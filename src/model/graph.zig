//! Composable Model Graph & Recurrent-Depth Executor (`W9`, `W10`, `W16`, ADR-0015 & ADR-0019).
//!
//! Executes frontier hybrid-sparse architectures (`Qwen3.8-Flash-Next` / `qwen4_exp`),
//! native looped transformers (`LoopLM`: `Prelude -> Core x R -> Coda` with `inject_x_init`
//! and opt-in `sigmoid_linear` early exit), and adapter-extended looped models
//! (`gated_loop_adapter` with L2-resident ARMv9 SME2 `SmeStepLoraAdapter` and `alpha_0 == 0.0`
//! bit-exact identity) over one unified 16 KiB-aligned wired memory substrate with zero
//! hot-path heap allocations.
//!
//! Enforces all four mandatory `ADR-0019` hybrid-sparse loop isolations on `r < R - 1`:
//!   1. `GDN` (`R6`/`G8`): evaluates `r < R - 1` into wired `S_tmp` without mutating
//!      committed `S_t`; commits `S_tmp -> S_t` once at `r = R - 1` (or `commitTree`).
//!   2. `PLE` (`C10`): calls `PleStore.lookupForLoopStep` so `r = 0` populates wired
//!      `LoopPleScratch` and `r >= 1` performs `0` extra NVMe reads (`nvme_reads_delta == 0`).
//!   3. `QSA` (`G9`): runs `ami_qsa_index_blocks` at `r = 0` and passes
//!      `reuse_block_indices = true` for `r >= 1`.
//!   4. `MoE` (`G7`, `C5`): deduplicates active experts across `M = 1..64` nodes via CSR
//!      gather (`E <= 1024`, `top_k <= 16`, optional `shared_expert`) and restricts `r >= 1`
//!      routing to wired/resident `ExpertSlotPool` slots (`0` cold NVMe stalls on `r >= 1`).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const expert = @import("../expert/source.zig");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const dequant = @import("../gpu/dequant.zig");
const attention = @import("../gpu/attention.zig");
const moe = @import("../gpu/moe.zig");
const hybrid = @import("../gpu/hybrid.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const aio = @import("../storage/aio.zig");
const cache_mod = @import("../storage/cache.zig");
const ple_mod = @import("../storage/ple.zig");
const sme = @import("../cpu/sme.zig");
const accept_mod = @import("../spec/accept.zig");
const qwen3 = @import("qwen3.zig");
const vision_mod = @import("../gpu/vision.zig");
const prefix_cache_mod = @import("../mem/prefix_cache.zig");
const batch_sched = @import("../runtime/batch_scheduler.zig");
const vmstat = @import("../platform/vmstat.zig");
const rng_mod = @import("../runtime/rng.zig");

/// Maximum wired bytes per GPU weight shard (matches `qwen3.zig` to stay below
/// macOS single-`MTLBuffer` maxBufferLength on 24 GiB–512 GiB Apple Silicon).
const MAX_SHARD_BYTES: usize = 12 * 1024 * 1024 * 1024;
const MAX_ADAPTER_PROJS_PER_LAYER: usize = 7;

// ===========================================================================
// Section 1: Composable Operator Registry (`W9`, ADR-0015 & ADR-0019)
// ===========================================================================

pub const BackboneKind = enum {
    decoder_only,
    /// Reserved for M7 Tier 3 (`DeepSeek-V4.1-Flash-1017`).
    causal_enc_dec,

    pub fn label(self: BackboneKind) []const u8 {
        return switch (self) {
            .decoder_only => "decoder_only",
            .causal_enc_dec => "causal_enc_dec",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!BackboneKind {
        if (std.mem.eql(u8, s, "decoder_only")) return .decoder_only;
        if (std.mem.eql(u8, s, "causal_enc_dec")) return .causal_enc_dec;
        return error.InvalidOperatorAxis;
    }
};

pub const AttentionKind = enum {
    gqa_full,
    gated_deltanet,
    qsa_sparse,
    /// Reserved for M7 Tier 3 (`CSA2`).
    mla_csa2,

    pub fn label(self: AttentionKind) []const u8 {
        return switch (self) {
            .gqa_full => "gqa_full",
            .gated_deltanet => "gated_deltanet",
            .qsa_sparse => "qsa_sparse",
            .mla_csa2 => "mla_csa2",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!AttentionKind {
        if (std.mem.eql(u8, s, "gqa_full") or std.mem.eql(u8, s, "full_attention")) return .gqa_full;
        if (std.mem.eql(u8, s, "gated_deltanet") or std.mem.eql(u8, s, "linear_attention")) return .gated_deltanet;
        if (std.mem.eql(u8, s, "qsa_sparse") or std.mem.eql(u8, s, "qsa")) return .qsa_sparse;
        if (std.mem.eql(u8, s, "mla_csa2") or std.mem.eql(u8, s, "csa2")) return .mla_csa2;
        return error.InvalidOperatorAxis;
    }
};

pub const FfnKind = enum {
    dense_swiglu,
    moe_swiglu,

    pub fn label(self: FfnKind) []const u8 {
        return switch (self) {
            .dense_swiglu => "dense_swiglu",
            .moe_swiglu => "moe_swiglu",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!FfnKind {
        if (std.mem.eql(u8, s, "dense_swiglu")) return .dense_swiglu;
        if (std.mem.eql(u8, s, "moe_swiglu")) return .moe_swiglu;
        return error.InvalidOperatorAxis;
    }
};

pub const ResidualKind = enum {
    single_stream,
    gated_4branch,
    /// Reserved for M7 Tier 3 (`Single-Pass mHC`).
    mhc_single_pass,

    pub fn label(self: ResidualKind) []const u8 {
        return switch (self) {
            .single_stream => "single_stream",
            .gated_4branch => "gated_4branch",
            .mhc_single_pass => "mhc_single_pass",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!ResidualKind {
        if (std.mem.eql(u8, s, "single_stream")) return .single_stream;
        if (std.mem.eql(u8, s, "gated_4branch")) return .gated_4branch;
        if (std.mem.eql(u8, s, "mhc_single_pass")) return .mhc_single_pass;
        return error.InvalidOperatorAxis;
    }
};

pub const MemoryLookupKind = enum {
    none,
    ple_ngram,
    /// Reserved for M7 Tier 3 (`196B Engram`).
    engram,

    pub fn label(self: MemoryLookupKind) []const u8 {
        return switch (self) {
            .none => "none",
            .ple_ngram => "ple_ngram",
            .engram => "engram",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!MemoryLookupKind {
        if (std.mem.eql(u8, s, "none")) return .none;
        if (std.mem.eql(u8, s, "ple_ngram")) return .ple_ngram;
        if (std.mem.eql(u8, s, "engram")) return .engram;
        return error.InvalidOperatorAxis;
    }
};

pub const LoopMode = enum {
    single_pass,
    native_recurrent,
    gated_loop_adapter,

    pub fn label(self: LoopMode) []const u8 {
        return switch (self) {
            .single_pass => "single_pass",
            .native_recurrent => "native_recurrent",
            .gated_loop_adapter => "gated_loop_adapter",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!LoopMode {
        if (std.mem.eql(u8, s, "single_pass")) return .single_pass;
        if (std.mem.eql(u8, s, "native_recurrent")) return .native_recurrent;
        if (std.mem.eql(u8, s, "gated_loop_adapter")) return .gated_loop_adapter;
        return error.InvalidOperatorAxis;
    }
};

pub const LoopAdapterKind = enum {
    none,
    sme2_step_lora,

    pub fn label(self: LoopAdapterKind) []const u8 {
        return switch (self) {
            .none => "none",
            .sme2_step_lora => "sme2_step_lora",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!LoopAdapterKind {
        if (std.mem.eql(u8, s, "none")) return .none;
        if (std.mem.eql(u8, s, "sme2_step_lora")) return .sme2_step_lora;
        return error.InvalidOperatorAxis;
    }
};

pub const EarlyExitGateKind = enum {
    none,
    sigmoid_linear,

    pub fn label(self: EarlyExitGateKind) []const u8 {
        return switch (self) {
            .none => "none",
            .sigmoid_linear => "sigmoid_linear",
        };
    }

    pub fn parse(s: []const u8) error{InvalidOperatorAxis}!EarlyExitGateKind {
        if (std.mem.eql(u8, s, "none")) return .none;
        if (std.mem.eql(u8, s, "sigmoid_linear")) return .sigmoid_linear;
        return error.InvalidOperatorAxis;
    }
};

pub const AdapterProjKind = enum(u8) {
    q_proj = 0,
    k_proj = 1,
    v_proj = 2,
    o_proj = 3,
    gate_proj = 4,
    up_proj = 5,
    down_proj = 6,

    pub fn parse(s: []const u8) ?AdapterProjKind {
        if (std.mem.eql(u8, s, "q_proj")) return .q_proj;
        if (std.mem.eql(u8, s, "k_proj")) return .k_proj;
        if (std.mem.eql(u8, s, "v_proj")) return .v_proj;
        if (std.mem.eql(u8, s, "o_proj")) return .o_proj;
        if (std.mem.eql(u8, s, "gate_proj")) return .gate_proj;
        if (std.mem.eql(u8, s, "up_proj")) return .up_proj;
        if (std.mem.eql(u8, s, "down_proj")) return .down_proj;
        return null;
    }
};

/// Canonical recurrent-depth descriptor (`ADR-0019`, `W9`/`W16`).
pub const LoopSchedule = struct {
    mode: LoopMode = .single_pass,
    start_layer: u32 = 0,
    end_layer: u32 = 0,
    max_loops: u32 = 1,
    inject_x_init: bool = false,
    adapter_kind: LoopAdapterKind = .none,
    adapter_rank: u32 = 0,
    adapter_packed_rank: u32 = 0,
    enforce_zero_alpha_r0: bool = true,
    early_exit_gate: EarlyExitGateKind = .none,
    early_exit_threshold: f32 = 0.5,

    pub fn isRecurrent(self: LoopSchedule) bool {
        return self.mode != .single_pass and self.max_loops > 1;
    }

    pub fn validate(self: LoopSchedule, num_layers: u32) GraphConfig.ConfigError!void {
        if (self.max_loops == 0) return error.InvalidConfig;
        if (num_layers == 0) return error.InvalidConfig;
        const end_l = if (self.end_layer == 0) num_layers else self.end_layer;
        if (self.start_layer >= end_l or end_l > num_layers) return error.InvalidConfig;
        if (self.adapter_kind == .sme2_step_lora) {
            if (self.adapter_rank == 0 or self.adapter_rank > sme.MAX_STEP_LORA_RANK) return error.InvalidConfig;
            if (self.adapter_packed_rank == 0 or (self.adapter_packed_rank % sme.GROUP_SIZE) != 0) return error.InvalidConfig;
            if (self.adapter_packed_rank < self.adapter_rank) return error.InvalidConfig;
            if (!self.enforce_zero_alpha_r0) return error.InvalidConfig;
        }
    }
};

/// Composable model graph operator axes (`W9`, ADR-0015).
pub const OperatorAxes = struct {
    backbone_kind: BackboneKind = .decoder_only,
    layer_attention_kinds: []AttentionKind = &.{},
    ffn_kind: FfnKind = .dense_swiglu,
    has_shared_expert: bool = false,
    residual_kind: ResidualKind = .single_stream,
    residual_branches: u32 = 1,
    memory_lookup_kind: MemoryLookupKind = .none,
    loop_schedule: LoopSchedule = .{},

    pub fn hasGdn(self: OperatorAxes) bool {
        for (self.layer_attention_kinds) |k| if (k == .gated_deltanet) return true;
        return false;
    }

    pub fn hasQsa(self: OperatorAxes) bool {
        for (self.layer_attention_kinds) |k| if (k == .qsa_sparse) return true;
        return false;
    }

    pub fn hasGqa(self: OperatorAxes) bool {
        for (self.layer_attention_kinds) |k| if (k == .gqa_full) return true;
        return false;
    }

    pub fn attentionSummaryLabel(self: OperatorAxes) []const u8 {
        if (self.layer_attention_kinds.len == 0) return "gqa_full";
        const first = self.layer_attention_kinds[0];
        for (self.layer_attention_kinds[1..]) |k| {
            if (k != first) return "hybrid_gdn_qsa";
        }
        return first.label();
    }
};

// ===========================================================================
// Section 2: `GraphConfig` & JSON Parsing (`qwen4_exp`, `LoopLM`, `Step-LoRA`)
// ===========================================================================

pub const GraphConfig = struct {
    axes: OperatorAxes,
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    norm_topk_prob: bool = true,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
    max_position_embeddings: u32 = 32768,
    tie_word_embeddings: bool = false,
    bos_token_id: u32 = 151643,
    eos_token_id: u32 = 151645,
    quant_bits: u8 = 3,
    group_size: u16 = 64,

    // Gated DeltaNet (`R6`/`G8`) geometry
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 0,
    linear_value_head_dim: u32 = 0,
    linear_conv_kernel_dim: u32 = 4,

    // QSA (`G9`/`R5`) geometry + YaRN + interleaved MRoPE
    qsa_block_size: u32 = hybrid.QSA_DEFAULT_BLOCK_TOKENS,
    qsa_top_k_blocks: u32 = 8,
    qsa_window_size: u32 = 0,
    qsa_index_dim: u32 = 64,
    yarn: hybrid.YarnConfig = .{},
    mrope_section: hybrid.MropeSection = hybrid.MropeSection.QWEN38_DEFAULT,
    mrope_interleaved: bool = true,

    // PLE (`C10`) metadata
    ple_num_rows: u64 = 0,
    ple_embed_dim: u32 = 0,
    ple_ngram_order: u8 = 2,
    ple_num_shards: u32 = 0,
    ple_shard_files: [][]u8 = &.{},

    // MTP metadata (optional file name relative to config directory)
    mtp_amf_file: ?[]u8 = null,
    mtp_config_file: ?[]u8 = null,

    pub const ConfigError = error{
        InvalidJson,
        InvalidConfig,
        UnsupportedOperatorAxis,
        OutOfMemory,
    };

    pub fn deinit(self: *GraphConfig, allocator: std.mem.Allocator) void {
        if (self.axes.layer_attention_kinds.len > 0) {
            allocator.free(self.axes.layer_attention_kinds);
            self.axes.layer_attention_kinds = &.{};
        }
        for (self.ple_shard_files) |s| allocator.free(s);
        if (self.ple_shard_files.len > 0) {
            allocator.free(self.ple_shard_files);
            self.ple_shard_files = &.{};
        }
        if (self.mtp_amf_file) |s| {
            allocator.free(s);
            self.mtp_amf_file = null;
        }
        if (self.mtp_config_file) |s| {
            allocator.free(s);
            self.mtp_config_file = null;
        }
    }

    pub fn qDim(self: GraphConfig) u32 {
        return self.num_attention_heads * self.head_dim;
    }

    pub fn kvDim(self: GraphConfig) u32 {
        return self.num_key_value_heads * self.head_dim;
    }

    pub fn gdnQkDim(self: GraphConfig) u32 {
        return self.linear_num_key_heads * self.linear_key_head_dim;
    }

    pub fn gdnVDim(self: GraphConfig) u32 {
        return self.linear_num_value_heads * self.linear_value_head_dim;
    }

    pub fn validate(self: GraphConfig) ConfigError!void {
        if (self.axes.backbone_kind != .decoder_only) return error.UnsupportedOperatorAxis;
        if (self.axes.residual_kind == .mhc_single_pass) return error.UnsupportedOperatorAxis;
        if (self.axes.memory_lookup_kind == .engram) return error.UnsupportedOperatorAxis;
        if (self.vocab_size == 0 or self.hidden_size == 0 or self.num_hidden_layers == 0) return error.InvalidConfig;
        if (self.axes.layer_attention_kinds.len != self.num_hidden_layers) return error.InvalidConfig;
        for (self.axes.layer_attention_kinds) |ak| {
            if (ak == .mla_csa2) return error.UnsupportedOperatorAxis;
        }
        if (self.axes.hasQsa() or self.axes.hasGqa()) {
            if (self.num_attention_heads == 0 or self.num_key_value_heads == 0 or self.head_dim == 0) return error.InvalidConfig;
            if ((self.num_attention_heads % self.num_key_value_heads) != 0) return error.InvalidConfig;
            if ((self.head_dim % 4) != 0 or self.head_dim > hybrid.MAX_QSA_HEAD_DIM) return error.InvalidConfig;
        }
        if (self.axes.hasGdn()) {
            if (self.linear_num_key_heads == 0 or self.linear_num_value_heads == 0 or
                self.linear_key_head_dim == 0 or self.linear_value_head_dim == 0) return error.InvalidConfig;
            if ((self.linear_num_value_heads % self.linear_num_key_heads) != 0) return error.InvalidConfig;
            if (self.linear_key_head_dim > hybrid.MAX_GDN_HEAD_DIM or self.linear_value_head_dim > hybrid.MAX_GDN_HEAD_DIM) {
                return error.InvalidConfig;
            }
            if (self.linear_conv_kernel_dim > 8) return error.InvalidConfig;
        }
        switch (self.axes.ffn_kind) {
            .dense_swiglu => {
                if (self.intermediate_size == 0) return error.InvalidConfig;
            },
            .moe_swiglu => {
                if (self.num_experts == 0 or self.num_experts > moe.MAX_EXPERTS) return error.InvalidConfig;
                if (self.num_experts_per_tok == 0 or self.num_experts_per_tok > moe.MAX_TOP_K or
                    self.num_experts_per_tok > self.num_experts) return error.InvalidConfig;
                if (self.moe_intermediate_size == 0) return error.InvalidConfig;
                if (self.axes.has_shared_expert and self.shared_expert_intermediate_size == 0) return error.InvalidConfig;
            },
        }
        if (self.axes.residual_kind == .gated_4branch and self.axes.residual_branches != hybrid.GR_NUM_BRANCHES) {
            return error.InvalidConfig;
        }
        if (self.axes.memory_lookup_kind == .ple_ngram) {
            if (self.ple_embed_dim != 0 and self.ple_embed_dim != self.hidden_size) return error.InvalidConfig;
        }
        try self.axes.loop_schedule.validate(self.num_hidden_layers);
    }

    /// Lightweight detector telling CLI / server whether a `config.json` requires
    /// the composable `GraphModel` executor (`qwen4_exp`, `LoopLM`, `Step-LoRA`,
    /// hybrid GDN/QSA, 4-branch GR, PLE, or `max_loops > 1`).
    pub fn isGraphConfigJson(json_bytes: []const u8) bool {
        if (std.mem.indexOf(u8, json_bytes, "\"qwen4_exp\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"looplm\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"gated_deltanet\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"qsa_sparse\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"gated_4branch\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"ple_ngram\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"native_recurrent\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"gated_loop_adapter\"") != null) return true;
        if (std.mem.indexOf(u8, json_bytes, "\"sme2_step_lora\"") != null) return true;
        return false;
    }

    pub fn fromJson(allocator: std.mem.Allocator, json_bytes: []const u8) ConfigError!GraphConfig {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch {
            return error.InvalidJson;
        };
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidJson;
        const root = parsed.value.object;
        const obj = if (root.get("text_config")) |tc|
            if (tc == .object) tc.object else root
        else
            root;

        const num_layers = jsonU32(obj, "num_hidden_layers") orelse return error.InvalidConfig;
        const vocab_size = jsonU32(obj, "vocab_size") orelse return error.InvalidConfig;
        const hidden_size = jsonU32(obj, "hidden_size") orelse return error.InvalidConfig;
        const num_attn_heads = jsonU32(obj, "num_attention_heads") orelse 0;
        const num_kv_heads = jsonU32(obj, "num_key_value_heads") orelse num_attn_heads;
        const head_dim = jsonU32(obj, "head_dim") orelse
            if (num_attn_heads > 0) (hidden_size / num_attn_heads) else 0;

        const model_type = jsonStr(obj, "model_type") orelse jsonStr(root, "model_type") orelse "";
        const is_qwen4 = std.mem.eql(u8, model_type, "qwen4_exp");
        const is_looplm = std.mem.eql(u8, model_type, "looplm") or
            std.mem.eql(u8, model_type, "huginn") or
            std.mem.eql(u8, model_type, "ouro");

        var backbone_kind: BackboneKind = .decoder_only;
        var ffn_kind: FfnKind = if ((jsonU32(obj, "num_experts") orelse 0) > 0) .moe_swiglu else .dense_swiglu;
        var has_shared_expert: bool = (jsonU32(obj, "shared_expert_intermediate_size") orelse 0) > 0;
        var residual_branches: u32 = jsonU32(obj, "residual_branches") orelse (if (is_qwen4) @as(u32, 4) else 1);
        var residual_kind: ResidualKind = if (residual_branches == 4) .gated_4branch else .single_stream;
        var mem_kind: MemoryLookupKind = if (obj.get("ple") != null or root.get("ple") != null) .ple_ngram else .none;

        const layer_attns = try allocator.alloc(AttentionKind, num_layers);
        errdefer allocator.free(layer_attns);
        const full_interval = jsonU32(obj, "full_attention_interval") orelse 4;
        for (0..num_layers) |l| {
            layer_attns[l] = if (is_qwen4)
                (if (((l + 1) % full_interval) == 0) .qsa_sparse else .gated_deltanet)
            else
                .gqa_full;
        }

        // Override from `layer_types` if provided.
        if (obj.get("layer_types") orelse root.get("layer_types")) |lt_val| {
            if (lt_val == .array and lt_val.array.items.len == num_layers) {
                for (lt_val.array.items, 0..) |item, l| {
                    if (item == .string) {
                        layer_attns[l] = AttentionKind.parse(item.string) catch return error.UnsupportedOperatorAxis;
                    }
                }
            }
        }

        // Override from canonical `operator_axes` if present.
        if (obj.get("operator_axes") orelse root.get("operator_axes")) |oa_val| {
            if (oa_val == .object) {
                const oa = oa_val.object;
                if (jsonStr(oa, "backbone_kind")) |s| {
                    backbone_kind = BackboneKind.parse(s) catch return error.UnsupportedOperatorAxis;
                }
                if (jsonStr(oa, "ffn_kind")) |s| {
                    ffn_kind = FfnKind.parse(s) catch return error.UnsupportedOperatorAxis;
                }
                if (jsonBool(oa, "has_shared_expert")) |b| has_shared_expert = b;
                if (jsonStr(oa, "residual_kind")) |s| {
                    residual_kind = ResidualKind.parse(s) catch return error.UnsupportedOperatorAxis;
                }
                if (jsonU32(oa, "residual_branches")) |rb| residual_branches = rb;
                if (jsonStr(oa, "memory_lookup_kind")) |s| {
                    mem_kind = MemoryLookupKind.parse(s) catch return error.UnsupportedOperatorAxis;
                }
                if (oa.get("layer_attention_kinds")) |lak| {
                    if (lak == .array and lak.array.items.len == num_layers) {
                        for (lak.array.items, 0..) |item, l| {
                            if (item == .string) {
                                layer_attns[l] = AttentionKind.parse(item.string) catch return error.UnsupportedOperatorAxis;
                            }
                        }
                    }
                }
            }
        }

        // Parse `loop_schedule` (or infer from `looplm` / `recurrent_depth`).
        var ls: LoopSchedule = .{
            .mode = .single_pass,
            .start_layer = 0,
            .end_layer = num_layers,
            .max_loops = 1,
        };
        if (obj.get("loop_schedule") orelse root.get("loop_schedule")) |ls_val| {
            if (ls_val == .object) {
                const lo = ls_val.object;
                if (jsonStr(lo, "mode")) |s| {
                    ls.mode = LoopMode.parse(s) catch return error.InvalidConfig;
                }
                ls.start_layer = jsonU32(lo, "start_layer") orelse 0;
                ls.end_layer = jsonU32(lo, "end_layer") orelse num_layers;
                ls.max_loops = jsonU32(lo, "max_loops") orelse 1;
                ls.inject_x_init = jsonBool(lo, "inject_x_init") orelse false;
                if (jsonStr(lo, "adapter_kind")) |s| {
                    ls.adapter_kind = LoopAdapterKind.parse(s) catch return error.InvalidConfig;
                }
                ls.adapter_rank = jsonU32(lo, "adapter_rank") orelse 0;
                ls.adapter_packed_rank = jsonU32(lo, "adapter_packed_rank") orelse
                    if (ls.adapter_rank > 0) @intCast(amf.align64(ls.adapter_rank)) else 0;
                ls.enforce_zero_alpha_r0 = jsonBool(lo, "enforce_zero_alpha_r0") orelse true;
                if (jsonStr(lo, "early_exit_gate")) |s| {
                    ls.early_exit_gate = EarlyExitGateKind.parse(s) catch return error.InvalidConfig;
                }
                if (jsonF32(lo, "early_exit_threshold")) |th| ls.early_exit_threshold = th;
            }
        } else if (is_looplm or (jsonU32(obj, "recurrent_depth") orelse 1) > 1) {
            const prelude = jsonU32(obj, "num_prelude_layers") orelse (if (num_layers >= 3) @as(u32, 1) else 0);
            const coda = jsonU32(obj, "num_coda_layers") orelse (if (num_layers >= 3) @as(u32, 1) else 0);
            ls = .{
                .mode = .native_recurrent,
                .start_layer = @min(prelude, num_layers - 1),
                .end_layer = @max(@min(prelude, num_layers - 1) + 1, num_layers -| coda),
                .max_loops = @max(2, jsonU32(obj, "recurrent_depth") orelse 3),
                .inject_x_init = jsonBool(obj, "inject_x_init") orelse true,
                .adapter_kind = .none,
                .early_exit_gate = .none,
            };
        }

        // Parse YaRN + MRoPE `rope_scaling`.
        var yarn: hybrid.YarnConfig = .{};
        var mrope_sec: hybrid.MropeSection = if (head_dim == 64)
            hybrid.MropeSection.QWEN38_DEFAULT
        else if (head_dim > 0)
            hybrid.MropeSection.text1d(head_dim)
        else
            hybrid.MropeSection.QWEN38_DEFAULT;
        var mrope_interleaved: bool = true;
        if (obj.get("rope_scaling")) |rs_val| {
            if (rs_val == .object) {
                const rs = rs_val.object;
                if (jsonF32(rs, "factor")) |f| yarn.factor = f;
                if (jsonF32(rs, "mscale")) |m| yarn.mscale = m;
                if (jsonF32(rs, "beta_fast")) |bf| yarn.beta_fast = bf;
                if (jsonF32(rs, "beta_slow")) |bs| yarn.beta_slow = bs;
                if (jsonU32(rs, "original_max_position_embeddings")) |omp| yarn.original_max_pos = omp;
                if (jsonBool(rs, "mrope_interleaved")) |mi| mrope_interleaved = mi;
                if (rs.get("mrope_section")) |ms_val| {
                    if (ms_val == .array and ms_val.array.items.len == 3) {
                        mrope_sec = .{ .section = .{
                            jsonItemU32(ms_val.array.items[0]) orelse 11,
                            jsonItemU32(ms_val.array.items[1]) orelse 11,
                            jsonItemU32(ms_val.array.items[2]) orelse 10,
                        } };
                    }
                }
            }
        }

        // Parse PLE metadata.
        var ple_num_rows: u64 = 0;
        var ple_embed_dim: u32 = hidden_size;
        var ple_ngram_order: u8 = 2;
        var ple_num_shards: u32 = 0;
        var ple_shard_files: [][]u8 = &.{};
        errdefer {
            for (ple_shard_files) |s| allocator.free(s);
            if (ple_shard_files.len > 0) allocator.free(ple_shard_files);
        }
        if (obj.get("ple") orelse root.get("ple")) |ple_val| {
            if (ple_val == .object) {
                const po = ple_val.object;
                ple_num_rows = jsonU64(po, "num_rows") orelse 0;
                ple_embed_dim = jsonU32(po, "embed_dim") orelse hidden_size;
                ple_ngram_order = @intCast(@min(8, @max(1, jsonU32(po, "ngram_order") orelse 2)));
                ple_num_shards = jsonU32(po, "num_shards") orelse 0;
                if (po.get("shard_files")) |sf_val| {
                    if (sf_val == .array and sf_val.array.items.len > 0) {
                        const files = try allocator.alloc([]u8, sf_val.array.items.len);
                        var built: usize = 0;
                        errdefer {
                            for (0..built) |i| allocator.free(files[i]);
                            allocator.free(files);
                        }
                        for (sf_val.array.items, 0..) |item, i| {
                            if (item != .string) return error.InvalidConfig;
                            files[i] = try allocator.dupe(u8, item.string);
                            built += 1;
                        }
                        ple_shard_files = files;
                        if (ple_num_shards == 0) ple_num_shards = @intCast(files.len);
                    }
                }
            }
        }

        // Parse optional MTP filenames.
        var mtp_amf_file: ?[]u8 = null;
        errdefer if (mtp_amf_file) |s| allocator.free(s);
        var mtp_config_file: ?[]u8 = null;
        errdefer if (mtp_config_file) |s| allocator.free(s);
        if (obj.get("mtp") orelse root.get("mtp")) |mtp_val| {
            if (mtp_val == .object) {
                if (jsonStr(mtp_val.object, "amf_file")) |s| mtp_amf_file = try allocator.dupe(u8, s);
                if (jsonStr(mtp_val.object, "config_file")) |s| mtp_config_file = try allocator.dupe(u8, s);
            }
        }

        // Quantization hints.
        var qbits: u8 = 3;
        var gsize: u16 = 64;
        if (root.get("quantization") orelse obj.get("quantization")) |qv| {
            if (qv == .object) {
                if (jsonU32(qv.object, "bits")) |b| qbits = @intCast(b);
                if (jsonU32(qv.object, "group_size")) |g| gsize = @intCast(g);
            }
        }

        const cfg: GraphConfig = .{
            .axes = .{
                .backbone_kind = backbone_kind,
                .layer_attention_kinds = layer_attns,
                .ffn_kind = ffn_kind,
                .has_shared_expert = has_shared_expert,
                .residual_kind = residual_kind,
                .residual_branches = residual_branches,
                .memory_lookup_kind = mem_kind,
                .loop_schedule = ls,
            },
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
            .num_hidden_layers = num_layers,
            .num_attention_heads = num_attn_heads,
            .num_key_value_heads = num_kv_heads,
            .head_dim = head_dim,
            .intermediate_size = jsonU32(obj, "intermediate_size") orelse 0,
            .moe_intermediate_size = jsonU32(obj, "moe_intermediate_size") orelse 0,
            .shared_expert_intermediate_size = jsonU32(obj, "shared_expert_intermediate_size") orelse 0,
            .num_experts = jsonU32(obj, "num_experts") orelse 0,
            .num_experts_per_tok = jsonU32(obj, "num_experts_per_tok") orelse 0,
            .norm_topk_prob = jsonBool(obj, "norm_topk_prob") orelse true,
            .rms_norm_eps = jsonF32(obj, "rms_norm_eps") orelse 1e-6,
            .rope_theta = jsonF32(obj, "rope_theta") orelse 1_000_000.0,
            .max_position_embeddings = jsonU32(obj, "max_position_embeddings") orelse 32768,
            .tie_word_embeddings = jsonBool(obj, "tie_word_embeddings") orelse false,
            .bos_token_id = jsonU32(obj, "bos_token_id") orelse jsonU32(root, "bos_token_id") orelse 151643,
            .eos_token_id = jsonU32(obj, "eos_token_id") orelse jsonU32(root, "eos_token_id") orelse 151645,
            .quant_bits = qbits,
            .group_size = gsize,
            .linear_num_key_heads = jsonU32(obj, "linear_num_key_heads") orelse 0,
            .linear_num_value_heads = jsonU32(obj, "linear_num_value_heads") orelse 0,
            .linear_key_head_dim = jsonU32(obj, "linear_key_head_dim") orelse 0,
            .linear_value_head_dim = jsonU32(obj, "linear_value_head_dim") orelse 0,
            .linear_conv_kernel_dim = jsonU32(obj, "linear_conv_kernel_dim") orelse 4,
            .qsa_block_size = jsonU32(obj, "qsa_block_size") orelse hybrid.QSA_DEFAULT_BLOCK_TOKENS,
            .qsa_top_k_blocks = jsonU32(obj, "qsa_top_k_blocks") orelse 8,
            .qsa_window_size = jsonU32(obj, "qsa_window_size") orelse 0,
            .qsa_index_dim = jsonU32(obj, "qsa_index_dim") orelse 64,
            .yarn = yarn,
            .mrope_section = mrope_sec,
            .mrope_interleaved = mrope_interleaved,
            .ple_num_rows = ple_num_rows,
            .ple_embed_dim = ple_embed_dim,
            .ple_ngram_order = ple_ngram_order,
            .ple_num_shards = ple_num_shards,
            .ple_shard_files = ple_shard_files,
            .mtp_amf_file = mtp_amf_file,
            .mtp_config_file = mtp_config_file,
        };
        try cfg.validate();
        return cfg;
    }
};

fn jsonStr(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn jsonBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const v = obj.get(key) orelse return null;
    return if (v == .bool) v.bool else null;
}

fn jsonU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = obj.get(key) orelse return null;
    return jsonItemU32(v);
}

fn jsonU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        else => null,
    };
}

fn jsonItemU32(v: std.json.Value) ?u32 {
    return switch (v) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

fn jsonF32(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

// ===========================================================================
// Section 3: Load Options, Errors & ADR-0019 Loop Isolation Telemetry
// ===========================================================================

pub const GraphLoadOptions = struct {
    max_seq_len: u32 = 2048,
    max_tree_nodes: u32 = 64,
    chunk_tokens: u32 = attention.DEFAULT_CHUNK_TOKENS,
    timeout_ms: u64 = 30_000,
    wired_budget_override: ?budget_mod.Budget = null,
    enable_feature_taps: bool = false,
    aux_layer_ids: ?[3]u32 = null,
    /// Optional explicit PLE shard `.amf` paths (`C10`, `F_NOCACHE`).
    ple_paths: []const [:0]const u8 = &.{},
    /// Optional external `Step-LoRA` adapter `.amf` file (`W16`).
    loop_adapter_amf_path: ?[:0]const u8 = null,
    /// Optional runtime override for `LoopSchedule.max_loops` (`--max-loops`).
    max_loops_override: ?u32 = null,
    /// Opt-in approximate token-adaptive early exit (`--loop-early-exit`).
    enable_loop_early_exit: bool = false,
    early_exit_threshold: ?f32 = null,
    /// Cold-expert `ExpertSlotPool` slot count (`0` auto-sizes when cold experts exist).
    cold_slot_count: u32 = 0,
    ple_cache_pages: usize = ple_mod.DEFAULT_CACHE_PAGES,
    /// M6.5 R5: When `> 0`, keep at most `N` active `RGBA16Float` IOSurface KV chunks
    /// per full-context `GQA` / `QSA` layer and retire older completed chunks on GPU
    /// into group-64 `q8` (`RetiredKvQ8Pool`, `-46.875%` retired KV RAM). `0` disables retirement.
    kv_retire_q8_after_chunks: u32 = 0,

    // M6.8 Wide GPU Prefill, Tail Pruning, Scratch Aliasing, Canary & Live RAM (`ADR-0020`)
    /// GPU prefill backend (`wide` = 512–1024 tok wide prefill, default; `tree64` = legacy `<= 64` A/B baseline).
    gpu_prefill_mode: qwen3.GpuPrefillMode = .wide,
    /// Wide GPU prefill chunk capacity (`512` default; up to `attention.MAX_WIDE_PREFILL_TOKENS = 1024`).
    wide_prefill_chunk_tokens: u32 = 512,
    /// Run startup `.metallib` numerical canary once during load (`G16`).
    startup_canary: bool = true,
    /// When `true` (used by continuous-batching `aminfer serve`), keep `max_seq_len`
    /// intact for `GraphBatchRuntime` validation while allocating only `1` minimal
    /// single-request KV chunk per layer on the base model (saving unused IOSurfaces
    /// and wired RAM when all serving traffic executes through `GraphBatchRuntime`).
    continuous_batch_only: bool = false,
    /// Alias single-request wide-prefill scratch with `verifyTree` scratch via `WiredArena.allocAliasedPair` (`P0-B2`).
    alias_prefill_tree_scratch: bool = true,
    /// Opt-in strict live non-anonymous RAM admission (`--strict-live-ram`, `P1-B`).
    strict_live_ram: bool = false,
    /// Safety margin subtracted from live reclaimable RAM (`P1-B`, default 512 MiB).
    live_safety_margin_bytes: u64 = budget_mod.DEFAULT_LIVE_SAFETY_MARGIN_BYTES,
    /// Optional synthetic live zero-swap RAM override for deterministic tests.
    live_reclaimable_override: ?u64 = null,
    /// Disable `P0-B1` final-layer `prompt_tail` compute pruning when full-prompt logits are requested.
    compute_all_logits: bool = false,
    /// Enable lossless `P0-B1` final-layer `prompt_tail` compute pruning (default `true`).
    enable_prompt_tail_pruning: bool = true,
};

pub const LoopIsolationStats = struct {
    total_loop_steps: u64 = 0,
    early_exit_tokens: u64 = 0,
    /// Isolation #1 (GDN R6/G8): scratch evaluations on `r < R-1` and final commits on `r == R-1`.
    gdn_scratch_evals: u64 = 0,
    gdn_final_commits: u64 = 0,
    /// Isolation #2 (PLE C10): `r = 0` fills vs `r >= 1` zero-NVMe-read scratch reuses.
    ple_r0_fills: u64 = 0,
    ple_r_ge1_reuses: u64 = 0,
    ple_r_ge1_nvme_reads: u64 = 0,
    /// Isolation #3 (QSA G9): `r = 0` block-index dispatches vs `r >= 1` index reuses.
    qsa_index_computes: u64 = 0,
    qsa_index_reuses: u64 = 0,
    /// Isolation #4 (MoE G7/C5): `r >= 1` pinned-only routing masks & cold stalls on `r >= 1`.
    moe_r_ge1_pinned_routes: u64 = 0,
    moe_r_ge1_cold_stalls: u64 = 0,
    /// W16 Step-LoRA: `r = 0` zero-alpha skips vs active SME2 evaluations.
    steplora_zero_alpha_skips: u64 = 0,
    steplora_active_evals: u64 = 0,
};

pub const GraphLoadError = error{
    MissingTensor,
    TensorShapeMismatch,
    UnsupportedDType,
    UnsupportedOperatorAxis,
    InvalidJson,
    InvalidConfig,
    NonZeroLoopZeroAlpha,
    BudgetExceeded,
    TooManyWeightShards,
    Overflow,
    OutOfMemory,
} || amf.ParseError || amf.IoError || metal.Error || sysinfo.Error || vmstat.Error ||
    budget_mod.PlanError || dequant.CanaryError ||
    page.ShareError || expert.NvmeExpert.InitError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError || hybrid.ArenaError || hybrid.HybridKvRing.InitError ||
    attention.RetiredKvQ8PoolError ||
    moe.WorkspaceError || ple_mod.Error || sme.WiredScratchError ||
    cache_mod.PoolError || cache_mod.CacheError || aio.EcorePoolError;

pub const GraphStepError = error{
    SequenceFull,
    TokenOutOfRange,
    TreeNotEnabled,
    TreeTooLarge,
    InvalidBatchSize,
    InvalidLoopStep,
    Overflow,
} || metal.Error || dequant.EncodeError || moe.EncodeError ||
    hybrid.EncodeError || hybrid.ArenaError || ple_mod.Error ||
    attention.RetiredKvQ8PoolError ||
    sme.Error || cache_mod.CacheError || accept_mod.VerifyError;

// ===========================================================================
// Section 4: Internal Layer & Runtime Structures
// ===========================================================================

const WeightShard = struct {
    arena: arena_mod.WiredArena,
    buffer: metal.Buffer,
};

const GrLayerWeights = struct {
    read_gate: metal.BufferBinding,
    write_gate: metal.BufferBinding,
    decay_gate: metal.BufferBinding,
    dtype: quant.FloatDType,
};

const GdnLayerWeights = struct {
    gdn_idx: u32,
    q_proj: qwen3.LinearBinding,
    k_proj: qwen3.LinearBinding,
    v_proj: qwen3.LinearBinding,
    g_proj: qwen3.LinearBinding,
    o_proj: qwen3.LinearBinding,
    a_proj: metal.BufferBinding,
    b_proj: metal.BufferBinding,
    ab_dtype: quant.FloatDType,
    a_log: metal.BufferBinding,
    dt_bias: metal.BufferBinding,
    conv1d_w: ?metal.BufferBinding,
    conv_k: u32,
    out_norm: ?qwen3.NormBinding,
    // Per-layer scratch offsets for tree commit (`k_tree`, `v_tree`, `g_tree`, `beta_tree`, raw QKV).
    tree_k_off: u64,
    tree_v_off: u64,
    tree_g_off: u64,
    tree_beta_off: u64,
    tree_q_raw_off: u64,
    tree_q_raw_host: []f16,
    tree_k_raw_off: u64,
    tree_v_raw_off: u64,
    conv_state_off: u64,
    conv_state_tmp_off: u64,
    // M6.8 wide-prefill per-layer `r = 0` loop-isolation scratch offsets (active when `wide_enabled and R > 1`).
    wp_k_off: u64 = 0,
    wp_v_off: u64 = 0,
    wp_g_off: u64 = 0,
    wp_beta_off: u64 = 0,
    wp_q_raw_off: u64 = 0,
};

const QsaLayerWeights = struct {
    qsa_idx: u32,
    q_proj: qwen3.LinearBinding,
    k_proj: qwen3.LinearBinding,
    v_proj: qwen3.LinearBinding,
    o_proj: qwen3.LinearBinding,
    q_norm: ?metal.BufferBinding,
    k_norm: ?metal.BufferBinding,
    qk_norm_dtype: quant.FloatDType,
    kv_ring: hybrid.HybridKvRing,
    index_buf: hybrid.QsaIndexBuffer,
    wp_index_buf: ?hybrid.QsaIndexBuffer = null,
    block_keys_off: u64,
    max_blocks: u32,
    tree_k_off: u64,
    tree_v_off: u64,
    // M6.8 wide-prefill circular-SWA `r = 0` K/V staging (active when `wide_enabled and ring_capacity > 0 and R > 1`).
    wp_k_off: u64 = 0,
    wp_v_off: u64 = 0,
    // M6.5 R5 retired q8 KV tier (active when `supports_retire_q8 == true`)
    supports_retire_q8: bool = false,
    full_tile_table_off: u64 = 0,
    full_tile_table_host: []attention.KvTilePair = &.{},
    retired_ids_binding: metal.BufferBinding = undefined,
    retired_ids_host: []u32 = &.{},
};

const GqaLayerWeights = struct {
    gqa_idx: u32,
    q_proj: qwen3.LinearBinding,
    k_proj: qwen3.LinearBinding,
    v_proj: qwen3.LinearBinding,
    o_proj: qwen3.LinearBinding,
    q_norm: ?metal.BufferBinding,
    k_norm: ?metal.BufferBinding,
    qk_norm_dtype: quant.FloatDType,
    kv_tiles_binding: metal.BufferBinding,
    kv_tiles_host: []attention.KvTilePair = &.{},
    retired_ids_binding: metal.BufferBinding = undefined,
    retired_ids_host: []u32 = &.{},
    kv_tile_start: usize,
    tree_k_off: u64,
    tree_v_off: u64,
};

const LayerAttention = union(enum) {
    gqa: GqaLayerWeights,
    gdn: GdnLayerWeights,
    qsa: QsaLayerWeights,
};

const GraphLayerWeights = struct {
    shard_idx: u16,
    input_layernorm: qwen3.NormBinding,
    post_attention_layernorm: qwen3.NormBinding,
    gr: ?GrLayerWeights = null,
    attn: LayerAttention,

    // Dense SwiGLU (when `ffn_kind == .dense_swiglu`)
    dense_gate: qwen3.LinearBinding = undefined,
    dense_up: qwen3.LinearBinding = undefined,
    dense_down: qwen3.LinearBinding = undefined,

    // MoE SwiGLU (when `ffn_kind == .moe_swiglu`)
    router_binding: metal.BufferBinding = undefined,
    router_dtype: moe.RouterWeightDType = .bf16,
    router_quant_dtype: ?quant.WeightDType = null,
    router_scale_bias: ?metal.BufferBinding = null,
    expert_weights_buf: metal.Buffer = undefined,
    expert_table: metal.BufferBinding = undefined,
    expert_table_host: []moe.ExpertWeights = &.{},
    expert_weight_dtype: quant.WeightDType = .q3_affine,
    expert_dtypes_binding: ?metal.BufferBinding = null,
    expert_scheme_counts: [moe.NUM_SCHEME_BUCKETS]u32 = @splat(0),
    is_hetero_moe: bool = false,
    has_cold_experts: bool = false,
    cold_expert_count: u32 = 0,
    shared_expert: ?moe.ExpertWeights = null,
    shared_expert_buf: ?metal.Buffer = null,
    shared_expert_dtype: ?quant.WeightDType = null,
    shared_gate_weight: ?metal.BufferBinding = null,
    shared_gate_dtype: moe.RouterWeightDType = .bf16,
};

const PleRuntime = struct {
    store: ple_mod.PleStore,
    scratch: ple_mod.LoopPleScratch,
    f32_tmp: []f32,
};

const ColdRuntime = struct {
    pool: cache_mod.ExpertSlotPool,
    reader: aio.AsyncExpertReader,
    expert_cache: cache_mod.ExpertCache,
    amf_file: ?amf.AmfFile,
    slot_count: u32,
    /// Per-layer, per-node `r = 0` selected top-k expert IDs:
    /// flat `[num_layers * max_tree_nodes * top_k]`, preallocated at load time
    /// so `r >= 1` routing never sees speculative sibling cold-cache entries.
    r0_selected: []u16,
};

const LoopAdapterRuntime = struct {
    external_arena: ?arena_mod.WiredArena = null,
    dispatcher: sme.SmeDispatcher,
    scratch: sme.StepLoraScratch,
    delta_f32: []f32,
    max_steps: u32,
    num_layers: u32,
    /// Flat `[max_steps * num_layers * MAX_ADAPTER_PROJS_PER_LAYER]` table.
    entries: []?sme.SmeStepLoraAdapter,

    fn get(self: *const LoopAdapterRuntime, step: u32, layer: u32, proj: AdapterProjKind) ?sme.SmeStepLoraAdapter {
        if (step >= self.max_steps or layer >= self.num_layers) return null;
        const idx = (@as(usize, step) * self.num_layers + layer) * MAX_ADAPTER_PROJS_PER_LAYER + @backingInt(proj);
        return self.entries[idx];
    }

    fn set(self: *LoopAdapterRuntime, step: u32, layer: u32, proj: AdapterProjKind, adapter: sme.SmeStepLoraAdapter) void {
        const idx = (@as(usize, step) * self.num_layers + layer) * MAX_ADAPTER_PROJS_PER_LAYER + @backingInt(proj);
        self.entries[idx] = adapter;
    }

    fn deinit(self: *LoopAdapterRuntime, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        allocator.free(self.delta_f32);
        self.scratch.deinit();
        self.dispatcher.deinit();
        if (self.external_arena) |*ea| ea.deinit();
    }
};

// ===========================================================================
// Section 5: `GraphModel` Composable Executor
// ===========================================================================

pub const GraphModel = struct {
    config: GraphConfig,
    effective_max_loops: u32,
    enable_loop_early_exit: bool,
    early_exit_threshold: f32,
    max_seq_len: u32,
    max_tree_nodes: u32,
    chunk_tokens: u32,
    num_chunks: u32,
    seq_len: u32,
    timeout_ms: u64,
    wired_budget_bytes: u64,
    total_wired_bytes: u64,
    enable_feature_taps: bool,
    aux_layer_ids: [3]u32,

    // Hardware handles (public for `DraftHead` & `SpecEngine` interop)
    device: metal.Device,
    library: metal.Library,
    compute: metal.Compute,
    dequant_pipes: dequant.DequantPipelines,
    attn_pipes: attention.AttentionPipelines,
    moe_pipes: moe.MoePipelines,
    hybrid_pipes: hybrid.HybridPipelines,

    // Multi-shard wired weight arenas + scratch arena
    weight_shards: []WeightShard,
    scratch_arena: arena_mod.WiredArena,
    scratch_buf: metal.Buffer,

    // Optional hybrid state arenas
    gdn_arena: ?hybrid.GdnStateArena,
    gr_arena: ?hybrid.GrStreamArena,
    gqa_kv_tiles: []metal.KvTile,

    // Global weights
    embed: qwen3.LinearBinding,
    final_norm: qwen3.NormBinding,
    lm_head: qwen3.LinearBinding,
    early_exit_gate_host: ?[]const u8,
    early_exit_gate_dtype: quant.FloatDType,
    layers: []GraphLayerWeights,

    // Optional subsystems (heap-pinned where internal pointers exist)
    ple_rt: ?*PleRuntime,
    cold_rt: ?*ColdRuntime,
    adapter_rt: ?*LoopAdapterRuntime,

    // Preallocated hot-path scratch host slices & GPU bindings (zero alloc on decode/verify)
    token_history: []u32,
    token_ids_host: []u32,
    positions_host: []u32,
    pos3d_host: []hybrid.Pos3D,
    ancestor_masks_host: []u64,
    out_token_ids_host: []u32,
    logits_host: []const f16,
    topk_host: []moe.TopKCandidate,
    ple_row_indices_host: []u64,
    ple_f16_host: []f16,
    x_host: []f16,
    x_norm_host: []f16,
    q_host: []f16,
    k_host: []f16,
    v_host: []f16,
    attn_out_host: []f16,
    proj_out_host: []f16,
    dense_gate_host: []f16,
    dense_up_host: []f16,
    dense_act_host: []f16,
    accepted_top_host: []f16,
    accepted_aux_host: []f16,

    token_ids_bind: metal.BufferBinding,
    positions_bind: metal.BufferBinding,
    pos3d_bind: metal.BufferBinding,
    ancestor_masks_bind: metal.BufferBinding,
    out_token_id_bind: metal.BufferBinding,
    logits_bind: metal.BufferBinding,
    topk_bind: metal.BufferBinding,
    ple_f16_base_off: u64,
    x_bind: metal.BufferBinding,
    x_init_bind: metal.BufferBinding,
    x_norm_bind: metal.BufferBinding,
    q_bind: metal.BufferBinding,
    k_bind: metal.BufferBinding,
    v_bind: metal.BufferBinding,
    q_conv_bind: metal.BufferBinding,
    k_conv_bind: metal.BufferBinding,
    v_conv_bind: metal.BufferBinding,
    gdn_z_bind: metal.BufferBinding,
    gdn_g_bind: metal.BufferBinding,
    gdn_beta_bind: metal.BufferBinding,
    gr_g_read_bind: metal.BufferBinding,
    gr_g_write_bind: metal.BufferBinding,
    gr_decay_bind: metal.BufferBinding,
    attn_out_bind: metal.BufferBinding,
    proj_out_bind: metal.BufferBinding,
    dense_gate_bind: metal.BufferBinding,
    dense_up_bind: metal.BufferBinding,
    dense_act_bind: metal.BufferBinding,
    gqa_scratch_bind: metal.BufferBinding,
    moe_ws: ?moe.MoeWorkspace,
    layer_sched_hdr_raw: []u8,
    layer_sched_hdr_off: u64,
    accepted_top_bind: metal.BufferBinding,
    accepted_aux_bind: metal.BufferBinding,
    aux_tree_base_off: u64,

    // Accounting for `SpecEngine` & `bench-m6`
    single_expert_bytes: u64,
    total_layers_non_expert_bytes: u64,
    total_layers_static_bytes: u64,
    global_head_bytes: u64,
    embed_row_alg_bytes: u64,
    last_logits_row: usize,
    last_verified_nodes: u32,
    last_verified_tokens: [hybrid.MAX_TREE_NODES]u32,
    last_verified_masks: [hybrid.MAX_TREE_NODES]u64,
    loop_stats: LoopIsolationStats,

    // M6.5 R5 retired group-64 `q8` KV tier + G14 3D MRoPE next text position
    kv_retire_after_chunks: u32 = 0,
    retire_chunk_tokens: u32 = attention.DEFAULT_CHUNK_TOKENS,
    active_kv_chunks: u32 = 0,
    num_retired_chunks: u32 = 0,
    max_retired_per_layer: u32 = 0,
    kv_retire_events: u64 = 0,
    active_kv_wired_bytes: u64 = 0,
    retired_kv_wired_bytes: u64 = 0,
    retired_kv_pool: ?attention.RetiredKvQ8Pool = null,
    active_tile_phys_ring: []u32 = &.{},
    mrope_next_pos: u32 = 0,

    // M6.8 Wide GPU Prefill (`G15`), Tail Pruning (`P0-B1`), Aliased Scratch (`P0-B2`),
    // Live RAM Admission (`P1-B`) & Startup `.metallib` Canary (`G16`)
    gpu_prefill_mode: qwen3.GpuPrefillMode = .wide,
    wide_prefill_max_tokens: u32 = 0,
    supports_wide_prefill: bool = false,
    compute_all_logits: bool = false,
    enable_prompt_tail_pruning: bool = true,
    canary_result: ?dequant.CanaryResult = null,
    live_admission: budget_mod.LiveAdmissionPlan = std.mem.zeroes(budget_mod.LiveAdmissionPlan),
    aliased_scratch_saved_bytes: usize = 0,
    wp_token_ids_host: []u32 = &.{},
    wp_pos3d_host: []hybrid.Pos3D = &.{},
    wp_ple_f16_host: []f16 = &.{},
    wp_token_ids_bind: metal.BufferBinding = undefined,
    wp_pos3d_bind: metal.BufferBinding = undefined,
    wp_ple_f16_base_off: u64 = 0,
    wp_gr_arena: ?hybrid.GrStreamArena = null,
    wp_gr_g_read_bind: metal.BufferBinding = undefined,
    wp_gr_g_write_bind: metal.BufferBinding = undefined,
    wp_gr_decay_bind: metal.BufferBinding = undefined,
    wp_x_bind: metal.BufferBinding = undefined,
    wp_x_init_bind: metal.BufferBinding = undefined,
    wp_x_norm_bind: metal.BufferBinding = undefined,
    wp_q_bind: metal.BufferBinding = undefined,
    wp_k_bind: metal.BufferBinding = undefined,
    wp_v_bind: metal.BufferBinding = undefined,
    wp_q_conv_bind: metal.BufferBinding = undefined,
    wp_k_conv_bind: metal.BufferBinding = undefined,
    wp_v_conv_bind: metal.BufferBinding = undefined,
    wp_gdn_z_bind: metal.BufferBinding = undefined,
    wp_gdn_g_bind: metal.BufferBinding = undefined,
    wp_gdn_beta_bind: metal.BufferBinding = undefined,
    wp_attn_out_bind: metal.BufferBinding = undefined,
    wp_proj_out_bind: metal.BufferBinding = undefined,
    wp_dense_gate_bind: metal.BufferBinding = undefined,
    wp_dense_up_bind: metal.BufferBinding = undefined,
    wp_dense_act_bind: metal.BufferBinding = undefined,
    wp_moe_ws: ?moe.MoeWorkspace = null,

    // -----------------------------------------------------------------------
    // Loading & Teardown
    // -----------------------------------------------------------------------

    const AmfSource = union(enum) {
        bytes: struct { parsed: *const amf.Parsed, raw: []const u8 },
        file: struct { file: *amf.AmfFile, path: [:0]const u8 },

        fn entries(self: AmfSource) []const amf.TensorEntry {
            return switch (self) {
                .bytes => |b| b.parsed.entries,
                .file => |f| f.file.parsed.entries,
            };
        }

        fn find(self: AmfSource, name: []const u8) ?*const amf.TensorEntry {
            return switch (self) {
                .bytes => |b| b.parsed.find(name),
                .file => |f| f.file.parsed.find(name),
            };
        }

        fn loadIntoArena(self: AmfSource, entry: *const amf.TensorEntry, arena: *arena_mod.WiredArena) (amf.ParseError || amf.IoError || arena_mod.WiredArena.AllocError)![]align(page.APPLE_PAGE_SIZE) u8 {
            return switch (self) {
                .bytes => |b| b.parsed.loadBlobFromBytes(b.raw, entry, arena),
                .file => |f| f.file.loadBlob(entry, arena),
            };
        }
    };

    pub fn loadFromBytes(
        allocator: std.mem.Allocator,
        amf_bytes: []const u8,
        config: GraphConfig,
        options: GraphLoadOptions,
    ) GraphLoadError!GraphModel {
        var parsed = try amf.Parsed.parseBytes(allocator, amf_bytes);
        defer parsed.deinit(allocator);
        return loadInternal(allocator, .{ .bytes = .{ .parsed = &parsed, .raw = amf_bytes } }, config, options);
    }

    pub fn loadFromFile(
        allocator: std.mem.Allocator,
        amf_path: [:0]const u8,
        config: GraphConfig,
        options: GraphLoadOptions,
    ) GraphLoadError!GraphModel {
        var file = try amf.AmfFile.open(allocator, amf_path);
        defer file.close(allocator);
        return loadInternal(allocator, .{ .file = .{ .file = &file, .path = amf_path } }, config, options);
    }

    pub fn deinit(self: *GraphModel, allocator: std.mem.Allocator) void {
        if (self.adapter_rt) |art| {
            art.deinit(allocator);
            allocator.destroy(art);
            self.adapter_rt = null;
        }
        if (self.cold_rt) |crt| {
            allocator.free(crt.r0_selected);
            crt.expert_cache.deinit(allocator);
            crt.reader.deinit(allocator);
            crt.pool.deinit();
            if (crt.amf_file) |*f| f.close(allocator);
            allocator.destroy(crt);
            self.cold_rt = null;
        }
        if (self.ple_rt) |prt| {
            allocator.free(prt.f32_tmp);
            prt.scratch.deinit(allocator);
            prt.store.deinit(allocator);
            allocator.destroy(prt);
            self.ple_rt = null;
        }
        for (self.layers) |*layer| {
            switch (layer.attn) {
                .qsa => |*qsa| {
                    qsa.kv_ring.removeResident(self.compute) catch |err| {
                        std.debug.panic("QSA kv_ring.removeResident failed: {}", .{err});
                    };
                    qsa.kv_ring.deinit(allocator);
                },
                else => {},
            }
        }
        if (self.gqa_kv_tiles.len > 0) {
            self.compute.removeTilesResident(self.gqa_kv_tiles) catch |err| {
                std.debug.panic("removeTilesResident failed in GraphModel.deinit: {}", .{err});
            };
            for (self.gqa_kv_tiles) |tile| tile.close();
            allocator.free(self.gqa_kv_tiles);
        }
        self.compute.removeResident(&.{self.scratch_buf}) catch |err| {
            std.debug.panic("removeResident(scratch) failed in GraphModel.deinit: {}", .{err});
        };
        for (self.weight_shards) |ws| {
            self.compute.removeResident(&.{ws.buffer}) catch |err| {
                std.debug.panic("removeResident(weight_shard) failed in GraphModel.deinit: {}", .{err});
            };
        }
        self.scratch_buf.close();
        self.scratch_arena.deinit();
        for (self.weight_shards) |*ws| {
            ws.buffer.close();
            ws.arena.deinit();
        }
        allocator.free(self.weight_shards);
        allocator.free(self.layers);
        allocator.free(self.token_history);
        allocator.free(self.ple_row_indices_host);
        if (self.active_tile_phys_ring.len > 0) allocator.free(self.active_tile_phys_ring);

        self.hybrid_pipes.close();
        self.moe_pipes.close();
        self.attn_pipes.close();
        self.dequant_pipes.close();
        self.compute.close();
        self.library.close();
        self.device.close();
    }

    pub fn resetKvCache(self: *GraphModel) void {
        self.seq_len = 0;
        self.mrope_next_pos = 0;
        self.last_logits_row = 0;
        self.last_verified_nodes = 0;
        self.num_retired_chunks = 0;
        self.kv_retire_events = 0;
        if (self.retired_kv_pool) |*pool| pool.reset();
        for (0..self.active_tile_phys_ring.len) |i| {
            self.active_tile_phys_ring[i] = @intCast(i);
        }
        self.syncActiveKvTileTables();
        if (self.gdn_arena) |*ga| ga.resetAll();
        for (self.layers) |layer| {
            switch (layer.attn) {
                .gdn => |g| {
                    if (g.conv_k > 1) {
                        const conv_dim = 2 * self.config.gdnQkDim() + self.config.gdnVDim();
                        const state_bytes = @as(usize, g.conv_k - 1) * conv_dim * @sizeOf(f16);
                        const base = self.scratch_arena.bytes();
                        @memset(base[@intCast(g.conv_state_off)..][0..state_bytes], 0);
                        @memset(base[@intCast(g.conv_state_tmp_off)..][0..state_bytes], 0);
                    }
                },
                else => {},
            }
        }
        if (self.ple_rt) |prt| {
            prt.scratch.resetStep();
        }
    }

    pub fn kvRetireEnabled(self: *const GraphModel) bool {
        return self.retired_kv_pool != null;
    }

    pub fn retiredKvChunks(self: *const GraphModel) u32 {
        return self.num_retired_chunks;
    }

    pub fn kvRetireEvents(self: *const GraphModel) u64 {
        return self.kv_retire_events;
    }

    pub fn activeKvWiredBytes(self: *const GraphModel) u64 {
        return self.active_kv_wired_bytes;
    }

    pub fn retiredKvWiredBytes(self: *const GraphModel) u64 {
        return self.retired_kv_wired_bytes;
    }

    fn syncActiveKvTileTables(self: *GraphModel) void {
        if (self.active_tile_phys_ring.len == 0) return;
        const active_n: usize = self.active_kv_chunks;
        for (self.layers) |*layer| {
            switch (layer.attn) {
                .gqa => |*gqa| {
                    if (gqa.kv_tiles_host.len == 0) continue;
                    for (0..active_n) |slot| {
                        const phys: usize = self.active_tile_phys_ring[slot];
                        const idx = gqa.kv_tile_start + phys * 2;
                        gqa.kv_tiles_host[slot] = attention.KvTilePair.fromTiles(
                            self.gqa_kv_tiles[idx],
                            self.gqa_kv_tiles[idx + 1],
                            .rgba16f,
                        );
                    }
                },
                .qsa => |*qsa| {
                    if (!qsa.supports_retire_q8 or qsa.full_tile_table_host.len == 0) continue;
                    for (0..active_n) |slot| {
                        const c = self.num_retired_chunks + @as(u32, @intCast(slot));
                        if (c >= qsa.full_tile_table_host.len) break;
                        const phys: usize = self.active_tile_phys_ring[slot];
                        qsa.full_tile_table_host[c] = attention.KvTilePair.fromTiles(
                            qsa.kv_ring.k_tiles[phys],
                            qsa.kv_ring.v_tiles[phys],
                            .rgba16f,
                        );
                    }
                },
                .gdn => {},
            }
        }
    }

    fn retireOldestActiveChunk(self: *GraphModel) GraphStepError!void {
        var pool = &(self.retired_kv_pool.?);
        std.debug.assert(self.num_retired_chunks < self.max_retired_per_layer);
        const retired_idx: usize = self.num_retired_chunks;
        const phys_oldest: usize = self.active_tile_phys_ring[0];
        const active_n: usize = self.active_kv_chunks;

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();
        var dispatched = false;

        for (self.layers) |*layer| {
            switch (layer.attn) {
                .gqa => |*gqa| {
                    const dst_slot = try pool.acquireSlot();
                    gqa.retired_ids_host[retired_idx] = dst_slot;
                    const idx = gqa.kv_tile_start + phys_oldest * 2;
                    try self.attn_pipes.encodeKvRetireTileToQ8(&frame, .{
                        .k_tile = self.gqa_kv_tiles[idx],
                        .v_tile = self.gqa_kv_tiles[idx + 1],
                        .pool = pool,
                        .pool_buf = self.scratch_buf,
                        .dst_slot = dst_slot,
                    }, false);
                    dispatched = true;
                },
                .qsa => |*qsa| {
                    if (!qsa.supports_retire_q8) continue;
                    const dst_slot = try pool.acquireSlot();
                    qsa.retired_ids_host[retired_idx] = dst_slot;
                    try self.attn_pipes.encodeKvRetireTileToQ8(&frame, .{
                        .k_tile = qsa.kv_ring.k_tiles[phys_oldest],
                        .v_tile = qsa.kv_ring.v_tiles[phys_oldest],
                        .pool = pool,
                        .pool_buf = self.scratch_buf,
                        .dst_slot = dst_slot,
                    }, false);
                    dispatched = true;
                },
                .gdn => {},
            }
        }

        if (dispatched) {
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
        } else {
            frame.abort();
        }

        if (active_n > 1) {
            for (0..active_n - 1) |i| {
                self.active_tile_phys_ring[i] = self.active_tile_phys_ring[i + 1];
            }
            self.active_tile_phys_ring[active_n - 1] = @intCast(phys_oldest);
        }
        self.num_retired_chunks += 1;
        self.kv_retire_events += 1;
        self.syncActiveKvTileTables();
    }

    fn retireBeforeCommitRange(self: *GraphModel, base_pos: u32, count: u32) GraphStepError!void {
        if (self.retired_kv_pool == null or count == 0) return;
        const last_chunk: u32 = (base_pos + count - 1) / self.retire_chunk_tokens;
        while (last_chunk >= self.num_retired_chunks + self.active_kv_chunks and
            self.num_retired_chunks < self.max_retired_per_layer and
            base_pos >= (self.num_retired_chunks + 1) * self.retire_chunk_tokens)
        {
            try self.retireOldestActiveChunk();
        }
    }

    fn retireExcessCompletedChunks(self: *GraphModel) GraphStepError!void {
        if (self.retired_kv_pool == null or self.seq_len == 0) return;
        const total_live_chunks: u32 = (self.seq_len + self.retire_chunk_tokens - 1) / self.retire_chunk_tokens;
        while (total_live_chunks > self.num_retired_chunks + self.kv_retire_after_chunks and
            self.num_retired_chunks < self.max_retired_per_layer and
            self.seq_len >= (self.num_retired_chunks + 1) * self.retire_chunk_tokens)
        {
            try self.retireOldestActiveChunk();
        }
    }

    pub fn resetLoopIsolationStats(self: *GraphModel) void {
        self.loop_stats = .{};
    }

    pub fn loopIsolationStats(self: *const GraphModel) LoopIsolationStats {
        return self.loop_stats;
    }

    pub fn pleStats(self: *const GraphModel) ?ple_mod.Stats {
        const prt = self.ple_rt orelse return null;
        return prt.store.stats();
    }

    pub fn hasColdExperts(self: *const GraphModel) bool {
        return self.cold_rt != null;
    }

    pub fn coldExpertStats(self: *const GraphModel) ?qwen3.ColdExpertStats {
        const crt = self.cold_rt orelse return null;
        var total_cold: u32 = 0;
        for (self.layers) |l| total_cold += l.cold_expert_count;
        return .{
            .total_cold_experts = total_cold,
            .slot_count = crt.slot_count,
            .io_backend = crt.reader.activeBackend(),
            .ane_predictor_active = false,
            .ane_predictor_uses_ne = false,
            .ane_predictor_compile_ms = 0.0,
            .cache_telemetry = crt.expert_cache.telemetry(),
        };
    }

    pub fn totalWiredBytes(self: *const GraphModel) u64 {
        return self.total_wired_bytes;
    }

    pub fn shardCount(self: *const GraphModel) usize {
        return self.weight_shards.len;
    }

    pub fn resetColdTelemetry(self: *GraphModel) void {
        if (self.cold_rt) |crt| {
            crt.expert_cache.resetTelemetry();
        }
    }

    pub fn singleExpertBytes(self: *const GraphModel) u64 {
        return self.single_expert_bytes;
    }

    pub fn totalLayersNonExpertBytes(self: *const GraphModel) u64 {
        return self.total_layers_non_expert_bytes;
    }

    pub fn bytesPerDecodeToken(self: *const GraphModel, seq_len: u32) u64 {
        const loops: u64 = self.effective_max_loops;
        var kv_bytes: u64 = 0;
        for (self.layers) |layer| {
            switch (layer.attn) {
                .gqa => {
                    kv_bytes += 2 * @as(u64, self.config.num_key_value_heads) * seq_len * self.config.head_dim * @sizeOf(f16);
                },
                .qsa => |qsa| {
                    const vis_tok: u64 = if (qsa.kv_ring.window_size > 0)
                        @min(@as(u64, seq_len), qsa.kv_ring.window_size)
                    else
                        @min(@as(u64, seq_len), @as(u64, self.config.qsa_top_k_blocks) * self.config.qsa_block_size);
                    kv_bytes += 2 * @as(u64, self.config.num_key_value_heads) * vis_tok * self.config.head_dim * @sizeOf(f16);
                },
                .gdn => {
                    const state_b = @as(u64, self.config.linear_num_value_heads) *
                        self.config.linear_key_head_dim * self.config.linear_value_head_dim * @sizeOf(f32);
                    kv_bytes += 2 * state_b; // read + write recurrent state
                },
            }
        }
        const layer_bytes: u64 = if (self.config.axes.ffn_kind == .moe_swiglu)
            self.total_layers_non_expert_bytes +
                @as(u64, self.config.num_hidden_layers) * self.config.num_experts_per_tok * self.single_expert_bytes
        else
            self.total_layers_static_bytes;
        const ls = self.config.axes.loop_schedule;
        const core_layers: u64 = if (ls.end_layer > ls.start_layer) (ls.end_layer - ls.start_layer) else self.config.num_hidden_layers;
        const extra_loop_bytes: u64 = if (loops > 1)
            (loops - 1) * (layer_bytes * core_layers / self.config.num_hidden_layers)
        else
            0;
        return layer_bytes + extra_loop_bytes + kv_bytes + self.global_head_bytes;
    }

    pub fn readLogitsF16(self: *const GraphModel) []const f16 {
        return self.logits_host[0..self.config.vocab_size];
    }

    pub fn readLastLogitsF16(self: *const GraphModel) []const f16 {
        const v: usize = self.config.vocab_size;
        return self.logits_host[self.last_logits_row * v .. (self.last_logits_row + 1) * v];
    }

    pub fn readTopHiddenF16(self: *const GraphModel, num_nodes: u32) []const f16 {
        return self.accepted_top_host[0 .. @as(usize, num_nodes) * self.config.hidden_size];
    }

    pub fn readAuxConcatF16(self: *const GraphModel, num_nodes: u32) []const f16 {
        return self.accepted_aux_host[0 .. @as(usize, num_nodes) * 3 * self.config.hidden_size];
    }

    pub fn sampleFromLastLogits(
        self: *const GraphModel,
        temperature: f32,
        rng: std.Random,
        p_res: []f32,
    ) accept_mod.VerifyError!u32 {
        try accept_mod.softmaxRow(f16, self.readLastLogitsF16(), temperature, p_res);
        return accept_mod.sampleCategorical(p_res[0..self.config.vocab_size], rng);
    }

    // -----------------------------------------------------------------------
    // GPU Linear Encoding Helpers (Compatible with `DraftHead` / `Qwen3Model`)
    // -----------------------------------------------------------------------

    pub fn encodeLinearGemv(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        lin: qwen3.LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        barrier_before: bool,
    ) dequant.EncodeError!void {
        switch (lin.dtype) {
            .q2_affine => try self.dequant_pipes.encodeGemv(frame, .q2, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .q3_affine => try self.dequant_pipes.encodeGemv(frame, .q3, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .q4_affine => try self.dequant_pipes.encodeGemv(frame, .q4, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .f16 => try self.dequant_pipes.encodeDenseGemv(frame, .f16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .bf16 => try self.dequant_pipes.encodeDenseGemv(frame, .bf16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
        }
    }

    pub fn encodeLinearBatched(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        lin: qwen3.LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        m: u32,
        barrier_before: bool,
    ) dequant.EncodeError!void {
        if (m == 1) {
            return self.encodeLinearGemv(frame, lin, x, y, barrier_before);
        }
        switch (lin.dtype) {
            .q2_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q2, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .q3_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q3, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .q4_affine => try self.dequant_pipes.encodeGemvBatch(frame, .q4, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .f16 => try self.dequant_pipes.encodeDenseGemvBatch(frame, .f16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
            .bf16 => try self.dequant_pipes.encodeDenseGemvBatch(frame, .bf16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
            }, barrier_before),
        }
    }

    pub fn encodeLinearGemmWithTile(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        lin: qwen3.LinearBinding,
        x: metal.BufferBinding,
        y: metal.BufferBinding,
        m: u32,
        wide_tile: bool,
        barrier_before: bool,
    ) dequant.EncodeError!void {
        if (m < 16) {
            return self.encodeLinearBatched(frame, lin, x, y, m, barrier_before);
        }
        switch (lin.dtype) {
            .q2_affine => try self.dequant_pipes.encodeGemm(frame, .q2, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
                .wide_tile = wide_tile,
            }, barrier_before),
            .q3_affine => try self.dequant_pipes.encodeGemm(frame, .q3, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
                .wide_tile = wide_tile,
            }, barrier_before),
            .q4_affine => try self.dequant_pipes.encodeGemm(frame, .q4, .{
                .x = x,
                .qweight = lin.weight,
                .scale_bias = lin.scale_bias.?,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
                .wide_tile = wide_tile,
            }, barrier_before),
            .f16 => try self.dequant_pipes.encodeDenseGemm(frame, .f16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
                .wide_tile = wide_tile,
            }, barrier_before),
            .bf16 => try self.dequant_pipes.encodeDenseGemm(frame, .bf16, .{
                .x = x,
                .weight = lin.weight,
                .y = y,
                .m = m,
                .n = lin.n,
                .k = lin.k,
                .wide_tile = wide_tile,
            }, barrier_before),
        }
    }

    pub fn usesWideGpuPrefill(self: *const GraphModel) bool {
        return self.gpu_prefill_mode == .wide and
            self.wide_prefill_max_tokens > 0 and
            self.supports_wide_prefill and
            self.cold_rt == null and
            self.adapter_rt == null and
            !self.enable_loop_early_exit;
    }

    pub fn prefillChunkCap(self: *const GraphModel) u32 {
        if (self.usesWideGpuPrefill()) {
            return if (self.retired_kv_pool != null)
                @min(self.wide_prefill_max_tokens, self.retire_chunk_tokens)
            else
                self.wide_prefill_max_tokens;
        }
        return @max(@as(u32, 1), self.max_tree_nodes);
    }

    // -----------------------------------------------------------------------
    // Public Inference Entrypoints: Prefill, Decode, VerifyTree, CommitTree, DraftShallow
    // -----------------------------------------------------------------------

    /// Batched prompt prefill across `tokens` starting at `start_pos`:
    /// uses wide-chunk GPU prefill (`G15` + `P0-B1` tail pruning) when
    /// `gpu_prefill_mode == .wide`, or legacy `<= max_tree_nodes` chunked
    /// `verifyTree`-style prefill when `.tree64`.
    pub fn prefillChunk(
        self: *GraphModel,
        tokens: []const u32,
        start_pos: u32,
        compute_logits: bool,
    ) GraphStepError!u32 {
        return self.prefillChunkWithOptions(tokens, start_pos, .{
            .compute_logits = compute_logits,
            .preserve_all_top_hidden = self.enable_feature_taps or self.compute_all_logits,
        });
    }

    pub fn prefillChunkWithOptions(
        self: *GraphModel,
        tokens: []const u32,
        start_pos: u32,
        opts: qwen3.PrefillChunkOptions,
    ) GraphStepError!u32 {
        if (tokens.len == 0) return error.InvalidBatchSize;
        if (start_pos + tokens.len > self.max_seq_len) return error.SequenceFull;
        for (tokens) |tid| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
        }

        if (self.usesWideGpuPrefill()) {
            const cap: usize = self.prefillChunkCap();
            var offset: usize = 0;
            var last_argmax: u32 = 0;
            while (offset < tokens.len) {
                const take: usize = @min(cap, tokens.len - offset);
                const cur_base: u32 = start_pos + @as(u32, @intCast(offset));
                const is_last_chunk = (offset + take == tokens.len);
                last_argmax = try self.prefillWideChunk(
                    tokens[offset .. offset + take],
                    cur_base,
                    is_last_chunk,
                    opts.compute_logits and is_last_chunk,
                    opts.preserve_all_top_hidden,
                );
                offset += take;
            }
            return last_argmax;
        }

        const chunk_cap: usize = @max(@as(usize, 1), @as(usize, self.max_tree_nodes));
        var offset: usize = 0;
        var last_argmax: u32 = 0;
        while (offset < tokens.len) {
            const take: usize = @min(chunk_cap, tokens.len - offset);
            const cur_base: u32 = start_pos + @as(u32, @intCast(offset));
            const is_last_chunk = (offset + take == tokens.len);

            var pos_buf: [hybrid.MAX_TREE_NODES]u32 = undefined;
            var mask_buf: [hybrid.MAX_TREE_NODES]u64 = undefined;
            var accept_nodes: [hybrid.MAX_TREE_NODES]u16 = undefined;
            for (0..take) |i| {
                pos_buf[i] = cur_base + @as(u32, @intCast(i));
                // Causal chain within the chunk: node `i` attends to `0..=i`.
                mask_buf[i] = if (i == 63) std.math.maxInt(u64) else ((@as(u64, 1) << @intCast(i + 1)) - 1);
                accept_nodes[i] = @intCast(i);
            }

            try self.retireBeforeCommitRange(cur_base, @intCast(take));
            const vr = try self.runForwardInternal(
                tokens[offset .. offset + take],
                pos_buf[0..take],
                mask_buf[0..take],
                cur_base,
                self.effective_max_loops,
                is_last_chunk and opts.compute_logits,
                0,
                null,
                true, // commit state on final loop step
                null,
                null,
            );
            // Record committed token history for subsequent PLE N-gram lookups.
            for (0..take) |i| {
                self.token_history[cur_base + @as(u32, @intCast(i))] = tokens[offset + i];
            }
            if (self.enable_feature_taps) {
                try self.commitFeatureTaps(accept_nodes[0..take]);
            }
            self.seq_len = cur_base + @as(u32, @intCast(take));
            self.mrope_next_pos = self.seq_len;
            try self.retireExcessCompletedChunks();
            if (is_last_chunk and opts.compute_logits) {
                last_argmax = vr.argmax_tokens[take - 1];
            }
            offset += take;
        }
        return last_argmax;
    }

    fn preparePleLayerWide(
        self: *GraphModel,
        prt: *PleRuntime,
        l: usize,
        token_ids: []const u32,
        base_pos: u32,
        m: u32,
        loop_step: u32,
    ) GraphStepError!void {
        if (loop_step >= 1) {
            // ADR-0019 Isolation #2: `wp_ple_f16_host[l * layer_stride ..]` was already
            // populated on `loop_step == 0` for this layer and chunk (`tokens[0..m]`),
            // and all `r = 0..R-1` passes encode into the same uncommitted `Compute.Frame`.
            // Reusing the `r = 0` binding directly avoids overwriting host-mapped memory
            // before `frame.commit()` and guarantees zero NVMe reads on `r >= 1`.
            self.loop_stats.ple_r_ge1_reuses += 1;
            return;
        }
        const layer_rows = try prt.store.layerTotalRows(@intCast(l));
        const order = self.config.ple_ngram_order;
        const h: usize = self.config.hidden_size;

        for (0..m) |node| {
            var ngram_buf: [8]u32 = undefined;
            var collected: usize = 0;
            ngram_buf[0] = token_ids[node];
            collected = 1;
            if (order > 1) {
                var back: usize = 1;
                while (collected < order and back <= node) : (back += 1) {
                    ngram_buf[collected] = token_ids[node - back];
                    collected += 1;
                }
                var past_back: u32 = 1;
                while (collected < order and base_pos >= past_back) : (past_back += 1) {
                    ngram_buf[collected] = self.token_history[base_pos - past_back];
                    collected += 1;
                }
            }
            var chrono: [8]u32 = undefined;
            for (0..collected) |k| {
                chrono[k] = ngram_buf[collected - 1 - k];
            }
            self.ple_row_indices_host[node] = ple_mod.hashPleNgram(chrono[0..collected], order, layer_rows);
        }

        const f32_slice = prt.f32_tmp[0 .. @as(usize, m) * h];
        @memset(f32_slice, 0.0);

        const before_nvme = prt.store.stats().nvme_reads;
        try prt.store.lookupForLoopStep(
            @intCast(l),
            self.ple_row_indices_host[0..m],
            1.0,
            loop_step,
            &prt.scratch,
            f32_slice,
        );
        const after_nvme = prt.store.stats().nvme_reads;
        if (loop_step == 0) {
            self.loop_stats.ple_r0_fills += 1;
        } else {
            self.loop_stats.ple_r_ge1_reuses += 1;
            self.loop_stats.ple_r_ge1_nvme_reads += (after_nvme - before_nvme);
        }

        const layer_stride = @as(usize, self.wide_prefill_max_tokens) * h;
        const dst_f16 = self.wp_ple_f16_host[l * layer_stride ..][0 .. @as(usize, m) * h];
        for (0..@as(usize, m) * h) |i| {
            dst_f16[i] = @floatCast(f32_slice[i]);
        }
    }

    fn wpPleLayerBinding(self: *const GraphModel, l: usize) metal.BufferBinding {
        const stride = @as(u64, self.wide_prefill_max_tokens) * self.config.hidden_size * 2;
        return .{ .buffer = self.scratch_buf, .offset = self.wp_ple_f16_base_off + @as(u64, l) * stride };
    }

    fn prefillWideChunk(
        self: *GraphModel,
        tokens: []const u32,
        base_pos: u32,
        is_final_chunk: bool,
        compute_logits: bool,
        preserve_all_top_hidden: bool,
    ) GraphStepError!u32 {
        const m: u32 = @intCast(tokens.len);
        if (m == 0 or m > self.wide_prefill_max_tokens) return error.InvalidBatchSize;
        if (base_pos + m > self.max_seq_len) return error.SequenceFull;
        try self.retireBeforeCommitRange(base_pos, m);

        const h: u32 = self.config.hidden_size;
        const w_stride: u32 = self.wide_prefill_max_tokens;
        const ls = self.config.axes.loop_schedule;
        const R: u32 = @max(@as(u32, 1), self.effective_max_loops);
        const core_start: u32 = if (R > 1) ls.start_layer else 0;
        const core_end: u32 = if (R > 1)
            (if (ls.end_layer > ls.start_layer) ls.end_layer else self.config.num_hidden_layers)
        else
            self.config.num_hidden_layers;
        const last_l: usize = self.config.num_hidden_layers - 1;
        const aux_needs_last = self.enable_feature_taps and
            (self.aux_layer_ids[0] == last_l or self.aux_layer_ids[1] == last_l or self.aux_layer_ids[2] == last_l);
        const allow_tail_prune = self.enable_prompt_tail_pruning and !self.compute_all_logits and
            !preserve_all_top_hidden and !aux_needs_last;
        const prune_non_final_last = allow_tail_prune and (!is_final_chunk or !compute_logits);
        const prune_final_last_to_m1 = allow_tail_prune and is_final_chunk and compute_logits and (m > 1);

        for (0..m) |i| {
            self.wp_token_ids_host[i] = tokens[i];
            self.wp_pos3d_host[i] = hybrid.Pos3D.text(base_pos + @as(u32, @intCast(i)));
        }
        if (self.ple_rt) |prt| {
            prt.scratch.resetStep();
        }

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();

        try self.moe_pipes.encodeEmbedLookup(&frame, self.embed.dtype, .{
            .weight = self.embed.weight,
            .scale_bias = self.embed.scale_bias,
            .tokens = .{ .token_ids = self.wp_token_ids_bind },
            .out = self.wp_x_bind,
            .m = m,
            .vocab_size = self.config.vocab_size,
            .hidden_size = h,
        }, false);

        if (self.wp_gr_arena) |*gr| {
            try self.hybrid_pipes.encodeGrInitStream0(&frame, .{
                .x0 = self.wp_x_bind,
                .stream = gr.binding(self.scratch_buf),
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = w_stride,
            }, true);
        }

        // Prelude layers `[0 .. core_start)`.
        for (0..core_start) |l| {
            try self.execLayerWide(&frame, l, tokens, base_pos, m, 0, R, true, false, false);
        }

        // Save `x_init` after Prelude when `R > 1 and ls.inject_x_init`.
        if (R > 1 and ls.inject_x_init) {
            if (self.wp_gr_arena) |*gr| {
                try self.hybrid_pipes.encodeGrCollapse(&frame, .{
                    .stream = gr.binding(self.scratch_buf),
                    .out = self.wp_x_init_bind,
                    .mode = .stream0,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = w_stride,
                }, true);
            } else {
                try self.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = self.wp_x_bind,
                    .dst = self.wp_x_init_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            }
        }

        // Core recurrent block `[core_start .. core_end) x R`.
        const has_coda = (core_end < self.config.num_hidden_layers);
        var r: u32 = 0;
        while (r < R) : (r += 1) {
            self.loop_stats.total_loop_steps += 1;
            if (r >= 1 and ls.inject_x_init) {
                if (self.wp_gr_arena) |*gr| {
                    try self.hybrid_pipes.encodeBlendF16(&frame, .{
                        .dst = gr.binding(self.scratch_buf),
                        .src = self.wp_x_init_bind,
                        .count = m * h,
                        .a = 0.5,
                        .b = 0.5,
                    }, true);
                } else {
                    try self.hybrid_pipes.encodeBlendF16(&frame, .{
                        .dst = self.wp_x_bind,
                        .src = self.wp_x_init_bind,
                        .count = m * h,
                        .a = 0.5,
                        .b = 0.5,
                    }, true);
                }
            }
            const is_final_pass = (r == R - 1);
            for (core_start..core_end) |l| {
                const is_terminal_exec = (!has_coda) and is_final_pass and (l == last_l);
                try self.execLayerWide(
                    &frame,
                    l,
                    tokens,
                    base_pos,
                    m,
                    r,
                    R,
                    is_final_pass,
                    is_terminal_exec and prune_non_final_last,
                    is_terminal_exec and prune_final_last_to_m1,
                );
            }
        }

        // Coda layers `[core_end .. num_hidden_layers)`.
        for (core_end..self.config.num_hidden_layers) |l| {
            const is_terminal_exec = (l == last_l);
            try self.execLayerWide(
                &frame,
                l,
                tokens,
                base_pos,
                m,
                0,
                1,
                true,
                is_terminal_exec and prune_non_final_last,
                is_terminal_exec and prune_final_last_to_m1,
            );
        }

        if (!prune_non_final_last) {
            if (self.wp_gr_arena) |*gr| {
                if (prune_final_last_to_m1) {
                    const stream_last: metal.BufferBinding = .{
                        .buffer = self.scratch_buf,
                        .offset = gr.base_offset + @as(u64, m - 1) * h * @sizeOf(f16),
                    };
                    const last_x_bind: metal.BufferBinding = .{
                        .buffer = self.scratch_buf,
                        .offset = self.wp_x_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
                    };
                    try self.hybrid_pipes.encodeGrCollapse(&frame, .{
                        .stream = stream_last,
                        .out = last_x_bind,
                        .mode = .mean,
                        .m = 1,
                        .hidden_dim = h,
                        .branch_token_stride = w_stride,
                    }, true);
                } else {
                    try self.hybrid_pipes.encodeGrCollapse(&frame, .{
                        .stream = gr.binding(self.scratch_buf),
                        .out = self.wp_x_bind,
                        .mode = .mean,
                        .m = m,
                        .hidden_dim = h,
                        .branch_token_stride = w_stride,
                    }, true);
                }
            }
            if (self.enable_feature_taps and preserve_all_top_hidden) {
                try self.moe_pipes.encodeCopyStrided(&frame, .{
                    .src = self.wp_x_bind,
                    .dst = self.accepted_top_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            }
        }

        var logits_rows_computed: u32 = 0;
        if (compute_logits and (is_final_chunk or self.compute_all_logits)) {
            if (self.compute_all_logits) {
                try self.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = self.wp_x_bind,
                    .w = self.final_norm.binding,
                    .y = self.wp_x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = self.final_norm.dtype,
                }, true);
                try self.encodeLinearGemmWithTile(&frame, self.lm_head, self.wp_x_norm_bind, self.logits_bind, m, true, true);
                try self.moe_pipes.encodeArgmaxRows(&frame, .{
                    .logits = self.logits_bind,
                    .out_token_ids = self.out_token_id_bind,
                    .num_rows = m,
                    .vocab_size = self.config.vocab_size,
                    .logits_dtype = .f16,
                }, true);
                logits_rows_computed = m;
            } else {
                const last_x_bind: metal.BufferBinding = .{
                    .buffer = self.scratch_buf,
                    .offset = self.wp_x_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
                };
                try self.attn_pipes.encodeRmsNorm(&frame, .{
                    .x = last_x_bind,
                    .w = self.final_norm.binding,
                    .y = self.wp_x_norm_bind,
                    .num_rows = 1,
                    .hidden_dim = h,
                    .eps = self.config.rms_norm_eps,
                    .w_dtype = self.final_norm.dtype,
                }, true);
                try self.encodeLinearGemv(&frame, self.lm_head, self.wp_x_norm_bind, self.logits_bind, true);
                try self.moe_pipes.encodeArgmaxRows(&frame, .{
                    .logits = self.logits_bind,
                    .out_token_ids = self.out_token_id_bind,
                    .num_rows = 1,
                    .vocab_size = self.config.vocab_size,
                    .logits_dtype = .f16,
                }, true);
                logits_rows_computed = 1;
            }
        }

        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);

        for (0..m) |i| {
            self.token_history[base_pos + @as(u32, @intCast(i))] = tokens[i];
        }
        self.seq_len = base_pos + m;
        self.mrope_next_pos = self.seq_len;
        try self.retireExcessCompletedChunks();

        if (logits_rows_computed > 0) {
            self.last_logits_row = @as(usize, logits_rows_computed) - 1;
            return self.out_token_ids_host[self.last_logits_row];
        }
        return 0;
    }

    fn execLayerWide(
        self: *GraphModel,
        frame: *metal.Compute.Frame,
        l: usize,
        tokens: []const u32,
        base_pos: u32,
        m: u32,
        loop_step: u32,
        total_loops: u32,
        is_final_pass: bool,
        skip_last_rest: bool,
        tail_m1_last: bool,
    ) GraphStepError!void {
        const layer = &self.layers[l];
        const h: u32 = self.config.hidden_size;
        const w_stride: u32 = self.wide_prefill_max_tokens;
        const multi_loop = (total_loops > 1);
        const kv_len_after: u32 = base_pos + m;

        if (self.ple_rt) |prt| {
            try self.preparePleLayerWide(prt, l, tokens, base_pos, m, loop_step);
        }

        // Pre-attention 4-branch GR read gate.
        if (self.wp_gr_arena) |*gr| {
            const gr_w = layer.gr.?;
            try self.hybrid_pipes.encodeGrComputeGates(frame, .{
                .stream = gr.binding(self.scratch_buf),
                .w_read = gr_w.read_gate,
                .w_write = gr_w.write_gate,
                .w_decay = gr_w.decay_gate,
                .g_read_out = self.wp_gr_g_read_bind,
                .g_write_out = self.wp_gr_g_write_bind,
                .decay_out = self.wp_gr_decay_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = w_stride,
                .weight_dtype = gr_w.dtype,
            }, true);
            try self.hybrid_pipes.encodeGrReadGate(frame, .{
                .stream = gr.binding(self.scratch_buf),
                .g_read = self.wp_gr_g_read_bind,
                .x_in = self.wp_x_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = w_stride,
            }, true);
        }

        if (self.ple_rt != null) {
            try self.moe_pipes.encodeResidualAdd(frame, .{
                .x = self.wp_x_bind,
                .delta = self.wpPleLayerBinding(l),
                .count = m * h,
            }, true);
        }

        try self.attn_pipes.encodeRmsNorm(frame, .{
            .x = self.wp_x_bind,
            .w = layer.input_layernorm.binding,
            .y = self.wp_x_norm_bind,
            .num_rows = m,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = layer.input_layernorm.dtype,
        }, true);

        const last_x_bind: metal.BufferBinding = .{
            .buffer = self.scratch_buf,
            .offset = self.wp_x_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
        };
        const last_x_norm_bind: metal.BufferBinding = .{
            .buffer = self.scratch_buf,
            .offset = self.wp_x_norm_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
        };
        const eff_m: u32 = if (tail_m1_last) 1 else m;
        const eff_x_bind: metal.BufferBinding = if (tail_m1_last) last_x_bind else self.wp_x_bind;

        switch (layer.attn) {
            .gdn => |g| {
                const ga = &(self.gdn_arena.?);
                const qk_dim = self.config.gdnQkDim();
                const v_dim = self.config.gdnVDim();
                const k_eff_bind: metal.BufferBinding = if (multi_loop)
                    .{ .buffer = self.scratch_buf, .offset = g.wp_k_off }
                else
                    self.wp_k_conv_bind;
                const v_eff_bind: metal.BufferBinding = if (multi_loop)
                    .{ .buffer = self.scratch_buf, .offset = g.wp_v_off }
                else
                    self.wp_v_conv_bind;
                const g_bind: metal.BufferBinding = if (multi_loop)
                    .{ .buffer = self.scratch_buf, .offset = g.wp_g_off }
                else
                    self.wp_gdn_g_bind;
                const beta_bind: metal.BufferBinding = if (multi_loop)
                    .{ .buffer = self.scratch_buf, .offset = g.wp_beta_off }
                else
                    self.wp_gdn_beta_bind;
                const q_raw_r0_bind: metal.BufferBinding = if (multi_loop and g.conv1d_w != null)
                    .{ .buffer = self.scratch_buf, .offset = g.wp_q_raw_off }
                else
                    self.wp_q_bind;
                const k_raw_r0_bind: metal.BufferBinding = if (g.conv1d_w != null) self.wp_k_bind else k_eff_bind;
                const v_raw_r0_bind: metal.BufferBinding = if (g.conv1d_w != null) self.wp_v_bind else v_eff_bind;
                const q_eff_bind: metal.BufferBinding = if (g.conv1d_w != null) self.wp_q_conv_bind else (if (loop_step == 0) q_raw_r0_bind else self.wp_q_bind);

                if (loop_step == 0) {
                    const need_q_r0 = (!skip_last_rest) or (g.conv1d_w != null);
                    if (need_q_r0) {
                        try self.encodeLinearGemmWithTile(frame, g.q_proj, self.wp_x_norm_bind, q_raw_r0_bind, m, true, true);
                        try self.encodeLinearGemmWithTile(frame, g.k_proj, self.wp_x_norm_bind, k_raw_r0_bind, m, true, false);
                    } else {
                        try self.encodeLinearGemmWithTile(frame, g.k_proj, self.wp_x_norm_bind, k_raw_r0_bind, m, true, true);
                    }
                    try self.encodeLinearGemmWithTile(frame, g.v_proj, self.wp_x_norm_bind, v_raw_r0_bind, m, true, false);
                    try self.hybrid_pipes.encodeGdnComputeGates(frame, .{
                        .x = self.wp_x_norm_bind,
                        .w_a = g.a_proj,
                        .w_b = g.b_proj,
                        .a_log = g.a_log,
                        .dt_bias = g.dt_bias,
                        .g_out = g_bind,
                        .beta_out = beta_bind,
                        .m = m,
                        .hidden_dim = h,
                        .num_v_heads = self.config.linear_num_value_heads,
                        .weight_dtype = g.ab_dtype,
                    }, false);
                    if (!skip_last_rest) {
                        if (tail_m1_last) {
                            const last_z_bind: metal.BufferBinding = .{
                                .buffer = self.scratch_buf,
                                .offset = self.wp_gdn_z_bind.offset + @as(u64, m - 1) * v_dim * @sizeOf(f16),
                            };
                            try self.encodeLinearGemv(frame, g.g_proj, last_x_norm_bind, last_z_bind, false);
                        } else {
                            try self.encodeLinearGemmWithTile(frame, g.g_proj, self.wp_x_norm_bind, self.wp_gdn_z_bind, m, true, false);
                        }
                    }
                    if (g.conv1d_w) |cw| {
                        try self.hybrid_pipes.encodeGdnConv1dSilu(frame, .{
                            .q_in = q_raw_r0_bind,
                            .k_in = k_raw_r0_bind,
                            .v_in = v_raw_r0_bind,
                            .conv_w = cw,
                            .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .conv_state_out = .{ .buffer = self.scratch_buf, .offset = g.conv_state_tmp_off },
                            .q_out = q_eff_bind,
                            .k_out = k_eff_bind,
                            .v_out = v_eff_bind,
                            .m = m,
                            .gdn_qk = qk_dim,
                            .gdn_v = v_dim,
                            .conv_k = g.conv_k,
                            .is_tree = false,
                            .write_state = true,
                        }, true);
                    }
                } else {
                    if (!skip_last_rest) {
                        try self.encodeLinearGemmWithTile(frame, g.q_proj, self.wp_x_norm_bind, self.wp_q_bind, m, true, true);
                        if (tail_m1_last) {
                            const last_z_bind: metal.BufferBinding = .{
                                .buffer = self.scratch_buf,
                                .offset = self.wp_gdn_z_bind.offset + @as(u64, m - 1) * v_dim * @sizeOf(f16),
                            };
                            try self.encodeLinearGemv(frame, g.g_proj, last_x_norm_bind, last_z_bind, false);
                        } else {
                            try self.encodeLinearGemmWithTile(frame, g.g_proj, self.wp_x_norm_bind, self.wp_gdn_z_bind, m, true, false);
                        }
                        if (g.conv1d_w) |cw| {
                            try self.hybrid_pipes.encodeGdnConv1dQSilu(frame, .{
                                .q_cur = self.wp_q_bind,
                                .q_hist_r0 = q_raw_r0_bind,
                                .conv_w = cw,
                                .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                                .q_out = q_eff_bind,
                                .m = m,
                                .gdn_qk = qk_dim,
                                .gdn_v = v_dim,
                                .conv_k = g.conv_k,
                                .is_tree = false,
                            }, true);
                        }
                    }
                }
                if (is_final_pass and g.conv1d_w != null and g.conv_k > 1) {
                    const conv_hist = g.conv_k - 1;
                    const conv_d = 2 * qk_dim + v_dim;
                    try self.moe_pipes.encodeCopyStrided(frame, .{
                        .src = .{ .buffer = self.scratch_buf, .offset = g.conv_state_tmp_off },
                        .dst = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                        .num_rows = conv_hist,
                        .copy_cols = conv_d,
                        .src_stride = conv_d,
                        .dst_stride = conv_d,
                    }, true);
                }

                const committed_s = try ga.committedBinding(self.scratch_buf, g.gdn_idx);
                const scratch_s = try ga.scratchBinding(self.scratch_buf, g.gdn_idx);
                try self.hybrid_pipes.encodeGdnPrefill(frame, .{
                    .q = q_eff_bind,
                    .k = k_eff_bind,
                    .v = v_eff_bind,
                    .g = g_bind,
                    .beta = beta_bind,
                    .z = self.wp_gdn_z_bind,
                    .norm_w = if (g.out_norm) |n| n.binding else null,
                    .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                    .state_in = committed_s,
                    .state_out = scratch_s,
                    .out = self.wp_attn_out_bind,
                    .seq_len = m,
                    .num_k_heads = self.config.linear_num_key_heads,
                    .num_v_heads = self.config.linear_num_value_heads,
                    .head_k_dim = self.config.linear_key_head_dim,
                    .head_v_dim = self.config.linear_value_head_dim,
                    .rms_eps = self.config.rms_norm_eps,
                    .write_state_only = skip_last_rest,
                    .out_start_token = if (tail_m1_last) m - 1 else 0,
                }, true);
                if (is_final_pass) {
                    try self.hybrid_pipes.encodeCommitScratchState(frame, ga, self.scratch_buf, g.gdn_idx, true);
                    self.loop_stats.gdn_final_commits += 1;
                } else {
                    self.loop_stats.gdn_scratch_evals += 1;
                }

                if (skip_last_rest) return;
                try self.encodeLinearGemmWithTile(frame, g.o_proj, self.wp_attn_out_bind, self.wp_proj_out_bind, eff_m, true, true);
            },
            .qsa => |*qsa| {
                const mrope_rot: u32 = (self.config.mrope_section.section[0] + self.config.mrope_section.section[1] + self.config.mrope_section.section[2]) * 2;
                const eff_rot_dim: u32 = if (mrope_rot > 0 and mrope_rot <= self.config.head_dim) mrope_rot else self.config.head_dim;
                const is_circular_swa = (qsa.kv_ring.ring_capacity > 0);
                const use_qsa_tiered = (qsa.supports_retire_q8 and self.retired_kv_pool != null);
                const write_kv_tiles: metal.BufferBinding = if (use_qsa_tiered)
                    .{ .buffer = self.scratch_buf, .offset = qsa.full_tile_table_off }
                else
                    qsa.kv_ring.tileTableBinding(self.scratch_buf);
                const blk_tok = self.config.qsa_block_size;

                if (is_circular_swa) {
                    const k_chunk_bind: metal.BufferBinding = if (multi_loop)
                        .{ .buffer = self.scratch_buf, .offset = qsa.wp_k_off }
                    else
                        self.wp_k_bind;
                    const v_chunk_bind: metal.BufferBinding = if (multi_loop)
                        .{ .buffer = self.scratch_buf, .offset = qsa.wp_v_off }
                    else
                        self.wp_v_bind;

                    if (loop_step == 0) {
                        if (skip_last_rest) {
                            try self.encodeLinearGemmWithTile(frame, qsa.k_proj, self.wp_x_norm_bind, k_chunk_bind, m, true, true);
                            try self.encodeLinearGemmWithTile(frame, qsa.v_proj, self.wp_x_norm_bind, v_chunk_bind, m, true, false);
                            try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                                .q = self.wp_q_bind,
                                .k = k_chunk_bind,
                                .v = v_chunk_bind,
                                .q_norm = qsa.q_norm,
                                .k_norm = qsa.k_norm,
                                .norm_dtype = qsa.qk_norm_dtype,
                                .positions_3d = self.wp_pos3d_bind,
                                .num_tokens = m,
                                .num_q_heads = self.config.num_attention_heads,
                                .num_kv_heads = self.config.num_key_value_heads,
                                .head_dim = self.config.head_dim,
                                .rotary_dim = eff_rot_dim,
                                .start_pos = base_pos,
                                .mrope = self.config.mrope_section,
                                .yarn = self.config.yarn,
                                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                                .ring_capacity = qsa.kv_ring.ring_capacity,
                                .kv_only = true,
                                .write_to_tiles = false,
                                .rms_eps = self.config.rms_norm_eps,
                                .rope_theta = self.config.rope_theta,
                            }, true);
                        } else if (tail_m1_last) {
                            try self.encodeLinearGemmWithTile(frame, qsa.k_proj, self.wp_x_norm_bind, k_chunk_bind, m, true, true);
                            try self.encodeLinearGemmWithTile(frame, qsa.v_proj, self.wp_x_norm_bind, v_chunk_bind, m, true, false);
                            try self.encodeLinearGemv(frame, qsa.q_proj, last_x_norm_bind, self.wp_q_bind, false);
                            try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                                .q = self.wp_q_bind,
                                .k = k_chunk_bind,
                                .v = v_chunk_bind,
                                .q_norm = qsa.q_norm,
                                .k_norm = qsa.k_norm,
                                .norm_dtype = qsa.qk_norm_dtype,
                                .positions_3d = self.wp_pos3d_bind,
                                .num_tokens = m - 1,
                                .num_q_heads = self.config.num_attention_heads,
                                .num_kv_heads = self.config.num_key_value_heads,
                                .head_dim = self.config.head_dim,
                                .rotary_dim = eff_rot_dim,
                                .start_pos = base_pos,
                                .mrope = self.config.mrope_section,
                                .yarn = self.config.yarn,
                                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                                .ring_capacity = qsa.kv_ring.ring_capacity,
                                .kv_only = true,
                                .write_to_tiles = false,
                                .rms_eps = self.config.rms_norm_eps,
                                .rope_theta = self.config.rope_theta,
                            }, true);
                            const kv_d = self.config.kvDim();
                            const last_k_bind: metal.BufferBinding = .{
                                .buffer = self.scratch_buf,
                                .offset = k_chunk_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16),
                            };
                            const last_v_bind: metal.BufferBinding = .{
                                .buffer = self.scratch_buf,
                                .offset = v_chunk_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16),
                            };
                            const last_p3d_bind: metal.BufferBinding = .{
                                .buffer = self.scratch_buf,
                                .offset = self.wp_pos3d_bind.offset + @as(u64, m - 1) * @sizeOf(hybrid.Pos3D),
                            };
                            try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                                .q = self.wp_q_bind,
                                .k = last_k_bind,
                                .v = last_v_bind,
                                .q_norm = qsa.q_norm,
                                .k_norm = qsa.k_norm,
                                .norm_dtype = qsa.qk_norm_dtype,
                                .positions_3d = last_p3d_bind,
                                .num_tokens = 1,
                                .num_q_heads = self.config.num_attention_heads,
                                .num_kv_heads = self.config.num_key_value_heads,
                                .head_dim = self.config.head_dim,
                                .rotary_dim = eff_rot_dim,
                                .start_pos = base_pos + m - 1,
                                .mrope = self.config.mrope_section,
                                .yarn = self.config.yarn,
                                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                                .ring_capacity = qsa.kv_ring.ring_capacity,
                                .kv_only = false,
                                .write_to_tiles = false,
                                .rms_eps = self.config.rms_norm_eps,
                                .rope_theta = self.config.rope_theta,
                            }, false);
                        } else {
                            try self.encodeLinearGemmWithTile(frame, qsa.q_proj, self.wp_x_norm_bind, self.wp_q_bind, m, true, true);
                            try self.encodeLinearGemmWithTile(frame, qsa.k_proj, self.wp_x_norm_bind, k_chunk_bind, m, true, false);
                            try self.encodeLinearGemmWithTile(frame, qsa.v_proj, self.wp_x_norm_bind, v_chunk_bind, m, true, false);
                            try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                                .q = self.wp_q_bind,
                                .k = k_chunk_bind,
                                .v = v_chunk_bind,
                                .q_norm = qsa.q_norm,
                                .k_norm = qsa.k_norm,
                                .norm_dtype = qsa.qk_norm_dtype,
                                .positions_3d = self.wp_pos3d_bind,
                                .num_tokens = m,
                                .num_q_heads = self.config.num_attention_heads,
                                .num_kv_heads = self.config.num_key_value_heads,
                                .head_dim = self.config.head_dim,
                                .rotary_dim = eff_rot_dim,
                                .start_pos = base_pos,
                                .mrope = self.config.mrope_section,
                                .yarn = self.config.yarn,
                                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                                .ring_capacity = qsa.kv_ring.ring_capacity,
                                .kv_only = false,
                                .write_to_tiles = false,
                                .rms_eps = self.config.rms_norm_eps,
                                .rope_theta = self.config.rope_theta,
                            }, true);
                        }
                    } else if (!skip_last_rest) {
                        const q_src = if (tail_m1_last) last_x_norm_bind else self.wp_x_norm_bind;
                        const p3d_src: metal.BufferBinding = if (tail_m1_last)
                            .{ .buffer = self.scratch_buf, .offset = self.wp_pos3d_bind.offset + @as(u64, m - 1) * @sizeOf(hybrid.Pos3D) }
                        else
                            self.wp_pos3d_bind;
                        try self.encodeLinearGemmWithTile(frame, qsa.q_proj, q_src, self.wp_q_bind, eff_m, true, true);
                        try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                            .q = self.wp_q_bind,
                            .k = self.wp_k_bind,
                            .v = self.wp_v_bind,
                            .q_norm = qsa.q_norm,
                            .k_norm = qsa.k_norm,
                            .norm_dtype = qsa.qk_norm_dtype,
                            .positions_3d = p3d_src,
                            .num_tokens = eff_m,
                            .num_q_heads = self.config.num_attention_heads,
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .rotary_dim = eff_rot_dim,
                            .start_pos = if (tail_m1_last) base_pos + m - 1 else base_pos,
                            .mrope = self.config.mrope_section,
                            .yarn = self.config.yarn,
                            .chunk_tokens = qsa.kv_ring.chunk_tokens,
                            .ring_capacity = qsa.kv_ring.ring_capacity,
                            .kv_only = false,
                            .write_to_tiles = false,
                            .rms_eps = self.config.rms_norm_eps,
                            .rope_theta = self.config.rope_theta,
                        }, true);
                    }

                    if (!skip_last_rest) {
                        try self.hybrid_pipes.encodeQsaSwaPrefillAttn(frame, .{
                            .q = self.wp_q_bind,
                            .kv_tiles = if (base_pos > 0) write_kv_tiles else null,
                            .k_chunk = k_chunk_bind,
                            .v_chunk = v_chunk_bind,
                            .out = self.wp_attn_out_bind,
                            .num_q = eff_m,
                            .q_start_pos = if (tail_m1_last) base_pos + m - 1 else base_pos,
                            .chunk_start_pos = base_pos,
                            .kv_len = kv_len_after,
                            .num_q_heads = self.config.num_attention_heads,
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .window_size = qsa.kv_ring.window_size,
                            .ring_capacity = qsa.kv_ring.ring_capacity,
                            .chunk_tokens = qsa.kv_ring.chunk_tokens,
                        }, true);
                    }

                    if (is_final_pass) {
                        try self.hybrid_pipes.encodeQsaPrefillKvRingCommit(frame, .{
                            .k = k_chunk_bind,
                            .v = v_chunk_bind,
                            .kv_tiles = write_kv_tiles,
                            .num_tokens = m,
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .start_pos = base_pos,
                            .chunk_tokens = qsa.kv_ring.chunk_tokens,
                            .ring_capacity = qsa.kv_ring.ring_capacity,
                        }, true);
                        const tot_blocks: u32 = (@min(kv_len_after, qsa.kv_ring.window_size) + blk_tok - 1) / blk_tok;
                        try self.hybrid_pipes.encodeQsaUpdateBlockSummary(frame, .{
                            .kv_tiles = write_kv_tiles,
                            .block_keys = .{ .buffer = self.scratch_buf, .offset = qsa.block_keys_off },
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .kv_len = kv_len_after,
                            .window_size = qsa.kv_ring.window_size,
                            .ring_capacity = qsa.kv_ring.ring_capacity,
                            .chunk_tokens = qsa.kv_ring.chunk_tokens,
                            .block_tokens = blk_tok,
                            .start_block = 0,
                            .num_blocks = @max(tot_blocks, 1),
                        }, true);
                    }
                    if (skip_last_rest) return;
                    try self.encodeLinearGemmWithTile(frame, qsa.o_proj, self.wp_attn_out_bind, self.wp_proj_out_bind, eff_m, true, true);
                } else {
                    // Full-context QSA (`ring_capacity == 0`): write K/V directly to KvTiles at r == 0.
                    if (loop_step == 0) {
                        if (skip_last_rest) {
                            try self.encodeLinearGemmWithTile(frame, qsa.k_proj, self.wp_x_norm_bind, self.wp_k_bind, m, true, true);
                            try self.encodeLinearGemmWithTile(frame, qsa.v_proj, self.wp_x_norm_bind, self.wp_v_bind, m, true, false);
                            try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                                .q = self.wp_q_bind,
                                .k = self.wp_k_bind,
                                .v = self.wp_v_bind,
                                .q_norm = qsa.q_norm,
                                .k_norm = qsa.k_norm,
                                .norm_dtype = qsa.qk_norm_dtype,
                                .positions_3d = self.wp_pos3d_bind,
                                .kv_tiles = write_kv_tiles,
                                .num_tokens = m,
                                .num_q_heads = self.config.num_attention_heads,
                                .num_kv_heads = self.config.num_key_value_heads,
                                .head_dim = self.config.head_dim,
                                .rotary_dim = eff_rot_dim,
                                .start_pos = base_pos,
                                .mrope = self.config.mrope_section,
                                .yarn = self.config.yarn,
                                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                                .num_retired_chunks = 0,
                                .ring_capacity = 0,
                                .kv_only = true,
                                .write_to_tiles = true,
                                .rms_eps = self.config.rms_norm_eps,
                                .rope_theta = self.config.rope_theta,
                            }, true);
                        } else if (tail_m1_last) {
                            try self.encodeLinearGemmWithTile(frame, qsa.k_proj, self.wp_x_norm_bind, self.wp_k_bind, m, true, true);
                            try self.encodeLinearGemmWithTile(frame, qsa.v_proj, self.wp_x_norm_bind, self.wp_v_bind, m, true, false);
                            try self.encodeLinearGemv(frame, qsa.q_proj, last_x_norm_bind, self.wp_q_bind, false);
                            try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                                .q = self.wp_q_bind,
                                .k = self.wp_k_bind,
                                .v = self.wp_v_bind,
                                .q_norm = qsa.q_norm,
                                .k_norm = qsa.k_norm,
                                .norm_dtype = qsa.qk_norm_dtype,
                                .positions_3d = self.wp_pos3d_bind,
                                .kv_tiles = write_kv_tiles,
                                .num_tokens = m - 1,
                                .num_q_heads = self.config.num_attention_heads,
                                .num_kv_heads = self.config.num_key_value_heads,
                                .head_dim = self.config.head_dim,
                                .rotary_dim = eff_rot_dim,
                                .start_pos = base_pos,
                                .mrope = self.config.mrope_section,
                                .yarn = self.config.yarn,
                                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                                .num_retired_chunks = 0,
                                .ring_capacity = 0,
                                .kv_only = true,
                                .write_to_tiles = true,
                                .rms_eps = self.config.rms_norm_eps,
                                .rope_theta = self.config.rope_theta,
                            }, true);
                            const kv_d = self.config.kvDim();
                            const last_k_bind: metal.BufferBinding = .{
                                .buffer = self.scratch_buf,
                                .offset = self.wp_k_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16),
                            };
                            const last_v_bind: metal.BufferBinding = .{
                                .buffer = self.scratch_buf,
                                .offset = self.wp_v_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16),
                            };
                            const last_p3d_bind: metal.BufferBinding = .{
                                .buffer = self.scratch_buf,
                                .offset = self.wp_pos3d_bind.offset + @as(u64, m - 1) * @sizeOf(hybrid.Pos3D),
                            };
                            try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                                .q = self.wp_q_bind,
                                .k = last_k_bind,
                                .v = last_v_bind,
                                .q_norm = qsa.q_norm,
                                .k_norm = qsa.k_norm,
                                .norm_dtype = qsa.qk_norm_dtype,
                                .positions_3d = last_p3d_bind,
                                .kv_tiles = write_kv_tiles,
                                .num_tokens = 1,
                                .num_q_heads = self.config.num_attention_heads,
                                .num_kv_heads = self.config.num_key_value_heads,
                                .head_dim = self.config.head_dim,
                                .rotary_dim = eff_rot_dim,
                                .start_pos = base_pos + m - 1,
                                .mrope = self.config.mrope_section,
                                .yarn = self.config.yarn,
                                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                                .num_retired_chunks = 0,
                                .ring_capacity = 0,
                                .kv_only = false,
                                .write_to_tiles = true,
                                .rms_eps = self.config.rms_norm_eps,
                                .rope_theta = self.config.rope_theta,
                            }, false);
                        } else {
                            try self.encodeLinearGemmWithTile(frame, qsa.q_proj, self.wp_x_norm_bind, self.wp_q_bind, m, true, true);
                            try self.encodeLinearGemmWithTile(frame, qsa.k_proj, self.wp_x_norm_bind, self.wp_k_bind, m, true, false);
                            try self.encodeLinearGemmWithTile(frame, qsa.v_proj, self.wp_x_norm_bind, self.wp_v_bind, m, true, false);
                            try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                                .q = self.wp_q_bind,
                                .k = self.wp_k_bind,
                                .v = self.wp_v_bind,
                                .q_norm = qsa.q_norm,
                                .k_norm = qsa.k_norm,
                                .norm_dtype = qsa.qk_norm_dtype,
                                .positions_3d = self.wp_pos3d_bind,
                                .kv_tiles = write_kv_tiles,
                                .num_tokens = m,
                                .num_q_heads = self.config.num_attention_heads,
                                .num_kv_heads = self.config.num_key_value_heads,
                                .head_dim = self.config.head_dim,
                                .rotary_dim = eff_rot_dim,
                                .start_pos = base_pos,
                                .mrope = self.config.mrope_section,
                                .yarn = self.config.yarn,
                                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                                .num_retired_chunks = 0,
                                .ring_capacity = 0,
                                .kv_only = false,
                                .write_to_tiles = true,
                                .rms_eps = self.config.rms_norm_eps,
                                .rope_theta = self.config.rope_theta,
                            }, true);
                        }

                        const start_block: u32 = base_pos / blk_tok;
                        const total_blocks: u32 = (kv_len_after + blk_tok - 1) / blk_tok;
                        try self.hybrid_pipes.encodeQsaUpdateBlockSummary(frame, .{
                            .kv_tiles = write_kv_tiles,
                            .block_keys = .{ .buffer = self.scratch_buf, .offset = qsa.block_keys_off },
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .kv_len = kv_len_after,
                            .window_size = 0,
                            .ring_capacity = 0,
                            .chunk_tokens = qsa.kv_ring.chunk_tokens,
                            .block_tokens = blk_tok,
                            .start_block = start_block,
                            .num_blocks = if (total_blocks > start_block) (total_blocks - start_block) else 1,
                        }, true);
                    } else if (!skip_last_rest) {
                        const q_src = if (tail_m1_last) last_x_norm_bind else self.wp_x_norm_bind;
                        const p3d_src: metal.BufferBinding = if (tail_m1_last)
                            .{ .buffer = self.scratch_buf, .offset = self.wp_pos3d_bind.offset + @as(u64, m - 1) * @sizeOf(hybrid.Pos3D) }
                        else
                            self.wp_pos3d_bind;
                        try self.encodeLinearGemmWithTile(frame, qsa.q_proj, q_src, self.wp_q_bind, eff_m, true, true);
                        try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                            .q = self.wp_q_bind,
                            .k = self.wp_k_bind,
                            .v = self.wp_v_bind,
                            .q_norm = qsa.q_norm,
                            .k_norm = qsa.k_norm,
                            .norm_dtype = qsa.qk_norm_dtype,
                            .positions_3d = p3d_src,
                            .num_tokens = eff_m,
                            .num_q_heads = self.config.num_attention_heads,
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .rotary_dim = eff_rot_dim,
                            .start_pos = if (tail_m1_last) base_pos + m - 1 else base_pos,
                            .mrope = self.config.mrope_section,
                            .yarn = self.config.yarn,
                            .chunk_tokens = qsa.kv_ring.chunk_tokens,
                            .ring_capacity = 0,
                            .kv_only = false,
                            .write_to_tiles = false,
                            .rms_eps = self.config.rms_norm_eps,
                            .rope_theta = self.config.rope_theta,
                        }, true);
                    }

                    if (skip_last_rest) return;

                    const reuse_idx = (loop_step >= 1);
                    if (reuse_idx) {
                        self.loop_stats.qsa_index_reuses += 1;
                    } else {
                        self.loop_stats.qsa_index_computes += 1;
                    }
                    var tail_idx_view: hybrid.QsaIndexBuffer = undefined;
                    const idx_buf_ptr: *const hybrid.QsaIndexBuffer = if (reuse_idx and tail_m1_last) blk: {
                        tail_idx_view = try qsa.wp_index_buf.?.nodeSlice(m - 1, 1);
                        break :blk &tail_idx_view;
                    } else &(qsa.wp_index_buf.?);
                    const qsa_kv_tiles_bind: metal.BufferBinding = if (use_qsa_tiered)
                        .{
                            .buffer = self.scratch_buf,
                            .offset = qsa.full_tile_table_off + @as(u64, self.num_retired_chunks) * @sizeOf(attention.KvTilePair),
                        }
                    else
                        qsa.kv_ring.tileTableBinding(self.scratch_buf);
                    const qsa_ret_pool_bind: ?metal.BufferBinding = if (use_qsa_tiered and self.num_retired_chunks > 0)
                        self.retired_kv_pool.?.binding(self.scratch_buf)
                    else
                        null;
                    try self.hybrid_pipes.encodeQsaAttention(frame, .{
                        .q = self.wp_q_bind,
                        .kv_tiles = qsa_kv_tiles_bind,
                        .block_keys = .{ .buffer = self.scratch_buf, .offset = qsa.block_keys_off },
                        .index_buf = idx_buf_ptr,
                        .index_metal_buf = self.scratch_buf,
                        .out = self.wp_attn_out_bind,
                        .retired_pool = qsa_ret_pool_bind,
                        .retired_chunk_ids = if (qsa_ret_pool_bind != null) qsa.retired_ids_binding else null,
                        .num_nodes = eff_m,
                        .num_q_heads = self.config.num_attention_heads,
                        .num_kv_heads = self.config.num_key_value_heads,
                        .head_dim = self.config.head_dim,
                        .kv_len = kv_len_after,
                        .window_size = 0,
                        .ring_capacity = 0,
                        .chunk_tokens = qsa.kv_ring.chunk_tokens,
                        .block_tokens = blk_tok,
                        .top_k_blocks = self.config.qsa_top_k_blocks,
                        .num_retired_chunks = if (qsa_ret_pool_bind != null) self.num_retired_chunks else 0,
                        .retired_plane_stride_bytes = if (qsa_ret_pool_bind != null) self.retired_kv_pool.?.plane_stride_bytes else 0,
                        .retired_chunk_stride_bytes = if (qsa_ret_pool_bind != null) self.retired_kv_pool.?.chunk_stride_bytes else 0,
                        .reuse_block_indices = reuse_idx,
                        .causal_q_start_pos = if (tail_m1_last) base_pos + m - 1 else base_pos,
                    }, true);
                    try self.encodeLinearGemmWithTile(frame, qsa.o_proj, self.wp_attn_out_bind, self.wp_proj_out_bind, eff_m, true, true);
                }
            },
            .gqa => |gqa| {
                const kv_d = self.config.kvDim();
                const has_retired = (self.retired_kv_pool != null and self.num_retired_chunks > 0);
                const rpool_bind: ?metal.BufferBinding = if (has_retired) self.retired_kv_pool.?.binding(self.scratch_buf) else null;
                const ret_plane_stride: u32 = if (has_retired) self.retired_kv_pool.?.plane_stride_bytes else 0;
                const ret_chunk_stride: u32 = if (has_retired) self.retired_kv_pool.?.chunk_stride_bytes else 0;

                if (loop_step == 0) {
                    if (skip_last_rest) {
                        try self.encodeLinearGemmWithTile(frame, gqa.k_proj, self.wp_x_norm_bind, self.wp_k_bind, m, true, true);
                        try self.encodeLinearGemmWithTile(frame, gqa.v_proj, self.wp_x_norm_bind, self.wp_v_bind, m, true, false);
                        try self.attn_pipes.encodePrefillQkNormRopeKvWrite(frame, .{
                            .q = self.wp_q_bind,
                            .k = self.wp_k_bind,
                            .v = self.wp_v_bind,
                            .q_norm = gqa.q_norm,
                            .k_norm = gqa.k_norm,
                            .kv_tiles = gqa.kv_tiles_binding,
                            .num_tokens = m,
                            .num_q_heads = self.config.num_attention_heads,
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .start_pos = base_pos,
                            .chunk_tokens = self.chunk_tokens,
                            .num_retired_chunks = self.num_retired_chunks,
                            .kv_only = true,
                            .rms_eps = self.config.rms_norm_eps,
                            .rope_theta = self.config.rope_theta,
                            .norm_dtype = gqa.qk_norm_dtype,
                        }, true);
                        return;
                    } else if (tail_m1_last) {
                        try self.encodeLinearGemmWithTile(frame, gqa.k_proj, self.wp_x_norm_bind, self.wp_k_bind, m, true, true);
                        try self.encodeLinearGemmWithTile(frame, gqa.v_proj, self.wp_x_norm_bind, self.wp_v_bind, m, true, false);
                        try self.encodeLinearGemv(frame, gqa.q_proj, last_x_norm_bind, self.wp_q_bind, false);
                        try self.attn_pipes.encodePrefillQkNormRopeKvWrite(frame, .{
                            .q = self.wp_q_bind,
                            .k = self.wp_k_bind,
                            .v = self.wp_v_bind,
                            .q_norm = gqa.q_norm,
                            .k_norm = gqa.k_norm,
                            .kv_tiles = gqa.kv_tiles_binding,
                            .num_tokens = m - 1,
                            .num_q_heads = self.config.num_attention_heads,
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .start_pos = base_pos,
                            .chunk_tokens = self.chunk_tokens,
                            .num_retired_chunks = self.num_retired_chunks,
                            .kv_only = true,
                            .rms_eps = self.config.rms_norm_eps,
                            .rope_theta = self.config.rope_theta,
                            .norm_dtype = gqa.qk_norm_dtype,
                        }, true);
                        const last_k_bind: metal.BufferBinding = .{
                            .buffer = self.scratch_buf,
                            .offset = self.wp_k_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16),
                        };
                        const last_v_bind: metal.BufferBinding = .{
                            .buffer = self.scratch_buf,
                            .offset = self.wp_v_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16),
                        };
                        try self.attn_pipes.encodePrefillQkNormRopeKvWrite(frame, .{
                            .q = self.wp_q_bind,
                            .k = last_k_bind,
                            .v = last_v_bind,
                            .q_norm = gqa.q_norm,
                            .k_norm = gqa.k_norm,
                            .kv_tiles = gqa.kv_tiles_binding,
                            .num_tokens = 1,
                            .num_q_heads = self.config.num_attention_heads,
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .start_pos = base_pos + m - 1,
                            .chunk_tokens = self.chunk_tokens,
                            .num_retired_chunks = self.num_retired_chunks,
                            .kv_only = false,
                            .rms_eps = self.config.rms_norm_eps,
                            .rope_theta = self.config.rope_theta,
                            .norm_dtype = gqa.qk_norm_dtype,
                        }, false);
                    } else {
                        try self.encodeLinearGemmWithTile(frame, gqa.q_proj, self.wp_x_norm_bind, self.wp_q_bind, m, true, true);
                        try self.encodeLinearGemmWithTile(frame, gqa.k_proj, self.wp_x_norm_bind, self.wp_k_bind, m, true, false);
                        try self.encodeLinearGemmWithTile(frame, gqa.v_proj, self.wp_x_norm_bind, self.wp_v_bind, m, true, false);
                        try self.attn_pipes.encodePrefillQkNormRopeKvWrite(frame, .{
                            .q = self.wp_q_bind,
                            .k = self.wp_k_bind,
                            .v = self.wp_v_bind,
                            .q_norm = gqa.q_norm,
                            .k_norm = gqa.k_norm,
                            .kv_tiles = gqa.kv_tiles_binding,
                            .num_tokens = m,
                            .num_q_heads = self.config.num_attention_heads,
                            .num_kv_heads = self.config.num_key_value_heads,
                            .head_dim = self.config.head_dim,
                            .start_pos = base_pos,
                            .chunk_tokens = self.chunk_tokens,
                            .num_retired_chunks = self.num_retired_chunks,
                            .kv_only = false,
                            .rms_eps = self.config.rms_norm_eps,
                            .rope_theta = self.config.rope_theta,
                            .norm_dtype = gqa.qk_norm_dtype,
                        }, true);
                    }
                } else {
                    if (skip_last_rest) return;
                    const q_src = if (tail_m1_last) last_x_norm_bind else self.wp_x_norm_bind;
                    try self.encodeLinearGemmWithTile(frame, gqa.q_proj, q_src, self.wp_q_bind, eff_m, true, true);
                    // Rotate Q in-place without overwriting committed r = 0 KvTiles using QsaPrefillRope (write_to_tiles = false, 1D RoPE).
                    try self.hybrid_pipes.encodeQsaPrefillRope(frame, .{
                        .q = self.wp_q_bind,
                        .k = self.wp_k_bind,
                        .v = self.wp_v_bind,
                        .q_norm = gqa.q_norm,
                        .k_norm = gqa.k_norm,
                        .norm_dtype = gqa.qk_norm_dtype,
                        .num_tokens = eff_m,
                        .num_q_heads = self.config.num_attention_heads,
                        .num_kv_heads = self.config.num_key_value_heads,
                        .head_dim = self.config.head_dim,
                        .start_pos = if (tail_m1_last) base_pos + m - 1 else base_pos,
                        .mrope = .{ .section = .{ self.config.head_dim / 2, 0, 0 } },
                        .chunk_tokens = self.chunk_tokens,
                        .kv_only = false,
                        .write_to_tiles = false,
                        .rms_eps = self.config.rms_norm_eps,
                        .rope_theta = self.config.rope_theta,
                    }, true);
                }

                try self.attn_pipes.encodeGqaCausalPrefillFlashAttn(frame, .{
                    .q = self.wp_q_bind,
                    .kv_tiles = gqa.kv_tiles_binding,
                    .retired_pool = rpool_bind,
                    .retired_chunk_ids = if (has_retired) gqa.retired_ids_binding else null,
                    .out = self.wp_attn_out_bind,
                    .num_q = eff_m,
                    .q_start_pos = if (tail_m1_last) base_pos + m - 1 else base_pos,
                    .kv_len = kv_len_after,
                    .num_q_heads = self.config.num_attention_heads,
                    .num_kv_heads = self.config.num_key_value_heads,
                    .head_dim = self.config.head_dim,
                    .chunk_tokens = self.chunk_tokens,
                    .num_retired_chunks = if (has_retired) self.num_retired_chunks else 0,
                    .retired_plane_stride_bytes = ret_plane_stride,
                    .retired_chunk_stride_bytes = ret_chunk_stride,
                }, true);
                try self.encodeLinearGemmWithTile(frame, gqa.o_proj, self.wp_attn_out_bind, self.wp_proj_out_bind, eff_m, true, true);
            },
        }

        // Post-attention residual update (4-branch GR or standard residual add).
        if (self.wp_gr_arena) |*gr| {
            const stream_eff: metal.BufferBinding = if (tail_m1_last)
                .{ .buffer = self.scratch_buf, .offset = gr.base_offset + @as(u64, m - 1) * h * @sizeOf(f16) }
            else
                gr.binding(self.scratch_buf);
            const decay_eff: metal.BufferBinding = if (tail_m1_last)
                .{ .buffer = self.scratch_buf, .offset = self.wp_gr_decay_bind.offset + @as(u64, m - 1) * 4 * 4 }
            else
                self.wp_gr_decay_bind;
            const gwrite_eff: metal.BufferBinding = if (tail_m1_last)
                .{ .buffer = self.scratch_buf, .offset = self.wp_gr_g_write_bind.offset + @as(u64, m - 1) * 4 * 4 }
            else
                self.wp_gr_g_write_bind;
            const gread_eff: metal.BufferBinding = if (tail_m1_last)
                .{ .buffer = self.scratch_buf, .offset = self.wp_gr_g_read_bind.offset + @as(u64, m - 1) * 4 * 4 }
            else
                self.wp_gr_g_read_bind;
            try self.hybrid_pipes.encodeGrWriteGate(frame, .{
                .stream = stream_eff,
                .y_delta = self.wp_proj_out_bind,
                .decay = decay_eff,
                .g_write = gwrite_eff,
                .m = eff_m,
                .hidden_dim = h,
                .branch_token_stride = w_stride,
            }, true);
            try self.hybrid_pipes.encodeGrReadGate(frame, .{
                .stream = stream_eff,
                .g_read = gread_eff,
                .x_in = eff_x_bind,
                .m = eff_m,
                .hidden_dim = h,
                .branch_token_stride = w_stride,
            }, true);
        } else {
            try self.moe_pipes.encodeResidualAdd(frame, .{
                .x = eff_x_bind,
                .delta = self.wp_proj_out_bind,
                .count = eff_m * h,
            }, true);
        }

        try self.attn_pipes.encodeRmsNorm(frame, .{
            .x = eff_x_bind,
            .w = layer.post_attention_layernorm.binding,
            .y = self.wp_x_norm_bind,
            .num_rows = eff_m,
            .hidden_dim = h,
            .eps = self.config.rms_norm_eps,
            .w_dtype = layer.post_attention_layernorm.dtype,
        }, true);

        switch (self.config.axes.ffn_kind) {
            .dense_swiglu => {
                try self.encodeLinearGemmWithTile(frame, layer.dense_gate, self.wp_x_norm_bind, self.wp_dense_gate_bind, eff_m, true, true);
                try self.encodeLinearGemmWithTile(frame, layer.dense_up, self.wp_x_norm_bind, self.wp_dense_up_bind, eff_m, true, false);
                try self.moe_pipes.encodeSiluMul(frame, .{
                    .gate = self.wp_dense_gate_bind,
                    .up = self.wp_dense_up_bind,
                    .out = self.wp_dense_act_bind,
                    .count = eff_m * self.config.intermediate_size,
                }, true);
                try self.encodeLinearGemmWithTile(frame, layer.dense_down, self.wp_dense_act_bind, self.wp_proj_out_bind, eff_m, true, true);
                if (self.wp_gr_arena) |*gr| {
                    const stream_eff: metal.BufferBinding = if (tail_m1_last)
                        .{ .buffer = self.scratch_buf, .offset = gr.base_offset + @as(u64, m - 1) * h * @sizeOf(f16) }
                    else
                        gr.binding(self.scratch_buf);
                    const decay_eff: metal.BufferBinding = if (tail_m1_last)
                        .{ .buffer = self.scratch_buf, .offset = self.wp_gr_decay_bind.offset + @as(u64, m - 1) * 4 * 4 }
                    else
                        self.wp_gr_decay_bind;
                    const gwrite_eff: metal.BufferBinding = if (tail_m1_last)
                        .{ .buffer = self.scratch_buf, .offset = self.wp_gr_g_write_bind.offset + @as(u64, m - 1) * 4 * 4 }
                    else
                        self.wp_gr_g_write_bind;
                    try self.hybrid_pipes.encodeGrWriteGate(frame, .{
                        .stream = stream_eff,
                        .y_delta = self.wp_proj_out_bind,
                        .decay = decay_eff,
                        .g_write = gwrite_eff,
                        .m = eff_m,
                        .hidden_dim = h,
                        .branch_token_stride = w_stride,
                    }, true);
                } else {
                    try self.moe_pipes.encodeResidualAdd(frame, .{
                        .x = eff_x_bind,
                        .delta = self.wp_proj_out_bind,
                        .count = eff_m * h,
                    }, true);
                }
            },
            .moe_swiglu => {
                const ws_ptr = &(self.wp_moe_ws.?);
                const router_src: moe.RouterSource = if (layer.router_quant_dtype) |rqd| .{
                    .quant_weights = .{
                        .qweight = layer.router_binding,
                        .scale_bias = layer.router_scale_bias.?,
                        .bits = switch (rqd) {
                            .q2_affine => .q2,
                            .q3_affine => .q3,
                            .q4_affine => .q4,
                            else => return error.InvalidBatchSize,
                        },
                    },
                } else .{
                    .weights = .{
                        .binding = layer.router_binding,
                        .dtype = layer.router_dtype,
                    },
                };
                const residual_bind: ?metal.BufferBinding = if (self.wp_gr_arena != null) null else eff_x_bind;
                const out_bind: metal.BufferBinding = if (self.wp_gr_arena != null) self.wp_proj_out_bind else eff_x_bind;
                const moe_args: moe.MoELayerArgs = .{
                    .x = self.wp_x_norm_bind,
                    .router = router_src,
                    .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                    .expert_table = layer.expert_table,
                    .workspace = ws_ptr,
                    .workspace_buf = self.scratch_buf,
                    .residual = residual_bind,
                    .out = out_bind,
                    .m = eff_m,
                    .num_experts = self.config.num_experts,
                    .top_k = self.config.num_experts_per_tok,
                    .hidden_size = h,
                    .intermediate_size = self.config.moe_intermediate_size,
                    .norm_topk_prob = self.config.norm_topk_prob,
                    .shared_expert = layer.shared_expert,
                    .shared_weights_base = if (layer.shared_expert_buf) |sb| .{ .buffer = sb, .offset = 0 } else null,
                    .shared_expert_dtype = layer.shared_expert_dtype,
                    .shared_intermediate_size = self.config.shared_expert_intermediate_size,
                    .shared_gate_weight = layer.shared_gate_weight,
                    .shared_gate_dtype = layer.shared_gate_dtype,
                };
                if (layer.is_hetero_moe) {
                    try self.moe_pipes.encodeMoELayerHetero(frame, .{
                        .base = moe_args,
                        .expert_schemes = layer.expert_dtypes_binding.?,
                        .scheme_counts = layer.expert_scheme_counts,
                    }, true);
                } else {
                    switch (layer.expert_weight_dtype) {
                        .q2_affine => try self.moe_pipes.encodeMoELayer(frame, .q2, moe_args, true),
                        .q3_affine => try self.moe_pipes.encodeMoELayer(frame, .q3, moe_args, true),
                        .q4_affine => try self.moe_pipes.encodeMoELayer(frame, .q4, moe_args, true),
                        .f16 => try self.moe_pipes.encodeMoELayerFloat(frame, .f16, moe_args, true),
                        .bf16 => try self.moe_pipes.encodeMoELayerFloat(frame, .bf16, moe_args, true),
                    }
                }
                if (self.wp_gr_arena) |*gr| {
                    const stream_eff: metal.BufferBinding = if (tail_m1_last)
                        .{ .buffer = self.scratch_buf, .offset = gr.base_offset + @as(u64, m - 1) * h * @sizeOf(f16) }
                    else
                        gr.binding(self.scratch_buf);
                    const decay_eff: metal.BufferBinding = if (tail_m1_last)
                        .{ .buffer = self.scratch_buf, .offset = self.wp_gr_decay_bind.offset + @as(u64, m - 1) * 4 * 4 }
                    else
                        self.wp_gr_decay_bind;
                    const gwrite_eff: metal.BufferBinding = if (tail_m1_last)
                        .{ .buffer = self.scratch_buf, .offset = self.wp_gr_g_write_bind.offset + @as(u64, m - 1) * 4 * 4 }
                    else
                        self.wp_gr_g_write_bind;
                    try self.hybrid_pipes.encodeGrWriteGate(frame, .{
                        .stream = stream_eff,
                        .y_delta = self.wp_proj_out_bind,
                        .decay = decay_eff,
                        .g_write = gwrite_eff,
                        .m = eff_m,
                        .hidden_dim = h,
                        .branch_token_stride = w_stride,
                    }, true);
                }
            },
        }

        // Capture EAGLE-3 aux taps on final loop pass if enabled.
        if (self.enable_feature_taps and is_final_pass and !tail_m1_last) {
            for (self.aux_layer_ids, 0..) |tap_l, slot| {
                if (tap_l == @as(u32, @intCast(l))) {
                    if (self.wp_gr_arena) |*gr| {
                        try self.hybrid_pipes.encodeGrCollapse(frame, .{
                            .stream = gr.binding(self.scratch_buf),
                            .out = self.wp_x_bind,
                            .mode = .mean,
                            .m = m,
                            .hidden_dim = h,
                            .branch_token_stride = w_stride,
                        }, true);
                    }
                    try self.moe_pipes.encodeCopyStrided(frame, .{
                        .src = self.wp_x_bind,
                        .dst = self.accepted_aux_bind,
                        .num_rows = m,
                        .copy_cols = h,
                        .src_stride = h,
                        .dst_stride = 3 * h,
                        .dst_col_off = @as(u32, @intCast(slot)) * h,
                    }, true);
                }
            }
        }
    }

    pub fn prefillToken(self: *GraphModel, token_id: u32, pos: u32) GraphStepError!void {
        const single = [_]u32{token_id};
        _ = try self.prefillChunk(&single, pos, false);
    }

    /// Single-step decode at `pos` with full `LoopSchedule` (`R = effective_max_loops`),
    /// enforcing all four `ADR-0019` loop isolations and committing KV/GDN state once at `r = R - 1`.
    pub fn stepDecode(self: *GraphModel, token_id: u32, pos: u32) GraphStepError!u32 {
        if (pos >= self.max_seq_len) return error.SequenceFull;
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;

        try self.retireBeforeCommitRange(pos, 1);
        const tok = [_]u32{token_id};
        const p = [_]u32{pos};
        const mask = [_]u64{1};
        const use_custom_p3d = (self.mrope_next_pos > 0 and self.mrope_next_pos != pos);
        const custom_p3d = [_]hybrid.Pos3D{hybrid.Pos3D.text(if (use_custom_p3d) self.mrope_next_pos else pos)};
        const vr = try self.runForwardInternal(
            &tok,
            &p,
            &mask,
            pos,
            self.effective_max_loops,
            true,
            0,
            null,
            true, // commit state on final pass `r == R - 1`
            if (use_custom_p3d) &custom_p3d else null,
            null,
        );
        self.token_history[pos] = token_id;
        if (self.enable_feature_taps) {
            const acc = [_]u16{0};
            try self.commitFeatureTaps(&acc);
        }
        self.seq_len = pos + 1;
        self.mrope_next_pos = if (use_custom_p3d) (custom_p3d[0].t + 1) else self.seq_len;
        try self.retireExcessCompletedChunks();
        return vr.argmax_tokens[0];
    }

    /// Evaluates an `M = 1..64` speculative candidate tree under full `Core x R`
    /// without mutating committed KV tiles or committed `GDN` state `S_t`.
    /// Intermediate loop steps `r < R - 1` skip full-vocabulary `lm_head`.
    pub fn verifyTree(
        self: *GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
    ) GraphStepError!qwen3.VerifyTreeResult {
        return self.verifyTreeWithLoops(token_ids, positions, ancestor_masks, base_pos, self.effective_max_loops);
    }

    pub fn verifyTreeWithLoops(
        self: *GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        num_loops: u32,
    ) GraphStepError!qwen3.VerifyTreeResult {
        if (self.max_tree_nodes <= 1) return error.TreeNotEnabled;
        if (token_ids.len == 0 or token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (positions.len != token_ids.len or ancestor_masks.len != token_ids.len) return error.InvalidBatchSize;
        for (token_ids, positions) |tid, pos| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
            if (pos >= self.max_seq_len) return error.SequenceFull;
        }
        const vr = try self.runForwardInternal(
            token_ids,
            positions,
            ancestor_masks,
            base_pos,
            num_loops,
            true,
            0,
            null,
            false, // NEVER mutate committed KV / GDN state during tree verification
            null,
            null,
        );
        self.last_verified_nodes = @intCast(token_ids.len);
        for (token_ids, 0..) |tid, i| {
            self.last_verified_tokens[i] = tid;
            self.last_verified_masks[i] = ancestor_masks[i];
        }
        return vr;
    }

    /// Commits the accepted path `accepted_nodes[0..k]` from the most recent
    /// `verifyTree` pass into committed `KvTile`s, `HybridKvRing` + `block_keys`,
    /// `GDN` committed `S_t` + `conv_state`, feature taps, and PLE `token_history`.
    pub fn commitTree(self: *GraphModel, base_pos: u32, accepted_nodes: []const u16) GraphStepError!void {
        return self.commitTreeKv(base_pos, accepted_nodes);
    }

    pub fn commitTreeKv(self: *GraphModel, base_pos: u32, accepted_nodes: []const u16) GraphStepError!void {
        if (accepted_nodes.len == 0) return;
        if (accepted_nodes.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (base_pos + accepted_nodes.len > self.max_seq_len) return error.SequenceFull;
        try self.retireBeforeCommitRange(base_pos, @intCast(accepted_nodes.len));

        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();
        var had_dispatch = false;

        for (self.layers) |*layer| {
            switch (layer.attn) {
                .gdn => |g| {
                    const ga = &(self.gdn_arena.?);
                    try self.hybrid_pipes.encodeCommitTreeState(&frame, .{
                        .k_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_k_off },
                        .v_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_v_off },
                        .g_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                        .beta_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                        .accepted_nodes = .{ .inline_u16 = accepted_nodes },
                        .accepted_count = @intCast(accepted_nodes.len),
                        .state_in = try ga.committedBinding(self.scratch_buf, g.gdn_idx),
                        .state_out = try ga.committedBinding(self.scratch_buf, g.gdn_idx),
                        .num_k_heads = self.config.linear_num_key_heads,
                        .num_v_heads = self.config.linear_num_value_heads,
                        .head_k_dim = self.config.linear_key_head_dim,
                        .head_v_dim = self.config.linear_value_head_dim,
                    }, had_dispatch);
                    had_dispatch = true;
                    if (g.conv1d_w != null and g.conv_k > 1) {
                        try self.hybrid_pipes.encodeGdnConv1dCommit(&frame, .{
                            .q_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_q_raw_off },
                            .k_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_k_raw_off },
                            .v_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_v_raw_off },
                            .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .conv_state_out = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .accepted_nodes = accepted_nodes,
                            .gdn_qk = self.config.gdnQkDim(),
                            .gdn_v = self.config.gdnVDim(),
                            .conv_k = g.conv_k,
                        }, true);
                    }
                },
                .qsa => |*qsa| {
                    try self.encodeQsaCommitAccepted(&frame, qsa, base_pos, accepted_nodes, had_dispatch);
                    had_dispatch = true;
                },
                .gqa => |gqa| {
                    try self.encodeGqaCommitAccepted(&frame, gqa, base_pos, accepted_nodes, had_dispatch);
                    had_dispatch = true;
                },
            }
        }

        if (self.enable_feature_taps) {
            try self.encodeCommitFeatureTaps(&frame, accepted_nodes, had_dispatch);
            had_dispatch = true;
        }

        if (had_dispatch) {
            const sig = try frame.commit();
            try self.compute.wait(sig, self.timeout_ms);
        } else {
            frame.abort();
        }

        // Advance committed token history for PLE N-gram hashes.
        for (accepted_nodes, 0..) |nid, i| {
            if (nid < self.last_verified_nodes) {
                self.token_history[base_pos + @as(u32, @intCast(i))] = self.last_verified_tokens[nid];
            }
        }
        self.seq_len = base_pos + @as(u32, @intCast(accepted_nodes.len));
        self.mrope_next_pos = self.seq_len;
        try self.retireExcessCompletedChunks();
    }

    /// M6.5 G14 multimodal prefill options for zero-copy `IOSurface` vision/video ingress.
    pub const MultimodalPrefillOptions = struct {
        patch_size: u32 = 16,
        t_patch: u32 = 1,
        spatial_merge: u32 = 2,
        norm: vision_mod.NormParams = vision_mod.NormParams.QWEN_VL_DEFAULT,
        compute_logits: bool = true,
        budget_override: ?budget_mod.Budget = null,
    };

    pub const MultimodalPrefillResult = struct {
        first_token: u32,
        prefix_tokens: u32,
        vision_tokens: u32,
        suffix_tokens: u32,
        total_prompt_tokens: u32,
        next_mrope_pos: u32,
        grid: vision_mod.VisionGridSpec,
        frame_width: u32,
        frame_height: u32,
        frame_count: u32,
        all_frames_zero_copy: bool,
        all_frames_page_aligned: bool,
        vision_encode_us: f64,
    };

    pub const MultimodalPrefillError = GraphStepError || vision_mod.VisionIngress.InitError ||
        vision_mod.EncodeError || vision_mod.MropeError || sysinfo.Error || vmstat.Error ||
        budget_mod.PlanError || arena_mod.WiredArena.InitError ||
        arena_mod.WiredArena.AllocError || metal.WrapError || error{
        EmptyFrames,
        FrameGeometryMismatch,
        UnsupportedVisionFormat,
        OutOfMemory,
    };

    /// Zero-copy multimodal prefill (`G14`): reads `IOSurface`-backed `VisionFrame`s
    /// (`ImageIO` PNG or hardware `VideoToolbox` H.264/HEVC) directly as `MTLTexture`
    /// planes (`0` host copies), runs Metal 4 patchify + `2x2` spatial-temporal merge +
    /// linear embed into visual token positions, builds true 3D `[t, h, w]` interleaved
    /// MRoPE (`[11, 11, 10]`) positions via `buildMultimodalMropePositions`, and prefills
    /// `[prefix_tokens | vision_tokens | suffix_tokens]` through `GraphModel`.
    pub fn prefillMultimodal(
        self: *GraphModel,
        allocator: std.mem.Allocator,
        prefix_tokens: []const u32,
        suffix_tokens: []const u32,
        frames: []const metal.VisionFrame,
        opts: MultimodalPrefillOptions,
    ) MultimodalPrefillError!MultimodalPrefillResult {
        if (frames.len == 0) return error.EmptyFrames;
        const info0 = try frames[0].info();
        const fmt0 = info0.format();
        if (fmt0 != .bgra8 and fmt0 != .nv12) return error.UnsupportedVisionFormat;
        const width = info0.width();
        const height = info0.height();
        const is_full_range = info0.isFullRange();

        var all_zc = info0.zeroCopyVerified();
        var all_aligned = (info0.baseAddress() % page.APPLE_PAGE_SIZE) == 0;
        for (frames[1..]) |fr| {
            const inf = try fr.info();
            if (inf.width() != width or inf.height() != height or inf.format() != fmt0) {
                return error.FrameGeometryMismatch;
            }
            all_zc = all_zc and inf.zeroCopyVerified();
            all_aligned = all_aligned and ((inf.baseAddress() % page.APPLE_PAGE_SIZE) == 0);
        }

        const eff_t_patch: u32 = if (opts.t_patch == 2 and frames.len >= 2) 2 else 1;
        const tube_count: u32 = @intCast((frames.len + eff_t_patch - 1) / eff_t_patch);
        const geom: vision_mod.PatchGeometry = .{
            .width = width,
            .height = height,
            .patch_size = opts.patch_size,
            .t_patch = eff_t_patch,
            .spatial_merge = opts.spatial_merge,
        };
        try geom.validate();
        const tokens_per_tube = try geom.numTokens();
        const total_vis_tokens = std.math.mul(u32, tube_count, tokens_per_tube) catch return error.Overflow;
        const token_in_dim = try geom.tokenInDim();

        const prefix_u32: u32 = @intCast(prefix_tokens.len);
        const suffix_u32: u32 = @intCast(suffix_tokens.len);
        const total_tokens = prefix_u32 + total_vis_tokens + suffix_u32;
        if (total_tokens == 0) return error.InvalidBatchSize;
        if (total_tokens > self.max_seq_len) return error.SequenceFull;

        const grid: vision_mod.VisionGridSpec = .{
            .grid_t = tube_count,
            .grid_h = geom.mergedH(),
            .grid_w = geom.mergedW(),
        };

        const facts = try sysinfo.Facts.read();
        const dev_info = try self.device.info();
        const wired_budget = opts.budget_override orelse budget_mod.Budget.fromHost(facts, dev_info, .{});

        // Preallocate `VisionIngress` workspace + deterministic wired projection matrix `[H, token_in_dim] f16`.
        var ingress = try vision_mod.VisionIngress.init(self.device, self.library, wired_budget, .{
            .max_tokens = total_vis_tokens,
            .max_patch_dim = token_in_dim,
            .max_hidden_size = self.config.hidden_size,
            .max_mrope_tokens = total_tokens,
        });
        defer ingress.deinit();

        const proj_elems = std.math.mul(usize, self.config.hidden_size, token_in_dim) catch return error.Overflow;
        const proj_bytes = try page.alignUp(proj_elems * @sizeOf(f16));
        var proj_arena = try arena_mod.WiredArena.init(proj_bytes, wired_budget);
        defer proj_arena.deinit();
        const proj_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, proj_arena.bytes()[0 .. proj_elems * @sizeOf(f16)]));
        const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(token_in_dim)));
        for (proj_f16, 0..) |*w, idx| {
            const h_mix = (@as(u64, idx) *% 0x9e3779b97f4a7c15) ^ 0x517cc1b727220a95;
            const unit: f32 = (@as(f32, @floatFromInt(h_mix & 0xFFFF)) / 32767.5) - 1.0;
            w.* = @floatCast(unit * inv_scale * 0.25);
        }
        const proj_buf = try metal.Buffer.wrap(self.device, proj_arena.bytes());
        defer proj_buf.close();

        try self.compute.makeResident(&.{ ingress.buffer(), proj_buf });
        defer self.compute.removeResident(&.{ ingress.buffer(), proj_buf }) catch |err| {
            std.debug.panic("removeResident(vision_ingress) failed: {}", .{err});
        };
        try self.compute.makeVisionFramesResident(frames);
        defer self.compute.removeVisionFramesResident(frames) catch |err| {
            std.debug.panic("removeVisionFramesResident failed: {}", .{err});
        };

        // 1. Run zero-copy Metal 4 patchify + 2x2 merge + linear embed directly from IOSurface textures.
        const vis_t0 = try vmstat.nowNs();
        {
            var vframe = try self.compute.beginFrame(self.timeout_ms);
            errdefer vframe.abort();
            var tube: u32 = 0;
            while (tube < tube_count) : (tube += 1) {
                const f0_idx: usize = @as(usize, tube) * eff_t_patch;
                const f1_opt: ?metal.VisionFrame = if (eff_t_patch == 2)
                    frames[@min(f0_idx + 1, frames.len - 1)]
                else
                    null;
                const tok_off: u32 = tube * tokens_per_tube;
                const proj_bind: metal.BufferBinding = .{ .buffer = proj_buf, .offset = 0 };
                switch (fmt0) {
                    .bgra8 => try ingress.encodePatchifyEmbedBgra8(
                        &vframe,
                        frames[f0_idx],
                        f1_opt,
                        proj_bind,
                        null,
                        .f16,
                        .f16,
                        self.config.hidden_size,
                        geom,
                        opts.norm,
                        tok_off,
                        tube > 0,
                    ),
                    .nv12 => try ingress.encodePatchifyEmbedNv12(
                        &vframe,
                        frames[f0_idx],
                        f1_opt,
                        proj_bind,
                        null,
                        .f16,
                        .f16,
                        self.config.hidden_size,
                        geom,
                        is_full_range,
                        opts.norm,
                        tok_off,
                        tube > 0,
                    ),
                    else => unreachable,
                }
            }
            const vsig = try vframe.commit();
            try self.compute.wait(vsig, self.timeout_ms);
        }
        const vis_t1 = try vmstat.nowNs();
        const vision_encode_us: f64 = @as(f64, @floatFromInt(vis_t1 - vis_t0)) / 1000.0;

        // 2. Build deterministic 3D interleaved MRoPE positions `[prefix | grid(t,h,w) | suffix]`.
        const mrope_res = try ingress.buildMropePositions(prefix_u32, grid, suffix_u32);
        const all_pos3d = ingress.mropePos3d()[0..total_tokens];

        // Build contiguous token ID sequence (placeholder ID 0 for visual positions; replaced on GPU by embedOutBinding).
        const full_toks = try allocator.alloc(u32, total_tokens);
        defer allocator.free(full_toks);
        @memcpy(full_toks[0..prefix_u32], prefix_tokens);
        @memset(full_toks[prefix_u32 .. prefix_u32 + total_vis_tokens], 0);
        @memcpy(full_toks[prefix_u32 + total_vis_tokens .. total_tokens], suffix_tokens);

        // 3. Prefill `[prefix | vision | suffix]` in causal chunks with true 3D MRoPE + GPU visual embed injection.
        self.resetKvCache();
        const chunk_cap: usize = @max(@as(usize, 1), @as(usize, self.max_tree_nodes));
        const vis_start: usize = prefix_u32;
        const vis_end: usize = prefix_u32 + total_vis_tokens;
        var offset: usize = 0;
        var first_tok: u32 = 0;

        while (offset < total_tokens) {
            const take: usize = @min(chunk_cap, total_tokens - offset);
            const cur_base: u32 = @intCast(offset);
            const is_last_chunk = (offset + take == total_tokens);

            var pos_buf: [hybrid.MAX_TREE_NODES]u32 = undefined;
            var mask_buf: [hybrid.MAX_TREE_NODES]u64 = undefined;
            var accept_nodes: [hybrid.MAX_TREE_NODES]u16 = undefined;
            for (0..take) |i| {
                pos_buf[i] = cur_base + @as(u32, @intCast(i));
                mask_buf[i] = if (i == 63) std.math.maxInt(u64) else ((@as(u64, 1) << @intCast(i + 1)) - 1);
                accept_nodes[i] = @intCast(i);
            }

            // Determine overlap of `[offset, offset + take)` with visual token range `[vis_start, vis_end)`.
            const ov_lo = @max(offset, vis_start);
            const ov_hi = @min(offset + take, vis_end);
            const vis_ov: ?VisualEmbedOverride = if (ov_lo < ov_hi) .{
                .local_start = @intCast(ov_lo - offset),
                .count = @intCast(ov_hi - ov_lo),
                .src_bind = ingress.embedOutBinding(),
                .src_row_offset = @intCast(ov_lo - vis_start),
            } else null;

            try self.retireBeforeCommitRange(cur_base, @intCast(take));
            const vr = try self.runForwardInternal(
                full_toks[offset .. offset + take],
                pos_buf[0..take],
                mask_buf[0..take],
                cur_base,
                self.effective_max_loops,
                is_last_chunk and opts.compute_logits,
                0,
                null,
                true,
                all_pos3d[offset .. offset + take],
                vis_ov,
            );
            for (0..take) |i| {
                self.token_history[cur_base + @as(u32, @intCast(i))] = full_toks[offset + i];
            }
            if (self.enable_feature_taps) {
                try self.commitFeatureTaps(accept_nodes[0..take]);
            }
            self.seq_len = cur_base + @as(u32, @intCast(take));
            try self.retireExcessCompletedChunks();
            if (is_last_chunk and opts.compute_logits) {
                first_tok = vr.argmax_tokens[take - 1];
            }
            offset += take;
        }
        self.mrope_next_pos = mrope_res.next_text_pos;

        return .{
            .first_token = first_tok,
            .prefix_tokens = prefix_u32,
            .vision_tokens = total_vis_tokens,
            .suffix_tokens = suffix_u32,
            .total_prompt_tokens = total_tokens,
            .next_mrope_pos = mrope_res.next_text_pos,
            .grid = grid,
            .frame_width = width,
            .frame_height = height,
            .frame_count = @intCast(frames.len),
            .all_frames_zero_copy = all_zc,
            .all_frames_page_aligned = all_aligned,
            .vision_encode_us = vision_encode_us,
        };
    }

    const VisualEmbedOverride = struct {
        local_start: u32,
        count: u32,
        src_bind: metal.BufferBinding,
        src_row_offset: u32,
    };

    /// `D12` `SelfSpecRecurrent` shallow drafting primitive (`0` extra draft-model RAM):
    /// evaluates `token_ids[0..m]` with `loops = 1` (`r = 0` only, without mutating
    /// committed KV/GDN state) and fills `out_candidates[0 .. m * eff_k]` via a single
    /// GPU softmax top-K dispatch.
    pub fn draftShallowBatch(
        self: *GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        top_k: u32,
        out_candidates: []moe.TopKCandidate,
    ) GraphStepError!u32 {
        if (token_ids.len == 0 or token_ids.len > self.max_tree_nodes) return error.TreeTooLarge;
        if (positions.len != token_ids.len or ancestor_masks.len != token_ids.len) return error.InvalidBatchSize;
        if (top_k == 0 or top_k > moe.MAX_TOP_K) return error.InvalidBatchSize;
        const eff_k = @min(top_k, self.config.vocab_size);
        if (out_candidates.len < token_ids.len * eff_k) return error.InvalidBatchSize;
        for (token_ids, positions) |tid, pos| {
            if (tid >= self.config.vocab_size) return error.TokenOutOfRange;
            if (pos >= self.max_seq_len) return error.SequenceFull;
        }

        _ = try self.runForwardInternal(
            token_ids,
            positions,
            ancestor_masks,
            base_pos,
            1, // shallow r = 1 / r = 0 pass (0 extra draft-model RAM)
            true,
            eff_k,
            out_candidates[0 .. token_ids.len * eff_k],
            false, // NEVER mutate committed state while drafting
            null,
            null,
        );
        return eff_k;
    }

    // -----------------------------------------------------------------------
    // Core Forward Execution with All 4 ADR-0019 Loop Isolations
    // -----------------------------------------------------------------------

    fn runForwardInternal(
        self: *GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        num_loops: u32,
        compute_logits: bool,
        topk_k: u32,
        topk_out: ?[]moe.TopKCandidate,
        commit_state_on_final: bool,
        custom_pos3d: ?[]const hybrid.Pos3D,
        vis_override: ?VisualEmbedOverride,
    ) GraphStepError!qwen3.VerifyTreeResult {
        const m: u32 = @intCast(token_ids.len);
        const h: u32 = self.config.hidden_size;
        const ls = self.config.axes.loop_schedule;
        const R: u32 = @max(@as(u32, 1), num_loops);
        const core_start: u32 = if (R > 1) ls.start_layer else 0;
        const core_end: u32 = if (R > 1)
            (if (ls.end_layer > ls.start_layer) ls.end_layer else self.config.num_hidden_layers)
        else
            self.config.num_hidden_layers;

        // Stage inputs into preallocated wired host slices (0 heap allocs).
        for (0..m) |i| {
            self.token_ids_host[i] = token_ids[i];
            self.positions_host[i] = positions[i];
            self.pos3d_host[i] = if (custom_pos3d) |p3d| p3d[i] else hybrid.Pos3D.text(positions[i]);
            self.ancestor_masks_host[i] = ancestor_masks[i];
        }

        // Reset `LoopPleScratch` at the start of every step (`r = 0` will populate).
        if (self.ple_rt) |prt| {
            prt.scratch.resetStep();
        }

        var frame_opt: ?metal.Compute.Frame = null;
        errdefer if (frame_opt) |*f| f.abort();

        // 1. Token embedding lookup `[m, H]` + optional GPU visual token override (`G14`).
        {
            const f = try self.ensureFrame(&frame_opt);
            try self.moe_pipes.encodeEmbedLookup(f, self.embed.dtype, .{
                .weight = self.embed.weight,
                .scale_bias = self.embed.scale_bias,
                .tokens = .{ .token_ids = self.token_ids_bind },
                .out = self.x_bind,
                .m = m,
                .vocab_size = self.config.vocab_size,
                .hidden_size = h,
            }, false);

            if (vis_override) |vov| {
                const row_bytes: u64 = @as(u64, h) * @sizeOf(f16);
                const src_bind: metal.BufferBinding = .{
                    .buffer = vov.src_bind.buffer,
                    .offset = vov.src_bind.offset + @as(u64, vov.src_row_offset) * row_bytes,
                };
                const dst_bind: metal.BufferBinding = .{
                    .buffer = self.x_bind.buffer,
                    .offset = self.x_bind.offset + @as(u64, vov.local_start) * row_bytes,
                };
                try self.moe_pipes.encodeCopyStrided(f, .{
                    .src = src_bind,
                    .dst = dst_bind,
                    .num_rows = vov.count,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            }

            // Initialize 4-branch GR stream (`G10`) if active.
            if (self.gr_arena) |*gr| {
                try self.hybrid_pipes.encodeGrInitStream0(f, .{
                    .x0 = self.x_bind,
                    .stream = gr.binding(self.scratch_buf),
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = self.max_tree_nodes,
                }, true);
            }
        }

        // 2. Prelude layers `[0 .. core_start)` (executed once at `r = 0`).
        for (0..core_start) |l| {
            try self.execLayer(
                &frame_opt,
                l,
                token_ids,
                ancestor_masks,
                base_pos,
                m,
                0,
                true, // prelude always commits on single pass when `commit_state_on_final`
                commit_state_on_final,
            );
        }

        // Save `x_init` after Prelude when `inject_x_init` is active (`LoopLM`).
        if (R > 1 and ls.inject_x_init) {
            const f = try self.ensureFrame(&frame_opt);
            if (self.gr_arena) |*gr| {
                try self.hybrid_pipes.encodeGrCollapse(f, .{
                    .stream = gr.binding(self.scratch_buf),
                    .out = self.x_init_bind,
                    .mode = .stream0,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = self.max_tree_nodes,
                }, true);
            } else {
                try self.moe_pipes.encodeCopyStrided(f, .{
                    .src = self.x_bind,
                    .dst = self.x_init_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            }
        }

        // 3. Core recurrent block `[core_start .. core_end) x R`.
        //    Enforces all 4 ADR-0019 loop isolations on `r < R - 1` and skips `lm_head` on `r < R - 1`.
        var r: u32 = 0;
        while (r < R) : (r += 1) {
            self.loop_stats.total_loop_steps += 1;

            // Inject saved `x_init` on loop re-entry `r >= 1` (`0.5 * h + 0.5 * x_init`).
            if (r >= 1 and ls.inject_x_init) {
                const f = try self.ensureFrame(&frame_opt);
                if (self.gr_arena) |*gr| {
                    try self.hybrid_pipes.encodeBlendF16(f, .{
                        .dst = gr.binding(self.scratch_buf), // branch 0 starts at offset 0
                        .src = self.x_init_bind,
                        .count = m * h,
                        .a = 0.5,
                        .b = 0.5,
                    }, true);
                } else {
                    try self.hybrid_pipes.encodeBlendF16(f, .{
                        .dst = self.x_bind,
                        .src = self.x_init_bind,
                        .count = m * h,
                        .a = 0.5,
                        .b = 0.5,
                    }, true);
                }
            }

            const is_final_pass = (r == R - 1);
            for (core_start..core_end) |l| {
                try self.execLayer(
                    &frame_opt,
                    l,
                    token_ids,
                    ancestor_masks,
                    base_pos,
                    m,
                    r,
                    is_final_pass,
                    commit_state_on_final and is_final_pass,
                );
            }

            // Opt-in approximate token-adaptive early exit (`--loop-early-exit`) after `r = 0`.
            if (r == 0 and R > 1 and self.enable_loop_early_exit and
                ls.early_exit_gate == .sigmoid_linear and self.early_exit_gate_host != null)
            {
                if (self.gr_arena) |*gr| {
                    const f = try self.ensureFrame(&frame_opt);
                    try self.hybrid_pipes.encodeGrCollapse(f, .{
                        .stream = gr.binding(self.scratch_buf),
                        .out = self.x_bind,
                        .mode = .stream0,
                        .m = m,
                        .hidden_dim = h,
                        .branch_token_stride = self.max_tree_nodes,
                    }, true);
                }
                try self.flushFrame(&frame_opt);
                if (self.shouldEarlyExitCpu(m)) {
                    self.loop_stats.early_exit_tokens += m;
                    // Commit core layers' `r = 0` scratch states if `commit_state_on_final`!
                    if (commit_state_on_final) {
                        try self.commitCoreLayersAfterEarlyExit(&frame_opt, core_start, core_end, base_pos, m);
                    }
                    break;
                }
            }
        }

        // 4. Coda layers `[core_end .. num_hidden_layers)` (executed once after Core).
        for (core_end..self.config.num_hidden_layers) |l| {
            try self.execLayer(
                &frame_opt,
                l,
                token_ids,
                ancestor_masks,
                base_pos,
                m,
                0,
                true,
                commit_state_on_final,
            );
        }

        // Collapse 4-branch GR stream to single-stream `x_bind` (`mean` of 4 branches).
        if (self.gr_arena) |*gr| {
            const f = try self.ensureFrame(&frame_opt);
            try self.hybrid_pipes.encodeGrCollapse(f, .{
                .stream = gr.binding(self.scratch_buf),
                .out = self.x_bind,
                .mode = .mean,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
        }

        // 5. Final norm + `lm_head` + argmax / optional top-K (NEVER run on `r < R - 1`).
        if (compute_logits) {
            const f = try self.ensureFrame(&frame_opt);
            try self.attn_pipes.encodeRmsNorm(f, .{
                .x = self.x_bind,
                .w = self.final_norm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = self.final_norm.dtype,
            }, true);
            try self.encodeLinearBatched(f, self.lm_head, self.x_norm_bind, self.logits_bind, m, true);
            try self.moe_pipes.encodeArgmaxRows(f, .{
                .logits = self.logits_bind,
                .out_token_ids = self.out_token_id_bind,
                .num_rows = m,
                .vocab_size = self.config.vocab_size,
                .logits_dtype = .f16,
            }, true);
            if (topk_k > 0) {
                try self.moe_pipes.encodeTopKLogitsRows(f, .{
                    .logits = self.logits_bind,
                    .out_candidates = self.topk_bind,
                    .num_rows = m,
                    .vocab_size = self.config.vocab_size,
                    .top_k = topk_k,
                    .logits_dtype = .f16,
                }, true);
            }
        }

        try self.flushFrame(&frame_opt);
        if (compute_logits) {
            self.last_logits_row = @as(usize, m) - 1;
            if (topk_k > 0) {
                if (topk_out) |dst| {
                    const count = @as(usize, m) * topk_k;
                    @memcpy(dst[0..count], self.topk_host[0..count]);
                }
            }
        }

        // Gather measured unique active experts across MoE layers.
        var total_unique: u64 = 0;
        if (self.config.axes.ffn_kind == .moe_swiglu) {
            for (0..self.layers.len) |l| {
                const hdr: *const moe.ScheduleHeader = @ptrCast(@alignCast(self.layer_sched_hdr_raw.ptr + l * 64));
                total_unique += hdr.num_unique_experts;
            }
        }
        const avg_unique: f64 = if (self.config.axes.ffn_kind == .moe_swiglu)
            @as(f64, @floatFromInt(total_unique)) / @as(f64, @floatFromInt(self.layers.len))
        else
            0.0;

        const core_layers: u64 = core_end - core_start;
        const base_layer_bytes: u64 = if (self.config.axes.ffn_kind == .moe_swiglu)
            self.total_layers_non_expert_bytes + total_unique * self.single_expert_bytes
        else
            self.total_layers_static_bytes;
        const loop_extra_bytes: u64 = if (R > 1)
            (R - 1) * (base_layer_bytes * core_layers / self.config.num_hidden_layers)
        else
            0;
        const head_bytes: u64 = if (compute_logits)
            @as(u64, m) * self.embed_row_alg_bytes + (self.global_head_bytes - self.embed_row_alg_bytes)
        else
            @as(u64, m) * self.embed_row_alg_bytes;

        return .{
            .num_nodes = m,
            .argmax_tokens = self.out_token_ids_host[0..m],
            .logits_f16 = self.logits_host[0 .. @as(usize, m) * self.config.vocab_size],
            .total_unique_experts = total_unique,
            .avg_unique_experts_per_layer = avg_unique,
            .actual_bytes_moved = base_layer_bytes + loop_extra_bytes + head_bytes,
        };
    }

    fn execLayer(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        token_ids: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        m: u32,
        loop_step: u32,
        is_final_pass: bool,
        commit_kv_and_gdn: bool,
    ) GraphStepError!void {
        const layer = &self.layers[l];
        const h = self.config.hidden_size;

        // --- Isolation #2 (PLE C10): single lookup at `r = 0`, 0 NVMe reads at `r >= 1` ---
        if (self.ple_rt) |prt| {
            try self.preparePleLayer(prt, l, token_ids, ancestor_masks, base_pos, m, loop_step);
        }

        // --- Pre-Attention 4-Branch GR Read Gate (G10) ---
        if (self.gr_arena) |*gr| {
            const gr_w = layer.gr.?;
            const f = try self.ensureFrame(frame_opt);
            try self.hybrid_pipes.encodeGrComputeGates(f, .{
                .stream = gr.binding(self.scratch_buf),
                .w_read = gr_w.read_gate,
                .w_write = gr_w.write_gate,
                .w_decay = gr_w.decay_gate,
                .g_read_out = self.gr_g_read_bind,
                .g_write_out = self.gr_g_write_bind,
                .decay_out = self.gr_decay_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
                .weight_dtype = gr_w.dtype,
            }, true);
            try self.hybrid_pipes.encodeGrReadGate(f, .{
                .stream = gr.binding(self.scratch_buf),
                .g_read = self.gr_g_read_bind,
                .x_in = self.x_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
        }

        // Accumulate PLE contribution into `x_bind` (inside the same Metal CB, 0 mid-pass sync).
        if (self.ple_rt != null) {
            const f = try self.ensureFrame(frame_opt);
            try self.moe_pipes.encodeResidualAdd(f, .{
                .x = self.x_bind,
                .delta = self.pleLayerBinding(l),
                .count = m * h,
            }, true);
        }

        // Input RMSNorm -> `x_norm_bind`.
        {
            const f = try self.ensureFrame(frame_opt);
            try self.attn_pipes.encodeRmsNorm(f, .{
                .x = self.x_bind,
                .w = layer.input_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);
        }

        // --- Attention / Sequence Mixer Dispatch ---
        switch (layer.attn) {
            .gdn => |g| {
                try self.execGdnAttention(frame_opt, l, g, m, loop_step, is_final_pass, commit_kv_and_gdn);
            },
            .qsa => |*qsa| {
                try self.execQsaAttention(frame_opt, l, qsa, base_pos, m, loop_step, commit_kv_and_gdn);
            },
            .gqa => |gqa| {
                try self.execGqaAttention(frame_opt, l, gqa, base_pos, m, loop_step, commit_kv_and_gdn);
            },
        }

        // Residual update after attention (`G10` 4-branch write-gate or standard residual add).
        if (self.gr_arena) |*gr| {
            const f = try self.ensureFrame(frame_opt);
            try self.hybrid_pipes.encodeGrWriteGate(f, .{
                .stream = gr.binding(self.scratch_buf),
                .y_delta = self.proj_out_bind,
                .decay = self.gr_decay_bind,
                .g_write = self.gr_g_write_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
            // Read updated stream back into `x_bind` for FFN sub-block.
            try self.hybrid_pipes.encodeGrReadGate(f, .{
                .stream = gr.binding(self.scratch_buf),
                .g_read = self.gr_g_read_bind,
                .x_in = self.x_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
        } else {
            const f = try self.ensureFrame(frame_opt);
            try self.moe_pipes.encodeResidualAdd(f, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = m * h,
            }, true);
        }

        // Post-attention RMSNorm -> `x_norm_bind`.
        {
            const f = try self.ensureFrame(frame_opt);
            try self.attn_pipes.encodeRmsNorm(f, .{
                .x = self.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = self.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = self.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);
        }

        // --- FFN / MoE Dispatch ---
        switch (self.config.axes.ffn_kind) {
            .dense_swiglu => {
                try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .gate_proj, layer.dense_gate, self.x_norm_bind, self.x_norm_host, self.dense_gate_bind, self.dense_gate_host, m, true);
                try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .up_proj, layer.dense_up, self.x_norm_bind, self.x_norm_host, self.dense_up_bind, self.dense_up_host, m, false);
                {
                    const f = try self.ensureFrame(frame_opt);
                    try self.moe_pipes.encodeSiluMul(f, .{
                        .gate = self.dense_gate_bind,
                        .up = self.dense_up_bind,
                        .out = self.dense_act_bind,
                        .count = m * self.config.intermediate_size,
                    }, true);
                }
                try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .down_proj, layer.dense_down, self.dense_act_bind, self.dense_act_host, self.proj_out_bind, self.proj_out_host, m, true);
                try self.applyFfnResidual(frame_opt, m, h);
            },
            .moe_swiglu => {
                try self.execMoeFfn(frame_opt, l, layer, m, loop_step);
            },
        }

        // Capture EAGLE-3 aux taps on final loop pass if enabled.
        if (self.enable_feature_taps and is_final_pass) {
            for (self.aux_layer_ids, 0..) |tap_l, slot| {
                if (tap_l == @as(u32, @intCast(l))) {
                    const f = try self.ensureFrame(frame_opt);
                    if (self.gr_arena) |*gr| {
                        try self.hybrid_pipes.encodeGrCollapse(f, .{
                            .stream = gr.binding(self.scratch_buf),
                            .out = self.x_bind,
                            .mode = .mean,
                            .m = m,
                            .hidden_dim = h,
                            .branch_token_stride = self.max_tree_nodes,
                        }, true);
                    }
                    try self.moe_pipes.encodeCopyStrided(f, .{
                        .src = self.x_bind,
                        .dst = .{ .buffer = self.scratch_buf, .offset = self.aux_tree_base_off },
                        .num_rows = m,
                        .copy_cols = h,
                        .src_stride = h,
                        .dst_stride = 3 * h,
                        .dst_col_off = @as(u32, @intCast(slot)) * h,
                    }, true);
                }
            }
        }
    }

    fn applyFfnResidual(self: *GraphModel, frame_opt: *?metal.Compute.Frame, m: u32, h: u32) GraphStepError!void {
        const f = try self.ensureFrame(frame_opt);
        if (self.gr_arena) |*gr| {
            try self.hybrid_pipes.encodeGrWriteGate(f, .{
                .stream = gr.binding(self.scratch_buf),
                .y_delta = self.proj_out_bind,
                .decay = self.gr_decay_bind,
                .g_write = self.gr_g_write_bind,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = self.max_tree_nodes,
            }, true);
        } else {
            try self.moe_pipes.encodeResidualAdd(f, .{
                .x = self.x_bind,
                .delta = self.proj_out_bind,
                .count = m * h,
            }, true);
        }
    }

    fn execGdnAttention(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        g: GdnLayerWeights,
        m: u32,
        loop_step: u32,
        is_final_pass: bool,
        commit_state: bool,
    ) GraphStepError!void {
        const ga = &(self.gdn_arena.?);
        const qk_dim = self.config.gdnQkDim();
        const v_dim = self.config.gdnVDim();

        // Shared r=0 cross-token GDN state transition inputs (`LoopLM`, ADR-0019):
        // - At `loop_step == 0`: compute `q_raw^(0), k_raw^(0), v_raw^(0), g^(0), beta^(0)`,
        //   run `conv1d + SiLU` with `is_tree = true, write_state = false` into
        //   `(q_conv_bind, tree_k_off, tree_v_off)` while preserving raw `r = 0` inputs in
        //   `tree_{q,k,v}_raw_off` (for `r >= 1` causal Q conv taps & final `conv_state` commit).
        // - At `loop_step >= 1`: NEVER overwrite `tree_{q,k,v}_raw_off` or `tree_{k,v,g,beta}_off`.
        //   Compute only `q_raw^(r) -> self.q_bind` (with optional Step-LoRA) and `z^(r) -> self.gdn_z_bind`,
        //   and if `conv1d_w != null` run `ami_gdn_conv1d_q_silu` using `q_cur = self.q_bind` at `tap == hist`
        //   and `r = 0` history `tree_q_raw_off` / `conv_state_off` at `tap < hist`, writing ONLY `q_conv_bind`.
        const q_raw_r0_bind: metal.BufferBinding = if (g.conv1d_w != null)
            .{ .buffer = self.scratch_buf, .offset = g.tree_q_raw_off }
        else
            self.q_bind;
        const q_proj_out_bind: metal.BufferBinding = if (loop_step == 0) q_raw_r0_bind else self.q_bind;
        const q_proj_out_host: []f16 = if (loop_step == 0 and g.conv1d_w != null) g.tree_q_raw_host else self.q_host;
        const k_raw_r0_bind: metal.BufferBinding = if (g.conv1d_w != null)
            .{ .buffer = self.scratch_buf, .offset = g.tree_k_raw_off }
        else
            .{ .buffer = self.scratch_buf, .offset = g.tree_k_off };
        const v_raw_r0_bind: metal.BufferBinding = if (g.conv1d_w != null)
            .{ .buffer = self.scratch_buf, .offset = g.tree_v_raw_off }
        else
            .{ .buffer = self.scratch_buf, .offset = g.tree_v_off };

        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .q_proj, g.q_proj, self.x_norm_bind, self.x_norm_host, q_proj_out_bind, q_proj_out_host, m, true);
        {
            const f = try self.ensureFrame(frame_opt);
            if (loop_step == 0) {
                try self.encodeLinearBatched(f, g.k_proj, self.x_norm_bind, k_raw_r0_bind, m, false);
                try self.encodeLinearBatched(f, g.v_proj, self.x_norm_bind, v_raw_r0_bind, m, false);
                try self.hybrid_pipes.encodeGdnComputeGates(f, .{
                    .x = self.x_norm_bind,
                    .w_a = g.a_proj,
                    .w_b = g.b_proj,
                    .a_log = g.a_log,
                    .dt_bias = g.dt_bias,
                    .g_out = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                    .beta_out = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                    .m = m,
                    .hidden_dim = self.config.hidden_size,
                    .num_v_heads = self.config.linear_num_value_heads,
                    .weight_dtype = g.ab_dtype,
                }, false);
            }
            try self.encodeLinearBatched(f, g.g_proj, self.x_norm_bind, self.gdn_z_bind, m, false);

            // Optional depthwise causal Conv1D + SiLU with loop-isolated `r = 0` history.
            const q_eff_bind: metal.BufferBinding = if (g.conv1d_w != null) self.q_conv_bind else q_proj_out_bind;
            const k_eff_bind: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = g.tree_k_off };
            const v_eff_bind: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = g.tree_v_off };
            if (g.conv1d_w) |cw| {
                if (loop_step == 0) {
                    try self.hybrid_pipes.encodeGdnConv1dSilu(f, .{
                        .q_in = q_raw_r0_bind,
                        .k_in = k_raw_r0_bind,
                        .v_in = v_raw_r0_bind,
                        .conv_w = cw,
                        .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                        .conv_state_out = .{ .buffer = self.scratch_buf, .offset = g.conv_state_tmp_off },
                        .ancestor_masks = self.ancestor_masks_bind,
                        .q_out = q_eff_bind,
                        .k_out = k_eff_bind,
                        .v_out = v_eff_bind,
                        .m = m,
                        .gdn_qk = qk_dim,
                        .gdn_v = v_dim,
                        .conv_k = g.conv_k,
                        .is_tree = true,
                        .write_state = false,
                    }, true);
                } else {
                    try self.hybrid_pipes.encodeGdnConv1dQSilu(f, .{
                        .q_cur = self.q_bind,
                        .q_hist_r0 = q_raw_r0_bind,
                        .conv_w = cw,
                        .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                        .ancestor_masks = self.ancestor_masks_bind,
                        .q_out = q_eff_bind,
                        .m = m,
                        .gdn_qk = qk_dim,
                        .gdn_v = v_dim,
                        .conv_k = g.conv_k,
                    }, true);
                }
                // Advance committed Conv1D history once on `commit_state` using `r = 0` raw QKV
                // (unified for both `m = 1` and `m > 1` so chunk size never affects state).
                if (commit_state and g.conv_k > 1) {
                    var chain: [hybrid.MAX_TREE_NODES]u16 = undefined;
                    for (0..m) |i| chain[i] = @intCast(i);
                    try self.hybrid_pipes.encodeGdnConv1dCommit(f, .{
                        .q_raw = q_raw_r0_bind,
                        .k_raw = k_raw_r0_bind,
                        .v_raw = v_raw_r0_bind,
                        .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                        .conv_state_out = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                        .accepted_nodes = chain[0..m],
                        .gdn_qk = qk_dim,
                        .gdn_v = v_dim,
                        .conv_k = g.conv_k,
                    }, true);
                }
            }

            // --- Isolation #1 (GDN R6/G8): NEVER mutate `S_t` on `r < R - 1` or tree verify ---
            const committed_s = try ga.committedBinding(self.scratch_buf, g.gdn_idx);
            const scratch_s = try ga.scratchBinding(self.scratch_buf, g.gdn_idx);
            if (m == 1) {
                try self.hybrid_pipes.encodeGdnDecode(f, .{
                    .q = q_eff_bind,
                    .k = k_eff_bind,
                    .v = v_eff_bind,
                    .g = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                    .beta = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                    .z = self.gdn_z_bind,
                    .norm_w = if (g.out_norm) |n| n.binding else null,
                    .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                    .state_in = committed_s,
                    .state_out = scratch_s,
                    .out = self.attn_out_bind,
                    .num_k_heads = self.config.linear_num_key_heads,
                    .num_v_heads = self.config.linear_num_value_heads,
                    .head_k_dim = self.config.linear_key_head_dim,
                    .head_v_dim = self.config.linear_value_head_dim,
                    .rms_eps = self.config.rms_norm_eps,
                }, true);
                if (!is_final_pass or !commit_state) {
                    self.loop_stats.gdn_scratch_evals += 1;
                } else {
                    try self.hybrid_pipes.encodeCommitScratchState(f, ga, self.scratch_buf, g.gdn_idx, true);
                    self.loop_stats.gdn_final_commits += 1;
                }
            } else if (commit_state) {
                try self.hybrid_pipes.encodeGdnPrefill(f, .{
                    .q = q_eff_bind,
                    .k = k_eff_bind,
                    .v = v_eff_bind,
                    .g = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                    .beta = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                    .z = self.gdn_z_bind,
                    .norm_w = if (g.out_norm) |n| n.binding else null,
                    .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                    .state_in = committed_s,
                    .state_out = scratch_s,
                    .out = self.attn_out_bind,
                    .seq_len = m,
                    .num_k_heads = self.config.linear_num_key_heads,
                    .num_v_heads = self.config.linear_num_value_heads,
                    .head_k_dim = self.config.linear_key_head_dim,
                    .head_v_dim = self.config.linear_value_head_dim,
                    .rms_eps = self.config.rms_norm_eps,
                }, true);
                try self.hybrid_pipes.encodeCommitScratchState(f, ga, self.scratch_buf, g.gdn_idx, true);
                self.loop_stats.gdn_final_commits += 1;
            } else {
                // Tree verify or intermediate loop pass (`r < R - 1`): committed `S_t` stays untouched!
                try self.hybrid_pipes.encodeGdnTreeVerify(f, .{
                    .q = q_eff_bind,
                    .k = k_eff_bind,
                    .v = v_eff_bind,
                    .g = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                    .beta = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                    .z = self.gdn_z_bind,
                    .norm_w = if (g.out_norm) |n| n.binding else null,
                    .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                    .ancestor_masks = self.ancestor_masks_bind,
                    .state_in = committed_s,
                    .state_out = scratch_s,
                    .scratch_leaf_node = null,
                    .out = self.attn_out_bind,
                    .num_nodes = m,
                    .num_k_heads = self.config.linear_num_key_heads,
                    .num_v_heads = self.config.linear_num_value_heads,
                    .head_k_dim = self.config.linear_key_head_dim,
                    .head_v_dim = self.config.linear_value_head_dim,
                    .rms_eps = self.config.rms_norm_eps,
                }, true);
                self.loop_stats.gdn_scratch_evals += 1;
            }
        }

        // Output projection `o_proj` (with optional Step-LoRA).
        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .o_proj, g.o_proj, self.attn_out_bind, self.attn_out_host, self.proj_out_bind, self.proj_out_host, m, true);
    }

    fn execQsaAttention(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        qsa: *const QsaLayerWeights,
        base_pos: u32,
        m: u32,
        loop_step: u32,
        commit_kv: bool,
    ) GraphStepError!void {
        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .q_proj, qsa.q_proj, self.x_norm_bind, self.x_norm_host, self.q_bind, self.q_host, m, true);
        {
            const f = try self.ensureFrame(frame_opt);
            const tree_k: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = qsa.tree_k_off };
            const tree_v: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = qsa.tree_v_off };

            // Shared r=0 KV across recurrent passes (`LoopLM` KV sharing, ADR-0019):
            // compute `K, V` once at `r = 0` into `tree_k/tree_v` and reuse them on `r >= 1`
            // so batched `verifyTree(M > 1)` matches serial `stepDecode(M = 1)` bit-for-bit
            // and QSA `reuse_block_indices` sees identical KV keys across loops.
            if (loop_step == 0) {
                try self.encodeLinearBatched(f, qsa.k_proj, self.x_norm_bind, self.k_bind, m, false);
                try self.encodeLinearBatched(f, qsa.v_proj, self.x_norm_bind, self.v_bind, m, false);
            }

            // QK-Norm + YaRN + Interleaved 3D MRoPE: at `r = 0` write into `tree_k/tree_v`;
            // at `r >= 1` rotate `Q` in-place and route K/V output to scratch `k_bind/v_bind`
            // so `tree_k/tree_v` preserve `r = 0` K/V untouched.
            const mrope_rot: u32 = (self.config.mrope_section.section[0] + self.config.mrope_section.section[1] + self.config.mrope_section.section[2]) * 2;
            const eff_rot_dim: u32 = if (mrope_rot > 0 and mrope_rot <= self.config.head_dim) mrope_rot else self.config.head_dim;
            try self.hybrid_pipes.encodeQsaTreeRope(f, .{
                .q = self.q_bind,
                .k = self.k_bind,
                .v = self.v_bind,
                .q_norm = qsa.q_norm,
                .k_norm = qsa.k_norm,
                .positions_3d = self.pos3d_bind,
                .k_tree_out = if (loop_step == 0) tree_k else self.k_bind,
                .v_tree_out = if (loop_step == 0) tree_v else self.v_bind,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .rotary_dim = eff_rot_dim,
                .mrope = self.config.mrope_section,
                .rms_eps = self.config.rms_norm_eps,
                .rope_theta = self.config.rope_theta,
                .yarn = self.config.yarn,
                .norm_dtype = qsa.qk_norm_dtype,
            }, true);

            // --- Isolation #3 (QSA G9): compute block indices at `r = 0`, reuse on `r >= 1` ---
            const reuse_idx = (loop_step >= 1);
            if (base_pos > 0) {
                if (reuse_idx) {
                    self.loop_stats.qsa_index_reuses += 1;
                } else {
                    self.loop_stats.qsa_index_computes += 1;
                }
            }
            const use_qsa_tiered = (qsa.supports_retire_q8 and self.retired_kv_pool != null);
            const qsa_kv_tiles_bind: ?metal.BufferBinding = if (base_pos == 0)
                null
            else if (use_qsa_tiered) blk: {
                const ret_toks: u64 = @as(u64, self.num_retired_chunks) * @as(u64, qsa.kv_ring.chunk_tokens);
                if (@as(u64, base_pos) <= ret_toks) break :blk null;
                break :blk metal.BufferBinding{
                    .buffer = self.scratch_buf,
                    .offset = qsa.full_tile_table_off + @as(u64, self.num_retired_chunks) * @sizeOf(attention.KvTilePair),
                };
            } else qsa.kv_ring.tileTableBinding(self.scratch_buf);
            const qsa_ret_pool_bind: ?metal.BufferBinding = if (use_qsa_tiered and self.num_retired_chunks > 0)
                self.retired_kv_pool.?.binding(self.scratch_buf)
            else
                null;
            try self.hybrid_pipes.encodeQsaAttention(f, .{
                .q = self.q_bind,
                .kv_tiles = qsa_kv_tiles_bind,
                .block_keys = .{ .buffer = self.scratch_buf, .offset = qsa.block_keys_off },
                .index_buf = &qsa.index_buf,
                .index_metal_buf = self.scratch_buf,
                .k_tree = tree_k,
                .v_tree = tree_v,
                .ancestor_masks = self.ancestor_masks_bind,
                .out = self.attn_out_bind,
                .retired_pool = qsa_ret_pool_bind,
                .retired_chunk_ids = if (qsa_ret_pool_bind != null) qsa.retired_ids_binding else null,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .kv_len = base_pos,
                .window_size = qsa.kv_ring.window_size,
                .ring_capacity = qsa.kv_ring.ring_capacity,
                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                .block_tokens = self.config.qsa_block_size,
                .top_k_blocks = self.config.qsa_top_k_blocks,
                .num_retired_chunks = if (qsa_ret_pool_bind != null) self.num_retired_chunks else 0,
                .retired_plane_stride_bytes = if (qsa_ret_pool_bind != null) self.retired_kv_pool.?.plane_stride_bytes else 0,
                .retired_chunk_stride_bytes = if (qsa_ret_pool_bind != null) self.retired_kv_pool.?.chunk_stride_bytes else 0,
                .reuse_block_indices = reuse_idx,
            }, true);

            if (commit_kv) {
                var chain: [hybrid.MAX_TREE_NODES]u16 = undefined;
                for (0..m) |i| chain[i] = @intCast(i);
                try self.encodeQsaCommitAccepted(f, qsa, base_pos, chain[0..m], true);
            }
        }

        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .o_proj, qsa.o_proj, self.attn_out_bind, self.attn_out_host, self.proj_out_bind, self.proj_out_host, m, true);
    }

    fn execGqaAttention(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        gqa: GqaLayerWeights,
        base_pos: u32,
        m: u32,
        loop_step: u32,
        commit_kv: bool,
    ) GraphStepError!void {
        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .q_proj, gqa.q_proj, self.x_norm_bind, self.x_norm_host, self.q_bind, self.q_host, m, true);
        {
            const f = try self.ensureFrame(frame_opt);
            const tree_k: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = gqa.tree_k_off };
            const tree_v: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = gqa.tree_v_off };

            // Shared r=0 KV across recurrent passes (`LoopLM` KV sharing, ADR-0019):
            // compute `K, V` once at `r = 0` into `tree_k/tree_v` and reuse on `r >= 1`
            // so parallel `prefill` / `verifyTree(M > 1)` matches serial `stepDecode(M = 1)` bit-for-bit.
            if (loop_step == 0) {
                try self.encodeLinearBatched(f, gqa.k_proj, self.x_norm_bind, self.k_bind, m, false);
                try self.encodeLinearBatched(f, gqa.v_proj, self.x_norm_bind, self.v_bind, m, false);
            }

            try self.attn_pipes.encodeTreeQkNormRope(f, .{
                .q = self.q_bind,
                .k = self.k_bind,
                .v = self.v_bind,
                .q_norm = gqa.q_norm,
                .k_norm = gqa.k_norm,
                .positions = self.positions_bind,
                .k_tree_out = if (loop_step == 0) tree_k else self.k_bind,
                .v_tree_out = if (loop_step == 0) tree_v else self.v_bind,
                .num_nodes = m,
                .num_q_heads = self.config.num_attention_heads,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .rms_eps = self.config.rms_norm_eps,
                .rope_theta = self.config.rope_theta,
                .norm_dtype = gqa.qk_norm_dtype,
            }, true);

            if (self.num_retired_chunks > 0 and base_pos > 0 and self.retired_kv_pool != null) {
                const rpool = self.retired_kv_pool.?;
                const ret_toks: u64 = @as(u64, self.num_retired_chunks) * @as(u64, self.chunk_tokens);
                try self.attn_pipes.encodeGqaTieredTreeVerifyAttn(f, .{
                    .q = self.q_bind,
                    .kv_tiles = if (@as(u64, base_pos) > ret_toks) gqa.kv_tiles_binding else null,
                    .retired_pool = rpool.binding(self.scratch_buf),
                    .retired_chunk_ids = gqa.retired_ids_binding,
                    .base_pos = base_pos,
                    .k_tree = tree_k,
                    .v_tree = tree_v,
                    .ancestor_masks = self.ancestor_masks_bind,
                    .out = self.attn_out_bind,
                    .scratch = self.gqa_scratch_bind,
                    .max_splits = attention.MAX_SPLITS,
                    .num_nodes = m,
                    .num_q_heads = self.config.num_attention_heads,
                    .num_kv_heads = self.config.num_key_value_heads,
                    .head_dim = self.config.head_dim,
                    .chunk_tokens = self.chunk_tokens,
                    .num_retired_chunks = self.num_retired_chunks,
                    .retired_plane_stride_bytes = rpool.plane_stride_bytes,
                    .retired_chunk_stride_bytes = rpool.chunk_stride_bytes,
                }, true);
            } else {
                try self.attn_pipes.encodeGqaTreeVerifyAttn(f, .{
                    .q = self.q_bind,
                    .kv_tiles = if (base_pos > 0) gqa.kv_tiles_binding else null,
                    .base_pos = base_pos,
                    .k_tree = tree_k,
                    .v_tree = tree_v,
                    .ancestor_masks = self.ancestor_masks_bind,
                    .out = self.attn_out_bind,
                    .scratch = self.gqa_scratch_bind,
                    .max_splits = attention.MAX_SPLITS,
                    .num_nodes = m,
                    .num_q_heads = self.config.num_attention_heads,
                    .num_kv_heads = self.config.num_key_value_heads,
                    .head_dim = self.config.head_dim,
                    .chunk_tokens = self.chunk_tokens,
                }, true);
            }

            if (commit_kv) {
                var chain: [hybrid.MAX_TREE_NODES]u16 = undefined;
                for (0..m) |i| chain[i] = @intCast(i);
                try self.encodeGqaCommitAccepted(f, gqa, base_pos, chain[0..m], true);
            }
        }

        try self.execLinearWithOptionalStepLora(frame_opt, l, loop_step, .o_proj, gqa.o_proj, self.attn_out_bind, self.attn_out_host, self.proj_out_bind, self.proj_out_host, m, true);
    }

    fn execMoeFfn(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        layer: *GraphLayerWeights,
        m: u32,
        loop_step: u32,
    ) GraphStepError!void {
        const h = self.config.hidden_size;
        var layer_ws = self.moe_ws.?;
        layer_ws.schedule_header_off = self.layer_sched_hdr_off + @as(u64, @intCast(l)) * 64;
        layer_ws.schedule_header = @ptrCast(@alignCast(self.layer_sched_hdr_raw.ptr + l * 64));

        const router_src: moe.RouterSource = if (layer.router_quant_dtype) |rqd| .{
            .quant_weights = .{
                .qweight = layer.router_binding,
                .scale_bias = layer.router_scale_bias.?,
                .bits = switch (rqd) {
                    .q2_affine => .q2,
                    .q3_affine => .q3,
                    .q4_affine => .q4,
                    else => return error.InvalidConfig,
                },
            },
        } else .{
            .weights = .{
                .binding = layer.router_binding,
                .dtype = layer.router_dtype,
            },
        };

        // Residual target: for 4-branch GR, write FFN delta into `proj_out_bind` (zero residual)
        // and feed into `ami_gr_write_gate`; for single-stream, fuse `x + ffn` directly into `x_bind`.
        const moe_residual: ?metal.BufferBinding = if (self.gr_arena != null) null else self.x_bind;
        const moe_out: metal.BufferBinding = if (self.gr_arena != null) self.proj_out_bind else self.x_bind;

        // --- Isolation #4 (MoE G7, C5): cold-expert resolution on `r = 0`, pinned-only routing on `r >= 1` ---
        if (layer.has_cold_experts and self.cold_rt != null) {
            const crt = self.cold_rt.?;
            const logits_bind: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = layer_ws.router_logits_off };
            // 1. Run router projection on GPU and read `router_logits`.
            {
                const f = try self.ensureFrame(frame_opt);
                switch (router_src) {
                    .weights => |w| try self.moe_pipes.encodeRouterProj(f, w.dtype, .{
                        .x = self.x_norm_bind,
                        .router_weights = w.binding,
                        .logits = logits_bind,
                        .m = m,
                        .num_experts = self.config.num_experts,
                        .hidden_size = h,
                    }, true),
                    .quant_weights => |qw| try self.moe_pipes.encodeRouterProjQuant(f, qw.bits, .{
                        .x = self.x_norm_bind,
                        .qweight = qw.qweight,
                        .scale_bias = qw.scale_bias,
                        .logits = logits_bind,
                        .m = m,
                        .num_experts = self.config.num_experts,
                        .hidden_size = h,
                    }, true),
                    .logits => {},
                }
            }
            try self.flushFrame(frame_opt);

            const n_exp: usize = self.config.num_experts;
            const top_k: usize = self.config.num_experts_per_tok;
            const max_m: usize = self.max_tree_nodes;
            const before_misses = crt.expert_cache.telemetry().sync_misses;
            if (loop_step >= 1) {
                // ADR-0019 Isolation #4: restrict `r >= 1` routing for `(l, tok)` strictly to
                // wired `.hot` experts plus `(l, tok)`'s own `r = 0` selected cold experts (when
                // still `.cached_ready`), preventing unaccepted speculative siblings (or prior
                // tokens) from polluting `r >= 1` routing while guaranteeing 0 cold NVMe stalls.
                for (0..m) |tok| {
                    const row = layer_ws.router_logits[tok * n_exp ..][0..n_exp];
                    const r0_ids = crt.r0_selected[(l * max_m + tok) * top_k ..][0..top_k];
                    var allowed_count: usize = 0;
                    for (0..n_exp) |eid| {
                        const allow = switch (crt.expert_cache.residencyState(@intCast(l), @intCast(eid))) {
                            .hot => true,
                            .cached_ready => blk: {
                                var own = false;
                                for (r0_ids) |rid| {
                                    if (rid == eid) {
                                        own = true;
                                        break;
                                    }
                                }
                                break :blk own;
                            },
                            else => false,
                        };
                        if (allow) {
                            allowed_count += 1;
                        } else {
                            row[eid] = -std.math.inf(f32);
                        }
                    }
                    // Defensive guard only reachable if `num_hot < top_k` and a tiny slot pool
                    // evicted `tok`'s own `r = 0` cold experts; keeps 0 cold stalls & finite softmax.
                    if (allowed_count < top_k) {
                        for (0..n_exp) |eid| {
                            if (row[eid] == -std.math.inf(f32) and crt.expert_cache.isResident(@intCast(l), @intCast(eid))) {
                                row[eid] = -1.0e30;
                                allowed_count += 1;
                                if (allowed_count >= top_k) break;
                            }
                        }
                    }
                }
                self.loop_stats.moe_r_ge1_pinned_routes += 1;
            }

            // 2. Run top-k CSR schedule on GPU (`ami_moe_route_topk`), then pin active experts.
            {
                const f = try self.ensureFrame(frame_opt);
                try self.moe_pipes.encodeRouteTopK(f, .f32, .{
                    .logits = logits_bind,
                    .workspace = &layer_ws,
                    .workspace_buf = self.scratch_buf,
                    .m = m,
                    .num_experts = self.config.num_experts,
                    .top_k = self.config.num_experts_per_tok,
                    .norm_topk_prob = self.config.norm_topk_prob,
                }, false);
            }
            try self.flushFrame(frame_opt);

            if (loop_step == 0) {
                for (0..m) |tok| {
                    const dst = crt.r0_selected[(l * max_m + tok) * top_k ..][0..top_k];
                    const src = layer_ws.scratch_topk_experts[tok * top_k ..][0..top_k];
                    for (0..top_k) |k| {
                        dst[k] = @intCast(src[k]);
                    }
                }
            }

            const num_unique: usize = layer_ws.schedule_header.num_unique_experts;
            for (0..num_unique) |u| {
                const eid: u16 = @intCast(layer_ws.unique_experts[u]);
                const pin = try crt.expert_cache.acquirePinned(@intCast(l), eid);
                layer.expert_table_host[eid] = pin.weights;
            }
            errdefer crt.expert_cache.releasePins();

            if (loop_step >= 1) {
                const after_misses = crt.expert_cache.telemetry().sync_misses;
                self.loop_stats.moe_r_ge1_cold_stalls += (after_misses - before_misses);
            }

            // 3. Execute deduplicated SwiGLU FFN + optional `shared_expert` on GPU using pinned slots!
            const moe_args: moe.MoELayerArgs = .{
                .x = self.x_norm_bind,
                .router = .{ .logits = .{ .binding = logits_bind, .dtype = .f32 } },
                .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                .expert_table = layer.expert_table,
                .workspace = &layer_ws,
                .workspace_buf = self.scratch_buf,
                .residual = moe_residual,
                .out = moe_out,
                .m = m,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .hidden_size = h,
                .intermediate_size = self.config.moe_intermediate_size,
                .norm_topk_prob = self.config.norm_topk_prob,
                .shared_expert = layer.shared_expert,
                .shared_weights_base = if (layer.shared_expert_buf) |sb| .{ .buffer = sb, .offset = 0 } else null,
                .shared_expert_dtype = layer.shared_expert_dtype,
                .shared_intermediate_size = self.config.shared_expert_intermediate_size,
                .shared_gate_weight = layer.shared_gate_weight,
                .shared_gate_dtype = layer.shared_gate_dtype,
            };
            {
                const f = try self.ensureFrame(frame_opt);
                try self.encodeMoEFfnOnlyDispatch(f, layer, moe_args, false);
            }
            try self.flushFrame(frame_opt);
            crt.expert_cache.releasePins();
        } else {
            // Fast all-wired MoE path: single command buffer, 0 CPU-GPU syncs!
            if (loop_step >= 1) {
                self.loop_stats.moe_r_ge1_pinned_routes += 1;
            }
            const f = try self.ensureFrame(frame_opt);
            const moe_args: moe.MoELayerArgs = .{
                .x = self.x_norm_bind,
                .router = router_src,
                .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                .expert_table = layer.expert_table,
                .workspace = &layer_ws,
                .workspace_buf = self.scratch_buf,
                .residual = moe_residual,
                .out = moe_out,
                .m = m,
                .num_experts = self.config.num_experts,
                .top_k = self.config.num_experts_per_tok,
                .hidden_size = h,
                .intermediate_size = self.config.moe_intermediate_size,
                .norm_topk_prob = self.config.norm_topk_prob,
                .shared_expert = layer.shared_expert,
                .shared_weights_base = if (layer.shared_expert_buf) |sb| .{ .buffer = sb, .offset = 0 } else null,
                .shared_expert_dtype = layer.shared_expert_dtype,
                .shared_intermediate_size = self.config.shared_expert_intermediate_size,
                .shared_gate_weight = layer.shared_gate_weight,
                .shared_gate_dtype = layer.shared_gate_dtype,
            };
            try self.encodeMoELayerDispatch(f, layer, moe_args, true);
        }

        if (self.gr_arena != null) {
            try self.applyFfnResidual(frame_opt, m, h);
        }
    }

    fn encodeMoELayerDispatch(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        layer: *const GraphLayerWeights,
        args: moe.MoELayerArgs,
        barrier_before: bool,
    ) moe.EncodeError!void {
        if (layer.is_hetero_moe) {
            return self.moe_pipes.encodeMoELayerHetero(frame, .{
                .base = args,
                .expert_schemes = layer.expert_dtypes_binding.?,
                .scheme_counts = layer.expert_scheme_counts,
            }, barrier_before);
        }
        return switch (layer.expert_weight_dtype) {
            .q2_affine => self.moe_pipes.encodeMoELayer(frame, .q2, args, barrier_before),
            .q3_affine => self.moe_pipes.encodeMoELayer(frame, .q3, args, barrier_before),
            .q4_affine => self.moe_pipes.encodeMoELayer(frame, .q4, args, barrier_before),
            .f16 => self.moe_pipes.encodeMoELayerFloat(frame, .f16, args, barrier_before),
            .bf16 => self.moe_pipes.encodeMoELayerFloat(frame, .bf16, args, barrier_before),
        };
    }

    fn encodeMoEFfnOnlyDispatch(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        layer: *const GraphLayerWeights,
        args: moe.MoELayerArgs,
        barrier_before: bool,
    ) moe.EncodeError!void {
        // Router logits & schedule header are already populated; dispatch directly.
        return self.encodeMoELayerDispatch(frame, layer, args, barrier_before);
    }

    /// Evaluates a base linear projection on GPU plus optional L2-resident SME2
    /// `Step-LoRA` adapter (`W16`, ADR-0019).
    /// When `alpha == 0.0` (`r = 0`), executes strictly on GPU with zero CPU syncs
    /// and zero writes (100% bit-exact identity).
    fn execLinearWithOptionalStepLora(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        loop_step: u32,
        proj: AdapterProjKind,
        lin: qwen3.LinearBinding,
        x_bind: metal.BufferBinding,
        x_host: []const f16,
        y_bind: metal.BufferBinding,
        y_host: []f16,
        m: u32,
        barrier_before: bool,
    ) GraphStepError!void {
        const adapter_opt: ?sme.SmeStepLoraAdapter = if (self.adapter_rt) |art|
            art.get(loop_step, @intCast(l), proj)
        else
            null;

        if (adapter_opt) |adapter| {
            if (adapter.alpha == 0.0) {
                // Mandatory ADR-0019 r = 0 bit-exact identity fast path!
                self.loop_stats.steplora_zero_alpha_skips += 1;
                const f = try self.ensureFrame(frame_opt);
                try self.encodeLinearBatched(f, lin, x_bind, y_bind, m, barrier_before);
                return;
            }

            // Active `r >= 1` Step-LoRA: flush input readiness, launch GPU base linear
            // concurrently with L2-resident P-core SME2 `SmeStepLoraAdapter.evalU16`!
            const art = self.adapter_rt.?;
            try self.flushFrame(frame_opt);

            var base_frame = try self.compute.beginFrame(self.timeout_ms);
            errdefer base_frame.abort();
            try self.encodeLinearBatched(&base_frame, lin, x_bind, y_bind, m, false);
            const sig = try base_frame.commit();

            const mn: usize = @as(usize, m) * adapter.n;
            const mk: usize = @as(usize, m) * adapter.k;
            const delta_slice = art.delta_f32[0..mn];
            @memset(delta_slice, 0.0);
            const x_u16: []const u16 = @ptrCast(x_host[0..mk]);
            try adapter.evalU16(&art.dispatcher, &art.scratch, x_u16, delta_slice, m, null, null);

            try self.compute.wait(sig, self.timeout_ms);
            for (0..mn) |i| {
                const sum: f32 = @as(f32, @floatCast(y_host[i])) + delta_slice[i];
                y_host[i] = @floatCast(sum);
            }
            self.loop_stats.steplora_active_evals += 1;
            return;
        }

        const f = try self.ensureFrame(frame_opt);
        try self.encodeLinearBatched(f, lin, x_bind, y_bind, m, barrier_before);
    }

    /// Populates `ple_f16_host` for `layer l` via `PleStore.lookupForLoopStep` (`C10`, ADR-0019 #2).
    fn preparePleLayer(
        self: *GraphModel,
        prt: *PleRuntime,
        l: usize,
        token_ids: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        m: u32,
        loop_step: u32,
    ) GraphStepError!void {
        const layer_rows = try prt.store.layerTotalRows(@intCast(l));
        const order = self.config.ple_ngram_order;
        const h: usize = self.config.hidden_size;

        // Compute deterministic N-gram row indices per node using ancestor masks + history.
        for (0..m) |node| {
            var ngram_buf: [8]u32 = undefined;
            var collected: usize = 0;
            ngram_buf[0] = token_ids[node];
            collected = 1;

            if (order > 1) {
                var anc_nodes: [8]u32 = undefined;
                var n_anc: usize = 0;
                const mask = ancestor_masks[node];
                if (node > 0) {
                    var cand: isize = @as(isize, @intCast(node)) - 1;
                    while (cand >= 0 and n_anc < order - 1) : (cand -= 1) {
                        if (((mask >> @intCast(cand)) & 1) != 0) {
                            anc_nodes[n_anc] = @intCast(cand);
                            n_anc += 1;
                        }
                    }
                }
                for (0..n_anc) |a| {
                    ngram_buf[collected] = token_ids[anc_nodes[a]];
                    collected += 1;
                }
                var past_back: u32 = 1;
                while (collected < order and base_pos >= past_back) : (past_back += 1) {
                    ngram_buf[collected] = self.token_history[base_pos - past_back];
                    collected += 1;
                }
            }
            // Reverse collected suffix into chronological order `[oldest .. newest]`.
            var chrono: [8]u32 = undefined;
            for (0..collected) |k| {
                chrono[k] = ngram_buf[collected - 1 - k];
            }
            self.ple_row_indices_host[node] = ple_mod.hashPleNgram(chrono[0..collected], order, layer_rows);
        }

        const f32_slice = prt.f32_tmp[0 .. @as(usize, m) * h];
        @memset(f32_slice, 0.0);

        const before_nvme = prt.store.stats().nvme_reads;
        try prt.store.lookupForLoopStep(
            @intCast(l),
            self.ple_row_indices_host[0..m],
            1.0,
            loop_step,
            &prt.scratch,
            f32_slice,
        );
        const after_nvme = prt.store.stats().nvme_reads;
        if (loop_step == 0) {
            self.loop_stats.ple_r0_fills += 1;
        } else {
            self.loop_stats.ple_r_ge1_reuses += 1;
            self.loop_stats.ple_r_ge1_nvme_reads += (after_nvme - before_nvme);
        }

        const layer_stride = @as(usize, self.max_tree_nodes) * h;
        const dst_f16 = self.ple_f16_host[l * layer_stride ..][0 .. @as(usize, m) * h];
        for (0..@as(usize, m) * h) |i| {
            dst_f16[i] = @floatCast(f32_slice[i]);
        }
    }

    fn pleLayerBinding(self: *const GraphModel, l: usize) metal.BufferBinding {
        const layer_bytes = @as(u64, self.max_tree_nodes) * self.config.hidden_size * @sizeOf(f16);
        return .{
            .buffer = self.scratch_buf,
            .offset = self.ple_f16_base_off + @as(u64, @intCast(l)) * layer_bytes,
        };
    }

    fn shouldEarlyExitCpu(self: *const GraphModel, m: u32) bool {
        const w_bytes = self.early_exit_gate_host orelse return false;
        const h: usize = self.config.hidden_size;
        for (0..m) |tok| {
            const x_row = self.x_host[tok * h ..][0..h];
            var dot: f32 = 0.0;
            for (0..h) |d| {
                const wv = readFloatWeightAt(w_bytes, self.early_exit_gate_dtype, d);
                dot += @as(f32, @floatCast(x_row[d])) * wv;
            }
            const g: f32 = 1.0 / (1.0 + @exp(-dot));
            if (g < self.early_exit_threshold) return false;
        }
        return true;
    }

    fn commitCoreLayersAfterEarlyExit(
        self: *GraphModel,
        frame_opt: *?metal.Compute.Frame,
        core_start: u32,
        core_end: u32,
        base_pos: u32,
        m: u32,
    ) GraphStepError!void {
        var chain: [hybrid.MAX_TREE_NODES]u16 = undefined;
        for (0..m) |i| chain[i] = @intCast(i);
        const f = try self.ensureFrame(frame_opt);
        for (core_start..core_end) |l| {
            switch (self.layers[l].attn) {
                .gdn => |g| {
                    const ga = &(self.gdn_arena.?);
                    try self.hybrid_pipes.encodeCommitTreeState(f, .{
                        .k_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_k_off },
                        .v_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_v_off },
                        .g_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_g_off },
                        .beta_tree = .{ .buffer = self.scratch_buf, .offset = g.tree_beta_off },
                        .accepted_nodes = .{ .inline_u16 = chain[0..m] },
                        .accepted_count = m,
                        .state_in = try ga.committedBinding(self.scratch_buf, g.gdn_idx),
                        .state_out = try ga.committedBinding(self.scratch_buf, g.gdn_idx),
                        .num_k_heads = self.config.linear_num_key_heads,
                        .num_v_heads = self.config.linear_num_value_heads,
                        .head_k_dim = self.config.linear_key_head_dim,
                        .head_v_dim = self.config.linear_value_head_dim,
                    }, true);
                    self.loop_stats.gdn_final_commits += 1;
                    if (g.conv1d_w != null and g.conv_k > 1) {
                        try self.hybrid_pipes.encodeGdnConv1dCommit(f, .{
                            .q_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_q_raw_off },
                            .k_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_k_raw_off },
                            .v_raw = .{ .buffer = self.scratch_buf, .offset = g.tree_v_raw_off },
                            .conv_state_in = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .conv_state_out = .{ .buffer = self.scratch_buf, .offset = g.conv_state_off },
                            .accepted_nodes = chain[0..m],
                            .gdn_qk = self.config.gdnQkDim(),
                            .gdn_v = self.config.gdnVDim(),
                            .conv_k = g.conv_k,
                        }, true);
                    }
                },
                .qsa => |*qsa| {
                    try self.encodeQsaCommitAccepted(f, qsa, base_pos, chain[0..m], true);
                },
                .gqa => |gqa| {
                    try self.encodeGqaCommitAccepted(f, gqa, base_pos, chain[0..m], true);
                },
            }
        }
    }

    fn encodeQsaCommitAccepted(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        qsa: *const QsaLayerWeights,
        base_pos: u32,
        accepted_nodes: []const u16,
        barrier_before: bool,
    ) GraphStepError!void {
        const tree_k: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = qsa.tree_k_off };
        const tree_v: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = qsa.tree_v_off };
        var offset: usize = 0;
        var first = barrier_before;
        while (offset < accepted_nodes.len) {
            const cur_base = base_pos + @as(u32, @intCast(offset));
            const max_span = @min(accepted_nodes.len - offset, @as(usize, qsa.kv_ring.chunk_tokens));
            const take: u32 = @intCast(max_span);
            var k_tile0: metal.KvTile = undefined;
            var v_tile0: metal.KvTile = undefined;
            var k_tile1: ?metal.KvTile = null;
            var v_tile1: ?metal.KvTile = null;
            if (qsa.supports_retire_q8 and self.retired_kv_pool != null) {
                const c0: u32 = cur_base / qsa.kv_ring.chunk_tokens;
                const c1: u32 = (cur_base + take - 1) / qsa.kv_ring.chunk_tokens;
                const p0 = self.qsaTilePair(qsa, c0);
                k_tile0 = p0.k;
                v_tile0 = p0.v;
                if (c1 != c0) {
                    const p1 = self.qsaTilePair(qsa, c1);
                    k_tile1 = p1.k;
                    v_tile1 = p1.v;
                }
            } else {
                const span = try qsa.kv_ring.commitTilesForSpan(cur_base, take);
                k_tile0 = span.k_tile0;
                v_tile0 = span.v_tile0;
                k_tile1 = span.k_tile1;
                v_tile1 = span.v_tile1;
            }
            try self.hybrid_pipes.encodeQsaKvTreeCommit(frame, .{
                .k_tree = tree_k,
                .v_tree = tree_v,
                .accepted_nodes = .{ .inline_u16 = accepted_nodes[offset .. offset + take] },
                .accepted_count = take,
                .k_tile0 = k_tile0,
                .v_tile0 = v_tile0,
                .k_tile1 = k_tile1,
                .v_tile1 = v_tile1,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .base_pos = cur_base,
                .chunk_tokens = qsa.kv_ring.chunk_tokens,
                .ring_capacity = qsa.kv_ring.ring_capacity,
            }, first);
            first = true;
            offset += take;
        }

        // Refresh touched micro-block key summaries for `ami_qsa_index_blocks`.
        const new_kv_len: u32 = base_pos + @as(u32, @intCast(accepted_nodes.len));
        const blk_tok = self.config.qsa_block_size;
        const start_block: u32 = if (qsa.kv_ring.window_size > 0) 0 else (base_pos / blk_tok);
        const total_blocks: u32 = if (qsa.kv_ring.window_size > 0)
            (@min(new_kv_len, qsa.kv_ring.window_size) + blk_tok - 1) / blk_tok
        else
            (new_kv_len + blk_tok - 1) / blk_tok;
        const num_blocks: u32 = if (total_blocks > start_block) (total_blocks - start_block) else 1;
        const summary_kv_tiles: metal.BufferBinding = if (qsa.supports_retire_q8 and self.retired_kv_pool != null)
            .{ .buffer = self.scratch_buf, .offset = qsa.full_tile_table_off }
        else
            qsa.kv_ring.tileTableBinding(self.scratch_buf);
        try self.hybrid_pipes.encodeQsaUpdateBlockSummary(frame, .{
            .kv_tiles = summary_kv_tiles,
            .block_keys = .{ .buffer = self.scratch_buf, .offset = qsa.block_keys_off },
            .num_kv_heads = self.config.num_key_value_heads,
            .head_dim = self.config.head_dim,
            .kv_len = new_kv_len,
            .window_size = qsa.kv_ring.window_size,
            .ring_capacity = qsa.kv_ring.ring_capacity,
            .chunk_tokens = qsa.kv_ring.chunk_tokens,
            .block_tokens = blk_tok,
            .start_block = start_block,
            .num_blocks = num_blocks,
        }, true);
    }

    fn qsaTilePair(self: *const GraphModel, qsa: *const QsaLayerWeights, chunk: u32) struct { k: metal.KvTile, v: metal.KvTile } {
        std.debug.assert(chunk >= self.num_retired_chunks);
        const active_idx: usize = chunk - self.num_retired_chunks;
        std.debug.assert(active_idx < self.active_tile_phys_ring.len);
        const phys: usize = self.active_tile_phys_ring[active_idx];
        return .{ .k = qsa.kv_ring.k_tiles[phys], .v = qsa.kv_ring.v_tiles[phys] };
    }

    fn encodeGqaCommitAccepted(
        self: *const GraphModel,
        frame: *metal.Compute.Frame,
        gqa: GqaLayerWeights,
        base_pos: u32,
        accepted_nodes: []const u16,
        barrier_before: bool,
    ) GraphStepError!void {
        const tree_k: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = gqa.tree_k_off };
        const tree_v: metal.BufferBinding = .{ .buffer = self.scratch_buf, .offset = gqa.tree_v_off };
        var offset: usize = 0;
        var first = barrier_before;
        while (offset < accepted_nodes.len) {
            const cur_base = base_pos + @as(u32, @intCast(offset));
            const chunk0: u32 = cur_base / self.chunk_tokens;
            const max_end: u32 = (chunk0 + 2) * self.chunk_tokens;
            const take: u32 = @intCast(@min(accepted_nodes.len - offset, @as(usize, max_end - cur_base)));
            const crosses = ((cur_base + take - 1) / self.chunk_tokens) > chunk0;
            const t0 = self.gqaTilePair(gqa, chunk0);
            const t1 = if (crosses) self.gqaTilePair(gqa, chunk0 + 1) else null;
            try self.attn_pipes.encodeKvTreeCommit(frame, .{
                .k_tree = tree_k,
                .v_tree = tree_v,
                .accepted_nodes = .{ .inline_u16 = accepted_nodes[offset .. offset + take] },
                .accepted_count = take,
                .k_tile0 = t0.k,
                .v_tile0 = t0.v,
                .k_tile1 = if (t1) |p| p.k else null,
                .v_tile1 = if (t1) |p| p.v else null,
                .num_kv_heads = self.config.num_key_value_heads,
                .head_dim = self.config.head_dim,
                .base_pos = cur_base,
                .chunk_tokens = self.chunk_tokens,
            }, first);
            first = true;
            offset += take;
        }
    }

    fn gqaTilePair(self: *const GraphModel, gqa: GqaLayerWeights, chunk: u32) struct { k: metal.KvTile, v: metal.KvTile } {
        if (self.retired_kv_pool != null) {
            std.debug.assert(chunk >= self.num_retired_chunks);
            const active_idx: usize = chunk - self.num_retired_chunks;
            const phys: usize = self.active_tile_phys_ring[active_idx];
            const idx = gqa.kv_tile_start + phys * 2;
            return .{ .k = self.gqa_kv_tiles[idx], .v = self.gqa_kv_tiles[idx + 1] };
        }
        const idx = gqa.kv_tile_start + @as(usize, chunk) * 2;
        return .{ .k = self.gqa_kv_tiles[idx], .v = self.gqa_kv_tiles[idx + 1] };
    }

    fn commitFeatureTaps(self: *GraphModel, accepted_nodes: []const u16) GraphStepError!void {
        var frame = try self.compute.beginFrame(self.timeout_ms);
        errdefer frame.abort();
        try self.encodeCommitFeatureTaps(&frame, accepted_nodes, false);
        const sig = try frame.commit();
        try self.compute.wait(sig, self.timeout_ms);
    }

    fn encodeCommitFeatureTaps(
        self: *GraphModel,
        frame: *metal.Compute.Frame,
        accepted_nodes: []const u16,
        barrier_before: bool,
    ) GraphStepError!void {
        const h = self.config.hidden_size;
        const count: u32 = @intCast(accepted_nodes.len);
        for (accepted_nodes, 0..) |nid, i| {
            self.token_ids_host[i] = nid;
        }
        try self.moe_pipes.encodeGatherRows(frame, .{
            .src = self.x_bind,
            .indices = self.token_ids_bind,
            .dst = self.accepted_top_bind,
            .count = count,
            .row_elems = h,
        }, barrier_before);
        try self.moe_pipes.encodeGatherRows(frame, .{
            .src = .{ .buffer = self.scratch_buf, .offset = self.aux_tree_base_off },
            .indices = self.token_ids_bind,
            .dst = self.accepted_aux_bind,
            .count = count,
            .row_elems = 3 * h,
        }, false);
    }

    fn ensureFrame(self: *GraphModel, frame_opt: *?metal.Compute.Frame) metal.Error!*metal.Compute.Frame {
        if (frame_opt.* == null) {
            frame_opt.* = try self.compute.beginFrame(self.timeout_ms);
        }
        return &(frame_opt.*.?);
    }

    fn flushFrame(self: *GraphModel, frame_opt: *?metal.Compute.Frame) metal.Error!void {
        if (frame_opt.*) |*f| {
            const sig = try f.commit();
            frame_opt.* = null;
            try self.compute.wait(sig, self.timeout_ms);
        }
    }

    // -----------------------------------------------------------------------
    // Internal Loader
    // -----------------------------------------------------------------------

    fn loadInternal(
        allocator: std.mem.Allocator,
        source: AmfSource,
        config: GraphConfig,
        options: GraphLoadOptions,
    ) GraphLoadError!GraphModel {
        try config.validate();
        if (options.max_seq_len == 0 or options.chunk_tokens == 0) return error.InvalidConfig;
        if (options.max_tree_nodes == 0 or options.max_tree_nodes > hybrid.MAX_TREE_NODES) return error.InvalidConfig;

        const effective_max_loops: u32 = options.max_loops_override orelse config.axes.loop_schedule.max_loops;
        if (effective_max_loops == 0) return error.InvalidConfig;

        const device = try metal.Device.open();
        errdefer device.close();
        const dev_info = try device.info();
        const library = try metal.Library.openEmbedded(device);
        errdefer library.close();
        var compute = try metal.Compute.init(device);
        errdefer compute.close();
        const dequant_pipes = try dequant.DequantPipelines.init(device, library);
        errdefer dequant_pipes.close();
        const attn_pipes = try attention.AttentionPipelines.init(device, library);
        errdefer attn_pipes.close();
        const moe_pipes = try moe.MoePipelines.init(device, library);
        errdefer moe_pipes.close();
        const hybrid_pipes = try hybrid.HybridPipelines.init(device, library);
        errdefer hybrid_pipes.close();

        const facts = try sysinfo.Facts.read();
        const wired_budget = options.wired_budget_override orelse budget_mod.Budget.compute(.{
            .memory_bytes = facts.memory_bytes,
            .user_wire_limit = facts.user_wire_limit,
            .global_user_wire_limit = facts.global_user_wire_limit,
            .gpu_working_set_bytes = dev_info.recommendedMaxWorkingSetSize(),
            .configured_cap = null,
            .headroom_bytes = budget_mod.DEFAULT_HEADROOM_BYTES,
        });

        // --- 1. Partition wired tensors into <= MAX_SHARD_BYTES shards (skipping cold/sparse) ---
        const entries = source.entries();
        var shard_sizes: std.ArrayListUnmanaged(usize) = .empty;
        defer shard_sizes.deinit(allocator);
        var entry_shard = try allocator.alloc(u16, entries.len);
        defer allocator.free(entry_shard);
        @memset(entry_shard, std.math.maxInt(u16));

        var has_any_cold = false;
        var cold_entry_count: usize = 0;
        var max_cold_triple_bytes: usize = 0;
        var max_cold_gate_bytes: usize = 0;
        var max_cold_up_bytes: usize = 0;
        var max_cold_down_bytes: usize = 0;
        var inline_loop_adapters: usize = 0;

        // First scan: compute total bytes and identify cold experts & inline loop adapters.
        var cur_shard: usize = 0;
        try shard_sizes.append(allocator, 0);
        for (entries, 0..) |*e, idx| {
            const kind = try e.tensorKind();
            if (kind == .cold_expert) {
                has_any_cold = true;
                cold_entry_count += 1;
                const aligned = try page.alignUp(@intCast(e.blob_bytes));
                const nm = e.nameSlice();
                if (std.mem.endsWith(u8, nm, ".gate_proj")) {
                    max_cold_gate_bytes = @max(max_cold_gate_bytes, aligned);
                } else if (std.mem.endsWith(u8, nm, ".up_proj")) {
                    max_cold_up_bytes = @max(max_cold_up_bytes, aligned);
                } else if (std.mem.endsWith(u8, nm, ".down_proj")) {
                    max_cold_down_bytes = @max(max_cold_down_bytes, aligned);
                }
                continue;
            }
            if (kind == .sparse_table) continue;
            if (kind == .loop_adapter) inline_loop_adapters += 1;

            const aligned = try page.alignUp(@intCast(e.blob_bytes));
            if (shard_sizes.items[cur_shard] > 0 and shard_sizes.items[cur_shard] + aligned > MAX_SHARD_BYTES) {
                cur_shard += 1;
                if (cur_shard >= 64) return error.TooManyWeightShards;
                try shard_sizes.append(allocator, 0);
            }
            entry_shard[idx] = @intCast(cur_shard);
            shard_sizes.items[cur_shard] += aligned;
        }
        max_cold_triple_bytes = max_cold_gate_bytes + max_cold_up_bytes + max_cold_down_bytes;

        // Ensure shard 0 has room for at least 1 page even for empty edge cases.
        if (shard_sizes.items[0] == 0) shard_sizes.items[0] = page.APPLE_PAGE_SIZE;

        // If cold experts exist, reserve `ExpertSlotPool` space at the end of shard 0 so
        // hot experts and pinned cold experts share shard 0's `MTLBuffer` zero-copy!
        // P1-B (`R9`, ADR-0020): auto-scale optional extra cold-expert slots and PLE
        // page-cache pages against live non-anonymous Mach RAM minus safety margin.
        var cold_slot_count: u32 = 0;
        var cold_min_slots: u32 = 0;
        var cold_pool_spec: cache_mod.PoolSpec = undefined;
        var cold_pool_bytes: usize = 0;
        var cold_min_pool_bytes: u64 = 0;
        var cold_slot_stride_bytes: usize = 0;
        if (has_any_cold) {
            if (source == .bytes) return error.InvalidConfig; // Cold experts require file fd
            cold_min_slots = @max(config.num_experts_per_tok, @min(config.num_experts, options.max_tree_nodes * config.num_experts_per_tok));
            cold_slot_count = if (options.cold_slot_count > 0)
                @max(options.cold_slot_count, cold_min_slots)
            else
                @max(@max(config.num_experts_per_tok * 2, 32), cold_min_slots);
            cold_pool_spec = .{
                .num_slots = cold_slot_count,
                .gate_blob_bytes = max_cold_gate_bytes,
                .up_blob_bytes = max_cold_up_bytes,
                .down_blob_bytes = max_cold_down_bytes,
            };
            cold_pool_bytes = try cold_pool_spec.totalBytes();
            const min_spec: cache_mod.PoolSpec = .{
                .num_slots = cold_min_slots,
                .gate_blob_bytes = max_cold_gate_bytes,
                .up_blob_bytes = max_cold_up_bytes,
                .down_blob_bytes = max_cold_down_bytes,
            };
            cold_min_pool_bytes = try min_spec.totalBytes();
            cold_slot_stride_bytes = try cold_pool_spec.slotBytes();
        }

        var hot_shards_bytes: u64 = 0;
        for (shard_sizes.items) |sz| hot_shards_bytes += try page.alignUp(sz);
        const ple_min_pages: usize = if (options.ple_paths.len > 0 and options.ple_cache_pages > 0) 1 else 0;
        const ple_req_extra_pages: usize = if (options.ple_paths.len > 0 and options.ple_cache_pages > ple_min_pages)
            options.ple_cache_pages - ple_min_pages
        else
            0;
        const optional_cold_bytes: u64 = @as(u64, cold_pool_bytes) -| cold_min_pool_bytes;
        const optional_ple_bytes: u64 = @as(u64, ple_req_extra_pages) * page.APPLE_PAGE_SIZE;
        const live_reclaimable_bytes: u64 = options.live_reclaimable_override orelse
            (try vmstat.Snapshot.capture()).liveZeroSwapReclaimableBytes();
        var live_admission = try wired_budget.planWithLiveRam(
            hot_shards_bytes + cold_min_pool_bytes + @as(u64, ple_min_pages) * page.APPLE_PAGE_SIZE,
            optional_cold_bytes + optional_ple_bytes,
            live_reclaimable_bytes,
            .{
                .live_safety_margin_bytes = options.live_safety_margin_bytes,
                .strict_live_ram = options.strict_live_ram,
            },
        );
        var eff_ple_cache_pages: usize = options.ple_cache_pages;
        if (live_admission.cache_scaled_down) {
            var rem_grant: u64 = live_admission.granted_optional_cache_bytes;
            if (has_any_cold and cold_slot_count > cold_min_slots) {
                const extra_req: usize = cold_slot_count - cold_min_slots;
                const granted_extra: usize = budget_mod.scaleOptionalCount(extra_req, cold_slot_stride_bytes, rem_grant, 0);
                cold_slot_count = cold_min_slots + @as(u32, @intCast(granted_extra));
                cold_pool_spec.num_slots = cold_slot_count;
                cold_pool_bytes = try cold_pool_spec.totalBytes();
                rem_grant -|= @as(u64, granted_extra) * cold_slot_stride_bytes;
            }
            if (ple_req_extra_pages > 0) {
                const granted_ple_extra: usize = budget_mod.scaleOptionalCount(ple_req_extra_pages, page.APPLE_PAGE_SIZE, rem_grant, 0);
                eff_ple_cache_pages = ple_min_pages + granted_ple_extra;
            }
        }
        if (has_any_cold) {
            shard_sizes.items[0] += cold_pool_bytes;
        }

        // Allocate wired weight shards & load every wired tensor once.
        const weight_shards = try allocator.alloc(WeightShard, shard_sizes.items.len);
        var shards_built: usize = 0;
        errdefer {
            for (0..shards_built) |si| {
                compute.removeResident(&.{weight_shards[si].buffer}) catch |err| {
                    std.debug.panic("removeResident failed during GraphModel errdefer: {}", .{err});
                };
                weight_shards[si].buffer.close();
                weight_shards[si].arena.deinit();
            }
            allocator.free(weight_shards);
        }

        var total_wired_bytes: u64 = 0;
        for (shard_sizes.items, 0..) |sz, si| {
            const aligned_sz = try page.alignUp(sz);
            try wired_budget.checkPlan(total_wired_bytes + aligned_sz);
            var w_arena = try arena_mod.WiredArena.init(aligned_sz, wired_budget);
            errdefer w_arena.deinit();
            const w_buf = try metal.Buffer.wrap(device, w_arena.bytes());
            errdefer w_buf.close();
            try compute.makeResident(&.{w_buf});
            weight_shards[si] = .{ .arena = w_arena, .buffer = w_buf };
            shards_built += 1;
            total_wired_bytes += aligned_sz;
        }

        const entry_offsets = try allocator.alloc(u64, entries.len);
        defer allocator.free(entry_offsets);
        const entry_slices = try allocator.alloc([]u8, entries.len);
        defer allocator.free(entry_slices);
        const entry_page_slices = try allocator.alloc([]align(page.APPLE_PAGE_SIZE) u8, entries.len);
        defer allocator.free(entry_page_slices);
        @memset(entry_offsets, 0);
        @memset(entry_slices, &.{});
        @memset(entry_page_slices, &.{});

        for (entries, 0..) |*e, idx| {
            const si = entry_shard[idx];
            if (si == std.math.maxInt(u16)) continue;
            const len: usize = @intCast(e.blob_bytes);
            const raw = try source.loadIntoArena(e, &weight_shards[si].arena);
            entry_offsets[idx] = weight_shards[si].arena.offsetOf(raw.ptr);
            entry_slices[idx] = raw[0..len];
            entry_page_slices[idx] = raw;
        }

        // Build lookup map: entry pointer -> index.
        const entry_base_ptr = @intFromPtr(entries.ptr);
        const entryIndexOf = struct {
            fn call(base: usize, e: *const amf.TensorEntry) usize {
                return (@intFromPtr(e) - base) / @sizeOf(amf.TensorEntry);
            }
        }.call;

        const bindLinear = struct {
            fn call(
                src: AmfSource,
                shards: []const WeightShard,
                e_shard: []const u16,
                e_off: []const u64,
                e_slice: []const []u8,
                base_ptr: usize,
                name: []const u8,
                expected_n: u32,
                expected_k: u32,
            ) GraphLoadError!qwen3.LinearBinding {
                const e = src.find(name) orelse blk: {
                    var alt_buf: [96]u8 = undefined;
                    const alt = std.fmt.bufPrint(&alt_buf, "{s}.weight", .{name}) catch return error.MissingTensor;
                    break :blk src.find(alt) orelse return error.MissingTensor;
                };
                const idx = entryIndexOf(base_ptr, e);
                const si = e_shard[idx];
                if (si == std.math.maxInt(u16)) return error.MissingTensor;
                if (e.shape[0] != expected_n or e.shape[1] != expected_k) return error.TensorShapeMismatch;
                const dt = try e.dataType();
                const wdt = amfToWeightDType(dt) orelse return error.UnsupportedDType;
                const buf = shards[si].buffer;
                const off = e_off[idx];
                const raw = e_slice[idx];
                const sb_bind: ?metal.BufferBinding = if (wdt.isQuantized())
                    .{ .buffer = buf, .offset = off + e.params_offset }
                else
                    null;
                const alg_b: u64 = e.blob_bytes;
                return .{
                    .weight = .{ .buffer = buf, .offset = off },
                    .scale_bias = sb_bind,
                    .n = expected_n,
                    .k = expected_k,
                    .dtype = wdt,
                    .alg_bytes = alg_b,
                    .layout = try e.tensorLayout(),
                    .param_kind = try e.quantParamKind(),
                    .weight_host = raw,
                };
            }
        }.call;

        const bindNorm = struct {
            fn call(
                src: AmfSource,
                shards: []const WeightShard,
                e_shard: []const u16,
                e_off: []const u64,
                e_slice: []const []u8,
                base_ptr: usize,
                name: []const u8,
                expected_len: u32,
            ) GraphLoadError!qwen3.NormBinding {
                const e = src.find(name) orelse blk: {
                    var alt_buf: [96]u8 = undefined;
                    const alt = std.fmt.bufPrint(&alt_buf, "{s}.weight", .{name}) catch return error.MissingTensor;
                    break :blk src.find(alt) orelse return error.MissingTensor;
                };
                const idx = entryIndexOf(base_ptr, e);
                const si = e_shard[idx];
                if (si == std.math.maxInt(u16)) return error.MissingTensor;
                if (e.shape[0] != expected_len) return error.TensorShapeMismatch;
                const dt = try e.dataType();
                const fdt: quant.FloatDType = switch (dt) {
                    .f16 => .f16,
                    .bf16 => .bf16,
                    else => return error.UnsupportedDType,
                };
                return .{
                    .binding = .{ .buffer = shards[si].buffer, .offset = e_off[idx] },
                    .dtype = fdt,
                    .host_bytes = e_slice[idx],
                };
            }
        }.call;

        // --- 2. Global embeddings, final norm, lm_head, optional early-exit gate ---
        const h = config.hidden_size;
        const embed = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, "embed_tokens", config.vocab_size, h);
        const final_norm = try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, "norm.weight", h);
        const lm_head = if (config.tie_word_embeddings)
            embed
        else
            try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, "lm_head", config.vocab_size, h);

        var early_exit_gate_host: ?[]const u8 = null;
        var early_exit_gate_dtype: quant.FloatDType = .bf16;
        if (source.find("loop.early_exit_gate.weight")) |eg_e| {
            const idx = entryIndexOf(entry_base_ptr, eg_e);
            early_exit_gate_host = entry_slices[idx];
            early_exit_gate_dtype = switch (try eg_e.dataType()) {
                .f16 => .f16,
                .bf16 => .bf16,
                else => return error.UnsupportedDType,
            };
        }

        // --- 3. Compute exact scratch arena size ---
        const max_m = options.max_tree_nodes;
        var num_gdn_layers: u32 = 0;
        var num_qsa_layers: u32 = 0;
        var num_gqa_layers: u32 = 0;
        for (config.axes.layer_attention_kinds) |ak| {
            switch (ak) {
                .gated_deltanet => num_gdn_layers += 1,
                .qsa_sparse => num_qsa_layers += 1,
                .gqa_full => num_gqa_layers += 1,
                .mla_csa2 => return error.UnsupportedOperatorAxis,
            }
        }
        const num_chunks: u32 = (options.max_seq_len + options.chunk_tokens - 1) / options.chunk_tokens;
        const qsa_ring_mode: hybrid.HybridKvRingMode = if (config.qsa_window_size > 0 and config.qsa_window_size <= hybrid.QSA_SWA_WINDOW_TOKENS)
            .sliding_window_128
        else
            .full_chunk_512;
        const qsa_full_chunks: u32 = switch (qsa_ring_mode) {
            .sliding_window_128 => 1,
            .full_chunk_512 => (options.max_seq_len + hybrid.QSA_FULL_CHUNK_TOKENS - 1) / hybrid.QSA_FULL_CHUNK_TOKENS,
        };
        const qsa_supports_retire = (num_qsa_layers > 0 and qsa_ring_mode == .full_chunk_512);
        const retire_after: u32 = if (options.continuous_batch_only) 0 else options.kv_retire_q8_after_chunks;
        const retire_chunk_tokens: u32 = if (qsa_supports_retire) hybrid.QSA_FULL_CHUNK_TOKENS else options.chunk_tokens;
        const retire_total_chunks: u32 = if (qsa_supports_retire) qsa_full_chunks else num_chunks;
        const num_retirable_kv_layers: u32 = num_gqa_layers + (if (qsa_supports_retire) num_qsa_layers else 0);
        const use_retired_q8 = (retire_after > 0 and num_retirable_kv_layers > 0 and retire_total_chunks > retire_after);
        if (use_retired_q8) {
            if ((config.head_dim % attention.KV_Q8_GROUP_SIZE) != 0) return error.InvalidConfig;
            if (qsa_supports_retire and num_gqa_layers > 0 and options.chunk_tokens != hybrid.QSA_FULL_CHUNK_TOKENS) {
                return error.InvalidConfig;
            }
        }
        const active_kv_chunks: u32 = if (options.continuous_batch_only)
            1
        else if (use_retired_q8)
            @min(retire_total_chunks, if (max_m > 1) @max(retire_after, 2) else retire_after)
        else
            retire_total_chunks;
        const max_retired_per_layer: u32 = if (use_retired_q8) retire_total_chunks - retire_after else 0;
        const qsa_num_tiles: u32 = if (options.continuous_batch_only)
            1
        else if (qsa_supports_retire and use_retired_q8)
            active_kv_chunks
        else
            qsa_full_chunks;
        const gqa_num_tiles: u32 = if (options.continuous_batch_only)
            1
        else if (use_retired_q8)
            active_kv_chunks
        else
            num_chunks;
        const retired_pool_spec: attention.RetiredKvQ8PoolSpec = .{
            .num_kv_heads = config.num_key_value_heads,
            .head_dim = config.head_dim,
            .chunk_tokens = retire_chunk_tokens,
            .max_chunks = if (use_retired_q8) num_retirable_kv_layers * max_retired_per_layer else 1,
        };
        const qsa_max_blocks: u32 = switch (qsa_ring_mode) {
            .sliding_window_128 => (hybrid.QSA_SWA_WINDOW_TOKENS + config.qsa_block_size - 1) / config.qsa_block_size,
            .full_chunk_512 => (options.max_seq_len + config.qsa_block_size - 1) / config.qsa_block_size,
        };

        const supports_wide_prefill = (num_gqa_layers == 0) or (config.head_dim == 64 or config.head_dim == 128);
        const req_wide_cap: u32 = @min(attention.MAX_WIDE_PREFILL_TOKENS, @max(@as(u32, 1), options.wide_prefill_chunk_tokens));
        const ple_limited_wide: u32 = if (options.ple_paths.len > 0)
            @min(req_wide_cap, @as(u32, ple_mod.MAX_BATCH_TOKENS))
        else
            req_wide_cap;
        const wide_prefill_max_tokens: u32 = if (options.gpu_prefill_mode == .wide and supports_wide_prefill)
            @min(ple_limited_wide, options.max_seq_len)
        else
            0;
        const wide_enabled = (wide_prefill_max_tokens > 0);
        const w_cap: usize = wide_prefill_max_tokens;
        const multi_loop = (effective_max_loops > 1);
        const use_aliased_scratch = wide_enabled and options.alias_prefill_tree_scratch;
        const logits_cap: usize = if (options.compute_all_logits and wide_enabled) @max(@as(usize, max_m), w_cap) else max_m;
        const feat_cap: usize = if (wide_enabled and options.enable_feature_taps) @max(@as(usize, max_m), w_cap) else max_m;

        const addAligned = struct {
            fn call(total: *usize, n_bytes: usize) void {
                total.* += amf.align64(n_bytes) + 64;
            }
        }.call;

        const max_q_dim: u32 = @max(@max(config.qDim(), config.gdnQkDim()), h);
        const max_kv_dim: u32 = @max(@max(config.kvDim(), config.gdnQkDim()), @max(config.gdnVDim(), h));
        const max_attn_out: u32 = @max(@max(config.qDim(), config.gdnVDim()), h);
        const max_inter: u32 = @max(@max(config.intermediate_size, config.moe_intermediate_size), @max(config.shared_expert_intermediate_size, h));

        const moe_ws_spec: moe.WorkspaceSpec = .{
            .max_tokens = max_m,
            .num_experts = if (config.axes.ffn_kind == .moe_swiglu) config.num_experts else 1,
            .top_k = if (config.axes.ffn_kind == .moe_swiglu) config.num_experts_per_tok else 1,
            .hidden_size = h,
            .intermediate_size = if (config.axes.ffn_kind == .moe_swiglu) config.moe_intermediate_size else h,
            .shared_intermediate_size = config.shared_expert_intermediate_size,
        };
        const wp_moe_ws_spec: moe.WorkspaceSpec = .{
            .max_tokens = if (wide_enabled) wide_prefill_max_tokens else 1,
            .num_experts = if (config.axes.ffn_kind == .moe_swiglu) config.num_experts else 1,
            .top_k = if (config.axes.ffn_kind == .moe_swiglu) config.num_experts_per_tok else 1,
            .hidden_size = h,
            .intermediate_size = if (config.axes.ffn_kind == .moe_swiglu) config.moe_intermediate_size else h,
            .shared_intermediate_size = config.shared_expert_intermediate_size,
        };

        // Region B (`tree_scratch_bytes`): decode & `verifyTree` scratch (aliased with Region A under `P0-B2`).
        var tree_scratch_bytes: usize = 0;
        if (config.axes.residual_kind == .gated_4branch) {
            tree_scratch_bytes += try hybrid.GrStreamArena.requiredBytes(.{
                .max_tokens = max_m,
                .hidden_dim = h,
            });
            addAligned(&tree_scratch_bytes, @as(usize, max_m) * 4 * 4); // gr_g_read
            addAligned(&tree_scratch_bytes, @as(usize, max_m) * 4 * 4); // gr_g_write
            addAligned(&tree_scratch_bytes, @as(usize, max_m) * 4 * 4); // gr_decay
        } else {
            addAligned(&tree_scratch_bytes, @as(usize, max_m) * 4 * 4);
            addAligned(&tree_scratch_bytes, @as(usize, max_m) * 4 * 4);
            addAligned(&tree_scratch_bytes, @as(usize, max_m) * 4 * 4);
        }
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * 4); // token_ids
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * 4); // positions
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * @sizeOf(hybrid.Pos3D)); // pos3d
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * 8); // ancestor_masks
        addAligned(&tree_scratch_bytes, @as(usize, config.num_hidden_layers) * max_m * h * 2); // ple_f16
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * h * 2); // x
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * h * 2); // x_init
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * h * 2); // x_norm
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_q_dim * 2); // q
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_kv_dim * 2); // k
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_kv_dim * 2); // v
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_q_dim * 2); // q_conv
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_kv_dim * 2); // k_conv
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_kv_dim * 2); // v_conv
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_attn_out * 2); // gdn_z
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * @max(@as(u32, 1), config.linear_num_value_heads) * 4); // gdn_g
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * @max(@as(u32, 1), config.linear_num_value_heads) * 4); // gdn_beta
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_attn_out * 2); // attn_out
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * h * 2); // proj_out
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_inter * 2); // dense_gate
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_inter * 2); // dense_up
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * max_inter * 2); // dense_act
        addAligned(&tree_scratch_bytes, @as(usize, max_m) * 3 * h * 2); // aux_tree
        if (num_gqa_layers > 0) {
            addAligned(&tree_scratch_bytes, @as(usize, max_m) * config.num_attention_heads * attention.MAX_SPLITS * (config.head_dim + 2) * 4);
        } else {
            addAligned(&tree_scratch_bytes, 64);
        }
        if (config.axes.ffn_kind == .moe_swiglu) {
            addAligned(&tree_scratch_bytes, (try moe.MoeWorkspace.requiredBytes(moe_ws_spec)) + 4096);
        }
        if (num_gdn_layers > 0) {
            const qk_d = config.gdnQkDim();
            const v_d = config.gdnVDim();
            for (0..num_gdn_layers) |_| {
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * qk_d * 2); // tree_k
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * v_d * 2); // tree_v
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * config.linear_num_value_heads * 4); // tree_g
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * config.linear_num_value_heads * 4); // tree_beta
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * qk_d * 2); // tree_q_raw
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * qk_d * 2); // tree_k_raw
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * v_d * 2); // tree_v_raw
            }
        }
        if (num_qsa_layers > 0) {
            for (0..num_qsa_layers) |_| {
                tree_scratch_bytes += (try hybrid.QsaIndexBuffer.requiredBytes(.{
                    .max_nodes = max_m,
                    .num_kv_heads = config.num_key_value_heads,
                    .top_k_blocks = config.qsa_top_k_blocks,
                })) + 128;
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * config.kvDim() * 2); // tree_k
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * config.kvDim() * 2); // tree_v
            }
        }
        if (num_gqa_layers > 0) {
            for (0..num_gqa_layers) |_| {
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * config.kvDim() * 2);
                addAligned(&tree_scratch_bytes, @as(usize, max_m) * config.kvDim() * 2);
            }
        }
        tree_scratch_bytes += 4 * page.APPLE_PAGE_SIZE;

        // Region A (`wide_scratch_bytes`): wide-prefill (`512–1024`-token) rolling activation scratch + MoeWorkspace.
        var wide_scratch_bytes: usize = 0;
        if (wide_enabled) {
            if (config.axes.residual_kind == .gated_4branch) {
                wide_scratch_bytes += try hybrid.GrStreamArena.requiredBytes(.{
                    .max_tokens = wide_prefill_max_tokens,
                    .hidden_dim = h,
                });
                addAligned(&wide_scratch_bytes, w_cap * 4 * 4);
                addAligned(&wide_scratch_bytes, w_cap * 4 * 4);
                addAligned(&wide_scratch_bytes, w_cap * 4 * 4);
            }
            addAligned(&wide_scratch_bytes, w_cap * 4); // wp_token_ids
            addAligned(&wide_scratch_bytes, w_cap * @sizeOf(hybrid.Pos3D)); // wp_pos3d
            if (options.ple_paths.len > 0) {
                addAligned(&wide_scratch_bytes, @as(usize, config.num_hidden_layers) * w_cap * h * 2);
            }
            addAligned(&wide_scratch_bytes, w_cap * h * 2); // wp_x
            if (multi_loop) addAligned(&wide_scratch_bytes, w_cap * h * 2); // wp_x_init
            addAligned(&wide_scratch_bytes, w_cap * h * 2); // wp_x_norm
            addAligned(&wide_scratch_bytes, w_cap * max_q_dim * 2); // wp_q
            addAligned(&wide_scratch_bytes, w_cap * max_kv_dim * 2); // wp_k
            addAligned(&wide_scratch_bytes, w_cap * max_kv_dim * 2); // wp_v
            if (num_gdn_layers > 0) {
                addAligned(&wide_scratch_bytes, w_cap * max_q_dim * 2); // wp_q_conv
                addAligned(&wide_scratch_bytes, w_cap * max_kv_dim * 2); // wp_k_conv
                addAligned(&wide_scratch_bytes, w_cap * max_kv_dim * 2); // wp_v_conv
                addAligned(&wide_scratch_bytes, w_cap * max_attn_out * 2); // wp_gdn_z
                addAligned(&wide_scratch_bytes, w_cap * @max(@as(u32, 1), config.linear_num_value_heads) * 4); // wp_gdn_g
                addAligned(&wide_scratch_bytes, w_cap * @max(@as(u32, 1), config.linear_num_value_heads) * 4); // wp_gdn_beta
                if (multi_loop) {
                    const qk_d = config.gdnQkDim();
                    const v_d = config.gdnVDim();
                    for (0..num_gdn_layers) |_| {
                        addAligned(&wide_scratch_bytes, w_cap * qk_d * 2); // wp_k
                        addAligned(&wide_scratch_bytes, w_cap * v_d * 2); // wp_v
                        addAligned(&wide_scratch_bytes, w_cap * config.linear_num_value_heads * 4); // wp_g
                        addAligned(&wide_scratch_bytes, w_cap * config.linear_num_value_heads * 4); // wp_beta
                        addAligned(&wide_scratch_bytes, w_cap * qk_d * 2); // wp_q_raw
                    }
                }
            }
            addAligned(&wide_scratch_bytes, w_cap * max_attn_out * 2); // wp_attn_out
            addAligned(&wide_scratch_bytes, w_cap * h * 2); // wp_proj_out
            if (config.axes.ffn_kind == .dense_swiglu) {
                addAligned(&wide_scratch_bytes, w_cap * max_inter * 2); // wp_dense_gate
                addAligned(&wide_scratch_bytes, w_cap * max_inter * 2); // wp_dense_up
                addAligned(&wide_scratch_bytes, w_cap * max_inter * 2); // wp_dense_act
            } else {
                addAligned(&wide_scratch_bytes, (try moe.MoeWorkspace.requiredBytes(wp_moe_ws_spec)) + 4096);
            }
            if (num_qsa_layers > 0) {
                for (0..num_qsa_layers) |_| {
                    wide_scratch_bytes += (try hybrid.QsaIndexBuffer.requiredBytes(.{
                        .max_nodes = wide_prefill_max_tokens,
                        .num_kv_heads = config.num_key_value_heads,
                        .top_k_blocks = config.qsa_top_k_blocks,
                    })) + 128;
                    if (qsa_ring_mode == .sliding_window_128 and multi_loop) {
                        addAligned(&wide_scratch_bytes, w_cap * config.kvDim() * 2);
                        addAligned(&wide_scratch_bytes, w_cap * config.kvDim() * 2);
                    }
                }
            }
            wide_scratch_bytes += 4 * page.APPLE_PAGE_SIZE;
        }

        // Persistent / non-aliased scratch bytes (GDN states, Conv1D states, KV tile tables,
        // QSA block_keys, R5 retired pool, logits, out_token_ids, accepted taps, expert tables).
        var persist_bytes: usize = 0;
        if (num_gdn_layers > 0) {
            persist_bytes += try hybrid.GdnStateArena.requiredBytes(.{
                .num_gdn_layers = num_gdn_layers,
                .num_v_heads = config.linear_num_value_heads,
                .head_k_dim = config.linear_key_head_dim,
                .head_v_dim = config.linear_value_head_dim,
            });
            const qk_d = config.gdnQkDim();
            const v_d = config.gdnVDim();
            const conv_d = 2 * qk_d + v_d;
            const conv_hist = if (config.linear_conv_kernel_dim > 1) config.linear_conv_kernel_dim - 1 else 1;
            for (0..num_gdn_layers) |_| {
                addAligned(&persist_bytes, @as(usize, conv_hist) * conv_d * 2); // conv_state
                addAligned(&persist_bytes, @as(usize, conv_hist) * conv_d * 2); // conv_state_tmp
            }
        }
        if (use_retired_q8) {
            persist_bytes += try attention.RetiredKvQ8Pool.requiredArenaBytes(retired_pool_spec);
        }
        if (num_qsa_layers > 0) {
            for (0..num_qsa_layers) |_| {
                persist_bytes += try hybrid.HybridKvRing.tableRequiredBytes(qsa_num_tiles) + 64;
                if (qsa_supports_retire and use_retired_q8) {
                    addAligned(&persist_bytes, @as(usize, qsa_full_chunks) * @sizeOf(attention.KvTilePair));
                    addAligned(&persist_bytes, @as(usize, max_retired_per_layer) * @sizeOf(u32));
                }
                addAligned(&persist_bytes, @as(usize, config.num_key_value_heads) * qsa_max_blocks * config.head_dim * 2); // block_keys
            }
        }
        if (num_gqa_layers > 0) {
            for (0..num_gqa_layers) |_| {
                addAligned(&persist_bytes, @as(usize, gqa_num_tiles) * @sizeOf(attention.KvTilePair));
                if (use_retired_q8) {
                    addAligned(&persist_bytes, @as(usize, max_retired_per_layer) * @sizeOf(u32));
                }
            }
        }
        addAligned(&persist_bytes, logits_cap * 4); // out_token_ids
        addAligned(&persist_bytes, logits_cap * config.vocab_size * 2); // logits
        addAligned(&persist_bytes, @as(usize, max_m) * moe.MAX_TOP_K * @sizeOf(moe.TopKCandidate)); // topk
        addAligned(&persist_bytes, feat_cap * h * 2); // accepted_top
        addAligned(&persist_bytes, feat_cap * 3 * h * 2); // accepted_aux
        addAligned(&persist_bytes, @as(usize, config.num_hidden_layers) * 64); // per-layer sched headers
        if (config.axes.ffn_kind == .moe_swiglu) {
            for (0..config.num_hidden_layers) |_| {
                addAligned(&persist_bytes, @as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights));
                addAligned(&persist_bytes, @as(usize, config.num_experts)); // hetero expert_dtypes
            }
        }

        const aliased_block_bytes: usize = if (use_aliased_scratch)
            @max(try page.alignUp(wide_scratch_bytes), try page.alignUp(tree_scratch_bytes))
        else if (wide_enabled)
            (try page.alignUp(wide_scratch_bytes)) + (try page.alignUp(tree_scratch_bytes))
        else
            try page.alignUp(tree_scratch_bytes);

        // Generous page margin for sub-arena page rounding.
        const scratch_bytes = try page.alignUp(aliased_block_bytes + persist_bytes + 32 * page.APPLE_PAGE_SIZE);
        // Update live admission with exact required wired footprint (enforcing `--strict-live-ram` before mlock).
        live_admission = try wired_budget.planWithLiveRam(
            total_wired_bytes + scratch_bytes,
            optional_cold_bytes + optional_ple_bytes,
            live_reclaimable_bytes,
            .{
                .live_safety_margin_bytes = options.live_safety_margin_bytes,
                .strict_live_ram = options.strict_live_ram,
            },
        );
        try wired_budget.checkPlan(total_wired_bytes + scratch_bytes);

        var scratch_arena = try arena_mod.WiredArena.init(scratch_bytes, wired_budget);
        errdefer scratch_arena.deinit();
        const scratch_buf = try metal.Buffer.wrap(device, scratch_arena.bytes());
        errdefer scratch_buf.close();
        try compute.makeResident(&.{scratch_buf});
        total_wired_bytes += scratch_bytes;

        // G16 (`P2`, ADR-0020): startup `.metallib` numerical canary over
        // the resident `scratch_buf` BEFORE carving any sub-arena.
        var canary_result: ?dequant.CanaryResult = null;
        if (options.startup_canary) {
            canary_result = try dequant_pipes.runStartupNumericalCanaryInBuffer(&compute, scratch_buf, scratch_arena.bytes());
            @memset(scratch_arena.bytes()[0..page.APPLE_PAGE_SIZE], 0);
        }

        // P0-B2 (`R9`, ADR-0020): Allocate aliased wide-prefill / verifyTree pair FIRST
        // (at offset 0 of `scratch_arena`) so `arena_a.offsetOf(p) == arena_b.offsetOf(p) ==
        // scratch_arena.offsetOf(p)` for `GrStreamArena`, `QsaIndexBuffer`, `MoeWorkspace`,
        // and all Metal buffer bindings while keeping `lm_head` 100% wired!
        var aliased_scratch_saved_bytes: usize = 0;
        var aliased_pair: ?arena_mod.AliasedScratchPair = null;
        if (use_aliased_scratch) {
            const pair = try scratch_arena.allocAliasedPair(wide_scratch_bytes, tree_scratch_bytes);
            std.debug.assert(scratch_arena.offsetOf(pair.backing.ptr) == 0);
            aliased_scratch_saved_bytes = pair.saved_bytes;
            aliased_pair = pair;
        }
        const wp_arena: *arena_mod.WiredArena = if (aliased_pair != null) &aliased_pair.?.arena_a else &scratch_arena;
        const tr_arena: *arena_mod.WiredArena = if (aliased_pair != null) &aliased_pair.?.arena_b else &scratch_arena;

        // --- Carve Region A: wide-prefill scratch (`wp_arena`) ---
        var wp_token_ids_host: []u32 = &.{};
        var wp_pos3d_host: []hybrid.Pos3D = &.{};
        var wp_ple_f16_host: []f16 = &.{};
        var wp_token_ids_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_pos3d_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_ple_f16_base_off: u64 = 0;
        var wp_gr_arena: ?hybrid.GrStreamArena = null;
        var wp_gr_g_read_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_gr_g_write_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_gr_decay_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_x_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_x_init_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_x_norm_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_q_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_k_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_v_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_q_conv_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_k_conv_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_v_conv_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_gdn_z_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_gdn_g_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_gdn_beta_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_attn_out_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_proj_out_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_dense_gate_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_dense_up_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_dense_act_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
        var wp_moe_ws: ?moe.MoeWorkspace = null;
        if (wide_enabled) {
            if (config.axes.residual_kind == .gated_4branch) {
                wp_gr_arena = try hybrid.GrStreamArena.init(wp_arena, .{
                    .max_tokens = wide_prefill_max_tokens,
                    .hidden_dim = h,
                });
                const gr_r = try wp_arena.allocAligned(w_cap * 4 * 4, 64);
                const gr_w = try wp_arena.allocAligned(w_cap * 4 * 4, 64);
                const gr_d = try wp_arena.allocAligned(w_cap * 4 * 4, 64);
                wp_gr_g_read_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_r.ptr) };
                wp_gr_g_write_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_w.ptr) };
                wp_gr_decay_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_d.ptr) };
            }
            const wt_raw = try wp_arena.allocAligned(w_cap * 4, 64);
            const wp3_raw = try wp_arena.allocAligned(w_cap * @sizeOf(hybrid.Pos3D), 64);
            @memset(wt_raw, 0);
            wp_token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, wt_raw[0 .. w_cap * 4]));
            wp_pos3d_host = @alignCast(std.mem.bytesAsSlice(hybrid.Pos3D, wp3_raw[0 .. w_cap * @sizeOf(hybrid.Pos3D)]));
            wp_token_ids_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wt_raw.ptr) };
            wp_pos3d_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wp3_raw.ptr) };
            if (options.ple_paths.len > 0) {
                const wple_raw = try wp_arena.allocAligned(@as(usize, config.num_hidden_layers) * w_cap * h * 2, 64);
                @memset(wple_raw, 0);
                wp_ple_f16_host = @alignCast(std.mem.bytesAsSlice(f16, wple_raw[0 .. @as(usize, config.num_hidden_layers) * w_cap * h * 2]));
                wp_ple_f16_base_off = scratch_arena.offsetOf(wple_raw.ptr);
            }
            const wx_raw = try wp_arena.allocAligned(w_cap * h * 2, 64);
            wp_x_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wx_raw.ptr) };
            if (multi_loop) {
                const wxi_raw = try wp_arena.allocAligned(w_cap * h * 2, 64);
                wp_x_init_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wxi_raw.ptr) };
            }
            const wxn_raw = try wp_arena.allocAligned(w_cap * h * 2, 64);
            const wq_raw = try wp_arena.allocAligned(w_cap * max_q_dim * 2, 64);
            const wk_raw = try wp_arena.allocAligned(w_cap * max_kv_dim * 2, 64);
            const wv_raw = try wp_arena.allocAligned(w_cap * max_kv_dim * 2, 64);
            wp_x_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wxn_raw.ptr) };
            wp_q_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wq_raw.ptr) };
            wp_k_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wk_raw.ptr) };
            wp_v_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wv_raw.ptr) };
            if (num_gdn_layers > 0) {
                const wqc = try wp_arena.allocAligned(w_cap * max_q_dim * 2, 64);
                const wkc = try wp_arena.allocAligned(w_cap * max_kv_dim * 2, 64);
                const wvc = try wp_arena.allocAligned(w_cap * max_kv_dim * 2, 64);
                const wgz = try wp_arena.allocAligned(w_cap * max_attn_out * 2, 64);
                const wgg = try wp_arena.allocAligned(w_cap * @max(@as(u32, 1), config.linear_num_value_heads) * 4, 64);
                const wgb = try wp_arena.allocAligned(w_cap * @max(@as(u32, 1), config.linear_num_value_heads) * 4, 64);
                wp_q_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wqc.ptr) };
                wp_k_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wkc.ptr) };
                wp_v_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wvc.ptr) };
                wp_gdn_z_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wgz.ptr) };
                wp_gdn_g_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wgg.ptr) };
                wp_gdn_beta_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wgb.ptr) };
            }
            const wao_raw = try wp_arena.allocAligned(w_cap * max_attn_out * 2, 64);
            const wpo_raw = try wp_arena.allocAligned(w_cap * h * 2, 64);
            wp_attn_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wao_raw.ptr) };
            wp_proj_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wpo_raw.ptr) };
            if (config.axes.ffn_kind == .dense_swiglu) {
                const wdg = try wp_arena.allocAligned(w_cap * max_inter * 2, 64);
                const wdu = try wp_arena.allocAligned(w_cap * max_inter * 2, 64);
                const wda = try wp_arena.allocAligned(w_cap * max_inter * 2, 64);
                wp_dense_gate_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wdg.ptr) };
                wp_dense_up_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wdu.ptr) };
                wp_dense_act_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(wda.ptr) };
            } else {
                wp_moe_ws = try moe.MoeWorkspace.init(wp_arena, wp_moe_ws_spec);
            }
        }

        // --- Carve Region B: decode & `verifyTree` scratch (`tr_arena`) ---
        const gr_arena: ?hybrid.GrStreamArena = if (config.axes.residual_kind == .gated_4branch)
            try hybrid.GrStreamArena.init(tr_arena, .{
                .max_tokens = max_m,
                .hidden_dim = h,
            })
        else
            null;

        const gr_g_read_raw = try tr_arena.allocAligned(@as(usize, max_m) * 4 * 4, 64);
        const gr_g_write_raw = try tr_arena.allocAligned(@as(usize, max_m) * 4 * 4, 64);
        const gr_decay_raw = try tr_arena.allocAligned(@as(usize, max_m) * 4 * 4, 64);

        const tok_raw = try tr_arena.allocAligned(@as(usize, max_m) * 4, 64);
        const pos_raw = try tr_arena.allocAligned(@as(usize, max_m) * 4, 64);
        const pos3d_raw = try tr_arena.allocAligned(@as(usize, max_m) * @sizeOf(hybrid.Pos3D), 64);
        const msk_raw = try tr_arena.allocAligned(@as(usize, max_m) * 8, 64);
        const ple_f16_raw = try tr_arena.allocAligned(@as(usize, config.num_hidden_layers) * max_m * h * 2, 64);
        @memset(ple_f16_raw, 0);
        const x_raw = try tr_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const x_init_raw = try tr_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const x_norm_raw = try tr_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const q_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_q_dim * 2, 64);
        const k_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_kv_dim * 2, 64);
        const v_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_kv_dim * 2, 64);
        const q_conv_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_q_dim * 2, 64);
        const k_conv_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_kv_dim * 2, 64);
        const v_conv_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_kv_dim * 2, 64);
        const gdn_z_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_attn_out * 2, 64);
        const gdn_g_raw = try tr_arena.allocAligned(@as(usize, max_m) * @max(@as(u32, 1), config.linear_num_value_heads) * 4, 64);
        const gdn_beta_raw = try tr_arena.allocAligned(@as(usize, max_m) * @max(@as(u32, 1), config.linear_num_value_heads) * 4, 64);
        const attn_out_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_attn_out * 2, 64);
        const proj_out_raw = try tr_arena.allocAligned(@as(usize, max_m) * h * 2, 64);
        const dgate_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_inter * 2, 64);
        const dup_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_inter * 2, 64);
        const dact_raw = try tr_arena.allocAligned(@as(usize, max_m) * max_inter * 2, 64);
        const aux_tree_raw = try tr_arena.allocAligned(@as(usize, max_m) * 3 * h * 2, 64);
        const gqa_scratch_raw = if (num_gqa_layers > 0)
            try tr_arena.allocAligned(@as(usize, max_m) * config.num_attention_heads * attention.MAX_SPLITS * (config.head_dim + 2) * 4, 64)
        else
            try tr_arena.allocAligned(64, 64);

        const moe_ws: ?moe.MoeWorkspace = if (config.axes.ffn_kind == .moe_swiglu)
            try moe.MoeWorkspace.init(tr_arena, moe_ws_spec)
        else
            null;

        // --- Carve persistent non-aliased sub-arenas inside `scratch_arena` ---
        const retired_kv_pool: ?attention.RetiredKvQ8Pool = if (use_retired_q8)
            try attention.RetiredKvQ8Pool.init(&scratch_arena, retired_pool_spec)
        else
            null;
        const retired_kv_wired_bytes: u64 = if (use_retired_q8)
            @as(u64, try attention.RetiredKvQ8Pool.dataRequiredBytes(retired_pool_spec))
        else
            0;

        const gdn_arena: ?hybrid.GdnStateArena = if (num_gdn_layers > 0)
            try hybrid.GdnStateArena.init(&scratch_arena, .{
                .num_gdn_layers = num_gdn_layers,
                .num_v_heads = config.linear_num_value_heads,
                .head_k_dim = config.linear_key_head_dim,
                .head_v_dim = config.linear_value_head_dim,
            })
        else
            null;

        const out_tok_raw = try scratch_arena.allocAligned(logits_cap * 4, 64);
        const logits_raw = try scratch_arena.allocAligned(logits_cap * config.vocab_size * 2, 64);
        const topk_raw = try scratch_arena.allocAligned(@as(usize, max_m) * moe.MAX_TOP_K * @sizeOf(moe.TopKCandidate), 64);
        const acc_top_raw = try scratch_arena.allocAligned(feat_cap * h * 2, 64);
        const acc_aux_raw = try scratch_arena.allocAligned(feat_cap * 3 * h * 2, 64);
        const layer_sched_hdr_raw = try scratch_arena.allocAligned(@as(usize, config.num_hidden_layers) * 64, 64);
        @memset(layer_sched_hdr_raw, 0);

        // --- 4. Initialize optional ColdRuntime (`C3`-`C5`) ---
        var cold_rt: ?*ColdRuntime = null;
        errdefer if (cold_rt) |crt| {
            allocator.free(crt.r0_selected);
            crt.expert_cache.deinit(allocator);
            crt.reader.deinit(allocator);
            crt.pool.deinit();
            if (crt.amf_file) |*f| f.close(allocator);
            allocator.destroy(crt);
        };
        if (has_any_cold) {
            const crt = try allocator.create(ColdRuntime);
            errdefer allocator.destroy(crt);
            var pool = try cache_mod.ExpertSlotPool.initFromArena(&weight_shards[0].arena, weight_shards[0].buffer, cold_pool_spec);
            errdefer pool.deinit();
            var reader = try aio.AsyncExpertReader.init(allocator, .{ .max_inflight = 16 });
            errdefer reader.deinit(allocator);
            var cold_amf = try amf.AmfFile.open(allocator, source.file.path);
            errdefer cold_amf.close(allocator);
            const r0_len: usize = @as(usize, config.num_hidden_layers) * max_m * config.num_experts_per_tok;
            const r0_selected = try allocator.alloc(u16, r0_len);
            errdefer allocator.free(r0_selected);
            @memset(r0_selected, 0);
            crt.* = .{
                .pool = pool,
                .reader = reader,
                .expert_cache = undefined,
                .amf_file = cold_amf,
                .slot_count = cold_slot_count,
                .r0_selected = r0_selected,
            };
            crt.expert_cache = try cache_mod.ExpertCache.init(allocator, &crt.pool, &crt.reader, .{
                .num_layers = @intCast(config.num_hidden_layers),
                .num_experts = @intCast(config.num_experts),
            });
            cold_rt = crt;
        }

        // --- 5. Bind per-layer weights, KV tiles/rings, and MoE expert tables ---
        const gqa_kv_tiles = if (num_gqa_layers > 0)
            try allocator.alloc(metal.KvTile, @as(usize, num_gqa_layers) * gqa_num_tiles * 2)
        else
            try allocator.alloc(metal.KvTile, 0);
        var active_kv_wired_bytes: u64 = 0;
        var gqa_tiles_built: usize = 0;
        errdefer {
            if (gqa_tiles_built > 0) {
                compute.removeTilesResident(gqa_kv_tiles[0..gqa_tiles_built]) catch |err| {
                    std.debug.panic("removeTilesResident failed during GraphModel errdefer: {}", .{err});
                };
                for (0..gqa_tiles_built) |ti| gqa_kv_tiles[ti].close();
            }
            allocator.free(gqa_kv_tiles);
        }

        const layers = try allocator.alloc(GraphLayerWeights, config.num_hidden_layers);
        var layers_built: usize = 0;
        errdefer {
            for (0..layers_built) |li| {
                switch (layers[li].attn) {
                    .qsa => |*qsa| {
                        qsa.kv_ring.removeResident(compute) catch |err| {
                            std.debug.panic("QSA kv_ring.removeResident failed during errdefer: {}", .{err});
                        };
                        qsa.kv_ring.deinit(allocator);
                    },
                    else => {},
                }
            }
            allocator.free(layers);
        }

        var gdn_cursor: u32 = 0;
        var qsa_cursor: u32 = 0;
        var gqa_cursor: u32 = 0;
        var name_buf: [160]u8 = undefined;
        var single_expert_bytes: u64 = 0;
        var total_layers_non_expert_bytes: u64 = 0;
        var total_layers_static_bytes: u64 = 0;

        for (0..config.num_hidden_layers) |l| {
            const in_ln_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable;
            const in_ln = try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, in_ln_nm, h);
            const post_ln_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable;
            const post_ln = try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, post_ln_nm, h);

            var gr_w: ?GrLayerWeights = null;
            if (config.axes.residual_kind == .gated_4branch) {
                const r_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.gr.read_gate.weight", .{l}) catch unreachable;
                const re = source.find(r_nm) orelse return error.MissingTensor;
                const r_idx = entryIndexOf(entry_base_ptr, re);
                const w_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.gr.write_gate.weight", .{l}) catch unreachable;
                const we = source.find(w_nm) orelse return error.MissingTensor;
                const w_idx = entryIndexOf(entry_base_ptr, we);
                const d_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.gr.decay_gate.weight", .{l}) catch unreachable;
                const de = source.find(d_nm) orelse return error.MissingTensor;
                const d_idx = entryIndexOf(entry_base_ptr, de);
                const gdt: quant.FloatDType = if ((try re.dataType()) == .bf16) .bf16 else .f16;
                gr_w = .{
                    .read_gate = .{ .buffer = weight_shards[entry_shard[r_idx]].buffer, .offset = entry_offsets[r_idx] },
                    .write_gate = .{ .buffer = weight_shards[entry_shard[w_idx]].buffer, .offset = entry_offsets[w_idx] },
                    .decay_gate = .{ .buffer = weight_shards[entry_shard[d_idx]].buffer, .offset = entry_offsets[d_idx] },
                    .dtype = gdt,
                };
                total_layers_non_expert_bytes += 3 * 4 * @as(u64, h) * 2;
            }

            var attn_union: LayerAttention = undefined;
            var layer_attn_bytes: u64 = 0;
            switch (config.axes.layer_attention_kinds[l]) {
                .gated_deltanet => {
                    const qk_d = config.gdnQkDim();
                    const v_d = config.gdnVDim();
                    const q_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.q_proj", .{l}) catch unreachable;
                    const q_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, q_nm, qk_d, h);
                    const k_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.k_proj", .{l}) catch unreachable;
                    const k_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, k_nm, qk_d, h);
                    const v_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.v_proj", .{l}) catch unreachable;
                    const v_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, v_nm, v_d, h);
                    const g_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.g_proj", .{l}) catch unreachable;
                    const g_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, g_nm, v_d, h);
                    const o_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.o_proj", .{l}) catch unreachable;
                    const o_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, o_nm, h, v_d);

                    const a_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.a_proj.weight", .{l}) catch unreachable;
                    const ae = source.find(a_nm) orelse return error.MissingTensor;
                    const a_idx = entryIndexOf(entry_base_ptr, ae);
                    const b_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.b_proj.weight", .{l}) catch unreachable;
                    const be = source.find(b_nm) orelse return error.MissingTensor;
                    const b_idx = entryIndexOf(entry_base_ptr, be);
                    const al_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.A_log", .{l}) catch unreachable;
                    const ale = source.find(al_nm) orelse return error.MissingTensor;
                    const al_idx = entryIndexOf(entry_base_ptr, ale);
                    const dt_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.dt_bias", .{l}) catch unreachable;
                    const dte = source.find(dt_nm) orelse return error.MissingTensor;
                    const dt_idx = entryIndexOf(entry_base_ptr, dte);

                    var conv_w: ?metal.BufferBinding = null;
                    var conv_k: u32 = 0;
                    const cw_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.conv1d.weight", .{l}) catch unreachable;
                    if (source.find(cw_nm)) |cwe| {
                        const cw_idx = entryIndexOf(entry_base_ptr, cwe);
                        conv_w = .{ .buffer = weight_shards[entry_shard[cw_idx]].buffer, .offset = entry_offsets[cw_idx] };
                        conv_k = @intCast(cwe.shape[1]);
                    }
                    const norm_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.linear_attn.norm.weight", .{l}) catch unreachable;
                    const out_norm: ?qwen3.NormBinding = if (source.find(norm_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, norm_nm, config.linear_value_head_dim)
                    else
                        null;

                    const tk_r = try tr_arena.allocAligned(@as(usize, max_m) * qk_d * 2, 64);
                    const tv_r = try tr_arena.allocAligned(@as(usize, max_m) * v_d * 2, 64);
                    const tg_r = try tr_arena.allocAligned(@as(usize, max_m) * config.linear_num_value_heads * 4, 64);
                    const tb_r = try tr_arena.allocAligned(@as(usize, max_m) * config.linear_num_value_heads * 4, 64);
                    const tqr_r = try tr_arena.allocAligned(@as(usize, max_m) * qk_d * 2, 64);
                    const tkr_r = try tr_arena.allocAligned(@as(usize, max_m) * qk_d * 2, 64);
                    const tvr_r = try tr_arena.allocAligned(@as(usize, max_m) * v_d * 2, 64);
                    var wp_k_off: u64 = 0;
                    var wp_v_off: u64 = 0;
                    var wp_g_off: u64 = 0;
                    var wp_beta_off: u64 = 0;
                    var wp_q_raw_off: u64 = 0;
                    if (wide_enabled and multi_loop) {
                        const wk_r = try wp_arena.allocAligned(w_cap * qk_d * 2, 64);
                        const wv_r = try wp_arena.allocAligned(w_cap * v_d * 2, 64);
                        const wg_r = try wp_arena.allocAligned(w_cap * config.linear_num_value_heads * 4, 64);
                        const wb_r = try wp_arena.allocAligned(w_cap * config.linear_num_value_heads * 4, 64);
                        const wqr_r = try wp_arena.allocAligned(w_cap * qk_d * 2, 64);
                        wp_k_off = scratch_arena.offsetOf(wk_r.ptr);
                        wp_v_off = scratch_arena.offsetOf(wv_r.ptr);
                        wp_g_off = scratch_arena.offsetOf(wg_r.ptr);
                        wp_beta_off = scratch_arena.offsetOf(wb_r.ptr);
                        wp_q_raw_off = scratch_arena.offsetOf(wqr_r.ptr);
                    }
                    const conv_hist = if (conv_k > 1) conv_k - 1 else 1;
                    const conv_d = 2 * qk_d + v_d;
                    const cs_r = try scratch_arena.allocAligned(@as(usize, conv_hist) * conv_d * 2, 64);
                    const cst_r = try scratch_arena.allocAligned(@as(usize, conv_hist) * conv_d * 2, 64);
                    @memset(cs_r, 0);
                    @memset(cst_r, 0);

                    attn_union = .{ .gdn = .{
                        .gdn_idx = gdn_cursor,
                        .q_proj = q_p,
                        .k_proj = k_p,
                        .v_proj = v_p,
                        .g_proj = g_p,
                        .o_proj = o_p,
                        .a_proj = .{ .buffer = weight_shards[entry_shard[a_idx]].buffer, .offset = entry_offsets[a_idx] },
                        .b_proj = .{ .buffer = weight_shards[entry_shard[b_idx]].buffer, .offset = entry_offsets[b_idx] },
                        .ab_dtype = if ((try ae.dataType()) == .bf16) .bf16 else .f16,
                        .a_log = .{ .buffer = weight_shards[entry_shard[al_idx]].buffer, .offset = entry_offsets[al_idx] },
                        .dt_bias = .{ .buffer = weight_shards[entry_shard[dt_idx]].buffer, .offset = entry_offsets[dt_idx] },
                        .conv1d_w = conv_w,
                        .conv_k = conv_k,
                        .out_norm = out_norm,
                        .tree_k_off = scratch_arena.offsetOf(tk_r.ptr),
                        .tree_v_off = scratch_arena.offsetOf(tv_r.ptr),
                        .tree_g_off = scratch_arena.offsetOf(tg_r.ptr),
                        .tree_beta_off = scratch_arena.offsetOf(tb_r.ptr),
                        .tree_q_raw_off = scratch_arena.offsetOf(tqr_r.ptr),
                        .tree_q_raw_host = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(tqr_r))),
                        .tree_k_raw_off = scratch_arena.offsetOf(tkr_r.ptr),
                        .tree_v_raw_off = scratch_arena.offsetOf(tvr_r.ptr),
                        .conv_state_off = scratch_arena.offsetOf(cs_r.ptr),
                        .conv_state_tmp_off = scratch_arena.offsetOf(cst_r.ptr),
                        .wp_k_off = wp_k_off,
                        .wp_v_off = wp_v_off,
                        .wp_g_off = wp_g_off,
                        .wp_beta_off = wp_beta_off,
                        .wp_q_raw_off = wp_q_raw_off,
                    } };
                    gdn_cursor += 1;
                    layer_attn_bytes = q_p.alg_bytes + k_p.alg_bytes + v_p.alg_bytes + g_p.alg_bytes + o_p.alg_bytes;
                },
                .qsa_sparse => {
                    const q_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable;
                    const q_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, q_nm, config.qDim(), h);
                    const k_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable;
                    const k_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, k_nm, config.kvDim(), h);
                    const v_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable;
                    const v_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, v_nm, config.kvDim(), h);
                    const o_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable;
                    const o_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, o_nm, h, config.qDim());

                    const qn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable;
                    const qn: ?qwen3.NormBinding = if (source.find(qn_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, qn_nm, config.head_dim)
                    else
                        null;
                    const kn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable;
                    const kn: ?qwen3.NormBinding = if (source.find(kn_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, kn_nm, config.head_dim)
                    else
                        null;

                    var kv_ring = try hybrid.HybridKvRing.init(allocator, device, &scratch_arena, .{
                        .mode = qsa_ring_mode,
                        .num_kv_heads = config.num_key_value_heads,
                        .head_dim = config.head_dim,
                        .num_tiles = qsa_num_tiles,
                        .window_size_override = if (config.qsa_window_size > 0 and qsa_ring_mode == .sliding_window_128) config.qsa_window_size else null,
                    });
                    errdefer kv_ring.deinit(allocator);
                    try kv_ring.makeResident(compute);
                    for (kv_ring.k_tiles) |kt| {
                        const tb = (try kt.info()).allocBytes();
                        total_wired_bytes += tb;
                        active_kv_wired_bytes += tb;
                    }
                    for (kv_ring.v_tiles) |vt| {
                        const tb = (try vt.info()).allocBytes();
                        total_wired_bytes += tb;
                        active_kv_wired_bytes += tb;
                    }

                    var qsa_full_tbl_off: u64 = 0;
                    var qsa_full_tbl_host: []attention.KvTilePair = &.{};
                    var qsa_ret_bind: metal.BufferBinding = .{ .buffer = scratch_buf, .offset = 0 };
                    var qsa_ret_host: []u32 = &.{};
                    const qsa_use_ret = (qsa_supports_retire and use_retired_q8);
                    if (qsa_use_ret) {
                        const ft_raw = try scratch_arena.allocAligned(@as(usize, qsa_full_chunks) * @sizeOf(attention.KvTilePair), 64);
                        @memset(ft_raw, 0);
                        qsa_full_tbl_off = scratch_arena.offsetOf(ft_raw.ptr);
                        qsa_full_tbl_host = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, ft_raw[0 .. @as(usize, qsa_full_chunks) * @sizeOf(attention.KvTilePair)]));
                        for (0..qsa_num_tiles) |ti| {
                            qsa_full_tbl_host[ti] = attention.KvTilePair.fromTiles(kv_ring.k_tiles[ti], kv_ring.v_tiles[ti], .rgba16f);
                        }
                        const r_raw = try scratch_arena.allocAligned(@as(usize, max_retired_per_layer) * @sizeOf(u32), 64);
                        @memset(r_raw, 0);
                        qsa_ret_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(r_raw.ptr) };
                        qsa_ret_host = @alignCast(std.mem.bytesAsSlice(u32, r_raw[0 .. @as(usize, max_retired_per_layer) * @sizeOf(u32)]));
                    }

                    const idx_buf = try hybrid.QsaIndexBuffer.init(tr_arena, .{
                        .max_nodes = max_m,
                        .num_kv_heads = config.num_key_value_heads,
                        .top_k_blocks = config.qsa_top_k_blocks,
                    });
                    const wp_idx_buf: ?hybrid.QsaIndexBuffer = if (wide_enabled)
                        try hybrid.QsaIndexBuffer.init(wp_arena, .{
                            .max_nodes = wide_prefill_max_tokens,
                            .num_kv_heads = config.num_key_value_heads,
                            .top_k_blocks = config.qsa_top_k_blocks,
                        })
                    else
                        null;
                    var qsa_wp_k_off: u64 = 0;
                    var qsa_wp_v_off: u64 = 0;
                    if (wide_enabled and qsa_ring_mode == .sliding_window_128 and multi_loop) {
                        const qwk = try wp_arena.allocAligned(w_cap * config.kvDim() * 2, 64);
                        const qwv = try wp_arena.allocAligned(w_cap * config.kvDim() * 2, 64);
                        qsa_wp_k_off = scratch_arena.offsetOf(qwk.ptr);
                        qsa_wp_v_off = scratch_arena.offsetOf(qwv.ptr);
                    }
                    const bk_raw = try scratch_arena.allocAligned(@as(usize, config.num_key_value_heads) * qsa_max_blocks * config.head_dim * 2, 64);
                    @memset(bk_raw, 0);
                    const tk_r = try tr_arena.allocAligned(@as(usize, max_m) * config.kvDim() * 2, 64);
                    const tv_r = try tr_arena.allocAligned(@as(usize, max_m) * config.kvDim() * 2, 64);

                    attn_union = .{ .qsa = .{
                        .qsa_idx = qsa_cursor,
                        .q_proj = q_p,
                        .k_proj = k_p,
                        .v_proj = v_p,
                        .o_proj = o_p,
                        .q_norm = if (qn) |n| n.binding else null,
                        .k_norm = if (kn) |n| n.binding else null,
                        .qk_norm_dtype = if (qn) |n| n.dtype else .bf16,
                        .kv_ring = kv_ring,
                        .index_buf = idx_buf,
                        .wp_index_buf = wp_idx_buf,
                        .block_keys_off = scratch_arena.offsetOf(bk_raw.ptr),
                        .max_blocks = qsa_max_blocks,
                        .tree_k_off = scratch_arena.offsetOf(tk_r.ptr),
                        .tree_v_off = scratch_arena.offsetOf(tv_r.ptr),
                        .wp_k_off = qsa_wp_k_off,
                        .wp_v_off = qsa_wp_v_off,
                        .supports_retire_q8 = qsa_use_ret,
                        .full_tile_table_off = qsa_full_tbl_off,
                        .full_tile_table_host = qsa_full_tbl_host,
                        .retired_ids_binding = qsa_ret_bind,
                        .retired_ids_host = qsa_ret_host,
                    } };
                    qsa_cursor += 1;
                    layer_attn_bytes = q_p.alg_bytes + k_p.alg_bytes + v_p.alg_bytes + o_p.alg_bytes;
                },
                .gqa_full => {
                    const q_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable;
                    const q_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, q_nm, config.qDim(), h);
                    const k_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable;
                    const k_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, k_nm, config.kvDim(), h);
                    const v_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable;
                    const v_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, v_nm, config.kvDim(), h);
                    const o_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable;
                    const o_p = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, o_nm, h, config.qDim());

                    const qn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable;
                    const qn: ?qwen3.NormBinding = if (source.find(qn_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, qn_nm, config.head_dim)
                    else
                        null;
                    const kn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable;
                    const kn: ?qwen3.NormBinding = if (source.find(kn_nm) != null)
                        try bindNorm(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, kn_nm, config.head_dim)
                    else
                        null;

                    const tile_start = gqa_tiles_built;
                    const tbl_raw = try scratch_arena.allocAligned(@as(usize, gqa_num_tiles) * @sizeOf(attention.KvTilePair), 64);
                    const tbl: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, tbl_raw[0 .. @as(usize, gqa_num_tiles) * @sizeOf(attention.KvTilePair)]));
                    const rows = config.num_key_value_heads * options.chunk_tokens;
                    for (0..gqa_num_tiles) |c| {
                        gqa_kv_tiles[gqa_tiles_built] = try metal.KvTile.init(device, config.head_dim, rows, .{});
                        const tbk = (try gqa_kv_tiles[gqa_tiles_built].info()).allocBytes();
                        total_wired_bytes += tbk;
                        active_kv_wired_bytes += tbk;
                        gqa_tiles_built += 1;
                        gqa_kv_tiles[gqa_tiles_built] = try metal.KvTile.init(device, config.head_dim, rows, .{});
                        const tbv = (try gqa_kv_tiles[gqa_tiles_built].info()).allocBytes();
                        total_wired_bytes += tbv;
                        active_kv_wired_bytes += tbv;
                        gqa_tiles_built += 1;
                        tbl[c] = attention.KvTilePair.fromTiles(gqa_kv_tiles[gqa_tiles_built - 2], gqa_kv_tiles[gqa_tiles_built - 1], .rgba16f);
                    }
                    try compute.makeTilesResident(gqa_kv_tiles[tile_start..gqa_tiles_built]);
                    const gqa_ret_bind: metal.BufferBinding, const gqa_ret_host: []u32 = if (use_retired_q8) blk: {
                        const r_raw = try scratch_arena.allocAligned(@as(usize, max_retired_per_layer) * @sizeOf(u32), 64);
                        @memset(r_raw, 0);
                        break :blk .{
                            .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(r_raw.ptr) },
                            @alignCast(std.mem.bytesAsSlice(u32, r_raw[0 .. @as(usize, max_retired_per_layer) * @sizeOf(u32)])),
                        };
                    } else .{ .{ .buffer = scratch_buf, .offset = 0 }, &.{} };
                    const tk_r = try tr_arena.allocAligned(@as(usize, max_m) * config.kvDim() * 2, 64);
                    const tv_r = try tr_arena.allocAligned(@as(usize, max_m) * config.kvDim() * 2, 64);

                    attn_union = .{ .gqa = .{
                        .gqa_idx = gqa_cursor,
                        .q_proj = q_p,
                        .k_proj = k_p,
                        .v_proj = v_p,
                        .o_proj = o_p,
                        .q_norm = if (qn) |n| n.binding else null,
                        .k_norm = if (kn) |n| n.binding else null,
                        .qk_norm_dtype = if (qn) |n| n.dtype else .bf16,
                        .kv_tiles_binding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tbl_raw.ptr) },
                        .kv_tiles_host = tbl,
                        .retired_ids_binding = gqa_ret_bind,
                        .retired_ids_host = gqa_ret_host,
                        .kv_tile_start = tile_start,
                        .tree_k_off = scratch_arena.offsetOf(tk_r.ptr),
                        .tree_v_off = scratch_arena.offsetOf(tv_r.ptr),
                    } };
                    gqa_cursor += 1;
                    layer_attn_bytes = q_p.alg_bytes + k_p.alg_bytes + v_p.alg_bytes + o_p.alg_bytes;
                },
                .mla_csa2 => return error.UnsupportedOperatorAxis,
            }

            var lw: GraphLayerWeights = .{
                .shard_idx = 0,
                .input_layernorm = in_ln,
                .post_attention_layernorm = post_ln,
                .gr = gr_w,
                .attn = attn_union,
            };

            switch (config.axes.ffn_kind) {
                .dense_swiglu => {
                    const gp_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable;
                    lw.dense_gate = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, gp_nm, config.intermediate_size, h);
                    const up_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable;
                    lw.dense_up = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, up_nm, config.intermediate_size, h);
                    const dp_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable;
                    lw.dense_down = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, dp_nm, h, config.intermediate_size);
                    total_layers_static_bytes += layer_attn_bytes + lw.dense_gate.alg_bytes + lw.dense_up.alg_bytes + lw.dense_down.alg_bytes;
                },
                .moe_swiglu => {
                    // Router (`layers.{l}.mlp.gate.weight` or `layers.{l}.mlp.gate`)
                    const r_nm_w = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate.weight", .{l}) catch unreachable;
                    const re = source.find(r_nm_w) orelse blk: {
                        const r_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate", .{l}) catch unreachable;
                        break :blk source.find(r_nm) orelse return error.MissingTensor;
                    };
                    const r_idx = entryIndexOf(entry_base_ptr, re);
                    const r_si = entry_shard[r_idx];
                    lw.router_binding = .{ .buffer = weight_shards[r_si].buffer, .offset = entry_offsets[r_idx] };
                    const rdt = try re.dataType();
                    if (amfToWeightDType(rdt)) |wdt| {
                        if (wdt.isQuantized()) {
                            lw.router_quant_dtype = wdt;
                            lw.router_scale_bias = .{ .buffer = weight_shards[r_si].buffer, .offset = entry_offsets[r_idx] + re.params_offset };
                        } else {
                            lw.router_dtype = if (wdt == .bf16) .bf16 else .f16;
                        }
                    } else if (rdt == .f32) {
                        lw.router_dtype = .f32;
                    } else return error.UnsupportedDType;

                    // Optional `shared_expert` + `shared_expert_gate` (`G7`).
                    const sg_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_expert.gate_proj", .{l}) catch unreachable;
                    if (source.find(sg_nm) != null) {
                        const sh_inter = if (config.shared_expert_intermediate_size > 0) config.shared_expert_intermediate_size else config.moe_intermediate_size;
                        const sg = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, sg_nm, sh_inter, h);
                        const su_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_expert.up_proj", .{l}) catch unreachable;
                        const su = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, su_nm, sh_inter, h);
                        const sd_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_expert.down_proj", .{l}) catch unreachable;
                        const sd = try bindLinear(source, weight_shards, entry_shard, entry_offsets, entry_slices, entry_base_ptr, sd_nm, h, sh_inter);
                        lw.shared_expert = .{
                            .gate_qweight_offset = sg.weight.offset,
                            .gate_sb_offset = if (sg.scale_bias) |b| b.offset else 0,
                            .up_qweight_offset = su.weight.offset,
                            .up_sb_offset = if (su.scale_bias) |b| b.offset else 0,
                            .down_qweight_offset = sd.weight.offset,
                            .down_sb_offset = if (sd.scale_bias) |b| b.offset else 0,
                        };
                        lw.shared_expert_buf = sg.weight.buffer;
                        lw.shared_expert_dtype = sg.dtype;
                        layer_attn_bytes += sg.alg_bytes + su.alg_bytes + sd.alg_bytes;

                        const sgate_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.shared_expert_gate.weight", .{l}) catch unreachable;
                        if (source.find(sgate_nm)) |sge| {
                            const sg_idx = entryIndexOf(entry_base_ptr, sge);
                            lw.shared_gate_weight = .{ .buffer = weight_shards[entry_shard[sg_idx]].buffer, .offset = entry_offsets[sg_idx] };
                            lw.shared_gate_dtype = switch (try sge.dataType()) {
                                .bf16 => .bf16,
                                .f16 => .f16,
                                .f32 => .f32,
                                else => return error.UnsupportedDType,
                            };
                        }
                    }

                    // Routed experts table `[num_experts]ExpertWeights` (`E <= 1024`).
                    const tbl_raw = try scratch_arena.allocAligned(@as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights), 64);
                    const tbl: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, tbl_raw[0 .. @as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights)]));
                    const dt_raw = try scratch_arena.allocAligned(config.num_experts, 64);
                    lw.expert_table = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tbl_raw.ptr) };
                    lw.expert_table_host = tbl;

                    var first_dt: ?quant.WeightDType = null;
                    var hetero = false;
                    var exp_buf: metal.Buffer = weight_shards[0].buffer;
                    for (0..config.num_experts) |e| {
                        const g_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }) catch unreachable;
                        const ge = source.find(g_nm) orelse return error.MissingTensor;
                        const u_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }) catch unreachable;
                        const ue = source.find(u_nm) orelse return error.MissingTensor;
                        const d_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }) catch unreachable;
                        const de = source.find(d_nm) orelse return error.MissingTensor;

                        const wdt = amfToWeightDType(try ge.dataType()) orelse return error.UnsupportedDType;
                        if (first_dt) |fdt| {
                            if (fdt != wdt) hetero = true;
                        } else {
                            first_dt = wdt;
                        }
                        dt_raw[e] = @backingInt(wdt);
                        if (wdt.isQuantized()) {
                            const bucket: usize = switch (wdt) {
                                .q2_affine => 0,
                                .q3_affine => 1,
                                .q4_affine => 2,
                                else => 0,
                            };
                            lw.expert_scheme_counts[bucket] += 1;
                        }

                        if ((try ge.tensorKind()) == .cold_expert) {
                            lw.has_cold_experts = true;
                            lw.cold_expert_count += 1;
                            const triple = try cache_mod.ColdExpertTriple.fromAmfEntries(cold_rt.?.amf_file.?.fd, 0, ge, ue, de);
                            try cold_rt.?.expert_cache.registerCold(@intCast(l), @intCast(e), triple);
                            tbl[e] = std.mem.zeroes(moe.ExpertWeights);
                            if (single_expert_bytes == 0) single_expert_bytes = triple.totalBlobBytes();
                        } else {
                            const g_idx = entryIndexOf(entry_base_ptr, ge);
                            const u_idx = entryIndexOf(entry_base_ptr, ue);
                            const d_idx = entryIndexOf(entry_base_ptr, de);
                            exp_buf = weight_shards[entry_shard[g_idx]].buffer;
                            const ew = makeExpertWeights(wdt, entry_offsets[g_idx], ge.params_offset, entry_offsets[u_idx], ue.params_offset, entry_offsets[d_idx], de.params_offset);
                            tbl[e] = ew;
                            if (cold_rt) |crt| {
                                try crt.expert_cache.registerHot(@intCast(l), @intCast(e), .{
                                    .gate = try expert.WiredExpert.init(entry_page_slices[g_idx]),
                                    .up = try expert.WiredExpert.init(entry_page_slices[u_idx]),
                                    .down = try expert.WiredExpert.init(entry_page_slices[d_idx]),
                                    .weights = ew,
                                    .metal_buffer = exp_buf,
                                });
                            }
                            if (single_expert_bytes == 0) {
                                single_expert_bytes = ge.blob_bytes + ue.blob_bytes + de.blob_bytes;
                            }
                        }
                    }
                    lw.expert_weights_buf = exp_buf;
                    lw.expert_weight_dtype = first_dt orelse .q3_affine;
                    lw.is_hetero_moe = hetero;
                    if (hetero) {
                        lw.expert_dtypes_binding = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dt_raw.ptr) };
                    }
                    total_layers_non_expert_bytes += layer_attn_bytes;
                },
            }

            layers[l] = lw;
            layers_built += 1;
        }

        // --- 6. Optional PLE Store (`C10`, `F_NOCACHE`) ---
        var ple_rt: ?*PleRuntime = null;
        errdefer if (ple_rt) |prt| {
            allocator.free(prt.f32_tmp);
            prt.scratch.deinit(allocator);
            prt.store.deinit(allocator);
            allocator.destroy(prt);
        };
        const ple_max_m: u32 = @max(max_m, wide_prefill_max_tokens);
        if (options.ple_paths.len > 0) {
            const prt = try allocator.create(PleRuntime);
            errdefer allocator.destroy(prt);
            var store = try ple_mod.PleStore.open(allocator, options.ple_paths, .{
                .cache = .{
                    .cache_pages = eff_ple_cache_pages,
                    .budget_override = wired_budget,
                },
            });
            errdefer store.deinit(allocator);
            var ple_scratch = try ple_mod.LoopPleScratch.init(
                allocator,
                @intCast(config.num_hidden_layers),
                @intCast(ple_max_m),
                h,
                wired_budget,
            );
            errdefer ple_scratch.deinit(allocator);
            const f32_tmp = try allocator.alloc(f32, @as(usize, ple_max_m) * h);
            errdefer allocator.free(f32_tmp);
            total_wired_bytes += store.totalWiredBytes() + ple_scratch.totalWiredBytes();
            prt.* = .{ .store = store, .scratch = ple_scratch, .f32_tmp = f32_tmp };
            ple_rt = prt;
        }

        // --- 7. Optional SME2 Step-LoRA Loop Adapters (`W16`, inline or external `.amf`) ---
        var adapter_rt: ?*LoopAdapterRuntime = null;
        errdefer if (adapter_rt) |art| {
            art.deinit(allocator);
            allocator.destroy(art);
        };
        if (inline_loop_adapters > 0 or options.loop_adapter_amf_path != null) {
            adapter_rt = try loadStepLoraAdapters(
                allocator,
                source,
                entry_slices,
                entry_base_ptr,
                options.loop_adapter_amf_path,
                config,
                effective_max_loops,
                max_m,
                wired_budget,
                &total_wired_bytes,
            );
        }

        const token_history = try allocator.alloc(u32, options.max_seq_len + @max(max_m, wide_prefill_max_tokens));
        errdefer allocator.free(token_history);
        @memset(token_history, 0);

        const ple_row_indices_host = try allocator.alloc(u64, ple_max_m);
        errdefer allocator.free(ple_row_indices_host);
        @memset(ple_row_indices_host, 0);

        const active_tile_phys_ring: []u32 = if (use_retired_q8) blk: {
            const ring = try allocator.alloc(u32, active_kv_chunks);
            for (0..active_kv_chunks) |ri| ring[ri] = @intCast(ri);
            break :blk ring;
        } else &.{};
        errdefer if (active_tile_phys_ring.len > 0) allocator.free(active_tile_phys_ring);

        const aux_ids = options.aux_layer_ids orelse qwen3.defaultEagle3AuxLayerIds(config.num_hidden_layers);
        const embed_row_alg_bytes = embed.alg_bytes / config.vocab_size;
        const global_head_bytes = embed_row_alg_bytes + lm_head.alg_bytes;

        return .{
            .config = config,
            .effective_max_loops = effective_max_loops,
            .enable_loop_early_exit = options.enable_loop_early_exit,
            .early_exit_threshold = options.early_exit_threshold orelse config.axes.loop_schedule.early_exit_threshold,
            .max_seq_len = options.max_seq_len,
            .max_tree_nodes = max_m,
            .chunk_tokens = options.chunk_tokens,
            .num_chunks = num_chunks,
            .seq_len = 0,
            .timeout_ms = options.timeout_ms,
            .wired_budget_bytes = wired_budget.limit_bytes,
            .total_wired_bytes = total_wired_bytes,
            .enable_feature_taps = options.enable_feature_taps,
            .aux_layer_ids = aux_ids,
            .device = device,
            .library = library,
            .compute = compute,
            .dequant_pipes = dequant_pipes,
            .attn_pipes = attn_pipes,
            .moe_pipes = moe_pipes,
            .hybrid_pipes = hybrid_pipes,
            .weight_shards = weight_shards,
            .scratch_arena = scratch_arena,
            .scratch_buf = scratch_buf,
            .gdn_arena = gdn_arena,
            .gr_arena = gr_arena,
            .gqa_kv_tiles = gqa_kv_tiles,
            .embed = embed,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .early_exit_gate_host = early_exit_gate_host,
            .early_exit_gate_dtype = early_exit_gate_dtype,
            .layers = layers,
            .ple_rt = ple_rt,
            .cold_rt = cold_rt,
            .adapter_rt = adapter_rt,
            .token_history = token_history,
            .token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, tok_raw[0 .. @as(usize, max_m) * 4])),
            .positions_host = @alignCast(std.mem.bytesAsSlice(u32, pos_raw[0 .. @as(usize, max_m) * 4])),
            .pos3d_host = @alignCast(std.mem.bytesAsSlice(hybrid.Pos3D, pos3d_raw[0 .. @as(usize, max_m) * @sizeOf(hybrid.Pos3D)])),
            .ancestor_masks_host = @alignCast(std.mem.bytesAsSlice(u64, msk_raw[0 .. @as(usize, max_m) * 8])),
            .out_token_ids_host = @alignCast(std.mem.bytesAsSlice(u32, out_tok_raw[0 .. logits_cap * 4])),
            .logits_host = @alignCast(std.mem.bytesAsSlice(f16, logits_raw[0 .. logits_cap * config.vocab_size * 2])),
            .topk_host = @alignCast(std.mem.bytesAsSlice(moe.TopKCandidate, topk_raw[0 .. @as(usize, max_m) * moe.MAX_TOP_K * @sizeOf(moe.TopKCandidate)])),
            .ple_row_indices_host = ple_row_indices_host,
            .ple_f16_host = @alignCast(std.mem.bytesAsSlice(f16, ple_f16_raw[0 .. @as(usize, config.num_hidden_layers) * max_m * h * 2])),
            .x_host = @alignCast(std.mem.bytesAsSlice(f16, x_raw[0 .. @as(usize, max_m) * h * 2])),
            .x_norm_host = @alignCast(std.mem.bytesAsSlice(f16, x_norm_raw[0 .. @as(usize, max_m) * h * 2])),
            .q_host = @alignCast(std.mem.bytesAsSlice(f16, q_raw[0 .. @as(usize, max_m) * max_q_dim * 2])),
            .k_host = @alignCast(std.mem.bytesAsSlice(f16, k_raw[0 .. @as(usize, max_m) * max_kv_dim * 2])),
            .v_host = @alignCast(std.mem.bytesAsSlice(f16, v_raw[0 .. @as(usize, max_m) * max_kv_dim * 2])),
            .attn_out_host = @alignCast(std.mem.bytesAsSlice(f16, attn_out_raw[0 .. @as(usize, max_m) * max_attn_out * 2])),
            .proj_out_host = @alignCast(std.mem.bytesAsSlice(f16, proj_out_raw[0 .. @as(usize, max_m) * h * 2])),
            .dense_gate_host = @alignCast(std.mem.bytesAsSlice(f16, dgate_raw[0 .. @as(usize, max_m) * max_inter * 2])),
            .dense_up_host = @alignCast(std.mem.bytesAsSlice(f16, dup_raw[0 .. @as(usize, max_m) * max_inter * 2])),
            .dense_act_host = @alignCast(std.mem.bytesAsSlice(f16, dact_raw[0 .. @as(usize, max_m) * max_inter * 2])),
            .accepted_top_host = @alignCast(std.mem.bytesAsSlice(f16, acc_top_raw[0 .. feat_cap * h * 2])),
            .accepted_aux_host = @alignCast(std.mem.bytesAsSlice(f16, acc_aux_raw[0 .. feat_cap * 3 * h * 2])),
            .token_ids_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(tok_raw.ptr) },
            .positions_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(pos_raw.ptr) },
            .pos3d_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(pos3d_raw.ptr) },
            .ancestor_masks_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(msk_raw.ptr) },
            .out_token_id_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(out_tok_raw.ptr) },
            .logits_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(logits_raw.ptr) },
            .topk_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(topk_raw.ptr) },
            .ple_f16_base_off = scratch_arena.offsetOf(ple_f16_raw.ptr),
            .x_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_raw.ptr) },
            .x_init_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_init_raw.ptr) },
            .x_norm_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(x_norm_raw.ptr) },
            .q_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(q_raw.ptr) },
            .k_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(k_raw.ptr) },
            .v_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(v_raw.ptr) },
            .q_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(q_conv_raw.ptr) },
            .k_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(k_conv_raw.ptr) },
            .v_conv_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(v_conv_raw.ptr) },
            .gdn_z_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gdn_z_raw.ptr) },
            .gdn_g_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gdn_g_raw.ptr) },
            .gdn_beta_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gdn_beta_raw.ptr) },
            .gr_g_read_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_g_read_raw.ptr) },
            .gr_g_write_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_g_write_raw.ptr) },
            .gr_decay_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gr_decay_raw.ptr) },
            .attn_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(attn_out_raw.ptr) },
            .proj_out_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(proj_out_raw.ptr) },
            .dense_gate_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dgate_raw.ptr) },
            .dense_up_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dup_raw.ptr) },
            .dense_act_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(dact_raw.ptr) },
            .gqa_scratch_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(gqa_scratch_raw.ptr) },
            .moe_ws = moe_ws,
            .layer_sched_hdr_raw = layer_sched_hdr_raw,
            .layer_sched_hdr_off = scratch_arena.offsetOf(layer_sched_hdr_raw.ptr),
            .accepted_top_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(acc_top_raw.ptr) },
            .accepted_aux_bind = .{ .buffer = scratch_buf, .offset = scratch_arena.offsetOf(acc_aux_raw.ptr) },
            .aux_tree_base_off = scratch_arena.offsetOf(aux_tree_raw.ptr),
            .single_expert_bytes = single_expert_bytes,
            .total_layers_non_expert_bytes = total_layers_non_expert_bytes,
            .total_layers_static_bytes = total_layers_static_bytes,
            .global_head_bytes = global_head_bytes,
            .embed_row_alg_bytes = embed_row_alg_bytes,
            .last_logits_row = 0,
            .last_verified_nodes = 0,
            .last_verified_tokens = @splat(0),
            .last_verified_masks = @splat(0),
            .loop_stats = .{},
            .kv_retire_after_chunks = retire_after,
            .retire_chunk_tokens = retire_chunk_tokens,
            .active_kv_chunks = active_kv_chunks,
            .num_retired_chunks = 0,
            .max_retired_per_layer = max_retired_per_layer,
            .kv_retire_events = 0,
            .active_kv_wired_bytes = active_kv_wired_bytes,
            .retired_kv_wired_bytes = retired_kv_wired_bytes,
            .retired_kv_pool = retired_kv_pool,
            .active_tile_phys_ring = active_tile_phys_ring,
            .mrope_next_pos = 0,
            .gpu_prefill_mode = options.gpu_prefill_mode,
            .wide_prefill_max_tokens = wide_prefill_max_tokens,
            .supports_wide_prefill = supports_wide_prefill,
            .compute_all_logits = options.compute_all_logits,
            .enable_prompt_tail_pruning = options.enable_prompt_tail_pruning,
            .canary_result = canary_result,
            .live_admission = live_admission,
            .aliased_scratch_saved_bytes = aliased_scratch_saved_bytes,
            .wp_token_ids_host = wp_token_ids_host,
            .wp_pos3d_host = wp_pos3d_host,
            .wp_ple_f16_host = wp_ple_f16_host,
            .wp_token_ids_bind = wp_token_ids_bind,
            .wp_pos3d_bind = wp_pos3d_bind,
            .wp_ple_f16_base_off = wp_ple_f16_base_off,
            .wp_gr_arena = wp_gr_arena,
            .wp_gr_g_read_bind = wp_gr_g_read_bind,
            .wp_gr_g_write_bind = wp_gr_g_write_bind,
            .wp_gr_decay_bind = wp_gr_decay_bind,
            .wp_x_bind = wp_x_bind,
            .wp_x_init_bind = wp_x_init_bind,
            .wp_x_norm_bind = wp_x_norm_bind,
            .wp_q_bind = wp_q_bind,
            .wp_k_bind = wp_k_bind,
            .wp_v_bind = wp_v_bind,
            .wp_q_conv_bind = wp_q_conv_bind,
            .wp_k_conv_bind = wp_k_conv_bind,
            .wp_v_conv_bind = wp_v_conv_bind,
            .wp_gdn_z_bind = wp_gdn_z_bind,
            .wp_gdn_g_bind = wp_gdn_g_bind,
            .wp_gdn_beta_bind = wp_gdn_beta_bind,
            .wp_attn_out_bind = wp_attn_out_bind,
            .wp_proj_out_bind = wp_proj_out_bind,
            .wp_dense_gate_bind = wp_dense_gate_bind,
            .wp_dense_up_bind = wp_dense_up_bind,
            .wp_dense_act_bind = wp_dense_act_bind,
            .wp_moe_ws = wp_moe_ws,
        };
    }
};

// ===========================================================================
// Section 5b: M6.5 C14 + R8 Multi-Slot Continuous Batching & GDN PrefixCache
// ===========================================================================

pub const GraphBatchConfig = struct {
    max_slots: u32 = 8,
    prefix_cache_chunks: u32 = 0,
    kv_pool_chunk_groups: u32 = 0,
    budget_override: ?budget_mod.Budget = null,
    /// M6.8 (`G15`): GPU prefill mode (`null` inherits `model.gpu_prefill_mode`, default `.wide`).
    gpu_prefill_mode: ?qwen3.GpuPrefillMode = null,
    /// M6.8 (`P1-B`): Reject startup if required wired bytes exceed live zero-swap RAM minus margin.
    strict_live_ram: bool = false,
    /// M6.8 (`P1-B`): Safety margin held back from live zero-swap reclaimable RAM (default 512 MiB).
    live_safety_margin_bytes: u64 = budget_mod.DEFAULT_LIVE_SAFETY_MARGIN_BYTES,
    /// Optional synthetic live zero-swap RAM override for deterministic tests.
    live_reclaimable_override: ?u64 = null,
};

pub const GraphBatchStats = struct {
    max_slots: u32,
    pool_chunk_groups: u32,
    pool_free_groups: u32,
    prefix_cache_capacity: u32,
    prefix_cache_live_entries: u32,
    prefix_cache_pinned_entries: u32,
    gdn_snapshots_active: u32,
    prefix_hits: u64,
    prefix_hit_tokens: u64,
    prefix_cow_copies: u64,
    boundary_hits: u64,
    evict_first_evictions: u64,
    gdn_snapshot_clones: u64,
    decode_batch_steps: u64,
    decode_batch_tokens: u64,
    total_wired_bytes: u64,
    live_admission: budget_mod.LiveAdmissionPlan,
};

pub const GraphBatchError = error{
    InvalidConfig,
    TreeNotEnabled,
    BudgetExceeded,
    OutOfMemory,
} || sysinfo.Error || vmstat.Error || budget_mod.PlanError || arena_mod.WiredArena.InitError ||
    arena_mod.WiredArena.AllocError || metal.WrapError || metal.Error ||
    hybrid.ArenaError || hybrid.GdnSnapshotPoolError || prefix_cache_mod.Error;

const GRAPH_NONE_GROUP: u32 = std.math.maxInt(u32);

const GraphBatchSlotState = struct {
    in_use: bool = false,
    seq_len: u32 = 0,
    matched_prefix_tokens: u32 = 0,
    prefix_cached_full_chunks: u32 = 0,
    last_cached_parent_id: u32 = prefix_cache_mod.NONE_ID,
    pinned_full_count: u32 = 0,
    chunk_group_ids: []u32 = &.{},
    chunk_is_shared: []bool = &.{},
    pinned_full_ids: []u32 = &.{},
    prompt_tokens: []u32 = &.{},
};

/// Preallocated zero-allocation continuous-batching (`C14`) & shared `PrefixCache` +
/// `GdnSnapshotPool` (`R8`) runtime attached to a loaded hybrid `GraphModel`
/// (`GDN + QSA/GQA + MoE/Dense + 4-branch GR`).
pub const GraphBatchRuntime = struct {
    model: *GraphModel,
    max_slots: u32,
    max_chunks: u32,
    kv_chunk_tokens: u32,
    num_kv_layers: u32,
    num_gdn_layers: u32,
    num_qsa_layers: u32,
    num_gqa_layers: u32,
    pool_chunk_groups: u32,
    total_wired_bytes: u64,

    /// Maps transformer layer index `l` -> compact KV layer index (`0..num_kv_layers-1` or `maxInt`).
    layer_to_kv_idx: []u32,
    kv_idx_to_layer: []u32,

    batch_arena: arena_mod.WiredArena,
    batch_buf: metal.Buffer,

    // Multi-slot GDN recurrent state & conv1d history pools (`[num_gdn_layers, max_slots, ...]`)
    gdn_pool_base_off: u64,
    gdn_slot_stride_bytes: u64,
    gdn_layer_stride_bytes: u64,
    gdn_conv_pool_base_off: u64,
    gdn_conv_slot_stride_bytes: u64,
    gdn_conv_layer_stride_bytes: u64,
    gdn_conv_f16_per_layer: u32,
    gdn_slot_ids_off: u64,
    gdn_slot_ids_host: []u32,
    gdn_snapshot_pool: ?hybrid.GdnSnapshotPool,

    // Shared IOSurface KV chunk pool (`[pool_chunk_groups * num_kv_layers * 2]`)
    pool_tiles: []metal.KvTile,
    pool_free_stack: []u32,
    pool_free_top: u32,

    // Per-KV-layer `[max_slots * max_chunks]` active KvTilePair tables
    kv_tiles_base_off: u64,
    kv_tiles_layer_stride_bytes: u64,
    kv_tiles_host: []attention.KvTilePair,

    // Per-GQA-layer batch write slots & slot descs + shared batch GQA scratch
    write_slots_base_off: u64,
    write_slots_layer_stride_bytes: u64,
    write_slots_host: []attention.BatchKvWriteSlot,
    gqa_descs_base_off: u64,
    gqa_descs_layer_stride_bytes: u64,
    gqa_descs_host: []attention.BatchGqaSlotDesc,
    gqa_batch_scratch_bind: metal.BufferBinding,

    // Per-QSA-layer per-slot `block_keys` (`[num_qsa_layers, max_slots, kv_heads * max_blocks * d]`)
    qsa_block_keys_base_off: u64,
    qsa_block_keys_slot_stride_bytes: u64,
    qsa_block_keys_layer_stride_bytes: u64,
    qsa_max_blocks: u32,

    prefix_cache: ?prefix_cache_mod.PrefixCache,
    slots: []GraphBatchSlotState,
    scratch_handles: []prefix_cache_mod.KvBlockHandle,
    scratch_evicted_handles: []prefix_cache_mod.KvBlockHandle,
    p_res: []f32,

    // Telemetry & M6.8 state
    gpu_prefill_mode: qwen3.GpuPrefillMode,
    live_admission: budget_mod.LiveAdmissionPlan,
    prefix_hits: u64 = 0,
    prefix_hit_tokens: u64 = 0,
    prefix_cow_copies: u64 = 0,
    gdn_snapshot_clones: u64 = 0,
    decode_batch_steps: u64 = 0,
    decode_batch_tokens: u64 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *GraphModel,
        cfg: GraphBatchConfig,
    ) GraphBatchError!GraphBatchRuntime {
        if (cfg.max_slots == 0 or cfg.max_slots > batch_sched.MAX_BATCH_SLOTS) {
            return error.InvalidConfig;
        }
        const needed_tree: u32 = @max(cfg.max_slots, 2);
        if (model.max_tree_nodes < needed_tree) {
            return error.TreeNotEnabled;
        }
        const eff_gpu_prefill_mode: qwen3.GpuPrefillMode = cfg.gpu_prefill_mode orelse model.gpu_prefill_mode;

        const n_layers = model.layers.len;
        const layer_to_kv_idx = try allocator.alloc(u32, n_layers);
        errdefer allocator.free(layer_to_kv_idx);
        @memset(layer_to_kv_idx, std.math.maxInt(u32));

        var num_kv_layers: u32 = 0;
        var num_gdn_layers: u32 = 0;
        var num_qsa_layers: u32 = 0;
        var num_gqa_layers: u32 = 0;
        var qsa_chunk_tokens: u32 = 0;
        var qsa_max_blocks: u32 = 0;
        var max_conv_k: u32 = 0;
        for (model.layers, 0..) |layer, l| {
            switch (layer.attn) {
                .gdn => |g| {
                    num_gdn_layers += 1;
                    if (g.conv1d_w != null and g.conv_k > max_conv_k) max_conv_k = g.conv_k;
                },
                .qsa => |qsa| {
                    layer_to_kv_idx[l] = num_kv_layers;
                    num_kv_layers += 1;
                    num_qsa_layers += 1;
                    qsa_chunk_tokens = qsa.kv_ring.chunk_tokens;
                    qsa_max_blocks = @max(qsa_max_blocks, qsa.max_blocks);
                },
                .gqa => {
                    layer_to_kv_idx[l] = num_kv_layers;
                    num_kv_layers += 1;
                    num_gqa_layers += 1;
                },
            }
        }
        const kv_idx_to_layer = try allocator.alloc(u32, num_kv_layers);
        errdefer allocator.free(kv_idx_to_layer);
        for (layer_to_kv_idx, 0..) |ki, l| {
            if (ki != std.math.maxInt(u32)) kv_idx_to_layer[ki] = @intCast(l);
        }

        const kv_chunk_tokens: u32 = if (num_qsa_layers > 0) qsa_chunk_tokens else model.chunk_tokens;
        const max_chunks: u32 = (model.max_seq_len + kv_chunk_tokens - 1) / kv_chunk_tokens;

        const facts = try sysinfo.Facts.read();
        const dev_info = try model.device.info();
        const wired_budget = cfg.budget_override orelse budget_mod.Budget.fromHost(facts, dev_info, .{});

        // Compute multi-slot GDN state & conv1d pool strides.
        const gdn_state_elems: usize = if (num_gdn_layers > 0)
            @as(usize, model.config.linear_num_value_heads) *
                model.config.linear_key_head_dim * model.config.linear_value_head_dim
        else
            0;
        const gdn_slot_stride_bytes: usize = if (num_gdn_layers > 0) amf.align64(gdn_state_elems * @sizeOf(f32)) else 0;
        const gdn_layer_stride_bytes: usize = gdn_slot_stride_bytes * cfg.max_slots;
        const gdn_pool_bytes: usize = @as(usize, num_gdn_layers) * gdn_layer_stride_bytes;

        const conv_hist: u32 = if (max_conv_k > 1) max_conv_k - 1 else 0;
        const conv_dim: u32 = if (num_gdn_layers > 0) 2 * model.config.gdnQkDim() + model.config.gdnVDim() else 0;
        const gdn_conv_f16_per_layer: u32 = conv_hist * conv_dim;
        const gdn_conv_slot_stride_bytes: usize = if (gdn_conv_f16_per_layer > 0)
            amf.align64(@as(usize, gdn_conv_f16_per_layer) * @sizeOf(f16))
        else
            0;
        const gdn_conv_layer_stride_bytes: usize = gdn_conv_slot_stride_bytes * cfg.max_slots;
        const gdn_conv_pool_bytes: usize = @as(usize, num_gdn_layers) * gdn_conv_layer_stride_bytes;

        const kv_tiles_layer_stride_bytes: usize = if (num_kv_layers > 0)
            amf.align64(@as(usize, cfg.max_slots) * @as(usize, max_chunks) * @sizeOf(attention.KvTilePair))
        else
            0;
        const write_slots_layer_stride_bytes: usize = if (num_gqa_layers > 0)
            amf.align64(@as(usize, cfg.max_slots) * @sizeOf(attention.BatchKvWriteSlot))
        else
            0;
        const gqa_descs_layer_stride_bytes: usize = if (num_gqa_layers > 0)
            amf.align64(@as(usize, cfg.max_slots) * @sizeOf(attention.BatchGqaSlotDesc))
        else
            0;
        const gqa_batch_layout = attention.gqaBatchScratchLayout(
            cfg.max_slots,
            model.config.num_attention_heads,
            model.config.head_dim,
            attention.MAX_SPLITS,
        );
        const gqa_batch_bytes: usize = if (num_gqa_layers > 0) amf.align64(gqa_batch_layout.total_bytes) else 0;
        const qsa_block_keys_slot_stride_bytes: usize = if (num_qsa_layers > 0)
            amf.align64(@as(usize, model.config.num_key_value_heads) * qsa_max_blocks * model.config.head_dim * @sizeOf(f16))
        else
            0;
        const qsa_block_keys_layer_stride_bytes: usize = qsa_block_keys_slot_stride_bytes * cfg.max_slots;
        const qsa_block_keys_total_bytes: usize = @as(usize, num_qsa_layers) * qsa_block_keys_layer_stride_bytes;
        const prompt_hist_bytes: usize = amf.align64(@as(usize, cfg.max_slots) * model.max_seq_len * @sizeOf(u32));

        // P1-B (`R9`, ADR-0020): Evaluate required runtime bytes vs live non-anonymous Mach RAM
        // and softly auto-scale optional `PrefixCache` + `GdnSnapshotPool` entries if tighter.
        const kv_tile_rows: u32 = model.config.num_key_value_heads * kv_chunk_tokens;
        const bytes_per_group: u64 = if (num_kv_layers > 0) blk: {
            const tile_bytes: u64 = try page.alignUp(@as(usize, kv_tile_rows) * model.config.head_dim * @sizeOf(f16));
            break :blk @as(u64, num_kv_layers) * 2 * tile_bytes;
        } else 0;
        const base_arena_raw: usize =
            gdn_pool_bytes +
            gdn_conv_pool_bytes +
            amf.align64(@as(usize, cfg.max_slots) * @sizeOf(u32)) +
            @as(usize, num_kv_layers) * kv_tiles_layer_stride_bytes +
            @as(usize, num_gqa_layers) * write_slots_layer_stride_bytes +
            @as(usize, num_gqa_layers) * gqa_descs_layer_stride_bytes +
            gqa_batch_bytes +
            qsa_block_keys_total_bytes +
            prompt_hist_bytes +
            16 * page.APPLE_PAGE_SIZE;
        const base_arena_bytes: u64 = try page.alignUp(base_arena_raw);
        const min_req_groups: u32 = if (num_kv_layers > 0)
            (if (cfg.kv_pool_chunk_groups > 0) cfg.kv_pool_chunk_groups else cfg.max_slots)
        else
            0;
        const required_runtime_bytes: u64 = base_arena_bytes + @as(u64, min_req_groups) * bytes_per_group;

        const req_use_gdn_snap = (num_gdn_layers > 0 and cfg.prefix_cache_chunks > 0);
        const req_gdn_snap_bytes: u64 = if (req_use_gdn_snap)
            try hybrid.GdnSnapshotPool.requiredArenaBytes(.{
                .state = .{
                    .num_gdn_layers = num_gdn_layers,
                    .num_v_heads = model.config.linear_num_value_heads,
                    .head_k_dim = model.config.linear_key_head_dim,
                    .head_v_dim = model.config.linear_value_head_dim,
                },
                .conv_f16_per_layer = gdn_conv_f16_per_layer,
                .max_snapshots = cfg.prefix_cache_chunks,
            })
        else
            0;
        const req_pc_meta_bytes: u64 = if (cfg.prefix_cache_chunks > 0)
            try prefix_cache_mod.PrefixCache.requiredBytes(.{
                .max_entries = cfg.prefix_cache_chunks,
                .block_tokens = kv_chunk_tokens,
                .num_kv_layers = @max(num_kv_layers, 1),
            })
        else
            0;
        const optional_pc_bytes: u64 = if (cfg.prefix_cache_chunks > 0)
            req_gdn_snap_bytes + req_pc_meta_bytes + @as(u64, cfg.prefix_cache_chunks) * bytes_per_group
        else
            0;

        if (model.totalWiredBytes() > wired_budget.limit_bytes) return error.BudgetExceeded;
        const rem_limit_raw: u64 = wired_budget.limit_bytes - model.totalWiredBytes();
        const rem_budget: budget_mod.Budget = .{
            .limit_bytes = rem_limit_raw - (rem_limit_raw % page.APPLE_PAGE_SIZE),
            .binding_limit = wired_budget.binding_limit,
            .raw_limit_bytes = rem_limit_raw,
        };
        const live_reclaimable_bytes: u64 = cfg.live_reclaimable_override orelse
            (try vmstat.Snapshot.capture()).liveZeroSwapReclaimableBytes();
        const live_admission = try rem_budget.planWithLiveRam(
            required_runtime_bytes,
            optional_pc_bytes,
            live_reclaimable_bytes,
            .{
                .live_safety_margin_bytes = cfg.live_safety_margin_bytes,
                .strict_live_ram = cfg.strict_live_ram,
            },
        );
        const eff_prefix_cache_chunks: u32 = if (cfg.prefix_cache_chunks > 0 and live_admission.cache_scaled_down) blk: {
            const unit_bytes: usize = @intCast(@max(@as(u64, 1), optional_pc_bytes / cfg.prefix_cache_chunks));
            break :blk @intCast(budget_mod.scaleOptionalCount(
                cfg.prefix_cache_chunks,
                unit_bytes,
                live_admission.granted_optional_cache_bytes,
                0,
            ));
        } else cfg.prefix_cache_chunks;

        const use_gdn_snap = (num_gdn_layers > 0 and eff_prefix_cache_chunks > 0);
        const gdn_snap_spec: hybrid.GdnSnapshotPoolSpec = .{
            .state = .{
                .num_gdn_layers = if (num_gdn_layers > 0) num_gdn_layers else 1,
                .num_v_heads = if (num_gdn_layers > 0) model.config.linear_num_value_heads else 1,
                .head_k_dim = if (num_gdn_layers > 0) model.config.linear_key_head_dim else 64,
                .head_v_dim = if (num_gdn_layers > 0) model.config.linear_value_head_dim else 64,
            },
            .conv_f16_per_layer = gdn_conv_f16_per_layer,
            .max_snapshots = if (use_gdn_snap) eff_prefix_cache_chunks else 1,
        };
        const gdn_snap_bytes: usize = if (use_gdn_snap)
            try hybrid.GdnSnapshotPool.requiredArenaBytes(gdn_snap_spec)
        else
            0;

        const pc_spec: prefix_cache_mod.PrefixCacheSpec = .{
            .max_entries = if (eff_prefix_cache_chunks > 0) eff_prefix_cache_chunks else 1,
            .block_tokens = kv_chunk_tokens,
            .num_kv_layers = @max(num_kv_layers, 1),
        };
        const pc_bytes: usize = if (eff_prefix_cache_chunks > 0)
            try prefix_cache_mod.PrefixCache.requiredBytes(pc_spec)
        else
            0;

        const arena_raw: usize =
            gdn_snap_bytes +
            pc_bytes +
            gdn_pool_bytes +
            gdn_conv_pool_bytes +
            amf.align64(@as(usize, cfg.max_slots) * @sizeOf(u32)) + // gdn_slot_ids
            @as(usize, num_kv_layers) * kv_tiles_layer_stride_bytes +
            @as(usize, num_gqa_layers) * write_slots_layer_stride_bytes +
            @as(usize, num_gqa_layers) * gqa_descs_layer_stride_bytes +
            gqa_batch_bytes +
            qsa_block_keys_total_bytes +
            prompt_hist_bytes +
            16 * page.APPLE_PAGE_SIZE;
        const batch_arena_bytes = try page.alignUp(arena_raw);

        // Size shared IOSurface chunk group pool when KV layers exist.
        var pool_chunk_groups: u32 = 0;
        var pool_kv_bytes: u64 = 0;
        if (num_kv_layers > 0) {
            const desired_groups: u32 = if (cfg.kv_pool_chunk_groups > 0)
                cfg.kv_pool_chunk_groups
            else
                cfg.max_slots * max_chunks + eff_prefix_cache_chunks;
            const base_used: u64 = model.totalWiredBytes() + batch_arena_bytes;
            if (base_used + @as(u64, cfg.max_slots) * bytes_per_group > wired_budget.limit_bytes) {
                return error.BudgetExceeded;
            }
            const max_total_iosurfaces: u64 = 4608;
            var model_iosurfaces: u64 = @as(u64, model.gqa_kv_tiles.len);
            for (model.layers) |layer| {
                if (layer.attn == .qsa) {
                    model_iosurfaces += @as(u64, layer.attn.qsa.kv_ring.k_tiles.len) + layer.attn.qsa.kv_ring.v_tiles.len;
                }
            }
            const surf_per_group: u64 = @as(u64, num_kv_layers) * 2;
            const iosurface_cap_groups: u64 = if (max_total_iosurfaces > model_iosurfaces and surf_per_group > 0)
                (max_total_iosurfaces - model_iosurfaces) / surf_per_group
            else
                cfg.max_slots;
            const ram_cap_groups: u64 = (wired_budget.limit_bytes - base_used) / bytes_per_group;
            const headroom_groups: u32 = @intCast(@min(
                @as(u64, desired_groups),
                @min(ram_cap_groups, if (cfg.kv_pool_chunk_groups > 0) @as(u64, desired_groups) else iosurface_cap_groups),
            ));
            pool_chunk_groups = if (cfg.kv_pool_chunk_groups > 0) blk: {
                if (cfg.kv_pool_chunk_groups > headroom_groups) return error.BudgetExceeded;
                break :blk cfg.kv_pool_chunk_groups;
            } else @max(cfg.max_slots, headroom_groups);
            pool_kv_bytes = @as(u64, pool_chunk_groups) * bytes_per_group;
        }

        const total_wired_bytes: u64 = batch_arena_bytes + pool_kv_bytes;
        try wired_budget.checkPlan(model.totalWiredBytes() + total_wired_bytes);

        var batch_arena = try arena_mod.WiredArena.init(batch_arena_bytes, wired_budget);
        errdefer batch_arena.deinit();
        const batch_buf = try metal.Buffer.wrap(model.device, batch_arena.bytes());
        errdefer batch_buf.close();

        const gdn_snapshot_pool: ?hybrid.GdnSnapshotPool = if (use_gdn_snap)
            try hybrid.GdnSnapshotPool.init(&batch_arena, gdn_snap_spec)
        else
            null;
        const prefix_cache: ?prefix_cache_mod.PrefixCache = if (eff_prefix_cache_chunks > 0)
            try prefix_cache_mod.PrefixCache.init(&batch_arena, pc_spec)
        else
            null;

        var gdn_pool_base_off: u64 = 0;
        if (gdn_pool_bytes > 0) {
            const raw = try batch_arena.allocAligned(gdn_pool_bytes, 64);
            @memset(raw, 0);
            gdn_pool_base_off = @intCast(batch_arena.offsetOf(raw.ptr));
        }
        var gdn_conv_pool_base_off: u64 = 0;
        if (gdn_conv_pool_bytes > 0) {
            const raw = try batch_arena.allocAligned(gdn_conv_pool_bytes, 64);
            @memset(raw, 0);
            gdn_conv_pool_base_off = @intCast(batch_arena.offsetOf(raw.ptr));
        }

        const slot_ids_raw = try batch_arena.allocAligned(@as(usize, cfg.max_slots) * @sizeOf(u32), 64);
        @memset(slot_ids_raw, 0);
        const gdn_slot_ids_off: u64 = @intCast(batch_arena.offsetOf(slot_ids_raw.ptr));
        const gdn_slot_ids_host: []u32 = @alignCast(std.mem.bytesAsSlice(u32, slot_ids_raw[0 .. @as(usize, cfg.max_slots) * @sizeOf(u32)]));

        var kv_tiles_base_off: u64 = 0;
        var kv_tiles_host: []attention.KvTilePair = &.{};
        if (num_kv_layers > 0) {
            const raw = try batch_arena.allocAligned(@as(usize, num_kv_layers) * kv_tiles_layer_stride_bytes, 64);
            @memset(raw, 0);
            kv_tiles_base_off = @intCast(batch_arena.offsetOf(raw.ptr));
            kv_tiles_host = @alignCast(std.mem.bytesAsSlice(attention.KvTilePair, raw));
        }

        var write_slots_base_off: u64 = 0;
        var write_slots_host: []attention.BatchKvWriteSlot = &.{};
        var gqa_descs_base_off: u64 = 0;
        var gqa_descs_host: []attention.BatchGqaSlotDesc = &.{};
        var gqa_batch_scratch_bind: metal.BufferBinding = .{ .buffer = batch_buf, .offset = 0 };
        if (num_gqa_layers > 0) {
            const ws_raw = try batch_arena.allocAligned(@as(usize, num_gqa_layers) * write_slots_layer_stride_bytes, 64);
            @memset(ws_raw, 0);
            write_slots_base_off = @intCast(batch_arena.offsetOf(ws_raw.ptr));
            write_slots_host = @alignCast(std.mem.bytesAsSlice(attention.BatchKvWriteSlot, ws_raw));

            const gd_raw = try batch_arena.allocAligned(@as(usize, num_gqa_layers) * gqa_descs_layer_stride_bytes, 64);
            @memset(gd_raw, 0);
            gqa_descs_base_off = @intCast(batch_arena.offsetOf(gd_raw.ptr));
            gqa_descs_host = @alignCast(std.mem.bytesAsSlice(attention.BatchGqaSlotDesc, gd_raw));

            const sc_raw = try batch_arena.allocAligned(gqa_batch_layout.total_bytes, 64);
            gqa_batch_scratch_bind = .{ .buffer = batch_buf, .offset = @intCast(batch_arena.offsetOf(sc_raw.ptr)) };
        }

        var qsa_block_keys_base_off: u64 = 0;
        if (qsa_block_keys_total_bytes > 0) {
            const raw = try batch_arena.allocAligned(qsa_block_keys_total_bytes, 64);
            @memset(raw, 0);
            qsa_block_keys_base_off = @intCast(batch_arena.offsetOf(raw.ptr));
        }

        const prompt_hist_raw = try batch_arena.allocAligned(prompt_hist_bytes, 64);
        const prompt_hist_all: []u32 = @alignCast(std.mem.bytesAsSlice(u32, prompt_hist_raw));

        // Allocate shared IOSurface KV tiles.
        const total_pool_tiles: usize = @as(usize, pool_chunk_groups) * num_kv_layers * 2;
        const pool_tiles = try allocator.alloc(metal.KvTile, total_pool_tiles);
        var tiles_built: usize = 0;
        errdefer {
            for (0..tiles_built) |ti| pool_tiles[ti].close();
            allocator.free(pool_tiles);
        }
        for (0..total_pool_tiles) |ti| {
            pool_tiles[ti] = try metal.KvTile.init(model.device, model.config.head_dim, kv_tile_rows, .{});
            tiles_built += 1;
        }

        try model.compute.makeResident(&.{batch_buf});
        if (pool_tiles.len > 0) try model.compute.makeTilesResident(pool_tiles);

        const pool_free_stack = try allocator.alloc(u32, pool_chunk_groups);
        errdefer allocator.free(pool_free_stack);
        for (0..pool_chunk_groups) |i| {
            pool_free_stack[i] = pool_chunk_groups - 1 - @as(u32, @intCast(i));
        }

        const slots = try allocator.alloc(GraphBatchSlotState, cfg.max_slots);
        var slots_built: usize = 0;
        errdefer {
            for (0..slots_built) |si| {
                allocator.free(slots[si].chunk_group_ids);
                allocator.free(slots[si].chunk_is_shared);
                allocator.free(slots[si].pinned_full_ids);
            }
            allocator.free(slots);
        }
        for (0..cfg.max_slots) |si| {
            const gids = try allocator.alloc(u32, max_chunks);
            errdefer allocator.free(gids);
            @memset(gids, GRAPH_NONE_GROUP);

            const shared = try allocator.alloc(bool, max_chunks);
            errdefer allocator.free(shared);
            @memset(shared, false);

            const pids = try allocator.alloc(u32, max_chunks);
            errdefer allocator.free(pids);
            @memset(pids, prefix_cache_mod.NONE_ID);

            const p_off = si * @as(usize, model.max_seq_len);
            slots[si] = .{
                .chunk_group_ids = gids,
                .chunk_is_shared = shared,
                .pinned_full_ids = pids,
                .prompt_tokens = prompt_hist_all[p_off .. p_off + model.max_seq_len],
            };
            slots_built += 1;
        }

        const handle_len: usize = @max(num_kv_layers, 1);
        const scratch_handles = try allocator.alloc(prefix_cache_mod.KvBlockHandle, handle_len);
        errdefer allocator.free(scratch_handles);
        const scratch_evicted_handles = try allocator.alloc(prefix_cache_mod.KvBlockHandle, handle_len);
        errdefer allocator.free(scratch_evicted_handles);
        const p_res = try allocator.alloc(f32, model.config.vocab_size);
        errdefer allocator.free(p_res);

        return .{
            .model = model,
            .max_slots = cfg.max_slots,
            .max_chunks = max_chunks,
            .kv_chunk_tokens = kv_chunk_tokens,
            .num_kv_layers = num_kv_layers,
            .num_gdn_layers = num_gdn_layers,
            .num_qsa_layers = num_qsa_layers,
            .num_gqa_layers = num_gqa_layers,
            .pool_chunk_groups = pool_chunk_groups,
            .total_wired_bytes = total_wired_bytes,
            .layer_to_kv_idx = layer_to_kv_idx,
            .kv_idx_to_layer = kv_idx_to_layer,
            .batch_arena = batch_arena,
            .batch_buf = batch_buf,
            .gdn_pool_base_off = gdn_pool_base_off,
            .gdn_slot_stride_bytes = gdn_slot_stride_bytes,
            .gdn_layer_stride_bytes = gdn_layer_stride_bytes,
            .gdn_conv_pool_base_off = gdn_conv_pool_base_off,
            .gdn_conv_slot_stride_bytes = gdn_conv_slot_stride_bytes,
            .gdn_conv_layer_stride_bytes = gdn_conv_layer_stride_bytes,
            .gdn_conv_f16_per_layer = gdn_conv_f16_per_layer,
            .gdn_slot_ids_off = gdn_slot_ids_off,
            .gdn_slot_ids_host = gdn_slot_ids_host,
            .gdn_snapshot_pool = gdn_snapshot_pool,
            .pool_tiles = pool_tiles,
            .pool_free_stack = pool_free_stack,
            .pool_free_top = pool_chunk_groups,
            .kv_tiles_base_off = kv_tiles_base_off,
            .kv_tiles_layer_stride_bytes = kv_tiles_layer_stride_bytes,
            .kv_tiles_host = kv_tiles_host,
            .write_slots_base_off = write_slots_base_off,
            .write_slots_layer_stride_bytes = write_slots_layer_stride_bytes,
            .write_slots_host = write_slots_host,
            .gqa_descs_base_off = gqa_descs_base_off,
            .gqa_descs_layer_stride_bytes = gqa_descs_layer_stride_bytes,
            .gqa_descs_host = gqa_descs_host,
            .gqa_batch_scratch_bind = gqa_batch_scratch_bind,
            .qsa_block_keys_base_off = qsa_block_keys_base_off,
            .qsa_block_keys_slot_stride_bytes = qsa_block_keys_slot_stride_bytes,
            .qsa_block_keys_layer_stride_bytes = qsa_block_keys_layer_stride_bytes,
            .qsa_max_blocks = qsa_max_blocks,
            .prefix_cache = prefix_cache,
            .slots = slots,
            .scratch_handles = scratch_handles,
            .scratch_evicted_handles = scratch_evicted_handles,
            .p_res = p_res,
            .gpu_prefill_mode = eff_gpu_prefill_mode,
            .live_admission = live_admission,
        };
    }

    pub fn deinit(self: *GraphBatchRuntime, allocator: std.mem.Allocator) void {
        allocator.free(self.p_res);
        allocator.free(self.scratch_evicted_handles);
        allocator.free(self.scratch_handles);
        for (self.slots) |slot| {
            allocator.free(slot.chunk_group_ids);
            allocator.free(slot.chunk_is_shared);
            allocator.free(slot.pinned_full_ids);
        }
        allocator.free(self.slots);
        allocator.free(self.pool_free_stack);
        if (self.pool_tiles.len > 0) {
            self.model.compute.removeTilesResident(self.pool_tiles) catch |err| {
                std.debug.panic("GraphBatchRuntime removeTilesResident failed: {}", .{err});
            };
            for (self.pool_tiles) |tile| tile.close();
        }
        allocator.free(self.pool_tiles);
        self.model.compute.removeResident(&.{self.batch_buf}) catch |err| {
            std.debug.panic("GraphBatchRuntime removeResident failed: {}", .{err});
        };
        self.batch_buf.close();
        self.batch_arena.deinit();
        allocator.free(self.kv_idx_to_layer);
        allocator.free(self.layer_to_kv_idx);
    }

    pub fn totalWiredBytes(self: *const GraphBatchRuntime) u64 {
        return self.total_wired_bytes;
    }

    pub fn stats(self: *const GraphBatchRuntime) GraphBatchStats {
        const pc_cap: u32 = if (self.prefix_cache) |pc| pc.spec.max_entries else 0;
        const pc_live: u32 = if (self.prefix_cache) |pc| pc.liveEntries() else 0;
        const pc_pinned: u32 = if (self.prefix_cache) |pc| pc.pinnedEntries() else 0;
        const b_hits: u64 = if (self.prefix_cache) |pc| pc.boundary_hits else 0;
        const ef_evicts: u64 = if (self.prefix_cache) |pc| pc.evict_first_evictions else 0;
        const gdn_act: u32 = if (self.gdn_snapshot_pool) |sp| sp.activeCount() else 0;
        return .{
            .max_slots = self.max_slots,
            .pool_chunk_groups = self.pool_chunk_groups,
            .pool_free_groups = self.pool_free_top,
            .prefix_cache_capacity = pc_cap,
            .prefix_cache_live_entries = pc_live,
            .prefix_cache_pinned_entries = pc_pinned,
            .gdn_snapshots_active = gdn_act,
            .prefix_hits = self.prefix_hits,
            .prefix_hit_tokens = self.prefix_hit_tokens,
            .prefix_cow_copies = self.prefix_cow_copies,
            .boundary_hits = b_hits,
            .evict_first_evictions = ef_evicts,
            .gdn_snapshot_clones = self.gdn_snapshot_clones,
            .decode_batch_steps = self.decode_batch_steps,
            .decode_batch_tokens = self.decode_batch_tokens,
            .total_wired_bytes = self.total_wired_bytes,
            .live_admission = self.live_admission,
        };
    }

    pub fn batchBackend(self: *GraphBatchRuntime) batch_sched.BatchBackend {
        return .{
            .ctx = self,
            .lookupPrefixFn = if (self.prefix_cache != null) lookupPrefixCb else null,
            .prefillSlotFn = prefillSlotCb,
            .stepDecodeBatchFn = stepDecodeBatchCb,
            .releaseSlotFn = releaseSlotCb,
        };
    }

    fn poolTilePair(self: *const GraphBatchRuntime, group_id: u32, kv_idx: u32) struct { k: metal.KvTile, v: metal.KvTile } {
        const base = (@as(usize, group_id) * self.num_kv_layers + kv_idx) * 2;
        return .{ .k = self.pool_tiles[base], .v = self.pool_tiles[base + 1] };
    }

    fn slotKvTilesSlice(self: *const GraphBatchRuntime, kv_idx: u32, slot_id: u32) []attention.KvTilePair {
        const stride_elems = self.kv_tiles_layer_stride_bytes / @sizeOf(attention.KvTilePair);
        const start = @as(usize, kv_idx) * stride_elems + @as(usize, slot_id) * self.max_chunks;
        return self.kv_tiles_host[start .. start + self.max_chunks];
    }

    fn slotKvTilesBinding(self: *const GraphBatchRuntime, kv_idx: u32, slot_id: u32) metal.BufferBinding {
        const off = self.kv_tiles_base_off +
            @as(u64, kv_idx) * self.kv_tiles_layer_stride_bytes +
            @as(u64, slot_id) * @as(u64, self.max_chunks) * @sizeOf(attention.KvTilePair);
        return .{ .buffer = self.batch_buf, .offset = off };
    }

    fn layerKvTilesBinding(self: *const GraphBatchRuntime, kv_idx: u32) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.kv_tiles_base_off + @as(u64, kv_idx) * self.kv_tiles_layer_stride_bytes,
        };
    }

    fn gdnLayerPoolBinding(self: *const GraphBatchRuntime, gdn_idx: u32) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.gdn_pool_base_off + @as(u64, gdn_idx) * self.gdn_layer_stride_bytes,
        };
    }

    fn gdnSlotStateBinding(self: *const GraphBatchRuntime, gdn_idx: u32, slot_id: u32) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.gdn_pool_base_off +
                @as(u64, gdn_idx) * self.gdn_layer_stride_bytes +
                @as(u64, slot_id) * self.gdn_slot_stride_bytes,
        };
    }

    fn gdnSlotConvBinding(self: *const GraphBatchRuntime, gdn_idx: u32, slot_id: u32) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.gdn_conv_pool_base_off +
                @as(u64, gdn_idx) * self.gdn_conv_layer_stride_bytes +
                @as(u64, slot_id) * self.gdn_conv_slot_stride_bytes,
        };
    }

    fn qsaSlotBlockKeysBinding(self: *const GraphBatchRuntime, qsa_idx: u32, slot_id: u32) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.qsa_block_keys_base_off +
                @as(u64, qsa_idx) * self.qsa_block_keys_layer_stride_bytes +
                @as(u64, slot_id) * self.qsa_block_keys_slot_stride_bytes,
        };
    }

    fn gqaWriteSlotsSlice(self: *const GraphBatchRuntime, gqa_idx: u32) []attention.BatchKvWriteSlot {
        const stride_elems = self.write_slots_layer_stride_bytes / @sizeOf(attention.BatchKvWriteSlot);
        const start = @as(usize, gqa_idx) * stride_elems;
        return self.write_slots_host[start .. start + self.max_slots];
    }

    fn gqaWriteSlotsBinding(self: *const GraphBatchRuntime, gqa_idx: u32) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.write_slots_base_off + @as(u64, gqa_idx) * self.write_slots_layer_stride_bytes,
        };
    }

    fn gqaDescsSlice(self: *const GraphBatchRuntime, gqa_idx: u32) []attention.BatchGqaSlotDesc {
        const stride_elems = self.gqa_descs_layer_stride_bytes / @sizeOf(attention.BatchGqaSlotDesc);
        const start = @as(usize, gqa_idx) * stride_elems;
        return self.gqa_descs_host[start .. start + self.max_slots];
    }

    fn gqaDescsBinding(self: *const GraphBatchRuntime, gqa_idx: u32) metal.BufferBinding {
        return .{
            .buffer = self.batch_buf,
            .offset = self.gqa_descs_base_off + @as(u64, gqa_idx) * self.gqa_descs_layer_stride_bytes,
        };
    }

    fn releaseChunkGroup(self: *GraphBatchRuntime, group_id: u32) void {
        if (group_id == GRAPH_NONE_GROUP or self.num_kv_layers == 0) return;
        std.debug.assert(group_id < self.pool_chunk_groups);
        self.pool_free_stack[self.pool_free_top] = group_id;
        self.pool_free_top += 1;
    }

    fn reclaimEvicted(self: *GraphBatchRuntime, ev: prefix_cache_mod.EvictedEntry, handles: []const prefix_cache_mod.KvBlockHandle) void {
        if (self.num_kv_layers > 0 and handles.len > 0 and handles[0].storage == .active_tile and handles[0].pool_slot_id < self.pool_chunk_groups) {
            self.releaseChunkGroup(handles[0].pool_slot_id);
        }
        if (ev.gdn_snapshot_id != prefix_cache_mod.NO_SNAPSHOT) {
            if (self.gdn_snapshot_pool) |*sp| {
                sp.releaseSlot(ev.gdn_snapshot_id) catch unreachable;
            }
        }
    }

    fn acquireChunkGroup(self: *GraphBatchRuntime) ?u32 {
        if (self.num_kv_layers == 0) return GRAPH_NONE_GROUP;
        if (self.pool_free_top == 0) {
            if (self.prefix_cache) |*pc| {
                while (self.pool_free_top == 0) {
                    const ev = pc.evictLru(self.scratch_evicted_handles) catch break;
                    self.reclaimEvicted(ev, self.scratch_evicted_handles);
                }
            }
        }
        if (self.pool_free_top == 0) return null;
        self.pool_free_top -= 1;
        return self.pool_free_stack[self.pool_free_top];
    }

    fn acquireGdnSnapshotSlot(self: *GraphBatchRuntime) ?u32 {
        var sp = &(self.gdn_snapshot_pool orelse return null);
        if (sp.freeCount() == 0) {
            if (self.prefix_cache) |*pc| {
                while (sp.freeCount() == 0) {
                    const ev = pc.evictLru(self.scratch_evicted_handles) catch break;
                    self.reclaimEvicted(ev, self.scratch_evicted_handles);
                }
            }
        }
        return sp.acquireSlot() catch null;
    }

    fn clearSlotGdnState(self: *GraphBatchRuntime, slot_id: u32) void {
        if (self.num_gdn_layers == 0) return;
        const base = self.batch_arena.bytes();
        for (0..self.num_gdn_layers) |g| {
            const s_off: usize = @intCast(self.gdn_pool_base_off + g * self.gdn_layer_stride_bytes + @as(usize, slot_id) * self.gdn_slot_stride_bytes);
            @memset(base[s_off .. s_off + self.gdn_slot_stride_bytes], 0);
            if (self.gdn_conv_slot_stride_bytes > 0) {
                const c_off: usize = @intCast(self.gdn_conv_pool_base_off + g * self.gdn_conv_layer_stride_bytes + @as(usize, slot_id) * self.gdn_conv_slot_stride_bytes);
                @memset(base[c_off .. c_off + self.gdn_conv_slot_stride_bytes], 0);
            }
        }
    }

    fn saveOrRestoreGdnSnapshotGpu(self: *GraphBatchRuntime, slot_id: u32, snap_id: u32, is_restore: bool) bool {
        const sp = &(self.gdn_snapshot_pool orelse return false);
        const snap_state = sp.stateBinding(self.batch_buf, snap_id) catch return false;
        const snap_conv = sp.convBinding(self.batch_buf, snap_id) catch return false;
        const slot_state: metal.BufferBinding = .{
            .buffer = self.batch_buf,
            .offset = self.gdn_pool_base_off + @as(u64, slot_id) * self.gdn_slot_stride_bytes,
        };
        const slot_conv: ?metal.BufferBinding = if (self.gdn_conv_f16_per_layer > 0) .{
            .buffer = self.batch_buf,
            .offset = self.gdn_conv_pool_base_off + @as(u64, slot_id) * self.gdn_conv_slot_stride_bytes,
        } else null;
        const slot_state_layer_stride_f32: u32 = @intCast(self.gdn_layer_stride_bytes / @sizeOf(f32));
        const slot_conv_layer_stride_f16: u32 = @intCast(self.gdn_conv_layer_stride_bytes / @sizeOf(f16));

        var frame = self.model.compute.beginFrame(self.model.timeout_ms) catch return false;
        self.model.hybrid_pipes.encodeGdnCloneState(&frame, .{
            .src_state = if (is_restore) snap_state else slot_state,
            .dst_state = if (is_restore) slot_state else snap_state,
            .src_conv = if (is_restore) snap_conv else slot_conv,
            .dst_conv = if (is_restore) slot_conv else snap_conv,
            .num_gdn_layers = self.num_gdn_layers,
            .state_f32_per_layer = sp.layer_state_elems,
            .src_state_layer_stride_f32 = if (is_restore) sp.stateLayerStrideF32() else slot_state_layer_stride_f32,
            .dst_state_layer_stride_f32 = if (is_restore) slot_state_layer_stride_f32 else sp.stateLayerStrideF32(),
            .conv_f16_per_layer = self.gdn_conv_f16_per_layer,
            .src_conv_layer_stride_f16 = if (is_restore) sp.convLayerStrideF16() else slot_conv_layer_stride_f16,
            .dst_conv_layer_stride_f16 = if (is_restore) slot_conv_layer_stride_f16 else sp.convLayerStrideF16(),
        }, false) catch {
            frame.abort();
            return false;
        };
        const sig = frame.commit() catch return false;
        self.model.compute.wait(sig, self.model.timeout_ms) catch return false;
        self.gdn_snapshot_clones += 1;
        return true;
    }

    fn copyTailGroupGpu(self: *GraphBatchRuntime, src_group: u32, dst_group: u32, valid_tokens: u32) bool {
        if (self.num_kv_layers == 0) return true;
        var frame = self.model.compute.beginFrame(self.model.timeout_ms) catch return false;
        for (0..self.num_kv_layers) |ki| {
            const src_pair = self.poolTilePair(src_group, @intCast(ki));
            const dst_pair = self.poolTilePair(dst_group, @intCast(ki));
            prefix_cache_mod.PrefixCache.encodeTailCowCopy(
                self.model.attn_pipes,
                &frame,
                src_pair.k,
                src_pair.v,
                dst_pair.k,
                dst_pair.v,
                self.model.config.num_key_value_heads,
                self.model.config.head_dim,
                self.kv_chunk_tokens,
                valid_tokens,
                false,
            ) catch {
                frame.abort();
                return false;
            };
        }
        const sig = frame.commit() catch return false;
        self.model.compute.wait(sig, self.model.timeout_ms) catch return false;
        self.prefix_cow_copies += 1;
        return true;
    }

    fn refreshSlotQsaBlockSummaries(self: *GraphBatchRuntime, slot_id: u32, start_pos: u32, new_kv_len: u32) bool {
        if (self.num_qsa_layers == 0 or new_kv_len == 0) return true;
        var frame = self.model.compute.beginFrame(self.model.timeout_ms) catch return false;
        var had = false;
        const blk_tok = self.model.config.qsa_block_size;
        for (self.model.layers, 0..) |layer, l| {
            switch (layer.attn) {
                .qsa => |qsa| {
                    const kv_idx = self.layer_to_kv_idx[l];
                    const start_block: u32 = start_pos / blk_tok;
                    const total_blocks: u32 = (new_kv_len + blk_tok - 1) / blk_tok;
                    const num_blocks: u32 = if (total_blocks > start_block) (total_blocks - start_block) else 1;
                    self.model.hybrid_pipes.encodeQsaUpdateBlockSummary(&frame, .{
                        .kv_tiles = self.slotKvTilesBinding(kv_idx, slot_id),
                        .block_keys = self.qsaSlotBlockKeysBinding(qsa.qsa_idx, slot_id),
                        .num_kv_heads = self.model.config.num_key_value_heads,
                        .head_dim = self.model.config.head_dim,
                        .kv_len = new_kv_len,
                        .window_size = 0,
                        .ring_capacity = 0,
                        .chunk_tokens = self.kv_chunk_tokens,
                        .block_tokens = blk_tok,
                        .start_block = start_block,
                        .num_blocks = num_blocks,
                    }, had) catch {
                        frame.abort();
                        return false;
                    };
                    had = true;
                },
                else => {},
            }
        }
        if (had) {
            const sig = frame.commit() catch return false;
            self.model.compute.wait(sig, self.model.timeout_ms) catch return false;
        } else {
            frame.abort();
        }
        return true;
    }

    fn ensureSlotWritableChunk(self: *GraphBatchRuntime, slot_id: u32, chunk_idx: u32) bool {
        if (self.num_kv_layers == 0) return true;
        if (chunk_idx >= self.max_chunks) return false;
        const slot = &self.slots[slot_id];
        if (slot.chunk_group_ids[chunk_idx] != GRAPH_NONE_GROUP) {
            return !slot.chunk_is_shared[chunk_idx];
        }
        const g = self.acquireChunkGroup() orelse return false;
        slot.chunk_group_ids[chunk_idx] = g;
        slot.chunk_is_shared[chunk_idx] = false;
        for (0..self.num_kv_layers) |ki| {
            const pair = self.poolTilePair(g, @intCast(ki));
            self.slotKvTilesSlice(@intCast(ki), slot_id)[chunk_idx] = attention.KvTilePair.fromTiles(pair.k, pair.v, .rgba16f);
        }
        return true;
    }

    fn resetSlotState(self: *GraphBatchRuntime, slot_id: u32) void {
        if (slot_id >= self.max_slots) return;
        const slot = &self.slots[slot_id];
        if (self.prefix_cache) |*pc| {
            for (slot.pinned_full_ids[0..slot.pinned_full_count]) |eid| {
                pc.unpin(eid) catch unreachable;
            }
        }
        slot.pinned_full_count = 0;
        for (0..self.max_chunks) |c| {
            const g = slot.chunk_group_ids[c];
            if (g != GRAPH_NONE_GROUP and !slot.chunk_is_shared[c]) {
                self.releaseChunkGroup(g);
            }
            slot.chunk_group_ids[c] = GRAPH_NONE_GROUP;
            slot.chunk_is_shared[c] = false;
        }
        slot.seq_len = 0;
        slot.matched_prefix_tokens = 0;
        slot.prefix_cached_full_chunks = 0;
        slot.last_cached_parent_id = prefix_cache_mod.NONE_ID;
        slot.in_use = false;
        self.clearSlotGdnState(slot_id);
        for (0..self.num_kv_layers) |ki| {
            @memset(self.slotKvTilesSlice(@intCast(ki), slot_id), .{ .k_tex_id = 0, .v_tex_id = 0 });
        }
    }

    fn lookupPrefixCb(ctx: *anyopaque, slot_id: u16, prompt_tokens: []const u32) batch_sched.PrefixLookupResult {
        const self: *GraphBatchRuntime = @ptrCast(@alignCast(ctx));
        return self.lookupPrefix(slot_id, prompt_tokens);
    }

    pub fn lookupPrefix(self: *GraphBatchRuntime, slot_id: u16, prompt_tokens: []const u32) batch_sched.PrefixLookupResult {
        if (slot_id >= self.max_slots) return .{};
        self.resetSlotState(slot_id);
        const slot = &self.slots[slot_id];
        slot.in_use = true;
        const handle: u64 = @as(u64, slot_id) + 1;

        var pc = &(self.prefix_cache orelse return .{ .matched_tokens = 0, .prefix_handle = handle });
        if (prompt_tokens.len <= 1 or prompt_tokens.len > self.model.max_seq_len) {
            return .{ .matched_tokens = 0, .prefix_handle = handle };
        }

        const require_gdn = (self.num_gdn_layers > 0);
        var match = pc.matchAndPin(prompt_tokens, slot.pinned_full_ids[0..self.max_chunks], .{
            .max_match_tokens = prompt_tokens.len - 1,
            .allow_partial_tail = true,
            .require_gdn_snapshot = require_gdn,
        }) catch return .{ .matched_tokens = 0, .prefix_handle = handle };

        if (match.matched_tokens == 0) {
            return .{ .matched_tokens = 0, .prefix_handle = handle };
        }

        // Bind shared full KV chunks (`0` copy).
        if (self.num_kv_layers > 0) {
            for (0..match.full_block_count) |ci| {
                const eid = slot.pinned_full_ids[ci];
                const h0 = pc.entryHandle(eid, 0) catch {
                    pc.releaseMatch(&match, slot.pinned_full_ids[0..match.full_block_count]) catch unreachable;
                    self.resetSlotState(slot_id);
                    slot.in_use = true;
                    return .{ .matched_tokens = 0, .prefix_handle = handle };
                };
                slot.chunk_group_ids[ci] = h0.pool_slot_id;
                slot.chunk_is_shared[ci] = true;
                for (0..self.num_kv_layers) |ki| {
                    const hl = pc.entryHandle(eid, @intCast(ki)) catch unreachable;
                    self.slotKvTilesSlice(@intCast(ki), slot_id)[ci] = hl.toTilePair();
                }
            }
        }
        slot.pinned_full_count = match.full_block_count;
        slot.prefix_cached_full_chunks = match.full_block_count;
        slot.last_cached_parent_id = if (match.full_block_count > 0)
            slot.pinned_full_ids[match.full_block_count - 1]
        else
            prefix_cache_mod.NONE_ID;

        var final_matched = match.full_block_count * self.kv_chunk_tokens;
        var effective_snap = if (match.full_block_count > 0)
            (pc.entryInfo(slot.pinned_full_ids[match.full_block_count - 1]) catch unreachable).gdn_snapshot_id
        else
            prefix_cache_mod.NO_SNAPSHOT;

        if (match.tail_tokens > 0 and match.tail_entry_id != prefix_cache_mod.NONE_ID) {
            if (self.num_kv_layers == 0) {
                final_matched += match.tail_tokens;
                effective_snap = match.gdn_snapshot_id;
            } else {
                const tail_c = match.full_block_count;
                const tail_h0 = pc.entryHandle(match.tail_entry_id, 0) catch null;
                if (tail_h0 != null and tail_h0.?.storage == .active_tile and tail_c < self.max_chunks) {
                    if (self.acquireChunkGroup()) |dst_g| {
                        if (self.copyTailGroupGpu(tail_h0.?.pool_slot_id, dst_g, match.tail_tokens)) {
                            slot.chunk_group_ids[tail_c] = dst_g;
                            slot.chunk_is_shared[tail_c] = false;
                            for (0..self.num_kv_layers) |ki| {
                                const pair = self.poolTilePair(dst_g, @intCast(ki));
                                self.slotKvTilesSlice(@intCast(ki), slot_id)[tail_c] = attention.KvTilePair.fromTiles(pair.k, pair.v, .rgba16f);
                            }
                            final_matched += match.tail_tokens;
                            effective_snap = match.gdn_snapshot_id;
                        } else {
                            self.releaseChunkGroup(dst_g);
                        }
                    }
                }
            }
            pc.releaseTailPin(&match) catch unreachable;
        }

        // Restore GDN recurrent state + conv1d history snapshot on GPU (`ami_gdn_clone_state`).
        if (require_gdn and final_matched > 0) {
            if (effective_snap == prefix_cache_mod.NO_SNAPSHOT or
                !self.saveOrRestoreGdnSnapshotGpu(slot_id, effective_snap, true))
            {
                self.resetSlotState(slot_id);
                slot.in_use = true;
                return .{ .matched_tokens = 0, .prefix_handle = handle };
            }
        }

        // Refresh QSA block key summaries for the matched prefix tokens.
        if (final_matched > 0 and self.num_qsa_layers > 0) {
            if (!self.refreshSlotQsaBlockSummaries(slot_id, 0, final_matched)) {
                self.resetSlotState(slot_id);
                slot.in_use = true;
                return .{ .matched_tokens = 0, .prefix_handle = handle };
            }
        }

        @memcpy(slot.prompt_tokens[0..final_matched], prompt_tokens[0..final_matched]);
        slot.seq_len = final_matched;
        slot.matched_prefix_tokens = final_matched;
        if (final_matched > 0) {
            self.prefix_hits += 1;
            self.prefix_hit_tokens += final_matched;
        }
        return .{ .matched_tokens = final_matched, .prefix_handle = handle };
    }

    fn populatePrefixCache(
        self: *GraphBatchRuntime,
        slot_id: u32,
        is_final_chunk: bool,
        is_boundary_checkpoint: bool,
    ) void {
        var pc = &(self.prefix_cache orelse return);
        const slot = &self.slots[slot_id];
        const bs = self.kv_chunk_tokens;

        while ((slot.prefix_cached_full_chunks + 1) * bs <= slot.seq_len) {
            const c = slot.prefix_cached_full_chunks;
            if (!slot.chunk_is_shared[c]) {
                var snap_id: u32 = prefix_cache_mod.NO_SNAPSHOT;
                if (self.num_gdn_layers > 0) {
                    // Only exact when `slot.seq_len == (c + 1) * bs` (GDN state matches exact boundary).
                    if (slot.seq_len == (c + 1) * bs) {
                        if (self.acquireGdnSnapshotSlot()) |sid| {
                            if (self.saveOrRestoreGdnSnapshotGpu(slot_id, sid, false)) {
                                snap_id = sid;
                            } else {
                                self.gdn_snapshot_pool.?.releaseSlot(sid) catch unreachable;
                            }
                        }
                    }
                    if (snap_id == prefix_cache_mod.NO_SNAPSHOT) {
                        slot.prefix_cached_full_chunks += 1;
                        continue;
                    }
                }

                const gid = slot.chunk_group_ids[c];
                if (self.num_kv_layers > 0) {
                    if (gid == GRAPH_NONE_GROUP) {
                        if (snap_id != prefix_cache_mod.NO_SNAPSHOT) self.gdn_snapshot_pool.?.releaseSlot(snap_id) catch unreachable;
                        break;
                    }
                    for (0..self.num_kv_layers) |ki| {
                        const pair = self.poolTilePair(gid, @intCast(ki));
                        self.scratch_handles[ki] = prefix_cache_mod.KvBlockHandle.activeFromTiles(pair.k, pair.v, gid);
                    }
                } else {
                    self.scratch_handles[0] = .{ .k_tex_id = 0, .v_tex_id = 0, .pool_slot_id = 0, .storage = .active_tile };
                }

                const tok_start: usize = @as(usize, c) * bs;
                const ins = pc.insertFullBlock(
                    slot.last_cached_parent_id,
                    slot.prompt_tokens[tok_start .. tok_start + bs],
                    self.scratch_handles,
                    .{
                        .gdn_snapshot_id = snap_id,
                        .evict_if_full = true,
                        .evicted_handles_out = self.scratch_evicted_handles,
                        .pin_on_insert = true,
                    },
                ) catch {
                    if (snap_id != prefix_cache_mod.NO_SNAPSHOT) self.gdn_snapshot_pool.?.releaseSlot(snap_id) catch unreachable;
                    break;
                };
                if (ins.evicted) |ev| self.reclaimEvicted(ev, self.scratch_evicted_handles);
                if (ins.already_existed) {
                    if (snap_id != prefix_cache_mod.NO_SNAPSHOT) self.gdn_snapshot_pool.?.releaseSlot(snap_id) catch unreachable;
                    if (self.num_kv_layers > 0) {
                        const eh0 = pc.entryHandle(ins.entry_id, 0) catch break;
                        self.releaseChunkGroup(gid);
                        slot.chunk_group_ids[c] = eh0.pool_slot_id;
                        for (0..self.num_kv_layers) |ki| {
                            const hl = pc.entryHandle(ins.entry_id, @intCast(ki)) catch break;
                            self.slotKvTilesSlice(@intCast(ki), slot_id)[c] = hl.toTilePair();
                        }
                    }
                }
                slot.chunk_is_shared[c] = true;
                slot.pinned_full_ids[slot.pinned_full_count] = ins.entry_id;
                slot.pinned_full_count += 1;
                slot.last_cached_parent_id = ins.entry_id;
            }
            slot.prefix_cached_full_chunks += 1;
        }

        // Snapshot partial tail (`1..bs-1`) with optional GDN checkpoint when either
        // the prompt completes OR a ChatML message-boundary checkpoint (`C15`, ADR-0020)
        // lands mid-tile (`insertBoundaryCheckpoint` marks intra-tile checkpoints `evict_first = true`).
        if (is_final_chunk or is_boundary_checkpoint) {
            const tail_len = slot.seq_len % bs;
            const tail_c = slot.seq_len / bs;
            if (tail_len > 0 and tail_c < self.max_chunks and !slot.chunk_is_shared[tail_c]) {
                var snap_id: u32 = prefix_cache_mod.NO_SNAPSHOT;
                if (self.num_gdn_layers > 0) {
                    if (self.acquireGdnSnapshotSlot()) |sid| {
                        if (self.saveOrRestoreGdnSnapshotGpu(slot_id, sid, false)) {
                            snap_id = sid;
                        } else {
                            self.gdn_snapshot_pool.?.releaseSlot(sid) catch unreachable;
                            return;
                        }
                    } else return;
                }
                var snap_g: u32 = GRAPH_NONE_GROUP;
                if (self.num_kv_layers > 0) {
                    const src_g = slot.chunk_group_ids[tail_c];
                    if (src_g == GRAPH_NONE_GROUP) {
                        if (snap_id != prefix_cache_mod.NO_SNAPSHOT) self.gdn_snapshot_pool.?.releaseSlot(snap_id) catch unreachable;
                        return;
                    }
                    snap_g = self.acquireChunkGroup() orelse {
                        if (snap_id != prefix_cache_mod.NO_SNAPSHOT) self.gdn_snapshot_pool.?.releaseSlot(snap_id) catch unreachable;
                        return;
                    };
                    if (!self.copyTailGroupGpu(src_g, snap_g, tail_len)) {
                        self.releaseChunkGroup(snap_g);
                        if (snap_id != prefix_cache_mod.NO_SNAPSHOT) self.gdn_snapshot_pool.?.releaseSlot(snap_id) catch unreachable;
                        return;
                    }
                    for (0..self.num_kv_layers) |ki| {
                        const pair = self.poolTilePair(snap_g, @intCast(ki));
                        self.scratch_handles[ki] = prefix_cache_mod.KvBlockHandle.activeFromTiles(pair.k, pair.v, snap_g);
                    }
                } else {
                    self.scratch_handles[0] = .{ .k_tex_id = 0, .v_tex_id = 0, .pool_slot_id = 0, .storage = .active_tile };
                }

                const tok_start: usize = @as(usize, tail_c) * bs;
                const ins_res = if (is_boundary_checkpoint)
                    pc.insertBoundaryCheckpoint(
                        slot.last_cached_parent_id,
                        slot.prompt_tokens[tok_start .. tok_start + tail_len],
                        self.scratch_handles,
                        .{
                            .gdn_snapshot_id = snap_id,
                            .evict_if_full = true,
                            .evicted_handles_out = self.scratch_evicted_handles,
                            .pin_on_insert = false,
                            .evict_first = true,
                        },
                    )
                else
                    pc.insertTailBlock(
                        slot.last_cached_parent_id,
                        slot.prompt_tokens[tok_start .. tok_start + tail_len],
                        self.scratch_handles,
                        .{
                            .gdn_snapshot_id = snap_id,
                            .evict_if_full = true,
                            .evicted_handles_out = self.scratch_evicted_handles,
                            .pin_on_insert = false,
                        },
                    );
                if (ins_res) |ins| {
                    if (ins.evicted) |ev| self.reclaimEvicted(ev, self.scratch_evicted_handles);
                    if (ins.already_existed) {
                        self.releaseChunkGroup(snap_g);
                        if (snap_id != prefix_cache_mod.NO_SNAPSHOT) self.gdn_snapshot_pool.?.releaseSlot(snap_id) catch unreachable;
                    }
                } else |_| {
                    self.releaseChunkGroup(snap_g);
                    if (snap_id != prefix_cache_mod.NO_SNAPSHOT) self.gdn_snapshot_pool.?.releaseSlot(snap_id) catch unreachable;
                }
            }
        }
    }

    fn runSlotPrefillWideChunk(
        self: *GraphBatchRuntime,
        slot_id: u32,
        tokens: []const u32,
        base_pos: u32,
        is_final_chunk: bool,
        compute_logits: bool,
    ) GraphStepError!u32 {
        const model = self.model;
        const slot = &self.slots[slot_id];
        const m: u32 = @intCast(tokens.len);
        const h = model.config.hidden_size;
        const w_stride = model.wide_prefill_max_tokens;
        const kv_len_after: u32 = base_pos + m;

        for (0..m) |i| {
            model.wp_token_ids_host[i] = tokens[i];
            model.wp_pos3d_host[i] = hybrid.Pos3D.text(base_pos + @as(u32, @intCast(i)));
        }
        if (self.num_kv_layers > 0) {
            const c0: u32 = base_pos / self.kv_chunk_tokens;
            const c1: u32 = (base_pos + m - 1) / self.kv_chunk_tokens;
            var ci: u32 = c0;
            while (ci <= c1) : (ci += 1) {
                if (!self.ensureSlotWritableChunk(slot_id, ci)) return error.SequenceFull;
            }
        }

        const last_l: usize = model.layers.len - 1;
        const allow_tail_prune = model.enable_prompt_tail_pruning and !model.compute_all_logits;
        const prune_non_final_last = allow_tail_prune and (!is_final_chunk or !compute_logits);
        const prune_final_last_to_m1 = allow_tail_prune and is_final_chunk and compute_logits and (m > 1);

        var frame = try model.compute.beginFrame(model.timeout_ms);
        errdefer frame.abort();

        try model.moe_pipes.encodeEmbedLookup(&frame, model.embed.dtype, .{
            .weight = model.embed.weight,
            .scale_bias = model.embed.scale_bias,
            .tokens = .{ .token_ids = model.wp_token_ids_bind },
            .out = model.wp_x_bind,
            .m = m,
            .vocab_size = model.config.vocab_size,
            .hidden_size = h,
        }, false);

        if (model.wp_gr_arena) |*gr| {
            try model.hybrid_pipes.encodeGrInitStream0(&frame, .{
                .x0 = model.wp_x_bind,
                .stream = gr.binding(model.scratch_buf),
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = w_stride,
            }, true);
        }

        for (model.layers, 0..) |*layer, l| {
            const is_last_layer = (l == last_l);
            const skip_last_rest = is_last_layer and prune_non_final_last;
            const tail_m1_last = is_last_layer and prune_final_last_to_m1;
            const eff_m: u32 = if (tail_m1_last) 1 else m;

            if (model.wp_gr_arena) |*gr| {
                const grw = layer.gr.?;
                try model.hybrid_pipes.encodeGrComputeGates(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .w_read = grw.read_gate,
                    .w_write = grw.write_gate,
                    .w_decay = grw.decay_gate,
                    .g_read_out = model.wp_gr_g_read_bind,
                    .g_write_out = model.wp_gr_g_write_bind,
                    .decay_out = model.wp_gr_decay_bind,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = w_stride,
                    .weight_dtype = grw.dtype,
                }, true);
                try model.hybrid_pipes.encodeGrReadGate(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .g_read = model.wp_gr_g_read_bind,
                    .x_in = model.wp_x_bind,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = w_stride,
                }, true);
            }

            try model.attn_pipes.encodeRmsNorm(&frame, .{
                .x = model.wp_x_bind,
                .w = layer.input_layernorm.binding,
                .y = model.wp_x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);

            const last_x_bind: metal.BufferBinding = .{
                .buffer = model.scratch_buf,
                .offset = model.wp_x_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
            };
            const last_x_norm_bind: metal.BufferBinding = .{
                .buffer = model.scratch_buf,
                .offset = model.wp_x_norm_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
            };
            const eff_x_bind: metal.BufferBinding = if (tail_m1_last) last_x_bind else model.wp_x_bind;

            switch (layer.attn) {
                .gdn => |g| {
                    const qk_dim = model.config.gdnQkDim();
                    const v_dim = model.config.gdnVDim();
                    const need_q = (!skip_last_rest) or (g.conv1d_w != null);
                    const q_eff_bind = if (g.conv1d_w != null) model.wp_q_conv_bind else model.wp_q_bind;
                    const k_eff_bind = if (g.conv1d_w != null) model.wp_k_conv_bind else model.wp_k_bind;
                    const v_eff_bind = if (g.conv1d_w != null) model.wp_v_conv_bind else model.wp_v_bind;

                    if (need_q) {
                        try model.encodeLinearGemmWithTile(&frame, g.q_proj, model.wp_x_norm_bind, model.wp_q_bind, m, true, true);
                        try model.encodeLinearGemmWithTile(&frame, g.k_proj, model.wp_x_norm_bind, model.wp_k_bind, m, true, false);
                    } else {
                        try model.encodeLinearGemmWithTile(&frame, g.k_proj, model.wp_x_norm_bind, model.wp_k_bind, m, true, true);
                    }
                    try model.encodeLinearGemmWithTile(&frame, g.v_proj, model.wp_x_norm_bind, model.wp_v_bind, m, true, false);
                    try model.hybrid_pipes.encodeGdnComputeGates(&frame, .{
                        .x = model.wp_x_norm_bind,
                        .w_a = g.a_proj,
                        .w_b = g.b_proj,
                        .a_log = g.a_log,
                        .dt_bias = g.dt_bias,
                        .g_out = model.wp_gdn_g_bind,
                        .beta_out = model.wp_gdn_beta_bind,
                        .m = m,
                        .hidden_dim = h,
                        .num_v_heads = model.config.linear_num_value_heads,
                        .weight_dtype = g.ab_dtype,
                    }, false);
                    if (!skip_last_rest) {
                        if (tail_m1_last) {
                            const last_z: metal.BufferBinding = .{
                                .buffer = model.scratch_buf,
                                .offset = model.wp_gdn_z_bind.offset + @as(u64, m - 1) * v_dim * @sizeOf(f16),
                            };
                            try model.encodeLinearGemv(&frame, g.g_proj, last_x_norm_bind, last_z, false);
                        } else {
                            try model.encodeLinearGemmWithTile(&frame, g.g_proj, model.wp_x_norm_bind, model.wp_gdn_z_bind, m, true, false);
                        }
                    }

                    if (g.conv1d_w) |cw| {
                        const conv_bind = self.gdnSlotConvBinding(g.gdn_idx, slot_id);
                        try model.hybrid_pipes.encodeGdnConv1dSilu(&frame, .{
                            .q_in = model.wp_q_bind,
                            .k_in = model.wp_k_bind,
                            .v_in = model.wp_v_bind,
                            .conv_w = cw,
                            .conv_state_in = conv_bind,
                            .conv_state_out = conv_bind,
                            .q_out = q_eff_bind,
                            .k_out = k_eff_bind,
                            .v_out = v_eff_bind,
                            .m = m,
                            .gdn_qk = qk_dim,
                            .gdn_v = v_dim,
                            .conv_k = g.conv_k,
                            .is_tree = false,
                            .write_state = true,
                        }, true);
                    }

                    const slot_s = self.gdnSlotStateBinding(g.gdn_idx, slot_id);
                    try model.hybrid_pipes.encodeGdnPrefill(&frame, .{
                        .q = q_eff_bind,
                        .k = k_eff_bind,
                        .v = v_eff_bind,
                        .g = model.wp_gdn_g_bind,
                        .beta = model.wp_gdn_beta_bind,
                        .z = model.wp_gdn_z_bind,
                        .norm_w = if (g.out_norm) |n| n.binding else null,
                        .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                        .state_in = slot_s,
                        .state_out = slot_s,
                        .out = model.wp_attn_out_bind,
                        .seq_len = m,
                        .num_k_heads = model.config.linear_num_key_heads,
                        .num_v_heads = model.config.linear_num_value_heads,
                        .head_k_dim = model.config.linear_key_head_dim,
                        .head_v_dim = model.config.linear_value_head_dim,
                        .rms_eps = model.config.rms_norm_eps,
                        .write_state_only = skip_last_rest,
                        .out_start_token = if (tail_m1_last) m - 1 else 0,
                    }, true);
                    if (skip_last_rest) break;
                    try model.encodeLinearGemmWithTile(&frame, g.o_proj, model.wp_attn_out_bind, model.wp_proj_out_bind, eff_m, true, true);
                },
                .qsa => |*qsa| {
                    const kv_idx = self.layer_to_kv_idx[l];
                    const kv_d = model.config.kvDim();
                    const slot_kv_bind = self.slotKvTilesBinding(kv_idx, slot_id);
                    const bk_bind = self.qsaSlotBlockKeysBinding(qsa.qsa_idx, slot_id);
                    const mrope_rot: u32 = (model.config.mrope_section.section[0] + model.config.mrope_section.section[1] + model.config.mrope_section.section[2]) * 2;
                    const eff_rot_dim: u32 = if (mrope_rot > 0 and mrope_rot <= model.config.head_dim) mrope_rot else model.config.head_dim;
                    const blk_tok = model.config.qsa_block_size;

                    if (skip_last_rest) {
                        try model.encodeLinearGemmWithTile(&frame, qsa.k_proj, model.wp_x_norm_bind, model.wp_k_bind, m, true, true);
                        try model.encodeLinearGemmWithTile(&frame, qsa.v_proj, model.wp_x_norm_bind, model.wp_v_bind, m, true, false);
                        try model.hybrid_pipes.encodeQsaPrefillRope(&frame, .{
                            .q = model.wp_q_bind,
                            .k = model.wp_k_bind,
                            .v = model.wp_v_bind,
                            .q_norm = qsa.q_norm,
                            .k_norm = qsa.k_norm,
                            .norm_dtype = qsa.qk_norm_dtype,
                            .positions_3d = model.wp_pos3d_bind,
                            .kv_tiles = slot_kv_bind,
                            .num_tokens = m,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .rotary_dim = eff_rot_dim,
                            .start_pos = base_pos,
                            .mrope = model.config.mrope_section,
                            .yarn = model.config.yarn,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .kv_only = true,
                            .write_to_tiles = true,
                            .rms_eps = model.config.rms_norm_eps,
                            .rope_theta = model.config.rope_theta,
                        }, true);
                        const start_blk = base_pos / blk_tok;
                        const tot_blk = (kv_len_after + blk_tok - 1) / blk_tok;
                        try model.hybrid_pipes.encodeQsaUpdateBlockSummary(&frame, .{
                            .kv_tiles = slot_kv_bind,
                            .block_keys = bk_bind,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .kv_len = kv_len_after,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .block_tokens = blk_tok,
                            .start_block = start_blk,
                            .num_blocks = if (tot_blk > start_blk) tot_blk - start_blk else 1,
                        }, true);
                        break;
                    }

                    if (tail_m1_last) {
                        try model.encodeLinearGemmWithTile(&frame, qsa.k_proj, model.wp_x_norm_bind, model.wp_k_bind, m, true, true);
                        try model.encodeLinearGemmWithTile(&frame, qsa.v_proj, model.wp_x_norm_bind, model.wp_v_bind, m, true, false);
                        try model.encodeLinearGemv(&frame, qsa.q_proj, last_x_norm_bind, model.wp_q_bind, false);
                        try model.hybrid_pipes.encodeQsaPrefillRope(&frame, .{
                            .q = model.wp_q_bind,
                            .k = model.wp_k_bind,
                            .v = model.wp_v_bind,
                            .q_norm = qsa.q_norm,
                            .k_norm = qsa.k_norm,
                            .norm_dtype = qsa.qk_norm_dtype,
                            .positions_3d = model.wp_pos3d_bind,
                            .kv_tiles = slot_kv_bind,
                            .num_tokens = m - 1,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .rotary_dim = eff_rot_dim,
                            .start_pos = base_pos,
                            .mrope = model.config.mrope_section,
                            .yarn = model.config.yarn,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .kv_only = true,
                            .write_to_tiles = true,
                            .rms_eps = model.config.rms_norm_eps,
                            .rope_theta = model.config.rope_theta,
                        }, true);
                        try model.hybrid_pipes.encodeQsaPrefillRope(&frame, .{
                            .q = model.wp_q_bind,
                            .k = .{ .buffer = model.scratch_buf, .offset = model.wp_k_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16) },
                            .v = .{ .buffer = model.scratch_buf, .offset = model.wp_v_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16) },
                            .q_norm = qsa.q_norm,
                            .k_norm = qsa.k_norm,
                            .norm_dtype = qsa.qk_norm_dtype,
                            .positions_3d = .{ .buffer = model.scratch_buf, .offset = model.wp_pos3d_bind.offset + @as(u64, m - 1) * @sizeOf(hybrid.Pos3D) },
                            .kv_tiles = slot_kv_bind,
                            .num_tokens = 1,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .rotary_dim = eff_rot_dim,
                            .start_pos = base_pos + m - 1,
                            .mrope = model.config.mrope_section,
                            .yarn = model.config.yarn,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .kv_only = false,
                            .write_to_tiles = true,
                            .rms_eps = model.config.rms_norm_eps,
                            .rope_theta = model.config.rope_theta,
                        }, false);
                    } else {
                        try model.encodeLinearGemmWithTile(&frame, qsa.q_proj, model.wp_x_norm_bind, model.wp_q_bind, m, true, true);
                        try model.encodeLinearGemmWithTile(&frame, qsa.k_proj, model.wp_x_norm_bind, model.wp_k_bind, m, true, false);
                        try model.encodeLinearGemmWithTile(&frame, qsa.v_proj, model.wp_x_norm_bind, model.wp_v_bind, m, true, false);
                        try model.hybrid_pipes.encodeQsaPrefillRope(&frame, .{
                            .q = model.wp_q_bind,
                            .k = model.wp_k_bind,
                            .v = model.wp_v_bind,
                            .q_norm = qsa.q_norm,
                            .k_norm = qsa.k_norm,
                            .norm_dtype = qsa.qk_norm_dtype,
                            .positions_3d = model.wp_pos3d_bind,
                            .kv_tiles = slot_kv_bind,
                            .num_tokens = m,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .rotary_dim = eff_rot_dim,
                            .start_pos = base_pos,
                            .mrope = model.config.mrope_section,
                            .yarn = model.config.yarn,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .kv_only = false,
                            .write_to_tiles = true,
                            .rms_eps = model.config.rms_norm_eps,
                            .rope_theta = model.config.rope_theta,
                        }, true);
                    }

                    const start_blk = base_pos / blk_tok;
                    const tot_blk = (kv_len_after + blk_tok - 1) / blk_tok;
                    try model.hybrid_pipes.encodeQsaUpdateBlockSummary(&frame, .{
                        .kv_tiles = slot_kv_bind,
                        .block_keys = bk_bind,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .kv_len = kv_len_after,
                        .chunk_tokens = self.kv_chunk_tokens,
                        .block_tokens = blk_tok,
                        .start_block = start_blk,
                        .num_blocks = if (tot_blk > start_blk) tot_blk - start_blk else 1,
                    }, true);
                    try model.hybrid_pipes.encodeQsaAttention(&frame, .{
                        .q = model.wp_q_bind,
                        .kv_tiles = slot_kv_bind,
                        .block_keys = bk_bind,
                        .index_buf = &(qsa.wp_index_buf.?),
                        .index_metal_buf = model.scratch_buf,
                        .out = model.wp_attn_out_bind,
                        .num_nodes = eff_m,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .kv_len = kv_len_after,
                        .chunk_tokens = self.kv_chunk_tokens,
                        .block_tokens = blk_tok,
                        .top_k_blocks = model.config.qsa_top_k_blocks,
                        .reuse_block_indices = false,
                        .causal_q_start_pos = if (tail_m1_last) base_pos + m - 1 else base_pos,
                    }, true);
                    try model.encodeLinearGemmWithTile(&frame, qsa.o_proj, model.wp_attn_out_bind, model.wp_proj_out_bind, eff_m, true, true);
                },
                .gqa => |gqa| {
                    const kv_idx = self.layer_to_kv_idx[l];
                    const kv_d = model.config.kvDim();
                    const slot_kv_bind = self.slotKvTilesBinding(kv_idx, slot_id);
                    if (skip_last_rest) {
                        try model.encodeLinearGemmWithTile(&frame, gqa.k_proj, model.wp_x_norm_bind, model.wp_k_bind, m, true, true);
                        try model.encodeLinearGemmWithTile(&frame, gqa.v_proj, model.wp_x_norm_bind, model.wp_v_bind, m, true, false);
                        try model.attn_pipes.encodePrefillQkNormRopeKvWrite(&frame, .{
                            .q = model.wp_q_bind,
                            .k = model.wp_k_bind,
                            .v = model.wp_v_bind,
                            .q_norm = gqa.q_norm,
                            .k_norm = gqa.k_norm,
                            .kv_tiles = slot_kv_bind,
                            .num_tokens = m,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .start_pos = base_pos,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .kv_only = true,
                            .rms_eps = model.config.rms_norm_eps,
                            .rope_theta = model.config.rope_theta,
                            .norm_dtype = gqa.qk_norm_dtype,
                        }, true);
                        break;
                    }
                    if (tail_m1_last) {
                        try model.encodeLinearGemmWithTile(&frame, gqa.k_proj, model.wp_x_norm_bind, model.wp_k_bind, m, true, true);
                        try model.encodeLinearGemmWithTile(&frame, gqa.v_proj, model.wp_x_norm_bind, model.wp_v_bind, m, true, false);
                        try model.encodeLinearGemv(&frame, gqa.q_proj, last_x_norm_bind, model.wp_q_bind, false);
                        try model.attn_pipes.encodePrefillQkNormRopeKvWrite(&frame, .{
                            .q = model.wp_q_bind,
                            .k = model.wp_k_bind,
                            .v = model.wp_v_bind,
                            .q_norm = gqa.q_norm,
                            .k_norm = gqa.k_norm,
                            .kv_tiles = slot_kv_bind,
                            .num_tokens = m - 1,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .start_pos = base_pos,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .kv_only = true,
                            .rms_eps = model.config.rms_norm_eps,
                            .rope_theta = model.config.rope_theta,
                            .norm_dtype = gqa.qk_norm_dtype,
                        }, true);
                        try model.attn_pipes.encodePrefillQkNormRopeKvWrite(&frame, .{
                            .q = model.wp_q_bind,
                            .k = .{ .buffer = model.scratch_buf, .offset = model.wp_k_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16) },
                            .v = .{ .buffer = model.scratch_buf, .offset = model.wp_v_bind.offset + @as(u64, m - 1) * kv_d * @sizeOf(f16) },
                            .q_norm = gqa.q_norm,
                            .k_norm = gqa.k_norm,
                            .kv_tiles = slot_kv_bind,
                            .num_tokens = 1,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .start_pos = base_pos + m - 1,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .kv_only = false,
                            .rms_eps = model.config.rms_norm_eps,
                            .rope_theta = model.config.rope_theta,
                            .norm_dtype = gqa.qk_norm_dtype,
                        }, false);
                    } else {
                        try model.encodeLinearGemmWithTile(&frame, gqa.q_proj, model.wp_x_norm_bind, model.wp_q_bind, m, true, true);
                        try model.encodeLinearGemmWithTile(&frame, gqa.k_proj, model.wp_x_norm_bind, model.wp_k_bind, m, true, false);
                        try model.encodeLinearGemmWithTile(&frame, gqa.v_proj, model.wp_x_norm_bind, model.wp_v_bind, m, true, false);
                        try model.attn_pipes.encodePrefillQkNormRopeKvWrite(&frame, .{
                            .q = model.wp_q_bind,
                            .k = model.wp_k_bind,
                            .v = model.wp_v_bind,
                            .q_norm = gqa.q_norm,
                            .k_norm = gqa.k_norm,
                            .kv_tiles = slot_kv_bind,
                            .num_tokens = m,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .start_pos = base_pos,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .kv_only = false,
                            .rms_eps = model.config.rms_norm_eps,
                            .rope_theta = model.config.rope_theta,
                            .norm_dtype = gqa.qk_norm_dtype,
                        }, true);
                    }
                    try model.attn_pipes.encodeGqaCausalPrefillFlashAttn(&frame, .{
                        .q = model.wp_q_bind,
                        .kv_tiles = slot_kv_bind,
                        .out = model.wp_attn_out_bind,
                        .num_q = eff_m,
                        .q_start_pos = if (tail_m1_last) base_pos + m - 1 else base_pos,
                        .kv_len = kv_len_after,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .chunk_tokens = self.kv_chunk_tokens,
                    }, true);
                    try model.encodeLinearGemmWithTile(&frame, gqa.o_proj, model.wp_attn_out_bind, model.wp_proj_out_bind, eff_m, true, true);
                },
            }

            if (model.wp_gr_arena) |*gr| {
                const stream_eff: metal.BufferBinding = if (tail_m1_last)
                    .{ .buffer = model.scratch_buf, .offset = gr.base_offset + @as(u64, m - 1) * h * @sizeOf(f16) }
                else
                    gr.binding(model.scratch_buf);
                const decay_eff: metal.BufferBinding = if (tail_m1_last)
                    .{ .buffer = model.scratch_buf, .offset = model.wp_gr_decay_bind.offset + @as(u64, m - 1) * 4 * 4 }
                else
                    model.wp_gr_decay_bind;
                const gwrite_eff: metal.BufferBinding = if (tail_m1_last)
                    .{ .buffer = model.scratch_buf, .offset = model.wp_gr_g_write_bind.offset + @as(u64, m - 1) * 4 * 4 }
                else
                    model.wp_gr_g_write_bind;
                const gread_eff: metal.BufferBinding = if (tail_m1_last)
                    .{ .buffer = model.scratch_buf, .offset = model.wp_gr_g_read_bind.offset + @as(u64, m - 1) * 4 * 4 }
                else
                    model.wp_gr_g_read_bind;
                try model.hybrid_pipes.encodeGrWriteGate(&frame, .{
                    .stream = stream_eff,
                    .y_delta = model.wp_proj_out_bind,
                    .decay = decay_eff,
                    .g_write = gwrite_eff,
                    .m = eff_m,
                    .hidden_dim = h,
                    .branch_token_stride = w_stride,
                }, true);
                try model.hybrid_pipes.encodeGrReadGate(&frame, .{
                    .stream = stream_eff,
                    .g_read = gread_eff,
                    .x_in = eff_x_bind,
                    .m = eff_m,
                    .hidden_dim = h,
                    .branch_token_stride = w_stride,
                }, true);
            } else {
                try model.moe_pipes.encodeResidualAdd(&frame, .{
                    .x = eff_x_bind,
                    .delta = model.wp_proj_out_bind,
                    .count = eff_m * h,
                }, true);
            }

            try model.attn_pipes.encodeRmsNorm(&frame, .{
                .x = eff_x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = model.wp_x_norm_bind,
                .num_rows = eff_m,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);

            switch (model.config.axes.ffn_kind) {
                .dense_swiglu => {
                    try model.encodeLinearGemmWithTile(&frame, layer.dense_gate, model.wp_x_norm_bind, model.wp_dense_gate_bind, eff_m, true, true);
                    try model.encodeLinearGemmWithTile(&frame, layer.dense_up, model.wp_x_norm_bind, model.wp_dense_up_bind, eff_m, true, false);
                    try model.moe_pipes.encodeSiluMul(&frame, .{
                        .gate = model.wp_dense_gate_bind,
                        .up = model.wp_dense_up_bind,
                        .out = model.wp_dense_act_bind,
                        .count = eff_m * model.config.intermediate_size,
                    }, true);
                    try model.encodeLinearGemmWithTile(&frame, layer.dense_down, model.wp_dense_act_bind, model.wp_proj_out_bind, eff_m, true, true);
                    if (model.wp_gr_arena) |*gr| {
                        const stream_eff: metal.BufferBinding = if (tail_m1_last)
                            .{ .buffer = model.scratch_buf, .offset = gr.base_offset + @as(u64, m - 1) * h * @sizeOf(f16) }
                        else
                            gr.binding(model.scratch_buf);
                        const decay_eff: metal.BufferBinding = if (tail_m1_last)
                            .{ .buffer = model.scratch_buf, .offset = model.wp_gr_decay_bind.offset + @as(u64, m - 1) * 4 * 4 }
                        else
                            model.wp_gr_decay_bind;
                        const gwrite_eff: metal.BufferBinding = if (tail_m1_last)
                            .{ .buffer = model.scratch_buf, .offset = model.wp_gr_g_write_bind.offset + @as(u64, m - 1) * 4 * 4 }
                        else
                            model.wp_gr_g_write_bind;
                        try model.hybrid_pipes.encodeGrWriteGate(&frame, .{
                            .stream = stream_eff,
                            .y_delta = model.wp_proj_out_bind,
                            .decay = decay_eff,
                            .g_write = gwrite_eff,
                            .m = eff_m,
                            .hidden_dim = h,
                            .branch_token_stride = w_stride,
                        }, true);
                    } else {
                        try model.moe_pipes.encodeResidualAdd(&frame, .{
                            .x = eff_x_bind,
                            .delta = model.wp_proj_out_bind,
                            .count = eff_m * h,
                        }, true);
                    }
                },
                .moe_swiglu => {
                    const ws_ptr = &(model.wp_moe_ws.?);
                    const router_src: moe.RouterSource = if (layer.router_quant_dtype) |rqd| .{
                        .quant_weights = .{
                            .qweight = layer.router_binding,
                            .scale_bias = layer.router_scale_bias.?,
                            .bits = switch (rqd) {
                                .q2_affine => .q2,
                                .q3_affine => .q3,
                                .q4_affine => .q4,
                                else => return error.InvalidBatchSize,
                            },
                        },
                    } else .{
                        .weights = .{
                            .binding = layer.router_binding,
                            .dtype = layer.router_dtype,
                        },
                    };
                    const residual_bind: ?metal.BufferBinding = if (model.wp_gr_arena != null) null else eff_x_bind;
                    const out_bind: metal.BufferBinding = if (model.wp_gr_arena != null) model.wp_proj_out_bind else eff_x_bind;
                    const moe_args: moe.MoELayerArgs = .{
                        .x = model.wp_x_norm_bind,
                        .router = router_src,
                        .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                        .expert_table = layer.expert_table,
                        .workspace = ws_ptr,
                        .workspace_buf = model.scratch_buf,
                        .residual = residual_bind,
                        .out = out_bind,
                        .m = eff_m,
                        .num_experts = model.config.num_experts,
                        .top_k = model.config.num_experts_per_tok,
                        .hidden_size = h,
                        .intermediate_size = model.config.moe_intermediate_size,
                        .norm_topk_prob = model.config.norm_topk_prob,
                        .shared_expert = layer.shared_expert,
                        .shared_weights_base = if (layer.shared_expert_buf) |sb| .{ .buffer = sb, .offset = 0 } else null,
                        .shared_expert_dtype = layer.shared_expert_dtype,
                        .shared_intermediate_size = model.config.shared_expert_intermediate_size,
                        .shared_gate_weight = layer.shared_gate_weight,
                        .shared_gate_dtype = layer.shared_gate_dtype,
                    };
                    if (layer.is_hetero_moe) {
                        try model.moe_pipes.encodeMoELayerHetero(&frame, .{
                            .base = moe_args,
                            .expert_schemes = layer.expert_dtypes_binding.?,
                            .scheme_counts = layer.expert_scheme_counts,
                        }, true);
                    } else {
                        switch (layer.expert_weight_dtype) {
                            .q2_affine => try model.moe_pipes.encodeMoELayer(&frame, .q2, moe_args, true),
                            .q3_affine => try model.moe_pipes.encodeMoELayer(&frame, .q3, moe_args, true),
                            .q4_affine => try model.moe_pipes.encodeMoELayer(&frame, .q4, moe_args, true),
                            .f16 => try model.moe_pipes.encodeMoELayerFloat(&frame, .f16, moe_args, true),
                            .bf16 => try model.moe_pipes.encodeMoELayerFloat(&frame, .bf16, moe_args, true),
                        }
                    }
                    if (model.wp_gr_arena) |*gr| {
                        const stream_eff: metal.BufferBinding = if (tail_m1_last)
                            .{ .buffer = model.scratch_buf, .offset = gr.base_offset + @as(u64, m - 1) * h * @sizeOf(f16) }
                        else
                            gr.binding(model.scratch_buf);
                        const decay_eff: metal.BufferBinding = if (tail_m1_last)
                            .{ .buffer = model.scratch_buf, .offset = model.wp_gr_decay_bind.offset + @as(u64, m - 1) * 4 * 4 }
                        else
                            model.wp_gr_decay_bind;
                        const gwrite_eff: metal.BufferBinding = if (tail_m1_last)
                            .{ .buffer = model.scratch_buf, .offset = model.wp_gr_g_write_bind.offset + @as(u64, m - 1) * 4 * 4 }
                        else
                            model.wp_gr_g_write_bind;
                        try model.hybrid_pipes.encodeGrWriteGate(&frame, .{
                            .stream = stream_eff,
                            .y_delta = model.wp_proj_out_bind,
                            .decay = decay_eff,
                            .g_write = gwrite_eff,
                            .m = eff_m,
                            .hidden_dim = h,
                            .branch_token_stride = w_stride,
                        }, true);
                    }
                },
            }
        }

        if (compute_logits) {
            const last_x_bind: metal.BufferBinding = .{
                .buffer = model.scratch_buf,
                .offset = model.wp_x_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
            };
            if (model.wp_gr_arena) |*gr| {
                const stream_last: metal.BufferBinding = .{
                    .buffer = model.scratch_buf,
                    .offset = gr.base_offset + @as(u64, m - 1) * h * @sizeOf(f16),
                };
                try model.hybrid_pipes.encodeGrCollapse(&frame, .{
                    .stream = stream_last,
                    .out = last_x_bind,
                    .mode = .mean,
                    .m = 1,
                    .hidden_dim = h,
                    .branch_token_stride = w_stride,
                }, true);
            }
            try model.attn_pipes.encodeRmsNorm(&frame, .{
                .x = last_x_bind,
                .w = model.final_norm.binding,
                .y = model.wp_x_norm_bind,
                .num_rows = 1,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = model.final_norm.dtype,
            }, true);
            try model.encodeLinearGemv(&frame, model.lm_head, model.wp_x_norm_bind, model.logits_bind, true);
            try model.moe_pipes.encodeArgmaxRows(&frame, .{
                .logits = model.logits_bind,
                .out_token_ids = model.out_token_id_bind,
                .num_rows = 1,
                .vocab_size = model.config.vocab_size,
                .logits_dtype = .f16,
            }, true);
        }

        const sig = try frame.commit();
        try model.compute.wait(sig, model.timeout_ms);
        slot.seq_len = kv_len_after;
        if (compute_logits) model.last_logits_row = 0;
        return if (compute_logits) model.out_token_ids_host[0] else 0;
    }

    fn runSlotPrefillSubChunk(
        self: *GraphBatchRuntime,
        slot_id: u32,
        tokens: []const u32,
        base_pos: u32,
        compute_logits: bool,
    ) GraphStepError!u32 {
        const model = self.model;
        const slot = &self.slots[slot_id];
        const m: u32 = @intCast(tokens.len);
        const h = model.config.hidden_size;

        var acc_nodes: [hybrid.MAX_TREE_NODES]u16 = undefined;
        for (0..m) |i| {
            const p = base_pos + @as(u32, @intCast(i));
            model.token_ids_host[i] = tokens[i];
            model.positions_host[i] = p;
            model.pos3d_host[i] = hybrid.Pos3D.text(p);
            model.ancestor_masks_host[i] = if (i == 63) std.math.maxInt(u64) else ((@as(u64, 1) << @intCast(i + 1)) - 1);
            acc_nodes[i] = @intCast(i);
        }

        if (self.num_kv_layers > 0) {
            const c0: u32 = base_pos / self.kv_chunk_tokens;
            const c1: u32 = (base_pos + m - 1) / self.kv_chunk_tokens;
            var ci: u32 = c0;
            while (ci <= c1) : (ci += 1) {
                if (!self.ensureSlotWritableChunk(slot_id, ci)) return error.SequenceFull;
            }
        }

        var frame = try model.compute.beginFrame(model.timeout_ms);
        errdefer frame.abort();

        try model.moe_pipes.encodeEmbedLookup(&frame, model.embed.dtype, .{
            .weight = model.embed.weight,
            .scale_bias = model.embed.scale_bias,
            .tokens = .{ .token_ids = model.token_ids_bind },
            .out = model.x_bind,
            .m = m,
            .vocab_size = model.config.vocab_size,
            .hidden_size = h,
        }, false);

        if (model.gr_arena) |*gr| {
            try model.hybrid_pipes.encodeGrInitStream0(&frame, .{
                .x0 = model.x_bind,
                .stream = gr.binding(model.scratch_buf),
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = model.max_tree_nodes,
            }, true);
        }

        for (model.layers, 0..) |*layer, l| {
            // Optional 4-branch GR read gate
            if (model.gr_arena) |*gr| {
                const grw = layer.gr.?;
                try model.hybrid_pipes.encodeGrComputeGates(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .w_read = grw.read_gate,
                    .w_write = grw.write_gate,
                    .w_decay = grw.decay_gate,
                    .g_read_out = model.gr_g_read_bind,
                    .g_write_out = model.gr_g_write_bind,
                    .decay_out = model.gr_decay_bind,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                    .weight_dtype = grw.dtype,
                }, true);
                try model.hybrid_pipes.encodeGrReadGate(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .g_read = model.gr_g_read_bind,
                    .x_in = model.x_bind,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                }, true);
            }

            try model.attn_pipes.encodeRmsNorm(&frame, .{
                .x = model.x_bind,
                .w = layer.input_layernorm.binding,
                .y = model.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);

            switch (layer.attn) {
                .gdn => |g| {
                    try model.encodeLinearBatched(&frame, g.q_proj, model.x_norm_bind, model.q_bind, m, true);
                    try model.encodeLinearBatched(&frame, g.k_proj, model.x_norm_bind, model.k_bind, m, false);
                    try model.encodeLinearBatched(&frame, g.v_proj, model.x_norm_bind, model.v_bind, m, false);
                    try model.encodeLinearBatched(&frame, g.g_proj, model.x_norm_bind, model.gdn_z_bind, m, false);
                    try model.hybrid_pipes.encodeGdnComputeGates(&frame, .{
                        .x = model.x_norm_bind,
                        .w_a = g.a_proj,
                        .w_b = g.b_proj,
                        .a_log = g.a_log,
                        .dt_bias = g.dt_bias,
                        .g_out = .{ .buffer = model.scratch_buf, .offset = g.tree_g_off },
                        .beta_out = .{ .buffer = model.scratch_buf, .offset = g.tree_beta_off },
                        .m = m,
                        .hidden_dim = h,
                        .num_v_heads = model.config.linear_num_value_heads,
                        .weight_dtype = g.ab_dtype,
                    }, false);

                    var q_eff = model.q_bind;
                    var k_eff: metal.BufferBinding = .{ .buffer = model.scratch_buf, .offset = g.tree_k_off };
                    var v_eff: metal.BufferBinding = .{ .buffer = model.scratch_buf, .offset = g.tree_v_off };
                    if (g.conv1d_w) |cw| {
                        const conv_bind = self.gdnSlotConvBinding(g.gdn_idx, slot_id);
                        try model.hybrid_pipes.encodeGdnConv1dSilu(&frame, .{
                            .q_in = model.q_bind,
                            .k_in = model.k_bind,
                            .v_in = model.v_bind,
                            .conv_w = cw,
                            .conv_state_in = conv_bind,
                            .conv_state_out = conv_bind,
                            .q_out = model.q_conv_bind,
                            .k_out = k_eff,
                            .v_out = v_eff,
                            .m = m,
                            .gdn_qk = model.config.gdnQkDim(),
                            .gdn_v = model.config.gdnVDim(),
                            .conv_k = g.conv_k,
                            .is_tree = false,
                            .write_state = true,
                        }, true);
                        q_eff = model.q_conv_bind;
                    } else {
                        k_eff = model.k_bind;
                        v_eff = model.v_bind;
                    }

                    const slot_s = self.gdnSlotStateBinding(g.gdn_idx, slot_id);
                    if (m == 1) {
                        try model.hybrid_pipes.encodeGdnDecode(&frame, .{
                            .q = q_eff,
                            .k = k_eff,
                            .v = v_eff,
                            .g = .{ .buffer = model.scratch_buf, .offset = g.tree_g_off },
                            .beta = .{ .buffer = model.scratch_buf, .offset = g.tree_beta_off },
                            .z = model.gdn_z_bind,
                            .norm_w = if (g.out_norm) |n| n.binding else null,
                            .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                            .state_in = slot_s,
                            .state_out = slot_s,
                            .out = model.attn_out_bind,
                            .num_k_heads = model.config.linear_num_key_heads,
                            .num_v_heads = model.config.linear_num_value_heads,
                            .head_k_dim = model.config.linear_key_head_dim,
                            .head_v_dim = model.config.linear_value_head_dim,
                            .rms_eps = model.config.rms_norm_eps,
                        }, true);
                    } else {
                        try model.hybrid_pipes.encodeGdnPrefill(&frame, .{
                            .q = q_eff,
                            .k = k_eff,
                            .v = v_eff,
                            .g = .{ .buffer = model.scratch_buf, .offset = g.tree_g_off },
                            .beta = .{ .buffer = model.scratch_buf, .offset = g.tree_beta_off },
                            .z = model.gdn_z_bind,
                            .norm_w = if (g.out_norm) |n| n.binding else null,
                            .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                            .state_in = slot_s,
                            .state_out = slot_s,
                            .out = model.attn_out_bind,
                            .seq_len = m,
                            .num_k_heads = model.config.linear_num_key_heads,
                            .num_v_heads = model.config.linear_num_value_heads,
                            .head_k_dim = model.config.linear_key_head_dim,
                            .head_v_dim = model.config.linear_value_head_dim,
                            .rms_eps = model.config.rms_norm_eps,
                        }, true);
                    }
                    try model.encodeLinearBatched(&frame, g.o_proj, model.attn_out_bind, model.proj_out_bind, m, true);
                },
                .qsa => |*qsa| {
                    const kv_idx = self.layer_to_kv_idx[l];
                    const tree_k: metal.BufferBinding = .{ .buffer = model.scratch_buf, .offset = qsa.tree_k_off };
                    const tree_v: metal.BufferBinding = .{ .buffer = model.scratch_buf, .offset = qsa.tree_v_off };
                    try model.encodeLinearBatched(&frame, qsa.q_proj, model.x_norm_bind, model.q_bind, m, true);
                    try model.encodeLinearBatched(&frame, qsa.k_proj, model.x_norm_bind, model.k_bind, m, false);
                    try model.encodeLinearBatched(&frame, qsa.v_proj, model.x_norm_bind, model.v_bind, m, false);

                    const mrope_rot: u32 = (model.config.mrope_section.section[0] + model.config.mrope_section.section[1] + model.config.mrope_section.section[2]) * 2;
                    const eff_rot_dim: u32 = if (mrope_rot > 0 and mrope_rot <= model.config.head_dim) mrope_rot else model.config.head_dim;
                    try model.hybrid_pipes.encodeQsaTreeRope(&frame, .{
                        .q = model.q_bind,
                        .k = model.k_bind,
                        .v = model.v_bind,
                        .q_norm = qsa.q_norm,
                        .k_norm = qsa.k_norm,
                        .positions_3d = model.pos3d_bind,
                        .k_tree_out = tree_k,
                        .v_tree_out = tree_v,
                        .num_nodes = m,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .rotary_dim = eff_rot_dim,
                        .mrope = model.config.mrope_section,
                        .rms_eps = model.config.rms_norm_eps,
                        .rope_theta = model.config.rope_theta,
                        .yarn = model.config.yarn,
                        .norm_dtype = qsa.qk_norm_dtype,
                    }, true);

                    const bk_bind = self.qsaSlotBlockKeysBinding(qsa.qsa_idx, slot_id);
                    try model.hybrid_pipes.encodeQsaAttention(&frame, .{
                        .q = model.q_bind,
                        .kv_tiles = if (base_pos > 0) self.slotKvTilesBinding(kv_idx, slot_id) else null,
                        .block_keys = bk_bind,
                        .index_buf = &qsa.index_buf,
                        .index_metal_buf = model.scratch_buf,
                        .k_tree = tree_k,
                        .v_tree = tree_v,
                        .ancestor_masks = model.ancestor_masks_bind,
                        .out = model.attn_out_bind,
                        .num_nodes = m,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .kv_len = base_pos,
                        .window_size = 0,
                        .ring_capacity = 0,
                        .chunk_tokens = self.kv_chunk_tokens,
                        .block_tokens = model.config.qsa_block_size,
                        .top_k_blocks = model.config.qsa_top_k_blocks,
                        .reuse_block_indices = false,
                    }, true);

                    // Commit KV rows into this slot's private tiles & refresh block summaries.
                    const c0: u32 = base_pos / self.kv_chunk_tokens;
                    const c1: u32 = (base_pos + m - 1) / self.kv_chunk_tokens;
                    const p0 = self.poolTilePair(slot.chunk_group_ids[c0], kv_idx);
                    const p1 = if (c1 != c0) self.poolTilePair(slot.chunk_group_ids[c1], kv_idx) else null;
                    try model.hybrid_pipes.encodeQsaKvTreeCommit(&frame, .{
                        .k_tree = tree_k,
                        .v_tree = tree_v,
                        .accepted_nodes = .{ .inline_u16 = acc_nodes[0..m] },
                        .accepted_count = m,
                        .k_tile0 = p0.k,
                        .v_tile0 = p0.v,
                        .k_tile1 = if (p1) |pp| pp.k else null,
                        .v_tile1 = if (p1) |pp| pp.v else null,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .base_pos = base_pos,
                        .chunk_tokens = self.kv_chunk_tokens,
                        .ring_capacity = 0,
                    }, true);

                    const new_kv_len = base_pos + m;
                    const blk_tok = model.config.qsa_block_size;
                    const start_blk = base_pos / blk_tok;
                    const tot_blk = (new_kv_len + blk_tok - 1) / blk_tok;
                    try model.hybrid_pipes.encodeQsaUpdateBlockSummary(&frame, .{
                        .kv_tiles = self.slotKvTilesBinding(kv_idx, slot_id),
                        .block_keys = bk_bind,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .kv_len = new_kv_len,
                        .window_size = 0,
                        .ring_capacity = 0,
                        .chunk_tokens = self.kv_chunk_tokens,
                        .block_tokens = blk_tok,
                        .start_block = start_blk,
                        .num_blocks = if (tot_blk > start_blk) tot_blk - start_blk else 1,
                    }, true);

                    try model.encodeLinearBatched(&frame, qsa.o_proj, model.attn_out_bind, model.proj_out_bind, m, true);
                },
                .gqa => |gqa| {
                    const kv_idx = self.layer_to_kv_idx[l];
                    const tree_k: metal.BufferBinding = .{ .buffer = model.scratch_buf, .offset = gqa.tree_k_off };
                    const tree_v: metal.BufferBinding = .{ .buffer = model.scratch_buf, .offset = gqa.tree_v_off };
                    try model.encodeLinearBatched(&frame, gqa.q_proj, model.x_norm_bind, model.q_bind, m, true);
                    try model.encodeLinearBatched(&frame, gqa.k_proj, model.x_norm_bind, model.k_bind, m, false);
                    try model.encodeLinearBatched(&frame, gqa.v_proj, model.x_norm_bind, model.v_bind, m, false);
                    try model.attn_pipes.encodeTreeQkNormRope(&frame, .{
                        .q = model.q_bind,
                        .k = model.k_bind,
                        .v = model.v_bind,
                        .q_norm = gqa.q_norm,
                        .k_norm = gqa.k_norm,
                        .positions = model.positions_bind,
                        .k_tree_out = tree_k,
                        .v_tree_out = tree_v,
                        .num_nodes = m,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .rms_eps = model.config.rms_norm_eps,
                        .rope_theta = model.config.rope_theta,
                        .norm_dtype = gqa.qk_norm_dtype,
                    }, true);
                    try model.attn_pipes.encodeGqaTreeVerifyAttn(&frame, .{
                        .q = model.q_bind,
                        .kv_tiles = if (base_pos > 0) self.slotKvTilesBinding(kv_idx, slot_id) else null,
                        .base_pos = base_pos,
                        .k_tree = tree_k,
                        .v_tree = tree_v,
                        .ancestor_masks = model.ancestor_masks_bind,
                        .out = model.attn_out_bind,
                        .scratch = model.gqa_scratch_bind,
                        .max_splits = attention.MAX_SPLITS,
                        .num_nodes = m,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .chunk_tokens = self.kv_chunk_tokens,
                    }, true);

                    const c0: u32 = base_pos / self.kv_chunk_tokens;
                    const c1: u32 = (base_pos + m - 1) / self.kv_chunk_tokens;
                    const p0 = self.poolTilePair(slot.chunk_group_ids[c0], kv_idx);
                    const p1 = if (c1 != c0) self.poolTilePair(slot.chunk_group_ids[c1], kv_idx) else null;
                    try model.attn_pipes.encodeKvTreeCommit(&frame, .{
                        .k_tree = tree_k,
                        .v_tree = tree_v,
                        .accepted_nodes = .{ .inline_u16 = acc_nodes[0..m] },
                        .accepted_count = m,
                        .k_tile0 = p0.k,
                        .v_tile0 = p0.v,
                        .k_tile1 = if (p1) |pp| pp.k else null,
                        .v_tile1 = if (p1) |pp| pp.v else null,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .base_pos = base_pos,
                        .chunk_tokens = self.kv_chunk_tokens,
                    }, true);

                    try model.encodeLinearBatched(&frame, gqa.o_proj, model.attn_out_bind, model.proj_out_bind, m, true);
                },
            }

            // Residual add (4-branch GR or standard)
            if (model.gr_arena) |*gr| {
                try model.hybrid_pipes.encodeGrWriteGate(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .y_delta = model.proj_out_bind,
                    .decay = model.gr_decay_bind,
                    .g_write = model.gr_g_write_bind,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                }, true);
                const grw = layer.gr.?;
                try model.hybrid_pipes.encodeGrComputeGates(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .w_read = grw.read_gate,
                    .w_write = grw.write_gate,
                    .w_decay = grw.decay_gate,
                    .g_read_out = model.gr_g_read_bind,
                    .g_write_out = model.gr_g_write_bind,
                    .decay_out = model.gr_decay_bind,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                    .weight_dtype = grw.dtype,
                }, true);
                try model.hybrid_pipes.encodeGrReadGate(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .g_read = model.gr_g_read_bind,
                    .x_in = model.x_bind,
                    .m = m,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                }, true);
            } else {
                try model.moe_pipes.encodeResidualAdd(&frame, .{
                    .x = model.x_bind,
                    .delta = model.proj_out_bind,
                    .count = m * h,
                }, true);
            }

            try model.attn_pipes.encodeRmsNorm(&frame, .{
                .x = model.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = model.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);

            try self.encodeLayerFfnBatched(&frame, l, layer, m);
        }

        if (model.gr_arena) |*gr| {
            try model.hybrid_pipes.encodeGrCollapse(&frame, .{
                .stream = gr.binding(model.scratch_buf),
                .out = model.x_bind,
                .mode = .mean,
                .m = m,
                .hidden_dim = h,
                .branch_token_stride = model.max_tree_nodes,
            }, true);
        }

        if (compute_logits) {
            const last_x_bind: metal.BufferBinding = .{
                .buffer = model.scratch_buf,
                .offset = model.x_bind.offset + @as(u64, m - 1) * h * @sizeOf(f16),
            };
            try model.attn_pipes.encodeRmsNorm(&frame, .{
                .x = last_x_bind,
                .w = model.final_norm.binding,
                .y = model.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = model.final_norm.dtype,
            }, true);
            try model.encodeLinearBatched(&frame, model.lm_head, model.x_norm_bind, model.logits_bind, 1, true);
            try model.moe_pipes.encodeArgmaxRows(&frame, .{
                .logits = model.logits_bind,
                .out_token_ids = model.out_token_id_bind,
                .num_rows = 1,
                .vocab_size = model.config.vocab_size,
                .logits_dtype = .f16,
            }, true);
        }

        const sig = try frame.commit();
        try model.compute.wait(sig, model.timeout_ms);
        slot.seq_len = base_pos + m;
        if (compute_logits) model.last_logits_row = 0;
        return if (compute_logits) model.out_token_ids_host[0] else 0;
    }

    fn encodeLayerFfnBatched(
        self: *GraphBatchRuntime,
        frame: *metal.Compute.Frame,
        l: usize,
        layer: *GraphLayerWeights,
        m: u32,
    ) GraphStepError!void {
        const model = self.model;
        const h = model.config.hidden_size;
        switch (model.config.axes.ffn_kind) {
            .dense_swiglu => {
                try model.encodeLinearBatched(frame, layer.dense_gate, model.x_norm_bind, model.dense_gate_bind, m, true);
                try model.encodeLinearBatched(frame, layer.dense_up, model.x_norm_bind, model.dense_up_bind, m, false);
                try model.moe_pipes.encodeSiluMul(frame, .{
                    .gate = model.dense_gate_bind,
                    .up = model.dense_up_bind,
                    .out = model.dense_act_bind,
                    .count = m * model.config.intermediate_size,
                }, true);
                try model.encodeLinearBatched(frame, layer.dense_down, model.dense_act_bind, model.proj_out_bind, m, true);
                if (model.gr_arena) |*gr| {
                    try model.hybrid_pipes.encodeGrWriteGate(frame, .{
                        .stream = gr.binding(model.scratch_buf),
                        .y_delta = model.proj_out_bind,
                        .decay = model.gr_decay_bind,
                        .g_write = model.gr_g_write_bind,
                        .m = m,
                        .hidden_dim = h,
                        .branch_token_stride = model.max_tree_nodes,
                    }, true);
                } else {
                    try model.moe_pipes.encodeResidualAdd(frame, .{
                        .x = model.x_bind,
                        .delta = model.proj_out_bind,
                        .count = m * h,
                    }, true);
                }
            },
            .moe_swiglu => {
                var layer_ws = model.moe_ws.?;
                layer_ws.schedule_header_off = model.layer_sched_hdr_off + @as(u64, l) * 64;
                const router_src: moe.RouterSource = if (layer.router_quant_dtype) |rqd| .{
                    .quant_weights = .{
                        .qweight = layer.router_binding,
                        .scale_bias = layer.router_scale_bias.?,
                        .bits = switch (rqd) {
                            .q2_affine => .q2,
                            .q3_affine => .q3,
                            .q4_affine => .q4,
                            else => return error.InvalidBatchSize,
                        },
                    },
                } else .{
                    .weights = .{
                        .binding = layer.router_binding,
                        .dtype = layer.router_dtype,
                    },
                };
                const residual_bind: ?metal.BufferBinding = if (model.gr_arena != null) null else model.x_bind;
                const out_bind: metal.BufferBinding = if (model.gr_arena != null) model.proj_out_bind else model.x_bind;
                const moe_args: moe.MoELayerArgs = .{
                    .x = model.x_norm_bind,
                    .router = router_src,
                    .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                    .expert_table = layer.expert_table,
                    .workspace = &layer_ws,
                    .workspace_buf = model.scratch_buf,
                    .residual = residual_bind,
                    .out = out_bind,
                    .m = m,
                    .num_experts = model.config.num_experts,
                    .top_k = model.config.num_experts_per_tok,
                    .hidden_size = h,
                    .intermediate_size = model.config.moe_intermediate_size,
                    .norm_topk_prob = model.config.norm_topk_prob,
                    .shared_expert = layer.shared_expert,
                    .shared_weights_base = if (layer.shared_expert_buf) |sb| .{ .buffer = sb, .offset = 0 } else null,
                    .shared_expert_dtype = layer.shared_expert_dtype,
                    .shared_intermediate_size = model.config.shared_expert_intermediate_size,
                    .shared_gate_weight = layer.shared_gate_weight,
                    .shared_gate_dtype = layer.shared_gate_dtype,
                };
                if (layer.is_hetero_moe) {
                    try model.moe_pipes.encodeMoELayerHetero(frame, .{
                        .base = moe_args,
                        .expert_schemes = layer.expert_dtypes_binding.?,
                        .scheme_counts = layer.expert_scheme_counts,
                    }, true);
                } else {
                    switch (layer.expert_weight_dtype) {
                        .q2_affine => try model.moe_pipes.encodeMoELayer(frame, .q2, moe_args, true),
                        .q3_affine => try model.moe_pipes.encodeMoELayer(frame, .q3, moe_args, true),
                        .q4_affine => try model.moe_pipes.encodeMoELayer(frame, .q4, moe_args, true),
                        .f16 => try model.moe_pipes.encodeMoELayerFloat(frame, .f16, moe_args, true),
                        .bf16 => try model.moe_pipes.encodeMoELayerFloat(frame, .bf16, moe_args, true),
                    }
                }
                if (model.gr_arena) |*gr| {
                    try model.hybrid_pipes.encodeGrWriteGate(frame, .{
                        .stream = gr.binding(model.scratch_buf),
                        .y_delta = model.proj_out_bind,
                        .decay = model.gr_decay_bind,
                        .g_write = model.gr_g_write_bind,
                        .m = m,
                        .hidden_dim = h,
                        .branch_token_stride = model.max_tree_nodes,
                    }, true);
                }
            },
        }
    }

    fn prefillSlotCb(ctx: *anyopaque, chunk: *const batch_sched.PrefillChunkView) batch_sched.BackendError!batch_sched.PrefillResult {
        const self: *GraphBatchRuntime = @ptrCast(@alignCast(ctx));
        return self.prefillSlot(chunk);
    }

    pub fn prefillSlot(self: *GraphBatchRuntime, chunk: *const batch_sched.PrefillChunkView) batch_sched.BackendError!batch_sched.PrefillResult {
        if (chunk.slot_id >= self.max_slots or chunk.tokens.len == 0) return error.InvalidState;
        if (chunk.start_pos + chunk.tokens.len > self.model.max_seq_len) return error.PrefillFailed;
        const slot = &self.slots[chunk.slot_id];
        if (!slot.in_use) {
            self.resetSlotState(chunk.slot_id);
            slot.in_use = true;
        }
        @memcpy(slot.prompt_tokens[chunk.start_pos .. chunk.start_pos + chunk.tokens.len], chunk.tokens);

        const use_wide = (self.gpu_prefill_mode == .wide and self.model.usesWideGpuPrefill() and self.model.effective_max_loops == 1);
        const sub_cap: usize = if (use_wide) self.model.wide_prefill_max_tokens else self.model.max_tree_nodes;
        const bs: usize = self.kv_chunk_tokens;
        var off: usize = 0;
        var last_argmax: u32 = 0;
        while (off < chunk.tokens.len) {
            const sub_pos = chunk.start_pos + @as(u32, @intCast(off));
            // Align sub-chunk end to full-block boundary when GDN + PrefixCache are active so GDN snapshots land on exact block boundaries!
            const to_boundary = bs - (@as(usize, sub_pos) % bs);
            const take = if (self.num_gdn_layers > 0 and self.prefix_cache != null)
                @min(@min(sub_cap, chunk.tokens.len - off), to_boundary)
            else
                @min(sub_cap, chunk.tokens.len - off);
            const at_chunk_end = (off + take == chunk.tokens.len);
            const is_last = at_chunk_end and chunk.is_final_chunk;
            const is_boundary = at_chunk_end and chunk.is_message_boundary_checkpoint;
            last_argmax = if (use_wide)
                self.runSlotPrefillWideChunk(
                    chunk.slot_id,
                    chunk.tokens[off .. off + take],
                    sub_pos,
                    is_last,
                    is_last,
                ) catch return error.PrefillFailed
            else
                self.runSlotPrefillSubChunk(
                    chunk.slot_id,
                    chunk.tokens[off .. off + take],
                    sub_pos,
                    is_last,
                ) catch return error.PrefillFailed;
            self.populatePrefixCache(chunk.slot_id, is_last, is_boundary);
            off += take;
        }

        if (chunk.is_final_chunk) {
            const first_tok: u32 = if (chunk.temperature > 0.0)
                self.model.sampleFromLastLogits(chunk.temperature, chunk.rng.random(), self.p_res) catch return error.PrefillFailed
            else
                last_argmax;
            return .{ .first_token = first_tok };
        }
        return .{};
    }

    fn stepDecodeBatchCb(
        ctx: *anyopaque,
        view: *const batch_sched.BatchStepView,
        out_next_tokens: []u32,
    ) batch_sched.BackendError!void {
        const self: *GraphBatchRuntime = @ptrCast(@alignCast(ctx));
        return self.stepDecodeBatch(view, out_next_tokens);
    }

    /// Advances all `B = view.batchSize()` (`1 <= B <= max_slots`) active hybrid
    /// `GDN + QSA/GQA + MoE/Dense + GR` decode slots in ONE batched `M = B` Metal 4 pass!
    pub fn stepDecodeBatch(
        self: *GraphBatchRuntime,
        view: *const batch_sched.BatchStepView,
        out_next_tokens: []u32,
    ) batch_sched.BackendError!void {
        const b_usize = view.batchSize();
        if (b_usize == 0 or b_usize > self.max_slots or out_next_tokens.len < b_usize) {
            return error.InvalidState;
        }
        const b_u32: u32 = @intCast(b_usize);
        const model = self.model;
        const h = model.config.hidden_size;
        const v_size: usize = model.config.vocab_size;

        var max_seq: u32 = 0;
        for (0..b_usize) |bi| {
            const sid: u32 = view.slot_ids[bi];
            const tok = view.input_tokens[bi];
            const pos = view.seq_positions[bi];
            if (sid >= self.max_slots or tok >= model.config.vocab_size or pos >= model.max_seq_len) {
                return error.DecodeFailed;
            }
            if (self.num_kv_layers > 0) {
                if (!self.ensureSlotWritableChunk(sid, pos / self.kv_chunk_tokens)) return error.SlotUnavailable;
            }
            model.token_ids_host[bi] = tok;
            model.positions_host[bi] = pos;
            model.pos3d_host[bi] = hybrid.Pos3D.text(pos);
            model.ancestor_masks_host[bi] = 1;
            self.gdn_slot_ids_host[bi] = sid;
            max_seq = @max(max_seq, pos + 1);
        }

        // Stage per-GQA-layer batch write slots & slot descriptors.
        for (model.layers, 0..) |layer, l| {
            switch (layer.attn) {
                .gqa => |gqa| {
                    const kv_idx = self.layer_to_kv_idx[l];
                    const w_slots = self.gqaWriteSlotsSlice(gqa.gqa_idx);
                    const g_descs = self.gqaDescsSlice(gqa.gqa_idx);
                    for (0..b_usize) |bi| {
                        const sid: u32 = view.slot_ids[bi];
                        const pos = view.seq_positions[bi];
                        const c = pos / self.kv_chunk_tokens;
                        const pair = self.poolTilePair(self.slots[sid].chunk_group_ids[c], kv_idx);
                        w_slots[bi] = attention.BatchKvWriteSlot.fromTiles(pair.k, pair.v, pos, self.kv_chunk_tokens);
                        g_descs[bi] = .{
                            .seq_len = pos + 1,
                            .tile_table_offset = sid * self.max_chunks,
                            .retired_offset = 0,
                            .num_retired_chunks = 0,
                        };
                    }
                },
                else => {},
            }
        }

        var frame = model.compute.beginFrame(model.timeout_ms) catch return error.DecodeFailed;
        errdefer frame.abort();

        model.moe_pipes.encodeEmbedLookup(&frame, model.embed.dtype, .{
            .weight = model.embed.weight,
            .scale_bias = model.embed.scale_bias,
            .tokens = .{ .token_ids = model.token_ids_bind },
            .out = model.x_bind,
            .m = b_u32,
            .vocab_size = model.config.vocab_size,
            .hidden_size = h,
        }, false) catch return error.DecodeFailed;

        if (model.gr_arena) |*gr| {
            model.hybrid_pipes.encodeGrInitStream0(&frame, .{
                .x0 = model.x_bind,
                .stream = gr.binding(model.scratch_buf),
                .m = b_u32,
                .hidden_dim = h,
                .branch_token_stride = model.max_tree_nodes,
            }, true) catch return error.DecodeFailed;
        }

        const gdn_slot_ids_bind: metal.BufferBinding = .{ .buffer = self.batch_buf, .offset = self.gdn_slot_ids_off };
        const gdn_slot_stride_f32: u32 = if (self.gdn_slot_stride_bytes > 0)
            @intCast(self.gdn_slot_stride_bytes / @sizeOf(f32))
        else
            0;

        for (model.layers, 0..) |*layer, l| {
            if (model.gr_arena) |*gr| {
                const grw = layer.gr.?;
                model.hybrid_pipes.encodeGrComputeGates(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .w_read = grw.read_gate,
                    .w_write = grw.write_gate,
                    .w_decay = grw.decay_gate,
                    .g_read_out = model.gr_g_read_bind,
                    .g_write_out = model.gr_g_write_bind,
                    .decay_out = model.gr_decay_bind,
                    .m = b_u32,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                    .weight_dtype = grw.dtype,
                }, true) catch return error.DecodeFailed;
                model.hybrid_pipes.encodeGrReadGate(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .g_read = model.gr_g_read_bind,
                    .x_in = model.x_bind,
                    .m = b_u32,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                }, true) catch return error.DecodeFailed;
            }

            model.attn_pipes.encodeRmsNorm(&frame, .{
                .x = model.x_bind,
                .w = layer.input_layernorm.binding,
                .y = model.x_norm_bind,
                .num_rows = b_u32,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true) catch return error.DecodeFailed;

            switch (layer.attn) {
                .gdn => |g| {
                    model.encodeLinearBatched(&frame, g.q_proj, model.x_norm_bind, model.q_bind, b_u32, true) catch return error.DecodeFailed;
                    model.encodeLinearBatched(&frame, g.k_proj, model.x_norm_bind, model.k_bind, b_u32, false) catch return error.DecodeFailed;
                    model.encodeLinearBatched(&frame, g.v_proj, model.x_norm_bind, model.v_bind, b_u32, false) catch return error.DecodeFailed;
                    model.encodeLinearBatched(&frame, g.g_proj, model.x_norm_bind, model.gdn_z_bind, b_u32, false) catch return error.DecodeFailed;
                    model.hybrid_pipes.encodeGdnComputeGates(&frame, .{
                        .x = model.x_norm_bind,
                        .w_a = g.a_proj,
                        .w_b = g.b_proj,
                        .a_log = g.a_log,
                        .dt_bias = g.dt_bias,
                        .g_out = .{ .buffer = model.scratch_buf, .offset = g.tree_g_off },
                        .beta_out = .{ .buffer = model.scratch_buf, .offset = g.tree_beta_off },
                        .m = b_u32,
                        .hidden_dim = h,
                        .num_v_heads = model.config.linear_num_value_heads,
                        .weight_dtype = g.ab_dtype,
                    }, false) catch return error.DecodeFailed;

                    var q_eff = model.q_bind;
                    var k_eff = model.k_bind;
                    var v_eff = model.v_bind;
                    if (g.conv1d_w) |cw| {
                        const qk_d: u64 = model.config.gdnQkDim();
                        const v_d: u64 = model.config.gdnVDim();
                        for (0..b_usize) |bi| {
                            const sid: u32 = view.slot_ids[bi];
                            const conv_bind = self.gdnSlotConvBinding(g.gdn_idx, sid);
                            model.hybrid_pipes.encodeGdnConv1dSilu(&frame, .{
                                .q_in = .{ .buffer = model.q_bind.buffer, .offset = model.q_bind.offset + @as(u64, bi) * qk_d * 2 },
                                .k_in = .{ .buffer = model.k_bind.buffer, .offset = model.k_bind.offset + @as(u64, bi) * qk_d * 2 },
                                .v_in = .{ .buffer = model.v_bind.buffer, .offset = model.v_bind.offset + @as(u64, bi) * v_d * 2 },
                                .conv_w = cw,
                                .conv_state_in = conv_bind,
                                .conv_state_out = conv_bind,
                                .q_out = .{ .buffer = model.q_conv_bind.buffer, .offset = model.q_conv_bind.offset + @as(u64, bi) * qk_d * 2 },
                                .k_out = .{ .buffer = model.scratch_buf, .offset = g.tree_k_off + @as(u64, bi) * qk_d * 2 },
                                .v_out = .{ .buffer = model.scratch_buf, .offset = g.tree_v_off + @as(u64, bi) * v_d * 2 },
                                .m = 1,
                                .gdn_qk = model.config.gdnQkDim(),
                                .gdn_v = model.config.gdnVDim(),
                                .conv_k = g.conv_k,
                                .is_tree = false,
                                .write_state = true,
                            }, bi == 0) catch return error.DecodeFailed;
                        }
                        q_eff = model.q_conv_bind;
                        k_eff = .{ .buffer = model.scratch_buf, .offset = g.tree_k_off };
                        v_eff = .{ .buffer = model.scratch_buf, .offset = g.tree_v_off };
                    }

                    // ONE batched `ami_gdn_batch_decode` dispatch updating all `B` slots concurrently!
                    const pool_bind = self.gdnLayerPoolBinding(g.gdn_idx);
                    model.hybrid_pipes.encodeGdnBatchDecode(&frame, .{
                        .q = q_eff,
                        .k = k_eff,
                        .v = v_eff,
                        .g = .{ .buffer = model.scratch_buf, .offset = g.tree_g_off },
                        .beta = .{ .buffer = model.scratch_buf, .offset = g.tree_beta_off },
                        .z = model.gdn_z_bind,
                        .norm_w = if (g.out_norm) |n| n.binding else null,
                        .norm_dtype = if (g.out_norm) |n| n.dtype else .f16,
                        .state_in_pool = pool_bind,
                        .state_out_pool = pool_bind,
                        .out = model.attn_out_bind,
                        .state_in_slots = gdn_slot_ids_bind,
                        .batch_size = b_u32,
                        .num_k_heads = model.config.linear_num_key_heads,
                        .num_v_heads = model.config.linear_num_value_heads,
                        .head_k_dim = model.config.linear_key_head_dim,
                        .head_v_dim = model.config.linear_value_head_dim,
                        .slot_stride_f32 = gdn_slot_stride_f32,
                        .rms_eps = model.config.rms_norm_eps,
                    }, true) catch return error.DecodeFailed;

                    model.encodeLinearBatched(&frame, g.o_proj, model.attn_out_bind, model.proj_out_bind, b_u32, true) catch return error.DecodeFailed;
                },
                .qsa => |*qsa| {
                    const kv_idx = self.layer_to_kv_idx[l];
                    const tree_k: metal.BufferBinding = .{ .buffer = model.scratch_buf, .offset = qsa.tree_k_off };
                    const tree_v: metal.BufferBinding = .{ .buffer = model.scratch_buf, .offset = qsa.tree_v_off };
                    model.encodeLinearBatched(&frame, qsa.q_proj, model.x_norm_bind, model.q_bind, b_u32, true) catch return error.DecodeFailed;
                    model.encodeLinearBatched(&frame, qsa.k_proj, model.x_norm_bind, model.k_bind, b_u32, false) catch return error.DecodeFailed;
                    model.encodeLinearBatched(&frame, qsa.v_proj, model.x_norm_bind, model.v_bind, b_u32, false) catch return error.DecodeFailed;

                    const mrope_rot: u32 = (model.config.mrope_section.section[0] + model.config.mrope_section.section[1] + model.config.mrope_section.section[2]) * 2;
                    const eff_rot_dim: u32 = if (mrope_rot > 0 and mrope_rot <= model.config.head_dim) mrope_rot else model.config.head_dim;
                    model.hybrid_pipes.encodeQsaTreeRope(&frame, .{
                        .q = model.q_bind,
                        .k = model.k_bind,
                        .v = model.v_bind,
                        .q_norm = qsa.q_norm,
                        .k_norm = qsa.k_norm,
                        .positions_3d = model.pos3d_bind,
                        .k_tree_out = tree_k,
                        .v_tree_out = tree_v,
                        .num_nodes = b_u32,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .rotary_dim = eff_rot_dim,
                        .mrope = model.config.mrope_section,
                        .rms_eps = model.config.rms_norm_eps,
                        .rope_theta = model.config.rope_theta,
                        .yarn = model.config.yarn,
                        .norm_dtype = qsa.qk_norm_dtype,
                    }, true) catch return error.DecodeFailed;

                    const q_row_bytes: u64 = @as(u64, model.config.qDim()) * @sizeOf(f16);
                    const kv_row_bytes: u64 = @as(u64, model.config.kvDim()) * @sizeOf(f16);
                    const blk_tok = model.config.qsa_block_size;
                    for (0..b_usize) |bi| {
                        const sid: u32 = view.slot_ids[bi];
                        const pos = view.seq_positions[bi];
                        const bk_bind = self.qsaSlotBlockKeysBinding(qsa.qsa_idx, sid);
                        const q_sub: metal.BufferBinding = .{ .buffer = model.q_bind.buffer, .offset = model.q_bind.offset + @as(u64, bi) * q_row_bytes };
                        const out_sub: metal.BufferBinding = .{ .buffer = model.attn_out_bind.buffer, .offset = model.attn_out_bind.offset + @as(u64, bi) * q_row_bytes };
                        const kt_sub: metal.BufferBinding = .{ .buffer = tree_k.buffer, .offset = tree_k.offset + @as(u64, bi) * kv_row_bytes };
                        const vt_sub: metal.BufferBinding = .{ .buffer = tree_v.buffer, .offset = tree_v.offset + @as(u64, bi) * kv_row_bytes };
                        model.hybrid_pipes.encodeQsaAttention(&frame, .{
                            .q = q_sub,
                            .kv_tiles = if (pos > 0) self.slotKvTilesBinding(kv_idx, sid) else null,
                            .block_keys = bk_bind,
                            .index_buf = &qsa.index_buf,
                            .index_metal_buf = model.scratch_buf,
                            .k_tree = kt_sub,
                            .v_tree = vt_sub,
                            .ancestor_masks = model.ancestor_masks_bind,
                            .out = out_sub,
                            .num_nodes = 1,
                            .num_q_heads = model.config.num_attention_heads,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .kv_len = pos,
                            .window_size = 0,
                            .ring_capacity = 0,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .block_tokens = blk_tok,
                            .top_k_blocks = model.config.qsa_top_k_blocks,
                            .reuse_block_indices = false,
                        }, true) catch return error.DecodeFailed;

                        const c = pos / self.kv_chunk_tokens;
                        const pair = self.poolTilePair(self.slots[sid].chunk_group_ids[c], kv_idx);
                        const acc1 = [_]u16{0};
                        model.hybrid_pipes.encodeQsaKvTreeCommit(&frame, .{
                            .k_tree = kt_sub,
                            .v_tree = vt_sub,
                            .accepted_nodes = .{ .inline_u16 = &acc1 },
                            .accepted_count = 1,
                            .k_tile0 = pair.k,
                            .v_tile0 = pair.v,
                            .k_tile1 = null,
                            .v_tile1 = null,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .base_pos = pos,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .ring_capacity = 0,
                        }, true) catch return error.DecodeFailed;

                        model.hybrid_pipes.encodeQsaUpdateBlockSummary(&frame, .{
                            .kv_tiles = self.slotKvTilesBinding(kv_idx, sid),
                            .block_keys = bk_bind,
                            .num_kv_heads = model.config.num_key_value_heads,
                            .head_dim = model.config.head_dim,
                            .kv_len = pos + 1,
                            .window_size = 0,
                            .ring_capacity = 0,
                            .chunk_tokens = self.kv_chunk_tokens,
                            .block_tokens = blk_tok,
                            .start_block = pos / blk_tok,
                            .num_blocks = 1,
                        }, true) catch return error.DecodeFailed;
                    }

                    model.encodeLinearBatched(&frame, qsa.o_proj, model.attn_out_bind, model.proj_out_bind, b_u32, true) catch return error.DecodeFailed;
                },
                .gqa => |gqa| {
                    const kv_idx = self.layer_to_kv_idx[l];
                    model.encodeLinearBatched(&frame, gqa.q_proj, model.x_norm_bind, model.q_bind, b_u32, true) catch return error.DecodeFailed;
                    model.encodeLinearBatched(&frame, gqa.k_proj, model.x_norm_bind, model.k_bind, b_u32, false) catch return error.DecodeFailed;
                    model.encodeLinearBatched(&frame, gqa.v_proj, model.x_norm_bind, model.v_bind, b_u32, false) catch return error.DecodeFailed;

                    model.attn_pipes.encodeBatchQkNormRopeKvWrite(&frame, .{
                        .q = model.q_bind,
                        .k = model.k_bind,
                        .v = model.v_bind,
                        .q_norm = gqa.q_norm,
                        .k_norm = gqa.k_norm,
                        .slots = self.gqaWriteSlotsBinding(gqa.gqa_idx),
                        .batch_size = b_u32,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .chunk_tokens = self.kv_chunk_tokens,
                        .rms_eps = model.config.rms_norm_eps,
                        .rope_theta = model.config.rope_theta,
                        .norm_dtype = gqa.qk_norm_dtype,
                    }, true) catch return error.DecodeFailed;

                    model.attn_pipes.encodeBatchGqaDecodeAttn(&frame, .{
                        .q = model.q_bind,
                        .kv_tiles = self.layerKvTilesBinding(kv_idx),
                        .slots = self.gqaDescsBinding(gqa.gqa_idx),
                        .out = model.attn_out_bind,
                        .scratch = self.gqa_batch_scratch_bind,
                        .max_splits = attention.MAX_SPLITS,
                        .batch_size = b_u32,
                        .num_q_heads = model.config.num_attention_heads,
                        .num_kv_heads = model.config.num_key_value_heads,
                        .head_dim = model.config.head_dim,
                        .max_seq_len = max_seq,
                        .chunk_tokens = self.kv_chunk_tokens,
                    }, true) catch return error.DecodeFailed;

                    model.encodeLinearBatched(&frame, gqa.o_proj, model.attn_out_bind, model.proj_out_bind, b_u32, true) catch return error.DecodeFailed;
                },
            }

            if (model.gr_arena) |*gr| {
                model.hybrid_pipes.encodeGrWriteGate(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .y_delta = model.proj_out_bind,
                    .decay = model.gr_decay_bind,
                    .g_write = model.gr_g_write_bind,
                    .m = b_u32,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                }, true) catch return error.DecodeFailed;
                const grw = layer.gr.?;
                model.hybrid_pipes.encodeGrComputeGates(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .w_read = grw.read_gate,
                    .w_write = grw.write_gate,
                    .w_decay = grw.decay_gate,
                    .g_read_out = model.gr_g_read_bind,
                    .g_write_out = model.gr_g_write_bind,
                    .decay_out = model.gr_decay_bind,
                    .m = b_u32,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                    .weight_dtype = grw.dtype,
                }, true) catch return error.DecodeFailed;
                model.hybrid_pipes.encodeGrReadGate(&frame, .{
                    .stream = gr.binding(model.scratch_buf),
                    .g_read = model.gr_g_read_bind,
                    .x_in = model.x_bind,
                    .m = b_u32,
                    .hidden_dim = h,
                    .branch_token_stride = model.max_tree_nodes,
                }, true) catch return error.DecodeFailed;
            } else {
                model.moe_pipes.encodeResidualAdd(&frame, .{
                    .x = model.x_bind,
                    .delta = model.proj_out_bind,
                    .count = b_u32 * h,
                }, true) catch return error.DecodeFailed;
            }

            model.attn_pipes.encodeRmsNorm(&frame, .{
                .x = model.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = model.x_norm_bind,
                .num_rows = b_u32,
                .hidden_dim = h,
                .eps = model.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true) catch return error.DecodeFailed;

            self.encodeLayerFfnBatched(&frame, l, layer, b_u32) catch return error.DecodeFailed;
        }

        if (model.gr_arena) |*gr| {
            model.hybrid_pipes.encodeGrCollapse(&frame, .{
                .stream = gr.binding(model.scratch_buf),
                .out = model.x_bind,
                .mode = .mean,
                .m = b_u32,
                .hidden_dim = h,
                .branch_token_stride = model.max_tree_nodes,
            }, true) catch return error.DecodeFailed;
        }

        model.attn_pipes.encodeRmsNorm(&frame, .{
            .x = model.x_bind,
            .w = model.final_norm.binding,
            .y = model.x_norm_bind,
            .num_rows = b_u32,
            .hidden_dim = h,
            .eps = model.config.rms_norm_eps,
            .w_dtype = model.final_norm.dtype,
        }, true) catch return error.DecodeFailed;
        model.encodeLinearBatched(&frame, model.lm_head, model.x_norm_bind, model.logits_bind, b_u32, true) catch return error.DecodeFailed;
        model.moe_pipes.encodeArgmaxRows(&frame, .{
            .logits = model.logits_bind,
            .out_token_ids = model.out_token_id_bind,
            .num_rows = b_u32,
            .vocab_size = model.config.vocab_size,
            .logits_dtype = .f16,
        }, true) catch return error.DecodeFailed;

        const sig = frame.commit() catch return error.DecodeFailed;
        model.compute.wait(sig, model.timeout_ms) catch return error.DecodeFailed;

        for (0..b_usize) |bi| {
            const sid: u32 = view.slot_ids[bi];
            const temp = view.temperatures[bi];
            if (temp > 0.0) {
                const row = model.logits_host[bi * v_size .. (bi + 1) * v_size];
                accept_mod.softmaxRow(f16, row, temp, self.p_res) catch return error.DecodeFailed;
                out_next_tokens[bi] = accept_mod.sampleCategorical(self.p_res[0..v_size], view.rngs[bi].random());
            } else {
                out_next_tokens[bi] = model.out_token_ids_host[bi];
            }
            self.slots[sid].seq_len = view.seq_positions[bi] + 1;
        }

        self.decode_batch_steps += 1;
        self.decode_batch_tokens += b_usize;
    }

    fn releaseSlotCb(ctx: *anyopaque, slot_id: u16, prefix_handle: u64) void {
        _ = prefix_handle;
        const self: *GraphBatchRuntime = @ptrCast(@alignCast(ctx));
        self.resetSlotState(slot_id);
    }
};

// ===========================================================================
// Section 6: Loader Helpers (Step-LoRA Parsing, DType Conversion, Expert Weights)
// ===========================================================================

fn amfToWeightDType(dt: amf.DType) ?quant.WeightDType {
    return switch (dt) {
        .q2_affine => .q2_affine,
        .q3_affine => .q3_affine,
        .q4_affine => .q4_affine,
        .f16 => .f16,
        .bf16 => .bf16,
        else => null,
    };
}

fn readFloatWeightAt(raw: []const u8, dt: quant.FloatDType, idx: usize) f32 {
    const bits = std.mem.readInt(u16, raw[idx * 2 ..][0..2], .little);
    return quant.floatBitsToF32(dt, bits);
}

fn makeExpertWeights(
    wdt: quant.WeightDType,
    gate_off: u64,
    gate_params_off: u64,
    up_off: u64,
    up_params_off: u64,
    down_off: u64,
    down_params_off: u64,
) moe.ExpertWeights {
    if (wdt.isQuantized()) {
        return .{
            .gate_qweight_offset = gate_off,
            .gate_sb_offset = gate_off + gate_params_off,
            .up_qweight_offset = up_off,
            .up_sb_offset = up_off + up_params_off,
            .down_qweight_offset = down_off,
            .down_sb_offset = down_off + down_params_off,
        };
    }
    return .{
        .gate_qweight_offset = gate_off,
        .gate_sb_offset = 0,
        .up_qweight_offset = up_off,
        .up_sb_offset = 0,
        .down_qweight_offset = down_off,
        .down_sb_offset = 0,
    };
}

const ParsedAdapterName = struct {
    step: u32,
    layer: u32,
    proj: AdapterProjKind,
    part: enum { lora_a, lora_b, alpha },
};

fn parseLoopAdapterName(name: []const u8) ?ParsedAdapterName {
    // Expected: `loop.step.{r}.layers.{l}.{proj}.{lora_a|lora_b|alpha}`
    const prefix = "loop.step.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    var rest = name[prefix.len..];
    const dot1 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const step = std.fmt.parseInt(u32, rest[0..dot1], 10) catch return null;
    rest = rest[dot1 + 1 ..];
    const l_pref = "layers.";
    if (!std.mem.startsWith(u8, rest, l_pref)) return null;
    rest = rest[l_pref.len..];
    const dot2 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot2], 10) catch return null;
    rest = rest[dot2 + 1 ..];
    const dot3 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const proj = AdapterProjKind.parse(rest[0..dot3]) orelse return null;
    const tail = rest[dot3 + 1 ..];
    const part: @TypeOf(@as(ParsedAdapterName, undefined).part) = if (std.mem.eql(u8, tail, "lora_a"))
        .lora_a
    else if (std.mem.eql(u8, tail, "lora_b"))
        .lora_b
    else if (std.mem.eql(u8, tail, "alpha"))
        .alpha
    else
        return null;
    return .{ .step = step, .layer = layer, .proj = proj, .part = part };
}

fn loadStepLoraAdapters(
    allocator: std.mem.Allocator,
    main_source: GraphModel.AmfSource,
    main_entry_slices: []const []u8,
    main_base_ptr: usize,
    external_path: ?[:0]const u8,
    config: GraphConfig,
    effective_max_loops: u32,
    max_m: u32,
    wired_budget: budget_mod.Budget,
    total_wired_bytes: *u64,
) GraphLoadError!*LoopAdapterRuntime {
    var ext_file: ?amf.AmfFile = null;
    defer if (ext_file) |*f| f.close(allocator);
    var ext_arena: ?arena_mod.WiredArena = null;
    errdefer if (ext_arena) |*ea| ea.deinit();
    var ext_slices: [][]u8 = &.{};
    defer if (ext_slices.len > 0) allocator.free(ext_slices);

    if (external_path) |ep| {
        var f = try amf.AmfFile.open(allocator, ep);
        errdefer f.close(allocator);
        var need_bytes: usize = 0;
        for (f.parsed.entries) |*e| {
            need_bytes += try page.alignUp(@intCast(e.blob_bytes));
        }
        const aligned = try page.alignUp(@max(need_bytes, page.APPLE_PAGE_SIZE));
        try wired_budget.checkPlan(total_wired_bytes.* + aligned);
        var ea = try arena_mod.WiredArena.init(aligned, wired_budget);
        errdefer ea.deinit();
        const slices = try allocator.alloc([]u8, f.parsed.entries.len);
        errdefer allocator.free(slices);
        for (f.parsed.entries, 0..) |*e, i| {
            const len: usize = @intCast(e.blob_bytes);
            const raw = try f.loadBlob(e, &ea);
            slices[i] = raw[0..len];
        }
        total_wired_bytes.* += aligned;
        ext_file = f;
        ext_arena = ea;
        ext_slices = slices;
    }

    const use_entries: []const amf.TensorEntry = if (ext_file) |*f| f.parsed.entries else main_source.entries();
    const use_slices: []const []u8 = if (ext_file != null) ext_slices else main_entry_slices;
    const use_base: usize = if (ext_file != null) @intFromPtr(use_entries.ptr) else main_base_ptr;
    const findIn = struct {
        fn call(ext: ?*const amf.AmfFile, main_src: GraphModel.AmfSource, name: []const u8) ?*const amf.TensorEntry {
            if (ext) |f| return f.parsed.find(name);
            return main_src.find(name);
        }
    }.call;

    // Discover max_steps, max_k, max_n, max_rank across all `loop_adapter` tensors.
    var max_steps: u32 = @max(effective_max_loops, config.axes.loop_schedule.max_loops);
    var max_k: usize = @max(config.hidden_size, @max(config.qDim(), config.gdnVDim()));
    var max_n: usize = max_k;
    var max_rank: usize = sme.GROUP_SIZE;

    for (use_entries) |*e| {
        if ((try e.tensorKind()) != .loop_adapter) continue;
        const p = parseLoopAdapterName(e.nameSlice()) orelse continue;
        if (p.step + 1 > max_steps) max_steps = p.step + 1;
        if (p.part == .lora_a) {
            max_rank = @max(max_rank, @as(usize, @intCast(e.shape[0])));
            max_k = @max(max_k, @as(usize, @intCast(e.shape[1])));
        } else if (p.part == .lora_b) {
            max_n = @max(max_n, @as(usize, @intCast(e.shape[0])));
        }
    }

    const total_slots = @as(usize, max_steps) * config.num_hidden_layers * MAX_ADAPTER_PROJS_PER_LAYER;
    const table = try allocator.alloc(?sme.SmeStepLoraAdapter, total_slots);
    errdefer allocator.free(table);
    @memset(table, null);

    var nm_buf: [160]u8 = undefined;
    const ext_ptr: ?*const amf.AmfFile = if (ext_file) |*f| f else null;
    for (use_entries) |*e| {
        if ((try e.tensorKind()) != .loop_adapter) continue;
        const p = parseLoopAdapterName(e.nameSlice()) orelse continue;
        if (p.part != .lora_a) continue;
        if (p.layer >= config.num_hidden_layers) continue;

        const proj_name = @tagName(p.proj);
        const b_nm = std.fmt.bufPrint(&nm_buf, "loop.step.{d}.layers.{d}.{s}.lora_b", .{ p.step, p.layer, proj_name }) catch unreachable;
        const b_e = findIn(ext_ptr, main_source, b_nm) orelse return error.MissingTensor;
        const al_nm = std.fmt.bufPrint(&nm_buf, "loop.step.{d}.layers.{d}.{s}.alpha", .{ p.step, p.layer, proj_name }) catch unreachable;
        const al_e = findIn(ext_ptr, main_source, al_nm) orelse return error.MissingTensor;

        const a_idx = (@intFromPtr(e) - use_base) / @sizeOf(amf.TensorEntry);
        const b_idx = (@intFromPtr(b_e) - use_base) / @sizeOf(amf.TensorEntry);
        const al_idx = (@intFromPtr(al_e) - use_base) / @sizeOf(amf.TensorEntry);
        if (use_slices[al_idx].len < 4) return error.TensorShapeMismatch;
        const alpha: f32 = @bitCast(std.mem.readInt(u32, use_slices[al_idx][0..4], .little));
        if (p.step == 0 and config.axes.loop_schedule.enforce_zero_alpha_r0 and alpha != 0.0) {
            return error.NonZeroLoopZeroAlpha;
        }
        const adapter = try sme.SmeStepLoraAdapter.fromAmfEntries(e, use_slices[a_idx], b_e, use_slices[b_idx], alpha);
        const slot_idx = (@as(usize, p.step) * config.num_hidden_layers + p.layer) * MAX_ADAPTER_PROJS_PER_LAYER + @backingInt(p.proj);
        table[slot_idx] = adapter;
    }

    var scratch = try sme.StepLoraScratch.init(max_m, @min(max_rank, sme.MAX_STEP_LORA_RANK), max_k, wired_budget);
    errdefer scratch.deinit();
    total_wired_bytes.* += scratch.totalWiredBytes();

    const delta_f32 = try allocator.alloc(f32, @as(usize, max_m) * max_n);
    errdefer allocator.free(delta_f32);
    @memset(delta_f32, 0.0);

    var disp = try sme.SmeDispatcher.init(allocator, .{});
    errdefer disp.deinit();

    const art = try allocator.create(LoopAdapterRuntime);
    art.* = .{
        .external_arena = ext_arena,
        .dispatcher = disp,
        .scratch = scratch,
        .delta_f32 = delta_f32,
        .max_steps = max_steps,
        .num_layers = config.num_hidden_layers,
        .entries = table,
    };
    ext_arena = null; // ownership transferred
    return art;
}

// ===========================================================================
// Section 7: Unit Tests — CPU FP32 Parity, Qwen4/512E, LoopLM, Step-LoRA &
//            All Four ADR-0019 Hybrid-Sparse Loop Isolations
// ===========================================================================

const TestTensorBlob = struct {
    name: []const u8,
    kind: amf.TensorKind,
    dtype: amf.DType,
    layout: amf.TensorLayout = .row_major,
    layer: u16 = amf.NO_LAYER,
    expert: u16 = amf.NO_EXPERT,
    shape: [4]u32,
    ndim: u8,
    bytes: []u8,
    data_bytes_override: u64 = 0,
    params_offset: u64 = 0,
    params_bytes: u64 = 0,
};

fn buildTestAmf(allocator: std.mem.Allocator, model_id: []const u8, blobs: []const TestTensorBlob) ![]u8 {
    const dir_offset: u64 = page.APPLE_PAGE_SIZE;
    const raw_dir_bytes: usize = blobs.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: u64 = try page.alignUp(@max(raw_dir_bytes, page.APPLE_PAGE_SIZE));
    var offset: u64 = dir_offset + dir_bytes;

    var entries = try allocator.alloc(amf.TensorEntry, blobs.len);
    defer allocator.free(entries);

    for (blobs, 0..) |b, i| {
        const data_len: u64 = if (b.data_bytes_override > 0) b.data_bytes_override else b.bytes.len;
        const blob_len: u64 = try page.alignUp(@max(b.bytes.len, 1));
        var e = amf.TensorEntry.zeroed(b.name, b.kind, b.dtype);
        e.setLayoutAndParamKind(b.layout, .affine_scale_bias);
        e.quant_bits = switch (b.dtype) {
            .q4_affine => 4,
            .q3_affine => 3,
            .q2_affine => 2,
            else => 0,
        };
        e.group_size = if (e.quant_bits > 0) 64 else 0;
        e.layer = b.layer;
        e.expert = b.expert;
        e.rank = b.ndim;
        e.shape = b.shape;
        e.blob_offset = offset;
        e.blob_bytes = blob_len;
        e.data_bytes = data_len;
        e.params_offset = b.params_offset;
        e.params_bytes = b.params_bytes;
        entries[i] = e;
        offset += blob_len;
    }

    const total: usize = @intCast(offset);
    const out = try allocator.alloc(u8, total);
    @memset(out, 0);

    for (blobs, 0..) |b, i| {
        const bo: usize = @intCast(entries[i].blob_offset);
        const bl: usize = @intCast(entries[i].blob_bytes);
        @memcpy(out[bo .. bo + b.bytes.len], b.bytes);
        entries[i].blob_crc32 = std.hash.Crc32.hash(out[bo .. bo + bl]);
    }

    const dir_slice = std.mem.sliceAsBytes(entries);
    var hdr: amf.Header = std.mem.zeroes(amf.Header);
    hdr.magic = amf.MAGIC;
    hdr.version = amf.FORMAT_VERSION;
    hdr.flags = 0;
    hdr.page_size = page.APPLE_PAGE_SIZE;
    hdr.checksum_algo = amf.CHECKSUM_CRC32_IEEE;
    hdr.tensor_count = @intCast(blobs.len);
    hdr.directory_offset = dir_offset;
    hdr.directory_bytes = dir_bytes;
    hdr.directory_crc32 = std.hash.Crc32.hash(dir_slice);
    const id_len = @min(model_id.len, hdr.model_id.len - 1);
    @memcpy(hdr.model_id[0..id_len], model_id[0..id_len]);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);

    @memcpy(out[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    @memcpy(out[@intCast(dir_offset)..][0..dir_slice.len], dir_slice);
    return out;
}

fn fillDeterministicF16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const v: f32 = (r.float(f32) * 2.0 - 1.0) * amp;
        const bits: u16 = @bitCast(@as(f16, @floatCast(v)));
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn fillDeterministicBf16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const v: f32 = (r.float(f32) * 2.0 - 1.0) * amp;
        const bits = quant.f32ToFloatBits(.bf16, v);
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn fillConstantBf16(allocator: std.mem.Allocator, count: usize, val: f32) ![]u8 {
    const out = try allocator.alloc(u8, count * 2);
    const bits = quant.f32ToFloatBits(.bf16, val);
    for (0..count) |i| {
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn fillDeterministicF32(allocator: std.mem.Allocator, seed: u64, count: usize, lo: f32, hi: f32) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try allocator.alloc(u8, count * 4);
    for (0..count) |i| {
        const v: f32 = lo + r.float(f32) * (hi - lo);
        std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little);
    }
    return out;
}

fn fillDeterministicStepLoraSme2F16(
    allocator: std.mem.Allocator,
    seed: u64,
    n_rows: usize,
    k_cols: usize,
    active_rows: usize,
    active_cols: usize,
    amp: f32,
) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const row_major = try allocator.alloc(u16, n_rows * k_cols);
    defer allocator.free(row_major);
    @memset(row_major, 0);
    for (0..active_rows) |row| {
        for (0..active_cols) |col| {
            const v: f32 = (r.float(f32) * 2.0 - 1.0) * amp;
            row_major[row * k_cols + col] = @bitCast(@as(f16, @floatCast(v)));
        }
    }
    const packed_len = try sme.packedBlobBytes(.f16, n_rows, k_cols);
    const packed_buf = try allocator.alloc(u8, packed_len);
    try sme.packDensePanel64(.f16, row_major, n_rows, k_cols, packed_buf);
    return packed_buf;
}

/// CPU FP32 reference helpers for exact step-by-step numerical parity checks.
fn cpuRmsNorm(x: []const f32, w_bytes: []const u8, w_dt: quant.FloatDType, eps: f32, out: []f32) void {
    var ss: f32 = 0.0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (0..x.len) |i| {
        out[i] = x[i] * inv * readFloatWeightAt(w_bytes, w_dt, i);
    }
}

fn cpuLinearF16(x: []const f32, w_bytes: []const u8, w_dt: quant.FloatDType, n: usize, k: usize, out: []f32) void {
    for (0..n) |row| {
        var acc: f32 = 0.0;
        for (0..k) |col| {
            acc += x[col] * readFloatWeightAt(w_bytes, w_dt, row * k + col);
        }
        out[row] = acc;
    }
}

fn cpuSilu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn cpuSigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn fillDeterministicQ4(
    allocator: std.mem.Allocator,
    seed: u64,
    n: usize,
    k: usize,
    amp: f32,
) !struct { bytes: []u8, data_bytes: u64, params_offset: u64, params_bytes: u64 } {
    const groups_per_row = k / 64;
    const q_bytes = n * (k / 2);
    const p_off: usize = amf.align64(q_bytes);
    const p_bytes = n * groups_per_row * 4;
    const out = try allocator.alloc(u8, p_off + p_bytes);
    @memset(out, 0);
    // Fast deterministic SplitMix64 fill for nibbles so large `>= 50 MiB` bench tensors build in milliseconds.
    var s: u64 = seed ^ 0x9e3779b97f4a7c15;
    const words = std.mem.bytesAsSlice(u64, out[0 .. (q_bytes / 8) * 8]);
    for (words) |*w| {
        s +%= 0x9e3779b97f4a7c15;
        var z = s;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        w.* = z ^ (z >> 31);
    }
    const rem_start = words.len * 8;
    for (rem_start..q_bytes) |i| {
        s +%= 0x9e3779b97f4a7c15;
        out[i] = @truncate(s >> 24);
    }
    const scale_f16: u16 = @bitCast(@as(f16, @floatCast(amp / 7.5)));
    const bias_f16: u16 = @bitCast(@as(f16, @floatCast(-amp)));
    var pi: usize = p_off;
    for (0..n * groups_per_row) |_| {
        std.mem.writeInt(u16, out[pi..][0..2], scale_f16, .little);
        std.mem.writeInt(u16, out[pi + 2 ..][0..2], bias_f16, .little);
        pi += 4;
    }
    return .{
        .bytes = out,
        .data_bytes = @intCast(q_bytes),
        .params_offset = @intCast(p_off),
        .params_bytes = @intCast(p_bytes),
    };
}

pub const SynthQwen4BenchSpec = struct {
    hidden_size: u32 = 64,
    vocab_size: u32 = 128,
    num_attention_heads: u32 = 2,
    num_key_value_heads: u32 = 1,
    head_dim: u32 = 64,
    gdn_kh: u32 = 2,
    gdn_vh: u32 = 2,
    gdn_kd: u32 = 32,
    gdn_vd: u32 = 32,
    moe_inter: u32 = 64,
    shared_inter: u32 = 64,
    num_experts: u32 = 8,
    top_k: u32 = 2,
    cold_experts_per_layer: u32 = 0,
    max_loops: u32 = 1,
    ple_num_shards: u32 = 0,
    ple_rows: u64 = 64,
    use_q4_experts: bool = false,
    include_steplora: bool = false,
    core_start_layer: ?u32 = null,
    tmp_dir: ?[]const u8 = null,
};

pub const SynthQwen4Bundle = struct {
    cfg: GraphConfig,
    amf_bytes: []u8,
    ple_paths: [][:0]u8,
    amf_file_path: ?[:0]u8,

    pub fn deinit(self: *SynthQwen4Bundle, allocator: std.mem.Allocator) void {
        self.cfg.deinit(allocator);
        allocator.free(self.amf_bytes);
        for (self.ple_paths) |p| allocator.free(p);
        allocator.free(self.ple_paths);
        if (self.amf_file_path) |p| allocator.free(p);
    }
};

pub fn buildSynthQwen4Bundle(
    allocator: std.mem.Allocator,
    spec: SynthQwen4BenchSpec,
) !SynthQwen4Bundle {
    const h: u32 = spec.hidden_size;
    const vocab: u32 = spec.vocab_size;
    const n_layers: u32 = 4; // 3:1 GDN:QSA (L0,L1,L2 = GDN, L3 = QSA)
    const n_heads: u32 = spec.num_attention_heads;
    const n_kv: u32 = spec.num_key_value_heads;
    const head_dim: u32 = spec.head_dim;
    const gdn_kh: u32 = spec.gdn_kh;
    const gdn_vh: u32 = spec.gdn_vh;
    const gdn_kd: u32 = spec.gdn_kd;
    const gdn_vd: u32 = spec.gdn_vd;
    const moe_inter: u32 = spec.moe_inter;
    const sh_inter: u32 = spec.shared_inter;
    const ple_rows: u64 = spec.ple_rows;
    const num_experts = spec.num_experts;
    const top_k = spec.top_k;
    const cold_experts_per_layer = spec.cold_experts_per_layer;
    const max_loops = spec.max_loops;
    const ple_num_shards = spec.ple_num_shards;
    const tmp_dir = spec.tmp_dir;

    const layer_attns = try allocator.alloc(AttentionKind, n_layers);
    layer_attns[0] = .gated_deltanet;
    layer_attns[1] = .gated_deltanet;
    layer_attns[2] = .gated_deltanet;
    layer_attns[3] = .qsa_sparse;

    const core_start: u32 = spec.core_start_layer orelse (if (max_loops > 1) @as(u32, 1) else 0);
    const cfg: GraphConfig = .{
        .axes = .{
            .backbone_kind = .decoder_only,
            .layer_attention_kinds = layer_attns,
            .ffn_kind = .moe_swiglu,
            .has_shared_expert = true,
            .residual_kind = .gated_4branch,
            .residual_branches = 4,
            .memory_lookup_kind = if (ple_num_shards > 0) .ple_ngram else .none,
            .loop_schedule = .{
                .mode = if (spec.include_steplora) .gated_loop_adapter else if (max_loops > 1) .native_recurrent else .single_pass,
                .start_layer = core_start,
                .end_layer = n_layers,
                .max_loops = max_loops,
                .inject_x_init = (max_loops > 1 and !spec.include_steplora),
                .adapter_kind = if (spec.include_steplora) .sme2_step_lora else .none,
                .adapter_rank = if (spec.include_steplora) 16 else 0,
                .adapter_packed_rank = if (spec.include_steplora) 64 else 0,
            },
        },
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = n_heads,
        .num_key_value_heads = n_kv,
        .head_dim = head_dim,
        .moe_intermediate_size = moe_inter,
        .shared_expert_intermediate_size = sh_inter,
        .num_experts = num_experts,
        .num_experts_per_tok = top_k,
        .norm_topk_prob = true,
        .linear_num_key_heads = gdn_kh,
        .linear_num_value_heads = gdn_vh,
        .linear_key_head_dim = gdn_kd,
        .linear_value_head_dim = gdn_vd,
        .linear_conv_kernel_dim = 4,
        .qsa_block_size = 32,
        .qsa_top_k_blocks = 4,
        .qsa_window_size = 128,
        .yarn = .{ .factor = 2.0, .original_max_pos = 1024 },
        .mrope_section = hybrid.MropeSection.QWEN38_DEFAULT,
        .ple_num_rows = ple_rows,
        .ple_embed_dim = h,
        .ple_ngram_order = 2,
        .ple_num_shards = ple_num_shards,
    };

    var blobs: std.ArrayListUnmanaged(TestTensorBlob) = .empty;
    var owned_names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
        for (owned_names.items) |n| allocator.free(n);
        owned_names.deinit(allocator);
    }

    const addBlob = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(TestTensorBlob),
            names: *std.ArrayListUnmanaged([]u8),
            nm: []const u8,
            kind: amf.TensorKind,
            dt: amf.DType,
            layer: u16,
            expert_id: u16,
            shape: [4]u32,
            ndim: u8,
            data: []u8,
            p_off: u64,
            p_bytes: u64,
        ) !void {
            const dup = try alloc.dupe(u8, nm);
            try names.append(alloc, dup);
            try list.append(alloc, .{
                .name = dup,
                .kind = kind,
                .dtype = dt,
                .layer = layer,
                .expert = expert_id,
                .shape = shape,
                .ndim = ndim,
                .bytes = data,
                .data_bytes_override = if (p_bytes > 0) @as(u64, shape[0]) * (@as(u64, shape[1]) / 2) else 0,
                .params_offset = p_off,
                .params_bytes = p_bytes,
            });
        }
    }.call;

    try addBlob(allocator, &blobs, &owned_names, "embed_tokens", .embedding, .f16, amf.NO_LAYER, amf.NO_EXPERT, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 101, vocab * h, 0.2), 0, 0);

    var nbuf: [160]u8 = undefined;
    for (0..n_layers) |l| {
        const sbase: u64 = 1000 + l * 100;
        const ln_in = try std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, ln_in, .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
        const ln_po = try std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, ln_po, .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);

        const gr_r = try std.fmt.bufPrint(&nbuf, "layers.{d}.gr.read_gate.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, gr_r, .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ 4, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 1, 4 * h, 0.1), 0, 0);
        const gr_w = try std.fmt.bufPrint(&nbuf, "layers.{d}.gr.write_gate.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, gr_w, .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ 4, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 2, 4 * h, 0.1), 0, 0);
        const gr_d = try std.fmt.bufPrint(&nbuf, "layers.{d}.gr.decay_gate.weight", .{l});
        try addBlob(allocator, &blobs, &owned_names, gr_d, .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ 4, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 3, 4 * h, 0.1), 0, 0);

        if (layer_attns[l] == .gated_deltanet) {
            const qk = gdn_kh * gdn_kd;
            const vd = gdn_vh * gdn_vd;
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.q_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ qk, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 10, qk * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.k_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ qk, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 11, qk * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.v_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ vd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 12, vd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.g_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ vd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 13, vd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.o_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ h, vd, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 14, h * vd, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.a_proj.weight", .{l}), .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ gdn_vh, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 15, gdn_vh * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.b_proj.weight", .{l}), .dense, .bf16, @intCast(l), amf.NO_EXPERT, .{ gdn_vh, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 16, gdn_vh * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.A_log", .{l}), .dense, .f32, @intCast(l), amf.NO_EXPERT, .{ gdn_vh, 0, 0, 0 }, 1, try fillDeterministicF32(allocator, sbase + 17, gdn_vh, -1.0, 0.0), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.dt_bias", .{l}), .dense, .f32, @intCast(l), amf.NO_EXPERT, .{ gdn_vh, 0, 0, 0 }, 1, try fillDeterministicF32(allocator, sbase + 18, gdn_vh, -0.2, 0.2), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.conv1d.weight", .{l}), .dense, .f32, @intCast(l), amf.NO_EXPERT, .{ 2 * qk + vd, 4, 0, 0 }, 2, try fillDeterministicF32(allocator, sbase + 19, (2 * qk + vd) * 4, -0.25, 0.25), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.linear_attn.norm.weight", .{l}), .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ gdn_vd, 0, 0, 0 }, 1, try fillConstantBf16(allocator, gdn_vd, 1.0), 0, 0);
        } else {
            const qd = n_heads * head_dim;
            const kvd = n_kv * head_dim;
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ qd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 20, qd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ kvd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 21, kvd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ kvd, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 22, kvd * h, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ h, qd, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 23, h * qd, 0.08), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}), .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ head_dim, 0, 0, 0 }, 1, try fillConstantBf16(allocator, head_dim, 1.0), 0, 0);
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}), .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ head_dim, 0, 0, 0 }, 1, try fillConstantBf16(allocator, head_dim, 1.0), 0, 0);
        }

        // Router + Shared Expert + E Routed Experts
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate.weight", .{l}), .router, .bf16, @intCast(l), amf.NO_EXPERT, .{ num_experts, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 30, num_experts * h, 0.15), 0, 0);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_expert.gate_proj", .{l}), .shared_expert, .f16, @intCast(l), amf.NO_EXPERT, .{ sh_inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 31, sh_inter * h, 0.06), 0, 0);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_expert.up_proj", .{l}), .shared_expert, .f16, @intCast(l), amf.NO_EXPERT, .{ sh_inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 32, sh_inter * h, 0.06), 0, 0);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_expert.down_proj", .{l}), .shared_expert, .f16, @intCast(l), amf.NO_EXPERT, .{ h, sh_inter, 0, 0 }, 2, try fillDeterministicF16(allocator, sbase + 33, h * sh_inter, 0.06), 0, 0);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.shared_expert_gate.weight", .{l}), .shared_expert, .bf16, @intCast(l), amf.NO_EXPERT, .{ 1, h, 0, 0 }, 2, try fillDeterministicBf16(allocator, sbase + 34, h, 0.1), 0, 0);

        for (0..num_experts) |e| {
            const is_cold = (cold_experts_per_layer > 0 and e >= num_experts - cold_experts_per_layer);
            const ekind: amf.TensorKind = if (is_cold) .cold_expert else .hot_expert;
            const eseed = sbase * 1000 + e * 7;
            if (spec.use_q4_experts) {
                const gq = try fillDeterministicQ4(allocator, eseed + 1, moe_inter, h, 0.05);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }), ekind, .q4_affine, @intCast(l), @intCast(e), .{ moe_inter, h, 0, 0 }, 2, gq.bytes, gq.params_offset, gq.params_bytes);
                const uq = try fillDeterministicQ4(allocator, eseed + 2, moe_inter, h, 0.05);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }), ekind, .q4_affine, @intCast(l), @intCast(e), .{ moe_inter, h, 0, 0 }, 2, uq.bytes, uq.params_offset, uq.params_bytes);
                const dq = try fillDeterministicQ4(allocator, eseed + 3, h, moe_inter, 0.05);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }), ekind, .q4_affine, @intCast(l), @intCast(e), .{ h, moe_inter, 0, 0 }, 2, dq.bytes, dq.params_offset, dq.params_bytes);
            } else {
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }), ekind, .f16, @intCast(l), @intCast(e), .{ moe_inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, eseed + 1, moe_inter * h, 0.05), 0, 0);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }), ekind, .f16, @intCast(l), @intCast(e), .{ moe_inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, eseed + 2, moe_inter * h, 0.05), 0, 0);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }), ekind, .f16, @intCast(l), @intCast(e), .{ h, moe_inter, 0, 0 }, 2, try fillDeterministicF16(allocator, eseed + 3, h * moe_inter, 0.05), 0, 0);
            }
        }
    }

    try addBlob(allocator, &blobs, &owned_names, "norm.weight", .norm, .bf16, amf.NO_LAYER, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
    try addBlob(allocator, &blobs, &owned_names, "lm_head", .lm_head, .f16, amf.NO_LAYER, amf.NO_EXPERT, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 999, vocab * h, 0.12), 0, 0);

    if (spec.include_steplora and max_loops > 0) {
        const gdn_qk_dim = gdn_kh * gdn_kd;
        const gdn_v_dim = gdn_vh * gdn_vd;
        const qsa_q_dim = n_heads * head_dim;
        for (0..max_loops) |step| {
            const alpha_val: f32 = if (step == 0) 0.0 else 0.15 * @as(f32, @floatFromInt(step));
            for (core_start..n_layers) |l| {
                const is_gdn = (layer_attns[l] == .gated_deltanet);
                const q_out_d: u32 = if (is_gdn) gdn_qk_dim else qsa_q_dim;
                const o_in_d: u32 = if (is_gdn) gdn_v_dim else qsa_q_dim;
                const sd: u64 = 12000 + step * 300 + l * 30;

                const qa_nm = try allocator.dupe(u8, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.lora_a", .{ step, l }));
                try owned_names.append(allocator, qa_nm);
                try blobs.append(allocator, .{ .name = qa_nm, .kind = .loop_adapter, .dtype = .f16, .layout = .sme2_panel64, .layer = @intCast(l), .expert = amf.NO_EXPERT, .shape = .{ 64, h, 0, 0 }, .ndim = 2, .bytes = try fillDeterministicStepLoraSme2F16(allocator, sd + 1, 64, h, 16, h, 0.08) });

                const qb_nm = try allocator.dupe(u8, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.lora_b", .{ step, l }));
                try owned_names.append(allocator, qb_nm);
                try blobs.append(allocator, .{ .name = qb_nm, .kind = .loop_adapter, .dtype = .f16, .layout = .sme2_panel64, .layer = @intCast(l), .expert = amf.NO_EXPERT, .shape = .{ q_out_d, 64, 0, 0 }, .ndim = 2, .bytes = try fillDeterministicStepLoraSme2F16(allocator, sd + 2, q_out_d, 64, q_out_d, 16, 0.08) });

                const al_q = try allocator.alloc(u8, 4);
                std.mem.writeInt(u32, al_q[0..4], @bitCast(alpha_val), .little);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.alpha", .{ step, l }), .loop_adapter, .f32, @intCast(l), amf.NO_EXPERT, .{ 1, 0, 0, 0 }, 1, al_q, 0, 0);

                const oa_nm = try allocator.dupe(u8, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.lora_a", .{ step, l }));
                try owned_names.append(allocator, oa_nm);
                try blobs.append(allocator, .{ .name = oa_nm, .kind = .loop_adapter, .dtype = .f16, .layout = .sme2_panel64, .layer = @intCast(l), .expert = amf.NO_EXPERT, .shape = .{ 64, o_in_d, 0, 0 }, .ndim = 2, .bytes = try fillDeterministicStepLoraSme2F16(allocator, sd + 3, 64, o_in_d, 16, o_in_d, 0.08) });

                const ob_nm = try allocator.dupe(u8, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.lora_b", .{ step, l }));
                try owned_names.append(allocator, ob_nm);
                try blobs.append(allocator, .{ .name = ob_nm, .kind = .loop_adapter, .dtype = .f16, .layout = .sme2_panel64, .layer = @intCast(l), .expert = amf.NO_EXPERT, .shape = .{ h, 64, 0, 0 }, .ndim = 2, .bytes = try fillDeterministicStepLoraSme2F16(allocator, sd + 4, h, 64, h, 16, 0.08) });

                const al_o = try allocator.alloc(u8, 4);
                std.mem.writeInt(u32, al_o[0..4], @bitCast(alpha_val), .little);
                try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.alpha", .{ step, l }), .loop_adapter, .f32, @intCast(l), amf.NO_EXPERT, .{ 1, 0, 0, 0 }, 1, al_o, 0, 0);
            }
        }
    }

    const amf_bytes = try buildTestAmf(allocator, "test-qwen4-exp", blobs.items);

    var amf_file_path: ?[:0]u8 = null;
    if (tmp_dir) |td| {
        const p = try std.fmt.allocPrintSentinel(allocator, "{s}/model.amf", .{td}, 0);
        try writeBytesFile(p, amf_bytes);
        amf_file_path = p;
    }

    var ple_paths = try allocator.alloc([:0]u8, ple_num_shards);
    if (ple_num_shards > 0) {
        const td = tmp_dir.?;
        const rows_per_shard = ple_rows / ple_num_shards;
        for (0..ple_num_shards) |s| {
            var pblobs: std.ArrayListUnmanaged(TestTensorBlob) = .empty;
            var pnames: std.ArrayListUnmanaged([]u8) = .empty;
            defer {
                for (pblobs.items) |b| allocator.free(b.bytes);
                pblobs.deinit(allocator);
                for (pnames.items) |n| allocator.free(n);
                pnames.deinit(allocator);
            }
            for (0..n_layers) |l| {
                const pnm = if (ple_num_shards == 1)
                    try std.fmt.bufPrint(&nbuf, "ple.layers.{d}.weight", .{l})
                else
                    try std.fmt.bufPrint(&nbuf, "ple.layers.{d}.shard.{d}.weight", .{ l, s });
                try addBlob(allocator, &pblobs, &pnames, pnm, .sparse_table, .bf16, @intCast(l), amf.NO_EXPERT, .{ @intCast(rows_per_shard), h, 0, 0 }, 2, try fillDeterministicBf16(allocator, 77700 + l * 17 + s, @intCast(rows_per_shard * h), 0.08), 0, 0);
            }
            const pbytes = try buildTestAmf(allocator, "test-qwen4-ple", pblobs.items);
            defer allocator.free(pbytes);
            const pp = try std.fmt.allocPrintSentinel(allocator, "{s}/ple_{d}.amf", .{ td, s }, 0);
            try writeBytesFile(pp, pbytes);
            ple_paths[s] = pp;
        }
    }

    return .{ .cfg = cfg, .amf_bytes = amf_bytes, .ple_paths = ple_paths, .amf_file_path = amf_file_path };
}

fn buildSynthQwen4TestBundle(
    allocator: std.mem.Allocator,
    num_experts: u32,
    top_k: u32,
    cold_experts_per_layer: u32,
    max_loops: u32,
    ple_num_shards: u32,
    tmp_dir: ?[]const u8,
) !SynthQwen4Bundle {
    return buildSynthQwen4Bundle(allocator, .{
        .num_experts = num_experts,
        .top_k = top_k,
        .cold_experts_per_layer = cold_experts_per_layer,
        .max_loops = max_loops,
        .ple_num_shards = ple_num_shards,
        .tmp_dir = tmp_dir,
    });
}

pub const SynthLoopDenseBenchSpec = struct {
    hidden_size: u32 = 64,
    vocab_size: u32 = 128,
    num_layers: u32 = 4,
    num_attention_heads: u32 = 2,
    num_key_value_heads: u32 = 1,
    head_dim: u32 = 64,
    intermediate_size: u32 = 128,
    start_layer: u32 = 1,
    end_layer: u32 = 3,
    mode: LoopMode = .native_recurrent,
    max_loops: u32 = 3,
    use_q4: bool = false,
    include_early_exit: bool = false,
    include_steplora: bool = false,
};

pub const SynthLoopDenseBundle = struct {
    cfg: GraphConfig,
    amf_bytes: []u8,

    pub fn deinit(self: *SynthLoopDenseBundle, allocator: std.mem.Allocator) void {
        self.cfg.deinit(allocator);
        allocator.free(self.amf_bytes);
    }
};

pub fn buildSynthLoopDenseBundle(
    allocator: std.mem.Allocator,
    spec: SynthLoopDenseBenchSpec,
) !SynthLoopDenseBundle {
    const h: u32 = spec.hidden_size;
    const vocab: u32 = spec.vocab_size;
    const n_layers: u32 = spec.num_layers;
    const n_heads: u32 = spec.num_attention_heads;
    const n_kv: u32 = spec.num_key_value_heads;
    const head_dim: u32 = spec.head_dim;
    const inter: u32 = spec.intermediate_size;
    const start_l: u32 = spec.start_layer;
    const end_l: u32 = spec.end_layer;
    const mode = spec.mode;
    const max_loops = spec.max_loops;
    const include_early_exit = spec.include_early_exit;
    const include_steplora = spec.include_steplora;

    const layer_attns = try allocator.alloc(AttentionKind, n_layers);
    @memset(layer_attns, .gqa_full);

    const cfg: GraphConfig = .{
        .axes = .{
            .backbone_kind = .decoder_only,
            .layer_attention_kinds = layer_attns,
            .ffn_kind = .dense_swiglu,
            .has_shared_expert = false,
            .residual_kind = .single_stream,
            .residual_branches = 1,
            .memory_lookup_kind = .none,
            .loop_schedule = .{
                .mode = mode,
                .start_layer = start_l,
                .end_layer = end_l,
                .max_loops = max_loops,
                .inject_x_init = (mode == .native_recurrent),
                .adapter_kind = if (include_steplora) .sme2_step_lora else .none,
                .adapter_rank = if (include_steplora) 16 else 0,
                .adapter_packed_rank = if (include_steplora) 64 else 0,
                .enforce_zero_alpha_r0 = true,
                .early_exit_gate = if (include_early_exit) .sigmoid_linear else .none,
                .early_exit_threshold = 0.5,
            },
        },
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = n_heads,
        .num_key_value_heads = n_kv,
        .head_dim = head_dim,
        .intermediate_size = inter,
        .rope_theta = 10_000.0,
    };

    var blobs: std.ArrayListUnmanaged(TestTensorBlob) = .empty;
    var owned_names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
        for (owned_names.items) |n| allocator.free(n);
        owned_names.deinit(allocator);
    }

    const addB = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(TestTensorBlob),
            names: *std.ArrayListUnmanaged([]u8),
            nm: []const u8,
            kind: amf.TensorKind,
            dt: amf.DType,
            layout: amf.TensorLayout,
            layer: u16,
            shape: [4]u32,
            ndim: u8,
            data: []u8,
            p_off: u64,
            p_bytes: u64,
        ) !void {
            const dup = try alloc.dupe(u8, nm);
            try names.append(alloc, dup);
            try list.append(alloc, .{
                .name = dup,
                .kind = kind,
                .dtype = dt,
                .layout = layout,
                .layer = layer,
                .expert = amf.NO_EXPERT,
                .shape = shape,
                .ndim = ndim,
                .bytes = data,
                .data_bytes_override = if (p_bytes > 0) @as(u64, shape[0]) * (@as(u64, shape[1]) / 2) else 0,
                .params_offset = p_off,
                .params_bytes = p_bytes,
            });
        }
    }.call;

    const addLin = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(TestTensorBlob),
            names: *std.ArrayListUnmanaged([]u8),
            nm: []const u8,
            kind: amf.TensorKind,
            layer: u16,
            n_out: u32,
            k_in: u32,
            seed: u64,
            amp: f32,
            use_q4: bool,
        ) !void {
            if (use_q4) {
                const q = try fillDeterministicQ4(alloc, seed, n_out, k_in, amp);
                try addB(alloc, list, names, nm, kind, .q4_affine, .row_major, layer, .{ n_out, k_in, 0, 0 }, 2, q.bytes, q.params_offset, q.params_bytes);
            } else {
                try addB(alloc, list, names, nm, kind, .f16, .row_major, layer, .{ n_out, k_in, 0, 0 }, 2, try fillDeterministicF16(alloc, seed, @as(usize, n_out) * k_in, amp), 0, 0);
            }
        }
    }.call;

    try addB(allocator, &blobs, &owned_names, "embed_tokens", .embedding, .f16, .row_major, amf.NO_LAYER, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 501, vocab * h, 0.25), 0, 0);

    var nbuf: [160]u8 = undefined;
    for (0..n_layers) |l| {
        const sb: u64 = 5000 + l * 50;
        const qd = n_heads * head_dim;
        const kvd = n_kv * head_dim;
        try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}), .norm, .bf16, .row_major, @intCast(l), .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj", .{l}), .dense, @intCast(l), qd, h, sb + 1, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj", .{l}), .dense, @intCast(l), kvd, h, sb + 2, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj", .{l}), .dense, @intCast(l), kvd, h, sb + 3, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj", .{l}), .dense, @intCast(l), h, qd, sb + 4, 0.08, spec.use_q4);
        try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}), .norm, .bf16, .row_major, @intCast(l), .{ head_dim, 0, 0, 0 }, 1, try fillConstantBf16(allocator, head_dim, 1.0), 0, 0);
        try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}), .norm, .bf16, .row_major, @intCast(l), .{ head_dim, 0, 0, 0 }, 1, try fillConstantBf16(allocator, head_dim, 1.0), 0, 0);
        try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}), .norm, .bf16, .row_major, @intCast(l), .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj", .{l}), .dense, @intCast(l), inter, h, sb + 5, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj", .{l}), .dense, @intCast(l), inter, h, sb + 6, 0.08, spec.use_q4);
        try addLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj", .{l}), .dense, @intCast(l), h, inter, sb + 7, 0.08, spec.use_q4);
    }

    try addB(allocator, &blobs, &owned_names, "norm.weight", .norm, .bf16, .row_major, amf.NO_LAYER, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0), 0, 0);
    try addB(allocator, &blobs, &owned_names, "lm_head", .lm_head, .f16, .row_major, amf.NO_LAYER, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 599, vocab * h, 0.15), 0, 0);

    if (include_early_exit) {
        try addB(allocator, &blobs, &owned_names, "loop.early_exit_gate.weight", .dense, .bf16, .row_major, amf.NO_LAYER, .{ 1, h, 0, 0 }, 2, try fillConstantBf16(allocator, h, 0.5), 0, 0);
    }

    if (include_steplora) {
        const qd = n_heads * head_dim;
        for (0..max_loops) |step| {
            const alpha_val: f32 = if (step == 0) 0.0 else 0.15 * @as(f32, @floatFromInt(step));
            for (start_l..end_l) |l| {
                const sd: u64 = 9000 + step * 300 + l * 30;
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.lora_a", .{ step, l }), .loop_adapter, .f16, .sme2_panel64, @intCast(l), .{ 64, h, 0, 0 }, 2, try fillDeterministicStepLoraSme2F16(allocator, sd + 1, 64, h, 16, h, 0.08), 0, 0);
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.lora_b", .{ step, l }), .loop_adapter, .f16, .sme2_panel64, @intCast(l), .{ qd, 64, 0, 0 }, 2, try fillDeterministicStepLoraSme2F16(allocator, sd + 2, qd, 64, qd, 16, 0.08), 0, 0);
                const al_q = try allocator.alloc(u8, 4);
                std.mem.writeInt(u32, al_q[0..4], @bitCast(alpha_val), .little);
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.q_proj.alpha", .{ step, l }), .loop_adapter, .f32, .row_major, @intCast(l), .{ 1, 0, 0, 0 }, 1, al_q, 0, 0);

                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.lora_a", .{ step, l }), .loop_adapter, .f16, .sme2_panel64, @intCast(l), .{ 64, qd, 0, 0 }, 2, try fillDeterministicStepLoraSme2F16(allocator, sd + 3, 64, qd, 16, qd, 0.08), 0, 0);
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.lora_b", .{ step, l }), .loop_adapter, .f16, .sme2_panel64, @intCast(l), .{ h, 64, 0, 0 }, 2, try fillDeterministicStepLoraSme2F16(allocator, sd + 4, h, 64, h, 16, 0.08), 0, 0);
                const al_o = try allocator.alloc(u8, 4);
                std.mem.writeInt(u32, al_o[0..4], @bitCast(alpha_val), .little);
                try addB(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "loop.step.{d}.layers.{d}.o_proj.alpha", .{ step, l }), .loop_adapter, .f32, .row_major, @intCast(l), .{ 1, 0, 0, 0 }, 1, al_o, 0, 0);
            }
        }
    }

    const amf_bytes = try buildTestAmf(allocator, "test-loop-dense", blobs.items);
    return .{ .cfg = cfg, .amf_bytes = amf_bytes };
}

fn buildSynthLoopDenseTestBundle(
    allocator: std.mem.Allocator,
    mode: LoopMode,
    max_loops: u32,
    include_early_exit: bool,
    include_steplora: bool,
) !SynthLoopDenseBundle {
    return buildSynthLoopDenseBundle(allocator, .{
        .mode = mode,
        .max_loops = max_loops,
        .include_early_exit = include_early_exit,
        .include_steplora = include_steplora,
    });
}

fn writeBytesFile(path: [:0]const u8, data: []const u8) !void {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.IoFailed;
    defer _ = std.c.close(fd);
    var written: usize = 0;
    while (written < data.len) {
        const rc = std.c.write(fd, data.ptr + written, data.len - written);
        if (rc <= 0) return error.IoFailed;
        written += @intCast(rc);
    }
}

pub fn makeTestTempDir(allocator: std.mem.Allocator, tag: []const u8) ![]u8 {
    const base = std.c.getenv("TMPDIR");
    const root_slice: []const u8 = if (base) |p| std.mem.span(p) else "/tmp";
    const dir_path = try std.fmt.allocPrint(allocator, "{s}/graph_test_{s}_{d}", .{ root_slice, tag, std.c.getpid() });
    errdefer allocator.free(dir_path);
    const dir_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{dir_path}, 0);
    defer allocator.free(dir_z);
    _ = std.c.mkdir(dir_z.ptr, 0o755);
    return dir_path;
}

pub fn removeTestFile(path: [:0]const u8) void {
    _ = std.c.unlink(path.ptr);
}

pub fn removeTestDir(allocator: std.mem.Allocator, path: []const u8) void {
    const z = std.fmt.allocPrintSentinel(allocator, "{s}", .{path}, 0) catch return;
    defer allocator.free(z);
    _ = std.c.rmdir(z.ptr);
}

/// CPU FP32 reference forward pass for dense `gqa_full` + `LoopLM` (`R = 1..3`)
/// verifying exact numerical agreement with `GraphModel`.
fn evalLoopDenseCpuRef(
    allocator: std.mem.Allocator,
    amf_bytes: []const u8,
    cfg: GraphConfig,
    tokens: []const u32,
    num_loops: u32,
) ![]f32 {
    var parsed = try amf.Parsed.parseBytes(allocator, amf_bytes);
    defer parsed.deinit(allocator);
    const blobOf = struct {
        fn call(p: *const amf.Parsed, raw: []const u8, name: []const u8) []const u8 {
            const e = p.find(name).?;
            return raw[@intCast(e.blob_offset)..][0..@intCast(e.blob_bytes)];
        }
    }.call;

    const h: usize = cfg.hidden_size;
    const qd: usize = cfg.qDim();
    const kvd: usize = cfg.kvDim();
    const inter: usize = cfg.intermediate_size;
    const vocab: usize = cfg.vocab_size;
    const n_layers: usize = cfg.num_hidden_layers;
    const max_t = tokens.len;

    const k_cache = try allocator.alloc(f32, n_layers * max_t * kvd);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, n_layers * max_t * kvd);
    defer allocator.free(v_cache);
    @memset(k_cache, 0.0);
    @memset(v_cache, 0.0);

    const x = try allocator.alloc(f32, h);
    defer allocator.free(x);
    const x_init = try allocator.alloc(f32, h);
    defer allocator.free(x_init);
    const xn = try allocator.alloc(f32, h);
    defer allocator.free(xn);
    const q = try allocator.alloc(f32, qd);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kvd);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kvd);
    defer allocator.free(v);
    const attn_out = try allocator.alloc(f32, qd);
    defer allocator.free(attn_out);
    const proj = try allocator.alloc(f32, h);
    defer allocator.free(proj);
    const gbuf = try allocator.alloc(f32, inter);
    defer allocator.free(gbuf);
    const ubuf = try allocator.alloc(f32, inter);
    defer allocator.free(ubuf);
    const logits = try allocator.alloc(f32, vocab);

    const ls = cfg.axes.loop_schedule;
    const core_start: usize = if (num_loops > 1) ls.start_layer else 0;
    const core_end: usize = if (num_loops > 1) ls.end_layer else n_layers;

    const emb_raw = blobOf(&parsed, amf_bytes, "embed_tokens");
    var nbuf: [160]u8 = undefined;

    for (tokens, 0..) |tid, pos| {
        for (0..h) |d| x[d] = readFloatWeightAt(emb_raw, .f16, @as(usize, tid) * h + d);

        const runLayer = struct {
            fn call(
                p: *const amf.Parsed,
                raw: []const u8,
                c: GraphConfig,
                l: usize,
                pos_i: usize,
                commit_kv: bool,
                kc: []f32,
                vc: []f32,
                xv: []f32,
                xnv: []f32,
                qv: []f32,
                kv: []f32,
                vv: []f32,
                aov: []f32,
                pv: []f32,
                gv: []f32,
                uv: []f32,
                nb: *[160]u8,
            ) void {
                const hh: usize = c.hidden_size;
                const qdd: usize = c.qDim();
                const kvdd: usize = c.kvDim();
                const hdd: usize = c.head_dim;
                const nhh: usize = c.num_attention_heads;
                const nkvv: usize = c.num_key_value_heads;
                const ii: usize = c.intermediate_size;
                const mt: usize = kc.len / (c.num_hidden_layers * kvdd);

                cpuRmsNorm(xv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable), .bf16, c.rms_norm_eps, xnv);
                cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable), .f16, qdd, hh, qv);
                if (commit_kv) {
                    cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable), .f16, kvdd, hh, kv);
                    cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable), .f16, kvdd, hh, vv);
                }

                const qnw = blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable);
                const knw = blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable);
                const half_hd = hdd / 2;
                for (0..nhh) |hd_i| {
                    const seg = qv[hd_i * hdd ..][0..hdd];
                    var tmp: [256]f32 = undefined;
                    cpuRmsNorm(seg, qnw, .bf16, c.rms_norm_eps, tmp[0..hdd]);
                    for (0..half_hd) |j| {
                        const inv_f = std.math.pow(f32, c.rope_theta, -2.0 * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(hdd)));
                        const ang = @as(f32, @floatFromInt(pos_i)) * inv_f;
                        const cs = @cos(ang);
                        const sn = @sin(ang);
                        const a = tmp[j];
                        const b = tmp[j + half_hd];
                        seg[j] = a * cs - b * sn;
                        seg[j + half_hd] = a * sn + b * cs;
                    }
                }
                if (commit_kv) {
                    for (0..nkvv) |hd_i| {
                        const seg = kv[hd_i * hdd ..][0..hdd];
                        var tmp: [256]f32 = undefined;
                        cpuRmsNorm(seg, knw, .bf16, c.rms_norm_eps, tmp[0..hdd]);
                        for (0..half_hd) |j| {
                            const inv_f = std.math.pow(f32, c.rope_theta, -2.0 * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(hdd)));
                            const ang = @as(f32, @floatFromInt(pos_i)) * inv_f;
                            const cs = @cos(ang);
                            const sn = @sin(ang);
                            const a = tmp[j];
                            const b = tmp[j + half_hd];
                            seg[j] = a * cs - b * sn;
                            seg[j + half_hd] = a * sn + b * cs;
                        }
                    }
                    @memcpy(kc[(l * mt + pos_i) * kvdd ..][0..kvdd], kv);
                    @memcpy(vc[(l * mt + pos_i) * kvdd ..][0..kvdd], vv);
                }

                // Causal GQA attending over committed `0..=pos_i` (using r=0 shared KV at `pos_i`).
                const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hdd)));
                const group = nhh / nkvv;
                for (0..nhh) |qh| {
                    const kvh = qh / group;
                    var max_s: f32 = -std.math.inf(f32);
                    var scores: [64]f32 = undefined;
                    for (0..pos_i + 1) |t| {
                        const k_src = kc[(l * mt + t) * kvdd + kvh * hdd ..][0..hdd];
                        var dot: f32 = 0.0;
                        for (0..hdd) |d| dot += qv[qh * hdd + d] * k_src[d];
                        scores[t] = dot * scale;
                        if (scores[t] > max_s) max_s = scores[t];
                    }
                    var sum_e: f32 = 0.0;
                    for (0..pos_i + 1) |t| {
                        scores[t] = @exp(scores[t] - max_s);
                        sum_e += scores[t];
                    }
                    for (0..hdd) |d| {
                        var acc: f32 = 0.0;
                        for (0..pos_i + 1) |t| {
                            const v_src = vc[(l * mt + t) * kvdd + kvh * hdd ..][0..hdd];
                            acc += (scores[t] / sum_e) * v_src[d];
                        }
                        aov[qh * hdd + d] = acc;
                    }
                }

                cpuLinearF16(aov, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable), .f16, hh, qdd, pv);
                for (0..hh) |d| xv[d] += pv[d];

                cpuRmsNorm(xv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable), .bf16, c.rms_norm_eps, xnv);
                cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable), .f16, ii, hh, gv);
                cpuLinearF16(xnv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.mlp.up_proj", .{l}) catch unreachable), .f16, ii, hh, uv);
                for (0..ii) |d| gv[d] = cpuSilu(gv[d]) * uv[d];
                cpuLinearF16(gv, blobOf(p, raw, std.fmt.bufPrint(nb, "layers.{d}.mlp.down_proj", .{l}) catch unreachable), .f16, hh, ii, pv);
                for (0..hh) |d| xv[d] += pv[d];
            }
        }.call;

        for (0..core_start) |l| runLayer(&parsed, amf_bytes, cfg, l, pos, true, k_cache, v_cache, x, xn, q, k, v, attn_out, proj, gbuf, ubuf, &nbuf);
        if (num_loops > 1 and ls.inject_x_init) @memcpy(x_init, x);
        for (0..num_loops) |r| {
            if (r >= 1 and ls.inject_x_init) {
                for (0..h) |d| x[d] = 0.5 * x[d] + 0.5 * x_init[d];
            }
            const is_r0 = (r == 0);
            for (core_start..core_end) |l| runLayer(&parsed, amf_bytes, cfg, l, pos, is_r0, k_cache, v_cache, x, xn, q, k, v, attn_out, proj, gbuf, ubuf, &nbuf);
        }
        for (core_end..n_layers) |l| runLayer(&parsed, amf_bytes, cfg, l, pos, true, k_cache, v_cache, x, xn, q, k, v, attn_out, proj, gbuf, ubuf, &nbuf);

        if (pos == tokens.len - 1) {
            cpuRmsNorm(x, blobOf(&parsed, amf_bytes, "norm.weight"), .bf16, cfg.rms_norm_eps, xn);
            cpuLinearF16(xn, blobOf(&parsed, amf_bytes, "lm_head"), .f16, vocab, h, logits);
        }
    }
    return logits;
}

test "W9+W10: tiny-qwen4 (3:1 GDN:QSA, 4-branch GR, shared MoE, PLE) prefill, decode & tree parity" {
    const allocator = std.testing.allocator;
    const tmp_dir = try makeTestTempDir(allocator, "tiny_qwen4");
    defer {
        removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    var b = try buildSynthQwen4TestBundle(allocator, 16, 4, 0, 1, 1, tmp_dir);
    defer {
        b.cfg.deinit(allocator);
        allocator.free(b.amf_bytes);
        if (b.amf_file_path) |p| {
            removeTestFile(p);
            allocator.free(p);
        }
        for (b.ple_paths) |p| {
            removeTestFile(p);
            allocator.free(p);
        }
        allocator.free(b.ple_paths);
    }

    var ple_const = try allocator.alloc([:0]const u8, b.ple_paths.len);
    defer allocator.free(ple_const);
    for (b.ple_paths, 0..) |p, i| ple_const[i] = p;

    var serial_model = try GraphModel.loadFromBytes(allocator, b.amf_bytes, b.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .ple_paths = ple_const,
    });
    defer serial_model.deinit(allocator);

    var tree_model = try GraphModel.loadFromBytes(allocator, b.amf_bytes, b.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .ple_paths = ple_const,
    });
    defer tree_model.deinit(allocator);

    // Prefill prompt tokens on both models.
    const prompt = [_]u32{ 3, 17, 42, 7 };
    const s_next = try serial_model.prefillChunk(&prompt, 0, true);
    const t_next = try tree_model.prefillChunk(&prompt, 0, true);
    try std.testing.expectEqual(s_next, t_next);

    // Run 3 serial decode steps vs 1 batched verifyTree + commitTree on a 4-node branching tree!
    // Tree layout: 0 (root = s_next at pos 4) -> 1 (pos 5, child of 0) -> 2 (pos 6, child of 1);
    //              3 (pos 5, sibling child of 0)
    const d0 = try serial_model.stepDecode(s_next, 4);
    const d1 = try serial_model.stepDecode(d0, 5);
    const d2 = try serial_model.stepDecode(d1, 6);
    var serial_logits_last: [128]f16 = undefined;
    @memcpy(&serial_logits_last, serial_model.readLastLogitsF16());

    const tree_toks = [_]u32{ s_next, d0, d1, (d0 + 5) % 128 };
    const tree_pos = [_]u32{ 4, 5, 6, 5 };
    const tree_masks = [_]u64{ 0b0001, 0b0011, 0b0111, 0b1001 };
    const vr = try tree_model.verifyTree(&tree_toks, &tree_pos, &tree_masks, 4);
    try std.testing.expectEqual(d0, vr.argmax_tokens[0]);
    try std.testing.expectEqual(d1, vr.argmax_tokens[1]);
    try std.testing.expectEqual(d2, vr.argmax_tokens[2]);

    // Check tight numerical parity on node 2's full logit vector vs serial decode!
    const v_row = vr.logits_f16[2 * 128 .. 3 * 128];
    var max_err: f32 = 0.0;
    for (0..128) |i| {
        const diff = @abs(@as(f32, @floatCast(v_row[i])) - @as(f32, @floatCast(serial_logits_last[i])));
        if (diff > max_err) max_err = diff;
    }
    try std.testing.expect(max_err < 1e-2);

    // Commit accepted path [0, 1, 2] and verify subsequent decode matches serial!
    try tree_model.commitTree(4, &.{ 0, 1, 2 });
    const s_after = try serial_model.stepDecode(d2, 7);
    const t_after = try tree_model.stepDecode(d2, 7);
    try std.testing.expectEqual(s_after, t_after);
}

test "W10+G7+C10+ADR-0019: scale-qwen4-512e (E=512, top-10 + shared, 4-shard PLE, cold experts, R=3) 4 loop isolations" {
    const allocator = std.testing.allocator;
    const tmp_dir = try makeTestTempDir(allocator, "scale_512e_iso");
    defer {
        removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    // E=512, top_k=10, 64 cold experts/layer, R=3 recurrent loops, 4-shard PLE!
    var b = try buildSynthQwen4TestBundle(allocator, 512, 10, 64, 3, 4, tmp_dir);
    defer {
        b.cfg.deinit(allocator);
        allocator.free(b.amf_bytes);
        if (b.amf_file_path) |p| {
            removeTestFile(p);
            allocator.free(p);
        }
        for (b.ple_paths) |p| {
            removeTestFile(p);
            allocator.free(p);
        }
        allocator.free(b.ple_paths);
    }

    var ple_const = try allocator.alloc([:0]const u8, b.ple_paths.len);
    defer allocator.free(ple_const);
    for (b.ple_paths, 0..) |p, i| ple_const[i] = p;

    var model = try GraphModel.loadFromFile(allocator, b.amf_file_path.?, b.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .ple_paths = ple_const,
        .cold_slot_count = 32,
    });
    defer model.deinit(allocator);

    try std.testing.expect(model.hasColdExperts());
    const prompt = [_]u32{ 5, 11, 23, 37 };
    const t0 = try model.prefillChunk(&prompt, 0, true);

    // Snapshot committed GDN state `S_t` before tree verify!
    const ga = &(model.gdn_arena.?);
    const s_before = try allocator.dupe(f32, try ga.committedLayerSlice(1));
    defer allocator.free(s_before);

    model.resetLoopIsolationStats();
    // 1. Run `verifyTree` (`M = 4`, `R = 3`): committed GDN `S_t` MUST be byte-exact untouched!
    const toks = [_]u32{ t0, (t0 + 1) % 128, (t0 + 2) % 128, (t0 + 3) % 128 };
    const pos = [_]u32{ 4, 5, 6, 5 };
    const masks = [_]u64{ 0b0001, 0b0011, 0b0111, 0b1001 };
    const vr = try model.verifyTree(&toks, &pos, &masks, 4);
    try std.testing.expectEqual(@as(u32, 4), vr.num_nodes);
    try std.testing.expect(vr.total_unique_experts > 0);

    const s_after_verify = try ga.committedLayerSlice(1);
    try std.testing.expectEqualSlices(f32, s_before, s_after_verify);

    // Check ALL FOUR ADR-0019 loop isolations on `R = 3`:
    const iso_verify = model.loopIsolationStats();
    // #1 GDN: scratch evals on r=0,1,2 (`0` commits during verifyTree)
    try std.testing.expect(iso_verify.gdn_scratch_evals >= 6);
    try std.testing.expectEqual(@as(u64, 0), iso_verify.gdn_final_commits);
    // #2 PLE: `r = 0` fills, `r >= 1` reuses with EXACTLY `0` extra NVMe reads!
    try std.testing.expect(iso_verify.ple_r0_fills > 0);
    try std.testing.expect(iso_verify.ple_r_ge1_reuses > 0);
    try std.testing.expectEqual(@as(u64, 0), iso_verify.ple_r_ge1_nvme_reads);
    // #3 QSA: `r = 0` computes indices once, `r >= 1` reuses indices (`reuse_block_indices = true`)!
    try std.testing.expectEqual(@as(u64, 1), iso_verify.qsa_index_computes);
    try std.testing.expectEqual(@as(u64, 2), iso_verify.qsa_index_reuses);
    // #4 MoE: `r >= 1` pinned-only routes with `0` cold NVMe stalls!
    try std.testing.expect(iso_verify.moe_r_ge1_pinned_routes > 0);
    try std.testing.expectEqual(@as(u64, 0), iso_verify.moe_r_ge1_cold_stalls);

    // 2. Run single-step `stepDecode` (`R = 3`): commits `S_tmp -> S_t` ONCE at `r = R - 1`!
    model.resetLoopIsolationStats();
    _ = try model.stepDecode(t0, 4);
    const iso_dec = model.loopIsolationStats();
    // 2 core GDN layers (L1, L2) x 2 intermediate steps (r=0,1) = 4 scratch evals,
    // plus L0 prelude (1) + L1,L2 final r=2 (2) = 3 final commits!
    try std.testing.expectEqual(@as(u64, 4), iso_dec.gdn_scratch_evals);
    try std.testing.expectEqual(@as(u64, 3), iso_dec.gdn_final_commits);
    try std.testing.expectEqual(@as(u64, 0), iso_dec.ple_r_ge1_nvme_reads);
    try std.testing.expectEqual(@as(u64, 1), iso_dec.qsa_index_computes);
    try std.testing.expectEqual(@as(u64, 2), iso_dec.qsa_index_reuses);
    try std.testing.expectEqual(@as(u64, 0), iso_dec.moe_r_ge1_cold_stalls);
}

test "W9+ADR-0019: native tiny-looplm (R=1..3, inject_x_init) CPU fp32 parity, tree parity & early-exit gate" {
    const allocator = std.testing.allocator;
    var b = try buildSynthLoopDenseTestBundle(allocator, .native_recurrent, 3, true, false);
    defer {
        b.cfg.deinit(allocator);
        allocator.free(b.amf_bytes);
    }

    const prompt = [_]u32{ 4, 19, 63 };
    for ([_]u32{ 1, 2, 3 }) |R| {
        const ref_logits = try evalLoopDenseCpuRef(allocator, b.amf_bytes, b.cfg, &prompt, R);
        defer allocator.free(ref_logits);

        var gpu_model = try GraphModel.loadFromBytes(allocator, b.amf_bytes, b.cfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 16,
            .max_loops_override = R,
            .enable_loop_early_exit = false,
        });
        defer gpu_model.deinit(allocator);

        var last_tok: u32 = 0;
        for (prompt, 0..) |tid, i| {
            last_tok = try gpu_model.stepDecode(tid, @intCast(i));
        }
        const gpu_logits = gpu_model.readLastLogitsF16();
        var max_err: f32 = 0.0;
        var ref_argmax: u32 = 0;
        var ref_max: f32 = -std.math.inf(f32);
        for (0..b.cfg.vocab_size) |i| {
            if (ref_logits[i] > ref_max) {
                ref_max = ref_logits[i];
                ref_argmax = @intCast(i);
            }
            const diff = @abs(@as(f32, @floatCast(gpu_logits[i])) - ref_logits[i]);
            if (diff > max_err) max_err = diff;
        }
        try std.testing.expect(max_err < 2e-2);
        try std.testing.expectEqual(ref_argmax, last_tok);
    }

    // Verify opt-in `--loop-early-exit` (`sigmoid_linear`) triggers at threshold 0.0 (always >= 0.0).
    var ee_model = try GraphModel.loadFromBytes(allocator, b.amf_bytes, b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 3,
        .enable_loop_early_exit = true,
        .early_exit_threshold = 0.0,
    });
    defer ee_model.deinit(allocator);
    _ = try ee_model.stepDecode(4, 0);
    const ee_stats = ee_model.loopIsolationStats();
    try std.testing.expectEqual(@as(u64, 1), ee_stats.total_loop_steps);
    try std.testing.expectEqual(@as(u64, 1), ee_stats.early_exit_tokens);
}

test "W16: tiny-steplora alpha_0 == 0 bit-exact identity at R=1 & active SME2 Step-LoRA at R=2..3" {
    const allocator = std.testing.allocator;
    var base_b = try buildSynthLoopDenseTestBundle(allocator, .single_pass, 1, false, false);
    defer {
        base_b.cfg.deinit(allocator);
        allocator.free(base_b.amf_bytes);
    }
    var lora_b = try buildSynthLoopDenseTestBundle(allocator, .gated_loop_adapter, 3, false, true);
    defer {
        lora_b.cfg.deinit(allocator);
        allocator.free(lora_b.amf_bytes);
    }

    var base_model = try GraphModel.loadFromBytes(allocator, base_b.amf_bytes, base_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 1,
    });
    defer base_model.deinit(allocator);

    var lora_r1 = try GraphModel.loadFromBytes(allocator, lora_b.amf_bytes, lora_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 1,
    });
    defer lora_r1.deinit(allocator);

    const prompt = [_]u32{ 9, 22, 51 };
    for (prompt, 0..) |tid, i| {
        const b_tok = try base_model.stepDecode(tid, @intCast(i));
        const l_tok = try lora_r1.stepDecode(tid, @intCast(i));
        try std.testing.expectEqual(b_tok, l_tok);
    }
    // Bit-exact `f16` logits identity at `R = 1` (`alpha_0 == 0.0`)!
    const base_logits = base_model.readLastLogitsF16();
    const lora_r1_logits = lora_r1.readLastLogitsF16();
    for (0..base_b.cfg.vocab_size) |i| {
        try std.testing.expectEqual(@as(u16, @bitCast(base_logits[i])), @as(u16, @bitCast(lora_r1_logits[i])));
    }
    const iso_r1 = lora_r1.loopIsolationStats();
    try std.testing.expect(iso_r1.steplora_zero_alpha_skips > 0);
    try std.testing.expectEqual(@as(u64, 0), iso_r1.steplora_active_evals);

    // At R = 3, SME2 Step-LoRA runs actively on r = 1, 2 and `verifyTree` matches `stepDecode`!
    var lora_r3_serial = try GraphModel.loadFromBytes(allocator, lora_b.amf_bytes, lora_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 3,
    });
    defer lora_r3_serial.deinit(allocator);

    var lora_r3_tree = try GraphModel.loadFromBytes(allocator, lora_b.amf_bytes, lora_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .max_loops_override = 3,
    });
    defer lora_r3_tree.deinit(allocator);

    const s0 = try lora_r3_serial.prefillChunk(&prompt, 0, true);
    const t0 = try lora_r3_tree.prefillChunk(&prompt, 0, true);
    try std.testing.expectEqual(s0, t0);

    const iso_r3 = lora_r3_serial.loopIsolationStats();
    try std.testing.expect(iso_r3.steplora_zero_alpha_skips > 0);
    try std.testing.expect(iso_r3.steplora_active_evals > 0);

    const sd1 = try lora_r3_serial.stepDecode(s0, 3);
    const sd2 = try lora_r3_serial.stepDecode(sd1, 4);
    const vr = try lora_r3_tree.verifyTree(&.{ t0, sd1 }, &.{ 3, 4 }, &.{ 0b01, 0b11 }, 3);
    try std.testing.expectEqual(sd1, vr.argmax_tokens[0]);
    try std.testing.expectEqual(sd2, vr.argmax_tokens[1]);
}

test "M6.5 R5 GraphModel QSA tiered q8 retired KV recycles IOSurface tiles and matches fp16" {
    const allocator = std.testing.allocator;
    var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 1, 0, null);
    defer bundle.deinit(allocator);
    // Full-context QSA (`qsa_window_size = 0`, `chunk_tokens = 512`) enables R5 retired q8 tier!
    bundle.cfg.qsa_window_size = 0;

    var fp16_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
        .max_seq_len = 1536,
        .max_tree_nodes = 32,
        .kv_retire_q8_after_chunks = 0,
    });
    defer fp16_m.deinit(allocator);

    var tiered_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
        .max_seq_len = 1536,
        .max_tree_nodes = 32,
        .kv_retire_q8_after_chunks = 1,
    });
    defer tiered_m.deinit(allocator);

    try std.testing.expect(tiered_m.kvRetireEnabled());
    try std.testing.expect(tiered_m.activeKvWiredBytes() < fp16_m.activeKvWiredBytes());
    try std.testing.expect(tiered_m.retiredKvWiredBytes() > 0);

    // Prefill 540 tokens (> 1 full 512-token chunk -> chunk 0 retires to group-64 q8!).
    var prompt: [540]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast((i * 13 + 7) % bundle.cfg.vocab_size);

    const fp16_first = try fp16_m.prefillChunk(&prompt, 0, true);
    const tiered_first = try tiered_m.prefillChunk(&prompt, 0, true);
    try std.testing.expectEqual(@as(u32, 1), tiered_m.retiredKvChunks());
    try std.testing.expectEqual(@as(u64, 1), tiered_m.kvRetireEvents());
    try std.testing.expectEqual(fp16_first, tiered_first);

    var tf = fp16_first;
    var tt = tiered_first;
    for (0..4) |step| {
        const pos: u32 = @intCast(prompt.len + step);
        tf = try fp16_m.stepDecode(tf, pos);
        tt = try tiered_m.stepDecode(tt, pos);
        try std.testing.expectEqual(tf, tt);
    }
}

test "M6.5 G14 GraphModel prefillMultimodal with HW VideoToolbox HEVC -> zero-copy IOSurface -> 3D MRoPE" {
    const allocator = std.testing.allocator;
    var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 1, 0, null);
    defer bundle.deinit(allocator);
    bundle.cfg.qsa_window_size = 0;

    var model = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 32,
    });
    defer model.deinit(allocator);

    // Real M4 hardware VideoToolbox HEVC clip -> zero-copy IOSurface NV12 VisionFrames.
    const clip = try metal.VtClip.encodeTestClip(.hevc, 128, 128, 2, 30);
    defer clip.close();
    const dec = try metal.VtDecodeSession.init(model.device, clip, .nv12);
    defer dec.close();
    const f0 = try dec.decodeFrame(clip, 0);
    defer f0.close();
    const f1 = try dec.decodeFrame(clip, 1);
    defer f1.close();

    const prefix = [_]u32{ 1, 5, 9 };
    const suffix = [_]u32{ 11, 14 };
    const frames = [_]metal.VisionFrame{ f0, f1 };

    const res = try model.prefillMultimodal(allocator, &prefix, &suffix, &frames, .{
        .patch_size = 16,
        .t_patch = 2,
        .spatial_merge = 2,
    });
    try std.testing.expect(res.all_frames_zero_copy);
    try std.testing.expect(res.all_frames_page_aligned);
    // 128x128 with patch=16 -> 8x8 raw patches -> 2x2 merge -> 4x4 = 16 visual tokens, t_patch=2 -> grid_t=1.
    try std.testing.expectEqual(@as(u32, 16), res.vision_tokens);
    try std.testing.expectEqual(@as(u32, 21), res.total_prompt_tokens);
    // MRoPE next_text_pos: prefix(3) + max(grid_t=1, grid_h=4, grid_w=4) + suffix(2) = 9!
    try std.testing.expectEqual(@as(u32, 9), res.next_mrope_pos);
    try std.testing.expect(res.first_token < bundle.cfg.vocab_size);
    try std.testing.expect(res.vision_encode_us > 0.0);

    // Subsequent text decode must advance seamlessly with 3D-aligned MRoPE position!
    const next_tok = try model.stepDecode(res.first_token, res.total_prompt_tokens);
    try std.testing.expect(next_tok < bundle.cfg.vocab_size);
    try std.testing.expectEqual(@as(u32, 10), model.mrope_next_pos);
}

test "M6.5 C14+R8 GraphBatchRuntime hybrid GDN+QSA+MoE continuous batching + GdnSnapshotPool parity" {
    const allocator = std.testing.allocator;
    var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 1, 0, null);
    defer bundle.deinit(allocator);
    bundle.cfg.qsa_window_size = 0;

    var serial_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
    });
    defer serial_m.deinit(allocator);

    var batch_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
    });
    defer batch_m.deinit(allocator);

    var rt = try GraphBatchRuntime.init(allocator, &batch_m, .{
        .max_slots = 4,
        .prefix_cache_chunks = 8,
    });
    defer rt.deinit(allocator);

    const p_common = [_]u32{ 3, 17, 42, 9, 61, 88, 12, 5 };
    const p0 = p_common;
    const p1 = p_common ++ [_]u32{ 4, 5, 6 };
    const p2 = p_common ++ [_]u32{ 7, 8, 9 };
    const prompts = [_][]const u32{ &p0, &p1, &p2 };
    const gen_len: usize = 5;

    var ref_out: [3][gen_len]u32 = undefined;
    for (prompts, 0..) |pr, idx| {
        serial_m.resetKvCache();
        var tok = try serial_m.prefillChunk(pr, 0, true);
        ref_out[idx][0] = tok;
        for (1..gen_len) |step| {
            tok = try serial_m.stepDecode(tok, @intCast(pr.len + step - 1));
            ref_out[idx][step] = tok;
        }
    }

    var sched = try batch_sched.BatchScheduler.init(allocator, .{
        .max_batch_size = 4,
        .max_prefill_chunk_tokens = 32,
        .max_prompt_tokens = 256,
    });
    defer sched.deinit(allocator);

    const Collector = struct {
        tokens: [3][gen_len]u32 = @splat(@splat(0)),
        counts: [3]usize = @splat(0),
        hits: [3]u32 = @splat(0),
        fn onPrefillDone(ctx: *anyopaque, _: u16, tag: u64, _: u32, hit: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.hits[@intCast(tag)] = hit;
            return true;
        }
        fn onToken(ctx: *anyopaque, _: u16, tag: u64, tid: u32, _: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const i: usize = @intCast(tag);
            if (self.counts[i] < gen_len) {
                self.tokens[i][self.counts[i]] = tid;
                self.counts[i] += 1;
            }
            return true;
        }
    };
    var col: Collector = .{};
    const obs: batch_sched.StepObserver = .{
        .ctx = &col,
        .onPrefillDoneFn = Collector.onPrefillDone,
        .onTokenFn = Collector.onToken,
    };
    const backend = rt.batchBackend();

    // Warm prefix checkpoint `p0` (`8` tokens with exact GDN state snapshot!).
    _ = try sched.enqueue(prompts[0], .{ .request_id = 1, .user_tag = 0, .max_tokens = gen_len, .ignore_eos = true });
    while (sched.activeSlotCount() > 0) {
        _ = sched.step(backend, obs);
    }
    try std.testing.expectEqualSlices(u32, &ref_out[0], &col.tokens[0]);

    // Now run `p1` and `p2` concurrently: both hit the `8`-token GDN+QSA checkpoint!
    _ = try sched.enqueue(prompts[1], .{ .request_id = 2, .user_tag = 1, .max_tokens = gen_len, .ignore_eos = true });
    _ = try sched.enqueue(prompts[2], .{ .request_id = 3, .user_tag = 2, .max_tokens = gen_len, .ignore_eos = true });
    while (sched.activeSlotCount() > 0) {
        _ = sched.step(backend, obs);
    }

    try std.testing.expectEqual(@as(u32, 8), col.hits[1]);
    try std.testing.expectEqual(@as(u32, 8), col.hits[2]);
    try std.testing.expect(sched.telemetry().max_observed_batch_size >= 2);
    try std.testing.expectEqualSlices(u32, &ref_out[1], &col.tokens[1]);
    try std.testing.expectEqualSlices(u32, &ref_out[2], &col.tokens[2]);
    const st = rt.stats();
    try std.testing.expect(st.gdn_snapshot_clones >= 3);
    try std.testing.expectEqual(@as(u32, 0), st.prefix_cache_pinned_entries);
}

test "M6.8 G15+G16+R9 GraphModel wide prefill + tail pruning + 128-tok SWA wrap + LoopLM R=2 + aliasing + canary + live-RAM" {
    const allocator = std.testing.allocator;

    // 1a. Full-context hybrid 3:1 GDN+QSA+GR+MoE (`qsa_window_size = 0`, 150 tokens across 128+22 chunks):
    // verifies `wide` + `P0-B1` pruning vs legacy `tree64` parity, `G16` canary, `P0-B2` scratch aliasing,
    // `verifyTree` transition over aliased pages, and `P1-B` live RAM guard.
    {
        var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 1, 0, null);
        defer bundle.deinit(allocator);
        bundle.cfg.qsa_window_size = 0;
        bundle.cfg.qsa_top_k_blocks = 8;

        var tree_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 256,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .tree64,
            .startup_canary = true,
        });
        defer tree_m.deinit(allocator);
        try std.testing.expect(!tree_m.usesWideGpuPrefill());

        var wide_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 256,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 128,
            .alias_prefill_tree_scratch = true,
            .enable_prompt_tail_pruning = true,
            .startup_canary = true,
        });
        defer wide_m.deinit(allocator);
        try std.testing.expect(wide_m.usesWideGpuPrefill());
        try std.testing.expect(wide_m.canary_result != null);
        try std.testing.expect(wide_m.canary_result.?.max_abs_err_f16 < 1.0e-2);
        try std.testing.expect(wide_m.aliased_scratch_saved_bytes > 0);

        var prompt: [150]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast((i * 13 + 7) % bundle.cfg.vocab_size);

        const t0_tree = try tree_m.prefillChunk(&prompt, 0, true);
        const t0_wide = try wide_m.prefillChunk(&prompt, 0, true);
        const tl0 = tree_m.readLastLogitsF16();
        const wl0 = wide_m.readLastLogitsF16();
        var max_diff: f32 = 0.0;
        for (tl0, wl0) |a, b| {
            const d = @abs(@as(f32, a) - @as(f32, b));
            if (d > max_diff) max_diff = d;
        }
        try std.testing.expect(max_diff < 5.0e-2);
        try std.testing.expect(@abs(@as(f32, wl0[t0_wide]) - @as(f32, wl0[t0_tree])) <= max_diff * 2.0 + 1.0e-3);

        // Also prove `P0-B1` tail pruning vs unpruned `wide` is tight (`< 5e-3`), and
        // `P0-B2` aliased scratch (`alias_prefill_tree_scratch = true`) is bit-exact vs unaliased!
        var wide_unaliased = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 256,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 128,
            .alias_prefill_tree_scratch = false,
            .enable_prompt_tail_pruning = false,
            .startup_canary = false,
        });
        defer wide_unaliased.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 0), wide_unaliased.aliased_scratch_saved_bytes);
        _ = try wide_unaliased.prefillChunk(&prompt, 0, true);
        var prune_diff: f32 = 0.0;
        for (wide_unaliased.readLastLogitsF16(), wl0) |a, b| {
            const d = @abs(@as(f32, a) - @as(f32, b));
            if (d > prune_diff) prune_diff = d;
        }
        try std.testing.expect(prune_diff < 5.0e-3);

        // 3 decode steps fed identical tokens after wide prefill + `verifyTree` over `P0-B2` aliased scratch.
        var cur = t0_wide;
        for (0..3) |s| {
            const pos: u32 = @intCast(prompt.len + s);
            _ = try tree_m.stepDecode(cur, pos);
            _ = try wide_unaliased.stepDecode(cur, pos);
            cur = try wide_m.stepDecode(cur, pos);
            var step_diff: f32 = 0.0;
            for (tree_m.readLastLogitsF16(), wide_m.readLastLogitsF16()) |a, b| {
                const d = @abs(@as(f32, a) - @as(f32, b));
                if (d > step_diff) step_diff = d;
            }
            try std.testing.expect(step_diff < 5.0e-2);
        }
        const bp: u32 = @intCast(prompt.len + 3);
        const ctoks = [_]u32{ cur, 5, 9 };
        const cpos = [_]u32{ bp, bp + 1, bp + 2 };
        const cmasks = [_]u64{ 0b001, 0b011, 0b111 };
        const vru = try wide_unaliased.verifyTree(&ctoks, &cpos, &cmasks, bp);
        const vrw = try wide_m.verifyTree(&ctoks, &cpos, &cmasks, bp);
        var tree_diff: f32 = 0.0;
        for (vru.logits_f16, vrw.logits_f16) |a, b| {
            const d = @abs(@as(f32, a) - @as(f32, b));
            if (d > tree_diff) tree_diff = d;
        }
        try std.testing.expect(tree_diff < 5.0e-3);
        try std.testing.expectEqualSlices(u32, vru.argmax_tokens, vrw.argmax_tokens);

        // `P1-B` live RAM guard check.
        try std.testing.expectError(
            error.ExceedsLiveZeroSwapRam,
            GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
                .max_seq_len = 64,
                .strict_live_ram = true,
                .live_reclaimable_override = 16384,
            }),
        );
    }

    // 1b. Circular 128-tok SWA wrap (`150` tokens > `128` ring capacity):
    // proves 2-chunk (`128 + 22`) pruned wide prefill matches single-shot unpruned `150`-tok wide SWA
    // with zero ring overwrite and identical post-wrap decode continuation!
    {
        var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 1, 0, null);
        defer bundle.deinit(allocator);
        bundle.cfg.qsa_window_size = hybrid.QSA_SWA_WINDOW_TOKENS;

        var single_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 256,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 256,
            .enable_prompt_tail_pruning = false,
            .startup_canary = false,
        });
        defer single_m.deinit(allocator);

        var chunked_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 256,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 128,
            .enable_prompt_tail_pruning = true,
            .startup_canary = false,
        });
        defer chunked_m.deinit(allocator);

        var prompt: [150]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast((i * 13 + 7) % bundle.cfg.vocab_size);
        const t0_single = try single_m.prefillChunk(&prompt, 0, true);
        const t0_chunked = try chunked_m.prefillChunk(&prompt, 0, true);
        var max_diff: f32 = 0.0;
        for (single_m.readLastLogitsF16(), chunked_m.readLastLogitsF16()) |a, b| {
            const d = @abs(@as(f32, a) - @as(f32, b));
            if (d > max_diff) max_diff = d;
        }
        try std.testing.expect(max_diff < 5.0e-2);
        try std.testing.expectEqual(t0_single, t0_chunked);

        var cur_s = t0_single;
        var cur_c = t0_chunked;
        for (0..3) |s| {
            const pos: u32 = @intCast(prompt.len + s);
            cur_s = try single_m.stepDecode(cur_s, pos);
            cur_c = try chunked_m.stepDecode(cur_c, pos);
            try std.testing.expectEqual(cur_s, cur_c);
        }
    }

    // 2. Hybrid `LoopSchedule R = 2` wide prefill parity (`r = 0` write, `r = 1` read-only + `P0-B1` final-pass pruning).
    {
        var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 2, 0, null);
        defer bundle.deinit(allocator);
        bundle.cfg.qsa_window_size = 0;

        var tree_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .tree64,
            .startup_canary = false,
        });
        defer tree_m.deinit(allocator);

        var wide_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 32,
            .enable_prompt_tail_pruning = true,
            .startup_canary = false,
        });
        defer wide_m.deinit(allocator);

        var prompt: [44]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast((i * 19 + 1) % bundle.cfg.vocab_size);
        const t0_tree = try tree_m.prefillChunk(&prompt, 0, true);
        const t0_wide = try wide_m.prefillChunk(&prompt, 0, true);
        const tl0 = tree_m.readLastLogitsF16();
        const wl0 = wide_m.readLastLogitsF16();
        var max_diff: f32 = 0.0;
        for (tl0, wl0) |a, b| {
            const d = @abs(@as(f32, a) - @as(f32, b));
            if (d > max_diff) max_diff = d;
        }
        try std.testing.expect(max_diff < 5.0e-2);
        try std.testing.expect(@abs(@as(f32, wl0[t0_wide]) - @as(f32, wl0[t0_tree])) <= max_diff * 2.0 + 1.0e-3);
        _ = try tree_m.stepDecode(t0_wide, @intCast(prompt.len));
        _ = try wide_m.stepDecode(t0_wide, @intCast(prompt.len));
        var d1: f32 = 0.0;
        for (tree_m.readLastLogitsF16(), wide_m.readLastLogitsF16()) |a, b| {
            const d = @abs(@as(f32, a) - @as(f32, b));
            if (d > d1) d1 = d;
        }
        try std.testing.expect(d1 < 5.0e-2);
    }
}

test "M6.8 C15 GraphBatchRuntime wide prefill + ChatML message-boundary PrefixCache checkpoint with GDN snapshot" {
    const allocator = std.testing.allocator;
    var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 1, 0, null);
    defer bundle.deinit(allocator);
    bundle.cfg.qsa_window_size = 0;

    var serial_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .gpu_prefill_mode = .wide,
        .wide_prefill_chunk_tokens = 128,
    });
    defer serial_m.deinit(allocator);

    var batch_m = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .gpu_prefill_mode = .wide,
        .wide_prefill_chunk_tokens = 128,
    });
    defer batch_m.deinit(allocator);

    var rt = try GraphBatchRuntime.init(allocator, &batch_m, .{
        .max_slots = 4,
        .prefix_cache_chunks = 8,
        .gpu_prefill_mode = .wide,
    });
    defer rt.deinit(allocator);

    // 18-token shared multi-turn prefix (strictly intra-tile `< 512`!), followed by edited final user turn.
    var p0: [26]u32 = undefined;
    var p1: [28]u32 = undefined;
    for (0..18) |i| {
        const tok: u32 = @intCast((i * 7 + 3) % bundle.cfg.vocab_size);
        p0[i] = tok;
        p1[i] = tok;
    }
    for (18..26) |i| p0[i] = @intCast((i * 3 + 11) % bundle.cfg.vocab_size);
    for (18..28) |i| p1[i] = @intCast((i * 5 + 23) % bundle.cfg.vocab_size);

    const gen_len: usize = 4;
    var ref_p1: [gen_len]u32 = undefined;
    serial_m.resetKvCache();
    var tok = try serial_m.prefillChunk(&p1, 0, true);
    ref_p1[0] = tok;
    for (1..gen_len) |s| {
        tok = try serial_m.stepDecode(tok, @intCast(p1.len + s - 1));
        ref_p1[s] = tok;
    }

    var sched = try batch_sched.BatchScheduler.init(allocator, .{
        .max_batch_size = 4,
        .max_prefill_chunk_tokens = 128,
        .max_prompt_tokens = 256,
        .message_boundary_cache = true,
    });
    defer sched.deinit(allocator);

    const Collector = struct {
        tokens: [2][gen_len]u32 = @splat(@splat(0)),
        counts: [2]usize = @splat(0),
        hits: [2]u32 = @splat(0),
        fn onPrefillDone(ctx: *anyopaque, _: u16, tag: u64, _: u32, hit: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.hits[@intCast(tag)] = hit;
            return true;
        }
        fn onToken(ctx: *anyopaque, _: u16, tag: u64, tid: u32, _: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const idx: usize = @intCast(tag);
            if (self.counts[idx] < gen_len) {
                self.tokens[idx][self.counts[idx]] = tid;
                self.counts[idx] += 1;
            }
            return true;
        }
    };
    var col: Collector = .{};
    const obs: batch_sched.StepObserver = .{
        .ctx = &col,
        .onPrefillDoneFn = Collector.onPrefillDone,
        .onTokenFn = Collector.onToken,
    };
    const backend = rt.batchBackend();

    _ = try sched.enqueue(&p0, .{
        .request_id = 1,
        .user_tag = 0,
        .max_tokens = gen_len,
        .ignore_eos = true,
        .message_boundary_token = 18,
    });
    while (sched.activeSlotCount() > 0) _ = sched.step(backend, obs);
    try std.testing.expectEqual(@as(u32, 0), col.hits[0]);

    _ = try sched.enqueue(&p1, .{
        .request_id = 2,
        .user_tag = 1,
        .max_tokens = gen_len,
        .ignore_eos = true,
        .message_boundary_token = 18,
    });
    while (sched.activeSlotCount() > 0) _ = sched.step(backend, obs);

    try std.testing.expectEqual(@as(u32, 18), col.hits[1]);
    try std.testing.expectEqualSlices(u32, &ref_p1, &col.tokens[1]);
    const st = rt.stats();
    try std.testing.expect(st.boundary_hits >= 1);
    try std.testing.expect(st.gdn_snapshot_clones >= 2);
    try std.testing.expectEqual(@as(u32, 0), st.prefix_cache_pinned_entries);
}

test "M6.8 ADR-0019 Isolation #2 & #3: wide prefill R=3 QSA sparse tail_m1_last reuse and PLE zero-overwrite reuse" {
    const allocator = std.testing.allocator;

    // Part A: R = 3, qsa_window_size = 0, top_k_blocks = 2, m = 160 (> 128 so sparse top-k
    // block indexing is active on the final QSA layer `L = 3`), verifying `tail_m1_last`
    // reuses `r = 0`'s row `m - 1` selected blocks (`qsa_index_computes == 1`, `qsa_index_reuses == 2`).
    {
        var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 3, 0, null);
        defer bundle.deinit(allocator);
        bundle.cfg.qsa_window_size = 0;
        bundle.cfg.qsa_top_k_blocks = 2;

        var m_pruned = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 256,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 256,
            .enable_prompt_tail_pruning = true,
        });
        defer m_pruned.deinit(allocator);

        var m_unpruned = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 256,
            .max_tree_nodes = 16,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 256,
            .enable_prompt_tail_pruning = false,
        });
        defer m_unpruned.deinit(allocator);

        var prompt: [160]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast((i * 13 + 7) % bundle.cfg.vocab_size);

        m_pruned.resetLoopIsolationStats();
        m_unpruned.resetLoopIsolationStats();
        const tok_p = try m_pruned.prefillChunk(&prompt, 0, true);
        const tok_u = try m_unpruned.prefillChunk(&prompt, 0, true);

        const iso_p = m_pruned.loopIsolationStats();
        const iso_u = m_unpruned.loopIsolationStats();
        try std.testing.expectEqual(@as(u64, 1), iso_p.qsa_index_computes);
        try std.testing.expectEqual(@as(u64, 2), iso_p.qsa_index_reuses);
        try std.testing.expectEqual(@as(u64, 1), iso_u.qsa_index_computes);
        try std.testing.expectEqual(@as(u64, 2), iso_u.qsa_index_reuses);
        try std.testing.expectEqual(tok_u, tok_p);

        var max_diff: f32 = 0.0;
        for (m_pruned.readLastLogitsF16(), m_unpruned.readLastLogitsF16()) |a, b| {
            const d = @abs(@as(f32, a) - @as(f32, b));
            if (d > max_diff) max_diff = d;
        }
        try std.testing.expect(max_diff < 5.0e-3);
    }

    // Part B: R = 3 with NVMe PLE (`ple_num_shards = 1`): `preparePleLayerWide` populates
    // host-mapped `wp_ple_f16_host` only at `loop_step == 0` and reuses it on `r >= 1`
    // without overwriting before `frame.commit()` (`ple_r_ge1_reuses == 6`, `0` extra NVMe reads).
    {
        const tmp_dir = try makeTestTempDir(allocator, "m68_ple_r3_wide");
        defer {
            removeTestDir(allocator, tmp_dir);
            allocator.free(tmp_dir);
        }
        var bundle = try buildSynthQwen4TestBundle(allocator, 8, 2, 0, 3, 1, tmp_dir);
        defer bundle.deinit(allocator);
        bundle.cfg.qsa_window_size = 0;

        var m_wide = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 16,
            .ple_paths = bundle.ple_paths,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 64,
            .enable_prompt_tail_pruning = true,
        });
        defer m_wide.deinit(allocator);

        var m_tree = try GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 32,
            .ple_paths = bundle.ple_paths,
            .gpu_prefill_mode = .tree64,
        });
        defer m_tree.deinit(allocator);

        var prompt: [32]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast((i * 11 + 5) % bundle.cfg.vocab_size);

        m_wide.resetLoopIsolationStats();
        const tok_w = try m_wide.prefillChunk(&prompt, 0, true);
        const tok_t = try m_tree.prefillChunk(&prompt, 0, true);

        const iso_w = m_wide.loopIsolationStats();
        try std.testing.expectEqual(@as(u64, 4), iso_w.ple_r0_fills);
        try std.testing.expectEqual(@as(u64, 6), iso_w.ple_r_ge1_reuses);
        try std.testing.expectEqual(@as(u64, 0), iso_w.ple_r_ge1_nvme_reads);
        try std.testing.expectEqual(@as(u64, 1), iso_w.qsa_index_computes);
        try std.testing.expectEqual(@as(u64, 2), iso_w.qsa_index_reuses);
        try std.testing.expectEqual(tok_t, tok_w);

        var max_diff: f32 = 0.0;
        for (m_wide.readLastLogitsF16(), m_tree.readLastLogitsF16()) |a, b| {
            const d = @abs(@as(f32, a) - @as(f32, b));
            if (d > max_diff) max_diff = d;
        }
        try std.testing.expect(max_diff < 5.0e-2);
    }
}
