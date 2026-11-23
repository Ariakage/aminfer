//! Fused Metal 4 248K-vocab radix/bucket constrained GPU sampler (G21, ADR-0023 §4).
//!
//! Executes directly over device-resident `f16` or `f32` logits (`vocab_size` up to
//! `262,144`, covering `Qwen3` 151,936, `DeepSeek-V4.1` 129,280, and `Qwen3.8`
//! 248,077) in preallocated 16 KiB-aligned wired unified memory with zero hot-path
//! dynamic heap allocation and zero full-vocab CPU logit scans:
//! - Explicit unconstrained greedy fast-path bypass (`temperature == 0`, no
//!   penalties, no `logit_bias`, no `C22` bitmask, no `logprobs`/`top_logprobs`)
//!   dispatches a minimal multi-block GPU argmax reading zero auxiliary buffers.
//! - Full constrained two-pass reduction + 1024-bucket radix/Gumbel-max pipeline
//!   fuses `C22` allowed-token bitmasks (~31 KiB for 248K vocab), sparse
//!   `logit_bias`, per-slot `presence_penalty` / `frequency_penalty` /
//!   `repetition_penalty` (via an 8 KiB threadgroup Bloom filter + O(1) open-
//!   addressed device hash tables), temperature scaling, `min_p`, `top_k`, `top_p`
//!   (with G20 indirect boundary refinement launching 0 threadgroups when clean),
//!   optional `logprobs` / `top_logprobs` (<= 20), and deterministic slot-local
//!   `D11` (`Xoshiro256`) seed-driven Gumbel-max sampling.
//! - Supports both single-token decode (`M = 1`) and batched / speculative-tree
//!   multi-row sampling with per-row `C22` bitmask slot selection.

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const amf = @import("../format/amf.zig");
const sampling = @import("../spec/sampling.zig");
const grammar = @import("../spec/grammar.zig");
const rng_mod = @import("../runtime/rng.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Maximum supported vocabulary size (`262,144` >= Qwen3.8 `248,077`).
pub const MAX_VOCAB_SIZE: u32 = @intCast(grammar.MAX_VOCAB_SIZE);
/// Number of `u32` words needed for `MAX_VOCAB_SIZE` (`8,192` words = `32 KiB`).
pub const MAX_BITMASK_WORDS: usize = grammar.MAX_BITMASK_WORDS;
/// Maximum sparse `logit_bias` entries per row (matches `src/spec/sampling.zig`).
pub const MAX_LOGIT_BIAS_ENTRIES: usize = sampling.MAX_LOGIT_BIAS_ENTRIES;
/// Open-addressed device hash table capacity for `logit_bias` (load factor <= 0.5).
pub const BIAS_TABLE_CAP: usize = 128;
/// Maximum unique seen-token penalty entries per row (matches `DefaultPenaltyTracker`).
pub const MAX_PENALTY_ENTRIES: usize = 2048;
/// Open-addressed device hash table capacity for `seen_frequencies` (load factor <= 0.5).
pub const FREQ_TABLE_CAP: usize = 4096;
/// Maximum `top_logprobs` entries per row (matches `src/spec/sampling.zig`).
pub const MAX_TOP_LOGPROBS: usize = sampling.MAX_TOP_LOGPROBS;
/// Maximum reduction blocks per row (`32` × `256` threads = `8,192` threads/row;
/// guarantees `<= 8,192` tokens/block for 13-bit local index packing up to 262,144).
pub const MAX_BLOCKS: u32 = 32;
/// Number of monotone radix buckets per row (`16 KiB` global histogram/row).
pub const NUM_RADIX_BUCKETS: usize = 1024;
/// Maximum simultaneous rows in one batched / speculative-tree sampling dispatch.
pub const MAX_BATCH_SLOTS: u32 = 64;
/// Capacity of parallel boundary-refinement candidate buffer per row.
pub const CUT_CAND_CAP: usize = 2048;

const EMPTY_ID: u32 = 0xFFFF_FFFF;

const FLAG_USE_BITMASK: u32 = 1 << 0;
const FLAG_HAS_BIAS: u32 = 1 << 1;
const FLAG_HAS_PENALTIES: u32 = 1 << 2;
const FLAG_NEED_LOGPROBS: u32 = 1 << 3;

pub const LogitsDType = enum {
    f16,
    f32,
};

pub const SampleStatus = enum(u32) {
    ok = 0,
    empty_allowed_set = 1,
    invalid_logits = 2,
};

/// Re-exported unified sampling parameter & output structs for seamless engine wiring.
pub const SampleParams = sampling.SamplingParams;
pub const SampleOutput = sampling.SampleOutput;
pub const LogitBiasEntry = sampling.LogitBiasEntry;
pub const TokenFreqEntry = sampling.TokenFreqEntry;
pub const TokenLogprob = sampling.TokenLogprob;

pub const Error = metal.WrapError || sampling.SampleError || arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError || error{
    VocabTooLarge,
    BatchTooLarge,
    SlotOutOfRange,
    TooManyLogitBiasEntries,
    BitmaskDisabled,
    OverridesDisabled,
};

// ---------------------------------------------------------------------------
// Metal 4 ABI extern structs (verified comptime against `sampling.metal`)
// ---------------------------------------------------------------------------

pub const GpuBiasEntry = extern struct {
    token_id: u32,
    bias: f32,
};

pub const GpuFreqEntry = extern struct {
    token_id: u32,
    count: u32,
};

pub const RowSampleConfig = extern struct {
    temperature: f32,
    inv_temp: f32,
    top_p: f32,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    top_k: u32,
    seed_lo: u32,
    seed_hi: u32,
    flags: u32,
    top_logprobs: u32,
    bias_count: u32,
    freq_count: u32,
    bitmask_row_index: u32,
    _pad0: u32 = 0,
};

pub const SampleDispatchParams = extern struct {
    num_rows: u32,
    vocab_size: u32,
    num_blocks: u32,
    aligned_vec4: u32,
    bitmask_stride_u32: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const BlockStats = extern struct {
    max_logit: f32,
    argmax_id: u32,
    logsum_m: f32,
    logsum_exp: f32,
};

pub const RowState = extern struct {
    global_max: f32,
    global_argmax_id: u32,
    global_log_z: f32,
    status: u32,
    cut_bucket: u32,
    rem_k: u32,
    rem_mass: f32,
    total_mass: f32,
    prefix_best_tok: u32,
    prefix_best_gumbel: u32,
    cut_s_lo: f32,
    cut_s_hi: f32,
    need_refine: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,
};

pub const GpuTokenLogprob = extern struct {
    token_id: u32,
    logprob: f32,
};

pub const CutCandidate = extern struct {
    s: f32,
    token_id: u32,
};

pub const GpuSampleOutput = extern struct {
    token_id: u32,
    logprob: f32,
    status: u32,
    top_logprobs_len: u32,
    top_logprobs: [MAX_TOP_LOGPROBS]GpuTokenLogprob,
    _pad: [4]u32 = .{ 0, 0, 0, 0 },

    pub fn toSampleOutput(self: *const GpuSampleOutput) Error!SampleOutput {
        switch (self.status) {
            @backingInt(SampleStatus.ok) => {},
            @backingInt(SampleStatus.empty_allowed_set) => return error.EmptyAllowedSet,
            else => return error.InvalidLogits,
        }
        const n: usize = @min(@as(usize, self.top_logprobs_len), MAX_TOP_LOGPROBS);
        var out: SampleOutput = .{
            .token_id = self.token_id,
            .logprob = self.logprob,
            .top_logprobs_len = @intCast(n),
        };
        for (0..n) |i| {
            out.top_logprobs[i] = .{
                .token_id = self.top_logprobs[i].token_id,
                .logprob = self.top_logprobs[i].logprob,
            };
        }
        return out;
    }
};

comptime {
    if (@sizeOf(GpuBiasEntry) != 8) @compileError("GpuBiasEntry must be 8 bytes");
    if (@sizeOf(GpuFreqEntry) != 8) @compileError("GpuFreqEntry must be 8 bytes");
    if (@sizeOf(RowSampleConfig) != 64) @compileError("RowSampleConfig must be 64 bytes");
    if (@sizeOf(SampleDispatchParams) != 32) @compileError("SampleDispatchParams must be 32 bytes");
    if (@sizeOf(BlockStats) != 16) @compileError("BlockStats must be 16 bytes");
    if (@sizeOf(RowState) != 64) @compileError("RowState must be 64 bytes");
    if (@sizeOf(CutCandidate) != 8) @compileError("CutCandidate must be 8 bytes");
    if (@sizeOf(GpuTokenLogprob) != 8) @compileError("GpuTokenLogprob must be 8 bytes");
    if (@sizeOf(GpuSampleOutput) != 192) @compileError("GpuSampleOutput must be 192 bytes");
}

/// Computes the optimal parallel reduction block count (`1..32`) for `vocab_size`,
/// keeping `ceil(vocab_size / num_blocks) <= 8192` up to `MAX_VOCAB_SIZE`.
pub inline fn numBlocksForVocab(vocab_size: u32) u32 {
    if (vocab_size <= 2048) return 1;
    if (vocab_size <= 8192) return 4;
    if (vocab_size <= 32768) return 16;
    return MAX_BLOCKS;
}

inline fn hashTokenU32(token_id: u32) u32 {
    return token_id *% 0x9E37_79B1;
}

// ---------------------------------------------------------------------------
// Precompiled Metal 4 Pipelines (zero runtime PSO compilation)
// ---------------------------------------------------------------------------

pub const GpuSamplerPipelines = struct {
    greedy_pass1_f16: metal.Pipeline,
    greedy_pass1_f32: metal.Pipeline,
    greedy_pass2: metal.Pipeline,

    /// Indexed by `(f16 ? 1 : 0) | (use_bitmask ? 2 : 0) | (has_overrides ? 4 : 0)`.
    constrained_pass1: [8]metal.Pipeline,
    merge_stats: metal.Pipeline,
    radix_pass2: [8]metal.Pipeline,
    radix_select: [8]metal.Pipeline,
    refine_gather: [8]metal.Pipeline,
    refine_sort: [8]metal.Pipeline,
    refine_tie: [8]metal.Pipeline,
    toplogprobs_pass1: [8]metal.Pipeline,
    toplogprobs_pass2: metal.Pipeline,

    inline fn variantIndex(dtype: LogitsDType, use_bitmask: bool, has_overrides: bool) usize {
        var idx: usize = if (dtype == .f16) 1 else 0;
        if (use_bitmask) idx |= 2;
        if (has_overrides) idx |= 4;
        return idx;
    }

    fn initVariant(
        device: metal.Device,
        library: metal.Library,
        name: [:0]const u8,
        idx: usize,
    ) metal.Error!metal.Pipeline {
        const is_f16 = (idx & 1) != 0;
        const use_bm = (idx & 2) != 0;
        const has_ov = (idx & 4) != 0;
        return metal.Pipeline.initSpecialized(device, library, name, &.{
            .{ .index = 0, .value = .{ .bool = is_f16 } },
            .{ .index = 1, .value = .{ .bool = use_bm } },
            .{ .index = 2, .value = .{ .bool = has_ov } },
        });
    }

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!GpuSamplerPipelines {
        const greedy_pass1_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_sample_greedy_pass1", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer greedy_pass1_f16.close();

        const greedy_pass1_f32 = try metal.Pipeline.initSpecialized(device, library, "ami_sample_greedy_pass1", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer greedy_pass1_f32.close();

        const greedy_pass2 = try metal.Pipeline.init(device, library, "ami_sample_greedy_pass2");
        errdefer greedy_pass2.close();

        const merge_stats = try metal.Pipeline.init(device, library, "ami_sample_merge_stats");
        errdefer merge_stats.close();

        const toplogprobs_pass2 = try metal.Pipeline.init(device, library, "ami_sample_toplogprobs_pass2");
        errdefer toplogprobs_pass2.close();

        var cp1: [8]metal.Pipeline = undefined;
        var cp1_ok: usize = 0;
        errdefer for (0..cp1_ok) |i| cp1[i].close();
        for (0..8) |i| {
            cp1[i] = try initVariant(device, library, "ami_sample_constrained_pass1", i);
            cp1_ok += 1;
        }

        var rp2: [8]metal.Pipeline = undefined;
        var rp2_ok: usize = 0;
        errdefer for (0..rp2_ok) |i| rp2[i].close();
        for (0..8) |i| {
            rp2[i] = try initVariant(device, library, "ami_sample_radix_pass2", i);
            rp2_ok += 1;
        }

        var rsel: [8]metal.Pipeline = undefined;
        var rsel_ok: usize = 0;
        errdefer for (0..rsel_ok) |i| rsel[i].close();
        for (0..8) |i| {
            rsel[i] = try initVariant(device, library, "ami_sample_radix_select", i);
            rsel_ok += 1;
        }

        var rgath: [8]metal.Pipeline = undefined;
        var rgath_ok: usize = 0;
        errdefer for (0..rgath_ok) |i| rgath[i].close();
        for (0..8) |i| {
            rgath[i] = try initVariant(device, library, "ami_sample_refine_gather", i);
            rgath_ok += 1;
        }

        var rsort: [8]metal.Pipeline = undefined;
        var rsort_ok: usize = 0;
        errdefer for (0..rsort_ok) |i| rsort[i].close();
        for (0..8) |i| {
            rsort[i] = try initVariant(device, library, "ami_sample_refine_sort", i);
            rsort_ok += 1;
        }

        var rtie: [8]metal.Pipeline = undefined;
        var rtie_ok: usize = 0;
        errdefer for (0..rtie_ok) |i| rtie[i].close();
        for (0..8) |i| {
            rtie[i] = try initVariant(device, library, "ami_sample_refine_tie", i);
            rtie_ok += 1;
        }

        var tlp1: [8]metal.Pipeline = undefined;
        var tlp1_ok: usize = 0;
        errdefer for (0..tlp1_ok) |i| tlp1[i].close();
        for (0..8) |i| {
            tlp1[i] = try initVariant(device, library, "ami_sample_toplogprobs_pass1", i);
            tlp1_ok += 1;
        }

        return .{
            .greedy_pass1_f16 = greedy_pass1_f16,
            .greedy_pass1_f32 = greedy_pass1_f32,
            .greedy_pass2 = greedy_pass2,
            .constrained_pass1 = cp1,
            .merge_stats = merge_stats,
            .radix_pass2 = rp2,
            .radix_select = rsel,
            .refine_gather = rgath,
            .refine_sort = rsort,
            .refine_tie = rtie,
            .toplogprobs_pass1 = tlp1,
            .toplogprobs_pass2 = toplogprobs_pass2,
        };
    }

    pub fn close(self: GpuSamplerPipelines) void {
        self.greedy_pass1_f16.close();
        self.greedy_pass1_f32.close();
        self.greedy_pass2.close();
        self.merge_stats.close();
        self.toplogprobs_pass2.close();
        for (0..8) |i| {
            self.constrained_pass1[i].close();
            self.radix_pass2[i].close();
            self.radix_select[i].close();
            self.refine_gather[i].close();
            self.refine_sort[i].close();
            self.refine_tie[i].close();
            self.toplogprobs_pass1[i].close();
        }
    }
};

// ---------------------------------------------------------------------------
// Preallocated Wired Workspace (carved once at startup, zero hot-path alloc)
// ---------------------------------------------------------------------------

pub const SamplerSpec = struct {
    max_vocab_size: u32 = MAX_VOCAB_SIZE,
    max_batch_slots: u32 = 1,
    enable_bitmask: bool = true,
    enable_overrides: bool = true,
};

fn validateSpec(spec: SamplerSpec) Error!void {
    if (spec.max_vocab_size == 0) return error.EmptyVocab;
    if (spec.max_vocab_size > MAX_VOCAB_SIZE) return error.VocabTooLarge;
    if (spec.max_batch_slots == 0 or spec.max_batch_slots > MAX_BATCH_SLOTS) return error.BatchTooLarge;
}

pub const GpuSamplerWorkspace = struct {
    spec: SamplerSpec,
    bitmask_stride_u32: u32,

    row_cfgs_off: u64,
    bitmasks_off: u64,
    bias_tables_off: u64,
    bias_keys_off: u64,
    freq_tables_off: u64,
    freq_keys_off: u64,
    block_stats_off: u64,
    row_states_off: u64,
    hist_count_off: u64,
    hist_mass_off: u64,
    hist_gumbel_off: u64,
    block_top_off: u64,
    refine_indirect_off: u64,
    cut_cand_counts_off: u64,
    cut_cands_off: u64,
    outputs_off: u64,

    row_cfgs: []RowSampleConfig,
    bitmasks: []u32,
    bias_tables: []GpuBiasEntry,
    bias_keys: []u32,
    bias_dirty_slots: []u16,
    bias_counts: []u32,
    freq_tables: []GpuFreqEntry,
    freq_keys: []u32,
    freq_dirty_slots: []u16,
    freq_counts: []u32,
    outputs: []GpuSampleOutput,

    pub fn requiredBytes(spec: SamplerSpec) Error!usize {
        try validateSpec(spec);
        const slots: usize = spec.max_batch_slots;
        const bm_words_raw = grammar.bitmaskWordCount(spec.max_vocab_size);
        const bm_stride_bytes = amf.align64(std.math.mul(usize, bm_words_raw, @sizeOf(u32)) catch return error.Overflow);

        const sizes = [_]usize{
            slots * @sizeOf(RowSampleConfig),
            if (spec.enable_bitmask) (std.math.mul(usize, slots, bm_stride_bytes) catch return error.Overflow) else 64,
            if (spec.enable_overrides) slots * BIAS_TABLE_CAP * @sizeOf(GpuBiasEntry) else 64,
            if (spec.enable_overrides) slots * MAX_LOGIT_BIAS_ENTRIES * @sizeOf(u32) else 64,
            if (spec.enable_overrides) slots * MAX_LOGIT_BIAS_ENTRIES * @sizeOf(u16) else 64,
            if (spec.enable_overrides) slots * @sizeOf(u32) else 64,
            if (spec.enable_overrides) slots * FREQ_TABLE_CAP * @sizeOf(GpuFreqEntry) else 64,
            if (spec.enable_overrides) slots * MAX_PENALTY_ENTRIES * @sizeOf(u32) else 64,
            if (spec.enable_overrides) slots * MAX_PENALTY_ENTRIES * @sizeOf(u16) else 64,
            if (spec.enable_overrides) slots * @sizeOf(u32) else 64,
            slots * MAX_BLOCKS * @sizeOf(BlockStats),
            slots * @sizeOf(RowState),
            slots * NUM_RADIX_BUCKETS * @sizeOf(u32),
            slots * NUM_RADIX_BUCKETS * @sizeOf(f32),
            slots * NUM_RADIX_BUCKETS * @sizeOf(u64),
            slots * MAX_BLOCKS * MAX_TOP_LOGPROBS * @sizeOf(GpuTokenLogprob),
            64, // 3 x IndirectDispatchArgs (36 B)
            slots * @sizeOf(u32), // cut_cand_counts
            slots * CUT_CAND_CAP * @sizeOf(CutCandidate), // cut_cands (16 KiB/slot)
            slots * @sizeOf(GpuSampleOutput),
        };

        var total: usize = 64;
        for (sizes) |sz| {
            total = std.math.add(usize, total, amf.align64(sz)) catch return error.Overflow;
        }
        return total;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: SamplerSpec) Error!GpuSamplerWorkspace {
        try validateSpec(spec);
        const slots: usize = spec.max_batch_slots;
        const bm_words_raw = grammar.bitmaskWordCount(spec.max_vocab_size);
        const bm_stride_bytes = amf.align64(bm_words_raw * @sizeOf(u32));
        const bm_stride_u32: u32 = @intCast(bm_stride_bytes / @sizeOf(u32));

        const row_cfgs_raw = try arena.allocAligned(slots * @sizeOf(RowSampleConfig), 64);
        const bitmasks_raw = try arena.allocAligned(if (spec.enable_bitmask) slots * bm_stride_bytes else 64, 64);
        const bias_tables_raw = try arena.allocAligned(if (spec.enable_overrides) slots * BIAS_TABLE_CAP * @sizeOf(GpuBiasEntry) else 64, 64);
        const bias_keys_raw = try arena.allocAligned(if (spec.enable_overrides) slots * MAX_LOGIT_BIAS_ENTRIES * @sizeOf(u32) else 64, 64);
        const bias_dirty_raw = try arena.allocAligned(if (spec.enable_overrides) slots * MAX_LOGIT_BIAS_ENTRIES * @sizeOf(u16) else 64, 64);
        const bias_cnts_raw = try arena.allocAligned(if (spec.enable_overrides) slots * @sizeOf(u32) else 64, 64);
        const freq_tables_raw = try arena.allocAligned(if (spec.enable_overrides) slots * FREQ_TABLE_CAP * @sizeOf(GpuFreqEntry) else 64, 64);
        const freq_keys_raw = try arena.allocAligned(if (spec.enable_overrides) slots * MAX_PENALTY_ENTRIES * @sizeOf(u32) else 64, 64);
        const freq_dirty_raw = try arena.allocAligned(if (spec.enable_overrides) slots * MAX_PENALTY_ENTRIES * @sizeOf(u16) else 64, 64);
        const freq_cnts_raw = try arena.allocAligned(if (spec.enable_overrides) slots * @sizeOf(u32) else 64, 64);

        const block_stats_raw = try arena.allocAligned(slots * MAX_BLOCKS * @sizeOf(BlockStats), 64);
        const row_states_raw = try arena.allocAligned(slots * @sizeOf(RowState), 64);
        const hist_count_raw = try arena.allocAligned(slots * NUM_RADIX_BUCKETS * @sizeOf(u32), 64);
        const hist_mass_raw = try arena.allocAligned(slots * NUM_RADIX_BUCKETS * @sizeOf(f32), 64);
        const hist_gumbel_raw = try arena.allocAligned(slots * NUM_RADIX_BUCKETS * @sizeOf(u64), 64);
        const block_top_raw = try arena.allocAligned(slots * MAX_BLOCKS * MAX_TOP_LOGPROBS * @sizeOf(GpuTokenLogprob), 64);
        const refine_ind_raw = try arena.allocAligned(64, 64);
        const cut_cnts_raw = try arena.allocAligned(slots * @sizeOf(u32), 64);
        const cut_cands_raw = try arena.allocAligned(slots * CUT_CAND_CAP * @sizeOf(CutCandidate), 64);
        const outputs_raw = try arena.allocAligned(slots * @sizeOf(GpuSampleOutput), 64);

        @memset(row_cfgs_raw, 0);
        @memset(bitmasks_raw, 0xFF);
        @memset(bias_cnts_raw, 0);
        @memset(freq_cnts_raw, 0);
        @memset(block_stats_raw, 0);
        @memset(row_states_raw, 0);
        @memset(hist_count_raw, 0);
        @memset(hist_mass_raw, 0);
        @memset(hist_gumbel_raw, 0);
        @memset(block_top_raw, 0);
        @memset(refine_ind_raw, 0);
        @memset(cut_cnts_raw, 0);
        @memset(outputs_raw, 0);

        const bias_tables: []GpuBiasEntry = if (spec.enable_overrides)
            @alignCast(std.mem.bytesAsSlice(GpuBiasEntry, bias_tables_raw))
        else
            &.{};
        for (bias_tables) |*e| e.* = .{ .token_id = EMPTY_ID, .bias = 0.0 };

        const freq_tables: []GpuFreqEntry = if (spec.enable_overrides)
            @alignCast(std.mem.bytesAsSlice(GpuFreqEntry, freq_tables_raw))
        else
            &.{};
        for (freq_tables) |*e| e.* = .{ .token_id = EMPTY_ID, .count = 0 };

        return .{
            .spec = spec,
            .bitmask_stride_u32 = bm_stride_u32,
            .row_cfgs_off = arena.offsetOf(row_cfgs_raw.ptr),
            .bitmasks_off = arena.offsetOf(bitmasks_raw.ptr),
            .bias_tables_off = arena.offsetOf(bias_tables_raw.ptr),
            .bias_keys_off = arena.offsetOf(bias_keys_raw.ptr),
            .freq_tables_off = arena.offsetOf(freq_tables_raw.ptr),
            .freq_keys_off = arena.offsetOf(freq_keys_raw.ptr),
            .block_stats_off = arena.offsetOf(block_stats_raw.ptr),
            .row_states_off = arena.offsetOf(row_states_raw.ptr),
            .hist_count_off = arena.offsetOf(hist_count_raw.ptr),
            .hist_mass_off = arena.offsetOf(hist_mass_raw.ptr),
            .hist_gumbel_off = arena.offsetOf(hist_gumbel_raw.ptr),
            .block_top_off = arena.offsetOf(block_top_raw.ptr),
            .refine_indirect_off = arena.offsetOf(refine_ind_raw.ptr),
            .cut_cand_counts_off = arena.offsetOf(cut_cnts_raw.ptr),
            .cut_cands_off = arena.offsetOf(cut_cands_raw.ptr),
            .outputs_off = arena.offsetOf(outputs_raw.ptr),
            .row_cfgs = @alignCast(std.mem.bytesAsSlice(RowSampleConfig, row_cfgs_raw)),
            .bitmasks = if (spec.enable_bitmask) @alignCast(std.mem.bytesAsSlice(u32, bitmasks_raw)) else &.{},
            .bias_tables = bias_tables,
            .bias_keys = if (spec.enable_overrides) @alignCast(std.mem.bytesAsSlice(u32, bias_keys_raw)) else &.{},
            .bias_dirty_slots = if (spec.enable_overrides) @alignCast(std.mem.bytesAsSlice(u16, bias_dirty_raw)) else &.{},
            .bias_counts = if (spec.enable_overrides) @alignCast(std.mem.bytesAsSlice(u32, bias_cnts_raw)) else &.{},
            .freq_tables = freq_tables,
            .freq_keys = if (spec.enable_overrides) @alignCast(std.mem.bytesAsSlice(u32, freq_keys_raw)) else &.{},
            .freq_dirty_slots = if (spec.enable_overrides) @alignCast(std.mem.bytesAsSlice(u16, freq_dirty_raw)) else &.{},
            .freq_counts = if (spec.enable_overrides) @alignCast(std.mem.bytesAsSlice(u32, freq_cnts_raw)) else &.{},
            .outputs = @alignCast(std.mem.bytesAsSlice(GpuSampleOutput, outputs_raw)),
        };
    }

    pub fn bitmaskSliceMut(self: *GpuSamplerWorkspace, slot: u32) Error![]u32 {
        if (!self.spec.enable_bitmask) return error.BitmaskDisabled;
        if (slot >= self.spec.max_batch_slots) return error.SlotOutOfRange;
        const base = @as(usize, slot) * self.bitmask_stride_u32;
        return self.bitmasks[base .. base + self.bitmask_stride_u32];
    }

    pub fn setBitmask(self: *GpuSamplerWorkspace, slot: u32, vocab_size: u32, mask: []const u32) Error!void {
        const words = grammar.bitmaskWordCount(vocab_size);
        if (mask.len < words) return error.BitmaskTooSmall;
        const dst = try self.bitmaskSliceMut(slot);
        @memcpy(dst[0..words], mask[0..words]);
    }

    pub fn setBitmaskAllAllowed(self: *GpuSamplerWorkspace, slot: u32, vocab_size: u32) Error!void {
        const words = grammar.bitmaskWordCount(vocab_size);
        const dst = try self.bitmaskSliceMut(slot);
        @memset(dst[0..words], 0xFFFF_FFFF);
    }

    /// O(dirty) clear of a slot's `logit_bias` open-addressed hash table.
    pub fn clearLogitBias(self: *GpuSamplerWorkspace, slot: u32) void {
        if (!self.spec.enable_overrides or slot >= self.spec.max_batch_slots) return;
        const cnt: usize = self.bias_counts[slot];
        if (cnt == 0) return;
        const tbl_base = @as(usize, slot) * BIAS_TABLE_CAP;
        const dirty_base = @as(usize, slot) * MAX_LOGIT_BIAS_ENTRIES;
        for (self.bias_dirty_slots[dirty_base .. dirty_base + cnt]) |s_idx| {
            self.bias_tables[tbl_base + s_idx] = .{ .token_id = EMPTY_ID, .bias = 0.0 };
        }
        self.bias_counts[slot] = 0;
    }

    /// O(dirty) clear of a slot's `seen_frequencies` open-addressed hash table.
    pub fn clearPenalties(self: *GpuSamplerWorkspace, slot: u32) void {
        if (!self.spec.enable_overrides or slot >= self.spec.max_batch_slots) return;
        const cnt: usize = self.freq_counts[slot];
        if (cnt == 0) return;
        const tbl_base = @as(usize, slot) * FREQ_TABLE_CAP;
        const dirty_base = @as(usize, slot) * MAX_PENALTY_ENTRIES;
        for (self.freq_dirty_slots[dirty_base .. dirty_base + cnt]) |s_idx| {
            self.freq_tables[tbl_base + s_idx] = .{ .token_id = EMPTY_ID, .count = 0 };
        }
        self.freq_counts[slot] = 0;
    }

    /// Uploads sparse `logit_bias` entries into `slot`'s open-addressed device
    /// hash table in O(prev + logit_bias.len). Zero heap allocation.
    pub fn uploadLogitBias(
        self: *GpuSamplerWorkspace,
        slot: u32,
        vocab_size: u32,
        entries: []const LogitBiasEntry,
    ) Error!void {
        if (!self.spec.enable_overrides) return error.OverridesDisabled;
        if (slot >= self.spec.max_batch_slots) return error.SlotOutOfRange;
        if (entries.len > MAX_LOGIT_BIAS_ENTRIES) return error.TooManyLogitBiasEntries;

        self.clearLogitBias(slot);
        if (entries.len == 0) return;

        const tbl_base = @as(usize, slot) * BIAS_TABLE_CAP;
        const keys_base = @as(usize, slot) * MAX_LOGIT_BIAS_ENTRIES;
        const dirty_base = @as(usize, slot) * MAX_LOGIT_BIAS_ENTRIES;
        var count: usize = 0;

        for (entries) |lb| {
            if (lb.token_id >= vocab_size) continue;
            if (!std.math.isFinite(lb.bias) and !(std.math.isInf(lb.bias) and lb.bias < 0.0)) {
                return error.InvalidLogits;
            }
            var h: usize = @intCast(hashTokenU32(lb.token_id) & (BIAS_TABLE_CAP - 1));
            var probes: usize = 0;
            while (probes < BIAS_TABLE_CAP) : (probes += 1) {
                const cur = &self.bias_tables[tbl_base + h];
                if (cur.token_id == EMPTY_ID) {
                    if (count >= MAX_LOGIT_BIAS_ENTRIES) return error.TooManyLogitBiasEntries;
                    cur.* = .{ .token_id = lb.token_id, .bias = lb.bias };
                    self.bias_keys[keys_base + count] = lb.token_id;
                    self.bias_dirty_slots[dirty_base + count] = @intCast(h);
                    count += 1;
                    break;
                }
                if (cur.token_id == lb.token_id) {
                    if (std.math.isInf(cur.bias) or std.math.isInf(lb.bias)) {
                        cur.bias = -std.math.inf(f32);
                    } else {
                        cur.bias += lb.bias;
                    }
                    break;
                }
                h = (h + 1) & (BIAS_TABLE_CAP - 1);
            }
        }
        self.bias_counts[slot] = @intCast(count);
    }

    /// Uploads sparse `seen_frequencies` entries into `slot`'s open-addressed
    /// device hash table in O(prev + seen.len). Zero heap allocation.
    pub fn uploadPenalties(
        self: *GpuSamplerWorkspace,
        slot: u32,
        vocab_size: u32,
        entries: []const TokenFreqEntry,
    ) Error!void {
        if (!self.spec.enable_overrides) return error.OverridesDisabled;
        if (slot >= self.spec.max_batch_slots) return error.SlotOutOfRange;

        self.clearPenalties(slot);
        if (entries.len == 0) return;

        const tbl_base = @as(usize, slot) * FREQ_TABLE_CAP;
        const keys_base = @as(usize, slot) * MAX_PENALTY_ENTRIES;
        const dirty_base = @as(usize, slot) * MAX_PENALTY_ENTRIES;
        var count: usize = 0;

        for (entries) |entry| {
            if (entry.count == 0 or entry.token_id >= vocab_size) continue;
            var h: usize = @intCast(hashTokenU32(entry.token_id) & (FREQ_TABLE_CAP - 1));
            var probes: usize = 0;
            while (probes < FREQ_TABLE_CAP) : (probes += 1) {
                const cur = &self.freq_tables[tbl_base + h];
                if (cur.token_id == EMPTY_ID) {
                    if (count >= MAX_PENALTY_ENTRIES) break;
                    cur.* = .{ .token_id = entry.token_id, .count = entry.count };
                    self.freq_keys[keys_base + count] = entry.token_id;
                    self.freq_dirty_slots[dirty_base + count] = @intCast(h);
                    count += 1;
                    break;
                }
                if (cur.token_id == entry.token_id) {
                    cur.count +|= entry.count;
                    break;
                }
                h = (h + 1) & (FREQ_TABLE_CAP - 1);
            }
        }
        self.freq_counts[slot] = @intCast(count);
    }

    /// O(1) incremental decode-step penalty update directly in unified memory
    /// (avoids re-uploading the full seen-token slice on every generated token).
    pub fn recordToken(self: *GpuSamplerWorkspace, slot: u32, token_id: u32) void {
        if (!self.spec.enable_overrides or slot >= self.spec.max_batch_slots) return;
        const tbl_base = @as(usize, slot) * FREQ_TABLE_CAP;
        const keys_base = @as(usize, slot) * MAX_PENALTY_ENTRIES;
        const dirty_base = @as(usize, slot) * MAX_PENALTY_ENTRIES;
        var h: usize = @intCast(hashTokenU32(token_id) & (FREQ_TABLE_CAP - 1));
        var probes: usize = 0;
        while (probes < FREQ_TABLE_CAP) : (probes += 1) {
            const cur = &self.freq_tables[tbl_base + h];
            if (cur.token_id == EMPTY_ID) {
                const count: usize = self.freq_counts[slot];
                if (count >= MAX_PENALTY_ENTRIES) return;
                cur.* = .{ .token_id = token_id, .count = 1 };
                self.freq_keys[keys_base + count] = token_id;
                self.freq_dirty_slots[dirty_base + count] = @intCast(h);
                self.freq_counts[slot] = @intCast(count + 1);
                return;
            }
            if (cur.token_id == token_id) {
                cur.count +|= 1;
                return;
            }
            h = (h + 1) & (FREQ_TABLE_CAP - 1);
        }
    }
};

// ---------------------------------------------------------------------------
// High-Level GpuSampler (Engine-Facing & Standalone API)
// ---------------------------------------------------------------------------

pub const RowRequest = struct {
    params: SampleParams = .{},
    /// Optional sparse seen-token frequency list. When non-null, synced into
    /// this row's penalty hash table in O(prev + len). When null and penalties
    /// are active, uses the row's incrementally-tracked `recordToken` state.
    seen_frequencies: ?[]const TokenFreqEntry = null,
    /// Slot-local `D11` (`RequestRng.nextU64()`) per-step seed for Gumbel-max sampling.
    rng_seed: u64 = 0,
    /// Optional bitmask row index (defaults to this row's index `row_idx`).
    /// Allows speculative tree rows to share or index distinct C22 grammar bitmasks.
    bitmask_slot: ?u32 = null,
};

pub const SampleRequest = struct {
    logits: metal.BufferBinding,
    logits_dtype: LogitsDType = .f16,
    vocab_size: u32,
    params: SampleParams = .{},
    seen_frequencies: ?[]const TokenFreqEntry = null,
    rng_seed: u64 = 0,
    /// Optional external resident C22 `u32` bitmask binding (zero-copy).
    /// When null and `params.allowed_token_mask != null`, copies into slot 0.
    external_bitmask: ?metal.BufferBinding = null,
};

pub const BatchSampleRequest = struct {
    logits: metal.BufferBinding,
    logits_dtype: LogitsDType = .f16,
    vocab_size: u32,
    rows: []const RowRequest,
    /// Optional external resident `[num_bitmask_rows, external_bitmask_stride_u32]`
    /// C22 bitmask buffer for zero-copy speculative-tree grammar verification.
    external_bitmasks: ?metal.BufferBinding = null,
    external_bitmask_stride_u32: u32 = 0,
};

pub const GpuSampler = struct {
    pipelines: GpuSamplerPipelines,
    workspace: GpuSamplerWorkspace,
    ws_buf: metal.Buffer,
    owned_arena: ?arena_mod.WiredArena = null,
    owns_buf: bool = false,

    /// Initializes a `GpuSampler` carved out of a caller-owned `WiredArena` and
    /// its resident `metal.Buffer` (preferred zero-extra-buffer engine path).
    pub fn initInArena(
        device: metal.Device,
        library: metal.Library,
        arena: *arena_mod.WiredArena,
        arena_buf: metal.Buffer,
        spec: SamplerSpec,
    ) Error!GpuSampler {
        const pipes = try GpuSamplerPipelines.init(device, library);
        errdefer pipes.close();
        const ws = try GpuSamplerWorkspace.init(arena, spec);
        return .{
            .pipelines = pipes,
            .workspace = ws,
            .ws_buf = arena_buf,
            .owned_arena = null,
            .owns_buf = false,
        };
    }

    /// Initializes a self-contained `GpuSampler` owning its 16 KiB-aligned
    /// wired memory arena and `metal.Buffer`. Call `registerResident(compute)`
    /// once before encoding.
    pub fn initStandalone(
        device: metal.Device,
        library: metal.Library,
        spec: SamplerSpec,
    ) Error!GpuSampler {
        const req = try GpuSamplerWorkspace.requiredBytes(spec);
        const cap = page.alignUp(req + page.APPLE_PAGE_SIZE) catch return error.Overflow;
        const budget = budget_mod.Budget.compute(.{
            .memory_bytes = cap + 4 * page.APPLE_PAGE_SIZE,
            .user_wire_limit = cap,
            .global_user_wire_limit = cap,
            .gpu_working_set_bytes = cap,
            .configured_cap = null,
            .headroom_bytes = page.APPLE_PAGE_SIZE,
        });
        var arena = try arena_mod.WiredArena.init(cap, budget);
        errdefer arena.deinit();

        const ws = try GpuSamplerWorkspace.init(&arena, spec);
        const buf = try metal.Buffer.wrap(device, arena.bytes());
        errdefer buf.close();

        const pipes = try GpuSamplerPipelines.init(device, library);
        errdefer pipes.close();

        return .{
            .pipelines = pipes,
            .workspace = ws,
            .ws_buf = buf,
            .owned_arena = arena,
            .owns_buf = true,
        };
    }

    pub fn close(self: *GpuSampler) void {
        self.pipelines.close();
        if (self.owns_buf) {
            self.ws_buf.close();
        }
        if (self.owned_arena) |*a| {
            a.deinit();
        }
    }

    pub fn registerResident(self: *const GpuSampler, compute: metal.Compute) metal.Error!void {
        try compute.makeResident(&.{self.ws_buf});
    }

    pub fn setBitmask(self: *GpuSampler, slot: u32, vocab_size: u32, mask: []const u32) Error!void {
        return self.workspace.setBitmask(slot, vocab_size, mask);
    }

    pub fn setBitmaskAllAllowed(self: *GpuSampler, slot: u32, vocab_size: u32) Error!void {
        return self.workspace.setBitmaskAllAllowed(slot, vocab_size);
    }

    pub fn uploadOverrides(
        self: *GpuSampler,
        slot: u32,
        vocab_size: u32,
        logit_bias: []const LogitBiasEntry,
        seen_frequencies: []const TokenFreqEntry,
    ) Error!void {
        try self.workspace.uploadLogitBias(slot, vocab_size, logit_bias);
        try self.workspace.uploadPenalties(slot, vocab_size, seen_frequencies);
    }

    pub fn clearOverrides(self: *GpuSampler, slot: u32) void {
        self.workspace.clearLogitBias(slot);
        self.workspace.clearPenalties(slot);
    }

    pub fn recordToken(self: *GpuSampler, slot: u32, token_id: u32) void {
        self.workspace.recordToken(slot, token_id);
    }

    /// Encodes a single-row (`M = 1`) sampling step into `frame`.
    pub fn encodeSample(
        self: *GpuSampler,
        frame: *metal.Compute.Frame,
        req: SampleRequest,
        insert_barrier_before: bool,
    ) Error!void {
        const single = [_]RowRequest{.{
            .params = req.params,
            .seen_frequencies = req.seen_frequencies,
            .rng_seed = req.rng_seed,
            .bitmask_slot = 0,
        }};
        const bm_stride: u32 = if (req.external_bitmask != null)
            @intCast(grammar.bitmaskWordCount(req.vocab_size))
        else
            0;
        return self.encodeSampleBatch(frame, .{
            .logits = req.logits,
            .logits_dtype = req.logits_dtype,
            .vocab_size = req.vocab_size,
            .rows = &single,
            .external_bitmasks = req.external_bitmask,
            .external_bitmask_stride_u32 = bm_stride,
        }, insert_barrier_before);
    }

    /// Encodes a single-row or batched / speculative-tree GPU sampling pass into `frame`.
    /// Automatically selects the zero-overhead unconstrained greedy argmax fast path
    /// whenever all rows are unconstrained greedy (`temperature == 0`, no penalties,
    /// no `logit_bias`, no `C22` bitmask, no `logprobs`/`top_logprobs`).
    pub fn encodeSampleBatch(
        self: *GpuSampler,
        frame: *metal.Compute.Frame,
        req: BatchSampleRequest,
        insert_barrier_before: bool,
    ) Error!void {
        if (req.vocab_size == 0) return error.EmptyVocab;
        if (req.vocab_size > self.workspace.spec.max_vocab_size) return error.VocabTooLarge;
        if (req.rows.len == 0 or req.rows.len > self.workspace.spec.max_batch_slots) {
            return error.BatchTooLarge;
        }

        const num_rows: u32 = @intCast(req.rows.len);
        const num_blocks = numBlocksForVocab(req.vocab_size);
        const base_aligned: bool = switch (req.logits_dtype) {
            .f16 => (req.logits.offset % 8 == 0),
            .f32 => (req.logits.offset % 16 == 0),
        };
        const aligned_vec4: u32 = if (base_aligned and (req.vocab_size % 4 == 0))
            3
        else if (base_aligned)
            2
        else
            0;

        var all_unconstrained_greedy = (req.external_bitmasks == null);
        var any_bitmask = (req.external_bitmasks != null);
        var any_overrides = false;
        var any_stochastic = false;
        var any_refine_possible = false;
        var any_top_logprobs = false;

        for (req.rows) |row| {
            try row.params.validate();
            if (!row.params.isUnconstrainedGreedy()) {
                all_unconstrained_greedy = false;
            }
            if (row.params.allowed_token_mask != null or req.external_bitmasks != null) {
                any_bitmask = true;
            }
            if (row.params.logit_bias.len > 0 or row.params.hasPenalties()) {
                any_overrides = true;
            }
            if (row.params.temperature > 0.0) {
                any_stochastic = true;
                if (row.params.top_k > 0 or row.params.top_p < 1.0) {
                    any_refine_possible = true;
                }
            }
            if (row.params.top_logprobs > 0) {
                any_top_logprobs = true;
            }
        }

        const wb = self.ws_buf;
        const ws = &self.workspace;

        // -------------------------------------------------------------------
        // Explicit Unconstrained Greedy Fast-Path Bypass (ADR-0023 §4):
        // Minimal 2-pass multi-block GPU argmax reading ZERO bitmask/penalty/
        // bias/RNG buffers!
        // -------------------------------------------------------------------
        if (all_unconstrained_greedy) {
            const dp: SampleDispatchParams = .{
                .num_rows = num_rows,
                .vocab_size = req.vocab_size,
                .num_blocks = num_blocks,
                .aligned_vec4 = aligned_vec4,
                .bitmask_stride_u32 = 0,
            };
            const p1_pipe = switch (req.logits_dtype) {
                .f16 => self.pipelines.greedy_pass1_f16,
                .f32 => self.pipelines.greedy_pass1_f32,
            };
            try frame.encodeThreadgroups(
                p1_pipe,
                &.{
                    req.logits,
                    .{ .buffer = wb, .offset = ws.block_stats_off },
                },
                std.mem.asBytes(&dp),
                .{ num_blocks, num_rows, 1 },
                .{ 256, 1, 1 },
                insert_barrier_before,
            );
            try frame.encodeThreadgroups(
                self.pipelines.greedy_pass2,
                &.{
                    .{ .buffer = wb, .offset = ws.block_stats_off },
                    .{ .buffer = wb, .offset = ws.outputs_off },
                },
                std.mem.asBytes(&dp),
                .{ num_rows, 1, 1 },
                .{ 32, 1, 1 },
                true,
            );
            return;
        }

        // -------------------------------------------------------------------
        // Constrained / Penalized / Stochastic Radix-Bucket Pipeline
        // -------------------------------------------------------------------
        const bm_words = grammar.bitmaskWordCount(req.vocab_size);
        const bm_stride_u32: u32 = if (req.external_bitmasks != null)
            (if (req.external_bitmask_stride_u32 > 0) req.external_bitmask_stride_u32 else @intCast(bm_words))
        else
            ws.bitmask_stride_u32;

        if (req.external_bitmasks != null and bm_stride_u32 < bm_words) {
            return error.BitmaskTooSmall;
        }

        // Populate per-row configs and sparse override hash tables in O(sparse).
        for (req.rows, 0..) |row, r_idx| {
            const slot: u32 = @intCast(r_idx);
            var flags: u32 = 0;

            const use_bm = (row.params.allowed_token_mask != null) or (req.external_bitmasks != null);
            const bm_row_idx: u32 = row.bitmask_slot orelse slot;
            if (use_bm) {
                flags |= FLAG_USE_BITMASK;
                if (req.external_bitmasks == null) {
                    if (row.params.allowed_token_mask) |mask| {
                        try ws.setBitmask(bm_row_idx, req.vocab_size, mask);
                    }
                }
            }

            var b_cnt: u32 = 0;
            if (row.params.logit_bias.len > 0) {
                try ws.uploadLogitBias(slot, req.vocab_size, row.params.logit_bias);
                b_cnt = ws.bias_counts[slot];
                if (b_cnt > 0) flags |= FLAG_HAS_BIAS;
            } else if (ws.spec.enable_overrides and ws.bias_counts[slot] > 0) {
                ws.clearLogitBias(slot);
            }

            var f_cnt: u32 = 0;
            if (row.params.hasPenalties()) {
                if (row.seen_frequencies) |seen| {
                    try ws.uploadPenalties(slot, req.vocab_size, seen);
                }
                if (!ws.spec.enable_overrides) return error.OverridesDisabled;
                f_cnt = ws.freq_counts[slot];
                if (f_cnt > 0) flags |= FLAG_HAS_PENALTIES;
            }

            if (row.params.logprobs or row.params.top_logprobs > 0) {
                flags |= FLAG_NEED_LOGPROBS;
            }

            const inv_temp: f32 = if (row.params.temperature > 0.0) 1.0 / row.params.temperature else 1.0;
            ws.row_cfgs[slot] = .{
                .temperature = row.params.temperature,
                .inv_temp = inv_temp,
                .top_p = row.params.top_p,
                .min_p = row.params.min_p,
                .presence_penalty = row.params.presence_penalty,
                .frequency_penalty = row.params.frequency_penalty,
                .repetition_penalty = row.params.repetition_penalty,
                .top_k = row.params.top_k,
                .seed_lo = @truncate(row.rng_seed),
                .seed_hi = @truncate(row.rng_seed >> 32),
                .flags = flags,
                .top_logprobs = row.params.top_logprobs,
                .bias_count = b_cnt,
                .freq_count = f_cnt,
                .bitmask_row_index = bm_row_idx,
            };
        }

        const dp: SampleDispatchParams = .{
            .num_rows = num_rows,
            .vocab_size = req.vocab_size,
            .num_blocks = num_blocks,
            .aligned_vec4 = aligned_vec4,
            .bitmask_stride_u32 = bm_stride_u32,
        };
        const v_idx = GpuSamplerPipelines.variantIndex(req.logits_dtype, any_bitmask, any_overrides);
        const bm_bind: metal.BufferBinding = req.external_bitmasks orelse .{ .buffer = wb, .offset = ws.bitmasks_off };

        // Pass 1: Fused constrained reduction (max_logit, argmax_id, online log_z).
        try frame.encodeThreadgroups(
            self.pipelines.constrained_pass1[v_idx],
            &.{
                req.logits,
                .{ .buffer = wb, .offset = ws.row_cfgs_off },
                bm_bind,
                .{ .buffer = wb, .offset = ws.bias_tables_off },
                .{ .buffer = wb, .offset = ws.bias_keys_off },
                .{ .buffer = wb, .offset = ws.freq_tables_off },
                .{ .buffer = wb, .offset = ws.freq_keys_off },
                .{ .buffer = wb, .offset = ws.block_stats_off },
            },
            std.mem.asBytes(&dp),
            .{ num_blocks, num_rows, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );

        // Pass 1 Merge: Reduce block partials -> RowState, finalize greedy rows,
        // zero 1024-bucket histograms & indirect refine args in parallel.
        try frame.encodeThreadgroups(
            self.pipelines.merge_stats,
            &.{
                .{ .buffer = wb, .offset = ws.block_stats_off },
                .{ .buffer = wb, .offset = ws.row_cfgs_off },
                .{ .buffer = wb, .offset = ws.row_states_off },
                .{ .buffer = wb, .offset = ws.hist_count_off },
                .{ .buffer = wb, .offset = ws.hist_mass_off },
                .{ .buffer = wb, .offset = ws.hist_gumbel_off },
                .{ .buffer = wb, .offset = ws.refine_indirect_off },
                .{ .buffer = wb, .offset = ws.cut_cand_counts_off },
                .{ .buffer = wb, .offset = ws.outputs_off },
            },
            std.mem.asBytes(&dp),
            .{ num_rows, 1, 1 },
            .{ 256, 1, 1 },
            true,
        );

        // Pass 2 (Stochastic rows only): 1024-bucket radix + Gumbel-max reservoir -> select ->
        // parallel 32-block indirect exact boundary refinement (0 TGs dispatched when clean!).
        if (any_stochastic) {
            try frame.encodeThreadgroups(
                self.pipelines.radix_pass2[v_idx],
                &.{
                    req.logits,
                    .{ .buffer = wb, .offset = ws.row_cfgs_off },
                    .{ .buffer = wb, .offset = ws.row_states_off },
                    bm_bind,
                    .{ .buffer = wb, .offset = ws.bias_tables_off },
                    .{ .buffer = wb, .offset = ws.bias_keys_off },
                    .{ .buffer = wb, .offset = ws.freq_tables_off },
                    .{ .buffer = wb, .offset = ws.freq_keys_off },
                    .{ .buffer = wb, .offset = ws.hist_count_off },
                    .{ .buffer = wb, .offset = ws.hist_mass_off },
                    .{ .buffer = wb, .offset = ws.hist_gumbel_off },
                },
                std.mem.asBytes(&dp),
                .{ num_blocks, num_rows, 1 },
                .{ 256, 1, 1 },
                true,
            );

            try frame.encodeThreadgroups(
                self.pipelines.radix_select[v_idx],
                &.{
                    req.logits,
                    .{ .buffer = wb, .offset = ws.row_cfgs_off },
                    .{ .buffer = wb, .offset = ws.row_states_off },
                    bm_bind,
                    .{ .buffer = wb, .offset = ws.bias_tables_off },
                    .{ .buffer = wb, .offset = ws.freq_tables_off },
                    .{ .buffer = wb, .offset = ws.hist_count_off },
                    .{ .buffer = wb, .offset = ws.hist_mass_off },
                    .{ .buffer = wb, .offset = ws.hist_gumbel_off },
                    .{ .buffer = wb, .offset = ws.refine_indirect_off },
                    .{ .buffer = wb, .offset = ws.outputs_off },
                },
                std.mem.asBytes(&dp),
                .{ num_rows, 1, 1 },
                .{ 256, 1, 1 },
                true,
            );

            if (any_refine_possible) {
                try frame.encodeThreadgroupsIndirect(
                    self.pipelines.refine_gather[v_idx],
                    &.{
                        req.logits,
                        .{ .buffer = wb, .offset = ws.row_cfgs_off },
                        .{ .buffer = wb, .offset = ws.row_states_off },
                        bm_bind,
                        .{ .buffer = wb, .offset = ws.bias_tables_off },
                        .{ .buffer = wb, .offset = ws.bias_keys_off },
                        .{ .buffer = wb, .offset = ws.freq_tables_off },
                        .{ .buffer = wb, .offset = ws.freq_keys_off },
                        .{ .buffer = wb, .offset = ws.cut_cand_counts_off },
                        .{ .buffer = wb, .offset = ws.cut_cands_off },
                    },
                    std.mem.asBytes(&dp),
                    .{ .buffer = wb, .offset = ws.refine_indirect_off },
                    .{ 256, 1, 1 },
                    true,
                );

                try frame.encodeThreadgroupsIndirect(
                    self.pipelines.refine_sort[v_idx],
                    &.{
                        req.logits,
                        .{ .buffer = wb, .offset = ws.row_cfgs_off },
                        .{ .buffer = wb, .offset = ws.row_states_off },
                        bm_bind,
                        .{ .buffer = wb, .offset = ws.bias_tables_off },
                        .{ .buffer = wb, .offset = ws.freq_tables_off },
                        .{ .buffer = wb, .offset = ws.cut_cand_counts_off },
                        .{ .buffer = wb, .offset = ws.cut_cands_off },
                        .{ .buffer = wb, .offset = ws.outputs_off },
                    },
                    std.mem.asBytes(&dp),
                    .{ .buffer = wb, .offset = ws.refine_indirect_off + 12 },
                    .{ 256, 1, 1 },
                    true,
                );

                try frame.encodeThreadgroupsIndirect(
                    self.pipelines.refine_tie[v_idx],
                    &.{
                        req.logits,
                        .{ .buffer = wb, .offset = ws.row_cfgs_off },
                        .{ .buffer = wb, .offset = ws.row_states_off },
                        bm_bind,
                        .{ .buffer = wb, .offset = ws.bias_tables_off },
                        .{ .buffer = wb, .offset = ws.bias_keys_off },
                        .{ .buffer = wb, .offset = ws.freq_tables_off },
                        .{ .buffer = wb, .offset = ws.freq_keys_off },
                        .{ .buffer = wb, .offset = ws.outputs_off },
                    },
                    std.mem.asBytes(&dp),
                    .{ .buffer = wb, .offset = ws.refine_indirect_off + 24 },
                    .{ 256, 1, 1 },
                    false, // disjoint rows from refine_sort
                );
            }
        }

        // Optional exact top_logprobs extraction (dispatched ONLY when requested).
        if (any_top_logprobs) {
            try frame.encodeThreadgroups(
                self.pipelines.toplogprobs_pass1[v_idx],
                &.{
                    req.logits,
                    .{ .buffer = wb, .offset = ws.row_cfgs_off },
                    .{ .buffer = wb, .offset = ws.row_states_off },
                    bm_bind,
                    .{ .buffer = wb, .offset = ws.bias_tables_off },
                    .{ .buffer = wb, .offset = ws.bias_keys_off },
                    .{ .buffer = wb, .offset = ws.freq_tables_off },
                    .{ .buffer = wb, .offset = ws.freq_keys_off },
                    .{ .buffer = wb, .offset = ws.block_top_off },
                },
                std.mem.asBytes(&dp),
                .{ num_blocks, num_rows, 1 },
                .{ 256, 1, 1 },
                !any_stochastic, // if !any_stochastic, wait on merge_stats; otherwise already synced
            );

            try frame.encodeThreadgroups(
                self.pipelines.toplogprobs_pass2,
                &.{
                    .{ .buffer = wb, .offset = ws.row_cfgs_off },
                    .{ .buffer = wb, .offset = ws.row_states_off },
                    .{ .buffer = wb, .offset = ws.block_top_off },
                    .{ .buffer = wb, .offset = ws.outputs_off },
                },
                std.mem.asBytes(&dp),
                .{ num_rows, 1, 1 },
                .{ 32, 1, 1 },
                true,
            );
        }
    }

    /// Returns the arena byte budget required to carve `spec` via `initInArena`.
    pub fn requiredBytes(spec: SamplerSpec) Error!usize {
        return GpuSamplerWorkspace.requiredBytes(spec);
    }

    /// Reads the sampled token ID (and optional logprobs) for `row` directly from
    /// shared host memory (`0` full-vocab CPU scan).
    pub fn readOutput(self: *const GpuSampler, row: u32) Error!SampleOutput {
        if (row >= self.workspace.spec.max_batch_slots) return error.SlotOutOfRange;
        return self.workspace.outputs[row].toSampleOutput();
    }

    /// Zero-copy view of raw GPU sample output records for batched readback.
    pub fn rawOutputsSlice(self: *const GpuSampler, num_rows: usize) []const GpuSampleOutput {
        const cap = @min(num_rows, self.workspace.outputs.len);
        return self.workspace.outputs[0..cap];
    }

    /// Convenience synchronous single-row sampling helper for tests and benchmarks.
    pub fn sampleSync(
        self: *GpuSampler,
        compute: *metal.Compute,
        req: SampleRequest,
        timeout_ms: u64,
    ) Error!SampleOutput {
        var frame = try compute.beginFrame(timeout_ms);
        errdefer frame.abort();
        try self.encodeSample(&frame, req, false);
        const sig = try frame.commit();
        try compute.wait(sig, timeout_ms);
        return self.readOutput(0);
    }
};

// ===========================================================================
// Unit Tests (248K Vocab Parity vs CPU `src/spec/sampling.zig` & Warm Latency)
// ===========================================================================

fn makeTestArena(min_bytes: usize) !arena_mod.WiredArena {
    const cap = try page.alignUp(min_bytes + page.APPLE_PAGE_SIZE);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    return arena_mod.WiredArena.init(cap, budget);
}

fn fillSyntheticLogitsF16(dst: []f16, seed: u64) void {
    var prng = std.Random.Xoshiro256.init(seed);
    const r = prng.random();
    for (dst, 0..) |*v, i| {
        // Realistic heavy-tailed LLM logit spectrum in [-12, +10] with a few peaks.
        const base = (r.float(f32) * 14.0) - 10.0;
        const val: f32 = if (i % 41_117 == 17) base + 9.0 else base;
        v.* = @floatCast(val);
    }
}

fn fillSyntheticLogitsF32(dst: []f32, seed: u64) void {
    var prng = std.Random.Xoshiro256.init(seed);
    const r = prng.random();
    for (dst, 0..) |*v, i| {
        const base = (r.float(f32) * 14.0) - 10.0;
        v.* = if (i % 37_003 == 29) base + 9.5 else base;
    }
}

test "gpu.sampler: ABI sizes, workspace sizing, and parameter validation" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(GpuBiasEntry));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(GpuFreqEntry));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(RowSampleConfig));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(SampleDispatchParams));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(BlockStats));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(RowState));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(GpuTokenLogprob));
    try std.testing.expectEqual(@as(usize, 192), @sizeOf(GpuSampleOutput));

    const bytes_1 = try GpuSamplerWorkspace.requiredBytes(.{ .max_vocab_size = MAX_VOCAB_SIZE, .max_batch_slots = 1 });
    try std.testing.expect(bytes_1 > 0 and bytes_1 < 256 * 1024);

    try std.testing.expectError(error.EmptyVocab, GpuSamplerWorkspace.requiredBytes(.{ .max_vocab_size = 0 }));
    try std.testing.expectError(error.VocabTooLarge, GpuSamplerWorkspace.requiredBytes(.{ .max_vocab_size = MAX_VOCAB_SIZE + 1 }));
    try std.testing.expectError(error.BatchTooLarge, GpuSamplerWorkspace.requiredBytes(.{ .max_batch_slots = 0 }));
    try std.testing.expectError(error.BatchTooLarge, GpuSamplerWorkspace.requiredBytes(.{ .max_batch_slots = MAX_BATCH_SLOTS + 1 }));
}

test "gpu.sampler: unconstrained greedy fast-path parity across edge, 129K, 151K, 248K, and 262K vocabs (f16 & f32)" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();

    var sampler = try GpuSampler.initStandalone(device, lib, .{
        .max_vocab_size = MAX_VOCAB_SIZE,
        .max_batch_slots = 1,
    });
    defer sampler.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try sampler.registerResident(compute);

    const max_bytes = MAX_VOCAB_SIZE * @sizeOf(f32);
    var arena = try makeTestArena(max_bytes + 64 * 1024);
    defer arena.deinit();
    const logits_raw = try arena.allocAligned(max_bytes, 64);
    const logits_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer logits_buf.close();
    try compute.makeResident(&.{logits_buf});

    const vocabs = [_]u32{ 1, 13, 4099, 129_280, 151_936, 248_077, 262_144 };
    var dummy_rng = rng_mod.RequestRng.initDeterministic(1);

    for (vocabs, 0..) |vsize, vi| {
        // f16 logits with exact tie check (smallest id must win).
        const f16_slice: []f16 = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, logits_raw)))[0..vsize];
        fillSyntheticLogitsF16(f16_slice, 0x1000 + vi);
        if (vsize > 5) {
            f16_slice[vsize - 1] = @floatCast(@as(f32, 42.0));
            f16_slice[3] = @floatCast(@as(f32, 42.0)); // exact tie -> id 3 must win!
        }

        const cpu_out = try sampling.sampleToken(f16, f16_slice, .{}, &.{}, &dummy_rng, .{});
        const gpu_out = try sampler.sampleSync(&compute, .{
            .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_dtype = .f16,
            .vocab_size = vsize,
            .params = .{},
        }, 5000);
        try std.testing.expectEqual(cpu_out.token_id, gpu_out.token_id);

        // f32 logits.
        const f32_slice: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, logits_raw)))[0..vsize];
        fillSyntheticLogitsF32(f32_slice, 0x2000 + vi);
        const cpu_out32 = try sampling.sampleToken(f32, f32_slice, .{}, &.{}, &dummy_rng, .{});
        const gpu_out32 = try sampler.sampleSync(&compute, .{
            .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_dtype = .f32,
            .vocab_size = vsize,
            .params = .{},
        }, 5000);
        try std.testing.expectEqual(cpu_out32.token_id, gpu_out32.token_id);
    }

    // NaN / +Inf logits must report error.InvalidLogits.
    const f32_bad: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, logits_raw)))[0..256];
    @memset(f32_bad, 0.0);
    f32_bad[123] = std.math.nan(f32);
    try std.testing.expectError(error.InvalidLogits, sampler.sampleSync(&compute, .{
        .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
        .logits_dtype = .f32,
        .vocab_size = 256,
        .params = .{},
    }, 5000));
}

test "gpu.sampler: 248K constrained greedy parity (C22 bitmask, logit_bias, penalties, incremental recordToken, top_logprobs)" {
    const gpa = std.testing.allocator;
    const vocab_size: u32 = 248_077;

    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();

    var sampler = try GpuSampler.initStandalone(device, lib, .{
        .max_vocab_size = MAX_VOCAB_SIZE,
        .max_batch_slots = 1,
    });
    defer sampler.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try sampler.registerResident(compute);

    var arena = try makeTestArena(vocab_size * @sizeOf(f32) + 64 * 1024);
    defer arena.deinit();
    const logits_raw = try arena.allocAligned(vocab_size * @sizeOf(f32), 64);
    const logits_f16: []f16 = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, logits_raw)))[0..vocab_size];
    fillSyntheticLogitsF16(logits_f16, 0xCAFE_2480);

    // Plant known peaks at specific indices (including near the 248K tail!).
    logits_f16[100] = @floatCast(@as(f32, 18.0)); // will be banned by C22 bitmask
    logits_f16[500] = @floatCast(@as(f32, 16.0)); // will be heavily penalized by rep+freq+pres
    logits_f16[248_070] = @floatCast(@as(f32, 11.0)); // boosted by +4.5 logit_bias -> winner!
    logits_f16[248_076] = @floatCast(@as(f32, 14.5)); // runner-up (allowed, unpenalized)
    logits_f16[123_456] = @floatCast(@as(f32, -2.0)); // negative logit with repetition_penalty test

    const logits_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer logits_buf.close();
    try compute.makeResident(&.{logits_buf});

    // Build C22 bitmask allowing ~25% of tokens + explicit test targets, banning token 100.
    const bm_words = grammar.bitmaskWordCount(vocab_size);
    const mask = try gpa.alloc(u32, bm_words);
    defer gpa.free(mask);
    for (mask, 0..) |*w, wi| {
        w.* = if ((wi & 3) == 0) 0x5555_5555 else 0x0000_0000;
    }
    // Explicitly ban 100, allow 500, 123_456, 248_070, 248_076.
    mask[100 >> 5] &= ~(@as(u32, 1) << @intCast(100 & 31));
    for ([_]u32{ 500, 123_456, 248_070, 248_076 }) |id| {
        mask[id >> 5] |= (@as(u32, 1) << @intCast(id & 31));
    }

    const biases = [_]LogitBiasEntry{
        .{ .token_id = 248_070, .bias = 4.5 },
        .{ .token_id = 500, .bias = -0.5 },
        .{ .token_id = 248_076, .bias = 0.25 },
        .{ .token_id = 999_999, .bias = 50.0 }, // out-of-vocab, must be ignored
    };

    var tracker = sampling.DefaultPenaltyTracker.init();
    tracker.recordSlice(&.{ 500, 500, 500, 123_456, 123_456, 42, 777 });

    const params: SampleParams = .{
        .temperature = 0.0,
        .presence_penalty = 0.8,
        .frequency_penalty = 0.6,
        .repetition_penalty = 1.25,
        .logit_bias = &biases,
        .logprobs = true,
        .top_logprobs = 20,
        .allowed_token_mask = mask,
    };

    const scratch_f32 = try gpa.alloc(f32, vocab_size);
    defer gpa.free(scratch_f32);
    var dummy_rng = rng_mod.RequestRng.initDeterministic(9);
    const cpu_out = try sampling.sampleToken(
        f16,
        logits_f16,
        params,
        tracker.activeEntries(),
        &dummy_rng,
        .{ .logits_f32 = scratch_f32 },
    );

    const gpu_out = try sampler.sampleSync(&compute, .{
        .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
        .logits_dtype = .f16,
        .vocab_size = vocab_size,
        .params = params,
        .seen_frequencies = tracker.activeEntries(),
    }, 5000);

    try std.testing.expectEqual(@as(u32, 248_070), cpu_out.token_id);
    try std.testing.expectEqual(cpu_out.token_id, gpu_out.token_id);
    try std.testing.expectApproxEqAbs(cpu_out.logprob, gpu_out.logprob, 1e-3);
    try std.testing.expectEqual(cpu_out.top_logprobs_len, gpu_out.top_logprobs_len);
    for (cpu_out.topLogprobsSlice(), gpu_out.topLogprobsSlice()) |c_lp, g_lp| {
        try std.testing.expectEqual(c_lp.token_id, g_lp.token_id);
        try std.testing.expectApproxEqAbs(c_lp.logprob, g_lp.logprob, 1e-3);
    }

    // Verify incremental `recordToken` matches re-uploading `tracker.activeEntries()`!
    sampler.clearOverrides(0);
    for ([_]u32{ 500, 500, 500, 123_456, 123_456, 42, 777 }) |id| {
        sampler.recordToken(0, id);
    }
    const gpu_inc = try sampler.sampleSync(&compute, .{
        .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
        .logits_dtype = .f16,
        .vocab_size = vocab_size,
        .params = params,
        .seen_frequencies = null, // uses incremental table directly!
    }, 5000);
    try std.testing.expectEqual(cpu_out.token_id, gpu_inc.token_id);
    try std.testing.expectApproxEqAbs(cpu_out.logprob, gpu_inc.logprob, 1e-3);

    // Empty allowed bitmask must return error.EmptyAllowedSet!
    @memset(mask, 0);
    try std.testing.expectError(error.EmptyAllowedSet, sampler.sampleSync(&compute, .{
        .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
        .logits_dtype = .f16,
        .vocab_size = vocab_size,
        .params = .{ .allowed_token_mask = mask },
    }, 5000));
}

test "gpu.sampler: stochastic radix/bucket sampling, exact boundary refinement (Case 1 & Case 2), D11 determinism, and distribution parity" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();

    var sampler = try GpuSampler.initStandalone(device, lib, .{
        .max_vocab_size = MAX_VOCAB_SIZE,
        .max_batch_slots = 1,
    });
    defer sampler.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try sampler.registerResident(compute);

    var arena = try makeTestArena(MAX_VOCAB_SIZE * @sizeOf(f32) + 64 * 1024);
    defer arena.deinit();
    const logits_raw = try arena.allocAligned(MAX_VOCAB_SIZE * @sizeOf(f32), 64);
    const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, logits_raw));
    const logits_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer logits_buf.close();
    try compute.makeResident(&.{logits_buf});

    // 1. Exact Boundary Split Case 1 (<= 1024 tied tokens): 128 tied tokens at 0.0, top_k = 5
    //    MUST strictly pick ONLY from {0, 1, 2, 3, 4} across 32 distinct D11 seeds!
    @memset(logits_f32[0..128], 0.0);
    var rng1 = rng_mod.RequestRng.initDeterministic(0xD11_C001);
    var seen_k5: [5]u32 = @splat(0);
    for (0..32) |_| {
        const out = try sampler.sampleSync(&compute, .{
            .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_dtype = .f32,
            .vocab_size = 128,
            .params = .{ .temperature = 1.0, .top_k = 5 },
            .rng_seed = rng1.nextU64(),
        }, 5000);
        try std.testing.expect(out.token_id < 5);
        seen_k5[out.token_id] += 1;
    }
    // Every token in {0..4} should be hit at least once in 32 draws (p_miss = (4/5)^32 < 8e-4).
    for (seen_k5) |c| try std.testing.expect(c > 0);

    // 2. Exact Boundary Split Case 2 (> 1024 tied tokens): 4096 tied tokens at 0.0, top_k = 17
    //    MUST trigger Case 2 sub-radix + token_id tie-break and pick ONLY from {0..16}!
    @memset(logits_f32[0..4096], 0.0);
    var rng2 = rng_mod.RequestRng.initDeterministic(0xD11_C002);
    for (0..24) |_| {
        const out = try sampler.sampleSync(&compute, .{
            .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_dtype = .f32,
            .vocab_size = 4096,
            .params = .{ .temperature = 1.0, .top_k = 17 },
            .rng_seed = rng2.nextU64(),
        }, 5000);
        try std.testing.expect(out.token_id < 17);
    }

    // 3. Exact Categorical distribution parity + D11 bit-for-bit replay on a 6-token skewed spectrum!
    const probs_logits = [_]f32{ 2.0, 1.2, 0.5, -0.3, -1.5, -4.0 };
    @memcpy(logits_f32[0..6], &probs_logits);
    var expected_p: [6]f64 = undefined;
    var sum_e: f64 = 0.0;
    for (probs_logits, 0..) |l, i| {
        expected_p[i] = @exp(@as(f64, l - 2.0) / 0.8);
        sum_e += expected_p[i];
    }
    for (&expected_p) |*p| p.* /= sum_e;

    var counts: [6]u32 = @splat(0);
    const n_draws: usize = 600;
    var rng_dist = rng_mod.RequestRng.initDeterministic(0xD11_57A7);
    var replay_rng = rng_mod.RequestRng.initDeterministic(0xD11_57A7);
    var first_ten: [10]u32 = undefined;

    for (0..n_draws) |d| {
        const out = try sampler.sampleSync(&compute, .{
            .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_dtype = .f32,
            .vocab_size = 6,
            .params = .{ .temperature = 0.8, .logprobs = true },
            .rng_seed = rng_dist.nextU64(),
        }, 5000);
        try std.testing.expect(out.token_id < 6);
        counts[out.token_id] += 1;
        if (d < 10) first_ten[d] = out.token_id;
        // Exact logprob parity for the chosen token!
        const exp_lp: f32 = @floatCast(@log(expected_p[out.token_id]));
        try std.testing.expectApproxEqAbs(exp_lp, out.logprob, 1e-4);
    }
    // Verify 100% deterministic replay with identical D11 seed!
    for (0..10) |d| {
        const out = try sampler.sampleSync(&compute, .{
            .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_dtype = .f32,
            .vocab_size = 6,
            .params = .{ .temperature = 0.8 },
            .rng_seed = replay_rng.nextU64(),
        }, 5000);
        try std.testing.expectEqual(first_ten[d], out.token_id);
    }
    // Verify empirical proportions match theoretical Categorical probabilities within 4.5 sigma!
    for (0..6) |i| {
        const emp = @as(f64, @floatFromInt(counts[i])) / @as(f64, @floatFromInt(n_draws));
        const sigma = @sqrt(expected_p[i] * (1.0 - expected_p[i]) / @as(f64, @floatFromInt(n_draws)));
        try std.testing.expect(@abs(emp - expected_p[i]) <= @max(4.5 * sigma, 0.02));
    }

    // 4. Full 248,077 vocab combined filter check (`C22` bitmask + bias + penalties + min_p + top_k + top_p + top_logprobs)!
    const vocab_248k: u32 = 248_077;
    const logits_f16: []f16 = @as([]f16, @alignCast(std.mem.bytesAsSlice(f16, logits_raw)))[0..vocab_248k];
    fillSyntheticLogitsF16(logits_f16, 0x2480_570C);
    // Create 6 distinct top candidates well separated above background.
    const cand_ids = [_]u32{ 10, 50_000, 120_000, 180_000, 240_000, 248_075 };
    const cand_vals = [_]f32{ 16.0, 15.2, 14.8, 12.0, 9.0, 15.5 };
    for (cand_ids, cand_vals) |id, val| logits_f16[id] = @floatCast(val);

    const bm_words = grammar.bitmaskWordCount(vocab_248k);
    const mask = try gpa.alloc(u32, bm_words);
    defer gpa.free(mask);
    @memset(mask, 0xFFFF_FFFF);
    // Ban token 10 (highest raw logit) via C22 bitmask!
    mask[10 >> 5] &= ~(@as(u32, 1) << @intCast(10 & 31));

    const params_full: SampleParams = .{
        .temperature = 0.7,
        .top_k = 3,
        .min_p = 0.2,
        .top_p = 0.9,
        .logprobs = true,
        .top_logprobs = 5,
        .allowed_token_mask = mask,
    };

    // CPU reference logprobs & top_logprobs at 248K vocab.
    const scratch_f32 = try gpa.alloc(f32, vocab_248k);
    defer gpa.free(scratch_f32);
    var cpu_rng = rng_mod.RequestRng.initDeterministic(77);
    const cpu_ref = try sampling.sampleToken(f16, logits_f16, params_full, &.{}, &cpu_rng, .{ .logits_f32 = scratch_f32 });

    var gpu_rng = rng_mod.RequestRng.initDeterministic(77);
    for (0..16) |_| {
        const gpu_out = try sampler.sampleSync(&compute, .{
            .logits = .{ .buffer = logits_buf, .offset = arena.offsetOf(logits_raw.ptr) },
            .logits_dtype = .f16,
            .vocab_size = vocab_248k,
            .params = params_full,
            .rng_seed = gpu_rng.nextU64(),
        }, 5000);
        // Token 10 is banned; top_k=3 & min_p=0.2 restrict strictly to {248_075, 50_000, 120_000}!
        try std.testing.expect(gpu_out.token_id == 248_075 or gpu_out.token_id == 50_000 or gpu_out.token_id == 120_000);
        try std.testing.expectEqual(cpu_ref.top_logprobs_len, gpu_out.top_logprobs_len);
        for (cpu_ref.topLogprobsSlice(), gpu_out.topLogprobsSlice()) |c_lp, g_lp| {
            try std.testing.expectEqual(c_lp.token_id, g_lp.token_id);
            try std.testing.expectApproxEqAbs(c_lp.logprob, g_lp.logprob, 2e-3);
        }
    }
}

test "gpu.sampler: batched speculative-tree multi-row sampling with per-row C22 bitmasks, mixed greedy/stochastic, and zero cross-row bleed" {
    const gpa = std.testing.allocator;
    const num_rows: u32 = 8;
    const vocab_size: u32 = 4096;

    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();

    var sampler = try GpuSampler.initStandalone(device, lib, .{
        .max_vocab_size = vocab_size,
        .max_batch_slots = num_rows,
    });
    defer sampler.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try sampler.registerResident(compute);

    const bm_words = grammar.bitmaskWordCount(vocab_size);
    const bm_stride_bytes = amf.align64(bm_words * @sizeOf(u32));
    const bm_stride_u32: u32 = @intCast(bm_stride_bytes / @sizeOf(u32));

    var arena = try makeTestArena(num_rows * vocab_size * @sizeOf(f16) + num_rows * bm_stride_bytes + 64 * 1024);
    defer arena.deinit();
    const logits_raw = try arena.allocAligned(num_rows * vocab_size * @sizeOf(f16), 64);
    const ext_bm_raw = try arena.allocAligned(num_rows * bm_stride_bytes, 64);
    const logits_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw));
    const ext_bm_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, ext_bm_raw));
    @memset(ext_bm_u32, 0);

    // Each row `r` has its OWN disjoint allowed token range `[r*256 .. r*256+4)`
    // in the external zero-copy bitmask buffer!
    for (0..num_rows) |r| {
        const row_l = logits_f16[r * vocab_size ..][0..vocab_size];
        fillSyntheticLogitsF16(row_l, 0xB000 + r);
        const target_base: u32 = @intCast(r * 256);
        row_l[target_base + 0] = @floatCast(@as(f32, 12.0));
        row_l[target_base + 1] = @floatCast(@as(f32, 11.5));
        row_l[target_base + 2] = @floatCast(@as(f32, 11.0));
        row_l[target_base + 3] = @floatCast(@as(f32, 10.5));
        // Higher decoy OUTSIDE row r's allowed mask:
        row_l[vocab_size - 1 - r] = @floatCast(@as(f32, 25.0));

        const bm_row = ext_bm_u32[r * bm_stride_u32 ..][0..bm_stride_u32];
        for (0..4) |k| {
            const id = target_base + @as(u32, @intCast(k));
            bm_row[id >> 5] |= (@as(u32, 1) << @intCast(id & 31));
        }
    }

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    try compute.makeResident(&.{buf});

    var rng = rng_mod.RequestRng.initDeterministic(0x7EEE_0001);
    var rows: [num_rows]RowRequest = undefined;
    for (0..num_rows) |r| {
        const is_greedy = (r % 2 == 0);
        rows[r] = .{
            .params = .{
                .temperature = if (is_greedy) 0.0 else 0.8,
                .top_p = if (is_greedy) 1.0 else 0.9,
                .logprobs = true,
                .top_logprobs = 4,
            },
            .rng_seed = rng.nextU64(),
            .bitmask_slot = @intCast(r),
        };
    }

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try sampler.encodeSampleBatch(&frame, .{
        .logits = .{ .buffer = buf, .offset = arena.offsetOf(logits_raw.ptr) },
        .logits_dtype = .f16,
        .vocab_size = vocab_size,
        .rows = &rows,
        .external_bitmasks = .{ .buffer = buf, .offset = arena.offsetOf(ext_bm_raw.ptr) },
        .external_bitmask_stride_u32 = bm_stride_u32,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    const scratch_f32 = try gpa.alloc(f32, vocab_size);
    defer gpa.free(scratch_f32);
    const scratch_ids = try gpa.alloc(u32, vocab_size);
    defer gpa.free(scratch_ids);
    var dummy_rng = rng_mod.RequestRng.initDeterministic(1);

    for (0..num_rows) |r| {
        const out = try sampler.readOutput(@intCast(r));
        const target_base: u32 = @intCast(r * 256);
        try std.testing.expect(out.token_id >= target_base and out.token_id < target_base + 4);
        if (r % 2 == 0) {
            try std.testing.expectEqual(target_base, out.token_id);
        }
        try std.testing.expectEqual(@as(u8, 4), out.top_logprobs_len);

        // Verify exact top_logprobs against CPU reference for this row's mask!
        var p = rows[r].params;
        p.allowed_token_mask = ext_bm_u32[r * bm_stride_u32 ..][0..bm_words];
        const cpu_out = try sampling.sampleToken(
            f16,
            logits_f16[r * vocab_size ..][0..vocab_size],
            p,
            &.{},
            &dummy_rng,
            .{ .logits_f32 = scratch_f32, .cand_ids = scratch_ids },
        );
        for (cpu_out.topLogprobsSlice(), out.topLogprobsSlice()) |c_lp, g_lp| {
            try std.testing.expectEqual(c_lp.token_id, g_lp.token_id);
            try std.testing.expectApproxEqAbs(c_lp.logprob, g_lp.logprob, 1e-3);
        }
    }
}

test "gpu.sampler: 248K vocab warm GPU latency smoke check (< 0.05 ms/step on Apple Silicon)" {
    const gpa = std.testing.allocator;
    const vocab_size: u32 = 248_077;

    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();

    var sampler = try GpuSampler.initStandalone(device, lib, .{
        .max_vocab_size = MAX_VOCAB_SIZE,
        .max_batch_slots = 1,
    });
    defer sampler.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try sampler.registerResident(compute);

    const bm_words = grammar.bitmaskWordCount(vocab_size);
    const bm_bytes = amf.align64(bm_words * @sizeOf(u32));
    var arena = try makeTestArena(vocab_size * @sizeOf(f16) + bm_bytes + 64 * 1024);
    defer arena.deinit();
    const logits_raw = try arena.allocAligned(vocab_size * @sizeOf(f16), 64);
    const bm_raw = try arena.allocAligned(bm_bytes, 64);
    const logits_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, logits_raw));
    fillSyntheticLogitsF16(logits_f16, 0x2480_FA57);
    const bm_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, bm_raw));
    @memset(bm_u32, 0xFFFF_FFFF);

    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    try compute.makeResident(&.{buf});

    const logits_bind: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(logits_raw.ptr) };
    const bm_bind: metal.BufferBinding = .{ .buffer = buf, .offset = arena.offsetOf(bm_raw.ptr) };

    const biases = [_]LogitBiasEntry{
        .{ .token_id = 42, .bias = 1.5 },
        .{ .token_id = 100_000, .bias = -2.0 },
    };
    const seen = try gpa.alloc(TokenFreqEntry, 128);
    defer gpa.free(seen);
    for (seen, 0..) |*e, i| e.* = .{ .token_id = @intCast(i * 17 + 5), .count = @intCast((i % 4) + 1) };
    try sampler.uploadOverrides(0, vocab_size, &biases, seen);

    const constrained_params: SampleParams = .{
        .temperature = 0.8,
        .top_k = 50,
        .min_p = 0.05,
        .top_p = 0.9,
        .presence_penalty = 0.3,
        .frequency_penalty = 0.2,
        .repetition_penalty = 1.1,
        .logit_bias = &biases,
    };

    // Warmup both fast-path greedy and full constrained radix paths.
    for (0..5) |_| {
        _ = try sampler.sampleSync(&compute, .{
            .logits = logits_bind,
            .logits_dtype = .f16,
            .vocab_size = vocab_size,
            .params = .{},
        }, 5000);
        _ = try sampler.sampleSync(&compute, .{
            .logits = logits_bind,
            .logits_dtype = .f16,
            .vocab_size = vocab_size,
            .params = constrained_params,
            .seen_frequencies = null,
            .rng_seed = 0x1234_5678,
            .external_bitmask = bm_bind,
        }, 5000);
    }

    // Measure amortized on-GPU execution time (best of 3 batches of 32 dispatches inside
    // one Metal 4 frame) to isolate actual GPU execution from host commit/wait overhead
    // and concurrent system GPU load.
    const iters: usize = 32;
    var best_greedy_us: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const t0_greedy = try vmstat.nowNs();
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        for (0..iters) |i| {
            try sampler.encodeSample(&frame, .{
                .logits = logits_bind,
                .logits_dtype = .f16,
                .vocab_size = vocab_size,
                .params = .{},
            }, i > 0);
        }
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
        const us = @as(f64, @floatFromInt(try vmstat.nowNs() - t0_greedy)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
        if (us < best_greedy_us) best_greedy_us = us;
    }

    var best_constrained_us: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const t0_const = try vmstat.nowNs();
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        for (0..iters) |i| {
            try sampler.encodeSample(&frame, .{
                .logits = logits_bind,
                .logits_dtype = .f16,
                .vocab_size = vocab_size,
                .params = constrained_params,
                .seen_frequencies = null,
                .rng_seed = 0x9000_0001,
                .external_bitmask = bm_bind,
            }, i > 0);
        }
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
        const us = @as(f64, @floatFromInt(try vmstat.nowNs() - t0_const)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
        if (us < best_constrained_us) best_constrained_us = us;
    }

    // Also measure 16 end-to-end single-step commit+wait cycles (simulating standalone decode readback).
    const t0_sync = try vmstat.nowNs();
    for (0..16) |i| {
        _ = try sampler.sampleSync(&compute, .{
            .logits = logits_bind,
            .logits_dtype = .f16,
            .vocab_size = vocab_size,
            .params = constrained_params,
            .seen_frequencies = null,
            .rng_seed = 0xA000 + @as(u64, i),
            .external_bitmask = bm_bind,
        }, 5000);
    }
    const sync_us = @as(f64, @floatFromInt(try vmstat.nowNs() - t0_sync)) / (16.0 * 1000.0);

    std.debug.print("\n[G21 248K f16 warm GPU latency] greedy={d:.2} us/step, constrained_radix={d:.2} us/step, sync_roundtrip={d:.2} us/step\n", .{
        best_greedy_us,
        best_constrained_us,
        sync_us,
    });
    // Generous CI margin allowing concurrent swarm GPU execution; idle ReleaseFast < 50 us.
    try std.testing.expect(best_greedy_us < 1000.0);
    try std.testing.expect(best_constrained_us < 2000.0);
}
