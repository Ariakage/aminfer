//! C14 Zero-Allocation Iteration-Level Continuous Batch Scheduler
//! (`docs/decisions/0019-recurrent-depth-and-post-m5-accelerators.md`,
//! `docs/architecture/runtime-pipeline.md`).
//!
//! Manages up to `MAX_BATCH_SLOTS` (`16`) preallocated request slots across the
//! full lifecycle (`free -> queued -> prefilling -> decoding -> finished/cancelled`)
//! with:
//! - Fair round-robin chunked prefill budgeting (`max_prefill_chunk_tokens`,
//!   default `512`) so long prompts do not starve short prompts or stall active
//!   decode steps.
//! - Optional `PrefixLookupFn` hook (`R8` `PrefixCache`) that skips matched
//!   prefix tokens before prefill and passes the pinned `prefix_handle` back to
//!   `releaseSlot` on completion or mid-generation cancellation.
//! - Iteration-level batched decode (`M = B`, `1 <= B <= max_batch_size`)
//!   gathering all `.decoding` slots into one contiguous `BatchStepView` so all
//!   active requests advance together in ONE batched forward pass per iteration.
//! - Strict per-slot `RequestRng` (`D11` Secure Enclave TRNG / explicit `seed`
//!   Xoshiro256) isolation so bit-exact replay holds regardless of batch peers.
//! - Zero dynamic heap allocations after `BatchScheduler.init`.

const std = @import("std");
const page = @import("../mem/page.zig");
const rng_mod = @import("rng.zig");

/// Maximum concurrent request slots managed by one `BatchScheduler`.
pub const MAX_BATCH_SLOTS: usize = 16;
/// Maximum byte length of a per-request `.ang` domain pack identifier.
pub const MAX_DOMAIN_ID_BYTES: usize = 64;
/// Maximum per-request / default stop (EOS) token IDs.
pub const MAX_STOP_TOKENS: usize = 4;
/// Default per-iteration chunked prefill token budget (matches 512-token KV tile / ANE chunk).
pub const DEFAULT_MAX_PREFILL_CHUNK_TOKENS: u32 = 512;
/// Default preallocated prompt token capacity per slot.
pub const DEFAULT_MAX_PROMPT_TOKENS: usize = 4096;

pub const InitError = error{
    InvalidConfig,
    OutOfMemory,
};

pub const SubmitError = error{
    BatchFull,
    InvalidSlot,
    SlotNotFree,
    EmptyPrompt,
    PromptTooLong,
    InvalidMaxTokens,
    InvalidDomainId,
    EntropyUnavailable,
};

pub const BackendError = error{
    PrefillFailed,
    DecodeFailed,
    TokenizeFailed,
    SlotUnavailable,
    InvalidState,
};

pub const SlotState = enum(u8) {
    free = 0,
    queued = 1,
    prefilling = 2,
    decoding = 3,
    finished = 4,
    cancelled = 5,
};

pub const FinishReason = enum(u8) {
    none = 0,
    stop = 1,
    length = 2,
    cancelled = 3,
    err = 4,
};

pub const BatchConfig = struct {
    /// Maximum active slots (`1..=MAX_BATCH_SLOTS`).
    max_batch_size: usize = MAX_BATCH_SLOTS,
    /// Per-iteration chunked prefill token budget (`>= 1`).
    max_prefill_chunk_tokens: u32 = DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
    /// Preallocated `u32` prompt token capacity per slot (`>= 1`).
    max_prompt_tokens: usize = DEFAULT_MAX_PROMPT_TOKENS,
    /// Default EOS/stop token IDs applied when a request omits explicit stop IDs.
    default_eos_token_ids: [MAX_STOP_TOKENS]u32 = @splat(0),
    default_eos_count: u8 = 0,
};

/// Result returned by `PrefixLookupFn` (`R8` shared `KvTile` / `GDN` prefix cache).
pub const PrefixLookupResult = struct {
    /// Number of leading prompt tokens matched in cache (`0` on miss).
    /// When `matched_tokens >= prompt_len` and `prompt_len > 0`, the scheduler
    /// clamps the skip to `prompt_len - 1` so the final prompt token is still
    /// evaluated to produce first-token logits.
    matched_tokens: u32 = 0,
    /// Opaque cache handle pinned by the backend; passed back to `releaseSlot`.
    prefix_handle: u64 = 0,
};

/// Optional `R8` shared prefix cache lookup callback.
pub const PrefixLookupFn = *const fn (
    ctx: *anyopaque,
    slot_id: u16,
    prompt_tokens: []const u32,
) PrefixLookupResult;

/// Immutable view of one chunked prefill slice passed to `BatchBackend.prefillSlot`.
pub const PrefillChunkView = struct {
    slot_id: u16,
    request_id: u64,
    user_tag: u64,
    tokens: []const u32,
    start_pos: u32,
    total_prompt_len: u32,
    is_final_chunk: bool,
    prefix_handle: u64,
    temperature: f32,
    rng: *rng_mod.RequestRng,
    ngram_domain: []const u8,
};

/// Output returned by `BatchBackend.prefillSlot`.
pub const PrefillResult = struct {
    /// Sampled first generated token when `chunk.is_final_chunk == true`.
    first_token: u32 = 0,
    /// Optional updated opaque handle (e.g. newly pinned prefix cache entry).
    prefix_handle: u64 = 0,
    /// Set `true` to replace the slot's stored `prefix_handle` with `prefix_handle`.
    update_prefix_handle: bool = false,
};

/// Contiguous view of all `B` (`1 <= B <= max_batch_size`) slots currently in
/// `.decoding` handed to `BatchBackend.stepDecodeBatch` for ONE batched `M = B`
/// forward pass per iteration.
pub const BatchStepView = struct {
    slot_ids: []const u16,
    request_ids: []const u64,
    user_tags: []const u64,
    input_tokens: []const u32,
    seq_positions: []const u32,
    temperatures: []const f32,
    rngs: []const *rng_mod.RequestRng,
    ngram_domains: []const []const u8,
    prefix_handles: []const u64,

    pub inline fn batchSize(self: *const BatchStepView) usize {
        return self.slot_ids.len;
    }
};

/// Pluggable execution vtable driven by `BatchScheduler`.
pub const BatchBackend = struct {
    ctx: *anyopaque,
    /// Optional `R8` shared prefix cache lookup hook (`null` disables prefix skipping).
    lookupPrefixFn: ?PrefixLookupFn = null,
    /// Runs one chunk of prefill (`tokens.len <= max_prefill_chunk_tokens`) for `slot_id`.
    prefillSlotFn: *const fn (
        ctx: *anyopaque,
        chunk: *const PrefillChunkView,
    ) BackendError!PrefillResult,
    /// Advances all `B = view.batchSize()` active decode slots in ONE batched step,
    /// writing one sampled next token per slot into `out_next_tokens[0..B]`.
    stepDecodeBatchFn: *const fn (
        ctx: *anyopaque,
        view: *const BatchStepView,
        out_next_tokens: []u32,
    ) BackendError!void,
    /// Releases backend KV/GDN state for `slot_id` and unpins `prefix_handle`.
    releaseSlotFn: *const fn (
        ctx: *anyopaque,
        slot_id: u16,
        prefix_handle: u64,
    ) void,

    pub fn lookupPrefix(self: BatchBackend, slot_id: u16, prompt_tokens: []const u32) PrefixLookupResult {
        if (self.lookupPrefixFn) |f| return f(self.ctx, slot_id, prompt_tokens);
        return .{};
    }

    pub fn prefillSlot(self: BatchBackend, chunk: *const PrefillChunkView) BackendError!PrefillResult {
        return self.prefillSlotFn(self.ctx, chunk);
    }

    pub fn stepDecodeBatch(self: BatchBackend, view: *const BatchStepView, out_next_tokens: []u32) BackendError!void {
        return self.stepDecodeBatchFn(self.ctx, view, out_next_tokens);
    }

    pub fn releaseSlot(self: BatchBackend, slot_id: u16, prefix_handle: u64) void {
        self.releaseSlotFn(self.ctx, slot_id, prefix_handle);
    }
};

/// Optional observer callbacks invoked synchronously by `BatchScheduler.step`
/// without heap allocation.
pub const StepObserver = struct {
    ctx: *anyopaque,
    /// Invoked once when a slot finishes its final prefill chunk (before emitting
    /// the first generated token). Return `false` to cancel the slot immediately.
    onPrefillDoneFn: ?*const fn (
        ctx: *anyopaque,
        slot_id: u16,
        user_tag: u64,
        prompt_tokens: u32,
        prefix_hit_tokens: u32,
    ) bool = null,
    /// Invoked for each emitted non-stop token. Return `false` if the client
    /// disconnected so the scheduler immediately cancels and calls `releaseSlot`.
    onTokenFn: ?*const fn (
        ctx: *anyopaque,
        slot_id: u16,
        user_tag: u64,
        token_id: u32,
        generated_count: u32,
    ) bool = null,
    /// Invoked right after a slot transitions to `.finished` or `.cancelled` and
    /// `backend.releaseSlot` has run (immediately before the slot returns to `.free`).
    onFinishFn: ?*const fn (
        ctx: *anyopaque,
        slot_id: u16,
        user_tag: u64,
        reason: FinishReason,
        prompt_tokens: u32,
        generated_tokens: u32,
    ) void = null,
    /// Invoked when a backend error terminates a slot.
    onErrorFn: ?*const fn (
        ctx: *anyopaque,
        slot_id: u16,
        user_tag: u64,
        err: BackendError,
    ) void = null,
};

/// Submission descriptor for `BatchScheduler.enqueue` / `enqueueInPlace`.
pub const SubmitOptions = struct {
    request_id: u64 = 0,
    user_tag: u64 = 0,
    max_tokens: u32 = 256,
    temperature: f32 = 0.0,
    /// Explicit deterministic seed (`null` draws Secure Enclave TRNG when
    /// `temperature > 0` or greedy seed `0` when `temperature == 0`, unless
    /// `external_rng` is supplied).
    explicit_seed: ?u64 = null,
    /// Optional caller-owned `RequestRng` (e.g. `http_sse.ConnectionSlot.rng`).
    /// When non-null, `BatchStepView.rngs[b]` points directly to this RNG.
    external_rng: ?*rng_mod.RequestRng = null,
    ignore_eos: bool = false,
    eos_token_ids: []const u32 = &.{},
    /// Optional per-request `.ang` domain pack ID (empty slice selects active default).
    ngram_domain: []const u8 = "",
};

/// Cumulative iteration-level batching telemetry.
pub const BatchTelemetry = struct {
    total_batch_steps: u64 = 0,
    total_slot_tokens: u64 = 0,
    total_prefill_chunks: u64 = 0,
    total_prefill_tokens: u64 = 0,
    completed_requests: u64 = 0,
    cancelled_requests: u64 = 0,
    failed_requests: u64 = 0,
    max_observed_batch_size: u32 = 0,
    active_batch_size: u32 = 0,
    active_slots: u32 = 0,
    prefix_hit_tokens: u64 = 0,
    prefix_hit_requests: u64 = 0,

    pub fn meanBatchSize(self: BatchTelemetry) f64 {
        if (self.total_batch_steps == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_slot_tokens)) / @as(f64, @floatFromInt(self.total_batch_steps));
    }
};

/// Summary returned by one `BatchScheduler.step` iteration.
pub const StepSummary = struct {
    prefill_chunks_run: u32 = 0,
    prefill_tokens_run: u32 = 0,
    decode_batch_size: u32 = 0,
    tokens_emitted: u32 = 0,
    slots_finished: u32 = 0,
    slots_cancelled: u32 = 0,
    slots_failed: u32 = 0,
};

/// Preallocated per-slot state inside `BatchScheduler`.
pub const BatchSlot = struct {
    slot_id: u16 = 0,
    state: SlotState = .free,
    finish_reason: FinishReason = .none,
    generation: u64 = 0,
    request_id: u64 = 0,
    user_tag: u64 = 0,

    prompt_buf: []u32 = &.{},
    prompt_len: u32 = 0,
    prefill_pos: u32 = 0,
    prefix_handle: u64 = 0,
    prefix_hit_tokens: u32 = 0,

    seq_pos: u32 = 0,
    last_token: u32 = 0,
    generated_tokens: u32 = 0,
    max_tokens: u32 = 0,
    temperature: f32 = 0.0,
    ignore_eos: bool = false,
    eos_token_ids: [MAX_STOP_TOKENS]u32 = @splat(0),
    eos_count: u8 = 0,

    owned_rng: rng_mod.RequestRng = rng_mod.RequestRng.initDeterministic(0),
    rng_ptr: ?*rng_mod.RequestRng = null,

    ngram_domain_buf: [MAX_DOMAIN_ID_BYTES]u8 = @splat(0),
    ngram_domain_len: u8 = 0,

    pub fn ngramDomainSlice(self: *const BatchSlot) []const u8 {
        return self.ngram_domain_buf[0..self.ngram_domain_len];
    }

    pub fn promptSlice(self: *const BatchSlot) []const u32 {
        return self.prompt_buf[0..self.prompt_len];
    }

    pub fn rng(self: *BatchSlot) *rng_mod.RequestRng {
        return self.rng_ptr orelse &self.owned_rng;
    }

    fn isStopToken(self: *const BatchSlot, tok: u32) bool {
        if (self.ignore_eos) return false;
        for (self.eos_token_ids[0..self.eos_count]) |eos| {
            if (eos == tok) return true;
        }
        return false;
    }

    fn resetToFree(self: *BatchSlot) void {
        self.state = .free;
        self.finish_reason = .none;
        self.request_id = 0;
        self.user_tag = 0;
        self.prompt_len = 0;
        self.prefill_pos = 0;
        self.prefix_handle = 0;
        self.prefix_hit_tokens = 0;
        self.seq_pos = 0;
        self.last_token = 0;
        self.generated_tokens = 0;
        self.max_tokens = 0;
        self.temperature = 0.0;
        self.ignore_eos = false;
        self.eos_count = 0;
        self.rng_ptr = null;
        self.ngram_domain_len = 0;
    }
};

/// Zero-allocation iteration-level continuous batch scheduler (`C14`).
pub const BatchScheduler = struct {
    config: BatchConfig,
    /// Single 16 KiB-aligned backing slab holding all `max_batch_size * max_prompt_tokens`
    /// prompt token buffers (allocated once in `init`, freed once in `deinit`).
    prompt_slab: []align(page.APPLE_PAGE_SIZE) u8,
    slots: [MAX_BATCH_SLOTS]BatchSlot,
    prefill_cursor: usize,
    stats: BatchTelemetry,

    // Preallocated scratch arrays for `BatchStepView` (`0` heap allocs per step).
    view_slot_ids: [MAX_BATCH_SLOTS]u16,
    view_request_ids: [MAX_BATCH_SLOTS]u64,
    view_user_tags: [MAX_BATCH_SLOTS]u64,
    view_input_tokens: [MAX_BATCH_SLOTS]u32,
    view_seq_positions: [MAX_BATCH_SLOTS]u32,
    view_temperatures: [MAX_BATCH_SLOTS]f32,
    view_rngs: [MAX_BATCH_SLOTS]*rng_mod.RequestRng,
    view_ngram_domains: [MAX_BATCH_SLOTS][]const u8,
    view_prefix_handles: [MAX_BATCH_SLOTS]u64,
    step_out_tokens: [MAX_BATCH_SLOTS]u32,

    pub fn init(allocator: std.mem.Allocator, config: BatchConfig) InitError!BatchScheduler {
        if (config.max_batch_size == 0 or config.max_batch_size > MAX_BATCH_SLOTS) return error.InvalidConfig;
        if (config.max_prefill_chunk_tokens == 0) return error.InvalidConfig;
        if (config.max_prompt_tokens == 0) return error.InvalidConfig;
        if (config.default_eos_count > MAX_STOP_TOKENS) return error.InvalidConfig;

        const per_slot_bytes = std.math.mul(usize, config.max_prompt_tokens, @sizeOf(u32)) catch return error.InvalidConfig;
        const raw_slab_bytes = std.math.mul(usize, per_slot_bytes, config.max_batch_size) catch return error.InvalidConfig;
        const aligned_slab_bytes = std.mem.alignForward(usize, raw_slab_bytes, page.APPLE_PAGE_SIZE);

        const prompt_slab = allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), aligned_slab_bytes) catch {
            return error.OutOfMemory;
        };
        @memset(prompt_slab, 0);

        var slots: [MAX_BATCH_SLOTS]BatchSlot = undefined;
        for (0..MAX_BATCH_SLOTS) |i| {
            slots[i] = .{ .slot_id = @intCast(i) };
            if (i < config.max_batch_size) {
                const start = i * per_slot_bytes;
                const slice_bytes = prompt_slab[start .. start + per_slot_bytes];
                slots[i].prompt_buf = @alignCast(std.mem.bytesAsSlice(u32, slice_bytes));
            }
        }

        return .{
            .config = config,
            .prompt_slab = prompt_slab,
            .slots = slots,
            .prefill_cursor = 0,
            .stats = .{},
            .view_slot_ids = undefined,
            .view_request_ids = undefined,
            .view_user_tags = undefined,
            .view_input_tokens = undefined,
            .view_seq_positions = undefined,
            .view_temperatures = undefined,
            .view_rngs = undefined,
            .view_ngram_domains = undefined,
            .view_prefix_handles = undefined,
            .step_out_tokens = undefined,
        };
    }

    pub fn deinit(self: *BatchScheduler, allocator: std.mem.Allocator) void {
        if (self.prompt_slab.len > 0) {
            allocator.free(self.prompt_slab);
            self.prompt_slab = &.{};
        }
        for (&self.slots) |*s| {
            s.prompt_buf = &.{};
            s.resetToFree();
        }
    }

    pub fn maxBatchSize(self: *const BatchScheduler) usize {
        return self.config.max_batch_size;
    }

    pub fn promptSlabBytes(self: *const BatchScheduler) usize {
        return self.prompt_slab.len;
    }

    pub fn findFreeSlot(self: *const BatchScheduler) ?u16 {
        for (0..self.config.max_batch_size) |i| {
            if (self.slots[i].state == .free) return @intCast(i);
        }
        return null;
    }

    pub fn hasFreeSlot(self: *const BatchScheduler) bool {
        return self.findFreeSlot() != null;
    }

    /// Returns the preallocated prompt token buffer for a `.free` slot so callers
    /// (such as `http_sse.Server`) can tokenize directly into slot storage with zero copy.
    pub fn slotPromptBuffer(self: *BatchScheduler, slot_id: u16) []u32 {
        std.debug.assert(slot_id < self.config.max_batch_size);
        return self.slots[slot_id].prompt_buf;
    }

    pub fn slotState(self: *const BatchScheduler, slot_id: u16) SlotState {
        std.debug.assert(slot_id < self.config.max_batch_size);
        return self.slots[slot_id].state;
    }

    pub fn slotPtr(self: *BatchScheduler, slot_id: u16) *BatchSlot {
        std.debug.assert(slot_id < self.config.max_batch_size);
        return &self.slots[slot_id];
    }

    pub fn activeSlotCount(self: *const BatchScheduler) usize {
        var n: usize = 0;
        for (0..self.config.max_batch_size) |i| {
            if (self.slots[i].state != .free) n += 1;
        }
        return n;
    }

    pub fn decodingSlotCount(self: *const BatchScheduler) usize {
        var n: usize = 0;
        for (0..self.config.max_batch_size) |i| {
            if (self.slots[i].state == .decoding) n += 1;
        }
        return n;
    }

    pub fn telemetry(self: *const BatchScheduler) BatchTelemetry {
        var t = self.stats;
        t.active_slots = @intCast(self.activeSlotCount());
        t.active_batch_size = @intCast(self.decodingSlotCount());
        return t;
    }

    /// Enqueues a request by copying `prompt_tokens` into a free slot's
    /// preallocated buffer and transitioning `free -> queued`. Zero heap allocations.
    pub fn enqueue(
        self: *BatchScheduler,
        prompt_tokens: []const u32,
        opts: SubmitOptions,
    ) SubmitError!u16 {
        const slot_id = self.findFreeSlot() orelse return error.BatchFull;
        if (prompt_tokens.len == 0) return error.EmptyPrompt;
        if (prompt_tokens.len > self.config.max_prompt_tokens) return error.PromptTooLong;
        @memcpy(self.slots[slot_id].prompt_buf[0..prompt_tokens.len], prompt_tokens);
        try self.enqueueInPlace(slot_id, prompt_tokens.len, opts);
        return slot_id;
    }

    /// Commits a request whose tokens were already written into `slotPromptBuffer(slot_id)`
    /// (`0..prompt_len`), transitioning `free -> queued`. Zero heap allocations.
    pub fn enqueueInPlace(
        self: *BatchScheduler,
        slot_id: u16,
        prompt_len: usize,
        opts: SubmitOptions,
    ) SubmitError!void {
        if (slot_id >= self.config.max_batch_size) return error.InvalidSlot;
        const slot = &self.slots[slot_id];
        if (slot.state != .free) return error.SlotNotFree;
        if (prompt_len == 0) return error.EmptyPrompt;
        if (prompt_len > self.config.max_prompt_tokens) return error.PromptTooLong;
        if (opts.max_tokens == 0) return error.InvalidMaxTokens;
        if (opts.ngram_domain.len > MAX_DOMAIN_ID_BYTES) return error.InvalidDomainId;
        for (opts.ngram_domain) |c| {
            if (c < 0x20 or c == 0x7f) return error.InvalidDomainId;
        }

        // Initialize D11 slot-local RequestRng when caller does not supply an external one.
        if (opts.external_rng) |ext| {
            slot.rng_ptr = ext;
        } else {
            if (opts.explicit_seed) |s| {
                slot.owned_rng = rng_mod.RequestRng.initDeterministic(s);
            } else if (opts.temperature > 0.0) {
                slot.owned_rng = rng_mod.RequestRng.init(null) catch return error.EntropyUnavailable;
            } else {
                slot.owned_rng = rng_mod.RequestRng.initDeterministic(0);
            }
            slot.rng_ptr = null;
        }

        slot.generation +%= 1;
        slot.state = .queued;
        slot.finish_reason = .none;
        slot.request_id = opts.request_id;
        slot.user_tag = opts.user_tag;
        slot.prompt_len = @intCast(prompt_len);
        slot.prefill_pos = 0;
        slot.prefix_handle = 0;
        slot.prefix_hit_tokens = 0;
        slot.seq_pos = 0;
        slot.last_token = 0;
        slot.generated_tokens = 0;
        slot.max_tokens = opts.max_tokens;
        slot.temperature = opts.temperature;
        slot.ignore_eos = opts.ignore_eos;

        if (opts.eos_token_ids.len > 0) {
            const n = @min(opts.eos_token_ids.len, MAX_STOP_TOKENS);
            @memcpy(slot.eos_token_ids[0..n], opts.eos_token_ids[0..n]);
            slot.eos_count = @intCast(n);
        } else {
            slot.eos_token_ids = self.config.default_eos_token_ids;
            slot.eos_count = self.config.default_eos_count;
        }

        @memset(&slot.ngram_domain_buf, 0);
        if (opts.ngram_domain.len > 0) {
            @memcpy(slot.ngram_domain_buf[0..opts.ngram_domain.len], opts.ngram_domain);
        }
        slot.ngram_domain_len = @intCast(opts.ngram_domain.len);
        self.refreshActiveCounts();
    }

    /// Immediately cancels an active slot (e.g. on client TCP disconnect),
    /// transitions it through `.cancelled`, invokes `backend.releaseSlot` to
    /// free KV/GDN state & unpin any shared prefix handle, notifies `observer`
    /// (if provided), and recycles the slot back to `.free`.
    pub fn cancelSlot(
        self: *BatchScheduler,
        slot_id: u16,
        backend: BatchBackend,
        observer: ?StepObserver,
    ) bool {
        if (slot_id >= self.config.max_batch_size) return false;
        const slot = &self.slots[slot_id];
        if (slot.state == .free) return false;

        const user_tag = slot.user_tag;
        const prompt_tokens = slot.prompt_len;
        const generated_tokens = slot.generated_tokens;
        const prefix_handle = slot.prefix_handle;

        slot.state = .cancelled;
        slot.finish_reason = .cancelled;
        backend.releaseSlot(slot_id, prefix_handle);
        self.stats.cancelled_requests += 1;

        if (observer) |obs| {
            if (obs.onFinishFn) |f| {
                f(obs.ctx, slot_id, user_tag, .cancelled, prompt_tokens, generated_tokens);
            }
        }

        slot.resetToFree();
        self.refreshActiveCounts();
        return true;
    }

    fn finishSlotInternal(
        self: *BatchScheduler,
        slot_id: u16,
        reason: FinishReason,
        backend: BatchBackend,
        observer: ?StepObserver,
    ) void {
        const slot = &self.slots[slot_id];
        const user_tag = slot.user_tag;
        const prompt_tokens = slot.prompt_len;
        const generated_tokens = slot.generated_tokens;
        const prefix_handle = slot.prefix_handle;

        slot.state = if (reason == .cancelled) .cancelled else .finished;
        slot.finish_reason = reason;
        backend.releaseSlot(slot_id, prefix_handle);
        switch (reason) {
            .cancelled => self.stats.cancelled_requests += 1,
            .err => self.stats.failed_requests += 1,
            .stop, .length, .none => self.stats.completed_requests += 1,
        }

        if (observer) |obs| {
            if (obs.onFinishFn) |f| {
                f(obs.ctx, slot_id, user_tag, reason, prompt_tokens, generated_tokens);
            }
        }

        slot.resetToFree();
    }

    fn failSlotInternal(
        self: *BatchScheduler,
        slot_id: u16,
        err: BackendError,
        backend: BatchBackend,
        observer: ?StepObserver,
    ) void {
        const slot = &self.slots[slot_id];
        const user_tag = slot.user_tag;
        const prefix_handle = slot.prefix_handle;

        slot.state = .finished;
        slot.finish_reason = .err;
        backend.releaseSlot(slot_id, prefix_handle);
        self.stats.failed_requests += 1;

        if (observer) |obs| {
            if (obs.onErrorFn) |f| {
                f(obs.ctx, slot_id, user_tag, err);
            }
        }

        slot.resetToFree();
    }

    fn refreshActiveCounts(self: *BatchScheduler) void {
        self.stats.active_slots = @intCast(self.activeSlotCount());
        self.stats.active_batch_size = @intCast(self.decodingSlotCount());
    }

    /// Executes one iteration-level continuous batching tick:
    /// 1. Fair round-robin chunked prefill up to `config.max_prefill_chunk_tokens`,
    ///    invoking optional `PrefixLookupFn` on `.queued -> .prefilling` admission
    ///    to skip matched prefix tokens.
    /// 2. Batched `M = B` decode step gathering all currently `.decoding` slots
    ///    into one contiguous `BatchStepView` and calling `backend.stepDecodeBatch`.
    ///
    /// Performs zero dynamic heap allocations.
    pub fn step(
        self: *BatchScheduler,
        backend: BatchBackend,
        observer: ?StepObserver,
    ) StepSummary {
        var summary: StepSummary = .{};
        const max_b = self.config.max_batch_size;

        // -------------------------------------------------------------------
        // Phase 1: Fair round-robin chunked prefill within `max_prefill_chunk_tokens`
        // -------------------------------------------------------------------
        var prefill_budget: u32 = self.config.max_prefill_chunk_tokens;
        var visited: usize = 0;
        var cursor: usize = self.prefill_cursor % max_b;

        while (prefill_budget > 0 and visited < max_b) : (visited += 1) {
            const slot_id: u16 = @intCast(cursor);
            cursor = (cursor + 1) % max_b;
            const slot = &self.slots[slot_id];

            if (slot.state == .queued) {
                // R8 shared prefix lookup hook: skip matched prefix tokens before prefill.
                const hit = backend.lookupPrefix(slot_id, slot.promptSlice());
                slot.prefix_handle = hit.prefix_handle;
                const skip: u32 = if (slot.prompt_len > 1)
                    @min(hit.matched_tokens, slot.prompt_len - 1)
                else
                    0;
                slot.prefix_hit_tokens = skip;
                slot.prefill_pos = skip;
                if (skip > 0) {
                    self.stats.prefix_hit_tokens += skip;
                    self.stats.prefix_hit_requests += 1;
                }
                slot.state = .prefilling;
            }

            if (slot.state != .prefilling) continue;

            const remaining: u32 = slot.prompt_len - slot.prefill_pos;
            std.debug.assert(remaining > 0);
            const take: u32 = @min(remaining, prefill_budget);
            const start_pos: u32 = slot.prefill_pos;
            const is_final: bool = (start_pos + take == slot.prompt_len);

            const chunk_view = PrefillChunkView{
                .slot_id = slot_id,
                .request_id = slot.request_id,
                .user_tag = slot.user_tag,
                .tokens = slot.prompt_buf[start_pos .. start_pos + take],
                .start_pos = start_pos,
                .total_prompt_len = slot.prompt_len,
                .is_final_chunk = is_final,
                .prefix_handle = slot.prefix_handle,
                .temperature = slot.temperature,
                .rng = slot.rng(),
                .ngram_domain = slot.ngramDomainSlice(),
            };

            const res = backend.prefillSlot(&chunk_view) catch |err| {
                self.failSlotInternal(slot_id, err, backend, observer);
                summary.slots_failed += 1;
                self.prefill_cursor = cursor;
                continue;
            };

            if (res.update_prefix_handle) {
                slot.prefix_handle = res.prefix_handle;
            }
            slot.prefill_pos += take;
            prefill_budget -= take;
            summary.prefill_chunks_run += 1;
            summary.prefill_tokens_run += take;
            self.stats.total_prefill_chunks += 1;
            self.stats.total_prefill_tokens += take;
            self.prefill_cursor = cursor;

            if (!is_final) continue;

            // Final prefill chunk finished: notify observer and emit first generated token.
            if (observer) |obs| {
                if (obs.onPrefillDoneFn) |f| {
                    if (!f(obs.ctx, slot_id, slot.user_tag, slot.prompt_len, slot.prefix_hit_tokens)) {
                        self.finishSlotInternal(slot_id, .cancelled, backend, observer);
                        summary.slots_cancelled += 1;
                        continue;
                    }
                }
            }

            const first_tok = res.first_token;
            if (slot.isStopToken(first_tok)) {
                self.finishSlotInternal(slot_id, .stop, backend, observer);
                summary.slots_finished += 1;
                continue;
            }

            slot.generated_tokens = 1;
            slot.last_token = first_tok;
            slot.seq_pos = slot.prompt_len;
            summary.tokens_emitted += 1;

            if (observer) |obs| {
                if (obs.onTokenFn) |f| {
                    if (!f(obs.ctx, slot_id, slot.user_tag, first_tok, slot.generated_tokens)) {
                        self.finishSlotInternal(slot_id, .cancelled, backend, observer);
                        summary.slots_cancelled += 1;
                        continue;
                    }
                }
            }

            if (slot.generated_tokens >= slot.max_tokens) {
                self.finishSlotInternal(slot_id, .length, backend, observer);
                summary.slots_finished += 1;
                continue;
            }

            slot.state = .decoding;
        }

        // -------------------------------------------------------------------
        // Phase 2: Gather all `.decoding` slots into ONE batched `M = B` step
        // -------------------------------------------------------------------
        var batch_len: usize = 0;
        for (0..max_b) |i| {
            const slot = &self.slots[i];
            if (slot.state != .decoding) continue;
            self.view_slot_ids[batch_len] = slot.slot_id;
            self.view_request_ids[batch_len] = slot.request_id;
            self.view_user_tags[batch_len] = slot.user_tag;
            self.view_input_tokens[batch_len] = slot.last_token;
            self.view_seq_positions[batch_len] = slot.seq_pos;
            self.view_temperatures[batch_len] = slot.temperature;
            self.view_rngs[batch_len] = slot.rng();
            self.view_ngram_domains[batch_len] = slot.ngramDomainSlice();
            self.view_prefix_handles[batch_len] = slot.prefix_handle;
            batch_len += 1;
        }

        if (batch_len > 0) {
            const view = BatchStepView{
                .slot_ids = self.view_slot_ids[0..batch_len],
                .request_ids = self.view_request_ids[0..batch_len],
                .user_tags = self.view_user_tags[0..batch_len],
                .input_tokens = self.view_input_tokens[0..batch_len],
                .seq_positions = self.view_seq_positions[0..batch_len],
                .temperatures = self.view_temperatures[0..batch_len],
                .rngs = self.view_rngs[0..batch_len],
                .ngram_domains = self.view_ngram_domains[0..batch_len],
                .prefix_handles = self.view_prefix_handles[0..batch_len],
            };

            const out_slice = self.step_out_tokens[0..batch_len];
            backend.stepDecodeBatch(&view, out_slice) catch |err| {
                for (self.view_slot_ids[0..batch_len]) |sid| {
                    self.failSlotInternal(sid, err, backend, observer);
                    summary.slots_failed += 1;
                }
                self.refreshActiveCounts();
                return summary;
            };

            const b_u32: u32 = @intCast(batch_len);
            summary.decode_batch_size = b_u32;
            self.stats.total_batch_steps += 1;
            self.stats.total_slot_tokens += b_u32;
            if (b_u32 > self.stats.max_observed_batch_size) {
                self.stats.max_observed_batch_size = b_u32;
            }

            for (0..batch_len) |b| {
                const sid = self.view_slot_ids[b];
                const slot = &self.slots[sid];
                const next_tok = out_slice[b];

                if (slot.isStopToken(next_tok)) {
                    self.finishSlotInternal(sid, .stop, backend, observer);
                    summary.slots_finished += 1;
                    continue;
                }

                slot.generated_tokens += 1;
                slot.last_token = next_tok;
                slot.seq_pos += 1;
                summary.tokens_emitted += 1;

                if (observer) |obs| {
                    if (obs.onTokenFn) |f| {
                        if (!f(obs.ctx, sid, slot.user_tag, next_tok, slot.generated_tokens)) {
                            self.finishSlotInternal(sid, .cancelled, backend, observer);
                            summary.slots_cancelled += 1;
                            continue;
                        }
                    }
                }

                if (slot.generated_tokens >= slot.max_tokens) {
                    self.finishSlotInternal(sid, .length, backend, observer);
                    summary.slots_finished += 1;
                }
            }
        }

        self.refreshActiveCounts();
        return summary;
    }
};

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

const TestMallocStatistics = extern struct {
    blocks_in_use: c_uint,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *TestMallocStatistics) void;

test "BatchScheduler fair chunked prefill, R8 prefix skip/unpin, M=8 batched decode, D11 seed parity, and mid-stream cancel" {
    const gpa = std.testing.allocator;
    const EOS_TOK: u32 = 9999;

    const MockBackend = struct {
        shared_prefix_len: u32 = 600,
        pinned_handles: [MAX_BATCH_SLOTS]u64 = @splat(0),
        released_handles: [64]u64 = @splat(0),
        released_count: usize = 0,
        next_handle: u64 = 100,
        prefill_starts: [MAX_BATCH_SLOTS]u32 = @splat(std.math.maxInt(u32)),
        prefill_chunks_seen: u32 = 0,
        max_seen_batch_size: usize = 0,
        saw_domain_repo: bool = false,
        saw_domain_docs: bool = false,
        /// Optional EOS trigger on a specific slot's sequence offset.
        eos_on_user_tag: u64 = std.math.maxInt(u64),
        eos_at_pos: u32 = 0,

        fn lookupPrefix(ctx_ptr: *anyopaque, slot_id: u16, prompt_tokens: []const u32) PrefixLookupResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            // Match whenever prompt begins with sentinel token 7777 and has > shared_prefix_len tokens.
            if (prompt_tokens.len > self.shared_prefix_len and prompt_tokens[0] == 7777) {
                const handle = self.next_handle;
                self.next_handle += 1;
                self.pinned_handles[slot_id] = handle;
                return .{
                    .matched_tokens = self.shared_prefix_len,
                    .prefix_handle = handle,
                };
            }
            return .{};
        }

        fn sampleToken(rng_handle: *rng_mod.RequestRng, temp: f32, pos: u32, prev: u32) u32 {
            if (temp > 0.0) {
                const draw = rng_handle.nextU64();
                return @intCast((draw ^ (@as(u64, pos) *% 31) ^ prev) % 4096 + 1);
            }
            return (pos * 17 + prev) % 4096 + 1;
        }

        fn prefillSlot(ctx_ptr: *anyopaque, chunk: *const PrefillChunkView) BackendError!PrefillResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            self.prefill_chunks_seen += 1;
            if (self.prefill_starts[chunk.slot_id] == std.math.maxInt(u32)) {
                self.prefill_starts[chunk.slot_id] = chunk.start_pos;
            }
            if (std.mem.eql(u8, chunk.ngram_domain, "repo")) self.saw_domain_repo = true;
            if (std.mem.eql(u8, chunk.ngram_domain, "docs")) self.saw_domain_docs = true;
            if (!chunk.is_final_chunk) return .{};

            const last_prompt_tok = chunk.tokens[chunk.tokens.len - 1];
            const tok = sampleToken(chunk.rng, chunk.temperature, chunk.total_prompt_len, last_prompt_tok);
            return .{ .first_token = tok };
        }

        fn stepDecodeBatch(ctx_ptr: *anyopaque, view: *const BatchStepView, out_next: []u32) BackendError!void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (view.batchSize() > self.max_seen_batch_size) {
                self.max_seen_batch_size = view.batchSize();
            }
            for (0..view.batchSize()) |b| {
                if (std.mem.eql(u8, view.ngram_domains[b], "repo")) self.saw_domain_repo = true;
                if (std.mem.eql(u8, view.ngram_domains[b], "docs")) self.saw_domain_docs = true;
                if (view.user_tags[b] == self.eos_on_user_tag and view.seq_positions[b] == self.eos_at_pos) {
                    out_next[b] = EOS_TOK;
                } else {
                    out_next[b] = sampleToken(
                        view.rngs[b],
                        view.temperatures[b],
                        view.seq_positions[b],
                        view.input_tokens[b],
                    );
                }
            }
        }

        fn releaseSlot(ctx_ptr: *anyopaque, slot_id: u16, prefix_handle: u64) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (prefix_handle != 0) {
                std.debug.assert(self.pinned_handles[slot_id] == prefix_handle);
                self.pinned_handles[slot_id] = 0;
                if (self.released_count < self.released_handles.len) {
                    self.released_handles[self.released_count] = prefix_handle;
                    self.released_count += 1;
                }
            }
            self.prefill_starts[slot_id] = std.math.maxInt(u32);
        }

        fn backend(self: *@This()) BatchBackend {
            return .{
                .ctx = self,
                .lookupPrefixFn = lookupPrefix,
                .prefillSlotFn = prefillSlot,
                .stepDecodeBatchFn = stepDecodeBatch,
                .releaseSlotFn = releaseSlot,
            };
        }
    };

    const Recorder = struct {
        tokens: [8][32]u32 = undefined,
        counts: [8]usize = @splat(0),
        reasons: [8]FinishReason = @splat(.none),
        prefill_hits: [8]u32 = @splat(0),

        fn onPrefillDone(ctx_ptr: *anyopaque, _: u16, user_tag: u64, _: u32, prefix_hit: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            self.prefill_hits[@intCast(user_tag)] = prefix_hit;
            return true;
        }

        fn onToken(ctx_ptr: *anyopaque, _: u16, user_tag: u64, tok: u32, _: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            const idx: usize = @intCast(user_tag);
            if (self.counts[idx] < 32) {
                self.tokens[idx][self.counts[idx]] = tok;
                self.counts[idx] += 1;
            }
            return true;
        }

        fn onFinish(ctx_ptr: *anyopaque, _: u16, user_tag: u64, reason: FinishReason, _: u32, _: u32) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            self.reasons[@intCast(user_tag)] = reason;
        }

        fn observer(self: *@This()) StepObserver {
            return .{
                .ctx = self,
                .onPrefillDoneFn = onPrefillDone,
                .onTokenFn = onToken,
                .onFinishFn = onFinish,
            };
        }
    };

    // 1. Capture single-request D11 explicit-seed reference trajectory (B = 1).
    var solo_sched = try BatchScheduler.init(gpa, .{
        .max_batch_size = 8,
        .max_prefill_chunk_tokens = 256,
        .max_prompt_tokens = 1024,
        .default_eos_token_ids = .{ EOS_TOK, 0, 0, 0 },
        .default_eos_count = 1,
    });
    defer solo_sched.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), solo_sched.promptSlabBytes() % page.APPLE_PAGE_SIZE);

    var solo_mock = MockBackend{};
    var solo_rec = Recorder{};
    const ref_prompt = [_]u32{ 11, 22, 33, 44, 55, 66, 77, 88 };
    _ = try solo_sched.enqueue(&ref_prompt, .{
        .request_id = 1,
        .user_tag = 0,
        .max_tokens = 10,
        .temperature = 0.8,
        .explicit_seed = 0xCAFE_BABE_1234_5678,
        .ngram_domain = "repo",
    });
    while (solo_sched.activeSlotCount() > 0) {
        _ = solo_sched.step(solo_mock.backend(), solo_rec.observer());
    }
    try std.testing.expectEqual(@as(usize, 10), solo_rec.counts[0]);
    try std.testing.expectEqual(FinishReason.length, solo_rec.reasons[0]);

    // 2. Now run 8 concurrent requests together (including shared 600-tok prefix skip,
    //    EOS early stop, mid-stream cancel, and the exact same explicit-seed request!).
    var sched = try BatchScheduler.init(gpa, .{
        .max_batch_size = 8,
        .max_prefill_chunk_tokens = 512,
        .max_prompt_tokens = 1024,
        .default_eos_token_ids = .{ EOS_TOK, 0, 0, 0 },
        .default_eos_count = 1,
    });
    defer sched.deinit(gpa);

    var mock = MockBackend{
        .shared_prefix_len = 600,
        .eos_on_user_tag = 2,
        .eos_at_pos = 8 + 4, // EOS after 5 tokens on tag 2
    };
    var rec = Recorder{};

    var long_prefix_prompt: [700]u32 = undefined;
    for (&long_prefix_prompt, 0..) |*t, i| t.* = if (i == 0) 7777 else @intCast(i + 10);

    // Tag 0: identical to solo_rec (explicit seed -> MUST be bit-exact even in B=8 batch!).
    _ = try sched.enqueue(&ref_prompt, .{
        .request_id = 100,
        .user_tag = 0,
        .max_tokens = 10,
        .temperature = 0.8,
        .explicit_seed = 0xCAFE_BABE_1234_5678,
        .ngram_domain = "repo",
    });
    // Tag 1: 700-token prompt hitting 600-token prefix cache -> only 100 uncached suffix tokens prefilled!
    const prefix_slot = try sched.enqueue(&long_prefix_prompt, .{
        .request_id = 101,
        .user_tag = 1,
        .max_tokens = 10,
        .temperature = 0.7,
        .ngram_domain = "docs",
    });
    // Tag 2: hits EOS early.
    _ = try sched.enqueue(&ref_prompt, .{
        .request_id = 102,
        .user_tag = 2,
        .max_tokens = 12,
        .temperature = 0.0,
        .ngram_domain = "repo",
    });
    // Tag 3: will be cancelled mid-generation while holding a pinned prefix handle!
    const cancel_slot = try sched.enqueue(&long_prefix_prompt, .{
        .request_id = 103,
        .user_tag = 3,
        .max_tokens = 12,
        .temperature = 0.7,
        .ngram_domain = "docs",
    });
    // Tags 4..7: 4 more concurrent decoding requests to hit B = 8!
    for (4..8) |tag| {
        _ = try sched.enqueue(&ref_prompt, .{
            .request_id = 100 + tag,
            .user_tag = tag,
            .max_tokens = 10,
            .temperature = 0.6,
            .ngram_domain = if ((tag & 1) == 0) "repo" else "docs",
        });
    }

    try std.testing.expectEqual(@as(usize, 8), sched.activeSlotCount());
    try std.testing.expectEqual(SlotState.queued, sched.slotState(prefix_slot));

    // Step 1: total uncached prefill tokens = 6 * 8 + 2 * 100 = 248 <= 512 budget ->
    // ALL 8 slots complete prefill AND immediately execute ONE batched B = 8 decode step!
    const s1 = sched.step(mock.backend(), rec.observer());
    try std.testing.expectEqual(@as(u32, 248), s1.prefill_tokens_run);
    try std.testing.expectEqual(@as(u32, 8), s1.decode_batch_size);
    try std.testing.expectEqual(@as(u32, 600), rec.prefill_hits[1]);
    try std.testing.expectEqual(@as(u32, 600), rec.prefill_hits[3]);
    try std.testing.expectEqual(@as(u32, 600), mock.prefill_starts[prefix_slot]);

    // Cancel Tag 3 mid-generation -> must immediately releaseSlot & unpin its prefix handle!
    const cancel_handle = sched.slotPtr(cancel_slot).prefix_handle;
    try std.testing.expect(cancel_handle != 0);
    try std.testing.expect(sched.cancelSlot(cancel_slot, mock.backend(), rec.observer()));
    try std.testing.expectEqual(SlotState.free, sched.slotState(cancel_slot));
    try std.testing.expectEqual(FinishReason.cancelled, rec.reasons[3]);
    try std.testing.expectEqual(@as(u64, 0), mock.pinned_handles[cancel_slot]);

    // Run remaining steps until all slots drain.
    while (sched.activeSlotCount() > 0) {
        _ = sched.step(mock.backend(), rec.observer());
    }

    // Verify D11 explicit-seed bit-exactness under concurrent M=8 batching!
    try std.testing.expectEqual(@as(usize, 10), rec.counts[0]);
    try std.testing.expectEqualSlices(u32, solo_rec.tokens[0][0..10], rec.tokens[0][0..10]);
    try std.testing.expectEqual(FinishReason.length, rec.reasons[0]);
    try std.testing.expectEqual(FinishReason.stop, rec.reasons[2]);
    try std.testing.expectEqual(@as(usize, 5), rec.counts[2]);
    try std.testing.expect(mock.saw_domain_repo);
    try std.testing.expect(mock.saw_domain_docs);
    try std.testing.expectEqual(@as(usize, 2), mock.released_count);

    const tel = sched.telemetry();
    try std.testing.expectEqual(@as(u32, 8), tel.max_observed_batch_size);
    try std.testing.expect(tel.meanBatchSize() > 5.0);
    try std.testing.expectEqual(@as(u64, 1200), tel.prefix_hit_tokens);
    try std.testing.expectEqual(@as(u64, 2), tel.prefix_hit_requests);
    try std.testing.expectEqual(@as(u64, 1), tel.cancelled_requests);
    try std.testing.expectEqual(@as(u64, 7), tel.completed_requests);
    try std.testing.expectEqual(@as(u32, 0), tel.active_slots);
    try std.testing.expectEqual(@as(u32, 0), tel.active_batch_size);
}

test "BatchScheduler zero steady-state heap allocations (malloc_zone_statistics)" {
    const gpa = std.testing.allocator;
    var sched = try BatchScheduler.init(gpa, .{
        .max_batch_size = 8,
        .max_prefill_chunk_tokens = 512,
        .max_prompt_tokens = 256,
    });
    defer sched.deinit(gpa);

    const FastBackend = struct {
        fn prefill(_: *anyopaque, chunk: *const PrefillChunkView) BackendError!PrefillResult {
            return .{ .first_token = @intCast((chunk.rng.nextU64() % 1000) + 1) };
        }
        fn decode(_: *anyopaque, view: *const BatchStepView, out: []u32) BackendError!void {
            for (0..view.batchSize()) |b| {
                out[b] = @intCast((view.rngs[b].nextU64() % 1000) + 1);
            }
        }
        fn release(_: *anyopaque, _: u16, _: u64) void {}
    };
    var dummy: u8 = 0;
    const be = BatchBackend{
        .ctx = &dummy,
        .prefillSlotFn = FastBackend.prefill,
        .stepDecodeBatchFn = FastBackend.decode,
        .releaseSlotFn = FastBackend.release,
    };

    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // Warmup one full wave.
    for (0..8) |i| {
        _ = try sched.enqueue(&prompt, .{ .request_id = i, .max_tokens = 8, .temperature = 0.7 });
    }
    while (sched.activeSlotCount() > 0) {
        _ = sched.step(be, null);
    }

    var m_warm: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_warm);
    var m_before: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_before);

    for (0..8) |wave| {
        for (0..8) |i| {
            _ = try sched.enqueue(&prompt, .{
                .request_id = wave * 8 + i,
                .max_tokens = 12,
                .temperature = 0.7,
                .explicit_seed = if ((i & 1) == 0) @as(u64, 42 + i) else null,
            });
        }
        while (sched.activeSlotCount() > 0) {
            _ = sched.step(be, null);
        }
    }

    var m_after: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_after);
    try std.testing.expectEqual(m_before.blocks_in_use, m_after.blocks_in_use);
    try std.testing.expectEqual(m_before.size_in_use, m_after.size_in_use);
    try std.testing.expectEqual(@as(u32, 8), sched.telemetry().max_observed_batch_size);
}

test "BatchScheduler fair round-robin chunked prefill avoids head-of-line starvation and releases slots on backend error" {
    const gpa = std.testing.allocator;
    var sched = try BatchScheduler.init(gpa, .{
        .max_batch_size = 4,
        .max_prefill_chunk_tokens = 128,
        .max_prompt_tokens = 512,
    });
    defer sched.deinit(gpa);

    const State = struct {
        fail_slot: ?u16 = null,
        releases: u32 = 0,
        error_events: u32 = 0,
        short_finished_step: u32 = 0,
        long_finished_step: u32 = 0,
        cur_step: u32 = 0,

        fn prefill(ctx_ptr: *anyopaque, chunk: *const PrefillChunkView) BackendError!PrefillResult {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (self.fail_slot != null and self.fail_slot.? == chunk.slot_id) {
                return error.PrefillFailed;
            }
            if (!chunk.is_final_chunk) return .{};
            return .{ .first_token = 42 };
        }
        fn decode(_: *anyopaque, view: *const BatchStepView, out: []u32) BackendError!void {
            for (0..view.batchSize()) |b| out[b] = 43;
        }
        fn release(ctx_ptr: *anyopaque, _: u16, _: u64) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            self.releases += 1;
        }
        fn onFinish(ctx_ptr: *anyopaque, _: u16, tag: u64, _: FinishReason, _: u32, _: u32) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            if (tag == 1) self.short_finished_step = self.cur_step;
            if (tag == 0) self.long_finished_step = self.cur_step;
        }
        fn onError(ctx_ptr: *anyopaque, _: u16, _: u64, _: BackendError) void {
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
            self.error_events += 1;
        }
    };

    var st = State{};
    const be = BatchBackend{
        .ctx = &st,
        .prefillSlotFn = State.prefill,
        .stepDecodeBatchFn = State.decode,
        .releaseSlotFn = State.release,
    };
    const obs = StepObserver{
        .ctx = &st,
        .onFinishFn = State.onFinish,
        .onErrorFn = State.onError,
    };

    var long_p: [320]u32 = undefined;
    @memset(&long_p, 7);
    const short_p = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // Enqueue long (320 tok -> 3 chunks of <=128) then short (8 tok, max_tokens=2).
    _ = try sched.enqueue(&long_p, .{ .request_id = 1, .user_tag = 0, .max_tokens = 2 });
    _ = try sched.enqueue(&short_p, .{ .request_id = 2, .user_tag = 1, .max_tokens = 2 });

    while (sched.activeSlotCount() > 0) {
        st.cur_step += 1;
        _ = sched.step(be, obs);
    }
    // Step 1: long gets 128/320. Step 2: round-robin serves short (8/8 -> finishes in step 2!) AND long gets 120/320!
    // Step 3: long gets remaining 72/320 and finishes in step 3!
    try std.testing.expectEqual(@as(u32, 2), st.short_finished_step);
    try std.testing.expectEqual(@as(u32, 3), st.long_finished_step);
    try std.testing.expectEqual(@as(u32, 2), st.releases);

    // Backend prefill error cleanly calls releaseSlot, onError, and frees the slot.
    const err_slot = try sched.enqueue(&short_p, .{ .request_id = 3, .user_tag = 2, .max_tokens = 4 });
    st.fail_slot = err_slot;
    const err_sum = sched.step(be, obs);
    try std.testing.expectEqual(@as(u32, 1), err_sum.slots_failed);
    try std.testing.expectEqual(@as(u32, 1), st.error_events);
    try std.testing.expectEqual(@as(u32, 3), st.releases);
    try std.testing.expectEqual(SlotState.free, sched.slotState(err_slot));
    try std.testing.expectEqual(@as(u64, 1), sched.telemetry().failed_requests);
}
