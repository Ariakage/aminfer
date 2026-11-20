//! Zero-allocation L1 Suffix Automaton over `u32` token IDs (status row D1,
//! ADR-0005, docs/architecture/speculative-decoding.md).
//!
//! Online Blumer et al. construction with amortized O(1) work per appended
//! token. All state storage, open-addressing transition slots (load factor
//! strictly <= 0.5), and flat token history live in a single preallocated
//! caller-provided buffer (carvable from `WiredArena`). Nothing is allocated
//! on the heap after initialization.

const std = @import("std");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Minimum suffix match length treated as a high-confidence L1 seed prefix (ADR-0005).
pub const SEED_MIN_MATCH_LEN: u32 = 4;

/// Sentinel for an empty open-addressing transition slot.
pub const EMPTY_STATE: u32 = std.math.maxInt(u32);

/// Sentinel for the end of a state's outgoing transition linked list.
pub const NO_EDGE: u32 = std.math.maxInt(u32);

/// Upper bound on `max_tokens` ensuring state and transition counts stay within
/// `u32` indices without overlapping `EMPTY_STATE` / `NO_EDGE`.
pub const MAX_SUPPORTED_TOKENS: u32 = 1 << 28;

pub const LayoutError = error{
    /// `max_tokens` is zero or exceeds `MAX_SUPPORTED_TOKENS`.
    InvalidCapacity,
    /// Required byte size overflows `usize`.
    Overflow,
};

pub const InitError = LayoutError || error{
    /// Backing buffer pointer is not aligned to `SuffixAutomaton.backing_alignment`.
    UnalignedBacking,
    /// Backing buffer length is smaller than `requiredBytes(max_tokens)`.
    BufferTooSmall,
};

pub const AppendError = error{
    /// Appending would exceed `max_tokens`.
    CapacityExceeded,
};

/// Suffix automaton equivalence class state (16 bytes, zero padding).
pub const State = extern struct {
    /// Length of the longest substring in this equivalence class.
    len: u32,
    /// Suffix link state index (`-1` for root state `0`).
    link: i32,
    /// End index in `history` of the earliest occurrence of this class.
    first_end_pos: u32,
    /// Head slot index in `table` of outgoing transitions from this state, or `NO_EDGE`.
    first_edge: u32,
};

/// Open-addressing hash slot for transition `(state, token) -> next_state`
/// threaded into `state`'s outgoing edge list (16 bytes, zero padding).
/// Transitions are never deleted between resets, so slot indices remain stable
/// for `first_edge` / `next_edge` links.
pub const TransitionSlot = extern struct {
    state: u32,
    token: u32,
    next_state: u32,
    next_edge: u32,
};

comptime {
    if (@sizeOf(State) != 16) @compileError("State must be 16 bytes");
    if (@sizeOf(TransitionSlot) != 16) @compileError("TransitionSlot must be 16 bytes");
}

/// Result of querying the L1 suffix automaton for a draft continuation.
pub const QueryResult = struct {
    /// Length of the longest matched suffix that has an earlier occurrence.
    match_len: u32,
    /// Number of continuation tokens written to `out_tokens`.
    draft_len: u32,
    /// True when `match_len >= SEED_MIN_MATCH_LEN` (4 tokens, ADR-0005).
    is_seed: bool,
};

pub const SuffixAutomaton = struct {
    states: []State,
    table: []TransitionSlot,
    table_mask: usize,
    history: []u32,
    max_tokens: u32,
    count: u32,
    state_count: u32,
    transition_count: u32,
    last: u32,
    owned_backing: ?[]align(backing_alignment) u8,

    /// Required byte alignment of the backing buffer passed to `initFromBuffer`.
    pub const backing_alignment: usize = @alignOf(State);

    /// Maximum number of states for `max_tokens` appended tokens (`<= 2n + 1`).
    pub fn maxStates(max_tokens: u32) LayoutError!usize {
        if (max_tokens == 0 or max_tokens > MAX_SUPPORTED_TOKENS) {
            return error.InvalidCapacity;
        }
        return @as(usize, max_tokens) * 2 + 1;
    }

    /// Upper bound on transitions for `max_tokens` appended tokens (`<= 3n + 4`).
    pub fn maxTransitions(max_tokens: u32) LayoutError!usize {
        if (max_tokens == 0 or max_tokens > MAX_SUPPORTED_TOKENS) {
            return error.InvalidCapacity;
        }
        return @as(usize, max_tokens) * 3 + 4;
    }

    /// Power-of-two hash table slot count guaranteeing load factor `<= 0.5`.
    pub fn tableCapacity(max_tokens: u32) LayoutError!usize {
        const max_edges = try maxTransitions(max_tokens);
        const min_slots = try std.math.mul(usize, max_edges, 2);
        const clamped = @max(@as(usize, 16), min_slots);
        return std.math.ceilPowerOfTwo(usize, clamped) catch return error.Overflow;
    }

    /// Exact backing buffer size in bytes needed by `initFromBuffer(buf, max_tokens)`.
    pub fn requiredBytes(max_tokens: u32) LayoutError!usize {
        const n_states = try maxStates(max_tokens);
        const n_slots = try tableCapacity(max_tokens);
        const states_bytes = std.math.mul(usize, n_states, @sizeOf(State)) catch return error.Overflow;
        const table_bytes = std.math.mul(usize, n_slots, @sizeOf(TransitionSlot)) catch return error.Overflow;
        const history_bytes = std.math.mul(usize, @as(usize, max_tokens), @sizeOf(u32)) catch return error.Overflow;
        const s1 = std.math.add(usize, states_bytes, table_bytes) catch return error.Overflow;
        return std.math.add(usize, s1, history_bytes) catch return error.Overflow;
    }

    /// Initializes a zero-allocation `SuffixAutomaton` over a caller-provided
    /// aligned byte slice (e.g., carved from `WiredArena.allocAligned`).
    pub fn initFromBuffer(buf: []u8, max_tokens: u32) InitError!SuffixAutomaton {
        const req = try requiredBytes(max_tokens);
        if (!std.mem.isAligned(@intFromPtr(buf.ptr), backing_alignment)) {
            return error.UnalignedBacking;
        }
        if (buf.len < req) return error.BufferTooSmall;

        const n_states = try maxStates(max_tokens);
        const n_slots = try tableCapacity(max_tokens);
        const states_bytes = n_states * @sizeOf(State);
        const table_bytes = n_slots * @sizeOf(TransitionSlot);
        const history_bytes = @as(usize, max_tokens) * @sizeOf(u32);

        const states_raw: []align(@alignOf(State)) u8 = @alignCast(buf[0..states_bytes]);
        const table_raw: []align(@alignOf(TransitionSlot)) u8 = @alignCast(buf[states_bytes .. states_bytes + table_bytes]);
        const history_raw: []align(@alignOf(u32)) u8 = @alignCast(buf[states_bytes + table_bytes .. states_bytes + table_bytes + history_bytes]);

        var self = SuffixAutomaton{
            .states = std.mem.bytesAsSlice(State, states_raw),
            .table = std.mem.bytesAsSlice(TransitionSlot, table_raw),
            .table_mask = n_slots - 1,
            .history = std.mem.bytesAsSlice(u32, history_raw),
            .max_tokens = max_tokens,
            .count = 0,
            .state_count = 0,
            .transition_count = 0,
            .last = 0,
            .owned_backing = null,
        };
        self.reset();
        return self;
    }

    /// Convenience allocator-backed initializer for tests and offline tools.
    /// Runtime hot paths should carve a buffer from `WiredArena` and call
    /// `initFromBuffer`.
    pub fn initAlloc(
        allocator: std.mem.Allocator,
        max_tokens: u32,
    ) (InitError || error{OutOfMemory})!SuffixAutomaton {
        const req = try requiredBytes(max_tokens);
        const backing = try allocator.alignedAlloc(u8, .fromByteUnits(backing_alignment), req);
        errdefer allocator.free(backing);
        var self = try initFromBuffer(backing, max_tokens);
        self.owned_backing = backing;
        return self;
    }

    /// Releases the backing allocation if created via `initAlloc`. No-op for
    /// buffer-backed instances initialized via `initFromBuffer`.
    pub fn deinit(self: *SuffixAutomaton, allocator: std.mem.Allocator) void {
        if (self.owned_backing) |backing| {
            allocator.free(backing);
            self.owned_backing = null;
        }
        self.states = &.{};
        self.table = &.{};
        self.history = &.{};
        self.count = 0;
        self.state_count = 0;
        self.transition_count = 0;
        self.last = 0;
    }

    /// Resets the automaton to the empty state without reallocating memory.
    pub fn reset(self: *SuffixAutomaton) void {
        // Fill transition table bytes with 0xFF so every `state` is `EMPTY_STATE`
        // and every `next_edge` is `NO_EDGE`.
        @memset(std.mem.sliceAsBytes(self.table), 0xFF);
        self.states[0] = .{
            .len = 0,
            .link = -1,
            .first_end_pos = 0,
            .first_edge = NO_EDGE,
        };
        self.count = 0;
        self.state_count = 1;
        self.transition_count = 0;
        self.last = 0;
    }

    pub fn len(self: *const SuffixAutomaton) u32 {
        return self.count;
    }

    pub fn capacity(self: *const SuffixAutomaton) u32 {
        return self.max_tokens;
    }

    pub fn remainingCapacity(self: *const SuffixAutomaton) u32 {
        return self.max_tokens - self.count;
    }

    pub fn historySlice(self: *const SuffixAutomaton) []const u32 {
        return self.history[0..self.count];
    }

    /// Appends one token to the suffix automaton in amortized O(1) time with
    /// zero dynamic allocation.
    pub fn append(self: *SuffixAutomaton, token_id: u32) AppendError!void {
        if (self.count >= self.max_tokens) return error.CapacityExceeded;

        const pos = self.count;
        self.history[pos] = token_id;
        self.count = pos + 1;

        const cur = self.state_count;
        std.debug.assert(cur < self.states.len);
        self.state_count = cur + 1;
        self.states[cur] = .{
            .len = self.states[self.last].len + 1,
            .link = 0,
            .first_end_pos = pos,
            .first_edge = NO_EDGE,
        };

        var p: i32 = @intCast(self.last);
        while (p != -1) {
            const pu: u32 = @intCast(p);
            if (self.findSlot(pu, token_id) != null) break;
            self.insertTransition(pu, token_id, cur);
            p = self.states[pu].link;
        }

        if (p == -1) {
            self.states[cur].link = 0;
        } else {
            const pu: u32 = @intCast(p);
            const q_slot = self.findSlot(pu, token_id) orelse unreachable;
            const q = self.table[q_slot].next_state;
            if (self.states[pu].len + 1 == self.states[q].len) {
                self.states[cur].link = @intCast(q);
            } else {
                const clone = self.state_count;
                std.debug.assert(clone < self.states.len);
                self.state_count = clone + 1;
                self.states[clone] = .{
                    .len = self.states[pu].len + 1,
                    .link = self.states[q].link,
                    .first_end_pos = self.states[q].first_end_pos,
                    .first_edge = NO_EDGE,
                };

                // Clone outgoing transitions of `q` in O(outdeg(q)).
                var edge = self.states[q].first_edge;
                while (edge != NO_EDGE) {
                    const slot = self.table[edge];
                    self.insertTransition(clone, slot.token, slot.next_state);
                    edge = slot.next_edge;
                }

                // Redirect `(r, token_id) -> q` transitions along suffix links to `clone`.
                var r: i32 = p;
                while (r != -1) {
                    const ru: u32 = @intCast(r);
                    const r_slot = self.findSlot(ru, token_id) orelse break;
                    if (self.table[r_slot].next_state != q) break;
                    self.table[r_slot].next_state = clone;
                    r = self.states[ru].link;
                }

                self.states[q].link = @intCast(clone);
                self.states[cur].link = @intCast(clone);
            }
        }

        self.last = cur;
    }

    /// Appends a slice of tokens atomically with respect to capacity bounds.
    pub fn appendSlice(self: *SuffixAutomaton, tokens: []const u32) AppendError!void {
        if (tokens.len > self.remainingCapacity()) return error.CapacityExceeded;
        for (tokens) |tok| {
            try self.append(tok);
        }
    }

    /// Queries the longest suffix of the current appended history that also
    /// occurred earlier, copying up to `min(max_draft, out_tokens.len)` tokens
    /// following that earlier occurrence into `out_tokens`.
    pub fn queryActive(
        self: *const SuffixAutomaton,
        max_draft: usize,
        out_tokens: []u32,
    ) QueryResult {
        if (self.count == 0 or self.last == 0) {
            return .{ .match_len = 0, .draft_len = 0, .is_seed = false };
        }
        return self.extractContinuation(self.last, self.states[self.last].len, max_draft, out_tokens);
    }

    /// Matches the longest suffix of `tail_tokens` present in `history` with a
    /// valid continuation, copying up to `min(max_draft, out_tokens.len)` tokens
    /// following its earliest occurrence into `out_tokens`.
    pub fn querySuffix(
        self: *const SuffixAutomaton,
        tail_tokens: []const u32,
        max_draft: usize,
        out_tokens: []u32,
    ) QueryResult {
        if (self.count == 0 or tail_tokens.len == 0) {
            return .{ .match_len = 0, .draft_len = 0, .is_seed = false };
        }

        var state: u32 = 0;
        var match_len: u32 = 0;

        for (tail_tokens) |tok| {
            if (self.getTransition(state, tok)) |next| {
                state = next;
                match_len += 1;
            } else {
                var p: i32 = self.states[state].link;
                while (p != -1) {
                    const pu: u32 = @intCast(p);
                    if (self.getTransition(pu, tok)) |next| {
                        state = next;
                        match_len = self.states[pu].len + 1;
                        break;
                    }
                    p = self.states[pu].link;
                } else {
                    state = 0;
                    match_len = 0;
                }
            }
        }

        return self.extractContinuation(state, match_len, max_draft, out_tokens);
    }

    fn extractContinuation(
        self: *const SuffixAutomaton,
        start_state: u32,
        start_match_len: u32,
        max_draft: usize,
        out_tokens: []u32,
    ) QueryResult {
        if (self.count == 0 or start_state == 0 or start_match_len == 0) {
            return .{ .match_len = 0, .draft_len = 0, .is_seed = false };
        }

        var state: u32 = start_state;
        var match_len: u32 = start_match_len;

        // When the matched state only occurred at the very end of `history`
        // (`first_end_pos + 1 >= count`), walk the suffix link so we reach the
        // longest suffix that also has an earlier occurrence with a continuation.
        while (state != 0 and self.states[state].first_end_pos + 1 >= self.count) {
            const link = self.states[state].link;
            if (link <= 0) {
                return .{ .match_len = 0, .draft_len = 0, .is_seed = false };
            }
            state = @intCast(link);
            match_len = @min(match_len, self.states[state].len);
        }

        if (state == 0 or match_len == 0) {
            return .{ .match_len = 0, .draft_len = 0, .is_seed = false };
        }

        const start_pos: usize = self.states[state].first_end_pos + 1;
        const avail: usize = self.count - start_pos;
        const draft_len: usize = @min(@min(avail, max_draft), out_tokens.len);
        if (draft_len > 0) {
            @memcpy(out_tokens[0..draft_len], self.history[start_pos .. start_pos + draft_len]);
        }

        return .{
            .match_len = match_len,
            .draft_len = @intCast(draft_len),
            .is_seed = match_len >= SEED_MIN_MATCH_LEN,
        };
    }

    inline fn hashKey(state: u32, token: u32) usize {
        var x: u64 = (@as(u64, state) << 32) | @as(u64, token);
        // SplitMix64 finalizer for low probe variance across sequential token/state IDs.
        x = (x ^ (x >> 30)) *% 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) *% 0x94d049bb133111eb;
        x = x ^ (x >> 31);
        return @truncate(x);
    }

    inline fn findSlot(self: *const SuffixAutomaton, state: u32, token: u32) ?usize {
        var idx = hashKey(state, token) & self.table_mask;
        while (true) {
            const slot = self.table[idx];
            if (slot.state == EMPTY_STATE) return null;
            if (slot.state == state and slot.token == token) return idx;
            idx = (idx + 1) & self.table_mask;
        }
    }

    inline fn getTransition(self: *const SuffixAutomaton, state: u32, token: u32) ?u32 {
        const idx = self.findSlot(state, token) orelse return null;
        return self.table[idx].next_state;
    }

    inline fn insertTransition(self: *SuffixAutomaton, state: u32, token: u32, next_state: u32) void {
        var idx = hashKey(state, token) & self.table_mask;
        while (self.table[idx].state != EMPTY_STATE) {
            std.debug.assert(!(self.table[idx].state == state and self.table[idx].token == token));
            idx = (idx + 1) & self.table_mask;
        }
        self.table[idx] = .{
            .state = state,
            .token = token,
            .next_state = next_state,
            .next_edge = self.states[state].first_edge,
        };
        self.states[state].first_edge = @intCast(idx);
        self.transition_count += 1;
        std.debug.assert(self.transition_count * 2 <= self.table.len);
    }
};

test "SuffixAutomaton validates capacity, alignment, and buffer sizing" {
    try std.testing.expectError(error.InvalidCapacity, SuffixAutomaton.requiredBytes(0));
    try std.testing.expectError(error.InvalidCapacity, SuffixAutomaton.requiredBytes(MAX_SUPPORTED_TOKENS + 1));

    const req = try SuffixAutomaton.requiredBytes(64);
    try std.testing.expect(req > 0);
    const tcap = try SuffixAutomaton.tableCapacity(64);
    try std.testing.expect(std.math.isPowerOfTwo(tcap));
    try std.testing.expect((try SuffixAutomaton.maxTransitions(64)) * 2 <= tcap);

    const backing = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(16), req + 4);
    defer std.testing.allocator.free(backing);

    try std.testing.expectError(error.UnalignedBacking, SuffixAutomaton.initFromBuffer(backing[1 .. req + 1], 64));
    try std.testing.expectError(error.BufferTooSmall, SuffixAutomaton.initFromBuffer(backing[0 .. req - 1], 64));

    var sam = try SuffixAutomaton.initFromBuffer(backing[0..req], 64);
    try std.testing.expectEqual(@as(u32, 0), sam.len());
    try std.testing.expectEqual(@as(u32, 64), sam.capacity());
    try std.testing.expectEqual(@as(u32, 64), sam.remainingCapacity());
}

test "SuffixAutomaton carves cleanly from WiredArena with zero dynamic heap allocation" {
    const max_toks: u32 = 1024;
    const req = try SuffixAutomaton.requiredBytes(max_toks);
    const cap = try page.alignUp(req);
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = cap * 4,
        .user_wire_limit = cap * 2,
        .global_user_wire_limit = cap * 2,
        .gpu_working_set_bytes = cap * 2,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(cap, budget);
    defer arena.deinit();

    const buf = try arena.allocAligned(req, SuffixAutomaton.backing_alignment);
    var sam = try SuffixAutomaton.initFromBuffer(buf, max_toks);
    try sam.appendSlice(&[_]u32{ 10, 20, 30, 40, 50, 60, 10, 20, 30, 40 });

    var out: [8]u32 = undefined;
    const res = sam.queryActive(8, &out);
    try std.testing.expectEqual(@as(u32, 4), res.match_len);
    try std.testing.expect(res.is_seed);
    try std.testing.expectEqual(@as(u32, 6), res.draft_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 50, 60, 10, 20, 30, 40 }, out[0..res.draft_len]);
}

test "SuffixAutomaton matches repetitions, seed threshold, and suffix link fallback at end of history" {
    var sam = try SuffixAutomaton.initAlloc(std.testing.allocator, 256);
    defer sam.deinit(std.testing.allocator);

    var out: [16]u32 = undefined;
    // Empty automaton returns no match.
    const empty_res = sam.queryActive(8, &out);
    try std.testing.expectEqual(@as(u32, 0), empty_res.match_len);
    try std.testing.expectEqual(@as(u32, 0), empty_res.draft_len);
    try std.testing.expect(!empty_res.is_seed);

    // Append a 6-token phrase followed by a distinct separator and partial repetitions.
    try sam.appendSlice(&[_]u32{ 1, 2, 3, 4, 5, 6, 99 });
    // Last token 99 has never appeared before -> no match.
    const no_match = sam.queryActive(8, &out);
    try std.testing.expectEqual(@as(u32, 0), no_match.match_len);
    try std.testing.expectEqual(@as(u32, 0), no_match.draft_len);
    try std.testing.expect(!no_match.is_seed);

    // Append 1, 2, 3 -> match_len = 3 (< 4 -> branch, not seed), continuation starts at 4.
    try sam.appendSlice(&[_]u32{ 1, 2, 3 });
    const branch = sam.queryActive(4, &out);
    try std.testing.expectEqual(@as(u32, 3), branch.match_len);
    try std.testing.expect(!branch.is_seed);
    try std.testing.expectEqual(@as(u32, 4), branch.draft_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 4, 5, 6, 99 }, out[0..branch.draft_len]);

    // Append 4 -> match_len = 4 (>= 4 -> seed!), continuation starts at 5.
    try sam.append(4);
    const seed = sam.queryActive(4, &out);
    try std.testing.expectEqual(@as(u32, 4), seed.match_len);
    try std.testing.expect(seed.is_seed);
    try std.testing.expectEqual(@as(u32, 4), seed.draft_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 5, 6, 99, 1 }, out[0..seed.draft_len]);

    // querySuffix with a tail whose full match `[99, 1, 2, 3, 4]` only ends at
    // `count - 1` must walk the suffix link once to `[1, 2, 3, 4]` so a valid
    // continuation `[5, 6, ...]` is still returned.
    const suffix_end = sam.querySuffix(&[_]u32{ 99, 1, 2, 3, 4 }, 3, &out);
    try std.testing.expectEqual(@as(u32, 4), suffix_end.match_len);
    try std.testing.expect(suffix_end.is_seed);
    try std.testing.expectEqual(@as(u32, 3), suffix_end.draft_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 5, 6, 99 }, out[0..suffix_end.draft_len]);

    // querySuffix with an unseen prefix `[777, 888, 2, 3]` falls back to suffix `[2, 3]`.
    const suffix_partial = sam.querySuffix(&[_]u32{ 777, 888, 2, 3 }, 3, &out);
    try std.testing.expectEqual(@as(u32, 2), suffix_partial.match_len);
    try std.testing.expect(!suffix_partial.is_seed);
    try std.testing.expectEqual(@as(u32, 3), suffix_partial.draft_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 4, 5, 6 }, out[0..suffix_partial.draft_len]);
}

test "SuffixAutomaton handles code-block repetition, capacity bounds, and reset reuse" {
    const cap: u32 = 128;
    var sam = try SuffixAutomaton.initAlloc(std.testing.allocator, cap);
    defer sam.deinit(std.testing.allocator);

    // Simulate a repeated function header / loop boilerplate block.
    const block = [_]u32{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111 };
    try sam.appendSlice(&block);
    // Insert unique body tokens.
    try sam.appendSlice(&[_]u32{ 200, 201, 202, 203, 204 });
    // Start repeating the header.
    try sam.appendSlice(block[0..5]);

    var out: [12]u32 = undefined;
    const res = sam.queryActive(8, &out);
    try std.testing.expectEqual(@as(u32, 5), res.match_len);
    try std.testing.expect(res.is_seed);
    try std.testing.expectEqual(@as(u32, 8), res.draft_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 105, 106, 107, 108, 109, 110, 111, 200 }, out[0..res.draft_len]);

    // Stress worst-case state/transition growth (e.g. a b b b ... b c) up to exact capacity.
    sam.reset();
    try std.testing.expectEqual(@as(u32, 0), sam.len());
    try std.testing.expectEqual(@as(u32, 1), sam.state_count);
    try std.testing.expectEqual(@as(u32, 0), sam.transition_count);

    // Classic worst-case SAM string: `0, 1, 1, ..., 1, 2` reaches 2n - 1 states and 3n - 4 edges.
    try sam.append(0);
    for (1..cap - 1) |_| try sam.append(1);
    try sam.append(2);
    try std.testing.expectEqual(cap, sam.len());
    try std.testing.expect(sam.state_count <= 2 * cap - 1);
    try std.testing.expect(sam.transition_count <= 3 * cap - 4);
    try std.testing.expect(sam.transition_count * 2 <= sam.table.len);

    // Exceeding capacity fails cleanly without mutating count.
    try std.testing.expectError(error.CapacityExceeded, sam.append(3));
    try std.testing.expectError(error.CapacityExceeded, sam.appendSlice(&[_]u32{ 3, 4 }));
    try std.testing.expectEqual(cap, sam.len());

    // Reset allows clean reuse without stale transitions from the previous sequence.
    sam.reset();
    try sam.appendSlice(&[_]u32{ 42, 43, 44, 45, 46, 42, 43, 44, 45 });
    const reused = sam.queryActive(4, &out);
    try std.testing.expectEqual(@as(u32, 4), reused.match_len);
    try std.testing.expect(reused.is_seed);
    try std.testing.expectEqual(@as(u32, 4), reused.draft_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 46, 42, 43, 44 }, out[0..reused.draft_len]);
    // Old tokens from before reset must not match.
    const stale = sam.querySuffix(&[_]u32{ 0, 1, 1, 1 }, 4, &out);
    try std.testing.expectEqual(@as(u32, 0), stale.match_len);
    try std.testing.expectEqual(@as(u32, 0), stale.draft_len);
}

test "SuffixAutomaton query latency stays well below 50 us at 32K tokens" {
    const n_tokens: u32 = 32_768;
    var sam = try SuffixAutomaton.initAlloc(std.testing.allocator, n_tokens);
    defer sam.deinit(std.testing.allocator);

    // Populate 32K tokens with mixed pseudo-random content and periodic repeated
    // code blocks so clone paths and deep suffix links are heavily exercised.
    var prng = std.Random.DefaultPrng.init(0x5A11_C0DE);
    const rng = prng.random();
    const motif = [_]u32{ 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012 };

    var written: u32 = 0;
    while (written < n_tokens - 16) {
        if ((written % 512) == 0 and written + motif.len <= n_tokens - 16) {
            try sam.appendSlice(&motif);
            written += @intCast(motif.len);
        } else {
            try sam.append(rng.uintLessThan(u32, 4096));
            written += 1;
        }
    }
    // End with a prefix of `motif` so active query finds a strong >=4-token seed.
    try sam.appendSlice(motif[0..6]);
    try std.testing.expect(sam.state_count <= 2 * sam.len() + 1);
    try std.testing.expect(sam.transition_count * 2 <= sam.table.len);

    var out: [16]u32 = undefined;
    const check = sam.queryActive(8, &out);
    try std.testing.expect(check.match_len >= 6);
    try std.testing.expect(check.is_seed);
    try std.testing.expectEqualSlices(u32, motif[6..12], out[0..6]);

    // Measure 1,000 active queries and 1,000 32-token suffix queries at 32K context.
    const iters: usize = 1000;
    var sink: u64 = 0;

    const t0 = try vmstat.nowNs();
    for (0..iters) |_| {
        const r = sam.queryActive(16, &out);
        sink +%= r.match_len +% r.draft_len +% out[0];
    }
    const t1 = try vmstat.nowNs();
    const active_ns_per_query = (t1 - t0) / iters;

    const tail = sam.historySlice()[sam.len() - 32 .. sam.len()];
    const t2 = try vmstat.nowNs();
    for (0..iters) |_| {
        const r = sam.querySuffix(tail, 16, &out);
        sink +%= r.match_len +% r.draft_len +% out[0];
    }
    const t3 = try vmstat.nowNs();
    const suffix_ns_per_query = (t3 - t2) / iters;

    std.mem.doNotOptimizeAway(sink);

    // Charter target: < 50 us (50,000 ns) per query at 32K tokens.
    try std.testing.expect(active_ns_per_query < 50_000);
    try std.testing.expect(suffix_ns_per_query < 50_000);
}
