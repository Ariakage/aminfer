//! Preallocated grammar, JSON Schema (`strict: true`), tool-call, and GBNF
//! token-bitmask engine (capability C22, Milestone M7.5, ADR-0022).
//!
//! Compiles `json_object`, `json_schema` (objects with required/optional keys,
//! strings with escapes and valid UTF-8, numbers, integers, booleans, null,
//! arrays, enums, const, anyOf/oneOf, `$defs`/`$ref`, nested schemas, and strict
//! OpenAI/Anthropic tool-call envelopes), and GBNF grammars at request admission
//! into a preallocated pushdown state machine (`GrammarProgram`) and evaluates
//! `u32` allowed-token bitmasks (`vocab_size` up to `262,144`) on P-cores with:
//! - zero per-token dynamic heap allocation;
//! - O(1) `GrammarState` value snapshot/restore for `G5`/`D9` speculative trees;
//! - strict zero-overhead bypass when `grammar == null`.

const std = @import("std");

/// Maximum supported vocabulary size (`262,144` >= Qwen3.8 `248,077`).
pub const MAX_VOCAB_SIZE: usize = 262_144;
/// Number of `u32` words needed for `MAX_VOCAB_SIZE` (`8,192` words = `32 KiB`).
pub const MAX_BITMASK_WORDS: usize = MAX_VOCAB_SIZE / 32;

pub const MAX_RULES: usize = 256;
pub const MAX_INSTS: usize = 4096;
pub const MAX_CHARSETS: usize = 160;
pub const MAX_STACK_DEPTH: usize = 40;
pub const MAX_ACTIVE_STACKS: usize = 16;
pub const MAX_SCHEMA_DEPTH: usize = 32;
pub const MAX_OBJECT_PROPS: usize = 48;
pub const MAX_DEFS: usize = 48;
pub const MAX_GBNF_RULE_NAME: usize = 64;
pub const MAX_EOS_TOKENS: usize = 16;

const ACCEPT_PC: u16 = std.math.maxInt(u16);
const INVALID_RULE: u16 = std.math.maxInt(u16);

pub const CompileError = error{
    EmptyGrammar,
    InvalidSyntax,
    InvalidJsonSchema,
    UnsupportedSchemaKeyword,
    UnknownRuleReference,
    UnknownSchemaRef,
    ProgramCapacityExceeded,
    CharsetCapacityExceeded,
    RuleCapacityExceeded,
    NestingDepthExceeded,
    TooManyProperties,
};

pub const VocabError = error{
    EmptyVocab,
    VocabTooLarge,
    OutOfMemory,
};

/// 256-bit byte-set (`[4]u64` = 32 bytes) testing byte membership in O(1).
pub const ByteSet = struct {
    words: [4]u64 = .{ 0, 0, 0, 0 },

    pub const empty: ByteSet = .{ .words = .{ 0, 0, 0, 0 } };
    pub const all: ByteSet = .{ .words = .{
        std.math.maxInt(u64),
        std.math.maxInt(u64),
        std.math.maxInt(u64),
        std.math.maxInt(u64),
    } };

    pub inline fn insert(self: *ByteSet, b: u8) void {
        self.words[b >> 6] |= (@as(u64, 1) << @intCast(b & 63));
    }

    pub inline fn remove(self: *ByteSet, b: u8) void {
        self.words[b >> 6] &= ~(@as(u64, 1) << @intCast(b & 63));
    }

    pub inline fn insertRange(self: *ByteSet, lo: u8, hi: u8) void {
        if (lo > hi) return;
        var b: u16 = lo;
        while (b <= hi) : (b += 1) {
            self.insert(@intCast(b));
        }
    }

    pub inline fn contains(self: ByteSet, b: u8) bool {
        return ((self.words[b >> 6] >> @intCast(b & 63)) & 1) != 0;
    }

    pub inline fn unionWith(self: *ByteSet, other: ByteSet) void {
        self.words[0] |= other.words[0];
        self.words[1] |= other.words[1];
        self.words[2] |= other.words[2];
        self.words[3] |= other.words[3];
    }

    pub inline fn invert(self: *ByteSet) void {
        self.words[0] = ~self.words[0];
        self.words[1] = ~self.words[1];
        self.words[2] = ~self.words[2];
        self.words[3] = ~self.words[3];
    }

    pub inline fn isEmpty(self: ByteSet) bool {
        return (self.words[0] | self.words[1] | self.words[2] | self.words[3]) == 0;
    }

    pub inline fn eql(a: ByteSet, b: ByteSet) bool {
        return a.words[0] == b.words[0] and
            a.words[1] == b.words[1] and
            a.words[2] == b.words[2] and
            a.words[3] == b.words[3];
    }
};

pub inline fn isPlainAsciiStringByte(b: u8) bool {
    return b >= 0x20 and b <= 0x7E and b != '"' and b != '\\';
}

const PLAIN_ASCII_STRING_SET: ByteSet = blk: {
    var s = ByteSet.empty;
    var b: u8 = 0x20;
    while (b <= 0x7E) : (b += 1) {
        if (b != '"' and b != '\\') s.insert(b);
    }
    break :blk s;
};

// ---------------------------------------------------------------------------
// Bitmask utilities (`u32` words up to `MAX_VOCAB_SIZE = 262,144`)
// ---------------------------------------------------------------------------

pub inline fn bitmaskWordCount(vocab_size: usize) usize {
    return (vocab_size + 31) / 32;
}

pub inline fn isTokenAllowed(mask: []const u32, token_id: u32) bool {
    const word_idx: usize = @as(usize, token_id) >> 5;
    if (word_idx >= mask.len) return false;
    const bit: u5 = @intCast(token_id & 31);
    return ((mask[word_idx] >> bit) & 1) != 0;
}

pub inline fn setTokenAllowed(mask: []u32, token_id: u32) void {
    const word_idx: usize = @as(usize, token_id) >> 5;
    if (word_idx < mask.len) {
        const bit: u5 = @intCast(token_id & 31);
        mask[word_idx] |= (@as(u32, 1) << bit);
    }
}

pub inline fn clearTokenAllowed(mask: []u32, token_id: u32) void {
    const word_idx: usize = @as(usize, token_id) >> 5;
    if (word_idx < mask.len) {
        const bit: u5 = @intCast(token_id & 31);
        mask[word_idx] &= ~(@as(u32, 1) << bit);
    }
}

pub fn clearBitmask(mask: []u32, vocab_size: usize) void {
    const words = @min(mask.len, bitmaskWordCount(vocab_size));
    @memset(mask[0..words], 0);
}

pub fn allowAllTokens(mask: []u32, vocab_size: usize) void {
    const words = bitmaskWordCount(vocab_size);
    if (words == 0 or mask.len < words) return;
    @memset(mask[0..words], std.math.maxInt(u32));
    const rem: u5 = @intCast(vocab_size & 31);
    if (rem != 0) {
        mask[words - 1] = (@as(u32, 1) << rem) - 1;
    }
}

pub fn countAllowedTokens(mask: []const u32, vocab_size: usize) usize {
    const words = @min(mask.len, bitmaskWordCount(vocab_size));
    var total: usize = 0;
    for (mask[0..words], 0..) |w, idx| {
        var masked = w;
        if (idx + 1 == words) {
            const rem: u5 = @intCast(vocab_size & 31);
            if (rem != 0) masked &= (@as(u32, 1) << rem) - 1;
        }
        total += @popCount(masked);
    }
    return total;
}

// ---------------------------------------------------------------------------
// Pushdown Automaton IR & O(1) Snapshot/Restore Runtime
// ---------------------------------------------------------------------------

pub const InstKind = enum(u8) {
    /// End of rule; pops return address off PDA stack (or accepts EOF if depth == 0).
    end = 0,
    /// Terminal matching exact byte `arg` (`0..255`), advancing to `next`.
    byte = 1,
    /// Terminal matching `charsets[arg]`, advancing to `next`.
    charset = 2,
    /// Pushes `next` (unless `insts[next].kind == .end`, tail-call optimized) and jumps to `rule_entries[arg]`.
    rule_ref = 3,
    /// Epsilon split exploring both `arg` and `next`.
    branch = 4,
    /// Unconditional epsilon jump to `arg`.
    jump = 5,
};

pub const Inst = struct {
    kind: InstKind = .end,
    /// True on `.charset` instructions that loop back to themselves on every plain ASCII string byte.
    plain_ascii_self_loop: bool = false,
    arg: u16 = 0,
    next: u16 = 0,
};

pub const Stack = struct {
    frames: [MAX_STACK_DEPTH]u16 = undefined,
    depth: u8 = 0,
    pc: u16 = 0,

    fn eql(a: *const Stack, b: *const Stack) bool {
        if (a.pc != b.pc or a.depth != b.depth) return false;
        return std.mem.eql(u16, a.frames[0..a.depth], b.frames[0..b.depth]);
    }
};

/// Value-type PDA runtime state: zero pointers, zero heap allocations.
/// Copying `GrammarState` via `snapshot()` / `restore()` is O(1).
pub const GrammarState = struct {
    stacks: [MAX_ACTIVE_STACKS]Stack = undefined,
    stack_count: u8 = 0,
    can_accept_eof: bool = false,
    has_plain_ascii_self_loop: bool = false,
    allowed_bytes: ByteSet = ByteSet.empty,

    pub inline fn snapshot(self: *const GrammarState) GrammarState {
        return self.*;
    }

    pub inline fn restore(self: *GrammarState, snap: GrammarState) void {
        self.* = snap;
    }

    pub inline fn isAccepting(self: *const GrammarState) bool {
        return self.can_accept_eof;
    }

    pub inline fn isRejected(self: *const GrammarState) bool {
        return self.stack_count == 0 and !self.can_accept_eof;
    }
};

/// Preallocated compiled grammar program.
pub const GrammarProgram = struct {
    insts: [MAX_INSTS]Inst = undefined,
    inst_count: u16 = 0,
    charsets: [MAX_CHARSETS]ByteSet = undefined,
    charset_count: u16 = 0,
    rule_entries: [MAX_RULES]u16 = @splat(INVALID_RULE),
    rule_count: u16 = 0,
    root_rule: u16 = 0,

    pub fn reset(self: *GrammarProgram) void {
        self.inst_count = 0;
        self.charset_count = 0;
        self.rule_entries = @splat(INVALID_RULE);
        self.rule_count = 0;
        self.root_rule = 0;
    }

    fn allocRule(self: *GrammarProgram) CompileError!u16 {
        if (self.rule_count >= MAX_RULES) return error.RuleCapacityExceeded;
        const r = self.rule_count;
        self.rule_count += 1;
        return r;
    }

    fn emitInst(self: *GrammarProgram, inst: Inst) CompileError!u16 {
        if (self.inst_count >= MAX_INSTS) return error.ProgramCapacityExceeded;
        const idx = self.inst_count;
        self.insts[idx] = inst;
        self.inst_count += 1;
        return idx;
    }

    fn internCharset(self: *GrammarProgram, cs: ByteSet) CompileError!u16 {
        for (self.charsets[0..self.charset_count], 0..) |existing, idx| {
            if (existing.eql(cs)) return @intCast(idx);
        }
        if (self.charset_count >= MAX_CHARSETS) return error.CharsetCapacityExceeded;
        const idx = self.charset_count;
        self.charsets[idx] = cs;
        self.charset_count += 1;
        return idx;
    }

    /// Marks `.charset` instructions that include all plain ASCII string bytes and
    /// immediately epsilon-reach themselves, enabling O(vocab/32) bitmask bulk OR.
    fn finalizeSelfLoops(self: *GrammarProgram) void {
        for (0..self.inst_count) |idx| {
            const inst = &self.insts[idx];
            if (inst.kind != .charset) continue;
            const cs = self.charsets[inst.arg];
            var covers_plain = true;
            for (0..4) |w| {
                if ((cs.words[w] & PLAIN_ASCII_STRING_SET.words[w]) != PLAIN_ASCII_STRING_SET.words[w]) {
                    covers_plain = false;
                    break;
                }
            }
            if (!covers_plain) continue;
            if (self.canEpsilonReach(inst.next, @intCast(idx))) {
                inst.plain_ascii_self_loop = true;
            }
        }
    }

    fn canEpsilonReach(self: *const GrammarProgram, start_pc: u16, target_pc: u16) bool {
        var visited: [MAX_INSTS / 64]u64 = @splat(0);
        var work: [32]u16 = undefined;
        var sp: usize = 0;
        work[0] = start_pc;
        sp = 1;
        while (sp > 0) {
            sp -= 1;
            const pc = work[sp];
            if (pc == target_pc) return true;
            if (pc >= self.inst_count) continue;
            const w = pc >> 6;
            const b: u6 = @intCast(pc & 63);
            if (((visited[w] >> b) & 1) != 0) continue;
            visited[w] |= (@as(u64, 1) << b);

            const inst = self.insts[pc];
            switch (inst.kind) {
                .jump => {
                    if (sp < work.len) {
                        work[sp] = inst.arg;
                        sp += 1;
                    }
                },
                .branch => {
                    if (sp + 2 <= work.len) {
                        work[sp] = inst.arg;
                        work[sp + 1] = inst.next;
                        sp += 2;
                    }
                },
                else => {},
            }
        }
        return false;
    }

    /// Initializes the starting `GrammarState` at `root_rule` with zero heap allocation.
    pub fn initialState(self: *const GrammarProgram) GrammarState {
        var out = GrammarState{};
        if (self.rule_count == 0 or self.root_rule >= self.rule_count) return out;
        const entry = self.rule_entries[self.root_rule];
        if (entry == INVALID_RULE) return out;

        var seed = Stack{
            .depth = 0,
            .pc = entry,
        };
        self.expandEpsilon(&seed, &out);
        return out;
    }

    fn pushTerminalStack(self: *const GrammarProgram, st: *const Stack, out: *GrammarState) void {
        if (st.pc == ACCEPT_PC) {
            out.can_accept_eof = true;
            return;
        }
        for (out.stacks[0..out.stack_count]) |*existing| {
            if (existing.eql(st)) return;
        }
        if (out.stack_count >= MAX_ACTIVE_STACKS) return;

        const dst = &out.stacks[out.stack_count];
        dst.depth = st.depth;
        dst.pc = st.pc;
        if (st.depth > 0) {
            @memcpy(dst.frames[0..st.depth], st.frames[0..st.depth]);
        }
        out.stack_count += 1;

        const inst = self.insts[st.pc];
        switch (inst.kind) {
            .byte => out.allowed_bytes.insert(@intCast(inst.arg)),
            .charset => {
                out.allowed_bytes.unionWith(self.charsets[inst.arg]);
                if (inst.plain_ascii_self_loop) {
                    out.has_plain_ascii_self_loop = true;
                }
            },
            else => unreachable,
        }
    }

    fn expandEpsilon(self: *const GrammarProgram, seed: *const Stack, out: *GrammarState) void {
        var work: [48]Stack = undefined;
        work[0].depth = seed.depth;
        work[0].pc = seed.pc;
        if (seed.depth > 0) {
            @memcpy(work[0].frames[0..seed.depth], seed.frames[0..seed.depth]);
        }
        var sp: usize = 1;
        var steps: usize = 0;

        while (sp > 0 and steps < 512) : (steps += 1) {
            sp -= 1;
            var cur = work[sp];

            if (cur.pc == ACCEPT_PC) {
                out.can_accept_eof = true;
                continue;
            }
            if (cur.pc >= self.inst_count) continue;

            const inst = self.insts[cur.pc];
            switch (inst.kind) {
                .byte, .charset => {
                    self.pushTerminalStack(&cur, out);
                },
                .end => {
                    if (cur.depth == 0) {
                        out.can_accept_eof = true;
                    } else {
                        cur.depth -= 1;
                        cur.pc = cur.frames[cur.depth];
                        if (sp < work.len) {
                            work[sp] = cur;
                            sp += 1;
                        }
                    }
                },
                .jump => {
                    cur.pc = inst.arg;
                    if (sp < work.len) {
                        work[sp] = cur;
                        sp += 1;
                    }
                },
                .branch => {
                    // Push `next` first so `arg` is explored first (priority order).
                    if (sp < work.len) {
                        var second = cur;
                        second.pc = inst.next;
                        work[sp] = second;
                        sp += 1;
                    }
                    if (sp < work.len) {
                        cur.pc = inst.arg;
                        work[sp] = cur;
                        sp += 1;
                    }
                },
                .rule_ref => {
                    if (inst.arg >= self.rule_count) continue;
                    const target_entry = self.rule_entries[inst.arg];
                    if (target_entry == INVALID_RULE) continue;

                    // Tail-call optimization: skip pushing frame when return instruction is `.end`.
                    const is_tail = (inst.next < self.inst_count and self.insts[inst.next].kind == .end);
                    if (!is_tail) {
                        if (cur.depth >= MAX_STACK_DEPTH) continue;
                        cur.frames[cur.depth] = inst.next;
                        cur.depth += 1;
                    }
                    cur.pc = target_entry;
                    if (sp < work.len) {
                        work[sp] = cur;
                        sp += 1;
                    }
                },
            }
        }
    }

    /// Advances `state` in-place by one raw UTF-8/token byte. Returns `false` and
    /// leaves `state` unchanged if `b` is rejected by the grammar.
    pub fn advanceByte(self: *const GrammarProgram, state: *GrammarState, b: u8) bool {
        if (!state.allowed_bytes.contains(b)) return false;

        var next_state = GrammarState{};
        for (state.stacks[0..state.stack_count]) |*st| {
            const inst = self.insts[st.pc];
            const matched = switch (inst.kind) {
                .byte => (@as(u8, @intCast(inst.arg)) == b),
                .charset => self.charsets[inst.arg].contains(b),
                else => false,
            };
            if (matched) {
                var stepped: Stack = undefined;
                stepped.depth = st.depth;
                stepped.pc = inst.next;
                if (st.depth > 0) {
                    @memcpy(stepped.frames[0..st.depth], st.frames[0..st.depth]);
                }
                self.expandEpsilon(&stepped, &next_state);
            }
        }

        if (next_state.isRejected()) return false;
        state.* = next_state;
        return true;
    }

    /// Advances `state` across all bytes of `slice`. Atomic: on rejection, `state`
    /// remains untouched and `false` is returned.
    pub fn advanceBytes(self: *const GrammarProgram, state: *GrammarState, slice: []const u8) bool {
        var cur = state.*;
        for (slice) |b| {
            if (!self.advanceByte(&cur, b)) return false;
        }
        state.* = cur;
        return true;
    }

    /// Returns `true` iff `full_input` is accepted from start to EOF by `self`.
    pub fn matchesComplete(self: *const GrammarProgram, full_input: []const u8) bool {
        var st = self.initialState();
        if (!self.advanceBytes(&st, full_input)) return false;
        return st.isAccepting();
    }
};

// ---------------------------------------------------------------------------
// Preallocated Vocabulary Prefix Trie (`VocabIndex`) for <= 262,144 tokens
// ---------------------------------------------------------------------------

pub const TrieEdge = struct {
    byte: u8,
    child: u32,
};

pub const TrieNode = struct {
    first_edge: u32 = 0,
    edge_count: u16 = 0,
    first_token: u32 = 0,
    token_count: u32 = 0,
    /// True iff path from root to this node AND all descendant edges below it
    /// contain only `isPlainAsciiStringByte` bytes.
    all_plain_from_root: bool = false,
};

/// Startup-allocated flat prefix trie + plain-ASCII-string bitmask for fast
/// zero-allocation per-token `fillAllowedTokenBitmask` up to `262,144` tokens.
pub const VocabIndex = struct {
    vocab_size: usize,
    nodes: []TrieNode,
    edges: []TrieEdge,
    node_tokens: []u32,
    ascii_string_safe_mask: []u32,
    eos_tokens: [MAX_EOS_TOKENS]u32 = @splat(0),
    eos_count: u8 = 0,

    /// Builds a `VocabIndex` from a native `Tokenizer` (`W5`, duck-typed `vocabSize`, `isSpecialToken`, `tokenBytes`).
    pub fn initFromTokenizer(
        allocator: std.mem.Allocator,
        tok: anytype,
        eos_token_ids: []const u32,
    ) VocabError!VocabIndex {
        const TokPtr = @TypeOf(tok);
        const n = tok.vocabSize();
        if (n == 0) return error.EmptyVocab;
        if (n > MAX_VOCAB_SIZE) return error.VocabTooLarge;

        const Ctx = struct {
            t: TokPtr,
            fn bytesAt(self: @This(), id: u32) ?[]const u8 {
                if (self.t.isSpecialToken(id)) return null;
                return self.t.tokenBytes(id) catch null;
            }
        };
        return initGeneric(allocator, n, Ctx{ .t = tok }, Ctx.bytesAt, eos_token_ids);
    }

    /// Builds a `VocabIndex` from explicit token byte slices (`null` entries denote special/skipped tokens).
    pub fn initFromSlices(
        allocator: std.mem.Allocator,
        token_slices: []const ?[]const u8,
        eos_token_ids: []const u32,
    ) VocabError!VocabIndex {
        if (token_slices.len == 0) return error.EmptyVocab;
        if (token_slices.len > MAX_VOCAB_SIZE) return error.VocabTooLarge;

        const Ctx = struct {
            slices: []const ?[]const u8,
            fn bytesAt(self: @This(), id: u32) ?[]const u8 {
                return self.slices[id];
            }
        };
        return initGeneric(allocator, token_slices.len, Ctx{ .slices = token_slices }, Ctx.bytesAt, eos_token_ids);
    }

    fn initGeneric(
        allocator: std.mem.Allocator,
        vocab_size: usize,
        ctx: anytype,
        comptime get_bytes_fn: anytype,
        eos_token_ids: []const u32,
    ) VocabError!VocabIndex {
        const BuilderEdge = struct {
            byte: u8,
            child: u32,
        };
        const BuilderNode = struct {
            edges: std.ArrayListUnmanaged(BuilderEdge) = .empty,
            tokens: std.ArrayListUnmanaged(u32) = .empty,
            path_all_plain: bool = true,
            subtree_all_plain: bool = true,
        };

        var bnodes: std.ArrayListUnmanaged(BuilderNode) = .empty;
        defer {
            for (bnodes.items) |*bn| {
                bn.edges.deinit(allocator);
                bn.tokens.deinit(allocator);
            }
            bnodes.deinit(allocator);
        }

        try bnodes.append(allocator, .{});

        const mask_words = bitmaskWordCount(vocab_size);
        const safe_mask = try allocator.alloc(u32, mask_words);
        errdefer allocator.free(safe_mask);
        @memset(safe_mask, 0);

        var total_tokens_inserted: usize = 0;

        for (0..vocab_size) |tid_usize| {
            const tid: u32 = @intCast(tid_usize);
            const maybe_bytes = get_bytes_fn(ctx, tid);
            const raw = maybe_bytes orelse continue;
            if (raw.len == 0) continue;

            var all_plain = true;
            var cur: u32 = 0;
            for (raw) |b| {
                const plain_b = isPlainAsciiStringByte(b);
                if (!plain_b) all_plain = false;

                var found_child: ?u32 = null;
                for (bnodes.items[cur].edges.items) |e| {
                    if (e.byte == b) {
                        found_child = e.child;
                        break;
                    }
                }
                if (found_child) |c| {
                    cur = c;
                } else {
                    const new_idx: u32 = @intCast(bnodes.items.len);
                    const parent_plain = bnodes.items[cur].path_all_plain;
                    try bnodes.append(allocator, .{
                        .path_all_plain = parent_plain and plain_b,
                        .subtree_all_plain = parent_plain and plain_b,
                    });
                    try bnodes.items[cur].edges.append(allocator, .{ .byte = b, .child = new_idx });
                    cur = new_idx;
                }
            }

            try bnodes.items[cur].tokens.append(allocator, tid);
            total_tokens_inserted += 1;
            if (all_plain) {
                setTokenAllowed(safe_mask, tid);
            }
        }

        // Sort edges by byte at each node and propagate `subtree_all_plain` bottom-up
        // in reverse topological order (child index is always > parent index).
        var rev: usize = bnodes.items.len;
        while (rev > 0) {
            rev -= 1;
            const bn = &bnodes.items[rev];
            std.mem.sort(BuilderEdge, bn.edges.items, {}, struct {
                fn lessThan(_: void, a: BuilderEdge, b: BuilderEdge) bool {
                    return a.byte < b.byte;
                }
            }.lessThan);
            var sub_plain = bn.path_all_plain;
            for (bn.edges.items) |e| {
                if (!isPlainAsciiStringByte(e.byte) or !bnodes.items[e.child].subtree_all_plain) {
                    sub_plain = false;
                }
            }
            bn.subtree_all_plain = sub_plain;
        }

        var total_edges: usize = 0;
        for (bnodes.items) |*bn| total_edges += bn.edges.items.len;

        const nodes = try allocator.alloc(TrieNode, bnodes.items.len);
        errdefer allocator.free(nodes);
        const edges = try allocator.alloc(TrieEdge, total_edges);
        errdefer allocator.free(edges);
        const node_tokens = try allocator.alloc(u32, total_tokens_inserted);
        errdefer allocator.free(node_tokens);

        var edge_off: u32 = 0;
        var tok_off: u32 = 0;
        for (bnodes.items, 0..) |*bn, idx| {
            nodes[idx] = .{
                .first_edge = edge_off,
                .edge_count = @intCast(bn.edges.items.len),
                .first_token = tok_off,
                .token_count = @intCast(bn.tokens.items.len),
                .all_plain_from_root = (idx > 0) and bn.subtree_all_plain,
            };
            for (bn.edges.items) |e| {
                edges[edge_off] = .{ .byte = e.byte, .child = e.child };
                edge_off += 1;
            }
            for (bn.tokens.items) |tid| {
                node_tokens[tok_off] = tid;
                tok_off += 1;
            }
        }

        var res = VocabIndex{
            .vocab_size = vocab_size,
            .nodes = nodes,
            .edges = edges,
            .node_tokens = node_tokens,
            .ascii_string_safe_mask = safe_mask,
        };
        const n_eos = @min(eos_token_ids.len, MAX_EOS_TOKENS);
        for (eos_token_ids[0..n_eos], 0..) |eid, i| {
            if (eid < vocab_size) {
                res.eos_tokens[res.eos_count] = eid;
                res.eos_count += 1;
            }
            _ = i;
        }
        return res;
    }

    pub fn deinit(self: *VocabIndex, allocator: std.mem.Allocator) void {
        allocator.free(self.nodes);
        allocator.free(self.edges);
        allocator.free(self.node_tokens);
        allocator.free(self.ascii_string_safe_mask);
        self.* = undefined;
    }

    /// Advances `state` by `token_id` (handling EOS tokens when `isAccepting()`).
    pub fn advanceToken(
        self: *const VocabIndex,
        program: *const GrammarProgram,
        state: *GrammarState,
        tok: anytype,
        token_id: u32,
    ) bool {
        if (token_id >= self.vocab_size) return false;
        for (self.eos_tokens[0..self.eos_count]) |eid| {
            if (eid == token_id) {
                return state.isAccepting();
            }
        }
        if (tok.isSpecialToken(token_id)) return false;
        const bytes = tok.tokenBytes(token_id) catch return false;
        if (bytes.len == 0) return false;
        return program.advanceBytes(state, bytes);
    }

    /// Zero-allocation P-core bitmask evaluation over `self` (`vocab_size <= 262,144`).
    /// When `program == null`, marks all tokens `0..vocab_size` allowed.
    pub fn fillAllowedTokenBitmask(
        self: *const VocabIndex,
        program_opt: ?*const GrammarProgram,
        state: *const GrammarState,
        out_mask: []u32,
    ) void {
        const words = bitmaskWordCount(self.vocab_size);
        std.debug.assert(out_mask.len >= words);

        const program = program_opt orelse {
            allowAllTokens(out_mask, self.vocab_size);
            return;
        };

        @memset(out_mask[0..words], 0);

        if (state.can_accept_eof) {
            for (self.eos_tokens[0..self.eos_count]) |eid| {
                setTokenAllowed(out_mask, eid);
            }
        }
        if (state.stack_count == 0 or self.nodes.len == 0) return;

        // Fast-path bulk OR when root state has a plain-ASCII-string self-loop.
        const root_plain_loop = state.has_plain_ascii_self_loop;
        if (root_plain_loop) {
            for (0..words) |w| {
                out_mask[w] |= self.ascii_string_safe_mask[w];
            }
        }

        const Frame = struct {
            node_idx: u32,
            edge_cursor: u16,
            state: GrammarState,
        };

        // Preallocated stack frame array bounded by max token byte length (<= 128).
        var dfs: [128]Frame = undefined;
        dfs[0] = .{
            .node_idx = 0,
            .edge_cursor = 0,
            .state = state.*,
        };
        var depth: usize = 1;

        while (depth > 0) {
            const top = &dfs[depth - 1];
            const node = self.nodes[top.node_idx];
            if (top.edge_cursor >= node.edge_count) {
                depth -= 1;
                continue;
            }

            const edge = self.edges[node.first_edge + top.edge_cursor];
            top.edge_cursor += 1;

            if (!top.state.allowed_bytes.contains(edge.byte)) continue;

            const child_node = self.nodes[edge.child];
            // Exact pruning: every token in `child_node`'s subtree consists purely of
            // plain ASCII string bytes from root to leaf and was already marked `1` above.
            if (root_plain_loop and child_node.all_plain_from_root) continue;

            var next_st = top.state;
            if (!program.advanceByte(&next_st, edge.byte)) continue;

            if (child_node.token_count > 0) {
                const tids = self.node_tokens[child_node.first_token .. child_node.first_token + child_node.token_count];
                for (tids) |tid| {
                    setTokenAllowed(out_mask, tid);
                }
            }

            if (child_node.edge_count > 0 and depth < dfs.len) {
                dfs[depth] = .{
                    .node_idx = edge.child,
                    .edge_cursor = 0,
                    .state = next_st,
                };
                depth += 1;
            }
        }
    }
};

/// Direct zero-allocation bitmask evaluation over raw token slices without a prebuilt
/// `VocabIndex` (convenient for small vocabularies and standalone unit tests).
pub fn fillAllowedTokenBitmaskDirect(
    program_opt: ?*const GrammarProgram,
    state: *const GrammarState,
    token_slices: []const ?[]const u8,
    eos_token_ids: []const u32,
    out_mask: []u32,
) void {
    const vocab_size = token_slices.len;
    const words = bitmaskWordCount(vocab_size);
    std.debug.assert(out_mask.len >= words);

    const program = program_opt orelse {
        allowAllTokens(out_mask, vocab_size);
        return;
    };

    @memset(out_mask[0..words], 0);
    if (state.can_accept_eof) {
        for (eos_token_ids) |eid| {
            if (eid < vocab_size) setTokenAllowed(out_mask, eid);
        }
    }
    if (state.stack_count == 0) return;

    for (token_slices, 0..) |maybe_tok, idx| {
        const raw = maybe_tok orelse continue;
        if (raw.len == 0) continue;
        if (!state.allowed_bytes.contains(raw[0])) continue;
        var st = state.*;
        if (program.advanceBytes(&st, raw)) {
            setTokenAllowed(out_mask, @intCast(idx));
        }
    }
}

// ---------------------------------------------------------------------------
// Fragment Builder Helpers (Thompson NFA/PDA construction)
// ---------------------------------------------------------------------------

const Frag = struct {
    start: u16,
    /// Instruction index whose `.next` (or `.arg` if `patch_arg`) awaits patching.
    /// Instead of open patch lists, we chain sequentially with an explicit target `end_pc`!
    end_pc: u16,
};

/// Emits a literal byte sequence ending at `after_pc`, returning the entry PC.
fn emitLiteral(prog: *GrammarProgram, bytes: []const u8, after_pc: u16) CompileError!u16 {
    var next_pc = after_pc;
    var i = bytes.len;
    while (i > 0) {
        i -= 1;
        next_pc = try prog.emitInst(.{
            .kind = .byte,
            .arg = bytes[i],
            .next = next_pc,
        });
    }
    return next_pc;
}

/// Emits valid non-control UTF-8 character transitions (excluding `"` and `\`)
/// leading to `after_pc`, returning the entry PC.
fn emitJsonStringChar(prog: *GrammarProgram, after_pc: u16) CompileError!u16 {
    // 1. Plain ASCII `[0x20..0x7E]` minus `"` and `\`, plus `0x7F` if allowed (RFC 8259 allows 0x20..0x10FFFF).
    var ascii_set = PLAIN_ASCII_STRING_SET;
    ascii_set.insert(0x7F);
    const cs_ascii = try prog.internCharset(ascii_set);
    const p_ascii = try prog.emitInst(.{ .kind = .charset, .arg = cs_ascii, .next = after_pc });

    var cont_set = ByteSet.empty;
    cont_set.insertRange(0x80, 0xBF);
    const cs_cont = try prog.internCharset(cont_set);

    // 2-byte UTF-8: [0xC2..0xDF] [0x80..0xBF]
    var lead2 = ByteSet.empty;
    lead2.insertRange(0xC2, 0xDF);
    const cs_lead2 = try prog.internCharset(lead2);
    const p_cont1 = try prog.emitInst(.{ .kind = .charset, .arg = cs_cont, .next = after_pc });
    const p_utf2 = try prog.emitInst(.{ .kind = .charset, .arg = cs_lead2, .next = p_cont1 });

    // 3-byte UTF-8:
    //   0xE0 [0xA0..0xBF] [0x80..0xBF]
    //   [0xE1..0xEC, 0xEE..0xEF] [0x80..0xBF] [0x80..0xBF]
    //   0xED [0x80..0x9F] [0x80..0xBF]
    var e0_mid = ByteSet.empty;
    e0_mid.insertRange(0xA0, 0xBF);
    const cs_e0_mid = try prog.internCharset(e0_mid);
    const p_e0_2 = try prog.emitInst(.{ .kind = .charset, .arg = cs_e0_mid, .next = p_cont1 });
    const p_e0 = try prog.emitInst(.{ .kind = .byte, .arg = 0xE0, .next = p_e0_2 });

    var e_norm = ByteSet.empty;
    e_norm.insertRange(0xE1, 0xEC);
    e_norm.insertRange(0xEE, 0xEF);
    const cs_e_norm = try prog.internCharset(e_norm);
    const p_cont2 = try prog.emitInst(.{ .kind = .charset, .arg = cs_cont, .next = p_cont1 });
    const p_enorm = try prog.emitInst(.{ .kind = .charset, .arg = cs_e_norm, .next = p_cont2 });

    var ed_mid = ByteSet.empty;
    ed_mid.insertRange(0x80, 0x9F);
    const cs_ed_mid = try prog.internCharset(ed_mid);
    const p_ed_2 = try prog.emitInst(.{ .kind = .charset, .arg = cs_ed_mid, .next = p_cont1 });
    const p_ed = try prog.emitInst(.{ .kind = .byte, .arg = 0xED, .next = p_ed_2 });

    // 4-byte UTF-8:
    //   0xF0 [0x90..0xBF] cont cont
    //   [0xF1..0xF3] cont cont cont
    //   0xF4 [0x80..0x8F] cont cont
    var f0_mid = ByteSet.empty;
    f0_mid.insertRange(0x90, 0xBF);
    const cs_f0_mid = try prog.internCharset(f0_mid);
    const p_f0_2 = try prog.emitInst(.{ .kind = .charset, .arg = cs_f0_mid, .next = p_cont2 });
    const p_f0 = try prog.emitInst(.{ .kind = .byte, .arg = 0xF0, .next = p_f0_2 });

    var f_norm = ByteSet.empty;
    f_norm.insertRange(0xF1, 0xF3);
    const cs_f_norm = try prog.internCharset(f_norm);
    const p_cont3 = try prog.emitInst(.{ .kind = .charset, .arg = cs_cont, .next = p_cont2 });
    const p_fnorm = try prog.emitInst(.{ .kind = .charset, .arg = cs_f_norm, .next = p_cont3 });

    var f4_mid = ByteSet.empty;
    f4_mid.insertRange(0x80, 0x8F);
    const cs_f4_mid = try prog.internCharset(f4_mid);
    const p_f4_2 = try prog.emitInst(.{ .kind = .charset, .arg = cs_f4_mid, .next = p_cont2 });
    const p_f4 = try prog.emitInst(.{ .kind = .byte, .arg = 0xF4, .next = p_f4_2 });

    // Escape sequence: `\` followed by `["\\/bfnrt]` or `u` + 4 hex digits
    var esc_simple = ByteSet.empty;
    for ("\"\\/bfnrt") |ch| esc_simple.insert(ch);
    const cs_esc_simple = try prog.internCharset(esc_simple);
    const p_esc_s = try prog.emitInst(.{ .kind = .charset, .arg = cs_esc_simple, .next = after_pc });

    var hex_set = ByteSet.empty;
    hex_set.insertRange('0', '9');
    hex_set.insertRange('a', 'f');
    hex_set.insertRange('A', 'F');
    const cs_hex = try prog.internCharset(hex_set);
    const h4 = try prog.emitInst(.{ .kind = .charset, .arg = cs_hex, .next = after_pc });
    const h3 = try prog.emitInst(.{ .kind = .charset, .arg = cs_hex, .next = h4 });
    const h2 = try prog.emitInst(.{ .kind = .charset, .arg = cs_hex, .next = h3 });
    const h1 = try prog.emitInst(.{ .kind = .charset, .arg = cs_hex, .next = h2 });
    const p_esc_u = try prog.emitInst(.{ .kind = .byte, .arg = 'u', .next = h1 });
    const p_esc_branch = try prog.emitInst(.{ .kind = .branch, .arg = p_esc_s, .next = p_esc_u });
    const p_bslash = try prog.emitInst(.{ .kind = .byte, .arg = '\\', .next = p_esc_branch });

    // Chain alternatives: ascii | bslash | utf2 | e0 | enorm | ed | f0 | fnorm | f4
    const b7 = try prog.emitInst(.{ .kind = .branch, .arg = p_fnorm, .next = p_f4 });
    const b6 = try prog.emitInst(.{ .kind = .branch, .arg = p_f0, .next = b7 });
    const b5 = try prog.emitInst(.{ .kind = .branch, .arg = p_ed, .next = b6 });
    const b4 = try prog.emitInst(.{ .kind = .branch, .arg = p_enorm, .next = b5 });
    const b3 = try prog.emitInst(.{ .kind = .branch, .arg = p_e0, .next = b4 });
    const b2 = try prog.emitInst(.{ .kind = .branch, .arg = p_utf2, .next = b3 });
    const b1 = try prog.emitInst(.{ .kind = .branch, .arg = p_bslash, .next = b2 });
    return prog.emitInst(.{ .kind = .branch, .arg = p_ascii, .next = b1 });
}

/// Core reusable JSON primitive rule IDs inside a `GrammarProgram`.
const JsonBaseRules = struct {
    ws: u16,
    string: u16,
    integer: u16,
    number: u16,
    boolean: u16,
    null_lit: u16,
    any_value: u16,
    any_object: u16,
    any_array: u16,
};

fn buildJsonBaseRules(prog: *GrammarProgram) CompileError!JsonBaseRules {
    const r_ws = try prog.allocRule();
    const r_string = try prog.allocRule();
    const r_integer = try prog.allocRule();
    const r_number = try prog.allocRule();
    const r_boolean = try prog.allocRule();
    const r_null = try prog.allocRule();
    const r_any_val = try prog.allocRule();
    const r_any_obj = try prog.allocRule();
    const r_any_arr = try prog.allocRule();

    // 1. `ws ::= [ \t\n\r]*` (capped whitespace loop)
    {
        var ws_set = ByteSet.empty;
        ws_set.insert(' ');
        ws_set.insert('\t');
        ws_set.insert('\n');
        ws_set.insert('\r');
        const cs_ws = try prog.internCharset(ws_set);
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const loop_branch = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = end_pc });
        const step_ws = try prog.emitInst(.{ .kind = .charset, .arg = cs_ws, .next = loop_branch });
        prog.insts[loop_branch].arg = step_ws;
        prog.rule_entries[r_ws] = loop_branch;
    }

    // 2. `string ::= '"' char* '"' ws`
    {
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const trail_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = end_pc });
        const close_q = try prog.emitInst(.{ .kind = .byte, .arg = '"', .next = trail_ws });
        const loop_branch = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = close_q });
        const char_entry = try emitJsonStringChar(prog, loop_branch);
        prog.insts[loop_branch].arg = char_entry;
        const open_q = try prog.emitInst(.{ .kind = .byte, .arg = '"', .next = loop_branch });
        prog.rule_entries[r_string] = open_q;
    }

    var digit_set = ByteSet.empty;
    digit_set.insertRange('0', '9');
    const cs_digit = try prog.internCharset(digit_set);

    var nonzero_set = ByteSet.empty;
    nonzero_set.insertRange('1', '9');
    const cs_nonzero = try prog.internCharset(nonzero_set);

    // 3. `integer ::= '-'? ('0' | [1-9] [0-9]*) ws`
    {
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const trail_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = end_pc });

        const dig_loop = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = trail_ws });
        const dig_step = try prog.emitInst(.{ .kind = .charset, .arg = cs_digit, .next = dig_loop });
        prog.insts[dig_loop].arg = dig_step;

        const nz_start = try prog.emitInst(.{ .kind = .charset, .arg = cs_nonzero, .next = dig_loop });
        const zero_lit = try prog.emitInst(.{ .kind = .byte, .arg = '0', .next = trail_ws });
        const int_core = try prog.emitInst(.{ .kind = .branch, .arg = zero_lit, .next = nz_start });

        const minus = try prog.emitInst(.{ .kind = .byte, .arg = '-', .next = int_core });
        const opt_minus = try prog.emitInst(.{ .kind = .branch, .arg = minus, .next = int_core });
        prog.rule_entries[r_integer] = opt_minus;
    }

    // 4. `number ::= '-'? ('0' | [1-9] [0-9]*) ('.' [0-9]+)? ([eE] [+-]? [0-9]+)? ws`
    {
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const trail_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = end_pc });

        // Exponent `([eE] [+-]? [0-9]+)?`
        const exp_loop = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = trail_ws });
        const exp_dig_more = try prog.emitInst(.{ .kind = .charset, .arg = cs_digit, .next = exp_loop });
        prog.insts[exp_loop].arg = exp_dig_more;
        const exp_dig1 = try prog.emitInst(.{ .kind = .charset, .arg = cs_digit, .next = exp_loop });

        var sign_set = ByteSet.empty;
        sign_set.insert('+');
        sign_set.insert('-');
        const cs_sign = try prog.internCharset(sign_set);
        const exp_sign = try prog.emitInst(.{ .kind = .charset, .arg = cs_sign, .next = exp_dig1 });
        const opt_sign = try prog.emitInst(.{ .kind = .branch, .arg = exp_sign, .next = exp_dig1 });

        var e_set = ByteSet.empty;
        e_set.insert('e');
        e_set.insert('E');
        const cs_e = try prog.internCharset(e_set);
        const exp_start = try prog.emitInst(.{ .kind = .charset, .arg = cs_e, .next = opt_sign });
        const opt_exp = try prog.emitInst(.{ .kind = .branch, .arg = exp_start, .next = trail_ws });

        // Fraction `('.' [0-9]+)?`
        const frac_loop = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = opt_exp });
        const frac_more = try prog.emitInst(.{ .kind = .charset, .arg = cs_digit, .next = frac_loop });
        prog.insts[frac_loop].arg = frac_more;
        const frac_dig1 = try prog.emitInst(.{ .kind = .charset, .arg = cs_digit, .next = frac_loop });
        const frac_dot = try prog.emitInst(.{ .kind = .byte, .arg = '.', .next = frac_dig1 });
        const opt_frac = try prog.emitInst(.{ .kind = .branch, .arg = frac_dot, .next = opt_exp });

        // Integer core
        const int_loop = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = opt_frac });
        const int_more = try prog.emitInst(.{ .kind = .charset, .arg = cs_digit, .next = int_loop });
        prog.insts[int_loop].arg = int_more;
        const nz_start = try prog.emitInst(.{ .kind = .charset, .arg = cs_nonzero, .next = int_loop });
        const zero_lit = try prog.emitInst(.{ .kind = .byte, .arg = '0', .next = opt_frac });
        const int_core = try prog.emitInst(.{ .kind = .branch, .arg = zero_lit, .next = nz_start });

        const minus = try prog.emitInst(.{ .kind = .byte, .arg = '-', .next = int_core });
        const opt_minus = try prog.emitInst(.{ .kind = .branch, .arg = minus, .next = int_core });
        prog.rule_entries[r_number] = opt_minus;
    }

    // 5. `boolean ::= ("true" | "false") ws`
    {
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const trail_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = end_pc });
        const t_lit = try emitLiteral(prog, "true", trail_ws);
        const f_lit = try emitLiteral(prog, "false", trail_ws);
        prog.rule_entries[r_boolean] = try prog.emitInst(.{ .kind = .branch, .arg = t_lit, .next = f_lit });
    }

    // 6. `null_lit ::= "null" ws`
    {
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const trail_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = end_pc });
        prog.rule_entries[r_null] = try emitLiteral(prog, "null", trail_ws);
    }

    // 7. `any_object ::= '{' ws (string ':' ws any_value (',' ws string ':' ws any_value)*)? '}' ws`
    {
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const trail_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = end_pc });
        const close_brace = try prog.emitInst(.{ .kind = .byte, .arg = '}', .next = trail_ws });

        const more_loop = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = close_brace });
        const kv_val2 = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_any_val, .next = more_loop });
        const kv_ws2 = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = kv_val2 });
        const kv_colon2 = try prog.emitInst(.{ .kind = .byte, .arg = ':', .next = kv_ws2 });
        const kv_key2 = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_string, .next = kv_colon2 });
        const comma_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = kv_key2 });
        const comma = try prog.emitInst(.{ .kind = .byte, .arg = ',', .next = comma_ws });
        prog.insts[more_loop].arg = comma;

        const kv_val1 = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_any_val, .next = more_loop });
        const kv_ws1 = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = kv_val1 });
        const kv_colon1 = try prog.emitInst(.{ .kind = .byte, .arg = ':', .next = kv_ws1 });
        const kv_key1 = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_string, .next = kv_colon1 });

        const opt_body = try prog.emitInst(.{ .kind = .branch, .arg = kv_key1, .next = close_brace });
        const open_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = opt_body });
        const open_brace = try prog.emitInst(.{ .kind = .byte, .arg = '{', .next = open_ws });
        prog.rule_entries[r_any_obj] = open_brace;
    }

    // 8. `any_array ::= '[' ws (any_value (',' ws any_value)*)? ']' ws`
    {
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const trail_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = end_pc });
        const close_br = try prog.emitInst(.{ .kind = .byte, .arg = ']', .next = trail_ws });

        const more_loop = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = close_br });
        const elem2 = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_any_val, .next = more_loop });
        const comma_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = elem2 });
        const comma = try prog.emitInst(.{ .kind = .byte, .arg = ',', .next = comma_ws });
        prog.insts[more_loop].arg = comma;

        const elem1 = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_any_val, .next = more_loop });
        const opt_elems = try prog.emitInst(.{ .kind = .branch, .arg = elem1, .next = close_br });
        const open_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_ws, .next = opt_elems });
        const open_br = try prog.emitInst(.{ .kind = .byte, .arg = '[', .next = open_ws });
        prog.rule_entries[r_any_arr] = open_br;
    }

    // 9. `any_value ::= any_object | any_array | string | number | boolean | null_lit`
    {
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const v_obj = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_any_obj, .next = end_pc });
        const v_arr = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_any_arr, .next = end_pc });
        const v_str = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_string, .next = end_pc });
        const v_num = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_number, .next = end_pc });
        const v_bool = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_boolean, .next = end_pc });
        const v_null = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_null, .next = end_pc });

        const b4 = try prog.emitInst(.{ .kind = .branch, .arg = v_bool, .next = v_null });
        const b3 = try prog.emitInst(.{ .kind = .branch, .arg = v_num, .next = b4 });
        const b2 = try prog.emitInst(.{ .kind = .branch, .arg = v_str, .next = b3 });
        const b1 = try prog.emitInst(.{ .kind = .branch, .arg = v_arr, .next = b2 });
        prog.rule_entries[r_any_val] = try prog.emitInst(.{ .kind = .branch, .arg = v_obj, .next = b1 });
    }

    return .{
        .ws = r_ws,
        .string = r_string,
        .integer = r_integer,
        .number = r_number,
        .boolean = r_boolean,
        .null_lit = r_null,
        .any_value = r_any_val,
        .any_object = r_any_obj,
        .any_array = r_any_arr,
    };
}

// ---------------------------------------------------------------------------
// 1. `json_object` Compiler
// ---------------------------------------------------------------------------

pub fn compileJsonObject(prog: *GrammarProgram) CompileError!void {
    prog.reset();
    const base = try buildJsonBaseRules(prog);
    const r_root = try prog.allocRule();
    const end_pc = try prog.emitInst(.{ .kind = .end });
    const obj_call = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.any_object, .next = end_pc });
    const lead_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = obj_call });
    prog.rule_entries[r_root] = lead_ws;
    prog.root_rule = r_root;
    prog.finalizeSelfLoops();
}

// ---------------------------------------------------------------------------
// 2. Zero-Allocation `json_schema` (`strict: true`) & Tool-Call Compiler
// ---------------------------------------------------------------------------

const SchemaJsonScanner = struct {
    src: []const u8,
    pos: usize = 0,

    fn skipWs(self: *SchemaJsonScanner) void {
        while (self.pos < self.src.len) {
            switch (self.src[self.pos]) {
                ' ', '\t', '\r', '\n' => self.pos += 1,
                else => return,
            }
        }
    }

    fn peek(self: *SchemaJsonScanner) ?u8 {
        self.skipWs();
        return if (self.pos < self.src.len) self.src[self.pos] else null;
    }

    fn expect(self: *SchemaJsonScanner, ch: u8) CompileError!void {
        self.skipWs();
        if (self.pos >= self.src.len or self.src[self.pos] != ch) return error.InvalidJsonSchema;
        self.pos += 1;
    }

    fn consumeKeyword(self: *SchemaJsonScanner, kw: []const u8) bool {
        self.skipWs();
        if (self.src.len - self.pos < kw.len) return false;
        if (!std.mem.eql(u8, self.src[self.pos .. self.pos + kw.len], kw)) return false;
        self.pos += kw.len;
        return true;
    }

    /// Captures the exact JSON string token including quotes (`"..."`) while validating escapes.
    fn captureQuotedString(self: *SchemaJsonScanner) CompileError![]const u8 {
        self.skipWs();
        const start = self.pos;
        if (self.pos >= self.src.len or self.src[self.pos] != '"') return error.InvalidJsonSchema;
        self.pos += 1;
        while (self.pos < self.src.len) {
            const b = self.src[self.pos];
            if (b == '"') {
                self.pos += 1;
                return self.src[start..self.pos];
            }
            if (b < 0x20) return error.InvalidJsonSchema;
            if (b == '\\') {
                self.pos += 1;
                if (self.pos >= self.src.len) return error.InvalidJsonSchema;
                const esc = self.src[self.pos];
                self.pos += 1;
                if (esc == 'u') {
                    if (self.src.len - self.pos < 4) return error.InvalidJsonSchema;
                    for (self.src[self.pos .. self.pos + 4]) |hc| {
                        if (!std.ascii.isHex(hc)) return error.InvalidJsonSchema;
                    }
                    self.pos += 4;
                }
            } else {
                self.pos += 1;
            }
        }
        return error.InvalidJsonSchema;
    }

    /// Unescapes a JSON string token into `dst`.
    fn readUnescapedString(self: *SchemaJsonScanner, dst: []u8) CompileError![]const u8 {
        const quoted = try self.captureQuotedString();
        return unescapeJsonToken(quoted, dst);
    }

    fn readU32(self: *SchemaJsonScanner) CompileError!u32 {
        const raw = try self.captureValue(1);
        return std.fmt.parseInt(u32, raw, 10) catch error.InvalidJsonSchema;
    }

    fn readBool(self: *SchemaJsonScanner) CompileError!bool {
        if (self.consumeKeyword("true")) return true;
        if (self.consumeKeyword("false")) return false;
        return error.InvalidJsonSchema;
    }

    fn captureValue(self: *SchemaJsonScanner, depth: usize) CompileError![]const u8 {
        if (depth > MAX_SCHEMA_DEPTH * 2) return error.NestingDepthExceeded;
        self.skipWs();
        const start = self.pos;
        const ch = self.peek() orelse return error.InvalidJsonSchema;
        switch (ch) {
            '"' => _ = try self.captureQuotedString(),
            '{' => {
                self.pos += 1;
                if (self.peek() == '}') {
                    self.pos += 1;
                } else {
                    while (true) {
                        _ = try self.captureQuotedString();
                        try self.expect(':');
                        _ = try self.captureValue(depth + 1);
                        const sep = self.peek() orelse return error.InvalidJsonSchema;
                        if (sep == ',') {
                            self.pos += 1;
                        } else if (sep == '}') {
                            self.pos += 1;
                            break;
                        } else return error.InvalidJsonSchema;
                    }
                }
            },
            '[' => {
                self.pos += 1;
                if (self.peek() == ']') {
                    self.pos += 1;
                } else {
                    while (true) {
                        _ = try self.captureValue(depth + 1);
                        const sep = self.peek() orelse return error.InvalidJsonSchema;
                        if (sep == ',') {
                            self.pos += 1;
                        } else if (sep == ']') {
                            self.pos += 1;
                            break;
                        } else return error.InvalidJsonSchema;
                    }
                }
            },
            't' => if (!self.consumeKeyword("true")) return error.InvalidJsonSchema,
            'f' => if (!self.consumeKeyword("false")) return error.InvalidJsonSchema,
            'n' => if (!self.consumeKeyword("null")) return error.InvalidJsonSchema,
            '-', '0'...'9' => {
                if (self.src[self.pos] == '-') self.pos += 1;
                if (self.pos >= self.src.len or !std.ascii.isDigit(self.src[self.pos])) return error.InvalidJsonSchema;
                while (self.pos < self.src.len) {
                    const c = self.src[self.pos];
                    if (std.ascii.isDigit(c) or c == '.' or c == 'e' or c == 'E' or c == '+' or c == '-') {
                        self.pos += 1;
                    } else break;
                }
            },
            else => return error.InvalidJsonSchema,
        }
        return std.mem.trim(u8, self.src[start..self.pos], " \t\r\n");
    }
};

fn unescapeJsonToken(quoted: []const u8, dst: []u8) CompileError![]const u8 {
    if (quoted.len < 2 or quoted[0] != '"' or quoted[quoted.len - 1] != '"') return error.InvalidJsonSchema;
    const inner = quoted[1 .. quoted.len - 1];
    var i: usize = 0;
    var out: usize = 0;
    while (i < inner.len) {
        if (inner[i] != '\\') {
            if (out >= dst.len) return error.ProgramCapacityExceeded;
            dst[out] = inner[i];
            out += 1;
            i += 1;
            continue;
        }
        i += 1;
        if (i >= inner.len) return error.InvalidJsonSchema;
        const esc = inner[i];
        i += 1;
        const b: u8 = switch (esc) {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => 0x08,
            'f' => 0x0C,
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'u' => {
                if (inner.len - i < 4) return error.InvalidJsonSchema;
                const cp = std.fmt.parseInt(u21, inner[i .. i + 4], 16) catch return error.InvalidJsonSchema;
                i += 4;
                var ubuf: [4]u8 = undefined;
                const un = std.unicode.utf8Encode(cp, &ubuf) catch return error.InvalidJsonSchema;
                if (out + un > dst.len) return error.ProgramCapacityExceeded;
                @memcpy(dst[out .. out + un], ubuf[0..un]);
                out += un;
                continue;
            },
            else => return error.InvalidJsonSchema,
        };
        if (out >= dst.len) return error.ProgramCapacityExceeded;
        dst[out] = b;
        out += 1;
    }
    return dst[0..out];
}

const DefEntry = struct {
    name_quoted: []const u8,
    schema_json: []const u8,
    rule_id: u16 = INVALID_RULE,
    compiling: bool = false,
};

const SchemaCompiler = struct {
    prog: *GrammarProgram,
    base: JsonBaseRules,
    defs: [MAX_DEFS]DefEntry = undefined,
    def_count: usize = 0,

    fn collectRootDefs(self: *SchemaCompiler, root_json: []const u8) CompileError!void {
        var s = SchemaJsonScanner{ .src = root_json };
        if (s.peek() != '{') return;
        try s.expect('{');
        if (s.peek() == '}') return;

        while (true) {
            var kbuf: [64]u8 = undefined;
            const key = s.readUnescapedString(&kbuf) catch |err| switch (err) {
                error.ProgramCapacityExceeded => blk: {
                    _ = try s.captureValue(1);
                    break :blk "";
                },
                else => return err,
            };
            try s.expect(':');
            if (std.mem.eql(u8, key, "$defs") or std.mem.eql(u8, key, "definitions")) {
                const defs_obj = try s.captureValue(1);
                try self.parseDefsMap(defs_obj);
            } else {
                _ = try s.captureValue(1);
            }
            const sep = s.peek() orelse return error.InvalidJsonSchema;
            if (sep == ',') {
                s.pos += 1;
            } else if (sep == '}') {
                s.pos += 1;
                break;
            } else return error.InvalidJsonSchema;
        }
    }

    fn parseDefsMap(self: *SchemaCompiler, defs_json: []const u8) CompileError!void {
        var s = SchemaJsonScanner{ .src = defs_json };
        try s.expect('{');
        if (s.peek() == '}') return;
        while (true) {
            const name_q = try s.captureQuotedString();
            try s.expect(':');
            const val_slice = try s.captureValue(2);
            if (self.def_count >= MAX_DEFS) return error.RuleCapacityExceeded;
            self.defs[self.def_count] = .{
                .name_quoted = name_q,
                .schema_json = val_slice,
                .rule_id = INVALID_RULE,
            };
            self.def_count += 1;
            const sep = s.peek() orelse return error.InvalidJsonSchema;
            if (sep == ',') {
                s.pos += 1;
            } else if (sep == '}') {
                s.pos += 1;
                break;
            } else return error.InvalidJsonSchema;
        }
    }

    fn resolveRef(self: *SchemaCompiler, ref_str: []const u8, depth: usize) CompileError!u16 {
        const name = if (std.mem.startsWith(u8, ref_str, "#/$defs/"))
            ref_str["#/$defs/".len..]
        else if (std.mem.startsWith(u8, ref_str, "#/definitions/"))
            ref_str["#/definitions/".len..]
        else
            return error.UnknownSchemaRef;

        for (self.defs[0..self.def_count]) |*d| {
            var nbuf: [128]u8 = undefined;
            const dname = try unescapeJsonToken(d.name_quoted, &nbuf);
            if (std.mem.eql(u8, dname, name)) {
                if (d.rule_id != INVALID_RULE) return d.rule_id;
                // Allocate rule ID upfront so recursive `$ref`s bind cleanly!
                const r = try self.prog.allocRule();
                d.rule_id = r;
                try self.compileSchemaIntoRule(r, d.schema_json, depth + 1);
                return r;
            }
        }
        return error.UnknownSchemaRef;
    }

    fn compileSchemaNode(self: *SchemaCompiler, schema_json: []const u8, depth: usize) CompileError!u16 {
        const r = try self.prog.allocRule();
        try self.compileSchemaIntoRule(r, schema_json, depth);
        return r;
    }

    fn compileSchemaIntoRule(self: *SchemaCompiler, rule_id: u16, schema_json: []const u8, depth: usize) CompileError!void {
        if (depth > MAX_SCHEMA_DEPTH) return error.NestingDepthExceeded;
        var s = SchemaJsonScanner{ .src = schema_json };
        const first = s.peek() orelse return error.InvalidJsonSchema;

        // Boolean JSON Schema (`true` = any JSON value, `false` = unsatisfiable).
        if (first == 't' or first == 'f') {
            const allow = try s.readBool();
            if (!allow) return error.InvalidJsonSchema;
            const end_pc = try self.prog.emitInst(.{ .kind = .end });
            self.prog.rule_entries[rule_id] = try self.prog.emitInst(.{
                .kind = .rule_ref,
                .arg = self.base.any_value,
                .next = end_pc,
            });
            return;
        }

        try s.expect('{');

        var type_slice: ?[]const u8 = null;
        var props_slice: ?[]const u8 = null;
        var required_slice: ?[]const u8 = null;
        var items_slice: ?[]const u8 = null;
        var prefix_items_slice: ?[]const u8 = null;
        var enum_slice: ?[]const u8 = null;
        var const_slice: ?[]const u8 = null;
        var any_of_slice: ?[]const u8 = null;
        var ref_buf: [128]u8 = undefined;
        var ref_str: ?[]const u8 = null;
        var additional_props_bool: ?bool = null;
        var additional_props_schema: ?[]const u8 = null;
        var min_items: u32 = 0;
        var max_items: ?u32 = null;
        var min_length: u32 = 0;
        var max_length: ?u32 = null;
        var nullable_flag: bool = false;

        if (s.peek() != '}') {
            while (true) {
                var kbuf: [64]u8 = undefined;
                const key = s.readUnescapedString(&kbuf) catch |err| switch (err) {
                    error.ProgramCapacityExceeded => blk: {
                        _ = try s.captureValue(depth + 1);
                        break :blk "";
                    },
                    else => return err,
                };
                try s.expect(':');

                if (std.mem.eql(u8, key, "type")) {
                    type_slice = try s.captureValue(depth + 1);
                } else if (std.mem.eql(u8, key, "properties")) {
                    props_slice = try s.captureValue(depth + 1);
                } else if (std.mem.eql(u8, key, "required")) {
                    required_slice = try s.captureValue(depth + 1);
                } else if (std.mem.eql(u8, key, "additionalProperties")) {
                    const ap = try s.captureValue(depth + 1);
                    if (std.mem.eql(u8, ap, "false")) {
                        additional_props_bool = false;
                    } else if (std.mem.eql(u8, ap, "true")) {
                        additional_props_bool = true;
                    } else if (ap.len > 0 and ap[0] == '{') {
                        additional_props_schema = ap;
                    } else return error.InvalidJsonSchema;
                } else if (std.mem.eql(u8, key, "items")) {
                    items_slice = try s.captureValue(depth + 1);
                } else if (std.mem.eql(u8, key, "prefixItems")) {
                    prefix_items_slice = try s.captureValue(depth + 1);
                } else if (std.mem.eql(u8, key, "minItems")) {
                    min_items = try s.readU32();
                } else if (std.mem.eql(u8, key, "maxItems")) {
                    max_items = try s.readU32();
                } else if (std.mem.eql(u8, key, "minLength")) {
                    min_length = try s.readU32();
                } else if (std.mem.eql(u8, key, "maxLength")) {
                    max_length = try s.readU32();
                } else if (std.mem.eql(u8, key, "enum")) {
                    enum_slice = try s.captureValue(depth + 1);
                } else if (std.mem.eql(u8, key, "const")) {
                    const_slice = try s.captureValue(depth + 1);
                } else if (std.mem.eql(u8, key, "anyOf") or std.mem.eql(u8, key, "oneOf")) {
                    any_of_slice = try s.captureValue(depth + 1);
                } else if (std.mem.eql(u8, key, "$ref")) {
                    ref_str = try s.readUnescapedString(&ref_buf);
                } else if (std.mem.eql(u8, key, "nullable")) {
                    nullable_flag = try s.readBool();
                } else {
                    _ = try s.captureValue(depth + 1);
                }

                const sep = s.peek() orelse return error.InvalidJsonSchema;
                if (sep == ',') {
                    s.pos += 1;
                } else if (sep == '}') {
                    s.pos += 1;
                    break;
                } else return error.InvalidJsonSchema;
            }
        } else {
            s.pos += 1;
        }

        const end_pc = try self.prog.emitInst(.{ .kind = .end });

        // 1. `$ref`
        if (ref_str) |rf| {
            const target = try self.resolveRef(rf, depth);
            var entry = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = target, .next = end_pc });
            if (nullable_flag) {
                const n_pc = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.null_lit, .next = end_pc });
                entry = try self.prog.emitInst(.{ .kind = .branch, .arg = entry, .next = n_pc });
            }
            self.prog.rule_entries[rule_id] = entry;
            return;
        }

        // 2. `const`
        if (const_slice) |cs_lit| {
            const trail_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = end_pc });
            self.prog.rule_entries[rule_id] = try emitCanonicalJsonValueLiteral(self.prog, cs_lit, trail_ws);
            return;
        }

        // 3. `enum`
        if (enum_slice) |es| {
            self.prog.rule_entries[rule_id] = try self.compileEnumAlternatives(es, end_pc);
            return;
        }

        // 4. `anyOf` / `oneOf`
        if (any_of_slice) |ao| {
            var entry = try self.compileSubschemaUnion(ao, end_pc, depth);
            if (nullable_flag) {
                const n_pc = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.null_lit, .next = end_pc });
                entry = try self.prog.emitInst(.{ .kind = .branch, .arg = entry, .next = n_pc });
            }
            self.prog.rule_entries[rule_id] = entry;
            return;
        }

        // 5. Infer or parse `"type"` (string or array of type strings e.g. `["string", "null"]`).
        var type_mask: u8 = 0;
        const T_OBJECT: u8 = 1 << 0;
        const T_ARRAY: u8 = 1 << 1;
        const T_STRING: u8 = 1 << 2;
        const T_NUMBER: u8 = 1 << 3;
        const T_INTEGER: u8 = 1 << 4;
        const T_BOOLEAN: u8 = 1 << 5;
        const T_NULL: u8 = 1 << 6;

        if (type_slice) |ts| {
            if (ts.len > 0 and ts[0] == '"') {
                var tbuf: [32]u8 = undefined;
                const tname = try unescapeJsonToken(ts, &tbuf);
                type_mask |= try parseJsonTypeFlag(tname);
            } else if (ts.len > 0 and ts[0] == '[') {
                var ts_scan = SchemaJsonScanner{ .src = ts };
                try ts_scan.expect('[');
                if (ts_scan.peek() == ']') return error.InvalidJsonSchema;
                while (true) {
                    var tbuf: [32]u8 = undefined;
                    const tname = try ts_scan.readUnescapedString(&tbuf);
                    type_mask |= try parseJsonTypeFlag(tname);
                    const sep = ts_scan.peek() orelse return error.InvalidJsonSchema;
                    if (sep == ',') {
                        ts_scan.pos += 1;
                    } else if (sep == ']') {
                        ts_scan.pos += 1;
                        break;
                    } else return error.InvalidJsonSchema;
                }
            } else return error.InvalidJsonSchema;
        } else {
            if (props_slice != null or required_slice != null or additional_props_bool != null or additional_props_schema != null) {
                type_mask |= T_OBJECT;
            } else if (items_slice != null or prefix_items_slice != null) {
                type_mask |= T_ARRAY;
            } else {
                // Unconstrained schema `{}` accepts any valid JSON value.
                self.prog.rule_entries[rule_id] = try self.prog.emitInst(.{
                    .kind = .rule_ref,
                    .arg = self.base.any_value,
                    .next = end_pc,
                });
                return;
            }
        }
        if (nullable_flag) type_mask |= T_NULL;

        var alts: [7]u16 = undefined;
        var alt_count: usize = 0;

        if ((type_mask & T_OBJECT) != 0) {
            const obj_entry = try self.compileObjectBody(
                props_slice,
                required_slice,
                additional_props_bool,
                additional_props_schema,
                end_pc,
                depth,
            );
            alts[alt_count] = obj_entry;
            alt_count += 1;
        }
        if ((type_mask & T_ARRAY) != 0) {
            const arr_entry = try self.compileArrayBody(
                items_slice,
                prefix_items_slice,
                min_items,
                max_items,
                end_pc,
                depth,
            );
            alts[alt_count] = arr_entry;
            alt_count += 1;
        }
        if ((type_mask & T_STRING) != 0) {
            const str_entry = try self.compileStringBody(min_length, max_length, end_pc);
            alts[alt_count] = str_entry;
            alt_count += 1;
        }
        if ((type_mask & T_NUMBER) != 0) {
            alts[alt_count] = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.number, .next = end_pc });
            alt_count += 1;
        } else if ((type_mask & T_INTEGER) != 0) {
            alts[alt_count] = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.integer, .next = end_pc });
            alt_count += 1;
        }
        if ((type_mask & T_BOOLEAN) != 0) {
            alts[alt_count] = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.boolean, .next = end_pc });
            alt_count += 1;
        }
        if ((type_mask & T_NULL) != 0) {
            alts[alt_count] = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.null_lit, .next = end_pc });
            alt_count += 1;
        }

        if (alt_count == 0) return error.InvalidJsonSchema;
        var combined = alts[alt_count - 1];
        var idx = alt_count - 1;
        while (idx > 0) {
            idx -= 1;
            combined = try self.prog.emitInst(.{ .kind = .branch, .arg = alts[idx], .next = combined });
        }
        self.prog.rule_entries[rule_id] = combined;
    }

    fn compileStringBody(self: *SchemaCompiler, min_len: u32, max_len: ?u32, end_pc: u16) CompileError!u16 {
        if (min_len == 0 and max_len == null) {
            return self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.string, .next = end_pc });
        }
        const capped_min = @min(min_len, 32);
        if (max_len) |mx| {
            if (mx < min_len) return error.InvalidJsonSchema;
            if (mx <= 32) {
                const trail_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = end_pc });
                const close_q = try self.prog.emitInst(.{ .kind = .byte, .arg = '"', .next = trail_ws });
                var cur_pc = close_q;
                const opt_extra = mx - capped_min;
                var o: u32 = 0;
                while (o < opt_extra) : (o += 1) {
                    const ch_pc = try emitJsonStringChar(self.prog, cur_pc);
                    cur_pc = try self.prog.emitInst(.{ .kind = .branch, .arg = ch_pc, .next = close_q });
                }
                var req: u32 = 0;
                while (req < capped_min) : (req += 1) {
                    cur_pc = try emitJsonStringChar(self.prog, cur_pc);
                }
                return self.prog.emitInst(.{ .kind = .byte, .arg = '"', .next = cur_pc });
            }
        }
        // Unbounded max with `min_len >= 1`: emit `capped_min` required chars then star loop.
        const trail_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = end_pc });
        const close_q = try self.prog.emitInst(.{ .kind = .byte, .arg = '"', .next = trail_ws });
        const loop_br = try self.prog.emitInst(.{ .kind = .branch, .arg = 0, .next = close_q });
        const loop_ch = try emitJsonStringChar(self.prog, loop_br);
        self.prog.insts[loop_br].arg = loop_ch;
        var cur_pc = loop_br;
        var req: u32 = 0;
        while (req < capped_min) : (req += 1) {
            cur_pc = try emitJsonStringChar(self.prog, cur_pc);
        }
        return self.prog.emitInst(.{ .kind = .byte, .arg = '"', .next = cur_pc });
    }

    fn compileEnumAlternatives(self: *SchemaCompiler, enum_json: []const u8, end_pc: u16) CompileError!u16 {
        var s = SchemaJsonScanner{ .src = enum_json };
        try s.expect('[');
        if (s.peek() == ']') return error.InvalidJsonSchema;

        const trail_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = end_pc });
        var alts: [64]u16 = undefined;
        var count: usize = 0;

        while (true) {
            if (count >= alts.len) return error.ProgramCapacityExceeded;
            const lit = try s.captureValue(2);
            alts[count] = try emitCanonicalJsonValueLiteral(self.prog, lit, trail_ws);
            count += 1;

            const sep = s.peek() orelse return error.InvalidJsonSchema;
            if (sep == ',') {
                s.pos += 1;
            } else if (sep == ']') {
                s.pos += 1;
                break;
            } else return error.InvalidJsonSchema;
        }

        var cur = alts[count - 1];
        var idx = count - 1;
        while (idx > 0) {
            idx -= 1;
            cur = try self.prog.emitInst(.{ .kind = .branch, .arg = alts[idx], .next = cur });
        }
        return cur;
    }

    fn compileSubschemaUnion(self: *SchemaCompiler, arr_json: []const u8, end_pc: u16, depth: usize) CompileError!u16 {
        var s = SchemaJsonScanner{ .src = arr_json };
        try s.expect('[');
        if (s.peek() == ']') return error.InvalidJsonSchema;

        var alts: [32]u16 = undefined;
        var count: usize = 0;
        while (true) {
            if (count >= alts.len) return error.ProgramCapacityExceeded;
            const sub_json = try s.captureValue(depth + 1);
            const sub_rule = try self.compileSchemaNode(sub_json, depth + 1);
            alts[count] = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = sub_rule, .next = end_pc });
            count += 1;

            const sep = s.peek() orelse return error.InvalidJsonSchema;
            if (sep == ',') {
                s.pos += 1;
            } else if (sep == ']') {
                s.pos += 1;
                break;
            } else return error.InvalidJsonSchema;
        }

        var cur = alts[count - 1];
        var idx = count - 1;
        while (idx > 0) {
            idx -= 1;
            cur = try self.prog.emitInst(.{ .kind = .branch, .arg = alts[idx], .next = cur });
        }
        return cur;
    }

    fn compileArrayBody(
        self: *SchemaCompiler,
        items_slice: ?[]const u8,
        prefix_items_slice: ?[]const u8,
        min_items: u32,
        max_items: ?u32,
        end_pc: u16,
        depth: usize,
    ) CompileError!u16 {
        if (max_items) |mx| {
            if (mx < min_items) return error.InvalidJsonSchema;
        }
        if (items_slice == null and prefix_items_slice == null and min_items == 0 and max_items == null) {
            return self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.any_array, .next = end_pc });
        }

        const trail_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = end_pc });
        const close_br = try self.prog.emitInst(.{ .kind = .byte, .arg = ']', .next = trail_ws });

        // Tuple / prefixItems support
        if (prefix_items_slice) |pi_json| {
            var ps = SchemaJsonScanner{ .src = pi_json };
            try ps.expect('[');
            var item_rules: [32]u16 = undefined;
            var pcount: usize = 0;
            if (ps.peek() != ']') {
                while (true) {
                    if (pcount >= item_rules.len) return error.ProgramCapacityExceeded;
                    const sub = try ps.captureValue(depth + 1);
                    item_rules[pcount] = try self.compileSchemaNode(sub, depth + 1);
                    pcount += 1;
                    const sep = ps.peek() orelse return error.InvalidJsonSchema;
                    if (sep == ',') {
                        ps.pos += 1;
                    } else if (sep == ']') {
                        ps.pos += 1;
                        break;
                    } else return error.InvalidJsonSchema;
                }
            } else {
                ps.pos += 1;
            }
            if (pcount == 0) {
                const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = close_br });
                return self.prog.emitInst(.{ .kind = .byte, .arg = '[', .next = open_ws });
            }
            var cur = close_br;
            var k = pcount;
            while (k > 0) {
                k -= 1;
                cur = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = item_rules[k], .next = cur });
                if (k > 0) {
                    cur = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = cur });
                    cur = try self.prog.emitInst(.{ .kind = .byte, .arg = ',', .next = cur });
                }
            }
            const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = cur });
            return self.prog.emitInst(.{ .kind = .byte, .arg = '[', .next = open_ws });
        }

        const item_rule: u16 = if (items_slice) |is|
            try self.compileSchemaNode(is, depth + 1)
        else
            self.base.any_value;

        const capped_min = @min(min_items, 16);
        if (max_items) |mx| {
            if (mx == 0) {
                const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = close_br });
                return self.prog.emitInst(.{ .kind = .byte, .arg = '[', .next = open_ws });
            }
            if (mx <= 16) {
                // Exact bounded count `[min_items .. mx]`
                var cur = close_br;
                const opt_count = mx - capped_min;
                var o: u32 = 0;
                while (o < opt_count) : (o += 1) {
                    const elem_pc = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = item_rule, .next = cur });
                    if (capped_min > 0 or o + 1 < opt_count) {
                        const c_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = elem_pc });
                        const c_pc = try self.prog.emitInst(.{ .kind = .byte, .arg = ',', .next = c_ws });
                        cur = try self.prog.emitInst(.{ .kind = .branch, .arg = c_pc, .next = close_br });
                    } else {
                        cur = try self.prog.emitInst(.{ .kind = .branch, .arg = elem_pc, .next = close_br });
                    }
                }
                if (capped_min > 0) {
                    var r: u32 = capped_min;
                    while (r > 0) {
                        r -= 1;
                        cur = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = item_rule, .next = cur });
                        if (r > 0) {
                            cur = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = cur });
                            cur = try self.prog.emitInst(.{ .kind = .byte, .arg = ',', .next = cur });
                        }
                    }
                }
                const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = cur });
                return self.prog.emitInst(.{ .kind = .byte, .arg = '[', .next = open_ws });
            }
        }

        // General unbounded tail `(',' ws item)*`
        const tail_loop = try self.prog.emitInst(.{ .kind = .branch, .arg = 0, .next = close_br });
        const tail_elem = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = item_rule, .next = tail_loop });
        const tail_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = tail_elem });
        const tail_comma = try self.prog.emitInst(.{ .kind = .byte, .arg = ',', .next = tail_ws });
        self.prog.insts[tail_loop].arg = tail_comma;

        if (capped_min == 0) {
            const first_elem = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = item_rule, .next = tail_loop });
            const opt_first = try self.prog.emitInst(.{ .kind = .branch, .arg = first_elem, .next = close_br });
            const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = opt_first });
            return self.prog.emitInst(.{ .kind = .byte, .arg = '[', .next = open_ws });
        } else {
            var cur = tail_loop;
            var r: u32 = capped_min;
            while (r > 0) {
                r -= 1;
                cur = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = item_rule, .next = cur });
                if (r > 0) {
                    cur = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = cur });
                    cur = try self.prog.emitInst(.{ .kind = .byte, .arg = ',', .next = cur });
                }
            }
            const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = cur });
            return self.prog.emitInst(.{ .kind = .byte, .arg = '[', .next = open_ws });
        }
    }

    fn compileObjectBody(
        self: *SchemaCompiler,
        props_slice: ?[]const u8,
        required_slice: ?[]const u8,
        additional_props_bool: ?bool,
        additional_props_schema: ?[]const u8,
        end_pc: u16,
        depth: usize,
    ) CompileError!u16 {
        const PropSpec = struct {
            key_quoted: []const u8,
            val_rule: u16,
            required: bool,
        };

        var props: [MAX_OBJECT_PROPS]PropSpec = undefined;
        var prop_count: usize = 0;

        var req_names: [MAX_OBJECT_PROPS][]const u8 = undefined;
        var req_count: usize = 0;
        if (required_slice) |rs| {
            var rscan = SchemaJsonScanner{ .src = rs };
            try rscan.expect('[');
            if (rscan.peek() != ']') {
                while (true) {
                    if (req_count >= MAX_OBJECT_PROPS) return error.TooManyProperties;
                    req_names[req_count] = try rscan.captureQuotedString();
                    req_count += 1;
                    const sep = rscan.peek() orelse return error.InvalidJsonSchema;
                    if (sep == ',') {
                        rscan.pos += 1;
                    } else if (sep == ']') {
                        rscan.pos += 1;
                        break;
                    } else return error.InvalidJsonSchema;
                }
            } else {
                rscan.pos += 1;
            }
        }

        if (props_slice) |ps| {
            var pscan = SchemaJsonScanner{ .src = ps };
            try pscan.expect('{');
            if (pscan.peek() != '}') {
                while (true) {
                    if (prop_count >= MAX_OBJECT_PROPS) return error.TooManyProperties;
                    const k_quoted = try pscan.captureQuotedString();
                    try pscan.expect(':');
                    const v_schema = try pscan.captureValue(depth + 1);
                    const v_rule = try self.compileSchemaNode(v_schema, depth + 1);

                    var is_req = false;
                    var k_ubuf: [128]u8 = undefined;
                    const k_plain = try unescapeJsonToken(k_quoted, &k_ubuf);
                    for (req_names[0..req_count]) |rq| {
                        var r_ubuf: [128]u8 = undefined;
                        const r_plain = try unescapeJsonToken(rq, &r_ubuf);
                        if (std.mem.eql(u8, k_plain, r_plain)) {
                            is_req = true;
                            break;
                        }
                    }

                    props[prop_count] = .{
                        .key_quoted = k_quoted,
                        .val_rule = v_rule,
                        .required = is_req,
                    };
                    prop_count += 1;

                    const sep = pscan.peek() orelse return error.InvalidJsonSchema;
                    if (sep == ',') {
                        pscan.pos += 1;
                    } else if (sep == '}') {
                        pscan.pos += 1;
                        break;
                    } else return error.InvalidJsonSchema;
                }
            } else {
                pscan.pos += 1;
            }
        }

        // Also add any required keys that were not listed in `properties` (mapped to `any_value`).
        for (req_names[0..req_count]) |rq| {
            var r_ubuf: [128]u8 = undefined;
            const r_plain = try unescapeJsonToken(rq, &r_ubuf);
            var exists = false;
            for (props[0..prop_count]) |p| {
                var k_ubuf: [128]u8 = undefined;
                const k_plain = try unescapeJsonToken(p.key_quoted, &k_ubuf);
                if (std.mem.eql(u8, k_plain, r_plain)) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                if (prop_count >= MAX_OBJECT_PROPS) return error.TooManyProperties;
                props[prop_count] = .{
                    .key_quoted = rq,
                    .val_rule = self.base.any_value,
                    .required = true,
                };
                prop_count += 1;
            }
        }

        const allow_additional = (additional_props_bool == true) or
            (additional_props_schema != null) or
            (prop_count == 0 and additional_props_bool == null);

        if (prop_count == 0 and allow_additional and additional_props_schema == null) {
            return self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.any_object, .next = end_pc });
        }

        const trail_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = end_pc });
        const close_brace = try self.prog.emitInst(.{ .kind = .byte, .arg = '}', .next = trail_ws });

        // Optional `additionalProperties` tail after known properties.
        var tail_after_last_prop: u16 = close_brace;
        var empty_additional_start: ?u16 = null;
        if (allow_additional) {
            const add_val_rule: u16 = if (additional_props_schema) |aps|
                try self.compileSchemaNode(aps, depth + 1)
            else
                self.base.any_value;

            const add_loop = try self.prog.emitInst(.{ .kind = .branch, .arg = 0, .next = close_brace });
            const add_v = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = add_val_rule, .next = add_loop });
            const add_c_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = add_v });
            const add_colon = try self.prog.emitInst(.{ .kind = .byte, .arg = ':', .next = add_c_ws });
            const add_k = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.string, .next = add_colon });
            const add_comma_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = add_k });
            const add_comma = try self.prog.emitInst(.{ .kind = .byte, .arg = ',', .next = add_comma_ws });
            self.prog.insts[add_loop].arg = add_comma;

            tail_after_last_prop = add_loop;
            empty_additional_start = add_k;
        }

        if (prop_count == 0) {
            // `additionalProperties: false` with 0 properties => empty object `{}`!
            if (empty_additional_start) |eas| {
                const opt_add = try self.prog.emitInst(.{ .kind = .branch, .arg = eas, .next = close_brace });
                const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = opt_add });
                return self.prog.emitInst(.{ .kind = .byte, .arg = '{', .next = open_ws });
            } else {
                const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = close_brace });
                return self.prog.emitInst(.{ .kind = .byte, .arg = '{', .next = open_ws });
            }
        }

        // Build exact schema-ordered DAG for required + optional properties!
        // `after_emit[i]`: continuation PC right after property `i`'s value has been emitted
        //                  (at least one property has now been emitted, so any next prop `j > i` needs `,`).
        // `start_prop_no_comma[i]`: emits `"key_i" ws ':' ws val_i -> after_emit[i]`.
        // `start_prop_with_comma[i]`: emits `',' ws "key_i" ws ':' ws val_i -> after_emit[i]`.
        var start_no_comma: [MAX_OBJECT_PROPS]u16 = undefined;
        var start_with_comma: [MAX_OBJECT_PROPS]u16 = undefined;

        // Backward pass from `i = prop_count - 1` down to `0`:
        // `after_nonempty_chain` represents the continuation from index `i + 1` when at least one
        // earlier property was already emitted (`','` required before the next emitted property).
        var after_nonempty_chain: u16 = tail_after_last_prop;

        var idx = prop_count;
        while (idx > 0) {
            idx -= 1;
            const p = props[idx];
            const val_pc = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = p.val_rule, .next = after_nonempty_chain });
            const colon_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = val_pc });
            const colon = try self.prog.emitInst(.{ .kind = .byte, .arg = ':', .next = colon_ws });
            const key_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = colon });
            const key_lit = try emitLiteral(self.prog, p.key_quoted, key_ws);
            start_no_comma[idx] = key_lit;

            const c_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = key_lit });
            start_with_comma[idx] = try self.prog.emitInst(.{ .kind = .byte, .arg = ',', .next = c_ws });

            if (p.required) {
                after_nonempty_chain = start_with_comma[idx];
            } else {
                after_nonempty_chain = try self.prog.emitInst(.{
                    .kind = .branch,
                    .arg = start_with_comma[idx],
                    .next = after_nonempty_chain,
                });
            }
        }

        // Now build the entry chain when NO property has been emitted yet (`first` property has NO leading comma!):
        var first_chain: u16 = if (empty_additional_start) |eas|
            try self.prog.emitInst(.{ .kind = .branch, .arg = eas, .next = close_brace })
        else
            close_brace;

        idx = prop_count;
        while (idx > 0) {
            idx -= 1;
            if (props[idx].required) {
                first_chain = start_no_comma[idx];
            } else {
                first_chain = try self.prog.emitInst(.{
                    .kind = .branch,
                    .arg = start_no_comma[idx],
                    .next = first_chain,
                });
            }
        }

        const open_ws = try self.prog.emitInst(.{ .kind = .rule_ref, .arg = self.base.ws, .next = first_chain });
        return self.prog.emitInst(.{ .kind = .byte, .arg = '{', .next = open_ws });
    }
};

fn parseJsonTypeFlag(tname: []const u8) CompileError!u8 {
    if (std.mem.eql(u8, tname, "object")) return 1 << 0;
    if (std.mem.eql(u8, tname, "array")) return 1 << 1;
    if (std.mem.eql(u8, tname, "string")) return 1 << 2;
    if (std.mem.eql(u8, tname, "number")) return 1 << 3;
    if (std.mem.eql(u8, tname, "integer")) return 1 << 4;
    if (std.mem.eql(u8, tname, "boolean")) return 1 << 5;
    if (std.mem.eql(u8, tname, "null")) return 1 << 6;
    return error.InvalidJsonSchema;
}

/// Emits a JSON literal (`"str"`, `123`, `true`, `false`, `null`) normalized with no internal ws.
fn emitCanonicalJsonValueLiteral(prog: *GrammarProgram, raw_json_lit: []const u8, after_pc: u16) CompileError!u16 {
    const trimmed = std.mem.trim(u8, raw_json_lit, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidJsonSchema;
    return emitLiteral(prog, trimmed, after_pc);
}

/// Compiles a JSON Schema (`strict: true` structured output or tool argument schema)
/// into `prog` with zero heap allocation.
pub fn compileJsonSchema(prog: *GrammarProgram, schema_json: []const u8) CompileError!void {
    const trimmed = std.mem.trim(u8, schema_json, " \t\r\n");
    if (trimmed.len == 0) return error.EmptyGrammar;

    prog.reset();
    const base = try buildJsonBaseRules(prog);
    var compiler = SchemaCompiler{
        .prog = prog,
        .base = base,
    };
    try compiler.collectRootDefs(trimmed);
    const val_rule = try compiler.compileSchemaNode(trimmed, 0);

    // Ensure full input is consumed cleanly ( scanner rejects trailing garbage in schema_json ).
    var chk = SchemaJsonScanner{ .src = trimmed };
    _ = try chk.captureValue(0);
    if (chk.peek() != null) return error.InvalidJsonSchema;

    const r_root = try prog.allocRule();
    const end_pc = try prog.emitInst(.{ .kind = .end });
    const body_call = try prog.emitInst(.{ .kind = .rule_ref, .arg = val_rule, .next = end_pc });
    const lead_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = body_call });
    prog.rule_entries[r_root] = lead_ws;
    prog.root_rule = r_root;
    prog.finalizeSelfLoops();
}

/// Descriptor for compiling OpenAI / Anthropic strict tool-call envelopes (`<tool_call>...</tool_call>`).
pub const ToolSchemaSpec = struct {
    name: []const u8,
    /// Raw JSON Schema object for `parameters` (OpenAI) or `input_schema` (Anthropic).
    parameters_json: []const u8 = "{\"type\":\"object\"}",
};

/// Compiles one or more strict tool specifications into a Qwen3 `<tool_call>` envelope grammar:
/// `ws ("<think>" [^<]* "</think>" ws)? "<tool_call>" ws '{"name":' ws '"<tool>"' ws ',"arguments":' ws <schema> '}' ws "</tool_call>" (ws "<tool_call>" ...)* ws`
pub fn compileStrictToolCalls(
    prog: *GrammarProgram,
    tools: []const ToolSchemaSpec,
    parallel_tool_calls: bool,
    allow_thinking_prefix: bool,
) CompileError!void {
    if (tools.len == 0) return error.EmptyGrammar;

    prog.reset();
    const base = try buildJsonBaseRules(prog);
    var compiler = SchemaCompiler{
        .prog = prog,
        .base = base,
    };

    const r_single_call = try prog.allocRule();
    const call_end = try prog.emitInst(.{ .kind = .end });
    const close_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = call_end });
    const close_tag = try emitLiteral(prog, "</tool_call>", close_ws);
    const close_obj_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = close_tag });
    const close_obj = try prog.emitInst(.{ .kind = .byte, .arg = '}', .next = close_obj_ws });

    var tool_alts: [32]u16 = undefined;
    if (tools.len > tool_alts.len) return error.RuleCapacityExceeded;

    for (tools, 0..) |tspec, idx| {
        if (tspec.name.len == 0) return error.InvalidJsonSchema;
        compiler.def_count = 0;
        const p_json = if (tspec.parameters_json.len > 0) tspec.parameters_json else "{\"type\":\"object\"}";
        try compiler.collectRootDefs(p_json);
        const args_rule = try compiler.compileSchemaNode(p_json, 0);

        const args_call = try prog.emitInst(.{ .kind = .rule_ref, .arg = args_rule, .next = close_obj });
        const args_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = args_call });
        const args_key = try emitLiteral(prog, "\"arguments\":", args_ws);
        const comma_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = args_key });
        const comma = try prog.emitInst(.{ .kind = .byte, .arg = ',', .next = comma_ws });
        const name_end_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = comma });
        const name_q2 = try prog.emitInst(.{ .kind = .byte, .arg = '"', .next = name_end_ws });
        const name_bytes = try emitLiteral(prog, tspec.name, name_q2);
        const name_q1 = try prog.emitInst(.{ .kind = .byte, .arg = '"', .next = name_bytes });
        const name_colon_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = name_q1 });
        const name_key = try emitLiteral(prog, "\"name\":", name_colon_ws);
        tool_alts[idx] = name_key;
    }

    var body_union = tool_alts[tools.len - 1];
    var i = tools.len - 1;
    while (i > 0) {
        i -= 1;
        body_union = try prog.emitInst(.{ .kind = .branch, .arg = tool_alts[i], .next = body_union });
    }

    const open_obj_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = body_union });
    const open_obj = try prog.emitInst(.{ .kind = .byte, .arg = '{', .next = open_obj_ws });
    const open_tag_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = open_obj });
    const open_tag = try emitLiteral(prog, "<tool_call>", open_tag_ws);
    prog.rule_entries[r_single_call] = open_tag;

    const r_root = try prog.allocRule();
    const root_end = try prog.emitInst(.{ .kind = .end });

    var calls_entry: u16 = undefined;
    if (parallel_tool_calls) {
        const more_loop = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = root_end });
        const next_call = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_single_call, .next = more_loop });
        prog.insts[more_loop].arg = next_call;
        calls_entry = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_single_call, .next = more_loop });
    } else {
        calls_entry = try prog.emitInst(.{ .kind = .rule_ref, .arg = r_single_call, .next = root_end });
    }

    if (allow_thinking_prefix) {
        var non_lt = ByteSet.all;
        non_lt.remove('<');
        const cs_non_lt = try prog.internCharset(non_lt);

        const after_think_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = calls_entry });
        const close_think = try emitLiteral(prog, "</think>", after_think_ws);
        const think_loop = try prog.emitInst(.{ .kind = .branch, .arg = 0, .next = close_think });
        const think_ch = try prog.emitInst(.{ .kind = .charset, .arg = cs_non_lt, .next = think_loop });
        prog.insts[think_loop].arg = think_ch;
        const open_think = try emitLiteral(prog, "<think>", think_loop);
        calls_entry = try prog.emitInst(.{ .kind = .branch, .arg = open_think, .next = calls_entry });
    }

    const root_ws = try prog.emitInst(.{ .kind = .rule_ref, .arg = base.ws, .next = calls_entry });
    prog.rule_entries[r_root] = root_ws;
    prog.root_rule = r_root;
    prog.finalizeSelfLoops();
}

// ---------------------------------------------------------------------------
// 3. Zero-Allocation GBNF Compiler (`root ::= ...`)
// ---------------------------------------------------------------------------

const GbnfRuleSlot = struct {
    name_buf: [MAX_GBNF_RULE_NAME]u8 = undefined,
    name_len: u8 = 0,
    defined: bool = false,
    rule_id: u16 = INVALID_RULE,

    fn nameSlice(self: *const GbnfRuleSlot) []const u8 {
        return self.name_buf[0..self.name_len];
    }
};

const GbnfCompiler = struct {
    src: []const u8,
    pos: usize = 0,
    prog: *GrammarProgram,
    rules: [MAX_RULES]GbnfRuleSlot = undefined,
    rule_map_count: usize = 0,

    fn isIdentChar(c: u8) bool {
        return std.ascii.isAlphanumeric(c) or c == '-' or c == '_';
    }

    fn skipSpace(self: *GbnfCompiler, skip_newlines: bool) void {
        while (self.pos < self.src.len) {
            const b = self.src[self.pos];
            if (b == ' ' or b == '\t' or (skip_newlines and (b == '\n' or b == '\r'))) {
                self.pos += 1;
            } else if (b == '#') {
                while (self.pos < self.src.len and self.src[self.pos] != '\n') : (self.pos += 1) {}
            } else {
                return;
            }
        }
    }

    fn internRule(self: *GbnfCompiler, name: []const u8) CompileError!usize {
        if (name.len == 0 or name.len > MAX_GBNF_RULE_NAME) return error.InvalidSyntax;
        for (self.rules[0..self.rule_map_count], 0..) |*slot, i| {
            if (std.mem.eql(u8, slot.nameSlice(), name)) return i;
        }
        if (self.rule_map_count >= MAX_RULES) return error.RuleCapacityExceeded;
        const r_id = try self.prog.allocRule();
        const idx = self.rule_map_count;
        self.rules[idx] = .{
            .name_len = @intCast(name.len),
            .defined = false,
            .rule_id = r_id,
        };
        @memcpy(self.rules[idx].name_buf[0..name.len], name);
        self.rule_map_count += 1;
        return idx;
    }

    fn readIdent(self: *GbnfCompiler) []const u8 {
        const start = self.pos;
        while (self.pos < self.src.len and isIdentChar(self.src[self.pos])) : (self.pos += 1) {}
        return self.src[start..self.pos];
    }

    /// Lookahead check whether an identifier is followed by `::=` (starts the next rule!).
    fn isRuleDefinitionAhead(self: *const GbnfCompiler, after_ident_pos: usize) bool {
        var p = after_ident_pos;
        while (p < self.src.len) {
            const c = self.src[p];
            if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
                p += 1;
            } else if (c == '#') {
                while (p < self.src.len and self.src[p] != '\n') : (p += 1) {}
            } else break;
        }
        return (self.src.len - p >= 3 and std.mem.eql(u8, self.src[p .. p + 3], "::="));
    }

    fn readEscapedCodepoint(self: *GbnfCompiler) CompileError!u21 {
        if (self.pos >= self.src.len) return error.InvalidSyntax;
        const b = self.src[self.pos];
        self.pos += 1;
        if (b != '\\') {
            // Decode UTF-8 codepoint starting at `b`
            const seq_len = std.unicode.utf8ByteSequenceLength(b) catch return b;
            if (seq_len == 1) return b;
            if (self.src.len - (self.pos - 1) < seq_len) return error.InvalidSyntax;
            const slice = self.src[self.pos - 1 .. self.pos - 1 + seq_len];
            const cp = std.unicode.utf8Decode(slice) catch return error.InvalidSyntax;
            self.pos += seq_len - 1;
            return cp;
        }
        if (self.pos >= self.src.len) return error.InvalidSyntax;
        const esc = self.src[self.pos];
        self.pos += 1;
        return switch (esc) {
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            '"' => '"',
            '\\' => '\\',
            '[' => '[',
            ']' => ']',
            'x' => blk: {
                if (self.src.len - self.pos < 2) return error.InvalidSyntax;
                const v = std.fmt.parseInt(u8, self.src[self.pos .. self.pos + 2], 16) catch return error.InvalidSyntax;
                self.pos += 2;
                break :blk v;
            },
            'u' => blk: {
                if (self.src.len - self.pos < 4) return error.InvalidSyntax;
                const v = std.fmt.parseInt(u21, self.src[self.pos .. self.pos + 4], 16) catch return error.InvalidSyntax;
                self.pos += 4;
                break :blk v;
            },
            'U' => blk: {
                if (self.src.len - self.pos < 8) return error.InvalidSyntax;
                const v = std.fmt.parseInt(u21, self.src[self.pos .. self.pos + 8], 16) catch return error.InvalidSyntax;
                self.pos += 8;
                break :blk v;
            },
            else => esc,
        };
    }

    /// Parses alternation `seq ('|' seq)*` where continuation ends at `after_pc`.
    fn parseAlternation(self: *GbnfCompiler, after_pc: u16, inside_parens: bool, depth: usize) CompileError!u16 {
        if (depth > MAX_SCHEMA_DEPTH) return error.NestingDepthExceeded;
        var alts: [32]u16 = undefined;
        var count: usize = 0;

        while (true) {
            if (count >= alts.len) return error.ProgramCapacityExceeded;
            alts[count] = try self.parseSequence(after_pc, inside_parens, depth + 1);
            count += 1;

            self.skipSpace(true);
            if (self.pos < self.src.len and self.src[self.pos] == '|') {
                self.pos += 1;
                continue;
            }
            break;
        }

        var cur = alts[count - 1];
        var idx = count - 1;
        while (idx > 0) {
            idx -= 1;
            cur = try self.prog.emitInst(.{ .kind = .branch, .arg = alts[idx], .next = cur });
        }
        return cur;
    }

    fn parseSequence(self: *GbnfCompiler, after_pc: u16, inside_parens: bool, depth: usize) CompileError!u16 {
        // Parse atoms left-to-right by capturing their source spans, then emit right-to-left
        // so each atom directly receives its exact `next_pc` without patch lists!
        const AtomSpan = struct {
            start_pos: usize,
            end_pos: usize,
        };
        var spans: [64]AtomSpan = undefined;
        var span_count: usize = 0;

        while (true) {
            self.skipSpace(inside_parens);
            if (!inside_parens) {
                // Newlines inside a top-level rule are allowed if not starting a new `rule ::=` or EOF.
                const saved = self.pos;
                self.skipSpace(true);
                if (self.pos >= self.src.len) break;
                const ch = self.src[self.pos];
                if (ch == '|') {
                    break;
                }
                if (isIdentChar(ch)) {
                    var p = self.pos;
                    while (p < self.src.len and isIdentChar(self.src[p])) : (p += 1) {}
                    if (self.isRuleDefinitionAhead(p)) {
                        self.pos = saved;
                        break;
                    }
                }
            } else {
                if (self.pos >= self.src.len) return error.InvalidSyntax;
                const ch = self.src[self.pos];
                if (ch == ')' or ch == '|') break;
            }

            if (self.pos >= self.src.len) break;
            const ch = self.src[self.pos];
            if (ch == '|' or (inside_parens and ch == ')')) break;

            if (span_count >= spans.len) return error.ProgramCapacityExceeded;
            const atom_start = self.pos;
            try self.skipAtomWithQuantifier(depth + 1);
            spans[span_count] = .{ .start_pos = atom_start, .end_pos = self.pos };
            span_count += 1;
        }

        const end_after_seq = self.pos;
        var next_pc = after_pc;
        var k = span_count;
        while (k > 0) {
            k -= 1;
            self.pos = spans[k].start_pos;
            next_pc = try self.emitAtomWithQuantifier(next_pc, depth + 1);
        }
        self.pos = end_after_seq;
        return next_pc;
    }

    fn skipAtomWithQuantifier(self: *GbnfCompiler, depth: usize) CompileError!void {
        if (depth > MAX_SCHEMA_DEPTH * 2) return error.NestingDepthExceeded;
        if (self.pos >= self.src.len) return error.InvalidSyntax;
        const ch = self.src[self.pos];
        switch (ch) {
            '"' => {
                self.pos += 1;
                while (self.pos < self.src.len and self.src[self.pos] != '"') {
                    _ = try self.readEscapedCodepoint();
                }
                if (self.pos >= self.src.len or self.src[self.pos] != '"') return error.InvalidSyntax;
                self.pos += 1;
            },
            '[' => {
                self.pos += 1;
                if (self.pos < self.src.len and self.src[self.pos] == '^') self.pos += 1;
                if (self.pos >= self.src.len or self.src[self.pos] == ']') return error.InvalidSyntax;
                while (self.pos < self.src.len and self.src[self.pos] != ']') {
                    _ = try self.readEscapedCodepoint();
                    if (self.pos + 1 < self.src.len and self.src[self.pos] == '-' and self.src[self.pos + 1] != ']') {
                        self.pos += 1;
                        _ = try self.readEscapedCodepoint();
                    }
                }
                if (self.pos >= self.src.len or self.src[self.pos] != ']') return error.InvalidSyntax;
                self.pos += 1;
            },
            '(' => {
                self.pos += 1;
                while (self.pos < self.src.len) {
                    self.skipSpace(true);
                    if (self.pos < self.src.len and self.src[self.pos] == ')') {
                        self.pos += 1;
                        break;
                    }
                    if (self.pos < self.src.len and self.src[self.pos] == '|') {
                        self.pos += 1;
                        continue;
                    }
                    try self.skipAtomWithQuantifier(depth + 1);
                } else return error.InvalidSyntax;
            },
            '.' => self.pos += 1,
            else => {
                if (!isIdentChar(ch)) return error.InvalidSyntax;
                _ = self.readIdent();
            },
        }

        // Optional quantifier `*`, `+`, `?`, `{m}`, `{m,n}`, `{m,}`
        self.skipSpace(false);
        if (self.pos < self.src.len) {
            switch (self.src[self.pos]) {
                '*', '+', '?' => self.pos += 1,
                '{' => {
                    self.pos += 1;
                    while (self.pos < self.src.len and self.src[self.pos] != '}') : (self.pos += 1) {}
                    if (self.pos >= self.src.len) return error.InvalidSyntax;
                    self.pos += 1;
                },
                else => {},
            }
        }
    }

    fn emitSingleAtom(self: *GbnfCompiler, atom_slice: []const u8, after_pc: u16, depth: usize) CompileError!u16 {
        var sub = GbnfCompiler{
            .src = atom_slice,
            .pos = 0,
            .prog = self.prog,
            .rules = self.rules,
            .rule_map_count = self.rule_map_count,
        };
        defer {
            self.rules = sub.rules;
            self.rule_map_count = sub.rule_map_count;
        }

        const ch = sub.src[0];
        switch (ch) {
            '"' => {
                sub.pos = 1;
                var buf: [256]u8 = undefined;
                var len: usize = 0;
                while (sub.pos < sub.src.len and sub.src[sub.pos] != '"') {
                    const cp = try sub.readEscapedCodepoint();
                    if (cp <= 0xFF and sub.src[0] == '"' and cp < 0x80) {
                        if (len >= buf.len) return error.ProgramCapacityExceeded;
                        buf[len] = @intCast(cp);
                        len += 1;
                    } else {
                        var ubuf: [4]u8 = undefined;
                        const un = std.unicode.utf8Encode(cp, &ubuf) catch return error.InvalidSyntax;
                        if (len + un > buf.len) return error.ProgramCapacityExceeded;
                        @memcpy(buf[len .. len + un], ubuf[0..un]);
                        len += un;
                    }
                }
                return emitLiteral(self.prog, buf[0..len], after_pc);
            },
            '[' => {
                sub.pos = 1;
                var negated = false;
                if (sub.pos < sub.src.len and sub.src[sub.pos] == '^') {
                    negated = true;
                    sub.pos += 1;
                }
                var bset = ByteSet.empty;
                while (sub.pos < sub.src.len and sub.src[sub.pos] != ']') {
                    const lo = try sub.readEscapedCodepoint();
                    if (sub.pos + 1 < sub.src.len and sub.src[sub.pos] == '-' and sub.src[sub.pos + 1] != ']') {
                        sub.pos += 1;
                        const hi = try sub.readEscapedCodepoint();
                        if (lo > hi) return error.InvalidSyntax;
                        if (hi <= 0x7F) {
                            bset.insertRange(@intCast(lo), @intCast(hi));
                        } else if (lo <= 0xFF and hi <= 0xFF) {
                            bset.insertRange(@intCast(lo), @intCast(hi));
                        } else {
                            // Non-ASCII codepoint range: permit valid multi-byte UTF-8 when negated/wide.
                            bset.insertRange(0x80, 0xF4);
                        }
                    } else {
                        if (lo <= 0xFF) {
                            bset.insert(@intCast(lo));
                        } else {
                            bset.insertRange(0x80, 0xF4);
                        }
                    }
                }
                if (negated) {
                    bset.invert();
                    // Never match invalid UTF-8 0xF5..0xFF or NUL unless explicitly positive.
                    bset.remove(0x00);
                    bset.remove(0xFF);
                    // For negated ASCII classes like `[^"\\]`, multi-byte UTF-8 sequences are valid:
                    // convert negated class into full valid UTF-8 codepoint OR single matching ASCII byte!
                    var ascii_only = ByteSet.empty;
                    for (0..128) |bi| {
                        const b8: u8 = @intCast(bi);
                        if (bset.contains(b8)) ascii_only.insert(b8);
                    }
                    const cs_ascii = try self.prog.internCharset(ascii_only);
                    const p_ascii = try self.prog.emitInst(.{ .kind = .charset, .arg = cs_ascii, .next = after_pc });

                    var cont_set = ByteSet.empty;
                    cont_set.insertRange(0x80, 0xBF);
                    const cs_cont = try self.prog.internCharset(cont_set);
                    const p_c1 = try self.prog.emitInst(.{ .kind = .charset, .arg = cs_cont, .next = after_pc });
                    const p_c2 = try self.prog.emitInst(.{ .kind = .charset, .arg = cs_cont, .next = p_c1 });
                    const p_c3 = try self.prog.emitInst(.{ .kind = .charset, .arg = cs_cont, .next = p_c2 });

                    var l2 = ByteSet.empty;
                    l2.insertRange(0xC2, 0xDF);
                    const p_u2 = try self.prog.emitInst(.{ .kind = .charset, .arg = try self.prog.internCharset(l2), .next = p_c1 });

                    var l3 = ByteSet.empty;
                    l3.insertRange(0xE0, 0xEF);
                    const p_u3 = try self.prog.emitInst(.{ .kind = .charset, .arg = try self.prog.internCharset(l3), .next = p_c2 });

                    var l4 = ByteSet.empty;
                    l4.insertRange(0xF0, 0xF4);
                    const p_u4 = try self.prog.emitInst(.{ .kind = .charset, .arg = try self.prog.internCharset(l4), .next = p_c3 });

                    const b_34 = try self.prog.emitInst(.{ .kind = .branch, .arg = p_u3, .next = p_u4 });
                    const b_234 = try self.prog.emitInst(.{ .kind = .branch, .arg = p_u2, .next = b_34 });
                    return self.prog.emitInst(.{ .kind = .branch, .arg = p_ascii, .next = b_234 });
                }
                const cs = try self.prog.internCharset(bset);
                return self.prog.emitInst(.{ .kind = .charset, .arg = cs, .next = after_pc });
            },
            '(' => {
                // Strip outer `(` and `)`
                const inner = atom_slice[1 .. atom_slice.len - 1];
                var grp = GbnfCompiler{
                    .src = inner,
                    .pos = 0,
                    .prog = self.prog,
                    .rules = sub.rules,
                    .rule_map_count = sub.rule_map_count,
                };
                const entry = try grp.parseAlternation(after_pc, false, depth + 1);
                sub.rules = grp.rules;
                sub.rule_map_count = grp.rule_map_count;
                return entry;
            },
            '.' => {
                var any_ascii = ByteSet.empty;
                any_ascii.insertRange(0x01, 0x7F);
                const cs_a = try self.prog.internCharset(any_ascii);
                const p_a = try self.prog.emitInst(.{ .kind = .charset, .arg = cs_a, .next = after_pc });

                var cont = ByteSet.empty;
                cont.insertRange(0x80, 0xBF);
                const cs_c = try self.prog.internCharset(cont);
                const c1 = try self.prog.emitInst(.{ .kind = .charset, .arg = cs_c, .next = after_pc });
                const c2 = try self.prog.emitInst(.{ .kind = .charset, .arg = cs_c, .next = c1 });
                const c3 = try self.prog.emitInst(.{ .kind = .charset, .arg = cs_c, .next = c2 });

                var l2 = ByteSet.empty;
                l2.insertRange(0xC2, 0xDF);
                const u2_p = try self.prog.emitInst(.{ .kind = .charset, .arg = try self.prog.internCharset(l2), .next = c1 });
                var l3 = ByteSet.empty;
                l3.insertRange(0xE0, 0xEF);
                const u3_p = try self.prog.emitInst(.{ .kind = .charset, .arg = try self.prog.internCharset(l3), .next = c2 });
                var l4 = ByteSet.empty;
                l4.insertRange(0xF0, 0xF4);
                const u4_p = try self.prog.emitInst(.{ .kind = .charset, .arg = try self.prog.internCharset(l4), .next = c3 });

                const b2 = try self.prog.emitInst(.{ .kind = .branch, .arg = u3_p, .next = u4_p });
                const b1 = try self.prog.emitInst(.{ .kind = .branch, .arg = u2_p, .next = b2 });
                return self.prog.emitInst(.{ .kind = .branch, .arg = p_a, .next = b1 });
            },
            else => {
                const slot_idx = try sub.internRule(atom_slice);
                const r_id = sub.rules[slot_idx].rule_id;
                return self.prog.emitInst(.{ .kind = .rule_ref, .arg = r_id, .next = after_pc });
            },
        }
    }

    fn emitAtomWithQuantifier(self: *GbnfCompiler, after_pc: u16, depth: usize) CompileError!u16 {
        const atom_start = self.pos;
        // Skip base atom only
        const ch = self.src[self.pos];
        switch (ch) {
            '"' => {
                self.pos += 1;
                while (self.pos < self.src.len and self.src[self.pos] != '"') {
                    _ = try self.readEscapedCodepoint();
                }
                self.pos += 1;
            },
            '[' => {
                self.pos += 1;
                if (self.pos < self.src.len and self.src[self.pos] == '^') self.pos += 1;
                while (self.pos < self.src.len and self.src[self.pos] != ']') {
                    _ = try self.readEscapedCodepoint();
                    if (self.pos + 1 < self.src.len and self.src[self.pos] == '-' and self.src[self.pos + 1] != ']') {
                        self.pos += 1;
                        _ = try self.readEscapedCodepoint();
                    }
                }
                self.pos += 1;
            },
            '(' => {
                var pdepth: usize = 1;
                self.pos += 1;
                while (self.pos < self.src.len and pdepth > 0) {
                    const c = self.src[self.pos];
                    if (c == '"') {
                        self.pos += 1;
                        while (self.pos < self.src.len and self.src[self.pos] != '"') {
                            _ = try self.readEscapedCodepoint();
                        }
                        self.pos += 1;
                    } else if (c == '[') {
                        self.pos += 1;
                        while (self.pos < self.src.len and self.src[self.pos] != ']') {
                            _ = try self.readEscapedCodepoint();
                        }
                        self.pos += 1;
                    } else if (c == '(') {
                        pdepth += 1;
                        self.pos += 1;
                    } else if (c == ')') {
                        pdepth -= 1;
                        self.pos += 1;
                    } else {
                        self.pos += 1;
                    }
                }
            },
            '.' => self.pos += 1,
            else => _ = self.readIdent(),
        }
        const atom_slice = self.src[atom_start..self.pos];

        self.skipSpace(false);
        if (self.pos >= self.src.len) {
            return self.emitSingleAtom(atom_slice, after_pc, depth);
        }

        switch (self.src[self.pos]) {
            '?' => {
                self.pos += 1;
                const inner = try self.emitSingleAtom(atom_slice, after_pc, depth);
                return self.prog.emitInst(.{ .kind = .branch, .arg = inner, .next = after_pc });
            },
            '*' => {
                self.pos += 1;
                const loop_br = try self.prog.emitInst(.{ .kind = .branch, .arg = 0, .next = after_pc });
                const inner = try self.emitSingleAtom(atom_slice, loop_br, depth);
                self.prog.insts[loop_br].arg = inner;
                return loop_br;
            },
            '+' => {
                self.pos += 1;
                const loop_br = try self.prog.emitInst(.{ .kind = .branch, .arg = 0, .next = after_pc });
                const inner_more = try self.emitSingleAtom(atom_slice, loop_br, depth);
                self.prog.insts[loop_br].arg = inner_more;
                return self.emitSingleAtom(atom_slice, loop_br, depth);
            },
            '{' => {
                self.pos += 1;
                const q_start = self.pos;
                while (self.pos < self.src.len and self.src[self.pos] != '}') : (self.pos += 1) {}
                const q_spec = std.mem.trim(u8, self.src[q_start..self.pos], " \t");
                self.pos += 1;

                var min_rep: u32 = 0;
                var max_rep: ?u32 = null;
                if (std.mem.indexOfScalar(u8, q_spec, ',')) |comma_idx| {
                    const left = std.mem.trim(u8, q_spec[0..comma_idx], " \t");
                    const right = std.mem.trim(u8, q_spec[comma_idx + 1 ..], " \t");
                    min_rep = if (left.len > 0) (std.fmt.parseInt(u32, left, 10) catch return error.InvalidSyntax) else 0;
                    max_rep = if (right.len > 0) (std.fmt.parseInt(u32, right, 10) catch return error.InvalidSyntax) else null;
                } else {
                    const exact = std.fmt.parseInt(u32, q_spec, 10) catch return error.InvalidSyntax;
                    min_rep = exact;
                    max_rep = exact;
                }
                if (min_rep > 32) return error.ProgramCapacityExceeded;
                if (max_rep) |mx| {
                    if (mx < min_rep or mx > 32) return error.InvalidSyntax;
                }

                var cur_pc: u16 = after_pc;
                if (max_rep) |mx| {
                    var o: u32 = 0;
                    while (o < mx - min_rep) : (o += 1) {
                        const step_pc = try self.emitSingleAtom(atom_slice, cur_pc, depth);
                        cur_pc = try self.prog.emitInst(.{ .kind = .branch, .arg = step_pc, .next = after_pc });
                    }
                } else {
                    const loop_br = try self.prog.emitInst(.{ .kind = .branch, .arg = 0, .next = after_pc });
                    const step_pc = try self.emitSingleAtom(atom_slice, loop_br, depth);
                    self.prog.insts[loop_br].arg = step_pc;
                    cur_pc = loop_br;
                }

                var r: u32 = 0;
                while (r < min_rep) : (r += 1) {
                    cur_pc = try self.emitSingleAtom(atom_slice, cur_pc, depth);
                }
                return cur_pc;
            },
            else => return self.emitSingleAtom(atom_slice, after_pc, depth),
        }
    }
};

/// Compiles a GBNF grammar string (`root ::= ...`) into `prog` with zero heap allocation.
pub fn compileGbnf(prog: *GrammarProgram, gbnf_text: []const u8) CompileError!void {
    prog.reset();
    var c = GbnfCompiler{
        .src = gbnf_text,
        .prog = prog,
    };

    var defined_any = false;
    while (true) {
        c.skipSpace(true);
        if (c.pos >= c.src.len) break;

        const rname = c.readIdent();
        if (rname.len == 0) return error.InvalidSyntax;
        c.skipSpace(true);
        if (c.src.len - c.pos < 3 or !std.mem.eql(u8, c.src[c.pos .. c.pos + 3], "::=")) {
            return error.InvalidSyntax;
        }
        c.pos += 3;

        const slot_idx = try c.internRule(rname);
        if (c.rules[slot_idx].defined) return error.InvalidSyntax;
        c.rules[slot_idx].defined = true;

        const r_id = c.rules[slot_idx].rule_id;
        const end_pc = try prog.emitInst(.{ .kind = .end });
        const entry_pc = try c.parseAlternation(end_pc, false, 0);
        prog.rule_entries[r_id] = entry_pc;
        defined_any = true;
    }

    if (!defined_any) return error.EmptyGrammar;

    var root_rule_id: ?u16 = null;
    for (c.rules[0..c.rule_map_count]) |slot| {
        if (!slot.defined) return error.UnknownRuleReference;
        if (std.mem.eql(u8, slot.nameSlice(), "root")) {
            root_rule_id = slot.rule_id;
        }
    }
    prog.root_rule = root_rule_id orelse return error.UnknownRuleReference;
    prog.finalizeSelfLoops();
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

const MallocStatistics = extern struct {
    blocks_in_use: c_uint,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};
extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;

test "C22 json_object grammar validates escapes, UTF-8, numbers, arrays, and rejects invalid JSON" {
    var prog: GrammarProgram = undefined;
    try compileJsonObject(&prog);

    try std.testing.expect(prog.matchesComplete("{}"));
    try std.testing.expect(prog.matchesComplete("  { \"a\" : 123, \"b\": -0.45e+2, \"c\": [true, false, null, {\"nested\": \"hi\\n\\u4f60\"}] } "));
    try std.testing.expect(prog.matchesComplete("{\"cjk\":\"你好世界🚀\"}"));

    try std.testing.expect(!prog.matchesComplete(""));
    try std.testing.expect(!prog.matchesComplete("[]"));
    try std.testing.expect(!prog.matchesComplete("\"string\""));
    try std.testing.expect(!prog.matchesComplete("{\"a\": 01}"));
    try std.testing.expect(!prog.matchesComplete("{\"a\": 1,}"));
    try std.testing.expect(!prog.matchesComplete("{\"a\": \"unterminated}"));
}

test "C22 json_schema strict compiler handles required/optional keys, enums, nested $defs/$ref, arrays, and types" {
    var prog: GrammarProgram = undefined;
    const schema =
        \\{
        \\  "$defs": {
        \\    "Address": {
        \\      "type": "object",
        \\      "properties": {
        \\        "city": {"type": "string", "minLength": 1},
        \\        "zip": {"type": ["integer", "null"]}
        \\      },
        \\      "required": ["city", "zip"],
        \\      "additionalProperties": false
        \\    }
        \\  },
        \\  "type": "object",
        \\  "properties": {
        \\    "id": {"type": "integer"},
        \\    "role": {"enum": ["admin", "viewer"]},
        \\    "nickname": {"type": "string"},
        \\    "active": {"type": "boolean"},
        \\    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        \\    "address": {"$ref": "#/$defs/Address"}
        \\  },
        \\  "required": ["id", "role", "active", "tags", "address"],
        \\  "additionalProperties": false
        \\}
    ;
    try compileJsonSchema(&prog, schema);

    // Valid with optional `nickname` omitted
    try std.testing.expect(prog.matchesComplete(
        \\{"id": 42, "role": "admin", "active": true, "tags": ["gpu", "ane"], "address": {"city": "Tokyo", "zip": null}}
    ));
    // Valid with optional `nickname` present
    try std.testing.expect(prog.matchesComplete(
        \\{"id": -7, "role": "viewer", "nickname": "akari", "active": false, "tags": ["sme2"], "address": {"city": "Osaka", "zip": 5300001}}
    ));

    // Reject float for integer `id`
    try std.testing.expect(!prog.matchesComplete(
        \\{"id": 42.5, "role": "admin", "active": true, "tags": ["gpu"], "address": {"city": "Tokyo", "zip": null}}
    ));
    // Reject invalid enum value
    try std.testing.expect(!prog.matchesComplete(
        \\{"id": 42, "role": "guest", "active": true, "tags": ["gpu"], "address": {"city": "Tokyo", "zip": null}}
    ));
    // Reject empty `tags` (`minItems: 1`)
    try std.testing.expect(!prog.matchesComplete(
        \\{"id": 42, "role": "admin", "active": true, "tags": [], "address": {"city": "Tokyo", "zip": null}}
    ));
    // Reject > 3 `tags` (`maxItems: 3`)
    try std.testing.expect(!prog.matchesComplete(
        \\{"id": 42, "role": "admin", "active": true, "tags": ["a","b","c","d"], "address": {"city": "Tokyo", "zip": null}}
    ));
    // Reject empty `city` (`minLength: 1`)
    try std.testing.expect(!prog.matchesComplete(
        \\{"id": 42, "role": "admin", "active": true, "tags": ["a"], "address": {"city": "", "zip": 1}}
    ));
    // Reject undeclared extra key (`additionalProperties: false`)
    try std.testing.expect(!prog.matchesComplete(
        \\{"id": 42, "role": "admin", "active": true, "tags": ["a"], "address": {"city": "NY", "zip": 1}, "extra": 1}
    ));
}

test "C22 strict tool-call envelope grammar and GBNF compiler" {
    var prog: GrammarProgram = undefined;
    const tools = [_]ToolSchemaSpec{
        .{
            .name = "get_weather",
            .parameters_json =
            \\{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["location","unit"],"additionalProperties":false}
            ,
        },
        .{
            .name = "read_clock",
            .parameters_json =
            \\{"type":"object","properties":{},"additionalProperties":false}
            ,
        },
    };
    try compileStrictToolCalls(&prog, &tools, true, true);

    try std.testing.expect(prog.matchesComplete(
        "<think>Need weather</think>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Kyoto\", \"unit\": \"celsius\"}}\n</tool_call>",
    ));
    try std.testing.expect(prog.matchesComplete(
        "<tool_call>{\"name\":\"read_clock\",\"arguments\":{}}</tool_call>\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"SF\",\"unit\":\"fahrenheit\"}}</tool_call>",
    ));
    try std.testing.expect(!prog.matchesComplete(
        "<tool_call>{\"name\":\"unknown_fn\",\"arguments\":{}}</tool_call>",
    ));

    // GBNF grammar test: SQL-like SELECT subset with quantifiers, negation, grouping, rule refs
    const gbnf =
        \\# Simple arithmetic / assignment GBNF
        \\root   ::= stmt ("\n" stmt)*
        \\stmt   ::= ident ws "=" ws expr
        \\expr   ::= term (ws [+-] ws term)*
        \\term   ::= [0-9]+ | ident | "(" ws expr ws ")"
        \\ident  ::= [a-zA-Z_] [a-zA-Z0-9_]{0,7}
        \\ws     ::= [ \t]*
    ;
    try compileGbnf(&prog, gbnf);
    try std.testing.expect(prog.matchesComplete("x = 42 + (y_1 - 7)\nz = x"));
    try std.testing.expect(!prog.matchesComplete("123 = x"));
    try std.testing.expect(!prog.matchesComplete("x = 42 + "));
}

test "C22 VocabIndex 248,077-token bitmask (Qwen3.8 scale), O(1) branch snapshot/restore, and zero heap alloc" {
    const qwen38_vocab: usize = 248_077;
    const slices = try std.testing.allocator.alloc(?[]const u8, qwen38_vocab);
    defer std.testing.allocator.free(slices);

    const static_tokens = [_][]const u8{
        "{",         "}",         "\"",         ":",
        ",",         " ",         "\"status\"", "\"ok\"",
        "\"error\"", "\"count\"", "0",          "1",
        "42",        "true",      "false",      "null",
        "hello",     "world",     "invalid}",   "[",
    };
    for (0..qwen38_vocab) |i| {
        if (i == 0) {
            slices[i] = null; // special EOS token 0
        } else if (i - 1 < static_tokens.len) {
            slices[i] = static_tokens[i - 1];
        } else {
            // Populate full 248,077 vocabulary with realistic ASCII word tokens
            slices[i] = static_tokens[16 + (i & 1)]; // "hello" / "world"
        }
    }

    const eos = [_]u32{0};
    var vindex = try VocabIndex.initFromSlices(std.testing.allocator, slices, &eos);
    defer vindex.deinit(std.testing.allocator);

    const mask = try std.testing.allocator.alloc(u32, MAX_BITMASK_WORDS);
    defer std.testing.allocator.free(mask);

    // 1. Null grammar bypass allows all 248,077 tokens.
    const empty_state = GrammarState{};
    vindex.fillAllowedTokenBitmask(null, &empty_state, mask);
    try std.testing.expectEqual(qwen38_vocab, countAllowedTokens(mask, qwen38_vocab));

    // 2. Compile strict schema `{"status": "ok"|"error", "count": integer}`
    var prog = try std.testing.allocator.create(GrammarProgram);
    defer std.testing.allocator.destroy(prog);
    try compileJsonSchema(prog,
        \\{"type":"object","properties":{"status":{"type":"string"},"count":{"type":"integer"}},"required":["status","count"],"additionalProperties":false}
    );

    var st = prog.initialState();

    var m_before: MallocStatistics = undefined;
    var m_after: MallocStatistics = undefined;
    malloc_zone_statistics(null, &m_before);

    // At initial state: `{` (tok 1) and `" "` (tok 6) are allowed; `}` (tok 2) and `"hello"` (tok 17) are disallowed!
    vindex.fillAllowedTokenBitmask(prog, &st, mask);
    try std.testing.expect(isTokenAllowed(mask, 1)); // "{"
    try std.testing.expect(isTokenAllowed(mask, 6)); // " "
    try std.testing.expect(!isTokenAllowed(mask, 2)); // "}"
    try std.testing.expect(!isTokenAllowed(mask, 17)); // "hello"
    try std.testing.expect(!isTokenAllowed(mask, 0)); // EOS not allowed yet

    // Advance into `{"status": "` so we enter the self-looping string body!
    try std.testing.expect(prog.advanceBytes(&st, "{\"status\": \""));
    const snap_in_str = st.snapshot();

    vindex.fillAllowedTokenBitmask(prog, &st, mask);
    // Inside string: all ~248K plain ASCII string tokens ("hello", "world", "42", etc.) are allowed via fast-path!
    try std.testing.expect(isTokenAllowed(mask, 17)); // "hello"
    try std.testing.expect(isTokenAllowed(mask, 18)); // "world"
    try std.testing.expect(isTokenAllowed(mask, 3)); // closing "\""

    // Branch A: advance `"hello"` then reject illegal unquoted close
    try std.testing.expect(prog.advanceBytes(&st, "hello"));
    try std.testing.expect(!st.isAccepting());

    // O(1) restore back to `snap_in_str` for Branch B: finish valid object `ok", "count": 42}`
    st.restore(snap_in_str);
    try std.testing.expect(prog.advanceBytes(&st, "ok\", \"count\": 42}"));
    try std.testing.expect(st.isAccepting());

    vindex.fillAllowedTokenBitmask(prog, &st, mask);
    try std.testing.expect(isTokenAllowed(mask, 0)); // EOS token 0 now allowed!
    try std.testing.expect(!isTokenAllowed(mask, 17)); // trailing identifier rejected!

    malloc_zone_statistics(null, &m_after);
    try std.testing.expectEqual(m_before.size_allocated, m_after.size_allocated);
}
