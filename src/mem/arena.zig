//! Preallocated startup wired arena backed by a single 16 KiB-aligned
//! `WiredRegion` capped by a `Budget` (ADR-0002, ADR-0012).
//! Startup-only carving: no dynamic heap allocation ever occurs here.

const std = @import("std");
const page = @import("page.zig");
const budget_mod = @import("budget.zig");

/// Pair of non-owning `WiredArena` sub-arenas aliasing the exact same
/// 16 KiB-aligned physical backing pages (`P0-B2`, ADR-0020) so mutually
/// exclusive single-request workspaces (e.g., wide-prefill scratch and
/// speculative `verifyTree` scratch) share RAM without unloading model weights.
pub const AliasedScratchPair = struct {
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    arena_a: WiredArena,
    arena_b: WiredArena,
    saved_bytes: usize,
};

pub const WiredArena = struct {
    region: page.WiredRegion,
    used: usize,
    owns_mapping: bool = true,

    pub const InitError = page.WiredRegion.InitError || budget_mod.PlanError;
    pub const AliasedPair = AliasedScratchPair;

    pub const AllocError = error{
        ZeroLength,
        InvalidAlignment,
        Overflow,
        OutOfArenaSpace,
    };

    /// Maps and locks a single `WiredRegion` after checking `len` (rounded up
    /// to whole 16 KiB pages) against `budget`.
    pub fn init(len: usize, budget: budget_mod.Budget) InitError!WiredArena {
        if (len == 0) return error.ZeroLength;
        const total = try page.alignUp(len);
        try budget.checkPlan(total);
        const region = try page.WiredRegion.init(total);
        return .{ .region = region, .used = 0, .owns_mapping = true };
    }

    /// Creates a non-owning sub-arena over an already-wired `16 KiB`-aligned
    /// slice (address and length both 16 KiB aligned). Calling `deinit()` on a
    /// sub-arena resets its cursor without unmapping the parent `WiredRegion`.
    pub fn initFromAlignedSlice(slice: []align(page.APPLE_PAGE_SIZE) u8) WiredArena {
        std.debug.assert(std.mem.isAligned(slice.len, page.APPLE_PAGE_SIZE));
        return .{
            .region = .{ .bytes = slice },
            .used = 0,
            .owns_mapping = false,
        };
    }

    pub fn deinit(self: *WiredArena) void {
        if (self.owns_mapping) {
            self.region.deinit();
            self.owns_mapping = false;
        }
        self.used = 0;
    }

    /// Full backing byte slice (16 KiB aligned in both address and length).
    pub fn bytes(self: *const WiredArena) []align(page.APPLE_PAGE_SIZE) u8 {
        return self.region.bytes;
    }

    pub fn capacity(self: *const WiredArena) usize {
        return self.region.bytes.len;
    }

    pub fn remaining(self: *const WiredArena) usize {
        return self.region.bytes.len - self.used;
    }

    /// Byte offset of a pointer carved from this arena relative to its base.
    pub fn offsetOf(self: *const WiredArena, ptr: [*]const u8) usize {
        const base = @intFromPtr(self.region.bytes.ptr);
        const addr = @intFromPtr(ptr);
        std.debug.assert(addr >= base and addr <= base + self.region.bytes.len);
        return addr - base;
    }

    /// Carves a slice aligned to `alignment` (a power of two <= APPLE_PAGE_SIZE).
    /// Startup use only; performs zero heap allocation.
    pub fn allocAligned(self: *WiredArena, len: usize, alignment: usize) AllocError![]u8 {
        if (len == 0) return error.ZeroLength;
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment) or alignment > page.APPLE_PAGE_SIZE) {
            return error.InvalidAlignment;
        }
        const mask = alignment - 1;
        const padded = try std.math.add(usize, self.used, mask);
        const start = padded & ~mask;
        const end = try std.math.add(usize, start, len);
        if (end > self.region.bytes.len) return error.OutOfArenaSpace;
        self.used = end;
        return self.region.bytes[start..end];
    }

    /// Carves whole 16 KiB pages (start and length both 16 KiB aligned),
    /// directly usable with Metal `NoCopy` buffers or cold-expert slots.
    pub fn allocPages(self: *WiredArena, len: usize) AllocError![]align(page.APPLE_PAGE_SIZE) u8 {
        if (len == 0) return error.ZeroLength;
        const aligned_len = try page.alignUp(len);
        const slice = try self.allocAligned(aligned_len, page.APPLE_PAGE_SIZE);
        return @alignCast(slice);
    }

    /// Allocates ONE `16 KiB`-aligned region of `alignUp(@max(bytes_a, bytes_b), APPLE_PAGE_SIZE)`
    /// inside this `WiredArena` and returns two non-owning sub-arenas (`arena_a`,
    /// `arena_b`) aliasing the exact same physical pages (`P0-B2`, ADR-0020).
    /// `saved_bytes = alignUp(bytes_a, 16384) + alignUp(bytes_b, 16384) - backing.len`.
    pub fn allocAliasedPair(self: *WiredArena, bytes_a: usize, bytes_b: usize) AllocError!AliasedScratchPair {
        const aligned_a = try page.alignUp(bytes_a);
        const aligned_b = try page.alignUp(bytes_b);
        const needed = @max(aligned_a, aligned_b);
        if (needed == 0) return error.ZeroLength;

        const backing = try self.allocPages(needed);
        // Mathematically identical to `aligned_a + aligned_b - backing.len` without overflow risk.
        const saved_bytes = @min(aligned_a, aligned_b);
        return .{
            .backing = backing,
            .arena_a = WiredArena.initFromAlignedSlice(backing[0..aligned_a]),
            .arena_b = WiredArena.initFromAlignedSlice(backing[0..aligned_b]),
            .saved_bytes = saved_bytes,
        };
    }

    /// Reclaims all carved offsets without unmapping or unlocking the region.
    pub fn reset(self: *WiredArena) void {
        self.used = 0;
    }
};

test "WiredArena enforces the wired budget before mapping" {
    const tight = budget_mod.Budget.compute(.{
        .memory_bytes = 8 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = 2 * page.APPLE_PAGE_SIZE,
        .global_user_wire_limit = 8 * page.APPLE_PAGE_SIZE,
        .gpu_working_set_bytes = 8 * page.APPLE_PAGE_SIZE,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    try std.testing.expectError(error.BudgetExceeded, WiredArena.init(3 * page.APPLE_PAGE_SIZE, tight));
}

test "WiredArena carves aligned sub-slices and whole pages without overlap" {
    const cap = 4 * page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = 16 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try WiredArena.init(cap, budget);
    defer arena.deinit();

    const first_page = try arena.allocPages(1);
    try std.testing.expectEqual(page.APPLE_PAGE_SIZE, first_page.len);
    try std.testing.expect(std.mem.isAligned(@intFromPtr(first_page.ptr), page.APPLE_PAGE_SIZE));

    const small = try arena.allocAligned(100, 64);
    try std.testing.expectEqual(@as(usize, 100), small.len);
    try std.testing.expect(std.mem.isAligned(@intFromPtr(small.ptr), 64));
    try std.testing.expectEqual(page.APPLE_PAGE_SIZE, arena.offsetOf(small.ptr));

    const second = try arena.allocAligned(200, 64);
    try std.testing.expect(std.mem.isAligned(@intFromPtr(second.ptr), 64));
    try std.testing.expectEqual(page.APPLE_PAGE_SIZE + 128, arena.offsetOf(second.ptr));

    // Next page allocation advances to the next 16 KiB boundary.
    const next_page = try arena.allocPages(page.APPLE_PAGE_SIZE);
    try std.testing.expect(std.mem.isAligned(@intFromPtr(next_page.ptr), page.APPLE_PAGE_SIZE));
    try std.testing.expectEqual(2 * page.APPLE_PAGE_SIZE, arena.offsetOf(next_page.ptr));

    // Only 1 page remains; asking for 2 pages must fail cleanly.
    try std.testing.expectEqual(page.APPLE_PAGE_SIZE, arena.remaining());
    try std.testing.expectError(error.OutOfArenaSpace, arena.allocPages(2 * page.APPLE_PAGE_SIZE));
    try std.testing.expectError(error.ZeroLength, arena.allocPages(0));
    try std.testing.expectError(error.ZeroLength, arena.allocAligned(0, 64));
}

test "WiredArena reset reclaims all space without remapping" {
    const cap = 2 * page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = 8 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try WiredArena.init(cap, budget);
    defer arena.deinit();

    const a = try arena.allocPages(cap);
    try std.testing.expectEqual(@as(usize, 0), arena.remaining());
    arena.reset();
    try std.testing.expectEqual(cap, arena.remaining());
    const b = try arena.allocPages(cap);
    try std.testing.expectEqual(a.ptr, b.ptr);
}

test "WiredArena.allocAliasedPair enforces 16 KiB alignment, aliases pages, and accounts saved_bytes" {
    const cap = 8 * page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = 32 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try WiredArena.init(cap, budget);
    defer arena.deinit();

    // Advance by a small unaligned-to-page slice first so allocAliasedPair must realign to 16 KiB.
    _ = try arena.allocAligned(128, 64);

    const bytes_a: usize = 2 * page.APPLE_PAGE_SIZE + 17; // rounds up to 3 pages
    const bytes_b: usize = 4 * page.APPLE_PAGE_SIZE + 5; // rounds up to 5 pages
    var pair = try arena.allocAliasedPair(bytes_a, bytes_b);

    try std.testing.expectEqual(5 * page.APPLE_PAGE_SIZE, pair.backing.len);
    try std.testing.expectEqual(3 * page.APPLE_PAGE_SIZE, pair.arena_a.capacity());
    try std.testing.expectEqual(5 * page.APPLE_PAGE_SIZE, pair.arena_b.capacity());
    try std.testing.expectEqual(3 * page.APPLE_PAGE_SIZE, pair.saved_bytes);
    try std.testing.expectEqual(
        (try page.alignUp(bytes_a)) + (try page.alignUp(bytes_b)) - pair.backing.len,
        pair.saved_bytes,
    );
    try std.testing.expect(std.mem.isAligned(@intFromPtr(pair.backing.ptr), page.APPLE_PAGE_SIZE));
    try std.testing.expectEqual(pair.backing.ptr, pair.arena_a.bytes().ptr);
    try std.testing.expectEqual(pair.backing.ptr, pair.arena_b.bytes().ptr);

    // Writes via arena_a are physically aliased in arena_b at the exact same address.
    const sa = try pair.arena_a.allocPages(2 * page.APPLE_PAGE_SIZE);
    const sb = try pair.arena_b.allocPages(2 * page.APPLE_PAGE_SIZE);
    try std.testing.expectEqual(sa.ptr, sb.ptr);
    sa[0] = 0xDE;
    sa[sa.len - 1] = 0xAD;
    try std.testing.expectEqual(@as(u8, 0xDE), sb[0]);
    try std.testing.expectEqual(@as(u8, 0xAD), sb[sb.len - 1]);

    // Sub-arena capacity limits still guard against accidental overflow.
    try std.testing.expectError(error.OutOfArenaSpace, pair.arena_a.allocPages(2 * page.APPLE_PAGE_SIZE));
    _ = try pair.arena_b.allocPages(3 * page.APPLE_PAGE_SIZE);
    try std.testing.expectEqual(@as(usize, 0), pair.arena_b.remaining());

    // Calling deinit() on borrowed sub-arenas must NOT unmap the parent arena.
    pair.arena_a.deinit();
    pair.arena_b.deinit();
    try std.testing.expectEqual(@as(usize, 0), pair.arena_a.used);
    try std.testing.expectEqual(@as(usize, 0), pair.arena_b.used);

    // Remaining 2 pages of parent arena remain wired and usable.
    try std.testing.expectEqual(2 * page.APPLE_PAGE_SIZE, arena.remaining());
    var one_sided = try arena.allocAliasedPair(page.APPLE_PAGE_SIZE, 0);
    try std.testing.expectEqual(page.APPLE_PAGE_SIZE, one_sided.backing.len);
    try std.testing.expectEqual(@as(usize, 0), one_sided.saved_bytes);
    try std.testing.expectEqual(@as(usize, 0), one_sided.arena_b.capacity());
    one_sided.arena_a.deinit();
    one_sided.arena_b.deinit();

    // Error cases: zero/zero and insufficient space.
    try std.testing.expectError(error.ZeroLength, arena.allocAliasedPair(0, 0));
    try std.testing.expectError(error.OutOfArenaSpace, arena.allocAliasedPair(2 * page.APPLE_PAGE_SIZE, page.APPLE_PAGE_SIZE));
}
