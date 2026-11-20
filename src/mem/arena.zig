//! Preallocated startup wired arena backed by a single 16 KiB-aligned
//! `WiredRegion` capped by a `Budget` (ADR-0002, ADR-0012).
//! Startup-only carving: no dynamic heap allocation ever occurs here.

const std = @import("std");
const page = @import("page.zig");
const budget_mod = @import("budget.zig");

pub const WiredArena = struct {
    region: page.WiredRegion,
    used: usize,

    pub const InitError = page.WiredRegion.InitError || budget_mod.PlanError;

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
        return .{ .region = region, .used = 0 };
    }

    pub fn deinit(self: *WiredArena) void {
        self.region.deinit();
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
