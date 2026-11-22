//! 16 KiB page arithmetic and wired (`mlock`ed) memory regions (ADR-0002).

const std = @import("std");

/// Apple Silicon page size (`hw.pagesize`). Every buffer shared with Metal or
/// through IOSurface starts on this boundary and spans whole pages.
pub const APPLE_PAGE_SIZE: usize = 16384;

comptime {
    if (std.heap.page_size_min != APPLE_PAGE_SIZE) {
        @compileError("aminfer requires the 16 KiB pages of aarch64-macos");
    }
}

/// Rounds `len` up to a whole number of pages.
pub fn alignUp(len: usize) error{Overflow}!usize {
    const padded = try std.math.add(usize, len, APPLE_PAGE_SIZE - 1);
    return padded & ~@as(usize, APPLE_PAGE_SIZE - 1);
}

pub const ShareError = error{ ZeroLength, MisalignedAddress, MisalignedLength };

/// Checks the 16 KiB rule for a span about to be shared with Metal or through
/// IOSurface. Metal does not reliably reject violations, so every bridge
/// boundary calls this, in every build mode.
pub fn checkShareable(ptr: [*]const u8, len: usize) ShareError!void {
    if (len == 0) return error.ZeroLength;
    if (!std.mem.isAligned(@intFromPtr(ptr), APPLE_PAGE_SIZE)) return error.MisalignedAddress;
    if (!std.mem.isAligned(len, APPLE_PAGE_SIZE)) return error.MisalignedLength;
}

/// Anonymous memory pinned with `mlock`: never paged out and never swapped.
/// Created at startup only; the hot path never allocates.
pub const WiredRegion = struct {
    bytes: []align(APPLE_PAGE_SIZE) u8,

    pub const InitError = std.posix.MMapError || error{
        ZeroLength,
        Overflow,
        /// `mlock` would exceed `vm.user_wire_limit` or `vm.global_user_wire_limit`.
        WireLimitExceeded,
    };

    /// Maps `len` bytes rounded up to whole pages and wires them. A failure to
    /// wire is an error, never a silent fallback to pageable memory.
    pub fn init(len: usize) InitError!WiredRegion {
        if (len == 0) return error.ZeroLength;
        const size = try alignUp(len);
        const mapping = try std.posix.mmap(
            null,
            size,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        errdefer std.posix.munmap(mapping);
        switch (std.c.errno(std.c.mlock(mapping.ptr, mapping.len))) {
            .SUCCESS => {},
            .AGAIN => return error.WireLimitExceeded,
            .NOMEM => return error.OutOfMemory,
            .PERM => return error.PermissionDenied,
            else => |err| return std.posix.unexpectedErrno(err),
        }
        return .{ .bytes = mapping };
    }

    /// Explicitly unwires and unmaps the pages.
    pub fn deinit(self: WiredRegion) void {
        _ = std.c.munlock(self.bytes.ptr, self.bytes.len);
        std.posix.munmap(self.bytes);
    }
};

test "alignUp rounds lengths up to whole 16 KiB pages" {
    try std.testing.expectEqual(@as(usize, 0), try alignUp(0));
    try std.testing.expectEqual(@as(usize, 16384), try alignUp(1));
    try std.testing.expectEqual(@as(usize, 16384), try alignUp(16384));
    try std.testing.expectEqual(@as(usize, 32768), try alignUp(16385));
    try std.testing.expectError(error.Overflow, alignUp(std.math.maxInt(usize)));
}

test "checkShareable accepts page-aligned spans and rejects everything else" {
    const region = try WiredRegion.init(2 * APPLE_PAGE_SIZE);
    defer region.deinit();
    const base = region.bytes.ptr;
    try checkShareable(base, APPLE_PAGE_SIZE);
    try std.testing.expectError(error.MisalignedAddress, checkShareable(base + 64, APPLE_PAGE_SIZE));
    // The IOSurface case measured on the baseline: 128 KiB of data + 64 B.
    try std.testing.expectError(error.MisalignedLength, checkShareable(base, 131_136));
    try std.testing.expectError(error.ZeroLength, checkShareable(base, 0));
}

test "WiredRegion is 16 KiB aligned in address and length" {
    const region = try WiredRegion.init(1);
    defer region.deinit();
    try std.testing.expectEqual(APPLE_PAGE_SIZE, region.bytes.len);
    try std.testing.expect(std.mem.isAligned(@intFromPtr(region.bytes.ptr), APPLE_PAGE_SIZE));
    @memset(region.bytes, 0x5A);
    try std.testing.expectEqual(@as(u8, 0x5A), region.bytes[region.bytes.len - 1]);
}

test "WiredRegion rejects a zero length" {
    try std.testing.expectError(error.ZeroLength, WiredRegion.init(0));
}
