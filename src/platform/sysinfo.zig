//! Read-only kernel facts the engine budgets against, read through
//! `sysctlbyname` (docs/architecture/hardware.md).

const std = @import("std");
const page = @import("../mem/page.zig");

pub const Error = std.posix.UnexpectedError || error{
    /// The kernel does not know this sysctl name.
    UnknownName,
    /// The value is not a 32- or 64-bit integer.
    UnexpectedSize,
};

/// Reads an unsigned 32- or 64-bit integer sysctl.
pub fn readUnsigned(name: [:0]const u8) Error!u64 {
    var value: u64 = 0;
    var size: usize = @sizeOf(u64);
    switch (std.c.errno(std.c.sysctlbyname(name.ptr, &value, &size, null, 0))) {
        .SUCCESS => {},
        .NOENT => return error.UnknownName,
        .NOMEM => return error.UnexpectedSize,
        else => |err| return std.posix.unexpectedErrno(err),
    }
    return switch (size) {
        // aarch64 is little-endian: a 4-byte value fills the low half.
        @sizeOf(u32), @sizeOf(u64) => value,
        else => error.UnexpectedSize,
    };
}

pub const Facts = struct {
    /// `hw.pagesize`
    page_size: u64,
    /// `hw.memsize`
    memory_bytes: u64,
    /// `hw.perflevel0.logicalcpu`
    performance_cores: u64,
    /// `hw.perflevel1.logicalcpu`
    efficiency_cores: u64,
    /// `vm.user_wire_limit`: per-process `mlock` limit in bytes.
    user_wire_limit: u64,
    /// `vm.global_user_wire_limit`: system-wide `mlock` limit in bytes.
    global_user_wire_limit: u64,
    /// `iogpu.wired_limit_mb`: GPU wired limit in MiB; 0 means the system default.
    gpu_wired_limit_mib: u64,

    pub fn read() Error!Facts {
        return .{
            .page_size = try readUnsigned("hw.pagesize"),
            .memory_bytes = try readUnsigned("hw.memsize"),
            .performance_cores = try readUnsigned("hw.perflevel0.logicalcpu"),
            .efficiency_cores = try readUnsigned("hw.perflevel1.logicalcpu"),
            .user_wire_limit = try readUnsigned("vm.user_wire_limit"),
            .global_user_wire_limit = try readUnsigned("vm.global_user_wire_limit"),
            .gpu_wired_limit_mib = try readUnsigned("iogpu.wired_limit_mb"),
        };
    }
};

test "the kernel page size is the Apple Silicon page size" {
    const facts = try Facts.read();
    try std.testing.expectEqual(@as(u64, page.APPLE_PAGE_SIZE), facts.page_size);
}

test "memory, core counts and wire limits are reported" {
    const facts = try Facts.read();
    try std.testing.expect(facts.memory_bytes > 0);
    try std.testing.expect(facts.performance_cores > 0);
    try std.testing.expect(facts.efficiency_cores > 0);
    try std.testing.expect(facts.user_wire_limit > 0 and facts.user_wire_limit <= facts.memory_bytes);
    try std.testing.expect(facts.global_user_wire_limit > 0 and facts.global_user_wire_limit <= facts.memory_bytes);
}

test "an unknown sysctl name is an error, not a zero" {
    try std.testing.expectError(error.UnknownName, readUnsigned("aminfer.no_such_name"));
}
