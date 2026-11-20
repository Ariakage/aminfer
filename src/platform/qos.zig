//! macOS QoS-based core steering and thread disk I/O policy overrides
//! (status row C2, docs/architecture/hardware.md#core-steering,
//! docs/architecture/runtime-pipeline.md#threading-model).
//!
//! Apple Silicon exposes no hard CPU affinity. Threads are steered toward
//! P-cores (`user_interactive` / `user_initiated`) or E-cores (`utility` /
//! `background`) through `pthread_set_qos_class_self_np`, and E-core prefetch
//! threads override kernel I/O throttling with `setiopolicy_np(..., IOPOL_IMPORTANT)`.

const std = @import("std");

/// From `<sys/resource.h>`.
pub const IOPOL_TYPE_DISK: c_int = 0;
pub const IOPOL_SCOPE_PROCESS: c_int = 0;
pub const IOPOL_SCOPE_THREAD: c_int = 1;

/// From `<sys/qos.h>`: minimum legal relative priority offset within a QoS class.
pub const MIN_RELATIVE_PRIORITY: i32 = -15;

pub const Error = std.posix.UnexpectedError || error{
    InvalidArgument,
    PermissionDenied,
    UnknownQosClass,
    UnknownIoPolicy,
};

extern "c" fn setiopolicy_np(iotype: c_int, scope: c_int, policy: c_int) c_int;
extern "c" fn getiopolicy_np(iotype: c_int, scope: c_int) c_int;

pub const QosClass = enum(c_uint) {
    user_interactive = 0x21,
    user_initiated = 0x19,
    default = 0x15,
    utility = 0x11,
    background = 0x09,
    unspecified = 0x00,

    pub fn toC(self: QosClass) std.c.qos_class_t {
        return @fromBackingInt(@backingInt(self));
    }

    pub fn fromC(raw: std.c.qos_class_t) Error!QosClass {
        return std.enums.fromInt(QosClass, @backingInt(raw)) orelse error.UnknownQosClass;
    }
};

pub const DiskIoPolicy = enum(c_int) {
    default = 0,
    important = 1,
    passive = 2,
    throttle = 3,
    utility = 4,
    standard = 5,

    /// XNU stores `IOPOL_IMPORTANT` (1) as unrestricted default (`IOPOL_DEFAULT` = 0)
    /// when read back through `getiopolicy_np`. Both indicate unthrottled disk I/O.
    pub fn isUnthrottled(self: DiskIoPolicy) bool {
        return self == .default or self == .important;
    }
};

pub const IoScope = enum(c_int) {
    process = IOPOL_SCOPE_PROCESS,
    thread = IOPOL_SCOPE_THREAD,
};

pub const ThreadQos = struct {
    class: QosClass,
    relative_priority: i32,
};

/// Engine thread roles from `docs/architecture/runtime-pipeline.md`.
pub const Role = enum {
    scheduler,
    l1_draft,
    ane_driver,
    server,
    io_prefetch,
};

fn checkPthreadRc(rc: c_int) Error!void {
    if (rc == 0) return;
    if (rc < 0) return error.Unexpected;
    const err = std.enums.fromInt(std.c.E, rc) orelse return error.Unexpected;
    return switch (err) {
        .INVAL => error.InvalidArgument,
        .PERM => error.PermissionDenied,
        else => std.posix.unexpectedErrno(err),
    };
}

/// Sets the requested QoS class and relative priority (`MIN_RELATIVE_PRIORITY..=0`)
/// of the calling thread via `pthread_set_qos_class_self_np`.
pub fn setCurrentThreadQos(class: QosClass, relative_priority: i32) Error!void {
    if (class == .unspecified) return error.InvalidArgument;
    if (relative_priority > 0 or relative_priority < MIN_RELATIVE_PRIORITY) {
        return error.InvalidArgument;
    }
    const rc = std.c.pthread_set_qos_class_self_np(class.toC(), @intCast(relative_priority));
    try checkPthreadRc(rc);
}

/// Reads back the requested QoS class and relative priority of `thread`
/// via `pthread_get_qos_class_np`.
pub fn getThreadQos(thread: std.c.pthread_t) Error!ThreadQos {
    var raw_class: std.c.qos_class_t = .UNSPECIFIED;
    var raw_priority: c_int = 0;
    const rc = std.c.pthread_get_qos_class_np(thread, &raw_class, &raw_priority);
    try checkPthreadRc(rc);
    return .{
        .class = try QosClass.fromC(raw_class),
        .relative_priority = @intCast(raw_priority),
    };
}

/// Reads back the requested QoS class and relative priority of the calling thread.
pub fn getCurrentThreadQos() Error!ThreadQos {
    return getThreadQos(std.c.pthread_self());
}

/// Sets the disk I/O policy (`IOPOL_TYPE_DISK`) for the given scope (`process` or `thread`)
/// via `setiopolicy_np`.
pub fn setDiskIoPolicy(scope: IoScope, policy: DiskIoPolicy) Error!void {
    const rc = setiopolicy_np(IOPOL_TYPE_DISK, @backingInt(scope), @backingInt(policy));
    if (rc != 0) {
        return switch (std.c.errno(rc)) {
            .INVAL => error.InvalidArgument,
            .PERM => error.PermissionDenied,
            else => |err| std.posix.unexpectedErrno(err),
        };
    }
}

/// Reads back the disk I/O policy (`IOPOL_TYPE_DISK`) for the given scope
/// via `getiopolicy_np`.
pub fn getDiskIoPolicy(scope: IoScope) Error!DiskIoPolicy {
    const rc = getiopolicy_np(IOPOL_TYPE_DISK, @backingInt(scope));
    if (rc < 0) {
        return switch (std.c.errno(rc)) {
            .INVAL => error.InvalidArgument,
            .PERM => error.PermissionDenied,
            else => |err| std.posix.unexpectedErrno(err),
        };
    }
    return std.enums.fromInt(DiskIoPolicy, rc) orelse error.UnknownIoPolicy;
}

/// Sets the calling thread's disk I/O policy (`IOPOL_SCOPE_THREAD`).
pub fn setThreadDiskIoPolicy(policy: DiskIoPolicy) Error!void {
    return setDiskIoPolicy(.thread, policy);
}

/// Reads back the calling thread's disk I/O policy (`IOPOL_SCOPE_THREAD`).
pub fn getThreadDiskIoPolicy() Error!DiskIoPolicy {
    return getDiskIoPolicy(.thread);
}

/// Steers the calling thread toward Apple Silicon P-cores
/// (`user_interactive` or `user_initiated`).
pub fn steerPerformanceCore(class: QosClass) Error!void {
    switch (class) {
        .user_interactive, .user_initiated => {},
        else => return error.InvalidArgument,
    }
    try setCurrentThreadQos(class, 0);
}

/// Steers the calling thread toward Apple Silicon E-cores (`utility` or `background`)
/// and overrides kernel disk I/O throttling to `IOPOL_IMPORTANT` so cold-expert
/// NVMe prefetches are not throttled.
///
/// Order note (macOS XNU): `setiopolicy_np(..., IOPOL_SCOPE_THREAD, ...)` marks
/// the thread as using an explicit legacy I/O policy, which clears its pthread
/// QoS class (`pthread_get_qos_class_np` thereafter returns `.unspecified` and
/// subsequent `pthread_set_qos_class_self_np` calls fail with `EPERM`). Setting
/// QoS first ensures the QoS request runs before the I/O policy override.
pub fn steerEfficiencyCorePrefetch(class: QosClass) Error!void {
    switch (class) {
        .utility, .background => {},
        else => return error.InvalidArgument,
    }
    try setCurrentThreadQos(class, 0);
    try setThreadDiskIoPolicy(.important);
}

/// Applies the documented QoS and I/O steering for an engine thread role.
pub fn applyRole(role: Role) Error!void {
    return switch (role) {
        .scheduler, .l1_draft => steerPerformanceCore(.user_interactive),
        .ane_driver, .server => steerPerformanceCore(.user_initiated),
        .io_prefetch => steerEfficiencyCorePrefetch(.utility),
    };
}

test "P-core QoS classes and relative priority round-trip on spawned worker threads" {
    const Outcome = struct {
        err: ?Error = null,
        got_qos: ThreadQos = .{ .class = .unspecified, .relative_priority = 0 },
    };

    const Worker = struct {
        fn run(out: *Outcome, class: QosClass, rel_prio: i32) void {
            if (rel_prio == 0) {
                steerPerformanceCore(class) catch |err| {
                    out.err = err;
                    return;
                };
            } else {
                setCurrentThreadQos(class, rel_prio) catch |err| {
                    out.err = err;
                    return;
                };
            }
            out.got_qos = getCurrentThreadQos() catch |err| {
                out.err = err;
                return;
            };
        }
    };

    const cases = [_]struct { class: QosClass, rel: i32 }{
        .{ .class = .user_interactive, .rel = 0 },
        .{ .class = .user_initiated, .rel = 0 },
        .{ .class = .user_initiated, .rel = -3 },
    };

    for (cases) |c| {
        var outcome: Outcome = .{};
        const t = try std.Thread.spawn(.{}, Worker.run, .{ &outcome, c.class, c.rel });
        t.join();
        if (outcome.err) |err| return err;
        try std.testing.expectEqual(c.class, outcome.got_qos.class);
        try std.testing.expectEqual(c.rel, outcome.got_qos.relative_priority);
    }
}

test "E-core QoS classes and thread disk I/O policy transitions read back accurately" {
    const Outcome = struct {
        err: ?Error = null,
        qos_before_io: ThreadQos = .{ .class = .unspecified, .relative_priority = 0 },
        io_throttled: DiskIoPolicy = .default,
        io_utility: DiskIoPolicy = .default,
        io_restored: DiskIoPolicy = .default,
        qos_after_io_err: ?Error = null,
    };

    const Worker = struct {
        fn run(out: *Outcome, class: QosClass) void {
            setCurrentThreadQos(class, 0) catch |err| {
                out.err = err;
                return;
            };
            out.qos_before_io = getThreadQos(std.c.pthread_self()) catch |err| {
                out.err = err;
                return;
            };

            // Verify setiopolicy_np / getiopolicy_np genuinely mutate kernel state.
            setThreadDiskIoPolicy(.throttle) catch |err| {
                out.err = err;
                return;
            };
            out.io_throttled = getThreadDiskIoPolicy() catch |err| {
                out.err = err;
                return;
            };

            setThreadDiskIoPolicy(.utility) catch |err| {
                out.err = err;
                return;
            };
            out.io_utility = getThreadDiskIoPolicy() catch |err| {
                out.err = err;
                return;
            };

            // Override back to IOPOL_IMPORTANT (unthrottled).
            setThreadDiskIoPolicy(.important) catch |err| {
                out.err = err;
                return;
            };
            out.io_restored = getThreadDiskIoPolicy() catch |err| {
                out.err = err;
                return;
            };

            // XNU opts threads with explicit thread iopolicy out of pthread QoS.
            setCurrentThreadQos(class, 0) catch |err| {
                out.qos_after_io_err = err;
            };
        }
    };

    for ([_]QosClass{ .utility, .background }) |class| {
        var outcome: Outcome = .{};
        const t = try std.Thread.spawn(.{}, Worker.run, .{ &outcome, class });
        t.join();
        if (outcome.err) |err| return err;
        try std.testing.expectEqual(class, outcome.qos_before_io.class);
        try std.testing.expectEqual(@as(i32, 0), outcome.qos_before_io.relative_priority);
        try std.testing.expectEqual(DiskIoPolicy.throttle, outcome.io_throttled);
        try std.testing.expectEqual(DiskIoPolicy.utility, outcome.io_utility);
        try std.testing.expect(outcome.io_restored.isUnthrottled());
        try std.testing.expectEqual(@as(?Error, error.PermissionDenied), outcome.qos_after_io_err);
    }
}

test "steerEfficiencyCorePrefetch and applyRole succeed and leave unthrottled disk I/O" {
    const Outcome = struct {
        err: ?Error = null,
        got_qos: ThreadQos = .{ .class = .unspecified, .relative_priority = 0 },
        got_io: DiskIoPolicy = .default,
    };

    const PrefetchWorker = struct {
        fn run(out: *Outcome, class: QosClass) void {
            steerEfficiencyCorePrefetch(class) catch |err| {
                out.err = err;
                return;
            };
            out.got_qos = getCurrentThreadQos() catch |err| {
                out.err = err;
                return;
            };
            out.got_io = getThreadDiskIoPolicy() catch |err| {
                out.err = err;
                return;
            };
        }
    };

    for ([_]QosClass{ .utility, .background }) |class| {
        var outcome: Outcome = .{};
        const t = try std.Thread.spawn(.{}, PrefetchWorker.run, .{ &outcome, class });
        t.join();
        if (outcome.err) |err| return err;
        try std.testing.expect(outcome.got_qos.class == class or outcome.got_qos.class == .unspecified);
        try std.testing.expect(outcome.got_io.isUnthrottled());
    }

    const RoleWorker = struct {
        fn run(out: *Outcome, role: Role) void {
            applyRole(role) catch |err| {
                out.err = err;
                return;
            };
            out.got_qos = getCurrentThreadQos() catch |err| {
                out.err = err;
                return;
            };
            out.got_io = getThreadDiskIoPolicy() catch |err| {
                out.err = err;
                return;
            };
        }
    };

    const expectations = [_]struct { role: Role, class: QosClass, overrides_io: bool }{
        .{ .role = .scheduler, .class = .user_interactive, .overrides_io = false },
        .{ .role = .l1_draft, .class = .user_interactive, .overrides_io = false },
        .{ .role = .ane_driver, .class = .user_initiated, .overrides_io = false },
        .{ .role = .server, .class = .user_initiated, .overrides_io = false },
        .{ .role = .io_prefetch, .class = .utility, .overrides_io = true },
    };

    for (expectations) |exp| {
        var outcome: Outcome = .{};
        const t = try std.Thread.spawn(.{}, RoleWorker.run, .{ &outcome, exp.role });
        t.join();
        if (outcome.err) |err| return err;
        if (exp.overrides_io) {
            try std.testing.expect(outcome.got_qos.class == exp.class or outcome.got_qos.class == .unspecified);
            try std.testing.expect(outcome.got_io.isUnthrottled());
        } else {
            try std.testing.expectEqual(exp.class, outcome.got_qos.class);
        }
    }
}

test "invalid QoS classes, priorities, and steering mismatches are rejected" {
    try std.testing.expectError(error.InvalidArgument, setCurrentThreadQos(.unspecified, 0));
    try std.testing.expectError(error.InvalidArgument, setCurrentThreadQos(.user_interactive, 1));
    try std.testing.expectError(error.InvalidArgument, setCurrentThreadQos(.user_interactive, MIN_RELATIVE_PRIORITY - 1));
    try std.testing.expectError(error.InvalidArgument, steerPerformanceCore(.utility));
    try std.testing.expectError(error.InvalidArgument, steerPerformanceCore(.background));
    try std.testing.expectError(error.InvalidArgument, steerEfficiencyCorePrefetch(.user_interactive));
    try std.testing.expectError(error.InvalidArgument, steerEfficiencyCorePrefetch(.user_initiated));
}
