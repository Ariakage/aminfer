//! Startup wired-memory budget computation and enforcement (ADR-0012).
//! Computes the usable wired ceiling from physical RAM minus headroom,
//! `vm.user_wire_limit`, `vm.global_user_wire_limit`, Metal's
//! `recommendedMaxWorkingSetSize` and any caller-configured cap.

const std = @import("std");
const page = @import("page.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const metal = @import("../bridge/metal.zig");

/// Default physical memory reserved for macOS and other processes (ADR-0012).
pub const DEFAULT_HEADROOM_BYTES: u64 = 3 * 1024 * 1024 * 1024;

pub const BindingLimit = enum {
    configured_cap,
    physical_headroom,
    user_wire_limit,
    global_user_wire_limit,
    gpu_working_set,

    pub fn label(self: BindingLimit) []const u8 {
        return switch (self) {
            .configured_cap => "configured_cap",
            .physical_headroom => "physical_headroom",
            .user_wire_limit => "vm.user_wire_limit",
            .global_user_wire_limit => "vm.global_user_wire_limit",
            .gpu_working_set => "MTLDevice.recommendedMaxWorkingSetSize",
        };
    }
};

pub const Config = struct {
    /// Optional hard cap requested by the caller (bytes).
    configured_cap: ?u64 = null,
    /// Physical RAM reserved for macOS (bytes); defaults to 3 GiB.
    headroom_bytes: u64 = DEFAULT_HEADROOM_BYTES,
};

pub const Inputs = struct {
    memory_bytes: u64,
    user_wire_limit: u64,
    global_user_wire_limit: u64,
    gpu_working_set_bytes: u64,
    configured_cap: ?u64 = null,
    headroom_bytes: u64 = DEFAULT_HEADROOM_BYTES,
};

pub const PlanError = error{
    /// Requested wired bytes exceed `Budget.limit_bytes` (inspect
    /// `Budget.binding_limit` for the exact constraint).
    BudgetExceeded,
};

pub const Budget = struct {
    /// Maximum bytes aminfer may wire, rounded down to whole 16 KiB pages.
    limit_bytes: u64,
    /// Which system or configuration constraint set `limit_bytes`.
    binding_limit: BindingLimit,
    /// The raw (pre-alignment) ceiling reported by `binding_limit`.
    raw_limit_bytes: u64,

    pub fn compute(inputs: Inputs) Budget {
        const physical_usable = if (inputs.memory_bytes > inputs.headroom_bytes)
            inputs.memory_bytes - inputs.headroom_bytes
        else
            0;

        var raw_limit: u64 = physical_usable;
        var binding: BindingLimit = .physical_headroom;

        if (inputs.user_wire_limit < raw_limit) {
            raw_limit = inputs.user_wire_limit;
            binding = .user_wire_limit;
        }
        if (inputs.global_user_wire_limit < raw_limit) {
            raw_limit = inputs.global_user_wire_limit;
            binding = .global_user_wire_limit;
        }
        if (inputs.gpu_working_set_bytes < raw_limit) {
            raw_limit = inputs.gpu_working_set_bytes;
            binding = .gpu_working_set;
        }
        if (inputs.configured_cap) |cap| {
            if (cap < raw_limit) {
                raw_limit = cap;
                binding = .configured_cap;
            }
        }

        const page_size: u64 = page.APPLE_PAGE_SIZE;
        const aligned = raw_limit - (raw_limit % page_size);
        return .{
            .limit_bytes = aligned,
            .binding_limit = binding,
            .raw_limit_bytes = raw_limit,
        };
    }

    pub fn fromHost(facts: sysinfo.Facts, device_info: metal.DeviceInfo, config: Config) Budget {
        return compute(.{
            .memory_bytes = facts.memory_bytes,
            .user_wire_limit = facts.user_wire_limit,
            .global_user_wire_limit = facts.global_user_wire_limit,
            .gpu_working_set_bytes = device_info.recommendedMaxWorkingSetSize(),
            .configured_cap = config.configured_cap,
            .headroom_bytes = config.headroom_bytes,
        });
    }

    /// Rejects any model/arena plan that exceeds the wired budget instead of
    /// silently degrading to pageable memory (ADR-0012).
    pub fn checkPlan(self: Budget, required_bytes: u64) PlanError!void {
        if (required_bytes > self.limit_bytes) return error.BudgetExceeded;
    }
};

test "compute selects the tightest limit and identifies it" {
    const gib: u64 = 1024 * 1024 * 1024;
    const base = Inputs{
        .memory_bytes = 24 * gib,
        .user_wire_limit = 20 * gib,
        .global_user_wire_limit = 20 * gib,
        .gpu_working_set_bytes = 18 * gib,
        .configured_cap = null,
        .headroom_bytes = DEFAULT_HEADROOM_BYTES,
    };

    const gpu_bound = Budget.compute(base);
    try std.testing.expectEqual(@as(u64, 18 * gib), gpu_bound.limit_bytes);
    try std.testing.expectEqual(BindingLimit.gpu_working_set, gpu_bound.binding_limit);

    var user_in = base;
    user_in.user_wire_limit = 16 * gib;
    const user_bound = Budget.compute(user_in);
    try std.testing.expectEqual(@as(u64, 16 * gib), user_bound.limit_bytes);
    try std.testing.expectEqual(BindingLimit.user_wire_limit, user_bound.binding_limit);

    var global_in = base;
    global_in.global_user_wire_limit = 15 * gib;
    const global_bound = Budget.compute(global_in);
    try std.testing.expectEqual(@as(u64, 15 * gib), global_bound.limit_bytes);
    try std.testing.expectEqual(BindingLimit.global_user_wire_limit, global_bound.binding_limit);

    var headroom_in = base;
    headroom_in.headroom_bytes = 10 * gib;
    const headroom_bound = Budget.compute(headroom_in);
    try std.testing.expectEqual(@as(u64, 14 * gib), headroom_bound.limit_bytes);
    try std.testing.expectEqual(BindingLimit.physical_headroom, headroom_bound.binding_limit);

    var cap_in = base;
    cap_in.configured_cap = 12 * gib;
    const cap_bound = Budget.compute(cap_in);
    try std.testing.expectEqual(@as(u64, 12 * gib), cap_bound.limit_bytes);
    try std.testing.expectEqual(BindingLimit.configured_cap, cap_bound.binding_limit);
}

test "compute rounds down to whole 16 KiB pages and saturates when headroom exceeds RAM" {
    const weird = Budget.compute(.{
        .memory_bytes = 10 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = 5 * page.APPLE_PAGE_SIZE + 123,
        .global_user_wire_limit = 10 * page.APPLE_PAGE_SIZE,
        .gpu_working_set_bytes = 10 * page.APPLE_PAGE_SIZE,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    try std.testing.expectEqual(@as(u64, 5 * page.APPLE_PAGE_SIZE), weird.limit_bytes);
    try std.testing.expectEqual(BindingLimit.user_wire_limit, weird.binding_limit);

    const zero = Budget.compute(.{
        .memory_bytes = 2 * 1024 * 1024 * 1024,
        .user_wire_limit = 2 * 1024 * 1024 * 1024,
        .global_user_wire_limit = 2 * 1024 * 1024 * 1024,
        .gpu_working_set_bytes = 2 * 1024 * 1024 * 1024,
        .configured_cap = null,
        .headroom_bytes = DEFAULT_HEADROOM_BYTES,
    });
    try std.testing.expectEqual(@as(u64, 0), zero.limit_bytes);
    try std.testing.expectEqual(BindingLimit.physical_headroom, zero.binding_limit);
}

test "checkPlan accepts plans within budget and rejects excess plans" {
    const budget = Budget.compute(.{
        .memory_bytes = 24 * 1024 * 1024 * 1024,
        .user_wire_limit = 16 * 1024 * 1024,
        .global_user_wire_limit = 32 * 1024 * 1024,
        .gpu_working_set_bytes = 32 * 1024 * 1024,
        .configured_cap = null,
        .headroom_bytes = DEFAULT_HEADROOM_BYTES,
    });
    try budget.checkPlan(16 * 1024 * 1024);
    try std.testing.expectError(error.BudgetExceeded, budget.checkPlan(16 * 1024 * 1024 + 1));
}

test "host budget is positive, 16 KiB aligned and respects default headroom" {
    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const info = try device.info();

    const budget = Budget.fromHost(facts, info, .{});
    try std.testing.expect(budget.limit_bytes > 0);
    try std.testing.expectEqual(@as(u64, 0), budget.limit_bytes % page.APPLE_PAGE_SIZE);
    try std.testing.expect(budget.limit_bytes <= facts.memory_bytes - DEFAULT_HEADROOM_BYTES);
    try std.testing.expect(budget.limit_bytes <= facts.user_wire_limit);
    try std.testing.expect(budget.limit_bytes <= facts.global_user_wire_limit);
    try std.testing.expect(budget.limit_bytes <= info.recommendedMaxWorkingSetSize());
}
