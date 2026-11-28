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
/// Default safety margin held back from the live zero-swap reclaimable pool
/// when sizing optional wired caches (`P1-B`, ADR-0020).
pub const DEFAULT_LIVE_SAFETY_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

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
    /// Under `LiveRamConfig.strict_live_ram = true`, required wired bytes exceed
    /// live zero-swap reclaimable RAM minus `live_safety_margin_bytes` (ADR-0020).
    ExceedsLiveZeroSwapRam,
};

pub const LiveRamConfig = struct {
    /// Safety margin subtracted from `live_reclaimable_bytes` before sizing
    /// optional caches or checking strict live RAM admission (default 512 MiB).
    live_safety_margin_bytes: u64 = DEFAULT_LIVE_SAFETY_MARGIN_BYTES,
    /// When `true`, reject startup with `error.ExceedsLiveZeroSwapRam` if
    /// `required_wired_bytes > live_budget_bytes`. When `false`, required wired
    /// plans stay governed deterministically by the static `ADR-0012` budget
    /// while optional caches auto-scale down softly.
    strict_live_ram: bool = false,
};

pub const LiveAdmissionPlan = struct {
    required_wired_bytes: u64,
    requested_optional_cache_bytes: u64,
    granted_optional_cache_bytes: u64,
    static_limit_bytes: u64,
    live_reclaimable_bytes: u64,
    live_budget_bytes: u64,
    cache_scaled_down: bool,
};

/// Returns the total bytes currently wired across all live `WiredRegion`s in
/// this process.
pub fn currentProcessWiredBytes() u64 {
    return page.currentProcessWiredBytes();
}

pub const Budget = struct {
    /// Maximum bytes aminfer may wire, rounded down to whole 16 KiB pages.
    limit_bytes: u64,
    /// Which system or configuration constraint set `limit_bytes`.
    binding_limit: BindingLimit,
    /// The raw (pre-alignment) ceiling reported by `binding_limit`.
    raw_limit_bytes: u64,
    /// When `true`, `checkPlan` enforces `currentProcessWiredBytes() + required_bytes <= limit_bytes`.
    enforce_process_ledger: bool = false,

    /// Returns a copy of `self` with process-wide wired-ledger accounting enabled.
    pub fn withProcessLedger(self: Budget) Budget {
        var copy = self;
        copy.enforce_process_ledger = true;
        return copy;
    }

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
        }).withProcessLedger();
    }

    /// Rejects any model/arena plan that exceeds the wired budget instead of
    /// silently degrading to pageable memory (ADR-0012).
    pub fn checkPlan(self: Budget, required_bytes: u64) PlanError!void {
        if (self.enforce_process_ledger) {
            const total = std.math.add(u64, currentProcessWiredBytes(), required_bytes) catch {
                return error.BudgetExceeded;
            };
            if (total > self.limit_bytes) return error.BudgetExceeded;
        } else {
            if (required_bytes > self.limit_bytes) return error.BudgetExceeded;
        }
    }

    /// Alias of `checkPlan` enforcing the static `ADR-0012` wired ceiling.
    pub fn check(self: Budget, required_bytes: u64) PlanError!void {
        return self.checkPlan(required_bytes);
    }

    /// Evaluates required wired bytes against the static `ADR-0012` budget and
    /// auto-scales optional wired caches against the live zero-swap reclaimable
    /// pool (`P1-B`, ADR-0020).
    pub fn planWithLiveRam(
        self: Budget,
        required_wired_bytes: u64,
        optional_cache_bytes: u64,
        live_reclaimable_bytes: u64,
        cfg: LiveRamConfig,
    ) PlanError!LiveAdmissionPlan {
        return planWithLiveRamImpl(self, required_wired_bytes, optional_cache_bytes, live_reclaimable_bytes, cfg);
    }
};

fn planWithLiveRamImpl(
    budget: Budget,
    required_wired_bytes: u64,
    optional_cache_bytes: u64,
    live_reclaimable_bytes: u64,
    cfg: LiveRamConfig,
) PlanError!LiveAdmissionPlan {
    try budget.check(required_wired_bytes);
    const live_budget_bytes: u64 = live_reclaimable_bytes -| cfg.live_safety_margin_bytes;
    if (cfg.strict_live_ram and required_wired_bytes > live_budget_bytes) {
        return error.ExceedsLiveZeroSwapRam;
    }
    const base_wired: u64 = if (budget.enforce_process_ledger) currentProcessWiredBytes() else 0;
    const projected_wired: u64 = base_wired +| required_wired_bytes;
    const static_headroom: u64 = budget.limit_bytes -| projected_wired;
    const live_headroom: u64 = live_budget_bytes -| required_wired_bytes;
    const granted_optional_cache_bytes: u64 = @min(optional_cache_bytes, @min(static_headroom, live_headroom));
    return .{
        .required_wired_bytes = required_wired_bytes,
        .requested_optional_cache_bytes = optional_cache_bytes,
        .granted_optional_cache_bytes = granted_optional_cache_bytes,
        .static_limit_bytes = budget.limit_bytes,
        .live_reclaimable_bytes = live_reclaimable_bytes,
        .live_budget_bytes = live_budget_bytes,
        .cache_scaled_down = granted_optional_cache_bytes < optional_cache_bytes,
    };
}

/// Evaluates required wired bytes against the static `ADR-0012` `budget`,
/// optionally enforces `--strict-live-ram`, and caps optional wired caches
/// (`PrefixCache`, `PlePageCache`, `NgramCache`, extra `ExpertSlotPool` slots)
/// within both static headroom and live zero-swap reclaimable headroom (`P1-B`, ADR-0020).
pub fn planWithLiveRam(
    budget: Budget,
    required_wired_bytes: u64,
    optional_cache_bytes: u64,
    live_reclaimable_bytes: u64,
    cfg: LiveRamConfig,
) PlanError!LiveAdmissionPlan {
    return planWithLiveRamImpl(budget, required_wired_bytes, optional_cache_bytes, live_reclaimable_bytes, cfg);
}

/// Scales an optional cache unit count (e.g., `PrefixCache` blocks, `PlePageCache`
/// pages, `NgramCache` slots, or extra `ExpertSlotPool` slots) down to fit
/// `available_bytes`, clamping to `[min(min_count, requested_count), requested_count]`.
pub fn scaleOptionalCount(
    requested_count: usize,
    bytes_per_unit: usize,
    available_bytes: u64,
    min_count: usize,
) usize {
    if (requested_count == 0) return 0;
    if (bytes_per_unit == 0) return requested_count;
    const floor: usize = @min(min_count, requested_count);
    const fit_u64: u64 = available_bytes / @as(u64, bytes_per_unit);
    const fit: usize = @intCast(@min(fit_u64, @as(u64, requested_count)));
    return @max(fit, floor);
}

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

test "planWithLiveRam scales optional caches softly and enforces strict_live_ram" {
    const mib: u64 = 1024 * 1024;
    const gib: u64 = 1024 * mib;
    const budget = Budget.compute(.{
        .memory_bytes = 24 * gib,
        .user_wire_limit = 18 * gib,
        .global_user_wire_limit = 18 * gib,
        .gpu_working_set_bytes = 18 * gib,
        .configured_cap = null,
        .headroom_bytes = DEFAULT_HEADROOM_BYTES,
    });

    // 1. Ample static and live headroom: full optional cache granted without scaling.
    const ample = try planWithLiveRam(budget, 10 * gib, 2 * gib, 16 * gib, .{});
    try std.testing.expectEqual(@as(u64, 10 * gib), ample.required_wired_bytes);
    try std.testing.expectEqual(@as(u64, 2 * gib), ample.requested_optional_cache_bytes);
    try std.testing.expectEqual(@as(u64, 2 * gib), ample.granted_optional_cache_bytes);
    try std.testing.expectEqual(@as(u64, 18 * gib), ample.static_limit_bytes);
    try std.testing.expectEqual(@as(u64, 16 * gib), ample.live_reclaimable_bytes);
    try std.testing.expectEqual(@as(u64, 16 * gib - DEFAULT_LIVE_SAFETY_MARGIN_BYTES), ample.live_budget_bytes);
    try std.testing.expect(!ample.cache_scaled_down);

    // 2. Desktop processes occupy RAM: live_reclaimable = 11.5 GiB -> live_budget = 11 GiB ->
    //    only 1 GiB optional cache headroom left, so 2 GiB optional cache softly scales to 1 GiB.
    const scaled = try budget.planWithLiveRam(10 * gib, 2 * gib, 11 * gib + 512 * mib, .{});
    try std.testing.expectEqual(@as(u64, 11 * gib), scaled.live_budget_bytes);
    try std.testing.expectEqual(@as(u64, 1 * gib), scaled.granted_optional_cache_bytes);
    try std.testing.expect(scaled.cache_scaled_down);

    // 3. Required wired plan fits static budget (12 <= 18 GiB) but exceeds live budget (9.5 GiB):
    //    default soft mode succeeds with 0 optional cache; strict_live_ram rejects immediately.
    const soft_zero = try planWithLiveRam(budget, 12 * gib, 1 * gib, 10 * gib, .{ .strict_live_ram = false });
    try std.testing.expectEqual(@as(u64, 0), soft_zero.granted_optional_cache_bytes);
    try std.testing.expect(soft_zero.cache_scaled_down);

    try std.testing.expectError(
        error.ExceedsLiveZeroSwapRam,
        planWithLiveRam(budget, 12 * gib, 1 * gib, 10 * gib, .{ .strict_live_ram = true }),
    );

    // 4. Static ADR-0012 limit still rejects excess required wired plans regardless of live RAM.
    try std.testing.expectError(
        error.BudgetExceeded,
        planWithLiveRam(budget, 18 * gib + page.APPLE_PAGE_SIZE, 0, 22 * gib, .{ .strict_live_ram = false }),
    );

    // 5. Static headroom tighter than live headroom clamps optional cache to static headroom.
    const static_clamped = try planWithLiveRam(budget, 17 * gib, 2 * gib, 22 * gib, .{ .strict_live_ram = true });
    try std.testing.expectEqual(@as(u64, 1 * gib), static_clamped.granted_optional_cache_bytes);
    try std.testing.expect(static_clamped.cache_scaled_down);
}

test "scaleOptionalCount clamps cleanly between min_count and requested_count" {
    const unit: usize = 64 * 1024;
    try std.testing.expectEqual(@as(usize, 16), scaleOptionalCount(16, unit, 2 * 1024 * 1024, 2));
    try std.testing.expectEqual(@as(usize, 5), scaleOptionalCount(16, unit, 5 * unit + 1234, 2));
    try std.testing.expectEqual(@as(usize, 2), scaleOptionalCount(16, unit, unit, 2));
    try std.testing.expectEqual(@as(usize, 0), scaleOptionalCount(16, unit, 0, 0));
    try std.testing.expectEqual(@as(usize, 0), scaleOptionalCount(0, unit, 1024 * 1024, 4));
    try std.testing.expectEqual(@as(usize, 3), scaleOptionalCount(3, unit, 0, 8));
    try std.testing.expectEqual(@as(usize, 16), scaleOptionalCount(16, 0, 0, 2));
}

test "process ledger enforces combined component A + component B wired ceiling" {
    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const info = try device.info();

    const base_wired = currentProcessWiredBytes();
    const page_sz: u64 = page.APPLE_PAGE_SIZE;
    // Cap at baseline + 5 pages so component A (3 pages) + component B (3 pages) exceeds budget.
    const host_budget = Budget.fromHost(facts, info, .{
        .configured_cap = base_wired + 5 * page_sz,
    });
    try std.testing.expect(host_budget.enforce_process_ledger);

    const comp_a_bytes: usize = 3 * page.APPLE_PAGE_SIZE;
    const comp_b_bytes: usize = 3 * page.APPLE_PAGE_SIZE;

    try host_budget.checkPlan(comp_a_bytes);
    const region_a = try page.WiredRegion.init(comp_a_bytes);
    var region_a_active = true;
    defer if (region_a_active) region_a.deinit();

    try std.testing.expectEqual(base_wired + comp_a_bytes, currentProcessWiredBytes());

    // A fresh Budget.fromHost recomputed by another subsystem still sees component A wired.
    const second_budget = Budget.fromHost(facts, info, .{
        .configured_cap = base_wired + 5 * page_sz,
    });
    try std.testing.expectError(error.BudgetExceeded, second_budget.checkPlan(comp_b_bytes));
    try std.testing.expectError(error.BudgetExceeded, second_budget.checkPlan(std.math.maxInt(u64)));
    try second_budget.checkPlan(2 * page_sz);

    region_a.deinit();
    region_a_active = false;
    try std.testing.expectEqual(base_wired, currentProcessWiredBytes());
    try second_budget.checkPlan(comp_b_bytes);
}
