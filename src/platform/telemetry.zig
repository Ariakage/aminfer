//! Rootless Apple Silicon host telemetry (`C19`, ADR-0022):
//! - Per-core Mach `PROCESSOR_CPU_LOAD_INFO` ticks split across P-cores and
//!   E-cores (`IODeviceTree:/cpus/cpuN` `cluster-type` + `sysinfo.Facts` fallback).
//! - Unprivileged `IOKit` `IOAccelerator` (`PerformanceStatistics`) GPU utilization
//!   and memory counters paired with `NSProcessInfo.thermalState` (`0` `sudo`).
//! - Unprivileged `IOKit` `IOBlockStorageDriver` (`Statistics`) cumulative NVMe
//!   bytes and IOPS (`0` `sudo`).
//! - Live `HOST_VM_INFO64` + `ADR-0012` wired budget + `R9` zero-swap reclaimable
//!   pool + `vm.swapusage` / `swapouts` delta tracking (`0` per-tick heap allocs).

const std = @import("std");
const c = @import("apple_bridge");
const sysinfo = @import("sysinfo.zig");
const vmstat = @import("vmstat.zig");
const chip_profile = @import("chip_profile.zig");
const qos = @import("qos.zig");
const metal = @import("../bridge/metal.zig");
const budget = @import("../mem/budget.zig");

pub const MAX_CPU_CORES: usize = c.AMI_MAX_CPU_CORES;

pub const Error = sysinfo.Error || vmstat.Error || metal.Error || error{
    InvalidArgument,
    TelemetryFailed,
    TtyFailed,
};

fn checkBridgeStatus(st: c.ami_status) Error!void {
    return switch (st) {
        c.AMI_OK => {},
        c.AMI_ERR_INVALID_ARGUMENT => error.InvalidArgument,
        c.AMI_ERR_TTY => error.TtyFailed,
        else => error.TelemetryFailed,
    };
}

pub const CoreKind = enum(u8) {
    unknown = 0,
    efficiency = 1,
    performance = 2,

    pub fn shortLabel(self: CoreKind) []const u8 {
        return switch (self) {
            .performance => "P",
            .efficiency => "E",
            .unknown => "C",
        };
    }
};

pub const ThermalState = enum(u8) {
    nominal = 0,
    fair = 1,
    serious = 2,
    critical = 3,

    pub fn fromBridge(raw: u32) ThermalState {
        return switch (raw) {
            c.AMI_THERMAL_NOMINAL => .nominal,
            c.AMI_THERMAL_FAIR => .fair,
            c.AMI_THERMAL_SERIOUS => .serious,
            c.AMI_THERMAL_CRITICAL => .critical,
            else => .nominal,
        };
    }

    pub fn label(self: ThermalState) []const u8 {
        return switch (self) {
            .nominal => "Nominal",
            .fair => "Fair",
            .serious => "Serious",
            .critical => "Critical",
        };
    }

    /// Maps thermal pressure to `[0.0, 1.0]` for smooth `temp` gradient coloring.
    pub fn severityRatio(self: ThermalState) f32 {
        return switch (self) {
            .nominal => 0.10,
            .fair => 0.45,
            .serious => 0.80,
            .critical => 1.00,
        };
    }
};

pub const CoreTicks = struct {
    user: u64 = 0,
    system: u64 = 0,
    idle: u64 = 0,
    nice: u64 = 0,
    kind: CoreKind = .unknown,

    pub fn busyTicks(self: CoreTicks) u64 {
        return self.user +| self.system +| self.nice;
    }

    pub fn totalTicks(self: CoreTicks) u64 {
        return self.busyTicks() +| self.idle;
    }

    pub fn utilizationDeltaPct(prev: CoreTicks, curr: CoreTicks) f32 {
        const d_busy: u64 = curr.busyTicks() -| prev.busyTicks();
        const d_idle: u64 = curr.idle -| prev.idle;
        const d_total: u64 = d_busy +| d_idle;
        if (d_total == 0) return 0.0;
        const pct = (@as(f64, @floatFromInt(d_busy)) * 100.0) / @as(f64, @floatFromInt(d_total));
        return @floatCast(std.math.clamp(pct, 0.0, 100.0));
    }
};

pub const RawCpuSample = struct {
    core_count: u32 = 0,
    loadavg_1m: f64 = 0.0,
    loadavg_5m: f64 = 0.0,
    loadavg_15m: f64 = 0.0,
    cores: [MAX_CPU_CORES]CoreTicks = @splat(.{}),

    pub fn capture() Error!RawCpuSample {
        var raw: c.ami_telemetry_cpu_sample = std.mem.zeroes(c.ami_telemetry_cpu_sample);
        try checkBridgeStatus(c.ami_telemetry_sample_cpu(&raw));
        var out: RawCpuSample = .{
            .core_count = @min(raw.core_count, @as(u32, MAX_CPU_CORES)),
            .loadavg_1m = raw.loadavg_1m,
            .loadavg_5m = raw.loadavg_5m,
            .loadavg_15m = raw.loadavg_15m,
        };
        const n: usize = out.core_count;
        for (0..n) |i| {
            const k: CoreKind = switch (raw.cores[i].cluster_kind) {
                c.AMI_CPU_CLUSTER_E => .efficiency,
                c.AMI_CPU_CLUSTER_P => .performance,
                else => .unknown,
            };
            out.cores[i] = .{
                .user = raw.cores[i].user,
                .system = raw.cores[i].system,
                .idle = raw.cores[i].idle,
                .nice = raw.cores[i].nice,
                .kind = k,
            };
        }
        return out;
    }
};

pub const CpuTelemetry = struct {
    core_count: u32 = 0,
    p_core_count: u32 = 0,
    e_core_count: u32 = 0,
    total_pct: f32 = 0.0,
    p_core_pct: f32 = 0.0,
    e_core_pct: f32 = 0.0,
    loadavg_1m: f64 = 0.0,
    loadavg_5m: f64 = 0.0,
    loadavg_15m: f64 = 0.0,
    per_core_pct: [MAX_CPU_CORES]f32 = @splat(0.0),
    per_core_kind: [MAX_CPU_CORES]CoreKind = @splat(.unknown),
};

pub const MemoryTelemetry = struct {
    total_bytes: u64 = 0,
    page_size: u64 = 16384,
    wired_bytes: u64 = 0,
    wired_budget_bytes: u64 = 0,
    binding_limit: budget.BindingLimit = .physical_headroom,
    user_wire_limit_bytes: u64 = 0,
    global_user_wire_limit_bytes: u64 = 0,
    gpu_wired_limit_bytes: u64 = 0,
    reclaimable_zero_swap_bytes: u64 = 0,
    free_bytes: u64 = 0,
    speculative_bytes: u64 = 0,
    purgeable_bytes: u64 = 0,
    external_bytes: u64 = 0,
    active_bytes: u64 = 0,
    inactive_bytes: u64 = 0,
    internal_bytes: u64 = 0,
    compressor_bytes: u64 = 0,
    swap_used_bytes: u64 = 0,
    swap_total_bytes: u64 = 0,
    swap_avail_bytes: u64 = 0,
    swapouts_total: u64 = 0,
    swapouts_delta: u64 = 0,
    pageouts_total: u64 = 0,
    pageouts_delta: u64 = 0,

    pub fn usedBytes(self: MemoryTelemetry) u64 {
        return self.total_bytes -| self.reclaimable_zero_swap_bytes;
    }

    pub fn wiredRatioPct(self: MemoryTelemetry) f32 {
        if (self.wired_budget_bytes == 0) return 0.0;
        const pct = (@as(f64, @floatFromInt(self.wired_bytes)) * 100.0) /
            @as(f64, @floatFromInt(self.wired_budget_bytes));
        return @floatCast(std.math.clamp(pct, 0.0, 100.0));
    }

    pub fn usedRatioPct(self: MemoryTelemetry) f32 {
        if (self.total_bytes == 0) return 0.0;
        const pct = (@as(f64, @floatFromInt(self.usedBytes())) * 100.0) /
            @as(f64, @floatFromInt(self.total_bytes));
        return @floatCast(std.math.clamp(pct, 0.0, 100.0));
    }
};

pub const GpuTelemetry = struct {
    available: bool = false,
    device_utilization_pct: u32 = 0,
    renderer_utilization_pct: u32 = 0,
    tiler_utilization_pct: u32 = 0,
    thermal_state: ThermalState = .nominal,
    recovery_count: u32 = 0,
    in_use_system_memory: u64 = 0,
    alloc_system_memory: u64 = 0,
    driver_in_use_memory: u64 = 0,
    tiled_scene_bytes: u64 = 0,
    accelerator_class_buf: [64]u8 = @splat(0),
    accelerator_class_len: u8 = 0,

    pub fn acceleratorClass(self: *const GpuTelemetry) []const u8 {
        const n: usize = @min(@as(usize, self.accelerator_class_len), self.accelerator_class_buf.len);
        return self.accelerator_class_buf[0..n];
    }

    pub fn capture() Error!GpuTelemetry {
        var raw: c.ami_telemetry_gpu_sample = std.mem.zeroes(c.ami_telemetry_gpu_sample);
        try checkBridgeStatus(c.ami_telemetry_sample_gpu(&raw));
        var out: GpuTelemetry = .{
            .available = raw.available != 0,
            .device_utilization_pct = @min(raw.device_utilization_pct, 100),
            .renderer_utilization_pct = @min(raw.renderer_utilization_pct, 100),
            .tiler_utilization_pct = @min(raw.tiler_utilization_pct, 100),
            .thermal_state = ThermalState.fromBridge(raw.thermal_state),
            .recovery_count = raw.recovery_count,
            .in_use_system_memory = raw.in_use_system_memory,
            .alloc_system_memory = raw.alloc_system_memory,
            .driver_in_use_memory = raw.driver_in_use_memory,
            .tiled_scene_bytes = raw.tiled_scene_bytes,
        };
        const raw_bytes: *const [64]u8 = @ptrCast(&raw.accelerator_class);
        const slen = std.mem.indexOfScalar(u8, raw_bytes, 0) orelse raw_bytes.len;
        @memcpy(out.accelerator_class_buf[0..slen], raw_bytes[0..slen]);
        out.accelerator_class_len = @intCast(slen);
        return out;
    }
};

pub const RawNvmeSample = struct {
    available: bool = false,
    driver_count: u32 = 0,
    bytes_read: u64 = 0,
    bytes_written: u64 = 0,
    ops_read: u64 = 0,
    ops_written: u64 = 0,
    read_time_ns: u64 = 0,
    write_time_ns: u64 = 0,

    pub fn capture() Error!RawNvmeSample {
        var raw: c.ami_telemetry_nvme_sample = std.mem.zeroes(c.ami_telemetry_nvme_sample);
        try checkBridgeStatus(c.ami_telemetry_sample_nvme(&raw));
        return .{
            .available = raw.available != 0,
            .driver_count = raw.driver_count,
            .bytes_read = raw.bytes_read,
            .bytes_written = raw.bytes_written,
            .ops_read = raw.ops_read,
            .ops_written = raw.ops_written,
            .read_time_ns = raw.read_time_ns,
            .write_time_ns = raw.write_time_ns,
        };
    }
};

pub const NvmeTelemetry = struct {
    available: bool = false,
    driver_count: u32 = 0,
    bytes_read_total: u64 = 0,
    bytes_written_total: u64 = 0,
    bytes_read_delta: u64 = 0,
    bytes_written_delta: u64 = 0,
    ops_read_total: u64 = 0,
    ops_written_total: u64 = 0,
    ops_read_delta: u64 = 0,
    ops_written_delta: u64 = 0,
    read_mb_s: f64 = 0.0,
    write_mb_s: f64 = 0.0,
    read_iops: f64 = 0.0,
    write_iops: f64 = 0.0,
};

pub const HostSnapshot = struct {
    timestamp_ns: u64 = 0,
    uptime_s: f64 = 0.0,
    chip: chip_profile.ChipProfile,
    facts: sysinfo.Facts,
    device_name_buf: [128]u8 = @splat(0),
    device_name_len: u8 = 0,
    cpu: CpuTelemetry = .{},
    mem: MemoryTelemetry = .{},
    gpu: GpuTelemetry = .{},
    nvme: NvmeTelemetry = .{},

    pub fn deviceName(self: *const HostSnapshot) []const u8 {
        const n: usize = @min(@as(usize, self.device_name_len), self.device_name_buf.len);
        const s = self.device_name_buf[0..n];
        if (s.len > 0) return s;
        return self.facts.cpuBrand();
    }
};

/// Zero-allocation stateful host telemetry sampler tracking per-tick rates and
/// resettable (`r`) cumulative baseline deltas.
pub const Sampler = struct {
    facts: sysinfo.Facts,
    chip: chip_profile.ChipProfile,
    wired_budget: budget.Budget,
    device_name_buf: [128]u8 = @splat(0),
    device_name_len: u8 = 0,

    baseline_ns: u64 = 0,
    baseline_swapouts: u64 = 0,
    baseline_pageouts: u64 = 0,
    baseline_nvme: RawNvmeSample = .{},
    has_baseline: bool = false,

    prev_ns: u64 = 0,
    prev_cpu: RawCpuSample = .{},
    prev_nvme: RawNvmeSample = .{},
    has_prev: bool = false,

    /// Initializes the sampler once at startup. Queries `sysinfo.Facts` and
    /// optionally probes `metal.Device` once (releasing it immediately) so
    /// `ChipProfile` and `Budget` reflect exact Metal working-set & Apple GPU family.
    pub fn init() Error!Sampler {
        const facts = try sysinfo.Facts.collect();
        var dev_name_buf: [128]u8 = @splat(0);
        var dev_name_len: u8 = 0;
        var gpu_apple_family: u8 = 0;
        var gpu_working_set: u64 = if (facts.gpu_wired_limit_mib > 0)
            facts.gpu_wired_limit_mib * 1024 * 1024
        else
            facts.user_wire_limit;

        if (metal.Device.open()) |dev| {
            defer dev.close();
            const info = try dev.info();
            const name = info.name();
            const copy_len = @min(name.len, dev_name_buf.len);
            @memcpy(dev_name_buf[0..copy_len], name[0..copy_len]);
            dev_name_len = @intCast(copy_len);
            gpu_apple_family = info.appleGpuFamily();
            if (info.recommendedMaxWorkingSetSize() > 0) {
                gpu_working_set = info.recommendedMaxWorkingSetSize();
            }
        } else |err| switch (err) {
            error.NoDevice => {},
            else => return err,
        }

        const fallback_brand = facts.cpuBrand();
        const name_for_chip = if (dev_name_len > 0) dev_name_buf[0..dev_name_len] else fallback_brand;
        const chip = chip_profile.ChipProfile.resolve(facts, name_for_chip, gpu_apple_family, null);
        const wb = budget.Budget.compute(.{
            .memory_bytes = facts.memory_bytes,
            .user_wire_limit = facts.user_wire_limit,
            .global_user_wire_limit = facts.global_user_wire_limit,
            .gpu_working_set_bytes = gpu_working_set,
        });

        var s: Sampler = .{
            .facts = facts,
            .chip = chip,
            .wired_budget = wb,
            .device_name_buf = dev_name_buf,
            .device_name_len = dev_name_len,
        };
        // Prime `prev_cpu` / `prev_nvme` + baseline so the first tick has a valid baseline.
        const t0 = try vmstat.nowNs();
        const vm0 = try vmstat.Snapshot.capture();
        const cpu0 = try RawCpuSample.capture();
        const nvme0 = try RawNvmeSample.capture();
        s.baseline_ns = t0;
        s.baseline_swapouts = vm0.swapouts;
        s.baseline_pageouts = vm0.pageouts;
        s.baseline_nvme = nvme0;
        s.has_baseline = true;
        s.prev_ns = t0;
        s.prev_cpu = cpu0;
        s.prev_nvme = nvme0;
        s.has_prev = true;
        return s;
    }

    /// Steers the current monitor thread to `QOS_CLASS_UTILITY` E-cores (`ADR-0022`).
    pub fn steerUtilityQos() qos.Error!void {
        try qos.setCurrentThreadQos(.utility, 0);
    }

    /// Resets cumulative baseline deltas (`r` interactive keybinding).
    pub fn resetDeltas(self: *Sampler) Error!void {
        const now = try vmstat.nowNs();
        const vm = try vmstat.Snapshot.capture();
        const nvme = try RawNvmeSample.capture();
        self.baseline_ns = now;
        self.baseline_swapouts = vm.swapouts;
        self.baseline_pageouts = vm.pageouts;
        self.baseline_nvme = nvme;
        self.has_baseline = true;
    }

    /// Captures one complete host telemetry snapshot with zero heap allocations.
    pub fn sample(self: *Sampler) Error!HostSnapshot {
        const now = try vmstat.nowNs();
        const vm = try vmstat.Snapshot.capture();
        const raw_cpu = try RawCpuSample.capture();
        const gpu = try GpuTelemetry.capture();
        const raw_nvme = try RawNvmeSample.capture();

        if (!self.has_baseline) {
            self.baseline_ns = now;
            self.baseline_swapouts = vm.swapouts;
            self.baseline_pageouts = vm.pageouts;
            self.baseline_nvme = raw_nvme;
            self.has_baseline = true;
        }

        const dt_ns: u64 = if (self.has_prev and now > self.prev_ns) now - self.prev_ns else 0;
        const dt_s: f64 = @as(f64, @floatFromInt(dt_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));

        const cpu = computeCpuDelta(
            if (self.has_prev) &self.prev_cpu else null,
            &raw_cpu,
            self.facts,
        );

        const ps: u64 = if (vm.page_size > 0) vm.page_size else 16384;
        const gpu_wired_limit_bytes: u64 = if (self.facts.gpu_wired_limit_mib > 0)
            self.facts.gpu_wired_limit_mib * 1024 * 1024
        else
            self.wired_budget.limit_bytes;

        const mem: MemoryTelemetry = .{
            .total_bytes = self.facts.memory_bytes,
            .page_size = ps,
            .wired_bytes = vm.wire_pages *| ps,
            .wired_budget_bytes = self.wired_budget.limit_bytes,
            .binding_limit = self.wired_budget.binding_limit,
            .user_wire_limit_bytes = self.facts.user_wire_limit,
            .global_user_wire_limit_bytes = self.facts.global_user_wire_limit,
            .gpu_wired_limit_bytes = gpu_wired_limit_bytes,
            .reclaimable_zero_swap_bytes = vm.liveZeroSwapReclaimableBytes(),
            .free_bytes = vm.free_pages *| ps,
            .speculative_bytes = vm.speculative_pages *| ps,
            .purgeable_bytes = vm.purgeable_pages *| ps,
            .external_bytes = vm.external_pages *| ps,
            .active_bytes = vm.active_pages *| ps,
            .inactive_bytes = vm.inactive_pages *| ps,
            .internal_bytes = vm.internal_pages *| ps,
            .compressor_bytes = vm.compressor_pages *| ps,
            .swap_used_bytes = vm.swap_used_bytes,
            .swap_total_bytes = vm.swap_total_bytes,
            .swap_avail_bytes = vm.swap_avail_bytes,
            .swapouts_total = vm.swapouts,
            .swapouts_delta = vm.swapouts -| self.baseline_swapouts,
            .pageouts_total = vm.pageouts,
            .pageouts_delta = vm.pageouts -| self.baseline_pageouts,
        };

        const tick_r_bytes: u64 = if (self.has_prev) raw_nvme.bytes_read -| self.prev_nvme.bytes_read else 0;
        const tick_w_bytes: u64 = if (self.has_prev) raw_nvme.bytes_written -| self.prev_nvme.bytes_written else 0;
        const tick_r_ops: u64 = if (self.has_prev) raw_nvme.ops_read -| self.prev_nvme.ops_read else 0;
        const tick_w_ops: u64 = if (self.has_prev) raw_nvme.ops_written -| self.prev_nvme.ops_written else 0;

        const mib: f64 = 1024.0 * 1024.0;
        const nvme: NvmeTelemetry = .{
            .available = raw_nvme.available,
            .driver_count = raw_nvme.driver_count,
            .bytes_read_total = raw_nvme.bytes_read,
            .bytes_written_total = raw_nvme.bytes_written,
            .bytes_read_delta = raw_nvme.bytes_read -| self.baseline_nvme.bytes_read,
            .bytes_written_delta = raw_nvme.bytes_written -| self.baseline_nvme.bytes_written,
            .ops_read_total = raw_nvme.ops_read,
            .ops_written_total = raw_nvme.ops_written,
            .ops_read_delta = raw_nvme.ops_read -| self.baseline_nvme.ops_read,
            .ops_written_delta = raw_nvme.ops_written -| self.baseline_nvme.ops_written,
            .read_mb_s = if (dt_s > 0.0) (@as(f64, @floatFromInt(tick_r_bytes)) / mib) / dt_s else 0.0,
            .write_mb_s = if (dt_s > 0.0) (@as(f64, @floatFromInt(tick_w_bytes)) / mib) / dt_s else 0.0,
            .read_iops = if (dt_s > 0.0) @as(f64, @floatFromInt(tick_r_ops)) / dt_s else 0.0,
            .write_iops = if (dt_s > 0.0) @as(f64, @floatFromInt(tick_w_ops)) / dt_s else 0.0,
        };

        self.prev_ns = now;
        self.prev_cpu = raw_cpu;
        self.prev_nvme = raw_nvme;
        self.has_prev = true;

        return .{
            .timestamp_ns = now,
            .uptime_s = @as(f64, @floatFromInt(now -| self.baseline_ns)) /
                @as(f64, @floatFromInt(std.time.ns_per_s)),
            .chip = self.chip,
            .facts = self.facts,
            .device_name_buf = self.device_name_buf,
            .device_name_len = self.device_name_len,
            .cpu = cpu,
            .mem = mem,
            .gpu = gpu,
            .nvme = nvme,
        };
    }
};

pub fn computeCpuDelta(
    maybe_prev: ?*const RawCpuSample,
    curr: *const RawCpuSample,
    facts: sysinfo.Facts,
) CpuTelemetry {
    var out: CpuTelemetry = .{
        .core_count = @min(curr.core_count, @as(u32, MAX_CPU_CORES)),
        .loadavg_1m = curr.loadavg_1m,
        .loadavg_5m = curr.loadavg_5m,
        .loadavg_15m = curr.loadavg_15m,
    };
    const n: usize = out.core_count;
    const fallback_e: usize = @intCast(@min(facts.efficiency_cores, n));

    var sum_all: f64 = 0.0;
    var sum_p: f64 = 0.0;
    var sum_e: f64 = 0.0;
    var cnt_p: u32 = 0;
    var cnt_e: u32 = 0;

    for (0..n) |i| {
        const resolved_kind: CoreKind = switch (curr.cores[i].kind) {
            .efficiency => .efficiency,
            .performance => .performance,
            .unknown => if (i < fallback_e) .efficiency else .performance,
        };
        out.per_core_kind[i] = resolved_kind;

        const pct: f32 = if (maybe_prev) |prev|
            if (i < prev.core_count)
                CoreTicks.utilizationDeltaPct(prev.cores[i], curr.cores[i])
            else
                0.0
        else
            0.0;

        out.per_core_pct[i] = pct;
        sum_all += pct;
        switch (resolved_kind) {
            .performance => {
                sum_p += pct;
                cnt_p += 1;
            },
            .efficiency => {
                sum_e += pct;
                cnt_e += 1;
            },
            .unknown => {},
        }
    }

    out.p_core_count = cnt_p;
    out.e_core_count = cnt_e;
    out.total_pct = if (n > 0) @floatCast(sum_all / @as(f64, @floatFromInt(n))) else 0.0;
    out.p_core_pct = if (cnt_p > 0) @floatCast(sum_p / @as(f64, @floatFromInt(cnt_p))) else 0.0;
    out.e_core_pct = if (cnt_e > 0) @floatCast(sum_e / @as(f64, @floatFromInt(cnt_e))) else 0.0;
    return out;
}

// ---------------------------------------------------------------------------
// Terminal Raw-Mode & Window Size Wrappers (`termios` + `TIOCGWINSZ`)
// ---------------------------------------------------------------------------

pub const WinSize = struct {
    cols: u16,
    rows: u16,
};

pub fn isTty(fd: std.c.fd_t) bool {
    return c.ami_tty_isatty(fd) != 0;
}

pub fn getWinSize(fd: std.c.fd_t) Error!WinSize {
    var cols: u16 = 0;
    var rows: u16 = 0;
    try checkBridgeStatus(c.ami_tty_get_winsize(fd, &cols, &rows));
    return .{ .cols = cols, .rows = rows };
}

pub const TtyRawGuard = struct {
    raw: c.ami_tty_raw_guard = std.mem.zeroes(c.ami_tty_raw_guard),

    pub fn enter(fd: std.c.fd_t) Error!TtyRawGuard {
        var g: TtyRawGuard = .{};
        try checkBridgeStatus(c.ami_tty_enter_raw(fd, &g.raw));
        return g;
    }

    pub fn leave(self: *TtyRawGuard) void {
        _ = c.ami_tty_leave_raw(&self.raw);
    }
};

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

test "live rootless M4 CPU, GPU IOAccelerator, ThermalState, and NVMe probes succeed with zero steady-state heap growth" {
    var sampler = try Sampler.init();

    // Short sleep so tick delta has real elapsed nanoseconds.
    const req: std.c.timespec = .{ .sec = 0, .nsec = 15 * std.time.ns_per_ms };
    _ = std.c.nanosleep(&req, null);

    const snap = try sampler.sample();

    // 1. CPU & P/E cluster split must match sysinfo.Facts on Apple Silicon.
    const expected_cores: u32 = @intCast(snap.facts.performance_cores + snap.facts.efficiency_cores);
    try std.testing.expectEqual(expected_cores, snap.cpu.core_count);
    try std.testing.expectEqual(@as(u32, @intCast(snap.facts.performance_cores)), snap.cpu.p_core_count);
    try std.testing.expectEqual(@as(u32, @intCast(snap.facts.efficiency_cores)), snap.cpu.e_core_count);
    try std.testing.expect(snap.cpu.total_pct >= 0.0 and snap.cpu.total_pct <= 100.0);
    try std.testing.expect(snap.cpu.loadavg_1m >= 0.0);

    // 2. Unified memory & R9 zero-swap pool invariants.
    try std.testing.expect(snap.mem.total_bytes > 0);
    try std.testing.expect(snap.mem.wired_bytes > 0);
    try std.testing.expect(snap.mem.wired_budget_bytes > 0);
    try std.testing.expectEqual(
        snap.mem.free_bytes + snap.mem.speculative_bytes + snap.mem.purgeable_bytes + snap.mem.external_bytes,
        snap.mem.reclaimable_zero_swap_bytes,
    );

    // 3. Rootless Metal GPU IOAccelerator & NSProcessInfo.thermalState (`0` sudo).
    try std.testing.expect(snap.gpu.available);
    try std.testing.expect(snap.gpu.device_utilization_pct <= 100);
    try std.testing.expect(snap.gpu.renderer_utilization_pct <= 100);
    try std.testing.expect(snap.gpu.tiler_utilization_pct <= 100);
    try std.testing.expect(snap.gpu.alloc_system_memory > 0);
    try std.testing.expect(snap.gpu.acceleratorClass().len > 0);
    try std.testing.expect(snap.gpu.thermal_state.label().len > 0);

    // 4. Rootless NVMe IOBlockStorageDriver cumulative counters.
    try std.testing.expect(snap.nvme.available);
    try std.testing.expect(snap.nvme.driver_count >= 1);
    try std.testing.expect(snap.nvme.bytes_read_total > 0);
    try std.testing.expect(snap.nvme.ops_read_total > 0);

    // 5. Verify resetDeltas (`r` keybinding) zeroes baseline deltas cleanly.
    try sampler.resetDeltas();
    const after_reset = try sampler.sample();
    try std.testing.expectEqual(@as(u64, 0), after_reset.mem.swapouts_delta);

    // 6. Run 16 consecutive steady-state samples and verify bounded C heap usage
    // (confirming Mach processor_info vm_deallocate + CFRelease + @autoreleasepool leak 0 pages).
    const vm_before = try vmstat.Snapshot.capture();
    for (0..16) |_| {
        _ = try sampler.sample();
    }
    const vm_after = try vmstat.Snapshot.capture();
    const heap_delta = vmstat.heapDelta(vm_before, vm_after);
    try std.testing.expect(heap_delta.size_in_use < 256 * 1024);
}

test "computeCpuDelta accurately computes P-core and E-core utilization percentages" {
    var facts: sysinfo.Facts = try sysinfo.Facts.collect();
    facts.efficiency_cores = 2;
    facts.performance_cores = 2;

    var prev: RawCpuSample = .{ .core_count = 4 };
    var curr: RawCpuSample = .{ .core_count = 4 };
    // Core 0 (E): 25% busy, Core 1 (E): 75% busy -> E mean = 50%
    prev.cores[0] = .{ .user = 100, .idle = 100, .kind = .efficiency };
    curr.cores[0] = .{ .user = 125, .idle = 175, .kind = .efficiency };
    prev.cores[1] = .{ .user = 100, .idle = 100, .kind = .efficiency };
    curr.cores[1] = .{ .user = 175, .idle = 125, .kind = .efficiency };

    // Core 2 (P): 100% busy, Core 3 (P): 50% busy -> P mean = 75%
    prev.cores[2] = .{ .user = 200, .idle = 200, .kind = .performance };
    curr.cores[2] = .{ .user = 300, .idle = 200, .kind = .performance };
    prev.cores[3] = .{ .user = 200, .idle = 200, .kind = .performance };
    curr.cores[3] = .{ .user = 250, .idle = 250, .kind = .performance };

    const delta = computeCpuDelta(&prev, &curr, facts);
    try std.testing.expectEqual(@as(u32, 2), delta.e_core_count);
    try std.testing.expectEqual(@as(u32, 2), delta.p_core_count);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), delta.e_core_pct, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 75.0), delta.p_core_pct, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 62.5), delta.total_pct, 0.01);
}
