//! Native macOS VM, swap, malloc heap, and monotonic clock probe.
//! Reads Mach `host_statistics64(HOST_VM_INFO64)`, `sysctlbyname("vm.swapusage")`,
//! `malloc_zone_statistics(null, ...)`, and `clock_gettime(CLOCK_UPTIME_RAW)`
//! without spawning subprocesses or allocating on the heap.

const std = @import("std");
const page = @import("../mem/page.zig");

const HOST_VM_INFO64: c_int = 4;

/// Matches `struct vm_statistics64` through rev1 (`HOST_VM_INFO64_REV1_COUNT`)
/// in `<mach/vm_statistics.h>`, which carries `pageouts` and `swapouts`.
const VmStatistics64 = extern struct {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
};

const HOST_VM_INFO64_REV1_COUNT: u32 = @sizeOf(VmStatistics64) / @sizeOf(c_int);

/// Matches `struct xsw_usage` in `<sys/sysctl.h>`.
const XswUsage = extern struct {
    xsu_total: u64,
    xsu_avail: u64,
    xsu_used: u64,
    xsu_pagesize: u32,
    xsu_encrypted: c_int,
};

/// Matches `malloc_statistics_t` in `<malloc/malloc.h>`.
const MallocStatistics = extern struct {
    blocks_in_use: c_uint,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

comptime {
    if (@sizeOf(VmStatistics64) != 152) @compileError("VmStatistics64 rev1 must be 152 bytes");
    if (@offsetOf(VmStatistics64, "pageouts") != 40) @compileError("VmStatistics64.pageouts offset mismatch");
    if (@offsetOf(VmStatistics64, "purgeable_count") != 88) @compileError("VmStatistics64.purgeable_count offset mismatch");
    if (@offsetOf(VmStatistics64, "speculative_count") != 92) @compileError("VmStatistics64.speculative_count offset mismatch");
    if (@offsetOf(VmStatistics64, "swapouts") != 120) @compileError("VmStatistics64.swapouts offset mismatch");
    if (@offsetOf(VmStatistics64, "compressor_page_count") != 128) @compileError("VmStatistics64.compressor_page_count offset mismatch");
    if (@offsetOf(VmStatistics64, "external_page_count") != 136) @compileError("VmStatistics64.external_page_count offset mismatch");
    if (@offsetOf(VmStatistics64, "internal_page_count") != 140) @compileError("VmStatistics64.internal_page_count offset mismatch");
    if (@sizeOf(XswUsage) != 32) @compileError("XswUsage must be 32 bytes");
    if (@sizeOf(MallocStatistics) != 32) @compileError("MallocStatistics must be 32 bytes");
}

extern "c" var vm_kernel_page_size: usize;

extern "c" fn host_statistics64(
    host_priv: std.c.mach_port_t,
    flavor: c_int,
    host_info64_out: *anyopaque,
    host_info64_outCnt: *u32,
) c_int;

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;

pub const Error = error{
    HostStatsFailed,
    ShortHostStats,
    SwapUsageFailed,
    UnexpectedSize,
    ClockFailed,
};

pub const SwapError = error{
    /// System swapout counter increased during the measured window.
    SwapOccurred,
};

pub const HeapStats = struct {
    blocks_in_use: u64,
    size_in_use: u64,
    max_size_in_use: u64,
    size_allocated: u64,
};

pub const HeapDelta = struct {
    blocks_in_use: i64,
    size_in_use: i64,
    max_size_in_use: i64,
    size_allocated: i64,
};

pub const Snapshot = struct {
    swapouts: u64,
    pageouts: u64,
    swap_total_bytes: u64,
    swap_used_bytes: u64,
    swap_avail_bytes: u64,
    heap: HeapStats,
    /// Kernel VM page size (`vm_kernel_page_size`, `16 KiB` on Apple Silicon).
    page_size: u64 = page.APPLE_PAGE_SIZE,
    /// Live Mach `HOST_VM_INFO64` page breakdown (ADR-0020 P1-B).
    /// Note: XNU's `vm_statistics64.free_count` includes `speculative_count`, so
    /// `capture()` normalizes `free_pages = vm.free_count -| vm.speculative_count`
    /// (matching macOS `vm_stat`) so `free_pages + speculative_pages` never
    /// double-counts speculative pages.
    free_pages: u64 = 0,
    speculative_pages: u64 = 0,
    purgeable_pages: u64 = 0,
    external_pages: u64 = 0,
    inactive_pages: u64 = 0,
    internal_pages: u64 = 0,
    compressor_pages: u64 = 0,
    wire_pages: u64 = 0,
    active_pages: u64 = 0,

    pub fn capture() Error!Snapshot {
        const host = std.c.mach_host_self();
        defer _ = std.c.mach_port_deallocate(std.c.mach_task_self(), host);

        var vm: VmStatistics64 = std.mem.zeroes(VmStatistics64);
        var count: u32 = HOST_VM_INFO64_REV1_COUNT;
        if (host_statistics64(host, HOST_VM_INFO64, &vm, &count) != 0) {
            return error.HostStatsFailed;
        }
        if (count < HOST_VM_INFO64_REV1_COUNT) {
            return error.ShortHostStats;
        }

        var swap: XswUsage = std.mem.zeroes(XswUsage);
        var swap_size: usize = @sizeOf(XswUsage);
        if (std.c.sysctlbyname("vm.swapusage", &swap, &swap_size, null, 0) != 0) {
            return error.SwapUsageFailed;
        }
        if (swap_size != @sizeOf(XswUsage)) {
            return error.UnexpectedSize;
        }

        var mstats: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &mstats);

        const kernel_page_size: u64 = if (vm_kernel_page_size > 0)
            @intCast(vm_kernel_page_size)
        else
            page.APPLE_PAGE_SIZE;

        return .{
            .swapouts = vm.swapouts,
            .pageouts = vm.pageouts,
            .swap_total_bytes = swap.xsu_total,
            .swap_used_bytes = swap.xsu_used,
            .swap_avail_bytes = swap.xsu_avail,
            .heap = .{
                .blocks_in_use = mstats.blocks_in_use,
                .size_in_use = mstats.size_in_use,
                .max_size_in_use = mstats.max_size_in_use,
                .size_allocated = mstats.size_allocated,
            },
            .page_size = kernel_page_size,
            .free_pages = vm.free_count -| vm.speculative_count,
            .speculative_pages = vm.speculative_count,
            .purgeable_pages = vm.purgeable_count,
            .external_pages = vm.external_page_count,
            .inactive_pages = vm.inactive_count,
            .internal_pages = vm.internal_page_count,
            .compressor_pages = vm.compressor_page_count,
            .wire_pages = vm.wire_count,
            .active_pages = vm.active_count,
        };
    }

    /// Returns live RAM bytes that XNU can reclaim without compressing or
    /// swapping anonymous memory (ADR-0020 P1-B):
    /// strictly `(free_pages + speculative_pages + purgeable_pages + external_pages) * page_size`.
    /// Explicitly excludes anonymous `inactive_pages` and `internal_pages`.
    pub fn liveZeroSwapReclaimableBytes(self: Snapshot) u64 {
        const ps: u64 = if (self.page_size > 0) self.page_size else page.APPLE_PAGE_SIZE;
        const reclaimable_pages: u64 = self.free_pages +|
            self.speculative_pages +|
            self.purgeable_pages +|
            self.external_pages;
        return reclaimable_pages *| ps;
    }

    pub fn heapDelta(before: Snapshot, after: Snapshot) HeapDelta {
        return .{
            .blocks_in_use = signedDiff(after.heap.blocks_in_use, before.heap.blocks_in_use),
            .size_in_use = signedDiff(after.heap.size_in_use, before.heap.size_in_use),
            .max_size_in_use = signedDiff(after.heap.max_size_in_use, before.heap.max_size_in_use),
            .size_allocated = signedDiff(after.heap.size_allocated, before.heap.size_allocated),
        };
    }

    pub fn swapoutDelta(before: Snapshot, after: Snapshot) u64 {
        return after.swapouts -| before.swapouts;
    }

    pub fn pageoutDelta(before: Snapshot, after: Snapshot) u64 {
        return after.pageouts -| before.pageouts;
    }
};

fn signedDiff(after: u64, before: u64) i64 {
    if (after >= before) {
        const diff = after - before;
        return @intCast(@min(diff, @as(u64, std.math.maxInt(i64))));
    } else {
        const diff = before - after;
        if (diff > @as(u64, std.math.maxInt(i64))) return std.math.minInt(i64);
        return -@as(i64, @intCast(diff));
    }
}

/// Returns `error.SwapOccurred` if the system swapout counter grew between
/// `before` and `after`.
pub fn checkNoSwap(before: Snapshot, after: Snapshot) SwapError!void {
    if (after.swapouts > before.swapouts) return error.SwapOccurred;
}

/// Computes signed malloc-zone deltas from `before` to `after`.
pub fn heapDelta(before: Snapshot, after: Snapshot) HeapDelta {
    return Snapshot.heapDelta(before, after);
}

/// Returns monotonic uptime in nanoseconds (`CLOCK_UPTIME_RAW`).
pub fn nowNs() Error!u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.UPTIME_RAW, &ts) != 0) {
        return error.ClockFailed;
    }
    if (ts.sec < 0 or ts.nsec < 0) return error.ClockFailed;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "nowNs returns a positive monotonic nanosecond timestamp" {
    const t0 = try nowNs();
    const t1 = try nowNs();
    try std.testing.expect(t0 > 0);
    try std.testing.expect(t1 >= t0);
}

test "Snapshot.capture reads live Mach VM stats, swap usage and malloc zone statistics" {
    const snap = try Snapshot.capture();
    try std.testing.expect(snap.heap.blocks_in_use > 0);
    try std.testing.expect(snap.heap.size_in_use > 0);
    // Note: macOS xzone malloc reports max_size_in_use as 0, while
    // blocks_in_use, size_in_use and size_allocated are live counters.
    try std.testing.expect(snap.heap.size_allocated >= snap.heap.size_in_use);
    try std.testing.expect(snap.swap_used_bytes <= snap.swap_total_bytes);
}

test "checkNoSwap accepts stable swapouts and rejects any increase" {
    const base = Snapshot{
        .swapouts = 100,
        .pageouts = 50,
        .swap_total_bytes = 4 * 1024 * 1024 * 1024,
        .swap_used_bytes = 0,
        .swap_avail_bytes = 4 * 1024 * 1024 * 1024,
        .heap = .{
            .blocks_in_use = 10,
            .size_in_use = 1024,
            .max_size_in_use = 2048,
            .size_allocated = 4096,
        },
    };
    try checkNoSwap(base, base);

    var increased = base;
    increased.swapouts = 101;
    try std.testing.expectError(error.SwapOccurred, checkNoSwap(base, increased));
    try std.testing.expectEqual(@as(u64, 1), Snapshot.swapoutDelta(base, increased));

    var paged = base;
    paged.pageouts = 53;
    try std.testing.expectEqual(@as(u64, 3), Snapshot.pageoutDelta(base, paged));
}

test "heapDelta computes positive and negative signed deltas and observes live malloc" {
    const a = Snapshot{
        .swapouts = 0,
        .pageouts = 0,
        .swap_total_bytes = 0,
        .swap_used_bytes = 0,
        .swap_avail_bytes = 0,
        .heap = .{
            .blocks_in_use = 20,
            .size_in_use = 4096,
            .max_size_in_use = 8192,
            .size_allocated = 16384,
        },
    };
    const b = Snapshot{
        .swapouts = 0,
        .pageouts = 0,
        .swap_total_bytes = 0,
        .swap_used_bytes = 0,
        .swap_avail_bytes = 0,
        .heap = .{
            .blocks_in_use = 15,
            .size_in_use = 6144,
            .max_size_in_use = 8192,
            .size_allocated = 12288,
        },
    };
    const d = heapDelta(a, b);
    try std.testing.expectEqual(@as(i64, -5), d.blocks_in_use);
    try std.testing.expectEqual(@as(i64, 2048), d.size_in_use);
    try std.testing.expectEqual(@as(i64, 0), d.max_size_in_use);
    try std.testing.expectEqual(@as(i64, -4096), d.size_allocated);

    // Live malloc_zone_statistics must observe a large C heap allocation and
    // its free (sized well above background Metal/GCD thread churn; volatile
    // indirection prevents LLVM malloc/free elision in optimized builds).
    const alloc_size: usize = 4 * 1024 * 1024;
    const min_delta: i64 = 2 * 1024 * 1024;
    const before = try Snapshot.capture();
    var anchor: ?*anyopaque = std.c.malloc(alloc_size);
    const ptr = @as(*volatile ?*anyopaque, &anchor).* orelse return error.OutOfMemory;
    const raw: [*]volatile u8 = @ptrCast(ptr);
    raw[0] = 0x5A;
    raw[alloc_size - 1] = 0xA5;
    const during = try Snapshot.capture();
    std.c.free(@as(*volatile ?*anyopaque, &anchor).* orelse unreachable);
    const after = try Snapshot.capture();

    const growth = heapDelta(before, during);
    try std.testing.expect(growth.size_in_use >= min_delta);

    const shrink = heapDelta(during, after);
    try std.testing.expect(shrink.size_in_use <= -min_delta);
}

test "liveZeroSwapReclaimableBytes sums free+speculative+purgeable+external and excludes inactive/internal" {
    const base = Snapshot{
        .swapouts = 0,
        .pageouts = 0,
        .swap_total_bytes = 0,
        .swap_used_bytes = 0,
        .swap_avail_bytes = 0,
        .heap = .{
            .blocks_in_use = 0,
            .size_in_use = 0,
            .max_size_in_use = 0,
            .size_allocated = 0,
        },
        .page_size = page.APPLE_PAGE_SIZE,
        .free_pages = 100,
        .speculative_pages = 25,
        .purgeable_pages = 40,
        .external_pages = 35,
        .inactive_pages = 50_000,
        .internal_pages = 80_000,
        .compressor_pages = 12_000,
        .wire_pages = 90_000,
        .active_pages = 70_000,
    };

    const expected: u64 = (100 + 25 + 40 + 35) * @as(u64, page.APPLE_PAGE_SIZE);
    try std.testing.expectEqual(expected, base.liveZeroSwapReclaimableBytes());

    // Changing anonymous / non-reclaimable page buckets must not change the zero-swap pool.
    var mutated = base;
    mutated.inactive_pages = 999_999;
    mutated.internal_pages = 999_999;
    mutated.compressor_pages = 999_999;
    mutated.wire_pages = 999_999;
    mutated.active_pages = 999_999;
    try std.testing.expectEqual(expected, mutated.liveZeroSwapReclaimableBytes());

    // Live capture must report 16 KiB kernel page size and a consistent live page breakdown.
    const live = try Snapshot.capture();
    try std.testing.expectEqual(@as(u64, page.APPLE_PAGE_SIZE), live.page_size);
    try std.testing.expect(live.wire_pages > 0);
    try std.testing.expect(live.active_pages > 0);
    try std.testing.expectEqual(
        (live.free_pages + live.speculative_pages + live.purgeable_pages + live.external_pages) * live.page_size,
        live.liveZeroSwapReclaimableBytes(),
    );
}

test {
    _ = @import("telemetry.zig");
    _ = @import("../monitor/theme.zig");
    _ = @import("../monitor/tui.zig");
    _ = @import("../cli/chat_repl.zig");
}
