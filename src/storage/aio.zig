//! Direct NVMe cold-expert asynchronous I/O engines (ADR-0006, status rows C3, C4, Spike S3).
//!
//! All cold-expert reads bypass the macOS unified buffer cache (`F_NOCACHE`) and
//! land directly in 16 KiB page-aligned wired slots with zero dynamic heap
//! allocation on the steady-state submit/poll/wait hot path:
//! - `KqueueAioEngine` (C3): Darwin POSIX `aio_read` with `SIGEV_KEVENT` ->
//!   `kevent64(EVFILT_AIO)` auto-reap (`ext[0]` errno, `ext[1]` bytes; never
//!   calls `aio_return` afterwards).
//! - `EcorePreadPool` (C4): 1..8 E-core QoS worker threads (`utility` +
//!   `IOPOL_IMPORTANT` or `IOPOL_STANDARD`) consuming from a lock-free bounded
//!   ring (parking on a condition variable when idle for 0% idle CPU) and
//!   publishing completions through a lock-free `MpscRing`.
//! - `AsyncExpertReader`: unified reader selecting `kqueue_aio` or `ecore_pread`
//!   (with `.auto` calibrated by Spike S3).
//! - `runSpikeS3`: reproducible NVMe cold-expert benchmark sweeping backends,
//!   worker counts, disk I/O policies, and queue depths 1..16 with zero-swap
//!   enforcement.

const std = @import("std");
const page = @import("../mem/page.zig");
const qos = @import("../platform/qos.zig");
const ring = @import("../runtime/ring.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Maximum concurrent in-flight reads supported by a single engine instance.
pub const MAX_INFLIGHT: usize = 64;
/// Maximum E-core `pread` worker threads in `EcorePreadPool`.
pub const MAX_WORKERS: usize = 8;

/// Real Qwen3-30B-A3B 3-bit expert triple size (`3 * 688,128 B = 126 * 16 KiB`).
pub const EXPERT_BLOB_Q3_BYTES: usize = 2_064_384;
/// Real Qwen3-30B-A3B 4-bit expert triple size (`3 * 884,736 B = 162 * 16 KiB`).
pub const EXPERT_BLOB_Q4_BYTES: usize = 2_654_208;

comptime {
    if (EXPERT_BLOB_Q3_BYTES % page.APPLE_PAGE_SIZE != 0) @compileError("EXPERT_BLOB_Q3_BYTES must be 16 KiB aligned");
    if (EXPERT_BLOB_Q4_BYTES % page.APPLE_PAGE_SIZE != 0) @compileError("EXPERT_BLOB_Q4_BYTES must be 16 KiB aligned");
}

// ---------------------------------------------------------------------------
// Darwin POSIX AIO & kqueue ABI (`<sys/aio.h>`, `<sys/signal.h>`, `<sys/event.h>`)
// ---------------------------------------------------------------------------

pub const SIGEV_NONE: c_int = 0;
pub const SIGEV_SIGNAL: c_int = 1;
pub const SIGEV_THREAD: c_int = 3;
/// Darwin extension: deliver AIO completion directly to a `kqueue` fd stored in `sigev_signo`.
pub const SIGEV_KEVENT: c_int = 4;

pub const Sigval = extern union {
    sival_int: c_int,
    sival_ptr: ?*anyopaque,
    sival_u64: u64,
};

pub const Sigevent = extern struct {
    sigev_notify: c_int = SIGEV_NONE,
    /// On Darwin, carries the `kqueue` fd when `sigev_notify == SIGEV_KEVENT`.
    sigev_signo: c_int = 0,
    sigev_value: Sigval = .{ .sival_u64 = 0 },
    sigev_notify_function: ?*const anyopaque = null,
    sigev_notify_attributes: ?*anyopaque = null,
};

pub const Aiocb = extern struct {
    aio_fildes: c_int = -1,
    aio_offset: std.c.off_t = 0,
    aio_buf: ?*volatile anyopaque = null,
    aio_nbytes: usize = 0,
    aio_reqprio: c_int = 0,
    aio_sigevent: Sigevent = .{},
    aio_lio_opcode: c_int = 0,
};

comptime {
    if (@sizeOf(Sigval) != 8) @compileError("Darwin union sigval must be 8 bytes");
    if (@sizeOf(Sigevent) != 32) @compileError("Darwin struct sigevent must be 32 bytes");
    if (@offsetOf(Sigevent, "sigev_notify") != 0) @compileError("sigev_notify offset mismatch");
    if (@offsetOf(Sigevent, "sigev_signo") != 4) @compileError("sigev_signo offset mismatch");
    if (@offsetOf(Sigevent, "sigev_value") != 8) @compileError("sigev_value offset mismatch");
    if (@offsetOf(Sigevent, "sigev_notify_function") != 16) @compileError("sigev_notify_function offset mismatch");
    if (@offsetOf(Sigevent, "sigev_notify_attributes") != 24) @compileError("sigev_notify_attributes offset mismatch");

    if (@sizeOf(Aiocb) != 80) @compileError("Darwin struct aiocb must be 80 bytes");
    if (@offsetOf(Aiocb, "aio_fildes") != 0) @compileError("aio_fildes offset mismatch");
    if (@offsetOf(Aiocb, "aio_offset") != 8) @compileError("aio_offset offset mismatch");
    if (@offsetOf(Aiocb, "aio_buf") != 16) @compileError("aio_buf offset mismatch");
    if (@offsetOf(Aiocb, "aio_nbytes") != 24) @compileError("aio_nbytes offset mismatch");
    if (@offsetOf(Aiocb, "aio_reqprio") != 32) @compileError("aio_reqprio offset mismatch");
    if (@offsetOf(Aiocb, "aio_sigevent") != 40) @compileError("aio_sigevent offset mismatch");
    if (@offsetOf(Aiocb, "aio_lio_opcode") != 72) @compileError("aio_lio_opcode offset mismatch");
}

extern "c" fn aio_read(aiocbp: *Aiocb) c_int;
extern "c" fn pthread_cond_timedwait_relative_np(
    cond: *std.c.pthread_cond_t,
    mutex: *std.c.pthread_mutex_t,
    reltime: *const std.c.timespec,
) std.c.E;

/// Queries kernel POSIX AIO concurrency limits (`kern.aiothreads`, `kern.aioprocmax`).
pub const KernelAioLimits = struct {
    aio_threads: u64,
    aio_proc_max: u64,

    pub fn query() sysinfo.Error!KernelAioLimits {
        return .{
            .aio_threads = try sysinfo.readUnsigned("kern.aiothreads"),
            .aio_proc_max = try sysinfo.readUnsigned("kern.aioprocmax"),
        };
    }
};

// ---------------------------------------------------------------------------
// Public errors, file handle, and I/O request/completion types
// ---------------------------------------------------------------------------

pub const IoBackend = enum {
    auto,
    kqueue_aio,
    ecore_pread,

    pub fn label(self: IoBackend) []const u8 {
        return switch (self) {
            .auto => "auto",
            .kqueue_aio => "kqueue_aio",
            .ecore_pread => "ecore_pread",
        };
    }
};

pub const AlignError = error{
    InvalidArgument,
    ZeroLength,
    MisalignedOffset,
    MisalignedAddress,
    MisalignedLength,
};

pub const IoResultError = error{
    ReadFailed,
    ShortRead,
    Canceled,
};

pub const EngineError = AlignError || IoResultError || error{
    OpenFailed,
    FcntlFailed,
    StatFailed,
    KqueueFailed,
    QueueFull,
    SubmitFailed,
    PollFailed,
    Timeout,
    OutOfMemory,
};

/// Sets `F_NOCACHE = 1` on an open file descriptor so reads and writes bypass
/// the macOS unified buffer cache (ADR-0006).
pub fn enableNoCache(fd: std.c.fd_t) EngineError!void {
    if (fd < 0) return error.InvalidArgument;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) {
        return error.FcntlFailed;
    }
}

/// Read-only `F_NOCACHE` file descriptor handle.
pub const DirectFile = struct {
    fd: std.c.fd_t,
    size: u64,

    pub fn openReadOnly(path: [:0]const u8) EngineError!DirectFile {
        if (path.len == 0) return error.InvalidArgument;
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.OpenFailed;
        errdefer _ = std.c.close(fd);

        try enableNoCache(fd);

        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
        if (st.size < 0) return error.StatFailed;

        return .{
            .fd = fd,
            .size = @intCast(st.size),
        };
    }

    pub fn close(self: *DirectFile) void {
        if (self.fd >= 0) {
            _ = std.c.close(self.fd);
            self.fd = -1;
        }
    }
};

/// Single 16 KiB-aligned direct NVMe read request.
pub const ReadRequest = struct {
    /// Open file descriptor (caller/DirectFile/AmfFile enables `F_NOCACHE`).
    fd: std.c.fd_t,
    /// File byte offset; must be a multiple of 16 KiB.
    offset: u64,
    /// Destination slice in wired memory; start address and length must be
    /// non-zero multiples of 16 KiB.
    dst: []align(page.APPLE_PAGE_SIZE) u8,
    /// Caller-defined opaque identifier returned verbatim in `Completion.user_data`.
    user_data: u64 = 0,
};

/// Validates 16 KiB page alignment on file offset, buffer pointer, and length.
pub fn validateReadRequest(req: ReadRequest) AlignError!void {
    if (req.fd < 0) return error.InvalidArgument;
    if (req.dst.len == 0) return error.ZeroLength;
    if (req.offset % page.APPLE_PAGE_SIZE != 0) return error.MisalignedOffset;
    if (!std.mem.isAligned(@intFromPtr(req.dst.ptr), page.APPLE_PAGE_SIZE)) return error.MisalignedAddress;
    if (req.dst.len % page.APPLE_PAGE_SIZE != 0) return error.MisalignedLength;
    const max_off: u64 = @intCast(std.math.maxInt(std.c.off_t));
    if (req.offset > max_off or req.dst.len > max_off - req.offset) return error.InvalidArgument;
}

/// Completed direct read result.
pub const Completion = struct {
    user_data: u64,
    bytes_read: usize,
    /// Zero on success, or the POSIX errno reported by `EVFILT_AIO` (`ext[0]`) / `pread`.
    err_code: i32,
    /// True when `err_code == 0` but `bytes_read != requested_len` (e.g. EOF).
    short_read: bool,
    /// Monotonic `CLOCK_UPTIME_RAW` timestamp captured at submit (`0` if unavailable).
    submit_ns: u64 = 0,
    /// Monotonic `CLOCK_UPTIME_RAW` timestamp captured at completion (`0` if unavailable).
    complete_ns: u64 = 0,

    pub fn isOk(self: Completion) bool {
        return self.err_code == 0 and !self.short_read;
    }

    pub fn latencyNs(self: Completion) u64 {
        return self.complete_ns -| self.submit_ns;
    }

    pub fn check(self: Completion) IoResultError!usize {
        if (self.err_code != 0) {
            if (self.err_code == @backingInt(std.c.E.CANCELED)) return error.Canceled;
            return error.ReadFailed;
        }
        if (self.short_read) return error.ShortRead;
        return self.bytes_read;
    }
};

fn monotonicNowNs() u64 {
    return vmstat.nowNs() catch 0;
}

fn nsToTimespec(timeout_ns: u64) std.c.timespec {
    const secs = timeout_ns / std.time.ns_per_s;
    const rem_ns = timeout_ns % std.time.ns_per_s;
    const max_sec: u64 = @intCast(std.math.maxInt(isize));
    return .{
        .sec = @intCast(@min(secs, max_sec)),
        .nsec = @intCast(rem_ns),
    };
}

// ---------------------------------------------------------------------------
// C3: KqueueAioEngine (`aio_read` + `SIGEV_KEVENT` + `kevent64(EVFILT_AIO)`)
// ---------------------------------------------------------------------------

/// Direct-I/O asynchronous reader using macOS POSIX `aio_read` delivering
/// completions into a dedicated `kqueue` via `SIGEV_KEVENT` (`EVFILT_AIO`).
///
/// All `aiocb` slots and `kevent64_s` scratch storage are preallocated once in
/// `init` so `&slot.cb` stays at a fixed address and steady-state `submit` /
/// `poll` / `wait` perform zero dynamic heap allocation.
pub const KqueueAioEngine = struct {
    const Slot = struct {
        cb: Aiocb,
        user_data: u64,
        expected_len: usize,
        submit_ns: u64,
        generation: u32,
        in_use: bool,
    };

    kq_fd: std.c.fd_t,
    slots: []Slot,
    free_stack: []u16,
    free_top: usize,
    events: []std.c.kevent64_s,
    inflight: usize,

    pub fn init(allocator: std.mem.Allocator, max_inflight: usize) EngineError!KqueueAioEngine {
        if (max_inflight == 0 or max_inflight > MAX_INFLIGHT) return error.InvalidArgument;

        const kq_fd = std.c.kqueue();
        if (kq_fd < 0) return error.KqueueFailed;
        errdefer _ = std.c.close(kq_fd);

        const slots = try allocator.alloc(Slot, max_inflight);
        errdefer allocator.free(slots);

        const free_stack = try allocator.alloc(u16, max_inflight);
        errdefer allocator.free(free_stack);

        const events = try allocator.alloc(std.c.kevent64_s, max_inflight);
        errdefer allocator.free(events);

        for (slots, 0..) |*slot, i| {
            slot.* = .{
                .cb = .{},
                .user_data = 0,
                .expected_len = 0,
                .submit_ns = 0,
                .generation = 1,
                .in_use = false,
            };
            // Push in reverse so slot 0 pops first.
            free_stack[max_inflight - 1 - i] = @intCast(i);
        }

        return .{
            .kq_fd = kq_fd,
            .slots = slots,
            .free_stack = free_stack,
            .free_top = max_inflight,
            .events = events,
            .inflight = 0,
        };
    }

    /// Drains all in-flight kernel AIO requests so no kernel reference to
    /// `&slot.cb` or destination wired buffers outlives the engine, then frees
    /// startup storage and closes the `kqueue` fd.
    pub fn deinit(self: *KqueueAioEngine, allocator: std.mem.Allocator) void {
        if (self.kq_fd >= 0) {
            var scratch: [MAX_INFLIGHT]Completion = undefined;
            while (self.inflight > 0) {
                const count = self.wait(&scratch, 5_000) catch break;
                if (count == 0) break;
            }
            _ = std.c.close(self.kq_fd);
            self.kq_fd = -1;
        }
        allocator.free(self.events);
        allocator.free(self.free_stack);
        allocator.free(self.slots);
        self.events = &.{};
        self.free_stack = &.{};
        self.slots = &.{};
        self.free_top = 0;
        self.inflight = 0;
    }

    pub fn maxInflight(self: *const KqueueAioEngine) usize {
        return self.slots.len;
    }

    pub fn inflightCount(self: *const KqueueAioEngine) usize {
        return self.inflight;
    }

    pub fn availableSlots(self: *const KqueueAioEngine) usize {
        return self.free_top;
    }

    /// Submits one 16 KiB-aligned asynchronous read with zero heap allocation.
    pub fn submit(self: *KqueueAioEngine, req: ReadRequest) EngineError!void {
        try validateReadRequest(req);
        if (self.free_top == 0) return error.QueueFull;

        const slot_idx: usize = self.free_stack[self.free_top - 1];
        const slot = &self.slots[slot_idx];
        std.debug.assert(!slot.in_use);

        const tag: u64 = (@as(u64, slot.generation) << 32) | @as(u64, slot_idx);
        slot.cb = .{
            .aio_fildes = req.fd,
            .aio_offset = @intCast(req.offset),
            .aio_buf = @ptrCast(@volatileCast(req.dst.ptr)),
            .aio_nbytes = req.dst.len,
            .aio_reqprio = 0,
            .aio_sigevent = .{
                .sigev_notify = SIGEV_KEVENT,
                .sigev_signo = self.kq_fd,
                .sigev_value = .{ .sival_u64 = tag },
                .sigev_notify_function = null,
                .sigev_notify_attributes = null,
            },
            .aio_lio_opcode = 0,
        };
        slot.user_data = req.user_data;
        slot.expected_len = req.dst.len;
        slot.submit_ns = monotonicNowNs();
        slot.in_use = true;

        // Commit slot ownership before calling `aio_read` so `slot` is valid if
        // the kernel completes immediately.
        self.free_top -= 1;
        self.inflight += 1;

        const rc = aio_read(&slot.cb);
        if (rc != 0) {
            const err = std.c.errno(rc);
            slot.in_use = false;
            self.inflight -= 1;
            self.free_stack[self.free_top] = @intCast(slot_idx);
            self.free_top += 1;
            return switch (err) {
                .AGAIN => error.QueueFull,
                .BADF, .INVAL => error.InvalidArgument,
                else => error.SubmitFailed,
            };
        }
    }

    /// Non-blocking completion reap (`timeout = 0`). Returns `0..out.len`.
    pub fn poll(self: *KqueueAioEngine, out: []Completion) EngineError!usize {
        if (out.len == 0 or self.inflight == 0) return 0;
        return self.reap(out, 0);
    }

    /// Waits for at least one completion (or `timeout_ms` elapsed; `null` waits
    /// indefinitely). Returns `0` on timeout or if `inflight == 0`.
    pub fn wait(self: *KqueueAioEngine, out: []Completion, timeout_ms: ?u64) EngineError!usize {
        if (out.len == 0 or self.inflight == 0) return 0;
        return self.reap(out, timeout_ms);
    }

    /// Submits one read and waits until it completes, returning bytes read.
    pub fn readSync(self: *KqueueAioEngine, req: ReadRequest, timeout_ms: u64) EngineError!usize {
        try self.submit(req);
        const start_ns = monotonicNowNs();
        const deadline_ns = start_ns +| (timeout_ms *| std.time.ns_per_ms);
        var comp: [1]Completion = undefined;
        while (true) {
            const now = monotonicNowNs();
            if (now >= deadline_ns) return error.Timeout;
            const rem_ms = @max(@as(u64, 1), (deadline_ns - now) / std.time.ns_per_ms);
            const got = try self.wait(&comp, rem_ms);
            if (got > 0) {
                return try comp[0].check();
            }
        }
    }

    fn reap(self: *KqueueAioEngine, out: []Completion, timeout_ms: ?u64) EngineError!usize {
        const cap = @min(out.len, @min(self.events.len, self.inflight));
        if (cap == 0) return 0;

        var empty_changes: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
        const deadline_ns: ?u64 = if (timeout_ms) |ms| blk: {
            if (ms == 0) break :blk 0;
            break :blk monotonicNowNs() +| (ms *| std.time.ns_per_ms);
        } else null;

        while (true) {
            var ts: std.c.timespec = undefined;
            const ts_ptr: ?*const std.c.timespec = if (timeout_ms) |ms| blk: {
                if (ms == 0) {
                    ts = .{ .sec = 0, .nsec = 0 };
                    break :blk &ts;
                }
                const now = monotonicNowNs();
                const dl = deadline_ns.?;
                if (now >= dl) return 0;
                ts = nsToTimespec(dl - now);
                break :blk &ts;
            } else null;

            const nev = std.c.kevent64(
                self.kq_fd,
                &empty_changes,
                0,
                self.events.ptr,
                @intCast(cap),
                std.c.KEVENT.FLAG.NONE,
                ts_ptr,
            );
            if (nev < 0) {
                if (std.c.errno(nev) == .INTR) {
                    if (timeout_ms != null and timeout_ms.? == 0) return 0;
                    continue;
                }
                return error.PollFailed;
            }
            if (nev == 0) return 0;

            const complete_ns = monotonicNowNs();
            const n_events: usize = @intCast(nev);
            var written: usize = 0;
            for (self.events[0..n_events]) |ev| {
                if (ev.filter != std.c.EVFILT.AIO) continue;

                const slot_idx: usize = @intCast(ev.udata & 0xFFFF_FFFF);
                const gen: u32 = @intCast(ev.udata >> 32);
                if (slot_idx >= self.slots.len) return error.PollFailed;

                const slot = &self.slots[slot_idx];
                if (!slot.in_use or slot.generation != gen or ev.ident != @intFromPtr(&slot.cb)) {
                    return error.PollFailed;
                }

                // Per ADR-0006 (measured on macOS 27.2): `SIGEV_KEVENT` auto-reaps
                // the AIO control block; `ext[0]` holds errno and `ext[1]` holds
                // transferred bytes. Never call `aio_return` afterwards.
                const err_code: i32 = @intCast(@min(ev.ext[0], @as(u64, std.math.maxInt(i32))));
                const bytes_read: usize = @intCast(ev.ext[1]);
                const short_read = (err_code == 0 and bytes_read != slot.expected_len);

                out[written] = .{
                    .user_data = slot.user_data,
                    .bytes_read = bytes_read,
                    .err_code = err_code,
                    .short_read = short_read,
                    .submit_ns = slot.submit_ns,
                    .complete_ns = complete_ns,
                };
                written += 1;

                slot.in_use = false;
                slot.generation +%= 1;
                if (slot.generation == 0) slot.generation = 1;
                self.inflight -= 1;
                self.free_stack[self.free_top] = @intCast(slot_idx);
                self.free_top += 1;
            }
            return written;
        }
    }
};

// ---------------------------------------------------------------------------
// Lock-free bounded MPMC work ring (128-byte cache-line padded, zero alloc)
// ---------------------------------------------------------------------------

fn WorkRing(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Slot = struct {
            sequence: std.atomic.Value(usize),
            value: T,
        };

        slots: []Slot,
        mask: usize,

        tail: std.atomic.Value(usize) align(ring.CACHE_LINE_BYTES),
        head: std.atomic.Value(usize) align(ring.CACHE_LINE_BYTES),

        pub fn init(slots: []Slot) ring.InitError!Self {
            if (slots.len == 0 or !std.math.isPowerOfTwo(slots.len)) {
                return error.InvalidCapacity;
            }
            for (slots, 0..) |*slot, i| {
                slot.sequence = .init(i << 1);
                slot.value = undefined;
            }
            return .{
                .slots = slots,
                .mask = slots.len - 1,
                .tail = .init(0),
                .head = .init(0),
            };
        }

        pub fn tryPush(self: *Self, item: T) bool {
            var pos = self.tail.load(.monotonic);
            while (true) {
                const slot = &self.slots[pos & self.mask];
                const seq = slot.sequence.load(.acquire);
                const expected = pos << 1;
                const diff: isize = @bitCast(seq -% expected);
                if (diff == 0) {
                    if (self.tail.cmpxchgWeak(pos, pos +% 1, .monotonic, .monotonic)) |actual| {
                        pos = actual;
                    } else {
                        slot.value = item;
                        slot.sequence.store(expected +% 1, .release);
                        return true;
                    }
                } else if (diff < 0) {
                    return false;
                } else {
                    pos = self.tail.load(.monotonic);
                }
            }
        }

        pub fn tryPop(self: *Self) ?T {
            var pos = self.head.load(.monotonic);
            while (true) {
                const slot = &self.slots[pos & self.mask];
                const seq = slot.sequence.load(.acquire);
                const expected = (pos << 1) +% 1;
                const diff: isize = @bitCast(seq -% expected);
                if (diff == 0) {
                    if (self.head.cmpxchgWeak(pos, pos +% 1, .monotonic, .monotonic)) |actual| {
                        pos = actual;
                    } else {
                        const item = slot.value;
                        slot.sequence.store((pos +% self.slots.len) << 1, .release);
                        return item;
                    }
                } else if (diff < 0) {
                    return null;
                } else {
                    pos = self.head.load(.monotonic);
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// C4: EcorePreadPool (1..8 E-core QoS workers issuing F_NOCACHE pread)
// ---------------------------------------------------------------------------

pub const EcorePoolConfig = struct {
    max_inflight: usize = 32,
    worker_count: usize = 4,
    /// Efficiency-core QoS class (`.utility` or `.background`).
    qos_class: qos.QosClass = .utility,
    /// Thread disk I/O policy override (`.important` or `.standard`).
    io_policy: qos.DiskIoPolicy = .important,
};

pub const EcorePoolError = EngineError || qos.Error || ring.InitError || std.Thread.SpawnError;

/// Pool of 1..8 E-core worker threads executing `F_NOCACHE` `pread` directly
/// into 16 KiB-aligned wired slots.
///
/// Work requests flow through a preallocated lock-free ring (`WorkRing`),
/// completions return through `ring.MpscRing(Completion)`, and idle workers
/// park on a POSIX condition variable (burning 0% CPU when idle). All memory is
/// allocated once in `init` inside a heap-stable `State` block; steady-state
/// `submit` / `poll` / `wait` perform zero dynamic heap allocation.
pub const EcorePreadPool = struct {
    const WorkItem = struct {
        req: ReadRequest,
        submit_ns: u64,
    };

    const ReqRing = WorkRing(WorkItem);
    const CompRing = ring.MpscRing(Completion);

    const State = struct {
        req_ring: ReqRing,
        comp_ring: CompRing,
        req_slots: []ReqRing.Slot,
        comp_slots: []CompRing.Slot,

        pending_reqs: std.atomic.Value(usize) align(ring.CACHE_LINE_BYTES),
        sleeping_workers: std.atomic.Value(u32),
        shutdown: std.atomic.Value(bool),
        work_mutex: std.c.pthread_mutex_t,
        work_cond: std.c.pthread_cond_t,

        pending_comps: std.atomic.Value(usize) align(ring.CACHE_LINE_BYTES),
        comp_waiters: std.atomic.Value(u32),
        comp_mutex: std.c.pthread_mutex_t,
        comp_cond: std.c.pthread_cond_t,

        // Startup handshake so `init` confirms every worker applied QoS + I/O policy.
        ready_count: std.atomic.Value(u32),
        init_mutex: std.c.pthread_mutex_t,
        init_cond: std.c.pthread_cond_t,
        worker_errors: [MAX_WORKERS]?qos.Error,
        qos_class: qos.QosClass,
        io_policy: qos.DiskIoPolicy,
        worker_count: usize,
    };

    state: *State,
    threads: [MAX_WORKERS]std.Thread,
    spawned: usize,
    max_inflight: usize,
    inflight: usize,

    pub fn init(allocator: std.mem.Allocator, config: EcorePoolConfig) EcorePoolError!EcorePreadPool {
        if (config.max_inflight == 0 or config.max_inflight > MAX_INFLIGHT) return error.InvalidArgument;
        if (config.worker_count == 0 or config.worker_count > MAX_WORKERS) return error.InvalidArgument;
        switch (config.qos_class) {
            .utility, .background => {},
            else => return error.InvalidArgument,
        }

        const ring_cap = std.math.ceilPowerOfTwo(usize, config.max_inflight) catch return error.InvalidArgument;

        const state = try allocator.create(State);
        errdefer allocator.destroy(state);

        const req_slots = try allocator.alloc(ReqRing.Slot, ring_cap);
        errdefer allocator.free(req_slots);

        const comp_slots = try allocator.alloc(CompRing.Slot, ring_cap);
        errdefer allocator.free(comp_slots);

        state.* = .{
            .req_ring = try ReqRing.init(req_slots),
            .comp_ring = try CompRing.init(comp_slots),
            .req_slots = req_slots,
            .comp_slots = comp_slots,
            .pending_reqs = .init(0),
            .sleeping_workers = .init(0),
            .shutdown = .init(false),
            .work_mutex = std.c.PTHREAD_MUTEX_INITIALIZER,
            .work_cond = std.c.PTHREAD_COND_INITIALIZER,
            .pending_comps = .init(0),
            .comp_waiters = .init(0),
            .comp_mutex = std.c.PTHREAD_MUTEX_INITIALIZER,
            .comp_cond = std.c.PTHREAD_COND_INITIALIZER,
            .ready_count = .init(0),
            .init_mutex = std.c.PTHREAD_MUTEX_INITIALIZER,
            .init_cond = std.c.PTHREAD_COND_INITIALIZER,
            .worker_errors = @splat(null),
            .qos_class = config.qos_class,
            .io_policy = config.io_policy,
            .worker_count = config.worker_count,
        };

        var pool: EcorePreadPool = .{
            .state = state,
            .threads = undefined,
            .spawned = 0,
            .max_inflight = config.max_inflight,
            .inflight = 0,
        };
        errdefer pool.deinit(allocator);

        for (0..config.worker_count) |i| {
            pool.threads[i] = try std.Thread.spawn(.{}, workerMain, .{ state, i });
            pool.spawned += 1;
        }

        // Wait for all workers to apply E-core QoS and disk I/O policy.
        _ = std.c.pthread_mutex_lock(&state.init_mutex);
        while (state.ready_count.load(.acquire) < config.worker_count) {
            _ = std.c.pthread_cond_wait(&state.init_cond, &state.init_mutex);
        }
        _ = std.c.pthread_mutex_unlock(&state.init_mutex);

        for (0..config.worker_count) |i| {
            if (state.worker_errors[i]) |err| return err;
        }

        return pool;
    }

    /// Signals workers to exit cleanly after finishing any already-queued
    /// reads, joins every worker thread, and frees startup storage.
    pub fn deinit(self: *EcorePreadPool, allocator: std.mem.Allocator) void {
        const state = self.state;
        _ = std.c.pthread_mutex_lock(&state.work_mutex);
        state.shutdown.store(true, .release);
        _ = std.c.pthread_cond_broadcast(&state.work_cond);
        _ = std.c.pthread_mutex_unlock(&state.work_mutex);

        for (0..self.spawned) |i| {
            self.threads[i].join();
        }
        self.spawned = 0;
        self.inflight = 0;

        _ = std.c.pthread_cond_destroy(&state.init_cond);
        _ = std.c.pthread_mutex_destroy(&state.init_mutex);
        _ = std.c.pthread_cond_destroy(&state.comp_cond);
        _ = std.c.pthread_mutex_destroy(&state.comp_mutex);
        _ = std.c.pthread_cond_destroy(&state.work_cond);
        _ = std.c.pthread_mutex_destroy(&state.work_mutex);

        allocator.free(state.comp_slots);
        allocator.free(state.req_slots);
        allocator.destroy(state);
    }

    pub fn maxInflight(self: *const EcorePreadPool) usize {
        return self.max_inflight;
    }

    pub fn inflightCount(self: *const EcorePreadPool) usize {
        return self.inflight;
    }

    pub fn availableSlots(self: *const EcorePreadPool) usize {
        return self.max_inflight - self.inflight;
    }

    pub fn workerCount(self: *const EcorePreadPool) usize {
        return self.state.worker_count;
    }

    /// Enqueues one 16 KiB-aligned direct `pread` request with zero heap allocation.
    pub fn submit(self: *EcorePreadPool, req: ReadRequest) EngineError!void {
        try validateReadRequest(req);
        if (self.inflight >= self.max_inflight) return error.QueueFull;

        const item: WorkItem = .{
            .req = req,
            .submit_ns = monotonicNowNs(),
        };
        const state = self.state;
        if (!state.req_ring.tryPush(item)) return error.QueueFull;

        self.inflight += 1;
        _ = state.pending_reqs.fetchAdd(1, .seq_cst);

        if (state.sleeping_workers.load(.seq_cst) > 0) {
            _ = std.c.pthread_mutex_lock(&state.work_mutex);
            _ = std.c.pthread_cond_signal(&state.work_cond);
            _ = std.c.pthread_mutex_unlock(&state.work_mutex);
        }
    }

    /// Lock-free, non-blocking reap of available completions.
    pub fn poll(self: *EcorePreadPool, out: []Completion) EngineError!usize {
        if (out.len == 0 or self.inflight == 0) return 0;
        const state = self.state;
        var count: usize = 0;
        while (count < out.len) {
            const comp = state.comp_ring.tryPop() orelse break;
            out[count] = comp;
            count += 1;
            _ = state.pending_comps.fetchSub(1, .acq_rel);
            self.inflight -= 1;
        }
        return count;
    }

    /// Waits for at least one completion (or until `timeout_ms` elapses).
    pub fn wait(self: *EcorePreadPool, out: []Completion, timeout_ms: ?u64) EngineError!usize {
        if (out.len == 0 or self.inflight == 0) return 0;
        const fast = try self.poll(out);
        if (fast > 0) return fast;
        if (timeout_ms != null and timeout_ms.? == 0) return 0;

        const state = self.state;
        const deadline_ns: ?u64 = if (timeout_ms) |ms| monotonicNowNs() +| (ms *| std.time.ns_per_ms) else null;

        while (true) {
            _ = std.c.pthread_mutex_lock(&state.comp_mutex);
            _ = state.comp_waiters.fetchAdd(1, .seq_cst);
            while (state.pending_comps.load(.seq_cst) == 0) {
                if (deadline_ns) |dl| {
                    const now = monotonicNowNs();
                    if (now >= dl) {
                        _ = state.comp_waiters.fetchSub(1, .seq_cst);
                        _ = std.c.pthread_mutex_unlock(&state.comp_mutex);
                        return try self.poll(out);
                    }
                    var rel = nsToTimespec(dl - now);
                    const rc = pthread_cond_timedwait_relative_np(&state.comp_cond, &state.comp_mutex, &rel);
                    if (rc == .TIMEDOUT) {
                        _ = state.comp_waiters.fetchSub(1, .seq_cst);
                        _ = std.c.pthread_mutex_unlock(&state.comp_mutex);
                        return try self.poll(out);
                    }
                } else {
                    _ = std.c.pthread_cond_wait(&state.comp_cond, &state.comp_mutex);
                }
            }
            _ = state.comp_waiters.fetchSub(1, .seq_cst);
            _ = std.c.pthread_mutex_unlock(&state.comp_mutex);

            const got = try self.poll(out);
            if (got > 0) return got;
            // A producer incremented `pending_comps` just after `tryPush`, or
            // `tryPop` briefly observed a transient slot publish lag: retry.
            std.atomic.spinLoopHint();
        }
    }

    /// Submits one read and waits for its completion, returning bytes read.
    pub fn readSync(self: *EcorePreadPool, req: ReadRequest, timeout_ms: u64) EngineError!usize {
        try self.submit(req);
        const start_ns = monotonicNowNs();
        const deadline_ns = start_ns +| (timeout_ms *| std.time.ns_per_ms);
        var comp: [1]Completion = undefined;
        while (true) {
            const now = monotonicNowNs();
            if (now >= deadline_ns) return error.Timeout;
            const rem_ms = @max(@as(u64, 1), (deadline_ns - now) / std.time.ns_per_ms);
            const got = try self.wait(&comp, rem_ms);
            if (got > 0) {
                return try comp[0].check();
            }
        }
    }

    fn workerMain(state: *State, worker_idx: usize) void {
        // Apply E-core QoS first, then thread disk I/O policy (order required by
        // macOS XNU; see `src/platform/qos.zig`).
        var init_err: ?qos.Error = null;
        if (qos.setCurrentThreadQos(state.qos_class, 0)) {
            if (qos.setThreadDiskIoPolicy(state.io_policy)) {} else |err| {
                init_err = err;
            }
        } else |err| {
            init_err = err;
        }
        state.worker_errors[worker_idx] = init_err;

        _ = std.c.pthread_mutex_lock(&state.init_mutex);
        _ = state.ready_count.fetchAdd(1, .release);
        _ = std.c.pthread_cond_signal(&state.init_cond);
        _ = std.c.pthread_mutex_unlock(&state.init_mutex);

        if (init_err != null) return;

        while (true) {
            if (state.req_ring.tryPop()) |item| {
                _ = state.pending_reqs.fetchSub(1, .acq_rel);
                const comp = executePread(item);
                while (!state.comp_ring.tryPush(comp)) {
                    std.atomic.spinLoopHint();
                }
                _ = state.pending_comps.fetchAdd(1, .seq_cst);
                if (state.comp_waiters.load(.seq_cst) > 0) {
                    _ = std.c.pthread_mutex_lock(&state.comp_mutex);
                    _ = std.c.pthread_cond_signal(&state.comp_cond);
                    _ = std.c.pthread_mutex_unlock(&state.comp_mutex);
                }
                continue;
            }

            if (state.pending_reqs.load(.acquire) > 0) {
                std.atomic.spinLoopHint();
                continue;
            }
            if (state.shutdown.load(.acquire)) break;

            // Park cleanly on condition variable so idle workers burn 0% CPU.
            _ = std.c.pthread_mutex_lock(&state.work_mutex);
            _ = state.sleeping_workers.fetchAdd(1, .seq_cst);
            if (state.pending_reqs.load(.seq_cst) == 0 and !state.shutdown.load(.seq_cst)) {
                _ = std.c.pthread_cond_wait(&state.work_cond, &state.work_mutex);
            }
            _ = state.sleeping_workers.fetchSub(1, .seq_cst);
            _ = std.c.pthread_mutex_unlock(&state.work_mutex);
        }
    }

    fn executePread(item: WorkItem) Completion {
        const req = item.req;
        var done: usize = 0;
        var err_code: i32 = 0;
        var short_read = false;

        while (done < req.dst.len) {
            const rc = std.c.pread(
                req.fd,
                req.dst.ptr + done,
                req.dst.len - done,
                @intCast(req.offset + done),
            );
            if (rc < 0) {
                const err = std.c.errno(rc);
                if (err == .INTR) continue;
                err_code = @backingInt(err);
                break;
            }
            if (rc == 0) {
                short_read = true;
                break;
            }
            done += @intCast(rc);
        }

        return .{
            .user_data = req.user_data,
            .bytes_read = done,
            .err_code = err_code,
            .short_read = short_read,
            .submit_ns = item.submit_ns,
            .complete_ns = monotonicNowNs(),
        };
    }
};

// ---------------------------------------------------------------------------
// Unified AsyncExpertReader
// ---------------------------------------------------------------------------

pub const ReaderConfig = struct {
    backend: IoBackend = .auto,
    max_inflight: usize = 32,
    /// Number of E-core `pread` workers when using `.ecore_pread` (1..8).
    worker_count: usize = 4,
    /// E-core QoS class for `pread` workers (`.utility` or `.background`).
    worker_qos: qos.QosClass = .utility,
    /// Disk I/O policy for `pread` workers (`.important` or `.standard`).
    io_policy: qos.DiskIoPolicy = .important,
    /// Queue-depth threshold at which `.auto` prefers `ecore_pread` over
    /// `kqueue_aio` when `kern.aiothreads` caps concurrency (calibrated by Spike S3).
    auto_pread_threshold: ?usize = null,
};

/// Selects between `kqueue_aio` and `ecore_pread` for `IoBackend.auto`.
/// Calibrated by Spike S3 on Apple M4 (`kern.aiothreads = 4`):
/// - For small queue depths (`<= kern.aiothreads`), `kqueue_aio` matches or
///   slightly exceeds `ecore_pread` throughput (`0.61–0.63 GB/s`) with tight
///   `p99/p50` latency and zero extra user threads.
/// - Above `kern.aiothreads` (e.g. `QD = 16` cold-expert prefetch queues),
///   `kqueue_aio` exhibits kernel queue `p99` tail spikes (`1.5–2.0x` p50),
///   whereas 4-worker `ecore_pread` maintains strictly FIFO tail latency
///   (`1.01x` p99/p50) at full link bandwidth.
pub fn resolveAutoBackend(config: ReaderConfig, limits: ?KernelAioLimits) IoBackend {
    if (config.backend != .auto) return config.backend;
    const aio_threads: usize = if (limits) |lim|
        @intCast(@max(@as(u64, 1), lim.aio_threads))
    else
        4;
    const threshold = config.auto_pread_threshold orelse aio_threads;
    if (config.max_inflight > threshold) {
        return .ecore_pread;
    }
    return .kqueue_aio;
}

/// Unified asynchronous direct-I/O cold-expert reader wrapping both
/// `KqueueAioEngine` (C3) and `EcorePreadPool` (C4).
pub const AsyncExpertReader = union(enum) {
    kqueue_aio: KqueueAioEngine,
    ecore_pread: EcorePreadPool,

    pub fn init(allocator: std.mem.Allocator, config: ReaderConfig) EcorePoolError!AsyncExpertReader {
        const limits: ?KernelAioLimits = KernelAioLimits.query() catch null;
        const chosen = resolveAutoBackend(config, limits);
        return switch (chosen) {
            .kqueue_aio => .{ .kqueue_aio = try KqueueAioEngine.init(allocator, config.max_inflight) },
            .ecore_pread => .{ .ecore_pread = try EcorePreadPool.init(allocator, .{
                .max_inflight = config.max_inflight,
                .worker_count = config.worker_count,
                .qos_class = config.worker_qos,
                .io_policy = config.io_policy,
            }) },
            .auto => unreachable,
        };
    }

    pub fn deinit(self: *AsyncExpertReader, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .kqueue_aio => |*e| e.deinit(allocator),
            .ecore_pread => |*p| p.deinit(allocator),
        }
    }

    pub fn activeBackend(self: *const AsyncExpertReader) IoBackend {
        return switch (self.*) {
            .kqueue_aio => .kqueue_aio,
            .ecore_pread => .ecore_pread,
        };
    }

    pub fn maxInflight(self: *const AsyncExpertReader) usize {
        return switch (self.*) {
            .kqueue_aio => |*e| e.maxInflight(),
            .ecore_pread => |*p| p.maxInflight(),
        };
    }

    pub fn inflightCount(self: *const AsyncExpertReader) usize {
        return switch (self.*) {
            .kqueue_aio => |*e| e.inflightCount(),
            .ecore_pread => |*p| p.inflightCount(),
        };
    }

    pub fn availableSlots(self: *const AsyncExpertReader) usize {
        return switch (self.*) {
            .kqueue_aio => |*e| e.availableSlots(),
            .ecore_pread => |*p| p.availableSlots(),
        };
    }

    pub fn submit(self: *AsyncExpertReader, req: ReadRequest) EngineError!void {
        return switch (self.*) {
            .kqueue_aio => |*e| e.submit(req),
            .ecore_pread => |*p| p.submit(req),
        };
    }

    pub fn poll(self: *AsyncExpertReader, out: []Completion) EngineError!usize {
        return switch (self.*) {
            .kqueue_aio => |*e| e.poll(out),
            .ecore_pread => |*p| p.poll(out),
        };
    }

    pub fn wait(self: *AsyncExpertReader, out: []Completion, timeout_ms: ?u64) EngineError!usize {
        return switch (self.*) {
            .kqueue_aio => |*e| e.wait(out, timeout_ms),
            .ecore_pread => |*p| p.wait(out, timeout_ms),
        };
    }

    pub fn readSync(self: *AsyncExpertReader, req: ReadRequest, timeout_ms: u64) EngineError!usize {
        return switch (self.*) {
            .kqueue_aio => |*e| e.readSync(req, timeout_ms),
            .ecore_pread => |*p| p.readSync(req, timeout_ms),
        };
    }
};

// ---------------------------------------------------------------------------
// Spike S3: NVMe F_NOCACHE kqueue AIO vs E-core pread benchmark
// ---------------------------------------------------------------------------

pub const DEFAULT_S3_QUEUE_DEPTHS = [_]usize{ 1, 2, 4, 8, 16 };
pub const DEFAULT_S3_WORKER_COUNTS = [_]usize{ 1, 2, 4 };
pub const DEFAULT_S3_BLOB_SIZES = [_]usize{ EXPERT_BLOB_Q3_BYTES, EXPERT_BLOB_Q4_BYTES };
pub const DEFAULT_S3_IO_POLICIES = [_]qos.DiskIoPolicy{ .important, .standard };

pub const SpikeS3Config = struct {
    /// Directory for the temporary `F_NOCACHE` test file (removed on exit).
    temp_dir: []const u8 = "/Volumes/Akari/.tmp",
    /// Total file size in bytes (16 KiB aligned; default 192 MiB).
    file_bytes: usize = 192 * 1024 * 1024,
    /// Expert blob sizes to benchmark (default: Q3 2,064,384 B and Q4 2,654,208 B).
    blob_sizes: []const usize = &DEFAULT_S3_BLOB_SIZES,
    /// Queue depths to sweep (default: 1, 2, 4, 8, 16).
    queue_depths: []const usize = &DEFAULT_S3_QUEUE_DEPTHS,
    /// Worker counts for `ecore_pread` (default: 1, 2, 4).
    pread_workers: []const usize = &DEFAULT_S3_WORKER_COUNTS,
    /// Disk I/O policies to compare (default: `IOPOL_IMPORTANT` vs `IOPOL_STANDARD`).
    io_policies: []const qos.DiskIoPolicy = &DEFAULT_S3_IO_POLICIES,
    /// Warmup reads issued per configuration before timing.
    warmup_reads: usize = 8,
    /// Timed reads issued per configuration.
    timed_reads: usize = 64,
    /// Enforce `swapouts_delta == 0` across the entire Spike S3 run.
    require_zero_swap: bool = true,
    timeout_ms: u64 = 30_000,
};

pub const SpikeS3CaseResult = struct {
    blob_bytes: usize,
    backend: IoBackend,
    workers: u8,
    io_policy: qos.DiskIoPolicy,
    queue_depth: u16,
    reads: usize,
    bytes_read: u64,
    elapsed_ns: u64,
    throughput_gbs: f64,
    p50_us: f64,
    p99_us: f64,
    cpu_us: u64,
};

pub const SpikeS3Report = struct {
    file_bytes: usize,
    kern_aiothreads: u64,
    kern_aioprocmax: u64,
    results: []SpikeS3CaseResult,
    recommended_backend: IoBackend,
    recommended_workers: u8,
    recommended_io_policy: qos.DiskIoPolicy,
    peak_aio_gbs: f64,
    peak_pread_gbs: f64,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,
    swapouts_delta: u64,
    pageouts_delta: u64,

    pub fn deinit(self: *SpikeS3Report, allocator: std.mem.Allocator) void {
        allocator.free(self.results);
        self.results = &.{};
    }

    pub fn write(self: *const SpikeS3Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\spike_s3.file_mib                {d}
            \\spike_s3.kern_aiothreads         {d}
            \\spike_s3.kern_aioprocmax         {d}
            \\spike_s3.peak_aio_gbs            {d:.3}
            \\spike_s3.peak_pread_gbs          {d:.3}
            \\spike_s3.recommended_backend     {s}
            \\spike_s3.recommended_workers     {d}
            \\spike_s3.recommended_io_policy   {s}
            \\spike_s3.swapouts_delta          {d}
            \\spike_s3.pageouts_delta          {d}
            \\
            \\Spike S3 NVMe Direct-I/O Sweep (F_NOCACHE)
            \\blob_B     backend      wk  iopol      qd   reads   GB/s     p50(us)   p99(us)   cpu(us)
            \\
        , .{
            self.file_bytes >> 20,
            self.kern_aiothreads,
            self.kern_aioprocmax,
            self.peak_aio_gbs,
            self.peak_pread_gbs,
            self.recommended_backend.label(),
            self.recommended_workers,
            ioPolicyLabel(self.recommended_io_policy),
            self.swapouts_delta,
            self.pageouts_delta,
        });

        for (self.results) |r| {
            try out.print(
                "{d:<10} {s:<12} {d:<3} {s:<10} {d:<4} {d:<7} {d:<8.3} {d:<9.1} {d:<9.1} {d}\n",
                .{
                    r.blob_bytes,
                    r.backend.label(),
                    r.workers,
                    ioPolicyLabel(r.io_policy),
                    r.queue_depth,
                    r.reads,
                    r.throughput_gbs,
                    r.p50_us,
                    r.p99_us,
                    r.cpu_us,
                },
            );
        }
    }
};

pub const SpikeS3Error = EcorePoolError || page.WiredRegion.InitError ||
    vmstat.Error || vmstat.SwapError || error{ WriteFailed, IntegrityMismatch };

fn ioPolicyLabel(policy: qos.DiskIoPolicy) []const u8 {
    return switch (policy) {
        .default => "default",
        .important => "important",
        .passive => "passive",
        .throttle => "throttle",
        .utility => "utility",
        .standard => "standard",
    };
}

fn processCpuUs() u64 {
    var ru: std.c.rusage = std.mem.zeroes(std.c.rusage);
    if (std.c.getrusage(std.c.rusage.SELF, &ru) != 0) return 0;
    const u_us: u64 = @as(u64, @intCast(@max(ru.utime.sec, 0))) * 1_000_000 +
        @as(u64, @intCast(@max(ru.utime.usec, 0)));
    const s_us: u64 = @as(u64, @intCast(@max(ru.stime.sec, 0))) * 1_000_000 +
        @as(u64, @intCast(@max(ru.stime.usec, 0)));
    return u_us + s_us;
}

fn pageSentinelWord(page_index: u64) u64 {
    return (page_index *% 0x9E3779B97F4A7C15) ^ 0xA55A_5AA5_C33C_3CC3;
}

fn fillDeterministicPages(buf: []align(page.APPLE_PAGE_SIZE) u8, first_page_idx: u64) void {
    const num_pages = buf.len / page.APPLE_PAGE_SIZE;
    for (0..num_pages) |p| {
        const page_slice = buf[p * page.APPLE_PAGE_SIZE .. (p + 1) * page.APPLE_PAGE_SIZE];
        const tag = pageSentinelWord(first_page_idx + p);
        const words: []u64 = @alignCast(std.mem.bytesAsSlice(u64, page_slice));
        words[0] = tag;
        words[words.len - 1] = ~tag;
        // Touch middle word so sparse/compressed storage cannot elide pages.
        words[words.len / 2] = tag ^ 0x5555_5555_AAAA_AAAA;
    }
}

fn verifyBlobSentinels(buf: []align(page.APPLE_PAGE_SIZE) const u8, file_offset: u64) bool {
    const first_page = file_offset / page.APPLE_PAGE_SIZE;
    const last_page = (file_offset + buf.len - page.APPLE_PAGE_SIZE) / page.APPLE_PAGE_SIZE;
    const first_words: []const u64 = @alignCast(std.mem.bytesAsSlice(u64, buf[0..page.APPLE_PAGE_SIZE]));
    const last_words: []const u64 = @alignCast(std.mem.bytesAsSlice(u64, buf[buf.len - page.APPLE_PAGE_SIZE ..]));
    const first_tag = pageSentinelWord(first_page);
    const last_tag = pageSentinelWord(last_page);
    return first_words[0] == first_tag and
        first_words[first_words.len - 1] == ~first_tag and
        last_words[0] == last_tag and
        last_words[last_words.len - 1] == ~last_tag;
}

fn pwriteExact(fd: std.c.fd_t, src: []const u8, offset: u64) SpikeS3Error!void {
    var done: usize = 0;
    while (done < src.len) {
        const rc = std.c.pwrite(fd, src.ptr + done, src.len - done, @intCast(offset + done));
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.WriteFailed;
        }
        if (rc == 0) return error.WriteFailed;
        done += @intCast(rc);
    }
}

fn createSpikeFile(path_z: [:0]const u8, file_bytes: usize) SpikeS3Error!void {
    const fd = std.c.open(
        path_z.ptr,
        .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true },
        @as(std.c.mode_t, 0o600),
    );
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);

    // Enable F_NOCACHE on the write fd too so the OS unified buffer cache never
    // retains the freshly-written pages before our read sweep begins.
    try enableNoCache(fd);

    const chunk_bytes: usize = @min(file_bytes, 4 * 1024 * 1024);
    const chunk = try page.WiredRegion.init(chunk_bytes);
    defer chunk.deinit();
    @memset(chunk.bytes, 0);

    var written: usize = 0;
    while (written < file_bytes) {
        const take = @min(chunk_bytes, file_bytes - written);
        fillDeterministicPages(chunk.bytes[0..take], written / page.APPLE_PAGE_SIZE);
        try pwriteExact(fd, chunk.bytes[0..take], written);
        written += take;
    }
    if (std.c.fsync(fd) != 0) return error.WriteFailed;
}

fn percentileUs(sorted_ns: []const u64, pct_num: usize, pct_den: usize) f64 {
    if (sorted_ns.len == 0) return 0.0;
    const idx = @min(sorted_ns.len - 1, (sorted_ns.len * pct_num) / pct_den);
    return @as(f64, @floatFromInt(sorted_ns[idx])) / 1000.0;
}

const SpikeCaseParams = struct {
    allocator: std.mem.Allocator,
    fd: std.c.fd_t,
    file_bytes: usize,
    blob_bytes: usize,
    backend: IoBackend,
    workers: usize,
    io_policy: qos.DiskIoPolicy,
    queue_depth: usize,
    warmup_reads: usize,
    timed_reads: usize,
    wired_dst: []align(page.APPLE_PAGE_SIZE) u8,
    latencies_ns: []u64,
    timeout_ms: u64,
};

fn runOneSpikeCase(params: SpikeCaseParams) SpikeS3Error!SpikeS3CaseResult {
    const qd = params.queue_depth;
    const blob_bytes = params.blob_bytes;
    const num_blobs = params.file_bytes / blob_bytes;
    if (num_blobs == 0) return error.InvalidArgument;

    var reader = try AsyncExpertReader.init(params.allocator, .{
        .backend = params.backend,
        .max_inflight = qd,
        .worker_count = if (params.workers == 0) 1 else params.workers,
        .worker_qos = .utility,
        .io_policy = params.io_policy,
    });
    defer reader.deinit(params.allocator);

    // Track which destination slots (`0..qd-1`) are free for new submissions.
    var free_slots: [MAX_INFLIGHT]u16 = undefined;
    var free_top: usize = qd;
    for (0..qd) |i| {
        free_slots[qd - 1 - i] = @intCast(i);
    }
    var slot_offsets: [MAX_INFLIGHT]u64 = @splat(0);
    var comps: [MAX_INFLIGHT]Completion = undefined;

    // Deterministic pseudo-random stride over all blobs in the file so we avoid
    // sequential read-ahead while touching every blob cleanly.
    var rng_state: u64 = 0xD1B54A32D192ED03 ^ (@as(u64, blob_bytes) << 16) ^ (@as(u64, qd) << 4);
    const nextBlobOffset = struct {
        fn next(st: *u64, count: usize, stride: usize) u64 {
            // SplitMix64 step.
            st.* +%= 0x9E3779B97F4A7C15;
            var z = st.*;
            z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
            z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
            z = z ^ (z >> 31);
            return (z % @as(u64, count)) * @as(u64, stride);
        }
    }.next;

    // 1. Warmup phase (unguided by timer).
    var warmup_submitted: usize = 0;
    var warmup_completed: usize = 0;
    while (warmup_completed < params.warmup_reads) {
        while (warmup_submitted < params.warmup_reads and free_top > 0) {
            const s = free_slots[free_top - 1];
            free_top -= 1;
            const off = nextBlobOffset(&rng_state, num_blobs, blob_bytes);
            slot_offsets[s] = off;
            const dst: []align(page.APPLE_PAGE_SIZE) u8 = @alignCast(
                params.wired_dst[@as(usize, s) * blob_bytes .. (@as(usize, s) + 1) * blob_bytes],
            );
            try reader.submit(.{
                .fd = params.fd,
                .offset = off,
                .dst = dst,
                .user_data = s,
            });
            warmup_submitted += 1;
        }
        const got = try reader.wait(comps[0..qd], params.timeout_ms);
        if (got == 0) return error.Timeout;
        for (comps[0..got]) |c| {
            _ = try c.check();
            const s: usize = @intCast(c.user_data);
            const dst: []align(page.APPLE_PAGE_SIZE) const u8 = @alignCast(
                params.wired_dst[s * blob_bytes .. (s + 1) * blob_bytes],
            );
            if (!verifyBlobSentinels(dst, slot_offsets[s])) return error.IntegrityMismatch;
            free_slots[free_top] = @intCast(s);
            free_top += 1;
            warmup_completed += 1;
        }
    }

    // 2. Timed sweep.
    const cpu_before = processCpuUs();
    const t0 = try vmstat.nowNs();

    var submitted: usize = 0;
    var completed: usize = 0;
    var total_bytes: u64 = 0;
    while (completed < params.timed_reads) {
        while (submitted < params.timed_reads and free_top > 0) {
            const s = free_slots[free_top - 1];
            free_top -= 1;
            const off = nextBlobOffset(&rng_state, num_blobs, blob_bytes);
            slot_offsets[s] = off;
            const dst: []align(page.APPLE_PAGE_SIZE) u8 = @alignCast(
                params.wired_dst[@as(usize, s) * blob_bytes .. (@as(usize, s) + 1) * blob_bytes],
            );
            try reader.submit(.{
                .fd = params.fd,
                .offset = off,
                .dst = dst,
                .user_data = s,
            });
            submitted += 1;
        }
        const got = try reader.wait(comps[0..qd], params.timeout_ms);
        if (got == 0) return error.Timeout;
        for (comps[0..got]) |c| {
            const nbytes = try c.check();
            const s: usize = @intCast(c.user_data);
            const dst: []align(page.APPLE_PAGE_SIZE) const u8 = @alignCast(
                params.wired_dst[s * blob_bytes .. (s + 1) * blob_bytes],
            );
            if (!verifyBlobSentinels(dst, slot_offsets[s])) return error.IntegrityMismatch;
            params.latencies_ns[completed] = c.latencyNs();
            total_bytes += nbytes;
            free_slots[free_top] = @intCast(s);
            free_top += 1;
            completed += 1;
        }
    }

    const t1 = try vmstat.nowNs();
    const cpu_after = processCpuUs();
    const elapsed_ns = @max(@as(u64, 1), t1 -| t0);

    std.mem.sort(u64, params.latencies_ns[0..params.timed_reads], {}, std.sort.asc(u64));
    const p50 = percentileUs(params.latencies_ns[0..params.timed_reads], 50, 100);
    const p99 = percentileUs(params.latencies_ns[0..params.timed_reads], 99, 100);
    const gbs = @as(f64, @floatFromInt(total_bytes)) / @as(f64, @floatFromInt(elapsed_ns));

    return .{
        .blob_bytes = blob_bytes,
        .backend = params.backend,
        .workers = @intCast(params.workers),
        .io_policy = params.io_policy,
        .queue_depth = @intCast(qd),
        .reads = params.timed_reads,
        .bytes_read = total_bytes,
        .elapsed_ns = elapsed_ns,
        .throughput_gbs = gbs,
        .p50_us = p50,
        .p99_us = p99,
        .cpu_us = cpu_after -| cpu_before,
    };
}

/// Runs one `kqueue_aio` Spike S3 case on a fresh E-core thread with explicit
/// `io_policy` so the caller thread's QoS / disk I/O policy is never mutated.
fn runKqueueAioOnDedicatedThread(params: SpikeCaseParams) SpikeS3Error!SpikeS3CaseResult {
    const Outcome = struct {
        result: ?SpikeS3CaseResult = null,
        err: ?SpikeS3Error = null,
    };
    const Runner = struct {
        fn run(p: SpikeCaseParams, out: *Outcome) void {
            qos.setCurrentThreadQos(.utility, 0) catch |err| {
                out.err = err;
                return;
            };
            qos.setThreadDiskIoPolicy(p.io_policy) catch |err| {
                out.err = err;
                return;
            };
            out.result = runOneSpikeCase(p) catch |err| {
                out.err = err;
                return;
            };
        }
    };
    var outcome: Outcome = .{};
    const t = try std.Thread.spawn(.{}, Runner.run, .{ params, &outcome });
    t.join();
    if (outcome.err) |err| return err;
    return outcome.result orelse error.PollFailed;
}

/// Runs Spike S3 (ADR-0006): sweeps `kqueue_aio` vs `ecore_pread` (`1, 2, 4`
/// workers) across queue depths `1..16` and `IOPOL_IMPORTANT` vs
/// `IOPOL_STANDARD` on a temporary `F_NOCACHE` file in `config.temp_dir`
/// (always removed via `defer`), returning measured throughput, p50/p99
/// latencies, recommended backend, and `swapouts_delta`.
pub fn runSpikeS3(allocator: std.mem.Allocator, config: SpikeS3Config) SpikeS3Error!SpikeS3Report {
    if (config.file_bytes == 0 or config.file_bytes % page.APPLE_PAGE_SIZE != 0) {
        return error.MisalignedLength;
    }
    if (config.blob_sizes.len == 0 or config.queue_depths.len == 0 or config.io_policies.len == 0 or config.timed_reads == 0) {
        return error.InvalidArgument;
    }

    var max_blob: usize = 0;
    for (config.blob_sizes) |b| {
        if (b == 0 or b % page.APPLE_PAGE_SIZE != 0 or b > config.file_bytes) return error.InvalidArgument;
        max_blob = @max(max_blob, b);
    }
    var max_qd: usize = 0;
    for (config.queue_depths) |qd| {
        if (qd == 0 or qd > MAX_INFLIGHT) return error.InvalidArgument;
        max_qd = @max(max_qd, qd);
    }
    for (config.pread_workers) |w| {
        if (w == 0 or w > MAX_WORKERS) return error.InvalidArgument;
    }

    const limits = KernelAioLimits.query() catch KernelAioLimits{ .aio_threads = 4, .aio_proc_max = 256 };

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const pid = std.c.getpid();
    const stamp = monotonicNowNs();
    const path_slice = std.fmt.bufPrint(
        path_buf[0 .. path_buf.len - 1],
        "{s}/aminfer_spike_s3_{d}_{d}.bin",
        .{ config.temp_dir, pid, stamp },
    ) catch return error.InvalidArgument;
    path_buf[path_slice.len] = 0;
    const path_z: [:0]const u8 = path_buf[0..path_slice.len :0];

    try createSpikeFile(path_z, config.file_bytes);
    defer _ = std.c.unlink(path_z.ptr);

    var dfile = try DirectFile.openReadOnly(path_z);
    defer dfile.close();

    // Single preallocated wired buffer holding `max_qd` slots of `max_blob` bytes.
    const dst_bytes = std.math.mul(usize, max_qd, max_blob) catch return error.InvalidArgument;
    const wired_region = try page.WiredRegion.init(dst_bytes);
    defer wired_region.deinit();

    const latencies_ns = try allocator.alloc(u64, config.timed_reads);
    defer allocator.free(latencies_ns);

    const total_cases = config.blob_sizes.len * config.io_policies.len * config.queue_depths.len *
        (1 + config.pread_workers.len);
    const results = try allocator.alloc(SpikeS3CaseResult, total_cases);
    errdefer allocator.free(results);

    const vm_before = try vmstat.Snapshot.capture();
    var out_idx: usize = 0;

    var peak_aio_gbs: f64 = 0.0;
    var peak_pread_gbs: f64 = 0.0;
    var best_aio_policy: qos.DiskIoPolicy = .important;
    var best_pread_workers: u8 = 4;
    var best_pread_policy: qos.DiskIoPolicy = .important;

    for (config.blob_sizes) |blob_bytes| {
        for (config.io_policies) |io_policy| {
            // 1) kqueue_aio sweep across queue depths.
            for (config.queue_depths) |qd| {
                const res = try runKqueueAioOnDedicatedThread(.{
                    .allocator = allocator,
                    .fd = dfile.fd,
                    .file_bytes = config.file_bytes,
                    .blob_bytes = blob_bytes,
                    .backend = .kqueue_aio,
                    .workers = 0,
                    .io_policy = io_policy,
                    .queue_depth = qd,
                    .warmup_reads = config.warmup_reads,
                    .timed_reads = config.timed_reads,
                    .wired_dst = wired_region.bytes,
                    .latencies_ns = latencies_ns,
                    .timeout_ms = config.timeout_ms,
                });
                results[out_idx] = res;
                out_idx += 1;
                if (res.throughput_gbs > peak_aio_gbs) {
                    peak_aio_gbs = res.throughput_gbs;
                    best_aio_policy = io_policy;
                }
            }

            // 2) ecore_pread sweep across worker counts and queue depths.
            for (config.pread_workers) |workers| {
                for (config.queue_depths) |qd| {
                    const res = try runOneSpikeCase(.{
                        .allocator = allocator,
                        .fd = dfile.fd,
                        .file_bytes = config.file_bytes,
                        .blob_bytes = blob_bytes,
                        .backend = .ecore_pread,
                        .workers = workers,
                        .io_policy = io_policy,
                        .queue_depth = qd,
                        .warmup_reads = config.warmup_reads,
                        .timed_reads = config.timed_reads,
                        .wired_dst = wired_region.bytes,
                        .latencies_ns = latencies_ns,
                        .timeout_ms = config.timeout_ms,
                    });
                    results[out_idx] = res;
                    out_idx += 1;
                    if (res.throughput_gbs > peak_pread_gbs) {
                        peak_pread_gbs = res.throughput_gbs;
                        best_pread_workers = @intCast(workers);
                        best_pread_policy = io_policy;
                    }
                }
            }
        }
    }

    const vm_after = try vmstat.Snapshot.capture();
    const swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after);
    const pageouts_delta = vmstat.Snapshot.pageoutDelta(vm_before, vm_after);
    if (config.require_zero_swap) {
        try vmstat.checkNoSwap(vm_before, vm_after);
    }

    const prefer_pread = peak_pread_gbs >= peak_aio_gbs;
    return .{
        .file_bytes = config.file_bytes,
        .kern_aiothreads = limits.aio_threads,
        .kern_aioprocmax = limits.aio_proc_max,
        .results = results,
        .recommended_backend = if (prefer_pread) .ecore_pread else .kqueue_aio,
        .recommended_workers = if (prefer_pread) best_pread_workers else 0,
        .recommended_io_policy = if (prefer_pread) best_pread_policy else best_aio_policy,
        .peak_aio_gbs = peak_aio_gbs,
        .peak_pread_gbs = peak_pread_gbs,
        .vm_before = vm_before,
        .vm_after = vm_after,
        .swapouts_delta = swapouts_delta,
        .pageouts_delta = pageouts_delta,
    };
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

fn makeTestTempFile(name_tag: []const u8, total_bytes: usize, path_buf: *[std.fs.max_path_bytes]u8) ![:0]const u8 {
    const pid = std.c.getpid();
    const stamp = monotonicNowNs();
    const s = try std.fmt.bufPrint(
        path_buf[0 .. path_buf.len - 1],
        "/Volumes/Akari/.tmp/aminfer_aio_test_{s}_{d}_{d}.bin",
        .{ name_tag, pid, stamp },
    );
    path_buf[s.len] = 0;
    const path_z: [:0]const u8 = path_buf[0..s.len :0];

    const fd = std.c.open(
        path_z.ptr,
        .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true },
        @as(std.c.mode_t, 0o600),
    );
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    try enableNoCache(fd);

    const buf = try page.WiredRegion.init(total_bytes);
    defer buf.deinit();
    for (buf.bytes, 0..) |*b, i| {
        // Non-repeating byte pattern covering full multi-MiB range.
        b.* = @truncate((i *% 131) ^ (i >> 8) ^ (i >> 16) ^ 0x5A);
    }
    try pwriteExact(fd, buf.bytes, 0);
    if (std.c.fsync(fd) != 0) return error.WriteFailed;
    return path_z;
}

fn verifyExactPattern(buf: []const u8, file_offset: u64) !void {
    for (buf, 0..) |got, j| {
        const i: usize = @intCast(file_offset + j);
        const expected: u8 = @truncate((i *% 131) ^ (i >> 8) ^ (i >> 16) ^ 0x5A);
        if (got != expected) {
            return std.testing.expectEqual(expected, got);
        }
    }
}

test "Darwin AIO structs and kernel limits match baseline ABI" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Sigevent));
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(Aiocb));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(std.c.kevent64_s));

    const limits = try KernelAioLimits.query();
    try std.testing.expect(limits.aio_threads >= 1);
    try std.testing.expect(limits.aio_proc_max >= MAX_INFLIGHT);
}

test "validateReadRequest enforces 16 KiB alignment on offset, pointer and length" {
    const region = try page.WiredRegion.init(2 * page.APPLE_PAGE_SIZE);
    defer region.deinit();

    // Valid aligned request passes.
    try validateReadRequest(.{
        .fd = 3,
        .offset = page.APPLE_PAGE_SIZE,
        .dst = region.bytes[0..page.APPLE_PAGE_SIZE],
    });

    // Bad fd.
    try std.testing.expectError(error.InvalidArgument, validateReadRequest(.{
        .fd = -1,
        .offset = 0,
        .dst = region.bytes[0..page.APPLE_PAGE_SIZE],
    }));

    // Zero length.
    try std.testing.expectError(error.ZeroLength, validateReadRequest(.{
        .fd = 3,
        .offset = 0,
        .dst = region.bytes[0..0],
    }));

    // Misaligned offset.
    try std.testing.expectError(error.MisalignedOffset, validateReadRequest(.{
        .fd = 3,
        .offset = 4096,
        .dst = region.bytes[0..page.APPLE_PAGE_SIZE],
    }));

    // Misaligned length.
    try std.testing.expectError(error.MisalignedLength, validateReadRequest(.{
        .fd = 3,
        .offset = 0,
        .dst = region.bytes[0..8192],
    }));
}

test "KqueueAioEngine performs exact multi-MiB F_NOCACHE concurrent reads and detects short reads" {
    const gpa = std.testing.allocator;
    const blob_bytes: usize = 512 * 1024; // 512 KiB = 32 * 16 KiB
    const num_blobs: usize = 8; // 4 MiB total
    const file_bytes: usize = blob_bytes * num_blobs;

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try makeTestTempFile("kqueue", file_bytes, &path_buf);
    defer _ = std.c.unlink(path_z.ptr);

    var dfile = try DirectFile.openReadOnly(path_z);
    defer dfile.close();
    try std.testing.expectEqual(@as(u64, file_bytes), dfile.size);

    const dst_region = try page.WiredRegion.init(file_bytes);
    defer dst_region.deinit();
    @memset(dst_region.bytes, 0);

    var engine = try KqueueAioEngine.init(gpa, num_blobs);
    defer engine.deinit(gpa);

    // Submit all 8 blobs in scrambled order to verify slot tagging & zero corruption.
    const order = [_]usize{ 5, 2, 7, 0, 4, 1, 6, 3 };
    for (order, 0..) |blob_idx, slot| {
        const dst: []align(page.APPLE_PAGE_SIZE) u8 = @alignCast(
            dst_region.bytes[slot * blob_bytes .. (slot + 1) * blob_bytes],
        );
        try engine.submit(.{
            .fd = dfile.fd,
            .offset = @as(u64, blob_idx) * blob_bytes,
            .dst = dst,
            .user_data = (@as(u64, blob_idx) << 16) | @as(u64, slot),
        });
    }
    try std.testing.expectEqual(num_blobs, engine.inflightCount());
    try std.testing.expectEqual(@as(usize, 0), engine.availableSlots());

    // Queue full must be rejected cleanly.
    try std.testing.expectError(error.QueueFull, engine.submit(.{
        .fd = dfile.fd,
        .offset = 0,
        .dst = dst_region.bytes[0..page.APPLE_PAGE_SIZE],
    }));

    var completed: usize = 0;
    var seen_mask: u8 = 0;
    var comps: [8]Completion = undefined;
    while (completed < num_blobs) {
        const n = try engine.wait(&comps, 5_000);
        try std.testing.expect(n > 0);
        for (comps[0..n]) |c| {
            const nbytes = try c.check();
            try std.testing.expectEqual(blob_bytes, nbytes);
            const blob_idx: usize = @intCast(c.user_data >> 16);
            const slot: usize = @intCast(c.user_data & 0xFFFF);
            seen_mask |= @as(u8, 1) << @intCast(blob_idx);
            const slice = dst_region.bytes[slot * blob_bytes .. (slot + 1) * blob_bytes];
            try verifyExactPattern(slice, @as(u64, blob_idx) * blob_bytes);
            completed += 1;
        }
    }
    try std.testing.expectEqual(@as(u8, 0xFF), seen_mask);
    try std.testing.expectEqual(@as(usize, 0), engine.inflightCount());
    try std.testing.expectEqual(num_blobs, engine.availableSlots());

    // Past-EOF read must report error.ShortRead via Completion.check().
    try engine.submit(.{
        .fd = dfile.fd,
        .offset = file_bytes - page.APPLE_PAGE_SIZE,
        .dst = dst_region.bytes[0 .. 2 * page.APPLE_PAGE_SIZE],
        .user_data = 999,
    });
    const got = try engine.wait(comps[0..1], 5_000);
    try std.testing.expectEqual(@as(usize, 1), got);
    try std.testing.expect(comps[0].short_read);
    try std.testing.expectEqual(page.APPLE_PAGE_SIZE, comps[0].bytes_read);
    try std.testing.expectError(error.ShortRead, comps[0].check());
}

test "EcorePreadPool performs exact multi-MiB F_NOCACHE reads across 1, 2, 4 workers and idle wakeups" {
    const gpa = std.testing.allocator;
    const blob_bytes: usize = 512 * 1024;
    const num_blobs: usize = 8; // 4 MiB total
    const file_bytes: usize = blob_bytes * num_blobs;

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try makeTestTempFile("pread", file_bytes, &path_buf);
    defer _ = std.c.unlink(path_z.ptr);

    var dfile = try DirectFile.openReadOnly(path_z);
    defer dfile.close();

    const dst_region = try page.WiredRegion.init(file_bytes);
    defer dst_region.deinit();

    const configs = [_]struct { workers: usize, policy: qos.DiskIoPolicy }{
        .{ .workers = 1, .policy = .important },
        .{ .workers = 2, .policy = .standard },
        .{ .workers = 4, .policy = .important },
    };

    for (configs) |cfg| {
        @memset(dst_region.bytes, 0);
        var pool = try EcorePreadPool.init(gpa, .{
            .max_inflight = num_blobs,
            .worker_count = cfg.workers,
            .qos_class = .utility,
            .io_policy = cfg.policy,
        });
        defer pool.deinit(gpa);

        // Wave 1: fill all slots and verify exact byte integrity.
        for (0..num_blobs) |i| {
            const dst: []align(page.APPLE_PAGE_SIZE) u8 = @alignCast(
                dst_region.bytes[i * blob_bytes .. (i + 1) * blob_bytes],
            );
            try pool.submit(.{
                .fd = dfile.fd,
                .offset = @as(u64, i) * blob_bytes,
                .dst = dst,
                .user_data = i,
            });
        }
        try std.testing.expectError(error.QueueFull, pool.submit(.{
            .fd = dfile.fd,
            .offset = 0,
            .dst = dst_region.bytes[0..page.APPLE_PAGE_SIZE],
        }));

        var comps: [8]Completion = undefined;
        var done: usize = 0;
        while (done < num_blobs) {
            const n = try pool.wait(&comps, 5_000);
            try std.testing.expect(n > 0);
            for (comps[0..n]) |c| {
                try std.testing.expectEqual(blob_bytes, try c.check());
                const idx: usize = @intCast(c.user_data);
                try verifyExactPattern(
                    dst_region.bytes[idx * blob_bytes .. (idx + 1) * blob_bytes],
                    @as(u64, idx) * blob_bytes,
                );
                done += 1;
            }
        }
        try std.testing.expectEqual(@as(usize, 0), pool.inflightCount());

        // Let workers park on the condition variable, then issue a second wave to
        // verify clean wakeup with zero lost signals.
        var sleep_ts: std.c.timespec = .{ .sec = 0, .nsec = 5 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&sleep_ts, null);

        @memset(dst_region.bytes[0..blob_bytes], 0);
        const sync_bytes = try pool.readSync(.{
            .fd = dfile.fd,
            .offset = (num_blobs - 1) * blob_bytes,
            .dst = dst_region.bytes[0..blob_bytes],
            .user_data = 42,
        }, 5_000);
        try std.testing.expectEqual(blob_bytes, sync_bytes);
        try verifyExactPattern(dst_region.bytes[0..blob_bytes], (num_blobs - 1) * blob_bytes);
    }
}

test "AsyncExpertReader resolves auto backend and reads identically across backends" {
    const gpa = std.testing.allocator;
    const limits: KernelAioLimits = .{ .aio_threads = 4, .aio_proc_max = 256 };
    try std.testing.expectEqual(IoBackend.kqueue_aio, resolveAutoBackend(.{ .backend = .auto, .max_inflight = 4 }, limits));
    try std.testing.expectEqual(IoBackend.ecore_pread, resolveAutoBackend(.{ .backend = .auto, .max_inflight = 16 }, limits));
    try std.testing.expectEqual(IoBackend.kqueue_aio, resolveAutoBackend(.{ .backend = .kqueue_aio, .max_inflight = 16 }, limits));
    try std.testing.expectEqual(IoBackend.ecore_pread, resolveAutoBackend(.{ .backend = .ecore_pread, .max_inflight = 1 }, limits));

    const file_bytes: usize = 2 * 1024 * 1024;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try makeTestTempFile("unified", file_bytes, &path_buf);
    defer _ = std.c.unlink(path_z.ptr);

    var dfile = try DirectFile.openReadOnly(path_z);
    defer dfile.close();

    const dst = try page.WiredRegion.init(file_bytes);
    defer dst.deinit();

    for ([_]IoBackend{ .kqueue_aio, .ecore_pread, .auto }) |b| {
        @memset(dst.bytes, 0);
        var reader = try AsyncExpertReader.init(gpa, .{
            .backend = b,
            .max_inflight = 8,
            .worker_count = 2,
        });
        defer reader.deinit(gpa);
        if (b != .auto) try std.testing.expectEqual(b, reader.activeBackend());

        const got = try reader.readSync(.{
            .fd = dfile.fd,
            .offset = 0,
            .dst = dst.bytes,
            .user_data = 7,
        }, 5_000);
        try std.testing.expectEqual(file_bytes, got);
        try verifyExactPattern(dst.bytes, 0);
    }
}

test "runSpikeS3 executes compact end-to-end sweep in /Volumes/Akari/.tmp with zero swapouts" {
    const gpa = std.testing.allocator;
    const blobs = [_]usize{256 * 1024};
    const qds = [_]usize{ 1, 4 };
    const workers = [_]usize{ 1, 2 };
    const policies = [_]qos.DiskIoPolicy{.important};

    var report = try runSpikeS3(gpa, .{
        .temp_dir = "/Volumes/Akari/.tmp",
        .file_bytes = 4 * 1024 * 1024,
        .blob_sizes = &blobs,
        .queue_depths = &qds,
        .pread_workers = &workers,
        .io_policies = &policies,
        .warmup_reads = 2,
        .timed_reads = 8,
        .require_zero_swap = true,
    });
    defer report.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 6), report.results.len);
    try std.testing.expectEqual(@as(u64, 0), report.swapouts_delta);
    try std.testing.expect(report.peak_aio_gbs > 0.0);
    try std.testing.expect(report.peak_pread_gbs > 0.0);

    for (report.results) |r| {
        try std.testing.expect(r.throughput_gbs > 0.0);
        try std.testing.expect(r.p99_us >= r.p50_us);
        try std.testing.expectEqual(@as(u64, 8 * 256 * 1024), r.bytes_read);
    }

    var text_buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&text_buf);
    try report.write(&writer);
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "spike_s3.recommended_backend") != null);
}
