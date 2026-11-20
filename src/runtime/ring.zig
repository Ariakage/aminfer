//! Preallocated, zero-heap-allocation lock-free SPSC and MPSC rings
//! (status row C1, docs/architecture/runtime-pipeline.md).
//! Backed by caller-provided slices (carvable from `WiredArena`), with
//! producer and consumer atomics padded to 128-byte Apple Silicon cache lines.

const std = @import("std");

/// Apple Silicon L1/L2 cache line size in bytes.
pub const CACHE_LINE_BYTES: usize = 128;

pub const InitError = error{
    /// Ring capacity must be a non-zero power of two.
    InvalidCapacity,
    /// Backing byte slice pointer is not aligned to the slot type's alignment.
    UnalignedBacking,
    /// Backing byte slice length does not hold an exact power-of-two slot count.
    InvalidBackingLength,
    /// Requested byte size overflows `usize`.
    Overflow,
};

fn validateCapacity(cap: usize) InitError!void {
    const max_cap: usize = @as(usize, 1) << (@bitSizeOf(usize) - 2);
    if (cap == 0 or !std.math.isPowerOfTwo(cap) or cap > max_cap) {
        return error.InvalidCapacity;
    }
}

pub fn SpscRing(comptime T: type) type {
    return struct {
        const Self = @This();

        slots: []T,
        mask: usize,

        // Producer-owned cache line (128-byte aligned on Apple Silicon).
        tail: std.atomic.Value(usize) align(CACHE_LINE_BYTES),
        cached_head: usize,

        // Consumer-owned cache line (128-byte aligned on Apple Silicon).
        head: std.atomic.Value(usize) align(CACHE_LINE_BYTES),
        cached_tail: usize,

        pub const backing_alignment: usize = @alignOf(T);

        /// Computes the exact backing slice length in bytes for `cap` slots.
        pub fn requiredBytes(cap: usize) InitError!usize {
            try validateCapacity(cap);
            return std.math.mul(usize, cap, @sizeOf(T)) catch return error.Overflow;
        }

        /// Initializes a lock-free SPSC ring over a caller-provided slice.
        /// Performs zero dynamic allocation.
        pub fn init(slots: []T) InitError!Self {
            try validateCapacity(slots.len);
            return .{
                .slots = slots,
                .mask = slots.len - 1,
                .tail = .init(0),
                .cached_head = 0,
                .head = .init(0),
                .cached_tail = 0,
            };
        }

        /// Initializes a lock-free SPSC ring over raw aligned bytes (for example
        /// carved from `WiredArena.allocAligned`).
        pub fn initFromBytes(backing: []u8) InitError!Self {
            if (@sizeOf(T) == 0) return error.InvalidBackingLength;
            if (!std.mem.isAligned(@intFromPtr(backing.ptr), backing_alignment)) {
                return error.UnalignedBacking;
            }
            if (backing.len == 0 or backing.len % @sizeOf(T) != 0) {
                return error.InvalidBackingLength;
            }
            const count = backing.len / @sizeOf(T);
            if (!std.math.isPowerOfTwo(count)) return error.InvalidBackingLength;
            const aligned: []align(@alignOf(T)) u8 = @alignCast(backing);
            return init(std.mem.bytesAsSlice(T, aligned));
        }

        pub fn capacity(self: *const Self) usize {
            return self.slots.len;
        }

        /// Returns the exact count when quiescent, or a bounded snapshot under concurrency.
        pub fn len(self: *const Self) usize {
            const h = self.head.load(.acquire);
            const t = self.tail.load(.acquire);
            return @min(t -% h, self.slots.len);
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.len() == 0;
        }

        pub fn isFull(self: *const Self) bool {
            return self.len() == self.slots.len;
        }

        /// Producer-only lock-free non-blocking push. Returns `false` if full.
        pub fn tryPush(self: *Self, item: T) bool {
            const tail = self.tail.load(.monotonic);
            if (tail -% self.cached_head == self.slots.len) {
                self.cached_head = self.head.load(.acquire);
                if (tail -% self.cached_head == self.slots.len) return false;
            }
            self.slots[tail & self.mask] = item;
            self.tail.store(tail +% 1, .release);
            return true;
        }

        /// Consumer-only lock-free non-blocking pop. Returns `null` if empty.
        pub fn tryPop(self: *Self) ?T {
            const head = self.head.load(.monotonic);
            if (self.cached_tail == head) {
                self.cached_tail = self.tail.load(.acquire);
                if (self.cached_tail == head) return null;
            }
            const item = self.slots[head & self.mask];
            self.head.store(head +% 1, .release);
            return item;
        }
    };
}

pub fn MpscRing(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Per-slot sequence + payload for Vyukov's bounded lock-free queue.
        /// Encoding `2 * pos` = writable for ticket `pos`, `2 * pos + 1` = readable
        /// for ticket `pos`, allowing every power-of-two capacity including 1.
        pub const Slot = struct {
            sequence: std.atomic.Value(usize),
            value: T,
        };

        slots: []Slot,
        mask: usize,

        // Multi-producer ticket counter on its own 128-byte Apple Silicon cache line.
        tail: std.atomic.Value(usize) align(CACHE_LINE_BYTES),

        // Single-consumer index on its own 128-byte Apple Silicon cache line.
        head: std.atomic.Value(usize) align(CACHE_LINE_BYTES),

        pub const backing_alignment: usize = @alignOf(Slot);

        /// Computes the exact backing slice length in bytes for `cap` slots.
        pub fn requiredBytes(cap: usize) InitError!usize {
            try validateCapacity(cap);
            return std.math.mul(usize, cap, @sizeOf(Slot)) catch return error.Overflow;
        }

        /// Initializes a lock-free MPSC ring over caller-provided slots.
        /// Performs zero dynamic allocation.
        pub fn init(slots: []Slot) InitError!Self {
            try validateCapacity(slots.len);
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

        /// Initializes a lock-free MPSC ring over raw aligned bytes (for example
        /// carved from `WiredArena.allocAligned`).
        pub fn initFromBytes(backing: []u8) InitError!Self {
            if (!std.mem.isAligned(@intFromPtr(backing.ptr), backing_alignment)) {
                return error.UnalignedBacking;
            }
            if (backing.len == 0 or backing.len % @sizeOf(Slot) != 0) {
                return error.InvalidBackingLength;
            }
            const count = backing.len / @sizeOf(Slot);
            if (!std.math.isPowerOfTwo(count)) return error.InvalidBackingLength;
            const aligned: []align(@alignOf(Slot)) u8 = @alignCast(backing);
            return init(std.mem.bytesAsSlice(Slot, aligned));
        }

        pub fn capacity(self: *const Self) usize {
            return self.slots.len;
        }

        /// Returns the exact count when quiescent, or a bounded snapshot under concurrency.
        pub fn len(self: *const Self) usize {
            const h = self.head.load(.acquire);
            const t = self.tail.load(.acquire);
            return @min(t -% h, self.slots.len);
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.len() == 0;
        }

        pub fn isFull(self: *const Self) bool {
            return self.len() == self.slots.len;
        }

        /// Multi-producer-safe lock-free non-blocking push. Returns `false` if full.
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

        /// Single-consumer lock-free non-blocking pop. Returns `null` if empty
        /// (or if the head ticket's producer has not yet finished publishing).
        pub fn tryPop(self: *Self) ?T {
            const pos = self.head.load(.monotonic);
            const slot = &self.slots[pos & self.mask];
            const expected = (pos << 1) +% 1;
            const seq = slot.sequence.load(.acquire);
            if (seq != expected) return null;
            const item = slot.value;
            slot.sequence.store((pos +% self.slots.len) << 1, .release);
            self.head.store(pos +% 1, .release);
            return item;
        }
    };
}

test "SpscRing and MpscRing pad producer and consumer atomics to 128-byte cache lines" {
    const Spsc = SpscRing(u64);
    try std.testing.expect(@alignOf(Spsc) >= CACHE_LINE_BYTES);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Spsc, "tail") % CACHE_LINE_BYTES);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Spsc, "head") % CACHE_LINE_BYTES);
    try std.testing.expect(@offsetOf(Spsc, "head") - @offsetOf(Spsc, "tail") >= CACHE_LINE_BYTES);

    const Mpsc = MpscRing(u64);
    try std.testing.expect(@alignOf(Mpsc) >= CACHE_LINE_BYTES);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Mpsc, "tail") % CACHE_LINE_BYTES);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Mpsc, "head") % CACHE_LINE_BYTES);
    try std.testing.expect(@offsetOf(Mpsc, "head") - @offsetOf(Mpsc, "tail") >= CACHE_LINE_BYTES);
}

test "SpscRing validates power-of-two capacity and aligned byte backing" {
    const Ring = SpscRing(u64);
    var empty: [0]u64 = .{};
    try std.testing.expectError(error.InvalidCapacity, Ring.init(&empty));

    var three: [3]u64 = undefined;
    try std.testing.expectError(error.InvalidCapacity, Ring.init(&three));

    var six: [6]u64 = undefined;
    try std.testing.expectError(error.InvalidCapacity, Ring.init(&six));

    try std.testing.expectError(error.InvalidCapacity, Ring.requiredBytes(0));
    try std.testing.expectError(error.InvalidCapacity, Ring.requiredBytes(5));
    try std.testing.expectEqual(@as(usize, 8 * @sizeOf(u64)), try Ring.requiredBytes(8));

    var raw: [8 * @sizeOf(u64) + 1]u8 align(Ring.backing_alignment) = undefined;
    try std.testing.expectError(error.UnalignedBacking, Ring.initFromBytes(raw[1..]));
    try std.testing.expectError(error.InvalidBackingLength, Ring.initFromBytes(raw[0 .. 3 * @sizeOf(u64)]));

    var ring = try Ring.initFromBytes(raw[0 .. 8 * @sizeOf(u64)]);
    try std.testing.expectEqual(@as(usize, 8), ring.capacity());
    try std.testing.expect(ring.isEmpty());
    try std.testing.expect(!ring.isFull());
}

test "SpscRing enforces empty/full boundaries, FIFO order, and wrap-around" {
    const Ring = SpscRing(u32);

    // Edge case: capacity = 1.
    var one_slot: [1]u32 = undefined;
    var single = try Ring.init(&one_slot);
    try std.testing.expectEqual(@as(?u32, null), single.tryPop());
    try std.testing.expect(single.tryPush(42));
    try std.testing.expect(single.isFull());
    try std.testing.expect(!single.tryPush(99));
    try std.testing.expectEqual(@as(?u32, 42), single.tryPop());
    try std.testing.expect(single.isEmpty());
    try std.testing.expect(single.tryPush(77));
    try std.testing.expectEqual(@as(?u32, 77), single.tryPop());

    // Multi-cycle wrap-around on capacity = 4.
    var storage: [4]u32 = undefined;
    var ring = try Ring.init(&storage);

    var next_push: u32 = 0;
    var next_pop: u32 = 0;
    for (0..32) |_| {
        while (ring.tryPush(next_push)) {
            next_push += 1;
        }
        try std.testing.expect(ring.isFull());
        try std.testing.expectEqual(@as(usize, 4), ring.len());

        // Pop 3 items so indices advance out of phase with capacity.
        for (0..3) |_| {
            try std.testing.expectEqual(@as(?u32, next_pop), ring.tryPop());
            next_pop += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), ring.len());
    }
    while (ring.tryPop()) |v| {
        try std.testing.expectEqual(next_pop, v);
        next_pop += 1;
    }
    try std.testing.expectEqual(next_push, next_pop);
    try std.testing.expect(ring.isEmpty());
}

test "SpscRing multi-threaded stress test proves lossless FIFO delivery" {
    const Ring = SpscRing(u64);
    const total_items: u64 = 200_000;

    var storage: [64]u64 = undefined;
    var ring = try Ring.init(&storage);

    const Producer = struct {
        fn run(r: *Ring) void {
            var i: u64 = 0;
            while (i < total_items) {
                if (r.tryPush(i)) {
                    i += 1;
                } else {
                    std.atomic.spinLoopHint();
                }
            }
        }
    };

    const prod_thread = try std.Thread.spawn(.{}, Producer.run, .{&ring});
    defer prod_thread.join();

    var expected: u64 = 0;
    while (expected < total_items) {
        if (ring.tryPop()) |got| {
            try std.testing.expectEqual(expected, got);
            expected += 1;
        } else {
            std.atomic.spinLoopHint();
        }
    }
    try std.testing.expectEqual(total_items, expected);
    try std.testing.expect(ring.isEmpty());
}

test "MpscRing validates power-of-two capacity and aligned byte backing" {
    const Ring = MpscRing(u64);
    var empty: [0]Ring.Slot = .{};
    try std.testing.expectError(error.InvalidCapacity, Ring.init(&empty));

    var three: [3]Ring.Slot = undefined;
    try std.testing.expectError(error.InvalidCapacity, Ring.init(&three));

    try std.testing.expectError(error.InvalidCapacity, Ring.requiredBytes(0));
    try std.testing.expectError(error.InvalidCapacity, Ring.requiredBytes(6));
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(Ring.Slot)), try Ring.requiredBytes(4));

    var raw: [4 * @sizeOf(Ring.Slot) + 1]u8 align(Ring.backing_alignment) = undefined;
    try std.testing.expectError(error.UnalignedBacking, Ring.initFromBytes(raw[1..]));
    try std.testing.expectError(error.InvalidBackingLength, Ring.initFromBytes(raw[0 .. 3 * @sizeOf(Ring.Slot)]));

    var ring = try Ring.initFromBytes(raw[0 .. 4 * @sizeOf(Ring.Slot)]);
    try std.testing.expectEqual(@as(usize, 4), ring.capacity());
    try std.testing.expect(ring.isEmpty());
}

test "MpscRing enforces empty/full boundaries and wrap-around" {
    const Ring = MpscRing(u32);

    // Edge case: capacity = 1.
    var one_slot: [1]Ring.Slot = undefined;
    var single = try Ring.init(&one_slot);
    try std.testing.expectEqual(@as(?u32, null), single.tryPop());
    try std.testing.expect(single.tryPush(11));
    try std.testing.expect(!single.tryPush(22));
    try std.testing.expectEqual(@as(?u32, 11), single.tryPop());
    try std.testing.expect(single.tryPush(33));
    try std.testing.expectEqual(@as(?u32, 33), single.tryPop());

    // Multi-cycle wrap-around on capacity = 4.
    var storage: [4]Ring.Slot = undefined;
    var ring = try Ring.init(&storage);

    var next_push: u32 = 0;
    var next_pop: u32 = 0;
    for (0..32) |_| {
        while (ring.tryPush(next_push)) {
            next_push += 1;
        }
        try std.testing.expect(ring.isFull());
        for (0..3) |_| {
            try std.testing.expectEqual(@as(?u32, next_pop), ring.tryPop());
            next_pop += 1;
        }
    }
    while (ring.tryPop()) |v| {
        try std.testing.expectEqual(next_pop, v);
        next_pop += 1;
    }
    try std.testing.expectEqual(next_push, next_pop);
    try std.testing.expect(ring.isEmpty());
}

test "MpscRing multi-threaded stress test proves lossless multi-producer delivery" {
    const Msg = struct {
        producer_id: u32,
        seq: u32,
    };
    const Ring = MpscRing(Msg);
    const num_producers: usize = 4;
    const per_producer: u32 = 50_000;
    const total_items: usize = num_producers * per_producer;

    var storage: [128]Ring.Slot = undefined;
    var ring = try Ring.init(&storage);

    const Producer = struct {
        fn run(r: *Ring, pid: u32) void {
            var seq: u32 = 0;
            while (seq < per_producer) {
                if (r.tryPush(.{ .producer_id = pid, .seq = seq })) {
                    seq += 1;
                } else {
                    std.atomic.spinLoopHint();
                }
            }
        }
    };

    var threads: [num_producers]std.Thread = undefined;
    var spawned: usize = 0;
    defer for (threads[0..spawned]) |t| t.join();
    for (&threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, Producer.run, .{ &ring, @as(u32, @intCast(i)) });
        spawned += 1;
    }

    var expected_seq: [num_producers]u32 = @splat(0);
    var received: usize = 0;
    while (received < total_items) {
        if (ring.tryPop()) |msg| {
            try std.testing.expect(msg.producer_id < num_producers);
            try std.testing.expectEqual(expected_seq[msg.producer_id], msg.seq);
            expected_seq[msg.producer_id] += 1;
            received += 1;
        } else {
            std.atomic.spinLoopHint();
        }
    }

    for (expected_seq) |s| {
        try std.testing.expectEqual(per_producer, s);
    }
    try std.testing.expectEqual(@as(?Msg, null), ring.tryPop());
    try std.testing.expect(ring.isEmpty());
}

test "SpscRing and MpscRing survive counter wrap-around across integer boundaries" {
    // SPSC across maxInt(usize) -> 0.
    var spsc_slots: [4]u32 = undefined;
    var spsc = try SpscRing(u32).init(&spsc_slots);
    const spsc_start = std.math.maxInt(usize) - 1;
    spsc.tail = .init(spsc_start);
    spsc.head = .init(spsc_start);
    spsc.cached_head = spsc_start;
    spsc.cached_tail = spsc_start;

    for (0..8) |i| {
        const v: u32 = @intCast(i + 100);
        try std.testing.expect(spsc.tryPush(v));
        try std.testing.expectEqual(@as(?u32, v), spsc.tryPop());
    }
    try std.testing.expect(spsc.isEmpty());

    // MPSC across 2^(N-1) where `(pos << 1)` wraps modulo 2^N.
    var mpsc_slots: [4]MpscRing(u32).Slot = undefined;
    var mpsc = try MpscRing(u32).init(&mpsc_slots);
    const mpsc_start: usize = (@as(usize, 1) << (@bitSizeOf(usize) - 1)) - 2;
    mpsc.tail = .init(mpsc_start);
    mpsc.head = .init(mpsc_start);
    for (0..mpsc_slots.len) |i| {
        const idx = (mpsc_start +% i) & mpsc.mask;
        mpsc.slots[idx].sequence = .init((mpsc_start +% i) << 1);
    }

    for (0..8) |i| {
        const v: u32 = @intCast(i + 200);
        try std.testing.expect(mpsc.tryPush(v));
        try std.testing.expectEqual(@as(?u32, v), mpsc.tryPop());
    }
    try std.testing.expect(mpsc.isEmpty());
}
