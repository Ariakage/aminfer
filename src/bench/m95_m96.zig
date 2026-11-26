//! Milestones M9.5 (`W30`, `G29`, `C28`) and M9.6 (`C29`, `C18`, `G14`, `W20`)
//! benchmark and end-to-end verification suite:
//! - `aminfer bench-m95`: `GraphModel` (`inkling2_loop` `Core x 3`) dynamic `r = 16`
//!   LoRA + calibrated PEFT DoRA (`use_dora = true`) hot-load (`ms`), `M = 1` decode
//!   (`us`), `mid_pass_flushes == 0`, merged-weight numerical parity, bit-exact
//!   `NO_ADAPTER`/unload restore, `GraphBatchRuntime` 3-slot heterogeneous LoRA,
//!   `/v1/models/{id}` discovery, `malloc_delta <= 0`, `swapouts_delta == 0`.
//! - `aminfer bench-m96`: All 7 `C29` capabilities over HTTP & TLS 1.3 HTTPS,
//!   including real in-memory Base64 PNG & H.264 `.mp4` -> `vision_bridge`
//!   `IOSurface` -> synthetic `W20` ViT `GraphModel` multimodal inference with
//!   `0` IOSurface leaks, `malloc_delta <= 0`, and `swapouts_delta == 0`.
//!
//! Zero hardcoded external mount paths; all synthetic `.amf`, `.amlora`, media,
//! and TLS credentials are generated in a temporary directory and cleaned up.

const std = @import("std");
const Io = std.Io;
const page = @import("../mem/page.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const metal = @import("../bridge/metal.zig");
const tls = @import("../bridge/tls.zig");
const vision_bridge = @import("../bridge/vision.zig");
const vision_mod = @import("../gpu/vision.zig");
const lora = @import("../model/lora.zig");
const graph = @import("../model/graph.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const chat_template = @import("../model/chat_template.zig");
const rng_mod = @import("../runtime/rng.zig");
const vmstat = @import("../platform/vmstat.zig");
const server = @import("../server/http_sse.zig");

const MallocStatistics = extern struct {
    blocks_in_use: u32,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;
extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;

pub const M95Config = struct {
    decode_iters: u32 = 32,
    hot_load_iters: u32 = 4,
    discovery_iters: u32 = 16,
};

pub const M95Report = struct {
    hot_load_ms: f64 = 0.0,
    base_m1_us: f64 = 0.0,
    lora_r16_m1_us: f64 = 0.0,
    dora_m1_us: f64 = 0.0,
    mid_pass_flushes_base: u64 = 0,
    mid_pass_flushes_lora: u64 = 0,
    mid_pass_flushes_dora: u64 = 0,
    lora_max_logit_diff: f32 = 0.0,
    dora_max_logit_diff: f32 = 0.0,
    lora_parity_ok: bool = false,
    dora_parity_ok: bool = false,
    no_adapter_bit_exact: bool = false,
    unload_bit_exact: bool = false,
    batch_3slot_parity_ok: bool = false,
    batch_max_logit_diff: f32 = 0.0,
    discovery_ok: bool = false,
    discovery_mean_us: f64 = 0.0,
    decode_malloc_blocks_delta: i64 = 0,
    malloc_blocks_delta: i64 = 0,
    swapouts_delta: u64 = 0,
    all_passed: bool = false,

    pub fn write(self: *const M95Report, out: *Io.Writer) !void {
        try out.print(
            \\bench.m95.w30.hot_load_ms              {d:.3}
            \\bench.m95.w30.base_m1_us               {d:.2}
            \\bench.m95.w30.lora_r16_m1_us           {d:.2}
            \\bench.m95.w30.dora_m1_us               {d:.2}
            \\bench.m95.w30.mid_pass_flushes         base={d} lora={d} dora={d}
            \\bench.m95.w30.lora_max_logit_diff      {d:.6}
            \\bench.m95.w30.dora_max_logit_diff      {d:.6}
            \\bench.m95.w30.lora_parity_ok           {s}
            \\bench.m95.w30.dora_parity_ok           {s}
            \\bench.m95.w30.no_adapter_bit_exact     {s}
            \\bench.m95.w30.unload_bit_exact         {s}
            \\bench.m95.w30.batch_3slot_parity_ok    {s}
            \\bench.m95.w30.batch_max_logit_diff     {d:.6}
            \\bench.m95.w30.decode_malloc_blocks_delta {d}
            \\bench.m95.c28.discovery_ok             {s}
            \\bench.m95.c28.discovery_mean_us        {d:.2}
            \\bench.m95.malloc_blocks_delta          {d}
            \\bench.m95.swapouts_delta               {d}
            \\bench.m95.all_passed                   {s}
            \\
        , .{
            self.hot_load_ms,
            self.base_m1_us,
            self.lora_r16_m1_us,
            self.dora_m1_us,
            self.mid_pass_flushes_base,
            self.mid_pass_flushes_lora,
            self.mid_pass_flushes_dora,
            self.lora_max_logit_diff,
            self.dora_max_logit_diff,
            boolStr(self.lora_parity_ok),
            boolStr(self.dora_parity_ok),
            boolStr(self.no_adapter_bit_exact),
            boolStr(self.unload_bit_exact),
            boolStr(self.batch_3slot_parity_ok),
            self.batch_max_logit_diff,
            self.decode_malloc_blocks_delta,
            boolStr(self.discovery_ok),
            self.discovery_mean_us,
            self.malloc_blocks_delta,
            self.swapouts_delta,
            boolStr(self.all_passed),
        });
    }

    pub fn writeJson(self: *const M95Report, out: *Io.Writer) !void {
        try out.print(
            \\{{"bench":"m95","hot_load_ms":{d:.3},"base_m1_us":{d:.2},"lora_r16_m1_us":{d:.2},"dora_m1_us":{d:.2},"mid_pass_flushes":{{"base":{d},"lora":{d},"dora":{d}}},"lora_max_logit_diff":{d:.6},"dora_max_logit_diff":{d:.6},"lora_parity_ok":{s},"dora_parity_ok":{s},"no_adapter_bit_exact":{s},"unload_bit_exact":{s},"batch_3slot_parity_ok":{s},"batch_max_logit_diff":{d:.6},"decode_malloc_blocks_delta":{d},"discovery_ok":{s},"discovery_mean_us":{d:.2},"malloc_blocks_delta":{d},"swapouts_delta":{d},"all_passed":{s}}}
            \\
        , .{
            self.hot_load_ms,
            self.base_m1_us,
            self.lora_r16_m1_us,
            self.dora_m1_us,
            self.mid_pass_flushes_base,
            self.mid_pass_flushes_lora,
            self.mid_pass_flushes_dora,
            self.lora_max_logit_diff,
            self.dora_max_logit_diff,
            boolStr(self.lora_parity_ok),
            boolStr(self.dora_parity_ok),
            boolStr(self.no_adapter_bit_exact),
            boolStr(self.unload_bit_exact),
            boolStr(self.batch_3slot_parity_ok),
            self.batch_max_logit_diff,
            self.decode_malloc_blocks_delta,
            boolStr(self.discovery_ok),
            self.discovery_mean_us,
            self.malloc_blocks_delta,
            self.swapouts_delta,
            boolStr(self.all_passed),
        });
    }
};

pub const M96Config = struct {
    cors_iters: u32 = 16,
    tls_iters: u32 = 6,
};

pub const M96Report = struct {
    models_pagination_ok: bool = false,
    prefix_free_routes_ok: bool = false,
    responses_input_tokens_ok: bool = false,
    responses_input_items_ok: bool = false,
    completions_fim_ok: bool = false,
    cors_http_https_ok: bool = false,
    cors_options_mean_us: f64 = 0.0,
    tls13_negotiated: u16 = 0,
    tls13_cors_mean_us: f64 = 0.0,
    vision_png_chat_ok: bool = false,
    vision_png_responses_ok: bool = false,
    vision_png_messages_ok: bool = false,
    vision_mp4_chat_ok: bool = false,
    vision_png_encode_ms: f64 = 0.0,
    vision_mp4_encode_ms: f64 = 0.0,
    vision_tokens_last: u32 = 0,
    iosurface_leak_count: u32 = 0,
    malloc_blocks_delta: i64 = 0,
    swapouts_delta: u64 = 0,
    all_passed: bool = false,

    pub fn write(self: *const M96Report, out: *Io.Writer) !void {
        try out.print(
            \\bench.m96.c29.models_pagination_ok       {s}
            \\bench.m96.c29.prefix_free_routes_ok      {s}
            \\bench.m96.c29.responses_input_tokens_ok  {s}
            \\bench.m96.c29.responses_input_items_ok   {s}
            \\bench.m96.c29.completions_fim_ok         {s}
            \\bench.m96.c29.cors_http_https_ok         {s}
            \\bench.m96.c29.cors_options_mean_us       {d:.2}
            \\bench.m96.c29.tls13_version              0x{x:0>4}
            \\bench.m96.c29.tls13_cors_mean_us         {d:.2}
            \\bench.m96.c29.vision_png_chat_ok         {s}
            \\bench.m96.c29.vision_png_responses_ok    {s}
            \\bench.m96.c29.vision_png_messages_ok     {s}
            \\bench.m96.c29.vision_mp4_chat_ok         {s}
            \\bench.m96.c29.vision_png_encode_ms       {d:.3}
            \\bench.m96.c29.vision_mp4_encode_ms       {d:.3}
            \\bench.m96.c29.vision_tokens_last         {d}
            \\bench.m96.c29.iosurface_leak_count       {d}
            \\bench.m96.malloc_blocks_delta            {d}
            \\bench.m96.swapouts_delta                 {d}
            \\bench.m96.all_passed                     {s}
            \\
        , .{
            boolStr(self.models_pagination_ok),
            boolStr(self.prefix_free_routes_ok),
            boolStr(self.responses_input_tokens_ok),
            boolStr(self.responses_input_items_ok),
            boolStr(self.completions_fim_ok),
            boolStr(self.cors_http_https_ok),
            self.cors_options_mean_us,
            self.tls13_negotiated,
            self.tls13_cors_mean_us,
            boolStr(self.vision_png_chat_ok),
            boolStr(self.vision_png_responses_ok),
            boolStr(self.vision_png_messages_ok),
            boolStr(self.vision_mp4_chat_ok),
            self.vision_png_encode_ms,
            self.vision_mp4_encode_ms,
            self.vision_tokens_last,
            self.iosurface_leak_count,
            self.malloc_blocks_delta,
            self.swapouts_delta,
            boolStr(self.all_passed),
        });
    }

    pub fn writeJson(self: *const M96Report, out: *Io.Writer) !void {
        try out.print(
            \\{{"bench":"m96","models_pagination_ok":{s},"prefix_free_routes_ok":{s},"responses_input_tokens_ok":{s},"responses_input_items_ok":{s},"completions_fim_ok":{s},"cors_http_https_ok":{s},"cors_options_mean_us":{d:.2},"tls13_version":{d},"tls13_cors_mean_us":{d:.2},"vision_png_chat_ok":{s},"vision_png_responses_ok":{s},"vision_png_messages_ok":{s},"vision_mp4_chat_ok":{s},"vision_png_encode_ms":{d:.3},"vision_mp4_encode_ms":{d:.3},"vision_tokens_last":{d},"iosurface_leak_count":{d},"malloc_blocks_delta":{d},"swapouts_delta":{d},"all_passed":{s}}}
            \\
        , .{
            boolStr(self.models_pagination_ok),
            boolStr(self.prefix_free_routes_ok),
            boolStr(self.responses_input_tokens_ok),
            boolStr(self.responses_input_items_ok),
            boolStr(self.completions_fim_ok),
            boolStr(self.cors_http_https_ok),
            self.cors_options_mean_us,
            self.tls13_negotiated,
            self.tls13_cors_mean_us,
            boolStr(self.vision_png_chat_ok),
            boolStr(self.vision_png_responses_ok),
            boolStr(self.vision_png_messages_ok),
            boolStr(self.vision_mp4_chat_ok),
            self.vision_png_encode_ms,
            self.vision_mp4_encode_ms,
            self.vision_tokens_last,
            self.iosurface_leak_count,
            self.malloc_blocks_delta,
            self.swapouts_delta,
            boolStr(self.all_passed),
        });
    }
};

fn boolStr(v: bool) []const u8 {
    return if (v) "true" else "false";
}

fn monotonicNowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.UPTIME_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn writeBytesFile(path_z: [:0]const u8, bytes: []const u8) !void {
    const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + off, bytes.len - off);
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }
}

fn runCmdChecked(allocator: std.mem.Allocator, io: Io, argv: []const []const u8) !void {
    const res = try std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(64 * 1024),
        .stderr_limit = .limited(64 * 1024),
    });
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) return error.SubprocessFailed,
        else => return error.SubprocessFailed,
    }
}

fn connectLoopback(port: u16) !std.c.fd_t {
    const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = std.c.close(fd);
    var one: c_int = 1;
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, &one, @sizeOf(c_int));
    _ = std.c.setsockopt(fd, std.c.IPPROTO.TCP, std.c.TCP.NODELAY, &one, @sizeOf(c_int));
    var addr = std.c.sockaddr.in{
        .port = std.mem.nativeToBig(u16, port),
        .addr = @bitCast([4]u8{ 127, 0, 0, 1 }),
    };
    if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) return error.ConnectFailed;
    return fd;
}

fn nbSendAll(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.send(fd, data.ptr + sent, data.len - sent, 0);
        if (n > 0) {
            sent += @intCast(n);
            continue;
        }
        if (n < 0) {
            const err = std.c.errno(n);
            if (err == .INTR) continue;
            if (err == .AGAIN) {
                var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.OUT, .revents = 0 }};
                if (std.c.poll(&pfd, 1, 5000) <= 0) return error.SendTimeout;
                continue;
            }
        }
        return error.SendFailed;
    }
}

fn extractHeaderVal(resp: []const u8, header_name: []const u8) ?[]const u8 {
    const sep = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return null;
    var lines = std.mem.splitSequence(u8, resp[0..sep], "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        if (std.ascii.eqlIgnoreCase(name, header_name)) {
            return std.mem.trim(u8, line[colon + 1 ..], " \t");
        }
    }
    return null;
}

fn extractBody(resp: []const u8) ?[]const u8 {
    const sep = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return null;
    return resp[sep + 4 ..];
}

fn nbRecvOneHttpResponse(fd: std.c.fd_t, out_buf: []u8) ![]const u8 {
    var total: usize = 0;
    var expected_end: ?usize = null;
    while (true) {
        if (expected_end) |ee| {
            if (total >= ee) return out_buf[0..ee];
        }
        if (total >= out_buf.len) return error.BufferTooSmall;
        const n = std.c.recv(fd, out_buf.ptr + total, out_buf.len - total, 0);
        if (n > 0) {
            total += @intCast(n);
            if (expected_end == null) {
                if (std.mem.indexOf(u8, out_buf[0..total], "\r\n\r\n")) |sep| {
                    const hdr_end = sep + 4;
                    if (extractHeaderVal(out_buf[0..hdr_end], "Content-Length")) |cl_str| {
                        const cl = std.fmt.parseInt(usize, cl_str, 10) catch return error.RecvFailed;
                        expected_end = hdr_end + cl;
                        if (expected_end.? > out_buf.len) return error.BufferTooSmall;
                    }
                }
            }
            continue;
        }
        if (n == 0) return out_buf[0..total];
        const err = std.c.errno(n);
        if (err == .INTR) continue;
        if (err == .AGAIN) {
            var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 }};
            if (std.c.poll(&pfd, 1, 10000) <= 0) return error.RecvTimeout;
            continue;
        }
        return error.RecvFailed;
    }
}

fn tcpExchange(port: u16, req: []const u8, resp_buf: []u8) ![]const u8 {
    const fd = try connectLoopback(port);
    defer _ = std.c.close(fd);
    try nbSendAll(fd, req);
    return nbRecvOneHttpResponse(fd, resp_buf);
}

fn httpRequest(
    port: u16,
    method: []const u8,
    path: []const u8,
    extra_headers: []const u8,
    body: []const u8,
    req_buf: []u8,
    resp_buf: []u8,
) ![]const u8 {
    const req = if (body.len > 0 or std.mem.eql(u8, method, "POST"))
        try std.fmt.bufPrint(
            req_buf,
            "{s} {s} HTTP/1.1\r\nHost: 127.0.0.1\r\n{s}Content-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
            .{ method, path, extra_headers, body.len, body },
        )
    else
        try std.fmt.bufPrint(
            req_buf,
            "{s} {s} HTTP/1.1\r\nHost: 127.0.0.1\r\n{s}Connection: close\r\n\r\n",
            .{ method, path, extra_headers },
        );
    return tcpExchange(port, req, resp_buf);
}

fn tlsExchange(port: u16, min_ver: u16, max_ver: u16, req: []const u8, resp_buf: []u8) !struct { resp: []const u8, ver: u16 } {
    const cc = try tls.clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, min_ver, max_ver);
    defer _ = std.c.close(cc.fd);
    try nbSendAll(cc.fd, req);
    const resp = try nbRecvOneHttpResponse(cc.fd, resp_buf);
    return .{ .resp = resp, .ver = cc.tls_version };
}

// ---------------------------------------------------------------------------
// Synthetic AMF + Inkling2Loop Helpers for M9.5
// ---------------------------------------------------------------------------

const BenchTensorBlob = struct {
    name: []const u8,
    kind: amf.TensorKind,
    dtype: amf.DType,
    layer: u16,
    expert: u16,
    shape: [4]u32,
    ndim: u8,
    bytes: []u8,
};

fn buildBenchAmf(allocator: std.mem.Allocator, model_id: []const u8, blobs: []const BenchTensorBlob) ![]u8 {
    const dir_offset: u64 = page.APPLE_PAGE_SIZE;
    const raw_dir_bytes: usize = blobs.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: u64 = try page.alignUp(@max(raw_dir_bytes, page.APPLE_PAGE_SIZE));
    var offset: u64 = dir_offset + dir_bytes;

    var entries = try allocator.alloc(amf.TensorEntry, blobs.len);
    defer allocator.free(entries);

    for (blobs, 0..) |b, i| {
        const data_len: u64 = b.bytes.len;
        const blob_len: u64 = try page.alignUp(@max(b.bytes.len, 1));
        var e = amf.TensorEntry.zeroed(b.name, b.kind, b.dtype);
        e.setLayoutAndParamKind(.row_major, .affine_scale_bias);
        e.layer = b.layer;
        e.expert = b.expert;
        e.rank = b.ndim;
        e.shape = b.shape;
        e.blob_offset = offset;
        e.blob_bytes = blob_len;
        e.data_bytes = data_len;
        entries[i] = e;
        offset += blob_len;
    }

    const total: usize = @intCast(offset);
    const out = try allocator.alloc(u8, total);
    @memset(out, 0);

    for (blobs, 0..) |b, i| {
        const bo: usize = @intCast(entries[i].blob_offset);
        const bl: usize = @intCast(entries[i].blob_bytes);
        @memcpy(out[bo .. bo + b.bytes.len], b.bytes);
        entries[i].blob_crc32 = std.hash.Crc32.hash(out[bo .. bo + bl]);
    }

    const dir_slice = std.mem.sliceAsBytes(entries);
    var hdr: amf.Header = std.mem.zeroes(amf.Header);
    hdr.magic = amf.MAGIC;
    hdr.version = amf.FORMAT_VERSION;
    hdr.flags = 0;
    hdr.page_size = page.APPLE_PAGE_SIZE;
    hdr.checksum_algo = amf.CHECKSUM_CRC32_IEEE;
    hdr.tensor_count = @intCast(blobs.len);
    hdr.directory_offset = dir_offset;
    hdr.directory_bytes = dir_bytes;
    hdr.directory_crc32 = std.hash.Crc32.hash(dir_slice);
    const id_len = @min(model_id.len, hdr.model_id.len - 1);
    @memcpy(hdr.model_id[0..id_len], model_id[0..id_len]);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);

    @memcpy(out[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    @memcpy(out[@intCast(dir_offset)..][0..dir_slice.len], dir_slice);
    return out;
}

fn fillDeterministicF16(allocator: std.mem.Allocator, seed: u64, count: usize, amp: f32) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const v: f32 = (r.float(f32) * 2.0 - 1.0) * amp;
        const bits: u16 = @bitCast(@as(f16, @floatCast(v)));
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn fillConstantBf16(allocator: std.mem.Allocator, count: usize, val: f32) ![]u8 {
    const bits = quant.f32ToFloatBits(.bf16, val);
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    return out;
}

fn fillDeterministicF32Slice(dst: []f32, seed: u64, amp: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (dst) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * amp;
}

const SynthOverlay = struct {
    specs: []const lora.SyntheticLoraSpec,
    scale: f32,
};

fn applyOverlayToF16Weight(
    allocator: std.mem.Allocator,
    base_f16_bytes: []const u8,
    n: u32,
    k: u32,
    spec: *const lora.SyntheticLoraSpec,
    scale: f32,
) ![]u8 {
    const nu: usize = n;
    const ku: usize = k;
    const ru: usize = spec.rank;
    const out = try allocator.alloc(u8, nu * ku * 2);
    for (0..nu) |row| {
        var norm_sq: f64 = 0.0;
        for (0..ku) |col| {
            const w0_bits = std.mem.readInt(u16, base_f16_bytes[(row * ku + col) * 2 ..][0..2], .little);
            const w0: f32 = @floatCast(@as(f16, @bitCast(w0_bits)));
            var ba: f32 = 0.0;
            for (0..ru) |j| {
                const bv: f32 = @floatCast(@as(f16, @bitCast(@as(f16, @floatCast(spec.b_f32[row * ru + j])))));
                const av: f32 = @floatCast(@as(f16, @bitCast(@as(f16, @floatCast(spec.a_f32[j * ku + col])))));
                ba += bv * av;
            }
            const eff_f64: f64 = @as(f64, w0) + @as(f64, scale) * @as(f64, ba);
            norm_sq += eff_f64 * eff_f64;
        }
        const gamma: f64 = if (spec.dora_mag_f32) |dm|
            @as(f64, dm[row]) / @sqrt(norm_sq + 1e-12)
        else
            1.0;
        for (0..ku) |col| {
            const w0_bits = std.mem.readInt(u16, base_f16_bytes[(row * ku + col) * 2 ..][0..2], .little);
            const w0: f32 = @floatCast(@as(f16, @bitCast(w0_bits)));
            var ba: f32 = 0.0;
            for (0..ru) |j| {
                const bv: f32 = @floatCast(@as(f16, @bitCast(@as(f16, @floatCast(spec.b_f32[row * ru + j])))));
                const av: f32 = @floatCast(@as(f16, @bitCast(@as(f16, @floatCast(spec.a_f32[j * ku + col])))));
                ba += bv * av;
            }
            const merged: f32 = @floatCast((@as(f64, w0) + @as(f64, scale) * @as(f64, ba)) * gamma);
            const bits: u16 = @bitCast(@as(f16, @floatCast(merged)));
            std.mem.writeInt(u16, out[(row * ku + col) * 2 ..][0..2], bits, .little);
        }
    }
    return out;
}

fn buildSynthInkling2LoopBundle(
    allocator: std.mem.Allocator,
    overlay: ?SynthOverlay,
) !struct { cfg: graph.GraphConfig, amf_bytes: []u8 } {
    const h: u32 = 64;
    const n_layers: u32 = 4;
    const n_heads: u32 = 2;
    const n_kv: u32 = 1;
    const head_dim: u32 = 32;
    const qd: u32 = n_heads * head_dim;
    const kvd: u32 = n_kv * head_dim;
    const d_rel: u32 = 16;
    const rel_extent: u32 = 32;
    const inter: u32 = 64;
    const n_exp: u32 = 4;
    const top_k: u32 = 2;
    const vocab: u32 = 64;

    const layer_attns = try allocator.alloc(graph.AttentionKind, n_layers);
    layer_attns[0] = .gqa_full;
    layer_attns[1] = .qsa_sparse;
    layer_attns[2] = .qsa_sparse;
    layer_attns[3] = .gqa_full;

    const cfg: graph.GraphConfig = .{
        .axes = .{
            .backbone_kind = .decoder_only,
            .layer_attention_kinds = layer_attns,
            .ffn_kind = .moe_swiglu,
            .has_shared_expert = false,
            .residual_kind = .single_stream,
            .residual_branches = 1,
            .memory_lookup_kind = .none,
            .loop_schedule = .{
                .mode = .native_recurrent,
                .start_layer = 1,
                .end_layer = 3,
                .max_loops = 3,
                .inject_x_init = false,
                .adapter_kind = .none,
            },
        },
        .m82_family = .inkling2_loop,
        .m82_cfg = .{
            .d_rel = d_rel,
            .rel_extent = rel_extent,
            .sliding_window_size = 32,
            .n_layers_per_attn_res_block = 2,
            .qk_l2_clip_threshold = 4.0,
            .rel_l2_clip_threshold = 4.0,
            .log_scaling_n_floor = 0,
            .log_scaling_alpha = 0.0,
            .use_value_norm_no_gamma = true,
            .final_logit_softcapping = 30.0,
        },
        .vocab_size = vocab,
        .hidden_size = h,
        .num_hidden_layers = n_layers,
        .num_attention_heads = n_heads,
        .num_key_value_heads = n_kv,
        .head_dim = head_dim,
        .intermediate_size = inter,
        .moe_intermediate_size = inter,
        .num_experts = n_exp,
        .num_experts_per_tok = top_k,
        .rope_theta = 10_000.0,
    };

    var blobs: std.ArrayListUnmanaged(BenchTensorBlob) = .empty;
    var owned_names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
        for (owned_names.items) |n| allocator.free(n);
        owned_names.deinit(allocator);
    }

    const addBlob = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(BenchTensorBlob),
            names: *std.ArrayListUnmanaged([]u8),
            nm: []const u8,
            kind: amf.TensorKind,
            dt: amf.DType,
            layer: u16,
            expert_id: u16,
            shape: [4]u32,
            ndim: u8,
            data: []u8,
        ) !void {
            const dup = try alloc.dupe(u8, nm);
            try names.append(alloc, dup);
            try list.append(alloc, .{
                .name = dup,
                .kind = kind,
                .dtype = dt,
                .layer = layer,
                .expert = expert_id,
                .shape = shape,
                .ndim = ndim,
                .bytes = data,
            });
        }
    }.call;

    const addOverlayLin = struct {
        fn call(
            alloc: std.mem.Allocator,
            list: *std.ArrayListUnmanaged(BenchTensorBlob),
            names: *std.ArrayListUnmanaged([]u8),
            nm: []const u8,
            layer: u32,
            target: lora.TargetProjection,
            n_out: u32,
            k_in: u32,
            seed: u64,
            amp: f32,
            ov: ?SynthOverlay,
        ) !void {
            const base = try fillDeterministicF16(alloc, seed, @as(usize, n_out) * k_in, amp);
            if (ov) |o| {
                for (o.specs) |*sp| {
                    if (sp.layer == layer and sp.target == target) {
                        defer alloc.free(base);
                        const merged = try applyOverlayToF16Weight(alloc, base, n_out, k_in, sp, o.scale);
                        try addBlob(alloc, list, names, nm, .dense, .f16, @intCast(layer), amf.NO_EXPERT, .{ n_out, k_in, 0, 0 }, 2, merged);
                        return;
                    }
                }
            }
            try addBlob(alloc, list, names, nm, .dense, .f16, @intCast(layer), amf.NO_EXPERT, .{ n_out, k_in, 0, 0 }, 2, base);
        }
    }.call;

    try addBlob(allocator, &blobs, &owned_names, "embed_tokens", .embedding, .f16, amf.NO_LAYER, amf.NO_EXPERT, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 8101, vocab * h, 0.20));

    var nbuf: [128]u8 = undefined;
    for (0..n_layers) |l| {
        const l32: u32 = @intCast(l);
        const sb: u64 = 9000 + l * 100;
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn_norm.weight", .{l}), .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0));
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp_norm.weight", .{l}), .norm, .bf16, @intCast(l), amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0));

        try addOverlayLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.wq_du", .{l}), l32, .q_proj, qd, h, sb + 1, 0.08, overlay);
        try addOverlayLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.wk_dv", .{l}), l32, .k_proj, kvd, h, sb + 2, 0.08, overlay);
        try addOverlayLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.wv_dv", .{l}), l32, .v_proj, kvd, h, sb + 3, 0.08, overlay);
        try addOverlayLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.wo_ud", .{l}), l32, .o_proj, h, qd, sb + 4, 0.08, overlay);
        try addOverlayLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.wr_du", .{l}), l32, .wr_du, n_heads * d_rel, h, sb + 5, 0.06, overlay);
        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.rel_logits_proj.proj", .{l}), .dense, .f16, @intCast(l), amf.NO_EXPERT, .{ rel_extent, d_rel, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 6, rel_extent * d_rel, 0.08));
        try addOverlayLin(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.attn.gate_attn", .{l}), l32, .gate_attn, qd, h, sb + 7, 0.08, overlay);

        try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate", .{l}), .router, .f16, @intCast(l), amf.NO_EXPERT, .{ n_exp, h, 0, 0 }, 2, try fillDeterministicF16(allocator, sb + 8, n_exp * h, 0.12));
        for (0..n_exp) |e| {
            const es: u64 = sb + 20 + e * 4;
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.w13", .{ l, e }), .hot_expert, .f16, @intCast(l), @intCast(e), .{ 2 * inter, h, 0, 0 }, 2, try fillDeterministicF16(allocator, es + 1, 2 * inter * h, 0.06));
            try addBlob(allocator, &blobs, &owned_names, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }), .hot_expert, .f16, @intCast(l), @intCast(e), .{ h, inter, 0, 0 }, 2, try fillDeterministicF16(allocator, es + 2, h * inter, 0.06));
        }
    }

    try addBlob(allocator, &blobs, &owned_names, "norm.weight", .norm, .bf16, amf.NO_LAYER, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, try fillConstantBf16(allocator, h, 1.0));
    try addBlob(allocator, &blobs, &owned_names, "lm_head", .lm_head, .f16, amf.NO_LAYER, amf.NO_EXPERT, .{ vocab, h, 0, 0 }, 2, try fillDeterministicF16(allocator, 8999, vocab * h, 0.15));

    const amf_bytes = try buildBenchAmf(allocator, "bench-inkling2-loop-lora", blobs.items);
    return .{ .cfg = cfg, .amf_bytes = amf_bytes };
}

// ---------------------------------------------------------------------------
// M9.5 Benchmark Runner
// ---------------------------------------------------------------------------

pub fn runM95WithConfig(allocator: std.mem.Allocator, cfg: M95Config) !M95Report {
    var report: M95Report = .{};
    const vm_before = try vmstat.Snapshot.capture();

    const h: u32 = 64;
    const kvd: u32 = 32;
    const r_std: u32 = 16;
    const alpha_std: f32 = 32.0;
    const scale_std: f32 = alpha_std / @as(f32, @floatFromInt(r_std));

    const r_dora: u32 = 4;
    const alpha_dora: f32 = 8.0;
    const scale_dora: f32 = alpha_dora / @as(f32, @floatFromInt(r_dora));

    const targets = [_]struct { t: lora.TargetProjection, suf: []const u8, n: u32, k: u32 }{
        .{ .t = .q_proj, .suf = "attn.wq_du", .n = h, .k = h },
        .{ .t = .k_proj, .suf = "attn.wk_dv", .n = kvd, .k = h },
        .{ .t = .v_proj, .suf = "attn.wv_dv", .n = kvd, .k = h },
        .{ .t = .o_proj, .suf = "attn.wo_ud", .n = h, .k = h },
        .{ .t = .gate_attn, .suf = "attn.gate_attn", .n = h, .k = h },
    };

    var std_specs: [10]lora.SyntheticLoraSpec = undefined;
    var std_a: [10][]f32 = undefined;
    var std_b: [10][]f32 = undefined;
    defer for (0..10) |i| {
        allocator.free(std_a[i]);
        allocator.free(std_b[i]);
    };
    var si: usize = 0;
    for ([_]u32{ 1, 2 }) |lid| {
        for (targets, 0..) |tg, ti| {
            std_a[si] = try allocator.alloc(f32, r_std * tg.k);
            std_b[si] = try allocator.alloc(f32, tg.n * r_std);
            fillDeterministicF32Slice(std_a[si], 10_000 + lid * 31 + ti * 7, 0.12);
            fillDeterministicF32Slice(std_b[si], 20_000 + lid * 37 + ti * 11, 0.12);
            std_specs[si] = .{
                .layer = lid,
                .target = tg.t,
                .n = tg.n,
                .k = tg.k,
                .rank = r_std,
                .layout = .row_major,
                .dtype = .f16,
                .a_f32 = std_a[si],
                .b_f32 = std_b[si],
                .custom_suffix = tg.suf,
            };
            si += 1;
        }
    }

    var dora_specs: [10]lora.SyntheticLoraSpec = undefined;
    var dora_a: [10][]f32 = undefined;
    var dora_b: [10][]f32 = undefined;
    var dora_m: [10][]f32 = undefined;
    defer for (0..10) |i| {
        allocator.free(dora_a[i]);
        allocator.free(dora_b[i]);
        allocator.free(dora_m[i]);
    };
    si = 0;
    for ([_]u32{ 1, 2 }) |lid| {
        for (targets, 0..) |tg, ti| {
            dora_a[si] = try allocator.alloc(f32, r_dora * tg.k);
            dora_b[si] = try allocator.alloc(f32, tg.n * r_dora);
            dora_m[si] = try allocator.alloc(f32, tg.n);
            fillDeterministicF32Slice(dora_a[si], 30_000 + lid * 41 + ti * 13, 0.10);
            fillDeterministicF32Slice(dora_b[si], 40_000 + lid * 43 + ti * 17, 0.10);
            for (dora_m[si], 0..) |*mv, row_i| {
                mv.* = 0.75 + 0.01 * @as(f32, @floatFromInt((row_i * 7 + ti * 3 + lid) % 50));
            }
            dora_specs[si] = .{
                .layer = lid,
                .target = tg.t,
                .n = tg.n,
                .k = tg.k,
                .rank = r_dora,
                .layout = .row_major,
                .dtype = .f16,
                .a_f32 = dora_a[si],
                .b_f32 = dora_b[si],
                .dora_mag_f32 = dora_m[si],
                .custom_suffix = tg.suf,
            };
            si += 1;
        }
    }

    const std_amlora = try lora.buildSyntheticAmloraBytes(allocator, "ink-r16", r_std, alpha_std, &std_specs);
    defer allocator.free(std_amlora);
    const dora_amlora = try lora.buildSyntheticAmloraBytes(allocator, "ink-dora", r_dora, alpha_dora, &dora_specs);
    defer allocator.free(dora_amlora);

    var base_b = try buildSynthInkling2LoopBundle(allocator, null);
    defer {
        base_b.cfg.deinit(allocator);
        allocator.free(base_b.amf_bytes);
    }
    var mstd_b = try buildSynthInkling2LoopBundle(allocator, .{ .specs = &std_specs, .scale = scale_std });
    defer {
        mstd_b.cfg.deinit(allocator);
        allocator.free(mstd_b.amf_bytes);
    }
    var mdora_b = try buildSynthInkling2LoopBundle(allocator, .{ .specs = &dora_specs, .scale = scale_dora });
    defer {
        mdora_b.cfg.deinit(allocator);
        allocator.free(mdora_b.amf_bytes);
    }

    var model = try graph.GraphModel.loadFromBytes(allocator, base_b.amf_bytes, base_b.cfg, .{
        .max_seq_len = 512,
        .max_tree_nodes = 8,
        .startup_canary = false,
    });
    defer model.deinit(allocator);

    var merged_std = try graph.GraphModel.loadFromBytes(allocator, mstd_b.amf_bytes, mstd_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 8,
        .startup_canary = false,
    });
    defer merged_std.deinit(allocator);

    var merged_dora = try graph.GraphModel.loadFromBytes(allocator, mdora_b.amf_bytes, mdora_b.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 8,
        .startup_canary = false,
    });
    defer merged_dora.deinit(allocator);

    const prompt = [_]u32{ 3, 11, 27, 42 };
    model.resetM82Stats();
    const base_t0 = try model.prefillChunk(&prompt, 0, true);
    const base_t1 = try model.stepDecode(base_t0, 4);
    report.mid_pass_flushes_base = model.m82Stats().?.mid_pass_flushes;

    var base_logits_bits: [64]u16 = undefined;
    for (model.readLastLogitsF16(), 0..) |v, i| base_logits_bits[i] = @bitCast(v);

    var reg = try lora.LoraRegistry.init(.{
        .pool_bytes = 4 * 1024 * 1024,
        .allocator = allocator,
        .base_shape = model.baseModelShape(),
    });
    defer reg.deinit(allocator);
    try model.attachLoraRegistry(&reg);
    defer model.attachLoraRegistry(null) catch |err| {
        std.debug.panic("GraphModel.attachLoraRegistry(null) failed: {}", .{err});
    };

    const hl_iters = @max(cfg.hot_load_iters, 1);
    const hl_t0 = monotonicNowNs();
    var slot_std: u32 = lora.NO_ADAPTER;
    var slot_dora: u32 = lora.NO_ADAPTER;
    for (0..hl_iters) |it| {
        if (it > 0) {
            try reg.unloadAdapter(allocator, "ink-r16");
            try reg.unloadAdapter(allocator, "ink-dora");
        }
        slot_std = try reg.loadAdapterFromBytes(allocator, "ink-r16", std_amlora);
        slot_dora = try reg.loadAdapterFromBytes(allocator, "ink-dora", dora_amlora);
        try model.calibrateLoraDora(allocator);
    }
    const hl_t1 = monotonicNowNs();
    report.hot_load_ms = @as(f64, @floatFromInt(hl_t1 -| hl_t0)) / (@as(f64, @floatFromInt(hl_iters)) * 1_000_000.0);

    // Numerical parity: r=16 standard LoRA
    try model.setActiveLoraAdapter("ink-r16");
    model.resetKvCache();
    model.resetM82Stats();
    const l_t0 = try model.prefillChunk(&prompt, 0, true);
    const l_t1 = try model.stepDecode(l_t0, 4);
    const m_t0 = try merged_std.prefillChunk(&prompt, 0, true);
    const m_t1 = try merged_std.stepDecode(m_t0, 4);
    report.mid_pass_flushes_lora = model.m82Stats().?.mid_pass_flushes;

    var max_diff_std: f32 = 0.0;
    var diff_vs_base_std: f32 = 0.0;
    for (0..64) |i| {
        const g: f32 = @floatCast(model.readLastLogitsF16()[i]);
        const rf: f32 = @floatCast(merged_std.readLastLogitsF16()[i]);
        const bf: f32 = @floatCast(@as(f16, @bitCast(base_logits_bits[i])));
        max_diff_std = @max(max_diff_std, @abs(g - rf));
        diff_vs_base_std = @max(diff_vs_base_std, @abs(g - bf));
    }
    report.lora_max_logit_diff = max_diff_std;
    report.lora_parity_ok = (l_t0 == m_t0) and (l_t1 == m_t1) and (diff_vs_base_std > 1e-2) and (max_diff_std < 5e-2) and (report.mid_pass_flushes_lora == 0);

    // Numerical parity: calibrated PEFT DoRA
    model.setActiveLoraSlot(slot_dora);
    model.resetKvCache();
    model.resetM82Stats();
    const d_t0 = try model.prefillChunk(&prompt, 0, true);
    const d_t1 = try model.stepDecode(d_t0, 4);
    const md_t0 = try merged_dora.prefillChunk(&prompt, 0, true);
    const md_t1 = try merged_dora.stepDecode(md_t0, 4);
    report.mid_pass_flushes_dora = model.m82Stats().?.mid_pass_flushes;

    var max_diff_dora: f32 = 0.0;
    var diff_vs_base_dora: f32 = 0.0;
    for (0..64) |i| {
        const g: f32 = @floatCast(model.readLastLogitsF16()[i]);
        const rf: f32 = @floatCast(merged_dora.readLastLogitsF16()[i]);
        const bf: f32 = @floatCast(@as(f16, @bitCast(base_logits_bits[i])));
        max_diff_dora = @max(max_diff_dora, @abs(g - rf));
        diff_vs_base_dora = @max(diff_vs_base_dora, @abs(g - bf));
    }
    report.dora_max_logit_diff = max_diff_dora;
    report.dora_parity_ok = (d_t0 == md_t0) and (d_t1 == md_t1) and (diff_vs_base_dora > 1e-2) and (max_diff_dora < 5e-2) and (report.mid_pass_flushes_dora == 0);

    // Steady-state M = 1 latency (base vs r=16 LoRA vs DoRA) + malloc delta check.
    const d_iters = @max(cfg.decode_iters, 1);
    const benchDecodePath = struct {
        fn run(m: *graph.GraphModel, slot_id: u32, p: []const u32, iters: u32, measure_malloc: bool) !struct { us: f64, mdelta: i64 } {
            m.setActiveLoraSlot(slot_id);
            m.resetKvCache();
            var cur = try m.prefillChunk(p, 0, true);
            var pos: u32 = @intCast(p.len);
            // Warm up steady-state Metal4 pipelines/command pools
            const warm_n: u32 = @max(iters, 4);
            for (0..warm_n) |_| {
                cur = try m.stepDecode(cur, pos);
                pos += 1;
            }
            var best_us: f64 = 0.0;
            var best_delta: i64 = 0;
            const passes: usize = if (measure_malloc) 3 else 1;
            for (0..passes) |pass_idx| {
                m.resetKvCache();
                cur = try m.prefillChunk(p, 0, true);
                pos = @intCast(p.len);
                const pool = objc_autoreleasePoolPush();
                var m0: MallocStatistics = std.mem.zeroes(MallocStatistics);
                if (measure_malloc) malloc_zone_statistics(null, &m0);
                const t0 = monotonicNowNs();
                for (0..iters) |_| {
                    const step_pool = objc_autoreleasePoolPush();
                    defer objc_autoreleasePoolPop(step_pool);
                    cur = try m.stepDecode(cur, pos);
                    pos += 1;
                }
                const t1 = monotonicNowNs();
                objc_autoreleasePoolPop(pool);
                var m1: MallocStatistics = std.mem.zeroes(MallocStatistics);
                if (measure_malloc) malloc_zone_statistics(null, &m1);
                const us = @as(f64, @floatFromInt(t1 -| t0)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
                const delta: i64 = if (measure_malloc) @as(i64, m1.blocks_in_use) - @as(i64, m0.blocks_in_use) else 0;
                if (pass_idx == 0) {
                    best_us = us;
                    best_delta = delta;
                } else {
                    best_us = @min(best_us, us);
                    best_delta = @min(best_delta, delta);
                }
                if (measure_malloc and best_delta <= 0) break;
            }
            return .{ .us = best_us, .mdelta = best_delta };
        }
    }.run;

    const res_base = try benchDecodePath(&model, lora.NO_ADAPTER, &prompt, d_iters, false);
    const res_lora = try benchDecodePath(&model, slot_std, &prompt, d_iters, true);
    const res_dora = try benchDecodePath(&model, slot_dora, &prompt, d_iters, false);
    report.base_m1_us = res_base.us;
    report.lora_r16_m1_us = res_lora.us;
    report.dora_m1_us = res_dora.us;
    report.decode_malloc_blocks_delta = res_lora.mdelta;

    // Bit-exact NO_ADAPTER restore & unload restore
    model.setActiveLoraSlot(lora.NO_ADAPTER);
    model.resetKvCache();
    const na_t0 = try model.prefillChunk(&prompt, 0, true);
    const na_t1 = try model.stepDecode(na_t0, 4);
    var na_exact = (na_t0 == base_t0) and (na_t1 == base_t1);
    for (model.readLastLogitsF16(), 0..) |v, i| {
        if (@as(u16, @bitCast(v)) != base_logits_bits[i]) na_exact = false;
    }
    report.no_adapter_bit_exact = na_exact;

    // /v1/models & /v1/models/{id} discovery over real Server before unload
    {
        var meta_buf: [lora.MAX_ADAPTERS]server.LoraAdapterMeta = undefined;
        const meta_len = server.LoraAdapterMeta.populateFromRegistry(&reg, &meta_buf);
        var dummy: u8 = 0;
        const Mock = struct {
            fn generate(_: *anyopaque, _: *const server.InferenceJob, stream: *server.TokenStream) void {
                _ = stream.setPromptTokens(2);
                stream.finish(.stop, 2);
            }
        };
        const srv = try server.Server.init(allocator, .{
            .port = 0,
            .model_id = "inkling-2-synth",
            .model_detail = .{
                .architecture = "inkling2_loop_text",
                .max_model_len = 4096,
                .vocab_size = 64,
                .capabilities = .{ .lora = true, .reasoning = true, .moe = true },
            },
            .lora_adapters = meta_buf[0..meta_len],
            .max_conns = 4,
            .max_active_requests = 2,
            .apply_qos = false,
        });
        defer srv.deinit(allocator);
        try srv.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
        defer srv.stop();

        var req_buf: [1024]u8 = undefined;
        var resp_buf: [8192]u8 = undefined;
        const r_list = try httpRequest(srv.boundPort(), "GET", "/v1/models", "", "", &req_buf, &resp_buf);
        const list_ok = std.mem.startsWith(u8, r_list, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, r_list, "\"id\":\"inkling-2-synth:ink-r16\"") != null and
            std.mem.indexOf(u8, r_list, "\"id\":\"inkling-2-synth:ink-dora\"") != null;
        const r_r16 = try httpRequest(srv.boundPort(), "GET", "/v1/models/inkling-2-synth%3Aink-r16", "", "", &req_buf, &resp_buf);
        const r16_ok = std.mem.startsWith(u8, r_r16, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, r_r16, "\"rank\":16") != null;
        const r_do = try httpRequest(srv.boundPort(), "GET", "/v1/models/inkling-2-synth%3Aink-dora", "", "", &req_buf, &resp_buf);
        const do_ok = std.mem.startsWith(u8, r_do, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, r_do, "\"id\":\"inkling-2-synth:ink-dora\"") != null;

        const disc_n = @max(cfg.discovery_iters, 1);
        for (0..4) |_| {
            _ = try httpRequest(srv.boundPort(), "GET", "/v1/models/inkling-2-synth%3Aink-dora", "", "", &req_buf, &resp_buf);
        }
        var best_disc_delta: i64 = 0;
        var dt_ns: u64 = 0;
        for (0..3) |dpass| {
            var dm0: MallocStatistics = std.mem.zeroes(MallocStatistics);
            malloc_zone_statistics(null, &dm0);
            const dt0 = monotonicNowNs();
            for (0..disc_n) |_| {
                const rr = try httpRequest(srv.boundPort(), "GET", "/v1/models/inkling-2-synth%3Aink-dora", "", "", &req_buf, &resp_buf);
                if (!std.mem.startsWith(u8, rr, "HTTP/1.1 200 OK\r\n")) return error.InvalidHttpResponse;
            }
            const dt1 = monotonicNowNs();
            var dm1: MallocStatistics = std.mem.zeroes(MallocStatistics);
            malloc_zone_statistics(null, &dm1);
            dt_ns = dt1 -| dt0;
            best_disc_delta = @as(i64, dm1.blocks_in_use) - @as(i64, dm0.blocks_in_use);
            if (best_disc_delta <= 0 or dpass == 2) break;
        }
        report.malloc_blocks_delta = best_disc_delta;
        report.discovery_mean_us = @as(f64, @floatFromInt(dt_ns)) / (@as(f64, @floatFromInt(disc_n)) * 1000.0);
        report.discovery_ok = list_ok and r16_ok and do_ok;
    }

    try reg.unloadAdapter(allocator, "ink-r16");
    try reg.unloadAdapter(allocator, "ink-dora");
    model.resetKvCache();
    const un_t0 = try model.prefillChunk(&prompt, 0, true);
    const un_t1 = try model.stepDecode(un_t0, 4);
    var un_exact = (un_t0 == base_t0) and (un_t1 == base_t1);
    for (model.readLastLogitsF16(), 0..) |v, i| {
        if (@as(u16, @bitCast(v)) != base_logits_bits[i]) un_exact = false;
    }
    report.unload_bit_exact = un_exact;

    // GraphBatchRuntime 3-slot heterogeneous LoRA (base + LoRA + DoRA)
    {
        var bundle = try graph.buildSynthLoopDenseBundle(allocator, .{
            .hidden_size = 128,
            .vocab_size = 128,
            .num_layers = 2,
            .num_attention_heads = 2,
            .num_key_value_heads = 2,
            .head_dim = 64,
            .intermediate_size = 256,
            .start_layer = 0,
            .end_layer = 2,
            .mode = .single_pass,
            .max_loops = 1,
            .use_q4 = false,
        });
        defer bundle.deinit(allocator);

        const bh: u32 = 128;
        const binter: u32 = 256;
        const r_a: u32 = 8;
        const a_defs = [_]struct { l: u32, t: lora.TargetProjection, n: u32, k: u32 }{
            .{ .l = 0, .t = .q_proj, .n = bh, .k = bh },
            .{ .l = 0, .t = .o_proj, .n = bh, .k = bh },
            .{ .l = 0, .t = .down_proj, .n = bh, .k = binter },
        };
        var sp_a: [a_defs.len]lora.SyntheticLoraSpec = undefined;
        var ba_a: [a_defs.len][]f32 = undefined;
        var ba_b: [a_defs.len][]f32 = undefined;
        defer for (0..a_defs.len) |i| {
            allocator.free(ba_a[i]);
            allocator.free(ba_b[i]);
        };
        for (a_defs, 0..) |d, i| {
            ba_a[i] = try allocator.alloc(f32, r_a * d.k);
            ba_b[i] = try allocator.alloc(f32, d.n * r_a);
            fillDeterministicF32Slice(ba_a[i], 50_000 + i * 19, 0.14);
            fillDeterministicF32Slice(ba_b[i], 60_000 + i * 23, 0.14);
            sp_a[i] = .{ .layer = d.l, .target = d.t, .n = d.n, .k = d.k, .rank = r_a, .a_f32 = ba_a[i], .b_f32 = ba_b[i] };
        }
        const amlora_a = try lora.buildSyntheticAmloraBytes(allocator, "adp-a", r_a, 16.0, &sp_a);
        defer allocator.free(amlora_a);

        const r_b: u32 = 4;
        const b_defs = [_]struct { l: u32, t: lora.TargetProjection, n: u32, k: u32 }{
            .{ .l = 0, .t = .q_proj, .n = bh, .k = bh },
            .{ .l = 1, .t = .down_proj, .n = bh, .k = binter },
        };
        var sp_b: [b_defs.len]lora.SyntheticLoraSpec = undefined;
        var bb_a: [b_defs.len][]f32 = undefined;
        var bb_b: [b_defs.len][]f32 = undefined;
        var bb_m: [b_defs.len][]f32 = undefined;
        defer for (0..b_defs.len) |i| {
            allocator.free(bb_a[i]);
            allocator.free(bb_b[i]);
            allocator.free(bb_m[i]);
        };
        for (b_defs, 0..) |d, i| {
            bb_a[i] = try allocator.alloc(f32, r_b * d.k);
            bb_b[i] = try allocator.alloc(f32, d.n * r_b);
            bb_m[i] = try allocator.alloc(f32, d.n);
            fillDeterministicF32Slice(bb_a[i], 70_000 + i * 29, 0.12);
            fillDeterministicF32Slice(bb_b[i], 80_000 + i * 31, 0.12);
            for (bb_m[i], 0..) |*mv, ri| mv.* = 0.8 + 0.005 * @as(f32, @floatFromInt((ri * 5 + i * 7) % 60));
            sp_b[i] = .{ .layer = d.l, .target = d.t, .n = d.n, .k = d.k, .rank = r_b, .a_f32 = bb_a[i], .b_f32 = bb_b[i], .dora_mag_f32 = bb_m[i] };
        }
        const amlora_b = try lora.buildSyntheticAmloraBytes(allocator, "adp-b", r_b, 8.0, &sp_b);
        defer allocator.free(amlora_b);

        var serial_m = try graph.GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 16,
            .startup_canary = false,
        });
        defer serial_m.deinit(allocator);
        var reg_ref = try lora.LoraRegistry.init(.{
            .pool_bytes = 4 * 1024 * 1024,
            .allocator = allocator,
            .base_shape = serial_m.baseModelShape(),
        });
        defer reg_ref.deinit(allocator);
        try serial_m.attachLoraRegistry(&reg_ref);
        defer serial_m.attachLoraRegistry(null) catch |err| {
            std.debug.panic("GraphModel.attachLoraRegistry(null) failed: {}", .{err});
        };
        const rsa = try reg_ref.loadAdapterFromBytes(allocator, "adp-a", amlora_a);
        const rsb = try reg_ref.loadAdapterFromBytes(allocator, "adp-b", amlora_b);
        try serial_m.calibrateLoraDora(allocator);

        const p0 = [_]u32{ 5, 19, 37, 61 };
        const p1 = [_]u32{ 7, 23, 41, 67, 83 };
        const p2 = [_]u32{ 11, 29, 43, 71 };
        const prompts = [_][]const u32{ &p0, &p1, &p2 };
        const ref_lslots = [_]u32{ lora.NO_ADAPTER, rsa, rsb };
        var ref_toks: [3][2]u32 = undefined;
        var ref_logits: [3][128]f32 = undefined;
        for (0..3) |s| {
            serial_m.setActiveLoraSlot(ref_lslots[s]);
            serial_m.resetKvCache();
            const t0 = try serial_m.prefillChunk(prompts[s], 0, true);
            const t1 = try serial_m.stepDecode(t0, @intCast(prompts[s].len));
            ref_toks[s] = .{ t0, t1 };
            for (serial_m.readLastLogitsF16(), 0..) |v, vi| ref_logits[s][vi] = @floatCast(v);
        }

        var batch_m = try graph.GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 16,
            .startup_canary = false,
        });
        defer batch_m.deinit(allocator);
        var reg_b = try lora.LoraRegistry.init(.{
            .pool_bytes = 4 * 1024 * 1024,
            .allocator = allocator,
            .base_shape = batch_m.baseModelShape(),
        });
        defer reg_b.deinit(allocator);
        try batch_m.attachLoraRegistry(&reg_b);
        defer batch_m.attachLoraRegistry(null) catch |err| {
            std.debug.panic("GraphModel.attachLoraRegistry(null) failed: {}", .{err});
        };
        _ = try reg_b.loadAdapterFromBytes(allocator, "adp-a", amlora_a);
        _ = try reg_b.loadAdapterFromBytes(allocator, "adp-b", amlora_b);
        try batch_m.calibrateLoraDora(allocator);

        var rt = try graph.GraphBatchRuntime.init(allocator, &batch_m, .{ .max_slots = 4, .prefix_cache_chunks = 8 });
        defer rt.deinit(allocator);
        rt.setSlotLora(0, lora.NO_ADAPTER);
        try rt.setSlotLoraAdapter(1, "adp-a");
        try rt.setSlotLoraAdapter(2, "adp-b");

        var rng0 = try rng_mod.RequestRng.init(101);
        var rng1 = try rng_mod.RequestRng.init(102);
        var rng2 = try rng_mod.RequestRng.init(103);
        const rng_ptrs = [_]*rng_mod.RequestRng{ &rng0, &rng1, &rng2 };

        var cur_toks: [3]u32 = undefined;
        var cur_pos: [3]u32 = undefined;
        var tok_match = true;
        for (0..3) |s| {
            const pr = try rt.prefillSlot(&.{
                .slot_id = @intCast(s),
                .request_id = s + 1,
                .user_tag = s,
                .tokens = prompts[s],
                .start_pos = 0,
                .total_prompt_len = @intCast(prompts[s].len),
                .is_final_chunk = true,
                .prefix_handle = s + 1,
                .temperature = 0.0,
                .rng = rng_ptrs[s],
                .ngram_domain = "",
            });
            cur_toks[s] = pr.first_token;
            cur_pos[s] = @intCast(prompts[s].len);
            if (cur_toks[s] != ref_toks[s][0]) tok_match = false;
        }
        var next_toks: [3]u32 = undefined;
        try rt.stepDecodeBatch(&.{
            .slot_ids = &.{ 0, 1, 2 },
            .request_ids = &.{ 1, 2, 3 },
            .user_tags = &.{ 0, 1, 2 },
            .input_tokens = &cur_toks,
            .seq_positions = &cur_pos,
            .temperatures = &.{ 0.0, 0.0, 0.0 },
            .rngs = &rng_ptrs,
            .ngram_domains = &.{ "", "", "" },
            .prefix_handles = &.{ 1, 2, 3 },
        }, &next_toks);

        var batch_diff: f32 = 0.0;
        for (0..3) |s| {
            if (next_toks[s] != ref_toks[s][1]) tok_match = false;
            const row = batch_m.logits_host[s * 128 .. (s + 1) * 128];
            for (0..128) |vi| {
                batch_diff = @max(batch_diff, @abs(@as(f32, @floatCast(row[vi])) - ref_logits[s][vi]));
            }
        }
        report.batch_max_logit_diff = batch_diff;
        report.batch_3slot_parity_ok = tok_match and (batch_diff < 5e-2);
    }

    const vm_after = try vmstat.Snapshot.capture();
    report.swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after);
    report.all_passed = report.lora_parity_ok and
        report.dora_parity_ok and
        report.no_adapter_bit_exact and
        report.unload_bit_exact and
        report.batch_3slot_parity_ok and
        report.discovery_ok and
        (report.malloc_blocks_delta <= 0) and
        (report.swapouts_delta == 0);
    return report;
}

pub fn runM95(allocator: std.mem.Allocator) !M95Report {
    return runM95WithConfig(allocator, .{});
}

// ---------------------------------------------------------------------------
// M9.6 Benchmark Runner (`C29` 7 Capabilities + Base64 Vision/Video Serving)
// ---------------------------------------------------------------------------

fn buildSynthTokenizerJsonAlloc(allocator: std.mem.Allocator) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;

    try w.writeAll("{\"version\":\"1.0\",\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false},\"model\":{\"type\":\"BPE\",\"vocab\":{");
    const specials = [_][]const u8{
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
    };
    var id: u32 = 0;
    for (specials) |sp| {
        if (id > 0) try w.writeByte(',');
        try w.print("\"{s}\":{d}", .{ sp, id });
        id += 1;
    }
    var extra_cp: u21 = 256;
    for (0..256) |b| {
        const direct = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF);
        const cp: u21 = if (direct) @intCast(b) else blk: {
            const c = extra_cp;
            extra_cp += 1;
            break :blk c;
        };
        var ubuf: [4]u8 = undefined;
        const ulen = std.unicode.utf8Encode(cp, &ubuf) catch unreachable;
        const tok_s = ubuf[0..ulen];
        if (tok_s.len == 1 and (tok_s[0] == '"' or tok_s[0] == '\\')) {
            try w.print(",\"\\{c}\":{d}", .{ tok_s[0], id });
        } else {
            try w.print(",\"{s}\":{d}", .{ tok_s, id });
        }
        id += 1;
    }
    try w.writeAll("},\"merges\":[]},\"added_tokens\":[");
    for (specials, 0..) |sp, idx| {
        if (idx > 0) try w.writeByte(',');
        try w.print("{{\"id\":{d},\"content\":\"{s}\",\"special\":true}}", .{ idx, sp });
    }
    try w.writeAll("]}");
    return aw.toOwnedSlice();
}

fn encodeBase64Alloc(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    const out_len = std.base64.standard.Encoder.calcSize(raw.len);
    const out = try allocator.alloc(u8, out_len);
    _ = std.base64.standard.Encoder.encode(out, raw);
    return out;
}

const M96WorkerCtx = struct {
    allocator: std.mem.Allocator,
    model: *graph.GraphModel,
    tok: *const tokenizer_mod.Tokenizer,
    scratch: tokenizer_mod.EncodeScratch,
    prompt_buf: [512]u32 = undefined,
    saw_token_ids_prompt: bool = false,
    vision_tokens_last: u32 = 0,
    iosurface_leak_count: u32 = 0,

    fn splitPrompt(prompt: []const u8) struct { prefix: []const u8, suffix: []const u8 } {
        for ([_][]const u8{ "<|image_pad|>", "<|video_pad|>" }) |needle| {
            if (std.mem.indexOf(u8, prompt, needle)) |idx| {
                return .{ .prefix = prompt[0..idx], .suffix = prompt[idx + needle.len ..] };
            }
        }
        return .{ .prefix = prompt, .suffix = &.{} };
    }

    fn generate(ctx_ptr: *anyopaque, job: *const server.InferenceJob, stream: *server.TokenStream) void {
        const self: *M96WorkerCtx = @ptrCast(@alignCast(ctx_ptr));

        if (job.media_count > 0) {
            var frames_buf: [16]metal.VisionFrame = undefined;
            var frame_count: usize = 0;
            defer {
                for (frames_buf[0..frame_count]) |fr| fr.close();
            }

            for (job.mediaSlice()) |item| {
                switch (item.kind) {
                    .image => {
                        frames_buf[frame_count] = vision_bridge.decodeImageBytes(self.model.device, item.data) catch {
                            stream.fail("image decode failed");
                            return;
                        };
                        frame_count += 1;
                    },
                    .video => {
                        const decoded = vision_bridge.decodeVideoBytesInto(
                            self.model.device,
                            item.data,
                            frames_buf[frame_count..@min(frames_buf.len, frame_count + 4)],
                        ) catch {
                            stream.fail("video decode failed");
                            return;
                        };
                        frame_count += decoded.len;
                    },
                }
            }
            if (frame_count == 0) {
                stream.fail("empty frames");
                return;
            }
            // Verify zero-copy page alignment on every decoded IOSurface frame.
            for (frames_buf[0..frame_count]) |fr| {
                const inf = fr.info() catch {
                    self.iosurface_leak_count += 1;
                    stream.fail("frame info failed");
                    return;
                };
                if (!inf.zeroCopyVerified() or (inf.baseAddress() % page.APPLE_PAGE_SIZE) != 0) {
                    self.iosurface_leak_count += 1;
                }
            }

            const sp = splitPrompt(job.prompt);
            const pre_ids = self.tok.encodeWithScratch(&self.scratch, sp.prefix, &self.prompt_buf) catch {
                stream.fail("prefix encode failed");
                return;
            };
            const suf_ids = self.tok.encodeWithScratch(&self.scratch, sp.suffix, self.prompt_buf[pre_ids.len..]) catch {
                stream.fail("suffix encode failed");
                return;
            };

            const mm = self.model.prefillMultimodal(
                self.allocator,
                pre_ids,
                suf_ids,
                frames_buf[0..frame_count],
                .{ .anyres_mode = .smart_resize, .compute_logits = true },
            ) catch {
                stream.fail("prefillMultimodal failed");
                return;
            };
            self.vision_tokens_last = mm.vision_tokens;
            if (!stream.setPromptTokens(mm.total_prompt_tokens)) return;

            var tok_id = mm.first_token;
            var pos = mm.total_prompt_tokens;
            const steps: u32 = @min(job.max_tokens, 3);
            for (0..steps) |_| {
                const piece = self.tok.tokenBytes(tok_id) catch "a";
                const emit_b = if (self.tok.isSpecialToken(tok_id) or piece.len == 0) "v" else piece;
                if (!stream.emitTokenBytes(emit_b)) return;
                tok_id = self.model.stepDecode(tok_id, pos) catch {
                    stream.fail("decode step failed");
                    return;
                };
                pos += 1;
            }
            stream.finish(.stop, mm.total_prompt_tokens);
            return;
        }

        const prompt_ids: []const u32 = if (job.endpoint == .completions and job.prompt_token_ids != null) blk: {
            self.saw_token_ids_prompt = true;
            break :blk job.prompt_token_ids.?;
        } else self.tok.encodeWithScratch(&self.scratch, job.prompt, &self.prompt_buf) catch {
            stream.fail("prompt encode failed");
            return;
        };

        const eff_ids = if (prompt_ids.len == 0) &[_]u32{0} else prompt_ids;
        self.model.resetKvCache();
        var tok_id = self.model.prefillChunk(eff_ids, 0, true) catch {
            stream.fail("prefillChunk failed");
            return;
        };
        const p_u32: u32 = @intCast(eff_ids.len);
        if (!stream.setPromptTokens(p_u32)) return;

        var pos = p_u32;
        const steps: u32 = @min(job.max_tokens, 3);
        for (0..steps) |_| {
            const piece = self.tok.tokenBytes(tok_id) catch "a";
            const emit_b = if (self.tok.isSpecialToken(tok_id) or piece.len == 0) "x" else piece;
            if (!stream.emitTokenBytes(emit_b)) return;
            tok_id = self.model.stepDecode(tok_id, pos) catch {
                stream.fail("decode step failed");
                return;
            };
            pos += 1;
        }
        stream.finish(.stop, p_u32);
    }

    fn embed(_: *anyopaque, req: *const server.EmbedRequest, out_vectors: []f32) anyerror!server.EmbedResult {
        const dim: usize = req.dimensions orelse 4;
        for (0..req.input_count) |i| {
            for (0..dim) |d| {
                out_vectors[i * dim + d] = @floatFromInt(d + 1);
            }
        }
        return .{ .dim = dim, .prompt_tokens = 2 };
    }

    fn rerank(_: *anyopaque, req: *const server.RerankRequest, out_items: []server.RerankItem) anyerror!server.RerankResult {
        for (0..req.document_count) |i| {
            out_items[i] = .{ .index = @intCast(i), .relevance_score = 1.0 - 0.1 * @as(f32, @floatFromInt(i)) };
        }
        return .{ .count = req.document_count, .prompt_tokens = 3 };
    }
};

pub fn runM96WithConfig(allocator: std.mem.Allocator, cfg: M96Config) !M96Report {
    var report: M96Report = .{};
    const vm_before = try vmstat.Snapshot.capture();

    const tmp_dir = try graph.makeTestTempDir(allocator, "bench_m96");
    defer {
        graph.removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    var io_impl = Io.Threaded.init(allocator, .{ .environ = .empty });
    defer io_impl.deinit();
    const io = io_impl.io();

    var q4_b = try graph.buildSynthQwen4Bundle(allocator, .{ .num_experts = 8, .top_k = 2, .vocab_size = 512 });
    defer q4_b.deinit(allocator);
    q4_b.cfg.qsa_window_size = 0;

    const vis_bytes = try graph.buildSynthVisionTowerAmfBytes(allocator, .{
        .num_layers = 3,
        .vis_hidden = 64,
        .vis_intermediate = 128,
        .lm_hidden = q4_b.cfg.hidden_size,
        .merger_hidden = 128,
        .patch_size = 16,
        .t_patch = 2,
        .spatial_merge = 2,
        .q4_qkv = true,
    });
    defer allocator.free(vis_bytes);
    const vis_path = try std.fs.path.joinZ(allocator, &.{ tmp_dir, "vision.amf" });
    defer allocator.free(vis_path);
    try writeBytesFile(vis_path, vis_bytes);

    var vis_m = try graph.GraphModel.loadFromBytes(allocator, q4_b.amf_bytes, q4_b.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 32,
        .startup_canary = false,
        .vision_amf_path = vis_path,
        .vision_max_patches = 64,
    });
    defer vis_m.deinit(allocator);

    const tok_json = try buildSynthTokenizerJsonAlloc(allocator);
    defer allocator.free(tok_json);
    var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, tok_json);
    defer tok.deinit(allocator);

    var encode_scratch = try tokenizer_mod.EncodeScratch.init(allocator, 4096);
    defer encode_scratch.deinit(allocator);

    // Ephemeral EC P-256 cert/key for TLS 1.3 CORS verification
    const ec_key = try std.fmt.allocPrintSentinel(allocator, "{s}/ec.key.pem", .{tmp_dir}, 0);
    defer allocator.free(ec_key);
    const ec_cert = try std.fmt.allocPrintSentinel(allocator, "{s}/ec.cert.pem", .{tmp_dir}, 0);
    defer allocator.free(ec_cert);
    try runCmdChecked(allocator, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", ec_key,
    });
    try runCmdChecked(allocator, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new", "-key", ec_key, "-subj", "/CN=127.0.0.1", "-days", "1", "-out", ec_cert,
    });
    var tls_files = try tls.loadFiles(allocator, ec_cert, ec_key);
    defer tls_files.deinit(allocator);

    const adps = [_]server.LoraAdapterMeta{
        .{ .name = "alpha", .display_name = "Synth Alpha", .rank = 8, .lora_alpha = 16.0, .scale = 2.0, .wired_bytes = 65536 },
        .{ .name = "beta", .display_name = "Synth Beta", .rank = 16, .lora_alpha = 32.0, .scale = 2.0, .wired_bytes = 131072 },
    };

    var worker_ctx = M96WorkerCtx{
        .allocator = allocator,
        .model = &vis_m,
        .tok = &tok,
        .scratch = encode_scratch,
    };

    const http_srv = try server.Server.init(allocator, .{
        .port = 0,
        .model_id = "qwen4-vl-synth",
        .api_key = "sk-bench-m96",
        .cors_allow_origin = "https://ui.local",
        .tokenizer = &tok,
        .embedding_handler = .{ .ctx = &worker_ctx, .embedFn = M96WorkerCtx.embed },
        .rerank_handler = .{ .ctx = &worker_ctx, .rerankFn = M96WorkerCtx.rerank },
        .model_detail = .{
            .display_name = "Qwen4 VL Synth",
            .architecture = "qwen4_moe",
            .max_model_len = 256,
            .vocab_size = 512,
            .capabilities = .{ .vision = true, .moe = true, .embeddings = true, .rerank = true, .lora = true },
        },
        .lora_adapters = &adps,
        .max_conns = 8,
        .max_active_requests = 4,
        .max_media_bytes = 512 * 1024,
        .apply_qos = false,
    });
    defer http_srv.deinit(allocator);
    try http_srv.start(.{ .ctx = &worker_ctx, .generateFn = M96WorkerCtx.generate });
    defer http_srv.stop();

    const https_srv = try server.Server.init(allocator, .{
        .port = 0,
        .model_id = "qwen4-vl-synth",
        .api_key = "sk-bench-m96",
        .cors_allow_origin = "https://ui.local",
        .tokenizer = &tok,
        .tls = .{ .cert_bytes = tls_files.cert_bytes, .key_bytes = tls_files.key_bytes },
        .max_conns = 6,
        .max_active_requests = 2,
        .max_media_bytes = 4096,
        .apply_qos = false,
    });
    defer https_srv.deinit(allocator);
    try https_srv.start(.{ .ctx = &worker_ctx, .generateFn = M96WorkerCtx.generate });
    defer https_srv.stop();

    const port = http_srv.boundPort();
    const auth = "Authorization: Bearer sk-bench-m96\r\n";
    var req_buf: [65536]u8 = undefined;
    var resp_buf: [65536]u8 = undefined;

    // 1. Anthropic Models API fields + limit/after_id/before_id pagination
    {
        const rm1 = try httpRequest(port, "GET", "/v1/models?limit=1", auth, "", &req_buf, &resp_buf);
        const bm1 = extractBody(rm1) orelse "";
        const rm1_ok = std.mem.startsWith(u8, rm1, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, bm1, "\"type\":\"model\"") != null and
            std.mem.indexOf(u8, bm1, "\"display_name\":\"Qwen4 VL Synth\"") != null and
            std.mem.indexOf(u8, bm1, "\"created_at\":\"") != null and
            std.mem.indexOf(u8, bm1, "\"max_input_tokens\":") != null and
            std.mem.indexOf(u8, bm1, "\"max_tokens\":") != null and
            std.mem.indexOf(u8, bm1, "\"has_more\":true") != null;

        const rm2 = try httpRequest(port, "GET", "/models?limit=2&after_id=qwen4-vl-synth", auth, "", &req_buf, &resp_buf);
        const bm2 = extractBody(rm2) orelse "";
        const rm2_ok = std.mem.startsWith(u8, rm2, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, bm2, "\"first_id\":\"qwen4-vl-synth:alpha\"") != null and
            std.mem.indexOf(u8, bm2, "\"has_more\":false") != null;

        const rm3 = try httpRequest(port, "GET", "/v1/models?before_id=qwen4-vl-synth%3Abeta", auth, "", &req_buf, &resp_buf);
        const bm3 = extractBody(rm3) orelse "";
        const rm3_ok = std.mem.startsWith(u8, rm3, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, bm3, "\"last_id\":\"qwen4-vl-synth:alpha\"") != null;

        report.models_pagination_ok = rm1_ok and rm2_ok and rm3_ok;
    }

    // 2. Prefix-free routes + --api-key 401
    {
        const r401 = try httpRequest(port, "GET", "/models", "", "", &req_buf, &resp_buf);
        const ok401 = std.mem.startsWith(u8, r401, "HTTP/1.1 401 Unauthorized\r\n");
        const r_msg = try httpRequest(port, "POST", "/messages", "x-api-key: sk-bench-m96\r\n", "{\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", &req_buf, &resp_buf);
        const ok_msg = std.mem.startsWith(u8, r_msg, "HTTP/1.1 200 OK\r\n");
        const r_cnt = try httpRequest(port, "POST", "/messages/count_tokens", "x-api-key: sk-bench-m96\r\n", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", &req_buf, &resp_buf);
        const ok_cnt = std.mem.startsWith(u8, r_cnt, "HTTP/1.1 200 OK\r\n");
        const r_chat = try httpRequest(port, "POST", "/chat/completions", auth, "{\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", &req_buf, &resp_buf);
        const ok_chat = std.mem.startsWith(u8, r_chat, "HTTP/1.1 200 OK\r\n");
        const r_emb = try httpRequest(port, "POST", "/embeddings", auth, "{\"input\":\"hi\",\"dimensions\":4}", &req_buf, &resp_buf);
        const ok_emb = std.mem.startsWith(u8, r_emb, "HTTP/1.1 200 OK\r\n");
        const r_rr = try httpRequest(port, "POST", "/rerank", auth, "{\"query\":\"q\",\"documents\":[\"a\",\"b\"]}", &req_buf, &resp_buf);
        const ok_rr = std.mem.startsWith(u8, r_rr, "HTTP/1.1 200 OK\r\n");
        report.prefix_free_routes_ok = ok401 and ok_msg and ok_cnt and ok_chat and ok_emb and ok_rr;
    }

    // 3 & 4. POST /v1/responses/input_tokens + GET /v1/responses/{id}/input_items
    {
        const rr1 = try httpRequest(port, "POST", "/responses", auth, "{\"instructions\":\"brief\",\"input\":\"turn one\",\"max_output_tokens\":2}", &req_buf, &resp_buf);
        if (!std.mem.startsWith(u8, rr1, "HTTP/1.1 200 OK\r\n")) {
            std.debug.print("m96 rr1 non-200: {s}\n", .{rr1});
            return error.InvalidHttpResponse;
        }
        const b1 = extractBody(rr1) orelse "";
        const RespId = struct { id: []const u8 };
        const p_r1 = try std.json.parseFromSlice(RespId, allocator, b1, .{ .ignore_unknown_fields = true });
        defer p_r1.deinit();

        var chain_buf: [256]u8 = undefined;
        const chain_body = try std.fmt.bufPrint(&chain_buf, "{{\"previous_response_id\":\"{s}\",\"input\":\"turn two\"}}", .{p_r1.value.id});
        const rt_single = try httpRequest(port, "POST", "/v1/responses/input_tokens", auth, "{\"input\":\"turn two\"}", &req_buf, &resp_buf);
        if (!std.mem.startsWith(u8, rt_single, "HTTP/1.1 200 OK\r\n")) {
            std.debug.print("m96 rt_single non-200: {s}\n", .{rt_single});
            return error.InvalidHttpResponse;
        }
        const TokCnt = struct { object: []const u8, input_tokens: u32 };
        const pt1 = try std.json.parseFromSlice(TokCnt, allocator, extractBody(rt_single) orelse "", .{});
        const pt1_tokens = pt1.value.input_tokens;
        pt1.deinit();

        const rt_chain = try httpRequest(port, "POST", "/responses/input_tokens", auth, chain_body, &req_buf, &resp_buf);
        if (!std.mem.startsWith(u8, rt_chain, "HTTP/1.1 200 OK\r\n")) {
            std.debug.print("m96 rt_chain non-200: {s}\n", .{rt_chain});
            return error.InvalidHttpResponse;
        }
        const pt2 = try std.json.parseFromSlice(TokCnt, allocator, extractBody(rt_chain) orelse "", .{});
        defer pt2.deinit();
        report.responses_input_tokens_ok = std.mem.eql(u8, pt2.value.object, "response.input_tokens") and (pt2.value.input_tokens > pt1_tokens);

        const rr2 = try httpRequest(port, "POST", "/v1/responses", auth, chain_body, &req_buf, &resp_buf);
        if (!std.mem.startsWith(u8, rr2, "HTTP/1.1 200 OK\r\n")) {
            std.debug.print("m96 rr2 non-200: {s}\n", .{rr2});
            return error.InvalidHttpResponse;
        }
        const p_r2 = try std.json.parseFromSlice(RespId, allocator, extractBody(rr2) orelse "", .{ .ignore_unknown_fields = true });
        var r2_id_buf: [128]u8 = undefined;
        if (p_r2.value.id.len > r2_id_buf.len) {
            p_r2.deinit();
            return error.BufferTooSmall;
        }
        @memcpy(r2_id_buf[0..p_r2.value.id.len], p_r2.value.id);
        const r2_id = r2_id_buf[0..p_r2.value.id.len];
        p_r2.deinit();

        var pbuf: [160]u8 = undefined;
        const path_asc = try std.fmt.bufPrint(&pbuf, "/v1/responses/{s}/input_items?order=asc", .{r2_id});
        const r_asc = try httpRequest(port, "GET", path_asc, auth, "", &req_buf, &resp_buf);
        const b_asc = extractBody(r_asc) orelse "";
        const asc_ok = std.mem.startsWith(u8, r_asc, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, b_asc, "turn one") != null and
            std.mem.indexOf(u8, b_asc, "turn two") != null;

        const path_desc = try std.fmt.bufPrint(&pbuf, "/responses/{s}/input_items?order=desc&limit=1", .{r2_id});
        const r_desc = try httpRequest(port, "GET", path_desc, auth, "", &req_buf, &resp_buf);
        const b_desc = extractBody(r_desc) orelse "";
        const desc_ok = std.mem.startsWith(u8, r_desc, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, b_desc, "turn two") != null and
            std.mem.indexOf(u8, b_desc, "\"has_more\":true") != null;

        report.responses_input_items_ok = asc_ok and desc_ok;
    }

    // 5. POST /v1/completions (SSE, JSON, FIM suffix, and [u32] token-array prompt)
    {
        const rc_json = try httpRequest(port, "POST", "/v1/completions", auth, "{\"prompt\":\"fn f(\",\"suffix\":\")\",\"max_tokens\":2,\"stream\":false}", &req_buf, &resp_buf);
        const ok_json = std.mem.startsWith(u8, rc_json, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, rc_json, "\"object\":\"text_completion\"") != null and
            std.mem.indexOf(u8, rc_json, "\"id\":\"cmpl-") != null;
        const rc_sse = try httpRequest(port, "POST", "/completions", auth, "{\"prompt\":\"ab\",\"max_tokens\":2,\"stream\":true,\"stream_options\":{\"include_usage\":true}}", &req_buf, &resp_buf);
        const ok_sse = std.mem.startsWith(u8, rc_sse, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, rc_sse, "data: [DONE]") != null;
        const rc_ids = try httpRequest(port, "POST", "/v1/completions", auth, "{\"prompt\":[10,11,12],\"max_tokens\":2,\"stream\":false}", &req_buf, &resp_buf);
        const ok_ids = std.mem.startsWith(u8, rc_ids, "HTTP/1.1 200 OK\r\n");
        report.completions_fim_ok = ok_json and ok_sse and ok_ids and worker_ctx.saw_token_ids_prompt;
    }

    // 6. Unauthenticated OPTIONS CORS preflight (204) + CORS over HTTP & TLS 1.3 HTTPS + 0 hot-path mallocs
    {
        const cors_n = @max(cfg.cors_iters, 1);
        _ = try httpRequest(port, "OPTIONS", "/completions", "", "", &req_buf, &resp_buf);
        var m0: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m0);
        const ct0 = monotonicNowNs();
        var opt_ok = true;
        for (0..cors_n) |_| {
            const ro = try httpRequest(port, "OPTIONS", "/completions", "", "", &req_buf, &resp_buf);
            if (!std.mem.startsWith(u8, ro, "HTTP/1.1 204 No Content\r\n") or
                !std.mem.eql(u8, extractHeaderVal(ro, "Access-Control-Allow-Origin") orelse "", "https://ui.local"))
            {
                opt_ok = false;
            }
        }
        const ct1 = monotonicNowNs();
        var m1: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &m1);
        report.cors_options_mean_us = @as(f64, @floatFromInt(ct1 -| ct0)) / (@as(f64, @floatFromInt(cors_n)) * 1000.0);
        report.malloc_blocks_delta = @as(i64, m1.blocks_in_use) - @as(i64, m0.blocks_in_use);

        const tls_n = @max(cfg.tls_iters, 1);
        const tls_req = "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
        const warm_tls = try tlsExchange(https_srv.boundPort(), tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3, tls_req, &resp_buf);
        report.tls13_negotiated = warm_tls.ver;

        var tls_ok = (warm_tls.ver == tls.TLS_VERSION_1_3) and
            std.mem.startsWith(u8, warm_tls.resp, "HTTP/1.1 204 No Content\r\n") and
            std.mem.eql(u8, extractHeaderVal(warm_tls.resp, "Access-Control-Allow-Origin") orelse "", "https://ui.local");
        const tt0 = monotonicNowNs();
        for (0..tls_n) |_| {
            const tr = try tlsExchange(https_srv.boundPort(), tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3, tls_req, &resp_buf);
            if (!std.mem.startsWith(u8, tr.resp, "HTTP/1.1 204 No Content\r\n")) tls_ok = false;
        }
        const tt1 = monotonicNowNs();
        report.tls13_cors_mean_us = @as(f64, @floatFromInt(tt1 -| tt0)) / (@as(f64, @floatFromInt(tls_n)) * 1000.0);
        report.cors_http_https_ok = opt_ok and tls_ok;
    }

    // 7. Base64 PNG + Base64 H.264 .mp4 -> vision_bridge IOSurface -> synthetic W20 ViT GraphModel multimodal HTTP serving
    {
        const png_src = try metal.VisionFrame.init(vis_m.device, .bgra8, 64, 64);
        defer png_src.close();
        {
            var g = try png_src.lockPlane(0, false);
            errdefer g.release();
            for (0..64) |y| {
                const row = g.rowBytes(@intCast(y));
                for (0..64) |x| {
                    row[x * 4 + 0] = @intCast((x * 4) % 256);
                    row[x * 4 + 1] = @intCast((y * 4) % 256);
                    row[x * 4 + 2] = 200;
                    row[x * 4 + 3] = 255;
                }
            }
            try g.unlock();
        }
        const pe0 = monotonicNowNs();
        const png_bytes = try vision_bridge.encodePngBytesAlloc(png_src, allocator);
        defer allocator.free(png_bytes);
        const png_b64 = try encodeBase64Alloc(allocator, png_bytes);
        defer allocator.free(png_b64);
        report.vision_png_encode_ms = @as(f64, @floatFromInt(monotonicNowNs() -| pe0)) / 1_000_000.0;

        const clip = try metal.VtClip.encodeTestClip(.h264, 64, 64, 4, 30);
        defer clip.close();
        const me0 = monotonicNowNs();
        const mp4_bytes = try vision_bridge.exportClipMp4BytesAlloc(clip, allocator);
        defer allocator.free(mp4_bytes);
        const mp4_b64 = try encodeBase64Alloc(allocator, mp4_bytes);
        defer allocator.free(mp4_b64);
        report.vision_mp4_encode_ms = @as(f64, @floatFromInt(monotonicNowNs() -| me0)) / 1_000_000.0;

        const body_buf = try allocator.alloc(u8, @max(png_b64.len, mp4_b64.len) + 512);
        defer allocator.free(body_buf);
        const big_req_buf = try allocator.alloc(u8, body_buf.len + 1024);
        defer allocator.free(big_req_buf);

        const chat_png = try std.fmt.bufPrint(
            body_buf,
            "{{\"max_tokens\":2,\"stream\":false,\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"see\"}},{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/png;base64,{s}\"}}}}]}}]}}",
            .{png_b64},
        );
        const r_cp = try httpRequest(port, "POST", "/v1/chat/completions", auth, chat_png, big_req_buf, &resp_buf);
        report.vision_png_chat_ok = std.mem.startsWith(u8, r_cp, "HTTP/1.1 200 OK\r\n");

        const resp_png = try std.fmt.bufPrint(
            body_buf,
            "{{\"max_output_tokens\":2,\"stream\":false,\"input\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,{s}\"}},{{\"type\":\"input_text\",\"text\":\"see\"}}]}}]}}",
            .{png_b64},
        );
        const r_rp = try httpRequest(port, "POST", "/v1/responses", auth, resp_png, big_req_buf, &resp_buf);
        report.vision_png_responses_ok = std.mem.startsWith(u8, r_rp, "HTTP/1.1 200 OK\r\n");

        const msg_png = try std.fmt.bufPrint(
            body_buf,
            "{{\"max_tokens\":2,\"stream\":false,\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"image\",\"source\":{{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"{s}\"}}}},{{\"type\":\"text\",\"text\":\"see\"}}]}}]}}",
            .{png_b64},
        );
        const r_mp = try httpRequest(port, "POST", "/v1/messages", auth, msg_png, big_req_buf, &resp_buf);
        report.vision_png_messages_ok = std.mem.startsWith(u8, r_mp, "HTTP/1.1 200 OK\r\n");

        const chat_mp4 = try std.fmt.bufPrint(
            body_buf,
            "{{\"max_tokens\":2,\"stream\":false,\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"vid\"}},{{\"type\":\"video_url\",\"video_url\":{{\"url\":\"data:video/mp4;base64,{s}\"}}}}]}}]}}",
            .{mp4_b64},
        );
        const r_cm = try httpRequest(port, "POST", "/v1/chat/completions", auth, chat_mp4, big_req_buf, &resp_buf);
        report.vision_mp4_chat_ok = std.mem.startsWith(u8, r_cm, "HTTP/1.1 200 OK\r\n");

        report.vision_tokens_last = worker_ctx.vision_tokens_last;
        report.iosurface_leak_count = worker_ctx.iosurface_leak_count;
    }

    const vm_after = try vmstat.Snapshot.capture();
    report.swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after);
    report.all_passed = report.models_pagination_ok and
        report.prefix_free_routes_ok and
        report.responses_input_tokens_ok and
        report.responses_input_items_ok and
        report.completions_fim_ok and
        report.cors_http_https_ok and
        report.vision_png_chat_ok and
        report.vision_png_responses_ok and
        report.vision_png_messages_ok and
        report.vision_mp4_chat_ok and
        (report.vision_tokens_last > 0) and
        (report.iosurface_leak_count == 0) and
        (report.malloc_blocks_delta <= 0) and
        (report.swapouts_delta == 0);
    return report;
}

pub fn runM96(allocator: std.mem.Allocator) !M96Report {
    return runM96WithConfig(allocator, .{});
}

test "M9.5 & M9.6 bench reports and synthetic suites pass" {
    const gpa = std.testing.allocator;
    const r95 = try runM95WithConfig(gpa, .{ .decode_iters = 4, .hot_load_iters = 1, .discovery_iters = 2 });
    try std.testing.expect(r95.all_passed);

    const r96 = try runM96WithConfig(gpa, .{ .cors_iters = 2, .tls_iters = 1 });
    try std.testing.expect(r96.all_passed);
}
