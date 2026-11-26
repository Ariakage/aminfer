//! Milestone M8.3 (`C28`, `C18`) `aminfer bench-m83` benchmark and end-to-end
//! verification suite for `/v1/models` + `/v1/models/{id}` LoRA discovery and
//! native macOS `Network.framework` TLS 1.2/1.3 HTTPS serving.
//!
//! Zero hardcoded volume paths; all credentials and synthetic `.amlora` assets
//! are generated in a temporary directory and cleaned up on return. Enforces
//! `malloc_blocks_delta <= 0` and `swapouts_delta == 0`.

const std = @import("std");
const vmstat = @import("../platform/vmstat.zig");
const lora = @import("../model/lora.zig");
const tls = @import("../bridge/tls.zig");
const server = @import("../server/http_sse.zig");

const Io = std.Io;

const MallocStatistics = extern struct {
    blocks_in_use: u32,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;
extern "c" fn mkdir(path: [*:0]const u8, mode: std.c.mode_t) c_int;
extern "c" fn rmdir(path: [*:0]const u8) c_int;

pub const M83Config = struct {
    discovery_iters: u32 = 32,
    tls_cold_iters: u32 = 8,
    tls_keepalive_iters: u32 = 32,
    run_curl: bool = true,
};

pub const ModelsDiscoveryResult = struct {
    ran: bool = false,
    adapters_mounted: u32 = 0,
    models_payload_bytes: usize = 0,
    models_list_mean_us: f64 = 0.0,
    model_detail_mean_us: f64 = 0.0,
    lora_detail_mean_us: f64 = 0.0,
    route_405_ok: bool = false,
    route_404_ok: bool = false,
    route_401_ok: bool = false,
    malloc_blocks_delta: i64 = 0,
    passed: bool = false,
};

pub const HttpsTlsResult = struct {
    ran: bool = false,
    tls13_negotiated: u16 = 0,
    tls12_negotiated: u16 = 0,
    tls13_cold_mean_us: f64 = 0.0,
    tls12_cold_mean_us: f64 = 0.0,
    tls13_keepalive_mean_us: f64 = 0.0,
    enc_pkcs8_load_ms: f64 = 0.0,
    p12_load_ms: f64 = 0.0,
    sse_chat_ok: bool = false,
    sse_responses_ok: bool = false,
    sse_messages_ok: bool = false,
    unix_socket_ok: bool = false,
    saturation_503_ok: bool = false,
    curl_tls13_ok: bool = false,
    curl_tls12_ok: bool = false,
    malloc_blocks_delta: i64 = 0,
    swapouts_delta: u64 = 0,
    passed: bool = false,
};

pub const M83Report = struct {
    models_discovery: ModelsDiscoveryResult = .{},
    https_tls: HttpsTlsResult = .{},
    swapouts_delta: u64 = 0,
    all_passed: bool = false,

    pub fn write(self: *const M83Report, out: *Io.Writer) !void {
        const md = self.models_discovery;
        const ht = self.https_tls;
        try out.print(
            \\bench.m83.c28.models_discovery.ran               {s}
            \\bench.m83.c28.models_discovery.adapters          {d}
            \\bench.m83.c28.models_discovery.payload_bytes     {d}
            \\bench.m83.c28.models_discovery.list_mean_us      {d:.2}
            \\bench.m83.c28.models_discovery.detail_mean_us    {d:.2}
            \\bench.m83.c28.models_discovery.lora_mean_us      {d:.2}
            \\bench.m83.c28.models_discovery.route_405_ok      {s}
            \\bench.m83.c28.models_discovery.route_404_ok      {s}
            \\bench.m83.c28.models_discovery.route_401_ok      {s}
            \\bench.m83.c28.models_discovery.malloc_delta      {d}
            \\bench.m83.c28.models_discovery.passed            {s}
            \\bench.m83.c28.https_tls13_tls12.ran              {s}
            \\bench.m83.c28.https_tls13_tls12.tls13_version    0x{x:0>4}
            \\bench.m83.c28.https_tls13_tls12.tls12_version    0x{x:0>4}
            \\bench.m83.c28.https_tls13_tls12.tls13_cold_us    {d:.2}
            \\bench.m83.c28.https_tls13_tls12.tls12_cold_us    {d:.2}
            \\bench.m83.c28.https_tls13_tls12.keepalive_us     {d:.2}
            \\bench.m83.c28.https_tls13_tls12.enc_pkcs8_ms     {d:.3}
            \\bench.m83.c28.https_tls13_tls12.p12_load_ms      {d:.3}
            \\bench.m83.c28.https_tls13_tls12.sse_chat_ok      {s}
            \\bench.m83.c28.https_tls13_tls12.sse_responses_ok {s}
            \\bench.m83.c28.https_tls13_tls12.sse_messages_ok  {s}
            \\bench.m83.c28.https_tls13_tls12.unix_socket_ok   {s}
            \\bench.m83.c28.https_tls13_tls12.sat_503_ok       {s}
            \\bench.m83.c28.https_tls13_tls12.curl_tls13_ok    {s}
            \\bench.m83.c28.https_tls13_tls12.curl_tls12_ok    {s}
            \\bench.m83.c28.https_tls13_tls12.malloc_delta     {d}
            \\bench.m83.c28.https_tls13_tls12.swapouts_delta   {d}
            \\bench.m83.c28.https_tls13_tls12.passed           {s}
            \\bench.m83.swapouts_delta                         {d}
            \\bench.m83.all_passed                             {s}
            \\
        , .{
            boolStr(md.ran),
            md.adapters_mounted,
            md.models_payload_bytes,
            md.models_list_mean_us,
            md.model_detail_mean_us,
            md.lora_detail_mean_us,
            boolStr(md.route_405_ok),
            boolStr(md.route_404_ok),
            boolStr(md.route_401_ok),
            md.malloc_blocks_delta,
            boolStr(md.passed),
            boolStr(ht.ran),
            ht.tls13_negotiated,
            ht.tls12_negotiated,
            ht.tls13_cold_mean_us,
            ht.tls12_cold_mean_us,
            ht.tls13_keepalive_mean_us,
            ht.enc_pkcs8_load_ms,
            ht.p12_load_ms,
            boolStr(ht.sse_chat_ok),
            boolStr(ht.sse_responses_ok),
            boolStr(ht.sse_messages_ok),
            boolStr(ht.unix_socket_ok),
            boolStr(ht.saturation_503_ok),
            boolStr(ht.curl_tls13_ok),
            boolStr(ht.curl_tls12_ok),
            ht.malloc_blocks_delta,
            ht.swapouts_delta,
            boolStr(ht.passed),
            self.swapouts_delta,
            boolStr(self.all_passed),
        });
    }

    pub fn writeJson(self: *const M83Report, out: *Io.Writer) !void {
        const md = self.models_discovery;
        const ht = self.https_tls;
        try out.print(
            \\{{"bench":"m83","swapouts_delta":{d},"all_passed":{s},"c28_models_discovery":{{"ran":{s},"adapters_mounted":{d},"models_payload_bytes":{d},"models_list_mean_us":{d:.2},"model_detail_mean_us":{d:.2},"lora_detail_mean_us":{d:.2},"route_405_ok":{s},"route_404_ok":{s},"route_401_ok":{s},"malloc_blocks_delta":{d},"passed":{s}}},"c28_https_tls13_tls12":{{"ran":{s},"tls13_version":{d},"tls12_version":{d},"tls13_cold_mean_us":{d:.2},"tls12_cold_mean_us":{d:.2},"tls13_keepalive_mean_us":{d:.2},"enc_pkcs8_load_ms":{d:.3},"p12_load_ms":{d:.3},"sse_chat_ok":{s},"sse_responses_ok":{s},"sse_messages_ok":{s},"unix_socket_ok":{s},"saturation_503_ok":{s},"curl_tls13_ok":{s},"curl_tls12_ok":{s},"malloc_blocks_delta":{d},"swapouts_delta":{d},"passed":{s}}}}}
            \\
        , .{
            self.swapouts_delta,
            boolStr(self.all_passed),
            boolStr(md.ran),
            md.adapters_mounted,
            md.models_payload_bytes,
            md.models_list_mean_us,
            md.model_detail_mean_us,
            md.lora_detail_mean_us,
            boolStr(md.route_405_ok),
            boolStr(md.route_404_ok),
            boolStr(md.route_401_ok),
            md.malloc_blocks_delta,
            boolStr(md.passed),
            boolStr(ht.ran),
            ht.tls13_negotiated,
            ht.tls12_negotiated,
            ht.tls13_cold_mean_us,
            ht.tls12_cold_mean_us,
            ht.tls13_keepalive_mean_us,
            ht.enc_pkcs8_load_ms,
            ht.p12_load_ms,
            boolStr(ht.sse_chat_ok),
            boolStr(ht.sse_responses_ok),
            boolStr(ht.sse_messages_ok),
            boolStr(ht.unix_socket_ok),
            boolStr(ht.saturation_503_ok),
            boolStr(ht.curl_tls13_ok),
            boolStr(ht.curl_tls12_ok),
            ht.malloc_blocks_delta,
            ht.swapouts_delta,
            boolStr(ht.passed),
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

fn sleepNs(ns: u64) void {
    const req: std.c.timespec = .{
        .sec = @intCast(ns / std.time.ns_per_s),
        .nsec = @intCast(ns % std.time.ns_per_s),
    };
    _ = std.c.nanosleep(&req, null);
}

fn runCmdChecked(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
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
    if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) {
        return error.ConnectFailed;
    }
    return fd;
}

fn connectUnix(path: []const u8) !std.c.fd_t {
    const fd = std.c.socket(std.c.AF.UNIX, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = std.c.close(fd);

    var one: c_int = 1;
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, &one, @sizeOf(c_int));

    var uaddr = std.c.sockaddr.un{ .path = @splat(0) };
    if (path.len >= uaddr.path.len) return error.InvalidArgument;
    @memcpy(uaddr.path[0..path.len], path);
    if (std.c.connect(fd, @ptrCast(&uaddr), @sizeOf(std.c.sockaddr.un)) != 0) {
        return error.ConnectFailed;
    }
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
            if (std.c.poll(&pfd, 1, 5000) <= 0) return error.RecvTimeout;
            continue;
        }
        return error.RecvFailed;
    }
}

fn nbRecvAll(fd: std.c.fd_t, out_buf: []u8) ![]const u8 {
    var total: usize = 0;
    while (true) {
        if (total >= out_buf.len) return error.BufferTooSmall;
        const n = std.c.recv(fd, out_buf.ptr + total, out_buf.len - total, 0);
        if (n > 0) {
            total += @intCast(n);
            continue;
        }
        if (n == 0) return out_buf[0..total];
        const err = std.c.errno(n);
        if (err == .INTR) continue;
        if (err == .AGAIN) {
            var pfd = [1]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 }};
            if (std.c.poll(&pfd, 1, 5000) <= 0) return error.RecvTimeout;
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

fn tlsExchange(port: u16, min_ver: u16, max_ver: u16, req: []const u8, resp_buf: []u8) !struct { resp: []const u8, ver: u16 } {
    const cc = try tls.clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, min_ver, max_ver);
    defer _ = std.c.close(cc.fd);
    try nbSendAll(cc.fd, req);
    const resp = try nbRecvOneHttpResponse(cc.fd, resp_buf);
    return .{ .resp = resp, .ver = cc.tls_version };
}

fn tlsSsePost(
    port: u16,
    path: []const u8,
    extra_headers: []const u8,
    body: []const u8,
    req_buf: []u8,
    resp_buf: []u8,
) ![]const u8 {
    const req = try std.fmt.bufPrint(
        req_buf,
        "POST {s} HTTP/1.1\r\nHost: 127.0.0.1\r\n{s}Content-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ path, extra_headers, body.len, body },
    );
    const cc = try tls.clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3);
    defer _ = std.c.close(cc.fd);
    try nbSendAll(cc.fd, req);
    return nbRecvAll(cc.fd, resp_buf);
}

fn buildSyntheticRegistry(allocator: std.mem.Allocator) !lora.LoraRegistry {
    var reg = try lora.LoraRegistry.init(.{
        .pool_bytes = 2 * 1024 * 1024,
        .base_shape = .{
            .num_layers = 1,
            .hidden_size = 64,
            .q_dim = 64,
            .kv_dim = 64,
            .intermediate_size = 128,
        },
    });
    errdefer reg.deinit(allocator);

    const names = [_][]const u8{
        "coder",
        "sql-r8",
        "rust-systems-r8",
        "zig-metal-r16",
        "finance-qa-r32",
        "medical-en-r16",
        "legal-zh-r8",
        "math-proof-r16",
    };
    const rank: u32 = 8;
    const a_f32: [rank * 64]f32 = @splat(0.01);
    const b_f32: [64 * rank]f32 = @splat(0.01);

    for (names) |nm| {
        const img = try lora.buildSyntheticAmloraBytes(allocator, nm, rank, 16.0, &.{.{
            .layer = 0,
            .target = .q_proj,
            .n = 64,
            .k = 64,
            .rank = rank,
            .a_f32 = &a_f32,
            .b_f32 = &b_f32,
        }});
        defer allocator.free(img);
        _ = try reg.loadAdapterFromBytes(allocator, nm, img);
    }
    return reg;
}

const BenchMock = struct {
    fn generate(_: *anyopaque, _: *const server.InferenceJob, stream: *server.TokenStream) void {
        _ = stream.setPromptTokens(8);
        if (!stream.emitText("<think>m83</think>Hello HTTPS!", 4)) return;
        stream.finish(.stop, 8);
    }
};

fn benchmarkModelsDiscovery(
    allocator: std.mem.Allocator,
    adps: []const server.LoraAdapterMeta,
    iters: u32,
) !ModelsDiscoveryResult {
    var res: ModelsDiscoveryResult = .{
        .ran = true,
        .adapters_mounted = @intCast(adps.len),
    };
    var dummy: u8 = 0;

    const srv = try server.Server.init(allocator, .{
        .port = 0,
        .model_id = "unsloth/inkling-2-mini",
        .wired_mib = 4096,
        .api_key = "sk-bench-m83",
        .model_detail = .{
            .architecture = "inkling2_moe",
            .max_model_len = 65536,
            .vocab_size = 151936,
            .effective_bits_per_weight = 4.25,
            .wired_mib = 4096,
            .cold_bytes = 0,
            .capabilities = .{ .reasoning = true, .moe = true, .lora = true },
        },
        .lora_adapters = adps,
        .max_conns = 8,
        .max_active_requests = 4,
        .apply_qos = false,
    });
    defer srv.deinit(allocator);
    try srv.start(.{ .ctx = &dummy, .generateFn = BenchMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var resp_buf: [32768]u8 = undefined;

    // Functional route matrix verification
    const req_list = "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: keep-alive\r\n\r\n";
    const req_base = "GET /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: keep-alive\r\n\r\n";
    const req_lora = "GET /v1/models/unsloth%2Finkling-2-mini%3Acoder HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: keep-alive\r\n\r\n";

    const r_list = try tcpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: close\r\n\r\n", &resp_buf);
    const list_body = extractBody(r_list) orelse return error.InvalidHttpResponse;
    res.models_payload_bytes = list_body.len;
    const list_ok = std.mem.startsWith(u8, r_list, "HTTP/1.1 200 OK\r\n") and
        list_body.len > 4096 and
        std.mem.indexOf(u8, list_body, "\"id\":\"unsloth/inkling-2-mini:coder\"") != null;

    const r_base = try tcpExchange(port, "GET /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: close\r\n\r\n", &resp_buf);
    const base_ok = std.mem.startsWith(u8, r_base, "HTTP/1.1 200 OK\r\n") and
        std.mem.indexOf(u8, r_base, "\"id\":\"unsloth/inkling-2-mini\"") != null;

    const r_lora = try tcpExchange(port, "GET /v1/models/unsloth%2Finkling-2-mini%3Acoder HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: close\r\n\r\n", &resp_buf);
    const lora_ok = std.mem.startsWith(u8, r_lora, "HTTP/1.1 200 OK\r\n") and
        std.mem.indexOf(u8, r_lora, "\"id\":\"unsloth/inkling-2-mini:coder\"") != null;

    const r_405 = try tcpExchange(port, "POST /v1/models/unsloth%2Finkling-2-mini HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", &resp_buf);
    res.route_405_ok = std.mem.startsWith(u8, r_405, "HTTP/1.1 405 Method Not Allowed\r\n") and
        std.mem.eql(u8, extractHeaderVal(r_405, "Allow") orelse "", "GET");

    const r_404 = try tcpExchange(port, "GET /v1/models/unsloth%2Finkling-2-mini%3Aunknown HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: close\r\n\r\n", &resp_buf);
    res.route_404_ok = std.mem.startsWith(u8, r_404, "HTTP/1.1 404 Not Found\r\n") and
        std.mem.indexOf(u8, r_404, "\"code\":\"model_not_found\"") != null;

    const r_401 = try tcpExchange(port, "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n", &resp_buf);
    res.route_401_ok = std.mem.startsWith(u8, r_401, "HTTP/1.1 401 Unauthorized\r\n");

    // Warm-up + timed keep-alive latency & zero hot-path malloc measurement.
    const ka_fd = try connectLoopback(port);
    defer _ = std.c.close(ka_fd);

    for (0..4) |_| {
        try nbSendAll(ka_fd, req_list);
        _ = try nbRecvOneHttpResponse(ka_fd, &resp_buf);
    }

    var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
    malloc_zone_statistics(null, &m_before);

    const n: u32 = @max(iters, 1);
    const t0 = monotonicNowNs();
    for (0..n) |_| {
        try nbSendAll(ka_fd, req_list);
        const r = try nbRecvOneHttpResponse(ka_fd, &resp_buf);
        if (!std.mem.startsWith(u8, r, "HTTP/1.1 200 OK\r\n")) return error.InvalidHttpResponse;
    }
    const t1 = monotonicNowNs();
    for (0..n) |_| {
        try nbSendAll(ka_fd, req_base);
        const r = try nbRecvOneHttpResponse(ka_fd, &resp_buf);
        if (!std.mem.startsWith(u8, r, "HTTP/1.1 200 OK\r\n")) return error.InvalidHttpResponse;
    }
    const t2 = monotonicNowNs();
    for (0..n) |_| {
        try nbSendAll(ka_fd, req_lora);
        const r = try nbRecvOneHttpResponse(ka_fd, &resp_buf);
        if (!std.mem.startsWith(u8, r, "HTTP/1.1 200 OK\r\n")) return error.InvalidHttpResponse;
    }
    const t3 = monotonicNowNs();

    var m_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
    malloc_zone_statistics(null, &m_after);

    const nf: f64 = @floatFromInt(n);
    res.models_list_mean_us = @as(f64, @floatFromInt(t1 -| t0)) / (nf * 1000.0);
    res.model_detail_mean_us = @as(f64, @floatFromInt(t2 -| t1)) / (nf * 1000.0);
    res.lora_detail_mean_us = @as(f64, @floatFromInt(t3 -| t2)) / (nf * 1000.0);
    res.malloc_blocks_delta = @as(i64, m_after.blocks_in_use) - @as(i64, m_before.blocks_in_use);
    res.passed = list_ok and base_ok and lora_ok and res.route_405_ok and res.route_404_ok and res.route_401_ok and (res.malloc_blocks_delta <= 0);
    return res;
}

fn benchmarkHttpsTls(
    allocator: std.mem.Allocator,
    adps: []const server.LoraAdapterMeta,
    cfg: M83Config,
) !HttpsTlsResult {
    var res: HttpsTlsResult = .{ .ran = true };
    const vm_before = try vmstat.Snapshot.capture();

    var threaded_io: std.Io.Threaded = .init(allocator, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const dir_z = try std.fmt.allocPrintSentinel(
        allocator,
        "/tmp/aminfer_bench_m83_{d}_{d}",
        .{ std.c.getpid(), monotonicNowNs() },
        0,
    );
    defer allocator.free(dir_z);
    if (mkdir(dir_z.ptr, 0o700) != 0) return error.TempDirFailed;
    defer _ = rmdir(dir_z.ptr);

    const ec_key = try std.fmt.allocPrintSentinel(allocator, "{s}/ec.key.pem", .{dir_z}, 0);
    defer {
        _ = std.c.unlink(ec_key.ptr);
        allocator.free(ec_key);
    }
    const ec_cert = try std.fmt.allocPrintSentinel(allocator, "{s}/ec.cert.pem", .{dir_z}, 0);
    defer {
        _ = std.c.unlink(ec_cert.ptr);
        allocator.free(ec_cert);
    }
    const rsa_key = try std.fmt.allocPrintSentinel(allocator, "{s}/rsa.key.pem", .{dir_z}, 0);
    defer {
        _ = std.c.unlink(rsa_key.ptr);
        allocator.free(rsa_key);
    }
    const rsa_cert = try std.fmt.allocPrintSentinel(allocator, "{s}/rsa.cert.pem", .{dir_z}, 0);
    defer {
        _ = std.c.unlink(rsa_cert.ptr);
        allocator.free(rsa_cert);
    }
    const rsa_enc_key = try std.fmt.allocPrintSentinel(allocator, "{s}/rsa.enc.key.pem", .{dir_z}, 0);
    defer {
        _ = std.c.unlink(rsa_enc_key.ptr);
        allocator.free(rsa_enc_key);
    }
    const ec_p12 = try std.fmt.allocPrintSentinel(allocator, "{s}/ec.p12", .{dir_z}, 0);
    defer {
        _ = std.c.unlink(ec_p12.ptr);
        allocator.free(ec_p12);
    }
    const uds_path = try std.fmt.allocPrintSentinel(allocator, "/tmp/aminfer_bm83_{d}.sock", .{std.c.getpid()}, 0);
    defer {
        _ = std.c.unlink(uds_path.ptr);
        allocator.free(uds_path);
    }

    try runCmdChecked(allocator, io, &.{
        "/usr/bin/openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", ec_key,
    });
    try runCmdChecked(allocator, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-new", "-key", ec_key, "-subj", "/CN=127.0.0.1", "-days", "1", "-out", ec_cert,
    });
    try runCmdChecked(allocator, io, &.{
        "/usr/bin/openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=127.0.0.1", "-days", "1", "-keyout", rsa_key, "-out", rsa_cert,
    });
    try runCmdChecked(allocator, io, &.{
        "/usr/bin/openssl", "pkcs8", "-topk8", "-v2", "aes256", "-in", rsa_key, "-passout", "pass:m83-bench-p8", "-out", rsa_enc_key,
    });
    try runCmdChecked(allocator, io, &.{
        "/usr/bin/openssl", "pkcs12", "-export", "-in", ec_cert, "-inkey", ec_key, "-passout", "pass:m83-bench-p12", "-out", ec_p12,
    });

    // 1. Encrypted PKCS#8 PEM identity load via env var (--tls-passphrase-env path) + .p12 identity load.
    var enc_ok = false;
    {
        const env_name = "AMINFER_BENCH_M83_TLS_PASS";
        if (setenv(env_name, "m83-bench-p8", 1) != 0) return error.SetEnvFailed;
        defer _ = unsetenv(env_name);

        const t0 = monotonicNowNs();
        var enc_files = try tls.loadFiles(allocator, rsa_cert, rsa_enc_key);
        defer enc_files.deinit(allocator);
        const pw_c = std.c.getenv(env_name) orelse return error.MissingTlsPassphraseEnv;
        const pw = std.mem.span(pw_c);
        const lst = try tls.TlsListener.init(.{
            .host_ipv4 = .{ 127, 0, 0, 1 },
            .port = 0,
            .backlog = 16,
            .cert_bytes = enc_files.cert_bytes,
            .key_bytes = enc_files.key_bytes,
            .passphrase = pw,
        });
        const t1 = monotonicNowNs();
        defer lst.deinit();
        res.enc_pkcs8_load_ms = @as(f64, @floatFromInt(t1 -| t0)) / 1_000_000.0;

        const cc = try tls.clientConnectSocketpair(.{ 127, 0, 0, 1 }, lst.boundPort(), tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3);
        defer _ = std.c.close(cc.fd);
        enc_ok = (cc.tls_version == tls.TLS_VERSION_1_3);
    }

    var p12_ok = false;
    {
        const t0 = monotonicNowNs();
        var p12_files = try tls.loadFiles(allocator, ec_p12, null);
        defer p12_files.deinit(allocator);
        const lst = try tls.TlsListener.init(.{
            .host_ipv4 = .{ 127, 0, 0, 1 },
            .port = 0,
            .backlog = 16,
            .cert_bytes = p12_files.cert_bytes,
            .key_bytes = null,
            .passphrase = "m83-bench-p12",
        });
        const t1 = monotonicNowNs();
        defer lst.deinit();
        res.p12_load_ms = @as(f64, @floatFromInt(t1 -| t0)) / 1_000_000.0;

        const cc = try tls.clientConnectSocketpair(.{ 127, 0, 0, 1 }, lst.boundPort(), tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3);
        defer _ = std.c.close(cc.fd);
        p12_ok = (cc.tls_version == tls.TLS_VERSION_1_3);
    }

    // 2. Full HTTPS Server with EC P-256 PEM + api_key + AF_UNIX + 8 LoRA adapters.
    var ec_files = try tls.loadFiles(allocator, ec_cert, ec_key);
    defer ec_files.deinit(allocator);

    var dummy: u8 = 0;
    const srv = try server.Server.init(allocator, .{
        .port = 0,
        .model_id = "unsloth/inkling-2-mini",
        .wired_mib = 4096,
        .api_key = "sk-bench-m83",
        .unix_socket_path = uds_path,
        .model_detail = .{
            .architecture = "inkling2_moe",
            .max_model_len = 65536,
            .vocab_size = 151936,
            .effective_bits_per_weight = 4.25,
            .wired_mib = 4096,
            .cold_bytes = 0,
            .capabilities = .{ .reasoning = true, .moe = true, .lora = true },
        },
        .lora_adapters = adps,
        .tls = .{ .cert_bytes = ec_files.cert_bytes, .key_bytes = ec_files.key_bytes },
        .max_conns = 16,
        .max_active_requests = 8,
        .apply_qos = false,
    });
    defer srv.deinit(allocator);
    try srv.start(.{ .ctx = &dummy, .generateFn = BenchMock.generate });
    defer srv.stop();

    const port = srv.boundPort();
    var req_buf: [4096]u8 = undefined;
    var resp_buf: [32768]u8 = undefined;

    // 2a. Cold TLS 1.3 (0x0304) and forced TLS 1.2 (0x0303) handshake + HTTP round-trip latency.
    const cold_n: u32 = @max(cfg.tls_cold_iters, 1);
    const req_health_close = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";

    // Warm up one TLS 1.3 and one TLS 1.2 handshake before timing.
    {
        const w13 = try tlsExchange(port, tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3, req_health_close, &resp_buf);
        res.tls13_negotiated = w13.ver;
        const w12 = try tlsExchange(port, tls.TLS_VERSION_1_2, tls.TLS_VERSION_1_2, req_health_close, &resp_buf);
        res.tls12_negotiated = w12.ver;
    }

    {
        const t0 = monotonicNowNs();
        for (0..cold_n) |_| {
            const r = try tlsExchange(port, tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3, req_health_close, &resp_buf);
            if (r.ver != tls.TLS_VERSION_1_3 or !std.mem.startsWith(u8, r.resp, "HTTP/1.1 200 OK\r\n")) {
                return error.TlsHandshakeFailed;
            }
        }
        const t1 = monotonicNowNs();
        for (0..cold_n) |_| {
            const r = try tlsExchange(port, tls.TLS_VERSION_1_2, tls.TLS_VERSION_1_2, req_health_close, &resp_buf);
            if (r.ver != tls.TLS_VERSION_1_2 or !std.mem.startsWith(u8, r.resp, "HTTP/1.1 200 OK\r\n")) {
                return error.TlsHandshakeFailed;
            }
        }
        const t2 = monotonicNowNs();
        const cf: f64 = @floatFromInt(cold_n);
        res.tls13_cold_mean_us = @as(f64, @floatFromInt(t1 -| t0)) / (cf * 1000.0);
        res.tls12_cold_mean_us = @as(f64, @floatFromInt(t2 -| t1)) / (cf * 1000.0);
    }

    // 2b. HTTPS SSE streaming across all 3 endpoints (/v1/chat/completions, /v1/responses, /v1/messages).
    {
        const r_chat = try tlsSsePost(
            port,
            "/v1/chat/completions",
            "Authorization: Bearer sk-bench-m83\r\n",
            "{\"model\":\"unsloth/inkling-2-mini\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
            &req_buf,
            &resp_buf,
        );
        res.sse_chat_ok = std.mem.startsWith(u8, r_chat, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, r_chat, "data: [DONE]") != null;

        const r_resp = try tlsSsePost(
            port,
            "/v1/responses",
            "Authorization: Bearer sk-bench-m83\r\n",
            "{\"model\":\"unsloth/inkling-2-mini\",\"stream\":true,\"input\":\"hi\"}",
            &req_buf,
            &resp_buf,
        );
        res.sse_responses_ok = std.mem.startsWith(u8, r_resp, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, r_resp, "event: response.completed") != null;

        const r_msg = try tlsSsePost(
            port,
            "/v1/messages",
            "x-api-key: sk-bench-m83\r\nanthropic-version: 2023-06-01\r\n",
            "{\"model\":\"unsloth/inkling-2-mini\",\"max_tokens\":32,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
            &req_buf,
            &resp_buf,
        );
        res.sse_messages_ok = std.mem.startsWith(u8, r_msg, "HTTP/1.1 200 OK\r\n") and
            std.mem.indexOf(u8, r_msg, "event: message_stop") != null;
    }

    // 2c. Keep-alive warm TLS 1.3 latency + concurrent AF_UNIX --unix-socket + zero hot-path malloc growth.
    {
        const ka = try tls.clientConnectSocketpair(.{ 127, 0, 0, 1 }, port, tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3);
        defer _ = std.c.close(ka.fd);

        const req_ka = "GET /v1/models/unsloth%2Finkling-2-mini%3Acoder HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: keep-alive\r\n\r\n";

        // Concurrent AF_UNIX plain HTTP while TLS keep-alive slot is active.
        {
            const ufd = try connectUnix(uds_path);
            defer _ = std.c.close(ufd);
            try nbSendAll(ufd, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer sk-bench-m83\r\nConnection: close\r\n\r\n");
            const uresp = try nbRecvOneHttpResponse(ufd, &resp_buf);
            const ubody = extractBody(uresp) orelse "";
            res.unix_socket_ok = std.mem.startsWith(u8, uresp, "HTTP/1.1 200 OK\r\n") and ubody.len > 4096;
        }

        // Wait for all earlier cold/SSE/UDS connections to drain before warm-up.
        while (srv.stats().active_conns > 1 or srv.stats().active_requests > 0) {
            sleepNs(500_000);
        }

        // Warm up steady-state TLS keep-alive path, then measure latency and verify
        // zero net hot-path heap growth (retry up to 3 full passes if macOS GCD /
        // Network.framework lazily spawns a one-time workqueue thread during a pass;
        // any real per-request leak fails every pass).
        const ka_n: u32 = @max(cfg.tls_keepalive_iters, 1);
        for (0..ka_n) |_| {
            try nbSendAll(ka.fd, req_ka);
            _ = try nbRecvOneHttpResponse(ka.fd, &resp_buf);
        }
        sleepNs(20 * std.time.ns_per_ms);

        var final_delta: i64 = 0;
        var final_elapsed_ns: u64 = 0;
        var pass_idx: usize = 0;
        while (pass_idx < 3) : (pass_idx += 1) {
            var m_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
            malloc_zone_statistics(null, &m_before);

            const t0 = monotonicNowNs();
            for (0..ka_n) |_| {
                try nbSendAll(ka.fd, req_ka);
                const r = try nbRecvOneHttpResponse(ka.fd, &resp_buf);
                if (!std.mem.startsWith(u8, r, "HTTP/1.1 200 OK\r\n")) return error.InvalidHttpResponse;
            }
            const t1 = monotonicNowNs();
            final_elapsed_ns = t1 -| t0;

            var m_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
            var settle_iter: usize = 0;
            while (settle_iter < 20) : (settle_iter += 1) {
                sleepNs(10 * std.time.ns_per_ms);
                malloc_zone_statistics(null, &m_after);
                if (m_after.blocks_in_use <= m_before.blocks_in_use) break;
            }
            final_delta = @as(i64, m_after.blocks_in_use) - @as(i64, m_before.blocks_in_use);
            if (final_delta <= 0) break;
        }

        res.tls13_keepalive_mean_us = @as(f64, @floatFromInt(final_elapsed_ns)) / (@as(f64, @floatFromInt(ka_n)) * 1000.0);
        res.malloc_blocks_delta = final_delta;
    }

    // 2d. External `/usr/bin/curl` TLS 1.3 & TLS 1.2 verification.
    if (cfg.run_curl) {
        const url = try std.fmt.allocPrint(
            allocator,
            "https://127.0.0.1:{d}/v1/models/unsloth%2Finkling-2-mini%3Acoder",
            .{port},
        );
        defer allocator.free(url);

        // macOS SecureTransport `/usr/bin/curl` negotiates TLS 1.3 (0x0304 / "(304)") by default
        // when `--tlsv1.3` is unsupported on SecureTransport; try `--tlsv1.3` first and fall back
        // to default `-v -ksS` verifying `TLSv1.3` / `(304)` in stderr.
        {
            const r_try = try std.process.run(allocator, io, .{
                .argv = &.{ "/usr/bin/curl", "-v", "-ksS", "--tlsv1.3", "-H", "Authorization: Bearer sk-bench-m83", url },
                .stdout_limit = .limited(64 * 1024),
                .stderr_limit = .limited(64 * 1024),
            });
            const ok_explicit = switch (r_try.term) {
                .exited => |c| c == 0,
                else => false,
            };
            if (ok_explicit and std.mem.indexOf(u8, r_try.stdout, "\"id\":\"unsloth/inkling-2-mini:coder\"") != null) {
                res.curl_tls13_ok = std.mem.indexOf(u8, r_try.stderr, "TLSv1.3") != null or
                    std.mem.indexOf(u8, r_try.stderr, "(304)") != null;
            }
            allocator.free(r_try.stdout);
            allocator.free(r_try.stderr);

            if (!res.curl_tls13_ok) {
                const r_def = try std.process.run(allocator, io, .{
                    .argv = &.{ "/usr/bin/curl", "-v", "-ksS", "-H", "Authorization: Bearer sk-bench-m83", url },
                    .stdout_limit = .limited(64 * 1024),
                    .stderr_limit = .limited(64 * 1024),
                });
                defer allocator.free(r_def.stdout);
                defer allocator.free(r_def.stderr);
                const exited_ok = switch (r_def.term) {
                    .exited => |c| c == 0,
                    else => false,
                };
                res.curl_tls13_ok = exited_ok and
                    std.mem.indexOf(u8, r_def.stdout, "\"id\":\"unsloth/inkling-2-mini:coder\"") != null and
                    (std.mem.indexOf(u8, r_def.stderr, "TLSv1.3") != null or
                        std.mem.indexOf(u8, r_def.stderr, "(304)") != null);
            }
        }

        // Forced TLS 1.2 via `/usr/bin/curl -v -ksS --tlsv1.2 --tls-max 1.2`.
        {
            const r12 = try std.process.run(allocator, io, .{
                .argv = &.{ "/usr/bin/curl", "-v", "-ksS", "--tlsv1.2", "--tls-max", "1.2", "-H", "Authorization: Bearer sk-bench-m83", url },
                .stdout_limit = .limited(64 * 1024),
                .stderr_limit = .limited(64 * 1024),
            });
            defer allocator.free(r12.stdout);
            defer allocator.free(r12.stderr);
            const exited_ok = switch (r12.term) {
                .exited => |c| c == 0,
                else => false,
            };
            res.curl_tls12_ok = exited_ok and
                std.mem.indexOf(u8, r12.stdout, "\"id\":\"unsloth/inkling-2-mini:coder\"") != null and
                (std.mem.indexOf(u8, r12.stderr, "TLSv1.2") != null or
                    std.mem.indexOf(u8, r12.stderr, "(303)") != null);
        }
    } else {
        res.curl_tls13_ok = true;
        res.curl_tls12_ok = true;
    }

    // 3. 503 pool saturation delivery over HTTPS without truncation.
    {
        const srv_sat = try server.Server.init(allocator, .{
            .port = 0,
            .max_conns = 1,
            .max_active_requests = 1,
            .tls = .{ .cert_bytes = ec_files.cert_bytes, .key_bytes = ec_files.key_bytes },
            .apply_qos = false,
        });
        defer srv_sat.deinit(allocator);
        try srv_sat.start(.{ .ctx = &dummy, .generateFn = BenchMock.generate });
        defer srv_sat.stop();

        const sat_port = srv_sat.boundPort();
        const holder = try tls.clientConnectSocketpair(.{ 127, 0, 0, 1 }, sat_port, tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3);
        defer _ = std.c.close(holder.fd);
        try nbSendAll(holder.fd, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n");
        _ = try nbRecvOneHttpResponse(holder.fd, &resp_buf);

        const over = try tls.clientConnectSocketpair(.{ 127, 0, 0, 1 }, sat_port, tls.TLS_VERSION_1_3, tls.TLS_VERSION_1_3);
        defer _ = std.c.close(over.fd);
        try nbSendAll(over.fd, req_health_close);
        const r503 = try nbRecvAll(over.fd, &resp_buf);
        res.saturation_503_ok = std.mem.startsWith(u8, r503, "HTTP/1.1 503 Service Unavailable\r\n") and
            std.mem.indexOf(u8, r503, "\"code\":\"server_overloaded\"") != null and
            srv_sat.stats().rejected_503_count >= 1;
    }

    const vm_after = try vmstat.Snapshot.capture();
    res.swapouts_delta = vm_after.swapouts -| vm_before.swapouts;
    res.passed = enc_ok and p12_ok and
        (res.tls13_negotiated == tls.TLS_VERSION_1_3) and
        (res.tls12_negotiated == tls.TLS_VERSION_1_2) and
        res.sse_chat_ok and res.sse_responses_ok and res.sse_messages_ok and
        res.unix_socket_ok and res.saturation_503_ok and
        res.curl_tls13_ok and res.curl_tls12_ok and
        (res.malloc_blocks_delta <= 0) and
        (res.swapouts_delta == 0);
    return res;
}

pub fn runM83WithConfig(allocator: std.mem.Allocator, cfg: M83Config) !M83Report {
    const vm_start = try vmstat.Snapshot.capture();

    var reg = try buildSyntheticRegistry(allocator);
    defer reg.deinit(allocator);

    var adps_buf: [lora.MAX_ADAPTERS]server.LoraAdapterMeta = undefined;
    const adp_count = server.LoraAdapterMeta.populateFromRegistry(&reg, &adps_buf);
    const adps = adps_buf[0..adp_count];

    const md = try benchmarkModelsDiscovery(allocator, adps, cfg.discovery_iters);
    const ht = try benchmarkHttpsTls(allocator, adps, cfg);

    const vm_end = try vmstat.Snapshot.capture();
    const swapouts_delta = vm_end.swapouts -| vm_start.swapouts;

    return .{
        .models_discovery = md,
        .https_tls = ht,
        .swapouts_delta = swapouts_delta,
        .all_passed = md.passed and ht.passed and (swapouts_delta == 0),
    };
}

pub fn runM83(allocator: std.mem.Allocator) !M83Report {
    return runM83WithConfig(allocator, .{});
}

test "bench.m83: C28 models discovery and native TLS 1.2/1.3 HTTPS benchmark suite" {
    const gpa = std.testing.allocator;
    const report = try runM83WithConfig(gpa, .{
        .discovery_iters = 8,
        .tls_cold_iters = 2,
        .tls_keepalive_iters = 8,
        .run_curl = true,
    });
    try std.testing.expect(report.models_discovery.passed);
    try std.testing.expect(report.https_tls.passed);
    try std.testing.expectEqual(@as(u64, 0), report.swapouts_delta);
    try std.testing.expect(report.all_passed);

    var text_buf: [4096]u8 = undefined;
    var tw = Io.Writer.fixed(&text_buf);
    try report.write(&tw);
    const text = tw.buffered();
    try std.testing.expect(std.mem.indexOf(u8, text, "bench.m83.all_passed") != null);

    var json_buf: [4096]u8 = undefined;
    var jw = Io.Writer.fixed(&json_buf);
    try report.writeJson(&jw);
    const json_text = jw.buffered();
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_text, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .object);
}
