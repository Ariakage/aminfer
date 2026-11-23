//! `aminfer monitor` (`C19`, ADR-0022): rootless Apple Silicon terminal TUI and
//! deterministic JSON/frame snapshot monitor.
//!
//! - Zero third-party dependencies; driven by Darwin `kqueue` (`EVFILT_TIMER`,
//!   `EVFILT_READ`, `EVFILT_SIGNAL` `SIGWINCH`) over raw `termios` alternate screen.
//! - Preallocated double-buffered TrueColor + Unicode Braille (`⠀..⣿`) cell canvas
//!   performing zero steady-state heap allocations per frame and flushing diffed
//!   ANSI writes in one syscall per tick.
//! - Built-in `trans-pride` default palette + `--theme <path.theme>` `btop` loader.
//! - Four panels:
//!   1. CPU P/E-cores & SME2 + `ChipProfile` (`C16`)
//!   2. `HOST_VM_INFO64` / `ADR-0012` wired budget / `R9` zero-swap reclaimable /
//!      compressor / `swapouts` delta
//!   3. Rootless `IOAccelerator` Metal GPU & `NSProcessInfo.thermalState`
//!   4. `IOBlockStorageDriver` NVMe MB/s & IOPS + optional `--server <host:port>`
//!      live `/health` attach.

const std = @import("std");
const theme_mod = @import("theme.zig");
const telemetry = @import("../platform/telemetry.zig");
const vmstat = @import("../platform/vmstat.zig");
const transport = @import("../cluster/transport.zig");

pub const Theme = theme_mod.Theme;
pub const Rgb = theme_mod.Rgb;
pub const Gradient3 = theme_mod.Gradient3;

pub const MAX_COLS: u16 = 320;
pub const MAX_ROWS: u16 = 120;
pub const MIN_COLS: u16 = 60;
pub const MIN_ROWS: u16 = 18;
pub const DEFAULT_COLS: u16 = 104;
pub const DEFAULT_ROWS: u16 = 36;
pub const HISTORY_CAP: usize = 256;
pub const OUT_BUF_CAP: usize = 768 * 1024;
pub const JSON_BUF_CAP: usize = 32 * 1024;

pub const DEFAULT_INTERVAL_MS: u32 = 1000;
pub const MIN_LOCAL_INTERVAL_MS: u32 = 250;
pub const MIN_SERVER_INTERVAL_MS: u32 = 500; // ADR-0022 guardrail: >= 500 ms when polling --server
pub const MAX_INTERVAL_MS: u32 = 10_000;
pub const INTERVAL_STEP_MS: u32 = 250;

pub const Error = telemetry.Error || theme_mod.Error || std.mem.Allocator.Error || error{
    InvalidCliArgument,
    MissingCliValue,
    InvalidServerEndpoint,
    BufferTooSmall,
    KqueueFailed,
    WriteFailed,
};

pub const HELP_TEXT: []const u8 =
    \\Usage: aminfer monitor [options]
    \\
    \\Options:
    \\  --once                  Render a single TrueColor telemetry frame and exit
    \\  --json                  Emit a single machine-readable JSON snapshot and exit
    \\  --theme <path.theme>    Load a custom btop .theme file (default: trans-pride)
    \\  --server <host:port>    Attach live HTTP /health polling to aminfer serve
    \\  -i, --interval-ms <ms>  Refresh interval in ms (default: 1000)
    \\  --cols <n>              Override terminal width (useful with --once)
    \\  --rows <n>              Override terminal height (useful with --once)
    \\  -h, --help              Show this help message
    \\
    \\Interactive keys:
    \\  q / Esc     Quit monitor
    \\  + / -       Adjust refresh interval
    \\  1 - 4       Toggle panels (1: CPU/SME2, 2: Memory/R9, 3: GPU/Thermal, 4: NVMe/Serve)
    \\  r           Reset cumulative baseline deltas
    \\
;

// ---------------------------------------------------------------------------
// CLI Configuration
// ---------------------------------------------------------------------------

pub const MonitorConfig = struct {
    interval_ms: u32 = DEFAULT_INTERVAL_MS,
    once: bool = false,
    json: bool = false,
    theme_path: ?[]const u8 = null,
    server_addr: ?[]const u8 = null,
    cols_override: ?u16 = null,
    rows_override: ?u16 = null,
    show_help: bool = false,

    pub fn minIntervalMs(self: MonitorConfig) u32 {
        return if (self.server_addr != null) MIN_SERVER_INTERVAL_MS else MIN_LOCAL_INTERVAL_MS;
    }

    pub fn effectiveIntervalMs(self: MonitorConfig) u32 {
        return std.math.clamp(self.interval_ms, self.minIntervalMs(), MAX_INTERVAL_MS);
    }

    pub fn parseArgs(args: []const []const u8) Error!MonitorConfig {
        var cfg: MonitorConfig = .{};
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            const arg = args[i];
            if (std.mem.eql(u8, arg, "--once")) {
                cfg.once = true;
            } else if (std.mem.eql(u8, arg, "--json")) {
                cfg.json = true;
            } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
                cfg.show_help = true;
            } else if (std.mem.eql(u8, arg, "--theme")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.theme_path = args[i];
            } else if (std.mem.eql(u8, arg, "--server")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                _ = try ServerEndpoint.parse(args[i]);
                cfg.server_addr = args[i];
            } else if (std.mem.eql(u8, arg, "--interval-ms") or std.mem.eql(u8, arg, "-i")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                const ms = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidCliArgument;
                if (ms == 0) return error.InvalidCliArgument;
                cfg.interval_ms = ms;
            } else if (std.mem.eql(u8, arg, "--cols")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.cols_override = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--rows")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.rows_override = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidCliArgument;
            } else {
                return error.InvalidCliArgument;
            }
        }
        cfg.interval_ms = cfg.effectiveIntervalMs();
        return cfg;
    }
};

// ---------------------------------------------------------------------------
// Fixed Buffer Writer & Single-Syscall Flush
// ---------------------------------------------------------------------------

pub const BufWriter = struct {
    buf: []u8,
    pos: usize = 0,

    pub fn init(buf: []u8) BufWriter {
        return .{ .buf = buf, .pos = 0 };
    }

    pub fn written(self: *const BufWriter) []const u8 {
        return self.buf[0..self.pos];
    }

    pub fn writeByte(self: *BufWriter, b: u8) error{BufferTooSmall}!void {
        if (self.pos >= self.buf.len) return error.BufferTooSmall;
        self.buf[self.pos] = b;
        self.pos += 1;
    }

    pub fn writeAll(self: *BufWriter, s: []const u8) error{BufferTooSmall}!void {
        if (s.len > self.buf.len - self.pos) return error.BufferTooSmall;
        @memcpy(self.buf[self.pos .. self.pos + s.len], s);
        self.pos += s.len;
    }

    pub fn print(self: *BufWriter, comptime fmt: []const u8, args: anytype) error{BufferTooSmall}!void {
        const slice = std.fmt.bufPrint(self.buf[self.pos..], fmt, args) catch return error.BufferTooSmall;
        self.pos += slice.len;
    }

    pub fn writeCodepointUtf8(self: *BufWriter, cp: u21) error{BufferTooSmall}!void {
        var utf8_buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &utf8_buf) catch blk: {
            utf8_buf[0] = '?';
            break :blk @as(u3, 1);
        };
        try self.writeAll(utf8_buf[0..n]);
    }

    pub fn writeJsonEscaped(self: *BufWriter, s: []const u8) error{BufferTooSmall}!void {
        const hex = "0123456789abcdef";
        for (s) |b| {
            switch (b) {
                '"' => try self.writeAll("\\\""),
                '\\' => try self.writeAll("\\\\"),
                '\n' => try self.writeAll("\\n"),
                '\r' => try self.writeAll("\\r"),
                '\t' => try self.writeAll("\\t"),
                else => {
                    if (b < 0x20) {
                        const esc = [6]u8{ '\\', 'u', '0', '0', hex[b >> 4], hex[b & 0x0F] };
                        try self.writeAll(&esc);
                    } else {
                        try self.writeByte(b);
                    }
                },
            }
        }
    }
};

pub fn writeAllFd(fd: std.c.fd_t, data: []const u8) error{WriteFailed}!void {
    var offset: usize = 0;
    while (offset < data.len) {
        const rc = std.c.write(fd, data[offset..].ptr, data.len - offset);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.WriteFailed;
        }
        if (rc == 0) return error.WriteFailed;
        offset += @intCast(rc);
    }
}

// ---------------------------------------------------------------------------
// Unicode Braille (`⠀..⣿`, U+2800..U+28FF) & Fixed Time-Series Ring
// ---------------------------------------------------------------------------

/// Encodes bottom-up left (`0..4`) and right (`0..4`) vertical dot columns into
/// a single Unicode Braille character in `U+2800..U+28FF` (`'⠀'..'⣿'`).
pub fn brailleCodepoint(left_dots: u8, right_dots: u8) u21 {
    const left_fill = [5]u8{ 0x00, 0x40, 0x44, 0x46, 0x47 };
    const right_fill = [5]u8{ 0x00, 0x80, 0xA0, 0xB0, 0xB8 };
    const l: usize = @min(left_dots, 4);
    const r: usize = @min(right_dots, 4);
    return @as(u21, 0x2800) + @as(u21, left_fill[l] | right_fill[r]);
}

pub fn HistoryRing(comptime cap: usize) type {
    return struct {
        const Self = @This();
        values: [cap]f32 = @splat(0.0),
        head: usize = 0,
        len: usize = 0,

        pub fn push(self: *Self, val_pct: f32) void {
            const clamped: f32 = if (std.math.isNan(val_pct)) 0.0 else std.math.clamp(val_pct, 0.0, 100.0);
            self.values[self.head] = clamped;
            self.head = (self.head + 1) % cap;
            if (self.len < cap) self.len += 1;
        }

        pub fn clear(self: *Self) void {
            self.values = @splat(0.0);
            self.head = 0;
            self.len = 0;
        }

        /// `age == 0` returns the newest sample; returns `0.0` when `age >= self.len`.
        pub fn getRecent(self: *const Self, age: usize) f32 {
            if (age >= self.len or self.len == 0) return 0.0;
            const idx = (self.head + cap - 1 - age) % cap;
            return self.values[idx];
        }
    };
}

// ---------------------------------------------------------------------------
// Live `aminfer serve` `/health` Telemetry Poller (zero heap allocation)
// ---------------------------------------------------------------------------

pub const ServerEndpoint = struct {
    endpoint: transport.Endpoint,
    label_buf: [64]u8 = @splat(0),
    label_len: u8 = 0,

    pub fn label(self: *const ServerEndpoint) []const u8 {
        return self.label_buf[0..@min(@as(usize, self.label_len), self.label_buf.len)];
    }

    pub fn parse(spec: []const u8) error{InvalidServerEndpoint}!ServerEndpoint {
        const trimmed = std.mem.trim(u8, spec, " \t\r\n");
        const without_scheme = if (std.mem.startsWith(u8, trimmed, "http://"))
            trimmed["http://".len..]
        else
            trimmed;
        const slash = std.mem.indexOfScalar(u8, without_scheme, '/') orelse without_scheme.len;
        const host_port = without_scheme[0..slash];
        const colon = std.mem.lastIndexOfScalar(u8, host_port, ':') orelse return error.InvalidServerEndpoint;
        const host = host_port[0..colon];
        const port_str = host_port[colon + 1 ..];
        if (host.len == 0 or port_str.len == 0) return error.InvalidServerEndpoint;
        const port = std.fmt.parseInt(u16, port_str, 10) catch return error.InvalidServerEndpoint;
        if (port == 0) return error.InvalidServerEndpoint;

        const ip: [4]u8 = if (std.mem.eql(u8, host, "localhost"))
            .{ 127, 0, 0, 1 }
        else
            transport.parseIpv4(host) catch return error.InvalidServerEndpoint;

        var ep: ServerEndpoint = .{
            .endpoint = .{ .host = ip, .port = port },
        };
        const clen = @min(host_port.len, ep.label_buf.len);
        @memcpy(ep.label_buf[0..clen], host_port[0..clen]);
        ep.label_len = @intCast(clen);
        return ep;
    }
};

pub const ServerTelemetry = struct {
    attached: bool = false,
    online: bool = false,
    address_buf: [64]u8 = @splat(0),
    address_len: u8 = 0,
    model_buf: [96]u8 = @splat(0),
    model_len: u8 = 0,
    wired_mib: u64 = 0,
    active_requests: u32 = 0,
    active_batch_size: u32 = 0,
    prefill_slots: u32 = 0,
    decode_slots: u32 = 0,
    mean_batch_size: f64 = 0.0,
    tok_per_s: f64 = 0.0,
    decode_share: f64 = 0.0,
    prefix_hit_tokens: u64 = 0,
    prefix_hit_rate: f64 = 0.0,
    retired_q8_kv_chunks: u64 = 0,
    active_ngram_pack_buf: [64]u8 = @splat(0),
    active_ngram_pack_len: u8 = 0,
    cluster_peers_online: u32 = 0,
    cluster_mode_buf: [32]u8 = @splat(0),
    cluster_mode_len: u8 = 0,

    pub fn address(self: *const ServerTelemetry) []const u8 {
        return self.address_buf[0..@min(@as(usize, self.address_len), self.address_buf.len)];
    }

    pub fn model(self: *const ServerTelemetry) []const u8 {
        return self.model_buf[0..@min(@as(usize, self.model_len), self.model_buf.len)];
    }

    pub fn activeNgramPack(self: *const ServerTelemetry) []const u8 {
        return self.active_ngram_pack_buf[0..@min(@as(usize, self.active_ngram_pack_len), self.active_ngram_pack_buf.len)];
    }

    pub fn clusterMode(self: *const ServerTelemetry) []const u8 {
        return self.cluster_mode_buf[0..@min(@as(usize, self.cluster_mode_len), self.cluster_mode_buf.len)];
    }

    pub fn offlineForEndpoint(ep: *const ServerEndpoint) ServerTelemetry {
        var out: ServerTelemetry = .{ .attached = true, .online = false };
        const lbl = ep.label();
        @memcpy(out.address_buf[0..lbl.len], lbl);
        out.address_len = @intCast(lbl.len);
        return out;
    }

    pub fn parseHealthJson(json_body: []const u8, addr_label: []const u8) ServerTelemetry {
        var out: ServerTelemetry = .{ .attached = true };
        const alen = @min(addr_label.len, out.address_buf.len);
        @memcpy(out.address_buf[0..alen], addr_label[0..alen]);
        out.address_len = @intCast(alen);

        var status_buf: [32]u8 = undefined;
        if (extractJsonString(json_body, "status", &status_buf)) |st| {
            if (std.mem.eql(u8, st, "ok")) out.online = true;
        }
        if (extractJsonString(json_body, "model", &out.model_buf)) |m| {
            out.model_len = @intCast(m.len);
            if (!out.online) out.online = true;
        }
        if (extractJsonU64(json_body, "wired_mib")) |v| out.wired_mib = v;
        if (extractJsonU64(json_body, "active_requests")) |v| {
            out.active_requests = @intCast(@min(v, std.math.maxInt(u32)));
        }
        if (extractJsonU64(json_body, "active_batch_size")) |v| {
            out.active_batch_size = @intCast(@min(v, std.math.maxInt(u32)));
        }
        if (extractJsonU64(json_body, "prefill_slots") orelse extractJsonU64(json_body, "active_prefill_slots")) |v| {
            out.prefill_slots = @intCast(@min(v, std.math.maxInt(u32)));
        }
        if (extractJsonU64(json_body, "decode_slots") orelse extractJsonU64(json_body, "active_decode_slots")) |v| {
            out.decode_slots = @intCast(@min(v, std.math.maxInt(u32)));
        } else {
            out.decode_slots = out.active_batch_size;
        }
        if (extractJsonF64(json_body, "mean_batch_size")) |v| out.mean_batch_size = v;
        if (extractJsonF64(json_body, "tok_per_s") orelse
            extractJsonF64(json_body, "tok_s") orelse
            extractJsonF64(json_body, "throughput_tok_s")) |v|
        {
            out.tok_per_s = v;
        }
        if (extractJsonF64(json_body, "decode_share") orelse extractJsonF64(json_body, "decode_share_qos")) |v| {
            out.decode_share = v;
        }
        if (extractJsonU64(json_body, "prefix_hit_tokens")) |v| out.prefix_hit_tokens = v;
        if (extractJsonF64(json_body, "prefix_hit_rate")) |v| out.prefix_hit_rate = v;
        if (extractJsonU64(json_body, "retired_q8_kv_chunks") orelse extractJsonU64(json_body, "q8_kv_chunks")) |v| {
            out.retired_q8_kv_chunks = v;
        }
        if (extractJsonString(json_body, "active_ngram_pack", &out.active_ngram_pack_buf) orelse
            extractJsonString(json_body, "default_domain", &out.active_ngram_pack_buf)) |p|
        {
            out.active_ngram_pack_len = @intCast(p.len);
        }
        if (extractJsonU64(json_body, "cluster_peers_online") orelse extractJsonU64(json_body, "tb_peers")) |v| {
            out.cluster_peers_online = @intCast(@min(v, std.math.maxInt(u32)));
        }
        if (extractJsonString(json_body, "cluster_mode", &out.cluster_mode_buf) orelse
            extractJsonString(json_body, "tb_transport", &out.cluster_mode_buf)) |cm|
        {
            out.cluster_mode_len = @intCast(cm.len);
        }
        return out;
    }
};

fn findJsonValueAfterKey(json: []const u8, key: []const u8) ?[]const u8 {
    var search_pos: usize = 0;
    while (search_pos + key.len + 2 < json.len) {
        const quote_idx = std.mem.indexOfScalarPos(u8, json, search_pos, '"') orelse return null;
        if (quote_idx + 1 + key.len >= json.len) return null;
        const candidate = json[quote_idx + 1 .. quote_idx + 1 + key.len];
        const close_quote_pos = quote_idx + 1 + key.len;
        if (std.mem.eql(u8, candidate, key) and json[close_quote_pos] == '"') {
            const after = std.mem.trimStart(u8, json[close_quote_pos + 1 ..], " \t\r\n");
            if (after.len > 0 and after[0] == ':') {
                return std.mem.trimStart(u8, after[1..], " \t\r\n");
            }
        }
        search_pos = quote_idx + 1;
    }
    return null;
}

fn extractJsonString(json: []const u8, key: []const u8, out_buf: []u8) ?[]const u8 {
    const val = findJsonValueAfterKey(json, key) orelse return null;
    if (val.len == 0 or val[0] != '"') return null;
    var i: usize = 1;
    var out_len: usize = 0;
    while (i < val.len) : (i += 1) {
        const c = val[i];
        if (c == '"') {
            return out_buf[0..out_len];
        }
        var decoded: u8 = c;
        if (c == '\\') {
            i += 1;
            if (i >= val.len) return null;
            decoded = switch (val[i]) {
                '"' => '"',
                '\\' => '\\',
                '/' => '/',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                else => val[i],
            };
        }
        if (out_len < out_buf.len) {
            out_buf[out_len] = decoded;
            out_len += 1;
        }
    }
    return null;
}

fn extractJsonNumberToken(json: []const u8, key: []const u8) ?[]const u8 {
    const val = findJsonValueAfterKey(json, key) orelse return null;
    var end: usize = 0;
    while (end < val.len) : (end += 1) {
        const ch = val[end];
        const ok = (ch >= '0' and ch <= '9') or ch == '.' or ch == '-' or ch == '+' or ch == 'e' or ch == 'E';
        if (!ok) break;
    }
    if (end == 0) return null;
    return val[0..end];
}

fn extractJsonU64(json: []const u8, key: []const u8) ?u64 {
    const tok = extractJsonNumberToken(json, key) orelse return null;
    return std.fmt.parseInt(u64, tok, 10) catch null;
}

fn extractJsonF64(json: []const u8, key: []const u8) ?f64 {
    const tok = extractJsonNumberToken(json, key) orelse return null;
    return std.fmt.parseFloat(f64, tok) catch null;
}

/// Polls `GET /health` from `ep` over non-blocking TCP + `kqueue` timeout (`0` heap allocs).
/// Returns an `online = false` `ServerTelemetry` snapshot if the server is unreachable.
pub fn pollServerHealth(ep: *const ServerEndpoint, timeout_ms: u32) ServerTelemetry {
    const fd = transport.connectNonBlocking(ep.endpoint, timeout_ms) catch {
        return ServerTelemetry.offlineForEndpoint(ep);
    };
    defer _ = std.c.close(fd);

    var req_buf: [256]u8 = undefined;
    var rw = BufWriter.init(&req_buf);
    rw.print("GET /health HTTP/1.1\r\nHost: {s}\r\nConnection: close\r\n\r\n", .{ep.label()}) catch {
        return ServerTelemetry.offlineForEndpoint(ep);
    };

    const req_bytes = rw.written();
    const kq = transport.kqueueCreate() catch return ServerTelemetry.offlineForEndpoint(ep);
    defer _ = std.c.close(kq);

    const start_ns = vmstat.nowNs() catch return ServerTelemetry.offlineForEndpoint(ep);
    const deadline_ns = start_ns + @as(u64, timeout_ms) * std.time.ns_per_ms;

    var sent: usize = 0;
    while (sent < req_bytes.len) {
        const rc = std.c.write(fd, req_bytes[sent..].ptr, req_bytes.len - sent);
        if (rc > 0) {
            sent += @intCast(rc);
            continue;
        }
        const err = std.c.errno(rc);
        if (err == .INTR) continue;
        if (err == .AGAIN) {
            const now = vmstat.nowNs() catch return ServerTelemetry.offlineForEndpoint(ep);
            if (now >= deadline_ns) return ServerTelemetry.offlineForEndpoint(ep);
            transport.kqueueSetFilter(kq, fd, std.c.EVFILT.WRITE, std.c.EV.ADD | std.c.EV.ENABLE | std.c.EV.ONESHOT, 1) catch {
                return ServerTelemetry.offlineForEndpoint(ep);
            };
            var wevs: [1]std.c.kevent64_s = undefined;
            const n = transport.kqueueWait(kq, &wevs, deadline_ns - now) catch {
                return ServerTelemetry.offlineForEndpoint(ep);
            };
            if (n == 0) return ServerTelemetry.offlineForEndpoint(ep);
            continue;
        }
        return ServerTelemetry.offlineForEndpoint(ep);
    }

    transport.kqueueSetFilter(kq, fd, std.c.EVFILT.READ, std.c.EV.ADD | std.c.EV.ENABLE, 1) catch {
        return ServerTelemetry.offlineForEndpoint(ep);
    };

    var resp_buf: [4096]u8 = undefined;
    var resp_len: usize = 0;
    while (resp_len < resp_buf.len) {
        const now = vmstat.nowNs() catch break;
        if (now >= deadline_ns) break;
        var evs: [1]std.c.kevent64_s = undefined;
        const nev = transport.kqueueWait(kq, &evs, deadline_ns - now) catch break;
        if (nev == 0) break;

        while (resp_len < resp_buf.len) {
            const n = std.c.read(fd, resp_buf[resp_len..].ptr, resp_buf.len - resp_len);
            if (n > 0) {
                resp_len += @intCast(n);
                continue;
            }
            if (n == 0) {
                return decodeHttpHealthResponse(resp_buf[0..resp_len], ep);
            }
            const rerr = std.c.errno(n);
            if (rerr == .INTR) continue;
            if (rerr == .AGAIN) break;
            if (resp_len > 0) return decodeHttpHealthResponse(resp_buf[0..resp_len], ep);
            return ServerTelemetry.offlineForEndpoint(ep);
        }
    }
    return decodeHttpHealthResponse(resp_buf[0..resp_len], ep);
}

fn decodeHttpHealthResponse(raw_http: []const u8, ep: *const ServerEndpoint) ServerTelemetry {
    if (!std.mem.startsWith(u8, raw_http, "HTTP/1.1 200") and
        !std.mem.startsWith(u8, raw_http, "HTTP/1.0 200"))
    {
        return ServerTelemetry.offlineForEndpoint(ep);
    }
    const body_sep = std.mem.indexOf(u8, raw_http, "\r\n\r\n") orelse return ServerTelemetry.offlineForEndpoint(ep);
    const body = raw_http[body_sep + 4 ..];
    if (body.len == 0) return ServerTelemetry.offlineForEndpoint(ep);
    return ServerTelemetry.parseHealthJson(body, ep.label());
}

// ---------------------------------------------------------------------------
// Preallocated Double-Buffered Cell Canvas (`0` per-frame heap allocation)
// ---------------------------------------------------------------------------

pub const Cell = struct {
    codepoint: u21 = ' ',
    fg: Rgb = Rgb.init(0xF7, 0xF7, 0xFC),
    bg: Rgb = Rgb.init(0x11, 0x13, 0x1D),
    bold: bool = false,
    dim: bool = false,

    pub fn eql(a: Cell, b: Cell) bool {
        return a.codepoint == b.codepoint and
            a.bold == b.bold and
            a.dim == b.dim and
            a.fg.eql(b.fg) and
            a.bg.eql(b.bg);
    }
};

pub const Rect = struct {
    x: u16,
    y: u16,
    w: u16,
    h: u16,
};

pub const Style = struct {
    fg: Rgb,
    bg: Rgb,
    bold: bool = false,
    dim: bool = false,
};

pub const Canvas = struct {
    cols: u16,
    rows: u16,
    front: []Cell,
    back: []Cell,
    out_buf: []u8,

    pub fn init(allocator: std.mem.Allocator, initial_cols: u16, initial_rows: u16) std.mem.Allocator.Error!Canvas {
        const cap: usize = @as(usize, MAX_COLS) * @as(usize, MAX_ROWS);
        const front = try allocator.alloc(Cell, cap);
        errdefer allocator.free(front);
        const back = try allocator.alloc(Cell, cap);
        errdefer allocator.free(back);
        const out_buf = try allocator.alloc(u8, OUT_BUF_CAP);

        const c = std.math.clamp(initial_cols, 20, MAX_COLS);
        const r = std.math.clamp(initial_rows, 8, MAX_ROWS);
        @memset(front, .{});
        @memset(back, .{});
        return .{
            .cols = c,
            .rows = r,
            .front = front,
            .back = back,
            .out_buf = out_buf,
        };
    }

    pub fn deinit(self: *Canvas, allocator: std.mem.Allocator) void {
        allocator.free(self.out_buf);
        allocator.free(self.back);
        allocator.free(self.front);
        self.* = undefined;
    }

    pub fn resize(self: *Canvas, new_cols: u16, new_rows: u16) void {
        self.cols = std.math.clamp(new_cols, 20, MAX_COLS);
        self.rows = std.math.clamp(new_rows, 8, MAX_ROWS);
    }

    pub fn clear(self: *Canvas, fg: Rgb, bg: Rgb) void {
        const active_len: usize = @as(usize, self.cols) * @as(usize, self.rows);
        @memset(self.back[0..active_len], .{
            .codepoint = ' ',
            .fg = fg,
            .bg = bg,
            .bold = false,
            .dim = false,
        });
    }

    pub fn putCell(self: *Canvas, x: u16, y: u16, cell: Cell) void {
        if (x >= self.cols or y >= self.rows) return;
        const idx: usize = @as(usize, y) * @as(usize, self.cols) + @as(usize, x);
        self.back[idx] = cell;
    }

    pub fn putChar(self: *Canvas, x: u16, y: u16, cp: u21, style: Style) void {
        self.putCell(x, y, .{
            .codepoint = cp,
            .fg = style.fg,
            .bg = style.bg,
            .bold = style.bold,
            .dim = style.dim,
        });
    }

    /// Writes UTF-8 text horizontally clipped to `max_cols_end`, returning new `x`.
    pub fn putTextClipped(
        self: *Canvas,
        x: u16,
        y: u16,
        max_x_excl: u16,
        text: []const u8,
        style: Style,
    ) u16 {
        if (y >= self.rows) return x;
        const limit = @min(max_x_excl, self.cols);
        var cx = x;
        var i: usize = 0;
        while (i < text.len and cx < limit) {
            const seq_len = std.unicode.utf8ByteSequenceLength(text[i]) catch {
                self.putChar(cx, y, '?', style);
                cx += 1;
                i += 1;
                continue;
            };
            if (i + seq_len > text.len) break;
            const cp = std.unicode.utf8Decode(text[i .. i + seq_len]) catch '?';
            self.putChar(cx, y, cp, style);
            cx += 1;
            i += seq_len;
        }
        return cx;
    }

    pub fn putFmtClipped(
        self: *Canvas,
        x: u16,
        y: u16,
        max_x_excl: u16,
        style: Style,
        comptime fmt: []const u8,
        args: anytype,
    ) u16 {
        var stack_buf: [256]u8 = undefined;
        const s = std.fmt.bufPrint(&stack_buf, fmt, args) catch stack_buf[0..0];
        return self.putTextClipped(x, y, max_x_excl, s, style);
    }

    /// Draws a rounded `btop`-style box with keybadge + title.
    pub fn drawBox(
        self: *Canvas,
        rect: Rect,
        key_digit: u8,
        title_text: []const u8,
        border_fg: Rgb,
        pal: *const Theme,
    ) void {
        if (rect.w < 4 or rect.h < 3) return;
        const x1: u16 = rect.x + rect.w - 1;
        const y1: u16 = rect.y + rect.h - 1;
        const bstyle: Style = .{ .fg = border_fg, .bg = pal.main_bg };

        self.putChar(rect.x, rect.y, '╭', bstyle);
        self.putChar(x1, rect.y, '╮', bstyle);
        self.putChar(rect.x, y1, '╰', bstyle);
        self.putChar(x1, y1, '╯', bstyle);

        var cx: u16 = rect.x + 1;
        while (cx < x1) : (cx += 1) {
            self.putChar(cx, rect.y, '─', bstyle);
            self.putChar(cx, y1, '─', bstyle);
        }
        var cy: u16 = rect.y + 1;
        while (cy < y1) : (cy += 1) {
            self.putChar(rect.x, cy, '│', bstyle);
            self.putChar(x1, cy, '│', bstyle);
        }

        // Embed `┤ <key><title> ├` in top border.
        if (rect.w > 10) {
            var tx: u16 = rect.x + 2;
            self.putChar(tx, rect.y, '┤', bstyle);
            tx += 1;
            self.putChar(tx, rect.y, ' ', .{ .fg = pal.main_fg, .bg = pal.main_bg });
            tx += 1;
            if (key_digit != 0 and tx < x1 - 2) {
                self.putChar(tx, rect.y, key_digit, .{ .fg = pal.hi_fg, .bg = pal.main_bg, .bold = true });
                tx += 1;
                self.putChar(tx, rect.y, ':', .{ .fg = pal.inactive_fg, .bg = pal.main_bg });
                tx += 1;
            }
            tx = self.putTextClipped(tx, rect.y, x1 - 2, title_text, .{
                .fg = pal.title,
                .bg = pal.main_bg,
                .bold = true,
            });
            if (tx + 1 < x1) {
                self.putChar(tx, rect.y, ' ', .{ .fg = pal.main_fg, .bg = pal.main_bg });
                tx += 1;
                self.putChar(tx, rect.y, '├', bstyle);
            }
        }
    }

    /// Draws a horizontal `btop` meter bar (`■` filled with smooth 3-stop gradient,
    /// `·` unfilled in `meter_bg`).
    pub fn drawMeter(
        self: *Canvas,
        x: u16,
        y: u16,
        width: u16,
        pct: f32,
        grad: Gradient3,
        pal: *const Theme,
    ) void {
        if (width == 0 or y >= self.rows) return;
        const clamped: f32 = if (std.math.isNan(pct)) 0.0 else std.math.clamp(pct, 0.0, 100.0);
        const filled_f: f32 = (clamped / 100.0) * @as(f32, @floatFromInt(width));
        const filled: u16 = @intFromFloat(@round(std.math.clamp(filled_f, 0.0, @as(f32, @floatFromInt(width)))));
        var i: u16 = 0;
        while (i < width) : (i += 1) {
            const pos_ratio: f32 = if (width > 1)
                @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(width - 1))
            else
                clamped / 100.0;
            if (i < filled) {
                self.putChar(x + i, y, '■', .{
                    .fg = grad.sample(pos_ratio),
                    .bg = pal.main_bg,
                });
            } else {
                self.putChar(x + i, y, '·', .{
                    .fg = pal.meter_bg,
                    .bg = pal.main_bg,
                });
            }
        }
    }

    /// Renders a multi-row Unicode Braille (`⠀..⣿`) time-series sparkline graph.
    pub fn drawBrailleGraph(
        self: *Canvas,
        rect: Rect,
        history: *const HistoryRing(HISTORY_CAP),
        grad: Gradient3,
        pal: *const Theme,
    ) void {
        if (rect.w == 0 or rect.h == 0) return;
        const total_sub: u32 = @as(u32, rect.h) * 4;
        var cx: u16 = 0;
        while (cx < rect.w) : (cx += 1) {
            const right_age: usize = @as(usize, rect.w - 1 - cx) * 2;
            const left_age: usize = right_age + 1;
            const l_val = history.getRecent(left_age);
            const r_val = history.getRecent(right_age);

            const l_sub = pctToSubpixels(l_val, total_sub);
            const r_sub = pctToSubpixels(r_val, total_sub);

            var ry: u16 = 0;
            while (ry < rect.h) : (ry += 1) {
                const row_from_bottom: u32 = @as(u32, rect.h - 1 - ry);
                const base_sub: u32 = row_from_bottom * 4;
                const l_dots: u8 = @intCast(@min(l_sub -| base_sub, 4));
                const r_dots: u8 = @intCast(@min(r_sub -| base_sub, 4));
                const cp = brailleCodepoint(l_dots, r_dots);
                const row_ratio: f32 = if (rect.h > 1)
                    @as(f32, @floatFromInt(row_from_bottom + 1)) / @as(f32, @floatFromInt(rect.h))
                else
                    @max(l_val, r_val) / 100.0;
                const fg = if (l_dots == 0 and r_dots == 0)
                    pal.meter_bg
                else
                    grad.sample(@max(row_ratio, @max(l_val, r_val) / 100.0));
                self.putChar(rect.x + cx, rect.y + ry, cp, .{ .fg = fg, .bg = pal.main_bg });
            }
        }
    }

    /// Encodes only cells whose `back` differs from `front` (or all cells when
    /// `force_full == true`) into `self.out_buf` with TrueColor SGR and DEC
    /// synchronized-output framing (`\x1b[?2026h ... \x1b[?2026l`).
    /// Zero heap allocations.
    pub fn encodeDiff(self: *Canvas, force_full: bool) error{BufferTooSmall}![]const u8 {
        var w = BufWriter.init(self.out_buf);
        try w.writeAll("\x1b[?2026h");

        var has_cursor = false;
        var cur_x: u16 = 0;
        var cur_y: u16 = 0;

        var has_style = false;
        var cur_fg: Rgb = Rgb.init(0, 0, 0);
        var cur_bg: Rgb = Rgb.init(0, 0, 0);
        var cur_bold = false;
        var cur_dim = false;

        var y: u16 = 0;
        while (y < self.rows) : (y += 1) {
            var x: u16 = 0;
            while (x < self.cols) : (x += 1) {
                const idx: usize = @as(usize, y) * @as(usize, self.cols) + @as(usize, x);
                const cell = self.back[idx];
                if (!force_full and Cell.eql(cell, self.front[idx])) {
                    continue;
                }
                self.front[idx] = cell;

                if (!has_cursor or cur_y != y or cur_x != x) {
                    try w.print("\x1b[{d};{d}H", .{ @as(u32, y) + 1, @as(u32, x) + 1 });
                    has_cursor = true;
                }

                if (!has_style or cur_bold != cell.bold or cur_dim != cell.dim or
                    !cur_fg.eql(cell.fg) or !cur_bg.eql(cell.bg))
                {
                    try w.writeAll("\x1b[0");
                    if (cell.bold) try w.writeAll(";1");
                    if (cell.dim) try w.writeAll(";2");
                    try w.print(";38;2;{d};{d};{d};48;2;{d};{d};{d}m", .{
                        cell.fg.r, cell.fg.g, cell.fg.b,
                        cell.bg.r, cell.bg.g, cell.bg.b,
                    });
                    has_style = true;
                    cur_fg = cell.fg;
                    cur_bg = cell.bg;
                    cur_bold = cell.bold;
                    cur_dim = cell.dim;
                }

                try w.writeCodepointUtf8(cell.codepoint);
                cur_x = x + 1;
                cur_y = y;
            }
        }

        try w.writeAll("\x1b[?2026l");
        return w.written();
    }

    /// Encodes a standalone non-alternate-screen TrueColor frame (`--once` mode),
    /// terminating each row with `\x1b[0m\n` so output scrolls cleanly in terminals
    /// and shell pipelines (`0` heap allocations).
    pub fn encodeOnceFrame(self: *Canvas) error{BufferTooSmall}![]const u8 {
        var w = BufWriter.init(self.out_buf);
        var y: u16 = 0;
        while (y < self.rows) : (y += 1) {
            var has_style = false;
            var cur_fg: Rgb = Rgb.init(0, 0, 0);
            var cur_bg: Rgb = Rgb.init(0, 0, 0);
            var cur_bold = false;
            var cur_dim = false;

            var x: u16 = 0;
            while (x < self.cols) : (x += 1) {
                const idx: usize = @as(usize, y) * @as(usize, self.cols) + @as(usize, x);
                const cell = self.back[idx];
                self.front[idx] = cell;

                if (!has_style or cur_bold != cell.bold or cur_dim != cell.dim or
                    !cur_fg.eql(cell.fg) or !cur_bg.eql(cell.bg))
                {
                    try w.writeAll("\x1b[0");
                    if (cell.bold) try w.writeAll(";1");
                    if (cell.dim) try w.writeAll(";2");
                    try w.print(";38;2;{d};{d};{d};48;2;{d};{d};{d}m", .{
                        cell.fg.r, cell.fg.g, cell.fg.b,
                        cell.bg.r, cell.bg.g, cell.bg.b,
                    });
                    has_style = true;
                    cur_fg = cell.fg;
                    cur_bg = cell.bg;
                    cur_bold = cell.bold;
                    cur_dim = cell.dim;
                }
                try w.writeCodepointUtf8(cell.codepoint);
            }
            try w.writeAll("\x1b[0m\n");
        }
        return w.written();
    }
};

fn pctToSubpixels(pct: f32, total_sub: u32) u32 {
    if (total_sub == 0 or std.math.isNan(pct) or pct <= 0.0) return 0;
    const clamped = std.math.clamp(pct, 0.0, 100.0);
    const scaled: f32 = (clamped / 100.0) * @as(f32, @floatFromInt(total_sub));
    var sub: u32 = @intFromFloat(@round(scaled));
    if (sub == 0 and clamped >= 0.5) sub = 1;
    return @min(sub, total_sub);
}

fn fmtGiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

fn fmtMiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

// ---------------------------------------------------------------------------
// Interactive State Machine & 4-Panel `btop` Renderer
// ---------------------------------------------------------------------------

pub const KeyAction = enum {
    none,
    quit,
    interval_changed,
    panel_toggled,
    reset_deltas,
};

pub const MonitorState = struct {
    palette: Theme = Theme.TRANS_PRIDE,
    interval_ms: u32 = DEFAULT_INTERVAL_MS,
    min_interval_ms: u32 = MIN_LOCAL_INTERVAL_MS,
    /// Bitmask for panels `1..4` (`bit 0 = CPU`, `bit 1 = Mem`, `bit 2 = GPU`, `bit 3 = NVMe/Serve`).
    panel_mask: u8 = 0x0F,
    server_endpoint: ?ServerEndpoint = null,

    cpu_hist: HistoryRing(HISTORY_CAP) = .{},
    pcore_hist: HistoryRing(HISTORY_CAP) = .{},
    ecore_hist: HistoryRing(HISTORY_CAP) = .{},
    mem_hist: HistoryRing(HISTORY_CAP) = .{},
    gpu_hist: HistoryRing(HISTORY_CAP) = .{},
    nvme_hist: HistoryRing(HISTORY_CAP) = .{},
    serve_hist: HistoryRing(HISTORY_CAP) = .{},

    peak_nvme_mb_s: f64 = 100.0,
    peak_tok_s: f64 = 50.0,

    pub fn init(cfg: MonitorConfig) Error!MonitorState {
        const pal = if (cfg.theme_path) |p| try Theme.loadFile(p) else Theme.TRANS_PRIDE;
        const ep = if (cfg.server_addr) |s| try ServerEndpoint.parse(s) else null;
        return .{
            .palette = pal,
            .interval_ms = cfg.effectiveIntervalMs(),
            .min_interval_ms = cfg.minIntervalMs(),
            .server_endpoint = ep,
        };
    }

    pub fn isPanelVisible(self: *const MonitorState, panel_idx_0_to_3: u2) bool {
        return (self.panel_mask & (@as(u8, 1) << panel_idx_0_to_3)) != 0;
    }

    pub fn visiblePanelCount(self: *const MonitorState) u8 {
        return @popCount(self.panel_mask & 0x0F);
    }

    pub fn recordSnapshot(
        self: *MonitorState,
        host: *const telemetry.HostSnapshot,
        srv: *const ServerTelemetry,
    ) void {
        self.cpu_hist.push(host.cpu.total_pct);
        self.pcore_hist.push(host.cpu.p_core_pct);
        self.ecore_hist.push(host.cpu.e_core_pct);
        self.mem_hist.push(host.mem.wiredRatioPct());
        self.gpu_hist.push(@floatFromInt(host.gpu.device_utilization_pct));

        const total_nvme_mb_s = host.nvme.read_mb_s + host.nvme.write_mb_s;
        if (total_nvme_mb_s > self.peak_nvme_mb_s) self.peak_nvme_mb_s = total_nvme_mb_s;
        const nvme_pct: f32 = if (self.peak_nvme_mb_s > 0.0)
            @floatCast(std.math.clamp((total_nvme_mb_s / self.peak_nvme_mb_s) * 100.0, 0.0, 100.0))
        else
            0.0;
        self.nvme_hist.push(nvme_pct);

        if (srv.online) {
            if (srv.tok_per_s > self.peak_tok_s) self.peak_tok_s = srv.tok_per_s;
            const tok_pct: f32 = if (self.peak_tok_s > 0.0)
                @floatCast(std.math.clamp((srv.tok_per_s / self.peak_tok_s) * 100.0, 0.0, 100.0))
            else
                0.0;
            self.serve_hist.push(tok_pct);
        } else {
            self.serve_hist.push(0.0);
        }
    }

    /// Processes raw keyboard input bytes (`q`/`Esc`/`Ctrl-C`, `+`/`-`, `1`–`4`, `r`).
    /// Standalone `Esc` (`0x1B` with no trailing `[`/`O` CSI/SS3 bytes) quits cleanly,
    /// while arrow/function escape sequences (`\x1b[...`) are consumed without quitting.
    pub fn handleInputBytes(self: *MonitorState, input: []const u8) KeyAction {
        var action: KeyAction = .none;
        var i: usize = 0;
        while (i < input.len) {
            const b = input[i];
            if (b == 0x1B) {
                if (i + 1 < input.len and (input[i + 1] == '[' or input[i + 1] == 'O')) {
                    // Skip ANSI CSI / SS3 sequence so arrow keys never trigger Esc quit.
                    i += 2;
                    while (i < input.len and input[i] >= 0x20 and input[i] <= 0x3F) : (i += 1) {}
                    if (i < input.len) i += 1;
                    continue;
                }
                return .quit;
            }
            i += 1;
            switch (b) {
                'q', 'Q', 0x03 => return .quit,
                '+', '=' => {
                    self.interval_ms = @min(self.interval_ms + INTERVAL_STEP_MS, MAX_INTERVAL_MS);
                    action = .interval_changed;
                },
                '-', '_' => {
                    self.interval_ms = if (self.interval_ms > self.min_interval_ms + INTERVAL_STEP_MS)
                        self.interval_ms - INTERVAL_STEP_MS
                    else
                        self.min_interval_ms;
                    action = .interval_changed;
                },
                '1'...'4' => {
                    const bit: u8 = @as(u8, 1) << @as(u3, @intCast(b - '1'));
                    const candidate = self.panel_mask ^ bit;
                    if ((candidate & 0x0F) != 0) {
                        self.panel_mask = candidate;
                        action = .panel_toggled;
                    }
                },
                'r', 'R' => {
                    self.peak_nvme_mb_s = 100.0;
                    self.peak_tok_s = 50.0;
                    action = .reset_deltas;
                },
                else => {},
            }
        }
        return action;
    }

    /// Renders the header bar and all visible panels into `canvas.back` (`0` heap allocs).
    pub fn render(
        self: *const MonitorState,
        canvas: *Canvas,
        host: *const telemetry.HostSnapshot,
        srv: *const ServerTelemetry,
    ) void {
        const pal = &self.palette;
        canvas.clear(pal.main_fg, pal.main_bg);

        // Top status/keybinding header bar (row 0).
        self.renderTopBar(canvas, host);

        if (canvas.rows < 8 or canvas.cols < 32) {
            _ = canvas.putTextClipped(1, 2, canvas.cols -| 1, "Terminal window too small — resize to >= 60x18", .{
                .fg = pal.title,
                .bg = pal.main_bg,
                .bold = true,
            });
            return;
        }

        // Compute dynamic panel rectangles for visible subset of panels 1..4.
        var active_ids: [4]u2 = undefined;
        var active_count: usize = 0;
        for (0..4) |idx| {
            const p: u2 = @intCast(idx);
            if (self.isPanelVisible(p)) {
                active_ids[active_count] = p;
                active_count += 1;
            }
        }
        if (active_count == 0) return;

        const body_y: u16 = 1;
        const body_h: u16 = canvas.rows - body_y;
        const left_w: u16 = canvas.cols / 2;
        const right_w: u16 = canvas.cols - left_w;
        const top_h: u16 = body_h / 2;
        const bot_h: u16 = body_h - top_h;

        var rects: [4]Rect = undefined;
        switch (active_count) {
            1 => {
                rects[0] = .{ .x = 0, .y = body_y, .w = canvas.cols, .h = body_h };
            },
            2 => {
                rects[0] = .{ .x = 0, .y = body_y, .w = canvas.cols, .h = top_h };
                rects[1] = .{ .x = 0, .y = body_y + top_h, .w = canvas.cols, .h = bot_h };
            },
            3 => {
                rects[0] = .{ .x = 0, .y = body_y, .w = canvas.cols, .h = top_h };
                rects[1] = .{ .x = 0, .y = body_y + top_h, .w = left_w, .h = bot_h };
                rects[2] = .{ .x = left_w, .y = body_y + top_h, .w = right_w, .h = bot_h };
            },
            else => {
                rects[0] = .{ .x = 0, .y = body_y, .w = left_w, .h = top_h };
                rects[1] = .{ .x = left_w, .y = body_y, .w = right_w, .h = top_h };
                rects[2] = .{ .x = 0, .y = body_y + top_h, .w = left_w, .h = bot_h };
                rects[3] = .{ .x = left_w, .y = body_y + top_h, .w = right_w, .h = bot_h };
            },
        }

        for (0..active_count) |slot| {
            const r = rects[slot];
            switch (active_ids[slot]) {
                0 => self.renderCpuPanel(canvas, r, host),
                1 => self.renderMemPanel(canvas, r, host),
                2 => self.renderGpuPanel(canvas, r, host),
                3 => self.renderNvmeServePanel(canvas, r, host, srv),
            }
        }
    }

    fn renderTopBar(self: *const MonitorState, canvas: *Canvas, host: *const telemetry.HostSnapshot) void {
        const pal = &self.palette;
        var x: u16 = 1;
        x = canvas.putTextClipped(x, 0, canvas.cols, "aminfer monitor ", .{
            .fg = pal.title,
            .bg = pal.main_bg,
            .bold = true,
        });
        x = canvas.putFmtClipped(
            x,
            0,
            canvas.cols,
            .{ .fg = pal.main_fg, .bg = pal.main_bg },
            "{s} {s} · {d}P+{d}E · SME2:{s} · {d:.0}GB/s ",
            .{
                host.chip.generation.label(),
                host.chip.tier.label(),
                host.chip.p_cores,
                host.chip.e_cores,
                if (host.facts.supportsSme2Engine()) "ON" else "OFF",
                host.chip.spec_bus_gbs,
            },
        );

        var rhs_buf: [96]u8 = undefined;
        const rhs = std.fmt.bufPrint(
            &rhs_buf,
            "+/- {d}ms  1-4 panels  r reset  q quit",
            .{self.interval_ms},
        ) catch "";
        if (rhs.len + 2 < canvas.cols and canvas.cols - @as(u16, @intCast(rhs.len)) - 1 > x) {
            const rx: u16 = canvas.cols - @as(u16, @intCast(rhs.len)) - 1;
            _ = canvas.putTextClipped(rx, 0, canvas.cols, rhs, .{
                .fg = pal.inactive_fg,
                .bg = pal.main_bg,
            });
        }
    }

    fn renderCpuPanel(self: *const MonitorState, canvas: *Canvas, r: Rect, host: *const telemetry.HostSnapshot) void {
        const pal = &self.palette;
        canvas.drawBox(r, '1', "cpu & sme2", pal.cpu_box, pal);
        if (r.w < 16 or r.h < 4) return;

        const ix: u16 = r.x + 2;
        const max_x: u16 = r.x + r.w - 2;
        const inner_w: u16 = max_x - ix;
        var y: u16 = r.y + 1;
        const max_y: u16 = r.y + r.h - 1;

        // Chip badge + loadavg row.
        var cx = canvas.putFmtClipped(
            ix,
            y,
            max_x,
            .{ .fg = pal.hi_fg, .bg = pal.main_bg, .bold = true },
            "{s}",
            .{host.deviceName()},
        );
        cx = canvas.putFmtClipped(
            cx,
            y,
            max_x,
            .{ .fg = pal.inactive_fg, .bg = pal.main_bg },
            "  SVL:{d}B  load {d:.2} {d:.2} {d:.2}",
            .{
                host.facts.sme_max_svl_bytes,
                host.cpu.loadavg_1m,
                host.cpu.loadavg_5m,
                host.cpu.loadavg_15m,
            },
        );
        y += 1;
        if (y >= max_y) return;

        // Total / P-cluster / E-cluster summary + meter.
        cx = canvas.putFmtClipped(
            ix,
            y,
            max_x,
            .{ .fg = pal.main_fg, .bg = pal.main_bg, .bold = true },
            "CPU {d:5.1}% ",
            .{host.cpu.total_pct},
        );
        const meter_w: u16 = if (max_x > cx + 22) max_x - cx - 21 else 0;
        if (meter_w > 0) {
            canvas.drawMeter(cx, y, meter_w, host.cpu.total_pct, pal.cpu, pal);
            cx += meter_w + 1;
        }
        _ = canvas.putFmtClipped(
            cx,
            y,
            max_x,
            .{ .fg = pal.proc_misc, .bg = pal.main_bg },
            "P:{d:4.1}% E:{d:4.1}%",
            .{ host.cpu.p_core_pct, host.cpu.e_core_pct },
        );
        y += 1;

        // Reserve bottom rows (`>= 2` if space permits) for Braille CPU history.
        const graph_h: u16 = if (max_y > y + 4) @min(@as(u16, 4), (max_y - y) / 2) else if (max_y > y) 1 else 0;
        const cores_end_y: u16 = max_y -| graph_h;

        // Per-core 2-column compact meters (`E0..` and `P6..`).
        const core_count: usize = host.cpu.core_count;
        const col_w: u16 = inner_w / 2;
        var core_idx: usize = 0;
        while (y < cores_end_y and core_idx < core_count) : (y += 1) {
            for (0..2) |col| {
                if (core_idx >= core_count) break;
                const col_x: u16 = ix + @as(u16, @intCast(col)) * col_w;
                const col_limit: u16 = if (col == 0) ix + col_w -| 1 else max_x;
                if (col_limit <= col_x + 10) continue;

                const kind = host.cpu.per_core_kind[core_idx];
                const pct = host.cpu.per_core_pct[core_idx];
                const label_fg = if (kind == .performance) pal.title else pal.hi_fg;
                var px = canvas.putFmtClipped(
                    col_x,
                    y,
                    col_limit,
                    .{ .fg = label_fg, .bg = pal.main_bg, .bold = kind == .performance },
                    "{s}{d:<2}",
                    .{ kind.shortLabel(), core_idx },
                );
                const bar_w: u16 = if (col_limit > px + 7) col_limit - px - 6 else 0;
                if (bar_w > 0) {
                    canvas.drawMeter(px + 1, y, bar_w, pct, pal.cpu, pal);
                    px += 1 + bar_w;
                }
                _ = canvas.putFmtClipped(
                    px,
                    y,
                    col_limit,
                    .{ .fg = pal.graph_text, .bg = pal.main_bg },
                    "{d:5.1}%",
                    .{pct},
                );
                core_idx += 1;
            }
        }

        if (graph_h > 0 and max_y > graph_h) {
            canvas.drawBrailleGraph(
                .{ .x = ix, .y = max_y - graph_h, .w = inner_w, .h = graph_h },
                &self.cpu_hist,
                pal.cpu,
                pal,
            );
        }
    }

    fn renderMemPanel(self: *const MonitorState, canvas: *Canvas, r: Rect, host: *const telemetry.HostSnapshot) void {
        const pal = &self.palette;
        canvas.drawBox(r, '2', "unified memory & r9 pool", pal.mem_box, pal);
        if (r.w < 16 or r.h < 4) return;

        const ix: u16 = r.x + 2;
        const max_x: u16 = r.x + r.w - 2;
        const inner_w: u16 = max_x - ix;
        var y: u16 = r.y + 1;
        const max_y: u16 = r.y + r.h - 1;

        const mem = &host.mem;
        // Row 1: Total RAM + ADR-0012 Wired vs Ceiling.
        const wired_pct = mem.wiredRatioPct();
        var cx = canvas.putFmtClipped(
            ix,
            y,
            max_x,
            .{ .fg = pal.main_fg, .bg = pal.main_bg, .bold = true },
            "Wired {d:5.2}/{d:5.2} GiB ",
            .{ fmtGiB(mem.wired_bytes), fmtGiB(mem.wired_budget_bytes) },
        );
        if (max_x > cx + 7) {
            const mw = max_x - cx - 6;
            canvas.drawMeter(cx, y, mw, wired_pct, pal.used, pal);
            cx += mw;
        }
        _ = canvas.putFmtClipped(cx, y, max_x, .{ .fg = pal.graph_text, .bg = pal.main_bg }, "{d:5.1}%", .{wired_pct});
        y += 1;
        if (y >= max_y) return;

        // Row 2: R9 Zero-Swap Reclaimable (`free + spec + purge + ext`).
        const recl_pct: f32 = if (mem.total_bytes > 0)
            @floatCast(std.math.clamp(
                (@as(f64, @floatFromInt(mem.reclaimable_zero_swap_bytes)) * 100.0) /
                    @as(f64, @floatFromInt(mem.total_bytes)),
                0.0,
                100.0,
            ))
        else
            0.0;
        cx = canvas.putFmtClipped(
            ix,
            y,
            max_x,
            .{ .fg = pal.hi_fg, .bg = pal.main_bg },
            "R9 Pool {d:5.2}/{d:4.0} GiB ",
            .{ fmtGiB(mem.reclaimable_zero_swap_bytes), fmtGiB(mem.total_bytes) },
        );
        if (max_x > cx + 7) {
            const mw = max_x - cx - 6;
            canvas.drawMeter(cx, y, mw, recl_pct, pal.available, pal);
            cx += mw;
        }
        _ = canvas.putFmtClipped(cx, y, max_x, .{ .fg = pal.graph_text, .bg = pal.main_bg }, "{d:5.1}%", .{recl_pct});
        y += 1;

        if (y < max_y) {
            _ = canvas.putFmtClipped(
                ix,
                y,
                max_x,
                .{ .fg = pal.inactive_fg, .bg = pal.main_bg },
                "free {d:.2}G  ext {d:.2}G  purge {d:.2}G  spec {d:.2}G",
                .{
                    fmtGiB(mem.free_bytes),
                    fmtGiB(mem.external_bytes),
                    fmtGiB(mem.purgeable_bytes),
                    fmtGiB(mem.speculative_bytes),
                },
            );
            y += 1;
        }
        if (y < max_y) {
            _ = canvas.putFmtClipped(
                ix,
                y,
                max_x,
                .{ .fg = pal.main_fg, .bg = pal.main_bg },
                "act {d:.2}G  inact {d:.2}G  comp {d:.2}G  cap:{s}",
                .{
                    fmtGiB(mem.active_bytes),
                    fmtGiB(mem.inactive_bytes),
                    fmtGiB(mem.compressor_bytes),
                    mem.binding_limit.label(),
                },
            );
            y += 1;
        }
        if (y < max_y) {
            const swap_fg = if (mem.swapouts_delta == 0) pal.hi_fg else pal.title;
            _ = canvas.putFmtClipped(
                ix,
                y,
                max_x,
                .{ .fg = swap_fg, .bg = pal.main_bg, .bold = mem.swapouts_delta > 0 },
                "swap {d:.1}/{d:.1} MiB  swapouts Δ:{d} (tot {d})",
                .{
                    fmtMiB(mem.swap_used_bytes),
                    fmtMiB(mem.swap_total_bytes),
                    mem.swapouts_delta,
                    mem.swapouts_total,
                },
            );
            y += 1;
        }

        if (max_y > y) {
            const gh: u16 = max_y - y;
            canvas.drawBrailleGraph(
                .{ .x = ix, .y = y, .w = inner_w, .h = gh },
                &self.mem_hist,
                pal.used,
                pal,
            );
        }
    }

    fn renderGpuPanel(self: *const MonitorState, canvas: *Canvas, r: Rect, host: *const telemetry.HostSnapshot) void {
        const pal = &self.palette;
        canvas.drawBox(r, '3', "metal gpu & thermal", pal.net_box, pal);
        if (r.w < 16 or r.h < 4) return;

        const ix: u16 = r.x + 2;
        const max_x: u16 = r.x + r.w - 2;
        const inner_w: u16 = max_x - ix;
        var y: u16 = r.y + 1;
        const max_y: u16 = r.y + r.h - 1;

        const gpu = &host.gpu;
        const therm_fg = pal.temp.sample(gpu.thermal_state.severityRatio());
        var cx = canvas.putFmtClipped(
            ix,
            y,
            max_x,
            .{ .fg = pal.hi_fg, .bg = pal.main_bg, .bold = true },
            "{s} (Apple{d})  ",
            .{
                if (gpu.acceleratorClass().len > 0) gpu.acceleratorClass() else "IOAccelerator",
                host.chip.gpu_apple_family,
            },
        );
        _ = canvas.putFmtClipped(
            cx,
            y,
            max_x,
            .{ .fg = therm_fg, .bg = pal.main_bg, .bold = true },
            "Thermal: {s}  NAX:{s}",
            .{ gpu.thermal_state.label(), if (host.chip.has_nax) "ON" else "OFF" },
        );
        y += 1;
        if (y >= max_y) return;

        const dev_pct: f32 = @floatFromInt(gpu.device_utilization_pct);
        cx = canvas.putFmtClipped(ix, y, max_x, .{ .fg = pal.main_fg, .bg = pal.main_bg, .bold = true }, "Device   ", .{});
        if (max_x > cx + 6) {
            const mw = max_x - cx - 5;
            canvas.drawMeter(cx, y, mw, dev_pct, pal.process, pal);
            cx += mw;
        }
        _ = canvas.putFmtClipped(cx, y, max_x, .{ .fg = pal.graph_text, .bg = pal.main_bg }, "{d:3}%", .{gpu.device_utilization_pct});
        y += 1;

        if (y < max_y) {
            const ren_pct: f32 = @floatFromInt(gpu.renderer_utilization_pct);
            const til_pct: f32 = @floatFromInt(gpu.tiler_utilization_pct);
            const half_w: u16 = inner_w / 2;
            var lx = canvas.putTextClipped(ix, y, ix + half_w, "Render ", .{ .fg = pal.inactive_fg, .bg = pal.main_bg });
            if (ix + half_w > lx + 6) {
                const bw = (ix + half_w) - lx - 5;
                canvas.drawMeter(lx, y, bw, ren_pct, pal.process, pal);
                lx += bw;
            }
            _ = canvas.putFmtClipped(lx, y, ix + half_w, .{ .fg = pal.main_fg, .bg = pal.main_bg }, "{d:3}%", .{gpu.renderer_utilization_pct});

            var rx = canvas.putTextClipped(ix + half_w + 1, y, max_x, "Tiler ", .{ .fg = pal.inactive_fg, .bg = pal.main_bg });
            if (max_x > rx + 5) {
                const bw = max_x - rx - 4;
                canvas.drawMeter(rx, y, bw, til_pct, pal.process, pal);
                rx += bw;
            }
            _ = canvas.putFmtClipped(rx, y, max_x, .{ .fg = pal.main_fg, .bg = pal.main_bg }, "{d:3}%", .{gpu.tiler_utilization_pct});
            y += 1;
        }

        if (y < max_y) {
            _ = canvas.putFmtClipped(
                ix,
                y,
                max_x,
                .{ .fg = pal.main_fg, .bg = pal.main_bg },
                "GPU SysMem in-use {d:.2} GiB / alloc {d:.2} GiB",
                .{ fmtGiB(gpu.in_use_system_memory), fmtGiB(gpu.alloc_system_memory) },
            );
            y += 1;
        }

        if (max_y > y) {
            canvas.drawBrailleGraph(
                .{ .x = ix, .y = y, .w = inner_w, .h = max_y - y },
                &self.gpu_hist,
                pal.process,
                pal,
            );
        }
    }

    fn renderNvmeServePanel(
        self: *const MonitorState,
        canvas: *Canvas,
        r: Rect,
        host: *const telemetry.HostSnapshot,
        srv: *const ServerTelemetry,
    ) void {
        const pal = &self.palette;
        canvas.drawBox(r, '4', "nvme & serve/cluster", pal.proc_box, pal);
        if (r.w < 16 or r.h < 4) return;

        const ix: u16 = r.x + 2;
        const max_x: u16 = r.x + r.w - 2;
        const inner_w: u16 = max_x - ix;
        var y: u16 = r.y + 1;
        const max_y: u16 = r.y + r.h - 1;

        const nv = &host.nvme;
        _ = canvas.putFmtClipped(
            ix,
            y,
            max_x,
            .{ .fg = pal.download.sample(0.3), .bg = pal.main_bg, .bold = true },
            "NVMe R {d:6.1} MB/s ({d:5.0} IOPS)  ",
            .{ nv.read_mb_s, nv.read_iops },
        );
        if (inner_w > 33) {
            _ = canvas.putFmtClipped(
                ix + 33,
                y,
                max_x,
                .{ .fg = pal.upload.sample(0.3), .bg = pal.main_bg, .bold = true },
                "W {d:6.1} MB/s ({d:5.0} IOPS)",
                .{ nv.write_mb_s, nv.write_iops },
            );
        }
        y += 1;

        if (y < max_y) {
            _ = canvas.putFmtClipped(
                ix,
                y,
                max_x,
                .{ .fg = pal.inactive_fg, .bg = pal.main_bg },
                "ΔR {d:.1} MiB ({d} ops) · ΔW {d:.1} MiB ({d} ops)",
                .{
                    fmtMiB(nv.bytes_read_delta),
                    nv.ops_read_delta,
                    fmtMiB(nv.bytes_written_delta),
                    nv.ops_written_delta,
                },
            );
            y += 1;
        }

        // Divider before live `aminfer serve` / cluster attach section.
        if (y < max_y) {
            var dx: u16 = ix;
            while (dx < max_x) : (dx += 1) {
                canvas.putChar(dx, y, '─', .{ .fg = pal.div_line, .bg = pal.main_bg });
            }
            y += 1;
        }

        if (!srv.attached) {
            if (y < max_y) {
                _ = canvas.putTextClipped(
                    ix,
                    y,
                    max_x,
                    "serve: detached (--server 127.0.0.1:8080 to attach)",
                    .{ .fg = pal.inactive_fg, .bg = pal.main_bg },
                );
                y += 1;
            }
        } else if (!srv.online) {
            if (y < max_y) {
                _ = canvas.putFmtClipped(
                    ix,
                    y,
                    max_x,
                    .{ .fg = pal.title, .bg = pal.main_bg, .bold = true },
                    "serve [{s}] OFFLINE — awaiting /health",
                    .{srv.address()},
                );
                y += 1;
            }
        } else {
            if (y < max_y) {
                _ = canvas.putFmtClipped(
                    ix,
                    y,
                    max_x,
                    .{ .fg = pal.hi_fg, .bg = pal.main_bg, .bold = true },
                    "serve [{s}] {s} · {d} MiB wired · {d:.1} tok/s",
                    .{
                        srv.address(),
                        if (srv.model().len > 0) srv.model() else "active",
                        srv.wired_mib,
                        srv.tok_per_s,
                    },
                );
                y += 1;
            }
            if (y < max_y) {
                _ = canvas.putFmtClipped(
                    ix,
                    y,
                    max_x,
                    .{ .fg = pal.main_fg, .bg = pal.main_bg },
                    "req:{d} batch:{d} (P:{d}/D:{d}) mean:{d:.2} qos:{d:.2}",
                    .{
                        srv.active_requests,
                        srv.active_batch_size,
                        srv.prefill_slots,
                        srv.decode_slots,
                        srv.mean_batch_size,
                        srv.decode_share,
                    },
                );
                y += 1;
            }
            if (y < max_y) {
                _ = canvas.putFmtClipped(
                    ix,
                    y,
                    max_x,
                    .{ .fg = pal.proc_misc, .bg = pal.main_bg },
                    "prefix:{d}tok q8kv:{d} .ang:{s} peers:{d}",
                    .{
                        srv.prefix_hit_tokens,
                        srv.retired_q8_kv_chunks,
                        if (srv.activeNgramPack().len > 0) srv.activeNgramPack() else "-",
                        srv.cluster_peers_online,
                    },
                );
                y += 1;
            }
        }

        if (max_y > y) {
            const hist = if (srv.online and srv.tok_per_s > 0.0) &self.serve_hist else &self.nvme_hist;
            canvas.drawBrailleGraph(
                .{ .x = ix, .y = y, .w = inner_w, .h = max_y - y },
                hist,
                pal.download,
                pal,
            );
        }
    }
};

// ---------------------------------------------------------------------------
// Deterministic `--json` Snapshot Serializer (`0` heap allocations)
// ---------------------------------------------------------------------------

pub fn writeJsonSnapshot(
    out_buf: []u8,
    pal: *const Theme,
    host: *const telemetry.HostSnapshot,
    srv: *const ServerTelemetry,
) error{BufferTooSmall}![]const u8 {
    var w = BufWriter.init(out_buf);
    try w.writeAll("{\"theme\":\"");
    try w.writeJsonEscaped(pal.name());
    try w.writeAll("\",\"chip\":{\"generation\":\"");
    try w.writeJsonEscaped(host.chip.generation.label());
    try w.writeAll("\",\"tier\":\"");
    try w.writeJsonEscaped(host.chip.tier.label());
    try w.writeAll("\",\"device_name\":\"");
    try w.writeJsonEscaped(host.deviceName());
    try w.print(
        "\",\"gpu_apple_family\":{d},\"p_cores\":{d},\"e_cores\":{d},\"has_sme2\":{s},\"has_nax\":{s},\"spec_bus_gbs\":{d:.1}}},",
        .{
            host.chip.gpu_apple_family,
            host.chip.p_cores,
            host.chip.e_cores,
            if (host.facts.supportsSme2Engine()) "true" else "false",
            if (host.chip.has_nax) "true" else "false",
            host.chip.spec_bus_gbs,
        },
    );

    try w.print(
        "\"cpu\":{{\"core_count\":{d},\"p_core_count\":{d},\"e_core_count\":{d},\"total_pct\":{d:.2},\"p_core_pct\":{d:.2},\"e_core_pct\":{d:.2},\"loadavg\":[{d:.2},{d:.2},{d:.2}],\"per_core_pct\":[",
        .{
            host.cpu.core_count,
            host.cpu.p_core_count,
            host.cpu.e_core_count,
            host.cpu.total_pct,
            host.cpu.p_core_pct,
            host.cpu.e_core_pct,
            host.cpu.loadavg_1m,
            host.cpu.loadavg_5m,
            host.cpu.loadavg_15m,
        },
    );
    for (0..host.cpu.core_count) |i| {
        if (i > 0) try w.writeByte(',');
        try w.print("{d:.2}", .{host.cpu.per_core_pct[i]});
    }
    try w.writeAll("]},");

    const mem = &host.mem;
    try w.print(
        "\"memory\":{{\"total_bytes\":{d},\"page_size\":{d},\"wired_bytes\":{d},\"wired_budget_bytes\":{d},\"binding_limit\":\"",
        .{ mem.total_bytes, mem.page_size, mem.wired_bytes, mem.wired_budget_bytes },
    );
    try w.writeJsonEscaped(mem.binding_limit.label());
    try w.print(
        "\",\"user_wire_limit_bytes\":{d},\"gpu_wired_limit_bytes\":{d},\"reclaimable_zero_swap_bytes\":{d},\"free_bytes\":{d},\"speculative_bytes\":{d},\"purgeable_bytes\":{d},\"external_bytes\":{d},\"active_bytes\":{d},\"inactive_bytes\":{d},\"compressor_bytes\":{d},\"swap_used_bytes\":{d},\"swap_total_bytes\":{d},\"swapouts_total\":{d},\"swapouts_delta\":{d}}},",
        .{
            mem.user_wire_limit_bytes,
            mem.gpu_wired_limit_bytes,
            mem.reclaimable_zero_swap_bytes,
            mem.free_bytes,
            mem.speculative_bytes,
            mem.purgeable_bytes,
            mem.external_bytes,
            mem.active_bytes,
            mem.inactive_bytes,
            mem.compressor_bytes,
            mem.swap_used_bytes,
            mem.swap_total_bytes,
            mem.swapouts_total,
            mem.swapouts_delta,
        },
    );

    const gpu = &host.gpu;
    try w.print(
        "\"gpu\":{{\"available\":{s},\"accelerator_class\":\"",
        .{if (gpu.available) "true" else "false"},
    );
    try w.writeJsonEscaped(gpu.acceleratorClass());
    try w.writeAll("\",\"thermal_state\":\"");
    try w.writeJsonEscaped(gpu.thermal_state.label());
    try w.print(
        "\",\"device_utilization_pct\":{d},\"renderer_utilization_pct\":{d},\"tiler_utilization_pct\":{d},\"in_use_system_memory_bytes\":{d},\"alloc_system_memory_bytes\":{d}}},",
        .{
            gpu.device_utilization_pct,
            gpu.renderer_utilization_pct,
            gpu.tiler_utilization_pct,
            gpu.in_use_system_memory,
            gpu.alloc_system_memory,
        },
    );

    const nv = &host.nvme;
    try w.print(
        "\"nvme\":{{\"available\":{s},\"driver_count\":{d},\"bytes_read_total\":{d},\"bytes_written_total\":{d},\"ops_read_total\":{d},\"ops_written_total\":{d},\"read_mb_s\":{d:.2},\"write_mb_s\":{d:.2},\"read_iops\":{d:.2},\"write_iops\":{d:.2}}},",
        .{
            if (nv.available) "true" else "false",
            nv.driver_count,
            nv.bytes_read_total,
            nv.bytes_written_total,
            nv.ops_read_total,
            nv.ops_written_total,
            nv.read_mb_s,
            nv.write_mb_s,
            nv.read_iops,
            nv.write_iops,
        },
    );

    try w.print(
        "\"server\":{{\"attached\":{s},\"online\":{s},\"address\":\"",
        .{
            if (srv.attached) "true" else "false",
            if (srv.online) "true" else "false",
        },
    );
    try w.writeJsonEscaped(srv.address());
    try w.writeAll("\",\"model\":\"");
    try w.writeJsonEscaped(srv.model());
    try w.print(
        "\",\"wired_mib\":{d},\"active_requests\":{d},\"active_batch_size\":{d},\"prefill_slots\":{d},\"decode_slots\":{d},\"mean_batch_size\":{d:.2},\"tok_per_s\":{d:.2},\"decode_share\":{d:.2},\"prefix_hit_tokens\":{d},\"retired_q8_kv_chunks\":{d},\"cluster_peers_online\":{d}}}}}\n",
        .{
            srv.wired_mib,
            srv.active_requests,
            srv.active_batch_size,
            srv.prefill_slots,
            srv.decode_slots,
            srv.mean_batch_size,
            srv.tok_per_s,
            srv.decode_share,
            srv.prefix_hit_tokens,
            srv.retired_q8_kv_chunks,
            srv.cluster_peers_online,
        },
    );

    return w.written();
}

// ---------------------------------------------------------------------------
// Top-Level Runner (`--json`, `--once`, and Interactive `kqueue` TUI)
// ---------------------------------------------------------------------------

const TIMER_IDENT: u64 = 1;

fn setTimerMs(kq_fd: std.c.fd_t, interval_ms: u32) Error!void {
    const change = [1]std.c.kevent64_s{.{
        .ident = TIMER_IDENT,
        .filter = std.c.EVFILT.TIMER,
        .flags = std.c.EV.ADD | std.c.EV.ENABLE,
        .fflags = 0, // default Darwin EVFILT_TIMER unit: milliseconds
        .data = @intCast(interval_ms),
        .udata = 0,
        .ext = .{ 0, 0 },
    }};
    var empty: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    while (true) {
        const rc = std.c.kevent64(kq_fd, &change, 1, &empty, 0, std.c.KEVENT.FLAG.NONE, null);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.KqueueFailed;
        }
        return;
    }
}

fn queryTerminalSize(out_fd: std.c.fd_t, cfg: MonitorConfig) telemetry.WinSize {
    var ws: telemetry.WinSize = .{ .cols = DEFAULT_COLS, .rows = DEFAULT_ROWS };
    if (telemetry.getWinSize(out_fd)) |live_ws| {
        ws = live_ws;
    } else |_| {}
    if (cfg.cols_override) |c| ws.cols = c;
    if (cfg.rows_override) |r| ws.rows = r;
    return ws;
}

/// Entry point for `aminfer monitor` ready for `src/main.zig` dispatch.
pub fn runCli(allocator: std.mem.Allocator, args: []const []const u8) Error!void {
    const cfg = try MonitorConfig.parseArgs(args);
    if (cfg.show_help) {
        try writeAllFd(1, HELP_TEXT);
        return;
    }
    return run(allocator, cfg, 0, 1);
}

fn restoreTerminalBestEffort(out_fd: std.c.fd_t) void {
    const seq = "\x1b[0m\x1b[?25h\x1b[?1049l";
    var off: usize = 0;
    while (off < seq.len) {
        const rc = std.c.write(out_fd, seq[off..].ptr, seq.len - off);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return;
        }
        if (rc == 0) return;
        off += @intCast(rc);
    }
}

pub fn run(
    allocator: std.mem.Allocator,
    cfg: MonitorConfig,
    in_fd: std.c.fd_t,
    out_fd: std.c.fd_t,
) Error!void {
    // Steer monitor polling to QOS_CLASS_UTILITY E-cores (`ADR-0022` L282).
    if (telemetry.Sampler.steerUtilityQos()) |_| {} else |_| {}

    var state = try MonitorState.init(cfg);
    var sampler = try telemetry.Sampler.init();

    // Non-interactive `--json` or `--once` mode (also used automatically if stdout is not a TTY).
    if (cfg.json) {
        const sleep_ts: std.c.timespec = .{ .sec = 0, .nsec = 15 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&sleep_ts, null);
        const host = try sampler.sample();
        const srv = if (state.server_endpoint) |*ep| pollServerHealth(ep, 150) else ServerTelemetry{};
        var json_buf: [JSON_BUF_CAP]u8 = undefined;
        const json_bytes = try writeJsonSnapshot(&json_buf, &state.palette, &host, &srv);
        try writeAllFd(out_fd, json_bytes);
        return;
    }

    const ws0 = queryTerminalSize(out_fd, cfg);
    var canvas = try Canvas.init(allocator, ws0.cols, ws0.rows);
    defer canvas.deinit(allocator);

    if (cfg.once or !telemetry.isTty(in_fd) or !telemetry.isTty(out_fd)) {
        const sleep_ts: std.c.timespec = .{ .sec = 0, .nsec = 15 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&sleep_ts, null);
        const host = try sampler.sample();
        const srv = if (state.server_endpoint) |*ep| pollServerHealth(ep, 150) else ServerTelemetry{};
        state.recordSnapshot(&host, &srv);
        state.render(&canvas, &host, &srv);
        const frame = try canvas.encodeOnceFrame();
        try writeAllFd(out_fd, frame);
        return;
    }

    // Interactive kqueue + raw termios + alternate screen loop.
    const kq_fd = std.c.kqueue();
    if (kq_fd < 0) return error.KqueueFailed;
    defer _ = std.c.close(kq_fd);

    const changes = [2]std.c.kevent64_s{
        .{
            .ident = @intCast(in_fd),
            .filter = std.c.EVFILT.READ,
            .flags = std.c.EV.ADD | std.c.EV.ENABLE,
            .fflags = 0,
            .data = 0,
            .udata = 0,
            .ext = .{ 0, 0 },
        },
        .{
            .ident = @backingInt(std.c.SIG.WINCH),
            .filter = std.c.EVFILT.SIGNAL,
            .flags = std.c.EV.ADD | std.c.EV.ENABLE,
            .fflags = 0,
            .data = 0,
            .udata = 0,
            .ext = .{ 0, 0 },
        },
    };
    var empty_ev: [1]std.c.kevent64_s = .{std.mem.zeroes(std.c.kevent64_s)};
    if (std.c.kevent64(kq_fd, &changes, changes.len, &empty_ev, 0, std.c.KEVENT.FLAG.NONE, null) < 0) {
        return error.KqueueFailed;
    }
    try setTimerMs(kq_fd, state.interval_ms);

    var raw_guard = try telemetry.TtyRawGuard.enter(in_fd);
    defer raw_guard.leave();

    // Enter alternate screen, hide cursor, clear screen.
    try writeAllFd(out_fd, "\x1b[?1049h\x1b[?25l\x1b[2J");
    defer restoreTerminalBestEffort(out_fd);

    // Render initial frame immediately and retain latest snapshots so pure UI
    // redraws (panel toggle, interval change, SIGWINCH) never perturb tick deltas
    // or violate the ADR-0022 >=500ms --server polling cadence.
    var force_full = true;
    var last_host = try sampler.sample();
    var last_srv = if (state.server_endpoint) |*ep| pollServerHealth(ep, 150) else ServerTelemetry{};
    state.recordSnapshot(&last_host, &last_srv);
    state.render(&canvas, &last_host, &last_srv);
    {
        const diff = try canvas.encodeDiff(force_full);
        try writeAllFd(out_fd, diff);
        force_full = false;
    }

    var evs: [8]std.c.kevent64_s = undefined;
    var key_buf: [64]u8 = undefined;

    while (true) {
        const nev_rc = std.c.kevent64(kq_fd, &empty_ev, 0, &evs, evs.len, std.c.KEVENT.FLAG.NONE, null);
        if (nev_rc < 0) {
            if (std.c.errno(nev_rc) == .INTR) continue;
            return error.KqueueFailed;
        }
        const nev: usize = @intCast(nev_rc);
        var need_redraw = false;
        var need_sample = false;

        for (evs[0..nev]) |ev| {
            if (ev.filter == std.c.EVFILT.TIMER) {
                need_sample = true;
                need_redraw = true;
            } else if (ev.filter == std.c.EVFILT.SIGNAL and ev.ident == @as(u64, @backingInt(std.c.SIG.WINCH))) {
                const ws = queryTerminalSize(out_fd, cfg);
                canvas.resize(ws.cols, ws.rows);
                force_full = true;
                need_redraw = true;
            } else if (ev.filter == std.c.EVFILT.READ) {
                const nread = std.c.read(in_fd, &key_buf, key_buf.len);
                if (nread > 0) {
                    const act = state.handleInputBytes(key_buf[0..@intCast(nread)]);
                    switch (act) {
                        .quit => return,
                        .interval_changed => {
                            try setTimerMs(kq_fd, state.interval_ms);
                            need_redraw = true;
                        },
                        .panel_toggled => {
                            force_full = true;
                            need_redraw = true;
                        },
                        .reset_deltas => {
                            try sampler.resetDeltas();
                            need_sample = true;
                            need_redraw = true;
                        },
                        .none => {},
                    }
                } else if (nread == 0) {
                    return;
                }
            }
        }

        if (need_redraw) {
            if (need_sample) {
                last_host = try sampler.sample();
                last_srv = if (state.server_endpoint) |*ep| pollServerHealth(ep, 150) else ServerTelemetry{};
                state.recordSnapshot(&last_host, &last_srv);
            }
            state.render(&canvas, &last_host, &last_srv);
            const diff = try canvas.encodeDiff(force_full);
            try writeAllFd(out_fd, diff);
            force_full = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

test "brailleCodepoint spans full Unicode Braille block U+2800..U+28FF" {
    try std.testing.expectEqual(@as(u21, '⠀'), brailleCodepoint(0, 0));
    try std.testing.expectEqual(@as(u21, '⣿'), brailleCodepoint(4, 4));
    try std.testing.expectEqual(@as(u21, 0x2840), brailleCodepoint(1, 0));
    try std.testing.expectEqual(@as(u21, 0x2880), brailleCodepoint(0, 1));
    try std.testing.expectEqual(@as(u21, 0x28C0), brailleCodepoint(1, 1));
}

test "MonitorState keybindings handle q/Esc/CSI/+/-/1-4/r and guard >=1 visible panel" {
    const cfg = try MonitorConfig.parseArgs(&.{ "--server", "127.0.0.1:8080", "--interval-ms", "600" });
    try std.testing.expectEqual(@as(u32, 600), cfg.interval_ms);

    var state = try MonitorState.init(cfg);
    // Arrow key CSI escape (`\x1b[A`) must NOT trigger standalone Esc quit.
    try std.testing.expectEqual(KeyAction.none, state.handleInputBytes("\x1b[A"));
    // `-` clamps at MIN_SERVER_INTERVAL_MS (`500 ms` per ADR-0022).
    try std.testing.expectEqual(KeyAction.interval_changed, state.handleInputBytes("-"));
    try std.testing.expectEqual(@as(u32, 500), state.interval_ms);
    try std.testing.expectEqual(KeyAction.interval_changed, state.handleInputBytes("+"));
    try std.testing.expectEqual(@as(u32, 750), state.interval_ms);

    // Toggle panels 1, 2, 3 off; toggling the last panel 4 off must be ignored so >=1 stays visible.
    try std.testing.expectEqual(KeyAction.panel_toggled, state.handleInputBytes("1"));
    try std.testing.expectEqual(KeyAction.panel_toggled, state.handleInputBytes("2"));
    try std.testing.expectEqual(KeyAction.panel_toggled, state.handleInputBytes("3"));
    try std.testing.expectEqual(@as(u8, 1), state.visiblePanelCount());
    try std.testing.expectEqual(KeyAction.none, state.handleInputBytes("4"));
    try std.testing.expectEqual(@as(u8, 1), state.visiblePanelCount());

    try std.testing.expectEqual(KeyAction.reset_deltas, state.handleInputBytes("r"));
    try std.testing.expectEqual(KeyAction.quit, state.handleInputBytes("\x1b"));
    try std.testing.expectEqual(KeyAction.quit, state.handleInputBytes("q"));
}

test "Canvas double-buffered diff rendering and --once/--json modes run with zero per-frame heap allocations" {
    const gpa = std.testing.allocator;
    var sampler = try telemetry.Sampler.init();
    const cfg = try MonitorConfig.parseArgs(&.{ "--once", "--cols", "100", "--rows", "34" });
    var state = try MonitorState.init(cfg);
    var canvas = try Canvas.init(gpa, 100, 34);
    defer canvas.deinit(gpa);

    const host = try sampler.sample();
    const srv: ServerTelemetry = .{};
    state.recordSnapshot(&host, &srv);
    state.render(&canvas, &host, &srv);

    // Full diff emits entire screen; identical second diff emits ONLY sync wrapper (`16` bytes)!
    const full_diff = try canvas.encodeDiff(true);
    try std.testing.expect(full_diff.len > 1024);
    try std.testing.expect(std.mem.indexOf(u8, full_diff, "aminfer monitor") != null);
    try std.testing.expect(std.mem.indexOf(u8, full_diff, "trans-pride") == null);
    try std.testing.expect(std.mem.indexOf(u8, full_diff, "17;19;29m") != null); // #11131D main_bg

    const noop_diff = try canvas.encodeDiff(false);
    try std.testing.expectEqualStrings("\x1b[?2026h\x1b[?2026l", noop_diff);

    // Verify 16 consecutive frames allocate 0 heap bytes.
    const vm_before = try vmstat.Snapshot.capture();
    for (0..16) |tick| {
        const snap = try sampler.sample();
        state.recordSnapshot(&snap, &srv);
        state.render(&canvas, &snap, &srv);
        const frame = if (tick == 0) try canvas.encodeOnceFrame() else try canvas.encodeDiff(false);
        try std.testing.expect(frame.len > 0);
    }
    const vm_after = try vmstat.Snapshot.capture();
    try std.testing.expect(vmstat.heapDelta(vm_before, vm_after).size_in_use < 256 * 1024);

    // Verify `--json` produces valid RFC 8259 JSON matching live telemetry.
    var json_buf: [JSON_BUF_CAP]u8 = undefined;
    const json_slice = try writeJsonSnapshot(&json_buf, &state.palette, &host, &srv);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_slice, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("trans-pride", parsed.value.object.get("theme").?.string);
    try std.testing.expect(parsed.value.object.get("gpu").?.object.get("available").?.bool);
}

test "pollServerHealth parses live loopback HTTP /health JSON without heap allocation" {
    var listener = try transport.TcpListener.bind(.{ .host = .{ 127, 0, 0, 1 }, .port = 0 }, 4);
    defer listener.close();

    const ServerThread = struct {
        fn serveOnce(listen_fd: std.c.fd_t) void {
            const kq = transport.kqueueCreate() catch return;
            defer _ = std.c.close(kq);
            transport.kqueueSetFilter(kq, listen_fd, std.c.EVFILT.READ, std.c.EV.ADD | std.c.EV.ENABLE, 1) catch return;
            var evs: [1]std.c.kevent64_s = undefined;
            const n = transport.kqueueWait(kq, &evs, 2 * std.time.ns_per_s) catch return;
            if (n == 0) return;

            var cli_addr: std.c.sockaddr.in = std.mem.zeroes(std.c.sockaddr.in);
            var cli_len: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
            const cfd = std.c.accept(listen_fd, @ptrCast(&cli_addr), &cli_len);
            if (cfd < 0) return;
            defer _ = std.c.close(cfd);

            // Wait for the client's HTTP GET /health request to arrive before replying,
            // since `accept` on Darwin inherits `O_NONBLOCK` from `listen_fd`.
            const ckq = transport.kqueueCreate() catch return;
            defer _ = std.c.close(ckq);
            transport.kqueueSetFilter(ckq, cfd, std.c.EVFILT.READ, std.c.EV.ADD | std.c.EV.ENABLE, 1) catch return;
            var cevs: [1]std.c.kevent64_s = undefined;
            var rbuf: [512]u8 = undefined;
            var rlen: usize = 0;
            while (rlen < rbuf.len) {
                if (std.mem.indexOf(u8, rbuf[0..rlen], "\r\n\r\n") != null) break;
                const cn = transport.kqueueWait(ckq, &cevs, 2 * std.time.ns_per_s) catch break;
                if (cn == 0) break;
                const got = std.c.read(cfd, rbuf[rlen..].ptr, rbuf.len - rlen);
                if (got > 0) {
                    rlen += @intCast(got);
                } else {
                    break;
                }
            }

            const body =
                "{\"status\":\"ok\",\"model\":\"Qwen3-4B-q4\",\"wired_mib\":2560,\"active_requests\":2," ++
                "\"active_batch_size\":2,\"mean_batch_size\":1.75,\"tok_per_s\":142.5,\"prefix_hit_tokens\":512}";
            var resp_buf: [512]u8 = undefined;
            const resp = std.fmt.bufPrint(
                &resp_buf,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
                .{ body.len, body },
            ) catch return;
            var woff: usize = 0;
            while (woff < resp.len) {
                const wrc = std.c.write(cfd, resp[woff..].ptr, resp.len - woff);
                if (wrc > 0) {
                    woff += @intCast(wrc);
                    continue;
                }
                const werr = std.c.errno(wrc);
                if (werr == .INTR) continue;
                if (werr == .AGAIN) {
                    transport.kqueueSetFilter(ckq, cfd, std.c.EVFILT.WRITE, std.c.EV.ADD | std.c.EV.ENABLE | std.c.EV.ONESHOT, 1) catch return;
                    _ = transport.kqueueWait(ckq, &cevs, 1 * std.time.ns_per_s) catch return;
                    continue;
                }
                return;
            }
            _ = std.c.shutdown(cfd, std.c.SHUT.WR);
        }
    };

    const th = try std.Thread.spawn(.{}, ServerThread.serveOnce, .{listener.fd});
    defer th.join();

    var addr_buf: [32]u8 = undefined;
    const addr_str = try std.fmt.bufPrint(&addr_buf, "127.0.0.1:{d}", .{listener.bound_port});
    const ep = try ServerEndpoint.parse(addr_str);
    const srv = pollServerHealth(&ep, 1000);

    try std.testing.expect(srv.attached);
    try std.testing.expect(srv.online);
    try std.testing.expectEqualStrings("Qwen3-4B-q4", srv.model());
    try std.testing.expectEqual(@as(u64, 2560), srv.wired_mib);
    try std.testing.expectEqual(@as(u32, 2), srv.active_requests);
    try std.testing.expectApproxEqAbs(@as(f64, 142.5), srv.tok_per_s, 0.01);
    try std.testing.expectEqual(@as(u64, 512), srv.prefix_hit_tokens);
}
