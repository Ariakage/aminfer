//! Zero-allocation declarative JSON `ChatTemplate` state machine and generalized
//! `<think>` / `<think:opensource>` / `<|content_thinking|>` splitter (`W24`, ADR-0024).
//!
//! Replaces hardcoded `<|im_start|>` builders across `chat_repl.zig` and `http_sse.zig`.
//! Parses offline-compiled `chat_template_spec` JSON (or auto-detects from
//! `tokenizer_config.json` / `Tokenizer` dialect) without a Jinja2 AST interpreter
//! and renders prompts directly into caller-supplied buffers with zero heap allocation.

const std = @import("std");

pub const SCHEMA_VERSION: u32 = 1;
pub const CHAT_TEMPLATE_METADATA_NAME: []const u8 = "__meta.chat_template_json";
pub const CHAT_TEMPLATE_METADATA_ALIAS: []const u8 = "__aminfer_chat_template__";

pub const TemplateError = error{
    BufferTooSmall,
    InvalidJson,
    InvalidSpec,
    UnsupportedDialect,
    InvalidReasoningEffort,
    UnsupportedContent,
    MissingToolName,
    DepthExceeded,
};

pub fn SmallStr(comptime cap: usize) type {
    return struct {
        buf: [cap]u8 = @splat(0),
        len: u16 = 0,

        pub fn fromSlice(s: []const u8) error{InvalidSpec}!@This() {
            if (s.len > cap) return error.InvalidSpec;
            var out: @This() = .{};
            if (s.len > 0) {
                @memcpy(out.buf[0..s.len], s);
                out.len = @intCast(s.len);
            }
            return out;
        }

        pub fn comptimeFrom(comptime s: []const u8) @This() {
            if (s.len > cap) @compileError("SmallStr literal exceeds capacity");
            var out: @This() = .{};
            if (s.len > 0) {
                @memcpy(out.buf[0..s.len], s);
                out.len = s.len;
            }
            return out;
        }

        pub fn slice(self: *const @This()) []const u8 {
            return self.buf[0..self.len];
        }

        pub fn isEmpty(self: *const @This()) bool {
            return self.len == 0;
        }
    };
}

/// Zero-allocation fixed-buffer writer with NEON-friendly fast JSON escaping.
pub const FixedWriter = struct {
    buf: []u8,
    pos: usize = 0,

    pub fn init(buf: []u8) FixedWriter {
        return .{ .buf = buf, .pos = 0 };
    }

    pub fn remaining(self: *const FixedWriter) usize {
        return self.buf.len - self.pos;
    }

    pub fn written(self: *const FixedWriter) []const u8 {
        return self.buf[0..self.pos];
    }

    pub fn writeByte(self: *FixedWriter, b: u8) error{BufferTooSmall}!void {
        if (self.pos >= self.buf.len) return error.BufferTooSmall;
        self.buf[self.pos] = b;
        self.pos += 1;
    }

    pub fn writeAll(self: *FixedWriter, bytes: []const u8) error{BufferTooSmall}!void {
        if (bytes.len > self.buf.len - self.pos) return error.BufferTooSmall;
        @memcpy(self.buf[self.pos .. self.pos + bytes.len], bytes);
        self.pos += bytes.len;
    }

    pub fn print(self: *FixedWriter, comptime fmt: []const u8, args: anytype) error{BufferTooSmall}!void {
        const s = std.fmt.bufPrint(self.buf[self.pos..], fmt, args) catch return error.BufferTooSmall;
        self.pos += s.len;
    }

    pub fn writeJsonEscaped(self: *FixedWriter, bytes: []const u8) error{BufferTooSmall}!void {
        const hex = "0123456789abcdef";
        var start: usize = 0;
        for (bytes, 0..) |b, i| {
            const need_esc = (b == '"') or (b == '\\') or (b < 0x20);
            if (!need_esc) continue;
            if (i > start) try self.writeAll(bytes[start..i]);
            switch (b) {
                '"' => try self.writeAll("\\\""),
                '\\' => try self.writeAll("\\\\"),
                0x08 => try self.writeAll("\\b"),
                0x0c => try self.writeAll("\\f"),
                '\n' => try self.writeAll("\\n"),
                '\r' => try self.writeAll("\\r"),
                '\t' => try self.writeAll("\\t"),
                else => {
                    const esc = [6]u8{ '\\', 'u', '0', '0', hex[b >> 4], hex[b & 0x0f] };
                    try self.writeAll(&esc);
                },
            }
            start = i + 1;
        }
        if (start < bytes.len) try self.writeAll(bytes[start..]);
    }
};

/// Unified reasoning effort domain (`W24` / ADR-0024):
/// named tiers (`none`, `no_think`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`)
/// plus arbitrary floats in `[0.0, 0.99]`.
pub const ReasoningEffort = union(enum) {
    none,
    no_think,
    minimal,
    low,
    medium,
    high,
    xhigh,
    max,
    numeric: f64,

    pub fn parseString(raw: []const u8) error{InvalidReasoningEffort}!ReasoningEffort {
        const s = std.mem.trim(u8, raw, " \t\r\n");
        if (std.mem.eql(u8, s, "none")) return .none;
        if (std.mem.eql(u8, s, "no_think")) return .no_think;
        if (std.mem.eql(u8, s, "minimal")) return .minimal;
        if (std.mem.eql(u8, s, "low")) return .low;
        if (std.mem.eql(u8, s, "medium")) return .medium;
        if (std.mem.eql(u8, s, "high")) return .high;
        if (std.mem.eql(u8, s, "xhigh")) return .xhigh;
        if (std.mem.eql(u8, s, "max")) return .max;
        return error.InvalidReasoningEffort;
    }

    pub fn fromFloat(v: f64) error{InvalidReasoningEffort}!ReasoningEffort {
        if (!std.math.isFinite(v) or v < 0.0 or v > 0.99) return error.InvalidReasoningEffort;
        return .{ .numeric = v };
    }

    /// Parses either a named effort tier or a numeric `[0.0, 0.99]` literal (for CLI `--reasoning-effort`).
    pub fn parseCliArg(raw: []const u8) error{InvalidReasoningEffort}!ReasoningEffort {
        if (parseString(raw)) |tier| return tier else |_| {}
        const s = std.mem.trim(u8, raw, " \t\r\n");
        if (s.len == 0) return error.InvalidReasoningEffort;
        const v = std.fmt.parseFloat(f64, s) catch return error.InvalidReasoningEffort;
        return fromFloat(v);
    }

    pub fn isThinkingEnabled(self: ReasoningEffort) bool {
        return switch (self) {
            .none, .no_think => false,
            .numeric => |v| v > 0.0,
            .minimal, .low, .medium, .high, .xhigh, .max => true,
        };
    }

    pub fn name(self: ReasoningEffort) []const u8 {
        return switch (self) {
            .none => "none",
            .no_think => "no_think",
            .minimal => "minimal",
            .low => "low",
            .medium => "medium",
            .high => "high",
            .xhigh => "xhigh",
            .max => "max",
            .numeric => "numeric",
        };
    }

    pub fn toInklingFloat(self: ReasoningEffort, map: *const EffortMap) f64 {
        return switch (self) {
            .none, .no_think => map.none,
            .minimal => map.minimal,
            .low => map.low,
            .medium => map.medium,
            .high => map.high,
            .xhigh => map.xhigh,
            .max => map.max,
            .numeric => |v| v,
        };
    }

    pub fn formatInklingValue(self: ReasoningEffort, map: *const EffortMap, buf: []u8) error{BufferTooSmall}![]const u8 {
        const num = self.toInklingFloat(map);
        if (num == 0.0) {
            if (buf.len < 1) return error.BufferTooSmall;
            buf[0] = '0';
            return buf[0..1];
        }
        return std.fmt.bufPrint(buf, "{d}", .{num}) catch return error.BufferTooSmall;
    }

    pub fn toHy4String(self: ReasoningEffort) []const u8 {
        return if (self.isThinkingEnabled()) "high" else "no_think";
    }

    pub fn eqlNormalized(a: ReasoningEffort, b: ReasoningEffort) bool {
        if (std.meta.activeTag(a) == std.meta.activeTag(b)) {
            return switch (a) {
                .numeric => |va| @abs(va - b.numeric) <= 1e-9,
                else => true,
            };
        }
        if ((a == .none or a == .no_think) and (b == .none or b == .no_think)) return true;
        if ((a == .xhigh or a == .max) and (b == .xhigh or b == .max)) return true;
        const map = EffortMap{};
        const va = a.toInklingFloat(&map);
        const vb = b.toInklingFloat(&map);
        return @abs(va - vb) <= 1e-9;
    }
};

pub const EffortMap = struct {
    none: f64 = 0.0,
    minimal: f64 = 0.1,
    low: f64 = 0.2,
    medium: f64 = 0.7,
    high: f64 = 0.9,
    xhigh: f64 = 0.99,
    max: f64 = 0.99,
};

pub const Dialect = enum(u8) {
    chatml = 0,
    deepseek = 1,
    inkling2 = 2,
    hy4 = 3,
    custom = 4,

    pub fn name(self: Dialect) []const u8 {
        return switch (self) {
            .chatml => "chatml",
            .deepseek => "deepseek",
            .inkling2 => "inkling2",
            .hy4 => "hy4",
            .custom => "custom",
        };
    }

    pub fn parse(s: []const u8) error{UnsupportedDialect}!Dialect {
        if (std.mem.eql(u8, s, "chatml") or std.mem.eql(u8, s, "qwen") or std.mem.eql(u8, s, "qwen3")) return .chatml;
        if (std.mem.eql(u8, s, "deepseek")) return .deepseek;
        if (std.mem.eql(u8, s, "inkling2") or std.mem.eql(u8, s, "inkling-2")) return .inkling2;
        if (std.mem.eql(u8, s, "hy4") or std.mem.eql(u8, s, "hy4-flash") or std.mem.eql(u8, s, "hy4-preview") or std.mem.eql(u8, s, "hy_v4")) return .hy4;
        if (std.mem.eql(u8, s, "custom")) return .custom;
        return error.UnsupportedDialect;
    }
};

pub const EffortMode = enum(u8) {
    none = 0,
    boolean_think_prefill = 1,
    inkling2_effort_directive = 2,
    hy4_reasoning_mode = 3,

    pub fn parse(s: []const u8) error{InvalidSpec}!EffortMode {
        if (std.mem.eql(u8, s, "none")) return .none;
        if (std.mem.eql(u8, s, "boolean_think_prefill")) return .boolean_think_prefill;
        if (std.mem.eql(u8, s, "inkling2_effort_directive")) return .inkling2_effort_directive;
        if (std.mem.eql(u8, s, "hy4_reasoning_mode")) return .hy4_reasoning_mode;
        return error.InvalidSpec;
    }
};

pub const ToolCallFormat = enum(u8) {
    qwen_xml_json = 0,
    inkling2_xml_json = 1,
    hy4_xml_arg = 2,
    deepseek = 3,
    custom = 4,

    pub fn parse(s: []const u8) error{InvalidSpec}!ToolCallFormat {
        if (std.mem.eql(u8, s, "qwen_xml_json")) return .qwen_xml_json;
        if (std.mem.eql(u8, s, "inkling2_xml_json")) return .inkling2_xml_json;
        if (std.mem.eql(u8, s, "hy4_xml_arg")) return .hy4_xml_arg;
        if (std.mem.eql(u8, s, "deepseek")) return .deepseek;
        if (std.mem.eql(u8, s, "custom")) return .custom;
        return error.InvalidSpec;
    }
};

pub const RoleTokens = struct {
    system_prefix: SmallStr(96) = .{},
    system_suffix: SmallStr(96) = .{},
    user_prefix: SmallStr(96) = .{},
    user_suffix: SmallStr(96) = .{},
    assistant_prefix: SmallStr(96) = .{},
    assistant_suffix: SmallStr(96) = .{},
    tool_prefix: SmallStr(96) = .{},
    tool_suffix: SmallStr(96) = .{},
    header_system: SmallStr(64) = .{},
    header_user: SmallStr(64) = .{},
    header_assistant: SmallStr(64) = .{},
    header_tool: SmallStr(64) = .{},
    content_text_tag: SmallStr(64) = .{},
    end_message_tag: SmallStr(64) = .{},
    hy4_start: SmallStr(64) = .{},
    hy4_middle: SmallStr(64) = .{},
    hy4_end: SmallStr(64) = .{},
};

pub const ThinkingSpec = struct {
    supported: bool = true,
    open_tag: SmallStr(64) = SmallStr(64).comptimeFrom("<think>"),
    close_tag: SmallStr(64) = SmallStr(64).comptimeFrom("</think>"),
    prefill_think: SmallStr(96) = .{},
    prefill_no_think: SmallStr(96) = SmallStr(96).comptimeFrom("<think>\n\n</think>\n\n"),
    effort_mode: EffortMode = .boolean_think_prefill,
    default_effort: ReasoningEffort = .high,
    effort_map: EffortMap = .{},
    preserved_thinking: ?bool = null,
    reasoning_mode_token: SmallStr(64) = .{},
    starts_in_think_on_think: bool = false,
};

pub const ToolCallingSpec = struct {
    format: ToolCallFormat = .qwen_xml_json,
    declare_prefix: SmallStr(128) = .{},
    declare_suffix: SmallStr(128) = .{},
    call_open: SmallStr(64) = SmallStr(64).comptimeFrom("<tool_call>\n"),
    call_close: SmallStr(64) = SmallStr(64).comptimeFrom("\n</tool_call>\n"),
    response_open: SmallStr(64) = SmallStr(64).comptimeFrom("<tool_response>\n"),
    response_close: SmallStr(64) = SmallStr(64).comptimeFrom("\n</tool_response>"),
};

pub const MultimodalSpec = struct {
    image_tag: SmallStr(128) = SmallStr(128).comptimeFrom("<|vision_start|><|image_pad|><|vision_end|>"),
    video_tag: SmallStr(128) = SmallStr(128).comptimeFrom("<|vision_start|><|video_pad|><|vision_end|>"),
};

pub const Role = enum(u8) {
    system = 0,
    user = 1,
    assistant = 2,
    tool = 3,

    pub fn parse(s: []const u8) ?Role {
        if (std.mem.eql(u8, s, "system") or std.mem.eql(u8, s, "developer")) return .system;
        if (std.mem.eql(u8, s, "user")) return .user;
        if (std.mem.eql(u8, s, "assistant")) return .assistant;
        if (std.mem.eql(u8, s, "tool")) return .tool;
        return null;
    }
};

pub const ContentPart = union(enum) {
    text: []const u8,
    image: void,
    video: void,
};

pub const ToolCallView = struct {
    id: []const u8 = "",
    name: []const u8,
    arguments_json: []const u8 = "{}",
};

pub const MessageView = struct {
    role: Role,
    name: []const u8 = "",
    content: []const u8 = "",
    parts: []const ContentPart = &.{},
    reasoning_content: []const u8 = "",
    tool_calls: []const ToolCallView = &.{},
    tool_call_id: []const u8 = "",
};

pub const ToolDeclView = struct {
    name: []const u8,
    description: []const u8 = "",
    parameters_json: []const u8 = "{}",
    tool_type: []const u8 = "function",
};

pub const ToolChoiceKind = enum(u8) {
    auto = 0,
    none = 1,
    required = 2,
    named = 3,
};

pub const RenderOptions = struct {
    reasoning_effort: ?ReasoningEffort = null,
    enable_thinking: ?bool = null,
    preserved_thinking: ?bool = null,
    add_generation_prompt: ?bool = null,
    tools: []const ToolDeclView = &.{},
    raw_tools_json: ?[]const u8 = null,
    tool_choice: ToolChoiceKind = .auto,
    tool_choice_name: []const u8 = "",
    parallel_tool_calls: bool = true,
    add_vision_id: bool = false,
    prior_prompt: ?[]const u8 = null,
    /// Preserves single-`\n` `<think>\n\n</think>\n` formatting for `chat_repl.zig` ChatML compatibility.
    repl_compact_nothink: bool = false,
};

pub const RenderResult = struct {
    prompt: []const u8,
    last_user_msg_byte_offset: ?usize,
    starts_in_think: bool,
};

// ---------------------------------------------------------------------------
// Zero-alloc JSON scanner & Inkling 2 / Hy4 canonical JSON helpers
// ---------------------------------------------------------------------------

const JsonScan = struct {
    src: []const u8,
    pos: usize = 0,

    fn skipWs(self: *JsonScan) void {
        while (self.pos < self.src.len) {
            switch (self.src[self.pos]) {
                ' ', '\t', '\r', '\n' => self.pos += 1,
                else => return,
            }
        }
    }

    fn peek(self: *JsonScan) ?u8 {
        self.skipWs();
        if (self.pos >= self.src.len) return null;
        return self.src[self.pos];
    }

    fn expectByte(self: *JsonScan, expected: u8) TemplateError!void {
        const ch = self.peek() orelse return error.InvalidJson;
        if (ch != expected) return error.InvalidJson;
        self.pos += 1;
    }

    fn consumeKeyword(self: *JsonScan, kw: []const u8) bool {
        self.skipWs();
        if (self.pos + kw.len > self.src.len) return false;
        if (!std.mem.eql(u8, self.src[self.pos .. self.pos + kw.len], kw)) return false;
        if (self.pos + kw.len < self.src.len) {
            const nxt = self.src[self.pos + kw.len];
            if ((nxt >= 'a' and nxt <= 'z') or (nxt >= 'A' and nxt <= 'Z') or (nxt >= '0' and nxt <= '9') or nxt == '_') {
                return false;
            }
        }
        self.pos += kw.len;
        return true;
    }

    fn readBool(self: *JsonScan) TemplateError!bool {
        if (self.consumeKeyword("true")) return true;
        if (self.consumeKeyword("false")) return false;
        return error.InvalidJson;
    }

    fn readU32(self: *JsonScan) TemplateError!u32 {
        self.skipWs();
        const start = self.pos;
        while (self.pos < self.src.len and self.src[self.pos] >= '0' and self.src[self.pos] <= '9') : (self.pos += 1) {}
        if (self.pos == start) return error.InvalidJson;
        return std.fmt.parseInt(u32, self.src[start..self.pos], 10) catch return error.InvalidSpec;
    }

    fn readF64(self: *JsonScan) TemplateError!f64 {
        self.skipWs();
        const start = self.pos;
        if (self.pos < self.src.len and (self.src[self.pos] == '-' or self.src[self.pos] == '+')) self.pos += 1;
        var saw_digit = false;
        while (self.pos < self.src.len) {
            const c = self.src[self.pos];
            if (c >= '0' and c <= '9') {
                saw_digit = true;
                self.pos += 1;
            } else if (c == '.' or c == 'e' or c == 'E' or c == '+' or c == '-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if (!saw_digit) return error.InvalidJson;
        const val = std.fmt.parseFloat(f64, self.src[start..self.pos]) catch return error.InvalidSpec;
        if (!std.math.isFinite(val)) return error.InvalidSpec;
        return val;
    }

    fn readStringInto(self: *JsonScan, dst: *FixedWriter) TemplateError!void {
        try self.expectByte('"');
        while (self.pos < self.src.len) {
            const b = self.src[self.pos];
            self.pos += 1;
            if (b == '"') return;
            if (b < 0x20) return error.InvalidJson;
            if (b != '\\') {
                try dst.writeByte(b);
                continue;
            }
            if (self.pos >= self.src.len) return error.InvalidJson;
            const esc = self.src[self.pos];
            self.pos += 1;
            switch (esc) {
                '"' => try dst.writeByte('"'),
                '\\' => try dst.writeByte('\\'),
                '/' => try dst.writeByte('/'),
                'b' => try dst.writeByte(0x08),
                'f' => try dst.writeByte(0x0c),
                'n' => try dst.writeByte('\n'),
                'r' => try dst.writeByte('\r'),
                't' => try dst.writeByte('\t'),
                'u' => {
                    if (self.pos + 4 > self.src.len) return error.InvalidJson;
                    const cp0 = std.fmt.parseInt(u21, self.src[self.pos .. self.pos + 4], 16) catch return error.InvalidJson;
                    self.pos += 4;
                    var cp = cp0;
                    if (cp0 >= 0xD800 and cp0 <= 0xDBFF) {
                        if (self.pos + 6 <= self.src.len and self.src[self.pos] == '\\' and self.src[self.pos + 1] == 'u') {
                            const lo = std.fmt.parseInt(u21, self.src[self.pos + 2 .. self.pos + 6], 16) catch return error.InvalidJson;
                            if (lo >= 0xDC00 and lo <= 0xDFFF) {
                                self.pos += 6;
                                cp = 0x10000 + ((cp0 - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                cp = 0xFFFD;
                            }
                        } else {
                            cp = 0xFFFD;
                        }
                    } else if (cp0 >= 0xDC00 and cp0 <= 0xDFFF) {
                        cp = 0xFFFD;
                    }
                    var ubuf: [4]u8 = undefined;
                    const ulen = std.unicode.utf8Encode(cp, &ubuf) catch return error.InvalidJson;
                    try dst.writeAll(ubuf[0..ulen]);
                },
                else => return error.InvalidJson,
            }
        }
        return error.InvalidJson;
    }

    fn readSmallString(self: *JsonScan, buf: []u8) TemplateError![]const u8 {
        var w = FixedWriter.init(buf);
        self.readStringInto(&w) catch |err| switch (err) {
            error.BufferTooSmall => return error.InvalidSpec,
            else => return err,
        };
        return w.written();
    }

    fn readKeyOrSkip(self: *JsonScan, buf: []u8) TemplateError!?[]const u8 {
        var w = FixedWriter.init(buf);
        self.readStringInto(&w) catch |err| switch (err) {
            error.BufferTooSmall => return null,
            else => return err,
        };
        return w.written();
    }

    fn skipValue(self: *JsonScan, depth: usize) TemplateError!void {
        if (depth > 64) return error.DepthExceeded;
        const ch = self.peek() orelse return error.InvalidJson;
        switch (ch) {
            '"' => {
                self.pos += 1;
                while (self.pos < self.src.len) {
                    const b = self.src[self.pos];
                    self.pos += 1;
                    if (b == '"') return;
                    if (b == '\\') {
                        if (self.pos >= self.src.len) return error.InvalidJson;
                        self.pos += 1;
                    }
                }
                return error.InvalidJson;
            },
            '{' => {
                self.pos += 1;
                if (self.peek() == '}') {
                    self.pos += 1;
                    return;
                }
                while (true) {
                    try self.skipValue(depth + 1);
                    try self.expectByte(':');
                    try self.skipValue(depth + 1);
                    const sep = self.peek() orelse return error.InvalidJson;
                    if (sep == ',') {
                        self.pos += 1;
                    } else if (sep == '}') {
                        self.pos += 1;
                        return;
                    } else return error.InvalidJson;
                }
            },
            '[' => {
                self.pos += 1;
                if (self.peek() == ']') {
                    self.pos += 1;
                    return;
                }
                while (true) {
                    try self.skipValue(depth + 1);
                    const sep = self.peek() orelse return error.InvalidJson;
                    if (sep == ',') {
                        self.pos += 1;
                    } else if (sep == ']') {
                        self.pos += 1;
                        return;
                    } else return error.InvalidJson;
                }
            },
            't' => if (self.consumeKeyword("true")) return else return error.InvalidJson,
            'f' => if (self.consumeKeyword("false")) return else return error.InvalidJson,
            'n' => if (self.consumeKeyword("null")) return else return error.InvalidJson,
            '-', '0'...'9' => {
                _ = try self.readF64();
            },
            else => return error.InvalidJson,
        }
    }

    fn captureValueSlice(self: *JsonScan, depth: usize) TemplateError![]const u8 {
        self.skipWs();
        const start = self.pos;
        try self.skipValue(depth);
        return self.src[start..self.pos];
    }
};

fn readSmallStrOpt(r: *JsonScan, out: anytype) TemplateError!void {
    if (r.consumeKeyword("null")) {
        out.* = .{};
        return;
    }
    var tmp: [256]u8 = undefined;
    const s = try r.readSmallString(&tmp);
    const T = @TypeOf(out.*);
    out.* = try T.fromSlice(s);
}

/// Writes a JSON value canonically with sorted keys (`sort_keys=true`) and compact
/// or spaced separators using zero heap allocations (bounded stack keys per nesting level).
pub fn writeCanonicalJsonValue(
    r: *JsonScan,
    dst: *FixedWriter,
    depth: usize,
    comptime sort_keys: bool,
    comptime spaced: bool,
) TemplateError!void {
    if (depth > 32) return error.DepthExceeded;
    const ch = r.peek() orelse return error.InvalidJson;
    switch (ch) {
        '"' => {
            try dst.writeByte('"');
            const raw_slice = try r.captureValueSlice(depth + 1);
            // Re-decode string cleanly to normalize escapes identically to Python `json.dumps`.
            var sub = JsonScan{ .src = raw_slice };
            var ubuf: [2048]u8 = undefined;
            var uw = FixedWriter.init(&ubuf);
            sub.readStringInto(&uw) catch |err| switch (err) {
                error.BufferTooSmall => {
                    // Large string: emit raw unquoted slice contents verbatim.
                    try dst.writeAll(raw_slice[1 .. raw_slice.len - 1]);
                    try dst.writeByte('"');
                    return;
                },
                else => return err,
            };
            try dst.writeJsonEscaped(uw.written());
            try dst.writeByte('"');
        },
        '[' => {
            r.pos += 1;
            try dst.writeByte('[');
            if (r.peek() == ']') {
                r.pos += 1;
                try dst.writeByte(']');
                return;
            }
            var first = true;
            while (true) {
                if (!first) try dst.writeAll(if (spaced) ", " else ",");
                first = false;
                try writeCanonicalJsonValue(r, dst, depth + 1, sort_keys, spaced);
                const sep = r.peek() orelse return error.InvalidJson;
                if (sep == ',') {
                    r.pos += 1;
                } else if (sep == ']') {
                    r.pos += 1;
                    try dst.writeByte(']');
                    return;
                } else return error.InvalidJson;
            }
        },
        '{' => {
            r.pos += 1;
            try dst.writeByte('{');
            if (r.peek() == '}') {
                r.pos += 1;
                try dst.writeByte('}');
                return;
            }
            if (!sort_keys) {
                var first = true;
                while (true) {
                    if (!first) try dst.writeAll(if (spaced) ", " else ",");
                    first = false;
                    try writeCanonicalJsonValue(r, dst, depth + 1, false, spaced);
                    try r.expectByte(':');
                    try dst.writeAll(if (spaced) ": " else ":");
                    try writeCanonicalJsonValue(r, dst, depth + 1, false, spaced);
                    const sep = r.peek() orelse return error.InvalidJson;
                    if (sep == ',') {
                        r.pos += 1;
                    } else if (sep == '}') {
                        r.pos += 1;
                        try dst.writeByte('}');
                        return;
                    } else return error.InvalidJson;
                }
            }
            const KvEntry = struct {
                key_raw: []const u8,
                val_raw: []const u8,
                fn lessThan(_: void, a: @This(), b: @This()) bool {
                    return std.mem.order(u8, a.key_raw, b.key_raw) == .lt;
                }
            };
            var entries: [64]KvEntry = undefined;
            var count: usize = 0;
            var overflow = false;
            const obj_body_start = r.pos;
            while (true) {
                const k_raw = try r.captureValueSlice(depth + 1);
                try r.expectByte(':');
                const v_raw = try r.captureValueSlice(depth + 1);
                if (count < entries.len) {
                    entries[count] = .{ .key_raw = k_raw, .val_raw = v_raw };
                    count += 1;
                } else {
                    overflow = true;
                }
                const sep = r.peek() orelse return error.InvalidJson;
                if (sep == ',') {
                    r.pos += 1;
                } else if (sep == '}') {
                    r.pos += 1;
                    break;
                } else return error.InvalidJson;
            }
            if (overflow) {
                var fallback = JsonScan{ .src = r.src, .pos = obj_body_start };
                var first = true;
                while (true) {
                    if (!first) try dst.writeAll(if (spaced) ", " else ",");
                    first = false;
                    try writeCanonicalJsonValue(&fallback, dst, depth + 1, false, spaced);
                    try fallback.expectByte(':');
                    try dst.writeAll(if (spaced) ": " else ":");
                    try writeCanonicalJsonValue(&fallback, dst, depth + 1, false, spaced);
                    const sep = fallback.peek() orelse return error.InvalidJson;
                    if (sep == ',') {
                        fallback.pos += 1;
                    } else if (sep == '}') {
                        break;
                    } else return error.InvalidJson;
                }
                try dst.writeByte('}');
                return;
            }
            std.mem.sort(KvEntry, entries[0..count], {}, KvEntry.lessThan);
            for (entries[0..count], 0..) |e, idx| {
                if (idx > 0) try dst.writeAll(if (spaced) ", " else ",");
                var kr = JsonScan{ .src = e.key_raw };
                try writeCanonicalJsonValue(&kr, dst, depth + 1, true, spaced);
                try dst.writeAll(if (spaced) ": " else ":");
                var vr = JsonScan{ .src = e.val_raw };
                try writeCanonicalJsonValue(&vr, dst, depth + 1, true, spaced);
            }
            try dst.writeByte('}');
        },
        else => {
            const raw = try r.captureValueSlice(depth + 1);
            try dst.writeAll(raw);
        },
    }
}

/// Declarative zero-allocation chat template specification (`W24`).
pub const ChatTemplate = struct {
    schema_version: u32 = SCHEMA_VERSION,
    dialect: Dialect = .chatml,
    bos_token: SmallStr(64) = .{},
    eos_token: SmallStr(64) = SmallStr(64).comptimeFrom("<|im_end|>"),
    add_generation_prompt: bool = true,
    role_tokens: RoleTokens = .{},
    thinking: ThinkingSpec = .{},
    tool_calling: ToolCallingSpec = .{},
    multimodal: MultimodalSpec = .{},

    pub fn preset(d: Dialect) ChatTemplate {
        return switch (d) {
            .chatml, .custom => .{
                .schema_version = SCHEMA_VERSION,
                .dialect = d,
                .bos_token = .{},
                .eos_token = SmallStr(64).comptimeFrom("<|im_end|>"),
                .add_generation_prompt = true,
                .role_tokens = .{
                    .system_prefix = SmallStr(96).comptimeFrom("<|im_start|>system\n"),
                    .system_suffix = SmallStr(96).comptimeFrom("<|im_end|>\n"),
                    .user_prefix = SmallStr(96).comptimeFrom("<|im_start|>user\n"),
                    .user_suffix = SmallStr(96).comptimeFrom("<|im_end|>\n"),
                    .assistant_prefix = SmallStr(96).comptimeFrom("<|im_start|>assistant\n"),
                    .assistant_suffix = SmallStr(96).comptimeFrom("<|im_end|>\n"),
                    .tool_prefix = SmallStr(96).comptimeFrom("<|im_start|>user\n<tool_response>\n"),
                    .tool_suffix = SmallStr(96).comptimeFrom("\n</tool_response><|im_end|>\n"),
                },
                .thinking = .{
                    .supported = true,
                    .open_tag = SmallStr(64).comptimeFrom("<think>"),
                    .close_tag = SmallStr(64).comptimeFrom("</think>"),
                    .prefill_think = .{},
                    .prefill_no_think = SmallStr(96).comptimeFrom("<think>\n\n</think>\n\n"),
                    .effort_mode = .boolean_think_prefill,
                    .default_effort = .high,
                    .starts_in_think_on_think = false,
                },
                .tool_calling = .{
                    .format = .qwen_xml_json,
                    .call_open = SmallStr(64).comptimeFrom("<tool_call>\n"),
                    .call_close = SmallStr(64).comptimeFrom("\n</tool_call>\n"),
                    .response_open = SmallStr(64).comptimeFrom("<tool_response>\n"),
                    .response_close = SmallStr(64).comptimeFrom("\n</tool_response>"),
                },
                .multimodal = .{
                    .image_tag = SmallStr(128).comptimeFrom("<|vision_start|><|image_pad|><|vision_end|>"),
                    .video_tag = SmallStr(128).comptimeFrom("<|vision_start|><|video_pad|><|vision_end|>"),
                },
            },
            .deepseek => .{
                .schema_version = SCHEMA_VERSION,
                .dialect = .deepseek,
                .bos_token = SmallStr(64).comptimeFrom("<\xef\xbd\x9cbegin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"),
                .eos_token = SmallStr(64).comptimeFrom("<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"),
                .add_generation_prompt = true,
                .role_tokens = .{
                    .system_prefix = SmallStr(96).comptimeFrom("<\xef\xbd\x9cSystem\xef\xbd\x9c>"),
                    .system_suffix = .{},
                    .user_prefix = SmallStr(96).comptimeFrom("<\xef\xbd\x9cUser\xef\xbd\x9c>"),
                    .user_suffix = .{},
                    .assistant_prefix = SmallStr(96).comptimeFrom("<\xef\xbd\x9cAssistant\xef\xbd\x9c>"),
                    .assistant_suffix = SmallStr(96).comptimeFrom("<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"),
                    .tool_prefix = SmallStr(96).comptimeFrom("<\xef\xbd\x9ctool\xe2\x96\x81outputs\xe2\x96\x81begin\xef\xbd\x9c><\xef\xbd\x9ctool\xe2\x96\x81output\xe2\x96\x81begin\xef\xbd\x9c>"),
                    .tool_suffix = SmallStr(96).comptimeFrom("<\xef\xbd\x9ctool\xe2\x96\x81output\xe2\x96\x81end\xef\xbd\x9c><\xef\xbd\x9ctool\xe2\x96\x81outputs\xe2\x96\x81end\xef\xbd\x9c>"),
                },
                .thinking = .{
                    .supported = true,
                    .open_tag = SmallStr(64).comptimeFrom("<think>"),
                    .close_tag = SmallStr(64).comptimeFrom("</think>"),
                    .prefill_think = SmallStr(96).comptimeFrom("<think>\n"),
                    .prefill_no_think = SmallStr(96).comptimeFrom("</think>\n"),
                    .effort_mode = .boolean_think_prefill,
                    .default_effort = .high,
                    .starts_in_think_on_think = true,
                },
                .tool_calling = .{
                    .format = .deepseek,
                    .call_open = SmallStr(64).comptimeFrom("<\xef\xbd\x9ctool\xe2\x96\x81calls\xe2\x96\x81begin\xef\xbd\x9c><\xef\xbd\x9ctool\xe2\x96\x81call\xe2\x96\x81begin\xef\xbd\x9c>"),
                    .call_close = SmallStr(64).comptimeFrom("<\xef\xbd\x9ctool\xe2\x96\x81call\xe2\x96\x81end\xef\xbd\x9c><\xef\xbd\x9ctool\xe2\x96\x81calls\xe2\x96\x81end\xef\xbd\x9c>"),
                    .response_open = SmallStr(64).comptimeFrom("<\xef\xbd\x9ctool\xe2\x96\x81output\xe2\x96\x81begin\xef\xbd\x9c>"),
                    .response_close = SmallStr(64).comptimeFrom("<\xef\xbd\x9ctool\xe2\x96\x81output\xe2\x96\x81end\xef\xbd\x9c>"),
                },
                .multimodal = .{ .image_tag = .{}, .video_tag = .{} },
            },
            .inkling2 => .{
                .schema_version = SCHEMA_VERSION,
                .dialect = .inkling2,
                .bos_token = .{},
                .eos_token = SmallStr(64).comptimeFrom("<|content_model_end_sampling|>"),
                .add_generation_prompt = false,
                .role_tokens = .{
                    .system_prefix = SmallStr(96).comptimeFrom("<|message_system|><|content_text|>"),
                    .system_suffix = SmallStr(96).comptimeFrom("<|end_message|>"),
                    .user_prefix = SmallStr(96).comptimeFrom("<|message_user|><|content_text|>"),
                    .user_suffix = SmallStr(96).comptimeFrom("<|end_message|>"),
                    .assistant_prefix = SmallStr(96).comptimeFrom("<|message_model|><|content_text|>"),
                    .assistant_suffix = SmallStr(96).comptimeFrom("<|end_message|>"),
                    .tool_prefix = SmallStr(96).comptimeFrom("<|message_tool|><|content_text|>"),
                    .tool_suffix = SmallStr(96).comptimeFrom("<|end_message|>"),
                    .header_system = SmallStr(64).comptimeFrom("<|message_system|>"),
                    .header_user = SmallStr(64).comptimeFrom("<|message_user|>"),
                    .header_assistant = SmallStr(64).comptimeFrom("<|message_model|>"),
                    .header_tool = SmallStr(64).comptimeFrom("<|message_tool|>"),
                    .content_text_tag = SmallStr(64).comptimeFrom("<|content_text|>"),
                    .end_message_tag = SmallStr(64).comptimeFrom("<|end_message|>"),
                },
                .thinking = .{
                    .supported = true,
                    .open_tag = SmallStr(64).comptimeFrom("<|content_thinking|>"),
                    .close_tag = SmallStr(64).comptimeFrom("<|end_message|>"),
                    .prefill_think = .{},
                    .prefill_no_think = .{},
                    .effort_mode = .inkling2_effort_directive,
                    .default_effort = .{ .numeric = 0.9 },
                    .effort_map = .{},
                    .starts_in_think_on_think = false,
                },
                .tool_calling = .{
                    .format = .inkling2_xml_json,
                    .declare_prefix = SmallStr(128).comptimeFrom("<|message_system|>tool_declare<|content_xml|>"),
                    .declare_suffix = SmallStr(128).comptimeFrom("<|end_message|>"),
                    .call_open = SmallStr(64).comptimeFrom("<|content_invoke_tool_json|>"),
                    .call_close = SmallStr(64).comptimeFrom("<|end_message|>"),
                    .response_open = SmallStr(64).comptimeFrom("<|message_tool|>"),
                    .response_close = SmallStr(64).comptimeFrom("<|end_message|>"),
                },
                .multimodal = .{
                    .image_tag = SmallStr(128).comptimeFrom("<|content_image|><|unused_200054|>"),
                    .video_tag = .{},
                },
            },
            .hy4 => .{
                .schema_version = SCHEMA_VERSION,
                .dialect = .hy4,
                .bos_token = SmallStr(64).comptimeFrom("<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>"),
                .eos_token = SmallStr(64).comptimeFrom("<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"),
                .add_generation_prompt = true,
                .role_tokens = .{
                    .system_prefix = SmallStr(96).comptimeFrom("<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>system<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>"),
                    .system_suffix = SmallStr(96).comptimeFrom("<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"),
                    .user_prefix = SmallStr(96).comptimeFrom("<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>user<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>"),
                    .user_suffix = SmallStr(96).comptimeFrom("<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"),
                    .assistant_prefix = SmallStr(96).comptimeFrom("<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>assistant<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>"),
                    .assistant_suffix = SmallStr(96).comptimeFrom("<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"),
                    .tool_prefix = SmallStr(96).comptimeFrom("<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>tool<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>"),
                    .tool_suffix = SmallStr(96).comptimeFrom("<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"),
                    .hy4_start = SmallStr(64).comptimeFrom("<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>"),
                    .hy4_middle = SmallStr(64).comptimeFrom("<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>"),
                    .hy4_end = SmallStr(64).comptimeFrom("<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"),
                },
                .thinking = .{
                    .supported = true,
                    .open_tag = SmallStr(64).comptimeFrom("<think:opensource>"),
                    .close_tag = SmallStr(64).comptimeFrom("</think:opensource>"),
                    .prefill_think = SmallStr(96).comptimeFrom("<think:opensource>"),
                    .prefill_no_think = SmallStr(96).comptimeFrom("<think:opensource></think:opensource>"),
                    .effort_mode = .hy4_reasoning_mode,
                    .default_effort = .high,
                    .reasoning_mode_token = SmallStr(64).comptimeFrom("<\xef\xbd\x9creasoning_mode:opensource\xef\xbd\x9c>"),
                    .starts_in_think_on_think = true,
                },
                .tool_calling = .{
                    .format = .hy4_xml_arg,
                    .call_open = SmallStr(64).comptimeFrom("<tool_calls:opensource>"),
                    .call_close = SmallStr(64).comptimeFrom("</tool_calls:opensource>"),
                    .response_open = SmallStr(64).comptimeFrom("<tool_response:opensource>"),
                    .response_close = SmallStr(64).comptimeFrom("</tool_response:opensource>"),
                },
                .multimodal = .{
                    .image_tag = SmallStr(128).comptimeFrom("<\xef\xbd\x9chy_image\xe2\x96\x81start:opensource\xef\xbd\x9c><\xef\xbd\x9chy_image\xe2\x96\x81pad:opensource\xef\xbd\x9c><\xef\xbd\x9chy_image\xe2\x96\x81end:opensource\xef\xbd\x9c>"),
                    .video_tag = SmallStr(128).comptimeFrom("<\xef\xbd\x9chy_video\xe2\x96\x81start:opensource\xef\xbd\x9c><\xef\xbd\x9chy_video\xe2\x96\x81pad:opensource\xef\xbd\x9c><\xef\xbd\x9chy_video\xe2\x96\x81end:opensource\xef\xbd\x9c>"),
                },
            },
        };
    }

    /// Auto-detects `ChatTemplate` preset from any duck-typed `Tokenizer` (`dialect` or `addedTokenId`).
    pub fn fromTokenizer(tok: anytype) ChatTemplate {
        const T = @TypeOf(tok.*);
        if (@hasField(T, "dialect")) {
            return switch (tok.dialect) {
                .qwen => preset(.chatml),
                .deepseek => preset(.deepseek),
                .inkling2 => preset(.inkling2),
                .hy4 => preset(.hy4),
            };
        }
        if (tok.addedTokenId("<|content_model_end_sampling|>") != null or
            tok.addedTokenId("<|message_user|>") != null)
        {
            return preset(.inkling2);
        }
        if (tok.addedTokenId("<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>") != null) {
            return preset(.hy4);
        }
        if (tok.addedTokenId("<\xef\xbd\x9cbegin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>") != null) {
            return preset(.deepseek);
        }
        return preset(.chatml);
    }

    /// Auto-detects preset (or parses declarative `chat_template_spec`) from `tokenizer_config.json` or `.amf` metadata JSON.
    pub fn detectFromConfigJson(json_bytes: []const u8) TemplateError!ChatTemplate {
        if (std.mem.indexOf(u8, json_bytes, "\"schema_version\"") != null and
            std.mem.indexOf(u8, json_bytes, "\"dialect\"") != null)
        {
            return fromSpecJson(json_bytes);
        }
        if (std.mem.indexOf(u8, json_bytes, "<|content_model_end_sampling|>") != null or
            std.mem.indexOf(u8, json_bytes, "\"content_thinking\"") != null or
            std.mem.indexOf(u8, json_bytes, "<|message_user|>") != null)
        {
            return preset(.inkling2);
        }
        if (std.mem.indexOf(u8, json_bytes, "<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>") != null or
            std.mem.indexOf(u8, json_bytes, "<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>") != null or
            std.mem.indexOf(u8, json_bytes, "HYV4") != null)
        {
            return preset(.hy4);
        }
        if (std.mem.indexOf(u8, json_bytes, "<\xef\xbd\x9cbegin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>") != null or
            std.mem.indexOf(u8, json_bytes, "<\xef\xbd\x9cUser\xef\xbd\x9c>") != null or
            std.mem.indexOf(u8, json_bytes, "\\uff5cbegin\\u2581of\\u2581sentence\\uff5c") != null)
        {
            return preset(.deepseek);
        }
        return preset(.chatml);
    }

    /// Parses declarative JSON `chat_template_spec` (`schema_version = 1`) with zero heap allocation.
    pub fn fromSpecJson(json_bytes: []const u8) TemplateError!ChatTemplate {
        // Pass 1: locate top-level `"dialect"` regardless of JSON key ordering.
        var detected_dialect: Dialect = .chatml;
        {
            var r1 = JsonScan{ .src = json_bytes };
            try r1.expectByte('{');
            if (r1.peek() != '}') {
                while (true) {
                    var kbuf: [64]u8 = undefined;
                    const mk = try r1.readKeyOrSkip(&kbuf);
                    try r1.expectByte(':');
                    if (mk != null and std.mem.eql(u8, mk.?, "dialect")) {
                        var dbuf: [32]u8 = undefined;
                        const ds = try r1.readSmallString(&dbuf);
                        detected_dialect = try Dialect.parse(ds);
                    } else {
                        try r1.skipValue(1);
                    }
                    const sep = r1.peek() orelse return error.InvalidJson;
                    if (sep == ',') {
                        r1.pos += 1;
                    } else if (sep == '}') {
                        r1.pos += 1;
                        break;
                    } else return error.InvalidJson;
                }
            }
        }

        // Pass 2: start from dialect preset and apply every present override.
        var tpl = preset(detected_dialect);
        var r = JsonScan{ .src = json_bytes };
        try r.expectByte('{');
        if (r.peek() != '}') {
            while (true) {
                var kbuf: [64]u8 = undefined;
                const mk = try r.readKeyOrSkip(&kbuf);
                try r.expectByte(':');
                if (mk) |key| {
                    if (std.mem.eql(u8, key, "schema_version")) {
                        tpl.schema_version = try r.readU32();
                        if (tpl.schema_version == 0 or tpl.schema_version > SCHEMA_VERSION) return error.InvalidSpec;
                    } else if (std.mem.eql(u8, key, "dialect")) {
                        var dbuf: [32]u8 = undefined;
                        const ds = try r.readSmallString(&dbuf);
                        tpl.dialect = try Dialect.parse(ds);
                    } else if (std.mem.eql(u8, key, "bos_token")) {
                        try readSmallStrOpt(&r, &tpl.bos_token);
                    } else if (std.mem.eql(u8, key, "eos_token")) {
                        try readSmallStrOpt(&r, &tpl.eos_token);
                    } else if (std.mem.eql(u8, key, "add_generation_prompt")) {
                        if (!r.consumeKeyword("null")) tpl.add_generation_prompt = try r.readBool();
                    } else if (std.mem.eql(u8, key, "role_tokens")) {
                        try parseRoleTokensObject(&r, &tpl.role_tokens);
                    } else if (std.mem.eql(u8, key, "thinking")) {
                        try parseThinkingObject(&r, &tpl.thinking);
                    } else if (std.mem.eql(u8, key, "tool_calling")) {
                        try parseToolCallingObject(&r, &tpl.tool_calling);
                    } else if (std.mem.eql(u8, key, "multimodal")) {
                        try parseMultimodalObject(&r, &tpl.multimodal);
                    } else {
                        try r.skipValue(1);
                    }
                } else {
                    try r.skipValue(1);
                }
                const sep = r.peek() orelse return error.InvalidJson;
                if (sep == ',') {
                    r.pos += 1;
                } else if (sep == '}') {
                    r.pos += 1;
                    break;
                } else return error.InvalidJson;
            }
        } else {
            r.pos += 1;
        }
        if (r.peek() != null) return error.InvalidJson;
        return tpl;
    }

    pub fn generationStartsInThink(self: *const ChatTemplate, effort_opt: ?ReasoningEffort, enable_thinking_opt: ?bool) bool {
        const eff = self.resolveEffort(effort_opt, enable_thinking_opt);
        if (!eff.isThinkingEnabled() or !self.thinking.supported) return false;
        return self.thinking.starts_in_think_on_think;
    }

    pub fn resolveEffort(self: *const ChatTemplate, effort_opt: ?ReasoningEffort, enable_thinking_opt: ?bool) ReasoningEffort {
        if (effort_opt) |e| return e;
        if (enable_thinking_opt) |en| {
            if (!en) return .none;
        }
        return self.thinking.default_effort;
    }

    /// Formats a completed assistant turn suffix/wrapper for stateful `/v1/responses` transcript chaining.
    pub fn assistantEndDelimiter(self: *const ChatTemplate) []const u8 {
        return switch (self.dialect) {
            .inkling2 => "<|content_model_end_sampling|>",
            .hy4 => self.role_tokens.hy4_end.slice(),
            .deepseek => self.role_tokens.assistant_suffix.slice(),
            .chatml, .custom => self.role_tokens.assistant_suffix.slice(),
        };
    }

    /// Returns the active thinking close tag for budget cutoff or ThinkSplitter.
    pub fn thinkingCloseTag(self: *const ChatTemplate) []const u8 {
        return self.thinking.close_tag.slice();
    }

    /// Deterministic zero-allocation rendering of `messages` into `dst_buf`.
    pub fn render(
        self: *const ChatTemplate,
        dst_buf: []u8,
        messages: []const MessageView,
        opts: RenderOptions,
    ) TemplateError!RenderResult {
        var fw = FixedWriter.init(dst_buf);
        const has_leading_sys = messages.len > 0 and messages[0].role == .system;
        var last_user_idx: ?usize = null;
        for (messages, 0..) |m, idx| {
            if (m.role == .user) last_user_idx = idx;
        }
        var renderer = PromptRenderer.init(self, &fw, opts, has_leading_sys, last_user_idx);
        try renderer.begin();
        for (messages, 0..) |m, idx| {
            try renderer.pushMessage(m, idx);
        }
        try renderer.finish();
        return .{
            .prompt = fw.written(),
            .last_user_msg_byte_offset = renderer.last_user_msg_byte_offset,
            .starts_in_think = renderer.starts_in_think,
        };
    }
};

fn parseRoleTokensObject(r: *JsonScan, rt: *RoleTokens) TemplateError!void {
    if (r.consumeKeyword("null")) return;
    try r.expectByte('{');
    if (r.peek() == '}') {
        r.pos += 1;
        return;
    }
    while (true) {
        var kbuf: [64]u8 = undefined;
        const mk = try r.readKeyOrSkip(&kbuf);
        try r.expectByte(':');
        if (mk) |key| {
            if (std.mem.eql(u8, key, "system_prefix")) {
                try readSmallStrOpt(r, &rt.system_prefix);
            } else if (std.mem.eql(u8, key, "system_suffix")) {
                try readSmallStrOpt(r, &rt.system_suffix);
            } else if (std.mem.eql(u8, key, "user_prefix")) {
                try readSmallStrOpt(r, &rt.user_prefix);
            } else if (std.mem.eql(u8, key, "user_suffix")) {
                try readSmallStrOpt(r, &rt.user_suffix);
            } else if (std.mem.eql(u8, key, "assistant_prefix")) {
                try readSmallStrOpt(r, &rt.assistant_prefix);
            } else if (std.mem.eql(u8, key, "assistant_suffix")) {
                try readSmallStrOpt(r, &rt.assistant_suffix);
            } else if (std.mem.eql(u8, key, "tool_prefix")) {
                try readSmallStrOpt(r, &rt.tool_prefix);
            } else if (std.mem.eql(u8, key, "tool_suffix")) {
                try readSmallStrOpt(r, &rt.tool_suffix);
            } else if (std.mem.eql(u8, key, "content_text_tag")) {
                try readSmallStrOpt(r, &rt.content_text_tag);
            } else if (std.mem.eql(u8, key, "end_message_tag")) {
                try readSmallStrOpt(r, &rt.end_message_tag);
            } else if (std.mem.eql(u8, key, "hy4_start")) {
                try readSmallStrOpt(r, &rt.hy4_start);
            } else if (std.mem.eql(u8, key, "hy4_middle")) {
                try readSmallStrOpt(r, &rt.hy4_middle);
            } else if (std.mem.eql(u8, key, "hy4_end")) {
                try readSmallStrOpt(r, &rt.hy4_end);
            } else if (std.mem.eql(u8, key, "header_role_tokens")) {
                if (!r.consumeKeyword("null")) {
                    try r.expectByte('{');
                    if (r.peek() != '}') {
                        while (true) {
                            var hk_buf: [32]u8 = undefined;
                            const mhk = try r.readKeyOrSkip(&hk_buf);
                            try r.expectByte(':');
                            if (mhk) |hk| {
                                if (std.mem.eql(u8, hk, "system")) {
                                    try readSmallStrOpt(r, &rt.header_system);
                                } else if (std.mem.eql(u8, hk, "user")) {
                                    try readSmallStrOpt(r, &rt.header_user);
                                } else if (std.mem.eql(u8, hk, "assistant")) {
                                    try readSmallStrOpt(r, &rt.header_assistant);
                                } else if (std.mem.eql(u8, hk, "tool")) {
                                    try readSmallStrOpt(r, &rt.header_tool);
                                } else {
                                    try r.skipValue(3);
                                }
                            } else {
                                try r.skipValue(3);
                            }
                            const hsep = r.peek() orelse return error.InvalidJson;
                            if (hsep == ',') {
                                r.pos += 1;
                            } else if (hsep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                }
            } else {
                try r.skipValue(2);
            }
        } else {
            try r.skipValue(2);
        }
        const sep = r.peek() orelse return error.InvalidJson;
        if (sep == ',') {
            r.pos += 1;
        } else if (sep == '}') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

fn parseThinkingObject(r: *JsonScan, th: *ThinkingSpec) TemplateError!void {
    if (r.consumeKeyword("null")) return;
    try r.expectByte('{');
    if (r.peek() == '}') {
        r.pos += 1;
        return;
    }
    while (true) {
        var kbuf: [64]u8 = undefined;
        const mk = try r.readKeyOrSkip(&kbuf);
        try r.expectByte(':');
        if (mk) |key| {
            if (std.mem.eql(u8, key, "supported")) {
                th.supported = try r.readBool();
            } else if (std.mem.eql(u8, key, "open_tag")) {
                try readSmallStrOpt(r, &th.open_tag);
            } else if (std.mem.eql(u8, key, "close_tag")) {
                try readSmallStrOpt(r, &th.close_tag);
            } else if (std.mem.eql(u8, key, "prefill_think")) {
                try readSmallStrOpt(r, &th.prefill_think);
            } else if (std.mem.eql(u8, key, "prefill_no_think")) {
                try readSmallStrOpt(r, &th.prefill_no_think);
            } else if (std.mem.eql(u8, key, "reasoning_mode_token")) {
                try readSmallStrOpt(r, &th.reasoning_mode_token);
            } else if (std.mem.eql(u8, key, "starts_in_think_on_think")) {
                th.starts_in_think_on_think = try r.readBool();
            } else if (std.mem.eql(u8, key, "preserved_thinking")) {
                if (r.consumeKeyword("null")) {
                    th.preserved_thinking = null;
                } else {
                    th.preserved_thinking = try r.readBool();
                }
            } else if (std.mem.eql(u8, key, "effort_mode")) {
                var ebuf: [64]u8 = undefined;
                const es = try r.readSmallString(&ebuf);
                th.effort_mode = try EffortMode.parse(es);
            } else if (std.mem.eql(u8, key, "default_effort")) {
                if (!r.consumeKeyword("null")) {
                    if (r.peek() == '"') {
                        var dbuf: [32]u8 = undefined;
                        const ds = try r.readSmallString(&dbuf);
                        th.default_effort = ReasoningEffort.parseCliArg(ds) catch return error.InvalidSpec;
                    } else {
                        const fv = try r.readF64();
                        th.default_effort = ReasoningEffort.fromFloat(fv) catch return error.InvalidSpec;
                    }
                }
            } else if (std.mem.eql(u8, key, "effort_map")) {
                if (!r.consumeKeyword("null")) {
                    try r.expectByte('{');
                    if (r.peek() != '}') {
                        while (true) {
                            var mk_buf: [32]u8 = undefined;
                            const mkey = try r.readKeyOrSkip(&mk_buf);
                            try r.expectByte(':');
                            const fv = try r.readF64();
                            if (fv < 0.0 or fv > 0.99) return error.InvalidSpec;
                            if (mkey) |ek| {
                                if (std.mem.eql(u8, ek, "none")) th.effort_map.none = fv else if (std.mem.eql(u8, ek, "minimal")) th.effort_map.minimal = fv else if (std.mem.eql(u8, ek, "low")) th.effort_map.low = fv else if (std.mem.eql(u8, ek, "medium")) th.effort_map.medium = fv else if (std.mem.eql(u8, ek, "high")) th.effort_map.high = fv else if (std.mem.eql(u8, ek, "xhigh")) th.effort_map.xhigh = fv else if (std.mem.eql(u8, ek, "max")) th.effort_map.max = fv;
                            }
                            const msep = r.peek() orelse return error.InvalidJson;
                            if (msep == ',') {
                                r.pos += 1;
                            } else if (msep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                }
            } else {
                try r.skipValue(2);
            }
        } else {
            try r.skipValue(2);
        }
        const sep = r.peek() orelse return error.InvalidJson;
        if (sep == ',') {
            r.pos += 1;
        } else if (sep == '}') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

fn parseToolCallingObject(r: *JsonScan, tc: *ToolCallingSpec) TemplateError!void {
    if (r.consumeKeyword("null")) return;
    try r.expectByte('{');
    if (r.peek() == '}') {
        r.pos += 1;
        return;
    }
    while (true) {
        var kbuf: [64]u8 = undefined;
        const mk = try r.readKeyOrSkip(&kbuf);
        try r.expectByte(':');
        if (mk) |key| {
            if (std.mem.eql(u8, key, "format")) {
                var fbuf: [64]u8 = undefined;
                const fs = try r.readSmallString(&fbuf);
                tc.format = try ToolCallFormat.parse(fs);
            } else if (std.mem.eql(u8, key, "declare_prefix")) {
                try readSmallStrOpt(r, &tc.declare_prefix);
            } else if (std.mem.eql(u8, key, "declare_suffix")) {
                try readSmallStrOpt(r, &tc.declare_suffix);
            } else if (std.mem.eql(u8, key, "call_open")) {
                try readSmallStrOpt(r, &tc.call_open);
            } else if (std.mem.eql(u8, key, "call_close")) {
                try readSmallStrOpt(r, &tc.call_close);
            } else if (std.mem.eql(u8, key, "response_open")) {
                try readSmallStrOpt(r, &tc.response_open);
            } else if (std.mem.eql(u8, key, "response_close")) {
                try readSmallStrOpt(r, &tc.response_close);
            } else {
                try r.skipValue(2);
            }
        } else {
            try r.skipValue(2);
        }
        const sep = r.peek() orelse return error.InvalidJson;
        if (sep == ',') {
            r.pos += 1;
        } else if (sep == '}') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

fn parseMultimodalObject(r: *JsonScan, mm: *MultimodalSpec) TemplateError!void {
    if (r.consumeKeyword("null")) return;
    try r.expectByte('{');
    if (r.peek() == '}') {
        r.pos += 1;
        return;
    }
    while (true) {
        var kbuf: [64]u8 = undefined;
        const mk = try r.readKeyOrSkip(&kbuf);
        try r.expectByte(':');
        if (mk) |key| {
            if (std.mem.eql(u8, key, "image_tag")) {
                try readSmallStrOpt(r, &mm.image_tag);
            } else if (std.mem.eql(u8, key, "video_tag")) {
                try readSmallStrOpt(r, &mm.video_tag);
            } else {
                try r.skipValue(2);
            }
        } else {
            try r.skipValue(2);
        }
        const sep = r.peek() orelse return error.InvalidJson;
        if (sep == ',') {
            r.pos += 1;
        } else if (sep == '}') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

// ---------------------------------------------------------------------------
// Zero-allocation incremental `PromptRenderer` state machine
// ---------------------------------------------------------------------------

pub const MAX_TRACKED_TOOL_CALL_NAMES: usize = 64;

pub const PromptRenderer = struct {
    tpl: *const ChatTemplate,
    dst: *FixedWriter,
    opts: RenderOptions,
    effort: ReasoningEffort,
    enable_thinking: bool,
    preserved_thinking: bool,
    add_generation_prompt: bool,
    has_leading_system: bool,
    last_user_idx: ?usize,

    effort_emitted: bool = false,
    in_hy4_tool_cluster: bool = false,
    last_is_assistant: bool = false,
    has_prior_messages: bool = false,
    message_count: usize = 0,
    last_user_msg_byte_offset: ?usize = null,
    starts_in_think: bool = false,
    image_count: u32 = 0,
    video_count: u32 = 0,

    // Bounded history map `tool_call_id -> function_name` for Inkling 2 `role="tool"` header resolution.
    tracked_ids: [MAX_TRACKED_TOOL_CALL_NAMES]SmallStr(48) = @splat(.{}),
    tracked_names: [MAX_TRACKED_TOOL_CALL_NAMES]SmallStr(64) = @splat(.{}),
    tracked_count: usize = 0,

    pub fn init(
        tpl: *const ChatTemplate,
        dst: *FixedWriter,
        opts: RenderOptions,
        has_leading_system: bool,
        last_user_idx: ?usize,
    ) PromptRenderer {
        const has_tools = (opts.tools.len > 0 or opts.raw_tools_json != null) and opts.tool_choice != .none;
        const eff = tpl.resolveEffort(opts.reasoning_effort, opts.enable_thinking);
        const en_think = if (opts.enable_thinking) |b| b else eff.isThinkingEnabled();
        const preserved = opts.preserved_thinking orelse
            tpl.thinking.preserved_thinking orelse
            (tpl.dialect == .hy4 and has_tools);
        const add_gen = opts.add_generation_prompt orelse tpl.add_generation_prompt;

        return .{
            .tpl = tpl,
            .dst = dst,
            .opts = opts,
            .effort = eff,
            .enable_thinking = en_think,
            .preserved_thinking = preserved,
            .add_generation_prompt = add_gen,
            .has_leading_system = has_leading_system,
            .last_user_idx = last_user_idx,
            .starts_in_think = tpl.generationStartsInThink(eff, en_think),
        };
    }

    pub fn begin(self: *PromptRenderer) TemplateError!void {
        if (self.opts.prior_prompt) |prior| {
            if (prior.len > 0) {
                try self.dst.writeAll(prior);
                self.has_prior_messages = true;
                self.effort_emitted = true;
            }
        } else if (!self.tpl.bos_token.isEmpty() and self.tpl.dialect != .hy4) {
            try self.dst.writeAll(self.tpl.bos_token.slice());
        }

        const has_tools = (self.opts.tools.len > 0 or self.opts.raw_tools_json != null) and self.opts.tool_choice != .none;

        switch (self.tpl.dialect) {
            .inkling2 => {
                if (has_tools) {
                    try self.emitInkling2ToolsDeclare();
                    self.has_prior_messages = true;
                }
            },
            .hy4 => {
                if (!self.has_leading_system and self.opts.prior_prompt == null) {
                    try self.dst.writeAll(self.tpl.role_tokens.hy4_start.slice());
                    try self.dst.writeAll("system");
                    try self.dst.writeAll(self.tpl.role_tokens.hy4_middle.slice());
                    if (has_tools) {
                        try self.emitHy4ToolsPrompt();
                    }
                    try self.emitHy4ReasoningMode();
                    try self.dst.writeAll(self.tpl.role_tokens.hy4_end.slice());
                    self.effort_emitted = true;
                    self.has_prior_messages = true;
                }
            },
            .chatml, .deepseek, .custom => {
                if (has_tools) {
                    try self.emitQwenToolsPrompt();
                    self.has_prior_messages = true;
                }
            },
        }
    }

    pub fn pushMessage(self: *PromptRenderer, msg: MessageView, msg_idx: usize) TemplateError!void {
        if (self.tpl.dialect == .hy4 and msg.role != .tool and self.in_hy4_tool_cluster) {
            try self.dst.writeAll(self.tpl.role_tokens.hy4_end.slice());
            self.in_hy4_tool_cluster = false;
        }

        switch (self.tpl.dialect) {
            .inkling2 => try self.pushInkling2Message(msg),
            .hy4 => try self.pushHy4Message(msg, msg_idx),
            .chatml, .deepseek, .custom => try self.pushGenericMessage(msg),
        }

        self.last_is_assistant = (msg.role == .assistant);
        self.message_count += 1;
        self.has_prior_messages = true;
    }

    pub fn finish(self: *PromptRenderer) TemplateError!void {
        if (self.in_hy4_tool_cluster) {
            try self.dst.writeAll(self.tpl.role_tokens.hy4_end.slice());
            self.in_hy4_tool_cluster = false;
        }
        if (self.tpl.dialect == .inkling2 and !self.effort_emitted) {
            try self.emitInkling2EffortDirective();
            self.effort_emitted = true;
        }

        if (!self.add_generation_prompt or self.last_is_assistant) return;

        switch (self.tpl.dialect) {
            .inkling2 => {},
            .hy4 => {
                try self.dst.writeAll(self.tpl.role_tokens.assistant_prefix.slice());
                if (self.enable_thinking and self.effort.isThinkingEnabled()) {
                    try self.dst.writeAll(self.tpl.thinking.prefill_think.slice());
                } else {
                    try self.dst.writeAll(self.tpl.thinking.prefill_no_think.slice());
                }
            },
            .chatml, .deepseek, .custom => {
                try self.dst.writeAll(self.tpl.role_tokens.assistant_prefix.slice());
                if (self.enable_thinking and self.effort.isThinkingEnabled()) {
                    try self.dst.writeAll(self.tpl.thinking.prefill_think.slice());
                } else if (self.opts.repl_compact_nothink and self.tpl.dialect == .chatml) {
                    try self.dst.writeAll("<think>\n\n</think>\n");
                } else {
                    try self.dst.writeAll(self.tpl.thinking.prefill_no_think.slice());
                }
            },
        }
    }

    fn recordToolCall(self: *PromptRenderer, tc: ToolCallView) void {
        if (tc.id.len == 0 or tc.name.len == 0 or self.tracked_count >= MAX_TRACKED_TOOL_CALL_NAMES) return;
        self.tracked_ids[self.tracked_count] = SmallStr(48).fromSlice(tc.id) catch return;
        self.tracked_names[self.tracked_count] = SmallStr(64).fromSlice(tc.name) catch return;
        self.tracked_count += 1;
    }

    fn resolveToolName(self: *const PromptRenderer, msg: MessageView) []const u8 {
        if (msg.name.len > 0) return msg.name;
        if (msg.tool_call_id.len > 0) {
            var idx = self.tracked_count;
            while (idx > 0) {
                idx -= 1;
                if (std.mem.eql(u8, self.tracked_ids[idx].slice(), msg.tool_call_id)) {
                    return self.tracked_names[idx].slice();
                }
            }
        }
        return "";
    }

    fn emitInkling2EffortDirective(self: *PromptRenderer) TemplateError!void {
        var eff_buf: [32]u8 = undefined;
        const eff_str = try self.effort.formatInklingValue(&self.tpl.thinking.effort_map, &eff_buf);
        try self.dst.writeAll(self.tpl.role_tokens.header_system.slice());
        try self.dst.writeAll(self.tpl.role_tokens.content_text_tag.slice());
        try self.dst.writeAll("Thinking effort level: ");
        try self.dst.writeAll(eff_str);
        try self.dst.writeAll(self.tpl.role_tokens.end_message_tag.slice());
    }

    fn emitInkling2ToolsDeclare(self: *PromptRenderer) TemplateError!void {
        try self.dst.writeAll(self.tpl.tool_calling.declare_prefix.slice());
        try self.dst.writeByte('[');
        if (self.opts.tools.len > 0) {
            for (self.opts.tools, 0..) |t, i| {
                if (i > 0) try self.dst.writeByte(',');
                try self.dst.writeAll("{\"description\":\"");
                try self.dst.writeJsonEscaped(t.description);
                try self.dst.writeAll("\",\"name\":\"");
                try self.dst.writeJsonEscaped(t.name);
                try self.dst.writeAll("\",\"parameters\":");
                var pr = JsonScan{ .src = if (t.parameters_json.len > 0) t.parameters_json else "{}" };
                try writeCanonicalJsonValue(&pr, self.dst, 0, true, false);
                try self.dst.writeAll(",\"type\":\"");
                try self.dst.writeJsonEscaped(t.tool_type);
                try self.dst.writeAll("\"}");
            }
        } else if (self.opts.raw_tools_json) |raw| {
            var rr = JsonScan{ .src = raw };
            try rr.expectByte('[');
            if (rr.peek() != ']') {
                var idx: usize = 0;
                while (true) {
                    const obj = try rr.captureValueSlice(1);
                    if (idx > 0) try self.dst.writeByte(',');
                    idx += 1;
                    try emitInkling2NormalizedToolObj(obj, self.dst);
                    const sep = rr.peek() orelse return error.InvalidJson;
                    if (sep == ',') {
                        rr.pos += 1;
                    } else if (sep == ']') {
                        break;
                    } else return error.InvalidJson;
                }
            }
        }
        try self.dst.writeByte(']');
        try self.dst.writeAll(self.tpl.tool_calling.declare_suffix.slice());
    }

    fn emitHy4ReasoningMode(self: *PromptRenderer) TemplateError!void {
        try self.dst.writeAll(self.tpl.thinking.reasoning_mode_token.slice());
        try self.dst.writeAll("reasoning_effort:");
        try self.dst.writeAll(self.effort.toHy4String());
    }

    fn emitHy4ToolsPrompt(self: *PromptRenderer) TemplateError!void {
        try self.dst.writeAll("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        if (self.opts.tools.len > 0) {
            for (self.opts.tools, 0..) |t, i| {
                if (i > 0) try self.dst.writeByte('\n');
                try self.dst.writeAll("{\"name\": \"");
                try self.dst.writeJsonEscaped(t.name);
                try self.dst.writeAll("\", \"description\": \"");
                try self.dst.writeJsonEscaped(t.description);
                try self.dst.writeAll("\", \"parameters\": ");
                var pr = JsonScan{ .src = if (t.parameters_json.len > 0) t.parameters_json else "{}" };
                try writeCanonicalJsonValue(&pr, self.dst, 0, false, true);
                try self.dst.writeByte('}');
            }
        } else if (self.opts.raw_tools_json) |raw| {
            var rr = JsonScan{ .src = raw };
            try rr.expectByte('[');
            if (rr.peek() != ']') {
                var idx: usize = 0;
                while (true) {
                    const obj = try rr.captureValueSlice(1);
                    if (idx > 0) try self.dst.writeByte('\n');
                    idx += 1;
                    try emitHy4NormalizedToolObj(obj, self.dst);
                    const sep = rr.peek() orelse return error.InvalidJson;
                    if (sep == ',') {
                        rr.pos += 1;
                    } else if (sep == ']') {
                        break;
                    } else return error.InvalidJson;
                }
            }
        }
        try self.dst.writeAll(
            "\n</tools>\n\nFor function call returns, you should first print <tool_calls:opensource>\n" ++
                "For each function call, you should return object like:\n" ++
                "<tool_call:opensource>{function-name}" ++
                "<arg_key:opensource>{arg-key-1}</arg_key:opensource>" ++
                "<arg_value:opensource>{arg-value-1}</arg_value:opensource>" ++
                "<arg_key:opensource>{arg-key-2}</arg_key:opensource>" ++
                "<arg_value:opensource>{arg-value-2}</arg_value:opensource>...</tool_call:opensource>\n" ++
                "At the end of function call returns, you should print </tool_calls:opensource>",
        );
    }

    fn emitQwenToolsPrompt(self: *PromptRenderer) TemplateError!void {
        try self.dst.writeAll(self.tpl.role_tokens.system_prefix.slice());
        try self.dst.writeAll("# Tools\nYou may call one or more functions to assist with the user query.\n");
        switch (self.opts.tool_choice) {
            .auto => {},
            .none => try self.dst.writeAll("Do NOT call any tools.\n"),
            .required => try self.dst.writeAll("You MUST call at least one tool.\n"),
            .named => {
                try self.dst.writeAll("You MUST call the tool named \"");
                try self.dst.writeAll(self.opts.tool_choice_name);
                try self.dst.writeAll("\".\n");
            },
        }
        if (!self.opts.parallel_tool_calls) {
            try self.dst.writeAll("Call at most one tool per turn.\n");
        }
        try self.dst.writeAll("Return each tool call inside <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>.\n<tools>\n");
        if (self.opts.raw_tools_json) |raw| {
            try self.dst.writeAll(raw);
        } else {
            try self.dst.writeByte('[');
            for (self.opts.tools, 0..) |t, i| {
                if (i > 0) try self.dst.writeByte(',');
                try self.dst.writeAll("{\"type\":\"");
                try self.dst.writeJsonEscaped(t.tool_type);
                try self.dst.writeAll("\",\"function\":{\"name\":\"");
                try self.dst.writeJsonEscaped(t.name);
                try self.dst.writeAll("\",\"description\":\"");
                try self.dst.writeJsonEscaped(t.description);
                try self.dst.writeAll("\",\"parameters\":");
                try self.dst.writeAll(if (t.parameters_json.len > 0) t.parameters_json else "{}");
                try self.dst.writeAll("}}");
            }
            try self.dst.writeByte(']');
        }
        try self.dst.writeAll("\n</tools>");
        try self.dst.writeAll(self.tpl.role_tokens.system_suffix.slice());
    }

    fn pushInkling2Message(self: *PromptRenderer, msg: MessageView) TemplateError!void {
        if (!self.effort_emitted and msg.role != .system) {
            try self.emitInkling2EffortDirective();
            self.effort_emitted = true;
            self.has_prior_messages = true;
        }
        if (msg.role == .user and self.has_prior_messages) {
            self.last_user_msg_byte_offset = self.dst.pos;
        }

        const role_tok = switch (msg.role) {
            .system => self.tpl.role_tokens.header_system.slice(),
            .user => self.tpl.role_tokens.header_user.slice(),
            .assistant => self.tpl.role_tokens.header_assistant.slice(),
            .tool => self.tpl.role_tokens.header_tool.slice(),
        };
        const end_tok = self.tpl.role_tokens.end_message_tag.slice();
        const text_tag = self.tpl.role_tokens.content_text_tag.slice();

        var name_suffix: []const u8 = "";
        if (msg.role == .tool) {
            name_suffix = self.resolveToolName(msg);
        } else if (msg.role == .assistant and msg.name.len > 0) {
            name_suffix = msg.name;
        }

        if (msg.role == .assistant and msg.reasoning_content.len > 0) {
            try self.dst.writeAll(role_tok);
            try self.dst.writeAll(name_suffix);
            try self.dst.writeAll(self.tpl.thinking.open_tag.slice());
            try self.dst.writeAll(msg.reasoning_content);
            try self.dst.writeAll(end_tok);
        }

        if (msg.parts.len > 0) {
            for (msg.parts) |part| {
                switch (part) {
                    .text => |t| {
                        try self.dst.writeAll(role_tok);
                        try self.dst.writeAll(name_suffix);
                        try self.dst.writeAll(text_tag);
                        try self.dst.writeAll(t);
                        try self.dst.writeAll(end_tok);
                    },
                    .image => {
                        if (msg.role != .user and msg.role != .tool) return error.UnsupportedContent;
                        try self.dst.writeAll(role_tok);
                        try self.dst.writeAll(name_suffix);
                        try self.dst.writeAll(self.tpl.multimodal.image_tag.slice());
                        try self.dst.writeAll(end_tok);
                    },
                    .video => return error.UnsupportedContent,
                }
            }
        } else if (msg.content.len > 0 or !(msg.role == .assistant and (msg.tool_calls.len > 0 or msg.reasoning_content.len > 0))) {
            try self.dst.writeAll(role_tok);
            try self.dst.writeAll(name_suffix);
            try self.dst.writeAll(text_tag);
            try self.dst.writeAll(msg.content);
            try self.dst.writeAll(end_tok);
        }

        if (msg.role == .assistant and msg.tool_calls.len > 0) {
            for (msg.tool_calls) |tc| {
                if (tc.name.len == 0) return error.MissingToolName;
                if (msg.name.len > 0 and !std.mem.eql(u8, msg.name, tc.name)) return error.InvalidSpec;
                self.recordToolCall(tc);
                try self.dst.writeAll(role_tok);
                try self.dst.writeAll(tc.name);
                try self.dst.writeAll(self.tpl.tool_calling.call_open.slice());
                try self.dst.writeAll("{\"name\":\"");
                try self.dst.writeJsonEscaped(tc.name);
                try self.dst.writeAll("\",\"args\":");
                var ar = JsonScan{ .src = if (tc.arguments_json.len > 0) tc.arguments_json else "{}" };
                try writeCanonicalJsonValue(&ar, self.dst, 0, true, false);
                try self.dst.writeByte('}');
                try self.dst.writeAll(self.tpl.tool_calling.call_close.slice());
            }
        }

        if (msg.role == .assistant) {
            try self.dst.writeAll(self.tpl.eos_token.slice());
        }
    }

    fn pushHy4Message(self: *PromptRenderer, msg: MessageView, msg_idx: usize) TemplateError!void {
        const hy_start = self.tpl.role_tokens.hy4_start.slice();
        const hy_mid = self.tpl.role_tokens.hy4_middle.slice();
        const hy_end = self.tpl.role_tokens.hy4_end.slice();
        const has_tools = (self.opts.tools.len > 0 or self.opts.raw_tools_json != null) and self.opts.tool_choice != .none;

        switch (msg.role) {
            .system => {
                try self.dst.writeAll(hy_start);
                try self.dst.writeAll("system");
                try self.dst.writeAll(hy_mid);
                if (self.message_count == 0 and self.opts.prior_prompt == null) {
                    if (has_tools) {
                        try self.emitHy4ToolsPrompt();
                    }
                    const sys_pos = self.dst.pos;
                    try self.writePartsOrText(msg, true);
                    if (has_tools and self.dst.pos > sys_pos) {
                        // Insert `\n\n` before system content when tools are also present.
                        const wrote_len = self.dst.pos - sys_pos;
                        if (self.dst.remaining() < 2) return error.BufferTooSmall;
                        std.mem.copyBackwards(u8, self.dst.buf[sys_pos + 2 .. sys_pos + 2 + wrote_len], self.dst.buf[sys_pos .. sys_pos + wrote_len]);
                        self.dst.buf[sys_pos] = '\n';
                        self.dst.buf[sys_pos + 1] = '\n';
                        self.dst.pos += 2;
                    }
                    try self.emitHy4ReasoningMode();
                    self.effort_emitted = true;
                } else {
                    try self.writePartsOrText(msg, true);
                }
                try self.dst.writeAll(hy_end);
            },
            .user => {
                if (self.has_prior_messages) {
                    self.last_user_msg_byte_offset = self.dst.pos;
                }
                try self.dst.writeAll(hy_start);
                try self.dst.writeAll("user");
                try self.dst.writeAll(hy_mid);
                try self.writePartsOrText(msg, false);
                try self.dst.writeAll(hy_end);
            },
            .assistant => {
                try self.dst.writeAll(hy_start);
                try self.dst.writeAll("assistant");
                try self.dst.writeAll(hy_mid);
                const keep_reason = self.effort.isThinkingEnabled() and
                    (self.preserved_thinking or (if (self.last_user_idx) |lui| msg_idx > lui else true)) and
                    msg.reasoning_content.len > 0;
                try self.dst.writeAll(self.tpl.thinking.open_tag.slice());
                if (keep_reason) {
                    try self.dst.writeAll(msg.reasoning_content);
                }
                try self.dst.writeAll(self.tpl.thinking.close_tag.slice());
                try self.writePartsOrText(msg, false);
                if (msg.tool_calls.len > 0) {
                    try self.dst.writeAll(self.tpl.tool_calling.call_open.slice());
                    for (msg.tool_calls) |tc| {
                        self.recordToolCall(tc);
                        try self.dst.writeAll("<tool_call:opensource>");
                        try self.dst.writeAll(tc.name);
                        try emitHy4ToolArgs(if (tc.arguments_json.len > 0) tc.arguments_json else "{}", self.dst);
                        try self.dst.writeAll("</tool_call:opensource>");
                    }
                    try self.dst.writeAll(self.tpl.tool_calling.call_close.slice());
                }
                try self.dst.writeAll(hy_end);
            },
            .tool => {
                if (!self.in_hy4_tool_cluster) {
                    try self.dst.writeAll(hy_start);
                    try self.dst.writeAll("tool");
                    try self.dst.writeAll(hy_mid);
                    self.in_hy4_tool_cluster = true;
                }
                try self.dst.writeAll(self.tpl.tool_calling.response_open.slice());
                try self.writePartsOrText(msg, false);
                try self.dst.writeAll(self.tpl.tool_calling.response_close.slice());
            },
        }
    }

    fn pushGenericMessage(self: *PromptRenderer, msg: MessageView) TemplateError!void {
        switch (msg.role) {
            .system => {
                try self.dst.writeAll(self.tpl.role_tokens.system_prefix.slice());
                try self.writePartsOrText(msg, false);
                try self.dst.writeAll(self.tpl.role_tokens.system_suffix.slice());
            },
            .user => {
                if (self.has_prior_messages) {
                    self.last_user_msg_byte_offset = self.dst.pos;
                }
                try self.dst.writeAll(self.tpl.role_tokens.user_prefix.slice());
                try self.writePartsOrText(msg, false);
                try self.dst.writeAll(self.tpl.role_tokens.user_suffix.slice());
            },
            .assistant => {
                try self.dst.writeAll(self.tpl.role_tokens.assistant_prefix.slice());
                try self.writePartsOrText(msg, false);
                for (msg.tool_calls) |tc| {
                    try self.dst.writeAll(self.tpl.tool_calling.call_open.slice());
                    try self.dst.writeAll("{\"name\":\"");
                    try self.dst.writeJsonEscaped(tc.name);
                    try self.dst.writeAll("\",\"arguments\":");
                    try self.dst.writeAll(if (tc.arguments_json.len > 0) tc.arguments_json else "{}");
                    try self.dst.writeByte('}');
                    try self.dst.writeAll(self.tpl.tool_calling.call_close.slice());
                }
                try self.dst.writeAll(self.tpl.role_tokens.assistant_suffix.slice());
            },
            .tool => {
                if (self.has_prior_messages and self.tpl.dialect == .chatml) {
                    self.last_user_msg_byte_offset = self.dst.pos;
                }
                try self.dst.writeAll(self.tpl.role_tokens.tool_prefix.slice());
                if (msg.tool_call_id.len > 0 and self.tpl.dialect == .chatml) {
                    try self.dst.writeAll("[tool_call_id=");
                    try self.dst.writeAll(msg.tool_call_id);
                    try self.dst.writeAll("]\n");
                }
                try self.writePartsOrText(msg, false);
                try self.dst.writeAll(self.tpl.role_tokens.tool_suffix.slice());
            },
        }
    }

    fn writePartsOrText(self: *PromptRenderer, msg: MessageView, is_system: bool) TemplateError!void {
        if (msg.parts.len == 0) {
            try self.dst.writeAll(msg.content);
            return;
        }
        for (msg.parts) |part| {
            switch (part) {
                .text => |t| try self.dst.writeAll(t),
                .image => {
                    if (is_system and self.tpl.dialect == .hy4) return error.UnsupportedContent;
                    self.image_count += 1;
                    if (self.opts.add_vision_id and self.tpl.dialect == .hy4) {
                        try self.dst.print("Picture {d}: ", .{self.image_count});
                    }
                    try self.dst.writeAll(self.tpl.multimodal.image_tag.slice());
                },
                .video => {
                    if (is_system and self.tpl.dialect == .hy4) return error.UnsupportedContent;
                    self.video_count += 1;
                    if (self.opts.add_vision_id and self.tpl.dialect == .hy4) {
                        try self.dst.print("Video {d}: ", .{self.video_count});
                    }
                    try self.dst.writeAll(self.tpl.multimodal.video_tag.slice());
                },
            }
        }
    }
};

fn emitInkling2NormalizedToolObj(obj_json: []const u8, dst: *FixedWriter) TemplateError!void {
    var r = JsonScan{ .src = obj_json };
    try r.expectByte('{');
    var name_raw: []const u8 = "\"\"";
    var desc_raw: []const u8 = "\"\"";
    var params_raw: []const u8 = "{}";
    var type_raw: []const u8 = "\"function\"";
    if (r.peek() != '}') {
        while (true) {
            var kbuf: [64]u8 = undefined;
            const mk = try r.readKeyOrSkip(&kbuf);
            try r.expectByte(':');
            if (mk) |key| {
                if (std.mem.eql(u8, key, "type")) {
                    type_raw = try r.captureValueSlice(2);
                } else if (std.mem.eql(u8, key, "function") and r.peek() == '{') {
                    r.pos += 1;
                    if (r.peek() != '}') {
                        while (true) {
                            var fk_buf: [64]u8 = undefined;
                            const mfk = try r.readKeyOrSkip(&fk_buf);
                            try r.expectByte(':');
                            if (mfk) |fkey| {
                                if (std.mem.eql(u8, fkey, "name")) {
                                    name_raw = try r.captureValueSlice(3);
                                } else if (std.mem.eql(u8, fkey, "description")) {
                                    desc_raw = try r.captureValueSlice(3);
                                } else if (std.mem.eql(u8, fkey, "parameters")) {
                                    params_raw = try r.captureValueSlice(3);
                                } else {
                                    try r.skipValue(3);
                                }
                            } else {
                                try r.skipValue(3);
                            }
                            const fsep = r.peek() orelse return error.InvalidJson;
                            if (fsep == ',') {
                                r.pos += 1;
                            } else if (fsep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                } else if (std.mem.eql(u8, key, "name")) {
                    name_raw = try r.captureValueSlice(2);
                } else if (std.mem.eql(u8, key, "description")) {
                    desc_raw = try r.captureValueSlice(2);
                } else if (std.mem.eql(u8, key, "parameters") or std.mem.eql(u8, key, "input_schema")) {
                    params_raw = try r.captureValueSlice(2);
                } else {
                    try r.skipValue(2);
                }
            } else {
                try r.skipValue(2);
            }
            const sep = r.peek() orelse return error.InvalidJson;
            if (sep == ',') {
                r.pos += 1;
            } else if (sep == '}') {
                r.pos += 1;
                break;
            } else return error.InvalidJson;
        }
    }
    try dst.writeAll("{\"description\":");
    var dr = JsonScan{ .src = desc_raw };
    try writeCanonicalJsonValue(&dr, dst, 0, true, false);
    try dst.writeAll(",\"name\":");
    var nr = JsonScan{ .src = name_raw };
    try writeCanonicalJsonValue(&nr, dst, 0, true, false);
    try dst.writeAll(",\"parameters\":");
    var pr = JsonScan{ .src = params_raw };
    try writeCanonicalJsonValue(&pr, dst, 0, true, false);
    try dst.writeAll(",\"type\":");
    var tr = JsonScan{ .src = type_raw };
    try writeCanonicalJsonValue(&tr, dst, 0, true, false);
    try dst.writeByte('}');
}

fn emitHy4NormalizedToolObj(obj_json: []const u8, dst: *FixedWriter) TemplateError!void {
    var r = JsonScan{ .src = obj_json };
    try r.expectByte('{');
    var name_raw: []const u8 = "\"\"";
    var desc_raw: []const u8 = "\"\"";
    var params_raw: []const u8 = "{}";
    if (r.peek() != '}') {
        while (true) {
            var kbuf: [64]u8 = undefined;
            const mk = try r.readKeyOrSkip(&kbuf);
            try r.expectByte(':');
            if (mk) |key| {
                if (std.mem.eql(u8, key, "function") and r.peek() == '{') {
                    r.pos += 1;
                    if (r.peek() != '}') {
                        while (true) {
                            var fk_buf: [64]u8 = undefined;
                            const mfk = try r.readKeyOrSkip(&fk_buf);
                            try r.expectByte(':');
                            if (mfk) |fkey| {
                                if (std.mem.eql(u8, fkey, "name")) {
                                    name_raw = try r.captureValueSlice(3);
                                } else if (std.mem.eql(u8, fkey, "description")) {
                                    desc_raw = try r.captureValueSlice(3);
                                } else if (std.mem.eql(u8, fkey, "parameters")) {
                                    params_raw = try r.captureValueSlice(3);
                                } else {
                                    try r.skipValue(3);
                                }
                            } else {
                                try r.skipValue(3);
                            }
                            const fsep = r.peek() orelse return error.InvalidJson;
                            if (fsep == ',') {
                                r.pos += 1;
                            } else if (fsep == '}') {
                                r.pos += 1;
                                break;
                            } else return error.InvalidJson;
                        }
                    } else r.pos += 1;
                } else if (std.mem.eql(u8, key, "name")) {
                    name_raw = try r.captureValueSlice(2);
                } else if (std.mem.eql(u8, key, "description")) {
                    desc_raw = try r.captureValueSlice(2);
                } else if (std.mem.eql(u8, key, "parameters") or std.mem.eql(u8, key, "input_schema")) {
                    params_raw = try r.captureValueSlice(2);
                } else {
                    try r.skipValue(2);
                }
            } else {
                try r.skipValue(2);
            }
            const sep = r.peek() orelse return error.InvalidJson;
            if (sep == ',') {
                r.pos += 1;
            } else if (sep == '}') {
                r.pos += 1;
                break;
            } else return error.InvalidJson;
        }
    }
    try dst.writeAll("{\"name\": ");
    var nr = JsonScan{ .src = name_raw };
    try writeCanonicalJsonValue(&nr, dst, 0, false, true);
    try dst.writeAll(", \"description\": ");
    var dr = JsonScan{ .src = desc_raw };
    try writeCanonicalJsonValue(&dr, dst, 0, false, true);
    try dst.writeAll(", \"parameters\": ");
    var pr = JsonScan{ .src = params_raw };
    try writeCanonicalJsonValue(&pr, dst, 0, false, true);
    try dst.writeByte('}');
}

fn emitHy4ToolArgs(args_json: []const u8, dst: *FixedWriter) TemplateError!void {
    var r = JsonScan{ .src = args_json };
    if (r.peek() == '"') {
        var unesc_buf: [2048]u8 = undefined;
        var uw = FixedWriter.init(&unesc_buf);
        try r.readStringInto(&uw);
        try emitHy4ToolArgs(uw.written(), dst);
        return;
    }
    try r.expectByte('{');
    if (r.peek() == '}') {
        r.pos += 1;
        return;
    }
    while (true) {
        try dst.writeAll("<arg_key:opensource>");
        try r.readStringInto(dst);
        try dst.writeAll("</arg_key:opensource>");
        try r.expectByte(':');
        try dst.writeAll("<arg_value:opensource>");
        if (r.peek() == '"') {
            try r.readStringInto(dst);
        } else {
            try writeCanonicalJsonValue(&r, dst, 1, false, true);
        }
        try dst.writeAll("</arg_value:opensource>");
        const sep = r.peek() orelse return error.InvalidJson;
        if (sep == ',') {
            r.pos += 1;
        } else if (sep == '}') {
            r.pos += 1;
            return;
        } else return error.InvalidJson;
    }
}

// ---------------------------------------------------------------------------
// Unified Zero-Alloc Streaming `ThinkSplitter`
// (`<think>..</think>`, `<think:opensource>..</think:opensource>`,
//  and Inkling 2 `<|content_thinking|>..<|end_message|>`)
// ---------------------------------------------------------------------------

pub const StreamChannel = enum(u8) {
    content = 0,
    reasoning = 1,
};

pub const ThinkTagKind = enum(u8) {
    auto = 0,
    standard = 1,
    hy4 = 2,
    inkling2 = 3,
};

pub const ThinkSplitter = struct {
    const STANDARD_OPEN = "<think>";
    const STANDARD_CLOSE = "</think>";
    const HY4_OPEN = "<think:opensource>";
    const HY4_CLOSE = "</think:opensource>";
    const INKLING_OPEN = "<|content_thinking|>";
    const INKLING_CLOSE = "<|end_message|>";
    const MAX_TAG_LEN: usize = INKLING_OPEN.len; // 20 bytes

    in_think: bool = false,
    mode: ThinkTagKind = .auto,
    active_kind: ThinkTagKind = .standard,
    pending: [MAX_TAG_LEN]u8 = undefined,
    pending_len: u8 = 0,

    pub fn reset(self: *ThinkSplitter, start_in_think: bool) void {
        self.resetWithMode(.auto, start_in_think);
    }

    pub fn resetForDialect(self: *ThinkSplitter, d: Dialect, start_in_think: bool) void {
        const m: ThinkTagKind = switch (d) {
            .hy4 => .hy4,
            .inkling2 => .inkling2,
            .chatml, .deepseek, .custom => .standard,
        };
        self.resetWithMode(m, start_in_think);
    }

    pub fn resetWithMode(self: *ThinkSplitter, mode: ThinkTagKind, start_in_think: bool) void {
        self.mode = mode;
        self.in_think = start_in_think;
        self.active_kind = if (mode == .auto) .standard else mode;
        self.pending_len = 0;
    }

    inline fn activeCloseTag(self: *const ThinkSplitter) []const u8 {
        return switch (self.active_kind) {
            .hy4 => HY4_CLOSE,
            .inkling2 => INKLING_CLOSE,
            .standard, .auto => STANDARD_CLOSE,
        };
    }

    fn matchOpenTagPrefix(self: *const ThinkSplitter, cand: []const u8) enum { none, prefix, exact_standard, exact_hy4, exact_inkling } {
        const allow_std = (self.mode == .auto or self.mode == .standard);
        const allow_hy4 = (self.mode == .auto or self.mode == .hy4);
        const allow_ink = (self.mode == .auto or self.mode == .inkling2);

        if (allow_std and std.mem.eql(u8, cand, STANDARD_OPEN)) return .exact_standard;
        if (allow_hy4 and std.mem.eql(u8, cand, HY4_OPEN)) return .exact_hy4;
        if (allow_ink and std.mem.eql(u8, cand, INKLING_OPEN)) return .exact_inkling;

        if ((allow_std and cand.len < STANDARD_OPEN.len and std.mem.startsWith(u8, STANDARD_OPEN, cand)) or
            (allow_hy4 and cand.len < HY4_OPEN.len and std.mem.startsWith(u8, HY4_OPEN, cand)) or
            (allow_ink and cand.len < INKLING_OPEN.len and std.mem.startsWith(u8, INKLING_OPEN, cand)))
        {
            return .prefix;
        }
        return .none;
    }

    pub fn feed(
        self: *ThinkSplitter,
        chunk: []const u8,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        var i: usize = 0;
        while (i < chunk.len) {
            const channel: StreamChannel = if (self.in_think) .reasoning else .content;
            if (self.pending_len == 0) {
                if (std.mem.indexOfScalarPos(u8, chunk, i, '<')) |lt_pos| {
                    if (lt_pos > i) {
                        try emit_fn(ctx, channel, chunk[i..lt_pos]);
                    }
                    self.pending[0] = '<';
                    self.pending_len = 1;
                    i = lt_pos + 1;
                } else {
                    try emit_fn(ctx, channel, chunk[i..]);
                    return;
                }
            } else {
                const b = chunk[i];
                if (self.in_think) {
                    const ctag = self.activeCloseTag();
                    if (b == ctag[self.pending_len]) {
                        self.pending[self.pending_len] = b;
                        self.pending_len += 1;
                        i += 1;
                        if (self.pending_len == ctag.len) {
                            self.pending_len = 0;
                            self.in_think = false;
                        }
                    } else {
                        const flushed = self.pending[0..self.pending_len];
                        self.pending_len = 0;
                        try emit_fn(ctx, channel, flushed);
                    }
                } else {
                    if (self.pending_len < MAX_TAG_LEN) {
                        self.pending[self.pending_len] = b;
                        const cand = self.pending[0 .. self.pending_len + 1];
                        switch (self.matchOpenTagPrefix(cand)) {
                            .prefix => {
                                self.pending_len += 1;
                                i += 1;
                            },
                            .exact_standard => {
                                self.pending_len = 0;
                                self.in_think = true;
                                self.active_kind = .standard;
                                i += 1;
                            },
                            .exact_hy4 => {
                                self.pending_len = 0;
                                self.in_think = true;
                                self.active_kind = .hy4;
                                i += 1;
                            },
                            .exact_inkling => {
                                self.pending_len = 0;
                                self.in_think = true;
                                self.active_kind = .inkling2;
                                i += 1;
                            },
                            .none => {
                                const flushed = self.pending[0..self.pending_len];
                                self.pending_len = 0;
                                try emit_fn(ctx, channel, flushed);
                            },
                        }
                    } else {
                        const flushed = self.pending[0..self.pending_len];
                        self.pending_len = 0;
                        try emit_fn(ctx, channel, flushed);
                    }
                }
            }
        }
    }

    pub fn flush(
        self: *ThinkSplitter,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        if (self.pending_len == 0) return;
        const channel: StreamChannel = if (self.in_think) .reasoning else .content;
        const flushed = self.pending[0..self.pending_len];
        self.pending_len = 0;
        try emit_fn(ctx, channel, flushed);
    }
};

// ---------------------------------------------------------------------------
// Unit Tests (`W24`)
// ---------------------------------------------------------------------------

test "ChatTemplate renders Inkling 2 exact prompt, effort_map, tool_declare, tool_calls, and image tags" {
    const tpl = ChatTemplate.preset(.inkling2);
    var buf: [4096]u8 = undefined;

    const tools = [_]ToolDeclView{.{
        .name = "get_weather",
        .description = "Get weather",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}",
    }};
    const tool_calls = [_]ToolCallView{.{
        .id = "call_1",
        .name = "get_weather",
        .arguments_json = "{\"city\":\"Tokyo\"}",
    }};
    const msgs = [_]MessageView{
        .{ .role = .system, .content = "Be brief." },
        .{ .role = .user, .content = "Weather in Tokyo?" },
        .{
            .role = .assistant,
            .reasoning_content = "Need tool.",
            .tool_calls = &tool_calls,
        },
        .{ .role = .tool, .tool_call_id = "call_1", .content = "Sunny 22C" },
        .{ .role = .user, .content = "Thanks!" },
    };

    const res = try tpl.render(&buf, &msgs, .{
        .reasoning_effort = .medium,
        .tools = &tools,
    });

    const expected =
        "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Get weather\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"type\":\"object\"},\"type\":\"function\"}]<|end_message|>" ++
        "<|message_system|><|content_text|>Be brief.<|end_message|>" ++
        "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>" ++
        "<|message_user|><|content_text|>Weather in Tokyo?<|end_message|>" ++
        "<|message_model|><|content_thinking|>Need tool.<|end_message|>" ++
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Tokyo\"}}<|end_message|><|content_model_end_sampling|>" ++
        "<|message_tool|>get_weather<|content_text|>Sunny 22C<|end_message|>" ++
        "<|message_user|><|content_text|>Thanks!<|end_message|>";

    try std.testing.expectEqualStrings(expected, res.prompt);
    try std.testing.expect(!res.starts_in_think);
    try std.testing.expect(res.last_user_msg_byte_offset != null);

    // System-only prompt emits Thinking effort level at end; effort `none` emits `0`; float 0.5 emits `0.5`.
    const sys_only = [_]MessageView{.{ .role = .system, .content = "Policy" }};
    const r0 = try tpl.render(&buf, &sys_only, .{ .reasoning_effort = .none });
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Policy<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|>",
        r0.prompt,
    );
    const r05 = try tpl.render(&buf, &sys_only, .{ .reasoning_effort = .{ .numeric = 0.5 } });
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Policy<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.5<|end_message|>",
        r05.prompt,
    );

    // Image part emits <|content_image|><|unused_200054|>; system image rejected.
    const img_parts = [_]ContentPart{ .{ .text = "Look:" }, .image };
    const img_msgs = [_]MessageView{.{ .role = .user, .parts = &img_parts }};
    const ri = try tpl.render(&buf, &img_msgs, .{});
    try std.testing.expect(std.mem.indexOf(u8, ri.prompt, "<|message_user|><|content_image|><|unused_200054|><|end_message|>") != null);
}

test "ChatTemplate renders Hy4 exact prompt, reasoning_effort normalization, preserved_thinking, and tool clustering" {
    const tpl = ChatTemplate.preset(.hy4);
    var buf: [8192]u8 = undefined;

    const tools = [_]ToolDeclView{.{
        .name = "get_weather",
        .description = "Get weather",
        .parameters_json = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}",
    }};
    const tool_calls = [_]ToolCallView{.{
        .id = "call_1",
        .name = "get_weather",
        .arguments_json = "{\"city\":\"Tokyo\"}",
    }};
    const msgs = [_]MessageView{
        .{ .role = .system, .content = "Be brief." },
        .{ .role = .user, .content = "Weather in Tokyo?" },
        .{
            .role = .assistant,
            .reasoning_content = "Need tool.",
            .tool_calls = &tool_calls,
        },
        .{ .role = .tool, .tool_call_id = "call_1", .content = "Sunny 22C" },
        .{ .role = .user, .content = "Thanks!" },
    };

    // Positive tier (`low`) normalizes to `high` (never emitting `'low'`), tools default `preserved_thinking = true`.
    const r_high = try tpl.render(&buf, &msgs, .{
        .reasoning_effort = .low,
        .tools = &tools,
        .add_generation_prompt = true,
    });
    try std.testing.expect(r_high.starts_in_think);
    try std.testing.expect(std.mem.indexOf(u8, r_high.prompt, "<\xef\xbd\x9creasoning_mode:opensource\xef\xbd\x9c>reasoning_effort:high<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>") != null);
    try std.testing.expect(std.mem.indexOf(u8, r_high.prompt, "reasoning_effort:low") == null);
    try std.testing.expect(std.mem.indexOf(u8, r_high.prompt, "<think:opensource>Need tool.</think:opensource>") != null);
    try std.testing.expect(std.mem.indexOf(u8, r_high.prompt, "<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tokyo</arg_value:opensource></tool_call:opensource></tool_calls:opensource>") != null);
    try std.testing.expect(std.mem.endsWith(u8, r_high.prompt, "<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>assistant<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c><think:opensource>"));

    // `no_think` / `none` / `0.0` normalizes to `no_think`, strips history thinking, and emits `<think:opensource></think:opensource>`.
    const r_no = try tpl.render(&buf, &msgs, .{
        .reasoning_effort = .{ .numeric = 0.0 },
        .tools = &tools,
        .add_generation_prompt = true,
    });
    try std.testing.expect(!r_no.starts_in_think);
    try std.testing.expect(std.mem.indexOf(u8, r_no.prompt, "reasoning_effort:no_think") != null);
    try std.testing.expect(std.mem.indexOf(u8, r_no.prompt, "Need tool.") == null);
    try std.testing.expect(std.mem.endsWith(u8, r_no.prompt, "<think:opensource></think:opensource>"));

    // Without leading system message, synthesizes one!
    const no_sys = [_]MessageView{.{ .role = .user, .content = "Hi" }};
    const r_nosys = try tpl.render(&buf, &no_sys, .{ .reasoning_effort = .minimal });
    try std.testing.expect(std.mem.startsWith(
        u8,
        r_nosys.prompt,
        "<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>system<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c><\xef\xbd\x9creasoning_mode:opensource\xef\xbd\x9c>reasoning_effort:high<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>",
    ));
}

test "ChatTemplate declarative JSON parsing, custom delimiters, and ThinkSplitter multi-dialect tags" {
    const spec_json =
        \\{
        \\  "schema_version": 1,
        \\  "dialect": "custom",
        \\  "eos_token": "<|stop|>",
        \\  "role_tokens": {
        \\    "system_prefix": "<<SYS>>",
        \\    "system_suffix": "<</SYS>>\n",
        \\    "user_prefix": "<<USR>>",
        \\    "user_suffix": "<</USR>>\n",
        \\    "assistant_prefix": "<<AST>>",
        \\    "assistant_suffix": "<|stop|>\n"
        \\  },
        \\  "thinking": {
        \\    "supported": true,
        \\    "open_tag": "<thought>",
        \\    "close_tag": "</thought>",
        \\    "prefill_no_think": "<thought></thought>\n"
        \\  }
        \\}
    ;
    const tpl = try ChatTemplate.fromSpecJson(spec_json);
    var buf: [512]u8 = undefined;
    const msgs = [_]MessageView{
        .{ .role = .system, .content = "Rules" },
        .{ .role = .user, .content = "Hi" },
    };
    const res = try tpl.render(&buf, &msgs, .{ .enable_thinking = false });
    try std.testing.expectEqualStrings("<<SYS>>Rules<</SYS>>\n<<USR>>Hi<</USR>>\n<<AST>><thought></thought>\n", res.prompt);

    // ThinkSplitter handles fragmented <think>, <think:opensource>, and <|content_thinking|>..<|end_message|>.
    const Cap = struct {
        reason: [128]u8 = undefined,
        r_len: usize = 0,
        content: [128]u8 = undefined,
        c_len: usize = 0,
        fn emit(self: *@This(), ch: StreamChannel, s: []const u8) error{BufferTooSmall}!void {
            switch (ch) {
                .reasoning => {
                    if (self.r_len + s.len > self.reason.len) return error.BufferTooSmall;
                    @memcpy(self.reason[self.r_len .. self.r_len + s.len], s);
                    self.r_len += s.len;
                },
                .content => {
                    if (self.c_len + s.len > self.content.len) return error.BufferTooSmall;
                    @memcpy(self.content[self.c_len .. self.c_len + s.len], s);
                    self.c_len += s.len;
                },
            }
        }
    };

    var sp: ThinkSplitter = .{};
    // 1. Hy4 fragmented tags
    sp.resetForDialect(.hy4, true);
    var c1: Cap = .{};
    try sp.feed("step1</think", &c1, Cap.emit);
    try sp.feed(":open", &c1, Cap.emit);
    try sp.feed("source>Answer", &c1, Cap.emit);
    try sp.flush(&c1, Cap.emit);
    try std.testing.expectEqualStrings("step1", c1.reason[0..c1.r_len]);
    try std.testing.expectEqualStrings("Answer", c1.content[0..c1.c_len]);

    // 2. Inkling 2 fragmented tags (ignoring stray `</think>` inside `<|content_thinking|>`)
    sp.reset(false);
    var c2: Cap = .{};
    try sp.feed("pre<|content_", &c2, Cap.emit);
    try sp.feed("thinking|>plan</think>2<|end_", &c2, Cap.emit);
    try sp.feed("message|>post", &c2, Cap.emit);
    try sp.flush(&c2, Cap.emit);
    try std.testing.expectEqualStrings("plan</think>2", c2.reason[0..c2.r_len]);
    try std.testing.expectEqualStrings("prepost", c2.content[0..c2.c_len]);
}
