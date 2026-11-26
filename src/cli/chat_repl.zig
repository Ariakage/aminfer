//! `aminfer chat` (`C20`, ADR-0022): plain scrolling terminal line-REPL and
//! non-interactive `stdin` pipe runner.
//!
//! Design invariants (`ADR-0022` Section 5):
//! - Explicitly NOT an alternate-screen full-screen TUI (`\x1b[?1049h` is never
//!   emitted; full-screen terminal rendering belongs exclusively to `aminfer monitor`).
//! - Inline streaming token output with zero per-token heap allocation.
//! - Optional dimmed `<think>...</think>` reasoning output (`ThinkSplitter` safe
//!   across arbitrary token boundaries) and historical `<think>` stripping on
//!   subsequent multi-turn ChatML prompts.
//! - Compact 1-line post-turn summary (`tok/s`, TTFT, `PrefixCache` hit tokens).
//! - Interactive slash commands `/clear`, `/retry`, `/edit` reusing `C15` ChatML
//!   message-boundary `PrefixCache` checkpoints, plus `/image` and `/video`
//!   attachment queuing for `G14` zero-copy vision/video ingress.

const std = @import("std");
const telemetry = @import("../platform/telemetry.zig");
const vmstat = @import("../platform/vmstat.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const prefix_cache = @import("../mem/prefix_cache.zig");
const chat_template = @import("../model/chat_template.zig");

pub const MAX_TURNS: usize = 128;
pub const MAX_ATTACHMENTS_PER_TURN: usize = 8;
pub const MAX_TOTAL_ATTACHMENTS: usize = 256;
pub const DEFAULT_TEXT_POOL_BYTES: usize = 512 * 1024;
pub const DEFAULT_PROMPT_BUF_BYTES: usize = 512 * 1024;
pub const DEFAULT_LINE_BUF_BYTES: usize = 64 * 1024;

pub const Error = std.mem.Allocator.Error || vmstat.Error || prefix_cache.Error || error{
    InvalidCliArgument,
    MissingCliValue,
    HistoryFull,
    TextPoolFull,
    PromptTooLarge,
    AttachmentLimitExceeded,
    EmptyAttachmentPath,
    NoTurnToRetry,
    NoTurnToEdit,
    EmptyEditMessage,
    UnknownSlashCommand,
    ReadFailed,
    WriteFailed,
    BackendFailed,
};

pub const HELP_TEXT: []const u8 =
    \\usage: aminfer chat [-CHhqNSV] -m path.amf [-c config.json] [-D cache_dir]
    \\                    [-k tokenizer.json] [-L name=path.amlora] [-n tokens]
    \\                    [-s system_prompt] [-t temp] [-W wired_mib]
    \\                    [--image path] [--video path] [--seed u64]
    \\
    \\Options:
    \\  -m, --model <path>              Model .amf path or directory
    \\  -c, --config <path>             Explicit config.json override (W18)
    \\  -k, --tokenizer <path>          Explicit tokenizer.json override (W18)
    \\  -L, --lora <name>=<path.amlora> Load and activate first runtime LoRA adapter (W19/W30)
    \\  -D, --disk-prefix-cache <dir>   Enable R10 F_NOCACHE persistent prefix cache
    \\  -W, --wired-cap-mib <n>         Override wired memory budget ceiling in MiB
    \\  -s, --system <prompt>           System prompt (default: "You are a helpful assistant.")
    \\  -n, --max-tokens <n>            Max generated tokens per turn (default: 1024)
    \\  -t, --temperature <f>           Sampling temperature (default: 0.6)
    \\      --top-k <n>                 Top-k sampling cutoff, 0 = disabled (default: 40)
    \\      --top-p <f>                 Nucleus sampling cutoff in (0, 1] (default: 0.95)
    \\      --min-p <f>                 Min-p relative cutoff in [0, 1) (default: 0.05)
    \\      --repetition-penalty <f>    Repetition penalty > 0 (default: 1.05 when T>0, 1.0 when T=0)
    \\      --presence-penalty <f>      OpenAI additive presence penalty (default: 0.0)
    \\      --frequency-penalty <f>     OpenAI additive frequency penalty (default: 0.0)
    \\      --seed <u64>                Deterministic RNG seed
    \\  -N, --no-think                  Request non-thinking turn prefill
    \\      --reasoning-effort <level>  Reasoning effort (none|minimal|low|medium|high|xhigh|max|0..0.99)
    \\  -H, --hide-think                Suppress <think>...</think> blocks from output
    \\  -C, --no-color                  Disable ANSI dimming/colors
    \\  -q, --no-stats                  Suppress 1-line post-turn throughput summary
    \\  -S, --strict-live-ram           Enforce strict R9 live non-anonymous RAM admission
    \\  -V, --no-verify-crc             Skip .amf CRC-32 payload verification
    \\      --image <path>              Attach an image to the initial turn (G14)
    \\      --video <path>              Attach a video to the initial turn (G14)
    \\  -h, --help                      Show this help message
    \\
    \\Interactive slash commands:
    \\  /clear                   Clear conversation history & reset PrefixCache state
    \\  /retry                   Rewind last turn to C15 boundary checkpoint & regenerate
    \\  /edit <new message>      Replace last user message from C15 boundary & regenerate
    \\  /image <path>            Attach image (G14 zero-copy IOSurface ingress) to next turn
    \\  /video <path>            Attach video (G14 VideoToolbox ingress) to next turn
    \\  /lora [name|off]         Show active/loaded LoRA adapters, switch adapter, or revert to base
    \\  /help                    Print interactive slash-command help
    \\  /exit, /quit             Exit REPL
    \\
;

pub const ThinkDisplayMode = enum(u8) {
    dimmed = 0,
    hidden = 1,
    plain = 2,
};

pub const AttachmentKind = enum(u8) {
    image = 0,
    video = 1,

    pub fn label(self: AttachmentKind) []const u8 {
        return switch (self) {
            .image => "image",
            .video => "video",
        };
    }

    pub fn chatMlTag(self: AttachmentKind) []const u8 {
        return switch (self) {
            .image => "<|vision_start|><|image_pad|><|vision_end|>",
            .video => "<|vision_start|><|video_pad|><|vision_end|>",
        };
    }
};

pub const Attachment = struct {
    kind: AttachmentKind,
    path_off: u32,
    path_len: u32,
    /// Optional caller-owned zero-copy `G14` `ami_vision_frame*` / `ami_vt_clip*` handle.
    native_handle: ?*anyopaque = null,
};

pub const AttachmentView = struct {
    kind: AttachmentKind,
    path: []const u8,
    native_handle: ?*anyopaque = null,
};

pub const ChatConfig = struct {
    model_path: ?[]const u8 = null,
    system_prompt: []const u8 = "You are a helpful assistant.",
    max_tokens: u32 = 1024,
    temperature: f32 = 0.6,
    top_k: u32 = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    repetition_penalty: f32 = 1.05,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    seed: ?u64 = null,
    enable_thinking: bool = true,
    reasoning_effort: ?chat_template.ReasoningEffort = null,
    chat_template: chat_template.ChatTemplate = chat_template.ChatTemplate.preset(.chatml),
    think_display: ThinkDisplayMode = .dimmed,
    use_color: bool = true,
    show_stats: bool = true,
    show_help: bool = false,
    initial_images: [MAX_ATTACHMENTS_PER_TURN][]const u8 = @splat(""),
    initial_image_count: u8 = 0,
    initial_videos: [MAX_ATTACHMENTS_PER_TURN][]const u8 = @splat(""),
    initial_video_count: u8 = 0,

    pub fn parseArgs(args: []const []const u8) Error!ChatConfig {
        var cfg: ChatConfig = .{};
        var explicit_no_think = false;
        var explicit_rep_penalty = false;
        var explicit_effort: ?chat_template.ReasoningEffort = null;
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            const arg = args[i];
            if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
                cfg.show_help = true;
            } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.model_path = args[i];
            } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--system")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.system_prompt = args[i];
            } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.max_tokens = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidCliArgument;
                if (cfg.max_tokens == 0) return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temperature")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.temperature = std.fmt.parseFloat(f32, args[i]) catch return error.InvalidCliArgument;
                if (cfg.temperature < 0.0 or !std.math.isFinite(cfg.temperature)) return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--top-k")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.top_k = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidCliArgument;
            } else if (std.mem.startsWith(u8, arg, "--top-k=")) {
                cfg.top_k = std.fmt.parseInt(u32, arg["--top-k=".len..], 10) catch return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--top-p")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.top_p = std.fmt.parseFloat(f32, args[i]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.top_p) or !(cfg.top_p > 0.0) or cfg.top_p > 1.0) return error.InvalidCliArgument;
            } else if (std.mem.startsWith(u8, arg, "--top-p=")) {
                cfg.top_p = std.fmt.parseFloat(f32, arg["--top-p=".len..]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.top_p) or !(cfg.top_p > 0.0) or cfg.top_p > 1.0) return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--min-p")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.min_p = std.fmt.parseFloat(f32, args[i]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.min_p) or cfg.min_p < 0.0 or cfg.min_p >= 1.0) return error.InvalidCliArgument;
            } else if (std.mem.startsWith(u8, arg, "--min-p=")) {
                cfg.min_p = std.fmt.parseFloat(f32, arg["--min-p=".len..]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.min_p) or cfg.min_p < 0.0 or cfg.min_p >= 1.0) return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--repetition-penalty")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.repetition_penalty = std.fmt.parseFloat(f32, args[i]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.repetition_penalty) or !(cfg.repetition_penalty > 0.0)) return error.InvalidCliArgument;
                explicit_rep_penalty = true;
            } else if (std.mem.startsWith(u8, arg, "--repetition-penalty=")) {
                cfg.repetition_penalty = std.fmt.parseFloat(f32, arg["--repetition-penalty=".len..]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.repetition_penalty) or !(cfg.repetition_penalty > 0.0)) return error.InvalidCliArgument;
                explicit_rep_penalty = true;
            } else if (std.mem.eql(u8, arg, "--presence-penalty")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.presence_penalty = std.fmt.parseFloat(f32, args[i]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.presence_penalty)) return error.InvalidCliArgument;
            } else if (std.mem.startsWith(u8, arg, "--presence-penalty=")) {
                cfg.presence_penalty = std.fmt.parseFloat(f32, arg["--presence-penalty=".len..]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.presence_penalty)) return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--frequency-penalty")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.frequency_penalty = std.fmt.parseFloat(f32, args[i]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.frequency_penalty)) return error.InvalidCliArgument;
            } else if (std.mem.startsWith(u8, arg, "--frequency-penalty=")) {
                cfg.frequency_penalty = std.fmt.parseFloat(f32, arg["--frequency-penalty=".len..]) catch return error.InvalidCliArgument;
                if (!std.math.isFinite(cfg.frequency_penalty)) return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--seed")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                cfg.seed = std.fmt.parseInt(u64, args[i], 10) catch return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--no-think") or std.mem.eql(u8, arg, "-N")) {
                explicit_no_think = true;
                cfg.enable_thinking = false;
            } else if (std.mem.eql(u8, arg, "--reasoning-effort")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                explicit_effort = chat_template.ReasoningEffort.parseCliArg(args[i]) catch return error.InvalidCliArgument;
            } else if (std.mem.startsWith(u8, arg, "--reasoning-effort=")) {
                const val = arg["--reasoning-effort=".len..];
                explicit_effort = chat_template.ReasoningEffort.parseCliArg(val) catch return error.InvalidCliArgument;
            } else if (std.mem.eql(u8, arg, "--hide-think") or std.mem.eql(u8, arg, "-H")) {
                cfg.think_display = .hidden;
            } else if (std.mem.eql(u8, arg, "--no-color") or std.mem.eql(u8, arg, "-C")) {
                cfg.use_color = false;
                if (cfg.think_display == .dimmed) cfg.think_display = .plain;
            } else if (std.mem.eql(u8, arg, "--no-stats") or std.mem.eql(u8, arg, "-q")) {
                cfg.show_stats = false;
            } else if (std.mem.eql(u8, arg, "--image")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                if (cfg.initial_image_count >= MAX_ATTACHMENTS_PER_TURN) return error.AttachmentLimitExceeded;
                cfg.initial_images[cfg.initial_image_count] = args[i];
                cfg.initial_image_count += 1;
            } else if (std.mem.eql(u8, arg, "--video")) {
                i += 1;
                if (i >= args.len) return error.MissingCliValue;
                if (cfg.initial_video_count >= MAX_ATTACHMENTS_PER_TURN) return error.AttachmentLimitExceeded;
                cfg.initial_videos[cfg.initial_video_count] = args[i];
                cfg.initial_video_count += 1;
            } else if (arg.len >= 2 and arg[0] == '-' and arg[1] != '-') {
                for (arg[1..]) |ch| {
                    switch (ch) {
                        'C' => {
                            cfg.use_color = false;
                            if (cfg.think_display == .dimmed) cfg.think_display = .plain;
                        },
                        'H' => cfg.think_display = .hidden,
                        'h' => cfg.show_help = true,
                        'N' => {
                            explicit_no_think = true;
                            cfg.enable_thinking = false;
                        },
                        'q' => cfg.show_stats = false,
                        else => return error.InvalidCliArgument,
                    }
                }
            } else {
                return error.InvalidCliArgument;
            }
        }
        if (cfg.temperature == 0.0 and !explicit_rep_penalty) {
            cfg.repetition_penalty = 1.0;
        }
        if (explicit_effort) |eff| {
            if (explicit_no_think and eff.isThinkingEnabled()) return error.InvalidCliArgument;
            cfg.reasoning_effort = eff;
            cfg.enable_thinking = eff.isThinkingEnabled();
        }
        return cfg;
    }
};

// ---------------------------------------------------------------------------
// Streaming Think-Block Splitter (shared with `chat_template.zig`, zero heap alloc)
// ---------------------------------------------------------------------------

pub const StreamChannel = chat_template.StreamChannel;
pub const ThinkSplitter = chat_template.ThinkSplitter;

// ---------------------------------------------------------------------------
// Writer Sink & Turn-Runner Abstraction (ready for `main.zig` wiring)
// ---------------------------------------------------------------------------

pub const ByteSink = struct {
    ctx: *anyopaque,
    writeFn: *const fn (ctx: *anyopaque, bytes: []const u8) error{WriteFailed}!void,

    pub fn writeAll(self: ByteSink, bytes: []const u8) error{WriteFailed}!void {
        if (bytes.len == 0) return;
        return self.writeFn(self.ctx, bytes);
    }

    pub fn print(self: ByteSink, comptime fmt: []const u8, args: anytype) error{WriteFailed}!void {
        var stack_buf: [512]u8 = undefined;
        const s = std.fmt.bufPrint(&stack_buf, fmt, args) catch return error.WriteFailed;
        return self.writeAll(s);
    }

    pub fn fromFd(fd_ptr: *const std.c.fd_t) ByteSink {
        return .{
            .ctx = @constCast(fd_ptr),
            .writeFn = struct {
                fn writeFd(ctx: *anyopaque, bytes: []const u8) error{WriteFailed}!void {
                    const fd: *const std.c.fd_t = @ptrCast(@alignCast(ctx));
                    var off: usize = 0;
                    while (off < bytes.len) {
                        const rc = std.c.write(fd.*, bytes[off..].ptr, bytes.len - off);
                        if (rc < 0) {
                            if (std.c.errno(rc) == .INTR) continue;
                            return error.WriteFailed;
                        }
                        if (rc == 0) return error.WriteFailed;
                        off += @intCast(rc);
                    }
                }
            }.writeFd,
        };
    }
};

pub const TurnRequest = struct {
    turn_index: u32,
    /// Full Qwen3 ChatML formatted prompt in preallocated session buffer.
    chatml_prompt: []const u8,
    /// Raw user message text for the current turn.
    user_message: []const u8,
    /// Exact byte offset in `chatml_prompt` right before the final `<|im_start|>user\n`
    /// turn on multi-turn chats (`turn_index >= 1`, `C15` ADR-0020); `null` on turn 0.
    last_user_msg_byte_offset: ?usize,
    /// `C15` `PrefixCache` message-boundary checkpoint ID at the end of `turn_index - 1`
    /// (`prefix_cache.NONE_ID` on turn 0 or after `/clear`). May be either a full block
    /// or an intra-tile CoW tail (`evict_first = true`).
    parent_checkpoint_entry_id: u32,
    /// Last canonical full-tile `PrefixCache` block ID at the end of `turn_index - 1`
    /// (`prefix_cache.NONE_ID` when no full tile has completed yet). Valid parent for
    /// chaining `PrefixCache.insertFullBlock` / `insertBoundaryCheckpoint`.
    parent_full_block_id: u32 = prefix_cache.NONE_ID,
    /// Zero-copy `G14` image/video attachments bound to this turn.
    attachments: []const AttachmentView,
    max_tokens: u32,
    temperature: f32,
    top_k: u32 = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
    repetition_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    seed: ?u64,
    enable_thinking: bool,
    reasoning_effort: ?chat_template.ReasoningEffort = null,
};

pub const TurnStats = struct {
    prompt_tokens: u32 = 0,
    generated_tokens: u32 = 0,
    prefix_hit_tokens: u32 = 0,
    ttft_ns: u64 = 0,
    total_ns: u64 = 0,
    /// Optional `C15` `PrefixCache` boundary checkpoint ID after completing this turn.
    checkpoint_entry_id: u32 = prefix_cache.NONE_ID,
    /// Optional last canonical full-tile `PrefixCache` block ID after completing this turn.
    full_block_entry_id: u32 = prefix_cache.NONE_ID,

    pub fn tokensPerSecond(self: TurnStats) f64 {
        if (self.total_ns == 0 or self.generated_tokens == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.generated_tokens)) * @as(f64, @floatFromInt(std.time.ns_per_s))) /
            @as(f64, @floatFromInt(self.total_ns));
    }

    pub fn ttftMs(self: TurnStats) f64 {
        return @as(f64, @floatFromInt(self.ttft_ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    }
};

/// Zero-allocation streaming sink handed to `TurnRunner.runTurnFn`.
pub const TurnStream = struct {
    out_sink: ByteSink,
    think_display: ThinkDisplayMode,
    use_color: bool,
    splitter: ThinkSplitter = .{},
    dim_active: bool = false,
    had_reasoning: bool = false,
    had_content: bool = false,
    last_reasoning_byte: u8 = 0,

    start_ns: u64,
    first_token_ns: ?u64 = null,
    emitted_chunks: u32 = 0,

    assistant_content_buf: []u8,
    assistant_content_len: usize = 0,

    pub fn emitText(self: *TurnStream, chunk: []const u8) Error!void {
        if (chunk.len == 0) return;
        if (self.first_token_ns == null) {
            self.first_token_ns = try vmstat.nowNs();
        }
        self.emitted_chunks +|= 1;
        try self.splitter.feed(chunk, self, onClassifiedSpan);
    }

    fn finish(self: *TurnStream) Error!void {
        try self.splitter.flush(self, onClassifiedSpan);
        if (self.dim_active) {
            try self.out_sink.writeAll("\x1b[0m");
            self.dim_active = false;
        }
    }

    fn onClassifiedSpan(self: *TurnStream, channel: StreamChannel, span: []const u8) Error!void {
        if (span.len == 0) return;
        switch (channel) {
            .reasoning => {
                self.had_reasoning = true;
                switch (self.think_display) {
                    .hidden => return,
                    .dimmed => {
                        if (self.use_color and !self.dim_active) {
                            try self.out_sink.writeAll("\x1b[2m");
                            self.dim_active = true;
                        }
                        try self.out_sink.writeAll(span);
                        self.last_reasoning_byte = span[span.len - 1];
                    },
                    .plain => {
                        try self.out_sink.writeAll(span);
                        self.last_reasoning_byte = span[span.len - 1];
                    },
                }
            },
            .content => {
                if (self.dim_active) {
                    try self.out_sink.writeAll("\x1b[0m");
                    self.dim_active = false;
                }
                // Strip leading single newline right after `</think>\n` when reasoning was hidden.
                const clean_span = if (!self.had_content and self.had_reasoning and self.think_display == .hidden and span[0] == '\n')
                    span[1..]
                else
                    span;
                if (clean_span.len == 0) return;

                // Visual separation between visible reasoning and answer when neither
                // ended nor started with a newline (e.g. Inkling 2 thinking blocks).
                if (!self.had_content and self.had_reasoning and self.think_display != .hidden and
                    self.last_reasoning_byte != '\n' and clean_span[0] != '\n')
                {
                    try self.out_sink.writeAll("\n\n");
                }

                self.had_content = true;
                try self.out_sink.writeAll(clean_span);

                // Record non-thinking assistant content into history pool for clean multi-turn ChatML.
                if (self.assistant_content_len + clean_span.len > self.assistant_content_buf.len) {
                    return error.TextPoolFull;
                }
                @memcpy(
                    self.assistant_content_buf[self.assistant_content_len .. self.assistant_content_len + clean_span.len],
                    clean_span,
                );
                self.assistant_content_len += clean_span.len;
            },
        }
    }
};

pub const TurnRunner = struct {
    ctx: *anyopaque,
    runTurnFn: *const fn (
        ctx: *anyopaque,
        req: *const TurnRequest,
        stream: *TurnStream,
    ) Error!TurnStats,
    onClearFn: ?*const fn (ctx: *anyopaque) void = null,
    loadAttachmentFn: ?*const fn (
        ctx: *anyopaque,
        kind: AttachmentKind,
        path: []const u8,
    ) Error!?*anyopaque = null,
    onLoraCommandFn: ?*const fn (
        ctx: *anyopaque,
        arg: []const u8,
        info_sink: ByteSink,
    ) Error!void = null,
};

// ---------------------------------------------------------------------------
// Slash Command Parser
// ---------------------------------------------------------------------------

pub const SlashCommand = union(enum) {
    clear: void,
    retry: void,
    edit: []const u8,
    image: []const u8,
    video: []const u8,
    lora: []const u8,
    help: void,
    exit: void,
};

pub fn parseSlashCommand(line: []const u8) Error!?SlashCommand {
    const trimmed = std.mem.trim(u8, line, " \t\r\n");
    if (trimmed.len == 0 or trimmed[0] != '/') return null;

    const space_idx = std.mem.indexOfAny(u8, trimmed, " \t");
    const cmd = if (space_idx) |idx| trimmed[0..idx] else trimmed;
    const raw_arg = if (space_idx) |idx| std.mem.trim(u8, trimmed[idx + 1 ..], " \t") else "";

    if (std.mem.eql(u8, cmd, "/clear")) return .clear;
    if (std.mem.eql(u8, cmd, "/retry")) return .retry;
    if (std.mem.eql(u8, cmd, "/help")) return .help;
    if (std.mem.eql(u8, cmd, "/exit") or std.mem.eql(u8, cmd, "/quit")) return .exit;

    if (std.mem.eql(u8, cmd, "/edit")) {
        if (raw_arg.len == 0) return error.EmptyEditMessage;
        return .{ .edit = unquoteArg(raw_arg) };
    }
    if (std.mem.eql(u8, cmd, "/image")) {
        const path = unquoteArg(raw_arg);
        if (path.len == 0) return error.EmptyAttachmentPath;
        return .{ .image = path };
    }
    if (std.mem.eql(u8, cmd, "/video")) {
        const path = unquoteArg(raw_arg);
        if (path.len == 0) return error.EmptyAttachmentPath;
        return .{ .video = path };
    }
    if (std.mem.eql(u8, cmd, "/lora")) {
        return .{ .lora = unquoteArg(raw_arg) };
    }

    return error.UnknownSlashCommand;
}

fn unquoteArg(raw: []const u8) []const u8 {
    if (raw.len >= 2 and ((raw[0] == '"' and raw[raw.len - 1] == '"') or
        (raw[0] == '\'' and raw[raw.len - 1] == '\'')))
    {
        return raw[1 .. raw.len - 1];
    }
    return raw;
}

// ---------------------------------------------------------------------------
// Preallocated Multi-Turn Chat Session State Machine (`0` steady-state allocs)
// ---------------------------------------------------------------------------

pub const TurnRecord = struct {
    text_pool_mark_before: usize,
    attachment_mark_before: usize,
    user_off: u32,
    user_len: u32,
    assistant_off: u32,
    assistant_len: u32,
    attach_start: u16,
    attach_count: u8,
    /// `C15` `PrefixCache` checkpoint ID at the message boundary BEFORE this turn.
    parent_checkpoint_entry_id: u32,
    /// Last canonical full-tile `PrefixCache` block ID BEFORE this turn.
    parent_full_block_id: u32,
    /// `C15` `PrefixCache` checkpoint ID recorded AFTER completing this turn.
    post_turn_checkpoint_entry_id: u32,
    /// Last canonical full-tile `PrefixCache` block ID AFTER completing this turn.
    post_turn_full_block_id: u32,
    last_user_msg_byte_offset: ?usize,
};

pub const LineOutcome = enum {
    continued,
    exited,
};

pub const ChatSession = struct {
    cfg: ChatConfig,
    turns: [MAX_TURNS]TurnRecord = undefined,
    turn_count: usize = 0,

    attachments: [MAX_TOTAL_ATTACHMENTS]Attachment = undefined,
    attachment_count: usize = 0,

    pending_attachments: [MAX_ATTACHMENTS_PER_TURN]Attachment = undefined,
    pending_attachment_count: u8 = 0,

    text_pool: []u8,
    text_pool_used: usize = 0,
    prompt_buf: []u8,

    pub fn init(allocator: std.mem.Allocator, cfg: ChatConfig) Error!ChatSession {
        const text_pool = try allocator.alloc(u8, DEFAULT_TEXT_POOL_BYTES);
        errdefer allocator.free(text_pool);
        const prompt_buf = try allocator.alloc(u8, DEFAULT_PROMPT_BUF_BYTES);

        var session: ChatSession = .{
            .cfg = cfg,
            .text_pool = text_pool,
            .prompt_buf = prompt_buf,
        };

        for (0..cfg.initial_image_count) |i| {
            try session.queueAttachment(.image, cfg.initial_images[i], null);
        }
        for (0..cfg.initial_video_count) |i| {
            try session.queueAttachment(.video, cfg.initial_videos[i], null);
        }
        return session;
    }

    pub fn deinit(self: *ChatSession, allocator: std.mem.Allocator) void {
        allocator.free(self.prompt_buf);
        allocator.free(self.text_pool);
        self.* = undefined;
    }

    pub fn clear(self: *ChatSession, runner: ?TurnRunner) void {
        self.turn_count = 0;
        self.attachment_count = 0;
        self.pending_attachment_count = 0;
        self.text_pool_used = 0;
        if (runner) |r| {
            if (r.onClearFn) |cb| cb(r.ctx);
        }
    }

    pub fn activeCheckpointId(self: *const ChatSession) u32 {
        if (self.turn_count == 0) return prefix_cache.NONE_ID;
        return self.turns[self.turn_count - 1].post_turn_checkpoint_entry_id;
    }

    pub fn activeFullBlockId(self: *const ChatSession) u32 {
        if (self.turn_count == 0) return prefix_cache.NONE_ID;
        return self.turns[self.turn_count - 1].post_turn_full_block_id;
    }

    /// Helper wrapping `PrefixCache.insertBoundaryCheckpoint` (`C15`, ADR-0020)
    /// so `main.zig` runners and tests can persist exact ChatML turn-boundary
    /// checkpoints and chain them across multi-turn `/retry` & `/edit`.
    pub fn recordBoundaryCheckpoint(
        cache: *prefix_cache.PrefixCache,
        parent_checkpoint_id: u32,
        boundary_block_tokens: []const u32,
        layer_handles: []const prefix_cache.KvBlockHandle,
    ) prefix_cache.Error!u32 {
        const res = try cache.insertBoundaryCheckpoint(
            parent_checkpoint_id,
            boundary_block_tokens,
            layer_handles,
            .{},
        );
        return res.entry_id;
    }

    pub fn queueAttachment(
        self: *ChatSession,
        kind: AttachmentKind,
        path: []const u8,
        runner: ?TurnRunner,
    ) Error!void {
        if (path.len == 0) return error.EmptyAttachmentPath;
        if (self.pending_attachment_count >= MAX_ATTACHMENTS_PER_TURN) {
            return error.AttachmentLimitExceeded;
        }
        const native_handle: ?*anyopaque = if (runner) |r|
            if (r.loadAttachmentFn) |load_fn| try load_fn(r.ctx, kind, path) else null
        else
            null;

        const off = try self.pushText(path);
        self.pending_attachments[self.pending_attachment_count] = .{
            .kind = kind,
            .path_off = off,
            .path_len = @intCast(path.len),
            .native_handle = native_handle,
        };
        self.pending_attachment_count += 1;
    }

    fn pushText(self: *ChatSession, bytes: []const u8) Error!u32 {
        if (self.text_pool_used + bytes.len > self.text_pool.len) {
            return error.TextPoolFull;
        }
        const start = self.text_pool_used;
        @memcpy(self.text_pool[start .. start + bytes.len], bytes);
        self.text_pool_used += bytes.len;
        return @intCast(start);
    }

    fn sliceText(self: *const ChatSession, off: u32, len: u32) []const u8 {
        const s: usize = off;
        const e: usize = s + @as(usize, len);
        return self.text_pool[s..e];
    }

    /// Processes a single interactive line (either a slash command or user message).
    pub fn handleInteractiveLine(
        self: *ChatSession,
        line: []const u8,
        runner: TurnRunner,
        out_sink: ByteSink,
        info_sink: ByteSink,
    ) Error!LineOutcome {
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) return .continued;

        if (try parseSlashCommand(trimmed)) |cmd| {
            switch (cmd) {
                .exit => return .exited,
                .help => {
                    try info_sink.writeAll(HELP_TEXT);
                    return .continued;
                },
                .clear => {
                    self.clear(runner);
                    try info_sink.writeAll("[conversation cleared]\n");
                    return .continued;
                },
                .image => |p| {
                    try self.queueAttachment(.image, p, runner);
                    try info_sink.print("[attached image: {s}]\n", .{p});
                    return .continued;
                },
                .video => |p| {
                    try self.queueAttachment(.video, p, runner);
                    try info_sink.print("[attached video: {s}]\n", .{p});
                    return .continued;
                },
                .lora => |arg| {
                    if (runner.onLoraCommandFn) |cb| {
                        try cb(runner.ctx, arg, info_sink);
                    } else {
                        try info_sink.writeAll("[no LoRA adapters loaded]\n");
                    }
                    return .continued;
                },
                .retry => {
                    try self.retryLastTurn(runner, out_sink, info_sink);
                    return .continued;
                },
                .edit => |new_msg| {
                    try self.editLastTurn(new_msg, runner, out_sink, info_sink);
                    return .continued;
                },
            }
        }

        try self.submitUserTurn(trimmed, runner, out_sink, info_sink);
        return .continued;
    }

    /// Submits a new user turn, formats Qwen3 ChatML with `C15` message-boundary
    /// tracking and `G14` attachment markers, streams output, and prints 1-line stats.
    pub fn submitUserTurn(
        self: *ChatSession,
        user_message: []const u8,
        runner: TurnRunner,
        out_sink: ByteSink,
        summary_sink: ByteSink,
    ) Error!void {
        if (self.turn_count >= MAX_TURNS) return error.HistoryFull;
        if (self.attachment_count + self.pending_attachment_count > MAX_TOTAL_ATTACHMENTS) {
            return error.AttachmentLimitExceeded;
        }

        // Capture pool marks BEFORE pushing this turn's user message so `/edit`
        // and failed turns can rewind cleanly without leaking text pool bytes.
        // Note: pending attachments were pushed into `text_pool` earlier; copy their
        // paths cleanly after `pool_mark` if any exist.
        var staged_paths_buf: [MAX_ATTACHMENTS_PER_TURN][512]u8 = undefined;
        var staged_lens: [MAX_ATTACHMENTS_PER_TURN]usize = @splat(0);
        var staged_kinds: [MAX_ATTACHMENTS_PER_TURN]AttachmentKind = @splat(.image);
        var staged_handles: [MAX_ATTACHMENTS_PER_TURN]?*anyopaque = @splat(null);
        const pend_n: usize = self.pending_attachment_count;

        const base_mark = if (self.turn_count == 0)
            0
        else
            @as(usize, self.turns[self.turn_count - 1].assistant_off) +
                @as(usize, self.turns[self.turn_count - 1].assistant_len);

        for (0..pend_n) |i| {
            const pa = self.pending_attachments[i];
            const pslice = self.sliceText(pa.path_off, pa.path_len);
            if (pslice.len > staged_paths_buf[i].len) return error.TextPoolFull;
            @memcpy(staged_paths_buf[i][0..pslice.len], pslice);
            staged_lens[i] = pslice.len;
            staged_kinds[i] = pa.kind;
            staged_handles[i] = pa.native_handle;
        }
        self.pending_attachment_count = 0;
        self.text_pool_used = base_mark;

        const attach_mark = self.attachment_count;
        for (0..pend_n) |i| {
            const poff = try self.pushText(staged_paths_buf[i][0..staged_lens[i]]);
            self.attachments[self.attachment_count] = .{
                .kind = staged_kinds[i],
                .path_off = poff,
                .path_len = @intCast(staged_lens[i]),
                .native_handle = staged_handles[i],
            };
            self.attachment_count += 1;
        }

        const user_off = try self.pushText(user_message);
        const user_len: u32 = @intCast(user_message.len);

        try self.executePreparedTurn(
            base_mark,
            attach_mark,
            user_off,
            user_len,
            @intCast(attach_mark),
            @intCast(pend_n),
            runner,
            out_sink,
            summary_sink,
        );
    }

    /// Rewinds the last completed turn back to its `C15` parent boundary checkpoint
    /// and regenerates it with identical user text and `G14` attachments.
    pub fn retryLastTurn(
        self: *ChatSession,
        runner: TurnRunner,
        out_sink: ByteSink,
        summary_sink: ByteSink,
    ) Error!void {
        if (self.turn_count == 0) return error.NoTurnToRetry;
        const last = self.turns[self.turn_count - 1];
        // Truncate assistant text of `last`, keeping `last`'s attachments + `user_off..user_off+user_len`.
        self.turn_count -= 1;
        self.text_pool_used = @as(usize, last.user_off) + @as(usize, last.user_len);
        self.attachment_count = @as(usize, last.attach_start) + @as(usize, last.attach_count);

        try self.executePreparedTurn(
            last.text_pool_mark_before,
            last.attachment_mark_before,
            last.user_off,
            last.user_len,
            last.attach_start,
            last.attach_count,
            runner,
            out_sink,
            summary_sink,
        );
    }

    /// Rewinds the last completed turn back to its `C15` parent boundary checkpoint,
    /// replaces the user message text with `new_message` (keeping turn attachments),
    /// and regenerates immediately.
    pub fn editLastTurn(
        self: *ChatSession,
        new_message: []const u8,
        runner: TurnRunner,
        out_sink: ByteSink,
        summary_sink: ByteSink,
    ) Error!void {
        if (self.turn_count == 0) return error.NoTurnToEdit;
        if (new_message.len == 0) return error.EmptyEditMessage;

        const last = self.turns[self.turn_count - 1];
        self.turn_count -= 1;
        // Rewind text pool right after `last`'s attachment paths (which equals `last.user_off`).
        self.text_pool_used = @as(usize, last.user_off);
        self.attachment_count = @as(usize, last.attach_start) + @as(usize, last.attach_count);

        const new_user_off = try self.pushText(new_message);
        const new_user_len: u32 = @intCast(new_message.len);

        try self.executePreparedTurn(
            last.text_pool_mark_before,
            last.attachment_mark_before,
            new_user_off,
            new_user_len,
            last.attach_start,
            last.attach_count,
            runner,
            out_sink,
            summary_sink,
        );
    }

    fn executePreparedTurn(
        self: *ChatSession,
        pool_mark_before: usize,
        attach_mark_before: usize,
        user_off: u32,
        user_len: u32,
        attach_start: u16,
        attach_count: u8,
        runner: TurnRunner,
        out_sink: ByteSink,
        summary_sink: ByteSink,
    ) Error!void {
        const parent_ckpt = self.activeCheckpointId();
        const parent_full_blk = self.activeFullBlockId();
        const built = try self.buildChatMlPrompt(user_off, user_len, attach_start, attach_count);

        var attach_views: [MAX_ATTACHMENTS_PER_TURN]AttachmentView = undefined;
        for (0..attach_count) |i| {
            const a = self.attachments[@as(usize, attach_start) + i];
            attach_views[i] = .{
                .kind = a.kind,
                .path = self.sliceText(a.path_off, a.path_len),
                .native_handle = a.native_handle,
            };
        }

        const effective_enable_thinking = if (self.cfg.reasoning_effort) |re|
            (self.cfg.enable_thinking and re.isThinkingEnabled())
        else
            self.cfg.enable_thinking;

        const req: TurnRequest = .{
            .turn_index = @intCast(self.turn_count),
            .chatml_prompt = built.prompt,
            .user_message = self.sliceText(user_off, user_len),
            .last_user_msg_byte_offset = built.last_user_msg_byte_offset,
            .parent_checkpoint_entry_id = parent_ckpt,
            .parent_full_block_id = parent_full_blk,
            .attachments = attach_views[0..attach_count],
            .max_tokens = self.cfg.max_tokens,
            .temperature = self.cfg.temperature,
            .top_k = self.cfg.top_k,
            .top_p = self.cfg.top_p,
            .min_p = self.cfg.min_p,
            .repetition_penalty = self.cfg.repetition_penalty,
            .presence_penalty = self.cfg.presence_penalty,
            .frequency_penalty = self.cfg.frequency_penalty,
            .seed = self.cfg.seed,
            .enable_thinking = effective_enable_thinking,
            .reasoning_effort = self.cfg.reasoning_effort,
        };

        const assistant_start = self.text_pool_used;
        const start_ns = try vmstat.nowNs();
        var stream: TurnStream = .{
            .out_sink = out_sink,
            .think_display = self.cfg.think_display,
            .use_color = self.cfg.use_color,
            .start_ns = start_ns,
            .assistant_content_buf = self.text_pool[assistant_start..],
        };
        // If `enable_thinking == false`, Qwen3 ChatML pre-closes `<think>\n\n</think>\n`
        // in the prompt, so streaming starts in `.content`; otherwise `.content`
        // transitions into `.reasoning` when `<think>` arrives.
        stream.splitter.resetForDialect(self.cfg.chat_template.dialect, built.starts_in_think);

        var stats = try runner.runTurnFn(runner.ctx, &req, &stream);
        try stream.finish();

        const end_ns = try vmstat.nowNs();
        if (stats.total_ns == 0) {
            stats.total_ns = end_ns -| start_ns;
        }
        if (stats.ttft_ns == 0) {
            if (stream.first_token_ns) |ft| {
                stats.ttft_ns = ft -| start_ns;
            }
        }
        if (stats.generated_tokens == 0 and stream.emitted_chunks > 0) {
            stats.generated_tokens = stream.emitted_chunks;
        }

        // Ensure trailing newline after streamed assistant reply in scrolling terminal.
        const assistant_slice = stream.assistant_content_buf[0..stream.assistant_content_len];
        if (assistant_slice.len == 0 or assistant_slice[assistant_slice.len - 1] != '\n') {
            try out_sink.writeAll("\n");
        }

        self.text_pool_used = assistant_start + stream.assistant_content_len;
        self.turns[self.turn_count] = .{
            .text_pool_mark_before = pool_mark_before,
            .attachment_mark_before = attach_mark_before,
            .user_off = user_off,
            .user_len = user_len,
            .assistant_off = @intCast(assistant_start),
            .assistant_len = @intCast(stream.assistant_content_len),
            .attach_start = attach_start,
            .attach_count = attach_count,
            .parent_checkpoint_entry_id = parent_ckpt,
            .parent_full_block_id = parent_full_blk,
            .post_turn_checkpoint_entry_id = stats.checkpoint_entry_id,
            .post_turn_full_block_id = stats.full_block_entry_id,
            .last_user_msg_byte_offset = built.last_user_msg_byte_offset,
        };
        self.turn_count += 1;

        if (self.cfg.show_stats) {
            try self.writePostTurnSummary(summary_sink, stats);
        }
    }

    fn writePostTurnSummary(self: *const ChatSession, sink: ByteSink, stats: TurnStats) Error!void {
        if (self.cfg.use_color) {
            try sink.print(
                "\x1b[2m[{d} tok | {d:.1} tok/s | TTFT {d:.1} ms | PrefixCache hit {d}/{d} tok]\x1b[0m\n",
                .{
                    stats.generated_tokens,
                    stats.tokensPerSecond(),
                    stats.ttftMs(),
                    stats.prefix_hit_tokens,
                    stats.prompt_tokens,
                },
            );
        } else {
            try sink.print(
                "[{d} tok | {d:.1} tok/s | TTFT {d:.1} ms | PrefixCache hit {d}/{d} tok]\n",
                .{
                    stats.generated_tokens,
                    stats.tokensPerSecond(),
                    stats.ttftMs(),
                    stats.prefix_hit_tokens,
                    stats.prompt_tokens,
                },
            );
        }
    }

    const BuiltPrompt = struct {
        prompt: []const u8,
        last_user_msg_byte_offset: ?usize,
        starts_in_think: bool,
    };

    fn mapTemplateErr(err: chat_template.TemplateError) Error {
        return switch (err) {
            error.BufferTooSmall => error.PromptTooLarge,
            error.DepthExceeded,
            error.InvalidJson,
            error.InvalidReasoningEffort,
            error.InvalidSpec,
            error.MissingToolName,
            error.UnsupportedContent,
            error.UnsupportedDialect,
            => error.InvalidCliArgument,
        };
    }

    fn pushUserTurnWithAttachments(
        self: *const ChatSession,
        renderer: *chat_template.PromptRenderer,
        user_off: u32,
        user_len: u32,
        attach_start: u16,
        attach_count: u8,
        msg_idx: usize,
    ) Error!void {
        const user_txt = self.sliceText(user_off, user_len);
        if (attach_count > 0) {
            var parts_buf: [MAX_ATTACHMENTS_PER_TURN * 2 + 1]chat_template.ContentPart = undefined;
            var p_len: usize = 0;
            for (0..attach_count) |ai| {
                const att = self.attachments[@as(usize, attach_start) + ai];
                parts_buf[p_len] = switch (att.kind) {
                    .image => .image,
                    .video => .video,
                };
                p_len += 1;
                parts_buf[p_len] = .{ .text = "\n" };
                p_len += 1;
            }
            parts_buf[p_len] = .{ .text = user_txt };
            p_len += 1;
            renderer.pushMessage(.{
                .role = .user,
                .parts = parts_buf[0..p_len],
            }, msg_idx) catch |err| return mapTemplateErr(err);
        } else {
            renderer.pushMessage(.{
                .role = .user,
                .content = user_txt,
            }, msg_idx) catch |err| return mapTemplateErr(err);
        }
    }

    fn buildChatMlPrompt(
        self: *ChatSession,
        curr_user_off: u32,
        curr_user_len: u32,
        curr_attach_start: u16,
        curr_attach_count: u8,
    ) Error!BuiltPrompt {
        const effective_enable_thinking = if (self.cfg.reasoning_effort) |re|
            (self.cfg.enable_thinking and re.isThinkingEnabled())
        else
            self.cfg.enable_thinking;
        const has_leading_sys = self.cfg.system_prompt.len > 0;
        const last_user_idx = (if (has_leading_sys) @as(usize, 1) else 0) + (self.turn_count * 2);

        var fw = chat_template.FixedWriter.init(self.prompt_buf);
        var renderer = chat_template.PromptRenderer.init(
            &self.cfg.chat_template,
            &fw,
            .{
                .reasoning_effort = self.cfg.reasoning_effort,
                .enable_thinking = effective_enable_thinking,
                .repl_compact_nothink = (self.cfg.chat_template.dialect == .chatml),
            },
            has_leading_sys,
            last_user_idx,
        );
        renderer.begin() catch |err| return mapTemplateErr(err);

        var msg_idx: usize = 0;
        if (has_leading_sys) {
            renderer.pushMessage(.{
                .role = .system,
                .content = self.cfg.system_prompt,
            }, msg_idx) catch |err| return mapTemplateErr(err);
            msg_idx += 1;
        }

        for (self.turns[0..self.turn_count]) |t| {
            try self.pushUserTurnWithAttachments(&renderer, t.user_off, t.user_len, t.attach_start, t.attach_count, msg_idx);
            msg_idx += 1;

            renderer.pushMessage(.{
                .role = .assistant,
                .content = self.sliceText(t.assistant_off, t.assistant_len),
            }, msg_idx) catch |err| return mapTemplateErr(err);
            msg_idx += 1;
        }

        try self.pushUserTurnWithAttachments(&renderer, curr_user_off, curr_user_len, curr_attach_start, curr_attach_count, msg_idx);
        renderer.finish() catch |err| return mapTemplateErr(err);

        return .{
            .prompt = fw.written(),
            .last_user_msg_byte_offset = renderer.last_user_msg_byte_offset,
            .starts_in_think = renderer.starts_in_think,
        };
    }
};

// ---------------------------------------------------------------------------
// Zero-Alloc UTF-8 Display Width & Interactive Line Editor (CJK-Safe Backspace)
// ---------------------------------------------------------------------------

/// Terminal column width for a Unicode scalar value (`0`, `1`, or `2`).
/// Zero-alloc replacement for `wcwidth` covering C0/C1 controls, combining marks,
/// variation selectors, ZWJ, East Asian Wide/Fullwidth/CJK/Hangul/Kana, and Emoji.
pub fn utf8CodepointWidth(cp: u21) u2 {
    if (cp < 0x20 or (cp >= 0x7F and cp <= 0x9F)) return 0;
    if ((cp >= 0x0300 and cp <= 0x036F) or
        (cp >= 0x1AB0 and cp <= 0x1AFF) or
        (cp >= 0x1DC0 and cp <= 0x1DFF) or
        (cp >= 0x200B and cp <= 0x200F) or
        (cp >= 0x20D0 and cp <= 0x20FF) or
        (cp >= 0xFE00 and cp <= 0xFE0F) or
        (cp >= 0xFE20 and cp <= 0xFE2F))
    {
        return 0;
    }

    if ((cp >= 0x1100 and cp <= 0x115F) or
        (cp >= 0x2329 and cp <= 0x232A) or
        (cp >= 0x2E80 and cp <= 0x303E) or
        (cp >= 0x3040 and cp <= 0xA4CF) or
        (cp >= 0xA960 and cp <= 0xA97C) or
        (cp >= 0xAC00 and cp <= 0xD7A3) or
        (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0xFE10 and cp <= 0xFE19) or
        (cp >= 0xFE30 and cp <= 0xFE6F) or
        (cp >= 0xFF01 and cp <= 0xFF60) or
        (cp >= 0xFFE0 and cp <= 0xFFE6) or
        (cp >= 0x1F300 and cp <= 0x1FAFF) or
        (cp >= 0x20000 and cp <= 0x3FFFD))
    {
        return 2;
    }

    return 1;
}

pub const PopGraphemeResult = struct {
    new_len: usize,
    cols: usize,
};

fn decodePrevUtf8(buf: []const u8) struct { start: usize, cp: ?u21 } {
    std.debug.assert(buf.len > 0);
    var start = buf.len - 1;
    var back: usize = 0;
    while (start > 0 and (buf[start] & 0xC0) == 0x80 and back < 3) {
        start -= 1;
        back += 1;
    }
    const seq = buf[start..];
    const expected_len = std.unicode.utf8ByteSequenceLength(seq[0]) catch {
        return .{ .start = buf.len - 1, .cp = null };
    };
    if (expected_len != seq.len) {
        return .{ .start = buf.len - 1, .cp = null };
    }
    const cp = std.unicode.utf8Decode(seq) catch {
        return .{ .start = buf.len - 1, .cp = null };
    };
    return .{ .start = start, .cp = cp };
}

/// Pops the trailing grapheme cluster (trailing width-0 codepoints plus 1 base
/// codepoint, continuing across any preceding `U+200D` ZWJ joiner) and returns
/// the new byte length and exact display columns to erase. Falls back safely to
/// popping 1 byte / 1 column on malformed UTF-8.
pub fn popLastGrapheme(buf: []const u8) PopGraphemeResult {
    if (buf.len == 0) return .{ .new_len = 0, .cols = 0 };
    var pos: usize = buf.len;
    var cols: usize = 0;

    while (pos > 0) {
        while (pos > 0) {
            const prev = decodePrevUtf8(buf[0..pos]);
            const cp = prev.cp orelse {
                return .{ .new_len = prev.start, .cols = if (cols == 0) 1 else cols };
            };
            const w = utf8CodepointWidth(cp);
            if (w == 0) {
                pos = prev.start;
                continue;
            }
            pos = prev.start;
            cols = @max(cols, @as(usize, w));
            break;
        }
        if (pos == 0) break;

        const maybe_zwj = decodePrevUtf8(buf[0..pos]);
        if (maybe_zwj.cp) |cp| {
            if (cp == 0x200D) {
                pos = maybe_zwj.start;
                continue;
            }
        }
        break;
    }

    return .{ .new_len = pos, .cols = cols };
}

/// Computes total terminal display columns for `buf` consistently with `popLastGrapheme`.
pub fn displayWidth(buf: []const u8) usize {
    var pos: usize = buf.len;
    var total: usize = 0;
    while (pos > 0) {
        const res = popLastGrapheme(buf[0..pos]);
        total += res.cols;
        if (res.new_len >= pos) break;
        pos = res.new_len;
    }
    return total;
}

/// Zero-allocation terminal line editor for interactive TTY mode (`!ICANON`, `!ECHO`).
/// Accumulates UTF-8 multi-byte characters, performs exact column-width erasure for
/// CJK/wide glyphs and emoji on Backspace/`Ctrl-W`/`Ctrl-U`, handles `Ctrl-D` EOF on
/// empty lines, and swallows ANSI CSI/SS3 escape sequences (e.g. arrow keys).
pub const InteractiveLineEditor = struct {
    pub const EscState = enum { none, esc, csi, ss3 };

    pub const FeedOutcome = union(enum) {
        pending,
        line: []const u8,
        eof,
    };

    line_buf: []u8,
    len: usize = 0,
    utf8_pending: [4]u8 = @splat(0),
    utf8_need: u3 = 0,
    utf8_got: u3 = 0,
    esc_state: EscState = .none,
    last_was_cr: bool = false,

    pub fn currentLine(self: *const InteractiveLineEditor) []const u8 {
        return self.line_buf[0..self.len];
    }

    fn writeRepeatedByte(out_sink: ByteSink, byte: u8, count: usize) Error!void {
        const bs_block: [64]u8 = @splat(0x08);
        const sp_block: [64]u8 = @splat(' ');
        const block: *const [64]u8 = if (byte == 0x08) &bs_block else &sp_block;
        var rem = count;
        while (rem > 0) {
            const n = @min(rem, block.len);
            try out_sink.writeAll(block[0..n]);
            rem -= n;
        }
    }

    fn emitEraseCols(out_sink: ByteSink, cols: usize) Error!void {
        if (cols == 0) return;
        try writeRepeatedByte(out_sink, 0x08, cols);
        try writeRepeatedByte(out_sink, ' ', cols);
        try writeRepeatedByte(out_sink, 0x08, cols);
    }

    pub fn feedByte(self: *InteractiveLineEditor, b: u8, out_sink: ByteSink) Error!FeedOutcome {
        // Absorb ANSI escape sequences (`\x1b[...` CSI, `\x1bO.` SS3, or 2-byte ESC+ch).
        switch (self.esc_state) {
            .none => {},
            .esc => {
                self.last_was_cr = false;
                if (b == '[') {
                    self.esc_state = .csi;
                } else if (b == 'O') {
                    self.esc_state = .ss3;
                } else if (b == 0x1b) {
                    self.esc_state = .esc;
                } else {
                    self.esc_state = .none;
                }
                return .pending;
            },
            .csi => {
                self.last_was_cr = false;
                if (b >= 0x40 and b <= 0x7E) {
                    self.esc_state = .none;
                    return .pending;
                } else if (b == 0x1b) {
                    self.esc_state = .esc;
                    return .pending;
                } else if (b >= 0x20 and b <= 0x3F) {
                    return .pending;
                } else {
                    // Malformed/aborted CSI (e.g. Enter/Ctrl-D): exit CSI and handle `b`.
                    self.esc_state = .none;
                }
            },
            .ss3 => {
                self.last_was_cr = false;
                self.esc_state = .none;
                return .pending;
            },
        }

        // Collapse `\r\n` into a single line submission.
        if (b == '\n' and self.last_was_cr) {
            self.last_was_cr = false;
            return .pending;
        }
        self.last_was_cr = false;

        // Multi-byte UTF-8 continuation / lead bytes (`>= 0x80`).
        if (b >= 0x80) {
            if ((b & 0xC0) == 0x80) {
                if (self.utf8_need > 0 and self.utf8_got < self.utf8_need) {
                    self.utf8_pending[self.utf8_got] = b;
                    self.utf8_got += 1;
                    if (self.utf8_got == self.utf8_need) {
                        const seq = self.utf8_pending[0..self.utf8_need];
                        self.utf8_need = 0;
                        self.utf8_got = 0;
                        const decoded_cp: ?u21 = std.unicode.utf8Decode(seq) catch null;
                        if (decoded_cp != null) {
                            if (self.len + seq.len > self.line_buf.len) return error.PromptTooLarge;
                            @memcpy(self.line_buf[self.len .. self.len + seq.len], seq);
                            self.len += seq.len;
                            try out_sink.writeAll(seq);
                        }
                    }
                } else {
                    self.utf8_need = 0;
                    self.utf8_got = 0;
                }
            } else {
                if (std.unicode.utf8ByteSequenceLength(b)) |seq_len| {
                    if (seq_len >= 2 and seq_len <= 4) {
                        self.utf8_pending[0] = b;
                        self.utf8_need = @intCast(seq_len);
                        self.utf8_got = 1;
                    } else {
                        self.utf8_need = 0;
                        self.utf8_got = 0;
                    }
                } else |_| {
                    self.utf8_need = 0;
                    self.utf8_got = 0;
                }
            }
            return .pending;
        }

        // Any ASCII byte terminates an incomplete UTF-8 sequence.
        self.utf8_need = 0;
        self.utf8_got = 0;

        switch (b) {
            0x1b => {
                self.esc_state = .esc;
                return .pending;
            },
            '\r', '\n' => {
                self.last_was_cr = (b == '\r');
                self.esc_state = .none;
                try out_sink.writeAll("\n");
                const submitted = self.line_buf[0..self.len];
                self.len = 0;
                return .{ .line = submitted };
            },
            0x08, 0x7f => {
                if (self.len > 0) {
                    const res = popLastGrapheme(self.line_buf[0..self.len]);
                    self.len = res.new_len;
                    try emitEraseCols(out_sink, res.cols);
                }
                return .pending;
            },
            0x15 => {
                // Ctrl-U: kill entire line with exact display-column erase.
                const total_cols = displayWidth(self.line_buf[0..self.len]);
                self.len = 0;
                try emitEraseCols(out_sink, total_cols);
                return .pending;
            },
            0x17 => {
                // Ctrl-W: erase trailing spaces, then preceding word / CJK run.
                var total_cols: usize = 0;
                while (self.len > 0 and (self.line_buf[self.len - 1] == ' ' or self.line_buf[self.len - 1] == '\t')) {
                    const res = popLastGrapheme(self.line_buf[0..self.len]);
                    self.len = res.new_len;
                    total_cols += res.cols;
                }
                while (self.len > 0 and self.line_buf[self.len - 1] != ' ' and self.line_buf[self.len - 1] != '\t') {
                    const res = popLastGrapheme(self.line_buf[0..self.len]);
                    self.len = res.new_len;
                    total_cols += res.cols;
                }
                try emitEraseCols(out_sink, total_cols);
                return .pending;
            },
            0x04 => {
                // Ctrl-D: EOF on empty line.
                if (self.len == 0) return .eof;
                return .pending;
            },
            '\t' => {
                // Normalize tab to 1-column space so backspace column math stays exact.
                if (self.len >= self.line_buf.len) return error.PromptTooLarge;
                self.line_buf[self.len] = ' ';
                self.len += 1;
                try out_sink.writeAll(" ");
                return .pending;
            },
            0x20...0x7E => {
                if (self.len >= self.line_buf.len) return error.PromptTooLarge;
                self.line_buf[self.len] = b;
                self.len += 1;
                const one = [1]u8{b};
                try out_sink.writeAll(&one);
                return .pending;
            },
            else => return .pending,
        }
    }
};

// ---------------------------------------------------------------------------
// Scrolling Terminal REPL & Non-Interactive `stdin` Pipe Entry Point
// ---------------------------------------------------------------------------

/// Runs `aminfer chat` over `in_fd`, `out_fd`, and `err_fd`:
/// - When `in_fd` is a TTY (`is_interactive == true`), runs the plain scrolling
///   line-REPL (`> ` prompt, slash commands, dimmed `<think>`, 1-line summary).
/// - When `in_fd` is piped (`is_interactive == false`), reads all of `in_fd`
///   as a single non-interactive turn (supporting optional leading `/image` /
///   `/video` lines), streams tokens to `out_fd`, and writes the 1-line summary
///   to `err_fd` so `stdout` stays clean for shell pipelines.
pub fn runReplOrPipe(
    allocator: std.mem.Allocator,
    cfg: ChatConfig,
    runner: TurnRunner,
    in_fd: std.c.fd_t,
    out_fd: std.c.fd_t,
    err_fd: std.c.fd_t,
    force_interactive: ?bool,
) Error!void {
    const is_interactive = force_interactive orelse (telemetry.isTty(in_fd) and telemetry.isTty(out_fd));
    var effective_cfg = cfg;
    if (!telemetry.isTty(out_fd) and force_interactive == null) {
        effective_cfg.use_color = false;
        if (effective_cfg.think_display == .dimmed) {
            effective_cfg.think_display = .plain;
        }
    }

    var session = try ChatSession.init(allocator, effective_cfg);
    defer session.deinit(allocator);

    const out_fd_val = out_fd;
    const err_fd_val = err_fd;
    const out_sink = ByteSink.fromFd(&out_fd_val);
    const err_sink = ByteSink.fromFd(&err_fd_val);

    const line_buf = try allocator.alloc(u8, DEFAULT_LINE_BUF_BYTES);
    defer allocator.free(line_buf);

    if (!is_interactive) {
        // Non-interactive `stdin` pipe mode: read entire stdin up to EOF.
        var total: usize = 0;
        while (total < line_buf.len) {
            const rc = std.c.read(in_fd, line_buf[total..].ptr, line_buf.len - total);
            if (rc < 0) {
                if (std.c.errno(rc) == .INTR) continue;
                return error.ReadFailed;
            }
            if (rc == 0) break;
            total += @intCast(rc);
        }
        const piped = line_buf[0..total];
        try runPipedInput(&session, piped, runner, out_sink, err_sink);
        return;
    }

    // Plain scrolling terminal line-REPL (zero alternate screen!).
    const prompt_str = if (effective_cfg.use_color) "\x1b[1;36m>\x1b[0m " else "> ";

    // On a real interactive TTY, disable Darwin canonical echo (`ICANON | ECHO | IEXTEN`)
    // and drive `InteractiveLineEditor` so wide CJK/Emoji glyphs erase 2 columns cleanly
    // instead of macOS's 1-column `\b \b` ghosting. Keep `ISIG` enabled for `Ctrl-C`.
    if (telemetry.isTty(in_fd) and telemetry.isTty(out_fd)) {
        var orig_termios: std.c.termios = undefined;
        if (std.c.tcgetattr(in_fd, &orig_termios) == 0) {
            var edit_termios = orig_termios;
            edit_termios.lflag.ICANON = false;
            edit_termios.lflag.ECHO = false;
            edit_termios.lflag.IEXTEN = false;
            edit_termios.lflag.ISIG = true;
            edit_termios.iflag.ICRNL = true;
            if (@hasField(@TypeOf(edit_termios.iflag), "IUTF8")) {
                edit_termios.iflag.IUTF8 = true;
            }
            edit_termios.cc[@backingInt(std.c.V.MIN)] = 1;
            edit_termios.cc[@backingInt(std.c.V.TIME)] = 0;

            if (std.c.tcsetattr(in_fd, .NOW, &edit_termios) == 0) {
                defer _ = std.c.tcsetattr(in_fd, .NOW, &orig_termios);
                var editor: InteractiveLineEditor = .{ .line_buf = line_buf };
                var read_buf: [256]u8 = undefined;

                try out_sink.writeAll(prompt_str);
                while (true) {
                    const rc = std.c.read(in_fd, read_buf[0..].ptr, read_buf.len);
                    if (rc < 0) {
                        if (std.c.errno(rc) == .INTR) continue;
                        return error.ReadFailed;
                    }
                    if (rc == 0) {
                        try out_sink.writeAll("\n");
                        if (editor.len > 0) {
                            _ = try session.handleInteractiveLine(editor.currentLine(), runner, out_sink, out_sink);
                        }
                        return;
                    }
                    const n: usize = @intCast(rc);
                    for (read_buf[0..n]) |b| {
                        switch (try editor.feedByte(b, out_sink)) {
                            .pending => {},
                            .eof => {
                                try out_sink.writeAll("\n");
                                return;
                            },
                            .line => |one_line| {
                                const outcome = session.handleInteractiveLine(one_line, runner, out_sink, out_sink) catch |err| switch (err) {
                                    error.UnknownSlashCommand,
                                    error.EmptyAttachmentPath,
                                    error.EmptyEditMessage,
                                    error.NoTurnToRetry,
                                    error.NoTurnToEdit,
                                    => blk: {
                                        try err_sink.print("[error: {s}]\n", .{@errorName(err)});
                                        break :blk LineOutcome.continued;
                                    },
                                    else => return err,
                                };
                                if (outcome == .exited) return;
                                try out_sink.writeAll(prompt_str);
                            },
                        }
                    }
                }
            }
        }
    }

    var pending_len: usize = 0;

    try out_sink.writeAll(prompt_str);
    while (true) {
        if (std.mem.indexOfScalar(u8, line_buf[0..pending_len], '\n')) |nl_idx| {
            const one_line = line_buf[0..nl_idx];
            // Copy line before compacting or process first then compact!
            // Note: `handleInteractiveLine` copies any needed slices into `session.text_pool`
            // before returning, so `one_line` only needs to live across `handleInteractiveLine`.
            const outcome = session.handleInteractiveLine(one_line, runner, out_sink, out_sink) catch |err| switch (err) {
                error.UnknownSlashCommand,
                error.EmptyAttachmentPath,
                error.EmptyEditMessage,
                error.NoTurnToRetry,
                error.NoTurnToEdit,
                => blk: {
                    try err_sink.print("[error: {s}]\n", .{@errorName(err)});
                    break :blk LineOutcome.continued;
                },
                else => return err,
            };

            const rem = pending_len - (nl_idx + 1);
            if (rem > 0) {
                std.mem.copyForwards(u8, line_buf[0..rem], line_buf[nl_idx + 1 .. pending_len]);
            }
            pending_len = rem;

            if (outcome == .exited) return;
            try out_sink.writeAll(prompt_str);
            continue;
        }

        if (pending_len >= line_buf.len) return error.PromptTooLarge;
        const rc = std.c.read(in_fd, line_buf[pending_len..].ptr, line_buf.len - pending_len);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) {
            if (pending_len > 0) {
                _ = try session.handleInteractiveLine(line_buf[0..pending_len], runner, out_sink, out_sink);
            } else {
                try out_sink.writeAll("\n");
            }
            return;
        }
        pending_len += @intCast(rc);
    }
}

pub fn runPipedInput(
    session: *ChatSession,
    raw_input: []const u8,
    runner: TurnRunner,
    out_sink: ByteSink,
    summary_sink: ByteSink,
) Error!void {
    // Support optional leading `/image <path>`, `/video <path>`, and `/lora [name|off]`
    // lines in piped stdin followed by the multi-line prompt body.
    var cursor: usize = 0;
    while (cursor < raw_input.len) {
        const line_end = std.mem.indexOfScalarPos(u8, raw_input, cursor, '\n') orelse raw_input.len;
        const line = std.mem.trim(u8, raw_input[cursor..line_end], " \t\r");
        if (std.mem.startsWith(u8, line, "/image ") or
            std.mem.startsWith(u8, line, "/video ") or
            std.mem.eql(u8, line, "/lora") or
            std.mem.startsWith(u8, line, "/lora "))
        {
            const cmd = (try parseSlashCommand(line)) orelse break;
            switch (cmd) {
                .image => |p| try session.queueAttachment(.image, p, runner),
                .video => |p| try session.queueAttachment(.video, p, runner),
                .lora => |arg| {
                    if (runner.onLoraCommandFn) |cb| {
                        try cb(runner.ctx, arg, summary_sink);
                    } else {
                        try summary_sink.writeAll("[no LoRA adapters loaded]\n");
                    }
                },
                else => break,
            }
            cursor = if (line_end < raw_input.len) line_end + 1 else raw_input.len;
        } else {
            break;
        }
    }

    const body = std.mem.trim(u8, raw_input[cursor..], " \t\r\n");
    if (body.len == 0) return;
    try session.submitUserTurn(body, runner, out_sink, summary_sink);
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

const CaptureBuf = struct {
    buf: [16384]u8 = undefined,
    len: usize = 0,

    fn reset(self: *CaptureBuf) void {
        self.len = 0;
    }

    fn written(self: *const CaptureBuf) []const u8 {
        return self.buf[0..self.len];
    }

    fn sink(self: *CaptureBuf) ByteSink {
        return .{
            .ctx = self,
            .writeFn = struct {
                fn write(ctx: *anyopaque, bytes: []const u8) error{WriteFailed}!void {
                    const c: *CaptureBuf = @ptrCast(@alignCast(ctx));
                    if (c.len + bytes.len > c.buf.len) return error.WriteFailed;
                    @memcpy(c.buf[c.len .. c.len + bytes.len], bytes);
                    c.len += bytes.len;
                }
            }.write,
        };
    }
};

test "ChatSession streaming <think> dimming, G14 /image & /video attachments, and C15 /retry + /edit + /clear PrefixCache checkpoints" {
    const gpa = std.testing.allocator;

    // Real PrefixCache (`C15`) verifying message-boundary checkpoint reuse across turns!
    const cache_spec: prefix_cache.PrefixCacheSpec = .{
        .max_entries = 16,
        .block_tokens = 16,
        .num_kv_layers = 2,
    };
    const need_bytes = try prefix_cache.PrefixCache.requiredBytes(cache_spec);
    const test_budget = budget_mod.Budget.compute(.{
        .memory_bytes = need_bytes + 4 * 1024 * 1024 * 1024,
        .user_wire_limit = need_bytes * 4,
        .global_user_wire_limit = need_bytes * 4,
        .gpu_working_set_bytes = need_bytes * 4,
        .headroom_bytes = 0,
    });
    var arena = try arena_mod.WiredArena.init(need_bytes, test_budget);
    defer arena.deinit();
    var cache = try prefix_cache.PrefixCache.init(&arena, cache_spec);

    const MockBackend = struct {
        cache: *prefix_cache.PrefixCache,
        call_count: u32 = 0,
        last_parent_ckpt: u32 = prefix_cache.NONE_ID,
        last_parent_full_blk: u32 = prefix_cache.NONE_ID,
        last_boundary_offset: ?usize = null,
        last_attach_count: usize = 0,
        last_prompt_had_think: bool = false,
        loaded_images: u32 = 0,
        loaded_videos: u32 = 0,
        cleared_count: u32 = 0,

        fn onLoadAttachment(ctx: *anyopaque, kind: AttachmentKind, path: []const u8) Error!?*anyopaque {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (path.len == 0) return error.EmptyAttachmentPath;
            switch (kind) {
                .image => self.loaded_images += 1,
                .video => self.loaded_videos += 1,
            }
            return @ptrFromInt(0xCAFE_0000 + @as(usize, self.loaded_images + self.loaded_videos));
        }

        fn onClear(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.cleared_count += 1;
        }

        fn runTurn(ctx: *anyopaque, req: *const TurnRequest, stream: *TurnStream) Error!TurnStats {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.call_count += 1;
            self.last_parent_ckpt = req.parent_checkpoint_entry_id;
            self.last_parent_full_blk = req.parent_full_block_id;
            self.last_boundary_offset = req.last_user_msg_byte_offset;
            self.last_attach_count = req.attachments.len;
            self.last_prompt_had_think = std.mem.indexOf(u8, req.chatml_prompt, "plan step 1") != null;

            // Emit fragmented `<think>...</think>` across token chunks + final answer.
            try stream.emitText("<thi");
            try stream.emitText("nk>plan step ");
            try stream.emitText("1</th");
            try stream.emitText("ink>Answer: ");
            try stream.emitText(req.user_message);

            // Real C15 PrefixCache tree layout:
            // - Append one canonical full tile (`16` tokens) chained to `req.parent_full_block_id`,
            // - Then insert an intra-tile ChatML boundary checkpoint (`3` tokens, `evict_first = true`)
            //   under that canonical full tile.
            var full_block_tokens: [16]u32 = undefined;
            for (0..16) |i| {
                full_block_tokens[i] = self.call_count * 100 + @as(u32, @intCast(i));
            }
            const full_handles = [_]prefix_cache.KvBlockHandle{
                .{ .k_tex_id = self.call_count * 10 + 1, .v_tex_id = self.call_count * 10 + 2, .pool_slot_id = self.call_count, .storage = .active_tile },
                .{ .k_tex_id = self.call_count * 10 + 3, .v_tex_id = self.call_count * 10 + 4, .pool_slot_id = self.call_count + 100, .storage = .active_tile },
            };
            const full_blk_id = try ChatSession.recordBoundaryCheckpoint(
                self.cache,
                req.parent_full_block_id,
                &full_block_tokens,
                &full_handles,
            );

            const boundary_tail = [_]u32{ 9000 + self.call_count, 9100 + self.call_count, 9200 + self.call_count };
            const tail_handles = [_]prefix_cache.KvBlockHandle{
                .{ .k_tex_id = self.call_count * 10 + 5, .v_tex_id = self.call_count * 10 + 6, .pool_slot_id = self.call_count + 200, .storage = .active_tile },
                .{ .k_tex_id = self.call_count * 10 + 7, .v_tex_id = self.call_count * 10 + 8, .pool_slot_id = self.call_count + 300, .storage = .active_tile },
            };
            const ckpt_id = try ChatSession.recordBoundaryCheckpoint(
                self.cache,
                full_blk_id,
                &boundary_tail,
                &tail_handles,
            );

            return .{
                .prompt_tokens = 64,
                .generated_tokens = 12,
                .prefix_hit_tokens = if (req.parent_checkpoint_entry_id != prefix_cache.NONE_ID) 48 else 16,
                .ttft_ns = 8_500_000,
                .total_ns = 100_000_000,
                .checkpoint_entry_id = ckpt_id,
                .full_block_entry_id = full_blk_id,
            };
        }
    };

    var backend: MockBackend = .{ .cache = &cache };
    const runner: TurnRunner = .{
        .ctx = &backend,
        .runTurnFn = MockBackend.runTurn,
        .onClearFn = MockBackend.onClear,
        .loadAttachmentFn = MockBackend.onLoadAttachment,
    };

    const cfg = try ChatConfig.parseArgs(&.{ "--max-tokens", "256", "--temperature", "0.2" });
    var session = try ChatSession.init(gpa, cfg);
    defer session.deinit(gpa);

    var out_cap: CaptureBuf = .{};
    var info_cap: CaptureBuf = .{};

    // 1. Queue `/image` + `/video` (`G14`) and run Turn 1.
    _ = try session.handleInteractiveLine("/image \"/tmp/frame0.png\"", runner, out_cap.sink(), info_cap.sink());
    _ = try session.handleInteractiveLine("/video /tmp/clip.mp4", runner, out_cap.sink(), info_cap.sink());
    try std.testing.expectEqual(@as(u32, 1), backend.loaded_images);
    try std.testing.expectEqual(@as(u32, 1), backend.loaded_videos);

    _ = try session.handleInteractiveLine("Describe media", runner, out_cap.sink(), info_cap.sink());
    try std.testing.expectEqual(@as(usize, 2), backend.last_attach_count);
    try std.testing.expectEqual(prefix_cache.NONE_ID, backend.last_parent_ckpt);
    const turn1_ckpt = session.activeCheckpointId();
    const turn1_full_blk = session.activeFullBlockId();
    try std.testing.expect(turn1_ckpt != prefix_cache.NONE_ID);
    try std.testing.expect(turn1_full_blk != prefix_cache.NONE_ID);
    try std.testing.expect((try cache.entryInfo(turn1_ckpt)).evict_first);
    try std.testing.expect(!(try cache.entryInfo(turn1_full_blk)).evict_first);

    // Verify NO alternate-screen sequence (`\x1b[?1049h`) was emitted, while ANSI dim `\x1b[2m` wrapped `<think>`
    // with clean `\n\n` visual separation before the answer!
    try std.testing.expect(std.mem.indexOf(u8, out_cap.written(), "\x1b[?1049h") == null);
    try std.testing.expect(std.mem.indexOf(u8, out_cap.written(), "\x1b[2mplan step 1\x1b[0m\n\nAnswer: Describe media\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, info_cap.written(), "tok/s | TTFT 8.5 ms | PrefixCache hit 16/64 tok") != null);

    // 2. Run Turn 2 -> must chain from `turn1_ckpt` / `turn1_full_blk` and strip Turn 1's `<think>` from ChatML history!
    out_cap.reset();
    info_cap.reset();
    _ = try session.handleInteractiveLine("Follow-up question", runner, out_cap.sink(), info_cap.sink());
    try std.testing.expectEqual(turn1_ckpt, backend.last_parent_ckpt);
    try std.testing.expectEqual(turn1_full_blk, backend.last_parent_full_blk);
    try std.testing.expect(!backend.last_prompt_had_think);
    try std.testing.expect(backend.last_boundary_offset != null);
    try std.testing.expect(std.mem.indexOf(u8, info_cap.written(), "PrefixCache hit 48/64 tok") != null);

    // 3. `/retry` rewinds Turn 2 back to `turn1_ckpt` / `turn1_full_blk` and re-runs "Follow-up question".
    out_cap.reset();
    _ = try session.handleInteractiveLine("/retry", runner, out_cap.sink(), info_cap.sink());
    try std.testing.expectEqual(turn1_ckpt, backend.last_parent_ckpt);
    try std.testing.expectEqual(turn1_full_blk, backend.last_parent_full_blk);
    try std.testing.expect(std.mem.indexOf(u8, out_cap.written(), "Answer: Follow-up question\n") != null);

    // 4. `/edit` rewinds Turn 2 back to `turn1_ckpt` / `turn1_full_blk` and runs the edited user message!
    out_cap.reset();
    _ = try session.handleInteractiveLine("/edit Updated question", runner, out_cap.sink(), info_cap.sink());
    try std.testing.expectEqual(turn1_ckpt, backend.last_parent_ckpt);
    try std.testing.expectEqual(turn1_full_blk, backend.last_parent_full_blk);
    try std.testing.expect(std.mem.indexOf(u8, out_cap.written(), "Answer: Updated question\n") != null);
    try std.testing.expectEqual(@as(usize, 2), session.turn_count);

    // 5. `/clear` resets history and active checkpoint to `NONE_ID`.
    _ = try session.handleInteractiveLine("/clear", runner, out_cap.sink(), info_cap.sink());
    try std.testing.expectEqual(@as(usize, 0), session.turn_count);
    try std.testing.expectEqual(prefix_cache.NONE_ID, session.activeCheckpointId());
    try std.testing.expectEqual(prefix_cache.NONE_ID, session.activeFullBlockId());
    try std.testing.expectEqual(@as(u32, 1), backend.cleared_count);
}

test "stdin pipe mode streams clean content to stdout and summary to stderr over real OS pipes" {
    const gpa = std.testing.allocator;

    var in_pipe: [2]std.c.fd_t = undefined;
    var out_pipe: [2]std.c.fd_t = undefined;
    var err_pipe: [2]std.c.fd_t = undefined;
    if (std.c.pipe(&in_pipe) != 0) return error.ReadFailed;
    defer if (in_pipe[0] >= 0) {
        _ = std.c.close(in_pipe[0]);
    };
    if (std.c.pipe(&out_pipe) != 0) return error.WriteFailed;
    defer {
        _ = std.c.close(out_pipe[0]);
    }
    if (std.c.pipe(&err_pipe) != 0) return error.WriteFailed;
    defer {
        _ = std.c.close(err_pipe[0]);
    }

    const piped_prompt = "/image /tmp/diagram.png\nExplain zero-copy IOSurface bridging\n";
    _ = std.c.write(in_pipe[1], piped_prompt.ptr, piped_prompt.len);
    _ = std.c.close(in_pipe[1]);

    const PipeRunner = struct {
        seen_attachments: usize = 0,

        fn runTurn(ctx: *anyopaque, req: *const TurnRequest, stream: *TurnStream) Error!TurnStats {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.seen_attachments = req.attachments.len;
            try stream.emitText("<think>hidden internal reasoning</think>\nUnified physical pages.");
            return .{
                .prompt_tokens = 32,
                .generated_tokens = 6,
                .prefix_hit_tokens = 0,
                .ttft_ns = 5_000_000,
                .total_ns = 50_000_000,
            };
        }
    };

    var pr: PipeRunner = .{};
    const cfg = try ChatConfig.parseArgs(&.{"--hide-think"});
    try runReplOrPipe(
        gpa,
        cfg,
        .{ .ctx = &pr, .runTurnFn = PipeRunner.runTurn },
        in_pipe[0],
        out_pipe[1],
        err_pipe[1],
        false,
    );
    _ = std.c.close(out_pipe[1]);
    _ = std.c.close(err_pipe[1]);

    var stdout_buf: [512]u8 = undefined;
    const out_n = std.c.read(out_pipe[0], &stdout_buf, stdout_buf.len);
    try std.testing.expect(out_n > 0);
    const stdout_str = stdout_buf[0..@intCast(out_n)];
    // `stdout` contains pure assistant output with zero `> ` prompt and zero `<think>` block!
    try std.testing.expectEqualStrings("Unified physical pages.\n", stdout_str);
    try std.testing.expectEqual(@as(usize, 1), pr.seen_attachments);

    var stderr_buf: [512]u8 = undefined;
    const err_n = std.c.read(err_pipe[0], &stderr_buf, stderr_buf.len);
    try std.testing.expect(err_n > 0);
    const stderr_str = stderr_buf[0..@intCast(err_n)];
    try std.testing.expect(std.mem.indexOf(u8, stderr_str, "120.0 tok/s") != null);
}

test "ChatConfig --reasoning-effort CLI parsing and multi-dialect Inkling 2 / Hy4 ChatSession turns" {
    const gpa = std.testing.allocator;

    // 1. `--reasoning-effort` parsing, numeric floats, `none` disable, and contradiction check.
    const c_med = try ChatConfig.parseArgs(&.{ "--reasoning-effort", "medium" });
    try std.testing.expectEqual(@as(?chat_template.ReasoningEffort, .medium), c_med.reasoning_effort);
    try std.testing.expect(c_med.enable_thinking);

    const c_num = try ChatConfig.parseArgs(&.{"--reasoning-effort=0.35"});
    try std.testing.expect(c_num.enable_thinking);
    try std.testing.expectApproxEqAbs(@as(f64, 0.35), c_num.reasoning_effort.?.numeric, 1e-6);

    const c_none = try ChatConfig.parseArgs(&.{ "--reasoning-effort", "none" });
    try std.testing.expect(!c_none.enable_thinking);

    const c_nothink_ok = try ChatConfig.parseArgs(&.{ "--no-think", "--reasoning-effort=no_think" });
    try std.testing.expect(!c_nothink_ok.enable_thinking);

    try std.testing.expectError(
        error.InvalidCliArgument,
        ChatConfig.parseArgs(&.{ "--no-think", "--reasoning-effort", "high" }),
    );
    try std.testing.expectError(
        error.InvalidCliArgument,
        ChatConfig.parseArgs(&.{ "-N", "--reasoning-effort=low" }),
    );
    try std.testing.expectError(
        error.InvalidCliArgument,
        ChatConfig.parseArgs(&.{ "--reasoning-effort", "turbo" }),
    );

    // 2. Inkling 2 dialect turn: prompt + `<|content_thinking|>..<|end_message|>` splitter
    const DialectRunner = struct {
        mode: enum { inkling2, hy4 },
        captured_prompt: [2048]u8 = undefined,
        captured_len: usize = 0,
        fn runTurn(ctx: *anyopaque, req: *const TurnRequest, stream: *TurnStream) Error!TurnStats {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(self.captured_prompt[0..req.chatml_prompt.len], req.chatml_prompt);
            self.captured_len = req.chatml_prompt.len;
            switch (self.mode) {
                .inkling2 => {
                    try stream.emitText("<|content_thinking|>");
                    try stream.emitText("plan");
                    try stream.emitText("<|end_message|>");
                    try stream.emitText("42");
                },
                .hy4 => {
                    try stream.emitText("step");
                    try stream.emitText("</think:opensource>");
                    try stream.emitText("Ok");
                },
            }
            return .{ .generated_tokens = 4, .prompt_tokens = 12, .ttft_ns = 1_000_000, .total_ns = 10_000_000 };
        }
    };

    const OutSink = struct {
        buf: [1024]u8 = undefined,
        len: usize = 0,
        fn byteSink(self: *@This()) ByteSink {
            return .{ .ctx = self, .writeFn = write };
        }
        fn write(ctx: *anyopaque, bytes: []const u8) error{WriteFailed}!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (self.len + bytes.len > self.buf.len) return error.WriteFailed;
            @memcpy(self.buf[self.len .. self.len + bytes.len], bytes);
            self.len += bytes.len;
        }
    };

    var ink_runner = DialectRunner{ .mode = .inkling2 };
    var ink_cfg = try ChatConfig.parseArgs(&.{ "--hide-think", "--reasoning-effort", "medium", "--no-stats" });
    ink_cfg.chat_template = chat_template.ChatTemplate.preset(.inkling2);
    ink_cfg.system_prompt = "Sys";
    var ink_sess = try ChatSession.init(gpa, ink_cfg);
    defer ink_sess.deinit(gpa);
    var ink_out = OutSink{};
    var ink_sum = OutSink{};
    try ink_sess.submitUserTurn("Hi", .{ .ctx = &ink_runner, .runTurnFn = DialectRunner.runTurn }, ink_out.byteSink(), ink_sum.byteSink());
    try std.testing.expectEqualStrings(
        "<|message_system|><|content_text|>Sys<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>Hi<|end_message|>",
        ink_runner.captured_prompt[0..ink_runner.captured_len],
    );
    try std.testing.expectEqualStrings("42\n", ink_out.buf[0..ink_out.len]);

    // 3. Hy4 dialect turn: prompt ends with `<think:opensource>` -> splitter starts in think!
    var hy_runner = DialectRunner{ .mode = .hy4 };
    var hy_cfg = try ChatConfig.parseArgs(&.{ "--hide-think", "--reasoning-effort", "low", "--no-stats" });
    hy_cfg.chat_template = chat_template.ChatTemplate.preset(.hy4);
    var hy_sess = try ChatSession.init(gpa, hy_cfg);
    defer hy_sess.deinit(gpa);
    var hy_out = OutSink{};
    var hy_sum = OutSink{};
    try hy_sess.submitUserTurn("Yo", .{ .ctx = &hy_runner, .runTurnFn = DialectRunner.runTurn }, hy_out.byteSink(), hy_sum.byteSink());
    try std.testing.expectEqualStrings(
        "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>You are a helpful assistant.<｜reasoning_mode:opensource｜>reasoning_effort:high<｜hy_end:opensource｜><｜hy_start:opensource｜>user<｜hy_middle:opensource｜>Yo<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>",
        hy_runner.captured_prompt[0..hy_runner.captured_len],
    );
    try std.testing.expectEqualStrings("Ok\n", hy_out.buf[0..hy_out.len]);
}

test "ChatConfig sampling defaults, greedy T=0 parity, and TurnRequest penalty forwarding" {
    const gpa = std.testing.allocator;

    const def_cfg = try ChatConfig.parseArgs(&.{});
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), def_cfg.temperature, 1e-6);
    try std.testing.expectEqual(@as(u32, 40), def_cfg.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), def_cfg.top_p, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), def_cfg.min_p, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.05), def_cfg.repetition_penalty, 1e-6);

    const greedy_cfg = try ChatConfig.parseArgs(&.{ "-t", "0" });
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), greedy_cfg.temperature, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), greedy_cfg.repetition_penalty, 1e-6);

    const greedy_rep_cfg = try ChatConfig.parseArgs(&.{ "-t", "0", "--repetition-penalty", "1.15" });
    try std.testing.expectApproxEqAbs(@as(f32, 1.15), greedy_rep_cfg.repetition_penalty, 1e-6);

    try std.testing.expectError(error.InvalidCliArgument, ChatConfig.parseArgs(&.{ "--top-p", "0.0" }));
    try std.testing.expectError(error.InvalidCliArgument, ChatConfig.parseArgs(&.{ "--min-p", "1.0" }));
    try std.testing.expectError(error.InvalidCliArgument, ChatConfig.parseArgs(&.{ "--repetition-penalty", "0.0" }));

    const CaptureReqRunner = struct {
        last_top_k: u32 = 0,
        last_top_p: f32 = 0,
        last_min_p: f32 = 0,
        last_rep: f32 = 0,
        last_pres: f32 = 0,
        last_freq: f32 = 0,
        fn runTurn(ctx: *anyopaque, req: *const TurnRequest, stream: *TurnStream) Error!TurnStats {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.last_top_k = req.top_k;
            self.last_top_p = req.top_p;
            self.last_min_p = req.min_p;
            self.last_rep = req.repetition_penalty;
            self.last_pres = req.presence_penalty;
            self.last_freq = req.frequency_penalty;
            try stream.emitText("Done.");
            return .{ .prompt_tokens = 4, .generated_tokens = 1 };
        }
    };
    var cap_runner = CaptureReqRunner{};
    const custom_cfg = try ChatConfig.parseArgs(&.{
        "--top-k=64",
        "--top-p=0.9",
        "--min-p=0.08",
        "--repetition-penalty=1.12",
        "--presence-penalty=0.4",
        "--frequency-penalty=0.25",
        "--no-stats",
    });
    var sess = try ChatSession.init(gpa, custom_cfg);
    defer sess.deinit(gpa);
    var out_b = CaptureBuf{};
    var sum_b = CaptureBuf{};
    try sess.submitUserTurn("test", .{ .ctx = &cap_runner, .runTurnFn = CaptureReqRunner.runTurn }, out_b.sink(), sum_b.sink());
    try std.testing.expectEqual(@as(u32, 64), cap_runner.last_top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), cap_runner.last_top_p, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.08), cap_runner.last_min_p, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.12), cap_runner.last_rep, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), cap_runner.last_pres, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), cap_runner.last_freq, 1e-6);
}

test "Inkling 2 TurnStream strips control tags, separates dimmed reasoning, and keeps clean history" {
    const gpa = std.testing.allocator;
    const tokens = [_][]const u8{
        "<|message_model|>",
        "<|content_thinking|>",
        "The user asks...",
        "<|end_message|>",
        "<|message_model|>",
        "<|content_text|>",
        "你好！我是 Inkling。",
        "<|end_message|>",
        "<|content_model_end_sampling|>",
    };

    const InklingRunner = struct {
        byte_by_byte: bool,
        fn runTurn(ctx: *anyopaque, _: *const TurnRequest, stream: *TurnStream) Error!TurnStats {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (tokens) |tok| {
                if (self.byte_by_byte) {
                    for (tok) |b| {
                        const one = [1]u8{b};
                        try stream.emitText(&one);
                    }
                } else {
                    try stream.emitText(tok);
                }
            }
            return .{ .prompt_tokens = 12, .generated_tokens = 9 };
        }
    };

    for ([_]bool{ false, true }) |byte_mode| {
        var runner_ctx = InklingRunner{ .byte_by_byte = byte_mode };
        var cfg = try ChatConfig.parseArgs(&.{"--no-stats"});
        cfg.chat_template = chat_template.ChatTemplate.preset(.inkling2);
        cfg.think_display = .dimmed;
        cfg.use_color = true;

        var sess = try ChatSession.init(gpa, cfg);
        defer sess.deinit(gpa);

        var out_b = CaptureBuf{};
        var sum_b = CaptureBuf{};
        try sess.submitUserTurn(
            "你好你是什么模型",
            .{ .ctx = &runner_ctx, .runTurnFn = InklingRunner.runTurn },
            out_b.sink(),
            sum_b.sink(),
        );

        const rendered = out_b.written();
        try std.testing.expect(std.mem.indexOf(u8, rendered, "<|") == null);
        try std.testing.expect(std.mem.indexOf(u8, rendered, "|>") == null);
        try std.testing.expectEqualStrings(
            "\x1b[2mThe user asks...\x1b[0m\n\n你好！我是 Inkling。\n",
            rendered,
        );

        // Session history must contain ONLY the clean assistant answer (no `\n\n`, no tags, no reasoning).
        try std.testing.expectEqual(@as(usize, 1), sess.turn_count);
        const turn0 = sess.turns[0];
        const recorded_assistant = sess.text_pool[turn0.assistant_off .. turn0.assistant_off + turn0.assistant_len];
        try std.testing.expectEqualStrings("你好！我是 Inkling。", recorded_assistant);
    }
}

test "InteractiveLineEditor handles CJK wide-char backspace, emoji, Ctrl-W, Ctrl-U, Ctrl-D, and CSI arrows" {
    // 1. Display width sanity checks
    try std.testing.expectEqual(@as(u2, 0), utf8CodepointWidth(0x00));
    try std.testing.expectEqual(@as(u2, 0), utf8CodepointWidth(0x200D));
    try std.testing.expectEqual(@as(u2, 0), utf8CodepointWidth(0xFE0F));
    try std.testing.expectEqual(@as(u2, 0), utf8CodepointWidth(0x0301));
    try std.testing.expectEqual(@as(u2, 1), utf8CodepointWidth('A'));
    try std.testing.expectEqual(@as(u2, 1), utf8CodepointWidth(0x303F));
    try std.testing.expectEqual(@as(u2, 2), utf8CodepointWidth(0x4F60)); // 你
    try std.testing.expectEqual(@as(u2, 2), utf8CodepointWidth(0x1F680)); // 🚀

    // 2. CJK wide-character Backspace: `"你好"` + `0x7f` -> `"你"` and `\x08\x08  \x08\x08`
    {
        var line_buf: [256]u8 = undefined;
        var editor: InteractiveLineEditor = .{ .line_buf = &line_buf };
        var out = CaptureBuf{};

        for ("你好") |b| {
            try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte(b, out.sink()));
        }
        try std.testing.expectEqualStrings("你好", editor.currentLine());

        try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte(0x7f, out.sink()));
        try std.testing.expectEqualStrings("你", editor.currentLine());
        try std.testing.expect(std.mem.endsWith(u8, out.written(), "\x08\x08  \x08\x08"));
        try std.testing.expectEqualStrings("你好\x08\x08  \x08\x08", out.written());

        // Backspace again wipes `"你"` and does not underflow on a 3rd Backspace.
        try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte(0x08, out.sink()));
        try std.testing.expectEqualStrings("", editor.currentLine());
        const len_after_second_bs = out.len;
        try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte(0x7f, out.sink()));
        try std.testing.expectEqual(len_after_second_bs, out.len);
    }

    // 3. Mixed ASCII + CJK + Emoji + arrow CSI/SS3 ignore + Ctrl-W + Ctrl-U + CRLF dedup
    {
        var line_buf: [256]u8 = undefined;
        var editor: InteractiveLineEditor = .{ .line_buf = &line_buf };
        var out = CaptureBuf{};

        // Type `"hi 你好🚀"`, inject arrow keys `\x1b[D`, `\x1b[1;5C`, `\x1bOD`, verify ignored.
        for ("hi \x1b[D你好\x1b[1;5C🚀\x1bOD") |b| {
            try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte(b, out.sink()));
        }
        try std.testing.expectEqualStrings("hi 你好🚀", editor.currentLine());
        try std.testing.expectEqualStrings("hi 你好🚀", out.written());

        // Backspace removes 4-byte 🚀 (2 display cols).
        out.reset();
        try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte(0x7f, out.sink()));
        try std.testing.expectEqualStrings("hi 你好", editor.currentLine());
        try std.testing.expectEqualStrings("\x08\x08  \x08\x08", out.written());

        // Append trailing spaces then Ctrl-W pops `"你好  "` (2 + 4 = 6 cols).
        for ("  ") |b| {
            _ = try editor.feedByte(b, out.sink());
        }
        out.reset();
        try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte(0x17, out.sink()));
        try std.testing.expectEqualStrings("hi ", editor.currentLine());
        try std.testing.expectEqualStrings("\x08\x08\x08\x08\x08\x08      \x08\x08\x08\x08\x08\x08", out.written());

        // Ctrl-U clears remaining `"hi "` (3 cols).
        out.reset();
        try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte(0x15, out.sink()));
        try std.testing.expectEqualStrings("", editor.currentLine());
        try std.testing.expectEqualStrings("\x08\x08\x08   \x08\x08\x08", out.written());

        // Type `"测试"`, submit with `\r\n`, ensure single line submission and `\n` is ignored.
        for ("测试") |b| {
            _ = try editor.feedByte(b, out.sink());
        }
        const cr_outcome = try editor.feedByte('\r', out.sink());
        switch (cr_outcome) {
            .line => |s| try std.testing.expectEqualStrings("测试", s),
            else => return error.TestUnexpectedResult,
        }
        try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.pending, try editor.feedByte('\n', out.sink()));

        // Ctrl-D on empty line returns `.eof`.
        try std.testing.expectEqual(InteractiveLineEditor.FeedOutcome.eof, try editor.feedByte(0x04, out.sink()));
    }

    // 4. Combining marks, ZWJ emoji cluster, and malformed UTF-8 fallback in `popLastGrapheme`
    {
        // `"e\u{0301}"` (3 bytes, 1 col)
        const comb = "ae\xcc\x81";
        const r1 = popLastGrapheme(comb);
        try std.testing.expectEqual(@as(usize, 1), r1.new_len);
        try std.testing.expectEqual(@as(usize, 1), r1.cols);

        // `"👨\u{200D}👩"` ZWJ cluster (11 bytes, 2 cols)
        const zwj = "x\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x91\xa9";
        const r2 = popLastGrapheme(zwj);
        try std.testing.expectEqual(@as(usize, 1), r2.new_len);
        try std.testing.expectEqual(@as(usize, 2), r2.cols);

        // Malformed UTF-8 trailing byte -> 1 byte, 1 col
        const bad = "a\xff";
        const r3 = popLastGrapheme(bad);
        try std.testing.expectEqual(@as(usize, 1), r3.new_len);
        try std.testing.expectEqual(@as(usize, 1), r3.cols);
    }
}

test "M9.5 chat_repl parses /lora commands and dispatches onLoraCommandFn" {
    const gpa = std.testing.allocator;

    const c0 = (try parseSlashCommand("/lora")).?;
    try std.testing.expectEqualStrings("", c0.lora);
    const c1 = (try parseSlashCommand("/lora \"unbound\"")).?;
    try std.testing.expectEqualStrings("unbound", c1.lora);
    const c2 = (try parseSlashCommand("/lora off")).?;
    try std.testing.expectEqualStrings("off", c2.lora);

    const LoraMock = struct {
        active: []const u8 = "base",
        fn runTurn(ctx: *anyopaque, req: *const TurnRequest, stream: *TurnStream) Error!TurnStats {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = req;
            try stream.emitText(self.active);
            return .{
                .prompt_tokens = 4,
                .generated_tokens = 1,
                .prefix_hit_tokens = 0,
                .ttft_ns = 1_000_000,
                .total_ns = 2_000_000,
            };
        }
        fn onLora(ctx: *anyopaque, arg: []const u8, info_sink: ByteSink) Error!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (arg.len == 0) {
                try info_sink.print("[LoRA active: {s}]\n", .{self.active});
            } else if (std.mem.eql(u8, arg, "off")) {
                self.active = "base";
                try info_sink.writeAll("[LoRA disabled (base model)]\n");
            } else {
                self.active = arg;
                try info_sink.print("[LoRA active: {s}]\n", .{arg});
            }
        }
    };

    var mock = LoraMock{};
    const runner: TurnRunner = .{
        .ctx = &mock,
        .runTurnFn = LoraMock.runTurn,
        .onLoraCommandFn = LoraMock.onLora,
    };

    const cfg = try ChatConfig.parseArgs(&.{"--no-stats"});
    var sess = try ChatSession.init(gpa, cfg);
    defer sess.deinit(gpa);

    var out_b = CaptureBuf{};
    var info_b = CaptureBuf{};
    try std.testing.expectEqual(LineOutcome.continued, try sess.handleInteractiveLine("/lora", runner, out_b.sink(), info_b.sink()));
    try std.testing.expectEqualStrings("[LoRA active: base]\n", info_b.written());

    info_b.reset();
    try std.testing.expectEqual(LineOutcome.continued, try sess.handleInteractiveLine("/lora unbound", runner, out_b.sink(), info_b.sink()));
    try std.testing.expectEqualStrings("[LoRA active: unbound]\n", info_b.written());

    try std.testing.expectEqual(LineOutcome.continued, try sess.handleInteractiveLine("hello", runner, out_b.sink(), info_b.sink()));
    try std.testing.expect(std.mem.indexOf(u8, out_b.written(), "unbound") != null);

    info_b.reset();
    try std.testing.expectEqual(LineOutcome.continued, try sess.handleInteractiveLine("/lora off", runner, out_b.sink(), info_b.sink()));
    try std.testing.expectEqualStrings("[LoRA disabled (base model)]\n", info_b.written());
    try std.testing.expectEqualStrings("base", mock.active);
}
