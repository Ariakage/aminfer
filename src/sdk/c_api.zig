//! Public Plain-C ABI (`libaminfer.dylib`) implementation (`ADR-0001`, `ADR-0022` `C17`/`C21`).
//!
//! Invariants:
//! - Zero Python or CPython headers/symbols linked.
//! - Zero swallowed errors; thread-local diagnostics via `ami_sdk_last_error()`.
//! - Zero dynamic heap allocation on steady-state tokenize, prefill, decode, view, embed, and rerank paths.
//! - Explicit `@autoreleasepool` draining and `WiredArena.deinit()` (`munlock`) teardown.
//! - Zero process-wide POSIX signal handler modifications; cancellation uses atomic `ami_sdk_cancel_token`.

const std = @import("std");
const aminfer = @import("aminfer");
const bridge_c = @import("apple_bridge");
const sdk_c = @import("aminfer_sdk_c");

const page = aminfer.page;
const budget_mod = aminfer.budget;
const arena_mod = aminfer.arena;
const sysinfo = aminfer.sysinfo;
const vmstat = aminfer.vmstat;
const metal = aminfer.metal;
const amf_mod = aminfer.amf;
const tokenizer_mod = aminfer.tokenizer;
const qwen3 = aminfer.qwen3;
const graph = aminfer.graph;
const vision_mod = aminfer.vision;
const attention = aminfer.attention;
const moe = aminfer.moe;
const rng_mod = aminfer.rng;

// ---------------------------------------------------------------------------
// Objective-C Runtime, CoreVideo & IOSurface C Declarations (macOS Frameworks)
// ---------------------------------------------------------------------------

extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;
extern "c" fn objc_getClass(name: [*:0]const u8) ?*anyopaque;
extern "c" fn sel_registerName(str: [*:0]const u8) ?*anyopaque;
extern "c" fn objc_msgSend() void;

extern "c" fn IOSurfaceLookup(csid: u32) ?*anyopaque;
extern "c" fn IOSurfaceGetID(buffer: *anyopaque) u32;
extern "c" fn IOSurfaceGetBaseAddress(buffer: *anyopaque) ?*anyopaque;
extern "c" fn IOSurfaceGetAllocSize(buffer: *anyopaque) usize;

extern "c" fn CVPixelBufferCreateWithIOSurface(
    allocator: ?*anyopaque,
    surface: *anyopaque,
    pixelBufferAttributes: ?*anyopaque,
    pixelBufferOut: *?*anyopaque,
) i32;
extern "c" fn CVPixelBufferRetain(texture: *anyopaque) ?*anyopaque;
extern "c" fn CVPixelBufferRelease(texture: *anyopaque) void;
extern "c" fn CVPixelBufferGetIOSurface(pixelBuffer: *anyopaque) ?*anyopaque;
extern "c" fn CVPixelBufferGetPixelFormatType(pixelBuffer: *anyopaque) u32;
extern "c" fn CVPixelBufferGetWidth(pixelBuffer: *anyopaque) usize;
extern "c" fn CVPixelBufferGetHeight(pixelBuffer: *anyopaque) usize;
extern "c" fn CVPixelBufferGetPlaneCount(pixelBuffer: *anyopaque) usize;
extern "c" fn CVPixelBufferGetBytesPerRow(pixelBuffer: *anyopaque) usize;
extern "c" fn CVPixelBufferGetWidthOfPlane(pixelBuffer: *anyopaque, planeIndex: usize) usize;
extern "c" fn CVPixelBufferGetHeightOfPlane(pixelBuffer: *anyopaque, planeIndex: usize) usize;
extern "c" fn CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer: *anyopaque, planeIndex: usize) usize;
extern "c" fn CVPixelBufferLockBaseAddress(pixelBuffer: *anyopaque, lockFlags: u64) i32;
extern "c" fn CVPixelBufferUnlockBaseAddress(pixelBuffer: *anyopaque, unlockFlags: u64) i32;
extern "c" fn CFRelease(cf: *anyopaque) void;

const CV_RETURN_SUCCESS: i32 = 0;
const CV_LOCK_READ_ONLY: u64 = 0x00000001;
const CV_FMT_BGRA8: u32 = 0x42475241; // 'BGRA'
const CV_FMT_NV12_FULL: u32 = 0x34323066; // '420f'
const CV_FMT_NV12_VIDEO: u32 = 0x34323076; // '420v'
const CV_FMT_R16F: u32 = 0x4C303068; // 'L00h'

const MTL_PIXEL_FORMAT_R8_UNORM: usize = 10;
const MTL_PIXEL_FORMAT_R16_FLOAT: usize = 25;
const MTL_PIXEL_FORMAT_RG8_UNORM: usize = 30;
const MTL_PIXEL_FORMAT_BGRA8_UNORM: usize = 80;
const MTL_STORAGE_MODE_SHARED: usize = 0;
const MTL_TEXTURE_USAGE_SHADER_READ: usize = 0x0001;

// Exact internal bridge handle layouts (`src/bridge/apple_bridge.m`) used solely to
// attach external `CVPixelBufferRef` / `IOSurfaceRef` handles zero-copy without
// touching `src/bridge/*`.
const AbiDeviceLayout = extern struct {
    mtl_device: ?*anyopaque,
};

const AbiVisionFrameLayout = extern struct {
    pixel_buffer: ?*anyopaque,
    surface: ?*anyopaque,
    mtl_textures: [2]?*anyopaque,
    format: u32,
    cv_pixel_format: u32,
    width: u32,
    height: u32,
    plane_count: u32,
    is_full_range: u32,
    zero_copy_verified: u32,
    is_page_aligned: u32,
    iosurface_id: u32,
    vt_decoded: u32,
    base_address: u64,
    alloc_size: u64,
    surface_alloc_size: u64,
    plane_width: [2]u32,
    plane_height: [2]u32,
    plane_bytes_per_row: [2]u32,
    pts_value: i64,
    pts_timescale: i32,
};

// ---------------------------------------------------------------------------
// Public C-ABI Constants, Status Codes & Struct Definitions
// ---------------------------------------------------------------------------

pub const ABI_VERSION: u32 = 1;
pub const MAX_STOP_TOKENS: usize = 8;
pub const TOKEN_NONE: u32 = 0xFFFF_FFFF;

pub const Status = i32;
pub const AMI_SDK_OK: Status = 0;
pub const AMI_SDK_ERR_INVALID_ARGUMENT: Status = -1;
pub const AMI_SDK_ERR_ABI_MISMATCH: Status = -2;
pub const AMI_SDK_ERR_OUT_OF_MEMORY: Status = -3;
pub const AMI_SDK_ERR_IO: Status = -4;
pub const AMI_SDK_ERR_FORMAT: Status = -5;
pub const AMI_SDK_ERR_BUDGET_EXCEEDED: Status = -6;
pub const AMI_SDK_ERR_LIVE_RAM_EXCEEDED: Status = -7;
pub const AMI_SDK_ERR_METAL: Status = -8;
pub const AMI_SDK_ERR_MODEL: Status = -9;
pub const AMI_SDK_ERR_TOKENIZER: Status = -10;
pub const AMI_SDK_ERR_BUFFER_TOO_SMALL: Status = -11;
pub const AMI_SDK_ERR_SEQUENCE_FULL: Status = -12;
pub const AMI_SDK_ERR_VISION: Status = -13;
pub const AMI_SDK_ERR_CANCELLED: Status = -14;
pub const AMI_SDK_ERR_UNSUPPORTED: Status = -15;

pub const DType = enum(u32) {
    f16 = 0,
    f32 = 1,
};

pub const Dialect = enum(u32) {
    qwen = 0,
    deepseek = 1,
};

pub const DeltaKind = enum(u32) {
    text = 0,
    reasoning = 1,
    tool_call = 2,
    tool_call_end = 3,
};

pub const FinishReason = enum(u32) {
    length = 0,
    stop = 1,
    tool_calls = 2,
    cancelled = 3,
    callback_stop = 4,
};

pub const PoolingMode = enum(u32) {
    last_token = 0,
    mean = 1,
    cls = 2,
};

pub const RerankMode = enum(u32) {
    logit_diff = 0,
    prob_yes = 1,
    raw_logit = 2,
};

pub const HostVmInfo = extern struct {
    struct_size: u32,
    reserved: u32,
    wired_bytes: u64,
    free_bytes: u64,
    live_reclaimable_bytes: u64,
    swapouts: u64,
    pageouts: u64,
    phys_footprint_bytes: u64,
};

pub const EngineOptions = extern struct {
    struct_size: u32,
    abi_version: u32,
    amf_path: ?[*:0]const u8,
    config_json_path: ?[*:0]const u8,
    tokenizer_json_path: ?[*:0]const u8,
    wired_budget_bytes: u64,
    headroom_bytes: u64,
    live_safety_margin_bytes: u64,
    live_reclaimable_override_bytes: u64,
    has_live_reclaimable_override: u32,
    strict_live_ram: u32,
    max_seq_len: u32,
    verify_crcs: u32,
    startup_canary: u32,
    reserved: u32,
};

pub const EngineStats = extern struct {
    struct_size: u32,
    is_graph_model: u32,
    vocab_size: u32,
    hidden_size: u32,
    num_layers: u32,
    max_seq_len: u32,
    seq_len: u32,
    reserved: u32,
    total_wired_bytes: u64,
    sdk_wired_bytes: u64,
    wired_budget_bytes: u64,
    live_reclaimable_bytes: u64,
    live_budget_bytes: u64,
};

pub const GenParams = extern struct {
    struct_size: u32,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    has_seed: u32,
    seed: u64,
    stop_token_ids: [MAX_STOP_TOKENS]u32,
    stop_token_count: u32,
    ignore_eos: u32,
    start_in_think: u32,
    reset_kv_cache: u32,
    skip_special_tokens: u32,
    reserved: u32,
    cancel_token: ?*const SdkCancelToken,
};

pub const StreamEvent = extern struct {
    struct_size: u32,
    token_id: u32,
    step_index: u32,
    delta_kind: u32,
    utf8_ptr: [*]const u8,
    utf8_len: usize,
    is_final_flush: u32,
    reserved: u32,
};

pub const StreamCallback = *const fn (event: *const StreamEvent, user_data: ?*anyopaque) callconv(.c) i32;

pub const GenResult = extern struct {
    struct_size: u32,
    prompt_tokens: u32,
    generated_tokens: u32,
    finish_reason: u32,
    seed_used: u64,
    prefill_ns: f64,
    decode_ns: f64,
};

pub const TensorView = extern struct {
    struct_size: u32,
    dtype: u32,
    data: ?*const anyopaque,
    byte_len: usize,
    ndim: u32,
    reserved: u32,
    shape: [2]usize,
};

pub const EmbedParams = extern struct {
    struct_size: u32,
    pooling: u32,
    dimensions: u32,
    normalize: u32,
    cancel_token: ?*const SdkCancelToken,
};

pub const RerankParams = extern struct {
    struct_size: u32,
    score_mode: u32,
    positive_token_id: u32,
    negative_token_id: u32,
    cancel_token: ?*const SdkCancelToken,
};

pub const VisionInfo = extern struct {
    struct_size: u32,
    format: u32,
    cv_pixel_format: u32,
    width: u32,
    height: u32,
    plane_count: u32,
    iosurface_id: u32,
    zero_copy_verified: u32,
    is_page_aligned: u32,
    reserved: u32,
    base_address: u64,
    surface_alloc_size: u64,
};

pub const VisionParams = extern struct {
    struct_size: u32,
    patch_size: u32,
    t_patch: u32,
    spatial_merge: u32,
    mean_rgb: [3]f32,
    inv_std_rgb: [3]f32,
    compute_logits: u32,
    reserved: u32,
    cancel_token: ?*const SdkCancelToken,
};

pub const VisionResult = extern struct {
    struct_size: u32,
    first_token_id: u32,
    prefix_tokens: u32,
    vision_tokens: u32,
    suffix_tokens: u32,
    total_prompt_tokens: u32,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    all_frames_zero_copy: u32,
    all_frames_page_aligned: u32,
    reserved: u32,
    vision_encode_us: f64,
};

comptime {
    std.debug.assert(ABI_VERSION == sdk_c.AMI_SDK_ABI_VERSION);
    std.debug.assert(@sizeOf(HostVmInfo) == @sizeOf(sdk_c.ami_sdk_host_vm_info));
    std.debug.assert(@sizeOf(EngineOptions) == @sizeOf(sdk_c.ami_sdk_engine_options));
    std.debug.assert(@sizeOf(EngineStats) == @sizeOf(sdk_c.ami_sdk_engine_stats));
    std.debug.assert(@sizeOf(GenParams) == @sizeOf(sdk_c.ami_sdk_gen_params));
    std.debug.assert(@sizeOf(StreamEvent) == @sizeOf(sdk_c.ami_sdk_stream_event));
    std.debug.assert(@sizeOf(GenResult) == @sizeOf(sdk_c.ami_sdk_gen_result));
    std.debug.assert(@sizeOf(TensorView) == @sizeOf(sdk_c.ami_sdk_tensor_view));
    std.debug.assert(@sizeOf(EmbedParams) == @sizeOf(sdk_c.ami_sdk_embed_params));
    std.debug.assert(@sizeOf(RerankParams) == @sizeOf(sdk_c.ami_sdk_rerank_params));
    std.debug.assert(@sizeOf(VisionInfo) == @sizeOf(sdk_c.ami_sdk_vision_info));
    std.debug.assert(@sizeOf(VisionParams) == @sizeOf(sdk_c.ami_sdk_vision_params));
    std.debug.assert(@sizeOf(VisionResult) == @sizeOf(sdk_c.ami_sdk_vision_result));
}

// ---------------------------------------------------------------------------
// Thread-Local Diagnostic State (Zero Heap Allocation)
// ---------------------------------------------------------------------------

threadlocal var tls_error_buf: [512:0]u8 = @splat(0);
threadlocal var tls_error_len: usize = 0;

fn clearLastError() void {
    tls_error_buf[0] = 0;
    tls_error_len = 0;
}

fn fail(status: Status, msg: []const u8) Status {
    const n = @min(msg.len, tls_error_buf.len - 1);
    @memcpy(tls_error_buf[0..n], msg[0..n]);
    tls_error_buf[n] = 0;
    tls_error_len = n;
    return status;
}

fn failFmt(status: Status, comptime fmt: []const u8, args: anytype) Status {
    const slice = std.fmt.bufPrint(&tls_error_buf, fmt, args) catch |err| switch (err) {
        error.NoSpaceLeft => blk: {
            tls_error_buf[tls_error_buf.len - 1] = 0;
            break :blk tls_error_buf[0 .. tls_error_buf.len - 1];
        },
    };
    tls_error_buf[slice.len] = 0;
    tls_error_len = slice.len;
    return status;
}

// ---------------------------------------------------------------------------
// Zero-Allocation Streaming UTF-8 Assembler & <think> / <tool_call> Classifier
// ---------------------------------------------------------------------------

pub const Utf8Assembler = struct {
    carry: [4]u8 = undefined,
    carry_len: u8 = 0,

    pub fn reset(self: *Utf8Assembler) void {
        self.carry_len = 0;
    }

    pub fn feed(
        self: *Utf8Assembler,
        bytes: []const u8,
        ctx: anytype,
        comptime emit_fn: fn (@TypeOf(ctx), []const u8) bool,
    ) bool {
        var input = bytes;
        if (self.carry_len > 0 and input.len > 0) {
            const need = expectedUtf8Len(self.carry[0]);
            var consumed: usize = 0;
            while (self.carry_len < need and consumed < input.len) {
                const b = input[consumed];
                if ((b & 0xC0) != 0x80) {
                    self.carry_len = 0;
                    if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                    break;
                }
                self.carry[self.carry_len] = b;
                self.carry_len += 1;
                consumed += 1;
            }
            if (self.carry_len > 0) {
                if (self.carry_len < need) return true;
                const completed = self.carry[0..need];
                self.carry_len = 0;
                if (isValidUtf8(completed)) {
                    if (!emit_fn(ctx, completed)) return false;
                } else {
                    if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                }
            }
            input = input[consumed..];
        }

        var valid_start: usize = 0;
        var pos: usize = 0;
        while (pos < input.len) {
            const b = input[pos];
            if (b < 0x80) {
                pos += 1;
                continue;
            }
            const seq_len = expectedUtf8Len(b);
            if (seq_len == 0) {
                if (pos > valid_start) {
                    if (!emit_fn(ctx, input[valid_start..pos])) return false;
                }
                if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                pos += 1;
                valid_start = pos;
                continue;
            }
            if (pos + seq_len > input.len) {
                const rem = input[pos..];
                var all_cont = true;
                for (rem[1..]) |cb| {
                    if ((cb & 0xC0) != 0x80) {
                        all_cont = false;
                        break;
                    }
                }
                if (pos > valid_start) {
                    if (!emit_fn(ctx, input[valid_start..pos])) return false;
                }
                if (all_cont) {
                    @memcpy(self.carry[0..rem.len], rem);
                    self.carry_len = @intCast(rem.len);
                    return true;
                }
                if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                pos += 1;
                valid_start = pos;
                continue;
            }
            if (isValidUtf8(input[pos .. pos + seq_len])) {
                pos += seq_len;
            } else {
                if (pos > valid_start) {
                    if (!emit_fn(ctx, input[valid_start..pos])) return false;
                }
                if (!emit_fn(ctx, "\xEF\xBF\xBD")) return false;
                pos += 1;
                valid_start = pos;
            }
        }
        if (pos > valid_start) {
            if (!emit_fn(ctx, input[valid_start..pos])) return false;
        }
        return true;
    }

    pub fn flush(
        self: *Utf8Assembler,
        ctx: anytype,
        comptime emit_fn: fn (@TypeOf(ctx), []const u8) bool,
    ) bool {
        if (self.carry_len == 0) return true;
        self.carry_len = 0;
        return emit_fn(ctx, "\xEF\xBF\xBD");
    }

    fn expectedUtf8Len(first: u8) u8 {
        if (first < 0x80) return 1;
        if (first >= 0xC2 and first <= 0xDF) return 2;
        if (first >= 0xE0 and first <= 0xEF) return 3;
        if (first >= 0xF0 and first <= 0xF4) return 4;
        return 0;
    }

    fn isValidUtf8(seq: []const u8) bool {
        _ = std.unicode.utf8Decode(seq) catch return false;
        return true;
    }
};

pub const StreamTagClassifier = struct {
    const THINK_OPEN = "<think>";
    const THINK_CLOSE = "</think>";
    const TOOL_OPEN = "<tool_call>";
    const TOOL_CLOSE = "</tool_call>";

    const Mode = enum(u8) {
        text = 0,
        reasoning = 1,
        tool_call = 2,
    };

    mode: Mode = .text,
    pending: [TOOL_CLOSE.len]u8 = undefined,
    pending_len: u8 = 0,
    saw_tool_call: bool = false,

    pub fn reset(self: *StreamTagClassifier, start_in_think: bool) void {
        self.mode = if (start_in_think) .reasoning else .text;
        self.pending_len = 0;
        self.saw_tool_call = false;
    }

    fn currentDeltaKind(mode: Mode) DeltaKind {
        return switch (mode) {
            .text => .text,
            .reasoning => .reasoning,
            .tool_call => .tool_call,
        };
    }

    pub fn feed(
        self: *StreamTagClassifier,
        chunk: []const u8,
        ctx: anytype,
        comptime emit_fn: fn (@TypeOf(ctx), DeltaKind, []const u8) bool,
    ) bool {
        var i: usize = 0;
        while (i < chunk.len) {
            if (self.pending_len == 0) {
                if (std.mem.indexOfScalarPos(u8, chunk, i, '<')) |lt_pos| {
                    if (lt_pos > i) {
                        if (!emit_fn(ctx, currentDeltaKind(self.mode), chunk[i..lt_pos])) return false;
                    }
                    self.pending[0] = '<';
                    self.pending_len = 1;
                    i = lt_pos + 1;
                } else {
                    if (i < chunk.len) {
                        if (!emit_fn(ctx, currentDeltaKind(self.mode), chunk[i..])) return false;
                    }
                    return true;
                }
                continue;
            }

            const b = chunk[i];
            switch (self.mode) {
                .text => {
                    const plen: usize = self.pending_len;
                    const match_think = plen < THINK_OPEN.len and
                        std.mem.eql(u8, self.pending[0..plen], THINK_OPEN[0..plen]) and
                        b == THINK_OPEN[plen];
                    const match_tool = plen < TOOL_OPEN.len and
                        std.mem.eql(u8, self.pending[0..plen], TOOL_OPEN[0..plen]) and
                        b == TOOL_OPEN[plen];

                    if (match_think or match_tool) {
                        self.pending[plen] = b;
                        self.pending_len = @intCast(plen + 1);
                        i += 1;
                        if (self.pending_len == THINK_OPEN.len and
                            std.mem.eql(u8, self.pending[0..THINK_OPEN.len], THINK_OPEN))
                        {
                            self.pending_len = 0;
                            self.mode = .reasoning;
                        } else if (self.pending_len == TOOL_OPEN.len and
                            std.mem.eql(u8, self.pending[0..TOOL_OPEN.len], TOOL_OPEN))
                        {
                            self.pending_len = 0;
                            self.mode = .tool_call;
                            self.saw_tool_call = true;
                        }
                    } else {
                        const flushed = self.pending[0..self.pending_len];
                        self.pending_len = 0;
                        if (!emit_fn(ctx, .text, flushed)) return false;
                    }
                },
                .reasoning => {
                    const plen: usize = self.pending_len;
                    if (plen < THINK_CLOSE.len and b == THINK_CLOSE[plen]) {
                        self.pending[plen] = b;
                        self.pending_len = @intCast(plen + 1);
                        i += 1;
                        if (self.pending_len == THINK_CLOSE.len) {
                            self.pending_len = 0;
                            self.mode = .text;
                        }
                    } else {
                        const flushed = self.pending[0..self.pending_len];
                        self.pending_len = 0;
                        if (!emit_fn(ctx, .reasoning, flushed)) return false;
                    }
                },
                .tool_call => {
                    const plen: usize = self.pending_len;
                    if (plen < TOOL_CLOSE.len and b == TOOL_CLOSE[plen]) {
                        self.pending[plen] = b;
                        self.pending_len = @intCast(plen + 1);
                        i += 1;
                        if (self.pending_len == TOOL_CLOSE.len) {
                            self.pending_len = 0;
                            self.mode = .text;
                            if (!emit_fn(ctx, .tool_call_end, "")) return false;
                        }
                    } else {
                        const flushed = self.pending[0..self.pending_len];
                        self.pending_len = 0;
                        if (!emit_fn(ctx, .tool_call, flushed)) return false;
                    }
                },
            }
        }
        return true;
    }

    pub fn flush(
        self: *StreamTagClassifier,
        ctx: anytype,
        comptime emit_fn: fn (@TypeOf(ctx), DeltaKind, []const u8) bool,
    ) bool {
        if (self.pending_len == 0) return true;
        const kind = currentDeltaKind(self.mode);
        const flushed = self.pending[0..self.pending_len];
        self.pending_len = 0;
        return emit_fn(ctx, kind, flushed);
    }
};

// ---------------------------------------------------------------------------
// Opaque Handle Types: CancelToken, Tokenizer, VisionInput, Engine
// ---------------------------------------------------------------------------

const gpa = std.heap.c_allocator;

pub const SdkCancelToken = struct {
    cancelled: std.atomic.Value(u8) = .init(0),

    pub fn isCancelled(self: ?*const SdkCancelToken) bool {
        const tok = self orelse return false;
        return tok.cancelled.load(.acquire) != 0;
    }
};

const TOK_SCRATCH_CHUNK_BYTES: usize = 64 * 1024;
const TOK_COUNT_SCRATCH_CAP: usize = 16 * 1024;

pub const SdkTokenizer = struct {
    tok: tokenizer_mod.Tokenizer,
    scratch: tokenizer_mod.EncodeScratch,
    count_buf: []u32,
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    fn createFromBytes(allocator: std.mem.Allocator, json_bytes: []const u8) !*SdkTokenizer {
        var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, json_bytes);
        errdefer tok.deinit(allocator);

        var scratch = try tokenizer_mod.EncodeScratch.init(allocator, TOK_SCRATCH_CHUNK_BYTES);
        errdefer scratch.deinit(allocator);

        const count_buf = try allocator.alloc(u32, TOK_COUNT_SCRATCH_CAP);
        errdefer allocator.free(count_buf);

        const handle = try allocator.create(SdkTokenizer);
        handle.* = .{
            .tok = tok,
            .scratch = scratch,
            .count_buf = count_buf,
        };
        return handle;
    }

    fn destroy(self: *SdkTokenizer, allocator: std.mem.Allocator) void {
        _ = std.c.pthread_mutex_destroy(&self.mutex);
        allocator.free(self.count_buf);
        self.scratch.deinit(allocator);
        self.tok.deinit(allocator);
        allocator.destroy(self);
    }

    fn encodeInto(
        self: *SdkTokenizer,
        text: []const u8,
        allow_special: bool,
        out_ids: ?[]u32,
    ) (tokenizer_mod.EncodeError || error{OutOfMemory})!usize {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        defer _ = std.c.pthread_mutex_unlock(&self.mutex);

        var view_tok = self.tok;
        if (!allow_special) {
            view_tok.added_tokens = &.{};
        }

        if (out_ids) |dst| {
            if (text.len <= TOK_SCRATCH_CHUNK_BYTES) {
                const written = view_tok.encodeWithScratch(&self.scratch, text, dst) catch |err| switch (err) {
                    error.BufferTooSmall => {
                        return try self.encodeCountLocked(&view_tok, text);
                    },
                    else => return err,
                };
                return written.len;
            }
            const written = view_tok.encode(gpa, text, dst) catch |err| switch (err) {
                error.BufferTooSmall => return try self.encodeCountLocked(&view_tok, text),
                else => return err,
            };
            return written.len;
        }
        return self.encodeCountLocked(&view_tok, text);
    }

    fn encodeCountLocked(
        self: *SdkTokenizer,
        view_tok: *const tokenizer_mod.Tokenizer,
        text: []const u8,
    ) (tokenizer_mod.EncodeError || error{OutOfMemory})!usize {
        if (text.len <= TOK_COUNT_SCRATCH_CAP and text.len <= TOK_SCRATCH_CHUNK_BYTES) {
            const written = try view_tok.encodeWithScratch(&self.scratch, text, self.count_buf);
            return written.len;
        }
        const owned = try view_tok.encodeAlloc(gpa, text);
        defer gpa.free(owned);
        return owned.len;
    }
};

pub const SdkVisionInput = struct {
    frame: metal.VisionFrame,
};

const ModelBackend = union(enum) {
    qwen3: qwen3.Qwen3Model,
    graph: graph.GraphModel,
};

const RERANK_PROMPT_CAP: usize = 64 * 1024;

pub const SdkEngine = struct {
    backend: ModelBackend,
    graph_cfg: ?graph.GraphConfig,
    owned_tokenizer: ?*SdkTokenizer,

    vocab_size: u32,
    hidden_size: u32,
    num_layers: u32,
    max_seq_len: u32,
    eos_token_id: u32,
    wired_budget_bytes: u64,
    live_reclaimable_bytes: u64,
    live_budget_bytes: u64,

    // Preallocated 16 KiB-aligned `mlock`ed SDK workspace (0 steady-state heap allocs).
    sdk_arena: arena_mod.WiredArena,
    logits_f32: []f32,
    hidden_norm_f16: []f16,
    hidden_norm_f32: []f32,
    hidden_raw_f32: []f32,
    sample_probs: []f32,
    sample_indices: []u32,
    token_counts: []u32,
    history_ids: []u32,
    prompt_ids_scratch: []u32,
    rerank_buf: []u8,
    embed_accum_f32: []f32,

    history_len: u32 = 0,
    has_step_state: bool = false,
    logits_f32_valid: bool = false,
    hidden_norm_f16_valid: bool = false,
    hidden_norm_f32_valid: bool = false,
    hidden_raw_f32_valid: bool = false,

    fn invalidateViews(self: *SdkEngine) void {
        self.logits_f32_valid = false;
        self.hidden_norm_f16_valid = false;
        self.hidden_norm_f32_valid = false;
        self.hidden_raw_f32_valid = false;
    }

    fn currentSeqLen(self: *const SdkEngine) u32 {
        return switch (self.backend) {
            .qwen3 => |*m| m.seq_len,
            .graph => |*m| m.seq_len,
        };
    }

    fn totalModelWiredBytes(self: *const SdkEngine) u64 {
        return switch (self.backend) {
            .qwen3 => |*m| m.totalWiredBytes(),
            .graph => |*m| m.totalWiredBytes(),
        };
    }

    fn metalDevice(self: *const SdkEngine) metal.Device {
        return switch (self.backend) {
            .qwen3 => |*m| m.device,
            .graph => |*m| m.device,
        };
    }

    fn resetKv(self: *SdkEngine) void {
        switch (self.backend) {
            .qwen3 => |*m| m.resetKvCache(),
            .graph => |*m| m.resetKvCache(),
        }
        self.history_len = 0;
        self.has_step_state = false;
        self.invalidateViews();
    }

    fn lastLogitsF16(self: *const SdkEngine) []const f16 {
        return switch (self.backend) {
            .qwen3 => |*m| m.readLastLogitsF16(),
            .graph => |*m| m.readLastLogitsF16(),
        };
    }

    fn lastTopHiddenF16(self: *const SdkEngine) []const f16 {
        return switch (self.backend) {
            .qwen3 => |*m| blk: {
                if (m.usesWideGpuPrefill() and m.last_logits_row == 0 and m.seq_len > 0) {
                    // For single-token stepDecode `x_host[0..h]` is written; for wide prefill,
                    // track whichever buffer was most recently populated via `readTopHiddenF16(1)`
                    // or `wp_x_bind` captured into `hidden_norm_f16`.
                }
                break :blk m.readTopHiddenF16(1);
            },
            .graph => |*m| m.readTopHiddenF16(1),
        };
    }

    fn recordToken(self: *SdkEngine, tok: u32) void {
        if (self.history_len < self.max_seq_len) {
            self.history_ids[self.history_len] = tok;
            self.history_len += 1;
        }
    }

    fn destroy(self: *SdkEngine, allocator: std.mem.Allocator) void {
        if (self.owned_tokenizer) |t| {
            t.destroy(allocator);
            self.owned_tokenizer = null;
        }
        self.sdk_arena.deinit();
        switch (self.backend) {
            .qwen3 => |*m| m.deinit(allocator),
            .graph => |*m| m.deinit(allocator),
        }
        if (self.graph_cfg) |*gcfg| {
            gcfg.deinit(allocator);
            self.graph_cfg = null;
        }
        allocator.destroy(self);
    }
};

// ---------------------------------------------------------------------------
// POSIX File Helpers (Zig 0.17.1 `std.c.open` / `fstat` / `read`)
// ---------------------------------------------------------------------------

const FileIoError = error{
    FileOpenFailed,
    FileStatFailed,
    InvalidFileSize,
    FileTooLarge,
    FileReadFailed,
    UnexpectedEof,
    OutOfMemory,
};

fn fileExistsPosix(path: [:0]const u8) bool {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return false;
    _ = std.c.close(fd);
    return true;
}

fn readFileAllocPosix(allocator: std.mem.Allocator, path: [:0]const u8, max_bytes: usize) FileIoError![]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileOpenFailed;
    defer _ = std.c.close(fd);

    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.FileStatFailed;
    if (st.size < 0) return error.InvalidFileSize;
    const size: usize = @intCast(st.size);
    if (size > max_bytes) return error.FileTooLarge;

    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);

    var done: usize = 0;
    while (done < size) {
        const rc = std.c.read(fd, buf.ptr + done, size - done);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.FileReadFailed;
        }
        if (rc == 0) return error.UnexpectedEof;
        done += @intCast(rc);
    }
    return buf;
}

fn buildSiblingPathZ(
    allocator: std.mem.Allocator,
    amf_path: []const u8,
    leaf: []const u8,
) error{OutOfMemory}![:0]u8 {
    const dir = if (std.mem.lastIndexOfScalar(u8, amf_path, '/')) |slash|
        amf_path[0 .. slash + 1]
    else
        "";
    const out = try allocator.allocSentinel(u8, dir.len + leaf.len, 0);
    @memcpy(out[0..dir.len], dir);
    @memcpy(out[dir.len .. dir.len + leaf.len], leaf);
    return out;
}

fn mapAnyError(err: anyerror) Status {
    const name = @errorName(err);
    var status: Status = AMI_SDK_ERR_MODEL;
    if (std.mem.eql(u8, name, "BudgetExceeded")) {
        status = AMI_SDK_ERR_BUDGET_EXCEEDED;
    } else if (std.mem.eql(u8, name, "ExceedsLiveZeroSwapRam")) {
        status = AMI_SDK_ERR_LIVE_RAM_EXCEEDED;
    } else if (std.mem.eql(u8, name, "OutOfMemory")) {
        status = AMI_SDK_ERR_OUT_OF_MEMORY;
    } else if (std.mem.eql(u8, name, "FileOpenFailed") or
        std.mem.eql(u8, name, "FileStatFailed") or
        std.mem.eql(u8, name, "FileReadFailed") or
        std.mem.eql(u8, name, "UnexpectedEof") or
        std.mem.eql(u8, name, "OpenFailed") or
        std.mem.eql(u8, name, "StatFailed") or
        std.mem.eql(u8, name, "PreadFailed") or
        std.mem.eql(u8, name, "MetadataNotFound"))
    {
        status = AMI_SDK_ERR_IO;
    } else if (std.mem.eql(u8, name, "BadMagic") or
        std.mem.eql(u8, name, "UnsupportedVersion") or
        std.mem.eql(u8, name, "ChecksumMismatch") or
        std.mem.eql(u8, name, "TruncatedFile") or
        std.mem.eql(u8, name, "MissingTensor"))
    {
        status = AMI_SDK_ERR_FORMAT;
    } else if (std.mem.eql(u8, name, "PositionOutOfRange") or
        std.mem.eql(u8, name, "SequenceFull"))
    {
        status = AMI_SDK_ERR_SEQUENCE_FULL;
    } else if (std.mem.eql(u8, name, "TokenOutOfRange") or
        std.mem.eql(u8, name, "InvalidArgument") or
        std.mem.eql(u8, name, "InvalidBatchSize"))
    {
        status = AMI_SDK_ERR_INVALID_ARGUMENT;
    } else if (std.mem.eql(u8, name, "VisionFailed") or
        std.mem.eql(u8, name, "UnsupportedVisionFormat") or
        std.mem.eql(u8, name, "EmptyFrames") or
        std.mem.eql(u8, name, "FrameGeometryMismatch"))
    {
        status = AMI_SDK_ERR_VISION;
    }
    return failFmt(status, "aminfer error: {s}", .{name});
}

// ---------------------------------------------------------------------------
// ABI Version, Status Strings, Thread-Local Diagnostics & Struct Initializers
// ---------------------------------------------------------------------------

pub export fn ami_sdk_abi_version() callconv(.c) u32 {
    return ABI_VERSION;
}

pub export fn ami_sdk_version_string() callconv(.c) [*:0]const u8 {
    return "0.7.5";
}

pub export fn ami_sdk_status_string(status: Status) callconv(.c) [*:0]const u8 {
    return switch (status) {
        AMI_SDK_OK => "AMI_SDK_OK",
        AMI_SDK_ERR_INVALID_ARGUMENT => "AMI_SDK_ERR_INVALID_ARGUMENT",
        AMI_SDK_ERR_ABI_MISMATCH => "AMI_SDK_ERR_ABI_MISMATCH",
        AMI_SDK_ERR_OUT_OF_MEMORY => "AMI_SDK_ERR_OUT_OF_MEMORY",
        AMI_SDK_ERR_IO => "AMI_SDK_ERR_IO",
        AMI_SDK_ERR_FORMAT => "AMI_SDK_ERR_FORMAT",
        AMI_SDK_ERR_BUDGET_EXCEEDED => "AMI_SDK_ERR_BUDGET_EXCEEDED",
        AMI_SDK_ERR_LIVE_RAM_EXCEEDED => "AMI_SDK_ERR_LIVE_RAM_EXCEEDED",
        AMI_SDK_ERR_METAL => "AMI_SDK_ERR_METAL",
        AMI_SDK_ERR_MODEL => "AMI_SDK_ERR_MODEL",
        AMI_SDK_ERR_TOKENIZER => "AMI_SDK_ERR_TOKENIZER",
        AMI_SDK_ERR_BUFFER_TOO_SMALL => "AMI_SDK_ERR_BUFFER_TOO_SMALL",
        AMI_SDK_ERR_SEQUENCE_FULL => "AMI_SDK_ERR_SEQUENCE_FULL",
        AMI_SDK_ERR_VISION => "AMI_SDK_ERR_VISION",
        AMI_SDK_ERR_CANCELLED => "AMI_SDK_ERR_CANCELLED",
        AMI_SDK_ERR_UNSUPPORTED => "AMI_SDK_ERR_UNSUPPORTED",
        else => "AMI_SDK_ERR_UNKNOWN",
    };
}

pub export fn ami_sdk_last_error() callconv(.c) [*:0]const u8 {
    return &tls_error_buf;
}

pub export fn ami_sdk_clear_last_error() callconv(.c) void {
    clearLastError();
}

pub export fn ami_sdk_host_vm_snapshot(out_info: ?*HostVmInfo) callconv(.c) Status {
    const info = out_info orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_info is NULL");
    if (info.struct_size != 0 and info.struct_size < @sizeOf(HostVmInfo)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "HostVmInfo struct_size mismatch");
    }
    const snap = vmstat.Snapshot.capture() catch |err| return mapAnyError(err);
    info.* = .{
        .struct_size = @sizeOf(HostVmInfo),
        .reserved = 0,
        .wired_bytes = snap.wire_pages * snap.page_size,
        .free_bytes = snap.free_pages * snap.page_size,
        .live_reclaimable_bytes = snap.liveZeroSwapReclaimableBytes(),
        .swapouts = snap.swapouts,
        .pageouts = snap.pageouts,
        .phys_footprint_bytes = snap.heap.size_in_use,
    };
    return AMI_SDK_OK;
}

pub export fn ami_sdk_engine_options_init(opts_ptr: ?*EngineOptions) callconv(.c) void {
    const opts = opts_ptr orelse return;
    opts.* = .{
        .struct_size = @sizeOf(EngineOptions),
        .abi_version = ABI_VERSION,
        .amf_path = null,
        .config_json_path = null,
        .tokenizer_json_path = null,
        .wired_budget_bytes = 0,
        .headroom_bytes = 0,
        .live_safety_margin_bytes = 0,
        .live_reclaimable_override_bytes = 0,
        .has_live_reclaimable_override = 0,
        .strict_live_ram = 1,
        .max_seq_len = 2048,
        .verify_crcs = 1,
        .startup_canary = 1,
        .reserved = 0,
    };
}

pub export fn ami_sdk_gen_params_init(params_ptr: ?*GenParams) callconv(.c) void {
    const params = params_ptr orelse return;
    params.* = .{
        .struct_size = @sizeOf(GenParams),
        .max_tokens = 64,
        .temperature = 0.0,
        .top_p = 1.0,
        .top_k = 0,
        .min_p = 0.0,
        .repetition_penalty = 1.0,
        .presence_penalty = 0.0,
        .frequency_penalty = 0.0,
        .has_seed = 0,
        .seed = 0,
        .stop_token_ids = @splat(0),
        .stop_token_count = 0,
        .ignore_eos = 0,
        .start_in_think = 0,
        .reset_kv_cache = 1,
        .skip_special_tokens = 1,
        .reserved = 0,
        .cancel_token = null,
    };
}

pub export fn ami_sdk_embed_params_init(params_ptr: ?*EmbedParams) callconv(.c) void {
    const params = params_ptr orelse return;
    params.* = .{
        .struct_size = @sizeOf(EmbedParams),
        .pooling = @backingInt(PoolingMode.last_token),
        .dimensions = 0,
        .normalize = 1,
        .cancel_token = null,
    };
}

pub export fn ami_sdk_rerank_params_init(params_ptr: ?*RerankParams) callconv(.c) void {
    const params = params_ptr orelse return;
    params.* = .{
        .struct_size = @sizeOf(RerankParams),
        .score_mode = @backingInt(RerankMode.logit_diff),
        .positive_token_id = TOKEN_NONE,
        .negative_token_id = TOKEN_NONE,
        .cancel_token = null,
    };
}

pub export fn ami_sdk_vision_params_init(params_ptr: ?*VisionParams) callconv(.c) void {
    const params = params_ptr orelse return;
    params.* = .{
        .struct_size = @sizeOf(VisionParams),
        .patch_size = 16,
        .t_patch = 1,
        .spatial_merge = 2,
        .mean_rgb = .{ 0.48145466, 0.4578275, 0.40821073 },
        .inv_std_rgb = .{ 1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711 },
        .compute_logits = 1,
        .reserved = 0,
        .cancel_token = null,
    };
}

// ---------------------------------------------------------------------------
// Cooperative Cancellation Token (0 POSIX Signal Handler Modifications)
// ---------------------------------------------------------------------------

pub export fn ami_sdk_cancel_token_create(out_token: ?*?*SdkCancelToken) callconv(.c) Status {
    const out = out_token orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_token is NULL");
    out.* = null;
    const tok = gpa.create(SdkCancelToken) catch return fail(AMI_SDK_ERR_OUT_OF_MEMORY, "cancel token alloc failed");
    tok.* = .{};
    out.* = tok;
    return AMI_SDK_OK;
}

pub export fn ami_sdk_cancel_token_destroy(token: ?*SdkCancelToken) callconv(.c) void {
    const tok = token orelse return;
    gpa.destroy(tok);
}

pub export fn ami_sdk_cancel_token_cancel(token: ?*SdkCancelToken) callconv(.c) void {
    const tok = token orelse return;
    tok.cancelled.store(1, .release);
}

pub export fn ami_sdk_cancel_token_reset(token: ?*SdkCancelToken) callconv(.c) void {
    const tok = token orelse return;
    tok.cancelled.store(0, .release);
}

pub export fn ami_sdk_cancel_token_is_cancelled(token: ?*const SdkCancelToken) callconv(.c) u32 {
    return if (SdkCancelToken.isCancelled(token)) 1 else 0;
}

// ---------------------------------------------------------------------------
// Native BPE Tokenizer Entry Points (`W5`: Qwen & DeepSeek Dialects)
// ---------------------------------------------------------------------------

pub export fn ami_sdk_tokenizer_from_file(
    path_ptr: ?[*:0]const u8,
    out_tokenizer: ?*?*SdkTokenizer,
) callconv(.c) Status {
    const out = out_tokenizer orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_tokenizer is NULL");
    out.* = null;
    const raw_path = path_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "path is NULL");
    const path = std.mem.span(raw_path);
    if (path.len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "path is empty");

    if (std.mem.endsWith(u8, path, ".amf")) {
        var amf_file = amf_mod.AmfFile.open(gpa, path) catch |err| return mapAnyError(err);
        defer amf_file.close(gpa);
        var resolved_tok = amf_mod.resolveTokenizerJsonAlloc(gpa, &amf_file, path, null) catch |err| return mapAnyError(err);
        defer resolved_tok.deinit(gpa);
        const handle = SdkTokenizer.createFromBytes(gpa, resolved_tok.bytes) catch |err| {
            return failFmt(AMI_SDK_ERR_TOKENIZER, "tokenizer load failed: {s}", .{@errorName(err)});
        };
        out.* = handle;
        return AMI_SDK_OK;
    }

    const json_bytes = readFileAllocPosix(gpa, path, 64 * 1024 * 1024) catch |err| return mapAnyError(err);
    defer gpa.free(json_bytes);

    const handle = SdkTokenizer.createFromBytes(gpa, json_bytes) catch |err| {
        return failFmt(AMI_SDK_ERR_TOKENIZER, "tokenizer load failed: {s}", .{@errorName(err)});
    };
    out.* = handle;
    return AMI_SDK_OK;
}

pub export fn ami_sdk_tokenizer_from_json(
    json_ptr: ?[*]const u8,
    json_len: usize,
    out_tokenizer: ?*?*SdkTokenizer,
) callconv(.c) Status {
    const out = out_tokenizer orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_tokenizer is NULL");
    out.* = null;
    if (json_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "json_len is 0");
    const ptr = json_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "json_ptr is NULL");

    const handle = SdkTokenizer.createFromBytes(gpa, ptr[0..json_len]) catch |err| {
        return failFmt(AMI_SDK_ERR_TOKENIZER, "tokenizer parse failed: {s}", .{@errorName(err)});
    };
    out.* = handle;
    return AMI_SDK_OK;
}

pub export fn ami_sdk_tokenizer_destroy(tokenizer: ?*SdkTokenizer) callconv(.c) void {
    const tok = tokenizer orelse return;
    tok.destroy(gpa);
}

pub export fn ami_sdk_tokenizer_vocab_size(tokenizer: ?*const SdkTokenizer) callconv(.c) u32 {
    const tok = tokenizer orelse return 0;
    return @intCast(tok.tok.vocabSize());
}

pub export fn ami_sdk_tokenizer_dialect(tokenizer: ?*const SdkTokenizer) callconv(.c) u32 {
    const tok = tokenizer orelse return @backingInt(Dialect.qwen);
    return switch (tok.tok.pre_tokenizer_kind) {
        .qwen => @backingInt(Dialect.qwen),
        .deepseek => @backingInt(Dialect.deepseek),
    };
}

pub export fn ami_sdk_tokenizer_special_id(
    tokenizer: ?*const SdkTokenizer,
    token_str: ?[*]const u8,
    token_len: usize,
    out_id: ?*u32,
) callconv(.c) Status {
    const tok = tokenizer orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "tokenizer is NULL");
    const out = out_id orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_id is NULL");
    if (token_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "token_len is 0");
    const ptr = token_str orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "token_str is NULL");
    if (tok.tok.addedTokenId(ptr[0..token_len])) |id| {
        out.* = id;
        return AMI_SDK_OK;
    }
    out.* = TOKEN_NONE;
    return fail(AMI_SDK_ERR_TOKENIZER, "special token not found");
}

pub export fn ami_sdk_tokenize(
    tokenizer: ?*SdkTokenizer,
    text_ptr: ?[*]const u8,
    text_len: usize,
    allow_special_tokens: u32,
    out_ids: ?[*]u32,
    out_cap: usize,
    out_len: ?*usize,
) callconv(.c) Status {
    const tok = tokenizer orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "tokenizer is NULL");
    const len_out = out_len orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_len is NULL");
    len_out.* = 0;

    if (text_len == 0) return AMI_SDK_OK;
    const tptr = text_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "text_ptr is NULL");
    const text = tptr[0..text_len];

    if (out_ids == null) {
        if (out_cap != 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_ids is NULL with non-zero out_cap");
        const count = tok.encodeInto(text, allow_special_tokens != 0, null) catch |err| {
            return failFmt(AMI_SDK_ERR_TOKENIZER, "tokenize count failed: {s}", .{@errorName(err)});
        };
        len_out.* = count;
        return AMI_SDK_OK;
    }

    const dst = out_ids.?[0..out_cap];
    _ = std.c.pthread_mutex_lock(&tok.mutex);
    defer _ = std.c.pthread_mutex_unlock(&tok.mutex);

    var view_tok = tok.tok;
    if (allow_special_tokens == 0) view_tok.added_tokens = &.{};

    const res = if (text.len <= TOK_SCRATCH_CHUNK_BYTES)
        view_tok.encodeWithScratch(&tok.scratch, text, dst)
    else
        view_tok.encode(gpa, text, dst);

    if (res) |written| {
        len_out.* = written.len;
        return AMI_SDK_OK;
    } else |err| switch (err) {
        error.BufferTooSmall => {
            const needed = tok.encodeCountLocked(&view_tok, text) catch |cerr| {
                return failFmt(AMI_SDK_ERR_TOKENIZER, "tokenize failed: {s}", .{@errorName(cerr)});
            };
            len_out.* = needed;
            return fail(AMI_SDK_ERR_BUFFER_TOO_SMALL, "out_ids buffer too small");
        },
        else => return failFmt(AMI_SDK_ERR_TOKENIZER, "tokenize failed: {s}", .{@errorName(err)}),
    }
}

pub export fn ami_sdk_count_tokens(
    tokenizer: ?*SdkTokenizer,
    text_ptr: ?[*]const u8,
    text_len: usize,
    allow_special_tokens: u32,
    out_count: ?*usize,
) callconv(.c) Status {
    return ami_sdk_tokenize(tokenizer, text_ptr, text_len, allow_special_tokens, null, 0, out_count);
}

pub export fn ami_sdk_detokenize(
    tokenizer: ?*const SdkTokenizer,
    ids_ptr: ?[*]const u32,
    ids_len: usize,
    skip_special_tokens: u32,
    out_utf8: ?[*]u8,
    out_cap: usize,
    out_len: ?*usize,
) callconv(.c) Status {
    const tok = tokenizer orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "tokenizer is NULL");
    const len_out = out_len orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_len is NULL");
    len_out.* = 0;
    if (ids_len == 0) return AMI_SDK_OK;
    const iptr = ids_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "ids_ptr is NULL");
    const ids = iptr[0..ids_len];
    const skip = (skip_special_tokens != 0);

    var needed: usize = 0;
    for (ids) |id| {
        if (skip and tok.tok.isSpecialToken(id)) continue;
        const piece = tok.tok.tokenBytes(id) catch |err| {
            return failFmt(AMI_SDK_ERR_TOKENIZER, "detokenize failed: {s}", .{@errorName(err)});
        };
        needed += piece.len;
    }
    len_out.* = needed;

    if (out_utf8 == null) {
        if (out_cap != 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_utf8 is NULL with non-zero out_cap");
        return AMI_SDK_OK;
    }
    if (out_cap < needed) {
        return fail(AMI_SDK_ERR_BUFFER_TOO_SMALL, "out_utf8 buffer too small");
    }

    const dst = out_utf8.?[0..out_cap];
    const written = (if (skip) tok.tok.decodeSkippingSpecial(ids, dst) else tok.tok.decode(ids, dst)) catch |err| {
        return failFmt(AMI_SDK_ERR_TOKENIZER, "detokenize failed: {s}", .{@errorName(err)});
    };
    len_out.* = written.len;
    return AMI_SDK_OK;
}

// ---------------------------------------------------------------------------
// Engine Lifecycle & Co-Tenant Memory Governance (`ADR-0012`, `R9`)
// ---------------------------------------------------------------------------

fn computeSdkArenaBytes(vocab_size: u32, hidden_size: u32, max_seq_len: u32) error{Overflow}!usize {
    const v: usize = vocab_size;
    const h: usize = hidden_size;
    const seq: usize = max_seq_len;
    const raw = v * @sizeOf(f32) + // logits_f32
        h * @sizeOf(f16) + // hidden_norm_f16
        h * @sizeOf(f32) + // hidden_norm_f32
        h * @sizeOf(f32) + // hidden_raw_f32
        v * @sizeOf(f32) + // sample_probs
        v * @sizeOf(u32) + // sample_indices
        v * @sizeOf(u32) + // token_counts
        seq * @sizeOf(u32) + // history_ids
        seq * @sizeOf(u32) + // prompt_ids_scratch
        RERANK_PROMPT_CAP + // rerank_buf
        h * @sizeOf(f32) + // embed_accum_f32
        2048; // 64-byte alignment paddingslack
    return page.alignUp(raw);
}

fn allocArenaSlice(
    comptime T: type,
    arena: *arena_mod.WiredArena,
    count: usize,
) arena_mod.WiredArena.AllocError![]T {
    const raw = try arena.allocAligned(count * @sizeOf(T), 64);
    return @alignCast(std.mem.bytesAsSlice(T, raw));
}

pub export fn ami_sdk_engine_create(
    opts_ptr: ?*const EngineOptions,
    out_engine: ?*?*SdkEngine,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const out = out_engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_engine is NULL");
    out.* = null;
    const opts = opts_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "opts is NULL");
    if (opts.struct_size != @sizeOf(EngineOptions) or opts.abi_version != ABI_VERSION) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "EngineOptions struct_size or abi_version mismatch");
    }

    const raw_amf = opts.amf_path orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "amf_path is NULL");
    const amf_path = std.mem.span(raw_amf);
    if (amf_path.len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "amf_path is empty");

    const max_seq_len: u32 = if (opts.max_seq_len == 0) 2048 else opts.max_seq_len;

    // W18 embedded-metadata auto-resolution (explicit path -> embedded __meta.* -> sidecar).
    var amf_file = amf_mod.AmfFile.open(gpa, amf_path) catch |err| return mapAnyError(err);
    defer amf_file.close(gpa);

    const explicit_cfg: ?[]const u8 = if (opts.config_json_path) |cp| blk: {
        const s = std.mem.span(cp);
        break :blk if (s.len > 0) s else null;
    } else null;
    var resolved_cfg = amf_mod.resolveConfigJsonAlloc(gpa, &amf_file, amf_path, explicit_cfg) catch |err| return mapAnyError(err);
    defer resolved_cfg.deinit(gpa);
    const cfg_bytes = resolved_cfg.bytes;

    // Resolve optional tokenizer.json (explicit -> embedded __meta.tokenizer_json -> sibling).
    const explicit_tok: ?[]const u8 = if (opts.tokenizer_json_path) |tp| blk: {
        const s = std.mem.span(tp);
        break :blk if (s.len > 0) s else null;
    } else null;
    var owned_tokenizer: ?*SdkTokenizer = null;
    errdefer if (owned_tokenizer) |t| t.destroy(gpa);

    if (amf_mod.resolveTokenizerJsonAlloc(gpa, &amf_file, amf_path, explicit_tok)) |resolved_tok_val| {
        var resolved_tok = resolved_tok_val;
        defer resolved_tok.deinit(gpa);
        owned_tokenizer = SdkTokenizer.createFromBytes(gpa, resolved_tok.bytes) catch |err| {
            return failFmt(AMI_SDK_ERR_TOKENIZER, "tokenizer load failed: {s}", .{@errorName(err)});
        };
    } else |err| switch (err) {
        error.MetadataNotFound => {
            if (explicit_tok != null) return mapAnyError(err);
        },
        else => return mapAnyError(err),
    }

    const is_graph = graph.GraphConfig.isGraphConfigJson(cfg_bytes);
    var maybe_gcfg: ?graph.GraphConfig = null;
    errdefer if (maybe_gcfg) |*gc| gc.deinit(gpa);
    var qcfg: qwen3.ModelConfig = undefined;

    var vocab_size: u32 = 0;
    var hidden_size: u32 = 0;
    var num_layers: u32 = 0;
    var eos_token_id: u32 = TOKEN_NONE;

    if (is_graph) {
        const gcfg = graph.GraphConfig.fromJson(gpa, cfg_bytes) catch |err| return mapAnyError(err);
        vocab_size = gcfg.vocab_size;
        hidden_size = gcfg.hidden_size;
        num_layers = gcfg.num_hidden_layers;
        eos_token_id = gcfg.eos_token_id;
        maybe_gcfg = gcfg;
    } else {
        qcfg = qwen3.ModelConfig.fromJson(gpa, cfg_bytes) catch |err| return mapAnyError(err);
        vocab_size = qcfg.vocab_size;
        hidden_size = qcfg.hidden_size;
        num_layers = qcfg.num_hidden_layers;
        eos_token_id = qcfg.eos_token_id orelse TOKEN_NONE;
    }

    if (eos_token_id == TOKEN_NONE) {
        if (owned_tokenizer) |t| {
            eos_token_id = t.tok.addedTokenId("<|im_end|>") orelse
                t.tok.addedTokenId("<|endoftext|>") orelse
                t.tok.addedTokenId("<｜end▁of▁sentence｜>") orelse TOKEN_NONE;
        }
    }

    // Compute host + co-tenant wired budget and R9 live non-anonymous RAM ceiling.
    const facts = sysinfo.Facts.read() catch |err| return mapAnyError(err);
    const probe_dev = metal.Device.open() catch |err| return mapAnyError(err);
    const dev_info = probe_dev.info() catch |err| {
        probe_dev.close();
        return mapAnyError(err);
    };
    probe_dev.close();

    const headroom = if (opts.headroom_bytes == 0) budget_mod.DEFAULT_HEADROOM_BYTES else opts.headroom_bytes;
    const configured_cap: ?u64 = if (opts.wired_budget_bytes == 0) null else opts.wired_budget_bytes;
    const full_budget = budget_mod.Budget.fromHost(facts, dev_info, .{
        .configured_cap = configured_cap,
        .headroom_bytes = headroom,
    });

    const vm_snap = vmstat.Snapshot.capture() catch |err| return mapAnyError(err);
    const live_reclaimable: u64 = if (opts.has_live_reclaimable_override != 0)
        opts.live_reclaimable_override_bytes
    else
        vm_snap.liveZeroSwapReclaimableBytes();
    const safety_margin: u64 = if (opts.live_safety_margin_bytes == 0)
        budget_mod.DEFAULT_LIVE_SAFETY_MARGIN_BYTES
    else
        opts.live_safety_margin_bytes;
    const strict_live = (opts.strict_live_ram != 0);

    const sdk_arena_bytes = computeSdkArenaBytes(vocab_size, hidden_size, max_seq_len) catch |err| return mapAnyError(err);
    const sdk_live_plan = full_budget.planWithLiveRam(sdk_arena_bytes, 0, live_reclaimable, .{
        .live_safety_margin_bytes = safety_margin,
        .strict_live_ram = strict_live,
    }) catch |err| return mapAnyError(err);

    var sdk_arena = arena_mod.WiredArena.init(sdk_arena_bytes, full_budget) catch |err| return mapAnyError(err);
    errdefer sdk_arena.deinit();

    const rem_limit = full_budget.limit_bytes - sdk_arena.bytes().len;
    const rem_budget: budget_mod.Budget = .{
        .limit_bytes = rem_limit,
        .binding_limit = full_budget.binding_limit,
        .raw_limit_bytes = rem_limit,
    };
    const rem_live_reclaimable: u64 = live_reclaimable -| sdk_arena.bytes().len;

    const wide_chunk_toks: u32 = @min(@as(u32, 512), max_seq_len);
    const tree_nodes: u32 = @min(@as(u32, 64), max_seq_len);

    var backend: ModelBackend = undefined;
    if (is_graph) {
        const gmodel = graph.GraphModel.loadFromFile(gpa, amf_path, maybe_gcfg.?, .{
            .max_seq_len = max_seq_len,
            .max_tree_nodes = tree_nodes,
            .enable_feature_taps = true,
            .wired_budget_override = rem_budget,
            .startup_canary = opts.startup_canary != 0,
            .strict_live_ram = strict_live,
            .live_safety_margin_bytes = safety_margin,
            .live_reclaimable_override = rem_live_reclaimable,
            .wide_prefill_chunk_tokens = wide_chunk_toks,
        }) catch |err| return mapAnyError(err);
        backend = .{ .graph = gmodel };
    } else {
        const qmodel = qwen3.Qwen3Model.loadFromFile(gpa, amf_path, qcfg, .{
            .max_seq_len = max_seq_len,
            .max_tree_nodes = tree_nodes,
            .enable_feature_taps = true,
            .budget_override = rem_budget,
            .verify_crcs = opts.verify_crcs != 0,
            .startup_canary = opts.startup_canary != 0,
            .strict_live_ram = strict_live,
            .live_safety_margin_bytes = safety_margin,
            .live_reclaimable_override = rem_live_reclaimable,
            .wide_prefill_chunk_tokens = wide_chunk_toks,
        }) catch |err| return mapAnyError(err);
        backend = .{ .qwen3 = qmodel };
    }
    errdefer switch (backend) {
        .qwen3 => |*m| m.deinit(gpa),
        .graph => |*m| m.deinit(gpa),
    };

    const logits_f32 = allocArenaSlice(f32, &sdk_arena, vocab_size) catch |err| return mapAnyError(err);
    const hidden_norm_f16 = allocArenaSlice(f16, &sdk_arena, hidden_size) catch |err| return mapAnyError(err);
    const hidden_norm_f32 = allocArenaSlice(f32, &sdk_arena, hidden_size) catch |err| return mapAnyError(err);
    const hidden_raw_f32 = allocArenaSlice(f32, &sdk_arena, hidden_size) catch |err| return mapAnyError(err);
    const sample_probs = allocArenaSlice(f32, &sdk_arena, vocab_size) catch |err| return mapAnyError(err);
    const sample_indices = allocArenaSlice(u32, &sdk_arena, vocab_size) catch |err| return mapAnyError(err);
    const token_counts = allocArenaSlice(u32, &sdk_arena, vocab_size) catch |err| return mapAnyError(err);
    @memset(token_counts, 0);
    const history_ids = allocArenaSlice(u32, &sdk_arena, max_seq_len) catch |err| return mapAnyError(err);
    const prompt_ids_scratch = allocArenaSlice(u32, &sdk_arena, max_seq_len) catch |err| return mapAnyError(err);
    const rerank_buf = sdk_arena.allocAligned(RERANK_PROMPT_CAP, 64) catch |err| return mapAnyError(err);
    const embed_accum_f32 = allocArenaSlice(f32, &sdk_arena, hidden_size) catch |err| return mapAnyError(err);

    const eng = gpa.create(SdkEngine) catch return fail(AMI_SDK_ERR_OUT_OF_MEMORY, "SdkEngine alloc failed");
    eng.* = .{
        .backend = backend,
        .graph_cfg = maybe_gcfg,
        .owned_tokenizer = owned_tokenizer,
        .vocab_size = vocab_size,
        .hidden_size = hidden_size,
        .num_layers = num_layers,
        .max_seq_len = max_seq_len,
        .eos_token_id = eos_token_id,
        .wired_budget_bytes = full_budget.limit_bytes,
        .live_reclaimable_bytes = live_reclaimable,
        .live_budget_bytes = sdk_live_plan.live_budget_bytes,
        .sdk_arena = sdk_arena,
        .logits_f32 = logits_f32,
        .hidden_norm_f16 = hidden_norm_f16,
        .hidden_norm_f32 = hidden_norm_f32,
        .hidden_raw_f32 = hidden_raw_f32,
        .sample_probs = sample_probs,
        .sample_indices = sample_indices,
        .token_counts = token_counts,
        .history_ids = history_ids,
        .prompt_ids_scratch = prompt_ids_scratch,
        .rerank_buf = rerank_buf,
        .embed_accum_f32 = embed_accum_f32,
    };

    out.* = eng;
    return AMI_SDK_OK;
}

pub export fn ami_sdk_engine_destroy(engine: ?*SdkEngine) callconv(.c) void {
    const eng = engine orelse return;
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);
    eng.destroy(gpa);
}

pub export fn ami_sdk_engine_reset_kv(engine: ?*SdkEngine) callconv(.c) Status {
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    eng.resetKv();
    return AMI_SDK_OK;
}

pub export fn ami_sdk_engine_query_stats(
    engine: ?*const SdkEngine,
    out_stats: ?*EngineStats,
) callconv(.c) Status {
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const stats = out_stats orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_stats is NULL");
    if (stats.struct_size != 0 and stats.struct_size < @sizeOf(EngineStats)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "EngineStats struct_size mismatch");
    }
    const sdk_bytes: u64 = eng.sdk_arena.bytes().len;
    stats.* = .{
        .struct_size = @sizeOf(EngineStats),
        .is_graph_model = switch (eng.backend) {
            .qwen3 => 0,
            .graph => 1,
        },
        .vocab_size = eng.vocab_size,
        .hidden_size = eng.hidden_size,
        .num_layers = eng.num_layers,
        .max_seq_len = eng.max_seq_len,
        .seq_len = eng.currentSeqLen(),
        .reserved = 0,
        .total_wired_bytes = eng.totalModelWiredBytes() + sdk_bytes,
        .sdk_wired_bytes = sdk_bytes,
        .wired_budget_bytes = eng.wired_budget_bytes,
        .live_reclaimable_bytes = eng.live_reclaimable_bytes,
        .live_budget_bytes = eng.live_budget_bytes,
    };
    return AMI_SDK_OK;
}

pub export fn ami_sdk_engine_tokenizer(engine: ?*const SdkEngine) callconv(.c) ?*const SdkTokenizer {
    const eng = engine orelse return null;
    return eng.owned_tokenizer;
}

// ---------------------------------------------------------------------------
// Prefill, Single-Step Decode, Sampling & Streaming Generation
// ---------------------------------------------------------------------------

fn normWeightsU16(norm: qwen3.NormBinding) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, norm.host_bytes));
}

fn captureLatestHiddenState(eng: *SdkEngine, m_chunk: u32) void {
    const h: usize = eng.hidden_size;
    switch (eng.backend) {
        .qwen3 => |*m| {
            const top = if (m.enable_feature_taps and m_chunk > 1 and m.usesWideGpuPrefill())
                m.accepted_top_host[(@as(usize, m_chunk) - 1) * h .. @as(usize, m_chunk) * h]
            else
                m.readTopHiddenF16(1);
            attention.rmsNormCpuRefDType(
                top,
                m.final_norm.dtype,
                normWeightsU16(m.final_norm),
                eng.hidden_norm_f32,
                1,
                m.config.hidden_size,
                m.config.rms_norm_eps,
            );
            for (0..h) |i| {
                eng.hidden_raw_f32[i] = @floatCast(top[i]);
                eng.hidden_norm_f16[i] = @floatCast(eng.hidden_norm_f32[i]);
            }
            eng.hidden_raw_f32_valid = true;
            eng.hidden_norm_f16_valid = true;
            eng.hidden_norm_f32_valid = true;
        },
        .graph => |*m| {
            if (m.config.isCausalEncDec()) {
                const dec_h = m.lastDecoderHiddenF32();
                for (0..h) |i| {
                    eng.hidden_raw_f32[i] = dec_h[i];
                    eng.hidden_norm_f32[i] = dec_h[i];
                    eng.hidden_norm_f16[i] = @floatCast(dec_h[i]);
                }
            } else {
                const top = if (m.enable_feature_taps and m_chunk > 1 and m.usesWideGpuPrefill())
                    m.accepted_top_host[(@as(usize, m_chunk) - 1) * h .. @as(usize, m_chunk) * h]
                else
                    m.readTopHiddenF16(1);
                attention.rmsNormCpuRefDType(
                    top,
                    m.final_norm.dtype,
                    normWeightsU16(m.final_norm),
                    eng.hidden_norm_f32,
                    1,
                    m.config.hidden_size,
                    m.config.rms_norm_eps,
                );
                for (0..h) |i| {
                    eng.hidden_raw_f32[i] = @floatCast(top[i]);
                    eng.hidden_norm_f16[i] = @floatCast(eng.hidden_norm_f32[i]);
                }
            }
            eng.hidden_raw_f32_valid = true;
            eng.hidden_norm_f16_valid = true;
            eng.hidden_norm_f32_valid = true;
        },
    }
}

fn prefillTokensInternal(
    eng: *SdkEngine,
    prompt_ids: []const u32,
    compute_logits: bool,
    reset_kv: bool,
    cancel_token: ?*const SdkCancelToken,
    out_greedy_next: ?*u32,
) Status {
    if (prompt_ids.len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prompt_len is 0");
    if (SdkCancelToken.isCancelled(cancel_token)) return fail(AMI_SDK_ERR_CANCELLED, "request cancelled");

    if (reset_kv) eng.resetKv();

    const base_pos = eng.currentSeqLen();
    if (base_pos + prompt_ids.len > eng.max_seq_len) {
        return fail(AMI_SDK_ERR_SEQUENCE_FULL, "prompt exceeds max_seq_len");
    }
    for (prompt_ids) |tid| {
        if (tid >= eng.vocab_size) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "token_id out of range");
    }

    eng.invalidateViews();
    const cap: usize = switch (eng.backend) {
        .qwen3 => |*m| @max(@as(usize, 1), @as(usize, m.prefillChunkCap())),
        .graph => |*m| @max(@as(usize, 1), @as(usize, m.prefillChunkCap())),
    };

    var off: usize = 0;
    var last_argmax: u32 = 0;
    var last_chunk_len: u32 = 1;
    while (off < prompt_ids.len) {
        if (SdkCancelToken.isCancelled(cancel_token)) {
            return fail(AMI_SDK_ERR_CANCELLED, "request cancelled during prefill");
        }
        const take = @min(cap, prompt_ids.len - off);
        const is_last = (off + take == prompt_ids.len);
        const start_pos: u32 = base_pos + @as(u32, @intCast(off));
        const slice = prompt_ids[off .. off + take];
        last_chunk_len = @intCast(take);

        switch (eng.backend) {
            .qwen3 => |*m| {
                if (take == 1 and !m.usesWideGpuPrefill()) {
                    if (compute_logits and is_last) {
                        last_argmax = m.stepDecode(slice[0], start_pos) catch |err| return mapAnyError(err);
                    } else {
                        m.prefillToken(slice[0], start_pos) catch |err| return mapAnyError(err);
                    }
                } else {
                    last_argmax = m.prefillChunkWithOptions(slice, start_pos, .{
                        .compute_logits = compute_logits and is_last,
                        .preserve_all_top_hidden = is_last,
                    }) catch |err| return mapAnyError(err);
                }
            },
            .graph => |*m| {
                last_argmax = m.prefillChunkWithOptions(slice, start_pos, .{
                    .compute_logits = compute_logits and is_last,
                    .preserve_all_top_hidden = is_last,
                }) catch |err| return mapAnyError(err);
            },
        }
        for (slice) |tid| eng.recordToken(tid);
        off += take;
    }

    if (compute_logits) {
        eng.has_step_state = true;
        captureLatestHiddenState(eng, last_chunk_len);
    }
    if (out_greedy_next) |dst| dst.* = last_argmax;
    return AMI_SDK_OK;
}

pub export fn ami_sdk_prefill_tokens(
    engine: ?*SdkEngine,
    prompt_ids: ?[*]const u32,
    prompt_len: usize,
    compute_logits: u32,
    reset_kv_cache: u32,
    cancel_token: ?*const SdkCancelToken,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    if (prompt_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prompt_len is 0");
    const ptr = prompt_ids orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prompt_ids is NULL");
    return prefillTokensInternal(eng, ptr[0..prompt_len], compute_logits != 0, reset_kv_cache != 0, cancel_token, null);
}

pub export fn ami_sdk_step_decode(
    engine: ?*SdkEngine,
    token_id: u32,
    out_next_token_id: ?*u32,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    if (token_id >= eng.vocab_size) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "token_id out of range");
    const pos = eng.currentSeqLen();
    if (pos >= eng.max_seq_len) return fail(AMI_SDK_ERR_SEQUENCE_FULL, "KV cache is full");

    eng.invalidateViews();
    const next_tok: u32 = switch (eng.backend) {
        .qwen3 => |*m| m.stepDecode(token_id, pos) catch |err| return mapAnyError(err),
        .graph => |*m| m.stepDecode(token_id, pos) catch |err| return mapAnyError(err),
    };
    eng.recordToken(token_id);
    eng.has_step_state = true;
    captureLatestHiddenState(eng, 1);
    if (out_next_token_id) |dst| dst.* = next_tok;
    return AMI_SDK_OK;
}

fn sampleNextToken(
    eng: *SdkEngine,
    greedy_argmax: u32,
    params: *const GenParams,
    req_rng: *rng_mod.RequestRng,
) u32 {
    const has_rep = (params.repetition_penalty != 1.0 and params.repetition_penalty > 0.0);
    const has_pres = (params.presence_penalty != 0.0);
    const has_freq = (params.frequency_penalty != 0.0);
    const any_penalty = has_rep or has_pres or has_freq;

    if (!any_penalty and params.temperature <= 0.0) {
        return greedy_argmax;
    }

    const v: usize = eng.vocab_size;
    const logits_f16 = eng.lastLogitsF16();
    const probs = eng.sample_probs[0..v];
    for (0..v) |i| probs[i] = @floatCast(logits_f16[i]);

    if (any_penalty and eng.history_len > 0) {
        const hist = eng.history_ids[0..eng.history_len];
        for (hist) |tid| {
            if (tid < v) eng.token_counts[tid] += 1;
        }
        defer for (hist) |tid| {
            if (tid < v) eng.token_counts[tid] = 0;
        };

        for (hist) |tid| {
            if (tid >= v) continue;
            const c = eng.token_counts[tid];
            if (c == 0) continue;
            // Apply once per unique seen token ID by negating/marking via MSB? Even cleaner:
            // clear count immediately after applying so duplicates in `hist` are O(1) skipped!
            eng.token_counts[tid] = 0;
            var val = probs[tid];
            if (has_rep) {
                if (val > 0.0) val /= params.repetition_penalty else val *= params.repetition_penalty;
            }
            if (has_freq) val -= params.frequency_penalty * @as(f32, @floatFromInt(c));
            if (has_pres) val -= params.presence_penalty;
            probs[tid] = val;
        }
    }

    if (params.temperature <= 0.0 or params.top_k == 1) {
        var best_idx: u32 = 0;
        var best_val: f32 = probs[0];
        for (1..v) |i| {
            if (probs[i] > best_val) {
                best_val = probs[i];
                best_idx = @intCast(i);
            }
        }
        return best_idx;
    }

    const inv_temp: f32 = 1.0 / params.temperature;
    var max_logit: f32 = -std.math.inf(f32);
    for (probs) |val| {
        const scaled = val * inv_temp;
        if (scaled > max_logit) max_logit = scaled;
    }

    var sum: f32 = 0.0;
    var max_unnorm_prob: f32 = 0.0;
    for (probs) |*val| {
        const p = @exp(val.* * inv_temp - max_logit);
        val.* = p;
        sum += p;
        if (p > max_unnorm_prob) max_unnorm_prob = p;
    }
    if (!(sum > 0.0) or std.math.isNan(sum)) return greedy_argmax;

    const use_top_k = (params.top_k > 0 and params.top_k < v);
    const use_top_p = (params.top_p > 0.0 and params.top_p < 1.0);
    const use_min_p = (params.min_p > 0.0 and params.min_p < 1.0);

    if (!use_top_k and !use_top_p and !use_min_p) {
        const inv_sum = 1.0 / sum;
        const u = req_rng.random().float(f32);
        var cdf: f32 = 0.0;
        var last_idx: u32 = 0;
        for (probs, 0..) |p, idx| {
            if (p > 0.0) {
                last_idx = @intCast(idx);
                cdf += p * inv_sum;
                if (u <= cdf) return @intCast(idx);
            }
        }
        return last_idx;
    }

    const min_unnorm_thresh: f32 = if (use_min_p) params.min_p * max_unnorm_prob else 0.0;
    var cand_count: usize = 0;
    const indices = eng.sample_indices[0..v];
    for (probs, 0..) |p, idx| {
        if (p > 0.0 and p >= min_unnorm_thresh) {
            indices[cand_count] = @intCast(idx);
            cand_count += 1;
        }
    }
    if (cand_count == 0) return greedy_argmax;

    const DescCtx = struct {
        weights: []const f32,
        pub fn lessThan(ctx: @This(), a: u32, b: u32) bool {
            if (ctx.weights[a] == ctx.weights[b]) return a < b;
            return ctx.weights[a] > ctx.weights[b];
        }
    };
    std.mem.sort(u32, indices[0..cand_count], DescCtx{ .weights = probs }, DescCtx.lessThan);

    if (use_top_k and params.top_k < cand_count) {
        cand_count = params.top_k;
    }

    var kept_sum: f32 = 0.0;
    for (indices[0..cand_count]) |idx| kept_sum += probs[idx];

    if (use_top_p and kept_sum > 0.0) {
        const target = params.top_p * kept_sum;
        var prefix: f32 = 0.0;
        var cut: usize = 0;
        while (cut < cand_count) : (cut += 1) {
            prefix += probs[indices[cut]];
            if (prefix >= target) {
                cut += 1;
                break;
            }
        }
        cand_count = @max(@as(usize, 1), cut);
        kept_sum = prefix;
    }

    if (!(kept_sum > 0.0)) return indices[0];
    const r = req_rng.random().float(f32) * kept_sum;
    var acc: f32 = 0.0;
    for (indices[0..cand_count]) |idx| {
        acc += probs[idx];
        if (r <= acc) return idx;
    }
    return indices[cand_count - 1];
}

fn isStopToken(eng: *const SdkEngine, params: *const GenParams, tok: u32) bool {
    if (params.ignore_eos == 0 and eng.eos_token_id != TOKEN_NONE and tok == eng.eos_token_id) {
        return true;
    }
    const n = @min(@as(usize, params.stop_token_count), MAX_STOP_TOKENS);
    for (params.stop_token_ids[0..n]) |sid| {
        if (tok == sid) return true;
    }
    return false;
}

const StreamDispatchCtx = struct {
    cb: StreamCallback,
    user_data: ?*anyopaque,
    classifier: *StreamTagClassifier,
    token_id: u32,
    step_index: u32,
    is_final_flush: u32,
    stopped_by_cb: bool = false,

    fn onUtf8(self: *StreamDispatchCtx, valid_utf8: []const u8) bool {
        return self.classifier.feed(valid_utf8, self, onClassified);
    }

    fn onClassified(self: *StreamDispatchCtx, kind: DeltaKind, text: []const u8) bool {
        if (text.len == 0 and kind != .tool_call_end) return true;
        const empty: [1]u8 = .{0};
        const ev: StreamEvent = .{
            .struct_size = @sizeOf(StreamEvent),
            .token_id = self.token_id,
            .step_index = self.step_index,
            .delta_kind = @backingInt(kind),
            .utf8_ptr = if (text.len > 0) text.ptr else &empty,
            .utf8_len = text.len,
            .is_final_flush = self.is_final_flush,
            .reserved = 0,
        };
        if (self.cb(&ev, self.user_data) != 0) {
            self.stopped_by_cb = true;
            return false;
        }
        return true;
    }
};

fn generateTokensImpl(
    eng: *SdkEngine,
    prompt_ids: []const u32,
    params: *const GenParams,
    stream_cb: ?StreamCallback,
    user_data: ?*anyopaque,
    out_token_ids: ?[]u32,
    out_result: ?*GenResult,
) Status {
    if (params.struct_size != @sizeOf(GenParams)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "GenParams struct_size mismatch");
    }
    if (out_result) |res| {
        if (res.struct_size != 0 and res.struct_size < @sizeOf(GenResult)) {
            return fail(AMI_SDK_ERR_ABI_MISMATCH, "GenResult struct_size mismatch");
        }
    }

    const explicit_seed: ?u64 = if (params.has_seed != 0) params.seed else null;
    var req_rng = rng_mod.RequestRng.init(explicit_seed) catch |err| return mapAnyError(err);

    const t0 = vmstat.nowNs() catch |err| return mapAnyError(err);
    var greedy_first: u32 = 0;
    const pre_st = prefillTokensInternal(
        eng,
        prompt_ids,
        true,
        params.reset_kv_cache != 0,
        params.cancel_token,
        &greedy_first,
    );
    if (pre_st != AMI_SDK_OK) return pre_st;
    const t_prefill = vmstat.nowNs() catch |err| return mapAnyError(err);

    var utf8_asm: Utf8Assembler = .{};
    var tag_cls: StreamTagClassifier = .{};
    tag_cls.reset(params.start_in_think != 0);

    var generated: u32 = 0;
    var finish: FinishReason = .length;
    var prev_greedy: u32 = greedy_first;

    while (generated < params.max_tokens) {
        if (SdkCancelToken.isCancelled(params.cancel_token)) {
            finish = .cancelled;
            break;
        }

        if (generated > 0) {
            // Feed previous generated token into single-step decode.
            const prev_tok = eng.history_ids[eng.history_len - 1];
            const pos = eng.currentSeqLen();
            if (pos >= eng.max_seq_len) {
                finish = .length;
                break;
            }
            eng.invalidateViews();
            prev_greedy = switch (eng.backend) {
                .qwen3 => |*m| m.stepDecode(prev_tok, pos) catch |err| return mapAnyError(err),
                .graph => |*m| m.stepDecode(prev_tok, pos) catch |err| return mapAnyError(err),
            };
            eng.has_step_state = true;
            captureLatestHiddenState(eng, 1);
        }

        const sampled = sampleNextToken(eng, prev_greedy, params, &req_rng);
        const hit_stop = isStopToken(eng, params, sampled);

        if (hit_stop) {
            finish = if (tag_cls.saw_tool_call) .tool_calls else .stop;
            break;
        }

        if (out_token_ids) |dst| {
            if (generated < dst.len) dst[generated] = sampled;
        }
        eng.recordToken(sampled);
        const cur_step = generated;
        generated += 1;

        if (stream_cb) |cb| {
            if (eng.owned_tokenizer) |stok| {
                const skip_spec = (params.skip_special_tokens != 0) and stok.tok.isSpecialToken(sampled);
                if (!skip_spec) {
                    const piece = stok.tok.tokenBytes(sampled) catch |err| return mapAnyError(err);
                    var dctx: StreamDispatchCtx = .{
                        .cb = cb,
                        .user_data = user_data,
                        .classifier = &tag_cls,
                        .token_id = sampled,
                        .step_index = cur_step,
                        .is_final_flush = 0,
                    };
                    if (!utf8_asm.feed(piece, &dctx, StreamDispatchCtx.onUtf8)) {
                        if (dctx.stopped_by_cb) {
                            finish = .callback_stop;
                            break;
                        }
                    }
                }
            } else {
                const empty: [1]u8 = .{0};
                const ev: StreamEvent = .{
                    .struct_size = @sizeOf(StreamEvent),
                    .token_id = sampled,
                    .step_index = cur_step,
                    .delta_kind = @backingInt(DeltaKind.text),
                    .utf8_ptr = &empty,
                    .utf8_len = 0,
                    .is_final_flush = 0,
                    .reserved = 0,
                };
                if (cb(&ev, user_data) != 0) {
                    finish = .callback_stop;
                    break;
                }
            }
        }
    }

    if (stream_cb) |cb| {
        if (finish != .callback_stop and finish != .cancelled and eng.owned_tokenizer != null) {
            var fctx: StreamDispatchCtx = .{
                .cb = cb,
                .user_data = user_data,
                .classifier = &tag_cls,
                .token_id = TOKEN_NONE,
                .step_index = generated,
                .is_final_flush = 1,
            };
            _ = utf8_asm.flush(&fctx, StreamDispatchCtx.onUtf8);
            if (!fctx.stopped_by_cb) {
                _ = tag_cls.flush(&fctx, StreamDispatchCtx.onClassified);
            }
            if (fctx.stopped_by_cb) finish = .callback_stop;
        }
    }

    if (finish == .length and tag_cls.saw_tool_call and generated < params.max_tokens) {
        finish = .tool_calls;
    }

    const t_end = vmstat.nowNs() catch |err| return mapAnyError(err);
    if (out_result) |res| {
        res.* = .{
            .struct_size = @sizeOf(GenResult),
            .prompt_tokens = @intCast(prompt_ids.len),
            .generated_tokens = generated,
            .finish_reason = @backingInt(finish),
            .seed_used = req_rng.seed_used,
            .prefill_ns = @floatFromInt(t_prefill -| t0),
            .decode_ns = @floatFromInt(t_end -| t_prefill),
        };
    }

    if (finish == .cancelled) return fail(AMI_SDK_ERR_CANCELLED, "generation cancelled");
    return AMI_SDK_OK;
}

pub export fn ami_sdk_generate_tokens(
    engine: ?*SdkEngine,
    prompt_ids: ?[*]const u32,
    prompt_len: usize,
    params_ptr: ?*const GenParams,
    stream_cb: ?StreamCallback,
    user_data: ?*anyopaque,
    out_token_ids: ?[*]u32,
    out_token_cap: usize,
    out_result: ?*GenResult,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    if (prompt_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prompt_len is 0");
    const pids = prompt_ids orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prompt_ids is NULL");
    const params = params_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "params is NULL");

    const out_slice: ?[]u32 = if (out_token_ids) |p| p[0..out_token_cap] else null;
    return generateTokensImpl(eng, pids[0..prompt_len], params, stream_cb, user_data, out_slice, out_result);
}

pub export fn ami_sdk_generate_text(
    engine: ?*SdkEngine,
    prompt_utf8: ?[*]const u8,
    prompt_len: usize,
    params_ptr: ?*const GenParams,
    stream_cb: ?StreamCallback,
    user_data: ?*anyopaque,
    out_token_ids: ?[*]u32,
    out_token_cap: usize,
    out_result: ?*GenResult,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const stok = eng.owned_tokenizer orelse return fail(AMI_SDK_ERR_TOKENIZER, "engine has no loaded tokenizer");
    if (prompt_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prompt_len is 0");
    const pstr = prompt_utf8 orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prompt_utf8 is NULL");
    const params = params_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "params is NULL");

    var tok_len: usize = 0;
    const tok_st = ami_sdk_tokenize(
        stok,
        pstr,
        prompt_len,
        1,
        eng.prompt_ids_scratch.ptr,
        eng.prompt_ids_scratch.len,
        &tok_len,
    );
    if (tok_st == AMI_SDK_ERR_BUFFER_TOO_SMALL) {
        return fail(AMI_SDK_ERR_SEQUENCE_FULL, "tokenized prompt exceeds max_seq_len");
    }
    if (tok_st != AMI_SDK_OK) return tok_st;
    if (tok_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prompt tokenized to 0 tokens");

    const out_slice: ?[]u32 = if (out_token_ids) |p| p[0..out_token_cap] else null;
    return generateTokensImpl(eng, eng.prompt_ids_scratch[0..tok_len], params, stream_cb, user_data, out_slice, out_result);
}

// ---------------------------------------------------------------------------
// Read-Only Zero-Copy Logits & Final Hidden-State Views
// ---------------------------------------------------------------------------

pub export fn ami_sdk_get_logits_view(
    engine: ?*SdkEngine,
    dtype: u32,
    out_view: ?*TensorView,
) callconv(.c) Status {
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const view = out_view orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_view is NULL");
    if (view.struct_size != 0 and view.struct_size < @sizeOf(TensorView)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "TensorView struct_size mismatch");
    }
    if (!eng.has_step_state) {
        return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "no logits computed yet; run prefill(compute_logits=1) or generate first");
    }

    const v: usize = eng.vocab_size;
    const logits_f16 = eng.lastLogitsF16();

    if (dtype == @backingInt(DType.f16)) {
        view.* = .{
            .struct_size = @sizeOf(TensorView),
            .dtype = @backingInt(DType.f16),
            .data = @ptrCast(logits_f16.ptr),
            .byte_len = v * @sizeOf(f16),
            .ndim = 2,
            .reserved = 0,
            .shape = .{ 1, v },
        };
        return AMI_SDK_OK;
    }
    if (dtype == @backingInt(DType.f32)) {
        if (!eng.logits_f32_valid) {
            for (0..v) |i| eng.logits_f32[i] = @floatCast(logits_f16[i]);
            eng.logits_f32_valid = true;
        }
        view.* = .{
            .struct_size = @sizeOf(TensorView),
            .dtype = @backingInt(DType.f32),
            .data = @ptrCast(eng.logits_f32.ptr),
            .byte_len = v * @sizeOf(f32),
            .ndim = 2,
            .reserved = 0,
            .shape = .{ 1, v },
        };
        return AMI_SDK_OK;
    }
    return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "unsupported dtype for logits view");
}

pub export fn ami_sdk_get_hidden_state_view(
    engine: ?*SdkEngine,
    dtype: u32,
    normalized: u32,
    out_view: ?*TensorView,
) callconv(.c) Status {
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const view = out_view orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_view is NULL");
    if (view.struct_size != 0 and view.struct_size < @sizeOf(TensorView)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "TensorView struct_size mismatch");
    }
    if (!eng.has_step_state or !eng.hidden_norm_f32_valid) {
        return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "no hidden state computed yet");
    }

    const h: usize = eng.hidden_size;
    if (normalized != 0) {
        if (dtype == @backingInt(DType.f16)) {
            view.* = .{
                .struct_size = @sizeOf(TensorView),
                .dtype = @backingInt(DType.f16),
                .data = @ptrCast(eng.hidden_norm_f16.ptr),
                .byte_len = h * @sizeOf(f16),
                .ndim = 2,
                .reserved = 0,
                .shape = .{ 1, h },
            };
            return AMI_SDK_OK;
        }
        if (dtype == @backingInt(DType.f32)) {
            view.* = .{
                .struct_size = @sizeOf(TensorView),
                .dtype = @backingInt(DType.f32),
                .data = @ptrCast(eng.hidden_norm_f32.ptr),
                .byte_len = h * @sizeOf(f32),
                .ndim = 2,
                .reserved = 0,
                .shape = .{ 1, h },
            };
            return AMI_SDK_OK;
        }
    } else {
        if (dtype == @backingInt(DType.f16)) {
            const raw_f16 = eng.lastTopHiddenF16();
            view.* = .{
                .struct_size = @sizeOf(TensorView),
                .dtype = @backingInt(DType.f16),
                .data = @ptrCast(raw_f16.ptr),
                .byte_len = h * @sizeOf(f16),
                .ndim = 2,
                .reserved = 0,
                .shape = .{ 1, h },
            };
            return AMI_SDK_OK;
        }
        if (dtype == @backingInt(DType.f32)) {
            view.* = .{
                .struct_size = @sizeOf(TensorView),
                .dtype = @backingInt(DType.f32),
                .data = @ptrCast(eng.hidden_raw_f32.ptr),
                .byte_len = h * @sizeOf(f32),
                .ndim = 2,
                .reserved = 0,
                .shape = .{ 1, h },
            };
            return AMI_SDK_OK;
        }
    }
    return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "unsupported dtype for hidden-state view");
}

// ---------------------------------------------------------------------------
// Native Embeddings (`lm_head` elided) & Cross-Encoder Reranking (`C21`)
// ---------------------------------------------------------------------------

fn parsePoolingMode(raw: u32) ?PoolingMode {
    return switch (raw) {
        0 => .last_token,
        1 => .mean,
        2 => .cls,
        else => null,
    };
}

fn parseRerankMode(raw: u32) ?RerankMode {
    return switch (raw) {
        0 => .logit_diff,
        1 => .prob_yes,
        2 => .raw_logit,
        else => null,
    };
}

fn embedTokensImpl(
    eng: *SdkEngine,
    token_ids: []const u32,
    params: *const EmbedParams,
    out_embedding: ?[*]f32,
    out_cap: usize,
    out_dims: *usize,
) Status {
    if (params.struct_size != @sizeOf(EmbedParams)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "EmbedParams struct_size mismatch");
    }
    if (token_ids.len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "token_len is 0");
    if (token_ids.len > eng.max_seq_len) return fail(AMI_SDK_ERR_SEQUENCE_FULL, "token_len exceeds max_seq_len");
    for (token_ids) |tid| {
        if (tid >= eng.vocab_size) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "token_id out of range");
    }

    const h: usize = eng.hidden_size;
    const eff_dims: usize = if (params.dimensions == 0) h else params.dimensions;
    if (eff_dims == 0 or eff_dims > h) {
        return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "dimensions must be in 0..hidden_size");
    }
    out_dims.* = eff_dims;

    const pool_mode = parsePoolingMode(params.pooling) orelse {
        return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "invalid pooling mode");
    };

    if (out_embedding == null) {
        if (out_cap != 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_embedding is NULL with non-zero out_cap");
        return AMI_SDK_OK;
    }
    if (out_cap < eff_dims) {
        return fail(AMI_SDK_ERR_BUFFER_TOO_SMALL, "out_embedding buffer smaller than eff_dims");
    }
    if (SdkCancelToken.isCancelled(params.cancel_token)) {
        return fail(AMI_SDK_ERR_CANCELLED, "embed cancelled");
    }

    eng.resetKv();
    const accum = eng.embed_accum_f32[0..h];
    @memset(accum, 0.0);

    const cap: usize = switch (eng.backend) {
        .qwen3 => |*m| @max(@as(usize, 1), @as(usize, m.prefillChunkCap())),
        .graph => |*m| @max(@as(usize, 1), @as(usize, m.prefillChunkCap())),
    };

    var off: usize = 0;
    while (off < token_ids.len) {
        if (SdkCancelToken.isCancelled(params.cancel_token)) {
            return fail(AMI_SDK_ERR_CANCELLED, "embed cancelled");
        }
        const take = @min(cap, token_ids.len - off);
        const is_first = (off == 0);
        const is_last = (off + take == token_ids.len);
        const need_all = (pool_mode == .mean) or (pool_mode == .cls and is_first) or (pool_mode == .last_token and is_last);
        const slice = token_ids[off .. off + take];
        const start_pos: u32 = @intCast(off);

        switch (eng.backend) {
            .qwen3 => |*m| {
                // Wide prefill with `compute_logits = false` completely elides `lm_head`!
                _ = m.prefillChunkWithOptions(slice, start_pos, .{
                    .compute_logits = false,
                    .preserve_all_top_hidden = need_all,
                }) catch |err| return mapAnyError(err);

                if (need_all) {
                    const top_rows: []const f16 = if (m.usesWideGpuPrefill() and m.enable_feature_taps)
                        m.accepted_top_host[0 .. take * h]
                    else
                        m.readTopHiddenF16(@intCast(take));

                    const row_lo: usize = switch (pool_mode) {
                        .cls => 0,
                        .last_token => take - 1,
                        .mean => 0,
                    };
                    const row_hi: usize = switch (pool_mode) {
                        .cls => 1,
                        .last_token => take,
                        .mean => take,
                    };
                    for (row_lo..row_hi) |r| {
                        const row_f16 = top_rows[r * h .. (r + 1) * h];
                        attention.rmsNormCpuRefDType(
                            row_f16,
                            m.final_norm.dtype,
                            normWeightsU16(m.final_norm),
                            eng.hidden_norm_f32,
                            1,
                            m.config.hidden_size,
                            m.config.rms_norm_eps,
                        );
                        for (0..h) |d| accum[d] += eng.hidden_norm_f32[d];
                    }
                }
            },
            .graph => |*m| {
                _ = m.prefillChunkWithOptions(slice, start_pos, .{
                    .compute_logits = false,
                    .preserve_all_top_hidden = need_all,
                }) catch |err| return mapAnyError(err);

                if (need_all) {
                    const top_rows: []const f16 = if (m.usesWideGpuPrefill() and m.enable_feature_taps)
                        m.accepted_top_host[0 .. take * h]
                    else
                        m.readTopHiddenF16(@intCast(take));

                    const row_lo: usize = switch (pool_mode) {
                        .cls => 0,
                        .last_token => take - 1,
                        .mean => 0,
                    };
                    const row_hi: usize = switch (pool_mode) {
                        .cls => 1,
                        .last_token => take,
                        .mean => take,
                    };
                    for (row_lo..row_hi) |r| {
                        const row_f16 = top_rows[r * h .. (r + 1) * h];
                        attention.rmsNormCpuRefDType(
                            row_f16,
                            m.final_norm.dtype,
                            normWeightsU16(m.final_norm),
                            eng.hidden_norm_f32,
                            1,
                            m.config.hidden_size,
                            m.config.rms_norm_eps,
                        );
                        for (0..h) |d| accum[d] += eng.hidden_norm_f32[d];
                    }
                }
            },
        }
        off += take;
    }

    if (pool_mode == .mean) {
        const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(token_ids.len));
        for (accum) |*v| v.* *= inv_n;
    }

    // Store final pooled hidden state so `get_hidden_state_view` also reflects `embed`.
    for (0..h) |d| {
        eng.hidden_norm_f32[d] = accum[d];
        eng.hidden_norm_f16[d] = @floatCast(accum[d]);
    }
    eng.hidden_norm_f32_valid = true;
    eng.hidden_norm_f16_valid = true;
    eng.has_step_state = false;

    const dst = out_embedding.?[0..eff_dims];
    @memcpy(dst, accum[0..eff_dims]);

    // Matryoshka prefix slice is applied BEFORE L2 normalization (C21).
    if (params.normalize != 0) {
        var sq_sum: f64 = 0.0;
        for (dst) |v| {
            const vf: f64 = v;
            sq_sum += vf * vf;
        }
        const norm = @sqrt(sq_sum);
        if (norm > 1e-12) {
            const inv_norm: f32 = @floatCast(1.0 / norm);
            for (dst) |*v| v.* *= inv_norm;
        }
    }
    return AMI_SDK_OK;
}

pub export fn ami_sdk_embed_tokens(
    engine: ?*SdkEngine,
    token_ids: ?[*]const u32,
    token_len: usize,
    params_ptr: ?*const EmbedParams,
    out_embedding: ?[*]f32,
    out_cap: usize,
    out_dims: ?*usize,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const dims_out = out_dims orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_dims is NULL");
    dims_out.* = 0;
    if (token_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "token_len is 0");
    const tids = token_ids orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "token_ids is NULL");
    const params = params_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "params is NULL");

    return embedTokensImpl(eng, tids[0..token_len], params, out_embedding, out_cap, dims_out);
}

pub export fn ami_sdk_embed_text(
    engine: ?*SdkEngine,
    text_ptr: ?[*]const u8,
    text_len: usize,
    params_ptr: ?*const EmbedParams,
    out_embedding: ?[*]f32,
    out_cap: usize,
    out_dims: ?*usize,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const dims_out = out_dims orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_dims is NULL");
    dims_out.* = 0;
    const stok = eng.owned_tokenizer orelse return fail(AMI_SDK_ERR_TOKENIZER, "engine has no loaded tokenizer");
    if (text_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "text_len is 0");
    const tptr = text_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "text_ptr is NULL");
    const params = params_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "params is NULL");

    var tok_len: usize = 0;
    const tok_st = ami_sdk_tokenize(
        stok,
        tptr,
        text_len,
        1,
        eng.prompt_ids_scratch.ptr,
        eng.prompt_ids_scratch.len,
        &tok_len,
    );
    if (tok_st == AMI_SDK_ERR_BUFFER_TOO_SMALL) {
        return fail(AMI_SDK_ERR_SEQUENCE_FULL, "text exceeds max_seq_len");
    }
    if (tok_st != AMI_SDK_OK) return tok_st;
    return embedTokensImpl(eng, eng.prompt_ids_scratch[0..tok_len], params, out_embedding, out_cap, dims_out);
}

fn resolveRerankTokens(eng: *SdkEngine, params: *const RerankParams) error{InvalidTokenId}!struct { pos: u32, neg: u32 } {
    var pos_id = params.positive_token_id;
    var neg_id = params.negative_token_id;

    if (pos_id == TOKEN_NONE) {
        if (eng.owned_tokenizer) |stok| {
            var tmp: [4]u32 = undefined;
            var n: usize = 0;
            if (ami_sdk_tokenize(stok, "yes", 3, 0, &tmp, tmp.len, &n) == AMI_SDK_OK and n >= 1) {
                pos_id = tmp[0];
            }
        }
        if (pos_id == TOKEN_NONE) pos_id = @min(@as(u32, 1), eng.vocab_size - 1);
    }
    if (neg_id == TOKEN_NONE) {
        if (eng.owned_tokenizer) |stok| {
            var tmp: [4]u32 = undefined;
            var n: usize = 0;
            if (ami_sdk_tokenize(stok, "no", 2, 0, &tmp, tmp.len, &n) == AMI_SDK_OK and n >= 1) {
                neg_id = tmp[0];
            }
        }
        if (neg_id == TOKEN_NONE) neg_id = 0;
    }
    if (pos_id >= eng.vocab_size or neg_id >= eng.vocab_size) return error.InvalidTokenId;
    return .{ .pos = pos_id, .neg = neg_id };
}

fn computeRerankScore(logits: []const f16, mode: RerankMode, pos_id: u32, neg_id: u32) f32 {
    const pos_val: f32 = @floatCast(logits[pos_id]);
    const neg_val: f32 = @floatCast(logits[neg_id]);
    return switch (mode) {
        .logit_diff => pos_val - neg_val,
        .prob_yes => blk: {
            const diff = pos_val - neg_val;
            if (diff >= 0.0) {
                const e = @exp(-diff);
                break :blk 1.0 / (1.0 + e);
            } else {
                const e = @exp(diff);
                break :blk e / (1.0 + e);
            }
        },
        .raw_logit => pos_val,
    };
}

pub export fn ami_sdk_rerank_tokens(
    engine: ?*SdkEngine,
    query_ids_ptr: ?[*]const u32,
    query_len: usize,
    doc_id_ptrs: ?[*]const [*]const u32,
    doc_id_lens: ?[*]const usize,
    num_docs: usize,
    params_ptr: ?*const RerankParams,
    out_scores_ptr: ?[*]f32,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const params = params_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "params is NULL");
    if (params.struct_size != @sizeOf(RerankParams)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "RerankParams struct_size mismatch");
    }
    if (num_docs == 0) return AMI_SDK_OK;
    if (query_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "query_len is 0");
    const qptr = query_ids_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "query_ids is NULL");
    const dptrs = doc_id_ptrs orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "doc_id_ptrs is NULL");
    const dlens = doc_id_lens orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "doc_id_lens is NULL");
    const out_scores = out_scores_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_scores is NULL");

    const mode = parseRerankMode(params.score_mode) orelse {
        return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "invalid rerank score_mode");
    };
    const pair_ids = resolveRerankTokens(eng, params) catch {
        return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "rerank positive/negative token_id out of range");
    };

    const query_ids = qptr[0..query_len];
    if (query_len >= eng.max_seq_len) return fail(AMI_SDK_ERR_SEQUENCE_FULL, "query_len exceeds max_seq_len");

    // Reuse query prefix KV cache across all candidate documents when running pure GQA Qwen3
    // with zero KV retirement; otherwise run exact full prefill per pair.
    const can_reuse_query_kv = switch (eng.backend) {
        .qwen3 => |*m| m.kv_retire_after_chunks == 0,
        .graph => false,
    };

    if (can_reuse_query_kv) {
        const qst = prefillTokensInternal(eng, query_ids, false, true, params.cancel_token, null);
        if (qst != AMI_SDK_OK) return qst;
    }

    for (0..num_docs) |i| {
        if (SdkCancelToken.isCancelled(params.cancel_token)) {
            return fail(AMI_SDK_ERR_CANCELLED, "rerank cancelled");
        }
        const dlen = dlens[i];
        if (dlen == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "document token length is 0");
        const doc_ids = dptrs[i][0..dlen];

        if (can_reuse_query_kv) {
            eng.backend.qwen3.seq_len = @intCast(query_len);
            eng.history_len = @intCast(query_len);
            const dst_st = prefillTokensInternal(eng, doc_ids, true, false, params.cancel_token, null);
            if (dst_st != AMI_SDK_OK) return dst_st;
        } else {
            if (query_len + dlen > eng.prompt_ids_scratch.len) {
                return fail(AMI_SDK_ERR_SEQUENCE_FULL, "query + doc exceeds max_seq_len");
            }
            @memcpy(eng.prompt_ids_scratch[0..query_len], query_ids);
            @memcpy(eng.prompt_ids_scratch[query_len .. query_len + dlen], doc_ids);
            const pst = prefillTokensInternal(
                eng,
                eng.prompt_ids_scratch[0 .. query_len + dlen],
                true,
                true,
                params.cancel_token,
                null,
            );
            if (pst != AMI_SDK_OK) return pst;
        }
        out_scores[i] = computeRerankScore(eng.lastLogitsF16(), mode, pair_ids.pos, pair_ids.neg);
    }
    return AMI_SDK_OK;
}

pub export fn ami_sdk_rerank_texts(
    engine: ?*SdkEngine,
    query_ptr: ?[*]const u8,
    query_len: usize,
    doc_ptrs: ?[*]const [*]const u8,
    doc_lens: ?[*]const usize,
    num_docs: usize,
    params_ptr: ?*const RerankParams,
    out_scores_ptr: ?[*]f32,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const stok = eng.owned_tokenizer orelse return fail(AMI_SDK_ERR_TOKENIZER, "engine has no loaded tokenizer");
    const params = params_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "params is NULL");
    if (params.struct_size != @sizeOf(RerankParams)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "RerankParams struct_size mismatch");
    }
    if (num_docs == 0) return AMI_SDK_OK;
    if (query_len == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "query_len is 0");
    const qptr = query_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "query_ptr is NULL");
    const dptrs = doc_ptrs orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "doc_ptrs is NULL");
    const dlens = doc_lens orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "doc_lens is NULL");
    const out_scores = out_scores_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_scores is NULL");

    const mode = parseRerankMode(params.score_mode) orelse {
        return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "invalid rerank score_mode");
    };
    const pair_ids = resolveRerankTokens(eng, params) catch {
        return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "rerank positive/negative token_id out of range");
    };

    const prefix_hdr = "<|im_start|>system\nJudge whether the Document is relevant to the Query. Answer yes or no.<|im_end|>\n<|im_start|>user\n<Query>: ";
    const mid_sep = "\n<Document>: ";
    const suffix_ftr = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    const query_str = qptr[0..query_len];

    for (0..num_docs) |i| {
        if (SdkCancelToken.isCancelled(params.cancel_token)) {
            return fail(AMI_SDK_ERR_CANCELLED, "rerank cancelled");
        }
        const doc_str = dptrs[i][0..dlens[i]];
        const total_bytes = prefix_hdr.len + query_str.len + mid_sep.len + doc_str.len + suffix_ftr.len;
        if (total_bytes > eng.rerank_buf.len) {
            return fail(AMI_SDK_ERR_SEQUENCE_FULL, "formatted rerank prompt exceeds scratch capacity");
        }

        var cursor: usize = 0;
        @memcpy(eng.rerank_buf[cursor .. cursor + prefix_hdr.len], prefix_hdr);
        cursor += prefix_hdr.len;
        @memcpy(eng.rerank_buf[cursor .. cursor + query_str.len], query_str);
        cursor += query_str.len;
        @memcpy(eng.rerank_buf[cursor .. cursor + mid_sep.len], mid_sep);
        cursor += mid_sep.len;
        @memcpy(eng.rerank_buf[cursor .. cursor + doc_str.len], doc_str);
        cursor += doc_str.len;
        @memcpy(eng.rerank_buf[cursor .. cursor + suffix_ftr.len], suffix_ftr);
        cursor += suffix_ftr.len;

        var tok_len: usize = 0;
        const tst = ami_sdk_tokenize(
            stok,
            eng.rerank_buf.ptr,
            cursor,
            1,
            eng.prompt_ids_scratch.ptr,
            eng.prompt_ids_scratch.len,
            &tok_len,
        );
        if (tst == AMI_SDK_ERR_BUFFER_TOO_SMALL) {
            return fail(AMI_SDK_ERR_SEQUENCE_FULL, "rerank prompt exceeds max_seq_len");
        }
        if (tst != AMI_SDK_OK) return tst;

        const pst = prefillTokensInternal(
            eng,
            eng.prompt_ids_scratch[0..tok_len],
            true,
            true,
            params.cancel_token,
            null,
        );
        if (pst != AMI_SDK_OK) return pst;
        out_scores[i] = computeRerankScore(eng.lastLogitsF16(), mode, pair_ids.pos, pair_ids.neg);
    }
    return AMI_SDK_OK;
}

// ---------------------------------------------------------------------------
// Zero-Copy Vision Ingress (`G14`: CVPixelBufferRef / IOSurfaceRef / IOSurfaceID)
// ---------------------------------------------------------------------------

fn wrapExternalPixelBufferZeroCopy(
    dev: metal.Device,
    pixel_buffer: *anyopaque,
) error{ VisionFailed, UnsupportedFormat, OutOfMemory }!metal.VisionFrame {
    const abi_dev: *const AbiDeviceLayout = @ptrCast(@alignCast(dev.handle));
    const mtl_device = abi_dev.mtl_device orelse return error.VisionFailed;

    const cv_fmt = CVPixelBufferGetPixelFormatType(pixel_buffer);
    var fmt_code: u32 = 0;
    var plane_count: u32 = 0;
    var is_full_range: u32 = 1;
    var mtl_fmts: [2]usize = .{ 0, 0 };

    if (cv_fmt == CV_FMT_BGRA8) {
        fmt_code = bridge_c.AMI_VISION_FMT_BGRA8;
        plane_count = 1;
        is_full_range = 1;
        mtl_fmts[0] = MTL_PIXEL_FORMAT_BGRA8_UNORM;
    } else if (cv_fmt == CV_FMT_NV12_FULL) {
        fmt_code = bridge_c.AMI_VISION_FMT_NV12;
        plane_count = 2;
        is_full_range = 1;
        mtl_fmts[0] = MTL_PIXEL_FORMAT_R8_UNORM;
        mtl_fmts[1] = MTL_PIXEL_FORMAT_RG8_UNORM;
    } else if (cv_fmt == CV_FMT_NV12_VIDEO) {
        fmt_code = bridge_c.AMI_VISION_FMT_NV12;
        plane_count = 2;
        is_full_range = 0;
        mtl_fmts[0] = MTL_PIXEL_FORMAT_R8_UNORM;
        mtl_fmts[1] = MTL_PIXEL_FORMAT_RG8_UNORM;
    } else if (cv_fmt == CV_FMT_R16F) {
        fmt_code = bridge_c.AMI_VISION_FMT_F16;
        plane_count = 1;
        is_full_range = 1;
        mtl_fmts[0] = MTL_PIXEL_FORMAT_R16_FLOAT;
    } else {
        return error.UnsupportedFormat;
    }

    const surface = CVPixelBufferGetIOSurface(pixel_buffer) orelse return error.VisionFailed;
    const width = CVPixelBufferGetWidth(pixel_buffer);
    const height = CVPixelBufferGetHeight(pixel_buffer);
    if (width == 0 or height == 0 or width > metal.MAX_TEXTURE_DIM or height > metal.MAX_TEXTURE_DIM) {
        return error.VisionFailed;
    }

    if (CVPixelBufferLockBaseAddress(pixel_buffer, CV_LOCK_READ_ONLY) != CV_RETURN_SUCCESS) {
        return error.VisionFailed;
    }
    const base_addr = IOSurfaceGetBaseAddress(surface);
    const surface_alloc = IOSurfaceGetAllocSize(surface);
    const sid = IOSurfaceGetID(surface);

    var plane_w: [2]u32 = .{ 0, 0 };
    var plane_h: [2]u32 = .{ 0, 0 };
    var plane_bpr: [2]u32 = .{ 0, 0 };
    var planes_ok = true;

    if (plane_count == 1) {
        plane_w[0] = @intCast(width);
        plane_h[0] = @intCast(height);
        const bpr = CVPixelBufferGetBytesPerRow(pixel_buffer);
        if (bpr == 0) planes_ok = false else plane_bpr[0] = @intCast(bpr);
    } else {
        if (CVPixelBufferGetPlaneCount(pixel_buffer) < 2) {
            planes_ok = false;
        } else {
            for (0..2) |p| {
                const pw = CVPixelBufferGetWidthOfPlane(pixel_buffer, p);
                const ph = CVPixelBufferGetHeightOfPlane(pixel_buffer, p);
                const pbpr = CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, p);
                if (pw == 0 or ph == 0 or pbpr == 0 or pw > metal.MAX_TEXTURE_DIM or ph > metal.MAX_TEXTURE_DIM) {
                    planes_ok = false;
                    break;
                }
                plane_w[p] = @intCast(pw);
                plane_h[p] = @intCast(ph);
                plane_bpr[p] = @intCast(pbpr);
            }
        }
    }
    _ = CVPixelBufferUnlockBaseAddress(pixel_buffer, CV_LOCK_READ_ONLY);

    const raw_base = base_addr orelse return error.VisionFailed;
    if (!planes_ok or surface_alloc == 0) return error.VisionFailed;
    const base_int = @intFromPtr(raw_base);
    const base_aligned = (base_int % page.APPLE_PAGE_SIZE == 0);
    const size_aligned = (surface_alloc % page.APPLE_PAGE_SIZE == 0);
    if (!base_aligned) return error.VisionFailed;

    const desc_cls = objc_getClass("MTLTextureDescriptor") orelse return error.VisionFailed;
    const sel_desc = sel_registerName("texture2DDescriptorWithPixelFormat:width:height:mipmapped:") orelse return error.VisionFailed;
    const sel_storage = sel_registerName("setStorageMode:") orelse return error.VisionFailed;
    const sel_usage = sel_registerName("setUsage:") orelse return error.VisionFailed;
    const sel_new_tex = sel_registerName("newTextureWithDescriptor:iosurface:plane:") orelse return error.VisionFailed;
    const sel_iosurface = sel_registerName("iosurface") orelse return error.VisionFailed;

    const MsgDescFn = *const fn (?*anyopaque, ?*anyopaque, usize, usize, usize, bool) callconv(.c) ?*anyopaque;
    const MsgSetUsizeFn = *const fn (?*anyopaque, ?*anyopaque, usize) callconv(.c) void;
    const MsgNewTexFn = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, *anyopaque, usize) callconv(.c) ?*anyopaque;
    const MsgGetPtrFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;

    const msgDesc: MsgDescFn = @ptrCast(&objc_msgSend);
    const msgSetUsize: MsgSetUsizeFn = @ptrCast(&objc_msgSend);
    const msgNewTex: MsgNewTexFn = @ptrCast(&objc_msgSend);
    const msgGetPtr: MsgGetPtrFn = @ptrCast(&objc_msgSend);

    var mtl_textures: [2]?*anyopaque = .{ null, null };
    errdefer for (mtl_textures) |maybe_t| {
        if (maybe_t) |t| CFRelease(t);
    };

    for (0..plane_count) |p| {
        const desc = msgDesc(desc_cls, sel_desc, mtl_fmts[p], plane_w[p], plane_h[p], false) orelse return error.VisionFailed;
        msgSetUsize(desc, sel_storage, MTL_STORAGE_MODE_SHARED);
        msgSetUsize(desc, sel_usage, MTL_TEXTURE_USAGE_SHADER_READ);
        const tex = msgNewTex(mtl_device, sel_new_tex, desc, surface, p) orelse return error.VisionFailed;
        const tex_surface = msgGetPtr(tex, sel_iosurface) orelse {
            CFRelease(tex);
            return error.VisionFailed;
        };
        if (tex_surface != surface or IOSurfaceGetID(tex_surface) != sid) {
            CFRelease(tex);
            return error.VisionFailed;
        }
        mtl_textures[p] = tex;
    }

    const raw_mem = std.c.malloc(@sizeOf(AbiVisionFrameLayout)) orelse return error.OutOfMemory;
    const raw_frame: *AbiVisionFrameLayout = @ptrCast(@alignCast(raw_mem));
    _ = CVPixelBufferRetain(pixel_buffer);

    raw_frame.* = .{
        .pixel_buffer = pixel_buffer,
        .surface = surface,
        .mtl_textures = mtl_textures,
        .format = fmt_code,
        .cv_pixel_format = cv_fmt,
        .width = @intCast(width),
        .height = @intCast(height),
        .plane_count = plane_count,
        .is_full_range = is_full_range,
        .zero_copy_verified = 1,
        .is_page_aligned = if (base_aligned and size_aligned) 1 else 0,
        .iosurface_id = sid,
        .vt_decoded = 0,
        .base_address = @intCast(base_int),
        .alloc_size = @intCast(surface_alloc),
        .surface_alloc_size = @intCast(surface_alloc),
        .plane_width = plane_w,
        .plane_height = plane_h,
        .plane_bytes_per_row = plane_bpr,
        .pts_value = 0,
        .pts_timescale = 0,
    };
    return .{ .handle = @ptrCast(raw_frame) };
}

pub export fn ami_sdk_vision_from_cvpixelbuffer(
    engine: ?*SdkEngine,
    cv_pixel_buffer_ref: ?*anyopaque,
    out_vision: ?*?*SdkVisionInput,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const out = out_vision orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_vision is NULL");
    out.* = null;
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const pb = cv_pixel_buffer_ref orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "cv_pixel_buffer_ref is NULL");

    const frame = wrapExternalPixelBufferZeroCopy(eng.metalDevice(), pb) catch |err| switch (err) {
        error.UnsupportedFormat => return fail(AMI_SDK_ERR_UNSUPPORTED, "unsupported CVPixelBuffer format"),
        error.OutOfMemory => return fail(AMI_SDK_ERR_OUT_OF_MEMORY, "vision handle alloc failed"),
        error.VisionFailed => return fail(AMI_SDK_ERR_VISION, "CVPixelBuffer must be 16 KiB-aligned IOSurface-backed"),
    };
    errdefer frame.close();

    const vis = gpa.create(SdkVisionInput) catch return fail(AMI_SDK_ERR_OUT_OF_MEMORY, "SdkVisionInput alloc failed");
    vis.* = .{ .frame = frame };
    out.* = vis;
    return AMI_SDK_OK;
}

pub export fn ami_sdk_vision_from_iosurface(
    engine: ?*SdkEngine,
    iosurface_ref: ?*anyopaque,
    out_vision: ?*?*SdkVisionInput,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const out = out_vision orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_vision is NULL");
    out.* = null;
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const surface = iosurface_ref orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "iosurface_ref is NULL");

    var pb: ?*anyopaque = null;
    if (CVPixelBufferCreateWithIOSurface(null, surface, null, &pb) != CV_RETURN_SUCCESS or pb == null) {
        return fail(AMI_SDK_ERR_VISION, "CVPixelBufferCreateWithIOSurface failed");
    }
    defer CVPixelBufferRelease(pb.?);

    return ami_sdk_vision_from_cvpixelbuffer(eng, pb.?, out);
}

pub export fn ami_sdk_vision_from_iosurface_id(
    engine: ?*SdkEngine,
    iosurface_id: u32,
    out_vision: ?*?*SdkVisionInput,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const out = out_vision orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_vision is NULL");
    out.* = null;
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    if (iosurface_id == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "iosurface_id is 0");

    const surface = IOSurfaceLookup(iosurface_id) orelse {
        return fail(AMI_SDK_ERR_VISION, "IOSurfaceLookup returned NULL");
    };
    defer CFRelease(surface);

    return ami_sdk_vision_from_iosurface(eng, surface, out);
}

pub export fn ami_sdk_vision_create_iosurface(
    engine: ?*SdkEngine,
    format: u32,
    width: u32,
    height: u32,
    out_vision: ?*?*SdkVisionInput,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const out = out_vision orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_vision is NULL");
    out.* = null;
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const fmt: metal.VisionFormat = switch (format) {
        sdk_c.AMI_SDK_VISION_FMT_BGRA8 => .bgra8,
        sdk_c.AMI_SDK_VISION_FMT_NV12 => .nv12,
        sdk_c.AMI_SDK_VISION_FMT_F16 => .f16,
        else => return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "unsupported vision format"),
    };

    const frame = metal.VisionFrame.init(eng.metalDevice(), fmt, width, height) catch |err| return mapAnyError(err);
    errdefer frame.close();

    const vis = gpa.create(SdkVisionInput) catch return fail(AMI_SDK_ERR_OUT_OF_MEMORY, "SdkVisionInput alloc failed");
    vis.* = .{ .frame = frame };
    out.* = vis;
    return AMI_SDK_OK;
}

pub export fn ami_sdk_vision_load_image(
    engine: ?*SdkEngine,
    image_path: ?[*:0]const u8,
    out_vision: ?*?*SdkVisionInput,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const out = out_vision orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_vision is NULL");
    out.* = null;
    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const raw_path = image_path orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "image_path is NULL");

    const frame = metal.VisionFrame.loadImage(eng.metalDevice(), std.mem.span(raw_path)) catch |err| return mapAnyError(err);
    errdefer frame.close();

    const vis = gpa.create(SdkVisionInput) catch return fail(AMI_SDK_ERR_OUT_OF_MEMORY, "SdkVisionInput alloc failed");
    vis.* = .{ .frame = frame };
    out.* = vis;
    return AMI_SDK_OK;
}

pub export fn ami_sdk_vision_destroy(vision: ?*SdkVisionInput) callconv(.c) void {
    const vis = vision orelse return;
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);
    vis.frame.close();
    gpa.destroy(vis);
}

pub export fn ami_sdk_vision_query(
    vision: ?*const SdkVisionInput,
    out_info: ?*VisionInfo,
) callconv(.c) Status {
    const vis = vision orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "vision is NULL");
    const info = out_info orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_info is NULL");
    if (info.struct_size != 0 and info.struct_size < @sizeOf(VisionInfo)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "VisionInfo struct_size mismatch");
    }
    const finfo = vis.frame.info() catch |err| return mapAnyError(err);
    const fmt_out: u32 = switch (finfo.format()) {
        .bgra8 => sdk_c.AMI_SDK_VISION_FMT_BGRA8,
        .nv12 => sdk_c.AMI_SDK_VISION_FMT_NV12,
        .f16 => sdk_c.AMI_SDK_VISION_FMT_F16,
    };
    info.* = .{
        .struct_size = @sizeOf(VisionInfo),
        .format = fmt_out,
        .cv_pixel_format = finfo.raw.cv_pixel_format,
        .width = finfo.raw.width,
        .height = finfo.raw.height,
        .plane_count = finfo.raw.plane_count,
        .iosurface_id = finfo.raw.iosurface_id,
        .zero_copy_verified = finfo.raw.zero_copy_verified,
        .is_page_aligned = finfo.raw.is_page_aligned,
        .reserved = 0,
        .base_address = finfo.raw.base_address,
        .surface_alloc_size = finfo.raw.surface_alloc_size,
    };
    return AMI_SDK_OK;
}

pub export fn ami_sdk_vision_cvpixelbuffer(vision: ?*const SdkVisionInput) callconv(.c) ?*anyopaque {
    const vis = vision orelse return null;
    const raw: *const AbiVisionFrameLayout = @ptrCast(@alignCast(vis.frame.handle));
    return raw.pixel_buffer;
}

pub export fn ami_sdk_vision_iosurface(vision: ?*const SdkVisionInput) callconv(.c) ?*anyopaque {
    const vis = vision orelse return null;
    const raw: *const AbiVisionFrameLayout = @ptrCast(@alignCast(vis.frame.handle));
    return raw.surface;
}

pub export fn ami_sdk_vision_lock_plane(
    vision: ?*SdkVisionInput,
    plane: u32,
    read_only: u32,
    out_base_ptr: ?*?*anyopaque,
    out_bytes_per_row: ?*usize,
    out_height: ?*u32,
) callconv(.c) Status {
    const vis = vision orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "vision is NULL");
    const bptr = out_base_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_base_ptr is NULL");
    const bpr = out_bytes_per_row orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_bytes_per_row is NULL");
    const hgt = out_height orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "out_height is NULL");

    const rc = bridge_c.ami_vision_frame_lock_plane(vis.frame.handle, plane, if (read_only != 0) 1 else 0, bptr, bpr, hgt);
    if (rc != bridge_c.AMI_OK) return fail(AMI_SDK_ERR_VISION, "vision lock_plane failed");
    return AMI_SDK_OK;
}

pub export fn ami_sdk_vision_unlock(
    vision: ?*SdkVisionInput,
    read_only: u32,
) callconv(.c) Status {
    const vis = vision orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "vision is NULL");
    const rc = bridge_c.ami_vision_frame_unlock(vis.frame.handle, if (read_only != 0) 1 else 0);
    if (rc != bridge_c.AMI_OK) return fail(AMI_SDK_ERR_VISION, "vision unlock failed");
    return AMI_SDK_OK;
}

fn prefillVisionQwen3(
    eng: *SdkEngine,
    m: *qwen3.Qwen3Model,
    prefix_ids: []const u32,
    frames: []const metal.VisionFrame,
    suffix_ids: []const u32,
    params: *const VisionParams,
    out_res: *VisionResult,
) !void {
    if (frames.len == 0) return error.EmptyFrames;
    const info0 = try frames[0].info();
    const fmt0 = info0.format();
    if (fmt0 != .bgra8 and fmt0 != .nv12) return error.UnsupportedVisionFormat;
    const width = info0.width();
    const height = info0.height();
    const is_full_range = info0.isFullRange();

    var all_zc = info0.zeroCopyVerified();
    var all_aligned = (info0.baseAddress() % page.APPLE_PAGE_SIZE) == 0;
    for (frames[1..]) |fr| {
        const inf = try fr.info();
        if (inf.width() != width or inf.height() != height or inf.format() != fmt0) {
            return error.FrameGeometryMismatch;
        }
        all_zc = all_zc and inf.zeroCopyVerified();
        all_aligned = all_aligned and ((inf.baseAddress() % page.APPLE_PAGE_SIZE) == 0);
    }

    const eff_t_patch: u32 = if (params.t_patch == 2 and frames.len >= 2) 2 else 1;
    const tube_count: u32 = @intCast((frames.len + eff_t_patch - 1) / eff_t_patch);
    const geom: vision_mod.PatchGeometry = .{
        .width = width,
        .height = height,
        .patch_size = params.patch_size,
        .t_patch = eff_t_patch,
        .spatial_merge = params.spatial_merge,
    };
    try geom.validate();
    const tokens_per_tube = try geom.numTokens();
    const total_vis_tokens = std.math.mul(u32, tube_count, tokens_per_tube) catch return error.Overflow;
    const token_in_dim = try geom.tokenInDim();

    const prefix_u32: u32 = @intCast(prefix_ids.len);
    const suffix_u32: u32 = @intCast(suffix_ids.len);
    const total_tokens = prefix_u32 + total_vis_tokens + suffix_u32;
    if (total_tokens == 0) return error.InvalidArgument;
    if (total_tokens > m.max_seq_len) return error.SequenceFull;

    const grid: vision_mod.VisionGridSpec = .{
        .grid_t = tube_count,
        .grid_h = geom.mergedH(),
        .grid_w = geom.mergedW(),
    };

    const facts = try sysinfo.Facts.read();
    const dev_info = try m.device.info();
    const vis_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    var ingress = try vision_mod.VisionIngress.init(m.device, m.library, vis_budget, .{
        .max_tokens = total_vis_tokens,
        .max_patch_dim = token_in_dim,
        .max_hidden_size = m.config.hidden_size,
        .max_mrope_tokens = total_tokens,
    });
    defer ingress.deinit();

    const proj_elems = std.math.mul(usize, m.config.hidden_size, token_in_dim) catch return error.Overflow;
    const proj_bytes = try page.alignUp(proj_elems * @sizeOf(f16));
    var proj_arena = try arena_mod.WiredArena.init(proj_bytes, vis_budget);
    defer proj_arena.deinit();
    const proj_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, proj_arena.bytes()[0 .. proj_elems * @sizeOf(f16)]));
    const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(token_in_dim)));
    for (proj_f16, 0..) |*w, idx| {
        const h_mix = (@as(u64, idx) *% 0x9e3779b97f4a7c15) ^ 0x517cc1b727220a95;
        const unit: f32 = (@as(f32, @floatFromInt(h_mix & 0xFFFF)) / 32767.5) - 1.0;
        w.* = @floatCast(unit * inv_scale * 0.25);
    }
    const proj_buf = try metal.Buffer.wrap(m.device, proj_arena.bytes());
    defer proj_buf.close();

    try m.compute.makeResident(&.{ ingress.buffer(), proj_buf });
    defer m.compute.removeResident(&.{ ingress.buffer(), proj_buf }) catch |err| {
        std.debug.panic("removeResident(sdk_vision) failed: {}", .{err});
    };
    try m.compute.makeVisionFramesResident(frames);
    defer m.compute.removeVisionFramesResident(frames) catch |err| {
        std.debug.panic("removeVisionFramesResident(sdk_vision) failed: {}", .{err});
    };

    const norm: vision_mod.NormParams = .{
        .mean = params.mean_rgb,
        .inv_std = params.inv_std_rgb,
    };

    const vis_t0 = try vmstat.nowNs();
    {
        var vframe = try m.compute.beginFrame(m.timeout_ms);
        errdefer vframe.abort();
        var tube: u32 = 0;
        while (tube < tube_count) : (tube += 1) {
            const f0_idx: usize = @as(usize, tube) * eff_t_patch;
            const f1_opt: ?metal.VisionFrame = if (eff_t_patch == 2)
                frames[@min(f0_idx + 1, frames.len - 1)]
            else
                null;
            const tok_off: u32 = tube * tokens_per_tube;
            const proj_bind: metal.BufferBinding = .{ .buffer = proj_buf, .offset = 0 };
            switch (fmt0) {
                .bgra8 => try ingress.encodePatchifyEmbedBgra8(
                    &vframe,
                    frames[f0_idx],
                    f1_opt,
                    proj_bind,
                    null,
                    .f16,
                    .f16,
                    m.config.hidden_size,
                    geom,
                    norm,
                    tok_off,
                    tube > 0,
                ),
                .nv12 => try ingress.encodePatchifyEmbedNv12(
                    &vframe,
                    frames[f0_idx],
                    f1_opt,
                    proj_bind,
                    null,
                    .f16,
                    .f16,
                    m.config.hidden_size,
                    geom,
                    is_full_range,
                    norm,
                    tok_off,
                    tube > 0,
                ),
                else => unreachable,
            }
        }
        const vsig = try vframe.commit();
        try m.compute.wait(vsig, m.timeout_ms);
    }
    const vis_t1 = try vmstat.nowNs();
    const vision_encode_us: f64 = @as(f64, @floatFromInt(vis_t1 -| vis_t0)) / 1000.0;

    _ = try ingress.buildMropePositions(prefix_u32, grid, suffix_u32);

    // Execute `[prefix | vision_embeddings | suffix]` through Qwen3 layers on GPU,
    // injecting `ingress.embedOutBinding()` directly into `x_bind` for visual tokens (`0` host copy).
    eng.resetKv();
    const h = m.config.hidden_size;
    const compute_logits = (params.compute_logits != 0);
    var first_token: u32 = 0;

    if (prefix_u32 > 0) {
        first_token = try m.prefillChunkWithOptions(prefix_ids, 0, .{
            .compute_logits = compute_logits and total_vis_tokens == 0 and suffix_u32 == 0,
            .preserve_all_top_hidden = total_vis_tokens == 0 and suffix_u32 == 0,
        });
        for (prefix_ids) |tid| eng.recordToken(tid);
    }

    // Stream each visual token embedding `[1, h] f16` from `ingress.embedOutBinding()` into `m.x_bind`
    // and run Qwen3's GQA + MoE/Dense layers on GPU (`0` host pixel or embedding memcpy).
    for (0..total_vis_tokens) |vi| {
        const pos: u32 = prefix_u32 + @as(u32, @intCast(vi));
        const is_final_overall = (suffix_u32 == 0 and vi + 1 == total_vis_tokens);
        const want_logits = compute_logits and is_final_overall;

        var frame = try m.compute.beginFrame(m.timeout_ms);
        errdefer frame.abort();

        try m.moe_pipes.encodeCopyStrided(&frame, .{
            .src = .{
                .buffer = ingress.embedOutBinding().buffer,
                .offset = ingress.embedOutBinding().offset + @as(u64, @intCast(vi)) * h * @sizeOf(f16),
            },
            .dst = m.x_bind,
            .num_rows = 1,
            .copy_cols = h,
            .src_stride = h,
            .dst_stride = h,
        }, false);

        for (m.layers) |layer| {
            try m.attn_pipes.encodeRmsNorm(&frame, .{
                .x = m.x_bind,
                .w = layer.input_layernorm.binding,
                .y = m.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = h,
                .eps = m.config.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);

            try m.encodeLinearGemv(&frame, layer.q_proj, m.x_norm_bind, m.q_bind, true);
            try m.encodeLinearGemv(&frame, layer.k_proj, m.x_norm_bind, m.k_bind, false);
            try m.encodeLinearGemv(&frame, layer.v_proj, m.x_norm_bind, m.v_bind, false);

            try m.attn_pipes.encodePrefillQkNormRopeKvWrite(&frame, .{
                .q = m.q_bind,
                .k = m.k_bind,
                .v = m.v_bind,
                .q_norm = layer.q_norm,
                .k_norm = layer.k_norm,
                .kv_tiles = layer.kv_tiles_binding,
                .num_tokens = 1,
                .num_q_heads = m.config.num_attention_heads,
                .num_kv_heads = m.config.num_key_value_heads,
                .head_dim = m.config.head_dim,
                .start_pos = pos,
                .chunk_tokens = m.chunk_tokens,
                .num_retired_chunks = m.num_retired_chunks,
                .kv_only = false,
                .rms_eps = m.config.rms_norm_eps,
                .rope_theta = m.config.rope_theta,
                .norm_dtype = layer.qk_norm_dtype,
            }, true);

            try m.attn_pipes.encodeGqaCausalPrefillFlashAttn(&frame, .{
                .q = m.q_bind,
                .kv_tiles = layer.kv_tiles_binding,
                .retired_pool = null,
                .retired_chunk_ids = null,
                .out = m.attn_out_bind,
                .num_q = 1,
                .q_start_pos = pos,
                .kv_len = pos + 1,
                .num_q_heads = m.config.num_attention_heads,
                .num_kv_heads = m.config.num_key_value_heads,
                .head_dim = m.config.head_dim,
                .chunk_tokens = m.chunk_tokens,
                .num_retired_chunks = 0,
                .retired_plane_stride_bytes = 0,
                .retired_chunk_stride_bytes = 0,
            }, true);

            try m.encodeLinearGemv(&frame, layer.o_proj, m.attn_out_bind, m.proj_out_bind, true);
            try m.moe_pipes.encodeResidualAdd(&frame, .{
                .x = m.x_bind,
                .delta = m.proj_out_bind,
                .count = h,
            }, true);

            try m.attn_pipes.encodeRmsNorm(&frame, .{
                .x = m.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = m.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = h,
                .eps = m.config.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);

            switch (m.config.arch) {
                .qwen3_moe => {
                    const moe_args: moe.MoELayerArgs = .{
                        .x = m.x_norm_bind,
                        .router = layer.router,
                        .weights_base = .{ .buffer = layer.expert_weights_buf, .offset = 0 },
                        .expert_table = layer.expert_table,
                        .workspace = &(m.moe_ws.?),
                        .workspace_buf = m.scratch_buf,
                        .residual = m.x_bind,
                        .out = m.x_bind,
                        .m = 1,
                        .num_experts = m.config.num_experts,
                        .top_k = m.config.num_experts_per_tok,
                        .hidden_size = h,
                        .intermediate_size = m.config.moe_intermediate_size,
                        .norm_topk_prob = m.config.norm_topk_prob,
                    };
                    if (layer.is_heterogeneous_moe) {
                        try m.moe_pipes.encodeMoELayerHetero(&frame, .{
                            .base = moe_args,
                            .expert_schemes = layer.expert_schemes_bind,
                            .scheme_counts = layer.expert_scheme_counts,
                        }, true);
                    } else {
                        switch (layer.expert_dtype) {
                            .q2_affine => try m.moe_pipes.encodeMoELayer(&frame, .q2, moe_args, true),
                            .q3_affine => try m.moe_pipes.encodeMoELayer(&frame, .q3, moe_args, true),
                            .q4_affine => try m.moe_pipes.encodeMoELayer(&frame, .q4, moe_args, true),
                            .f16 => try m.moe_pipes.encodeMoELayerFloat(&frame, .f16, moe_args, true),
                            .bf16 => try m.moe_pipes.encodeMoELayerFloat(&frame, .bf16, moe_args, true),
                        }
                    }
                },
                .qwen3_dense => {
                    try m.encodeLinearGemv(&frame, layer.dense_gate, m.x_norm_bind, m.dense_gate_bind, true);
                    try m.encodeLinearGemv(&frame, layer.dense_up, m.x_norm_bind, m.dense_up_bind, false);
                    try m.moe_pipes.encodeSiluMul(&frame, .{
                        .gate = m.dense_gate_bind,
                        .up = m.dense_up_bind,
                        .out = m.dense_act_bind,
                        .count = m.config.intermediate_size,
                    }, true);
                    try m.encodeLinearGemv(&frame, layer.dense_down, m.dense_act_bind, m.proj_out_bind, true);
                    try m.moe_pipes.encodeResidualAdd(&frame, .{
                        .x = m.x_bind,
                        .delta = m.proj_out_bind,
                        .count = h,
                    }, true);
                },
            }
        }

        if (want_logits) {
            try m.attn_pipes.encodeRmsNorm(&frame, .{
                .x = m.x_bind,
                .w = m.final_norm.binding,
                .y = m.x_norm_bind,
                .num_rows = 1,
                .hidden_dim = h,
                .eps = m.config.rms_norm_eps,
                .w_dtype = m.final_norm.dtype,
            }, true);
            try m.encodeLinearGemv(&frame, m.lm_head, m.x_norm_bind, m.logits_bind, true);
            try m.moe_pipes.encodeArgmaxRows(&frame, .{
                .logits = m.logits_bind,
                .out_token_ids = m.out_token_id_bind,
                .num_rows = 1,
                .vocab_size = m.config.vocab_size,
                .logits_dtype = .f16,
            }, true);
        }

        const sig = try frame.commit();
        try m.compute.wait(sig, m.timeout_ms);
        m.seq_len = pos + 1;
        eng.recordToken(0);

        if (want_logits) {
            m.last_logits_row = 0;
            first_token = m.out_token_ids_host[0];
        }
    }

    if (suffix_u32 > 0) {
        first_token = try m.prefillChunkWithOptions(suffix_ids, prefix_u32 + total_vis_tokens, .{
            .compute_logits = compute_logits,
            .preserve_all_top_hidden = true,
        });
        for (suffix_ids) |tid| eng.recordToken(tid);
    }

    if (compute_logits) {
        eng.has_step_state = true;
        captureLatestHiddenState(eng, if (suffix_u32 > 0) suffix_u32 else 1);
    }

    out_res.* = .{
        .struct_size = @sizeOf(VisionResult),
        .first_token_id = first_token,
        .prefix_tokens = prefix_u32,
        .vision_tokens = total_vis_tokens,
        .suffix_tokens = suffix_u32,
        .total_prompt_tokens = total_tokens,
        .grid_t = grid.grid_t,
        .grid_h = grid.grid_h,
        .grid_w = grid.grid_w,
        .all_frames_zero_copy = if (all_zc) 1 else 0,
        .all_frames_page_aligned = if (all_aligned) 1 else 0,
        .reserved = 0,
        .vision_encode_us = vision_encode_us,
    };
}

pub export fn ami_sdk_prefill_vision(
    engine: ?*SdkEngine,
    prefix_ids: ?[*]const u32,
    prefix_len: usize,
    frames_ptr: ?[*]const *const SdkVisionInput,
    frame_count: usize,
    suffix_ids: ?[*]const u32,
    suffix_len: usize,
    params_ptr: ?*const VisionParams,
    out_result: ?*VisionResult,
) callconv(.c) Status {
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);

    const eng = engine orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "engine is NULL");
    const params = params_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "params is NULL");
    if (params.struct_size != @sizeOf(VisionParams)) {
        return fail(AMI_SDK_ERR_ABI_MISMATCH, "VisionParams struct_size mismatch");
    }
    if (frame_count == 0) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "frame_count is 0");
    const fptrs = frames_ptr orelse return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "frames is NULL");
    if (prefix_len > 0 and prefix_ids == null) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "prefix_ids is NULL");
    if (suffix_len > 0 and suffix_ids == null) return fail(AMI_SDK_ERR_INVALID_ARGUMENT, "suffix_ids is NULL");
    if (SdkCancelToken.isCancelled(params.cancel_token)) return fail(AMI_SDK_ERR_CANCELLED, "vision prefill cancelled");

    const prefix_slice: []const u32 = if (prefix_len > 0) prefix_ids.?[0..prefix_len] else &.{};
    const suffix_slice: []const u32 = if (suffix_len > 0) suffix_ids.?[0..suffix_len] else &.{};

    var stack_frames: [32]metal.VisionFrame = undefined;
    var heap_frames: ?[]metal.VisionFrame = null;
    defer if (heap_frames) |hf| gpa.free(hf);

    const mframes: []metal.VisionFrame = if (frame_count <= stack_frames.len)
        stack_frames[0..frame_count]
    else blk: {
        heap_frames = gpa.alloc(metal.VisionFrame, frame_count) catch {
            return fail(AMI_SDK_ERR_OUT_OF_MEMORY, "vision frame list alloc failed");
        };
        break :blk heap_frames.?;
    };
    for (0..frame_count) |i| mframes[i] = fptrs[i].frame;

    eng.invalidateViews();
    var local_res: VisionResult = undefined;

    switch (eng.backend) {
        .graph => |*gm| {
            const gres = gm.prefillMultimodal(gpa, prefix_slice, suffix_slice, mframes, .{
                .patch_size = params.patch_size,
                .t_patch = params.t_patch,
                .spatial_merge = params.spatial_merge,
                .norm = .{
                    .mean = params.mean_rgb,
                    .inv_std = params.inv_std_rgb,
                },
                .compute_logits = params.compute_logits != 0,
            }) catch |err| return mapAnyError(err);

            if (params.compute_logits != 0) {
                eng.has_step_state = true;
                captureLatestHiddenState(eng, 1);
            }
            local_res = .{
                .struct_size = @sizeOf(VisionResult),
                .first_token_id = gres.first_token,
                .prefix_tokens = gres.prefix_tokens,
                .vision_tokens = gres.vision_tokens,
                .suffix_tokens = gres.suffix_tokens,
                .total_prompt_tokens = gres.total_prompt_tokens,
                .grid_t = gres.grid.grid_t,
                .grid_h = gres.grid.grid_h,
                .grid_w = gres.grid.grid_w,
                .all_frames_zero_copy = if (gres.all_frames_zero_copy) 1 else 0,
                .all_frames_page_aligned = if (gres.all_frames_page_aligned) 1 else 0,
                .reserved = 0,
                .vision_encode_us = gres.vision_encode_us,
            };
        },
        .qwen3 => |*qm| {
            prefillVisionQwen3(eng, qm, prefix_slice, mframes, suffix_slice, params, &local_res) catch |err| {
                return mapAnyError(err);
            };
        },
    }

    if (out_result) |dst| dst.* = local_res;
    return AMI_SDK_OK;
}

// ---------------------------------------------------------------------------
// Zig Unit Tests (`zig build test`)
// ---------------------------------------------------------------------------

test "sdk.c_api: ABI struct sizes and fragmented <think> + <tool_call> stream classification" {
    try std.testing.expectEqual(@as(u32, 1), ami_sdk_abi_version());
    try std.testing.expectEqualStrings("AMI_SDK_OK", std.mem.span(ami_sdk_status_string(AMI_SDK_OK)));

    const Record = struct {
        kind: DeltaKind,
        text: []const u8,
    };
    const Sink = struct {
        buf: [512]u8 = undefined,
        used: usize = 0,
        recs: [32]Record = undefined,
        count: usize = 0,
        utf8: Utf8Assembler = .{},
        cls: StreamTagClassifier = .{},

        fn onUtf8(self: *@This(), s: []const u8) bool {
            return self.cls.feed(s, self, onKind);
        }

        fn onKind(self: *@This(), k: DeltaKind, s: []const u8) bool {
            const dst = self.buf[self.used .. self.used + s.len];
            @memcpy(dst, s);
            self.used += s.len;
            self.recs[self.count] = .{ .kind = k, .text = dst };
            self.count += 1;
            return true;
        }
    };

    var sink: Sink = .{};
    sink.cls.reset(false);
    // Feed 1..2 bytes at a time including split multi-byte UTF-8 ("你好"), false `<to` prefix,
    // `<think>...</think>`, and `<tool_call>...</tool_call>`.
    const stream = "<think>plan 你好</think>Hi <to!<tool_call>{\"name\":\"ls\"}</tool_call>";
    var idx: usize = 0;
    while (idx < stream.len) {
        const step: usize = if ((idx % 2) == 0) 1 else @min(@as(usize, 2), stream.len - idx);
        try std.testing.expect(sink.utf8.feed(stream[idx .. idx + step], &sink, Sink.onUtf8));
        idx += step;
    }
    try std.testing.expect(sink.utf8.flush(&sink, Sink.onUtf8));
    try std.testing.expect(sink.cls.flush(&sink, Sink.onKind));
    try std.testing.expect(sink.cls.saw_tool_call);

    var reasoning_buf: [64]u8 = undefined;
    var r_len: usize = 0;
    var text_buf: [64]u8 = undefined;
    var t_len: usize = 0;
    var tool_buf: [64]u8 = undefined;
    var tc_len: usize = 0;
    var tool_ends: usize = 0;

    for (sink.recs[0..sink.count]) |rec| {
        switch (rec.kind) {
            .reasoning => {
                @memcpy(reasoning_buf[r_len .. r_len + rec.text.len], rec.text);
                r_len += rec.text.len;
            },
            .text => {
                @memcpy(text_buf[t_len .. t_len + rec.text.len], rec.text);
                t_len += rec.text.len;
            },
            .tool_call => {
                @memcpy(tool_buf[tc_len .. tc_len + rec.text.len], rec.text);
                tc_len += rec.text.len;
            },
            .tool_call_end => tool_ends += 1,
        }
    }
    try std.testing.expectEqualStrings("plan 你好", reasoning_buf[0..r_len]);
    try std.testing.expectEqualStrings("Hi <to!", text_buf[0..t_len]);
    try std.testing.expectEqualStrings("{\"name\":\"ls\"}", tool_buf[0..tc_len]);
    try std.testing.expectEqual(@as(usize, 1), tool_ends);
}

test "sdk.c_api: native BPE tokenizer C-ABI (Qwen & DeepSeek dialects) and CancelToken" {
    // Build minimal byte-level BPE tokenizer JSON in memory (DeepSeek 3-stage split dialect).
    var json_aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer json_aw.deinit();
    const w = &json_aw.writer;
    try w.writeAll("{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (0..256) |b| {
        if (b > 0) try w.writeAll(",");
        var u_buf: [4]u8 = undefined;
        const u_len = try std.unicode.utf8Encode(tokenizer_mod.BYTE_TO_UNICODE[b], &u_buf);
        const s = u_buf[0..u_len];
        try w.writeAll("\"");
        for (s) |ch| {
            if (ch == '"' or ch == '\\') try w.writeByte('\\');
            try w.writeByte(ch);
        }
        try w.print("\":{d}", .{b});
    }
    try w.writeAll(",\"<|im_start|>\":256,\"<|im_end|>\":257},\"merges\":[]},");
    try w.writeAll("\"added_tokens\":[{\"id\":256,\"content\":\"<|im_start|>\",\"special\":true},{\"id\":257,\"content\":\"<|im_end|>\",\"special\":true}],");
    try w.writeAll("\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\p{N}{1,3}\"}}]}}");

    const json = json_aw.written();
    var tok: ?*SdkTokenizer = null;
    try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_tokenizer_from_json(json.ptr, json.len, &tok));
    defer ami_sdk_tokenizer_destroy(tok);

    try std.testing.expectEqual(@as(u32, 258), ami_sdk_tokenizer_vocab_size(tok));
    try std.testing.expectEqual(@backingInt(Dialect.deepseek), ami_sdk_tokenizer_dialect(tok));

    var spec_id: u32 = 0;
    try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_tokenizer_special_id(tok, "<|im_start|>", 12, &spec_id));
    try std.testing.expectEqual(@as(u32, 256), spec_id);

    const sample = "<|im_start|>Hi 1234!<|im_end|>";
    var count: usize = 0;
    try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_count_tokens(tok, sample.ptr, sample.len, 1, &count));
    try std.testing.expectEqual(@as(usize, 1 + 8 + 1), count);

    var ids: [32]u32 = undefined;
    var enc_len: usize = 0;
    try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_tokenize(tok, sample.ptr, sample.len, 1, &ids, ids.len, &enc_len));
    try std.testing.expectEqual(count, enc_len);
    try std.testing.expectEqual(@as(u32, 256), ids[0]);
    try std.testing.expectEqual(@as(u32, 257), ids[enc_len - 1]);

    var dec_buf: [64]u8 = undefined;
    var dec_len: usize = 0;
    try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_detokenize(tok, &ids, enc_len, 1, &dec_buf, dec_buf.len, &dec_len));
    try std.testing.expectEqualStrings("Hi 1234!", dec_buf[0..dec_len]);

    var ctok: ?*SdkCancelToken = null;
    try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_cancel_token_create(&ctok));
    defer ami_sdk_cancel_token_destroy(ctok);
    try std.testing.expectEqual(@as(u32, 0), ami_sdk_cancel_token_is_cancelled(ctok));
    ami_sdk_cancel_token_cancel(ctok);
    try std.testing.expectEqual(@as(u32, 1), ami_sdk_cancel_token_is_cancelled(ctok));
    ami_sdk_cancel_token_reset(ctok);
    try std.testing.expectEqual(@as(u32, 0), ami_sdk_cancel_token_is_cancelled(ctok));
}

test "sdk.c_api: end-to-end engine create, R9 budget guard, views, embed, rerank, vision & zero-leak teardown" {
    const qcfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = .q4,
        .vocab_size = 320,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 256,
        .tie_word_embeddings = false,
    };
    const amf_bytes = try qwen3.buildSyntheticAmfImageWithSpec(std.testing.allocator, qcfg, .{
        .linear_dtype = .q4_affine,
        .embed_dtype = .f16,
        .norm_dtype = .f16,
    }, 77);
    defer std.testing.allocator.free(amf_bytes);

    const cfg_json =
        \\{
        \\  "model_type": "qwen3",
        \\  "architectures": ["Qwen3ForCausalLM"],
        \\  "vocab_size": 320,
        \\  "hidden_size": 128,
        \\  "num_hidden_layers": 2,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 64,
        \\  "intermediate_size": 256,
        \\  "rms_norm_eps": 1e-6,
        \\  "rope_theta": 1000000.0,
        \\  "max_position_embeddings": 2048,
        \\  "tie_word_embeddings": false,
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;

    // Write temporary `.amf` + `config.json` in `TMPDIR` (or `/tmp` fallback) and clean up unconditionally.
    const tmp_root_z = std.c.getenv("TMPDIR") orelse "/tmp";
    const tmp_root = std.mem.span(tmp_root_z);
    const pid: u32 = @bitCast(std.c.getpid());
    var amf_path_buf: [512:0]u8 = undefined;
    var cfg_path_buf: [512:0]u8 = undefined;
    const amf_slice = try std.fmt.bufPrint(&amf_path_buf, "{s}/ami_sdk_test_{d}.amf", .{ std.mem.trimEnd(u8, tmp_root, "/"), pid });
    amf_path_buf[amf_slice.len] = 0;
    const cfg_slice = try std.fmt.bufPrint(&cfg_path_buf, "{s}/ami_sdk_test_{d}_config.json", .{ std.mem.trimEnd(u8, tmp_root, "/"), pid });
    cfg_path_buf[cfg_slice.len] = 0;

    {
        const fd = std.c.open(&amf_path_buf, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        var written: usize = 0;
        while (written < amf_bytes.len) {
            const rc = std.c.write(fd, amf_bytes.ptr + written, amf_bytes.len - written);
            try std.testing.expect(rc > 0);
            written += @intCast(rc);
        }
    }
    defer _ = std.c.unlink(&amf_path_buf);

    {
        const fd = std.c.open(&cfg_path_buf, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        var written: usize = 0;
        while (written < cfg_json.len) {
            const rc = std.c.write(fd, cfg_json.ptr + written, cfg_json.len - written);
            try std.testing.expect(rc > 0);
            written += @intCast(rc);
        }
    }
    defer _ = std.c.unlink(&cfg_path_buf);

    // 1. Verify co-tenant wired_budget_bytes ceiling & R9 strict_live_ram rejection.
    {
        var tight_opts: EngineOptions = undefined;
        ami_sdk_engine_options_init(&tight_opts);
        tight_opts.amf_path = &amf_path_buf;
        tight_opts.config_json_path = &cfg_path_buf;
        tight_opts.max_seq_len = 64;
        tight_opts.wired_budget_bytes = 16 * 1024;
        var bad_eng: ?*SdkEngine = null;
        try std.testing.expectEqual(AMI_SDK_ERR_BUDGET_EXCEEDED, ami_sdk_engine_create(&tight_opts, &bad_eng));
        try std.testing.expect(bad_eng == null);

        ami_sdk_engine_options_init(&tight_opts);
        tight_opts.amf_path = &amf_path_buf;
        tight_opts.config_json_path = &cfg_path_buf;
        tight_opts.max_seq_len = 64;
        tight_opts.strict_live_ram = 1;
        tight_opts.has_live_reclaimable_override = 1;
        tight_opts.live_reclaimable_override_bytes = 64 * 1024;
        tight_opts.live_safety_margin_bytes = 32 * 1024;
        try std.testing.expectEqual(AMI_SDK_ERR_LIVE_RAM_EXCEEDED, ami_sdk_engine_create(&tight_opts, &bad_eng));
        try std.testing.expect(std.mem.span(ami_sdk_last_error()).len > 0);
    }

    // 2. Repeated create/destroy loop + full inference, views, embed, rerank, cancellation, and zero-copy vision.
    var opts: EngineOptions = undefined;
    ami_sdk_engine_options_init(&opts);
    opts.amf_path = &amf_path_buf;
    opts.config_json_path = &cfg_path_buf;
    opts.max_seq_len = 64;
    opts.startup_canary = 0;

    for (0..3) |iter| {
        var eng: ?*SdkEngine = null;
        try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_engine_create(&opts, &eng));
        defer ami_sdk_engine_destroy(eng);

        var stats: EngineStats = .{
            .struct_size = @sizeOf(EngineStats),
            .is_graph_model = 0,
            .vocab_size = 0,
            .hidden_size = 0,
            .num_layers = 0,
            .max_seq_len = 0,
            .seq_len = 0,
            .reserved = 0,
            .total_wired_bytes = 0,
            .sdk_wired_bytes = 0,
            .wired_budget_bytes = 0,
            .live_reclaimable_bytes = 0,
            .live_budget_bytes = 0,
        };
        try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_engine_query_stats(eng, &stats));
        try std.testing.expectEqual(@as(u32, 320), stats.vocab_size);
        try std.testing.expectEqual(@as(u32, 128), stats.hidden_size);
        try std.testing.expect(stats.total_wired_bytes > stats.sdk_wired_bytes);

        if (iter == 0) {
            const prompt = [_]u32{ 5, 11, 23 };
            var gparams: GenParams = undefined;
            ami_sdk_gen_params_init(&gparams);
            gparams.max_tokens = 4;
            gparams.temperature = 0.0;
            gparams.ignore_eos = 1;

            var out_ids: [4]u32 = undefined;
            var gres: GenResult = std.mem.zeroes(GenResult);
            gres.struct_size = @sizeOf(GenResult);

            const CbState = struct {
                calls: u32 = 0,
                fn onToken(ev: *const StreamEvent, ud: ?*anyopaque) callconv(.c) i32 {
                    const st: *@This() = @ptrCast(@alignCast(ud.?));
                    st.calls += 1;
                    _ = ev;
                    return 0;
                }
            };
            var cb_state: CbState = .{};
            try std.testing.expectEqual(
                AMI_SDK_OK,
                ami_sdk_generate_tokens(eng, &prompt, prompt.len, &gparams, CbState.onToken, &cb_state, &out_ids, out_ids.len, &gres),
            );
            try std.testing.expectEqual(@as(u32, 4), gres.generated_tokens);
            try std.testing.expectEqual(@as(u32, 4), cb_state.calls);

            // Zero-copy fp16 & fp32 logits and hidden-state views.
            var v_f16: TensorView = std.mem.zeroes(TensorView);
            var v_f32: TensorView = std.mem.zeroes(TensorView);
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_get_logits_view(eng, @backingInt(DType.f16), &v_f16));
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_get_logits_view(eng, @backingInt(DType.f32), &v_f32));
            try std.testing.expectEqual(@as(usize, 320 * 2), v_f16.byte_len);
            try std.testing.expectEqual(@as(usize, 320 * 4), v_f32.byte_len);
            const lf16: [*]const f16 = @ptrCast(@alignCast(v_f16.data.?));
            const lf32: [*]const f32 = @ptrCast(@alignCast(v_f32.data.?));
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(lf16[0])), lf32[0], 1e-6);

            var h_f32: TensorView = std.mem.zeroes(TensorView);
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_get_hidden_state_view(eng, @backingInt(DType.f32), 1, &h_f32));
            try std.testing.expectEqual(@as(usize, 128 * 4), h_f32.byte_len);

            // Native embed (`last_token` & `mean` + Matryoshka `dimensions = 32` + L2 norm).
            var eparams: EmbedParams = undefined;
            ami_sdk_embed_params_init(&eparams);
            eparams.pooling = @backingInt(PoolingMode.mean);
            eparams.dimensions = 32;
            eparams.normalize = 1;
            var emb: [32]f32 = undefined;
            var emb_dims: usize = 0;
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_embed_tokens(eng, &prompt, prompt.len, &eparams, &emb, emb.len, &emb_dims));
            try std.testing.expectEqual(@as(usize, 32), emb_dims);
            var norm_sq: f32 = 0.0;
            for (emb) |x| norm_sq += x * x;
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm_sq, 1e-4);

            // Native rerank (`prob_yes`).
            var rparams: RerankParams = undefined;
            ami_sdk_rerank_params_init(&rparams);
            rparams.score_mode = @backingInt(RerankMode.prob_yes);
            rparams.positive_token_id = 10;
            rparams.negative_token_id = 11;
            const doc0 = [_]u32{ 12, 13 };
            const doc1 = [_]u32{ 24, 25, 26 };
            const doc_ptrs = [_][*]const u32{ &doc0, &doc1 };
            const doc_lens = [_]usize{ doc0.len, doc1.len };
            var scores: [2]f32 = undefined;
            try std.testing.expectEqual(
                AMI_SDK_OK,
                ami_sdk_rerank_tokens(eng, &prompt, prompt.len, &doc_ptrs, &doc_lens, 2, &rparams, &scores),
            );
            try std.testing.expect(scores[0] >= 0.0 and scores[0] <= 1.0);
            try std.testing.expect(scores[1] >= 0.0 and scores[1] <= 1.0);

            // Cooperative CancelToken stops generation cleanly without signals.
            var ctok: ?*SdkCancelToken = null;
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_cancel_token_create(&ctok));
            defer ami_sdk_cancel_token_destroy(ctok);
            ami_sdk_cancel_token_cancel(ctok);
            gparams.cancel_token = ctok;
            try std.testing.expectEqual(
                AMI_SDK_ERR_CANCELLED,
                ami_sdk_generate_tokens(eng, &prompt, prompt.len, &gparams, null, null, &out_ids, out_ids.len, &gres),
            );

            // Zero-copy IOSurface / CVPixelBufferRef / IOSurfaceID vision ingress (`G14`).
            var vis_src: ?*SdkVisionInput = null;
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_vision_create_iosurface(eng, 1, 32, 32, &vis_src));
            defer ami_sdk_vision_destroy(vis_src);

            var vinfo0: VisionInfo = std.mem.zeroes(VisionInfo);
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_vision_query(vis_src, &vinfo0));
            try std.testing.expectEqual(@as(u32, 1), vinfo0.zero_copy_verified);
            try std.testing.expectEqual(@as(u32, 1), vinfo0.is_page_aligned);

            // Wrap the exact same physical IOSurface by IOSurfaceID and by CVPixelBufferRef (`0` copy).
            var vis_by_id: ?*SdkVisionInput = null;
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_vision_from_iosurface_id(eng, vinfo0.iosurface_id, &vis_by_id));
            defer ami_sdk_vision_destroy(vis_by_id);

            var vis_by_pb: ?*SdkVisionInput = null;
            try std.testing.expectEqual(
                AMI_SDK_OK,
                ami_sdk_vision_from_cvpixelbuffer(eng, ami_sdk_vision_cvpixelbuffer(vis_src), &vis_by_pb),
            );
            defer ami_sdk_vision_destroy(vis_by_pb);

            var vinfo_id: VisionInfo = std.mem.zeroes(VisionInfo);
            try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_vision_query(vis_by_id, &vinfo_id));
            try std.testing.expectEqual(vinfo0.base_address, vinfo_id.base_address);
            try std.testing.expectEqual(@as(u32, 1), vinfo_id.zero_copy_verified);

            var vparams: VisionParams = undefined;
            ami_sdk_vision_params_init(&vparams);
            vparams.patch_size = 16;
            vparams.spatial_merge = 2;
            vparams.t_patch = 1;
            const frames_arr = [_]*const SdkVisionInput{vis_by_id.?};
            var vres: VisionResult = std.mem.zeroes(VisionResult);
            try std.testing.expectEqual(
                AMI_SDK_OK,
                ami_sdk_prefill_vision(eng, &prompt, prompt.len, &frames_arr, 1, &prompt, prompt.len, &vparams, &vres),
            );
            try std.testing.expectEqual(@as(u32, 1), vres.vision_tokens);
            try std.testing.expectEqual(@as(u32, 1), vres.all_frames_zero_copy);
            try std.testing.expectEqual(@as(u32, 1), vres.all_frames_page_aligned);
        }
    }

    // 3. W18 single-file .amf embedded-metadata auto-resolution (NULL config & tokenizer paths, zero sidecars).
    {
        var tok_aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer tok_aw.deinit();
        const tw = &tok_aw.writer;
        try tw.writeAll("{\"model\":{\"type\":\"BPE\",\"vocab\":{");
        for (0..256) |b| {
            if (b > 0) try tw.writeAll(",");
            var u_buf: [4]u8 = undefined;
            const u_len = try std.unicode.utf8Encode(tokenizer_mod.BYTE_TO_UNICODE[b], &u_buf);
            try tw.writeAll("\"");
            for (u_buf[0..u_len]) |ch| {
                if (ch == '"' or ch == '\\') try tw.writeByte('\\');
                try tw.writeByte(ch);
            }
            try tw.print("\":{d}", .{b});
        }
        try tw.writeAll(",\"<|im_start|>\":256,\"<|im_end|>\":257},\"merges\":[]},\"added_tokens\":[{\"id\":256,\"content\":\"<|im_start|>\",\"special\":true},{\"id\":257,\"content\":\"<|im_end|>\",\"special\":true}]}");

        const sf_bytes = try amf_mod.embedMetadataIntoAmfBytes(std.testing.allocator, amf_bytes, &.{
            .{ .name = amf_mod.META_CONFIG_JSON, .payload = cfg_json },
            .{ .name = amf_mod.META_TOKENIZER_JSON, .payload = tok_aw.written() },
        });
        defer std.testing.allocator.free(sf_bytes);

        var sf_path_buf: [512:0]u8 = undefined;
        const sf_slice = try std.fmt.bufPrint(&sf_path_buf, "{s}/ami_sdk_sf_{d}.amf", .{ std.mem.trimEnd(u8, tmp_root, "/"), pid });
        sf_path_buf[sf_slice.len] = 0;
        {
            const fd = std.c.open(&sf_path_buf, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
            try std.testing.expect(fd >= 0);
            defer _ = std.c.close(fd);
            var written: usize = 0;
            while (written < sf_bytes.len) {
                const rc = std.c.write(fd, sf_bytes.ptr + written, sf_bytes.len - written);
                try std.testing.expect(rc > 0);
                written += @intCast(rc);
            }
        }
        defer _ = std.c.unlink(&sf_path_buf);

        var sf_opts: EngineOptions = undefined;
        ami_sdk_engine_options_init(&sf_opts);
        sf_opts.amf_path = &sf_path_buf;
        sf_opts.config_json_path = null;
        sf_opts.tokenizer_json_path = null;
        sf_opts.max_seq_len = 64;
        sf_opts.startup_canary = 0;

        var sf_eng: ?*SdkEngine = null;
        try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_engine_create(&sf_opts, &sf_eng));
        defer ami_sdk_engine_destroy(sf_eng);

        var sf_tok: ?*SdkTokenizer = null;
        try std.testing.expectEqual(AMI_SDK_OK, ami_sdk_tokenizer_from_file(&sf_path_buf, &sf_tok));
        defer ami_sdk_tokenizer_destroy(sf_tok);
        try std.testing.expectEqual(@as(u32, 258), ami_sdk_tokenizer_vocab_size(sf_tok));
    }
}
