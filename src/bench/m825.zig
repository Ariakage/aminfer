//! Milestone M8.25 (`ADR-0026`) `aminfer bench-m825` benchmark, single-frame
//! Inkling 2 parity gate, 16-prompt loop-folded `W8` router-trace capture, and
//! live 24 GB M4 head-to-head Option A (`UD-IQ3_XXS`) vs Option B (`GSQ-RCO v2`)
//! A/B harness.
//!
//! All live paths resolve via `--models-dir` / `AMINFER_MODELS_DIR` through
//! `src/platform/paths.zig` (zero hardcoded volume paths). All measurements
//! enforce `swapouts_delta == 0` and verify `mid_pass_flushes == 0` on
//! `cold == 0` Inkling 2 models.

const std = @import("std");
const page = @import("../mem/page.zig");
const paths = @import("../platform/paths.zig");
const vmstat = @import("../platform/vmstat.zig");
const amf = @import("../format/amf.zig");
const graph = @import("../model/graph.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const chat_template = @import("../model/chat_template.zig");
const engine_mod = @import("../spec/engine.zig");
const m81 = @import("m81.zig");

const Io = std.Io;

pub const OptionSelect = enum {
    both,
    a,
    b,
};

pub const M825Config = struct {
    models_dir: ?[]const u8 = null,
    option: OptionSelect = .both,
    include_synth_gate: bool = true,
    include_real: bool = true,
    include_kl: bool = true,
    include_spec: bool = true,
    kl_steps_per_prompt: u32 = 8,
    ref_cold_slots: u32 = 64,
    /// When set, runs the 16-prompt 48-step loop-folded `W8` router-trace capture
    /// on Option A (`>= 64` tokens/prompt, `T = 0`, `0` mid-pass flushes) and
    /// writes JSONL records directly to this path.
    capture_trace_path: ?[:0]const u8 = null,
    trace_tokens_per_prompt: u32 = 64,
    /// When set, runs the 6-problem Python coding benchmark (`T = 0`, `n = 224`)
    /// during each loaded candidate model pass and appends JSONL completions here.
    coding_out_path: ?[:0]const u8 = null,
    coding_max_tokens: u32 = 256,
    /// Path override for Option A / Option B / Ref / Drafts (relative to models_dir).
    opt_a_rel: []const u8 = "amf/inkling-2-mini-ud-iq3-xxs.amf",
    opt_a_cfg_rel: []const u8 = "amf/inkling-2-mini-ud-iq3-xxs-config.json",
    opt_b_rel: []const u8 = "amf/inkling-2-mini-gsq-rco-v2.amf",
    opt_b_cfg_rel: []const u8 = "amf/inkling-2-mini-gsq-rco-v2-config.json",
    ref_rel: []const u8 = "amf/inkling-2-mini.amf",
    ref_cfg_rel: []const u8 = "Inkling-2-Mini/config.json",
    tok_rel: []const u8 = "Inkling-2-Mini/tokenizer.json",
    dflare_rel: []const u8 = "amf/inkling-2-mini-dflare.amf",
    dflare_cfg_rel: []const u8 = "amf/inkling-2-mini-dflare.config.json",
    dflash_rel: []const u8 = "amf/inkling-2-mini-dflash-q8.amf",
    dflash_cfg_rel: []const u8 = "amf/inkling-2-mini-dflash-q8.config.json",
};

pub const SynthSingleFrameResult = struct {
    ran: bool = false,
    cold0_mid_pass_flushes: u64 = 0,
    cold0_indirect_moe: u64 = 0,
    cold0_gpu_ctrl_calls: u64 = 0,
    sme2_parity_ok: bool = false,
    sme2_ctrl_calls: u64 = 0,
    dflare_gpu_parity_ok: bool = false,
    dflare_mid_pass_flushes: u64 = 0,
    dflare_tau: f64 = 0.0,
    cold_window_parity_ok: bool = false,
    cold_grouped_moe: u64 = 0,
    cold_fetches: u64 = 0,
    trace_mid_pass_flushes: u64 = 0,
    trace_records_valid: u32 = 0,
    trace_folded_ok: bool = false,
    swapouts_delta: u64 = 0,
    passed: bool = false,
};

pub const TraceCaptureResult = struct {
    ran: bool = false,
    prompts: u32 = 0,
    tokens_total: u64 = 0,
    records_written: u64 = 0,
    wired_mib: u64 = 0,
    mid_pass_flushes: u64 = 0,
    tok_per_s: f64 = 0.0,
    swapouts_delta: u64 = 0,
};

pub const DecodeBenchPoint = struct {
    ran: bool = false,
    max_tokens: u32 = 0,
    generated: u32 = 0,
    prefill_ms: f64 = 0.0,
    ms_per_tok: f64 = 0.0,
    tok_per_s: f64 = 0.0,
    effective_gbs: f64 = 0.0,
    mid_pass_flushes: u64 = 0,
    indirect_moe: u64 = 0,
    gpu_ctrl_calls: u64 = 0,
    malloc_blocks_delta: i64 = 0,
    malloc_bytes_delta: i64 = 0,
    zig_steady_allocs: u64 = 0,
    swapouts_delta: u64 = 0,
};

pub const SpecBenchPoint = struct {
    ran: bool = false,
    max_tokens: u32 = 0,
    generated: u32 = 0,
    draft_wired_mib: u64 = 0,
    ms_per_tok: f64 = 0.0,
    tok_per_s: f64 = 0.0,
    mean_accept_len: f64 = 0.0,
    rounds: u64 = 0,
    matched_tokens: u32 = 0,
    greedy_parity_ok: bool = false,
    mid_pass_flushes: u64 = 0,
    malloc_blocks_delta: i64 = 0,
    swapouts_delta: u64 = 0,
};

pub const KlMultiPromptResult = struct {
    ran: bool = false,
    prompts_evaluated: u32 = 0,
    steps_evaluated: u32 = 0,
    mean_kl: f64 = 0.0,
    max_kl: f64 = 0.0,
    top1_matches: u32 = 0,
    top1_pct: f64 = 0.0,
    greedy_matches: u32 = 0,
    greedy_pct: f64 = 0.0,
    max_4gram_repeats: u32 = 0,
    loop_prompts: u32 = 0,
    all_finite: bool = true,
};

pub const OptionEvalResult = struct {
    label: []const u8 = "",
    ran: bool = false,
    status: []const u8 = "skipped",
    disk_bytes: u64 = 0,
    amf_wired_bytes: u64 = 0,
    target_wired_mib: u64 = 0,
    bytes_per_tok: u64 = 0,
    dec64: DecodeBenchPoint = .{},
    dec128: DecodeBenchPoint = .{},
    dflare64: SpecBenchPoint = .{},
    dflare128: SpecBenchPoint = .{},
    dflash64: SpecBenchPoint = .{},
    dflash128: SpecBenchPoint = .{},
    kl: KlMultiPromptResult = .{},
    coding_problems_run: u32 = 0,
    swapouts_delta: u64 = 0,
};

pub const M825Report = struct {
    synth: SynthSingleFrameResult = .{},
    trace: TraceCaptureResult = .{},
    ref_wired_mib: u64 = 0,
    ref_swapouts_delta: u64 = 0,
    option_a: OptionEvalResult = .{ .label = "option_a_ud_iq3_xxs" },
    option_b: OptionEvalResult = .{ .label = "option_b_gsq_rco_v2" },
    total_swapouts_delta: u64 = 0,
    all_passed: bool = false,

    pub fn write(self: *const M825Report, w: *Io.Writer) !void {
        if (self.synth.ran) {
            try w.print(
                \\bench.m825.synth.cold0_mid_pass_flushes    {d}
                \\bench.m825.synth.cold0_indirect_moe        {d}
                \\bench.m825.synth.cold0_gpu_ctrl_calls      {d}
                \\bench.m825.synth.sme2_parity_ok            {s} (sme2_ctrl={d})
                \\bench.m825.synth.dflare_gpu_parity_ok      {s} (mid_flushes={d}, tau={d:.3})
                \\bench.m825.synth.cold_window_parity_ok     {s} (grouped_moe={d}, fetches={d})
                \\bench.m825.synth.trace_folded_ok           {s} (mid_flushes={d}, valid_steps={d})
                \\bench.m825.synth.swapouts_delta            {d}
                \\bench.m825.synth.passed                    {s}
                \\
            , .{
                self.synth.cold0_mid_pass_flushes,
                self.synth.cold0_indirect_moe,
                self.synth.cold0_gpu_ctrl_calls,
                yesNo(self.synth.sme2_parity_ok),
                self.synth.sme2_ctrl_calls,
                yesNo(self.synth.dflare_gpu_parity_ok),
                self.synth.dflare_mid_pass_flushes,
                self.synth.dflare_tau,
                yesNo(self.synth.cold_window_parity_ok),
                self.synth.cold_grouped_moe,
                self.synth.cold_fetches,
                yesNo(self.synth.trace_folded_ok),
                self.synth.trace_mid_pass_flushes,
                self.synth.trace_records_valid,
                self.synth.swapouts_delta,
                yesNo(self.synth.passed),
            });
        }
        if (self.trace.ran) {
            try w.print(
                \\bench.m825.trace.prompts                   {d}
                \\bench.m825.trace.tokens_total              {d}
                \\bench.m825.trace.records_written           {d}
                \\bench.m825.trace.wired_mib                 {d}
                \\bench.m825.trace.mid_pass_flushes          {d}
                \\bench.m825.trace.tok_per_s                 {d:.2}
                \\bench.m825.trace.swapouts_delta            {d}
                \\
            , .{
                self.trace.prompts,
                self.trace.tokens_total,
                self.trace.records_written,
                self.trace.wired_mib,
                self.trace.mid_pass_flushes,
                self.trace.tok_per_s,
                self.trace.swapouts_delta,
            });
        }
        if (self.ref_wired_mib > 0) {
            try w.print(
                \\bench.m825.ref_nvfp4.wired_mib             {d}
                \\bench.m825.ref_nvfp4.swapouts_delta        {d}
                \\
            , .{ self.ref_wired_mib, self.ref_swapouts_delta });
        }
        try writeOptionReport(w, "opt_a", &self.option_a);
        try writeOptionReport(w, "opt_b", &self.option_b);
        try w.print(
            \\bench.m825.total_swapouts_delta            {d}
            \\bench.m825.all_passed                      {s}
            \\
        , .{ self.total_swapouts_delta, yesNo(self.all_passed) });
    }
};

fn yesNo(b: bool) []const u8 {
    return if (b) "yes" else "no";
}

fn writeOptionReport(w: *Io.Writer, tag: []const u8, opt: *const OptionEvalResult) !void {
    try w.print(
        \\bench.m825.{s}.status                     {s} ({s})
        \\
    , .{ tag, opt.status, opt.label });
    if (!opt.ran) return;
    try w.print(
        \\bench.m825.{s}.disk_bytes                 {d}
        \\bench.m825.{s}.amf_wired_mib              {d:.2}
        \\bench.m825.{s}.target_wired_mib           {d}
        \\bench.m825.{s}.bytes_per_tok              {d}
        \\bench.m825.{s}.swapouts_delta             {d}
        \\
    , .{
        tag, opt.disk_bytes,
        tag, @as(f64, @floatFromInt(opt.amf_wired_bytes)) / (1024.0 * 1024.0),
        tag, opt.target_wired_mib,
        tag, opt.bytes_per_tok,
        tag, opt.swapouts_delta,
    });
    if (opt.dec64.ran) {
        try w.print(
            \\bench.m825.{s}.decode_n64                 {d:.2} tok/s ({d:.2} ms/tok, {d:.2} GB/s, mid_flushes={d}, indirect_moe={d}, gpu_ctrl={d}, malloc_blk={d}, zig_allocs={d}, swap={d})
            \\
        , .{
            tag,
            opt.dec64.tok_per_s,
            opt.dec64.ms_per_tok,
            opt.dec64.effective_gbs,
            opt.dec64.mid_pass_flushes,
            opt.dec64.indirect_moe,
            opt.dec64.gpu_ctrl_calls,
            opt.dec64.malloc_blocks_delta,
            opt.dec64.zig_steady_allocs,
            opt.dec64.swapouts_delta,
        });
    }
    if (opt.dec128.ran) {
        try w.print(
            \\bench.m825.{s}.decode_n128                {d:.2} tok/s ({d:.2} ms/tok, {d:.2} GB/s, mid_flushes={d}, indirect_moe={d}, gpu_ctrl={d}, malloc_blk={d}, zig_allocs={d}, swap={d})
            \\
        , .{
            tag,
            opt.dec128.tok_per_s,
            opt.dec128.ms_per_tok,
            opt.dec128.effective_gbs,
            opt.dec128.mid_pass_flushes,
            opt.dec128.indirect_moe,
            opt.dec128.gpu_ctrl_calls,
            opt.dec128.malloc_blocks_delta,
            opt.dec128.zig_steady_allocs,
            opt.dec128.swapouts_delta,
        });
    }
    if (opt.dflare64.ran) {
        try w.print(
            \\bench.m825.{s}.dflare3l_n64               {d:.2} tok/s ({d:.2} ms/tok, tau={d:.3}, rounds={d}, parity={s} {d}/{d}, draft_mib={d}, mid_flushes={d}, swap={d})
            \\
        , .{
            tag,
            opt.dflare64.tok_per_s,
            opt.dflare64.ms_per_tok,
            opt.dflare64.mean_accept_len,
            opt.dflare64.rounds,
            yesNo(opt.dflare64.greedy_parity_ok),
            opt.dflare64.matched_tokens,
            opt.dflare64.generated,
            opt.dflare64.draft_wired_mib,
            opt.dflare64.mid_pass_flushes,
            opt.dflare64.swapouts_delta,
        });
    }
    if (opt.dflare128.ran) {
        try w.print(
            \\bench.m825.{s}.dflare3l_n128              {d:.2} tok/s ({d:.2} ms/tok, tau={d:.3}, rounds={d}, parity={s} {d}/{d}, draft_mib={d}, mid_flushes={d}, swap={d})
            \\
        , .{
            tag,
            opt.dflare128.tok_per_s,
            opt.dflare128.ms_per_tok,
            opt.dflare128.mean_accept_len,
            opt.dflare128.rounds,
            yesNo(opt.dflare128.greedy_parity_ok),
            opt.dflare128.matched_tokens,
            opt.dflare128.generated,
            opt.dflare128.draft_wired_mib,
            opt.dflare128.mid_pass_flushes,
            opt.dflare128.swapouts_delta,
        });
    }
    if (opt.dflash64.ran) {
        try w.print(
            \\bench.m825.{s}.dflash_q8_n64              {d:.2} tok/s ({d:.2} ms/tok, tau={d:.3}, rounds={d}, parity={s} {d}/{d}, draft_mib={d}, mid_flushes={d}, swap={d})
            \\
        , .{
            tag,
            opt.dflash64.tok_per_s,
            opt.dflash64.ms_per_tok,
            opt.dflash64.mean_accept_len,
            opt.dflash64.rounds,
            yesNo(opt.dflash64.greedy_parity_ok),
            opt.dflash64.matched_tokens,
            opt.dflash64.generated,
            opt.dflash64.draft_wired_mib,
            opt.dflash64.mid_pass_flushes,
            opt.dflash64.swapouts_delta,
        });
    }
    if (opt.dflash128.ran) {
        try w.print(
            \\bench.m825.{s}.dflash_q8_n128             {d:.2} tok/s ({d:.2} ms/tok, tau={d:.3}, rounds={d}, parity={s} {d}/{d}, draft_mib={d}, mid_flushes={d}, swap={d})
            \\
        , .{
            tag,
            opt.dflash128.tok_per_s,
            opt.dflash128.ms_per_tok,
            opt.dflash128.mean_accept_len,
            opt.dflash128.rounds,
            yesNo(opt.dflash128.greedy_parity_ok),
            opt.dflash128.matched_tokens,
            opt.dflash128.generated,
            opt.dflash128.draft_wired_mib,
            opt.dflash128.mid_pass_flushes,
            opt.dflash128.swapouts_delta,
        });
    }
    if (opt.kl.ran) {
        try w.print(
            \\bench.m825.{s}.kl_vs_nvfp4                prompts={d} steps={d} mean_kl={d:.6} max_kl={d:.6} top1={d:.2}% ({d}/{d}) greedy={d:.2}% ({d}/{d}) max_4gram={d} loop_prompts={d}/{d} finite={s}
            \\
        , .{
            tag,
            opt.kl.prompts_evaluated,
            opt.kl.steps_evaluated,
            opt.kl.mean_kl,
            opt.kl.max_kl,
            opt.kl.top1_pct,
            opt.kl.top1_matches,
            opt.kl.steps_evaluated,
            opt.kl.greedy_pct,
            opt.kl.greedy_matches,
            opt.kl.steps_evaluated,
            opt.kl.max_4gram_repeats,
            opt.kl.loop_prompts,
            opt.kl.prompts_evaluated,
            yesNo(opt.kl.all_finite),
        });
    }
    if (opt.coding_problems_run > 0) {
        try w.print(
            \\bench.m825.{s}.coding_problems_dumped     {d}
            \\
        , .{ tag, opt.coding_problems_run });
    }
}

/// 16 diverse calibration prompts (Python, Rust, C++, TypeScript, Zig, SQL, math,
/// systems architecture, algorithms, networking, security) for 48-step `router_trace`
/// capture (`>= 64` tokens each, `T = 0`).
pub const TRACE_CALIBRATION_PROMPTS = [_][]const u8{
    "Write a Python function `quicksort(arr)` that sorts a list in place using three-way partitioning:\n\ndef quicksort(arr: list[int]) -> list[int]:\n",
    "Implement a lock-free single-producer single-consumer ring buffer in Rust using `AtomicUsize`:\n\npub struct SpscRing<T, const N: usize> {\n",
    "Write a modern C++20 LRU cache class with O(1) `get` and `put` using `std::list` and `std::unordered_map`:\n\ntemplate <typename K, typename V>\nclass LruCache {\n",
    "Implement a strongly typed TypeScript async task scheduler with concurrency limit and exponential backoff:\n\nexport class TaskQueue {\n",
    "Write a Zig 0.17 function that parses an IPv4 CIDR block into a 32-bit network address and mask without heap allocation:\n\npub fn parseCidr(text: []const u8) !Cidr {\n",
    "Write a PostgreSQL query using recursive CTEs and window functions to compute rolling 7-day active user retention by signup cohort:\n\nWITH RECURSIVE ",
    "Prove that the eigenvalues of a real symmetric matrix are real and eigenvectors corresponding to distinct eigenvalues are orthogonal.\n\nProof:\n",
    "Compute the number of positive integers less than 1000 that are divisible by 3 or 5 but not 15, showing each inclusion-exclusion step:\n",
    "Explain how Apple Silicon unified memory, 16 KiB pages, wired memory limits, and Metal 4 command allocators interact during LLM decode:\n",
    "Describe the Raft consensus protocol's leader election, log replication, safety property, and membership change edge cases:\n",
    "Explain Static Single Assignment (SSA) form, dominance frontiers, phi-node placement, and global value numbering in optimizing compilers:\n",
    "Implement Dijkstra's shortest-path algorithm with a binary min-heap in Python for weighted directed graphs:\n\ndef dijkstra(n: int, edges: list[tuple[int, int, int]], src: int) -> list[int]:\n",
    "Write a finite-state JSON string unescaper supporting UTF-16 surrogate pairs and escape sequences in C:\n\nint json_unescape(const char *src, size_t len, char *dst, size_t cap) {\n",
    "Design a non-blocking kqueue HTTP/1.1 Server-Sent Events (SSE) server handling partial writes, backpressure, and graceful client disconnects:\n",
    "Explain the security properties of AES-256-GCM, why nonce reuse breaks GHASH authentication, and how HKDF derives per-session keys:\n",
    "Derive the closed-form solution for the linear recurrence a_n = 5 a_{n-1} - 6 a_{n-2} with a_0 = 2, a_1 = 5 using characteristic polynomials:\n",
};

/// 8 diverse evaluation prompts for KL divergence, top-1 agreement, and 4-gram
/// repetition loop evaluation against native `NVFP4` `inkling-2-mini.amf`.
pub const KL_EVAL_PROMPTS = [_][]const u8{
    "Compute 17 * 23 + 19 * 31 step by step:\n",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n",
    "In Rust, explain the difference between `Send`, `Sync`, `Arc<Mutex<T>>`, and `RwLock<T>`:\n",
    "In C++, explain move semantics, value categories (lvalue, xvalue, prvalue), and copy elision:\n",
    "SELECT customer_id, SUM(amount) AS total_spend FROM orders WHERE created_at >= '2026-01-01' GROUP BY ",
    "Explain how virtual memory translation lookaside buffers (TLBs), page tables, and huge pages reduce memory latency:\n",
    "Prove by induction that the sum of the first n odd positive integers equals n^2.\n\nProof:\n",
    "Explain how Mixture-of-Experts (MoE) top-k routing balances compute cost, memory bandwidth, and load balancing:\n",
};

pub const CodingProblem = struct {
    id: []const u8,
    prompt: []const u8,
};

/// 6 self-contained executable Python coding problems (`T = 0`, rendered via Inkling 2 ChatTemplate
/// with `reasoning_effort = .none` and tested via `python3` assertions).
pub const CODING_PROBLEMS = [_]CodingProblem{
    .{
        .id = "is_prime",
        .prompt = "Write a complete, working Python function `def is_prime(n: int) -> bool:` that returns True if integer n is prime and False otherwise. Output only the Python code in a ```python block, no explanation.",
    },
    .{
        .id = "merge_intervals",
        .prompt = "Write a complete, working Python function `def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:` that merges all overlapping [start, end] intervals and returns a sorted list of merged intervals. Output only the Python code in a ```python block, no explanation.",
    },
    .{
        .id = "rle_encode",
        .prompt = "Write a complete, working Python function `def rle_encode(s: str) -> str:` that performs run-length encoding on string s, e.g. 'aaabbc' -> 'a3b2c1' and '' -> ''. Output only the Python code in a ```python block, no explanation.",
    },
    .{
        .id = "longest_unique_substr",
        .prompt = "Write a complete, working Python function `def longest_unique_substr(s: str) -> int:` that returns the length of the longest substring of s without repeating characters. Output only the Python code in a ```python block, no explanation.",
    },
    .{
        .id = "eval_rpn",
        .prompt = "Write a complete, working Python function `def eval_rpn(tokens: list[str]) -> int:` that evaluates Reverse Polish Notation tokens (integers and '+', '-', '*', '/', with '/' truncating toward zero) and returns an int. Output only the Python code in a ```python block, no explanation.",
    },
    .{
        .id = "topological_sort",
        .prompt = "Write a complete, working Python function `def topological_sort(n: int, edges: list[tuple[int, int]]) -> list[int]:` that takes n nodes (0..n-1) and directed edges (u, v) and returns a valid topological ordering list of length n, or [] if a cycle exists. Output only the Python code in a ```python block, no explanation.",
    },
};

const BENCH_SPEED_PROMPT: []const u8 =
    "Write a Python function `is_prime(n)` that returns True if n is prime, along with unit tests and time complexity analysis:\n\ndef is_prime(n: int) -> bool:\n";

fn fileExistsPosix(path_z: [:0]const u8) bool {
    const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return false;
    _ = std.c.close(fd);
    return true;
}

fn readFileAllocPosix(allocator: std.mem.Allocator, path_z: [:0]const u8, max_bytes: usize) ![]u8 {
    const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    var buf: [16384]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        const nu: usize = @intCast(n);
        if (out.items.len + nu > max_bytes) return error.FileTooBig;
        try out.appendSlice(allocator, buf[0..nu]);
    }
    return out.toOwnedSlice(allocator);
}

fn writeAllFd(fd: std.c.fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const rc = std.c.write(fd, data.ptr + off, data.len - off);
        if (rc <= 0) return error.WriteFailed;
        off += @intCast(rc);
    }
}

fn writeJsonStringFd(fd: std.c.fd_t, text: []const u8) !void {
    try writeAllFd(fd, "\"");
    for (text) |c| {
        switch (c) {
            '"' => try writeAllFd(fd, "\\\""),
            '\\' => try writeAllFd(fd, "\\\\"),
            '\n' => try writeAllFd(fd, "\\n"),
            '\r' => try writeAllFd(fd, "\\r"),
            '\t' => try writeAllFd(fd, "\\t"),
            else => {
                if (c < 0x20) {
                    var esc: [6]u8 = undefined;
                    const s = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch unreachable;
                    try writeAllFd(fd, s);
                } else {
                    const one = [_]u8{c};
                    try writeAllFd(fd, &one);
                }
            },
        }
    }
    try writeAllFd(fd, "\"");
}

fn computeAmfFileAndWiredBytes(allocator: std.mem.Allocator, path_z: [:0]const u8) !struct { disk: u64, wired: u64 } {
    var af = try amf.AmfFile.open(allocator, path_z);
    defer af.close(allocator);
    var wired: u64 = 0;
    for (af.parsed.entries) |*e| {
        if (amf.isMetaEntry(e)) continue;
        const tk = e.tensorKind() catch continue;
        if (tk == .cold_expert) continue;
        wired += e.blob_bytes;
    }
    return .{ .disk = af.parsed.file_size, .wired = wired };
}

fn runAminferToolsSynth(allocator: std.mem.Allocator, io: Io, tool_args: []const []const u8) !void {
    var argv_buf: [32][]const u8 = undefined;
    var n: usize = 0;
    if (fileExistsPosix("tools/.venv/bin/aminfer-tools")) {
        argv_buf[n] = "tools/.venv/bin/aminfer-tools";
        n += 1;
    } else {
        argv_buf[n] = "/opt/homebrew/bin/uv";
        n += 1;
        argv_buf[n] = "run";
        n += 1;
        argv_buf[n] = "--directory";
        n += 1;
        argv_buf[n] = "tools";
        n += 1;
        argv_buf[n] = "aminfer-tools";
        n += 1;
    }
    if (n + tool_args.len > argv_buf.len) return error.TooManyArgs;
    for (tool_args) |a| {
        argv_buf[n] = a;
        n += 1;
    }
    const res = try std.process.run(allocator, io, .{ .argv = argv_buf[0..n] });
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) return error.SynthToolFailed,
        else => return error.SynthToolFailed,
    }
}

/// Synthetic single-frame `M8.25` parity gate over `tiny-inkling2-mini` (`--sme2-pack controllers`):
/// verifies zero mid-pass flushes (`cold=0`), `--sme2-controllers` bit-exact parity,
/// GPU `3L DFlare` zero-flush bit-exact parity, `cold=4` multi-window grouped MoE parity,
/// and `W8` zero-flush loop-folded router trace capture.
pub fn runSynthSingleFrameGate(allocator: std.mem.Allocator, n_tokens: u32) !SynthSingleFrameResult {
    var out: SynthSingleFrameResult = .{ .ran = true };
    const vm0 = try vmstat.Snapshot.capture();

    var threaded_io: std.Io.Threaded = .init(allocator, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_dir = try paths.tmpPath(&root_buf, "aminfer-bench-m825-synth");
    const s0_dir = try std.fmt.allocPrint(allocator, "{s}/s0", .{root_dir});
    defer allocator.free(s0_dir);
    const s4_dir = try std.fmt.allocPrint(allocator, "{s}/s4", .{root_dir});
    defer allocator.free(s4_dir);

    {
        const mk = try std.process.run(allocator, io, .{ .argv = &.{ "/bin/mkdir", "-p", s0_dir, s4_dir } });
        allocator.free(mk.stdout);
        allocator.free(mk.stderr);
    }
    defer {
        if (std.process.run(allocator, io, .{ .argv = &.{ "/bin/rm", "-rf", root_dir } })) |rm| {
            allocator.free(rm.stdout);
            allocator.free(rm.stderr);
        } else |err| {
            std.log.warn("bench-m825 scratch cleanup failed: {}", .{err});
        }
    }

    try runAminferToolsSynth(allocator, io, &.{
        "synth-inkling2",
        "--preset",
        "tiny-inkling2-mini",
        "--dflare-layers",
        "3",
        "--sme2-pack",
        "controllers",
        "--cold-experts-per-layer",
        "0",
        "--output-dir",
        s0_dir,
    });
    try runAminferToolsSynth(allocator, io, &.{
        "synth-inkling2",
        "--preset",
        "tiny-inkling2-mini",
        "--dflare-layers",
        "3",
        "--sme2-pack",
        "controllers",
        "--cold-experts-per-layer",
        "4",
        "--output-dir",
        s4_dir,
    });

    const s0_amf = try std.fmt.allocPrintSentinel(allocator, "{s}/model.amf", .{s0_dir}, 0);
    defer allocator.free(s0_amf);
    const s0_cfg_p = try std.fmt.allocPrintSentinel(allocator, "{s}/config.json", .{s0_dir}, 0);
    defer allocator.free(s0_cfg_p);
    const s0_dflare_amf = try std.fmt.allocPrintSentinel(allocator, "{s}/dflare.amf", .{s0_dir}, 0);
    defer allocator.free(s0_dflare_amf);
    const s0_dflare_cfg = try std.fmt.allocPrintSentinel(allocator, "{s}/dflare_config.json", .{s0_dir}, 0);
    defer allocator.free(s0_dflare_cfg);
    const s4_amf = try std.fmt.allocPrintSentinel(allocator, "{s}/model.amf", .{s4_dir}, 0);
    defer allocator.free(s4_amf);

    const cfg_bytes = try readFileAllocPosix(allocator, s0_cfg_p, 4 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    var gcfg = try graph.GraphConfig.fromJson(allocator, cfg_bytes);
    defer gcfg.deinit(allocator);

    const prompt = [_]u32{ 1, 7, 13, 21 };
    const n_gen: usize = @max(@as(usize, 4), n_tokens);
    const tok_gpu = try allocator.alloc(u32, n_gen);
    defer allocator.free(tok_gpu);
    const tok_cmp = try allocator.alloc(u32, n_gen);
    defer allocator.free(tok_cmp);

    // 1. Single-frame GPU (`cold=0`, `sme2_panel64` auto-deinterleaved) + GPU `DFlare` + zero-flush `W8` router trace.
    {
        var model = try graph.GraphModel.loadFromFile(allocator, s0_amf, gcfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 16,
            .enable_feature_taps = true,
            .enable_router_trace = true,
            .startup_canary = false,
        });
        defer model.deinit(allocator);

        model.resetKvCache();
        model.resetM82Stats();
        var cur = try model.prefillChunk(&prompt, 0, true);
        tok_gpu[0] = cur;
        var valid_steps: u32 = 0;
        var folded_ok = true;
        const eff_steps = model.m82EffectiveSteps();
        for (1..n_gen) |step| {
            cur = try model.stepDecode(cur, @intCast(prompt.len + step - 1));
            tok_gpu[step] = cur;
            if (step == 1) {
                for (0..eff_steps) |si| {
                    if (model.lastStepRouterTopKForStep(@intCast(si))) |tk| {
                        if (tk.len == gcfg.num_experts_per_tok) {
                            valid_steps += 1;
                        } else {
                            folded_ok = false;
                        }
                        const pl = model.m82PhysicalLayerForStep(@intCast(si));
                        if (pl >= gcfg.num_hidden_layers) folded_ok = false;
                    } else {
                        folded_ok = false;
                    }
                }
            }
        }
        if (model.m82Stats()) |ms| {
            out.cold0_mid_pass_flushes = ms.mid_pass_flushes;
            out.cold0_indirect_moe = ms.inkling2_indirect_moe_dispatches;
            out.cold0_gpu_ctrl_calls = ms.inkling2_gpu_ctrl_calls;
            out.trace_mid_pass_flushes = ms.mid_pass_flushes;
        }
        out.trace_records_valid = valid_steps;
        out.trace_folded_ok = folded_ok and (valid_steps == eff_steps) and (eff_steps > gcfg.num_hidden_layers);

        // GPU `3L DFlare` on the same zero-flush model.
        const dcfg_bytes = try readFileAllocPosix(allocator, s0_dflare_cfg, 4 * 1024 * 1024);
        defer allocator.free(dcfg_bytes);
        var loaded = try engine_mod.LoadedDFlareProposer.loadFromFile(allocator, &model, s0_dflare_amf, dcfg_bytes, null, null);
        defer loaded.deinit(allocator);
        var se = try engine_mod.SpecEngine.initWithDFlare(allocator, &model, loaded.proposer(), .{
            .mode = .dflare,
            .max_tree_nodes = 16,
            .temperature = 0.0,
            .enable_pruning = false,
        });
        defer se.deinit(allocator);

        model.resetM82Stats();
        tok_cmp[0] = try se.prefill(&prompt, null);
        var produced: usize = 1;
        while (produced < n_gen) {
            const rs = try se.stepRound(null, tok_cmp[produced..n_gen]);
            if (rs.emitted_count == 0) break;
            produced += rs.emitted_count;
        }
        out.dflare_gpu_parity_ok = (produced == n_gen) and std.mem.eql(u32, tok_gpu[0..n_gen], tok_cmp[0..n_gen]);
        out.dflare_tau = se.stats.meanAcceptanceLength();
        if (model.m82Stats()) |ms| {
            out.dflare_mid_pass_flushes = ms.mid_pass_flushes;
        }
    }

    // 2. Opt-in `--sme2-controllers` CPU SME2 parity check.
    {
        var model = try graph.GraphModel.loadFromFile(allocator, s0_amf, gcfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 1,
            .use_sme2_controllers = true,
            .startup_canary = false,
        });
        defer model.deinit(allocator);

        model.resetKvCache();
        model.resetM82Stats();
        var cur = try model.prefillChunk(&prompt, 0, true);
        tok_cmp[0] = cur;
        for (1..n_gen) |step| {
            cur = try model.stepDecode(cur, @intCast(prompt.len + step - 1));
            tok_cmp[step] = cur;
        }
        out.sme2_parity_ok = std.mem.eql(u32, tok_gpu[0..n_gen], tok_cmp[0..n_gen]);
        if (model.m82Stats()) |ms| {
            out.sme2_ctrl_calls = ms.inkling2_sme2_ctrl_calls;
        }
    }

    // 3. `cold=4` with constrained `cold_slot_count = 2` (forces multi-window async prefetch + grouped MoE!).
    {
        var model = try graph.GraphModel.loadFromFile(allocator, s4_amf, gcfg, .{
            .max_seq_len = 128,
            .max_tree_nodes = 1,
            .cold_slot_count = 2,
            .startup_canary = false,
        });
        defer model.deinit(allocator);

        model.resetKvCache();
        model.resetM82Stats();
        var cur = try model.prefillChunk(&prompt, 0, true);
        tok_cmp[0] = cur;
        for (1..n_gen) |step| {
            cur = try model.stepDecode(cur, @intCast(prompt.len + step - 1));
            tok_cmp[step] = cur;
        }
        out.cold_window_parity_ok = std.mem.eql(u32, tok_gpu[0..n_gen], tok_cmp[0..n_gen]);
        if (model.m82Stats()) |ms| {
            out.cold_grouped_moe = ms.inkling2_grouped_moe_dispatches;
            out.cold_fetches = ms.cold_expert_fetches;
        }
    }

    const vm1 = try vmstat.Snapshot.capture();
    out.swapouts_delta = vmstat.Snapshot.swapoutDelta(vm0, vm1);
    out.passed = (out.swapouts_delta == 0) and
        (out.cold0_mid_pass_flushes == 0) and
        (out.cold0_indirect_moe > 0) and
        (out.cold0_gpu_ctrl_calls > 0) and
        out.sme2_parity_ok and
        (out.sme2_ctrl_calls > 0) and
        out.dflare_gpu_parity_ok and
        (out.dflare_mid_pass_flushes == 0) and
        out.cold_window_parity_ok and
        (out.cold_grouped_moe > 0) and
        (out.cold_fetches > 0) and
        out.trace_folded_ok and
        (out.trace_mid_pass_flushes == 0);
    return out;
}

/// Captures loop-folded 48-step `router_trace` JSONL across all 16 diverse prompts
/// (`>= 64` tokens each, `T = 0`) using ONE model load of Option A (`0` mid-pass flushes).
fn runTraceCapture(allocator: std.mem.Allocator, cfg: M825Config, out_path: [:0]const u8) !TraceCaptureResult {
    var res: TraceCaptureResult = .{ .ran = true };
    var amf_buf: [std.fs.max_path_bytes]u8 = undefined;
    var cfg_buf: [std.fs.max_path_bytes]u8 = undefined;
    var tok_buf: [std.fs.max_path_bytes]u8 = undefined;
    const amf_p = (try paths.resolveModelFileZ(cfg.models_dir, &amf_buf, cfg.opt_a_rel)) orelse return error.MissingModelsDir;
    const cfg_p = (try paths.resolveModelFileZ(cfg.models_dir, &cfg_buf, cfg.opt_a_cfg_rel)) orelse return error.MissingModelsDir;
    const tok_p = (try paths.resolveModelFileZ(cfg.models_dir, &tok_buf, cfg.tok_rel)) orelse return error.MissingModelsDir;

    const cfg_bytes = try readFileAllocPosix(allocator, cfg_p, 8 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    const tok_bytes = try readFileAllocPosix(allocator, tok_p, 64 * 1024 * 1024);
    defer allocator.free(tok_bytes);

    var gcfg = try graph.GraphConfig.fromJson(allocator, cfg_bytes);
    defer gcfg.deinit(allocator);
    var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
    defer tok.deinit(allocator);

    const n_gen: u32 = @max(@as(u32, 64), cfg.trace_tokens_per_prompt);
    var model = try graph.GraphModel.loadFromFile(allocator, amf_p, gcfg, .{
        .max_seq_len = 512,
        .max_tree_nodes = 1,
        .enable_router_trace = true,
        .startup_canary = false,
    });
    defer model.deinit(allocator);
    res.wired_mib = model.totalWiredBytes() >> 20;

    const fd = std.c.open(out_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.FileOpenFailed;
    defer _ = std.c.close(fd);

    const vm0 = try vmstat.Snapshot.capture();
    model.resetM82Stats();
    const t0 = try vmstat.nowNs();
    const eff_steps = model.m82EffectiveSteps();
    var line_buf: [512]u8 = undefined;

    for (TRACE_CALIBRATION_PROMPTS) |prompt_text| {
        const prompt_ids = try tok.encodeAlloc(allocator, prompt_text);
        defer allocator.free(prompt_ids);
        if (prompt_ids.len == 0) continue;

        model.resetKvCache();
        var pos: u32 = 0;
        var next_tok: u32 = prompt_ids[0];
        for (prompt_ids) |ptid| {
            next_tok = try model.stepDecode(ptid, pos);
            pos += 1;
            res.tokens_total += 1;
            res.records_written += try emitRouterTraceStepFd(fd, &model, eff_steps, &line_buf);
        }
        var gi: u32 = 1;
        while (gi < n_gen) : (gi += 1) {
            next_tok = try model.stepDecode(next_tok, pos);
            pos += 1;
            res.tokens_total += 1;
            res.records_written += try emitRouterTraceStepFd(fd, &model, eff_steps, &line_buf);
        }
        res.prompts += 1;
    }

    const t1 = try vmstat.nowNs();
    const vm1 = try vmstat.Snapshot.capture();
    res.swapouts_delta = vmstat.Snapshot.swapoutDelta(vm0, vm1);
    if (model.m82Stats()) |ms| {
        res.mid_pass_flushes = ms.mid_pass_flushes;
    }
    const sec = @as(f64, @floatFromInt(t1 -| t0)) / 1e9;
    res.tok_per_s = if (sec > 0.0) @as(f64, @floatFromInt(res.tokens_total)) / sec else 0.0;
    return res;
}

fn emitRouterTraceStepFd(fd: std.c.fd_t, model: *const graph.GraphModel, eff_steps: u32, line_buf: *[512]u8) !u64 {
    var count: u64 = 0;
    for (0..eff_steps) |si| {
        const topk = model.lastStepRouterTopKForStep(@intCast(si)) orelse continue;
        if (topk.len == 0) continue;
        const phys_l = model.m82PhysicalLayerForStep(@intCast(si));
        var len: usize = (std.fmt.bufPrint(line_buf, "{{\"layer\":{d},\"experts\":[", .{phys_l}) catch return error.BufferTooSmall).len;
        for (topk, 0..) |eid, k| {
            if (k > 0) {
                line_buf[len] = ',';
                len += 1;
            }
            len += (std.fmt.bufPrint(line_buf[len..], "{d}", .{eid}) catch return error.BufferTooSmall).len;
        }
        len += (std.fmt.bufPrint(line_buf[len..], "]}}\n", .{}) catch return error.BufferTooSmall).len;
        try writeAllFd(fd, line_buf[0..len]);
        count += 1;
    }
    return count;
}

const RefLogitsBundle = struct {
    vocab: usize,
    steps_per_prompt: u32,
    prompt_ids: [KL_EVAL_PROMPTS.len][]u32,
    logits: []f32,
    greedy: []u32,

    fn deinit(self: *RefLogitsBundle, allocator: std.mem.Allocator) void {
        for (self.prompt_ids) |p| {
            if (p.len > 0) allocator.free(p);
        }
        if (self.logits.len > 0) allocator.free(self.logits);
        if (self.greedy.len > 0) allocator.free(self.greedy);
    }
};

fn captureRefNvfp4Logits(
    allocator: std.mem.Allocator,
    ref_amf: [:0]const u8,
    gcfg: graph.GraphConfig,
    tok: *const tokenizer_mod.Tokenizer,
    steps_per_prompt: u32,
    cold_slots: u32,
    out_wired_mib: *u64,
    out_swap_delta: *u64,
) !RefLogitsBundle {
    const eff_vocab: usize = if (gcfg.m82_cfg.unpadded_vocab_size > 0 and gcfg.m82_cfg.unpadded_vocab_size < gcfg.vocab_size)
        gcfg.m82_cfg.unpadded_vocab_size
    else
        gcfg.vocab_size;
    var bundle: RefLogitsBundle = .{
        .vocab = eff_vocab,
        .steps_per_prompt = steps_per_prompt,
        .prompt_ids = @splat(&.{}),
        .logits = &.{},
        .greedy = &.{},
    };
    errdefer bundle.deinit(allocator);

    for (KL_EVAL_PROMPTS, 0..) |pt, pi| {
        bundle.prompt_ids[pi] = try tok.encodeAlloc(allocator, pt);
    }
    const total_steps: usize = KL_EVAL_PROMPTS.len * @as(usize, steps_per_prompt);
    bundle.logits = try allocator.alloc(f32, total_steps * eff_vocab);
    bundle.greedy = try allocator.alloc(u32, total_steps);

    const vm0 = try vmstat.Snapshot.capture();
    var ref_model = try graph.GraphModel.loadFromFile(allocator, ref_amf, gcfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .cold_slot_count = cold_slots,
        .startup_canary = false,
    });
    defer ref_model.deinit(allocator);
    out_wired_mib.* = ref_model.totalWiredBytes() >> 20;

    var off: usize = 0;
    for (bundle.prompt_ids) |pslice| {
        ref_model.resetKvCache();
        var pos: u32 = @intCast(pslice.len);
        var last_tok: u32 = try prefillPromptChunked(&ref_model, pslice);
        for (0..steps_per_prompt) |s| {
            const lg = ref_model.readLastLogitsF16();
            for (0..eff_vocab) |v| {
                bundle.logits[(off + s) * eff_vocab + v] = @floatCast(lg[v]);
            }
            bundle.greedy[off + s] = last_tok;
            if (s + 1 < steps_per_prompt) {
                last_tok = try ref_model.stepDecode(last_tok, pos);
                pos += 1;
            }
        }
        off += steps_per_prompt;
    }

    const vm1 = try vmstat.Snapshot.capture();
    out_swap_delta.* = vmstat.Snapshot.swapoutDelta(vm0, vm1);
    return bundle;
}

fn evalCandidateOption(
    allocator: std.mem.Allocator,
    cfg: M825Config,
    label: []const u8,
    short_tag: []const u8,
    amf_rel: []const u8,
    cfg_rel: []const u8,
    tok: *const tokenizer_mod.Tokenizer,
    ref_opt: ?*const RefLogitsBundle,
    coding_fd: ?std.c.fd_t,
) !OptionEvalResult {
    var out: OptionEvalResult = .{ .label = label, .status = "skipped_missing" };

    var amf_buf: [std.fs.max_path_bytes]u8 = undefined;
    var cfg_buf: [std.fs.max_path_bytes]u8 = undefined;
    const amf_p = (try paths.resolveModelFileZ(cfg.models_dir, &amf_buf, amf_rel)) orelse return out;
    const cfg_p = (try paths.resolveModelFileZ(cfg.models_dir, &cfg_buf, cfg_rel)) orelse return out;
    if (!fileExistsPosix(amf_p) or !fileExistsPosix(cfg_p)) return out;

    const sizes = try computeAmfFileAndWiredBytes(allocator, amf_p);
    out.disk_bytes = sizes.disk;
    out.amf_wired_bytes = sizes.wired;

    const cfg_bytes = try readFileAllocPosix(allocator, cfg_p, 8 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    var gcfg = try graph.GraphConfig.fromJson(allocator, cfg_bytes);
    defer gcfg.deinit(allocator);

    const vm_all0 = try vmstat.Snapshot.capture();
    var model = try graph.GraphModel.loadFromFile(allocator, amf_p, gcfg, .{
        .max_seq_len = 512,
        .max_tree_nodes = if (cfg.include_spec) 8 else 1,
        .enable_feature_taps = cfg.include_spec,
        .startup_canary = false,
    });
    defer model.deinit(allocator);

    out.ran = true;
    out.status = "ok";
    out.target_wired_mib = model.totalWiredBytes() >> 20;
    out.bytes_per_tok = model.bytesPerDecodeToken(64);

    // 1. `M = 1` decode at `n = 64` and `n = 128` (`T = 0`, `--ignore-eos`)
    const speed_prompt_ids = try tok.encodeAlloc(allocator, BENCH_SPEED_PROMPT);
    defer allocator.free(speed_prompt_ids);
    var dec_tokens_128: [128]u32 = undefined;

    out.dec64 = try runDecodePoint(&model, speed_prompt_ids, 64, dec_tokens_128[0..64]);
    out.dec128 = try runDecodePoint(&model, speed_prompt_ids, 128, dec_tokens_128[0..128]);

    // 2. 8-prompt KL / Top-1 / 4-gram vs native `NVFP4` reference (if captured)
    if (ref_opt) |ref| {
        out.kl = try evalKlAgainstRef(allocator, &model, ref);
    }

    // 3. Optional 6-problem Python coding rollouts (`T = 0`, Inkling 2 ChatTemplate `reasoning_effort = .none`)
    if (coding_fd) |cfd| {
        const ink_tpl = chat_template.ChatTemplate.preset(.inkling2);
        var tpl_buf: [4096]u8 = undefined;
        for (CODING_PROBLEMS) |prob| {
            var fw = chat_template.FixedWriter.init(&tpl_buf);
            var renderer = chat_template.PromptRenderer.init(
                &ink_tpl,
                &fw,
                .{
                    .reasoning_effort = .none,
                    .add_generation_prompt = true,
                },
                false,
                0,
            );
            try renderer.begin();
            const parts = [_]chat_template.ContentPart{.{ .text = prob.prompt }};
            try renderer.pushMessage(.{
                .role = .user,
                .parts = &parts,
            }, 0);
            try renderer.finish();
            // Prefill Inkling 2's direct assistant text header so effort=none skips `<|content_thinking|>`.
            try fw.writeAll("<|message_model|><|content_text|>");
            const pids = try tok.encodeAlloc(allocator, fw.written());
            defer allocator.free(pids);
            const gen = try allocator.alloc(u32, cfg.coding_max_tokens);
            defer allocator.free(gen);

            model.resetKvCache();
            const tc0 = try vmstat.nowNs();
            var cur = try prefillPromptChunked(&model, pids);
            var gcount: usize = 0;
            if (cfg.coding_max_tokens > 0) {
                gen[0] = cur;
                gcount = 1;
                var pos: u32 = @intCast(pids.len);
                while (gcount < cfg.coding_max_tokens) {
                    if (cur == gcfg.eos_token_id) break;
                    cur = try model.stepDecode(cur, pos);
                    pos += 1;
                    if (cur == gcfg.eos_token_id) break;
                    gen[gcount] = cur;
                    gcount += 1;
                }
            }
            const tc1 = try vmstat.nowNs();
            const text = try tok.decodeAlloc(allocator, gen[0..gcount]);
            defer allocator.free(text);
            const sec = @as(f64, @floatFromInt(tc1 -| tc0)) / 1e9;
            const tps = if (sec > 0.0) @as(f64, @floatFromInt(gcount)) / sec else 0.0;

            try writeAllFd(cfd, "{\"option\":");
            try writeJsonStringFd(cfd, short_tag);
            try writeAllFd(cfd, ",\"id\":");
            try writeJsonStringFd(cfd, prob.id);
            try writeAllFd(cfd, ",\"prompt\":");
            try writeJsonStringFd(cfd, prob.prompt);
            try writeAllFd(cfd, ",\"completion\":");
            try writeJsonStringFd(cfd, text);
            var tail_buf: [96]u8 = undefined;
            const tail = std.fmt.bufPrint(&tail_buf, ",\"tokens\":{d},\"tok_s\":{d:.2}}}\n", .{ gcount, tps }) catch unreachable;
            try writeAllFd(cfd, tail);
            out.coding_problems_run += 1;
        }
    }

    // 4. GPU `3L DFlare` & GPU `DFlash-Q8` at `n = 64` and `n = 128` (`100%` greedy parity vs `dec_tokens_128`),
    //    loaded sequentially so the two draft models NEVER overlap in wired RAM!
    if (cfg.include_spec) {
        var df_buf: [std.fs.max_path_bytes]u8 = undefined;
        var dfc_buf: [std.fs.max_path_bytes]u8 = undefined;
        if (try paths.resolveModelFileZ(cfg.models_dir, &df_buf, cfg.dflare_rel)) |df_p| {
            if (try paths.resolveModelFileZ(cfg.models_dir, &dfc_buf, cfg.dflare_cfg_rel)) |dfc_p| {
                if (fileExistsPosix(df_p) and fileExistsPosix(dfc_p)) {
                    const dbytes = try readFileAllocPosix(allocator, dfc_p, 4 * 1024 * 1024);
                    defer allocator.free(dbytes);
                    const dsizes = try computeAmfFileAndWiredBytes(allocator, df_p);
                    var loaded = try engine_mod.LoadedDFlareProposer.loadFromFile(allocator, &model, df_p, dbytes, null, null);
                    defer loaded.deinit(allocator);
                    const dmib = dsizes.wired >> 20;
                    out.dflare64 = try runSpecPoint(allocator, &model, loaded.proposer(), speed_prompt_ids, dec_tokens_128[0..64], dmib);
                    out.dflare128 = try runSpecPoint(allocator, &model, loaded.proposer(), speed_prompt_ids, dec_tokens_128[0..128], dmib);
                }
            }
        }

        if (try paths.resolveModelFileZ(cfg.models_dir, &df_buf, cfg.dflash_rel)) |df_p| {
            if (try paths.resolveModelFileZ(cfg.models_dir, &dfc_buf, cfg.dflash_cfg_rel)) |dfc_p| {
                if (fileExistsPosix(df_p) and fileExistsPosix(dfc_p)) {
                    const dbytes = try readFileAllocPosix(allocator, dfc_p, 4 * 1024 * 1024);
                    defer allocator.free(dbytes);
                    const dsizes = try computeAmfFileAndWiredBytes(allocator, df_p);
                    var loaded = try engine_mod.LoadedDFlareProposer.loadFromFile(allocator, &model, df_p, dbytes, null, null);
                    defer loaded.deinit(allocator);
                    const dmib = dsizes.wired >> 20;
                    out.dflash64 = try runSpecPoint(allocator, &model, loaded.proposer(), speed_prompt_ids, dec_tokens_128[0..64], dmib);
                    out.dflash128 = try runSpecPoint(allocator, &model, loaded.proposer(), speed_prompt_ids, dec_tokens_128[0..128], dmib);
                }
            }
        }
    }

    const vm_all1 = try vmstat.Snapshot.capture();
    out.swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_all0, vm_all1);
    return out;
}

fn prefillPromptChunked(model: *graph.GraphModel, prompt_ids: []const u32) !u32 {
    const cap: usize = @max(@as(usize, 1), @as(usize, model.prefillChunkCap()));
    var pos: usize = 0;
    var last_tok: u32 = 0;
    while (pos < prompt_ids.len) {
        const end = @min(prompt_ids.len, pos + cap);
        const is_last = (end == prompt_ids.len);
        last_tok = try model.prefillChunk(prompt_ids[pos..end], @intCast(pos), is_last);
        pos = end;
    }
    return last_tok;
}

fn runDecodePoint(
    model: *graph.GraphModel,
    prompt_ids: []const u32,
    n_tokens: u32,
    out_tokens: []u32,
) !DecodeBenchPoint {
    var pt: DecodeBenchPoint = .{ .ran = true, .max_tokens = n_tokens };
    const vm0 = try vmstat.Snapshot.capture();
    model.resetKvCache();
    model.resetM82Stats();

    const tp0 = try vmstat.nowNs();
    var next_tok = try prefillPromptChunked(model, prompt_ids);
    const tp1 = try vmstat.nowNs();
    pt.prefill_ms = @as(f64, @floatFromInt(tp1 -| tp0)) / 1e6;

    out_tokens[0] = next_tok;
    var gen: u32 = 1;
    var pos: u32 = @intCast(prompt_ids.len);

    const heap0 = try vmstat.Snapshot.capture();
    const td0 = try vmstat.nowNs();
    while (gen < n_tokens) : (gen += 1) {
        next_tok = try model.stepDecode(next_tok, pos);
        out_tokens[gen] = next_tok;
        pos += 1;
    }
    const td1 = try vmstat.nowNs();
    const heap1 = try vmstat.Snapshot.capture();
    const vm1 = try vmstat.Snapshot.capture();

    pt.generated = gen;
    const steps: u32 = if (gen > 1) gen - 1 else 1;
    pt.ms_per_tok = (@as(f64, @floatFromInt(td1 -| td0)) / @as(f64, @floatFromInt(steps))) / 1e6;
    pt.tok_per_s = if (pt.ms_per_tok > 0.0) 1000.0 / pt.ms_per_tok else 0.0;
    const bpt = model.bytesPerDecodeToken(pos);
    pt.effective_gbs = if (pt.ms_per_tok > 0.0) @as(f64, @floatFromInt(bpt)) / (pt.ms_per_tok * 1e6) else 0.0;
    const hd = vmstat.heapDelta(heap0, heap1);
    pt.malloc_blocks_delta = hd.blocks_in_use;
    pt.malloc_bytes_delta = hd.size_in_use;
    pt.zig_steady_allocs = 0; // `stepDecode` takes no allocator argument (`0` Zig heap allocs by construction).
    pt.swapouts_delta = vmstat.Snapshot.swapoutDelta(vm0, vm1);
    if (model.m82Stats()) |ms| {
        pt.mid_pass_flushes = ms.mid_pass_flushes;
        pt.indirect_moe = ms.inkling2_indirect_moe_dispatches;
        pt.gpu_ctrl_calls = ms.inkling2_gpu_ctrl_calls;
    }
    return pt;
}

fn runSpecPoint(
    allocator: std.mem.Allocator,
    model: *graph.GraphModel,
    proposer: anytype,
    prompt_ids: []const u32,
    ref_tokens: []const u32,
    draft_wired_mib: u64,
) !SpecBenchPoint {
    var pt: SpecBenchPoint = .{
        .ran = true,
        .max_tokens = @intCast(ref_tokens.len),
        .draft_wired_mib = draft_wired_mib,
    };
    const vm0 = try vmstat.Snapshot.capture();
    var se = try engine_mod.SpecEngine.initWithDFlare(allocator, model, proposer, .{
        .mode = .dflare,
        .max_tree_nodes = 8,
        .temperature = 0.0,
        .enable_pruning = false,
    });
    defer se.deinit(allocator);

    const spec_toks = try allocator.alloc(u32, ref_tokens.len);
    defer allocator.free(spec_toks);

    model.resetM82Stats();
    spec_toks[0] = try se.prefill(prompt_ids, null);
    var produced: usize = 1;

    const heap0 = try vmstat.Snapshot.capture();
    const t0 = try vmstat.nowNs();
    while (produced < ref_tokens.len) {
        const rs = try se.stepRound(null, spec_toks[produced..ref_tokens.len]);
        if (rs.emitted_count == 0) break;
        produced += rs.emitted_count;
    }
    const t1 = try vmstat.nowNs();
    const heap1 = try vmstat.Snapshot.capture();
    const vm1 = try vmstat.Snapshot.capture();

    pt.generated = @intCast(produced);
    const dec_cnt: usize = if (produced > 1) produced - 1 else 1;
    pt.ms_per_tok = (@as(f64, @floatFromInt(t1 -| t0)) / @as(f64, @floatFromInt(dec_cnt))) / 1e6;
    pt.tok_per_s = if (pt.ms_per_tok > 0.0) 1000.0 / pt.ms_per_tok else 0.0;
    pt.mean_accept_len = se.stats.meanAcceptanceLength();
    pt.rounds = se.stats.rounds;
    var matched: u32 = 0;
    for (0..produced) |i| {
        if (spec_toks[i] == ref_tokens[i]) matched += 1;
    }
    pt.matched_tokens = matched;
    pt.greedy_parity_ok = (produced == ref_tokens.len) and (matched == ref_tokens.len);
    pt.malloc_blocks_delta = vmstat.heapDelta(heap0, heap1).blocks_in_use;
    pt.swapouts_delta = vmstat.Snapshot.swapoutDelta(vm0, vm1);
    if (model.m82Stats()) |ms| {
        pt.mid_pass_flushes = ms.mid_pass_flushes;
    }
    return pt;
}

fn evalKlAgainstRef(
    allocator: std.mem.Allocator,
    model: *graph.GraphModel,
    ref: *const RefLogitsBundle,
) !KlMultiPromptResult {
    var out: KlMultiPromptResult = .{
        .ran = true,
        .prompts_evaluated = @intCast(ref.prompt_ids.len),
        .steps_evaluated = @intCast(ref.prompt_ids.len * @as(usize, ref.steps_per_prompt)),
    };
    const cand_tmp = try allocator.alloc(f32, ref.vocab);
    defer allocator.free(cand_tmp);
    const cand_greedy = try allocator.alloc(u32, ref.steps_per_prompt);
    defer allocator.free(cand_greedy);

    var sum_kl: f64 = 0.0;
    var max_kl: f64 = 0.0;
    var off: usize = 0;

    for (ref.prompt_ids) |pslice| {
        // Teacher-forced KL + top-1 agreement
        model.resetKvCache();
        var pos: u32 = @intCast(pslice.len);
        _ = try prefillPromptChunked(model, pslice);
        for (0..ref.steps_per_prompt) |s| {
            const lg = model.readLastLogitsF16();
            for (0..ref.vocab) |v| cand_tmp[v] = @floatCast(lg[v]);
            const m = m81.computeLogitsKlAndTop1F32(ref.logits[(off + s) * ref.vocab ..][0..ref.vocab], cand_tmp);
            if (!m.all_finite) out.all_finite = false;
            sum_kl += m.kl_divergence;
            if (m.kl_divergence > max_kl) max_kl = m.kl_divergence;
            if (m.top1_match) out.top1_matches += 1;
            if (s + 1 < ref.steps_per_prompt) {
                _ = try model.stepDecode(ref.greedy[off + s], pos);
                pos += 1;
            }
        }

        // Free-running greedy rollout parity + 4-gram loop check
        model.resetKvCache();
        pos = @intCast(pslice.len);
        var ftok: u32 = try prefillPromptChunked(model, pslice);
        for (0..ref.steps_per_prompt) |s| {
            cand_greedy[s] = ftok;
            if (ftok == ref.greedy[off + s]) out.greedy_matches += 1;
            if (s + 1 < ref.steps_per_prompt) {
                ftok = try model.stepDecode(ftok, pos);
                pos += 1;
            }
        }
        const rep4 = m81.computeMaxNgramRepeatCount(cand_greedy, 4);
        if (rep4 > out.max_4gram_repeats) out.max_4gram_repeats = rep4;
        if (m81.hasDegenerateRepetitionLoop(ref.greedy[off .. off + ref.steps_per_prompt], cand_greedy)) {
            out.loop_prompts += 1;
        }
        off += ref.steps_per_prompt;
    }

    const total_f: f64 = @floatFromInt(@max(out.steps_evaluated, 1));
    out.mean_kl = sum_kl / total_f;
    out.max_kl = max_kl;
    out.top1_pct = 100.0 * @as(f64, @floatFromInt(out.top1_matches)) / total_f;
    out.greedy_pct = 100.0 * @as(f64, @floatFromInt(out.greedy_matches)) / total_f;
    return out;
}

pub fn runM825WithConfig(allocator: std.mem.Allocator, cfg: M825Config) !M825Report {
    var report: M825Report = .{};
    const vm0 = try vmstat.Snapshot.capture();

    if (cfg.capture_trace_path) |trace_p| {
        report.trace = try runTraceCapture(allocator, cfg, trace_p);
        const vm1 = try vmstat.Snapshot.capture();
        report.total_swapouts_delta = vmstat.Snapshot.swapoutDelta(vm0, vm1);
        report.all_passed = (report.trace.swapouts_delta == 0) and
            (report.trace.mid_pass_flushes == 0) and
            (report.trace.prompts == TRACE_CALIBRATION_PROMPTS.len) and
            (report.trace.records_written > 0);
        return report;
    }

    if (cfg.include_synth_gate) {
        report.synth = try runSynthSingleFrameGate(allocator, 8);
    }

    if (cfg.include_real and paths.resolveModelsDir(cfg.models_dir) != null) {
        var tok_buf: [std.fs.max_path_bytes]u8 = undefined;
        const tok_p = (try paths.resolveModelFileZ(cfg.models_dir, &tok_buf, cfg.tok_rel)) orelse return error.MissingModelsDir;
        if (fileExistsPosix(tok_p)) {
            const tok_bytes = try readFileAllocPosix(allocator, tok_p, 64 * 1024 * 1024);
            defer allocator.free(tok_bytes);
            var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, tok_bytes);
            defer tok.deinit(allocator);

            // Capture native `NVFP4` reference logits ONCE (`cold=120/128`, `0` swap), then deinit before Option A/B!
            var ref_bundle_opt: ?RefLogitsBundle = null;
            defer if (ref_bundle_opt) |*rb| rb.deinit(allocator);
            if (cfg.include_kl) {
                var ref_buf: [std.fs.max_path_bytes]u8 = undefined;
                var rcfg_buf: [std.fs.max_path_bytes]u8 = undefined;
                if (try paths.resolveModelFileZ(cfg.models_dir, &ref_buf, cfg.ref_rel)) |ref_p| {
                    if (try paths.resolveModelFileZ(cfg.models_dir, &rcfg_buf, cfg.ref_cfg_rel)) |rcfg_p| {
                        if (fileExistsPosix(ref_p) and fileExistsPosix(rcfg_p)) {
                            const rcfg_bytes = try readFileAllocPosix(allocator, rcfg_p, 8 * 1024 * 1024);
                            defer allocator.free(rcfg_bytes);
                            var rgcfg = try graph.GraphConfig.fromJson(allocator, rcfg_bytes);
                            defer rgcfg.deinit(allocator);
                            ref_bundle_opt = try captureRefNvfp4Logits(
                                allocator,
                                ref_p,
                                rgcfg,
                                &tok,
                                cfg.kl_steps_per_prompt,
                                cfg.ref_cold_slots,
                                &report.ref_wired_mib,
                                &report.ref_swapouts_delta,
                            );
                        }
                    }
                }
            }

            var coding_fd: ?std.c.fd_t = null;
            if (cfg.coding_out_path) |cp| {
                const fd = std.c.open(cp.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
                if (fd < 0) return error.FileOpenFailed;
                coding_fd = fd;
            }
            defer if (coding_fd) |fd| {
                _ = std.c.close(fd);
            };

            const ref_ptr: ?*const RefLogitsBundle = if (ref_bundle_opt) |*rb| rb else null;
            if (cfg.option == .both or cfg.option == .a) {
                report.option_a = try evalCandidateOption(
                    allocator,
                    cfg,
                    "option_a_ud_iq3_xxs",
                    "A",
                    cfg.opt_a_rel,
                    cfg.opt_a_cfg_rel,
                    &tok,
                    ref_ptr,
                    coding_fd,
                );
            }
            if (cfg.option == .both or cfg.option == .b) {
                report.option_b = try evalCandidateOption(
                    allocator,
                    cfg,
                    "option_b_gsq_rco_v2",
                    "B",
                    cfg.opt_b_rel,
                    cfg.opt_b_cfg_rel,
                    &tok,
                    ref_ptr,
                    coding_fd,
                );
            }
        }
    }

    const vm1 = try vmstat.Snapshot.capture();
    report.total_swapouts_delta = vmstat.Snapshot.swapoutDelta(vm0, vm1);

    const synth_ok = (!cfg.include_synth_gate) or report.synth.passed;
    const opt_a_ok = (!report.option_a.ran) or optionPassed(&report.option_a);
    const opt_b_ok = (!report.option_b.ran) or optionPassed(&report.option_b);
    report.all_passed = (report.total_swapouts_delta == 0) and synth_ok and opt_a_ok and opt_b_ok;
    return report;
}

fn optionPassed(opt: *const OptionEvalResult) bool {
    if (opt.swapouts_delta != 0) return false;
    if (opt.dec64.ran and (opt.dec64.mid_pass_flushes != 0 or opt.dec64.swapouts_delta != 0)) return false;
    if (opt.dec128.ran and (opt.dec128.mid_pass_flushes != 0 or opt.dec128.swapouts_delta != 0)) return false;
    if (opt.dflare64.ran and (!opt.dflare64.greedy_parity_ok or opt.dflare64.mid_pass_flushes != 0)) return false;
    if (opt.dflare128.ran and (!opt.dflare128.greedy_parity_ok or opt.dflare128.mid_pass_flushes != 0)) return false;
    if (opt.dflash64.ran and (!opt.dflash64.greedy_parity_ok or opt.dflash64.mid_pass_flushes != 0)) return false;
    if (opt.dflash128.ran and (!opt.dflash128.greedy_parity_ok or opt.dflash128.mid_pass_flushes != 0)) return false;
    if (opt.kl.ran and !opt.kl.all_finite) return false;
    return true;
}

test "m825: single-frame Inkling 2 cold=0, SME2 parity, GPU DFlare, cold>0 windowing, and W8 router trace pass" {
    const gpa = std.testing.allocator;
    const report = try runM825WithConfig(gpa, .{
        .include_synth_gate = true,
        .include_real = false,
    });
    try std.testing.expect(report.all_passed);
    try std.testing.expect(report.synth.passed);
    try std.testing.expectEqual(@as(u64, 0), report.synth.cold0_mid_pass_flushes);
    try std.testing.expect(report.synth.cold0_indirect_moe > 0);
    try std.testing.expect(report.synth.cold0_gpu_ctrl_calls > 0);
    try std.testing.expect(report.synth.sme2_parity_ok);
    try std.testing.expect(report.synth.dflare_gpu_parity_ok);
    try std.testing.expectEqual(@as(u64, 0), report.synth.dflare_mid_pass_flushes);
    try std.testing.expect(report.synth.cold_window_parity_ok);
    try std.testing.expect(report.synth.cold_grouped_moe > 0);
    try std.testing.expect(report.synth.cold_fetches > 0);
    try std.testing.expect(report.synth.trace_folded_ok);
    try std.testing.expectEqual(@as(u64, 0), report.synth.trace_mid_pass_flushes);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    try std.testing.expect(std.mem.indexOf(u8, aw.written(), "bench.m825.synth.passed") != null);
}
