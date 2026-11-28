//! Deterministic parser fuzz & property regression tests (`T1`, 2026-11-28 review).
//!
//! Exercises every untrusted ingress parser across 1,024+ deterministic pseudo-random
//! mutations (`std.Random.Xoshiro256`) in < 200 ms without panics, OOB reads, or leaks:
//!   - HTTP header parser & chunked transfer decoder (`http_sse`)
//!   - JSON reader & request parsers (`/v1/chat/completions`, `/v1/completions`,
//!     `/v1/messages`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`) (`http_sse`)
//!   - `.amf` binary header & directory parser (`amf.Parsed.parseBytes`)
//!   - `.ang` N-gram NVMe header & slot continuation validator (`ngram_nvme`)
//!   - HuggingFace BPE `tokenizer.json` & ChatTemplate JSON/scanner (`tokenizer`, `chat_template`)

const std = @import("std");
const amf = @import("format/amf.zig");
const page = @import("mem/page.zig");
const chat_template = @import("model/chat_template.zig");
const tokenizer = @import("model/tokenizer.zig");
const http_sse = @import("server/http_sse.zig");
const ngram_nvme = @import("spec/ngram_nvme.zig");

const FUZZ_ITERS: usize = 1024;

fn mutateBytes(rng: std.Random, src: []const u8, out: []u8) []u8 {
    if (out.len == 0) return out[0..0];
    const max_len = @min(out.len, @max(@as(usize, 64), src.len * 2));
    var len = if (src.len > 0 and rng.uintLessThan(u8, 4) != 0) blk: {
        const copy_len = @min(src.len, out.len);
        @memcpy(out[0..copy_len], src[0..copy_len]);
        break :blk copy_len;
    } else blk: {
        const rand_len = rng.uintLessThan(usize, max_len + 1);
        rng.bytes(out[0..rand_len]);
        break :blk rand_len;
    };

    const ops = 1 + rng.uintLessThan(usize, 6);
    for (0..ops) |_| {
        switch (rng.uintLessThan(u8, 5)) {
            0 => {
                if (len > 0) {
                    out[rng.uintLessThan(usize, len)] ^= rng.int(u8);
                }
            },
            1 => {
                if (len > 0) {
                    const idx = rng.uintLessThan(usize, len);
                    const special = [_]u8{ 0, '\r', '\n', '"', '\\', '{', '}', '[', ']', ':', ',', 0xFF, 0x7F };
                    out[idx] = special[rng.uintLessThan(usize, special.len)];
                }
            },
            2 => {
                if (len > 1) {
                    len = rng.uintLessThan(usize, len);
                }
            },
            3 => {
                if (len < out.len) {
                    const add_len = @min(out.len - len, 1 + rng.uintLessThan(usize, 24));
                    rng.bytes(out[len .. len + add_len]);
                    len += add_len;
                }
            },
            else => {
                if (len > 4) {
                    const a = rng.uintLessThan(usize, len);
                    const b = rng.uintLessThan(usize, len);
                    const tmp = out[a];
                    out[a] = out[b];
                    out[b] = tmp;
                }
            },
        }
    }
    return out[0..len];
}

test "T1 parser fuzz: HTTP headers and chunked transfer decoder never panic" {
    var prng = std.Random.DefaultPrng.init(0x2026_11_28_0001);
    const rng = prng.random();
    const seeds = [_][]const u8{
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nContent-Length: 12\r\n\r\n",
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\nAuthorization: Bearer secret\r\n\r\n",
        "GET /health HTTP/1.0\r\nConnection: close\r\n\r\n",
        "OPTIONS /v1/models HTTP/1.1\r\nOrigin: https://app.local\r\nAccess-Control-Request-Method: POST\r\n\r\n",
    };
    const chunk_seeds = [_][]const u8{
        "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n",
        "4;ext=1\r\ntest\r\n0\r\nX-Trailer: 1\r\n\r\n",
        "ffffffffffffffff\r\nabc",
        "0\r\n\r\n",
    };

    var buf: [512]u8 = undefined;
    var header_errors: usize = 0;
    var chunk_errors: usize = 0;
    for (0..FUZZ_ITERS) |iter| {
        const h_seed = seeds[iter % seeds.len];
        const h_sample = mutateBytes(rng, h_seed, &buf);
        if (http_sse.fuzzParseHttpHeaders(h_sample)) |_| {} else |_| {
            header_errors += 1;
        }

        const c_seed = chunk_seeds[iter % chunk_seeds.len];
        const c_sample = mutateBytes(rng, c_seed, &buf);
        if (http_sse.fuzzDecodeChunkedInPlace(c_sample, 256)) |_| {} else |_| {
            chunk_errors += 1;
        }
    }
    try std.testing.expect(header_errors > 0);
    try std.testing.expect(chunk_errors > 0);
}

test "T1 parser fuzz: HTTP JSON reader and endpoint request parsers never panic" {
    var prng = std.Random.DefaultPrng.init(0x2026_11_28_0002);
    const rng = prng.random();
    const json_seeds = [_][]const u8{
        \\{"model":"qwen3","messages":[{"role":"user","content":"hi\\u0041\n"}],"max_tokens":16,"stream":true}
        ,
        \\{"model":"m","prompt":[1,2,3],"suffix":"end","temperature":0.5,"seed":42,"stop":["\n"]}
        ,
        \\{"model":"claude","system":"sys","messages":[{"role":"user","content":[{"type":"text","text":"a"}]}],"max_tokens":8}
        ,
        \\{"model":"gpt","input":[{"role":"user","content":[{"type":"input_text","text":"ok"}]}],"reasoning":{"effort":"high"}}
        ,
        \\{"model":"emb","input":["alpha","beta"],"dimensions":64}
        ,
        \\{"model":"rerank","query":"q","documents":["d1","d2"],"top_n":1}
        ,
    };

    var json_buf: [512]u8 = undefined;
    var prompt_buf: [1024]u8 = undefined;
    var model_buf: [128]u8 = undefined;
    var text_scratch: [1024]u8 = undefined;

    var reader_errs: usize = 0;
    var req_errs: usize = 0;
    const endpoints = [_]http_sse.Endpoint{ .chat_completions, .completions, .messages, .responses };

    for (0..FUZZ_ITERS) |iter| {
        const seed = json_seeds[iter % json_seeds.len];
        const sample = mutateBytes(rng, seed, &json_buf);

        if (http_sse.fuzzJsonReader(sample)) |_| {} else |_| {
            reader_errs += 1;
        }
        const ep = endpoints[iter % endpoints.len];
        if (http_sse.parseInferenceRequest(ep, sample, "default", &prompt_buf, &model_buf)) |_| {} else |_| {
            req_errs += 1;
        }
        if (http_sse.parseEmbeddingsRequest(sample, "default", &model_buf, &text_scratch)) |_| {} else |_| {
            req_errs += 1;
        }
        if (http_sse.parseRerankRequest(sample, "default", &model_buf, &text_scratch)) |_| {} else |_| {
            req_errs += 1;
        }
    }
    try std.testing.expect(reader_errs > 0);
    try std.testing.expect(req_errs > 0);
}

test "T1 parser fuzz: AMF header/directory and ANG header/slot validators never panic or leak" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2026_11_28_0003);
    const rng = prng.random();

    const aligned_buf = try gpa.alignedAlloc(u8, .@"16", page.APPLE_PAGE_SIZE);
    defer gpa.free(aligned_buf);

    var amf_errs: usize = 0;
    var ang_errs: usize = 0;
    var short_buf: [512]u8 = undefined;

    for (0..FUZZ_ITERS) |iter| {
        if (iter % 4 == 0) {
            const s_len = rng.uintLessThan(usize, short_buf.len + 1);
            rng.bytes(short_buf[0..s_len]);
            if (amf.Parsed.parseBytes(gpa, short_buf[0..s_len])) |p_val| {
                var p = p_val;
                p.deinit(gpa);
            } else |_| {
                amf_errs += 1;
            }
        } else {
            @memset(aligned_buf, 0);
            var hdr = amf.Header{
                .magic = amf.MAGIC,
                .version = amf.FORMAT_VERSION,
                .flags = rng.int(u32),
                .page_size = page.APPLE_PAGE_SIZE,
                .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
                .tensor_count = rng.uintLessThan(u32, 32),
                .reserved0 = 0,
                .directory_offset = @sizeOf(amf.Header),
                .directory_bytes = rng.uintLessThan(u64, page.APPLE_PAGE_SIZE),
                .directory_crc32 = rng.int(u32),
                .header_crc32 = 0,
                .model_id = @splat('m'),
                .reserved1 = 0,
            };
            if (iter % 8 == 1) rng.bytes(&hdr.magic);
            @memcpy(aligned_buf[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
            const dir_mut_len = @min(@as(usize, 512), aligned_buf.len - @sizeOf(amf.Header));
            rng.bytes(aligned_buf[@sizeOf(amf.Header) .. @sizeOf(amf.Header) + dir_mut_len]);

            if (amf.Parsed.parseBytes(gpa, aligned_buf)) |p_val| {
                var p = p_val;
                p.deinit(gpa);
            } else |_| {
                amf_errs += 1;
            }
        }

        // Exercise `.ang` header and slot continuation validators with mutated structs.
        var ang_hdr: ngram_nvme.Header = undefined;
        rng.bytes(std.mem.asBytes(&ang_hdr));
        if (iter % 2 == 0) {
            ang_hdr.magic = ngram_nvme.MAGIC;
            ang_hdr.version = ngram_nvme.FORMAT_VERSION;
            ang_hdr.page_size = page.APPLE_PAGE_SIZE;
            ang_hdr.checksum_algo = ngram_nvme.CHECKSUM_CRC32_IEEE;
        }
        const fake_sz = (@as(u64, 1) + rng.uintLessThan(u64, 8)) * page.APPLE_PAGE_SIZE;
        if (ngram_nvme.validateHeader(&ang_hdr, fake_sz)) |_| {} else |_| {
            ang_errs += 1;
        }

        var slot: ngram_nvme.NgramSlot = undefined;
        rng.bytes(std.mem.asBytes(&slot));
        _ = ngram_nvme.validateSlotContinuation(&slot, 1 + rng.uintLessThan(u32, 65536));
    }

    try std.testing.expect(amf_errs > 0);
    try std.testing.expect(ang_errs > 0);
}

test "T1 parser fuzz: Tokenizer JSON and ChatTemplate parser/scanner never panic or leak" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2026_11_28_0004);
    const rng = prng.random();

    const tok_seeds = [_][]const u8{
        \\{"model":{"type":"BPE","vocab":{"<|endoftext|>":0,"a":1,"b":2,"ab":3},"merges":["a b"]},"added_tokens":[{"id":0,"content":"<|endoftext|>","special":true}]}
        ,
        \\{"model":{"vocab":{"x":0,"y":1},"merges":[["x","y"]]},"added_tokens":[{"id":2,"content":"<|im_start|>","special":true}]}
        ,
    };
    const tpl_seeds = [_][]const u8{
        \\{"dialect":"chatml","bos_token":"<s>","eos_token":"<|im_end|>","thinking":{"supported":true}}
        ,
        \\{"dialect":"inkling2","eos_token":"<|content_model_end_sampling|>","thinking":{"supported":true}}
        ,
        \\{"dialect":"deepseek_v4","bos_token":"<｜begin▁of▁sentence｜>","eos_token":"<｜end▁of▁sentence｜>"}
        ,
    };

    var buf: [512]u8 = undefined;
    var tok_errs: usize = 0;
    var tpl_errs: usize = 0;
    var scan_errs: usize = 0;

    for (0..FUZZ_ITERS) |iter| {
        const tok_sample = mutateBytes(rng, tok_seeds[iter % tok_seeds.len], &buf);
        if (tokenizer.Tokenizer.fromHuggingFaceJson(gpa, tok_sample)) |tok_val| {
            var tok = tok_val;
            tok.deinit(gpa);
        } else |_| {
            tok_errs += 1;
        }

        const tpl_sample = mutateBytes(rng, tpl_seeds[iter % tpl_seeds.len], &buf);
        if (chat_template.ChatTemplate.fromSpecJson(tpl_sample)) |_| {} else |_| {
            tpl_errs += 1;
        }
        if (chat_template.fuzzJsonScan(tpl_sample)) |_| {} else |_| {
            scan_errs += 1;
        }
    }

    try std.testing.expect(tok_errs > 0);
    try std.testing.expect(tpl_errs > 0);
    try std.testing.expect(scan_errs > 0);
}
