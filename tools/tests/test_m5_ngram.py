"""Unit tests for C6 offline `.ang` v1 NVMe N-gram table builder (`aminfer_tools.ngram`)."""

from __future__ import annotations

import json
import os
import pathlib
import struct
import tempfile
import unittest
import zlib

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.ngram import (
    ANG_CHECKSUM_CRC32_IEEE,
    ANG_FLAG_HAS_TABLE_CRC32,
    ANG_FORMAT_VERSION,
    ANG_MAGIC,
    ANG_PAGE_SIZE,
    ANG_SLOT_BYTES,
    ANG_SLOTS_PER_BUCKET,
    HEADER_STRUCT,
    SLOT_STRUCT,
    ByteLevelBpeTokenizer,
    build_ngram_table,
    compute_header_crc32,
    extract_ngram_records,
    f32_to_f16_bits,
    hash_context,
    iter_qwen_pretokens,
    main,
    write_ang_table,
)


def _make_tiny_tokenizer_json(path: pathlib.Path) -> pathlib.Path:
    """Writes a minimal valid HuggingFace byte-level BPE tokenizer.json."""
    from aminfer_tools.ngram import _BYTE_TO_UNICODE

    vocab: dict[str, int] = {}
    for b, sym in enumerate(_BYTE_TO_UNICODE):
        vocab[sym] = b

    # Add a few multi-byte merges for realistic code tokens (`pub`, `fn`, `const`).
    merges: list[list[str]] = []

    def add_merge(a: str, b: str) -> None:
        merged = a + b
        if merged not in vocab:
            vocab[merged] = len(vocab)
        merges.append([a, b])

    add_merge("p", "u")
    add_merge("pu", "b")
    add_merge("f", "n")
    add_merge("c", "o")
    add_merge("co", "n")
    add_merge("con", "s")
    add_merge("cons", "t")
    add_merge("Ġ", "f")
    add_merge("Ġf", "n")
    add_merge("Ġ", "p")
    add_merge("Ġp", "u")
    add_merge("Ġpu", "b")

    doc = {
        "version": "1.0",
        "added_tokens": [{"id": len(vocab), "content": "<|endoftext|>", "special": True}],
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
        },
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class M5NgramTests(unittest.TestCase):
    def test_hash_context_matches_zig_golden_vectors(self) -> None:
        # Must match `src/spec/ngram_nvme.zig` test vectors bit-for-bit!
        self.assertEqual(hash_context([1, 2]), 0x1E0365D932B40171)
        self.assertEqual(hash_context([10, 20, 30, 40]), 0x48458ECD6D748F49)
        self.assertEqual(hash_context([0]), 0xC3E6F058F572B82B)
        self.assertEqual(hash_context([151643, 151644, 151645]), 0xF82427FE34E25F3B)
        self.assertNotEqual(hash_context([1, 2]), hash_context([2, 1]))
        self.assertNotEqual(hash_context([1, 2, 0]), hash_context([1, 2]))

    def test_qwen_pretokenizer_ascii_and_unicode(self) -> None:
        sample_ascii = "pub fn init(x: u32) !void {\n    // don't panic 42\n}\n"
        chunks = list(iter_qwen_pretokens(sample_ascii))
        self.assertEqual("".join(chunks), sample_ascii)
        self.assertIn("'t", chunks)
        self.assertIn("4", chunks)
        self.assertIn("2", chunks)

        sample_unicode = "你好世界! 123 don't\n  foo"
        u_chunks = list(iter_qwen_pretokens(sample_unicode))
        self.assertEqual("".join(u_chunks), sample_unicode)
        self.assertIn("你好世界", u_chunks)
        self.assertIn("'t", u_chunks)

    def test_greedy_prefix_mode_extracts_highest_probability_continuation(self) -> None:
        # Context [1, 2, 3, 4] followed 4x by [10, 11, 12, 13, 14] (with varying 5th token)
        # and 1x by [99, 98, 97, 96, 95]. Greedy prefix mode must select [10, 11, 12, 13, ...]
        # with hit_count = 5 and step_prob = 4/5 = 0.8!
        seqs = [
            [1, 2, 3, 4, 10, 11, 12, 13, 21],
            [1, 2, 3, 4, 10, 11, 12, 13, 22],
            [1, 2, 3, 4, 10, 11, 12, 13, 21],
            [1, 2, 3, 4, 10, 11, 12, 13, 23],
            [1, 2, 3, 4, 99, 98, 97, 96, 95],
        ]
        records, total_toks = extract_ngram_records(seqs, min_ngram=2, max_ngram=4, max_cont_len=8)
        self.assertEqual(total_toks, 45)
        by_ctx = {r.ctx_tail: r for r in records}
        rec = by_ctx[(1, 2, 3, 4)]
        self.assertEqual(rec.hit_count, 5)
        self.assertAlmostEqual(rec.step_prob, 0.8, places=6)
        self.assertEqual(rec.cont_tokens, (10, 11, 12, 13, 21))

    def test_write_ang_binary_layout_crc_and_bucket_sorting(self) -> None:
        seqs = [
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ]
        records, total_toks = extract_ngram_records(seqs, min_ngram=2, max_ngram=4, max_cont_len=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = pathlib.Path(tmpdir) / "test.ang"
            summary = write_ang_table(
                out_path,
                records,
                num_buckets=16,
                vocab_size=256,
                corpus_id="unit-layout",
                include_table_crc=True,
                source_files=1,
                total_tokens=total_toks,
            )
            raw = out_path.read_bytes()
            self.assertEqual(len(raw), summary.file_bytes)
            self.assertEqual(len(raw), (1 + 16) * APPLE_PAGE_SIZE)
            self.assertEqual(len(raw) % APPLE_PAGE_SIZE, 0)

            hdr = HEADER_STRUCT.unpack_from(raw, 0)
            (
                magic,
                version,
                flags,
                page_size,
                checksum_algo,
                num_buckets,
                slots_per_bucket,
                min_ngram,
                max_ngram,
                max_cont_len,
                reserved_u8,
                vocab_size,
                total_entries,
                table_offset,
                table_bytes,
                table_crc32,
                header_crc32,
                corpus_id,
                reserved1,
            ) = hdr
            self.assertEqual(magic, ANG_MAGIC)
            self.assertEqual(version, ANG_FORMAT_VERSION)
            self.assertEqual(flags, ANG_FLAG_HAS_TABLE_CRC32)
            self.assertEqual(page_size, ANG_PAGE_SIZE)
            self.assertEqual(checksum_algo, ANG_CHECKSUM_CRC32_IEEE)
            self.assertEqual(num_buckets, 16)
            self.assertEqual(slots_per_bucket, ANG_SLOTS_PER_BUCKET)
            self.assertEqual((min_ngram, max_ngram, max_cont_len, reserved_u8), (2, 4, 8, 0))
            self.assertEqual(vocab_size, 256)
            self.assertEqual(total_entries, summary.total_entries)
            self.assertEqual(table_offset, ANG_PAGE_SIZE)
            self.assertEqual(table_bytes, 16 * ANG_PAGE_SIZE)
            self.assertEqual(table_crc32, zlib.crc32(raw[ANG_PAGE_SIZE:]) & 0xFFFFFFFF)
            self.assertEqual(header_crc32, compute_header_crc32(raw[:128]))
            self.assertEqual(corpus_id.rstrip(b"\x00"), b"unit-layout")
            self.assertEqual(reserved1, 0)

            # Verify every bucket page has contiguous occupied slots sorted by hit_count descending.
            counted = 0
            for b in range(num_buckets):
                page_off = ANG_PAGE_SIZE + b * ANG_PAGE_SIZE
                saw_empty = False
                prev_hits = 65536
                for s in range(ANG_SLOTS_PER_BUCKET):
                    rec_u = SLOT_STRUCT.unpack_from(raw, page_off + s * ANG_SLOT_BYTES)
                    ctx_hash = rec_u[0]
                    ctx_tail = rec_u[1:5]
                    cont_len = rec_u[13]
                    ctx_len = rec_u[14]
                    hit_count = rec_u[15]
                    if cont_len == 0:
                        saw_empty = True
                        continue
                    self.assertFalse(saw_empty, "non-empty slot appeared after empty slot")
                    self.assertLessEqual(hit_count, prev_hits)
                    prev_hits = hit_count
                    self.assertEqual(ctx_hash, hash_context(ctx_tail[:ctx_len]))
                    self.assertEqual(ctx_hash & (num_buckets - 1), b)
                    counted += 1
            self.assertEqual(counted, total_entries)

    def test_cli_build_from_source_files_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            tok_path = _make_tiny_tokenizer_json(tmp / "tokenizer.json")
            src_dir = tmp / "repo"
            src_dir.mkdir()
            (src_dir / "sample.zig").write_text(
                "pub fn init(x: u32) void { const y = x + 1; }\n" * 4,
                encoding="utf-8",
            )
            jsonl_path = tmp / "ids.jsonl"
            jsonl_path.write_text(
                json.dumps({"token_ids": [100, 101, 102, 103, 104, 105, 106, 107]}) + "\n",
                encoding="utf-8",
            )
            out_ang = tmp / "out.ang"

            rc = main(
                [
                    "--input",
                    str(src_dir),
                    "--tokenizer",
                    str(tok_path),
                    "--token-ids-jsonl",
                    str(jsonl_path),
                    "--output",
                    str(out_ang),
                    "--corpus-id",
                    "cli-smoke",
                    "--num-buckets",
                    "16",
                    "--json",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_ang.exists())
            self.assertEqual(out_ang.stat().st_size, 17 * APPLE_PAGE_SIZE)

            # Verify JSONL entry [100, 101, 102, 103] -> [104, 105, 106, 107] landed in its bucket!
            raw = out_ang.read_bytes()
            h = hash_context([100, 101, 102, 103])
            b = h & 15
            page_off = ANG_PAGE_SIZE + b * ANG_PAGE_SIZE
            found = False
            for s in range(ANG_SLOTS_PER_BUCKET):
                u = SLOT_STRUCT.unpack_from(raw, page_off + s * ANG_SLOT_BYTES)
                if u[13] == 0:
                    break
                if u[0] == h and u[14] == 4 and u[1:5] == (100, 101, 102, 103):
                    self.assertEqual(u[5 : 5 + u[13]], (104, 105, 106, 107))
                    found = True
                    break
            self.assertTrue(found)

    def test_real_qwen3_tokenizer_if_present(self) -> None:
        models_dir = os.environ.get("AMINFER_MODELS_DIR")
        if not models_dir:
            self.skipTest("AMINFER_MODELS_DIR not set")
        tok_path = pathlib.Path(models_dir) / "Qwen3-0.6B" / "tokenizer.json"
        if not tok_path.exists():
            self.skipTest("Qwen3-0.6B tokenizer.json not present")
        tok = ByteLevelBpeTokenizer.from_file(tok_path)
        ids = tok.encode("pub fn stepRound(self: *SpecEngine) !void {\n    return;\n}\n")
        self.assertGreater(len(ids), 8)
        self.assertTrue(all(0 <= t < tok.vocab_size for t in ids))
        # Verify f32_to_f16_bits round-trip sanity
        bits = f32_to_f16_bits(0.875)
        self.assertAlmostEqual(struct.unpack("<e", struct.pack("<H", bits))[0], 0.875, places=3)

    def test_cli_build_ngram_subcommand(self) -> None:
        from aminfer_tools.cli import main as cli_main

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            jsonl_path = root / "tokens.jsonl"
            jsonl_path.write_text(
                json.dumps({"token_ids": [10, 20, 30, 40, 50, 60, 70, 80]}) + "\n",
                encoding="utf-8",
            )
            out_ang = root / "cli_repo.ang"
            rc = cli_main(
                [
                    "build-ngram",
                    "--token-ids-jsonl",
                    str(jsonl_path),
                    "--output",
                    str(out_ang),
                    "--corpus-id",
                    "cli-delegate",
                    "--num-buckets",
                    "8",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_ang.exists())
            self.assertEqual(out_ang.stat().st_size, 9 * APPLE_PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
