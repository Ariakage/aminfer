"""Offline C6 NVMe N-gram / suffix table builder (`.ang` v1, ADR-0006).

Tokenizes source repositories or pre-tokenized token streams into 16 KiB
page-aligned `.ang` v1 lookup tables queried by `src/spec/ngram_nvme.zig`
via direct I/O (`F_NOCACHE`). Pure Python standard library only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import os
import pathlib
import re
import struct
import sys
import tempfile
import unicodedata
import zlib

from aminfer_tools import APPLE_PAGE_SIZE

ANG_MAGIC = b"ANG0\x00\x00\x00\x00"
ANG_FORMAT_VERSION = 1
ANG_CHECKSUM_CRC32_IEEE = 1
ANG_FLAG_HAS_TABLE_CRC32 = 1 << 0
ANG_PAGE_SIZE = APPLE_PAGE_SIZE
ANG_MAX_CTX_LEN = 4
ANG_MAX_CONT_LEN = 8
ANG_SLOT_BYTES = 64
ANG_SLOTS_PER_BUCKET = ANG_PAGE_SIZE // ANG_SLOT_BYTES  # 256

HEADER_STRUCT = struct.Struct("<8s6I4BI3Q2I48sQ")
SLOT_STRUCT = struct.Struct("<Q4I8IBBHHH")
assert HEADER_STRUCT.size == 128
assert SLOT_STRUCT.size == 64

_U64_MASK = (1 << 64) - 1

DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".zig",
    ".py",
    ".md",
    ".metal",
    ".h",
    ".m",
    ".c",
    ".s",
    ".txt",
    ".zon",
)

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".ikat",
        ".zig-cache",
        "zig-out",
        "__pycache__",
        ".venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
    }
)


def hash_context(ctx: Sequence[int]) -> int:
    """Deterministic 64-bit SplitMix64 context hash matching `ngram_nvme.hashContext`."""
    n = len(ctx)
    if n < 1 or n > ANG_MAX_CTX_LEN:
        raise ValueError(f"context length must be in 1..={ANG_MAX_CTX_LEN}, got {n}")
    h = (0x9E3779B97F4A7C15 ^ ((n * 0xBF58476D1CE4E5B9) & _U64_MASK)) & _U64_MASK
    for tok in ctx:
        if tok < 0 or tok > 0xFFFFFFFF:
            raise ValueError(f"token ID out of u32 range: {tok}")
        x = (tok + 0x9E3779B97F4A7C15) & _U64_MASK
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
        x = (x ^ (x >> 31)) & _U64_MASK
        h = (((h ^ x) * 0xD1B54A32D192ED03) + 0x9E3779B97F4A7C15) & _U64_MASK
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    h = (h ^ (h >> 31)) & _U64_MASK
    return h


def f32_to_f16_bits(value: float) -> int:
    """Encodes a probability in `[0, 1]` as IEEE 754 binary16 bits."""
    clamped = 0.0 if value != value else min(1.0, max(0.0, float(value)))
    return struct.unpack("<H", struct.pack("<e", clamped))[0]


def compute_header_crc32(header_128b: bytes) -> int:
    """Computes IEEE CRC-32 over the 128-byte header with `header_crc32` zeroed."""
    if len(header_128b) != 128:
        raise ValueError(f"expected 128-byte header, got {len(header_128b)}")
    zeroed = header_128b[:68] + b"\x00\x00\x00\x00" + header_128b[72:]
    return zlib.crc32(zeroed) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Pure-Python Qwen2 / Qwen3 Byte-Level BPE Tokenizer (matching tokenizer.zig)
# ---------------------------------------------------------------------------


def _build_byte_to_unicode() -> tuple[str, ...]:
    table: list[str] = [""] * 256
    next_cp = 256
    for b in range(256):
        if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF):
            table[b] = chr(b)
        else:
            table[b] = chr(next_cp)
            next_cp += 1
    return tuple(table)


_BYTE_TO_UNICODE: tuple[str, ...] = _build_byte_to_unicode()

_ASCII_QWEN_RE = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\nA-Za-z0-9]?[A-Za-z]+"
    r"|[0-9]"
    r"| ?[^\sA-Za-z0-9]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _is_letter_or_mark(ch: str) -> bool:
    o = ord(ch)
    if o < 0x80:
        return (65 <= o <= 90) or (97 <= o <= 122)
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "M")


def _is_number(ch: str) -> bool:
    o = ord(ch)
    if o < 0x80:
        return 48 <= o <= 57
    return unicodedata.category(ch)[0] == "N"


def _is_ws(ch: str) -> bool:
    o = ord(ch)
    if o < 0x80:
        return o == 32 or (9 <= o <= 13)
    return ch.isspace()


def _is_punct_or_symbol(ch: str) -> bool:
    return not _is_ws(ch) and not _is_letter_or_mark(ch) and not _is_number(ch)


def iter_qwen_pretokens(text: str) -> Iterable[str]:
    """Yields pre-token slices matching `src/model/tokenizer.zig`'s Qwen scanner."""
    if not text:
        return
    if text.isascii():
        for m in _ASCII_QWEN_RE.finditer(text):
            yield m.group(0)
        return

    n = len(text)
    pos = 0
    while pos < n:
        ch0 = text[pos]
        # 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if ch0 == "'" and pos + 1 < n:
            c1 = text[pos + 1].lower()
            if c1 in ("s", "t", "m", "d"):
                yield text[pos : pos + 2]
                pos += 2
                continue
            if pos + 2 < n:
                pair = text[pos + 1 : pos + 3].lower()
                if pair in ("re", "ve", "ll"):
                    yield text[pos : pos + 3]
                    pos += 3
                    continue

        # 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if _is_letter_or_mark(ch0):
            end = pos + 1
            while end < n and _is_letter_or_mark(text[end]):
                end += 1
            yield text[pos:end]
            pos = end
            continue
        if ch0 not in ("\r", "\n") and not _is_number(ch0) and pos + 1 < n and _is_letter_or_mark(text[pos + 1]):
            end = pos + 2
            while end < n and _is_letter_or_mark(text[end]):
                end += 1
            yield text[pos:end]
            pos = end
            continue

        # 3. \p{N}
        if _is_number(ch0):
            yield ch0
            pos += 1
            continue

        # 4. " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
        p_start: int | None = None
        if _is_punct_or_symbol(ch0):
            p_start = pos + 1
        elif ch0 == " " and pos + 1 < n and _is_punct_or_symbol(text[pos + 1]):
            p_start = pos + 2
        if p_start is not None:
            end = p_start
            while end < n and _is_punct_or_symbol(text[end]):
                end += 1
            while end < n and text[end] in ("\r", "\n"):
                end += 1
            yield text[pos:end]
            pos = end
            continue

        # 5. \s*[\r\n]+ , 6. \s+(?!\S) , 7. \s+
        if _is_ws(ch0):
            ws_end = pos
            last_ws_start = pos
            after_last_nl: int | None = None
            while ws_end < n and _is_ws(text[ws_end]):
                last_ws_start = ws_end
                if text[ws_end] in ("\r", "\n"):
                    after_last_nl = ws_end + 1
                ws_end += 1
            if after_last_nl is not None:
                yield text[pos:after_last_nl]
                pos = after_last_nl
                continue
            if ws_end == n:
                yield text[pos:ws_end]
                pos = ws_end
                continue
            if last_ws_start > pos:
                yield text[pos:last_ws_start]
                pos = last_ws_start
                continue
            yield text[pos:ws_end]
            pos = ws_end
            continue

        yield ch0
        pos += 1


class ByteLevelBpeTokenizer:
    """Minimal pure-Python HuggingFace `tokenizer.json` byte-level BPE encoder."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: Sequence[str | Sequence[str]],
        added_tokens: Sequence[dict[str, object]] | None = None,
    ) -> None:
        self.vocab = vocab
        self.vocab_size = (max(vocab.values()) + 1) if vocab else 0
        self.byte_ids: list[int] = [0] * 256
        for b, sym in enumerate(_BYTE_TO_UNICODE):
            tid = vocab.get(sym)
            if tid is None:
                raise ValueError(f"tokenizer vocab missing byte-level symbol {sym!r} for byte {b}")
            self.byte_ids[b] = tid

        # Map `(left_id, right_id) -> (rank, merged_id)` for fast integer BPE merges.
        self.merges: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, item in enumerate(merges):
            if isinstance(item, str):
                parts = item.split(" ")
                if len(parts) != 2:
                    continue
                left_s, right_s = parts[0], parts[1]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                left_s, right_s = str(item[0]), str(item[1])
            else:
                continue
            lid = vocab.get(left_s)
            rid = vocab.get(right_s)
            mid = vocab.get(left_s + right_s)
            if lid is not None and rid is not None and mid is not None:
                self.merges[(lid, rid)] = (rank, mid)

        self.added_map: dict[str, int] = {}
        if added_tokens:
            for entry in added_tokens:
                content = entry.get("content")
                tid = entry.get("id")
                if isinstance(content, str) and content and isinstance(tid, int):
                    self.added_map[content] = tid
                    if tid + 1 > self.vocab_size:
                        self.vocab_size = tid + 1

        self._added_re: re.Pattern[str] | None = None
        if self.added_map:
            ordered = sorted(self.added_map.keys(), key=lambda s: (-len(s), s))
            self._added_re = re.compile("|".join(re.escape(s) for s in ordered))

        self._cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_file(cls, path: pathlib.Path) -> ByteLevelBpeTokenizer:
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
            raise ValueError(f"unsupported tokenizer.json format in {path}")
        vocab = {str(k): int(v) for k, v in model["vocab"].items()}
        merges = model.get("merges", [])
        added = data.get("added_tokens", [])
        return cls(vocab=vocab, merges=merges, added_tokens=added)

    def _encode_pretoken(self, chunk: str) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        raw = chunk.encode("utf-8")
        if not raw:
            return ()
        byte_ids = self.byte_ids
        syms = [byte_ids[b] for b in raw]
        if len(syms) == 1:
            res = (syms[0],)
            self._cache[chunk] = res
            return res

        merges = self.merges
        while len(syms) >= 2:
            best_rank = 1 << 30
            best_idx = -1
            best_mid = -1
            prev = syms[0]
            for i in range(len(syms) - 1):
                nxt = syms[i + 1]
                m = merges.get((prev, nxt))
                if m is not None and m[0] < best_rank:
                    best_rank = m[0]
                    best_idx = i
                    best_mid = m[1]
                prev = nxt
            if best_idx < 0:
                break
            # Replace all non-overlapping occurrences of the winning pair? No —
            # standard BPE merges the winning rank left-to-right; merging every
            # adjacent pair matching the exact winning pair `(a, b)` in one scan
            # is equivalent and faster.
            a = syms[best_idx]
            b = syms[best_idx + 1]
            out: list[int] = []
            i = 0
            limit = len(syms)
            while i < limit:
                if i + 1 < limit and syms[i] == a and syms[i + 1] == b:
                    out.append(best_mid)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out

        res = tuple(syms)
        if len(self._cache) < 262_144:
            self._cache[chunk] = res
        return res

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        if self._added_re is not None:
            pos = 0
            for m in self._added_re.finditer(text):
                if m.start() > pos:
                    for pre in iter_qwen_pretokens(text[pos : m.start()]):
                        ids.extend(self._encode_pretoken(pre))
                ids.append(self.added_map[m.group(0)])
                pos = m.end()
            if pos < len(text):
                for pre in iter_qwen_pretokens(text[pos:]):
                    ids.extend(self._encode_pretoken(pre))
            return ids

        for pre in iter_qwen_pretokens(text):
            ids.extend(self._encode_pretoken(pre))
        return ids


# ---------------------------------------------------------------------------
# Corpus Collection, N-Gram Extraction & `.ang` v1 Packing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NgramRecord:
    ctx_hash: int
    ctx_tail: tuple[int, ...]
    cont_tokens: tuple[int, ...]
    hit_count: int
    step_prob: float


@dataclass(frozen=True)
class BuildSummary:
    output_path: pathlib.Path
    file_bytes: int
    num_buckets: int
    total_entries: int
    dropped_entries: int
    source_files: int
    total_tokens: int
    min_ngram: int
    max_ngram: int
    max_cont_len: int
    vocab_size: int
    table_crc32: int
    header_crc32: int
    corpus_id: str


def collect_source_files(
    inputs: Sequence[pathlib.Path],
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> list[pathlib.Path]:
    """Deterministically collects matching source files from paths and directories."""
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    found: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for raw in inputs:
        p = raw.resolve()
        if not p.exists():
            raise FileNotFoundError(f"input path does not exist: {raw}")
        if p.is_file():
            if p not in seen:
                seen.add(p)
                found.append(p)
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
            for fname in sorted(files):
                fp = pathlib.Path(root) / fname
                if ext_set and fp.suffix.lower() not in ext_set:
                    continue
                if fp not in seen:
                    seen.add(fp)
                    found.append(fp)
    return found


def load_token_sequences_jsonl(path: pathlib.Path) -> list[list[int]]:
    """Loads pre-tokenized ID sequences from a JSONL file."""
    seqs: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                seq = [int(t) for t in obj]
            elif isinstance(obj, dict) and isinstance(obj.get("token_ids"), list):
                seq = [int(t) for t in obj["token_ids"]]
            else:
                raise ValueError(f"invalid JSONL token record at {path}:{line_no}")
            if seq:
                seqs.append(seq)
    return seqs


def extract_ngram_records(
    token_sequences: Iterable[Sequence[int]],
    *,
    min_ngram: int = 2,
    max_ngram: int = 4,
    max_cont_len: int = 8,
    min_count: int = 1,
) -> tuple[list[NgramRecord], int]:
    """Extracts top-frequency `min_ngram..max_ngram` contexts -> up to `max_cont_len` continuations."""
    if not (1 <= min_ngram <= max_ngram <= ANG_MAX_CTX_LEN):
        raise ValueError(f"invalid ngram bounds: min={min_ngram}, max={max_ngram}")
    if not (1 <= max_cont_len <= ANG_MAX_CONT_LEN):
        raise ValueError(f"invalid max_cont_len: {max_cont_len}")
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")

    # Map `ctx -> Counter[cont_tuple]`
    ctx_conts: dict[tuple[int, ...], Counter[tuple[int, ...]]] = defaultdict(Counter)
    total_tokens = 0

    for seq in token_sequences:
        seq_t = tuple(int(t) for t in seq)
        seq_len = len(seq_t)
        total_tokens += seq_len
        if seq_len <= min_ngram:
            continue
        for i in range(min_ngram, seq_len):
            cont = seq_t[i : min(seq_len, i + max_cont_len)]
            max_n = min(max_ngram, i)
            for n in range(min_ngram, max_n + 1):
                ctx = seq_t[i - n : i]
                ctx_conts[ctx][cont] += 1

    records: list[NgramRecord] = []
    for ctx, counter in ctx_conts.items():
        total_hits = sum(counter.values())
        if total_hits < min_count:
            continue

        # Greedy prefix-mode continuation selection: at each step k=0..max_cont_len-1,
        # pick the highest-frequency token among remaining real corpus continuations.
        # Guarantees every emitted continuation is a real corpus substring and
        # maximizes step-by-step prefix acceptance probability.
        candidates = list(counter.items())
        best_cont: list[int] = []
        first_tok_freq = total_hits
        for pos in range(max_cont_len):
            tok_freq: dict[int, int] = {}
            for cont, freq in candidates:
                if len(cont) > pos:
                    tok = cont[pos]
                    tok_freq[tok] = tok_freq.get(tok, 0) + freq
            if not tok_freq:
                break
            best_tok, best_f = min(tok_freq.items(), key=lambda kv: (-kv[1], kv[0]))
            if pos == 0:
                first_tok_freq = best_f
            best_cont.append(best_tok)
            candidates = [(c, f) for c, f in candidates if len(c) > pos and c[pos] == best_tok]
            if len(candidates) == 1 and len(candidates[0][0]) > pos + 1:
                best_cont.extend(candidates[0][0][pos + 1 :])
                break

        if not best_cont:
            continue

        records.append(
            NgramRecord(
                ctx_hash=hash_context(ctx),
                ctx_tail=ctx,
                cont_tokens=tuple(best_cont[:max_cont_len]),
                hit_count=min(65535, total_hits),
                step_prob=first_tok_freq / float(total_hits),
            )
        )

    return records, total_tokens


def _choose_num_buckets(num_records: int, requested: int) -> int:
    if requested > 0:
        if (requested & (requested - 1)) != 0:
            raise ValueError(f"num_buckets must be a power of two, got {requested}")
        return requested
    # Target <= 192 slots/bucket (75% of 256) with floor 16 and ceiling 65536.
    needed = max(16, (num_records + 191) // 192)
    b = 1
    while b < needed:
        b <<= 1
    return min(65536, b)


def write_ang_table(
    output_path: pathlib.Path,
    records: Sequence[NgramRecord],
    *,
    num_buckets: int = 0,
    min_ngram: int = 2,
    max_ngram: int = 4,
    max_cont_len: int = 8,
    vocab_size: int = 0,
    corpus_id: str = "aminfer-ngram",
    include_table_crc: bool = True,
    source_files: int = 0,
    total_tokens: int = 0,
) -> BuildSummary:
    """Packs `records` into 16 KiB bucket pages and writes `.ang` v1 atomically."""
    if not (1 <= min_ngram <= max_ngram <= ANG_MAX_CTX_LEN):
        raise ValueError(f"invalid ngram bounds: min={min_ngram}, max={max_ngram}")
    if not (1 <= max_cont_len <= ANG_MAX_CONT_LEN):
        raise ValueError(f"invalid max_cont_len: {max_cont_len}")

    buckets_count = _choose_num_buckets(len(records), num_buckets)
    mask = buckets_count - 1
    buckets: list[list[NgramRecord]] = [[] for _ in range(buckets_count)]
    for rec in records:
        buckets[rec.ctx_hash & mask].append(rec)

    total_entries = 0
    dropped_entries = 0
    table_crc = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".ang_tmp_", dir=str(output_path.parent))
    tmp_path = pathlib.Path(tmp_name)

    try:
        with os.fdopen(tmp_fd, "wb") as f:
            # Reserve page 0 for header (rewritten once table_crc and counts are final).
            f.write(b"\x00" * ANG_PAGE_SIZE)

            for bucket in buckets:
                page_buf = bytearray(ANG_PAGE_SIZE)
                if bucket:
                    bucket.sort(
                        key=lambda r: (-r.hit_count, -len(r.ctx_tail), r.ctx_hash, r.ctx_tail)
                    )
                    if len(bucket) > ANG_SLOTS_PER_BUCKET:
                        dropped_entries += len(bucket) - ANG_SLOTS_PER_BUCKET
                        kept = bucket[:ANG_SLOTS_PER_BUCKET]
                    else:
                        kept = bucket
                    total_entries += len(kept)
                    for idx, r in enumerate(kept):
                        ctx4 = list(r.ctx_tail) + [0] * (ANG_MAX_CTX_LEN - len(r.ctx_tail))
                        cont8 = list(r.cont_tokens) + [0] * (ANG_MAX_CONT_LEN - len(r.cont_tokens))
                        SLOT_STRUCT.pack_into(
                            page_buf,
                            idx * ANG_SLOT_BYTES,
                            r.ctx_hash,
                            ctx4[0],
                            ctx4[1],
                            ctx4[2],
                            ctx4[3],
                            cont8[0],
                            cont8[1],
                            cont8[2],
                            cont8[3],
                            cont8[4],
                            cont8[5],
                            cont8[6],
                            cont8[7],
                            len(r.cont_tokens),
                            len(r.ctx_tail),
                            r.hit_count,
                            f32_to_f16_bits(r.step_prob),
                            0,
                        )
                if include_table_crc:
                    table_crc = zlib.crc32(page_buf, table_crc)
                f.write(page_buf)

            table_crc &= 0xFFFFFFFF
            table_bytes = buckets_count * ANG_PAGE_SIZE
            flags = ANG_FLAG_HAS_TABLE_CRC32 if include_table_crc else 0
            cid_raw = corpus_id.encode("utf-8")[:47].ljust(48, b"\x00")
            hdr_no_crc = HEADER_STRUCT.pack(
                ANG_MAGIC,
                ANG_FORMAT_VERSION,
                flags,
                ANG_PAGE_SIZE,
                ANG_CHECKSUM_CRC32_IEEE,
                buckets_count,
                ANG_SLOTS_PER_BUCKET,
                min_ngram,
                max_ngram,
                max_cont_len,
                0,
                vocab_size,
                total_entries,
                ANG_PAGE_SIZE,
                table_bytes,
                table_crc if include_table_crc else 0,
                0,
                cid_raw,
                0,
            )
            header_crc = compute_header_crc32(hdr_no_crc)
            hdr_final = hdr_no_crc[:68] + struct.pack("<I", header_crc) + hdr_no_crc[72:]

            f.seek(0)
            f.write(hdr_final)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return BuildSummary(
        output_path=output_path,
        file_bytes=ANG_PAGE_SIZE + buckets_count * ANG_PAGE_SIZE,
        num_buckets=buckets_count,
        total_entries=total_entries,
        dropped_entries=dropped_entries,
        source_files=source_files,
        total_tokens=total_tokens,
        min_ngram=min_ngram,
        max_ngram=max_ngram,
        max_cont_len=max_cont_len,
        vocab_size=vocab_size,
        table_crc32=table_crc if include_table_crc else 0,
        header_crc32=header_crc,
        corpus_id=corpus_id,
    )


def build_ngram_table(
    output_path: pathlib.Path,
    *,
    inputs: Sequence[pathlib.Path] = (),
    tokenizer_path: pathlib.Path | None = None,
    token_ids_jsonl: pathlib.Path | None = None,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    min_ngram: int = 2,
    max_ngram: int = 4,
    max_cont_len: int = 8,
    min_count: int = 1,
    num_buckets: int = 0,
    corpus_id: str = "aminfer-ngram",
    include_table_crc: bool = True,
) -> BuildSummary:
    """End-to-end offline `.ang` v1 builder from source files or pre-tokenized JSONL."""
    sequences: list[list[int]] = []
    source_files = 0
    vocab_size = 0

    if token_ids_jsonl is not None:
        sequences.extend(load_token_sequences_jsonl(token_ids_jsonl))
        source_files += 1

    if inputs:
        if tokenizer_path is None:
            raise ValueError("--tokenizer is required when building from text/source --input paths")
        tok = ByteLevelBpeTokenizer.from_file(tokenizer_path)
        vocab_size = tok.vocab_size
        files = collect_source_files(inputs, extensions=extensions)
        source_files += len(files)
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            ids = tok.encode(text)
            if ids:
                sequences.append(ids)

    if not sequences:
        raise ValueError("no input tokens produced; provide --input + --tokenizer or --token-ids-jsonl")

    records, total_tokens = extract_ngram_records(
        sequences,
        min_ngram=min_ngram,
        max_ngram=max_ngram,
        max_cont_len=max_cont_len,
        min_count=min_count,
    )
    return write_ang_table(
        output_path,
        records,
        num_buckets=num_buckets,
        min_ngram=min_ngram,
        max_ngram=max_ngram,
        max_cont_len=max_cont_len,
        vocab_size=vocab_size,
        corpus_id=corpus_id,
        include_table_crc=include_table_crc,
        source_files=source_files,
        total_tokens=total_tokens,
    )


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Registers the `build-ngram` subcommand on a parent argparse subparsers group."""
    p = subparsers.add_parser(
        "build-ngram",
        help="Build a 16 KiB-aligned C6 NVMe N-gram table (.ang v1) from source files or token IDs",
    )
    _configure_parser(p)
    return p


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        action="extend",
        nargs="+",
        type=pathlib.Path,
        default=[],
        help="Source files or directories to tokenize (repeatable or space-separated)",
    )
    parser.add_argument(
        "--tokenizer",
        type=pathlib.Path,
        default=None,
        help="HuggingFace tokenizer.json path for byte-level BPE tokenization",
    )
    parser.add_argument(
        "--token-ids-jsonl",
        type=pathlib.Path,
        default=None,
        help="Optional pre-tokenized JSONL file ({'token_ids': [...]} or [...] per line)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Output .ang v1 path (written atomically, 16 KiB page aligned)",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help=f"Comma-separated file extensions when walking directories (default: {','.join(DEFAULT_EXTENSIONS)})",
    )
    parser.add_argument("--min-ngram", type=int, default=2, help="Minimum context n-gram length 1..4 (default: 2)")
    parser.add_argument("--max-ngram", type=int, default=4, help="Maximum context n-gram length 1..4 (default: 4)")
    parser.add_argument("--cont-len", type=int, default=8, help="Maximum continuation length 1..8 (default: 8)")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum corpus hit count to keep (default: 1)")
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=0,
        help="Power-of-two 16 KiB bucket page count (0 = auto-size, default: 0)",
    )
    parser.add_argument(
        "--corpus-id",
        default="aminfer-ngram",
        help="ASCII corpus identifier stored in .ang header (default: aminfer-ngram)",
    )
    parser.add_argument(
        "--no-table-crc",
        action="store_true",
        help="Skip computing optional full-table CRC-32 (header CRC-32 is always written)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON build summary")


def run_args(args: argparse.Namespace) -> int:
    """Executes `build-ngram` from a parsed `argparse.Namespace`."""
    exts = [e.strip() for e in str(args.extensions).split(",") if e.strip()]
    summary = build_ngram_table(
        args.output,
        inputs=list(args.input or ()),
        tokenizer_path=args.tokenizer,
        token_ids_jsonl=args.token_ids_jsonl,
        extensions=exts,
        min_ngram=args.min_ngram,
        max_ngram=args.max_ngram,
        max_cont_len=args.cont_len,
        min_count=args.min_count,
        num_buckets=args.num_buckets,
        corpus_id=args.corpus_id,
        include_table_crc=not bool(args.no_table_crc),
    )
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "output": str(summary.output_path),
                    "file_bytes": summary.file_bytes,
                    "num_buckets": summary.num_buckets,
                    "total_entries": summary.total_entries,
                    "dropped_entries": summary.dropped_entries,
                    "source_files": summary.source_files,
                    "total_tokens": summary.total_tokens,
                    "min_ngram": summary.min_ngram,
                    "max_ngram": summary.max_ngram,
                    "max_cont_len": summary.max_cont_len,
                    "vocab_size": summary.vocab_size,
                    "table_crc32": f"0x{summary.table_crc32:08x}",
                    "header_crc32": f"0x{summary.header_crc32:08x}",
                    "corpus_id": summary.corpus_id,
                },
                indent=2,
            )
        )
    else:
        print(
            f"wrote {summary.output_path} "
            f"({summary.file_bytes / (1024 * 1024):.2f} MiB, "
            f"{summary.num_buckets} buckets, "
            f"{summary.total_entries} entries, "
            f"{summary.dropped_entries} dropped, "
            f"{summary.source_files} files, "
            f"{summary.total_tokens} tokens, "
            f"table_crc32=0x{summary.table_crc32:08x})"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone CLI entry point for `python -m aminfer_tools.ngram`."""
    parser = argparse.ArgumentParser(
        prog="aminfer-tools build-ngram",
        description="Build a 16 KiB-aligned C6 NVMe N-gram table (.ang v1).",
    )
    _configure_parser(parser)
    args = parser.parse_args(argv)
    return run_args(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
