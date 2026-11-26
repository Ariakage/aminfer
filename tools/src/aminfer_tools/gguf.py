"""Pure standard-library GGUF v2/v3 streaming reader and `.amf` v0 repacker (W22).

Zero third-party dependencies (`ADR-0006`, `ADR-0024`). Streams GGUF tensors directly
into 16 KiB-aligned `.amf` blobs with IEEE CRC-32 without double-quantizing raw
GGML blocks, while verifying per-tensor byte spans against sorted offsets / EOF to
hard-fail on unknown or colliding fork types (`STQ1_0` on type 43, `VQ-GGUF`, `EXL3`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DType,
    ENTRY_STRUCT,
    FLAG_EMBEDDED_METADATA,
    LEGACY_CHAT_TEMPLATE_METADATA_NAME,
    META_CHAT_TEMPLATE_JSON,
    META_CONFIG_JSON,
    META_PREFIX,
    META_TOKENIZER_JSON,
    NO_EXPERT,
    NO_LAYER,
    SME2_GROUP_K,
    SME2_PACK_MODES,
    SME2_PANEL_COLS,
    PackedTensorMeta,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    _align_up,
    _bf16_to_f32_bytes,
    _BlobWriter,
    _classify_tensor,
    _crc32,
    _natural_key,
    _normalize_name,
    _pack_directory,
    _pack_header,
    _set_nocache,
    _should_sme2_pack,
    bf16_bytes_to_f16_bytes,
    bf16_bytes_to_f32_list,
    f16_bytes_to_bf16_bytes,
    f16_bytes_to_f32_list,
    f32_to_bf16_bytes,
    f32_to_f16_bytes,
    is_inkling2_controller_tensor,
    is_vision_tensor,
    load_cold_experts_from_heat_profile,
    pack_sme2_panel64_f16_panel,
    pack_vision_tower_amf,
    quantize_row_f32_to_e4m3_row,
    sme2_panel64_data_bytes,
)
from aminfer_tools.chat_template import compile_chat_template_spec

GGUF_MAGIC = b"GGUF"
GGUF_SUPPORTED_VERSIONS = (2, 3)

# Standard upstream GGUF Metadata Value Types
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_SCALAR_STRUCTS: dict[int, struct.Struct] = {
    _GGUF_TYPE_UINT8: struct.Struct("<B"),
    _GGUF_TYPE_INT8: struct.Struct("<b"),
    _GGUF_TYPE_UINT16: struct.Struct("<H"),
    _GGUF_TYPE_INT16: struct.Struct("<h"),
    _GGUF_TYPE_UINT32: struct.Struct("<I"),
    _GGUF_TYPE_INT32: struct.Struct("<i"),
    _GGUF_TYPE_FLOAT32: struct.Struct("<f"),
    _GGUF_TYPE_BOOL: struct.Struct("<B"),
    _GGUF_TYPE_UINT64: struct.Struct("<Q"),
    _GGUF_TYPE_INT64: struct.Struct("<q"),
    _GGUF_TYPE_FLOAT64: struct.Struct("<d"),
}

# Canonical upstream ggml_type IDs -> (name, block_size, block_bytes, amf_dtype_or_None)
_GGML_TYPE_INFO: dict[int, tuple[str, int, int, DType | None]] = {
    0: ("F32", 1, 4, DType.F32),
    1: ("F16", 1, 2, DType.F16),
    2: ("Q4_0", 32, 18, DType.GGUF_Q4_0),
    3: ("Q4_1", 32, 20, DType.GGUF_Q4_1),
    6: ("Q5_0", 32, 22, DType.GGUF_Q5_0),
    7: ("Q5_1", 32, 24, DType.GGUF_Q5_1),
    8: ("Q8_0", 32, 34, DType.GGUF_Q8_0),
    10: ("Q2_K", 256, 84, DType.GGUF_Q2_K),
    11: ("Q3_K", 256, 110, DType.GGUF_Q3_K),
    12: ("Q4_K", 256, 144, DType.GGUF_Q4_K),
    13: ("Q5_K", 256, 176, DType.GGUF_Q5_K),
    14: ("Q6_K", 256, 210, DType.GGUF_Q6_K),
    16: ("IQ2_XXS", 256, 66, DType.GGUF_IQ2_XXS),
    17: ("IQ2_XS", 256, 74, DType.GGUF_IQ2_XS),
    18: ("IQ3_XXS", 256, 98, DType.GGUF_IQ3_XXS),
    19: ("IQ1_S", 256, 50, DType.GGUF_IQ1_S),
    20: ("IQ4_NL", 32, 18, DType.GGUF_IQ4_NL),
    21: ("IQ3_S", 256, 110, DType.GGUF_IQ3_S),
    22: ("IQ2_S", 256, 82, DType.GGUF_IQ2_S),
    23: ("IQ4_XS", 256, 136, DType.GGUF_IQ4_XS),
    29: ("IQ1_M", 256, 56, DType.GGUF_IQ1_M),
    30: ("BF16", 1, 2, DType.BF16),
    34: ("TQ1_0", 256, 54, DType.GGUF_TQ1_0),
    35: ("TQ2_0", 256, 66, DType.GGUF_TQ2_0),
    39: ("MXFP4", 32, 17, DType.GGUF_MXFP4),
    40: ("NVFP4", 64, 36, DType.NVFP4_G16),
    42: ("Q2_0", 64, 18, DType.GGUF_Q2_0),
    43: ("F8_E4M3", 1, 1, DType.FP8_E4M3_TENSOR),
}

_AMF_GGUF_BLOCK_INFO: dict[DType, tuple[int, int]] = {
    amf_dt: (bs, bb)
    for g_id, (_gname, bs, bb, amf_dt) in _GGML_TYPE_INFO.items()
    if amf_dt is not None and amf_dt >= DType.GGUF_Q4_0
}

# Nominal quant_bits recorded in `.amf` directory (`src/quant/formats.zig:ggufBlockInfo`).
_AMF_GGUF_QUANT_BITS: dict[DType, int] = {
    DType.GGUF_Q4_0: 4,
    DType.GGUF_Q4_1: 4,
    DType.GGUF_Q5_0: 5,
    DType.GGUF_Q5_1: 5,
    DType.GGUF_Q8_0: 8,
    DType.GGUF_Q2_K: 2,
    DType.GGUF_Q3_K: 3,
    DType.GGUF_Q4_K: 4,
    DType.GGUF_Q5_K: 5,
    DType.GGUF_Q6_K: 6,
    DType.GGUF_IQ1_S: 1,
    DType.GGUF_IQ1_M: 1,
    DType.GGUF_IQ2_XXS: 2,
    DType.GGUF_IQ2_XS: 2,
    DType.GGUF_IQ2_S: 2,
    DType.GGUF_IQ3_XXS: 3,
    DType.GGUF_IQ3_S: 3,
    DType.GGUF_IQ4_NL: 4,
    DType.GGUF_IQ4_XS: 4,
    DType.GGUF_TQ1_0: 2,
    DType.GGUF_TQ2_0: 2,
    DType.GGUF_MXFP4: 4,
    DType.GGUF_Q2_0: 2,
}

_FORK_SIGNATURES = ("stq1_0", "vq-gguf", "vq_gguf", "exl3", "exllama3")

# Verbatim IQ lookup tables extracted from `src/quant/formats.zig` (`33792 B` = 256+512+1024 u64 + 256+512 u32 + 2048 u64).
_IQ_TABLES_B64 = (
    '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'
    't2jAcqLA/vnARsKg6SkeGLw/JRDscIcaGz2mYgfeYJVyggEleGiV1KhX8I16PTICahYtAQEDa3V6NWct1etheHOiwkQJNXQc+69Q'
    'B5/VWpmIoz8E4QrwMbuxvsLAbyU8RNLaaspWcyVKBjsH0CxHpfV+CWFWnd/vqqXTB35px1D//+G/O7cnt8vl5+fB7fciN+/cLg/u'
    '+cfH78/Pi9uFPw9u4+E7t8v98vO6y4t3bpfn5ef5WoOMz5Pb5SID3rldHvefB7dfbpcX99xev/c58i+3Memd2+vO338fPPSTu+fP'
    '88n968lrefLzT37+xc+/eF5ur9fP6/H+eb7f/P7vz5Pbm9vj+ftzef/OFT9k+WPJYzljGXMzPM3lyU2HnJsbKx5/Y5P3u3wZqxpv'
    'j8+4+fPgb7+y+bsS4PJ4XN5Pbm/+zqt4cft5cbu/Lk9ubybO5ffF9988BBPpKYQa1LkICSfRBqned5mU6T8nHkvl5U1ijhWMz5j6'
    'wUP9cmOaTyKP1y9zdCH4+GNaMyPuTCZZ+Nj98yJMeHJ7PZ5MRm5M9d87c/T54vuDs7987/fOtLy/nrLZn7lz2fRgJdPs8XrzPWXe'
    'XRk4Pk+e5M1tbI1vTmrzEnhCbr9CsPHHjGK2Pvkec2Jy8sUrfj1eL17Z5f0YnwcT9fFggXgPaZiUe74e3HNjEeDlsoj8srDc55ZY'
    'YLi97ywSfM0TvJ68HH7k8mJJej95h+POawoVS9JzSMB9SMFj9D/MMBUwltL3aPcLL4ffvszVC6F46+PDEsQLfrIY87C/zyHRr8tj'
    'COWbqfo7ifm8//CsP+83t+d7CMDlPT7MJ76+MwXnDt9jdeNFZpZs+ud3cnhoAy/ml9VwCLQwhlfI65oCP6RxTDZWNjSWF/A7CPbz'
    'fvAUgzLPqcVD21gWmSTvX+K/Hbc9tzO3m97YcdtzO3G76gM7bmduhwNfn/gWtzO3G7cdv3wYA1z5PrcdD3TidtMBx2fP7aSD77id'
    'dZIDtxu33YHvcTvxpDue9Mxtf5PJx9tnbjddyIHbldtuz/f3Y1F7OnHbn/ia243bjhd7GAu+cT8af3a7A8/LG9gfaM9tx5s5cLty'
    'O534/pnvc7td+bcbP3uTzR52J7px2/GmT9z2B970ie9z2ykRxgpPg4p7Xim3K7cDE2V3OvOazvwI37vyvRv3/Nntrjwu94crv8L9'
    '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'
)

_IQ_LOADED = False
_IQ2XXS_GRID: tuple[int, ...] = ()
_IQ2XS_GRID: tuple[int, ...] = ()
_IQ2S_GRID: tuple[int, ...] = ()
_IQ3XXS_GRID: tuple[int, ...] = ()
_IQ3S_GRID: tuple[int, ...] = ()
_IQ1S_GRID: tuple[int, ...] = ()

_KSIGNS_IQ2XS: tuple[int, ...] = tuple(i | (0x80 if i.bit_count() & 1 else 0) for i in range(128))
_KMASK_IQ2XS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
_KVALUES_IQ4NL: tuple[int, ...] = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
_KVALUES_MXFP4: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _ensure_iq_tables() -> None:
    global _IQ_LOADED, _IQ2XXS_GRID, _IQ2XS_GRID, _IQ2S_GRID, _IQ3XXS_GRID, _IQ3S_GRID, _IQ1S_GRID
    if _IQ_LOADED:
        return
    raw = zlib.decompress(base64.b64decode(_IQ_TABLES_B64))
    off = 0

    def take_u64(n: int) -> tuple[int, ...]:
        nonlocal off
        res = struct.unpack_from(f"<{n}Q", raw, off)
        off += n * 8
        return res

    def take_u32(n: int) -> tuple[int, ...]:
        nonlocal off
        res = struct.unpack_from(f"<{n}I", raw, off)
        off += n * 4
        return res

    _IQ2XXS_GRID = take_u64(256)
    _IQ2XS_GRID = take_u64(512)
    _IQ2S_GRID = take_u64(1024)
    _IQ3XXS_GRID = take_u32(256)
    _IQ3S_GRID = take_u32(512)
    _IQ1S_GRID = take_u64(2048)
    assert off == len(raw)
    _IQ_LOADED = True


@dataclass(frozen=True)
class GgufTensorInfo:
    name: str
    dims: tuple[int, ...]  # GGML order `[ne0, ne1, ...]` (`ne0` = innermost row width K)
    ggml_type: int
    offset: int  # relative to shard `data_offset`
    expected_bytes: int
    actual_span_bytes: int

    @property
    def amf_shape(self) -> tuple[int, ...]:
        """Convert GGML inner-first `dims` to row-major `.amf` `[N, K]` shape."""
        return tuple(reversed(self.dims))

    @property
    def numel(self) -> int:
        p = 1
        for d in self.dims:
            p *= d
        return p


@dataclass(frozen=True)
class GgufShard:
    path: pathlib.Path
    version: int
    tensor_count: int
    kv: dict[str, Any]
    alignment: int
    data_offset: int
    file_size: int
    tensors: tuple[GgufTensorInfo, ...]


class _GgufStreamCursor:
    """Bounded-buffer forward reader for GGUF headers and KV sections."""

    def __init__(self, fh) -> None:
        self._fh = fh
        self.pos = 0

    def read_exact(self, n: int) -> bytes:
        if n == 0:
            return b""
        buf = self._fh.read(n)
        if len(buf) != n:
            raise ValueError(f"unexpected EOF in GGUF header at offset {self.pos} (wanted {n}, got {len(buf)})")
        self.pos += n
        return buf

    def read_u8(self) -> int:
        return self.read_exact(1)[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def read_str(self) -> str:
        n = self.read_u64()
        if n > (64 << 20):
            raise ValueError(f"implausible GGUF string length {n}")
        return self.read_exact(n).decode("utf-8", errors="replace")

    def read_value(self, vtype: int) -> Any:
        if vtype == _GGUF_TYPE_STRING:
            return self.read_str()
        if vtype == _GGUF_TYPE_BOOL:
            return bool(self.read_u8())
        if vtype == _GGUF_TYPE_ARRAY:
            elem_type = self.read_u32()
            count = self.read_u64()
            if count > 10_000_000:
                raise ValueError(f"implausible GGUF array length {count}")
            if elem_type in _SCALAR_STRUCTS:
                st = _SCALAR_STRUCTS[elem_type]
                raw = self.read_exact(count * st.size)
                return list(struct.unpack(f"<{count}{st.format[-1]}", raw)) if count else []
            if elem_type == _GGUF_TYPE_STRING:
                return [self.read_str() for _ in range(count)]
            return [self.read_value(elem_type) for _ in range(count)]
        st = _SCALAR_STRUCTS.get(vtype)
        if st is None:
            raise ValueError(f"unknown GGUF metadata value type {vtype}")
        return st.unpack(self.read_exact(st.size))[0]


def _check_fork_metadata(kv: dict[str, Any]) -> None:
    for k, v in kv.items():
        lk = k.lower()
        lv = str(v).lower() if not isinstance(v, list) else ""
        for sig in _FORK_SIGNATURES:
            if sig in lk or sig in lv:
                raise ValueError(f"unsupported GGUF fork metadata detected ({sig!r} in {k!r})")


def expected_ggml_tensor_bytes(ggml_type: int, dims: Sequence[int], name: str = "<tensor>") -> int:
    """Compute exact GGML packed byte count for `dims = (ne0, ne1, ...)`."""
    spec = _GGML_TYPE_INFO.get(ggml_type)
    if spec is None:
        raise ValueError(
            f"unsupported or colliding fork ggml_type {ggml_type} on tensor {name!r} "
            "(supported: F32/F16/BF16, standard Q*/K*/IQ*/TQ*/MXFP4/Q2_0, NVFP4=40, F8_E4M3=43)"
        )
    tname, bs, bb, _ = spec
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"tensor {name!r} has invalid non-positive dims {list(dims)}")
    ne0 = dims[0]
    if ne0 % bs != 0:
        raise ValueError(f"tensor {name!r} ({tname}) row width ne0={ne0} not divisible by block size {bs}")
    rows = 1
    for d in dims[1:]:
        rows *= d
    return (ne0 // bs) * bb * rows


def read_gguf_header(path: pathlib.Path) -> GgufShard:
    """Read and validate one GGUF v2/v3 shard header, KV table, and exact tensor byte spans."""
    p = pathlib.Path(path)
    file_size = p.stat().st_size
    if file_size < 24:
        raise ValueError(f"file is too small to be GGUF: {p}")

    with p.open("rb") as fh:
        cur = _GgufStreamCursor(fh)
        magic = cur.read_exact(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"invalid GGUF magic {magic!r} in {p}")
        version = cur.read_u32()
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported GGUF version {version} in {p} (expected 2 or 3)")
        tensor_count = cur.read_u64()
        kv_count = cur.read_u64()
        if tensor_count > 1_000_000 or kv_count > 1_000_000:
            raise ValueError(f"implausible GGUF counts tensors={tensor_count} kv={kv_count} in {p}")

        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = cur.read_str()
            vtype = cur.read_u32()
            kv[key] = cur.read_value(vtype)
        _check_fork_metadata(kv)

        alignment = int(kv.get("general.alignment", 32))
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError(f"invalid GGUF general.alignment={alignment} in {p}")

        raw_infos: list[tuple[str, tuple[int, ...], int, int]] = []
        seen_names: set[str] = set()
        for _ in range(tensor_count):
            tname = cur.read_str()
            if not tname or tname in seen_names:
                raise ValueError(f"invalid or duplicate GGUF tensor name {tname!r} in {p}")
            seen_names.add(tname)
            ndims = cur.read_u32()
            if not (1 <= ndims <= 4):
                raise ValueError(f"GGUF tensor {tname!r} has invalid n_dims={ndims} in {p}")
            dims = tuple(cur.read_u64() for _ in range(ndims))
            gtype = cur.read_u32()
            toff = cur.read_u64()
            raw_infos.append((tname, dims, gtype, toff))

        data_offset = _align_up(cur.pos, alignment)
        if data_offset > file_size:
            raise ValueError(f"GGUF data_offset {data_offset} exceeds file size {file_size} in {p}")

    # Mandatory per-tensor actual_tensor_bytes == expected_ggml_bytes verification across sorted offsets/EOF.
    sorted_idx = sorted(range(len(raw_infos)), key=lambda i: raw_infos[i][3])
    validated: list[GgufTensorInfo | None] = [None] * len(raw_infos)
    total_data_area = file_size - data_offset
    for rank_i, orig_idx in enumerate(sorted_idx):
        tname, dims, gtype, toff = raw_infos[orig_idx]
        if toff % alignment != 0:
            raise ValueError(f"GGUF tensor {tname!r} offset {toff} misaligned to general.alignment={alignment}")
        exp_bytes = expected_ggml_tensor_bytes(gtype, dims, name=tname)
        next_off = raw_infos[sorted_idx[rank_i + 1]][3] if rank_i + 1 < len(sorted_idx) else total_data_area
        actual_span = next_off - toff
        padded_expected = _align_up(exp_bytes, alignment)
        if actual_span < exp_bytes:
            extra_hint = (
                " (colliding fork type such as STQ1_0 overloading type 43 at ~1.312 bpw rejected)"
                if gtype == 43
                else ""
            )
            raise ValueError(
                f"GGUF byte-size mismatch for {tname!r} (ggml_type={gtype} {_GGML_TYPE_INFO[gtype][0]}): "
                f"actual_tensor_bytes={actual_span} < expected_ggml_bytes={exp_bytes}{extra_hint}"
            )
        if actual_span > padded_expected:
            raise ValueError(
                f"GGUF byte-size mismatch for {tname!r} (ggml_type={gtype} {_GGML_TYPE_INFO[gtype][0]}): "
                f"actual_tensor_bytes={actual_span} > expected_ggml_bytes={exp_bytes} (aligned {padded_expected})"
            )
        validated[orig_idx] = GgufTensorInfo(
            name=tname,
            dims=dims,
            ggml_type=gtype,
            offset=toff,
            expected_bytes=exp_bytes,
            actual_span_bytes=actual_span,
        )

    return GgufShard(
        path=p,
        version=version,
        tensor_count=tensor_count,
        kv=kv,
        alignment=alignment,
        data_offset=data_offset,
        file_size=file_size,
        tensors=tuple(v for v in validated if v is not None),
    )


_SPLIT_GGUF_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def resolve_gguf_shards(input_path: pathlib.Path) -> list[GgufShard]:
    """Resolve a single `.gguf`, split multi-shard `-00001-of-0000N.gguf`, or directory."""
    p = pathlib.Path(input_path)
    if p.is_dir():
        ggufs = sorted(p.glob("*.gguf"))
        if not ggufs:
            raise FileNotFoundError(f"no .gguf files found in directory {p}")
        split_matches = [m for m in (_SPLIT_GGUF_RE.match(g.name) for g in ggufs) if m is not None]
        if split_matches:
            first = min((g for g in ggufs if _SPLIT_GGUF_RE.match(g.name)), key=lambda g: g.name)
            return resolve_gguf_shards(first)
        if len(ggufs) == 1:
            return [read_gguf_header(ggufs[0])]
        raise ValueError(f"ambiguous multiple non-split .gguf files in {p}: {[g.name for g in ggufs]}")

    if not p.is_file():
        raise FileNotFoundError(f"GGUF file not found: {p}")

    m = _SPLIT_GGUF_RE.match(p.name)
    if m is not None:
        prefix, _, total_str = m.group(1), int(m.group(2)), m.group(3)
        total = int(total_str)
        shard_paths: list[pathlib.Path] = []
        for i in range(1, total + 1):
            cand = p.with_name(f"{prefix}-{i:05d}-of-{total_str}.gguf")
            if not cand.is_file():
                raise FileNotFoundError(f"missing split GGUF shard: {cand}")
            shard_paths.append(cand)
        return [read_gguf_header(sp) for sp in shard_paths]
    return [read_gguf_header(p)]


# ---------------------------------------------------------------------------
# GGUF Block Layout Metadata & Pure-Stdlib Exact fp32 Row Decoders (`15..37`)
# ---------------------------------------------------------------------------


def gguf_row_bytes_for_amf_dtype(dtype: DType, k: int) -> int:
    spec = _AMF_GGUF_BLOCK_INFO.get(dtype)
    if spec is None:
        raise ValueError(f"{dtype.name} is not a raw GGUF block dtype")
    bs, bb = spec
    if k <= 0 or k % bs != 0:
        raise ValueError(f"{dtype.name} requires k > 0 divisible by {bs}, got {k}")
    return (k // bs) * bb


def _read_f16(raw: memoryview | bytes, off: int) -> float:
    return float(struct.unpack_from("<e", raw, off)[0])


def _read_u16(raw: memoryview | bytes, off: int) -> int:
    return struct.unpack_from("<H", raw, off)[0]


def _read_u32(raw: memoryview | bytes, off: int) -> int:
    return struct.unpack_from("<I", raw, off)[0]


def _read_i8(raw: memoryview | bytes, off: int) -> int:
    b = raw[off]
    return b - 256 if b >= 128 else b


def _get_scale_min_k4(j: int, q: memoryview) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequantize_gguf_amf_row_f32(dtype: DType, row_bytes: bytes | bytearray | memoryview, k: int) -> list[float]:
    """Decode one row of raw GGML blocks (`DType.GGUF_Q4_0..GGUF_Q2_0`) to fp32."""
    bs, bb = _AMF_GGUF_BLOCK_INFO[dtype]
    if k <= 0 or k % bs != 0:
        raise ValueError(f"{dtype.name} row width k={k} not divisible by block size {bs}")
    nb = k // bs
    if len(row_bytes) != nb * bb:
        raise ValueError(f"{dtype.name} row length {len(row_bytes)} != expected {nb * bb}")
    mv = memoryview(row_bytes)
    out = [0.0] * k

    if dtype == DType.GGUF_Q4_0:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            base = b * 32
            for j in range(16):
                byte = blk[2 + j]
                out[base + j] = ((byte & 0x0F) - 8) * d
                out[base + j + 16] = ((byte >> 4) - 8) * d
        return out

    if dtype == DType.GGUF_Q4_1:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            m = _read_f16(blk, 2)
            base = b * 32
            for j in range(16):
                byte = blk[4 + j]
                out[base + j] = (byte & 0x0F) * d + m
                out[base + j + 16] = (byte >> 4) * d + m
        return out

    if dtype == DType.GGUF_Q5_0:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            qh = _read_u32(blk, 2)
            base = b * 32
            for j in range(16):
                byte = blk[6 + j]
                xh0 = ((qh >> j) << 4) & 0x10
                xh1 = (qh >> (j + 12)) & 0x10
                out[base + j] = (((byte & 0x0F) | xh0) - 16) * d
                out[base + j + 16] = (((byte >> 4) | xh1) - 16) * d
        return out

    if dtype == DType.GGUF_Q5_1:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            m = _read_f16(blk, 2)
            qh = _read_u32(blk, 4)
            base = b * 32
            for j in range(16):
                byte = blk[8 + j]
                xh0 = ((qh >> j) << 4) & 0x10
                xh1 = (qh >> (j + 12)) & 0x10
                out[base + j] = ((byte & 0x0F) | xh0) * d + m
                out[base + j + 16] = ((byte >> 4) | xh1) * d + m
        return out

    if dtype == DType.GGUF_Q8_0:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            base = b * 32
            for j in range(32):
                out[base + j] = _read_i8(blk, 2 + j) * d
        return out

    if dtype == DType.GGUF_Q2_0:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            base = b * 64
            for j in range(64):
                q = (blk[2 + j // 4] >> ((j % 4) * 2)) & 0x03
                out[base + j] = (q - 1) * d
        return out

    if dtype == DType.GGUF_MXFP4:
        tbl = _KVALUES_MXFP4
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            e = blk[0]
            d = math.nan if e == 0xFF else math.ldexp(1.0, e - 127) * 0.5
            base = b * 32
            for j in range(16):
                byte = blk[1 + j]
                out[base + j] = tbl[byte & 0x0F] * d
                out[base + j + 16] = tbl[byte >> 4] * d
        return out

    if dtype == DType.GGUF_Q2_K:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            scales = blk[0:16]
            d = _read_f16(blk, 80)
            mn = _read_f16(blk, 82)
            y_off = b * 256
            q_off = 16
            is_i = 0
            for _ in range(2):
                shift = 0
                for _ in range(4):
                    sc0 = scales[is_i]
                    is_i += 1
                    dl0 = d * (sc0 & 0x0F)
                    ml0 = mn * (sc0 >> 4)
                    for l in range(16):
                        out[y_off + l] = dl0 * ((blk[q_off + l] >> shift) & 3) - ml0
                    y_off += 16
                    sc1 = scales[is_i]
                    is_i += 1
                    dl1 = d * (sc1 & 0x0F)
                    ml1 = mn * (sc1 >> 4)
                    for l in range(16):
                        out[y_off + l] = dl1 * ((blk[q_off + 16 + l] >> shift) & 3) - ml1
                    y_off += 16
                    shift += 2
                q_off += 32
        return out

    if dtype == DType.GGUF_Q3_K:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            hm = blk[0:32]
            qs = blk[32:96]
            d_all = _read_f16(blk, 108)
            a0, a1, a2 = struct.unpack_from("<3I", blk, 96)
            km1, km2 = 0x03030303, 0x0F0F0F0F
            aux = [
                (a0 & km2) | (((a2 >> 0) & km1) << 4),
                (a1 & km2) | (((a2 >> 2) & km1) << 4),
                ((a0 >> 4) & km2) | (((a2 >> 4) & km1) << 4),
                ((a1 >> 4) & km2) | (((a2 >> 6) & km1) << 4),
            ]
            sc_bytes = struct.unpack("<16b", struct.pack("<4I", *aux))
            y_off = b * 256
            q_off = 0
            is_i = 0
            m = 1
            for _ in range(2):
                shift = 0
                for _ in range(4):
                    dl0 = d_all * (sc_bytes[is_i] - 32)
                    is_i += 1
                    for l in range(16):
                        q2 = (qs[q_off + l] >> shift) & 3
                        out[y_off + l] = dl0 * (q2 - (0 if (hm[l] & m) else 4))
                    y_off += 16
                    dl1 = d_all * (sc_bytes[is_i] - 32)
                    is_i += 1
                    for l in range(16):
                        q2 = (qs[q_off + 16 + l] >> shift) & 3
                        out[y_off + l] = dl1 * (q2 - (0 if (hm[16 + l] & m) else 4))
                    y_off += 16
                    shift += 2
                    m <<= 1
                q_off += 32
        return out

    if dtype == DType.GGUF_Q4_K:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            mn = _read_f16(blk, 2)
            scales = blk[4:16]
            qs = blk[16:144]
            y_off = b * 256
            q_off = 0
            is_i = 0
            for _ in range(4):
                sc0, m0 = _get_scale_min_k4(is_i, scales)
                sc1, m1 = _get_scale_min_k4(is_i + 1, scales)
                d1, mm1 = d * sc0, mn * m0
                d2, mm2 = d * sc1, mn * m1
                for l in range(32):
                    out[y_off + l] = d1 * (qs[q_off + l] & 0x0F) - mm1
                    out[y_off + 32 + l] = d2 * (qs[q_off + l] >> 4) - mm2
                q_off += 32
                y_off += 64
                is_i += 2
        return out

    if dtype == DType.GGUF_Q5_K:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            mn = _read_f16(blk, 2)
            scales = blk[4:16]
            qh = blk[16:48]
            ql = blk[48:176]
            y_off = b * 256
            q_off = 0
            is_i = 0
            mask1, mask2 = 1, 2
            for _ in range(4):
                sc0, m0 = _get_scale_min_k4(is_i, scales)
                sc1, m1 = _get_scale_min_k4(is_i + 1, scales)
                d1, mm1 = d * sc0, mn * m0
                d2, mm2 = d * sc1, mn * m1
                for l in range(32):
                    h1 = 16 if (qh[l] & mask1) else 0
                    h2 = 16 if (qh[l] & mask2) else 0
                    out[y_off + l] = d1 * ((ql[q_off + l] & 0x0F) + h1) - mm1
                    out[y_off + 32 + l] = d2 * ((ql[q_off + l] >> 4) + h2) - mm2
                q_off += 32
                y_off += 64
                is_i += 2
                mask1 <<= 2
                mask2 <<= 2
        return out

    if dtype == DType.GGUF_Q6_K:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 208)
            ql_off, qh_off, sc_off, y_off = 0, 128, 192, b * 256
            for _ in range(2):
                for l in range(32):
                    is_i = l // 16
                    qh_b = blk[qh_off + l]
                    q1 = ((blk[ql_off + l] & 0x0F) | (((qh_b >> 0) & 3) << 4)) - 32
                    q2 = ((blk[ql_off + l + 32] & 0x0F) | (((qh_b >> 2) & 3) << 4)) - 32
                    q3 = ((blk[ql_off + l] >> 4) | (((qh_b >> 4) & 3) << 4)) - 32
                    q4 = ((blk[ql_off + l + 32] >> 4) | (((qh_b >> 6) & 3) << 4)) - 32
                    out[y_off + l + 0] = d * _read_i8(blk, sc_off + is_i + 0) * q1
                    out[y_off + l + 32] = d * _read_i8(blk, sc_off + is_i + 2) * q2
                    out[y_off + l + 64] = d * _read_i8(blk, sc_off + is_i + 4) * q3
                    out[y_off + l + 96] = d * _read_i8(blk, sc_off + is_i + 6) * q4
                y_off += 128
                ql_off += 64
                qh_off += 32
                sc_off += 8
        return out

    if dtype == DType.GGUF_IQ4_NL:
        tbl4 = _KVALUES_IQ4NL
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            base = b * 32
            for j in range(16):
                byte = blk[2 + j]
                out[base + j] = d * tbl4[byte & 0x0F]
                out[base + j + 16] = d * tbl4[byte >> 4]
        return out

    if dtype == DType.GGUF_IQ4_XS:
        tbl4 = _KVALUES_IQ4NL
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            scales_h = _read_u16(blk, 2)
            scales_l = blk[4:8]
            qs = blk[8:136]
            base = b * 256
            for ib in range(8):
                ls = ((scales_l[ib // 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4)
                dl = d * (ls - 32)
                q_sub = qs[ib * 16 : (ib + 1) * 16]
                y_sub = base + ib * 32
                for j in range(16):
                    out[y_sub + j] = dl * tbl4[q_sub[j] & 0x0F]
                    out[y_sub + j + 16] = dl * tbl4[q_sub[j] >> 4]
        return out

    if dtype == DType.GGUF_TQ1_0:
        pow3 = (1, 3, 9, 27, 81)
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            qs, qh = blk[0:48], blk[48:52]
            d = _read_f16(blk, 52)
            y_off = b * 256
            for n in range(5):
                for m in range(32):
                    xi = (((qs[m] * pow3[n]) & 0xFF) * 3) >> 8
                    out[y_off] = (xi - 1) * d
                    y_off += 1
            for n in range(5):
                for m in range(16):
                    xi = (((qs[32 + m] * pow3[n]) & 0xFF) * 3) >> 8
                    out[y_off] = (xi - 1) * d
                    y_off += 1
            for n in range(4):
                for j in range(4):
                    xi = (((qh[j] * pow3[n]) & 0xFF) * 3) >> 8
                    out[y_off] = (xi - 1) * d
                    y_off += 1
        return out

    if dtype == DType.GGUF_TQ2_0:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            qs = blk[0:64]
            d = _read_f16(blk, 64)
            y_off = b * 256
            for j in (0, 32):
                for l in range(4):
                    for m in range(32):
                        q = (qs[j + m] >> (l * 2)) & 3
                        out[y_off] = (q - 1) * d
                        y_off += 1
        return out

    _ensure_iq_tables()
    ksigns = _KSIGNS_IQ2XS
    kmask = _KMASK_IQ2XS

    if dtype == DType.GGUF_IQ1_S:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            qs = blk[2:34]
            q_off, y_off = 0, b * 256
            for ib in range(8):
                qh_w = _read_u16(blk, 34 + ib * 2)
                dl = d * (2 * ((qh_w >> 12) & 7) + 1)
                delta = -0.125 if (qh_w & 0x8000) else 0.125
                for l in range(4):
                    grid = _IQ1S_GRID[qs[q_off + l] | (((qh_w >> (3 * l)) & 7) << 8)]
                    for j in range(8):
                        gb = (grid >> (j * 8)) & 0xFF
                        gs = gb - 256 if gb >= 128 else gb
                        out[y_off + j] = dl * (gs + delta)
                    y_off += 8
                q_off += 4
        return out

    if dtype == DType.GGUF_IQ1_M:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            sc_bytes = blk[48:56]
            sc0, sc1, sc2, sc3 = struct.unpack_from("<4H", sc_bytes, 0)
            d_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
            d = float(struct.unpack("<e", struct.pack("<H", d_bits))[0])
            qs, qh = blk[0:32], blk[32:48]
            q_off, qh_off, y_off = 0, 0, b * 256
            for ib in range(8):
                sc_w = _read_u16(sc_bytes, (ib // 2) * 2)
                sb = 6 * (ib % 2)
                dl1 = d * (2 * ((sc_w >> sb) & 7) + 1)
                dl2 = d * (2 * ((sc_w >> (sb + 3)) & 7) + 1)
                idxs = (
                    qs[q_off + 0] | ((qh[qh_off + 0] << 8) & 0x700),
                    qs[q_off + 1] | ((qh[qh_off + 0] << 4) & 0x700),
                    qs[q_off + 2] | ((qh[qh_off + 1] << 8) & 0x700),
                    qs[q_off + 3] | ((qh[qh_off + 1] << 4) & 0x700),
                )
                deltas = (
                    -0.125 if (qh[qh_off + 0] & 0x08) else 0.125,
                    -0.125 if (qh[qh_off + 0] & 0x80) else 0.125,
                    -0.125 if (qh[qh_off + 1] & 0x08) else 0.125,
                    -0.125 if (qh[qh_off + 1] & 0x80) else 0.125,
                )
                for l in range(4):
                    dl = dl1 if l < 2 else dl2
                    grid = _IQ1S_GRID[idxs[l]]
                    dl_d = deltas[l]
                    for j in range(8):
                        gb = (grid >> (j * 8)) & 0xFF
                        gs = gb - 256 if gb >= 128 else gb
                        out[y_off + j] = dl * (gs + dl_d)
                    y_off += 8
                q_off += 4
                qh_off += 2
        return out

    if dtype == DType.GGUF_IQ2_XXS:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            y_off = b * 256
            for ib32 in range(8):
                aux0, aux1 = struct.unpack_from("<2I", blk, 2 + ib32 * 8)
                db = d * (0.5 + (aux1 >> 28)) * 0.25
                for l in range(4):
                    grid = _IQ2XXS_GRID[(aux0 >> (l * 8)) & 0xFF]
                    signs = ksigns[(aux1 >> (7 * l)) & 127]
                    for j in range(8):
                        gb = (grid >> (j * 8)) & 0xFF
                        out[y_off + j] = db * gb * (-1.0 if (signs & kmask[j]) else 1.0)
                    y_off += 8
        return out

    if dtype == DType.GGUF_IQ2_XS:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            scales = blk[66:74]
            y_off = b * 256
            for ib32 in range(8):
                db = (d * (0.5 + (scales[ib32] & 0x0F)) * 0.25, d * (0.5 + (scales[ib32] >> 4)) * 0.25)
                for l in range(4):
                    qsw = _read_u16(blk, 2 + (4 * ib32 + l) * 2)
                    grid = _IQ2XS_GRID[qsw & 511]
                    signs = ksigns[qsw >> 9]
                    dl = db[l // 2]
                    for j in range(8):
                        gb = (grid >> (j * 8)) & 0xFF
                        out[y_off + j] = dl * gb * (-1.0 if (signs & kmask[j]) else 1.0)
                    y_off += 8
        return out

    if dtype == DType.GGUF_IQ2_S:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            qs_all, qh, scales = blk[2:66], blk[66:74], blk[74:82]
            qs_off, signs_off, y_off = 0, 32, b * 256
            for ib32 in range(8):
                db = (d * (0.5 + (scales[ib32] & 0x0F)) * 0.25, d * (0.5 + (scales[ib32] >> 4)) * 0.25)
                for l in range(4):
                    dl = db[l // 2]
                    qh_hi = (qh[ib32] << (8 - 2 * l)) & 0x300
                    grid = _IQ2S_GRID[qs_all[qs_off + l] | qh_hi]
                    sb = qs_all[signs_off + l]
                    for j in range(8):
                        gb = (grid >> (j * 8)) & 0xFF
                        out[y_off + j] = dl * gb * (-1.0 if (sb & kmask[j]) else 1.0)
                    y_off += 8
                qs_off += 4
                signs_off += 4
        return out

    if dtype == DType.GGUF_IQ3_XXS:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            qs, sc_signs = blk[2:66], blk[66:98]
            qs_off, y_off = 0, b * 256
            for ib32 in range(8):
                aux32 = _read_u32(sc_signs, 4 * ib32)
                db = d * (0.5 + (aux32 >> 28)) * 0.5
                for l in range(4):
                    signs = ksigns[(aux32 >> (7 * l)) & 127]
                    g1 = _IQ3XXS_GRID[qs[qs_off + 2 * l]]
                    g2 = _IQ3XXS_GRID[qs[qs_off + 2 * l + 1]]
                    for j in range(4):
                        out[y_off + j] = db * ((g1 >> (j * 8)) & 0xFF) * (-1.0 if (signs & kmask[j]) else 1.0)
                        out[y_off + 4 + j] = db * ((g2 >> (j * 8)) & 0xFF) * (-1.0 if (signs & kmask[4 + j]) else 1.0)
                    y_off += 8
                qs_off += 8
        return out

    if dtype == DType.GGUF_IQ3_S:
        for b in range(nb):
            blk = mv[b * bb : (b + 1) * bb]
            d = _read_f16(blk, 0)
            qs, qh, signs_all, scales = blk[2:66], blk[66:74], blk[74:106], blk[106:110]
            qs_off, qh_off, signs_off, y_off = 0, 0, 0, b * 256
            for pair in range(4):
                sc_b = scales[pair]
                dbs = (d * (1 + 2 * (sc_b & 0x0F)), d * (1 + 2 * (sc_b >> 4)))
                for half_i in range(2):
                    db = dbs[half_i]
                    qb = qh[qh_off + half_i]
                    for l in range(4):
                        g1 = _IQ3S_GRID[qs[qs_off + 2 * l] | ((qb << (8 - 2 * l)) & 256)]
                        g2 = _IQ3S_GRID[qs[qs_off + 2 * l + 1] | ((qb << (7 - 2 * l)) & 256)]
                        sb = signs_all[signs_off + l]
                        for j in range(4):
                            out[y_off + j] = db * ((g1 >> (j * 8)) & 0xFF) * (-1.0 if (sb & kmask[j]) else 1.0)
                            out[y_off + 4 + j] = db * ((g2 >> (j * 8)) & 0xFF) * (-1.0 if (sb & kmask[4 + j]) else 1.0)
                        y_off += 8
                    qs_off += 8
                    signs_off += 4
                qh_off += 2
        return out

    raise ValueError(f"unsupported GGUF AMF dtype {dtype.name}")


# ---------------------------------------------------------------------------
# Tensor Name Normalization, Unrolled Recurrent Deduplication & Metadata Sync
# ---------------------------------------------------------------------------

_GGUF_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
_GGUF_VISION_BLK_RE = re.compile(r"^v\.blk\.(\d+)\.(.+)$")
_GGUF_EXP_RE = re.compile(r"^(ffn_gate_up|ffn_gate|ffn_up|ffn_down)\.(\d+)\.(weight|bias)$")
_GGUF_INKLING2_CTRL_RE = re.compile(
    r"^loop_(?:(q|k|v|ffn)_ctrl_(down|forget|gain)|(attn|k|v|mlp)_(down|forget|gain))\.weight$"
)

# Per-block `(offset_of_d, optional_offset_of_min)` for scaling GGUF blocks by a per-expert F32 multiplier.
_GGUF_BLOCK_SCALE_OFFSETS: dict[DType, tuple[int, ...]] = {
    DType.GGUF_Q4_0: (0,),
    DType.GGUF_Q4_1: (0, 2),
    DType.GGUF_Q5_0: (0,),
    DType.GGUF_Q5_1: (0, 2),
    DType.GGUF_Q8_0: (0,),
    DType.GGUF_Q2_K: (80, 82),
    DType.GGUF_Q3_K: (108,),
    DType.GGUF_Q4_K: (0, 2),
    DType.GGUF_Q5_K: (0, 2),
    DType.GGUF_Q6_K: (208,),
    DType.GGUF_IQ1_S: (0,),
    DType.GGUF_IQ2_XXS: (0,),
    DType.GGUF_IQ2_XS: (0,),
    DType.GGUF_IQ2_S: (0,),
    DType.GGUF_IQ3_XXS: (0,),
    DType.GGUF_IQ3_S: (0,),
    DType.GGUF_IQ4_NL: (0,),
    DType.GGUF_IQ4_XS: (0,),
    DType.GGUF_TQ1_0: (52,),
    DType.GGUF_TQ2_0: (64,),
    DType.GGUF_Q2_0: (0,),
}

_GGUF_LEAF_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_norm.bias": "input_layernorm.bias",
    "attn_q_inp_norm.weight": "q_input_norm.weight",
    "attn_k_inp_norm.weight": "k_input_norm.weight",
    "attn_v_inp_norm.weight": "v_input_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.bias": "post_attention_layernorm.bias",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_post_norm.weight": "mlp_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_r.weight": "attn.wr_du.weight",
    "attn_rel.weight": "attn.wr_du.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "attn_qkv.weight": "self_attn.qkv_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_v_norm.weight": "v_input_norm.weight",
    "attn_gate.weight": "self_attn.gate_proj.weight",
    "attn_sink.weight": "attn.attn_sink_proj.weight",
    "attn_sink_proj.weight": "attn.attn_sink_proj.weight",
    "attn_sinks.weight": "attn.attn_sink_proj.weight",
    "attn_rel_logits.weight": "attn.rel_logits_proj.proj",
    "attn_rel_b.weight": "attn.rel_logits_proj.proj",
    "attn_rel_bias.weight": "attn.rel_logits_proj.proj",
    "attn_q_clip.scale": "attn.q_l2_clip.scale",
    "attn_q_clip.weight": "attn.q_l2_clip.scale",
    "attn_rel_clip.scale": "attn.rel_l2_clip.scale",
    "attn_rel_clip.weight": "attn.rel_l2_clip.scale",
    "attn_sconv.weight": "attn_sconv.weight",
    "attn_post_conv.weight": "attn_sconv.weight",
    "mlp_sconv.weight": "mlp_sconv.weight",
    "ffn_post_conv.weight": "mlp_sconv.weight",
    "attn_k_sconv.weight": "attn.k_sconv.weight",
    "attn_k_conv.weight": "attn.k_sconv.weight",
    "attn_v_sconv.weight": "attn.v_sconv.weight",
    "attn_v_conv.weight": "attn.v_sconv.weight",
    "attn_k_conv_proj.weight": "attn.k_sconv.weight_proj",
    "attn_v_conv_proj.weight": "attn.v_sconv.weight_proj",
    "dflare_norm.weight": "dflare_norm.weight",
    "attn_k_norm_ctx.weight": "self_attn.k_norm_ctx.weight",
    "attn_k_ctx.weight": "self_attn.k_proj_ctx.weight",
    "attn_v_ctx.weight": "self_attn.v_proj_ctx.weight",
    # Hy4 / Hy4-flash GGUF mappings
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attn_kv_b.weight": "self_attn.kv_b_proj.weight",
    "attn_linear_gate.weight": "self_attn.linear_gate.weight",
    "attn_post_norm.weight": "self_attn.post_proj_norm.weight",
    "ffn_combine_norm.weight": "mlp.combine_output_norm.weight",
    "ffn_routed_up.weight": "mlp.routed_expert_up_proj.weight",
    "ffn_routed_down.weight": "mlp.routed_expert_down_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp.bias": "mlp.gate.bias",
    "exp_probs_b.bias": "mlp.gate.bias",
    "ffn_exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_shexp_inp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
}

# Inkling 2 `M82Runtime` (`src/model/m82_runtime.zig`) overrides generic `self_attn.*` names.
_GGUF_INKLING2_LEAF_MAP = {
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "mlp_norm.weight",
    "attn_q.weight": "attn.wq_du.weight",
    "attn_k.weight": "attn.wk_dv.weight",
    "attn_v.weight": "attn.wv_dv.weight",
    "attn_output.weight": "attn.wo_ud.weight",
    "attn_rel.weight": "attn.wr_du.weight",
    "attn_gate.weight": "attn.gate_attn.weight",
}

_GGUF_VISION_LEAF_MAP = {
    "attn_qkv.weight": "attn.qkv.weight",
    "attn_qkv.bias": "attn.qkv.bias",
    "attn_q.weight": "attn.q_proj.weight",
    "attn_q.bias": "attn.q_proj.bias",
    "attn_k.weight": "attn.k_proj.weight",
    "attn_k.bias": "attn.k_proj.bias",
    "attn_v.weight": "attn.v_proj.weight",
    "attn_v.bias": "attn.v_proj.bias",
    "attn_out.weight": "attn.proj.weight",
    "attn_out.bias": "attn.proj.bias",
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
    "ffn_up.weight": "mlp.fc1.weight",
    "ffn_up.bias": "mlp.fc1.bias",
    "ffn_down.weight": "mlp.fc2.weight",
    "ffn_down.bias": "mlp.fc2.bias",
}

_GGUF_TOP_MAP = {
    "token_embd.weight": "embed_tokens.weight",
    "token_embd_norm.weight": "embed_norm.weight",
    "embed_norm.weight": "embed_norm.weight",
    "output_norm.weight": "norm.weight",
    "output_norm.bias": "norm.bias",
    "output.weight": "lm_head.weight",
    "output.bias": "lm_head.bias",
    "output_res_query.weight": "attn_res.output_query",
    "per_layer_token_embd.weight": "engram.per_layer_token_embd.weight",
    "dflash.fuse_weight": "fuse_weight",
    "fuse_weight": "fuse_weight",
    "dflash.mask_embed": "mask_embed",
    "mask_embed": "mask_embed",
    "middle_cycle_cross_pass_queries": "middle_cycle_cross_pass_queries",
    "middle_cycle_cross_pass_kv_queries": "middle_cycle_cross_pass_kv_queries",
    "v.patch_embd.weight": "vision.patch_embed.proj.weight",
    "v.patch_embd.bias": "vision.patch_embed.proj.bias",
    "v.position_embd.weight": "vision.pos_embed.weight",
    "v.post_ln.weight": "vision.norm.weight",
    "v.post_ln.bias": "vision.norm.bias",
}


def normalize_gguf_tensor_name(
    raw_name: str,
    *,
    is_clip: bool = False,
    layer_remap: dict[int, int] | None = None,
    inkling2_runtime_names: bool = False,
    attn_res_shifted_by_1: bool = False,
    arch: str = "",
    mimo_mtp_mode: bool = False,
    linear_layers: set[int] | None = None,
) -> str:
    """Normalize a GGUF tensor name (`blk.N.*`, `v.blk.N.*`, `mm.*`) to canonical `.amf` naming."""
    eff_arch = arch.lower()
    if eff_arch == "dflash":
        dflash_top = {
            "fc.weight": "fc.weight",
            "dflash.fc.weight": "fc.weight",
            "dflare.fc.weight": "fc.weight",
            "hidden_norm.weight": "hidden_norm.weight",
            "enc.output_norm.weight": "hidden_norm.weight",
            "fc_norm.weight": "hidden_norm.weight",
            "dflash.hidden_norm.weight": "hidden_norm.weight",
            "dflare.fc_norm.weight": "hidden_norm.weight",
            "output_norm.weight": "norm.weight",
            "norm.weight": "norm.weight",
            "dflare.norm.weight": "norm.weight",
            "mask_embed": "mask_embed.weight",
            "dflash.mask_embed": "mask_embed.weight",
            "mask_embd.weight": "mask_embed.weight",
            "mask_embed.weight": "mask_embed.weight",
            "candidate_selector.hidden_projection.weight": "candidate_selector.hidden_projection.weight",
            "candidate_selector.predecessor_codebook": "candidate_selector.predecessor_codebook",
            "candidate_selector.predecessor_codebook.weight": "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook": "candidate_selector.successor_codebook",
            "candidate_selector.successor_codebook.weight": "candidate_selector.successor_codebook",
        }
        if raw_name in dflash_top:
            return _normalize_name(dflash_top[raw_name])
    if eff_arch in ("mimo2", "mimo_v2") and mimo_mtp_mode:
        mn = re.match(r"^(?:model\.)?(?:mtp\.)?nextn\.(\d+)\.(.+)$", raw_name)
        if mn is not None:
            k_idx = str(int(mn.group(1)))
            n_rest = mn.group(2)
            mtp_rest_map = {
                "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
                "self_attn.attention_sink_bias": "self_attn.sink_bias",
                "output_norm.weight": "final_layernorm.weight",
            }
            n_rest = mtp_rest_map.get(n_rest, n_rest)
            return _normalize_name(f"layers.{k_idx}.{n_rest}")
    if raw_name in _GGUF_TOP_MAP:
        return _normalize_name(_GGUF_TOP_MAP[raw_name])
    mv = _GGUF_VISION_BLK_RE.match(raw_name)
    if mv is not None:
        blk_idx, rest = mv.group(1), mv.group(2)
        leaf = _GGUF_VISION_LEAF_MAP.get(rest, rest)
        return _normalize_name(f"vision.blocks.{blk_idx}.{leaf}")
    if raw_name.startswith("mm."):
        return _normalize_name("vision.merger." + raw_name[3:])
    if is_clip and not is_vision_tensor(raw_name):
        return _normalize_name(f"vision.{raw_name}")
    mb = _GGUF_BLK_RE.match(raw_name)
    if mb is not None:
        raw_blk_idx = int(mb.group(1))
        phys_blk_idx = layer_remap.get(raw_blk_idx, raw_blk_idx) if layer_remap is not None else raw_blk_idx
        blk_idx = str(phys_blk_idx)
        rest = mb.group(2)
        if eff_arch in ("dflash", "mimo2", "mimo_v2"):
            if rest in (
                "attn_sinks.weight",
                "attn_sink.weight",
                "attn_sink_bias",
                "self_attn.attention_sink_bias",
                "self_attn.sink_bias",
            ):
                return _normalize_name(f"layers.{blk_idx}.self_attn.sink_bias")
            if eff_arch in ("mimo2", "mimo_v2"):
                if rest in ("exp_probs_b.bias", "ffn_exp_probs_b.bias", "ffn_gate_inp.bias"):
                    return _normalize_name(f"layers.{blk_idx}.mlp.gate.bias")
                if mimo_mtp_mode:
                    if rest in ("nextn.enorm.weight", "enorm.weight"):
                        return _normalize_name(f"layers.{blk_idx}.enorm.weight")
                    if rest in ("nextn.hnorm.weight", "hnorm.weight"):
                        return _normalize_name(f"layers.{blk_idx}.hnorm.weight")
                    if rest in ("nextn.eh_proj.weight", "eh_proj.weight"):
                        return _normalize_name(f"layers.{blk_idx}.eh_proj.weight")
                    if rest in ("pre_mlp_layernorm.weight", "ffn_norm.weight", "post_attention_norm.weight"):
                        return _normalize_name(f"layers.{blk_idx}.post_attention_layernorm.weight")
                    if rest in ("final_layernorm.weight", "nextn.norm.weight", "ffn_post_norm.weight", "output_norm.weight"):
                        return _normalize_name(f"layers.{blk_idx}.final_layernorm.weight")
        if eff_arch == "dflash" and ("attention_conv." in rest or "attn_conv." in rest or "mlp_conv." in rest):
            df_rest = rest.replace("attention_conv.", "attn_conv.")
            if len(f"layers.{blk_idx}.{df_rest}".encode("utf-8")) > 47:
                df_rest = df_rest.replace(".kernel_projection.", ".k_proj.")
            return _normalize_name(f"layers.{blk_idx}.{df_rest}")
        if eff_arch in ("glm-dsa", "glm_moe_dsa", "glm5-next", "glm5_next"):
            glm_leaf_map = {
                "indexer.attn_k.weight": "attn.idx.wk.weight",
                "indexer.wk.weight": "attn.idx.wk.weight",
                "indexer.attn_q_b.weight": "attn.idx.wq_b.weight",
                "indexer.wq_b.weight": "attn.idx.wq_b.weight",
                "indexer.k_norm.weight": "attn.idx.k_norm.weight",
                "indexer.k_norm.bias": "attn.idx.k_norm.bias",
                "indexer.norm.weight": "attn.idx.k_norm.weight",
                "indexer.norm.bias": "attn.idx.k_norm.bias",
                "indexer.proj.weight": "attn.idx.weights_proj.weight",
                "indexer.weights_proj.weight": "attn.idx.weights_proj.weight",
                "indexer_compressor_ape": "attn.idx.kpool_ape",
                "indexer_compressor_ape.weight": "attn.idx.kpool_ape",
                "indexer.kpool_ape": "attn.idx.kpool_ape",
                "indexer_compressor_gate": "attn.idx.kpool_gate",
                "indexer_compressor_gate.weight": "attn.idx.kpool_gate",
                "indexer.kpool_gate": "attn.idx.kpool_gate",
                "exp_probs_b.bias": "mlp.gate.bias",
                "ffn_exp_probs_b.bias": "mlp.gate.bias",
                "ffn_gate_inp.bias": "mlp.gate.bias",
                "hc_attn_base": "hc_attn_base",
                "hc_attn_base.weight": "hc_attn_base",
                "hc_attn_fn": "hc_attn_fn",
                "hc_attn_fn.weight": "hc_attn_fn",
                "hc_attn_scale": "hc_attn_scale",
                "hc_attn_scale.weight": "hc_attn_scale",
                "hc_ffn_base": "hc_ffn_base",
                "hc_ffn_base.weight": "hc_ffn_base",
                "hc_ffn_fn": "hc_ffn_fn",
                "hc_ffn_fn.weight": "hc_ffn_fn",
                "hc_ffn_scale": "hc_ffn_scale",
                "hc_ffn_scale.weight": "hc_ffn_scale",
                "attn_q_a.weight": "self_attn.q_a_proj.weight",
                "attn_q_b.weight": "self_attn.q_b_proj.weight",
                "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
                "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
                "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
                "attn_kv_b.weight": "self_attn.kv_b_proj.weight",
            }
            if eff_arch in ("glm5-next", "glm5_next"):
                glm_leaf_map.update(
                    {
                        "ssm_a": "linear_attn.A_log",
                        "ssm_a.weight": "linear_attn.A_log",
                        "ssm_dt.bias": "linear_attn.dt_bias",
                        "ssm_dt_bias": "linear_attn.dt_bias",
                        "ssm_dt.weight": "linear_attn.dt_bias",
                        "ssm_beta.weight": "linear_attn.b_proj.weight",
                        "ssm_f_a.weight": "linear_attn.f_a_proj.weight",
                        "ssm_f_b.weight": "linear_attn.f_b_proj.weight",
                        "ssm_g_a.weight": "linear_attn.g_a_proj.weight",
                        "ssm_g_b.weight": "linear_attn.g_b_proj.weight",
                        "ssm_conv1d_q.weight": "linear_attn.q_conv1d.weight",
                        "ssm_conv1d_k.weight": "linear_attn.k_conv1d.weight",
                        "ssm_conv1d_v.weight": "linear_attn.v_conv1d.weight",
                        "ssm_norm.weight": "linear_attn.o_norm.weight",
                        "ssm_norm": "linear_attn.o_norm.weight",
                        "ssm_out.weight": "linear_attn.o_proj.weight",
                    }
                )
                if linear_layers is not None and raw_blk_idx in linear_layers:
                    glm_leaf_map.update(
                        {
                            "attn_q.weight": "linear_attn.q_proj.weight",
                            "attn_k.weight": "linear_attn.k_proj.weight",
                            "attn_v.weight": "linear_attn.v_proj.weight",
                            "attn_output.weight": "linear_attn.o_proj.weight",
                        }
                    )
            if rest in glm_leaf_map:
                return _normalize_name(f"layers.{blk_idx}.{glm_leaf_map[rest]}")
            if mimo_mtp_mode:
                if rest in ("nextn.enorm.weight", "enorm.weight"):
                    return _normalize_name(f"layers.{blk_idx}.enorm.weight")
                if rest in ("nextn.hnorm.weight", "hnorm.weight"):
                    return _normalize_name(f"layers.{blk_idx}.hnorm.weight")
                if rest in ("nextn.eh_proj.weight", "eh_proj.weight"):
                    return _normalize_name(f"layers.{blk_idx}.eh_proj.weight")
                if rest in ("nextn.norm.weight", "shared_head.norm.weight", "output_norm.weight"):
                    return _normalize_name("shared_head.norm.weight")
        if eff_arch in ("qwen35", "qwen3_5", "qwen35moe"):
            qwen35_leaf_map = {
                "attn_gate.weight": "linear_attn.g_proj.weight",
                "ssm_out.weight": "linear_attn.o_proj.weight",
                "ssm_alpha.weight": "linear_attn.a_proj.weight",
                "ssm_beta.weight": "linear_attn.b_proj.weight",
                "ssm_conv1d.weight": "linear_attn.conv1d.weight",
                "ssm_conv1d": "linear_attn.conv1d.weight",
                "ssm_a": "linear_attn.A_log",
                "ssm_a.weight": "linear_attn.A_log",
                "ssm_dt.bias": "linear_attn.dt_bias",
                "ssm_dt_bias": "linear_attn.dt_bias",
                "ssm_dt.weight": "linear_attn.dt_bias",
                "ssm_norm.weight": "linear_attn.norm.weight",
                "ssm_norm": "linear_attn.norm.weight",
            }
            if rest in qwen35_leaf_map:
                return _normalize_name(f"layers.{blk_idx}.{qwen35_leaf_map[rest]}")
        if rest == "ffn_res_query.weight":
            return _normalize_name(f"attn_res.mlp_queries.{blk_idx}")
        if rest == "attn_res_query.weight":
            aq_idx = max(0, phys_blk_idx - 1) if attn_res_shifted_by_1 else phys_blk_idx
            return _normalize_name(f"attn_res.attn_queries.{aq_idx}")
        mc = _GGUF_INKLING2_CTRL_RE.match(rest)
        if mc is not None:
            if mc.group(1) is not None:
                slot = {"q": "attn", "k": "k", "v": "v", "ffn": "mlp"}[mc.group(1)]
                part = mc.group(2)
            else:
                slot = mc.group(3)
                part = mc.group(4)
            return _normalize_name(f"layers.{blk_idx}.controllers.{slot}.{part}_weight")
        if inkling2_runtime_names and rest in _GGUF_INKLING2_LEAF_MAP:
            return _normalize_name(f"layers.{blk_idx}.{_GGUF_INKLING2_LEAF_MAP[rest]}")
        if rest in _GGUF_LEAF_MAP:
            return _normalize_name(f"layers.{blk_idx}.{_GGUF_LEAF_MAP[rest]}")
        me = _GGUF_EXP_RE.match(rest)
        if me is not None:
            proj_map = {
                "ffn_gate": "gate_proj",
                "ffn_up": "up_proj",
                "ffn_down": "down_proj",
                "ffn_gate_up": "w13" if inkling2_runtime_names else "gate_up_proj",
            }
            return _normalize_name(f"layers.{blk_idx}.mlp.experts.{me.group(2)}.{proj_map[me.group(1)]}.{me.group(3)}")
        return _normalize_name(f"layers.{blk_idx}.{rest}")
    return _normalize_name(raw_name)


def _tensor_content_digest(fh, data_offset: int, t: GgufTensorInfo) -> bytes:
    """Compute SHA-256 of one tensor payload incrementally without buffering huge arrays."""
    h = hashlib.sha256()
    fh.seek(data_offset + t.offset)
    left = t.expected_bytes
    while left > 0:
        chunk = fh.read(min(1 << 20, left))
        if not chunk:
            raise ValueError(f"unexpected EOF reading {t.name!r}")
        h.update(chunk)
        left -= len(chunk)
    return h.digest()


def _inkling2_loop_params(kv: dict[str, Any], arch: str, total_layers: int) -> tuple[int, int, int] | None:
    """Return `(core_start, core_end, num_passes)` if GGUF KV or layer count reflects an Inkling 2 loop."""
    c_start = kv.get(f"{arch}.loop.start", kv.get(f"{arch}.loop.core_start"))
    c_end = kv.get(f"{arch}.loop.end", kv.get(f"{arch}.loop.core_end"))
    n_pass = kv.get(f"{arch}.loop.count", kv.get(f"{arch}.loop.num_passes"))
    if c_start is not None and c_end is not None and n_pass is not None:
        return int(c_start), int(c_end), int(n_pass)
    if arch in ("inkling2", "inkling2_loop", "inkling2_loop_text") and total_layers == 48:
        return 6, 18, 3
    return None


def _is_inkling2_per_pass_leaf(leaf: str) -> bool:
    if leaf in ("attn_res_query.weight", "ffn_res_query.weight", "loop_msg_query.weight"):
        return True
    if _GGUF_INKLING2_CTRL_RE.match(leaf) is not None:
        return True
    return leaf.endswith("_norm.weight") or leaf.endswith("_norm.bias")


def _deduplicate_unrolled_recurrent_layers_ex(
    entries: Sequence[tuple[int, GgufShard, GgufTensorInfo]],
    kv: dict[str, Any],
) -> tuple[
    list[tuple[int, GgufShard, GgufTensorInfo]],
    dict[tuple[int, str], list[tuple[int, tuple[int, GgufShard, GgufTensorInfo]]]],
    int,
    int,
]:
    """Deduplicate unrolled recurrent `blk.N.*` aliases (`48 -> 24`), preserving per-pass deltas."""
    arch = str(kv.get("general.architecture") or "").lower()
    reported_blocks = kv.get(f"{arch}.block_count") if arch else None
    layer_buckets: dict[int, list[tuple[int, GgufShard, GgufTensorInfo]]] = {}
    non_layer: list[tuple[int, GgufShard, GgufTensorInfo]] = []
    for item in entries:
        _, _, t = item
        m = _GGUF_BLK_RE.match(t.name)
        if m is None:
            non_layer.append(item)
        else:
            layer_buckets.setdefault(int(m.group(1)), []).append(item)

    if len(layer_buckets) <= 1:
        return list(entries), {}, len(layer_buckets), 0

    loop_params = _inkling2_loop_params(kv, arch, len(layer_buckets))
    core_start, core_end, n_passes = loop_params if loop_params is not None else (0, 0, 1)
    core_len = max(0, core_end - core_start)
    max_unrolled_idx = core_start + core_len * n_passes if core_len > 0 and n_passes > 1 else -1

    sig_to_first_layer: dict[tuple[tuple[str, tuple[int, ...], int, int | bytes], ...], int] = {}
    first_core_sigs: dict[int, dict[str, tuple[tuple[int, ...], int, bytes]]] = {}
    per_pass_deltas: dict[tuple[int, str], list[tuple[int, tuple[int, GgufShard, GgufTensorInfo]]]] = {}
    opened: dict[int, Any] = {}
    kept_layers: dict[int, list[tuple[int, GgufShard, GgufTensorInfo]]] = {}
    dedup_count = 0
    try:
        for _, sh, _ in entries:
            if id(sh) not in opened:
                opened[id(sh)] = sh.path.open("rb")
        for l_idx in sorted(layer_buckets):
            items = sorted(layer_buckets[l_idx], key=lambda x: x[2].name)
            # Build layer signature using (leaf_name, dims, ggml_type, physical_ptr_or_digest).
            sig_parts: list[tuple[str, tuple[int, ...], int, int | bytes]] = []
            leaf_sigs: dict[str, tuple[tuple[int, ...], int, bytes]] = {}
            for s_idx, sh, t in items:
                m = _GGUF_BLK_RE.match(t.name)
                assert m is not None
                leaf = m.group(2)
                digest = _tensor_content_digest(opened[id(sh)], sh.data_offset, t)
                sig_parts.append((leaf, t.dims, t.ggml_type, digest))
                leaf_sigs[leaf] = (t.dims, t.ggml_type, digest)
            sig = tuple(sig_parts)
            # Only collapse duplicate layers in the repeated unrolled core band (when loop schedule is known)
            # or in the upper half / alias band for generic unrolled GGUFs.
            prev_layer = sig_to_first_layer.get(sig)
            in_unrolled_loop_tail = (
                core_len > 0
                and n_passes > 1
                and core_start + core_len <= l_idx < max_unrolled_idx
            )
            in_generic_alias_band = (core_len == 0 or n_passes <= 1) and (
                reported_blocks is None or l_idx >= len(layer_buckets) // 2 or int(reported_blocks) > 24
            )
            if prev_layer is not None and (in_unrolled_loop_tail or in_generic_alias_band):
                dedup_count += 1
                continue

            # Partial match inside unrolled Inkling 2 loop core (`blk.18..41` vs `blk.6..17`):
            # deduplicate shared linear/expert weights while preserving differing per-pass tensors.
            if (
                core_len > 0
                and n_passes > 1
                and core_start + core_len <= l_idx < max_unrolled_idx
            ):
                base_core = core_start + ((l_idx - core_start) % core_len)
                pass_idx = (l_idx - core_start) // core_len
                base_map = first_core_sigs.get(base_core)
                if base_map is not None:
                    matched_shared = False
                    divergent_shared: list[str] = []
                    pass_only: list[tuple[str, tuple[int, GgufShard, GgufTensorInfo]]] = []
                    for item in items:
                        leaf = _GGUF_BLK_RE.match(item[2].name).group(2)  # type: ignore[union-attr]
                        if leaf_sigs[leaf] == base_map.get(leaf):
                            if not _is_inkling2_per_pass_leaf(leaf):
                                matched_shared = True
                        elif _is_inkling2_per_pass_leaf(leaf):
                            pass_only.append((leaf, item))
                        else:
                            divergent_shared.append(leaf)
                    if matched_shared and divergent_shared:
                        raise ValueError(
                            f"unrolled core block blk.{l_idx} shares weights with blk.{base_core} "
                            f"but diverges on shared tensor(s): {divergent_shared}"
                        )
                    if matched_shared and not divergent_shared:
                        for leaf, item in pass_only:
                            per_pass_deltas.setdefault((base_core, leaf), []).append((pass_idx, item))
                        dedup_count += 1
                        continue

            sig_to_first_layer[sig] = l_idx
            kept_layers[l_idx] = items
            if core_len > 0 and core_start <= l_idx < core_end:
                first_core_sigs[l_idx] = leaf_sigs
    finally:
        for fh in opened.values():
            fh.close()

    kept: list[tuple[int, GgufShard, GgufTensorInfo]] = list(non_layer)
    for l_idx in sorted(kept_layers):
        kept.extend(kept_layers[l_idx])
    return kept, per_pass_deltas, len(kept_layers), dedup_count


def _deduplicate_unrolled_recurrent_layers(
    entries: Sequence[tuple[int, GgufShard, GgufTensorInfo]],
    kv: dict[str, Any],
) -> tuple[list[tuple[int, GgufShard, GgufTensorInfo]], int, int]:
    """Deduplicate unrolled recurrent `blk.N.*` aliases (e.g. `block_count=48` -> `24` physical layers)."""
    kept, _deltas, phys_layers, dedup_count = _deduplicate_unrolled_recurrent_layers_ex(entries, kv)
    return kept, phys_layers, dedup_count


def _build_synthesized_hf_tokenizer_bytes(kv: dict[str, Any]) -> bytes | None:
    tokens = kv.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens:
        return None
    vocab = {str(tok): idx for idx, tok in enumerate(tokens)}
    scores = kv.get("tokenizer.ggml.scores")
    merges = kv.get("tokenizer.ggml.merges")
    tok_model = str(kv.get("tokenizer.ggml.model") or "gpt2")
    added: list[dict[str, Any]] = []
    for id_key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.unk_token_id"):
        if id_key in kv:
            tid = int(kv[id_key])
            if 0 <= tid < len(tokens):
                added.append({"id": tid, "content": str(tokens[tid]), "special": True})
    model_obj: dict[str, Any]
    if tok_model == "llama" and isinstance(scores, list) and len(scores) == len(tokens):
        model_obj = {
            "type": "Unigram",
            "unk_id": int(kv.get("tokenizer.ggml.unk_token_id", 0)),
            "vocab": [[str(tok), float(sc)] for tok, sc in zip(tokens, scores)],
        }
    else:
        model_obj = {
            "type": "BPE",
            "vocab": vocab,
            "merges": [str(m) for m in merges] if isinstance(merges, list) else [],
        }
    payload = {
        "version": "1.0",
        "added_tokens": added,
        "model": model_obj,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _gguf_scalar_kv(val: Any) -> Any:
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _build_synthesized_config_bytes(
    kv: dict[str, Any],
    *,
    physical_layers: int,
    dedup_layers: int,
) -> bytes:
    arch = str(kv.get("general.architecture") or "llama")
    arch_lower = arch.lower()
    cfg: dict[str, Any] = {
        "model_type": arch,
    }
    if arch == "dflash":
        is_dflash2 = any(
            k in kv
            for k in (
                "dflash.conv_kernel_size",
                "dflash.conv_group_size",
                "dflash.conv_groups",
                "dflash.selector_rank",
                "dflash.selector_top_k",
            )
        ) or int(kv.get("dflash.version", 1) or 1) == 2
        is_mimo_dflash = (
            "dflash.target_layers" in kv
            or "dflash.attention.value_scale" in kv
            or "tokenizer.ggml.mask_token_id" in kv
        )
        cfg["architectures"] = ["DFlash2DraftModel"] if is_dflash2 else ["DFlashForCausalLM"]
        cfg["draft_kind"] = "dflash2" if is_dflash2 else "dflash"
        cfg["dflare_format"] = "dflash2" if is_dflash2 else ("dflash" if is_mimo_dflash else "dflare")
        if is_dflash2:
            cfg["dflash_version"] = 2
            cfg["conv_kernel_size"] = int(kv.get("dflash.conv_kernel_size", 2) or 2)
            conv_gs = int(kv.get("dflash.conv_group_size", kv.get("dflash.conv_groups", 16)) or 16)
            cfg["conv_groups"] = conv_gs
            cfg["conv_group_size"] = conv_gs
            cfg["selector_rank"] = int(kv.get("dflash.selector_rank", 256) or 256)
            cfg["selector_top_k"] = int(kv.get("dflash.selector_top_k", 16) or 16)
            cfg["sliding_window"] = int(kv.get("dflash.attention.sliding_window", 2048) or 2048)
        cfg["block_size"] = int(kv.get("dflash.block_size", kv.get("dflare.block_size", 8)) or 8)
        default_mrank = 0 if (is_mimo_dflash or is_dflash2) else 256
        default_mtype = "none" if (is_mimo_dflash or is_dflash2) else "rnn_hidden"
        cfg["markov_rank"] = int(
            kv.get("dflash.markov_rank", kv.get("dflare.markov_rank", default_mrank)) or default_mrank
        )
        cfg["markov_head_type"] = str(
            kv.get("dflash.markov_head_type", kv.get("dflare.markov_head_type", default_mtype))
        )
        tids = kv.get("dflash.target_layer_ids", kv.get("dflare.target_layer_ids"))
        if isinstance(tids, list):
            cfg["target_layer_ids"] = [int(x) for x in tids]
        else:
            # llama.cpp `convert_hf_to_gguf.py` emits 1-indexed `dflash.target_layers` (`[1, 12, 24, 36, 48]`).
            t_layers = kv.get("dflash.target_layers")
            if isinstance(t_layers, list) and t_layers:
                vals = [int(x) for x in t_layers]
                cfg["target_layer_ids"] = [v - 1 for v in vals] if min(vals) >= 1 else vals
        lids = kv.get("dflash.local_layer_ids", kv.get("dflare.local_layer_ids"))
        if isinstance(lids, list):
            cfg["local_layer_ids"] = [int(x) for x in lids]
        if "dflash.attention.value_scale" in kv:
            cfg["attention_value_scale"] = float(kv["dflash.attention.value_scale"])
        if "dflash.attention.causal" in kv:
            cfg["attention_causal"] = bool(kv["dflash.attention.causal"])
        if "dflash.rope.dimension_count" in kv:
            cfg["rope_dim"] = int(kv["dflash.rope.dimension_count"])
        if "tokenizer.ggml.mask_token_id" in kv:
            cfg["mask_token_id"] = int(kv["tokenizer.ggml.mask_token_id"])
        cfg["add_swa_attention_sink_bias"] = True
    elif arch in ("glm-dsa", "glm_moe_dsa"):
        cfg["model_type"] = "glm_moe_dsa"
        cfg["architectures"] = ["GlmMoeDsaForCausalLM"]
        cfg["first_k_dense_replace"] = int(kv.get(f"{arch}.leading_dense_block_count", 3) or 3)
        cfg["index_n_heads"] = int(kv.get(f"{arch}.attention.indexer.head_count", 32) or 32)
        cfg["index_head_dim"] = int(kv.get(f"{arch}.attention.indexer.key_length", 128) or 128)
        cfg["index_topk"] = int(kv.get(f"{arch}.attention.indexer.top_k", 2048) or 2048)
        cfg["index_topk_freq"] = int(kv.get(f"{arch}.attention.indexer.freq", 4) or 4)
        cfg["index_skip_topk_offset"] = int(kv.get(f"{arch}.attention.indexer.skip_offset", 3))
        cfg["index_share_for_mtp_iteration"] = True
        cfg["rope_interleave"] = True
        cfg["indexer_rope_interleave"] = True
        if f"{arch}.attention.q_lora_rank" in kv:
            cfg["q_lora_rank"] = int(kv[f"{arch}.attention.q_lora_rank"])
        if f"{arch}.attention.kv_lora_rank" in kv:
            cfg["kv_lora_rank"] = int(kv[f"{arch}.attention.kv_lora_rank"])
        cfg["qk_rope_head_dim"] = int(kv.get(f"{arch}.rope.dimension_count", 64) or 64)
        mla_k = int(kv.get(f"{arch}.attention.key_length_mla", 256) or 256)
        cfg["qk_nope_head_dim"] = max(0, mla_k - cfg["qk_rope_head_dim"])
        cfg["v_head_dim"] = int(kv.get(f"{arch}.attention.value_length_mla", 256) or 256)
        cfg["moe_intermediate_size"] = int(kv.get(f"{arch}.expert_feed_forward_length", 2048) or 2048)
        cfg["n_shared_experts"] = int(kv.get(f"{arch}.expert_shared_count", 1) or 1)
        cfg["routed_scaling_factor"] = float(kv.get(f"{arch}.expert_weights_scale", 2.5) or 2.5)
        cfg["scoring_func"] = "sigmoid"
        cfg["norm_topk_prob"] = bool(kv.get(f"{arch}.expert_weights_norm", True))
    elif arch in ("glm5-next", "glm5_next"):
        cfg["model_type"] = "glm5_next"
        cfg["architectures"] = ["Glm5NextForConditionalGeneration"]
        cfg["first_k_dense_replace"] = int(kv.get(f"{arch}.leading_dense_block_count", 3) or 3)
        cfg["mhc"] = True
        cfg["hc_mult"] = int(kv.get(f"{arch}.hyper_connection.count", 4) or 4)
        hc_iters = int(kv.get(f"{arch}.hyper_connection.sinkhorn_iterations", 20) or 20)
        cfg["sinkhorn_iters"] = hc_iters
        cfg["hc_sinkhorn_iters"] = hc_iters
        cfg["hc_eps"] = float(kv.get(f"{arch}.hyper_connection.epsilon", 1e-6) or 1e-6)
        cfg["linear_attn_kind"] = "kda"
        cfg["linear_num_heads"] = int(kv.get(f"{arch}.attention.head_count", 64) or 64)
        cfg["linear_head_dim"] = int(kv.get(f"{arch}.kda.head_dim", 128) or 128)
        cfg["linear_lora_rank"] = int(kv.get(f"{arch}.kda.head_dim", 128) or 128)
        cfg["linear_conv_kernel_dim"] = int(kv.get(f"{arch}.ssm.conv_kernel", 4) or 4)
        cfg["linear_lower_bound"] = float(kv.get(f"{arch}.kda.gate_lower_bound", -5.0) or -5.0)
        cfg["q_lora_rank"] = int(kv.get(f"{arch}.attention.q_lora_rank", 1536) or 1536)
        cfg["kv_lora_rank"] = int(kv.get(f"{arch}.attention.kv_lora_rank", 512) or 512)
        cfg["qk_rope_head_dim"] = int(kv.get(f"{arch}.rope.dimension_count", 0) or 0)
        cfg["qk_nope_head_dim"] = int(kv.get(f"{arch}.attention.key_length_mla", 256) or 256)
        cfg["v_head_dim"] = int(kv.get(f"{arch}.attention.value_length_mla", 256) or 256)
        cfg["index_n_heads"] = int(kv.get(f"{arch}.attention.indexer.head_count", 32) or 32)
        cfg["index_head_dim"] = int(kv.get(f"{arch}.attention.indexer.key_length", 128) or 128)
        cfg["index_topk"] = int(kv.get(f"{arch}.attention.indexer.top_k", 2048) or 2048)
        cfg["index_kpool"] = int(kv.get(f"{arch}.attention.indexer.kpool", 4) or 4)
        cfg["index_kpool_always_select_tail"] = True
        cfg["moe_intermediate_size"] = int(kv.get(f"{arch}.expert_feed_forward_length", 2048) or 2048)
        cfg["n_shared_experts"] = int(kv.get(f"{arch}.expert_shared_count", 1) or 1)
        cfg["routed_scaling_factor"] = float(kv.get(f"{arch}.expert_weights_scale", 2.5) or 2.5)
        cfg["swiglu_limit"] = 10.0
        cfg["scoring_func"] = "sigmoid"
        cfg["norm_topk_prob"] = bool(kv.get(f"{arch}.expert_weights_norm", True))
        h_kv = kv.get(f"{arch}.attention.head_count_kv")
        if isinstance(h_kv, list) and h_kv:
            cfg["layer_types"] = ["linear_attention" if int(x) == 0 else "mla_attention" for x in h_kv]
    elif arch in ("mimo2", "mimo_v2"):
        cfg["model_type"] = "mimo_v2"
        cfg["architectures"] = ["MiMoV2ForCausalLM"]
        if f"{arch}.attention.value_length" in kv:
            cfg["v_head_dim"] = int(kv[f"{arch}.attention.value_length"])
            cfg["swa_v_head_dim"] = cfg["v_head_dim"]
        if f"{arch}.attention.key_length" in kv:
            cfg["swa_head_dim"] = int(kv[f"{arch}.attention.key_length"])
        if f"{arch}.attention.value_scale" in kv:
            cfg["attention_value_scale"] = float(kv[f"{arch}.attention.value_scale"])
        if f"{arch}.rope.freq_base_swa" in kv:
            cfg["swa_rope_theta"] = float(kv[f"{arch}.rope.freq_base_swa"])
        if f"{arch}.rope.dimension_count" in kv:
            cfg["rope_dim"] = int(kv[f"{arch}.rope.dimension_count"])
        h_kv = kv.get(f"{arch}.attention.head_count_kv")
        swa_pat_m = kv.get(f"{arch}.attention.sliding_window_pattern")
        if isinstance(swa_pat_m, list) and swa_pat_m:
            cfg["hybrid_layer_pattern"] = [int(x) for x in swa_pat_m]
            cfg["layer_types"] = ["full_attention" if int(x) == 0 else "sliding_attention" for x in swa_pat_m]
            if isinstance(h_kv, list) and len(h_kv) == len(swa_pat_m):
                glob_kvh = next((int(k_v) for k_v, fl in zip(h_kv, swa_pat_m) if int(fl) == 0), int(h_kv[0]))
                swa_kvh = next((int(k_v) for k_v, fl in zip(h_kv, swa_pat_m) if int(fl) != 0), int(h_kv[-1]))
                cfg["num_key_value_heads"] = glob_kvh
                cfg["swa_num_key_value_heads"] = swa_kvh
        cfg["add_swa_attention_sink_bias"] = True
        cfg["scoring_func"] = "sigmoid"
        cfg["norm_topk_prob"] = True
    elif arch in ("qwen35", "qwen3_5", "qwen35moe"):
        cfg["model_type"] = "qwen3_5"
        cfg["architectures"] = ["Qwen3_5ForCausalLM"]
        cfg["rmsnorm_unit_offset"] = True
        cfg["attn_output_gate"] = True
    elif arch == "clip":
        cfg["architectures"] = ["ClipVisionModel"]
    else:
        cfg["architectures"] = [f"{arch[:1].upper()}{arch[1:]}ForCausalLM"]

    scalar_map_fields = (
        (f"{arch}.embedding_length", "hidden_size"),
        (f"{arch}.feed_forward_length", "intermediate_size"),
        (f"{arch}.attention.head_count", "num_attention_heads"),
        (f"{arch}.attention.head_count_kv", "num_key_value_heads"),
        (f"{arch}.attention.key_length", "head_dim"),
        (f"{arch}.attention.layer_norm_rms_epsilon", "rms_norm_eps"),
        (f"{arch}.attention.sliding_window", "sliding_window_size"),
        (f"{arch}.context_length", "max_position_embeddings"),
        (f"{arch}.rope.freq_base", "rope_theta"),
        (f"{arch}.expert_count", "num_local_experts"),
        (f"{arch}.expert_used_count", "num_experts_per_tok"),
        (f"{arch}.block_count", "num_hidden_layers"),
        (f"{arch}.vocab_size", "vocab_size"),
        (f"{arch}.vocab_size_unpadded", "unpadded_vocab_size"),
        (f"{arch}.attention.relative_length", "d_rel"),
        (f"{arch}.attention.relative_extent", "rel_extent"),
        (f"{arch}.shortconv.l_cache", "sconv_kernel_size"),
        (f"{arch}.attn_res.block_size", "n_layers_per_attn_res_block"),
        (f"{arch}.attention.log_scale_floor", "log_scaling_n_floor"),
        (f"{arch}.attention.log_scale_alpha", "log_scaling_alpha"),
        (f"{arch}.attention.qk_clip", "qk_l2_clip_threshold"),
        (f"{arch}.attention.relative_clip", "rel_l2_clip_threshold"),
        (f"{arch}.activation.situ_beta", "mlp_gate_softcap"),
        (f"{arch}.activation.situ_linear_beta", "mlp_up_softcap"),
        (f"{arch}.final_logit_softcapping", "final_logit_softcapping"),
        (f"{arch}.expert_weights_scale", "route_scale"),
    )
    for g_k, h_k in scalar_map_fields:
        if g_k in kv:
            sv = _gguf_scalar_kv(kv[g_k])
            if sv is not None:
                cfg[h_k] = sv
    if "num_local_experts" in cfg and "n_routed_experts" not in cfg:
        cfg["n_routed_experts"] = cfg["num_local_experts"]

    swa_pat = kv.get(f"{arch}.attention.sliding_window_pattern")
    if isinstance(swa_pat, list) and swa_pat:
        cfg["local_layer_ids"] = [idx for idx, flag in enumerate(swa_pat) if int(flag) != 0]

    if physical_layers > 0:
        cfg["physical_num_hidden_layers"] = physical_layers
        if dedup_layers > 0:
            cfg["deduplicated_unrolled_layers"] = dedup_layers
    reported_blocks = int(kv.get(f"{arch}.block_count", 0) or 0)
    eff_layers = physical_layers if physical_layers > 0 else reported_blocks
    has_loop_kv = any(
        k in kv
        for k in (
            f"{arch}.loop.start",
            f"{arch}.loop.core_start",
            f"{arch}.loop.count",
            f"{arch}.loop.num_passes",
        )
    )
    if arch_lower in ("inkling2", "inkling2_loop", "inkling2_loop_text") and (
        (physical_layers > 0 and reported_blocks > physical_layers) or has_loop_kv or arch_lower == "inkling2_loop"
    ):
        cfg["model_type"] = "inkling2_loop"
        if eff_layers > 0:
            cfg["num_hidden_layers"] = eff_layers
        c_start = int(kv.get(f"{arch}.loop.start", kv.get(f"{arch}.loop.core_start", 6)) or 6)
        c_end = int(kv.get(f"{arch}.loop.end", kv.get(f"{arch}.loop.core_end", 18)) or 18)
        n_passes = int(kv.get(f"{arch}.loop.count", kv.get(f"{arch}.loop.num_passes", 3)) or 3)
        g_rank = int(kv.get(f"{arch}.loop.gate_rank", 64) or 64)
        is_var = bool(kv.get(f"{arch}.loop.varying", False))
        eff_steps = eff_layers + max(0, c_end - c_start) * max(0, n_passes - 1) if eff_layers > 0 else reported_blocks
        cfg["effective_num_steps"] = max(eff_steps, reported_blocks)
        cfg["loop_config"] = {
            "core_start": c_start,
            "core_end": c_end,
            "num_passes": n_passes,
            "gate_rank": g_rank,
            "is_varloop": is_var,
        }
        cfg["loop_schedule"] = {
            "core_start": c_start,
            "core_end": c_end,
            "num_passes": n_passes,
            "gate_rank": g_rank,
            "effective_num_steps": cfg["effective_num_steps"],
            "controllers": ["attn", "k", "v", "mlp"],
        }
        cfg["operator_axes"] = {
            "attention": "rel_gqa_swa_sconv",
            "ffn": "moe_softcap",
            "kv_layout": "hybrid_swa_1024",
            "sparse_memory": "none",
            "residual": "attn_res_6",
            "loop_controllers": "sme2_panel64" if g_rank > 0 else "row_major",
        }
        cfg.setdefault("use_gate_bias", True)
        cfg.setdefault("attn_res_normalize_values", True)
    tokens = kv.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list) and tokens:
        if "vocab_size" not in cfg:
            cfg["vocab_size"] = len(tokens)
        elif arch_lower in ("inkling2", "inkling2_loop") and "unpadded_vocab_size" not in cfg:
            cfg["unpadded_vocab_size"] = len(tokens)
    for g_k, h_k in (
        ("tokenizer.ggml.bos_token_id", "bos_token_id"),
        ("tokenizer.ggml.eos_token_id", "eos_token_id"),
    ):
        if g_k in kv and kv[g_k] is not None:
            cfg[h_k] = int(kv[g_k])
    return json.dumps(cfg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _extract_hf_vocab_size_and_specials(tokenizer_json_bytes: bytes) -> tuple[int, dict[str, int]]:
    obj = json.loads(tokenizer_json_bytes.decode("utf-8"))
    vocab_obj = obj.get("model", {}).get("vocab") if isinstance(obj, dict) else None
    specials: dict[str, int] = {}
    max_id = -1
    if isinstance(vocab_obj, dict):
        for tok, tid in vocab_obj.items():
            if isinstance(tid, int) and tid > max_id:
                max_id = tid
        base_len = len(vocab_obj)
    elif isinstance(vocab_obj, list):
        base_len = len(vocab_obj)
        max_id = base_len - 1
    else:
        raise ValueError("external tokenizer.json missing valid model.vocab")
    added = obj.get("added_tokens") or []
    added_extra = 0
    if isinstance(added, list):
        for item in added:
            if isinstance(item, dict) and isinstance(item.get("id"), int) and isinstance(item.get("content"), str):
                tid = int(item["id"])
                specials[item["content"]] = tid
                if tid > max_id:
                    max_id = tid
                if isinstance(vocab_obj, dict) and item["content"] not in vocab_obj:
                    added_extra += 1
    return max(base_len + added_extra, max_id + 1), specials


def _cross_check_hf_tokenizer_with_gguf(tokenizer_json_bytes: bytes, kv: dict[str, Any]) -> None:
    ggml_tokens = kv.get("tokenizer.ggml.tokens")
    if not isinstance(ggml_tokens, list) or not ggml_tokens:
        return
    hf_vocab_size, hf_specials = _extract_hf_vocab_size_and_specials(tokenizer_json_bytes)
    if hf_vocab_size != len(ggml_tokens):
        raise ValueError(
            f"external HF tokenizer.json vocab size ({hf_vocab_size}) disagrees with "
            f"GGUF tokenizer.ggml.tokens ({len(ggml_tokens)})"
        )
    for id_key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
        if id_key in kv:
            tid = int(kv[id_key])
            if 0 <= tid < len(ggml_tokens):
                expected_tok = str(ggml_tokens[tid])
                if expected_tok in hf_specials and hf_specials[expected_tok] != tid:
                    raise ValueError(
                        f"external HF tokenizer.json token {expected_tok!r} id={hf_specials[expected_tok]} "
                        f"conflicts with GGUF {id_key}={tid}"
                    )


def _unsplit_nvfp4_row(row_raw: bytes | memoryview, k: int) -> tuple[bytes, bytes]:
    """Losslessly convert one GGUF `NVFP4` row (`36B/64` split-half nibbles) to planar `nvfp4_g16`."""
    blocks = k // 64
    data_out = bytearray(k // 2)
    scale_out = bytearray(k // 16)
    mv = memoryview(row_raw)
    d_pos = 0
    s_pos = 0
    for b in range(blocks):
        blk = mv[b * 36 : (b + 1) * 36]
        scale_out[s_pos : s_pos + 4] = blk[0:4]
        s_pos += 4
        qs = blk[4:36]
        for sb in range(4):
            qsub = qs[sb * 8 : (sb + 1) * 8]
            nib = [qsub[j] & 0x0F for j in range(8)] + [qsub[j] >> 4 for j in range(8)]
            for i in range(8):
                data_out[d_pos + i] = nib[2 * i] | (nib[2 * i + 1] << 4)
            d_pos += 8
    return bytes(data_out), bytes(scale_out)


@dataclass(frozen=True)
class _PlannedGgufEntry:
    name: str
    kind: TensorKind
    dtype: DType
    quant_bits: int
    group_size: int
    layer: int
    expert: int
    shape: tuple[int, ...]
    shard: GgufShard | None
    src_offset: int
    src_bytes: int
    src_ggml_type: int
    data_convert: str | None  # None, "f16_to_bf16", "bf16_to_f16", "f32_to_bf16", "f32_to_f16", "gguf_nvfp4", "sme2_*"
    scale_shard: GgufShard | None
    scale_info: GgufTensorInfo | None
    fp8_mode: str | None  # None, "fp8_row", "fp8_tensor"
    data_bytes: int
    params_offset: int
    params_bytes: int
    blob_bytes: int
    scale_row_offset: int = 0
    scale_row_count: int = 0
    layout: TensorLayout = TensorLayout.ROW_MAJOR
    quant_param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS
    pair_interleave_rows: bool = False
    second_shard: GgufShard | None = None
    second_src_offset: int = 0
    expert_scale: float | None = None
    second_expert_scale: float | None = None
    synthetic_bytes: bytes | None = None
    row_gather_runs: tuple[tuple[int, int], ...] | None = None


def _scale_gguf_blocks_in_place(buf: bytearray, dtype: DType, bb: int, scale_f: float, name: str) -> None:
    """Multiply every block's fp16 `d` (and optional `dmin`) in-place by `scale_f`."""
    if not math.isfinite(scale_f):
        raise ValueError(f"non-finite expert scale {scale_f} for {name!r}")
    if scale_f == 1.0:
        return
    offsets = _GGUF_BLOCK_SCALE_OFFSETS.get(dtype)
    if offsets is None:
        raise ValueError(f"cannot apply expert .scale to unsupported GGUF dtype {dtype.name} on {name!r}")
    for off in offsets:
        b0 = buf[off::bb]
        b1 = buf[off + 1 :: bb]
        cache: dict[tuple[int, int], bytes] = {}
        out_chunks: list[bytes] = []
        append = out_chunks.append
        for pair in zip(b0, b1):
            packed = cache.get(pair)
            if packed is None:
                val = struct.unpack("<e", bytes(pair))[0] * scale_f
                if val > 65504.0:
                    val = 65504.0
                elif val < -65504.0:
                    val = -65504.0
                packed = struct.pack("<e", val)
                cache[pair] = packed
            append(packed)
        merged = b"".join(out_chunks)
        buf[off::bb] = merged[0::2]
        buf[off + 1 :: bb] = merged[1::2]


def _interleave_pair_rows(raw_concat: bytes | bytearray | memoryview, n_rows: int, row_bytes: int) -> bytes:
    """Reorder `[gate[0..H-1], up[0..H-1]]` row blocks into `pair_interleaved` `[g0, u0, g1, u1, ...]`."""
    if n_rows % 2 != 0:
        raise ValueError(f"pair_interleaved w13 requires even row count, got {n_rows}")
    half = n_rows // 2
    mv = memoryview(raw_concat)
    out = bytearray(n_rows * row_bytes)
    out_mv = memoryview(out)
    for r in range(half):
        src_g = r * row_bytes
        src_u = (half + r) * row_bytes
        dst_p = (2 * r) * row_bytes
        out_mv[dst_p : dst_p + row_bytes] = mv[src_g : src_g + row_bytes]
        out_mv[dst_p + row_bytes : dst_p + 2 * row_bytes] = mv[src_u : src_u + row_bytes]
    return bytes(out)


def _interleave_split_rows(
    gate_buf: bytes | bytearray | memoryview,
    up_buf: bytes | bytearray | memoryview,
    half_rows: int,
    row_bytes: int,
) -> bytes:
    """Interleave separate `gate[0..H-1]` and `up[0..H-1]` row buffers into `[g0, u0, g1, u1, ...]`."""
    g_mv = memoryview(gate_buf)
    u_mv = memoryview(up_buf)
    out = bytearray(2 * half_rows * row_bytes)
    out_mv = memoryview(out)
    for r in range(half_rows):
        src = r * row_bytes
        dst_p = (2 * r) * row_bytes
        out_mv[dst_p : dst_p + row_bytes] = g_mv[src : src + row_bytes]
        out_mv[dst_p + row_bytes : dst_p + 2 * row_bytes] = u_mv[src : src + row_bytes]
    return bytes(out)


_NVFP4_MASK_CACHE: dict[int, tuple[int, int]] = {}


def _unsplit_nvfp4_batch(raw_buf: bytes | bytearray | memoryview) -> tuple[bytes, bytes]:
    """Vectorized C-speed conversion of N packed GGUF NVFP4 blocks (`N * 36B`) to planar `nvfp4_g16`."""
    n = len(raw_buf) // 36
    if n == 0:
        return b"", b""
    masks = _NVFP4_MASK_CACHE.get(n)
    if masks is None:
        all_ones = ((1 << (8 * n)) - 1) // 255
        masks = (all_ones * 0x0F, all_ones * 0xF0)
        if len(_NVFP4_MASK_CACHE) < 64:
            _NVFP4_MASK_CACHE[n] = masks
    mask0f, maskf0 = masks
    mv = memoryview(raw_buf)
    scales = bytearray(n * 4)
    for j in range(4):
        scales[j::4] = mv[j::36]
    data_out = bytearray(n * 32)
    for sb in range(4):
        sbase = 4 + sb * 8
        dbase = sb * 8
        for i in range(4):
            i0 = int.from_bytes(mv[sbase + 2 * i :: 36], "little")
            i1 = int.from_bytes(mv[sbase + 2 * i + 1 :: 36], "little")
            data_out[dbase + i :: 32] = ((i0 & mask0f) | ((i1 & mask0f) << 4)).to_bytes(n, "little")
            data_out[dbase + 4 + i :: 32] = (((i0 & maskf0) >> 4) | (i1 & maskf0)).to_bytes(n, "little")
    return bytes(data_out), bytes(scales)


def _decode_gguf_float_array(shard: GgufShard, info: GgufTensorInfo) -> bytes:
    with shard.path.open("rb") as fh:
        fh.seek(shard.data_offset + info.offset)
        raw = fh.read(info.expected_bytes)
    if info.ggml_type == 0:  # F32
        return raw
    if info.ggml_type == 30:  # BF16 -> F32 lossless
        return bytes(_bf16_to_f32_bytes(raw))
    if info.ggml_type == 1:  # F16 -> F32 exact
        cnt = info.expected_bytes // 2
        return struct.pack(f"<{cnt}f", *struct.unpack(f"<{cnt}e", raw))
    raise ValueError(f"unsupported FP8 companion scale type {info.ggml_type} on {info.name!r}")


def _extract_expert_scales(shard: GgufShard, info: GgufTensorInfo) -> tuple[float, ...]:
    raw = _decode_gguf_float_array(shard, info)
    cnt = len(raw) // 4
    return struct.unpack(f"<{cnt}f", raw)


def _decode_gguf_tensor_f32(shard: GgufShard, info: GgufTensorInfo) -> tuple[float, ...]:
    if info.ggml_type == 0:
        with shard.path.open("rb") as fh:
            fh.seek(shard.data_offset + info.offset)
            raw = fh.read(info.expected_bytes)
        return struct.unpack(f"<{len(raw) // 4}f", raw)
    if info.ggml_type == 30:
        with shard.path.open("rb") as fh:
            fh.seek(shard.data_offset + info.offset)
            raw = fh.read(info.expected_bytes)
        f32_b = _bf16_to_f32_bytes(raw)
        return struct.unpack(f"<{len(f32_b) // 4}f", f32_b)
    if info.ggml_type == 1:
        with shard.path.open("rb") as fh:
            fh.seek(shard.data_offset + info.offset)
            raw = fh.read(info.expected_bytes)
        return tuple(f16_bytes_to_f32_list(raw))
    raise ValueError(f"expected float tensor (F32/F16/BF16) for {info.name!r}, got ggml_type={info.ggml_type}")


def pack_amf_from_gguf(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str | None = None,
    dtype_mode: str = "auto",
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int | None = None,
    heat_profile: pathlib.Path | None = None,
    all_experts_cold: bool = False,
    exclude_tensors: Iterable[str] = (),
    embed_metadata: bool = False,
    config_path: pathlib.Path | None = None,
    tokenizer_path: pathlib.Path | None = None,
    tokenizer_config_path: pathlib.Path | None = None,
    chat_template_path: pathlib.Path | None = None,
    strip_vision: bool = False,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
    group_size: int = 64,
    strict_vocab_check: bool = True,
    deduplicate_recurrent: bool = True,
    sme2_pack: str | None = None,
    draft_config_output: pathlib.Path | None = None,
    engram_output: pathlib.Path | None = None,
    engram_shards: int = 1,
    meta_config_override: bytes | None = None,
    stats_out: dict[str, int] | None = None,
    mimo_mtp_mode: bool = False,
) -> list[PackedTensorMeta]:
    """Stream-convert a GGUF v2/v3 file (or split shards) into a 16 KiB-aligned `.amf` v0."""
    in_path = pathlib.Path(input_path)
    out_path = pathlib.Path(output_path)
    if engram_shards < 1:
        raise ValueError(f"engram_shards must be >= 1, got {engram_shards}")
    shards = resolve_gguf_shards(in_path)
    kv = dict(shards[0].kv)
    arch = str(kv.get("general.architecture") or "llama").lower()
    is_clip = arch == "clip"

    eff_model_id = model_id or str(kv.get("general.name") or out_path.stem)[:63]
    eff_model_id = eff_model_id[:63]
    if not eff_model_id:
        raise ValueError("model_id must be non-empty")

    flat_items: list[tuple[int, GgufShard, GgufTensorInfo]] = []
    for s_idx, sh in enumerate(shards):
        for t in sh.tensors:
            flat_items.append((s_idx, sh, t))

    physical_layers = 0
    dedup_layers = 0
    layer_remap: dict[int, int] = {}
    pass_deltas: dict[tuple[int, str], list[tuple[int, tuple[int, GgufShard, GgufTensorInfo]]]] = {}
    if deduplicate_recurrent and not is_clip:
        flat_items, pass_deltas, physical_layers, dedup_layers = _deduplicate_unrolled_recurrent_layers_ex(
            flat_items, kv
        )
        kept_ids = sorted(
            {
                int(m.group(1))
                for _, _, t in flat_items
                if (m := _GGUF_BLK_RE.match(t.name)) is not None
            }
        )
        layer_remap = {old_l: new_l for new_l, old_l in enumerate(kept_ids)}
    if mimo_mtp_mode and arch.lower() in ("mimo2", "mimo_v2", "glm-dsa", "glm_moe_dsa", "glm5-next", "glm5_next"):
        mtp_blk_ids = sorted(
            {
                int(m.group(1))
                for _, _, t in flat_items
                if (m := _GGUF_BLK_RE.match(t.name)) is not None
            }
        )
        blk_cnt = int(kv.get(f"{arch}.block_count", 0) or 0)
        default_nextn = 1 if arch.lower().startswith("glm") else 3
        nextn_n = int(kv.get(f"{arch}.nextn_predict_layers", default_nextn) or default_nextn)
        if blk_cnt > nextn_n > 0 and mtp_blk_ids and max(mtp_blk_ids) >= blk_cnt - nextn_n and min(mtp_blk_ids) < blk_cnt - nextn_n:
            mtp_start = blk_cnt - nextn_n
            flat_items = [
                item
                for item in flat_items
                if (m := _GGUF_BLK_RE.match(item[2].name)) is not None and int(m.group(1)) >= mtp_start
            ]
            mtp_blk_ids = [idx for idx in mtp_blk_ids if idx >= mtp_start]
        if mtp_blk_ids:
            layer_remap = {old_l: new_l for new_l, old_l in enumerate(mtp_blk_ids)}
            physical_layers = len(mtp_blk_ids)
    elif not mimo_mtp_mode and arch.lower() in ("glm-dsa", "glm_moe_dsa", "glm5-next", "glm5_next"):
        blk_cnt = int(kv.get(f"{arch}.block_count", 0) or 0)
        nextn_n = int(kv.get(f"{arch}.nextn_predict_layers", 0) or 0)
        if blk_cnt > nextn_n > 0:
            backbone_lim = blk_cnt - nextn_n
            flat_items = [
                item
                for item in flat_items
                if not ((m := _GGUF_BLK_RE.match(item[2].name)) is not None and int(m.group(1)) >= backbone_lim)
            ]

    glm5_linear_layers: set[int] | None = None
    if arch.lower() in ("glm5-next", "glm5_next"):
        glm5_linear_layers = set()
        h_kv = kv.get(f"{arch}.attention.head_count_kv")
        if isinstance(h_kv, list):
            for idx_k, v_k in enumerate(h_kv):
                if int(v_k) == 0:
                    glm5_linear_layers.add(idx_k)
        for _, _, t in flat_items:
            m_kda = _GGUF_BLK_RE.match(t.name)
            if m_kda is not None and m_kda.group(2).startswith(("ssm_", "linear_attn.")):
                glm5_linear_layers.add(int(m_kda.group(1)))

    if stats_out is not None:
        stats_out["physical_layers"] = physical_layers
        stats_out["deduplicated_layers"] = dedup_layers
        stats_out["gguf_arch"] = hash(arch) & 0x7FFFFFFF

    is_inkling2_arch = arch in ("inkling2", "inkling2_loop", "inkling2_text", "inkling2_loop_text")
    use_inkling2_runtime_names = (
        arch in ("inkling2_loop", "inkling2_loop_text")
        or (
            is_inkling2_arch
            and any(
                any(
                    tok in t.name
                    for tok in (
                        "attn_q_clip",
                        "attn_post_conv",
                        "attn_rel_clip",
                        "attn_rel_bias",
                        "attn_res_query",
                        "ffn_res_query",
                        "loop_msg_query",
                        "loop_q_ctrl",
                        "loop_ffn_ctrl",
                        "attn_k_conv",
                        "ffn_gate_up_exps",
                        "ffn_gate_exps",
                        "token_embd_norm",
                    )
                )
                for _, _, t in flat_items
            )
        )
    )
    blk_res_layers = {
        layer_remap.get(int(m.group(1)), int(m.group(1)))
        for _, _, t in flat_items
        if (m := _GGUF_BLK_RE.match(t.name)) is not None and m.group(2) == "attn_res_query.weight"
    }
    attn_res_shifted_by_1 = use_inkling2_runtime_names and (0 not in blk_res_layers) and (1 in blk_res_layers)

    if sme2_pack is None:
        eff_sme2_pack = (
            "controllers"
            if any(
                is_inkling2_controller_tensor(
                    normalize_gguf_tensor_name(
                        t.name,
                        is_clip=is_clip,
                        layer_remap=layer_remap,
                        inkling2_runtime_names=use_inkling2_runtime_names,
                        attn_res_shifted_by_1=attn_res_shifted_by_1,
                    )
                )
                for _, _, t in flat_items
            )
            else "none"
        )
    else:
        if sme2_pack not in SME2_PACK_MODES:
            raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
        eff_sme2_pack = sme2_pack

    raw_by_name = {t.name: (s_idx, sh, t) for s_idx, sh, t in flat_items}
    excluded = {tok.strip() for tok in exclude_tensors if tok.strip()}
    cold_set = set(cold_experts)
    if not cold_set and not all_experts_cold:
        if heat_profile is not None:
            cold_set = load_cold_experts_from_heat_profile(
                heat_profile,
                cold_experts_per_layer=cold_experts_per_layer,
            )
        elif cold_experts_per_layer is not None and cold_experts_per_layer > 0:
            n_l = physical_layers if physical_layers > 0 else int(kv.get(f"{arch}.block_count", 0) or 0)
            n_e = int(kv.get(f"{arch}.expert_count", 0) or 0)
            if n_e <= 0:
                for _, _, t in flat_items:
                    if len(t.amf_shape) == 3 and "_exps" in t.name:
                        n_e = max(n_e, t.amf_shape[0])
            take = min(cold_experts_per_layer, n_e)
            if n_l > 0 and take > 0:
                cold_set = {(l, e) for l in range(n_l) for e in range(n_e - take, n_e)}

    consumed_names: set[str] = set()
    planned: list[_PlannedGgufEntry] = []
    planned_vision: list[_PlannedGgufEntry] = []
    planned_engram: list[_PlannedGgufEntry] = []

    def _find_scalar_or_vec_scale(t_name: str, *, exps_mode: bool = False) -> tuple[int, GgufShard, GgufTensorInfo] | None:
        base = t_name[:-7] if t_name.endswith(".weight") else t_name
        cands = (
            (f"{base}.scale", f"{t_name}.scale", f"{base}.weight_scale")
            if exps_mode
            else (f"{t_name}.scale", f"{t_name}_scale", f"{base}.scale", f"{base}.weight_scale", f"{base}_scale")
        )
        for cand in cands:
            hit = raw_by_name.get(cand)
            if hit is not None and hit[2].ggml_type in (0, 1, 30):
                return hit
        return None

    # Pre-scan F8_E4M3 companion scales and expert `.scale` arrays cleanly before planning.
    fp8_pairs: dict[str, tuple[int, GgufShard, GgufTensorInfo] | None] = {}
    exps_scale_vals: dict[str, tuple[float, ...]] = {}
    for _, _, t in flat_items:
        if t.ggml_type == 43:
            sc_hit = _find_scalar_or_vec_scale(t.name)
            fp8_pairs[t.name] = sc_hit
            if sc_hit is not None:
                consumed_names.add(sc_hit[2].name)
        elif re.match(r"^blk\.\d+\.(ffn_gate_up|ffn_gate|ffn_up|ffn_down)_exps(?:\.weight)?$", t.name):
            sc_hit = _find_scalar_or_vec_scale(t.name, exps_mode=True)
            if sc_hit is not None:
                consumed_names.add(sc_hit[2].name)
                exps_scale_vals[t.name] = _extract_expert_scales(sc_hit[1], sc_hit[2])

    # Pre-scan Inkling 2 `blk.N.loop_msg_query.weight` -> assemble `middle_cycle_cross_pass_{,kv_}queries`.
    if use_inkling2_runtime_names:
        loop_msg_entries: list[tuple[int, GgufShard, GgufTensorInfo]] = []
        for _, sh, t in flat_items:
            mm = _GGUF_BLK_RE.match(t.name)
            if mm is not None and mm.group(2) == "loop_msg_query.weight":
                raw_li = int(mm.group(1))
                phys_li = layer_remap.get(raw_li, raw_li)
                loop_msg_entries.append((phys_li, sh, t))
        if (
            loop_msg_entries
            and "middle_cycle_cross_pass_queries" not in raw_by_name
            and "middle_cycle_cross_pass_kv_queries" not in raw_by_name
        ):
            loop_msg_entries.sort(key=lambda x: x[0])
            cpq_floats: list[float] = []
            cpkv_floats: list[float] = []
            h_dim = loop_msg_entries[0][2].amf_shape[-1]
            for _, sh, t in loop_msg_entries:
                if len(t.amf_shape) != 2 or t.amf_shape[0] != 4 or t.amf_shape[1] != h_dim:
                    raise ValueError(f"expected (4, {h_dim}) for {t.name!r}, got {t.amf_shape}")
                vals = _decode_gguf_tensor_f32(sh, t)
                cpq_floats.extend(vals[0:h_dim])
                cpq_floats.extend(vals[3 * h_dim : 4 * h_dim])
                cpkv_floats.extend(vals[h_dim : 2 * h_dim])
                cpkv_floats.extend(vals[2 * h_dim : 3 * h_dim])
                consumed_names.add(t.name)
            k_rows = 2 * len(loop_msg_entries)
            for synth_name, flt_list in (
                ("middle_cycle_cross_pass_queries", cpq_floats),
                ("middle_cycle_cross_pass_kv_queries", cpkv_floats),
            ):
                if synth_name in excluded:
                    continue
                synth_b = struct.pack(f"<{len(flt_list)}f", *flt_list)
                s_kind, s_layer, s_exp = _classify_tensor(synth_name, cold_set, all_experts_cold)
                planned.append(
                    _PlannedGgufEntry(
                        name=synth_name,
                        kind=s_kind,
                        dtype=DType.F32,
                        quant_bits=0,
                        group_size=0,
                        layer=s_layer,
                        expert=s_exp,
                        shape=(k_rows, h_dim),
                        shard=None,
                        src_offset=0,
                        src_bytes=len(synth_b),
                        src_ggml_type=0,
                        data_convert=None,
                        scale_shard=None,
                        scale_info=None,
                        fp8_mode=None,
                        data_bytes=len(synth_b),
                        params_offset=0,
                        params_bytes=0,
                        blob_bytes=_align_up(max(len(synth_b), 1), APPLE_PAGE_SIZE),
                        synthetic_bytes=synth_b,
                    )
                )

    # Pre-scan split `blk.N.ffn_gate_exps.weight` + `blk.N.ffn_up_exps.weight` -> fused `w13.weight` if matching.
    fused_split_gate_ups: dict[str, tuple[int, GgufShard, GgufTensorInfo]] = {}
    if use_inkling2_runtime_names:
        for t_name, (_s_i, _g_sh, g_t) in raw_by_name.items():
            gm = re.match(r"^blk\.(\d+)\.ffn_gate_exps(?:\.(weight))?$", t_name)
            if gm is None or len(g_t.amf_shape) != 3:
                continue
            up_cand = f"blk.{gm.group(1)}.ffn_up_exps" + (".weight" if gm.group(2) else "")
            up_hit = raw_by_name.get(up_cand)
            if up_hit is None:
                continue
            up_t = up_hit[2]
            if up_t.ggml_type != g_t.ggml_type or up_t.amf_shape != g_t.amf_shape:
                continue
            if g_t.ggml_type == 40:
                g_sc = exps_scale_vals.get(g_t.name)
                u_sc = exps_scale_vals.get(up_t.name)
                if g_sc != u_sc and (g_sc is not None or u_sc is not None):
                    continue
            fused_split_gate_ups[t_name] = up_hit
            consumed_names.add(up_cand)

    # Synthesize stacked multi-pass `attn_res_query.weight` / `ffn_res_query.weight` or pass deltas if present.
    synthetic_override_items: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    extra_pass_items: list[tuple[str, tuple[int, GgufShard, GgufTensorInfo]]] = []
    if pass_deltas:
        for (base_core, leaf), delta_list in sorted(pass_deltas.items()):
            delta_list_sorted = sorted(delta_list, key=lambda x: x[0])
            base_name = f"blk.{base_core}.{leaf}"
            base_hit = raw_by_name.get(base_name)
            if (
                use_inkling2_runtime_names
                and base_hit is not None
                and leaf in ("attn_res_query.weight", "ffn_res_query.weight")
                and len(base_hit[2].amf_shape) == 1
                and all(len(it[1][2].amf_shape) == 1 and it[1][2].amf_shape == base_hit[2].amf_shape for it in delta_list_sorted)
            ):
                base_vals = _decode_gguf_tensor_f32(base_hit[1], base_hit[2])
                stacked_vals = list(base_vals)
                for _p_i, (_si, d_sh, d_t) in delta_list_sorted:
                    stacked_vals.extend(_decode_gguf_tensor_f32(d_sh, d_t))
                out_nm = normalize_gguf_tensor_name(
                    base_name,
                    is_clip=is_clip,
                    layer_remap=layer_remap,
                    inkling2_runtime_names=use_inkling2_runtime_names,
                    attn_res_shifted_by_1=attn_res_shifted_by_1,
                )
                n_rows_p = 1 + len(delta_list_sorted)
                synthetic_override_items[base_name] = (
                    out_nm,
                    (n_rows_p, base_hit[2].amf_shape[0]),
                    struct.pack(f"<{len(stacked_vals)}f", *stacked_vals),
                )
            else:
                for p_i, item in delta_list_sorted:
                    base_out = normalize_gguf_tensor_name(
                        f"blk.{base_core}.{leaf}",
                        is_clip=is_clip,
                        layer_remap=layer_remap,
                        inkling2_runtime_names=use_inkling2_runtime_names,
                        attn_res_shifted_by_1=attn_res_shifted_by_1,
                        arch=arch,
                        mimo_mtp_mode=mimo_mtp_mode,
                    )
                    extra_pass_items.append((f"{base_out}.pass{p_i}", item))

    def _convert_ink2_f32_needed(out_nm: str, out_k: TensorKind, g_type: int) -> bool:
        if not use_inkling2_runtime_names or g_type != 0:
            return False
        if out_k == TensorKind.NORM or is_inkling2_controller_tensor(out_nm):
            return True
        return out_nm.endswith(".mlp.gate.weight") or out_nm.endswith(".mlp.gate")

    target_f16_dt = DType.F16 if dtype_mode == "f16" else DType.BF16

    all_planning_items: list[tuple[str | None, int, GgufShard, GgufTensorInfo]] = [
        (None, s_idx, sh, t) for s_idx, sh, t in flat_items
    ] + [(ov_name, s_idx, sh, t) for ov_name, (s_idx, sh, t) in extra_pass_items]

    for override_out_name, s_idx, sh, t in all_planning_items:
        if override_out_name is None and (t.name in excluded or t.name in consumed_names):
            continue
        if override_out_name is None and t.name in synthetic_override_items:
            ov_nm, ov_shp, ov_bytes = synthetic_override_items[t.name]
            if ov_nm in excluded:
                continue
            ov_kind, ov_layer, ov_exp = _classify_tensor(ov_nm, cold_set, all_experts_cold)
            planned.append(
                _PlannedGgufEntry(
                    name=ov_nm,
                    kind=ov_kind,
                    dtype=DType.F32,
                    quant_bits=0,
                    group_size=0,
                    layer=ov_layer,
                    expert=ov_exp,
                    shape=ov_shp,
                    shard=None,
                    src_offset=0,
                    src_bytes=len(ov_bytes),
                    src_ggml_type=0,
                    data_convert=None,
                    scale_shard=None,
                    scale_info=None,
                    fp8_mode=None,
                    data_bytes=len(ov_bytes),
                    params_offset=0,
                    params_bytes=0,
                    blob_bytes=_align_up(max(len(ov_bytes), 1), APPLE_PAGE_SIZE),
                    synthetic_bytes=ov_bytes,
                )
            )
            continue

        spec = _GGML_TYPE_INFO[t.ggml_type]
        _gname, bs, bb, base_amf_dt = spec
        assert base_amf_dt is not None

        amf_shape_full = t.amf_shape
        ex_m = (
            re.match(r"^blk\.(\d+)\.(ffn_gate_up|ffn_gate|ffn_up|ffn_down)_exps(?:\.(weight))?$", t.name)
            if override_out_name is None
            else None
        )
        # Each tuple: (out_name, out_kind, layer_i, exp_i, shape, src_off, src_len,
        #              pair_interleave, second_sh, second_off, exp_scale, second_exp_scale)
        slices: list[
            tuple[
                str,
                TensorKind,
                int,
                int,
                tuple[int, ...],
                int,
                int,
                bool,
                GgufShard | None,
                int,
                float | None,
                float | None,
            ]
        ]
        split_extras: dict[str, tuple[int, int, tuple[tuple[int, int], ...] | None]] = {}
        if ex_m is not None and len(amf_shape_full) == 3:
            raw_layer_i = int(ex_m.group(1))
            layer_i = layer_remap.get(raw_layer_i, raw_layer_i)
            raw_proj = ex_m.group(2)
            num_experts, n_r, k_c = amf_shape_full
            exp_slice_bytes = t.expected_bytes // num_experts
            sc_arr = exps_scale_vals.get(t.name)

            if raw_proj == "ffn_gate" and t.name in fused_split_gate_ups:
                _, up_sh, up_t = fused_split_gate_ups[t.name]
                up_exp_bytes = up_t.expected_bytes // num_experts
                up_sc_arr = exps_scale_vals.get(up_t.name)
                if exp_slice_bytes != up_exp_bytes or exp_slice_bytes % n_r != 0:
                    raise ValueError(f"cannot fuse split gate/up experts for blk.{raw_layer_i}: {exp_slice_bytes} vs {up_exp_bytes}")
                slices = []
                for e in range(num_experts):
                    e_name = f"layers.{layer_i}.mlp.experts.{e}.w13.weight"
                    if e_name in excluded:
                        continue
                    e_kind = TensorKind.COLD_EXPERT if (all_experts_cold or (layer_i, e) in cold_set) else TensorKind.HOT_EXPERT
                    slices.append(
                        (
                            e_name,
                            e_kind,
                            layer_i,
                            e,
                            (2 * n_r, k_c),
                            t.offset + e * exp_slice_bytes,
                            exp_slice_bytes + up_exp_bytes,
                            False,
                            up_sh,
                            up_t.offset + e * up_exp_bytes,
                            sc_arr[e] if sc_arr is not None else None,
                            up_sc_arr[e] if up_sc_arr is not None else None,
                        )
                    )
            else:
                proj = {
                    "ffn_gate": "gate_proj",
                    "ffn_up": "up_proj",
                    "ffn_down": "down_proj",
                    "ffn_gate_up": "w13" if use_inkling2_runtime_names else "gate_up_proj",
                }[raw_proj]
                is_pair_interleave = raw_proj == "ffn_gate_up" and use_inkling2_runtime_names
                if is_pair_interleave and (n_r % 2 != 0 or exp_slice_bytes % n_r != 0):
                    raise ValueError(
                        f"cannot pair_interleave {t.name!r} with shape {(n_r, k_c)} and {exp_slice_bytes} B/expert"
                    )
                slices = []
                for e in range(num_experts):
                    e_name = f"layers.{layer_i}.mlp.experts.{e}.{proj}.weight"
                    if e_name in excluded:
                        continue
                    e_kind = TensorKind.COLD_EXPERT if (all_experts_cold or (layer_i, e) in cold_set) else TensorKind.HOT_EXPERT
                    slices.append(
                        (
                            e_name,
                            e_kind,
                            layer_i,
                            e,
                            (n_r, k_c),
                            t.offset + e * exp_slice_bytes,
                            exp_slice_bytes,
                            is_pair_interleave,
                            None,
                            0,
                            sc_arr[e] if sc_arr is not None else None,
                            None,
                        )
                    )
        else:
            norm_name = (
                override_out_name
                if override_out_name is not None
                else normalize_gguf_tensor_name(
                    t.name,
                    is_clip=is_clip,
                    layer_remap=layer_remap,
                    inkling2_runtime_names=use_inkling2_runtime_names,
                    attn_res_shifted_by_1=attn_res_shifted_by_1,
                    arch=arch,
                    mimo_mtp_mode=mimo_mtp_mode,
                    linear_layers=glm5_linear_layers,
                )
            )
            if norm_name in excluded:
                continue
            eff_shape = amf_shape_full
            if (
                arch.lower() in ("qwen35", "qwen3_5", "qwen35moe", "glm5-next", "glm5_next")
                and norm_name.endswith(
                    (
                        ".linear_attn.conv1d.weight",
                        ".linear_attn.q_conv1d.weight",
                        ".linear_attn.k_conv1d.weight",
                        ".linear_attn.v_conv1d.weight",
                    )
                )
                and len(eff_shape) == 3
                and eff_shape[1] == 1
            ):
                eff_shape = (eff_shape[0], eff_shape[2])
            kind, layer_i, exp_i = _classify_tensor(norm_name, cold_set, all_experts_cold)
            slices = [(norm_name, kind, layer_i, exp_i, eff_shape, t.offset, t.expected_bytes, False, None, 0, None, None)]
            eff_arch_l = arch.lower()
            raw_mb = _GGUF_BLK_RE.match(t.name)
            raw_li = int(raw_mb.group(1)) if raw_mb is not None else layer_i
            if eff_arch_l in ("mimo2", "mimo_v2") and norm_name.endswith(".self_attn.qkv_proj.weight") and len(eff_shape) == 2:
                n_r, k_c = eff_shape
                q_heads = int(kv.get(f"{arch}.attention.head_count", 64) or 64)
                kv_raw = kv.get(f"{arch}.attention.head_count_kv", 8)
                if isinstance(kv_raw, Sequence) and not isinstance(kv_raw, (str, bytes)):
                    kv_heads = int(kv_raw[raw_li] if raw_li < len(kv_raw) else kv_raw[min(layer_i, len(kv_raw) - 1)])
                else:
                    kv_heads = int(kv_raw or 8)
                k_hd = int(kv.get(f"{arch}.attention.key_length", 192) or 192)
                v_hd = int(kv.get(f"{arch}.attention.value_length", 128) or 128)
                q_rows, k_rows, v_rows = q_heads * k_hd, kv_heads * k_hd, kv_heads * v_hd
                if q_rows + k_rows + v_rows == n_r and t.expected_bytes % n_r == 0:
                    row_b = t.expected_bytes // n_r
                    pref_qkv = norm_name[: -len("qkv_proj.weight")]
                    slices = []
                    cur_r = 0
                    for sub_proj, sub_rows in (("q_proj.weight", q_rows), ("k_proj.weight", k_rows), ("v_proj.weight", v_rows)):
                        sub_name = _normalize_name(f"{pref_qkv}{sub_proj}")
                        if sub_name not in excluded:
                            sub_kind, sub_li, sub_ei = _classify_tensor(sub_name, cold_set, all_experts_cold)
                            slices.append(
                                (
                                    sub_name,
                                    sub_kind,
                                    sub_li,
                                    sub_ei,
                                    (sub_rows, k_c),
                                    t.offset + cur_r * row_b,
                                    sub_rows * row_b,
                                    False,
                                    None,
                                    0,
                                    None,
                                    None,
                                )
                            )
                            split_extras[sub_name] = (cur_r, sub_rows, None)
                        cur_r += sub_rows
            elif eff_arch_l in ("qwen35", "qwen3_5", "qwen35moe") and norm_name.endswith(".self_attn.qkv_proj.weight") and len(eff_shape) == 2:
                n_r, k_c = eff_shape
                k_heads = int(kv.get(f"{arch}.ssm.group_count", 16) or 16)
                v_heads = int(kv.get(f"{arch}.ssm.time_step_rank", 32) or 32)
                state_sz = int(kv.get(f"{arch}.ssm.state_size", 128) or 128)
                v_total = int(kv.get(f"{arch}.ssm.inner_size", v_heads * state_sz) or (v_heads * state_sz))
                q_rows = k_heads * state_sz
                k_rows = k_heads * state_sz
                if q_rows + k_rows + v_total == n_r and t.expected_bytes % n_r == 0:
                    row_b = t.expected_bytes // n_r
                    slices = []
                    cur_r = 0
                    for sub_proj, sub_rows in (("q_proj.weight", q_rows), ("k_proj.weight", k_rows), ("v_proj.weight", v_total)):
                        sub_name = _normalize_name(f"layers.{layer_i}.linear_attn.{sub_proj}")
                        if sub_name not in excluded:
                            sub_kind, sub_li, sub_ei = _classify_tensor(sub_name, cold_set, all_experts_cold)
                            slices.append(
                                (
                                    sub_name,
                                    sub_kind,
                                    sub_li,
                                    sub_ei,
                                    (sub_rows, k_c),
                                    t.offset + cur_r * row_b,
                                    sub_rows * row_b,
                                    False,
                                    None,
                                    0,
                                    None,
                                    None,
                                )
                            )
                            split_extras[sub_name] = (cur_r, sub_rows, None)
                        cur_r += sub_rows
            elif eff_arch_l in ("qwen35", "qwen3_5", "qwen35moe") and norm_name.endswith(".self_attn.q_proj.weight") and len(eff_shape) == 2:
                n_r, k_c = eff_shape
                n_heads = int(kv.get(f"{arch}.attention.head_count", 0) or 0)
                hd = int(kv.get(f"{arch}.attention.key_length", 0) or 0)
                if n_heads > 0 and hd > 0 and n_r == 2 * n_heads * hd and t.expected_bytes % n_r == 0:
                    row_b = t.expected_bytes // n_r
                    half_r = n_heads * hd
                    q_runs = tuple((h * 2 * hd, hd) for h in range(n_heads))
                    g_runs = tuple((h * 2 * hd + hd, hd) for h in range(n_heads))
                    pref_sa = norm_name[: -len("q_proj.weight")]
                    slices = []
                    for sub_proj, runs in (("q_proj.weight", q_runs), ("q_gate_proj.weight", g_runs)):
                        sub_name = _normalize_name(f"{pref_sa}{sub_proj}")
                        if sub_name not in excluded:
                            sub_kind, sub_li, sub_ei = _classify_tensor(sub_name, cold_set, all_experts_cold)
                            slices.append(
                                (
                                    sub_name,
                                    sub_kind,
                                    sub_li,
                                    sub_ei,
                                    (half_r, k_c),
                                    t.offset,
                                    half_r * row_b,
                                    False,
                                    None,
                                    0,
                                    None,
                                    None,
                                )
                            )
                            split_extras[sub_name] = (0, 0, runs)

        for (
            out_name,
            out_kind,
            layer_i,
            exp_i,
            shp,
            src_off,
            src_len,
            pair_interleave,
            second_sh,
            second_off,
            exp_scale,
            second_exp_scale,
        ) in slices:
            convert: str | None = None
            out_dt = base_amf_dt
            q_bits = 0
            g_size = 0
            data_bytes = src_len
            params_offset = 0
            params_bytes = 0
            scale_sh: GgufShard | None = None
            scale_info: GgufTensorInfo | None = None
            fp8_mode: str | None = None

            if (
                engram_output is not None
                and out_kind == TensorKind.SPARSE_TABLE
                and len(shp) == 2
                and t.ggml_type in (0, 1, 30)
            ):
                out_dt = DType.FP8_E4M3_ROW
                q_bits = 8
                g_size = 0
                convert = "quant_fp8_row"
                data_bytes = shp[0] * shp[1]
                params_offset = _align_up(data_bytes, 64)
                params_bytes = shp[0] * 4
            elif _convert_ink2_f32_needed(out_name, out_kind, t.ggml_type):
                out_dt = target_f16_dt
                convert = "f32_to_f16" if target_f16_dt == DType.F16 else "f32_to_bf16"
                data_bytes = src_len // 2
            elif out_dt == DType.F16 and dtype_mode == "bf16":
                out_dt = DType.BF16
                convert = "f16_to_bf16"
            elif out_dt == DType.BF16 and dtype_mode == "f16":
                out_dt = DType.F16
                convert = "bf16_to_f16"
            elif t.ggml_type == 40:  # GGUF NVFP4 -> planar nvfp4_g16
                if len(shp) != 2:
                    raise ValueError(f"GGUF NVFP4 tensor {t.name!r} must be 2D (or 3D expert stack), got {shp}")
                n_r, k_c = shp
                out_dt = DType.NVFP4_G16
                q_bits = 4
                g_size = 16
                convert = "gguf_nvfp4"
                data_bytes = n_r * (k_c // 2)
                params_offset = _align_up(data_bytes, 64)
                params_bytes = _align_up(n_r * (k_c // 16), 4) + 4
            elif t.ggml_type == 43:  # GGUF F8_E4M3 -> fp8_e4m3_tensor or fp8_e4m3_row
                q_bits = 8
                g_size = 0
                sc_hit = fp8_pairs.get(t.name)
                if sc_hit is not None:
                    _, scale_sh, scale_info = sc_hit
                    if len(shp) == 2 and (scale_info.amf_shape == (shp[0],) or scale_info.amf_shape == (shp[0], 1)):
                        out_dt = DType.FP8_E4M3_ROW
                        fp8_mode = "fp8_row"
                        params_offset = _align_up(data_bytes, 64)
                        params_bytes = shp[0] * 4
                    elif scale_info.numel == 1:
                        out_dt = DType.FP8_E4M3_TENSOR
                        fp8_mode = "fp8_tensor"
                        params_offset = _align_up(data_bytes, 64)
                        params_bytes = 4
                    else:
                        raise ValueError(
                            f"GGUF F8_E4M3 companion scale {scale_info.name!r} shape {scale_info.amf_shape} "
                            f"incompatible with weight shape {shp}"
                        )
                elif out_kind == TensorKind.SPARSE_TABLE and len(shp) == 2:
                    out_dt = DType.FP8_E4M3_ROW
                    fp8_mode = "fp8_row_ones"
                    params_offset = _align_up(data_bytes, 64)
                    params_bytes = shp[0] * 4
                else:
                    out_dt = DType.FP8_E4M3_TENSOR
                    fp8_mode = "fp8_tensor"
                    params_offset = _align_up(data_bytes, 64)
                    params_bytes = 4
            elif out_dt >= DType.GGUF_Q4_0:
                if len(shp) != 2:
                    raise ValueError(f"GGUF quantized tensor {t.name!r} must be 2D or 3D exps, got {shp}")
                q_bits = _AMF_GGUF_QUANT_BITS[out_dt]
                g_size = bs

            out_layout = TensorLayout.ROW_MAJOR
            if (
                not is_vision_tensor(out_name)
                and out_dt in (DType.F16, DType.BF16)
                and _should_sme2_pack(out_kind, out_dt, shp, eff_sme2_pack, name=out_name)
            ):
                out_layout = TensorLayout.SME2_PANEL64
                data_bytes = sme2_panel64_data_bytes(out_dt, shp[0], shp[1], 0)
                params_offset = 0
                params_bytes = 0
                if convert == "f16_to_bf16":
                    convert = "sme2_f16_to_bf16"
                elif convert == "bf16_to_f16":
                    convert = "sme2_bf16_to_f16"
                elif convert == "f32_to_bf16":
                    convert = "sme2_f32_to_bf16"
                elif convert == "f32_to_f16":
                    convert = "sme2_f32_to_f16"
                else:
                    convert = "sme2_f16"

            content = (params_offset + params_bytes) if params_bytes > 0 else data_bytes
            sc_row_off, sc_row_cnt, gather_runs = split_extras.get(out_name, (0, 0, None))
            entry_obj = _PlannedGgufEntry(
                name=out_name,
                kind=out_kind,
                dtype=out_dt,
                quant_bits=q_bits,
                group_size=g_size,
                layer=layer_i,
                expert=exp_i,
                shape=shp,
                shard=sh,
                src_offset=src_off,
                src_bytes=src_len,
                src_ggml_type=t.ggml_type,
                data_convert=convert,
                scale_shard=scale_sh,
                scale_info=scale_info,
                fp8_mode=fp8_mode,
                data_bytes=data_bytes,
                params_offset=params_offset,
                params_bytes=params_bytes,
                blob_bytes=_align_up(max(content, 1), APPLE_PAGE_SIZE),
                scale_row_offset=sc_row_off if fp8_mode == "fp8_row" else 0,
                scale_row_count=sc_row_cnt if fp8_mode == "fp8_row" else 0,
                layout=out_layout,
                pair_interleave_rows=pair_interleave,
                second_shard=second_sh,
                second_src_offset=second_off,
                expert_scale=exp_scale,
                second_expert_scale=second_exp_scale,
                row_gather_runs=gather_runs,
            )
            if engram_output is not None and out_kind == TensorKind.SPARSE_TABLE:
                planned_engram.append(entry_obj)
            elif not is_clip and is_vision_tensor(out_name) and (strip_vision or vision_sidecar is not None):
                if vision_sidecar is not None:
                    planned_vision.append(entry_obj)
            else:
                planned.append(entry_obj)

    planned.sort(key=lambda e: _natural_key(e.name))
    seen_out: set[str] = set()
    for e in planned:
        if e.name in seen_out:
            raise ValueError(f"duplicate normalized output tensor name {e.name!r} in GGUF {in_path}")
        seen_out.add(e.name)

    # Resolve metadata (external HF tokenizer.json / chat_template.jinja preferred over GGUF KV).
    base_dir = in_path if in_path.is_dir() else in_path.parent
    meta_blobs: list[tuple[str, bytes]] = []
    want_meta = (
        embed_metadata
        or is_clip
        or arch == "dflash"
        or mimo_mtp_mode
        or config_path is not None
        or tokenizer_path is not None
        or chat_template_path is not None
        or meta_config_override is not None
    )
    if want_meta or draft_config_output is not None:
        ext_cfg = pathlib.Path(config_path) if config_path is not None else (base_dir / "config.json")
        if meta_config_override is not None:
            cfg_bytes = meta_config_override
        elif ext_cfg.is_file():
            cfg_bytes = ext_cfg.read_bytes()
        else:
            cfg_bytes = _build_synthesized_config_bytes(kv, physical_layers=physical_layers, dedup_layers=dedup_layers)
        if meta_config_override is None and mimo_mtp_mode and arch.lower() in ("mimo2", "mimo_v2"):
            from aminfer_tools.mimo_v2 import build_mimo_mtp_config_dict

            cfg_dict = json.loads(cfg_bytes)
            mtp_dict = build_mimo_mtp_config_dict(cfg_dict, mtp_layers=max(1, physical_layers))
            cfg_bytes = json.dumps(mtp_dict, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        elif meta_config_override is None and arch.lower() in ("glm-dsa", "glm_moe_dsa", "glm5-next", "glm5_next"):
            from aminfer_tools.glm5 import (
                build_glm5_mtp_config_dict,
                build_glm5_next_config_json,
                build_glm_moe_dsa_config_json,
            )

            cfg_dict = json.loads(cfg_bytes)
            is_g5_next = arch.lower() in ("glm5-next", "glm5_next")
            if mimo_mtp_mode:
                out_cfg_d = build_glm5_mtp_config_dict(cfg_dict, is_glm5_next=is_g5_next)
            elif is_g5_next:
                out_cfg_d = build_glm5_next_config_json(cfg_dict, include_vision=not (strip_vision or vision_sidecar is not None))
            else:
                out_cfg_d = build_glm_moe_dsa_config_json(cfg_dict)
            cfg_bytes = json.dumps(out_cfg_d, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        if draft_config_output is not None:
            dcfg_p = pathlib.Path(draft_config_output)
            dcfg_p.parent.mkdir(parents=True, exist_ok=True)
            dcfg_p.write_bytes(cfg_bytes + (b"" if cfg_bytes.endswith(b"\n") else b"\n"))
        if want_meta and (
            embed_metadata
            or is_clip
            or arch == "dflash"
            or mimo_mtp_mode
            or config_path is not None
            or meta_config_override is not None
        ):
            meta_blobs.append((META_CONFIG_JSON, cfg_bytes))

        ext_tok = pathlib.Path(tokenizer_path) if tokenizer_path is not None else (base_dir / "tokenizer.json")
        tok_bytes: bytes | None = None
        if ext_tok.is_file():
            tok_bytes = ext_tok.read_bytes()
            if strict_vocab_check:
                _cross_check_hf_tokenizer_with_gguf(tok_bytes, kv)
        elif tokenizer_path is not None:
            raise FileNotFoundError(f"tokenizer.json not found: {ext_tok}")
        elif embed_metadata:
            tok_bytes = _build_synthesized_hf_tokenizer_bytes(kv)
        if tok_bytes is not None and (embed_metadata or tokenizer_path is not None):
            meta_blobs.append((META_TOKENIZER_JSON, tok_bytes))

        ext_tpl = pathlib.Path(chat_template_path) if chat_template_path is not None else (base_dir / "chat_template.jinja")
        ext_tok_cfg = (
            pathlib.Path(tokenizer_config_path)
            if tokenizer_config_path is not None
            else (base_dir / "tokenizer_config.json")
        )
        tpl_text: str | None = None
        if ext_tpl.is_file():
            tpl_text = ext_tpl.read_text(encoding="utf-8")
        elif chat_template_path is not None:
            raise FileNotFoundError(f"chat_template.jinja not found: {ext_tpl}")
        tok_cfg_dict = json.loads(ext_tok_cfg.read_text(encoding="utf-8")) if ext_tok_cfg.is_file() else {}
        mod_cfg_dict = json.loads(cfg_bytes.decode("utf-8"))
        if not tpl_text and isinstance(kv.get("tokenizer.chat_template"), str):
            tpl_text = kv["tokenizer.chat_template"]
        if want_meta and (tpl_text or (isinstance(tok_cfg_dict, dict) and tok_cfg_dict.get("chat_template"))):
            spec = compile_chat_template_spec(
                jinja_text=tpl_text,
                tokenizer_config=tok_cfg_dict if isinstance(tok_cfg_dict, dict) else {},
                model_config=mod_cfg_dict if isinstance(mod_cfg_dict, dict) else {},
            )
            meta_blobs.append(
                (META_CHAT_TEMPLATE_JSON, json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            )

    def _read_exact(sfh, n_bytes: int, what: str) -> bytes:
        buf = sfh.read(n_bytes)
        if len(buf) != n_bytes:
            raise ValueError(f"unexpected EOF reading {what}")
        return buf

    def _prepare_expert_raw(e: _PlannedGgufEntry, fhs: dict[pathlib.Path, Any]) -> bytes:
        """Assemble optionally-scaled / pair-interleaved raw bytes for GGUF MoE experts."""
        if e.shard is None:
            assert e.synthetic_bytes is not None
            return e.synthetic_bytes
        if e.second_shard is not None:
            half_len = e.src_bytes // 2
            sfh1 = fhs[e.shard.path]
            sfh1.seek(e.shard.data_offset + e.src_offset)
            g_buf = bytearray(_read_exact(sfh1, half_len, e.name))
            sfh2 = fhs[e.second_shard.path]
            sfh2.seek(e.second_shard.data_offset + e.second_src_offset)
            u_buf = bytearray(_read_exact(sfh2, half_len, e.name))
            if e.dtype >= DType.GGUF_Q4_0:
                _, bb_exp = _AMF_GGUF_BLOCK_INFO[e.dtype]
                if e.expert_scale is not None and e.expert_scale != 1.0:
                    _scale_gguf_blocks_in_place(g_buf, e.dtype, bb_exp, e.expert_scale, e.name)
                if e.second_expert_scale is not None and e.second_expert_scale != 1.0:
                    _scale_gguf_blocks_in_place(u_buf, e.dtype, bb_exp, e.second_expert_scale, e.name)
            half_rows = e.shape[0] // 2
            row_b = half_len // half_rows
            return _interleave_split_rows(g_buf, u_buf, half_rows, row_b)

        sfh = fhs[e.shard.path]
        sfh.seek(e.shard.data_offset + e.src_offset)
        buf = bytearray(_read_exact(sfh, e.src_bytes, e.name))
        if e.dtype >= DType.GGUF_Q4_0 and e.expert_scale is not None and e.expert_scale != 1.0:
            _, bb_exp = _AMF_GGUF_BLOCK_INFO[e.dtype]
            _scale_gguf_blocks_in_place(buf, e.dtype, bb_exp, e.expert_scale, e.name)
        if e.pair_interleave_rows:
            n_r = e.shape[0]
            return _interleave_pair_rows(buf, n_r, e.src_bytes // n_r)
        return bytes(buf)

    def _write_staged(
        target_amf: pathlib.Path,
        entries_list: Sequence[_PlannedGgufEntry],
        target_meta: Sequence[tuple[str, bytes]],
        target_model_id: str,
    ) -> list[PackedTensorMeta]:
        tensor_count = len(entries_list) + len(target_meta)
        dir_offset = APPLE_PAGE_SIZE
        dir_raw_len = tensor_count * ENTRY_STRUCT.size
        dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)
        cursor = dir_offset + dir_bytes
        offsets: list[int] = []
        for e in entries_list:
            offsets.append(cursor)
            cursor += e.blob_bytes
        m_offsets: list[tuple[str, bytes, int, int]] = []
        for m_name, m_bytes in target_meta:
            m_blob = _align_up(len(m_bytes), APPLE_PAGE_SIZE)
            m_offsets.append((m_name, m_bytes, cursor, m_blob))
            cursor += m_blob

        target_amf.parent.mkdir(parents=True, exist_ok=True)
        partial = target_amf.with_name(target_amf.name + ".partial")
        in_fhs = {sh.path: sh.path.open("rb") for sh in shards}
        metas: list[PackedTensorMeta] = []
        try:
            with partial.open("wb") as out_fh:
                _set_nocache(out_fh.fileno())
                _BlobWriter(out_fh).zeros(dir_offset + dir_bytes)

                for e, blob_offset in zip(entries_list, offsets):
                    w = _BlobWriter(out_fh)
                    needs_expert_buf = (
                        e.synthetic_bytes is not None
                        or e.pair_interleave_rows
                        or e.second_shard is not None
                        or (e.expert_scale is not None and e.expert_scale != 1.0 and e.data_convert != "gguf_nvfp4")
                    )
                    def _read_gathered_source_bytes(entry: _PlannedGgufEntry) -> bytes:
                        assert entry.shard is not None and entry.row_gather_runs is not None
                        sfh_g = in_fhs[entry.shard.path]
                        if entry.src_bytes % entry.shape[0] != 0:
                            raise AssertionError(
                                f"row_gather src_bytes {entry.src_bytes} not divisible by rows {entry.shape[0]}"
                            )
                        row_src_b_g = entry.src_bytes // entry.shape[0]
                        chunks_g: list[bytes] = []
                        for r0_g, cnt_g in entry.row_gather_runs:
                            sfh_g.seek(entry.shard.data_offset + entry.src_offset + r0_g * row_src_b_g)
                            chunks_g.append(_read_exact(sfh_g, cnt_g * row_src_b_g, entry.name))
                        return b"".join(chunks_g)

                    if e.synthetic_bytes is not None:
                        w.write(e.synthetic_bytes)
                    elif e.row_gather_runs is not None and e.data_convert is None:
                        assert e.shard is not None
                        sfh = in_fhs[e.shard.path]
                        row_src_b = e.src_bytes // e.shape[0]
                        if e.src_bytes % e.shape[0] != 0:
                            raise AssertionError(f"row_gather src_bytes {e.src_bytes} not divisible by rows {e.shape[0]}")
                        for r0, cnt in e.row_gather_runs:
                            sfh.seek(e.shard.data_offset + e.src_offset + r0 * row_src_b)
                            left_g = cnt * row_src_b
                            while left_g > 0:
                                take = min(8 * 1024 * 1024, left_g)
                                w.write(_read_exact(sfh, take, e.name))
                                left_g -= take
                    elif e.data_convert is None and needs_expert_buf:
                        w.write(_prepare_expert_raw(e, in_fhs))
                    elif e.data_convert is None:
                        assert e.shard is not None
                        sfh = in_fhs[e.shard.path]
                        sfh.seek(e.shard.data_offset + e.src_offset)
                        left = e.src_bytes
                        while left > 0:
                            take = min(8 * 1024 * 1024, left)
                            chunk = _read_exact(sfh, take, e.name)
                            w.write(chunk)
                            left -= take
                    elif e.data_convert in ("f16_to_bf16", "bf16_to_f16"):
                        if e.row_gather_runs is not None:
                            chunk = _read_gathered_source_bytes(e)
                            w.write(
                                f16_bytes_to_bf16_bytes(chunk)
                                if e.data_convert == "f16_to_bf16"
                                else bf16_bytes_to_f16_bytes(chunk, what=e.name)
                            )
                        else:
                            assert e.shard is not None
                            sfh = in_fhs[e.shard.path]
                            sfh.seek(e.shard.data_offset + e.src_offset)
                            left = e.src_bytes
                            while left > 0:
                                take = min((1 << 20) * 2, left)
                                chunk = _read_exact(sfh, take, e.name)
                                w.write(
                                    f16_bytes_to_bf16_bytes(chunk)
                                    if e.data_convert == "f16_to_bf16"
                                    else bf16_bytes_to_f16_bytes(chunk, what=e.name)
                                )
                                left -= take
                    elif e.data_convert in ("f32_to_bf16", "f32_to_f16"):
                        if e.row_gather_runs is not None:
                            raw = _read_gathered_source_bytes(e)
                        else:
                            assert e.shard is not None
                            sfh = in_fhs[e.shard.path]
                            sfh.seek(e.shard.data_offset + e.src_offset)
                            raw = _read_exact(sfh, e.src_bytes, e.name)
                        vals = struct.unpack(f"<{e.src_bytes // 4}f", raw)
                        w.write(f32_to_f16_bytes(vals) if e.data_convert == "f32_to_f16" else f32_to_bf16_bytes(vals))
                    elif e.data_convert in (
                        "sme2_f16",
                        "sme2_f16_to_bf16",
                        "sme2_bf16_to_f16",
                        "sme2_f32_to_bf16",
                        "sme2_f32_to_f16",
                    ):
                        n_r, k_c = e.shape
                        src_row_bytes = k_c * 4 if e.data_convert in ("sme2_f32_to_bf16", "sme2_f32_to_f16") else k_c * 2
                        dst_row_bytes = k_c * 2
                        gathered_buf = memoryview(_read_gathered_source_bytes(e)) if e.row_gather_runs is not None else None
                        sfh = None
                        if gathered_buf is None:
                            assert e.shard is not None
                            sfh = in_fhs[e.shard.path]
                            sfh.seek(e.shard.data_offset + e.src_offset)
                        panels = (n_r + SME2_PANEL_COLS - 1) // SME2_PANEL_COLS
                        for p in range(panels):
                            r0 = p * SME2_PANEL_COLS
                            cnt = min(SME2_PANEL_COLS, n_r - r0)
                            if gathered_buf is not None:
                                raw = bytes(gathered_buf[r0 * src_row_bytes : (r0 + cnt) * src_row_bytes])
                            else:
                                assert sfh is not None
                                raw = _read_exact(sfh, cnt * src_row_bytes, e.name)
                            if e.data_convert == "sme2_f16_to_bf16":
                                raw = f16_bytes_to_bf16_bytes(raw)
                            elif e.data_convert == "sme2_bf16_to_f16":
                                raw = bf16_bytes_to_f16_bytes(raw, what=e.name)
                            elif e.data_convert == "sme2_f32_to_bf16":
                                raw = f32_to_bf16_bytes(struct.unpack(f"<{cnt * k_c}f", raw))
                            elif e.data_convert == "sme2_f32_to_f16":
                                raw = f32_to_f16_bytes(struct.unpack(f"<{cnt * k_c}f", raw))
                            mv = memoryview(raw)
                            rows_mv = [mv[c * dst_row_bytes : (c + 1) * dst_row_bytes] for c in range(cnt)]
                            w.write(pack_sme2_panel64_f16_panel(rows_mv, k_c))
                    elif e.data_convert == "gguf_nvfp4":
                        n_r, k_c = e.shape
                        row_src_b = (k_c // 64) * 36
                        scales_acc = bytearray()
                        if e.pair_interleave_rows or e.second_shard is not None:
                            full_raw = memoryview(_prepare_expert_raw(e, in_fhs))
                            step = 4096
                            for r0 in range(0, n_r, step):
                                r1 = min(n_r, r0 + step)
                                d_chunk, s_chunk = _unsplit_nvfp4_batch(full_raw[r0 * row_src_b : r1 * row_src_b])
                                w.write(d_chunk)
                                scales_acc.extend(s_chunk)
                        else:
                            assert e.shard is not None
                            sfh = in_fhs[e.shard.path]
                            sfh.seek(e.shard.data_offset + e.src_offset)
                            left_rows = n_r
                            while left_rows > 0:
                                take_rows = min(4096, left_rows)
                                chunk_raw = _read_exact(sfh, take_rows * row_src_b, e.name)
                                d_chunk, s_chunk = _unsplit_nvfp4_batch(chunk_raw)
                                w.write(d_chunk)
                                scales_acc.extend(s_chunk)
                                left_rows -= take_rows
                        w.zeros(e.params_offset - e.data_bytes)
                        w.write(scales_acc)
                        pad4 = _align_up(len(scales_acc), 4) - len(scales_acc)
                        if pad4 > 0:
                            w.zeros(pad4)
                        outer_sc = e.expert_scale if e.expert_scale is not None else 1.0
                        w.write(struct.pack("<f", outer_sc))
                    elif e.data_convert == "quant_fp8_row":
                        n_r, k_c = e.shape
                        elem_b = 4 if e.src_ggml_type == 0 else 2
                        row_src_b = k_c * elem_b
                        gathered_qbuf = memoryview(_read_gathered_source_bytes(e)) if e.row_gather_runs is not None else None
                        sfh = None
                        if gathered_qbuf is None:
                            assert e.shard is not None
                            sfh = in_fhs[e.shard.path]
                            sfh.seek(e.shard.data_offset + e.src_offset)
                        scales_f32: list[float] = []
                        for r_i in range(n_r):
                            if gathered_qbuf is not None:
                                r_raw = bytes(gathered_qbuf[r_i * row_src_b : (r_i + 1) * row_src_b])
                            else:
                                assert sfh is not None
                                r_raw = _read_exact(sfh, row_src_b, e.name)
                            if e.src_ggml_type == 0:
                                vals = list(struct.unpack(f"<{k_c}f", r_raw))
                            elif e.src_ggml_type == 1:
                                vals = f16_bytes_to_f32_list(r_raw)
                            else:
                                vals = bf16_bytes_to_f32_list(r_raw)
                            q_row, r_sc = quantize_row_f32_to_e4m3_row(vals)
                            w.write(q_row)
                            scales_f32.append(r_sc)
                        w.zeros(e.params_offset - e.data_bytes)
                        w.write(struct.pack(f"<{n_r}f", *scales_f32))

                    if e.fp8_mode is not None:
                        w.zeros(e.params_offset - e.data_bytes)
                        if e.scale_shard is not None and e.scale_info is not None:
                            sc_bytes = _decode_gguf_float_array(e.scale_shard, e.scale_info)
                            if e.row_gather_runs is not None and e.fp8_mode == "fp8_row":
                                sc_bytes = b"".join(sc_bytes[r0 * 4 : (r0 + cnt) * 4] for r0, cnt in e.row_gather_runs)
                            elif e.scale_row_count > 0:
                                sc_bytes = sc_bytes[e.scale_row_offset * 4 : (e.scale_row_offset + e.scale_row_count) * 4]
                            w.write(sc_bytes)
                        elif e.fp8_mode == "fp8_row_ones":
                            w.write(struct.pack(f"<{e.shape[0]}f", *([1.0] * e.shape[0])))
                        else:
                            w.write(struct.pack("<f", 1.0))

                    content_done = (e.params_offset + e.params_bytes) if e.params_bytes > 0 else e.data_bytes
                    w.zeros(e.blob_bytes - content_done)
                    if w.written != e.blob_bytes:
                        raise AssertionError(f"wrote {w.written} != planned {e.blob_bytes} for {e.name!r}")

                    metas.append(
                        PackedTensorMeta(
                            name=e.name,
                            kind=e.kind,
                            dtype=e.dtype,
                            quant_bits=e.quant_bits,
                            group_size=e.group_size,
                            layer=e.layer,
                            expert=e.expert,
                            shape=e.shape,
                            blob_offset=blob_offset,
                            blob_bytes=e.blob_bytes,
                            data_bytes=e.data_bytes,
                            params_offset=e.params_offset,
                            params_bytes=e.params_bytes,
                            blob_crc32=w.crc & 0xFFFFFFFF,
                            layout=e.layout,
                            quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                        )
                    )

                for m_name, m_bytes, m_off, m_blob in m_offsets:
                    w = _BlobWriter(out_fh)
                    w.write(m_bytes)
                    if m_blob > len(m_bytes):
                        w.zeros(m_blob - len(m_bytes))
                    metas.append(
                        PackedTensorMeta(
                            name=m_name,
                            kind=TensorKind.METADATA,
                            dtype=DType.U32,
                            quant_bits=0,
                            group_size=0,
                            layer=NO_LAYER,
                            expert=NO_EXPERT,
                            shape=(len(m_bytes),),
                            blob_offset=m_off,
                            blob_bytes=m_blob,
                            data_bytes=len(m_bytes),
                            params_offset=0,
                            params_bytes=0,
                            blob_crc32=w.crc & 0xFFFFFFFF,
                            layout=TensorLayout.ROW_MAJOR,
                            quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                        )
                    )

                if out_fh.tell() != cursor:
                    raise AssertionError(f"GGUF pack final size {out_fh.tell()} != planned {cursor}")
                dir_raw = _pack_directory(metas)
                hdr_flags = FLAG_EMBEDDED_METADATA if m_offsets else 0
                hdr = _pack_header(
                    target_model_id.encode("utf-8"),
                    tensor_count,
                    dir_offset,
                    dir_bytes,
                    _crc32(dir_raw),
                    flags=hdr_flags,
                )
                out_fh.seek(0)
                out_fh.write(hdr)
                if dir_raw:
                    out_fh.seek(dir_offset)
                    out_fh.write(dir_raw)
                out_fh.flush()
                os.fsync(out_fh.fileno())
            os.replace(partial, target_amf)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        finally:
            for fh in in_fhs.values():
                fh.close()
        return metas

    if engram_output is not None and planned_engram:
        planned_engram.sort(key=lambda e: _natural_key(e.name))
        eng_base = pathlib.Path(engram_output)
        if engram_shards == 1:
            _write_staged(eng_base, planned_engram, [], f"{eff_model_id[:55]}-engram")
        else:
            # Partition multi-row Engram tables deterministically across shard files.
            for s_idx in range(engram_shards):
                shard_path = eng_base.with_name(f"{eng_base.stem}-{s_idx:05d}-of-{engram_shards:05d}{eng_base.suffix}")
                shard_entries: list[_PlannedGgufEntry] = []
                for pe in planned_engram:
                    if len(pe.shape) == 2 and pe.shape[0] >= engram_shards:
                        n_rows, k_c = pe.shape
                        base_rows = n_rows // engram_shards
                        rem = n_rows % engram_shards
                        r0 = s_idx * base_rows + min(s_idx, rem)
                        r1 = r0 + base_rows + (1 if s_idx < rem else 0)
                        cnt = r1 - r0
                        row_b = pe.src_bytes // n_rows
                        src_shard_b = cnt * row_b
                        if pe.data_convert == "quant_fp8_row":
                            d_b = cnt * k_c
                            p_off = _align_up(d_b, 64)
                            p_b = cnt * 4
                        else:
                            d_b = src_shard_b
                            p_off = _align_up(d_b, 64) if pe.params_bytes > 0 else 0
                            p_b = cnt * 4 if pe.fp8_mode in ("fp8_row", "fp8_row_ones") else pe.params_bytes
                        shard_entries.append(
                            _PlannedGgufEntry(
                                name=f"{pe.name}.s{s_idx}" if len(f"{pe.name}.s{s_idx}") <= 47 else pe.name[:44] + f".{s_idx}",
                                kind=pe.kind,
                                dtype=pe.dtype,
                                quant_bits=pe.quant_bits,
                                group_size=pe.group_size,
                                layer=pe.layer,
                                expert=pe.expert,
                                shape=(cnt, k_c),
                                shard=pe.shard,
                                src_offset=pe.src_offset + r0 * row_b,
                                src_bytes=src_shard_b,
                                src_ggml_type=pe.src_ggml_type,
                                data_convert=pe.data_convert,
                                scale_shard=pe.scale_shard,
                                scale_info=pe.scale_info,
                                fp8_mode=pe.fp8_mode,
                                data_bytes=d_b,
                                params_offset=p_off,
                                params_bytes=p_b,
                                blob_bytes=_align_up(max((p_off + p_b) if p_b > 0 else d_b, 1), APPLE_PAGE_SIZE),
                                scale_row_offset=r0 if pe.fp8_mode == "fp8_row" else 0,
                                scale_row_count=cnt if pe.fp8_mode == "fp8_row" else 0,
                                layout=pe.layout,
                                quant_param_kind=pe.quant_param_kind,
                            )
                        )
                    elif s_idx == 0:
                        shard_entries.append(pe)
                if shard_entries:
                    _write_staged(shard_path, shard_entries, [], f"{eff_model_id[:52]}-engram-{s_idx}")

    if vision_sidecar is not None and planned_vision:
        planned_vision.sort(key=lambda e: _natural_key(e.name))
        vis_out = pathlib.Path(vision_sidecar)
        vis_staged = vis_out.with_name(vis_out.name + ".vision_stage.amf")
        try:
            _write_staged(vis_staged, planned_vision, [], f"{eff_model_id[:55]}-vision")
            pack_vision_tower_amf(
                vis_staged,
                vis_out,
                model_id=f"{eff_model_id[:55]}-vision",
                dtype_mode=dtype_mode,
                vision_block_dtype=vision_block_dtype,
                group_size=group_size,
                embed_metadata=embed_metadata,
                config_path=config_path,
                stats_out=stats_out,
            )
        finally:
            vis_staged.unlink(missing_ok=True)

    if is_clip:
        staged = out_path.with_name(out_path.name + ".clip_stage.amf")
        try:
            _write_staged(staged, planned, meta_blobs, eff_model_id)
            return pack_vision_tower_amf(
                staged,
                out_path,
                model_id=eff_model_id,
                dtype_mode=dtype_mode,
                vision_block_dtype=vision_block_dtype,
                group_size=group_size,
                embed_metadata=embed_metadata,
                config_path=config_path,
                stats_out=stats_out,
            )
        finally:
            staged.unlink(missing_ok=True)

    return _write_staged(out_path, planned, meta_blobs, eff_model_id)


def export_dflare_bundle_from_gguf(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    model_id: str | None = None,
    dtype_mode: str = "auto",
    config_path: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path, list[PackedTensorMeta]]:
    """Export a standalone `arch = dflash` GGUF into a `D14` bundle (`dflare.amf` + `dflare_config.json`)."""
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    amf_path = out_dir / "dflare.amf"
    cfg_path = out_dir / "dflare_config.json"
    metas = pack_amf_from_gguf(
        input_path,
        amf_path,
        model_id=model_id,
        dtype_mode=dtype_mode,
        embed_metadata=True,
        config_path=config_path,
        draft_config_output=cfg_path,
    )
    return amf_path, cfg_path, metas
