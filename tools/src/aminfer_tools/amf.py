"""aminfer Model Format (.amf v0) writer and MLX / Hugging Face safetensors importer.

Pure standard-library Python implementation (ADR-0006, ADR-0016) so offline
conversion and testing require zero third-party packages.

`pack_amf_from_safetensors` converts a single `.safetensors` file, a
`model.safetensors.index.json` multi-shard checkpoint, or a checkpoint
directory in two passes with bounded host memory:

1. Pass 1 reads only the shard JSON headers and plans every output entry
   (name, dtype, shape, 16 KiB-aligned blob offset and size).
2. Pass 2 streams each tensor's byte range from its owning shard straight into
   the `.amf` file in fixed-size chunks, computing CRC-32 incrementally, then
   writes page 0 (header) and the directory last.
"""

from __future__ import annotations

import array
import enum
import fcntl
import json
import os
import pathlib
import re
import struct
import sys
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from aminfer_tools import APPLE_PAGE_SIZE

AMF_MAGIC = b"AMF0\x00\x00\x00\x00"
AMF_VERSION = 1
CHECKSUM_CRC32_IEEE = 1
NO_LAYER = 0xFFFF
NO_EXPERT = 0xFFFF

HEADER_STRUCT = struct.Struct("<8sIIIIIIQQII64sQ")
ENTRY_STRUCT = struct.Struct("<48sBBBBHHHHI4IQQQQQQ")
assert HEADER_STRUCT.size == 128
assert ENTRY_STRUCT.size == 128

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
_SWITCH_MLP_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)$")

# Streaming granularity: raw copies move 8 MiB at a time; dtype conversions
# process 1 Mi elements at a time. Both bound Python RSS independent of model size.
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_CONVERT_CHUNK_ELEMS = 1 << 20
_ZERO_CHUNK = bytes(1 << 20)
_F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class TensorKind(enum.IntEnum):
    DENSE = 0
    HOT_EXPERT = 1
    COLD_EXPERT = 2
    EMBEDDING = 3
    NORM = 4
    ROUTER = 5
    LM_HEAD = 6


class DType(enum.IntEnum):
    F16 = 0
    F32 = 1
    Q4_AFFINE = 2
    Q3_AFFINE = 3
    BF16 = 4
    U32 = 5


@dataclass(frozen=True)
class PackedTensorMeta:
    name: str
    kind: TensorKind
    dtype: DType
    quant_bits: int
    group_size: int
    layer: int
    expert: int
    shape: tuple[int, ...]
    blob_offset: int
    blob_bytes: int
    data_bytes: int
    params_offset: int
    params_bytes: int
    blob_crc32: int


def _align_up(length: int, alignment: int) -> int:
    return (length + alignment - 1) & ~(alignment - 1)


def _crc32(data: bytes | bytearray | memoryview) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _bf16_to_f32_bytes(raw: bytes | bytearray | memoryview) -> bytearray:
    """Widen little-endian bfloat16 bytes to little-endian float32 bytes (exact)."""
    out = bytearray(len(raw) * 2)
    out[2::4] = raw[0::2]
    out[3::4] = raw[1::2]
    return out


def _f32_bytes_to_f16_bytes(raw: bytes | bytearray | memoryview, what: str) -> bytes:
    count = len(raw) // 4
    try:
        return struct.pack(f"<{count}e", *struct.unpack(f"<{count}f", raw))
    except OverflowError as exc:
        raise ValueError(f"{what}: value exceeds the IEEE float16 range (|x| > 65504)") from exc


def bf16_bytes_to_f16_bytes(raw: bytes | bytearray | memoryview, *, what: str = "bf16 tensor") -> bytes:
    """Convert little-endian bfloat16 bytes to little-endian IEEE 754 float16 bytes.

    Rounds to nearest, ties to even. Raises `ValueError` if any finite value
    overflows float16 (instead of silently producing infinities).
    """
    if len(raw) % 2 != 0:
        raise ValueError(f"bf16 byte length must be even, got {len(raw)}")
    return _f32_bytes_to_f16_bytes(_bf16_to_f32_bytes(raw), what)


def f16_bytes_to_bf16_bytes(raw: bytes | bytearray | memoryview) -> bytes:
    """Convert little-endian IEEE float16 bytes to bfloat16 bytes (round to nearest even).

    Infinities are preserved and NaNs stay (quiet) NaNs.
    """
    if len(raw) % 2 != 0:
        raise ValueError(f"f16 byte length must be even, got {len(raw)}")
    count = len(raw) // 2
    bits = array.array("I", struct.pack(f"<{count}f", *struct.unpack(f"<{count}e", raw)))
    if sys.byteorder != "little":
        bits.byteswap()
    out = array.array(
        "H",
        [
            ((u >> 16) | 0x0040) if (u & 0x7FFFFFFF) > 0x7F800000 else ((u + 0x7FFF + ((u >> 16) & 1)) >> 16)
            for u in bits
        ],
    )
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def d2t_deltas_to_u32_bytes(
    raw: bytes | bytearray | memoryview,
    *,
    elem_bytes: int,
    start_index: int = 0,
    what: str = "d2t",
) -> bytes:
    """Convert EAGLE-3 signed `d2t` offset deltas (I64 or I32) to direct u32 target token IDs.

    Each entry `j` (global draft token index `i = start_index + j`) maps to
    `target_id = i + int(d2t[i])`, validated to lie within `0 <= target_id < 2^32`.
    """
    if elem_bytes not in (4, 8):
        raise ValueError(f"{what}: expected 4- or 8-byte signed integers, got {elem_bytes}")
    if len(raw) % elem_bytes != 0:
        raise ValueError(f"{what}: byte length {len(raw)} is not a multiple of {elem_bytes}")
    count = len(raw) // elem_bytes
    fmt_in = "q" if elem_bytes == 8 else "i"
    deltas = struct.unpack(f"<{count}{fmt_in}", raw)
    out = array.array("I")
    for j, delta in enumerate(deltas):
        idx = start_index + j
        target_id = idx + delta
        if target_id < 0 or target_id > 0xFFFFFFFF:
            raise ValueError(
                f"{what}[{idx}]: target token id {idx} + {delta} = {target_id} is outside u32 range"
            )
        out.append(target_id)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def interleave_scale_bias_f16(scales_f16: bytes, biases_f16: bytes) -> bytes:
    """Interleave separate fp16 scales and biases into half2(scale, bias) pairs."""
    if len(scales_f16) != len(biases_f16) or len(scales_f16) % 2 != 0:
        raise ValueError("scales and biases must have identical even byte lengths")
    out = bytearray(len(scales_f16) * 2)
    out[0::4] = scales_f16[0::2]
    out[1::4] = scales_f16[1::2]
    out[2::4] = biases_f16[0::2]
    out[3::4] = biases_f16[1::2]
    return bytes(out)


def pack_mlx_affine_4bit(codes: Sequence[int]) -> bytes:
    """Pack 4-bit unsigned codes LSB-first (2 values per byte, MLX uint32 compatible)."""
    if len(codes) % 8 != 0:
        raise ValueError("4-bit code count must be a multiple of 8")
    out = bytearray(len(codes) // 2)
    for i in range(len(out)):
        lo = codes[2 * i] & 0x0F
        hi = codes[2 * i + 1] & 0x0F
        out[i] = lo | (hi << 4)
    return bytes(out)


def pack_mlx_affine_3bit(codes: Sequence[int]) -> bytes:
    """Pack 3-bit unsigned codes LSB-first (8 values in 3 bytes, MLX uint32 compatible)."""
    if len(codes) % 8 != 0:
        raise ValueError("3-bit code count must be a multiple of 8")
    out = bytearray((len(codes) // 8) * 3)
    for blk in range(len(codes) // 8):
        c = codes[blk * 8 : blk * 8 + 8]
        word = 0
        for j in range(8):
            word |= (c[j] & 0x07) << (3 * j)
        out[blk * 3 + 0] = word & 0xFF
        out[blk * 3 + 1] = (word >> 8) & 0xFF
        out[blk * 3 + 2] = (word >> 16) & 0xFF
    return bytes(out)


def _normalize_name(raw_name: str) -> str:
    name = raw_name[6:] if raw_name.startswith("model.") else raw_name
    encoded = name.encode("utf-8")
    if len(encoded) == 0 or len(encoded) > 47:
        raise ValueError(f"AMF tensor name must be 1..47 UTF-8 bytes, got {len(encoded)}: {name!r}")
    return name


def _classify_tensor(
    name: str,
    cold_experts: set[tuple[int, int]],
    all_experts_cold: bool,
) -> tuple[TensorKind, int, int]:
    layer_match = _LAYER_RE.search(name)
    layer = int(layer_match.group(1)) if layer_match else NO_LAYER
    expert_match = _EXPERT_RE.search(name)
    expert = int(expert_match.group(1)) if expert_match else NO_EXPERT

    if expert != NO_EXPERT:
        is_cold = all_experts_cold or (layer, expert) in cold_experts
        return (TensorKind.COLD_EXPERT if is_cold else TensorKind.HOT_EXPERT, layer, expert)
    lower = name.lower()
    if "norm" in lower:
        return (TensorKind.NORM, layer, expert)
    if "embed" in lower:
        return (TensorKind.EMBEDDING, layer, expert)
    if "lm_head" in lower:
        return (TensorKind.LM_HEAD, layer, expert)
    if lower.endswith(".mlp.gate") or lower.endswith(".mlp.gate.weight"):
        return (TensorKind.ROUTER, layer, expert)
    return (TensorKind.DENSE, layer, expert)


def write_amf(
    output_path: pathlib.Path,
    model_id: str,
    records: Sequence[tuple[str, TensorKind, DType, int, int, int, int, tuple[int, ...], bytes, bytes]],
) -> list[PackedTensorMeta]:
    """Write a validated, 16 KiB-aligned .amf v0 file.

    Each record is:
      (name, kind, dtype, quant_bits, group_size, layer, expert, shape, data_bytes, params_bytes)
    """
    model_id_bytes = model_id.encode("utf-8")
    if len(model_id_bytes) > 63:
        raise ValueError(f"model_id exceeds 63 UTF-8 bytes: {model_id!r}")

    tensor_count = len(records)
    dir_offset = APPLE_PAGE_SIZE
    dir_raw_len = tensor_count * ENTRY_STRUCT.size
    dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)

    cursor = dir_offset + dir_bytes
    metas: list[PackedTensorMeta] = []
    blobs: list[bytes] = []
    seen_names: set[str] = set()

    for (
        name,
        kind,
        dtype,
        quant_bits,
        group_size,
        layer,
        expert,
        shape,
        data,
        params,
    ) in records:
        if name in seen_names:
            raise ValueError(f"duplicate tensor name: {name!r}")
        seen_names.add(name)
        name_bytes = name.encode("utf-8")
        if len(name_bytes) == 0 or len(name_bytes) > 47:
            raise ValueError(f"invalid tensor name length: {name!r}")
        if not (1 <= len(shape) <= 4):
            raise ValueError(f"tensor rank must be 1..4 for {name!r}, got {len(shape)}")

        if params:
            params_offset = _align_up(len(data), 64)
            content_len = params_offset + len(params)
        else:
            params_offset = 0
            content_len = len(data)

        blob_bytes = _align_up(content_len, APPLE_PAGE_SIZE)
        blob = bytearray(blob_bytes)
        blob[0 : len(data)] = data
        if params:
            blob[params_offset : params_offset + len(params)] = params
        blob_crc = _crc32(blob)

        metas.append(
            PackedTensorMeta(
                name=name,
                kind=kind,
                dtype=dtype,
                quant_bits=quant_bits,
                group_size=group_size,
                layer=layer,
                expert=expert,
                shape=shape,
                blob_offset=cursor,
                blob_bytes=blob_bytes,
                data_bytes=len(data),
                params_offset=params_offset,
                params_bytes=len(params),
                blob_crc32=blob_crc,
            )
        )
        blobs.append(bytes(blob))
        cursor += blob_bytes

    dir_raw = bytearray(dir_raw_len)
    for i, m in enumerate(metas):
        padded_shape = [m.shape[d] if d < len(m.shape) else 0 for d in range(4)]
        ENTRY_STRUCT.pack_into(
            dir_raw,
            i * ENTRY_STRUCT.size,
            m.name.encode("utf-8"),
            int(m.kind),
            int(m.dtype),
            m.quant_bits,
            len(m.shape),
            m.group_size,
            m.layer,
            m.expert,
            0,
            m.blob_crc32,
            padded_shape[0],
            padded_shape[1],
            padded_shape[2],
            padded_shape[3],
            m.blob_offset,
            m.blob_bytes,
            m.data_bytes,
            m.params_offset,
            m.params_bytes,
            0,
        )
    dir_crc = _crc32(dir_raw)

    header_zero_crc = HEADER_STRUCT.pack(
        AMF_MAGIC,
        AMF_VERSION,
        0,
        APPLE_PAGE_SIZE,
        CHECKSUM_CRC32_IEEE,
        tensor_count,
        0,
        dir_offset,
        dir_bytes,
        dir_crc,
        0,
        model_id_bytes,
        0,
    )
    header_crc = _crc32(header_zero_crc)
    header_packed = HEADER_STRUCT.pack(
        AMF_MAGIC,
        AMF_VERSION,
        0,
        APPLE_PAGE_SIZE,
        CHECKSUM_CRC32_IEEE,
        tensor_count,
        0,
        dir_offset,
        dir_bytes,
        dir_crc,
        header_crc,
        model_id_bytes,
        0,
    )

    page0 = bytearray(APPLE_PAGE_SIZE)
    page0[0 : len(header_packed)] = header_packed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out:
        out.write(page0)
        if dir_bytes > 0:
            dir_page = bytearray(dir_bytes)
            dir_page[0 : len(dir_raw)] = dir_raw
            out.write(dir_page)
        for blob in blobs:
            out.write(blob)

    return metas


# ---------------------------------------------------------------------------
# Streaming multi-shard safetensors importer (ADR-0016, W12)
# ---------------------------------------------------------------------------

DTYPE_MODES = ("auto", "f16", "bf16")


@dataclass(frozen=True)
class _ShardTensor:
    """One tensor as described by its shard's JSON header (no data loaded)."""

    name: str
    path: pathlib.Path
    dtype: str
    shape: tuple[int, ...]
    offset: int  # absolute byte offset of the tensor data inside `path`
    length: int


@dataclass(frozen=True)
class _ByteRange:
    path: pathlib.Path
    offset: int
    length: int
    dtype: str


@dataclass(frozen=True)
class _PlannedEntry:
    name: str
    kind: TensorKind
    dtype: DType
    quant_bits: int
    group_size: int
    layer: int
    expert: int
    shape: tuple[int, ...]
    data: _ByteRange
    data_convert: str | None  # None, "bf16_to_f16", "f16_to_bf16", "d2t_i64_to_u32", "d2t_i32_to_u32"
    scales: _ByteRange | None
    biases: _ByteRange | None
    data_bytes: int
    params_offset: int
    params_bytes: int
    blob_bytes: int


@dataclass(frozen=True)
class ResolvedCheckpoint:
    shard_paths: tuple[pathlib.Path, ...]
    config_path: pathlib.Path | None


def resolve_checkpoint(input_path: pathlib.Path) -> ResolvedCheckpoint:
    """Resolve a `.safetensors` file, an `index.json` file, or a checkpoint directory."""
    input_path = pathlib.Path(input_path)
    if input_path.is_dir():
        index_path = input_path / "model.safetensors.index.json"
        single_path = input_path / "model.safetensors"
        if index_path.is_file():
            return resolve_checkpoint(index_path)
        if single_path.is_file():
            return resolve_checkpoint(single_path)
        raise FileNotFoundError(
            f"{input_path} contains neither model.safetensors.index.json nor model.safetensors"
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"input checkpoint not found: {input_path}")

    parent = input_path.parent
    config_path = parent / "config.json"
    config = config_path if config_path.is_file() else None
    if input_path.name.endswith(".json"):
        index = json.loads(input_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{input_path} has no non-empty 'weight_map'")
        shard_names = sorted({str(v) for v in weight_map.values()})
        shards = []
        for shard in shard_names:
            shard_path = parent / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"shard listed in {input_path.name} is missing: {shard_path}")
            shards.append(shard_path)
        return ResolvedCheckpoint(tuple(shards), config)
    return ResolvedCheckpoint((input_path,), config)


def _read_shard_header(path: pathlib.Path) -> list[_ShardTensor]:
    """Read only the JSON header of a safetensors shard (kilobytes, never tensor data)."""
    file_size = path.stat().st_size
    with path.open("rb") as f:
        hdr_len_bytes = f.read(8)
        if len(hdr_len_bytes) != 8:
            raise ValueError(f"truncated safetensors header in {path}")
        (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
        if hdr_len > file_size - 8 or hdr_len > (256 << 20):
            raise ValueError(f"implausible safetensors header length {hdr_len} in {path}")
        hdr_raw = f.read(hdr_len)
        if len(hdr_raw) != hdr_len:
            raise ValueError(f"truncated safetensors JSON in {path}")
    header = json.loads(hdr_raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be a JSON object in {path}")
    data_base = 8 + hdr_len
    out: list[_ShardTensor] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = str(spec["dtype"])
        shape = tuple(int(x) for x in spec["shape"])
        start, end = (int(x) for x in spec["data_offsets"])
        elem = _SAFETENSORS_DTYPE_BYTES.get(dtype)
        if elem is None:
            raise ValueError(f"unsupported safetensors dtype {dtype} for {name} in {path}")
        numel = 1
        for d in shape:
            numel *= d
        if end < start or end - start != numel * elem:
            raise ValueError(f"tensor {name} in {path}: data_offsets {start}..{end} do not match {dtype}{list(shape)}")
        if data_base + end > file_size:
            raise ValueError(f"tensor {name} in {path} extends past end of file")
        out.append(_ShardTensor(name, path, dtype, shape, data_base + start, end - start))
    return out


def _collect_tensors(shards: Sequence[pathlib.Path]) -> dict[str, _ShardTensor]:
    tensors: dict[str, _ShardTensor] = {}
    for shard in shards:
        for t in _read_shard_header(shard):
            if t.name in tensors:
                raise ValueError(f"tensor {t.name!r} appears in both {tensors[t.name].path} and {shard}")
            tensors[t.name] = t
    return tensors


def _natural_key(name: str) -> tuple[tuple[int, int, str], ...]:
    return tuple((0, int(tok), "") if tok.isdigit() else (1, 0, tok) for tok in name.split("."))


def _quant_bits_for(prefix: str, packed_cols: int, groups: int, group_size: int) -> tuple[int, DType]:
    k = groups * group_size
    bits_num = packed_cols * 32
    if k == 0 or bits_num % k != 0:
        raise ValueError(f"cannot infer bit width for {prefix}: packed={packed_cols}, k={k}")
    bits = bits_num // k
    if bits == 4:
        return 4, DType.Q4_AFFINE
    if bits == 3:
        return 3, DType.Q3_AFFINE
    raise ValueError(f"unsupported bit width {bits} for {prefix} (expected 3 or 4)")


def _unquantized_target(dtype_str: str, dtype_mode: str, name: str) -> tuple[DType, str | None]:
    if dtype_str == "F16":
        return (DType.BF16, "f16_to_bf16") if dtype_mode == "bf16" else (DType.F16, None)
    if dtype_str == "BF16":
        return (DType.F16, "bf16_to_f16") if dtype_mode == "f16" else (DType.BF16, None)
    if dtype_str == "F32":
        return DType.F32, None
    if dtype_str == "U32":
        return DType.U32, None
    raise ValueError(f"unsupported unquantized dtype {dtype_str} for {name}")


def _is_d2t_name(norm_name: str) -> bool:
    return norm_name == "d2t" or norm_name.endswith(".d2t")


def _is_t2d_name(norm_name: str) -> bool:
    return norm_name == "t2d" or norm_name.endswith(".t2d")


def _make_entry(
    name: str,
    kind: TensorKind,
    dtype: DType,
    quant_bits: int,
    group_size: int,
    layer: int,
    expert: int,
    shape: tuple[int, ...],
    data: _ByteRange,
    data_convert: str | None,
    scales: _ByteRange | None = None,
    biases: _ByteRange | None = None,
) -> _PlannedEntry:
    name_bytes = name.encode("utf-8")
    if len(name_bytes) == 0 or len(name_bytes) > 47:
        raise ValueError(f"AMF tensor name must be 1..47 UTF-8 bytes, got {len(name_bytes)}: {name!r}")
    if not (1 <= len(shape) <= 4):
        raise ValueError(f"tensor rank must be 1..4 for {name!r}, got {len(shape)}")
    if data_convert == "d2t_i64_to_u32":
        data_bytes = (data.length // 8) * 4
    else:
        data_bytes = data.length  # f16<->bf16, i32->u32 and raw copies preserve byte length
    if scales is not None:
        params_offset = _align_up(data_bytes, 64)
        groups = scales.length // _SAFETENSORS_DTYPE_BYTES[scales.dtype]
        params_bytes = groups * 4  # half2(scale, bias)
        content = params_offset + params_bytes
    else:
        params_offset = 0
        params_bytes = 0
        content = data_bytes
    return _PlannedEntry(
        name=name,
        kind=kind,
        dtype=dtype,
        quant_bits=quant_bits,
        group_size=group_size,
        layer=layer,
        expert=expert,
        shape=shape,
        data=data,
        data_convert=data_convert,
        scales=scales,
        biases=biases,
        data_bytes=data_bytes,
        params_offset=params_offset,
        params_bytes=params_bytes,
        blob_bytes=_align_up(max(content, 1), APPLE_PAGE_SIZE),
    )


def _expert_kind(layer: int, expert: int, cold: set[tuple[int, int]], all_cold: bool) -> TensorKind:
    return TensorKind.COLD_EXPERT if all_cold or (layer, expert) in cold else TensorKind.HOT_EXPERT


def _slice_range(t: _ShardTensor, index: int, count: int) -> _ByteRange:
    step = t.length // count
    return _ByteRange(t.path, t.offset + index * step, step, t.dtype)


def _full_range(t: _ShardTensor) -> _ByteRange:
    return _ByteRange(t.path, t.offset, t.length, t.dtype)


def plan_amf_entries(
    tensors: dict[str, _ShardTensor],
    *,
    group_size: int = 64,
    dtype_mode: str = "auto",
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
    tie_word_embeddings: bool = False,
    exclude_tensors: Iterable[str] = (),
) -> list[_PlannedEntry]:
    """Pass 1: plan every output entry from shard headers alone."""
    if group_size != 64:
        raise ValueError(f"aminfer v0 supports group_size=64 only, got {group_size}")
    if dtype_mode not in DTYPE_MODES:
        raise ValueError(f"dtype_mode must be one of {DTYPE_MODES}, got {dtype_mode!r}")
    cold_set = set(cold_experts)
    excluded: set[str] = set()
    for item in exclude_tensors:
        tok = item.strip()
        if tok:
            excluded.add(tok)
            excluded.add(_normalize_lenient(tok))

    def is_excluded(raw_name: str) -> bool:
        return raw_name in excluded or _normalize_lenient(raw_name) in excluded

    quant_prefixes: set[str] = set()
    for name in tensors:
        if name.endswith(".scales"):
            prefix = name[: -len(".scales")]
            if f"{prefix}.weight" in tensors and f"{prefix}.biases" in tensors:
                quant_prefixes.add(prefix)

    def is_lm_head(name: str) -> bool:
        norm = _normalize_lenient(name)
        return norm in ("lm_head", "lm_head.weight") or norm.startswith("lm_head.")

    embed_rows: int | None = None
    for n, t in tensors.items():
        if _normalize_lenient(n) in ("embed_tokens.weight", "embed_tokens.scales"):
            if not is_excluded(n) and not is_excluded("embed_tokens"):
                embed_rows = t.shape[0] if t.shape else None
                break
    has_embed = embed_rows is not None
    skip_lm_head = tie_word_embeddings and has_embed

    entries: list[_PlannedEntry] = []
    consumed: set[str] = set()

    for prefix in sorted(quant_prefixes):
        w = tensors[f"{prefix}.weight"]
        s = tensors[f"{prefix}.scales"]
        b = tensors[f"{prefix}.biases"]
        consumed.update((w.name, s.name, b.name))
        if is_excluded(prefix) or is_excluded(w.name):
            continue
        if skip_lm_head and is_lm_head(prefix) and (embed_rows is None or w.shape[0] == embed_rows):
            continue
        if w.dtype != "U32":
            raise ValueError(f"expected U32 packed weights for {w.name}, got {w.dtype}")
        for t in (s, b):
            if t.dtype not in ("F16", "BF16", "F32"):
                raise ValueError(f"expected F16/BF16/F32 scales/biases for {t.name}, got {t.dtype}")
        if s.shape != b.shape or s.dtype != b.dtype:
            raise ValueError(f"scales/biases mismatch for {prefix}: {s.dtype}{list(s.shape)} vs {b.dtype}{list(b.shape)}")

        switch = _SWITCH_MLP_RE.match(prefix)
        if len(w.shape) == 3:
            if switch is None:
                raise ValueError(f"3D quantized tensor {prefix} is not an MLX switch_mlp expert stack")
            if len(s.shape) != 3 or w.shape[:2] != s.shape[:2]:
                raise ValueError(f"inconsistent stacked expert shapes for {prefix}: {list(w.shape)}, {list(s.shape)}")
            num_experts, n, packed = w.shape
            groups = s.shape[2]
            bits, amf_dtype = _quant_bits_for(prefix, packed, groups, group_size)
            layer = int(switch.group(1))
            proj = switch.group(2)
            for e in range(num_experts):
                entries.append(
                    _make_entry(
                        f"layers.{layer}.mlp.experts.{e}.{proj}",
                        _expert_kind(layer, e, cold_set, all_experts_cold),
                        amf_dtype,
                        bits,
                        group_size,
                        layer,
                        e,
                        (n, groups * group_size),
                        _slice_range(w, e, num_experts),
                        None,
                        _slice_range(s, e, num_experts),
                        _slice_range(b, e, num_experts),
                    )
                )
            continue

        if len(w.shape) != 2 or len(s.shape) != 2 or w.shape[0] != s.shape[0]:
            raise ValueError(f"inconsistent quantized shapes for {prefix}: {list(w.shape)}, {list(s.shape)}")
        n, packed = w.shape
        groups = s.shape[1]
        bits, amf_dtype = _quant_bits_for(prefix, packed, groups, group_size)
        norm_name = _normalize_name(prefix)
        kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
        entries.append(
            _make_entry(
                norm_name,
                kind,
                amf_dtype,
                bits,
                group_size,
                layer,
                expert,
                (n, groups * group_size),
                _full_range(w),
                None,
                _full_range(s),
                _full_range(b),
            )
        )

    for name in sorted(tensors):
        if name in consumed:
            continue
        t = tensors[name]
        norm_lenient = _normalize_lenient(name)
        if is_excluded(name):
            continue
        # EAGLE-3 training-only target-to-draft boolean mask is omitted at inference.
        if t.dtype == "BOOL" and _is_t2d_name(norm_lenient):
            continue
        if skip_lm_head and is_lm_head(name) and (embed_rows is None or t.shape[0] == embed_rows):
            continue
        if _is_d2t_name(norm_lenient) and t.dtype in ("I64", "I32"):
            if len(t.shape) != 1 or t.shape[0] == 0:
                raise ValueError(f"EAGLE-3 d2t tensor {name!r} must be non-empty 1D, got shape {list(t.shape)}")
            amf_dtype = DType.U32
            convert: str | None = "d2t_i64_to_u32" if t.dtype == "I64" else "d2t_i32_to_u32"
        else:
            amf_dtype, convert = _unquantized_target(t.dtype, dtype_mode, name)
        if name.endswith(".weight"):
            switch = _SWITCH_MLP_RE.match(name[: -len(".weight")])
        else:
            switch = None
        if switch is not None and len(t.shape) == 3:
            num_experts, n, k = t.shape
            layer = int(switch.group(1))
            proj = switch.group(2)
            for e in range(num_experts):
                entries.append(
                    _make_entry(
                        f"layers.{layer}.mlp.experts.{e}.{proj}.weight",
                        _expert_kind(layer, e, cold_set, all_experts_cold),
                        amf_dtype,
                        0,
                        0,
                        layer,
                        e,
                        (n, k),
                        _slice_range(t, e, num_experts),
                        convert,
                    )
                )
            continue
        norm_name = _normalize_name(name)
        kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
        entries.append(
            _make_entry(norm_name, kind, amf_dtype, 0, 0, layer, expert, t.shape, _full_range(t), convert)
        )

    entries.sort(key=lambda e: _natural_key(e.name))
    seen: set[str] = set()
    for e in entries:
        if e.name in seen:
            raise ValueError(f"duplicate output tensor name: {e.name!r}")
        seen.add(e.name)
    return entries


def _normalize_lenient(raw_name: str) -> str:
    return raw_name[6:] if raw_name.startswith("model.") else raw_name


class _ShardReaders:
    """Lazily opened, F_NOCACHE read handles for each shard file."""

    def __init__(self) -> None:
        self._handles: dict[pathlib.Path, object] = {}

    def get(self, path: pathlib.Path):
        fh = self._handles.get(path)
        if fh is None:
            fh = path.open("rb", buffering=0)
            _set_nocache(fh.fileno())
            self._handles[path] = fh
        return fh

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()  # type: ignore[attr-defined]
        self._handles.clear()


def _set_nocache(fd: int) -> None:
    try:
        fcntl.fcntl(fd, _F_NOCACHE, 1)
    except OSError as exc:
        print(f"aminfer-tools: warning: F_NOCACHE unavailable ({exc}); using the page cache", file=sys.stderr)


def _read_exact(fh, offset: int, length: int) -> bytes:
    data = os.pread(fh.fileno(), length, offset)
    if len(data) != length:
        # pread may return short reads for very large requests; finish the rest.
        parts = [data]
        got = len(data)
        while got < length:
            more = os.pread(fh.fileno(), length - got, offset + got)
            if not more:
                raise ValueError(f"unexpected EOF reading {length} bytes at {offset} from {fh.name}")
            parts.append(more)
            got += len(more)
        data = b"".join(parts)
    return data


class _BlobWriter:
    def __init__(self, out) -> None:
        self.out = out
        self.crc = 0
        self.written = 0

    def write(self, chunk: bytes | bytearray | memoryview) -> None:
        self.out.write(chunk)
        self.crc = zlib.crc32(chunk, self.crc)
        self.written += len(chunk)

    def zeros(self, count: int) -> None:
        while count > 0:
            n = min(count, len(_ZERO_CHUNK))
            self.write(_ZERO_CHUNK[:n] if n < len(_ZERO_CHUNK) else _ZERO_CHUNK)
            count -= n


def _to_f16_chunk(raw: bytes, dtype: str, what: str) -> bytes:
    if dtype == "F16":
        return raw
    if dtype == "BF16":
        return bf16_bytes_to_f16_bytes(raw, what=what)
    if dtype == "F32":
        return _f32_bytes_to_f16_bytes(raw, what)
    raise ValueError(f"{what}: unsupported scale/bias dtype {dtype}")


def _stream_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    src = readers.get(entry.data.path)
    if entry.data_convert is None:
        done = 0
        while done < entry.data.length:
            take = min(_COPY_CHUNK_BYTES, entry.data.length - done)
            w.write(_read_exact(src, entry.data.offset + done, take))
            done += take
    elif entry.data_convert in ("d2t_i64_to_u32", "d2t_i32_to_u32"):
        elem_bytes = 8 if entry.data_convert == "d2t_i64_to_u32" else 4
        step = _CONVERT_CHUNK_ELEMS * elem_bytes
        done = 0
        elem_idx = 0
        while done < entry.data.length:
            take = min(step, entry.data.length - done)
            raw = _read_exact(src, entry.data.offset + done, take)
            w.write(
                d2t_deltas_to_u32_bytes(
                    raw,
                    elem_bytes=elem_bytes,
                    start_index=elem_idx,
                    what=entry.name,
                )
            )
            done += take
            elem_idx += take // elem_bytes
    else:
        step = _CONVERT_CHUNK_ELEMS * 2
        done = 0
        while done < entry.data.length:
            take = min(step, entry.data.length - done)
            raw = _read_exact(src, entry.data.offset + done, take)
            if entry.data_convert == "bf16_to_f16":
                w.write(bf16_bytes_to_f16_bytes(raw, what=entry.name))
            elif entry.data_convert == "f16_to_bf16":
                w.write(f16_bytes_to_bf16_bytes(raw))
            else:
                raise ValueError(f"unknown conversion {entry.data_convert!r}")
            done += take

    if entry.scales is not None and entry.biases is not None:
        w.zeros(entry.params_offset - entry.data_bytes)
        elem = _SAFETENSORS_DTYPE_BYTES[entry.scales.dtype]
        total = entry.scales.length // elem
        s_fh = readers.get(entry.scales.path)
        b_fh = readers.get(entry.biases.path)
        done = 0
        while done < total:
            count = min(_CONVERT_CHUNK_ELEMS, total - done)
            s_raw = _read_exact(s_fh, entry.scales.offset + done * elem, count * elem)
            b_raw = _read_exact(b_fh, entry.biases.offset + done * elem, count * elem)
            w.write(
                interleave_scale_bias_f16(
                    _to_f16_chunk(s_raw, entry.scales.dtype, f"{entry.name} scales"),
                    _to_f16_chunk(b_raw, entry.biases.dtype, f"{entry.name} biases"),
                )
            )
            done += count
        content = entry.params_offset + entry.params_bytes
    else:
        content = entry.data_bytes
    w.zeros(entry.blob_bytes - content)


def _load_tie_flag(config_path: pathlib.Path | None) -> bool:
    if config_path is None:
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return bool(config.get("tie_word_embeddings", False)) if isinstance(config, dict) else False


def pack_amf_from_safetensors(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str,
    group_size: int = 64,
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
    dtype_mode: str = "auto",
    tie_word_embeddings: bool | None = None,
    exclude_tensors: Iterable[str] = (),
) -> list[PackedTensorMeta]:
    """Convert a safetensors checkpoint into a 16 KiB-aligned `.amf` v0 file.

    `input_path` may be a `.safetensors` file, a `model.safetensors.index.json`
    file, or a checkpoint directory. `dtype_mode` is `auto` (preserve unquantized
    F16 / BF16 / F32 / U32 bit-exact), `f16` (convert unquantized BF16 to F16),
    or `bf16` (convert unquantized F16 to BF16). EAGLE-3 `d2t` signed offset
    tables (`I64` / `I32`, 1D) are converted to direct `u32` target token IDs
    (`i + d2t[i]`), and training-only `BOOL` `t2d` masks are skipped. MLX affine
    `.weight/.scales/.biases` triples always pack as `q3_affine` / `q4_affine`
    with `half2(scale, bias)` parameters. `tie_word_embeddings=None` reads
    `config.json` next to the input; when tied (and shapes match), a duplicate
    `lm_head` is not emitted.
    """
    model_id_bytes = model_id.encode("utf-8")
    if len(model_id_bytes) == 0 or len(model_id_bytes) > 63:
        raise ValueError(f"model_id must be 1..63 UTF-8 bytes: {model_id!r}")

    resolved = resolve_checkpoint(pathlib.Path(input_path))
    tie = _load_tie_flag(resolved.config_path) if tie_word_embeddings is None else tie_word_embeddings
    tensors = _collect_tensors(resolved.shard_paths)
    plan = plan_amf_entries(
        tensors,
        group_size=group_size,
        dtype_mode=dtype_mode,
        cold_experts=cold_experts,
        all_experts_cold=all_experts_cold,
        tie_word_embeddings=tie,
        exclude_tensors=exclude_tensors,
    )

    tensor_count = len(plan)
    dir_offset = APPLE_PAGE_SIZE
    dir_raw_len = tensor_count * ENTRY_STRUCT.size
    dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)
    offsets: list[int] = []
    cursor = dir_offset + dir_bytes
    for e in plan:
        offsets.append(cursor)
        cursor += e.blob_bytes

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    readers = _ShardReaders()
    metas: list[PackedTensorMeta] = []
    try:
        with partial.open("wb") as out:
            _set_nocache(out.fileno())
            prefix = _BlobWriter(out)
            prefix.zeros(dir_offset + dir_bytes)
            for e, blob_offset in zip(plan, offsets):
                if out.tell() != blob_offset:
                    raise AssertionError(f"planned offset {blob_offset} != file position {out.tell()} for {e.name}")
                w = _BlobWriter(out)
                _stream_entry(e, readers, w)
                if w.written != e.blob_bytes:
                    raise AssertionError(f"wrote {w.written} bytes for {e.name}, planned {e.blob_bytes}")
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
                    )
                )
            if out.tell() != cursor:
                raise AssertionError(f"final size {out.tell()} != planned {cursor}")

            dir_raw = _pack_directory(metas)
            header = _pack_header(model_id_bytes, tensor_count, dir_offset, dir_bytes, _crc32(dir_raw))
            out.seek(0)
            out.write(header)
            if dir_raw:
                out.seek(dir_offset)
                out.write(dir_raw)
            out.flush()
            os.fsync(out.fileno())
        os.replace(partial, output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        readers.close()
    return metas


def _pack_directory(metas: Sequence[PackedTensorMeta]) -> bytes:
    dir_raw = bytearray(len(metas) * ENTRY_STRUCT.size)
    for i, m in enumerate(metas):
        padded_shape = [m.shape[d] if d < len(m.shape) else 0 for d in range(4)]
        ENTRY_STRUCT.pack_into(
            dir_raw,
            i * ENTRY_STRUCT.size,
            m.name.encode("utf-8"),
            int(m.kind),
            int(m.dtype),
            m.quant_bits,
            len(m.shape),
            m.group_size,
            m.layer,
            m.expert,
            0,
            m.blob_crc32,
            *padded_shape,
            m.blob_offset,
            m.blob_bytes,
            m.data_bytes,
            m.params_offset,
            m.params_bytes,
            0,
        )
    return bytes(dir_raw)


def _pack_header(model_id_bytes: bytes, tensor_count: int, dir_offset: int, dir_bytes: int, dir_crc: int) -> bytes:
    fields = [
        AMF_MAGIC,
        AMF_VERSION,
        0,
        APPLE_PAGE_SIZE,
        CHECKSUM_CRC32_IEEE,
        tensor_count,
        0,
        dir_offset,
        dir_bytes,
        dir_crc,
        0,
        model_id_bytes,
        0,
    ]
    fields[10] = _crc32(HEADER_STRUCT.pack(*fields))
    return HEADER_STRUCT.pack(*fields)
