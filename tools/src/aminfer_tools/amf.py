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


class TensorLayout(enum.IntEnum):
    """Single-owner tensor byte layout in low byte of TensorEntry.reserved0 (ADR-0018, W15)."""

    ROW_MAJOR = 0
    SME2_PANEL64 = 1


class QuantParamKind(enum.IntEnum):
    """Quantization parameter encoding in high byte of TensorEntry.reserved0 (ADR-0017, ADR-0018)."""

    AFFINE_SCALE_BIAS = 0
    LUT16_CODEBOOK = 1


SME2_PANEL_COLS = 16
SME2_GROUP_K = 64
SME2_QUANT_HEADER_BYTES = 128
SME2_F16_GROUP_BYTES = 2048  # 32 kp chunks * 32 u16 lanes
SME2_Q4_GROUP_BYTES = 640  # 128 B inline header + 512 B packed codes
SME2_Q3_GROUP_BYTES = 512  # 128 B inline header + 384 B packed codes
SME2_Q2_GROUP_BYTES = 384  # 128 B inline header + 256 B packed codes
SME2_PACK_MODES = ("none", "routers", "all-linear")
SME2_REPACK_TARGETS = ("routers", "all-linear")


def encode_reserved0(
    layout: TensorLayout = TensorLayout.ROW_MAJOR,
    param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS,
) -> int:
    """Pack `(layout, param_kind)` into the 16-bit `TensorEntry.reserved0` field."""
    return (int(layout) & 0xFF) | ((int(param_kind) & 0xFF) << 8)


def decode_reserved0(reserved0: int) -> tuple[TensorLayout, QuantParamKind]:
    """Decode `TensorEntry.reserved0` into `(TensorLayout, QuantParamKind)`."""
    layout_raw = reserved0 & 0xFF
    param_raw = (reserved0 >> 8) & 0xFF
    try:
        layout = TensorLayout(layout_raw)
    except ValueError as exc:
        raise ValueError(f"invalid AMF TensorLayout {layout_raw} in reserved0={reserved0:#06x}") from exc
    try:
        param_kind = QuantParamKind(param_raw)
    except ValueError as exc:
        raise ValueError(f"invalid AMF QuantParamKind {param_raw} in reserved0={reserved0:#06x}") from exc
    return layout, param_kind


def sme2_panel64_data_bytes(dtype: DType, n: int, k: int, group_size: int = 0) -> int:
    """Return exact packed `data_bytes` for a 2D `[n, k]` `sme2_panel64` tensor (W15)."""
    if n <= 0 or k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"invalid sme2_panel64 shape ({n}, {k}): require n>0, k>0, k%64==0")
    if dtype in (DType.F16, DType.BF16):
        if group_size != 0:
            raise ValueError(f"unquantized sme2_panel64 {dtype.name} requires group_size=0, got {group_size}")
        group_bytes = SME2_F16_GROUP_BYTES
    elif dtype == DType.Q4_AFFINE:
        if group_size != SME2_GROUP_K:
            raise ValueError(f"q4_affine sme2_panel64 requires group_size=64, got {group_size}")
        group_bytes = SME2_Q4_GROUP_BYTES
    elif dtype == DType.Q3_AFFINE:
        if group_size != SME2_GROUP_K:
            raise ValueError(f"q3_affine sme2_panel64 requires group_size=64, got {group_size}")
        group_bytes = SME2_Q3_GROUP_BYTES
    else:
        raise ValueError(f"dtype {dtype.name} is not supported in sme2_panel64 layout")
    panels = (n + SME2_PANEL_COLS - 1) // SME2_PANEL_COLS
    groups = k // SME2_GROUP_K
    return panels * groups * group_bytes


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
    layout: TensorLayout = TensorLayout.ROW_MAJOR
    quant_param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS


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
    records: Sequence[tuple],
) -> list[PackedTensorMeta]:
    """Write a validated, 16 KiB-aligned .amf v0 file.

    Each record is either a 10-tuple:
      (name, kind, dtype, quant_bits, group_size, layer, expert, shape, data_bytes, params_bytes)
    or a 12-tuple appending `(layout, quant_param_kind)`.
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

    for rec in records:
        if len(rec) == 10:
            name, kind, dtype, quant_bits, group_size, layer, expert, shape, data, params = rec
            layout = TensorLayout.ROW_MAJOR
            param_kind = QuantParamKind.AFFINE_SCALE_BIAS
        elif len(rec) == 12:
            name, kind, dtype, quant_bits, group_size, layer, expert, shape, data, params, layout, param_kind = rec
        else:
            raise ValueError(f"write_amf record must have 10 or 12 fields, got {len(rec)}")
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
                layout=layout,
                quant_param_kind=param_kind,
            )
        )
        blobs.append(bytes(blob))
        cursor += blob_bytes

    dir_raw = _pack_directory(metas)
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
    layout: TensorLayout = TensorLayout.ROW_MAJOR
    quant_param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS


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


def _should_sme2_pack(
    kind: TensorKind,
    dtype: DType,
    shape: tuple[int, ...],
    sme2_pack: str,
) -> bool:
    if sme2_pack == "none":
        return False
    if sme2_pack not in SME2_PACK_MODES:
        raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 or shape[1] % SME2_GROUP_K != 0:
        return False
    if dtype not in (DType.F16, DType.BF16, DType.Q4_AFFINE, DType.Q3_AFFINE):
        return False
    if sme2_pack == "routers":
        return kind == TensorKind.ROUTER
    return kind in (
        TensorKind.ROUTER,
        TensorKind.DENSE,
        TensorKind.HOT_EXPERT,
        TensorKind.COLD_EXPERT,
        TensorKind.LM_HEAD,
    )


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
    *,
    sme2_pack: str = "none",
) -> _PlannedEntry:
    name_bytes = name.encode("utf-8")
    if len(name_bytes) == 0 or len(name_bytes) > 47:
        raise ValueError(f"AMF tensor name must be 1..47 UTF-8 bytes, got {len(name_bytes)}: {name!r}")
    if not (1 <= len(shape) <= 4):
        raise ValueError(f"tensor rank must be 1..4 for {name!r}, got {len(shape)}")
    use_sme2 = _should_sme2_pack(kind, dtype, shape, sme2_pack)
    if use_sme2:
        layout = TensorLayout.SME2_PANEL64
        param_kind = QuantParamKind.AFFINE_SCALE_BIAS
        data_bytes = sme2_panel64_data_bytes(dtype, shape[0], shape[1], group_size)
        params_offset = 0
        params_bytes = 0
        content = data_bytes
    else:
        layout = TensorLayout.ROW_MAJOR
        param_kind = QuantParamKind.AFFINE_SCALE_BIAS
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
        layout=layout,
        quant_param_kind=param_kind,
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
    sme2_pack: str = "none",
) -> list[_PlannedEntry]:
    """Pass 1: plan every output entry from shard headers alone."""
    if group_size != 64:
        raise ValueError(f"aminfer v0 supports group_size=64 only, got {group_size}")
    if dtype_mode not in DTYPE_MODES:
        raise ValueError(f"dtype_mode must be one of {DTYPE_MODES}, got {dtype_mode!r}")
    if sme2_pack not in SME2_PACK_MODES:
        raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
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
                        sme2_pack=sme2_pack,
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
                sme2_pack=sme2_pack,
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
                        sme2_pack=sme2_pack,
                    )
                )
            continue
        norm_name = _normalize_name(name)
        kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
        entries.append(
            _make_entry(
                norm_name,
                kind,
                amf_dtype,
                0,
                0,
                layer,
                expert,
                t.shape,
                _full_range(t),
                convert,
                sme2_pack=sme2_pack,
            )
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


def _stream_sme2_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    """Stream 16 rows at a time from shard(s) and emit `sme2_panel64` panels."""
    n_rows, k = entry.shape
    panels = (n_rows + SME2_PANEL_COLS - 1) // SME2_PANEL_COLS
    src = readers.get(entry.data.path)

    if entry.dtype in (DType.F16, DType.BF16):
        row_bytes = k * 2
        for p in range(panels):
            r0 = p * SME2_PANEL_COLS
            count = min(SME2_PANEL_COLS, n_rows - r0)
            raw = _read_exact(src, entry.data.offset + r0 * row_bytes, count * row_bytes)
            if entry.data_convert == "bf16_to_f16":
                raw = bf16_bytes_to_f16_bytes(raw, what=entry.name)
            elif entry.data_convert == "f16_to_bf16":
                raw = f16_bytes_to_bf16_bytes(raw)
            elif entry.data_convert is not None:
                raise ValueError(f"unknown conversion {entry.data_convert!r}")
            mv = memoryview(raw)
            rows = [mv[c * row_bytes : (c + 1) * row_bytes] for c in range(count)]
            w.write(pack_sme2_panel64_f16_panel(rows, k))
    elif entry.dtype in (DType.Q4_AFFINE, DType.Q3_AFFINE):
        if entry.scales is None or entry.biases is None:
            raise ValueError(f"quantized sme2_panel64 tensor {entry.name!r} is missing scales/biases")
        groups = k // entry.group_size
        row_data_bytes = k // 2 if entry.dtype == DType.Q4_AFFINE else (k * 3) // 8
        sb_elem = _SAFETENSORS_DTYPE_BYTES[entry.scales.dtype]
        row_sb_src_bytes = groups * sb_elem
        row_param_bytes = groups * 4
        s_fh = readers.get(entry.scales.path)
        b_fh = readers.get(entry.biases.path)
        packer = pack_sme2_panel64_q4_panel if entry.dtype == DType.Q4_AFFINE else pack_sme2_panel64_q3_panel
        for p in range(panels):
            r0 = p * SME2_PANEL_COLS
            count = min(SME2_PANEL_COLS, n_rows - r0)
            raw_d = _read_exact(src, entry.data.offset + r0 * row_data_bytes, count * row_data_bytes)
            raw_s = _read_exact(s_fh, entry.scales.offset + r0 * row_sb_src_bytes, count * row_sb_src_bytes)
            raw_b = _read_exact(b_fh, entry.biases.offset + r0 * row_sb_src_bytes, count * row_sb_src_bytes)
            params_block = interleave_scale_bias_f16(
                _to_f16_chunk(raw_s, entry.scales.dtype, f"{entry.name} scales"),
                _to_f16_chunk(raw_b, entry.biases.dtype, f"{entry.name} biases"),
            )
            d_mv = memoryview(raw_d)
            p_mv = memoryview(params_block)
            d_rows = [d_mv[c * row_data_bytes : (c + 1) * row_data_bytes] for c in range(count)]
            p_rows = [p_mv[c * row_param_bytes : (c + 1) * row_param_bytes] for c in range(count)]
            w.write(packer(d_rows, p_rows, k))
    else:
        raise ValueError(f"unsupported sme2_panel64 dtype {entry.dtype.name} for {entry.name!r}")

    w.zeros(entry.blob_bytes - entry.data_bytes)


def _stream_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    if entry.layout == TensorLayout.SME2_PANEL64:
        _stream_sme2_entry(entry, readers, w)
        return
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
    heat_profile: pathlib.Path | None = None,
    cold_experts_per_layer: int | None = None,
    sme2_pack: str = "none",
) -> list[PackedTensorMeta]:
    """Convert a safetensors checkpoint into a 16 KiB-aligned `.amf` v0 file.

    `input_path` may be a `.safetensors` file, a `model.safetensors.index.json`
    file, or a checkpoint directory. `dtype_mode` is `auto` (preserve unquantized
    F16 / BF16 / F32 / U32 bit-exact), `f16` (convert unquantized BF16 to F16),
    or `bf16` (convert unquantized F16 to BF16). EAGLE-3 `d2t` signed offset
    tables (`I64` / `I32`, 1D) are converted to direct `u32` target token IDs
    (`i + d2t[i]`), and training-only `BOOL` `t2d` masks are skipped. MLX affine
    `.weight/.scales/.biases` triples always pack as `q3_affine` / `q4_affine`
    with `half2(scale, bias)` parameters (or inline 128 B headers when packed
    with `sme2_pack != 'none'`). `tie_word_embeddings=None` reads
    `config.json` next to the input; when tied (and shapes match), a duplicate
    `lm_head` is not emitted. When `heat_profile` is given, its cold-expert set
    (optionally overridden by `cold_experts_per_layer`) is unioned with
    `cold_experts`. `sme2_pack` selects single-owner `sme2_panel64` pre-packing
    (`none`, `routers`, or `all-linear`; ADR-0018, W15).
    """
    model_id_bytes = model_id.encode("utf-8")
    if len(model_id_bytes) == 0 or len(model_id_bytes) > 63:
        raise ValueError(f"model_id must be 1..63 UTF-8 bytes: {model_id!r}")

    effective_cold: set[tuple[int, int]] = set(cold_experts)
    if heat_profile is not None:
        effective_cold.update(
            load_cold_experts_from_heat_profile(
                pathlib.Path(heat_profile),
                cold_experts_per_layer=cold_experts_per_layer,
            )
        )
    elif cold_experts_per_layer is not None:
        raise ValueError("cold_experts_per_layer requires heat_profile in pack_amf_from_safetensors")

    resolved = resolve_checkpoint(pathlib.Path(input_path))
    tie = _load_tie_flag(resolved.config_path) if tie_word_embeddings is None else tie_word_embeddings
    tensors = _collect_tensors(resolved.shard_paths)
    plan = plan_amf_entries(
        tensors,
        group_size=group_size,
        dtype_mode=dtype_mode,
        cold_experts=effective_cold,
        all_experts_cold=all_experts_cold,
        tie_word_embeddings=tie,
        exclude_tensors=exclude_tensors,
        sme2_pack=sme2_pack,
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
                        layout=e.layout,
                        quant_param_kind=e.quant_param_kind,
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
            encode_reserved0(m.layout, m.quant_param_kind),
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


# ---------------------------------------------------------------------------
# Streaming .amf v0 reader, pure-stdlib dequantizers, and cold-expert retagging
# ---------------------------------------------------------------------------

# 12-bit lookup table mapping 4 packed 3-bit unsigned codes -> (c0, c1, c2, c3).
_Q3_LUT12: tuple[tuple[int, int, int, int], ...] = tuple(
    (v & 0x07, (v >> 3) & 0x07, (v >> 6) & 0x07, (v >> 9) & 0x07) for v in range(4096)
)

# 12-bit lookup table mapping 4 packed 3-bit codes -> 2 q4-nibble bytes
# `(c0 | (c1 << 4), c2 | (c3 << 4))` for direct sme2_panel64 q3 quad assembly.
_Q3_12BIT_TO_Q4_PAIR: tuple[tuple[int, int], ...] = tuple(
    (
        (v & 0x07) | (((v >> 3) & 0x07) << 4),
        ((v >> 6) & 0x07) | (((v >> 9) & 0x07) << 4),
    )
    for v in range(4096)
)

# 8-bit lookup table mapping 2 packed 4-bit unsigned codes -> (lo, hi).
_Q4_LUT8: tuple[tuple[int, int], ...] = tuple((b & 0x0F, b >> 4) for b in range(256))


# ---------------------------------------------------------------------------
# ARMv9 SME2 `sme2_panel64` packers and reference unpackers (ADR-0018, W15)
# ---------------------------------------------------------------------------


def pack_sme2_lut16_header(
    codebook_f16: bytes | bytearray | memoryview,
    bias_dup_f16: bytes | bytearray | memoryview | None = None,
) -> bytes:
    """Build a 128 B `QuantParamKind.LUT16_CODEBOOK` inline `(panel, group)` header.

    Hardware M4 `luti4 .h` / `luti2 .h` index `ZT0` at 32-bit stride (`zt0_u16[2*idx]`),
    so 16 `f16` codebook entries (32 B) are duplicated into 64 B `[v0, v0, ..., v15, v15]`
    at bytes `0..64`, followed by 64 B `bias_dup_f16` (zeros when omitted) at `64..128`.
    """
    cb = memoryview(codebook_f16)
    if len(cb) != 32:
        raise ValueError(f"lut16 codebook must be 32 bytes (16 f16 entries), got {len(cb)}")
    out = bytearray(SME2_QUANT_HEADER_BYTES)
    for i in range(16):
        lo = cb[i * 2]
        hi = cb[i * 2 + 1]
        out[i * 4 + 0] = lo
        out[i * 4 + 1] = hi
        out[i * 4 + 2] = lo
        out[i * 4 + 3] = hi
    if bias_dup_f16 is not None:
        b_mv = memoryview(bias_dup_f16)
        if len(b_mv) != 64:
            raise ValueError(f"bias_dup_f16 must be 64 bytes (32 f16 entries), got {len(b_mv)}")
        out[64:128] = b_mv
    return bytes(out)


def _write_sme2_quant_group_header(
    out: bytearray,
    g_off: int,
    g: int,
    params_mvs: Sequence[memoryview] | None,
    param_kind: QuantParamKind,
    group_headers: Sequence[bytes | bytearray | memoryview] | None,
) -> None:
    if param_kind == QuantParamKind.LUT16_CODEBOOK:
        if group_headers is None:
            raise ValueError("group_headers is required when param_kind is LUT16_CODEBOOK")
        hdr = memoryview(group_headers[g])
        if len(hdr) != SME2_QUANT_HEADER_BYTES:
            raise ValueError(f"expected 128 B inline header for group {g}, got {len(hdr)}")
        out[g_off : g_off + SME2_QUANT_HEADER_BYTES] = hdr
        return
    if params_mvs is None:
        raise ValueError("panel_params_rows is required when param_kind is AFFINE_SCALE_BIAS")
    sb_off = g * 4
    for c, p_row in enumerate(params_mvs):
        s0 = p_row[sb_off]
        s1 = p_row[sb_off + 1]
        b0 = p_row[sb_off + 2]
        b1 = p_row[sb_off + 3]
        s_dst = g_off + c * 4
        b_dst = g_off + 64 + c * 4
        out[s_dst + 0] = s0
        out[s_dst + 1] = s1
        out[s_dst + 2] = s0
        out[s_dst + 3] = s1
        out[b_dst + 0] = b0
        out[b_dst + 1] = b1
        out[b_dst + 2] = b0
        out[b_dst + 3] = b1


def pack_sme2_panel64_f16_panel(
    panel_rows: Sequence[bytes | bytearray | memoryview],
    k: int,
) -> bytes:
    """Pack 1..16 rows of `[N_panel, K]` `bf16`/`f16` weights into one `sme2_panel64` panel.

    Produces `(K // 64) * 2048` bytes: for each group and `kp in 0..32`, 32 `u16` lanes
    `[W[col_0, k0], W[col_0, k1], ..., W[col_15, k0], W[col_15, k1]]`. Rows beyond
    `len(panel_rows)` are zero-padded.
    """
    if not (1 <= len(panel_rows) <= SME2_PANEL_COLS):
        raise ValueError(f"panel_rows must contain 1..16 rows, got {len(panel_rows)}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    groups = k // SME2_GROUP_K
    row_bytes = k * 2
    out = bytearray(groups * SME2_F16_GROUP_BYTES)
    out_u32 = memoryview(out).cast("I")
    try:
        for c, row in enumerate(panel_rows):
            r_mv = memoryview(row)
            if len(r_mv) != row_bytes:
                raise ValueError(f"panel row {c} has {len(r_mv)} bytes, expected {row_bytes}")
            out_u32[c::SME2_PANEL_COLS] = r_mv.cast("I")
    finally:
        out_u32.release()
    return bytes(out)


def pack_sme2_panel64_q4_panel(
    panel_data_rows: Sequence[bytes | bytearray | memoryview],
    panel_params_rows: Sequence[bytes | bytearray | memoryview] | None,
    k: int,
    *,
    param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS,
    group_headers: Sequence[bytes | bytearray | memoryview] | None = None,
) -> bytes:
    """Pack 1..16 MLX-format `q4_affine` rows into one `sme2_panel64` panel (`640 B`/group).

    Per `(panel, group)`: 128 B inline header (`scale_dup_f16[32]` + `bias_dup_f16[32]`,
    or `lut16_codebook` header) followed by 512 B codes where for `kp in 0..32`, `c in 0..16`:
    `codes[kp*16 + c] = (code[col_c, k0] & 0xF) | ((code[col_c, k1] & 0xF) << 4)`.
    """
    n_active = len(panel_data_rows)
    if not (1 <= n_active <= SME2_PANEL_COLS):
        raise ValueError(f"panel_data_rows must contain 1..16 rows, got {n_active}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    groups = k // SME2_GROUP_K
    row_data_bytes = k // 2
    row_param_bytes = groups * 4

    data_mvs = [memoryview(r) for r in panel_data_rows]
    for c, mv in enumerate(data_mvs):
        if len(mv) != row_data_bytes:
            raise ValueError(f"q4 panel row {c} has {len(mv)} data bytes, expected {row_data_bytes}")

    params_mvs: list[memoryview] | None = None
    if param_kind == QuantParamKind.AFFINE_SCALE_BIAS:
        if panel_params_rows is None or len(panel_params_rows) != n_active:
            raise ValueError("panel_params_rows must match panel_data_rows length")
        params_mvs = [memoryview(r) for r in panel_params_rows]
        for c, mv in enumerate(params_mvs):
            if len(mv) != row_param_bytes:
                raise ValueError(f"q4 panel row {c} has {len(mv)} param bytes, expected {row_param_bytes}")

    out = bytearray(groups * SME2_Q4_GROUP_BYTES)
    for g in range(groups):
        g_off = g * SME2_Q4_GROUP_BYTES
        _write_sme2_quant_group_header(out, g_off, g, params_mvs, param_kind, group_headers)
        code_off = g_off + SME2_QUANT_HEADER_BYTES
        src_start = g * 32
        src_end = src_start + 32
        for c, rd in enumerate(data_mvs):
            out[code_off + c : code_off + 512 : SME2_PANEL_COLS] = rd[src_start:src_end]
    return bytes(out)


def pack_sme2_panel64_q3_panel(
    panel_data_rows: Sequence[bytes | bytearray | memoryview],
    panel_params_rows: Sequence[bytes | bytearray | memoryview] | None,
    k: int,
    *,
    param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS,
    group_headers: Sequence[bytes | bytearray | memoryview] | None = None,
) -> bytes:
    """Pack 1..16 MLX-format `q3_affine` rows into one `sme2_panel64` panel (`512 B`/group).

    Per `(panel, group)`: 128 B inline header followed by 384 B codes (`2` halves × `3 × 64 B`).
    In each half (`kp_base = h * 16`), 4 `q4`-format 64 B quads `Q[0..3]` (3-bit codes in
    bits `0..2` and `4..6`) are packed into 3 × 64 B planes (`B0`, `B1`, `B2`) by embedding
    `Q[3]`'s 6 payload bits into the spare bits `3` and `7` of `Q[0..2]`.
    """
    n_active = len(panel_data_rows)
    if not (1 <= n_active <= SME2_PANEL_COLS):
        raise ValueError(f"panel_data_rows must contain 1..16 rows, got {n_active}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    groups = k // SME2_GROUP_K
    row_data_bytes = (k * 3) // 8
    row_param_bytes = groups * 4

    data_mvs = [memoryview(r) for r in panel_data_rows]
    for c, mv in enumerate(data_mvs):
        if len(mv) != row_data_bytes:
            raise ValueError(f"q3 panel row {c} has {len(mv)} data bytes, expected {row_data_bytes}")

    params_mvs: list[memoryview] | None = None
    if param_kind == QuantParamKind.AFFINE_SCALE_BIAS:
        if panel_params_rows is None or len(panel_params_rows) != n_active:
            raise ValueError("panel_params_rows must match panel_data_rows length")
        params_mvs = [memoryview(r) for r in panel_params_rows]
        for c, mv in enumerate(params_mvs):
            if len(mv) != row_param_bytes:
                raise ValueError(f"q3 panel row {c} has {len(mv)} param bytes, expected {row_param_bytes}")

    out = bytearray(groups * SME2_Q3_GROUP_BYTES)
    q3_lut = _Q3_12BIT_TO_Q4_PAIR
    for g in range(groups):
        g_off = g * SME2_Q3_GROUP_BYTES
        _write_sme2_quant_group_header(out, g_off, g, params_mvs, param_kind, group_headers)
        code_off = g_off + SME2_QUANT_HEADER_BYTES
        g_src = g * 24
        for h in range(2):
            q0 = bytearray(64)
            q1 = bytearray(64)
            q2 = bytearray(64)
            q3 = bytearray(64)
            quads = (q0, q1, q2, q3)
            h_blk = h * 4
            for c, rd in enumerate(data_mvs):
                for q_idx in range(4):
                    off = g_src + (h_blk + q_idx) * 3
                    word24 = rd[off] | (rd[off + 1] << 8) | (rd[off + 2] << 16)
                    p0, p1 = q3_lut[word24 & 0xFFF]
                    p2, p3 = q3_lut[word24 >> 12]
                    dst = quads[q_idx]
                    dst[c] = p0
                    dst[16 + c] = p1
                    dst[32 + c] = p2
                    dst[48 + c] = p3
            half_off = code_off + h * 192
            for i in range(64):
                v3 = q3[i]
                out[half_off + i] = q0[i] | ((v3 & 0x01) << 3) | ((v3 & 0x10) << 3)
                out[half_off + 64 + i] = q1[i] | ((v3 & 0x02) << 2) | ((v3 & 0x20) << 2)
                out[half_off + 128 + i] = q2[i] | ((v3 & 0x04) << 1) | ((v3 & 0x40) << 1)
    return bytes(out)


def pack_sme2_panel64_q2_panel(
    panel_codes_rows: Sequence[Sequence[int]],
    panel_params_rows: Sequence[bytes | bytearray | memoryview] | None,
    k: int,
    *,
    param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS,
    group_headers: Sequence[bytes | bytearray | memoryview] | None = None,
) -> bytes:
    """Pack 1..16 rows of 2-bit codes (`0..3`) into one `sme2_panel64` `q2` panel (`384 B`/group).

    Matches M4 `luti2 {z0.h-z3.h}, zt0, Zn[0..1]` lane order: for `kp in 0..32`, `b in 0..8`
    (`c0 = 2*b`, `c1 = 2*b + 1`),
    `codes[kp*8 + b] = c0_k0 | (c0_k1 << 2) | (c1_k0 << 4) | (c1_k1 << 6)`.
    """
    n_active = len(panel_codes_rows)
    if not (1 <= n_active <= SME2_PANEL_COLS):
        raise ValueError(f"panel_codes_rows must contain 1..16 rows, got {n_active}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    for c, row in enumerate(panel_codes_rows):
        if len(row) != k:
            raise ValueError(f"q2 panel row {c} has {len(row)} codes, expected {k}")
    groups = k // SME2_GROUP_K
    row_param_bytes = groups * 4

    params_mvs: list[memoryview] | None = None
    if param_kind == QuantParamKind.AFFINE_SCALE_BIAS:
        if panel_params_rows is None or len(panel_params_rows) != n_active:
            raise ValueError("panel_params_rows must match panel_codes_rows length")
        params_mvs = [memoryview(r) for r in panel_params_rows]
        for c, mv in enumerate(params_mvs):
            if len(mv) != row_param_bytes:
                raise ValueError(f"q2 panel row {c} has {len(mv)} param bytes, expected {row_param_bytes}")

    out = bytearray(groups * SME2_Q2_GROUP_BYTES)
    for g in range(groups):
        g_off = g * SME2_Q2_GROUP_BYTES
        _write_sme2_quant_group_header(out, g_off, g, params_mvs, param_kind, group_headers)
        code_off = g_off + SME2_QUANT_HEADER_BYTES
        k_base = g * SME2_GROUP_K
        for kp in range(32):
            k0 = k_base + 2 * kp
            k1 = k0 + 1
            row_off = code_off + kp * 8
            for b in range(8):
                c0 = 2 * b
                c1 = c0 + 1
                v00 = (panel_codes_rows[c0][k0] & 0x03) if c0 < n_active else 0
                v01 = (panel_codes_rows[c0][k1] & 0x03) if c0 < n_active else 0
                v10 = (panel_codes_rows[c1][k0] & 0x03) if c1 < n_active else 0
                v11 = (panel_codes_rows[c1][k1] & 0x03) if c1 < n_active else 0
                out[row_off + b] = v00 | (v01 << 2) | (v10 << 4) | (v11 << 6)
    return bytes(out)


def unpack_sme2_panel64_f16_panel(
    panel_bytes: bytes | bytearray | memoryview,
    k: int,
    *,
    num_rows: int = SME2_PANEL_COLS,
) -> list[bytes]:
    """Unpack one `sme2_panel64` `bf16`/`f16` panel back to `num_rows` contiguous rows."""
    if not (1 <= num_rows <= SME2_PANEL_COLS):
        raise ValueError(f"num_rows must be 1..16, got {num_rows}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    groups = k // SME2_GROUP_K
    expected = groups * SME2_F16_GROUP_BYTES
    mv = memoryview(panel_bytes)
    if len(mv) != expected:
        raise ValueError(f"f16/bf16 panel length {len(mv)} != expected {expected}")
    panel_u32 = mv.cast("I")
    rows: list[bytes] = []
    try:
        for c in range(num_rows):
            row_buf = bytearray(k * 2)
            row_u32 = memoryview(row_buf).cast("I")
            try:
                row_u32[:] = panel_u32[c::SME2_PANEL_COLS]
            finally:
                row_u32.release()
            rows.append(bytes(row_buf))
    finally:
        panel_u32.release()
    return rows


def _decode_sme2_header_tables(
    hdr_mv: memoryview,
    param_kind: QuantParamKind,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return `(first_16, bias_16)` from a 128 B inline header.

    For `AFFINE_SCALE_BIAS`, `first_16` is per-column scales `s[0..15]`;
    for `LUT16_CODEBOOK`, `first_16` is the 16-entry codebook `lut[0..15]`.
    `bias_16` is always per-column bias `b[0..15]`.
    """
    raw_first = struct.unpack_from("<32e", hdr_mv, 0)
    raw_bias = struct.unpack_from("<32e", hdr_mv, 64)
    first_16 = tuple(raw_first[2 * i] for i in range(16))
    bias_16 = tuple(raw_bias[2 * i] for i in range(16))
    if param_kind not in (QuantParamKind.AFFINE_SCALE_BIAS, QuantParamKind.LUT16_CODEBOOK):
        raise ValueError(f"unsupported QuantParamKind {param_kind!r}")
    return first_16, bias_16


def dequantize_sme2_panel64_q4_panel_f32(
    panel_bytes: bytes | bytearray | memoryview,
    k: int,
    *,
    num_rows: int = SME2_PANEL_COLS,
    param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS,
) -> list[list[float]]:
    """Dequantize one `sme2_panel64` `q4_affine` panel into `num_rows` lists of `k` floats."""
    if not (1 <= num_rows <= SME2_PANEL_COLS):
        raise ValueError(f"num_rows must be 1..16, got {num_rows}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    groups = k // SME2_GROUP_K
    mv = memoryview(panel_bytes)
    expected = groups * SME2_Q4_GROUP_BYTES
    if len(mv) != expected:
        raise ValueError(f"q4 panel length {len(mv)} != expected {expected}")

    rows = [[0.0] * k for _ in range(num_rows)]
    lut8 = _Q4_LUT8
    for g in range(groups):
        g_off = g * SME2_Q4_GROUP_BYTES
        first_16, bias_16 = _decode_sme2_header_tables(mv[g_off : g_off + SME2_QUANT_HEADER_BYTES], param_kind)
        code_off = g_off + SME2_QUANT_HEADER_BYTES
        k_base = g * SME2_GROUP_K
        if param_kind == QuantParamKind.AFFINE_SCALE_BIAS:
            for c in range(num_rows):
                s = first_16[c]
                b = bias_16[c]
                row = rows[c]
                for kp in range(32):
                    lo, hi = lut8[mv[code_off + kp * SME2_PANEL_COLS + c]]
                    row[k_base + 2 * kp] = b + lo * s
                    row[k_base + 2 * kp + 1] = b + hi * s
        else:
            for c in range(num_rows):
                b = bias_16[c]
                row = rows[c]
                for kp in range(32):
                    lo, hi = lut8[mv[code_off + kp * SME2_PANEL_COLS + c]]
                    row[k_base + 2 * kp] = first_16[lo] + b
                    row[k_base + 2 * kp + 1] = first_16[hi] + b
    return rows


def dequantize_sme2_panel64_q3_panel_f32(
    panel_bytes: bytes | bytearray | memoryview,
    k: int,
    *,
    num_rows: int = SME2_PANEL_COLS,
    param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS,
) -> list[list[float]]:
    """Dequantize one `sme2_panel64` `q3_affine` panel into `num_rows` lists of `k` floats."""
    if not (1 <= num_rows <= SME2_PANEL_COLS):
        raise ValueError(f"num_rows must be 1..16, got {num_rows}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    groups = k // SME2_GROUP_K
    mv = memoryview(panel_bytes)
    expected = groups * SME2_Q3_GROUP_BYTES
    if len(mv) != expected:
        raise ValueError(f"q3 panel length {len(mv)} != expected {expected}")

    rows = [[0.0] * k for _ in range(num_rows)]
    for g in range(groups):
        g_off = g * SME2_Q3_GROUP_BYTES
        first_16, bias_16 = _decode_sme2_header_tables(mv[g_off : g_off + SME2_QUANT_HEADER_BYTES], param_kind)
        code_off = g_off + SME2_QUANT_HEADER_BYTES
        k_base = g * SME2_GROUP_K
        for h in range(2):
            half_off = code_off + h * 192
            b0 = mv[half_off : half_off + 64]
            b1 = mv[half_off + 64 : half_off + 128]
            b2 = mv[half_off + 128 : half_off + 192]
            q0 = [b0[i] & 0x77 for i in range(64)]
            q1 = [b1[i] & 0x77 for i in range(64)]
            q2 = [b2[i] & 0x77 for i in range(64)]
            q3 = [
                ((b0[i] & 0x88) >> 3) | ((b1[i] & 0x88) >> 2) | ((b2[i] & 0x88) >> 1)
                for i in range(64)
            ]
            quads = (q0, q1, q2, q3)
            kp_base = h * 16
            for q_idx, quad in enumerate(quads):
                for kp_local in range(4):
                    kp = kp_base + q_idx * 4 + kp_local
                    k0 = k_base + 2 * kp
                    k1 = k0 + 1
                    row_i = kp_local * SME2_PANEL_COLS
                    if param_kind == QuantParamKind.AFFINE_SCALE_BIAS:
                        for c in range(num_rows):
                            byte = quad[row_i + c]
                            s = first_16[c]
                            b = bias_16[c]
                            rows[c][k0] = b + (byte & 0x07) * s
                            rows[c][k1] = b + ((byte >> 4) & 0x07) * s
                    else:
                        for c in range(num_rows):
                            byte = quad[row_i + c]
                            b = bias_16[c]
                            rows[c][k0] = first_16[byte & 0x07] + b
                            rows[c][k1] = first_16[(byte >> 4) & 0x07] + b
    return rows


def dequantize_sme2_panel64_q2_panel_f32(
    panel_bytes: bytes | bytearray | memoryview,
    k: int,
    *,
    num_rows: int = SME2_PANEL_COLS,
    param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS,
) -> list[list[float]]:
    """Dequantize one `sme2_panel64` `q2` panel into `num_rows` lists of `k` floats."""
    if not (1 <= num_rows <= SME2_PANEL_COLS):
        raise ValueError(f"num_rows must be 1..16, got {num_rows}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    groups = k // SME2_GROUP_K
    mv = memoryview(panel_bytes)
    expected = groups * SME2_Q2_GROUP_BYTES
    if len(mv) != expected:
        raise ValueError(f"q2 panel length {len(mv)} != expected {expected}")

    rows = [[0.0] * k for _ in range(num_rows)]
    for g in range(groups):
        g_off = g * SME2_Q2_GROUP_BYTES
        first_16, bias_16 = _decode_sme2_header_tables(mv[g_off : g_off + SME2_QUANT_HEADER_BYTES], param_kind)
        code_off = g_off + SME2_QUANT_HEADER_BYTES
        k_base = g * SME2_GROUP_K
        for kp in range(32):
            k0 = k_base + 2 * kp
            k1 = k0 + 1
            row_off = code_off + kp * 8
            for b_idx in range(8):
                byte = mv[row_off + b_idx]
                c0 = 2 * b_idx
                c1 = c0 + 1
                if c0 < num_rows:
                    v0 = byte & 0x03
                    v1 = (byte >> 2) & 0x03
                    if param_kind == QuantParamKind.AFFINE_SCALE_BIAS:
                        rows[c0][k0] = bias_16[c0] + v0 * first_16[c0]
                        rows[c0][k1] = bias_16[c0] + v1 * first_16[c0]
                    else:
                        rows[c0][k0] = first_16[v0] + bias_16[c0]
                        rows[c0][k1] = first_16[v1] + bias_16[c0]
                if c1 < num_rows:
                    v0 = (byte >> 4) & 0x03
                    v1 = (byte >> 6) & 0x03
                    if param_kind == QuantParamKind.AFFINE_SCALE_BIAS:
                        rows[c1][k0] = bias_16[c1] + v0 * first_16[c1]
                        rows[c1][k1] = bias_16[c1] + v1 * first_16[c1]
                    else:
                        rows[c1][k0] = first_16[v0] + bias_16[c1]
                        rows[c1][k1] = first_16[v1] + bias_16[c1]
    return rows


@dataclass(frozen=True)
class AmfHeader:
    version: int
    flags: int
    page_size: int
    checksum_algo: int
    tensor_count: int
    directory_offset: int
    directory_bytes: int
    directory_crc32: int
    header_crc32: int
    model_id: str


def bf16_bytes_to_f32_list(raw: bytes | bytearray | memoryview) -> list[float]:
    """Widen little-endian bfloat16 bytes to Python floats (exact)."""
    if len(raw) % 2 != 0:
        raise ValueError(f"bf16 byte length must be even, got {len(raw)}")
    f32_bytes = _bf16_to_f32_bytes(raw)
    return list(struct.unpack(f"<{len(raw) // 2}f", f32_bytes))


def f16_bytes_to_f32_list(raw: bytes | bytearray | memoryview) -> list[float]:
    """Widen little-endian IEEE float16 bytes to Python floats."""
    if len(raw) % 2 != 0:
        raise ValueError(f"f16 byte length must be even, got {len(raw)}")
    return list(struct.unpack(f"<{len(raw) // 2}e", raw))


def f32_bytes_to_f32_list(raw: bytes | bytearray | memoryview) -> list[float]:
    """Unpack little-endian IEEE float32 bytes to Python floats."""
    if len(raw) % 4 != 0:
        raise ValueError(f"f32 byte length must be a multiple of 4, got {len(raw)}")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def f32_values_to_f16_bytes(values: Sequence[float], *, what: str = "tensor") -> bytes:
    """Pack float values as little-endian IEEE float16 bytes."""
    try:
        return struct.pack(f"<{len(values)}e", *values)
    except OverflowError as exc:
        raise ValueError(f"{what}: value exceeds IEEE float16 range (|x| > 65504)") from exc


def dequantize_q3_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_params: bytes | bytearray | memoryview,
    k: int,
    *,
    group_size: int = 64,
) -> list[float]:
    """Dequantize one row of MLX-compatible 3-bit affine packed weights to floats."""
    if k <= 0 or group_size <= 0 or group_size % 8 != 0 or k % group_size != 0:
        raise ValueError(f"invalid q3 geometry: k={k}, group_size={group_size}")
    groups = k // group_size
    expected_data = (k * 3) // 8
    expected_params = groups * 4
    if len(row_data) != expected_data:
        raise ValueError(f"q3 row_data length {len(row_data)} != expected {expected_data}")
    if len(row_params) != expected_params:
        raise ValueError(f"q3 row_params length {len(row_params)} != expected {expected_params}")

    sbs = struct.unpack(f"<{groups * 2}e", row_params)
    blocks_per_group = group_size // 8
    lut12 = _Q3_LUT12
    out = [0.0] * k
    pos = 0
    off = 0
    sb_idx = 0
    for _ in range(groups):
        s = sbs[sb_idx]
        b = sbs[sb_idx + 1]
        sb_idx += 2
        v = (b, b + s, b + 2.0 * s, b + 3.0 * s, b + 4.0 * s, b + 5.0 * s, b + 6.0 * s, b + 7.0 * s)
        for _ in range(blocks_per_group):
            word = row_data[off] | (row_data[off + 1] << 8) | (row_data[off + 2] << 16)
            off += 3
            a0, a1, a2, a3 = lut12[word & 0xFFF]
            b0, b1, b2, b3 = lut12[word >> 12]
            out[pos] = v[a0]
            out[pos + 1] = v[a1]
            out[pos + 2] = v[a2]
            out[pos + 3] = v[a3]
            out[pos + 4] = v[b0]
            out[pos + 5] = v[b1]
            out[pos + 6] = v[b2]
            out[pos + 7] = v[b3]
            pos += 8
    return out


def dequantize_q4_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_params: bytes | bytearray | memoryview,
    k: int,
    *,
    group_size: int = 64,
) -> list[float]:
    """Dequantize one row of MLX-compatible 4-bit affine packed weights to floats."""
    if k <= 0 or group_size <= 0 or group_size % 2 != 0 or k % group_size != 0:
        raise ValueError(f"invalid q4 geometry: k={k}, group_size={group_size}")
    groups = k // group_size
    expected_data = k // 2
    expected_params = groups * 4
    if len(row_data) != expected_data:
        raise ValueError(f"q4 row_data length {len(row_data)} != expected {expected_data}")
    if len(row_params) != expected_params:
        raise ValueError(f"q4 row_params length {len(row_params)} != expected {expected_params}")

    sbs = struct.unpack(f"<{groups * 2}e", row_params)
    bytes_per_group = group_size // 2
    lut8 = _Q4_LUT8
    out = [0.0] * k
    pos = 0
    off = 0
    sb_idx = 0
    for _ in range(groups):
        s = sbs[sb_idx]
        b = sbs[sb_idx + 1]
        sb_idx += 2
        v = tuple(b + q * s for q in range(16))
        for _ in range(bytes_per_group):
            lo, hi = lut8[row_data[off]]
            off += 1
            out[pos] = v[lo]
            out[pos + 1] = v[hi]
            pos += 2
    return out


def _unpack_header(raw_header: bytes, file_size: int, *, verify_crc: bool = True) -> AmfHeader:
    if len(raw_header) != HEADER_STRUCT.size:
        raise ValueError(f"expected {HEADER_STRUCT.size}-byte AMF header, got {len(raw_header)}")
    (
        magic,
        version,
        flags,
        page_size,
        checksum_algo,
        tensor_count,
        _reserved0,
        dir_offset,
        dir_bytes,
        dir_crc,
        header_crc,
        model_id_raw,
        _reserved1,
    ) = HEADER_STRUCT.unpack(raw_header)
    if magic != AMF_MAGIC:
        raise ValueError(f"invalid AMF magic: {magic!r}")
    if version != AMF_VERSION:
        raise ValueError(f"unsupported AMF version: {version}")
    if page_size != APPLE_PAGE_SIZE:
        raise ValueError(f"unsupported AMF page_size: {page_size}")
    if checksum_algo != CHECKSUM_CRC32_IEEE:
        raise ValueError(f"unsupported AMF checksum_algo: {checksum_algo}")
    if verify_crc:
        zeroed = bytearray(raw_header)
        struct.pack_into("<I", zeroed, 52, 0)
        got_crc = _crc32(zeroed)
        if got_crc != header_crc:
            raise ValueError(f"AMF header CRC mismatch: expected {header_crc:#010x}, got {got_crc:#010x}")
    if dir_offset != APPLE_PAGE_SIZE or dir_bytes % APPLE_PAGE_SIZE != 0:
        raise ValueError(f"misaligned AMF directory: offset={dir_offset}, bytes={dir_bytes}")
    dir_raw_len = tensor_count * ENTRY_STRUCT.size
    if dir_raw_len > dir_bytes or dir_offset + dir_bytes > file_size:
        raise ValueError(
            f"AMF directory out of bounds: count={tensor_count}, dir_bytes={dir_bytes}, file_size={file_size}"
        )
    model_id = model_id_raw.split(b"\x00", 1)[0].decode("utf-8")
    return AmfHeader(
        version=version,
        flags=flags,
        page_size=page_size,
        checksum_algo=checksum_algo,
        tensor_count=tensor_count,
        directory_offset=dir_offset,
        directory_bytes=dir_bytes,
        directory_crc32=dir_crc,
        header_crc32=header_crc,
        model_id=model_id,
    )


def _unpack_directory(
    dir_raw: bytes,
    header: AmfHeader,
    file_size: int,
    *,
    verify_crc: bool = True,
) -> list[PackedTensorMeta]:
    if verify_crc:
        got_crc = _crc32(dir_raw)
        if got_crc != header.directory_crc32:
            raise ValueError(
                f"AMF directory CRC mismatch: expected {header.directory_crc32:#010x}, got {got_crc:#010x}"
            )
    entries: list[PackedTensorMeta] = []
    for i in range(header.tensor_count):
        (
            name_raw,
            kind_raw,
            dtype_raw,
            quant_bits,
            rank,
            group_size,
            layer,
            expert,
            reserved0,
            blob_crc32,
            s0,
            s1,
            s2,
            s3,
            blob_offset,
            blob_bytes,
            data_bytes,
            params_offset,
            params_bytes,
            _reserved1,
        ) = ENTRY_STRUCT.unpack_from(dir_raw, i * ENTRY_STRUCT.size)
        name = name_raw.split(b"\x00", 1)[0].decode("utf-8")
        if not name:
            raise ValueError(f"empty AMF tensor name at directory index {i}")
        if not (1 <= rank <= 4):
            raise ValueError(f"invalid AMF rank {rank} for {name!r}")
        if blob_offset % APPLE_PAGE_SIZE != 0 or blob_bytes % APPLE_PAGE_SIZE != 0:
            raise ValueError(f"misaligned blob for {name!r}: offset={blob_offset}, bytes={blob_bytes}")
        if blob_offset + blob_bytes > file_size:
            raise ValueError(f"blob out of bounds for {name!r}")
        layout, param_kind = decode_reserved0(reserved0)
        shape = (s0, s1, s2, s3)[:rank]
        entries.append(
            PackedTensorMeta(
                name=name,
                kind=TensorKind(kind_raw),
                dtype=DType(dtype_raw),
                quant_bits=quant_bits,
                group_size=group_size,
                layer=layer,
                expert=expert,
                shape=shape,
                blob_offset=blob_offset,
                blob_bytes=blob_bytes,
                data_bytes=data_bytes,
                params_offset=params_offset,
                params_bytes=params_bytes,
                blob_crc32=blob_crc32,
                layout=layout,
                quant_param_kind=param_kind,
            )
        )
    return entries


class AmfReader:
    """Bounded-memory, F_NOCACHE random-access reader for `.amf` v0 files."""

    def __init__(
        self,
        path: pathlib.Path,
        *,
        verify_header_crc: bool = True,
        verify_dir_crc: bool = True,
    ) -> None:
        self.path = pathlib.Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"AMF file not found: {self.path}")
        self.file_size = self.path.stat().st_size
        if self.file_size < APPLE_PAGE_SIZE or self.file_size % APPLE_PAGE_SIZE != 0:
            raise ValueError(f"invalid AMF file size {self.file_size} for {self.path}")
        self._fh = self.path.open("rb", buffering=0)
        try:
            _set_nocache(self._fh.fileno())
            hdr_raw = _read_exact(self._fh, 0, HEADER_STRUCT.size)
            self.header = _unpack_header(hdr_raw, self.file_size, verify_crc=verify_header_crc)
            dir_raw_len = self.header.tensor_count * ENTRY_STRUCT.size
            dir_raw = (
                _read_exact(self._fh, self.header.directory_offset, dir_raw_len)
                if dir_raw_len > 0
                else b""
            )
            entries = _unpack_directory(dir_raw, self.header, self.file_size, verify_crc=verify_dir_crc)
            self.entries: tuple[PackedTensorMeta, ...] = tuple(entries)
            self.by_name: dict[str, PackedTensorMeta] = {e.name: e for e in self.entries}
        except BaseException:
            self._fh.close()
            raise

    def __enter__(self) -> AmfReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def read_blob(self, entry: PackedTensorMeta, *, verify_crc: bool = False) -> bytes:
        raw = _read_exact(self._fh, entry.blob_offset, entry.blob_bytes)
        if verify_crc:
            got = _crc32(raw)
            if got != entry.blob_crc32:
                raise ValueError(
                    f"blob CRC mismatch for {entry.name!r}: expected {entry.blob_crc32:#010x}, got {got:#010x}"
                )
        return raw

    def _read_sme2_panel_f32(
        self,
        entry: PackedTensorMeta,
        panel_idx: int,
        num_rows: int,
    ) -> list[list[float]]:
        _, k = entry.shape
        groups = k // SME2_GROUP_K
        if entry.dtype in (DType.F16, DType.BF16):
            panel_bytes = groups * SME2_F16_GROUP_BYTES
            raw = _read_exact(self._fh, entry.blob_offset + panel_idx * panel_bytes, panel_bytes)
            row_raws = unpack_sme2_panel64_f16_panel(raw, k, num_rows=num_rows)
            conv = f16_bytes_to_f32_list if entry.dtype == DType.F16 else bf16_bytes_to_f32_list
            return [conv(r) for r in row_raws]
        if entry.dtype == DType.Q4_AFFINE:
            panel_bytes = groups * SME2_Q4_GROUP_BYTES
            raw = _read_exact(self._fh, entry.blob_offset + panel_idx * panel_bytes, panel_bytes)
            return dequantize_sme2_panel64_q4_panel_f32(
                raw, k, num_rows=num_rows, param_kind=entry.quant_param_kind
            )
        if entry.dtype == DType.Q3_AFFINE:
            panel_bytes = groups * SME2_Q3_GROUP_BYTES
            raw = _read_exact(self._fh, entry.blob_offset + panel_idx * panel_bytes, panel_bytes)
            return dequantize_sme2_panel64_q3_panel_f32(
                raw, k, num_rows=num_rows, param_kind=entry.quant_param_kind
            )
        raise ValueError(f"unsupported sme2_panel64 dtype {entry.dtype.name} for {entry.name!r}")

    def read_rows_f32(self, entry: PackedTensorMeta, row_indices: Sequence[int]) -> list[list[float]]:
        """Stream only selected rows of a 2D tensor into fp32 lists via `os.pread`."""
        if len(entry.shape) != 2:
            raise ValueError(f"read_rows_f32 requires a 2D tensor, got {entry.name!r} {entry.shape}")
        n_rows, k = entry.shape
        if entry.layout == TensorLayout.SME2_PANEL64:
            panel_cache: dict[int, list[list[float]]] = {}
            out_sme: list[list[float]] = []
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                p = r // SME2_PANEL_COLS
                c = r % SME2_PANEL_COLS
                if p not in panel_cache:
                    valid_in_panel = min(SME2_PANEL_COLS, n_rows - p * SME2_PANEL_COLS)
                    panel_cache[p] = self._read_sme2_panel_f32(entry, p, valid_in_panel)
                out_sme.append(panel_cache[p][c])
            return out_sme
        out: list[list[float]] = []

        if entry.dtype in (DType.Q3_AFFINE, DType.Q4_AFFINE):
            gs = entry.group_size
            groups = k // gs
            row_data_bytes = (k * 3) // 8 if entry.dtype == DType.Q3_AFFINE else k // 2
            row_param_bytes = groups * 4
            deq = dequantize_q3_row_f32 if entry.dtype == DType.Q3_AFFINE else dequantize_q4_row_f32
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                d_raw = _read_exact(self._fh, entry.blob_offset + r * row_data_bytes, row_data_bytes)
                p_raw = _read_exact(
                    self._fh,
                    entry.blob_offset + entry.params_offset + r * row_param_bytes,
                    row_param_bytes,
                )
                out.append(deq(d_raw, p_raw, k, group_size=gs))
            return out

        if entry.dtype == DType.F16:
            row_bytes = k * 2
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                raw = _read_exact(self._fh, entry.blob_offset + r * row_bytes, row_bytes)
                out.append(f16_bytes_to_f32_list(raw))
            return out

        if entry.dtype == DType.BF16:
            row_bytes = k * 2
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                raw = _read_exact(self._fh, entry.blob_offset + r * row_bytes, row_bytes)
                out.append(bf16_bytes_to_f32_list(raw))
            return out

        if entry.dtype == DType.F32:
            row_bytes = k * 4
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                raw = _read_exact(self._fh, entry.blob_offset + r * row_bytes, row_bytes)
                out.append(f32_bytes_to_f32_list(raw))
            return out

        raise ValueError(f"unsupported dtype {entry.dtype.name} for float row read on {entry.name!r}")

    def read_matrix_f32(self, entry: PackedTensorMeta) -> list[list[float]]:
        """Read and dequantize an entire 2D tensor into `N` rows of `K` floats."""
        if len(entry.shape) != 2:
            raise ValueError(f"read_matrix_f32 requires a 2D tensor, got {entry.name!r} {entry.shape}")
        n_rows, k = entry.shape
        if entry.layout == TensorLayout.SME2_PANEL64:
            panels = (n_rows + SME2_PANEL_COLS - 1) // SME2_PANEL_COLS
            rows_out: list[list[float]] = []
            for p in range(panels):
                valid = min(SME2_PANEL_COLS, n_rows - p * SME2_PANEL_COLS)
                rows_out.extend(self._read_sme2_panel_f32(entry, p, valid))
            return rows_out
        if entry.dtype in (DType.Q3_AFFINE, DType.Q4_AFFINE):
            data_mv = memoryview(_read_exact(self._fh, entry.blob_offset, entry.data_bytes))
            params_mv = memoryview(
                _read_exact(self._fh, entry.blob_offset + entry.params_offset, entry.params_bytes)
            )
            gs = entry.group_size
            row_data_bytes = (k * 3) // 8 if entry.dtype == DType.Q3_AFFINE else k // 2
            row_param_bytes = (k // gs) * 4
            deq = dequantize_q3_row_f32 if entry.dtype == DType.Q3_AFFINE else dequantize_q4_row_f32
            return [
                deq(
                    data_mv[r * row_data_bytes : (r + 1) * row_data_bytes],
                    params_mv[r * row_param_bytes : (r + 1) * row_param_bytes],
                    k,
                    group_size=gs,
                )
                for r in range(n_rows)
            ]

        raw_mv = memoryview(_read_exact(self._fh, entry.blob_offset, entry.data_bytes))
        if entry.dtype == DType.F16:
            row_bytes = k * 2
            return [f16_bytes_to_f32_list(raw_mv[r * row_bytes : (r + 1) * row_bytes]) for r in range(n_rows)]
        if entry.dtype == DType.BF16:
            row_bytes = k * 2
            return [bf16_bytes_to_f32_list(raw_mv[r * row_bytes : (r + 1) * row_bytes]) for r in range(n_rows)]
        if entry.dtype == DType.F32:
            row_bytes = k * 4
            return [f32_bytes_to_f32_list(raw_mv[r * row_bytes : (r + 1) * row_bytes]) for r in range(n_rows)]
        raise ValueError(f"unsupported dtype {entry.dtype.name} for matrix read on {entry.name!r}")

    def read_tensor_f16_bytes(self, entry: PackedTensorMeta) -> bytes:
        """Read and convert/dequantize any floating/quantized tensor to little-endian fp16 bytes."""
        if entry.layout == TensorLayout.SME2_PANEL64:
            if len(entry.shape) != 2:
                raise ValueError(f"sme2_panel64 tensor {entry.name!r} must be 2D, got {entry.shape}")
            n_rows, k = entry.shape
            panels = (n_rows + SME2_PANEL_COLS - 1) // SME2_PANEL_COLS
            if entry.dtype in (DType.F16, DType.BF16):
                groups = k // SME2_GROUP_K
                panel_bytes = groups * SME2_F16_GROUP_BYTES
                chunks_f16: list[bytes] = []
                for p in range(panels):
                    valid = min(SME2_PANEL_COLS, n_rows - p * SME2_PANEL_COLS)
                    raw = _read_exact(self._fh, entry.blob_offset + p * panel_bytes, panel_bytes)
                    for row_raw in unpack_sme2_panel64_f16_panel(raw, k, num_rows=valid):
                        chunks_f16.append(
                            row_raw if entry.dtype == DType.F16 else bf16_bytes_to_f16_bytes(row_raw, what=entry.name)
                        )
                return b"".join(chunks_f16)
            out_f16 = bytearray(n_rows * k * 2)
            row_packer = struct.Struct(f"<{k}e")
            row_out_bytes = k * 2
            row_cursor = 0
            for p in range(panels):
                valid = min(SME2_PANEL_COLS, n_rows - p * SME2_PANEL_COLS)
                for vals in self._read_sme2_panel_f32(entry, p, valid):
                    try:
                        row_packer.pack_into(out_f16, row_cursor * row_out_bytes, *vals)
                    except OverflowError as exc:
                        raise ValueError(
                            f"{entry.name}[{row_cursor}]: dequantized value exceeds IEEE float16 range"
                        ) from exc
                    row_cursor += 1
            return bytes(out_f16)
        if entry.dtype == DType.F16:
            return _read_exact(self._fh, entry.blob_offset, entry.data_bytes)
        if entry.dtype == DType.BF16:
            chunks: list[bytes] = []
            step = _CONVERT_CHUNK_ELEMS * 2
            done = 0
            while done < entry.data_bytes:
                take = min(step, entry.data_bytes - done)
                raw = _read_exact(self._fh, entry.blob_offset + done, take)
                chunks.append(bf16_bytes_to_f16_bytes(raw, what=entry.name))
                done += take
            return b"".join(chunks)
        if entry.dtype == DType.F32:
            chunks = []
            step = _CONVERT_CHUNK_ELEMS * 4
            done = 0
            while done < entry.data_bytes:
                take = min(step, entry.data_bytes - done)
                raw = _read_exact(self._fh, entry.blob_offset + done, take)
                chunks.append(_f32_bytes_to_f16_bytes(raw, entry.name))
                done += take
            return b"".join(chunks)
        if entry.dtype in (DType.Q3_AFFINE, DType.Q4_AFFINE):
            if len(entry.shape) != 2:
                raise ValueError(f"quantized tensor {entry.name!r} must be 2D, got {entry.shape}")
            n_rows, k = entry.shape
            gs = entry.group_size
            row_data_bytes = (k * 3) // 8 if entry.dtype == DType.Q3_AFFINE else k // 2
            row_param_bytes = (k // gs) * 4
            deq = dequantize_q3_row_f32 if entry.dtype == DType.Q3_AFFINE else dequantize_q4_row_f32
            data_mv = memoryview(_read_exact(self._fh, entry.blob_offset, entry.data_bytes))
            params_mv = memoryview(
                _read_exact(self._fh, entry.blob_offset + entry.params_offset, entry.params_bytes)
            )
            out = bytearray(n_rows * k * 2)
            row_packer = struct.Struct(f"<{k}e")
            row_out_bytes = k * 2
            for r in range(n_rows):
                vals = deq(
                    data_mv[r * row_data_bytes : (r + 1) * row_data_bytes],
                    params_mv[r * row_param_bytes : (r + 1) * row_param_bytes],
                    k,
                    group_size=gs,
                )
                try:
                    row_packer.pack_into(out, r * row_out_bytes, *vals)
                except OverflowError as exc:
                    raise ValueError(
                        f"{entry.name}[{r}]: dequantized value exceeds IEEE float16 range"
                    ) from exc
            return bytes(out)
        raise ValueError(f"cannot convert dtype {entry.dtype.name} to float16 for {entry.name!r}")


def load_amf_metadata(path: pathlib.Path) -> tuple[AmfHeader, list[PackedTensorMeta]]:
    """Read and validate only the header and directory of an `.amf` v0 file."""
    with AmfReader(path) as reader:
        return reader.header, list(reader.entries)


def load_cold_experts_from_heat_profile(
    heat_profile_path: pathlib.Path,
    *,
    cold_experts_per_layer: int | None = None,
) -> set[tuple[int, int]]:
    """Load `(layer, expert)` cold pairs from a schema v1 heat profile JSON."""
    raw = json.loads(pathlib.Path(heat_profile_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"heat profile {heat_profile_path} must be a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported heat profile schema_version {schema_version!r} (expected 1)")
    layers = raw.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"heat profile {heat_profile_path} is missing a 'layers' list")

    cold: set[tuple[int, int]] = set()
    for item in layers:
        if not isinstance(item, dict) or "layer" not in item:
            raise ValueError(f"malformed layer entry in heat profile {heat_profile_path}")
        layer_idx = int(item["layer"])
        if cold_experts_per_layer is not None:
            ranked = item.get("ranked_hot_to_cold")
            if not isinstance(ranked, list):
                raise ValueError(
                    f"layer {layer_idx} in {heat_profile_path} is missing 'ranked_hot_to_cold'"
                )
            if cold_experts_per_layer < 0 or cold_experts_per_layer > len(ranked):
                raise ValueError(
                    f"cold_experts_per_layer={cold_experts_per_layer} out of range 0..{len(ranked)} "
                    f"for layer {layer_idx}"
                )
            selected = ranked[len(ranked) - cold_experts_per_layer :] if cold_experts_per_layer > 0 else []
            for e in selected:
                cold.add((layer_idx, int(e)))
        else:
            cold_list = item.get("cold_experts")
            if not isinstance(cold_list, list):
                raise ValueError(f"layer {layer_idx} in {heat_profile_path} is missing 'cold_experts'")
            for e in cold_list:
                cold.add((layer_idx, int(e)))
    return cold


def retag_amf_cold_experts(
    amf_path: pathlib.Path,
    cold_experts: Iterable[tuple[int, int]] = (),
    *,
    output_path: pathlib.Path | None = None,
    all_experts_cold: bool = False,
) -> tuple[int, int]:
    """Retag `HOT_EXPERT` / `COLD_EXPERT` directory entries without touching tensor blobs.

    If `output_path` is omitted or resolves to `amf_path`, updates Page 0 and
    the directory in place (`O(directory_bytes)` I/O, zero blob writes).
    Otherwise writes a complete validated copy to `output_path` atomically.

    Returns `(hot_expert_tensor_count, cold_expert_tensor_count)`.
    """
    src_path = pathlib.Path(amf_path)
    dst_path = src_path if output_path is None else pathlib.Path(output_path)
    cold_set = {(int(l), int(e)) for l, e in cold_experts}

    header, entries = load_amf_metadata(src_path)
    updated: list[PackedTensorMeta] = []
    hot_count = 0
    cold_count = 0
    for m in entries:
        if m.kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT) or m.expert != NO_EXPERT:
            is_cold = all_experts_cold or (m.layer, m.expert) in cold_set
            new_kind = TensorKind.COLD_EXPERT if is_cold else TensorKind.HOT_EXPERT
            if is_cold:
                cold_count += 1
            else:
                hot_count += 1
            if new_kind != m.kind:
                m = PackedTensorMeta(
                    name=m.name,
                    kind=new_kind,
                    dtype=m.dtype,
                    quant_bits=m.quant_bits,
                    group_size=m.group_size,
                    layer=m.layer,
                    expert=m.expert,
                    shape=m.shape,
                    blob_offset=m.blob_offset,
                    blob_bytes=m.blob_bytes,
                    data_bytes=m.data_bytes,
                    params_offset=m.params_offset,
                    params_bytes=m.params_bytes,
                    blob_crc32=m.blob_crc32,
                    layout=m.layout,
                    quant_param_kind=m.quant_param_kind,
                )
        updated.append(m)

    dir_raw = _pack_directory(updated)
    new_header = _pack_header(
        header.model_id.encode("utf-8"),
        header.tensor_count,
        header.directory_offset,
        header.directory_bytes,
        _crc32(dir_raw),
    )

    in_place = dst_path.exists() and dst_path.resolve() == src_path.resolve()
    if in_place:
        with src_path.open("r+b") as fh:
            fh.seek(0)
            fh.write(new_header)
            if dir_raw:
                fh.seek(header.directory_offset)
                fh.write(dir_raw)
            fh.flush()
            os.fsync(fh.fileno())
        return hot_count, cold_count

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    partial = dst_path.with_name(dst_path.name + ".partial")
    blobs_start = header.directory_offset + header.directory_bytes
    file_size = src_path.stat().st_size
    try:
        with src_path.open("rb", buffering=0) as src_fh, partial.open("wb") as out_fh:
            _set_nocache(src_fh.fileno())
            _set_nocache(out_fh.fileno())
            page0 = bytearray(APPLE_PAGE_SIZE)
            page0[0 : len(new_header)] = new_header
            out_fh.write(page0)
            if header.directory_bytes > 0:
                dir_page = bytearray(header.directory_bytes)
                dir_page[0 : len(dir_raw)] = dir_raw
                out_fh.write(dir_page)
            pos = blobs_start
            while pos < file_size:
                take = min(_COPY_CHUNK_BYTES, file_size - pos)
                out_fh.write(_read_exact(src_fh, pos, take))
                pos += take
            out_fh.flush()
            os.fsync(out_fh.fileno())
        os.replace(partial, dst_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return hot_count, cold_count


def _verify_source_blob_crc(src_fh, entry: PackedTensorMeta) -> None:
    crc = 0
    done = 0
    while done < entry.blob_bytes:
        take = min(_COPY_CHUNK_BYTES, entry.blob_bytes - done)
        chunk = _read_exact(src_fh, entry.blob_offset + done, take)
        crc = zlib.crc32(chunk, crc)
        done += take
    got = crc & 0xFFFFFFFF
    if got != entry.blob_crc32:
        raise ValueError(
            f"source blob CRC mismatch for {entry.name!r}: expected {entry.blob_crc32:#010x}, got {got:#010x}"
        )


def _stream_repack_amf_entry(src_fh, entry: PackedTensorMeta, new_blob_bytes: int, w: _BlobWriter) -> None:
    """Stream one `ROW_MAJOR` 2D `.amf` tensor 16 rows at a time into `SME2_PANEL64`."""
    n_rows, k = entry.shape
    panels = (n_rows + SME2_PANEL_COLS - 1) // SME2_PANEL_COLS
    if entry.dtype in (DType.F16, DType.BF16):
        row_bytes = k * 2
        for p in range(panels):
            r0 = p * SME2_PANEL_COLS
            count = min(SME2_PANEL_COLS, n_rows - r0)
            raw = _read_exact(src_fh, entry.blob_offset + r0 * row_bytes, count * row_bytes)
            mv = memoryview(raw)
            rows = [mv[c * row_bytes : (c + 1) * row_bytes] for c in range(count)]
            w.write(pack_sme2_panel64_f16_panel(rows, k))
    elif entry.dtype in (DType.Q4_AFFINE, DType.Q3_AFFINE):
        groups = k // entry.group_size
        row_data_bytes = k // 2 if entry.dtype == DType.Q4_AFFINE else (k * 3) // 8
        row_param_bytes = groups * 4
        packer = pack_sme2_panel64_q4_panel if entry.dtype == DType.Q4_AFFINE else pack_sme2_panel64_q3_panel
        for p in range(panels):
            r0 = p * SME2_PANEL_COLS
            count = min(SME2_PANEL_COLS, n_rows - r0)
            raw_d = _read_exact(src_fh, entry.blob_offset + r0 * row_data_bytes, count * row_data_bytes)
            raw_p = _read_exact(
                src_fh,
                entry.blob_offset + entry.params_offset + r0 * row_param_bytes,
                count * row_param_bytes,
            )
            d_mv = memoryview(raw_d)
            p_mv = memoryview(raw_p)
            d_rows = [d_mv[c * row_data_bytes : (c + 1) * row_data_bytes] for c in range(count)]
            p_rows = [p_mv[c * row_param_bytes : (c + 1) * row_param_bytes] for c in range(count)]
            w.write(packer(d_rows, p_rows, k))
    else:
        raise ValueError(f"cannot repack unsupported dtype {entry.dtype.name} for {entry.name!r}")

    expected_data = sme2_panel64_data_bytes(entry.dtype, n_rows, k, entry.group_size)
    if w.written != expected_data:
        raise AssertionError(f"repacked {w.written} bytes for {entry.name!r}, expected {expected_data}")
    w.zeros(new_blob_bytes - w.written)


def repack_amf_sme2(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    targets: str = "routers",
    verify_source_crc: bool = True,
) -> list[PackedTensorMeta]:
    """Stream-convert selected 2D tensors in an `.amf` v0 file into single-owner `sme2_panel64`.

    Preserves single-owner layout (every tensor appears once — never duplicated in both
    `row_major` and `sme2_panel64`), 16 KiB page alignment, IEEE CRC-32 checksums, and
    all metadata (`kind`, `dtype`, `layer`, `expert`, `shape`). Already-packed
    `sme2_panel64` entries are copied verbatim so repacking is idempotent.
    """
    if targets not in SME2_REPACK_TARGETS:
        raise ValueError(f"targets must be one of {SME2_REPACK_TARGETS}, got {targets!r}")

    src_path = pathlib.Path(input_path)
    dst_path = pathlib.Path(output_path)
    header, entries = load_amf_metadata(src_path)

    tensor_count = len(entries)
    dir_offset = APPLE_PAGE_SIZE
    dir_raw_len = tensor_count * ENTRY_STRUCT.size
    dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)

    planned: list[tuple[PackedTensorMeta, bool, int, int, int, int, int, TensorLayout, QuantParamKind]] = []
    cursor = dir_offset + dir_bytes
    for m in entries:
        needs_repack = m.layout == TensorLayout.ROW_MAJOR and _should_sme2_pack(m.kind, m.dtype, m.shape, targets)
        if needs_repack:
            new_layout = TensorLayout.SME2_PANEL64
            new_param_kind = QuantParamKind.AFFINE_SCALE_BIAS
            new_data_bytes = sme2_panel64_data_bytes(m.dtype, m.shape[0], m.shape[1], m.group_size)
            new_params_offset = 0
            new_params_bytes = 0
            new_blob_bytes = _align_up(max(new_data_bytes, 1), APPLE_PAGE_SIZE)
        else:
            new_layout = m.layout
            new_param_kind = m.quant_param_kind
            new_data_bytes = m.data_bytes
            new_params_offset = m.params_offset
            new_params_bytes = m.params_bytes
            new_blob_bytes = m.blob_bytes
        planned.append(
            (
                m,
                needs_repack,
                cursor,
                new_blob_bytes,
                new_data_bytes,
                new_params_offset,
                new_params_bytes,
                new_layout,
                new_param_kind,
            )
        )
        cursor += new_blob_bytes

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    partial = dst_path.with_name(dst_path.name + ".partial")
    new_metas: list[PackedTensorMeta] = []
    try:
        with src_path.open("rb", buffering=0) as src_fh, partial.open("wb") as out_fh:
            _set_nocache(src_fh.fileno())
            _set_nocache(out_fh.fileno())
            prefix = _BlobWriter(out_fh)
            prefix.zeros(dir_offset + dir_bytes)

            for (
                m,
                needs_repack,
                blob_offset,
                new_blob_bytes,
                new_data_bytes,
                new_params_offset,
                new_params_bytes,
                new_layout,
                new_param_kind,
            ) in planned:
                if out_fh.tell() != blob_offset:
                    raise AssertionError(f"planned offset {blob_offset} != file pos {out_fh.tell()} for {m.name!r}")
                w = _BlobWriter(out_fh)
                if needs_repack:
                    if verify_source_crc:
                        _verify_source_blob_crc(src_fh, m)
                    _stream_repack_amf_entry(src_fh, m, new_blob_bytes, w)
                else:
                    done = 0
                    while done < m.blob_bytes:
                        take = min(_COPY_CHUNK_BYTES, m.blob_bytes - done)
                        w.write(_read_exact(src_fh, m.blob_offset + done, take))
                        done += take
                    if verify_source_crc and (w.crc & 0xFFFFFFFF) != m.blob_crc32:
                        raise ValueError(
                            f"source blob CRC mismatch for {m.name!r}: "
                            f"expected {m.blob_crc32:#010x}, got {w.crc & 0xFFFFFFFF:#010x}"
                        )
                if w.written != new_blob_bytes:
                    raise AssertionError(f"wrote {w.written} bytes for {m.name!r}, planned {new_blob_bytes}")
                new_metas.append(
                    PackedTensorMeta(
                        name=m.name,
                        kind=m.kind,
                        dtype=m.dtype,
                        quant_bits=m.quant_bits,
                        group_size=m.group_size,
                        layer=m.layer,
                        expert=m.expert,
                        shape=m.shape,
                        blob_offset=blob_offset,
                        blob_bytes=new_blob_bytes,
                        data_bytes=new_data_bytes,
                        params_offset=new_params_offset,
                        params_bytes=new_params_bytes,
                        blob_crc32=w.crc & 0xFFFFFFFF,
                        layout=new_layout,
                        quant_param_kind=new_param_kind,
                    )
                )

            if out_fh.tell() != cursor:
                raise AssertionError(f"final size {out_fh.tell()} != planned {cursor}")

            dir_raw = _pack_directory(new_metas)
            new_header = _pack_header(
                header.model_id.encode("utf-8"),
                tensor_count,
                dir_offset,
                dir_bytes,
                _crc32(dir_raw),
            )
            out_fh.seek(0)
            out_fh.write(new_header)
            if dir_raw:
                out_fh.seek(dir_offset)
                out_fh.write(dir_raw)
            out_fh.flush()
            os.fsync(out_fh.fileno())
        os.replace(partial, dst_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return new_metas
