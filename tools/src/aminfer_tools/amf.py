"""aminfer Model Format (.amf v0) writer and MLX safetensors importer (ADR-0006).

Pure standard-library Python implementation so offline conversion and testing
require zero third-party packages in M1.
"""

from __future__ import annotations

import enum
import json
import pathlib
import re
import struct
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


def bf16_bytes_to_f16_bytes(raw: bytes) -> bytes:
    """Convert little-endian bfloat16 bytes to little-endian IEEE 754 float16 bytes."""
    if len(raw) % 2 != 0:
        raise ValueError(f"bf16 byte length must be even, got {len(raw)}")
    count = len(raw) // 2
    out = bytearray(count * 2)
    for i in range(count):
        f32_val = struct.unpack_from("<f", b"\x00\x00" + raw[2 * i : 2 * i + 2], 0)[0]
        struct.pack_into("<e", out, 2 * i, f32_val)
    return bytes(out)


def interleave_scale_bias_f16(scales_f16: bytes, biases_f16: bytes) -> bytes:
    """Interleave separate fp16 scales and biases into half2(scale, bias) pairs."""
    if len(scales_f16) != len(biases_f16) or len(scales_f16) % 2 != 0:
        raise ValueError("scales and biases must have identical even byte lengths")
    count = len(scales_f16) // 2
    out = bytearray(count * 4)
    for i in range(count):
        out[4 * i : 4 * i + 2] = scales_f16[2 * i : 2 * i + 2]
        out[4 * i + 2 : 4 * i + 4] = biases_f16[2 * i : 2 * i + 2]
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
    if "embed" in lower:
        return (TensorKind.EMBEDDING, layer, expert)
    if "lm_head" in lower:
        return (TensorKind.LM_HEAD, layer, expert)
    if "norm" in lower:
        return (TensorKind.NORM, layer, expert)
    if lower.endswith(".mlp.gate") or lower.endswith(".mlp.gate.weight"):
        return (TensorKind.ROUTER, layer, expert)
    return (TensorKind.DENSE, layer, expert)


def _read_safetensors(path: pathlib.Path) -> tuple[dict[str, dict[str, object]], int]:
    with path.open("rb") as f:
        hdr_len_bytes = f.read(8)
        if len(hdr_len_bytes) != 8:
            raise ValueError(f"truncated safetensors header in {path}")
        (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
        hdr_raw = f.read(hdr_len)
        if len(hdr_raw) != hdr_len:
            raise ValueError(f"truncated safetensors JSON in {path}")
    header = json.loads(hdr_raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return tensors, 8 + hdr_len


def _read_slice(fh, data_base: int, spec: dict[str, object]) -> bytes:
    offsets = spec["data_offsets"]
    assert isinstance(offsets, list) and len(offsets) == 2
    start, end = int(offsets[0]), int(offsets[1])
    fh.seek(data_base + start)
    raw = fh.read(end - start)
    if len(raw) != end - start:
        raise ValueError("unexpected EOF reading safetensors tensor data")
    return raw


def _to_f16_bytes(dtype_str: str, raw: bytes) -> bytes:
    if dtype_str == "F16":
        return raw
    if dtype_str == "BF16":
        return bf16_bytes_to_f16_bytes(raw)
    raise ValueError(f"expected F16 or BF16 scales/biases, got {dtype_str}")


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


def pack_amf_from_safetensors(
    safetensors_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str,
    group_size: int = 64,
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
) -> list[PackedTensorMeta]:
    """Convert an MLX affine-quantized safetensors file into a 16 KiB-aligned .amf v0 file."""
    if group_size != 64:
        raise ValueError(f"aminfer v0 supports group_size=64 only, got {group_size}")
    cold_set = set(cold_experts)
    tensors, data_base = _read_safetensors(safetensors_path)

    quant_prefixes: set[str] = set()
    for name in tensors:
        if name.endswith(".scales"):
            prefix = name[: -len(".scales")]
            if f"{prefix}.weight" in tensors and f"{prefix}.biases" in tensors:
                quant_prefixes.add(prefix)

    consumed: set[str] = set()
    records: list[
        tuple[str, TensorKind, DType, int, int, int, int, tuple[int, ...], bytes, bytes]
    ] = []

    with safetensors_path.open("rb") as fh:
        for prefix in sorted(quant_prefixes):
            w_name = f"{prefix}.weight"
            s_name = f"{prefix}.scales"
            b_name = f"{prefix}.biases"
            consumed.update((w_name, s_name, b_name))

            w_spec = tensors[w_name]
            s_spec = tensors[s_name]
            b_spec = tensors[b_name]

            if w_spec["dtype"] != "U32":
                raise ValueError(f"expected U32 packed weights for {w_name}, got {w_spec['dtype']}")
            w_shape = [int(x) for x in w_spec["shape"]]  # type: ignore[union-attr]
            s_shape = [int(x) for x in s_spec["shape"]]  # type: ignore[union-attr]
            b_shape = [int(x) for x in b_spec["shape"]]  # type: ignore[union-attr]
            if len(w_shape) != 2 or s_shape != b_shape or len(s_shape) != 2 or w_shape[0] != s_shape[0]:
                raise ValueError(f"inconsistent quantized shapes for {prefix}: {w_shape}, {s_shape}, {b_shape}")

            n = w_shape[0]
            groups = s_shape[1]
            k = groups * group_size
            bits_num = w_shape[1] * 32
            if bits_num % k != 0:
                raise ValueError(f"cannot infer bit width for {prefix}: packed={w_shape[1]}, k={k}")
            bits = bits_num // k
            if bits == 4:
                amf_dtype = DType.Q4_AFFINE
            elif bits == 3:
                amf_dtype = DType.Q3_AFFINE
            else:
                raise ValueError(f"unsupported bit width {bits} for {prefix} (expected 3 or 4)")

            qweight_raw = _read_slice(fh, data_base, w_spec)
            scales_f16 = _to_f16_bytes(str(s_spec["dtype"]), _read_slice(fh, data_base, s_spec))
            biases_f16 = _to_f16_bytes(str(b_spec["dtype"]), _read_slice(fh, data_base, b_spec))
            params_raw = interleave_scale_bias_f16(scales_f16, biases_f16)

            norm_name = _normalize_name(prefix)
            kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
            records.append(
                (norm_name, kind, amf_dtype, bits, group_size, layer, expert, (n, k), qweight_raw, params_raw)
            )

        for name in sorted(tensors):
            if name in consumed:
                continue
            spec = tensors[name]
            dtype_str = str(spec["dtype"])
            shape = tuple(int(x) for x in spec["shape"])  # type: ignore[union-attr]
            raw = _read_slice(fh, data_base, spec)
            if dtype_str == "F16":
                amf_dtype = DType.F16
            elif dtype_str == "BF16":
                amf_dtype = DType.F16
                raw = bf16_bytes_to_f16_bytes(raw)
            elif dtype_str == "F32":
                amf_dtype = DType.F32
            else:
                raise ValueError(f"unsupported unquantized dtype {dtype_str} for {name}")

            norm_name = _normalize_name(name)
            kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
            records.append((norm_name, kind, amf_dtype, 0, 0, layer, expert, shape, raw, b""))

    return write_amf(output_path, model_id, records)
