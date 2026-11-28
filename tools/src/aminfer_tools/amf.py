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
import math
import os
import pathlib
import random
import re
import stat
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from aminfer_tools import APPLE_PAGE_SIZE

AMF_MAGIC = b"AMF0\x00\x00\x00\x00"
AMF_VERSION = 1
CHECKSUM_CRC32_IEEE = 1
NO_LAYER = 0xFFFF
NO_EXPERT = 0xFFFF
FLAG_EMBEDDED_METADATA = 1 << 0
KNOWN_HEADER_FLAGS = FLAG_EMBEDDED_METADATA

META_PREFIX = "__meta."
META_CONFIG_JSON = "__meta.config_json"
META_TOKENIZER_JSON = "__meta.tokenizer_json"
META_ADAPTER_CONFIG_JSON = "__meta.adapter_config_json"
META_CHAT_TEMPLATE_JSON = "__meta.chat_template_json"
CHAT_TEMPLATE_METADATA_NAME = META_CHAT_TEMPLATE_JSON
LEGACY_CHAT_TEMPLATE_METADATA_NAME = "__aminfer_chat_template__"

HEADER_STRUCT = struct.Struct("<8sIIIIIIQQII64sQ")
ENTRY_STRUCT = struct.Struct("<48sBBBBHHHHI4IQQQQQQ")
assert HEADER_STRUCT.size == 128
assert ENTRY_STRUCT.size == 128

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_VISION_BLOCK_RE = re.compile(r"(?:^|\.)(?:visual|vision)\.blocks\.(\d+)\.")
_VISION_TOWER_BLOCK_RE = re.compile(
    r"^(?:model\.)?(?:"
    r"(?:visual|vision|vision_tower\.vision_model\.encoder|vision_model\.encoder)"
    r"\.(?:blocks|layers)\.\d+\.(?:norm\d?|attn|self_attn|mlp|ffn|layernorm|layer_norm)\."
    r"|visual\.backbone\.layers\.\d+\.(?:ln_\d|ls_\d|attn|self_attn|mlp)\."
    r"|vision_tower\.layers\.\d+\.(?:layernorm_before|layernorm_after|norm\d?|attn|self_attn|mlp|lambda_\d)"
    r"|vit\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm|norm\d?|attn|self_attn|mlp)\."
    r")"
)
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
_LOOP_STEP_RE = re.compile(r"(?:^|\.)loop\.step\.(\d+)\.")
_SWITCH_MLP_RE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?(?:model\.)?(?:llm\.)?layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)$"
)
_EXPERTS_STACK_RE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?(?:model\.)?(?:llm\.)?layers\.(\d+)\.(?:mlp|ffn)\.experts\."
    r"(gate_proj|up_proj|down_proj|gate_up_proj|w13|w2|w1|w3)$"
)
VISION_TENSOR_PREFIXES = (
    "visual.",
    "vision.",
    "vision_tower.",
    "vit.",
    "multi_modal_projector.",
    "aligner.",
    "model.visual.",
    "model.vision.",
    "model.vision_tower.",
    "model.vit.",
    "model.multi_modal_projector.",
    "model.aligner.",
    "vision_model.",
    "model.vision_model.",
)
VISION_EXACT_TENSOR_NAMES = (
    "image_start",
    "image_end",
    "image_newline",
)
VISION_BLOCK_DTYPES = ("auto", "f16", "bf16", "q4_affine")


def is_vision_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` belongs to a multimodal vision tower / merger / aligner (W10, W11, W20)."""
    if raw_name.startswith(VISION_TENSOR_PREFIXES):
        return True
    stripped = raw_name[6:] if raw_name.startswith("model.") else raw_name
    return stripped in VISION_EXACT_TENSOR_NAMES


def has_vision_tower(tensor_names: Iterable[str]) -> bool:
    """Return True when `tensor_names` contain multi-layer ViT encoder blocks (`visual.blocks.*` / `vision.blocks.*`).

    Distinguishes full multimodal checkpoints (`Qwen-VL`, `Qwen3.8`, `DeepSeek-V4.1`, SigLIP)
    from legacy text-only test fixtures carrying 1-3 dangling stub vision weights.
    """
    return any(_VISION_TOWER_BLOCK_RE.match(name) is not None for name in tensor_names)


def resolve_vision_mode(
    tensor_names: Iterable[str],
    *,
    include_vision: bool | None = None,
    strip_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
) -> str:
    """Resolve effective W20 vision packing mode: `'include'`, `'strip'`, or `'sidecar'`.

    Precedence:
    - Conflicting explicit flags (`include_vision=True` + `strip_vision=True`, or
      `strip_vision=True` + `vision_sidecar`) raise `ValueError`.
    - `vision_sidecar` extracts vision tensors into a standalone 16 KiB-aligned `.amf`.
    - Explicit `include_vision=True` or `strip_vision=False` keeps all vision tensors.
    - Explicit `strip_vision=True` or `include_vision=False` strips all vision tensors.
    - Default (`None`): includes full ViT towers automatically when `has_vision_tower` holds.
    """
    if include_vision is True and strip_vision is True:
        raise ValueError("cannot specify both include_vision=True and strip_vision=True")
    if vision_sidecar is not None:
        if strip_vision is True or include_vision is False:
            raise ValueError("cannot combine strip_vision with vision_sidecar")
        return "sidecar"
    if include_vision is True or strip_vision is False:
        return "include"
    if strip_vision is True or include_vision is False:
        return "strip"
    return "include" if has_vision_tower(tensor_names) else "strip"


_VERBOSE_VISION_PREFIX_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\."), r"vision.blocks.\1."),
    (re.compile(r"^vision_model\.encoder\.layers\.(\d+)\."), r"vision.blocks.\1."),
    (re.compile(r"^vision_tower\.encoder\.layers\.(\d+)\."), r"vision.blocks.\1."),
    (re.compile(r"^vision_tower\.vision_model\.embeddings\.patch_embedding\."), "vision.patch_embed.proj."),
    (re.compile(r"^vision_model\.embeddings\.patch_embedding\."), "vision.patch_embed.proj."),
    (re.compile(r"^vision_tower\.vision_model\.embeddings\.position_embedding\."), "vision.pos_embed."),
    (re.compile(r"^vision_model\.embeddings\.position_embedding\."), "vision.pos_embed."),
    (re.compile(r"^vision_tower\.vision_model\.post_layernorm\."), "vision.norm."),
    (re.compile(r"^vision_model\.post_layernorm\."), "vision.norm."),
    # Inkling 2 (`visual.backbone.*` & MLX `vision_tower.layers.*`) and Hy4 (`vit.*`)
    (re.compile(r"^visual\.backbone\.layers\.(\d+)\."), r"visual.blocks.\1."),
    (re.compile(r"^visual\.backbone\.embeddings\."), "visual.embed."),
    (re.compile(r"^visual\.backbone\.ln_pre\."), "visual.ln_pre."),
    (re.compile(r"^vision_tower\.layers\.(\d+)\."), r"vision.blocks.\1."),
    (re.compile(r"^vision_tower\.embeddings\."), "vision.embed."),
    (re.compile(r"^vision_tower\.pre_layernorm\."), "vision.ln_pre."),
    (re.compile(r"^vision_tower\.merger\."), "vision.merger."),
    (re.compile(r"^vit\.layers\.(\d+)\."), r"vision.blocks.\1."),
    (re.compile(r"^vit\.embeddings\.patch_embedding\."), "vision.patch_embed.proj."),
    (re.compile(r"^vit\.embeddings\.position_embedding\."), "vision.pos_embed."),
    (re.compile(r"^vit\.perceive\."), "vision.merger."),
)


def normalize_vision_name(raw_name: str) -> str:
    """Normalize a vision-tower tensor key to canonical `<= 47 B` `.amf` form (W20).

    Preserves native Qwen-VL / Qwen3.8 (`visual.*`) and DeepSeek-V4.1 (`vision.*`,
    `aligner.*`, `image_*`) keys verbatim while compacting verbose HF SigLIP /
    LLaVA / `vision_tower.*` / Inkling-2 `visual.backbone.*` / Hy4 `vit.*` paths
    so every name fits the 48-byte `.amf` slot.
    """
    name = _strip_model_prefix(raw_name)
    for pat, repl in _VERBOSE_VISION_PREFIX_RULES:
        updated = pat.sub(repl, name, count=1)
        if updated != name:
            name = updated
            break

    # Compact standard sub-module names on verbose vision keys (or whenever > 47 B).
    if len(name.encode("utf-8")) > 47 or name.startswith(("vision.blocks.", "visual.blocks.")):
        name = name.replace(".self_attn.", ".attn.")
        name = name.replace(".layer_norm1.", ".norm1.")
        name = name.replace(".layer_norm2.", ".norm2.")
        name = name.replace(".input_layernorm.", ".norm1.")
        name = name.replace(".post_attention_layernorm.", ".norm2.")
        name = name.replace(".layernorm_before.", ".norm1.")
        name = name.replace(".layernorm_after.", ".norm2.")
        name = name.replace(".ln_1.", ".norm1.")
        name = name.replace(".ln_2.", ".norm2.")
        name = name.replace(".out_proj.", ".proj.")
        name = name.replace(".o_proj.", ".proj.")
        name = name.replace(".dense_h_to_4h.", ".fc1.")
        name = name.replace(".dense_4h_to_h.", ".fc2.")
        name = name.replace(".c_fc.", ".fc1.")
        name = name.replace(".c_proj.", ".fc2.")
        if len(name.encode("utf-8")) > 47:
            name = name.replace(".linear_fc1.", ".fc1.")
            name = name.replace(".linear_fc2.", ".fc2.")
            if name.startswith("multi_modal_projector."):
                name = "visual.merger." + name[len("multi_modal_projector.") :]
    return name


def is_vision_pinned_float(name: str, shape: Sequence[int] | None = None) -> bool:
    """Return True if a vision tensor must remain unquantized (`f16`/`bf16`) under W14/W20.

    Keeps `patch_embed`, `pos_embed`, all norms, 1D biases, special image tokens
    (`image_start`/`image_end`/`image_newline`), and final `merger`/`aligner`
    projection weights in full 16-bit float precision.
    """
    if not is_vision_tensor(name):
        return False
    if shape is not None and len(shape) != 2:
        return True
    norm_name = normalize_vision_name(name)
    lower = norm_name.lower()
    if norm_name in VISION_EXACT_TENSOR_NAMES:
        return True
    if (
        "patch_embed" in lower
        or "pos_embed" in lower
        or "position_embedding" in lower
        or "cls_position_embedding" in lower
        or "class_embedding" in lower
        or ".embed." in lower
    ):
        return True
    if "norm" in lower or ".ln_" in lower or ".ls_" in lower or ".lambda_" in lower:
        return True
    if lower.startswith(
        (
            "visual.merger.",
            "vision.merger.",
            "aligner.",
            "multi_modal_projector.",
            "visual.proj_in.",
            "visual.proj_out.",
            "vit.perceive.",
        )
    ):
        return True
    if ".merger." in lower or ".aligner." in lower or ".perceive." in lower:
        return True
    return False


def is_vision_block_linear(name: str, shape: Sequence[int] | None = None) -> bool:
    """Return True for 2D linear projections inside ViT encoder blocks (`visual.blocks.*` / `vision.blocks.*`)."""
    if not is_vision_tensor(name) or is_vision_pinned_float(name, shape):
        return False
    norm_name = normalize_vision_name(name)
    if _VISION_BLOCK_RE.search(norm_name) is None:
        return False
    return shape is None or len(shape) == 2

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
    "F8_E8M0": 1,
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
    SHARED_EXPERT = 7
    SPARSE_TABLE = 8
    LOOP_ADAPTER = 9
    METADATA = 10


def is_meta_name(name: str) -> bool:
    return name.startswith(META_PREFIX) or name == LEGACY_CHAT_TEMPLATE_METADATA_NAME


def make_metadata_record(name: str, payload: bytes | str) -> tuple:
    """Build a 12-tuple `write_amf` record for a cold `__meta.*` UTF-8 JSON blob (W18)."""
    if not is_meta_name(name):
        raise ValueError(f"metadata entry name must start with {META_PREFIX!r}, got {name!r}")
    raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    if len(raw) == 0:
        raise ValueError(f"metadata payload for {name!r} must be non-empty")
    return (
        name,
        TensorKind.METADATA,
        DType.U32,
        0,
        0,
        NO_LAYER,
        NO_EXPERT,
        (len(raw),),
        raw,
        b"",
        TensorLayout.ROW_MAJOR,
        QuantParamKind.AFFINE_SCALE_BIAS,
    )


class DType(enum.IntEnum):
    F16 = 0
    F32 = 1
    Q4_AFFINE = 2
    Q3_AFFINE = 3
    BF16 = 4
    U32 = 5
    Q2_AFFINE = 6
    Q5_AFFINE = 7
    Q6_AFFINE = 8
    Q8_AFFINE = 9
    FP8_E4M3_BLOCK = 10
    FP8_E4M3_ROW = 11
    FP8_E4M3_TENSOR = 12
    MXFP8_G32 = 13
    NVFP4_G16 = 14
    GGUF_Q4_0 = 15
    GGUF_Q4_1 = 16
    GGUF_Q5_0 = 17
    GGUF_Q5_1 = 18
    GGUF_Q8_0 = 19
    GGUF_Q2_K = 20
    GGUF_Q3_K = 21
    GGUF_Q4_K = 22
    GGUF_Q5_K = 23
    GGUF_Q6_K = 24
    GGUF_IQ1_S = 25
    GGUF_IQ1_M = 26
    GGUF_IQ2_XXS = 27
    GGUF_IQ2_XS = 28
    GGUF_IQ2_S = 29
    GGUF_IQ3_XXS = 30
    GGUF_IQ3_S = 31
    GGUF_IQ4_NL = 32
    GGUF_IQ4_XS = 33
    GGUF_TQ1_0 = 34
    GGUF_TQ2_0 = 35
    GGUF_MXFP4 = 36
    GGUF_Q2_0 = 37


AFFINE_DTYPES = (
    DType.Q2_AFFINE,
    DType.Q3_AFFINE,
    DType.Q4_AFFINE,
    DType.Q5_AFFINE,
    DType.Q6_AFFINE,
    DType.Q8_AFFINE,
)

AFFINE_GROUP_SIZES = (32, 64, 128)


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
SME2_PACK_MODES = ("none", "controllers", "routers", "eh-proj", "all-linear")
SME2_REPACK_TARGETS = ("controllers", "routers", "eh-proj", "all-linear")


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
    elif dtype == DType.Q2_AFFINE:
        if group_size != SME2_GROUP_K:
            raise ValueError(f"q2_affine sme2_panel64 requires group_size=64, got {group_size}")
        group_bytes = SME2_Q2_GROUP_BYTES
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


def _f16_to_f32_bytes(raw: bytes | bytearray | memoryview) -> bytes:
    """Widen little-endian IEEE float16 bytes to little-endian float32 bytes."""
    if len(raw) % 2 != 0:
        raise ValueError(f"f16 byte length must be even, got {len(raw)}")
    count = len(raw) // 2
    return struct.pack(f"<{count}f", *struct.unpack(f"<{count}e", raw))


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


def pack_mlx_affine_2bit(codes: Sequence[int]) -> bytes:
    """Pack 2-bit unsigned codes LSB-first (4 values per byte: c0|(c1<<2)|(c2<<4)|(c3<<6))."""
    if len(codes) % 16 != 0:
        raise ValueError("2-bit code count must be a multiple of 16")
    out = bytearray(len(codes) // 4)
    for i in range(len(out)):
        base = 4 * i
        out[i] = (
            (codes[base] & 0x03)
            | ((codes[base + 1] & 0x03) << 2)
            | ((codes[base + 2] & 0x03) << 4)
            | ((codes[base + 3] & 0x03) << 6)
        )
    return bytes(out)


def pack_mlx_affine_5bit(codes: Sequence[int]) -> bytes:
    """Pack 5-bit unsigned codes LSB-first (8 values in 5 bytes, MLX uint32 compatible)."""
    if len(codes) % 8 != 0:
        raise ValueError("5-bit code count must be a multiple of 8")
    out = bytearray((len(codes) // 8) * 5)
    for blk in range(len(codes) // 8):
        c = codes[blk * 8 : blk * 8 + 8]
        word = 0
        for j in range(8):
            word |= (c[j] & 0x1F) << (5 * j)
        for b_i in range(5):
            out[blk * 5 + b_i] = (word >> (8 * b_i)) & 0xFF
    return bytes(out)


def pack_mlx_affine_6bit(codes: Sequence[int]) -> bytes:
    """Pack 6-bit unsigned codes LSB-first (4 values in 3 bytes, MLX uint32 compatible)."""
    if len(codes) % 4 != 0:
        raise ValueError("6-bit code count must be a multiple of 4")
    out = bytearray((len(codes) // 4) * 3)
    for blk in range(len(codes) // 4):
        c = codes[blk * 4 : blk * 4 + 4]
        word = 0
        for j in range(4):
            word |= (c[j] & 0x3F) << (6 * j)
        out[blk * 3 + 0] = word & 0xFF
        out[blk * 3 + 1] = (word >> 8) & 0xFF
        out[blk * 3 + 2] = (word >> 16) & 0xFF
    return bytes(out)


def pack_mlx_affine_8bit(codes: Sequence[int]) -> bytes:
    """Pack 8-bit unsigned codes (1 byte per element, MLX uint32 compatible)."""
    if len(codes) % 4 != 0:
        raise ValueError("8-bit code count must be a multiple of 4")
    return bytes((c & 0xFF) for c in codes)


_E2M1_TABLE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def e2m1_nibble_to_f32(q: int) -> float:
    return _E2M1_TABLE[q & 0x0F]


def ue8m0_byte_to_f32(b: int) -> float:
    b &= 0xFF
    if b == 0xFF:
        return math.nan
    return math.ldexp(1.0, b - 127)


def e4m3_byte_to_f32(b: int) -> float:
    """Decode OCP FP8 E4M3 (S.E4.M3, bias=7, NaN=0x7F/0xFF, no Inf)."""
    b &= 0xFF
    if (b & 0x7F) == 0x7F:
        return math.nan
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0x0F
    mant = b & 0x07
    if exp == 0:
        return sign * math.ldexp(float(mant), -9)
    return sign * math.ldexp(1.0 + (mant / 8.0), exp - 7)


def f32_to_e4m3fn_byte(x: float) -> int:
    """Encode finite float to OCP FP8 E4M3FN byte (`S.E4.M3`, bias=7, max=448.0, NaN=0x7F/0xFF)."""
    if not math.isfinite(x):
        return 0x7F
    if x == 0.0:
        return 0x80 if math.copysign(1.0, x) < 0.0 else 0x00
    sign = 0x80 if x < 0.0 else 0x00
    ax = abs(x)
    if ax >= 448.0:
        return sign | 0x7E
    if ax < 0.015625:
        q = int(round(ax * 512.0))
        if q <= 0:
            return sign
        if q >= 8:
            return sign | 0x08
        return sign | q
    mant_f, exp_u = math.frexp(ax)
    e = exp_u - 1
    m = int(round((mant_f * 2.0 - 1.0) * 8.0))
    if m >= 8:
        m = 0
        e += 1
    exp_bits = e + 7
    if exp_bits >= 15 and m >= 7:
        return sign | 0x7E
    if exp_bits > 15:
        return sign | 0x7E
    return sign | (exp_bits << 3) | m


def quantize_row_f32_to_e4m3_row(vals: Sequence[float]) -> tuple[bytes, float]:
    """Quantize a row of floats into `(e4m3_bytes, row_f32_scale)`."""
    max_abs = max((abs(v) for v in vals), default=0.0)
    scale = (max_abs / 448.0) if max_abs > 1e-12 else 1.0
    inv = 1.0 / scale
    return bytes(f32_to_e4m3fn_byte(v * inv) for v in vals), scale


_E4M3_LUT: tuple[float, ...] = tuple(e4m3_byte_to_f32(i) for i in range(256))
_UE8M0_LUT: tuple[float, ...] = tuple(ue8m0_byte_to_f32(i) for i in range(256))


def _strip_model_prefix(raw_name: str) -> str:
    for prefix in ("model.language_model.", "model.llm.", "language_model.model.", "language_model.", "model."):
        if raw_name.startswith(prefix):
            return raw_name[len(prefix) :]
    return raw_name


_LONG_NAME_COMPACT_RULES: tuple[tuple[str, str], ...] = (
    ("hyper_connection_mixer.", "hc_mixer."),
    (".attn_hyper_connection.", ".attn_hc."),
    (".mlp_hyper_connection.", ".mlp_hc."),
    (".input_mix_weight_down.", ".in_mix_dn."),
    (".input_mix_weight_up.", ".in_mix_up."),
    (".block_inject_weight.", ".blk_inj."),
    (".self_attn.indexer.index_qk_proj.", ".attn.idx.qk_proj."),
    (".self_attn.indexer.q_layernorm.", ".attn.idx.q_norm."),
    (".self_attn.indexer.k_layernorm.", ".attn.idx.k_norm."),
    (".ple.ple_embedding.layer_multipliers", ".ple.layer_mult"),
    (".ple.ple_embedding.ngram_heads_offsets", ".ple.ngram_offsets"),
    (".ple.ple_embedding.ngram_heads_vocab_sizes", ".ple.ngram_vocabs"),
    (".ple_embedding.", "."),
    (".ngram_heads_offsets", ".ngram_offsets"),
    (".ngram_heads_vocab_sizes", ".ngram_vocabs"),
    (".layer_multipliers", ".layer_mult"),
    (".compressed_indexer.", ".cidx."),
    (".head_weight_projection.", ".hw_proj."),
    (".key_gate_projection.", ".kg_proj."),
    (".key_value_projection.", ".kv_proj."),
    (".query_projection.", ".q_proj."),
    (".token_key_projection.", ".tk_proj."),
    (".kv_a_proj_with_mqa.", ".kv_a_mqa."),
    (".engram_layer.", ".engram."),
    (".lookup_per_head_norms.", ".head_norms."),
    (".unified_per_hc_scales.", ".hc_scales."),
    (".mlp_message_gate.", ".mlp_mgate."),
    (".message_gate_proj.", ".mgate_proj."),
    (".self_attn.", ".attn."),
    (".shared_experts.", ".shared_exp."),
    (".input_layernorm.", ".in_norm."),
    (".post_attention_layernorm.", ".post_norm."),
    ("mtp_layers.", "mtp."),
    ("dfly_layers.", "dfly."),
)

_INKLING2_CTRL_RE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?(?:llm\.)?layers\.(\d+)\."
    r"(?:controllers\.(attn|k|v|mlp)\.(down|forget|gain)_weight"
    r"|(q|k|v|mlp)_message_gate\.(down_proj|forget_gate_proj|message_gate_proj)(?:\.weight)?)$"
)
_INKLING2_SCONV_RE = re.compile(
    r"^layers\.(\d+)\.(?:"
    r"post_(attention|mlp)_convolution\.conv1d\.weight"
    r"|self_attn\.(k|v)_sconv\.(conv1d\.weight|static_weight|weight_proj)"
    r")$"
)
_INKLING2_ATTN_RES_RE = re.compile(r"^layers\.(\d+)\.(attention|mlp)_input\.query$")


def is_inkling2_controller_tensor(name: str) -> bool:
    """Return True if `name` is an Inkling 2 `gate_rank=64` loop controller matrix."""
    return _INKLING2_CTRL_RE.match(name) is not None


def _canonicalize_m82_backbone_name(name: str) -> str:
    mc = _INKLING2_CTRL_RE.match(name)
    if mc is not None:
        layer = mc.group(1)
        if mc.group(2) is not None:
            return f"layers.{layer}.controllers.{mc.group(2)}.{mc.group(3)}_weight"
        slot_map = {"q": "attn", "k": "k", "v": "v", "mlp": "mlp"}
        part_map = {
            "down_proj": "down_weight",
            "forget_gate_proj": "forget_weight",
            "message_gate_proj": "gain_weight",
        }
        return f"layers.{layer}.controllers.{slot_map[mc.group(4)]}.{part_map[mc.group(5)]}"

    ms = _INKLING2_SCONV_RE.match(name)
    if ms is not None:
        layer = ms.group(1)
        if ms.group(2) is not None:
            kind = "attn" if ms.group(2) == "attention" else "mlp"
            return f"layers.{layer}.{kind}_sconv.weight"
        kv = ms.group(3)
        sub = {"conv1d.weight": "weight", "static_weight": "static_w", "weight_proj": "weight_proj"}[ms.group(4)]
        return f"layers.{layer}.attn.{kv}_sconv.{sub}"

    ma = _INKLING2_ATTN_RES_RE.match(name)
    if ma is not None:
        layer = ma.group(1)
        sub = "attn_queries" if ma.group(2) == "attention" else "mlp_queries"
        return f"attn_res.{sub}.{layer}"
    if name == "final_input.query":
        return "attn_res.output_query"
    if name == "unembed.weight":
        return "lm_head.weight"
    if name == "unembed":
        return "lm_head"
    return name


def _normalize_name(raw_name: str) -> str:
    if is_vision_tensor(raw_name):
        name = normalize_vision_name(raw_name)
    else:
        stripped = _strip_model_prefix(raw_name)
        name = _canonicalize_m82_backbone_name(stripped)
        if raw_name.startswith("model.llm.") and name == "embed.weight":
            name = "embed_tokens.weight"
    if len(name.encode("utf-8")) > 47:
        compacted = name
        for src_seg, dst_seg in _LONG_NAME_COMPACT_RULES:
            if len(compacted.encode("utf-8")) <= 47:
                break
            compacted = compacted.replace(src_seg, dst_seg)
        name = compacted
    encoded = name.encode("utf-8")
    if len(encoded) == 0 or len(encoded) > 47:
        raise ValueError(f"AMF tensor name must be 1..47 UTF-8 bytes, got {len(encoded)}: {name!r}")
    return name


def _classify_tensor(
    name: str,
    cold_experts: set[tuple[int, int]],
    all_experts_cold: bool,
) -> tuple[TensorKind, int, int]:
    layer_match = _LAYER_RE.search(name) or _VISION_BLOCK_RE.search(name)
    if layer_match is None and name.startswith("mtp."):
        layer_match = re.match(r"^mtp\.(\d+)\.", name)
    layer = int(layer_match.group(1)) if layer_match else NO_LAYER
    expert_match = _EXPERT_RE.search(name)
    expert = int(expert_match.group(1)) if expert_match else NO_EXPERT

    lower = name.lower()
    if _LOOP_STEP_RE.search(name) or ".loop_adapter." in lower:
        return (TensorKind.LOOP_ADAPTER, layer, NO_EXPERT)
    _is_dense_ple = ".ple." in lower and any(
        tok in lower
        for tok in (
            ".ple.conv1d",
            ".ple.key_proj",
            ".ple.value_proj",
            ".ple.norm_conv",
            ".ple.norm_key",
            ".ple.norm_query",
            ".ple.layer_mult",
            ".ple.ngram_offsets",
            ".ple.ngram_vocabs",
            ".ple_embedding.layer_multipliers",
            ".ple_embedding.ngram_heads_offsets",
            ".ple_embedding.ngram_heads_vocab_sizes",
        )
    )
    if (
        lower.startswith(("ple.", "engram.", "sparse_table."))
        or (".ple." in lower and not _is_dense_ple)
        or "multi_head_embedding" in lower
        or "per_layer_token_embd" in lower
        or lower.endswith(".lookup_table")
        or (
            ".engram." in lower
            and "key_projs" not in lower
            and "value_proj" not in lower
            and "wkv" not in lower
            and "q_weight" not in lower
            and "k_weight" not in lower
            and "norm" not in lower
            and "hc_scales" not in lower
            and "unified_pre_gamma" not in lower
        )
    ):
        return (TensorKind.SPARSE_TABLE, layer, NO_EXPERT)
    if expert != NO_EXPERT:
        is_cold = all_experts_cold or (layer, expert) in cold_experts
        return (TensorKind.COLD_EXPERT if is_cold else TensorKind.HOT_EXPERT, layer, expert)
    if (
        ".shared_expert." in lower
        or ".shared_experts." in lower
        or ".shared_exp." in lower
        or lower.endswith(".shared_expert_gate")
        or lower.endswith(".shared_expert_gate.weight")
    ):
        return (TensorKind.SHARED_EXPERT, layer, NO_EXPERT)
    if "norm" in lower or ".ln_" in lower or lower.endswith(".unified_pre_gamma"):
        return (TensorKind.NORM, layer, expert)
    if "lm_head" in lower or lower in ("head", "head.weight", "unembed", "unembed.weight"):
        return (TensorKind.LM_HEAD, layer, expert)
    if "embed" in lower and ".self_attn." not in lower and ".attn." not in lower:
        return (TensorKind.EMBEDDING, layer, expert)
    if (
        lower.endswith(".mlp.gate")
        or lower.endswith(".mlp.gate.weight")
        or lower.endswith(".mlp.gate.bias")
        or lower.endswith(".mlp.gate.e_score_correction_bias")
        or lower.endswith(".ffn.gate")
        or lower.endswith(".ffn.gate.weight")
        or lower.endswith(".ffn.gate.bias")
        or lower.endswith(".ffn.gate.bias_vl")
        or lower.endswith(".ffn.gate.e_score_correction_bias")
    ):
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
    flags = FLAG_EMBEDDED_METADATA if any(m.kind == TensorKind.METADATA for m in metas) else 0

    header_zero_crc = HEADER_STRUCT.pack(
        AMF_MAGIC,
        AMF_VERSION,
        flags,
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
        flags,
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
QUANT_PROFILES = ("uniform", "gsq-rco")


class RangeStreamReader(Protocol):
    """Random-access byte-range source for bounded-disk safetensors stream packing."""

    def size(self) -> int: ...

    def read_range(self, offset: int, length: int) -> bytes: ...

    def close(self) -> None: ...


class LocalFileRangeReader:
    """`F_NOCACHE` `os.pread`-backed RangeStreamReader for local shard files."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self._fh = self.path.open("rb", buffering=0)
        try:
            _set_nocache(self._fh.fileno())
            self._size = self.path.stat().st_size
        except BaseException:
            self._fh.close()
            raise

    def size(self) -> int:
        return self._size

    def read_range(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self._size:
            raise ValueError(f"range [{offset}, {offset + length}) out of bounds for {self.path} ({self._size} B)")
        if length == 0:
            return b""
        return _read_fd_exact(self._fh, offset, length)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class MemoryRangeReader:
    """In-memory RangeStreamReader (useful for stream testing and buffers)."""

    def __init__(self, data: bytes | bytearray | memoryview, name: str = "<memory>") -> None:
        self._mv = memoryview(data)
        self.name = name

    def size(self) -> int:
        return len(self._mv)

    def read_range(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > len(self._mv):
            raise ValueError(f"range [{offset}, {offset + length}) out of bounds for {self.name} ({len(self._mv)} B)")
        return bytes(self._mv[offset : offset + length])

    def close(self) -> None:
        pass


class HttpRangeStreamReader:
    """HTTP `Range` stream reader using stdlib `urllib.request` with bounded chunks."""

    _ALLOWED_SCHEMES = ("http", "https")

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_chunk_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme.lower() not in self._ALLOWED_SCHEMES or not parsed.netloc:
            raise ValueError(
                f"HttpRangeStreamReader only supports http/https URLs with host, got {url!r}"
            )
        self.url = url
        self._headers = dict(headers or {})
        self._timeout = float(timeout)
        self._max_chunk = int(max_chunk_bytes)
        self._size: int | None = None

    def _validate_response_url(self, resp: object) -> None:
        geturl = getattr(resp, "geturl", None)
        if callable(geturl):
            final_url = str(geturl())
            scheme = urllib.parse.urlsplit(final_url).scheme.lower()
            if scheme not in self._ALLOWED_SCHEMES:
                raise ValueError(f"HttpRangeStreamReader disallowed redirect scheme in {final_url!r}")

    def size(self) -> int:
        if self._size is not None:
            return self._size
        req = urllib.request.Request(self.url, headers=self._headers, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                self._validate_response_url(resp)
                cl = resp.headers.get("Content-Length")
                if cl is not None:
                    self._size = int(cl)
                    return self._size
        except urllib.error.HTTPError:
            pass
        # Probe with 1-byte Range GET (`Content-Range: bytes 0-0/<total>`).
        probe_headers = dict(self._headers)
        probe_headers["Range"] = "bytes=0-0"
        req2 = urllib.request.Request(self.url, headers=probe_headers, method="GET")
        with urllib.request.urlopen(req2, timeout=self._timeout) as resp:
            self._validate_response_url(resp)
            cr = resp.headers.get("Content-Range")
            if cr and "/" in cr:
                total_str = cr.rsplit("/", 1)[-1].strip()
                if total_str != "*":
                    self._size = int(total_str)
                    return self._size
            cl = resp.headers.get("Content-Length")
            if cl is not None and resp.status == 200:
                self._size = int(cl)
                return self._size
        raise ValueError(f"HTTP server at {self.url!r} did not report Content-Length or Content-Range")

    def read_range(self, offset: int, length: int) -> bytes:
        total = self.size()
        if offset < 0 or length < 0 or offset + length > total:
            raise ValueError(f"range [{offset}, {offset + length}) out of bounds for {self.url!r} ({total} B)")
        if length == 0:
            return b""
        parts: list[bytes] = []
        done = 0
        while done < length:
            take = min(self._max_chunk, length - done)
            start = offset + done
            end = start + take - 1
            hdrs = dict(self._headers)
            hdrs["Range"] = f"bytes={start}-{end}"
            req = urllib.request.Request(self.url, headers=hdrs, method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                self._validate_response_url(resp)
                status = getattr(resp, "status", 206)
                raw = resp.read()
                if status == 200:
                    if len(raw) >= start + take:
                        raw = raw[start : start + take]
                    else:
                        raise ValueError(f"short HTTP 200 response for {self.url!r}")
                elif status != 206:
                    raise ValueError(f"unexpected HTTP status {status} for Range {start}-{end} on {self.url!r}")
            if len(raw) != take:
                raise ValueError(
                    f"HTTP Range {start}-{end} on {self.url!r} returned {len(raw)} bytes, expected {take}"
                )
            parts.append(raw)
            done += take
        return b"".join(parts) if len(parts) > 1 else parts[0]

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class ShardStreamSource:
    """Named shard stream descriptor for bounded-disk `.amf` conversion."""

    source_id: str | pathlib.Path
    reader_factory: Callable[[], RangeStreamReader]


@dataclass(frozen=True)
class _ShardTensor:
    """One tensor as described by its shard's JSON header (no data loaded)."""

    name: str
    path: pathlib.Path | str
    dtype: str
    shape: tuple[int, ...]
    offset: int  # absolute byte offset of the tensor data inside `path`
    length: int


@dataclass(frozen=True)
class _ByteRange:
    path: pathlib.Path | str
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
    params_mode: str = "affine_scale_bias"  # affine_scale_bias, fp8_block, fp8_row, fp8_tensor, mxfp8, nvfp4
    outer_scale: _ByteRange | None = None
    outer_scale_denom: _ByteRange | None = None
    outer_scale_const: float = 1.0
    outer_scale_denom_const: float = 1.0
    outer_scale_divisor_mode: bool = False
    outer_scale_expert_index: int = -1
    scale_tile_shape: tuple[int, int] | None = None
    src_total_rows: int | None = None
    row_gather_runs: tuple[tuple[int, int], ...] | None = None
    mimo_qkv_slice: tuple[Any, int, int] | None = None  # `(MimoQkvSpec, out_start, out_count)`
    mlx_kv_b_pair: tuple[Any, ...] | None = None  # `(n_heads, kv_lora, qk_nope, v_hd, eq_w, eq_s, eq_b, eq_bits, eq_gs, uo_w, uo_s, uo_b, uo_bits, uo_gs)`


@dataclass(frozen=True)
class ResolvedCheckpoint:
    shard_paths: tuple[pathlib.Path, ...]
    config_path: pathlib.Path | None


def _resolve_contained_file(
    base_dir: pathlib.Path,
    rel_name: str,
    *,
    label: str = "file",
    missing_is_not_found: bool = True,
) -> pathlib.Path:
    """Resolve `rel_name` strictly inside `base_dir.resolve()`, rejecting traversal, symlinks escaping root, and special files."""
    if not isinstance(rel_name, str) or not rel_name:
        raise ValueError(f"{label} name must be a non-empty string")
    if "\x00" in rel_name or "\\" in rel_name:
        raise ValueError(f"unsafe {label} path {rel_name!r}")
    pure_posix = pathlib.PurePosixPath(rel_name)
    raw_path = pathlib.Path(rel_name)
    if (
        pure_posix.is_absolute()
        or raw_path.is_absolute()
        or os.path.isabs(rel_name)
        or ".." in pure_posix.parts
        or ".." in raw_path.parts
    ):
        raise ValueError(f"{label} path {rel_name!r} traverses outside {base_dir}")

    root_resolved = base_dir.resolve()
    candidate = base_dir / raw_path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"{label} path {rel_name!r} resolves outside {root_resolved}")
    if not resolved.exists():
        if missing_is_not_found:
            raise FileNotFoundError(f"{label} listed in checkpoint is missing: {candidate}")
        raise ValueError(f"{label} path {rel_name!r} does not exist")
    st = resolved.stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{label} path {rel_name!r} is not a regular file")
    return resolved


def resolve_checkpoint(input_path: pathlib.Path) -> ResolvedCheckpoint:
    """Resolve a `.safetensors` file, an `index.json` file, or a checkpoint directory."""
    input_path = pathlib.Path(input_path)
    if input_path.is_dir():
        index_cand = input_path / "model.safetensors.index.json"
        single_cand = input_path / "model.safetensors"
        if index_cand.exists() or index_cand.is_symlink():
            index_path = _resolve_contained_file(input_path, "model.safetensors.index.json", label="index")
            return resolve_checkpoint(index_path)
        if single_cand.exists() or single_cand.is_symlink():
            single_path = _resolve_contained_file(input_path, "model.safetensors", label="checkpoint")
            return resolve_checkpoint(single_path)
        raise FileNotFoundError(
            f"{input_path} contains neither model.safetensors.index.json nor model.safetensors"
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"input checkpoint not found: {input_path}")

    st_in = input_path.resolve().stat()
    if not stat.S_ISREG(st_in.st_mode):
        raise ValueError(f"input checkpoint is not a regular file: {input_path}")

    parent = input_path.parent
    config_cand = parent / "config.json"
    config: pathlib.Path | None = None
    if config_cand.exists() or config_cand.is_symlink():
        config = _resolve_contained_file(parent, "config.json", label="config")

    if input_path.name.endswith(".json"):
        index = json.loads(input_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{input_path} has no non-empty 'weight_map'")
        shard_names: list[str] = []
        seen_names: set[str] = set()
        for raw_val in weight_map.values():
            if not isinstance(raw_val, str) or not raw_val:
                raise ValueError(f"{input_path} contains invalid shard filename {raw_val!r}")
            if raw_val not in seen_names:
                seen_names.add(raw_val)
                shard_names.append(raw_val)
        shard_names.sort()
        shards = [
            _resolve_contained_file(parent, shard, label=f"shard listed in {input_path.name}")
            for shard in shard_names
        ]
        return ResolvedCheckpoint(tuple(shards), config)
    return ResolvedCheckpoint((input_path,), config)


def _read_shard_header_from_reader(source_id: pathlib.Path | str, reader: RangeStreamReader) -> list[_ShardTensor]:
    """Read only the JSON header from any `RangeStreamReader` (kilobytes, never tensor data)."""
    file_size = reader.size()
    if file_size < 8:
        raise ValueError(f"truncated safetensors header in {source_id}")
    hdr_len_bytes = reader.read_range(0, 8)
    if len(hdr_len_bytes) != 8:
        raise ValueError(f"truncated safetensors header in {source_id}")
    (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
    if hdr_len > file_size - 8 or hdr_len > (256 << 20):
        raise ValueError(f"implausible safetensors header length {hdr_len} in {source_id}")
    hdr_raw = reader.read_range(8, hdr_len)
    if len(hdr_raw) != hdr_len:
        raise ValueError(f"truncated safetensors JSON in {source_id}")
    header = json.loads(hdr_raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be a JSON object in {source_id}")
    data_base = 8 + hdr_len
    out: list[_ShardTensor] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"tensor {name} entry in {source_id} must be a JSON object")
        dtype = str(spec["dtype"])
        raw_shape = spec["shape"]
        if not isinstance(raw_shape, list):
            raise ValueError(f"tensor {name} shape in {source_id} must be a list")
        if any(isinstance(x, bool) for x in raw_shape):
            raise ValueError(f"tensor {name} shape in {source_id} contains invalid bool dimension")
        shape = tuple(int(x) for x in raw_shape)
        if any(d < 0 for d in shape):
            raise ValueError(f"tensor {name} in {source_id} has negative dimension in shape {list(shape)}")
        raw_offsets = spec["data_offsets"]
        if not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
            raise ValueError(f"tensor {name} in {source_id} has invalid data_offsets {raw_offsets!r}")
        if any(isinstance(x, bool) for x in raw_offsets):
            raise ValueError(f"tensor {name} in {source_id} has invalid bool data_offsets")
        start, end = int(raw_offsets[0]), int(raw_offsets[1])
        elem = _SAFETENSORS_DTYPE_BYTES.get(dtype)
        if elem is None:
            raise ValueError(f"unsupported safetensors dtype {dtype} for {name} in {source_id}")
        numel = 1
        for d in shape:
            numel *= d
        if start < 0 or end < start or end - start != numel * elem:
            raise ValueError(
                f"tensor {name} in {source_id}: data_offsets {start}..{end} do not match {dtype}{list(shape)}"
            )
        if data_base + end > file_size:
            raise ValueError(f"tensor {name} in {source_id} extends past end of file")
        out.append(_ShardTensor(name, source_id, dtype, shape, data_base + start, end - start))
    return out


def _read_shard_header(path: pathlib.Path) -> list[_ShardTensor]:
    """Read only the JSON header of a safetensors shard (kilobytes, never tensor data)."""
    reader = LocalFileRangeReader(path)
    try:
        return _read_shard_header_from_reader(path, reader)
    finally:
        reader.close()


def _collect_tensors(shards: Sequence[pathlib.Path]) -> dict[str, _ShardTensor]:
    tensors: dict[str, _ShardTensor] = {}
    for shard in shards:
        for t in _read_shard_header(shard):
            if t.name in tensors:
                raise ValueError(f"tensor {t.name!r} appears in both {tensors[t.name].path} and {shard}")
            tensors[t.name] = t
    return tensors


def _collect_tensors_from_sources(
    sources: Iterable[ShardStreamSource],
) -> tuple[dict[str, _ShardTensor], dict[pathlib.Path | str, Callable[[], RangeStreamReader]]]:
    tensors: dict[str, _ShardTensor] = {}
    factories: dict[pathlib.Path | str, Callable[[], RangeStreamReader]] = {}
    for src in sources:
        if src.source_id in factories:
            raise ValueError(f"duplicate shard source_id: {src.source_id!r}")
        factories[src.source_id] = src.reader_factory
        reader = src.reader_factory()
        try:
            for t in _read_shard_header_from_reader(src.source_id, reader):
                if t.name in tensors:
                    raise ValueError(f"tensor {t.name!r} appears in both {tensors[t.name].path} and {src.source_id}")
                tensors[t.name] = t
        finally:
            reader.close()
    return tensors, factories


def _natural_key(name: str) -> tuple[tuple[int, int, str], ...]:
    return tuple((0, int(tok), "") if tok.isdigit() else (1, 0, tok) for tok in name.split("."))


_AFFINE_BITS_DTYPE = {
    2: DType.Q2_AFFINE,
    3: DType.Q3_AFFINE,
    4: DType.Q4_AFFINE,
    5: DType.Q5_AFFINE,
    6: DType.Q6_AFFINE,
    8: DType.Q8_AFFINE,
}


def _quant_bits_for(prefix: str, packed_cols: int, groups: int, group_size: int) -> tuple[int, DType]:
    k = groups * group_size
    bits_num = packed_cols * 32
    if k == 0 or bits_num % k != 0:
        raise ValueError(f"cannot infer bit width for {prefix}: packed={packed_cols}, k={k}")
    bits = bits_num // k
    dt = _AFFINE_BITS_DTYPE.get(bits)
    if dt is not None:
        return bits, dt
    raise ValueError(f"unsupported bit width {bits} for {prefix} (expected 2, 3, 4, 5, 6, or 8)")


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


def _is_loop_r0_alpha(name: str) -> bool:
    m = _LOOP_STEP_RE.search(name)
    if m is None or int(m.group(1)) != 0:
        return False
    return name.endswith(".alpha") or name.endswith(".alpha.weight")


def _should_sme2_pack(
    kind: TensorKind,
    dtype: DType,
    shape: tuple[int, ...],
    sme2_pack: str,
    *,
    name: str = "",
    sme2_pack_loop_adapters: bool = False,
) -> bool:
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 or shape[1] % SME2_GROUP_K != 0:
        return False
    if dtype not in (DType.F16, DType.BF16, DType.Q4_AFFINE, DType.Q3_AFFINE, DType.Q2_AFFINE):
        return False
    if sme2_pack_loop_adapters and kind == TensorKind.LOOP_ADAPTER:
        return True
    if sme2_pack == "none":
        return False
    if sme2_pack not in SME2_PACK_MODES:
        raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
    if sme2_pack == "controllers":
        return bool(name) and is_inkling2_controller_tensor(name)
    if sme2_pack == "routers":
        return kind == TensorKind.ROUTER or (bool(name) and is_inkling2_controller_tensor(name))
    if sme2_pack == "eh-proj":
        return name in ("layers.0.eh_proj.weight", "eh_proj.weight", "layers.0.eh_proj", "eh_proj")
    return kind in (
        TensorKind.ROUTER,
        TensorKind.DENSE,
        TensorKind.HOT_EXPERT,
        TensorKind.COLD_EXPERT,
        TensorKind.LM_HEAD,
        TensorKind.SHARED_EXPERT,
        TensorKind.LOOP_ADAPTER,
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
    sme2_pack_loop_adapters: bool = True,
    params_mode: str = "affine_scale_bias",
    params_bytes_override: int | None = None,
    outer_scale: _ByteRange | None = None,
    outer_scale_denom: _ByteRange | None = None,
    outer_scale_const: float = 1.0,
    outer_scale_denom_const: float = 1.0,
    outer_scale_divisor_mode: bool = False,
    outer_scale_expert_index: int = -1,
    scale_tile_shape: tuple[int, int] | None = None,
    src_total_rows: int | None = None,
    row_gather_runs: tuple[tuple[int, int], ...] | None = None,
    mimo_qkv_slice: tuple[Any, int, int] | None = None,
    mlx_kv_b_pair: tuple[Any, ...] | None = None,
    name_already_normalized: bool = False,
) -> _PlannedEntry:
    if not name_already_normalized:
        name = _normalize_name(name)
    if not (1 <= len(shape) <= 4):
        raise ValueError(f"tensor rank must be 1..4 for {name!r}, got {len(shape)}")
    use_sme2 = (
        not is_vision_tensor(name)
        and dtype in (DType.F16, DType.BF16, DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE)
        and (dtype not in AFFINE_DTYPES or group_size == SME2_GROUP_K)
        and row_gather_runs is None
        and mimo_qkv_slice is None
        and mlx_kv_b_pair is None
        and not (data_convert is not None and data_convert.endswith("_to_bf16"))
        and _should_sme2_pack(
            kind,
            dtype,
            shape,
            sme2_pack,
            name=name,
            sme2_pack_loop_adapters=sme2_pack_loop_adapters,
        )
    )
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
        if row_gather_runs is not None:
            eff_src_rows = src_total_rows or shape[0]
            data_bytes = shape[0] * (data.length // eff_src_rows)
            if data_convert in ("bf16_to_f32", "f16_to_f32"):
                data_bytes *= 2
        elif data_convert == "d2t_i64_to_u32":
            data_bytes = (data.length // 8) * 4
        elif data_convert in ("bf16_to_f32", "f16_to_f32"):
            data_bytes = data.length * 2
        elif data_convert == "hf_mxfp4_to_gguf_mxfp4":
            data_bytes = (data.length // 16) * 17
        elif data_convert is not None and (
            (data_convert.startswith("affine") and data_convert.endswith("_to_bf16"))
            or data_convert in ("fp8_block_to_bf16", "mlx_kv_b_bf16")
        ):
            data_bytes = shape[0] * shape[1] * 2
        elif mimo_qkv_slice is not None:
            _spec, _out_s, out_cnt = mimo_qkv_slice
            if dtype == DType.FP8_E4M3_ROW:
                data_bytes = out_cnt * _spec.k_cols
            elif dtype == DType.F32:
                data_bytes = out_cnt * _spec.k_cols * 4
            else:
                data_bytes = out_cnt * _spec.k_cols * 2
        else:
            data_bytes = data.length  # f16<->bf16, i32->u32, packed nibbles and raw copies preserve length
        if params_bytes_override is not None:
            params_offset = _align_up(data_bytes, 64) if params_bytes_override > 0 else 0
            params_bytes = params_bytes_override
            content = params_offset + params_bytes if params_bytes > 0 else data_bytes
        elif scales is not None:
            params_offset = _align_up(data_bytes, 64)
            if row_gather_runs is not None and src_total_rows:
                groups = (scales.length // _SAFETENSORS_DTYPE_BYTES[scales.dtype]) // src_total_rows
                params_bytes = shape[0] * groups * 4
            else:
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
        params_mode=params_mode,
        outer_scale=outer_scale,
        outer_scale_denom=outer_scale_denom,
        outer_scale_const=outer_scale_const,
        outer_scale_denom_const=outer_scale_denom_const,
        outer_scale_divisor_mode=outer_scale_divisor_mode,
        outer_scale_expert_index=outer_scale_expert_index,
        scale_tile_shape=scale_tile_shape,
        src_total_rows=src_total_rows,
        row_gather_runs=row_gather_runs,
        mimo_qkv_slice=mimo_qkv_slice,
        mlx_kv_b_pair=mlx_kv_b_pair,
    )


def _expert_kind(layer: int, expert: int, cold: set[tuple[int, int]], all_cold: bool) -> TensorKind:
    return TensorKind.COLD_EXPERT if all_cold or (layer, expert) in cold else TensorKind.HOT_EXPERT


def _slice_range(t: _ShardTensor, index: int, count: int) -> _ByteRange:
    step = t.length // count
    return _ByteRange(t.path, t.offset + index * step, step, t.dtype)


def _slice_sub_half_range(t: _ShardTensor, index: int, count: int, half_idx: int) -> _ByteRange:
    exp_len = t.length // count
    half_len = exp_len // 2
    return _ByteRange(t.path, t.offset + index * exp_len + half_idx * half_len, half_len, t.dtype)


def _full_range(t: _ShardTensor) -> _ByteRange:
    return _ByteRange(t.path, t.offset, t.length, t.dtype)


def _split_quant_weight_base(weight_name: str) -> tuple[str, str] | None:
    """Split a candidate quantized weight key into `(canonical_prefix, suffix_style)`."""
    if weight_name.endswith(".weight_packed"):
        return weight_name[: -len(".weight_packed")], "dot_weight"
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")], "dot_weight"
    if weight_name.endswith("_weight"):
        return weight_name[: -len("_weight")], "underscore_weight"
    return None


def _pick_companion(tensors: dict[str, _ShardTensor], candidates: Sequence[str]) -> _ShardTensor | None:
    for key in candidates:
        t = tensors.get(key)
        if t is not None:
            return t
    return None


def _resolve_stacked_expert_meta(raw_prefix: str) -> tuple[int, str] | None:
    sw = _SWITCH_MLP_RE.match(raw_prefix)
    if sw is not None:
        return int(sw.group(1)), sw.group(2)
    ex = _EXPERTS_STACK_RE.match(raw_prefix)
    if ex is not None:
        proj_map = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
        raw_proj = ex.group(2)
        return int(ex.group(1)), proj_map.get(raw_proj, raw_proj)
    return None


def _plan_nvfp4_slice(
    *,
    name: str,
    kind: TensorKind,
    layer: int,
    expert: int,
    n_rows: int,
    k_packed_cols: int,
    w_dtype: str,
    w_range: _ByteRange,
    s_range: _ByteRange,
    s_shape_2d: tuple[int, ...],
    outer_range: _ByteRange | None,
    denom_range: _ByteRange | None,
    outer_const: float,
    denom_const: float,
    divisor_mode: bool,
    outer_expert_idx: int,
    prefix_desc: str,
) -> _PlannedEntry:
    if w_dtype == "U8":
        k = k_packed_cols * 2
    elif w_dtype in ("U32", "I32"):
        k = k_packed_cols * 8
    else:
        raise ValueError(f"unsupported NVFP4 weight dtype {w_dtype} for {prefix_desc}")
    if k == 0 or k % 16 != 0:
        raise ValueError(f"NVFP4 tensor {prefix_desc} K={k} must be positive and divisible by 16")
    if len(s_shape_2d) != 2 or s_shape_2d[0] != n_rows or s_shape_2d[1] != k // 16:
        raise ValueError(
            f"NVFP4 micro-scale shape mismatch for {prefix_desc}: weight={(n_rows, k)}, scales={list(s_shape_2d)}"
        )
    params_bytes = _align_up(n_rows * (k // 16), 4) + 4
    return _make_entry(
        name,
        kind,
        DType.NVFP4_G16,
        4,
        16,
        layer,
        expert,
        (n_rows, k),
        w_range,
        None,
        s_range,
        None,
        sme2_pack="none",
        params_mode="nvfp4",
        params_bytes_override=params_bytes,
        outer_scale=outer_range,
        outer_scale_denom=denom_range,
        outer_scale_const=outer_const,
        outer_scale_denom_const=denom_const,
        outer_scale_divisor_mode=divisor_mode,
        outer_scale_expert_index=outer_expert_idx,
    )


def _plan_fp8_or_mxfp8_slice(
    *,
    name: str,
    kind: TensorKind,
    layer: int,
    expert: int,
    n_rows: int,
    k: int,
    w_range: _ByteRange,
    s_tensor: _ShardTensor | None,
    s_range: _ByteRange | None,
    s_slice_shape: tuple[int, ...],
    prefix_desc: str,
) -> _PlannedEntry:
    if n_rows <= 0 or k <= 0:
        raise ValueError(f"FP8/MXFP8 tensor {prefix_desc} must have positive 2D shape, got {(n_rows, k)}")
    if s_tensor is None or s_range is None:
        return _make_entry(
            name,
            kind,
            DType.FP8_E4M3_TENSOR,
            8,
            0,
            layer,
            expert,
            (n_rows, k),
            w_range,
            None,
            None,
            None,
            sme2_pack="none",
            params_mode="fp8_tensor",
            params_bytes_override=4,
            outer_scale_const=1.0,
        )
    if s_tensor.dtype in ("U8", "F8_E8M0"):
        if k % 32 != 0:
            raise ValueError(f"MXFP8 tensor {prefix_desc} K={k} must be divisible by 32")
        tile_shape: tuple[int, int] | None = None
        nb32 = (n_rows + 31) // 32
        kb32 = k // 32
        if s_slice_shape == (n_rows, kb32):
            tile_shape = None
        elif len(s_slice_shape) == 2 and s_slice_shape == (nb32, kb32):
            tile_shape = s_slice_shape
        else:
            raise ValueError(
                f"MXFP8 UE8M0 scale shape mismatch for {prefix_desc}: weight={(n_rows, k)}, scales={list(s_slice_shape)}"
            )
        return _make_entry(
            name,
            kind,
            DType.MXFP8_G32,
            8,
            32,
            layer,
            expert,
            (n_rows, k),
            w_range,
            None,
            s_range,
            None,
            sme2_pack="none",
            params_mode="mxfp8",
            params_bytes_override=n_rows * kb32,
            scale_tile_shape=tile_shape,
        )
    if s_tensor.dtype not in ("F32", "BF16", "F16"):
        raise ValueError(f"unsupported FP8 scale dtype {s_tensor.dtype} for {s_tensor.name}")
    nb128 = (n_rows + 127) // 128
    kb128 = (k + 127) // 128
    if s_slice_shape in ((), (1,), (1, 1)) and not (n_rows == 1 and len(s_slice_shape) == 1):
        # Single scalar tensor scale (note `(1, 1)` when `nb128 == kb128 == 1` is numerically equivalent to block 1x1;
        # treat 2D `(nb128, kb128)` as block when K > 128 or N > 128, and `(1,)` / `()` as tensor scale).
        if len(s_slice_shape) == 2 and (nb128, kb128) == (1, 1) and "scale_inv" in s_tensor.name:
            pass
        else:
            return _make_entry(
                name,
                kind,
                DType.FP8_E4M3_TENSOR,
                8,
                0,
                layer,
                expert,
                (n_rows, k),
                w_range,
                None,
                s_range,
                None,
                sme2_pack="none",
                params_mode="fp8_tensor",
                params_bytes_override=4,
            )
    if s_slice_shape == (n_rows,) or (s_slice_shape == (n_rows, 1) and kb128 > 1):
        return _make_entry(
            name,
            kind,
            DType.FP8_E4M3_ROW,
            8,
            0,
            layer,
            expert,
            (n_rows, k),
            w_range,
            None,
            s_range,
            None,
            sme2_pack="none",
            params_mode="fp8_row",
            params_bytes_override=n_rows * 4,
        )
    if s_slice_shape == (nb128, kb128):
        return _make_entry(
            name,
            kind,
            DType.FP8_E4M3_BLOCK,
            8,
            128,
            layer,
            expert,
            (n_rows, k),
            w_range,
            None,
            s_range,
            None,
            sme2_pack="none",
            params_mode="fp8_block",
            params_bytes_override=nb128 * kb128 * 4,
        )
    if (
        len(s_slice_shape) == 2
        and s_slice_shape[0] > 0
        and s_slice_shape[1] > 0
        and s_slice_shape[0] <= n_rows
        and s_slice_shape[1] <= k
        and s_slice_shape != (n_rows, 1)
    ):
        return _make_entry(
            name,
            kind,
            DType.BF16,
            0,
            0,
            layer,
            expert,
            (n_rows, k),
            w_range,
            "fp8_block_to_bf16",
            s_range,
            None,
            sme2_pack="none",
            params_bytes_override=0,
            scale_tile_shape=(int(s_slice_shape[0]), int(s_slice_shape[1])),
        )
    raise ValueError(
        f"FP8 scale shape {list(s_slice_shape)} does not match tensor/row/128x128-block for {prefix_desc} {(n_rows, k)}"
    )


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
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    sme2_pack_loop_adapters: bool = True,
    nvfp4_global_scale_is_divisor: bool = False,
    stats_out: dict[str, int] | None = None,
    mimo_config: Mapping[str, Any] | None = None,
    mimo_qkv_tp_degree: int = 1,
    mimo_mtp_standalone: bool = False,
    mimo_dflash_mode: bool = False,
    qwen35_config: Mapping[str, Any] | None = None,
    qwen4_config: Mapping[str, Any] | None = None,
    qwen4_mtp_standalone: bool = False,
    split_stacked_gate_up: bool | None = None,
    glm5_config: Mapping[str, Any] | None = None,
    glm5_mtp_standalone: bool = False,
    glm5_mtp_source_layer: int | None = None,
    glm5_dflash2_mode: bool = False,
    ling3_config: Mapping[str, Any] | None = None,
    ling3_mtp_standalone: bool = False,
    ling3_mtp_source_layer: int | None = None,
) -> list[_PlannedEntry]:
    """Pass 1: plan every output entry from shard headers alone."""
    if group_size not in AFFINE_GROUP_SIZES:
        raise ValueError(f"aminfer supports group_size in {AFFINE_GROUP_SIZES}, got {group_size}")
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

    glm5_backbone_layers: int | None = None
    glm5_model_type = "glm_moe_dsa"
    glm5_layer_types: list[str] | None = None
    if glm5_config is not None:
        from aminfer_tools.glm5 import (
            _effective_glm5_cfg,
            is_glm5_next_config,
            resolve_glm5_next_layer_types,
        )

        eff_g5 = _effective_glm5_cfg(glm5_config)
        n_hl = eff_g5.get("num_hidden_layers")
        if isinstance(n_hl, int) and n_hl > 0:
            glm5_backbone_layers = n_hl
        if is_glm5_next_config(glm5_config):
            glm5_model_type = "glm5_next"
            glm5_layer_types = resolve_glm5_next_layer_types(eff_g5)

    ling3_backbone_layers: int | None = None
    ling3_layer_types: list[str] | None = None
    if ling3_config is not None:
        from aminfer_tools.ling3 import _effective_ling3_cfg, resolve_ling3_layer_types

        eff_l3 = _effective_ling3_cfg(ling3_config)
        n_l3 = eff_l3.get("num_hidden_layers")
        if isinstance(n_l3, int) and n_l3 > 0:
            ling3_backbone_layers = n_l3
        ling3_layer_types = resolve_ling3_layer_types(eff_l3)

    def is_excluded(raw_name: str) -> bool:
        if raw_name in excluded or _normalize_lenient(raw_name) in excluded:
            return True
        if mimo_config is not None:
            from aminfer_tools.mimo_v2 import is_mimo_audio_tensor, is_mimo_mtp_tensor

            if is_mimo_audio_tensor(raw_name):
                return True
            if not mimo_mtp_standalone and is_mimo_mtp_tensor(raw_name):
                return True
        if glm5_config is not None and not glm5_mtp_standalone and glm5_backbone_layers is not None:
            from aminfer_tools.glm5 import is_glm5_mtp_tensor

            if is_glm5_mtp_tensor(raw_name, glm5_backbone_layers):
                return True
        if ling3_config is not None and not ling3_mtp_standalone:
            from aminfer_tools.ling3 import is_ling3_mtp_tensor

            if is_ling3_mtp_tensor(raw_name, ling3_backbone_layers):
                return True
        return False

    def _plan_out_name(raw_name: str) -> str:
        if glm5_dflash2_mode:
            from aminfer_tools.glm5 import normalize_glm5_dflash2_tensor_name

            nm = normalize_glm5_dflash2_tensor_name(raw_name)
        elif glm5_config is not None or glm5_mtp_standalone:
            from aminfer_tools.glm5 import normalize_glm5_tensor_name

            nm = normalize_glm5_tensor_name(
                raw_name,
                model_type=glm5_model_type,
                layer_types=glm5_layer_types,
                mtp_source_layer=glm5_mtp_source_layer,
                mtp_standalone=glm5_mtp_standalone,
            )
        elif ling3_config is not None or ling3_mtp_standalone:
            from aminfer_tools.ling3 import normalize_ling3_tensor_name

            nm = normalize_ling3_tensor_name(
                raw_name,
                layer_types=ling3_layer_types,
                mtp_source_layer=ling3_mtp_source_layer,
                mtp_standalone=ling3_mtp_standalone,
            )
        elif mimo_dflash_mode:
            from aminfer_tools.mimo_v2 import normalize_mimo_dflash_tensor_name

            nm = normalize_mimo_dflash_tensor_name(raw_name)
        elif mimo_config is not None or mimo_mtp_standalone:
            from aminfer_tools.mimo_v2 import normalize_mimo_tensor_name

            nm = normalize_mimo_tensor_name(raw_name, mtp_standalone=mimo_mtp_standalone)
        elif qwen35_config is not None:
            from aminfer_tools.mimo_v2 import normalize_qwen35_tensor_name

            nm = normalize_qwen35_tensor_name(raw_name)
        elif qwen4_config is not None or qwen4_mtp_standalone:
            from aminfer_tools.qwen4_exp import normalize_qwen4_tensor_name

            nm = normalize_qwen4_tensor_name(raw_name, mtp_standalone=qwen4_mtp_standalone)
        else:
            return _normalize_name(raw_name)
        enc = nm.encode("utf-8")
        if len(enc) == 0 or len(enc) > 47:
            raise ValueError(f"AMF tensor name must be 1..47 UTF-8 bytes, got {len(enc)}: {nm!r}")
        return nm

    eff_split_gate_up = (
        (qwen4_config is not None or qwen4_mtp_standalone)
        if split_stacked_gate_up is None
        else bool(split_stacked_gate_up)
    )

    def _glm5_split_specs(out_base: str, shape: Sequence[int]):
        if glm5_config is None or is_vision_tensor(out_base):
            return None
        from aminfer_tools.glm5 import resolve_glm5_next_split_specs

        return resolve_glm5_next_split_specs(out_base, shape, glm5_config)

    is_mimo_or_q35 = (
        mimo_config is not None
        or qwen35_config is not None
        or qwen4_config is not None
        or glm5_config is not None
        or ling3_config is not None
        or mimo_dflash_mode
        or mimo_mtp_standalone
        or qwen4_mtp_standalone
        or glm5_dflash2_mode
        or glm5_mtp_standalone
        or ling3_mtp_standalone
    )

    def _qwen4_split_specs(out_base: str, n_rows: int) -> list[tuple[str, tuple[tuple[int, int], ...]]] | None:
        if qwen4_config is None or is_vision_tensor(out_base):
            return None
        base = out_base[:-7] if out_base.endswith(".weight") else out_base
        if base.endswith(".linear_attn.in_proj_qkv"):
            lk_h = int(qwen4_config.get("linear_num_key_heads") or 16)
            lk_d = int(qwen4_config.get("linear_key_head_dim") or 128)
            lv_h = int(qwen4_config.get("linear_num_value_heads") or 48)
            lv_d = int(qwen4_config.get("linear_value_head_dim") or 128)
            kq = lk_h * lk_d
            vd = lv_h * lv_d
            if n_rows != 2 * kq + vd:
                raise ValueError(f"qwen4_exp {out_base}: rows {n_rows} != 2*{kq}+{vd}")
            pfx = base[: -len(".in_proj_qkv")]
            return [
                (f"{pfx}.q_proj.weight", ((0, kq),)),
                (f"{pfx}.k_proj.weight", ((kq, kq),)),
                (f"{pfx}.v_proj.weight", ((2 * kq, vd),)),
            ]
        og_type = str(qwen4_config.get("output_gate_type") or "").lower()
        has_og = bool(qwen4_config.get("attn_output_gate", False)) or og_type == "sigmoid"
        if base.endswith(".self_attn.q_proj") and has_og:
            qh = int(qwen4_config.get("num_attention_heads") or 24)
            hd = int(qwen4_config.get("head_dim") or 256)
            if n_rows == 2 * qh * hd:
                pfx = base[: -len(".q_proj")]
                return [
                    (f"{pfx}.q_proj.weight", tuple((h_i * 2 * hd, hd) for h_i in range(qh))),
                    (f"{pfx}.q_gate_proj.weight", tuple((h_i * 2 * hd + hd, hd) for h_i in range(qh))),
                ]
        return None

    def _qwen35_split_specs(out_base: str, n_rows: int) -> list[tuple[str, tuple[tuple[int, int], ...]]] | None:
        if qwen35_config is None or is_vision_tensor(out_base):
            return None
        base = out_base[:-7] if out_base.endswith(".weight") else out_base
        if base.endswith(".linear_attn.in_proj_qkv"):
            lk_h = int(qwen35_config.get("linear_num_key_heads") or 16)
            lk_d = int(qwen35_config.get("linear_key_head_dim") or 128)
            lv_h = int(qwen35_config.get("linear_num_value_heads") or 32)
            lv_d = int(qwen35_config.get("linear_value_head_dim") or 128)
            kq = lk_h * lk_d
            vd = lv_h * lv_d
            if n_rows != 2 * kq + vd:
                raise ValueError(f"qwen3_5 {out_base}: rows {n_rows} != 2*{kq}+{vd}")
            pfx = base[: -len(".in_proj_qkv")]
            return [
                (f"{pfx}.q_proj.weight", ((0, kq),)),
                (f"{pfx}.k_proj.weight", ((kq, kq),)),
                (f"{pfx}.v_proj.weight", ((2 * kq, vd),)),
            ]
        if base.endswith(".self_attn.q_proj") and bool(qwen35_config.get("attn_output_gate", True)):
            qh = int(qwen35_config.get("num_attention_heads") or 16)
            hd = int(qwen35_config.get("head_dim") or 256)
            if n_rows == 2 * qh * hd:
                pfx = base[: -len(".q_proj")]
                return [
                    (f"{pfx}.q_proj.weight", tuple((h_i * 2 * hd, hd) for h_i in range(qh))),
                    (f"{pfx}.q_gate_proj.weight", tuple((h_i * 2 * hd + hd, hd) for h_i in range(qh))),
                ]
        return None

    def _emit_mimo_qkv_entries(
        raw_name: str,
        norm_base: str,
        w_t: _ShardTensor,
        s_t: _ShardTensor | None,
    ) -> bool:
        if mimo_config is None:
            return False
        from aminfer_tools.mimo_v2 import resolve_mimo_qkv_spec_for_tensor

        spec = resolve_mimo_qkv_spec_for_tensor(
            raw_name,
            w_t.shape[0],
            w_t.shape[1],
            s_t.shape[0] if s_t is not None else None,
            mimo_config,
            tp_degree=mimo_qkv_tp_degree,
        )
        if spec is None:
            return False
        out_pfx = norm_base
        for suf in (".weight",):
            if out_pfx.endswith(suf):
                out_pfx = out_pfx[: -len(suf)]
        out_pfx = out_pfx[: -len(".qkv_proj")] if out_pfx.endswith(".qkv_proj") else out_pfx
        if s_t is not None:
            if dtype_mode == "auto":
                tgt_dt = DType.FP8_E4M3_ROW
                qb = 8
            elif dtype_mode == "bf16":
                tgt_dt = DType.BF16
                qb = 0
            elif dtype_mode == "f16":
                tgt_dt = DType.F16
                qb = 0
            else:
                tgt_dt = DType.F32
                qb = 0
            dc = None
        else:
            tgt_dt, dc = _unquantized_target(w_t.dtype, dtype_mode, norm_base)
            qb = 0
        for comp, off_r, cnt_r in (
            ("q_proj", 0, spec.q_total),
            ("k_proj", spec.q_total, spec.k_total),
            ("v_proj", spec.q_total + spec.k_total, spec.v_total),
        ):
            out_nm = f"{out_pfx}.{comp}.weight"
            knd, lyr, exp = _classify_tensor(out_nm, cold_set, all_experts_cold)
            p_over = (cnt_r * 4) if tgt_dt == DType.FP8_E4M3_ROW else 0
            entries.append(
                _make_entry(
                    out_nm,
                    knd,
                    tgt_dt,
                    qb,
                    0,
                    lyr,
                    exp,
                    (cnt_r, spec.k_cols),
                    _full_range(w_t),
                    dc,
                    _full_range(s_t) if s_t is not None else None,
                    None,
                    sme2_pack="none",
                    params_mode="fp8_row" if tgt_dt == DType.FP8_E4M3_ROW else "affine_scale_bias",
                    params_bytes_override=p_over,
                    mimo_qkv_slice=(spec, off_r, cnt_r),
                    name_already_normalized=True,
                )
            )
        return True

    eff_vision_mode = resolve_vision_mode(
        tensors.keys(),
        include_vision=include_vision,
        strip_vision=strip_vision,
    )
    do_strip_vision = eff_vision_mode == "strip"

    stripped_vision = 0
    included_vision = 0
    filtered_tensors: dict[str, _ShardTensor] = {}
    for name, st in tensors.items():
        if is_vision_tensor(name):
            if do_strip_vision:
                stripped_vision += 1
                continue
            included_vision += 1
        filtered_tensors[name] = st
    if stats_out is not None:
        stats_out["stripped_vision_tensors"] = stripped_vision
        stats_out["included_vision_tensors"] = included_vision
    tensors = filtered_tensors

    # Separate MLX affine triples (`weight` + `scales` + `biases`) from MLX NVFP4 / ModelOpt / FP8 pairs.
    affine_prefixes: set[str] = set()
    mlx_nvfp4_prefixes: set[str] = set()
    for name in tensors:
        if name.endswith(".scales"):
            prefix = name[: -len(".scales")]
            w = tensors.get(f"{prefix}.weight")
            if w is not None and f"{prefix}.biases" in tensors:
                affine_prefixes.add(prefix)
            elif w is not None and w.dtype == "U32" and tensors[name].dtype in ("U8", "F8_E4M3"):
                mlx_nvfp4_prefixes.add(prefix)

    def is_lm_head(name: str) -> bool:
        norm = _normalize_lenient(name)
        return norm in ("lm_head", "lm_head.weight", "unembed", "unembed.weight") or norm.startswith("lm_head.")

    embed_rows: int | None = None
    for n, t in tensors.items():
        if _normalize_lenient(n) in ("embed_tokens.weight", "embed_tokens.scales", "embed.weight"):
            if not is_excluded(n) and not is_excluded("embed_tokens") and not is_excluded("embed.weight"):
                embed_rows = t.shape[0] if t.shape else None
                break
    has_embed = embed_rows is not None
    skip_lm_head = tie_word_embeddings and has_embed

    entries: list[_PlannedEntry] = []
    consumed: set[str] = set()

    # 1) MLX Affine quantized groups (`q2..q8_affine`, `group_size in {32, 64, 128}`)
    ling3_mlx_kv_pairs: dict[str, tuple[str | None, str | None]] = {}
    if ling3_config is not None or ling3_mtp_standalone:
        tmp_eq: dict[str, str] = {}
        tmp_uo: dict[str, str] = {}
        for prefix in sorted(affine_prefixes):
            if prefix.endswith((".attention.embed_q", ".self_attn.embed_q")):
                parent = prefix.rsplit(".", 1)[0]
                tmp_eq[parent] = prefix
            elif prefix.endswith((".attention.unembed_out", ".self_attn.unembed_out")):
                parent = prefix.rsplit(".", 1)[0]
                tmp_uo[parent] = prefix
        for parent in sorted(set(tmp_eq) & set(tmp_uo)):
            ling3_mlx_kv_pairs[parent] = (tmp_eq[parent], tmp_uo[parent])

    for parent, (eq_pref, uo_pref) in sorted(ling3_mlx_kv_pairs.items()):
        assert eq_pref is not None and uo_pref is not None
        eq_w, eq_s, eq_b = tensors[f"{eq_pref}.weight"], tensors[f"{eq_pref}.scales"], tensors[f"{eq_pref}.biases"]
        uo_w, uo_s, uo_b = tensors[f"{uo_pref}.weight"], tensors[f"{uo_pref}.scales"], tensors[f"{uo_pref}.biases"]
        consumed.update((eq_w.name, eq_s.name, eq_b.name, uo_w.name, uo_s.name, uo_b.name))
        if is_excluded(eq_pref) or is_excluded(eq_w.name) or is_excluded(uo_pref) or is_excluded(uo_w.name):
            continue
        if len(eq_w.shape) != 3 or len(uo_w.shape) != 3 or eq_w.shape[0] != uo_w.shape[0]:
            raise ValueError(f"inconsistent MLX MLA pair {eq_pref}={list(eq_w.shape)} vs {uo_pref}={list(uo_w.shape)}")
        n_h, kv_lora, eq_packed = eq_w.shape
        _, v_hd, uo_packed = uo_w.shape
        eq_groups = eq_s.shape[2]
        uo_groups = uo_s.shape[2]
        eq_bits, _ = _quant_bits_for(eq_pref, eq_packed, eq_groups, group_size)
        uo_bits, _ = _quant_bits_for(uo_pref, uo_packed, uo_groups, group_size)
        qk_nope = eq_groups * group_size
        uo_k_full = uo_groups * group_size
        if uo_k_full != kv_lora:
            raise ValueError(f"MLX MLA pair mismatch: {eq_pref} kv_lora={kv_lora} vs {uo_pref} k={uo_k_full}")
        out_kv_nm = _plan_out_name(f"{parent}.kv_b_proj.weight")
        kind, layer, expert = _classify_tensor(out_kv_nm, cold_set, all_experts_cold)
        entries.append(
            _make_entry(
                out_kv_nm,
                kind,
                DType.BF16,
                0,
                0,
                layer,
                expert,
                (n_h * (qk_nope + v_hd), kv_lora),
                _full_range(eq_w),
                "mlx_kv_b_bf16",
                None,
                None,
                sme2_pack="none",
                params_bytes_override=0,
                mlx_kv_b_pair=(
                    n_h,
                    kv_lora,
                    qk_nope,
                    v_hd,
                    _full_range(eq_w),
                    _full_range(eq_s),
                    _full_range(eq_b),
                    eq_bits,
                    group_size,
                    _full_range(uo_w),
                    _full_range(uo_s),
                    _full_range(uo_b),
                    uo_bits,
                    group_size,
                ),
                name_already_normalized=True,
            )
        )

    for prefix in sorted(affine_prefixes):
        if f"{prefix}.weight" in consumed:
            continue
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

        if len(w.shape) == 3:
            stacked = _resolve_stacked_expert_meta(prefix)
            if stacked is None and ling3_mtp_standalone:
                mtp_sw = re.match(r"^(?:model\.)?mtp\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)$", prefix)
                if mtp_sw is not None:
                    stacked = (0, mtp_sw.group(2))
            if stacked is None:
                raise ValueError(f"3D quantized tensor {prefix} is not an MLX switch_mlp expert stack")
            if len(s.shape) != 3 or w.shape[:2] != s.shape[:2]:
                raise ValueError(f"inconsistent stacked expert shapes for {prefix}: {list(w.shape)}, {list(s.shape)}")
            num_experts, n, packed = w.shape
            groups = s.shape[2]
            bits, amf_dtype = _quant_bits_for(prefix, packed, groups, group_size)
            layer, proj = stacked
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
        norm_name = _plan_out_name(f"{prefix}.weight") if is_mimo_or_q35 else _normalize_name(prefix)
        k_full = groups * group_size
        if (
            (qwen35_config is not None or glm5_config is not None or ling3_config is not None or ling3_mtp_standalone)
            and norm_name.endswith(
                (".linear_attn.a_proj.weight", ".linear_attn.b_proj.weight", ".linear_attn.f_a_proj.weight", ".linear_attn.g_a_proj.weight")
            )
        ) or (
            (ling3_config is not None or ling3_mtp_standalone)
            and norm_name.endswith((".self_attn.g_proj.weight", ".mlp.gate.weight"))
        ):
            kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
            entries.append(
                _make_entry(
                    norm_name,
                    kind,
                    DType.BF16,
                    0,
                    0,
                    layer,
                    expert,
                    (n, k_full),
                    _full_range(w),
                    f"affine{bits}_to_bf16",
                    _full_range(s),
                    _full_range(b),
                    sme2_pack="none",
                    params_bytes_override=0,
                    name_already_normalized=True,
                )
            )
            continue
        g5_splits = _glm5_split_specs(norm_name, (n, k_full))
        if g5_splits is not None:
            for split_nm, runs, opt_shp in g5_splits:
                split_rows = sum(rc for _, rc in runs)
                eff_shp = opt_shp if opt_shp is not None else (split_rows, k_full)
                kind, layer, expert = _classify_tensor(split_nm, cold_set, all_experts_cold)
                entries.append(
                    _make_entry(
                        split_nm,
                        kind,
                        amf_dtype,
                        bits,
                        group_size,
                        layer,
                        expert,
                        eff_shp,
                        _full_range(w),
                        None,
                        _full_range(s),
                        _full_range(b),
                        sme2_pack=sme2_pack,
                        sme2_pack_loop_adapters=sme2_pack_loop_adapters,
                        src_total_rows=n,
                        row_gather_runs=runs,
                        name_already_normalized=True,
                    )
                )
            continue
        q35_splits = _qwen35_split_specs(norm_name, n)
        if q35_splits is not None:
            for split_nm, runs in q35_splits:
                split_rows = sum(rc for _, rc in runs)
                kind, layer, expert = _classify_tensor(split_nm, cold_set, all_experts_cold)
                entries.append(
                    _make_entry(
                        split_nm,
                        kind,
                        amf_dtype,
                        bits,
                        group_size,
                        layer,
                        expert,
                        (split_rows, k_full),
                        _full_range(w),
                        None,
                        _full_range(s),
                        _full_range(b),
                        sme2_pack=sme2_pack,
                        sme2_pack_loop_adapters=sme2_pack_loop_adapters,
                        src_total_rows=n,
                        row_gather_runs=runs,
                        name_already_normalized=True,
                    )
                )
            continue
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
                (n, k_full),
                _full_range(w),
                None,
                _full_range(s),
                _full_range(b),
                sme2_pack=sme2_pack,
                sme2_pack_loop_adapters=sme2_pack_loop_adapters,
                name_already_normalized=True,
            )
        )

    # 2) MLX NVFP4 (`mode: "nvfp4"` — `.weight` U32 + `.scales` U8/F8_E4M3 + optional `.global_scale`)
    for prefix in sorted(mlx_nvfp4_prefixes):
        w = tensors[f"{prefix}.weight"]
        s = tensors[f"{prefix}.scales"]
        gs = tensors.get(f"{prefix}.global_scale")
        consumed.update((w.name, s.name))
        if gs is not None:
            consumed.add(gs.name)
        if is_excluded(prefix) or is_excluded(w.name):
            continue
        if skip_lm_head and is_lm_head(prefix) and (embed_rows is None or w.shape[0] == embed_rows):
            continue
        gs_range = _full_range(gs) if gs is not None else None
        denom_const = 2688.0 if gs is not None else 1.0  # mlx_fp_quantized.h: *global_scale / (448*6)
        if len(w.shape) == 3:
            stacked = _resolve_stacked_expert_meta(prefix)
            if stacked is None:
                raise ValueError(f"3D MLX nvfp4 tensor {prefix} is not an expert stack")
            if len(s.shape) != 3 or w.shape[:2] != s.shape[:2]:
                raise ValueError(f"inconsistent 3D MLX nvfp4 shapes for {prefix}: {list(w.shape)}, {list(s.shape)}")
            num_experts, n, k_packed = w.shape
            layer, proj = stacked
            gs_numel = 1
            if gs is not None:
                for d in gs.shape:
                    gs_numel *= d
                if gs_numel not in (1, num_experts):
                    raise ValueError(f"MLX global_scale shape {list(gs.shape)} incompatible with E={num_experts} on {prefix}")
            for e in range(num_experts):
                exp_idx = 0 if (gs is None or gs_numel == 1) else e
                entries.append(
                    _plan_nvfp4_slice(
                        name=f"layers.{layer}.mlp.experts.{e}.{proj}",
                        kind=_expert_kind(layer, e, cold_set, all_experts_cold),
                        layer=layer,
                        expert=e,
                        n_rows=n,
                        k_packed_cols=k_packed,
                        w_dtype=w.dtype,
                        w_range=_slice_range(w, e, num_experts),
                        s_range=_slice_range(s, e, num_experts),
                        s_shape_2d=s.shape[1:],
                        outer_range=gs_range,
                        denom_range=None,
                        outer_const=1.0,
                        denom_const=denom_const,
                        divisor_mode=False,
                        outer_expert_idx=exp_idx,
                        prefix_desc=prefix,
                    )
                )
            continue
        if len(w.shape) != 2:
            raise ValueError(f"expected 2D MLX nvfp4 tensor for {prefix}, got {list(w.shape)}")
        norm_name = _plan_out_name(f"{prefix}.weight") if is_mimo_or_q35 else _normalize_name(prefix)
        kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
        entries.append(
            _plan_nvfp4_slice(
                name=norm_name,
                kind=kind,
                layer=layer,
                expert=expert,
                n_rows=w.shape[0],
                k_packed_cols=w.shape[1],
                w_dtype=w.dtype,
                w_range=_full_range(w),
                s_range=_full_range(s),
                s_shape_2d=s.shape,
                outer_range=gs_range,
                denom_range=None,
                outer_const=1.0,
                denom_const=denom_const,
                divisor_mode=False,
                outer_expert_idx=0,
                prefix_desc=prefix,
            )
        )

    # 3) ModelOpt / compressed-tensors / Inkling-2 NVFP4 + ModelOpt MXFP8 + Block/Row/Tensor FP8
    for w_name in sorted(tensors):
        if w_name in consumed:
            continue
        w = tensors[w_name]
        if w.dtype not in ("U8", "I8", "U32", "I32", "F8_E4M3"):
            continue
        split = _split_quant_weight_base(w_name)
        if split is None:
            continue
        base, style = split
        if style == "dot_weight":
            s = _pick_companion(
                tensors,
                (
                    f"{base}.weight_scale_inv",
                    f"{base}.weight_scale",
                    f"{base}.scale_inv",
                    f"{base}.scale",
                ),
            )
            s2 = _pick_companion(tensors, (f"{base}.weight_scale_2", f"{base}.scale_2"))
            gscale = _pick_companion(tensors, (f"{base}.weight_global_scale", f"{base}.global_scale"))
            amax_t = _pick_companion(tensors, (f"{base}.weight_amax", f"{base}.amax"))
            denom_t = _pick_companion(
                tensors, (f"{base}.weight_outer_scale_denominator", f"{base}.outer_scale_denominator")
            )
            inp_s = _pick_companion(tensors, (f"{base}.input_scale", f"{base}.input_global_scale"))
        else:
            s = _pick_companion(
                tensors,
                (
                    f"{base}_weight_scale_inv",
                    f"{base}_weight_scale",
                    f"{base}_scale_inv",
                    f"{base}_scale",
                ),
            )
            s2 = _pick_companion(tensors, (f"{base}_weight_scale_2", f"{base}_scale_2"))
            gscale = _pick_companion(tensors, (f"{base}_weight_global_scale", f"{base}_global_scale"))
            amax_t = _pick_companion(tensors, (f"{base}_weight_amax", f"{base}_amax"))
            denom_t = _pick_companion(
                tensors, (f"{base}_weight_outer_scale_denominator", f"{base}_outer_scale_denominator")
            )
            inp_s = _pick_companion(tensors, (f"{base}_input_scale", f"{base}_input_global_scale"))

        if s is None and w.dtype in ("U8", "U32", "I32"):
            # Unpaired U32/I32/U8 tensors (e.g. token tables) fall through to unquantized handling.
            continue

        consumed.add(w_name)
        for aux in (s, s2, gscale, amax_t, denom_t, inp_s):
            if aux is not None:
                consumed.add(aux.name)

        if is_excluded(base) or is_excluded(w_name):
            continue
        if skip_lm_head and is_lm_head(base) and (embed_rows is None or w.shape[0] == embed_rows):
            continue

        if (
            w.dtype in ("U8", "I8")
            and s is not None
            and s.dtype in ("U8", "F8_E8M0")
            and s2 is None
            and amax_t is None
            and gscale is None
            and len(w.shape) == 2
            and len(s.shape) == 2
            and w.shape[0] == s.shape[0]
            and w.shape[1] == 16 * s.shape[1]
        ):
            k_full = w.shape[1] * 2
            norm_name = _plan_out_name(w_name if w_name.endswith(".weight") else f"{base}.weight")
            kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
            entries.append(
                _make_entry(
                    norm_name,
                    kind,
                    DType.GGUF_MXFP4,
                    4,
                    32,
                    layer,
                    expert,
                    (w.shape[0], k_full),
                    _full_range(w),
                    "hf_mxfp4_to_gguf_mxfp4",
                    _full_range(s),
                    None,
                    sme2_pack="none",
                    params_bytes_override=0,
                    name_already_normalized=True,
                )
            )
            continue

        if w.dtype in ("U8", "U32", "I32"):
            assert s is not None
            if s2 is not None:
                outer_r: _ByteRange | None = _full_range(s2)
                denom_r: _ByteRange | None = None
                div_mode = False
                d_const = 1.0
                src_outer = s2
            elif amax_t is not None:
                outer_r = _full_range(amax_t)
                denom_r = _full_range(denom_t) if denom_t is not None else None
                div_mode = False
                d_const = 2688.0 if denom_t is None else 1.0
                src_outer = amax_t
            elif gscale is not None:
                outer_r = _full_range(gscale)
                denom_r = None
                div_mode = nvfp4_global_scale_is_divisor
                d_const = 1.0
                src_outer = gscale
            else:
                outer_r = None
                denom_r = None
                div_mode = False
                d_const = 1.0
                src_outer = None

            if len(w.shape) == 3:
                stacked = _resolve_stacked_expert_meta(base)
                if stacked is None:
                    raise ValueError(f"3D NVFP4 tensor {w_name} is not an expert stack")
                if len(s.shape) != 3 or w.shape[:2] != s.shape[:2]:
                    raise ValueError(f"inconsistent 3D NVFP4 shapes for {w_name}: {list(w.shape)}, {list(s.shape)}")
                num_experts, n_r, k_p = w.shape
                layer, proj = stacked
                outer_numel = 1
                if src_outer is not None:
                    for d in src_outer.shape:
                        outer_numel *= d
                    if outer_numel not in (1, num_experts):
                        raise ValueError(
                            f"outer scale {src_outer.name} shape {list(src_outer.shape)} incompatible with E={num_experts}"
                        )
                if denom_t is not None:
                    denom_numel = 1
                    for d in denom_t.shape:
                        denom_numel *= d
                    if denom_numel not in (1, num_experts):
                        raise ValueError(
                            f"outer scale denominator {denom_t.name} shape {list(denom_t.shape)} incompatible with E={num_experts}"
                        )
                for e in range(num_experts):
                    exp_idx = 0 if (src_outer is None or outer_numel == 1) else e
                    entries.append(
                        _plan_nvfp4_slice(
                            name=f"layers.{layer}.mlp.experts.{e}.{proj}",
                            kind=_expert_kind(layer, e, cold_set, all_experts_cold),
                            layer=layer,
                            expert=e,
                            n_rows=n_r,
                            k_packed_cols=k_p,
                            w_dtype=w.dtype,
                            w_range=_slice_range(w, e, num_experts),
                            s_range=_slice_range(s, e, num_experts),
                            s_shape_2d=s.shape[1:],
                            outer_range=outer_r,
                            denom_range=denom_r,
                            outer_const=1.0,
                            denom_const=d_const,
                            divisor_mode=div_mode,
                            outer_expert_idx=exp_idx,
                            prefix_desc=base,
                        )
                    )
                continue

            if len(w.shape) != 2:
                raise ValueError(f"NVFP4 weight {w_name} must be 2D or 3D expert stack, got {list(w.shape)}")
            out_name = _plan_out_name(w_name if style == "underscore_weight" else base)
            kind, layer, expert = _classify_tensor(out_name, cold_set, all_experts_cold)
            entries.append(
                _plan_nvfp4_slice(
                    name=out_name,
                    kind=kind,
                    layer=layer,
                    expert=expert,
                    n_rows=w.shape[0],
                    k_packed_cols=w.shape[1],
                    w_dtype=w.dtype,
                    w_range=_full_range(w),
                    s_range=_full_range(s),
                    s_shape_2d=s.shape,
                    outer_range=outer_r,
                    denom_range=denom_r,
                    outer_const=1.0,
                    denom_const=d_const,
                    divisor_mode=div_mode,
                    outer_expert_idx=0,
                    prefix_desc=base,
                )
            )
            continue

        # w.dtype == "F8_E4M3"
        if len(w.shape) == 3:
            stacked = _resolve_stacked_expert_meta(base)
            if stacked is None and qwen4_mtp_standalone:
                mtp_clean = re.sub(r"^(?:model\.)?(?:language_model\.)?mtp\.", "", base)
                stacked = _resolve_stacked_expert_meta(mtp_clean)
            if stacked is None:
                raise ValueError(f"3D FP8/MXFP8 tensor {w_name} is not an expert stack")
            num_experts, n_r, k_c = w.shape
            layer, proj = stacked
            if s is not None:
                if len(s.shape) == 3 and s.shape[0] == num_experts:
                    s_slice_shape = s.shape[1:]
                    per_exp_scale = True
                elif s.shape in ((), (1,), (1, 1)):
                    s_slice_shape = s.shape
                    per_exp_scale = False
                else:
                    raise ValueError(f"inconsistent 3D FP8 scale shape {list(s.shape)} for {w_name}")
            else:
                s_slice_shape = ()
                per_exp_scale = False
            if eff_split_gate_up and proj in ("gate_up_proj", "w13"):
                if n_r % 2 != 0:
                    raise ValueError(f"3D stacked {w_name} rows {n_r} must be even to split gate_proj/up_proj")
                half_r = n_r // 2
                if per_exp_scale and len(s_slice_shape) == 2 and s_slice_shape[0] % 2 != 0 and s_slice_shape[0] > 1:
                    raise ValueError(f"3D stacked scale {s.name} rows {s_slice_shape[0]} must be even to split")
                for e in range(num_experts):
                    ek = _expert_kind(layer, e, cold_set, all_experts_cold)
                    for h_idx, split_proj in ((0, "gate_proj"), (1, "up_proj")):
                        if s is None:
                            sr = None
                            eff_s_shp: tuple[int, ...] = ()
                        elif per_exp_scale and len(s_slice_shape) == 2 and s_slice_shape[0] >= 2:
                            sr = _slice_sub_half_range(s, e, num_experts, h_idx)
                            eff_s_shp = (s_slice_shape[0] // 2, s_slice_shape[1])
                        else:
                            sr = _slice_range(s, e, num_experts) if per_exp_scale else _full_range(s)
                            eff_s_shp = s_slice_shape
                        entries.append(
                            _plan_fp8_or_mxfp8_slice(
                                name=f"layers.{layer}.mlp.experts.{e}.{split_proj}",
                                kind=ek,
                                layer=layer,
                                expert=e,
                                n_rows=half_r,
                                k=k_c,
                                w_range=_slice_sub_half_range(w, e, num_experts, h_idx),
                                s_tensor=s,
                                s_range=sr,
                                s_slice_shape=eff_s_shp,
                                prefix_desc=base,
                            )
                        )
                continue
            for e in range(num_experts):
                sr = None if s is None else (_slice_range(s, e, num_experts) if per_exp_scale else _full_range(s))
                entries.append(
                    _plan_fp8_or_mxfp8_slice(
                        name=f"layers.{layer}.mlp.experts.{e}.{proj}",
                        kind=_expert_kind(layer, e, cold_set, all_experts_cold),
                        layer=layer,
                        expert=e,
                        n_rows=n_r,
                        k=k_c,
                        w_range=_slice_range(w, e, num_experts),
                        s_tensor=s,
                        s_range=sr,
                        s_slice_shape=s_slice_shape,
                        prefix_desc=base,
                    )
                )
            continue

        if len(w.shape) != 2:
            raise ValueError(f"FP8/MXFP8 weight {w_name} must be 2D or 3D expert stack, got {list(w.shape)}")
        out_name = _plan_out_name(w_name if s is None else (w_name if style == "underscore_weight" else base))
        if _emit_mimo_qkv_entries(w_name, out_name, w, s):
            continue
        kind, layer, expert = _classify_tensor(out_name, cold_set, all_experts_cold)
        entries.append(
            _plan_fp8_or_mxfp8_slice(
                name=out_name,
                kind=kind,
                layer=layer,
                expert=expert,
                n_rows=w.shape[0],
                k=w.shape[1],
                w_range=_full_range(w),
                s_tensor=s,
                s_range=_full_range(s) if s is not None else None,
                s_slice_shape=s.shape if s is not None else (),
                prefix_desc=base,
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
        if t.dtype == "I64" and norm_lenient.endswith(
            (
                ".ple.ple_embedding.layer_multipliers",
                ".ple.ple_embedding.ngram_heads_offsets",
                ".ple.ple_embedding.ngram_heads_vocab_sizes",
                ".ple.layer_mult",
                ".ple.ngram_offsets",
                ".ple.ngram_vocabs",
            )
        ):
            if len(t.shape) != 1 or t.shape[0] == 0:
                raise ValueError(f"PLE I64 tensor {name!r} must be non-empty 1D, got shape {list(t.shape)}")
            norm_name = _plan_out_name(name)
            kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
            entries.append(
                _make_entry(
                    norm_name,
                    kind,
                    DType.U32,
                    0,
                    0,
                    layer,
                    expert,
                    (t.shape[0], 2),
                    _full_range(t),
                    "i64_raw_u32_pair",
                    sme2_pack="none",
                    name_already_normalized=True,
                )
            )
            continue
        if _is_d2t_name(norm_lenient) and t.dtype in ("I64", "I32"):
            if len(t.shape) != 1 or t.shape[0] == 0:
                raise ValueError(f"EAGLE-3 d2t tensor {name!r} must be non-empty 1D, got shape {list(t.shape)}")
            amf_dtype = DType.U32
            convert: str | None = "d2t_i64_to_u32" if t.dtype == "I64" else "d2t_i32_to_u32"
        elif t.dtype == "F8_E4M3":
            norm_name = _plan_out_name(name)
            kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
            entries.append(
                _make_entry(
                    norm_name,
                    kind,
                    DType.FP8_E4M3_TENSOR,
                    8,
                    0,
                    layer,
                    expert,
                    t.shape,
                    _full_range(t),
                    None,
                    None,
                    None,
                    sme2_pack="none",
                    params_mode="fp8_tensor",
                    params_bytes_override=4,
                    outer_scale_const=1.0,
                )
            )
            continue
        else:
            amf_dtype, convert = _unquantized_target(t.dtype, dtype_mode, name)
        if name.endswith(".weight"):
            stacked = _resolve_stacked_expert_meta(name[: -len(".weight")])
            if stacked is None and ling3_mtp_standalone:
                mtp_sw_f = re.match(r"^(?:model\.)?mtp\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.weight$", name)
                if mtp_sw_f is not None:
                    stacked = (0, mtp_sw_f.group(2))
            if stacked is None and qwen4_mtp_standalone:
                mtp_clean = re.sub(r"^(?:model\.)?(?:language_model\.)?mtp\.", "", name[: -len(".weight")])
                stacked = _resolve_stacked_expert_meta(mtp_clean)
        elif name.endswith("_weight"):
            stacked = _resolve_stacked_expert_meta(name[: -len("_weight")])
        else:
            stacked = _resolve_stacked_expert_meta(name)
            if stacked is None and qwen4_mtp_standalone:
                mtp_clean = re.sub(r"^(?:model\.)?(?:language_model\.)?mtp\.", "", name)
                stacked = _resolve_stacked_expert_meta(mtp_clean)
        if stacked is not None and len(t.shape) == 3:
            num_experts, n, k = t.shape
            layer, proj = stacked
            if eff_split_gate_up and proj in ("gate_up_proj", "w13"):
                if n % 2 != 0:
                    raise ValueError(f"3D stacked {name} rows {n} must be even to split gate_proj/up_proj")
                half_n = n // 2
                for e in range(num_experts):
                    ek = _expert_kind(layer, e, cold_set, all_experts_cold)
                    for h_idx, split_proj in ((0, "gate_proj"), (1, "up_proj")):
                        entries.append(
                            _make_entry(
                                f"layers.{layer}.mlp.experts.{e}.{split_proj}.weight",
                                ek,
                                amf_dtype,
                                0,
                                0,
                                layer,
                                e,
                                (half_n, k),
                                _slice_sub_half_range(t, e, num_experts, h_idx),
                                convert,
                                sme2_pack=sme2_pack,
                            )
                        )
                continue
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
        norm_name = _plan_out_name(name)
        if len(t.shape) == 2 and _emit_mimo_qkv_entries(name, norm_name, t, None):
            continue
        if len(t.shape) in (2, 3):
            g5_splits = _glm5_split_specs(norm_name, t.shape)
            if g5_splits is not None:
                for split_nm, runs, opt_shp in g5_splits:
                    split_rows = sum(rc for _, rc in runs)
                    eff_shp = opt_shp if opt_shp is not None else (split_rows, t.shape[1])
                    eff_dt = amf_dtype
                    eff_conv = convert
                    if split_nm.endswith(
                        (".linear_attn.q_conv1d.weight", ".linear_attn.k_conv1d.weight", ".linear_attn.v_conv1d.weight")
                    ):
                        if t.dtype == "BF16":
                            eff_dt = DType.F32
                            eff_conv = "bf16_to_f32"
                        elif t.dtype == "F16":
                            eff_dt = DType.F32
                            eff_conv = "f16_to_f32"
                    kind, layer, expert = _classify_tensor(split_nm, cold_set, all_experts_cold)
                    entries.append(
                        _make_entry(
                            split_nm,
                            kind,
                            eff_dt,
                            0,
                            0,
                            layer,
                            expert,
                            eff_shp,
                            _full_range(t),
                            eff_conv,
                            sme2_pack=sme2_pack,
                            sme2_pack_loop_adapters=sme2_pack_loop_adapters,
                            src_total_rows=t.shape[0],
                            row_gather_runs=runs,
                            name_already_normalized=True,
                        )
                    )
                continue
        if len(t.shape) == 2:
            q35_splits = _qwen35_split_specs(norm_name, t.shape[0]) or _qwen4_split_specs(norm_name, t.shape[0])
            if q35_splits is not None:
                for split_nm, runs in q35_splits:
                    split_rows = sum(rc for _, rc in runs)
                    kind, layer, expert = _classify_tensor(split_nm, cold_set, all_experts_cold)
                    entries.append(
                        _make_entry(
                            split_nm,
                            kind,
                            amf_dtype,
                            0,
                            0,
                            layer,
                            expert,
                            (split_rows, t.shape[1]),
                            _full_range(t),
                            convert,
                            sme2_pack=sme2_pack,
                            sme2_pack_loop_adapters=sme2_pack_loop_adapters,
                            src_total_rows=t.shape[0],
                            row_gather_runs=runs,
                            name_already_normalized=True,
                        )
                    )
                continue
        kind, layer, expert = _classify_tensor(norm_name, cold_set, all_experts_cold)
        # Flatten 5D Conv3d vision `patch_embed` weights `[out_c, 3, t_patch, kh, kw]`
        # into 2D `[out_c, 3 * t_patch * kh * kw]` to fit AMF v0's 4-dim shape slot
        # and match `vision_ingress.metal`'s exact contiguous tube layout.
        eff_shape = t.shape
        if len(eff_shape) == 5 and is_vision_tensor(norm_name) and "patch_embed" in norm_name.lower():
            flat_in = eff_shape[1] * eff_shape[2] * eff_shape[3] * eff_shape[4]
            eff_shape = (eff_shape[0], flat_in)
        elif (
            (
                qwen35_config is not None
                or qwen4_config is not None
                or qwen4_mtp_standalone
                or glm5_config is not None
                or ling3_config is not None
                or ling3_mtp_standalone
            )
            and norm_name.endswith(
                (
                    ".linear_attn.conv1d.weight",
                    ".linear_attn.q_conv1d.weight",
                    ".linear_attn.k_conv1d.weight",
                    ".linear_attn.v_conv1d.weight",
                    ".ple.conv1d.weight",
                )
            )
            and len(eff_shape) == 3
            and (eff_shape[1] == 1 or eff_shape[2] == 1)
        ):
            kern = eff_shape[2] if eff_shape[1] == 1 else eff_shape[1]
            eff_shape = (eff_shape[0], kern)
        if norm_name.endswith(
            (
                ".linear_attn.A_log",
                ".linear_attn.dt_bias",
                ".linear_attn.conv1d.weight",
                ".linear_attn.q_conv1d.weight",
                ".linear_attn.k_conv1d.weight",
                ".linear_attn.v_conv1d.weight",
                ".hc_attn_base",
                ".hc_attn_scale",
                ".hc_ffn_base",
                ".hc_ffn_scale",
            )
        ) or ((glm5_config is not None or ling3_config is not None or ling3_mtp_standalone) and norm_name.endswith(".mlp.gate.bias")):
            # GDN/KDA, Sinkhorn mHC scalar/base tables, and GLM-5 router biases bind float32 buffers.
            if t.dtype == "BF16":
                amf_dtype = DType.F32
                convert = "bf16_to_f32"
            elif t.dtype == "F16":
                amf_dtype = DType.F32
                convert = "f16_to_f32"
        entries.append(
            _make_entry(
                norm_name,
                kind,
                amf_dtype,
                0,
                0,
                layer,
                expert,
                eff_shape,
                _full_range(t),
                convert,
                sme2_pack=sme2_pack,
                sme2_pack_loop_adapters=sme2_pack_loop_adapters,
                name_already_normalized=True,
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
    return normalize_vision_name(raw_name) if is_vision_tensor(raw_name) else _strip_model_prefix(raw_name)


class _ShardReaders:
    """Lazily opened, F_NOCACHE or RangeStreamReader handles for each shard source."""

    def __init__(
        self,
        source_factories: dict[pathlib.Path | str, Callable[[], RangeStreamReader]] | None = None,
    ) -> None:
        self._handles: dict[pathlib.Path | str, object] = {}
        self._factories = dict(source_factories or {})

    def get(self, path: pathlib.Path | str):
        fh = self._handles.get(path)
        if fh is None:
            factory = self._factories.get(path)
            if factory is not None:
                fh = factory()
            else:
                p = pathlib.Path(path)
                fh = p.open("rb", buffering=0)
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


def _read_fd_exact(fh, offset: int, length: int) -> bytes:
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


def _read_exact(fh, offset: int, length: int) -> bytes:
    if hasattr(fh, "read_range"):
        return fh.read_range(offset, length)
    return _read_fd_exact(fh, offset, length)


def _decode_scalar_f32_at(readers: _ShardReaders, br: _ByteRange, index: int) -> float:
    elem = _SAFETENSORS_DTYPE_BYTES[br.dtype]
    if br.length < elem or (br.length % elem) != 0:
        raise ValueError(
            f"invalid scalar scale byte range length {br.length} for dtype {br.dtype}"
        )
    if index < 0:
        raise ValueError(f"scalar scale index {index} must be non-negative")
    count = br.length // elem
    if count == 1:
        eff_index = 0
    elif index < count:
        eff_index = index
    else:
        raise ValueError(
            f"scalar scale index {index} out of bounds for count {count} ({br.dtype}, len={br.length})"
        )
    fh = readers.get(br.path)
    raw = _read_exact(fh, br.offset + eff_index * elem, elem)
    if br.dtype == "F32":
        return float(struct.unpack("<f", raw)[0])
    if br.dtype == "BF16":
        return float(struct.unpack("<f", _bf16_to_f32_bytes(raw))[0])
    if br.dtype == "F16":
        return float(struct.unpack("<e", raw)[0])
    raise ValueError(f"unsupported float scale dtype {br.dtype}")


def _stream_promoted_f32(readers: _ShardReaders, br: _ByteRange, w: _BlobWriter) -> None:
    elem = _SAFETENSORS_DTYPE_BYTES[br.dtype]
    fh = readers.get(br.path)
    step = _CONVERT_CHUNK_ELEMS * elem
    done = 0
    while done < br.length:
        take = min(step, br.length - done)
        raw = _read_exact(fh, br.offset + done, take)
        if br.dtype == "F32":
            w.write(raw)
        elif br.dtype == "BF16":
            w.write(_bf16_to_f32_bytes(raw))
        elif br.dtype == "F16":
            cnt = take // 2
            w.write(struct.pack(f"<{cnt}f", *struct.unpack(f"<{cnt}e", raw)))
        else:
            raise ValueError(f"unsupported scale dtype {br.dtype} for f32 promotion")
        done += take


def _stream_raw_range_chunked(readers: _ShardReaders, br: _ByteRange, w: _BlobWriter) -> None:
    fh = readers.get(br.path)
    done = 0
    while done < br.length:
        take = min(_COPY_CHUNK_BYTES, br.length - done)
        w.write(_read_exact(fh, br.offset + done, take))
        done += take


def _stream_mxfp8_scales(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    assert entry.scales is not None
    if entry.scale_tile_shape is None:
        _stream_raw_range_chunked(readers, entry.scales, w)
        return
    nb32, kb32 = entry.scale_tile_shape
    n_rows = entry.shape[0]
    fh = readers.get(entry.scales.path)
    for t_row in range(nb32):
        tile_raw = _read_exact(fh, entry.scales.offset + t_row * kb32, kb32)
        rep = min(32, n_rows - t_row * 32)
        for _ in range(rep):
            w.write(tile_raw)


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
    elif entry.dtype in (DType.Q4_AFFINE, DType.Q3_AFFINE, DType.Q2_AFFINE):
        if entry.scales is None or entry.biases is None:
            raise ValueError(f"quantized sme2_panel64 tensor {entry.name!r} is missing scales/biases")
        groups = k // entry.group_size
        if entry.dtype == DType.Q4_AFFINE:
            row_data_bytes = k // 2
            packer = pack_sme2_panel64_q4_panel
        elif entry.dtype == DType.Q3_AFFINE:
            row_data_bytes = (k * 3) // 8
            packer = pack_sme2_panel64_q3_panel
        else:
            row_data_bytes = k // 4
            packer = pack_sme2_panel64_q2_from_mlx_panel
        sb_elem = _SAFETENSORS_DTYPE_BYTES[entry.scales.dtype]
        row_sb_src_bytes = groups * sb_elem
        row_param_bytes = groups * 4
        s_fh = readers.get(entry.scales.path)
        b_fh = readers.get(entry.biases.path)
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


def _stream_mimo_qkv_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    """Stream TP-deinterleaved and sliced `mimo_v2` `q_proj`/`k_proj`/`v_proj` rows directly into `.amf`."""
    from aminfer_tools.mimo_v2 import dequantize_mimo_qkv_row_f32

    assert entry.mimo_qkv_slice is not None
    spec, out_start, out_count = entry.mimo_qkv_slice
    src_fh = readers.get(entry.data.path)
    k = spec.k_cols

    # Decompose `[out_start, out_start + out_count)` across TP ranks into contiguous rank chunks.
    def _rank_chunks() -> list[tuple[int, int, int]]:
        chunks: list[tuple[int, int, int]] = []  # (src_row_start, row_count, sc_block_start_row)
        if out_start == 0 and out_count == spec.q_total:
            for r in range(spec.tp_degree):
                chunks.append((r * spec.rank_rows, spec.q_tp, r * spec.sb_per_rank))
        elif out_start == spec.q_total and out_count == spec.k_total:
            for r in range(spec.tp_degree):
                chunks.append((r * spec.rank_rows + spec.q_tp, spec.k_tp, r * spec.sb_per_rank + spec.sb_q_tp))
        elif out_start == spec.q_total + spec.k_total and out_count == spec.v_total:
            for r in range(spec.tp_degree):
                chunks.append(
                    (
                        r * spec.rank_rows + spec.q_tp + spec.k_tp,
                        spec.v_tp,
                        r * spec.sb_per_rank + spec.sb_q_tp + spec.sb_k_tp,
                    )
                )
        else:
            raise ValueError(f"unsupported mimo_qkv_slice ({out_start}, {out_count}) for {entry.name!r}")
        return chunks

    chunks = _rank_chunks()
    if entry.scales is not None:
        s_fh = readers.get(entry.scales.path)
        sc_raw = _read_exact(s_fh, entry.scales.offset, entry.scales.length)
        if entry.scales.dtype == "F32":
            scales_f32 = f32_bytes_to_f32_list(sc_raw)
        elif entry.scales.dtype == "BF16":
            scales_f32 = bf16_bytes_to_f32_list(sc_raw)
        elif entry.scales.dtype == "F16":
            scales_f32 = f16_bytes_to_f32_list(sc_raw)
        else:
            raise ValueError(f"unsupported scale_inv dtype {entry.scales.dtype} for {entry.name}")
        row_scales: list[float] = []
        for src_r0, cnt, sc_base in chunks:
            done = 0
            while done < cnt:
                take = min(128, cnt - done)
                raw = memoryview(_read_exact(src_fh, entry.data.offset + (src_r0 + done) * k, take * k))
                for i in range(take):
                    vals = dequantize_mimo_qkv_row_f32(
                        raw[i * k : (i + 1) * k],
                        sc_base + ((done + i) // 128),
                        scales_f32,
                        k,
                    )
                    if entry.dtype == DType.FP8_E4M3_ROW:
                        q_row, r_sc = quantize_row_f32_to_e4m3_row(vals)
                        w.write(q_row)
                        row_scales.append(r_sc)
                    elif entry.dtype == DType.BF16:
                        w.write(f32_to_bf16_bytes(vals))
                    elif entry.dtype == DType.F16:
                        w.write(f32_values_to_f16_bytes(vals))
                    elif entry.dtype == DType.F32:
                        w.write(struct.pack(f"<{len(vals)}f", *vals))
                    else:
                        raise ValueError(f"unsupported mimo_qkv target dtype {entry.dtype.name}")
                done += take
        if entry.dtype == DType.FP8_E4M3_ROW:
            w.zeros(entry.params_offset - entry.data_bytes)
            w.write(struct.pack(f"<{len(row_scales)}f", *row_scales))
            content = entry.params_offset + entry.params_bytes
        else:
            content = entry.data_bytes
        w.zeros(entry.blob_bytes - content)
        return

    # Unquantized float source `qkv_proj` split cleanly across ranks.
    elem_b = 4 if entry.data.dtype == "F32" else 2
    row_b = k * elem_b
    for src_r0, cnt, _ in chunks:
        done = 0
        while done < cnt:
            take = min(256, cnt - done)
            raw = _read_exact(src_fh, entry.data.offset + (src_r0 + done) * row_b, take * row_b)
            if entry.data_convert is None:
                w.write(raw)
            elif entry.data_convert == "bf16_to_f16":
                w.write(bf16_bytes_to_f16_bytes(raw, what=entry.name))
            elif entry.data_convert == "f16_to_bf16":
                w.write(f16_bytes_to_bf16_bytes(raw))
            elif entry.data_convert == "bf16_to_f32":
                w.write(_bf16_to_f32_bytes(raw))
            elif entry.data_convert == "f16_to_f32":
                w.write(_f16_to_f32_bytes(raw))
            else:
                raise ValueError(f"unsupported float qkv conversion {entry.data_convert!r}")
            done += take
    w.zeros(entry.blob_bytes - entry.data_bytes)


def _stream_hf_mxfp4_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    """Repack HF `MXFP4` (`U8 [N, K//2]` + `U8 [N, K//32]` UE8M0) into inline `GGUF_MXFP4` blocks."""
    from aminfer_tools.mimo_v2 import repack_hf_mxfp4_row_to_gguf

    assert entry.scales is not None
    n_rows, k = entry.shape
    w_row_b = k // 2
    s_row_b = k // 32
    src_fh = readers.get(entry.data.path)
    s_fh = readers.get(entry.scales.path)
    done = 0
    while done < n_rows:
        take = min(128, n_rows - done)
        w_mv = memoryview(_read_exact(src_fh, entry.data.offset + done * w_row_b, take * w_row_b))
        s_mv = memoryview(_read_exact(s_fh, entry.scales.offset + done * s_row_b, take * s_row_b))
        for i in range(take):
            w.write(repack_hf_mxfp4_row_to_gguf(w_mv[i * w_row_b : (i + 1) * w_row_b], s_mv[i * s_row_b : (i + 1) * s_row_b], k))
        done += take
    w.zeros(entry.blob_bytes - entry.data_bytes)


def _stream_affine_to_bf16_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    """Dequantize small MLX affine tensors (e.g. `qwen3_5` `in_proj_a/b`) to `BF16`."""
    assert entry.scales is not None and entry.biases is not None
    bits = int(entry.data_convert[len("affine") : -len("_to_bf16")])  # type: ignore[index]
    aff_dt = _AFFINE_BITS_DTYPE[bits]
    row_data_fn, deq = _AFFINE_DEQUANT_SPEC[aff_dt]
    n_rows, k = entry.shape
    sb_elem = _SAFETENSORS_DTYPE_BYTES[entry.scales.dtype]
    groups = (entry.scales.length // sb_elem) // n_rows
    gs = k // groups
    row_d = row_data_fn(k)
    row_sb = groups * sb_elem
    src_fh = readers.get(entry.data.path)
    s_fh = readers.get(entry.scales.path)
    b_fh = readers.get(entry.biases.path)
    for r in range(n_rows):
        d_raw = _read_exact(src_fh, entry.data.offset + r * row_d, row_d)
        s_raw = _read_exact(s_fh, entry.scales.offset + r * row_sb, row_sb)
        b_raw = _read_exact(b_fh, entry.biases.offset + r * row_sb, row_sb)
        s_f = bf16_bytes_to_f32_list(s_raw) if entry.scales.dtype == "BF16" else (f16_bytes_to_f32_list(s_raw) if entry.scales.dtype == "F16" else f32_bytes_to_f32_list(s_raw))
        b_f = bf16_bytes_to_f32_list(b_raw) if entry.biases.dtype == "BF16" else (f16_bytes_to_f32_list(b_raw) if entry.biases.dtype == "F16" else f32_bytes_to_f32_list(b_raw))
        p_buf = bytearray(groups * 4)
        p_f16 = interleave_scale_bias_f16(
            f32_values_to_f16_bytes(s_f),
            f32_values_to_f16_bytes(b_f),
        )
        p_buf[:] = p_f16
        vals = deq(d_raw, p_buf, k, group_size=gs)
        w.write(f32_to_bf16_bytes(vals))
    w.zeros(entry.blob_bytes - entry.data_bytes)


def _stream_row_gathered_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    """Stream contiguous `(row_start, row_count)` row runs cleanly for weights and optional affine scales/biases."""
    assert entry.row_gather_runs is not None and entry.src_total_rows
    src_fh = readers.get(entry.data.path)
    row_d = entry.data.length // entry.src_total_rows
    for r_start, r_cnt in entry.row_gather_runs:
        off = entry.data.offset + r_start * row_d
        tot = r_cnt * row_d
        done = 0
        while done < tot:
            take = min(_COPY_CHUNK_BYTES, tot - done)
            raw = _read_exact(src_fh, off + done, take)
            if entry.data_convert is None:
                w.write(raw)
            elif entry.data_convert == "bf16_to_f16":
                w.write(bf16_bytes_to_f16_bytes(raw, what=entry.name))
            elif entry.data_convert == "f16_to_bf16":
                w.write(f16_bytes_to_bf16_bytes(raw))
            elif entry.data_convert == "bf16_to_f32":
                w.write(_bf16_to_f32_bytes(raw))
            elif entry.data_convert == "f16_to_f32":
                w.write(_f16_to_f32_bytes(raw))
            else:
                raise ValueError(f"unknown conversion {entry.data_convert!r}")
            done += take
    if entry.params_bytes > 0:
        assert entry.scales is not None and entry.biases is not None
        w.zeros(entry.params_offset - entry.data_bytes)
        elem = _SAFETENSORS_DTYPE_BYTES[entry.scales.dtype]
        row_sb = (entry.scales.length // entry.src_total_rows)
        s_fh = readers.get(entry.scales.path)
        b_fh = readers.get(entry.biases.path)
        for r_start, r_cnt in entry.row_gather_runs:
            s_raw = _read_exact(s_fh, entry.scales.offset + r_start * row_sb, r_cnt * row_sb)
            b_raw = _read_exact(b_fh, entry.biases.offset + r_start * row_sb, r_cnt * row_sb)
            w.write(
                interleave_scale_bias_f16(
                    _to_f16_chunk(s_raw, entry.scales.dtype, f"{entry.name} scales"),
                    _to_f16_chunk(b_raw, entry.biases.dtype, f"{entry.name} biases"),
                )
            )
        content = entry.params_offset + entry.params_bytes
    else:
        content = entry.data_bytes
    w.zeros(entry.blob_bytes - content)


def _stream_fp8_block_to_bf16_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    """Dequantize non-128x128 FP8 block tensors (e.g. tiny `[32, 32]` HF fixtures) to `BF16`."""
    assert entry.scales is not None and entry.scale_tile_shape is not None
    n_rows, k = entry.shape
    nb, kb = int(entry.scale_tile_shape[0]), int(entry.scale_tile_shape[1])
    bh, bw = ((n_rows + nb - 1) // nb, (k + kb - 1) // kb)
    for cand in (32, 64, 16):
        if (n_rows + cand - 1) // cand == nb and (k + cand - 1) // cand == kb:
            bh, bw = cand, cand
            break
    s_raw = _read_exact(readers.get(entry.scales.path), entry.scales.offset, entry.scales.length)
    s_f = (
        f32_bytes_to_f32_list(s_raw)
        if entry.scales.dtype == "F32"
        else (bf16_bytes_to_f32_list(s_raw) if entry.scales.dtype == "BF16" else f16_bytes_to_f32_list(s_raw))
    )
    src_fh = readers.get(entry.data.path)
    for r in range(n_rows):
        row_u8 = _read_exact(src_fh, entry.data.offset + r * k, k)
        s_row = (r // bh) * kb
        vals = [e4m3_byte_to_f32(b_u8) * s_f[s_row + (c // bw)] for c, b_u8 in enumerate(row_u8)]
        w.write(f32_to_bf16_bytes(vals))
    w.zeros(entry.blob_bytes - entry.data_bytes)


def decode_mlx_affine_3d_head_f32(
    readers: _ShardReaders,
    w_range: _ByteRange,
    s_range: _ByteRange,
    b_range: _ByteRange,
    bits: int,
    group_size: int,
    n_heads: int,
    head_rows: int,
    cols: int,
    head_idx: int,
) -> list[list[float]]:
    """Dequantize one head slice `[head_rows, cols]` from a 3D MLX affine tensor."""
    aff_dt = _AFFINE_BITS_DTYPE[bits]
    row_data_fn, deq = _AFFINE_DEQUANT_SPEC[aff_dt]
    groups = cols // group_size
    row_d = row_data_fn(cols)
    sb_elem = _SAFETENSORS_DTYPE_BYTES[s_range.dtype]
    row_sb = groups * sb_elem
    head_w_b = w_range.length // n_heads
    head_s_b = s_range.length // n_heads
    head_b_b = b_range.length // n_heads
    d_all = memoryview(_read_exact(readers.get(w_range.path), w_range.offset + head_idx * head_w_b, head_w_b))
    s_all = memoryview(_read_exact(readers.get(s_range.path), s_range.offset + head_idx * head_s_b, head_s_b))
    b_all = memoryview(_read_exact(readers.get(b_range.path), b_range.offset + head_idx * head_b_b, head_b_b))
    out: list[list[float]] = []
    for r in range(head_rows):
        d_raw = d_all[r * row_d : (r + 1) * row_d]
        s_raw = bytes(s_all[r * row_sb : (r + 1) * row_sb])
        b_raw = bytes(b_all[r * row_sb : (r + 1) * row_sb])
        s_f = (
            bf16_bytes_to_f32_list(s_raw)
            if s_range.dtype == "BF16"
            else (f16_bytes_to_f32_list(s_raw) if s_range.dtype == "F16" else f32_bytes_to_f32_list(s_raw))
        )
        b_f = (
            bf16_bytes_to_f32_list(b_raw)
            if b_range.dtype == "BF16"
            else (f16_bytes_to_f32_list(b_raw) if b_range.dtype == "F16" else f32_bytes_to_f32_list(b_raw))
        )
        p_buf = bytearray(groups * 4)
        p_buf[:] = interleave_scale_bias_f16(
            f32_values_to_f16_bytes(s_f),
            f32_values_to_f16_bytes(b_f),
        )
        out.append(deq(d_raw, p_buf, cols, group_size=group_size))
    return out


def _stream_mlx_kv_b_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    """Reconstruct MLA `kv_b_proj.weight` (`BF16`) from MLX 3D `embed_q` + `unembed_out`."""
    assert entry.mlx_kv_b_pair is not None
    (
        n_heads,
        kv_lora,
        qk_nope,
        v_hd,
        eq_w,
        eq_s,
        eq_b,
        eq_bits,
        eq_gs,
        uo_w,
        uo_s,
        uo_b,
        uo_bits,
        uo_gs,
    ) = entry.mlx_kv_b_pair
    for h in range(n_heads):
        eq_head = decode_mlx_affine_3d_head_f32(
            readers, eq_w, eq_s, eq_b, eq_bits, eq_gs, n_heads, kv_lora, qk_nope, h
        )
        for rn in range(qk_nope):
            w.write(f32_to_bf16_bytes([eq_head[kl][rn] for kl in range(kv_lora)]))
        uo_head = decode_mlx_affine_3d_head_f32(
            readers, uo_w, uo_s, uo_b, uo_bits, uo_gs, n_heads, v_hd, kv_lora, h
        )
        for rv in range(v_hd):
            w.write(f32_to_bf16_bytes(uo_head[rv]))
    w.zeros(entry.blob_bytes - entry.data_bytes)


def _stream_entry(entry: _PlannedEntry, readers: _ShardReaders, w: _BlobWriter) -> None:
    if entry.layout == TensorLayout.SME2_PANEL64:
        _stream_sme2_entry(entry, readers, w)
        return
    if entry.mimo_qkv_slice is not None:
        _stream_mimo_qkv_entry(entry, readers, w)
        return
    if entry.mlx_kv_b_pair is not None:
        _stream_mlx_kv_b_entry(entry, readers, w)
        return
    if entry.data_convert == "hf_mxfp4_to_gguf_mxfp4":
        _stream_hf_mxfp4_entry(entry, readers, w)
        return
    if entry.data_convert == "fp8_block_to_bf16":
        _stream_fp8_block_to_bf16_entry(entry, readers, w)
        return
    if entry.data_convert is not None and entry.data_convert.startswith("affine") and entry.data_convert.endswith("_to_bf16"):
        _stream_affine_to_bf16_entry(entry, readers, w)
        return
    if entry.row_gather_runs is not None:
        _stream_row_gathered_entry(entry, readers, w)
        return
    src = readers.get(entry.data.path)
    if entry.data_convert in (None, "i64_raw_u32_pair"):
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
            elif entry.data_convert == "bf16_to_f32":
                w.write(_bf16_to_f32_bytes(raw))
            elif entry.data_convert == "f16_to_f32":
                w.write(_f16_to_f32_bytes(raw))
            else:
                raise ValueError(f"unknown conversion {entry.data_convert!r}")
            done += take

    if entry.params_bytes > 0:
        w.zeros(entry.params_offset - entry.data_bytes)
        if entry.params_mode == "affine_scale_bias":
            assert entry.scales is not None and entry.biases is not None
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
        elif entry.params_mode in ("fp8_block", "fp8_row"):
            assert entry.scales is not None
            _stream_promoted_f32(readers, entry.scales, w)
        elif entry.params_mode == "fp8_tensor":
            val = (
                _decode_scalar_f32_at(readers, entry.scales, 0)
                if entry.scales is not None
                else entry.outer_scale_const
            )
            w.write(struct.pack("<f", val))
        elif entry.params_mode == "mxfp8":
            _stream_mxfp8_scales(entry, readers, w)
        elif entry.params_mode == "nvfp4":
            assert entry.scales is not None
            _stream_raw_range_chunked(readers, entry.scales, w)
            pad4 = _align_up(entry.scales.length, 4) - entry.scales.length
            if pad4 > 0:
                w.zeros(pad4)
            idx = max(0, entry.outer_scale_expert_index)
            if entry.outer_scale is not None:
                val = _decode_scalar_f32_at(readers, entry.outer_scale, idx)
                if entry.outer_scale_denom is not None:
                    denom = _decode_scalar_f32_at(readers, entry.outer_scale_denom, idx)
                    if denom == 0.0:
                        raise ValueError(f"{entry.name}: outer_scale_denominator is zero")
                    eff = val / denom
                elif entry.outer_scale_divisor_mode:
                    if val == 0.0:
                        raise ValueError(f"{entry.name}: NVFP4 weight_global_scale is zero in divisor mode")
                    eff = 1.0 / val
                else:
                    eff = val / entry.outer_scale_denom_const
            else:
                eff = entry.outer_scale_const
            w.write(struct.pack("<f", eff))
        else:
            raise ValueError(f"unknown params_mode {entry.params_mode!r} for {entry.name}")
        content = entry.params_offset + entry.params_bytes
    else:
        content = entry.data_bytes
    w.zeros(entry.blob_bytes - content)


def _load_tie_flag(config_path: pathlib.Path | None) -> bool:
    if config_path is None:
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return bool(config.get("tie_word_embeddings", False)) if isinstance(config, dict) else False


def _validate_r0_loop_alpha_zero(e: _PlannedEntry, readers: _ShardReaders) -> None:
    """Enforce ADR-0019 `alpha = 0.0` at loop step `r = 0` for bit-exact identity."""
    if e.kind != TensorKind.LOOP_ADAPTER or not _is_loop_r0_alpha(e.name):
        return
    fh = readers.get(e.data.path)
    raw = _read_exact(fh, e.data.offset, e.data.length)
    if e.data.dtype == "F32":
        vals = f32_bytes_to_f32_list(raw)
    elif e.data.dtype == "F16":
        vals = f16_bytes_to_f32_list(raw)
    elif e.data.dtype == "BF16":
        vals = bf16_bytes_to_f32_list(raw)
    else:
        raise ValueError(f"loop adapter alpha tensor {e.name!r} has unsupported dtype {e.data.dtype}")
    if any(v != 0.0 for v in vals):
        raise ValueError(
            f"ADR-0019 violation: loop adapter {e.name!r} at step r=0 must have alpha == 0.0, got {vals!r}"
        )


def _load_quant_config_hints(
    input_path: pathlib.Path,
    config_path: pathlib.Path | None,
) -> tuple[bool, int | None, bool]:
    """Read `(tie_word_embeddings, config_group_size, nvfp4_global_scale_is_divisor)`."""
    tie = False
    cfg_gs: int | None = None
    gs_is_divisor = False
    cfg: dict[str, object] = {}
    if config_path is not None and config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg = loaded
    tie = bool(cfg.get("tie_word_embeddings", False))
    qc = cfg.get("quantization") or cfg.get("quantization_config")
    if isinstance(qc, dict):
        if isinstance(qc.get("group_size"), int) and qc["group_size"] in AFFINE_GROUP_SIZES:
            cfg_gs = int(qc["group_size"])
        qm = str(qc.get("quant_method") or "").lower()
        if qm == "compressed-tensors":
            gs_is_divisor = True
    base_dir = input_path if input_path.is_dir() else input_path.parent
    hf_qc = base_dir / "hf_quant_config.json"
    if hf_qc.is_file():
        try:
            loaded_hf = json.loads(hf_qc.read_text(encoding="utf-8"))
            if isinstance(loaded_hf, dict):
                qm = str(loaded_hf.get("quant_method") or "").lower()
                if qm == "compressed-tensors":
                    gs_is_divisor = True
        except ValueError:
            pass
    return tie, cfg_gs, gs_is_divisor


def _write_planned_amf(
    plan: Sequence[_PlannedEntry],
    readers: _ShardReaders,
    output_path: pathlib.Path,
    *,
    model_id: str,
    meta_blobs: Sequence[tuple[str, bytes]] = (),
    lut16_sme2: bool = False,
    workers: int = 1,
) -> list[PackedTensorMeta]:
    """Write planned tensors and metadata blobs directly into a 16 KiB-aligned `.amf` v0 file."""
    model_id_bytes = model_id.encode("utf-8")
    if len(model_id_bytes) == 0 or len(model_id_bytes) > 63:
        raise ValueError(f"model_id must be 1..63 UTF-8 bytes: {model_id!r}")

    tensor_count = len(plan) + len(meta_blobs)
    dir_offset = APPLE_PAGE_SIZE
    dir_raw_len = tensor_count * ENTRY_STRUCT.size
    dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)
    offsets: list[int] = []
    cursor = dir_offset + dir_bytes
    for e in plan:
        offsets.append(cursor)
        cursor += e.blob_bytes
    meta_offsets: list[tuple[str, bytes, int, int]] = []
    for m_name, m_bytes in meta_blobs:
        if len(m_bytes) == 0:
            raise ValueError(f"metadata payload for {m_name!r} must be non-empty")
        m_blob_bytes = _align_up(len(m_bytes), APPLE_PAGE_SIZE)
        meta_offsets.append((m_name, m_bytes, cursor, m_blob_bytes))
        cursor += m_blob_bytes

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    metas: list[PackedTensorMeta] = []
    try:
        with partial.open("wb") as out:
            _set_nocache(out.fileno())
            prefix = _BlobWriter(out)
            prefix.zeros(dir_offset + dir_bytes)
            for e, blob_offset in zip(plan, offsets):
                if out.tell() != blob_offset:
                    raise AssertionError(f"planned offset {blob_offset} != file position {out.tell()} for {e.name}")
                _validate_r0_loop_alpha_zero(e, readers)
                w = _BlobWriter(out)
                eff_param_kind = e.quant_param_kind
                if (
                    lut16_sme2
                    and e.layout == TensorLayout.SME2_PANEL64
                    and e.dtype in (DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE)
                ):
                    from aminfer_tools.gsq_rco import stream_safetensors_entry_lut16_sme2

                    stream_safetensors_entry_lut16_sme2(e, readers, w, workers=workers)
                    eff_param_kind = QuantParamKind.LUT16_CODEBOOK
                else:
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
                        quant_param_kind=eff_param_kind,
                    )
                )
            for m_name, m_bytes, m_offset, m_blob_bytes in meta_offsets:
                if out.tell() != m_offset:
                    raise AssertionError(f"planned meta offset {m_offset} != file position {out.tell()} for {m_name}")
                w = _BlobWriter(out)
                w.write(m_bytes)
                pad_len = m_blob_bytes - len(m_bytes)
                if pad_len > 0:
                    w.zeros(pad_len)
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
                        blob_offset=m_offset,
                        blob_bytes=m_blob_bytes,
                        data_bytes=len(m_bytes),
                        params_offset=0,
                        params_bytes=0,
                        blob_crc32=w.crc & 0xFFFFFFFF,
                        layout=TensorLayout.ROW_MAJOR,
                        quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                    )
                )
            if out.tell() != cursor:
                raise AssertionError(f"final size {out.tell()} != planned {cursor}")

            dir_raw = _pack_directory(metas)
            hdr_flags = FLAG_EMBEDDED_METADATA if meta_offsets else 0
            header = _pack_header(
                model_id_bytes,
                tensor_count,
                dir_offset,
                dir_bytes,
                _crc32(dir_raw),
                flags=hdr_flags,
            )
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
    return metas


def pack_amf_from_stream_sources(
    sources: Iterable[ShardStreamSource],
    output_path: pathlib.Path,
    *,
    model_id: str,
    group_size: int = 64,
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
    dtype_mode: str = "auto",
    tie_word_embeddings: bool = False,
    exclude_tensors: Iterable[str] = (),
    sme2_pack: str = "none",
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    sme2_pack_loop_adapters: bool = True,
    nvfp4_global_scale_is_divisor: bool = False,
    config_bytes: bytes | None = None,
    tokenizer_bytes: bytes | None = None,
    chat_template_bytes: bytes | None = None,
    extra_metadata: dict[str, bytes] | None = None,
    stats_out: dict[str, int] | None = None,
) -> list[PackedTensorMeta]:
    """Stream-pack safetensors shards straight from local iterators or HTTP `Range` readers without staging copies."""
    tensors, factories = _collect_tensors_from_sources(sources)
    eff_tie = tie_word_embeddings
    eff_divisor = nvfp4_global_scale_is_divisor
    eff_gs = group_size
    if config_bytes:
        try:
            cfg = json.loads(config_bytes.decode("utf-8"))
            if isinstance(cfg, dict):
                if not tie_word_embeddings and cfg.get("tie_word_embeddings"):
                    eff_tie = True
                qc = cfg.get("quantization") or cfg.get("quantization_config")
                if isinstance(qc, dict):
                    if str(qc.get("quant_method") or "").lower() == "compressed-tensors":
                        eff_divisor = True
        except ValueError:
            pass
    plan = plan_amf_entries(
        tensors,
        group_size=eff_gs,
        dtype_mode=dtype_mode,
        cold_experts=cold_experts,
        all_experts_cold=all_experts_cold,
        tie_word_embeddings=eff_tie,
        exclude_tensors=exclude_tensors,
        sme2_pack=sme2_pack,
        strip_vision=strip_vision,
        include_vision=include_vision,
        sme2_pack_loop_adapters=sme2_pack_loop_adapters,
        nvfp4_global_scale_is_divisor=eff_divisor,
        stats_out=stats_out,
    )
    meta_blobs: list[tuple[str, bytes]] = []
    if config_bytes is not None:
        meta_blobs.append((META_CONFIG_JSON, bytes(config_bytes)))
    if tokenizer_bytes is not None:
        meta_blobs.append((META_TOKENIZER_JSON, bytes(tokenizer_bytes)))
    if chat_template_bytes is not None:
        meta_blobs.append((META_CHAT_TEMPLATE_JSON, bytes(chat_template_bytes)))
    if extra_metadata:
        for k, v in sorted(extra_metadata.items()):
            if not is_meta_name(k):
                raise ValueError(f"metadata key must start with {META_PREFIX!r}, got {k!r}")
            meta_blobs.append((k, bytes(v)))

    readers = _ShardReaders(source_factories=factories)
    try:
        return _write_planned_amf(
            plan,
            readers,
            pathlib.Path(output_path),
            model_id=model_id,
            meta_blobs=meta_blobs,
        )
    finally:
        readers.close()


def _maybe_compile_chat_template_bytes(
    input_path: pathlib.Path,
    *,
    embed_metadata: bool,
    config_path: pathlib.Path | None,
    chat_template_path: pathlib.Path | None,
) -> bytes | None:
    if not embed_metadata and chat_template_path is None:
        return None
    base_dir = input_path if input_path.is_dir() else input_path.parent
    has_tpl = chat_template_path is not None or (base_dir / "chat_template.jinja").is_file() or (
        base_dir / "tokenizer_config.json"
    ).is_file()
    if not has_tpl:
        return None
    from aminfer_tools.chat_template import compile_chat_template_bytes

    return compile_chat_template_bytes(
        input_path,
        config_path=config_path,
        chat_template_path=chat_template_path,
    )


def pack_amf_from_safetensors(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str,
    group_size: int | None = None,
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
    dtype_mode: str = "auto",
    tie_word_embeddings: bool | None = None,
    exclude_tensors: Iterable[str] = (),
    heat_profile: pathlib.Path | None = None,
    cold_experts_per_layer: int | None = None,
    sme2_pack: str | None = None,
    lut16_sme2: bool = False,
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    use_rco: bool = True,
    manifest_path: pathlib.Path | None = None,
    workers: int = 1,
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
    sme2_pack_loop_adapters: bool = True,
    embed_metadata: bool = False,
    config_path: pathlib.Path | None = None,
    tokenizer_path: pathlib.Path | None = None,
    chat_template_path: pathlib.Path | None = None,
    nvfp4_global_scale_is_divisor: bool | None = None,
    imatrix_path: pathlib.Path | None = None,
    enforce_backbone_floor: bool = True,
    max_q2_share: float | None = None,
    allow_lowbit_q2_retranscode: bool = True,
    fine_rungs: bool = False,
    stats_out: dict[str, int] | None = None,
    mimo_config: Mapping[str, Any] | None = None,
    mimo_qkv_tp_degree: int | None = None,
    mimo_mtp_standalone: bool = False,
    mimo_dflash_mode: bool = False,
    qwen35_config: Mapping[str, Any] | None = None,
    qwen4_config: Mapping[str, Any] | None = None,
    qwen4_mtp_standalone: bool = False,
    split_stacked_gate_up: bool | None = None,
    glm5_config: Mapping[str, Any] | None = None,
    glm5_mtp_standalone: bool = False,
    glm5_mtp_source_layer: int | None = None,
    glm5_dflash2_mode: bool = False,
    ling3_config: Mapping[str, Any] | None = None,
    ling3_mtp_standalone: bool = False,
    ling3_mtp_source_layer: int | None = None,
    dense_kda_rco_q3: bool | None = None,
) -> list[PackedTensorMeta]:
    """Convert a safetensors checkpoint into a 16 KiB-aligned `.amf` v0 file.

    `input_path` may be a `.safetensors` file, a `model.safetensors.index.json`
    file, or a checkpoint directory. `quant_profile='uniform'` (default)
    preserves existing uniform packing behavior bit-exact; `quant_profile='gsq-rco'`
    runs optional ADR-0017 offline GSQ + RCO per-tensor / per-expert hybrid
    quantization targeting `target_bits` (auto: 4.15 Dense / 3.50 MoE when omitted).

    W20 vision-tower modes:
    - default (`include_vision=None, strip_vision=None, vision_sidecar=None`):
      automatically includes complete multi-layer ViT towers (`visual.blocks.*` /
      `vision.blocks.*`) while stripping dangling legacy vision stubs.
    - `include_vision=True` / `strip_vision=False`: keeps all vision tensors in
      `output_path`.
    - `strip_vision=True` / `include_vision=False`: strips all vision tensors.
    - `vision_sidecar=<path>`: extracts all vision tensors into standalone
      16 KiB-aligned `vision.amf` and omits them from the backbone `.amf`.
    """
    if quant_profile not in QUANT_PROFILES:
        raise ValueError(f"quant_profile must be one of {QUANT_PROFILES}, got {quant_profile!r}")
    if vision_block_dtype not in VISION_BLOCK_DTYPES:
        raise ValueError(
            f"vision_block_dtype must be one of {VISION_BLOCK_DTYPES}, got {vision_block_dtype!r}"
        )

    in_path = pathlib.Path(input_path)
    resolved_probe = resolve_checkpoint(in_path)
    tie_cfg, gs_cfg, div_cfg = _load_quant_config_hints(in_path, config_path or resolved_probe.config_path)
    eff_group_size = gs_cfg if group_size is None else group_size
    if eff_group_size is None:
        eff_group_size = 64
    eff_divisor = div_cfg if nvfp4_global_scale_is_divisor is None else nvfp4_global_scale_is_divisor
    probe_tensors = _collect_tensors(resolved_probe.shard_paths)
    if sme2_pack is None:
        eff_sme2_requested = (
            "controllers"
            if any(is_inkling2_controller_tensor(n) for n in probe_tensors)
            else "none"
        )
    else:
        eff_sme2_requested = sme2_pack
    vision_mode = resolve_vision_mode(
        probe_tensors.keys(),
        include_vision=include_vision,
        strip_vision=strip_vision,
        vision_sidecar=vision_sidecar,
    )
    vision_present = any(is_vision_tensor(n) for n in probe_tensors)

    if vision_mode == "sidecar" and vision_sidecar is not None and vision_present:
        vis_stats: dict[str, int] = {}
        sidecar_metas = pack_vision_tower_amf(
            input_path,
            vision_sidecar,
            model_id=f"{model_id[:55]}-vision",
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            vision_block_dtype=vision_block_dtype,
            group_size=eff_group_size,
            use_rco=use_rco,
            workers=workers,
            embed_metadata=embed_metadata,
            config_path=config_path,
            stats_out=vis_stats,
        )
        if stats_out is not None:
            stats_out["sidecar_vision_tensors"] = len(sidecar_metas)

    eff_strip_for_backbone = vision_mode in ("strip", "sidecar")

    if quant_profile == "gsq-rco":
        metas_gsq = gsq_convert_with_vision(
            input_path,
            output_path,
            target_bits=target_bits,
            model_id=model_id,
            group_size=eff_group_size,
            cold_experts=cold_experts,
            all_experts_cold=all_experts_cold,
            dtype_mode=dtype_mode,
            tie_word_embeddings=tie_word_embeddings,
            exclude_tensors=exclude_tensors,
            heat_profile=heat_profile,
            cold_experts_per_layer=cold_experts_per_layer,
            sme2_pack=eff_sme2_requested,
            lut16_sme2=lut16_sme2,
            use_rco=use_rco,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=eff_strip_for_backbone,
            include_vision=(vision_mode == "include"),
            vision_block_dtype=vision_block_dtype,
            imatrix_path=imatrix_path,
            enforce_backbone_floor=enforce_backbone_floor,
            max_q2_share=max_q2_share,
            allow_lowbit_q2_retranscode=allow_lowbit_q2_retranscode,
            fine_rungs=fine_rungs,
            dense_kda_rco_q3=dense_kda_rco_q3,
            stats_out=stats_out,
        )
        ct_bytes = _maybe_compile_chat_template_bytes(
            in_path,
            embed_metadata=embed_metadata,
            config_path=config_path or resolved_probe.config_path,
            chat_template_path=chat_template_path,
        )
        if embed_metadata or config_path is not None or tokenizer_path is not None or ct_bytes is not None:
            cfg_p, tok_p = _resolve_metadata_paths(
                in_path,
                resolved_probe.config_path,
                config_path=config_path,
                tokenizer_path=tokenizer_path,
            )
            eff_cfg_bytes: bytes | None = cfg_p.read_bytes() if cfg_p is not None else None
            if cfg_p is not None and not glm5_dflash2_mode and not glm5_mtp_standalone and not ling3_mtp_standalone:
                try:
                    raw_g5_obj = json.loads(cfg_p.read_text(encoding="utf-8"))
                except ValueError:
                    raw_g5_obj = None
                if isinstance(raw_g5_obj, dict):
                    from aminfer_tools.glm5 import (
                        build_glm5_next_config_json,
                        build_glm_moe_dsa_config_json,
                        is_glm5_config,
                        is_glm5_next_config,
                    )
                    from aminfer_tools.ling3 import build_ling3_config_json, is_ling3_config

                    if is_glm5_config(raw_g5_obj):
                        rt_g5 = (
                            build_glm5_next_config_json(
                                raw_g5_obj,
                                include_vision=not eff_strip_for_backbone,
                            )
                            if is_glm5_next_config(raw_g5_obj)
                            else build_glm_moe_dsa_config_json(raw_g5_obj)
                        )
                        eff_cfg_bytes = json.dumps(rt_g5, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
                    elif is_ling3_config(raw_g5_obj):
                        rt_l3 = build_ling3_config_json(
                            raw_g5_obj,
                            cold_experts_count=len({m.expert for m in metas_gsq if m.kind == TensorKind.COLD_EXPERT}),
                            dense_kda_rco_q3=bool(dense_kda_rco_q3),
                        )
                        eff_cfg_bytes = json.dumps(rt_l3, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
            return embed_metadata_into_amf(
                output_path,
                config_bytes=eff_cfg_bytes,
                tokenizer_bytes=tok_p.read_bytes() if tok_p is not None else None,
                chat_template_bytes=ct_bytes,
                output_path=output_path,
            )
        return metas_gsq

    if vision_mode == "include" and vision_present and vision_block_dtype == "q4_affine":
        # Pack backbone uniformly + vision with W20 q4_affine block quantization (pins patch_embed/norms/merger).
        return _pack_uniform_backbone_with_quant_vision(
            input_path,
            output_path,
            resolved=resolved_probe,
            tensors=probe_tensors,
            model_id=model_id,
            group_size=eff_group_size,
            cold_experts=cold_experts,
            all_experts_cold=all_experts_cold,
            dtype_mode=dtype_mode,
            tie_word_embeddings=tie_word_embeddings,
            exclude_tensors=exclude_tensors,
            heat_profile=heat_profile,
            cold_experts_per_layer=cold_experts_per_layer,
            sme2_pack=eff_sme2_requested,
            lut16_sme2=lut16_sme2,
            use_rco=use_rco,
            workers=workers,
            vision_block_dtype=vision_block_dtype,
            sme2_pack_loop_adapters=sme2_pack_loop_adapters,
            embed_metadata=embed_metadata,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            stats_out=stats_out,
        )

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

    eff_sme2_pack = "all-linear" if (lut16_sme2 and eff_sme2_requested == "none") else eff_sme2_requested
    resolved = resolved_probe
    tie = tie_cfg if tie_word_embeddings is None else tie_word_embeddings
    eff_mimo_cfg = mimo_config
    eff_mimo_tp = 1 if mimo_qkv_tp_degree is None else mimo_qkv_tp_degree
    eff_qwen35_cfg = qwen35_config
    eff_qwen4_cfg = qwen4_config
    eff_glm5_cfg = glm5_config
    eff_ling3_cfg = ling3_config
    cfg_probe_p = config_path or resolved.config_path
    if (
        eff_mimo_cfg is None
        and eff_qwen35_cfg is None
        and eff_qwen4_cfg is None
        and eff_glm5_cfg is None
        and eff_ling3_cfg is None
        and cfg_probe_p is not None
        and cfg_probe_p.is_file()
    ):
        probe_cfg_obj = json.loads(cfg_probe_p.read_text(encoding="utf-8"))
        if isinstance(probe_cfg_obj, dict):
            from aminfer_tools.glm5 import _effective_glm5_cfg, is_glm5_config
            from aminfer_tools.ling3 import is_ling3_config
            from aminfer_tools.mimo_v2 import (
                _effective_mimo_cfg,
                infer_mimo_qkv_tp_degree,
                is_mimo_v2_config,
                is_qwen35_config,
            )
            from aminfer_tools.qwen4_exp import _effective_text_config, is_qwen4_config

            if is_glm5_config(probe_cfg_obj) and not glm5_dflash2_mode:
                eff_glm5_cfg = probe_cfg_obj
            elif is_ling3_config(probe_cfg_obj):
                eff_ling3_cfg = probe_cfg_obj
            elif is_mimo_v2_config(probe_cfg_obj) and not mimo_dflash_mode:
                eff_mimo_cfg = _effective_mimo_cfg(probe_cfg_obj)
                if mimo_qkv_tp_degree is None:
                    eff_mimo_tp = infer_mimo_qkv_tp_degree(probe_cfg_obj, probe_tensors)
            elif is_qwen35_config(probe_cfg_obj):
                eff_qwen35_cfg = _effective_mimo_cfg(probe_cfg_obj)
            elif is_qwen4_config(probe_cfg_obj):
                eff_qwen4_cfg = _effective_text_config(probe_cfg_obj)
    tensors = probe_tensors
    plan = plan_amf_entries(
        tensors,
        group_size=eff_group_size,
        dtype_mode=dtype_mode,
        cold_experts=effective_cold,
        all_experts_cold=all_experts_cold,
        tie_word_embeddings=tie,
        exclude_tensors=exclude_tensors,
        sme2_pack=eff_sme2_pack,
        strip_vision=eff_strip_for_backbone,
        sme2_pack_loop_adapters=sme2_pack_loop_adapters,
        nvfp4_global_scale_is_divisor=eff_divisor,
        stats_out=stats_out,
        mimo_config=eff_mimo_cfg,
        mimo_qkv_tp_degree=eff_mimo_tp,
        mimo_mtp_standalone=mimo_mtp_standalone,
        mimo_dflash_mode=mimo_dflash_mode,
        qwen35_config=eff_qwen35_cfg,
        qwen4_config=eff_qwen4_cfg,
        qwen4_mtp_standalone=qwen4_mtp_standalone,
        split_stacked_gate_up=split_stacked_gate_up,
        glm5_config=eff_glm5_cfg,
        glm5_mtp_standalone=glm5_mtp_standalone,
        glm5_mtp_source_layer=glm5_mtp_source_layer,
        glm5_dflash2_mode=glm5_dflash2_mode,
        ling3_config=eff_ling3_cfg,
        ling3_mtp_standalone=ling3_mtp_standalone,
        ling3_mtp_source_layer=ling3_mtp_source_layer,
    )

    meta_blobs: list[tuple[str, bytes]] = []
    if embed_metadata or config_path is not None or tokenizer_path is not None or chat_template_path is not None:
        cfg_p, tok_p = _resolve_metadata_paths(
            in_path,
            resolved.config_path,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
        )
        if cfg_p is not None:
            if eff_glm5_cfg is not None and not glm5_dflash2_mode:
                from aminfer_tools.glm5 import (
                    build_glm5_mtp_config_dict,
                    build_glm5_next_config_json,
                    build_glm_moe_dsa_config_json,
                    is_glm5_next_config,
                )

                is_g5_next = is_glm5_next_config(eff_glm5_cfg)
                if glm5_mtp_standalone:
                    rt_g5 = build_glm5_mtp_config_dict(eff_glm5_cfg, is_glm5_next=is_g5_next)
                elif is_g5_next:
                    rt_g5 = build_glm5_next_config_json(
                        eff_glm5_cfg,
                        cold_experts_count=len({e for (_, e) in effective_cold}),
                        include_vision=not eff_strip_for_backbone,
                    )
                else:
                    rt_g5 = build_glm_moe_dsa_config_json(
                        eff_glm5_cfg,
                        cold_experts_count=len({e for (_, e) in effective_cold}),
                    )
                cfg_b = json.dumps(rt_g5, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
                meta_blobs.append((META_CONFIG_JSON, cfg_b))
            elif eff_ling3_cfg is not None:
                from aminfer_tools.ling3 import build_ling3_config_json, build_ling3_mtp_config_dict

                if ling3_mtp_standalone:
                    rt_l3 = build_ling3_mtp_config_dict(eff_ling3_cfg)
                else:
                    rt_l3 = build_ling3_config_json(
                        eff_ling3_cfg,
                        cold_experts_count=len({e for (_, e) in effective_cold}),
                        dense_kda_rco_q3=bool(dense_kda_rco_q3),
                    )
                cfg_b = json.dumps(rt_l3, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
                meta_blobs.append((META_CONFIG_JSON, cfg_b))
            elif eff_mimo_cfg is not None:
                from aminfer_tools.mimo_v2 import build_mimo_v2_config_json

                cfg_b = json.dumps(build_mimo_v2_config_json(eff_mimo_cfg), ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
                meta_blobs.append((META_CONFIG_JSON, cfg_b))
            elif eff_qwen35_cfg is not None:
                from aminfer_tools.mimo_v2 import build_qwen35_config_json

                cfg_b = json.dumps(build_qwen35_config_json(eff_qwen35_cfg), ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
                meta_blobs.append((META_CONFIG_JSON, cfg_b))
            else:
                meta_blobs.append((META_CONFIG_JSON, cfg_p.read_bytes()))
        if tok_p is not None:
            meta_blobs.append((META_TOKENIZER_JSON, tok_p.read_bytes()))
        ct_bytes = _maybe_compile_chat_template_bytes(
            in_path,
            embed_metadata=embed_metadata,
            config_path=cfg_p,
            chat_template_path=chat_template_path,
        )
        if ct_bytes is not None:
            meta_blobs.append((META_CHAT_TEMPLATE_JSON, ct_bytes))

    readers = _ShardReaders()
    try:
        return _write_planned_amf(
            plan,
            readers,
            pathlib.Path(output_path),
            model_id=model_id,
            meta_blobs=meta_blobs,
            lut16_sme2=lut16_sme2,
            workers=workers,
        )
    finally:
        readers.close()


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


def _pack_header(
    model_id_bytes: bytes,
    tensor_count: int,
    dir_offset: int,
    dir_bytes: int,
    dir_crc: int,
    *,
    flags: int = 0,
) -> bytes:
    fields = [
        AMF_MAGIC,
        AMF_VERSION,
        flags,
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


def _resolve_metadata_paths(
    input_path: pathlib.Path,
    resolved_config: pathlib.Path | None,
    *,
    config_path: pathlib.Path | None = None,
    tokenizer_path: pathlib.Path | None = None,
) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    base_dir = input_path if input_path.is_dir() else input_path.parent
    cfg = pathlib.Path(config_path) if config_path is not None else resolved_config
    if cfg is None:
        cand = base_dir / "config.json"
        if cand.is_file():
            cfg = cand
    elif not cfg.is_file():
        raise FileNotFoundError(f"config.json not found: {cfg}")

    tok = pathlib.Path(tokenizer_path) if tokenizer_path is not None else None
    if tok is None:
        cand = base_dir / "tokenizer.json"
        if cand.is_file():
            tok = cand
    elif not tok.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {tok}")
    return cfg, tok


def read_amf_metadata(amf_path: pathlib.Path, meta_name: str) -> bytes | None:
    """Read and CRC32-verify a cold embedded `__meta.*` payload from an `.amf` file (W18)."""
    header, metas = load_amf_metadata(pathlib.Path(amf_path))
    _ = header
    target = next((m for m in metas if m.name == meta_name), None)
    if target is None:
        return None
    if target.kind != TensorKind.METADATA:
        raise ValueError(f"entry {meta_name!r} is kind={target.kind}, expected TensorKind.METADATA")
    with pathlib.Path(amf_path).open("rb", buffering=0) as fh:
        raw_blob = _read_exact(fh, target.blob_offset, target.blob_bytes)
    got_crc = _crc32(raw_blob)
    if got_crc != target.blob_crc32:
        raise ValueError(
            f"metadata CRC mismatch for {meta_name!r}: expected {target.blob_crc32:#010x}, got {got_crc:#010x}"
        )
    return raw_blob[: target.data_bytes]


def embed_metadata_into_amf(
    amf_path: pathlib.Path,
    *,
    config_bytes: bytes | None = None,
    tokenizer_bytes: bytes | None = None,
    chat_template_bytes: bytes | None = None,
    extra_metadata: dict[str, bytes] | None = None,
    output_path: pathlib.Path | None = None,
) -> list[PackedTensorMeta]:
    """Attach or replace cold `__meta.*` JSON payloads inside an existing `.amf` file (W18)."""
    src_path = pathlib.Path(amf_path)
    dst_path = pathlib.Path(output_path) if output_path is not None else src_path
    attachments: dict[str, bytes] = {}
    if config_bytes is not None:
        attachments[META_CONFIG_JSON] = bytes(config_bytes)
    if tokenizer_bytes is not None:
        attachments[META_TOKENIZER_JSON] = bytes(tokenizer_bytes)
    if chat_template_bytes is not None:
        attachments[META_CHAT_TEMPLATE_JSON] = bytes(chat_template_bytes)
    if extra_metadata:
        for k, v in extra_metadata.items():
            if not is_meta_name(k):
                raise ValueError(f"metadata key must start with {META_PREFIX!r}, got {k!r}")
            attachments[k] = bytes(v)
    for k, v in attachments.items():
        if len(v) == 0:
            raise ValueError(f"metadata payload for {k!r} must be non-empty")

    header, old_metas = load_amf_metadata(src_path)
    kept = [m for m in old_metas if m.name not in attachments]
    total_count = len(kept) + len(attachments)
    dir_offset = APPLE_PAGE_SIZE
    dir_bytes = 0 if total_count == 0 else _align_up(total_count * ENTRY_STRUCT.size, APPLE_PAGE_SIZE)
    cursor = dir_offset + dir_bytes

    planned_kept: list[tuple[PackedTensorMeta, int]] = []
    for m in kept:
        planned_kept.append((m, cursor))
        cursor += m.blob_bytes
    planned_meta: list[tuple[str, bytes, int, int]] = []
    for m_name, m_bytes in sorted(attachments.items()):
        m_blob_bytes = _align_up(len(m_bytes), APPLE_PAGE_SIZE)
        planned_meta.append((m_name, m_bytes, cursor, m_blob_bytes))
        cursor += m_blob_bytes

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    partial = dst_path.with_name(dst_path.name + ".partial")
    new_metas: list[PackedTensorMeta] = []
    try:
        with src_path.open("rb", buffering=0) as src_fh, partial.open("wb") as out_fh:
            _set_nocache(src_fh.fileno())
            _set_nocache(out_fh.fileno())
            _BlobWriter(out_fh).zeros(dir_offset + dir_bytes)

            for m, new_off in planned_kept:
                w = _BlobWriter(out_fh)
                done = 0
                while done < m.blob_bytes:
                    take = min(_COPY_CHUNK_BYTES, m.blob_bytes - done)
                    w.write(_read_exact(src_fh, m.blob_offset + done, take))
                    done += take
                if (w.crc & 0xFFFFFFFF) != m.blob_crc32:
                    raise ValueError(f"source blob CRC mismatch for {m.name!r}")
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
                        blob_offset=new_off,
                        blob_bytes=m.blob_bytes,
                        data_bytes=m.data_bytes,
                        params_offset=m.params_offset,
                        params_bytes=m.params_bytes,
                        blob_crc32=m.blob_crc32,
                        layout=m.layout,
                        quant_param_kind=m.quant_param_kind,
                    )
                )

            for m_name, m_bytes, m_off, m_blob_bytes in planned_meta:
                w = _BlobWriter(out_fh)
                w.write(m_bytes)
                pad_len = m_blob_bytes - len(m_bytes)
                if pad_len > 0:
                    w.zeros(pad_len)
                new_metas.append(
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
                        blob_bytes=m_blob_bytes,
                        data_bytes=len(m_bytes),
                        params_offset=0,
                        params_bytes=0,
                        blob_crc32=w.crc & 0xFFFFFFFF,
                        layout=TensorLayout.ROW_MAJOR,
                        quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                    )
                )

            dir_raw = _pack_directory(new_metas)
            has_meta = any(m.kind == TensorKind.METADATA for m in new_metas)
            hdr_flags = (header.flags | FLAG_EMBEDDED_METADATA) if has_meta else (header.flags & ~FLAG_EMBEDDED_METADATA)
            new_hdr = _pack_header(
                header.model_id.encode("utf-8"),
                len(new_metas),
                dir_offset,
                dir_bytes,
                _crc32(dir_raw),
                flags=hdr_flags,
            )
            out_fh.seek(0)
            out_fh.write(new_hdr)
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

# 8-bit lookup table mapping 4 packed 2-bit unsigned codes -> (c0, c1, c2, c3).
_Q2_LUT8: tuple[tuple[int, int, int, int], ...] = tuple(
    (b & 0x03, (b >> 2) & 0x03, (b >> 4) & 0x03, (b >> 6) & 0x03) for b in range(256)
)


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


def pack_sme2_panel64_q2_from_mlx_panel(
    panel_data_rows: Sequence[bytes | bytearray | memoryview],
    panel_params_rows: Sequence[bytes | bytearray | memoryview] | None,
    k: int,
    *,
    param_kind: QuantParamKind = QuantParamKind.AFFINE_SCALE_BIAS,
    group_headers: Sequence[bytes | bytearray | memoryview] | None = None,
) -> bytes:
    """Pack 1..16 MLX-format row-major `q2_affine` rows (`k // 4` bytes/row) into one `sme2_panel64` panel."""
    n_active = len(panel_data_rows)
    if not (1 <= n_active <= SME2_PANEL_COLS):
        raise ValueError(f"panel_data_rows must contain 1..16 rows, got {n_active}")
    if k <= 0 or k % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires k > 0 and k % 64 == 0, got {k}")
    row_data_bytes = k // 4
    lut = _Q2_LUT8
    codes_rows: list[list[int]] = []
    for c, row in enumerate(panel_data_rows):
        mv = memoryview(row)
        if len(mv) != row_data_bytes:
            raise ValueError(f"q2 panel row {c} has {len(mv)} data bytes, expected {row_data_bytes}")
        codes = [0] * k
        for i in range(row_data_bytes):
            c0, c1, c2, c3 = lut[mv[i]]
            base = 4 * i
            codes[base] = c0
            codes[base + 1] = c1
            codes[base + 2] = c2
            codes[base + 3] = c3
        codes_rows.append(codes)
    return pack_sme2_panel64_q2_panel(
        codes_rows,
        panel_params_rows,
        k,
        param_kind=param_kind,
        group_headers=group_headers,
    )


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


def dequantize_q2_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_params: bytes | bytearray | memoryview,
    k: int,
    *,
    group_size: int = 64,
) -> list[float]:
    """Dequantize one row of MLX-compatible 2-bit affine packed weights to floats."""
    if k <= 0 or group_size <= 0 or group_size % 4 != 0 or k % group_size != 0:
        raise ValueError(f"invalid q2 geometry: k={k}, group_size={group_size}")
    groups = k // group_size
    expected_data = k // 4
    expected_params = groups * 4
    if len(row_data) != expected_data:
        raise ValueError(f"q2 row_data length {len(row_data)} != expected {expected_data}")
    if len(row_params) != expected_params:
        raise ValueError(f"q2 row_params length {len(row_params)} != expected {expected_params}")

    sbs = struct.unpack(f"<{groups * 2}e", row_params)
    bytes_per_group = group_size // 4
    lut8 = _Q2_LUT8
    out = [0.0] * k
    pos = 0
    off = 0
    sb_idx = 0
    for _ in range(groups):
        s = sbs[sb_idx]
        b = sbs[sb_idx + 1]
        sb_idx += 2
        v = (b, b + s, b + 2.0 * s, b + 3.0 * s)
        for _ in range(bytes_per_group):
            c0, c1, c2, c3 = lut8[row_data[off]]
            off += 1
            out[pos] = v[c0]
            out[pos + 1] = v[c1]
            out[pos + 2] = v[c2]
            out[pos + 3] = v[c3]
            pos += 4
    return out


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


def dequantize_q5_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_params: bytes | bytearray | memoryview,
    k: int,
    *,
    group_size: int = 64,
) -> list[float]:
    """Dequantize one row of MLX-compatible 5-bit affine packed weights to floats."""
    if k <= 0 or group_size <= 0 or group_size % 8 != 0 or k % group_size != 0:
        raise ValueError(f"invalid q5 geometry: k={k}, group_size={group_size}")
    groups = k // group_size
    expected_data = (k * 5) // 8
    expected_params = groups * 4
    if len(row_data) != expected_data:
        raise ValueError(f"q5 row_data length {len(row_data)} != expected {expected_data}")
    if len(row_params) != expected_params:
        raise ValueError(f"q5 row_params length {len(row_params)} != expected {expected_params}")

    sbs = struct.unpack(f"<{groups * 2}e", row_params)
    blocks_per_group = group_size // 8
    out = [0.0] * k
    pos = 0
    off = 0
    sb_idx = 0
    for _ in range(groups):
        s = sbs[sb_idx]
        b = sbs[sb_idx + 1]
        sb_idx += 2
        for _ in range(blocks_per_group):
            w = (
                row_data[off]
                | (row_data[off + 1] << 8)
                | (row_data[off + 2] << 16)
                | (row_data[off + 3] << 24)
                | (row_data[off + 4] << 32)
            )
            off += 5
            for j in range(8):
                out[pos + j] = b + ((w >> (5 * j)) & 0x1F) * s
            pos += 8
    return out


def dequantize_q6_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_params: bytes | bytearray | memoryview,
    k: int,
    *,
    group_size: int = 64,
) -> list[float]:
    """Dequantize one row of MLX-compatible 6-bit affine packed weights to floats."""
    if k <= 0 or group_size <= 0 or group_size % 4 != 0 or k % group_size != 0:
        raise ValueError(f"invalid q6 geometry: k={k}, group_size={group_size}")
    groups = k // group_size
    expected_data = (k * 3) // 4
    expected_params = groups * 4
    if len(row_data) != expected_data:
        raise ValueError(f"q6 row_data length {len(row_data)} != expected {expected_data}")
    if len(row_params) != expected_params:
        raise ValueError(f"q6 row_params length {len(row_params)} != expected {expected_params}")

    sbs = struct.unpack(f"<{groups * 2}e", row_params)
    blocks_per_group = group_size // 4
    out = [0.0] * k
    pos = 0
    off = 0
    sb_idx = 0
    for _ in range(groups):
        s = sbs[sb_idx]
        b = sbs[sb_idx + 1]
        sb_idx += 2
        for _ in range(blocks_per_group):
            w = row_data[off] | (row_data[off + 1] << 8) | (row_data[off + 2] << 16)
            off += 3
            for j in range(4):
                out[pos + j] = b + ((w >> (6 * j)) & 0x3F) * s
            pos += 4
    return out


def dequantize_q8_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_params: bytes | bytearray | memoryview,
    k: int,
    *,
    group_size: int = 64,
) -> list[float]:
    """Dequantize one row of MLX-compatible 8-bit affine packed weights to floats."""
    if k <= 0 or group_size <= 0 or group_size % 4 != 0 or k % group_size != 0:
        raise ValueError(f"invalid q8 geometry: k={k}, group_size={group_size}")
    groups = k // group_size
    expected_data = k
    expected_params = groups * 4
    if len(row_data) != expected_data:
        raise ValueError(f"q8 row_data length {len(row_data)} != expected {expected_data}")
    if len(row_params) != expected_params:
        raise ValueError(f"q8 row_params length {len(row_params)} != expected {expected_params}")

    sbs = struct.unpack(f"<{groups * 2}e", row_params)
    out = [0.0] * k
    pos = 0
    sb_idx = 0
    for _ in range(groups):
        s = sbs[sb_idx]
        b = sbs[sb_idx + 1]
        sb_idx += 2
        for j in range(group_size):
            out[pos + j] = b + row_data[pos + j] * s
        pos += group_size
    return out


def dequantize_fp8_block_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_idx: int,
    block_scales_f32: Sequence[float],
    n_rows: int,
    k: int,
) -> list[float]:
    if len(row_data) != k:
        raise ValueError(f"fp8 block row length {len(row_data)} != {k}")
    kb = (k + 127) // 128
    nb_idx = row_idx // 128
    base = nb_idx * kb
    lut = _E4M3_LUT
    out = [0.0] * k
    for col in range(k):
        out[col] = lut[row_data[col]] * block_scales_f32[base + col // 128]
    return out


def dequantize_fp8_row_f32(row_data: bytes | bytearray | memoryview, row_scale: float, k: int) -> list[float]:
    if len(row_data) != k:
        raise ValueError(f"fp8 row length {len(row_data)} != {k}")
    lut = _E4M3_LUT
    return [lut[row_data[c]] * row_scale for c in range(k)]


def dequantize_fp8_tensor_f32(row_data: bytes | bytearray | memoryview, scale: float, k: int) -> list[float]:
    if len(row_data) != k:
        raise ValueError(f"fp8 tensor row length {len(row_data)} != {k}")
    lut = _E4M3_LUT
    return [lut[row_data[c]] * scale for c in range(k)]


def dequantize_mxfp8_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_params: bytes | bytearray | memoryview,
    k: int,
) -> list[float]:
    if k <= 0 or k % 32 != 0:
        raise ValueError(f"invalid mxfp8 row width k={k}")
    groups = k // 32
    if len(row_data) != k or len(row_params) != groups:
        raise ValueError(f"mxfp8 row length mismatch: data={len(row_data)}, params={len(row_params)}")
    e_lut = _E4M3_LUT
    u_lut = _UE8M0_LUT
    out = [0.0] * k
    for g in range(groups):
        sc = u_lut[row_params[g]]
        base = g * 32
        for j in range(32):
            out[base + j] = e_lut[row_data[base + j]] * sc
    return out


def dequantize_nvfp4_row_f32(
    row_data: bytes | bytearray | memoryview,
    row_scales_e4m3: bytes | bytearray | memoryview,
    outer_scale: float,
    k: int,
) -> list[float]:
    if k <= 0 or k % 16 != 0:
        raise ValueError(f"invalid nvfp4 row width k={k}")
    groups = k // 16
    if len(row_data) != k // 2 or len(row_scales_e4m3) != groups:
        raise ValueError(f"nvfp4 row length mismatch: data={len(row_data)}, scales={len(row_scales_e4m3)}")
    e_lut = _E4M3_LUT
    n_lut = _E2M1_TABLE
    out = [0.0] * k
    for g in range(groups):
        sc = e_lut[row_scales_e4m3[g]] * outer_scale
        d_off = g * 8
        p_off = g * 16
        for b_i in range(8):
            byte = row_data[d_off + b_i]
            out[p_off + 2 * b_i] = n_lut[byte & 0x0F] * sc
            out[p_off + 2 * b_i + 1] = n_lut[(byte >> 4) & 0x0F] * sc
    return out


_AFFINE_DEQUANT_SPEC = {
    DType.Q2_AFFINE: (lambda k: k // 4, dequantize_q2_row_f32),
    DType.Q3_AFFINE: (lambda k: (k * 3) // 8, dequantize_q3_row_f32),
    DType.Q4_AFFINE: (lambda k: k // 2, dequantize_q4_row_f32),
    DType.Q5_AFFINE: (lambda k: (k * 5) // 8, dequantize_q5_row_f32),
    DType.Q6_AFFINE: (lambda k: (k * 3) // 4, dequantize_q6_row_f32),
    DType.Q8_AFFINE: (lambda k: k, dequantize_q8_row_f32),
}


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

    def read_metadata_bytes(self, name: str, *, verify_crc: bool = True) -> bytes:
        """Read and unpad a cold `__meta.*` JSON/binary blob by name (supports W24 chat template alias)."""
        entry = self.by_name.get(name)
        if entry is None and name == META_CHAT_TEMPLATE_JSON:
            entry = self.by_name.get(LEGACY_CHAT_TEMPLATE_METADATA_NAME)
        elif entry is None and name == LEGACY_CHAT_TEMPLATE_METADATA_NAME:
            entry = self.by_name.get(META_CHAT_TEMPLATE_JSON)
        if entry is None:
            raise KeyError(f"metadata entry {name!r} not found in {self.path}")
        if entry.kind != TensorKind.METADATA:
            raise ValueError(f"entry {name!r} has kind={entry.kind}, expected TensorKind.METADATA")
        raw = self.read_blob(entry, verify_crc=verify_crc)
        return raw[: entry.data_bytes]

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
        if entry.dtype == DType.Q2_AFFINE:
            panel_bytes = groups * SME2_Q2_GROUP_BYTES
            raw = _read_exact(self._fh, entry.blob_offset + panel_idx * panel_bytes, panel_bytes)
            return dequantize_sme2_panel64_q2_panel_f32(
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

        if entry.dtype in _AFFINE_DEQUANT_SPEC:
            row_data_fn, deq = _AFFINE_DEQUANT_SPEC[entry.dtype]
            gs = entry.group_size
            groups = k // gs
            row_data_bytes = row_data_fn(k)
            row_param_bytes = groups * 4
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

        if entry.dtype == DType.FP8_E4M3_BLOCK:
            kb = (k + 127) // 128
            nb = (n_rows + 127) // 128
            scales_raw = _read_exact(self._fh, entry.blob_offset + entry.params_offset, nb * kb * 4)
            scales_f32 = struct.unpack(f"<{nb * kb}f", scales_raw)
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                d_raw = _read_exact(self._fh, entry.blob_offset + r * k, k)
                out.append(dequantize_fp8_block_row_f32(d_raw, r, scales_f32, n_rows, k))
            return out

        if entry.dtype == DType.FP8_E4M3_ROW:
            scales_raw = _read_exact(self._fh, entry.blob_offset + entry.params_offset, n_rows * 4)
            scales_f32 = struct.unpack(f"<{n_rows}f", scales_raw)
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                d_raw = _read_exact(self._fh, entry.blob_offset + r * k, k)
                out.append(dequantize_fp8_row_f32(d_raw, scales_f32[r], k))
            return out

        if entry.dtype == DType.FP8_E4M3_TENSOR:
            sc = struct.unpack("<f", _read_exact(self._fh, entry.blob_offset + entry.params_offset, 4))[0]
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                d_raw = _read_exact(self._fh, entry.blob_offset + r * k, k)
                out.append(dequantize_fp8_tensor_f32(d_raw, sc, k))
            return out

        if entry.dtype == DType.MXFP8_G32:
            groups = k // 32
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                d_raw = _read_exact(self._fh, entry.blob_offset + r * k, k)
                p_raw = _read_exact(self._fh, entry.blob_offset + entry.params_offset + r * groups, groups)
                out.append(dequantize_mxfp8_row_f32(d_raw, p_raw, k))
            return out

        if entry.dtype == DType.NVFP4_G16:
            groups = k // 16
            row_data_bytes = k // 2
            outer_off = entry.blob_offset + entry.params_offset + _align_up(n_rows * groups, 4)
            outer_scale = struct.unpack("<f", _read_exact(self._fh, outer_off, 4))[0]
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                d_raw = _read_exact(self._fh, entry.blob_offset + r * row_data_bytes, row_data_bytes)
                p_raw = _read_exact(self._fh, entry.blob_offset + entry.params_offset + r * groups, groups)
                out.append(dequantize_nvfp4_row_f32(d_raw, p_raw, outer_scale, k))
            return out

        if entry.dtype >= DType.GGUF_Q4_0:
            from aminfer_tools.gguf import dequantize_gguf_amf_row_f32, gguf_row_bytes_for_amf_dtype

            row_b = gguf_row_bytes_for_amf_dtype(entry.dtype, k)
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {entry.name!r} with {n_rows} rows")
                raw = _read_exact(self._fh, entry.blob_offset + r * row_b, row_b)
                out.append(dequantize_gguf_amf_row_f32(entry.dtype, raw, k))
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
        if entry.dtype in (
            *_AFFINE_DEQUANT_SPEC,
            DType.FP8_E4M3_BLOCK,
            DType.FP8_E4M3_ROW,
            DType.FP8_E4M3_TENSOR,
            DType.MXFP8_G32,
            DType.NVFP4_G16,
        ) or entry.dtype >= DType.GGUF_Q4_0:
            return self.read_rows_f32(entry, range(n_rows))

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
        if (
            entry.dtype in _AFFINE_DEQUANT_SPEC
            or entry.dtype
            in (DType.FP8_E4M3_BLOCK, DType.FP8_E4M3_ROW, DType.FP8_E4M3_TENSOR, DType.MXFP8_G32, DType.NVFP4_G16)
            or entry.dtype >= DType.GGUF_Q4_0
        ):
            if len(entry.shape) != 2:
                raise ValueError(f"quantized tensor {entry.name!r} must be 2D, got {entry.shape}")
            n_rows, k = entry.shape
            out = bytearray(n_rows * k * 2)
            row_packer = struct.Struct(f"<{k}e")
            row_out_bytes = k * 2
            for r, vals in enumerate(self.read_rows_f32(entry, range(n_rows))):
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
    elif entry.dtype in (DType.Q4_AFFINE, DType.Q3_AFFINE, DType.Q2_AFFINE):
        groups = k // entry.group_size
        if entry.dtype == DType.Q4_AFFINE:
            row_data_bytes = k // 2
            packer = pack_sme2_panel64_q4_panel
        elif entry.dtype == DType.Q3_AFFINE:
            row_data_bytes = (k * 3) // 8
            packer = pack_sme2_panel64_q3_panel
        else:
            row_data_bytes = k // 4
            packer = pack_sme2_panel64_q2_from_mlx_panel
        row_param_bytes = groups * 4
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
    lut16: bool = False,
    lloyd_iters: int = 10,
    workers: int = 1,
    verify_source_crc: bool = True,
) -> list[PackedTensorMeta]:
    """Stream-convert selected 2D tensors in an `.amf` v0 file into single-owner `sme2_panel64`.

    Preserves single-owner layout (every tensor appears once — never duplicated in both
    `row_major` and `sme2_panel64`), 16 KiB page alignment, IEEE CRC-32 checksums, and
    all metadata (`kind`, `dtype`, `layer`, `expert`, `shape`). When `lut16=True`,
    eligible quantized (`q2`/`q3`/`q4`) tensors are fitted with per-`(panel, group)`
    non-uniform Lloyd-Max `ZT0` codebooks + per-column `f16` RCO biases and tagged
    with `QuantParamKind.LUT16_CODEBOOK` (`reserved0 == 0x0101`). Already-packed
    entries matching the target `(layout, param_kind)` are copied verbatim so
    repacking is idempotent.
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
        eligible = _should_sme2_pack(m.kind, m.dtype, m.shape, targets, name=m.name)
        want_lut16 = lut16 and m.dtype in (DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE)
        needs_repack = eligible and (
            m.layout == TensorLayout.ROW_MAJOR
            or (want_lut16 and m.quant_param_kind != QuantParamKind.LUT16_CODEBOOK)
        )
        if needs_repack:
            new_layout = TensorLayout.SME2_PANEL64
            new_param_kind = QuantParamKind.LUT16_CODEBOOK if want_lut16 else QuantParamKind.AFFINE_SCALE_BIAS
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
                    if new_param_kind == QuantParamKind.LUT16_CODEBOOK:
                        from aminfer_tools.gsq_rco import repack_amf_entry_lut16_sme2

                        repack_amf_entry_lut16_sme2(
                            src_fh,
                            m,
                            new_blob_bytes,
                            w,
                            lloyd_iters=lloyd_iters,
                            workers=workers,
                        )
                    else:
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
                flags=header.flags,
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


# ---------------------------------------------------------------------------
# W20 Vision Tower Packing, W14 GSQ+RCO Vision Policy & Synthetic ViT Bundle
# ---------------------------------------------------------------------------


def _f32_values_to_bf16_bytes(values: Sequence[float]) -> bytes:
    out = bytearray(len(values) * 2)
    for i, v in enumerate(values):
        bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        # Round-to-nearest-even to BF16 (matching PyTorch / IEEE 754).
        if (bits & 0x7FFFFFFF) > 0x7F800000:
            bf = (bits >> 16) | 0x0040
        else:
            rounding = 0x7FFF + ((bits >> 16) & 1)
            bf = (bits + rounding) >> 16
        struct.pack_into("<H", out, i * 2, bf & 0xFFFF)
    return bytes(out)


def f32_to_bf16_bytes(values: Sequence[float]) -> bytes:
    """Pack float values as little-endian IEEE bfloat16 bytes."""
    return _f32_values_to_bf16_bytes(values)


def f32_to_f16_bytes(values: Sequence[float], *, what: str = "tensor") -> bytes:
    """Pack float values as little-endian IEEE float16 bytes."""
    return f32_values_to_f16_bytes(values, what=what)


def _read_planned_entry_bytes(
    e: _PlannedEntry,
    readers: _ShardReaders,
) -> tuple[bytes, bytes]:
    """Read a planned entry's converted data and optional params bytes from safetensors."""
    fh = readers.get(e.data.path)
    raw = _read_exact(fh, e.data.offset, e.data.length)
    if e.data_convert == "bf16_to_f16":
        data = bf16_bytes_to_f16_bytes(raw, what=e.name)
    elif e.data_convert == "f16_to_bf16":
        data = f16_bytes_to_bf16_bytes(raw)
    elif e.data_convert == "bf16_to_f32":
        data = bytes(_bf16_to_f32_bytes(raw))
    elif e.data_convert == "f16_to_f32":
        data = _f16_to_f32_bytes(raw)
    elif e.data_convert == "d2t_i64_to_u32":
        vals = struct.unpack(f"<{len(raw) // 8}q", raw)
        data = struct.pack(f"<{len(vals)}I", *vals)
    elif e.data_convert == "d2t_i32_to_u32":
        vals = struct.unpack(f"<{len(raw) // 4}i", raw)
        data = struct.pack(f"<{len(vals)}I", *vals)
    else:
        data = raw

    params = b""
    if e.scales is not None and e.biases is not None:
        sfh = readers.get(e.scales.path)
        bfh = readers.get(e.biases.path)
        s_raw = _read_exact(sfh, e.scales.offset, e.scales.length)
        b_raw = _read_exact(bfh, e.biases.offset, e.biases.length)
        s_f16 = _to_f16_chunk(s_raw, e.scales.dtype, f"{e.name}.scales")
        b_f16 = _to_f16_chunk(b_raw, e.biases.dtype, f"{e.name}.biases")
        params = interleave_scale_bias_f16(s_f16, b_f16)
    return data, params


def _decode_raw_to_f32_rows(raw: bytes, dtype_str: str, rows: int, cols: int) -> list[list[float]]:
    if dtype_str == "F16":
        flat = f16_bytes_to_f32_list(raw)
    elif dtype_str == "BF16":
        flat = bf16_bytes_to_f32_list(raw)
    elif dtype_str == "F32":
        flat = f32_bytes_to_f32_list(raw)
    else:
        raise ValueError(f"unsupported float dtype {dtype_str!r} for vision quantization")
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def _build_vision_records_from_safetensors(
    vision_tensors: dict[str, _ShardTensor],
    readers: _ShardReaders,
    *,
    dtype_mode: str = "auto",
    quantize_blocks_q4: bool = False,
    vision_block_dtype: str = "auto",
    group_size: int = 64,
    use_rco: bool = True,
    workers: int = 1,
) -> list[tuple]:
    """Build 12-tuple `write_amf` records for vision-tower tensors with W20 GSQ+RCO policy.

    - Pinned tensors (`patch_embed`, `pos_embed`, all norms, 1D biases, `merger.*`,
      `aligner.*`, `image_start`/`image_end`/`image_newline`) always remain in `f16`/`bf16`
      (or source `f32` when `dtype_mode == 'auto'` and source is `F32`).
    - Vision encoder block 2D linear projections (`visual.blocks.*` / `vision.blocks.*`)
      are quantized to `q4_affine` (via W14 RCO when `use_rco=True`) when
      `quantize_blocks_q4` or `vision_block_dtype == 'q4_affine'` and `K % group_size == 0`,
      or kept in `f16`/`bf16`.
    """
    if vision_block_dtype not in VISION_BLOCK_DTYPES:
        raise ValueError(
            f"vision_block_dtype must be one of {VISION_BLOCK_DTYPES}, got {vision_block_dtype!r}"
        )
    plan = plan_amf_entries(
        vision_tensors,
        group_size=group_size,
        dtype_mode=dtype_mode,
        tie_word_embeddings=False,
        sme2_pack="none",
        strip_vision=False,
    )
    should_q4_blocks = (vision_block_dtype == "q4_affine") or (
        quantize_blocks_q4 and vision_block_dtype == "auto"
    )
    quant_fn = None
    if should_q4_blocks:
        from aminfer_tools.gsq_rco import quantize_rows_affine

        quant_fn = quantize_rows_affine

    used_names: set[str] = set()
    records: list[tuple] = []
    for e in plan:
        pinned = is_vision_pinned_float(e.name, e.shape)
        is_block_lin = is_vision_block_linear(e.name, e.shape)

        if (
            e.quant_bits == 0
            and is_block_lin
            and not pinned
            and should_q4_blocks
            and quant_fn is not None
            and len(e.shape) == 2
            and e.shape[1] > 0
            and e.shape[1] % group_size == 0
            and e.data.dtype in ("F16", "BF16", "F32")
        ):
            fh = readers.get(e.data.path)
            raw = _read_exact(fh, e.data.offset, e.data.length)
            rows = _decode_raw_to_f32_rows(raw, e.data.dtype, e.shape[0], e.shape[1])
            q_data, q_params, _mse, _mm_mse = quant_fn(
                rows,
                bits=4,
                group_size=group_size,
                use_rco=use_rco,
            )
            out_name = e.name[: -len(".weight")] if e.name.endswith(".weight") else e.name
            if out_name in used_names:
                out_name = e.name
            used_names.add(out_name)
            records.append(
                (
                    out_name,
                    e.kind,
                    DType.Q4_AFFINE,
                    4,
                    group_size,
                    e.layer,
                    e.expert,
                    e.shape,
                    q_data,
                    q_params,
                    TensorLayout.ROW_MAJOR,
                    QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )
            continue

        data, params = _read_planned_entry_bytes(e, readers)
        out_dtype = e.dtype
        if e.quant_bits == 0 and is_block_lin and vision_block_dtype in ("f16", "bf16"):
            if vision_block_dtype == "f16" and out_dtype == DType.BF16:
                data = bf16_bytes_to_f16_bytes(data, what=e.name)
                out_dtype = DType.F16
            elif vision_block_dtype == "bf16" and out_dtype == DType.F16:
                data = f16_bytes_to_bf16_bytes(data)
                out_dtype = DType.BF16
        elif e.quant_bits == 0 and pinned and out_dtype == DType.F32 and dtype_mode in ("f16", "bf16"):
            f32_vals = f32_bytes_to_f32_list(data)
            if dtype_mode == "f16":
                data = f32_values_to_f16_bytes(f32_vals, what=e.name)
                out_dtype = DType.F16
            else:
                data = _f32_values_to_bf16_bytes(f32_vals)
                out_dtype = DType.BF16
        used_names.add(e.name)
        records.append(
            (
                e.name,
                e.kind,
                out_dtype,
                e.quant_bits,
                e.group_size,
                e.layer,
                e.expert,
                e.shape,
                data,
                params,
                e.layout,
                e.quant_param_kind,
            )
        )
    records.sort(key=lambda r: _natural_key(r[0]))
    return records


def _build_vision_records_from_amf(
    reader: AmfReader,
    *,
    dtype_mode: str = "auto",
    quantize_blocks_q4: bool = False,
    vision_block_dtype: str = "auto",
    group_size: int = 64,
    use_rco: bool = True,
    workers: int = 1,
) -> list[tuple]:
    """Extract and optionally W20-quantize vision entries from an existing `.amf`."""
    should_q4_blocks = (vision_block_dtype == "q4_affine") or (
        quantize_blocks_q4 and vision_block_dtype == "auto"
    )
    quant_fn = None
    if should_q4_blocks:
        from aminfer_tools.gsq_rco import quantize_rows_affine

        quant_fn = quantize_rows_affine

    used_names: set[str] = set()
    records: list[tuple] = []
    for m in reader.entries:
        if not is_vision_tensor(m.name):
            continue
        norm_name = _normalize_name(m.name)
        pinned = is_vision_pinned_float(norm_name, m.shape)
        is_block_lin = is_vision_block_linear(norm_name, m.shape)
        blob = reader.read_blob(m, verify_crc=True)
        data = blob[: m.data_bytes]
        params = blob[m.params_offset : m.params_offset + m.params_bytes] if m.params_bytes > 0 else b""

        if (
            m.quant_bits == 0
            and is_block_lin
            and not pinned
            and should_q4_blocks
            and quant_fn is not None
            and len(m.shape) == 2
            and m.shape[1] > 0
            and m.shape[1] % group_size == 0
            and m.dtype in (DType.F16, DType.BF16, DType.F32)
        ):
            src_dt_str = m.dtype.name
            rows = _decode_raw_to_f32_rows(data, src_dt_str, m.shape[0], m.shape[1])
            q_data, q_params, _mse, _mm_mse = quant_fn(
                rows,
                bits=4,
                group_size=group_size,
                use_rco=use_rco,
            )
            out_name = norm_name[: -len(".weight")] if norm_name.endswith(".weight") else norm_name
            if out_name in used_names:
                out_name = norm_name
            used_names.add(out_name)
            records.append(
                (
                    out_name,
                    m.kind,
                    DType.Q4_AFFINE,
                    4,
                    group_size,
                    m.layer,
                    m.expert,
                    m.shape,
                    q_data,
                    q_params,
                    TensorLayout.ROW_MAJOR,
                    QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )
            continue

        out_dtype = m.dtype
        if m.quant_bits == 0:
            want = vision_block_dtype if (is_block_lin and vision_block_dtype in ("f16", "bf16")) else dtype_mode
            if want == "f16" and out_dtype == DType.BF16:
                data = bf16_bytes_to_f16_bytes(data, what=norm_name)
                out_dtype = DType.F16
            elif want == "bf16" and out_dtype == DType.F16:
                data = f16_bytes_to_bf16_bytes(data)
                out_dtype = DType.BF16
        used_names.add(norm_name)
        records.append(
            (
                norm_name,
                m.kind,
                out_dtype,
                m.quant_bits,
                m.group_size,
                m.layer,
                m.expert,
                m.shape,
                data,
                params,
                m.layout,
                m.quant_param_kind,
            )
        )
    records.sort(key=lambda r: _natural_key(r[0]))
    return records


def _merge_amf_with_vision_records(
    backbone_amf_path: pathlib.Path,
    vision_records: Sequence[tuple],
    output_path: pathlib.Path,
    *,
    model_id: str,
) -> list[PackedTensorMeta]:
    """Merge an existing 16 KiB-aligned backbone `.amf` with W20 vision records cleanly."""
    combined_records: list[tuple] = []
    seen_names: set[str] = set()
    with AmfReader(backbone_amf_path, verify_header_crc=True, verify_dir_crc=True) as rdr:
        for m in rdr.entries:
            if is_vision_tensor(m.name):
                continue
            blob = rdr.read_blob(m, verify_crc=True)
            data = blob[: m.data_bytes]
            params = blob[m.params_offset : m.params_offset + m.params_bytes] if m.params_bytes > 0 else b""
            seen_names.add(m.name)
            combined_records.append(
                (
                    m.name,
                    m.kind,
                    m.dtype,
                    m.quant_bits,
                    m.group_size,
                    m.layer,
                    m.expert,
                    m.shape,
                    data,
                    params,
                    m.layout,
                    m.quant_param_kind,
                )
            )
    for rec in vision_records:
        if rec[0] in seen_names:
            raise ValueError(f"duplicate tensor name when merging vision tower: {rec[0]!r}")
        seen_names.add(rec[0])
        combined_records.append(rec)

    meta_recs = [r for r in combined_records if r[1] == TensorKind.METADATA]
    weight_recs = [r for r in combined_records if r[1] != TensorKind.METADATA]
    weight_recs.sort(key=lambda r: _natural_key(r[0]))
    ordered = weight_recs + meta_recs

    out_path = pathlib.Path(output_path)
    partial = out_path.with_name(out_path.name + ".merge.partial")
    try:
        metas = write_amf(partial, model_id, ordered)
        os.replace(partial, out_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return metas


def _pack_uniform_backbone_with_quant_vision(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    resolved: ResolvedCheckpoint,
    tensors: dict[str, _ShardTensor],
    model_id: str,
    group_size: int,
    cold_experts: Iterable[tuple[int, int]],
    all_experts_cold: bool,
    dtype_mode: str,
    tie_word_embeddings: bool | None,
    exclude_tensors: Iterable[str],
    heat_profile: pathlib.Path | None,
    cold_experts_per_layer: int | None,
    sme2_pack: str,
    lut16_sme2: bool,
    use_rco: bool,
    workers: int,
    vision_block_dtype: str,
    sme2_pack_loop_adapters: bool,
    embed_metadata: bool,
    config_path: pathlib.Path | None,
    tokenizer_path: pathlib.Path | None,
    stats_out: dict[str, int] | None,
) -> list[PackedTensorMeta]:
    vision_tensors = {k: v for k, v in tensors.items() if is_vision_tensor(k)}
    backbone_tensors = {k: v for k, v in tensors.items() if not is_vision_tensor(k)}
    if not backbone_tensors:
        return pack_vision_tower_amf(
            input_path,
            output_path,
            model_id=model_id,
            dtype_mode=dtype_mode,
            quant_profile="uniform",
            vision_block_dtype=vision_block_dtype,
            group_size=group_size,
            use_rco=use_rco,
            workers=workers,
            embed_metadata=embed_metadata,
            config_path=config_path,
            stats_out=stats_out,
        )

    out_path = pathlib.Path(output_path)
    tmp_backbone = out_path.with_name(out_path.name + ".bb.partial")
    readers = _ShardReaders()
    try:
        pack_amf_from_safetensors(
            input_path,
            tmp_backbone,
            model_id=model_id,
            group_size=group_size,
            cold_experts=cold_experts,
            all_experts_cold=all_experts_cold,
            dtype_mode=dtype_mode,
            tie_word_embeddings=tie_word_embeddings,
            exclude_tensors=exclude_tensors,
            heat_profile=heat_profile,
            cold_experts_per_layer=cold_experts_per_layer,
            sme2_pack=eff_sme2_requested,
            lut16_sme2=lut16_sme2,
            quant_profile="uniform",
            strip_vision=True,
            sme2_pack_loop_adapters=sme2_pack_loop_adapters,
            embed_metadata=embed_metadata,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
        )
        vis_recs = _build_vision_records_from_safetensors(
            vision_tensors,
            readers,
            dtype_mode=dtype_mode,
            quantize_blocks_q4=(vision_block_dtype == "q4_affine"),
            vision_block_dtype=vision_block_dtype,
            group_size=group_size,
            use_rco=use_rco,
            workers=workers,
        )
        metas = _merge_amf_with_vision_records(tmp_backbone, vis_recs, out_path, model_id=model_id)
        if stats_out is not None:
            stats_out["stripped_vision_tensors"] = 0
            stats_out["included_vision_tensors"] = len(vis_recs)
        return metas
    finally:
        readers.close()
        tmp_backbone.unlink(missing_ok=True)


def pack_vision_tower_amf(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str = "vision-tower",
    dtype_mode: str = "auto",
    quant_profile: str = "uniform",
    vision_block_dtype: str = "auto",
    group_size: int = 64,
    use_rco: bool = True,
    workers: int = 1,
    embed_metadata: bool = False,
    config_path: pathlib.Path | None = None,
    stats_out: dict[str, int] | None = None,
) -> list[PackedTensorMeta]:
    """Pack only the multimodal vision tower into a standalone 16 KiB-aligned `vision.amf` (W20).

    Supports both single/multi-shard safetensors checkpoints and existing `.amf` files.
    Under `quant_profile='gsq-rco'` (or `vision_block_dtype='q4_affine'`), pins
    `patch_embed`, `pos_embed`, all norms, biases, and `merger`/`aligner` in `f16`/`bf16`
    while quantizing vision encoder block 2D projections to `q4_affine`.
    """
    if quant_profile not in QUANT_PROFILES:
        raise ValueError(f"quant_profile must be one of {QUANT_PROFILES}, got {quant_profile!r}")
    if vision_block_dtype not in VISION_BLOCK_DTYPES:
        raise ValueError(
            f"vision_block_dtype must be one of {VISION_BLOCK_DTYPES}, got {vision_block_dtype!r}"
        )
    if group_size not in AFFINE_GROUP_SIZES:
        raise ValueError(f"aminfer supports group_size in {AFFINE_GROUP_SIZES}, got {group_size}")

    in_path = pathlib.Path(input_path)
    out_path = pathlib.Path(output_path)
    quantize_blocks_q4 = (quant_profile == "gsq-rco") or (vision_block_dtype == "q4_affine")
    src_config: dict[str, object] = {}

    if in_path.is_file() and in_path.suffix == ".amf":
        with AmfReader(in_path, verify_header_crc=True, verify_dir_crc=True) as rdr:
            records = _build_vision_records_from_amf(
                rdr,
                dtype_mode=dtype_mode,
                quantize_blocks_q4=quantize_blocks_q4,
                vision_block_dtype=vision_block_dtype,
                group_size=group_size,
                use_rco=use_rco,
                workers=workers,
            )
            if META_CONFIG_JSON in rdr.by_name:
                raw_cfg = rdr.read_metadata_bytes(META_CONFIG_JSON, verify_crc=True)
                loaded = json.loads(raw_cfg.decode("utf-8"))
                if isinstance(loaded, dict):
                    src_config = loaded
    else:
        resolved = resolve_checkpoint(in_path)
        if resolved.config_path is not None and resolved.config_path.is_file():
            loaded = json.loads(resolved.config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                src_config = loaded
        all_tensors = _collect_tensors(resolved.shard_paths)
        vision_tensors = {k: v for k, v in all_tensors.items() if is_vision_tensor(k)}
        if not vision_tensors:
            raise ValueError(f"no vision tensors found in {in_path}")
        readers = _ShardReaders()
        try:
            records = _build_vision_records_from_safetensors(
                vision_tensors,
                readers,
                dtype_mode=dtype_mode,
                quantize_blocks_q4=quantize_blocks_q4,
                vision_block_dtype=vision_block_dtype,
                group_size=group_size,
                use_rco=use_rco,
                workers=workers,
            )
        finally:
            readers.close()

    if not records:
        raise ValueError(f"no vision tensors found in {in_path}")

    if config_path is not None and pathlib.Path(config_path).is_file():
        loaded = json.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            src_config = loaded

    partial = out_path.with_name(out_path.name + ".vis.partial")
    try:
        metas = write_amf(partial, model_id[:63], records)
        if embed_metadata or config_path is not None:
            summary = extract_vision_config_summary(metas, src_config=src_config)
            cfg_payload = dict(src_config)
            cfg_payload["vision_tower_summary"] = summary
            metas = embed_metadata_into_amf(
                partial,
                config_bytes=(json.dumps(cfg_payload, indent=2) + "\n").encode("utf-8"),
                output_path=partial,
            )
        os.replace(partial, out_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    if stats_out is not None:
        stats_out["sidecar_vision_tensors"] = len([m for m in metas if m.kind != TensorKind.METADATA])
    return metas


def gsq_convert_with_vision(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    target_bits: float | None = None,
    model_id: str | None = None,
    group_size: int = 64,
    dtype_mode: str = "auto",
    heat_profile: pathlib.Path | None = None,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int | None = None,
    all_experts_cold: bool = False,
    tie_word_embeddings: bool | None = None,
    exclude_tensors: Iterable[str] = (),
    sme2_pack: str | None = None,
    lut16_sme2: bool = False,
    use_rco: bool = True,
    rco_iters: int = 3,
    lloyd_iters: int = 10,
    sample_rows: int = 32,
    manifest_path: pathlib.Path | None = None,
    workers: int = 1,
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
    imatrix_path: pathlib.Path | None = None,
    enforce_backbone_floor: bool = True,
    max_q2_share: float | None = None,
    allow_lowbit_q2_retranscode: bool = True,
    fine_rungs: bool = False,
    stats_out: dict[str, int] | None = None,
    mimo_config: Mapping[str, Any] | None = None,
    mimo_qkv_tp_degree: int | None = None,
    mimo_mtp_standalone: bool = False,
    mimo_dflash_mode: bool = False,
    qwen35_config: Mapping[str, Any] | None = None,
    qwen4_config: Mapping[str, Any] | None = None,
    qwen4_mtp_standalone: bool = False,
    split_stacked_gate_up: bool | None = None,
    glm5_config: Mapping[str, Any] | None = None,
    glm5_mtp_standalone: bool = False,
    glm5_mtp_source_layer: int | None = None,
    glm5_dflash2_mode: bool = False,
    ling3_config: Mapping[str, Any] | None = None,
    ling3_mtp_standalone: bool = False,
    ling3_mtp_source_layer: int | None = None,
    dense_kda_rco_q3: bool | None = None,
) -> list[PackedTensorMeta]:
    """Run W14 GSQ+RCO with W20 vision-tower precision policy.

    Keeps `patch_embed`, `pos_embed`, norms, 1D biases, and final `merger`/`aligner`
    tensors in `f16`/`bf16`, while quantizing vision encoder block 2D projections to
    `q4_affine` (unless `vision_block_dtype` pins them to `f16`/`bf16`).
    """
    from aminfer_tools.gsq_rco import gsq_convert

    in_path = pathlib.Path(input_path)
    out_path = pathlib.Path(output_path)
    eff_model_id = model_id or out_path.stem[:63]

    is_amf_in = in_path.is_file() and in_path.suffix == ".amf"
    if is_amf_in:
        with AmfReader(in_path, verify_header_crc=True, verify_dir_crc=True) as rdr:
            all_names = [m.name for m in rdr.entries if m.kind != TensorKind.METADATA]
            if model_id is None:
                eff_model_id = rdr.header.model_id or eff_model_id
    else:
        resolved = resolve_checkpoint(in_path)
        all_tensors = _collect_tensors(resolved.shard_paths)
        all_names = list(all_tensors.keys())

    vision_mode = resolve_vision_mode(
        all_names,
        include_vision=include_vision,
        strip_vision=strip_vision,
        vision_sidecar=vision_sidecar,
    )
    vision_names = [n for n in all_names if is_vision_tensor(n)]
    backbone_names = [n for n in all_names if not is_vision_tensor(n)]
    eff_sme2_pack = (
        "controllers"
        if (sme2_pack is None and any(is_inkling2_controller_tensor(n) for n in all_names))
        else (sme2_pack or "none")
    )

    if vision_mode == "sidecar" and vision_sidecar is not None and vision_names:
        pack_vision_tower_amf(
            in_path,
            vision_sidecar,
            model_id=f"{eff_model_id[:55]}-vision",
            dtype_mode=dtype_mode,
            quant_profile="gsq-rco",
            vision_block_dtype=vision_block_dtype,
            group_size=group_size,
            use_rco=use_rco,
            workers=workers,
            stats_out=stats_out,
        )

    if vision_mode == "strip" or not vision_names or vision_mode == "sidecar":
        if not backbone_names:
            raise ValueError(f"no non-vision tensors left to pack in {in_path}")
        rep = gsq_convert(
            in_path,
            out_path,
            target_bits=target_bits,
            model_id=eff_model_id,
            group_size=group_size,
            dtype_mode=dtype_mode,
            heat_profile=heat_profile,
            cold_experts=cold_experts,
            cold_experts_per_layer=cold_experts_per_layer,
            all_experts_cold=all_experts_cold,
            tie_word_embeddings=tie_word_embeddings,
            exclude_tensors=exclude_tensors,
            sme2_pack=eff_sme2_pack,
            lut16_sme2=lut16_sme2,
            use_rco=use_rco,
            rco_iters=rco_iters,
            lloyd_iters=lloyd_iters,
            sample_rows=sample_rows,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=True,
            imatrix_path=imatrix_path,
            enforce_backbone_floor=enforce_backbone_floor,
            max_q2_share=max_q2_share,
            allow_lowbit_q2_retranscode=allow_lowbit_q2_retranscode,
            fine_rungs=fine_rungs,
            dense_kda_rco_q3=dense_kda_rco_q3,
            stats_out=stats_out,
        )
        return rep.metas

    # vision_mode == "include" and vision_names is non-empty:
    if not backbone_names:
        return pack_vision_tower_amf(
            in_path,
            out_path,
            model_id=eff_model_id,
            dtype_mode=dtype_mode,
            quant_profile="gsq-rco",
            vision_block_dtype=vision_block_dtype,
            group_size=group_size,
            use_rco=use_rco,
            workers=workers,
            stats_out=stats_out,
        )

    tmp_bb = out_path.with_name(out_path.name + ".gsq_bb.partial")
    try:
        gsq_convert(
            in_path,
            tmp_bb,
            target_bits=target_bits,
            model_id=eff_model_id,
            group_size=group_size,
            dtype_mode=dtype_mode,
            heat_profile=heat_profile,
            cold_experts=cold_experts,
            cold_experts_per_layer=cold_experts_per_layer,
            all_experts_cold=all_experts_cold,
            tie_word_embeddings=tie_word_embeddings,
            exclude_tensors=exclude_tensors,
            sme2_pack=eff_sme2_pack,
            lut16_sme2=lut16_sme2,
            use_rco=use_rco,
            rco_iters=rco_iters,
            lloyd_iters=lloyd_iters,
            sample_rows=sample_rows,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=True,
            imatrix_path=imatrix_path,
            enforce_backbone_floor=enforce_backbone_floor,
            max_q2_share=max_q2_share,
            allow_lowbit_q2_retranscode=allow_lowbit_q2_retranscode,
            fine_rungs=fine_rungs,
            dense_kda_rco_q3=dense_kda_rco_q3,
        )
        if is_amf_in:
            with AmfReader(in_path, verify_header_crc=True, verify_dir_crc=True) as rdr:
                vis_recs = _build_vision_records_from_amf(
                    rdr,
                    dtype_mode=dtype_mode,
                    quantize_blocks_q4=True,
                    vision_block_dtype=vision_block_dtype,
                    group_size=group_size,
                    use_rco=use_rco,
                    workers=workers,
                )
        else:
            vision_only = {k: v for k, v in all_tensors.items() if is_vision_tensor(k)}
            readers = _ShardReaders()
            try:
                vis_recs = _build_vision_records_from_safetensors(
                    vision_only,
                    readers,
                    dtype_mode=dtype_mode,
                    quantize_blocks_q4=True,
                    vision_block_dtype=vision_block_dtype,
                    group_size=group_size,
                    use_rco=use_rco,
                    workers=workers,
                )
            finally:
                readers.close()

        metas = _merge_amf_with_vision_records(tmp_bb, vis_recs, out_path, model_id=eff_model_id)
        if stats_out is not None:
            stats_out["stripped_vision_tensors"] = 0
            stats_out["included_vision_tensors"] = len(vis_recs)
        if manifest_path is not None and pathlib.Path(manifest_path).is_file():
            mpath = pathlib.Path(manifest_path)
            manifest_obj = json.loads(mpath.read_text(encoding="utf-8"))
            if isinstance(manifest_obj, dict):
                manifest_obj["vision_tower"] = extract_vision_config_summary(
                    [m for m in metas if is_vision_tensor(m.name)]
                )
                mpath.write_text(json.dumps(manifest_obj, indent=2) + "\n", encoding="utf-8")
        return metas
    finally:
        tmp_bb.unlink(missing_ok=True)


def extract_vision_config_summary(
    metas: Sequence[PackedTensorMeta],
    *,
    src_config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize packed W20 vision-tower geometry and precision breakdown for `config.json`."""
    vis_metas = [m for m in metas if is_vision_tensor(m.name)]
    block_ids: set[int] = set()
    dtype_counts: dict[str, int] = {}
    has_qwen_visual = False
    has_deepseek_vision = False
    vis_hidden = 0
    vis_inter = 0
    out_hidden = 0

    for m in vis_metas:
        dt_key = m.dtype.name.lower()
        dtype_counts[dt_key] = dtype_counts.get(dt_key, 0) + 1
        if m.name.startswith("visual."):
            has_qwen_visual = True
        if m.name.startswith(("vision.", "aligner.")):
            has_deepseek_vision = True
        bm = _VISION_BLOCK_RE.search(m.name)
        if bm is not None:
            block_ids.add(int(bm.group(1)))
            if (".attn.proj" in m.name or ".attn.wo" in m.name) and len(m.shape) == 2:
                vis_hidden = max(vis_hidden, m.shape[0])
            if (".mlp.linear_fc1" in m.name or ".mlp.fc1" in m.name or ".mlp.w1" in m.name or ".mlp.gate_proj" in m.name) and len(m.shape) == 2:
                vis_inter = max(vis_inter, m.shape[0])
                if vis_hidden == 0:
                    vis_hidden = m.shape[1]
        if (m.name.startswith(("visual.merger.linear_fc2", "visual.merger.mlp.2", "aligner.w2", "visual.proj_out", "vision.merger.linear_2", "vision.merger.proj.2"))) and len(m.shape) == 2:
            out_hidden = max(out_hidden, m.shape[0])

    vcfg = src_config.get("vision_config") if isinstance(src_config, dict) and isinstance(src_config.get("vision_config"), dict) else {}
    v_mt = str(vcfg.get("model_type", "")).lower()
    if "inkling2" in v_mt or any(m.name.startswith("visual.proj_in") for m in vis_metas):
        family = "inkling2"
    elif "hy_v4" in v_mt or any("perceive" in m.name for m in vis_metas):
        family = "hy_v4"
    else:
        family = "deepseek_v41" if (has_deepseek_vision and not has_qwen_visual) else "qwen_vl"
    return {
        "family": family,
        "vision_tensor_count": len(vis_metas),
        "num_vision_layers": len(block_ids) if block_ids else int(vcfg.get("depth", vcfg.get("num_hidden_layers", 0)) or 0),
        "vision_hidden_size": vis_hidden or int(vcfg.get("hidden_size", 0) or 0),
        "vision_intermediate_size": vis_inter or int(vcfg.get("intermediate_size", 0) or 0),
        "out_hidden_size": out_hidden or int(vcfg.get("out_hidden_size", 0) or 0),
        "patch_size": int(vcfg.get("patch_size", 14 if family == "deepseek_v41" else 16) or 14),
        "temporal_patch_size": int(vcfg.get("temporal_patch_size", 1 if family == "deepseek_v41" else 2) or 1),
        "spatial_merge_size": int(vcfg.get("spatial_merge_size", vcfg.get("downsample_ratio", 2)) or 2),
        "dtype_counts": dtype_counts,
    }


@dataclass(frozen=True)
class SyntheticVitConfig:
    """Deterministic synthetic multi-layer ViT + PatchMerger/Aligner generator config (W20)."""

    family: str = "qwen_vl"  # "qwen_vl" or "deepseek_v41"
    num_vision_layers: int = 4
    vision_hidden_size: int = 128
    vision_intermediate_size: int = 256
    vision_num_heads: int = 4
    out_hidden_size: int = 128
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 16
    mlp_act: str = "gelu"  # "gelu" or "swiglu"
    dtype: str = "bf16"  # "bf16" or "f16"
    use_conv3d_patch_embed: bool = True
    include_language_backbone: bool = False
    backbone_layers: int = 2
    vocab_size: int = 128
    model_id: str = "synth-vit-tower"


def _write_simple_safetensors(
    path: pathlib.Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        dt, shape, payload = tensors[name]
        header[name] = {
            "dtype": dt,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        chunks.append(payload)
        offset += len(payload)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for c in chunks:
            fh.write(c)


def generate_synth_vit_checkpoint(
    output_dir: pathlib.Path,
    config: SyntheticVitConfig | None = None,
    *,
    seed: int = 42,
) -> tuple[pathlib.Path, dict[str, object]]:
    """Generate a deterministic safetensors checkpoint containing a full multi-layer ViT tower (W20).

    Supports both `qwen_vl` (`model.visual.*`: `patch_embed`, `pos_embed`, `blocks.0..N-1`,
    `merger.*`) and `deepseek_v41` (`vision.*` + `aligner.*` + `image_start/end/newline`),
    plus an optional minimal language backbone for end-to-end multimodal packing tests.
    """
    cfg = config or SyntheticVitConfig()
    if cfg.family not in ("qwen_vl", "deepseek_v41"):
        raise ValueError(f"family must be 'qwen_vl' or 'deepseek_v41', got {cfg.family!r}")
    if cfg.dtype not in ("bf16", "f16"):
        raise ValueError(f"dtype must be 'bf16' or 'f16', got {cfg.dtype!r}")
    if cfg.mlp_act not in ("gelu", "swiglu"):
        raise ValueError(f"mlp_act must be 'gelu' or 'swiglu', got {cfg.mlp_act!r}")
    if cfg.num_vision_layers < 1:
        raise ValueError(f"num_vision_layers must be >= 1, got {cfg.num_vision_layers}")

    rng = random.Random(seed)
    sf_dt = "BF16" if cfg.dtype == "bf16" else "F16"
    enc = _f32_values_to_bf16_bytes if cfg.dtype == "bf16" else lambda vals: f32_values_to_f16_bytes(vals, what="vit")

    def _mat(rows: int, cols: int, std: float = 0.05) -> bytes:
        return enc([rng.gauss(0.0, std) for _ in range(rows * cols)])

    def _vec(n: int, val: float = 1.0) -> bytes:
        return enc([val] * n)

    dv = cfg.vision_hidden_size
    df = cfg.vision_intermediate_size
    dl = cfg.out_hidden_size
    p = cfg.patch_size
    tp = cfg.temporal_patch_size if cfg.family == "qwen_vl" else 1
    sm = cfg.spatial_merge_size
    merger_in = dv * sm * sm

    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    if cfg.family == "qwen_vl":
        if cfg.use_conv3d_patch_embed:
            pe_shape: tuple[int, ...] = (dv, 3, tp, p, p)
        else:
            pe_shape = (dv, 3 * tp * p * p)
        pe_numel = dv * 3 * tp * p * p
        tensors["model.visual.patch_embed.proj.weight"] = (sf_dt, pe_shape, _mat(1, pe_numel, 0.03))
        tensors["model.visual.patch_embed.proj.bias"] = (sf_dt, (dv,), _vec(dv, 0.0))
        tensors["model.visual.pos_embed.weight"] = (sf_dt, (cfg.num_position_embeddings, dv), _mat(cfg.num_position_embeddings, dv, 0.02))
        for blk in range(cfg.num_vision_layers):
            bp = f"model.visual.blocks.{blk}"
            tensors[f"{bp}.norm1.weight"] = (sf_dt, (dv,), _vec(dv, 1.0))
            tensors[f"{bp}.norm1.bias"] = (sf_dt, (dv,), _vec(dv, 0.0))
            tensors[f"{bp}.attn.qkv.weight"] = (sf_dt, (3 * dv, dv), _mat(3 * dv, dv))
            tensors[f"{bp}.attn.qkv.bias"] = (sf_dt, (3 * dv,), _vec(3 * dv, 0.0))
            tensors[f"{bp}.attn.proj.weight"] = (sf_dt, (dv, dv), _mat(dv, dv))
            tensors[f"{bp}.attn.proj.bias"] = (sf_dt, (dv,), _vec(dv, 0.0))
            tensors[f"{bp}.norm2.weight"] = (sf_dt, (dv,), _vec(dv, 1.0))
            tensors[f"{bp}.norm2.bias"] = (sf_dt, (dv,), _vec(dv, 0.0))
            if cfg.mlp_act == "swiglu":
                tensors[f"{bp}.mlp.gate_proj.weight"] = (sf_dt, (df, dv), _mat(df, dv))
                tensors[f"{bp}.mlp.up_proj.weight"] = (sf_dt, (df, dv), _mat(df, dv))
                tensors[f"{bp}.mlp.down_proj.weight"] = (sf_dt, (dv, df), _mat(dv, df))
            else:
                tensors[f"{bp}.mlp.linear_fc1.weight"] = (sf_dt, (df, dv), _mat(df, dv))
                tensors[f"{bp}.mlp.linear_fc1.bias"] = (sf_dt, (df,), _vec(df, 0.0))
                tensors[f"{bp}.mlp.linear_fc2.weight"] = (sf_dt, (dv, df), _mat(dv, df))
                tensors[f"{bp}.mlp.linear_fc2.bias"] = (sf_dt, (dv,), _vec(dv, 0.0))
        tensors["model.visual.merger.norm.weight"] = (sf_dt, (dv,), _vec(dv, 1.0))
        tensors["model.visual.merger.norm.bias"] = (sf_dt, (dv,), _vec(dv, 0.0))
        tensors["model.visual.merger.linear_fc1.weight"] = (sf_dt, (merger_in, merger_in), _mat(merger_in, merger_in))
        tensors["model.visual.merger.linear_fc1.bias"] = (sf_dt, (merger_in,), _vec(merger_in, 0.0))
        tensors["model.visual.merger.linear_fc2.weight"] = (sf_dt, (dl, merger_in), _mat(dl, merger_in))
        tensors["model.visual.merger.linear_fc2.bias"] = (sf_dt, (dl,), _vec(dl, 0.0))
        vision_cfg_json: dict[str, object] = {
            "model_type": "qwen4_exp",
            "depth": cfg.num_vision_layers,
            "hidden_size": dv,
            "intermediate_size": df,
            "num_heads": cfg.vision_num_heads,
            "out_hidden_size": dl,
            "patch_size": p,
            "temporal_patch_size": tp,
            "spatial_merge_size": sm,
            "num_position_embeddings": cfg.num_position_embeddings,
            "hidden_act": "gelu_pytorch_tanh" if cfg.mlp_act == "gelu" else "silu",
        }
    else:
        # DeepSeek-V4.1 vision tower (`vision.*` + `aligner.*` + `image_*`).
        tensors["vision.patch_embed.proj.weight"] = (sf_dt, (dv, 3, p, p), _mat(1, dv * 3 * p * p, 0.03))
        tensors["vision.patch_embed.proj.bias"] = (sf_dt, (dv,), _vec(dv, 0.0))
        for blk in range(cfg.num_vision_layers):
            bp = f"vision.blocks.{blk}"
            tensors[f"{bp}.norm1.weight"] = (sf_dt, (dv,), _vec(dv, 1.0))
            tensors[f"{bp}.attn.wqkv.weight"] = (sf_dt, (3 * dv, dv), _mat(3 * dv, dv))
            tensors[f"{bp}.attn.wqkv.bias"] = (sf_dt, (3 * dv,), _vec(3 * dv, 0.0))
            tensors[f"{bp}.attn.wo.weight"] = (sf_dt, (dv, dv), _mat(dv, dv))
            tensors[f"{bp}.attn.wo.bias"] = (sf_dt, (dv,), _vec(dv, 0.0))
            tensors[f"{bp}.norm2.weight"] = (sf_dt, (dv,), _vec(dv, 1.0))
            tensors[f"{bp}.mlp.w1.weight"] = (sf_dt, (df, dv), _mat(df, dv))
            if cfg.mlp_act == "swiglu":
                tensors[f"{bp}.mlp.w3.weight"] = (sf_dt, (df, dv), _mat(df, dv))
            tensors[f"{bp}.mlp.w2.weight"] = (sf_dt, (dv, df), _mat(dv, df))
        tensors["vision.norm.weight"] = (sf_dt, (dv,), _vec(dv, 1.0))
        tensors["aligner.w1.weight"] = (sf_dt, (dl, merger_in), _mat(dl, merger_in))
        tensors["aligner.w1.bias"] = (sf_dt, (dl,), _vec(dl, 0.0))
        tensors["aligner.w2.weight"] = (sf_dt, (dl, dl), _mat(dl, dl))
        tensors["aligner.w2.bias"] = (sf_dt, (dl,), _vec(dl, 0.0))
        tensors["image_start"] = (sf_dt, (dl,), _vec(dl, 0.01))
        tensors["image_end"] = (sf_dt, (dl,), _vec(dl, -0.01))
        tensors["image_newline"] = (sf_dt, (dl,), _vec(dl, 0.02))
        vision_cfg_json = {
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": cfg.num_vision_layers,
            "hidden_size": dv,
            "intermediate_size": df,
            "num_attention_heads": cfg.vision_num_heads,
            "out_hidden_size": dl,
            "patch_size": p,
            "downsample_ratio": sm,
            "rope_theta": 10000,
        }

    cfg_json: dict[str, object] = {
        "model_type": "qwen4_exp" if cfg.family == "qwen_vl" else "deepseek41",
        "model_id": cfg.model_id,
        "hidden_size": dl,
        "vocab_size": cfg.vocab_size,
        "num_hidden_layers": cfg.backbone_layers if cfg.include_language_backbone else 0,
        "tie_word_embeddings": False,
        "vision_config": vision_cfg_json,
    }

    if cfg.include_language_backbone:
        tensors["model.embed_tokens.weight"] = (sf_dt, (cfg.vocab_size, dl), _mat(cfg.vocab_size, dl))
        tensors["model.norm.weight"] = (sf_dt, (dl,), _vec(dl, 1.0))
        tensors["lm_head.weight"] = (sf_dt, (cfg.vocab_size, dl), _mat(cfg.vocab_size, dl))
        for l in range(cfg.backbone_layers):
            lp = f"model.layers.{l}"
            tensors[f"{lp}.input_layernorm.weight"] = (sf_dt, (dl,), _vec(dl, 1.0))
            tensors[f"{lp}.self_attn.q_proj.weight"] = (sf_dt, (dl, dl), _mat(dl, dl))
            tensors[f"{lp}.self_attn.k_proj.weight"] = (sf_dt, (dl, dl), _mat(dl, dl))
            tensors[f"{lp}.self_attn.v_proj.weight"] = (sf_dt, (dl, dl), _mat(dl, dl))
            tensors[f"{lp}.self_attn.o_proj.weight"] = (sf_dt, (dl, dl), _mat(dl, dl))
            tensors[f"{lp}.post_attention_layernorm.weight"] = (sf_dt, (dl,), _vec(dl, 1.0))
            tensors[f"{lp}.mlp.gate_proj.weight"] = (sf_dt, (2 * dl, dl), _mat(2 * dl, dl))
            tensors[f"{lp}.mlp.up_proj.weight"] = (sf_dt, (2 * dl, dl), _mat(2 * dl, dl))
            tensors[f"{lp}.mlp.down_proj.weight"] = (sf_dt, (dl, 2 * dl), _mat(dl, 2 * dl))

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_simple_safetensors(out_dir / "model.safetensors", tensors)
    (out_dir / "config.json").write_text(json.dumps(cfg_json, indent=2) + "\n", encoding="utf-8")
    return out_dir, cfg_json


def generate_synth_vit_amf(
    output_path: pathlib.Path,
    config: SyntheticVitConfig | None = None,
    *,
    seed: int = 42,
    quant_profile: str = "uniform",
    vision_block_dtype: str = "auto",
    dtype_mode: str = "auto",
    embed_metadata: bool = False,
) -> list[PackedTensorMeta]:
    """Synthesize and pack a deterministic standalone ViT `.amf` bundle (W20)."""
    import tempfile

    cfg = config or SyntheticVitConfig()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir, _ = generate_synth_vit_checkpoint(pathlib.Path(tmpdir) / "ckpt", cfg, seed=seed)
        return pack_vision_tower_amf(
            ckpt_dir,
            output_path,
            model_id=cfg.model_id,
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            vision_block_dtype=vision_block_dtype,
            embed_metadata=embed_metadata,
        )
