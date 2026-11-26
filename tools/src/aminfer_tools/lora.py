"""Offline Hugging Face PEFT LoRA/DoRA & synthetic `.amlora` (`.amf`-based) packer (W19, W30, ADR-0022).

Pure standard-library Python implementation (zero third-party runtime/tool dependencies):
- Converts Hugging Face PEFT directories (`adapter_config.json` + `adapter_model.safetensors`)
  or standalone `.safetensors` files into a 16 KiB-aligned `.amlora` bundle readable directly
  by `src/model/lora.zig` (`LoraRegistry.loadAdapterFromFile`).
- Supports `row_major` (`lora_a [r, K]`, `lora_b [N, r]`), `sme2_panel64`
  (`lora_a [r, K]`, `lora_b [N, 64]` with cols `r..64` zero-padded for `W16`/`W19`/`W30` SME2),
  and `auto` (selects compact `row_major` when `K % 64 != 0` or `rank > 16`, else `sme2_panel64`).
- Supports universal graph projections (`Inkling 2`, `Qwen3`, `Hy4`, `MiMo`, `GLM`, `Ling`)
  and full PEFT DoRA (`use_dora: true`, packing magnitude vectors as `.dora_mag` `F32` within
  the 47-byte AMF `TensorEntry.name` ceiling).
- Embeds canonical `__meta.adapter_config_json` (`TensorKind.METADATA`, CRC-32 verified).
- Provides deterministic synthetic `.amlora` generation for integration tests and benchmarks.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import re
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from aminfer_tools.amf import (
    META_ADAPTER_CONFIG_JSON,
    SME2_GROUP_K,
    SME2_PANEL_COLS,
    DType,
    PackedTensorMeta,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    _bf16_to_f32_bytes,
    _read_exact,
    _read_shard_header,
    bf16_bytes_to_f16_bytes,
    f16_bytes_to_bf16_bytes,
    f32_values_to_f16_bytes,
    make_metadata_record,
    pack_sme2_panel64_f16_panel,
    write_amf,
)

MAX_LORA_RANK = 64
MAX_LORA_L2_POOL_BYTES = 4 * 1024 * 1024

LORA_LAYOUTS = ("row_major", "sme2_panel64", "auto")
LORA_DTYPES = ("f16", "bf16")
LORA_DORA_TENSOR_SUFFIX = "dora_mag"

TARGET_MODULE_SUFFIXES: dict[str, str] = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_attn": "self_attn.gate_attn",
    "wr_du": "self_attn.wr_du",
    "qkv_proj": "self_attn.qkv_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
    "gate_up_proj": "mlp.gate_up_proj",
    "q_a_proj": "self_attn.q_a_proj",
    "q_b_proj": "self_attn.q_b_proj",
    "kv_a_proj": "self_attn.kv_a_proj",
    "kv_b_proj": "self_attn.kv_b_proj",
    "eh_proj": "self_attn.eh_proj",
}

MODULE_ALIASES: dict[str, str] = {
    **{v: v for v in TARGET_MODULE_SUFFIXES.values()},
    **{k: v for k, v in TARGET_MODULE_SUFFIXES.items()},
    "q_gate_proj": "self_attn.q_gate_proj",
    "self_attn.q_gate_proj": "self_attn.q_gate_proj",
    "in_proj_qkv": "linear_attn.in_proj_qkv",
    "in_proj_z": "linear_attn.g_proj",
    "in_proj_b": "linear_attn.b_proj",
    "in_proj_a": "linear_attn.a_proj",
    "out_proj": "linear_attn.o_proj",
    "linear_attn.in_proj_qkv": "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z": "linear_attn.g_proj",
    "linear_attn.out_proj": "linear_attn.o_proj",
    "linear_attn.in_proj_a": "linear_attn.a_proj",
    "linear_attn.in_proj_b": "linear_attn.b_proj",
    "linear_attn.q_proj": "linear_attn.q_proj",
    "linear_attn.k_proj": "linear_attn.k_proj",
    "linear_attn.v_proj": "linear_attn.v_proj",
    "linear_attn.g_proj": "linear_attn.g_proj",
    "linear_attn.a_proj": "linear_attn.a_proj",
    "linear_attn.b_proj": "linear_attn.b_proj",
    "linear_attn.o_proj": "linear_attn.o_proj",
    "attn.wq": "self_attn.q_proj",
    "attn.wq_du": "self_attn.q_proj",
    "attn.wk": "self_attn.k_proj",
    "attn.wk_dv": "self_attn.k_proj",
    "attn.wv": "self_attn.v_proj",
    "attn.wv_dv": "self_attn.v_proj",
    "attn.wo": "self_attn.o_proj",
    "attn.wo_ud": "self_attn.o_proj",
    "attn.gate_attn": "self_attn.gate_attn",
    "attn.gate_proj": "self_attn.gate_attn",
    "self_attn.gate_proj": "self_attn.gate_attn",
    "self_attn.o_proj.gate_proj": "self_attn.gate_attn",
    "attn.wr": "self_attn.wr_du",
    "attn.wr_du": "self_attn.wr_du",
    "attn.wqkv": "self_attn.qkv_proj",
    "ffn.w1": "mlp.gate_proj",
    "ffn.w3": "mlp.up_proj",
    "ffn.w2": "mlp.down_proj",
    "ffn.w13": "mlp.gate_up_proj",
    "self_attn.kv_a_proj_with_mqa": "self_attn.kv_a_proj",
    "kv_a_proj_with_mqa": "self_attn.kv_a_proj",
    "self_attn.indexer.wq_b": "attn.idx.wq_b",
    "indexer.wq_b": "attn.idx.wq_b",
    "attn.idx.wq_b": "attn.idx.wq_b",
    "self_attn.indexer.wk": "attn.idx.wk",
    "indexer.wk": "attn.idx.wk",
    "attn.idx.wk": "attn.idx.wk",
    "linear_attn.f_a_proj": "linear_attn.f_a_proj",
    "linear_attn.f_b_proj": "linear_attn.f_b_proj",
    "linear_attn.g_a_proj": "linear_attn.g_a_proj",
    "linear_attn.g_b_proj": "linear_attn.g_b_proj",
    "f_a_proj": "linear_attn.f_a_proj",
    "f_b_proj": "linear_attn.f_b_proj",
    "g_a_proj": "linear_attn.g_a_proj",
    "g_b_proj": "linear_attn.g_b_proj",
    "mtp.eh_proj": "self_attn.eh_proj",
    "mtp_fusion.eh_proj": "self_attn.eh_proj",
}

_PEFT_LAYER_PREFIX_RE = re.compile(
    r"^(?:base_model\.(?:model\.)?)?"
    r"(?:model\.)?(?:(?:language_model|llm)\.)?(?:model\.)?(?:(?:language_model|llm)\.)?"
    r"layers\.(\d+)\.(.+)$"
)
_PEFT_TAIL_RE = re.compile(
    r"^(.+?)\.(lora_[ABab]|lora_magnitude_vector|weight_m_wdecomp)(?:\.[^.]+)?(?:\.weight)?$"
)


@dataclass(frozen=True)
class PackedLoraReport:
    output_path: pathlib.Path
    adapter_id: str
    rank: int
    lora_alpha: float
    scale: float
    layout: str
    dtype: str
    target_modules: tuple[str, ...]
    num_layers: int
    num_pairs: int
    total_wired_bytes: int
    total_file_bytes: int
    fits_l2_4mib: bool
    metas: tuple[PackedTensorMeta, ...]
    use_dora: bool = False


def _f32_bytes_to_target(raw: bytes, target_dtype: DType, what: str) -> bytes:
    if len(raw) % 4 != 0:
        raise ValueError(f"{what}: F32 byte length {len(raw)} not multiple of 4")
    if target_dtype == DType.BF16:
        out = bytearray(len(raw) // 2)
        for i in range(0, len(raw), 4):
            # Round-to-nearest-even from IEEE f32 to bf16
            bits = struct.unpack_from("<I", raw, i)[0]
            rounding = 0x7FFF + ((bits >> 16) & 1)
            bf16 = (bits + rounding) >> 16
            struct.pack_into("<H", out, i // 2, bf16 & 0xFFFF)
        return bytes(out)
    vals = struct.unpack(f"<{len(raw) // 4}f", raw)
    return f32_values_to_f16_bytes(vals, what=what)


def _convert_tensor_to_target(raw: bytes, src_dtype: str, target_dtype: DType, what: str) -> bytes:
    if src_dtype == "F16":
        return raw if target_dtype == DType.F16 else f16_bytes_to_bf16_bytes(raw)
    if src_dtype == "BF16":
        return raw if target_dtype == DType.BF16 else bf16_bytes_to_f16_bytes(raw, what=what)
    if src_dtype == "F32":
        return _f32_bytes_to_target(raw, target_dtype, what)
    raise ValueError(f"{what}: unsupported PEFT LoRA dtype {src_dtype!r} (expected F16, BF16, or F32)")


def _pack_dense_2d_panel64(raw_u16: bytes, n_rows: int, k_cols: int) -> bytes:
    if k_cols <= 0 or k_cols % SME2_GROUP_K != 0:
        raise ValueError(f"sme2_panel64 requires K > 0 and K % 64 == 0, got K={k_cols}")
    row_bytes = k_cols * 2
    if len(raw_u16) != n_rows * row_bytes:
        raise ValueError(f"expected {n_rows * row_bytes} bytes for [{n_rows}, {k_cols}], got {len(raw_u16)}")
    mv = memoryview(raw_u16)
    panels = (n_rows + SME2_PANEL_COLS - 1) // SME2_PANEL_COLS
    chunks: list[bytes] = []
    for p in range(panels):
        r0 = p * SME2_PANEL_COLS
        cnt = min(SME2_PANEL_COLS, n_rows - r0)
        rows = [mv[(r0 + c) * row_bytes : (r0 + c + 1) * row_bytes] for c in range(cnt)]
        chunks.append(pack_sme2_panel64_f16_panel(rows, k_cols))
    return b"".join(chunks)


def _pad_lora_b_to_64_and_pack_panel64(raw_b_u16: bytes, n_rows: int, rank: int) -> bytes:
    if not (1 <= rank <= MAX_LORA_RANK):
        raise ValueError(f"LoRA rank must be in 1..{MAX_LORA_RANK}, got {rank}")
    src_row_bytes = rank * 2
    dst_row_bytes = SME2_GROUP_K * 2
    if len(raw_b_u16) != n_rows * src_row_bytes:
        raise ValueError(f"expected {n_rows * src_row_bytes} bytes for lora_b [{n_rows}, {rank}], got {len(raw_b_u16)}")
    padded = bytearray(n_rows * dst_row_bytes)
    for r in range(n_rows):
        padded[r * dst_row_bytes : r * dst_row_bytes + src_row_bytes] = raw_b_u16[
            r * src_row_bytes : (r + 1) * src_row_bytes
        ]
    return _pack_dense_2d_panel64(bytes(padded), n_rows, SME2_GROUP_K)


def _convert_vector_to_f32(raw: bytes, src_dtype: str, expected_len: int, what: str) -> bytes:
    if src_dtype == "F32":
        if len(raw) != expected_len * 4:
            raise ValueError(f"{what}: expected {expected_len * 4} F32 bytes, got {len(raw)}")
        f32_raw = raw
    elif src_dtype == "F16":
        if len(raw) != expected_len * 2:
            raise ValueError(f"{what}: expected {expected_len * 2} F16 bytes, got {len(raw)}")
        vals = struct.unpack(f"<{expected_len}e", raw)
        f32_raw = struct.pack(f"<{expected_len}f", *vals)
    elif src_dtype == "BF16":
        if len(raw) != expected_len * 2:
            raise ValueError(f"{what}: expected {expected_len * 2} BF16 bytes, got {len(raw)}")
        f32_raw = bytes(_bf16_to_f32_bytes(raw))
    else:
        raise ValueError(f"{what}: unsupported DoRA magnitude dtype {src_dtype!r} (expected F16, BF16, or F32)")
    vals_f32 = struct.unpack(f"<{expected_len}f", f32_raw)
    if not all(math.isfinite(v) for v in vals_f32):
        raise ValueError(f"{what}: non-finite value in DoRA magnitude vector")
    return f32_raw


def _normalize_module_name(raw_mod: str) -> str | None:
    if raw_mod in MODULE_ALIASES:
        return MODULE_ALIASES[raw_mod]
    if raw_mod.endswith(".linear"):
        stripped = raw_mod[: -len(".linear")]
        if stripped in MODULE_ALIASES:
            return MODULE_ALIASES[stripped]
    return None


def _resolve_lora_layout(requested_layout: str, rank: int, k_dims: Iterable[int]) -> str:
    if requested_layout in ("row_major", "sme2_panel64"):
        return requested_layout
    if requested_layout != "auto":
        raise ValueError(f"layout must be one of {LORA_LAYOUTS}, got {requested_layout!r}")
    if rank > 16 or any(k <= 0 or (k % SME2_GROUP_K) != 0 for k in k_dims):
        return "row_major"
    return "sme2_panel64"


def _parse_peft_key(raw_name: str) -> tuple[int, str, str] | None:
    pm = _PEFT_LAYER_PREFIX_RE.match(raw_name)
    if pm is None:
        return None
    layer = int(pm.group(1))
    rest = pm.group(2)
    tm = _PEFT_TAIL_RE.match(rest)
    if tm is None:
        return None
    canonical_mod = _normalize_module_name(tm.group(1))
    if canonical_mod is None:
        return None
    tag = tm.group(2).lower()
    if tag == "lora_a":
        kind = "a"
    elif tag == "lora_b":
        kind = "b"
    else:
        kind = "mag"
    return (layer, canonical_mod, kind)


def pack_lora_from_peft(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    adapter_id: str | None = None,
    adapter_config_path: pathlib.Path | None = None,
    layout: str = "row_major",
    dtype: str = "f16",
    rank_override: int | None = None,
    alpha_override: float | None = None,
) -> PackedLoraReport:
    """Convert a Hugging Face PEFT LoRA/DoRA directory or `.safetensors` into a 16 KiB-aligned `.amlora`."""
    if layout not in LORA_LAYOUTS:
        raise ValueError(f"layout must be one of {LORA_LAYOUTS}, got {layout!r}")
    if dtype not in LORA_DTYPES:
        raise ValueError(f"dtype must be one of {LORA_DTYPES}, got {dtype!r}")

    src = pathlib.Path(input_path)
    if src.is_dir():
        st_path = src / "adapter_model.safetensors"
        if not st_path.is_file():
            raise FileNotFoundError(f"adapter_model.safetensors not found in {src}")
        cfg_path = adapter_config_path if adapter_config_path is not None else (src / "adapter_config.json")
    else:
        st_path = src
        cfg_path = adapter_config_path if adapter_config_path is not None else (src.parent / "adapter_config.json")

    cfg_dict: dict[str, object] = {}
    if cfg_path is not None and cfg_path.is_file():
        cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
        peft_type = str(cfg_dict.get("peft_type", "LORA")).upper()
        if peft_type != "LORA":
            raise ValueError(f"unsupported PEFT type {peft_type!r}; only 'LORA' is supported")

    cfg_use_dora = bool(cfg_dict.get("use_dora", False))
    target_dt = DType.BF16 if dtype == "bf16" else DType.F16

    shard_tensors = _read_shard_header(st_path)
    pairs: dict[tuple[int, str], dict[str, object]] = {}
    for st in shard_tensors:
        parsed = _parse_peft_key(st.name)
        if parsed is None:
            continue
        layer, mod_suffix, key = parsed
        slot = pairs.setdefault((layer, mod_suffix), {})
        if key in slot:
            raise ValueError(f"duplicate LoRA/DoRA tensor for layer {layer} {mod_suffix} ({key})")
        slot[key] = st

    if not pairs:
        raise ValueError(f"no supported LoRA tensors found in {st_path}")

    has_any_mag = any("mag" in slot for slot in pairs.values())
    effective_use_dora = cfg_use_dora or has_any_mag

    inferred_rank: int | None = rank_override
    if inferred_rank is None and "r" in cfg_dict:
        inferred_rank = int(cfg_dict["r"])

    eff_alpha: float | None = alpha_override
    if eff_alpha is None and "lora_alpha" in cfg_dict:
        eff_alpha = float(cfg_dict["lora_alpha"])

    sorted_keys = sorted(pairs.keys())
    k_dims: list[int] = []
    for layer, mod_suffix in sorted_keys:
        slot = pairs[(layer, mod_suffix)]
        if "a" not in slot or "b" not in slot:
            raise ValueError(f"unpaired LoRA tensor for layer {layer} {mod_suffix}")
        ta = slot["a"]
        tb = slot["b"]
        if len(ta.shape) != 2 or len(tb.shape) != 2:
            raise ValueError(f"LoRA tensors must be 2D: {ta.name}={ta.shape}, {tb.name}={tb.shape}")
        r_a, k_dim = ta.shape
        n_dim, r_b = tb.shape
        if r_a != r_b:
            raise ValueError(f"LoRA rank mismatch for layer {layer} {mod_suffix}: A={ta.shape}, B={tb.shape}")
        if not (1 <= r_a <= MAX_LORA_RANK):
            raise ValueError(f"LoRA rank {r_a} out of supported range 1..{MAX_LORA_RANK}")
        if inferred_rank is None:
            inferred_rank = r_a
        elif r_a != inferred_rank:
            raise ValueError(f"heterogeneous LoRA rank {r_a} != {inferred_rank} at layer {layer} {mod_suffix}")
        if effective_use_dora:
            if "mag" not in slot:
                raise ValueError(f"missing DoRA magnitude vector for layer {layer} {mod_suffix}")
            tm = slot["mag"]
            if len(tm.shape) != 1 or tm.shape[0] != n_dim:
                raise ValueError(
                    f"DoRA magnitude shape mismatch for layer {layer} {mod_suffix}: "
                    f"{tm.name}={tm.shape}, expected ({n_dim},)"
                )
        k_dims.append(k_dim)

    assert inferred_rank is not None
    resolved_layout = _resolve_lora_layout(layout, inferred_rank, k_dims)
    target_layout = TensorLayout.SME2_PANEL64 if resolved_layout == "sme2_panel64" else TensorLayout.ROW_MAJOR

    records: list[tuple] = []
    seen_modules: set[str] = set()
    max_layer = -1

    with st_path.open("rb", buffering=0) as fh:
        for layer, mod_suffix in sorted_keys:
            slot = pairs[(layer, mod_suffix)]
            ta = slot["a"]
            tb = slot["b"]
            r_a, k_dim = ta.shape
            n_dim, _ = tb.shape

            raw_a = _convert_tensor_to_target(_read_exact(fh, ta.offset, ta.length), ta.dtype, target_dt, ta.name)
            raw_b = _convert_tensor_to_target(_read_exact(fh, tb.offset, tb.length), tb.dtype, target_dt, tb.name)

            if target_layout == TensorLayout.SME2_PANEL64:
                blob_a = _pack_dense_2d_panel64(raw_a, r_a, k_dim)
                blob_b = _pad_lora_b_to_64_and_pack_panel64(raw_b, n_dim, r_a)
                shape_b = (n_dim, SME2_GROUP_K)
            else:
                blob_a = raw_a
                blob_b = raw_b
                shape_b = (n_dim, r_a)

            a_name = f"layers.{layer}.{mod_suffix}.lora_a"
            b_name = f"layers.{layer}.{mod_suffix}.lora_b"
            records.append(
                (
                    a_name,
                    TensorKind.LOOP_ADAPTER,
                    target_dt,
                    0,
                    0,
                    layer,
                    0xFFFF,
                    (r_a, k_dim),
                    blob_a,
                    b"",
                    target_layout,
                    QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )
            records.append(
                (
                    b_name,
                    TensorKind.LOOP_ADAPTER,
                    target_dt,
                    0,
                    0,
                    layer,
                    0xFFFF,
                    shape_b,
                    blob_b,
                    b"",
                    target_layout,
                    QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )
            if effective_use_dora:
                tm = slot["mag"]
                mag_raw = _read_exact(fh, tm.offset, tm.length)
                mag_f32 = _convert_vector_to_f32(mag_raw, tm.dtype, n_dim, tm.name)
                records.append(
                    (
                        f"layers.{layer}.{mod_suffix}.{LORA_DORA_TENSOR_SUFFIX}",
                        TensorKind.LOOP_ADAPTER,
                        DType.F32,
                        0,
                        0,
                        layer,
                        0xFFFF,
                        (n_dim,),
                        mag_f32,
                        b"",
                        TensorLayout.ROW_MAJOR,
                        QuantParamKind.AFFINE_SCALE_BIAS,
                    )
                )
            mod_short = mod_suffix.rsplit(".", 1)[-1]
            seen_modules.add(mod_short)
            max_layer = max(max_layer, layer)

    if eff_alpha is None:
        eff_alpha = float(inferred_rank)
    scale = eff_alpha / float(inferred_rank)

    eff_id = adapter_id if adapter_id is not None else pathlib.Path(output_path).stem
    sorted_mods = tuple(sorted(seen_modules))
    canonical_cfg: dict[str, object] = {
        "peft_type": "LORA",
        "r": inferred_rank,
        "lora_alpha": eff_alpha,
        "scale": scale,
        "target_modules": list(sorted_mods),
        "layout": resolved_layout,
        "dtype": dtype,
        "use_dora": effective_use_dora,
    }
    if "base_model_name_or_path" in cfg_dict:
        canonical_cfg["base_model_name_or_path"] = cfg_dict["base_model_name_or_path"]

    records.append(
        make_metadata_record(
            META_ADAPTER_CONFIG_JSON,
            json.dumps(canonical_cfg, separators=(",", ":"), sort_keys=True),
        )
    )

    out_path = pathlib.Path(output_path)
    metas = write_amf(out_path, eff_id, records)
    wired_bytes = sum(m.blob_bytes for m in metas if m.kind != TensorKind.METADATA)
    return PackedLoraReport(
        output_path=out_path,
        adapter_id=eff_id,
        rank=inferred_rank,
        lora_alpha=eff_alpha,
        scale=scale,
        layout=resolved_layout,
        dtype=dtype,
        target_modules=sorted_mods,
        num_layers=max_layer + 1,
        num_pairs=len(pairs),
        total_wired_bytes=wired_bytes,
        total_file_bytes=out_path.stat().st_size,
        fits_l2_4mib=wired_bytes <= MAX_LORA_L2_POOL_BYTES,
        metas=tuple(metas),
        use_dora=effective_use_dora,
    )


def generate_synthetic_lora(
    output_path: pathlib.Path,
    *,
    adapter_id: str = "synth-lora",
    num_layers: int = 2,
    hidden_size: int = 64,
    q_dim: int | None = None,
    kv_dim: int | None = None,
    intermediate_size: int = 128,
    rank: int = 4,
    lora_alpha: float = 8.0,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    extra_modules: Sequence[str] = (),
    layout: str = "row_major",
    dtype: str = "f16",
    seed: int = 42,
    zero_init_b: bool = False,
    use_dora: bool = False,
) -> PackedLoraReport:
    """Generate a deterministic 16 KiB-aligned synthetic `.amlora` bundle for testing & benchmarks."""
    if layout not in LORA_LAYOUTS:
        raise ValueError(f"layout must be one of {LORA_LAYOUTS}, got {layout!r}")
    if dtype not in LORA_DTYPES:
        raise ValueError(f"dtype must be one of {LORA_DTYPES}, got {dtype!r}")
    if not (1 <= rank <= MAX_LORA_RANK):
        raise ValueError(f"rank must be in 1..{MAX_LORA_RANK}, got {rank}")
    if num_layers <= 0 or hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("num_layers, hidden_size, and intermediate_size must be positive")
    combined_modules = [*target_modules, *extra_modules]
    if not combined_modules:
        raise ValueError("target_modules must be non-empty")

    eff_q = hidden_size if q_dim is None else q_dim
    eff_kv = hidden_size if kv_dim is None else kv_dim
    dims_map: dict[str, tuple[int, int]] = {
        "q_proj": (eff_q, hidden_size),
        "k_proj": (eff_kv, hidden_size),
        "v_proj": (eff_kv, hidden_size),
        "o_proj": (hidden_size, eff_q),
        "gate_attn": (eff_q, hidden_size),
        "wr_du": (eff_kv, hidden_size),
        "qkv_proj": (eff_q + 2 * eff_kv, hidden_size),
        "gate_proj": (intermediate_size, hidden_size),
        "up_proj": (intermediate_size, hidden_size),
        "down_proj": (hidden_size, intermediate_size),
        "gate_up_proj": (2 * intermediate_size, hidden_size),
        "q_a_proj": (eff_kv, hidden_size),
        "q_b_proj": (eff_q, eff_kv),
        "kv_a_proj": (eff_kv, hidden_size),
        "kv_b_proj": (eff_q, eff_kv),
        "eh_proj": (hidden_size, 2 * hidden_size),
        "q_gate_proj": (eff_q, hidden_size),
        "in_proj_qkv": (eff_q + 2 * eff_kv, hidden_size),
        "in_proj_z": (eff_q, hidden_size),
        "in_proj_b": (eff_kv, hidden_size),
        "in_proj_a": (eff_kv, hidden_size),
        "out_proj": (hidden_size, eff_q),
    }
    norm_targets: list[str] = []
    for raw_tm in combined_modules:
        if raw_tm in TARGET_MODULE_SUFFIXES:
            tm = raw_tm
        else:
            canonical = _normalize_module_name(raw_tm)
            if canonical is None:
                raise ValueError(
                    f"unsupported target module {raw_tm!r}; expected one of {sorted(TARGET_MODULE_SUFFIXES)}"
                )
            tm = canonical.rsplit(".", 1)[-1]
        if tm not in norm_targets:
            norm_targets.append(tm)

    resolved_layout = _resolve_lora_layout(layout, rank, [dims_map[tm][1] for tm in norm_targets])
    target_dt = DType.BF16 if dtype == "bf16" else DType.F16
    target_layout = TensorLayout.SME2_PANEL64 if resolved_layout == "sme2_panel64" else TensorLayout.ROW_MAJOR
    rng = random.Random(seed)

    records: list[tuple] = []
    num_pairs = 0
    for layer in range(num_layers):
        for tm in norm_targets:
            n_dim, k_dim = dims_map[tm]
            mod_suffix = TARGET_MODULE_SUFFIXES[tm]
            a_vals = [rng.uniform(-0.15, 0.15) for _ in range(rank * k_dim)]
            b_vals = [0.0] * (n_dim * rank) if zero_init_b else [rng.uniform(-0.15, 0.15) for _ in range(n_dim * rank)]
            raw_a = f32_values_to_f16_bytes(a_vals, what=f"layers.{layer}.{mod_suffix}.lora_a")
            raw_b = f32_values_to_f16_bytes(b_vals, what=f"layers.{layer}.{mod_suffix}.lora_b")
            if target_dt == DType.BF16:
                raw_a = f16_bytes_to_bf16_bytes(raw_a)
                raw_b = f16_bytes_to_bf16_bytes(raw_b)

            if target_layout == TensorLayout.SME2_PANEL64:
                blob_a = _pack_dense_2d_panel64(raw_a, rank, k_dim)
                blob_b = _pad_lora_b_to_64_and_pack_panel64(raw_b, n_dim, rank)
                shape_b = (n_dim, SME2_GROUP_K)
            else:
                blob_a = raw_a
                blob_b = raw_b
                shape_b = (n_dim, rank)

            records.append(
                (
                    f"layers.{layer}.{mod_suffix}.lora_a",
                    TensorKind.LOOP_ADAPTER,
                    target_dt,
                    0,
                    0,
                    layer,
                    0xFFFF,
                    (rank, k_dim),
                    blob_a,
                    b"",
                    target_layout,
                    QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )
            records.append(
                (
                    f"layers.{layer}.{mod_suffix}.lora_b",
                    TensorKind.LOOP_ADAPTER,
                    target_dt,
                    0,
                    0,
                    layer,
                    0xFFFF,
                    shape_b,
                    blob_b,
                    b"",
                    target_layout,
                    QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )
            if use_dora:
                mag_vals = [rng.uniform(0.5, 1.5) for _ in range(n_dim)]
                blob_mag = struct.pack(f"<{n_dim}f", *mag_vals)
                records.append(
                    (
                        f"layers.{layer}.{mod_suffix}.{LORA_DORA_TENSOR_SUFFIX}",
                        TensorKind.LOOP_ADAPTER,
                        DType.F32,
                        0,
                        0,
                        layer,
                        0xFFFF,
                        (n_dim,),
                        blob_mag,
                        b"",
                        TensorLayout.ROW_MAJOR,
                        QuantParamKind.AFFINE_SCALE_BIAS,
                    )
                )
            num_pairs += 1

    scale = float(lora_alpha) / float(rank)
    sorted_mods = tuple(sorted(norm_targets))
    cfg_json = json.dumps(
        {
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": float(lora_alpha),
            "scale": scale,
            "target_modules": list(sorted_mods),
            "layout": resolved_layout,
            "dtype": dtype,
            "use_dora": use_dora,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    records.append(make_metadata_record(META_ADAPTER_CONFIG_JSON, cfg_json))

    out_path = pathlib.Path(output_path)
    metas = write_amf(out_path, adapter_id, records)
    wired_bytes = sum(m.blob_bytes for m in metas if m.kind != TensorKind.METADATA)
    return PackedLoraReport(
        output_path=out_path,
        adapter_id=adapter_id,
        rank=rank,
        lora_alpha=float(lora_alpha),
        scale=scale,
        layout=resolved_layout,
        dtype=dtype,
        target_modules=sorted_mods,
        num_layers=num_layers,
        num_pairs=num_pairs,
        total_wired_bytes=wired_bytes,
        total_file_bytes=out_path.stat().st_size,
        fits_l2_4mib=wired_bytes <= MAX_LORA_L2_POOL_BYTES,
        metas=tuple(metas),
        use_dora=use_dora,
    )
