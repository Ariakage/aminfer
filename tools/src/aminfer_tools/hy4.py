"""Offline Tencent Hy4 (`hy_v4`, `hy_v4_vl`, `hy_v4_flash`) converter, Engram fp8_e4m3_row splitter, DFly2 packer, and synth CLI (W23, D14, C26).

Supports:
- `tencent/Hy4` (`78L`, `256E` top-8 + 1 shared, iHC `hc_mult=4`, MLA + `learnable_sink_param` + `linear_gate`,
  period-4 `IndexCache` `full`/`shared`, `swiglu_limit=10.0`, `40L` AnyRes ViT, `5L` `DFly2` `block_size=7`),
- `tencent/Hy4-preview` (incl. `ModelOpt` `MXFP8` `UE8M0` scales & `mtp_layers.*`),
- `tencent/Hy4-flash` (`42L`, `512E` `LatentMoE` `routed_expert_up/down_proj` + `combine_output_norm`,
  sigmoid router + `e_score_correction_bias`, `routed_scaling_factor=3.998`, `sandwich_norm`/`post_proj_norm`,
  `1:3` `IndexPool` `compressed_indexer` `compress_ratio=4` + SWA `512`, `fp8_e4m3_row` Engram shards,
  `[128,128]` `BF16`/`F32` `*_scale_inv` FP8, `NVFP4`, `27L` ViT, `5L` `DFly2`),
- `mlx-community/Hy4-preview-4bit` (`4-bit g64` affine + `switch_mlp` + `embed_q`/`unembed_out`),
- `yujiepan/hy4-preview-tiny-random` (real tiny test fixture),
- `AngelSlim/Hy4-flash-GGUF` `IQ2_AH` (standard `IQ2_XXS`/`Q4_K`/`Q8_0`/`BF16` + `F8_E4M3` `per_layer_token_embd`).
"""

from __future__ import annotations

import array
import json
import math
import os
import pathlib
import random
import re
import struct
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DTYPE_MODES,
    NO_EXPERT,
    NO_LAYER,
    QUANT_PROFILES,
    SME2_PACK_MODES,
    VISION_BLOCK_DTYPES,
    AmfReader,
    DType,
    PackedTensorMeta,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    _align_up,
    _bf16_to_f32_bytes,
    _BlobWriter,
    _collect_tensors,
    _crc32,
    _natural_key,
    _pack_directory,
    _pack_header,
    _read_exact,
    _set_nocache,
    _ShardReaders,
    _ShardTensor,
    _write_planned_amf,
    embed_metadata_into_amf,
    f32_to_bf16_bytes,
    f32_to_f16_bytes,
    is_vision_tensor,
    load_cold_experts_from_heat_profile,
    pack_amf_from_safetensors,
    plan_amf_entries,
    quantize_row_f32_to_e4m3_row,
    resolve_checkpoint,
    resolve_vision_mode,
)
from aminfer_tools.gguf import pack_amf_from_gguf

HY4_MODEL_TYPES = ("hy_v4", "hy_v4_vl", "hy_v4_flash")
VALID_HY4_TIERS = ("auto", "24gb-m4", "128gb-max", "256gb-studio", "512gb-ultra")
VALID_HY4_ENGRAM_DTYPES = ("auto", "fp8_e4m3_row", "bf16", "f16")

_HY4_DFLY_PREFIXES = ("model.dfly_layers.", "dfly_layers.", "model.mtp_layers.", "mtp_layers.")
_HY4_ENGRAM_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|engram\.layers)\.(\d+)\.")


def _effective_hy4_text_config(cfg: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(cfg, dict):
        return {}
    out = dict(cfg)
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        out.update(tc)
    return out


def is_hy4_engram_table_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` is a multi-GB Hy4 Engram embedding table (or its row scale)."""
    lower = raw_name.lower()
    if "multi_head_embedding" in lower or "per_layer_token_embd" in lower:
        return True
    if ".engram_layer.lookup_table" in lower:
        return True
    return False


def is_hy4_dfly_tensor(raw_name: str) -> bool:
    return raw_name.startswith(_HY4_DFLY_PREFIXES)


def strip_hy4_dfly_prefix(raw_name: str) -> str:
    for prefix in _HY4_DFLY_PREFIXES:
        if raw_name.startswith(prefix):
            rest = raw_name[len(prefix) :]
            if re.match(r"^\d+\.", rest):
                return f"layers.{rest}"
            return rest
    return raw_name


def _decode_sf_row_f32(raw_row: bytes, dtype_str: str) -> list[float]:
    if dtype_str == "F32":
        arr = array.array("f")
        arr.frombytes(raw_row)
        return list(arr)
    if dtype_str == "F16":
        n = len(raw_row) // 2
        return list(struct.unpack(f"<{n}e", raw_row))
    if dtype_str == "BF16":
        n = len(raw_row) // 2
        u16s = struct.unpack(f"<{n}H", raw_row)
        u32s = struct.pack(f"<{n}I", *(v << 16 for v in u16s))
        arr = array.array("f")
        arr.frombytes(u32s)
        return list(arr)
    raise ValueError(f"unsupported float row dtype {dtype_str!r}")


def engram_shard_paths(base_path: pathlib.Path, num_shards: int) -> list[pathlib.Path]:
    if num_shards < 1:
        raise ValueError(f"engram_shards must be >= 1, got {num_shards}")
    p = pathlib.Path(base_path)
    if num_shards == 1:
        return [p]
    return [p.with_name(f"{p.stem}-{idx:05d}-of-{num_shards:05d}{p.suffix}") for idx in range(num_shards)]


def _split_row_ranges(num_rows: int, num_shards: int) -> list[tuple[int, int]]:
    base = num_rows // num_shards
    rem = num_rows % num_shards
    out: list[tuple[int, int]] = []
    cur = 0
    for s in range(num_shards):
        cnt = base + (1 if s < rem else 0)
        out.append((cur, cur + cnt))
        cur += cnt
    return out


def _tail_cold_experts(num_layers: int, num_experts: int, per_layer: int) -> set[tuple[int, int]]:
    if per_layer <= 0 or num_experts <= 0:
        return set()
    take = min(per_layer, num_experts)
    start = num_experts - take
    return {(l, e) for l in range(num_layers) for e in range(start, num_experts)}


def _deterministic_f32(seed: int, count: int, scale: float = 0.04, *, centered_one: bool = False) -> list[float]:
    rng = random.Random(seed)
    if centered_one:
        return [1.0 + rng.uniform(-scale, scale) for _ in range(count)]
    return [rng.uniform(-scale, scale) for _ in range(count)]


def _encode_floats(vals: Sequence[float], dtype: str) -> bytes:
    if dtype == "f32":
        return struct.pack(f"<{len(vals)}f", *vals)
    if dtype == "f16":
        return f32_to_f16_bytes(vals)
    if dtype == "bf16":
        return f32_to_bf16_bytes(vals)
    raise ValueError(f"unsupported float dtype {dtype!r}")


def _write_safetensors_file(path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        dt, shape, payload = tensors[name]
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [offset, offset + len(payload)]}
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


# ---------------------------------------------------------------------------
# Engram Shard Packer (`fp8_e4m3_row` + aux scales for C26 NVMe streaming)
# ---------------------------------------------------------------------------


def pack_hy4_engram_shards(
    engram_tensors: dict[str, _ShardTensor],
    output_base_path: pathlib.Path,
    *,
    num_shards: int = 1,
    engram_dtype: str = "fp8_e4m3_row",
    model_id: str = "hy4-engram",
) -> tuple[list[pathlib.Path], list[list[PackedTensorMeta]]]:
    """Pack Hy4-flash Engram tables into `num_shards` 16 KiB-aligned `SPARSE_TABLE` `.amf` shards.

    Preserves native `F8_E4M3` + per-row `F32`/`BF16`/`F16` scales losslessly as
    `DType.FP8_E4M3_ROW`, or quantizes `BF16`/`F16`/`F32` rows to `fp8_e4m3_row`
    when `engram_dtype == 'fp8_e4m3_row'`.
    """
    if engram_dtype not in VALID_HY4_ENGRAM_DTYPES:
        raise ValueError(f"engram_dtype must be one of {VALID_HY4_ENGRAM_DTYPES}, got {engram_dtype!r}")
    if not engram_tensors:
        return [], []
    shard_files = engram_shard_paths(output_base_path, num_shards)

    # Identify weight tables vs companion row-scale tensors (`*.weight_scale`, `*.scale`).
    scale_names: set[str] = set()
    weight_to_scale: dict[str, _ShardTensor | None] = {}
    for name, st in sorted(engram_tensors.items()):
        if name.endswith((".weight_scale", ".scale", ".weight_scale_inv", ".scale_inv", "_scale")):
            continue
        if st.dtype in ("F32", "BF16", "F16") and len(st.shape) == 1:
            continue
        base = name[:-7] if name.endswith(".weight") else name
        sc = None
        for cand in (f"{name}_scale", f"{name}.scale", f"{base}.weight_scale", f"{base}.scale", f"{base}.weight_scale_inv"):
            if cand in engram_tensors:
                sc = engram_tensors[cand]
                scale_names.add(cand)
                break
        weight_to_scale[name] = sc

    @dataclass(frozen=True)
    class _ShardTask:
        out_name: str
        layer: int
        src_w: _ShardTensor
        src_scale: _ShardTensor | None
        r0: int
        r1: int
        out_dtype: DType
        quantize_f8: bool

    per_shard: list[list[_ShardTask]] = [[] for _ in range(num_shards)]
    layer_table_counts: dict[int, int] = {}
    for raw_name, sc in sorted(weight_to_scale.items()):
        w = engram_tensors[raw_name]
        if raw_name in scale_names or len(w.shape) != 2:
            continue
        n_rows, dim = w.shape
        m = _HY4_ENGRAM_LAYER_RE.search(raw_name)
        layer = int(m.group(1)) if m else 0
        tbl_idx = layer_table_counts.get(layer, 0)
        layer_table_counts[layer] = tbl_idx + 1

        if w.dtype == "F8_E4M3":
            out_dt = DType.FP8_E4M3_ROW
            quant_f8 = False
        elif engram_dtype in ("fp8_e4m3_row", "auto") and w.dtype in ("BF16", "F16", "F32"):
            out_dt = DType.FP8_E4M3_ROW
            quant_f8 = True
        elif w.dtype == "BF16" and engram_dtype in ("bf16", "auto"):
            out_dt = DType.BF16
            quant_f8 = False
        elif w.dtype == "F16" and engram_dtype in ("f16", "auto"):
            out_dt = DType.F16
            quant_f8 = False
        else:
            out_dt = DType.FP8_E4M3_ROW
            quant_f8 = True

        ranges = _split_row_ranges(n_rows, num_shards)
        for s_idx, (r0, r1) in enumerate(ranges):
            if r1 <= r0:
                continue
            if tbl_idx == 0:
                out_name = f"engram.layers.{layer}.shard.{s_idx}.weight" if num_shards > 1 else f"engram.layers.{layer}.weight"
            else:
                out_name = (
                    f"engram.layers.{layer}.t{tbl_idx}.shard.{s_idx}.weight"
                    if num_shards > 1
                    else f"engram.layers.{layer}.t{tbl_idx}.weight"
                )
            if len(out_name.encode("utf-8")) > 47:
                out_name = f"engram.l{layer}.t{tbl_idx}.s{s_idx}.weight" if tbl_idx > 0 else f"engram.l{layer}.s{s_idx}.weight"
            per_shard[s_idx].append(_ShardTask(out_name, layer, w, sc, r0, r1, out_dt, quant_f8))

    readers = _ShardReaders()
    all_metas: list[list[PackedTensorMeta]] = []
    try:
        for s_idx, tasks in enumerate(per_shard):
            out_file = shard_files[s_idx]
            out_file.parent.mkdir(parents=True, exist_ok=True)
            tasks.sort(key=lambda t: _natural_key(t.out_name))
            tensor_count = len(tasks)
            dir_offset = APPLE_PAGE_SIZE
            dir_bytes = 0 if tensor_count == 0 else _align_up(tensor_count * 128, APPLE_PAGE_SIZE)
            cursor = dir_offset + dir_bytes

            planned_layout: list[tuple[_ShardTask, int, int, int, int, int]] = []
            for t in tasks:
                cnt = t.r1 - t.r0
                dim = t.src_w.shape[1]
                if t.out_dtype == DType.FP8_E4M3_ROW:
                    d_bytes = cnt * dim
                    p_off = _align_up(d_bytes, 64)
                    p_bytes = cnt * 4
                    content = p_off + p_bytes
                else:
                    d_bytes = cnt * dim * 2
                    p_off = 0
                    p_bytes = 0
                    content = d_bytes
                b_bytes = _align_up(max(content, 1), APPLE_PAGE_SIZE)
                planned_layout.append((t, cursor, b_bytes, d_bytes, p_off, p_bytes))
                cursor += b_bytes

            partial = out_file.with_name(out_file.name + ".partial")
            metas: list[PackedTensorMeta] = []
            try:
                with partial.open("wb") as out_fh:
                    _set_nocache(out_fh.fileno())
                    _BlobWriter(out_fh).zeros(dir_offset + dir_bytes)
                    for t, blob_off, blob_b, data_b, p_off, p_b in planned_layout:
                        w = _BlobWriter(out_fh)
                        cnt = t.r1 - t.r0
                        dim = t.src_w.shape[1]
                        w_fh = readers.get(t.src_w.path)
                        if t.out_dtype == DType.FP8_E4M3_ROW and not t.quantize_f8:
                            row_bytes = dim
                            raw_w = _read_exact(w_fh, t.src_w.offset + t.r0 * row_bytes, cnt * row_bytes)
                            w.write(raw_w)
                            w.zeros(p_off - data_b)
                            if t.src_scale is not None:
                                sc_elem = 4 if t.src_scale.dtype == "F32" else 2
                                sc_fh = readers.get(t.src_scale.path)
                                raw_sc = _read_exact(sc_fh, t.src_scale.offset + t.r0 * sc_elem, cnt * sc_elem)
                                if t.src_scale.dtype == "F32":
                                    w.write(raw_sc)
                                elif t.src_scale.dtype == "BF16":
                                    w.write(bytes(_bf16_to_f32_bytes(raw_sc)))
                                else:
                                    w.write(struct.pack(f"<{cnt}f", *struct.unpack(f"<{cnt}e", raw_sc)))
                            else:
                                w.write(struct.pack(f"<{cnt}f", *([1.0] * cnt)))
                        elif t.out_dtype == DType.FP8_E4M3_ROW and t.quantize_f8:
                            elem_b = 4 if t.src_w.dtype == "F32" else 2
                            row_b = dim * elem_b
                            scales_f32: list[float] = []
                            for r in range(t.r0, t.r1):
                                raw_row = _read_exact(w_fh, t.src_w.offset + r * row_b, row_b)
                                vals = _decode_sf_row_f32(raw_row, t.src_w.dtype)
                                q_row, r_sc = quantize_row_f32_to_e4m3_row(vals)
                                w.write(q_row)
                                scales_f32.append(r_sc)
                            w.zeros(p_off - data_b)
                            w.write(struct.pack(f"<{cnt}f", *scales_f32))
                        else:
                            row_b = dim * 2
                            w.write(_read_exact(w_fh, t.src_w.offset + t.r0 * row_b, cnt * row_b))
                        content_done = (p_off + p_b) if p_b > 0 else data_b
                        w.zeros(blob_b - content_done)
                        metas.append(
                            PackedTensorMeta(
                                name=t.out_name,
                                kind=TensorKind.SPARSE_TABLE,
                                dtype=t.out_dtype,
                                quant_bits=8 if t.out_dtype == DType.FP8_E4M3_ROW else 0,
                                group_size=0,
                                layer=t.layer,
                                expert=NO_EXPERT,
                                shape=(cnt, dim),
                                blob_offset=blob_off,
                                blob_bytes=blob_b,
                                data_bytes=data_b,
                                params_offset=p_off,
                                params_bytes=p_b,
                                blob_crc32=w.crc & 0xFFFFFFFF,
                                layout=TensorLayout.ROW_MAJOR,
                                quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                            )
                        )
                    shard_id = f"{model_id[:48]}-engram-s{s_idx}" if num_shards > 1 else f"{model_id[:52]}-engram"
                    dir_raw = _pack_directory(metas)
                    hdr = _pack_header(shard_id.encode("utf-8"), tensor_count, dir_offset, dir_bytes, _crc32(dir_raw))
                    out_fh.seek(0)
                    out_fh.write(hdr)
                    if dir_raw:
                        out_fh.seek(dir_offset)
                        out_fh.write(dir_raw)
                    out_fh.flush()
                    os.fsync(out_fh.fileno())
                os.replace(partial, out_file)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            all_metas.append(metas)
    finally:
        readers.close()
    return shard_files, all_metas


# ---------------------------------------------------------------------------
# DFly2 (`5L`, `block_size=7`) Draft Exporter
# ---------------------------------------------------------------------------


def build_dfly2_config_dict(src_config: dict[str, object] | None) -> dict[str, object]:
    raw = src_config if isinstance(src_config, dict) else {}
    tc = _effective_hy4_text_config(raw)
    return {
        "draft_kind": "dfly2",
        "model_type": "dfly2",
        "num_hidden_layers": int(raw.get("dfly_num_hidden_layers", tc.get("num_nextn_predict_layers", 5)) or 5),
        "block_size": int(raw.get("dfly_block_size", 7) or 7),
        "hidden_size": int(tc.get("hidden_size", 6144) or 6144),
        "num_attention_heads": int(tc.get("num_attention_heads", 64) or 64),
        "num_key_value_heads": int(tc.get("num_key_value_heads", 8) or 8),
        "q_lora_rank": int(tc.get("q_lora_rank", 2048) or 2048),
        "kv_lora_rank": int(tc.get("kv_lora_rank", 512) or 512),
        "qk_nope_head_dim": int(tc.get("qk_nope_head_dim", 192) or 192),
        "qk_rope_head_dim": int(tc.get("qk_rope_head_dim", 64) or 64),
        "v_head_dim": int(tc.get("v_head_dim", 256) or 256),
        "rms_norm_eps": float(tc.get("rms_norm_eps", 1e-5) or 1e-5),
        "dfly_enable_ihc": bool(raw.get("dfly_enable_ihc", False)),
        "dfly_mtp_fusion": str(raw.get("dfly_mtp_fusion", "slot0") or "slot0"),
        "dfly_target_index_topk": int(raw.get("dfly_target_index_topk", 2048) or 2048),
        "dfly_enable_hidden_correction": bool(raw.get("dfly_enable_hidden_correction", True)),
        "dfly_mlp_type": str(raw.get("dfly_mlp_type", "dense") or "dense"),
        "dfly_mask_token_id": int(raw.get("dfly_mask_token_id", 120009) or 120009),
        "dfly_target_index_layer": [int(x) for x in (raw.get("dfly_target_index_layer") or [])],
        "dfly_target_kv_layers": [int(x) for x in (raw.get("dfly_target_kv_layers") or [])],
        "dfly_attn_causal": bool(raw.get("dfly_attn_causal", True)),
        "vocab_size": int(tc.get("vocab_size", 120832) or 120832),
    }


def export_dfly2_from_safetensors(
    dfly_tensors: dict[str, _ShardTensor],
    output_amf: pathlib.Path,
    *,
    config_json: dict[str, object] | None = None,
    config_output: pathlib.Path | None = None,
    model_id: str = "hy4-dfly2",
    dtype_mode: str = "auto",
    group_size: int = 64,
) -> tuple[pathlib.Path, pathlib.Path, list[PackedTensorMeta]]:
    """Export Hy4 `dfly_layers.*` / `mtp_layers.*` tensors into `dfly2.amf` + `dfly2_config.json` (`D14`)."""
    out_amf = pathlib.Path(output_amf)
    out_cfg = pathlib.Path(config_output) if config_output is not None else out_amf.with_name("dfly2_config.json")
    remapped: dict[str, _ShardTensor] = {}
    for raw_name, st in dfly_tensors.items():
        clean = strip_hy4_dfly_prefix(raw_name)
        remapped[clean] = _ShardTensor(
            name=clean,
            dtype=st.dtype,
            shape=st.shape,
            offset=st.offset,
            length=st.length,
            path=st.path,
        )

    dfly_cfg = build_dfly2_config_dict(config_json)
    layer_ids = {int(m.group(1)) for n in remapped if (m := re.match(r"^layers\.(\d+)\.", n)) is not None}
    if layer_ids:
        dfly_cfg["num_hidden_layers"] = max(layer_ids) + 1
    dfly_cfg_bytes = (json.dumps(dfly_cfg, indent=2) + "\n").encode("utf-8")
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_bytes(dfly_cfg_bytes)

    plan = plan_amf_entries(
        remapped,
        group_size=group_size,
        dtype_mode=dtype_mode,
        tie_word_embeddings=False,
        sme2_pack="none",
        strip_vision=True,
    )
    readers = _ShardReaders()
    try:
        metas = _write_planned_amf(
            plan,
            readers,
            out_amf,
            model_id=model_id[:63],
            meta_blobs=[("__meta.config_json", dfly_cfg_bytes)],
        )
    finally:
        readers.close()
    return out_amf, out_cfg, metas


# ---------------------------------------------------------------------------
# Metadata Validation (`tencent/Hy4*`, `mlx-community/Hy4-preview-4bit`, `yujiepan/hy4-preview-tiny-random`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hy4MetadataReport:
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    hc_mult: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    moe_intermediate_size: int
    routed_scaling_factor: float
    swiglu_limit: float
    uses_index_cache: bool
    index_cache_full_layers: int
    index_cache_shared_layers: int
    uses_index_pool: bool
    index_pool_compress_ratio: int
    index_pool_group_topk: int
    index_pool_candidate_topk: int
    sliding_window: int
    swa_layers: int
    dsa_layers: int
    engram_layer_ids: tuple[int, ...]
    engram_fp8_embedding: bool
    dfly_layers: int
    dfly_block_size: int
    dfly_target_kv_layers: tuple[int, ...]
    vision_layers: int
    total_index_entries: int
    shard_file_count: int
    index_total_size_bytes: int
    vision_tensor_count: int
    engram_table_tensor_count: int
    dfly_tensor_count: int
    latent_moe_proj_count: int
    long_backbone_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "arch": self.model_type,
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "hc_mult": self.hc_mult,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "n_shared_experts": self.n_shared_experts,
            "moe_intermediate_size": self.moe_intermediate_size,
            "routed_scaling_factor": self.routed_scaling_factor,
            "swiglu_limit": self.swiglu_limit,
            "index_cache": {
                "enabled": self.uses_index_cache,
                "full_layers": self.index_cache_full_layers,
                "shared_layers": self.index_cache_shared_layers,
            },
            "index_pool": {
                "enabled": self.uses_index_pool,
                "compress_ratio": self.index_pool_compress_ratio,
                "group_topk": self.index_pool_group_topk,
                "candidate_topk": self.index_pool_candidate_topk,
            },
            "sliding_window": self.sliding_window,
            "swa_layers": self.swa_layers,
            "dsa_layers": self.dsa_layers,
            "engram_layer_ids": list(self.engram_layer_ids),
            "engram_fp8_embedding": self.engram_fp8_embedding,
            "dfly2": {
                "num_hidden_layers": self.dfly_layers,
                "block_size": self.dfly_block_size,
                "target_kv_layers": list(self.dfly_target_kv_layers),
            },
            "vision_layers": self.vision_layers,
            "index": {
                "total_entries": self.total_index_entries,
                "shard_files": self.shard_file_count,
                "total_size_bytes": self.index_total_size_bytes,
                "vision": self.vision_tensor_count,
                "engram_tables": self.engram_table_tensor_count,
                "dfly_tensors": self.dfly_tensor_count,
                "latent_moe_projs": self.latent_moe_proj_count,
                "long_backbone_names": list(self.long_backbone_names),
            },
        }


def validate_hf_metadata_dir(metadata_dir: pathlib.Path) -> Hy4MetadataReport:
    """Dry-run validate a Hy4 / Hy4-preview / Hy4-flash / Hy4-VL HF metadata directory."""
    d = pathlib.Path(metadata_dir)
    cfg_path = d / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json in {d}")
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"config.json in {d} must be a JSON object")

    tc = _effective_hy4_text_config(raw_cfg)
    top_mt = str(raw_cfg.get("model_type") or tc.get("model_type") or "hy_v4").lower()
    if top_mt not in HY4_MODEL_TYPES:
        raise ValueError(f"expected Hy4 model_type in {HY4_MODEL_TYPES}, got {top_mt!r}")

    idx_types = tc.get("indexer_types") if isinstance(tc.get("indexer_types"), list) else []
    full_idx = sum(1 for x in idx_types if x == "full")
    shared_idx = sum(1 for x in idx_types if x == "shared")
    uses_pool = bool(tc.get("use_compressed_indexer", False))
    uses_cache = shared_idx > 0

    ltypes = tc.get("layer_types") if isinstance(tc.get("layer_types"), list) else []
    swa_cnt = sum(1 for x in ltypes if x == "sliding_attention")
    dsa_cnt = sum(1 for x in ltypes if "sparse" in str(x) or "indexed" in str(x))

    vis_cfg = raw_cfg.get("vision_config") if isinstance(raw_cfg.get("vision_config"), dict) else {}

    idx_path = d / "model.safetensors.index.json"
    single_sf = d / "model.safetensors"
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_cnt = 0
    if idx_path.is_file():
        idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
        weight_map = idx_obj.get("weight_map") or {}
        total_size = int((idx_obj.get("metadata") or {}).get("total_size", 0) or 0)
        shard_cnt = len(set(weight_map.values()))
    elif single_sf.is_file():
        file_size = single_sf.stat().st_size
        if file_size < 8:
            raise ValueError(f"truncated safetensors header in {single_sf}")
        with single_sf.open("rb") as fh:
            hdr_len_bytes = fh.read(8)
            if len(hdr_len_bytes) != 8:
                raise ValueError(f"truncated safetensors header in {single_sf}")
            (hlen,) = struct.unpack("<Q", hdr_len_bytes)
            if hlen > 256 * 1024 * 1024 or 8 + hlen > file_size:
                raise ValueError(f"implausible safetensors header length {hlen} in {single_sf}")
            hdr_raw = fh.read(hlen)
            if len(hdr_raw) != hlen:
                raise ValueError(f"truncated safetensors JSON in {single_sf}")
            hdr = json.loads(hdr_raw.decode("utf-8"))
        if not isinstance(hdr, dict):
            raise ValueError(f"safetensors header must be a JSON object in {single_sf}")
        data_base = 8 + hlen
        for tname, spec in hdr.items():
            if tname == "__metadata__":
                continue
            if not isinstance(spec, dict):
                raise ValueError(f"tensor {tname} entry in {single_sf} must be a JSON object")
            shape = spec.get("shape")
            offsets = spec.get("data_offsets")
            if not isinstance(shape, list) or any(isinstance(dim, bool) or int(dim) < 0 for dim in shape):
                raise ValueError(f"tensor {tname} in {single_sf} has invalid shape {shape!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(x, bool) for x in offsets)
            ):
                raise ValueError(f"tensor {tname} in {single_sf} has invalid data_offsets {offsets!r}")
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start or data_base + end > file_size:
                raise ValueError(
                    f"tensor {tname} in {single_sf} has out-of-bounds data_offsets {start}..{end}"
                )
        weight_map = {k: "model.safetensors" for k in hdr if k != "__metadata__"}
        total_size = file_size
        shard_cnt = 1

    from aminfer_tools.amf import _normalize_name

    vis_cnt = 0
    engram_tbl_cnt = 0
    dfly_cnt = 0
    latent_cnt = 0
    long_names: list[str] = []
    for raw_k in weight_map:
        if is_vision_tensor(raw_k):
            vis_cnt += 1
            nv = _normalize_name(raw_k)
            if len(nv.encode("utf-8")) > 47:
                long_names.append(nv)
            continue
        if is_hy4_engram_table_tensor(raw_k):
            engram_tbl_cnt += 1
            continue
        if is_hy4_dfly_tensor(raw_k):
            dfly_cnt += 1
            clean = strip_hy4_dfly_prefix(raw_k)
            if len(_normalize_name(clean).encode("utf-8")) > 47:
                long_names.append(clean)
            continue
        if "routed_expert_up_proj" in raw_k or "routed_expert_down_proj" in raw_k:
            latent_cnt += 1
        norm = _normalize_name(raw_k)
        if len(norm.encode("utf-8")) > 47:
            long_names.append(norm)

    return Hy4MetadataReport(
        model_type=top_mt,
        num_hidden_layers=int(tc.get("num_hidden_layers", 78) or 78),
        hidden_size=int(tc.get("hidden_size", 6144) or 6144),
        vocab_size=int(tc.get("vocab_size", 120832) or 120832),
        num_attention_heads=int(tc.get("num_attention_heads", 64) or 64),
        num_key_value_heads=int(tc.get("num_key_value_heads", 8) or 8),
        q_lora_rank=int(tc.get("q_lora_rank", 2048) or 2048),
        kv_lora_rank=int(tc.get("kv_lora_rank", 512) or 512),
        hc_mult=int(tc.get("hc_mult", 4) or 4),
        n_routed_experts=int(tc.get("n_routed_experts", tc.get("num_experts", 256)) or 256),
        num_experts_per_tok=int(tc.get("num_experts_per_tok", 8) or 8),
        n_shared_experts=int(tc.get("n_shared_experts", 1) or 0),
        moe_intermediate_size=int(tc.get("moe_intermediate_size", 2048) or 2048),
        routed_scaling_factor=float(tc.get("routed_scaling_factor", 2.827) or 2.827),
        swiglu_limit=float(tc.get("swiglu_limit", 10.0) or 10.0),
        uses_index_cache=uses_cache,
        index_cache_full_layers=full_idx,
        index_cache_shared_layers=shared_idx,
        uses_index_pool=uses_pool,
        index_pool_compress_ratio=int(tc.get("compressed_indexer_compress_ratio", 4) or 4),
        index_pool_group_topk=int(tc.get("compressed_indexer_group_topk", 512) or 512),
        index_pool_candidate_topk=int(tc.get("compressed_indexer_candidate_topk", 2048) or 2048),
        sliding_window=int(tc.get("sliding_window", 512) or 512),
        swa_layers=swa_cnt,
        dsa_layers=dsa_cnt,
        engram_layer_ids=tuple(int(x) for x in (tc.get("engram_layer_ids") or [])),
        engram_fp8_embedding=bool(tc.get("engram_fp8_embedding", False)),
        dfly_layers=int(raw_cfg.get("dfly_num_hidden_layers", tc.get("num_nextn_predict_layers", 0)) or 0),
        dfly_block_size=int(raw_cfg.get("dfly_block_size", 7) or 7),
        dfly_target_kv_layers=tuple(int(x) for x in (raw_cfg.get("dfly_target_kv_layers") or [])),
        vision_layers=int(vis_cfg.get("num_hidden_layers", 0) or 0),
        total_index_entries=len(weight_map),
        shard_file_count=shard_cnt,
        index_total_size_bytes=total_size,
        vision_tensor_count=vis_cnt,
        engram_table_tensor_count=engram_tbl_cnt,
        dfly_tensor_count=dfly_cnt,
        latent_moe_proj_count=latent_cnt,
        long_backbone_names=tuple(sorted(set(long_names))),
    )


# ---------------------------------------------------------------------------
# Checkpoint Converter (`convert_hy4_checkpoint`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hy4ConvertReport:
    backbone_amf: pathlib.Path
    backbone_metas: list[PackedTensorMeta]
    engram_paths: list[pathlib.Path]
    engram_metas: list[list[PackedTensorMeta]]
    dfly2_amf: pathlib.Path | None
    dfly2_config_path: pathlib.Path | None
    dfly2_metas: list[PackedTensorMeta]
    vision_amf: pathlib.Path | None
    vision_metas: list[PackedTensorMeta]
    config_path: pathlib.Path | None
    cold_expert_count: int


def convert_hy4_checkpoint(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str = "hy4",
    dtype_mode: str = "auto",
    group_size: int | None = None,
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    use_rco: bool = True,
    manifest_path: pathlib.Path | None = None,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int | None = None,
    heat_profile: pathlib.Path | None = None,
    all_experts_cold: bool = False,
    sme2_pack: str = "none",
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
    engram_output: pathlib.Path | None = None,
    engram_shards: int = 1,
    engram_dtype: str = "fp8_e4m3_row",
    dfly2_output: pathlib.Path | None = None,
    dfly2_config_output: pathlib.Path | None = None,
    config_output: pathlib.Path | None = None,
    embed_metadata: bool = True,
    workers: int = 1,
) -> Hy4ConvertReport:
    """Convert Hy4 / Hy4-preview / Hy4-flash / Hy4-VL safetensors or GGUF checkpoints into `.amf` bundles."""
    if quant_profile not in QUANT_PROFILES:
        raise ValueError(f"quant_profile must be one of {QUANT_PROFILES}, got {quant_profile!r}")
    if sme2_pack not in SME2_PACK_MODES:
        raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
    if vision_block_dtype not in VISION_BLOCK_DTYPES:
        raise ValueError(f"vision_block_dtype must be one of {VISION_BLOCK_DTYPES}, got {vision_block_dtype!r}")
    if engram_dtype not in VALID_HY4_ENGRAM_DTYPES:
        raise ValueError(f"engram_dtype must be one of {VALID_HY4_ENGRAM_DTYPES}, got {engram_dtype!r}")

    in_path = pathlib.Path(input_path)
    out_path = pathlib.Path(output_path)

    # GGUF pathway (e.g. `AngelSlim/Hy4-flash-GGUF` `IQ2_AH`).
    if (in_path.is_file() and in_path.suffix.lower() == ".gguf") or (
        in_path.is_dir() and any(in_path.glob("*.gguf")) and not (in_path / "model.safetensors.index.json").is_file()
    ):
        eff_eng = pathlib.Path(engram_output) if engram_output is not None else out_path.with_name("engram.amf")
        metas = pack_amf_from_gguf(
            in_path,
            out_path,
            model_id=model_id,
            dtype_mode=dtype_mode,
            cold_experts=cold_experts,
            all_experts_cold=all_experts_cold,
            embed_metadata=embed_metadata,
            vision_block_dtype=vision_block_dtype,
            sme2_pack=sme2_pack,
            engram_output=eff_eng,
            engram_shards=engram_shards,
        )
        eng_paths = [p for p in engram_shard_paths(eff_eng, engram_shards) if p.is_file()]
        eng_metas: list[list[PackedTensorMeta]] = []
        for ep in eng_paths:
            with AmfReader(ep) as er:
                eng_metas.append(list(er.entries))
        return Hy4ConvertReport(
            backbone_amf=out_path,
            backbone_metas=metas,
            engram_paths=eng_paths,
            engram_metas=eng_metas,
            dfly2_amf=None,
            dfly2_config_path=None,
            dfly2_metas=[],
            vision_amf=None,
            vision_metas=[],
            config_path=None,
            cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
        )

    resolved = resolve_checkpoint(in_path)
    src_cfg: dict[str, object] = {}
    if resolved.config_path is not None and resolved.config_path.is_file():
        loaded = json.loads(resolved.config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            src_cfg = loaded
    tc = _effective_hy4_text_config(src_cfg)
    num_layers = int(tc.get("num_hidden_layers", 10) or 10)
    num_experts = int(tc.get("n_routed_experts", tc.get("num_experts", 32)) or 32)

    eff_cold: set[tuple[int, int]] = set(cold_experts)
    if heat_profile is not None:
        eff_cold.update(
            load_cold_experts_from_heat_profile(
                pathlib.Path(heat_profile),
                cold_experts_per_layer=cold_experts_per_layer,
            )
        )
    elif cold_experts_per_layer is not None:
        eff_cold.update(_tail_cold_experts(num_layers, num_experts, cold_experts_per_layer))

    all_tensors = _collect_tensors(resolved.shard_paths)
    engram_tensors: dict[str, _ShardTensor] = {}
    dfly_tensors: dict[str, _ShardTensor] = {}
    excluded_from_bb: list[str] = []

    for k, v in all_tensors.items():
        if is_hy4_engram_table_tensor(k):
            engram_tensors[k] = v
            excluded_from_bb.append(k)
        elif is_hy4_dfly_tensor(k):
            dfly_tensors[k] = v
            excluded_from_bb.append(k)
        elif any(k.startswith(pref) and k.endswith((".weight_scale", ".scale", ".weight_scale_inv")) for pref in all_tensors if is_hy4_engram_table_tensor(pref)):
            engram_tensors[k] = v
            excluded_from_bb.append(k)

    # Catch companion row scale tensors whose weight is in `engram_tensors`.
    for ek in list(engram_tensors):
        base = ek[:-7] if ek.endswith(".weight") else ek
        for cand in (f"{ek}_scale", f"{ek}.scale", f"{base}.weight_scale", f"{base}.scale", f"{base}.weight_scale_inv"):
            if cand in all_tensors and cand not in engram_tensors:
                engram_tensors[cand] = all_tensors[cand]
                excluded_from_bb.append(cand)

    vision_mode = resolve_vision_mode(
        all_tensors.keys(),
        include_vision=include_vision,
        strip_vision=strip_vision,
        vision_sidecar=vision_sidecar,
    )
    eff_vis_sidecar = vision_sidecar
    if vision_mode == "sidecar" and eff_vis_sidecar is None:
        eff_vis_sidecar = out_path.with_name("vision.amf")

    metas = pack_amf_from_safetensors(
        in_path,
        out_path,
        model_id=model_id[:63],
        group_size=group_size,
        cold_experts=eff_cold,
        all_experts_cold=all_experts_cold,
        dtype_mode=dtype_mode,
        exclude_tensors=excluded_from_bb,
        sme2_pack=sme2_pack,
        quant_profile=quant_profile,
        target_bits=target_bits,
        use_rco=use_rco,
        manifest_path=manifest_path,
        workers=workers,
        strip_vision=strip_vision,
        include_vision=include_vision,
        vision_sidecar=eff_vis_sidecar,
        vision_block_dtype=vision_block_dtype,
        fine_rungs=(quant_profile == "gsq-rco"),
        embed_metadata=embed_metadata,
    )

    vis_metas: list[PackedTensorMeta] = []
    vis_out: pathlib.Path | None = None
    if vision_mode == "sidecar" and eff_vis_sidecar is not None and eff_vis_sidecar.is_file():
        vis_out = eff_vis_sidecar
        with AmfReader(vis_out) as vr:
            vis_metas = list(vr.entries)

    eng_paths: list[pathlib.Path] = []
    eng_metas: list[list[PackedTensorMeta]] = []
    if engram_tensors:
        eff_eng_out = pathlib.Path(engram_output) if engram_output is not None else out_path.with_name("engram.amf")
        eng_paths, eng_metas = pack_hy4_engram_shards(
            engram_tensors,
            eff_eng_out,
            num_shards=engram_shards,
            engram_dtype=engram_dtype,
            model_id=f"{model_id[:55]}-engram",
        )

    dfly_amf_path: pathlib.Path | None = None
    dfly_cfg_path: pathlib.Path | None = None
    dfly_metas: list[PackedTensorMeta] = []
    if dfly_tensors:
        eff_df_amf = pathlib.Path(dfly2_output) if dfly2_output is not None else out_path.with_name("dfly2.amf")
        eff_df_cfg = (
            pathlib.Path(dfly2_config_output)
            if dfly2_config_output is not None
            else eff_df_amf.with_name("dfly2_config.json")
        )
        dfly_amf_path, dfly_cfg_path, dfly_metas = export_dfly2_from_safetensors(
            dfly_tensors,
            eff_df_amf,
            config_json=src_cfg,
            config_output=eff_df_cfg,
            model_id=f"{model_id[:55]}-dfly2",
            dtype_mode=dtype_mode,
            group_size=group_size or 64,
        )

    cfg_out_path: pathlib.Path | None = None
    if config_output is not None:
        cfg_out_path = pathlib.Path(config_output)
        cfg_out_path.parent.mkdir(parents=True, exist_ok=True)
        enriched = dict(src_cfg)
        enriched["operator_axes"] = {
            "attention": "hy4_index_pool_swa" if tc.get("use_compressed_indexer") else "hy4_mla_index_cache",
            "ffn": "latent_moe" if str(src_cfg.get("model_type", "")) == "hy_v4_flash" else "moe_shared",
            "kv_layout": "fp4_latent",
            "sparse_memory": "nvme_engram_fp8_row" if eng_paths else "none",
            "residual": f"ihc_{int(tc.get('hc_mult', 4) or 4)}",
        }
        cfg_out_path.write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")
        if embed_metadata:
            metas = embed_metadata_into_amf(out_path, config_bytes=cfg_out_path.read_bytes(), output_path=out_path)

    return Hy4ConvertReport(
        backbone_amf=out_path,
        backbone_metas=metas,
        engram_paths=eng_paths,
        engram_metas=eng_metas,
        dfly2_amf=dfly_amf_path,
        dfly2_config_path=dfly_cfg_path,
        dfly2_metas=dfly_metas,
        vision_amf=vis_out,
        vision_metas=vis_metas,
        config_path=cfg_out_path,
        cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
    )


# ---------------------------------------------------------------------------
# Deterministic Tiny Hy4 Synthetic Bundle Generator (`synth-hy4`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hy4SynthConfig:
    model_type: str = "hy_v4"  # "hy_v4", "hy_v4_vl", "hy_v4_flash"
    num_hidden_layers: int = 6
    first_k_dense_replace: int = 1
    hidden_size: int = 128
    intermediate_size: int = 256
    moe_intermediate_size: int = 128
    latent_dim: int = 64  # used in LatentMoE (`hy_v4_flash`)
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    q_lora_rank: int = 64
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    hc_mult: int = 4
    n_routed_experts: int = 16
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.827
    swiglu_limit: float = 10.0
    sliding_window: int = 512
    index_n_heads: int = 4
    index_head_dim: int = 64
    compressed_indexer_compress_ratio: int = 4
    compressed_indexer_group_topk: int = 4
    compressed_indexer_candidate_topk: int = 8
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_rows: int = 64
    engram_dim: int = 128
    engram_shards: int = 2
    engram_dtype: str = "fp8_e4m3_row"
    dfly_layers: int = 5
    dfly_block_size: int = 7
    vision_layers: int = 0
    vision_hidden_size: int = 64
    vision_num_heads: int = 4
    vocab_size: int = 256
    dtype: str = "bf16"
    cold_experts_per_layer: int = 2
    vision_mode: str = "sidecar"
    include_dfly2: bool = True
    model_id: str = "tiny-hy4"

    def validate(self) -> None:
        if self.model_type not in HY4_MODEL_TYPES:
            raise ValueError(f"model_type must be in {HY4_MODEL_TYPES}, got {self.model_type!r}")
        if self.hidden_size <= 0 or self.hidden_size % 64 != 0:
            raise ValueError(f"hidden_size must be a positive multiple of 64, got {self.hidden_size}")
        if self.dtype not in ("bf16", "f16"):
            raise ValueError(f"dtype must be 'bf16' or 'f16', got {self.dtype!r}")
        if self.engram_dtype not in VALID_HY4_ENGRAM_DTYPES:
            raise ValueError(f"engram_dtype must be in {VALID_HY4_ENGRAM_DTYPES}, got {self.engram_dtype!r}")


HY4_SYNTH_PRESETS: dict[str, Hy4SynthConfig] = {
    "tiny-hy4": Hy4SynthConfig(
        model_type="hy_v4",
        num_hidden_layers=6,
        routed_scaling_factor=2.827,
        engram_layer_ids=(),
        vision_layers=0,
        dfly_layers=5,
        model_id="tiny-hy4",
    ),
    "tiny-hy4-flash": Hy4SynthConfig(
        model_type="hy_v4_flash",
        num_hidden_layers=6,
        routed_scaling_factor=3.998,
        engram_layer_ids=(1, 3),
        engram_num_rows=64,
        engram_dim=128,
        engram_shards=2,
        engram_dtype="fp8_e4m3_row",
        vision_layers=27,
        dfly_layers=5,
        model_id="tiny-hy4-flash",
    ),
    "tiny-hy4-vl": Hy4SynthConfig(
        model_type="hy_v4_vl",
        num_hidden_layers=6,
        routed_scaling_factor=2.827,
        engram_layer_ids=(),
        vision_layers=40,
        dfly_layers=5,
        model_id="tiny-hy4-vl",
    ),
}


@dataclass(frozen=True)
class Hy4SynthBundle:
    output_dir: pathlib.Path
    model_amf: pathlib.Path
    engram_paths: list[pathlib.Path]
    dfly2_amf: pathlib.Path | None
    dfly2_config_path: pathlib.Path | None
    vision_amf: pathlib.Path | None
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    backbone_metas: list[PackedTensorMeta]
    engram_metas: list[list[PackedTensorMeta]]
    dfly2_metas: list[PackedTensorMeta]
    vision_metas: list[PackedTensorMeta]
    total_bytes: int


def _build_hy4_tokenizer_dict(vocab_size: int) -> dict[str, object]:
    vocab: dict[str, int] = {
        "<｜begin▁of▁sentence｜>": 0,
        "<｜end▁of▁sentence｜>": 1,
        "<｜pad▁of▁sentence｜>": 2,
        "<｜User｜>": 3,
        "<｜Assistant｜>": 4,
    }
    for i in range(len(vocab), vocab_size):
        vocab[f"tok_{i}"] = i
    added = [{"id": tid, "content": tok, "special": True} for tok, tid in vocab.items() if tok.startswith("<｜")]
    return {"version": "1.0", "added_tokens": added, "model": {"type": "BPE", "vocab": vocab, "merges": []}}


def _build_hy4_jinja() -> str:
    return (
        "{%- for message in messages -%}"
        "{%- if message['role'] == 'user' -%}<｜User｜>{{ message['content'] }}"
        "{%- elif message['role'] == 'assistant' -%}<｜Assistant｜>{{ message['content'] }}<｜end▁of▁sentence｜>"
        "{%- endif -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}<｜Assistant｜>{%- endif -%}"
    )


def generate_synth_hy4_bundle(
    output_dir: pathlib.Path,
    *,
    preset: str = "tiny-hy4",
    config: Hy4SynthConfig | None = None,
    seed: int = 42,
) -> Hy4SynthBundle:
    """Generate a deterministic tiny `hy_v4`, `hy_v4_vl`, or `hy_v4_flash` bundle (`synth-hy4`)."""
    if config is not None:
        cfg = config
    else:
        if preset not in HY4_SYNTH_PRESETS:
            raise ValueError(f"unknown Hy4 preset {preset!r}; expected {sorted(HY4_SYNTH_PRESETS)}")
        cfg = HY4_SYNTH_PRESETS[preset]
    cfg.validate()

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf_dt = "BF16" if cfg.dtype == "bf16" else "F16"
    is_flash = cfg.model_type == "hy_v4_flash"
    h = cfg.hidden_size
    qk_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    q_b_out = cfg.num_attention_heads * qk_dim
    kv_b_out = cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
    o_in = cfg.num_attention_heads * cfg.v_head_dim
    exp_in = cfg.latent_dim if is_flash else h
    indexer_types = [
        ("full" if (l in (0, 1) or (l - 1) % 4 == 0) else "shared")
        for l in range(cfg.num_hidden_layers)
    ]
    layer_types = [
        ("deepseek_sparse_attention" if (not is_flash or l in (0, 1) or (l - 1) % 4 == 0) else "sliding_attention")
        for l in range(cfg.num_hidden_layers)
    ]
    mlp_types = [("dense" if l < cfg.first_k_dense_replace else "sparse") for l in range(cfg.num_hidden_layers)]
    dfly_targets = [l for l in range(max(0, cfg.num_hidden_layers - cfg.dfly_layers), cfg.num_hidden_layers)]

    text_cfg: dict[str, object] = {
        "model_type": "hy_v4_flash" if is_flash else "hy_v4",
        "hidden_size": h,
        "num_hidden_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "intermediate_size": cfg.intermediate_size,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "q_lora_rank": cfg.q_lora_rank,
        "kv_lora_rank": cfg.kv_lora_rank,
        "qk_head_dim": qk_dim,
        "qk_nope_head_dim": cfg.qk_nope_head_dim,
        "qk_rope_head_dim": cfg.qk_rope_head_dim,
        "v_head_dim": cfg.v_head_dim,
        "enable_ihc": True,
        "hc_mult": cfg.hc_mult,
        "gated_mla": True,
        "gating_type": "elementwise",
        "learnable_sink": True,
        "swiglu_limit": cfg.swiglu_limit,
        "first_k_dense_replace": cfg.first_k_dense_replace,
        "n_routed_experts": cfg.n_routed_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "n_shared_experts": cfg.n_shared_experts,
        "routed_scaling_factor": cfg.routed_scaling_factor,
        "moe_router_use_sigmoid": is_flash,
        "moe_router_enable_expert_bias": is_flash,
        "sandwich_norm": is_flash,
        "sliding_window": cfg.sliding_window,
        "indexer_types": indexer_types,
        "layer_types": layer_types,
        "mlp_layer_types": mlp_types,
        "use_compressed_indexer": is_flash,
        "compressed_indexer_compress_ratio": cfg.compressed_indexer_compress_ratio,
        "compressed_indexer_group_topk": cfg.compressed_indexer_group_topk,
        "compressed_indexer_candidate_topk": cfg.compressed_indexer_candidate_topk,
        "engram_layer_ids": list(cfg.engram_layer_ids),
        "engram_fp8_embedding": is_flash,
        "rms_norm_eps": 1e-5,
    }
    hf_cfg: dict[str, object] = {
        "architectures": [
            "HYV4FlashForConditionalGeneration"
            if is_flash
            else ("HYV4VLForConditionalGeneration" if cfg.model_type == "hy_v4_vl" else "HYV4ForCausalLM")
        ],
        "model_type": cfg.model_type,
        "text_config": text_cfg,
        "dfly_num_hidden_layers": cfg.dfly_layers,
        "dfly_block_size": cfg.dfly_block_size,
        "dfly_target_kv_layers": dfly_targets,
        "dfly_target_index_layer": dfly_targets,
        "dfly_target_index_topk": cfg.compressed_indexer_candidate_topk if is_flash else 2048,
        "dfly_enable_hidden_correction": True,
        "dfly_mtp_fusion": "slot0",
        "dfly_mask_token_id": 0,
    }
    if cfg.vision_layers > 0:
        hf_cfg["vision_config"] = {
            "model_type": cfg.model_type,
            "hidden_size": cfg.vision_hidden_size,
            "intermediate_size": cfg.vision_hidden_size * 4,
            "num_hidden_layers": cfg.vision_layers,
            "num_attention_heads": cfg.vision_num_heads,
            "out_hidden_size": h,
            "patch_size": 4,
            "spatial_merge_size": 2,
        }

    cfg_path = out_dir / "config.json"
    tok_path = out_dir / "tokenizer.json"
    tpl_path = out_dir / "chat_template.jinja"
    cfg_path.write_text(json.dumps(hf_cfg, indent=2) + "\n", encoding="utf-8")
    tok_path.write_text(json.dumps(_build_hy4_tokenizer_dict(cfg.vocab_size), indent=2) + "\n", encoding="utf-8")
    tpl_path.write_text(_build_hy4_jinja(), encoding="utf-8")

    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def _add(name: str, shape: tuple[int, ...], *, f32: bool = False, one: bool = False, scale: float = 0.04) -> None:
        n = 1
        for d_s in shape:
            n *= d_s
        vals = _deterministic_f32(seed + (zlib.crc32(name.encode("utf-8")) & 0xFFFF), n, scale=scale, centered_one=one)
        tensors[name] = ("F32" if f32 else sf_dt, shape, _encode_floats(vals, "f32" if f32 else cfg.dtype))

    _add("model.embed_tokens.weight", (cfg.vocab_size, h), scale=0.02)
    _add("lm_head.weight", (cfg.vocab_size, h), scale=0.02)
    _add("model.norm.weight", (h,), one=True)
    _add("model.hc_head.hc_head_base", (cfg.hc_mult,), f32=True, one=True)
    _add("model.hc_head.hc_head_fn", (cfg.hc_mult, h), f32=True)
    _add("model.hc_head.hc_head_scale", (1,), f32=True, one=True)

    engram_set = set(cfg.engram_layer_ids)
    for l in range(cfg.num_hidden_layers):
        p = f"model.layers.{l}"
        _add(f"{p}.hc_attn_layer.hc_pre.hc_base", (2 * cfg.hc_mult,), f32=True, one=True)
        _add(f"{p}.hc_attn_layer.hc_pre.hc_fn", (2 * cfg.hc_mult, h), f32=True)
        _add(f"{p}.hc_attn_layer.hc_pre.hc_scale", (2,), f32=True, one=True)
        _add(f"{p}.hc_mlp_layer.hc_pre.hc_base", (2 * cfg.hc_mult,), f32=True, one=True)
        _add(f"{p}.hc_mlp_layer.hc_pre.hc_fn", (2 * cfg.hc_mult, h), f32=True)
        _add(f"{p}.hc_mlp_layer.hc_pre.hc_scale", (2,), f32=True, one=True)
        _add(f"{p}.input_layernorm.weight", (h,), one=True)
        _add(f"{p}.post_attention_layernorm.weight", (h,), one=True)

        # MLA + sink + linear_gate + IndexCache / IndexPool
        _add(f"{p}.self_attn.q_a_proj.weight", (cfg.q_lora_rank, h))
        _add(f"{p}.self_attn.q_a_layernorm.weight", (cfg.q_lora_rank,), one=True)
        _add(f"{p}.self_attn.q_b_proj.weight", (q_b_out, cfg.q_lora_rank))
        _add(f"{p}.self_attn.kv_a_proj_with_mqa.weight", (cfg.kv_lora_rank + cfg.qk_rope_head_dim, h))
        _add(f"{p}.self_attn.kv_a_layernorm.weight", (cfg.kv_lora_rank,), one=True)
        _add(f"{p}.self_attn.kv_b_proj.weight", (kv_b_out, cfg.kv_lora_rank))
        _add(f"{p}.self_attn.o_proj.weight", (h, o_in))
        _add(f"{p}.self_attn.linear_gate.weight", (o_in, h))
        _add(f"{p}.self_attn.learnable_sink_param", (cfg.num_attention_heads,), f32=True, scale=0.01)
        if is_flash:
            _add(f"{p}.self_attn.post_proj_norm.weight", (h,), one=True)
        if layer_types[l] == "deepseek_sparse_attention":
            if is_flash:
                ci = f"{p}.self_attn.compressed_indexer"
                _add(f"{ci}.query_projection.weight", (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank))
                _add(f"{ci}.token_key_projection.weight", (cfg.index_head_dim, h))
                _add(f"{ci}.token_key_norm.weight", (cfg.index_head_dim,), one=True)
                _add(f"{ci}.key_value_projection.weight", (cfg.index_head_dim, h))
                _add(f"{ci}.key_gate_projection.weight", (cfg.index_head_dim, h))
                _add(f"{ci}.key_norm.weight", (cfg.index_head_dim,), one=True)
                _add(f"{ci}.head_weight_projection.weight", (cfg.index_n_heads, h))
            elif indexer_types[l] == "full":
                ix = f"{p}.self_attn.indexer"
                _add(f"{ix}.wq_b.weight", (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank))
                _add(f"{ix}.wk.weight", (cfg.index_head_dim, h))
                _add(f"{ix}.k_norm.weight", (cfg.index_head_dim,), one=True)
                _add(f"{ix}.k_norm.bias", (cfg.index_head_dim,), scale=0.01)
                _add(f"{ix}.weights_proj.weight", (cfg.index_n_heads, h))

        if l in engram_set:
            eg = f"{p}.engram_layer"
            _add(f"{eg}.multi_head_embedding.embedding.weight", (cfg.engram_num_rows, cfg.engram_dim))
            _add(f"{eg}.key_projs.0.weight", (h, cfg.engram_dim))
            _add(f"{eg}.value_proj.weight", (h, cfg.engram_dim))
            _add(f"{eg}.norm_key.0.weight", (h,), one=True)
            _add(f"{eg}.norm_query.0.weight", (h,), one=True)
            _add(f"{eg}.lookup_per_head_norms.0.weight", (cfg.engram_dim,), one=True)
            _add(f"{eg}.unified_per_hc_scales.0", (cfg.hc_mult,), f32=True, one=True)
            _add(f"{eg}.unified_pre_gamma", (h,), f32=True, one=True)

        if mlp_types[l] == "dense":
            _add(f"{p}.mlp.gate_proj.weight", (cfg.intermediate_size, h))
            _add(f"{p}.mlp.up_proj.weight", (cfg.intermediate_size, h))
            _add(f"{p}.mlp.down_proj.weight", (h, cfg.intermediate_size))
            if is_flash:
                _add(f"{p}.mlp.post_fc2_norm.weight", (h,), one=True)
        else:
            _add(f"{p}.mlp.gate.weight", (cfg.n_routed_experts, h))
            _add(f"{p}.mlp.gate.e_score_correction_bias", (cfg.n_routed_experts,), f32=True, scale=0.01)
            if is_flash:
                _add(f"{p}.mlp.routed_expert_down_proj.weight", (cfg.latent_dim, h))
                _add(f"{p}.mlp.routed_expert_up_proj.weight", (h, cfg.latent_dim))
                _add(f"{p}.mlp.combine_output_norm.weight", (h,), one=True)
            _add(f"{p}.mlp.experts.gate_up_proj", (cfg.n_routed_experts, 2 * cfg.moe_intermediate_size, exp_in))
            _add(f"{p}.mlp.experts.down_proj", (cfg.n_routed_experts, exp_in, cfg.moe_intermediate_size))
            if cfg.n_shared_experts > 0:
                _add(f"{p}.mlp.shared_experts.gate_proj.weight", (cfg.moe_intermediate_size, h))
                _add(f"{p}.mlp.shared_experts.up_proj.weight", (cfg.moe_intermediate_size, h))
                _add(f"{p}.mlp.shared_experts.down_proj.weight", (h, cfg.moe_intermediate_size))

    if cfg.include_dfly2 and cfg.dfly_layers > 0:
        dp_top = "model.dfly_layers"
        _add(f"{dp_top}.final_norm.weight", (h,), one=True)
        _add(f"{dp_top}.confidence_head.proj.weight", (1, h))
        _add(f"{dp_top}.confidence_head.proj.bias", (1,), scale=0.01)
        _add(f"{dp_top}.hidden_correction.embed_norm.weight", (h,), one=True)
        _add(f"{dp_top}.hidden_correction.hidden_norm.weight", (h,), one=True)
        _add(f"{dp_top}.hidden_correction.gate_proj.weight", (cfg.intermediate_size, h))
        _add(f"{dp_top}.hidden_correction.up_proj.weight", (cfg.intermediate_size, h))
        _add(f"{dp_top}.hidden_correction.down_proj.weight", (h, cfg.intermediate_size))
        _add(f"{dp_top}.mtp_fusion.eh_proj.weight", (h, 2 * h))
        _add(f"{dp_top}.mtp_fusion.enorm.weight", (h,), one=True)
        _add(f"{dp_top}.mtp_fusion.hnorm.weight", (h,), one=True)
        _add(f"{dp_top}.mtp_fusion.gate", (h,), scale=0.05)
        for dl in range(cfg.dfly_layers):
            dl_p = f"{dp_top}.{dl}"
            _add(f"{dl_p}.input_layernorm.weight", (h,), one=True)
            _add(f"{dl_p}.post_attention_layernorm.weight", (h,), one=True)
            _add(f"{dl_p}.self_attn.q_a_proj.weight", (cfg.q_lora_rank, h))
            _add(f"{dl_p}.self_attn.q_a_layernorm.weight", (cfg.q_lora_rank,), one=True)
            _add(f"{dl_p}.self_attn.q_b_proj.weight", (q_b_out, cfg.q_lora_rank))
            _add(f"{dl_p}.self_attn.kv_a_proj_with_mqa.weight", (cfg.kv_lora_rank + cfg.qk_rope_head_dim, h))
            _add(f"{dl_p}.self_attn.kv_a_layernorm.weight", (cfg.kv_lora_rank,), one=True)
            _add(f"{dl_p}.self_attn.kv_b_proj.weight", (kv_b_out, cfg.kv_lora_rank))
            _add(f"{dl_p}.self_attn.o_proj.weight", (h, o_in))
            _add(f"{dl_p}.self_attn.linear_gate.weight", (o_in, h))
            _add(f"{dl_p}.self_attn.learnable_sink_param", (cfg.num_attention_heads,), f32=True, scale=0.01)
            if is_flash:
                _add(f"{dl_p}.self_attn.post_proj_norm.weight", (h,), one=True)
            _add(f"{dl_p}.mlp.gate_proj.weight", (cfg.intermediate_size, h))
            _add(f"{dl_p}.mlp.up_proj.weight", (cfg.intermediate_size, h))
            _add(f"{dl_p}.mlp.down_proj.weight", (h, cfg.intermediate_size))

    if cfg.vision_layers > 0 and cfg.vision_mode != "strip":
        vh = cfg.vision_hidden_size
        v_inter = vh * 4
        _add("vit.embeddings.patch_embedding.weight", (vh, 3 * 4 * 4))
        _add("vit.embeddings.patch_embedding.bias", (vh,), scale=0.01)
        _add("vit.embeddings.position_embedding.weight", (16, vh), scale=0.02)
        _add("vit.perceive.image_newline", (h,), scale=0.02)
        _add("vit.perceive.norm.weight", (vh,), one=True)
        _add("vit.perceive.proj.0.weight", (h, vh))
        _add("vit.perceive.proj.0.bias", (h,), scale=0.01)
        for vl in range(cfg.vision_layers):
            vp = f"vit.layers.{vl}"
            _add(f"{vp}.input_layernorm.weight", (vh,), one=True)
            _add(f"{vp}.input_layernorm.bias", (vh,), scale=0.01)
            _add(f"{vp}.post_attention_layernorm.weight", (vh,), one=True)
            _add(f"{vp}.post_attention_layernorm.bias", (vh,), scale=0.01)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _add(f"{vp}.self_attn.{proj}.weight", (vh, vh))
                _add(f"{vp}.self_attn.{proj}.bias", (vh,), scale=0.01)
            _add(f"{vp}.mlp.dense_h_to_4h.weight", (v_inter, vh))
            _add(f"{vp}.mlp.dense_h_to_4h.bias", (v_inter,), scale=0.01)
            _add(f"{vp}.mlp.dense_4h_to_h.weight", (vh, v_inter))
            _add(f"{vp}.mlp.dense_4h_to_h.bias", (vh,), scale=0.01)

    src_sf_dir = out_dir / "_src_safetensors"
    src_sf_dir.mkdir(parents=True, exist_ok=True)
    _write_safetensors_file(src_sf_dir / "model.safetensors", tensors)
    (src_sf_dir / "config.json").write_bytes(cfg_path.read_bytes())
    (src_sf_dir / "tokenizer.json").write_bytes(tok_path.read_bytes())
    (src_sf_dir / "chat_template.jinja").write_bytes(tpl_path.read_bytes())

    model_amf = out_dir / "model.amf"
    vis_sidecar = (out_dir / "vision.amf") if (cfg.vision_layers > 0 and cfg.vision_mode == "sidecar") else None
    rep = convert_hy4_checkpoint(
        src_sf_dir,
        model_amf,
        model_id=cfg.model_id[:63],
        cold_experts_per_layer=cfg.cold_experts_per_layer,
        strip_vision=True if cfg.vision_mode == "strip" else None,
        include_vision=True if cfg.vision_mode == "include" else None,
        vision_sidecar=vis_sidecar,
        engram_output=out_dir / "engram.amf",
        engram_shards=cfg.engram_shards,
        engram_dtype=cfg.engram_dtype,
        dfly2_output=out_dir / "dfly2.amf",
        dfly2_config_output=out_dir / "dfly2_config.json",
        config_output=cfg_path,
        embed_metadata=True,
    )

    for p_item in sorted(src_sf_dir.iterdir()):
        p_item.unlink()
    src_sf_dir.rmdir()

    total_bytes = model_amf.stat().st_size + sum(p.stat().st_size for p in rep.engram_paths)
    if rep.dfly2_amf is not None and rep.dfly2_amf.is_file():
        total_bytes += rep.dfly2_amf.stat().st_size
    if rep.vision_amf is not None and rep.vision_amf.is_file():
        total_bytes += rep.vision_amf.stat().st_size

    return Hy4SynthBundle(
        output_dir=out_dir,
        model_amf=model_amf,
        engram_paths=rep.engram_paths,
        dfly2_amf=rep.dfly2_amf,
        dfly2_config_path=rep.dfly2_config_path,
        vision_amf=rep.vision_amf,
        config_path=cfg_path,
        tokenizer_path=tok_path,
        backbone_metas=rep.backbone_metas,
        engram_metas=rep.engram_metas,
        dfly2_metas=rep.dfly2_metas,
        vision_metas=rep.vision_metas,
        total_bytes=total_bytes,
    )
