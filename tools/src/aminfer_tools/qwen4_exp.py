"""Offline `qwen4_exp` (`Qwen3.8-Flash-Next`) and `LoopSchedule` (`W10` / `W16`) checkpoint tools.

Pure Python standard-library implementation (ADR-0001, ADR-0015, ADR-0019).
Provides:

- Canonical `operator_axes` and `loop_schedule` metadata descriptors matching
  `docs/architecture/model-roadmap.md` and ADR-0019.
- Offline checkpoint conversion (`pack_qwen4_checkpoint`) for `qwen4_exp` and
  native / adapter-extended `LoopLM` safetensors checkpoints:
  - Automatic vision-tower stripping (`visual.*`, `vision_tower.*`,
    `multi_modal_projector.*`) with exact `stripped_vision_tensors` accounting.
  - `TensorKind.SHARED_EXPERT = 7`, up to `>= 512` routed experts
    (`hot_expert`/`cold_expert` guided by `--heat-profile` / `--cold-experts-per-layer`),
    3:1 `gated_deltanet` / `qsa_sparse` layer tensors, and 4-branch GR gate tensors.
  - Sharded 16 KiB-aligned `TensorKind.SPARSE_TABLE = 8` PLE `.amf` files
    (`ple.layers.{l}.weight` or `ple.layers.{l}.shard.{s}.weight`,
    `shape = [num_rows, embed_dim]`).
  - Dedicated `4B MTP` draft head `.amf` extraction.
  - Optional Gated Loop Adapters (`TensorKind.LOOP_ADAPTER = 9`,
    `loop.step.{r}.layers.{l}.{proj}.{lora_a,lora_b,alpha}` packed into
    `sme2_panel64`, strictly enforcing `alpha == 0.0` at `r = 0`).
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
import struct
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DTYPE_MODES,
    ENTRY_STRUCT,
    NO_EXPERT,
    NO_LAYER,
    QUANT_PROFILES,
    SME2_GROUP_K,
    SME2_PACK_MODES,
    SME2_PANEL_COLS,
    DType,
    PackedTensorMeta,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    _align_up,
    _BlobWriter,
    _ByteRange,
    _collect_tensors,
    _crc32,
    _is_loop_r0_alpha,
    _load_tie_flag,
    _pack_directory,
    _pack_header,
    _PlannedEntry,
    _read_exact,
    _set_nocache,
    _ShardReaders,
    _ShardTensor,
    _stream_entry,
    _strip_model_prefix,
    _unquantized_target,
    bf16_bytes_to_f16_bytes,
    bf16_bytes_to_f32_list,
    f16_bytes_to_bf16_bytes,
    f16_bytes_to_f32_list,
    f32_bytes_to_f32_list,
    normalize_vision_name,
    VISION_BLOCK_DTYPES,
    extract_vision_config_summary,
    gsq_convert_with_vision,
    is_vision_tensor,
    load_cold_experts_from_heat_profile,
    pack_amf_from_safetensors,
    pack_sme2_panel64_f16_panel,
    pack_vision_tower_amf,
    plan_amf_entries,
    resolve_checkpoint,
    resolve_vision_mode,
    sme2_panel64_data_bytes,
)

_PLE_LAYER_RE = re.compile(r"(?:^|\.)(?:ple\.layers|layers)\.(\d+)\.")
_PLE_SHARD_RE = re.compile(r"\.shard[._](\d+)(?:\.|$)")
_LOOP_ADAPTER_RE = re.compile(
    r"^(?:language_model\.)?(?:model\.)?loop\.step\.(\d+)\.layers\.(\d+)\.([a-zA-Z0-9_]+)\.(lora_a|lora_b|alpha)(?:\.weight)?$"
)
_MTP_PREFIXES = ("mtp.", "model.mtp.", "language_model.mtp.", "language_model.model.mtp.", "nextn.", "model.nextn.")

VALID_ADAPTER_KINDS = ("none", "sme2_step_lora")
VALID_EARLY_EXIT_GATES = ("none", "sigmoid_linear")
VALID_LOOP_MODES = ("single_pass", "native_recurrent", "gated_loop_adapter")


@dataclass(frozen=True)
class OperatorAxes:
    """Canonical composable model-graph axes (ADR-0015, docs/architecture/model-roadmap.md)."""

    backbone_kind: str
    attention_kind: str
    layer_attention_kinds: tuple[str, ...]
    ffn_kind: str
    has_shared_expert: bool
    residual_kind: str
    residual_branches: int
    memory_lookup_kind: str
    loop_schedule_kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "backbone_kind": self.backbone_kind,
            "attention_kind": self.attention_kind,
            "layer_attention_kinds": list(self.layer_attention_kinds),
            "ffn_kind": self.ffn_kind,
            "has_shared_expert": self.has_shared_expert,
            "residual_kind": self.residual_kind,
            "residual_branches": self.residual_branches,
            "memory_lookup_kind": self.memory_lookup_kind,
            "loop_schedule_kind": self.loop_schedule_kind,
        }


@dataclass(frozen=True)
class LoopScheduleConfig:
    """Canonical `LoopSchedule` metadata descriptor (ADR-0019)."""

    mode: str = "single_pass"
    start_layer: int = 0
    end_layer: int = 0
    max_loops: int = 1
    inject_x_init: bool = False
    adapter_kind: str = "none"
    adapter_rank: int = 0
    adapter_packed_rank: int = 0
    adapter_projs: tuple[str, ...] = ()
    enforce_zero_alpha_r0: bool = True
    early_exit_gate: str = "none"

    def validate(self, num_layers: int) -> None:
        if self.mode not in VALID_LOOP_MODES:
            raise ValueError(f"unsupported loop mode {self.mode!r}; expected one of {VALID_LOOP_MODES}")
        if self.adapter_kind not in VALID_ADAPTER_KINDS:
            raise ValueError(
                f"unsupported adapter_kind {self.adapter_kind!r}; expected one of {VALID_ADAPTER_KINDS}"
            )
        if self.early_exit_gate not in VALID_EARLY_EXIT_GATES:
            raise ValueError(
                f"unsupported early_exit_gate {self.early_exit_gate!r}; expected one of {VALID_EARLY_EXIT_GATES}"
            )
        if self.max_loops < 1:
            raise ValueError(f"max_loops must be >= 1, got {self.max_loops}")
        if not (0 <= self.start_layer < self.end_layer <= num_layers):
            raise ValueError(
                f"invalid loop core range [{self.start_layer}, {self.end_layer}) for num_layers={num_layers}"
            )
        if self.adapter_kind == "sme2_step_lora":
            if self.adapter_rank <= 0 or self.adapter_rank > 64:
                raise ValueError(f"sme2_step_lora adapter_rank must be in 1..64, got {self.adapter_rank}")
            if self.adapter_packed_rank <= 0 or self.adapter_packed_rank % SME2_GROUP_K != 0:
                raise ValueError(
                    f"sme2_step_lora adapter_packed_rank must be a positive multiple of 64, got {self.adapter_packed_rank}"
                )
            if self.adapter_packed_rank < self.adapter_rank:
                raise ValueError("adapter_packed_rank must be >= adapter_rank")
            if not self.enforce_zero_alpha_r0:
                raise ValueError("ADR-0019 requires enforce_zero_alpha_r0=True for sme2_step_lora")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "max_loops": self.max_loops,
            "inject_x_init": self.inject_x_init,
            "adapter_kind": self.adapter_kind,
            "adapter_rank": self.adapter_rank,
            "adapter_packed_rank": self.adapter_packed_rank,
            "adapter_projs": list(self.adapter_projs),
            "enforce_zero_alpha_r0": self.enforce_zero_alpha_r0,
            "early_exit_gate": self.early_exit_gate,
        }


QWEN4_HC_COMPACT_MAP: tuple[tuple[str, str], ...] = (
    ("hyper_connection_mixer.", "hc_mixer."),
    (".attn_hyper_connection.", ".attn_hc."),
    (".mlp_hyper_connection.", ".mlp_hc."),
    (".input_mix_weight_down.", ".in_mix_dn."),
    (".input_mix_weight_up.", ".in_mix_up."),
    (".block_inject_weight.", ".blk_inj."),
)

QWEN4_IDX_COMPACT_MAP: tuple[tuple[str, str], ...] = (
    (".self_attn.indexer.index_qk_proj.", ".attn.idx.qk_proj."),
    (".self_attn.indexer.q_layernorm.", ".attn.idx.q_norm."),
    (".self_attn.indexer.k_layernorm.", ".attn.idx.k_norm."),
    (".self_attn.indexer.", ".attn.idx."),
)

QWEN4_PLE_COMPACT_MAP: tuple[tuple[str, str], ...] = (
    (".ple.ple_embedding.layer_multipliers", ".ple.layer_mult"),
    (".ple.ple_embedding.ngram_heads_offsets", ".ple.ngram_offsets"),
    (".ple.ple_embedding.ngram_heads_vocab_sizes", ".ple.ngram_vocabs"),
)

QWEN4_GDN_COMPACT_MAP: tuple[tuple[str, str], ...] = (
    (".linear_attn.in_proj_z.", ".linear_attn.g_proj."),
    (".linear_attn.out_proj.", ".linear_attn.o_proj."),
    (".linear_attn.in_proj_a.", ".linear_attn.a_proj."),
    (".linear_attn.in_proj_b.", ".linear_attn.b_proj."),
)

_DENSE_PLE_MARKERS: tuple[str, ...] = (
    ".ple.conv1d",
    ".ple.key_proj",
    ".ple.value_proj",
    ".ple.norm_conv",
    ".ple.norm_key",
    ".ple.norm_query",
    ".ple.layer_mult",
    ".ple.ngram_offsets",
    ".ple.ngram_vocabs",
    ".ple.ple_embedding.layer_multipliers",
    ".ple.ple_embedding.ngram_heads_offsets",
    ".ple.ple_embedding.ngram_heads_vocab_sizes",
)


def is_qwen4_config(config: Mapping[str, object] | None) -> bool:
    """Return True if `config` describes a `qwen4_exp` (`Qwen3.8-Flash-Next`) checkpoint."""
    if not isinstance(config, Mapping):
        return False
    mt = str(config.get("model_type") or "").lower()
    tc = config.get("text_config")
    tc_mt = str(tc.get("model_type") or "").lower() if isinstance(tc, Mapping) else ""
    if mt in ("qwen4_exp", "qwen4_exp_text") or tc_mt in ("qwen4_exp", "qwen4_exp_text"):
        return True
    archs = config.get("architectures")
    if isinstance(archs, list) and any("qwen4exp" in str(a).lower() for a in archs):
        return True
    return False


def is_dense_ple_tensor(raw_name: str) -> bool:
    """Return True for wired dense PLE conditioning weights/buffers (kept in `model.amf`)."""
    norm = _strip_model_prefix(raw_name).lower()
    return ".ple." in norm and any(tok in norm for tok in _DENSE_PLE_MARKERS)


def is_ple_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` is an external PLE / Engram sparse lookup table."""
    if is_dense_ple_tensor(raw_name):
        return False
    norm = _strip_model_prefix(raw_name).lower()
    return (
        norm.startswith("ple.")
        or ".ple." in norm
        or norm.startswith("engram.")
        or ".engram." in norm
        or norm.startswith("sparse_table.")
    )


def normalize_qwen4_tensor_name(raw_name: str, *, mtp_standalone: bool = False) -> str:
    """Normalize a `qwen4_exp` (`Qwen3.8-Flash-Next`) name deterministically to `<= 47 B` `.amf` form."""
    if is_vision_tensor(raw_name):
        return normalize_vision_name(raw_name)
    if is_ple_tensor(raw_name):
        return normalize_ple_name(raw_name)
    name = strip_mtp_prefix(raw_name) if mtp_standalone else _strip_model_prefix(raw_name)
    for suf in (".scales", ".biases", ".weight_packed", ".weight_scale", ".weight_scale_inv"):
        if name.endswith(suf):
            name = name[: -len(suf)] + ".weight"
            break
    for mapping in (QWEN4_GDN_COMPACT_MAP, QWEN4_HC_COMPACT_MAP, QWEN4_IDX_COMPACT_MAP, QWEN4_PLE_COMPACT_MAP):
        for src_seg, dst_seg in mapping:
            name = name.replace(src_seg, dst_seg)
    return name


def is_mtp_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` belongs to an embedded Multi-Token Prediction draft head."""
    return raw_name.startswith(_MTP_PREFIXES)


def is_loop_adapter_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` belongs to a Gated Loop Adapter (`Step-LoRA`)."""
    norm = _strip_model_prefix(raw_name).lower()
    return norm.startswith("loop.step.") or ".loop_adapter." in norm


def strip_mtp_prefix(raw_name: str) -> str:
    for prefix in _MTP_PREFIXES:
        if raw_name.startswith(prefix):
            return raw_name[len(prefix) :]
    return _strip_model_prefix(raw_name)


def normalize_ple_name(raw_name: str, *, shard_idx: int | None = None, quantized: bool = False) -> str:
    """Normalize a PLE tensor name to `ple.layers.{l}.weight` or `ple.layers.{l}.shard.{s}.weight`."""
    norm = _strip_model_prefix(raw_name)
    m = _PLE_LAYER_RE.search(norm)
    layer = int(m.group(1)) if m else 0
    suffix = "" if quantized else ".weight"
    if shard_idx is not None:
        return f"ple.layers.{layer}.shard.{shard_idx}{suffix}"
    sm = _PLE_SHARD_RE.search(norm)
    if sm is not None:
        return f"ple.layers.{layer}.shard.{int(sm.group(1))}{suffix}"
    return f"ple.layers.{layer}{suffix}"


def ple_shard_paths(base_path: pathlib.Path, num_shards: int) -> list[pathlib.Path]:
    """Return deterministic output path(s) for `num_shards` PLE `.amf` files."""
    if num_shards < 1:
        raise ValueError(f"ple num_shards must be >= 1, got {num_shards}")
    base = pathlib.Path(base_path)
    if num_shards == 1:
        return [base]
    stem = base.stem
    suffix = base.suffix or ".amf"
    return [base.with_name(f"{stem}-shard-{s:02d}-of-{num_shards:02d}{suffix}") for s in range(num_shards)]


def _split_row_ranges(num_rows: int, num_shards: int) -> list[tuple[int, int]]:
    if num_rows < num_shards:
        raise ValueError(f"cannot split {num_rows} PLE rows across {num_shards} shards")
    base = num_rows // num_shards
    rem = num_rows % num_shards
    ranges: list[tuple[int, int]] = []
    start = 0
    for s in range(num_shards):
        cnt = base + (1 if s < rem else 0)
        ranges.append((start, start + cnt))
        start += cnt
    return ranges


def _slice_row_byte_range(t: _ShardTensor, r0: int, r1: int) -> _ByteRange:
    total_rows = t.shape[0]
    if total_rows <= 0 or t.length % total_rows != 0:
        raise ValueError(f"cannot row-slice tensor {t.name!r} with shape {t.shape} and length {t.length}")
    row_bytes = t.length // total_rows
    return _ByteRange(t.path, t.offset + r0 * row_bytes, (r1 - r0) * row_bytes, t.dtype)


def pack_ple_shards_from_safetensors(
    ple_tensors: dict[str, _ShardTensor],
    output_base_path: pathlib.Path,
    *,
    num_shards: int = 1,
    model_id: str = "qwen4-ple",
    group_size: int = 64,
    dtype_mode: str = "auto",
) -> tuple[list[pathlib.Path], list[list[PackedTensorMeta]]]:
    """Stream PLE tables from safetensors into 1 or `num_shards` 16 KiB-aligned `.amf` files.

    Every entry is tagged with `TensorKind.SPARSE_TABLE = 8` and `shape = [rows_in_shard, embed_dim]`,
    named `ple.layers.{l}.weight` (1 shard) or `ple.layers.{l}.shard.{s}.weight` (`num_shards > 1`).
    """
    if group_size != 64:
        raise ValueError(f"aminfer v0 supports group_size=64 only, got {group_size}")
    if dtype_mode not in DTYPE_MODES:
        raise ValueError(f"dtype_mode must be one of {DTYPE_MODES}, got {dtype_mode!r}")
    shard_files = ple_shard_paths(output_base_path, num_shards)
    if not ple_tensors:
        return [], []

    quant_prefixes: set[str] = set()
    for name in ple_tensors:
        if name.endswith(".scales"):
            prefix = name[: -len(".scales")]
            if f"{prefix}.weight" in ple_tensors and f"{prefix}.biases" in ple_tensors:
                quant_prefixes.add(prefix)

    # Plan per-shard entries.
    per_shard_entries: list[list[_PlannedEntry]] = [[] for _ in range(num_shards)]
    consumed: set[str] = set()

    for prefix in sorted(quant_prefixes):
        w = ple_tensors[f"{prefix}.weight"]
        s = ple_tensors[f"{prefix}.scales"]
        b = ple_tensors[f"{prefix}.biases"]
        consumed.update((w.name, s.name, b.name))
        if len(w.shape) != 2 or len(s.shape) != 2 or w.shape[0] != s.shape[0]:
            raise ValueError(f"PLE quantized tensor {prefix} must be 2D, got {w.shape} / {s.shape}")
        n_rows, packed_cols = w.shape
        groups = s.shape[1]
        k = groups * group_size
        bits = (packed_cols * 32) // k
        amf_dtype = {2: DType.Q2_AFFINE, 3: DType.Q3_AFFINE, 4: DType.Q4_AFFINE}.get(bits)
        if amf_dtype is None:
            raise ValueError(f"unsupported PLE quant bits {bits} for {prefix}")
        m = _PLE_LAYER_RE.search(_strip_model_prefix(prefix))
        layer = int(m.group(1)) if m else 0
        ranges = _split_row_ranges(n_rows, num_shards)
        for s_idx, (r0, r1) in enumerate(ranges):
            cnt = r1 - r0
            entry_name = normalize_ple_name(prefix, shard_idx=(s_idx if num_shards > 1 else None), quantized=True)
            d_rng = _slice_row_byte_range(w, r0, r1)
            s_rng = _slice_row_byte_range(s, r0, r1)
            b_rng = _slice_row_byte_range(b, r0, r1)
            data_bytes = d_rng.length
            params_offset = _align_up(data_bytes, 64)
            params_bytes = cnt * groups * 4
            blob_bytes = _align_up(params_offset + params_bytes, APPLE_PAGE_SIZE)
            per_shard_entries[s_idx].append(
                _PlannedEntry(
                    name=entry_name,
                    kind=TensorKind.SPARSE_TABLE,
                    dtype=amf_dtype,
                    quant_bits=bits,
                    group_size=group_size,
                    layer=layer,
                    expert=NO_EXPERT,
                    shape=(cnt, k),
                    data=d_rng,
                    data_convert=None,
                    scales=s_rng,
                    biases=b_rng,
                    data_bytes=data_bytes,
                    params_offset=params_offset,
                    params_bytes=params_bytes,
                    blob_bytes=blob_bytes,
                    layout=TensorLayout.ROW_MAJOR,
                    quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )

    unconsumed = [name for name in sorted(ple_tensors) if name not in consumed]
    presharded_pairs: list[tuple[int, str]] = []
    for name in unconsumed:
        sm = _PLE_SHARD_RE.search(_strip_model_prefix(name))
        if sm is not None:
            presharded_pairs.append((int(sm.group(1)), name))
    use_presharded = len(unconsumed) > 1 and len(presharded_pairs) == len(unconsumed) and num_shards <= len(presharded_pairs)

    for idx_in_list, name in enumerate(unconsumed):
        t = ple_tensors[name]
        if len(t.shape) != 2:
            raise ValueError(f"PLE tensor {name!r} must be 2D [num_rows, embed_dim], got {t.shape}")
        amf_dtype, convert = _unquantized_target(t.dtype, dtype_mode, name)
        n_rows, embed_dim = t.shape
        m = _PLE_LAYER_RE.search(_strip_model_prefix(name))
        layer = int(m.group(1)) if m else 0
        if use_presharded:
            orig_s, _ = presharded_pairs[idx_in_list]
            s_idx = orig_s % num_shards if num_shards > 1 else 0
            entry_name = normalize_ple_name(name, shard_idx=orig_s, quantized=False)
            d_rng = _slice_row_byte_range(t, 0, n_rows)
            data_bytes = d_rng.length
            blob_bytes = _align_up(max(data_bytes, 1), APPLE_PAGE_SIZE)
            per_shard_entries[s_idx].append(
                _PlannedEntry(
                    name=entry_name,
                    kind=TensorKind.SPARSE_TABLE,
                    dtype=amf_dtype,
                    quant_bits=0,
                    group_size=0,
                    layer=layer,
                    expert=NO_EXPERT,
                    shape=(n_rows, embed_dim),
                    data=d_rng,
                    data_convert=convert,
                    scales=None,
                    biases=None,
                    data_bytes=data_bytes,
                    params_offset=0,
                    params_bytes=0,
                    blob_bytes=blob_bytes,
                    layout=TensorLayout.ROW_MAJOR,
                    quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )
            continue
        ranges = _split_row_ranges(n_rows, num_shards)
        for s_idx, (r0, r1) in enumerate(ranges):
            cnt = r1 - r0
            entry_name = normalize_ple_name(name, shard_idx=(s_idx if num_shards > 1 else None), quantized=False)
            d_rng = _slice_row_byte_range(t, r0, r1)
            data_bytes = d_rng.length
            blob_bytes = _align_up(max(data_bytes, 1), APPLE_PAGE_SIZE)
            per_shard_entries[s_idx].append(
                _PlannedEntry(
                    name=entry_name,
                    kind=TensorKind.SPARSE_TABLE,
                    dtype=amf_dtype,
                    quant_bits=0,
                    group_size=0,
                    layer=layer,
                    expert=NO_EXPERT,
                    shape=(cnt, embed_dim),
                    data=d_rng,
                    data_convert=convert,
                    scales=None,
                    biases=None,
                    data_bytes=data_bytes,
                    params_offset=0,
                    params_bytes=0,
                    blob_bytes=blob_bytes,
                    layout=TensorLayout.ROW_MAJOR,
                    quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )

    readers = _ShardReaders()
    all_metas: list[list[PackedTensorMeta]] = []
    try:
        for s_idx, (shard_path, plan) in enumerate(zip(shard_files, per_shard_entries)):
            shard_id = f"{model_id}-s{s_idx}" if num_shards > 1 else model_id
            shard_id_bytes = shard_id.encode("utf-8")[:63]
            tensor_count = len(plan)
            dir_offset = APPLE_PAGE_SIZE
            dir_raw_len = tensor_count * ENTRY_STRUCT.size
            dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)
            cursor = dir_offset + dir_bytes
            offsets: list[int] = []
            for e in plan:
                offsets.append(cursor)
                cursor += e.blob_bytes

            shard_path.parent.mkdir(parents=True, exist_ok=True)
            partial = shard_path.with_name(shard_path.name + ".partial")
            metas: list[PackedTensorMeta] = []
            try:
                with partial.open("wb") as out:
                    _set_nocache(out.fileno())
                    _BlobWriter(out).zeros(dir_offset + dir_bytes)
                    for e, blob_offset in zip(plan, offsets):
                        w = _BlobWriter(out)
                        _stream_entry(e, readers, w)
                        if w.written != e.blob_bytes:
                            raise AssertionError(f"wrote {w.written} B for {e.name}, planned {e.blob_bytes}")
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
                    dir_raw = _pack_directory(metas)
                    hdr = _pack_header(shard_id_bytes, tensor_count, dir_offset, dir_bytes, _crc32(dir_raw))
                    out.seek(0)
                    out.write(hdr)
                    if dir_raw:
                        out.seek(dir_offset)
                        out.write(dir_raw)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(partial, shard_path)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            all_metas.append(metas)
    finally:
        readers.close()
    return shard_files, all_metas


def _prepare_loop_adapter_records(
    adapter_tensors: dict[str, _ShardTensor],
    readers: _ShardReaders,
    *,
    dtype_mode: str = "auto",
    force_zero_r0_alpha: bool = False,
) -> tuple[list[tuple], dict[str, object]]:
    """Convert raw Step-LoRA safetensors into 12-tuple records for `sme2_panel64` packing.

    Enforces ADR-0019 `alpha == 0.0` at `r = 0`, packs `lora_a [packed_rank, K]` and
    `lora_b [N, packed_rank]` into `TensorLayout.SME2_PANEL64` (zero-padding rank up
    to `align_up(rank, 64)` when `rank % 64 != 0` so `(x @ A^T) @ B^T` is mathematically
    identical while satisfying `amf.zig`'s strict `k % 64 == 0` invariant).
    """
    records: list[tuple] = []
    steps_seen: set[int] = set()
    layers_seen: set[int] = set()
    projs_seen: set[str] = set()
    effective_rank = 0
    packed_rank = 0

    for raw_name in sorted(adapter_tensors):
        t = adapter_tensors[raw_name]
        m = _LOOP_ADAPTER_RE.match(raw_name)
        if m is None:
            raise ValueError(
                f"unrecognized loop adapter tensor name {raw_name!r}; expected "
                "'loop.step.{r}.layers.{l}.{proj}.{lora_a|lora_b|alpha}'"
            )
        step = int(m.group(1))
        layer = int(m.group(2))
        proj = m.group(3)
        part = m.group(4)
        steps_seen.add(step)
        layers_seen.add(layer)
        projs_seen.add(proj)
        canonical_name = f"loop.step.{step}.layers.{layer}.{proj}.{part}"
        if len(canonical_name.encode("utf-8")) > 47:
            raise ValueError(f"loop adapter name exceeds 47 bytes: {canonical_name!r}")

        fh = readers.get(t.path)
        raw = _read_exact(fh, t.offset, t.length)

        if part == "alpha":
            if t.dtype == "F32":
                vals = f32_bytes_to_f32_list(raw)
            elif t.dtype == "F16":
                vals = f16_bytes_to_f32_list(raw)
            elif t.dtype == "BF16":
                vals = bf16_bytes_to_f32_list(raw)
            else:
                raise ValueError(f"unsupported loop adapter alpha dtype {t.dtype} for {raw_name!r}")
            if not vals:
                raise ValueError(f"empty alpha tensor {raw_name!r}")
            if step == 0 and any(v != 0.0 for v in vals):
                if force_zero_r0_alpha:
                    vals = [0.0] * len(vals)
                else:
                    raise ValueError(
                        f"ADR-0019 violation: loop adapter {canonical_name!r} at r=0 must have alpha == 0.0, got {vals!r}"
                    )
            f32_bytes = struct.pack(f"<{len(vals)}f", *vals)
            records.append(
                (
                    canonical_name,
                    TensorKind.LOOP_ADAPTER,
                    DType.F32,
                    0,
                    0,
                    layer,
                    NO_EXPERT,
                    (len(vals),),
                    f32_bytes,
                    b"",
                    TensorLayout.ROW_MAJOR,
                    QuantParamKind.AFFINE_SCALE_BIAS,
                )
            )
            continue

        # lora_a [rank, K] or lora_b [N, rank]
        if len(t.shape) != 2:
            raise ValueError(f"loop adapter {raw_name!r} must be 2D, got {t.shape}")
        n_rows, k_cols = t.shape
        amf_dtype, convert = _unquantized_target(t.dtype, dtype_mode, raw_name)
        if amf_dtype == DType.F32:
            # Narrow f32 LoRA weights to f16 for hardware SME2 `fmopa` execution.
            vals = f32_bytes_to_f32_list(raw)
            raw_16 = struct.pack(f"<{len(vals)}e", *(max(-65504.0, min(65504.0, v)) for v in vals))
            amf_dtype = DType.F16
        elif convert == "bf16_to_f16":
            raw_16 = bf16_bytes_to_f16_bytes(raw, what=raw_name)
        elif convert == "f16_to_bf16":
            raw_16 = f16_bytes_to_bf16_bytes(raw)
        else:
            raw_16 = raw

        if part == "lora_a":
            rank = n_rows
            if rank <= 0 or rank > 64:
                raise ValueError(f"Step-LoRA rank must be in 1..64 for {raw_name!r}, got {rank}")
            if k_cols <= 0 or k_cols % SME2_GROUP_K != 0:
                raise ValueError(f"lora_a K={k_cols} must be a positive multiple of 64 for {raw_name!r}")
            p_rank = _align_up(rank, SME2_GROUP_K)
            effective_rank = max(effective_rank, rank)
            packed_rank = max(packed_rank, p_rank)
            row_bytes = k_cols * 2
            mv = memoryview(raw_16)
            rows_16 = [bytes(mv[r * row_bytes : (r + 1) * row_bytes]) for r in range(rank)]
            if p_rank > rank:
                zero_row = bytes(row_bytes)
                rows_16.extend(zero_row for _ in range(p_rank - rank))
            out_shape = (p_rank, k_cols)
        else:
            rank = k_cols
            if rank <= 0 or rank > 64:
                raise ValueError(f"Step-LoRA rank must be in 1..64 for {raw_name!r}, got {rank}")
            p_rank = _align_up(rank, SME2_GROUP_K)
            effective_rank = max(effective_rank, rank)
            packed_rank = max(packed_rank, p_rank)
            src_row_bytes = rank * 2
            pad_bytes = (p_rank - rank) * 2
            mv = memoryview(raw_16)
            if pad_bytes > 0:
                pad = bytes(pad_bytes)
                rows_16 = [bytes(mv[r * src_row_bytes : (r + 1) * src_row_bytes]) + pad for r in range(n_rows)]
            else:
                rows_16 = [bytes(mv[r * src_row_bytes : (r + 1) * src_row_bytes]) for r in range(n_rows)]
            out_shape = (n_rows, p_rank)

        panel_chunks: list[bytes] = []
        out_n, out_k = out_shape
        for p0 in range(0, out_n, SME2_PANEL_COLS):
            panel_chunks.append(pack_sme2_panel64_f16_panel(rows_16[p0 : p0 + SME2_PANEL_COLS], out_k))
        packed_blob = b"".join(panel_chunks)
        expected = sme2_panel64_data_bytes(amf_dtype, out_n, out_k, 0)
        if len(packed_blob) != expected:
            raise AssertionError(f"loop adapter {canonical_name}: packed {len(packed_blob)} B != {expected}")
        records.append(
            (
                canonical_name,
                TensorKind.LOOP_ADAPTER,
                amf_dtype,
                0,
                0,
                layer,
                NO_EXPERT,
                out_shape,
                packed_blob,
                b"",
                TensorLayout.SME2_PANEL64,
                QuantParamKind.AFFINE_SCALE_BIAS,
            )
        )

    summary: dict[str, object] = {
        "steps": sorted(steps_seen),
        "layers": sorted(layers_seen),
        "projs": sorted(projs_seen),
        "effective_rank": effective_rank,
        "packed_rank": packed_rank,
    }
    return records, summary


def _effective_text_config(config: dict[str, object] | None) -> dict[str, object]:
    """Flatten HF `config.json` by merging nested `text_config` over top-level fields."""
    if not isinstance(config, dict):
        return {}
    merged = dict(config)
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        merged.update(text_cfg)
    return merged


def infer_operator_axes_and_loop_schedule(
    config: dict[str, object] | None,
    tensor_names: Iterable[str],
    *,
    ple_shard_count: int = 0,
    loop_adapter_summary: dict[str, object] | None = None,
) -> tuple[OperatorAxes, LoopScheduleConfig]:
    """Infer canonical `OperatorAxes` and `LoopScheduleConfig` from HF config + packed tensor names."""
    cfg = _effective_text_config(config)
    names = [_strip_model_prefix(nm) for nm in tensor_names]

    # Determine layer count from config or tensor names.
    num_layers = int(cfg.get("num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        max_l = -1
        for nm in names:
            lm = re.search(r"(?:^|\.)layers\.(\d+)\.", nm)
            if lm:
                max_l = max(max_l, int(lm.group(1)))
        num_layers = max(1, max_l + 1)

    has_linear_attn = any(".linear_attn." in nm for nm in names)
    has_qsa_idx = (
        any(
            ".idx_wq" in nm or ".idx_wk" in nm or ".qsa_" in nm or ".self_attn.indexer." in nm or ".attn.idx." in nm
            for nm in names
        )
        or int(cfg.get("indexer_n_heads", 0) or 0) > 0
    )
    has_gr = (
        any(
            ".gr." in nm or "hyper_connection" in nm or ".attn_hc." in nm or ".mlp_hc." in nm or "hc_mixer." in nm
            for nm in names
        )
        or int(cfg.get("hc_count", 0) or 0) >= 2
    )
    has_shared_expert = (
        any(".shared_expert." in nm or ".shared_experts." in nm or ".shared_expert_gate" in nm for nm in names)
        or int(cfg.get("shared_expert_intermediate_size", 0) or 0) > 0
    )
    has_moe = any(".mlp.experts." in nm or ".mlp.gate" in nm for nm in names) or int(cfg.get("num_experts", 0) or 0) > 0
    has_ple = (
        ple_shard_count > 0
        or any(is_ple_tensor(nm) or is_dense_ple_tensor(nm) for nm in names)
        or bool(cfg.get("ple_layer_ids"))
        or int(cfg.get("ngram_vocab_size_base", 0) or 0) > 0
    )
    has_early_exit = any("early_exit_gate" in nm for nm in names)

    # Per-layer attention kinds (respect explicit config layer_types or 3:1 GDN:QSA interval).
    raw_layer_types = cfg.get("layer_types")
    layer_attns: list[str] = []
    if isinstance(raw_layer_types, list) and len(raw_layer_types) == num_layers:
        for lt in raw_layer_types:
            s = str(lt).lower()
            if "linear" in s or "delta" in s or "gdn" in s:
                layer_attns.append("gated_deltanet")
            elif "qsa" in s or "sparse" in s:
                layer_attns.append("qsa_sparse")
            else:
                layer_attns.append("qsa_sparse" if has_qsa_idx else "gqa_full")
    elif has_linear_attn:
        interval = int(cfg.get("full_attention_interval", 4) or 4)
        for l_idx in range(num_layers):
            if any(nm.startswith(f"layers.{l_idx}.linear_attn.") for nm in names):
                layer_attns.append("gated_deltanet")
            elif any(nm.startswith(f"layers.{l_idx}.self_attn.") for nm in names):
                layer_attns.append("qsa_sparse" if has_qsa_idx else "gqa_full")
            else:
                layer_attns.append("qsa_sparse" if (l_idx + 1) % interval == 0 else "gated_deltanet")
    else:
        default_attn = "qsa_sparse" if has_qsa_idx else "gqa_full"
        layer_attns = [default_attn] * num_layers

    unique_attns = set(layer_attns)
    if len(unique_attns) > 1:
        attention_kind = "hybrid_gdn_qsa"
    else:
        attention_kind = layer_attns[0] if layer_attns else "gqa_full"

    ffn_kind = "moe_swiglu" if has_moe else "dense_swiglu"
    residual_branches = int(cfg.get("hc_count", 4) or 4) if has_gr else 1
    residual_kind = "gated_4branch" if has_gr else "single_stream"
    memory_lookup_kind = "ple_ngram" if has_ple else "none"

    # Infer loop_schedule from explicit config or loop adapter tensors.
    cfg_loop = cfg.get("loop_schedule")
    if isinstance(cfg_loop, dict):
        ls = LoopScheduleConfig(
            mode=str(cfg_loop.get("mode", "single_pass")),
            start_layer=int(cfg_loop.get("start_layer", 0)),
            end_layer=int(cfg_loop.get("end_layer", num_layers)),
            max_loops=int(cfg_loop.get("max_loops", 1)),
            inject_x_init=bool(cfg_loop.get("inject_x_init", False)),
            adapter_kind=str(cfg_loop.get("adapter_kind", "none")),
            adapter_rank=int(cfg_loop.get("adapter_rank", 0)),
            adapter_packed_rank=int(cfg_loop.get("adapter_packed_rank", 0)),
            adapter_projs=tuple(str(p) for p in cfg_loop.get("adapter_projs", ())),
            enforce_zero_alpha_r0=bool(cfg_loop.get("enforce_zero_alpha_r0", True)),
            early_exit_gate=str(cfg_loop.get("early_exit_gate", "sigmoid_linear" if has_early_exit else "none")),
        )
    elif loop_adapter_summary and loop_adapter_summary.get("steps"):
        steps = list(loop_adapter_summary["steps"])  # type: ignore[arg-type]
        a_layers = list(loop_adapter_summary["layers"])  # type: ignore[arg-type]
        projs = tuple(str(p) for p in loop_adapter_summary["projs"])  # type: ignore[arg-type]
        ls = LoopScheduleConfig(
            mode="gated_loop_adapter",
            start_layer=min(a_layers) if a_layers else 0,
            end_layer=(max(a_layers) + 1) if a_layers else num_layers,
            max_loops=max(steps) + 1,
            inject_x_init=False,
            adapter_kind="sme2_step_lora",
            adapter_rank=int(loop_adapter_summary.get("effective_rank", 64)),  # type: ignore[arg-type]
            adapter_packed_rank=int(loop_adapter_summary.get("packed_rank", 64)),  # type: ignore[arg-type]
            adapter_projs=projs,
            enforce_zero_alpha_r0=True,
            early_exit_gate="sigmoid_linear" if has_early_exit else "none",
        )
    elif int(cfg.get("recurrent_depth", cfg.get("num_recurrent_steps", 1)) or 1) > 1 or str(
        cfg.get("model_type", "")
    ).lower() in ("looplm", "huginn", "ouro"):
        r_steps = max(2, int(cfg.get("recurrent_depth", cfg.get("num_recurrent_steps", 3)) or 3))
        prelude_n = int(cfg.get("num_prelude_layers", 1 if num_layers >= 3 else 0) or 0)
        coda_n = int(cfg.get("num_coda_layers", 1 if num_layers >= 3 else 0) or 0)
        start_l = min(prelude_n, num_layers - 1)
        end_l = max(start_l + 1, num_layers - coda_n)
        ls = LoopScheduleConfig(
            mode="native_recurrent",
            start_layer=start_l,
            end_layer=end_l,
            max_loops=r_steps,
            inject_x_init=bool(cfg.get("inject_x_init", True)),
            adapter_kind="none",
            adapter_rank=0,
            adapter_packed_rank=0,
            adapter_projs=(),
            enforce_zero_alpha_r0=True,
            early_exit_gate="sigmoid_linear" if has_early_exit else "none",
        )
    else:
        ls = LoopScheduleConfig(
            mode="single_pass",
            start_layer=0,
            end_layer=num_layers,
            max_loops=1,
            inject_x_init=False,
            adapter_kind="none",
            adapter_rank=0,
            adapter_packed_rank=0,
            adapter_projs=(),
            enforce_zero_alpha_r0=True,
            early_exit_gate="sigmoid_linear" if has_early_exit else "none",
        )

    # Upgrade to gated_loop_adapter if external adapters were merged onto an existing config.
    if loop_adapter_summary and loop_adapter_summary.get("steps") and ls.adapter_kind == "none":
        steps = list(loop_adapter_summary["steps"])  # type: ignore[arg-type]
        a_layers = list(loop_adapter_summary["layers"])  # type: ignore[arg-type]
        ls = LoopScheduleConfig(
            mode="gated_loop_adapter",
            start_layer=min(a_layers) if a_layers else ls.start_layer,
            end_layer=(max(a_layers) + 1) if a_layers else ls.end_layer,
            max_loops=max(ls.max_loops, max(steps) + 1),
            inject_x_init=ls.inject_x_init,
            adapter_kind="sme2_step_lora",
            adapter_rank=int(loop_adapter_summary.get("effective_rank", 64)),  # type: ignore[arg-type]
            adapter_packed_rank=int(loop_adapter_summary.get("packed_rank", 64)),  # type: ignore[arg-type]
            adapter_projs=tuple(str(p) for p in loop_adapter_summary["projs"]),  # type: ignore[arg-type]
            enforce_zero_alpha_r0=True,
            early_exit_gate=ls.early_exit_gate,
        )

    ls.validate(num_layers)
    axes = OperatorAxes(
        backbone_kind="decoder_only",
        attention_kind=attention_kind,
        layer_attention_kinds=tuple(layer_attns),
        ffn_kind=ffn_kind,
        has_shared_expert=has_shared_expert,
        residual_kind=residual_kind,
        residual_branches=residual_branches,
        memory_lookup_kind=memory_lookup_kind,
        loop_schedule_kind=ls.mode,
    )
    return axes, ls


@dataclass(frozen=True)
class Qwen4PackReport:
    amf_path: pathlib.Path
    metas: list[PackedTensorMeta]
    stripped_vision_tensors: int
    ple_shard_paths: list[pathlib.Path]
    ple_metas: list[list[PackedTensorMeta]]
    mtp_amf_path: pathlib.Path | None
    mtp_metas: list[PackedTensorMeta]
    loop_adapter_count: int
    config_path: pathlib.Path | None
    operator_axes: OperatorAxes
    loop_schedule: LoopScheduleConfig
    vision_amf_path: pathlib.Path | None = None
    vision_metas: list[PackedTensorMeta] = field(default_factory=list)
    included_vision_tensors: int = 0
    mtp_config_path: pathlib.Path | None = None


def _write_backbone_with_adapters(
    backbone_plan: list[_PlannedEntry],
    adapter_records: list[tuple],
    output_path: pathlib.Path,
    *,
    model_id: str,
    readers: _ShardReaders,
    lut16_sme2: bool = False,
    workers: int = 1,
) -> list[PackedTensorMeta]:
    """Stream planned backbone entries + pre-packed `LOOP_ADAPTER` records into one `.amf`."""
    model_id_bytes = model_id.encode("utf-8")
    if len(model_id_bytes) == 0 or len(model_id_bytes) > 63:
        raise ValueError(f"model_id must be 1..63 UTF-8 bytes: {model_id!r}")

    # Validate unique tensor names across backbone and adapters.
    seen_names: set[str] = {e.name for e in backbone_plan}
    for rec in adapter_records:
        if rec[0] in seen_names:
            raise ValueError(f"duplicate tensor name between backbone and loop adapter: {rec[0]!r}")
        seen_names.add(rec[0])

    tensor_count = len(backbone_plan) + len(adapter_records)
    dir_offset = APPLE_PAGE_SIZE
    dir_raw_len = tensor_count * ENTRY_STRUCT.size
    dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)

    cursor = dir_offset + dir_bytes
    bb_offsets: list[int] = []
    for e in backbone_plan:
        bb_offsets.append(cursor)
        cursor += e.blob_bytes

    ad_layouts: list[tuple[tuple, int, int, int, int, int]] = []
    for rec in adapter_records:
        data = rec[8]
        params = rec[9]
        if params:
            p_off = _align_up(len(data), 64)
            content = p_off + len(params)
        else:
            p_off = 0
            content = len(data)
        blob_b = _align_up(max(content, 1), APPLE_PAGE_SIZE)
        ad_layouts.append((rec, cursor, blob_b, len(data), p_off, len(params)))
        cursor += blob_b

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    metas: list[PackedTensorMeta] = []
    try:
        with partial.open("wb") as out:
            _set_nocache(out.fileno())
            _BlobWriter(out).zeros(dir_offset + dir_bytes)

            for e, blob_offset in zip(backbone_plan, bb_offsets):
                if out.tell() != blob_offset:
                    raise AssertionError(f"planned offset {blob_offset} != pos {out.tell()} for {e.name}")
                w = _BlobWriter(out)
                eff_pk = e.quant_param_kind
                if (
                    lut16_sme2
                    and e.layout == TensorLayout.SME2_PANEL64
                    and e.dtype in (DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE)
                ):
                    from aminfer_tools.gsq_rco import stream_safetensors_entry_lut16_sme2

                    stream_safetensors_entry_lut16_sme2(e, readers, w, workers=workers)
                    eff_pk = QuantParamKind.LUT16_CODEBOOK
                else:
                    _stream_entry(e, readers, w)
                if w.written != e.blob_bytes:
                    raise AssertionError(f"wrote {w.written} B for {e.name}, planned {e.blob_bytes}")
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
                        quant_param_kind=eff_pk,
                    )
                )

            for rec, blob_offset, blob_b, data_b, p_off, p_b in ad_layouts:
                name, kind, dtype, q_bits, g_size, layer, expert, shape, data, params, layout, pk = rec
                if out.tell() != blob_offset:
                    raise AssertionError(f"planned offset {blob_offset} != pos {out.tell()} for {name}")
                w = _BlobWriter(out)
                w.write(data)
                if params:
                    w.zeros(p_off - len(data))
                    w.write(params)
                w.zeros(blob_b - w.written)
                metas.append(
                    PackedTensorMeta(
                        name=name,
                        kind=kind,
                        dtype=dtype,
                        quant_bits=q_bits,
                        group_size=g_size,
                        layer=layer,
                        expert=expert,
                        shape=shape,
                        blob_offset=blob_offset,
                        blob_bytes=blob_b,
                        data_bytes=data_b,
                        params_offset=p_off,
                        params_bytes=p_b,
                        blob_crc32=w.crc & 0xFFFFFFFF,
                        layout=layout,
                        quant_param_kind=pk,
                    )
                )

            if out.tell() != cursor:
                raise AssertionError(f"final size {out.tell()} != planned {cursor}")
            dir_raw = _pack_directory(metas)
            hdr = _pack_header(model_id_bytes, tensor_count, dir_offset, dir_bytes, _crc32(dir_raw))
            out.seek(0)
            out.write(hdr)
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


def pack_qwen4_checkpoint(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str = "qwen4-exp",
    group_size: int = 64,
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
    dtype_mode: str = "auto",
    tie_word_embeddings: bool | None = None,
    exclude_tensors: Iterable[str] = (),
    heat_profile: pathlib.Path | None = None,
    cold_experts_per_layer: int | None = None,
    sme2_pack: str = "none",
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
    ple_output: pathlib.Path | None = None,
    ple_shards: int = 1,
    mtp_output: pathlib.Path | None = None,
    loop_adapter_input: pathlib.Path | None = None,
    force_zero_r0_alpha: bool = False,
    config_output: pathlib.Path | None = None,
    mtp_config_output: pathlib.Path | None = None,
    target_tier: str = "auto",
    host_ram_gib: float | None = None,
    split_stacked_gate_up: bool | None = None,
) -> Qwen4PackReport:
    """Import a `qwen4_exp` / `Qwen3.8` / `Qwen3-VL` or native/adapter-extended `LoopLM` safetensors checkpoint.

    Handles full multi-layer vision towers (`visual.patch_embed.*`, `visual.blocks.0..N.*`,
    `visual.merger.*`) via W20 default-include, `--strip-vision` opt-out, or standalone
    `--vision-sidecar vision.amf` extraction, extracts sharded `SPARSE_TABLE = 8` PLE
    `.amf` files and `4B MTP` draft head `.amf` files when present, packs `SHARED_EXPERT = 7`
    and `>= 512` routed experts, packs `LOOP_ADAPTER = 9` Step-LoRA weights into
    `sme2_panel64` (enforcing `alpha == 0.0` at `r = 0`), and emits canonical
    `operator_axes` + `loop_schedule` + `vision` metadata in `config.json`.
    """
    if quant_profile not in QUANT_PROFILES:
        raise ValueError(f"quant_profile must be one of {QUANT_PROFILES}, got {quant_profile!r}")
    if sme2_pack not in SME2_PACK_MODES:
        raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
    if vision_block_dtype not in VISION_BLOCK_DTYPES:
        raise ValueError(f"vision_block_dtype must be one of {VISION_BLOCK_DTYPES}, got {vision_block_dtype!r}")
    if ple_shards < 1:
        raise ValueError(f"ple_shards must be >= 1, got {ple_shards}")

    effective_cold: set[tuple[int, int]] = set(cold_experts)
    if heat_profile is not None:
        effective_cold.update(
            load_cold_experts_from_heat_profile(
                pathlib.Path(heat_profile),
                cold_experts_per_layer=cold_experts_per_layer,
            )
        )
    elif cold_experts_per_layer is not None:
        raise ValueError("cold_experts_per_layer requires heat_profile in pack_qwen4_checkpoint")

    resolved = resolve_checkpoint(pathlib.Path(input_path))
    tie = _load_tie_flag(resolved.config_path) if tie_word_embeddings is None else tie_word_embeddings
    src_config: dict[str, object] = {}
    if resolved.config_path is not None and resolved.config_path.is_file():
        loaded = json.loads(resolved.config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            src_config = loaded

    all_tensors = _collect_tensors(resolved.shard_paths)
    has_real_hf_q4_keys = any(
        (
            ".attn_hyper_connection." in k
            or ".mlp_hyper_connection." in k
            or "hyper_connection_mixer." in k
            or ".linear_attn.in_proj_" in k
            or ".mlp.experts.gate_up_proj" in k
            or ".self_attn.indexer." in k
            or is_dense_ple_tensor(k)
        )
        for k in all_tensors
    )
    eff_qwen4_cfg: dict[str, object] | None = (
        _effective_text_config(src_config) if (is_qwen4_config(src_config) or has_real_hf_q4_keys) else None
    )
    eff_split_gate_up = True if split_stacked_gate_up is None else bool(split_stacked_gate_up)
    eff_target_bits = target_bits
    if quant_profile == "gsq-rco" and eff_target_bits is None and target_tier != "auto":
        tier_bits_map = {
            "24gb-m4": 3.25,
            "128gb-max": 3.50,
            "512gb-ultra": 4.25,
        }
        if target_tier in tier_bits_map:
            eff_target_bits = tier_bits_map[target_tier]
    _ = host_ram_gib  # Reserved for tier-guided GSQ admission policies.
    vision_mode = resolve_vision_mode(
        all_tensors.keys(),
        include_vision=include_vision,
        strip_vision=strip_vision,
        vision_sidecar=vision_sidecar,
    )
    stripped_vision = 0
    included_vision = 0
    vision_tensors: dict[str, _ShardTensor] = {}
    ple_tensors: dict[str, _ShardTensor] = {}
    mtp_tensors: dict[str, _ShardTensor] = {}
    embedded_adapter_tensors: dict[str, _ShardTensor] = {}
    backbone_tensors: dict[str, _ShardTensor] = {}

    for raw_name, st in all_tensors.items():
        if is_vision_tensor(raw_name):
            if vision_mode == "strip":
                stripped_vision += 1
                continue
            if vision_mode == "sidecar":
                vision_tensors[raw_name] = st
                continue
            included_vision += 1
            backbone_tensors[raw_name] = st
            continue
        if is_ple_tensor(raw_name):
            ple_tensors[raw_name] = st
            continue
        if is_mtp_tensor(raw_name):
            mtp_tensors[raw_name] = st
            continue
        if is_loop_adapter_tensor(raw_name):
            embedded_adapter_tensors[raw_name] = st
            continue
        backbone_tensors[raw_name] = st

    out_path = pathlib.Path(output_path)

    # Pack standalone vision sidecar when requested.
    vision_amf_path: pathlib.Path | None = None
    vision_metas: list[PackedTensorMeta] = []
    if vision_mode == "sidecar" and vision_sidecar is not None and vision_tensors:
        vision_amf_path = pathlib.Path(vision_sidecar)
        vision_metas = pack_vision_tower_amf(
            input_path,
            vision_amf_path,
            model_id=f"{model_id[:55]}-vision",
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            vision_block_dtype=vision_block_dtype,
            group_size=group_size,
            use_rco=use_rco,
            workers=workers,
        )

    # Load optional external loop adapter safetensors.
    external_adapter_tensors: dict[str, _ShardTensor] = {}
    if loop_adapter_input is not None:
        ad_resolved = resolve_checkpoint(pathlib.Path(loop_adapter_input))
        external_adapter_tensors = _collect_tensors(ad_resolved.shard_paths)
    merged_adapter_tensors = {**embedded_adapter_tensors, **external_adapter_tensors}

    # Pack PLE shards when PLE tensors exist.
    ple_paths: list[pathlib.Path] = []
    ple_metas: list[list[PackedTensorMeta]] = []
    if ple_tensors:
        eff_ple_out = pathlib.Path(ple_output) if ple_output is not None else out_path.with_name(out_path.stem + "-ple.amf")
        ple_paths, ple_metas = pack_ple_shards_from_safetensors(
            ple_tensors,
            eff_ple_out,
            num_shards=ple_shards,
            model_id=f"{model_id[:55]}-ple",
            group_size=group_size,
            dtype_mode=dtype_mode,
        )

    # Pack MTP draft head when MTP tensors exist.
    mtp_path: pathlib.Path | None = None
    mtp_cfg_out_path: pathlib.Path | None = None
    mtp_metas: list[PackedTensorMeta] = []
    if mtp_tensors:
        eff_mtp_out = pathlib.Path(mtp_output) if mtp_output is not None else out_path.with_name(out_path.stem + "-mtp.amf")
        remapped_mtp: dict[str, _ShardTensor] = {}
        for raw_name, st in mtp_tensors.items():
            clean = strip_mtp_prefix(raw_name)
            remapped_mtp[clean] = _ShardTensor(
                name=clean,
                dtype=st.dtype,
                shape=st.shape,
                offset=st.offset,
                length=st.length,
                path=st.path,
            )
        mtp_plan = plan_amf_entries(
            remapped_mtp,
            group_size=group_size,
            dtype_mode=dtype_mode,
            tie_word_embeddings=False,
            sme2_pack=sme2_pack,
            strip_vision=False,
            qwen4_config=eff_qwen4_cfg,
            qwen4_mtp_standalone=True,
            split_stacked_gate_up=eff_split_gate_up,
        )
        readers_mtp = _ShardReaders()
        try:
            mtp_metas = _write_backbone_with_adapters(
                mtp_plan,
                [],
                eff_mtp_out,
                model_id=f"{model_id[:55]}-mtp",
                readers=readers_mtp,
                lut16_sme2=lut16_sme2,
                workers=workers,
            )
        finally:
            readers_mtp.close()
        mtp_path = eff_mtp_out
        if mtp_config_output is not None or config_output is not None or mtp_output is not None:
            mtp_cfg_out_path = (
                pathlib.Path(mtp_config_output)
                if mtp_config_output is not None
                else eff_mtp_out.with_name(
                    "mtp_config.json" if eff_mtp_out.name == "mtp.amf" else f"{eff_mtp_out.stem}_config.json"
                )
            )
            eff_tc = eff_qwen4_cfg if eff_qwen4_cfg is not None else _effective_text_config(src_config)
            mtp_cfg_obj = {
                "draft_format": "qwen4_exp_mtp",
                "amf_file": mtp_path.name,
                "num_hidden_layers": int(eff_tc.get("mtp_num_hidden_layers", 1) or 1),
                "hidden_size": int(eff_tc.get("hidden_size", 0) or 0),
                "vocab_size": int(eff_tc.get("vocab_size", 0) or 0),
                "num_attention_heads": int(eff_tc.get("num_attention_heads", 0) or 0),
                "num_key_value_heads": int(eff_tc.get("num_key_value_heads", 0) or 0),
                "head_dim": int(eff_tc.get("head_dim", 0) or 0),
                "hc_count": int(eff_tc.get("hc_count", 4) or 4),
                "hc_lowrank": int(eff_tc.get("hc_lowrank", 0) or 0),
                "output_gate_type": str(eff_tc.get("output_gate_type", "sigmoid") or "sigmoid"),
                "indexer_n_heads": int(eff_tc.get("indexer_n_heads", 0) or 0),
                "indexer_kv_heads": int(eff_tc.get("indexer_kv_heads", 1) or 1),
                "indexer_head_dim": int(eff_tc.get("indexer_head_dim", 0) or 0),
                "num_experts": int(eff_tc.get("num_experts", 0) or 0),
                "num_experts_per_tok": int(eff_tc.get("num_experts_per_tok", 0) or 0),
                "moe_intermediate_size": int(eff_tc.get("moe_intermediate_size", 0) or 0),
                "shared_expert_intermediate_size": int(eff_tc.get("shared_expert_intermediate_size", 0) or 0),
                "tensor_count": len(mtp_metas),
                "tensor_names_max_bytes": max((len(m.name.encode("utf-8")) for m in mtp_metas), default=0),
            }
            mtp_cfg_out_path.parent.mkdir(parents=True, exist_ok=True)
            mtp_cfg_out_path.write_text(json.dumps(mtp_cfg_obj, indent=2) + "\n", encoding="utf-8")

    # Prepare loop adapters (validates r=0 alpha == 0.0 and packs into sme2_panel64).
    readers = _ShardReaders()
    try:
        adapter_records: list[tuple] = []
        adapter_summary: dict[str, object] | None = None
        if merged_adapter_tensors:
            adapter_records, adapter_summary = _prepare_loop_adapter_records(
                merged_adapter_tensors,
                readers,
                dtype_mode=dtype_mode,
                force_zero_r0_alpha=force_zero_r0_alpha,
            )

        if quant_profile == "gsq-rco" and not adapter_records and not ple_tensors and not mtp_tensors:
            stats_map: dict[str, int] = {}
            metas = pack_amf_from_safetensors(
                input_path,
                out_path,
                model_id=model_id,
                group_size=group_size,
                cold_experts=effective_cold,
                all_experts_cold=all_experts_cold,
                dtype_mode=dtype_mode,
                tie_word_embeddings=tie,
                exclude_tensors=exclude_tensors,
                heat_profile=heat_profile,
                cold_experts_per_layer=cold_experts_per_layer,
                sme2_pack=sme2_pack,
                lut16_sme2=lut16_sme2,
                quant_profile="gsq-rco",
                target_bits=eff_target_bits,
                use_rco=use_rco,
                manifest_path=manifest_path,
                workers=workers,
                strip_vision=(vision_mode in ("strip", "sidecar")),
                include_vision=(vision_mode == "include"),
                vision_block_dtype=vision_block_dtype,
                stats_out=stats_map,
                qwen4_config=eff_qwen4_cfg,
                split_stacked_gate_up=eff_split_gate_up,
            )
            if vision_mode == "strip":
                stripped_vision = stats_map.get("stripped_vision_tensors", stripped_vision)
        else:
            eff_sme2 = "all-linear" if (lut16_sme2 and sme2_pack == "none") else sme2_pack
            backbone_plan = plan_amf_entries(
                backbone_tensors,
                group_size=group_size,
                dtype_mode=dtype_mode,
                cold_experts=effective_cold,
                all_experts_cold=all_experts_cold,
                tie_word_embeddings=tie,
                exclude_tensors=exclude_tensors,
                sme2_pack=eff_sme2,
                strip_vision=False,
                qwen4_config=eff_qwen4_cfg,
                split_stacked_gate_up=eff_split_gate_up,
            )
            metas = _write_backbone_with_adapters(
                backbone_plan,
                adapter_records,
                out_path,
                model_id=model_id,
                readers=readers,
                lut16_sme2=lut16_sme2,
                workers=workers,
            )
            if quant_profile == "gsq-rco":
                metas = gsq_convert_with_vision(
                    out_path,
                    out_path,
                    target_bits=eff_target_bits,
                    model_id=model_id,
                    group_size=group_size,
                    dtype_mode=dtype_mode,
                    heat_profile=heat_profile,
                    cold_experts=effective_cold,
                    cold_experts_per_layer=cold_experts_per_layer,
                    all_experts_cold=all_experts_cold,
                    sme2_pack=sme2_pack,
                    lut16_sme2=lut16_sme2,
                    use_rco=use_rco,
                    manifest_path=manifest_path,
                    workers=workers,
                    include_vision=(vision_mode == "include"),
                    strip_vision=(vision_mode != "include"),
                    vision_block_dtype=vision_block_dtype,
                    qwen4_config=eff_qwen4_cfg,
                    split_stacked_gate_up=eff_split_gate_up,
                )
    finally:
        readers.close()

    axes, ls = infer_operator_axes_and_loop_schedule(
        src_config,
        (m.name for m in metas),
        ple_shard_count=len(ple_paths),
        loop_adapter_summary=adapter_summary,
    )

    cfg_out_path: pathlib.Path | None = None
    if config_output is not None:
        cfg_out_path = pathlib.Path(config_output)
        out_cfg = dict(src_config)
        out_cfg["model_id"] = model_id
        out_cfg["operator_axes"] = axes.to_dict()
        out_cfg["loop_schedule"] = ls.to_dict()
        out_cfg["stripped_vision_tensors"] = stripped_vision
        if vision_amf_path is not None:
            vis_summary = extract_vision_config_summary(vision_metas, src_config=src_config)
            vis_summary["mode"] = "sidecar"
            vis_summary["amf_file"] = vision_amf_path.name
            out_cfg["vision"] = vis_summary
        elif included_vision > 0:
            vis_summary = extract_vision_config_summary(metas, src_config=src_config)
            vis_summary["mode"] = "embedded"
            out_cfg["vision"] = vis_summary
        if ple_paths or any(is_dense_ple_tensor(k) for k in backbone_tensors):
            ple_meta0 = ple_metas[0][0] if ple_metas and ple_metas[0] else None
            total_ple_rows = sum(
                m.shape[0]
                for sm in ple_metas
                for m in sm
                if ple_meta0 is not None and m.layer == ple_meta0.layer
            )
            eff_tc = eff_qwen4_cfg if eff_qwen4_cfg is not None else _effective_text_config(src_config)
            ple_obj: dict[str, object] = {
                "num_shards": len(ple_paths),
                "shard_files": [p.name for p in ple_paths],
                "num_rows": total_ple_rows,
                "embed_dim": ple_meta0.shape[1] if ple_meta0 else int(eff_tc.get("ple_embed_dim", 0) or 0),
                "dtype": ple_meta0.dtype.name.lower() if ple_meta0 else "f16",
                "ngram_order": int(eff_tc.get("ple_ngram_order", src_config.get("ple_ngram_order", 2)) or 2),
                "ple_layer_ids": list(eff_tc.get("ple_layer_ids", []))
                if isinstance(eff_tc.get("ple_layer_ids"), list)
                else [],
                "dense_conditioning": any(is_dense_ple_tensor(k) for k in backbone_tensors),
            }
            # Preserve exact 64-bit PLE primes / offsets / vocab sizes in config.json alongside U32x2 AMF entries.
            i64_readers = _ShardReaders()
            try:
                for raw_name, st in backbone_tensors.items():
                    if st.dtype == "I64" and len(st.shape) == 1 and is_dense_ple_tensor(raw_name):
                        raw_i64 = _read_exact(i64_readers.get(st.path), st.offset, st.length)
                        vals_i64 = list(struct.unpack(f"<{st.shape[0]}q", raw_i64))
                        if raw_name.endswith("layer_multipliers"):
                            ple_obj["layer_multipliers"] = vals_i64
                        elif raw_name.endswith("ngram_heads_offsets"):
                            ple_obj["ngram_heads_offsets"] = vals_i64
                        elif raw_name.endswith("ngram_heads_vocab_sizes"):
                            ple_obj["ngram_heads_vocab_sizes"] = vals_i64
            finally:
                i64_readers.close()
            out_cfg["ple"] = ple_obj
        if eff_qwen4_cfg is not None:
            og_t = str(eff_qwen4_cfg.get("output_gate_type") or "").lower()
            out_cfg.setdefault("attn_output_gate", bool(eff_qwen4_cfg.get("attn_output_gate", False)) or og_t == "sigmoid")
            if eff_qwen4_cfg.get("hc_count") is not None:
                out_cfg.setdefault("hc_count", eff_qwen4_cfg["hc_count"])
            if eff_qwen4_cfg.get("hc_lowrank") is not None:
                out_cfg.setdefault("hc_lowrank", eff_qwen4_cfg["hc_lowrank"])
            if eff_qwen4_cfg.get("mamba_ssm_dtype") is not None:
                out_cfg.setdefault("mamba_ssm_dtype", eff_qwen4_cfg["mamba_ssm_dtype"])
        if mtp_path is not None:
            mtp_entry_cfg: dict[str, object] = {
                "amf_file": mtp_path.name,
                "tensor_count": len(mtp_metas),
            }
            if mtp_cfg_out_path is not None:
                mtp_entry_cfg["config_file"] = mtp_cfg_out_path.name
            out_cfg["mtp"] = mtp_entry_cfg
        cfg_out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_out_path.write_text(json.dumps(out_cfg, indent=2) + "\n", encoding="utf-8")

    return Qwen4PackReport(
        amf_path=out_path,
        metas=metas,
        stripped_vision_tensors=stripped_vision,
        ple_shard_paths=ple_paths,
        ple_metas=ple_metas,
        mtp_amf_path=mtp_path,
        mtp_metas=mtp_metas,
        loop_adapter_count=len(adapter_records),
        config_path=cfg_out_path,
        operator_axes=axes,
        loop_schedule=ls,
        vision_amf_path=vision_amf_path,
        vision_metas=vision_metas,
        included_vision_tensors=included_vision,
        mtp_config_path=mtp_cfg_out_path,
    )


@dataclass(frozen=True)
class Qwen4MetadataReport:
    """Offline metadata-only dry-run report for HF `qwen4_exp` (`Qwen3.8-Flash-Next`) repos."""

    metadata_dir: pathlib.Path
    model_type: str
    architectures: tuple[str, ...]
    num_hidden_layers: int
    gdn_layers: int
    qsa_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    num_experts: int
    num_experts_per_tok: int
    has_shared_expert: bool
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    hc_count: int
    ple_layer_ids: tuple[int, ...]
    ple_shards_detected: int
    mtp_layers: int
    kv_bytes_per_token_fp16: int
    gdn_state_bytes_fp32: int
    total_index_entries: int
    shard_file_count: int
    index_total_size_bytes: int
    backbone_tensor_count: int
    vision_tensor_count: int
    ple_tensor_count: int
    mtp_tensor_count: int
    loop_adapter_tensor_count: int
    stacked_expert_tensor_count: int
    per_expert_tensor_count: int
    shared_expert_tensor_count: int
    hyper_connection_tensor_count: int
    qsa_indexer_tensor_count: int
    gdn_tensor_count: int
    long_backbone_names: tuple[str, ...]
    estimated_backbone_q3_bytes: int
    estimated_backbone_q4_bytes: int
    estimated_ple_bf16_bytes: int
    operator_axes: OperatorAxes
    loop_schedule: LoopScheduleConfig
    tier_admission: dict[str, dict[str, object]]
    tokenizer_present: bool
    dense_ple_tensor_count: int = 0
    header_summary_verified: bool = False
    header_summary_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "arch": "qwen4_exp",
            "metadata_dir": str(self.metadata_dir),
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "num_hidden_layers": self.num_hidden_layers,
            "gdn_layers": self.gdn_layers,
            "qsa_layers": self.qsa_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rotary_dim": self.rotary_dim,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "has_shared_expert": self.has_shared_expert,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "hc_count": self.hc_count,
            "ple_layer_ids": list(self.ple_layer_ids),
            "ple_shards_detected": self.ple_shards_detected,
            "mtp_layers": self.mtp_layers,
            "kv_bytes_per_token_fp16": self.kv_bytes_per_token_fp16,
            "gdn_state_bytes_fp32": self.gdn_state_bytes_fp32,
            "total_index_entries": self.total_index_entries,
            "shard_file_count": self.shard_file_count,
            "index_total_size_bytes": self.index_total_size_bytes,
            "tensor_buckets": {
                "backbone": self.backbone_tensor_count,
                "vision_strippable": self.vision_tensor_count,
                "ple_nvme_table": self.ple_tensor_count,
                "ple_dense_conditioning": self.dense_ple_tensor_count,
                "mtp_draft_head": self.mtp_tensor_count,
                "loop_adapter": self.loop_adapter_tensor_count,
                "stacked_experts": self.stacked_expert_tensor_count,
                "per_expert": self.per_expert_tensor_count,
                "shared_experts": self.shared_expert_tensor_count,
                "hyper_connections": self.hyper_connection_tensor_count,
                "qsa_indexer": self.qsa_indexer_tensor_count,
                "gdn_linear_attn": self.gdn_tensor_count,
            },
            "long_backbone_names_over_47b": len(self.long_backbone_names),
            "long_backbone_name_samples": list(self.long_backbone_names[:8]),
            "estimated_backbone_q3_bytes": self.estimated_backbone_q3_bytes,
            "estimated_backbone_q4_bytes": self.estimated_backbone_q4_bytes,
            "estimated_ple_bf16_bytes": self.estimated_ple_bf16_bytes,
            "operator_axes": self.operator_axes.to_dict(),
            "loop_schedule": self.loop_schedule.to_dict(),
            "tier_admission": self.tier_admission,
            "tokenizer_present": self.tokenizer_present,
            "header_summary_verified": self.header_summary_verified,
            "header_summary_notes": list(self.header_summary_notes),
        }


def _estimate_qwen4_wired_bytes(
    cfg: dict[str, object],
    *,
    bpw: float,
    gdn_layers: int,
    qsa_layers: int,
    has_shared_expert: bool,
) -> tuple[int, int, int]:
    """Return `(total_backbone_bytes, non_expert_bytes, routed_expert_bytes)` excluding NVMe PLE."""
    h = int(cfg.get("hidden_size", 2560) or 2560)
    vocab = int(cfg.get("vocab_size", 248320) or 248320)
    n_layers = gdn_layers + qsa_layers
    n_heads = int(cfg.get("num_attention_heads", 24) or 24)
    n_kv = int(cfg.get("num_key_value_heads", 2) or 2)
    head_dim = int(cfg.get("head_dim", 256) or 256)
    lin_kh = int(cfg.get("linear_num_key_heads", 16) or 16)
    lin_vh = int(cfg.get("linear_num_value_heads", 48) or 48)
    lin_kd = int(cfg.get("linear_key_head_dim", 128) or 128)
    lin_vd = int(cfg.get("linear_value_head_dim", 128) or 128)
    n_exp = int(cfg.get("num_experts", 512) or 512)
    moe_i = int(cfg.get("moe_intermediate_size", 640) or 640)
    shared_i = int(cfg.get("shared_expert_intermediate_size", moe_i) or moe_i) if has_shared_expert else 0
    hc_count = int(cfg.get("hc_count", 4) or 4)
    hc_lr = int(cfg.get("hc_lowrank", 320) or 320)
    tie = bool(cfg.get("tie_word_embeddings", False))

    qsa_elems = (n_heads * head_dim * 2 + 2 * n_kv * head_dim + n_heads * head_dim) * h
    gdn_qkv = (2 * lin_kh * lin_kd + lin_vh * lin_vd) * h
    gdn_z_out = (2 * lin_vh * lin_vd + 2 * lin_vh) * h
    gdn_elems = gdn_qkv + gdn_z_out
    hc_per_layer = 2 * (hc_count * h * hc_lr + hc_lr * hc_count * hc_count + hc_count * h)
    shared_per_layer = (3 * shared_i * h + h) if has_shared_expert else 0
    router_per_layer = n_exp * h

    non_exp_quant_elems = qsa_layers * qsa_elems + gdn_layers * gdn_elems + n_layers * shared_per_layer
    non_exp_fp16_elems = n_layers * (hc_per_layer + router_per_layer) + vocab * h
    lm_head_bytes = 0 if tie else int(vocab * h * bpw)
    non_expert_bytes = int(non_exp_quant_elems * bpw) + non_exp_fp16_elems * 2 + lm_head_bytes
    routed_expert_bytes = int(n_layers * n_exp * 3 * moe_i * h * bpw)
    return (non_expert_bytes + routed_expert_bytes, non_expert_bytes, routed_expert_bytes)


def validate_hf_metadata_dir(
    metadata_dir: pathlib.Path,
    *,
    host_ram_gib: float | None = None,
) -> Qwen4MetadataReport:
    """Validate a Hugging Face `qwen4_exp` (`Qwen3.8-Flash-Next`) metadata directory offline.

    Requires only `config.json` and `model.safetensors.index.json` (no `.safetensors` shards
    needed on disk). Maps every `weight_map` key, computes exact GDN/QSA/MoE/PLE geometry,
    KV bytes/token, and multi-tier Apple Silicon wired admission plans.
    """
    base = pathlib.Path(metadata_dir)
    if base.is_file() and base.name == "config.json":
        base = base.parent
    cfg_path = base / "config.json"
    idx_path = base / "model.safetensors.index.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json in {base}")
    if not idx_path.is_file():
        raise FileNotFoundError(f"missing model.safetensors.index.json in {base}")

    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cfg, dict):
        raise ValueError("config.json must be a JSON object")
    cfg = _effective_text_config(raw_cfg)

    idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
    if not isinstance(idx_obj, dict) or not isinstance(idx_obj.get("weight_map"), dict):
        raise ValueError("model.safetensors.index.json must contain a weight_map object")
    weight_map: dict[str, str] = idx_obj["weight_map"]
    idx_meta = idx_obj.get("metadata") if isinstance(idx_obj.get("metadata"), dict) else {}
    total_size = int(float(idx_meta.get("total_size", 0) or 0))
    shard_files = set(weight_map.values())

    vision_cnt = 0
    ple_cnt = 0
    dense_ple_cnt = 0
    mtp_cnt = 0
    loop_cnt = 0
    backbone_names: list[str] = []
    ple_shard_ids: set[int] = set()
    stacked_exp_cnt = 0
    per_exp_cnt = 0
    shared_exp_cnt = 0
    hc_cnt = 0
    qsa_idx_cnt = 0
    gdn_cnt = 0
    long_names: list[str] = []

    for raw_name in sorted(weight_map.keys()):
        if is_vision_tensor(raw_name):
            vision_cnt += 1
            continue
        if is_ple_tensor(raw_name):
            ple_cnt += 1
            sm = _PLE_SHARD_RE.search(raw_name)
            if sm is not None:
                ple_shard_ids.add(int(sm.group(1)))
            continue
        if is_mtp_tensor(raw_name):
            mtp_cnt += 1
            mtp_norm = normalize_qwen4_tensor_name(raw_name, mtp_standalone=True)
            if len(mtp_norm.encode("utf-8")) > 47:
                long_names.append(f"mtp:{mtp_norm}")
            continue
        if is_loop_adapter_tensor(raw_name):
            loop_cnt += 1
            continue
        if is_dense_ple_tensor(raw_name):
            dense_ple_cnt += 1
        norm = normalize_qwen4_tensor_name(raw_name)
        backbone_names.append(norm)
        if len(norm.encode("utf-8")) > 47:
            long_names.append(norm)
        lower = norm.lower()
        raw_lower = raw_name.lower()
        if ".linear_attn." in lower:
            gdn_cnt += 1
        if (
            ".self_attn.indexer." in raw_lower
            or ".attn.idx." in lower
            or ".idx_wq" in lower
            or ".idx_wk" in lower
        ):
            qsa_idx_cnt += 1
        if (
            "hyper_connection" in raw_lower
            or ".attn_hc." in lower
            or ".mlp_hc." in lower
            or lower.startswith("hc_mixer.")
            or ".gr." in lower
        ):
            hc_cnt += 1
        if ".shared_expert." in lower or ".shared_experts." in lower or "shared_expert_gate" in lower:
            shared_exp_cnt += 1
        if re.search(r"\.experts\.(\d+)\.", lower):
            per_exp_cnt += 1
        elif ".experts." in lower:
            stacked_exp_cnt += 1

    ple_shards_detected = len(ple_shard_ids) if ple_shard_ids else int(cfg.get("split_ngram_parts", 0) or 0)
    axes, ls = infer_operator_axes_and_loop_schedule(
        raw_cfg,
        backbone_names,
        ple_shard_count=ple_shards_detected,
    )
    gdn_layers = sum(1 for k in axes.layer_attention_kinds if k == "gated_deltanet")
    qsa_layers = sum(1 for k in axes.layer_attention_kinds if k in ("qsa_sparse", "gqa_full"))
    num_layers = len(axes.layer_attention_kinds)
    hidden_size = int(cfg.get("hidden_size", 0) or 0)
    vocab_size = int(cfg.get("vocab_size", 0) or 0)
    num_heads = int(cfg.get("num_attention_heads", 0) or 0)
    num_kv_heads = int(cfg.get("num_key_value_heads", 0) or 0)
    head_dim = int(cfg.get("head_dim", hidden_size // max(1, num_heads)) or 0)
    rope_params = cfg.get("rope_parameters") if isinstance(cfg.get("rope_parameters"), dict) else {}
    prf = float(cfg.get("partial_rotary_factor", rope_params.get("partial_rotary_factor", 1.0)) or 1.0)
    rotary_dim = int(round(head_dim * prf))
    num_experts = int(cfg.get("num_experts", 0) or 0)
    num_experts_per_tok = int(cfg.get("num_experts_per_tok", 0) or 0)
    moe_i = int(cfg.get("moe_intermediate_size", 0) or 0)
    shared_i = int(cfg.get("shared_expert_intermediate_size", 0) or 0)
    hc_count = int(cfg.get("hc_count", 4 if axes.residual_kind == "gated_4branch" else 1) or 1)
    raw_ple_ids = cfg.get("ple_layer_ids")
    ple_layer_ids = tuple(int(x) for x in raw_ple_ids) if isinstance(raw_ple_ids, list) else ()
    mtp_cfg = cfg.get("mtp") if isinstance(cfg.get("mtp"), dict) else {}
    mtp_layers = int(cfg.get("mtp_num_hidden_layers", mtp_cfg.get("num_hidden_layers", 0)) or 0)

    # FP16 KV bytes/token across QSA full-attention layers only (GDN layers hold O(1) fixed recurrent state).
    kv_bytes_per_token_fp16 = qsa_layers * 2 * num_kv_heads * head_dim * 2
    lin_vh = int(cfg.get("linear_num_value_heads", 0) or 0)
    lin_kd = int(cfg.get("linear_key_head_dim", 0) or 0)
    lin_vd = int(cfg.get("linear_value_head_dim", 0) or 0)
    gdn_state_bytes_fp32 = gdn_layers * lin_vh * lin_kd * lin_vd * 4

    q3_total, q3_non_exp, q3_exp = _estimate_qwen4_wired_bytes(
        cfg, bpw=(24 + 4) / 64.0, gdn_layers=gdn_layers, qsa_layers=qsa_layers, has_shared_expert=axes.has_shared_expert
    )
    q4_total, _, _ = _estimate_qwen4_wired_bytes(
        cfg, bpw=(32 + 4) / 64.0, gdn_layers=gdn_layers, qsa_layers=qsa_layers, has_shared_expert=axes.has_shared_expert
    )
    ngram_base = int(cfg.get("ngram_vocab_size_base", 0) or 0)
    ple_dim = int(cfg.get("ple_embed_dim", hidden_size) or hidden_size)
    est_ple_bf16 = ngram_base * ple_dim * 2 if ngram_base > 0 else 0

    tier_budgets = {
        "24gb-m4": (1, 18 * (1024**3)),
        "128gb-max": (1, 96 * (1024**3)),
        "512gb-ultra": (1, 384 * (1024**3)),
        "tb5-cluster-2x128gb": (2, 96 * (1024**3)),
        "tb5-cluster-4x128gb": (4, 96 * (1024**3)),
    }
    if host_ram_gib is not None and host_ram_gib > 0:
        tier_budgets["host-custom"] = (1, int(host_ram_gib * 0.75 * (1024**3)))

    tier_admission: dict[str, dict[str, object]] = {}
    for tier_name, (ws, budget_b) in tier_budgets.items():
        per_rank_q3 = q3_non_exp + (q3_exp + ws - 1) // ws
        fits_all_wired = per_rank_q3 <= budget_b
        fits_cold_offload = q3_non_exp < budget_b
        hot_expert_budget = max(0, budget_b - q3_non_exp)
        hot_frac = min(1.0, hot_expert_budget / q3_exp) if q3_exp > 0 else 1.0
        if fits_all_wired:
            mode = "full_wired_q3_ple_nvme"
        elif fits_cold_offload:
            mode = f"hybrid_hot_cold_experts_ple_nvme (~{hot_frac * 100.0:.1f}% routed experts hot)"
        else:
            mode = "exceeds_non_expert_budget"
        tier_admission[tier_name] = {
            "world_size": ws,
            "per_rank_budget_gib": round(budget_b / (1024**3), 2),
            "per_rank_q3_wired_gib": round(per_rank_q3 / (1024**3), 2),
            "non_expert_q3_gib": round(q3_non_exp / (1024**3), 2),
            "fits_all_wired_q3": fits_all_wired,
            "fits_with_nvme_cold_offload": fits_cold_offload,
            "mode": mode,
        }

    summary_verified = False
    summary_notes: list[str] = []
    summary_path = base / "safetensors_headers_summary.json"
    if summary_path.is_file():
        sum_obj = json.loads(summary_path.read_text(encoding="utf-8"))
        pat_map = sum_obj.get("pattern_summary") if isinstance(sum_obj, dict) else None
        if isinstance(pat_map, dict) and pat_map:
            if len(shard_files) != int(sum_obj.get("total_shards", len(shard_files)) or len(shard_files)):
                raise ValueError(
                    f"safetensors_headers_summary total_shards mismatch: {sum_obj.get('total_shards')} vs {len(shard_files)}"
                )
            hc_lr_cfg = int(cfg.get("hc_lowrank", 0) or 0)
            og_sigmoid = str(cfg.get("output_gate_type") or "").lower() == "sigmoid" or bool(
                cfg.get("attn_output_gate", False)
            )
            for pat_k, pat_v in pat_map.items():
                if not isinstance(pat_v, dict):
                    continue
                shp = pat_v.get("shape")
                dt_s = pat_v.get("dtype")
                if not isinstance(shp, list):
                    continue
                if pat_k.endswith("hyper_connection_mixer.hc_norm.weight") and hidden_size > 0:
                    exp = [hc_count * hidden_size]
                    if shp != exp:
                        raise ValueError(f"{pat_k} shape {shp} != expected {exp}")
                elif pat_k.endswith("hyper_connection_mixer.input_mix_weight_down.weight") and hc_lr_cfg > 0:
                    exp = [hc_lr_cfg, hc_count * hidden_size]
                    if shp != exp:
                        raise ValueError(f"{pat_k} shape {shp} != expected {exp}")
                elif pat_k.endswith(".attn_hyper_connection.block_inject_weight.weight") and hidden_size > 0:
                    exp = [hc_count, hc_count * hidden_size]
                    if shp != exp:
                        raise ValueError(f"{pat_k} shape {shp} != expected {exp}")
                elif ".self_attn.q_proj.weight" in pat_k and num_heads > 0 and head_dim > 0 and "mtp." not in pat_k:
                    exp_rows = (2 if og_sigmoid else 1) * num_heads * head_dim
                    if shp != [exp_rows, hidden_size]:
                        raise ValueError(f"{pat_k} shape {shp} != expected {[exp_rows, hidden_size]}")
                elif pat_k.endswith(".ple.conv1d.weight"):
                    if len(shp) != 3 or shp[1] != 1:
                        raise ValueError(f"{pat_k} expected 3D [C, 1, K], got {shp}")
                elif pat_k.endswith(
                    (
                        ".ple_embedding.layer_multipliers",
                        ".ple_embedding.ngram_heads_offsets",
                        ".ple_embedding.ngram_heads_vocab_sizes",
                    )
                ):
                    if dt_s != "I64":
                        raise ValueError(f"{pat_k} expected I64, got {dt_s}")
                elif pat_k.endswith(".mlp.experts.gate_up_proj") and num_experts > 0 and moe_i > 0:
                    exp = [num_experts, 2 * moe_i, hidden_size]
                    if shp != exp:
                        raise ValueError(f"{pat_k} shape {shp} != expected {exp}")
            summary_verified = True
            summary_notes.append(
                f"verified {len(pat_map)} patterns across {sum_obj.get('total_keys', len(weight_map))} tensors"
            )

    raw_archs = raw_cfg.get("architectures")
    archs = tuple(str(a) for a in raw_archs) if isinstance(raw_archs, list) else ()
    return Qwen4MetadataReport(
        metadata_dir=base,
        model_type=str(raw_cfg.get("model_type", cfg.get("model_type", "unknown"))),
        architectures=archs,
        num_hidden_layers=num_layers,
        gdn_layers=gdn_layers,
        qsa_layers=qsa_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        has_shared_expert=axes.has_shared_expert,
        moe_intermediate_size=moe_i,
        shared_expert_intermediate_size=shared_i,
        hc_count=hc_count,
        ple_layer_ids=ple_layer_ids,
        ple_shards_detected=ple_shards_detected,
        mtp_layers=mtp_layers,
        kv_bytes_per_token_fp16=kv_bytes_per_token_fp16,
        gdn_state_bytes_fp32=gdn_state_bytes_fp32,
        total_index_entries=len(weight_map),
        shard_file_count=len(shard_files),
        index_total_size_bytes=total_size,
        backbone_tensor_count=len(backbone_names),
        vision_tensor_count=vision_cnt,
        ple_tensor_count=ple_cnt,
        mtp_tensor_count=mtp_cnt,
        loop_adapter_tensor_count=loop_cnt,
        stacked_expert_tensor_count=stacked_exp_cnt,
        per_expert_tensor_count=per_exp_cnt,
        shared_expert_tensor_count=shared_exp_cnt,
        hyper_connection_tensor_count=hc_cnt,
        qsa_indexer_tensor_count=qsa_idx_cnt,
        gdn_tensor_count=gdn_cnt,
        long_backbone_names=tuple(long_names),
        estimated_backbone_q3_bytes=q3_total,
        estimated_backbone_q4_bytes=q4_total,
        estimated_ple_bf16_bytes=est_ple_bf16,
        operator_axes=axes,
        loop_schedule=ls,
        tier_admission=tier_admission,
        tokenizer_present=(base / "tokenizer.json").is_file(),
        dense_ple_tensor_count=dense_ple_cnt,
        header_summary_verified=summary_verified,
        header_summary_notes=tuple(summary_notes),
    )


@dataclass(frozen=True)
class Qwen4RealHfSynthBundle:
    """Deterministic real-HF-schema `qwen4_exp` synthetic bundle produced via `pack_qwen4_checkpoint`."""

    amf_path: pathlib.Path
    ple_paths: list[pathlib.Path]
    mtp_path: pathlib.Path | None
    mtp_config_path: pathlib.Path | None
    vision_path: pathlib.Path | None
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    report: Qwen4PackReport


def _write_simple_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    hdr: dict[str, object] = {}
    cur = 0
    chunks: list[bytes] = []
    for nm, (dt, shp, raw) in tensors.items():
        hdr[nm] = {"dtype": dt, "shape": list(shp), "data_offsets": [cur, cur + len(raw)]}
        chunks.append(raw)
        cur += len(raw)
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for c in chunks:
            fh.write(c)


def _det_bf16_bytes(rng: random.Random, count: int, scale: float = 0.08) -> bytes:
    u16s = []
    for _ in range(count):
        v = (rng.random() * 2.0 - 1.0) * scale
        u32 = struct.unpack("<I", struct.pack("<f", v))[0]
        u16s.append((u32 >> 16) & 0xFFFF)
    return struct.pack(f"<{count}H", *u16s)


def _det_f32_bytes(rng: random.Random, count: int, lo: float = -0.5, hi: float = 0.5) -> bytes:
    return struct.pack(f"<{count}f", *(lo + rng.random() * (hi - lo) for _ in range(count)))


def generate_real_hf_qwen4_synth_bundle(
    output_dir: pathlib.Path,
    *,
    num_layers: int = 4,
    hidden_size: int = 128,
    vocab_size: int = 512,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    moe_intermediate_size: int = 64,
    shared_expert_intermediate_size: int = 64,
    hc_count: int = 4,
    hc_lowrank: int = 16,
    ple_shards: int = 2,
    ple_rows_per_shard: int = 64,
    ple_embed_dim: int = 64,
    include_mtp: bool = True,
    include_vision_sidecar: bool = False,
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    seed: int = 42,
) -> Qwen4RealHfSynthBundle:
    """Synthesize a real-HF schema `Qwen4ExpForConditionalGeneration` checkpoint and pack it to `.amf`.

    Exercises:
    - `hc_count=4` `attn_hyper_connection`, `mlp_hyper_connection`, and `hyper_connection_mixer` (`<= 47 B`)
    - 3:1 `linear_attention` (`in_proj_qkv` split + `in_proj_z/a/b` + `out_proj` + `F32` SSM) : `full_attention`
    - QSA `self_attn.q_proj` doubled output-gate (`output_gate_type='sigmoid'`) + `self_attn.indexer.*`
    - Dense PLE conditioning (`conv1d`, `key_proj`, `value_proj`, `norm_*`, `I64` multipliers/offsets/vocabs) in backbone + external `ngram_embedding.shard_*` in `SPARSE_TABLE` `.amf`
    - 3D stacked `mlp.experts.{gate_up_proj,down_proj}` + `shared_expert_gate.weight`
    - Standalone 1L real-HF `mtp.amf` + `mtp_config.json`
    """
    from aminfer_tools.synth import QWEN4_PRESETS, build_qwen3_tokenizer_dict

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    h = hidden_size
    n_heads = 4
    n_kv = 2
    hd = 32
    lin_kh = 2
    lin_kd = 32
    lin_vh = 4
    lin_vd = 32
    kq = lin_kh * lin_kd
    vd = lin_vh * lin_vd
    idx_heads = 2
    idx_kv = 1
    idx_hd = 32
    ple_layer_ids = [min(2, num_layers)] if num_layers >= 1 else [1]
    ple_src_l = ple_layer_ids[0] - 1

    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)]
    cfg = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen4_exp_text",
            "vocab_size": vocab_size,
            "hidden_size": h,
            "num_hidden_layers": num_layers,
            "layer_types": layer_types,
            "full_attention_interval": 4,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv,
            "head_dim": hd,
            "output_gate_type": "sigmoid",
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": lin_kh,
            "linear_key_head_dim": lin_kd,
            "linear_num_value_heads": lin_vh,
            "linear_value_head_dim": lin_vd,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "indexer_n_heads": idx_heads,
            "indexer_kv_heads": idx_kv,
            "indexer_head_dim": idx_hd,
            "indexer_compress_ratio": 4,
            "indexer_budget": 64,
            "hc_count": hc_count,
            "hc_lowrank": hc_lowrank,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_intermediate_size": moe_intermediate_size,
            "shared_expert_intermediate_size": shared_expert_intermediate_size,
            "ple_layer_ids": ple_layer_ids,
            "ple_embed_dim": ple_embed_dim,
            "split_ngram_parts": ple_shards,
            "mtp_num_hidden_layers": 1 if include_mtp else 0,
            "tie_word_embeddings": False,
            "rms_norm_eps": 1e-6,
        },
    }
    if include_vision_sidecar:
        cfg["vision_config"] = {
            "model_type": "qwen4_exp",
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 2,
            "out_hidden_size": h,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
        }

    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    p_lm = "model.language_model"
    tensors[f"{p_lm}.embed_tokens.weight"] = ("BF16", (vocab_size, h), _det_bf16_bytes(rng, vocab_size * h))
    tensors["lm_head.weight"] = ("BF16", (vocab_size, h), _det_bf16_bytes(rng, vocab_size * h))
    tensors[f"{p_lm}.norm.weight"] = ("BF16", (h,), _det_bf16_bytes(rng, h, 0.25))
    tensors[f"{p_lm}.hyper_connection_mixer.hc_norm.weight"] = ("BF16", (hc_count * h,), _det_bf16_bytes(rng, hc_count * h, 0.25))
    tensors[f"{p_lm}.hyper_connection_mixer.input_mix_weight_down.weight"] = ("BF16", (hc_lowrank, hc_count * h), _det_bf16_bytes(rng, hc_lowrank * hc_count * h))
    tensors[f"{p_lm}.hyper_connection_mixer.input_mix_weight_up.weight"] = ("BF16", (hc_count * h, hc_lowrank), _det_bf16_bytes(rng, hc_count * h * hc_lowrank))

    def _add_block(prefix: str, ltype: str, include_ple: bool) -> None:
        tensors[f"{prefix}.input_layernorm.weight"] = ("BF16", (h,), _det_bf16_bytes(rng, h, 0.25))
        tensors[f"{prefix}.post_attention_layernorm.weight"] = ("BF16", (h,), _det_bf16_bytes(rng, h, 0.25))
        for hc_tag in ("attn_hyper_connection", "mlp_hyper_connection"):
            tensors[f"{prefix}.{hc_tag}.hc_norm.weight"] = ("BF16", (hc_count * h,), _det_bf16_bytes(rng, hc_count * h, 0.25))
            tensors[f"{prefix}.{hc_tag}.input_mix_weight_down.weight"] = ("BF16", (hc_lowrank, hc_count * h), _det_bf16_bytes(rng, hc_lowrank * hc_count * h))
            tensors[f"{prefix}.{hc_tag}.input_mix_weight_up.weight"] = ("BF16", (hc_count * h, hc_lowrank), _det_bf16_bytes(rng, hc_count * h * hc_lowrank))
            tensors[f"{prefix}.{hc_tag}.block_inject_weight.weight"] = ("BF16", (hc_count, hc_count * h), _det_bf16_bytes(rng, hc_count * hc_count * h))
        if ltype == "linear_attention":
            conv_c = 2 * kq + vd
            tensors[f"{prefix}.linear_attn.in_proj_qkv.weight"] = ("BF16", (conv_c, h), _det_bf16_bytes(rng, conv_c * h))
            tensors[f"{prefix}.linear_attn.in_proj_z.weight"] = ("BF16", (vd, h), _det_bf16_bytes(rng, vd * h))
            tensors[f"{prefix}.linear_attn.in_proj_a.weight"] = ("BF16", (lin_vh, h), _det_bf16_bytes(rng, lin_vh * h))
            tensors[f"{prefix}.linear_attn.in_proj_b.weight"] = ("BF16", (lin_vh, h), _det_bf16_bytes(rng, lin_vh * h))
            tensors[f"{prefix}.linear_attn.out_proj.weight"] = ("BF16", (h, vd), _det_bf16_bytes(rng, h * vd))
            tensors[f"{prefix}.linear_attn.A_log"] = ("F32", (lin_vh,), _det_f32_bytes(rng, lin_vh, -2.0, -0.1))
            tensors[f"{prefix}.linear_attn.dt_bias"] = ("BF16", (lin_vh,), _det_bf16_bytes(rng, lin_vh, 0.25))
            tensors[f"{prefix}.linear_attn.conv1d.weight"] = ("BF16", (conv_c, 1, 4), _det_bf16_bytes(rng, conv_c * 4))
            tensors[f"{prefix}.linear_attn.norm.weight"] = ("BF16", (lin_vd,), _det_bf16_bytes(rng, lin_vd, 0.25))
        else:
            tensors[f"{prefix}.self_attn.q_proj.weight"] = ("BF16", (2 * n_heads * hd, h), _det_bf16_bytes(rng, 2 * n_heads * hd * h))
            tensors[f"{prefix}.self_attn.k_proj.weight"] = ("BF16", (n_kv * hd, h), _det_bf16_bytes(rng, n_kv * hd * h))
            tensors[f"{prefix}.self_attn.v_proj.weight"] = ("BF16", (n_kv * hd, h), _det_bf16_bytes(rng, n_kv * hd * h))
            tensors[f"{prefix}.self_attn.o_proj.weight"] = ("BF16", (h, n_heads * hd), _det_bf16_bytes(rng, h * n_heads * hd))
            tensors[f"{prefix}.self_attn.q_norm.weight"] = ("BF16", (hd,), _det_bf16_bytes(rng, hd, 0.25))
            tensors[f"{prefix}.self_attn.k_norm.weight"] = ("BF16", (hd,), _det_bf16_bytes(rng, hd, 0.25))
            tensors[f"{prefix}.self_attn.indexer.index_qk_proj.weight"] = (
                "BF16",
                ((idx_heads + idx_kv) * idx_hd, h),
                _det_bf16_bytes(rng, (idx_heads + idx_kv) * idx_hd * h),
            )
            tensors[f"{prefix}.self_attn.indexer.q_layernorm.weight"] = ("BF16", (idx_hd,), _det_bf16_bytes(rng, idx_hd, 0.25))
            tensors[f"{prefix}.self_attn.indexer.k_layernorm.weight"] = ("BF16", (idx_hd,), _det_bf16_bytes(rng, idx_hd, 0.25))
        if include_ple:
            tensors[f"{prefix}.ple.conv1d.weight"] = ("BF16", (hc_count * h, 1, 4), _det_bf16_bytes(rng, hc_count * h * 4))
            tensors[f"{prefix}.ple.key_proj.weight"] = ("BF16", (hc_count * h, h), _det_bf16_bytes(rng, hc_count * h * h))
            tensors[f"{prefix}.ple.value_proj.weight"] = ("BF16", (h, h), _det_bf16_bytes(rng, h * h))
            tensors[f"{prefix}.ple.norm_conv.weight"] = ("BF16", (hc_count * h,), _det_bf16_bytes(rng, hc_count * h, 0.25))
            tensors[f"{prefix}.ple.norm_key.weight"] = ("BF16", (hc_count * h,), _det_bf16_bytes(rng, hc_count * h, 0.25))
            tensors[f"{prefix}.ple.norm_query.weight"] = ("BF16", (hc_count * h,), _det_bf16_bytes(rng, hc_count * h, 0.25))
            tensors[f"{prefix}.ple.ple_embedding.layer_multipliers"] = ("I64", (3,), struct.pack("<3q", 1000003, 2000003, 3000017))
            tensors[f"{prefix}.ple.ple_embedding.ngram_heads_offsets"] = ("I64", (4,), struct.pack("<4q", 0, 16, 32, 48))
            tensors[f"{prefix}.ple.ple_embedding.ngram_heads_vocab_sizes"] = ("I64", (4,), struct.pack("<4q", 16, 16, 16, 16))
            for s_i in range(ple_shards):
                tensors[f"{prefix}.ple.ple_embedding.ngram_embedding.shard_{s_i}.weight"] = (
                    "BF16",
                    (ple_rows_per_shard, ple_embed_dim),
                    _det_bf16_bytes(rng, ple_rows_per_shard * ple_embed_dim),
                )
        tensors[f"{prefix}.mlp.gate.weight"] = ("BF16", (num_experts, h), _det_bf16_bytes(rng, num_experts * h))
        tensors[f"{prefix}.mlp.experts.gate_up_proj"] = ("BF16", (num_experts, 2 * moe_intermediate_size, h), _det_bf16_bytes(rng, num_experts * 2 * moe_intermediate_size * h))
        tensors[f"{prefix}.mlp.experts.down_proj"] = ("BF16", (num_experts, h, moe_intermediate_size), _det_bf16_bytes(rng, num_experts * h * moe_intermediate_size))
        tensors[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = ("BF16", (shared_expert_intermediate_size, h), _det_bf16_bytes(rng, shared_expert_intermediate_size * h))
        tensors[f"{prefix}.mlp.shared_expert.up_proj.weight"] = ("BF16", (shared_expert_intermediate_size, h), _det_bf16_bytes(rng, shared_expert_intermediate_size * h))
        tensors[f"{prefix}.mlp.shared_expert.down_proj.weight"] = ("BF16", (h, shared_expert_intermediate_size), _det_bf16_bytes(rng, h * shared_expert_intermediate_size))
        tensors[f"{prefix}.mlp.shared_expert_gate.weight"] = ("BF16", (1, h), _det_bf16_bytes(rng, h))

    for l_idx in range(num_layers):
        _add_block(f"{p_lm}.layers.{l_idx}", layer_types[l_idx], include_ple=(l_idx == ple_src_l))

    if include_mtp:
        tensors["mtp.fc_embedding.weight"] = ("BF16", (h, h), _det_bf16_bytes(rng, h * h))
        tensors["mtp.fc_hidden.weight"] = ("BF16", (h, h), _det_bf16_bytes(rng, h * h))
        tensors["mtp.pre_fc_norm_embedding.weight"] = ("BF16", (h,), _det_bf16_bytes(rng, h, 0.25))
        tensors["mtp.pre_fc_norm_hidden.weight"] = ("BF16", (hc_count * h,), _det_bf16_bytes(rng, hc_count * h, 0.25))
        tensors["mtp.hyper_connection_mixer.hc_norm.weight"] = ("BF16", (hc_count * h,), _det_bf16_bytes(rng, hc_count * h, 0.25))
        tensors["mtp.hyper_connection_mixer.input_mix_weight_down.weight"] = ("BF16", (hc_lowrank, hc_count * h), _det_bf16_bytes(rng, hc_lowrank * hc_count * h))
        tensors["mtp.hyper_connection_mixer.input_mix_weight_up.weight"] = ("BF16", (hc_count * h, hc_lowrank), _det_bf16_bytes(rng, hc_count * h * hc_lowrank))
        _add_block("mtp.layers.0", "full_attention", include_ple=False)

    if include_vision_sidecar:
        tensors["model.visual.patch_embed.proj.weight"] = ("BF16", (64, 3, 2, 16, 16), _det_bf16_bytes(rng, 64 * 3 * 2 * 16 * 16))
        tensors["model.visual.blocks.0.norm1.weight"] = ("BF16", (64,), _det_bf16_bytes(rng, 64, 0.25))
        tensors["model.visual.blocks.0.attn.qkv.weight"] = ("BF16", (192, 64), _det_bf16_bytes(rng, 192 * 64))
        tensors["model.visual.blocks.0.attn.proj.weight"] = ("BF16", (64, 64), _det_bf16_bytes(rng, 64 * 64))
        tensors["model.visual.blocks.0.norm2.weight"] = ("BF16", (64,), _det_bf16_bytes(rng, 64, 0.25))
        tensors["model.visual.blocks.0.mlp.linear_fc1.weight"] = ("BF16", (128, 64), _det_bf16_bytes(rng, 128 * 64))
        tensors["model.visual.blocks.0.mlp.linear_fc2.weight"] = ("BF16", (64, 128), _det_bf16_bytes(rng, 64 * 128))
        tensors["model.visual.merger.norm.weight"] = ("BF16", (64,), _det_bf16_bytes(rng, 64, 0.25))
        tensors["model.visual.merger.linear_fc1.weight"] = ("BF16", (256, 256), _det_bf16_bytes(rng, 256 * 256))
        tensors["model.visual.merger.linear_fc2.weight"] = ("BF16", (h, 256), _det_bf16_bytes(rng, h * 256))

    tok_dict = build_qwen3_tokenizer_dict(QWEN4_PRESETS["tiny-qwen4"].with_overrides(vocab_size=vocab_size))  # type: ignore[arg-type]
    with tempfile.TemporaryDirectory() as tmp_sf:
        sf_dir = pathlib.Path(tmp_sf)
        _write_simple_safetensors(sf_dir / "model.safetensors", tensors)
        (sf_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        (sf_dir / "tokenizer.json").write_text(json.dumps(tok_dict) + "\n", encoding="utf-8")
        amf_p = out / "model.amf"
        ple_base = out / "ple.amf"
        mtp_p = (out / "mtp.amf") if include_mtp else None
        vis_p = (out / "vision.amf") if include_vision_sidecar else None
        cfg_p = out / "config.json"
        rep = pack_qwen4_checkpoint(
            sf_dir,
            amf_p,
            model_id="qwen4-real-hf-synth",
            quant_profile=quant_profile,
            target_bits=target_bits,
            cold_experts=[(l_i, num_experts - 1) for l_i in range(num_layers)],
            ple_output=ple_base,
            ple_shards=ple_shards,
            mtp_output=mtp_p,
            vision_sidecar=vis_p,
            config_output=cfg_p,
        )
    tok_out = out / "tokenizer.json"
    tok_out.write_text(json.dumps(tok_dict, indent=2) + "\n", encoding="utf-8")
    return Qwen4RealHfSynthBundle(
        amf_path=rep.amf_path,
        ple_paths=rep.ple_shard_paths,
        mtp_path=rep.mtp_amf_path,
        mtp_config_path=rep.mtp_config_path,
        vision_path=rep.vision_amf_path,
        config_path=cfg_p,
        tokenizer_path=tok_out,
        report=rep,
    )
