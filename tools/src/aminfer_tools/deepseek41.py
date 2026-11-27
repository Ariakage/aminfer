"""W11 DeepSeek-V4.1-Flash-1017 (`deepseek41`) offline converter, synth packer, and expert shard partitioner.

Pure standard-library Python implementation (ADR-0001, ADR-0015, ADR-0016):

- `DeepSeek41Config` and presets (`tiny-deepseek41`, `scale-deepseek41-512e`,
  `full-deepseek41-552b`) encoding the Causal Encoder-Decoder (CED) backbone,
  MLA + CSA2 (`full` / `reindex` / `reuse`) + FP4 (`E2M1` + `E4M3`, `890 B/tok`)
  KV + SWA bounded replay, Single-Pass Mega-mHC (`4` streams), 196B Engram
  conditional memory (`TensorKind.SPARSE_TABLE`, `[rows, 2 * hidden_size]`),
  DSpark semi-autoregressive proposer, and `expert_parallel` metadata (`X1`-`X3`).
- `generate_synth_deepseek41_bundle`: deterministic 16 KiB-aligned `.amf` v0
  backbone + sharded Engram `.amf` + optional `dspark.amf` + `config.json` +
  BPE `tokenizer.json`.
- `partition_deepseek41_amf_shards`: streaming multi-rank expert partitioner
  generating per-rank `.amf` shards and `cluster_manifest.json` carrying
  `ExpertSource.thunderbolt_peer` ownership metadata.
- `convert_deepseek41_checkpoint`: streaming multi-shard safetensors importer
  (`pack-deepseek41`) supporting `q3_affine` / `q4_affine` / `bf16` and
  `--quant-profile gsq-rco`, with explicit `ADR-0015` Tier 3 hardware-tier
  wired-memory validation rejecting non-viable 24 GB M4 full-552B local plans.
"""

from __future__ import annotations

import array
import json
import math
import pathlib
import random
import re
import struct
import tempfile
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    AMF_MAGIC,
    AMF_VERSION,
    CHECKSUM_CRC32_IEEE,
    DTYPE_MODES,
    ENTRY_STRUCT,
    HEADER_STRUCT,
    NO_EXPERT,
    NO_LAYER,
    QUANT_PROFILES,
    SME2_PACK_MODES,
    AmfReader,
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
    _load_tie_flag,
    _normalize_name,
    _pack_directory,
    _pack_header,
    _PlannedEntry,
    _read_exact,
    _ShardReaders,
    _ShardTensor,
    _stream_entry,
    _strip_model_prefix,
    dequantize_mxfp8_row_f32,
    encode_reserved0,
    f16_bytes_to_bf16_bytes,
    f32_to_bf16_bytes,
    interleave_scale_bias_f16,
    VISION_BLOCK_DTYPES,
    extract_vision_config_summary,
    gsq_convert_with_vision,
    is_vision_block_linear,
    is_vision_pinned_float,
    is_vision_tensor,
    load_cold_experts_from_heat_profile,
    pack_mlx_affine_3bit,
    pack_mlx_affine_4bit,
    pack_vision_tower_amf,
    plan_amf_entries,
    resolve_checkpoint,
    resolve_vision_mode,
)
from aminfer_tools.qwen4_exp import _slice_row_byte_range, _split_row_ranges
from aminfer_tools.synth import (
    _stream_zeros,
    _TensorPlan,
    _write_planned_amf,
    build_qwen3_tokenizer_dict,
)

VALID_CSA2_MODES = ("full", "reindex", "reuse")
VALID_DEEPSEEK41_DTYPES = ("q3_affine", "q4_affine", "bf16")
VALID_ENGRAM_DTYPES = ("f16", "bf16", "q4_affine", "q3_affine", "q2_affine")
VALID_PARTITION_STRATEGIES = ("contiguous_by_rank", "round_robin")
VALID_TARGET_TIERS = ("auto", "24gb-m4", "128gb-max", "256gb-ultra", "512gb-ultra", "cluster")

ENGRAM_MIN_HEADS = 2
ENGRAM_MAX_HEADS = 8
ENGRAM_MIN_ORDER = 2
ENGRAM_MAX_ORDER = 5
MAX_CLUSTER_WORLD_SIZE = 16

# Wired memory ceilings matching ADR-0012 / ADR-0015 hardware tiers (bytes).
TIER_WIRED_BUDGET_BYTES: dict[str, int] = {
    "24gb-m4": int(18.186 * (1024**3)),
    "128gb-max": int(96.0 * (1024**3)),
    "256gb-ultra": int(200.0 * (1024**3)),
    "512gb-ultra": int(400.0 * (1024**3)),
}

# Soft safety cap preventing accidental multi-hundred-GiB synthetic file generation.
DEFAULT_MAX_SYNTH_BYTES = 2 * (1024**3)

_ENGRAM_LAYER_RE = re.compile(r"(?:^|\.)(?:engram\.layers|layers)\.(\d+)(?:\.|$)")
_ENGRAM_SHARD_RE = re.compile(r"\.shard[._](\d+)(?:\.|$)")
_DSPARK_PREFIXES = (
    "dspark.",
    "model.dspark.",
    "language_model.dspark.",
    "language_model.model.dspark.",
    "mtp.",
    "model.mtp.",
    "language_model.mtp.",
    "language_model.model.mtp.",
)
_DEEPSEEK41_MODEL_TYPES = ("deepseek41", "deepseek_v41", "deepseek_v41_text", "deepseek_v3", "deepseek_v32")


def _effective_deepseek_text_config(src_config: dict[str, object] | None) -> dict[str, object]:
    """Flatten HF `config.json` by merging nested `text_config` over top-level fields."""
    if not isinstance(src_config, dict):
        return {}
    merged = dict(src_config)
    text_cfg = src_config.get("text_config")
    if isinstance(text_cfg, dict):
        merged.update(text_cfg)
    return merged


def _infer_enc_dec_layers(cfg: dict[str, object], total_layers: int) -> tuple[int, int]:
    """Infer `(num_encoder_layers, num_decoder_layers)` from explicit fields or `compress_ratios`."""
    if cfg.get("num_encoder_layers") is not None and cfg.get("num_decoder_layers") is not None:
        return (int(cfg["num_encoder_layers"]), int(cfg["num_decoder_layers"]))  # type: ignore[arg-type]
    ratios = cfg.get("compress_ratios")
    if isinstance(ratios, list) and len(ratios) >= total_layers and total_layers >= 2:
        main_ratios = [int(x) for x in ratios[:total_layers]]
        dec_l = sum(1 for r in main_ratios if r == 1)
        enc_l = total_layers - dec_l
        if enc_l >= 1 and dec_l >= 1:
            return (enc_l, dec_l)
    enc_l = max(1, total_layers // 2)
    dec_l = max(1, total_layers - enc_l)
    return (enc_l, dec_l)


class HardwareTierError(ValueError):
    """Raised when a `deepseek41` wired memory plan violates ADR-0015 hardware-tier ceilings."""


_DENSE_ENGRAM_MARKERS = ("wkv", "q_weight", "k_weight", "key_projs", "value_proj", "norm", "hc_scales", "unified_pre_gamma")


def is_engram_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` is an external Engram `SPARSE_TABLE` lookup tensor (never dense wkv/q_weight/k_weight)."""
    norm = _strip_model_prefix(raw_name).lower()
    if norm.startswith(("engram.", "sparse_table.")):
        return True
    if ".engram." in norm:
        if ".embed." in norm or ".lookup" in norm:
            return True
        return not any(tok in norm for tok in _DENSE_ENGRAM_MARKERS)
    return False


def is_dspark_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` belongs to the DSpark semi-autoregressive proposer head."""
    return raw_name.lower().startswith(_DSPARK_PREFIXES)


def strip_dspark_prefix(raw_name: str) -> str:
    lower = raw_name.lower()
    for prefix in ("mtp.", "model.mtp.", "language_model.mtp.", "language_model.model.mtp."):
        if lower.startswith(prefix):
            return "mtp." + raw_name[len(prefix) :]
    for prefix in ("dspark.", "model.dspark.", "language_model.dspark.", "language_model.model.dspark."):
        if lower.startswith(prefix):
            return "dspark." + raw_name[len(prefix) :]
    return _strip_model_prefix(raw_name)


def normalize_engram_name(raw_name: str, *, shard_idx: int | None = None, quantized: bool = False) -> str:
    """Normalize an Engram tensor name to `engram.layers.{l}[.shard.{s}][.weight]` (<= 47 bytes)."""
    norm = _strip_model_prefix(raw_name)
    m = _ENGRAM_LAYER_RE.search(norm)
    layer = int(m.group(1)) if m else 0
    suffix = "" if quantized else ".weight"
    if shard_idx is not None:
        return f"engram.layers.{layer}.shard.{shard_idx}{suffix}"
    sm = _ENGRAM_SHARD_RE.search(norm)
    if sm is not None:
        return f"engram.layers.{layer}.shard.{int(sm.group(1))}{suffix}"
    return f"engram.layers.{layer}{suffix}"


def engram_shard_paths(base_path: pathlib.Path, num_shards: int) -> list[pathlib.Path]:
    """Return deterministic output path(s) for `num_shards` Engram `.amf` shard files."""
    if num_shards < 1:
        raise ValueError(f"engram num_shards must be >= 1, got {num_shards}")
    base = pathlib.Path(base_path)
    if num_shards == 1:
        return [base]
    stem = base.stem
    suffix = base.suffix or ".amf"
    return [base.with_name(f"{stem}-shard-{s:02d}-of-{num_shards:02d}{suffix}") for s in range(num_shards)]


def _default_csa2_pattern(num_encoder_layers: int, num_decoder_layers: int) -> tuple[str, ...]:
    """Generate a valid CSA2 (`full`/`reindex`/`reuse`) pattern for CED encoder + decoder."""
    cycle = ("full", "reindex", "reuse")
    enc = tuple("full" if i == 0 else cycle[i % len(cycle)] for i in range(num_encoder_layers))
    dec = tuple("full" if i == 0 else cycle[i % len(cycle)] for i in range(num_decoder_layers))
    return enc + dec


@dataclass(frozen=True)
class DeepSeek41Config:
    """Canonical configuration for `DeepSeek-V4.1-Flash-1017` (`deepseek41`, ADR-0015)."""

    model_type: str = "deepseek41"
    model_id: str = "deepseek-v4.1-flash-1017-synth"
    backbone_kind: str = "causal_enc_dec"
    num_encoder_layers: int = 20
    num_decoder_layers: int = 20
    hidden_size: int = 7168
    vocab_size: int = 129280
    num_attention_heads: int = 128
    encoder_moe_top_k: int = 8
    decoder_moe_top_k: int = 16
    num_experts: int = 512
    has_shared_expert: bool = True
    moe_intermediate_size: int = 1280
    shared_expert_intermediate_size: int = 2560
    kv_lora_rank: int = 1536
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    csa2_modes: tuple[str, ...] = ()
    fp4_kv: bool = True
    # Compressed RoPE tail bytes per token for FP4 MLA KV (26 B gives exact 890 B/tok at rank 1536).
    fp4_rope_tail_bytes: int = 26
    swa_window: int = 4096
    residual_kind: str = "mhc_single_pass"
    mhc_num_streams: int = 4
    memory_lookup_kind: str = "engram"
    engram_num_heads: int = 3
    engram_orders: tuple[int, ...] = (2, 3, 4)
    engram_num_rows: int = 256
    engram_shards: int = 2
    engram_dtype: str = "q4_affine"
    include_dspark: bool = True
    dspark_block_width: int = 4
    dspark_top_k_per_pos: int = 2
    dspark_max_tree_nodes: int = 32
    dtype: str = "q3_affine"
    group_size: int = 64
    tie_word_embeddings: bool = False
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 1_048_576
    rms_norm_eps: float = 1e-6
    world_size: int = 1
    rank: int = 0
    owned_expert_ids: tuple[int, ...] = ()

    @property
    def total_layers(self) -> int:
        return self.num_encoder_layers + self.num_decoder_layers

    @property
    def quant_dtype(self) -> DType:
        if self.dtype == "q3_affine":
            return DType.Q3_AFFINE
        if self.dtype == "q4_affine":
            return DType.Q4_AFFINE
        if self.dtype == "bf16":
            return DType.BF16
        raise ValueError(f"unsupported deepseek41 dtype: {self.dtype!r}")

    @property
    def quant_bits(self) -> int:
        return {"q3_affine": 3, "q4_affine": 4, "bf16": 0}[self.dtype]

    @property
    def engram_amf_dtype(self) -> DType:
        return {
            "f16": DType.F16,
            "bf16": DType.BF16,
            "q4_affine": DType.Q4_AFFINE,
            "q3_affine": DType.Q3_AFFINE,
            "q2_affine": DType.Q2_AFFINE,
        }[self.engram_dtype]

    @property
    def engram_quant_bits(self) -> int:
        return {"f16": 0, "bf16": 0, "q4_affine": 4, "q3_affine": 3, "q2_affine": 2}[self.engram_dtype]

    def effective_csa2_modes(self) -> tuple[str, ...]:
        if self.csa2_modes:
            return tuple(self.csa2_modes)
        return _default_csa2_pattern(self.num_encoder_layers, self.num_decoder_layers)

    def effective_owned_experts(self) -> tuple[int, ...]:
        if self.owned_expert_ids:
            return tuple(self.owned_expert_ids)
        return tuple(range(self.num_experts))

    def special_token_ids(self) -> dict[str, int]:
        base = self.vocab_size - 5
        return {
            "<|endoftext|>": base,
            "<|im_start|>": base + 1,
            "<|im_end|>": base + 2,
            "<|think|>": base + 3,
            "</think>": base + 4,
        }

    def fp4_kv_bytes_per_token(self) -> int:
        """Return compressed FP4 (`E2M1` + `E4M3`/16-ch scale + RoPE tail) KV bytes/token.

        For full `deepseek41` (`kv_lora_rank=1536`, `fp4_rope_tail_bytes=26`):
        `1536/2 (768 B E2M1) + 1536/16 (96 B E4M3) + 26 B = 890 B/token`.
        """
        if not self.fp4_kv:
            return (self.kv_lora_rank + self.qk_rope_head_dim) * 2
        e2m1_bytes = self.kv_lora_rank // 2
        e4m3_scale_bytes = self.kv_lora_rank // 16
        rope_bytes = self.fp4_rope_tail_bytes if self.fp4_rope_tail_bytes > 0 else self.qk_rope_head_dim * 2
        return e2m1_bytes + e4m3_scale_bytes + rope_bytes

    def estimated_backbone_bytes(self) -> tuple[int, int, int]:
        """Return `(total_bytes, non_expert_bytes, routed_expert_bytes)` for this geometry."""
        h = self.hidden_size
        n_layers = self.total_layers
        n_heads = self.num_attention_heads
        q_dim = n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        kv_b_out = n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        v_tot = n_heads * self.v_head_dim
        kv_a_out = self.kv_lora_rank + self.qk_rope_head_dim

        # Effective bytes per weight element including group-64 affine scale+bias (4 B / 64 elems).
        if self.dtype == "q3_affine":
            bpw = (24 + 4) / 64.0  # 0.4375 B/elem (3.5 bits)
        elif self.dtype == "q4_affine":
            bpw = (32 + 4) / 64.0  # 0.5625 B/elem (4.5 bits)
        else:
            bpw = 2.0

        # Per-layer non-expert parameters: MLA projections + CSA2 indexer + mHC + router + shared expert.
        attn_elems = (
            self.q_lora_rank * h
            + q_dim * self.q_lora_rank
            + kv_a_out * h
            + kv_b_out * self.kv_lora_rank
            + h * v_tot
            + self.kv_lora_rank * h  # CSA2 indexer summary projection
        )
        mhc_elems = self.mhc_num_streams * h * 2 + self.mhc_num_streams * self.mhc_num_streams
        router_elems = self.num_experts * h
        shared_elems = (3 * self.shared_expert_intermediate_size * h + h) if self.has_shared_expert else 0
        layer_non_expert = int((attn_elems + shared_elems) * bpw) + (mhc_elems + router_elems + 4 * h) * 2

        # Encoder->decoder global KV projector + embeddings + lm_head.
        enc_dec_elems = kv_a_out * h + kv_b_out * self.kv_lora_rank
        embed_bytes = self.vocab_size * h * 2
        lm_head_bytes = 0 if self.tie_word_embeddings else int(self.vocab_size * h * bpw)
        non_expert_bytes = n_layers * layer_non_expert + int(enc_dec_elems * bpw) + embed_bytes + lm_head_bytes

        # Routed MoE experts across all encoder + decoder layers.
        per_expert_elems = 3 * self.moe_intermediate_size * h
        routed_expert_bytes = int(n_layers * self.num_experts * per_expert_elems * bpw)
        return (non_expert_bytes + routed_expert_bytes, non_expert_bytes, routed_expert_bytes)

    def with_overrides(self, **kwargs: object) -> DeepSeek41Config:
        return replace(self, **kwargs)

    def validate(self) -> None:
        if self.model_type != "deepseek41":
            raise ValueError(f"model_type must be 'deepseek41', got {self.model_type!r}")
        if self.backbone_kind != "causal_enc_dec":
            raise ValueError(f"backbone_kind must be 'causal_enc_dec', got {self.backbone_kind!r}")
        if self.residual_kind != "mhc_single_pass":
            raise ValueError(f"residual_kind must be 'mhc_single_pass', got {self.residual_kind!r}")
        if self.memory_lookup_kind != "engram":
            raise ValueError(f"memory_lookup_kind must be 'engram', got {self.memory_lookup_kind!r}")
        if self.num_encoder_layers < 1 or self.num_decoder_layers < 1:
            raise ValueError("num_encoder_layers and num_decoder_layers must both be >= 1")
        if self.dtype not in VALID_DEEPSEEK41_DTYPES:
            raise ValueError(f"dtype must be one of {VALID_DEEPSEEK41_DTYPES}, got {self.dtype!r}")
        if self.engram_dtype not in VALID_ENGRAM_DTYPES:
            raise ValueError(f"engram_dtype must be one of {VALID_ENGRAM_DTYPES}, got {self.engram_dtype!r}")
        if self.group_size != 64:
            raise ValueError(f"aminfer v0 requires group_size=64, got {self.group_size}")
        if self.hidden_size <= 0 or self.hidden_size % 64 != 0:
            raise ValueError(f"hidden_size must be a positive multiple of 64, got {self.hidden_size}")
        if self.kv_lora_rank <= 0 or self.kv_lora_rank % 32 != 0:
            raise ValueError(f"kv_lora_rank must be a positive multiple of 32, got {self.kv_lora_rank}")
        if self.q_lora_rank <= 0 or self.q_lora_rank % 64 != 0:
            raise ValueError(f"q_lora_rank must be a positive multiple of 64, got {self.q_lora_rank}")
        if self.moe_intermediate_size <= 0 or self.moe_intermediate_size % 64 != 0:
            raise ValueError(
                f"moe_intermediate_size must be a positive multiple of 64, got {self.moe_intermediate_size}"
            )
        if self.has_shared_expert and (
            self.shared_expert_intermediate_size <= 0 or self.shared_expert_intermediate_size % 64 != 0
        ):
            raise ValueError("shared_expert_intermediate_size must be a positive multiple of 64")
        if self.num_experts < 1 or self.num_experts > 1024:
            raise ValueError(f"num_experts must be in 1..1024, got {self.num_experts}")
        if not (1 <= self.encoder_moe_top_k <= self.num_experts):
            raise ValueError(f"encoder_moe_top_k must be in 1..{self.num_experts}, got {self.encoder_moe_top_k}")
        if not (1 <= self.decoder_moe_top_k <= self.num_experts):
            raise ValueError(f"decoder_moe_top_k must be in 1..{self.num_experts}, got {self.decoder_moe_top_k}")
        if not (2 <= self.mhc_num_streams <= 8):
            raise ValueError(f"mhc_num_streams must be in 2..8, got {self.mhc_num_streams}")
        if not (ENGRAM_MIN_HEADS <= self.engram_num_heads <= ENGRAM_MAX_HEADS):
            raise ValueError(
                f"engram_num_heads must be in {ENGRAM_MIN_HEADS}..{ENGRAM_MAX_HEADS}, got {self.engram_num_heads}"
            )
        if len(self.engram_orders) != self.engram_num_heads:
            raise ValueError(
                f"engram_orders length ({len(self.engram_orders)}) must match engram_num_heads ({self.engram_num_heads})"
            )
        for ord_val in self.engram_orders:
            if not (ENGRAM_MIN_ORDER <= ord_val <= ENGRAM_MAX_ORDER):
                raise ValueError(
                    f"each engram order must be in {ENGRAM_MIN_ORDER}..{ENGRAM_MAX_ORDER}, got {ord_val}"
                )
        if self.engram_shards < 1 or self.engram_num_rows < self.engram_shards:
            raise ValueError("engram_num_rows must be >= engram_shards >= 1")
        if self.vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256, got {self.vocab_size}")
        if self.swa_window < 0:
            raise ValueError(f"swa_window must be >= 0, got {self.swa_window}")

        modes = self.effective_csa2_modes()
        if len(modes) != self.total_layers:
            raise ValueError(f"csa2_modes length ({len(modes)}) must equal total_layers ({self.total_layers})")
        for m in modes:
            if m not in VALID_CSA2_MODES:
                raise ValueError(f"invalid CSA2 mode {m!r}; expected one of {VALID_CSA2_MODES}")
        if modes[0] != "full" or modes[self.num_encoder_layers] != "full":
            raise ValueError("encoder layer 0 and decoder layer 0 must use CSA2 mode 'full' as shared_kv_owner")

        if not (1 <= self.world_size <= MAX_CLUSTER_WORLD_SIZE):
            raise ValueError(f"world_size must be in 1..{MAX_CLUSTER_WORLD_SIZE}, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank must be in 0..{self.world_size - 1}, got {self.rank}")
        if len(self.model_id.encode("utf-8")) > 63:
            raise ValueError(f"model_id exceeds 63 UTF-8 bytes: {self.model_id!r}")

    def to_config_dict(
        self,
        *,
        engram_files: Sequence[str] = (),
        dspark_file: str | None = None,
        cluster_manifest_file: str | None = None,
    ) -> dict[str, object]:
        modes = list(self.effective_csa2_modes())
        owned = list(self.effective_owned_experts())
        total_b, non_exp_b, exp_b = self.estimated_backbone_bytes()
        return {
            "architectures": ["DeepSeekV41FlashForCausalLM"],
            "model_type": self.model_type,
            "backbone_kind": self.backbone_kind,
            "num_hidden_layers": self.total_layers,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "encoder_moe_top_k": self.encoder_moe_top_k,
            "decoder_moe_top_k": self.decoder_moe_top_k,
            "num_experts": self.num_experts,
            "has_shared_expert": self.has_shared_expert,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "kv_lora_rank": self.kv_lora_rank,
            "q_lora_rank": self.q_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "csa2_modes": modes,
            "fp4_kv": self.fp4_kv,
            "fp4_kv_bytes_per_token": self.fp4_kv_bytes_per_token(),
            "swa_window": self.swa_window,
            "residual_kind": self.residual_kind,
            "mhc_num_streams": self.mhc_num_streams,
            "memory_lookup_kind": self.memory_lookup_kind,
            "engram_num_heads": self.engram_num_heads,
            "engram_orders": list(self.engram_orders),
            "engram_num_rows": self.engram_num_rows,
            "engram_shards": self.engram_shards,
            "engram_dtype": self.engram_dtype,
            "engram_files": list(engram_files),
            "dspark": {
                "enabled": self.include_dspark,
                "amf_file": dspark_file,
                "block_width": self.dspark_block_width,
                "top_k_per_pos": self.dspark_top_k_per_pos,
                "max_tree_nodes": self.dspark_max_tree_nodes,
            },
            "operator_axes": {
                "backbone_kind": self.backbone_kind,
                "attention_kind": "mla_csa2",
                "layer_attention_kinds": ["mla_csa2"] * self.total_layers,
                "ffn_kind": "moe_swiglu",
                "has_shared_expert": self.has_shared_expert,
                "residual_kind": self.residual_kind,
                "residual_branches": self.mhc_num_streams,
                "memory_lookup_kind": self.memory_lookup_kind,
                "loop_schedule_kind": "single_pass",
            },
            "expert_parallel": {
                "world_size": self.world_size,
                "rank": self.rank,
                "owned_expert_ids": owned,
                "local_expert_source": "local_wired_ram",
                "remote_expert_source": "thunderbolt_peer",
                "cluster_manifest": cluster_manifest_file,
            },
            "quantization": {
                "dtype": self.dtype,
                "group_size": self.group_size,
                "bits": self.quant_bits,
            },
            "hardware_tier": {
                "tier": "tier3_scale_up_or_cluster",
                "estimated_backbone_bytes": total_b,
                "estimated_non_expert_bytes": non_exp_b,
                "estimated_routed_expert_bytes": exp_b,
            },
            "tie_word_embeddings": self.tie_word_embeddings,
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
        }


DEEPSEEK41_PRESETS: dict[str, DeepSeek41Config] = {
    "tiny-deepseek41": DeepSeek41Config(
        model_id="deepseek41-tiny-synth",
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_size=256,
        vocab_size=2048,
        num_attention_heads=4,
        encoder_moe_top_k=2,
        decoder_moe_top_k=4,
        num_experts=8,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        csa2_modes=("full", "reuse", "full", "reindex"),
        fp4_kv=True,
        fp4_rope_tail_bytes=0,
        swa_window=64,
        mhc_num_streams=4,
        engram_num_heads=3,
        engram_orders=(2, 3, 4),
        engram_num_rows=128,
        engram_shards=2,
        engram_dtype="q4_affine",
        dtype="q4_affine",
        max_position_embeddings=4096,
    ),
    "scale-deepseek41-512e": DeepSeek41Config(
        model_id="deepseek41-scale-512e-synth",
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_size=256,
        vocab_size=4096,
        num_attention_heads=4,
        encoder_moe_top_k=4,
        decoder_moe_top_k=8,
        num_experts=512,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        kv_lora_rank=128,
        q_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        csa2_modes=("full", "reindex", "full", "reuse"),
        fp4_kv=True,
        fp4_rope_tail_bytes=0,
        swa_window=128,
        mhc_num_streams=4,
        engram_num_heads=4,
        engram_orders=(2, 3, 4, 5),
        engram_num_rows=256,
        engram_shards=2,
        engram_dtype="q4_affine",
        dtype="q3_affine",
        max_position_embeddings=8192,
    ),
    "full-deepseek41-552b": DeepSeek41Config(
        model_id="deepseek-v4.1-flash-1017",
        num_encoder_layers=20,
        num_decoder_layers=20,
        hidden_size=7168,
        vocab_size=129280,
        num_attention_heads=128,
        encoder_moe_top_k=8,
        decoder_moe_top_k=16,
        num_experts=512,
        moe_intermediate_size=1280,
        shared_expert_intermediate_size=2560,
        kv_lora_rank=1536,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        csa2_modes=_default_csa2_pattern(20, 20),
        fp4_kv=True,
        fp4_rope_tail_bytes=26,  # 768 B E2M1 + 96 B E4M3 + 26 B RoPE = exact 890 B/tok
        swa_window=4096,
        mhc_num_streams=4,
        engram_num_heads=4,
        engram_orders=(2, 3, 4, 5),
        engram_num_rows=1_048_576,
        engram_shards=16,
        engram_dtype="q4_affine",
        dtype="q3_affine",
        max_position_embeddings=1_048_576,
    ),
}


@dataclass(frozen=True)
class HardwareTierCheck:
    target_tier: str
    world_size: int
    total_wired_bytes: int
    per_rank_wired_bytes: int
    per_rank_budget_bytes: int
    fits: bool
    recommendation: str


def evaluate_hardware_tier(
    total_wired_bytes: int,
    *,
    target_tier: str = "auto",
    host_ram_gib: float | None = None,
    world_size: int = 1,
    non_expert_bytes: int | None = None,
    routed_expert_bytes: int | None = None,
    allow_override: bool = False,
) -> HardwareTierCheck:
    """Validate a `deepseek41` wired memory plan against ADR-0015 Tier 3 limits.

    Rejects non-viable 24 GB M4 local full-552B wiring plans with a clear
    diagnostic explaining the supported 512 GB Ultra single-node and multi-Mac
    Thunderbolt 5 RDMA expert-parallel cluster paths.
    """
    if target_tier not in VALID_TARGET_TIERS:
        raise ValueError(f"target_tier must be one of {VALID_TARGET_TIERS}, got {target_tier!r}")
    if not (1 <= world_size <= MAX_CLUSTER_WORLD_SIZE):
        raise ValueError(f"world_size must be in 1..{MAX_CLUSTER_WORLD_SIZE}, got {world_size}")

    if non_expert_bytes is not None and routed_expert_bytes is not None:
        per_rank_wired = non_expert_bytes + math.ceil(routed_expert_bytes / world_size)
    else:
        per_rank_wired = math.ceil(total_wired_bytes / world_size)

    eff_tier = target_tier
    if eff_tier == "auto":
        if world_size > 1:
            eff_tier = "cluster"
        elif host_ram_gib is not None:
            if host_ram_gib <= 32.0:
                eff_tier = "24gb-m4"
            elif host_ram_gib <= 160.0:
                eff_tier = "128gb-max"
            elif host_ram_gib <= 320.0:
                eff_tier = "256gb-ultra"
            else:
                eff_tier = "512gb-ultra"
        else:
            eff_tier = "24gb-m4"

    if eff_tier == "cluster":
        if host_ram_gib is not None:
            per_rank_budget = int(host_ram_gib * 0.75 * (1024**3))
        else:
            per_rank_budget = TIER_WIRED_BUDGET_BYTES["128gb-max"]
    else:
        per_rank_budget = TIER_WIRED_BUDGET_BYTES[eff_tier]

    fits = per_rank_wired <= per_rank_budget
    total_gib = total_wired_bytes / (1024**3)
    per_rank_gib = per_rank_wired / (1024**3)
    budget_gib = per_rank_budget / (1024**3)

    if fits:
        rec = (
            f"wired plan fits {eff_tier} (world_size={world_size}: "
            f"{per_rank_gib:.2f} GiB/rank <= {budget_gib:.2f} GiB wired budget)"
        )
        return HardwareTierCheck(
            target_tier=eff_tier,
            world_size=world_size,
            total_wired_bytes=total_wired_bytes,
            per_rank_wired_bytes=per_rank_wired,
            per_rank_budget_bytes=per_rank_budget,
            fits=True,
            recommendation=rec,
        )

    rec = (
        f"ADR-0015 Tier 3 hardware-tier violation: deepseek41 wired backbone requires "
        f"{total_gib:.2f} GiB total ({per_rank_gib:.2f} GiB/rank at world_size={world_size}), "
        f"exceeding {eff_tier} wired budget ({budget_gib:.2f} GiB). "
        f"Full 552B+196B local wiring on 24 GB M4 is non-viable (<0.7 tok/s cold SSD thrash). "
        f"Use --target-tier 512gb-ultra on a 512 GB Apple Silicon Ultra node, or export "
        f"multi-Mac Thunderbolt 5 RDMA expert-parallel shards with --world-size 4 --target-tier cluster "
        f"(or `aminfer-tools partition-experts`)."
    )
    if not allow_override:
        raise HardwareTierError(rec)
    return HardwareTierCheck(
        target_tier=eff_tier,
        world_size=world_size,
        total_wired_bytes=total_wired_bytes,
        per_rank_wired_bytes=per_rank_wired,
        per_rank_budget_bytes=per_rank_budget,
        fits=False,
        recommendation=rec,
    )


def validate_config_hardware_tier(
    config: DeepSeek41Config,
    *,
    target_tier: str = "24gb-m4",
    host_ram_gib: float | None = None,
    world_size: int | None = None,
    allow_override: bool = False,
    all_experts_cold: bool = False,
    encoder_cold_decoder_hot: bool = False,
    decoder_hot_fraction: float = 0.0,
) -> HardwareTierCheck:
    total_b, non_exp_b, exp_b = config.estimated_backbone_bytes()
    if all_experts_cold:
        eff_routed = 0
        eff_total = non_exp_b
    elif encoder_cold_decoder_hot:
        dec_frac = config.num_decoder_layers / max(1, config.total_layers)
        dec_b = int(exp_b * dec_frac * max(0.0, min(1.0, decoder_hot_fraction)))
        eff_routed = dec_b
        eff_total = non_exp_b + dec_b
    else:
        eff_routed = exp_b
        eff_total = total_b
    return evaluate_hardware_tier(
        eff_total,
        target_tier=target_tier,
        host_ram_gib=host_ram_gib,
        world_size=config.world_size if world_size is None else world_size,
        non_expert_bytes=non_exp_b,
        routed_expert_bytes=eff_routed,
        allow_override=allow_override,
    )


# ---------------------------------------------------------------------------
# Synthetic `deepseek41` generator (backbone + Engram + DSpark + shards)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepSeek41SynthBundle:
    amf_path: pathlib.Path
    engram_paths: list[pathlib.Path]
    dspark_amf_path: pathlib.Path | None
    dspark_config_path: pathlib.Path | None
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    cluster_manifest_path: pathlib.Path | None
    rank_amf_paths: list[pathlib.Path]
    metas: list[PackedTensorMeta]
    engram_metas: list[list[PackedTensorMeta]]
    dspark_metas: list[PackedTensorMeta]
    total_bytes: int
    config: DeepSeek41Config


def _plan_deepseek41_backbone_tensors(
    config: DeepSeek41Config,
    cold_experts: set[tuple[int, int]],
) -> tuple[list[_TensorPlan], dict[str, bytes]]:
    h = config.hidden_size
    qdt = config.quant_dtype
    qb = config.quant_bits
    gs = config.group_size if qb > 0 else 0
    n_heads = config.num_attention_heads
    q_dim = n_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)
    kv_a_out = config.kv_lora_rank + config.qk_rope_head_dim
    kv_b_out = n_heads * (config.qk_nope_head_dim + config.v_head_dim)
    v_tot = n_heads * config.v_head_dim
    moe_inter = config.moe_intermediate_size
    sh_inter = config.shared_expert_intermediate_size
    streams = config.mhc_num_streams
    modes = config.effective_csa2_modes()

    inv_h = 1.0 / math.sqrt(h)
    inv_ql = 1.0 / math.sqrt(config.q_lora_rank)
    inv_kvl = 1.0 / math.sqrt(config.kv_lora_rank)
    inv_v = 1.0 / math.sqrt(v_tot)
    inv_moe = 1.0 / math.sqrt(moe_inter)
    inv_sh = 1.0 / math.sqrt(sh_inter)

    plans: list[_TensorPlan] = [
        _TensorPlan("embed_tokens.weight", TensorKind.EMBEDDING, DType.BF16, 0, 0, NO_LAYER, NO_EXPERT, (config.vocab_size, h), 0.25),
    ]
    raw_payloads: dict[str, bytes] = {}

    def _add_layer_tensors(layer: int) -> None:
        # Mega-mHC single-pass hyper-connection mixing weights (4 streams default).
        plans.extend(
            [
                _TensorPlan(f"layers.{layer}.mhc.pre_mix.weight", TensorKind.DENSE, DType.BF16, 0, 0, layer, NO_EXPERT, (streams, h), 0.25 * inv_h),
                _TensorPlan(f"layers.{layer}.mhc.post_mix.weight", TensorKind.DENSE, DType.BF16, 0, 0, layer, NO_EXPERT, (streams, h), 0.25 * inv_h),
                _TensorPlan(f"layers.{layer}.mhc.res_mix.weight", TensorKind.DENSE, DType.BF16, 0, 0, layer, NO_EXPERT, (streams, streams), 0.25),
                _TensorPlan(f"layers.{layer}.input_layernorm.weight", TensorKind.NORM, DType.BF16, 0, 0, layer, NO_EXPERT, (h,), 1.0),
                # MLA + CSA2 projections (all names <= 47 UTF-8 bytes).
                _TensorPlan(f"layers.{layer}.self_attn.q_a_proj", TensorKind.DENSE, qdt, qb, gs, layer, NO_EXPERT, (config.q_lora_rank, h), 0.5 * inv_h),
                _TensorPlan(f"layers.{layer}.self_attn.q_a_layernorm.weight", TensorKind.NORM, DType.BF16, 0, 0, layer, NO_EXPERT, (config.q_lora_rank,), 1.0),
                _TensorPlan(f"layers.{layer}.self_attn.q_b_proj", TensorKind.DENSE, qdt, qb, gs, layer, NO_EXPERT, (q_dim, config.q_lora_rank), 0.5 * inv_ql),
                _TensorPlan(f"layers.{layer}.self_attn.kv_a_proj_with_mqa", TensorKind.DENSE, qdt, qb, gs, layer, NO_EXPERT, (kv_a_out, h), 0.5 * inv_h),
                _TensorPlan(f"layers.{layer}.self_attn.kv_a_layernorm.weight", TensorKind.NORM, DType.BF16, 0, 0, layer, NO_EXPERT, (config.kv_lora_rank,), 1.0),
                _TensorPlan(f"layers.{layer}.self_attn.kv_b_proj", TensorKind.DENSE, qdt, qb, gs, layer, NO_EXPERT, (kv_b_out, config.kv_lora_rank), 0.5 * inv_kvl),
                _TensorPlan(f"layers.{layer}.self_attn.o_proj", TensorKind.DENSE, qdt, qb, gs, layer, NO_EXPERT, (h, v_tot), 0.5 * inv_v),
            ]
        )
        if modes[layer] in ("full", "reindex"):
            plans.append(
                _TensorPlan(
                    f"layers.{layer}.self_attn.csa2_idx_proj",
                    TensorKind.DENSE,
                    qdt,
                    qb,
                    gs,
                    layer,
                    NO_EXPERT,
                    (config.kv_lora_rank, h),
                    0.35 * inv_h,
                )
            )

        plans.extend(
            [
                _TensorPlan(f"layers.{layer}.post_attention_layernorm.weight", TensorKind.NORM, DType.BF16, 0, 0, layer, NO_EXPERT, (h,), 1.0),
                _TensorPlan(f"layers.{layer}.mlp.gate.weight", TensorKind.ROUTER, DType.BF16, 0, 0, layer, NO_EXPERT, (config.num_experts, h), 0.5 * inv_h),
            ]
        )

        if config.has_shared_expert:
            plans.extend(
                [
                    _TensorPlan(f"layers.{layer}.mlp.shared_expert.gate_proj", TensorKind.SHARED_EXPERT, qdt, qb, gs, layer, NO_EXPERT, (sh_inter, h), 0.5 * inv_h),
                    _TensorPlan(f"layers.{layer}.mlp.shared_expert.up_proj", TensorKind.SHARED_EXPERT, qdt, qb, gs, layer, NO_EXPERT, (sh_inter, h), 0.5 * inv_h),
                    _TensorPlan(f"layers.{layer}.mlp.shared_expert.down_proj", TensorKind.SHARED_EXPERT, qdt, qb, gs, layer, NO_EXPERT, (h, sh_inter), 0.5 * inv_sh),
                    _TensorPlan(f"layers.{layer}.mlp.shared_expert_gate.weight", TensorKind.SHARED_EXPERT, DType.BF16, 0, 0, layer, NO_EXPERT, (1, h), 0.25 * inv_h),
                ]
            )

        for e in range(config.num_experts):
            ekind = TensorKind.COLD_EXPERT if (layer, e) in cold_experts else TensorKind.HOT_EXPERT
            plans.extend(
                [
                    _TensorPlan(f"layers.{layer}.mlp.experts.{e}.gate_proj", ekind, qdt, qb, gs, layer, e, (moe_inter, h), 0.5 * inv_h),
                    _TensorPlan(f"layers.{layer}.mlp.experts.{e}.up_proj", ekind, qdt, qb, gs, layer, e, (moe_inter, h), 0.5 * inv_h),
                    _TensorPlan(f"layers.{layer}.mlp.experts.{e}.down_proj", ekind, qdt, qb, gs, layer, e, (h, moe_inter), 0.5 * inv_moe),
                ]
            )

    # 1. Causal Encoder layers [0 .. num_encoder_layers - 1]
    for layer in range(config.num_encoder_layers):
        _add_layer_tensors(layer)

    # 2. Encoder -> Decoder Global KV Projector (CED bridge)
    plans.extend(
        [
            _TensorPlan("enc_dec.norm.weight", TensorKind.NORM, DType.BF16, 0, 0, NO_LAYER, NO_EXPERT, (h,), 1.0),
            _TensorPlan("enc_dec.global_kv_a_proj", TensorKind.DENSE, qdt, qb, gs, NO_LAYER, NO_EXPERT, (kv_a_out, h), 0.5 * inv_h),
            _TensorPlan("enc_dec.global_kv_norm.weight", TensorKind.NORM, DType.BF16, 0, 0, NO_LAYER, NO_EXPERT, (config.kv_lora_rank,), 1.0),
            _TensorPlan("enc_dec.global_kv_b_proj", TensorKind.DENSE, qdt, qb, gs, NO_LAYER, NO_EXPERT, (kv_b_out, config.kv_lora_rank), 0.5 * inv_kvl),
        ]
    )

    # 3. Causal Decoder layers [num_encoder_layers .. total_layers - 1]
    for layer in range(config.num_encoder_layers, config.total_layers):
        _add_layer_tensors(layer)

    plans.append(_TensorPlan("norm.weight", TensorKind.NORM, DType.BF16, 0, 0, NO_LAYER, NO_EXPERT, (h,), 1.0))
    if not config.tie_word_embeddings:
        plans.append(_TensorPlan("lm_head", TensorKind.LM_HEAD, qdt, qb, gs, NO_LAYER, NO_EXPERT, (config.vocab_size, h), 0.5 * inv_h))

    return plans, raw_payloads


def _generate_engram_shards(
    output_dir: pathlib.Path,
    config: DeepSeek41Config,
    seed: int,
) -> tuple[list[pathlib.Path], list[list[PackedTensorMeta]]]:
    """Generate 16 KiB-aligned sharded Engram `.amf` files (`TensorKind.SPARSE_TABLE`).

    Each row has width `2 * hidden_size`, packing contiguous `[key_vec, value_vec]`
    so C10 `PleStore.lookupEngramConditional` fetches both key and value in one
    `F_NOCACHE` page lookup.
    """
    base_path = output_dir / "engram.amf"
    paths = engram_shard_paths(base_path, config.engram_shards)
    dt = config.engram_amf_dtype
    qb = config.engram_quant_bits
    gs = config.group_size if qb > 0 else 0
    packed_width = config.hidden_size * 2
    ranges = _split_row_ranges(config.engram_num_rows, config.engram_shards)

    all_metas: list[list[PackedTensorMeta]] = []
    for s_idx, (r0, r1) in enumerate(ranges):
        cnt = r1 - r0
        plans: list[_TensorPlan] = []
        for layer in range(config.total_layers):
            name = normalize_engram_name(
                f"engram.layers.{layer}",
                shard_idx=(s_idx if config.engram_shards > 1 else None),
                quantized=(qb > 0),
            )
            plans.append(
                _TensorPlan(
                    name=name,
                    kind=TensorKind.SPARSE_TABLE,
                    dtype=dt,
                    quant_bits=qb,
                    group_size=gs,
                    layer=layer,
                    expert=NO_EXPERT,
                    shape=(cnt, packed_width),
                    scale_hint=0.15,
                )
            )
        shard_id = f"{config.model_id[:48]}-engram-s{s_idx}" if config.engram_shards > 1 else f"{config.model_id[:52]}-engram"
        all_metas.append(_write_planned_amf(paths[s_idx], shard_id, plans, seed + 3001 + s_idx))
    return paths, all_metas


def _generate_dspark_head(
    output_dir: pathlib.Path,
    config: DeepSeek41Config,
    seed: int,
) -> tuple[pathlib.Path, pathlib.Path, list[PackedTensorMeta]]:
    h = config.hidden_size
    inv_h = 1.0 / math.sqrt(h)
    plans: list[_TensorPlan] = [
        _TensorPlan("dspark.enc_proj.weight", TensorKind.DENSE, DType.F16, 0, 0, NO_LAYER, NO_EXPERT, (h, h), 0.5 * inv_h),
        _TensorPlan("dspark.dec_proj.weight", TensorKind.DENSE, DType.F16, 0, 0, NO_LAYER, NO_EXPERT, (h, h), 0.5 * inv_h),
        _TensorPlan("dspark.norm.weight", TensorKind.NORM, DType.F16, 0, 0, NO_LAYER, NO_EXPERT, (h,), 1.0),
    ]
    for s in range(config.dspark_block_width):
        plans.append(
            _TensorPlan(f"dspark.slot.{s}.proj.weight", TensorKind.DENSE, DType.F16, 0, 0, s, NO_EXPERT, (h, h), 0.5 * inv_h)
        )
        plans.append(
            _TensorPlan(f"dspark.slot.{s}.embed.weight", TensorKind.EMBEDDING, DType.F16, 0, 0, s, NO_EXPERT, (h,), 0.25)
        )

    amf_path = output_dir / "dspark.amf"
    cfg_path = output_dir / "dspark_config.json"
    metas = _write_planned_amf(amf_path, f"{config.model_id[:52]}-dspark", plans, seed + 4007)
    dspark_cfg = {
        "draft_format": "dspark",
        "block_width": config.dspark_block_width,
        "top_k_per_pos": config.dspark_top_k_per_pos,
        "max_tree_nodes": config.dspark_max_tree_nodes,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "tied_lm_head": True,
    }
    cfg_path.write_text(json.dumps(dspark_cfg, indent=2) + "\n", encoding="utf-8")
    return amf_path, cfg_path, metas


def _encode_mxfp8_bf16_like(rng: random.Random, rows: int, cols: int, scale: float = 0.08) -> tuple[bytes, bytes]:
    """Synthesize deterministic `[32, 32]` UE8M0 FP8 (`F8_E4M3` + `F8_E8M0`) weight & scale payloads."""
    nb32 = (rows + 31) // 32
    kb32 = (cols + 31) // 32
    e4m3_raw = bytearray(rows * cols)
    sc_raw = bytearray(nb32 * kb32)
    # Use exact representable normal E4M3 codes (sign bit 0/1, exp 4..10 -> modest range) + exponent 125..128.
    for tr in range(nb32):
        for tc in range(kb32):
            exp_b = 123 + ((tr * 7 + tc * 13) % 5)  # 2^(exp-127)
            sc_raw[tr * kb32 + tc] = exp_b
    for i in range(rows * cols):
        sgn = 0x80 if rng.random() < 0.5 else 0x00
        mag = 0x20 + (i % 24)  # well within normal positive finite E4M3 codes [0.25..3.5]
        e4m3_raw[i] = sgn | mag
    _ = scale
    return bytes(e4m3_raw), bytes(sc_raw)


def _encode_mxfp8_rowwise(rng: random.Random, rows: int, cols: int) -> tuple[bytes, bytes]:
    """Synthesize deterministic `1x32` row-wise UE8M0 FP8 (`F8_E4M3 [R, C]` + `F8_E8M0 [R, C//32]`) Engram payloads."""
    kb32 = cols // 32
    w_raw = bytes(((0x80 if (i * 7 + 3) % 5 == 0 else 0x00) | (0x20 + (i % 20))) for i in range(rows * cols))
    s_raw = bytes((124 + ((i * 5) % 4)) for i in range(rows * kb32))
    return w_raw, s_raw


def _encode_hf_mxfp4(rng: random.Random, rows: int, cols: int) -> tuple[bytes, bytes]:
    """Synthesize deterministic HF `MXFP4` (`I8 [R, C//2]` packed e2m1fn_x2 + `F8_E8M0 [R, C//32]`)."""
    kb2 = cols // 2
    kb32 = cols // 32
    w_raw = bytes(((rng.randrange(0, 16) << 4) | rng.randrange(0, 16)) for _ in range(rows * kb2))
    s_raw = bytes((125 + rng.randrange(0, 4)) for _ in range(rows * kb32))
    return w_raw, s_raw


def _write_sf_file(path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
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


def _det_bf16_raw(rng: random.Random, count: int, scale: float = 0.08) -> bytes:
    u16s = []
    for _ in range(count):
        v = (rng.random() * 2.0 - 1.0) * scale
        u32 = struct.unpack("<I", struct.pack("<f", v))[0]
        u16s.append((u32 >> 16) & 0xFFFF)
    return struct.pack(f"<{count}H", *u16s)


def _det_f32_raw(rng: random.Random, count: int, lo: float = -0.25, hi: float = 0.25) -> bytes:
    return struct.pack(f"<{count}f", *(lo + rng.random() * (hi - lo) for _ in range(count)))


def _generate_real_hf_deepseek41_synth_bundle(
    out_path: pathlib.Path,
    cfg: DeepSeek41Config,
    *,
    seed: int,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int = 0,
    target_dtype: str = "auto",
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    world_size: int = 1,
    partition_strategy: str = "contiguous_by_rank",
    target_tier: str = "auto",
    encoder_cold_decoder_hot: bool = False,
    allow_tier_override: bool = False,
) -> DeepSeek41SynthBundle:
    """Synthesize a genuine real-HF `deepseek_v41` checkpoint and pack it via `convert_deepseek41_checkpoint`."""
    rng = random.Random(seed)
    h = max(64, cfg.hidden_size)
    vocab = cfg.vocab_size
    n_heads = max(2, min(8, cfg.num_attention_heads))
    q_lora = max(64, min(h, cfg.q_lora_rank))
    kv_lora = max(32, min(h, cfg.kv_lora_rank))
    qk_nope = 32
    qk_rope = 32
    o_groups = 2
    o_lora_cfg = int(getattr(cfg, "o_lora_rank", 512) or 512)
    o_lora = max(64, min(h, o_lora_cfg // 8 if o_lora_cfg >= 512 else 64))
    moe_i = max(64, cfg.moe_intermediate_size)
    n_exp = cfg.num_experts
    idx_hd = 64
    idx_heads = 4
    mhc_s = cfg.mhc_num_streams
    engram_dim = 64
    enc_l = cfg.num_encoder_layers
    dec_l = cfg.num_decoder_layers
    tot_l = enc_l + dec_l
    eng_l = 1 if tot_l > 1 else 0
    compress_ratios = [0 if i == 0 else (2 if i < enc_l else 1) for i in range(tot_l)]

    hf_cfg = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4", "scale_block_size": [32, 32]},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": vocab,
            "hidden_size": h,
            "num_hidden_layers": tot_l,
            "num_encoder_layers": enc_l,
            "num_decoder_layers": dec_l,
            "num_attention_heads": n_heads,
            "num_key_value_heads": 1,
            "head_dim": kv_lora,
            "kv_lora_rank": kv_lora,
            "q_lora_rank": q_lora,
            "qk_nope_head_dim": qk_nope,
            "qk_rope_head_dim": qk_rope,
            "v_head_dim": qk_nope,
            "o_lora_rank": o_lora,
            "o_groups": o_groups,
            "n_routed_experts": n_exp,
            "n_shared_experts": 1 if cfg.has_shared_expert else 0,
            "num_experts_per_tok": min(n_exp, cfg.encoder_moe_top_k),
            "moe_intermediate_size": moe_i,
            "hc_mult": mhc_s,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "compress_ratios": compress_ratios,
            "kv_source_layer_ids": [0, max(0, enc_l - 1)] if enc_l > 1 else [0],
            "engram_layer_ids": [eng_l],
            "engram_num_embeddings": [cfg.engram_num_rows],
            "engram_n_heads": 4,
            "engram_head_dim": engram_dim,
            "num_nextn_predict_layers": 3 if cfg.include_dspark else 0,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [0, max(0, tot_l - 2), tot_l - 1] if tot_l >= 2 else [0],
            "dspark_n_routed_experts": min(n_exp, 8),
            "dspark_routed_top_k": 3,
            "routed_scaling_factor": 1.5,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "swiglu_limit": 10.0,
            "tie_word_embeddings": False,
        },
    }

    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    tensors["embed.weight"] = ("BF16", (vocab, h), _det_bf16_raw(rng, vocab * h))
    tensors["head.weight"] = ("BF16", (vocab, h), _det_bf16_raw(rng, vocab * h))
    tensors["norm.weight"] = ("BF16", (h,), _det_bf16_raw(rng, h, 0.25))

    def _add_mxfp8(key: str, rows: int, cols: int) -> None:
        w_raw, s_raw = _encode_mxfp8_bf16_like(rng, rows, cols)
        tensors[f"{key}.weight"] = ("F8_E4M3", (rows, cols), w_raw)
        tensors[f"{key}.scale"] = ("F8_E8M0", ((rows + 31) // 32, (cols + 31) // 32), s_raw)

    def _add_mxfp4(key: str, rows: int, cols: int) -> None:
        w_raw, s_raw = _encode_hf_mxfp4(rng, rows, cols)
        tensors[f"{key}.weight"] = ("I8", (rows, cols // 2), w_raw)
        tensors[f"{key}.scale"] = ("F8_E8M0", (rows, cols // 32), s_raw)

    def _add_moe_block(pfx: str, block_exp: int, *, include_router_vl: bool = True) -> None:
        tensors[f"{pfx}.ffn.gate.weight"] = ("BF16", (block_exp, h), _det_bf16_raw(rng, block_exp * h))
        tensors[f"{pfx}.ffn.gate.bias"] = ("F32", (block_exp,), _det_f32_raw(rng, block_exp))
        if include_router_vl:
            tensors[f"{pfx}.ffn.gate.bias_vl"] = ("F32", (block_exp,), _det_f32_raw(rng, block_exp))
        if cfg.has_shared_expert:
            _add_mxfp8(f"{pfx}.ffn.shared_experts.w1", moe_i, h)
            _add_mxfp8(f"{pfx}.ffn.shared_experts.w2", h, moe_i)
            _add_mxfp8(f"{pfx}.ffn.shared_experts.w3", moe_i, h)
        for e in range(block_exp):
            _add_mxfp4(f"{pfx}.ffn.experts.{e}.w1", moe_i, h)
            _add_mxfp4(f"{pfx}.ffn.experts.{e}.w2", h, moe_i)
            _add_mxfp4(f"{pfx}.ffn.experts.{e}.w3", moe_i, h)

    for l_i in range(tot_l):
        lp = f"layers.{l_i}"
        tensors[f"{lp}.attn_norm.weight"] = ("BF16", (h,), _det_bf16_raw(rng, h, 0.25))
        tensors[f"{lp}.ffn_norm.weight"] = ("BF16", (h,), _det_bf16_raw(rng, h, 0.25))
        for tag in ("hc_attn", "hc_ffn"):
            tensors[f"{lp}.{tag}_base"] = ("BF16", (mhc_s, mhc_s), _det_bf16_raw(rng, mhc_s * mhc_s, 0.25))
            tensors[f"{lp}.{tag}_fn"] = ("BF16", (h, mhc_s), _det_bf16_raw(rng, h * mhc_s))
            tensors[f"{lp}.{tag}_scale"] = ("F32", (3,), _det_f32_raw(rng, 3, 0.05, 0.2))
        _add_mxfp8(f"{lp}.attn.wq_a", q_lora, h)
        _add_mxfp8(f"{lp}.attn.wq_b", n_heads * (qk_nope + qk_rope), q_lora)
        tensors[f"{lp}.attn.q_norm.weight"] = ("BF16", (q_lora,), _det_bf16_raw(rng, q_lora, 0.25))
        _add_mxfp8(f"{lp}.attn.wkv", kv_lora + qk_rope, h)
        tensors[f"{lp}.attn.kv_norm.weight"] = ("BF16", (kv_lora,), _det_bf16_raw(rng, kv_lora, 0.25))
        _add_mxfp8(f"{lp}.attn.wo_a", o_lora * o_groups, n_heads * qk_nope)
        _add_mxfp8(f"{lp}.attn.wo_b", h, o_lora)
        tensors[f"{lp}.attn.attn_sink"] = ("F32", (n_heads,), _det_f32_raw(rng, n_heads, -0.5, 0.0))
        if compress_ratios[l_i] > 1:
            tensors[f"{lp}.attn.compressor.norm.weight"] = ("BF16", (kv_lora + qk_rope,), _det_bf16_raw(rng, kv_lora + qk_rope, 0.25))
            tensors[f"{lp}.attn.compressor.wgate.weight"] = ("BF16", (compress_ratios[l_i], kv_lora + qk_rope), _det_bf16_raw(rng, compress_ratios[l_i] * (kv_lora + qk_rope)))
            tensors[f"{lp}.attn.compressor.wkv.weight"] = ("BF16", (kv_lora + qk_rope, kv_lora + qk_rope), _det_bf16_raw(rng, (kv_lora + qk_rope) ** 2))
        tensors[f"{lp}.attn.indexer.wk.weight"] = ("BF16", (idx_hd, h), _det_bf16_raw(rng, idx_hd * h))
        _add_mxfp8(f"{lp}.attn.indexer.wq_b", idx_heads * idx_hd, q_lora)
        tensors[f"{lp}.attn.indexer.k_norm.weight"] = ("BF16", (idx_hd,), _det_bf16_raw(rng, idx_hd, 0.25))
        tensors[f"{lp}.attn.indexer.weights_proj.weight"] = ("BF16", (idx_heads, h), _det_bf16_raw(rng, idx_heads * h))
        if l_i == eng_l:
            ew_w, ew_s = _encode_mxfp8_rowwise(rng, cfg.engram_num_rows, engram_dim)
            tensors[f"{lp}.engram.embed.weight"] = ("F8_E4M3", (cfg.engram_num_rows, engram_dim), ew_w)
            tensors[f"{lp}.engram.embed.scale"] = ("F8_E8M0", (cfg.engram_num_rows, engram_dim // 32), ew_s)
            _add_mxfp8(f"{lp}.engram.wkv", h, engram_dim * 2)
            tensors[f"{lp}.engram.q_weight"] = ("BF16", (4, h), _det_bf16_raw(rng, 4 * h))
            tensors[f"{lp}.engram.k_weight"] = ("BF16", (4, engram_dim * 2), _det_bf16_raw(rng, 4 * engram_dim * 2))
        _add_moe_block(lp, n_exp)

    if cfg.include_dspark:
        ds_exp = min(n_exp, 8)
        for m_idx in range(3):
            mp = f"mtp.{m_idx}"
            if m_idx == 0:
                tensors[f"{mp}.main_norm.weight"] = ("BF16", (h,), _det_bf16_raw(rng, h, 0.25))
                _add_mxfp8(f"{mp}.main_proj", h, h * 3)
            tensors[f"{mp}.attn_norm.weight"] = ("BF16", (h,), _det_bf16_raw(rng, h, 0.25))
            tensors[f"{mp}.ffn_norm.weight"] = ("BF16", (h,), _det_bf16_raw(rng, h, 0.25))
            for tag in ("hc_attn", "hc_ffn"):
                tensors[f"{mp}.{tag}_base"] = ("BF16", (mhc_s, mhc_s), _det_bf16_raw(rng, mhc_s * mhc_s, 0.25))
                tensors[f"{mp}.{tag}_fn"] = ("BF16", (h, mhc_s), _det_bf16_raw(rng, h * mhc_s))
                tensors[f"{mp}.{tag}_scale"] = ("F32", (3,), _det_f32_raw(rng, 3, 0.05, 0.2))
            _add_mxfp8(f"{mp}.attn.wq_a", q_lora, h)
            _add_mxfp8(f"{mp}.attn.wq_b", n_heads * (qk_nope + qk_rope), q_lora)
            tensors[f"{mp}.attn.q_norm.weight"] = ("BF16", (q_lora,), _det_bf16_raw(rng, q_lora, 0.25))
            _add_mxfp8(f"{mp}.attn.wkv", kv_lora + qk_rope, h)
            tensors[f"{mp}.attn.kv_norm.weight"] = ("BF16", (kv_lora,), _det_bf16_raw(rng, kv_lora, 0.25))
            _add_mxfp8(f"{mp}.attn.wo_a", o_lora * o_groups, n_heads * qk_nope)
            _add_mxfp8(f"{mp}.attn.wo_b", h, o_lora)
            _add_moe_block(mp, ds_exp, include_router_vl=False)
            if m_idx == 2:
                tensors[f"{mp}.norm.weight"] = ("BF16", (h,), _det_bf16_raw(rng, h, 0.25))
                tensors[f"{mp}.confidence_head.proj.weight"] = ("BF16", (1, h + 64), _det_bf16_raw(rng, h + 64))
                tensors[f"{mp}.markov_head.embed.weight"] = ("BF16", (vocab, 64), _det_bf16_raw(rng, vocab * 64))
                tensors[f"{mp}.markov_head.head.weight"] = ("BF16", (vocab, 64), _det_bf16_raw(rng, vocab * 64))

    cold_set = set(cold_experts)
    if cold_experts_per_layer > 0:
        for l_i in range(tot_l):
            for e_i in range(n_exp - cold_experts_per_layer, n_exp):
                cold_set.add((l_i, e_i))

    tok_dict = build_qwen3_tokenizer_dict(cfg)  # type: ignore[arg-type]
    with tempfile.TemporaryDirectory() as tmp_sf:
        sf_dir = pathlib.Path(tmp_sf)
        _write_sf_file(sf_dir / "model.safetensors", tensors)
        (sf_dir / "config.json").write_text(json.dumps(hf_cfg, indent=2) + "\n", encoding="utf-8")
        (sf_dir / "tokenizer.json").write_text(json.dumps(tok_dict) + "\n", encoding="utf-8")
        rep = convert_deepseek41_checkpoint(
            sf_dir,
            out_path / "model.amf",
            model_id=cfg.model_id,
            target_dtype=target_dtype,
            quant_profile=quant_profile,
            target_bits=target_bits,
            cold_experts=cold_set,
            engram_output=out_path / "engram.amf",
            engram_shards=cfg.engram_shards,
            dspark_output=(out_path / "dspark.amf") if cfg.include_dspark else None,
            config_output=out_path / "config.json",
            target_tier=target_tier,
            world_size=world_size,
            rank_dir=(out_path / "shards") if world_size > 1 else None,
            partition_strategy=partition_strategy,
            allow_tier_override=allow_tier_override,
            encoder_cold_decoder_hot=encoder_cold_decoder_hot,
        )
    tok_path = out_path / "tokenizer.json"
    tok_path.write_text(json.dumps(tok_dict, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    tot_b = rep.backbone_amf.stat().st_size + sum(p.stat().st_size for p in rep.engram_paths)
    if rep.dspark_amf is not None:
        tot_b += rep.dspark_amf.stat().st_size
    return DeepSeek41SynthBundle(
        amf_path=rep.backbone_amf,
        engram_paths=rep.engram_paths,
        dspark_amf_path=rep.dspark_amf,
        dspark_config_path=rep.dspark_config_path,
        config_path=rep.config_path or (out_path / "config.json"),
        tokenizer_path=tok_path,
        cluster_manifest_path=rep.partition_report.manifest_path if rep.partition_report is not None else None,
        rank_amf_paths=rep.partition_report.rank_amf_paths if rep.partition_report is not None else [],
        metas=rep.backbone_metas,
        engram_metas=rep.engram_metas,
        dspark_metas=rep.dspark_metas,
        total_bytes=tot_b,
        config=cfg,
    )


def generate_synth_deepseek41_bundle(
    out_dir: pathlib.Path,
    preset: str = "tiny-deepseek41",
    *,
    config: DeepSeek41Config | None = None,
    seed: int = 42,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int = 0,
    world_size: int = 1,
    partition_strategy: str = "contiguous_by_rank",
    allow_huge_synth: bool = False,
    real_hf_schema: bool = False,
    target_dtype: str | None = None,
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    target_tier: str = "auto",
    encoder_cold_decoder_hot: bool = False,
    allow_tier_override: bool = False,
) -> DeepSeek41SynthBundle:
    """Generate a deterministic 16 KiB-aligned `deepseek41` synthetic bundle.

    Produces `model.amf`, sharded `engram*.amf` (`SPARSE_TABLE`), optional
    `dspark.amf` + `dspark_config.json`, `config.json`, `tokenizer.json`, and
    (when `world_size > 1`) per-rank expert shards + `cluster_manifest.json`.
    """
    if config is None:
        if preset not in DEEPSEEK41_PRESETS:
            raise ValueError(f"unknown deepseek41 preset {preset!r}; expected one of {sorted(DEEPSEEK41_PRESETS)}")
        cfg = DEEPSEEK41_PRESETS[preset]
    else:
        cfg = config

    if world_size != cfg.world_size:
        cfg = cfg.with_overrides(world_size=world_size)
    cfg.validate()

    est_total, _, _ = cfg.estimated_backbone_bytes()
    if est_total > DEFAULT_MAX_SYNTH_BYTES and not allow_huge_synth:
        raise HardwareTierError(
            f"Refusing to synthesize {est_total / (1024**3):.2f} GiB synthetic backbone for preset "
            f"{preset!r} on disk (limit {DEFAULT_MAX_SYNTH_BYTES / (1024**3):.2f} GiB). "
            f"Use 'tiny-deepseek41' or 'scale-deepseek41-512e' for local testing, or pass allow_huge_synth=True."
        )

    if cold_experts_per_layer < 0 or cold_experts_per_layer > cfg.num_experts:
        raise ValueError(
            f"cold_experts_per_layer must be in 0..{cfg.num_experts}, got {cold_experts_per_layer}"
        )

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if real_hf_schema:
        return _generate_real_hf_deepseek41_synth_bundle(
            out_path,
            cfg,
            seed=seed,
            cold_experts=cold_experts,
            cold_experts_per_layer=cold_experts_per_layer,
            target_dtype=target_dtype if target_dtype is not None else "auto",
            quant_profile=quant_profile,
            target_bits=target_bits,
            world_size=world_size,
            partition_strategy=partition_strategy,
            target_tier=target_tier,
            encoder_cold_decoder_hot=encoder_cold_decoder_hot,
            allow_tier_override=allow_tier_override,
        )

    cold_set = set(cold_experts)
    if cold_experts_per_layer > 0:
        for layer in range(cfg.total_layers):
            for e in range(cfg.num_experts - cold_experts_per_layer, cfg.num_experts):
                cold_set.add((layer, e))

    amf_path = out_path / "model.amf"
    config_path = out_path / "config.json"
    tokenizer_path = out_path / "tokenizer.json"

    plans, raw_payloads = _plan_deepseek41_backbone_tensors(cfg, cold_set)
    metas = _write_planned_amf(amf_path, cfg.model_id, plans, seed, raw_payloads=raw_payloads)

    engram_paths, engram_metas = _generate_engram_shards(out_path, cfg, seed)

    dspark_amf_path: pathlib.Path | None = None
    dspark_config_path: pathlib.Path | None = None
    dspark_metas: list[PackedTensorMeta] = []
    if cfg.include_dspark:
        dspark_amf_path, dspark_config_path, dspark_metas = _generate_dspark_head(out_path, cfg, seed)

    cluster_manifest_path: pathlib.Path | None = None
    rank_amf_paths: list[pathlib.Path] = []
    if cfg.world_size > 1:
        part = partition_deepseek41_amf_shards(
            amf_path,
            cfg.world_size,
            out_path / "shards",
            strategy=partition_strategy,
        )
        cluster_manifest_path = part.manifest_path
        rank_amf_paths = part.rank_amf_paths

    cfg_dict = cfg.to_config_dict(
        engram_files=[p.name for p in engram_paths],
        dspark_file=(dspark_amf_path.name if dspark_amf_path is not None else None),
        cluster_manifest_file=(
            str(cluster_manifest_path.relative_to(out_path)) if cluster_manifest_path is not None else None
        ),
    )
    config_path.write_text(json.dumps(cfg_dict, indent=2) + "\n", encoding="utf-8")
    tokenizer_path.write_text(
        json.dumps(build_qwen3_tokenizer_dict(cfg), separators=(",", ":"), ensure_ascii=False) + "\n",  # type: ignore[arg-type]
        encoding="utf-8",
    )

    total_bytes = amf_path.stat().st_size + sum(p.stat().st_size for p in engram_paths)
    if dspark_amf_path is not None:
        total_bytes += dspark_amf_path.stat().st_size

    return DeepSeek41SynthBundle(
        amf_path=amf_path,
        engram_paths=engram_paths,
        dspark_amf_path=dspark_amf_path,
        dspark_config_path=dspark_config_path,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        cluster_manifest_path=cluster_manifest_path,
        rank_amf_paths=rank_amf_paths,
        metas=metas,
        engram_metas=engram_metas,
        dspark_metas=dspark_metas,
        total_bytes=total_bytes,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Multi-rank Expert-Parallel Shard Partitioner (`X1`-`X3` metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertPartitionReport:
    source_amf: pathlib.Path
    output_dir: pathlib.Path
    manifest_path: pathlib.Path
    world_size: int
    strategy: str
    num_layers: int
    num_experts: int
    rank_amf_paths: list[pathlib.Path]
    owned_experts_by_rank: list[list[int]]


def _assign_experts_to_ranks(
    expert_ids: list[int],
    world_size: int,
    strategy: str,
) -> list[list[int]]:
    if strategy not in VALID_PARTITION_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_PARTITION_STRATEGIES}, got {strategy!r}")
    buckets: list[list[int]] = [[] for _ in range(world_size)]
    if strategy == "contiguous_by_rank":
        n = len(expert_ids)
        base = n // world_size
        rem = n % world_size
        pos = 0
        for r in range(world_size):
            cnt = base + (1 if r < rem else 0)
            buckets[r] = expert_ids[pos : pos + cnt]
            pos += cnt
    else:
        for idx, e_id in enumerate(expert_ids):
            buckets[idx % world_size].append(e_id)
    return buckets


def _write_rank_shard_amf(
    reader: AmfReader,
    selected_entries: Sequence[PackedTensorMeta],
    out_path: pathlib.Path,
    model_id: str,
) -> list[PackedTensorMeta]:
    """Stream selected raw 16 KiB-aligned blobs from `reader` into a standalone `.amf` v0 shard."""
    model_id_bytes = model_id.encode("utf-8")
    if len(model_id_bytes) > 63:
        model_id_bytes = model_id_bytes[:63]

    tensor_count = len(selected_entries)
    dir_offset = APPLE_PAGE_SIZE
    dir_raw_len = tensor_count * ENTRY_STRUCT.size
    dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)

    metas: list[PackedTensorMeta] = []
    with out_path.open("wb") as fh:
        _stream_zeros(fh, dir_offset + dir_bytes, 0)
        cursor = dir_offset + dir_bytes
        for entry in selected_entries:
            blob = reader.read_blob(entry, verify_crc=True)
            if len(blob) != entry.blob_bytes or entry.blob_bytes % APPLE_PAGE_SIZE != 0:
                raise ValueError(f"misaligned blob in {entry.name!r}: {len(blob)} bytes")
            fh.write(blob)
            metas.append(
                PackedTensorMeta(
                    name=entry.name,
                    kind=entry.kind,
                    dtype=entry.dtype,
                    quant_bits=entry.quant_bits,
                    group_size=entry.group_size,
                    layer=entry.layer,
                    expert=entry.expert,
                    shape=entry.shape,
                    blob_offset=cursor,
                    blob_bytes=entry.blob_bytes,
                    data_bytes=entry.data_bytes,
                    params_offset=entry.params_offset,
                    params_bytes=entry.params_bytes,
                    blob_crc32=entry.blob_crc32,
                    layout=entry.layout,
                    quant_param_kind=entry.quant_param_kind,
                )
            )
            cursor += entry.blob_bytes

        dir_raw = _pack_directory(metas)
        dir_crc = _crc32(dir_raw)
        header = _pack_header(model_id_bytes, tensor_count, dir_offset, dir_bytes, dir_crc)

        fh.seek(0)
        fh.write(header)
        if len(header) < APPLE_PAGE_SIZE:
            fh.write(b"\x00" * (APPLE_PAGE_SIZE - len(header)))
        if dir_bytes > 0:
            fh.seek(dir_offset)
            fh.write(dir_raw)
            if len(dir_raw) < dir_bytes:
                fh.write(b"\x00" * (dir_bytes - len(dir_raw)))

    return metas


def partition_deepseek41_amf_shards(
    amf_path: pathlib.Path,
    world_size: int,
    out_dir: pathlib.Path,
    *,
    strategy: str = "contiguous_by_rank",
    replicate_backbone: bool = False,
    peer_endpoints: Sequence[str] | None = None,
) -> ExpertPartitionReport:
    """Split routed experts of a `deepseek41` `.amf` across `world_size` peer ranks.

    Produces `rank-XX-of-YY.amf` shards and `cluster_manifest.json` encoding
    `ExpertSource.local_wired_ram` vs `ExpertSource.thunderbolt_peer` ownership
    for Milestone M7 `X1`-`X3`.
    """
    if not (1 <= world_size <= MAX_CLUSTER_WORLD_SIZE):
        raise ValueError(f"world_size must be in 1..{MAX_CLUSTER_WORLD_SIZE}, got {world_size}")
    if strategy not in VALID_PARTITION_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_PARTITION_STRATEGIES}, got {strategy!r}")

    src = pathlib.Path(amf_path)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with AmfReader(src, verify_header_crc=True, verify_dir_crc=True) as reader:
        expert_set: set[int] = set()
        max_layer = -1
        non_expert_entries: list[PackedTensorMeta] = []
        expert_entries: list[PackedTensorMeta] = []

        for entry in reader.entries:
            if entry.layer != NO_LAYER and entry.layer > max_layer:
                max_layer = entry.layer
            if entry.expert != NO_EXPERT and entry.kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT):
                expert_set.add(entry.expert)
                expert_entries.append(entry)
            else:
                non_expert_entries.append(entry)

        expert_ids = sorted(expert_set)
        if not expert_ids:
            raise ValueError(f"no routed expert tensors found in {src}")
        if world_size > len(expert_ids):
            raise ValueError(f"world_size ({world_size}) exceeds available experts ({len(expert_ids)})")

        owned_by_rank = _assign_experts_to_ranks(expert_ids, world_size, strategy)
        owner_of: dict[int, int] = {}
        for r, e_list in enumerate(owned_by_rank):
            for e_id in e_list:
                owner_of[e_id] = r

        base_id = reader.header.model_id or src.stem
        rank_amf_paths: list[pathlib.Path] = []
        ranks_manifest: list[dict[str, object]] = []

        for r in range(world_size):
            owned_set = set(owned_by_rank[r])
            selected: list[PackedTensorMeta] = []
            if r == 0 or replicate_backbone:
                selected.extend(non_expert_entries)
            for e_entry in expert_entries:
                if e_entry.expert in owned_set:
                    selected.append(e_entry)

            shard_name = f"rank-{r:02d}-of-{world_size:02d}.amf"
            shard_path = out / shard_name
            metas = _write_rank_shard_amf(reader, selected, shard_path, f"{base_id[:52]}-r{r:02d}")
            rank_amf_paths.append(shard_path)

            endpoint = (
                peer_endpoints[r]
                if peer_endpoints is not None and r < len(peer_endpoints)
                else f"127.0.0.1:{7400 + r}"
            )
            ranks_manifest.append(
                {
                    "rank": r,
                    "role": "coordinator" if r == 0 else "peer_worker",
                    "amf_file": shard_name,
                    "amf_bytes": shard_path.stat().st_size,
                    "tensor_count": len(metas),
                    "has_backbone": bool(r == 0 or replicate_backbone),
                    "owned_expert_ids": owned_by_rank[r],
                    "owned_expert_count": len(owned_by_rank[r]),
                    "expert_source_on_rank0": "local_wired_ram" if r == 0 else "thunderbolt_peer",
                    "peer_endpoint_hint": endpoint,
                }
            )

    expert_map = [
        {
            "expert_id": e_id,
            "owner_rank": owner_of[e_id],
            "expert_source_on_rank0": "local_wired_ram" if owner_of[e_id] == 0 else "thunderbolt_peer",
        }
        for e_id in expert_ids
    ]

    manifest_path = out / "cluster_manifest.json"
    manifest = {
        "schema_version": 1,
        "model_type": "deepseek41",
        "source_amf": src.name,
        "world_size": world_size,
        "strategy": strategy,
        "replicate_backbone": replicate_backbone,
        "num_layers": max_layer + 1 if max_layer >= 0 else 0,
        "num_experts": len(expert_ids),
        "transport_preference": ["tb5_rdma", "tb4_tcp"],
        "ranks": ranks_manifest,
        "expert_map": expert_map,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return ExpertPartitionReport(
        source_amf=src,
        output_dir=out,
        manifest_path=manifest_path,
        world_size=world_size,
        strategy=strategy,
        num_layers=max_layer + 1 if max_layer >= 0 else 0,
        num_experts=len(expert_ids),
        rank_amf_paths=rank_amf_paths,
        owned_experts_by_rank=owned_by_rank,
    )


# ---------------------------------------------------------------------------
# Streaming Multi-Shard Safetensors Converter (`pack-deepseek41`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepSeek41ConvertReport:
    backbone_amf: pathlib.Path
    backbone_metas: list[PackedTensorMeta]
    engram_paths: list[pathlib.Path]
    engram_metas: list[list[PackedTensorMeta]]
    dspark_amf: pathlib.Path | None
    dspark_metas: list[PackedTensorMeta]
    config_path: pathlib.Path | None
    partition_report: ExpertPartitionReport | None
    tier_check: HardwareTierCheck
    stripped_vision_tensors: int
    vision_amf: pathlib.Path | None = None
    vision_metas: list[PackedTensorMeta] = field(default_factory=list)
    included_vision_tensors: int = 0
    dspark_config_path: pathlib.Path | None = None


def _decode_row_to_f32(raw_row: bytes, src_dtype: str) -> list[float]:
    if src_dtype == "F32":
        arr = array.array("f")
        arr.frombytes(raw_row)
        return list(arr)
    if src_dtype == "F16":
        n = len(raw_row) // 2
        return list(struct.unpack(f"<{n}e", raw_row))
    if src_dtype == "BF16":
        n = len(raw_row) // 2
        u16s = struct.unpack(f"<{n}H", raw_row)
        u32s = struct.pack(f"<{n}I", *(v << 16 for v in u16s))
        arr = array.array("f")
        arr.frombytes(u32s)
        return list(arr)
    raise ValueError(f"unsupported float source dtype for affine quantization: {src_dtype!r}")


def _quantize_2d_affine(
    raw_bytes: bytes,
    src_dtype: str,
    rows: int,
    cols: int,
    bits: int,
    group_size: int = 64,
) -> tuple[bytes, bytes]:
    """Quantize a row-major 2D float matrix into MLX-compatible group-64 `q3_affine` or `q4_affine`."""
    if cols % group_size != 0:
        raise ValueError(f"cols ({cols}) must be a multiple of group_size ({group_size})")
    qmax = (1 << bits) - 1
    elem_bytes = 4 if src_dtype == "F32" else 2
    row_stride = cols * elem_bytes
    packer = pack_mlx_affine_3bit if bits == 3 else pack_mlx_affine_4bit

    code_chunks: list[bytes] = []
    scale_f16 = bytearray()
    bias_f16 = bytearray()

    for r in range(rows):
        vals = _decode_row_to_f32(raw_bytes[r * row_stride : (r + 1) * row_stride], src_dtype)
        row_codes: list[int] = []
        for g0 in range(0, cols, group_size):
            grp = vals[g0 : g0 + group_size]
            vmin = min(grp)
            vmax = max(grp)
            span = vmax - vmin
            scale = span / qmax if span > 1e-8 else 1e-6
            inv_scale = 1.0 / scale
            scale_f16.extend(struct.pack("<e", scale))
            bias_f16.extend(struct.pack("<e", vmin))
            for x in grp:
                q = int(round((x - vmin) * inv_scale))
                if q < 0:
                    q = 0
                elif q > qmax:
                    q = qmax
                row_codes.append(q)
        code_chunks.append(packer(row_codes))

    params = interleave_scale_bias_f16(bytes(scale_f16), bytes(bias_f16))
    return b"".join(code_chunks), params


def _pack_engram_shards_from_safetensors(
    engram_tensors: dict[str, _ShardTensor],
    output_base_path: pathlib.Path,
    *,
    num_shards: int = 1,
    model_id: str = "deepseek41-engram",
    group_size: int = 64,
    dtype_mode: str = "auto",
) -> tuple[list[pathlib.Path], list[list[PackedTensorMeta]]]:
    """Stream Engram tables from safetensors into 1 or `num_shards` 16 KiB-aligned `.amf` files."""
    if not engram_tensors:
        return [], []
    shard_files = engram_shard_paths(output_base_path, num_shards)

    quant_prefixes: set[str] = set()
    mxfp8_prefixes: set[str] = set()
    for name in engram_tensors:
        if name.endswith(".scales"):
            prefix = name[: -len(".scales")]
            if f"{prefix}.weight" in engram_tensors and f"{prefix}.biases" in engram_tensors:
                quant_prefixes.add(prefix)
        elif name.endswith(".scale"):
            prefix = name[: -len(".scale")]
            if f"{prefix}.weight" in engram_tensors and engram_tensors[f"{prefix}.weight"].dtype == "F8_E4M3":
                mxfp8_prefixes.add(prefix)

    per_shard: list[list[_PlannedEntry]] = [[] for _ in range(num_shards)]
    consumed: set[str] = set()

    for prefix in sorted(mxfp8_prefixes):
        w = engram_tensors[f"{prefix}.weight"]
        s = engram_tensors[f"{prefix}.scale"]
        if s.dtype not in ("U8", "F8_E8M0") or len(w.shape) != 2 or len(s.shape) != 2:
            raise ValueError(f"Engram MXFP8 tensor {prefix} must be 2D F8_E4M3 + F8_E8M0/U8, got {w.dtype}{w.shape} / {s.dtype}{s.shape}")
        n_rows, dim = w.shape
        if w.shape[0] != s.shape[0] or dim % 32 != 0 or s.shape[1] != dim // 32:
            raise ValueError(f"Engram MXFP8 scale mismatch for {prefix}: weight={w.shape}, scale={s.shape}")
        consumed.update((w.name, s.name))
        m = _ENGRAM_LAYER_RE.search(_strip_model_prefix(prefix))
        layer = int(m.group(1)) if m else 0
        ranges = _split_row_ranges(n_rows, num_shards)
        for s_idx, (r0, r1) in enumerate(ranges):
            cnt = r1 - r0
            entry_name = normalize_engram_name(prefix, shard_idx=(s_idx if num_shards > 1 else None), quantized=False)
            d_rng = _slice_row_byte_range(w, r0, r1)
            s_rng = _slice_row_byte_range(s, r0, r1)
            data_bytes = d_rng.length
            params_bytes = s_rng.length
            params_offset = _align_up(data_bytes, 64)
            blob_bytes = _align_up(params_offset + params_bytes, APPLE_PAGE_SIZE)
            per_shard[s_idx].append(
                _PlannedEntry(
                    name=entry_name,
                    kind=TensorKind.SPARSE_TABLE,
                    dtype=DType.MXFP8_G32,
                    quant_bits=8,
                    group_size=32,
                    layer=layer,
                    expert=NO_EXPERT,
                    shape=(cnt, dim),
                    data=d_rng,
                    data_convert=None,
                    scales=s_rng,
                    biases=None,
                    data_bytes=data_bytes,
                    params_offset=params_offset,
                    params_bytes=params_bytes,
                    blob_bytes=blob_bytes,
                    params_mode="mxfp8",
                    scale_tile_shape=None,
                )
            )

    for prefix in sorted(quant_prefixes):
        w = engram_tensors[f"{prefix}.weight"]
        s = engram_tensors[f"{prefix}.scales"]
        b = engram_tensors[f"{prefix}.biases"]
        consumed.update((w.name, s.name, b.name))
        if len(w.shape) != 2 or len(s.shape) != 2 or w.shape[0] != s.shape[0]:
            raise ValueError(f"Engram quantized tensor {prefix} must be 2D, got {w.shape} / {s.shape}")
        n_rows, packed_cols = w.shape
        groups = s.shape[1]
        k = groups * group_size
        bits = (packed_cols * 32) // k
        amf_dtype = {2: DType.Q2_AFFINE, 3: DType.Q3_AFFINE, 4: DType.Q4_AFFINE}.get(bits)
        if amf_dtype is None:
            raise ValueError(f"unsupported Engram quant bits {bits} for {prefix}")
        m = _ENGRAM_LAYER_RE.search(_strip_model_prefix(prefix))
        layer = int(m.group(1)) if m else 0
        ranges = _split_row_ranges(n_rows, num_shards)
        for s_idx, (r0, r1) in enumerate(ranges):
            cnt = r1 - r0
            entry_name = normalize_engram_name(prefix, shard_idx=(s_idx if num_shards > 1 else None), quantized=True)
            d_rng = _slice_row_byte_range(w, r0, r1)
            s_rng = _slice_row_byte_range(s, r0, r1)
            b_rng = _slice_row_byte_range(b, r0, r1)
            data_bytes = d_rng.length
            params_bytes = cnt * groups * 4
            params_offset = _align_up(data_bytes, 64)
            blob_bytes = _align_up(params_offset + params_bytes, APPLE_PAGE_SIZE)
            per_shard[s_idx].append(
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
                )
            )

    for raw_name in sorted(engram_tensors):
        if raw_name in consumed:
            continue
        t = engram_tensors[raw_name]
        if len(t.shape) != 2:
            raise ValueError(f"Engram table {raw_name!r} must be 2D [num_rows, 2*D], got {t.shape}")
        n_rows, dim = t.shape
        if t.dtype == "F16":
            amf_dt, conv = (DType.BF16, "f16_to_bf16") if dtype_mode == "bf16" else (DType.F16, None)
        elif t.dtype == "BF16":
            amf_dt, conv = (DType.F16, "bf16_to_f16") if dtype_mode == "f16" else (DType.BF16, None)
        else:
            raise ValueError(f"unsupported Engram tensor dtype {t.dtype!r} for {raw_name!r}")
        m = _ENGRAM_LAYER_RE.search(_strip_model_prefix(raw_name))
        layer = int(m.group(1)) if m else 0
        ranges = _split_row_ranges(n_rows, num_shards)
        for s_idx, (r0, r1) in enumerate(ranges):
            cnt = r1 - r0
            entry_name = normalize_engram_name(raw_name, shard_idx=(s_idx if num_shards > 1 else None), quantized=False)
            d_rng = _slice_row_byte_range(t, r0, r1)
            data_bytes = cnt * dim * 2
            blob_bytes = _align_up(data_bytes, APPLE_PAGE_SIZE)
            per_shard[s_idx].append(
                _PlannedEntry(
                    name=entry_name,
                    kind=TensorKind.SPARSE_TABLE,
                    dtype=amf_dt,
                    quant_bits=0,
                    group_size=0,
                    layer=layer,
                    expert=NO_EXPERT,
                    shape=(cnt, dim),
                    data=d_rng,
                    data_convert=conv,
                    scales=None,
                    biases=None,
                    data_bytes=data_bytes,
                    params_offset=0,
                    params_bytes=0,
                    blob_bytes=blob_bytes,
                )
            )

    readers = _ShardReaders()
    all_metas: list[list[PackedTensorMeta]] = []
    try:
        for s_idx, entries in enumerate(per_shard):
            out_file = shard_files[s_idx]
            out_file.parent.mkdir(parents=True, exist_ok=True)
            dir_offset = APPLE_PAGE_SIZE
            dir_bytes = 0 if not entries else _align_up(len(entries) * ENTRY_STRUCT.size, APPLE_PAGE_SIZE)
            metas: list[PackedTensorMeta] = []
            with out_file.open("wb") as fh:
                _stream_zeros(fh, dir_offset + dir_bytes, 0)
                cursor = dir_offset + dir_bytes
                for e in entries:
                    w = _BlobWriter(fh)
                    _stream_entry(e, readers, w)
                    pad = e.blob_bytes - w.written
                    if pad > 0:
                        w.write(b"\x00" * pad)
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
                            blob_offset=cursor,
                            blob_bytes=e.blob_bytes,
                            data_bytes=e.data_bytes,
                            params_offset=e.params_offset,
                            params_bytes=e.params_bytes,
                            blob_crc32=w.crc,
                        )
                    )
                    cursor += e.blob_bytes
                dir_raw = _pack_directory(metas)
                dir_crc = _crc32(dir_raw)
                sid = (f"{model_id[:52]}-s{s_idx}" if num_shards > 1 else model_id[:63]).encode("utf-8")
                hdr = _pack_header(sid, len(metas), dir_offset, dir_bytes, dir_crc)
                fh.seek(0)
                fh.write(hdr)
                if len(hdr) < APPLE_PAGE_SIZE:
                    fh.write(b"\x00" * (APPLE_PAGE_SIZE - len(hdr)))
                if dir_bytes > 0:
                    fh.seek(dir_offset)
                    fh.write(dir_raw)
                    if len(dir_raw) < dir_bytes:
                        fh.write(b"\x00" * (dir_bytes - len(dir_raw)))
            all_metas.append(metas)
    finally:
        readers.close()

    return shard_files, all_metas


def _dequant_planned_entry_rows_f32(e: _PlannedEntry, readers: _ShardReaders) -> bytes:
    """Dequantize a 2D MXFP8_G32 or HF/GGUF MXFP4 planned entry to packed little-endian float32 bytes."""
    n_rows, k = e.shape
    src_fh = readers.get(e.data.path)
    out = bytearray()
    if e.dtype == DType.MXFP8_G32:
        if e.scales is None:
            for r in range(n_rows):
                d_raw = _read_exact(src_fh, e.data.offset + r * k, k)
                ones = b"\x7f" * (k // 32)
                vals = dequantize_mxfp8_row_f32(d_raw, ones, k)
                out.extend(struct.pack(f"<{k}f", *vals))
            return bytes(out)
        s_fh = readers.get(e.scales.path)
        kb32 = k // 32
        for r in range(n_rows):
            d_raw = _read_exact(src_fh, e.data.offset + r * k, k)
            if e.scale_tile_shape is not None:
                t_row = r // 32
                s_raw = _read_exact(s_fh, e.scales.offset + t_row * kb32, kb32)
            else:
                s_raw = _read_exact(s_fh, e.scales.offset + r * kb32, kb32)
            vals = dequantize_mxfp8_row_f32(d_raw, s_raw, k)
            out.extend(struct.pack(f"<{k}f", *vals))
        return bytes(out)
    if e.dtype == DType.GGUF_MXFP4:
        from aminfer_tools.mimo_v2 import dequantize_hf_mxfp4_row_f32

        assert e.scales is not None
        w_row_b = k // 2
        s_row_b = k // 32
        s_fh = readers.get(e.scales.path)
        for r in range(n_rows):
            d_raw = _read_exact(src_fh, e.data.offset + r * w_row_b, w_row_b)
            s_raw = _read_exact(s_fh, e.scales.offset + r * s_row_b, s_row_b)
            vals = dequantize_hf_mxfp4_row_f32(d_raw, s_raw, k)
            out.extend(struct.pack(f"<{k}f", *vals))
        return bytes(out)
    raise ValueError(f"unsupported dequant source dtype {e.dtype.name} for {e.name!r}")


def _write_backbone_stream(
    entries: Sequence[_PlannedEntry],
    output_path: pathlib.Path,
    model_id: str,
    readers: _ShardReaders,
    *,
    quantize_float_bits: int = 0,
    dequantize_quant_to_bf16: bool = False,
    group_size: int = 64,
    vision_block_dtype: str = "auto",
) -> list[PackedTensorMeta]:
    """Stream planned backbone entries to `output_path`, optionally quantizing or dequantizing 2D weights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dir_offset = APPLE_PAGE_SIZE
    dir_bytes = 0 if not entries else _align_up(len(entries) * ENTRY_STRUCT.size, APPLE_PAGE_SIZE)
    quant_kinds = {
        TensorKind.DENSE,
        TensorKind.HOT_EXPERT,
        TensorKind.COLD_EXPERT,
        TensorKind.SHARED_EXPERT,
        TensorKind.LM_HEAD,
    }

    metas: list[PackedTensorMeta] = []
    with output_path.open("wb") as fh:
        _stream_zeros(fh, dir_offset + dir_bytes, 0)
        cursor = dir_offset + dir_bytes

        for e in entries:
            is_vis = is_vision_tensor(e.name)
            vis_pinned = is_vision_pinned_float(e.name, e.shape)
            vis_block = is_vision_block_linear(e.name, e.shape)

            eff_bits = quantize_float_bits
            if is_vis:
                if vis_pinned:
                    eff_bits = 0
                elif vis_block:
                    if vision_block_dtype in ("f16", "bf16"):
                        eff_bits = 0
                    elif vision_block_dtype == "q4_affine":
                        eff_bits = 4
                    elif quantize_float_bits in (3, 4):
                        # W20 invariant: vision blocks stay in f16/bf16/q4_affine (never <= 3-bit).
                        eff_bits = 4
                    else:
                        eff_bits = 0
                else:
                    eff_bits = 0

            is_float_src = e.quant_bits == 0 and e.data.dtype in ("BF16", "F16", "F32")
            is_mxfp_src = e.dtype in (DType.MXFP8_G32, DType.GGUF_MXFP4)
            should_quant = (
                eff_bits in (3, 4)
                and e.kind in quant_kinds
                and len(e.shape) == 2
                and e.shape[1] % group_size == 0
                and (is_float_src or is_mxfp_src)
            )
            if should_quant:
                if is_float_src:
                    src_fh = readers.get(e.data.path)
                    raw = _read_exact(src_fh, e.data.offset, e.data.length)
                    raw_dt = e.data.dtype
                else:
                    raw = _dequant_planned_entry_rows_f32(e, readers)
                    raw_dt = "F32"
                code_bytes, param_bytes = _quantize_2d_affine(
                    raw, raw_dt, e.shape[0], e.shape[1], eff_bits, group_size
                )
                params_offset = _align_up(len(code_bytes), 64)
                blob_bytes = _align_up(params_offset + len(param_bytes), APPLE_PAGE_SIZE)
                w = _BlobWriter(fh)
                w.write(code_bytes)
                if params_offset > len(code_bytes):
                    w.write(b"\x00" * (params_offset - len(code_bytes)))
                w.write(param_bytes)
                if blob_bytes > w.written:
                    w.write(b"\x00" * (blob_bytes - w.written))
                amf_dt = DType.Q3_AFFINE if eff_bits == 3 else DType.Q4_AFFINE
                q_name = e.name[: -len(".weight")] if e.name.endswith(".weight") else e.name
                metas.append(
                    PackedTensorMeta(
                        name=q_name,
                        kind=e.kind,
                        dtype=amf_dt,
                        quant_bits=eff_bits,
                        group_size=group_size,
                        layer=e.layer,
                        expert=e.expert,
                        shape=e.shape,
                        blob_offset=cursor,
                        blob_bytes=blob_bytes,
                        data_bytes=len(code_bytes),
                        params_offset=params_offset,
                        params_bytes=len(param_bytes),
                        blob_crc32=w.crc,
                        layout=e.layout,
                        quant_param_kind=e.quant_param_kind,
                    )
                )
                cursor += blob_bytes
            elif dequantize_quant_to_bf16 and is_mxfp_src and len(e.shape) == 2 and e.kind != TensorKind.SPARSE_TABLE:
                f32_raw = _dequant_planned_entry_rows_f32(e, readers)
                n_elems = e.shape[0] * e.shape[1]
                bf16_raw = f32_to_bf16_bytes(list(struct.unpack(f"<{n_elems}f", f32_raw)))
                blob_bytes = _align_up(len(bf16_raw), APPLE_PAGE_SIZE)
                w = _BlobWriter(fh)
                w.write(bf16_raw)
                if blob_bytes > w.written:
                    w.write(b"\x00" * (blob_bytes - w.written))
                metas.append(
                    PackedTensorMeta(
                        name=e.name,
                        kind=e.kind,
                        dtype=DType.BF16,
                        quant_bits=0,
                        group_size=0,
                        layer=e.layer,
                        expert=e.expert,
                        shape=e.shape,
                        blob_offset=cursor,
                        blob_bytes=blob_bytes,
                        data_bytes=len(bf16_raw),
                        params_offset=0,
                        params_bytes=0,
                        blob_crc32=w.crc,
                        layout=TensorLayout.ROW_MAJOR,
                        quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                    )
                )
                cursor += blob_bytes
            else:
                w = _BlobWriter(fh)
                _stream_entry(e, readers, w)
                pad = e.blob_bytes - w.written
                if pad > 0:
                    w.write(b"\x00" * pad)
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
                        blob_offset=cursor,
                        blob_bytes=e.blob_bytes,
                        data_bytes=e.data_bytes,
                        params_offset=e.params_offset,
                        params_bytes=e.params_bytes,
                        blob_crc32=w.crc,
                        layout=e.layout,
                        quant_param_kind=e.quant_param_kind,
                    )
                )
                cursor += e.blob_bytes

        dir_raw = _pack_directory(metas)
        dir_crc = _crc32(dir_raw)
        hdr = _pack_header(model_id.encode("utf-8")[:63], len(metas), dir_offset, dir_bytes, dir_crc)
        fh.seek(0)
        fh.write(hdr)
        if len(hdr) < APPLE_PAGE_SIZE:
            fh.write(b"\x00" * (APPLE_PAGE_SIZE - len(hdr)))
        if dir_bytes > 0:
            fh.seek(dir_offset)
            fh.write(dir_raw)
            if len(dir_raw) < dir_bytes:
                fh.write(b"\x00" * (dir_bytes - len(dir_raw)))

    return metas


def _estimate_planned_wired_bytes(
    entries: Sequence[_PlannedEntry],
    quantize_float_bits: int,
    group_size: int,
) -> tuple[int, int, int]:
    """Return `(total_wired, non_expert_wired, routed_expert_wired)` for planned entries."""
    quant_kinds = {
        TensorKind.DENSE,
        TensorKind.HOT_EXPERT,
        TensorKind.COLD_EXPERT,
        TensorKind.SHARED_EXPERT,
        TensorKind.LM_HEAD,
    }
    non_expert = 0
    routed_expert = 0
    for e in entries:
        if e.kind in (TensorKind.COLD_EXPERT, TensorKind.SPARSE_TABLE):
            continue
        if (
            quantize_float_bits in (3, 4)
            and (e.quant_bits == 0 or e.dtype in (DType.MXFP8_G32, DType.GGUF_MXFP4))
            and e.kind in quant_kinds
            and len(e.shape) == 2
            and e.shape[1] % group_size == 0
        ):
            rows, cols = e.shape
            groups = cols // group_size
            code_b = rows * ((cols * quantize_float_bits) // 8)
            param_b = rows * groups * 4
            sz = _align_up(_align_up(code_b, 64) + param_b, APPLE_PAGE_SIZE)
        else:
            sz = e.blob_bytes

        if e.kind == TensorKind.HOT_EXPERT:
            routed_expert += sz
        else:
            non_expert += sz
    return non_expert + routed_expert, non_expert, routed_expert


def convert_deepseek41_checkpoint(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str = "deepseek41",
    target_dtype: str = "auto",
    dtype_mode: str = "auto",
    group_size: int = 64,
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    use_rco: bool = True,
    manifest_path: pathlib.Path | None = None,
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
    heat_profile: pathlib.Path | None = None,
    cold_experts_per_layer: int | None = None,
    sme2_pack: str = "none",
    lut16_sme2: bool = False,
    tie_word_embeddings: bool | None = None,
    exclude_tensors: Iterable[str] = (),
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
    engram_output: pathlib.Path | None = None,
    engram_shards: int = 1,
    dspark_output: pathlib.Path | None = None,
    config_output: pathlib.Path | None = None,
    target_tier: str = "auto",
    host_ram_gib: float | None = None,
    world_size: int = 1,
    rank_dir: pathlib.Path | None = None,
    partition_strategy: str = "contiguous_by_rank",
    allow_tier_override: bool = False,
    encoder_cold_decoder_hot: bool = False,
    dspark_quant_profile: str = "uniform",
    dspark_cold_experts_per_layer: int | None = None,
    dspark_config_output: pathlib.Path | None = None,
    workers: int = 1,
) -> DeepSeek41ConvertReport:
    """Streaming multi-shard safetensors converter for `DeepSeek-V4.1-Flash-1017` (`pack-deepseek41`).

    Supports `q3_affine`, `q4_affine`, `bf16`, and `--quant-profile gsq-rco`,
    W20 vision-tower packing (`vision.*` + `aligner.*` + `image_*` via default
    include, `--strip-vision`, or `--vision-sidecar vision.amf`),
    splits external `engram.*` tables into sharded `TensorKind.SPARSE_TABLE`
    `.amf` files, extracts optional `dspark.*` weights into `dspark.amf`, and
    validates ADR-0015 hardware-tier wired memory constraints before writing.
    """
    if target_dtype not in ("auto",) + VALID_DEEPSEEK41_DTYPES:
        raise ValueError(f"target_dtype must be 'auto' or one of {VALID_DEEPSEEK41_DTYPES}, got {target_dtype!r}")
    if dtype_mode not in DTYPE_MODES:
        raise ValueError(f"dtype_mode must be one of {DTYPE_MODES}, got {dtype_mode!r}")
    if quant_profile not in QUANT_PROFILES:
        raise ValueError(f"quant_profile must be one of {QUANT_PROFILES}, got {quant_profile!r}")
    if sme2_pack not in SME2_PACK_MODES:
        raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
    if vision_block_dtype not in VISION_BLOCK_DTYPES:
        raise ValueError(f"vision_block_dtype must be one of {VISION_BLOCK_DTYPES}, got {vision_block_dtype!r}")
    if group_size != 64:
        raise ValueError(f"aminfer v0 supports group_size=64 only, got {group_size}")
    if engram_shards < 1:
        raise ValueError(f"engram_shards must be >= 1, got {engram_shards}")

    effective_cold: set[tuple[int, int]] = set(cold_experts)
    if heat_profile is not None:
        effective_cold.update(
            load_cold_experts_from_heat_profile(
                pathlib.Path(heat_profile),
                cold_experts_per_layer=cold_experts_per_layer,
            )
        )
    elif cold_experts_per_layer is not None:
        raise ValueError("cold_experts_per_layer requires heat_profile in convert_deepseek41_checkpoint")

    resolved = resolve_checkpoint(pathlib.Path(input_path))
    tie = _load_tie_flag(resolved.config_path) if tie_word_embeddings is None else tie_word_embeddings
    src_config: dict[str, object] = {}
    if resolved.config_path is not None and resolved.config_path.is_file():
        loaded = json.loads(resolved.config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            src_config = loaded

    # Fast-fail hardware tier validation when config.json advertises full-scale DeepSeek-V4.1 geometry.
    text_cfg = _effective_deepseek_text_config(src_config)
    top_mt = str(src_config.get("model_type", text_cfg.get("model_type", ""))).lower()
    cfg_num_experts = int(text_cfg.get("num_experts", text_cfg.get("n_routed_experts", 0)) or 0)
    probe_total_l = int(
        text_cfg.get(
            "num_hidden_layers",
            int(text_cfg.get("num_encoder_layers", 20) or 20)
            + int(text_cfg.get("num_decoder_layers", 20) or 20),
        )
        or 40
    )
    probe_enc_l, probe_dec_l = _infer_enc_dec_layers(text_cfg, probe_total_l)
    if top_mt in _DEEPSEEK41_MODEL_TYPES and cfg_num_experts >= 64:
        h_sz = int(text_cfg.get("hidden_size", 5120) or 5120)
        n_exp = cfg_num_experts
        moe_i = int(text_cfg.get("moe_intermediate_size", 2304) or 2304)
        eff_dt = target_dtype if target_dtype in VALID_DEEPSEEK41_DTYPES else "q3_affine"
        top_k = int(text_cfg.get("num_experts_per_tok", 6) or 6)
        n_shared = int(text_cfg.get("n_shared_experts", 1 if text_cfg.get("has_shared_expert", True) else 0) or 0)
        kv_rank = int(text_cfg.get("kv_lora_rank", text_cfg.get("head_dim", 512)) or 512)
        q_rank = int(text_cfg.get("q_lora_rank", 1280) or 1280)
        probe_cfg = DeepSeek41Config(
            model_id=model_id[:63],
            num_encoder_layers=probe_enc_l,
            num_decoder_layers=probe_dec_l,
            hidden_size=h_sz,
            vocab_size=int(text_cfg.get("vocab_size", 129280) or 129280),
            num_attention_heads=int(text_cfg.get("num_attention_heads", 64) or 64),
            encoder_moe_top_k=int(text_cfg.get("encoder_moe_top_k", top_k) or top_k),
            decoder_moe_top_k=int(text_cfg.get("decoder_moe_top_k", top_k) or top_k),
            num_experts=n_exp,
            has_shared_expert=n_shared > 0,
            moe_intermediate_size=moe_i,
            shared_expert_intermediate_size=int(text_cfg.get("shared_expert_intermediate_size", moe_i) or moe_i),
            kv_lora_rank=kv_rank,
            q_lora_rank=q_rank,
            mhc_num_streams=int(text_cfg.get("mhc_num_streams", text_cfg.get("hc_mult", 4)) or 4),
            dtype=eff_dt,
            world_size=world_size,
        )
        validate_config_hardware_tier(
            probe_cfg,
            target_tier=target_tier,
            host_ram_gib=host_ram_gib,
            world_size=world_size,
            allow_override=allow_tier_override,
            all_experts_cold=all_experts_cold,
            encoder_cold_decoder_hot=encoder_cold_decoder_hot,
            decoder_hot_fraction=0.0,
        )

    all_tensors = _collect_tensors(resolved.shard_paths)
    vision_mode = resolve_vision_mode(
        all_tensors.keys(),
        include_vision=include_vision,
        strip_vision=strip_vision,
        vision_sidecar=vision_sidecar,
    )
    stripped_vision = 0
    included_vision = 0
    vision_tensors: dict[str, _ShardTensor] = {}
    engram_tensors: dict[str, _ShardTensor] = {}
    dspark_tensors: dict[str, _ShardTensor] = {}
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
        if is_engram_tensor(raw_name):
            engram_tensors[raw_name] = st
            continue
        if is_dspark_tensor(raw_name):
            dspark_tensors[raw_name] = st
            continue
        backbone_tensors[raw_name] = st

    eff_dtype_mode = "bf16" if target_dtype == "bf16" else dtype_mode
    dequantize_bf16 = target_dtype == "bf16"
    quantize_bits = {"q3_affine": 3, "q4_affine": 4}.get(target_dtype, 0)
    if quant_profile == "gsq-rco":
        # Pack uniform first, then delegate mixed-bit quantization to gsq_convert.
        quantize_bits = 0

    backbone_plan = plan_amf_entries(
        backbone_tensors,
        group_size=group_size,
        dtype_mode=eff_dtype_mode,
        cold_experts=effective_cold,
        all_experts_cold=all_experts_cold,
        tie_word_embeddings=tie,
        exclude_tensors=exclude_tensors,
        sme2_pack=sme2_pack if quantize_bits == 0 else "none",
        strip_vision=False,
    )

    if encoder_cold_decoder_hot and not all_experts_cold:
        # Step 1: tag all Causal Encoder experts (layers 0 .. enc_l-1) as cold NVMe experts.
        for e_entry in backbone_plan:
            if e_entry.expert != NO_EXPERT and 0 <= e_entry.layer < probe_enc_l:
                effective_cold.add((e_entry.layer, e_entry.expert))
        backbone_plan = plan_amf_entries(
            backbone_tensors,
            group_size=group_size,
            dtype_mode=eff_dtype_mode,
            cold_experts=effective_cold,
            all_experts_cold=all_experts_cold,
            tie_word_embeddings=tie,
            exclude_tensors=exclude_tensors,
            sme2_pack=sme2_pack if quantize_bits == 0 else "none",
            strip_vision=False,
        )
        # Step 2: if hot Decoder experts still exceed target wired budget, demote tail Decoder experts to cold.
        tot_w, non_exp_w, _ = _estimate_planned_wired_bytes(backbone_plan, quantize_bits, group_size)
        tier_key_eff = "24gb-m4" if target_tier == "auto" and (host_ram_gib is None or host_ram_gib <= 32.0) else target_tier
        tier_budget_b = (
            int(host_ram_gib * 0.75 * (1024**3))
            if host_ram_gib is not None
            else TIER_WIRED_BUDGET_BYTES.get(tier_key_eff, TIER_WIRED_BUDGET_BYTES["24gb-m4"])
        )
        if non_exp_w <= tier_budget_b and tot_w > tier_budget_b:
            dec_expert_sizes: dict[tuple[int, int], int] = {}
            for e_entry in backbone_plan:
                if e_entry.kind == TensorKind.HOT_EXPERT and e_entry.expert != NO_EXPERT:
                    dec_expert_sizes[(e_entry.layer, e_entry.expert)] = (
                        dec_expert_sizes.get((e_entry.layer, e_entry.expert), 0) + e_entry.blob_bytes
                    )
            cur_wired = tot_w
            for pair in sorted(dec_expert_sizes.keys(), key=lambda p: (-p[1], p[0])):
                if cur_wired <= tier_budget_b:
                    break
                effective_cold.add(pair)
                cur_wired -= dec_expert_sizes[pair]
            backbone_plan = plan_amf_entries(
                backbone_tensors,
                group_size=group_size,
                dtype_mode=eff_dtype_mode,
                cold_experts=effective_cold,
                all_experts_cold=all_experts_cold,
                tie_word_embeddings=tie,
                exclude_tensors=exclude_tensors,
                sme2_pack=sme2_pack if quantize_bits == 0 else "none",
                strip_vision=False,
            )

    total_wired, non_exp_wired, exp_wired = _estimate_planned_wired_bytes(backbone_plan, quantize_bits, group_size)
    tier_check = evaluate_hardware_tier(
        total_wired,
        target_tier=target_tier,
        host_ram_gib=host_ram_gib,
        world_size=world_size,
        non_expert_bytes=non_exp_wired,
        routed_expert_bytes=exp_wired,
        allow_override=allow_tier_override,
    )

    out_path = pathlib.Path(output_path)

    # 0. Pack standalone vision sidecar when requested.
    vision_amf_path: pathlib.Path | None = None
    vision_metas: list[PackedTensorMeta] = []
    if vision_mode == "sidecar" and vision_sidecar is not None and vision_tensors:
        vision_amf_path = pathlib.Path(vision_sidecar)
        eff_vis_block_dt = vision_block_dtype
        if eff_vis_block_dt == "auto" and target_dtype in ("q3_affine", "q4_affine"):
            eff_vis_block_dt = "q4_affine"
        vision_metas = pack_vision_tower_amf(
            input_path,
            vision_amf_path,
            model_id=f"{model_id[:55]}-vision",
            dtype_mode=eff_dtype_mode,
            quant_profile=quant_profile,
            vision_block_dtype=eff_vis_block_dt,
            group_size=group_size,
            use_rco=use_rco,
            workers=workers,
        )

    # 1. Pack external Engram SPARSE_TABLE shards when present.
    engram_paths: list[pathlib.Path] = []
    engram_metas: list[list[PackedTensorMeta]] = []
    if engram_tensors:
        eff_eng_out = (
            pathlib.Path(engram_output) if engram_output is not None else out_path.with_name(out_path.stem + "-engram.amf")
        )
        engram_paths, engram_metas = _pack_engram_shards_from_safetensors(
            engram_tensors,
            eff_eng_out,
            num_shards=engram_shards,
            model_id=f"{model_id[:50]}-engram",
            group_size=group_size,
            dtype_mode=eff_dtype_mode,
        )

    # 2. Pack optional DSpark head when present (M7 single-pass or real-HF 3L DSpark `mtp.0..2.*`).
    dspark_path: pathlib.Path | None = None
    dspark_cfg_out_path: pathlib.Path | None = None
    dspark_metas: list[PackedTensorMeta] = []
    if dspark_tensors:
        eff_ds_out = (
            pathlib.Path(dspark_output) if dspark_output is not None else out_path.with_name(out_path.stem + "-dspark.amf")
        )
        remapped: dict[str, _ShardTensor] = {}
        ds_exp_ids: set[int] = set()
        ds_layer_ids: set[int] = set()
        ds_exp_re = re.compile(r"\.experts\.(\d+)\.")
        ds_mtp_l_re = re.compile(r"^mtp\.(\d+)\.")
        for raw_name, st in dspark_tensors.items():
            clean = _normalize_name(strip_dspark_prefix(raw_name))
            remapped[clean] = _ShardTensor(
                name=clean,
                dtype=st.dtype,
                shape=st.shape,
                offset=st.offset,
                length=st.length,
                path=st.path,
            )
            em = ds_exp_re.search(clean)
            if em is not None:
                ds_exp_ids.add(int(em.group(1)))
            lm = ds_mtp_l_re.match(clean)
            if lm is not None:
                ds_layer_ids.add(int(lm.group(1)))
        ds_cold: set[tuple[int, int]] = set()
        if dspark_cold_experts_per_layer is not None and dspark_cold_experts_per_layer > 0 and ds_exp_ids:
            max_ds_e = max(ds_exp_ids)
            cutoff = max_ds_e + 1 - dspark_cold_experts_per_layer
            for l_i in ds_layer_ids or {0}:
                for e_i in ds_exp_ids:
                    if e_i >= cutoff:
                        ds_cold.add((l_i, e_i))
        ds_plan = plan_amf_entries(
            remapped,
            group_size=group_size,
            dtype_mode=eff_dtype_mode,
            cold_experts=ds_cold,
            all_experts_cold=all_experts_cold,
            tie_word_embeddings=False,
            sme2_pack="none",
            strip_vision=False,
        )
        ds_quant_bits = quantize_bits if (dspark_quant_profile == "uniform" and any(e.name.startswith("mtp.") for e in ds_plan)) else 0
        readers_ds = _ShardReaders()
        try:
            dspark_metas = _write_backbone_stream(
                ds_plan,
                eff_ds_out,
                f"{model_id[:52]}-dspark",
                readers_ds,
                quantize_float_bits=ds_quant_bits,
                dequantize_quant_to_bf16=dequantize_bf16 and any(e.name.startswith("mtp.") for e in ds_plan),
                group_size=group_size,
            )
        finally:
            readers_ds.close()
        if dspark_quant_profile == "gsq-rco" and any(m.name.startswith("mtp.") for m in dspark_metas):
            eff_ds_bits = 3.50 if target_bits is None else float(target_bits)
            dspark_metas = gsq_convert_with_vision(
                eff_ds_out,
                eff_ds_out,
                target_bits=eff_ds_bits,
                model_id=f"{model_id[:52]}-dspark",
                group_size=group_size,
                dtype_mode=eff_dtype_mode,
                cold_experts=ds_cold,
                all_experts_cold=all_experts_cold,
                sme2_pack=sme2_pack,
                lut16_sme2=lut16_sme2,
                use_rco=use_rco,
                workers=workers,
                strip_vision=True,
            )
        dspark_path = eff_ds_out
        if (
            dspark_config_output is not None
            or config_output is not None
            or dspark_output is not None
            or any(m.name.startswith("mtp.") for m in dspark_metas)
        ):
            dspark_cfg_out_path = (
                pathlib.Path(dspark_config_output)
                if dspark_config_output is not None
                else eff_ds_out.with_name(
                    "dspark_config.json" if eff_ds_out.name == "dspark.amf" else f"{eff_ds_out.stem}_config.json"
                )
            )
            is_3l_mtp = any(m.name.startswith("mtp.") for m in dspark_metas)
            dspark_cfg_obj = {
                "draft_format": "dspark_3l_mtp" if is_3l_mtp else "dspark",
                "amf_file": dspark_path.name,
                "num_nextn_predict_layers": int(text_cfg.get("num_nextn_predict_layers", len(ds_layer_ids) or 1) or 1),
                "dspark_block_size": int(text_cfg.get("dspark_block_size", text_cfg.get("dspark_block_width", 5)) or 5),
                "dspark_target_layer_ids": list(text_cfg.get("dspark_target_layer_ids", []))
                if isinstance(text_cfg.get("dspark_target_layer_ids"), list)
                else [],
                "dspark_n_routed_experts": int(text_cfg.get("dspark_n_routed_experts", len(ds_exp_ids)) or 0),
                "dspark_routed_top_k": int(text_cfg.get("dspark_routed_top_k", 3) or 3),
                "hidden_size": int(text_cfg.get("hidden_size", 5120) or 5120),
                "vocab_size": int(text_cfg.get("vocab_size", 129280) or 129280),
                "tensor_count": len(dspark_metas),
                "tensor_names_max_bytes": max((len(m.name.encode("utf-8")) for m in dspark_metas), default=0),
            }
            dspark_cfg_out_path.parent.mkdir(parents=True, exist_ok=True)
            dspark_cfg_out_path.write_text(json.dumps(dspark_cfg_obj, indent=2) + "\n", encoding="utf-8")

    # 3. Stream backbone .amf (+ optional GSQ+RCO pass).
    readers = _ShardReaders()
    try:
        backbone_metas = _write_backbone_stream(
            backbone_plan,
            out_path,
            model_id,
            readers,
            quantize_float_bits=quantize_bits,
            dequantize_quant_to_bf16=dequantize_bf16,
            group_size=group_size,
            vision_block_dtype=vision_block_dtype,
        )
    finally:
        readers.close()

    if quant_profile == "gsq-rco":
        eff_target = 3.25 if target_bits is None else float(target_bits)
        backbone_metas = gsq_convert_with_vision(
            out_path,
            out_path,
            target_bits=eff_target,
            model_id=model_id,
            group_size=group_size,
            dtype_mode=eff_dtype_mode,
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
        )

    # 4. Optional multi-rank expert-parallel shard export (`world_size > 1`).
    part_report: ExpertPartitionReport | None = None
    if world_size > 1:
        eff_rank_dir = pathlib.Path(rank_dir) if rank_dir is not None else out_path.parent / f"{out_path.stem}-shards"
        part_report = partition_deepseek41_amf_shards(
            out_path,
            world_size,
            eff_rank_dir,
            strategy=partition_strategy,
        )

    # 5. Emit canonical `deepseek41` config.json metadata when requested or source config exists.
    cfg_out_path: pathlib.Path | None = None
    if config_output is not None or src_config:
        cfg_out_path = pathlib.Path(config_output) if config_output is not None else out_path.with_name("config.json")
        out_cfg = dict(src_config)
        out_cfg.setdefault("model_type", "deepseek41")
        out_cfg.setdefault("backbone_kind", "causal_enc_dec")
        out_cfg.setdefault("residual_kind", "mhc_single_pass")
        out_cfg.setdefault("mhc_num_streams", 4)
        out_cfg.setdefault("memory_lookup_kind", "engram")
        out_cfg.setdefault("fp4_kv", True)
        out_cfg["stripped_vision_tensors"] = stripped_vision
        if vision_amf_path is not None:
            vis_summary = extract_vision_config_summary(vision_metas, src_config=src_config)
            vis_summary["mode"] = "sidecar"
            vis_summary["amf_file"] = vision_amf_path.name
            out_cfg["vision"] = vis_summary
        elif included_vision > 0:
            vis_summary = extract_vision_config_summary(backbone_metas, src_config=src_config)
            vis_summary["mode"] = "embedded"
            out_cfg["vision"] = vis_summary
        if engram_paths:
            out_cfg["engram_files"] = [p.name for p in engram_paths]
        if dspark_path is not None:
            ds_obj = dict(out_cfg.get("dspark", {})) if isinstance(out_cfg.get("dspark"), dict) else {}
            ds_obj["enabled"] = True
            ds_obj["amf_file"] = dspark_path.name
            out_cfg["dspark"] = ds_obj
        out_cfg["hardware_tier"] = {
            "target_tier": tier_check.target_tier,
            "world_size": tier_check.world_size,
            "total_wired_bytes": tier_check.total_wired_bytes,
            "per_rank_wired_bytes": tier_check.per_rank_wired_bytes,
            "per_rank_budget_bytes": tier_check.per_rank_budget_bytes,
            "fits": tier_check.fits,
            "encoder_cold_decoder_hot": encoder_cold_decoder_hot,
            "cold_experts_count": len(effective_cold),
        }
        if dspark_cfg_out_path is not None and isinstance(out_cfg.get("dspark"), dict):
            out_cfg["dspark"]["config_file"] = dspark_cfg_out_path.name
        if part_report is not None:
            out_cfg["expert_parallel"] = {
                "world_size": part_report.world_size,
                "strategy": part_report.strategy,
                "cluster_manifest": str(part_report.manifest_path),
                "local_expert_source": "local_wired_ram",
                "remote_expert_source": "thunderbolt_peer",
            }
        cfg_out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_out_path.write_text(json.dumps(out_cfg, indent=2) + "\n", encoding="utf-8")

    return DeepSeek41ConvertReport(
        backbone_amf=out_path,
        backbone_metas=backbone_metas,
        engram_paths=engram_paths,
        engram_metas=engram_metas,
        dspark_amf=dspark_path,
        dspark_metas=dspark_metas,
        config_path=cfg_out_path,
        partition_report=part_report,
        tier_check=tier_check,
        stripped_vision_tensors=stripped_vision,
        vision_amf=vision_amf_path,
        vision_metas=vision_metas,
        included_vision_tensors=included_vision,
        dspark_config_path=dspark_cfg_out_path,
    )


@dataclass(frozen=True)
class DeepSeek41MetadataReport:
    """Offline metadata-only dry-run report for HF `DeepSeek-V4.1-Flash-1017` (`deepseek_v41`) repos."""

    metadata_dir: pathlib.Path
    model_type: str
    architectures: tuple[str, ...]
    num_hidden_layers: int
    num_encoder_layers: int
    num_decoder_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    kv_latent_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    num_routed_experts: int
    num_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    mhc_num_streams: int
    sliding_window: int
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    engram_layer_ids: tuple[int, ...]
    engram_total_embeddings: int
    engram_n_heads: int
    engram_head_dim: int
    dspark_layers: int
    dspark_block_size: int
    dspark_n_routed_experts: int
    quantization_config: dict[str, object]
    kv_bytes_per_token_bf16: int
    kv_bytes_per_token_fp4: int
    total_index_entries: int
    shard_file_count: int
    index_total_size_bytes: int
    backbone_tensor_count: int
    vision_tensor_count: int
    engram_tensor_count: int
    dspark_mtp_tensor_count: int
    routed_expert_tensor_count: int
    routed_expert_ids_detected: int
    shared_expert_tensor_count: int
    mhc_tensor_count: int
    csa2_indexer_tensor_count: int
    quant_scale_tensor_count: int
    long_backbone_names: tuple[str, ...]
    estimated_backbone_q3_bytes: int
    estimated_non_expert_q3_bytes: int
    estimated_routed_expert_q3_bytes: int
    estimated_backbone_q4_bytes: int
    tier_admission: dict[str, dict[str, object]]
    tokenizer_present: bool
    header_summary_verified: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "arch": "deepseek41",
            "metadata_dir": str(self.metadata_dir),
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "num_hidden_layers": self.num_hidden_layers,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "kv_latent_dim": self.kv_latent_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "q_lora_rank": self.q_lora_rank,
            "o_lora_rank": self.o_lora_rank,
            "o_groups": self.o_groups,
            "num_routed_experts": self.num_routed_experts,
            "num_shared_experts": self.num_shared_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "mhc_num_streams": self.mhc_num_streams,
            "sliding_window": self.sliding_window,
            "kv_source_layer_ids": list(self.kv_source_layer_ids),
            "index_source_layer_ids": list(self.index_source_layer_ids),
            "engram_layer_ids": list(self.engram_layer_ids),
            "engram_total_embeddings": self.engram_total_embeddings,
            "engram_n_heads": self.engram_n_heads,
            "engram_head_dim": self.engram_head_dim,
            "dspark_layers": self.dspark_layers,
            "dspark_block_size": self.dspark_block_size,
            "dspark_n_routed_experts": self.dspark_n_experts if hasattr(self, "dspark_n_experts") else self.dspark_n_routed_experts,
            "quantization_config": dict(self.quantization_config),
            "kv_bytes_per_token_bf16": self.kv_bytes_per_token_bf16,
            "kv_bytes_per_token_fp4": self.kv_bytes_per_token_fp4,
            "total_index_entries": self.total_index_entries,
            "shard_file_count": self.shard_file_count,
            "index_total_size_bytes": self.index_total_size_bytes,
            "tensor_buckets": {
                "backbone": self.backbone_tensor_count,
                "vision_strippable": self.vision_tensor_count,
                "engram_nvme_table": self.engram_tensor_count,
                "dspark_mtp_head": self.dspark_mtp_tensor_count,
                "routed_expert_tensors": self.routed_expert_tensor_count,
                "routed_expert_ids_detected": self.routed_expert_ids_detected,
                "shared_expert_tensors": self.shared_expert_tensor_count,
                "mhc_tensors": self.mhc_tensor_count,
                "csa2_indexer_tensors": self.csa2_indexer_tensor_count,
                "quant_scale_tensors": self.quant_scale_tensor_count,
            },
            "long_backbone_names_over_47b": len(self.long_backbone_names),
            "long_backbone_name_samples": list(self.long_backbone_names[:8]),
            "estimated_backbone_q3_bytes": self.estimated_backbone_q3_bytes,
            "estimated_non_expert_q3_bytes": self.estimated_non_expert_q3_bytes,
            "estimated_routed_expert_q3_bytes": self.estimated_routed_expert_q3_bytes,
            "estimated_backbone_q4_bytes": self.estimated_backbone_q4_bytes,
            "tier_admission": self.tier_admission,
            "tokenizer_present": self.tokenizer_present,
            "header_summary_verified": self.header_summary_verified,
        }


def validate_hf_metadata_dir(
    metadata_dir: pathlib.Path,
    *,
    host_ram_gib: float | None = None,
) -> DeepSeek41MetadataReport:
    """Validate a Hugging Face `DeepSeek-V4.1-Flash-1017` (`deepseek_v41`) metadata directory offline.

    Requires only `config.json` and `model.safetensors.index.json` (no `.safetensors` shards
    needed on disk). Maps every `weight_map` key, reports CED/MLA/CSA2/mHC/Engram/DSpark
    parameters, KV bytes/token, and multi-tier + TB5 cluster wired admission plans.
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
    cfg = _effective_deepseek_text_config(raw_cfg)

    idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
    if not isinstance(idx_obj, dict) or not isinstance(idx_obj.get("weight_map"), dict):
        raise ValueError("model.safetensors.index.json must contain a weight_map object")
    weight_map: dict[str, str] = idx_obj["weight_map"]
    idx_meta = idx_obj.get("metadata") if isinstance(idx_obj.get("metadata"), dict) else {}
    total_size = int(float(idx_meta.get("total_size", 0) or 0))
    shard_files = set(weight_map.values())

    vision_cnt = 0
    engram_cnt = 0
    dspark_cnt = 0
    backbone_cnt = 0
    routed_exp_cnt = 0
    shared_exp_cnt = 0
    mhc_cnt = 0
    csa2_cnt = 0
    scale_cnt = 0
    expert_ids_seen: set[int] = set()
    long_names: list[str] = []

    exp_re = re.compile(r"\.experts\.(\d+)\.")
    for raw_name in sorted(weight_map.keys()):
        if is_vision_tensor(raw_name):
            vision_cnt += 1
            continue
        if is_engram_tensor(raw_name):
            engram_cnt += 1
            continue
        if is_dspark_tensor(raw_name):
            dspark_cnt += 1
            ds_clean = _normalize_name(strip_dspark_prefix(raw_name))
            if len(ds_clean.encode("utf-8")) > 47:
                long_names.append(f"dspark:{ds_clean}")
            continue
        norm = _strip_model_prefix(raw_name)
        backbone_cnt += 1
        if len(norm.encode("utf-8")) > 47:
            long_names.append(norm)
        lower = norm.lower()
        if lower.endswith(".scale"):
            scale_cnt += 1
        em = exp_re.search(lower)
        if em is not None:
            routed_exp_cnt += 1
            expert_ids_seen.add(int(em.group(1)))
        elif ".shared_expert" in lower:
            shared_exp_cnt += 1
        if ".hc_" in lower or ".mhc." in lower:
            mhc_cnt += 1
        if ".indexer." in lower or ".compressor." in lower or "csa2" in lower:
            csa2_cnt += 1

    total_layers = int(
        cfg.get(
            "num_hidden_layers",
            int(cfg.get("num_encoder_layers", 0) or 0) + int(cfg.get("num_decoder_layers", 0) or 0),
        )
        or 40
    )
    enc_l, dec_l = _infer_enc_dec_layers(cfg, total_layers)
    hidden_size = int(cfg.get("hidden_size", 5120) or 5120)
    vocab_size = int(cfg.get("vocab_size", 129280) or 129280)
    num_heads = int(cfg.get("num_attention_heads", 64) or 64)
    num_kv_heads = int(cfg.get("num_key_value_heads", 1) or 1)
    kv_latent_dim = int(cfg.get("kv_lora_rank", cfg.get("head_dim", 512)) or 512)
    qk_rope_dim = int(cfg.get("qk_rope_head_dim", 64) or 64)
    q_lora_rank = int(cfg.get("q_lora_rank", 1280) or 1280)
    o_lora_rank = int(cfg.get("o_lora_rank", 1024) or 1024)
    o_groups = int(cfg.get("o_groups", 8) or 8)
    num_routed_experts = int(cfg.get("n_routed_experts", cfg.get("num_experts", len(expert_ids_seen))) or 0)
    num_shared_experts = int(cfg.get("n_shared_experts", 1 if cfg.get("has_shared_expert", True) else 0) or 0)
    num_experts_per_tok = int(cfg.get("num_experts_per_tok", cfg.get("encoder_moe_top_k", 6)) or 6)
    moe_i = int(cfg.get("moe_intermediate_size", 2304) or 2304)
    mhc_streams = int(cfg.get("hc_mult", cfg.get("mhc_num_streams", 4)) or 4)
    swa = int(cfg.get("sliding_window", cfg.get("swa_window", 128)) or 128)
    raw_kv_src = cfg.get("kv_source_layer_ids")
    kv_source_ids = tuple(int(x) for x in raw_kv_src) if isinstance(raw_kv_src, list) else ()
    raw_idx_src = cfg.get("index_source_layer_ids")
    index_source_ids = tuple(int(x) for x in raw_idx_src) if isinstance(raw_idx_src, list) else ()
    raw_eng_ids = cfg.get("engram_layer_ids")
    engram_layer_ids = tuple(int(x) for x in raw_eng_ids) if isinstance(raw_eng_ids, list) else ()
    raw_eng_rows = cfg.get("engram_num_embeddings")
    engram_total_embeddings = (
        sum(int(x) for x in raw_eng_rows)
        if isinstance(raw_eng_rows, list)
        else int(cfg.get("engram_num_rows", 0) or 0)
    )
    engram_n_heads = int(cfg.get("engram_n_heads", cfg.get("engram_num_heads", 8)) or 8)
    engram_head_dim = int(cfg.get("engram_head_dim", 256) or 256)
    dspark_layers = int(cfg.get("num_nextn_predict_layers", 3 if dspark_cnt > 0 else 0) or 0)
    dspark_block_size = int(cfg.get("dspark_block_size", cfg.get("dspark_block_width", 5)) or 5)
    dspark_n_experts = int(cfg.get("dspark_n_routed_experts", 0) or 0)
    quant_cfg = raw_cfg.get("quantization_config") if isinstance(raw_cfg.get("quantization_config"), dict) else {}

    kv_slots = len(kv_source_ids) if kv_source_ids else 1
    kv_bytes_per_token_bf16 = kv_slots * (kv_latent_dim + qk_rope_dim) * 2
    fp4_per_slot = kv_latent_dim // 2 + kv_latent_dim // 16 + 26
    kv_bytes_per_token_fp4 = kv_slots * fp4_per_slot

    # Estimate backbone bytes under aminfer q3_affine (3.5 bpw) and q4_affine (4.5 bpw), excluding NVMe Engram.
    def _est_backbone(bpw: float) -> tuple[int, int, int]:
        h = hidden_size
        # Attention + low-rank Q/O + KV + compressor/indexer + shared expert + mHC + router
        q_elems = q_lora_rank * h + num_heads * (kv_latent_dim + qk_rope_dim) * q_lora_rank
        kv_elems = (kv_latent_dim + qk_rope_dim) * h
        o_elems = (o_lora_rank * num_heads * kv_latent_dim) // max(1, o_groups) + h * o_lora_rank
        shared_elems = num_shared_experts * 3 * moe_i * h
        mhc_elems = 2 * (mhc_streams * h + mhc_streams * mhc_streams + mhc_streams)
        router_elems = num_routed_experts * h
        per_layer_non_exp = int((q_elems + kv_elems + o_elems + shared_elems) * bpw) + (mhc_elems + router_elems + 4 * h) * 2
        embed_bytes = vocab_size * h * 2
        lm_head_bytes = 0 if bool(cfg.get("tie_word_embeddings", False)) else int(vocab_size * h * bpw)
        non_exp = total_layers * per_layer_non_exp + embed_bytes + lm_head_bytes
        routed_exp = int(total_layers * num_routed_experts * 3 * moe_i * h * bpw)
        return (non_exp + routed_exp, non_exp, routed_exp)

    q3_total, q3_non_exp, q3_exp = _est_backbone((24 + 4) / 64.0)
    q4_total, _, _ = _est_backbone((32 + 4) / 64.0)

    tier_specs: list[tuple[str, str, int, float | None]] = [
        ("24gb-m4", "24gb-m4", 1, 24.0),
        ("128gb-max", "128gb-max", 1, 128.0),
        ("256gb-ultra", "256gb-ultra", 1, 256.0),
        ("512gb-ultra", "512gb-ultra", 1, 512.0),
        ("tb5-cluster-2x128gb", "cluster", 2, 128.0),
        ("tb5-cluster-4x128gb", "cluster", 4, 128.0),
        ("tb5-cluster-2x256gb", "cluster", 2, 256.0),
    ]
    if host_ram_gib is not None and host_ram_gib > 0:
        tier_specs.append(("host-custom", "auto", 1, host_ram_gib))

    tier_admission: dict[str, dict[str, object]] = {}
    for label, tier_key, ws, ram_g in tier_specs:
        chk = evaluate_hardware_tier(
            q3_total,
            target_tier=tier_key,
            host_ram_gib=ram_g,
            world_size=ws,
            non_expert_bytes=q3_non_exp,
            routed_expert_bytes=q3_exp,
            allow_override=True,
        )
        entry_adm: dict[str, object] = {
            "target_tier": chk.target_tier,
            "world_size": chk.world_size,
            "per_rank_budget_gib": round(chk.per_rank_budget_bytes / (1024**3), 2),
            "per_rank_q3_wired_gib": round(chk.per_rank_wired_bytes / (1024**3), 2),
            "fits_all_wired_q3": chk.fits,
            "recommendation": chk.recommendation,
        }
        if label == "24gb-m4":
            enc_exp_q3 = (q3_exp * enc_l) // max(1, total_layers)
            dec_exp_q3 = max(0, q3_exp - enc_exp_q3)
            m4_budget = chk.per_rank_budget_bytes
            hot_budget = max(0, m4_budget - q3_non_exp)
            dec_hot_frac = min(1.0, hot_budget / dec_exp_q3) if dec_exp_q3 > 0 else 1.0
            fits_ecd = q3_non_exp <= m4_budget
            entry_adm["encoder_cold_decoder_hot"] = {
                "non_expert_q3_gib": round(q3_non_exp / (1024**3), 2),
                "encoder_routed_cold_q3_gib": round(enc_exp_q3 / (1024**3), 2),
                "decoder_routed_q3_gib": round(dec_exp_q3 / (1024**3), 2),
                "decoder_hot_cache_budget_gib": round(hot_budget / (1024**3), 2),
                "decoder_hot_expert_fraction": round(dec_hot_frac, 4),
                "fits_encoder_cold_decoder_hot": fits_ecd,
            }
        tier_admission[label] = entry_adm

    summary_verified = False
    summary_path = base / "safetensors_headers_summary.json"
    if summary_path.is_file():
        sum_obj = json.loads(summary_path.read_text(encoding="utf-8"))
        ratios_map = sum_obj.get("scale_weight_ratios") if isinstance(sum_obj, dict) else None
        if isinstance(ratios_map, dict) and ratios_map:
            for rk, rv in ratios_map.items():
                if not isinstance(rv, dict):
                    continue
                w_dt = str(rv.get("weight_dtype") or "")
                s_dt = str(rv.get("scale_dtype") or "")
                rats = rv.get("ratios")
                if not isinstance(rats, list) or len(rats) != 2:
                    continue
                if w_dt == "F8_E4M3" and s_dt == "F8_E8M0":
                    if ".engram.embed." in rk:
                        if rats != [1.0, 32.0]:
                            raise ValueError(f"{rk} expected [1, 32] row-wise UE8M0 scale ratio, got {rats}")
                    elif not (24.0 <= float(rats[0]) <= 32.0 and 24.0 <= float(rats[1]) <= 32.0):
                        raise ValueError(f"{rk} expected [32, 32] UE8M0 FP8 block scale ratio, got {rats}")
                elif w_dt == "I8" and s_dt == "F8_E8M0":
                    if rats != [1.0, 16.0]:
                        raise ValueError(f"{rk} expected [1, 16] packed UE8M0 FP4 scale ratio, got {rats}")
            summary_verified = True

    raw_archs = raw_cfg.get("architectures")
    archs = tuple(str(a) for a in raw_archs) if isinstance(raw_archs, list) else ()
    return DeepSeek41MetadataReport(
        metadata_dir=base,
        model_type=str(raw_cfg.get("model_type", cfg.get("model_type", "unknown"))),
        architectures=archs,
        num_hidden_layers=total_layers,
        num_encoder_layers=enc_l,
        num_decoder_layers=dec_l,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        kv_latent_dim=kv_latent_dim,
        qk_rope_head_dim=qk_rope_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        num_routed_experts=num_routed_experts,
        num_shared_experts=num_shared_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_i,
        mhc_num_streams=mhc_streams,
        sliding_window=swa,
        kv_source_layer_ids=kv_source_ids,
        index_source_layer_ids=index_source_ids,
        engram_layer_ids=engram_layer_ids,
        engram_total_embeddings=engram_total_embeddings,
        engram_n_heads=engram_n_heads,
        engram_head_dim=engram_head_dim,
        dspark_layers=dspark_layers,
        dspark_block_size=dspark_block_size,
        dspark_n_routed_experts=dspark_n_experts,
        quantization_config=quant_cfg,
        kv_bytes_per_token_bf16=kv_bytes_per_token_bf16,
        kv_bytes_per_token_fp4=kv_bytes_per_token_fp4,
        total_index_entries=len(weight_map),
        shard_file_count=len(shard_files),
        index_total_size_bytes=total_size,
        backbone_tensor_count=backbone_cnt,
        vision_tensor_count=vision_cnt,
        engram_tensor_count=engram_cnt,
        dspark_mtp_tensor_count=dspark_cnt,
        routed_expert_tensor_count=routed_exp_cnt,
        routed_expert_ids_detected=len(expert_ids_seen),
        shared_expert_tensor_count=shared_exp_cnt,
        mhc_tensor_count=mhc_cnt,
        csa2_indexer_tensor_count=csa2_cnt,
        quant_scale_tensor_count=scale_cnt,
        long_backbone_names=tuple(long_names),
        estimated_backbone_q3_bytes=q3_total,
        estimated_non_expert_q3_bytes=q3_non_exp,
        estimated_routed_expert_q3_bytes=q3_exp,
        estimated_backbone_q4_bytes=q4_total,
        tier_admission=tier_admission,
        tokenizer_present=(base / "tokenizer.json").is_file(),
        header_summary_verified=summary_verified,
    )
