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
import re
import struct
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

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
    _ShardReaders,
    _ShardTensor,
    _stream_entry,
    _strip_model_prefix,
    encode_reserved0,
    f16_bytes_to_bf16_bytes,
    interleave_scale_bias_f16,
    is_vision_tensor,
    load_cold_experts_from_heat_profile,
    pack_mlx_affine_3bit,
    pack_mlx_affine_4bit,
    plan_amf_entries,
    resolve_checkpoint,
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

_ENGRAM_LAYER_RE = re.compile(r"(?:^|\.)(?:engram\.layers|layers)\.(\d+)\.")
_ENGRAM_SHARD_RE = re.compile(r"\.shard\.(\d+)\.")
_DSPARK_PREFIXES = ("dspark.", "model.dspark.", "language_model.dspark.", "language_model.model.dspark.")


class HardwareTierError(ValueError):
    """Raised when a `deepseek41` wired memory plan violates ADR-0015 hardware-tier ceilings."""


def is_engram_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` belongs to the external 196B Engram conditional memory table."""
    norm = _strip_model_prefix(raw_name).lower()
    return norm.startswith("engram.") or ".engram." in norm or norm.startswith("sparse_table.")


def is_dspark_tensor(raw_name: str) -> bool:
    """Return True if `raw_name` belongs to the DSpark semi-autoregressive proposer head."""
    return raw_name.lower().startswith(_DSPARK_PREFIXES)


def strip_dspark_prefix(raw_name: str) -> str:
    lower = raw_name.lower()
    for prefix in _DSPARK_PREFIXES:
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
) -> HardwareTierCheck:
    total_b, non_exp_b, exp_b = config.estimated_backbone_bytes()
    return evaluate_hardware_tier(
        total_b,
        target_tier=target_tier,
        host_ram_gib=host_ram_gib,
        world_size=config.world_size if world_size is None else world_size,
        non_expert_bytes=non_exp_b,
        routed_expert_bytes=exp_b,
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
    for name in engram_tensors:
        if name.endswith(".scales"):
            prefix = name[: -len(".scales")]
            if f"{prefix}.weight" in engram_tensors and f"{prefix}.biases" in engram_tensors:
                quant_prefixes.add(prefix)

    per_shard: list[list[_PlannedEntry]] = [[] for _ in range(num_shards)]
    consumed: set[str] = set()

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


def _write_backbone_stream(
    entries: Sequence[_PlannedEntry],
    output_path: pathlib.Path,
    model_id: str,
    readers: _ShardReaders,
    *,
    quantize_float_bits: int = 0,
    group_size: int = 64,
) -> list[PackedTensorMeta]:
    """Stream planned backbone entries to `output_path`, optionally quantizing 2D float weights."""
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
            should_quant = (
                quantize_float_bits in (3, 4)
                and e.quant_bits == 0
                and e.kind in quant_kinds
                and len(e.shape) == 2
                and e.shape[1] % group_size == 0
                and e.data.dtype in ("BF16", "F16", "F32")
            )
            if should_quant:
                src_fh = readers.get(e.data.path)
                src_fh.seek(e.data.offset)
                raw = src_fh.read(e.data.length)
                if len(raw) != e.data.length:
                    raise EOFError(f"short read for {e.name!r}")
                code_bytes, param_bytes = _quantize_2d_affine(
                    raw, e.data.dtype, e.shape[0], e.shape[1], quantize_float_bits, group_size
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
                amf_dt = DType.Q3_AFFINE if quantize_float_bits == 3 else DType.Q4_AFFINE
                q_name = e.name[: -len(".weight")] if e.name.endswith(".weight") else e.name
                metas.append(
                    PackedTensorMeta(
                        name=q_name,
                        kind=e.kind,
                        dtype=amf_dt,
                        quant_bits=quantize_float_bits,
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
            and e.quant_bits == 0
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
    strip_vision: bool = True,
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
    workers: int = 1,
) -> DeepSeek41ConvertReport:
    """Streaming multi-shard safetensors converter for `DeepSeek-V4.1-Flash-1017` (`pack-deepseek41`).

    Supports `q3_affine`, `q4_affine`, `bf16`, and `--quant-profile gsq-rco`,
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

    # Fast-fail hardware tier validation when config.json advertises full-scale 552B geometry.
    if src_config.get("model_type") == "deepseek41" and int(src_config.get("num_experts", 0) or 0) >= 64:
        enc_l = int(src_config.get("num_encoder_layers", 20) or 20)
        dec_l = int(src_config.get("num_decoder_layers", 20) or 20)
        h_sz = int(src_config.get("hidden_size", 7168) or 7168)
        n_exp = int(src_config.get("num_experts", 512) or 512)
        moe_i = int(src_config.get("moe_intermediate_size", 2048) or 2048)
        eff_dt = target_dtype if target_dtype in VALID_DEEPSEEK41_DTYPES else "q3_affine"
        probe_cfg = DeepSeek41Config(
            model_id=model_id[:63],
            num_encoder_layers=enc_l,
            num_decoder_layers=dec_l,
            hidden_size=h_sz,
            vocab_size=int(src_config.get("vocab_size", 129280) or 129280),
            num_attention_heads=int(src_config.get("num_attention_heads", 128) or 128),
            encoder_moe_top_k=int(src_config.get("encoder_moe_top_k", 8) or 8),
            decoder_moe_top_k=int(src_config.get("decoder_moe_top_k", 16) or 16),
            num_experts=n_exp,
            moe_intermediate_size=moe_i,
            shared_expert_intermediate_size=int(src_config.get("shared_expert_intermediate_size", moe_i) or moe_i),
            kv_lora_rank=int(src_config.get("kv_lora_rank", 1536) or 1536),
            q_lora_rank=int(src_config.get("q_lora_rank", 1536) or 1536),
            dtype=eff_dt,
            world_size=world_size,
        )
        validate_config_hardware_tier(
            probe_cfg,
            target_tier=target_tier,
            host_ram_gib=host_ram_gib,
            world_size=world_size,
            allow_override=allow_tier_override,
        )

    all_tensors = _collect_tensors(resolved.shard_paths)
    stripped_vision = 0
    engram_tensors: dict[str, _ShardTensor] = {}
    dspark_tensors: dict[str, _ShardTensor] = {}
    backbone_tensors: dict[str, _ShardTensor] = {}

    for raw_name, st in all_tensors.items():
        if strip_vision and is_vision_tensor(raw_name):
            stripped_vision += 1
            continue
        if is_engram_tensor(raw_name):
            engram_tensors[raw_name] = st
            continue
        if is_dspark_tensor(raw_name):
            dspark_tensors[raw_name] = st
            continue
        backbone_tensors[raw_name] = st

    eff_dtype_mode = "bf16" if target_dtype == "bf16" else dtype_mode
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

    # 2. Pack optional DSpark head when present.
    dspark_path: pathlib.Path | None = None
    dspark_metas: list[PackedTensorMeta] = []
    if dspark_tensors:
        eff_ds_out = (
            pathlib.Path(dspark_output) if dspark_output is not None else out_path.with_name(out_path.stem + "-dspark.amf")
        )
        remapped: dict[str, _ShardTensor] = {}
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
        ds_plan = plan_amf_entries(
            remapped,
            group_size=group_size,
            dtype_mode=eff_dtype_mode,
            tie_word_embeddings=False,
            sme2_pack="none",
            strip_vision=False,
        )
        readers_ds = _ShardReaders()
        try:
            dspark_metas = _write_backbone_stream(ds_plan, eff_ds_out, f"{model_id[:52]}-dspark", readers_ds)
        finally:
            readers_ds.close()
        dspark_path = eff_ds_out

    # 3. Stream backbone .amf (+ optional GSQ+RCO pass).
    readers = _ShardReaders()
    try:
        backbone_metas = _write_backbone_stream(
            backbone_plan,
            out_path,
            model_id,
            readers,
            quantize_float_bits=quantize_bits,
            group_size=group_size,
        )
    finally:
        readers.close()

    if quant_profile == "gsq-rco":
        from aminfer_tools.gsq_rco import gsq_convert

        eff_target = 3.25 if target_bits is None else float(target_bits)
        rep = gsq_convert(
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
        )
        backbone_metas = rep.metas

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
        }
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
    )
