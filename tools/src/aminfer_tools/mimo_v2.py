"""MiMo-V2.6 (`mimo_v2`, `3L` MTP, `DFlash`) & `MiMo-V2.6-Distill-Qwen-9B` (`qwen3_5`) W28/W25 pipeline.

Locked Canonical `.amf` v0 Tensor Naming & `config.json` Contract (`#502`, Wave 2 `m91-g26-kernels` & `m91-d15-spec`)
====================================================================================================================
Every tensor name written into `.amf` is strictly `<= 47` UTF-8 bytes (`TensorEntry.name` is `[48]u8`
NUL-terminated; `long_backbone_names == 0`).

1. `mimo_v2` Backbone (`model.amf`, `MiMo-V2.6-Flash` 48L / `MiMo-V2.6-Pro` 70L)
---------------------------------------------------------------------------------
- `config.json`: `"model_type": "mimo_v2"`, `"layer_types"` (`"full_attention"` vs `"sliding_attention"`),
  `"num_attention_heads"`, `"num_key_value_heads"` (`4` global Flash / `8` Pro),
  `"swa_num_attention_heads"`, `"swa_num_key_value_heads"` (`8`),
  `"head_dim": 192`, `"v_head_dim": 128`, `"swa_head_dim": 192`, `"swa_v_head_dim": 128`,
  `"sliding_window": 128`, `"rope_theta": 1e7`, `"swa_rope_theta": 1e4`, `"partial_rotary_factor"` (`rope_dim = 64`),
  `"attention_value_scale"` (`0.707` Flash / `0.612` Pro), `"add_swa_attention_sink_bias": true`,
  `"n_routed_experts"` (`256`/`384`), `"num_experts_per_tok": 8`, `"scoring_func": "sigmoid"`,
  `"norm_topk_prob": true`, `"routed_scaling_factor"`.
- `.amf` tensors:
  - `embed_tokens.weight`, `norm.weight`, `lm_head.weight`
  - `layers.{L}.input_layernorm.weight`, `layers.{L}.post_attention_layernorm.weight`
  - TP-deinterleaved & split attention projections:
    `layers.{L}.self_attn.q_proj.weight`, `layers.{L}.self_attn.k_proj.weight`,
    `layers.{L}.self_attn.v_proj.weight`, `layers.{L}.self_attn.o_proj.weight`
  - SWA sink bias (`sliding_attention` layers only): `layers.{L}.self_attn.sink_bias` (`[num_q_heads]`)
  - Dense L0: `layers.0.mlp.{gate_proj,up_proj,down_proj}.weight`
  - MoE `L >= 1`: `layers.{L}.mlp.gate.weight`, `layers.{L}.mlp.gate.bias` (`e_score_correction_bias`),
    `layers.{L}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight` (`GGUF_MXFP4` or `Q2..Q8_AFFINE`/`BF16`)
  - Strips `audio_encoder.*` / `speech_embeddings.*` on backbone; splits/strips `visual.*` when requested.

2. `3L` Chained `MTP` (`mtp.amf`, `format="mtp"`, `mtp_chained=true`, `num_hidden_layers=3`)
--------------------------------------------------------------------------------------------
- Per stage `k in 0..2` (`SWA`, `sliding_window=128`, `head_dim=192, v_head_dim=128`, dense MLP):
  - `layers.{k}.enorm.weight`, `layers.{k}.hnorm.weight`, `layers.{k}.eh_proj.weight` (`[H, 2H]`)
  - `layers.{k}.input_layernorm.weight`, `layers.{k}.post_attention_layernorm.weight` (from `pre_mlp_layernorm`),
    `layers.{k}.final_layernorm.weight`
  - TP-deinterleaved & split: `layers.{k}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`,
    `layers.{k}.self_attn.sink_bias` (`[num_q_heads]`)
  - `layers.{k}.mlp.{gate_proj,up_proj,down_proj}.weight`

3. `DFlash` Companion (`dflash.amf`, `draft_kind="dflash"`, `block_size in {8, 16}`, `5L`)
------------------------------------------------------------------------------------------
- `config.json`: `"model_type": "dflash"`, `"draft_kind": "dflash"`, `"num_hidden_layers": 5`,
  `"block_size"` (`8` or `16`), `"mask_token_id": 151675`, `"target_layer_ids"` (0-indexed),
  `"sliding_window": 1024`, `"rope_theta": 10000.0`, `"partial_rotary_factor": 0.5`,
  `"attention_value_scale": 0.612`, `"add_swa_attention_sink_bias": true`, `"attention_causal": false`,
  `"markov_rank": 0`.
- `.amf` tensors:
  - `fc.weight` (`[H, 5*H]`), `hidden_norm.weight` (`[H]`), `norm.weight` (`[H]`)
  - `layers.{l}.{input_layernorm,post_attention_layernorm}.weight`
  - `layers.{l}.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight`, `layers.{l}.self_attn.sink_bias`
  - `layers.{l}.mlp.{gate_proj,up_proj,down_proj}.weight`

4. `MiMo-V2.6-Distill-Qwen-9B` (`qwen3_5`, `32L`: `24` GDN + `8` GQA + `27L` ViT sidecar)
------------------------------------------------------------------------------------------
- `config.json`: `"model_type": "qwen3_5"`, `"layer_types"` (`24` `"linear_attention"` + `8` `"full_attention"`),
  `"rmsnorm_unit_offset": true` (`(1 + w)` on `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `norm`;
  plain `w` on `linear_attn.norm`), `"attn_output_gate": true`, `"partial_rotary_factor": 0.25`,
  `"mrope_section": [11, 11, 10]`, `"rope_theta": 1e7`, `"linear_num_key_heads": 16`,
  `"linear_num_value_heads": 32`, `"linear_key_head_dim": 128`, `"linear_value_head_dim": 128`,
  `"linear_conv_kernel_dim": 4`.
- `.amf` tensors:
  - `embed_tokens.weight`, `norm.weight`, `lm_head.weight`
  - `layers.{L}.{input_layernorm,post_attention_layernorm}.weight`, `layers.{L}.mlp.{gate_proj,up_proj,down_proj}.weight`
  - `linear_attention`: `layers.{L}.linear_attn.{q_proj,k_proj,v_proj}.weight` (row-sliced losslessly from `in_proj_qkv`),
    `layers.{L}.linear_attn.g_proj.weight` (`in_proj_z`), `layers.{L}.linear_attn.o_proj.weight` (`out_proj`),
    `layers.{L}.linear_attn.{a_proj,b_proj}.weight` (dequantized to `BF16` from `in_proj_a`/`in_proj_b`),
    `layers.{L}.linear_attn.conv1d.weight` (`[conv_dim, 4]` 2D), `layers.{L}.linear_attn.A_log`,
    `layers.{L}.linear_attn.dt_bias`, `layers.{L}.linear_attn.norm.weight` (plain `w`, NO `+1`)
  - `full_attention`: `layers.{L}.self_attn.q_proj.weight` & `layers.{L}.self_attn.q_gate_proj.weight`
    (losslessly head-interleaved sliced per head `[h*2*hd : h*2*hd+hd]` vs `[h*2*hd+hd : (h+1)*2*hd]`),
    `layers.{L}.self_attn.{k_proj,v_proj,o_proj,q_norm,k_norm}.weight`
  - ViT sidecar (`vision.amf`): `visual.*` (`333` tensors).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import pathlib
import random
import re
import struct
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DType,
    META_CHAT_TEMPLATE_JSON,
    META_CONFIG_JSON,
    META_TOKENIZER_JSON,
    NO_EXPERT,
    NO_LAYER,
    PackedTensorMeta,
    TensorKind,
    _ShardTensor,
    _collect_tensors,
    e4m3_byte_to_f32,
    embed_metadata_into_amf,
    f32_to_bf16_bytes as f32_values_to_bf16_bytes,
    f32_values_to_f16_bytes,
    is_vision_tensor,
    make_metadata_record,
    normalize_vision_name,
    pack_amf_from_safetensors,
    resolve_checkpoint,
    ue8m0_byte_to_f32,
    write_amf,
)
from aminfer_tools.chat_template import compile_chat_template_spec
from aminfer_tools.deepseek41 import HardwareTierError, load_cold_experts_from_heat_profile


def f32_values_to_f32_bytes(values: Sequence[float]) -> bytes:
    """Pack a sequence of floats as little-endian float32 bytes."""
    return struct.pack(f"<{len(values)}f", *values)


# ---------------------------------------------------------------------------
# Tier Profiles (`24gb-m4`, `128gb-max`, `512gb-ultra`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MimoV2TierProfile:
    """Hardware-tier memory & cold-expert placement budget for `mimo_v2` / `qwen3_5`."""

    name: str
    host_memory_gib: float
    wired_budget_gib: float
    always_active_reserve_gib: float
    hot_expert_budget_gib: float
    target_bits_default: float
    allow_nvme_cold_experts: bool
    recommended_model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MIMO_TIER_PROFILES: dict[str, MimoV2TierProfile] = {
    "24gb-m4": MimoV2TierProfile(
        name="24gb-m4",
        host_memory_gib=24.0,
        wired_budget_gib=18.5,
        always_active_reserve_gib=6.4,
        hot_expert_budget_gib=12.1,
        target_bits_default=3.50,
        allow_nvme_cold_experts=True,
        recommended_model="MiMo-V2.6-Flash (NVMe cold experts) / Distill-Qwen-9B (4-bit ~5.5 GiB full-wired)",
    ),
    "128gb-max": MimoV2TierProfile(
        name="128gb-max",
        host_memory_gib=128.0,
        wired_budget_gib=104.0,
        always_active_reserve_gib=12.0,
        hot_expert_budget_gib=92.0,
        target_bits_default=4.0,
        allow_nvme_cold_experts=False,
        recommended_model="MiMo-V2.6-Flash (~92 GiB full-wired native MXFP4/FP8)",
    ),
    "512gb-ultra": MimoV2TierProfile(
        name="512gb-ultra",
        host_memory_gib=512.0,
        wired_budget_gib=420.0,
        always_active_reserve_gib=32.0,
        hot_expert_budget_gib=350.0,
        target_bits_default=4.0,
        allow_nvme_cold_experts=False,
        recommended_model="MiMo-V2.6-Pro (~350 GiB full-wired native MXFP4/FP8)",
    ),
}


def resolve_mimo_tier_profile(tier: str | None) -> MimoV2TierProfile | None:
    if tier is None:
        return None
    key = tier.strip().lower()
    if key not in MIMO_TIER_PROFILES:
        raise ValueError(f"unknown target_tier {tier!r}; expected one of {sorted(MIMO_TIER_PROFILES)}")
    return MIMO_TIER_PROFILES[key]


# ---------------------------------------------------------------------------
# Constants, Detection & Naming Helpers (`<= 47` UTF-8 bytes enforced)
# ---------------------------------------------------------------------------

_E4M3_LUT: tuple[float, ...] = tuple(e4m3_byte_to_f32(i) for i in range(256))
_UE8M0_LUT: tuple[float, ...] = tuple(ue8m0_byte_to_f32(i) for i in range(256))
_KVALUES_MXFP4: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

_AUDIO_PREFIXES = (
    "audio_encoder.",
    "speech_embeddings.",
    "audio_tower.",
    "model.audio_encoder.",
    "model.speech_embeddings.",
    "model.audio.",
)
_MTP_PREFIXES = ("mtp.", "model.mtp.", "language_model.mtp.", "language_model.model.mtp.")
_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", "_weight_scale_inv", "_weight_scale")
_LAYER_QKV_SCALE_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.self_attn\.qkv_proj\.weight_scale_inv$")
_MTP_QKV_SCALE_RE = re.compile(r"^(?:model\.)?mtp\.layers\.(\d+)\.self_attn\.qkv_proj\.weight_scale_inv$")
_LAYER_QKV_WEIGHT_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.self_attn\.qkv_proj\.weight$")
_MTP_QKV_WEIGHT_RE = re.compile(r"^(?:model\.)?mtp\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$")


def is_mimo_audio_tensor(raw_name: str) -> bool:
    """Return True for `mimo_v2` `audio_encoder.*` / `speech_embeddings.*` keys."""
    return raw_name.lower().startswith(_AUDIO_PREFIXES)


def is_mimo_mtp_tensor(raw_name: str) -> bool:
    """Return True for `mimo_v2` native 3-layer MTP tensors."""
    return raw_name.startswith(_MTP_PREFIXES)


def is_mimo_dflash_tensor(raw_name: str) -> bool:
    """Return True for `dflash` draft tensors."""
    return raw_name.startswith(("dflash.", "dflare.", "fc.", "hidden_norm.", "enc.output_norm."))


def is_mimo_v2_config(cfg: Mapping[str, Any] | None) -> bool:
    """Return True if parsed `config.json` describes a `mimo_v2` model or `dflash` draft."""
    if not isinstance(cfg, Mapping):
        return False
    mt = str(cfg.get("model_type") or "").lower()
    if mt in ("mimo_v2", "mimo2", "mimo_v2_mtp", "dflash"):
        return True
    if isinstance(cfg.get("dflash_config"), Mapping):
        return True
    tc = cfg.get("text_config")
    if isinstance(tc, Mapping) and str(tc.get("model_type") or "").lower() in ("mimo_v2", "mimo2"):
        return True
    return False


def is_qwen35_config(cfg: Mapping[str, Any] | None) -> bool:
    """Return True if parsed `config.json` describes a `qwen3_5` / `Distill-Qwen-9B` model."""
    if not isinstance(cfg, Mapping):
        return False
    mt = str(cfg.get("model_type") or "").lower()
    if mt in ("qwen3_5", "qwen3_5_text", "qwen35"):
        return True
    tc = cfg.get("text_config")
    if isinstance(tc, Mapping) and str(tc.get("model_type") or "").lower() in ("qwen3_5", "qwen3_5_text", "qwen35"):
        return True
    return False


def _strip_mimo_scale_suffix(raw_name: str) -> str:
    for suf in _SCALE_SUFFIXES:
        if raw_name.endswith(suf):
            return raw_name[: -len(suf)]
    return raw_name


def normalize_mimo_tensor_name(raw_name: str, *, mtp_standalone: bool = False) -> str:
    """Normalize a `mimo_v2` / `mtp` tensor name to locked canonical `<= 47 B` `.amf` form (`#502`)."""
    if is_vision_tensor(raw_name):
        return normalize_vision_name(raw_name)
    name = _strip_mimo_scale_suffix(raw_name)
    for pref in ("model.language_model.", "model.llm.", "language_model.model.", "language_model.", "model."):
        if name.startswith(pref):
            name = name[len(pref) :]
            break
    if name == "decoder.self_attn.o_proj.weight":
        name = "self_attn.o_proj.weight"
    if mtp_standalone and name.startswith("mtp."):
        name = name[len("mtp.") :]
    name = name.replace(".self_attn.attention_sink_bias", ".self_attn.sink_bias")
    name = name.replace(".mlp.gate.e_score_correction_bias", ".mlp.gate.bias")
    if mtp_standalone:
        name = name.replace(".pre_mlp_layernorm.weight", ".post_attention_layernorm.weight")
    if len(name.encode("utf-8")) > 47:
        name = name.replace(".self_attn.", ".attn.")
    if len(name.encode("utf-8")) > 47:
        name = name.replace("mtp.layers.", "mtp.")
    return name


def normalize_mimo_dflash_tensor_name(raw_name: str) -> str:
    """Normalize an HF or GGUF `dflash` draft tensor name to locked `<= 47 B` `.amf` form (`#502`)."""
    name = raw_name
    for pref in ("model.", "dflash."):
        if name.startswith(pref):
            name = name[len(pref) :]
            break
    exact_map = {
        "dflare.fc.weight": "fc.weight",
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "hidden_norm.weight",
        "enc.output_norm.weight": "hidden_norm.weight",
        "fc_norm.weight": "hidden_norm.weight",
        "dflare.fc_norm.weight": "hidden_norm.weight",
        "norm.weight": "norm.weight",
        "output_norm.weight": "norm.weight",
        "dflare.norm.weight": "norm.weight",
        "token_embd.weight": "embed_tokens.weight",
        "embed_tokens.weight": "embed_tokens.weight",
        "dflare.tok_embed.weight": "embed_tokens.weight",
        "mask_embed": "mask_embed.weight",
        "mask_embd.weight": "mask_embed.weight",
        "dflare.mask_embed.weight": "mask_embed.weight",
        "output.weight": "lm_head.weight",
    }
    if name in exact_map:
        return exact_map[name]
    name = name.replace(".self_attn.attention_sink_bias", ".self_attn.sink_bias")
    if len(name.encode("utf-8")) > 47:
        name = name.replace(".self_attn.", ".attn.")
    return name


def normalize_qwen35_tensor_name(raw_name: str) -> str:
    """Normalize a `qwen3_5` tensor name to locked canonical `<= 47 B` `.amf` form (`#502`)."""
    if is_vision_tensor(raw_name):
        return normalize_vision_name(raw_name)
    name = raw_name
    for pref in ("model.language_model.", "model.llm.", "language_model.model.", "language_model.", "model."):
        if name.startswith(pref):
            name = name[len(pref) :]
            break
    for suf in (".scales", ".biases", ".weight_packed", ".weight_scale", ".weight_scale_inv"):
        if name.endswith(suf):
            name = name[: -len(suf)] + ".weight"
            break
    name = name.replace(".linear_attn.in_proj_z.", ".linear_attn.g_proj.")
    name = name.replace(".linear_attn.out_proj.", ".linear_attn.o_proj.")
    name = name.replace(".linear_attn.in_proj_a.", ".linear_attn.a_proj.")
    name = name.replace(".linear_attn.in_proj_b.", ".linear_attn.b_proj.")
    return name


def _effective_mimo_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    tc = cfg.get("text_config")
    if isinstance(tc, Mapping):
        merged = dict(tc)
        for k, v in cfg.items():
            if k not in ("text_config", "vision_config", "audio_config") and k not in merged:
                merged[k] = v
        return merged
    return dict(cfg)


# ---------------------------------------------------------------------------
# MXFP4 & TP-Interleaved `qkv_proj` Helpers
# ---------------------------------------------------------------------------


def repack_hf_mxfp4_row_to_gguf(
    w_row: bytes | bytearray | memoryview,
    s_row: bytes | bytearray | memoryview,
    k: int,
) -> bytes:
    """Losslessly repack one HF `MXFP4` row (`k//2` adjacent nibbles + `k//32` UE8M0 bytes) to `GGUF_MXFP4`."""
    if k <= 0 or k % 32 != 0:
        raise ValueError(f"invalid MXFP4 row width k={k}; must be positive multiple of 32")
    groups = k // 32
    if len(w_row) != k // 2:
        raise ValueError(f"HF MXFP4 weight row bytes {len(w_row)} != {k // 2}")
    if len(s_row) != groups:
        raise ValueError(f"HF MXFP4 scale row bytes {len(s_row)} != {groups}")

    out = bytearray(groups * 17)
    for g in range(groups):
        out[g * 17] = s_row[g]
        w_base = g * 16
        out_base = g * 17 + 1
        nibs = [0] * 32
        for j in range(16):
            b = w_row[w_base + j]
            nibs[2 * j] = b & 0x0F
            nibs[2 * j + 1] = b >> 4
        for j in range(16):
            out[out_base + j] = nibs[j] | (nibs[j + 16] << 4)
    return bytes(out)


def dequantize_hf_mxfp4_row_f32(
    w_row: bytes | bytearray | memoryview,
    s_row: bytes | bytearray | memoryview,
    k: int,
) -> list[float]:
    """Dequantize one HF `MXFP4` row (`U8 [k//2]` adjacent nibbles + `U8 [k//32]` UE8M0) to float32."""
    if k <= 0 or k % 32 != 0:
        raise ValueError(f"invalid MXFP4 row width k={k}; must be positive multiple of 32")
    groups = k // 32
    if len(w_row) != k // 2 or len(s_row) != groups:
        raise ValueError(f"HF MXFP4 row length mismatch: data={len(w_row)}, scales={len(s_row)}, k={k}")
    kvals = _KVALUES_MXFP4
    u_lut = _UE8M0_LUT
    out = [0.0] * k
    for g in range(groups):
        d = u_lut[s_row[g]] * 0.5
        w_base = g * 16
        o_base = g * 32
        for j in range(16):
            b = w_row[w_base + j]
            out[o_base + 2 * j] = kvals[b & 0x0F] * d
            out[o_base + 2 * j + 1] = kvals[b >> 4] * d
    return out


@dataclass(frozen=True)
class MimoQkvSpec:
    """Geometry and optional TP-interleaving descriptor for `mimo_v2` `qkv_proj`."""

    tp_degree: int
    q_heads: int
    kv_heads: int
    head_dim: int
    v_head_dim: int
    k_cols: int

    @property
    def q_total(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def k_total(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def v_total(self) -> int:
        return self.kv_heads * self.v_head_dim

    @property
    def total_rows(self) -> int:
        return self.q_total + self.k_total + self.v_total

    @property
    def q_tp(self) -> int:
        return self.q_total // self.tp_degree

    @property
    def k_tp(self) -> int:
        return self.k_total // self.tp_degree

    @property
    def v_tp(self) -> int:
        return self.v_total // self.tp_degree

    @property
    def rank_rows(self) -> int:
        return self.total_rows // self.tp_degree

    @property
    def sb_q_tp(self) -> int:
        return (self.q_tp + 127) // 128

    @property
    def sb_k_tp(self) -> int:
        return (self.k_tp + 127) // 128

    @property
    def sb_v_tp(self) -> int:
        return (self.v_tp + 127) // 128

    @property
    def sb_per_rank(self) -> int:
        return self.sb_q_tp + self.sb_k_tp + self.sb_v_tp

    @property
    def expected_scale_rows(self) -> int:
        if self.tp_degree <= 1:
            return (self.total_rows + 127) // 128
        return self.tp_degree * self.sb_per_rank


def _layer_attn_geom(cfg: Mapping[str, Any], layer_idx: int, *, is_mtp: bool = False) -> tuple[int, int, int, int]:
    """Return `(q_heads, kv_heads, head_dim, v_head_dim)` for a `mimo_v2` backbone or MTP layer."""
    hidden = int(cfg.get("hidden_size", 4096) or 4096)
    g_q = int(cfg.get("num_attention_heads", 64) or 64)
    g_kv = int(cfg.get("num_key_value_heads", g_q) or g_q)
    g_hd = int(cfg.get("head_dim", hidden // max(g_q, 1)) or 192)
    g_vd = int(cfg.get("v_head_dim", g_hd) or g_hd)

    pat = cfg.get("hybrid_layer_pattern") or []
    is_swa = True if is_mtp else (layer_idx < len(pat) and int(pat[layer_idx]) != 0)
    if is_swa:
        s_q = int(cfg.get("swa_num_attention_heads") or g_q)
        s_kv = int(cfg.get("swa_num_key_value_heads") or g_kv)
        s_hd = int(cfg.get("swa_head_dim") or g_hd)
        s_vd = int(cfg.get("swa_v_head_dim") or g_vd)
        return s_q, s_kv, s_hd, s_vd
    return g_q, g_kv, g_hd, g_vd


def infer_mimo_qkv_tp_degree(
    cfg: Mapping[str, Any] | None,
    tensors: Mapping[str, Any],
    *,
    explicit_tp: int | None = None,
) -> int:
    """Infer TP-interleaving degree (`1`, `2`, `4`, `8`, `16`) from `qkv_proj.weight_scale_inv` ranks."""
    if explicit_tp is not None:
        if explicit_tp <= 0:
            raise ValueError(f"qkv_tp_degree must be positive, got {explicit_tp}")
        return explicit_tp
    if not cfg or not is_mimo_v2_config(cfg):
        return 1
    eff = _effective_mimo_cfg(cfg)

    def _get_shape(obj: Any) -> tuple[int, ...] | None:
        if isinstance(obj, _ShardTensor):
            return obj.shape
        if isinstance(obj, Mapping) and isinstance(obj.get("shape"), Sequence):
            return tuple(int(x) for x in obj["shape"])
        return None

    for name, obj in sorted(tensors.items()):
        m = _LAYER_QKV_SCALE_RE.match(name)
        is_mtp = False
        if m is None:
            m = _MTP_QKV_SCALE_RE.match(name)
            is_mtp = True
        if m is None:
            continue
        sh = _get_shape(obj)
        if not sh or len(sh) != 2:
            continue
        s_rows = sh[0]
        layer_idx = int(m.group(1))
        qh, kvh, hd, vd = _layer_attn_geom(eff, layer_idx, is_mtp=is_mtp)
        n_tot = qh * hd + kvh * (hd + vd)
        contig_rows = (n_tot + 127) // 128
        if s_rows == contig_rows:
            continue
        for tp in (4, 8, 2, 16):
            if qh % tp != 0 or kvh % tp != 0:
                continue
            spec = MimoQkvSpec(tp, qh, kvh, hd, vd, sh[1] * 128)
            if spec.expected_scale_rows == s_rows:
                return tp
    return 1


def resolve_mimo_qkv_spec_for_tensor(
    raw_name: str,
    n_rows: int,
    k_cols: int,
    s_rows: int | None,
    cfg: Mapping[str, Any] | None,
    *,
    tp_degree: int = 1,
) -> MimoQkvSpec | None:
    """Return `MimoQkvSpec` when `raw_name` is a `mimo_v2` `qkv_proj` that should be de-interleaved/split."""
    if cfg is None or not is_mimo_v2_config(cfg):
        return None
    eff = _effective_mimo_cfg(cfg)
    m = _LAYER_QKV_WEIGHT_RE.match(raw_name) or _LAYER_QKV_SCALE_RE.match(raw_name)
    is_mtp = False
    if m is None:
        m = _MTP_QKV_WEIGHT_RE.match(raw_name) or _MTP_QKV_SCALE_RE.match(raw_name)
        is_mtp = True
    if m is None:
        return None
    qh, kvh, hd, vd = _layer_attn_geom(eff, int(m.group(1)), is_mtp=is_mtp)
    if tp_degree > 1 and (qh % tp_degree != 0 or kvh % tp_degree != 0):
        return None
    spec = MimoQkvSpec(tp_degree, qh, kvh, hd, vd, k_cols)
    if spec.total_rows != n_rows:
        return None
    if s_rows is not None and spec.expected_scale_rows != s_rows:
        return None
    return spec


def map_mimo_contiguous_qkv_row(spec: MimoQkvSpec, out_row: int) -> tuple[int, int]:
    """Map contiguous `[Q_all, K_all, V_all]` row index -> `(src_weight_row, src_scale_block_row)`."""
    if out_row < 0 or out_row >= spec.total_rows:
        raise IndexError(f"qkv output row {out_row} out of bounds for {spec.total_rows}")
    if spec.tp_degree <= 1:
        return out_row, out_row // 128
    if out_row < spec.q_total:
        rank = out_row // spec.q_tp
        local = out_row % spec.q_tp
        src_r = rank * spec.rank_rows + local
        sc_r = rank * spec.sb_per_rank + (local // 128)
        return src_r, sc_r
    rem = out_row - spec.q_total
    if rem < spec.k_total:
        rank = rem // spec.k_tp
        local = rem % spec.k_tp
        src_r = rank * spec.rank_rows + spec.q_tp + local
        sc_r = rank * spec.sb_per_rank + spec.sb_q_tp + (local // 128)
        return src_r, sc_r
    rem_v = rem - spec.k_total
    rank = rem_v // spec.v_tp
    local = rem_v % spec.v_tp
    src_r = rank * spec.rank_rows + spec.q_tp + spec.k_tp + local
    sc_r = rank * spec.sb_per_rank + spec.sb_q_tp + spec.sb_k_tp + (local // 128)
    return src_r, sc_r


def dequantize_mimo_qkv_row_f32(
    raw_fp8_row: bytes | bytearray | memoryview,
    sc_row_idx: int,
    scales_f32: Sequence[float],
    k_cols: int,
) -> list[float]:
    """Dequantize one FP8-E4M3 `qkv_proj` source row using its resolved `weight_scale_inv` block row."""
    if len(raw_fp8_row) != k_cols:
        raise ValueError(f"FP8 row len {len(raw_fp8_row)} != {k_cols}")
    kb = (k_cols + 127) // 128
    base = sc_row_idx * kb
    lut = _E4M3_LUT
    out = [0.0] * k_cols
    for col in range(k_cols):
        out[col] = lut[raw_fp8_row[col]] * scales_f32[base + (col // 128)]
    return out


# ---------------------------------------------------------------------------
# W28 Metadata & Header Validators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MimoV2MetadataReport:
    """Verified W28 Hugging Face schema + safetensors/GGUF header audit report for `mimo_v2`."""

    metadata_dir: pathlib.Path
    variant: str
    model_type: str
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    swa_layers: int
    global_layers: int
    global_layer_indices: tuple[int, ...]
    sliding_window_size: int
    head_dim: int
    v_head_dim: int
    global_q_heads: int
    global_kv_heads: int
    swa_q_heads: int
    swa_kv_heads: int
    rope_theta: float
    swa_rope_theta: float
    partial_rotary_factor: float
    rope_dim: int
    attention_value_scale: float
    add_swa_attention_sink_bias: bool
    add_full_attention_sink_bias: bool
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    moe_dense_layers: int
    moe_layers: int
    mtp_layers: int
    dflash_present: bool
    dflash_block_size: int
    dflash_target_layer_ids: tuple[int, ...]
    active_params: int
    routed_expert_params: int
    total_params: int
    global_kv_bytes_per_token_fp16: int
    swa_kv_bytes_per_token_fp16: int
    swa_kv_window_cap_bytes_fp16: int
    qkv_tp_degree_inferred: int
    backbone_tensor_count: int
    vision_tensor_count: int
    audio_tensor_count: int
    mtp_tensor_count: int
    dflash_tensor_count: int
    long_backbone_names: int
    long_vision_names: int
    long_mtp_names: int
    long_dflash_names: int
    max_backbone_name_bytes: int
    header_summary_shards: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata_dir"] = str(self.metadata_dir)
        out["valid"] = len(self.errors) == 0
        return out


@dataclass(frozen=True)
class Qwen35MetadataReport:
    """Verified W28 Hugging Face schema + safetensors header audit report for `qwen3_5` (`Distill-Qwen-9B`)."""

    metadata_dir: pathlib.Path
    model_type: str
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    linear_attention_layers: int
    full_attention_layers: int
    full_attention_indices: tuple[int, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attn_output_gate: bool
    partial_rotary_factor: float
    rope_dim: int
    rope_theta: float
    mrope_section: tuple[int, ...]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rmsnorm_unit_offset: bool
    linear_attn_norm_offset: float
    tie_word_embeddings: bool
    quantization_mode: str | None
    vision_layers: int
    vision_hidden_size: int
    active_params: int
    total_params: int
    full_kv_bytes_per_token_fp16: int
    gdn_state_bytes_fp32_fp16: int
    backbone_tensor_count: int
    vision_tensor_count: int
    long_backbone_names: int
    long_vision_names: int
    max_backbone_name_bytes: int
    header_summary_shards: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata_dir"] = str(self.metadata_dir)
        out["valid"] = len(self.errors) == 0
        return out


def _read_weight_map_keys(meta_dir: pathlib.Path) -> tuple[dict[str, str], int]:
    idx_p = meta_dir / "model.safetensors.index.json"
    if idx_p.is_file():
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
        wm = idx.get("weight_map")
        if isinstance(wm, dict):
            shards = {str(v) for v in wm.values()}
            return {str(k): str(v) for k, v in wm.items()}, len(shards)
    st_p = meta_dir / "model.safetensors"
    if st_p.is_file():
        return {t.name: "model.safetensors" for t in _collect_tensors((st_p,))}, 1
    sum_p = meta_dir / "safetensors_headers_summary.json"
    if sum_p.is_file():
        summ = json.loads(sum_p.read_text(encoding="utf-8"))
        shards_obj = summ.get("shards", {})
        if isinstance(shards_obj, dict):
            wm_out: dict[str, str] = {}
            for sh_name, sh_info in shards_obj.items():
                tensors_map = sh_info.get("tensors", {}) if isinstance(sh_info, dict) else {}
                for k in tensors_map:
                    wm_out[str(k)] = str(sh_name)
            return wm_out, len(shards_obj)
    return {}, 0


def _read_header_summary_tensors(meta_dir: pathlib.Path) -> tuple[dict[str, dict[str, Any]], int]:
    sum_p = meta_dir / "safetensors_headers_summary.json"
    if not sum_p.is_file():
        return {}, 0
    summ = json.loads(sum_p.read_text(encoding="utf-8"))
    shards_obj = summ.get("shards", {})
    if not isinstance(shards_obj, dict):
        return {}, 0
    out: dict[str, dict[str, Any]] = {}
    for sh_info in shards_obj.values():
        tmap = sh_info.get("tensors", {}) if isinstance(sh_info, dict) else {}
        for k, spec in tmap.items():
            if isinstance(spec, dict):
                out[str(k)] = spec
    return out, len(shards_obj)


def validate_mimo_v2_metadata_dir(meta_dir: pathlib.Path, *, strict: bool = True) -> MimoV2MetadataReport:
    """Validate staged HF metadata + safetensors header manifests for `MiMo-V2.6-Flash` / `Pro` + `MTP` + `DFlash`."""
    meta_dir = pathlib.Path(meta_dir)
    cfg_p = meta_dir / "config.json"
    if not cfg_p.is_file():
        raise FileNotFoundError(f"config.json not found in {meta_dir}")
    raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    cfg = _effective_mimo_cfg(raw_cfg)
    errors: list[str] = []

    model_type = str(cfg.get("model_type") or "")
    if model_type not in ("mimo_v2", "mimo2"):
        errors.append(f"unexpected model_type {model_type!r} (expected 'mimo_v2')")

    hidden = int(cfg.get("hidden_size", 0) or 0)
    vocab_size = int(cfg.get("vocab_size", 0) or 0)
    num_layers = int(cfg.get("num_hidden_layers", 0) or 0)
    if hidden <= 0 or vocab_size <= 0 or num_layers <= 0:
        errors.append(f"invalid core geometry hidden={hidden}, vocab={vocab_size}, layers={num_layers}")

    pat = [int(x) for x in (cfg.get("hybrid_layer_pattern") or [])]
    if len(pat) != num_layers:
        errors.append(f"hybrid_layer_pattern length {len(pat)} != num_hidden_layers {num_layers}")
        swa_layers = 0
        global_indices: tuple[int, ...] = ()
    else:
        swa_layers = sum(1 for x in pat if x != 0)
        global_indices = tuple(i for i, x in enumerate(pat) if x == 0)
    global_layers = num_layers - swa_layers

    sliding_win = int(cfg.get("sliding_window_size") or cfg.get("sliding_window") or 0)
    if swa_layers > 0 and sliding_win <= 0:
        errors.append("SWA layers present but sliding_window <= 0")

    g_q = int(cfg.get("num_attention_heads", 0) or 0)
    g_kv = int(cfg.get("num_key_value_heads", g_q) or 0)
    hd = int(cfg.get("head_dim", 0) or 0)
    vd = int(cfg.get("v_head_dim", hd) or 0)
    s_q = int(cfg.get("swa_num_attention_heads") or g_q)
    s_kv = int(cfg.get("swa_num_key_value_heads") or g_kv)
    s_hd = int(cfg.get("swa_head_dim") or hd)
    s_vd = int(cfg.get("swa_v_head_dim") or vd)
    if hd <= 0 or vd <= 0 or s_hd != hd or s_vd != vd:
        errors.append(f"invalid head dims global=({hd},{vd}), swa=({s_hd},{s_vd})")

    rope_theta = float(cfg.get("rope_theta", 10000000.0) or 10000000.0)
    swa_rope_theta = float(cfg.get("swa_rope_theta", 10000.0) or 10000.0)
    prf = float(cfg.get("partial_rotary_factor", 1.0) or 1.0)
    rope_dim = int(hd * prf)
    v_scale = float(cfg.get("attention_value_scale", 1.0) or 1.0)
    if not (math.isfinite(v_scale) and v_scale > 0.0):
        errors.append(f"invalid attention_value_scale {v_scale}")
    add_swa_sink = bool(cfg.get("add_swa_attention_sink_bias", False))
    add_full_sink = bool(cfg.get("add_full_attention_sink_bias", False))
    if swa_layers > 0 and not add_swa_sink:
        errors.append("expected add_swa_attention_sink_bias=true for mimo_v2")
    if add_full_sink:
        errors.append("unexpected add_full_attention_sink_bias=true on mimo_v2")

    inter = int(cfg.get("intermediate_size", 0) or 0)
    moe_inter = int(cfg.get("moe_intermediate_size", 0) or 0)
    n_routed = int(cfg.get("n_routed_experts", 0) or 0)
    topk = int(cfg.get("num_experts_per_tok", 0) or 0)
    n_shared = int(cfg.get("n_shared_experts") or 0)
    moe_freq = [int(x) for x in (cfg.get("moe_layer_freq") or [1 if i > 0 else 0 for i in range(num_layers)])]
    if len(moe_freq) != num_layers:
        errors.append(f"moe_layer_freq length {len(moe_freq)} != num_hidden_layers {num_layers}")
        moe_layer_cnt = max(0, num_layers - 1)
    else:
        moe_layer_cnt = sum(1 for x in moe_freq if x != 0)
    moe_dense_cnt = num_layers - moe_layer_cnt

    if num_layers == 48 and n_routed == 256:
        variant = "flash"
    elif num_layers == 70 and n_routed == 384:
        variant = "pro"
    else:
        variant = "custom"

    tie_emb = bool(cfg.get("tie_word_embeddings", False))
    emb_params = vocab_size * hidden * (1 if tie_emb else 2)
    attn_global_params = hidden * (g_q * hd + g_kv * (hd + vd) + g_q * vd)
    attn_swa_params = hidden * (s_q * hd + s_kv * (hd + vd) + s_q * vd)
    dense_mlp_params = 3 * hidden * inter
    shared_mlp_params = 3 * hidden * moe_inter * n_shared
    one_expert_params = 3 * hidden * moe_inter
    active_params = (
        emb_params
        + global_layers * attn_global_params
        + swa_layers * attn_swa_params
        + moe_dense_cnt * dense_mlp_params
        + moe_layer_cnt * (shared_mlp_params + topk * one_expert_params)
    )
    routed_params = moe_layer_cnt * n_routed * one_expert_params
    total_params = (
        emb_params
        + global_layers * attn_global_params
        + swa_layers * attn_swa_params
        + moe_dense_cnt * dense_mlp_params
        + moe_layer_cnt * shared_mlp_params
        + routed_params
    )

    global_kv_bytes = global_layers * g_kv * (hd + vd) * 2
    swa_kv_bytes = swa_layers * s_kv * (hd + vd) * 2
    swa_kv_cap = swa_kv_bytes * max(0, sliding_win)

    wm, _ = _read_weight_map_keys(meta_dir)
    hdr_tensors, hdr_shards = _read_header_summary_tensors(meta_dir)
    all_keys = sorted(set(wm.keys()) | set(hdr_tensors.keys()))

    bb_cnt = vis_cnt = aud_cnt = mtp_cnt = 0
    long_bb = long_vis = long_mtp = max_bb_bytes = 0
    mtp_layer_ids: set[int] = set()

    for raw_k in all_keys:
        if is_mimo_audio_tensor(raw_k):
            aud_cnt += 1
            continue
        if is_vision_tensor(raw_k):
            vis_cnt += 1
            vn = normalize_vision_name(raw_k)
            if len(vn.encode("utf-8")) > 47:
                long_vis += 1
            continue
        if is_mimo_mtp_tensor(raw_k):
            mtp_cnt += 1
            mm = re.match(r"^(?:model\.)?mtp\.layers\.(\d+)\.", raw_k)
            if mm is not None:
                mtp_layer_ids.add(int(mm.group(1)))
            mn = normalize_mimo_tensor_name(raw_k, mtp_standalone=True)
            if len(mn.encode("utf-8")) > 47:
                long_mtp += 1
            continue
        bb_cnt += 1
        bn = normalize_mimo_tensor_name(raw_k)
        blen = len(bn.encode("utf-8"))
        if blen > max_bb_bytes:
            max_bb_bytes = blen
        if blen > 47:
            long_bb += 1

    if long_bb > 0:
        errors.append(f"long_backbone_names={long_bb} (> 47 UTF-8 bytes)")
    if long_vis > 0:
        errors.append(f"long_vision_names={long_vis} (> 47 UTF-8 bytes)")
    if long_mtp > 0:
        errors.append(f"long_mtp_names={long_mtp} (> 47 UTF-8 bytes)")

    mtp_layers_cfg = int(cfg.get("num_nextn_predict_layers") or 0)
    mtp_layers = len(mtp_layer_ids) if mtp_layer_ids else mtp_layers_cfg
    if mtp_layers_cfg > 0 and mtp_layer_ids and len(mtp_layer_ids) != mtp_layers_cfg:
        errors.append(f"MTP layer count {len(mtp_layer_ids)} != num_nextn_predict_layers {mtp_layers_cfg}")

    tp_deg = infer_mimo_qkv_tp_degree(cfg, hdr_tensors)
    for k, spec in hdr_tensors.items():
        dt = str(spec.get("dtype") or "")
        sh = [int(x) for x in (spec.get("shape") or [])]
        if k.endswith(".self_attn.qkv_proj.weight_scale_inv"):
            is_mtp_k = ".mtp." in k
            km = _MTP_QKV_SCALE_RE.match(k) if is_mtp_k else _LAYER_QKV_SCALE_RE.match(k)
            if km is not None and len(sh) == 2:
                l_idx = int(km.group(1))
                qh_l, kvh_l, hd_l, vd_l = _layer_attn_geom(cfg, l_idx, is_mtp=is_mtp_k)
                expected_rows = MimoQkvSpec(tp_deg, qh_l, kvh_l, hd_l, vd_l, hidden).expected_scale_rows
                if sh[0] != expected_rows:
                    errors.append(f"{k}: scale_inv rows {sh[0]} != expected {expected_rows} (tp={tp_deg})")
        elif ".mlp.experts." in k and k.endswith(".weight") and dt == "U8":
            sc_k = k[: -len(".weight")] + ".weight_scale"
            sc_spec = hdr_tensors.get(sc_k)
            if sc_spec is not None:
                sc_sh = [int(x) for x in (sc_spec.get("shape") or [])]
                if len(sh) != 2 or len(sc_sh) != 2 or sc_sh[0] != sh[0] or sh[1] != 16 * sc_sh[1]:
                    errors.append(f"{k}: invalid MXFP4 weight/scale shapes {sh} vs {sc_sh}")

    df_dir = meta_dir / "dflash"
    df_present = (df_dir / "config.json").is_file()
    df_block_size = 0
    df_taps: tuple[int, ...] = ()
    df_cnt = 0
    long_df = 0
    if df_present:
        df_cfg = json.loads((df_dir / "config.json").read_text(encoding="utf-8"))
        df_inner = df_cfg.get("dflash_config") if isinstance(df_cfg.get("dflash_config"), dict) else {}
        df_block_size = int(df_inner.get("block_size") or df_cfg.get("block_size") or 0)
        df_taps = tuple(int(x) for x in (df_inner.get("target_layer_ids") or df_cfg.get("target_layer_ids") or []))
        if df_block_size not in (8, 16):
            errors.append(f"unexpected dflash block_size={df_block_size} (expected 8 or 16)")
        if not df_taps or any(t < 0 or t >= num_layers for t in df_taps):
            errors.append(f"invalid dflash target_layer_ids={list(df_taps)} for num_layers={num_layers}")
        df_wm, _ = _read_weight_map_keys(df_dir)
        for dk in df_wm:
            df_cnt += 1
            dn = normalize_mimo_dflash_tensor_name(dk)
            if len(dn.encode("utf-8")) > 47:
                long_df += 1
        if long_df > 0:
            errors.append(f"long_dflash_names={long_df} (> 47 UTF-8 bytes)")

    rep = MimoV2MetadataReport(
        metadata_dir=meta_dir,
        variant=variant,
        model_type=model_type,
        hidden_size=hidden,
        vocab_size=vocab_size,
        num_hidden_layers=num_layers,
        swa_layers=swa_layers,
        global_layers=global_layers,
        global_layer_indices=global_indices,
        sliding_window_size=sliding_win,
        head_dim=hd,
        v_head_dim=vd,
        global_q_heads=g_q,
        global_kv_heads=g_kv,
        swa_q_heads=s_q,
        swa_kv_heads=s_kv,
        rope_theta=rope_theta,
        swa_rope_theta=swa_rope_theta,
        partial_rotary_factor=prf,
        rope_dim=rope_dim,
        attention_value_scale=v_scale,
        add_swa_attention_sink_bias=add_swa_sink,
        add_full_attention_sink_bias=add_full_sink,
        intermediate_size=inter,
        moe_intermediate_size=moe_inter,
        n_routed_experts=n_routed,
        num_experts_per_tok=topk,
        n_shared_experts=n_shared,
        moe_dense_layers=moe_dense_cnt,
        moe_layers=moe_layer_cnt,
        mtp_layers=mtp_layers,
        dflash_present=df_present,
        dflash_block_size=df_block_size,
        dflash_target_layer_ids=df_taps,
        active_params=active_params,
        routed_expert_params=routed_params,
        total_params=total_params,
        global_kv_bytes_per_token_fp16=global_kv_bytes,
        swa_kv_bytes_per_token_fp16=swa_kv_bytes,
        swa_kv_window_cap_bytes_fp16=swa_kv_cap,
        qkv_tp_degree_inferred=tp_deg,
        backbone_tensor_count=bb_cnt,
        vision_tensor_count=vis_cnt,
        audio_tensor_count=aud_cnt,
        mtp_tensor_count=mtp_cnt,
        dflash_tensor_count=df_cnt,
        long_backbone_names=long_bb,
        long_vision_names=long_vis,
        long_mtp_names=long_mtp,
        long_dflash_names=long_df,
        max_backbone_name_bytes=max_bb_bytes,
        header_summary_shards=hdr_shards,
        errors=tuple(errors),
    )
    if strict and rep.errors:
        raise ValueError(f"mimo_v2 W28 metadata validation failed for {meta_dir}: " + "; ".join(rep.errors))
    return rep


def validate_qwen35_metadata_dir(meta_dir: pathlib.Path, *, strict: bool = True) -> Qwen35MetadataReport:
    """Validate staged HF metadata + header manifests for `MiMo-V2.6-Distill-Qwen-9B` (`qwen3_5`)."""
    meta_dir = pathlib.Path(meta_dir)
    cfg_p = meta_dir / "config.json"
    if not cfg_p.is_file():
        raise FileNotFoundError(f"config.json not found in {meta_dir}")
    raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    cfg = _effective_mimo_cfg(raw_cfg)
    vis_cfg = raw_cfg.get("vision_config") if isinstance(raw_cfg.get("vision_config"), dict) else {}
    qc = raw_cfg.get("quantization") or raw_cfg.get("quantization_config")
    quant_mode = None
    if isinstance(qc, dict):
        if qc.get("mode") or qc.get("quant_method"):
            quant_mode = str(qc.get("mode") or qc.get("quant_method"))
        elif qc.get("bits") and qc.get("group_size"):
            quant_mode = f"mlx_{int(qc['bits'])}bit_g{int(qc['group_size'])}"
        else:
            quant_mode = "quantized"
    errors: list[str] = []

    model_type = str(raw_cfg.get("model_type") or cfg.get("model_type") or "")
    if model_type not in ("qwen3_5", "qwen3_5_text", "qwen35"):
        errors.append(f"unexpected model_type {model_type!r} (expected 'qwen3_5')")

    hidden = int(cfg.get("hidden_size", 0) or 0)
    inter = int(cfg.get("intermediate_size", 0) or 0)
    vocab_size = int(cfg.get("vocab_size", 0) or 0)
    num_layers = int(cfg.get("num_hidden_layers", 0) or 0)
    if hidden <= 0 or inter <= 0 or vocab_size <= 0 or num_layers <= 0:
        errors.append(f"invalid qwen3_5 dims hidden={hidden}, inter={inter}, vocab={vocab_size}, layers={num_layers}")

    layer_types = [str(x) for x in (cfg.get("layer_types") or [])]
    if not layer_types and num_layers > 0:
        interval = int(cfg.get("full_attention_interval", 4) or 4)
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(num_layers)
        ]
    if len(layer_types) != num_layers:
        errors.append(f"layer_types len {len(layer_types)} != num_hidden_layers {num_layers}")

    lin_cnt = sum(1 for t in layer_types if t == "linear_attention")
    full_indices = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    full_cnt = len(full_indices)

    q_heads = int(cfg.get("num_attention_heads", 0) or 0)
    kv_heads = int(cfg.get("num_key_value_heads", 0) or 0)
    hd = int(cfg.get("head_dim", 0) or 0)
    attn_gate = bool(cfg.get("attn_output_gate", True))

    rope_p = cfg.get("rope_parameters") if isinstance(cfg.get("rope_parameters"), dict) else {}
    prf = float(rope_p.get("partial_rotary_factor") or cfg.get("partial_rotary_factor") or 0.25)
    rope_theta = float(rope_p.get("rope_theta") or cfg.get("rope_theta") or 10000000.0)
    rope_dim = int(hd * prf)
    mrope_sec = tuple(int(x) for x in (rope_p.get("mrope_section") or ()))

    lk_heads = int(cfg.get("linear_num_key_heads", 0) or 0)
    lv_heads = int(cfg.get("linear_num_value_heads", 0) or 0)
    lk_dim = int(cfg.get("linear_key_head_dim", 0) or 0)
    lv_dim = int(cfg.get("linear_value_head_dim", 0) or 0)
    l_conv = int(cfg.get("linear_conv_kernel_dim", 4) or 4)
    if lin_cnt > 0 and (lk_heads <= 0 or lv_heads <= 0 or lk_dim <= 0 or lv_dim <= 0):
        errors.append("linear_attention layers present but GDN head dims missing/zero")

    tie_emb = bool(cfg.get("tie_word_embeddings", False))
    emb_params = vocab_size * hidden * (1 if tie_emb else 2)
    q_mult = 2 if attn_gate else 1
    full_attn_p = hidden * (q_heads * hd * q_mult + 2 * kv_heads * hd + q_heads * hd)
    gdn_qkv_dim = 2 * lk_heads * lk_dim + lv_heads * lv_dim
    gdn_z_dim = lv_heads * lv_dim
    gdn_p = hidden * (gdn_qkv_dim + gdn_z_dim + 2 * lv_heads + gdn_z_dim) + gdn_qkv_dim * l_conv
    mlp_p = 3 * hidden * inter
    total_params = emb_params + full_cnt * full_attn_p + lin_cnt * gdn_p + num_layers * mlp_p

    full_kv_bytes = full_cnt * kv_heads * (2 * hd) * 2
    gdn_state_bytes = lin_cnt * (
        lv_heads * lk_dim * lv_dim * 4
        + gdn_qkv_dim * max(0, l_conv - 1) * 2
    )

    wm, _ = _read_weight_map_keys(meta_dir)
    hdr_tensors, hdr_shards = _read_header_summary_tensors(meta_dir)
    all_keys = sorted(set(wm.keys()) | set(hdr_tensors.keys()))
    bb_cnt = vis_cnt = long_bb = long_vis = max_bb_bytes = 0
    for raw_k in all_keys:
        if is_vision_tensor(raw_k):
            vis_cnt += 1
            vn = normalize_vision_name(raw_k)
            if len(vn.encode("utf-8")) > 47:
                long_vis += 1
            continue
        bb_cnt += 1
        bn = normalize_qwen35_tensor_name(raw_k)
        blen = len(bn.encode("utf-8"))
        if blen > max_bb_bytes:
            max_bb_bytes = blen
        if blen > 47:
            long_bb += 1
    if long_bb > 0:
        errors.append(f"long_backbone_names={long_bb} (> 47 UTF-8 bytes)")
    if long_vis > 0:
        errors.append(f"long_vision_names={long_vis} (> 47 UTF-8 bytes)")

    rep = Qwen35MetadataReport(
        metadata_dir=meta_dir,
        model_type=model_type,
        hidden_size=hidden,
        intermediate_size=inter,
        vocab_size=vocab_size,
        num_hidden_layers=num_layers,
        linear_attention_layers=lin_cnt,
        full_attention_layers=full_cnt,
        full_attention_indices=full_indices,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
        head_dim=hd,
        attn_output_gate=attn_gate,
        partial_rotary_factor=prf,
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        mrope_section=mrope_sec,
        linear_num_key_heads=lk_heads,
        linear_num_value_heads=lv_heads,
        linear_key_head_dim=lk_dim,
        linear_value_head_dim=lv_dim,
        linear_conv_kernel_dim=l_conv,
        rmsnorm_unit_offset=True,
        linear_attn_norm_offset=0.0,
        tie_word_embeddings=tie_emb,
        quantization_mode=quant_mode,
        vision_layers=int(vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0),
        vision_hidden_size=int(vis_cfg.get("hidden_size") or 0),
        active_params=total_params,
        total_params=total_params,
        full_kv_bytes_per_token_fp16=full_kv_bytes,
        gdn_state_bytes_fp32_fp16=gdn_state_bytes,
        backbone_tensor_count=bb_cnt,
        vision_tensor_count=vis_cnt,
        long_backbone_names=long_bb,
        long_vision_names=long_vis,
        max_backbone_name_bytes=max_bb_bytes,
        header_summary_shards=hdr_shards,
        errors=tuple(errors),
    )
    if strict and rep.errors:
        raise ValueError(f"qwen3_5 W28 metadata validation failed for {meta_dir}: " + "; ".join(rep.errors))
    return rep


# ---------------------------------------------------------------------------
# Embedded Runtime `config.json` Builders (locked to `#502`)
# ---------------------------------------------------------------------------


def build_mimo_v2_config_json(
    src_cfg: Mapping[str, Any],
    *,
    num_hidden_layers_override: int | None = None,
    cold_experts_count: int = 0,
) -> dict[str, Any]:
    """Build canonical `.amf` runtime `config.json` for `mimo_v2` (`Flash` / `Pro`)."""
    cfg = _effective_mimo_cfg(src_cfg)
    n_layers = int(num_hidden_layers_override or cfg.get("num_hidden_layers") or 48)
    pat = list(cfg.get("hybrid_layer_pattern") or [0 if i == 0 else 1 for i in range(n_layers)])[:n_layers]
    ltypes = ["full_attention" if int(x) == 0 else "sliding_attention" for x in pat]
    moe_freq = list(cfg.get("moe_layer_freq") or [0 if i == 0 else 1 for i in range(n_layers)])[:n_layers]
    hd = int(cfg.get("head_dim") or 192)
    vd = int(cfg.get("v_head_dim") or 128)
    prf = float(cfg.get("partial_rotary_factor") or 0.334)
    local_ids = [i for i, x in enumerate(pat) if int(x) != 0]
    win = int(cfg.get("sliding_window_size") or cfg.get("sliding_window") or 128)
    out: dict[str, Any] = {
        "model_type": "mimo_v2",
        "architectures": ["MiMoV2ForCausalLM"],
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg.get("intermediate_size") or int(cfg["hidden_size"]) * 4),
        "moe_intermediate_size": int(cfg.get("moe_intermediate_size") or 2048),
        "num_hidden_layers": n_layers,
        "vocab_size": int(cfg["vocab_size"]),
        "layer_types": ltypes,
        "hybrid_layer_pattern": [int(x) for x in pat],
        "local_layer_ids": local_ids,
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "head_dim": hd,
        "v_head_dim": vd,
        "swa_num_attention_heads": int(cfg.get("swa_num_attention_heads") or cfg["num_attention_heads"]),
        "swa_num_key_value_heads": int(cfg.get("swa_num_key_value_heads") or cfg["num_key_value_heads"]),
        "swa_head_dim": int(cfg.get("swa_head_dim") or hd),
        "swa_v_head_dim": int(cfg.get("swa_v_head_dim") or vd),
        "sliding_window": win,
        "sliding_window_size": win,
        "moe_layer_freq": [int(x) for x in moe_freq],
        "n_routed_experts": int(cfg.get("n_routed_experts") or 0),
        "num_experts_per_tok": int(cfg.get("num_experts_per_tok") or 8),
        "n_shared_experts": cfg.get("n_shared_experts"),
        "scoring_func": str(cfg.get("scoring_func") or "sigmoid"),
        "topk_method": str(cfg.get("topk_method") or "noaux_tc"),
        "norm_topk_prob": bool(cfg.get("norm_topk_prob", True)),
        "routed_scaling_factor": cfg.get("routed_scaling_factor"),
        "n_group": int(cfg.get("n_group") or 1),
        "topk_group": int(cfg.get("topk_group") or 1),
        "rope_theta": float(cfg.get("rope_theta") or 10000000.0),
        "swa_rope_theta": float(cfg.get("swa_rope_theta") or 10000.0),
        "partial_rotary_factor": prf,
        "rope_dim": int(hd * prf),
        "rms_norm_eps": float(cfg.get("layernorm_epsilon") or cfg.get("rms_norm_eps") or 1e-6),
        "attention_value_scale": float(cfg.get("attention_value_scale") or 1.0),
        "add_swa_attention_sink_bias": bool(cfg.get("add_swa_attention_sink_bias", True)),
        "add_full_attention_sink_bias": bool(cfg.get("add_full_attention_sink_bias", False)),
        "attention_projection_layout": "split",
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "max_position_embeddings": int(cfg.get("max_position_embeddings") or 262144),
        "num_nextn_predict_layers": int(cfg.get("num_nextn_predict_layers") or 0),
    }
    if cold_experts_count > 0:
        out["cold_experts_count"] = int(cold_experts_count)
    return out


def build_mimo_mtp_config_dict(src_cfg: Mapping[str, Any], *, mtp_layers: int = 3) -> dict[str, Any]:
    """Build locked `.amf` metadata `config.json` for `3L` chained native MTP (`#502`)."""
    cfg = _effective_mimo_cfg(src_cfg)
    hd = int(cfg.get("swa_head_dim") or cfg.get("head_dim") or 192)
    vd = int(cfg.get("swa_v_head_dim") or cfg.get("v_head_dim") or 128)
    prf = float(cfg.get("partial_rotary_factor") or 0.334)
    win = int(cfg.get("sliding_window_size") or cfg.get("sliding_window") or 128)
    h_sz = int(cfg.get("hidden_size") or 4096)
    g_q = int(cfg.get("num_attention_heads") or 64)
    g_kv = int(cfg.get("num_key_value_heads") or 8)
    return {
        "model_type": "mimo_v2_mtp",
        "format": "mtp",
        "mtp_chained": True,
        "architectures": ["MiMoV2MTPDraftModel"],
        "hidden_size": h_sz,
        "intermediate_size": int(cfg.get("intermediate_size") or h_sz * 4),
        "num_hidden_layers": int(mtp_layers),
        "chained_mtp_steps": int(mtp_layers),
        "vocab_size": int(cfg.get("vocab_size") or 152576),
        "num_attention_heads": int(cfg.get("swa_num_attention_heads") or g_q),
        "num_key_value_heads": int(cfg.get("swa_num_key_value_heads") or g_kv),
        "head_dim": hd,
        "v_head_dim": vd,
        "sliding_window": win,
        "sliding_window_size": win,
        "rope_theta": float(cfg.get("swa_rope_theta") or 10000.0),
        "partial_rotary_factor": prf,
        "rope_dim": int(hd * prf),
        "rms_norm_eps": float(cfg.get("layernorm_epsilon") or cfg.get("rms_norm_eps") or 1e-6),
        "attention_value_scale": float(cfg.get("attention_value_scale") or 0.707),
        "add_swa_attention_sink_bias": bool(cfg.get("add_swa_attention_sink_bias", True)),
        "attention_projection_layout": "split",
        "is_swa": True,
        "is_moe": False,
    }


def build_mimo_dflash_config_dict(
    src_cfg: Mapping[str, Any],
    *,
    target_cfg: Mapping[str, Any] | None = None,
    block_size_override: int | None = None,
) -> dict[str, Any]:
    """Build `DFlareConfig.fromJson`-compatible `dflash` draft `config.json` (`#502`)."""
    dcfg = _effective_mimo_cfg(src_cfg)
    inner = dcfg.get("dflash_config") if isinstance(dcfg.get("dflash_config"), Mapping) else {}
    tcfg = _effective_mimo_cfg(target_cfg) if target_cfg else {}
    hd = int(dcfg.get("head_dim") or 128)
    prf = float(dcfg.get("partial_rotary_factor") or 0.5)
    taps = inner.get("target_layer_ids") or dcfg.get("target_layer_ids") or [0, 11, 23, 35, 47]
    bsize = int(block_size_override or inner.get("block_size") or dcfg.get("block_size") or 8)
    win = int(dcfg.get("sliding_window") or dcfg.get("sliding_window_size") or 1024)
    return {
        "model_type": "dflash",
        "architectures": ["DFlashForCausalLM"],
        "draft_kind": "dflash",
        "dflare_format": "dflash",
        "hidden_size": int(dcfg.get("hidden_size") or tcfg.get("hidden_size") or 4096),
        "intermediate_size": int(dcfg.get("intermediate_size") or 16384),
        "num_hidden_layers": int(dcfg.get("num_hidden_layers") or 5),
        "num_attention_heads": int(dcfg.get("num_attention_heads") or tcfg.get("num_attention_heads") or 64),
        "num_key_value_heads": int(dcfg.get("num_key_value_heads") or 8),
        "head_dim": hd,
        "v_head_dim": int(dcfg.get("v_head_dim") or hd),
        "vocab_size": int(dcfg.get("vocab_size") or tcfg.get("vocab_size") or 152576),
        "block_size": bsize,
        "mask_token_id": int(inner.get("mask_token_id") or dcfg.get("mask_token_id") or 151675),
        "target_layer_ids": [int(x) for x in taps],
        "sliding_window": win,
        "sliding_window_size": win,
        "rope_theta": float(dcfg.get("rope_theta") or 10000.0),
        "partial_rotary_factor": prf,
        "rope_dim": int(hd * prf),
        "rms_norm_eps": float(dcfg.get("rms_norm_eps") or 1e-6),
        "attention_value_scale": float(inner.get("attention_value_scale") or dcfg.get("attention_value_scale") or 0.612),
        "add_swa_attention_sink_bias": bool(
            inner.get("attention_sink_bias", dcfg.get("add_swa_attention_sink_bias", True))
        ),
        "attention_causal": False,
        "markov_rank": 0,
        "markov_head_type": "none",
    }


def build_qwen35_config_json(src_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build canonical `.amf` runtime `config.json` for `qwen3_5` (`Distill-Qwen-9B`, `#502`)."""
    cfg = _effective_mimo_cfg(src_cfg)
    n_layers = int(cfg["num_hidden_layers"])
    interval = int(cfg.get("full_attention_interval") or 4)
    ltypes = [str(x) for x in (cfg.get("layer_types") or [])]
    if len(ltypes) != n_layers:
        ltypes = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n_layers)]
    rope_p = cfg.get("rope_parameters") if isinstance(cfg.get("rope_parameters"), Mapping) else {}
    hd = int(cfg["head_dim"])
    prf = float(rope_p.get("partial_rotary_factor") or cfg.get("partial_rotary_factor") or 0.25)
    out: dict[str, Any] = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_hidden_layers": n_layers,
        "vocab_size": int(cfg["vocab_size"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "head_dim": hd,
        "attn_output_gate": bool(cfg.get("attn_output_gate", True)),
        "layer_types": ltypes,
        "full_attention_interval": interval,
        "linear_num_key_heads": int(cfg["linear_num_key_heads"]),
        "linear_num_value_heads": int(cfg["linear_num_value_heads"]),
        "linear_key_head_dim": int(cfg["linear_key_head_dim"]),
        "linear_value_head_dim": int(cfg["linear_value_head_dim"]),
        "linear_conv_kernel_dim": int(cfg.get("linear_conv_kernel_dim") or 4),
        "rope_theta": float(rope_p.get("rope_theta") or cfg.get("rope_theta") or 10000000.0),
        "partial_rotary_factor": prf,
        "rope_dim": int(hd * prf),
        "mrope_interleaved": bool(rope_p.get("mrope_interleaved", True)),
        "mrope_section": [int(x) for x in (rope_p.get("mrope_section") or [11, 11, 10])],
        "rms_norm_eps": float(cfg.get("rms_norm_eps") or 1e-6),
        "rmsnorm_unit_offset": True,
        "linear_attn_norm_offset": 0.0,
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "max_position_embeddings": int(cfg.get("max_position_embeddings") or 262144),
    }
    if isinstance(src_cfg.get("vision_config"), Mapping):
        vc = src_cfg["vision_config"]
        out["vision_config"] = {
            "model_type": str(vc.get("model_type") or "qwen3_5"),
            "depth": int(vc.get("depth") or vc.get("num_hidden_layers") or 27),
            "hidden_size": int(vc.get("hidden_size") or 1152),
            "intermediate_size": int(vc.get("intermediate_size") or 4304),
            "num_heads": int(vc.get("num_heads") or 16),
            "patch_size": int(vc.get("patch_size") or 16),
            "temporal_patch_size": int(vc.get("temporal_patch_size") or 2),
            "spatial_merge_size": int(vc.get("spatial_merge_size") or 2),
            "out_hidden_size": int(vc.get("out_hidden_size") or cfg["hidden_size"]),
        }
    return out


# ---------------------------------------------------------------------------
# Internal Helpers: Companion Metadata + Cold Experts + Companion Exporters
# ---------------------------------------------------------------------------


def _resolve_companion_files(
    src_path: pathlib.Path,
    *,
    config_path: pathlib.Path | None = None,
    tokenizer_path: pathlib.Path | None = None,
    tokenizer_config_path: pathlib.Path | None = None,
    chat_template_path: pathlib.Path | None = None,
) -> tuple[pathlib.Path | None, pathlib.Path | None, pathlib.Path | None, pathlib.Path | None]:
    base_dir = src_path if src_path.is_dir() else src_path.parent
    cfg = config_path if config_path is not None else (base_dir / "config.json")
    tok = tokenizer_path if tokenizer_path is not None else (base_dir / "tokenizer.json")
    tcfg = tokenizer_config_path if tokenizer_config_path is not None else (base_dir / "tokenizer_config.json")
    tpl = chat_template_path if chat_template_path is not None else (base_dir / "chat_template.jinja")
    return (
        cfg if cfg.is_file() else None,
        tok if tok.is_file() else None,
        tcfg if tcfg.is_file() else None,
        tpl if tpl.is_file() else None,
    )


def _select_mimo_cold_experts(
    *,
    n_layers: int,
    n_experts: int,
    moe_freq: Sequence[int],
    tier_profile: MimoV2TierProfile | None,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int | None = None,
    heat_profile: pathlib.Path | None = None,
    all_experts_cold: bool = False,
) -> set[tuple[int, int]]:
    out = set(cold_experts)
    if heat_profile is not None:
        out.update(
            load_cold_experts_from_heat_profile(
                pathlib.Path(heat_profile),
                cold_experts_per_layer=cold_experts_per_layer,
            )
        )
        return out
    eff_per_layer = cold_experts_per_layer
    if eff_per_layer is None and tier_profile is not None and tier_profile.allow_nvme_cold_experts and not all_experts_cold:
        if n_experts >= 32:
            eff_per_layer = max(1, int(round(n_experts * 0.80)))
    if eff_per_layer is not None and eff_per_layer > 0 and n_experts > 0:
        keep_hot = max(0, n_experts - eff_per_layer)
        for layer_idx in range(n_layers):
            if layer_idx < len(moe_freq) and int(moe_freq[layer_idx]) == 0:
                continue
            for e_idx in range(keep_hot, n_experts):
                out.add((layer_idx, e_idx))
    return out


@dataclass(frozen=True)
class MimoV2ConvertReport:
    """Summary returned by `convert_mimo_v2_checkpoint`."""

    backbone_amf: pathlib.Path
    backbone_metas: tuple[PackedTensorMeta, ...]
    mtp_amf: pathlib.Path | None = None
    mtp_metas: tuple[PackedTensorMeta, ...] = ()
    dflash_amf: pathlib.Path | None = None
    dflash_metas: tuple[PackedTensorMeta, ...] = ()
    vision_amf: pathlib.Path | None = None
    vision_metas: tuple[PackedTensorMeta, ...] = ()
    cold_expert_count: int = 0
    qkv_tp_degree: int = 1
    stripped_audio_tensors: int = 0
    stripped_vision_tensors: int = 0
    config_json_path: pathlib.Path | None = None


@dataclass(frozen=True)
class Qwen35ConvertReport:
    """Summary returned by `convert_qwen35_checkpoint`."""

    backbone_amf: pathlib.Path
    backbone_metas: tuple[PackedTensorMeta, ...]
    vision_amf: pathlib.Path | None = None
    vision_metas: tuple[PackedTensorMeta, ...] = ()
    stripped_vision_tensors: int = 0
    config_json_path: pathlib.Path | None = None


def export_mimo_mtp_from_safetensors(
    mtp_input: pathlib.Path,
    output_amf: pathlib.Path,
    *,
    target_config: Mapping[str, Any] | None = None,
    model_id: str = "mimo-v2-mtp",
    dtype_mode: str = "auto",
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    use_rco: bool = True,
    qkv_tp_degree: int | None = None,
    embed_metadata: bool = True,
    config_output: pathlib.Path | None = None,
    workers: int = 1,
) -> list[PackedTensorMeta]:
    """Convert `model_mtp.safetensors` (`3L` native MTP) into a standalone 16 KiB-aligned `.amf`."""
    mtp_p = pathlib.Path(mtp_input)
    if mtp_p.is_dir():
        cand = mtp_p / "model_mtp.safetensors"
        if not cand.is_file():
            raise FileNotFoundError(f"model_mtp.safetensors not found in {mtp_p}")
        mtp_p = cand
    raw = _collect_tensors(resolve_checkpoint(mtp_p).shard_paths)
    tcfg = dict(target_config) if target_config is not None else {}
    if not tcfg:
        cfg_p = mtp_p.parent / "config.json"
        if cfg_p.is_file():
            tcfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    mtp_layer_ids = sorted(
        {
            int(m.group(1))
            for k in raw
            for m in [re.match(r"^(?:model\.)?mtp\.layers\.(\d+)\.", k)]
            if m is not None
        }
    )
    n_mtp = max(1, len(mtp_layer_ids))
    eff_tp = infer_mimo_qkv_tp_degree(tcfg, raw, explicit_tp=qkv_tp_degree)
    if qkv_tp_degree is None and eff_tp == 1 and tcfg:
        eff_t = _effective_mimo_cfg(tcfg)
        for k, t in raw.items():
            if _MTP_QKV_SCALE_RE.match(k) and int(eff_t.get("num_hidden_layers", 0) or 0) == 48 and t.shape[0] == 116:
                eff_tp = 4
                break
            if _MTP_QKV_SCALE_RE.match(k) and int(eff_t.get("num_hidden_layers", 0) or 0) == 70 and t.shape[0] == 216:
                eff_tp = 8
                break

    mtp_cfg_dict = (
        build_mimo_mtp_config_dict(tcfg, mtp_layers=n_mtp)
        if tcfg
        else {"model_type": "mimo_v2_mtp", "format": "mtp", "mtp_chained": True, "num_hidden_layers": n_mtp}
    )
    mtp_cfg_bytes = json.dumps(mtp_cfg_dict, indent=2).encode("utf-8") + b"\n"
    if config_output is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_bytes(mtp_cfg_bytes)

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_cfg = pathlib.Path(tmp_d) / "mtp_config.json"
        tmp_cfg.write_bytes(mtp_cfg_bytes)
        metas = pack_amf_from_safetensors(
            mtp_p,
            output_amf,
            model_id=model_id[:63],
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            target_bits=target_bits,
            use_rco=use_rco,
            strip_vision=True,
            embed_metadata=embed_metadata,
            config_path=tmp_cfg if embed_metadata else None,
            workers=workers,
            mimo_config=tcfg,
            mimo_qkv_tp_degree=eff_tp,
            mimo_mtp_standalone=True,
        )
    if embed_metadata:
        metas = embed_metadata_into_amf(output_amf, config_bytes=mtp_cfg_bytes, output_path=output_amf)
    return metas


def export_mimo_dflash(
    dflash_input: pathlib.Path,
    output_amf: pathlib.Path,
    *,
    target_config: Mapping[str, Any] | None = None,
    block_size_override: int | None = None,
    model_id: str = "mimo-v2-dflash",
    dtype_mode: str = "auto",
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    use_rco: bool = True,
    embed_metadata: bool = True,
    config_output: pathlib.Path | None = None,
    workers: int = 1,
) -> list[PackedTensorMeta]:
    """Export `DFlash` draft bundle (`dflash/dflash_draft_model.safetensors` or `arch=dflash` `.gguf`) to `.amf`."""
    in_p = pathlib.Path(dflash_input)
    if in_p.is_dir() and (in_p / "dflash").is_dir():
        in_p = in_p / "dflash"
    if in_p.is_file() and in_p.suffix.lower() == ".gguf":
        from aminfer_tools.gguf import pack_amf_from_gguf

        return pack_amf_from_gguf(
            in_p,
            output_amf,
            model_id=model_id[:63],
            dtype_mode=dtype_mode,
            embed_metadata=embed_metadata,
            draft_config_output=config_output,
        )

    base_dir = in_p if in_p.is_dir() else in_p.parent
    st_in = in_p
    if st_in.is_dir():
        if (st_in / "dflash_draft_model.safetensors").is_file():
            st_in = st_in / "dflash_draft_model.safetensors"
        elif not (st_in / "model.safetensors").is_file() and not (st_in / "model.safetensors.index.json").is_file():
            cands = sorted(st_in.glob("*.safetensors"))
            if len(cands) == 1:
                st_in = cands[0]
    df_cfg_p = base_dir / "config.json"
    df_cfg = json.loads(df_cfg_p.read_text(encoding="utf-8")) if df_cfg_p.is_file() else {}
    tcfg = dict(target_config) if target_config is not None else {}
    if not tcfg and (base_dir.parent / "config.json").is_file():
        tcfg = json.loads((base_dir.parent / "config.json").read_text(encoding="utf-8"))
    cfg_dict = build_mimo_dflash_config_dict(df_cfg, target_cfg=tcfg, block_size_override=block_size_override)
    cfg_bytes = json.dumps(cfg_dict, indent=2).encode("utf-8") + b"\n"
    if config_output is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_bytes(cfg_bytes)

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_cfg = pathlib.Path(tmp_d) / "dflash_config.json"
        tmp_cfg.write_bytes(cfg_bytes)
        metas = pack_amf_from_safetensors(
            st_in,
            output_amf,
            model_id=model_id[:63],
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            target_bits=target_bits,
            use_rco=use_rco,
            strip_vision=True,
            embed_metadata=embed_metadata,
            config_path=tmp_cfg if embed_metadata else None,
            workers=workers,
            mimo_dflash_mode=True,
        )
    if embed_metadata:
        metas = embed_metadata_into_amf(output_amf, config_bytes=cfg_bytes, output_path=output_amf)
    return metas


# ---------------------------------------------------------------------------
# Checkpoint Converters (`mimo_v2` & `qwen3_5`)
# ---------------------------------------------------------------------------


def convert_mimo_v2_checkpoint(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str | None = None,
    dtype_mode: str = "auto",
    group_size: int = 64,
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    target_wired_gib: float | None = None,
    target_tier: str | None = None,
    use_rco: bool = True,
    manifest_path: pathlib.Path | None = None,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int | None = None,
    heat_profile: pathlib.Path | None = None,
    all_experts_cold: bool = False,
    qkv_tp_degree: int | None = None,
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
    mtp_input: pathlib.Path | None = None,
    mtp_output: pathlib.Path | None = None,
    mtp_config_output: pathlib.Path | None = None,
    dflash_input: pathlib.Path | None = None,
    dflash_output: pathlib.Path | None = None,
    dflash_config_output: pathlib.Path | None = None,
    dflash_block_size: int | None = None,
    config_output: pathlib.Path | None = None,
    config_path: pathlib.Path | None = None,
    tokenizer_path: pathlib.Path | None = None,
    tokenizer_config_path: pathlib.Path | None = None,
    chat_template_path: pathlib.Path | None = None,
    strict_vocab_check: bool = True,
    imatrix_path: pathlib.Path | None = None,
    enforce_backbone_floor: bool = True,
    max_q2_share: float | None = None,
    allow_lowbit_q2_retranscode: bool = True,
    fine_rungs: bool | None = None,
    embed_metadata: bool = True,
    workers: int = 1,
) -> MimoV2ConvertReport:
    """Convert `MiMo-V2.6-Flash` / `Pro` (+ `3L` MTP + `DFlash` + optional ViT) to 16 KiB-aligned `.amf`."""
    in_p = pathlib.Path(input_path)
    out_p = pathlib.Path(output_path)
    tier_prof = resolve_mimo_tier_profile(target_tier)
    eff_model_id = (model_id or (in_p.name if in_p.is_dir() else out_p.stem))[:63]

    if in_p.is_file() and in_p.suffix.lower() == ".gguf":
        from aminfer_tools.gguf import pack_amf_from_gguf, read_gguf_header

        sh = read_gguf_header(in_p)
        arch = str(sh.kv.get("general.architecture") or "mimo2")
        n_layers = int(sh.kv.get(f"{arch}.block_count", 48) or 48)
        n_exp = int(sh.kv.get(f"{arch}.expert_count", 0) or 0)
        cold_set = _select_mimo_cold_experts(
            n_layers=n_layers,
            n_experts=n_exp,
            moe_freq=[0 if i == 0 else 1 for i in range(n_layers)],
            tier_profile=tier_prof,
            cold_experts=cold_experts,
            cold_experts_per_layer=cold_experts_per_layer,
            heat_profile=heat_profile,
            all_experts_cold=all_experts_cold,
        )
        metas = pack_amf_from_gguf(
            in_p,
            out_p,
            model_id=eff_model_id,
            dtype_mode=dtype_mode,
            cold_experts=cold_set,
            all_experts_cold=all_experts_cold,
            embed_metadata=embed_metadata,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=tokenizer_config_path,
            chat_template_path=chat_template_path,
            vision_block_dtype=vision_block_dtype,
            group_size=group_size,
            strict_vocab_check=strict_vocab_check,
        )
        mtp_metas: list[PackedTensorMeta] = []
        if mtp_output is not None and mtp_input is not None:
            mtp_metas = export_mimo_mtp_from_safetensors(
                mtp_input,
                mtp_output,
                model_id=f"{eff_model_id[:55]}-mtp",
                dtype_mode=dtype_mode,
                embed_metadata=embed_metadata,
                config_output=mtp_config_output,
                workers=workers,
            )
        elif mtp_output is not None and arch == "mimo2" and in_p.name.lower().startswith("mtp-"):
            mtp_metas = pack_amf_from_gguf(
                in_p,
                mtp_output,
                model_id=f"{eff_model_id[:55]}-mtp",
                dtype_mode=dtype_mode,
                embed_metadata=embed_metadata,
                draft_config_output=mtp_config_output,
                mimo_mtp_mode=True,
            )
        df_metas: list[PackedTensorMeta] = []
        if dflash_output is not None and dflash_input is not None:
            df_metas = export_mimo_dflash(
                dflash_input,
                dflash_output,
                block_size_override=dflash_block_size,
                model_id=f"{eff_model_id[:52]}-dflash",
                dtype_mode=dtype_mode,
                embed_metadata=embed_metadata,
                config_output=dflash_config_output,
                workers=workers,
            )
        return MimoV2ConvertReport(
            backbone_amf=out_p,
            backbone_metas=tuple(metas),
            mtp_amf=mtp_output if mtp_metas else None,
            mtp_metas=tuple(mtp_metas),
            dflash_amf=dflash_output if df_metas else None,
            dflash_metas=tuple(df_metas),
            cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
        )

    if in_p.is_file() and in_p.suffix.lower() == ".amf":
        from aminfer_tools.amf import AmfReader, gsq_convert_with_vision, read_amf_metadata

        raw_cfg_b = pathlib.Path(config_path).read_bytes() if config_path is not None else read_amf_metadata(in_p, META_CONFIG_JSON)
        raw_cfg = json.loads(raw_cfg_b.decode("utf-8")) if raw_cfg_b else {}
        eff_cfg = _effective_mimo_cfg(raw_cfg)
        n_layers = int(eff_cfg.get("num_hidden_layers") or 48)
        n_exp = int(eff_cfg.get("n_routed_experts") or 0)
        moe_freq = list(eff_cfg.get("moe_layer_freq") or [0 if i == 0 else 1 for i in range(n_layers)])
        cold_set = _select_mimo_cold_experts(
            n_layers=n_layers,
            n_experts=n_exp,
            moe_freq=moe_freq,
            tier_profile=tier_prof,
            cold_experts=cold_experts,
            cold_experts_per_layer=cold_experts_per_layer,
            heat_profile=heat_profile,
            all_experts_cold=all_experts_cold,
        )
        eff_target_bits = target_bits
        if tier_prof is not None and quant_profile == "gsq-rco" and eff_target_bits is None and target_wired_gib is None:
            eff_target_bits = tier_prof.target_bits_default
        eff_fine_rungs = (tier_prof is not None and quant_profile == "gsq-rco") if fine_rungs is None else fine_rungs
        metas = gsq_convert_with_vision(
            in_p,
            out_p,
            target_bits=eff_target_bits,
            model_id=eff_model_id,
            group_size=group_size,
            dtype_mode=dtype_mode,
            heat_profile=heat_profile,
            cold_experts=cold_set,
            cold_experts_per_layer=cold_experts_per_layer,
            all_experts_cold=all_experts_cold,
            use_rco=use_rco,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=strip_vision,
            include_vision=include_vision,
            vision_sidecar=vision_sidecar,
            vision_block_dtype=vision_block_dtype,
            imatrix_path=imatrix_path,
            enforce_backbone_floor=enforce_backbone_floor,
            max_q2_share=max_q2_share,
            allow_lowbit_q2_retranscode=allow_lowbit_q2_retranscode,
            fine_rungs=eff_fine_rungs,
            mimo_config=eff_cfg or None,
        )
        vis_metas: tuple[PackedTensorMeta, ...] = ()
        if vision_sidecar is not None and pathlib.Path(vision_sidecar).is_file():
            with AmfReader(vision_sidecar) as vr:
                vis_metas = tuple(vr.entries)
        return MimoV2ConvertReport(
            backbone_amf=out_p,
            backbone_metas=tuple(metas),
            vision_amf=vision_sidecar if vis_metas else None,
            vision_metas=vis_metas,
            cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
            config_json_path=config_output,
        )

    resolved = resolve_checkpoint(in_p)
    base_dir = in_p if in_p.is_dir() else in_p.parent
    cfg_p, tok_p, tcfg_p, tpl_p = _resolve_companion_files(
        in_p,
        config_path=config_path or resolved.config_path,
        tokenizer_path=tokenizer_path,
        tokenizer_config_path=tokenizer_config_path,
        chat_template_path=chat_template_path,
    )
    raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p is not None else {}
    eff_cfg = _effective_mimo_cfg(raw_cfg)
    raw_tensors = _collect_tensors(resolved.shard_paths)

    eff_tp = infer_mimo_qkv_tp_degree(eff_cfg, raw_tensors, explicit_tp=qkv_tp_degree)
    n_layers = int(eff_cfg.get("num_hidden_layers") or 48)
    n_exp = int(eff_cfg.get("n_routed_experts") or 0)
    moe_freq = list(eff_cfg.get("moe_layer_freq") or [0 if i == 0 else 1 for i in range(n_layers)])
    cold_set = _select_mimo_cold_experts(
        n_layers=n_layers,
        n_experts=n_exp,
        moe_freq=moe_freq,
        tier_profile=tier_prof,
        cold_experts=cold_experts,
        cold_experts_per_layer=cold_experts_per_layer,
        heat_profile=heat_profile,
        all_experts_cold=all_experts_cold,
    )

    if tier_prof is not None and not tier_prof.allow_nvme_cold_experts and n_exp >= 384 and tier_prof.name == "128gb-max":
        raise HardwareTierError(
            f"target_tier={tier_prof.name!r} ({tier_prof.wired_budget_gib:.0f} GiB wired budget) "
            "cannot wire full MiMo-V2.6-Pro (384 experts); use '512gb-ultra' or enable cold experts"
        )

    audio_excludes = [k for k in raw_tensors if is_mimo_audio_tensor(k)]
    mtp_in_main = [k for k in raw_tensors if is_mimo_mtp_tensor(k)]
    excludes = [*audio_excludes, *mtp_in_main]

    eff_target_bits = target_bits
    if tier_prof is not None and quant_profile == "gsq-rco" and eff_target_bits is None and target_wired_gib is None:
        eff_target_bits = tier_prof.target_bits_default
    eff_fine_rungs = (tier_prof is not None and quant_profile == "gsq-rco") if fine_rungs is None else fine_rungs

    runtime_cfg = build_mimo_v2_config_json(eff_cfg, cold_experts_count=len(cold_set)) if eff_cfg else {}
    runtime_cfg_bytes = (
        json.dumps(runtime_cfg, ensure_ascii=False, indent=2).encode("utf-8") + b"\n" if runtime_cfg else None
    )
    if config_output is not None and runtime_cfg_bytes is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_bytes(runtime_cfg_bytes)

    stats: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_cfg_p: pathlib.Path | None = None
        if runtime_cfg_bytes is not None:
            tmp_cfg_p = pathlib.Path(tmp_d) / "config.json"
            tmp_cfg_p.write_bytes(runtime_cfg_bytes)
        metas = pack_amf_from_safetensors(
            in_p,
            out_p,
            model_id=eff_model_id,
            group_size=group_size,
            cold_experts=cold_set,
            all_experts_cold=all_experts_cold,
            dtype_mode=dtype_mode,
            tie_word_embeddings=bool(eff_cfg.get("tie_word_embeddings", False)),
            exclude_tensors=excludes,
            quant_profile=quant_profile,
            target_bits=eff_target_bits,
            use_rco=use_rco,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=strip_vision,
            include_vision=include_vision,
            vision_sidecar=vision_sidecar,
            vision_block_dtype=vision_block_dtype,
            embed_metadata=False,
            config_path=tmp_cfg_p,
            imatrix_path=imatrix_path,
            enforce_backbone_floor=enforce_backbone_floor,
            max_q2_share=max_q2_share,
            allow_lowbit_q2_retranscode=allow_lowbit_q2_retranscode,
            fine_rungs=eff_fine_rungs,
            stats_out=stats,
            mimo_config=eff_cfg,
            mimo_qkv_tp_degree=eff_tp,
        )

    if embed_metadata:
        tcfg_dict = json.loads(tcfg_p.read_text(encoding="utf-8")) if tcfg_p is not None else {}
        tpl_text = tpl_p.read_text(encoding="utf-8") if tpl_p is not None else None
        ct_bytes: bytes | None = None
        if tpl_text or (isinstance(tcfg_dict, dict) and tcfg_dict.get("chat_template")):
            spec = compile_chat_template_spec(
                jinja_text=tpl_text,
                tokenizer_config=tcfg_dict if isinstance(tcfg_dict, dict) else {},
                model_config=runtime_cfg or eff_cfg,
            )
            ct_bytes = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        metas = embed_metadata_into_amf(
            out_p,
            config_bytes=runtime_cfg_bytes,
            tokenizer_bytes=tok_p.read_bytes() if tok_p is not None else None,
            chat_template_bytes=ct_bytes,
            output_path=out_p,
        )

    vis_metas: tuple[PackedTensorMeta, ...] = ()
    if vision_sidecar is not None and pathlib.Path(vision_sidecar).is_file():
        from aminfer_tools.amf import AmfReader

        with AmfReader(vision_sidecar) as vr:
            vis_metas = tuple(vr.entries)

    mtp_metas_out: list[PackedTensorMeta] = []
    if mtp_output is not None:
        eff_mtp_in = mtp_input or (base_dir / "model_mtp.safetensors")
        if pathlib.Path(eff_mtp_in).is_file():
            mtp_metas_out = export_mimo_mtp_from_safetensors(
                pathlib.Path(eff_mtp_in),
                mtp_output,
                target_config=eff_cfg,
                model_id=f"{eff_model_id[:55]}-mtp",
                dtype_mode=dtype_mode,
                quant_profile=quant_profile,
                target_bits=eff_target_bits,
                use_rco=use_rco,
                qkv_tp_degree=eff_tp,
                embed_metadata=embed_metadata,
                config_output=mtp_config_output,
                workers=workers,
            )

    df_metas_out: list[PackedTensorMeta] = []
    if dflash_output is not None:
        eff_df_in = dflash_input or (base_dir / "dflash")
        if pathlib.Path(eff_df_in).exists():
            df_metas_out = export_mimo_dflash(
                pathlib.Path(eff_df_in),
                dflash_output,
                target_config=eff_cfg,
                block_size_override=dflash_block_size,
                model_id=f"{eff_model_id[:52]}-dflash",
                dtype_mode=dtype_mode,
                quant_profile=quant_profile,
                target_bits=eff_target_bits,
                use_rco=use_rco,
                embed_metadata=embed_metadata,
                config_output=dflash_config_output,
                workers=workers,
            )

    return MimoV2ConvertReport(
        backbone_amf=out_p,
        backbone_metas=tuple(metas),
        mtp_amf=mtp_output if mtp_metas_out else None,
        mtp_metas=tuple(mtp_metas_out),
        dflash_amf=dflash_output if df_metas_out else None,
        dflash_metas=tuple(df_metas_out),
        vision_amf=vision_sidecar if vis_metas else None,
        vision_metas=vis_metas,
        cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
        qkv_tp_degree=eff_tp,
        stripped_audio_tensors=len(audio_excludes),
        stripped_vision_tensors=stats.get("stripped_vision_tensors", 0),
        config_json_path=config_output,
    )


def convert_qwen35_checkpoint(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str | None = None,
    dtype_mode: str = "auto",
    group_size: int | None = None,
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    target_tier: str | None = None,
    use_rco: bool = True,
    manifest_path: pathlib.Path | None = None,
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
    config_output: pathlib.Path | None = None,
    config_path: pathlib.Path | None = None,
    tokenizer_path: pathlib.Path | None = None,
    tokenizer_config_path: pathlib.Path | None = None,
    chat_template_path: pathlib.Path | None = None,
    strict_vocab_check: bool = True,
    imatrix_path: pathlib.Path | None = None,
    embed_metadata: bool = True,
    workers: int = 1,
) -> Qwen35ConvertReport:
    """Convert `MiMo-V2.6-Distill-Qwen-9B` (`qwen3_5` BF16 / MLX 4-bit / GGUF) + ViT sidecar to `.amf`."""
    in_p = pathlib.Path(input_path)
    out_p = pathlib.Path(output_path)
    tier_prof = resolve_mimo_tier_profile(target_tier)
    eff_model_id = (model_id or (in_p.name if in_p.is_dir() else out_p.stem))[:63]

    if in_p.is_file() and in_p.suffix.lower() == ".gguf":
        from aminfer_tools.gguf import pack_amf_from_gguf

        metas = pack_amf_from_gguf(
            in_p,
            out_p,
            model_id=eff_model_id,
            dtype_mode=dtype_mode,
            embed_metadata=embed_metadata,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=tokenizer_config_path,
            chat_template_path=chat_template_path,
            vision_block_dtype=vision_block_dtype,
            group_size=group_size or 64,
            strict_vocab_check=strict_vocab_check,
        )
        return Qwen35ConvertReport(backbone_amf=out_p, backbone_metas=tuple(metas))

    if in_p.is_file() and in_p.suffix.lower() == ".amf":
        from aminfer_tools.amf import AmfReader, gsq_convert_with_vision, read_amf_metadata

        raw_cfg_b = pathlib.Path(config_path).read_bytes() if config_path is not None else read_amf_metadata(in_p, META_CONFIG_JSON)
        raw_cfg = json.loads(raw_cfg_b.decode("utf-8")) if raw_cfg_b else {}
        eff_cfg = _effective_mimo_cfg(raw_cfg)
        eff_target_bits = target_bits
        if tier_prof is not None and quant_profile == "gsq-rco" and eff_target_bits is None:
            eff_target_bits = 4.0 if tier_prof.name == "24gb-m4" else tier_prof.target_bits_default
        metas = gsq_convert_with_vision(
            in_p,
            out_p,
            target_bits=eff_target_bits,
            model_id=eff_model_id,
            group_size=group_size or 64,
            dtype_mode=dtype_mode,
            use_rco=use_rco,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=strip_vision,
            include_vision=include_vision,
            vision_sidecar=vision_sidecar,
            vision_block_dtype=vision_block_dtype,
            imatrix_path=imatrix_path,
            qwen35_config=eff_cfg or None,
        )
        vis_metas_a: tuple[PackedTensorMeta, ...] = ()
        if vision_sidecar is not None and pathlib.Path(vision_sidecar).is_file():
            with AmfReader(vision_sidecar) as vr:
                vis_metas_a = tuple(vr.entries)
        return Qwen35ConvertReport(
            backbone_amf=out_p,
            backbone_metas=tuple(metas),
            vision_amf=vision_sidecar if vis_metas_a else None,
            vision_metas=vis_metas_a,
            config_json_path=config_output,
        )

    resolved = resolve_checkpoint(in_p)
    cfg_p, tok_p, tcfg_p, tpl_p = _resolve_companion_files(
        in_p,
        config_path=config_path or resolved.config_path,
        tokenizer_path=tokenizer_path,
        tokenizer_config_path=tokenizer_config_path,
        chat_template_path=chat_template_path,
    )
    raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p is not None else {}
    eff_cfg = _effective_mimo_cfg(raw_cfg)
    runtime_cfg = build_qwen35_config_json(raw_cfg) if raw_cfg else {}
    runtime_cfg_bytes = (
        json.dumps(runtime_cfg, ensure_ascii=False, indent=2).encode("utf-8") + b"\n" if runtime_cfg else None
    )
    if config_output is not None and runtime_cfg_bytes is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_bytes(runtime_cfg_bytes)

    eff_target_bits = target_bits
    if tier_prof is not None and quant_profile == "gsq-rco" and eff_target_bits is None:
        eff_target_bits = 4.0 if tier_prof.name == "24gb-m4" else tier_prof.target_bits_default

    stats: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_cfg_p: pathlib.Path | None = None
        if runtime_cfg_bytes is not None:
            tmp_cfg_p = pathlib.Path(tmp_d) / "config.json"
            tmp_cfg_p.write_bytes(runtime_cfg_bytes)
        metas = pack_amf_from_safetensors(
            in_p,
            out_p,
            model_id=eff_model_id,
            group_size=group_size,
            dtype_mode=dtype_mode,
            tie_word_embeddings=bool(eff_cfg.get("tie_word_embeddings", False)),
            quant_profile=quant_profile,
            target_bits=eff_target_bits,
            use_rco=use_rco,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=strip_vision,
            include_vision=include_vision,
            vision_sidecar=vision_sidecar,
            vision_block_dtype=vision_block_dtype,
            embed_metadata=False,
            config_path=tmp_cfg_p,
            imatrix_path=imatrix_path,
            stats_out=stats,
            qwen35_config=eff_cfg,
        )

    if embed_metadata:
        tcfg_dict = json.loads(tcfg_p.read_text(encoding="utf-8")) if tcfg_p is not None else {}
        tpl_text = tpl_p.read_text(encoding="utf-8") if tpl_p is not None else None
        ct_bytes: bytes | None = None
        if tpl_text or (isinstance(tcfg_dict, dict) and tcfg_dict.get("chat_template")):
            spec = compile_chat_template_spec(
                jinja_text=tpl_text,
                tokenizer_config=tcfg_dict if isinstance(tcfg_dict, dict) else {},
                model_config=runtime_cfg or raw_cfg,
            )
            ct_bytes = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        metas = embed_metadata_into_amf(
            out_p,
            config_bytes=runtime_cfg_bytes,
            tokenizer_bytes=tok_p.read_bytes() if tok_p is not None else None,
            chat_template_bytes=ct_bytes,
            output_path=out_p,
        )

    vis_metas: tuple[PackedTensorMeta, ...] = ()
    if vision_sidecar is not None and pathlib.Path(vision_sidecar).is_file():
        from aminfer_tools.amf import AmfReader

        with AmfReader(vision_sidecar) as vr:
            vis_metas = tuple(vr.entries)

    return Qwen35ConvertReport(
        backbone_amf=out_p,
        backbone_metas=tuple(metas),
        vision_amf=vision_sidecar if vis_metas else None,
        vision_metas=vis_metas,
        stripped_vision_tensors=stats.get("stripped_vision_tensors", 0),
        config_json_path=config_output,
    )


# ---------------------------------------------------------------------------
# Deterministic Synthetic Bundle Generators (`synth-mimo` & `synth-qwen35`)
# ---------------------------------------------------------------------------


def _det_f32(rng: random.Random, n: int, *, std: float = 0.05, mean: float = 0.0) -> list[float]:
    return [mean + rng.uniform(-1.5, 1.5) * std for _ in range(n)]


def _encode_mxfp4_synthetic(vals: Sequence[float], rows: int, cols: int) -> bytes:
    """Quantize `[rows, cols]` floats into deterministic `GGUF_MXFP4` blocks (`[ue8m0, 16B split-half]`)."""
    if cols % 32 != 0:
        raise ValueError(f"MXFP4 cols must be multiple of 32, got {cols}")
    groups = cols // 32
    out = bytearray(rows * groups * 17)
    kvals = _KVALUES_MXFP4
    pos = 0
    for r in range(rows):
        row_off = r * cols
        for g in range(groups):
            blk = vals[row_off + g * 32 : row_off + (g + 1) * 32]
            amax = max(abs(x) for x in blk)
            if amax <= 0.0:
                eb = 0
                d = 0.0
            else:
                e_exp = max(-126, min(127, int(math.floor(math.log2(amax))) - 2))
                eb = e_exp + 127
                d = _UE8M0_LUT[eb] * 0.5
            out[pos] = eb
            codes = [0] * 32
            if d > 0.0:
                for i, v in enumerate(blk):
                    best_c = 0
                    best_err = abs(v)
                    for c_idx, kv in enumerate(kvals):
                        err = abs(v - kv * d)
                        if err < best_err:
                            best_err = err
                            best_c = c_idx
                    codes[i] = best_c
            for j in range(16):
                out[pos + 1 + j] = codes[j] | (codes[j + 16] << 4)
            pos += 17
    return bytes(out)


def _synth_tokenizer_bytes(vocab_size: int) -> bytes:
    vocab: dict[str, int] = {
        "<|endoftext|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|mask|>": 3,
    }
    for i in range(4, vocab_size):
        vocab[f"tok_{i}"] = i
    obj = {
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "special": True},
            {"id": 1, "content": "<|im_start|>", "special": True},
            {"id": 2, "content": "<|im_end|>", "special": True},
            {"id": 3, "content": "<|mask|>", "special": True},
        ],
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    }
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def _synth_chat_template_bytes(model_cfg: dict[str, Any]) -> bytes:
    jinja = (
        "{%- for message in messages -%}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' -}}{% endif -%}"
    )
    spec = compile_chat_template_spec(jinja_text=jinja, model_config=model_cfg)
    return json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class MimoV2SynthConfig:
    """Geometry preset for deterministic `synth-mimo` bundles (`mimo_v2` + `3L` MTP + `DFlash`)."""

    preset: str = "tiny-mimo-flash"
    model_id: str = "tiny-mimo-flash"
    num_hidden_layers: int = 4
    hidden_size: int = 256
    intermediate_size: int = 512
    moe_intermediate_size: int = 256
    vocab_size: int = 256
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    swa_num_attention_heads: int = 4
    swa_num_key_value_heads: int = 2
    head_dim: int = 192
    v_head_dim: int = 128
    sliding_window_size: int = 128
    partial_rotary_factor: float = 0.334
    attention_value_scale: float = 0.707
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    cold_experts_per_layer: int = 2
    mtp_layers: int = 3
    include_mtp: bool = True
    dflash_layers: int = 2
    dflash_block_size: int = 8
    include_dflash: bool = True
    expert_dtype: str = "mxfp4"
    dtype: str = "bf16"

    def validate(self) -> None:
        if self.num_hidden_layers < 2:
            raise ValueError("num_hidden_layers must be >= 2 (global L0 + >=1 SWA MoE)")
        if self.hidden_size % 64 != 0 or self.intermediate_size % 64 != 0 or self.moe_intermediate_size % 32 != 0:
            raise ValueError("hidden/intermediate dims must be multiples of 64 (32 for moe_inter)")
        if self.dflash_block_size not in (8, 16):
            raise ValueError(f"dflash_block_size must be 8 or 16, got {self.dflash_block_size}")
        if self.expert_dtype not in ("mxfp4", "bf16", "f16"):
            raise ValueError(f"expert_dtype must be mxfp4/bf16/f16, got {self.expert_dtype!r}")
        if self.dtype not in ("bf16", "f16"):
            raise ValueError(f"dtype must be bf16/f16, got {self.dtype!r}")


MIMO_SYNTH_PRESETS: dict[str, MimoV2SynthConfig] = {
    "tiny-mimo-flash": MimoV2SynthConfig(),
    "tiny-mimo-flash-b16": MimoV2SynthConfig(
        preset="tiny-mimo-flash-b16",
        model_id="tiny-mimo-flash-b16",
        dflash_block_size=16,
    ),
}


@dataclass(frozen=True)
class MimoV2SynthBundle:
    output_dir: pathlib.Path
    backbone_amf: pathlib.Path
    backbone_metas: tuple[PackedTensorMeta, ...]
    mtp_amf: pathlib.Path | None
    mtp_metas: tuple[PackedTensorMeta, ...]
    dflash_amf: pathlib.Path | None
    dflash_metas: tuple[PackedTensorMeta, ...]
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    total_bytes: int


def generate_synth_mimo_bundle(
    output_dir: pathlib.Path,
    *,
    config: MimoV2SynthConfig | None = None,
    seed: int = 42,
    embed_metadata: bool = True,
) -> MimoV2SynthBundle:
    """Synthesize deterministic 16 KiB-aligned `mimo_v2` + `3L` MTP + `DFlash` `.amf` bundle."""
    cfg = config or MimoV2SynthConfig()
    cfg.validate()
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    h = cfg.hidden_size
    inter = cfg.intermediate_size
    moe_inter = cfg.moe_intermediate_size
    vocab = cfg.vocab_size
    fdt = DType.BF16 if cfg.dtype == "bf16" else DType.F16

    def _fbytes(vals: Sequence[float]) -> bytes:
        return f32_values_to_bf16_bytes(vals) if fdt == DType.BF16 else f32_values_to_f16_bytes(vals)

    pat = [0 if i == 0 else 1 for i in range(cfg.num_hidden_layers)]
    moe_freq = [0 if i == 0 else 1 for i in range(cfg.num_hidden_layers)]
    src_cfg: dict[str, Any] = {
        "model_type": "mimo_v2",
        "hidden_size": h,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "num_hidden_layers": cfg.num_hidden_layers,
        "vocab_size": vocab,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "swa_num_attention_heads": cfg.swa_num_attention_heads,
        "swa_num_key_value_heads": cfg.swa_num_key_value_heads,
        "head_dim": cfg.head_dim,
        "v_head_dim": cfg.v_head_dim,
        "swa_head_dim": cfg.head_dim,
        "swa_v_head_dim": cfg.v_head_dim,
        "sliding_window": cfg.sliding_window_size,
        "hybrid_layer_pattern": pat,
        "moe_layer_freq": moe_freq,
        "n_routed_experts": cfg.n_routed_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "partial_rotary_factor": cfg.partial_rotary_factor,
        "attention_value_scale": cfg.attention_value_scale,
        "add_swa_attention_sink_bias": True,
        "add_full_attention_sink_bias": False,
        "rope_theta": 10000000.0,
        "swa_rope_theta": 10000.0,
        "num_nextn_predict_layers": cfg.mtp_layers if cfg.include_mtp else 0,
    }
    cold_tot = cfg.cold_experts_per_layer * sum(moe_freq)
    bb_cfg = build_mimo_v2_config_json(src_cfg, cold_experts_count=cold_tot)
    bb_cfg_bytes = json.dumps(bb_cfg, indent=2).encode("utf-8") + b"\n"
    tok_bytes = _synth_tokenizer_bytes(vocab)
    ct_bytes = _synth_chat_template_bytes(bb_cfg)
    (out_dir / "config.json").write_bytes(bb_cfg_bytes)
    (out_dir / "tokenizer.json").write_bytes(tok_bytes)

    bb_records: list[tuple] = []

    def _add_bb(name: str, kind: TensorKind, dt: DType, layer: int, expert: int, shape: tuple[int, ...], data: bytes, qb: int = 0, gs: int = 0) -> None:
        assert len(name.encode("utf-8")) <= 47, f"name too long: {name}"
        bb_records.append((name, kind, dt, qb, gs, layer, expert, shape, data, b""))

    _add_bb("embed_tokens.weight", TensorKind.EMBEDDING, fdt, NO_LAYER, NO_EXPERT, (vocab, h), _fbytes(_det_f32(rng, vocab * h)))
    _add_bb("norm.weight", TensorKind.NORM, DType.F32, NO_LAYER, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)))
    _add_bb("lm_head.weight", TensorKind.LM_HEAD, fdt, NO_LAYER, NO_EXPERT, (vocab, h), _fbytes(_det_f32(rng, vocab * h)))

    for l in range(cfg.num_hidden_layers):
        is_swa = pat[l] != 0
        qh = cfg.swa_num_attention_heads if is_swa else cfg.num_attention_heads
        kvh = cfg.swa_num_key_value_heads if is_swa else cfg.num_key_value_heads
        q_r = qh * cfg.head_dim
        k_r = kvh * cfg.head_dim
        v_r = kvh * cfg.v_head_dim
        _add_bb(f"layers.{l}.input_layernorm.weight", TensorKind.NORM, DType.F32, l, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add_bb(f"layers.{l}.post_attention_layernorm.weight", TensorKind.NORM, DType.F32, l, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add_bb(f"layers.{l}.self_attn.q_proj.weight", TensorKind.DENSE, fdt, l, NO_EXPERT, (q_r, h), _fbytes(_det_f32(rng, q_r * h)))
        _add_bb(f"layers.{l}.self_attn.k_proj.weight", TensorKind.DENSE, fdt, l, NO_EXPERT, (k_r, h), _fbytes(_det_f32(rng, k_r * h)))
        _add_bb(f"layers.{l}.self_attn.v_proj.weight", TensorKind.DENSE, fdt, l, NO_EXPERT, (v_r, h), _fbytes(_det_f32(rng, v_r * h)))
        _add_bb(f"layers.{l}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, l, NO_EXPERT, (h, qh * cfg.v_head_dim), _fbytes(_det_f32(rng, h * qh * cfg.v_head_dim)))
        if is_swa:
            _add_bb(f"layers.{l}.self_attn.sink_bias", TensorKind.NORM, fdt, l, NO_EXPERT, (qh,), _fbytes(_det_f32(rng, qh, std=0.1)))
        if moe_freq[l] == 0:
            _add_bb(f"layers.{l}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, l, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)))
            _add_bb(f"layers.{l}.mlp.up_proj.weight", TensorKind.DENSE, fdt, l, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)))
            _add_bb(f"layers.{l}.mlp.down_proj.weight", TensorKind.DENSE, fdt, l, NO_EXPERT, (h, inter), _fbytes(_det_f32(rng, h * inter)))
        else:
            ne = cfg.n_routed_experts
            _add_bb(f"layers.{l}.mlp.gate.weight", TensorKind.ROUTER, DType.F32, l, NO_EXPERT, (ne, h), f32_values_to_f32_bytes(_det_f32(rng, ne * h)))
            _add_bb(f"layers.{l}.mlp.gate.bias", TensorKind.ROUTER, DType.F32, l, NO_EXPERT, (ne,), f32_values_to_f32_bytes(_det_f32(rng, ne, std=0.05)))
            keep_hot = max(0, ne - cfg.cold_experts_per_layer)
            for e in range(ne):
                ek = TensorKind.COLD_EXPERT if e >= keep_hot else TensorKind.HOT_EXPERT
                for proj, pr, pc in (("gate_proj", moe_inter, h), ("up_proj", moe_inter, h), ("down_proj", h, moe_inter)):
                    vals = _det_f32(rng, pr * pc)
                    if cfg.expert_dtype == "mxfp4":
                        _add_bb(f"layers.{l}.mlp.experts.{e}.{proj}.weight", ek, DType.GGUF_MXFP4, l, e, (pr, pc), _encode_mxfp4_synthetic(vals, pr, pc), qb=4, gs=32)
                    else:
                        edt = DType.BF16 if cfg.expert_dtype == "bf16" else DType.F16
                        _add_bb(f"layers.{l}.mlp.experts.{e}.{proj}.weight", ek, edt, l, e, (pr, pc), f32_values_to_bf16_bytes(vals) if edt == DType.BF16 else f32_values_to_f16_bytes(vals))

    if embed_metadata:
        bb_records.extend([
            make_metadata_record(META_CONFIG_JSON, bb_cfg_bytes),
            make_metadata_record(META_TOKENIZER_JSON, tok_bytes),
            make_metadata_record(META_CHAT_TEMPLATE_JSON, ct_bytes),
        ])
    bb_amf = out_dir / "model.amf"
    bb_metas = write_amf(bb_amf, cfg.model_id, bb_records)
    total_bytes = bb_amf.stat().st_size

    mtp_amf: pathlib.Path | None = None
    mtp_metas: list[PackedTensorMeta] = []
    if cfg.include_mtp:
        mtp_cfg = build_mimo_mtp_config_dict(src_cfg, mtp_layers=cfg.mtp_layers)
        mtp_cfg_b = json.dumps(mtp_cfg, indent=2).encode("utf-8") + b"\n"
        (out_dir / "mtp_config.json").write_bytes(mtp_cfg_b)
        qh, kvh = cfg.swa_num_attention_heads, cfg.swa_num_key_value_heads
        q_r, k_r, v_r = qh * cfg.head_dim, kvh * cfg.head_dim, kvh * cfg.v_head_dim
        mtp_recs: list[tuple] = []
        for k in range(cfg.mtp_layers):
            mtp_recs.extend([
                (f"layers.{k}.enorm.weight", TensorKind.NORM, DType.F32, 0, 0, k, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
                (f"layers.{k}.hnorm.weight", TensorKind.NORM, DType.F32, 0, 0, k, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
                (f"layers.{k}.eh_proj.weight", TensorKind.DENSE, fdt, 0, 0, k, NO_EXPERT, (h, 2 * h), _fbytes(_det_f32(rng, h * 2 * h)), b""),
                (f"layers.{k}.input_layernorm.weight", TensorKind.NORM, DType.F32, 0, 0, k, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
                (f"layers.{k}.self_attn.q_proj.weight", TensorKind.DENSE, fdt, 0, 0, k, NO_EXPERT, (q_r, h), _fbytes(_det_f32(rng, q_r * h)), b""),
                (f"layers.{k}.self_attn.k_proj.weight", TensorKind.DENSE, fdt, 0, 0, k, NO_EXPERT, (k_r, h), _fbytes(_det_f32(rng, k_r * h)), b""),
                (f"layers.{k}.self_attn.v_proj.weight", TensorKind.DENSE, fdt, 0, 0, k, NO_EXPERT, (v_r, h), _fbytes(_det_f32(rng, v_r * h)), b""),
                (f"layers.{k}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, 0, 0, k, NO_EXPERT, (h, qh * cfg.v_head_dim), _fbytes(_det_f32(rng, h * qh * cfg.v_head_dim)), b""),
                (f"layers.{k}.self_attn.sink_bias", TensorKind.NORM, fdt, 0, 0, k, NO_EXPERT, (qh,), _fbytes(_det_f32(rng, qh, std=0.1)), b""),
                (f"layers.{k}.post_attention_layernorm.weight", TensorKind.NORM, DType.F32, 0, 0, k, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
                (f"layers.{k}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, 0, 0, k, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)), b""),
                (f"layers.{k}.mlp.up_proj.weight", TensorKind.DENSE, fdt, 0, 0, k, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)), b""),
                (f"layers.{k}.mlp.down_proj.weight", TensorKind.DENSE, fdt, 0, 0, k, NO_EXPERT, (h, inter), _fbytes(_det_f32(rng, h * inter)), b""),
                (f"layers.{k}.final_layernorm.weight", TensorKind.NORM, DType.F32, 0, 0, k, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
            ])
        if embed_metadata:
            mtp_recs.append(make_metadata_record(META_CONFIG_JSON, mtp_cfg_b))
        mtp_amf = out_dir / "mtp.amf"
        mtp_metas = write_amf(mtp_amf, f"{cfg.model_id[:55]}-mtp", mtp_recs)
        total_bytes += mtp_amf.stat().st_size

    df_amf: pathlib.Path | None = None
    df_metas: list[PackedTensorMeta] = []
    if cfg.include_dflash:
        taps = [0, cfg.num_hidden_layers - 1] if cfg.num_hidden_layers < 5 else [0, 1, 2, 3, 4]
        df_src = {
            "num_hidden_layers": cfg.dflash_layers,
            "hidden_size": h,
            "intermediate_size": inter,
            "num_attention_heads": cfg.swa_num_attention_heads,
            "num_key_value_heads": cfg.swa_num_key_value_heads,
            "head_dim": 128,
            "v_head_dim": 128,
            "vocab_size": vocab,
            "block_size": cfg.dflash_block_size,
            "target_layer_ids": taps,
            "sliding_window": 1024,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "attention_value_scale": 0.612,
            "mask_token_id": 3,
        }
        df_cfg = build_mimo_dflash_config_dict(df_src, target_cfg=src_cfg, block_size_override=cfg.dflash_block_size)
        df_cfg_b = json.dumps(df_cfg, indent=2).encode("utf-8") + b"\n"
        (out_dir / "dflash_config.json").write_bytes(df_cfg_b)
        df_hd = 128
        df_qh = cfg.swa_num_attention_heads
        df_kvh = cfg.swa_num_key_value_heads
        df_recs: list[tuple] = [
            ("fc.weight", TensorKind.DENSE, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h, len(taps) * h), _fbytes(_det_f32(rng, h * len(taps) * h)), b""),
            ("hidden_norm.weight", TensorKind.NORM, DType.F32, 0, 0, NO_LAYER, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
            ("norm.weight", TensorKind.NORM, DType.F32, 0, 0, NO_LAYER, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
            ("mask_embed.weight", TensorKind.EMBEDDING, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h)), b""),
            ("embed_tokens.weight", TensorKind.EMBEDDING, fdt, 0, 0, NO_LAYER, NO_EXPERT, (vocab, h), _fbytes(_det_f32(rng, vocab * h)), b""),
            ("lm_head.weight", TensorKind.LM_HEAD, fdt, 0, 0, NO_LAYER, NO_EXPERT, (vocab, h), _fbytes(_det_f32(rng, vocab * h)), b""),
        ]
        for l in range(cfg.dflash_layers):
            df_recs.extend([
                (f"layers.{l}.input_layernorm.weight", TensorKind.NORM, DType.F32, 0, 0, l, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
                (f"layers.{l}.post_attention_layernorm.weight", TensorKind.NORM, DType.F32, 0, 0, l, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)), b""),
                (f"layers.{l}.self_attn.q_proj.weight", TensorKind.DENSE, fdt, 0, 0, l, NO_EXPERT, (df_qh * df_hd, h), _fbytes(_det_f32(rng, df_qh * df_hd * h)), b""),
                (f"layers.{l}.self_attn.k_proj.weight", TensorKind.DENSE, fdt, 0, 0, l, NO_EXPERT, (df_kvh * df_hd, h), _fbytes(_det_f32(rng, df_kvh * df_hd * h)), b""),
                (f"layers.{l}.self_attn.v_proj.weight", TensorKind.DENSE, fdt, 0, 0, l, NO_EXPERT, (df_kvh * df_hd, h), _fbytes(_det_f32(rng, df_kvh * df_hd * h)), b""),
                (f"layers.{l}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, 0, 0, l, NO_EXPERT, (h, df_qh * df_hd), _fbytes(_det_f32(rng, h * df_qh * df_hd)), b""),
                (f"layers.{l}.self_attn.q_norm.weight", TensorKind.NORM, DType.F32, 0, 0, l, NO_EXPERT, (df_hd,), f32_values_to_f32_bytes(_det_f32(rng, df_hd, std=0.02, mean=1.0)), b""),
                (f"layers.{l}.self_attn.k_norm.weight", TensorKind.NORM, DType.F32, 0, 0, l, NO_EXPERT, (df_hd,), f32_values_to_f32_bytes(_det_f32(rng, df_hd, std=0.02, mean=1.0)), b""),
                (f"layers.{l}.self_attn.sink_bias", TensorKind.NORM, fdt, 0, 0, l, NO_EXPERT, (df_qh,), _fbytes(_det_f32(rng, df_qh, std=0.1)), b""),
                (f"layers.{l}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, 0, 0, l, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)), b""),
                (f"layers.{l}.mlp.up_proj.weight", TensorKind.DENSE, fdt, 0, 0, l, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)), b""),
                (f"layers.{l}.mlp.down_proj.weight", TensorKind.DENSE, fdt, 0, 0, l, NO_EXPERT, (h, inter), _fbytes(_det_f32(rng, h * inter)), b""),
            ])
        if embed_metadata:
            df_recs.append(make_metadata_record(META_CONFIG_JSON, df_cfg_b))
        df_amf = out_dir / "dflash.amf"
        df_metas = write_amf(df_amf, f"{cfg.model_id[:52]}-dflash", df_recs)
        total_bytes += df_amf.stat().st_size

    return MimoV2SynthBundle(
        output_dir=out_dir,
        backbone_amf=bb_amf,
        backbone_metas=tuple(bb_metas),
        mtp_amf=mtp_amf,
        mtp_metas=tuple(mtp_metas),
        dflash_amf=df_amf,
        dflash_metas=tuple(df_metas),
        config_path=out_dir / "config.json",
        tokenizer_path=out_dir / "tokenizer.json",
        total_bytes=total_bytes,
    )


@dataclass(frozen=True)
class Qwen35SynthConfig:
    """Geometry preset for deterministic `synth-qwen35` (`GDN + GQA + ViT`) bundle."""

    preset: str = "tiny-qwen35"
    model_id: str = "tiny-qwen35"
    num_hidden_layers: int = 4
    hidden_size: int = 256
    intermediate_size: int = 512
    vocab_size: int = 256
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 128
    linear_num_key_heads: int = 4
    linear_num_value_heads: int = 8
    linear_key_head_dim: int = 64
    linear_value_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    vision_layers: int = 2
    vision_hidden_size: int = 128
    vision_intermediate_size: int = 256
    vision_mode: str = "sidecar"
    dtype: str = "bf16"

    def validate(self) -> None:
        if self.num_hidden_layers < 2:
            raise ValueError("num_hidden_layers must be >= 2")
        if self.hidden_size % 64 != 0 or self.intermediate_size % 64 != 0:
            raise ValueError("hidden_size and intermediate_size must be multiples of 64")
        if self.vision_mode not in ("sidecar", "inline", "strip"):
            raise ValueError(f"vision_mode must be sidecar/inline/strip, got {self.vision_mode!r}")
        if self.dtype not in ("bf16", "f16"):
            raise ValueError(f"dtype must be bf16/f16, got {self.dtype!r}")


QWEN35_SYNTH_PRESETS: dict[str, Qwen35SynthConfig] = {
    "tiny-qwen35": Qwen35SynthConfig(),
}


@dataclass(frozen=True)
class Qwen35SynthBundle:
    output_dir: pathlib.Path
    backbone_amf: pathlib.Path
    backbone_metas: tuple[PackedTensorMeta, ...]
    vision_amf: pathlib.Path | None
    vision_metas: tuple[PackedTensorMeta, ...]
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    total_bytes: int


def generate_synth_qwen35_bundle(
    output_dir: pathlib.Path,
    *,
    config: Qwen35SynthConfig | None = None,
    seed: int = 42,
    embed_metadata: bool = True,
) -> Qwen35SynthBundle:
    """Synthesize deterministic 16 KiB-aligned `qwen3_5` (`GDN + GQA + ViT`) `.amf` bundle (`#502`)."""
    cfg = config or Qwen35SynthConfig()
    cfg.validate()
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    h = cfg.hidden_size
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    fdt = DType.BF16 if cfg.dtype == "bf16" else DType.F16

    def _fbytes(vals: Sequence[float]) -> bytes:
        return f32_values_to_bf16_bytes(vals) if fdt == DType.BF16 else f32_values_to_f16_bytes(vals)

    ltypes = [
        "full_attention" if i == cfg.num_hidden_layers - 1 or (i + 1) % 4 == 0 else "linear_attention"
        for i in range(cfg.num_hidden_layers)
    ]
    raw_cfg: dict[str, Any] = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": h,
            "intermediate_size": inter,
            "num_hidden_layers": cfg.num_hidden_layers,
            "vocab_size": vocab,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "attn_output_gate": True,
            "layer_types": ltypes,
            "full_attention_interval": 4,
            "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_num_value_heads": cfg.linear_num_value_heads,
            "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim,
            "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
        },
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": cfg.vision_layers,
            "hidden_size": cfg.vision_hidden_size,
            "intermediate_size": cfg.vision_intermediate_size,
            "num_heads": 4,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": h,
        },
    }
    rt_cfg = build_qwen35_config_json(raw_cfg)
    rt_cfg_b = json.dumps(rt_cfg, indent=2).encode("utf-8") + b"\n"
    tok_b = _synth_tokenizer_bytes(vocab)
    ct_b = _synth_chat_template_bytes(rt_cfg)
    (out_dir / "config.json").write_bytes(rt_cfg_b)
    (out_dir / "tokenizer.json").write_bytes(tok_b)

    records: list[tuple] = []

    def _add(name: str, kind: TensorKind, dt: DType, layer: int, shape: tuple[int, ...], data: bytes) -> None:
        assert len(name.encode("utf-8")) <= 47, f"name too long: {name}"
        records.append((name, kind, dt, 0, 0, layer, NO_EXPERT, shape, data, b""))

    _add("embed_tokens.weight", TensorKind.EMBEDDING, fdt, NO_LAYER, (vocab, h), _fbytes(_det_f32(rng, vocab * h)))
    _add("norm.weight", TensorKind.NORM, DType.F32, NO_LAYER, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=0.0)))
    _add("lm_head.weight", TensorKind.LM_HEAD, fdt, NO_LAYER, (vocab, h), _fbytes(_det_f32(rng, vocab * h)))

    kq_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    lv_h = cfg.linear_num_value_heads
    l_vd = cfg.linear_value_head_dim
    z_dim = lv_h * l_vd
    conv_dim = 2 * kq_dim + z_dim
    l_kern = cfg.linear_conv_kernel_dim
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim

    for l in range(cfg.num_hidden_layers):
        _add(f"layers.{l}.input_layernorm.weight", TensorKind.NORM, DType.F32, l, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=0.0)))
        _add(f"layers.{l}.post_attention_layernorm.weight", TensorKind.NORM, DType.F32, l, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=0.0)))
        _add(f"layers.{l}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, l, (inter, h), _fbytes(_det_f32(rng, inter * h)))
        _add(f"layers.{l}.mlp.up_proj.weight", TensorKind.DENSE, fdt, l, (inter, h), _fbytes(_det_f32(rng, inter * h)))
        _add(f"layers.{l}.mlp.down_proj.weight", TensorKind.DENSE, fdt, l, (h, inter), _fbytes(_det_f32(rng, h * inter)))
        if ltypes[l] == "linear_attention":
            _add(f"layers.{l}.linear_attn.q_proj.weight", TensorKind.DENSE, fdt, l, (kq_dim, h), _fbytes(_det_f32(rng, kq_dim * h)))
            _add(f"layers.{l}.linear_attn.k_proj.weight", TensorKind.DENSE, fdt, l, (kq_dim, h), _fbytes(_det_f32(rng, kq_dim * h)))
            _add(f"layers.{l}.linear_attn.v_proj.weight", TensorKind.DENSE, fdt, l, (z_dim, h), _fbytes(_det_f32(rng, z_dim * h)))
            _add(f"layers.{l}.linear_attn.g_proj.weight", TensorKind.DENSE, fdt, l, (z_dim, h), _fbytes(_det_f32(rng, z_dim * h)))
            _add(f"layers.{l}.linear_attn.b_proj.weight", TensorKind.DENSE, fdt, l, (lv_h, h), _fbytes(_det_f32(rng, lv_h * h)))
            _add(f"layers.{l}.linear_attn.a_proj.weight", TensorKind.DENSE, fdt, l, (lv_h, h), _fbytes(_det_f32(rng, lv_h * h)))
            _add(f"layers.{l}.linear_attn.conv1d.weight", TensorKind.DENSE, fdt, l, (conv_dim, l_kern), _fbytes(_det_f32(rng, conv_dim * l_kern)))
            _add(f"layers.{l}.linear_attn.A_log", TensorKind.NORM, DType.F32, l, (lv_h,), f32_values_to_f32_bytes(_det_f32(rng, lv_h, std=0.1, mean=-2.0)))
            _add(f"layers.{l}.linear_attn.dt_bias", TensorKind.NORM, DType.F32, l, (lv_h,), f32_values_to_f32_bytes(_det_f32(rng, lv_h, std=0.05)))
            _add(f"layers.{l}.linear_attn.norm.weight", TensorKind.NORM, DType.F32, l, (l_vd,), f32_values_to_f32_bytes(_det_f32(rng, l_vd, std=0.02, mean=1.0)))
            _add(f"layers.{l}.linear_attn.o_proj.weight", TensorKind.DENSE, fdt, l, (h, z_dim), _fbytes(_det_f32(rng, h * z_dim)))
        else:
            _add(f"layers.{l}.self_attn.q_proj.weight", TensorKind.DENSE, fdt, l, (q_dim, h), _fbytes(_det_f32(rng, q_dim * h)))
            _add(f"layers.{l}.self_attn.q_gate_proj.weight", TensorKind.DENSE, fdt, l, (q_dim, h), _fbytes(_det_f32(rng, q_dim * h)))
            _add(f"layers.{l}.self_attn.k_proj.weight", TensorKind.DENSE, fdt, l, (kv_dim, h), _fbytes(_det_f32(rng, kv_dim * h)))
            _add(f"layers.{l}.self_attn.v_proj.weight", TensorKind.DENSE, fdt, l, (kv_dim, h), _fbytes(_det_f32(rng, kv_dim * h)))
            _add(f"layers.{l}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, l, (h, q_dim), _fbytes(_det_f32(rng, h * q_dim)))
            _add(f"layers.{l}.self_attn.q_norm.weight", TensorKind.NORM, DType.F32, l, (cfg.head_dim,), f32_values_to_f32_bytes(_det_f32(rng, cfg.head_dim, std=0.02, mean=0.0)))
            _add(f"layers.{l}.self_attn.k_norm.weight", TensorKind.NORM, DType.F32, l, (cfg.head_dim,), f32_values_to_f32_bytes(_det_f32(rng, cfg.head_dim, std=0.02, mean=0.0)))

    vh = cfg.vision_hidden_size
    vi = cfg.vision_intermediate_size
    vis_recs: list[tuple] = []
    if cfg.vision_mode != "strip" and cfg.vision_layers > 0:
        patch_k = 3 * 2 * 16 * 16
        vis_recs.extend([
            ("visual.patch_embed.proj.weight", TensorKind.DENSE, fdt, 0, 0, NO_LAYER, NO_EXPERT, (vh, patch_k), _fbytes(_det_f32(rng, vh * patch_k)), b""),
            ("visual.pos_embed.weight", TensorKind.DENSE, fdt, 0, 0, NO_LAYER, NO_EXPERT, (64, vh), _fbytes(_det_f32(rng, 64 * vh)), b""),
            ("visual.merger.norm.weight", TensorKind.NORM, DType.F32, 0, 0, NO_LAYER, NO_EXPERT, (vh,), f32_values_to_f32_bytes(_det_f32(rng, vh, std=0.02, mean=1.0)), b""),
            ("visual.merger.linear_fc1.weight", TensorKind.DENSE, fdt, 0, 0, NO_LAYER, NO_EXPERT, (vh * 4, vh * 4), _fbytes(_det_f32(rng, vh * 4 * vh * 4)), b""),
            ("visual.merger.linear_fc2.weight", TensorKind.DENSE, fdt, 0, 0, NO_LAYER, NO_EXPERT, (h, vh * 4), _fbytes(_det_f32(rng, h * vh * 4)), b""),
        ])
        for vl in range(cfg.vision_layers):
            vis_recs.extend([
                (f"visual.blocks.{vl}.norm1.weight", TensorKind.NORM, DType.F32, 0, 0, vl, NO_EXPERT, (vh,), f32_values_to_f32_bytes(_det_f32(rng, vh, std=0.02, mean=1.0)), b""),
                (f"visual.blocks.{vl}.norm2.weight", TensorKind.NORM, DType.F32, 0, 0, vl, NO_EXPERT, (vh,), f32_values_to_f32_bytes(_det_f32(rng, vh, std=0.02, mean=1.0)), b""),
                (f"visual.blocks.{vl}.attn.qkv.weight", TensorKind.DENSE, fdt, 0, 0, vl, NO_EXPERT, (3 * vh, vh), _fbytes(_det_f32(rng, 3 * vh * vh)), b""),
                (f"visual.blocks.{vl}.attn.proj.weight", TensorKind.DENSE, fdt, 0, 0, vl, NO_EXPERT, (vh, vh), _fbytes(_det_f32(rng, vh * vh)), b""),
                (f"visual.blocks.{vl}.mlp.linear_fc1.weight", TensorKind.DENSE, fdt, 0, 0, vl, NO_EXPERT, (vi, vh), _fbytes(_det_f32(rng, vi * vh)), b""),
                (f"visual.blocks.{vl}.mlp.linear_fc2.weight", TensorKind.DENSE, fdt, 0, 0, vl, NO_EXPERT, (vh, vi), _fbytes(_det_f32(rng, vh * vi)), b""),
            ])
    if cfg.vision_mode == "inline":
        records.extend(vis_recs)

    if embed_metadata:
        records.extend([
            make_metadata_record(META_CONFIG_JSON, rt_cfg_b),
            make_metadata_record(META_TOKENIZER_JSON, tok_b),
            make_metadata_record(META_CHAT_TEMPLATE_JSON, ct_b),
        ])
    bb_amf = out_dir / "model.amf"
    bb_metas = write_amf(bb_amf, cfg.model_id, records)
    total_bytes = bb_amf.stat().st_size

    vis_amf: pathlib.Path | None = None
    vis_metas: list[PackedTensorMeta] = []
    if cfg.vision_mode == "sidecar" and vis_recs:
        if embed_metadata:
            vis_recs.append(make_metadata_record(META_CONFIG_JSON, rt_cfg_b))
        vis_amf = out_dir / "vision.amf"
        vis_metas = write_amf(vis_amf, f"{cfg.model_id[:55]}-vision", vis_recs)
        total_bytes += vis_amf.stat().st_size

    return Qwen35SynthBundle(
        output_dir=out_dir,
        backbone_amf=bb_amf,
        backbone_metas=tuple(bb_metas),
        vision_amf=vis_amf,
        vision_metas=tuple(vis_metas),
        config_path=out_dir / "config.json",
        tokenizer_path=out_dir / "tokenizer.json",
        total_bytes=total_bytes,
    )
