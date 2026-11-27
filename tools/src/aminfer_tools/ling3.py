"""W28/W27 Ling-3.1 (`bailing_hybrid`) validator, converter, `RCO-Q3` dense-`KDA`, and `synth-ling3` (`M9.3`).

Locked Canonical `.amf` v0 Tensor Naming & `config.json` Contract (`#534`, `#535`, M9.3)
========================================================================================
Every tensor name stored in `.amf` is strictly `<= 47` UTF-8 bytes (`TensorEntry.name` is `[48]u8`
NUL-terminated; `long_backbone_names == 0`; verified max across all 214,025 Ling-3.1-flash-1019
tensors is `45` UTF-8 bytes).

1. `inclusionAI/Ling-3.1-flash-1019` (`bailing_hybrid`, `572.12B` total)
------------------------------------------------------------------------
- `config.json`: `"model_type": "bailing_hybrid"`, `hidden_size=4096`, `intermediate_size=10240`,
  `moe_intermediate_size=1280`, `moe_shared_expert_intermediate_size=1280`, `num_hidden_layers=72`,
  `layer_group_size=8` (`63` `linear_attention` `KDA` layers `L%8!=7` + `9` full-causal `mla_attention`
  Gated `MLA` layers `L in {7,15,23,31,39,47,55,63,71}`), `first_k_dense_replace=4` (`L0..L3` dense,
  `L4..L71` MoE `512E` `top-8` + `1` shared, `n_group=8`, `topk_group=4` top-2 sum per group,
  `scoring_func="sigmoid"`, `norm_topk_prob=true`, `routed_scaling_factor=2.5`),
  `no_kda_lora=true`, `kda_safe_gate=true`, `kda_lower_bound=-5.0`, `linear_silu=true`,
  `short_conv_kernel_size=4`, `num_attention_heads=64`, `head_dim=128`, `q_lora_rank=1536`,
  `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `qk_head_dim=192`,
  `v_head_dim=128`, `gated_attention_proj_granularity_type="head_wise"`, `rope_interleave=true`,
  `rope_theta=6000000`, `rms_norm_eps=1e-6`, `num_nextn_predict_layers=1`, `mtp_use_kda=false`,
  `expert_swiglu_limit_list` & `share_expert_swiglu_limit_list` (`[0]*65 + [10]*8`).
- `.amf` tensors (`<= 45 B`):
  - `embed_tokens.weight`, `norm.weight`, `lm_head.weight`
  - `layers.{L}.input_layernorm.weight`, `layers.{L}.post_attention_layernorm.weight`
  - Dense `L0..L3`: `layers.{L}.mlp.{gate_proj,up_proj,down_proj}.weight`
  - MoE `L4..L71`: `layers.{L}.mlp.gate.weight`, `layers.{L}.mlp.gate.bias` (`F32`, from `expert_bias`),
    `layers.{L}.mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`,
    `layers.{L}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`
  - `63L` `KDA` (`L%8!=7`): `layers.{L}.linear_attn.{q,k,v,f,g,o,b}_proj.weight`,
    `layers.{L}.linear_attn.{q,k,v}_conv1d.weight` (`F32 [8192, 4]`),
    `layers.{L}.linear_attn.{A_log,dt_bias}` (`F32`),
    `layers.{L}.linear_attn.o_norm.weight` (`BF16 [128]`, shared across heads)
  - `9L` Gated `MLA` (`L%8==7`):
    `layers.{L}.self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,g_proj,o_proj}.weight`
    (upstream HF `attention.dense.weight` -> `self_attn.o_proj.weight`)
- Standalone `1L` `MTP` (`mtp.amf`):
  - Re-indexes `layers.72.*` / `mtp.0.*` -> `layers.0.*` (preserving optional root aliases
    `{enorm,hnorm,eh_proj,norm}.weight` when supplied), maps `layers.0.final_layernorm.weight`,
    uses Gated `MLA` (`mla_attention`) + `512E` MoE, and supports optional `--repack-sme2` on `eh_proj`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import pathlib
import random
import re
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
    TensorLayout,
    QuantParamKind,
    _collect_tensors,
    embed_metadata_into_amf,
    f32_to_bf16_bytes as f32_values_to_bf16_bytes,
    f32_values_to_f16_bytes,
    make_metadata_record,
    pack_amf_from_safetensors,
    repack_amf_sme2,
    resolve_checkpoint,
    write_amf,
)
from aminfer_tools.chat_template import compile_chat_template_spec
from aminfer_tools.deepseek41 import HardwareTierError, load_cold_experts_from_heat_profile
from aminfer_tools.mimo_v2 import (
    _det_f32,
    _encode_mxfp4_synthetic,
    _read_header_summary_tensors,
    _read_weight_map_keys,
    _resolve_companion_files,
    _synth_chat_template_bytes,
    _synth_tokenizer_bytes,
    f32_values_to_f32_bytes,
)


# ---------------------------------------------------------------------------
# Tier Profiles (`24gb-m4-rco-q3`, `24gb-m4`, `128gb-max`, `256gb-studio`, `512gb-ultra`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ling3TierProfile:
    """Hardware-tier wired RAM, `RCO-Q3` dense-`KDA`, & cold-expert placement budget."""

    name: str
    host_memory_gib: float
    wired_budget_gib: float
    always_active_reserve_gib: float
    hot_expert_budget_gib: float
    target_bits_default: float
    dense_kda_rco_q3: bool
    allow_nvme_cold_experts: bool
    recommended_model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_M4_24GB_PROFILE = Ling3TierProfile(
    name="24gb-m4-rco-q3",
    host_memory_gib=24.0,
    wired_budget_gib=18.5,
    always_active_reserve_gib=9.2,
    hot_expert_budget_gib=7.1,
    target_bits_default=3.25,
    dense_kda_rco_q3=True,
    allow_nvme_cold_experts=True,
    recommended_model="Ling-3.1-flash-1019 (RCO-Q3 dense-KDA ~9.2 GiB + ~7.1 GiB hot-expert cache + NVMe cold)",
)

LING3_TIER_PROFILES: dict[str, Ling3TierProfile] = {
    "24gb-m4-rco-q3": _M4_24GB_PROFILE,
    "24gb-m4": Ling3TierProfile(
        name="24gb-m4",
        host_memory_gib=_M4_24GB_PROFILE.host_memory_gib,
        wired_budget_gib=_M4_24GB_PROFILE.wired_budget_gib,
        always_active_reserve_gib=_M4_24GB_PROFILE.always_active_reserve_gib,
        hot_expert_budget_gib=_M4_24GB_PROFILE.hot_expert_budget_gib,
        target_bits_default=_M4_24GB_PROFILE.target_bits_default,
        dense_kda_rco_q3=True,
        allow_nvme_cold_experts=True,
        recommended_model=_M4_24GB_PROFILE.recommended_model,
    ),
    "128gb-max": Ling3TierProfile(
        name="128gb-max",
        host_memory_gib=128.0,
        wired_budget_gib=104.0,
        always_active_reserve_gib=12.8,
        hot_expert_budget_gib=91.2,
        target_bits_default=3.50,
        dense_kda_rco_q3=True,
        allow_nvme_cold_experts=True,
        recommended_model="Ling-3.1-flash-1019 (RCO-Q3 dense-KDA + ~91 GiB hot-expert cache + partial NVMe cold)",
    ),
    "256gb-studio": Ling3TierProfile(
        name="256gb-studio",
        host_memory_gib=256.0,
        wired_budget_gib=210.0,
        always_active_reserve_gib=13.0,
        hot_expert_budget_gib=172.0,
        target_bits_default=2.55,
        dense_kda_rco_q3=False,
        allow_nvme_cold_experts=False,
        recommended_model="Ling-3.1-flash-1019 (~172 GiB full-wired GSQ-RCO)",
    ),
    "512gb-ultra": Ling3TierProfile(
        name="512gb-ultra",
        host_memory_gib=512.0,
        wired_budget_gib=420.0,
        always_active_reserve_gib=18.0,
        hot_expert_budget_gib=320.0,
        target_bits_default=4.0,
        dense_kda_rco_q3=False,
        allow_nvme_cold_experts=False,
        recommended_model="Ling-3.1-flash-1019 (Q4 / FP8 full-wired)",
    ),
}


def resolve_ling3_tier_profile(tier: str | None) -> Ling3TierProfile | None:
    if tier is None:
        return None
    key = tier.strip().lower()
    if key not in LING3_TIER_PROFILES:
        raise ValueError(f"unknown target_tier {tier!r}; expected one of {sorted(LING3_TIER_PROFILES)}")
    return LING3_TIER_PROFILES[key]


# ---------------------------------------------------------------------------
# Config Detection, Schedules & Canonical `<= 47 B` Tensor Naming
# ---------------------------------------------------------------------------

_STRIP_QUANT_SUFFIXES = (
    ".weight_scale_inv",
    ".weight_scale_2",
    ".weight_scale",
    ".weight_global_scale",
    ".input_global_scale",
    ".input_scale",
    "_weight_scale_inv",
    "_weight_scale",
    ".scales",
    ".biases",
    ".global_scale",
)
_LAYER_IDX_RE = re.compile(r"^layers\.(\d+)\.")
_MTP_KEY_RE = re.compile(r"^(?:model\.)?(?:language_model\.)?(?:mtp\.\d+\.|mtp\.|nextn\.)")
_LING3_PREFIXES = (
    "model.language_model.",
    "model.llm.",
    "language_model.model.",
    "language_model.",
    "model.",
)
_KDA_LEAF_HINTS = (
    ".attention.A_log",
    ".attention.dt_bias",
    ".attention.b_proj.",
    ".attention.f_proj.",
    ".attention.o_norm.",
    ".attention.o_proj.",
    ".attention.q_conv1d.",
    ".attention.k_conv1d.",
    ".attention.v_conv1d.",
    ".linear_attn.",
)


def _effective_ling3_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        return {}
    tc = cfg.get("text_config")
    if isinstance(tc, Mapping):
        merged = dict(tc)
        for k, v in cfg.items():
            if k not in ("text_config", "vision_config", "audio_config") and k not in merged:
                merged[k] = v
        return merged
    return dict(cfg)


def is_ling3_config(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return False
    mt = str(cfg.get("model_type") or "").lower()
    if mt in ("bailing_hybrid", "bailing_hybrid_mtp", "ling3", "bailingmoe3"):
        return True
    archs = [str(a).lower() for a in (cfg.get("architectures") or [])]
    if any("bailingmoev3" in a for a in archs):
        return True
    tc = cfg.get("text_config")
    return isinstance(tc, Mapping) and str(tc.get("model_type") or "").lower() in ("bailing_hybrid", "bailingmoe3")


def resolve_ling3_layer_types(cfg: Mapping[str, Any]) -> list[str]:
    eff = _effective_ling3_cfg(cfg)
    n_layers = int(eff.get("num_hidden_layers") or 0)
    raw = eff.get("layer_types")
    if isinstance(raw, list) and len(raw) >= n_layers and n_layers > 0:
        out: list[str] = []
        for x in raw[:n_layers]:
            sx = str(x).lower()
            if sx in ("attention", "full_attention", "mla_attention"):
                out.append("mla_attention")
            else:
                out.append("linear_attention")
        return out
    lgs = max(int(eff.get("layer_group_size") or 8), 1)
    mla_floor = (n_layers // lgs) * lgs
    return [
        "mla_attention" if (((i + 1) % lgs == 0) or i >= mla_floor) else "linear_attention"
        for i in range(n_layers)
    ]


def resolve_ling3_mlp_layer_types(cfg: Mapping[str, Any]) -> list[str]:
    eff = _effective_ling3_cfg(cfg)
    n_layers = int(eff.get("num_hidden_layers") or 0)
    raw = eff.get("mlp_layer_types")
    if isinstance(raw, list) and len(raw) >= n_layers and n_layers > 0:
        return [str(x).lower() for x in raw[:n_layers]]
    first_k = int(eff.get("first_k_dense_replace") or 0)
    n_dense = min(max(first_k, 0), n_layers)
    return ["dense"] * n_dense + ["sparse"] * (n_layers - n_dense)


def is_ling3_mtp_tensor(raw_name: str, backbone_num_layers: int | None = None) -> bool:
    if _MTP_KEY_RE.match(raw_name) is not None:
        return True
    if raw_name in ("enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"):
        return True
    if backbone_num_layers is not None and backbone_num_layers > 0:
        s = raw_name
        for pref in _LING3_PREFIXES:
            if s.startswith(pref):
                s = s[len(pref) :]
                break
        if s.startswith(f"layers.{backbone_num_layers}."):
            return True
    return False


def _strip_ling3_quant_suffix(raw_name: str) -> str:
    if raw_name.endswith(".weight_packed"):
        return raw_name[: -len(".weight_packed")] + ".weight"
    for suf in _STRIP_QUANT_SUFFIXES:
        if raw_name.endswith(suf):
            return raw_name[: -len(suf)]
    return raw_name


def normalize_ling3_tensor_name(
    raw_name: str,
    *,
    layer_types: Sequence[str] | None = None,
    mtp_source_layer: int | None = None,
    mtp_standalone: bool = False,
) -> str:
    """Normalize a `bailing_hybrid` / `1L` MTP tensor name to canonical `<= 47 B` `.amf` form."""
    name = _strip_ling3_quant_suffix(raw_name)
    for pref in _LING3_PREFIXES:
        if name.startswith(pref):
            name = name[len(pref) :]
            break

    if name == "word_embeddings.weight":
        name = "embed_tokens.weight"
    elif name == "word_embeddings":
        name = "embed_tokens"

    if mtp_standalone:
        if name in ("enorm.weight", "hnorm.weight", "eh_proj.weight", "norm.weight", "shared_head.norm.weight"):
            enc_root = name.encode("utf-8")
            if len(enc_root) > 47:
                raise ValueError(f"AMF tensor name exceeds 47 B: {name!r}")
            return name
        if mtp_source_layer is not None and name.startswith(f"layers.{mtp_source_layer}."):
            name = "layers.0." + name[len(f"layers.{mtp_source_layer}.") :]
        elif re.match(r"^mtp\.\d+\.", name):
            name = "layers.0." + re.sub(r"^mtp\.\d+\.", "", name)
        elif name.startswith("mtp.layers."):
            name = name[len("mtp.") :]
        elif name.startswith("mtp."):
            name = "layers.0." + name[len("mtp.") :]
        if name == "layers.0.shared_head.norm.weight":
            name = "shared_head.norm.weight"

    name = name.replace(".mlp.gate.e_score_correction_bias", ".mlp.gate.bias")
    name = name.replace(".mlp.gate.expert_bias", ".mlp.gate.bias")
    name = name.replace(".mlp.gate.gate_proj.", ".mlp.gate.")
    if name.endswith(".mlp.gate.gate_proj"):
        name = name[: -len(".gate_proj")] + ".weight"

    if ".attention." in name:
        m_l = _LAYER_IDX_RE.match(name)
        is_linear = False
        if mtp_standalone:
            is_linear = False
        elif m_l is not None and layer_types is not None:
            li = int(m_l.group(1))
            if 0 <= li < len(layer_types):
                is_linear = layer_types[li] == "linear_attention"
        if not is_linear and any(tok in name for tok in _KDA_LEAF_HINTS):
            is_linear = True
        if is_linear:
            name = name.replace(".attention.", ".linear_attn.")
        else:
            if name.endswith(".attention.dense"):
                name = name[: -len(".attention.dense")] + ".self_attn.o_proj"
            else:
                name = name.replace(".attention.dense.", ".self_attn.o_proj.")
            name = name.replace(".attention.", ".self_attn.")
    elif name.endswith(".self_attn.dense"):
        name = name[: -len(".self_attn.dense")] + ".self_attn.o_proj"
    elif ".self_attn.dense." in name:
        name = name.replace(".self_attn.dense.", ".self_attn.o_proj.")

    enc = name.encode("utf-8")
    if len(enc) == 0 or len(enc) > 47:
        raise ValueError(f"AMF tensor name must be 1..47 UTF-8 bytes, got {len(enc)}: {name!r}")
    return name


# ---------------------------------------------------------------------------
# `W28` Metadata Dry-Run Validator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ling3MetadataReport:
    metadata_dir: pathlib.Path
    model_type: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    layer_group_size: int
    linear_attention_layers: int
    mla_layers: int
    dense_layers: int
    moe_layers: int
    no_kda_lora: bool
    kda_safe_gate: bool
    kda_lower_bound: float
    linear_silu: bool
    short_conv_kernel_size: int
    mla_gated_proj: bool
    gated_attention_granularity: str
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    mtp_layers: int
    always_active_backbone_params: int
    kda_dense_proj_params: int
    routed_expert_params: int
    routed_per_tok_params: int
    mtp_total_params: int
    always_active_plus_mtp_params: int
    total_params: int
    q4_always_active_bb_gib: float
    rco_q3_always_active_bb_gib: float
    q4_always_active_plus_mtp_gib: float
    rco_q3_always_active_plus_mtp_gib: float
    m4_24gb_hot_cache_q4_gib: float
    m4_24gb_hot_cache_rco_q3_gib: float
    mla_kv_bytes_per_token_fp16: int
    kda_state_bytes_fp32: int
    backbone_tensor_count: int
    mtp_tensor_count: int
    routed_expert_ids_detected: int
    long_backbone_names: int
    long_mtp_names: int
    max_backbone_name_bytes: int
    header_summary_shards: int
    tier_admission: dict[str, dict[str, Any]]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metadata_dir"] = str(self.metadata_dir)
        d["errors"] = list(self.errors)
        return d


def _matches_linear_shape(info: Mapping[str, Any], out_rows: int, in_cols: int) -> bool:
    shp = info.get("shape")
    if not isinstance(shp, list) or len(shp) != 2:
        return False
    if shp == [out_rows, in_cols]:
        return True
    dt = str(info.get("dtype") or "")
    if dt in ("U32", "I32") and shp[0] == out_rows and in_cols > 0:
        return (shp[1] * 8 == in_cols) or (shp[1] * 16 == in_cols) or (in_cols % max(shp[1], 1) == 0)
    if dt == "U8" and shp == [out_rows, in_cols // 2]:
        return True
    return False


def _q_bytes(rows: int, cols: int, bits: int, group_size: int = 64) -> int:
    data = rows * ((cols * bits + 7) // 8)
    params = rows * ((cols + group_size - 1) // group_size) * 4
    return data + params


def _estimate_ling3_tier_admission(
    *,
    always_active_bb_params: int,
    kda_dense_proj_params: int,
    routed_params: int,
    mtp_params: int,
    host_ram_gib: float | None = None,
) -> dict[str, dict[str, Any]]:
    tiers: list[tuple[str, float, float]] = [
        ("24gb-m4-rco-q3", 24.0, 17.76),
        ("128gb-max", 128.0, 104.0),
        ("256gb-studio", 256.0, 210.0),
        ("512gb-ultra", 512.0, 420.0),
    ]
    if host_ram_gib is not None:
        tiers.append((f"custom-{host_ram_gib:.0f}gb", host_ram_gib, host_ram_gib * 0.78))

    total_p = always_active_bb_params + routed_params + mtp_params
    q4_all_gib = (total_p * 0.5625) / (1024**3)
    rco_325_gib = ((always_active_bb_params - kda_dense_proj_params) * 0.5625 + kda_dense_proj_params * 0.4375 + (routed_params + mtp_params) * (3.25 / 8.0)) / (1024**3)
    rco_255_gib = (total_p * (2.55 / 8.0)) / (1024**3)
    out: dict[str, dict[str, Any]] = {}
    for name, _, budget_gib in tiers:
        out[name] = {
            "wired_budget_gib": round(budget_gib, 2),
            "q4_all_wired_gib": round(q4_all_gib, 2),
            "rco_325_all_wired_gib": round(rco_325_gib, 2),
            "rco_255_all_wired_gib": round(rco_255_gib, 2),
            "fits_all_wired": rco_255_gib <= budget_gib,
            "mode": (
                "full-wired-q4"
                if q4_all_gib <= budget_gib
                else ("full-wired-gsq-rco" if rco_255_gib <= budget_gib else "rco-q3-kda+nvme-cold-experts")
            ),
        }
    return out


def validate_ling3_metadata_dir(
    meta_dir: pathlib.Path,
    *,
    host_ram_gib: float | None = None,
) -> Ling3MetadataReport:
    """Validate staged HF metadata + Range safetensors headers for `Ling-3.1-flash-1019` (`bailing_hybrid`)."""
    meta_dir = pathlib.Path(meta_dir)
    cfg_p = meta_dir / "config.json"
    if not cfg_p.is_file():
        raise FileNotFoundError(f"config.json not found in {meta_dir}")
    raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    cfg = _effective_ling3_cfg(raw_cfg)
    errors: list[str] = []

    model_type = str(raw_cfg.get("model_type") or cfg.get("model_type") or "")
    if model_type not in ("bailing_hybrid", "bailing_hybrid_mtp"):
        errors.append(f"unexpected model_type {model_type!r} (expected 'bailing_hybrid')")

    hidden = int(cfg.get("hidden_size") or 0)
    inter = int(cfg.get("intermediate_size") or 0)
    moe_inter = int(cfg.get("moe_intermediate_size") or 0)
    sh_inter = int(cfg.get("moe_shared_expert_intermediate_size") or moe_inter)
    vocab = int(cfg.get("vocab_size") or 0)
    n_layers = int(cfg.get("num_hidden_layers") or 0)
    lgs = int(cfg.get("layer_group_size") or 8)
    if hidden <= 0 or inter <= 0 or vocab <= 0 or n_layers <= 0:
        errors.append(f"invalid geometry hidden={hidden}, inter={inter}, vocab={vocab}, layers={n_layers}")

    ltypes = resolve_ling3_layer_types(cfg)
    mlp_types = resolve_ling3_mlp_layer_types(cfg)
    lin_l = sum(1 for x in ltypes if x == "linear_attention")
    mla_l = sum(1 for x in ltypes if x == "mla_attention")
    dense_l = sum(1 for x in mlp_types if x == "dense")
    moe_l = sum(1 for x in mlp_types if x == "sparse")

    no_kda_lora = bool(cfg.get("no_kda_lora", True))
    kda_safe_gate = bool(cfg.get("kda_safe_gate", True))
    kda_lb = float(cfg.get("kda_lower_bound") if cfg.get("kda_lower_bound") is not None else -5.0)
    lin_silu = bool(cfg.get("linear_silu", True))
    conv_k = int(cfg.get("short_conv_kernel_size") or 4)
    if not no_kda_lora:
        errors.append("Ling-3.1 expects no_kda_lora=True (dense f_proj/g_proj)")
    if kda_lb >= 0.0:
        errors.append(f"expected negative kda_lower_bound (e.g. -5.0), got {kda_lb}")

    n_heads = int(cfg.get("num_attention_heads") or 0)
    head_dim = int(cfg.get("head_dim") or 128)
    q_lora = int(cfg.get("q_lora_rank") or 0)
    kv_lora = int(cfg.get("kv_lora_rank") or 0)
    qk_nope = int(cfg.get("qk_nope_head_dim") or 0)
    qk_rope = int(cfg.get("qk_rope_head_dim") or 0)
    v_hd = int(cfg.get("v_head_dim") or 0)
    gran = str(cfg.get("gated_attention_proj_granularity_type") or "head_wise")
    mla_gated = gran in ("head_wise", "element_wise")

    n_exp = int(cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 0)
    n_shared = int(cfg.get("num_shared_experts") or cfg.get("n_shared_experts") or 0)
    top_k = int(cfg.get("num_experts_per_tok") or 0)
    n_group = int(cfg.get("n_group") or 1)
    topk_group = int(cfg.get("topk_group") or 1)
    routed_scale = float(cfg.get("routed_scaling_factor") or 1.0)
    mtp_layers = int(cfg.get("num_nextn_predict_layers") or 0)

    wm, _ = _read_weight_map_keys(meta_dir)
    hdr_tensors, hdr_shards = _read_header_summary_tensors(meta_dir)
    all_keys = sorted(set(wm.keys()) | set(hdr_tensors.keys()))

    bb_cnt = 0
    mtp_cnt = 0
    exp_ids: set[int] = set()
    long_bb: list[str] = []
    long_mtp: list[str] = []
    max_bb_len = 0
    exp_re = re.compile(r"\.experts\.(\d+)\.")

    for raw_name in all_keys:
        if any(raw_name.endswith(s) for s in _STRIP_QUANT_SUFFIXES if s not in (".weight_packed",)):
            continue
        if is_ling3_mtp_tensor(raw_name, n_layers if mtp_layers > 0 else None):
            mtp_cnt += 1
            try:
                nm = normalize_ling3_tensor_name(
                    raw_name,
                    layer_types=ltypes,
                    mtp_source_layer=n_layers if mtp_layers > 0 else None,
                    mtp_standalone=True,
                )
            except ValueError:
                long_mtp.append(raw_name)
            else:
                if len(nm.encode("utf-8")) > 47:
                    long_mtp.append(nm)
            continue

        bb_cnt += 1
        em = exp_re.search(raw_name)
        if em is not None:
            exp_ids.add(int(em.group(1)))
        try:
            nm = normalize_ling3_tensor_name(raw_name, layer_types=ltypes)
        except ValueError:
            long_bb.append(raw_name)
        else:
            blen = len(nm.encode("utf-8"))
            max_bb_len = max(max_bb_len, blen)
            if blen > 47:
                long_bb.append(nm)

    if long_bb:
        errors.append(f"{len(long_bb)} backbone tensor name(s) exceed 47 B (sample: {long_bb[:3]})")
    if long_mtp:
        errors.append(f"{len(long_mtp)} MTP tensor name(s) exceed 47 B (sample: {long_mtp[:3]})")

    if hdr_tensors:
        qkv_dim = n_heads * head_dim
        g_rows = n_heads if gran == "head_wise" else n_heads * v_hd
        for l_idx in range(min(n_layers, 10)):
            for pfx in (f"model.layers.{l_idx}", f"layers.{l_idx}"):
                if l_idx < len(ltypes) and ltypes[l_idx] == "linear_attention":
                    alog = hdr_tensors.get(f"{pfx}.attention.A_log") or hdr_tensors.get(f"{pfx}.linear_attn.A_log")
                    if alog is not None and alog.get("shape") != [n_heads]:
                        errors.append(f"L{l_idx} A_log shape {alog.get('shape')} != [{n_heads}]")
                    dt = hdr_tensors.get(f"{pfx}.attention.dt_bias") or hdr_tensors.get(f"{pfx}.linear_attn.dt_bias")
                    if dt is not None and dt.get("shape") != [qkv_dim]:
                        errors.append(f"L{l_idx} dt_bias shape {dt.get('shape')} != [{qkv_dim}]")
                    qcv = hdr_tensors.get(f"{pfx}.attention.q_conv1d.weight") or hdr_tensors.get(f"{pfx}.linear_attn.q_conv1d.weight")
                    if qcv is not None and qcv.get("shape") not in ([qkv_dim, 1, conv_k], [qkv_dim, conv_k]):
                        errors.append(f"L{l_idx} q_conv1d shape {qcv.get('shape')} != [{qkv_dim}, 1, {conv_k}]")
                    fp = hdr_tensors.get(f"{pfx}.attention.f_proj.weight") or hdr_tensors.get(f"{pfx}.linear_attn.f_proj.weight")
                    if fp is not None and not _matches_linear_shape(fp, qkv_dim, hidden):
                        errors.append(f"L{l_idx} KDA f_proj shape {fp.get('shape')} != [{qkv_dim}, {hidden}]")
                    on = hdr_tensors.get(f"{pfx}.attention.o_norm.weight") or hdr_tensors.get(f"{pfx}.linear_attn.o_norm.weight")
                    if on is not None and on.get("shape") != [head_dim]:
                        errors.append(f"L{l_idx} KDA o_norm shape {on.get('shape')} != [{head_dim}]")
                elif l_idx < len(ltypes):
                    kva = hdr_tensors.get(f"{pfx}.attention.kv_a_proj_with_mqa.weight") or hdr_tensors.get(f"{pfx}.self_attn.kv_a_proj_with_mqa.weight")
                    if kva is not None and not _matches_linear_shape(kva, kv_lora + qk_rope, hidden):
                        errors.append(f"L{l_idx} MLA kv_a shape {kva.get('shape')} != [{kv_lora + qk_rope}, {hidden}]")
                    gp = hdr_tensors.get(f"{pfx}.attention.g_proj.weight") or hdr_tensors.get(f"{pfx}.self_attn.g_proj.weight")
                    if gp is not None and not _matches_linear_shape(gp, g_rows, hidden):
                        errors.append(f"L{l_idx} MLA g_proj shape {gp.get('shape')} != [{g_rows}, {hidden}]")
                    dp = hdr_tensors.get(f"{pfx}.attention.dense.weight") or hdr_tensors.get(f"{pfx}.self_attn.o_proj.weight")
                    if dp is not None and not _matches_linear_shape(dp, hidden, n_heads * v_hd):
                        errors.append(f"L{l_idx} MLA dense/o_proj shape {dp.get('shape')} != [{hidden}, {n_heads * v_hd}]")
        if mtp_layers > 0:
            eh = hdr_tensors.get(f"model.layers.{n_layers}.eh_proj.weight") or hdr_tensors.get("mtp.0.eh_proj.weight")
            if eh is not None and not _matches_linear_shape(eh, hidden, 2 * hidden):
                errors.append(f"MTP eh_proj shape {eh.get('shape')} != [{hidden}, {2 * hidden}]")

    # Exact parameter accounting matching 54/54 safetensors shards.
    qkv_dim = n_heads * head_dim
    embed_p = 2 * vocab * hidden
    kda_large_per_l = 6 * hidden * qkv_dim  # q, k, v, f, g, o_proj
    kda_small_per_l = hidden * n_heads + 3 * qkv_dim * conv_k + n_heads + qkv_dim + head_dim
    g_rows = n_heads if gran == "head_wise" else n_heads * v_hd
    mla_per_l = (
        hidden * q_lora
        + q_lora
        + q_lora * n_heads * (qk_nope + qk_rope)
        + hidden * (kv_lora + qk_rope)
        + kv_lora
        + kv_lora * n_heads * (qk_nope + v_hd)
        + hidden * g_rows
        + n_heads * v_hd * hidden
    )
    dense_mlp_p = 3 * hidden * inter
    shared_mlp_p = n_shared * 3 * hidden * sh_inter
    router_p = n_exp * (hidden + 1)
    one_exp_p = 3 * hidden * moe_inter
    norms_p = (2 * n_layers + 1) * hidden

    kda_dense_p = lin_l * kda_large_per_l
    active_bb_p = (
        embed_p
        + norms_p
        + lin_l * (kda_large_per_l + kda_small_per_l)
        + mla_l * mla_per_l
        + dense_l * dense_mlp_p
        + moe_l * (shared_mlp_p + router_p)
    )
    routed_bb_p = moe_l * n_exp * one_exp_p
    routed_tok_p = moe_l * top_k * one_exp_p
    mtp_active_p = mtp_layers * (4 * hidden + 2 * hidden * hidden + mla_per_l + shared_mlp_p + router_p)
    mtp_routed_p = mtp_layers * n_exp * one_exp_p
    mtp_total_p = mtp_active_p + mtp_routed_p
    always_plus_mtp_p = active_bb_p + mtp_total_p
    total_p = active_bb_p + routed_bb_p + mtp_total_p

    # Byte-budget comparison: Q4 baseline vs RCO-Q3 dense-KDA (M4 24GB 17.76 GiB wired ceiling).
    # Includes 16-bit/32-bit norm/small-gate overhead + Q4/Q8 embed & MLA/shared/dense + optional MTP residency.
    bb_non_kda_q4_b = (
        _q_bytes(2 * vocab, hidden, 4, 64)
        + norms_p * 2
        + lin_l * (hidden * n_heads * 2 + (3 * qkv_dim * conv_k + n_heads + qkv_dim) * 4 + head_dim * 2)
        + mla_l * (
            _q_bytes(q_lora, hidden, 4, 64)
            + _q_bytes(n_heads * (qk_nope + qk_rope), q_lora, 4, 64)
            + _q_bytes(kv_lora + qk_rope, hidden, 4, 64)
            + _q_bytes(n_heads * (qk_nope + v_hd), kv_lora, 4, 64)
            + _q_bytes(hidden, n_heads * v_hd, 4, 64)
            + (q_lora + kv_lora + g_rows * hidden) * 2
        )
        + dense_l * (2 * _q_bytes(inter, hidden, 4, 64) + _q_bytes(hidden, inter, 4, 64))
        + moe_l * (n_shared * (2 * _q_bytes(sh_inter, hidden, 4, 64) + _q_bytes(hidden, sh_inter, 4, 64)) + n_exp * hidden * 2 + n_exp * 4)
    )
    kda_q4_b = lin_l * (5 * _q_bytes(qkv_dim, hidden, 4, 64) + _q_bytes(hidden, qkv_dim, 4, 64))
    kda_q3_b = lin_l * (5 * _q_bytes(qkv_dim, hidden, 3, 64) + _q_bytes(hidden, qkv_dim, 3, 64))
    # 24.51B combined (16.34B always-active + 8.17B MTP) at Q4 vs RCO-Q3 + compact MTP.
    mtp_q4_b = mtp_layers * int(mtp_total_p * 0.5625)
    mtp_rco_b = mtp_layers * (int(mtp_active_p * 0.5625) + int(mtp_routed_p * (2.5 / 8.0)))

    q4_bb_gib = (bb_non_kda_q4_b + kda_q4_b) / (1024**3)
    q3_bb_gib = (bb_non_kda_q4_b + kda_q3_b) / (1024**3)
    q4_plus_mtp_gib = (bb_non_kda_q4_b + kda_q4_b + mtp_q4_b) / (1024**3)
    q3_plus_mtp_gib = (bb_non_kda_q4_b + kda_q3_b + mtp_rco_b) / (1024**3)

    m4_wired_usable_gib = 16.25  # 17.76 GiB M4 ceiling minus ~1.5 GiB KV/scratch/Metal workspace
    hot_q4_gib = max(0.0, m4_wired_usable_gib - q4_plus_mtp_gib)
    hot_q3_gib = max(0.0, m4_wired_usable_gib - q3_plus_mtp_gib)

    return Ling3MetadataReport(
        metadata_dir=meta_dir,
        model_type=model_type,
        hidden_size=hidden,
        intermediate_size=inter,
        moe_intermediate_size=moe_inter,
        vocab_size=vocab,
        num_hidden_layers=n_layers,
        layer_group_size=lgs,
        linear_attention_layers=lin_l,
        mla_layers=mla_l,
        dense_layers=dense_l,
        moe_layers=moe_l,
        no_kda_lora=no_kda_lora,
        kda_safe_gate=kda_safe_gate,
        kda_lower_bound=kda_lb,
        linear_silu=lin_silu,
        short_conv_kernel_size=conv_k,
        mla_gated_proj=mla_gated,
        gated_attention_granularity=gran,
        num_attention_heads=n_heads,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_hd,
        n_routed_experts=n_exp,
        n_shared_experts=n_shared,
        num_experts_per_tok=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scale,
        mtp_layers=mtp_layers,
        always_active_backbone_params=active_bb_p,
        kda_dense_proj_params=kda_dense_p,
        routed_expert_params=routed_bb_p,
        routed_per_tok_params=routed_tok_p,
        mtp_total_params=mtp_total_p,
        always_active_plus_mtp_params=always_plus_mtp_p,
        total_params=total_p,
        q4_always_active_bb_gib=round(q4_bb_gib, 3),
        rco_q3_always_active_bb_gib=round(q3_bb_gib, 3),
        q4_always_active_plus_mtp_gib=round(q4_plus_mtp_gib, 3),
        rco_q3_always_active_plus_mtp_gib=round(q3_plus_mtp_gib, 3),
        m4_24gb_hot_cache_q4_gib=round(hot_q4_gib, 3),
        m4_24gb_hot_cache_rco_q3_gib=round(hot_q3_gib, 3),
        mla_kv_bytes_per_token_fp16=mla_l * (kv_lora + qk_rope) * 2,
        kda_state_bytes_fp32=lin_l * n_heads * head_dim * head_dim * 4,
        backbone_tensor_count=bb_cnt,
        mtp_tensor_count=mtp_cnt,
        routed_expert_ids_detected=len(exp_ids),
        long_backbone_names=len(long_bb),
        long_mtp_names=len(long_mtp),
        max_backbone_name_bytes=max_bb_len,
        header_summary_shards=hdr_shards,
        tier_admission=_estimate_ling3_tier_admission(
            always_active_bb_params=active_bb_p,
            kda_dense_proj_params=kda_dense_p,
            routed_params=routed_bb_p,
            mtp_params=mtp_total_p,
            host_ram_gib=host_ram_gib,
        ),
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Runtime `.amf` `config.json` Builders
# ---------------------------------------------------------------------------


def build_ling3_config_json(
    src_cfg: Mapping[str, Any],
    *,
    cold_experts_count: int = 0,
    dense_kda_rco_q3: bool = False,
) -> dict[str, Any]:
    """Build canonical `.amf` runtime `config.json` for `Ling-3.1` (`bailing_hybrid`)."""
    cfg = _effective_ling3_cfg(src_cfg)
    n_layers = int(cfg.get("num_hidden_layers") or 72)
    ltypes = resolve_ling3_layer_types(cfg)
    mlp_types = resolve_ling3_mlp_layer_types(cfg)
    n_heads = int(cfg.get("num_attention_heads") or 64)
    head_dim = int(cfg.get("head_dim") or 128)
    qk_nope = int(cfg.get("qk_nope_head_dim") or 128)
    qk_rope = int(cfg.get("qk_rope_head_dim") or 64)
    v_hd = int(cfg.get("v_head_dim") or 128)
    moe_inter = int(cfg.get("moe_intermediate_size") or 1280)
    sh_inter = int(cfg.get("moe_shared_expert_intermediate_size") or moe_inter)
    gran = str(cfg.get("gated_attention_proj_granularity_type") or "head_wise")
    kda_lb = float(cfg.get("kda_lower_bound") if cfg.get("kda_lower_bound") is not None else -5.0)
    exp_limits = cfg.get("expert_swiglu_limit_list")
    sh_limits = cfg.get("share_expert_swiglu_limit_list")

    out: dict[str, Any] = {
        "model_type": "bailing_hybrid",
        "architectures": ["BailingMoeV3ForCausalLM"],
        "hidden_size": int(cfg.get("hidden_size") or 4096),
        "intermediate_size": int(cfg.get("intermediate_size") or 10240),
        "moe_intermediate_size": moe_inter,
        "moe_shared_expert_intermediate_size": sh_inter,
        "num_hidden_layers": n_layers,
        "layer_group_size": int(cfg.get("layer_group_size") or 8),
        "vocab_size": int(cfg.get("vocab_size") or 157184),
        "first_k_dense_replace": int(cfg.get("first_k_dense_replace") or 4),
        "layer_types": ltypes,
        "mlp_layer_types": mlp_types,
        "linear_attn_kind": "kda",
        "no_kda_lora": bool(cfg.get("no_kda_lora", True)),
        "kda_safe_gate": bool(cfg.get("kda_safe_gate", True)),
        "kda_lower_bound": kda_lb,
        "linear_lower_bound": kda_lb,
        "linear_silu": bool(cfg.get("linear_silu", True)),
        "short_conv_kernel_size": int(cfg.get("short_conv_kernel_size") or 4),
        "linear_conv_kernel_dim": int(cfg.get("short_conv_kernel_size") or 4),
        "linear_num_heads": n_heads,
        "linear_head_dim": head_dim,
        "linear_o_norm_gate": "sigmoid",
        "head_dim": head_dim,
        "num_attention_heads": n_heads,
        "num_key_value_heads": int(cfg.get("num_key_value_heads") or n_heads),
        "q_lora_rank": int(cfg.get("q_lora_rank") or 1536),
        "kv_lora_rank": int(cfg.get("kv_lora_rank") or 512),
        "qk_nope_head_dim": qk_nope,
        "qk_rope_head_dim": qk_rope,
        "qk_head_dim": qk_nope + qk_rope,
        "v_head_dim": v_hd,
        "mla_gated_proj": gran in ("head_wise", "element_wise"),
        "gated_attention_proj_granularity_type": gran,
        "mla_gate_act": "sigmoid",
        "rope_interleave": bool(cfg.get("rope_interleave", True)),
        "rope_theta": float(cfg.get("rope_theta") or 6000000.0),
        "rotary_dim": int(cfg.get("rotary_dim") or qk_rope),
        "n_routed_experts": int(cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 512),
        "num_experts": int(cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 512),
        "n_shared_experts": int(cfg.get("num_shared_experts") or cfg.get("n_shared_experts") or 1),
        "num_shared_experts": int(cfg.get("num_shared_experts") or cfg.get("n_shared_experts") or 1),
        "num_experts_per_tok": int(cfg.get("num_experts_per_tok") or 8),
        "n_group": int(cfg.get("n_group") or 8),
        "topk_group": int(cfg.get("topk_group") or 4),
        "group_score_topk": 2,
        "scoring_func": str(cfg.get("scoring_func") or cfg.get("score_function") or "sigmoid"),
        "topk_method": str(cfg.get("topk_method") or "noaux_tc"),
        "norm_topk_prob": bool(cfg.get("norm_topk_prob", True)),
        "routed_scaling_factor": float(cfg.get("routed_scaling_factor") or 2.5),
        "rms_norm_eps": float(cfg.get("rms_norm_eps") or 1e-6),
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "max_position_embeddings": int(cfg.get("max_position_embeddings") or 262144),
        "num_nextn_predict_layers": int(cfg.get("num_nextn_predict_layers") or 0),
        "mtp_use_kda": bool(cfg.get("mtp_use_kda", False)),
    }
    if isinstance(exp_limits, list):
        out["expert_swiglu_limit_list"] = [float(x) for x in exp_limits]
    if isinstance(sh_limits, list):
        out["share_expert_swiglu_limit_list"] = [float(x) for x in sh_limits]
    if dense_kda_rco_q3:
        out["dense_kda_rco_q3"] = True
    if cold_experts_count > 0:
        out["cold_experts_count"] = int(cold_experts_count)
    return out


def build_ling3_mtp_config_dict(src_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build standalone `1L` `mtp.amf` `config.json` for `Ling-3.1` (`bailing_hybrid_mtp`)."""
    base = build_ling3_config_json(src_cfg)
    eff = _effective_ling3_cfg(src_cfg)
    n_bb = int(eff.get("num_hidden_layers") or 72)
    exp_limits = eff.get("expert_swiglu_limit_list")
    sh_limits = eff.get("share_expert_swiglu_limit_list")
    mtp_exp_lim = float(exp_limits[n_bb]) if isinstance(exp_limits, list) and len(exp_limits) > n_bb else 10.0
    mtp_sh_lim = float(sh_limits[n_bb]) if isinstance(sh_limits, list) and len(sh_limits) > n_bb else 10.0
    return {
        **base,
        "model_type": "bailing_hybrid_mtp",
        "mtp_arch": "bailing_hybrid",
        "format": "mtp",
        "mtp_chained": False,
        "num_hidden_layers": 1,
        "first_k_dense_replace": 0,
        "layer_types": ["mla_attention"],
        "mlp_layer_types": ["sparse"],
        "is_moe": True,
        "swiglu_limit": mtp_exp_lim,
        "expert_swiglu_limit_list": [mtp_exp_lim],
        "share_expert_swiglu_limit_list": [mtp_sh_lim],
    }


# ---------------------------------------------------------------------------
# Cold-Expert Selection & Checkpoint Conversion (`W27`)
# ---------------------------------------------------------------------------


def _select_ling3_cold_experts(
    *,
    n_layers: int,
    n_experts: int,
    mlp_types: Sequence[str],
    tier_profile: Ling3TierProfile | None,
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
            frac = 0.88 if tier_profile.name.startswith("24gb-") else 0.60
            eff_per_layer = max(1, int(round(n_experts * frac)))
    if eff_per_layer is not None and eff_per_layer > 0 and n_experts > 0:
        keep_hot = max(0, n_experts - eff_per_layer)
        for l_idx in range(n_layers):
            if l_idx < len(mlp_types) and mlp_types[l_idx] == "dense":
                continue
            for e_idx in range(keep_hot, n_experts):
                out.add((l_idx, e_idx))
    return out


@dataclass(frozen=True)
class Ling3ConvertReport:
    """Summary returned by `convert_ling3_checkpoint`."""

    model_type: str
    backbone_amf: pathlib.Path
    backbone_metas: tuple[PackedTensorMeta, ...]
    mtp_amf: pathlib.Path | None = None
    mtp_metas: tuple[PackedTensorMeta, ...] = ()
    cold_expert_count: int = 0
    dense_kda_rco_q3: bool = False
    config_json_path: pathlib.Path | None = None


def _repack_quantize_eh_proj_sme2(
    amf_path: pathlib.Path,
    *,
    bits: int = 3,
    group_size: int = 64,
    use_rco: bool = True,
    repack_sme2: bool = True,
) -> list[PackedTensorMeta]:
    """Stream-quantize `layers.0.eh_proj.weight` (`eh_proj.weight`) in an MTP `.amf` to L2 `sme2_panel64`."""
    import os
    from aminfer_tools.amf import (
        AmfReader,
        _BlobWriter,
        _COPY_CHUNK_BYTES,
        _align_up,
        _crc32,
        _pack_directory,
        _pack_header,
        _read_exact,
        _set_nocache,
        pack_sme2_panel64_q2_panel,
        pack_sme2_panel64_q3_panel,
        pack_sme2_panel64_q4_panel,
        sme2_panel64_data_bytes,
    )
    from aminfer_tools.gsq_rco import quantize_rows_affine

    if bits not in (2, 3, 4):
        raise ValueError(f"eh_proj_bits must be 2, 3, or 4, got {bits}")
    q_dtype = {2: DType.Q2_AFFINE, 3: DType.Q3_AFFINE, 4: DType.Q4_AFFINE}[bits]
    pack_fn = {
        2: pack_sme2_panel64_q2_panel,
        3: pack_sme2_panel64_q3_panel,
        4: pack_sme2_panel64_q4_panel,
    }[bits]

    src_p = pathlib.Path(amf_path)
    with AmfReader(src_p) as rdr:
        header = rdr.header
        entries = list(rdr.entries)
        eh_targets = {"layers.0.eh_proj.weight", "eh_proj.weight", "layers.0.eh_proj", "eh_proj"}
        eh_m = next((m for m in entries if m.name in eh_targets), None)
        if eh_m is None:
            return entries
        if len(eh_m.shape) != 2 or eh_m.shape[1] % group_size != 0:
            raise ValueError(f"cannot RCO-quantize {eh_m.name!r} with shape {eh_m.shape} and group_size={group_size}")

        n_rows, k = eh_m.shape
        groups = k // group_size
        row_data_b = (k * bits) // 8
        row_param_b = groups * 4
        if repack_sme2:
            new_layout = TensorLayout.SME2_PANEL64
            new_data_b = sme2_panel64_data_bytes(q_dtype, n_rows, k, group_size)
            new_params_off = 0
            new_params_b = 0
            new_blob_b = _align_up(max(new_data_b, 1), APPLE_PAGE_SIZE)
        else:
            new_layout = TensorLayout.ROW_MAJOR
            new_data_b = n_rows * row_data_b
            new_params_off = _align_up(new_data_b, 64)
            new_params_b = n_rows * row_param_b
            new_blob_b = _align_up(new_params_off + new_params_b, APPLE_PAGE_SIZE)

        tensor_count = len(entries)
        dir_offset = APPLE_PAGE_SIZE
        dir_bytes = 0 if tensor_count == 0 else _align_up(tensor_count * 128, APPLE_PAGE_SIZE)
        cursor = dir_offset + dir_bytes
        planned: list[tuple[PackedTensorMeta, bool, int, int, int, int, int, TensorLayout]] = []
        for m in entries:
            is_eh = m.name == eh_m.name
            b_off = cursor
            if is_eh:
                planned.append((m, True, b_off, new_blob_b, new_data_b, new_params_off, new_params_b, new_layout))
                cursor += new_blob_b
            else:
                planned.append((m, False, b_off, m.blob_bytes, m.data_bytes, m.params_offset, m.params_bytes, m.layout))
                cursor += m.blob_bytes

        partial = src_p.with_name(src_p.name + ".ehq.partial")
        new_metas: list[PackedTensorMeta] = []
        try:
            with src_p.open("rb", buffering=0) as src_fh, partial.open("wb") as out_fh:
                _set_nocache(src_fh.fileno())
                _set_nocache(out_fh.fileno())
                _BlobWriter(out_fh).zeros(dir_offset + dir_bytes)

                for m, is_eh, b_off, nb_bytes, nd_bytes, np_off, np_bytes, n_lay in planned:
                    if out_fh.tell() != b_off:
                        raise AssertionError(f"planned offset {b_off} != file pos {out_fh.tell()} for {m.name!r}")
                    w = _BlobWriter(out_fh)
                    if is_eh:
                        if repack_sme2:
                            written_eh = 0
                            for r0 in range(0, n_rows, 16):
                                r1 = min(n_rows, r0 + 16)
                                panel_rows = rdr.read_rows_f32(m, range(r0, r1))
                                qb, pb, _, _ = quantize_rows_affine(
                                    panel_rows, bits=bits, group_size=group_size, use_rco=use_rco
                                )
                                cnt = r1 - r0
                                q_mv = memoryview(qb)
                                p_mv = memoryview(pb)
                                q_list = [q_mv[c * row_data_b : (c + 1) * row_data_b] for c in range(cnt)]
                                p_list = [p_mv[c * row_param_b : (c + 1) * row_param_b] for c in range(cnt)]
                                pnl = pack_fn(q_list, p_list, k)
                                w.write(pnl)
                                written_eh += len(pnl)
                            w.zeros(nb_bytes - written_eh)
                        else:
                            q_parts: list[bytes] = []
                            p_parts: list[bytes] = []
                            for r0 in range(0, n_rows, 64):
                                r1 = min(n_rows, r0 + 64)
                                rows = rdr.read_rows_f32(m, range(r0, r1))
                                qb, pb, _, _ = quantize_rows_affine(
                                    rows, bits=bits, group_size=group_size, use_rco=use_rco
                                )
                                q_parts.append(qb)
                                p_parts.append(pb)
                            q_all = b"".join(q_parts)
                            p_all = b"".join(p_parts)
                            w.write(q_all)
                            w.zeros(np_off - len(q_all))
                            w.write(p_all)
                            w.zeros(nb_bytes - (np_off + len(p_all)))
                        new_metas.append(
                            PackedTensorMeta(
                                name=m.name,
                                kind=m.kind,
                                dtype=q_dtype,
                                quant_bits=bits,
                                group_size=group_size,
                                layer=m.layer,
                                expert=m.expert,
                                shape=m.shape,
                                blob_offset=b_off,
                                blob_bytes=nb_bytes,
                                data_bytes=nd_bytes,
                                params_offset=np_off,
                                params_bytes=np_bytes,
                                blob_crc32=w.crc & 0xFFFFFFFF,
                                layout=n_lay,
                                quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
                            )
                        )
                    else:
                        done = 0
                        while done < m.blob_bytes:
                            take = min(_COPY_CHUNK_BYTES, m.blob_bytes - done)
                            w.write(_read_exact(src_fh, m.blob_offset + done, take))
                            done += take
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
                                blob_offset=b_off,
                                blob_bytes=m.blob_bytes,
                                data_bytes=m.data_bytes,
                                params_offset=m.params_offset,
                                params_bytes=m.params_bytes,
                                blob_crc32=w.crc & 0xFFFFFFFF,
                                layout=m.layout,
                                quant_param_kind=m.quant_param_kind,
                            )
                        )
                dir_raw = _pack_directory(new_metas)
                new_hdr = _pack_header(
                    header.model_id.encode("utf-8"),
                    tensor_count,
                    dir_offset,
                    dir_bytes,
                    _crc32(dir_raw),
                    flags=header.flags,
                )
                out_fh.seek(0)
                out_fh.write(new_hdr)
                if dir_raw:
                    out_fh.seek(dir_offset)
                    out_fh.write(dir_raw)
                out_fh.flush()
                os.fsync(out_fh.fileno())
            os.replace(partial, src_p)
            return new_metas
        except Exception:
            if partial.exists():
                partial.unlink()
            raise


def export_ling3_mtp(
    input_path: pathlib.Path,
    output_amf: pathlib.Path,
    *,
    target_config: Mapping[str, Any] | None = None,
    model_id: str = "ling3-mtp",
    dtype_mode: str = "auto",
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    use_rco: bool = True,
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
    repack_sme2: bool = False,
    eh_proj_bits: int | None = None,
    embed_metadata: bool = True,
    config_output: pathlib.Path | None = None,
    workers: int = 1,
) -> list[PackedTensorMeta]:
    """Extract & pack `Ling-3.1` `1L` Gated-`MLA` `MTP` (`layers.72.*` / `mtp.0.*`) into standalone `.amf`."""
    in_p = pathlib.Path(input_path)
    out_p = pathlib.Path(output_amf)
    mtp_cfg_b: bytes | None = None
    if in_p.is_file() and in_p.suffix.lower() == ".gguf":
        from aminfer_tools.gguf import pack_amf_from_gguf

        metas = pack_amf_from_gguf(
            in_p,
            out_p,
            model_id=model_id[:63],
            dtype_mode=dtype_mode,
            cold_experts=cold_experts,
            all_experts_cold=all_experts_cold,
            embed_metadata=embed_metadata,
            draft_config_output=config_output,
            mimo_mtp_mode=True,
            sme2_pack=("eh-proj" if repack_sme2 else None),
        )
    else:
        resolved = resolve_checkpoint(in_p)
        raw = _collect_tensors(resolved.shard_paths)
        tcfg = dict(target_config) if target_config is not None else {}
        if not tcfg and resolved.config_path is not None and resolved.config_path.is_file():
            tcfg = json.loads(resolved.config_path.read_text(encoding="utf-8"))
        eff_cfg = _effective_ling3_cfg(tcfg)
        bb_layers = int(eff_cfg.get("num_hidden_layers") or 0)

        has_explicit_mtp = any(
            _MTP_KEY_RE.match(k) is not None
            or (bb_layers > 0 and (k.startswith(f"layers.{bb_layers}.") or k.startswith(f"model.layers.{bb_layers}.")))
            for k in raw
        )
        has_standalone_mtp = any(
            k in ("eh_proj.weight", "layers.0.eh_proj.weight", "model.layers.0.eh_proj.weight") for k in raw
        )
        if has_explicit_mtp:
            mtp_keys = [k for k in raw if is_ling3_mtp_tensor(k, bb_layers if bb_layers > 0 else None)]
            non_mtp_excludes = [k for k in raw if k not in set(mtp_keys)]
        elif has_standalone_mtp:
            non_mtp_excludes = []
        else:
            raise ValueError(f"no Ling-3.1 MTP tensors found in {in_p}")

        mtp_cfg = build_ling3_mtp_config_dict(tcfg)
        mtp_cfg_b = json.dumps(mtp_cfg, indent=2).encode("utf-8") + b"\n"
        if config_output is not None:
            config_output.parent.mkdir(parents=True, exist_ok=True)
            config_output.write_bytes(mtp_cfg_b)

        with tempfile.TemporaryDirectory() as tmp_d:
            tmp_cfg = pathlib.Path(tmp_d) / "mtp_config.json"
            tmp_cfg.write_bytes(mtp_cfg_b)
            metas = pack_amf_from_safetensors(
                in_p,
                out_p,
                model_id=model_id[:63],
                dtype_mode=dtype_mode,
                quant_profile=quant_profile,
                target_bits=target_bits,
                use_rco=use_rco,
                cold_experts=cold_experts,
                all_experts_cold=all_experts_cold,
                exclude_tensors=non_mtp_excludes,
                strip_vision=True,
                sme2_pack=("eh-proj" if repack_sme2 else "none"),
                embed_metadata=embed_metadata,
                config_path=tmp_cfg if embed_metadata else None,
                workers=workers,
                ling3_config=eff_cfg,
                ling3_mtp_standalone=True,
                ling3_mtp_source_layer=bb_layers if (has_explicit_mtp and bb_layers > 0) else None,
            )
    eh_m = next(
        (m for m in metas if m.name in ("layers.0.eh_proj.weight", "eh_proj.weight", "layers.0.eh_proj", "eh_proj")),
        None,
    )
    eff_eh_bits = eh_proj_bits
    if (
        eff_eh_bits is None
        and repack_sme2
        and eh_m is not None
        and eh_m.dtype in (DType.F16, DType.BF16, DType.F32)
        and (quant_profile == "gsq-rco" or (len(eh_m.shape) == 2 and eh_m.shape[0] * eh_m.shape[1] * 2 > 16 * 1024 * 1024))
    ):
        eff_eh_bits = 3
    if eff_eh_bits is not None and eh_m is not None:
        metas = _repack_quantize_eh_proj_sme2(
            out_p,
            bits=eff_eh_bits,
            use_rco=use_rco,
            repack_sme2=repack_sme2,
        )
    elif repack_sme2 and eh_m is not None and eh_m.layout == TensorLayout.ROW_MAJOR:
        metas = repack_amf_sme2(out_p, out_p, targets="eh-proj", workers=workers)
    if embed_metadata and mtp_cfg_b is not None:
        metas = embed_metadata_into_amf(out_p, config_bytes=mtp_cfg_b, output_path=out_p)
    return metas


def convert_ling3_checkpoint(
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
    dense_kda_rco_q3: bool | None = None,
    use_rco: bool = True,
    manifest_path: pathlib.Path | None = None,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int | None = None,
    heat_profile: pathlib.Path | None = None,
    all_experts_cold: bool = False,
    mtp_output: pathlib.Path | None = None,
    mtp_config_output: pathlib.Path | None = None,
    mtp_repack_sme2: bool = False,
    mtp_eh_proj_bits: int | None = None,
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
) -> Ling3ConvertReport:
    """Convert `inclusionAI/Ling-3.1-flash-1019` (`bailing_hybrid` / `bailingmoe3`) + `1L` `MTP` to `.amf`."""
    in_p = pathlib.Path(input_path)
    out_p = pathlib.Path(output_path)
    tier_prof = resolve_ling3_tier_profile(target_tier)
    eff_model_id = (model_id or (in_p.name if in_p.is_dir() else out_p.stem))[:63]
    eff_dense_kda_q3 = (
        bool(dense_kda_rco_q3)
        if dense_kda_rco_q3 is not None
        else bool(tier_prof is not None and tier_prof.dense_kda_rco_q3)
    )
    eff_quant_profile = "gsq-rco" if (eff_dense_kda_q3 and quant_profile == "uniform") else quant_profile

    if in_p.is_file() and in_p.suffix.lower() == ".gguf":
        from aminfer_tools.gguf import pack_amf_from_gguf, read_gguf_header

        sh = read_gguf_header(in_p)
        arch = str(sh.kv.get("general.architecture") or "bailingmoe3")
        n_layers = int(sh.kv.get(f"{arch}.block_count", 73) or 73)
        nextn_n = int(sh.kv.get(f"{arch}.nextn_predict_layers", 1) or 0)
        bb_layers = max(1, n_layers - nextn_n) if n_layers > nextn_n > 0 else n_layers
        n_exp = int(sh.kv.get(f"{arch}.expert_count", 0) or 0)
        dense_cnt = int(sh.kv.get(f"{arch}.leading_dense_block_count", 4) or 4)
        mlp_types = ["dense" if i < dense_cnt else "sparse" for i in range(bb_layers)]
        cold_set = _select_ling3_cold_experts(
            n_layers=bb_layers,
            n_experts=n_exp,
            mlp_types=mlp_types,
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
            group_size=group_size,
            strict_vocab_check=strict_vocab_check,
        )
        mtp_metas_out: list[PackedTensorMeta] = []
        if mtp_output is not None and nextn_n > 0:
            mtp_metas_out = export_ling3_mtp(
                in_p,
                mtp_output,
                model_id=f"{eff_model_id[:55]}-mtp",
                dtype_mode=dtype_mode,
                cold_experts={(0, e) for (_, e) in cold_set} if cold_set else (),
                all_experts_cold=all_experts_cold,
                repack_sme2=mtp_repack_sme2,
                eh_proj_bits=mtp_eh_proj_bits,
                embed_metadata=embed_metadata,
                config_output=mtp_config_output,
                workers=workers,
            )
        return Ling3ConvertReport(
            model_type="bailing_hybrid",
            backbone_amf=out_p,
            backbone_metas=tuple(metas),
            mtp_amf=mtp_output if mtp_metas_out else None,
            mtp_metas=tuple(mtp_metas_out),
            cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
            dense_kda_rco_q3=False,
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
    eff_cfg = _effective_ling3_cfg(raw_cfg)
    raw_tensors = _collect_tensors(resolved.shard_paths)
    n_layers = int(eff_cfg.get("num_hidden_layers") or 72)
    n_exp = int(eff_cfg.get("num_experts") or eff_cfg.get("n_routed_experts") or eff_cfg.get("num_local_experts") or 0)
    mlp_types = resolve_ling3_mlp_layer_types(eff_cfg)

    if (
        tier_prof is not None
        and not tier_prof.allow_nvme_cold_experts
        and n_layers >= 48
        and n_exp >= 256
        and tier_prof.name.startswith("24gb-")
    ):
        raise HardwareTierError(
            f"target_tier={tier_prof.name!r} ({tier_prof.wired_budget_gib:.0f} GiB) cannot wire all {n_exp} experts"
        )

    cold_set = _select_ling3_cold_experts(
        n_layers=n_layers,
        n_experts=n_exp,
        mlp_types=mlp_types,
        tier_profile=tier_prof,
        cold_experts=cold_experts,
        cold_experts_per_layer=cold_experts_per_layer,
        heat_profile=heat_profile,
        all_experts_cold=all_experts_cold,
    )

    mtp_in_main = [k for k in raw_tensors if is_ling3_mtp_tensor(k, n_layers)]
    eff_target_bits = target_bits
    if tier_prof is not None and eff_quant_profile == "gsq-rco" and eff_target_bits is None and target_wired_gib is None:
        eff_target_bits = tier_prof.target_bits_default
    eff_fine_rungs = (tier_prof is not None and eff_quant_profile == "gsq-rco") if fine_rungs is None else fine_rungs

    runtime_cfg = build_ling3_config_json(
        raw_cfg,
        cold_experts_count=len(cold_set),
        dense_kda_rco_q3=eff_dense_kda_q3,
    )
    runtime_cfg_bytes = json.dumps(runtime_cfg, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    if config_output is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_bytes(runtime_cfg_bytes)

    with tempfile.TemporaryDirectory() as tmp_d:
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
            exclude_tensors=mtp_in_main,
            quant_profile=eff_quant_profile,
            target_bits=eff_target_bits,
            use_rco=use_rco,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=True,
            embed_metadata=False,
            config_path=tmp_cfg_p,
            imatrix_path=imatrix_path,
            enforce_backbone_floor=enforce_backbone_floor,
            max_q2_share=max_q2_share,
            allow_lowbit_q2_retranscode=allow_lowbit_q2_retranscode,
            fine_rungs=eff_fine_rungs,
            dense_kda_rco_q3=eff_dense_kda_q3,
            ling3_config=raw_cfg,
        )

    if embed_metadata:
        tcfg_dict = json.loads(tcfg_p.read_text(encoding="utf-8")) if tcfg_p is not None else {}
        tpl_text = tpl_p.read_text(encoding="utf-8") if tpl_p is not None else None
        ct_bytes: bytes | None = None
        if tpl_text or (isinstance(tcfg_dict, dict) and tcfg_dict.get("chat_template")):
            spec = compile_chat_template_spec(
                jinja_text=tpl_text,
                tokenizer_config=tcfg_dict if isinstance(tcfg_dict, dict) else {},
                model_config=runtime_cfg,
            )
            ct_bytes = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        metas = embed_metadata_into_amf(
            out_p,
            config_bytes=runtime_cfg_bytes,
            tokenizer_bytes=tok_p.read_bytes() if tok_p is not None else None,
            chat_template_bytes=ct_bytes,
            output_path=out_p,
        )

    mtp_metas_out: list[PackedTensorMeta] = []
    if mtp_output is not None and mtp_in_main:
        mtp_cold = {(0, e) for (li, e) in cold_set if li == n_layers}
        if not mtp_cold and cold_set:
            cold_exp_ids = sorted({e for (_, e) in cold_set})
            mtp_cold = {(0, e) for e in cold_exp_ids}
        mtp_metas_out = export_ling3_mtp(
            in_p,
            mtp_output,
            target_config=raw_cfg,
            model_id=f"{eff_model_id[:55]}-mtp",
            dtype_mode=dtype_mode,
            quant_profile=eff_quant_profile if eff_quant_profile != "gsq-rco" else "uniform",
            target_bits=eff_target_bits,
            use_rco=use_rco,
            cold_experts=mtp_cold,
            all_experts_cold=all_experts_cold,
            repack_sme2=mtp_repack_sme2,
            eh_proj_bits=mtp_eh_proj_bits,
            embed_metadata=embed_metadata,
            config_output=mtp_config_output,
            workers=workers,
        )

    return Ling3ConvertReport(
        model_type="bailing_hybrid",
        backbone_amf=out_p,
        backbone_metas=tuple(metas),
        mtp_amf=mtp_output if mtp_metas_out else None,
        mtp_metas=tuple(mtp_metas_out),
        cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
        dense_kda_rco_q3=eff_dense_kda_q3,
        config_json_path=config_output,
    )


# ---------------------------------------------------------------------------
# Deterministic Synthetic Generators (`synth-ling3`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ling3SynthConfig:
    """Geometry preset for deterministic `synth-ling3` (`tiny-ling3` / `tiny-ling3-512e`)."""

    preset: str = "tiny-ling3"
    model_id: str = "tiny-ling3"
    num_hidden_layers: int = 8
    layer_group_size: int = 4
    first_k_dense_replace: int = 1
    hidden_size: int = 128
    intermediate_size: int = 256
    moe_intermediate_size: int = 64
    moe_shared_expert_intermediate_size: int = 64
    vocab_size: int = 320
    num_attention_heads: int = 4
    head_dim: int = 32
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    short_conv_kernel_size: int = 4
    kda_lower_bound: float = -5.0
    gated_attention_granularity: str = "head_wise"
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    n_group: int = 2
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    cold_experts_per_layer: int = 2
    include_mtp: bool = True
    mtp_repack_sme2: bool = False
    mtp_eh_proj_bits: int | None = None
    emit_rco_q3_variant: bool = False
    rco_q3_target_bits: float = 3.50
    expert_dtype: str = "bf16"
    dtype: str = "bf16"

    def validate(self) -> None:
        if self.num_hidden_layers < 2:
            raise ValueError("num_hidden_layers must be >= 2")
        if self.hidden_size % 64 != 0 or self.intermediate_size % 64 != 0 or self.moe_intermediate_size % 32 != 0:
            raise ValueError("hidden/intermediate/moe_intermediate must be multiples of 64/64/32")
        if self.n_routed_experts < self.num_experts_per_tok:
            raise ValueError("n_routed_experts must be >= num_experts_per_tok")
        if self.n_routed_experts % max(self.n_group, 1) != 0:
            raise ValueError("n_routed_experts must be divisible by n_group")
        if self.expert_dtype not in ("mxfp4", "bf16", "f16"):
            raise ValueError(f"expert_dtype must be mxfp4/bf16/f16, got {self.expert_dtype!r}")
        if self.dtype not in ("bf16", "f16"):
            raise ValueError(f"dtype must be bf16/f16, got {self.dtype!r}")


LING3_SYNTH_PRESETS: dict[str, Ling3SynthConfig] = {
    "tiny-ling3": Ling3SynthConfig(),
    "tiny-ling3-512e": Ling3SynthConfig(
        preset="tiny-ling3-512e",
        model_id="tiny-ling3-512e",
        n_routed_experts=512,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        cold_experts_per_layer=384,
    ),
    "tiny-ling3-rco-q3": Ling3SynthConfig(
        preset="tiny-ling3-rco-q3",
        model_id="tiny-ling3-rco-q3",
        emit_rco_q3_variant=True,
    ),
}


@dataclass(frozen=True)
class Ling3SynthBundle:
    """Outputs written by `generate_synth_ling3_bundle`."""

    output_dir: pathlib.Path
    model_type: str
    backbone_amf: pathlib.Path
    backbone_metas: tuple[PackedTensorMeta, ...]
    rco_q3_amf: pathlib.Path | None
    rco_q3_metas: tuple[PackedTensorMeta, ...]
    mtp_amf: pathlib.Path | None
    mtp_metas: tuple[PackedTensorMeta, ...]
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    total_bytes: int


def generate_synth_ling3_bundle(
    output_dir: pathlib.Path,
    *,
    config: Ling3SynthConfig | None = None,
    preset: str | None = None,
    num_layers: int | None = None,
    hidden_size: int | None = None,
    num_experts: int | None = None,
    dtype: str | None = None,
    cold_experts_per_layer: int | None = None,
    with_mtp: bool | None = None,
    mtp_repack_sme2: bool | None = None,
    mtp_eh_proj_bits: int | None = None,
    with_rco_q3: bool | None = None,
    rco_q3_target_bits: float | None = None,
    model_id: str | None = None,
    seed: int = 42,
    embed_metadata: bool = True,
) -> Ling3SynthBundle:
    """Synthesize a deterministic, 16 KiB-aligned `Ling-3.1` (`bailing_hybrid`) + optional `RCO-Q3` & `1L` `MTP` bundle."""
    base_cfg = config if config is not None else LING3_SYNTH_PRESETS[preset or "tiny-ling3"]
    overrides: dict[str, Any] = {}
    if num_layers is not None:
        overrides["num_hidden_layers"] = num_layers
    if hidden_size is not None:
        overrides["hidden_size"] = hidden_size
    if num_experts is not None:
        overrides["n_routed_experts"] = num_experts
        if num_experts < base_cfg.n_group:
            overrides["n_group"] = 1
            overrides["topk_group"] = 1
    if dtype is not None:
        overrides["dtype"] = dtype
    if cold_experts_per_layer is not None:
        overrides["cold_experts_per_layer"] = cold_experts_per_layer
    if with_mtp is not None:
        overrides["include_mtp"] = with_mtp
    if mtp_repack_sme2 is not None:
        overrides["mtp_repack_sme2"] = mtp_repack_sme2
    if mtp_eh_proj_bits is not None:
        overrides["mtp_eh_proj_bits"] = mtp_eh_proj_bits
    if with_rco_q3 is not None:
        overrides["emit_rco_q3_variant"] = with_rco_q3
    if rco_q3_target_bits is not None:
        overrides["rco_q3_target_bits"] = rco_q3_target_bits
    if model_id is not None:
        overrides["model_id"] = model_id
    cfg = replace(base_cfg, **overrides) if overrides else base_cfg
    cfg.validate()

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    moe_inter = cfg.moe_intermediate_size
    sh_inter = cfg.moe_shared_expert_intermediate_size
    vocab = cfg.vocab_size
    fdt = DType.BF16 if cfg.dtype == "bf16" else DType.F16

    def _fbytes(vals: Sequence[float]) -> bytes:
        return f32_values_to_bf16_bytes(vals) if fdt == DType.BF16 else f32_values_to_f16_bytes(vals)

    def _add(
        recs: list[tuple],
        name: str,
        kind: TensorKind,
        dt: DType,
        layer: int,
        expert: int,
        shape: tuple[int, ...],
        data: bytes,
        qb: int = 0,
        gs: int = 0,
    ) -> None:
        assert len(name.encode("utf-8")) <= 47, f"name exceeds 47B: {name}"
        recs.append((name, kind, dt, qb, gs, layer, expert, shape, data, b""))

    raw_cfg: dict[str, Any] = {
        "model_type": "bailing_hybrid",
        "architectures": ["BailingMoeV3ForCausalLM"],
        "hidden_size": h,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "moe_shared_expert_intermediate_size": sh_inter,
        "num_hidden_layers": cfg.num_hidden_layers,
        "layer_group_size": cfg.layer_group_size,
        "vocab_size": vocab,
        "first_k_dense_replace": cfg.first_k_dense_replace,
        "no_kda_lora": True,
        "kda_safe_gate": True,
        "kda_lower_bound": cfg.kda_lower_bound,
        "linear_silu": True,
        "short_conv_kernel_size": cfg.short_conv_kernel_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_attention_heads,
        "head_dim": cfg.head_dim,
        "q_lora_rank": cfg.q_lora_rank,
        "kv_lora_rank": cfg.kv_lora_rank,
        "qk_nope_head_dim": cfg.qk_nope_head_dim,
        "qk_rope_head_dim": cfg.qk_rope_head_dim,
        "v_head_dim": cfg.v_head_dim,
        "gated_attention_proj_granularity_type": cfg.gated_attention_granularity,
        "rope_interleave": True,
        "rope_theta": 6000000.0,
        "num_experts": cfg.n_routed_experts,
        "num_shared_experts": cfg.n_shared_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "n_group": cfg.n_group,
        "topk_group": cfg.topk_group,
        "routed_scaling_factor": cfg.routed_scaling_factor,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "num_nextn_predict_layers": 1 if cfg.include_mtp else 0,
        "mtp_use_kda": False,
        "expert_swiglu_limit_list": [0.0] * max(0, cfg.num_hidden_layers - 1) + [10.0, 10.0],
        "share_expert_swiglu_limit_list": [0.0] * max(0, cfg.num_hidden_layers - 1) + [10.0, 10.0],
    }
    ltypes = resolve_ling3_layer_types(raw_cfg)
    mlp_types = resolve_ling3_mlp_layer_types(raw_cfg)
    cold_start = max(0, cfg.n_routed_experts - cfg.cold_experts_per_layer)
    cold_set = {
        (l_idx, e_idx)
        for l_idx in range(cfg.num_hidden_layers)
        if mlp_types[l_idx] == "sparse"
        for e_idx in range(cold_start, cfg.n_routed_experts)
    }

    rt_cfg = build_ling3_config_json(raw_cfg, cold_experts_count=len(cold_set), dense_kda_rco_q3=False)
    rt_cfg_b = json.dumps(rt_cfg, indent=2).encode("utf-8") + b"\n"
    tok_b = _synth_tokenizer_bytes(vocab)
    ct_b = _synth_chat_template_bytes(rt_cfg)
    (out_dir / "config.json").write_bytes(rt_cfg_b)
    (out_dir / "tokenizer.json").write_bytes(tok_b)

    def _emit_gated_mla(recs: list[tuple], l_idx: int) -> None:
        qh = cfg.num_attention_heads
        qk_hd = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        g_rows = qh if cfg.gated_attention_granularity == "head_wise" else qh * cfg.v_head_dim
        _add(recs, f"layers.{l_idx}.self_attn.q_a_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.q_lora_rank, h), _fbytes(_det_f32(rng, cfg.q_lora_rank * h)))
        _add(recs, f"layers.{l_idx}.self_attn.q_a_layernorm.weight", TensorKind.NORM, fdt, l_idx, NO_EXPERT, (cfg.q_lora_rank,), _fbytes(_det_f32(rng, cfg.q_lora_rank, std=0.02, mean=1.0)))
        _add(recs, f"layers.{l_idx}.self_attn.q_b_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (qh * qk_hd, cfg.q_lora_rank), _fbytes(_det_f32(rng, qh * qk_hd * cfg.q_lora_rank)))
        _add(recs, f"layers.{l_idx}.self_attn.kv_a_proj_with_mqa.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.kv_lora_rank + cfg.qk_rope_head_dim, h), _fbytes(_det_f32(rng, (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * h)))
        _add(recs, f"layers.{l_idx}.self_attn.kv_a_layernorm.weight", TensorKind.NORM, fdt, l_idx, NO_EXPERT, (cfg.kv_lora_rank,), _fbytes(_det_f32(rng, cfg.kv_lora_rank, std=0.02, mean=1.0)))
        _add(recs, f"layers.{l_idx}.self_attn.kv_b_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (qh * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank), _fbytes(_det_f32(rng, qh * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank)))
        _add(recs, f"layers.{l_idx}.self_attn.g_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (g_rows, h), _fbytes(_det_f32(rng, g_rows * h)))
        _add(recs, f"layers.{l_idx}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (h, qh * cfg.v_head_dim), _fbytes(_det_f32(rng, h * qh * cfg.v_head_dim)))

    def _emit_moe_or_dense(recs: list[tuple], l_idx: int, is_sparse: bool) -> None:
        if not is_sparse:
            _add(recs, f"layers.{l_idx}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)))
            _add(recs, f"layers.{l_idx}.mlp.up_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)))
            _add(recs, f"layers.{l_idx}.mlp.down_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (h, inter), _fbytes(_det_f32(rng, h * inter)))
            return
        _add(recs, f"layers.{l_idx}.mlp.gate.weight", TensorKind.ROUTER, fdt, l_idx, NO_EXPERT, (cfg.n_routed_experts, h), _fbytes(_det_f32(rng, cfg.n_routed_experts * h)))
        _add(recs, f"layers.{l_idx}.mlp.gate.bias", TensorKind.ROUTER, DType.F32, l_idx, NO_EXPERT, (cfg.n_routed_experts,), f32_values_to_f32_bytes(_det_f32(rng, cfg.n_routed_experts, std=0.05)))
        total_sh = sh_inter * cfg.n_shared_experts
        _add(recs, f"layers.{l_idx}.mlp.shared_experts.gate_proj.weight", TensorKind.SHARED_EXPERT, fdt, l_idx, NO_EXPERT, (total_sh, h), _fbytes(_det_f32(rng, total_sh * h)))
        _add(recs, f"layers.{l_idx}.mlp.shared_experts.up_proj.weight", TensorKind.SHARED_EXPERT, fdt, l_idx, NO_EXPERT, (total_sh, h), _fbytes(_det_f32(rng, total_sh * h)))
        _add(recs, f"layers.{l_idx}.mlp.shared_experts.down_proj.weight", TensorKind.SHARED_EXPERT, fdt, l_idx, NO_EXPERT, (h, total_sh), _fbytes(_det_f32(rng, h * total_sh)))
        for e in range(cfg.n_routed_experts):
            ek = TensorKind.COLD_EXPERT if (l_idx, e) in cold_set else TensorKind.HOT_EXPERT
            for proj, r_sz, c_sz in (("gate_proj", moe_inter, h), ("up_proj", moe_inter, h), ("down_proj", h, moe_inter)):
                vals = _det_f32(rng, r_sz * c_sz)
                if cfg.expert_dtype == "mxfp4":
                    _add(recs, f"layers.{l_idx}.mlp.experts.{e}.{proj}.weight", ek, DType.GGUF_MXFP4, l_idx, e, (r_sz, c_sz), _encode_mxfp4_synthetic(vals, r_sz, c_sz), qb=4, gs=32)
                else:
                    _add(recs, f"layers.{l_idx}.mlp.experts.{e}.{proj}.weight", ek, fdt, l_idx, e, (r_sz, c_sz), _fbytes(vals))

    bb_recs: list[tuple] = []
    _add(bb_recs, "embed_tokens.weight", TensorKind.EMBEDDING, fdt, NO_LAYER, NO_EXPERT, (vocab, h), _fbytes(_det_f32(rng, vocab * h)))
    _add(bb_recs, "norm.weight", TensorKind.NORM, fdt, NO_LAYER, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
    _add(bb_recs, "lm_head.weight", TensorKind.LM_HEAD, fdt, NO_LAYER, NO_EXPERT, (vocab, h), _fbytes(_det_f32(rng, vocab * h)))

    for l_idx in range(cfg.num_hidden_layers):
        _add(bb_recs, f"layers.{l_idx}.input_layernorm.weight", TensorKind.NORM, fdt, l_idx, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(bb_recs, f"layers.{l_idx}.post_attention_layernorm.weight", TensorKind.NORM, fdt, l_idx, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        if ltypes[l_idx] == "linear_attention":
            lh = cfg.num_attention_heads
            lhd = cfg.head_dim
            qkv_dim = lh * lhd
            kern = cfg.short_conv_kernel_size
            for proj in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
                _add(bb_recs, f"layers.{l_idx}.linear_attn.{proj}.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (qkv_dim, h), _fbytes(_det_f32(rng, qkv_dim * h)))
            for conv_nm in ("q_conv1d", "k_conv1d", "v_conv1d"):
                _add(bb_recs, f"layers.{l_idx}.linear_attn.{conv_nm}.weight", TensorKind.DENSE, DType.F32, l_idx, NO_EXPERT, (qkv_dim, kern), f32_values_to_f32_bytes(_det_f32(rng, qkv_dim * kern)))
            _add(bb_recs, f"layers.{l_idx}.linear_attn.b_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (lh, h), _fbytes(_det_f32(rng, lh * h)))
            _add(bb_recs, f"layers.{l_idx}.linear_attn.o_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (h, qkv_dim), _fbytes(_det_f32(rng, h * qkv_dim)))
            _add(bb_recs, f"layers.{l_idx}.linear_attn.o_norm.weight", TensorKind.NORM, fdt, l_idx, NO_EXPERT, (lhd,), _fbytes(_det_f32(rng, lhd, std=0.02, mean=1.0)))
            _add(bb_recs, f"layers.{l_idx}.linear_attn.A_log", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (lh,), f32_values_to_f32_bytes(_det_f32(rng, lh, std=0.1, mean=-2.0)))
            _add(bb_recs, f"layers.{l_idx}.linear_attn.dt_bias", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (qkv_dim,), f32_values_to_f32_bytes(_det_f32(rng, qkv_dim, std=0.1)))
        else:
            _emit_gated_mla(bb_recs, l_idx)
        _emit_moe_or_dense(bb_recs, l_idx, mlp_types[l_idx] == "sparse")

    if embed_metadata:
        bb_recs.extend([
            make_metadata_record(META_CONFIG_JSON, rt_cfg_b),
            make_metadata_record(META_TOKENIZER_JSON, tok_b),
            make_metadata_record(META_CHAT_TEMPLATE_JSON, ct_b),
        ])

    bb_amf = out_dir / "model.amf"
    bb_metas = write_amf(bb_amf, cfg.model_id[:63], bb_recs)
    total_bytes = bb_amf.stat().st_size

    rco_q3_amf: pathlib.Path | None = None
    rco_q3_metas: list[PackedTensorMeta] = []
    if cfg.emit_rco_q3_variant:
        from aminfer_tools.gsq_rco import gsq_convert

        rco_cfg = build_ling3_config_json(raw_cfg, cold_experts_count=len(cold_set), dense_kda_rco_q3=True)
        rco_cfg_b = json.dumps(rco_cfg, indent=2).encode("utf-8") + b"\n"
        (out_dir / "config-rco-q3.json").write_bytes(rco_cfg_b)
        rco_q3_amf = out_dir / "model-rco-q3.amf"
        gsq_rep = gsq_convert(
            bb_amf,
            rco_q3_amf,
            target_bits=cfg.rco_q3_target_bits,
            model_id=f"{cfg.model_id[:55]}-q3kda",
            group_size=64,
            dtype_mode=cfg.dtype,
            cold_experts=cold_set,
            use_rco=True,
            enforce_backbone_floor=True,
            dense_kda_rco_q3=True,
            meta_blobs=(
                [
                    (META_CONFIG_JSON, rco_cfg_b),
                    (META_TOKENIZER_JSON, tok_b),
                    (META_CHAT_TEMPLATE_JSON, ct_b),
                ]
                if embed_metadata
                else ()
            ),
        )
        rco_q3_metas = list(gsq_rep.metas)
        total_bytes += rco_q3_amf.stat().st_size

    mtp_amf: pathlib.Path | None = None
    mtp_metas: list[PackedTensorMeta] = []
    if cfg.include_mtp:
        mtp_cfg = build_ling3_mtp_config_dict(raw_cfg)
        mtp_cfg_b = json.dumps(mtp_cfg, indent=2).encode("utf-8") + b"\n"
        (out_dir / "mtp_config.json").write_bytes(mtp_cfg_b)
        mtp_recs: list[tuple] = []
        _add(mtp_recs, "layers.0.enorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(mtp_recs, "layers.0.hnorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(mtp_recs, "layers.0.eh_proj.weight", TensorKind.DENSE, fdt, 0, NO_EXPERT, (h, 2 * h), _fbytes(_det_f32(rng, h * 2 * h)))
        _add(mtp_recs, "layers.0.final_layernorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(mtp_recs, "layers.0.input_layernorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(mtp_recs, "layers.0.post_attention_layernorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _emit_gated_mla(mtp_recs, 0)
        _emit_moe_or_dense(mtp_recs, 0, True)
        if embed_metadata:
            mtp_recs.append(make_metadata_record(META_CONFIG_JSON, mtp_cfg_b))
        mtp_amf = out_dir / "mtp.amf"
        mtp_metas = write_amf(mtp_amf, f"{cfg.model_id[:55]}-mtp", mtp_recs)
        if cfg.mtp_eh_proj_bits is not None:
            mtp_metas = _repack_quantize_eh_proj_sme2(
                mtp_amf,
                bits=cfg.mtp_eh_proj_bits,
                use_rco=True,
                repack_sme2=cfg.mtp_repack_sme2,
            )
        elif cfg.mtp_repack_sme2:
            mtp_metas = repack_amf_sme2(mtp_amf, mtp_amf, targets="eh-proj")
        total_bytes += mtp_amf.stat().st_size

    return Ling3SynthBundle(
        output_dir=out_dir,
        model_type="bailing_hybrid",
        backbone_amf=bb_amf,
        backbone_metas=tuple(bb_metas),
        rco_q3_amf=rco_q3_amf,
        rco_q3_metas=tuple(rco_q3_metas),
        mtp_amf=mtp_amf,
        mtp_metas=tuple(mtp_metas),
        config_path=out_dir / "config.json",
        tokenizer_path=out_dir / "tokenizer.json",
        total_bytes=total_bytes,
    )
