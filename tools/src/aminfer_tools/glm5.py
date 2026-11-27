"""W28/W26 GLM-5 family (`glm_moe_dsa` GLM-5.2/5.3, `glm5_next` GLM-5.3-Flash, `24L` ViT, `1L` MTP, `DFlash2`).

Locked Canonical `.amf` v0 Tensor Naming & `config.json` Contract (`#514`, M9.2 Wave 2 `m92-g27-kernels` & `m92-d15-glm-spec`)
=============================================================================================================================
Every tensor name stored in `.amf` is strictly `<= 47` UTF-8 bytes (`TensorEntry.name` is `[48]u8`
NUL-terminated; `long_backbone_names == 0`).

1. `GLM-5.2` & `GLM-5.3` (`glm_moe_dsa`, `753B`)
------------------------------------------------
- `config.json`: `"model_type": "glm_moe_dsa"`, `hidden_size=6144`, `intermediate_size=12288`, `moe_intermediate_size=2048`,
  `num_hidden_layers=78` (`L0..L2` dense via `first_k_dense_replace=3`, `L3..L77` MoE `n_routed_experts=256`,
  `num_experts_per_tok=8`, `n_shared_experts=1`, `scoring_func="sigmoid"`, `norm_topk_prob=true`,
  `routed_scaling_factor=2.5`), `num_attention_heads=64`, `num_key_value_heads=64`, `q_lora_rank=2048`,
  `kv_lora_rank=512`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `qk_head_dim=256`, `v_head_dim=256`,
  `rope_interleave=true`, `rope_theta=8000000`, `index_n_heads=32`, `index_head_dim=128`, `index_topk=2048`,
  `index_topk_freq=4`, `index_skip_topk_offset=3`, `indexer_types` (`21` `"full"` + `57` `"shared"`),
  `index_share_for_mtp_iteration=true`, `num_nextn_predict_layers=1`.
- `.amf` tensors (`<= 46 B`):
  - `embed_tokens.weight`, `norm.weight`, `lm_head.weight`
  - `layers.{L}.input_layernorm.weight`, `layers.{L}.post_attention_layernorm.weight`
  - `layers.{L}.self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj}.weight`
  - `"full"` DSA indexer layers only: `layers.{L}.attn.idx.{wq_b,wk,k_norm,weights_proj}.weight` + `layers.{L}.attn.idx.k_norm.bias`
  - Dense `L0..L2`: `layers.{L}.mlp.{gate_proj,up_proj,down_proj}.weight`
  - MoE `L3..L77`: `layers.{L}.mlp.gate.weight`, `layers.{L}.mlp.gate.bias` (`F32`, from `e_score_correction_bias`),
    `layers.{L}.mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`,
    `layers.{L}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`

2. `GLM-5.3-Flash` (`glm5_next`, `320B`)
----------------------------------------
- `config.json`: `"model_type": "glm5_next"` (or `"glm5_next_text"`), `hidden_size=4096`, `intermediate_size=12288`,
  `moe_intermediate_size=2048`, `num_hidden_layers=45` (`L0..L2` dense, `L3..L44` MoE `288E` top-8 + `1` shared,
  `swiglu_limit=10.0` on dense, shared, and routed experts), `layer_types` (`34` `"linear_attention"` `L%4!=3` +
  `11` `"indexed_attention"` `L%4==3`), `hc_mult=4`, `hc_sinkhorn_iters=20`, `hc_eps=1e-6`, `linear_num_heads=64`,
  `linear_head_dim=128`, `linear_conv_kernel_dim=4`, `linear_lower_bound=-5.0`, `q_lora_rank=1536`, `kv_lora_rank=512`,
  `qk_nope_head_dim=256`, `qk_rope_head_dim=0`, `v_head_dim=256`, `index_n_heads=32`, `index_head_dim=128`,
  `index_topk=2048`, `index_kpool=4`, `index_kpool_always_select_tail=true`, `num_nextn_predict_layers=1`.
- `.amf` tensors (`<= 46 B`):
  - `embed_tokens.weight`, `norm.weight`, `lm_head.weight` (`hc_head` is unweighted `mean(dim=2)`)
  - `layers.{L}.hc_{attn,ffn}_{fn,base,scale}` (`base`/`scale` promoted to `F32`)
  - Linear KDA layers (`L%4!=3`): `layers.{L}.linear_attn.{q_proj,k_proj,v_proj,b_proj,f_a_proj,f_b_proj,g_a_proj,g_b_proj,o_proj,o_norm}.weight`,
    `layers.{L}.linear_attn.{q_conv1d,k_conv1d,v_conv1d}.weight` (`[8192, 4]` `F32`),
    `layers.{L}.linear_attn.{A_log,dt_bias}` (`F32`)
  - NoPE MLA + `index_kpool=4` layers (`L%4==3`): `layers.{L}.self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj}.weight`,
    `layers.{L}.attn.idx.{wq_b.weight,wk.weight,k_norm.weight,k_norm.bias,weights_proj.weight,kpool_ape,kpool_gate}`
  - `W20` `24L` ViT sidecar (`vision.amf`): `visual.*` (`<= 41 B`)

3. `1L` MTP (`mtp.amf`) & `DFlash2` (`dflash2.amf`)
---------------------------------------------------
- Standalone `1L` `mtp.amf`: re-indexes `layers.{78|45}.*` to `layers.0.*`, maps `layers.0.shared_head.norm.weight` ->
  `shared_head.norm.weight`.
- `DFlash2` (`draft_kind="dflash2"`, `block_size in {8, 16}`, `5L`):
  `fc.weight`, `hidden_norm.weight`, `norm.weight`, `candidate_selector.{hidden_projection.weight,predecessor_codebook,successor_codebook}`,
  `layers.{l}.{input_layernorm,post_attention_layernorm}.weight`, `layers.{l}.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight`,
  `layers.{l}.{attn_conv,mlp_conv}.{base_kernel,kernel_projection.weight}`, `layers.{l}.mlp.{gate_proj,up_proj,down_proj}.weight`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
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
    _collect_tensors,
    embed_metadata_into_amf,
    f32_to_bf16_bytes as f32_values_to_bf16_bytes,
    f32_values_to_f16_bytes,
    is_vision_tensor,
    make_metadata_record,
    normalize_vision_name,
    pack_amf_from_safetensors,
    resolve_checkpoint,
    resolve_vision_mode,
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
# Tier Profiles (`24gb-m4`, `128gb-max`, `256gb-studio`, `512gb-ultra`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Glm5TierProfile:
    """Hardware-tier wired RAM & cold-expert placement budget for GLM-5."""

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


GLM5_TIER_PROFILES: dict[str, Glm5TierProfile] = {
    "24gb-m4": Glm5TierProfile(
        name="24gb-m4",
        host_memory_gib=24.0,
        wired_budget_gib=18.5,
        always_active_reserve_gib=9.5,
        hot_expert_budget_gib=9.0,
        target_bits_default=3.25,
        allow_nvme_cold_experts=True,
        recommended_model="GLM-5.3-Flash (NVMe cold experts; strip/sidecar 24L ViT)",
    ),
    "128gb-max": Glm5TierProfile(
        name="128gb-max",
        host_memory_gib=128.0,
        wired_budget_gib=104.0,
        always_active_reserve_gib=10.0,
        hot_expert_budget_gib=94.0,
        target_bits_default=3.50,
        allow_nvme_cold_experts=False,
        recommended_model="GLM-5.3-Flash (~94 GiB full-wired GSQ-RCO)",
    ),
    "256gb-studio": Glm5TierProfile(
        name="256gb-studio",
        host_memory_gib=256.0,
        wired_budget_gib=210.0,
        always_active_reserve_gib=14.0,
        hot_expert_budget_gib=190.0,
        target_bits_default=4.0,
        allow_nvme_cold_experts=False,
        recommended_model="GLM-5.3-Flash (native 4-bit / NVFP4 full-wired)",
    ),
    "512gb-ultra": Glm5TierProfile(
        name="512gb-ultra",
        host_memory_gib=512.0,
        wired_budget_gib=420.0,
        always_active_reserve_gib=24.0,
        hot_expert_budget_gib=280.0,
        target_bits_default=3.25,
        allow_nvme_cold_experts=False,
        recommended_model="GLM-5.2 / GLM-5.3 (~280 GiB single-512GB Ultra full-wired GSQ-RCO)",
    ),
}


def resolve_glm5_tier_profile(tier: str | None) -> Glm5TierProfile | None:
    if tier is None:
        return None
    key = tier.strip().lower()
    if key not in GLM5_TIER_PROFILES:
        raise ValueError(f"unknown target_tier {tier!r}; expected one of {sorted(GLM5_TIER_PROFILES)}")
    return GLM5_TIER_PROFILES[key]


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
_MTP_KEY_RE = re.compile(r"^(?:model\.)?(?:language_model\.)?(?:mtp|nextn)\.")
_LINEAR_ATTN_LEAF_HINTS = (
    ".self_attn.A_log",
    ".self_attn.dt_bias",
    ".self_attn.b_proj.",
    ".self_attn.f_a_proj.",
    ".self_attn.f_b_proj.",
    ".self_attn.g_a_proj.",
    ".self_attn.g_b_proj.",
    ".self_attn.o_norm.",
    ".self_attn.q_conv1d.",
    ".self_attn.k_conv1d.",
    ".self_attn.v_conv1d.",
    ".self_attn.conv1d.",
    ".self_attn.fused_qkvbfg_a_proj.",
)
_GLM5_PREFIXES = (
    "model.language_model.",
    "model.llm.",
    "language_model.model.",
    "language_model.",
    "model.",
)


def _effective_glm5_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
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


def is_glm_moe_dsa_config(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return False
    mt = str(cfg.get("model_type") or "").lower()
    if mt in ("glm_moe_dsa", "glm-dsa", "glm_dsa"):
        return True
    archs = [str(a).lower() for a in (cfg.get("architectures") or [])]
    if any("glmmoedsa" in a for a in archs):
        return True
    tc = cfg.get("text_config")
    return isinstance(tc, Mapping) and str(tc.get("model_type") or "").lower() == "glm_moe_dsa"


def is_glm5_next_config(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return False
    mt = str(cfg.get("model_type") or "").lower()
    if mt in ("glm5_next", "glm5_next_text", "glm5-next"):
        return True
    archs = [str(a).lower() for a in (cfg.get("architectures") or [])]
    if any("glm5next" in a for a in archs):
        return True
    tc = cfg.get("text_config")
    return isinstance(tc, Mapping) and str(tc.get("model_type") or "").lower() in ("glm5_next", "glm5_next_text")


def is_glm5_dflash2_config(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return False
    archs = [str(a).lower() for a in (cfg.get("architectures") or [])]
    if any("dflash2" in a for a in archs):
        return True
    if str(cfg.get("draft_kind") or "").lower() == "dflash2":
        return True
    dc = cfg.get("dflash_config")
    if isinstance(dc, Mapping) and any(k in dc for k in ("selector_rank", "conv_kernel_size", "conv_group_size")):
        return True
    return False


def is_glm5_config(cfg: Mapping[str, Any] | None) -> bool:
    return is_glm_moe_dsa_config(cfg) or is_glm5_next_config(cfg)


def resolve_glm_moe_dsa_indexer_types(cfg: Mapping[str, Any]) -> list[str]:
    eff = _effective_glm5_cfg(cfg)
    n_layers = int(eff.get("num_hidden_layers") or 0)
    raw = eff.get("indexer_types")
    if isinstance(raw, list) and len(raw) >= n_layers and n_layers > 0:
        return [str(x).lower() for x in raw[:n_layers]]
    pat = eff.get("index_topk_pattern")
    if isinstance(pat, str) and len(pat) >= n_layers and n_layers > 0:
        return ["full" if c == "F" else "shared" for c in pat[:n_layers]]
    if isinstance(pat, list) and len(pat) >= n_layers and n_layers > 0:
        return [str(x).lower() for x in pat[:n_layers]]
    freq = max(int(eff.get("index_topk_freq") or 1), 1)
    offset = int(eff.get("index_skip_topk_offset") if eff.get("index_skip_topk_offset") is not None else 2)
    return ["full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(n_layers)]


def resolve_glm5_next_layer_types(cfg: Mapping[str, Any]) -> list[str]:
    eff = _effective_glm5_cfg(cfg)
    n_layers = int(eff.get("num_hidden_layers") or 0)
    raw = eff.get("layer_types")
    if isinstance(raw, list) and len(raw) >= n_layers and n_layers > 0:
        return [
            "indexed_attention" if str(x) in ("full_attention", "deepseek_sparse_attention", "indexed_attention") else str(x)
            for x in raw[:n_layers]
        ]
    la_cfg = eff.get("linear_attn_config")
    if isinstance(la_cfg, Mapping) and isinstance(la_cfg.get("kda_layers"), list):
        kda_set = {int(x) for x in la_cfg["kda_layers"]}
        return ["linear_attention" if i in kda_set else "indexed_attention" for i in range(n_layers)]
    return ["linear_attention" if (i % 4 != 3) else "indexed_attention" for i in range(n_layers)]


def resolve_glm5_mlp_layer_types(cfg: Mapping[str, Any]) -> list[str]:
    eff = _effective_glm5_cfg(cfg)
    n_layers = int(eff.get("num_hidden_layers") or 0)
    raw = eff.get("mlp_layer_types")
    if isinstance(raw, list) and len(raw) >= n_layers and n_layers > 0:
        return [str(x).lower() for x in raw[:n_layers]]
    first_k = int(eff.get("first_k_dense_replace") or 0)
    n_dense = min(max(first_k, 0), n_layers)
    return ["dense"] * n_dense + ["sparse"] * (n_layers - n_dense)


def is_glm5_mtp_tensor(raw_name: str, backbone_num_layers: int | None = None) -> bool:
    if _MTP_KEY_RE.match(raw_name) is not None:
        return True
    if backbone_num_layers is not None and backbone_num_layers > 0:
        s = raw_name
        for pref in _GLM5_PREFIXES:
            if s.startswith(pref):
                s = s[len(pref) :]
                break
        if s.startswith(f"layers.{backbone_num_layers}."):
            return True
    return False


def _strip_glm5_quant_suffix(raw_name: str) -> str:
    if raw_name.endswith(".weight_packed"):
        return raw_name[: -len(".weight_packed")] + ".weight"
    for suf in _STRIP_QUANT_SUFFIXES:
        if raw_name.endswith(suf):
            return raw_name[: -len(suf)]
    return raw_name


def normalize_glm5_tensor_name(
    raw_name: str,
    *,
    model_type: str = "glm_moe_dsa",
    layer_types: Sequence[str] | None = None,
    mtp_source_layer: int | None = None,
    mtp_standalone: bool = False,
) -> str:
    """Normalize a `glm_moe_dsa` / `glm5_next` / `1L` MTP tensor name to canonical `<= 47 B` `.amf` form."""
    if is_vision_tensor(raw_name):
        return normalize_vision_name(raw_name)
    name = _strip_glm5_quant_suffix(raw_name)
    for pref in _GLM5_PREFIXES:
        if name.startswith(pref):
            name = name[len(pref) :]
            break
    if mtp_standalone:
        if mtp_source_layer is not None and name.startswith(f"layers.{mtp_source_layer}."):
            name = "layers.0." + name[len(f"layers.{mtp_source_layer}.") :]
        elif name.startswith("mtp.layers."):
            name = name[len("mtp.") :]
        elif name.startswith("mtp."):
            name = name[len("mtp.") :]
        if name == "layers.0.shared_head.norm.weight":
            return "shared_head.norm.weight"
    name = name.replace(".mlp.gate.e_score_correction_bias", ".mlp.gate.bias")
    name = name.replace(".self_attn.indexer.index_kpool_compress_", ".attn.idx.kpool_")
    name = name.replace(".self_attn.indexer.", ".attn.idx.")

    is_next = "next" in model_type.lower() or layer_types is not None
    if is_next and ".self_attn." in name:
        m_l = _LAYER_IDX_RE.match(name)
        is_linear = False
        if m_l is not None and layer_types is not None and not mtp_standalone:
            li = int(m_l.group(1))
            if 0 <= li < len(layer_types) and layer_types[li] == "linear_attention":
                is_linear = True
        if not is_linear and any(tok in name for tok in _LINEAR_ATTN_LEAF_HINTS):
            is_linear = True
        if is_linear:
            name = name.replace(".self_attn.", ".linear_attn.")

    for src_seg, dst_seg in (
        (".self_attn.", ".attn."),
        (".shared_experts.", ".shared_exp."),
        (".input_layernorm.", ".in_norm."),
        (".post_attention_layernorm.", ".post_norm."),
    ):
        if len(name.encode("utf-8")) <= 47:
            break
        name = name.replace(src_seg, dst_seg)
    enc = name.encode("utf-8")
    if len(enc) == 0 or len(enc) > 47:
        raise ValueError(f"AMF tensor name must be 1..47 UTF-8 bytes, got {len(enc)}: {name!r}")
    return name


def normalize_glm5_dflash2_tensor_name(raw_name: str) -> str:
    """Normalize a `DFlash2` draft tensor name to canonical `<= 47 B` `.amf` form."""
    name = _strip_glm5_quant_suffix(raw_name)
    for pref in ("model.", "dflash2.", "dflash."):
        if name.startswith(pref):
            name = name[len(pref) :]
            break
    exact_map = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "hidden_norm.weight",
        "enc.output_norm.weight": "hidden_norm.weight",
        "norm.weight": "norm.weight",
        "output_norm.weight": "norm.weight",
        "token_embd.weight": "embed_tokens.weight",
        "embed_tokens.weight": "embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "lm_head.weight": "lm_head.weight",
        "mask_embed": "mask_embed.weight",
        "mask_embed.weight": "mask_embed.weight",
    }
    if name in exact_map:
        return exact_map[name]
    name = name.replace(".attention_conv.", ".attn_conv.")
    if len(name.encode("utf-8")) > 47:
        name = name.replace(".kernel_projection.", ".k_proj.")
    if len(name.encode("utf-8")) > 47:
        name = name.replace(".self_attn.", ".attn.")
    if len(name.encode("utf-8")) > 47:
        name = name.replace(".post_attention_layernorm.", ".post_norm.")
    enc = name.encode("utf-8")
    if len(enc) == 0 or len(enc) > 47:
        raise ValueError(f"AMF DFlash2 tensor name must be 1..47 UTF-8 bytes, got {len(enc)}: {name!r}")
    return name


def resolve_glm5_next_split_specs(
    out_base: str,
    shape: Sequence[int],
    glm5_cfg: Mapping[str, Any],
) -> list[tuple[str, tuple[tuple[int, int], ...], tuple[int, ...] | None]] | None:
    """Losslessly split fused `glm5_next` linear-attention projections or concatenated `conv1d` if present.

    Returns `[(split_name, ((row_start, row_count),), optional_shape_override), ...]` or `None`.
    """
    if is_vision_tensor(out_base):
        return None
    eff = _effective_glm5_cfg(glm5_cfg)
    la_cfg = eff.get("linear_attn_config") if isinstance(eff.get("linear_attn_config"), Mapping) else {}
    num_heads = int(eff.get("linear_num_heads") or la_cfg.get("num_heads") or 64)
    head_dim = int(eff.get("linear_head_dim") or la_cfg.get("head_dim") or 128)
    qkv_dim = num_heads * head_dim
    base = out_base[:-7] if out_base.endswith(".weight") else out_base

    if base.endswith(".linear_attn.fused_qkvbfg_a_proj") and len(shape) == 2:
        n_rows = int(shape[0])
        expected = 3 * qkv_dim + num_heads + 2 * head_dim
        if n_rows != expected:
            raise ValueError(f"glm5_next {out_base}: rows {n_rows} != expected {expected}")
        pfx = base[: -len(".fused_qkvbfg_a_proj")]
        off_b = 3 * qkv_dim
        off_fa = off_b + num_heads
        off_ga = off_fa + head_dim
        return [
            (f"{pfx}.q_proj.weight", ((0, qkv_dim),), None),
            (f"{pfx}.k_proj.weight", ((qkv_dim, qkv_dim),), None),
            (f"{pfx}.v_proj.weight", ((2 * qkv_dim, qkv_dim),), None),
            (f"{pfx}.b_proj.weight", ((off_b, num_heads),), None),
            (f"{pfx}.f_a_proj.weight", ((off_fa, head_dim),), None),
            (f"{pfx}.g_a_proj.weight", ((off_ga, head_dim),), None),
        ]

    if base.endswith(".linear_attn.conv1d") and len(shape) in (2, 3):
        n_rows = int(shape[0])
        if n_rows == 3 * qkv_dim:
            pfx = base[: -len(".conv1d")]
            kern = int(shape[-1])
            return [
                (f"{pfx}.q_conv1d.weight", ((0, qkv_dim),), (qkv_dim, kern)),
                (f"{pfx}.k_conv1d.weight", ((qkv_dim, qkv_dim),), (qkv_dim, kern)),
                (f"{pfx}.v_conv1d.weight", ((2 * qkv_dim, qkv_dim),), (qkv_dim, kern)),
            ]
    return None


# ---------------------------------------------------------------------------
# W28 Metadata Validators (`inspect-metadata`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlmMoeDsaMetadataReport:
    """Verified W28 HF metadata + safetensors header report for `glm_moe_dsa` (`GLM-5.2` / `GLM-5.3`)."""

    metadata_dir: pathlib.Path
    model_type: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    dense_layers: int
    moe_layers: int
    full_indexer_layers: int
    shared_indexer_layers: int
    index_topk_freq: int
    index_skip_topk_offset: int
    index_share_for_mtp_iteration: bool
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    rope_interleave: bool
    indexer_rope_interleave: bool
    mtp_layers: int
    active_params: int
    routed_expert_params: int
    total_params: int
    mla_kv_bytes_per_token_fp16: int
    indexer_k_bytes_per_token_fp16: int
    backbone_tensor_count: int
    mtp_tensor_count: int
    indexer_tensor_count: int
    routed_expert_ids_detected: int
    long_backbone_names: int
    long_mtp_names: int
    max_backbone_name_bytes: int
    header_summary_shards: int
    tier_admission: dict[str, dict[str, Any]]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata_dir"] = str(self.metadata_dir)
        out["valid"] = len(self.errors) == 0
        return out


@dataclass(frozen=True)
class Glm5NextMetadataReport:
    """Verified W28 HF metadata + safetensors header report for `glm5_next` (`GLM-5.3-Flash`)."""

    metadata_dir: pathlib.Path
    model_type: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    linear_attention_layers: int
    nope_mla_layers: int
    dense_layers: int
    moe_layers: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    swiglu_limit: float
    mtp_layers: int
    vision_layers: int
    vision_hidden_size: int
    active_params: int
    routed_expert_params: int
    total_params: int
    mla_kv_bytes_per_token_fp16: int
    kda_state_bytes_fp32: int
    backbone_tensor_count: int
    vision_tensor_count: int
    mtp_tensor_count: int
    hc_tensor_count: int
    indexer_tensor_count: int
    routed_expert_ids_detected: int
    has_fused_qkvbfg: bool
    long_backbone_names: int
    long_vision_names: int
    long_mtp_names: int
    max_backbone_name_bytes: int
    header_summary_shards: int
    tier_admission: dict[str, dict[str, Any]]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata_dir"] = str(self.metadata_dir)
        out["valid"] = len(self.errors) == 0
        return out


@dataclass(frozen=True)
class Glm5Dflash2MetadataReport:
    """Verified W28 HF metadata + safetensors header report for `DFlash2` draft models."""

    metadata_dir: pathlib.Path
    architectures: tuple[str, ...]
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    block_size: int
    mask_token_id: int
    sliding_window: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    target_layer_ids: tuple[int, ...]
    tensor_count: int
    long_names: int
    max_name_bytes: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metadata_dir"] = str(self.metadata_dir)
        out["valid"] = len(self.errors) == 0
        return out


def _matches_linear_shape(t_info: Mapping[str, Any], out_rows: int, in_cols: int) -> bool:
    """Return True if raw tensor shape matches `[out_rows, in_cols]` (raw or packed U8/U32)."""
    shp = t_info.get("shape")
    if not isinstance(shp, list) or len(shp) != 2 or int(shp[0]) != out_rows:
        return False
    dt = str(t_info.get("dtype") or "").upper()
    cols = int(shp[1])
    if cols == in_cols:
        return True
    if dt in ("U32", "I32"):
        return any((in_cols * bits) // 32 == cols for bits in (2, 3, 4, 8) if (in_cols * bits) % 32 == 0)
    if dt == "U8":
        return cols == in_cols // 2 and in_cols % 2 == 0
    return False


def _estimate_glm5_tier_admission(
    *,
    active_params: int,
    routed_expert_params: int,
    host_ram_gib: float | None = None,
) -> dict[str, dict[str, Any]]:
    # Always-active at ~0.625 B/param (mixed BF16 norms/embeds + FP8/Q4), routed at GSQ-RCO ~2.55 bpw (~0.320 B/param).
    active_q4_gib = (active_params * 0.625) / (1024**3)
    routed_rco_gib = (routed_expert_params * 0.320) / (1024**3)
    total_rco_gib = active_q4_gib + routed_rco_gib
    tiers: list[tuple[str, float]] = [
        ("24gb-m4", 18.186),
        ("128gb-max", 104.0),
        ("256gb-studio", 210.0),
        ("512gb-ultra", 420.0),
    ]
    if host_ram_gib is not None and host_ram_gib > 0:
        tiers.append(("custom", float(host_ram_gib) * 0.78))
    out: dict[str, dict[str, Any]] = {}
    for name, budget in tiers:
        fits_all_wired = total_rco_gib <= budget
        if fits_all_wired:
            mode = "full_wired"
        elif active_q4_gib + 2.0 <= budget:
            mode = "cold_nvme_hybrid"
        else:
            mode = "exceeds_always_active"
        out[name] = {
            "wired_budget_gib": round(budget, 2),
            "always_active_q4_gib": round(active_q4_gib, 2),
            "estimated_gsq_rco_gib": round(total_rco_gib, 2),
            "fits_all_wired": fits_all_wired,
            "mode": mode,
        }
    return out


def validate_glm_moe_dsa_metadata_dir(
    meta_dir: pathlib.Path,
    *,
    host_ram_gib: float | None = None,
) -> GlmMoeDsaMetadataReport:
    """Validate staged HF metadata + Range safetensors headers for `GLM-5.2` / `GLM-5.3` (`glm_moe_dsa`)."""
    meta_dir = pathlib.Path(meta_dir)
    cfg_p = meta_dir / "config.json"
    if not cfg_p.is_file():
        raise FileNotFoundError(f"config.json not found in {meta_dir}")
    raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    cfg = _effective_glm5_cfg(raw_cfg)
    errors: list[str] = []

    model_type = str(cfg.get("model_type") or "")
    if model_type != "glm_moe_dsa":
        errors.append(f"unexpected model_type {model_type!r} (expected 'glm_moe_dsa')")

    hidden = int(cfg.get("hidden_size") or 0)
    inter = int(cfg.get("intermediate_size") or 0)
    moe_inter = int(cfg.get("moe_intermediate_size") or 0)
    vocab = int(cfg.get("vocab_size") or 0)
    n_layers = int(cfg.get("num_hidden_layers") or 0)
    if hidden <= 0 or inter <= 0 or vocab <= 0 or n_layers <= 0:
        errors.append(f"invalid geometry hidden={hidden}, inter={inter}, vocab={vocab}, layers={n_layers}")

    idx_types = resolve_glm_moe_dsa_indexer_types(cfg)
    mlp_types = resolve_glm5_mlp_layer_types(cfg)
    full_idx = sum(1 for x in idx_types if x == "full")
    shared_idx = sum(1 for x in idx_types if x == "shared")
    dense_l = sum(1 for x in mlp_types if x == "dense")
    moe_l = sum(1 for x in mlp_types if x == "sparse")

    n_heads = int(cfg.get("num_attention_heads") or 0)
    n_kv = int(cfg.get("num_key_value_heads") or n_heads)
    q_lora = int(cfg.get("q_lora_rank") or 0)
    kv_lora = int(cfg.get("kv_lora_rank") or 0)
    qk_nope = int(cfg.get("qk_nope_head_dim") or 0)
    qk_rope = int(cfg.get("qk_rope_head_dim") or 0)
    v_hd = int(cfg.get("v_head_dim") or 0)
    idx_heads = int(cfg.get("index_n_heads") or 0)
    idx_hd = int(cfg.get("index_head_dim") or 0)
    idx_topk = int(cfg.get("index_topk") or 0)
    idx_freq = int(cfg.get("index_topk_freq") or 1)
    idx_off = int(cfg.get("index_skip_topk_offset") if cfg.get("index_skip_topk_offset") is not None else 2)
    idx_share_mtp = bool(cfg.get("index_share_for_mtp_iteration", False))
    n_exp = int(cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 0)
    n_shared = int(cfg.get("n_shared_experts") or 0)
    top_k = int(cfg.get("num_experts_per_tok") or 0)
    mtp_layers = int(cfg.get("num_nextn_predict_layers") or 0)
    rope_interleave = bool(cfg.get("rope_interleave", True))
    idx_rope_interleave = bool(cfg.get("indexer_rope_interleave", rope_interleave))

    wm, _ = _read_weight_map_keys(meta_dir)
    hdr_tensors, hdr_shards = _read_header_summary_tensors(meta_dir)
    all_keys = sorted(set(wm.keys()) | set(hdr_tensors.keys()))

    bb_cnt = 0
    mtp_cnt = 0
    idx_cnt = 0
    exp_ids: set[int] = set()
    long_bb: list[str] = []
    long_mtp: list[str] = []
    max_bb_len = 0
    exp_re = re.compile(r"\.experts\.(\d+)\.")

    for raw_name in all_keys:
        if any(raw_name.endswith(s) for s in _STRIP_QUANT_SUFFIXES if s not in (".weight_packed",)):
            continue
        if is_glm5_mtp_tensor(raw_name, n_layers if mtp_layers > 0 else None):
            mtp_cnt += 1
            try:
                nm = normalize_glm5_tensor_name(
                    raw_name,
                    model_type="glm_moe_dsa",
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
        if ".indexer." in raw_name:
            idx_cnt += 1
        em = exp_re.search(raw_name)
        if em is not None:
            exp_ids.add(int(em.group(1)))
        try:
            nm = normalize_glm5_tensor_name(raw_name, model_type="glm_moe_dsa")
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
        for l_idx in range(min(n_layers, 12)):
            for pfx in (f"model.layers.{l_idx}", f"layers.{l_idx}"):
                qb = hdr_tensors.get(f"{pfx}.self_attn.q_b_proj.weight")
                if qb is not None and not _matches_linear_shape(qb, n_heads * (qk_nope + qk_rope), q_lora):
                    errors.append(f"L{l_idx} q_b_proj shape {qb.get('shape')} != [{n_heads * (qk_nope + qk_rope)}, {q_lora}]")
                kva = hdr_tensors.get(f"{pfx}.self_attn.kv_a_proj_with_mqa.weight")
                if kva is not None and not _matches_linear_shape(kva, kv_lora + qk_rope, hidden):
                    errors.append(f"L{l_idx} kv_a_proj_with_mqa shape {kva.get('shape')} != [{kv_lora + qk_rope}, {hidden}]")
                kvb = hdr_tensors.get(f"{pfx}.self_attn.kv_b_proj.weight")
                if kvb is not None and not _matches_linear_shape(kvb, n_heads * (qk_nope + v_hd), kv_lora):
                    errors.append(f"L{l_idx} kv_b_proj shape {kvb.get('shape')} != [{n_heads * (qk_nope + v_hd)}, {kv_lora}]")
                if l_idx < len(idx_types) and idx_types[l_idx] == "full":
                    wqb = hdr_tensors.get(f"{pfx}.self_attn.indexer.wq_b.weight")
                    if wqb is not None and not _matches_linear_shape(wqb, idx_heads * idx_hd, q_lora):
                        errors.append(f"L{l_idx} indexer.wq_b shape {wqb.get('shape')} != [{idx_heads * idx_hd}, {q_lora}]")
                    wp = hdr_tensors.get(f"{pfx}.self_attn.indexer.weights_proj.weight")
                    if wp is not None and not _matches_linear_shape(wp, idx_heads, hidden):
                        errors.append(f"L{l_idx} indexer.weights_proj shape {wp.get('shape')} != [{idx_heads}, {hidden}]")
                elif l_idx < len(idx_types) and idx_types[l_idx] == "shared" and wm:
                    if f"{pfx}.self_attn.indexer.wk.weight" in wm:
                        errors.append(f"L{l_idx} is indexer_types='shared' but weight_map carries indexer.wk.weight")

    # Analytical parameter counts
    embed_p = 2 * vocab * hidden
    mla_per_l = (
        hidden * q_lora
        + q_lora * n_heads * (qk_nope + qk_rope)
        + hidden * (kv_lora + qk_rope)
        + kv_lora * n_heads * (qk_nope + v_hd)
        + n_heads * v_hd * hidden
    )
    idx_per_full = q_lora * (idx_heads * idx_hd) + hidden * idx_hd + hidden * idx_heads
    dense_mlp_p = 3 * hidden * inter
    shared_mlp_p = n_shared * 3 * hidden * moe_inter
    router_p = n_exp * hidden
    one_exp_p = 3 * hidden * moe_inter
    active_p = (
        embed_p
        + n_layers * mla_per_l
        + full_idx * idx_per_full
        + dense_l * dense_mlp_p
        + moe_l * (shared_mlp_p + router_p)
    )
    routed_p = moe_l * n_exp * one_exp_p
    total_p = active_p + routed_p

    return GlmMoeDsaMetadataReport(
        metadata_dir=meta_dir,
        model_type=model_type,
        hidden_size=hidden,
        intermediate_size=inter,
        moe_intermediate_size=moe_inter,
        vocab_size=vocab,
        num_hidden_layers=n_layers,
        dense_layers=dense_l,
        moe_layers=moe_l,
        full_indexer_layers=full_idx,
        shared_indexer_layers=shared_idx,
        index_topk_freq=idx_freq,
        index_skip_topk_offset=idx_off,
        index_share_for_mtp_iteration=idx_share_mtp,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_hd,
        index_n_heads=idx_heads,
        index_head_dim=idx_hd,
        index_topk=idx_topk,
        n_routed_experts=n_exp,
        n_shared_experts=n_shared,
        num_experts_per_tok=top_k,
        rope_interleave=rope_interleave,
        indexer_rope_interleave=idx_rope_interleave,
        mtp_layers=mtp_layers,
        active_params=active_p,
        routed_expert_params=routed_p,
        total_params=total_p,
        mla_kv_bytes_per_token_fp16=n_layers * (kv_lora + qk_rope) * 2,
        indexer_k_bytes_per_token_fp16=full_idx * idx_hd * 2,
        backbone_tensor_count=bb_cnt,
        mtp_tensor_count=mtp_cnt,
        indexer_tensor_count=idx_cnt,
        routed_expert_ids_detected=len(exp_ids),
        long_backbone_names=len(long_bb),
        long_mtp_names=len(long_mtp),
        max_backbone_name_bytes=max_bb_len,
        header_summary_shards=hdr_shards,
        tier_admission=_estimate_glm5_tier_admission(
            active_params=active_p,
            routed_expert_params=routed_p,
            host_ram_gib=host_ram_gib,
        ),
        errors=tuple(errors),
    )


def validate_glm5_next_metadata_dir(
    meta_dir: pathlib.Path,
    *,
    host_ram_gib: float | None = None,
) -> Glm5NextMetadataReport:
    """Validate staged HF metadata + Range safetensors headers for `GLM-5.3-Flash` (`glm5_next`)."""
    meta_dir = pathlib.Path(meta_dir)
    cfg_p = meta_dir / "config.json"
    if not cfg_p.is_file():
        raise FileNotFoundError(f"config.json not found in {meta_dir}")
    raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    cfg = _effective_glm5_cfg(raw_cfg)
    vis_cfg = raw_cfg.get("vision_config") if isinstance(raw_cfg.get("vision_config"), Mapping) else {}
    errors: list[str] = []

    model_type = str(raw_cfg.get("model_type") or cfg.get("model_type") or "")
    if model_type not in ("glm5_next", "glm5_next_text"):
        errors.append(f"unexpected model_type {model_type!r} (expected 'glm5_next')")

    hidden = int(cfg.get("hidden_size") or 0)
    inter = int(cfg.get("intermediate_size") or 0)
    moe_inter = int(cfg.get("moe_intermediate_size") or 0)
    vocab = int(cfg.get("vocab_size") or 0)
    n_layers = int(cfg.get("num_hidden_layers") or 0)
    if hidden <= 0 or inter <= 0 or vocab <= 0 or n_layers <= 0:
        errors.append(f"invalid geometry hidden={hidden}, inter={inter}, vocab={vocab}, layers={n_layers}")

    ltypes = resolve_glm5_next_layer_types(cfg)
    mlp_types = resolve_glm5_mlp_layer_types(cfg)
    lin_l = sum(1 for x in ltypes if x == "linear_attention")
    mla_l = sum(1 for x in ltypes if x != "linear_attention")
    dense_l = sum(1 for x in mlp_types if x == "dense")
    moe_l = sum(1 for x in mlp_types if x == "sparse")

    hc_mult = int(cfg.get("hc_mult") or 4)
    hc_iters = int(cfg.get("hc_sinkhorn_iters") or 20)
    hc_eps = float(cfg.get("hc_eps") or 1e-6)
    la_cfg = cfg.get("linear_attn_config") if isinstance(cfg.get("linear_attn_config"), Mapping) else {}
    lin_heads = int(cfg.get("linear_num_heads") or la_cfg.get("num_heads") or 64)
    lin_hd = int(cfg.get("linear_head_dim") or la_cfg.get("head_dim") or 128)
    lin_conv = int(cfg.get("linear_conv_kernel_dim") or la_cfg.get("short_conv_kernel_size") or 4)
    lin_lb = float(
        cfg.get("linear_lower_bound")
        if cfg.get("linear_lower_bound") is not None
        else (la_cfg.get("gate_lower_bound") if la_cfg.get("gate_lower_bound") is not None else -5.0)
    )

    n_heads = int(cfg.get("num_attention_heads") or 0)
    q_lora = int(cfg.get("q_lora_rank") or 0)
    kv_lora = int(cfg.get("kv_lora_rank") or 0)
    qk_nope = int(cfg.get("qk_nope_head_dim") or 0)
    qk_rope = int(cfg.get("qk_rope_head_dim") or 0)
    v_hd = int(cfg.get("v_head_dim") or 0)
    if qk_rope != 0:
        errors.append(f"glm5_next expects NoPE MLA qk_rope_head_dim == 0, got {qk_rope}")

    idx_heads = int(cfg.get("index_n_heads") or 0)
    idx_hd = int(cfg.get("index_head_dim") or 0)
    idx_topk = int(cfg.get("index_topk") or 0)
    idx_kpool = int(cfg.get("index_kpool") or 4)
    idx_tail = bool(cfg.get("index_kpool_always_select_tail", True))

    n_exp = int(cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 0)
    n_shared = int(cfg.get("n_shared_experts") or 0)
    top_k = int(cfg.get("num_experts_per_tok") or 0)
    swiglu_lim = float(cfg.get("swiglu_limit") or 10.0)
    mtp_layers = int(cfg.get("num_nextn_predict_layers") or 0)
    vis_layers = int(vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0)
    vis_hidden = int(vis_cfg.get("hidden_size") or 0)

    wm, _ = _read_weight_map_keys(meta_dir)
    hdr_tensors, hdr_shards = _read_header_summary_tensors(meta_dir)
    all_keys = sorted(set(wm.keys()) | set(hdr_tensors.keys()))

    bb_cnt = 0
    vis_cnt = 0
    mtp_cnt = 0
    hc_cnt = 0
    idx_cnt = 0
    has_fused_qkvbfg = False
    exp_ids: set[int] = set()
    long_bb: list[str] = []
    long_vis: list[str] = []
    long_mtp: list[str] = []
    max_bb_len = 0
    exp_re = re.compile(r"\.experts\.(\d+)\.")

    for raw_name in all_keys:
        if any(raw_name.endswith(s) for s in _STRIP_QUANT_SUFFIXES if s not in (".weight_packed",)):
            continue
        if "fused_qkvbfg_a_proj" in raw_name:
            has_fused_qkvbfg = True
        if is_vision_tensor(raw_name):
            vis_cnt += 1
            nm = normalize_vision_name(raw_name)
            if len(nm.encode("utf-8")) > 47:
                long_vis.append(nm)
            continue
        if is_glm5_mtp_tensor(raw_name, n_layers if mtp_layers > 0 else None):
            mtp_cnt += 1
            try:
                nm = normalize_glm5_tensor_name(
                    raw_name,
                    model_type="glm5_next",
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
        if ".hc_" in raw_name:
            hc_cnt += 1
        if ".indexer." in raw_name:
            idx_cnt += 1
        em = exp_re.search(raw_name)
        if em is not None:
            exp_ids.add(int(em.group(1)))
        try:
            nm = normalize_glm5_tensor_name(raw_name, model_type="glm5_next", layer_types=ltypes)
        except ValueError:
            long_bb.append(raw_name)
        else:
            blen = len(nm.encode("utf-8"))
            max_bb_len = max(max_bb_len, blen)
            if blen > 47:
                long_bb.append(nm)

    if long_bb:
        errors.append(f"{len(long_bb)} backbone tensor name(s) exceed 47 B (sample: {long_bb[:3]})")
    if long_vis:
        errors.append(f"{len(long_vis)} vision tensor name(s) exceed 47 B (sample: {long_vis[:3]})")
    if long_mtp:
        errors.append(f"{len(long_mtp)} MTP tensor name(s) exceed 47 B (sample: {long_mtp[:3]})")

    if hdr_tensors:
        hc_rows = (2 + hc_mult) * hc_mult
        qkv_dim = lin_heads * lin_hd
        for l_idx in range(min(n_layers, 6)):
            for pfx in (f"model.language_model.layers.{l_idx}", f"model.layers.{l_idx}", f"layers.{l_idx}"):
                hcf = hdr_tensors.get(f"{pfx}.hc_attn_fn")
                if hcf is not None and hcf.get("shape") != [hc_rows, hc_mult * hidden]:
                    errors.append(f"L{l_idx} hc_attn_fn shape {hcf.get('shape')} != [{hc_rows}, {hc_mult * hidden}]")
                hcb = hdr_tensors.get(f"{pfx}.hc_attn_base")
                if hcb is not None and hcb.get("shape") != [hc_rows]:
                    errors.append(f"L{l_idx} hc_attn_base shape {hcb.get('shape')} != [{hc_rows}]")
                if l_idx < len(ltypes) and ltypes[l_idx] == "linear_attention":
                    alog = hdr_tensors.get(f"{pfx}.self_attn.A_log")
                    if alog is not None and alog.get("shape") != [lin_heads]:
                        errors.append(f"L{l_idx} A_log shape {alog.get('shape')} != [{lin_heads}]")
                    dt = hdr_tensors.get(f"{pfx}.self_attn.dt_bias")
                    if dt is not None and dt.get("shape") != [qkv_dim]:
                        errors.append(f"L{l_idx} dt_bias shape {dt.get('shape')} != [{qkv_dim}]")
                    qcv = hdr_tensors.get(f"{pfx}.self_attn.q_conv1d.weight")
                    if qcv is not None and qcv.get("shape") not in ([qkv_dim, 1, lin_conv], [qkv_dim, lin_conv]):
                        errors.append(f"L{l_idx} q_conv1d shape {qcv.get('shape')} != [{qkv_dim}, 1, {lin_conv}]")
                elif l_idx < len(ltypes):
                    kva = hdr_tensors.get(f"{pfx}.self_attn.kv_a_proj_with_mqa.weight")
                    if kva is not None and not _matches_linear_shape(kva, kv_lora, hidden):
                        errors.append(f"L{l_idx} NoPE kv_a_proj_with_mqa shape {kva.get('shape')} != [{kv_lora}, {hidden}]")
                    ape = hdr_tensors.get(f"{pfx}.self_attn.indexer.index_kpool_compress_ape")
                    if ape is not None and ape.get("shape") != [idx_kpool, idx_hd]:
                        errors.append(f"L{l_idx} kpool_ape shape {ape.get('shape')} != [{idx_kpool}, {idx_hd}]")

    embed_p = 2 * vocab * hidden
    qkv_dim = lin_heads * lin_hd
    kda_per_l = 4 * hidden * qkv_dim + hidden * lin_heads + 2 * (hidden * lin_hd + lin_hd * qkv_dim)
    mla_per_l = hidden * q_lora + q_lora * n_heads * qk_nope + hidden * kv_lora + kv_lora * n_heads * (qk_nope + v_hd) + n_heads * v_hd * hidden
    idx_per_l = q_lora * (idx_heads * idx_hd) + 2 * hidden * idx_hd + hidden * idx_heads
    hc_per_l = 2 * ((2 + hc_mult) * hc_mult * hc_mult * hidden)
    dense_mlp_p = 3 * hidden * inter
    shared_mlp_p = n_shared * 3 * hidden * moe_inter
    router_p = n_exp * hidden
    one_exp_p = 3 * hidden * moe_inter
    active_p = (
        embed_p
        + lin_l * kda_per_l
        + mla_l * (mla_per_l + idx_per_l)
        + n_layers * hc_per_l
        + dense_l * dense_mlp_p
        + moe_l * (shared_mlp_p + router_p)
    )
    routed_p = moe_l * n_exp * one_exp_p
    total_p = active_p + routed_p

    return Glm5NextMetadataReport(
        metadata_dir=meta_dir,
        model_type=model_type,
        hidden_size=hidden,
        intermediate_size=inter,
        moe_intermediate_size=moe_inter,
        vocab_size=vocab,
        num_hidden_layers=n_layers,
        linear_attention_layers=lin_l,
        nope_mla_layers=mla_l,
        dense_layers=dense_l,
        moe_layers=moe_l,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_iters,
        hc_eps=hc_eps,
        linear_num_heads=lin_heads,
        linear_head_dim=lin_hd,
        linear_conv_kernel_dim=lin_conv,
        linear_lower_bound=lin_lb,
        num_attention_heads=n_heads,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_hd,
        index_n_heads=idx_heads,
        index_head_dim=idx_hd,
        index_topk=idx_topk,
        index_kpool=idx_kpool,
        index_kpool_always_select_tail=idx_tail,
        n_routed_experts=n_exp,
        n_shared_experts=n_shared,
        num_experts_per_tok=top_k,
        swiglu_limit=swiglu_lim,
        mtp_layers=mtp_layers,
        vision_layers=vis_layers,
        vision_hidden_size=vis_hidden,
        active_params=active_p,
        routed_expert_params=routed_p,
        total_params=total_p,
        mla_kv_bytes_per_token_fp16=mla_l * kv_lora * 2,
        kda_state_bytes_fp32=lin_l * lin_heads * lin_hd * lin_hd * 4,
        backbone_tensor_count=bb_cnt,
        vision_tensor_count=vis_cnt,
        mtp_tensor_count=mtp_cnt,
        hc_tensor_count=hc_cnt,
        indexer_tensor_count=idx_cnt,
        routed_expert_ids_detected=len(exp_ids),
        has_fused_qkvbfg=has_fused_qkvbfg,
        long_backbone_names=len(long_bb),
        long_vision_names=len(long_vis),
        long_mtp_names=len(long_mtp),
        max_backbone_name_bytes=max_bb_len,
        header_summary_shards=hdr_shards,
        tier_admission=_estimate_glm5_tier_admission(
            active_params=active_p,
            routed_expert_params=routed_p,
            host_ram_gib=host_ram_gib,
        ),
        errors=tuple(errors),
    )


def validate_glm5_dflash2_metadata_dir(meta_dir: pathlib.Path) -> Glm5Dflash2MetadataReport:
    """Validate staged HF metadata + safetensors header for a `DFlash2` draft checkpoint."""
    meta_dir = pathlib.Path(meta_dir)
    cfg_p = meta_dir / "config.json"
    if not cfg_p.is_file():
        raise FileNotFoundError(f"config.json not found in {meta_dir}")
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    dc = cfg.get("dflash_config") if isinstance(cfg.get("dflash_config"), Mapping) else {}
    errors: list[str] = []

    archs = tuple(str(a) for a in (cfg.get("architectures") or []))
    if not is_glm5_dflash2_config(cfg):
        errors.append(f"config.json does not describe DFlash2 (architectures={archs!r})")

    n_layers = int(cfg.get("num_hidden_layers") or 0)
    hidden = int(cfg.get("hidden_size") or 0)
    inter = int(cfg.get("intermediate_size") or 0)
    n_heads = int(cfg.get("num_attention_heads") or 0)
    n_kv = int(cfg.get("num_key_value_heads") or 0)
    hd = int(cfg.get("head_dim") or 0)
    vocab = int(cfg.get("vocab_size") or cfg.get("draft_vocab_size") or 0)
    bsize = int(dc.get("block_size") or cfg.get("block_size") or 8)
    mask_id = int(dc.get("mask_token_id") or cfg.get("mask_token_id") or 154856)
    win = int(cfg.get("sliding_window") or 2048)
    conv_k = int(dc.get("conv_kernel_size") or cfg.get("conv_kernel_size") or 2)
    conv_g = int(dc.get("conv_group_size") or cfg.get("conv_group_size") or 16)
    sel_r = int(dc.get("selector_rank") or cfg.get("selector_rank") or 256)
    sel_k = int(dc.get("selector_top_k") or cfg.get("selector_top_k") or 16)
    taps = tuple(int(x) for x in (dc.get("target_layer_ids") or cfg.get("target_layer_ids") or ()))

    wm, _ = _read_weight_map_keys(meta_dir)
    hdr_tensors, _ = _read_header_summary_tensors(meta_dir)
    all_keys = sorted(set(wm.keys()) | set(hdr_tensors.keys()))
    long_names: list[str] = []
    max_len = 0
    for raw_name in all_keys:
        try:
            nm = normalize_glm5_dflash2_tensor_name(raw_name)
        except ValueError:
            long_names.append(raw_name)
        else:
            max_len = max(max_len, len(nm.encode("utf-8")))
            if len(nm.encode("utf-8")) > 47:
                long_names.append(nm)

    if long_names:
        errors.append(f"{len(long_names)} DFlash2 tensor name(s) exceed 47 B (sample: {long_names[:3]})")
    if hdr_tensors:
        fc = hdr_tensors.get("fc.weight")
        if fc is not None and fc.get("shape") != [hidden, len(taps) * hidden]:
            errors.append(f"fc.weight shape {fc.get('shape')} != [{hidden}, {len(taps) * hidden}]")
        bk = hdr_tensors.get("layers.0.attention_conv.base_kernel")
        if bk is not None and bk.get("shape") != [2, conv_k, hidden]:
            errors.append(f"layers.0.attention_conv.base_kernel shape {bk.get('shape')} != [2, {conv_k}, {hidden}]")
        kp = hdr_tensors.get("layers.0.attention_conv.kernel_projection.weight")
        exp_kp = 2 * conv_k * (hidden // conv_g)
        if kp is not None and kp.get("shape") != [exp_kp, hidden]:
            errors.append(f"layers.0.attention_conv.kernel_projection shape {kp.get('shape')} != [{exp_kp}, {hidden}]")
        hp = hdr_tensors.get("candidate_selector.hidden_projection.weight")
        if hp is not None and hp.get("shape") != [sel_r, hidden]:
            errors.append(f"candidate_selector.hidden_projection shape {hp.get('shape')} != [{sel_r}, {hidden}]")

    return Glm5Dflash2MetadataReport(
        metadata_dir=meta_dir,
        architectures=archs,
        num_hidden_layers=n_layers,
        hidden_size=hidden,
        intermediate_size=inter,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=hd,
        vocab_size=vocab,
        block_size=bsize,
        mask_token_id=mask_id,
        sliding_window=win,
        conv_kernel_size=conv_k,
        conv_group_size=conv_g,
        selector_rank=sel_r,
        selector_top_k=sel_k,
        target_layer_ids=taps,
        tensor_count=len(all_keys),
        long_names=len(long_names),
        max_name_bytes=max_len,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Runtime `.amf` `config.json` Builders
# ---------------------------------------------------------------------------


def build_glm_moe_dsa_config_json(
    src_cfg: Mapping[str, Any],
    *,
    cold_experts_count: int = 0,
) -> dict[str, Any]:
    """Build canonical `.amf` runtime `config.json` for `glm_moe_dsa` (`GLM-5.2` / `GLM-5.3`)."""
    cfg = _effective_glm5_cfg(src_cfg)
    n_layers = int(cfg.get("num_hidden_layers") or 78)
    idx_types = resolve_glm_moe_dsa_indexer_types(cfg)
    mlp_types = resolve_glm5_mlp_layer_types(cfg)
    rope_p = cfg.get("rope_parameters") if isinstance(cfg.get("rope_parameters"), Mapping) else {}
    out: dict[str, Any] = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": int(cfg.get("hidden_size") or 6144),
        "intermediate_size": int(cfg.get("intermediate_size") or 12288),
        "moe_intermediate_size": int(cfg.get("moe_intermediate_size") or 2048),
        "num_hidden_layers": n_layers,
        "vocab_size": int(cfg.get("vocab_size") or 154880),
        "first_k_dense_replace": int(cfg.get("first_k_dense_replace") or 3),
        "mlp_layer_types": mlp_types,
        "layer_types": ["indexed_attention"] * n_layers,
        "indexer_types": idx_types,
        "index_n_heads": int(cfg.get("index_n_heads") or 32),
        "index_head_dim": int(cfg.get("index_head_dim") or 128),
        "index_topk": int(cfg.get("index_topk") or 2048),
        "index_topk_freq": int(cfg.get("index_topk_freq") or 4),
        "index_skip_topk_offset": int(cfg.get("index_skip_topk_offset") if cfg.get("index_skip_topk_offset") is not None else 3),
        "index_share_for_mtp_iteration": bool(cfg.get("index_share_for_mtp_iteration", True)),
        "indexer_rope_interleave": bool(cfg.get("indexer_rope_interleave", True)),
        "num_attention_heads": int(cfg.get("num_attention_heads") or 64),
        "num_key_value_heads": int(cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 64),
        "q_lora_rank": int(cfg.get("q_lora_rank") or 2048),
        "kv_lora_rank": int(cfg.get("kv_lora_rank") or 512),
        "qk_nope_head_dim": int(cfg.get("qk_nope_head_dim") or 192),
        "qk_rope_head_dim": int(cfg.get("qk_rope_head_dim") or 64),
        "qk_head_dim": int(cfg.get("qk_nope_head_dim") or 192) + int(cfg.get("qk_rope_head_dim") or 64),
        "v_head_dim": int(cfg.get("v_head_dim") or 256),
        "n_routed_experts": int(cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 256),
        "n_shared_experts": int(cfg.get("n_shared_experts") or 1),
        "num_experts_per_tok": int(cfg.get("num_experts_per_tok") or 8),
        "n_group": int(cfg.get("n_group") or 1),
        "topk_group": int(cfg.get("topk_group") or 1),
        "scoring_func": str(cfg.get("scoring_func") or "sigmoid"),
        "topk_method": str(cfg.get("topk_method") or "noaux_tc"),
        "norm_topk_prob": bool(cfg.get("norm_topk_prob", True)),
        "routed_scaling_factor": float(cfg.get("routed_scaling_factor") or 2.5),
        "rope_interleave": bool(cfg.get("rope_interleave", True)),
        "rope_theta": float(rope_p.get("rope_theta") or cfg.get("rope_theta") or 8000000.0),
        "rope_parameters": dict(rope_p) if rope_p else {"rope_theta": 8000000.0, "rope_type": "default"},
        "rms_norm_eps": float(cfg.get("rms_norm_eps") or 1e-5),
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "max_position_embeddings": int(cfg.get("max_position_embeddings") or 1048576),
        "num_nextn_predict_layers": int(cfg.get("num_nextn_predict_layers") or 0),
        "indexer_layout": "attn.idx",
    }
    if cold_experts_count > 0:
        out["cold_experts_count"] = int(cold_experts_count)
    return out


def build_glm5_next_config_json(
    src_cfg: Mapping[str, Any],
    *,
    cold_experts_count: int = 0,
    include_vision: bool = True,
) -> dict[str, Any]:
    """Build canonical `.amf` runtime `config.json` for `glm5_next` (`GLM-5.3-Flash`)."""
    cfg = _effective_glm5_cfg(src_cfg)
    n_layers = int(cfg.get("num_hidden_layers") or 45)
    ltypes = resolve_glm5_next_layer_types(cfg)
    mlp_types = resolve_glm5_mlp_layer_types(cfg)
    la_cfg = cfg.get("linear_attn_config") if isinstance(cfg.get("linear_attn_config"), Mapping) else {}
    lin_heads = int(cfg.get("linear_num_heads") or la_cfg.get("num_heads") or 64)
    lin_hd = int(cfg.get("linear_head_dim") or la_cfg.get("head_dim") or 128)
    lin_lora_rank = int(cfg.get("linear_lora_rank") or la_cfg.get("lora_rank") or lin_hd)
    lin_conv = int(cfg.get("linear_conv_kernel_dim") or la_cfg.get("short_conv_kernel_size") or 4)
    lin_lb = float(
        cfg.get("linear_lower_bound")
        if cfg.get("linear_lower_bound") is not None
        else (la_cfg.get("gate_lower_bound") if la_cfg.get("gate_lower_bound") is not None else -5.0)
    )
    sinkhorn_iters = int(cfg.get("hc_sinkhorn_iters") or cfg.get("sinkhorn_iters") or 20)
    out: dict[str, Any] = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "hidden_size": int(cfg.get("hidden_size") or 4096),
        "intermediate_size": int(cfg.get("intermediate_size") or 12288),
        "moe_intermediate_size": int(cfg.get("moe_intermediate_size") or 2048),
        "num_hidden_layers": n_layers,
        "vocab_size": int(cfg.get("vocab_size") or 154880),
        "first_k_dense_replace": int(cfg.get("first_k_dense_replace") or 3),
        "mlp_layer_types": mlp_types,
        "layer_types": ltypes,
        "indexer_types": resolve_glm_moe_dsa_indexer_types(
            {**cfg, "index_topk_freq": 1, "index_skip_topk_offset": 0}
        ),
        "mhc": True,
        "hc_mult": int(cfg.get("hc_mult") or 4),
        "sinkhorn_iters": sinkhorn_iters,
        "hc_sinkhorn_iters": sinkhorn_iters,
        "hc_eps": float(cfg.get("hc_eps") or 1e-6),
        "hc_head_type": "unweighted_mean",
        "linear_attn_kind": "kda",
        "linear_num_heads": lin_heads,
        "linear_head_dim": lin_hd,
        "linear_lora_rank": lin_lora_rank,
        "linear_conv_kernel_dim": lin_conv,
        "linear_lower_bound": lin_lb,
        "linear_o_norm_gate": "sigmoid",
        "mla_use_nope": True,
        "num_attention_heads": int(cfg.get("num_attention_heads") or 64),
        "num_key_value_heads": int(cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 64),
        "q_lora_rank": int(cfg.get("q_lora_rank") or 1536),
        "kv_lora_rank": int(cfg.get("kv_lora_rank") or 512),
        "qk_nope_head_dim": int(cfg.get("qk_nope_head_dim") or 256),
        "qk_rope_head_dim": 0,
        "qk_head_dim": int(cfg.get("qk_nope_head_dim") or 256),
        "v_head_dim": int(cfg.get("v_head_dim") or 256),
        "index_n_heads": int(cfg.get("index_n_heads") or 32),
        "index_head_dim": int(cfg.get("index_head_dim") or 128),
        "index_topk": int(cfg.get("index_topk") or 2048),
        "index_kpool": int(cfg.get("index_kpool") or 4),
        "index_kpool_always_select_tail": bool(cfg.get("index_kpool_always_select_tail", True)),
        "index_share_for_mtp_iteration": bool(cfg.get("index_share_for_mtp_iteration", True)),
        "n_routed_experts": int(cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 288),
        "n_shared_experts": int(cfg.get("n_shared_experts") or 1),
        "num_experts_per_tok": int(cfg.get("num_experts_per_tok") or 8),
        "n_group": int(cfg.get("n_group") or 1),
        "topk_group": int(cfg.get("topk_group") or 1),
        "scoring_func": str(cfg.get("scoring_func") or "sigmoid"),
        "topk_method": str(cfg.get("topk_method") or "noaux_tc"),
        "norm_topk_prob": bool(cfg.get("norm_topk_prob", True)),
        "routed_scaling_factor": float(cfg.get("routed_scaling_factor") or 2.5),
        "swiglu_limit": float(cfg.get("swiglu_limit") or 10.0),
        "rms_norm_eps": float(cfg.get("rms_norm_eps") or 1e-5),
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "max_position_embeddings": int(cfg.get("max_position_embeddings") or 1048576),
        "num_nextn_predict_layers": int(cfg.get("num_nextn_predict_layers") or 0),
        "indexer_layout": "attn.idx",
    }
    if include_vision and isinstance(src_cfg.get("vision_config"), Mapping):
        vc = src_cfg["vision_config"]
        out["vision_config"] = {
            "model_type": "glm5_next_vision",
            "depth": int(vc.get("depth") or vc.get("num_hidden_layers") or 24),
            "hidden_size": int(vc.get("hidden_size") or 1024),
            "intermediate_size": int(vc.get("intermediate_size") or 4096),
            "num_heads": int(vc.get("num_heads") or 16),
            "patch_size": int(vc.get("patch_size") or 14),
            "temporal_patch_size": int(vc.get("temporal_patch_size") or 2),
            "spatial_merge_size": int(vc.get("spatial_merge_size") or 2),
            "out_hidden_size": int(vc.get("out_hidden_size") or cfg["hidden_size"]),
            "projection_intermediate_size": int(vc.get("projection_intermediate_size") or 10240),
            "swiglu_limit": float(vc.get("swiglu_limit") or 10.0),
            "rope_type": "axial",
        }
    if cold_experts_count > 0:
        out["cold_experts_count"] = int(cold_experts_count)
    return out


def build_glm5_mtp_config_dict(src_cfg: Mapping[str, Any], *, is_glm5_next: bool) -> dict[str, Any]:
    """Build standalone `1L` `mtp.amf` `config.json` for `glm_moe_dsa` or `glm5_next`."""
    base = (
        build_glm5_next_config_json(src_cfg, include_vision=False)
        if is_glm5_next
        else build_glm_moe_dsa_config_json(src_cfg)
    )
    return {
        **base,
        "model_type": "glm5_next_mtp" if is_glm5_next else "glm_moe_dsa_mtp",
        "mtp_arch": "glm5_next" if is_glm5_next else "glm_moe_dsa",
        "format": "mtp",
        "mtp_chained": False,
        "num_hidden_layers": 1,
        "first_k_dense_replace": 0,
        "mlp_layer_types": ["sparse"],
        "layer_types": ["indexed_attention"],
        "indexer_types": ["full"],
        "mhc": False,
        "hc_mult": 1,
        "is_moe": True,
        "index_share_for_mtp_iteration": bool(base.get("index_share_for_mtp_iteration", True)),
    }


def build_glm5_dflash2_config_dict(
    src_cfg: Mapping[str, Any],
    *,
    target_cfg: Mapping[str, Any] | None = None,
    block_size_override: int | None = None,
) -> dict[str, Any]:
    """Build canonical `DFlash2` draft `config.json`."""
    dc = src_cfg.get("dflash_config") if isinstance(src_cfg.get("dflash_config"), Mapping) else {}
    tcfg = _effective_glm5_cfg(target_cfg) if target_cfg else {}
    bsize = int(block_size_override or dc.get("block_size") or src_cfg.get("block_size") or 8)
    hd = int(src_cfg.get("head_dim") or 128)
    taps = dc.get("target_layer_ids") or src_cfg.get("target_layer_ids") or [5, 14, 24, 33, 42]
    conv_gs = int(dc.get("conv_group_size") or src_cfg.get("conv_group_size") or dc.get("conv_groups") or src_cfg.get("conv_groups") or 16)
    return {
        "model_type": str(src_cfg.get("model_type") or "dflash2"),
        "architectures": ["DFlash2DraftModel"],
        "draft_kind": "dflash2",
        "dflare_format": "dflash2",
        "dflash_version": 2,
        "hidden_size": int(src_cfg.get("hidden_size") or tcfg.get("hidden_size") or 4096),
        "intermediate_size": int(src_cfg.get("intermediate_size") or 12288),
        "num_hidden_layers": int(src_cfg.get("num_hidden_layers") or 5),
        "num_attention_heads": int(src_cfg.get("num_attention_heads") or 32),
        "num_key_value_heads": int(src_cfg.get("num_key_value_heads") or 8),
        "head_dim": hd,
        "v_head_dim": int(src_cfg.get("v_head_dim") or hd),
        "vocab_size": int(src_cfg.get("vocab_size") or tcfg.get("vocab_size") or 154880),
        "block_size": bsize,
        "mask_token_id": int(dc.get("mask_token_id") or src_cfg.get("mask_token_id") or 154856),
        "target_layer_ids": [int(x) for x in taps],
        "sliding_window": int(src_cfg.get("sliding_window") or 2048),
        "sliding_window_size": int(src_cfg.get("sliding_window") or 2048),
        "rope_theta": float(src_cfg.get("rope_theta") or 10000.0),
        "partial_rotary_factor": float(src_cfg.get("partial_rotary_factor") or 1.0),
        "rope_dim": int(hd * float(src_cfg.get("partial_rotary_factor") or 1.0)),
        "rms_norm_eps": float(src_cfg.get("rms_norm_eps") or 1e-5),
        "attention_causal": False,
        "conv_kernel_size": int(dc.get("conv_kernel_size") or src_cfg.get("conv_kernel_size") or 2),
        "conv_groups": conv_gs,
        "conv_group_size": conv_gs,
        "selector_rank": int(dc.get("selector_rank") or src_cfg.get("selector_rank") or 256),
        "selector_top_k": int(dc.get("selector_top_k") or src_cfg.get("selector_top_k") or 16),
        "conv_layout": "attn_conv",
    }


# ---------------------------------------------------------------------------
# Cold-Expert Selection & Checkpoint Conversion (`W26`)
# ---------------------------------------------------------------------------


def _select_glm5_cold_experts(
    *,
    n_layers: int,
    n_experts: int,
    mlp_types: Sequence[str],
    tier_profile: Glm5TierProfile | None,
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
            eff_per_layer = max(1, int(round(n_experts * 0.85)))
    if eff_per_layer is not None and eff_per_layer > 0 and n_experts > 0:
        keep_hot = max(0, n_experts - eff_per_layer)
        for l_idx in range(n_layers):
            if l_idx < len(mlp_types) and mlp_types[l_idx] == "dense":
                continue
            for e_idx in range(keep_hot, n_experts):
                out.add((l_idx, e_idx))
    return out


@dataclass(frozen=True)
class Glm5ConvertReport:
    """Summary returned by `convert_glm5_checkpoint`."""

    model_type: str
    backbone_amf: pathlib.Path
    backbone_metas: tuple[PackedTensorMeta, ...]
    mtp_amf: pathlib.Path | None = None
    mtp_metas: tuple[PackedTensorMeta, ...] = ()
    dflash2_amf: pathlib.Path | None = None
    dflash2_metas: tuple[PackedTensorMeta, ...] = ()
    vision_amf: pathlib.Path | None = None
    vision_metas: tuple[PackedTensorMeta, ...] = ()
    cold_expert_count: int = 0
    stripped_vision_tensors: int = 0
    config_json_path: pathlib.Path | None = None


def export_glm5_mtp(
    input_path: pathlib.Path,
    output_amf: pathlib.Path,
    *,
    target_config: Mapping[str, Any] | None = None,
    model_id: str = "glm5-mtp",
    dtype_mode: str = "auto",
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    use_rco: bool = True,
    cold_experts: Iterable[tuple[int, int]] = (),
    all_experts_cold: bool = False,
    embed_metadata: bool = True,
    config_output: pathlib.Path | None = None,
    workers: int = 1,
) -> list[PackedTensorMeta]:
    """Extract & pack `1L` zero-reindex `MTP` (`layers.78.*` / `layers.45.*` / `mtp.*`) into standalone `.amf`."""
    in_p = pathlib.Path(input_path)
    resolved = resolve_checkpoint(in_p)
    raw = _collect_tensors(resolved.shard_paths)
    tcfg = dict(target_config) if target_config is not None else {}
    if not tcfg and resolved.config_path is not None and resolved.config_path.is_file():
        tcfg = json.loads(resolved.config_path.read_text(encoding="utf-8"))
    eff_cfg = _effective_glm5_cfg(tcfg)
    bb_layers = int(eff_cfg.get("num_hidden_layers") or 0)
    is_next = is_glm5_next_config(tcfg)

    mtp_keys = [k for k in raw if is_glm5_mtp_tensor(k, bb_layers if bb_layers > 0 else None)]
    if not mtp_keys:
        raise ValueError(f"no GLM-5 MTP tensors found in {in_p}")
    non_mtp_excludes = [k for k in raw if k not in set(mtp_keys)]

    mtp_cfg = build_glm5_mtp_config_dict(tcfg, is_glm5_next=is_next)
    mtp_cfg_b = json.dumps(mtp_cfg, indent=2).encode("utf-8") + b"\n"
    if config_output is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_bytes(mtp_cfg_b)

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_cfg = pathlib.Path(tmp_d) / "mtp_config.json"
        tmp_cfg.write_bytes(mtp_cfg_b)
        metas = pack_amf_from_safetensors(
            in_p,
            output_amf,
            model_id=model_id[:63],
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            target_bits=target_bits,
            use_rco=use_rco,
            cold_experts=cold_experts,
            all_experts_cold=all_experts_cold,
            exclude_tensors=non_mtp_excludes,
            strip_vision=True,
            embed_metadata=embed_metadata,
            config_path=tmp_cfg if embed_metadata else None,
            workers=workers,
            glm5_config=eff_cfg,
            glm5_mtp_standalone=True,
            glm5_mtp_source_layer=bb_layers if bb_layers > 0 else None,
        )
    if embed_metadata:
        metas = embed_metadata_into_amf(output_amf, config_bytes=mtp_cfg_b, output_path=output_amf)
    return metas


def export_glm5_dflash2(
    dflash_input: pathlib.Path,
    output_amf: pathlib.Path,
    *,
    target_config: Mapping[str, Any] | None = None,
    block_size_override: int | None = None,
    model_id: str = "glm5-dflash2",
    dtype_mode: str = "auto",
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    use_rco: bool = True,
    embed_metadata: bool = True,
    config_output: pathlib.Path | None = None,
    workers: int = 1,
) -> list[PackedTensorMeta]:
    """Export `DFlash2` draft checkpoint (safetensors or GGUF) to 16 KiB-aligned `.amf`."""
    in_p = pathlib.Path(dflash_input)
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
    df_cfg_p = base_dir / "config.json"
    df_cfg = json.loads(df_cfg_p.read_text(encoding="utf-8")) if df_cfg_p.is_file() else {}
    tcfg = dict(target_config) if target_config is not None else {}
    cfg_dict = build_glm5_dflash2_config_dict(df_cfg, target_cfg=tcfg, block_size_override=block_size_override)
    cfg_bytes = json.dumps(cfg_dict, indent=2).encode("utf-8") + b"\n"
    if config_output is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_bytes(cfg_bytes)

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_cfg = pathlib.Path(tmp_d) / "dflash2_config.json"
        tmp_cfg.write_bytes(cfg_bytes)
        metas = pack_amf_from_safetensors(
            in_p,
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
            glm5_dflash2_mode=True,
        )
    if embed_metadata:
        metas = embed_metadata_into_amf(output_amf, config_bytes=cfg_bytes, output_path=output_amf)
    return metas


def convert_glm5_checkpoint(
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
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
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
) -> Glm5ConvertReport:
    """Convert `GLM-5.2`/`5.3` (`glm_moe_dsa`) or `GLM-5.3-Flash` (`glm5_next`) + `24L` ViT + `1L` MTP + `DFlash2` to `.amf`."""
    in_p = pathlib.Path(input_path)
    out_p = pathlib.Path(output_path)
    tier_prof = resolve_glm5_tier_profile(target_tier)
    eff_model_id = (model_id or (in_p.name if in_p.is_dir() else out_p.stem))[:63]

    if in_p.is_file() and in_p.suffix.lower() == ".gguf":
        from aminfer_tools.gguf import pack_amf_from_gguf, read_gguf_header

        sh = read_gguf_header(in_p)
        arch = str(sh.kv.get("general.architecture") or "glm-dsa")
        n_layers = int(sh.kv.get(f"{arch}.block_count", 45) or 45)
        n_exp = int(sh.kv.get(f"{arch}.expert_count", 0) or 0)
        dense_cnt = int(sh.kv.get(f"{arch}.leading_dense_block_count", 3) or 3)
        mlp_types = ["dense" if i < dense_cnt else "sparse" for i in range(n_layers)]
        cold_set = _select_glm5_cold_experts(
            n_layers=n_layers,
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
            vision_block_dtype=vision_block_dtype,
            group_size=group_size,
            strict_vocab_check=strict_vocab_check,
        )
        df_metas: list[PackedTensorMeta] = []
        if dflash_output is not None and dflash_input is not None:
            df_metas = export_glm5_dflash2(
                dflash_input,
                dflash_output,
                block_size_override=dflash_block_size,
                model_id=f"{eff_model_id[:52]}-df2",
                dtype_mode=dtype_mode,
                embed_metadata=embed_metadata,
                config_output=dflash_config_output,
                workers=workers,
            )
        return Glm5ConvertReport(
            model_type="glm5_next" if "next" in arch.lower() else "glm_moe_dsa",
            backbone_amf=out_p,
            backbone_metas=tuple(metas),
            dflash2_amf=dflash_output if df_metas else None,
            dflash2_metas=tuple(df_metas),
            cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
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
    if is_glm5_dflash2_config(raw_cfg):
        df_metas = export_glm5_dflash2(
            in_p,
            out_p,
            block_size_override=dflash_block_size,
            model_id=eff_model_id,
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            target_bits=target_bits,
            use_rco=use_rco,
            embed_metadata=embed_metadata,
            config_output=config_output or dflash_config_output,
            workers=workers,
        )
        return Glm5ConvertReport(
            model_type="dflash2",
            backbone_amf=out_p,
            backbone_metas=tuple(df_metas),
            dflash2_amf=out_p,
            dflash2_metas=tuple(df_metas),
            config_json_path=config_output or dflash_config_output,
        )

    eff_cfg = _effective_glm5_cfg(raw_cfg)
    is_next = is_glm5_next_config(raw_cfg)
    mt_label = "glm5_next" if is_next else "glm_moe_dsa"
    raw_tensors = _collect_tensors(resolved.shard_paths)
    n_layers = int(eff_cfg.get("num_hidden_layers") or (45 if is_next else 78))
    n_exp = int(eff_cfg.get("n_routed_experts") or eff_cfg.get("num_local_experts") or 0)
    mlp_types = resolve_glm5_mlp_layer_types(eff_cfg)

    if (
        tier_prof is not None
        and not tier_prof.allow_nvme_cold_experts
        and not is_next
        and n_layers >= 60
        and n_exp >= 200
        and tier_prof.name in ("128gb-max", "256gb-studio")
    ):
        raise HardwareTierError(
            f"target_tier={tier_prof.name!r} ({tier_prof.wired_budget_gib:.0f} GiB wired budget) "
            "cannot wire full 753B GLM-5.2/5.3 (`glm_moe_dsa`); use '512gb-ultra' or enable cold experts"
        )

    cold_set = _select_glm5_cold_experts(
        n_layers=n_layers,
        n_experts=n_exp,
        mlp_types=mlp_types,
        tier_profile=tier_prof,
        cold_experts=cold_experts,
        cold_experts_per_layer=cold_experts_per_layer,
        heat_profile=heat_profile,
        all_experts_cold=all_experts_cold,
    )

    mtp_in_main = [k for k in raw_tensors if is_glm5_mtp_tensor(k, n_layers)]
    eff_target_bits = target_bits
    if tier_prof is not None and quant_profile == "gsq-rco" and eff_target_bits is None and target_wired_gib is None:
        eff_target_bits = tier_prof.target_bits_default
    eff_fine_rungs = (tier_prof is not None and quant_profile == "gsq-rco") if fine_rungs is None else fine_rungs

    vis_mode = resolve_vision_mode(
        raw_tensors.keys(),
        include_vision=include_vision,
        strip_vision=strip_vision,
        vision_sidecar=vision_sidecar,
    )
    runtime_cfg = (
        build_glm5_next_config_json(raw_cfg, cold_experts_count=len(cold_set), include_vision=(vis_mode != "strip"))
        if is_next
        else build_glm_moe_dsa_config_json(raw_cfg, cold_experts_count=len(cold_set))
    )
    runtime_cfg_bytes = json.dumps(runtime_cfg, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    if config_output is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_bytes(runtime_cfg_bytes)

    stats: dict[str, int] = {}
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
            glm5_config=raw_cfg,
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

    vis_metas: tuple[PackedTensorMeta, ...] = ()
    if vision_sidecar is not None and pathlib.Path(vision_sidecar).is_file():
        from aminfer_tools.amf import AmfReader

        with AmfReader(vision_sidecar) as vr:
            vis_metas = tuple(vr.entries)

    mtp_metas_out: list[PackedTensorMeta] = []
    if mtp_output is not None and mtp_in_main:
        mtp_metas_out = export_glm5_mtp(
            in_p,
            mtp_output,
            target_config=raw_cfg,
            model_id=f"{eff_model_id[:55]}-mtp",
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            target_bits=eff_target_bits,
            use_rco=use_rco,
            embed_metadata=embed_metadata,
            config_output=mtp_config_output,
            workers=workers,
        )

    df_metas_out: list[PackedTensorMeta] = []
    if dflash_output is not None and dflash_input is not None and pathlib.Path(dflash_input).exists():
        df_metas_out = export_glm5_dflash2(
            pathlib.Path(dflash_input),
            dflash_output,
            target_config=raw_cfg,
            block_size_override=dflash_block_size,
            model_id=f"{eff_model_id[:52]}-df2",
            dtype_mode=dtype_mode,
            quant_profile=quant_profile,
            target_bits=eff_target_bits,
            use_rco=use_rco,
            embed_metadata=embed_metadata,
            config_output=dflash_config_output,
            workers=workers,
        )

    return Glm5ConvertReport(
        model_type=mt_label,
        backbone_amf=out_p,
        backbone_metas=tuple(metas),
        mtp_amf=mtp_output if mtp_metas_out else None,
        mtp_metas=tuple(mtp_metas_out),
        dflash2_amf=dflash_output if df_metas_out else None,
        dflash2_metas=tuple(df_metas_out),
        vision_amf=vision_sidecar if vis_metas else None,
        vision_metas=vis_metas,
        cold_expert_count=sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT),
        stripped_vision_tensors=stats.get("stripped_vision_tensors", 0),
        config_json_path=config_output,
    )


# ---------------------------------------------------------------------------
# Deterministic Synthetic Generators (`synth-glm5`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Glm5SynthConfig:
    """Geometry preset for deterministic `synth-glm5` (`tiny-glm-dsa` / `tiny-glm5-next`) bundles."""

    preset: str = "tiny-glm-dsa"
    model_type: str = "glm_moe_dsa"
    model_id: str = "tiny-glm-dsa"
    num_hidden_layers: int = 6
    first_k_dense_replace: int = 1
    hidden_size: int = 128
    intermediate_size: int = 256
    moe_intermediate_size: int = 64
    vocab_size: int = 320
    num_attention_heads: int = 4
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    index_n_heads: int = 2
    index_head_dim: int = 32
    index_topk: int = 16
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 1
    index_kpool: int = 4
    linear_num_heads: int = 4
    linear_head_dim: int = 32
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float = -5.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    swiglu_limit: float = 10.0
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    cold_experts_per_layer: int = 2
    include_mtp: bool = True
    include_dflash2: bool = False
    dflash_layers: int = 2
    dflash_block_size: int = 8
    vision_layers: int = 0
    vision_hidden_size: int = 64
    vision_intermediate_size: int = 128
    vision_mode: str = "sidecar"
    expert_dtype: str = "bf16"
    dtype: str = "bf16"

    def validate(self) -> None:
        if self.model_type not in ("glm_moe_dsa", "glm5_next"):
            raise ValueError(f"unsupported model_type {self.model_type!r}")
        if self.num_hidden_layers < 2:
            raise ValueError("num_hidden_layers must be >= 2")
        if self.hidden_size % 64 != 0 or self.intermediate_size % 64 != 0 or self.moe_intermediate_size % 32 != 0:
            raise ValueError("hidden/intermediate/moe_intermediate must be multiples of 64/64/32")
        if self.dflash_block_size not in (8, 16):
            raise ValueError(f"dflash_block_size must be 8 or 16, got {self.dflash_block_size}")
        if self.vision_mode not in ("sidecar", "inline", "strip"):
            raise ValueError(f"vision_mode must be sidecar/inline/strip, got {self.vision_mode!r}")
        if self.expert_dtype not in ("mxfp4", "bf16", "f16"):
            raise ValueError(f"expert_dtype must be mxfp4/bf16/f16, got {self.expert_dtype!r}")
        if self.dtype not in ("bf16", "f16"):
            raise ValueError(f"dtype must be bf16/f16, got {self.dtype!r}")


GLM5_SYNTH_PRESETS: dict[str, Glm5SynthConfig] = {
    "tiny-glm-dsa": Glm5SynthConfig(),
    "tiny-glm5-next": Glm5SynthConfig(
        preset="tiny-glm5-next",
        model_type="glm5_next",
        model_id="tiny-glm5-next",
        num_hidden_layers=4,
        first_k_dense_replace=1,
        qk_rope_head_dim=0,
        index_topk_freq=1,
        index_skip_topk_offset=0,
        include_mtp=True,
        include_dflash2=True,
        vision_layers=2,
        vision_mode="sidecar",
    ),
    "tiny-glm5-next-b16": Glm5SynthConfig(
        preset="tiny-glm5-next-b16",
        model_type="glm5_next",
        model_id="tiny-glm5-next-b16",
        num_hidden_layers=4,
        first_k_dense_replace=1,
        qk_rope_head_dim=0,
        index_topk_freq=1,
        index_skip_topk_offset=0,
        include_mtp=True,
        include_dflash2=True,
        dflash_block_size=16,
        vision_layers=2,
        vision_mode="sidecar",
    ),
}


@dataclass(frozen=True)
class Glm5SynthBundle:
    """Outputs written by `generate_synth_glm5_bundle`."""

    output_dir: pathlib.Path
    model_type: str
    backbone_amf: pathlib.Path
    backbone_metas: tuple[PackedTensorMeta, ...]
    mtp_amf: pathlib.Path | None
    mtp_metas: tuple[PackedTensorMeta, ...]
    dflash2_amf: pathlib.Path | None
    dflash2_metas: tuple[PackedTensorMeta, ...]
    vision_amf: pathlib.Path | None
    vision_metas: tuple[PackedTensorMeta, ...]
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    total_bytes: int


def generate_synth_glm5_bundle(
    output_dir: pathlib.Path,
    *,
    config: Glm5SynthConfig | None = None,
    preset: str | None = None,
    num_layers: int | None = None,
    hidden_size: int | None = None,
    num_experts: int | None = None,
    vision_layers: int | None = None,
    vision_mode: str | None = None,
    dflash_block_size: int | None = None,
    dtype: str | None = None,
    cold_experts_per_layer: int | None = None,
    with_mtp: bool | None = None,
    with_dflash2: bool | None = None,
    model_id: str | None = None,
    seed: int = 42,
    embed_metadata: bool = True,
) -> Glm5SynthBundle:
    """Synthesize a deterministic, 16 KiB-aligned GLM-5 (`glm_moe_dsa` or `glm5_next`) `.amf` bundle."""
    base_cfg = config if config is not None else GLM5_SYNTH_PRESETS[preset or "tiny-glm-dsa"]
    overrides: dict[str, Any] = {}
    if num_layers is not None:
        overrides["num_hidden_layers"] = num_layers
    if hidden_size is not None:
        overrides["hidden_size"] = hidden_size
    if num_experts is not None:
        overrides["n_routed_experts"] = num_experts
    if vision_layers is not None:
        overrides["vision_layers"] = vision_layers
    if vision_mode is not None:
        overrides["vision_mode"] = vision_mode
    if dflash_block_size is not None:
        overrides["dflash_block_size"] = dflash_block_size
    if dtype is not None:
        overrides["dtype"] = dtype
    if cold_experts_per_layer is not None:
        overrides["cold_experts_per_layer"] = cold_experts_per_layer
    if with_mtp is not None:
        overrides["include_mtp"] = with_mtp
    if with_dflash2 is not None:
        overrides["include_dflash2"] = with_dflash2
    if model_id is not None:
        overrides["model_id"] = model_id
    if overrides:
        from dataclasses import replace

        cfg = replace(base_cfg, **overrides)
    else:
        cfg = base_cfg
    cfg.validate()
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    h = cfg.hidden_size
    inter = cfg.intermediate_size
    moe_inter = cfg.moe_intermediate_size
    vocab = cfg.vocab_size
    is_next = cfg.model_type == "glm5_next"
    fdt = DType.BF16 if cfg.dtype == "bf16" else DType.F16

    def _fbytes(vals: Sequence[float]) -> bytes:
        return f32_values_to_bf16_bytes(vals) if fdt == DType.BF16 else f32_values_to_f16_bytes(vals)

    def _add(recs: list[tuple], name: str, kind: TensorKind, dt: DType, layer: int, expert: int, shape: tuple[int, ...], data: bytes, qb: int = 0, gs: int = 0) -> None:
        assert len(name.encode("utf-8")) <= 47, f"name exceeds 47B: {name}"
        recs.append((name, kind, dt, qb, gs, layer, expert, shape, data, b""))

    if is_next:
        ltypes = ["linear_attention" if (i % 4 != 3) else "indexed_attention" for i in range(cfg.num_hidden_layers)]
        if all(x == "linear_attention" for x in ltypes):
            ltypes[-1] = "indexed_attention"
        idx_types = ["full"] * cfg.num_hidden_layers
        raw_cfg: dict[str, Any] = {
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "hidden_size": h,
                "intermediate_size": inter,
                "moe_intermediate_size": moe_inter,
                "num_hidden_layers": cfg.num_hidden_layers,
                "vocab_size": vocab,
                "first_k_dense_replace": cfg.first_k_dense_replace,
                "layer_types": ltypes,
                "hc_mult": cfg.hc_mult,
                "hc_sinkhorn_iters": cfg.hc_sinkhorn_iters,
                "hc_eps": 1e-6,
                "linear_num_heads": cfg.linear_num_heads,
                "linear_head_dim": cfg.linear_head_dim,
                "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
                "linear_lower_bound": cfg.linear_lower_bound,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_attention_heads,
                "q_lora_rank": cfg.q_lora_rank,
                "kv_lora_rank": cfg.kv_lora_rank,
                "qk_nope_head_dim": cfg.qk_nope_head_dim,
                "qk_rope_head_dim": 0,
                "v_head_dim": cfg.v_head_dim,
                "index_n_heads": cfg.index_n_heads,
                "index_head_dim": cfg.index_head_dim,
                "index_topk": cfg.index_topk,
                "index_kpool": cfg.index_kpool,
                "index_kpool_always_select_tail": True,
                "index_share_for_mtp_iteration": True,
                "n_routed_experts": cfg.n_routed_experts,
                "n_shared_experts": cfg.n_shared_experts,
                "num_experts_per_tok": cfg.num_experts_per_tok,
                "swiglu_limit": cfg.swiglu_limit,
                "num_nextn_predict_layers": 1 if cfg.include_mtp else 0,
            },
        }
        if cfg.vision_layers > 0:
            raw_cfg["vision_config"] = {
                "model_type": "glm5_next_vision",
                "depth": cfg.vision_layers,
                "hidden_size": cfg.vision_hidden_size,
                "intermediate_size": cfg.vision_intermediate_size,
                "num_heads": 4,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": h,
                "projection_intermediate_size": h * 2,
                "swiglu_limit": cfg.swiglu_limit,
            }
        rt_cfg = build_glm5_next_config_json(raw_cfg, cold_experts_count=0, include_vision=(cfg.vision_mode != "strip"))
    else:
        ltypes = ["indexed_attention"] * cfg.num_hidden_layers
        raw_cfg = {
            "model_type": "glm_moe_dsa",
            "hidden_size": h,
            "intermediate_size": inter,
            "moe_intermediate_size": moe_inter,
            "num_hidden_layers": cfg.num_hidden_layers,
            "vocab_size": vocab,
            "first_k_dense_replace": cfg.first_k_dense_replace,
            "index_topk_freq": cfg.index_topk_freq,
            "index_skip_topk_offset": cfg.index_skip_topk_offset,
            "index_share_for_mtp_iteration": True,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_attention_heads,
            "q_lora_rank": cfg.q_lora_rank,
            "kv_lora_rank": cfg.kv_lora_rank,
            "qk_nope_head_dim": cfg.qk_nope_head_dim,
            "qk_rope_head_dim": cfg.qk_rope_head_dim,
            "v_head_dim": cfg.v_head_dim,
            "index_n_heads": cfg.index_n_heads,
            "index_head_dim": cfg.index_head_dim,
            "index_topk": cfg.index_topk,
            "n_routed_experts": cfg.n_routed_experts,
            "n_shared_experts": cfg.n_shared_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "num_nextn_predict_layers": 1 if cfg.include_mtp else 0,
        }
        idx_types = resolve_glm_moe_dsa_indexer_types(raw_cfg)
        rt_cfg = build_glm_moe_dsa_config_json(raw_cfg)

    mlp_types = resolve_glm5_mlp_layer_types(raw_cfg)
    cold_start = max(0, cfg.n_routed_experts - cfg.cold_experts_per_layer)
    cold_set = {
        (l_idx, e_idx)
        for l_idx in range(cfg.num_hidden_layers)
        if mlp_types[l_idx] == "sparse"
        for e_idx in range(cold_start, cfg.n_routed_experts)
    }
    if cold_set:
        rt_cfg["cold_experts_count"] = len(cold_set)

    rt_cfg_b = json.dumps(rt_cfg, indent=2).encode("utf-8") + b"\n"
    tok_b = _synth_tokenizer_bytes(vocab)
    ct_b = _synth_chat_template_bytes(rt_cfg)
    (out_dir / "config.json").write_bytes(rt_cfg_b)
    (out_dir / "tokenizer.json").write_bytes(tok_b)

    def _emit_mla_and_idx(recs: list[tuple], l_idx: int, *, rope_hd: int, with_indexer: bool, with_kpool: bool) -> None:
        qh = cfg.num_attention_heads
        qk_hd = cfg.qk_nope_head_dim + rope_hd
        _add(recs, f"layers.{l_idx}.self_attn.q_a_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.q_lora_rank, h), _fbytes(_det_f32(rng, cfg.q_lora_rank * h)))
        _add(recs, f"layers.{l_idx}.self_attn.q_a_layernorm.weight", TensorKind.NORM, fdt, l_idx, NO_EXPERT, (cfg.q_lora_rank,), _fbytes(_det_f32(rng, cfg.q_lora_rank, std=0.02, mean=1.0)))
        _add(recs, f"layers.{l_idx}.self_attn.q_b_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (qh * qk_hd, cfg.q_lora_rank), _fbytes(_det_f32(rng, qh * qk_hd * cfg.q_lora_rank)))
        _add(recs, f"layers.{l_idx}.self_attn.kv_a_proj_with_mqa.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.kv_lora_rank + rope_hd, h), _fbytes(_det_f32(rng, (cfg.kv_lora_rank + rope_hd) * h)))
        _add(recs, f"layers.{l_idx}.self_attn.kv_a_layernorm.weight", TensorKind.NORM, fdt, l_idx, NO_EXPERT, (cfg.kv_lora_rank,), _fbytes(_det_f32(rng, cfg.kv_lora_rank, std=0.02, mean=1.0)))
        _add(recs, f"layers.{l_idx}.self_attn.kv_b_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (qh * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank), _fbytes(_det_f32(rng, qh * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank)))
        _add(recs, f"layers.{l_idx}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (h, qh * cfg.v_head_dim), _fbytes(_det_f32(rng, h * qh * cfg.v_head_dim)))
        if with_indexer:
            ix = f"layers.{l_idx}.attn.idx"
            _add(recs, f"{ix}.wq_b.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank), _fbytes(_det_f32(rng, cfg.index_n_heads * cfg.index_head_dim * cfg.q_lora_rank)))
            _add(recs, f"{ix}.wk.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.index_head_dim, h), _fbytes(_det_f32(rng, cfg.index_head_dim * h)))
            _add(recs, f"{ix}.k_norm.weight", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (cfg.index_head_dim,), f32_values_to_f32_bytes(_det_f32(rng, cfg.index_head_dim, std=0.02, mean=1.0)))
            _add(recs, f"{ix}.k_norm.bias", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (cfg.index_head_dim,), f32_values_to_f32_bytes(_det_f32(rng, cfg.index_head_dim, std=0.01)))
            _add(recs, f"{ix}.weights_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.index_n_heads, h), _fbytes(_det_f32(rng, cfg.index_n_heads * h)))
            if with_kpool:
                _add(recs, f"{ix}.kpool_ape", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.index_kpool, cfg.index_head_dim), _fbytes(_det_f32(rng, cfg.index_kpool * cfg.index_head_dim)))
                _add(recs, f"{ix}.kpool_gate", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (cfg.index_head_dim, h), _fbytes(_det_f32(rng, cfg.index_head_dim * h)))

    def _emit_moe_or_dense(recs: list[tuple], l_idx: int, is_sparse: bool) -> None:
        if not is_sparse:
            _add(recs, f"layers.{l_idx}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)))
            _add(recs, f"layers.{l_idx}.mlp.up_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)))
            _add(recs, f"layers.{l_idx}.mlp.down_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (h, inter), _fbytes(_det_f32(rng, h * inter)))
            return
        _add(recs, f"layers.{l_idx}.mlp.gate.weight", TensorKind.ROUTER, fdt, l_idx, NO_EXPERT, (cfg.n_routed_experts, h), _fbytes(_det_f32(rng, cfg.n_routed_experts * h)))
        _add(recs, f"layers.{l_idx}.mlp.gate.bias", TensorKind.ROUTER, DType.F32, l_idx, NO_EXPERT, (cfg.n_routed_experts,), f32_values_to_f32_bytes(_det_f32(rng, cfg.n_routed_experts, std=0.05)))
        sh_inter = moe_inter * cfg.n_shared_experts
        _add(recs, f"layers.{l_idx}.mlp.shared_experts.gate_proj.weight", TensorKind.SHARED_EXPERT, fdt, l_idx, NO_EXPERT, (sh_inter, h), _fbytes(_det_f32(rng, sh_inter * h)))
        _add(recs, f"layers.{l_idx}.mlp.shared_experts.up_proj.weight", TensorKind.SHARED_EXPERT, fdt, l_idx, NO_EXPERT, (sh_inter, h), _fbytes(_det_f32(rng, sh_inter * h)))
        _add(recs, f"layers.{l_idx}.mlp.shared_experts.down_proj.weight", TensorKind.SHARED_EXPERT, fdt, l_idx, NO_EXPERT, (h, sh_inter), _fbytes(_det_f32(rng, h * sh_inter)))
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
        if is_next:
            hc_rows = (2 + cfg.hc_mult) * cfg.hc_mult
            for sub in ("attn", "ffn"):
                _add(bb_recs, f"layers.{l_idx}.hc_{sub}_fn", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (hc_rows, cfg.hc_mult * h), _fbytes(_det_f32(rng, hc_rows * cfg.hc_mult * h)))
                _add(bb_recs, f"layers.{l_idx}.hc_{sub}_base", TensorKind.DENSE, DType.F32, l_idx, NO_EXPERT, (hc_rows,), f32_values_to_f32_bytes(_det_f32(rng, hc_rows, std=0.1)))
                _add(bb_recs, f"layers.{l_idx}.hc_{sub}_scale", TensorKind.DENSE, DType.F32, l_idx, NO_EXPERT, (3,), f32_values_to_f32_bytes([0.1, 0.1, 0.1]))
            if ltypes[l_idx] == "linear_attention":
                lh = cfg.linear_num_heads
                lhd = cfg.linear_head_dim
                qkv_dim = lh * lhd
                kern = cfg.linear_conv_kernel_dim
                for proj in ("q_proj", "k_proj", "v_proj"):
                    _add(bb_recs, f"layers.{l_idx}.linear_attn.{proj}.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (qkv_dim, h), _fbytes(_det_f32(rng, qkv_dim * h)))
                for conv_nm in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    _add(bb_recs, f"layers.{l_idx}.linear_attn.{conv_nm}.weight", TensorKind.DENSE, DType.F32, l_idx, NO_EXPERT, (qkv_dim, kern), f32_values_to_f32_bytes(_det_f32(rng, qkv_dim * kern)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.b_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (lh, h), _fbytes(_det_f32(rng, lh * h)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.f_a_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (lhd, h), _fbytes(_det_f32(rng, lhd * h)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.f_b_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (qkv_dim, lhd), _fbytes(_det_f32(rng, qkv_dim * lhd)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.g_a_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (lhd, h), _fbytes(_det_f32(rng, lhd * h)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.g_b_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (qkv_dim, lhd), _fbytes(_det_f32(rng, qkv_dim * lhd)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.o_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (h, qkv_dim), _fbytes(_det_f32(rng, h * qkv_dim)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.o_norm.weight", TensorKind.NORM, fdt, l_idx, NO_EXPERT, (lhd,), _fbytes(_det_f32(rng, lhd, std=0.02, mean=1.0)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.A_log", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (lh,), f32_values_to_f32_bytes(_det_f32(rng, lh, std=0.1, mean=-2.0)))
                _add(bb_recs, f"layers.{l_idx}.linear_attn.dt_bias", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (qkv_dim,), f32_values_to_f32_bytes(_det_f32(rng, qkv_dim, std=0.1)))
            else:
                _emit_mla_and_idx(bb_recs, l_idx, rope_hd=0, with_indexer=True, with_kpool=True)
        else:
            _emit_mla_and_idx(bb_recs, l_idx, rope_hd=cfg.qk_rope_head_dim, with_indexer=(idx_types[l_idx] == "full"), with_kpool=False)
        _emit_moe_or_dense(bb_recs, l_idx, mlp_types[l_idx] == "sparse")

    vis_recs: list[tuple] = []
    if cfg.vision_layers > 0 and cfg.vision_mode != "strip":
        vh = cfg.vision_hidden_size
        vi = cfg.vision_intermediate_size
        vhd = vh // 2 if vh >= 16 and (vh // 2) % 4 == 0 else vh
        _add(vis_recs, "visual.patch_embed.proj.weight", TensorKind.DENSE, fdt, NO_LAYER, NO_EXPERT, (vh, 3 * 2 * 14 * 14), _fbytes(_det_f32(rng, vh * 3 * 2 * 14 * 14)))
        _add(vis_recs, "visual.patch_embed.proj.bias", TensorKind.NORM, fdt, NO_LAYER, NO_EXPERT, (vh,), _fbytes(_det_f32(rng, vh, std=0.01)))
        for vl in range(cfg.vision_layers):
            _add(vis_recs, f"visual.blocks.{vl}.norm1.weight", TensorKind.NORM, fdt, vl, NO_EXPERT, (vh,), _fbytes(_det_f32(rng, vh, std=0.02, mean=1.0)))
            _add(vis_recs, f"visual.blocks.{vl}.norm2.weight", TensorKind.NORM, fdt, vl, NO_EXPERT, (vh,), _fbytes(_det_f32(rng, vh, std=0.02, mean=1.0)))
            _add(vis_recs, f"visual.blocks.{vl}.attn.q_norm.weight", TensorKind.NORM, fdt, vl, NO_EXPERT, (vhd,), _fbytes(_det_f32(rng, vhd, std=0.02, mean=1.0)))
            _add(vis_recs, f"visual.blocks.{vl}.attn.k_norm.weight", TensorKind.NORM, fdt, vl, NO_EXPERT, (vhd,), _fbytes(_det_f32(rng, vhd, std=0.02, mean=1.0)))
            _add(vis_recs, f"visual.blocks.{vl}.attn.qkv.weight", TensorKind.DENSE, fdt, vl, NO_EXPERT, (3 * vh, vh), _fbytes(_det_f32(rng, 3 * vh * vh)))
            _add(vis_recs, f"visual.blocks.{vl}.attn.proj.weight", TensorKind.DENSE, fdt, vl, NO_EXPERT, (vh, vh), _fbytes(_det_f32(rng, vh * vh)))
            _add(vis_recs, f"visual.blocks.{vl}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, vl, NO_EXPERT, (vi, vh), _fbytes(_det_f32(rng, vi * vh)))
            _add(vis_recs, f"visual.blocks.{vl}.mlp.up_proj.weight", TensorKind.DENSE, fdt, vl, NO_EXPERT, (vi, vh), _fbytes(_det_f32(rng, vi * vh)))
            _add(vis_recs, f"visual.blocks.{vl}.mlp.down_proj.weight", TensorKind.DENSE, fdt, vl, NO_EXPERT, (vh, vi), _fbytes(_det_f32(rng, vh * vi)))
        _add(vis_recs, "visual.post_layernorm.weight", TensorKind.NORM, fdt, NO_LAYER, NO_EXPERT, (vh,), _fbytes(_det_f32(rng, vh, std=0.02, mean=1.0)))
        _add(vis_recs, "visual.downsample.weight", TensorKind.DENSE, fdt, NO_LAYER, NO_EXPERT, (h, vh, 2, 2), _fbytes(_det_f32(rng, h * vh * 4)))
        _add(vis_recs, "visual.merger.proj.weight", TensorKind.DENSE, fdt, NO_LAYER, NO_EXPERT, (h, h), _fbytes(_det_f32(rng, h * h)))
        _add(vis_recs, "visual.merger.post_projection_norm.weight", TensorKind.NORM, fdt, NO_LAYER, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(vis_recs, "visual.merger.gate_proj.weight", TensorKind.DENSE, fdt, NO_LAYER, NO_EXPERT, (2 * h, h), _fbytes(_det_f32(rng, 2 * h * h)))
        _add(vis_recs, "visual.merger.up_proj.weight", TensorKind.DENSE, fdt, NO_LAYER, NO_EXPERT, (2 * h, h), _fbytes(_det_f32(rng, 2 * h * h)))
        _add(vis_recs, "visual.merger.down_proj.weight", TensorKind.DENSE, fdt, NO_LAYER, NO_EXPERT, (h, 2 * h), _fbytes(_det_f32(rng, h * 2 * h)))

    if cfg.vision_mode == "inline" and vis_recs:
        bb_recs.extend(vis_recs)

    if embed_metadata:
        bb_recs.extend([
            make_metadata_record(META_CONFIG_JSON, rt_cfg_b),
            make_metadata_record(META_TOKENIZER_JSON, tok_b),
            make_metadata_record(META_CHAT_TEMPLATE_JSON, ct_b),
        ])

    bb_amf = out_dir / "model.amf"
    bb_metas = write_amf(bb_amf, cfg.model_id[:63], bb_recs)
    total_bytes = bb_amf.stat().st_size

    vis_amf: pathlib.Path | None = None
    vis_metas: list[PackedTensorMeta] = []
    if cfg.vision_mode == "sidecar" and vis_recs:
        vis_cfg = json.dumps({"vision_config": rt_cfg.get("vision_config", {}), "model_type": "glm5_next_vision"}, indent=2).encode("utf-8") + b"\n"
        if embed_metadata:
            vis_recs.append(make_metadata_record(META_CONFIG_JSON, vis_cfg))
        vis_amf = out_dir / "vision.amf"
        vis_metas = write_amf(vis_amf, f"{cfg.model_id[:55]}-vision", vis_recs)
        total_bytes += vis_amf.stat().st_size

    mtp_amf: pathlib.Path | None = None
    mtp_metas: list[PackedTensorMeta] = []
    if cfg.include_mtp:
        mtp_cfg = build_glm5_mtp_config_dict(raw_cfg, is_glm5_next=is_next)
        mtp_cfg_b = json.dumps(mtp_cfg, indent=2).encode("utf-8") + b"\n"
        (out_dir / "mtp_config.json").write_bytes(mtp_cfg_b)
        mtp_recs: list[tuple] = []
        _add(mtp_recs, "layers.0.enorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(mtp_recs, "layers.0.hnorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(mtp_recs, "layers.0.eh_proj.weight", TensorKind.DENSE, fdt, 0, NO_EXPERT, (h, 2 * h), _fbytes(_det_f32(rng, h * 2 * h)))
        _add(mtp_recs, "shared_head.norm.weight", TensorKind.NORM, fdt, NO_LAYER, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(mtp_recs, "layers.0.input_layernorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(mtp_recs, "layers.0.post_attention_layernorm.weight", TensorKind.NORM, fdt, 0, NO_EXPERT, (h,), _fbytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _emit_mla_and_idx(mtp_recs, 0, rope_hd=(0 if is_next else cfg.qk_rope_head_dim), with_indexer=True, with_kpool=is_next)
        _emit_moe_or_dense(mtp_recs, 0, True)
        if embed_metadata:
            mtp_recs.append(make_metadata_record(META_CONFIG_JSON, mtp_cfg_b))
        mtp_amf = out_dir / "mtp.amf"
        mtp_metas = write_amf(mtp_amf, f"{cfg.model_id[:55]}-mtp", mtp_recs)
        total_bytes += mtp_amf.stat().st_size

    df_amf: pathlib.Path | None = None
    df_metas: list[PackedTensorMeta] = []
    if cfg.include_dflash2:
        taps = [0, 1, 2, 3][: min(cfg.num_hidden_layers, 4)]
        df_src = {
            "hidden_size": h,
            "intermediate_size": inter,
            "num_hidden_layers": cfg.dflash_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": vocab,
            "dflash_config": {
                "block_size": cfg.dflash_block_size,
                "mask_token_id": 3,
                "target_layer_ids": taps,
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 64,
                "selector_top_k": 8,
            },
        }
        df_cfg = build_glm5_dflash2_config_dict(df_src, target_cfg=raw_cfg, block_size_override=cfg.dflash_block_size)
        df_cfg_b = json.dumps(df_cfg, indent=2).encode("utf-8") + b"\n"
        (out_dir / "dflash2_config.json").write_bytes(df_cfg_b)
        df_hd = 64
        df_qh = cfg.num_attention_heads
        df_kvh = 2
        sel_r = 64
        conv_g = 16
        kp_rows = 2 * 2 * (h // conv_g)
        df_recs: list[tuple] = []
        _add(df_recs, "fc.weight", TensorKind.DENSE, fdt, NO_LAYER, NO_EXPERT, (h, len(taps) * h), _fbytes(_det_f32(rng, h * len(taps) * h)))
        _add(df_recs, "hidden_norm.weight", TensorKind.NORM, DType.F32, NO_LAYER, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(df_recs, "norm.weight", TensorKind.NORM, DType.F32, NO_LAYER, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)))
        _add(df_recs, "candidate_selector.hidden_projection.weight", TensorKind.DENSE, fdt, NO_LAYER, NO_EXPERT, (sel_r, h), _fbytes(_det_f32(rng, sel_r * h)))
        _add(df_recs, "candidate_selector.predecessor_codebook", TensorKind.EMBEDDING, fdt, NO_LAYER, NO_EXPERT, (vocab, sel_r), _fbytes(_det_f32(rng, vocab * sel_r)))
        _add(df_recs, "candidate_selector.successor_codebook", TensorKind.EMBEDDING, fdt, NO_LAYER, NO_EXPERT, (vocab, sel_r), _fbytes(_det_f32(rng, vocab * sel_r)))
        for l_idx in range(cfg.dflash_layers):
            _add(df_recs, f"layers.{l_idx}.input_layernorm.weight", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)))
            _add(df_recs, f"layers.{l_idx}.post_attention_layernorm.weight", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (h,), f32_values_to_f32_bytes(_det_f32(rng, h, std=0.02, mean=1.0)))
            _add(df_recs, f"layers.{l_idx}.self_attn.q_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (df_qh * df_hd, h), _fbytes(_det_f32(rng, df_qh * df_hd * h)))
            _add(df_recs, f"layers.{l_idx}.self_attn.k_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (df_kvh * df_hd, h), _fbytes(_det_f32(rng, df_kvh * df_hd * h)))
            _add(df_recs, f"layers.{l_idx}.self_attn.v_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (df_kvh * df_hd, h), _fbytes(_det_f32(rng, df_kvh * df_hd * h)))
            _add(df_recs, f"layers.{l_idx}.self_attn.o_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (h, df_qh * df_hd), _fbytes(_det_f32(rng, h * df_qh * df_hd)))
            _add(df_recs, f"layers.{l_idx}.self_attn.q_norm.weight", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (df_hd,), f32_values_to_f32_bytes(_det_f32(rng, df_hd, std=0.02, mean=1.0)))
            _add(df_recs, f"layers.{l_idx}.self_attn.k_norm.weight", TensorKind.NORM, DType.F32, l_idx, NO_EXPERT, (df_hd,), f32_values_to_f32_bytes(_det_f32(rng, df_hd, std=0.02, mean=1.0)))
            _add(df_recs, f"layers.{l_idx}.attn_conv.base_kernel", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (2, 2, h), _fbytes(_det_f32(rng, 4 * h)))
            _add(df_recs, f"layers.{l_idx}.attn_conv.kernel_projection.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (kp_rows, h), _fbytes(_det_f32(rng, kp_rows * h)))
            _add(df_recs, f"layers.{l_idx}.mlp_conv.base_kernel", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (2, 2, h), _fbytes(_det_f32(rng, 4 * h)))
            _add(df_recs, f"layers.{l_idx}.mlp_conv.kernel_projection.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (kp_rows, h), _fbytes(_det_f32(rng, kp_rows * h)))
            _add(df_recs, f"layers.{l_idx}.mlp.gate_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)))
            _add(df_recs, f"layers.{l_idx}.mlp.up_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (inter, h), _fbytes(_det_f32(rng, inter * h)))
            _add(df_recs, f"layers.{l_idx}.mlp.down_proj.weight", TensorKind.DENSE, fdt, l_idx, NO_EXPERT, (h, inter), _fbytes(_det_f32(rng, h * inter)))
        if embed_metadata:
            df_recs.append(make_metadata_record(META_CONFIG_JSON, df_cfg_b))
        df_amf = out_dir / "dflash2.amf"
        df_metas = write_amf(df_amf, f"{cfg.model_id[:52]}-df2", df_recs)
        total_bytes += df_amf.stat().st_size

    return Glm5SynthBundle(
        output_dir=out_dir,
        model_type=cfg.model_type,
        backbone_amf=bb_amf,
        backbone_metas=tuple(bb_metas),
        mtp_amf=mtp_amf,
        mtp_metas=tuple(mtp_metas),
        dflash2_amf=df_amf,
        dflash2_metas=tuple(df_metas),
        vision_amf=vis_amf,
        vision_metas=tuple(vis_metas),
        config_path=out_dir / "config.json",
        tokenizer_path=out_dir / "tokenizer.json",
        total_bytes=total_bytes,
    )
