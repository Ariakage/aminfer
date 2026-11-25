"""Offline Inkling 2 (`inkling2`, `inkling2_loop`) converter, DFlare packer, GSQ-RCO v2 tier profiles, and synth CLI (W23, D14, C26).

Supports:
- `thinkingmachines/Inkling-2-Mini` (`24` physical layers, `LoopSchedule` `core_start=6, core_end=18,
  num_passes=3` -> `48` effective steps, `128` experts top-8, `intermediate_size=1152`, `20/24` SWA `1024`,
  `k=4` `attn_sconv`/`mlp_sconv` + global dynamic `weight_proj`, `d_rel=16` `rel_logits_proj`, `q/rel_l2_clip`,
  `log_scaling`, `attn_sink_proj`, `gate_attn`, 6-layer `attn_res`, `4`/`25`/`30` softcaps, router `bias + route_scale`,
  `middle_cycle_cross_pass_{kv_}queries`, `gate_rank=64` `controllers.*`, native `NVFP4` + `MXFP8`, `23L` ViT, `3L` `DFlare`),
- `mlx-community/Inkling-2-Mini-4bit` (`4-bit g64` affine + `NVFP4` + `*_message_gate` + `post_*_convolution`),
- `thinkingmachines/Inkling-2-Small` (`48L`, `384E`, `40/48` SWA `1024`, `5L` `DFlare`, `23L` ViT),
- `unsloth/Inkling-2-Mini-GGUF` (`arch=inkling2` loop deduplication + `arch=dflash` `DFlare` bundle export).
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import re
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DTYPE_MODES,
    META_CHAT_TEMPLATE_JSON,
    META_CONFIG_JSON,
    META_TOKENIZER_JSON,
    NO_EXPERT,
    NO_LAYER,
    QUANT_PROFILES,
    SME2_GROUP_K,
    SME2_PACK_MODES,
    SME2_PANEL_COLS,
    VISION_BLOCK_DTYPES,
    AmfReader,
    DType,
    PackedTensorMeta,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    _align_up,
    _collect_tensors,
    _natural_key,
    _normalize_name,
    _ShardReaders,
    _ShardTensor,
    _write_planned_amf,
    embed_metadata_into_amf,
    f32_to_bf16_bytes,
    f32_to_f16_bytes,
    is_inkling2_controller_tensor,
    is_vision_tensor,
    load_cold_experts_from_heat_profile,
    pack_amf_from_safetensors,
    pack_sme2_panel64_f16_panel,
    pack_vision_tower_amf,
    plan_amf_entries,
    resolve_checkpoint,
    resolve_vision_mode,
    sme2_panel64_data_bytes,
    write_amf,
)
from aminfer_tools.chat_template import compile_chat_template_spec
from aminfer_tools.gguf import export_dflare_bundle_from_gguf, pack_amf_from_gguf

INKLING2_MODEL_TYPES = ("inkling2", "inkling2_loop", "inkling2_loop_text", "inkling2_text")
VALID_INKLING2_TIERS = (
    "auto",
    "24gb-m4",
    "24gb-m4-full-wired",
    "128gb-max",
    "256gb-studio",
    "512gb-ultra",
)

# Wired budgets mirror ADR-0012 (24 GB M4 = 18,186 MiB = 17.76 GiB) and Mac Studio/Ultra ceilings.
INKLING2_TIER_WIRED_BUDGET_BYTES: dict[str, int] = {
    "24gb-m4": 18_186 * 1024 * 1024,
    "24gb-m4-full-wired": int(10.8 * (1024**3)),
    "128gb-max": 108 * 1024 * 1024 * 1024,
    "256gb-studio": 218 * 1024 * 1024 * 1024,
    "512gb-ultra": 440 * 1024 * 1024 * 1024,
}


@dataclass(frozen=True)
class Inkling2TierProfile:
    """GSQ-RCO v2 tier-down profile for Inkling 2 (`Mini` and `Small`)."""

    tier: str
    wired_budget_bytes: int
    target_bits: float
    max_q2_share: float
    cold_tail_fraction: float
    vision_sidecar_recommended: bool
    enforce_backbone_floor: bool = True
    fine_rungs: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "wired_budget_gib": round(self.wired_budget_bytes / (1024**3), 2),
            "target_bits": self.target_bits,
            "max_q2_share": self.max_q2_share,
            "cold_tail_fraction": self.cold_tail_fraction,
            "vision_sidecar_recommended": self.vision_sidecar_recommended,
            "enforce_backbone_floor": self.enforce_backbone_floor,
            "fine_rungs": self.fine_rungs,
        }


INKLING2_GSQ_RCO_PROFILES: dict[str, Inkling2TierProfile] = {
    "24gb-m4": Inkling2TierProfile(
        tier="24gb-m4",
        wired_budget_bytes=INKLING2_TIER_WIRED_BUDGET_BYTES["24gb-m4"],
        target_bits=2.55,
        max_q2_share=0.78,
        cold_tail_fraction=0.75,
        vision_sidecar_recommended=True,
    ),
    "24gb-m4-full-wired": Inkling2TierProfile(
        tier="24gb-m4-full-wired",
        wired_budget_bytes=INKLING2_TIER_WIRED_BUDGET_BYTES["24gb-m4-full-wired"],
        target_bits=3.42,
        max_q2_share=0.30,
        cold_tail_fraction=0.0,
        vision_sidecar_recommended=True,
        enforce_backbone_floor=True,
        fine_rungs=True,
    ),
    "128gb-max": Inkling2TierProfile(
        tier="128gb-max",
        wired_budget_bytes=INKLING2_TIER_WIRED_BUDGET_BYTES["128gb-max"],
        target_bits=3.25,
        max_q2_share=0.35,
        cold_tail_fraction=0.25,
        vision_sidecar_recommended=True,
    ),
    "256gb-studio": Inkling2TierProfile(
        tier="256gb-studio",
        wired_budget_bytes=INKLING2_TIER_WIRED_BUDGET_BYTES["256gb-studio"],
        target_bits=3.65,
        max_q2_share=0.10,
        cold_tail_fraction=0.0,
        vision_sidecar_recommended=False,
    ),
    "512gb-ultra": Inkling2TierProfile(
        tier="512gb-ultra",
        wired_budget_bytes=INKLING2_TIER_WIRED_BUDGET_BYTES["512gb-ultra"],
        target_bits=4.50,
        max_q2_share=0.0,
        cold_tail_fraction=0.0,
        vision_sidecar_recommended=False,
    ),
}


def resolve_inkling2_tier_profile(
    tier: str,
    *,
    num_hidden_layers: int = 48,
    n_routed_experts: int = 384,
) -> Inkling2TierProfile:
    """Resolve `Inkling-2-Mini` / `Inkling-2-Small` `GSQ-RCO v2` tier-down policy."""
    norm_tier = "256gb-studio" if tier == "256gb-ultra" else tier
    if norm_tier == "auto":
        norm_tier = "24gb-m4"
    if norm_tier not in INKLING2_GSQ_RCO_PROFILES:
        raise ValueError(f"unsupported Inkling 2 tier {tier!r}; expected one of {VALID_INKLING2_TIERS}")
    base = INKLING2_GSQ_RCO_PROFILES[norm_tier]
    # Mini (24L, 128E) on 24 GB M4 needs only ~12.5% coldest experts on NVMe + high backbone precision.
    if norm_tier == "24gb-m4" and num_hidden_layers <= 24 and n_routed_experts <= 128:
        return Inkling2TierProfile(
            tier="24gb-m4",
            wired_budget_bytes=base.wired_budget_bytes,
            target_bits=3.50,
            max_q2_share=0.25,
            cold_tail_fraction=0.125,
            vision_sidecar_recommended=True,
            enforce_backbone_floor=True,
            fine_rungs=True,
        )
    return base


def _effective_text_config(cfg: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(cfg, dict):
        return {}
    out = dict(cfg)
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        out.update(tc)
    return out


def _is_inkling2_mtp_tensor(raw_name: str) -> bool:
    norm = raw_name
    for prefix in ("model.mtp.", "mtp.", "model.dflare.", "dflare."):
        if norm.startswith(prefix):
            return True
    return False


def _strip_inkling2_mtp_prefix(raw_name: str) -> str:
    for prefix in ("model.mtp.", "mtp.", "model.dflare.", "dflare."):
        if raw_name.startswith(prefix):
            return raw_name[len(prefix) :]
    return raw_name


def _canonicalize_inkling2_dict_name(raw_name: str) -> str:
    return _normalize_name(raw_name)


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


def _tail_cold_experts(num_layers: int, num_experts: int, per_layer: int) -> set[tuple[int, int]]:
    if per_layer <= 0 or num_experts <= 0:
        return set()
    take = min(per_layer, num_experts)
    start = num_experts - take
    return {(l, e) for l in range(num_layers) for e in range(start, num_experts)}


@dataclass(frozen=True)
class Inkling2MetadataReport:
    """Offline HF metadata inspection summary for Inkling 2 (`Mini` and `Small`)."""

    model_type: str
    num_hidden_layers: int
    effective_num_steps: int
    hidden_size: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sconv_kernel_size: int
    sliding_window_size: int
    local_layer_count: int
    global_layer_count: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    route_scale: float
    mlp_gate_softcap: float
    mlp_up_softcap: float
    final_logit_softcapping: float
    loop_core_start: int
    loop_core_end: int
    loop_num_passes: int
    loop_gate_rank: int
    dflare_layers: int
    dflare_block_size: int
    dflare_markov_rank: int
    vision_layers: int
    total_index_entries: int
    shard_file_count: int
    index_total_size_bytes: int
    controller_tensor_count: int
    sconv_tensor_count: int
    attn_res_tensor_count: int
    vision_tensor_count: int
    nvfp4_tensor_count: int
    mxfp8_tensor_count: int
    long_backbone_names: tuple[str, ...]
    tier_profiles: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "arch": self.model_type,
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "effective_num_steps": self.effective_num_steps,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "unpadded_vocab_size": self.unpadded_vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "d_rel": self.d_rel,
            "rel_extent": self.rel_extent,
            "sconv_kernel_size": self.sconv_kernel_size,
            "sliding_window_size": self.sliding_window_size,
            "local_layer_count": self.local_layer_count,
            "global_layer_count": self.global_layer_count,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "intermediate_size": self.intermediate_size,
            "route_scale": self.route_scale,
            "softcaps": {
                "mlp_gate": self.mlp_gate_softcap,
                "mlp_up": self.mlp_up_softcap,
                "final_logit": self.final_logit_softcapping,
            },
            "loop_config": {
                "core_start": self.loop_core_start,
                "core_end": self.loop_core_end,
                "num_passes": self.loop_num_passes,
                "gate_rank": self.loop_gate_rank,
            },
            "dflare": {
                "num_hidden_layers": self.dflare_layers,
                "block_size": self.dflare_block_size,
                "markov_rank": self.dflare_markov_rank,
            },
            "vision_layers": self.vision_layers,
            "index": {
                "total_entries": self.total_index_entries,
                "shard_files": self.shard_file_count,
                "total_size_bytes": self.index_total_size_bytes,
                "controllers": self.controller_tensor_count,
                "sconv": self.sconv_tensor_count,
                "attn_res": self.attn_res_tensor_count,
                "vision": self.vision_tensor_count,
                "nvfp4": self.nvfp4_tensor_count,
                "mxfp8": self.mxfp8_tensor_count,
                "long_backbone_names": list(self.long_backbone_names),
            },
            "tier_profiles": self.tier_profiles,
        }


def validate_hf_metadata_dir(
    metadata_dir: pathlib.Path,
    *,
    host_ram_gib: float | None = None,
) -> Inkling2MetadataReport:
    """Dry-run validate an Inkling 2 HF metadata directory (`config.json` + optional index/quant)."""
    d = pathlib.Path(metadata_dir)
    cfg_path = d / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json in {d}")
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"config.json in {d} must be a JSON object")

    tc = _effective_text_config(raw_cfg)
    mt = str(raw_cfg.get("model_type") or tc.get("model_type") or "inkling2").lower()
    if mt not in INKLING2_MODEL_TYPES:
        raise ValueError(f"expected Inkling 2 model_type in {INKLING2_MODEL_TYPES}, got {mt!r}")

    num_layers = int(tc.get("num_hidden_layers", 24) or 24)
    loop_cfg = tc.get("loop_config") if isinstance(tc.get("loop_config"), dict) else {}
    core_start = int(loop_cfg.get("core_start", 0) or 0)
    core_end = int(loop_cfg.get("core_end", 0) or 0)
    num_passes = int(loop_cfg.get("num_passes", 1) or 1)
    gate_rank = int(loop_cfg.get("gate_rank", 0) or 0)
    if core_end > core_start and num_passes > 1:
        eff_steps = num_layers + (core_end - core_start) * (num_passes - 1)
    else:
        eff_steps = num_layers

    local_ids = tc.get("local_layer_ids") if isinstance(tc.get("local_layer_ids"), list) else []
    local_count = len(local_ids)
    global_count = max(0, num_layers - local_count)

    mtp_cfg = raw_cfg.get("mtp_config") if isinstance(raw_cfg.get("mtp_config"), dict) else {}
    vis_cfg = raw_cfg.get("vision_config") if isinstance(raw_cfg.get("vision_config"), dict) else {}

    idx_path = d / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    total_size = 0
    if idx_path.is_file():
        idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
        weight_map = idx_obj.get("weight_map") or {}
        total_size = int((idx_obj.get("metadata") or {}).get("total_size", 0) or 0)

    hf_qc_path = d / "hf_quant_config.json"
    nvfp4_cnt = 0
    mxfp8_cnt = 0
    if hf_qc_path.is_file():
        hq = json.loads(hf_qc_path.read_text(encoding="utf-8"))
        for grp in (hq.get("model") or {}).get("parameter_groups") or []:
            cdt = str(grp.get("compute_dtype") or "").lower()
            tlist = grp.get("tensors") or []
            if cdt == "nvfp4":
                nvfp4_cnt += len(tlist)
            elif cdt == "mxfp8":
                mxfp8_cnt += len(tlist)

    ctrl_cnt = 0
    sconv_cnt = 0
    attn_res_cnt = 0
    vis_cnt = 0
    long_names: list[str] = []

    for raw_k in weight_map:
        if is_vision_tensor(raw_k):
            vis_cnt += 1
            norm_v = _normalize_name(raw_k)
            if len(norm_v.encode("utf-8")) > 47:
                long_names.append(norm_v)
            continue
        if raw_k.endswith(("_outer_scale_denominator", ".weight_scale", ".weight_amax")):
            continue
        norm = _normalize_name(raw_k)
        if len(norm.encode("utf-8")) > 47:
            long_names.append(norm)
        if is_inkling2_controller_tensor(norm):
            ctrl_cnt += 1
        if "sconv" in norm:
            sconv_cnt += 1
        if "attn_res." in norm:
            attn_res_cnt += 1

    h_sz = int(tc.get("hidden_size", 2304) or 2304)
    n_exp = int(tc.get("n_routed_experts", tc.get("num_experts", 128)) or 128)
    inter = int(tc.get("intermediate_size", 1152) or 1152)
    tier_info: dict[str, dict[str, object]] = {}
    for t_name in ("24gb-m4", "24gb-m4-full-wired", "128gb-max", "256gb-studio", "512gb-ultra"):
        prof = resolve_inkling2_tier_profile(t_name, num_hidden_layers=num_layers, n_routed_experts=n_exp)
        d_obj = prof.to_dict()
        # Estimate always-active + cold/hot expert split for quick admission summary.
        cold_exp = int(round(n_exp * prof.cold_tail_fraction))
        hot_exp = max(0, n_exp - cold_exp)
        bpw = prof.target_bits
        hot_bytes = int(num_layers * hot_exp * 3 * inter * h_sz * bpw / 8.0)
        dense_est = int(num_layers * (8 * h_sz * h_sz) * 4.25 / 8.0)
        d_obj["estimated_wired_gib"] = round((hot_bytes + dense_est) / (1024**3), 2)
        d_obj["cold_experts_per_layer_recommended"] = cold_exp
        if host_ram_gib is not None and t_name == "24gb-m4":
            d_obj["host_ram_gib"] = host_ram_gib
        tier_info[t_name] = d_obj

    return Inkling2MetadataReport(
        model_type=mt,
        num_hidden_layers=num_layers,
        effective_num_steps=eff_steps,
        hidden_size=h_sz,
        vocab_size=int(tc.get("vocab_size", 201024) or 201024),
        unpadded_vocab_size=int(tc.get("unpadded_vocab_size", 200058) or 200058),
        num_attention_heads=int(tc.get("num_attention_heads", 24) or 24),
        num_key_value_heads=int(tc.get("num_key_value_heads", 8) or 8),
        head_dim=int(tc.get("head_dim", 128) or 128),
        d_rel=int(tc.get("d_rel", 16) or 16),
        rel_extent=int(tc.get("rel_extent", 1024) or 1024),
        sconv_kernel_size=int(tc.get("sconv_kernel_size", 4) or 4),
        sliding_window_size=int(tc.get("sliding_window_size", 1024) or 1024),
        local_layer_count=local_count,
        global_layer_count=global_count,
        n_routed_experts=n_exp,
        num_experts_per_tok=int(tc.get("num_experts_per_tok", 8) or 8),
        intermediate_size=inter,
        route_scale=float(tc.get("route_scale", 8.0) or 8.0),
        mlp_gate_softcap=float(tc.get("mlp_gate_softcap", 4.0) or 4.0),
        mlp_up_softcap=float(tc.get("mlp_up_softcap", 25.0) or 25.0),
        final_logit_softcapping=float(tc.get("final_logit_softcapping", 30.0) or 30.0),
        loop_core_start=core_start,
        loop_core_end=core_end,
        loop_num_passes=num_passes,
        loop_gate_rank=gate_rank,
        dflare_layers=int(mtp_cfg.get("num_hidden_layers", 0) or 0),
        dflare_block_size=int(mtp_cfg.get("block_size", 8) or 8),
        dflare_markov_rank=int(mtp_cfg.get("markov_rank", 256) or 256),
        vision_layers=int(vis_cfg.get("num_hidden_layers", 0) or 0),
        total_index_entries=len(weight_map),
        shard_file_count=len(set(weight_map.values())),
        index_total_size_bytes=total_size,
        controller_tensor_count=ctrl_cnt,
        sconv_tensor_count=sconv_cnt,
        attn_res_tensor_count=attn_res_cnt,
        vision_tensor_count=vis_cnt,
        nvfp4_tensor_count=nvfp4_cnt,
        mxfp8_tensor_count=mxfp8_cnt,
        long_backbone_names=tuple(sorted(set(long_names))),
        tier_profiles=tier_info,
    )


def build_dflare_config_dict(src_config: dict[str, object] | None) -> dict[str, object]:
    """Build canonical `D14` `dflare_config.json` from Inkling 2 `config.json` (`mtp_config`)."""
    raw = src_config if isinstance(src_config, dict) else {}
    tc = _effective_text_config(raw)
    mtp = raw.get("mtp_config") if isinstance(raw.get("mtp_config"), dict) else {}
    return {
        "draft_kind": "dflare",
        "model_type": "dflare",
        "num_hidden_layers": int(mtp.get("num_hidden_layers", 3) or 3),
        "hidden_size": int(mtp.get("hidden_size", tc.get("hidden_size", 2304)) or 2304),
        "num_attention_heads": int(mtp.get("num_attention_heads", tc.get("num_attention_heads", 24)) or 24),
        "num_key_value_heads": int(mtp.get("num_key_value_heads", tc.get("num_key_value_heads", 8)) or 8),
        "head_dim": int(mtp.get("head_dim", tc.get("head_dim", 128)) or 128),
        "intermediate_size": int(mtp.get("intermediate_size", 9216) or 9216),
        "rms_norm_eps": float(mtp.get("rms_norm_eps", tc.get("rms_norm_eps", 1e-6)) or 1e-6),
        "rope_theta": float(mtp.get("rope_theta", 1_000_000.0) or 1_000_000.0),
        "local_layer_ids": [int(x) for x in (mtp.get("local_layer_ids") or [])],
        "sliding_window_size": int(mtp.get("sliding_window_size", 1024) or 1024),
        "block_size": int(mtp.get("block_size", 8) or 8),
        "markov_rank": int(mtp.get("markov_rank", 256) or 256),
        "markov_head_type": str(mtp.get("markov_head_type", "rnn_hidden") or "rnn_hidden"),
        "target_layer_ids": [int(x) for x in (mtp.get("target_layer_ids") or [])],
        "vocab_size": int(tc.get("vocab_size", 201024) or 201024),
        "unpadded_vocab_size": int(tc.get("unpadded_vocab_size", 200058) or 200058),
    }


def export_dflare_from_safetensors(
    mtp_source: pathlib.Path | dict[str, _ShardTensor],
    output_amf: pathlib.Path,
    *,
    config_json: dict[str, object] | None = None,
    config_output: pathlib.Path | None = None,
    model_id: str = "inkling2-dflare",
    dtype_mode: str = "auto",
) -> tuple[pathlib.Path, pathlib.Path, list[PackedTensorMeta]]:
    """Export native `3L`/`5L` `DFlare` tensors (`mtp.safetensors` or `mtp.*`) to `dflare.amf` + `dflare_config.json`."""
    out_amf = pathlib.Path(output_amf)
    out_cfg = pathlib.Path(config_output) if config_output is not None else out_amf.with_name("dflare_config.json")
    cfg_dict = dict(config_json or {})
    if isinstance(mtp_source, dict):
        raw_tensors = mtp_source
    else:
        src_p = pathlib.Path(mtp_source)
        resolved = resolve_checkpoint(src_p)
        if not cfg_dict and resolved.config_path is not None and resolved.config_path.is_file():
            loaded = json.loads(resolved.config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg_dict = loaded
        raw_tensors = _collect_tensors(resolved.shard_paths)

    remapped: dict[str, _ShardTensor] = {}
    for raw_name, st in raw_tensors.items():
        clean = _strip_inkling2_mtp_prefix(raw_name)
        remapped[clean] = _ShardTensor(
            name=clean,
            dtype=st.dtype,
            shape=st.shape,
            offset=st.offset,
            length=st.length,
            path=st.path,
        )

    dflare_cfg = build_dflare_config_dict(cfg_dict)
    if "mtp_config" not in cfg_dict:
        # Infer layer count if config.json omitted mtp_config.
        layer_ids = {int(m.group(1)) for n in remapped if (m := re.match(r"^layers\.(\d+)\.", n)) is not None}
        if layer_ids:
            dflare_cfg["num_hidden_layers"] = max(layer_ids) + 1
    dflare_cfg_bytes = (json.dumps(dflare_cfg, indent=2) + "\n").encode("utf-8")
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_bytes(dflare_cfg_bytes)

    plan = plan_amf_entries(
        remapped,
        group_size=64,
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
            meta_blobs=[("__meta.config_json", dflare_cfg_bytes)],
        )
    finally:
        readers.close()
    return out_amf, out_cfg, metas


def build_inkling2_config_json(
    src_config: dict[str, object] | None,
    *,
    num_layers: int,
    sme2_controllers: bool,
) -> dict[str, object]:
    """Build canonical runtime `config.json` enriched with `operator_axes` & `loop_schedule`."""
    raw = dict(src_config) if isinstance(src_config, dict) else {}
    tc = _effective_text_config(raw)
    loop_cfg = tc.get("loop_config") if isinstance(tc.get("loop_config"), dict) else {}
    core_start = int(loop_cfg.get("core_start", 0) or 0)
    core_end = int(loop_cfg.get("core_end", 0) or 0)
    num_passes = int(loop_cfg.get("num_passes", 1) or 1)
    gate_rank = int(loop_cfg.get("gate_rank", 0) or 0)

    is_loop = bool(core_end > core_start and num_passes > 1)
    raw.setdefault("model_type", "inkling2_loop" if is_loop else "inkling2")
    raw["operator_axes"] = {
        "attention": "rel_gqa_swa_sconv",
        "ffn": "moe_softcap",
        "kv_layout": "hybrid_swa_1024",
        "sparse_memory": "none",
        "residual": "attn_res_6",
        "loop_controllers": "sme2_panel64" if sme2_controllers else "row_major",
    }
    if is_loop:
        raw["loop_schedule"] = {
            "core_start": core_start,
            "core_end": core_end,
            "num_passes": num_passes,
            "gate_rank": gate_rank,
            "effective_num_steps": num_layers + (core_end - core_start) * (num_passes - 1),
            "controllers": ["attn", "k", "v", "mlp"],
        }
    return raw


@dataclass(frozen=True)
class Inkling2ConvertReport:
    backbone_amf: pathlib.Path
    backbone_metas: list[PackedTensorMeta]
    vision_amf: pathlib.Path | None
    vision_metas: list[PackedTensorMeta]
    dflare_amf: pathlib.Path | None
    dflare_config_path: pathlib.Path | None
    dflare_metas: list[PackedTensorMeta]
    config_path: pathlib.Path | None
    sme2_controller_count: int
    cold_expert_count: int
    tier_profile: Inkling2TierProfile | None


def _resolve_companion_metadata_files(
    *,
    base_dir: pathlib.Path,
    config_dir: pathlib.Path | None,
    config_path: pathlib.Path | None,
    tokenizer_path: pathlib.Path | None,
    tokenizer_config_path: pathlib.Path | None,
    chat_template_path: pathlib.Path | None,
) -> tuple[pathlib.Path | None, pathlib.Path | None, pathlib.Path | None, pathlib.Path | None]:
    cfg_root = pathlib.Path(config_dir) if config_dir is not None else base_dir
    eff_cfg = pathlib.Path(config_path) if config_path is not None else (cfg_root / "config.json")
    eff_tok = pathlib.Path(tokenizer_path) if tokenizer_path is not None else (cfg_root / "tokenizer.json")
    eff_tok_cfg = pathlib.Path(tokenizer_config_path) if tokenizer_config_path is not None else (cfg_root / "tokenizer_config.json")
    eff_tpl = pathlib.Path(chat_template_path) if chat_template_path is not None else (cfg_root / "chat_template.jinja")
    return (
        eff_cfg if eff_cfg.is_file() else None,
        eff_tok if eff_tok.is_file() else None,
        eff_tok_cfg if eff_tok_cfg.is_file() else None,
        eff_tpl if eff_tpl.is_file() else None,
    )


def convert_inkling2_checkpoint(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    model_id: str = "inkling2",
    dtype_mode: str = "auto",
    group_size: int | None = None,
    quant_profile: str = "uniform",
    target_bits: float | None = None,
    target_wired_gib: float | None = None,
    target_tier: str = "auto",
    use_rco: bool = True,
    manifest_path: pathlib.Path | None = None,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int | None = None,
    heat_profile: pathlib.Path | None = None,
    all_experts_cold: bool = False,
    sme2_pack: str | None = None,
    strip_vision: bool | None = None,
    include_vision: bool | None = None,
    vision_sidecar: pathlib.Path | None = None,
    vision_block_dtype: str = "auto",
    dflare_output: pathlib.Path | None = None,
    dflare_config_output: pathlib.Path | None = None,
    config_output: pathlib.Path | None = None,
    config_dir: pathlib.Path | None = None,
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
) -> Inkling2ConvertReport:
    """Convert an Inkling 2 safetensors, `.amf`, or GGUF checkpoint into 16 KiB-aligned `.amf` bundles (`W23`, `W29`)."""
    if quant_profile not in QUANT_PROFILES:
        raise ValueError(f"quant_profile must be one of {QUANT_PROFILES}, got {quant_profile!r}")
    if sme2_pack is not None and sme2_pack not in SME2_PACK_MODES:
        raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
    if vision_block_dtype not in VISION_BLOCK_DTYPES:
        raise ValueError(f"vision_block_dtype must be one of {VISION_BLOCK_DTYPES}, got {vision_block_dtype!r}")

    in_path = pathlib.Path(input_path)
    out_path = pathlib.Path(output_path)

    # GGUF pathway (`unsloth/Inkling-2-Mini-GGUF` `inkling2_loop`/`inkling2` deduplication or `dflash` draft).
    if (in_path.is_file() and in_path.suffix.lower() == ".gguf") or (
        in_path.is_dir() and any(in_path.glob("*.gguf")) and not (in_path / "model.safetensors.index.json").is_file()
    ):
        from aminfer_tools.gguf import resolve_gguf_shards

        shards = resolve_gguf_shards(in_path)
        arch = str(shards[0].kv.get("general.architecture") or "").lower()
        base_dir = in_path if in_path.is_dir() else in_path.parent
        eff_cfg_p, eff_tok_p, eff_tok_cfg_p, eff_tpl_p = _resolve_companion_metadata_files(
            base_dir=base_dir,
            config_dir=config_dir,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=tokenizer_config_path,
            chat_template_path=chat_template_path,
        )
        if arch == "dflash":
            d_amf, d_cfg, d_metas = export_dflare_bundle_from_gguf(
                in_path,
                out_path.parent if out_path.suffix == ".amf" else out_path,
                model_id=model_id,
                dtype_mode=dtype_mode,
                config_path=eff_cfg_p,
            )
            return Inkling2ConvertReport(
                backbone_amf=d_amf,
                backbone_metas=d_metas,
                vision_amf=None,
                vision_metas=[],
                dflare_amf=d_amf,
                dflare_config_path=d_cfg,
                dflare_metas=d_metas,
                config_path=d_cfg,
                sme2_controller_count=0,
                cold_expert_count=0,
                tier_profile=None,
            )

        tier_prof: Inkling2TierProfile | None = None
        if target_tier != "auto":
            kv0 = shards[0].kv
            n_l = int(kv0.get(f"{arch}.block_count", 24) or 24)
            n_e = int(kv0.get(f"{arch}.expert_count", 128) or 128)
            tier_prof = resolve_inkling2_tier_profile(target_tier, num_hidden_layers=n_l, n_routed_experts=n_e)

        eff_cold_per_layer = cold_experts_per_layer
        if eff_cold_per_layer is None and tier_prof is not None and target_tier == "24gb-m4-full-wired":
            eff_cold_per_layer = 0

        eff_strip_vis = (
            False
            if include_vision is True
            else (
                True
                if (
                    vision_sidecar is not None
                    or strip_vision is True
                    or (strip_vision is None and tier_prof is not None and tier_prof.vision_sidecar_recommended)
                )
                else False
            )
        )

        meta_override: bytes | None = None
        if eff_cfg_p is not None:
            loaded_cfg = json.loads(eff_cfg_p.read_text(encoding="utf-8"))
            if isinstance(loaded_cfg, dict):
                tc0 = _effective_text_config(loaded_cfg)
                n_layers_cfg = int(tc0.get("num_hidden_layers", shards[0].kv.get(f"{arch}.block_count", 24)) or 24)
                has_ctrls_gguf = any("_ctrl_" in t.name or "loop_attn_" in t.name for sh in shards for t in sh.tensors)
                sme2_ctrls = (sme2_pack != "none") and (has_ctrls_gguf or bool(tc0.get("loop_config")))
                enriched = build_inkling2_config_json(loaded_cfg, num_layers=n_layers_cfg, sme2_controllers=sme2_ctrls)
                meta_override = (json.dumps(enriched, indent=2) + "\n").encode("utf-8")

        stats: dict[str, int] = {}
        metas = pack_amf_from_gguf(
            in_path,
            out_path,
            model_id=model_id,
            dtype_mode=dtype_mode,
            cold_experts=cold_experts,
            cold_experts_per_layer=eff_cold_per_layer,
            heat_profile=heat_profile,
            all_experts_cold=all_experts_cold,
            embed_metadata=embed_metadata,
            config_path=eff_cfg_p,
            tokenizer_path=eff_tok_p,
            tokenizer_config_path=eff_tok_cfg_p,
            chat_template_path=eff_tpl_p,
            strip_vision=eff_strip_vis,
            vision_sidecar=vision_sidecar,
            vision_block_dtype=vision_block_dtype,
            strict_vocab_check=strict_vocab_check,
            sme2_pack=sme2_pack,
            draft_config_output=config_output,
            meta_config_override=meta_override,
            stats_out=stats,
        )
        vis_metas_gguf: list[PackedTensorMeta] = []
        vis_out_gguf: pathlib.Path | None = None
        if vision_sidecar is not None and pathlib.Path(vision_sidecar).is_file():
            vis_out_gguf = pathlib.Path(vision_sidecar)
            with AmfReader(vis_out_gguf) as vr:
                vis_metas_gguf = list(vr.entries)
        ctrl_cnt = sum(1 for m in metas if m.layout == TensorLayout.SME2_PANEL64 and is_inkling2_controller_tensor(m.name))
        cold_cnt = sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT)
        return Inkling2ConvertReport(
            backbone_amf=out_path,
            backbone_metas=metas,
            vision_amf=vis_out_gguf,
            vision_metas=vis_metas_gguf,
            dflare_amf=None,
            dflare_config_path=None,
            dflare_metas=[],
            config_path=pathlib.Path(config_output) if config_output is not None else None,
            sme2_controller_count=ctrl_cnt,
            cold_expert_count=cold_cnt,
            tier_profile=tier_prof,
        )

    is_amf_in = in_path.is_file() and in_path.suffix.lower() == ".amf"
    base_dir = in_path if in_path.is_dir() else in_path.parent
    eff_cfg_p, eff_tok_p, eff_tok_cfg_p, eff_tpl_p = _resolve_companion_metadata_files(
        base_dir=base_dir,
        config_dir=config_dir,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        tokenizer_config_path=tokenizer_config_path,
        chat_template_path=chat_template_path,
    )

    src_cfg: dict[str, object] = {}
    backbone_tensors: dict[str, _ShardTensor] = {}
    mtp_tensors: dict[str, _ShardTensor] = {}
    all_names: list[str] = []

    if is_amf_in:
        with AmfReader(in_path, verify_header_crc=False, verify_dir_crc=False) as amf_rdr:
            all_names = [m.name for m in amf_rdr.entries if m.kind != TensorKind.METADATA]
            if META_CONFIG_JSON in amf_rdr.by_name and eff_cfg_p is None:
                loaded = json.loads(amf_rdr.read_metadata_bytes(META_CONFIG_JSON, verify_crc=True).decode("utf-8"))
                if isinstance(loaded, dict):
                    src_cfg = loaded
    else:
        resolved = resolve_checkpoint(in_path)
        cfg_candidate = eff_cfg_p or resolved.config_path
        if cfg_candidate is not None and cfg_candidate.is_file():
            loaded = json.loads(cfg_candidate.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                src_cfg = loaded
        all_tensors = _collect_tensors(resolved.shard_paths)
        for k, v in all_tensors.items():
            if _is_inkling2_mtp_tensor(k):
                mtp_tensors[k] = v
            else:
                backbone_tensors[k] = v
        mtp_file = base_dir / "mtp.safetensors"
        if not mtp_tensors and mtp_file.is_file():
            mtp_tensors = _collect_tensors((mtp_file,))
        all_names = list(backbone_tensors.keys())

    if not src_cfg and eff_cfg_p is not None and eff_cfg_p.is_file():
        loaded = json.loads(eff_cfg_p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            src_cfg = loaded

    tc = _effective_text_config(src_cfg)
    if tc.get("num_hidden_layers"):
        num_layers = int(tc["num_hidden_layers"])
    else:
        layer_ids = {int(m.group(1)) for n in all_names if (m := re.search(r"layers\.(\d+)\.", n)) is not None}
        num_layers = (max(layer_ids) + 1) if layer_ids else 24
    if tc.get("n_routed_experts") or tc.get("num_experts"):
        num_experts = int(tc.get("n_routed_experts", tc.get("num_experts")) or 128)
    else:
        exp_ids = {int(m.group(1)) for n in all_names if (m := re.search(r"\.experts\.(\d+)\.", n)) is not None}
        num_experts = (max(exp_ids) + 1) if exp_ids else 128

    tier_prof = resolve_inkling2_tier_profile(target_tier, num_hidden_layers=num_layers, n_routed_experts=num_experts)

    eff_quant_profile = quant_profile
    if eff_quant_profile == "uniform" and (
        is_amf_in or target_tier == "24gb-m4-full-wired" or target_wired_gib is not None or imatrix_path is not None
    ):
        eff_quant_profile = "gsq-rco"

    eff_cold: set[tuple[int, int]] = set(cold_experts)
    eff_cold_per_layer = cold_experts_per_layer
    if eff_cold_per_layer is None and target_tier == "24gb-m4-full-wired" and not eff_cold:
        eff_cold_per_layer = 0
    if heat_profile is not None:
        eff_cold.update(
            load_cold_experts_from_heat_profile(
                pathlib.Path(heat_profile),
                cold_experts_per_layer=eff_cold_per_layer,
            )
        )
    elif eff_cold_per_layer is not None:
        eff_cold.update(_tail_cold_experts(num_layers, num_experts, eff_cold_per_layer))
    elif eff_quant_profile == "gsq-rco" and target_tier != "auto" and tier_prof.cold_tail_fraction > 0.0:
        tail_cnt = int(round(num_experts * tier_prof.cold_tail_fraction))
        eff_cold.update(_tail_cold_experts(num_layers, num_experts, tail_cnt))

    if sme2_pack is None:
        is_loop_mt = str(src_cfg.get("model_type", tc.get("model_type", ""))).lower() in ("inkling2_loop", "inkling2_loop_text")
        has_ctrls = any(is_inkling2_controller_tensor(k) for k in all_names)
        eff_sme2 = "controllers" if (is_loop_mt or has_ctrls) else "none"
    else:
        eff_sme2 = sme2_pack

    vision_mode = resolve_vision_mode(
        all_names,
        include_vision=include_vision,
        strip_vision=strip_vision,
        vision_sidecar=vision_sidecar,
    )
    eff_vis_sidecar = vision_sidecar
    if vision_mode == "sidecar" and eff_vis_sidecar is None:
        eff_vis_sidecar = out_path.with_name("vision.amf")

    eff_target_bits = target_bits
    eff_target_wired_gib = target_wired_gib
    eff_max_q2 = max_q2_share
    if eff_quant_profile == "gsq-rco" and target_tier != "auto":
        if eff_target_bits is None and eff_target_wired_gib is None:
            eff_target_bits = tier_prof.target_bits
        if eff_target_wired_gib is None and target_tier == "24gb-m4-full-wired":
            eff_target_wired_gib = tier_prof.wired_budget_bytes / (1024**3)
        if eff_max_q2 is None:
            eff_max_q2 = tier_prof.max_q2_share

    eff_fine_rungs = (eff_quant_profile == "gsq-rco") if fine_rungs is None else fine_rungs
    excl_mtp = tuple(mtp_tensors.keys()) if (mtp_tensors and all(_is_inkling2_mtp_tensor(k) for k in mtp_tensors)) else ()

    has_ctrls_eff = eff_sme2 != "none" and (
        any(is_inkling2_controller_tensor(k) for k in all_names)
        or str(src_cfg.get("model_type", tc.get("model_type", ""))).lower() in ("inkling2_loop", "inkling2_loop_text")
    )
    enriched_cfg = build_inkling2_config_json(src_cfg, num_layers=num_layers, sme2_controllers=has_ctrls_eff)
    enriched_bytes = (json.dumps(enriched_cfg, indent=2) + "\n").encode("utf-8")

    # Decide between direct single-pass `gsq_convert` vs `pack_amf_from_safetensors`.
    use_direct_gsq = eff_quant_profile == "gsq-rco" and (
        is_amf_in
        or eff_target_wired_gib is not None
        or imatrix_path is not None
        or target_tier == "24gb-m4-full-wired"
        or include_vision is not True
    )
    embedded_single_pass = False
    if use_direct_gsq:
        from aminfer_tools.gsq_rco import gsq_convert

        has_vision = any(is_vision_tensor(n) for n in all_names)
        if vision_mode == "sidecar" and eff_vis_sidecar is not None and has_vision:
            pack_vision_tower_amf(
                in_path,
                eff_vis_sidecar,
                model_id=f"{model_id[:55]}-vision",
                dtype_mode=dtype_mode,
                quant_profile="gsq-rco",
                vision_block_dtype=vision_block_dtype,
                group_size=group_size if group_size is not None else 64,
                use_rco=use_rco,
                workers=workers,
                embed_metadata=embed_metadata,
                config_path=eff_cfg_p,
            )

        gsq_meta_blobs: list[tuple[str, bytes]] = []
        if embed_metadata:
            gsq_meta_blobs.append((META_CONFIG_JSON, enriched_bytes))
            if eff_tok_p is not None:
                gsq_meta_blobs.append((META_TOKENIZER_JSON, eff_tok_p.read_bytes()))
            elif is_amf_in:
                with AmfReader(in_path, verify_header_crc=False, verify_dir_crc=False) as amf_rdr:
                    if META_TOKENIZER_JSON in amf_rdr.by_name:
                        gsq_meta_blobs.append((META_TOKENIZER_JSON, amf_rdr.read_metadata_bytes(META_TOKENIZER_JSON)))
                    if META_CHAT_TEMPLATE_JSON in amf_rdr.by_name:
                        gsq_meta_blobs.append((META_CHAT_TEMPLATE_JSON, amf_rdr.read_metadata_bytes(META_CHAT_TEMPLATE_JSON)))
            tpl_txt = eff_tpl_p.read_text(encoding="utf-8") if eff_tpl_p is not None else None
            tok_cfg_d = json.loads(eff_tok_cfg_p.read_text(encoding="utf-8")) if eff_tok_cfg_p is not None else {}
            if tpl_txt or (isinstance(tok_cfg_d, dict) and tok_cfg_d.get("chat_template")):
                spec = compile_chat_template_spec(
                    jinja_text=tpl_txt,
                    tokenizer_config=tok_cfg_d if isinstance(tok_cfg_d, dict) else {},
                    model_config=enriched_cfg,
                )
                gsq_meta_blobs.append(
                    (META_CHAT_TEMPLATE_JSON, json.dumps(spec, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                )
            embedded_single_pass = True

        eff_tie = bool(tc.get("tie_word_embeddings", False)) if not is_amf_in else None
        rep_gsq = gsq_convert(
            in_path,
            out_path,
            target_bits=eff_target_bits,
            target_wired_gib=eff_target_wired_gib,
            model_id=model_id[:63],
            group_size=group_size if group_size is not None else 64,
            dtype_mode=dtype_mode,
            heat_profile=heat_profile,
            cold_experts=eff_cold,
            cold_experts_per_layer=(eff_cold_per_layer if heat_profile is not None else None),
            all_experts_cold=all_experts_cold,
            tie_word_embeddings=eff_tie,
            exclude_tensors=excl_mtp,
            sme2_pack=eff_sme2,
            use_rco=use_rco,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=(include_vision is not True),
            imatrix_path=imatrix_path,
            enforce_backbone_floor=enforce_backbone_floor,
            max_q2_share=eff_max_q2,
            allow_lowbit_q2_retranscode=allow_lowbit_q2_retranscode,
            fine_rungs=eff_fine_rungs,
            meta_blobs=gsq_meta_blobs,
        )
        metas = rep_gsq.metas
    else:
        metas = pack_amf_from_safetensors(
            in_path,
            out_path,
            model_id=model_id[:63],
            group_size=group_size,
            cold_experts=eff_cold,
            all_experts_cold=all_experts_cold,
            dtype_mode=dtype_mode,
            exclude_tensors=excl_mtp,
            sme2_pack=eff_sme2,
            quant_profile=eff_quant_profile,
            target_bits=eff_target_bits,
            use_rco=use_rco,
            manifest_path=manifest_path,
            workers=workers,
            strip_vision=strip_vision,
            include_vision=include_vision,
            vision_sidecar=eff_vis_sidecar,
            vision_block_dtype=vision_block_dtype,
            max_q2_share=eff_max_q2,
            fine_rungs=eff_fine_rungs,
            embed_metadata=embed_metadata,
        )

    vis_metas: list[PackedTensorMeta] = []
    vis_out: pathlib.Path | None = None
    if vision_mode == "sidecar" and eff_vis_sidecar is not None and eff_vis_sidecar.is_file():
        vis_out = eff_vis_sidecar
        with AmfReader(vis_out) as vr:
            vis_metas = list(vr.entries)

    ctrl_cnt = sum(1 for m in metas if m.layout == TensorLayout.SME2_PANEL64 and is_inkling2_controller_tensor(m.name))
    cold_cnt = sum(1 for m in metas if m.kind == TensorKind.COLD_EXPERT)
    if ctrl_cnt == 0 and has_ctrls_eff:
        enriched_cfg = build_inkling2_config_json(src_cfg, num_layers=num_layers, sme2_controllers=False)
        enriched_bytes = (json.dumps(enriched_cfg, indent=2) + "\n").encode("utf-8")
    if embed_metadata and not embedded_single_pass:
        metas = embed_metadata_into_amf(out_path, config_bytes=enriched_bytes, output_path=out_path)

    cfg_out_path: pathlib.Path | None = None
    if config_output is not None:
        cfg_out_path = pathlib.Path(config_output)
        cfg_out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_out_path.write_bytes(enriched_bytes)

    dflare_amf_path: pathlib.Path | None = None
    dflare_cfg_path: pathlib.Path | None = None
    dflare_metas: list[PackedTensorMeta] = []
    if mtp_tensors:
        eff_df_amf = pathlib.Path(dflare_output) if dflare_output is not None else out_path.with_name("dflare.amf")
        eff_df_cfg = (
            pathlib.Path(dflare_config_output)
            if dflare_config_output is not None
            else eff_df_amf.with_name("dflare_config.json")
        )
        dflare_amf_path, dflare_cfg_path, dflare_metas = export_dflare_from_safetensors(
            mtp_tensors,
            eff_df_amf,
            config_json=src_cfg,
            config_output=eff_df_cfg,
            model_id=f"{model_id[:55]}-dflare",
            dtype_mode=dtype_mode,
        )

    return Inkling2ConvertReport(
        backbone_amf=out_path,
        backbone_metas=metas,
        vision_amf=vis_out,
        vision_metas=vis_metas,
        dflare_amf=dflare_amf_path,
        dflare_config_path=dflare_cfg_path,
        dflare_metas=dflare_metas,
        config_path=cfg_out_path,
        sme2_controller_count=ctrl_cnt,
        cold_expert_count=cold_cnt,
        tier_profile=tier_prof,
    )


# ---------------------------------------------------------------------------
# Deterministic Tiny Inkling 2 Synthetic Bundle Generator (`synth-inkling2`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inkling2SynthConfig:
    model_type: str = "inkling2_loop"
    num_hidden_layers: int = 6
    core_start: int = 2
    core_end: int = 4
    num_passes: int = 3
    gate_rank: int = 64
    hidden_size: int = 128
    num_attention_heads: int = 2
    num_key_value_heads: int = 1
    head_dim: int = 64
    d_rel: int = 16
    rel_extent: int = 64
    sconv_kernel_size: int = 4
    sliding_window_size: int = 1024
    local_layer_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_layers_per_attn_res_block: int = 2
    n_routed_experts: int = 16
    num_experts_per_tok: int = 4
    intermediate_size: int = 192  # non-256-divisible, multiple of 64 (mirrors Mini's 1152)
    route_scale: float = 8.0
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float = 30.0
    vocab_size: int = 256
    dflare_layers: int = 3
    dflare_block_size: int = 8
    dflare_markov_rank: int = 256
    dflare_intermediate_size: int = 256
    dflare_target_layer_ids: tuple[int, ...] = (1, 3, 5, 10)
    vision_layers: int = 23
    vision_hidden_size: int = 64
    vision_num_heads: int = 4
    dtype: str = "bf16"
    sme2_pack: str = "controllers"
    cold_experts_per_layer: int = 2
    vision_mode: str = "sidecar"  # "sidecar", "include", "strip"
    include_dflare: bool = True
    model_id: str = "tiny-inkling2"

    def validate(self) -> None:
        if self.model_type not in ("inkling2_loop", "inkling2"):
            raise ValueError(f"model_type must be 'inkling2_loop' or 'inkling2', got {self.model_type!r}")
        if self.hidden_size <= 0 or self.hidden_size % 64 != 0:
            raise ValueError(f"hidden_size must be a positive multiple of 64, got {self.hidden_size}")
        if self.intermediate_size <= 0 or self.intermediate_size % 64 != 0:
            raise ValueError(f"intermediate_size must be a positive multiple of 64, got {self.intermediate_size}")
        if self.model_type == "inkling2_loop" and self.gate_rank != 64:
            raise ValueError(f"inkling2_loop controller gate_rank must be 64 for sme2_panel64, got {self.gate_rank}")
        if self.dtype not in ("bf16", "f16"):
            raise ValueError(f"dtype must be 'bf16' or 'f16', got {self.dtype!r}")
        if self.vision_mode not in ("sidecar", "include", "strip"):
            raise ValueError(f"vision_mode must be 'sidecar', 'include', or 'strip', got {self.vision_mode!r}")


INKLING2_SYNTH_PRESETS: dict[str, Inkling2SynthConfig] = {
    "tiny-inkling2-mini": Inkling2SynthConfig(
        model_type="inkling2_loop",
        num_hidden_layers=6,
        core_start=2,
        core_end=4,
        num_passes=3,
        gate_rank=64,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=192,
        n_routed_experts=16,
        num_experts_per_tok=4,
        dflare_layers=3,
        dflare_target_layer_ids=(1, 3, 5, 10),
        vision_layers=23,
        sme2_pack="controllers",
        cold_experts_per_layer=2,
        model_id="tiny-inkling2-mini",
    ),
    "tiny-inkling2-small": Inkling2SynthConfig(
        model_type="inkling2",
        num_hidden_layers=6,
        core_start=0,
        core_end=0,
        num_passes=1,
        gate_rank=0,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=128,
        n_routed_experts=16,
        num_experts_per_tok=4,
        dflare_layers=5,
        dflare_target_layer_ids=(1, 3, 5),
        vision_layers=23,
        sme2_pack="none",
        cold_experts_per_layer=2,
        model_id="tiny-inkling2-small",
    ),
}


@dataclass(frozen=True)
class Inkling2SynthBundle:
    output_dir: pathlib.Path
    model_amf: pathlib.Path
    dflare_amf: pathlib.Path | None
    dflare_config_path: pathlib.Path | None
    vision_amf: pathlib.Path | None
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path
    backbone_metas: list[PackedTensorMeta]
    dflare_metas: list[PackedTensorMeta]
    vision_metas: list[PackedTensorMeta]
    total_bytes: int


def _build_inkling2_tokenizer_dict(vocab_size: int) -> dict[str, object]:
    vocab: dict[str, int] = {
        "<|endoftext|>": 0,
        "<|message_system|>": 1,
        "<|message_user|>": 2,
        "<|message_model|>": 3,
        "<|content_thinking|>": 4,
        "<|content_text|>": 5,
        "<|content_model_end_sampling|>": 6,
    }
    for i in range(len(vocab), vocab_size):
        vocab[f"tok_{i}"] = i
    added = [
        {"id": tid, "content": tok, "special": True}
        for tok, tid in vocab.items()
        if tok.startswith("<|")
    ]
    return {
        "version": "1.0",
        "added_tokens": added,
        "model": {"type": "BPE", "ignore_merges": True, "vocab": vocab, "merges": []},
    }


def _build_inkling2_jinja() -> str:
    return (
        "{{ bos_token }}"
        "{%- for message in messages -%}"
        "<|message_{{ message['role'] }}|><|content_text|>{{ message['content'] }}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|message_model|><|content_text|>{%- endif -%}"
    )


def generate_synth_inkling2_bundle(
    output_dir: pathlib.Path,
    *,
    preset: str = "tiny-inkling2-mini",
    config: Inkling2SynthConfig | None = None,
    seed: int = 42,
) -> Inkling2SynthBundle:
    """Generate a deterministic tiny Inkling 2 (`Mini` recurrent or `Small` non-recurrent) bundle."""
    if config is not None:
        cfg = config
    else:
        if preset not in INKLING2_SYNTH_PRESETS:
            raise ValueError(f"unknown Inkling 2 preset {preset!r}; expected {sorted(INKLING2_SYNTH_PRESETS)}")
        cfg = INKLING2_SYNTH_PRESETS[preset]
    cfg.validate()

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf_dt = "BF16" if cfg.dtype == "bf16" else "F16"
    h = cfg.hidden_size
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    local_set = set(cfg.local_layer_ids)
    is_loop = cfg.model_type == "inkling2_loop"

    hf_cfg: dict[str, object] = {
        "architectures": ["Inkling2LoopForConditionalGeneration" if is_loop else "Inkling2ForConditionalGeneration"],
        "model_type": cfg.model_type,
        "eos_token_id": 6,
        "text_config": {
            "model_type": f"{cfg.model_type}_text",
            "hidden_size": h,
            "num_hidden_layers": cfg.num_hidden_layers,
            "vocab_size": cfg.vocab_size,
            "unpadded_vocab_size": cfg.vocab_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "d_rel": cfg.d_rel,
            "rel_extent": cfg.rel_extent,
            "sconv_kernel_size": cfg.sconv_kernel_size,
            "sliding_window_size": cfg.sliding_window_size,
            "local_layer_ids": list(cfg.local_layer_ids),
            "n_layers_per_attn_res_block": cfg.n_layers_per_attn_res_block,
            "attn_res_normalize_values": True,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "log_scaling_n_floor": 8192,
            "log_scaling_alpha": 0.1,
            "n_routed_experts": cfg.n_routed_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "n_shared_experts": 0,
            "intermediate_size": cfg.intermediate_size,
            "mlp_gate_softcap": cfg.mlp_gate_softcap,
            "mlp_up_softcap": cfg.mlp_up_softcap,
            "final_logit_softcapping": cfg.final_logit_softcapping,
            "route_scale": cfg.route_scale,
            "use_gate_bias": True,
            "rms_norm_eps": 1e-6,
        },
        "mtp_config": {
            "hidden_size": h,
            "num_hidden_layers": cfg.dflare_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "intermediate_size": cfg.dflare_intermediate_size,
            "block_size": cfg.dflare_block_size,
            "markov_rank": cfg.dflare_markov_rank,
            "markov_head_type": "rnn_hidden",
            "target_layer_ids": list(cfg.dflare_target_layer_ids),
            "local_layer_ids": [i for i in range(cfg.dflare_layers) if i % 2 == 0],
            "sliding_window_size": cfg.sliding_window_size,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
        },
    }
    if is_loop:
        assert isinstance(hf_cfg["text_config"], dict)
        hf_cfg["text_config"]["loop_config"] = {
            "core_start": cfg.core_start,
            "core_end": cfg.core_end,
            "num_passes": cfg.num_passes,
            "gate_rank": cfg.gate_rank,
            "is_varloop": False,
        }
    if cfg.vision_layers > 0:
        hf_cfg["vision_config"] = {
            "model_type": "inkling2_vision",
            "text_hidden_size": h,
            "hidden_size": cfg.vision_hidden_size,
            "num_hidden_layers": cfg.vision_layers,
            "num_attention_heads": cfg.vision_num_heads,
            "mlp_ratio": 4.0,
            "native_patch_size": 4,
            "temporal_patch_size": 2,
            "adapter_merge_size": 2,
            "projection_hidden_dim": h,
        }

    cfg_path = out_dir / "config.json"
    tok_path = out_dir / "tokenizer.json"
    tpl_path = out_dir / "chat_template.jinja"
    cfg_path.write_text(json.dumps(hf_cfg, indent=2) + "\n", encoding="utf-8")
    tok_path.write_text(json.dumps(_build_inkling2_tokenizer_dict(cfg.vocab_size), indent=2) + "\n", encoding="utf-8")
    tpl_path.write_text(_build_inkling2_jinja(), encoding="utf-8")

    # Build deterministic backbone + optional ViT safetensors.
    bb_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def _add(name: str, shape: tuple[int, ...], *, f32: bool = False, one: bool = False, scale: float = 0.04) -> None:
        n = 1
        for d_s in shape:
            n *= d_s
        vals = _deterministic_f32(seed + (zlib_hash(name) & 0xFFFF), n, scale=scale, centered_one=one)
        if f32:
            bb_tensors[name] = ("F32", shape, _encode_floats(vals, "f32"))
        else:
            bb_tensors[name] = (sf_dt, shape, _encode_floats(vals, cfg.dtype))

    import zlib

    def zlib_hash(s: str) -> int:
        return zlib.crc32(s.encode("utf-8"))

    _add("model.llm.embed.weight", (cfg.vocab_size, h), scale=0.02)
    _add("model.llm.embed_norm.weight", (h,), one=True)
    _add("model.llm.norm.weight", (h,), one=True)
    _add("model.llm.unembed.weight", (cfg.vocab_size, h), scale=0.02)
    _add("model.llm.attn_res.output_query", (h,), f32=True, scale=0.05)
    if is_loop:
        _add("model.llm.middle_cycle_cross_pass_queries", (h,), f32=True, scale=0.05)
        _add("model.llm.middle_cycle_cross_pass_kv_queries", (h,), f32=True, scale=0.05)

    for l in range(cfg.num_hidden_layers):
        p = f"model.llm.layers.{l}"
        _add("model.llm.attn_res.attn_queries." + str(l), (h,), f32=True, scale=0.05)
        _add("model.llm.attn_res.mlp_queries." + str(l), (h,), f32=True, scale=0.05)
        _add(f"{p}.attn_norm.weight", (h,), one=True)
        _add(f"{p}.q_input_norm.weight", (h,), one=True)
        _add(f"{p}.k_input_norm.weight", (h,), one=True)
        _add(f"{p}.v_input_norm.weight", (h,), one=True)
        _add(f"{p}.mlp_norm.weight", (h,), one=True)
        _add(f"{p}.attn_sconv.weight", (h, cfg.sconv_kernel_size), scale=0.05)
        _add(f"{p}.mlp_sconv.weight", (h, cfg.sconv_kernel_size), scale=0.05)

        _add(f"{p}.attn.wq_du.weight", (q_out, h))
        _add(f"{p}.attn.wk_dv.weight", (kv_out, h))
        _add(f"{p}.attn.wv_dv.weight", (kv_out, h))
        _add(f"{p}.attn.wo_ud.weight", (h, q_out))
        _add(f"{p}.attn.wr_du.weight", (cfg.num_attention_heads * cfg.d_rel, h))
        _add(f"{p}.attn.rel_logits_proj.proj", (cfg.rel_extent, cfg.d_rel))
        _add(f"{p}.attn.q_l2_clip.scale", (1,), one=True)
        _add(f"{p}.attn.rel_l2_clip.scale", (1,), one=True)
        _add(f"{p}.attn.attn_sink_proj.weight", (cfg.num_attention_heads, h))
        _add(f"{p}.attn.gate_attn.weight", (q_out, h))

        if l not in local_set:
            # Global layer carries dynamic k/v sconv projections.
            _add(f"{p}.attn.k_sconv.static_w", (kv_out, cfg.sconv_kernel_size), scale=0.05)
            _add(f"{p}.attn.k_sconv.weight", (kv_out, cfg.sconv_kernel_size), scale=0.05)
            _add(f"{p}.attn.k_sconv.weight_proj", (cfg.sconv_kernel_size, h), scale=0.05)
            _add(f"{p}.attn.v_sconv.static_w", (kv_out, cfg.sconv_kernel_size), scale=0.05)
            _add(f"{p}.attn.v_sconv.weight", (kv_out, cfg.sconv_kernel_size), scale=0.05)
            _add(f"{p}.attn.v_sconv.weight_proj", (cfg.sconv_kernel_size, h), scale=0.05)

        if is_loop and cfg.core_start <= l < cfg.core_end:
            for slot, out_d in (("attn", h), ("k", kv_out), ("v", kv_out), ("mlp", h)):
                _add(f"{p}.controllers.{slot}.down_weight", (cfg.gate_rank, h))
                _add(f"{p}.controllers.{slot}.forget_weight", (out_d, cfg.gate_rank))
                _add(f"{p}.controllers.{slot}.gain_weight", (out_d, cfg.gate_rank))

        _add(f"{p}.mlp.gate.weight", (cfg.n_routed_experts, h))
        _add(f"{p}.mlp.gate.bias", (cfg.n_routed_experts,), f32=True, scale=0.01)
        _add(f"{p}.mlp.experts.w13_weight", (cfg.n_routed_experts, 2 * cfg.intermediate_size, h))
        _add(f"{p}.mlp.experts.w2_weight", (cfg.n_routed_experts, h, cfg.intermediate_size))

    if cfg.vision_layers > 0 and cfg.vision_mode != "strip":
        vh = cfg.vision_hidden_size
        v_inter = vh * 4
        _add("model.visual.backbone.embeddings.class_embedding", (vh,), scale=0.02)
        _add("model.visual.backbone.embeddings.cls_position_embedding", (vh,), scale=0.02)
        _add("model.visual.backbone.embeddings.patch_embedding.weight", (vh, 3 * 4 * 4))
        _add("model.visual.backbone.embeddings.position_embedding", (16, vh), scale=0.02)
        _add("model.visual.backbone.ln_pre.weight", (vh,), one=True)
        _add("model.visual.backbone.ln_pre.bias", (vh,), scale=0.01)
        _add("model.visual.proj_in.weight", (vh, vh))
        _add("model.visual.proj_out.weight", (h, vh))
        for vl in range(cfg.vision_layers):
            vp = f"model.visual.backbone.layers.{vl}"
            _add(f"{vp}.ln_1.weight", (vh,), one=True)
            _add(f"{vp}.ln_1.bias", (vh,), scale=0.01)
            _add(f"{vp}.ln_2.weight", (vh,), one=True)
            _add(f"{vp}.ln_2.bias", (vh,), scale=0.01)
            _add(f"{vp}.ls_1.gamma", (vh,), one=True)
            _add(f"{vp}.ls_2.gamma", (vh,), one=True)
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                _add(f"{vp}.self_attn.{proj}.weight", (vh, vh))
                _add(f"{vp}.self_attn.{proj}.bias", (vh,), scale=0.01)
            _add(f"{vp}.mlp.c_fc.weight", (v_inter, vh))
            _add(f"{vp}.mlp.c_fc.bias", (v_inter,), scale=0.01)
            _add(f"{vp}.mlp.c_proj.weight", (vh, v_inter))
            _add(f"{vp}.mlp.c_proj.bias", (vh,), scale=0.01)

    src_sf_dir = out_dir / "_src_safetensors"
    src_sf_dir.mkdir(parents=True, exist_ok=True)
    _write_safetensors_file(src_sf_dir / "model.safetensors", bb_tensors)
    (src_sf_dir / "config.json").write_bytes(cfg_path.read_bytes())
    (src_sf_dir / "tokenizer.json").write_bytes(tok_path.read_bytes())
    (src_sf_dir / "chat_template.jinja").write_bytes(tpl_path.read_bytes())

    model_amf = out_dir / "model.amf"
    vis_sidecar = (out_dir / "vision.amf") if (cfg.vision_layers > 0 and cfg.vision_mode == "sidecar") else None
    cold = _tail_cold_experts(cfg.num_hidden_layers, cfg.n_routed_experts, cfg.cold_experts_per_layer)
    bb_metas = pack_amf_from_safetensors(
        src_sf_dir,
        model_amf,
        model_id=cfg.model_id[:63],
        cold_experts=cold,
        sme2_pack=cfg.sme2_pack,
        strip_vision=True if cfg.vision_mode == "strip" else None,
        include_vision=True if cfg.vision_mode == "include" else None,
        vision_sidecar=vis_sidecar,
        embed_metadata=True,
    )
    enriched_cfg = build_inkling2_config_json(
        hf_cfg,
        num_layers=cfg.num_hidden_layers,
        sme2_controllers=(cfg.sme2_pack != "none" and is_loop),
    )
    enriched_bytes = (json.dumps(enriched_cfg, indent=2) + "\n").encode("utf-8")
    cfg_path.write_bytes(enriched_bytes)
    bb_metas = embed_metadata_into_amf(model_amf, config_bytes=enriched_bytes, output_path=model_amf)

    vis_metas: list[PackedTensorMeta] = []
    if vis_sidecar is not None and vis_sidecar.is_file():
        with AmfReader(vis_sidecar) as vr:
            vis_metas = list(vr.entries)

    dflare_amf: pathlib.Path | None = None
    dflare_cfg_path: pathlib.Path | None = None
    dflare_metas: list[PackedTensorMeta] = []
    if cfg.include_dflare and cfg.dflare_layers > 0:
        df_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

        def _add_df(name: str, shape: tuple[int, ...], *, one: bool = False) -> None:
            n = 1
            for d_s in shape:
                n *= d_s
            vals = _deterministic_f32(seed + 1000 + (zlib_hash(name) & 0xFFFF), n, scale=0.03, centered_one=one)
            df_tensors[name] = (sf_dt, shape, _encode_floats(vals, cfg.dtype))

        n_tgt = max(1, len(cfg.dflare_target_layer_ids))
        _add_df("fuse_weight", (h, n_tgt * h))
        _add_df("mask_embed", (h,))
        _add_df("norm.weight", (h,), one=True)
        for dl in range(cfg.dflare_layers):
            dp = f"layers.{dl}"
            _add_df(f"{dp}.dflare_norm.weight", (h,), one=True)
            _add_df(f"{dp}.input_layernorm.weight", (h,), one=True)
            _add_df(f"{dp}.post_attention_layernorm.weight", (h,), one=True)
            _add_df(f"{dp}.self_attn.q_proj.weight", (q_out, h))
            _add_df(f"{dp}.self_attn.k_proj.weight", (kv_out, h))
            _add_df(f"{dp}.self_attn.v_proj.weight", (kv_out, h))
            _add_df(f"{dp}.self_attn.k_proj_ctx.weight", (kv_out, h))
            _add_df(f"{dp}.self_attn.v_proj_ctx.weight", (kv_out, h))
            _add_df(f"{dp}.self_attn.o_proj.weight", (h, q_out))
            _add_df(f"{dp}.self_attn.q_norm.weight", (cfg.head_dim,), one=True)
            _add_df(f"{dp}.self_attn.k_norm.weight", (cfg.head_dim,), one=True)
            _add_df(f"{dp}.self_attn.k_norm_ctx.weight", (cfg.head_dim,), one=True)
            _add_df(f"{dp}.mlp.gate_proj.weight", (cfg.dflare_intermediate_size, h))
            _add_df(f"{dp}.mlp.up_proj.weight", (cfg.dflare_intermediate_size, h))
            _add_df(f"{dp}.mlp.down_proj.weight", (h, cfg.dflare_intermediate_size))
        _add_df("markov_head.dflare_norm.weight", (h,), one=True)
        _add_df("markov_head.markov_w1.weight", (cfg.dflare_markov_rank, h))
        _add_df("markov_head.rnn_proj.weight", (cfg.dflare_markov_rank, 2 * h))
        _add_df("markov_head.rnn_proj.bias", (cfg.dflare_markov_rank,))
        _add_df("markov_head.hidden_out.weight", (h, cfg.dflare_markov_rank))
        _add_df("markov_head.hidden_out.bias", (h,))

        mtp_sf = src_sf_dir / "mtp.safetensors"
        _write_safetensors_file(mtp_sf, df_tensors)
        dflare_amf, dflare_cfg_path, dflare_metas = export_dflare_from_safetensors(
            mtp_sf,
            out_dir / "dflare.amf",
            config_json=hf_cfg,
            config_output=out_dir / "dflare_config.json",
            model_id=f"{cfg.model_id[:55]}-dflare",
        )

    # Clean intermediate safetensors scratch dir so output_dir holds clean .amf + .json artifacts.
    for p_item in sorted(src_sf_dir.iterdir()):
        p_item.unlink()
    src_sf_dir.rmdir()

    total_bytes = model_amf.stat().st_size
    if vis_sidecar is not None and vis_sidecar.is_file():
        total_bytes += vis_sidecar.stat().st_size
    if dflare_amf is not None and dflare_amf.is_file():
        total_bytes += dflare_amf.stat().st_size

    return Inkling2SynthBundle(
        output_dir=out_dir,
        model_amf=model_amf,
        dflare_amf=dflare_amf,
        dflare_config_path=dflare_cfg_path,
        vision_amf=vis_sidecar if (vis_sidecar is not None and vis_sidecar.is_file()) else None,
        config_path=cfg_path,
        tokenizer_path=tok_path,
        backbone_metas=bb_metas,
        dflare_metas=dflare_metas,
        vision_metas=vis_metas,
        total_bytes=total_bytes,
    )
