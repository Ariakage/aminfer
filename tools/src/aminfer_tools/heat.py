"""Offline MoE expert activation heat profiler (W8, Milestone M4).

Pure standard-library Python implementation (ADR-0001). Reads `.amf` or
safetensors checkpoints with bounded host memory (`F_NOCACHE` + `os.pread`),
scores MoE router gates (`layers.{l}.mlp.gate`) over either:

1. A recorded router trace JSONL (`{"layer": l, "experts": [...]}`),
2. A token-ID corpus JSONL (`{"token_ids": [...]}` / `{"tokens": [...]}` /
   `{"input_ids": [...]}` / bare JSON int lists), scored against streamed
   `embed_tokens` rows + per-layer routers, or
3. A deterministic built-in representative multilingual/code token sample (fast:
   `< 5 s` on 48-layer `Qwen3-30B-A3B-3bit`),

and emits a schema-v1 JSON heat profile consumed by both `aminfer-tools
pack-amf --heat-profile` and Zig `Qwen3Model` startup loading.
"""

from __future__ import annotations

import collections
import json
import math
import os
import pathlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    NO_EXPERT,
    NO_LAYER,
    AmfReader,
    DType,
    PackedTensorMeta,
    TensorKind,
    _align_up,
    _collect_tensors,
    _quant_bits_for,
    _read_exact,
    _ShardReaders,
    _ShardTensor,
    _SWITCH_MLP_RE,
    _to_f16_chunk,
    bf16_bytes_to_f32_list,
    dequantize_q3_row_f32,
    dequantize_q4_row_f32,
    f16_bytes_to_f32_list,
    f32_bytes_to_f32_list,
    interleave_scale_bias_f16,
    resolve_checkpoint,
)

HEAT_SCHEMA_VERSION = 1
DEFAULT_TOP_K = 8
DEFAULT_SAMPLE_TOKENS = 48

_ROUTER_NAME_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.mlp\.gate(?:\.weight)?$")
_EXPERT_TENSOR_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.")

# Representative token ID anchors spanning ASCII/code syntax, high-frequency
# English/programming BPE merges, multilingual/CJK ranges, and long-tail domain
# vocabulary (calibrated for Qwen3 151,936 vocab; wrapped modulo `vocab_size`
# for smaller synthetic test models).
_REPRESENTATIVE_TOKEN_ANCHORS: tuple[int, ...] = (
    # ASCII / code syntax / whitespace / punctuation / digits
    11, 13, 15, 16, 17, 25, 26, 28, 40, 41, 61, 91, 93, 123, 125, 198,
    # High-frequency English & programming keywords/subwords
    220, 264, 279, 311, 315, 323, 369, 374, 389, 419, 458, 525, 700, 755, 872, 1084,
    # Code / technical / math / structured text subwords
    1493, 1988, 2684, 3452, 4710, 6144, 8192, 11234,
    # Multilingual / CJK / domain-specific vocabulary strata
    16384, 24576, 36864, 52428, 73728, 99840, 120000, 140288,
)


@dataclass(frozen=True)
class LayerHeatProfile:
    layer: int
    total_activations: int
    counts: tuple[int, ...]
    frequencies: tuple[float, ...]
    ranked_hot_to_cold: tuple[int, ...]
    cold_experts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "total_activations": self.total_activations,
            "counts": list(self.counts),
            "frequencies": list(self.frequencies),
            "ranked_hot_to_cold": list(self.ranked_hot_to_cold),
            "cold_experts": list(self.cold_experts),
        }


@dataclass(frozen=True)
class ExpertHeatProfile:
    schema_version: int
    model_id: str
    source_path: str
    calibration_mode: str
    sample_tokens: int
    num_layers: int
    num_experts: int
    top_k: int
    cold_experts_per_layer: int | None
    wired_budget_mib: float | None
    total_hot_experts: int
    total_cold_experts: int
    estimated_wired_bytes: int
    estimated_cold_bytes: int
    cold_spec_arg: str
    layers: tuple[LayerHeatProfile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "source_path": self.source_path,
            "calibration_mode": self.calibration_mode,
            "sample_tokens": self.sample_tokens,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "cold_experts_per_layer": self.cold_experts_per_layer,
            "wired_budget_mib": self.wired_budget_mib,
            "total_hot_experts": self.total_hot_experts,
            "total_cold_experts": self.total_cold_experts,
            "estimated_wired_bytes": self.estimated_wired_bytes,
            "estimated_cold_bytes": self.estimated_cold_bytes,
            "cold_spec_arg": self.cold_spec_arg,
            "layers": [lp.to_dict() for lp in self.layers],
        }


def _rms_normalize(vec: list[float], eps: float = 1e-6) -> list[float]:
    n = len(vec)
    if n == 0:
        return []
    mean_sq = math.sumprod(vec, vec) / n
    inv_rms = 1.0 / math.sqrt(mean_sq + eps)
    return [v * inv_rms for v in vec]


def _select_topk_indices(scores: Sequence[float], top_k: int) -> list[int]:
    """Select `top_k` indices with largest score (deterministic tie-break by lower index)."""
    n = len(scores)
    k = min(top_k, n)
    if k <= 0:
        return []
    if k == n:
        return sorted(range(n), key=lambda idx: (-scores[idx], idx))
    # Partial selection via sorted indices; n is typically 8..256 so sort is < 20 us.
    return sorted(range(n), key=lambda idx: (-scores[idx], idx))[:k]


def _default_sample_token_ids(vocab_size: int, count: int) -> list[int]:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    target = min(max(1, count), vocab_size)
    chosen: list[int] = []
    seen: set[int] = set()
    for anchor in _REPRESENTATIVE_TOKEN_ANCHORS:
        tid = anchor % vocab_size
        if tid not in seen:
            seen.add(tid)
            chosen.append(tid)
            if len(chosen) >= target:
                return chosen
    # Deterministically stride across remaining vocabulary if count > len(anchors).
    step = max(1, vocab_size // target)
    cur = 0
    while len(chosen) < target:
        tid = cur % vocab_size
        if tid not in seen:
            seen.add(tid)
            chosen.append(tid)
        cur += step
        if cur >= vocab_size * 2:
            # Linear sweep fallback if step and vocab_size share a common divisor.
            for fallback_tid in range(vocab_size):
                if fallback_tid not in seen:
                    seen.add(fallback_tid)
                    chosen.append(fallback_tid)
                    if len(chosen) >= target:
                        break
            break
    return chosen


def _synthetic_probe_vectors(hidden_size: int, count: int) -> list[list[float]]:
    """Deterministic normalized cosine/Walsh-like probe vectors when embed_tokens is absent."""
    probes: list[list[float]] = []
    inv_sqrt_h = 1.0 / math.sqrt(hidden_size)
    for p in range(count):
        freq = (p + 1) * math.pi / (hidden_size + 1)
        phase = 0.37 * (p + 1)
        vec = [math.cos(freq * (j + 0.5) + phase) * inv_sqrt_h for j in range(hidden_size)]
        probes.append(_rms_normalize(vec))
    return probes


class _ModelHeatSource:
    """Unified bounded-RAM view over `.amf` or safetensors routers and embeddings."""

    def __init__(self, path: pathlib.Path, *, model_id: str | None = None, group_size: int = 64) -> None:
        self.path = pathlib.Path(path)
        self.group_size = group_size
        self._amf: AmfReader | None = None
        self._shard_readers: _ShardReaders | None = None
        self._st_routers: dict[int, tuple[_ShardTensor, _ShardTensor | None, _ShardTensor | None]] = {}
        self._st_embed: tuple[_ShardTensor, _ShardTensor | None, _ShardTensor | None] | None = None
        self._amf_routers: dict[int, PackedTensorMeta] = {}
        self._amf_embed: PackedTensorMeta | None = None

        self.non_expert_bytes: int = 0
        self.expert_bytes: dict[tuple[int, int], int] = {}
        self.default_top_k: int | None = None

        if self.path.is_file() and self.path.suffix.lower() == ".amf":
            self._init_from_amf(model_id)
        else:
            self._init_from_safetensors(model_id)

        if not self.layer_ids:
            raise ValueError(f"no MoE router tensors (layers.*.mlp.gate) found in {self.path}")

    def _init_from_amf(self, model_id_override: str | None) -> None:
        self._amf = AmfReader(self.path)
        self.model_id = model_id_override or self._amf.header.model_id or self.path.stem
        for entry in self._amf.entries:
            if entry.kind == TensorKind.ROUTER:
                if entry.layer == NO_LAYER or len(entry.shape) != 2:
                    raise ValueError(f"malformed router tensor {entry.name!r} in {self.path}")
                self._amf_routers[entry.layer] = entry
            elif entry.kind == TensorKind.EMBEDDING and self._amf_embed is None:
                if len(entry.shape) == 2:
                    self._amf_embed = entry

            if entry.kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT) or entry.expert != NO_EXPERT:
                key = (entry.layer, entry.expert)
                self.expert_bytes[key] = self.expert_bytes.get(key, 0) + entry.blob_bytes
            else:
                self.non_expert_bytes += entry.blob_bytes

        self.layer_ids = sorted(self._amf_routers.keys())
        if self.layer_ids:
            first = self._amf_routers[self.layer_ids[0]]
            self.num_experts, self.hidden_size = first.shape
            for lid in self.layer_ids[1:]:
                sh = self._amf_routers[lid].shape
                if sh != (self.num_experts, self.hidden_size):
                    raise ValueError(
                        f"inconsistent router shapes: layer {self.layer_ids[0]} has {first.shape}, "
                        f"layer {lid} has {sh}"
                    )
        else:
            self.num_experts = 0
            self.hidden_size = 0
        self.vocab_size = self._amf_embed.shape[0] if self._amf_embed is not None else None

    def _init_from_safetensors(self, model_id_override: str | None) -> None:
        resolved = resolve_checkpoint(self.path)
        if resolved.config_path is not None:
            cfg = json.loads(resolved.config_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                if "num_experts_per_tok" in cfg:
                    self.default_top_k = int(cfg["num_experts_per_tok"])
                qcfg = cfg.get("quantization") or cfg.get("quantization_config")
                if isinstance(qcfg, dict) and "group_size" in qcfg:
                    self.group_size = int(qcfg["group_size"])
        self.model_id = model_id_override or (
            self.path.name if self.path.is_dir() else self.path.parent.name or self.path.stem
        )
        tensors = _collect_tensors(resolved.shard_paths)
        self._shard_readers = _ShardReaders()

        quant_prefixes: set[str] = set()
        for name in tensors:
            if name.endswith(".scales"):
                prefix = name[: -len(".scales")]
                if f"{prefix}.weight" in tensors and f"{prefix}.biases" in tensors:
                    quant_prefixes.add(prefix)

        consumed: set[str] = set()
        for prefix in sorted(quant_prefixes):
            w = tensors[f"{prefix}.weight"]
            s = tensors[f"{prefix}.scales"]
            b = tensors[f"{prefix}.biases"]
            consumed.update((w.name, s.name, b.name))
            rm = _ROUTER_NAME_RE.match(prefix)
            if rm is not None:
                self._st_routers[int(rm.group(1))] = (w, s, b)
            elif prefix in ("embed_tokens", "model.embed_tokens") and self._st_embed is None:
                self._st_embed = (w, s, b)

            # Estimate 16 KiB-aligned AMF blob sizes for wired-budget calculations.
            sw = _SWITCH_MLP_RE.match(prefix)
            if sw is not None and len(w.shape) == 3:
                num_e = w.shape[0]
                layer = int(sw.group(1))
                w_step = w.length // num_e
                s_step = (s.length // num_e) // (2 if s.dtype in ("F16", "BF16") else 4) * 4
                blob_sz = _align_up(_align_up(w_step, 64) + s_step, APPLE_PAGE_SIZE)
                for e in range(num_e):
                    self.expert_bytes[(layer, e)] = self.expert_bytes.get((layer, e), 0) + blob_sz
            else:
                em = _EXPERT_TENSOR_RE.match(prefix)
                params_sz = (s.length // (2 if s.dtype in ("F16", "BF16") else 4)) * 4
                blob_sz = _align_up(_align_up(w.length, 64) + params_sz, APPLE_PAGE_SIZE)
                if em is not None:
                    key = (int(em.group(1)), int(em.group(2)))
                    self.expert_bytes[key] = self.expert_bytes.get(key, 0) + blob_sz
                else:
                    self.non_expert_bytes += blob_sz

        for name, t in tensors.items():
            if name in consumed:
                continue
            rm = _ROUTER_NAME_RE.match(name)
            if rm is not None and int(rm.group(1)) not in self._st_routers:
                self._st_routers[int(rm.group(1))] = (t, None, None)
            elif name in ("embed_tokens.weight", "model.embed_tokens.weight") and self._st_embed is None:
                self._st_embed = (t, None, None)

            base = name[: -len(".weight")] if name.endswith(".weight") else name
            sw = _SWITCH_MLP_RE.match(base)
            if sw is not None and len(t.shape) == 3:
                num_e = t.shape[0]
                layer = int(sw.group(1))
                blob_sz = _align_up(t.length // num_e, APPLE_PAGE_SIZE)
                for e in range(num_e):
                    self.expert_bytes[(layer, e)] = self.expert_bytes.get((layer, e), 0) + blob_sz
            else:
                em = _EXPERT_TENSOR_RE.match(name)
                blob_sz = _align_up(max(t.length, 1), APPLE_PAGE_SIZE)
                if em is not None:
                    key = (int(em.group(1)), int(em.group(2)))
                    self.expert_bytes[key] = self.expert_bytes.get(key, 0) + blob_sz
                else:
                    self.non_expert_bytes += blob_sz

        self.layer_ids = sorted(self._st_routers.keys())
        if self.layer_ids:
            w0, s0, _ = self._st_routers[self.layer_ids[0]]
            self.num_experts, self.hidden_size = self._st_tensor_2d_shape(w0, s0)
        else:
            self.num_experts = 0
            self.hidden_size = 0
        if self._st_embed is not None:
            ew, es, _ = self._st_embed
            self.vocab_size = self._st_tensor_2d_shape(ew, es)[0]
        else:
            self.vocab_size = None

    def _st_tensor_2d_shape(self, w: _ShardTensor, s: _ShardTensor | None) -> tuple[int, int]:
        if len(w.shape) != 2:
            raise ValueError(f"expected 2D tensor for {w.name!r}, got {w.shape}")
        if s is not None:
            return (w.shape[0], s.shape[1] * self.group_size)
        return (w.shape[0], w.shape[1])

    def __enter__(self) -> _ModelHeatSource:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._amf is not None:
            self._amf.close()
            self._amf = None
        if self._shard_readers is not None:
            self._shard_readers.close()
            self._shard_readers = None

    def _read_st_rows_f32(
        self,
        triple: tuple[_ShardTensor, _ShardTensor | None, _ShardTensor | None],
        row_indices: Sequence[int],
    ) -> list[list[float]]:
        assert self._shard_readers is not None
        w, s, b = triple
        n_rows, k = self._st_tensor_2d_shape(w, s)
        w_fh = self._shard_readers.get(w.path)
        out: list[list[float]] = []

        if s is not None and b is not None:
            bits, _dtype = _quant_bits_for(w.name, w.shape[1], s.shape[1], self.group_size)
            row_w_bytes = w.length // n_rows
            row_s_bytes = s.length // n_rows
            s_fh = self._shard_readers.get(s.path)
            b_fh = self._shard_readers.get(b.path)
            deq = dequantize_q3_row_f32 if bits == 3 else dequantize_q4_row_f32
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {w.name!r} with {n_rows} rows")
                w_raw = _read_exact(w_fh, w.offset + r * row_w_bytes, row_w_bytes)
                s_raw = _read_exact(s_fh, s.offset + r * row_s_bytes, row_s_bytes)
                b_raw = _read_exact(b_fh, b.offset + r * row_s_bytes, row_s_bytes)
                params_f16 = interleave_scale_bias_f16(
                    _to_f16_chunk(s_raw, s.dtype, f"{w.name} scales"),
                    _to_f16_chunk(b_raw, b.dtype, f"{w.name} biases"),
                )
                out.append(deq(w_raw, params_f16, k, group_size=self.group_size))
            return out

        row_bytes = w.length // n_rows
        for r in row_indices:
            if r < 0 or r >= n_rows:
                raise IndexError(f"row {r} out of bounds for {w.name!r} with {n_rows} rows")
            raw = _read_exact(w_fh, w.offset + r * row_bytes, row_bytes)
            if w.dtype == "F16":
                out.append(f16_bytes_to_f32_list(raw))
            elif w.dtype == "BF16":
                out.append(bf16_bytes_to_f32_list(raw))
            elif w.dtype == "F32":
                out.append(f32_bytes_to_f32_list(raw))
            else:
                raise ValueError(f"unsupported router/embedding dtype {w.dtype} for {w.name!r}")
        return out

    def read_router_f32(self, layer: int) -> list[list[float]]:
        if self._amf is not None:
            return self._amf.read_matrix_f32(self._amf_routers[layer])
        triple = self._st_routers[layer]
        return self._read_st_rows_f32(triple, range(self.num_experts))

    def read_embed_rows_f32(self, token_ids: Sequence[int]) -> list[list[float]] | None:
        if self._amf is not None:
            if self._amf_embed is None:
                return None
            return self._amf.read_rows_f32(self._amf_embed, token_ids)
        if self._st_embed is None:
            return None
        return self._read_st_rows_f32(self._st_embed, token_ids)


def _parse_calibration_jsonl(
    path: pathlib.Path,
) -> tuple[str, dict[int, list[tuple[list[int], int]]], collections.Counter[int]]:
    """Parse calibration JSONL into either router traces or token-ID frequencies.

    Returns `(mode, router_traces, token_counter)` where `mode` is
    `"router_trace"` or `"token_jsonl"`.
    """
    router_traces: dict[int, list[tuple[list[int], int]]] = collections.defaultdict(list)
    token_counter: collections.Counter[int] = collections.Counter()
    mode: str | None = None

    with pathlib.Path(path).open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            if isinstance(obj, list):
                if mode is None:
                    mode = "token_jsonl"
                elif mode != "token_jsonl":
                    raise ValueError(f"{path}:{line_no}: cannot mix router-trace and token-ID records")
                for tok in obj:
                    if not isinstance(tok, int) or isinstance(tok, bool) or tok < 0:
                        raise ValueError(f"{path}:{line_no}: invalid token ID {tok!r}")
                    token_counter[tok] += 1
                continue

            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object or int list")

            if "layer" in obj and "experts" in obj:
                if mode is None:
                    mode = "router_trace"
                elif mode != "router_trace":
                    raise ValueError(f"{path}:{line_no}: cannot mix router-trace and token-ID records")
                layer = int(obj["layer"])
                weight = int(obj.get("weight", obj.get("count", 1)))
                if weight <= 0:
                    raise ValueError(f"{path}:{line_no}: weight/count must be positive, got {weight}")
                raw_experts = obj["experts"]
                if not isinstance(raw_experts, list):
                    raise ValueError(f"{path}:{line_no}: 'experts' must be a list")
                if raw_experts and isinstance(raw_experts[0], list):
                    for sub in raw_experts:
                        if not isinstance(sub, list):
                            raise ValueError(f"{path}:{line_no}: malformed nested 'experts' list")
                        router_traces[layer].append(([int(e) for e in sub], weight))
                else:
                    router_traces[layer].append(([int(e) for e in raw_experts], weight))
                continue

            tok_key = None
            for candidate in ("token_ids", "tokens", "input_ids"):
                if candidate in obj:
                    tok_key = candidate
                    break
            if tok_key is not None:
                if mode is None:
                    mode = "token_jsonl"
                elif mode != "token_jsonl":
                    raise ValueError(f"{path}:{line_no}: cannot mix router-trace and token-ID records")
                raw_toks = obj[tok_key]
                if not isinstance(raw_toks, list):
                    raise ValueError(f"{path}:{line_no}: {tok_key!r} must be a list of token IDs")
                weight = int(obj.get("weight", obj.get("count", 1)))
                if weight <= 0:
                    raise ValueError(f"{path}:{line_no}: weight/count must be positive, got {weight}")
                for tok in raw_toks:
                    if not isinstance(tok, int) or isinstance(tok, bool) or tok < 0:
                        raise ValueError(f"{path}:{line_no}: invalid token ID {tok!r}")
                    token_counter[tok] += weight
                continue

            raise ValueError(
                f"{path}:{line_no}: expected {{'layer', 'experts'}} or {{'token_ids'|'tokens'|'input_ids'}}"
            )

    if mode is None:
        raise ValueError(f"calibration JSONL {path} contains no records")
    return mode, router_traces, token_counter


def compute_expert_heat(
    input_path: pathlib.Path,
    *,
    calibration_jsonl: pathlib.Path | None = None,
    top_k: int | None = None,
    cold_experts_per_layer: int | None = None,
    wired_budget_mib: float | None = None,
    slot_reserve_mib: float = 0.0,
    sample_tokens: int = DEFAULT_SAMPLE_TOKENS,
    model_id: str | None = None,
    group_size: int = 64,
) -> ExpertHeatProfile:
    """Compute per-layer MoE expert activation heat statistics and hot/cold split."""
    if cold_experts_per_layer is not None and wired_budget_mib is not None:
        raise ValueError("specify at most one of cold_experts_per_layer and wired_budget_mib")
    if sample_tokens <= 0:
        raise ValueError(f"sample_tokens must be positive, got {sample_tokens}")
    if slot_reserve_mib < 0.0:
        raise ValueError(f"slot_reserve_mib must be non-negative, got {slot_reserve_mib}")

    with _ModelHeatSource(input_path, model_id=model_id, group_size=group_size) as src:
        eff_top_k = top_k if top_k is not None else (src.default_top_k or DEFAULT_TOP_K)
        if eff_top_k <= 0 or eff_top_k > src.num_experts:
            raise ValueError(f"top_k={eff_top_k} out of range 1..{src.num_experts}")
        if cold_experts_per_layer is not None and not (0 <= cold_experts_per_layer <= src.num_experts):
            raise ValueError(
                f"cold_experts_per_layer={cold_experts_per_layer} out of range 0..{src.num_experts}"
            )

        counts_by_layer: dict[int, list[int]] = {l: [0] * src.num_experts for l in src.layer_ids}
        norms_by_layer: dict[int, list[float]] = {l: [0.0] * src.num_experts for l in src.layer_ids}

        if calibration_jsonl is not None:
            cal_mode, router_traces, token_counter = _parse_calibration_jsonl(calibration_jsonl)
        else:
            cal_mode = "builtin_sample"
            router_traces = {}
            token_counter = collections.Counter()

        if cal_mode == "router_trace":
            total_sample_tokens = 0
            for layer_id, entries in router_traces.items():
                if layer_id not in counts_by_layer:
                    raise ValueError(
                        f"router trace references layer {layer_id}, not present in model ({src.layer_ids[0]}..{src.layer_ids[-1]})"
                    )
                layer_counts = counts_by_layer[layer_id]
                for experts, weight in entries:
                    total_sample_tokens += weight
                    for e in experts:
                        if e < 0 or e >= src.num_experts:
                            raise ValueError(
                                f"router trace expert index {e} out of range 0..{src.num_experts - 1}"
                            )
                        layer_counts[e] += weight
            # Still use router L2 norms as a deterministic secondary tie-breaker for zero-count experts.
            for layer_id in src.layer_ids:
                router = src.read_router_f32(layer_id)
                norms_by_layer[layer_id] = [math.sqrt(math.sumprod(row, row)) for row in router]
            eff_sample_tokens = max(1, total_sample_tokens // max(1, len(src.layer_ids)))
            calibration_mode = "router_trace"
        else:
            if cal_mode == "token_jsonl":
                if src.vocab_size is None:
                    raise ValueError(
                        f"token-ID calibration requires embed_tokens, which was not found in {input_path}"
                    )
                for tid in token_counter:
                    if tid >= src.vocab_size:
                        raise ValueError(
                            f"calibration token ID {tid} exceeds model vocab_size {src.vocab_size}"
                        )
                unique_tids = sorted(token_counter.keys())
                weights = [token_counter[tid] for tid in unique_tids]
                raw_embeds = src.read_embed_rows_f32(unique_tids)
                assert raw_embeds is not None
                probe_vecs = [_rms_normalize(v) for v in raw_embeds]
                eff_sample_tokens = sum(weights)
                calibration_mode = "token_jsonl_embed_router"
            else:
                if src.vocab_size is not None:
                    tids = _default_sample_token_ids(src.vocab_size, sample_tokens)
                    raw_embeds = src.read_embed_rows_f32(tids)
                    assert raw_embeds is not None
                    probe_vecs = [_rms_normalize(v) for v in raw_embeds]
                    weights = [1] * len(probe_vecs)
                    eff_sample_tokens = len(probe_vecs)
                    calibration_mode = "embed_router_sample"
                else:
                    probe_vecs = _synthetic_probe_vectors(src.hidden_size, sample_tokens)
                    weights = [1] * len(probe_vecs)
                    eff_sample_tokens = len(probe_vecs)
                    calibration_mode = "router_probe_sample"

            # Stream one layer router at a time to keep host RAM tiny (< 4 MiB).
            for layer_id in src.layer_ids:
                router = src.read_router_f32(layer_id)
                norms_by_layer[layer_id] = [math.sqrt(math.sumprod(row, row)) for row in router]
                layer_counts = counts_by_layer[layer_id]
                for x_vec, weight in zip(probe_vecs, weights):
                    logits = [math.sumprod(x_vec, r_row) for r_row in router]
                    for e in _select_topk_indices(logits, eff_top_k):
                        layer_counts[e] += weight

        # Rank hot -> cold per layer deterministically.
        ranked_by_layer: dict[int, list[int]] = {}
        freq_by_layer: dict[int, list[float]] = {}
        for layer_id in src.layer_ids:
            counts = counts_by_layer[layer_id]
            norms = norms_by_layer[layer_id]
            total = sum(counts)
            freq_by_layer[layer_id] = [
                round(c / total, 6) if total > 0 else 0.0 for c in counts
            ]
            ranked_by_layer[layer_id] = sorted(
                range(src.num_experts),
                key=lambda e: (-counts[e], -norms[e], e),
            )

        # Select cold experts per layer according to policy.
        cold_by_layer: dict[int, list[int]] = {l: [] for l in src.layer_ids}
        if cold_experts_per_layer is not None:
            for layer_id in src.layer_ids:
                ranked = ranked_by_layer[layer_id]
                if cold_experts_per_layer > 0:
                    cold_by_layer[layer_id] = sorted(ranked[src.num_experts - cold_experts_per_layer :])
        elif wired_budget_mib is not None:
            if wired_budget_mib <= 0.0:
                raise ValueError(f"wired_budget_mib must be positive, got {wired_budget_mib}")
            budget_bytes = int(wired_budget_mib * 1024.0 * 1024.0)
            reserve_bytes = int(slot_reserve_mib * 1024.0 * 1024.0)
            avail_hot_bytes = max(0, budget_bytes - src.non_expert_bytes - reserve_bytes)
            all_pairs = [
                (layer_id, e)
                for layer_id in src.layer_ids
                for e in range(src.num_experts)
            ]
            all_pairs.sort(
                key=lambda le: (
                    -freq_by_layer[le[0]][le[1]],
                    -counts_by_layer[le[0]][le[1]],
                    -norms_by_layer[le[0]][le[1]],
                    le[0],
                    le[1],
                )
            )
            used_hot_bytes = 0
            for layer_id, e in all_pairs:
                eb = src.expert_bytes.get((layer_id, e), 0)
                if used_hot_bytes + eb <= avail_hot_bytes:
                    used_hot_bytes += eb
                else:
                    cold_by_layer[layer_id].append(e)
            for layer_id in src.layer_ids:
                cold_by_layer[layer_id].sort()

        layer_profiles: list[LayerHeatProfile] = []
        cold_spec_parts: list[str] = []
        total_cold = 0
        total_hot = 0
        est_cold_bytes = 0
        est_hot_expert_bytes = 0
        for layer_id in src.layer_ids:
            counts = counts_by_layer[layer_id]
            freqs = freq_by_layer[layer_id]
            ranked = ranked_by_layer[layer_id]
            cold_list = cold_by_layer[layer_id]
            cold_set_l = set(cold_list)
            total_cold += len(cold_list)
            total_hot += src.num_experts - len(cold_list)
            for e in range(src.num_experts):
                eb = src.expert_bytes.get((layer_id, e), 0)
                if e in cold_set_l:
                    est_cold_bytes += eb
                else:
                    est_hot_expert_bytes += eb
            for e in cold_list:
                cold_spec_parts.append(f"{layer_id}:{e}")
            layer_profiles.append(
                LayerHeatProfile(
                    layer=layer_id,
                    total_activations=sum(counts),
                    counts=tuple(counts),
                    frequencies=tuple(freqs),
                    ranked_hot_to_cold=tuple(ranked),
                    cold_experts=tuple(cold_list),
                )
            )

        return ExpertHeatProfile(
            schema_version=HEAT_SCHEMA_VERSION,
            model_id=src.model_id,
            source_path=str(src.path),
            calibration_mode=calibration_mode,
            sample_tokens=eff_sample_tokens,
            num_layers=len(src.layer_ids),
            num_experts=src.num_experts,
            top_k=eff_top_k,
            cold_experts_per_layer=cold_experts_per_layer,
            wired_budget_mib=wired_budget_mib,
            total_hot_experts=total_hot,
            total_cold_experts=total_cold,
            estimated_wired_bytes=src.non_expert_bytes + est_hot_expert_bytes,
            estimated_cold_bytes=est_cold_bytes,
            cold_spec_arg=",".join(cold_spec_parts),
            layers=tuple(layer_profiles),
        )


def write_heat_profile(profile: ExpertHeatProfile, output_path: pathlib.Path) -> pathlib.Path:
    """Write a heat profile JSON atomically."""
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    payload = json.dumps(profile.to_dict(), indent=2) + "\n"
    try:
        partial.write_text(payload, encoding="utf-8")
        os.replace(partial, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return out
