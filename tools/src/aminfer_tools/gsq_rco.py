"""Offline GSQ + RCO per-tensor / per-expert hybrid quantization for `.amf` v0 (ADR-0017, W14).

Pure Python standard-library implementation (zero PyTorch/numpy runtime or offline
dependencies). Implements:

- **RCO (Residual Compensation Optimization)**: group-64 affine quantization for
  `q2_affine`, `q3_affine`, and `q4_affine` combining clipped range exploration,
  alternating closed-form ridge-stabilized OLS `(scale, bias)` refinement, and
  exact IEEE-754 float16 parameter compensation. Provably guarantees
  `mse_rco <= mse_minmax` on every group.
- **GSQ (Sensitivity-Guided Quantization)**: bounded-RAM streaming sensitivity
  scoring combining layer-norm Hessian proxies, sampled weight outlier/kurtosis
  statistics, empirical RCO distortion curves, architectural role weights,
  layer U-curve weighting, and optional `W8` `heat.json` MoE routing frequencies,
  followed by a lower-convex-hull discrete knapsack allocator across
  `{q2_affine, q3_affine, q4_affine, f16/bf16}`.
- **Strict per-tensor / per-expert granularity (ADR-0017)**: zero intra-tensor bit
  mixing; every MoE expert triple `(gate_proj, up_proj, down_proj)` is treated as
  one atomic quantization unit so runtime `G13` CSR bucketing dispatches one
  uniform `WeightDType` per expert.
"""

from __future__ import annotations

import concurrent.futures
import heapq
import json
import math
import os
import pathlib
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DTYPE_MODES,
    ENTRY_STRUCT,
    NO_EXPERT,
    NO_LAYER,
    SME2_GROUP_K,
    SME2_PACK_MODES,
    SME2_PANEL_COLS,
    AmfReader,
    DType,
    PackedTensorMeta,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    _align_up,
    _BlobWriter,
    _collect_tensors,
    _crc32,
    _load_tie_flag,
    _pack_directory,
    _pack_header,
    _PlannedEntry,
    _read_exact,
    _set_nocache,
    _ShardReaders,
    _should_sme2_pack,
    _stream_entry,
    bf16_bytes_to_f16_bytes,
    bf16_bytes_to_f32_list,
    dequantize_q2_row_f32,
    dequantize_q3_row_f32,
    dequantize_q4_row_f32,
    f16_bytes_to_bf16_bytes,
    f16_bytes_to_f32_list,
    f32_bytes_to_f32_list,
    load_cold_experts_from_heat_profile,
    pack_sme2_panel64_f16_panel,
    pack_sme2_panel64_q2_from_mlx_panel,
    pack_sme2_panel64_q3_panel,
    pack_sme2_panel64_q4_panel,
    plan_amf_entries,
    resolve_checkpoint,
    sme2_panel64_data_bytes,
)

_F16_STRUCT = struct.Struct("<e")
_F16_PAIR_STRUCT = struct.Struct("<2e")
_F16_MAX = 65504.0
_QUANT_DTYPES = (DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE)
_DTYPE_BITS_RANK = {
    DType.Q2_AFFINE: 2,
    DType.Q3_AFFINE: 3,
    DType.Q4_AFFINE: 4,
    DType.F16: 16,
    DType.BF16: 16,
    DType.F32: 32,
    DType.U32: 32,
}
_ROW_BATCH_SIZE = 128  # multiple of SME2_PANEL_COLS (16); bounds streaming RAM


def _clamp_f16(val: float) -> float:
    if not math.isfinite(val):
        return 0.0
    if val > _F16_MAX:
        return _F16_MAX
    if val < -_F16_MAX:
        return -_F16_MAX
    return _F16_STRUCT.unpack(_F16_STRUCT.pack(val))[0]


def _round_pair_f16(s: float, b: float) -> tuple[float, float]:
    if not math.isfinite(s) or s < 0.0:
        s = 0.0
    elif s > _F16_MAX:
        s = _F16_MAX
    if not math.isfinite(b):
        b = 0.0
    elif b > _F16_MAX:
        b = _F16_MAX
    elif b < -_F16_MAX:
        b = -_F16_MAX
    return _F16_PAIR_STRUCT.unpack(_F16_PAIR_STRUCT.pack(s, b))


def _eval_group_with_sb(
    values: Sequence[float],
    sum_w: float,
    sum_w2: float,
    s_h: float,
    b_h: float,
    qmax: int,
) -> tuple[list[int], float, int, int, float]:
    """Quantize `values` under exact IEEE-f16 `(s_h, b_h)` and compute exact MSE."""
    n = len(values)
    if s_h <= 0.0:
        codes = [0] * n
        mse = max(0.0, (sum_w2 - 2.0 * b_h * sum_w + n * b_h * b_h) / n)
        return codes, mse, 0, 0, 0.0

    inv_s = 1.0 / s_h
    codes = [0] * n
    sum_q = 0
    sum_q2 = 0
    for i, x in enumerate(values):
        v = (x - b_h) * inv_s
        if v <= 0.0:
            q = 0
        else:
            q = int(v + 0.5)
            if q > qmax:
                q = qmax
        codes[i] = q
        sum_q += q
        sum_q2 += q * q

    sum_wq = math.sumprod(values, codes)
    # Exact sum of squared errors via residual formula (immune to catastrophic cancellation).
    err2 = 0.0
    for x, q in zip(values, codes):
        d = x - (b_h + q * s_h)
        err2 += d * d
    return codes, err2 / n, sum_q, sum_q2, sum_wq


def quantize_group_minmax(
    values: Sequence[float],
    bits: int,
) -> tuple[float, float, list[int], float]:
    """Naive group affine min-max quantization (`w ~= bias_f16 + q * scale_f16`).

    Returns `(scale_f16, bias_f16, codes, mse)`.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported quant bits {bits} (expected 2, 3, or 4)")
    n = len(values)
    if n == 0:
        raise ValueError("group values must be non-empty")

    qmax = (1 << bits) - 1
    lo = min(values)
    hi = max(values)
    sum_w = math.fsum(values)
    sum_w2 = math.sumprod(values, values)

    if hi <= lo:
        b_h = _clamp_f16(lo)
        codes, mse, _, _, _ = _eval_group_with_sb(values, sum_w, sum_w2, 0.0, b_h, qmax)
        return 0.0, b_h, codes, mse

    s_h, b_h = _round_pair_f16((hi - lo) / qmax, lo)
    if s_h <= 0.0:
        # Sub-f16 dynamic range: represent by constant group mean.
        b_h = _clamp_f16(sum_w / n)
        codes, mse, _, _, _ = _eval_group_with_sb(values, sum_w, sum_w2, 0.0, b_h, qmax)
        return 0.0, b_h, codes, mse

    codes, mse, _, _, _ = _eval_group_with_sb(values, sum_w, sum_w2, s_h, b_h, qmax)
    return s_h, b_h, codes, mse


def quantize_group_rco(
    values: Sequence[float],
    bits: int,
    *,
    iters: int = 3,
) -> tuple[float, float, list[int], float]:
    """Residual Compensation Optimization (RCO) for group affine quantization (ADR-0017).

    Refines `(scale_f16, bias_f16, codes)` over clipped initializers using
    alternating ridge-stabilized closed-form OLS residual compensation and exact
    IEEE-754 float16 rounding verification. Provably guarantees `mse_rco <= mse_minmax`.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported quant bits {bits} (expected 2, 3, or 4)")
    n = len(values)
    if n == 0:
        raise ValueError("group values must be non-empty")

    qmax = (1 << bits) - 1
    lo = min(values)
    hi = max(values)
    sum_w = math.fsum(values)
    sum_w2 = math.sumprod(values, values)

    # Always initialize `best` with exact min-max so `mse_rco <= mse_minmax` is invariant.
    if hi <= lo:
        b_h = _clamp_f16(lo)
        codes, mse, _, _, _ = _eval_group_with_sb(values, sum_w, sum_w2, 0.0, b_h, qmax)
        return 0.0, b_h, codes, mse

    s_mm, b_mm = _round_pair_f16((hi - lo) / qmax, lo)
    if s_mm <= 0.0:
        b_mean = _clamp_f16(sum_w / n)
        codes, mse, _, _, _ = _eval_group_with_sb(values, sum_w, sum_w2, 0.0, b_mean, qmax)
        return 0.0, b_mean, codes, mse

    best_codes, best_mse, mm_sq, mm_sq2, mm_swq = _eval_group_with_sb(values, sum_w, sum_w2, s_mm, b_mm, qmax)
    best_s = s_mm
    best_b = b_mm
    if best_mse == 0.0 or iters <= 0:
        return best_s, best_b, best_codes, best_mse

    mean = sum_w / n
    var = max(0.0, sum_w2 / n - mean * mean)
    std = math.sqrt(var)
    span = hi - lo

    # Build compact candidate initializers targeting outlier shrinkage where it pays most.
    starts: list[tuple[float, float, list[int] | None, int, int, float]] = [
        (s_mm, b_mm, best_codes, mm_sq, mm_sq2, mm_swq),
    ]
    if std > 0.0:
        if bits == 2:
            # 2-bit has only 4 levels; Lloyd-Max Gaussian optimal span is ~+/-1.8..2.2 sigma.
            for mult in (1.75, 2.25):
                c_lo = max(lo, mean - mult * std)
                c_hi = min(hi, mean + mult * std)
                if c_hi > c_lo:
                    s0, b0 = _round_pair_f16((c_hi - c_lo) / qmax, c_lo)
                    if s0 > 0.0 and (s0 != s_mm or b0 != b_mm):
                        starts.append((s0, b0, None, 0, 0, 0.0))
        elif bits == 3:
            if span > 4.2 * std:
                for mult in (2.35, 2.85):
                    c_lo = max(lo, mean - mult * std)
                    c_hi = min(hi, mean + mult * std)
                    if c_hi > c_lo:
                        s0, b0 = _round_pair_f16((c_hi - c_lo) / qmax, c_lo)
                        if s0 > 0.0 and (s0 != s_mm or b0 != b_mm):
                            starts.append((s0, b0, None, 0, 0, 0.0))
        else:  # bits == 4
            if span > 4.8 * std:
                mid = 0.5 * (lo + hi)
                for alpha in (0.84, 0.92):
                    half = 0.5 * span * alpha
                    s0, b0 = _round_pair_f16((2.0 * half) / qmax, mid - half)
                    if s0 > 0.0 and (s0 != s_mm or b0 != b_mm):
                        starts.append((s0, b0, None, 0, 0, 0.0))

    inv_n = 1.0 / n
    for s_cur, b_cur, pre_codes, sum_q, sum_q2, sum_wq in starts:
        if pre_codes is None:
            codes, mse, sum_q, sum_q2, sum_wq = _eval_group_with_sb(values, sum_w, sum_w2, s_cur, b_cur, qmax)
            if mse < best_mse:
                best_mse = mse
                best_s = s_cur
                best_b = b_cur
                best_codes = codes
        else:
            codes = pre_codes

        for _ in range(iters):
            denom = n * sum_q2 - sum_q * sum_q
            if denom <= 0:
                break
            # Closed-form ridge-stabilized OLS residual compensation:
            # minimize sum_i (w_i - (b' + s' * q_i))^2 + lam * (s' - s_cur)^2
            lam = 1e-12 * max(1.0, float(sum_q2))
            s_new = (n * sum_wq - sum_q * sum_w + lam * s_cur) / (denom + lam)
            if s_new <= 0.0:
                break
            b_new = (sum_w - s_new * sum_q) * inv_n
            s_h, b_h = _round_pair_f16(s_new, b_new)
            if s_h <= 0.0:
                break
            if s_h == s_cur and b_h == b_cur:
                break
            new_codes, new_mse, sum_q, sum_q2, sum_wq = _eval_group_with_sb(values, sum_w, sum_w2, s_h, b_h, qmax)
            if new_mse < best_mse:
                best_mse = new_mse
                best_s = s_h
                best_b = b_h
                best_codes = new_codes
            same_codes = new_codes == codes
            s_cur = s_h
            b_cur = b_h
            codes = new_codes
            if same_codes:
                break

    # Final exact bias-residual compensation check under the chosen f16 scale.
    if best_s > 0.0:
        sum_q_best = sum(best_codes)
        b_opt = _clamp_f16((sum_w - best_s * sum_q_best) * inv_n)
        if b_opt != best_b:
            opt_codes, opt_mse, _, _, _ = _eval_group_with_sb(values, sum_w, sum_w2, best_s, b_opt, qmax)
            if opt_mse < best_mse:
                best_mse = opt_mse
                best_b = b_opt
                best_codes = opt_codes

    return best_s, best_b, best_codes, best_mse


def _pack_group_codes(codes: Sequence[int], bits: int, dst: bytearray, dst_off: int) -> None:
    """Pack one group's unsigned codes LSB-first straight into `dst[dst_off..]`."""
    n = len(codes)
    if bits == 4:
        for i in range(0, n, 2):
            dst[dst_off] = (codes[i] & 0x0F) | ((codes[i + 1] & 0x0F) << 4)
            dst_off += 1
    elif bits == 3:
        for i in range(0, n, 8):
            word = (
                (codes[i] & 0x07)
                | ((codes[i + 1] & 0x07) << 3)
                | ((codes[i + 2] & 0x07) << 6)
                | ((codes[i + 3] & 0x07) << 9)
                | ((codes[i + 4] & 0x07) << 12)
                | ((codes[i + 5] & 0x07) << 15)
                | ((codes[i + 6] & 0x07) << 18)
                | ((codes[i + 7] & 0x07) << 21)
            )
            dst[dst_off] = word & 0xFF
            dst[dst_off + 1] = (word >> 8) & 0xFF
            dst[dst_off + 2] = (word >> 16) & 0xFF
            dst_off += 3
    elif bits == 2:
        for i in range(0, n, 4):
            dst[dst_off] = (
                (codes[i] & 0x03)
                | ((codes[i + 1] & 0x03) << 2)
                | ((codes[i + 2] & 0x03) << 4)
                | ((codes[i + 3] & 0x03) << 6)
            )
            dst_off += 1
    else:
        raise ValueError(f"unsupported quant bits {bits}")


def quantize_rows_affine(
    rows: Sequence[Sequence[float]],
    bits: int,
    *,
    group_size: int = 64,
    use_rco: bool = True,
    rco_iters: int = 3,
    compute_minmax_mse: bool = False,
) -> tuple[bytes, bytes, float, float]:
    """Quantize a batch of 2D rows `[n_rows, k]` into row-major MLX affine bytes.

    Returns `(qweight_bytes, scale_bias_f16_bytes, mean_mse, mean_minmax_mse)`.
    When `compute_minmax_mse` is False and `use_rco` is True, `mean_minmax_mse`
    equals `mean_mse` (avoiding redundant baseline passes on large checkpoints).
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported quant bits {bits}")
    if not rows:
        return b"", b"", 0.0, 0.0
    k = len(rows[0])
    if k <= 0 or k % group_size != 0:
        raise ValueError(f"quantized row width k={k} must be a positive multiple of {group_size}")
    n_rows = len(rows)
    groups = k // group_size
    grp_data_bytes = (group_size * bits) // 8
    row_data_bytes = groups * grp_data_bytes
    row_param_bytes = groups * 4

    q_out = bytearray(n_rows * row_data_bytes)
    p_out = bytearray(n_rows * row_param_bytes)
    pack_sb = _F16_PAIR_STRUCT.pack_into

    mse_sum = 0.0
    mm_mse_sum = 0.0
    total_groups = n_rows * groups

    for r_idx, row in enumerate(rows):
        if len(row) != k:
            raise ValueError(f"row {r_idx} has length {len(row)}, expected {k}")
        d_base = r_idx * row_data_bytes
        p_base = r_idx * row_param_bytes
        for g in range(groups):
            grp = row[g * group_size : (g + 1) * group_size]
            if use_rco:
                s_h, b_h, codes, mse = quantize_group_rco(grp, bits, iters=rco_iters)
                mse_sum += mse
                if compute_minmax_mse:
                    _, _, _, mm_mse = quantize_group_minmax(grp, bits)
                    mm_mse_sum += mm_mse
            else:
                s_h, b_h, codes, mse = quantize_group_minmax(grp, bits)
                mse_sum += mse
                mm_mse_sum += mse
            _pack_group_codes(codes, bits, q_out, d_base + g * grp_data_bytes)
            pack_sb(p_out, p_base + g * 4, s_h, b_h)

    mean_mse = mse_sum / total_groups
    mean_mm = (mm_mse_sum / total_groups) if (compute_minmax_mse or not use_rco) else mean_mse
    return bytes(q_out), bytes(p_out), mean_mse, mean_mm


def _quantize_worker_task(args: tuple[list[list[float]], int, int, bool, int]) -> tuple[bytes, bytes, float]:
    rows, bits, group_size, use_rco, rco_iters = args
    q_bytes, p_bytes, mse, _ = quantize_rows_affine(
        rows, bits, group_size=group_size, use_rco=use_rco, rco_iters=rco_iters
    )
    return q_bytes, p_bytes, mse


# ---------------------------------------------------------------------------
# GSQ Sensitivity Scoring & Discrete Convex-Hull Knapsack Allocator
# ---------------------------------------------------------------------------


def _effective_payload_bytes(dtype: DType, shape: tuple[int, ...], group_size: int) -> tuple[int, int, int]:
    """Return `(data_bytes, params_offset, params_bytes)` for row-major storage."""
    numel = 1
    for d in shape:
        numel *= d
    if dtype in (DType.F16, DType.BF16):
        return numel * 2, 0, 0
    if dtype in (DType.F32, DType.U32):
        return numel * 4, 0, 0
    if len(shape) != 2:
        raise ValueError(f"quantized dtype {dtype.name} requires 2D shape, got {shape}")
    n, k = shape
    if k % group_size != 0:
        raise ValueError(f"shape ({n}, {k}) incompatible with group_size={group_size}")
    bits = _DTYPE_BITS_RANK[dtype]
    data_bytes = n * ((k * bits) // 8)
    params_offset = _align_up(data_bytes, 64)
    params_bytes = n * (k // group_size) * 4
    return data_bytes, params_offset, params_bytes


def _algorithmic_bytes(dtype: DType, shape: tuple[int, ...], group_size: int) -> int:
    d, _, p = _effective_payload_bytes(dtype, shape, group_size)
    return d + p


def _role_of_tensor(name: str, kind: TensorKind) -> str:
    if kind == TensorKind.NORM:
        return "norm"
    if kind == TensorKind.ROUTER:
        return "router"
    if kind == TensorKind.EMBEDDING:
        return "embedding"
    if kind == TensorKind.LM_HEAD:
        return "lm_head"
    if kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT):
        return "moe_expert"
    if ".self_attn." in name:
        if ".v_proj" in name or ".o_proj" in name:
            return "attn_vo"
        return "attn_qk"
    if ".mlp." in name:
        return "dense_ffn"
    return "dense"


_ROLE_WEIGHT = {
    "router": 10.0,
    "embedding": 4.0,
    "lm_head": 4.0,
    "attn_vo": 3.0,
    "attn_qk": 2.4,
    "dense_ffn": 1.0,
    "moe_expert": 1.0,
    "dense": 1.2,
    "norm": 100.0,
}


@dataclass(frozen=True)
class QuantUnitCandidate:
    dtype: DType
    alg_bytes: int
    weighted_distortion: float
    raw_mse: float


@dataclass(frozen=True)
class QuantUnitInput:
    unit_id: str
    tensor_names: tuple[str, ...]
    role: str
    layer: int
    expert: int
    numel: int
    sensitivity: float
    candidates: tuple[QuantUnitCandidate, ...]  # ordered by increasing alg_bytes


@dataclass(frozen=True)
class QuantUnitPlan:
    unit_id: str
    tensor_names: tuple[str, ...]
    role: str
    layer: int
    expert: int
    numel: int
    sensitivity: float
    chosen_dtype: DType
    alg_bytes: int
    weighted_distortion: float
    raw_mse: float


@dataclass(frozen=True)
class GsqAllocationReport:
    target_bits: float
    effective_bits: float
    total_weights: int
    total_alg_bytes: int
    pinned_weights: int
    pinned_alg_bytes: int
    units: tuple[QuantUnitPlan, ...]
    dtype_counts: dict[str, int]
    dtype_weight_share: dict[str, float]


def _best_hull_step(
    cands: Sequence[QuantUnitCandidate],
    cur_idx: int,
) -> tuple[float, int] | None:
    """Find the upgrade `j > cur_idx` maximizing marginal distortion reduction per byte."""
    cur = cands[cur_idx]
    best_gain = 0.0
    best_j = -1
    for j in range(cur_idx + 1, len(cands)):
        nxt = cands[j]
        db = nxt.alg_bytes - cur.alg_bytes
        if db <= 0:
            continue
        dd = cur.weighted_distortion - nxt.weighted_distortion
        gain = dd / db if dd > 0.0 else 0.0
        if gain > best_gain:
            best_gain = gain
            best_j = j
    if best_j < 0:
        return None
    return best_gain, best_j


def solve_gsq_allocation(
    units: Sequence[QuantUnitInput],
    *,
    target_bits: float,
    pinned_weights: int = 0,
    pinned_alg_bytes: int = 0,
) -> GsqAllocationReport:
    """Discrete convex-hull marginal-distortion-per-byte knapsack solver (ADR-0017).

    Starts every quantizable unit at its lowest-bit candidate (`q2_affine`),
    applies sensitivity-guided protection floors when the budget permits
    (routers -> highest precision, embeddings/lm_head -> >= `q4_affine`), and
    then greedily upgrades units with the highest distortion reduction per extra
    byte until `target_bits` is reached.
    """
    if target_bits <= 0.0:
        raise ValueError(f"target_bits must be positive, got {target_bits}")

    total_weights = pinned_weights + sum(u.numel for u in units)
    if total_weights <= 0:
        raise ValueError("model has zero weights")
    budget_bytes = int(math.floor(target_bits * total_weights / 8.0 + 1e-9))

    chosen_idx = [0] * len(units)
    cur_bytes = pinned_alg_bytes + sum(u.candidates[0].alg_bytes for u in units)

    # Protection pass 1: routers are tiny (<0.05% of model) and directly control MoE routing.
    for idx, u in enumerate(units):
        if u.role == "router" and len(u.candidates) > 1:
            top_j = len(u.candidates) - 1
            extra = u.candidates[top_j].alg_bytes - u.candidates[0].alg_bytes
            if cur_bytes + extra <= budget_bytes:
                chosen_idx[idx] = top_j
                cur_bytes += extra

    # Protection pass 2: protect embedding / lm_head at >= q4_affine when budget allows
    # while leaving room for remaining units at their base allocation.
    for idx, u in enumerate(units):
        if u.role in ("embedding", "lm_head"):
            target_j = chosen_idx[idx]
            for j, c in enumerate(u.candidates):
                if c.dtype == DType.Q4_AFFINE:
                    target_j = j
                    break
            if target_j > chosen_idx[idx]:
                extra = u.candidates[target_j].alg_bytes - u.candidates[chosen_idx[idx]].alg_bytes
                if cur_bytes + extra <= budget_bytes:
                    chosen_idx[idx] = target_j
                    cur_bytes += extra

    # Main greedy marginal-distortion-per-byte knapsack over lower convex hull.
    # Tie-break deterministically by (-sensitivity, unit_id).
    heap: list[tuple[float, float, str, int, int, int]] = []
    for idx, u in enumerate(units):
        step = _best_hull_step(u.candidates, chosen_idx[idx])
        if step is not None:
            gain, target_j = step
            heapq.heappush(heap, (-gain, -u.sensitivity, u.unit_id, idx, chosen_idx[idx], target_j))

    while heap:
        neg_gain, _, _, idx, from_j, target_j = heapq.heappop(heap)
        if chosen_idx[idx] != from_j:
            continue
        u = units[idx]
        cur_c = u.candidates[from_j]
        nxt_c = u.candidates[target_j]
        extra = nxt_c.alg_bytes - cur_c.alg_bytes
        if cur_bytes + extra <= budget_bytes:
            chosen_idx[idx] = target_j
            cur_bytes += extra
            nxt_step = _best_hull_step(u.candidates, target_j)
            if nxt_step is not None:
                gain, next_j = nxt_step
                heapq.heappush(heap, (-gain, -u.sensitivity, u.unit_id, idx, target_j, next_j))
        else:
            # Target jump (e.g., q4 -> bf16) did not fit; try any smaller intermediate step.
            best_small_gain = 0.0
            best_small_j = -1
            for j in range(from_j + 1, target_j):
                cand = u.candidates[j]
                db = cand.alg_bytes - cur_c.alg_bytes
                if db <= 0 or cur_bytes + db > budget_bytes:
                    continue
                dd = cur_c.weighted_distortion - cand.weighted_distortion
                g = dd / db if dd > 0.0 else 0.0
                if g > best_small_gain:
                    best_small_gain = g
                    best_small_j = j
            if best_small_j >= 0:
                heapq.heappush(
                    heap,
                    (-best_small_gain, -u.sensitivity, u.unit_id, idx, from_j, best_small_j),
                )

    plans: list[QuantUnitPlan] = []
    dtype_counts: dict[str, int] = {}
    dtype_weights: dict[str, int] = {}
    for idx, u in enumerate(units):
        c = u.candidates[chosen_idx[idx]]
        plans.append(
            QuantUnitPlan(
                unit_id=u.unit_id,
                tensor_names=u.tensor_names,
                role=u.role,
                layer=u.layer,
                expert=u.expert,
                numel=u.numel,
                sensitivity=u.sensitivity,
                chosen_dtype=c.dtype,
                alg_bytes=c.alg_bytes,
                weighted_distortion=c.weighted_distortion,
                raw_mse=c.raw_mse,
            )
        )
        key = c.dtype.name.lower()
        dtype_counts[key] = dtype_counts.get(key, 0) + len(u.tensor_names)
        dtype_weights[key] = dtype_weights.get(key, 0) + u.numel

    eff_bits = (8.0 * cur_bytes) / total_weights
    weight_share = {k: v / total_weights for k, v in sorted(dtype_weights.items())}
    return GsqAllocationReport(
        target_bits=target_bits,
        effective_bits=eff_bits,
        total_weights=total_weights,
        total_alg_bytes=cur_bytes,
        pinned_weights=pinned_weights,
        pinned_alg_bytes=pinned_alg_bytes,
        units=tuple(plans),
        dtype_counts=dict(sorted(dtype_counts.items())),
        dtype_weight_share=weight_share,
    )


# ---------------------------------------------------------------------------
# Unified Streaming Source Abstraction (`.amf` AND `.safetensors` checkpoints)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SourceTensor:
    name: str
    kind: TensorKind
    dtype: DType
    quant_bits: int
    group_size: int
    layer: int
    expert: int
    shape: tuple[int, ...]
    layout: TensorLayout
    quant_param_kind: QuantParamKind
    amf_meta: PackedTensorMeta | None = None
    sf_entry: _PlannedEntry | None = None


class _UnifiedSource:
    """Bounded-RAM streaming row reader over either an `.amf` v0 file or safetensors."""

    def __init__(
        self,
        input_path: pathlib.Path,
        *,
        group_size: int = 64,
        dtype_mode: str = "auto",
        cold_experts: set[tuple[int, int]],
        all_experts_cold: bool = False,
        tie_word_embeddings: bool | None = None,
        exclude_tensors: Iterable[str] = (),
        model_id_override: str | None = None,
    ) -> None:
        self.input_path = pathlib.Path(input_path)
        self.group_size = group_size
        self.dtype_mode = dtype_mode
        self._amf_reader: AmfReader | None = None
        self._sf_readers: _ShardReaders | None = None
        self.tensors: list[_SourceTensor] = []

        if self.input_path.is_file() and self.input_path.suffix == ".amf":
            reader = AmfReader(self.input_path)
            self._amf_reader = reader
            self.model_id = model_id_override or reader.header.model_id
            excl = set(exclude_tensors)
            for m in reader.entries:
                if m.name in excl:
                    continue
                kind = m.kind
                if m.kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT) or m.expert != NO_EXPERT:
                    if all_experts_cold or (m.layer, m.expert) in cold_experts:
                        kind = TensorKind.COLD_EXPERT
                    elif cold_experts:
                        kind = TensorKind.HOT_EXPERT
                self.tensors.append(
                    _SourceTensor(
                        name=m.name,
                        kind=kind,
                        dtype=m.dtype,
                        quant_bits=m.quant_bits,
                        group_size=m.group_size,
                        layer=m.layer,
                        expert=m.expert,
                        shape=m.shape,
                        layout=m.layout,
                        quant_param_kind=m.quant_param_kind,
                        amf_meta=m,
                    )
                )
        else:
            resolved = resolve_checkpoint(self.input_path)
            tie = _load_tie_flag(resolved.config_path) if tie_word_embeddings is None else tie_word_embeddings
            raw_tensors = _collect_tensors(resolved.shard_paths)
            planned = plan_amf_entries(
                raw_tensors,
                group_size=group_size,
                dtype_mode=dtype_mode,
                cold_experts=cold_experts,
                all_experts_cold=all_experts_cold,
                tie_word_embeddings=tie,
                exclude_tensors=exclude_tensors,
                sme2_pack="none",
            )
            self._sf_readers = _ShardReaders()
            self.model_id = model_id_override or self.input_path.stem
            for e in planned:
                self.tensors.append(
                    _SourceTensor(
                        name=e.name,
                        kind=e.kind,
                        dtype=e.dtype,
                        quant_bits=e.quant_bits,
                        group_size=e.group_size,
                        layer=e.layer,
                        expert=e.expert,
                        shape=e.shape,
                        layout=e.layout,
                        quant_param_kind=e.quant_param_kind,
                        sf_entry=e,
                    )
                )

    def close(self) -> None:
        if self._amf_reader is not None:
            self._amf_reader.close()
            self._amf_reader = None
        if self._sf_readers is not None:
            self._sf_readers.close()
            self._sf_readers = None

    def read_1d_f32(self, t: _SourceTensor) -> list[float]:
        if self._amf_reader is not None:
            m = t.amf_meta
            assert m is not None
            raw = _read_exact(self._amf_reader._fh, m.blob_offset, m.data_bytes)
            if m.dtype == DType.F16:
                return f16_bytes_to_f32_list(raw)
            if m.dtype == DType.BF16:
                return bf16_bytes_to_f32_list(raw)
            if m.dtype == DType.F32:
                return f32_bytes_to_f32_list(raw)
            raise ValueError(f"unsupported 1D dtype {m.dtype.name} for {m.name}")
        assert self._sf_readers is not None and t.sf_entry is not None
        e = t.sf_entry
        fh = self._sf_readers.get(e.data.path)
        raw = _read_exact(fh, e.data.offset, e.data.length)
        if e.data.dtype == "F16":
            return f16_bytes_to_f32_list(raw)
        if e.data.dtype == "BF16":
            return bf16_bytes_to_f32_list(raw)
        if e.data.dtype == "F32":
            return f32_bytes_to_f32_list(raw)
        raise ValueError(f"unsupported 1D safetensors dtype {e.data.dtype} for {e.name}")

    def read_rows_f32(self, t: _SourceTensor, row_indices: Sequence[int]) -> list[list[float]]:
        if self._amf_reader is not None:
            assert t.amf_meta is not None
            return self._amf_reader.read_rows_f32(t.amf_meta, row_indices)
        assert self._sf_readers is not None and t.sf_entry is not None
        e = t.sf_entry
        n_rows, k = e.shape
        fh = self._sf_readers.get(e.data.path)
        out: list[list[float]] = []

        if e.dtype in (DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE):
            assert e.scales is not None and e.biases is not None
            gs = e.group_size
            groups = k // gs
            if e.dtype == DType.Q2_AFFINE:
                row_d = k // 4
                deq = dequantize_q2_row_f32
            elif e.dtype == DType.Q3_AFFINE:
                row_d = (k * 3) // 8
                deq = dequantize_q3_row_f32
            else:
                row_d = k // 2
                deq = dequantize_q4_row_f32
            sb_elem = 2 if e.scales.dtype in ("F16", "BF16") else 4
            row_sb = groups * sb_elem
            s_fh = self._sf_readers.get(e.scales.path)
            b_fh = self._sf_readers.get(e.biases.path)
            for r in row_indices:
                if r < 0 or r >= n_rows:
                    raise IndexError(f"row {r} out of bounds for {e.name}")
                d_raw = _read_exact(fh, e.data.offset + r * row_d, row_d)
                s_raw = _read_exact(s_fh, e.scales.offset + r * row_sb, row_sb)
                b_raw = _read_exact(b_fh, e.biases.offset + r * row_sb, row_sb)
                s_vals = _decode_sf_floats(s_raw, e.scales.dtype)
                b_vals = _decode_sf_floats(b_raw, e.biases.dtype)
                p_buf = bytearray(groups * 4)
                for g_idx in range(groups):
                    _F16_PAIR_STRUCT.pack_into(p_buf, g_idx * 4, _clamp_f16(s_vals[g_idx]), _clamp_f16(b_vals[g_idx]))
                out.append(deq(d_raw, p_buf, k, group_size=gs))
            return out

        elem_b = 2 if e.data.dtype in ("F16", "BF16") else 4
        row_b = k * elem_b
        for r in row_indices:
            if r < 0 or r >= n_rows:
                raise IndexError(f"row {r} out of bounds for {e.name}")
            raw = _read_exact(fh, e.data.offset + r * row_b, row_b)
            out.append(_decode_sf_floats(raw, e.data.dtype))
        return out

    def iter_row_batches_f32(self, t: _SourceTensor, batch_rows: int = _ROW_BATCH_SIZE) -> Iterator[list[list[float]]]:
        """Stream a 2D tensor sequentially in contiguous chunks of `batch_rows` rows."""
        n_rows, k = t.shape
        if self._amf_reader is not None:
            m = t.amf_meta
            assert m is not None
            if m.layout == TensorLayout.SME2_PANEL64:
                step = max(SME2_PANEL_COLS, (batch_rows // SME2_PANEL_COLS) * SME2_PANEL_COLS)
                for r0 in range(0, n_rows, step):
                    r1 = min(n_rows, r0 + step)
                    rows_out: list[list[float]] = []
                    for p_start in range(r0, r1, SME2_PANEL_COLS):
                        p_idx = p_start // SME2_PANEL_COLS
                        valid = min(SME2_PANEL_COLS, n_rows - p_start)
                        rows_out.extend(self._amf_reader._read_sme2_panel_f32(m, p_idx, valid))
                    yield rows_out
                return

            fh = self._amf_reader._fh
            if m.dtype in (DType.F16, DType.BF16, DType.F32):
                elem_b = 4 if m.dtype == DType.F32 else 2
                row_b = k * elem_b
                conv = (
                    f16_bytes_to_f32_list
                    if m.dtype == DType.F16
                    else (bf16_bytes_to_f32_list if m.dtype == DType.BF16 else f32_bytes_to_f32_list)
                )
                for r0 in range(0, n_rows, batch_rows):
                    cnt = min(batch_rows, n_rows - r0)
                    raw = _read_exact(fh, m.blob_offset + r0 * row_b, cnt * row_b)
                    mv = memoryview(raw)
                    yield [conv(mv[i * row_b : (i + 1) * row_b]) for i in range(cnt)]
                return

            if m.dtype in (DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE):
                gs = m.group_size
                groups = k // gs
                if m.dtype == DType.Q2_AFFINE:
                    row_d = k // 4
                    deq = dequantize_q2_row_f32
                elif m.dtype == DType.Q3_AFFINE:
                    row_d = (k * 3) // 8
                    deq = dequantize_q3_row_f32
                else:
                    row_d = k // 2
                    deq = dequantize_q4_row_f32
                row_p = groups * 4
                for r0 in range(0, n_rows, batch_rows):
                    cnt = min(batch_rows, n_rows - r0)
                    d_raw = _read_exact(fh, m.blob_offset + r0 * row_d, cnt * row_d)
                    p_raw = _read_exact(fh, m.blob_offset + m.params_offset + r0 * row_p, cnt * row_p)
                    d_mv = memoryview(d_raw)
                    p_mv = memoryview(p_raw)
                    yield [
                        deq(d_mv[i * row_d : (i + 1) * row_d], p_mv[i * row_p : (i + 1) * row_p], k, group_size=gs)
                        for i in range(cnt)
                    ]
                return
            raise ValueError(f"unsupported AMF dtype {m.dtype.name} for {m.name}")

        assert self._sf_readers is not None and t.sf_entry is not None
        e = t.sf_entry
        fh = self._sf_readers.get(e.data.path)
        if e.dtype in (DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE):
            for r0 in range(0, n_rows, batch_rows):
                cnt = min(batch_rows, n_rows - r0)
                yield self.read_rows_f32(t, range(r0, r0 + cnt))
            return

        elem_b = 2 if e.data.dtype in ("F16", "BF16") else 4
        row_b = k * elem_b
        for r0 in range(0, n_rows, batch_rows):
            cnt = min(batch_rows, n_rows - r0)
            raw = _read_exact(fh, e.data.offset + r0 * row_b, cnt * row_b)
            mv = memoryview(raw)
            yield [_decode_sf_floats(mv[i * row_b : (i + 1) * row_b], e.data.dtype) for i in range(cnt)]

    def stream_raw_passthrough(self, t: _SourceTensor, w: _BlobWriter, target_dtype: DType, target_layout: TensorLayout, blob_bytes: int) -> None:
        """Stream-copy an unchanged tensor (possibly converting float dtype or SME2-packing)."""
        if self._amf_reader is not None:
            m = t.amf_meta
            assert m is not None
            src_fh = self._amf_reader._fh
            if m.dtype == target_dtype and m.layout == target_layout:
                done = 0
                while done < m.blob_bytes:
                    take = min(8 * 1024 * 1024, m.blob_bytes - done)
                    w.write(_read_exact(src_fh, m.blob_offset + done, take))
                    done += take
                return
            if m.dtype == target_dtype and m.layout == TensorLayout.ROW_MAJOR and target_layout == TensorLayout.SME2_PANEL64:
                from aminfer_tools.amf import _stream_repack_amf_entry

                _stream_repack_amf_entry(src_fh, m, blob_bytes, w)
                return
            # Float conversion (e.g. BF16 <-> F16) or SME2 -> ROW_MAJOR fallback via streaming rows.
            _stream_float_from_rows(self, t, target_dtype, target_layout, blob_bytes, w)
            return

        assert self._sf_readers is not None and t.sf_entry is not None
        e = t.sf_entry
        if e.dtype == target_dtype and e.layout == target_layout:
            _stream_entry(e, self._sf_readers, w)
            return
        if e.dtype == target_dtype and target_layout == TensorLayout.SME2_PANEL64:
            re_e = _PlannedEntry(
                name=e.name,
                kind=e.kind,
                dtype=e.dtype,
                quant_bits=e.quant_bits,
                group_size=e.group_size,
                layer=e.layer,
                expert=e.expert,
                shape=e.shape,
                data=e.data,
                data_convert=e.data_convert,
                scales=e.scales,
                biases=e.biases,
                data_bytes=sme2_panel64_data_bytes(e.dtype, e.shape[0], e.shape[1], e.group_size),
                params_offset=0,
                params_bytes=0,
                blob_bytes=blob_bytes,
                layout=TensorLayout.SME2_PANEL64,
                quant_param_kind=QuantParamKind.AFFINE_SCALE_BIAS,
            )
            _stream_entry(re_e, self._sf_readers, w)
            return
        _stream_float_from_rows(self, t, target_dtype, target_layout, blob_bytes, w)


def _decode_sf_floats(raw: bytes | bytearray | memoryview, dtype_str: str) -> list[float]:
    if dtype_str == "F16":
        return f16_bytes_to_f32_list(raw)
    if dtype_str == "BF16":
        return bf16_bytes_to_f32_list(raw)
    if dtype_str == "F32":
        return f32_bytes_to_f32_list(raw)
    raise ValueError(f"unsupported float dtype {dtype_str}")


def _stream_float_from_rows(
    src: _UnifiedSource,
    t: _SourceTensor,
    target_dtype: DType,
    target_layout: TensorLayout,
    blob_bytes: int,
    w: _BlobWriter,
) -> None:
    if len(t.shape) == 1:
        vals = src.read_1d_f32(t)
        if target_dtype == DType.F16:
            w.write(struct.pack(f"<{len(vals)}e", *(_clamp_f16(v) for v in vals)))
        elif target_dtype == DType.BF16:
            f32_raw = struct.pack(f"<{len(vals)}f", *vals)
            out = bytearray(len(vals) * 2)
            out[0::2] = f32_raw[2::4]
            out[1::2] = f32_raw[3::4]
            w.write(out)
        elif target_dtype == DType.F32:
            w.write(struct.pack(f"<{len(vals)}f", *vals))
        else:
            raise ValueError(f"unsupported 1D target dtype {target_dtype.name}")
        w.zeros(blob_bytes - w.written)
        return

    n_rows, k = t.shape
    row_packer_f16 = struct.Struct(f"<{k}e")
    row_packer_f32 = struct.Struct(f"<{k}f")

    def _encode_row(vals: list[float]) -> bytes:
        if target_dtype == DType.F16:
            return row_packer_f16.pack(*(_clamp_f16(v) for v in vals))
        if target_dtype == DType.BF16:
            f16_raw = row_packer_f16.pack(*(_clamp_f16(v) for v in vals))
            return f16_bytes_to_bf16_bytes(f16_raw)
        if target_dtype == DType.F32:
            return row_packer_f32.pack(*vals)
        raise ValueError(f"unsupported float target dtype {target_dtype.name}")

    for batch in src.iter_row_batches_f32(t, _ROW_BATCH_SIZE):
        encoded_rows = [_encode_row(r) for r in batch]
        if target_layout == TensorLayout.SME2_PANEL64:
            for p0 in range(0, len(encoded_rows), SME2_PANEL_COLS):
                w.write(pack_sme2_panel64_f16_panel(encoded_rows[p0 : p0 + SME2_PANEL_COLS], k))
        else:
            for rb in encoded_rows:
                w.write(rb)
    w.zeros(blob_bytes - w.written)


def _load_heat_routing_probs(heat_profile_path: pathlib.Path) -> dict[tuple[int, int], float]:
    """Extract normalized per-layer expert routing weights `mean == 1.0` from W8 `heat.json`."""
    raw = json.loads(pathlib.Path(heat_profile_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"unsupported heat profile {heat_profile_path} (expected schema_version == 1)")
    layers = raw.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"heat profile {heat_profile_path} missing 'layers'")

    weights: dict[tuple[int, int], float] = {}
    for item in layers:
        l_idx = int(item["layer"])
        probs = item.get("routing_prob")
        if isinstance(probs, list) and probs:
            vals = [max(0.0, float(x)) for x in probs]
            mean_v = sum(vals) / len(vals)
            denom = mean_v if mean_v > 0.0 else 1.0
            for e_idx, v in enumerate(vals):
                weights[(l_idx, e_idx)] = v / denom
            continue
        counts = item.get("counts")
        if isinstance(counts, list) and counts:
            vals = [max(0.0, float(x)) for x in counts]
            mean_v = sum(vals) / len(vals)
            denom = mean_v if mean_v > 0.0 else 1.0
            for e_idx, v in enumerate(vals):
                weights[(l_idx, e_idx)] = v / denom
            continue
        ranked = item.get("ranked_hot_to_cold")
        if isinstance(ranked, list) and ranked:
            n_e = len(ranked)
            for rank_pos, e_id in enumerate(ranked):
                # Smooth Zipf-like rank decay from ~2.0 (hottest) down to ~0.2 (coldest).
                score = 2.0 - 1.8 * (rank_pos / max(1, n_e - 1))
                weights[(l_idx, int(e_id))] = score
    return weights


def _sample_row_indices(n_rows: int, max_samples: int) -> list[int]:
    if n_rows <= max_samples:
        return list(range(n_rows))
    step = n_rows / max_samples
    return [min(n_rows - 1, int(i * step)) for i in range(max_samples)]


def _analyze_tensor_sample(
    src: _UnifiedSource,
    t: _SourceTensor,
    *,
    group_size: int,
    sample_rows: int,
    use_rco: bool,
    rco_iters: int,
    max_cand_bits: int,
) -> tuple[float, float, dict[DType, float]]:
    """Stream-sample a 2D tensor to estimate `(var, outlier_factor, mse_by_quant_dtype)`."""
    n_rows, k = t.shape
    row_ids = _sample_row_indices(n_rows, sample_rows)
    rows = src.read_rows_f32(t, row_ids)

    n_vals = 0
    sum_w = 0.0
    sum_w2 = 0.0
    max_abs = 0.0
    for row in rows:
        n_vals += len(row)
        sum_w += math.fsum(row)
        sum_w2 += math.sumprod(row, row)
        r_max = max(abs(min(row)), abs(max(row)))
        if r_max > max_abs:
            max_abs = r_max

    mean = sum_w / max(1, n_vals)
    var = max(1e-12, sum_w2 / max(1, n_vals) - mean * mean)
    rms = math.sqrt(sum_w2 / max(1, n_vals))
    outlier_ratio = max_abs / max(1e-8, rms)

    # Sample up to 64 groups across sampled rows for fast, accurate empirical distortion curves.
    groups_per_row = k // group_size
    sampled_groups: list[list[float]] = []
    max_groups = 64
    total_avail = len(rows) * groups_per_row
    stride = max(1, total_avail // max_groups)
    g_counter = 0
    for row in rows:
        for g in range(groups_per_row):
            if g_counter % stride == 0 and len(sampled_groups) < max_groups:
                sampled_groups.append(row[g * group_size : (g + 1) * group_size])
            g_counter += 1

    mse_by_dtype: dict[DType, float] = {}
    for dt, bits in ((DType.Q2_AFFINE, 2), (DType.Q3_AFFINE, 3), (DType.Q4_AFFINE, 4)):
        if bits > max_cand_bits:
            continue
        if t.dtype == dt:
            mse_by_dtype[dt] = 0.0
            continue
        m_sum = 0.0
        for grp in sampled_groups:
            if use_rco:
                _, _, _, m_val = quantize_group_rco(grp, bits, iters=rco_iters)
            else:
                _, _, _, m_val = quantize_group_minmax(grp, bits)
            m_sum += m_val
        mse_by_dtype[dt] = m_sum / max(1, len(sampled_groups))

    # Outlier multiplier boosts heavy-tailed tensors (activation-sensitive channels).
    outlier_factor = 1.0 + 0.12 * max(0.0, outlier_ratio - 3.0)
    return var, outlier_factor, mse_by_dtype


def _output_name_for_tensor(src_name: str, target_dtype: DType, existing_names: set[str]) -> str:
    """Normalize `.weight` suffix when newly quantizing a float tensor (matching MLX AMF naming)."""
    if target_dtype in _QUANT_DTYPES and src_name.endswith(".weight"):
        cand = src_name[:-7]
        if cand and cand not in existing_names:
            return cand
    return src_name


def _stream_quantize_tensor(
    src: _UnifiedSource,
    t: _SourceTensor,
    *,
    target_dtype: DType,
    target_layout: TensorLayout,
    group_size: int,
    blob_bytes: int,
    use_rco: bool,
    rco_iters: int,
    w: _BlobWriter,
    pool: concurrent.futures.ProcessPoolExecutor | None,
) -> float:
    """Stream-quantize a 2D tensor into `row_major` or `sme2_panel64` with bounded RAM."""
    n_rows, k = t.shape
    bits = _DTYPE_BITS_RANK[target_dtype]
    groups = k // group_size
    row_data_bytes = (k * bits) // 8
    row_param_bytes = groups * 4

    def _iter_quantized_batches() -> Iterator[tuple[bytes, bytes, float]]:
        if pool is None or n_rows < _ROW_BATCH_SIZE * 2:
            for batch in src.iter_row_batches_f32(t, _ROW_BATCH_SIZE):
                qb, pb, mse, _ = quantize_rows_affine(
                    batch, bits, group_size=group_size, use_rco=use_rco, rco_iters=rco_iters
                )
                yield qb, pb, mse
            return

        # Bounded in-flight futures to keep host RSS constant even on multi-GB tensors.
        max_inflight = max(2, (os.cpu_count() or 4))
        pending: list[concurrent.futures.Future[tuple[bytes, bytes, float]]] = []
        for batch in src.iter_row_batches_f32(t, _ROW_BATCH_SIZE):
            pending.append(pool.submit(_quantize_worker_task, (batch, bits, group_size, use_rco, rco_iters)))
            if len(pending) >= max_inflight:
                yield pending.pop(0).result()
        for fut in pending:
            yield fut.result()

    if target_layout == TensorLayout.SME2_PANEL64:
        if target_dtype == DType.Q4_AFFINE:
            packer = pack_sme2_panel64_q4_panel
        elif target_dtype == DType.Q3_AFFINE:
            packer = pack_sme2_panel64_q3_panel
        else:
            packer = pack_sme2_panel64_q2_from_mlx_panel
        mse_acc = 0.0
        n_batches = 0
        for qb, pb, mse in _iter_quantized_batches():
            mse_acc += mse
            n_batches += 1
            b_rows = len(qb) // row_data_bytes
            d_mv = memoryview(qb)
            p_mv = memoryview(pb)
            for p0 in range(0, b_rows, SME2_PANEL_COLS):
                cnt = min(SME2_PANEL_COLS, b_rows - p0)
                d_rows = [d_mv[(p0 + c) * row_data_bytes : (p0 + c + 1) * row_data_bytes] for c in range(cnt)]
                p_rows = [p_mv[(p0 + c) * row_param_bytes : (p0 + c + 1) * row_param_bytes] for c in range(cnt)]
                w.write(packer(d_rows, p_rows, k))
        w.zeros(blob_bytes - w.written)
        return mse_acc / max(1, n_batches)

    # ROW_MAJOR layout: `qweight` (`data_bytes`) -> 64 B pad -> `scale_bias` (`params_bytes`) -> 16 KiB pad.
    # Buffer only `params_bytes` (~4 B per 64 weights = 6.25% of float size, e.g. <= 10 MiB for 151936x1024).
    data_bytes, params_offset, params_bytes = _effective_payload_bytes(target_dtype, (n_rows, k), group_size)
    params_chunks: list[bytes] = []
    mse_acc = 0.0
    n_batches = 0
    for qb, pb, mse in _iter_quantized_batches():
        w.write(qb)
        params_chunks.append(pb)
        mse_acc += mse
        n_batches += 1
    if w.written != data_bytes:
        raise AssertionError(f"wrote {w.written} qweight bytes for {t.name}, expected {data_bytes}")
    w.zeros(params_offset - data_bytes)
    for pb in params_chunks:
        w.write(pb)
    if w.written != params_offset + params_bytes:
        raise AssertionError(f"wrote {w.written} payload bytes for {t.name}, expected {params_offset + params_bytes}")
    w.zeros(blob_bytes - w.written)
    return mse_acc / max(1, n_batches)


@dataclass(frozen=True)
class GsqConvertReport:
    input_path: pathlib.Path
    output_path: pathlib.Path
    model_id: str
    target_bits: float
    effective_bits: float
    use_rco: bool
    allocation: GsqAllocationReport
    metas: list[PackedTensorMeta]
    mean_quant_mse: float

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "input": str(self.input_path),
            "output": str(self.output_path),
            "model_id": self.model_id,
            "quant_profile": "gsq-rco" if self.use_rco else "gsq-minmax",
            "target_bits": self.target_bits,
            "effective_bits": round(self.effective_bits, 4),
            "total_weights": self.allocation.total_weights,
            "total_alg_bytes": self.allocation.total_alg_bytes,
            "mean_quant_mse": self.mean_quant_mse,
            "dtype_counts": self.allocation.dtype_counts,
            "dtype_weight_share": {k: round(v, 5) for k, v in self.allocation.dtype_weight_share.items()},
            "units": [
                {
                    "unit_id": u.unit_id,
                    "role": u.role,
                    "layer": None if u.layer == NO_LAYER else u.layer,
                    "expert": None if u.expert == NO_EXPERT else u.expert,
                    "dtype": u.chosen_dtype.name.lower(),
                    "numel": u.numel,
                    "alg_bytes": u.alg_bytes,
                    "sensitivity": round(u.sensitivity, 6),
                    "sample_mse": u.raw_mse,
                    "tensors": list(u.tensor_names),
                }
                for u in self.allocation.units
            ],
        }


def gsq_convert(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    target_bits: float,
    model_id: str | None = None,
    group_size: int = 64,
    dtype_mode: str = "auto",
    heat_profile: pathlib.Path | None = None,
    cold_experts: Iterable[tuple[int, int]] = (),
    cold_experts_per_layer: int | None = None,
    all_experts_cold: bool = False,
    tie_word_embeddings: bool | None = None,
    exclude_tensors: Iterable[str] = (),
    sme2_pack: str = "none",
    use_rco: bool = True,
    rco_iters: int = 3,
    sample_rows: int = 32,
    manifest_path: pathlib.Path | None = None,
    workers: int = 1,
) -> GsqConvertReport:
    """Convert a float/quantized safetensors checkpoint or `.amf` into a GSQ+RCO `.amf` v0.

    Streams weights row-batch by row-batch with bounded host RAM, enforces strict
    per-tensor and per-MoE-expert uniform `QuantScheme` (ADR-0017), writes 16 KiB
    page-aligned blobs with IEEE CRC-32 atomically, and optionally emits a JSON
    allocation manifest (`manifest_path`).
    """
    if group_size != 64:
        raise ValueError(f"ADR-0017 / ADR-0014 requires group_size == 64, got {group_size}")
    if dtype_mode not in DTYPE_MODES:
        raise ValueError(f"dtype_mode must be one of {DTYPE_MODES}, got {dtype_mode!r}")
    if sme2_pack not in SME2_PACK_MODES:
        raise ValueError(f"sme2_pack must be one of {SME2_PACK_MODES}, got {sme2_pack!r}")
    if target_bits <= 0.0:
        raise ValueError(f"target_bits must be positive, got {target_bits}")

    effective_cold: set[tuple[int, int]] = set(cold_experts)
    heat_probs: dict[tuple[int, int], float] = {}
    if heat_profile is not None:
        hp_path = pathlib.Path(heat_profile)
        heat_probs = _load_heat_routing_probs(hp_path)
        effective_cold.update(
            load_cold_experts_from_heat_profile(
                hp_path,
                cold_experts_per_layer=cold_experts_per_layer,
            )
        )
    elif cold_experts_per_layer is not None:
        raise ValueError("cold_experts_per_layer requires heat_profile in gsq_convert")

    src = _UnifiedSource(
        input_path,
        group_size=group_size,
        dtype_mode=dtype_mode,
        cold_experts=effective_cold,
        all_experts_cold=all_experts_cold,
        tie_word_embeddings=tie_word_embeddings,
        exclude_tensors=exclude_tensors,
        model_id_override=model_id,
    )
    try:
        model_id_str = src.model_id
        model_id_bytes = model_id_str.encode("utf-8")
        if len(model_id_bytes) == 0 or len(model_id_bytes) > 63:
            raise ValueError(f"model_id must be 1..63 UTF-8 bytes: {model_id_str!r}")

        # 1. Extract layer-norm Hessian proxies (mean(ln_w^2)) for input & post-attention norms.
        ln_in_energy: dict[int, float] = {}
        ln_post_energy: dict[int, float] = {}
        max_layer = 0
        for t in src.tensors:
            if t.layer != NO_LAYER and t.layer > max_layer:
                max_layer = t.layer
            if t.kind == TensorKind.NORM and len(t.shape) == 1 and t.layer != NO_LAYER:
                vals = src.read_1d_f32(t)
                energy = math.sumprod(vals, vals) / max(1, len(vals))
                if "input_layernorm" in t.name:
                    ln_in_energy[t.layer] = energy
                elif "post_attention_layernorm" in t.name:
                    ln_post_energy[t.layer] = energy

        num_layers = max_layer + 1
        mean_ln_in = (sum(ln_in_energy.values()) / len(ln_in_energy)) if ln_in_energy else 1.0
        mean_ln_post = (sum(ln_post_energy.values()) / len(ln_post_energy)) if ln_post_energy else 1.0

        # 2. Partition tensors into pinned (unquantizable) vs atomic QuantUnits.
        #    MoE experts `(layer, expert)` are grouped so gate/up/down share ONE uniform DType!
        pinned_tensors: list[_SourceTensor] = []
        expert_groups: dict[tuple[int, int], list[_SourceTensor]] = {}
        standalone_quant: list[_SourceTensor] = []

        for t in src.tensors:
            is_quantizable = (
                len(t.shape) == 2
                and t.kind != TensorKind.NORM
                and t.dtype in (DType.F16, DType.BF16, DType.F32, DType.Q4_AFFINE, DType.Q3_AFFINE, DType.Q2_AFFINE)
                and t.shape[1] > 0
                and t.shape[1] % group_size == 0
            )
            if not is_quantizable:
                pinned_tensors.append(t)
            elif t.kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT) and t.expert != NO_EXPERT:
                expert_groups.setdefault((t.layer, t.expert), []).append(t)
            else:
                standalone_quant.append(t)

        # Build QuantUnitInput list with sensitivity & empirical distortion curves.
        unit_inputs: list[QuantUnitInput] = []
        tensor_to_unit: dict[str, str] = {}

        def _float_target_for(t: _SourceTensor) -> DType:
            if dtype_mode == "f16":
                return DType.F16
            if dtype_mode == "bf16":
                return DType.BF16
            if t.dtype in (DType.F16, DType.BF16):
                return t.dtype
            return DType.F16

        def _layer_u_curve(layer: int) -> float:
            if layer == NO_LAYER or num_layers <= 1:
                return 1.25
            frac = layer / (num_layers - 1)
            return 1.0 + 0.5 * (1.0 - 4.0 * frac * (1.0 - frac))

        def _hessian_proxy(role: str, layer: int) -> float:
            if layer == NO_LAYER:
                return 1.0
            if role in ("attn_qk", "attn_vo") and layer in ln_in_energy and mean_ln_in > 0.0:
                return max(0.25, min(4.0, ln_in_energy[layer] / mean_ln_in))
            if role in ("dense_ffn", "moe_expert", "router") and layer in ln_post_energy and mean_ln_post > 0.0:
                return max(0.25, min(4.0, ln_post_energy[layer] / mean_ln_post))
            return 1.0

        def _build_unit(unit_id: str, members: list[_SourceTensor], role: str, layer: int, expert: int) -> QuantUnitInput:
            # Cap candidate precision at the lowest source precision among members so we never
            # fabricate bits by upscaling an already-quantized input tensor.
            max_bits = min(_DTYPE_BITS_RANK[m.dtype] for m in members)
            f_target = _float_target_for(members[0])

            total_numel = 0
            var_weighted = 0.0
            out_weighted = 0.0
            mse_weighted: dict[DType, float] = {DType.Q2_AFFINE: 0.0, DType.Q3_AFFINE: 0.0, DType.Q4_AFFINE: 0.0}

            for m in members:
                numel_m = m.shape[0] * m.shape[1]
                total_numel += numel_m
                var_m, out_m, mse_m = _analyze_tensor_sample(
                    src,
                    m,
                    group_size=group_size,
                    sample_rows=sample_rows,
                    use_rco=use_rco,
                    rco_iters=rco_iters,
                    max_cand_bits=max_bits,
                )
                var_weighted += var_m * numel_m
                out_weighted += out_m * numel_m
                for qdt, qmse in mse_m.items():
                    mse_weighted[qdt] += qmse * numel_m

            mean_var = max(1e-12, var_weighted / max(1, total_numel))
            mean_outlier = out_weighted / max(1, total_numel)

            heat_mult = 1.0
            if role == "moe_expert" and (layer, expert) in heat_probs:
                heat_mult = 0.25 + 2.5 * heat_probs[(layer, expert)]

            sensitivity = (
                _ROLE_WEIGHT.get(role, 1.0)
                * _layer_u_curve(layer)
                * _hessian_proxy(role, layer)
                * mean_outlier
                * heat_mult
            )

            cand_dtypes: list[DType] = []
            for qdt, qb in ((DType.Q2_AFFINE, 2), (DType.Q3_AFFINE, 3), (DType.Q4_AFFINE, 4)):
                if qb <= max_bits:
                    cand_dtypes.append(qdt)
            if max_bits >= 16:
                cand_dtypes.append(f_target)

            candidates: list[QuantUnitCandidate] = []
            for cdt in cand_dtypes:
                alg_b = sum(_algorithmic_bytes(cdt, m.shape, group_size) for m in members)
                if cdt in _QUANT_DTYPES:
                    raw_mse = mse_weighted[cdt] / max(1, total_numel)
                else:
                    raw_mse = 0.0
                # Relative squared error weighted by unit sensitivity and parameter count.
                w_dist = sensitivity * total_numel * (raw_mse / mean_var)
                candidates.append(
                    QuantUnitCandidate(
                        dtype=cdt,
                        alg_bytes=alg_b,
                        weighted_distortion=w_dist,
                        raw_mse=raw_mse,
                    )
                )

            candidates.sort(key=lambda c: c.alg_bytes)
            for m in members:
                tensor_to_unit[m.name] = unit_id
            return QuantUnitInput(
                unit_id=unit_id,
                tensor_names=tuple(m.name for m in members),
                role=role,
                layer=layer,
                expert=expert,
                numel=total_numel,
                sensitivity=sensitivity,
                candidates=tuple(candidates),
            )

        for t in standalone_quant:
            role = _role_of_tensor(t.name, t.kind)
            unit_inputs.append(_build_unit(t.name, [t], role, t.layer, t.expert))
        for (l_idx, e_idx), members in sorted(expert_groups.items()):
            uid = f"layers.{l_idx}.mlp.experts.{e_idx}"
            unit_inputs.append(_build_unit(uid, members, "moe_expert", l_idx, e_idx))

        pinned_weights = 0
        pinned_alg_bytes = 0
        for t in pinned_tensors:
            numel_t = 1
            for d in t.shape:
                numel_t *= d
            pinned_weights += numel_t
            pinned_alg_bytes += _algorithmic_bytes(t.dtype, t.shape, t.group_size or group_size)

        alloc = solve_gsq_allocation(
            unit_inputs,
            target_bits=target_bits,
            pinned_weights=pinned_weights,
            pinned_alg_bytes=pinned_alg_bytes,
        )
        chosen_by_unit = {u.unit_id: u.chosen_dtype for u in alloc.units}

        # 3. Plan final `.amf` directory layout (16 KiB-aligned blobs) before streaming writes.
        existing_names = {t.name for t in src.tensors}
        out_specs: list[
            tuple[
                _SourceTensor,
                str,
                DType,
                int,
                int,
                TensorLayout,
                QuantParamKind,
                int,
                int,
                int,
                int,
                int,
            ]
        ] = []
        tensor_count = len(src.tensors)
        dir_offset = APPLE_PAGE_SIZE
        dir_raw_len = tensor_count * ENTRY_STRUCT.size
        dir_bytes = 0 if tensor_count == 0 else _align_up(dir_raw_len, APPLE_PAGE_SIZE)
        cursor = dir_offset + dir_bytes
        used_out_names: set[str] = set()

        for t in src.tensors:
            uid = tensor_to_unit.get(t.name)
            target_dtype = chosen_by_unit[uid] if uid is not None else t.dtype
            out_name = _output_name_for_tensor(t.name, target_dtype, existing_names | used_out_names)
            used_out_names.add(out_name)

            q_bits = _DTYPE_BITS_RANK[target_dtype] if target_dtype in _QUANT_DTYPES else 0
            g_size = group_size if target_dtype in _QUANT_DTYPES else 0
            use_sme2 = _should_sme2_pack(t.kind, target_dtype, t.shape, sme2_pack)
            if use_sme2:
                layout = TensorLayout.SME2_PANEL64
                param_kind = QuantParamKind.AFFINE_SCALE_BIAS
                data_bytes = sme2_panel64_data_bytes(target_dtype, t.shape[0], t.shape[1], g_size)
                params_offset = 0
                params_bytes = 0
                content = data_bytes
            else:
                layout = TensorLayout.ROW_MAJOR
                param_kind = QuantParamKind.AFFINE_SCALE_BIAS
                data_bytes, params_offset, params_bytes = _effective_payload_bytes(target_dtype, t.shape, group_size)
                content = (params_offset + params_bytes) if params_bytes > 0 else data_bytes

            blob_bytes = _align_up(max(content, 1), APPLE_PAGE_SIZE)
            out_specs.append(
                (
                    t,
                    out_name,
                    target_dtype,
                    q_bits,
                    g_size,
                    layout,
                    param_kind,
                    cursor,
                    blob_bytes,
                    data_bytes,
                    params_offset,
                    params_bytes,
                )
            )
            cursor += blob_bytes

        # 4. Stream tensors straight into `<output>.partial`, compute CRC-32, fsync, atomic rename.
        out_path = pathlib.Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_name(out_path.name + ".partial")
        metas: list[PackedTensorMeta] = []
        quant_mse_sum = 0.0
        quant_tensor_count = 0

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            with partial.open("wb") as out_fh:
                _set_nocache(out_fh.fileno())
                prefix = _BlobWriter(out_fh)
                prefix.zeros(dir_offset + dir_bytes)

                for (
                    t,
                    out_name,
                    target_dtype,
                    q_bits,
                    g_size,
                    layout,
                    param_kind,
                    blob_offset,
                    blob_bytes,
                    data_bytes,
                    params_offset,
                    params_bytes,
                ) in out_specs:
                    if out_fh.tell() != blob_offset:
                        raise AssertionError(f"planned offset {blob_offset} != pos {out_fh.tell()} for {out_name}")
                    w = _BlobWriter(out_fh)
                    if target_dtype in _QUANT_DTYPES and target_dtype != t.dtype:
                        mse_t = _stream_quantize_tensor(
                            src,
                            t,
                            target_dtype=target_dtype,
                            target_layout=layout,
                            group_size=group_size,
                            blob_bytes=blob_bytes,
                            use_rco=use_rco,
                            rco_iters=rco_iters,
                            w=w,
                            pool=pool,
                        )
                        quant_mse_sum += mse_t
                        quant_tensor_count += 1
                    else:
                        src.stream_raw_passthrough(t, w, target_dtype, layout, blob_bytes)

                    if w.written != blob_bytes:
                        raise AssertionError(f"wrote {w.written} bytes for {out_name}, planned {blob_bytes}")
                    metas.append(
                        PackedTensorMeta(
                            name=out_name,
                            kind=t.kind,
                            dtype=target_dtype,
                            quant_bits=q_bits,
                            group_size=g_size,
                            layer=t.layer,
                            expert=t.expert,
                            shape=t.shape,
                            blob_offset=blob_offset,
                            blob_bytes=blob_bytes,
                            data_bytes=data_bytes,
                            params_offset=params_offset,
                            params_bytes=params_bytes,
                            blob_crc32=w.crc & 0xFFFFFFFF,
                            layout=layout,
                            quant_param_kind=param_kind,
                        )
                    )

                if out_fh.tell() != cursor:
                    raise AssertionError(f"final size {out_fh.tell()} != planned {cursor}")

                dir_raw = _pack_directory(metas)
                header = _pack_header(model_id_bytes, tensor_count, dir_offset, dir_bytes, _crc32(dir_raw))
                out_fh.seek(0)
                out_fh.write(header)
                if dir_raw:
                    out_fh.seek(dir_offset)
                    out_fh.write(dir_raw)
                out_fh.flush()
                os.fsync(out_fh.fileno())
            os.replace(partial, out_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        mean_quant_mse = quant_mse_sum / max(1, quant_tensor_count)
        report = GsqConvertReport(
            input_path=pathlib.Path(input_path),
            output_path=out_path,
            model_id=model_id_str,
            target_bits=target_bits,
            effective_bits=alloc.effective_bits,
            use_rco=use_rco,
            allocation=alloc,
            metas=metas,
            mean_quant_mse=mean_quant_mse,
        )
        if manifest_path is not None:
            m_path = pathlib.Path(manifest_path)
            m_path.parent.mkdir(parents=True, exist_ok=True)
            m_path.write_text(json.dumps(report.to_manifest_dict(), indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        src.close()
