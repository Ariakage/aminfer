"""Unit and end-to-end tests for M5 W14/G13: q2_affine, RCO, GSQ, and Python -> Zig parity."""

from __future__ import annotations

import io
import json
import math
import pathlib
import random
import re
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    AmfReader,
    DType,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    dequantize_q2_row_f32,
    dequantize_q3_row_f32,
    dequantize_q4_row_f32,
    dequantize_q5_row_f32,
    dequantize_q6_row_f32,
    dequantize_q8_row_f32,
    dequantize_sme2_panel64_q2_panel_f32,
    load_amf_metadata,
    pack_amf_from_safetensors,
    pack_mlx_affine_2bit,
    pack_mlx_affine_3bit,
    pack_sme2_panel64_q2_from_mlx_panel,
    repack_amf_sme2,
    sme2_panel64_data_bytes,
    write_amf,
)
from aminfer_tools.cli import _is_moe_checkpoint_input, main
from aminfer_tools.gsq_rco import (
    _BACKBONE_PROTECTED_ROLES,
    QuantUnitCandidate,
    QuantUnitInput,
    _imatrix_col_weights,
    _imatrix_tensor_multipliers,
    gsq_convert,
    parse_gguf_imatrix,
    quantize_group_minmax,
    quantize_group_rco,
    quantize_rows_affine,
    solve_gsq_allocation,
)
from aminfer_tools.synth import PRESETS, generate_qwen3_bundle

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    """Write a minimal valid `.safetensors` file using Python stdlib only."""
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    cursor = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(data)],
        }
        payloads.append(data)
        cursor += len(data)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for p in payloads:
            fh.write(p)


def _rand_bf16_bytes(rng: random.Random, count: int, scale: float, *, outlier_prob: float = 0.0) -> bytes:
    f32_buf = bytearray(count * 4)
    pack_f = struct.Struct("<f").pack_into
    for i in range(count):
        v = rng.gauss(0.0, scale)
        if outlier_prob > 0.0 and rng.random() < outlier_prob:
            v *= 6.0
        pack_f(f32_buf, i * 4, v)
    out = bytearray(count * 2)
    out[0::2] = f32_buf[2::4]
    out[1::2] = f32_buf[3::4]
    return bytes(out)


def _rand_f16_ones_bytes(rng: random.Random, count: int, base: float = 1.0) -> bytes:
    vals = [base * (0.9 + 0.2 * rng.random()) for _ in range(count)]
    return struct.pack(f"<{count}e", *vals)


def _build_synthetic_bf16_qwen3_checkpoint(
    out_dir: pathlib.Path,
    *,
    is_moe: bool,
    seed: int = 1234,
    num_layers: int = 2,
    hidden_size: int = 128,
    head_dim: int = 64,
    num_heads: int = 2,
    num_kv_heads: int = 1,
    vocab_size: int = 320,
    num_experts: int = 8,
    top_k: int = 2,
    moe_inter: int = 128,
    dense_inter: int = 192,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Create a valid BF16 Qwen3 safetensors dir + config.json + tokenizer.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    # Borrow a valid BPE tokenizer.json with matching vocab_size from synth helper.
    tok_bundle = generate_qwen3_bundle(
        out_dir / "_tok_tmp",
        PRESETS["tiny-moe" if is_moe else "tiny-dense"].with_overrides(
            vocab_size=vocab_size,
            num_hidden_layers=1,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        ),
        seed=seed,
    )
    tok_path = out_dir / "tokenizer.json"
    tok_path.write_bytes(tok_bundle.tokenizer_path.read_bytes())
    cfg_data = json.loads(tok_bundle.config_path.read_text(encoding="utf-8"))
    cfg_data.pop("quantization", None)
    cfg_data.pop("quantization_config", None)
    cfg_data["torch_dtype"] = "bfloat16"
    cfg_data["num_hidden_layers"] = num_layers
    cfg_data["tie_word_embeddings"] = False
    if is_moe:
        cfg_data["model_type"] = "qwen3_moe"
        cfg_data["num_experts"] = num_experts
        cfg_data["num_experts_per_tok"] = top_k
        cfg_data["moe_intermediate_size"] = moe_inter
    else:
        cfg_data["model_type"] = "qwen3"
        cfg_data["intermediate_size"] = dense_inter
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg_data, indent=2) + "\n", encoding="utf-8")

    inv_h = 1.0 / math.sqrt(hidden_size)
    inv_q = 1.0 / math.sqrt(q_dim)
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    tensors["model.embed_tokens.weight"] = (
        "BF16",
        (vocab_size, hidden_size),
        _rand_bf16_bytes(rng, vocab_size * hidden_size, 0.25),
    )
    for l in range(num_layers):
        ln_scale = 1.2 if l == 0 else 0.85
        tensors[f"model.layers.{l}.input_layernorm.weight"] = (
            "F16",
            (hidden_size,),
            _rand_f16_ones_bytes(rng, hidden_size, ln_scale),
        )
        tensors[f"model.layers.{l}.self_attn.q_proj.weight"] = (
            "BF16",
            (q_dim, hidden_size),
            _rand_bf16_bytes(rng, q_dim * hidden_size, 0.6 * inv_h, outlier_prob=0.02),
        )
        tensors[f"model.layers.{l}.self_attn.k_proj.weight"] = (
            "BF16",
            (kv_dim, hidden_size),
            _rand_bf16_bytes(rng, kv_dim * hidden_size, 0.6 * inv_h),
        )
        tensors[f"model.layers.{l}.self_attn.v_proj.weight"] = (
            "BF16",
            (kv_dim, hidden_size),
            _rand_bf16_bytes(rng, kv_dim * hidden_size, 0.6 * inv_h, outlier_prob=0.02),
        )
        tensors[f"model.layers.{l}.self_attn.o_proj.weight"] = (
            "BF16",
            (hidden_size, q_dim),
            _rand_bf16_bytes(rng, hidden_size * q_dim, 0.6 * inv_q),
        )
        tensors[f"model.layers.{l}.self_attn.q_norm.weight"] = (
            "F16",
            (head_dim,),
            _rand_f16_ones_bytes(rng, head_dim),
        )
        tensors[f"model.layers.{l}.self_attn.k_norm.weight"] = (
            "F16",
            (head_dim,),
            _rand_f16_ones_bytes(rng, head_dim),
        )
        tensors[f"model.layers.{l}.post_attention_layernorm.weight"] = (
            "F16",
            (hidden_size,),
            _rand_f16_ones_bytes(rng, hidden_size, ln_scale),
        )
        if is_moe:
            tensors[f"model.layers.{l}.mlp.gate.weight"] = (
                "BF16",
                (num_experts, hidden_size),
                _rand_bf16_bytes(rng, num_experts * hidden_size, 0.5 * inv_h),
            )
            inv_m = 1.0 / math.sqrt(moe_inter)
            for e in range(num_experts):
                out_p = 0.03 if e < 2 else 0.0
                tensors[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"] = (
                    "BF16",
                    (moe_inter, hidden_size),
                    _rand_bf16_bytes(rng, moe_inter * hidden_size, 0.5 * inv_h, outlier_prob=out_p),
                )
                tensors[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"] = (
                    "BF16",
                    (moe_inter, hidden_size),
                    _rand_bf16_bytes(rng, moe_inter * hidden_size, 0.5 * inv_h, outlier_prob=out_p),
                )
                tensors[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"] = (
                    "BF16",
                    (hidden_size, moe_inter),
                    _rand_bf16_bytes(rng, hidden_size * moe_inter, 0.5 * inv_m, outlier_prob=out_p),
                )
        else:
            inv_d = 1.0 / math.sqrt(dense_inter)
            tensors[f"model.layers.{l}.mlp.gate_proj.weight"] = (
                "BF16",
                (dense_inter, hidden_size),
                _rand_bf16_bytes(rng, dense_inter * hidden_size, 0.5 * inv_h, outlier_prob=0.02 if l == 0 else 0.0),
            )
            tensors[f"model.layers.{l}.mlp.up_proj.weight"] = (
                "BF16",
                (dense_inter, hidden_size),
                _rand_bf16_bytes(rng, dense_inter * hidden_size, 0.5 * inv_h),
            )
            tensors[f"model.layers.{l}.mlp.down_proj.weight"] = (
                "BF16",
                (hidden_size, dense_inter),
                _rand_bf16_bytes(rng, hidden_size * dense_inter, 0.5 * inv_d),
            )
    tensors["model.norm.weight"] = ("F16", (hidden_size,), _rand_f16_ones_bytes(rng, hidden_size))
    tensors["lm_head.weight"] = (
        "BF16",
        (vocab_size, hidden_size),
        _rand_bf16_bytes(rng, vocab_size * hidden_size, 0.4 * inv_h),
    )

    sf_path = out_dir / "model.safetensors"
    _write_safetensors(sf_path, tensors)
    return sf_path, cfg_path, tok_path


def _cpu_fp32_qwen3_greedy(amf_path: pathlib.Path, cfg: dict[str, object], prompt_tokens: list[int], max_new: int) -> list[int]:
    """Pure-Python CPU FP32 reference forward pass over any `.amf` v0 checkpoint."""
    h = int(cfg["hidden_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    theta = float(cfg.get("rope_theta", 1_000_000.0))
    is_moe = cfg["model_type"] == "qwen3_moe"
    n_exp = int(cfg.get("num_experts", 0))
    top_k = int(cfg.get("num_experts_per_tok", 0))
    norm_topk = bool(cfg.get("norm_topk_prob", True))
    tie = bool(cfg.get("tie_word_embeddings", False))
    group_heads = n_q // n_kv

    def rms_norm(x: list[float], w: list[float]) -> list[float]:
        inv = 1.0 / math.sqrt(math.sumprod(x, x) / len(x) + eps)
        return [xi * inv * wi for xi, wi in zip(x, w)]

    def apply_rope(vec: list[float], pos: int) -> list[float]:
        half = hd // 2
        out = list(vec)
        for i in range(half):
            freq = 1.0 / (theta ** ((2.0 * i) / hd))
            ang = pos * freq
            c = math.cos(ang)
            s = math.sin(ang)
            x0 = vec[i]
            x1 = vec[i + half]
            out[i] = x0 * c - x1 * s
            out[i + half] = x0 * s + x1 * c
        return out

    def matvec(mat: list[list[float]], x: list[float]) -> list[float]:
        return [math.sumprod(row, x) for row in mat]

    def silu(v: float) -> float:
        return v / (1.0 + math.exp(-v))

    with AmfReader(amf_path) as reader:
        def find(base: str):
            return reader.by_name.get(base) or reader.by_name[f"{base}.weight"]

        def load_1d(name: str) -> list[float]:
            e = reader.by_name[name]
            raw = reader.read_tensor_f16_bytes(e)
            return list(struct.unpack(f"<{e.shape[0]}e", raw))

        embed = reader.read_matrix_f32(find("embed_tokens"))
        final_norm = load_1d("norm.weight")
        lm_head = embed if tie else reader.read_matrix_f32(find("lm_head"))

        layers = []
        for l in range(n_layers):
            ld: dict[str, object] = {
                "in_ln": load_1d(f"layers.{l}.input_layernorm.weight"),
                "q": reader.read_matrix_f32(find(f"layers.{l}.self_attn.q_proj")),
                "k": reader.read_matrix_f32(find(f"layers.{l}.self_attn.k_proj")),
                "v": reader.read_matrix_f32(find(f"layers.{l}.self_attn.v_proj")),
                "o": reader.read_matrix_f32(find(f"layers.{l}.self_attn.o_proj")),
                "q_norm": load_1d(f"layers.{l}.self_attn.q_norm.weight"),
                "k_norm": load_1d(f"layers.{l}.self_attn.k_norm.weight"),
                "post_ln": load_1d(f"layers.{l}.post_attention_layernorm.weight"),
            }
            if is_moe:
                ld["router"] = reader.read_matrix_f32(find(f"layers.{l}.mlp.gate"))
                ld["experts"] = [
                    (
                        reader.read_matrix_f32(find(f"layers.{l}.mlp.experts.{e}.gate_proj")),
                        reader.read_matrix_f32(find(f"layers.{l}.mlp.experts.{e}.up_proj")),
                        reader.read_matrix_f32(find(f"layers.{l}.mlp.experts.{e}.down_proj")),
                    )
                    for e in range(n_exp)
                ]
            else:
                ld["gate"] = reader.read_matrix_f32(find(f"layers.{l}.mlp.gate_proj"))
                ld["up"] = reader.read_matrix_f32(find(f"layers.{l}.mlp.up_proj"))
                ld["down"] = reader.read_matrix_f32(find(f"layers.{l}.mlp.down_proj"))
            layers.append(ld)

    k_cache: list[list[list[float]]] = [[] for _ in range(n_layers)]
    v_cache: list[list[list[float]]] = [[] for _ in range(n_layers)]

    def step(tok: int, pos: int) -> int:
        x = list(embed[tok])
        for l in range(n_layers):
            ld = layers[l]
            xn = rms_norm(x, ld["in_ln"])  # type: ignore[arg-type]
            q = matvec(ld["q"], xn)  # type: ignore[arg-type]
            k = matvec(ld["k"], xn)  # type: ignore[arg-type]
            v = matvec(ld["v"], xn)  # type: ignore[arg-type]

            qn = ld["q_norm"]
            kn = ld["k_norm"]
            q_rot: list[float] = []
            for hi in range(n_q):
                qh = rms_norm(q[hi * hd : (hi + 1) * hd], qn)  # type: ignore[arg-type]
                q_rot.extend(apply_rope(qh, pos))
            k_rot: list[float] = []
            for hi in range(n_kv):
                kh = rms_norm(k[hi * hd : (hi + 1) * hd], kn)  # type: ignore[arg-type]
                k_rot.extend(apply_rope(kh, pos))

            k_cache[l].append(k_rot)
            v_cache[l].append(v)

            attn_out = [0.0] * (n_q * hd)
            inv_sqrt_hd = 1.0 / math.sqrt(hd)
            seq_len = pos + 1
            for qh in range(n_q):
                kvh = qh // group_heads
                qv = q_rot[qh * hd : (qh + 1) * hd]
                scores = [
                    math.sumprod(qv, k_cache[l][t][kvh * hd : (kvh + 1) * hd]) * inv_sqrt_hd
                    for t in range(seq_len)
                ]
                m_s = max(scores)
                exps = [math.exp(s - m_s) for s in scores]
                s_exp = sum(exps)
                for t in range(seq_len):
                    p = exps[t] / s_exp
                    vv = v_cache[l][t][kvh * hd : (kvh + 1) * hd]
                    base = qh * hd
                    for d in range(hd):
                        attn_out[base + d] += p * vv[d]

            o_out = matvec(ld["o"], attn_out)  # type: ignore[arg-type]
            x = [xi + oi for xi, oi in zip(x, o_out)]

            xn2 = rms_norm(x, ld["post_ln"])  # type: ignore[arg-type]
            if is_moe:
                r_logits = matvec(ld["router"], xn2)  # type: ignore[arg-type]
                m_r = max(r_logits)
                exps = [math.exp(v_r - m_r) for v_r in r_logits]
                s_r = sum(exps)
                probs = [v_e / s_r for v_e in exps]
                ranked = sorted(range(n_exp), key=lambda ei: (-probs[ei], ei))[:top_k]
                r_weights = [probs[ei] for ei in ranked]
                if norm_topk:
                    rw_sum = sum(r_weights)
                    r_weights = [rw / rw_sum for rw in r_weights]
                experts = ld["experts"]
                acc = [0.0] * h
                for ei, rw in zip(ranked, r_weights):
                    gw, uw, dw = experts[ei]  # type: ignore[index]
                    gv = matvec(gw, xn2)
                    uv = matvec(uw, xn2)
                    inter = [silu(a) * b for a, b in zip(gv, uv)]
                    dv = matvec(dw, inter)
                    for i in range(h):
                        acc[i] += rw * dv[i]
                x = [xi + ai for xi, ai in zip(x, acc)]
            else:
                gv = matvec(ld["gate"], xn2)  # type: ignore[arg-type]
                uv = matvec(ld["up"], xn2)  # type: ignore[arg-type]
                inter = [silu(a) * b for a, b in zip(gv, uv)]
                dv = matvec(ld["down"], inter)  # type: ignore[arg-type]
                x = [xi + di for xi, di in zip(x, dv)]

        xf = rms_norm(x, final_norm)
        logits = matvec(lm_head, xf)
        return max(range(len(logits)), key=lambda idx: (logits[idx], -idx))

    next_tok = 0
    for pos, tok in enumerate(prompt_tokens):
        next_tok = step(tok, pos)
    generated = [next_tok]
    for step_idx in range(1, max_new):
        next_tok = step(next_tok, len(prompt_tokens) - 1 + step_idx)
        generated.append(next_tok)
    return generated


class M5GsqRcoTests(unittest.TestCase):
    def test_q2_affine_pack_dequant_amf_and_sme2_roundtrip(self) -> None:
        n_rows, k = 19, 128
        groups = k // 64
        rng = random.Random(2026)

        codes_mat = [[rng.randrange(4) for _ in range(k)] for _ in range(n_rows)]
        q_rows = [pack_mlx_affine_2bit(r) for r in codes_mat]
        sbs_raw = bytearray()
        ref_f32: list[list[float]] = []
        for r in range(n_rows):
            row_vals: list[float] = []
            for g in range(groups):
                s = struct.unpack("<e", struct.pack("<e", 0.05 + 0.01 * ((r + g) % 5)))[0]
                b = struct.unpack("<e", struct.pack("<e", -0.075 - 0.01 * ((r * 3 + g) % 4)))[0]
                sbs_raw.extend(struct.pack("<2e", s, b))
                for j in range(64):
                    row_vals.append(b + codes_mat[r][g * 64 + j] * s)
            ref_f32.append(row_vals)

        # 1. Row-major dequantize_q2_row_f32 matches reference exactly.
        row_p_bytes = groups * 4
        for r in range(n_rows):
            got = dequantize_q2_row_f32(
                q_rows[r],
                sbs_raw[r * row_p_bytes : (r + 1) * row_p_bytes],
                k,
                group_size=64,
            )
            for a, b in zip(ref_f32[r], got):
                self.assertAlmostEqual(a, b, places=5)

        # 2. SME2 panel64 q2 packing & dequantization matches row-major exactly.
        self.assertEqual(sme2_panel64_data_bytes(DType.Q2_AFFINE, n_rows, k, 64), 2 * groups * 384)
        p0_bytes = pack_sme2_panel64_q2_from_mlx_panel(
            q_rows[:16],
            [sbs_raw[r * row_p_bytes : (r + 1) * row_p_bytes] for r in range(16)],
            k,
        )
        p0_deq = dequantize_sme2_panel64_q2_panel_f32(p0_bytes, k, num_rows=16)
        for r in range(16):
            for a, b in zip(ref_f32[r], p0_deq[r]):
                self.assertAlmostEqual(a, b, places=5)

        # 3. .amf v0 pack + AmfReader + repack_amf_sme2 + Zig verify-amf round-trip!
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            raw_amf = tmp / "q2_row.amf"
            sme_amf = tmp / "q2_sme.amf"
            write_amf(
                raw_amf,
                model_id="q2-test",
                records=[
                    (
                        "layers.0.mlp.gate",
                        TensorKind.ROUTER,
                        DType.Q2_AFFINE,
                        2,
                        64,
                        0,
                        0xFFFF,
                        (n_rows, k),
                        b"".join(q_rows),
                        bytes(sbs_raw),
                    )
                ],
            )
            _, metas = load_amf_metadata(raw_amf)
            self.assertEqual(metas[0].dtype, DType.Q2_AFFINE)
            self.assertEqual(int(metas[0].dtype), 6)
            self.assertEqual(metas[0].quant_bits, 2)
            self.assertEqual(metas[0].group_size, 64)

            with AmfReader(raw_amf) as rd:
                mat = rd.read_matrix_f32(rd.entries[0])
                for r in range(n_rows):
                    for a, b in zip(ref_f32[r], mat[r]):
                        self.assertAlmostEqual(a, b, places=5)

            repack_amf_sme2(raw_amf, sme_amf, targets="routers")
            with AmfReader(sme_amf) as rd:
                self.assertEqual(rd.entries[0].layout, TensorLayout.SME2_PANEL64)
                self.assertEqual(rd.entries[0].dtype, DType.Q2_AFFINE)
                mat_sme = rd.read_matrix_f32(rd.entries[0])
                for r in range(n_rows):
                    for a, b in zip(ref_f32[r], mat_sme[r]):
                        self.assertAlmostEqual(a, b, places=5)

            if AMINFER_BIN.exists():
                for p in (raw_amf, sme_amf):
                    proc = subprocess.run([str(AMINFER_BIN), "verify-amf", str(p)], capture_output=True, text=True)
                    self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
                    self.assertIn("amf.status       ok", proc.stdout)

    def test_rco_provably_reduces_mse_vs_minmax_across_q2_q3_q4(self) -> None:
        rng = random.Random(4242)
        groups: list[list[float]] = [
            # Gaussian groups
            [rng.gauss(0.0, 0.08) for _ in range(64)],
            [rng.gauss(0.05, 0.12) for _ in range(64)],
            # Outlier-contaminated groups (where RCO clipping + OLS yields large wins)
            [rng.gauss(0.0, 0.05) for _ in range(62)] + [1.25, -1.10],
            [rng.gauss(0.0, 0.04) for _ in range(63)] + [0.95],
            # Constant & near-zero edge cases
            [0.25] * 64,
            [0.0] * 64,
            [1e-7 * i for i in range(64)],
            # Skewed distribution
            [abs(rng.gauss(0.0, 0.1)) for _ in range(64)],
        ]

        for bits in (2, 3, 4):
            total_mm = 0.0
            total_rco = 0.0
            for grp in groups:
                _, _, _, mse_mm = quantize_group_minmax(grp, bits)
                s_rco, b_rco, codes_rco, mse_rco = quantize_group_rco(grp, bits, iters=3)
                # Hard invariant: RCO NEVER degrades MSE vs min-max.
                self.assertLessEqual(mse_rco, mse_mm + 1e-15, msg=f"bits={bits}")
                # Check reconstruction matches reported mse_rco.
                recon_mse = sum((x - (b_rco + q * s_rco)) ** 2 for x, q in zip(grp, codes_rco)) / 64.0
                self.assertAlmostEqual(mse_rco, recon_mse, places=10)
                total_mm += mse_mm
                total_rco += mse_rco

            # Aggregate MSE across Gaussian + outlier groups must be strictly lower with RCO!
            min_gain = 0.20 if bits == 2 else (0.12 if bits == 3 else 0.03)
            self.assertLess(
                total_rco,
                total_mm * (1.0 - min_gain),
                msg=f"expected >={int(min_gain * 100)}% MSE reduction for bits={bits} (rco={total_rco:.6g}, mm={total_mm:.6g})",
            )

        # Row-level quantize_rows_affine round-trip check.
        rows = [[rng.gauss(0.0, 0.1) for _ in range(128)] for _ in range(4)]
        qb, pb, rco_mse, mm_mse = quantize_rows_affine(rows, 2, use_rco=True, compute_minmax_mse=True)
        self.assertLessEqual(rco_mse, mm_mse)
        deq0 = dequantize_q2_row_f32(qb[:32], pb[:8], 128, group_size=64)
        self.assertEqual(len(deq0), 128)

    def test_gsq_knapsack_respects_budget_and_sensitivity_priority(self) -> None:
        # 4 identical-sized experts with distinct sensitivities (hot vs cold).
        units: list[QuantUnitInput] = []
        numel = 3 * 128 * 128
        for e, sens in enumerate((10.0, 4.0, 1.5, 0.3)):
            cands = (
                QuantUnitCandidate(DType.Q2_AFFINE, (numel * 20) // 64, sens * numel * 0.12, 0.12),
                QuantUnitCandidate(DType.Q3_AFFINE, (numel * 28) // 64, sens * numel * 0.03, 0.03),
                QuantUnitCandidate(DType.Q4_AFFINE, (numel * 36) // 64, sens * numel * 0.008, 0.008),
                QuantUnitCandidate(DType.BF16, numel * 2, 0.0, 0.0),
            )
            units.append(
                QuantUnitInput(
                    unit_id=f"layers.0.mlp.experts.{e}",
                    tensor_names=(f"e{e}.g", f"e{e}.u", f"e{e}.d"),
                    role="moe_expert",
                    layer=0,
                    expert=e,
                    numel=numel,
                    sensitivity=sens,
                    candidates=cands,
                )
            )
        rep = solve_gsq_allocation(units, target_bits=3.5)
        self.assertLessEqual(rep.effective_bits, 3.5 + 1e-6)
        bits_by_e = {u.expert: {DType.Q2_AFFINE: 2, DType.Q3_AFFINE: 3, DType.Q4_AFFINE: 4, DType.BF16: 16}[u.chosen_dtype] for u in rep.units}
        # Hottest expert (e=0) MUST receive strictly more bits than coldest expert (e=3)!
        self.assertGreater(bits_by_e[0], bits_by_e[3])
        self.assertGreaterEqual(bits_by_e[0], bits_by_e[1])
        self.assertGreaterEqual(bits_by_e[1], bits_by_e[2])
        self.assertGreaterEqual(bits_by_e[2], bits_by_e[3])

    def test_gsq_convert_moe_with_heat_and_pack_amf_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            sf_dir = tmp / "moe_bf16"
            sf_path, cfg_path, _ = _build_synthetic_bf16_qwen3_checkpoint(sf_dir, is_moe=True, num_layers=2, num_experts=8)

            # W8 schema-v1 heat profile: experts 0, 1 are hot; experts 6, 7 are cold.
            heat_path = tmp / "heat.json"
            heat_doc = {
                "schema_version": 1,
                "model_id": "test-moe",
                "num_layers": 2,
                "num_experts": 8,
                "top_k": 2,
                "layers": [
                    {
                        "layer": l,
                        "routing_prob": [0.35, 0.25, 0.14, 0.10, 0.07, 0.05, 0.025, 0.015],
                        "ranked_hot_to_cold": [0, 1, 2, 3, 4, 5, 6, 7],
                        "cold_experts": [6, 7],
                    }
                    for l in range(2)
                ],
            }
            heat_path.write_text(json.dumps(heat_doc), encoding="utf-8")

            # 1. pack-amf default (uniform) preserves BF16 untouched.
            uni_amf = tmp / "uniform.amf"
            uni_metas = pack_amf_from_safetensors(sf_dir, uni_amf, model_id="uni-moe")
            for m in uni_metas:
                if m.kind != TensorKind.NORM:
                    self.assertEqual(m.dtype, DType.BF16)

            # 2. pack-amf --quant-profile gsq-rco hits target-bits & respects heat.json!
            gsq_amf = tmp / "gsq_moe.amf"
            manifest_path = tmp / "gsq_manifest.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "pack-amf",
                        "--input",
                        str(sf_dir),
                        "--output",
                        str(gsq_amf),
                        "--model-id",
                        "gsq-moe",
                        "--quant-profile",
                        "gsq-rco",
                        "--target-bits",
                        "3.4",
                        "--heat-profile",
                        str(heat_path),
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(rc, 0)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertLessEqual(manifest["effective_bits"], 3.45)

            _, metas = load_amf_metadata(gsq_amf)
            by_name = {m.name: m for m in metas}
            # Norms stay unquantized F16; routers stay protected at BF16.
            self.assertEqual(by_name["norm.weight"].dtype, DType.F16)
            self.assertEqual(by_name["layers.0.mlp.gate.weight"].dtype, DType.BF16)
            # Every MoE expert triple MUST have one uniform DType (ADR-0017), and hot > cold!
            hot_bits_sum = 0
            cold_bits_sum = 0
            seen_exp_dtypes: set[DType] = set()
            dt_bits = {DType.Q2_AFFINE: 2, DType.Q3_AFFINE: 3, DType.Q4_AFFINE: 4, DType.BF16: 16, DType.F16: 16}
            for l in range(2):
                for e in range(8):
                    gp = by_name.get(f"layers.{l}.mlp.experts.{e}.gate_proj") or by_name[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                    up = by_name.get(f"layers.{l}.mlp.experts.{e}.up_proj") or by_name[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
                    dp = by_name.get(f"layers.{l}.mlp.experts.{e}.down_proj") or by_name[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
                    self.assertEqual(gp.dtype, up.dtype)
                    self.assertEqual(gp.dtype, dp.dtype)
                    seen_exp_dtypes.add(gp.dtype)
                    if e in (0, 1):
                        hot_bits_sum += dt_bits[gp.dtype]
                    if e in (6, 7):
                        cold_bits_sum += dt_bits[gp.dtype]
            self.assertGreater(len(seen_exp_dtypes), 1, msg="expected heterogeneous per-expert schemes")
            self.assertGreater(hot_bits_sum, cold_bits_sum, msg="hot experts should receive higher precision than cold")

            # 3. gsq-convert directly from an existing .amf file works cleanly too!
            re_gsq_amf = tmp / "from_amf_gsq.amf"
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = main(
                    [
                        "gsq-convert",
                        "--input",
                        str(uni_amf),
                        "--output",
                        str(re_gsq_amf),
                        "--target-bits",
                        "3.25",
                        "--heat-profile",
                        str(heat_path),
                    ]
                )
            self.assertEqual(rc2, 0)
            self.assertIn("gsq-converted", buf2.getvalue())

    def test_end_to_end_python_gsq_rco_amf_matches_zig_qwen3_decode_cpu_fp32_ref(self) -> None:
        self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            for is_moe, label in ((True, "hetero_moe"), (False, "mixed_dense")):
                sf_dir = tmp / label
                _, cfg_path, tok_path = _build_synthetic_bf16_qwen3_checkpoint(
                    sf_dir,
                    is_moe=is_moe,
                    seed=777 if is_moe else 888,
                    num_layers=2,
                    num_experts=8,
                    top_k=2,
                )
                heat_path: pathlib.Path | None = None
                if is_moe:
                    heat_path = tmp / f"{label}_heat.json"
                    heat_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "model_id": label,
                                "num_layers": 2,
                                "num_experts": 8,
                                "top_k": 2,
                                "layers": [
                                    {
                                        "layer": l,
                                        "routing_prob": [0.36, 0.24, 0.14, 0.10, 0.07, 0.05, 0.025, 0.015],
                                        "ranked_hot_to_cold": [0, 1, 2, 3, 4, 5, 6, 7],
                                        "cold_experts": [],
                                    }
                                    for l in range(2)
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )

                out_amf = tmp / f"{label}_gsq.amf"
                rep = gsq_convert(
                    sf_dir,
                    out_amf,
                    target_bits=3.35,
                    model_id=f"m5-{label}",
                    heat_profile=heat_path,
                    use_rco=True,
                )
                self.assertLessEqual(rep.effective_bits, 3.45)

                # Confirm the generated .amf genuinely exercises q2_affine + mixed schemes!
                present_dtypes = {m.dtype for m in rep.metas}
                self.assertIn(DType.Q2_AFFINE, present_dtypes, msg=f"{label} should contain q2_affine tensors")
                self.assertGreaterEqual(len(present_dtypes), 3)

                # 1. Zig verify-amf checks 16 KiB alignment, CRC-32, DType.q2_affine wire value 6.
                v_proc = subprocess.run(
                    [str(AMINFER_BIN), "verify-amf", str(out_amf)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(v_proc.returncode, 0, msg=v_proc.stdout + v_proc.stderr)
                self.assertIn("amf.status       ok", v_proc.stdout)

                # 2. Zig Qwen3Model Metal 4 decode matches pure-Python CPU FP32 reference token-for-token!
                d_proc = subprocess.run(
                    [
                        str(AMINFER_BIN),
                        "decode",
                        "--amf",
                        str(out_amf),
                        "--config",
                        str(cfg_path),
                        "--tokenizer",
                        str(tok_path),
                        "--prompt",
                        "",
                        "--max-tokens",
                        "4",
                        "--ignore-eos",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                self.assertEqual(d_proc.returncode, 0, msg=f"{label} decode failed:\n{d_proc.stdout}\n{d_proc.stderr}")
                self.assertIn("decode.swapouts_delta            0", d_proc.stdout)
                m_toks = re.search(r"decode\.token_ids\s+\[([^\]]*)\]", d_proc.stdout)
                self.assertIsNotNone(m_toks, msg=d_proc.stdout)
                assert m_toks is not None
                zig_tokens = [int(x.strip()) for x in m_toks.group(1).split(",") if x.strip()]
                self.assertEqual(len(zig_tokens), 4)

                cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
                bos_id = int(cfg_dict.get("bos_token_id", 0))
                ref_tokens = _cpu_fp32_qwen3_greedy(out_amf, cfg_dict, [bos_id], max_new=4)
                self.assertEqual(
                    zig_tokens,
                    ref_tokens,
                    msg=f"{label} Zig Metal 4 tokens {zig_tokens} != CPU FP32 ref {ref_tokens}",
                )

    def test_gsq_scale_invariant_ln_proxy_protects_boundary_layers_and_attention(self) -> None:
        """Depth-growing RMSNorm gammas (1000x larger at deep layers) must not starve layer 0."""
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            rng = random.Random(20261121)
            tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
            num_layers = 6
            h = 64
            inter = 128
            vocab = 128

            def _vals_to_bf16(vals: list[float]) -> bytes:
                f32_buf = struct.pack(f"<{len(vals)}f", *vals)
                out = bytearray(len(vals) * 2)
                out[0::2] = f32_buf[2::4]
                out[1::2] = f32_buf[3::4]
                return bytes(out)

            def _rand_bf16(rows: int, cols: int, scale: float = 0.05) -> bytes:
                return _rand_bf16_bytes(rng, rows * cols, scale)

            tensors["model.embed_tokens.weight"] = ("BF16", (vocab, h), _rand_bf16(vocab, h))
            tensors["model.norm.weight"] = ("BF16", (h,), _vals_to_bf16([1.0] * h))
            for l in range(num_layers):
                # Emulate Qwen3 depth growth: layer 0 has tiny gamma (0.05), deep layers grow 100x+.
                gamma_scale = 0.05 * (4.0 ** l)
                tensors[f"model.layers.{l}.input_layernorm.weight"] = (
                    "BF16",
                    (h,),
                    _vals_to_bf16([gamma_scale] * h),
                )
                tensors[f"model.layers.{l}.post_attention_layernorm.weight"] = (
                    "BF16",
                    (h,),
                    _vals_to_bf16([gamma_scale] * h),
                )
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    tensors[f"model.layers.{l}.self_attn.{proj}.weight"] = ("BF16", (h, h), _rand_bf16(h, h))
                for proj, shape in (
                    ("gate_proj", (inter, h)),
                    ("up_proj", (inter, h)),
                    ("down_proj", (h, inter)),
                ):
                    tensors[f"model.layers.{l}.mlp.{proj}.weight"] = ("BF16", shape, _rand_bf16(*shape))

            sf_dir = tmp / "model"
            sf_dir.mkdir()
            _write_safetensors(sf_dir / "model.safetensors", tensors)
            (sf_dir / "config.json").write_text(
                json.dumps({"tie_word_embeddings": True, "num_hidden_layers": num_layers}),
                encoding="utf-8",
            )
            out_amf = tmp / "out.amf"
            rep = gsq_convert(sf_dir, out_amf, target_bits=4.15, model_id="ln-scale-test", use_rco=True)
            chosen = {
                name.removesuffix(".weight"): u.chosen_dtype
                for u in rep.allocation.units
                for name in u.tensor_names
            }
            # Boundary layers (0 and num_layers - 1) and attention projections must be >= q4_affine.
            for name in (
                "embed_tokens",
                "layers.0.self_attn.o_proj",
                "layers.0.mlp.down_proj",
                "layers.0.mlp.gate_proj",
                f"layers.{num_layers - 1}.self_attn.o_proj",
                f"layers.{num_layers - 1}.mlp.down_proj",
            ):
                self.assertIn(
                    chosen[name],
                    (DType.Q4_AFFINE, DType.BF16, DType.F16),
                    msg=f"{name} unexpectedly dropped to {chosen[name]}",
                )
            # Layer 0 o_proj sensitivity must not be lower than interior layer 3 o_proj sensitivity.
            sens = {
                name.removesuffix(".weight"): u.sensitivity
                for u in rep.allocation.units
                for name in u.tensor_names
            }
            self.assertGreater(sens["layers.0.self_attn.o_proj"], sens["layers.3.self_attn.o_proj"])

    def test_weighted_ols_rco_and_extended_quant_rungs_g32_g64_g128(self) -> None:
        rng = random.Random(20261124)
        deq_by_bits = {
            2: dequantize_q2_row_f32,
            3: dequantize_q3_row_f32,
            4: dequantize_q4_row_f32,
            5: dequantize_q5_row_f32,
            6: dequantize_q6_row_f32,
            8: dequantize_q8_row_f32,
        }
        for gs in (32, 64, 128):
            rows = [[rng.gauss(0.0, 0.15) for _ in range(gs * 2)] for _ in range(3)]
            for bits, deq_fn in deq_by_bits.items():
                qb, pb, rco_mse, mm_mse = quantize_rows_affine(
                    rows, bits, group_size=gs, use_rco=True, compute_minmax_mse=True
                )
                self.assertLessEqual(rco_mse, mm_mse + 1e-15, msg=f"bits={bits}, gs={gs}")
                row_d_bytes = (len(rows[0]) * bits) // 8
                row_p_bytes = (len(rows[0]) // gs) * 4
                for r_idx, orig_row in enumerate(rows):
                    deq = deq_fn(
                        qb[r_idx * row_d_bytes : (r_idx + 1) * row_d_bytes],
                        pb[r_idx * row_p_bytes : (r_idx + 1) * row_p_bytes],
                        len(orig_row),
                        group_size=gs,
                    )
                    self.assertEqual(len(deq), len(orig_row))

        # High-E[x^2] channels live in [-0.14, 0.14], while low-E[x^2] channels carry large outliers [-2.2, 2.2].
        # Weighted OLS RCO must focus resolution onto the important channels and slash weighted error!
        grp = [rng.uniform(-0.14, 0.14) for _ in range(16)] + [rng.uniform(-0.08, 0.08) for _ in range(44)] + [2.2, -2.1, 1.9, -1.8]
        weights = [100.0] * 16 + [1e-4] * 48
        sum_w = sum(weights)

        def _wmse(s: float, b: float, codes: list[int]) -> float:
            return sum(w * (x - (b + q * s)) ** 2 for x, w, q in zip(grp, weights, codes)) / sum_w

        s_mm, b_mm, c_mm, _ = quantize_group_minmax(grp, 2)
        s_uw, b_uw, c_uw, _ = quantize_group_rco(grp, 2, iters=3)
        s_w, b_w, c_w, reported_wmse = quantize_group_rco(grp, 2, iters=3, weights=weights)
        self.assertAlmostEqual(reported_wmse, _wmse(s_w, b_w, c_w), places=10)
        self.assertLessEqual(reported_wmse, _wmse(s_mm, b_mm, c_mm) + 1e-15)
        self.assertLess(reported_wmse, _wmse(s_uw, b_uw, c_uw) * 0.5)

        # Row-level `col_weights` + end-to-end `gsq_convert(..., imatrix_path=...)` propagation check.
        rows_2d = [
            [rng.uniform(-0.14, 0.14) for _ in range(16)]
            + [rng.uniform(-0.08, 0.08) for _ in range(44)]
            + [2.2, -2.1, 1.9, -1.8]
            for _ in range(8)
        ]
        _, _, w_row_mse, w_mm_mse = quantize_rows_affine(
            rows_2d, 2, group_size=64, use_rco=True, compute_minmax_mse=True, col_weights=weights
        )
        self.assertLessEqual(w_row_mse, w_mm_mse + 1e-15)

        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            sf_dir = tmp / "imatrix_ckpt"
            sf_dir.mkdir()

            def _bf16_from_rows(mat: list[list[float]]) -> bytes:
                flat = [v for r in mat for v in r]
                f32 = struct.pack(f"<{len(flat)}f", *flat)
                out = bytearray(len(flat) * 2)
                out[0::2] = f32[2::4]
                out[1::2] = f32[3::4]
                return bytes(out)

            ffn_mat = [
                [rng.uniform(-0.14, 0.14) for _ in range(16)]
                + [rng.uniform(-0.08, 0.08) for _ in range(44)]
                + [2.2, -2.1, 1.9, -1.8]
                for _ in range(256)
            ]
            tensors = {
                "model.embed_tokens.weight": ("BF16", (64, 64), _rand_bf16_bytes(rng, 64 * 64, 0.1)),
                "model.layers.0.input_layernorm.weight": ("BF16", (64,), _bf16_from_rows([[1.0] * 64])),
                "model.layers.0.self_attn.q_proj.weight": ("BF16", (64, 64), _rand_bf16_bytes(rng, 64 * 64, 0.1)),
                "model.layers.0.mlp.gate_proj.weight": ("BF16", (256, 64), _bf16_from_rows(ffn_mat)),
                "model.norm.weight": ("BF16", (64,), _bf16_from_rows([[1.0] * 64])),
            }
            _write_safetensors(sf_dir / "model.safetensors", tensors)
            (sf_dir / "config.json").write_text(json.dumps({"tie_word_embeddings": True}), encoding="utf-8")

            im_path = tmp / "imatrix.json"
            im_path.write_text(json.dumps({"layers.0.mlp.gate_proj": weights}), encoding="utf-8")

            amf_uw = tmp / "uw.amf"
            amf_w = tmp / "w.amf"
            gsq_convert(sf_dir, amf_uw, target_bits=3.05, model_id="uw-test")
            gsq_convert(sf_dir, amf_w, target_bits=3.05, model_id="w-test", imatrix_path=im_path)

            with AmfReader(amf_uw) as r_uw, AmfReader(amf_w) as r_w:
                m_uw = r_uw.by_name["layers.0.mlp.gate_proj"]
                m_w = r_w.by_name["layers.0.mlp.gate_proj"]
                self.assertEqual(m_uw.dtype, DType.Q2_AFFINE)
                self.assertEqual(m_w.dtype, DType.Q2_AFFINE)
                mat_uw = r_uw.read_matrix_f32(m_uw)
                mat_w = r_w.read_matrix_f32(m_w)
                err_uw = sum(
                    weights[c] * (mat_uw[r][c] - ffn_mat[r][c]) ** 2
                    for r in range(256)
                    for c in range(16)
                )
                err_w = sum(
                    weights[c] * (mat_w[r][c] - ffn_mat[r][c]) ** 2
                    for r in range(256)
                    for c in range(16)
                )
                self.assertLess(err_w, err_uw * 0.5)

    def test_backbone_protection_floor_and_max_q2_share(self) -> None:
        for role in ("router", "embedding", "lm_head", "attn_qk", "attn_v", "attn_o", "vision", "ssm_dt_a", "embed_head", "attn_qkv_o", "ssm_proj"):
            self.assertIn(role, _BACKBONE_PROTECTED_ROLES)

        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            sf_dir = tmp / "dense_bf16"
            _build_synthetic_bf16_qwen3_checkpoint(sf_dir, is_moe=False, num_layers=2)
            # Backbone floor prevents dropping protected backbone roles to q2_affine at target_bits=2.1!
            with self.assertRaises(ValueError):
                gsq_convert(sf_dir, tmp / "too_low.amf", target_bits=2.1, model_id="too-low")

        # Unit-level `solve_gsq_allocation` checks for `enforce_backbone_floor` and `max_q2_share`.
        numel = 64 * 64
        def _mk_unit(uid: str, role: str, sens: float) -> QuantUnitInput:
            return QuantUnitInput(
                unit_id=uid,
                tensor_names=(uid,),
                role=role,
                layer=0,
                expert=0,
                numel=numel,
                sensitivity=sens,
                candidates=(
                    QuantUnitCandidate(DType.Q2_AFFINE, (numel * 20) // 64, sens * numel * 0.12, 0.12, 64),
                    QuantUnitCandidate(DType.Q3_AFFINE, (numel * 28) // 64, sens * numel * 0.03, 0.03, 64),
                    QuantUnitCandidate(DType.Q4_AFFINE, (numel * 36) // 64, sens * numel * 0.008, 0.008, 64),
                    QuantUnitCandidate(DType.BF16, numel * 2, 0.0, 0.0, 0),
                ),
            )

        units = [
            _mk_unit("layers.0.mlp.gate", "router", 10.0),
            _mk_unit("layers.0.self_attn.q_proj", "attn_qkv_o", 5.0),
            _mk_unit("embed_tokens", "embed_head", 5.0),
            _mk_unit("layers.0.ssm.dt_a", "ssm_dt_a", 5.0),
            _mk_unit("layers.0.ssm.in_proj", "ssm_proj", 5.0),
            _mk_unit("layers.0.mlp.experts.0", "moe_expert", 3.0),
            _mk_unit("layers.0.mlp.experts.1", "moe_expert", 2.0),
            _mk_unit("layers.0.mlp.experts.2", "moe_expert", 1.0),
            _mk_unit("layers.0.mlp.experts.3", "moe_expert", 0.5),
        ]
        with self.assertRaises(ValueError):
            solve_gsq_allocation(units, target_bits=2.1, enforce_backbone_floor=True)

        rep = solve_gsq_allocation(units, target_bits=3.5, enforce_backbone_floor=True, max_q2_share=0.25)
        for u in rep.units:
            if u.role in _BACKBONE_PROTECTED_ROLES:
                self.assertNotEqual(u.chosen_dtype, DType.Q2_AFFINE)
        q2_w = sum(u.numel for u in rep.units if u.chosen_dtype == DType.Q2_AFFINE)
        total_q = sum(u.numel for u in rep.units)
        self.assertLessEqual(q2_w / total_q, 0.25 + 1e-9)

        with self.assertRaises(ValueError):
            solve_gsq_allocation(units, target_bits=2.8, enforce_backbone_floor=True, max_q2_share=0.0)

    def test_cli_lowbit_retranscode_guard_and_auto_target_bits(self) -> None:
        rng = random.Random(3030)
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)

            # 1. Build a synthetic 3-bit MLX MoE safetensors checkpoint & pack to 3-bit .amf.
            h, inter, num_experts, num_layers = 128, 128, 4, 1
            def _make_q3(rows: int, cols: int) -> tuple[bytes, bytes, bytes]:
                groups = cols // 64
                qb = bytearray()
                sv: list[float] = []
                bv: list[float] = []
                for _ in range(rows):
                    qb.extend(pack_mlx_affine_3bit([rng.randint(0, 7) for _ in range(cols)]))
                    for _ in range(groups):
                        sv.append(rng.uniform(0.01, 0.04))
                        bv.append(rng.uniform(-0.1, 0.1))
                f32_s = struct.pack(f"<{len(sv)}f", *sv)
                f32_b = struct.pack(f"<{len(bv)}f", *bv)
                s_bf16 = bytearray(len(sv) * 2)
                b_bf16 = bytearray(len(bv) * 2)
                s_bf16[0::2] = f32_s[2::4]
                s_bf16[1::2] = f32_s[3::4]
                b_bf16[0::2] = f32_b[2::4]
                b_bf16[1::2] = f32_b[3::4]
                return bytes(qb), bytes(s_bf16), bytes(b_bf16)

            def _add_q3(td_map: dict[str, tuple[str, tuple[int, ...], bytes]], stem: str, rows: int, cols: int) -> None:
                wd, sd, bd = _make_q3(rows, cols)
                td_map[f"{stem}.weight"] = ("U32", (rows, (cols * 3) // 32), wd)
                td_map[f"{stem}.scales"] = ("BF16", (rows, cols // 64), sd)
                td_map[f"{stem}.biases"] = ("BF16", (rows, cols // 64), bd)

            q3_dir = tmp / "moe_3bit"
            q3_dir.mkdir()
            ones_bf16 = bytearray(h * 2)
            f32_1 = struct.pack(f"<{h}f", *([1.0] * h))
            ones_bf16[0::2] = f32_1[2::4]
            ones_bf16[1::2] = f32_1[3::4]
            q3_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
                "model.norm.weight": ("BF16", (h,), bytes(ones_bf16)),
                "model.layers.0.input_layernorm.weight": ("BF16", (h,), bytes(ones_bf16)),
                "model.layers.0.post_attention_layernorm.weight": ("BF16", (h,), bytes(ones_bf16)),
            }
            _add_q3(q3_tensors, "model.embed_tokens", 128, h)
            _add_q3(q3_tensors, "lm_head", 128, h)
            _add_q3(q3_tensors, "model.layers.0.self_attn.q_proj", h, h)
            _add_q3(q3_tensors, "model.layers.0.self_attn.k_proj", 64, h)
            _add_q3(q3_tensors, "model.layers.0.self_attn.v_proj", 64, h)
            _add_q3(q3_tensors, "model.layers.0.self_attn.o_proj", h, h)
            _add_q3(q3_tensors, "model.layers.0.mlp.gate", num_experts, h)
            for proj, r_dim, c_dim in (("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)):
                w_all, s_all, b_all = bytearray(), bytearray(), bytearray()
                for _ in range(num_experts):
                    wd, sd, bd = _make_q3(r_dim, c_dim)
                    w_all.extend(wd)
                    s_all.extend(sd)
                    b_all.extend(bd)
                q3_tensors[f"model.layers.0.mlp.switch_mlp.{proj}.weight"] = ("U32", (num_experts, r_dim, (c_dim * 3) // 32), bytes(w_all))
                q3_tensors[f"model.layers.0.mlp.switch_mlp.{proj}.scales"] = ("BF16", (num_experts, r_dim, c_dim // 64), bytes(s_all))
                q3_tensors[f"model.layers.0.mlp.switch_mlp.{proj}.biases"] = ("BF16", (num_experts, r_dim, c_dim // 64), bytes(b_all))
            _write_safetensors(q3_dir / "model.safetensors", q3_tensors)
            (q3_dir / "config.json").write_text(json.dumps({"tie_word_embeddings": False}), encoding="utf-8")

            # CLI `gsq-convert` refuses 3-bit -> q2_affine double-quantization by default!
            with self.assertRaises(ValueError):
                main([
                    "gsq-convert",
                    "--input", str(q3_dir),
                    "--output", str(tmp / "blocked.amf"),
                    "--target-bits", "2.95",
                ])

            # With `--allow-retranscode-lowbit` (and `--allow-unsafe-q2` since 2.95 needs >55% Q2 on small 4-expert model), succeeds!
            retrans_amf = tmp / "retrans.amf"
            buf_rt = io.StringIO()
            with redirect_stdout(buf_rt):
                rc_rt = main([
                    "gsq-convert",
                    "--input", str(q3_dir),
                    "--output", str(retrans_amf),
                    "--target-bits", "2.95",
                    "--allow-retranscode-lowbit",
                    "--allow-unsafe-q2",
                ])
            self.assertEqual(rc_rt, 0)
            _, rt_metas = load_amf_metadata(retrans_amf)
            self.assertTrue(any(m.dtype == DType.Q2_AFFINE for m in rt_metas))

            # 2. Auto `target_bits`: Dense `--quant-profile gsq-rco` without `--target-bits` -> 0 `q2_affine` & >= 4.0b!
            dense_dir = tmp / "dense_auto"
            _build_synthetic_bf16_qwen3_checkpoint(dense_dir, is_moe=False, num_layers=2)
            dense_pack_amf = tmp / "dense_pack_auto.amf"
            dense_manifest = tmp / "dense_auto_manifest.json"
            buf_dp = io.StringIO()
            with redirect_stdout(buf_dp):
                rc_dp = main([
                    "pack-amf",
                    "--input", str(dense_dir),
                    "--output", str(dense_pack_amf),
                    "--model-id", "dense-auto",
                    "--quant-profile", "gsq-rco",
                    "--manifest", str(dense_manifest),
                ])
            self.assertEqual(rc_dp, 0)
            dm = json.loads(dense_manifest.read_text(encoding="utf-8"))
            self.assertAlmostEqual(dm["target_bits"], 4.15, places=2)
            self.assertGreaterEqual(dm["effective_bits"], 4.0)
            self.assertLessEqual(dm["effective_bits"], 4.20)
            _, dp_metas = load_amf_metadata(dense_pack_amf)
            self.assertFalse(any(m.dtype == DType.Q2_AFFINE for m in dp_metas))

            dense_gsq_amf = tmp / "dense_gsq_auto.amf"
            buf_dg = io.StringIO()
            with redirect_stdout(buf_dg):
                rc_dg = main([
                    "gsq-convert",
                    "--input", str(dense_dir),
                    "--output", str(dense_gsq_amf),
                ])
            self.assertEqual(rc_dg, 0)
            _, dg_metas = load_amf_metadata(dense_gsq_amf)
            self.assertFalse(any(m.dtype == DType.Q2_AFFINE for m in dg_metas))

            # 3. `pack-amf --quant-profile gsq-rco` also blocks 3-bit -> q2_affine without `--allow-retranscode-lowbit`,
            #    and MoE auto `target_bits` resolves to 3.50 with CLI default `max_q2_share=0.55`.
            with self.assertRaises(ValueError):
                main([
                    "pack-amf",
                    "--input", str(q3_dir),
                    "--output", str(tmp / "pack_blocked.amf"),
                    "--quant-profile", "gsq-rco",
                    "--target-bits", "2.95",
                ])

            moe_dir = tmp / "moe_auto"
            _build_synthetic_bf16_qwen3_checkpoint(moe_dir, is_moe=True, num_layers=2, num_experts=8)
            moe_auto_amf = tmp / "moe_auto.amf"
            moe_manifest = tmp / "moe_auto_manifest.json"
            buf_ma = io.StringIO()
            with redirect_stdout(buf_ma):
                rc_ma = main([
                    "gsq-convert",
                    "--input", str(moe_dir),
                    "--output", str(moe_auto_amf),
                    "--manifest", str(moe_manifest),
                ])
            self.assertEqual(rc_ma, 0)
            mm = json.loads(moe_manifest.read_text(encoding="utf-8"))
            self.assertAlmostEqual(mm["target_bits"], 3.50, places=2)
            self.assertLessEqual(mm["effective_bits"], 3.55)
            self.assertLessEqual(mm["dtype_weight_share"].get("q2_affine", 0.0), 0.55 + 1e-6)

    def test_fine_rungs_dense_and_moe_verify_and_decode_parity(self) -> None:
        self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            for is_moe, label, t_bits in ((False, "dense_fine", "4.60"), (True, "moe_fine", "3.60")):
                sf_dir = tmp / label
                _, cfg_path, tok_path = _build_synthetic_bf16_qwen3_checkpoint(
                    sf_dir,
                    is_moe=is_moe,
                    seed=909 if is_moe else 808,
                    num_layers=2,
                    num_experts=8,
                    top_k=2,
                )
                out_amf = tmp / f"{label}.amf"
                manifest_p = tmp / f"{label}_manifest.json"
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main([
                        "gsq-convert",
                        "--input", str(sf_dir),
                        "--output", str(out_amf),
                        "--target-bits", t_bits,
                        "--fine-rungs",
                        "--manifest", str(manifest_p),
                    ])
                self.assertEqual(rc, 0)
                _, metas = load_amf_metadata(out_amf)
                # Confirm fine rungs actually surfaced g=32 or Q5/Q6/Q8_AFFINE!
                fine_hits = [
                    m for m in metas
                    if m.dtype in (DType.Q5_AFFINE, DType.Q6_AFFINE, DType.Q8_AFFINE) or m.group_size == 32
                ]
                self.assertGreater(len(fine_hits), 0, msg=f"{label} did not use any fine rungs")

                if is_moe:
                    for l in range(2):
                        layer_schemes = {
                            (m.dtype, m.group_size)
                            for m in metas
                            if m.layer == l and m.kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT)
                        }
                        self.assertLessEqual(len(layer_schemes), 5, msg=f"layer {l} exceeded NUM_SCHEME_BUCKETS=5")

                v_proc = subprocess.run([str(AMINFER_BIN), "verify-amf", str(out_amf)], capture_output=True, text=True)
                self.assertEqual(v_proc.returncode, 0, msg=v_proc.stdout + v_proc.stderr)
                self.assertIn("amf.status       ok", v_proc.stdout)

                d_proc = subprocess.run(
                    [
                        str(AMINFER_BIN),
                        "decode",
                        "--amf", str(out_amf),
                        "--config", str(cfg_path),
                        "--tokenizer", str(tok_path),
                        "--prompt", "",
                        "--max-tokens", "4",
                        "--ignore-eos",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self.assertEqual(d_proc.returncode, 0, msg=f"{label} decode failed:\n{d_proc.stdout}\n{d_proc.stderr}")
                self.assertIn("decode.swapouts_delta            0", d_proc.stdout)
                m_toks = re.search(r"decode\.token_ids\s+\[([^\]]*)\]", d_proc.stdout)
                self.assertIsNotNone(m_toks, msg=d_proc.stdout)
                assert m_toks is not None
                zig_tokens = [int(x.strip()) for x in m_toks.group(1).split(",") if x.strip()]
                cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
                bos_id = int(cfg_dict.get("bos_token_id", 0))
                ref_tokens = _cpu_fp32_qwen3_greedy(out_amf, cfg_dict, [bos_id], max_new=4)
                self.assertEqual(zig_tokens, ref_tokens, msg=f"{label} tokens {zig_tokens} != {ref_tokens}")

    def test_w8_router_trace_from_decode_consumed_by_expert_heat_no_warning(self) -> None:
        self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            sf_dir = tmp / "moe_src"
            _, cfg_path, tok_path = _build_synthetic_bf16_qwen3_checkpoint(
                sf_dir, is_moe=True, seed=5150, num_layers=2, num_experts=8, top_k=2
            )
            amf_path = tmp / "moe.amf"
            pack_amf_from_safetensors(sf_dir, amf_path, model_id="w8-trace-moe")

            # 1. Shallow `expert-heat` without `--calibration-jsonl` emits `WARNING:` on stderr.
            heat_shallow = tmp / "heat_shallow.json"
            out1, err1 = io.StringIO(), io.StringIO()
            with redirect_stdout(out1), redirect_stderr(err1):
                rc1 = main([
                    "expert-heat",
                    "--input", str(amf_path),
                    "--output", str(heat_shallow),
                ])
            self.assertEqual(rc1, 0)
            self.assertIn("WARNING:", err1.getvalue())
            self.assertIn("--dump-router-trace", err1.getvalue())
            self.assertIn("--calibration-jsonl", err1.getvalue())

            # 2. Capture real router trace via `aminfer decode --dump-router-trace`.
            trace_jsonl = tmp / "router_trace.jsonl"
            d_proc = subprocess.run(
                [
                    str(AMINFER_BIN),
                    "decode",
                    "--amf", str(amf_path),
                    "--config", str(cfg_path),
                    "--tokenizer", str(tok_path),
                    "--prompt", "",
                    "--max-tokens", "4",
                    "--ignore-eos",
                    "--dump-router-trace", str(trace_jsonl),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(d_proc.returncode, 0, msg=d_proc.stdout + d_proc.stderr)
            self.assertTrue(trace_jsonl.is_file())
            self.assertGreater(len(trace_jsonl.read_text(encoding="utf-8").splitlines()), 0)

            # 3. `expert-heat --calibration-jsonl <trace.jsonl>` yields `calibration_mode == "router_trace"` with NO warning!
            heat_trace = tmp / "heat_trace.json"
            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                rc2 = main([
                    "expert-heat",
                    "--input", str(amf_path),
                    "--calibration-jsonl", str(trace_jsonl),
                    "--output", str(heat_trace),
                ])
            self.assertEqual(rc2, 0)
            self.assertEqual(err2.getvalue(), "")
            heat_doc = json.loads(heat_trace.read_text(encoding="utf-8"))
            self.assertEqual(heat_doc["calibration_mode"], "router_trace")

    def test_weighted_rco_degenerate_imatrix_entries_and_nonfinite_target_bits(self) -> None:
        rng = random.Random(777)
        grp = [rng.uniform(-0.2, 0.2) for _ in range(60)] + [2.0, -2.0, 1.8, -1.7]
        s_uw, b_uw, c_uw, mse_uw = quantize_group_rco(grp, 2, iters=3)

        # Degenerate all-zero / negative / NaN / inf weight vectors must fall back cleanly to unweighted RCO.
        for bad_w in (
            [0.0] * 64,
            [-5.0] * 64,
            [float("nan")] * 64,
            [float("inf")] * 64,
            [float("-inf")] * 64,
            [0.0, -1.0, float("nan"), float("inf")] * 16,
        ):
            s_d, b_d, c_d, mse_d = quantize_group_rco(grp, 2, iters=3, weights=bad_w)
            self.assertEqual((s_d, b_d, c_d), (s_uw, b_uw, c_uw))
            self.assertAlmostEqual(mse_d, mse_uw, places=12)

        # Mixed finite + NaN/inf/negative/huge weights must remain finite and satisfy rco_wmse <= minmax_wmse.
        mixed_w = [1e308] * 8 + [1e307] * 8 + [float("nan"), float("inf"), float("-inf"), -3.0] * 12
        _, _, rco_mse, mm_mse = quantize_rows_affine(
            [grp, grp], 2, group_size=64, use_rco=True, compute_minmax_mse=True, col_weights=mixed_w
        )
        self.assertTrue(math.isfinite(rco_mse) and math.isfinite(mm_mse))
        self.assertLessEqual(rco_mse, mm_mse + 1e-15)

        # `_imatrix_tensor_multipliers` must ignore non-finite values without poisoning valid tensors.
        mults = _imatrix_tensor_multipliers(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [2.0, 4.0, 6.0],
                "bad_inf": [float("inf"), 1.0],
                "bad_nan": [float("nan"), float("-inf")],
                "huge": [1e308, 1e308],
            }
        )
        self.assertNotIn("bad_nan", mults)
        for k, v in mults.items():
            self.assertTrue(math.isfinite(v) and 0.5 <= v <= 4.0, msg=f"{k}={v}")
        self.assertAlmostEqual(mults["b"] / mults["a"], 2.0, places=6)

        # Non-finite target_bits must raise ValueError cleanly in both solver and gsq_convert.
        u = QuantUnitInput(
            unit_id="x",
            tensor_names=("x",),
            role="dense_ffn",
            layer=0,
            expert=0,
            numel=64 * 64,
            sensitivity=1.0,
            candidates=(QuantUnitCandidate(DType.Q4_AFFINE, (64 * 64 * 36) // 64, 1.0, 0.01, 64),),
        )
        for bad_tb in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
            with self.assertRaises(ValueError):
                solve_gsq_allocation([u], target_bits=bad_tb)
            with self.assertRaises(ValueError):
                gsq_convert(pathlib.Path("/nonexistent.amf"), pathlib.Path("/tmp/out.amf"), target_bits=bad_tb)

    def test_moe_stacked_3d_imatrix_matching_and_shape_mismatch_safety(self) -> None:
        rng = random.Random(202604)

        def _write_min_gguf_imatrix(
            path: pathlib.Path,
            tensors: list[tuple[str, tuple[int, ...], bytes]],
        ) -> None:
            align = 32
            buf = bytearray(b"GGUF")
            buf.extend(struct.pack("<IQQ", 3, len(tensors), 0))
            data = bytearray()
            offs: list[int] = []
            for _n, _d, payload in tensors:
                pad = (align - (len(data) % align)) % align
                data.extend(b"\x00" * pad)
                offs.append(len(data))
                data.extend(payload)
            for (name, dims, _), rel in zip(tensors, offs):
                nb = name.encode("utf-8")
                buf.extend(struct.pack("<Q", len(nb)) + nb)
                buf.extend(struct.pack("<I", len(dims)))
                for d in dims:
                    buf.extend(struct.pack("<Q", d))
                buf.extend(struct.pack("<IQ", 0, rel))  # GGML_TYPE_F32 = 0
            pad_hdr = (align - (len(buf) % align)) % align
            buf.extend(b"\x00" * pad_hdr)
            buf.extend(data)
            path.write_bytes(bytes(buf))

        def _bf16_rows(mat: list[list[float]]) -> bytes:
            flat = [v for r in mat for v in r]
            f32 = struct.pack(f"<{len(flat)}f", *flat)
            out = bytearray(len(flat) * 2)
            out[0::2] = f32[2::4]
            out[1::2] = f32[3::4]
            return bytes(out)

        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)

            # 1. GGUF stacked 3D MoE imatrix with per-expert `.counts` (4 experts x 64 cols) + shared broadcast (64 cols).
            e0_w = [200.0] * 16 + [2e-4] * 48  # divided by count=2 -> [100.0]*16 + [1e-4]*48
            e1_w = [4e-4] * 16 + [400.0] * 16 + [4e-4] * 32  # divided by count=4 -> [1e-4]*16 + [100.0]*16 + [1e-4]*32
            e2_w = [0.0] * 64  # cold/unrouted expert, count=0 -> falls back safely to unweighted
            e3_w = [1e-4] * 32 + [100.0] * 16 + [1e-4] * 16  # count=1
            stacked_gate = e0_w + e1_w + e2_w + e3_w
            shared_up = [50.0] * 16 + [1e-4] * 48

            gguf_im = tmp / "moe_imatrix.gguf"
            _write_min_gguf_imatrix(
                gguf_im,
                [
                    ("blk.0.ffn_gate_exps.weight.in_sum2", (64, 4), struct.pack(f"<{len(stacked_gate)}f", *stacked_gate)),
                    ("blk.0.ffn_gate_exps.weight.counts", (4,), struct.pack("<4f", 2.0, 4.0, 0.0, 1.0)),
                    ("blk.0.ffn_up_exps.weight.in_sum2", (64,), struct.pack("<64f", *shared_up)),
                    ("blk.0.ffn_up_exps.weight.counts", (1,), struct.pack("<1f", 1.0)),
                ],
            )
            parsed = parse_gguf_imatrix(gguf_im)
            cw0 = _imatrix_col_weights(parsed, "layers.0.mlp.experts.0.gate_proj.weight", 64, layer=0, expert=0)
            cw1 = _imatrix_col_weights(parsed, "layers.0.mlp.experts.1.gate_proj", 64, layer=0, expert=1)
            cw_up = _imatrix_col_weights(parsed, "layers.0.mlp.experts.3.up_proj", 64, layer=0, expert=3)
            self.assertIsNotNone(cw0)
            self.assertIsNotNone(cw1)
            self.assertIsNotNone(cw_up)
            assert cw0 is not None and cw1 is not None and cw_up is not None
            self.assertAlmostEqual(cw0[0], 100.0, places=3)
            self.assertAlmostEqual(cw0[20], 1e-4, places=6)
            self.assertAlmostEqual(cw1[0], 1e-4, places=6)
            self.assertAlmostEqual(cw1[20], 100.0, places=3)
            self.assertAlmostEqual(cw_up[0], 50.0, places=3)

            # Shape mismatches and out-of-bounds expert indices must safely return None.
            self.assertIsNone(_imatrix_col_weights(parsed, "layers.0.mlp.experts.4.gate_proj", 64, layer=0, expert=4))
            self.assertIsNone(_imatrix_col_weights(parsed, "layers.0.mlp.experts.0.gate_proj", 63, layer=0, expert=0))
            self.assertIsNone(_imatrix_col_weights({"layers.0.mlp.gate_proj": [1.0] * 128}, "layers.0.mlp.gate_proj", 64))
            self.assertIsNotNone(
                _imatrix_col_weights(
                    {"model.layers.0.mlp.switch_mlp.down_proj.weight": [2.0] * 64},
                    "layers.0.mlp.experts.2.down_proj",
                    64,
                    layer=0,
                    expert=2,
                )
            )

            # 2. End-to-end MoE `gsq_convert` with GGUF stacked 3D imatrix must lower per-expert weighted MSE.
            sf_dir = tmp / "moe_im_ckpt"
            sf_dir.mkdir()
            ones = _bf16_rows([[1.0] * 64])
            tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
                "model.embed_tokens.weight": ("BF16", (64, 64), _rand_bf16_bytes(rng, 64 * 64, 0.1)),
                "model.layers.0.input_layernorm.weight": ("BF16", (64,), ones),
                "model.layers.0.post_attention_layernorm.weight": ("BF16", (64,), ones),
                "model.layers.0.self_attn.q_proj.weight": ("BF16", (64, 64), _rand_bf16_bytes(rng, 64 * 64, 0.1)),
                "model.layers.0.mlp.gate.weight": ("BF16", (4, 64), _rand_bf16_bytes(rng, 4 * 64, 0.1)),
                "model.norm.weight": ("BF16", (64,), ones),
            }
            exp_mats: dict[int, list[list[float]]] = {}
            for e in range(4):
                # Put fine structure on expert-specific active cols and outliers on low-weight cols.
                mat = [
                    [rng.uniform(-0.12, 0.12) for _ in range(48)] + [2.2, -2.1, 1.9, -1.8] * 4
                    for _ in range(128)
                ]
                exp_mats[e] = mat
                tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = ("BF16", (128, 64), _bf16_rows(mat))
                tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = ("BF16", (128, 64), _rand_bf16_bytes(rng, 128 * 64, 0.1))
                tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = ("BF16", (64, 128), _rand_bf16_bytes(rng, 64 * 128, 0.1))
            _write_safetensors(sf_dir / "model.safetensors", tensors)
            (sf_dir / "config.json").write_text(json.dumps({"tie_word_embeddings": True}), encoding="utf-8")

            amf_uw = tmp / "moe_uw.amf"
            amf_w = tmp / "moe_w.amf"
            gsq_convert(sf_dir, amf_uw, target_bits=2.75, model_id="moe-uw")
            gsq_convert(sf_dir, amf_w, target_bits=2.75, model_id="moe-w", imatrix_path=gguf_im)

            with AmfReader(amf_uw) as r_uw, AmfReader(amf_w) as r_w:
                for e, active_slice in ((0, range(0, 16)), (1, range(16, 32))):
                    name = f"layers.0.mlp.experts.{e}.gate_proj"
                    m_uw = r_uw.by_name[name]
                    m_w = r_w.by_name[name]
                    self.assertEqual(m_uw.dtype, DType.Q2_AFFINE)
                    self.assertEqual(m_w.dtype, DType.Q2_AFFINE)
                    mat_uw = r_uw.read_matrix_f32(m_uw)
                    mat_w = r_w.read_matrix_f32(m_w)
                    err_uw = sum((mat_uw[r][c] - exp_mats[e][r][c]) ** 2 for r in range(128) for c in active_slice)
                    err_w = sum((mat_w[r][c] - exp_mats[e][r][c]) ** 2 for r in range(128) for c in active_slice)
                    self.assertLess(err_w, err_uw * 0.5, msg=f"expert {e} did not benefit from stacked GGUF imatrix")

            # 3. `_is_moe_checkpoint_input` detects HF 3D stacked `mlp.experts.gate_up_proj` checkpoints.
            hf_3d_dir = tmp / "hf_3d_moe"
            hf_3d_dir.mkdir()
            _write_safetensors(
                hf_3d_dir / "model.safetensors",
                {
                    "model.layers.0.mlp.experts.gate_up_proj.weight": (
                        "BF16",
                        (4, 128, 64),
                        _rand_bf16_bytes(rng, 4 * 128 * 64, 0.1),
                    ),
                },
            )
            self.assertTrue(_is_moe_checkpoint_input(hf_3d_dir))

    def test_fine_rungs_with_lut16_sme2_strict_gating_and_moe_uniform_layout_decode_parity(self) -> None:
        self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cases = (
                (False, "dense_fine_lut16", "4.60", True, False),
                (True, "moe_fine_lut16_mixed_q4_q6", "5.00", False, True),
                (True, "moe_fine_lut16_all_sme2", "3.60", True, False),
            )
            for is_moe, label, t_bits, expect_lut16, expect_mixed_non_sme2_moe in cases:
                sf_dir = tmp / label
                _, cfg_path, tok_path = _build_synthetic_bf16_qwen3_checkpoint(
                    sf_dir,
                    is_moe=is_moe,
                    seed=919 if is_moe else 818,
                    num_layers=2,
                    num_experts=8,
                    top_k=2,
                )
                out_amf = tmp / f"{label}.amf"
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main([
                        "gsq-convert",
                        "--input", str(sf_dir),
                        "--output", str(out_amf),
                        "--target-bits", t_bits,
                        "--fine-rungs",
                        "--lut16-sme2",
                    ])
                self.assertEqual(rc, 0)
                _, metas = load_amf_metadata(out_amf)

                # Strict gating: LUT16_CODEBOOK is ONLY emitted on Q2/Q3/Q4_AFFINE with group_size == 64!
                lut_metas = [m for m in metas if m.quant_param_kind == QuantParamKind.LUT16_CODEBOOK]
                if expect_lut16:
                    self.assertGreater(len(lut_metas), 0, msg=f"{label} expected at least one lut16 tensor")
                for m in lut_metas:
                    self.assertIn(m.dtype, (DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE))
                    self.assertEqual(m.group_size, 64)
                    self.assertEqual(m.layout, TensorLayout.SME2_PANEL64)

                # Every MoE layer must keep uniform TensorLayout across all its experts (matching `qwen3.zig`).
                if is_moe:
                    saw_non_sme2_exp = False
                    for l in range(2):
                        exp_l = [
                            m for m in metas
                            if m.layer == l and m.kind in (TensorKind.HOT_EXPERT, TensorKind.COLD_EXPERT)
                        ]
                        layouts = {m.layout for m in exp_l}
                        self.assertEqual(len(layouts), 1, msg=f"{label} layer {l} mixed expert layouts {layouts}")
                        if any(m.dtype in (DType.Q5_AFFINE, DType.Q6_AFFINE, DType.Q8_AFFINE) or m.group_size == 32 for m in exp_l):
                            saw_non_sme2_exp = True
                            self.assertEqual(layouts, {TensorLayout.ROW_MAJOR})
                    if expect_mixed_non_sme2_moe:
                        self.assertTrue(saw_non_sme2_exp, msg=f"{label} did not mix non-SME2 fine rungs into MoE layers")

                v_proc = subprocess.run([str(AMINFER_BIN), "verify-amf", str(out_amf)], capture_output=True, text=True)
                self.assertEqual(v_proc.returncode, 0, msg=v_proc.stdout + v_proc.stderr)
                self.assertIn("amf.status       ok", v_proc.stdout)

                d_proc = subprocess.run(
                    [
                        str(AMINFER_BIN),
                        "decode",
                        "--amf", str(out_amf),
                        "--config", str(cfg_path),
                        "--tokenizer", str(tok_path),
                        "--prompt", "",
                        "--max-tokens", "4",
                        "--ignore-eos",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self.assertEqual(d_proc.returncode, 0, msg=f"{label} decode failed:\n{d_proc.stdout}\n{d_proc.stderr}")
                self.assertIn("decode.swapouts_delta            0", d_proc.stdout)
                m_toks = re.search(r"decode\.token_ids\s+\[([^\]]*)\]", d_proc.stdout)
                self.assertIsNotNone(m_toks, msg=d_proc.stdout)
                assert m_toks is not None
                zig_tokens = [int(x.strip()) for x in m_toks.group(1).split(",") if x.strip()]
                cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
                bos_id = int(cfg_dict.get("bos_token_id", 0))
                ref_tokens = _cpu_fp32_qwen3_greedy(out_amf, cfg_dict, [bos_id], max_new=4)
                self.assertEqual(zig_tokens, ref_tokens, msg=f"{label} tokens {zig_tokens} != {ref_tokens}")


if __name__ == "__main__":
    unittest.main()
