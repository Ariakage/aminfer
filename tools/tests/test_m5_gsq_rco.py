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
from contextlib import redirect_stdout

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    AmfReader,
    DType,
    TensorKind,
    TensorLayout,
    dequantize_q2_row_f32,
    dequantize_sme2_panel64_q2_panel_f32,
    load_amf_metadata,
    pack_amf_from_safetensors,
    pack_mlx_affine_2bit,
    pack_sme2_panel64_q2_from_mlx_panel,
    repack_amf_sme2,
    sme2_panel64_data_bytes,
    write_amf,
)
from aminfer_tools.cli import main
from aminfer_tools.gsq_rco import (
    QuantUnitCandidate,
    QuantUnitInput,
    gsq_convert,
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


if __name__ == "__main__":
    unittest.main()
