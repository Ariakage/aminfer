"""Unit and integration tests for M6 tools (`W10`, `W16`, `W17`):

- `TensorKind` `7..9` (`SHARED_EXPERT`, `SPARSE_TABLE`, `LOOP_ADAPTER`) & `QuantParamKind.LUT16_CODEBOOK` (`0x0101`)
- `qwen4_exp` (`Qwen3.8-Flash-Next`) & `LoopLM` / `Step-LoRA` offline checkpoint converters
- `synth-qwen4` (`tiny-qwen4`, `scale-qwen4-512e`) & `synth-loop` (`tiny-looplm`, `tiny-steplora`)
- Heat-guided large-MoE GSQ+RCO on already-quantized MLX 3D `switch_mlp` `q3_affine` (lossless hot passthrough + SWAR 8-bin OLS `q3->q2`)
- Weighted LBG Lloyd-Max `LUT16` `ZT0` codebook packer (`gsq-convert --lut16-sme2` & `repack-sme2 --lut16`)
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import struct
import subprocess
import tempfile
import unittest

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    ENTRY_STRUCT,
    NO_EXPERT,
    SME2_Q2_GROUP_BYTES,
    SME2_Q3_GROUP_BYTES,
    SME2_Q4_GROUP_BYTES,
    AmfReader,
    DType,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    dequantize_q2_row_f32,
    dequantize_q3_row_f32,
    dequantize_sme2_panel64_q2_panel_f32,
    dequantize_sme2_panel64_q3_panel_f32,
    dequantize_sme2_panel64_q4_panel_f32,
    encode_reserved0,
    load_amf_metadata,
    pack_amf_from_safetensors,
    pack_mlx_affine_3bit,
    repack_amf_sme2,
    write_amf,
)
from aminfer_tools.cli import main as cli_main
from aminfer_tools.gsq_rco import (
    fit_lut16_panel_group,
    gsq_convert,
    pack_lut16_panel_sme2,
    quantize_q3_blob_to_q2_rco,
    quantize_rows_affine,
)
from aminfer_tools.qwen4_exp import pack_qwen4_checkpoint
from aminfer_tools.synth import (
    LOOP_PRESETS,
    QWEN4_PRESETS,
    generate_loop_bundle,
    generate_qwen4_bundle,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header: dict[str, object] = {}
    cursor = 0
    chunks: list[bytes] = []
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(payload)],
        }
        chunks.append(payload)
        cursor += len(payload)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for ch in chunks:
            fh.write(ch)


def _run_verify_amf(amf_path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(AMINFER_BIN), "verify-amf", str(amf_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _f16_bytes(vals: list[float]) -> bytes:
    return struct.pack(f"<{len(vals)}e", *vals)


def _bf16_bytes(vals: list[float]) -> bytes:
    f32 = struct.pack(f"<{len(vals)}f", *vals)
    out = bytearray(len(vals) * 2)
    out[0::2] = f32[2::4]
    out[1::2] = f32[3::4]
    return bytes(out)


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class M6ToolsTests(unittest.TestCase):
    def test_tensor_kind_and_lut16_binary_contract_matches_zig_abi(self) -> None:
        # TensorKind 0..6 unchanged; 7..9 match src/format/amf.zig M6 contract.
        self.assertEqual(int(TensorKind.DENSE), 0)
        self.assertEqual(int(TensorKind.HOT_EXPERT), 1)
        self.assertEqual(int(TensorKind.COLD_EXPERT), 2)
        self.assertEqual(int(TensorKind.EMBEDDING), 3)
        self.assertEqual(int(TensorKind.NORM), 4)
        self.assertEqual(int(TensorKind.ROUTER), 5)
        self.assertEqual(int(TensorKind.LM_HEAD), 6)
        self.assertEqual(int(TensorKind.SHARED_EXPERT), 7)
        self.assertEqual(int(TensorKind.SPARSE_TABLE), 8)
        self.assertEqual(int(TensorKind.LOOP_ADAPTER), 9)
        self.assertEqual(int(QuantParamKind.LUT16_CODEBOOK), 1)
        self.assertEqual(
            encode_reserved0(TensorLayout.SME2_PANEL64, QuantParamKind.LUT16_CODEBOOK),
            0x0101,
        )

        # Verify bit-for-bit LUT16 header layout (matches src/cpu/sme.zig:packQuantLut16Panel64)
        # across q2 (384 B), q3 (512 B), and q4 (640 B), including tail-row padding (n=13 < 16).
        rng = random.Random(101)
        n_rows, k = 13, 128
        mat = [[rng.gauss(0.0, 0.25) + 0.05 * (r % 5) for _ in range(k)] for r in range(n_rows)]

        for bits, dtype, grp_bytes, deq_fn in (
            (2, DType.Q2_AFFINE, SME2_Q2_GROUP_BYTES, dequantize_sme2_panel64_q2_panel_f32),
            (3, DType.Q3_AFFINE, SME2_Q3_GROUP_BYTES, dequantize_sme2_panel64_q3_panel_f32),
            (4, DType.Q4_AFFINE, SME2_Q4_GROUP_BYTES, dequantize_sme2_panel64_q4_panel_f32),
        ):
            panel_bytes, fit_mse = pack_lut16_panel_sme2(mat, bits, lloyd_iters=12)
            groups = k // 64
            self.assertEqual(len(panel_bytes), groups * grp_bytes)
            L = 1 << bits

            for g in range(groups):
                hdr = panel_bytes[g * grp_bytes : g * grp_bytes + 128]
                # Bytes 0..64: 16 f16 codebook entries, each u16 written TWICE.
                cb_u16 = struct.unpack_from("<32H", hdr, 0)
                for idx in range(16):
                    self.assertEqual(cb_u16[2 * idx], cb_u16[2 * idx + 1])
                cb_f16 = struct.unpack_from("<32e", hdr, 0)
                for idx in range(L, 16):
                    self.assertEqual(cb_f16[2 * idx], 0.0)
                # Bytes 64..128: 16 per-column f16 biases, each u16 written TWICE; padded cols zero.
                bias_u16 = struct.unpack_from("<32H", hdr, 64)
                for c in range(16):
                    self.assertEqual(bias_u16[2 * c], bias_u16[2 * c + 1])
                bias_f16 = struct.unpack_from("<32e", hdr, 64)
                for c in range(n_rows, 16):
                    self.assertEqual(bias_f16[2 * c], 0.0)

            deq = deq_fn(panel_bytes, k, num_rows=n_rows, param_kind=QuantParamKind.LUT16_CODEBOOK)
            ref_mse = sum(
                (a - b) ** 2 for ra, rb in zip(mat, deq) for a, b in zip(ra, rb)
            ) / (n_rows * k)
            self.assertAlmostEqual(fit_mse, ref_mse, places=10)

            # Write to .amf and verify with Zig `aminfer verify-amf`!
            with tempfile.TemporaryDirectory() as tmpdir:
                amf_path = pathlib.Path(tmpdir) / f"lut16_q{bits}.amf"
                write_amf(
                    amf_path,
                    f"lut16-q{bits}-abi",
                    [
                        (
                            "layers.0.self_attn.q_proj",
                            TensorKind.DENSE,
                            dtype,
                            bits,
                            64,
                            0,
                            NO_EXPERT,
                            (n_rows, k),
                            panel_bytes,
                            b"",
                            TensorLayout.SME2_PANEL64,
                            QuantParamKind.LUT16_CODEBOOK,
                        )
                    ],
                )
                # Check raw reserved0 == 0x0101 at directory entry byte offset 58.
                raw = amf_path.read_bytes()
                res0 = struct.unpack_from("<H", raw, APPLE_PAGE_SIZE + 58)[0]
                self.assertEqual(res0, 0x0101)
                self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
                proc = _run_verify_amf(amf_path)
                self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
                self.assertIn("amf.status       ok", proc.stdout)

    def test_lut16_lloyd_max_beats_uniform_affine_and_weighted_prioritizes_cols(self) -> None:
        # Bimodal / clustered non-uniform weights where uniform affine wastes levels in the empty gap.
        rng = random.Random(2026)
        tile = [
            [
                (rng.gauss(-0.8, 0.08) if rng.random() < 0.5 else rng.gauss(0.8, 0.08))
                + 0.03 * (c % 4)
                for _ in range(64)
            ]
            for c in range(16)
        ]
        for bits in (2, 3, 4):
            _, _, aff_rco_mse, aff_mm_mse = quantize_rows_affine(
                tile, bits, group_size=64, use_rco=True, compute_minmax_mse=True
            )
            _, _, _, lut_mse = fit_lut16_panel_group(tile, bits, lloyd_iters=14, bias_iters=2)
            self.assertLess(lut_mse, aff_mm_mse, msg=f"q{bits} LUT16 must beat min-max uniform affine")
            self.assertLess(lut_mse, aff_rco_mse, msg=f"q{bits} LUT16 must beat RCO uniform affine on bimodal weights")

        # Weighted Lloyd-Max: emphasize first 4 columns 16x -> their MSE decreases vs unweighted.
        tile_w = [
            [rng.gauss(0.0, 0.05 if c < 4 else 0.5) for _ in range(64)]
            for c in range(16)
        ]
        cb_u, b_u, codes_u, _ = fit_lut16_panel_group(tile_w, 3, col_weights=None)
        weights = [16.0 if c < 4 else 1.0 for c in range(16)]
        cb_w, b_w, codes_w, _ = fit_lut16_panel_group(tile_w, 3, col_weights=weights)
        err_hi_u = sum((tile_w[c][k] - (cb_u[codes_u[c][k]] + b_u[c])) ** 2 for c in range(4) for k in range(64))
        err_hi_w = sum((tile_w[c][k] - (cb_w[codes_w[c][k]] + b_w[c])) ** 2 for c in range(4) for k in range(64))
        self.assertLess(err_hi_w, err_hi_u)

    def test_repack_sme2_lut16_and_gsq_convert_lut16_cli_roundtrip(self) -> None:
        rng = random.Random(77)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            ckpt = tmp / "ckpt"
            ckpt.mkdir()
            h = 128
            # Bimodal weights so LUT16 strictly beats uniform affine SME2!
            def _bimodal_f16(rows: int, cols: int) -> bytes:
                vals = [
                    (rng.gauss(-0.6, 0.06) if rng.random() < 0.5 else rng.gauss(0.6, 0.06))
                    for _ in range(rows * cols)
                ]
                return _f16_bytes(vals)

            tensors = {
                "model.embed_tokens.weight": ("F16", (256, h), _f16_bytes([rng.gauss(0.0, 0.1) for _ in range(256 * h)])),
                "model.layers.0.input_layernorm.weight": ("BF16", (h,), _bf16_bytes([1.0] * h)),
                "model.layers.0.self_attn.q_proj.weight": ("F16", (h, h), _bimodal_f16(h, h)),
                "model.layers.0.self_attn.k_proj.weight": ("F16", (64, h), _bimodal_f16(64, h)),
                "model.layers.0.self_attn.v_proj.weight": ("F16", (64, h), _bimodal_f16(64, h)),
                "model.layers.0.self_attn.o_proj.weight": ("F16", (h, h), _bimodal_f16(h, h)),
                "model.layers.0.post_attention_layernorm.weight": ("BF16", (h,), _bf16_bytes([1.0] * h)),
                "model.layers.0.mlp.gate_proj.weight": ("F16", (256, h), _bimodal_f16(256, h)),
                "model.layers.0.mlp.up_proj.weight": ("F16", (256, h), _bimodal_f16(256, h)),
                "model.layers.0.mlp.down_proj.weight": ("F16", (h, 256), _bimodal_f16(h, 256)),
                "model.norm.weight": ("BF16", (h,), _bf16_bytes([1.0] * h)),
            }
            _write_safetensors(ckpt / "model.safetensors", tensors)
            (ckpt / "config.json").write_text(json.dumps({"tie_word_embeddings": True}), encoding="utf-8")

            aff_amf = tmp / "aff_sme2.amf"
            lut_amf = tmp / "lut16_sme2.amf"
            repacked_lut = tmp / "repacked_lut16.amf"
            repacked_lut_twice = tmp / "repacked_lut16_twice.amf"

            rc = cli_main([
                "gsq-convert",
                "--input", str(ckpt),
                "--output", str(aff_amf),
                "--target-bits", "3.5",
                "--sme2-pack", "all-linear",
            ])
            self.assertEqual(rc, 0)

            rc = cli_main([
                "gsq-convert",
                "--input", str(ckpt),
                "--output", str(lut_amf),
                "--target-bits", "3.5",
                "--lut16-sme2",
            ])
            self.assertEqual(rc, 0)

            # Verify Zig accepts both files and LUT16 has reserved0 == 0x0101 on all linear layers.
            for p in (aff_amf, lut_amf):
                proc = _run_verify_amf(p)
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertIn("amf.status       ok", proc.stdout)

            _, lut_metas = load_amf_metadata(lut_amf)
            lut_linear = [m for m in lut_metas if m.kind == TensorKind.DENSE]
            self.assertEqual(len(lut_linear), 7)
            for m in lut_linear:
                self.assertEqual(m.layout, TensorLayout.SME2_PANEL64)
                self.assertEqual(m.quant_param_kind, QuantParamKind.LUT16_CODEBOOK)

            # Verify strictly lower MSE for LUT16 vs uniform affine on bimodal weights!
            with AmfReader(aff_amf) as r_aff, AmfReader(lut_amf) as r_lut:
                aff_err = 0.0
                lut_err = 0.0
                n_w = 0
                for m_lut in lut_linear:
                    src_name = f"model.{m_lut.name}.weight"
                    _, shape, raw_f16 = tensors[src_name]
                    orig = list(struct.unpack(f"<{shape[0] * shape[1]}e", raw_f16))
                    aff_mat = r_aff.read_matrix_f32(r_aff.by_name[m_lut.name])
                    lut_mat = r_lut.read_matrix_f32(m_lut)
                    aff_flat = [v for row in aff_mat for v in row]
                    lut_flat = [v for row in lut_mat for v in row]
                    aff_err += sum((a - b) ** 2 for a, b in zip(orig, aff_flat))
                    lut_err += sum((a - b) ** 2 for a, b in zip(orig, lut_flat))
                    n_w += len(orig)
                self.assertLess(lut_err / n_w, aff_err / n_w)

            # Test `repack-sme2 --lut16` from affine SME2 + idempotency.
            rc = cli_main([
                "repack-sme2",
                "--input", str(aff_amf),
                "--output", str(repacked_lut),
                "--targets", "all-linear",
                "--lut16",
            ])
            self.assertEqual(rc, 0)
            rc = cli_main([
                "repack-sme2",
                "--input", str(repacked_lut),
                "--output", str(repacked_lut_twice),
                "--targets", "all-linear",
                "--lut16",
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(_sha256_file(repacked_lut), _sha256_file(repacked_lut_twice))
            proc = _run_verify_amf(repacked_lut)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_q3_switch_mlp_heat_guided_gsq_rco_lossless_hot_and_negative_scales(self) -> None:
        """Stream-import MLX 3D switch_mlp `q3_affine` (including negative scales!) + heat.json."""
        rng = random.Random(314)
        h = 256
        inter = 192  # q3 blob = 21,504 B (2 pages = 32 KiB) vs q2 blob = 15,360 B (1 page = 16 KiB)!
        num_experts = 8
        groups_h = h // 64

        def _make_q3_matrix(rows: int, cols: int, allow_neg_scale: bool = True) -> tuple[bytes, bytes, bytes]:
            groups = cols // 64
            q_buf = bytearray()
            s_vals: list[float] = []
            b_vals: list[float] = []
            for r in range(rows):
                codes = [rng.randint(0, 7) for _ in range(cols)]
                q_buf.extend(pack_mlx_affine_3bit(codes))
                for _ in range(groups):
                    sc = rng.uniform(0.008, 0.04)
                    if allow_neg_scale and rng.random() < 0.5:
                        sc = -sc
                    s_vals.append(sc)
                    b_vals.append(rng.uniform(-0.15, 0.15))
            return bytes(q_buf), _bf16_bytes(s_vals), _bf16_bytes(b_vals)

        # Unit test SWAR 8-bin OLS RCO directly against naive min-max (including negative scales!).
        q3_d, q3_s_bf16, q3_b_bf16 = _make_q3_matrix(32, h, allow_neg_scale=True)
        # Decode bf16 scales/biases to fp32
        def _dec_bf16(buf: bytes) -> list[float]:
            n = len(buf) // 2
            f32 = bytearray(n * 4)
            f32[2::4] = buf[0::2]
            f32[3::4] = buf[1::2]
            return list(struct.unpack(f"<{n}f", f32))

        sv = _dec_bf16(q3_s_bf16)
        bv = _dec_bf16(q3_b_bf16)
        q2_d, q2_p, mse_swar = quantize_q3_blob_to_q2_rco(q3_d, sv, bv, 32, h, compute_mse=True)
        # Reconstruct reference rows from q3_affine and compare MSE vs naive min-max.
        ref_rows: list[list[float]] = []
        row_q3_b = (h * 3) // 8
        for r in range(32):
            p_r = struct.pack(f"<{groups_h * 2}e", *[x for g in range(groups_h) for x in (sv[r * groups_h + g], bv[r * groups_h + g])])
            ref_rows.append(dequantize_q3_row_f32(q3_d[r * row_q3_b : (r + 1) * row_q3_b], p_r, h, group_size=64))
        _, _, mse_ref_rco, mse_ref_mm = quantize_rows_affine(ref_rows, 2, group_size=64, use_rco=True, compute_minmax_mse=True)
        # Exact MSE of swar output vs ref_rows:
        err2 = 0.0
        for r in range(32):
            hat = dequantize_q2_row_f32(q2_d[r * (h // 4) : (r + 1) * (h // 4)], q2_p[r * groups_h * 4 : (r + 1) * groups_h * 4], h, group_size=64)
            err2 += sum((a - b) ** 2 for a, b in zip(ref_rows[r], hat))
        exact_swar_mse = err2 / (32 * h)
        self.assertLess(exact_swar_mse, mse_ref_mm * 0.75)
        self.assertAlmostEqual(mse_swar, exact_swar_mse, delta=1e-6)

        # Full checkpoint conversion test with 3D switch_mlp + heat.json + lossless hot experts!
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            ckpt = tmp / "mlx_3bit_moe"
            ckpt.mkdir()
            tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

            def _add_q3_2d(prefix: str, rows: int, cols: int) -> None:
                wd, sd, bd = _make_q3_matrix(rows, cols)
                tensors[f"{prefix}.weight"] = ("U32", (rows, (cols * 3) // 32), wd)
                tensors[f"{prefix}.scales"] = ("BF16", (rows, cols // 64), sd)
                tensors[f"{prefix}.biases"] = ("BF16", (rows, cols // 64), bd)

            _add_q3_2d("model.embed_tokens", 128, h)
            _add_q3_2d("lm_head", 128, h)
            tensors["model.norm.weight"] = ("BF16", (h,), _bf16_bytes([1.0] * h))

            for l in range(2):
                tensors[f"model.layers.{l}.input_layernorm.weight"] = ("BF16", (h,), _bf16_bytes([1.0] * h))
                tensors[f"model.layers.{l}.post_attention_layernorm.weight"] = ("BF16", (h,), _bf16_bytes([1.0] * h))
                _add_q3_2d(f"model.layers.{l}.self_attn.q_proj", h, h)
                _add_q3_2d(f"model.layers.{l}.self_attn.k_proj", 64, h)
                _add_q3_2d(f"model.layers.{l}.self_attn.v_proj", 64, h)
                _add_q3_2d(f"model.layers.{l}.self_attn.o_proj", h, h)
                _add_q3_2d(f"model.layers.{l}.mlp.gate", num_experts, h)

                for proj, r_dim, c_dim in (("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)):
                    w_all, s_all, b_all = bytearray(), bytearray(), bytearray()
                    for _ in range(num_experts):
                        wd, sd, bd = _make_q3_matrix(r_dim, c_dim)
                        w_all.extend(wd)
                        s_all.extend(sd)
                        b_all.extend(bd)
                    tensors[f"model.layers.{l}.mlp.switch_mlp.{proj}.weight"] = ("U32", (num_experts, r_dim, (c_dim * 3) // 32), bytes(w_all))
                    tensors[f"model.layers.{l}.mlp.switch_mlp.{proj}.scales"] = ("BF16", (num_experts, r_dim, c_dim // 64), bytes(s_all))
                    tensors[f"model.layers.{l}.mlp.switch_mlp.{proj}.biases"] = ("BF16", (num_experts, r_dim, c_dim // 64), bytes(b_all))

            _write_safetensors(ckpt / "model.safetensors", tensors)
            (ckpt / "config.json").write_text(json.dumps({"tie_word_embeddings": False}), encoding="utf-8")

            # Heat profile: experts 0, 1 are very hot; experts 2..7 are cold.
            heat_path = tmp / "heat.json"
            heat_doc = {
                "schema_version": 1,
                "num_layers": 2,
                "num_experts": num_experts,
                "layers": [
                    {
                        "layer": l,
                        "counts": [200, 180, 10, 5, 2, 1, 0, 0],
                        "frequencies": [0.50, 0.45, 0.025, 0.0125, 0.005, 0.0025, 0.0, 0.0],
                        "ranked_hot_to_cold": [0, 1, 2, 3, 4, 5, 6, 7],
                        "cold_experts": [4, 5, 6, 7],
                    }
                    for l in range(2)
                ],
            }
            heat_path.write_text(json.dumps(heat_doc), encoding="utf-8")

            ref_amf = tmp / "uniform_3bit.amf"
            pack_amf_from_safetensors(ckpt, ref_amf, model_id="ref-3bit", heat_profile=heat_path, cold_experts_per_layer=0)

            gsq_amf = tmp / "gsq_rco.amf"
            rep = gsq_convert(
                ckpt,
                gsq_amf,
                target_bits=2.95,
                model_id="gsq-3bit-2bit",
                heat_profile=heat_path,
                cold_experts_per_layer=0,
                workers=2,
            )
            self.assertLessEqual(rep.effective_bits, 3.00)
            self.assertLess(gsq_amf.stat().st_size, ref_amf.stat().st_size)

            proc = _run_verify_amf(gsq_amf)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("amf.status       ok", proc.stdout)

            # Check protected tensors + hot expert 0 stayed lossless q3_affine (exact byte match vs ref_amf)!
            # Check cold expert 7 compressed to q2_affine!
            with AmfReader(ref_amf) as r_ref, AmfReader(gsq_amf) as r_gsq:
                for protected in (
                    "embed_tokens",
                    "lm_head",
                    "layers.0.self_attn.q_proj",
                    "layers.0.mlp.gate",
                    "layers.0.mlp.experts.0.gate_proj",
                    "layers.0.mlp.experts.0.up_proj",
                    "layers.0.mlp.experts.0.down_proj",
                ):
                    m_ref = r_ref.by_name[protected]
                    m_gsq = r_gsq.by_name[protected]
                    self.assertEqual(m_gsq.dtype, DType.Q3_AFFINE, msg=f"{protected} must stay q3_affine")
                    self.assertEqual(
                        r_gsq.read_blob(m_gsq, verify_crc=True),
                        r_ref.read_blob(m_ref, verify_crc=True),
                        msg=f"lossless passthrough blob mismatch for {protected}",
                    )
                for cold_proj in ("gate_proj", "up_proj", "down_proj"):
                    self.assertEqual(r_gsq.by_name[f"layers.0.mlp.experts.7.{cold_proj}"].dtype, DType.Q2_AFFINE)

    def test_qwen4_checkpoint_import_vision_strip_ple_shards_mtp_and_steplora(self) -> None:
        rng = random.Random(9001)
        h = 128
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            ckpt = tmp / "qwen4_src"
            ckpt.mkdir()

            tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
                # Vision tower weights -> must be stripped!
                "visual.patch_embed.proj.weight": ("BF16", (64, h), _bf16_bytes([0.1] * (64 * h))),
                "vision_tower.encoder.layers.0.weight": ("BF16", (64, h), _bf16_bytes([0.1] * (64 * h))),
                "multi_modal_projector.linear.weight": ("BF16", (h, 64), _bf16_bytes([0.1] * (h * 64))),
                # Backbone (4L 3:1 GDN:QSA, shared_expert, routed experts, 4-branch GR)
                "model.embed_tokens.weight": ("BF16", (256, h), _bf16_bytes([rng.gauss(0, 0.1) for _ in range(256 * h)])),
                "model.norm.weight": ("BF16", (h,), _bf16_bytes([1.0] * h)),
                "lm_head.weight": ("BF16", (256, h), _bf16_bytes([rng.gauss(0, 0.1) for _ in range(256 * h)])),
                # Embedded MTP draft head
                "mtp.enorm.weight": ("BF16", (h,), _bf16_bytes([1.0] * h)),
                "mtp.hnorm.weight": ("BF16", (h,), _bf16_bytes([1.0] * h)),
                "mtp.eh_proj.weight": ("BF16", (h, 2 * h), _bf16_bytes([0.05] * (h * 2 * h))),
                "mtp.shared_head.norm.weight": ("BF16", (h,), _bf16_bytes([1.0] * h)),
            }
            for l in range(4):
                tensors[f"model.layers.{l}.input_layernorm.weight"] = ("BF16", (h,), _bf16_bytes([1.0] * h))
                tensors[f"model.layers.{l}.post_attention_layernorm.weight"] = ("BF16", (h,), _bf16_bytes([1.0] * h))
                tensors[f"model.layers.{l}.gr.read_gate.weight"] = ("BF16", (4, h), _bf16_bytes([0.25] * (4 * h)))
                tensors[f"model.layers.{l}.gr.write_gate.weight"] = ("BF16", (4, h), _bf16_bytes([0.25] * (4 * h)))
                tensors[f"model.layers.{l}.gr.decay_gate.weight"] = ("BF16", (4, h), _bf16_bytes([0.9] * (4 * h)))
                if (l + 1) % 4 != 0:
                    tensors[f"model.layers.{l}.linear_attn.q_proj.weight"] = ("BF16", (h, h), _bf16_bytes([0.05] * (h * h)))
                    tensors[f"model.layers.{l}.linear_attn.o_proj.weight"] = ("BF16", (h, h), _bf16_bytes([0.05] * (h * h)))
                    tensors[f"model.layers.{l}.linear_attn.norm.weight"] = ("BF16", (32,), _bf16_bytes([1.0] * 32))
                else:
                    tensors[f"model.layers.{l}.self_attn.q_proj.weight"] = ("BF16", (h, h), _bf16_bytes([0.05] * (h * h)))
                    tensors[f"model.layers.{l}.self_attn.idx_wq.weight"] = ("BF16", (64, h), _bf16_bytes([0.05] * (64 * h)))
                tensors[f"model.layers.{l}.mlp.gate.weight"] = ("BF16", (8, h), _bf16_bytes([0.05] * (8 * h)))
                tensors[f"model.layers.{l}.mlp.shared_expert.gate_proj.weight"] = ("BF16", (h, h), _bf16_bytes([0.05] * (h * h)))
                tensors[f"model.layers.{l}.mlp.shared_expert_gate.weight"] = ("BF16", (1, h), _bf16_bytes([0.1] * h))
                tensors[f"model.layers.{l}.mlp.experts.0.gate_proj.weight"] = ("BF16", (64, h), _bf16_bytes([0.05] * (64 * h)))
                tensors[f"model.layers.{l}.mlp.experts.1.gate_proj.weight"] = ("BF16", (64, h), _bf16_bytes([0.05] * (64 * h)))
                # PLE table per layer (200 rows -> split evenly across 2 shards = 100 rows/shard)
                tensors[f"ple.layers.{l}.weight"] = ("BF16", (200, h), _bf16_bytes([0.02] * (200 * h)))

            _write_safetensors(ckpt / "model.safetensors", tensors)
            (ckpt / "config.json").write_text(
                json.dumps({"model_type": "qwen4_exp", "num_hidden_layers": 4, "num_experts": 8, "tie_word_embeddings": False}),
                encoding="utf-8",
            )

            # Separate Step-LoRA adapter checkpoint (rank=16 -> zero-padded to packed_rank=64 in sme2_panel64!).
            ad_good = tmp / "adapter_good.safetensors"
            _write_safetensors(
                ad_good,
                {
                    "loop.step.0.layers.1.q_proj.lora_a": ("F16", (16, h), _f16_bytes([0.1] * (16 * h))),
                    "loop.step.0.layers.1.q_proj.lora_b": ("F16", (h, 16), _f16_bytes([0.1] * (h * 16))),
                    "loop.step.0.layers.1.q_proj.alpha": ("F32", (1,), struct.pack("<f", 0.0)),
                    "loop.step.1.layers.1.q_proj.lora_a": ("F16", (16, h), _f16_bytes([0.2] * (16 * h))),
                    "loop.step.1.layers.1.q_proj.lora_b": ("F16", (h, 16), _f16_bytes([0.2] * (h * 16))),
                    "loop.step.1.layers.1.q_proj.alpha": ("F32", (1,), struct.pack("<f", 0.25)),
                },
            )

            # Non-zero r=0 alpha MUST be rejected!
            ad_bad = tmp / "adapter_bad.safetensors"
            _write_safetensors(
                ad_bad,
                {
                    "loop.step.0.layers.1.q_proj.alpha": ("F32", (1,), struct.pack("<f", 0.125)),
                },
            )
            with self.assertRaises(ValueError):
                pack_qwen4_checkpoint(ckpt, tmp / "bad.amf", loop_adapter_input=ad_bad)

            out_dir = tmp / "packed"
            out_dir.mkdir()
            rc = cli_main([
                "pack-amf",
                "--input", str(ckpt),
                "--output", str(out_dir / "model.amf"),
                "--model-id", "qwen4-test",
                "--cold-experts", "0:1,1:1",
                "--ple-output", str(out_dir / "ple.amf"),
                "--ple-shards", "2",
                "--mtp-output", str(out_dir / "mtp.amf"),
                "--loop-adapter-input", str(ad_good),
                "--config-output", str(out_dir / "config.json"),
            ])
            self.assertEqual(rc, 0)

            # Verify backbone .amf contents, kinds, SME2 Step-LoRA packing, and vision stripping.
            with AmfReader(out_dir / "model.amf") as rdr:
                names = set(rdr.by_name.keys())
                self.assertFalse(any(n.startswith(("visual.", "vision_tower.", "multi_modal_projector.")) for n in names))
                self.assertEqual(rdr.by_name["layers.0.mlp.shared_expert.gate_proj.weight"].kind, TensorKind.SHARED_EXPERT)
                self.assertEqual(rdr.by_name["layers.0.mlp.shared_expert_gate.weight"].kind, TensorKind.SHARED_EXPERT)
                self.assertEqual(rdr.by_name["layers.0.mlp.experts.0.gate_proj.weight"].kind, TensorKind.HOT_EXPERT)
                self.assertEqual(rdr.by_name["layers.0.mlp.experts.1.gate_proj.weight"].kind, TensorKind.COLD_EXPERT)
                la0 = rdr.by_name["loop.step.0.layers.1.q_proj.lora_a"]
                lb0 = rdr.by_name["loop.step.0.layers.1.q_proj.lora_b"]
                al0 = rdr.by_name["loop.step.0.layers.1.q_proj.alpha"]
                al1 = rdr.by_name["loop.step.1.layers.1.q_proj.alpha"]
                self.assertEqual(la0.kind, TensorKind.LOOP_ADAPTER)
                self.assertEqual(la0.layout, TensorLayout.SME2_PANEL64)
                self.assertEqual(la0.shape, (64, h))
                self.assertEqual(lb0.layout, TensorLayout.SME2_PANEL64)
                self.assertEqual(lb0.shape, (h, 64))
                # Verify zero-padding beyond rank=16 in lora_a rows 16..63 and lora_b cols 16..63!
                la0_mat = rdr.read_matrix_f32(la0)
                lb0_mat = rdr.read_matrix_f32(lb0)
                self.assertTrue(all(v == 0.0 for row in la0_mat[16:] for v in row))
                self.assertTrue(all(v == 0.0 for row in lb0_mat for v in row[16:]))
                # Verify exact alpha values
                al0_val = struct.unpack("<f", rdr.read_blob(al0, verify_crc=True)[:4])[0]
                al1_val = struct.unpack("<f", rdr.read_blob(al1, verify_crc=True)[:4])[0]
                self.assertEqual(al0_val, 0.0)
                self.assertAlmostEqual(al1_val, 0.25, places=6)

            # Verify 2 PLE shards: SPARSE_TABLE = 8, 16 KiB aligned, valid CRCs.
            for s_idx in range(2):
                p_shard = out_dir / f"ple-shard-{s_idx:02d}-of-02.amf"
                self.assertTrue(p_shard.is_file())
                with AmfReader(p_shard) as pr:
                    self.assertEqual(len(pr.entries), 4)
                    for l in range(4):
                        m = pr.by_name[f"ple.layers.{l}.shard.{s_idx}.weight"]
                        self.assertEqual(m.kind, TensorKind.SPARSE_TABLE)
                        self.assertEqual(m.layer, l)
                        self.assertEqual(m.shape, (100, h))
                        self.assertEqual(m.blob_offset % APPLE_PAGE_SIZE, 0)
                        self.assertEqual(m.blob_bytes % APPLE_PAGE_SIZE, 0)
                        pr.read_blob(m, verify_crc=True)

            # Verify extracted MTP .amf passes Zig `aminfer verify-amf`!
            proc = _run_verify_amf(out_dir / "mtp.amf")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("amf.status       ok", proc.stdout)

            # Verify emitted config.json has canonical operator_axes, loop_schedule & stripped_vision_tensors.
            cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["stripped_vision_tensors"], 3)
            self.assertEqual(cfg["operator_axes"]["attention_kind"], "hybrid_gdn_qsa")
            self.assertEqual(
                cfg["operator_axes"]["layer_attention_kinds"],
                ["gated_deltanet", "gated_deltanet", "gated_deltanet", "qsa_sparse"],
            )
            self.assertTrue(cfg["operator_axes"]["has_shared_expert"])
            self.assertEqual(cfg["operator_axes"]["residual_kind"], "gated_4branch")
            self.assertEqual(cfg["operator_axes"]["memory_lookup_kind"], "ple_ngram")
            self.assertEqual(cfg["loop_schedule"]["mode"], "gated_loop_adapter")
            self.assertEqual(cfg["loop_schedule"]["adapter_kind"], "sme2_step_lora")
            self.assertEqual(cfg["loop_schedule"]["adapter_rank"], 16)
            self.assertEqual(cfg["loop_schedule"]["adapter_packed_rank"], 64)
            self.assertTrue(cfg["loop_schedule"]["enforce_zero_alpha_r0"])

    def test_synth_qwen4_tiny_and_scale_512e_presets_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            d1 = tmp / "tiny1"
            d2 = tmp / "tiny2"
            rc1 = cli_main(["synth-qwen4", "--preset", "tiny-qwen4", "--output-dir", str(d1), "--seed", "7"])
            rc2 = cli_main(["synth-qwen4", "--preset", "tiny-qwen4", "--output-dir", str(d2), "--seed", "7"])
            self.assertEqual(rc1, 0)
            self.assertEqual(rc2, 0)
            self.assertEqual(_sha256_file(d1 / "model.amf"), _sha256_file(d2 / "model.amf"))
            self.assertEqual(_sha256_file(d1 / "ple.amf"), _sha256_file(d2 / "ple.amf"))
            self.assertEqual(_sha256_file(d1 / "mtp.amf"), _sha256_file(d2 / "mtp.amf"))

            # Verify MTP draft head passes Zig `aminfer verify-amf`!
            proc = _run_verify_amf(d1 / "mtp.amf")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            # Verify tiny-qwen4 with E=64 override
            d64 = tmp / "tiny64"
            rc64 = cli_main(["synth-qwen4", "--preset", "tiny-qwen4", "--num-experts", "64", "--output-dir", str(d64)])
            self.assertEqual(rc64, 0)
            with AmfReader(d64 / "model.amf") as r64:
                self.assertIn("layers.0.mlp.experts.63.down_proj", r64.by_name)
                self.assertEqual(r64.by_name["layers.0.mlp.shared_expert.gate_proj"].kind, TensorKind.SHARED_EXPERT)

            # Verify `scale-qwen4-512e`: E=512 routed experts, top_k=10, shared_expert, [11,11,10] MRoPE+YaRN, 4-shard PLE!
            d512 = tmp / "scale512"
            rc512 = cli_main([
                "synth-qwen4",
                "--preset", "scale-qwen4-512e",
                "--output-dir", str(d512),
                "--cold-experts-per-layer", "128",
            ])
            self.assertEqual(rc512, 0)
            cfg512 = json.loads((d512 / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg512["num_experts"], 512)
            self.assertEqual(cfg512["num_experts_per_tok"], 10)
            self.assertEqual(cfg512["rope_scaling"]["rope_type"], "yarn")
            self.assertEqual(cfg512["rope_scaling"]["mrope_section"], [11, 11, 10])
            self.assertEqual(cfg512["ple"]["num_shards"], 4)
            self.assertEqual(
                cfg512["operator_axes"]["layer_attention_kinds"],
                ["gated_deltanet", "gated_deltanet", "gated_deltanet", "qsa_sparse"],
            )
            with AmfReader(d512 / "model.amf") as r512:
                hot_cnt = sum(1 for m in r512.entries if m.kind == TensorKind.HOT_EXPERT)
                cold_cnt = sum(1 for m in r512.entries if m.kind == TensorKind.COLD_EXPERT)
                sh_cnt = sum(1 for m in r512.entries if m.kind == TensorKind.SHARED_EXPERT)
                self.assertEqual(hot_cnt, 4 * 384 * 3)
                self.assertEqual(cold_cnt, 4 * 128 * 3)
                self.assertEqual(sh_cnt, 4 * 4)  # gate, up, down, shared_expert_gate
                # Spot-check expert 511 and raw directory kind byte == 7 on shared_expert.
                e511 = r512.by_name["layers.3.mlp.experts.511.down_proj"]
                self.assertEqual(e511.expert, 511)
                self.assertEqual(e511.kind, TensorKind.COLD_EXPERT)
                r512.read_blob(e511, verify_crc=True)
            for s in range(4):
                sp = d512 / f"ple-shard-{s:02d}-of-04.amf"
                self.assertTrue(sp.is_file())
                with AmfReader(sp) as psr:
                    self.assertEqual(len(psr.entries), 4)
                    for m in psr.entries:
                        self.assertEqual(m.kind, TensorKind.SPARSE_TABLE)
                        self.assertEqual(m.shape, (256, 256))
                        psr.read_blob(m, verify_crc=True)

    def test_synth_loop_tiny_looplm_and_tiny_steplora(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)

            # 1. Native LoopLM (`Prelude -> Core x R -> Coda`) — uses kinds 0..6, so Zig verifies it too!
            d_lm = tmp / "looplm"
            rc = cli_main(["synth-loop", "--preset", "tiny-looplm", "--output-dir", str(d_lm)])
            self.assertEqual(rc, 0)
            proc = _run_verify_amf(d_lm / "model.amf")
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("amf.status       ok", proc.stdout)
            cfg_lm = json.loads((d_lm / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg_lm["loop_schedule"]["mode"], "native_recurrent")
            self.assertEqual(cfg_lm["loop_schedule"]["start_layer"], 1)
            self.assertEqual(cfg_lm["loop_schedule"]["end_layer"], 3)
            self.assertEqual(cfg_lm["loop_schedule"]["max_loops"], 3)
            self.assertTrue(cfg_lm["loop_schedule"]["inject_x_init"])
            self.assertEqual(cfg_lm["loop_schedule"]["early_exit_gate"], "sigmoid_linear")
            with AmfReader(d_lm / "model.amf") as rlm:
                self.assertIn("loop.early_exit_gate.weight", rlm.by_name)

            # 2. Step-LoRA on Qwen3 (`alpha_0 == 0.0`, `r >= 1` non-zero, `sme2_panel64` with rank=16 -> 64)
            d_sl = tmp / "steplora_q3"
            rc = cli_main([
                "synth-loop",
                "--preset", "tiny-steplora",
                "--rank", "16",
                "--output-dir", str(d_sl),
            ])
            self.assertEqual(rc, 0)
            cfg_sl = json.loads((d_sl / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg_sl["loop_schedule"]["adapter_kind"], "sme2_step_lora")
            self.assertEqual(cfg_sl["loop_schedule"]["adapter_rank"], 16)
            self.assertEqual(cfg_sl["loop_schedule"]["adapter_packed_rank"], 64)
            with AmfReader(d_sl / "model.amf") as rsl:
                adapters = [m for m in rsl.entries if m.kind == TensorKind.LOOP_ADAPTER]
                # 3 steps * 2 core layers * 2 projs * 3 tensors (lora_a, lora_b, alpha) = 36
                self.assertEqual(len(adapters), 36)
                for m in adapters:
                    if m.name.endswith(".alpha"):
                        val = struct.unpack("<f", rsl.read_blob(m, verify_crc=True)[:4])[0]
                        if ".step.0." in m.name:
                            self.assertEqual(val, 0.0)
                        else:
                            self.assertGreater(val, 0.0)
                    else:
                        self.assertEqual(m.layout, TensorLayout.SME2_PANEL64)
                        rsl.read_blob(m, verify_crc=True)

            # 3. Step-LoRA on `qwen4_exp` base architecture
            d_sl4 = tmp / "steplora_q4"
            rc = cli_main([
                "synth-loop",
                "--preset", "tiny-steplora",
                "--base-arch", "qwen4_exp",
                "--output-dir", str(d_sl4),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((d_sl4 / "ple.amf").is_file())
            cfg_sl4 = json.loads((d_sl4 / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg_sl4["operator_axes"]["attention_kind"], "hybrid_gdn_qsa")
            self.assertEqual(cfg_sl4["loop_schedule"]["adapter_kind"], "sme2_step_lora")

    def test_native_looplm_safetensors_import_and_no_strip_vision_flag(self) -> None:
        h = 128
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            ckpt = tmp / "looplm_ckpt"
            ckpt.mkdir()
            tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
                "visual.encoder.weight": ("BF16", (32, h), _bf16_bytes([0.1] * (32 * h))),
                "model.embed_tokens.weight": ("BF16", (256, h), _bf16_bytes([0.05] * (256 * h))),
                "model.norm.weight": ("BF16", (h,), _bf16_bytes([1.0] * h)),
                "lm_head.weight": ("BF16", (256, h), _bf16_bytes([0.05] * (256 * h))),
                "loop.early_exit_gate.weight": ("BF16", (1, h), _bf16_bytes([0.1] * h)),
            }
            for l in range(4):
                tensors[f"model.layers.{l}.input_layernorm.weight"] = ("BF16", (h,), _bf16_bytes([1.0] * h))
                tensors[f"model.layers.{l}.self_attn.q_proj.weight"] = ("BF16", (h, h), _bf16_bytes([0.05] * (h * h)))
                tensors[f"model.layers.{l}.self_attn.o_proj.weight"] = ("BF16", (h, h), _bf16_bytes([0.05] * (h * h)))
                tensors[f"model.layers.{l}.post_attention_layernorm.weight"] = ("BF16", (h,), _bf16_bytes([1.0] * h))
                tensors[f"model.layers.{l}.mlp.gate_proj.weight"] = ("BF16", (256, h), _bf16_bytes([0.05] * (256 * h)))
                tensors[f"model.layers.{l}.mlp.down_proj.weight"] = ("BF16", (h, 256), _bf16_bytes([0.05] * (h * 256)))
            _write_safetensors(ckpt / "model.safetensors", tensors)
            (ckpt / "config.json").write_text(
                json.dumps({
                    "model_type": "looplm",
                    "num_hidden_layers": 4,
                    "num_prelude_layers": 1,
                    "num_coda_layers": 1,
                    "recurrent_depth": 4,
                    "inject_x_init": True,
                    "tie_word_embeddings": False,
                }),
                encoding="utf-8",
            )

            out_amf = tmp / "looplm.amf"
            out_cfg = tmp / "config.json"
            rc = cli_main([
                "pack-amf",
                "--input", str(ckpt),
                "--output", str(out_amf),
                "--model-id", "looplm-import",
                "--config-output", str(out_cfg),
            ])
            self.assertEqual(rc, 0)
            proc = _run_verify_amf(out_amf)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            cfg = json.loads(out_cfg.read_text(encoding="utf-8"))
            self.assertEqual(cfg["stripped_vision_tensors"], 1)
            self.assertEqual(cfg["loop_schedule"]["mode"], "native_recurrent")
            self.assertEqual(cfg["loop_schedule"]["start_layer"], 1)
            self.assertEqual(cfg["loop_schedule"]["end_layer"], 3)
            self.assertEqual(cfg["loop_schedule"]["max_loops"], 4)
            self.assertEqual(cfg["loop_schedule"]["early_exit_gate"], "sigmoid_linear")
            self.assertEqual(cfg["operator_axes"]["loop_schedule_kind"], "native_recurrent")

            # Also verify `--no-strip-vision` retains the vision tensor.
            out_keep = tmp / "keep_vision.amf"
            rc2 = cli_main([
                "pack-amf",
                "--input", str(ckpt),
                "--output", str(out_keep),
                "--model-id", "looplm-keep-vis",
                "--no-strip-vision",
            ])
            self.assertEqual(rc2, 0)
            _, keep_metas = load_amf_metadata(out_keep)
            self.assertTrue(any(m.name == "visual.encoder.weight" for m in keep_metas))

            # Verify `pack-amf --quant-profile gsq-rco --lut16-sme2` CLI end-to-end.
            out_lut = tmp / "looplm_lut16.amf"
            rc3 = cli_main([
                "pack-amf",
                "--input", str(ckpt),
                "--output", str(out_lut),
                "--model-id", "looplm-lut16",
                "--quant-profile", "gsq-rco",
                "--target-bits", "4.0",
                "--lut16-sme2",
            ])
            self.assertEqual(rc3, 0)
            proc3 = _run_verify_amf(out_lut)
            self.assertEqual(proc3.returncode, 0, msg=proc3.stderr)
            _, lut_metas = load_amf_metadata(out_lut)
            self.assertGreater(
                sum(1 for m in lut_metas if m.quant_param_kind == QuantParamKind.LUT16_CODEBOOK),
                0,
            )


if __name__ == "__main__":
    unittest.main()
