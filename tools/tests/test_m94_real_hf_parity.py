"""W28 real-HF parity tests for Qwen3.8-Flash-Next (`qwen4_exp`) & DeepSeek-V4.1-Flash-1017 (`deepseek41`)."""

from __future__ import annotations

import json
import os
import pathlib
import random
import struct
import subprocess
import tempfile
import unittest

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import AmfReader, DType, TensorKind, e4m3_byte_to_f32
from aminfer_tools.cli import main as cli_main
from aminfer_tools.deepseek41 import (
    DEEPSEEK41_PRESETS,
    TIER_WIRED_BUDGET_BYTES,
    HardwareTierError,
    convert_deepseek41_checkpoint,
    generate_synth_deepseek41_bundle,
    is_engram_tensor,
)
from aminfer_tools.deepseek41 import validate_hf_metadata_dir as validate_ds_dir
from aminfer_tools.gsq_rco import resolve_gsq_arch_tier_profile
from aminfer_tools.mimo_v2 import dequantize_hf_mxfp4_row_f32
from aminfer_tools.qwen4_exp import (
    generate_real_hf_qwen4_synth_bundle,
    is_dense_ple_tensor,
    is_ple_tensor,
    normalize_qwen4_tensor_name,
    pack_qwen4_checkpoint,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def _write_safetensors(
    path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    header: dict[str, object] = {}
    cursor = 0
    chunks: list[bytes] = []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        chunks.append(raw)
        cursor += len(raw)
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for chunk in chunks:
            fh.write(chunk)


def _verify_amf_cli(tc: unittest.TestCase, amf_path: pathlib.Path) -> None:
    if not AMINFER_BIN.exists():
        tc.skipTest(f"missing {AMINFER_BIN}; run `zig build` first")
    res = subprocess.run(
        [str(AMINFER_BIN), "verify-amf", str(amf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    tc.assertEqual(
        res.returncode,
        0,
        msg=f"verify-amf failed for {amf_path}:\n{res.stdout}\n{res.stderr}",
    )


def _bf16_list(values: list[float]) -> bytes:
    u16s = [
        (struct.unpack("<I", struct.pack("<f", v))[0] >> 16) & 0xFFFF for v in values
    ]
    return struct.pack(f"<{len(u16s)}H", *u16s)


class TestQwen4RealHfParity(unittest.TestCase):
    def test_qwen4_name_compaction_and_ple_dense_vs_shard_classification(self) -> None:
        long_samples = [
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "model.language_model.layers.47.attn_hyper_connection.input_mix_weight_down.weight",
            "model.language_model.layers.47.mlp_hyper_connection.block_inject_weight.weight",
            "model.language_model.layers.47.self_attn.indexer.index_qk_proj.weight",
            "model.language_model.layers.47.self_attn.indexer.q_layernorm.weight",
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        ]
        for raw in long_samples:
            compacted = normalize_qwen4_tensor_name(raw)
            self.assertLessEqual(
                len(compacted.encode("utf-8")), 47, msg=f"name > 47B: {compacted}"
            )

        self.assertEqual(
            normalize_qwen4_tensor_name(
                "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
                mtp_standalone=True,
            ),
            "hc_mixer.in_mix_dn.weight",
        )
        self.assertEqual(
            normalize_qwen4_tensor_name(
                "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
                mtp_standalone=True,
            ),
            "layers.0.attn.idx.qk_proj.weight",
        )

        # Dense PLE conditioning stays wired; ONLY external N-gram lookup shards are SPARSE_TABLE.
        self.assertTrue(
            is_dense_ple_tensor("model.language_model.layers.1.ple.conv1d.weight")
        )
        self.assertTrue(
            is_dense_ple_tensor("model.language_model.layers.1.ple.key_proj.weight")
        )
        self.assertTrue(
            is_dense_ple_tensor(
                "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
            )
        )
        self.assertFalse(
            is_ple_tensor("model.language_model.layers.1.ple.conv1d.weight")
        )
        self.assertFalse(
            is_ple_tensor(
                "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"
            )
        )
        self.assertTrue(
            is_ple_tensor(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
            )
        )
        self.assertTrue(is_ple_tensor("model.layers.0.ple.weight"))

    def test_qwen4_real_hf_synth_conversion_slicing_and_verify_amf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "q4_real_hf"
            bundle = generate_real_hf_qwen4_synth_bundle(
                out_dir,
                num_layers=4,
                hidden_size=128,
                vocab_size=512,
                num_experts=4,
                moe_intermediate_size=64,
                hc_count=4,
                hc_lowrank=16,
                ple_shards=2,
                ple_rows_per_shard=64,
                ple_embed_dim=64,
                include_mtp=True,
                include_vision_sidecar=True,
                seed=77,
            )

            self.assertEqual(len(bundle.ple_paths), 2)
            self.assertIsNotNone(bundle.mtp_path)
            self.assertIsNotNone(bundle.mtp_config_path)
            self.assertIsNotNone(bundle.vision_path)
            self.assertTrue(bundle.config_path.is_file())

            with AmfReader(
                bundle.amf_path, verify_header_crc=True, verify_dir_crc=True
            ) as r:
                names = set(r.by_name.keys())
                for m in r.entries:
                    self.assertLessEqual(len(m.name.encode("utf-8")), 47)
                    self.assertEqual(m.blob_offset % APPLE_PAGE_SIZE, 0)
                    self.assertEqual(m.blob_bytes % APPLE_PAGE_SIZE, 0)
                    r.read_blob(m, verify_crc=True)

                # hc_count=4 hyper-connection projections present with compacted names.
                self.assertIn("hc_mixer.hc_norm.weight", names)
                self.assertIn("hc_mixer.in_mix_dn.weight", names)
                self.assertIn("layers.0.attn_hc.blk_inj.weight", names)
                self.assertIn("layers.0.mlp_hc.in_mix_up.weight", names)

                # GDN projection splits & F32 SSM parameters.
                self.assertIn("layers.0.linear_attn.q_proj.weight", names)
                self.assertIn("layers.0.linear_attn.k_proj.weight", names)
                self.assertIn("layers.0.linear_attn.v_proj.weight", names)
                self.assertIn("layers.0.linear_attn.g_proj.weight", names)
                self.assertIn("layers.0.linear_attn.a_proj.weight", names)
                self.assertIn("layers.0.linear_attn.b_proj.weight", names)
                self.assertIn("layers.0.linear_attn.o_proj.weight", names)
                self.assertEqual(
                    r.by_name["layers.0.linear_attn.conv1d.weight"].dtype, DType.F32
                )
                self.assertEqual(
                    r.by_name["layers.0.linear_attn.conv1d.weight"].shape[1], 4
                )
                self.assertEqual(
                    r.by_name["layers.0.linear_attn.dt_bias"].dtype, DType.F32
                )

                # QSA output-gated q_proj split & indexer weights.
                self.assertIn("layers.3.self_attn.q_proj.weight", names)
                self.assertIn("layers.3.self_attn.q_gate_proj.weight", names)
                self.assertIn("layers.3.attn.idx.qk_proj.weight", names)
                self.assertIn("layers.3.attn.idx.q_norm.weight", names)
                self.assertIn("layers.3.attn.idx.k_norm.weight", names)

                # Dense PLE conditioning stays wired in model.amf (NOT SPARSE_TABLE).
                self.assertEqual(
                    r.by_name["layers.1.ple.conv1d.weight"].kind, TensorKind.DENSE
                )
                self.assertEqual(
                    r.by_name["layers.1.ple.conv1d.weight"].shape, (4 * 128, 4)
                )
                self.assertEqual(
                    r.by_name["layers.1.ple.norm_conv.weight"].kind, TensorKind.NORM
                )
                self.assertEqual(r.by_name["layers.1.ple.layer_mult"].dtype, DType.U32)
                self.assertEqual(r.by_name["layers.1.ple.layer_mult"].shape, (3, 2))
                raw_mult = r.read_blob(
                    r.by_name["layers.1.ple.layer_mult"], verify_crc=True
                )[:24]
                self.assertEqual(
                    list(struct.unpack("<3q", raw_mult)), [1000003, 2000003, 3000017]
                )

                # 3D stacked experts sliced cleanly into per-expert gate_proj, up_proj, down_proj.
                self.assertEqual(
                    r.by_name["layers.0.mlp.experts.0.gate_proj.weight"].kind,
                    TensorKind.HOT_EXPERT,
                )
                self.assertEqual(
                    r.by_name["layers.0.mlp.experts.0.up_proj.weight"].shape, (64, 128)
                )
                self.assertEqual(
                    r.by_name["layers.0.mlp.experts.3.down_proj.weight"].kind,
                    TensorKind.COLD_EXPERT,
                )
                self.assertEqual(
                    r.by_name["layers.0.mlp.shared_expert_gate.weight"].kind,
                    TensorKind.SHARED_EXPERT,
                )

            cfg_json = json.loads(bundle.config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                cfg_json["ple"]["layer_multipliers"], [1000003, 2000003, 3000017]
            )
            self.assertEqual(
                cfg_json["ple"]["ngram_heads_vocab_sizes"], [16, 16, 16, 16]
            )
            self.assertTrue(cfg_json["ple"]["dense_conditioning"])

            # External NVMe PLE shards are strictly SPARSE_TABLE.
            for pp in bundle.ple_paths:
                with AmfReader(pp, verify_header_crc=True, verify_dir_crc=True) as pr:
                    self.assertEqual(len(pr.entries), 1)
                    self.assertEqual(pr.entries[0].kind, TensorKind.SPARSE_TABLE)
                _verify_amf_cli(self, pp)

            # Standalone 1L MTP .amf + mtp_config.json.
            assert bundle.mtp_path is not None and bundle.mtp_config_path is not None
            with AmfReader(
                bundle.mtp_path, verify_header_crc=True, verify_dir_crc=True
            ) as mr:
                self.assertIn("fc_embedding.weight", mr.by_name)
                self.assertIn("hc_mixer.in_mix_dn.weight", mr.by_name)
                self.assertIn("layers.0.self_attn.q_gate_proj.weight", mr.by_name)
                self.assertIn("layers.0.mlp.experts.0.gate_proj.weight", mr.by_name)
                for m in mr.entries:
                    self.assertLessEqual(len(m.name.encode("utf-8")), 47)
            _verify_amf_cli(self, bundle.mtp_path)
            _verify_amf_cli(self, bundle.amf_path)
            assert bundle.vision_path is not None
            _verify_amf_cli(self, bundle.vision_path)

    def test_qwen4_real_hf_exact_slicing_and_gsq_rco_pinning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sf_dir = pathlib.Path(tmp) / "sf"
            amf_gsq = pathlib.Path(tmp) / "model_gsq.amf"
            h = 128
            qh, hd = 2, 64
            moe_i = 64
            n_exp = 16

            # Distinct identifiable values per expert half and per Q/gate head.
            q_rows: list[float] = []
            for h_i in range(qh):
                for _ in range(hd):
                    q_rows.extend([0.25 if h_i == 0 else 0.75] * h)
                for _ in range(hd):
                    q_rows.extend([-0.25 if h_i == 0 else -0.75] * h)

            gate_up_vals: list[float] = []
            for e_idx in range(n_exp):
                g_v = 0.125 + 0.25 * (e_idx % 2)
                u_v = -(0.125 + 0.25 * (e_idx % 2))
                gate_up_vals.extend([g_v] * (moe_i * h))
                gate_up_vals.extend([u_v] * (moe_i * h))
            down_vals = [0.25] * (n_exp * h * moe_i)

            tensors = {
                "model.language_model.embed_tokens.weight": (
                    "BF16",
                    (64, h),
                    _bf16_list([0.05] * (64 * h)),
                ),
                "lm_head.weight": ("BF16", (64, h), _bf16_list([0.05] * (64 * h))),
                "model.language_model.norm.weight": (
                    "BF16",
                    (h,),
                    _bf16_list([1.0] * h),
                ),
                "model.language_model.layers.0.input_layernorm.weight": (
                    "BF16",
                    (h,),
                    _bf16_list([1.0] * h),
                ),
                "model.language_model.layers.0.post_attention_layernorm.weight": (
                    "BF16",
                    (h,),
                    _bf16_list([1.0] * h),
                ),
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": (
                    "BF16",
                    (4, h),
                    _bf16_list([0.1] * (4 * h)),
                ),
                "model.language_model.layers.0.linear_attn.in_proj_b.weight": (
                    "BF16",
                    (4, h),
                    _bf16_list([0.2] * (4 * h)),
                ),
                "model.language_model.layers.0.self_attn.q_proj.weight": (
                    "BF16",
                    (2 * qh * hd, h),
                    _bf16_list(q_rows),
                ),
                "model.language_model.layers.0.self_attn.o_proj.weight": (
                    "BF16",
                    (h, qh * hd),
                    _bf16_list([0.1] * (h * qh * hd)),
                ),
                "model.language_model.layers.0.self_attn.indexer.index_qk_proj.weight": (
                    "BF16",
                    (128, h),
                    _bf16_list([0.1] * (128 * h)),
                ),
                "model.language_model.layers.0.mlp.experts.gate_up_proj": (
                    "BF16",
                    (n_exp, 2 * moe_i, h),
                    _bf16_list(gate_up_vals),
                ),
                "model.language_model.layers.0.mlp.experts.down_proj": (
                    "BF16",
                    (n_exp, h, moe_i),
                    _bf16_list(down_vals),
                ),
            }
            cfg = {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "num_hidden_layers": 1,
                    "layer_types": ["full_attention"],
                    "hidden_size": h,
                    "vocab_size": 64,
                    "num_attention_heads": qh,
                    "num_key_value_heads": 1,
                    "head_dim": hd,
                    "output_gate_type": "sigmoid",
                    "num_experts": n_exp,
                    "moe_intermediate_size": moe_i,
                    "tie_word_embeddings": False,
                },
            }
            _write_safetensors(sf_dir / "model.safetensors", tensors)
            (sf_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            amf_u = pathlib.Path(tmp) / "model_uniform.amf"
            pack_qwen4_checkpoint(sf_dir, amf_u, quant_profile="uniform")
            with AmfReader(amf_u, verify_header_crc=True, verify_dir_crc=True) as ru:
                q_mat = ru.read_matrix_f32(
                    ru.by_name["layers.0.self_attn.q_proj.weight"]
                )
                qg_mat = ru.read_matrix_f32(
                    ru.by_name["layers.0.self_attn.q_gate_proj.weight"]
                )
                self.assertAlmostEqual(q_mat[0][0], 0.25, places=4)
                self.assertAlmostEqual(q_mat[hd][0], 0.75, places=4)
                self.assertAlmostEqual(qg_mat[0][0], -0.25, places=4)
                self.assertAlmostEqual(qg_mat[hd][0], -0.75, places=4)

                e0_g = ru.read_matrix_f32(
                    ru.by_name["layers.0.mlp.experts.0.gate_proj.weight"]
                )
                e0_u = ru.read_matrix_f32(
                    ru.by_name["layers.0.mlp.experts.0.up_proj.weight"]
                )
                e1_g = ru.read_matrix_f32(
                    ru.by_name["layers.0.mlp.experts.1.gate_proj.weight"]
                )
                e1_u = ru.read_matrix_f32(
                    ru.by_name["layers.0.mlp.experts.1.up_proj.weight"]
                )
                self.assertAlmostEqual(e0_g[0][0], 0.125, places=4)
                self.assertAlmostEqual(e0_u[0][0], -0.125, places=4)
                self.assertAlmostEqual(e1_g[0][0], 0.375, places=4)
                self.assertAlmostEqual(e1_u[0][0], -0.375, places=4)

            prof = resolve_gsq_arch_tier_profile("qwen4_exp", "24gb-m4")
            self.assertIsNotNone(prof)
            pack_qwen4_checkpoint(
                sf_dir, amf_gsq, quant_profile="gsq-rco", target_tier="24gb-m4"
            )
            with AmfReader(amf_gsq, verify_header_crc=True, verify_dir_crc=True) as rg:
                self.assertEqual(
                    rg.by_name["layers.0.linear_attn.a_proj.weight"].dtype, DType.BF16
                )
                self.assertEqual(
                    rg.by_name["layers.0.linear_attn.b_proj.weight"].dtype, DType.BF16
                )
                self.assertEqual(
                    rg.by_name["layers.0.attn.idx.qk_proj.weight"].dtype, DType.BF16
                )
            _verify_amf_cli(self, amf_gsq)

    def test_opportunistic_tiny_random_qwen4exp_conversion(self) -> None:
        models_env = os.environ.get("AMINFER_MODELS_DIR")
        if not models_env:
            self.skipTest("AMINFER_MODELS_DIR not set")
        root = pathlib.Path(models_env)
        malaiwah_dir = root / "qwen4-exp-tiny-random-bf16"
        hf_tiny_dir = root / "tiny-random-Qwen4ExpForConditionalGeneration"
        if (
            not (malaiwah_dir / "model.safetensors").is_file()
            and not (hf_tiny_dir / "model.safetensors").is_file()
        ):
            self.skipTest(
                "tiny-random Qwen4Exp fixtures not present under AMINFER_MODELS_DIR"
            )

        with tempfile.TemporaryDirectory() as tmp:
            if (malaiwah_dir / "model.safetensors").is_file():
                m_out = pathlib.Path(tmp) / "malaiwah"
                rep = pack_qwen4_checkpoint(
                    malaiwah_dir,
                    m_out / "model.amf",
                    model_id="qwen4-tiny-bf16",
                    ple_output=m_out / "ple.amf",
                    ple_shards=2,
                    mtp_output=m_out / "mtp.amf",
                    vision_sidecar=m_out / "vision.amf",
                    config_output=m_out / "config.json",
                )
                for m in rep.metas:
                    self.assertLessEqual(len(m.name.encode("utf-8")), 47)
                _verify_amf_cli(self, m_out / "model.amf")
                if (m_out / "vision.amf").is_file():
                    _verify_amf_cli(self, m_out / "vision.amf")
                for pp in rep.ple_shard_paths:
                    _verify_amf_cli(self, pp)
                if rep.mtp_amf_path is not None:
                    _verify_amf_cli(self, rep.mtp_amf_path)

            if (hf_tiny_dir / "model.safetensors").is_file():
                t_out = pathlib.Path(tmp) / "hf_tiny"
                rep_t = pack_qwen4_checkpoint(
                    hf_tiny_dir,
                    t_out / "model.amf",
                    model_id="qwen4-hf-tiny-f32",
                    ple_output=t_out / "ple.amf",
                    ple_shards=4,
                    mtp_output=t_out / "mtp.amf",
                    vision_sidecar=t_out / "vision.amf",
                    config_output=t_out / "config.json",
                )
                for m in rep_t.metas:
                    self.assertLessEqual(len(m.name.encode("utf-8")), 47)
                _verify_amf_cli(self, t_out / "model.amf")
                for pp in rep_t.ple_shard_paths:
                    _verify_amf_cli(self, pp)


class TestDeepSeek41RealHfParity(unittest.TestCase):
    def test_mxfp8_32x32_and_mxfp4_i8_numerical_parity_and_engram_split(self) -> None:
        self.assertFalse(is_engram_tensor("layers.1.engram.wkv.weight"))
        self.assertFalse(is_engram_tensor("layers.1.engram.q_weight"))
        self.assertFalse(is_engram_tensor("layers.1.engram.k_weight"))
        self.assertTrue(is_engram_tensor("layers.1.engram.embed.weight"))
        self.assertTrue(is_engram_tensor("layers.1.engram.embed.scale"))

        with tempfile.TemporaryDirectory() as tmp:
            sf_dir = pathlib.Path(tmp) / "ds_sf"
            rng = random.Random(314)
            n_rows, k_cols = (
                48,
                64,
            )  # non-multiple-of-32 rows tests [ceil(48/32), ceil(64/32)] = [2, 2] broadcast!
            nb32, kb32 = 2, 2
            w_e4m3 = bytes(
                ((0x80 if (i * 5 + 1) % 3 == 0 else 0x00) | (0x20 + (i % 16)))
                for i in range(n_rows * k_cols)
            )
            s_ue8m0 = bytes([124, 127, 126, 129])

            mxfp4_i8_raw = bytes(
                (rng.randrange(0, 16) << 4) | rng.randrange(0, 16)
                for _ in range(n_rows * (k_cols // 2))
            )
            mxfp4_scale = bytes(125 + (i % 4) for i in range(n_rows * kb32))

            eng_w = bytes(
                ((0x80 if i % 4 == 0 else 0x00) | (0x22 + (i % 12)))
                for i in range(32 * 64)
            )
            eng_s = bytes(126 + (i % 3) for i in range(32 * 2))

            tensors = {
                "embed.weight": ("BF16", (64, 64), _bf16_list([0.05] * (64 * 64))),
                "head.weight": ("BF16", (64, 64), _bf16_list([0.05] * (64 * 64))),
                "norm.weight": ("BF16", (64,), _bf16_list([1.0] * 64)),
                "layers.0.attn_norm.weight": ("BF16", (64,), _bf16_list([1.0] * 64)),
                "layers.0.ffn_norm.weight": ("BF16", (64,), _bf16_list([1.0] * 64)),
                "layers.0.attn.wkv.weight": ("F8_E4M3", (n_rows, k_cols), w_e4m3),
                "layers.0.attn.wkv.scale": ("F8_E8M0", (nb32, kb32), s_ue8m0),
                "layers.0.ffn.experts.0.w1.weight": (
                    "I8",
                    (n_rows, k_cols // 2),
                    mxfp4_i8_raw,
                ),
                "layers.0.ffn.experts.0.w1.scale": (
                    "F8_E8M0",
                    (n_rows, kb32),
                    mxfp4_scale,
                ),
                "layers.0.engram.wkv.weight": ("F8_E4M3", (n_rows, k_cols), w_e4m3),
                "layers.0.engram.wkv.scale": ("F8_E8M0", (nb32, kb32), s_ue8m0),
                "layers.0.engram.q_weight": ("BF16", (4, 64), _bf16_list([0.1] * 256)),
                "layers.0.engram.k_weight": ("BF16", (4, 64), _bf16_list([0.2] * 256)),
                "layers.0.engram.embed.weight": ("F8_E4M3", (32, 64), eng_w),
                "layers.0.engram.embed.scale": ("F8_E8M0", (32, 2), eng_s),
            }
            cfg = {
                "model_type": "deepseek_v41",
                "quantization_config": {
                    "quant_method": "fp8",
                    "expert_dtype": "fp4",
                    "scale_block_size": [32, 32],
                },
                "text_config": {
                    "model_type": "deepseek_v41_text",
                    "num_hidden_layers": 1,
                    "num_encoder_layers": 1,
                    "num_decoder_layers": 0,
                    "hidden_size": 64,
                    "vocab_size": 64,
                    "n_routed_experts": 1,
                    "tie_word_embeddings": False,
                },
            }
            _write_safetensors(sf_dir / "model.safetensors", tensors)
            (sf_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            # 1. Lossless auto mode preserves MXFP8_G32 + GGUF_MXFP4 bit-exact.
            out_auto = pathlib.Path(tmp) / "out_auto" / "model.amf"
            rep = convert_deepseek41_checkpoint(sf_dir, out_auto, target_dtype="auto")
            self.assertEqual(len(rep.engram_paths), 1)
            with AmfReader(out_auto, verify_header_crc=True, verify_dir_crc=True) as r:
                wkv_m = r.by_name["layers.0.attn.wkv"]
                self.assertEqual(wkv_m.dtype, DType.MXFP8_G32)
                mat_wkv = r.read_matrix_f32(wkv_m)
                for row_i in (0, 31, 32, 47):
                    for col_i in (0, 31, 32, 63):
                        exp_sc = 2.0 ** (
                            s_ue8m0[(row_i // 32) * kb32 + (col_i // 32)] - 127
                        )
                        exp_v = (
                            e4m3_byte_to_f32(w_e4m3[row_i * k_cols + col_i]) * exp_sc
                        )
                        self.assertAlmostEqual(mat_wkv[row_i][col_i], exp_v, places=6)

                exp_m = r.by_name["layers.0.ffn.experts.0.w1.weight"]
                self.assertEqual(exp_m.dtype, DType.GGUF_MXFP4)
                mat_exp = r.read_matrix_f32(exp_m)
                for row_i in (0, 17, 47):
                    expected_row = dequantize_hf_mxfp4_row_f32(
                        mxfp4_i8_raw[
                            row_i * (k_cols // 2) : (row_i + 1) * (k_cols // 2)
                        ],
                        mxfp4_scale[row_i * kb32 : (row_i + 1) * kb32],
                        k_cols,
                    )
                    self.assertEqual(mat_exp[row_i], expected_row)

                self.assertEqual(
                    r.by_name["layers.0.engram.wkv"].kind, TensorKind.DENSE
                )
                self.assertEqual(
                    r.by_name["layers.0.engram.q_weight"].kind, TensorKind.DENSE
                )
            with AmfReader(
                rep.engram_paths[0], verify_header_crc=True, verify_dir_crc=True
            ) as er:
                self.assertEqual(
                    er.by_name["engram.layers.0.weight"].dtype, DType.MXFP8_G32
                )
                self.assertEqual(
                    er.by_name["engram.layers.0.weight"].kind, TensorKind.SPARSE_TABLE
                )
            _verify_amf_cli(self, out_auto)
            _verify_amf_cli(self, rep.engram_paths[0])

            # 2. Re-quantizing to q4_affine & BF16 dequant works cleanly over MXFP8_G32 + GGUF_MXFP4!
            out_q4 = pathlib.Path(tmp) / "out_q4" / "model.amf"
            convert_deepseek41_checkpoint(sf_dir, out_q4, target_dtype="q4_affine")
            with AmfReader(out_q4, verify_header_crc=True, verify_dir_crc=True) as rq4:
                self.assertEqual(
                    rq4.by_name["layers.0.attn.wkv"].dtype, DType.Q4_AFFINE
                )
                self.assertEqual(
                    rq4.by_name["layers.0.ffn.experts.0.w1"].dtype, DType.Q4_AFFINE
                )
            _verify_amf_cli(self, out_q4)

            out_bf16 = pathlib.Path(tmp) / "out_bf16" / "model.amf"
            convert_deepseek41_checkpoint(sf_dir, out_bf16, target_dtype="bf16")
            with AmfReader(
                out_bf16, verify_header_crc=True, verify_dir_crc=True
            ) as rbf:
                self.assertEqual(
                    rbf.by_name["layers.0.ffn.experts.0.w1.weight"].dtype, DType.BF16
                )
            _verify_amf_cli(self, out_bf16)

    def test_24gb_m4_encoder_cold_decoder_hot_and_3l_dspark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Real-HF synth bundle with 2L Enc + 2L Dec, MXFP8 + GGUF_MXFP4 + 3L DSpark, encoder-cold on 24gb-m4.
            synth_dir = pathlib.Path(tmp) / "ds_real_hf"
            bundle = generate_synth_deepseek41_bundle(
                synth_dir,
                preset="tiny-deepseek41",
                seed=2026,
                real_hf_schema=True,
                target_tier="24gb-m4",
                encoder_cold_decoder_hot=True,
            )
            self.assertIsNotNone(bundle.dspark_amf_path)
            self.assertIsNotNone(bundle.dspark_config_path)
            with AmfReader(
                bundle.amf_path, verify_header_crc=True, verify_dir_crc=True
            ) as r:
                # Encoder layers (0, 1) must have ALL experts cold; Decoder layers (2, 3) have hot experts!
                enc_cold = {
                    e.expert
                    for e in r.entries
                    if e.layer < 2 and e.kind == TensorKind.COLD_EXPERT
                }
                enc_hot = {
                    e.expert
                    for e in r.entries
                    if e.layer < 2 and e.kind == TensorKind.HOT_EXPERT
                }
                dec_hot = {
                    e.expert
                    for e in r.entries
                    if e.layer >= 2 and e.kind == TensorKind.HOT_EXPERT
                }
                self.assertGreater(len(enc_cold), 0)
                self.assertEqual(len(enc_hot), 0)
                self.assertGreater(len(dec_hot), 0)
                self.assertIn("layers.0.hc_attn_scale", r.by_name)
                self.assertEqual(r.by_name["layers.0.hc_attn_scale"].dtype, DType.F32)
                self.assertIn("layers.0.ffn.gate.bias_vl", r.by_name)
                self.assertIn("layers.0.attn.attn_sink", r.by_name)

            # 3L DSpark preserves exact layer numbers (0, 1, 2) and verifies cleanly!
            assert (
                bundle.dspark_amf_path is not None
                and bundle.dspark_config_path is not None
            )
            ds_cfg = json.loads(bundle.dspark_config_path.read_text(encoding="utf-8"))
            self.assertEqual(ds_cfg["draft_format"], "dspark_3l_mtp")
            self.assertEqual(ds_cfg["num_nextn_predict_layers"], 3)
            with AmfReader(
                bundle.dspark_amf_path, verify_header_crc=True, verify_dir_crc=True
            ) as dr:
                self.assertIn("mtp.0.main_norm.weight", dr.by_name)
                self.assertIn("mtp.0.main_proj", dr.by_name)
                self.assertIn("mtp.2.markov_head.embed.weight", dr.by_name)
                self.assertIn("mtp.2.confidence_head.proj.weight", dr.by_name)
                self.assertEqual(dr.by_name["mtp.0.ffn.experts.0.w1.weight"].layer, 0)
                self.assertEqual(dr.by_name["mtp.1.ffn.experts.0.w1.weight"].layer, 1)
                self.assertEqual(dr.by_name["mtp.2.ffn.experts.0.w1.weight"].layer, 2)
                for m in dr.entries:
                    self.assertLessEqual(len(m.name.encode("utf-8")), 47)

            _verify_amf_cli(self, bundle.amf_path)
            _verify_amf_cli(self, bundle.engram_paths[0])
            _verify_amf_cli(self, bundle.dspark_amf_path)

            # 2. Verify CLI synth-qwen4 --real-hf-schema and synth-deepseek41 --real-hf-schema
            cli_q4 = pathlib.Path(tmp) / "cli_q4"
            self.assertEqual(
                cli_main(
                    [
                        "synth-qwen4",
                        "--preset",
                        "tiny-qwen4",
                        "--real-hf-schema",
                        "--output-dir",
                        str(cli_q4),
                    ]
                ),
                0,
            )
            _verify_amf_cli(self, cli_q4 / "model.amf")

            cli_ds = pathlib.Path(tmp) / "cli_ds"
            self.assertEqual(
                cli_main(
                    [
                        "synth-deepseek41",
                        "--preset",
                        "tiny-deepseek41",
                        "--real-hf-schema",
                        "--output-dir",
                        str(cli_ds),
                    ]
                ),
                0,
            )
            _verify_amf_cli(self, cli_ds / "model.amf")

            # 3. Verify 384E metadata config allows 24gb-m4 ONLY when encoder_cold_decoder_hot or allow_tier_override.
            meta_env = os.environ.get("AMINFER_METADATA_DIR")
            if (
                meta_env
                and (
                    pathlib.Path(meta_env)
                    / "deepseek-ai__DeepSeek-V4.1-Flash-1017"
                    / "config.json"
                ).is_file()
            ):
                ds_rep = validate_ds_dir(
                    pathlib.Path(meta_env) / "deepseek-ai__DeepSeek-V4.1-Flash-1017"
                )
                self.assertFalse(ds_rep.tier_admission["24gb-m4"]["fits_all_wired_q3"])
                self.assertLessEqual(
                    ds_rep.tier_admission["24gb-m4"]["encoder_cold_decoder_hot"][
                        "non_expert_q3_gib"
                    ],
                    TIER_WIRED_BUDGET_BYTES["24gb-m4"] / (1024**3),
                )
            with self.assertRaises(HardwareTierError):
                generate_synth_deepseek41_bundle(
                    pathlib.Path(tmp) / "reject_552b",
                    preset="full-deepseek41-552b",
                    config=DEEPSEEK41_PRESETS["full-deepseek41-552b"],
                )
