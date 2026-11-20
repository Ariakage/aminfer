"""Tests for deterministic synthetic Qwen3-MoE and Qwen3-Dense .amf/config/tokenizer generation."""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import DType, TensorKind
from aminfer_tools.cli import main
from aminfer_tools.synth import (
    DRAFT_PRESETS,
    PRESETS,
    DraftSynthConfig,
    Qwen3SynthConfig,
    bytes_to_unicode,
    generate_draft_bundle,
    generate_qwen3_bundle,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SynthTests(unittest.TestCase):
    def test_presets_match_target_qwen3_geometries(self) -> None:
        moe = PRESETS["qwen3-30b-a3b-3bit"]
        self.assertEqual(moe.model_type, "qwen3_moe")
        self.assertEqual(moe.quant_bits, 3)
        self.assertEqual(moe.group_size, 64)
        self.assertEqual(moe.hidden_size, 2048)
        self.assertEqual(moe.num_attention_heads, 32)
        self.assertEqual(moe.num_key_value_heads, 4)
        self.assertEqual(moe.head_dim, 128)
        self.assertEqual(moe.num_experts, 128)
        self.assertEqual(moe.num_experts_per_tok, 8)
        self.assertEqual(moe.moe_intermediate_size, 768)
        self.assertTrue(moe.norm_topk_prob)
        self.assertEqual(moe.rope_theta, 1_000_000.0)
        self.assertEqual(moe.vocab_size, 151936)

        dense = PRESETS["qwen3-14b-4bit"]
        self.assertEqual(dense.model_type, "qwen3")
        self.assertEqual(dense.quant_bits, 4)
        self.assertEqual(dense.group_size, 64)
        self.assertEqual(dense.hidden_size, 5120)
        self.assertEqual(dense.num_attention_heads, 40)
        self.assertEqual(dense.num_key_value_heads, 8)
        self.assertEqual(dense.head_dim, 128)
        self.assertEqual(dense.intermediate_size, 17408)
        self.assertEqual(dense.num_experts, 0)
        self.assertEqual(dense.vocab_size, 151936)

        self.assertIn("tiny-moe", PRESETS)
        self.assertIn("tiny-dense", PRESETS)

    def test_bytes_to_unicode_is_bijection(self) -> None:
        b2u = bytes_to_unicode()
        self.assertEqual(len(b2u), 256)
        self.assertEqual(len(set(b2u.values())), 256)
        self.assertEqual(b2u[ord(" ")], "\u0120")  # 'Ġ'
        self.assertEqual(b2u[ord("\n")], "\u010a")  # 'Ċ'
        self.assertEqual(b2u[ord("A")], "A")

    def test_tiny_moe_bundle_is_deterministic_and_verified_by_zig(self) -> None:
        cfg = PRESETS["tiny-moe"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            b1 = generate_qwen3_bundle(tmp / "run1", cfg, seed=42)
            b2 = generate_qwen3_bundle(tmp / "run2", cfg, seed=42)
            b3 = generate_qwen3_bundle(tmp / "run3", cfg, seed=99)

            self.assertEqual(_sha256(b1.amf_path), _sha256(b2.amf_path))
            self.assertEqual(_sha256(b1.config_path), _sha256(b2.config_path))
            self.assertEqual(_sha256(b1.tokenizer_path), _sha256(b2.tokenizer_path))
            self.assertNotEqual(_sha256(b1.amf_path), _sha256(b3.amf_path))

            self.assertEqual(b1.amf_path.stat().st_size % APPLE_PAGE_SIZE, 0)

            by_name = {m.name: m for m in b1.metas}
            self.assertEqual(by_name["embed_tokens"].kind, TensorKind.EMBEDDING)
            self.assertEqual(by_name["embed_tokens"].dtype, DType.Q3_AFFINE)
            self.assertEqual(by_name["embed_tokens"].shape, (cfg.vocab_size, cfg.hidden_size))
            self.assertEqual(by_name["lm_head"].kind, TensorKind.LM_HEAD)
            self.assertEqual(by_name["lm_head"].dtype, DType.Q3_AFFINE)
            self.assertEqual(by_name["norm.weight"].kind, TensorKind.NORM)
            self.assertEqual(by_name["norm.weight"].dtype, DType.F16)

            for layer in range(cfg.num_hidden_layers):
                self.assertEqual(
                    by_name[f"layers.{layer}.self_attn.q_proj"].shape,
                    (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
                )
                self.assertEqual(
                    by_name[f"layers.{layer}.self_attn.k_proj"].shape,
                    (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
                )
                self.assertEqual(
                    by_name[f"layers.{layer}.self_attn.o_proj"].shape,
                    (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
                )
                self.assertEqual(
                    by_name[f"layers.{layer}.self_attn.q_norm.weight"].shape,
                    (cfg.head_dim,),
                )
                self.assertEqual(
                    by_name[f"layers.{layer}.self_attn.k_norm.weight"].shape,
                    (cfg.head_dim,),
                )
                self.assertEqual(
                    by_name[f"layers.{layer}.mlp.gate.weight"].kind,
                    TensorKind.ROUTER,
                )
                self.assertEqual(
                    by_name[f"layers.{layer}.mlp.gate.weight"].dtype,
                    DType.F16,
                )
                self.assertEqual(
                    by_name[f"layers.{layer}.mlp.gate.weight"].shape,
                    (cfg.num_experts, cfg.hidden_size),
                )
                for expert in range(cfg.num_experts):
                    gp = by_name[f"layers.{layer}.mlp.experts.{expert}.gate_proj"]
                    up = by_name[f"layers.{layer}.mlp.experts.{expert}.up_proj"]
                    dp = by_name[f"layers.{layer}.mlp.experts.{expert}.down_proj"]
                    self.assertEqual(gp.kind, TensorKind.HOT_EXPERT)
                    self.assertEqual(gp.dtype, DType.Q3_AFFINE)
                    self.assertEqual(gp.shape, (cfg.moe_intermediate_size, cfg.hidden_size))
                    self.assertEqual(up.shape, (cfg.moe_intermediate_size, cfg.hidden_size))
                    self.assertEqual(dp.shape, (cfg.hidden_size, cfg.moe_intermediate_size))

            # Verify config.json schema.
            config_json = json.loads(b1.config_path.read_text(encoding="utf-8"))
            self.assertEqual(config_json["model_type"], "qwen3_moe")
            self.assertEqual(config_json["rope_theta"], 1_000_000.0)
            self.assertTrue(config_json["norm_topk_prob"])
            self.assertEqual(config_json["quantization"], {"bits": 3, "group_size": 64})
            self.assertFalse(config_json["tie_word_embeddings"])

            # Verify tokenizer.json schema and full vocab ID coverage.
            tok_json = json.loads(b1.tokenizer_path.read_text(encoding="utf-8"))
            vocab = tok_json["model"]["vocab"]
            merges = tok_json["model"]["merges"]
            added = tok_json["added_tokens"]
            self.assertGreater(len(merges), 10)
            for m in merges:
                left, right = m.split(" ")
                self.assertIn(left, vocab)
                self.assertIn(right, vocab)
                self.assertIn(left + right, vocab)
            added_ids = {t["id"] for t in added}
            added_contents = {t["content"] for t in added}
            self.assertTrue({"<|endoftext|>", "<|im_start|>", "<|im_end|>"}.issubset(added_contents))
            vocab_ids = set(vocab.values())
            self.assertTrue(vocab_ids.isdisjoint(added_ids))
            self.assertEqual(vocab_ids | added_ids, set(range(cfg.vocab_size)))

            # Cross-language verification with Zig aminfer verify-amf.
            self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
            proc = subprocess.run(
                [str(AMINFER_BIN), "verify-amf", str(b1.amf_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            self.assertIn("amf.status       ok", proc.stdout)

    def test_tiny_dense_tied_and_f16_embed_verified_by_zig(self) -> None:
        tied_cfg = PRESETS["tiny-dense"].with_overrides(tie_word_embeddings=True)
        f16_cfg = PRESETS["tiny-dense"].with_overrides(embed_f16=True, tie_word_embeddings=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            b_tied = generate_qwen3_bundle(tmp / "tied", tied_cfg, seed=7)
            tied_names = {m.name for m in b_tied.metas}
            self.assertNotIn("lm_head", tied_names)
            self.assertIn("layers.0.mlp.gate_proj", tied_names)
            self.assertIn("layers.0.mlp.up_proj", tied_names)
            self.assertIn("layers.0.mlp.down_proj", tied_names)
            self.assertNotIn("layers.0.mlp.gate.weight", tied_names)
            for m in b_tied.metas:
                if m.kind == TensorKind.DENSE:
                    self.assertEqual(m.dtype, DType.Q4_AFFINE)

            b_f16 = generate_qwen3_bundle(tmp / "f16", f16_cfg, seed=7)
            by_name = {m.name: m for m in b_f16.metas}
            self.assertEqual(by_name["embed_tokens"].dtype, DType.F16)
            self.assertEqual(by_name["lm_head"].dtype, DType.Q4_AFFINE)

            for bundle in (b_tied, b_f16):
                proc = subprocess.run(
                    [str(AMINFER_BIN), "verify-amf", str(bundle.amf_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
                self.assertIn("amf.status       ok", proc.stdout)

    def test_real_per_layer_geometries_pack_and_verify_fast(self) -> None:
        # 1 layer + small vocab exercises the exact Qwen3-30B-A3B-3bit and Qwen3-14B-4bit
        # transformer layer tensor dimensions while staying fast in unit tests.
        moe_1l = PRESETS["qwen3-30b-a3b-3bit"].with_overrides(num_hidden_layers=1, vocab_size=512)
        dense_1l = PRESETS["qwen3-14b-4bit"].with_overrides(num_hidden_layers=1, vocab_size=512)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            bm = generate_qwen3_bundle(tmp / "moe1l", moe_1l, seed=1)
            bd = generate_qwen3_bundle(tmp / "dense1l", dense_1l, seed=2)

            m_names = {m.name: m for m in bm.metas}
            self.assertEqual(m_names["layers.0.self_attn.q_proj"].shape, (4096, 2048))
            self.assertEqual(m_names["layers.0.self_attn.k_proj"].shape, (512, 2048))
            self.assertEqual(m_names["layers.0.self_attn.o_proj"].shape, (2048, 4096))
            self.assertEqual(m_names["layers.0.mlp.experts.127.gate_proj"].shape, (768, 2048))
            self.assertEqual(m_names["layers.0.mlp.experts.127.down_proj"].shape, (2048, 768))

            d_names = {m.name: m for m in bd.metas}
            self.assertEqual(d_names["layers.0.self_attn.q_proj"].shape, (5120, 5120))
            self.assertEqual(d_names["layers.0.self_attn.k_proj"].shape, (1024, 5120))
            self.assertEqual(d_names["layers.0.mlp.gate_proj"].shape, (17408, 5120))
            self.assertEqual(d_names["layers.0.mlp.down_proj"].shape, (5120, 17408))

            for bundle in (bm, bd):
                proc = subprocess.run(
                    [str(AMINFER_BIN), "verify-amf", str(bundle.amf_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
                self.assertIn("amf.status       ok", proc.stdout)

    def test_cli_synth_qwen3_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir) / "out"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "synth-qwen3",
                        "--preset",
                        "tiny-moe",
                        "--output-dir",
                        str(out_dir),
                        "--num-layers",
                        "1",
                        "--seed",
                        "123",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "model.amf").exists())
            self.assertTrue((out_dir / "config.json").exists())
            self.assertTrue((out_dir / "tokenizer.json").exists())
            self.assertIn("synthesized", buf.getvalue())

    def test_invalid_configs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PRESETS["tiny-moe"].with_overrides(quant_bits=2).validate()
        with self.assertRaises(ValueError):
            PRESETS["tiny-moe"].with_overrides(hidden_size=100).validate()
        with self.assertRaises(ValueError):
            PRESETS["tiny-moe"].with_overrides(num_experts_per_tok=16).validate()
        with self.assertRaises(ValueError):
            PRESETS["tiny-dense"].with_overrides(tie_word_embeddings=True, embed_f16=True).validate()
        with self.assertRaises(ValueError):
            Qwen3SynthConfig(
                model_id="bad",
                model_type="unknown",
                quant_bits=4,
                group_size=64,
                vocab_size=512,
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                intermediate_size=256,
                moe_intermediate_size=0,
                num_experts=0,
                num_experts_per_tok=0,
            ).validate()

    def test_synth_draft_eagle3_eagle2_mtp_and_zig_verify(self) -> None:
        self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)

            # 1. EAGLE-3: deterministic, exact RedHatAI/Eagle3DraftModel tensor layout + u32 d2t.
            e3_cfg = DRAFT_PRESETS["tiny-eagle3"]
            b1 = generate_draft_bundle(tmp / "e3_1", e3_cfg, seed=42)
            b2 = generate_draft_bundle(tmp / "e3_2", e3_cfg, seed=42)
            b3 = generate_draft_bundle(tmp / "e3_3", e3_cfg, seed=99)
            self.assertEqual(_sha256(b1.amf_path), _sha256(b2.amf_path))
            self.assertEqual(_sha256(b1.config_path), _sha256(b2.config_path))
            self.assertNotEqual(_sha256(b1.amf_path), _sha256(b3.amf_path))
            self.assertEqual(b1.amf_path.stat().st_size % APPLE_PAGE_SIZE, 0)

            e3_by_name = {m.name: m for m in b1.metas}
            self.assertEqual(len(e3_by_name), 15)
            h = e3_cfg.hidden_size
            q_dim = e3_cfg.num_attention_heads * e3_cfg.head_dim
            kv_dim = e3_cfg.num_key_value_heads * e3_cfg.head_dim
            self.assertEqual(e3_by_name["d2t"].dtype, DType.U32)
            self.assertEqual(e3_by_name["d2t"].kind, TensorKind.DENSE)
            self.assertEqual(e3_by_name["d2t"].shape, (e3_cfg.draft_vocab_size,))
            self.assertEqual(e3_by_name["fc.weight"].shape, (h, 3 * h))
            self.assertEqual(e3_by_name["layers.0.input_layernorm.weight"].shape, (h,))
            self.assertEqual(e3_by_name["layers.0.hidden_norm.weight"].shape, (h,))
            self.assertEqual(e3_by_name["layers.0.self_attn.q_proj.weight"].shape, (q_dim, 2 * h))
            self.assertEqual(e3_by_name["layers.0.self_attn.k_proj.weight"].shape, (kv_dim, 2 * h))
            self.assertEqual(e3_by_name["layers.0.self_attn.v_proj.weight"].shape, (kv_dim, 2 * h))
            self.assertEqual(e3_by_name["layers.0.self_attn.o_proj.weight"].shape, (h, q_dim))
            self.assertEqual(e3_by_name["layers.0.post_attention_layernorm.weight"].shape, (h,))
            self.assertEqual(e3_by_name["layers.0.mlp.gate_proj.weight"].shape, (e3_cfg.intermediate_size, h))
            self.assertEqual(e3_by_name["layers.0.mlp.up_proj.weight"].shape, (e3_cfg.intermediate_size, h))
            self.assertEqual(e3_by_name["layers.0.mlp.down_proj.weight"].shape, (h, e3_cfg.intermediate_size))
            self.assertEqual(e3_by_name["norm.weight"].shape, (h,))
            self.assertEqual(e3_by_name["lm_head.weight"].shape, (e3_cfg.draft_vocab_size, h))
            self.assertEqual(e3_by_name["embed_tokens.weight"].shape, (e3_cfg.vocab_size, h))

            # Validate d2t bytes are strictly increasing unique u32 target token IDs < vocab_size.
            raw_amf = b1.amf_path.read_bytes()
            d2t_m = e3_by_name["d2t"]
            d2t_ids = list(
                struct.unpack(
                    f"<{e3_cfg.draft_vocab_size}I",
                    raw_amf[d2t_m.blob_offset : d2t_m.blob_offset + d2t_m.data_bytes],
                )
            )
            self.assertEqual(len(set(d2t_ids)), e3_cfg.draft_vocab_size)
            self.assertEqual(d2t_ids, sorted(d2t_ids))
            self.assertLess(max(d2t_ids), e3_cfg.vocab_size)

            e3_json = json.loads(b1.config_path.read_text(encoding="utf-8"))
            self.assertEqual(e3_json["architectures"], ["Eagle3DraftModel"])
            self.assertEqual(e3_json["draft_vocab_size"], e3_cfg.draft_vocab_size)
            self.assertTrue(e3_json["norm_before_residual"])
            self.assertIsNone(e3_json["eagle_aux_hidden_state_layer_ids"])
            self.assertEqual(e3_json["transformer_layer_config"]["hidden_size"], h)
            self.assertEqual(e3_json["transformer_layer_config"]["vocab_size"], e3_cfg.vocab_size)

            # 2. EAGLE-2: fc.weight [H, 2*H] + 1 standard GQA decoder layer + norm + lm_head.
            e2_cfg = DRAFT_PRESETS["tiny-eagle2"]
            b_e2 = generate_draft_bundle(tmp / "e2", e2_cfg, seed=11)
            e2_by_name = {m.name: m for m in b_e2.metas}
            self.assertEqual(e2_by_name["fc.weight"].shape, (h, 2 * h))
            self.assertEqual(e2_by_name["layers.0.self_attn.q_proj.weight"].shape, (q_dim, h))
            self.assertEqual(e2_by_name["layers.0.self_attn.k_proj.weight"].shape, (kv_dim, h))
            self.assertEqual(e2_by_name["lm_head.weight"].shape, (e2_cfg.vocab_size, h))
            self.assertEqual(e2_by_name["norm.weight"].shape, (h,))
            self.assertNotIn("d2t", e2_by_name)
            self.assertNotIn("layers.0.hidden_norm.weight", e2_by_name)
            e2_json = json.loads(b_e2.config_path.read_text(encoding="utf-8"))
            self.assertEqual(e2_json["draft_format"], "eagle2")

            # 3. MTP (D3): enorm, hnorm, eh_proj [H, 2*H], QK-norm GQA layer, shared_head.norm, optional lm_head.
            mtp_cfg = DRAFT_PRESETS["tiny-mtp"]
            b_mtp = generate_draft_bundle(tmp / "mtp", mtp_cfg, seed=13)
            mtp_by_name = {m.name: m for m in b_mtp.metas}
            self.assertEqual(mtp_by_name["enorm.weight"].shape, (h,))
            self.assertEqual(mtp_by_name["hnorm.weight"].shape, (h,))
            self.assertEqual(mtp_by_name["eh_proj.weight"].shape, (h, 2 * h))
            self.assertEqual(mtp_by_name["layers.0.self_attn.q_norm.weight"].shape, (mtp_cfg.head_dim,))
            self.assertEqual(mtp_by_name["layers.0.self_attn.k_norm.weight"].shape, (mtp_cfg.head_dim,))
            self.assertEqual(mtp_by_name["shared_head.norm.weight"].shape, (h,))
            self.assertNotIn("lm_head.weight", mtp_by_name)
            self.assertNotIn("embed_tokens.weight", mtp_by_name)
            mtp_json = json.loads(b_mtp.config_path.read_text(encoding="utf-8"))
            self.assertEqual(mtp_json["draft_format"], "mtp")
            self.assertTrue(mtp_json["has_qk_norm"])
            self.assertTrue(mtp_json["share_lm_head"])

            # MTP with dedicated lm_head + f16 weights + no qk-norm via CLI.
            mtp_cli_dir = tmp / "mtp_cli"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "synth-draft",
                        "--format",
                        "mtp",
                        "--output-dir",
                        str(mtp_cli_dir),
                        "--dtype",
                        "f16",
                        "--no-qk-norm",
                        "--include-lm-head",
                        "--seed",
                        "7",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("synthesized mtp draft", buf.getvalue())
            mtp_cli_json = json.loads((mtp_cli_dir / "config.json").read_text(encoding="utf-8"))
            self.assertFalse(mtp_cli_json["has_qk_norm"])
            self.assertFalse(mtp_cli_json["share_lm_head"])

            # Cross-language verification of every generated draft .amf by Zig verify-amf.
            for amf_path in (b1.amf_path, b_e2.amf_path, b_mtp.amf_path, mtp_cli_dir / "model.amf"):
                proc = subprocess.run(
                    [str(AMINFER_BIN), "verify-amf", str(amf_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
                self.assertIn("amf.status       ok", proc.stdout)

    def test_invalid_draft_configs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DRAFT_PRESETS["tiny-eagle3"].with_overrides(draft_format="unknown").validate()
        with self.assertRaises(ValueError):
            DRAFT_PRESETS["tiny-eagle3"].with_overrides(draft_vocab_size=9999).validate()
        with self.assertRaises(ValueError):
            DRAFT_PRESETS["tiny-eagle2"].with_overrides(draft_vocab_size=256).validate()
        with self.assertRaises(ValueError):
            DRAFT_PRESETS["tiny-mtp"].with_overrides(dtype="fp8").validate()
        with self.assertRaises(ValueError):
            DraftSynthConfig(
                model_id="bad",
                draft_format="eagle3",
                vocab_size=512,
                draft_vocab_size=256,
                hidden_size=100,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                intermediate_size=256,
            ).validate()


if __name__ == "__main__":
    unittest.main()
