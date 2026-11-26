"""Milestone M9.2 GLM-5 (`glm_moe_dsa`, `glm5_next`, W20 ViT, 1L MTP, DFlash2) tests."""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aminfer_tools.amf import (  # noqa: E402
    AmfReader,
    DType,
    META_CONFIG_JSON,
    META_TOKENIZER_JSON,
    TensorKind,
    bf16_bytes_to_f32_list,
    f32_to_bf16_bytes as f32_values_to_bf16_bytes,
    f32_values_to_f16_bytes,
    pack_amf_from_safetensors,
)
from aminfer_tools.cli import main as cli_main  # noqa: E402
from aminfer_tools.deepseek41 import HardwareTierError  # noqa: E402
from aminfer_tools.gguf import pack_amf_from_gguf  # noqa: E402
from aminfer_tools.glm5 import (  # noqa: E402
    Glm5SynthConfig,
    convert_glm5_checkpoint,
    export_glm5_dflash2,
    export_glm5_mtp,
    generate_synth_glm5_bundle,
    normalize_glm5_dflash2_tensor_name,
    normalize_glm5_tensor_name,
)
from aminfer_tools.gsq_rco import gsq_convert  # noqa: E402
from aminfer_tools.lora import pack_lora_from_peft  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"
GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32


def _verify_amf_with_zig(test: unittest.TestCase, path: pathlib.Path, expected_tensors: int) -> None:
    test.assertTrue(
        AMINFER_BIN.is_file(),
        f"missing compiled binary {AMINFER_BIN} required for verify-amf checks; run `zig build` first",
    )
    res = subprocess.run(
        [str(AMINFER_BIN), "verify-amf", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    test.assertEqual(res.returncode, 0, f"aminfer verify-amf failed on {path}:\n{res.stdout}\n{res.stderr}")
    test.assertIsNotNone(
        re.search(rf"^amf\.tensors\s+{expected_tensors}$", res.stdout, re.MULTILINE),
        f"expected amf.tensors={expected_tensors} in:\n{res.stdout}",
    )


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    offset = 0
    payloads: list[bytes] = []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payloads.append(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hdr_bytes)) % 8
    hdr_padded = hdr_bytes + b" " * pad
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_padded)))
        fh.write(hdr_padded)
        for p in payloads:
            fh.write(p)


def _pack_gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _write_synth_gguf(
    path: pathlib.Path,
    kv_items: list[tuple[str, int, object]],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    buf += GGUF_MAGIC
    buf += struct.pack("<IQQ", 3, len(tensors), len(kv_items))
    for k, vtype, val in kv_items:
        buf += _pack_gguf_str(k)
        buf += struct.pack("<I", vtype)
        if vtype == 4:  # u32
            buf += struct.pack("<I", int(val))
        elif vtype == 6:  # f32
            buf += struct.pack("<f", float(val))
        elif vtype == 7:  # bool
            buf += struct.pack("<B", 1 if val else 0)
        elif vtype == 8:  # str
            buf += _pack_gguf_str(str(val))
        elif vtype == 9:  # array
            elem_t, items = val  # type: ignore[misc]
            buf += struct.pack("<IQ", elem_t, len(items))
            for item in items:
                if elem_t == 4:
                    buf += struct.pack("<I", int(item))
                elif elem_t == 5:
                    buf += struct.pack("<i", int(item))
                elif elem_t == 8:
                    buf += _pack_gguf_str(str(item))
                else:
                    raise ValueError(f"unsupported GGUF array elem type {elem_t}")
        else:
            raise ValueError(f"unsupported GGUF kv type {vtype}")
    off = 0
    for tname, shape_gguf, gtype, raw in tensors:
        buf += _pack_gguf_str(tname)
        buf += struct.pack("<I", len(shape_gguf))
        for d in shape_gguf:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<IQ", gtype, off)
        off += (len(raw) + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
    pad = (-len(buf)) % GGUF_DEFAULT_ALIGNMENT
    buf += b"\x00" * pad
    for _, _, _, raw in tensors:
        buf += raw
        tpad = (-len(raw)) % GGUF_DEFAULT_ALIGNMENT
        buf += b"\x00" * tpad
    path.write_bytes(bytes(buf))


def _f32_bytes(vals: list[float]) -> bytes:
    return struct.pack(f"<{len(vals)}f", *vals)


def _bf16_seq(n: int, *, seed: int = 1) -> bytes:
    rng = random.Random(seed)
    return f32_values_to_bf16_bytes([rng.uniform(-0.25, 0.25) for _ in range(n)])


def _write_minimal_tokenizer(root: pathlib.Path, vocab_size: int) -> None:
    vocab = {f"tok_{i}": i for i in range(vocab_size)}
    tok_obj = {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [],
    }
    (root / "tokenizer.json").write_text(json.dumps(tok_obj), encoding="utf-8")


class Glm5M92Tests(unittest.TestCase):
    def test_name_normalization_under_47_bytes(self) -> None:
        raw_dsa = [
            "model.layers.77.self_attn.indexer.weights_proj.weight",
            "model.layers.77.self_attn.indexer.k_norm.bias",
            "model.layers.77.mlp.gate.e_score_correction_bias",
            "model.layers.78.shared_head.norm.weight",
            "model.layers.78.mlp.shared_experts.down_proj.weight",
        ]
        for r in raw_dsa:
            out = normalize_glm5_tensor_name(
                r,
                model_type="glm_moe_dsa",
                mtp_source_layer=78,
                mtp_standalone=r.startswith("model.layers.78."),
            )
            self.assertLessEqual(len(out.encode("utf-8")), 47, f"{r} -> {out}")

        df2 = normalize_glm5_dflash2_tensor_name("model.layers.4.attention_conv.kernel_projection.weight")
        self.assertEqual(df2, "layers.4.attn_conv.kernel_projection.weight")
        self.assertLessEqual(len(df2.encode("utf-8")), 47)

    def test_synth_glm5_dsa_and_next_bundles_with_zig_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dsa_dir = root / "dsa_bundle"
            dsa_bundle = generate_synth_glm5_bundle(dsa_dir, preset="tiny-glm-dsa", with_mtp=True, seed=11)
            _verify_amf_with_zig(self, dsa_bundle.backbone_amf, len(dsa_bundle.backbone_metas))
            self.assertIsNotNone(dsa_bundle.mtp_amf)
            assert dsa_bundle.mtp_amf is not None
            _verify_amf_with_zig(self, dsa_bundle.mtp_amf, len(dsa_bundle.mtp_metas))

            with AmfReader(dsa_bundle.backbone_amf, verify_header_crc=True, verify_dir_crc=True) as rdr:
                cfg = json.loads(rdr.read_metadata_bytes(META_CONFIG_JSON, verify_crc=True))
                self.assertEqual(cfg["model_type"], "glm_moe_dsa")
                self.assertEqual(cfg["first_k_dense_replace"], 1)
                self.assertEqual(cfg["index_topk_freq"], 4)
                self.assertEqual(cfg["index_skip_topk_offset"], 1)
                self.assertTrue(cfg["index_share_for_mtp_iteration"])
                self.assertTrue(cfg["rope_interleave"])
                self.assertTrue(cfg["indexer_rope_interleave"])
                self.assertIn("layers.0.attn.idx.weights_proj.weight", rdr.by_name)
                self.assertNotIn("layers.3.attn.idx.wk.weight", rdr.by_name)
                self.assertEqual(rdr.by_name["layers.1.mlp.gate.bias"].dtype, DType.F32)

            with AmfReader(dsa_bundle.mtp_amf, verify_header_crc=True, verify_dir_crc=True) as m_rdr:
                mcfg = json.loads(m_rdr.read_metadata_bytes(META_CONFIG_JSON, verify_crc=True))
                self.assertEqual(mcfg["format"], "mtp")
                self.assertEqual(mcfg["mtp_arch"], "glm_moe_dsa")
                self.assertTrue(mcfg["index_share_for_mtp_iteration"])
                self.assertIn("layers.0.enorm.weight", m_rdr.by_name)
                self.assertIn("layers.0.hnorm.weight", m_rdr.by_name)
                self.assertIn("layers.0.eh_proj.weight", m_rdr.by_name)
                self.assertIn("shared_head.norm.weight", m_rdr.by_name)

            nxt_dir = root / "next_bundle"
            nxt_bundle = generate_synth_glm5_bundle(
                nxt_dir,
                preset="tiny-glm5-next-b16",
                vision_mode="sidecar",
                with_mtp=True,
                with_dflash2=True,
                seed=22,
            )
            _verify_amf_with_zig(self, nxt_bundle.backbone_amf, len(nxt_bundle.backbone_metas))
            assert nxt_bundle.vision_amf is not None
            assert nxt_bundle.mtp_amf is not None
            assert nxt_bundle.dflash2_amf is not None
            _verify_amf_with_zig(self, nxt_bundle.vision_amf, len(nxt_bundle.vision_metas))
            _verify_amf_with_zig(self, nxt_bundle.mtp_amf, len(nxt_bundle.mtp_metas))
            _verify_amf_with_zig(self, nxt_bundle.dflash2_amf, len(nxt_bundle.dflash2_metas))

            with AmfReader(nxt_bundle.backbone_amf, verify_header_crc=True, verify_dir_crc=True) as nr:
                ncfg = json.loads(nr.read_metadata_bytes(META_CONFIG_JSON, verify_crc=True))
                self.assertEqual(ncfg["model_type"], "glm5_next")
                self.assertEqual(ncfg["hc_mult"], 4)
                self.assertEqual(ncfg["sinkhorn_iters"], 20)
                self.assertEqual(ncfg["qk_rope_head_dim"], 0)
                self.assertEqual(ncfg["index_kpool"], 4)
                self.assertTrue(ncfg["index_kpool_always_select_tail"])
                self.assertAlmostEqual(ncfg["linear_lower_bound"], -5.0)
                self.assertAlmostEqual(ncfg["swiglu_limit"], 10.0)
                self.assertEqual(nr.by_name["layers.0.linear_attn.q_conv1d.weight"].shape, (128, 4))
                self.assertEqual(nr.by_name["layers.0.linear_attn.q_conv1d.weight"].dtype, DType.F32)
                self.assertEqual(nr.by_name["layers.0.hc_attn_base"].dtype, DType.F32)
                self.assertEqual(nr.by_name["layers.3.attn.idx.kpool_ape"].shape, (4, 32))
                self.assertFalse(any(name.startswith("vision.") for name in nr.by_name))

            with AmfReader(nxt_bundle.dflash2_amf, verify_header_crc=True, verify_dir_crc=True) as dr:
                dcfg = json.loads(dr.read_metadata_bytes(META_CONFIG_JSON, verify_crc=True))
                self.assertEqual(dcfg["draft_kind"], "dflash2")
                self.assertEqual(dcfg["dflash_version"], 2)
                self.assertEqual(dcfg["block_size"], 16)
                self.assertEqual(dcfg["conv_kernel_size"], 2)
                self.assertEqual(dcfg["conv_groups"], 16)
                self.assertEqual(dcfg["selector_top_k"], 8)
                self.assertIn("layers.0.attn_conv.kernel_projection.weight", dr.by_name)
                self.assertIn("candidate_selector.hidden_projection.weight", dr.by_name)

            cli_dir = root / "cli_synth"
            rc = cli_main(
                [
                    "synth-glm5",
                    "--preset",
                    "tiny-glm5-next",
                    "--output-dir",
                    str(cli_dir),
                    "--dflash-block-size",
                    "8",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((cli_dir / "dflash2.amf").is_file())

    def test_glm5_next_fused_qkvbfg_and_concat_conv1d_split_lossless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sf_dir = root / "hf_fused"
            sf_dir.mkdir()
            h = 64
            l_heads = 4
            l_hd = 16
            qkv = l_heads * l_hd  # 64
            vocab = 128
            cfg = {
                "model_type": "glm5_next",
                "architectures": ["Glm5NextForConditionalGeneration"],
                "text_config": {
                    "model_type": "glm5_next_text",
                    "hidden_size": h,
                    "intermediate_size": 128,
                    "moe_intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "first_k_dense_replace": 1,
                    "vocab_size": vocab,
                    "hc_mult": 4,
                    "hc_sinkhorn_iters": 20,
                    "hc_eps": 1e-6,
                    "layer_types": ["linear_attention"],
                    "linear_num_heads": l_heads,
                    "linear_head_dim": l_hd,
                    "linear_conv_kernel_dim": 4,
                    "linear_lower_bound": -5.0,
                },
            }
            (sf_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            _write_minimal_tokenizer(sf_dir, vocab)

            fused_rows = 3 * qkv + l_heads + 2 * l_hd  # 192 + 4 + 32 = 228
            fused_vals = [((r % 29) - 14) * 0.03125 for r in range(fused_rows * h)]
            conv_vals = [((r % 17) - 8) * 0.0625 for r in range(3 * qkv * 4)]
            tensors = {
                "model.language_model.embed_tokens.weight": ("BF16", (vocab, h), _bf16_seq(vocab * h, seed=1)),
                "lm_head.weight": ("BF16", (vocab, h), _bf16_seq(vocab * h, seed=2)),
                "model.language_model.norm.weight": ("BF16", (h,), _bf16_seq(h, seed=3)),
                "model.language_model.layers.0.hc_attn_base": ("BF16", (24,), _bf16_seq(24, seed=4)),
                "model.language_model.layers.0.hc_attn_fn": ("BF16", (24, h), _bf16_seq(24 * h, seed=5)),
                "model.language_model.layers.0.hc_attn_scale": ("BF16", (3,), _bf16_seq(3, seed=6)),
                "model.language_model.layers.0.hc_ffn_base": ("BF16", (24,), _bf16_seq(24, seed=7)),
                "model.language_model.layers.0.hc_ffn_fn": ("BF16", (24, h), _bf16_seq(24 * h, seed=8)),
                "model.language_model.layers.0.hc_ffn_scale": ("BF16", (3,), _bf16_seq(3, seed=9)),
                "model.language_model.layers.0.self_attn.fused_qkvbfg_a_proj.weight": (
                    "BF16",
                    (fused_rows, h),
                    f32_values_to_bf16_bytes(fused_vals),
                ),
                "model.language_model.layers.0.self_attn.conv1d.weight": (
                    "BF16",
                    (3 * qkv, 1, 4),
                    f32_values_to_bf16_bytes(conv_vals),
                ),
                "model.language_model.layers.0.self_attn.f_b_proj.weight": ("BF16", (qkv, l_hd), _bf16_seq(qkv * l_hd, seed=10)),
                "model.language_model.layers.0.self_attn.g_b_proj.weight": ("BF16", (qkv, l_hd), _bf16_seq(qkv * l_hd, seed=11)),
                "model.language_model.layers.0.self_attn.A_log": ("BF16", (l_heads,), _bf16_seq(l_heads, seed=12)),
                "model.language_model.layers.0.self_attn.dt_bias": ("BF16", (qkv,), _bf16_seq(qkv, seed=13)),
                "model.language_model.layers.0.self_attn.o_norm.weight": ("BF16", (l_hd,), _bf16_seq(l_hd, seed=14)),
                "model.language_model.layers.0.self_attn.o_proj.weight": ("BF16", (h, qkv), _bf16_seq(h * qkv, seed=15)),
                "model.language_model.layers.0.mlp.gate_proj.weight": ("BF16", (128, h), _bf16_seq(128 * h, seed=16)),
                "model.language_model.layers.0.mlp.up_proj.weight": ("BF16", (128, h), _bf16_seq(128 * h, seed=17)),
                "model.language_model.layers.0.mlp.down_proj.weight": ("BF16", (h, 128), _bf16_seq(h * 128, seed=18)),
            }
            _write_safetensors(sf_dir / "model.safetensors", tensors)
            out_amf = root / "split.amf"
            metas = pack_amf_from_safetensors(sf_dir, out_amf, model_id="split-test")
            _verify_amf_with_zig(self, out_amf, len(metas))

            with AmfReader(out_amf, verify_header_crc=True, verify_dir_crc=True) as rdr:
                for sub_nm in (
                    "layers.0.linear_attn.q_proj.weight",
                    "layers.0.linear_attn.k_proj.weight",
                    "layers.0.linear_attn.v_proj.weight",
                    "layers.0.linear_attn.b_proj.weight",
                    "layers.0.linear_attn.f_a_proj.weight",
                    "layers.0.linear_attn.g_a_proj.weight",
                    "layers.0.linear_attn.q_conv1d.weight",
                    "layers.0.linear_attn.k_conv1d.weight",
                    "layers.0.linear_attn.v_conv1d.weight",
                ):
                    self.assertIn(sub_nm, rdr.by_name)
                q_e = rdr.by_name["layers.0.linear_attn.q_proj.weight"]
                q_got = bf16_bytes_to_f32_list(rdr.read_blob(q_e)[: q_e.data_bytes])
                self.assertEqual(q_got, fused_vals[: qkv * h])
                ga_e = rdr.by_name["layers.0.linear_attn.g_a_proj.weight"]
                ga_got = bf16_bytes_to_f32_list(rdr.read_blob(ga_e)[: ga_e.data_bytes])
                off_ga = (3 * qkv + l_heads + l_hd) * h
                self.assertEqual(ga_got, fused_vals[off_ga : off_ga + l_hd * h])
                vcv_meta = rdr.by_name["layers.0.linear_attn.v_conv1d.weight"]
                self.assertEqual(vcv_meta.shape, (qkv, 4))
                self.assertEqual(vcv_meta.dtype, DType.F32)
                vcv_bytes = rdr.read_blob(vcv_meta)[: vcv_meta.data_bytes]
                vcv_got = list(struct.unpack(f"<{len(vcv_bytes) // 4}f", vcv_bytes))
                self.assertEqual(vcv_got, conv_vals[2 * qkv * 4 : 3 * qkv * 4])

    def test_glm5_nvfp4_fp8_mlx_and_tier_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            # 1) RedHatAI-style compressed-tensors NVFP4 on glm_moe_dsa (with input_global_scale & 48B indexer key)
            rh_dir = root / "rh_nvfp4"
            rh_dir.mkdir()
            h = 64
            (rh_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "glm_moe_dsa",
                        "architectures": ["GlmMoeDsaForCausalLM"],
                        "hidden_size": h,
                        "intermediate_size": 128,
                        "moe_intermediate_size": 64,
                        "num_hidden_layers": 1,
                        "first_k_dense_replace": 0,
                        "vocab_size": 128,
                        "n_routed_experts": 4,
                        "n_shared_experts": 1,
                        "num_experts_per_tok": 2,
                        "quantization_config": {"quant_method": "compressed-tensors"},
                    }
                ),
                encoding="utf-8",
            )
            _write_minimal_tokenizer(rh_dir, 128)
            _write_safetensors(
                rh_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("BF16", (128, h), _bf16_seq(128 * h, seed=1)),
                    "lm_head.weight": ("BF16", (128, h), _bf16_seq(128 * h, seed=2)),
                    "model.norm.weight": ("BF16", (h,), _bf16_seq(h, seed=3)),
                    "model.layers.0.self_attn.indexer.weights_proj.weight": ("BF16", (4, h), _bf16_seq(4 * h, seed=4)),
                    "model.layers.0.self_attn.q_a_proj.weight_packed": ("U8", (64, h // 2), bytes([0x21] * (64 * h // 2))),
                    "model.layers.0.self_attn.q_a_proj.weight_scale": ("F8_E4M3", (64, h // 16), bytes([0x38] * (64 * h // 16))),
                    "model.layers.0.self_attn.q_a_proj.weight_global_scale": ("F32", (1,), _f32_bytes([2.0])),
                    "model.layers.0.self_attn.q_a_proj.input_global_scale": ("F32", (1,), _f32_bytes([1.5])),
                    "model.layers.0.mlp.gate.weight": ("BF16", (4, h), _bf16_seq(4 * h, seed=5)),
                    "model.layers.0.mlp.gate.e_score_correction_bias": ("F32", (4,), _f32_bytes([0.1, 0.2, 0.3, 0.4])),
                    "model.layers.0.mlp.experts.0.gate_proj.weight_packed": ("U8", (64, h // 2), bytes([0x12] * (64 * h // 2))),
                    "model.layers.0.mlp.experts.0.gate_proj.weight_scale": ("F8_E4M3", (64, h // 16), bytes([0x38] * (64 * h // 16))),
                    "model.layers.0.mlp.experts.0.gate_proj.weight_global_scale": ("F32", (1,), _f32_bytes([1.0])),
                    "model.layers.0.mlp.experts.0.gate_proj.input_global_scale": ("F32", (1,), _f32_bytes([1.0])),
                },
            )
            rh_out = root / "rh.amf"
            rep_rh = convert_glm5_checkpoint(rh_dir, rh_out, strict_vocab_check=False)
            _verify_amf_with_zig(self, rh_out, len(rep_rh.backbone_metas))
            names_rh = {m.name: m for m in rep_rh.backbone_metas}
            self.assertEqual(names_rh["layers.0.self_attn.q_a_proj"].dtype, DType.NVFP4_G16)
            self.assertIn("layers.0.attn.idx.weights_proj.weight", names_rh)

            # 2) NVIDIA modelopt NVFP4 on glm5_next (0-D scalar input_scale & weight_scale_2)
            nv_dir = root / "nv_nvfp4"
            nv_dir.mkdir()
            (nv_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "glm5_next",
                        "architectures": ["Glm5NextForConditionalGeneration"],
                        "text_config": {
                            "model_type": "glm5_next_text",
                            "hidden_size": h,
                            "intermediate_size": 128,
                            "num_hidden_layers": 1,
                            "first_k_dense_replace": 1,
                            "vocab_size": 128,
                            "layer_types": ["indexed_attention"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            _write_minimal_tokenizer(nv_dir, 128)
            _write_safetensors(
                nv_dir / "model.safetensors",
                {
                    "model.language_model.embed_tokens.weight": ("BF16", (128, h), _bf16_seq(128 * h, seed=1)),
                    "lm_head.weight": ("BF16", (128, h), _bf16_seq(128 * h, seed=2)),
                    "model.language_model.norm.weight": ("BF16", (h,), _bf16_seq(h, seed=3)),
                    "model.language_model.layers.0.self_attn.kv_a_proj_with_mqa.weight": ("U8", (64, h // 2), bytes([0x11] * (64 * h // 2))),
                    "model.language_model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale": ("F8_E4M3", (64, h // 16), bytes([0x38] * (64 * h // 16))),
                    "model.language_model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_2": ("F32", (), _f32_bytes([0.25])),
                    "model.language_model.layers.0.self_attn.kv_a_proj_with_mqa.input_scale": ("F32", (), _f32_bytes([0.5])),
                },
            )
            nv_out = root / "nv.amf"
            rep_nv = convert_glm5_checkpoint(nv_dir, nv_out, strict_vocab_check=False)
            _verify_amf_with_zig(self, nv_out, len(rep_nv.backbone_metas))

            # 3) GSQ-RCO protection & 24gb-m4 NVMe cold-expert tier profile + tier admission guard
            dsa_synth = generate_synth_glm5_bundle(
                root / "synth_for_rco",
                preset="tiny-glm-dsa",
                num_layers=2,
                num_experts=8,
                with_mtp=True,
                seed=7,
            )
            rco_out = root / "rco_24gb.amf"
            gsq_metas = gsq_convert(
                dsa_synth.backbone_amf,
                rco_out,
                target_bits=3.6,
                cold_experts={(1, e) for e in range(2, 8)},
                allow_lowbit_q2_retranscode=True,
            ).metas
            _verify_amf_with_zig(self, rco_out, len(gsq_metas))
            by_nm = {m.name: m for m in gsq_metas}
            cold_m = by_nm.get("layers.1.mlp.experts.7.gate_proj") or by_nm["layers.1.mlp.experts.7.gate_proj.weight"]
            self.assertEqual(cold_m.kind, TensorKind.COLD_EXPERT)
            idx_m = by_nm.get("layers.0.attn.idx.wq_b") or by_nm["layers.0.attn.idx.wq_b.weight"]
            self.assertNotEqual(idx_m.dtype, DType.Q2_AFFINE)
            wp_m = by_nm.get("layers.0.attn.idx.weights_proj.weight") or by_nm["layers.0.attn.idx.weights_proj"]
            self.assertIn(wp_m.dtype, (DType.BF16, DType.F16))

            # Tier rejection on 128gb-max for 753B glm_moe_dsa full-wired!
            huge_dir = root / "huge_753b"
            huge_dir.mkdir()
            (huge_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "glm_moe_dsa",
                        "hidden_size": 6144,
                        "intermediate_size": 12288,
                        "moe_intermediate_size": 2048,
                        "num_hidden_layers": 78,
                        "first_k_dense_replace": 3,
                        "vocab_size": 154880,
                        "n_routed_experts": 256,
                        "n_shared_experts": 1,
                        "num_experts_per_tok": 8,
                    }
                ),
                encoding="utf-8",
            )
            _write_safetensors(huge_dir / "model.safetensors", {"model.norm.weight": ("BF16", (h,), _bf16_seq(h))})
            with self.assertRaises(HardwareTierError):
                convert_glm5_checkpoint(huge_dir, root / "should_fail.amf", target_tier="128gb-max")

    def test_glm5_gguf_and_lora_packing_with_zig_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            h = 64
            # 1) GLM-5.3-Flash (`glm5-next`) GGUF with linear_attention + NoPE MLA + 1L MTP
            kv_next = [
                ("general.architecture", 8, "glm5-next"),
                ("general.name", 8, "tiny-glm5-next"),
                ("glm5-next.block_count", 4, 3),  # 2 backbone + 1 MTP
                ("glm5-next.nextn_predict_layers", 4, 1),
                ("glm5-next.embedding_length", 4, h),
                ("glm5-next.feed_forward_length", 4, 128),
                ("glm5-next.attention.head_count", 4, 4),
                ("glm5-next.attention.head_count_kv", 9, (4, [0, 4, 4])),
                ("glm5-next.kda.head_dim", 4, 16),
                ("glm5-next.ssm.conv_kernel", 4, 4),
                ("glm5-next.kda.gate_lower_bound", 6, -5.0),
                ("glm5-next.hyper_connection.count", 4, 4),
                ("glm5-next.hyper_connection.sinkhorn_iterations", 4, 20),
                ("glm5-next.attention.indexer.head_count", 4, 2),
                ("glm5-next.attention.indexer.key_length", 4, 16),
                ("glm5-next.attention.indexer.kpool", 4, 4),
                ("glm5-next.leading_dense_block_count", 4, 1),
            ]
            gguf_tensors = [
                ("token_embd.weight", (h, 64), 1, f32_values_to_f16_bytes([0.01] * (64 * h))),
                ("output_norm.weight", (h,), 0, _f32_bytes([1.0] * h)),
                ("output.weight", (h, 64), 1, f32_values_to_f16_bytes([0.01] * (64 * h))),
                ("blk.0.hc_attn_base", (24,), 0, _f32_bytes([0.1] * 24)),
                ("blk.0.hc_attn_fn", (h, 24), 1, f32_values_to_f16_bytes([0.02] * (24 * h))),
                ("blk.0.attn_q.weight", (h, 64), 1, f32_values_to_f16_bytes([0.02] * (64 * h))),
                ("blk.0.ssm_a", (4,), 0, _f32_bytes([0.5] * 4)),
                ("blk.0.ssm_dt.bias", (64,), 0, _f32_bytes([0.0] * 64)),
                ("blk.0.ssm_conv1d_q.weight", (4, 1, 64), 0, _f32_bytes([0.1] * (64 * 4))),
                ("blk.0.ssm_f_a.weight", (h, 16), 1, f32_values_to_f16_bytes([0.02] * (16 * h))),
                ("blk.1.attn_q_a.weight", (h, 32), 1, f32_values_to_f16_bytes([0.02] * (32 * h))),
                ("blk.1.indexer.proj.weight", (h, 2), 1, f32_values_to_f16_bytes([0.02] * (2 * h))),
                ("blk.1.indexer_compressor_ape", (16, 4), 0, _f32_bytes([0.01] * 64)),
                ("blk.1.exp_probs_b.bias", (4,), 0, _f32_bytes([0.0] * 4)),
                ("blk.2.eh_proj.weight", (2 * h, h), 1, f32_values_to_f16_bytes([0.01] * (2 * h * h))),
                ("blk.2.nextn.enorm.weight", (h,), 0, _f32_bytes([1.0] * h)),
            ]
            gguf_p = root / "glm5_next.gguf"
            _write_synth_gguf(gguf_p, kv_next, gguf_tensors)
            bb_gguf_amf = root / "glm5_next_gguf.amf"
            bb_metas = pack_amf_from_gguf(gguf_p, bb_gguf_amf, dtype_mode="f16")
            _verify_amf_with_zig(self, bb_gguf_amf, len(bb_metas))
            bb_names = {m.name: m for m in bb_metas}
            self.assertIn("layers.0.linear_attn.q_proj.weight", bb_names)
            self.assertEqual(bb_names["layers.0.linear_attn.q_conv1d.weight"].shape, (64, 4))
            self.assertIn("layers.1.attn.idx.weights_proj.weight", bb_names)
            self.assertIn("layers.1.attn.idx.kpool_ape", bb_names)
            self.assertNotIn("layers.2.eh_proj.weight", bb_names)

            mtp_gguf_amf = root / "glm5_next_mtp_gguf.amf"
            mtp_metas = pack_amf_from_gguf(gguf_p, mtp_gguf_amf, dtype_mode="f16", mimo_mtp_mode=True)
            _verify_amf_with_zig(self, mtp_gguf_amf, len(mtp_metas))
            mtp_names = {m.name for m in mtp_metas}
            self.assertIn("layers.0.eh_proj.weight", mtp_names)

            # 2) DFlash2 GGUF
            kv_df2 = [
                ("general.architecture", 8, "dflash"),
                ("dflash.version", 4, 2),
                ("dflash.block_size", 4, 8),
                ("dflash.conv_kernel_size", 4, 2),
                ("dflash.conv_group_size", 4, 16),
                ("dflash.selector_rank", 4, 16),
                ("dflash.selector_top_k", 4, 16),
                ("dflash.target_layer_ids", 9, (5, [0, 1])),
                ("dflash.embedding_length", 4, h),
            ]
            df2_tensors = [
                ("fc.weight", (2 * h, h), 1, f32_values_to_f16_bytes([0.01] * (2 * h * h))),
                ("blk.0.attention_conv.base_kernel", (h, 2, 2), 0, _f32_bytes([0.05] * (4 * h))),
                ("candidate_selector.hidden_projection.weight", (h, 16), 1, f32_values_to_f16_bytes([0.01] * (16 * h))),
            ]
            df2_gguf = root / "dflash2.gguf"
            _write_synth_gguf(df2_gguf, kv_df2, df2_tensors)
            df2_amf = root / "dflash2_gguf.amf"
            df2_metas = pack_amf_from_gguf(df2_gguf, df2_amf, dtype_mode="f16")
            _verify_amf_with_zig(self, df2_amf, len(df2_metas))
            with AmfReader(df2_amf, verify_header_crc=True, verify_dir_crc=True) as dr:
                dcfg = json.loads(dr.read_metadata_bytes(META_CONFIG_JSON, verify_crc=True))
                self.assertEqual(dcfg["draft_kind"], "dflash2")
                self.assertEqual(dcfg["dflash_version"], 2)
                self.assertIn("layers.0.attn_conv.base_kernel", dr.by_name)
                self.assertIn("candidate_selector.hidden_projection.weight", dr.by_name)

            # 3) LoRA PEFT adapter packing for GLM-5 aliases (`f_a_proj`, `kv_a_proj_with_mqa`, `indexer.wq_b`)
            peft_dir = root / "glm5_lora"
            peft_dir.mkdir()
            r = 4
            (peft_dir / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": 8.0}),
                encoding="utf-8",
            )
            _write_safetensors(
                peft_dir / "adapter_model.safetensors",
                {
                    "base_model.model.layers.0.linear_attn.f_a_proj.lora_A.weight": ("BF16", (r, h), _bf16_seq(r * h, seed=1)),
                    "base_model.model.layers.0.linear_attn.f_a_proj.lora_B.weight": ("BF16", (16, r), _bf16_seq(16 * r, seed=2)),
                    "base_model.model.layers.3.self_attn.kv_a_proj_with_mqa.lora_A.weight": ("BF16", (r, h), _bf16_seq(r * h, seed=3)),
                    "base_model.model.layers.3.self_attn.kv_a_proj_with_mqa.lora_B.weight": ("BF16", (32, r), _bf16_seq(32 * r, seed=4)),
                    "base_model.model.layers.3.self_attn.indexer.wq_b.lora_A.weight": ("BF16", (r, 32), _bf16_seq(r * 32, seed=5)),
                    "base_model.model.layers.3.self_attn.indexer.wq_b.lora_B.weight": ("BF16", (32, r), _bf16_seq(32 * r, seed=6)),
                },
            )
            lora_amf = root / "glm5_lora.amf"
            lrep = pack_lora_from_peft(peft_dir, lora_amf, dtype="bf16", layout="row_major")
            _verify_amf_with_zig(self, lora_amf, len(lrep.metas))

    def test_opportunistic_real_tiny_random_hf_glm5_conversions(self) -> None:
        models_env = os.environ.get("AMINFER_MODELS_DIR")
        if not models_env:
            return
        models_root = pathlib.Path(models_env)
        amf_root = models_root / "amf"
        pairs = [
            (
                models_root / "glm-moe-dsa-tiny-random",
                amf_root / "glm-moe-dsa-tiny-random.amf",
                amf_root / "glm-moe-dsa-tiny-random-mtp.amf",
                None,
            ),
            (
                models_root / "tiny-Glm5NextForConditionalGeneration",
                amf_root / "glm5-next-tiny-random.amf",
                amf_root / "glm5-next-tiny-random-mtp.amf",
                amf_root / "glm5-next-tiny-random-vision.amf",
            ),
        ]
        for src_dir, bb_out, mtp_out, vis_out in pairs:
            if not src_dir.is_dir():
                continue
            rep = convert_glm5_checkpoint(
                src_dir,
                bb_out,
                mtp_output=mtp_out,
                vision_sidecar=vis_out,
                vision_block_dtype="bf16",
                strict_vocab_check=False,
            )
            _verify_amf_with_zig(self, rep.backbone_amf, len(rep.backbone_metas))
            if rep.mtp_amf is not None:
                _verify_amf_with_zig(self, rep.mtp_amf, len(rep.mtp_metas))
            if rep.vision_amf is not None:
                _verify_amf_with_zig(self, rep.vision_amf, len(rep.vision_metas))


if __name__ == "__main__":
    unittest.main()
