"""Milestone M7.8 Track W20 tests: full multi-layer ViT + PatchMerger/Aligner .amf packing
and pure-stdlib CoreML ANE `vision-tower` static-bucket export.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DType,
    SyntheticVitConfig,
    TensorKind,
    TensorLayout,
    AmfReader,
    generate_synth_vit_amf,
    generate_synth_vit_checkpoint,
    has_vision_tower,
    is_vision_block_linear,
    is_vision_pinned_float,
    is_vision_tensor,
    normalize_vision_name,
    pack_amf_from_safetensors,
    pack_vision_tower_amf,
    resolve_vision_mode,
)
from aminfer_tools.cli import main as cli_main
from aminfer_tools.coreml import compile_mlpackage, export_vision_tower
from aminfer_tools.deepseek41 import convert_deepseek41_checkpoint
from aminfer_tools.qwen4_exp import pack_qwen4_checkpoint

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def _verify_amf_with_zig(path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    if not AMINFER_BIN.is_file():
        raise AssertionError(f"missing {AMINFER_BIN}; run `zig build` first")
    return subprocess.run(
        [str(AMINFER_BIN), "verify-amf", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


class M78VisionTowerTests(unittest.TestCase):
    def test_canonical_vision_name_normalization_and_classification(self) -> None:
        qwen_keys = [
            "model.visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.bias",
            "model.visual.pos_embed.weight",
            "model.visual.blocks.26.norm1.weight",
            "model.visual.blocks.26.norm1.bias",
            "model.visual.blocks.26.attn.qkv.weight",
            "model.visual.blocks.26.attn.proj.weight",
            "model.visual.blocks.26.norm2.weight",
            "model.visual.blocks.26.mlp.linear_fc1.weight",
            "model.visual.blocks.26.mlp.linear_fc2.weight",
            "model.visual.merger.norm.weight",
            "model.visual.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc2.weight",
        ]
        for raw in qwen_keys:
            self.assertTrue(is_vision_tensor(raw))
            norm = normalize_vision_name(raw)
            self.assertLessEqual(len(norm.encode("utf-8")), 47, f"name too long: {norm!r}")
            self.assertEqual(norm, raw[len("model.") :])

        ds_keys = [
            "vision.patch_embed.proj.weight",
            "vision.blocks.31.norm1.weight",
            "vision.blocks.31.attn.wqkv.weight",
            "vision.blocks.31.attn.wo.weight",
            "vision.blocks.31.mlp.w1.weight",
            "vision.blocks.31.mlp.w2.weight",
            "vision.norm.weight",
            "aligner.w1.weight",
            "aligner.w2.weight",
            "image_start",
            "image_end",
            "image_newline",
        ]
        for raw in ds_keys:
            self.assertTrue(is_vision_tensor(raw))
            norm = normalize_vision_name(raw)
            self.assertLessEqual(len(norm.encode("utf-8")), 47)
            self.assertEqual(norm, raw)

        # Verbose SigLIP / HF vision_tower names exceeding 47 B must compact cleanly <= 47 B.
        verbose = [
            "model.vision_tower.vision_model.encoder.layers.31.self_attn.out_proj.weight",
            "model.vision_tower.vision_model.encoder.layers.31.layer_norm1.weight",
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
            "model.vision_tower.vision_model.post_layernorm.weight",
        ]
        for raw in verbose:
            self.assertTrue(is_vision_tensor(raw))
            norm = normalize_vision_name(raw)
            self.assertLessEqual(len(norm.encode("utf-8")), 47, f"failed to compact {raw!r} -> {norm!r}")
        self.assertEqual(
            normalize_vision_name("model.vision_tower.vision_model.encoder.layers.31.self_attn.out_proj.weight"),
            "vision.blocks.31.attn.proj.weight",
        )

        # Pinned float vs quantizable block linear rules.
        self.assertTrue(is_vision_pinned_float("visual.patch_embed.proj.weight", (128, 1176)))
        self.assertTrue(is_vision_pinned_float("visual.pos_embed.weight", (2304, 1152)))
        self.assertTrue(is_vision_pinned_float("visual.blocks.0.norm1.weight", (128,)))
        self.assertTrue(is_vision_pinned_float("visual.blocks.0.attn.qkv.bias", (384,)))
        self.assertTrue(is_vision_pinned_float("visual.merger.linear_fc1.weight", (512, 512)))
        self.assertTrue(is_vision_pinned_float("aligner.w1.weight", (256, 512)))
        self.assertTrue(is_vision_pinned_float("image_start", (256,)))

        self.assertTrue(is_vision_block_linear("visual.blocks.0.attn.qkv.weight", (384, 128)))
        self.assertTrue(is_vision_block_linear("visual.blocks.26.mlp.linear_fc1.weight", (512, 128)))
        self.assertTrue(is_vision_block_linear("vision.blocks.31.attn.wqkv.weight", (384, 128)))
        self.assertTrue(is_vision_block_linear("vision.blocks.31.mlp.w2.weight", (128, 256)))
        self.assertFalse(is_vision_block_linear("visual.merger.linear_fc2.weight", (256, 512)))

        # Mode resolution: full ViT tower defaults to include; stub defaults to strip.
        self.assertTrue(has_vision_tower(qwen_keys))
        self.assertTrue(has_vision_tower(ds_keys))
        self.assertFalse(has_vision_tower(["visual.encoder.weight", "model.embed_tokens.weight"]))
        self.assertEqual(resolve_vision_mode(qwen_keys), "include")
        self.assertEqual(resolve_vision_mode(qwen_keys, strip_vision=True), "strip")
        self.assertEqual(resolve_vision_mode(qwen_keys, vision_sidecar=pathlib.Path("vision.amf")), "sidecar")
        self.assertEqual(resolve_vision_mode(["visual.encoder.weight"]), "strip")
        self.assertEqual(resolve_vision_mode(["visual.encoder.weight"], include_vision=True), "include")
        with self.assertRaises(ValueError):
            resolve_vision_mode(qwen_keys, include_vision=True, strip_vision=True)
        with self.assertRaises(ValueError):
            resolve_vision_mode(qwen_keys, strip_vision=True, vision_sidecar=pathlib.Path("v.amf"))

        # Opportunistic check on live HF metadata JSONs if present on disk.
        meta_env = os.environ.get("AMINFER_METADATA_DIR")
        models_env = os.environ.get("AMINFER_MODELS_DIR")
        meta_root = (
            pathlib.Path(meta_env)
            if meta_env
            else (pathlib.Path(models_env) / "metadata" if models_env else None)
        )
        if meta_root is not None:
            for meta_dir in (
                meta_root / "Qwen__Qwen3.8-Flash-Next",
                meta_root / "deepseek-ai__DeepSeek-V4.1-Flash-1017",
            ):
                idx_path = meta_dir / "model.safetensors.index.json"
                if idx_path.is_file():
                    wm = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
                    vis_keys = [k for k in wm if is_vision_tensor(k)]
                    self.assertTrue(has_vision_tower(vis_keys))
                    normed = [normalize_vision_name(k) for k in vis_keys]
                    self.assertEqual(len(normed), len(set(normed)), f"duplicate normalized names in {meta_dir.name}")
                    for n in normed:
                        self.assertLessEqual(len(n.encode("utf-8")), 47, f"name > 47B in {meta_dir.name}: {n!r}")

    def test_qwen_vl_27layer_default_include_strip_sidecar_and_5d_patch_embed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            cfg = SyntheticVitConfig(
                family="qwen_vl",
                num_vision_layers=27,
                vision_hidden_size=64,
                vision_intermediate_size=128,
                vision_num_heads=4,
                out_hidden_size=64,
                patch_size=4,
                temporal_patch_size=2,
                spatial_merge_size=2,
                use_conv3d_patch_embed=True,
                include_language_backbone=True,
                backbone_layers=2,
                vocab_size=64,
                dtype="bf16",
                model_id="qwen38-vit27-test",
            )
            ckpt_dir, _ = generate_synth_vit_checkpoint(tmp / "ckpt", cfg, seed=101)

            # 1. Default pack-amf MUST auto-include the full 27-layer vision tower!
            out_default = tmp / "default.amf"
            out_cfg = tmp / "default_config.json"
            rc = cli_main([
                "pack-amf",
                "--input", str(ckpt_dir),
                "--output", str(out_default),
                "--model-id", "qwen38-vit27",
                "--config-output", str(out_cfg),
            ])
            self.assertEqual(rc, 0)
            proc = _verify_amf_with_zig(out_default)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("amf.status       ok", proc.stdout)

            with AmfReader(out_default, verify_header_crc=True, verify_dir_crc=True) as rdr:
                self.assertEqual(out_default.stat().st_size % APPLE_PAGE_SIZE, 0)
                pe = rdr.by_name["visual.patch_embed.proj.weight"]
                # 5D Conv3d [64, 3, 2, 4, 4] flattened to 2D [64, 96].
                self.assertEqual(pe.shape, (64, 3 * 2 * 4 * 4))
                self.assertEqual(pe.dtype, DType.BF16)
                self.assertIn("visual.blocks.0.attn.qkv.weight", rdr.by_name)
                self.assertIn("visual.blocks.26.mlp.linear_fc2.weight", rdr.by_name)
                self.assertEqual(rdr.by_name["visual.blocks.26.attn.qkv.weight"].layer, 26)
                self.assertIn("visual.merger.linear_fc1.weight", rdr.by_name)
                self.assertIn("visual.merger.linear_fc2.weight", rdr.by_name)
                for m in rdr.entries:
                    self.assertEqual(m.blob_offset % APPLE_PAGE_SIZE, 0)
                    self.assertEqual(m.blob_bytes % APPLE_PAGE_SIZE, 0)
                    self.assertLessEqual(len(m.name.encode("utf-8")), 47)

            emitted_cfg = json.loads(out_cfg.read_text(encoding="utf-8"))
            self.assertEqual(emitted_cfg["stripped_vision_tensors"], 0)
            self.assertEqual(emitted_cfg["vision"]["mode"], "embedded")
            self.assertEqual(emitted_cfg["vision"]["num_vision_layers"], 27)
            self.assertEqual(emitted_cfg["vision"]["family"], "qwen_vl")

            # 2. Explicit `--strip-vision` opts out cleanly.
            out_stripped = tmp / "stripped.amf"
            cfg_stripped = tmp / "stripped_config.json"
            rc2 = cli_main([
                "pack-amf",
                "--input", str(ckpt_dir),
                "--output", str(out_stripped),
                "--model-id", "qwen38-novis",
                "--strip-vision",
                "--config-output", str(cfg_stripped),
            ])
            self.assertEqual(rc2, 0)
            with AmfReader(out_stripped, verify_header_crc=True, verify_dir_crc=True) as rdr:
                self.assertFalse(any(is_vision_tensor(m.name) for m in rdr.entries))
            scfg = json.loads(cfg_stripped.read_text(encoding="utf-8"))
            self.assertGreater(scfg["stripped_vision_tensors"], 300)

            # 3. Standalone `--vision-sidecar vision.amf` mode.
            out_bb = tmp / "backbone.amf"
            out_vis = tmp / "vision.amf"
            cfg_sc = tmp / "sidecar_config.json"
            rc3 = cli_main([
                "pack-amf",
                "--input", str(ckpt_dir),
                "--output", str(out_bb),
                "--model-id", "qwen38-sc",
                "--vision-sidecar", str(out_vis),
                "--config-output", str(cfg_sc),
            ])
            self.assertEqual(rc3, 0)
            self.assertTrue(out_vis.is_file())
            self.assertEqual(_verify_amf_with_zig(out_bb).returncode, 0)
            self.assertEqual(_verify_amf_with_zig(out_vis).returncode, 0)
            with AmfReader(out_bb, verify_header_crc=True, verify_dir_crc=True) as r_bb:
                self.assertFalse(any(is_vision_tensor(m.name) for m in r_bb.entries))
            with AmfReader(out_vis, verify_header_crc=True, verify_dir_crc=True) as r_vis:
                self.assertTrue(all(is_vision_tensor(m.name) for m in r_vis.entries))
                self.assertIn("visual.blocks.26.attn.proj.weight", r_vis.by_name)
                self.assertIn("visual.merger.linear_fc2.weight", r_vis.by_name)
            sc_cfg = json.loads(cfg_sc.read_text(encoding="utf-8"))
            self.assertEqual(sc_cfg["vision"]["mode"], "sidecar")
            self.assertEqual(sc_cfg["vision"]["amf_file"], "vision.amf")
            self.assertEqual(sc_cfg["vision"]["num_vision_layers"], 27)

    def test_deepseek_v41_32layer_vision_and_aligner_packing_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            cfg = SyntheticVitConfig(
                family="deepseek_v41",
                num_vision_layers=32,
                vision_hidden_size=64,
                vision_intermediate_size=128,
                vision_num_heads=4,
                out_hidden_size=64,
                patch_size=4,
                temporal_patch_size=1,
                spatial_merge_size=2,
                include_language_backbone=True,
                backbone_layers=2,
                vocab_size=64,
                dtype="bf16",
                model_id="ds41-vit32-test",
            )
            ckpt_dir, _ = generate_synth_vit_checkpoint(tmp / "ckpt", cfg, seed=202)

            # 1. Default include under q3_affine backbone -> pinned vision stays BF16, blocks upgrade to Q4_AFFINE!
            out_q3 = tmp / "ds41_q3.amf"
            rep = convert_deepseek41_checkpoint(
                ckpt_dir,
                out_q3,
                model_id="ds41-vit32",
                target_dtype="q3_affine",
            )
            self.assertEqual(rep.stripped_vision_tensors, 0)
            self.assertGreater(rep.included_vision_tensors, 250)
            self.assertEqual(_verify_amf_with_zig(out_q3).returncode, 0)
            with AmfReader(out_q3, verify_header_crc=True, verify_dir_crc=True) as rdr:
                # Pinned vision tensors stay BF16.
                self.assertEqual(rdr.by_name["vision.patch_embed.proj.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["vision.norm.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["vision.blocks.0.norm1.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["aligner.w1.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["aligner.w2.weight"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["image_start"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["image_end"].dtype, DType.BF16)
                self.assertEqual(rdr.by_name["image_newline"].dtype, DType.BF16)
                # Vision block linears stay at Q4_AFFINE (never degraded to Q3_AFFINE!).
                self.assertEqual(rdr.by_name["vision.blocks.0.attn.wqkv"].dtype, DType.Q4_AFFINE)
                self.assertEqual(rdr.by_name["vision.blocks.31.mlp.w2"].dtype, DType.Q4_AFFINE)
                # Backbone linears still quantize to Q3_AFFINE.
                self.assertEqual(rdr.by_name["layers.0.self_attn.q_proj"].dtype, DType.Q3_AFFINE)

            # 2. CLI `pack-deepseek41 --vision-sidecar vision.amf --dtype q4_affine`.
            out_bb = tmp / "ds41_bb.amf"
            out_vis = tmp / "ds41_vision.amf"
            out_cfg = tmp / "ds41_config.json"
            rc = cli_main([
                "pack-deepseek41",
                "--input", str(ckpt_dir),
                "--output", str(out_bb),
                "--model-id", "ds41-sc",
                "--dtype", "q4_affine",
                "--vision-sidecar", str(out_vis),
                "--config-output", str(out_cfg),
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(_verify_amf_with_zig(out_bb).returncode, 0)
            self.assertEqual(_verify_amf_with_zig(out_vis).returncode, 0)
            with AmfReader(out_vis, verify_header_crc=True, verify_dir_crc=True) as rv:
                self.assertIn("vision.blocks.31.attn.wo", rv.by_name)
                self.assertEqual(rv.by_name["vision.blocks.31.attn.wo"].dtype, DType.Q4_AFFINE)
                self.assertEqual(rv.by_name["aligner.w1.weight"].dtype, DType.BF16)
                self.assertEqual(rv.by_name["vision.patch_embed.proj.weight"].dtype, DType.BF16)
            cfg_obj = json.loads(out_cfg.read_text(encoding="utf-8"))
            self.assertEqual(cfg_obj["vision"]["mode"], "sidecar")
            self.assertEqual(cfg_obj["vision"]["num_vision_layers"], 32)
            self.assertEqual(cfg_obj["vision"]["family"], "deepseek_v41")

    def test_w14_gsq_rco_vision_precision_policy_and_cli_gsq_convert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            cfg = SyntheticVitConfig(
                family="qwen_vl",
                num_vision_layers=4,
                vision_hidden_size=64,
                vision_intermediate_size=128,
                vision_num_heads=4,
                out_hidden_size=64,
                patch_size=4,
                temporal_patch_size=2,
                spatial_merge_size=2,
                include_language_backbone=True,
                backbone_layers=2,
                vocab_size=64,
                dtype="f16",
                model_id="qwen-gsq-vis",
            )
            ckpt_dir, _ = generate_synth_vit_checkpoint(tmp / "ckpt", cfg, seed=303)

            # 1. `pack-amf --quant-profile gsq-rco --target-bits 3.5` (auto-includes ViT with W20 policy).
            out_gsq = tmp / "gsq_embedded.amf"
            manifest = tmp / "gsq_manifest.json"
            rc = cli_main([
                "pack-amf",
                "--input", str(ckpt_dir),
                "--output", str(out_gsq),
                "--model-id", "qwen-gsq-vis",
                "--quant-profile", "gsq-rco",
                "--target-bits", "3.5",
                "--manifest", str(manifest),
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(_verify_amf_with_zig(out_gsq).returncode, 0)
            self.assertTrue(manifest.is_file())
            m_json = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("vision_tower", m_json)
            self.assertEqual(m_json["vision_tower"]["num_vision_layers"], 4)

            with AmfReader(out_gsq, verify_header_crc=True, verify_dir_crc=True) as rdr:
                # Pinned vision tensors stay F16 (never quantized!).
                for pinned_name in (
                    "visual.patch_embed.proj.weight",
                    "visual.pos_embed.weight",
                    "visual.blocks.0.norm1.weight",
                    "visual.blocks.0.norm2.weight",
                    "visual.merger.norm.weight",
                    "visual.merger.linear_fc1.weight",
                    "visual.merger.linear_fc2.weight",
                ):
                    self.assertEqual(rdr.by_name[pinned_name].dtype, DType.F16, msg=pinned_name)
                # Vision block 2D linears quantize cleanly to Q4_AFFINE (never Q2/Q3!).
                for blk in range(4):
                    for suffix in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
                        tname = f"visual.blocks.{blk}.{suffix}"
                        self.assertEqual(rdr.by_name[tname].dtype, DType.Q4_AFFINE, msg=tname)
                        self.assertEqual(rdr.by_name[tname].layout, TensorLayout.ROW_MAJOR)
                        rdr.read_blob(rdr.by_name[tname], verify_crc=True)

            # 2. `gsq-convert --vision-block-dtype f16` keeps vision blocks in F16 too.
            out_gsq_f16 = tmp / "gsq_vis_f16.amf"
            rc2 = cli_main([
                "gsq-convert",
                "--input", str(ckpt_dir),
                "--output", str(out_gsq_f16),
                "--target-bits", "3.5",
                "--vision-block-dtype", "f16",
            ])
            self.assertEqual(rc2, 0)
            self.assertEqual(_verify_amf_with_zig(out_gsq_f16).returncode, 0)
            with AmfReader(out_gsq_f16, verify_header_crc=True, verify_dir_crc=True) as rdr:
                self.assertEqual(rdr.by_name["visual.blocks.0.attn.qkv.weight"].dtype, DType.F16)
                self.assertEqual(rdr.by_name["visual.merger.linear_fc1.weight"].dtype, DType.F16)

    def test_coreml_vision_tower_static_buckets_and_real_compiler_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)

            # 1. Synthetic multi-bucket CLI export (`--component vision-tower`, inner-product, GELU).
            out_dir = tmp / "coreml_synth"
            compile_dir = tmp / "compiled_synth"
            rc = cli_main([
                "export-coreml",
                "--component", "vision-tower",
                "--output", str(out_dir),
                "--token-buckets", "64,128",
                "--num-vision-layers", "2",
                "--vision-hidden-size", "64",
                "--vision-intermediate-size", "128",
                "--out-hidden-size", "64",
                "--spatial-merge-size", "2",
                "--mlp-act", "gelu",
                "--verify-compile", str(compile_dir),
            ])
            self.assertEqual(rc, 0)
            pkg64 = out_dir / "vision_tower_t64.mlpackage"
            pkg128 = out_dir / "vision_tower_t128.mlpackage"
            self.assertTrue((pkg64 / "Manifest.json").is_file())
            self.assertTrue((pkg128 / "Manifest.json").is_file())
            self.assertTrue((compile_dir / "vision_tower_t64.mlmodelc").is_dir())
            self.assertTrue((compile_dir / "vision_tower_t128.mlmodelc").is_dir())

            meta64 = json.loads((pkg64 / "Data" / "com.apple.CoreML" / "Metadata.json").read_text(encoding="utf-8"))
            ud = meta64["MLModelCreatorDefinedKey"]
            self.assertEqual(ud["aminfer_kind"], "vision-tower")
            self.assertEqual(ud["token_bucket"], "64")
            self.assertEqual(ud["iosurface_row_align"], "32")
            self.assertEqual(ud["g14_zero_copy"], "true")

            # 2. Real weights from Q4-quantized Qwen-VL `vision.amf` sidecar -> conv1x1 CoreML `.mlpackage`.
            qwen_vis_amf = tmp / "qwen_vis.amf"
            generate_synth_vit_amf(
                qwen_vis_amf,
                SyntheticVitConfig(
                    family="qwen_vl",
                    num_vision_layers=2,
                    vision_hidden_size=64,
                    vision_intermediate_size=128,
                    out_hidden_size=96,
                    patch_size=4,
                    spatial_merge_size=2,
                    mlp_act="swiglu",
                    dtype="bf16",
                    model_id="qwen-vis-swiglu",
                ),
                seed=404,
                quant_profile="gsq-rco",
                embed_metadata=True,
            )
            self.assertEqual(_verify_amf_with_zig(qwen_vis_amf).returncode, 0)
            pkg_qwen = tmp / "qwen_vision_t64.mlpackage"
            res_qwen = export_vision_tower(
                pkg_qwen,
                amf_path=qwen_vis_amf,
                token_buckets=(64,),
                op="conv1x1",
            )
            self.assertEqual(res_qwen.kind, "vision-tower")
            self.assertEqual(res_qwen.op, "conv1x1")
            self.assertEqual(res_qwen.inputs[0].shape, (64, 1, 64))
            self.assertEqual(res_qwen.outputs[0].shape, (96, 1, 64))
            self.assertEqual(res_qwen.metadata["mlp_act"], "swiglu")
            self.assertEqual(res_qwen.metadata["vision_family"], "qwen_vl")
            compiled_qwen = compile_mlpackage(pkg_qwen, tmp / "compiled_qwen")
            self.assertTrue(compiled_qwen.is_dir())

            # 3. Real weights from DeepSeek-V4.1 `vision.amf` sidecar -> inner-product CoreML `.mlpackage`.
            ds_vis_amf = tmp / "ds_vis.amf"
            rc_svit = cli_main([
                "synth-vit",
                "--output", str(ds_vis_amf),
                "--family", "deepseek_v41",
                "--num-layers", "2",
                "--vision-hidden-size", "64",
                "--vision-intermediate-size", "128",
                "--out-hidden-size", "64",
                "--patch-size", "4",
                "--quant-profile", "gsq-rco",
            ])
            self.assertEqual(rc_svit, 0)
            pkg_ds = tmp / "ds_vision_t64.mlpackage"
            res_ds = export_vision_tower(
                pkg_ds,
                amf_path=ds_vis_amf,
                token_buckets=(64,),
                op="inner-product",
            )
            self.assertEqual(res_ds.metadata["vision_family"], "deepseek_v41")
            compiled_ds = compile_mlpackage(pkg_ds, tmp / "compiled_ds")
            self.assertTrue(compiled_ds.is_dir())

            # 4. IOSurface alignment rejection for unaligned conv1x1 token bucket.
            with self.assertRaises(ValueError):
                export_vision_tower(tmp / "bad.mlpackage", token_buckets=(50,), op="conv1x1")


if __name__ == "__main__":
    unittest.main()
