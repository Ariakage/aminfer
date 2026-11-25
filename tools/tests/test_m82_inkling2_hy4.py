"""Comprehensive Milestone M8.2 tests for W23 Inkling 2 & Hy4 converters, D14 DFlare/DFly2 packers, C26 SME2/Engram, GSQ-RCO v2, and synth CLIs."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pathlib
import random
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout

from aminfer_tools.amf import (
    AmfReader,
    DType,
    TensorKind,
    TensorLayout,
    _classify_tensor,
    e4m3_byte_to_f32,
    is_inkling2_controller_tensor,
    is_vision_tensor,
    pack_amf_from_safetensors,
)
from aminfer_tools.cli import main as cli_main
from aminfer_tools.gguf import export_dflare_bundle_from_gguf, pack_amf_from_gguf
from aminfer_tools.gsq_rco import (
    _PROTECTED_HIGH_PRECISION_ROLES,
    _role_of_tensor,
    gsq_convert,
)
from aminfer_tools.hy4 import (
    convert_hy4_checkpoint,
    generate_synth_hy4_bundle,
    validate_hf_metadata_dir as validate_hy4_metadata_dir,
)
from aminfer_tools.inkling2 import (
    INKLING2_SYNTH_PRESETS,
    convert_inkling2_checkpoint,
    generate_synth_inkling2_bundle,
    resolve_inkling2_tier_profile,
    validate_hf_metadata_dir as validate_inkling2_metadata_dir,
)

def _resolve_metadata_root() -> pathlib.Path | None:
    meta_env = os.environ.get("AMINFER_METADATA_DIR")
    if meta_env:
        return pathlib.Path(meta_env)
    models_env = os.environ.get("AMINFER_MODELS_DIR")
    if models_env:
        return pathlib.Path(models_env) / "metadata"
    return None


def _resolve_hy4_tiny_random() -> pathlib.Path | None:
    models_env = os.environ.get("AMINFER_MODELS_DIR")
    if models_env:
        return pathlib.Path(models_env) / "hy4-preview-tiny-random"
    return None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"
REAL_METADATA_ROOT = _resolve_metadata_root()
REAL_HY4_TINY_RANDOM = _resolve_hy4_tiny_random()


def _f32_to_bf16_bytes(vals: list[float]) -> bytes:
    out = bytearray()
    for v in vals:
        (u,) = struct.unpack("<I", struct.pack("<f", float(v)))
        rounding_bias = 0x7FFF + ((u >> 16) & 1)
        out.extend(struct.pack("<H", ((u + rounding_bias) >> 16) & 0xFFFF))
    return bytes(out)


def _bf16_bytes_to_f32(raw: bytes) -> list[float]:
    n = len(raw) // 2
    out: list[float] = []
    for (h16,) in struct.iter_unpack("<H", raw[: n * 2]):
        (val,) = struct.unpack("<f", struct.pack("<I", h16 << 16))
        out.append(val)
    return out


def _f32_to_f16_bytes(vals: list[float]) -> bytes:
    return b"".join(struct.pack("<e", float(v)) for v in vals)


def _f32_to_e4m3_byte(x: float) -> int:
    if not math.isfinite(x):
        return 0x7F
    if x == 0.0:
        return 0x80 if math.copysign(1.0, x) < 0.0 else 0x00
    sign = 0x80 if x < 0.0 else 0x00
    ax = min(abs(x), 448.0)
    best_code = 0
    best_err = float("inf")
    for mag_code in range(1, 127):
        cand = e4m3_byte_to_f32(mag_code)
        err = abs(ax - cand)
        if err < best_err:
            best_err = err
            best_code = mag_code
    return sign | best_code


def _make_f32_vals(seed: int, count: int, scale: float = 0.05, *, centered_one: bool = False) -> list[float]:
    rng = random.Random(seed)
    if centered_one:
        return [1.0 + rng.uniform(-scale, scale) for _ in range(count)]
    return [rng.uniform(-scale, scale) for _ in range(count)]


def _write_safetensors(
    path: pathlib.Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    *,
    metadata: dict[str, str] | None = None,
) -> bytes:
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    chunks: list[bytes] = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return blob


def _pack_gguf_kv_value(val_type: int, value: object) -> bytes:
    if val_type == 4:  # UINT32
        return struct.pack("<I", int(value))
    if val_type == 5:  # INT32
        return struct.pack("<i", int(value))
    if val_type == 6:  # FLOAT32
        return struct.pack("<f", float(value))
    if val_type == 7:  # BOOL
        return struct.pack("<B", 1 if value else 0)
    if val_type == 8:  # STRING
        b = str(value).encode("utf-8")
        return struct.pack("<Q", len(b)) + b
    if val_type == 9:  # ARRAY -> (elem_type, items)
        elem_type, items = value  # type: ignore[misc]
        buf = bytearray(struct.pack("<IQ", elem_type, len(items)))
        for item in items:
            buf.extend(_pack_gguf_kv_value(elem_type, item))
        return bytes(buf)
    raise ValueError(f"unsupported test gguf kv type {val_type}")


def _write_gguf_file(
    path: pathlib.Path,
    *,
    version: int,
    kv: list[tuple[str, int, object]],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    alignment: int = 32,
) -> None:
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", version))
    buf.extend(struct.pack("<QQ", len(tensors), len(kv)))
    for key, val_type, val in kv:
        kb = key.encode("utf-8")
        buf.extend(struct.pack("<Q", len(kb)) + kb)
        buf.extend(struct.pack("<I", val_type))
        buf.extend(_pack_gguf_kv_value(val_type, val))

    data_chunks = bytearray()
    rel_offsets: list[int] = []
    for _name, _dims, _t_type, payload in tensors:
        pad = (alignment - (len(data_chunks) % alignment)) % alignment
        data_chunks.extend(b"\x00" * pad)
        rel_offsets.append(len(data_chunks))
        data_chunks.extend(payload)

    for (name, dims, t_type, _payload), rel_off in zip(tensors, rel_offsets):
        nb = name.encode("utf-8")
        buf.extend(struct.pack("<Q", len(nb)) + nb)
        buf.extend(struct.pack("<I", len(dims)))
        for d in dims:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<IQ", t_type, rel_off))

    hdr_pad = (alignment - (len(buf) % alignment)) % alignment
    buf.extend(b"\x00" * hdr_pad)
    buf.extend(data_chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))


def _verify_amf_with_zig(test: unittest.TestCase, path: pathlib.Path, expected_tensors: int) -> None:
    if not AMINFER_BIN.is_file():
        return
    proc = subprocess.run([str(AMINFER_BIN), "verify-amf", str(path)], capture_output=True, text=True, check=False)
    test.assertEqual(proc.returncode, 0, msg=f"verify-amf failed for {path}:\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    test.assertIn("amf.status       ok", proc.stdout)
    test.assertIn(f"amf.tensors      {expected_tensors}\n", proc.stdout)


class TestSynthInkling2(unittest.TestCase):
    def test_synth_mini_bundle_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cfg = INKLING2_SYNTH_PRESETS["tiny-inkling2-mini"]
            b1 = generate_synth_inkling2_bundle(root / "run1", config=cfg, seed=7)
            b2 = generate_synth_inkling2_bundle(root / "run2", config=cfg, seed=7)

            self.assertEqual(
                hashlib.sha256(b1.model_amf.read_bytes()).digest(),
                hashlib.sha256(b2.model_amf.read_bytes()).digest(),
            )
            self.assertIsNotNone(b1.dflare_amf)
            self.assertIsNotNone(b1.dflare_config_path)
            self.assertIsNotNone(b1.vision_amf)
            assert b1.dflare_amf is not None and b1.dflare_config_path is not None and b1.vision_amf is not None

            self.assertEqual(
                hashlib.sha256(b1.dflare_amf.read_bytes()).digest(),
                hashlib.sha256(b2.dflare_amf.read_bytes()).digest(),
            )

            with AmfReader(b1.model_amf) as reader:
                metas = list(reader.entries)
                names = {m.name for m in metas}
                self.assertTrue(all(len(m.name.encode("utf-8")) <= 47 for m in metas))
                self.assertTrue(all(m.blob_offset % 16384 == 0 for m in metas))
                self.assertIn("middle_cycle_cross_pass_queries", names)
                self.assertIn("middle_cycle_cross_pass_kv_queries", names)

                controllers = [m for m in metas if is_inkling2_controller_tensor(m.name)]
                self.assertGreater(len(controllers), 0)
                self.assertTrue(
                    all(m.layout == TensorLayout.SME2_PANEL64 for m in controllers),
                    msg=f"expected controllers to default to SME2_PANEL64, got {[(c.name, c.layout) for c in controllers[:4]]}",
                )
                sconv = [m for m in metas if "sconv" in m.name]
                self.assertGreater(len(sconv), 0)
                cold = [m for m in metas if m.kind == TensorKind.COLD_EXPERT]
                self.assertGreater(len(cold), 0)

                cfg_blob = reader.read_metadata_bytes("__meta.config_json")
                self.assertIsNotNone(cfg_blob)
                cfg_meta = json.loads(cfg_blob.decode("utf-8"))
                self.assertEqual(cfg_meta["model_type"], "inkling2_loop")
                self.assertEqual(cfg_meta["operator_axes"]["loop_controllers"], "sme2_panel64")
                self.assertEqual(cfg_meta["loop_schedule"]["effective_num_steps"], 10)

            dflare_cfg = json.loads(b1.dflare_config_path.read_text(encoding="utf-8"))
            self.assertEqual(dflare_cfg["draft_kind"], "dflare")
            self.assertEqual(dflare_cfg["num_hidden_layers"], 3)
            self.assertEqual(dflare_cfg["block_size"], 8)
            self.assertEqual(dflare_cfg["markov_rank"], 256)
            self.assertEqual(dflare_cfg["markov_head_type"], "rnn_hidden")
            with AmfReader(b1.dflare_amf) as dreader:
                dnames = {m.name for m in dreader.entries}
                self.assertIn("mask_embed", dnames)
                self.assertIn("markov_head.markov_w1.weight", dnames)
                self.assertIn("markov_head.rnn_proj.weight", dnames)

            with AmfReader(b1.vision_amf) as vreader:
                vnames = {m.name for m in vreader.entries}
                self.assertTrue(any(n.startswith("visual.blocks.22.") for n in vnames))

            _verify_amf_with_zig(self, b1.model_amf, len(b1.backbone_metas))
            _verify_amf_with_zig(self, b1.dflare_amf, len(b1.dflare_metas))
            _verify_amf_with_zig(self, b1.vision_amf, len(b1.vision_metas))

    def test_synth_small_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "small_cli"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main([
                    "synth-inkling2",
                    "--preset",
                    "tiny-inkling2-small",
                    "--output-dir",
                    str(out_dir),
                ])
            self.assertEqual(rc, 0)
            self.assertIn("synthesized inkling2", buf.getvalue())
            dflare_cfg = json.loads((out_dir / "dflare_config.json").read_text(encoding="utf-8"))
            self.assertEqual(dflare_cfg["num_hidden_layers"], 5)
            model_cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(model_cfg["model_type"], "inkling2")
            with AmfReader(out_dir / "model.amf") as r:
                backbone_count = len(r.entries)
                self.assertFalse(any("controllers." in m.name for m in r.entries))
            with AmfReader(out_dir / "dflare.amf") as dr:
                df_count = len(dr.entries)
                self.assertTrue(any(m.name.startswith("layers.4.") for m in dr.entries))
            _verify_amf_with_zig(self, out_dir / "model.amf", backbone_count)
            _verify_amf_with_zig(self, out_dir / "dflare.amf", df_count)


class TestSynthHy4(unittest.TestCase):
    def test_synth_hy4_and_dfly2_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "hy4_base"
            b = generate_synth_hy4_bundle(out_dir, preset="tiny-hy4", seed=11)
            self.assertEqual(b.engram_paths, [])
            self.assertIsNotNone(b.dfly2_amf)
            self.assertIsNotNone(b.dfly2_config_path)
            assert b.dfly2_amf is not None and b.dfly2_config_path is not None

            with AmfReader(b.model_amf) as reader:
                names = {m.name for m in reader.entries}
                self.assertTrue(all(len(m.name.encode("utf-8")) <= 47 for m in reader.entries))
                self.assertIn("layers.0.input_layernorm.weight", names)
                self.assertTrue(any("learnable_sink_param" in n for n in names))
                self.assertTrue(any("linear_gate" in n for n in names))
                self.assertTrue(any("hc_" in n for n in names))
                self.assertTrue(any("kv_a_proj_with_mqa" in n or "kv_a_mqa" in n for n in names))
                self.assertTrue(any("experts." in n and ".down_proj." in n for n in names))

            dfly_cfg = json.loads(b.dfly2_config_path.read_text(encoding="utf-8"))
            self.assertEqual(dfly_cfg["draft_kind"], "dfly2")
            self.assertEqual(dfly_cfg["num_hidden_layers"], 5)
            self.assertEqual(dfly_cfg["block_size"], 7)
            _verify_amf_with_zig(self, b.model_amf, len(b.backbone_metas))
            _verify_amf_with_zig(self, b.dfly2_amf, len(b.dfly2_metas))

    def test_synth_hy4_flash_engram_shards_and_latent_moe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "hy4_flash_cli"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main([
                    "synth-hy4",
                    "--preset",
                    "tiny-hy4-flash",
                    "--engram-shards",
                    "2",
                    "--output-dir",
                    str(out_dir),
                ])
            self.assertEqual(rc, 0)
            self.assertIn("synthesized hy_v4_flash", buf.getvalue())

            cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
            tc = cfg["text_config"]
            self.assertEqual(tc["compressed_indexer_compress_ratio"], 4)
            self.assertEqual(tc["compressed_indexer_group_topk"], 4)
            self.assertEqual(tc["compressed_indexer_candidate_topk"], 8)
            self.assertAlmostEqual(float(tc["routed_scaling_factor"]), 3.998, places=3)

            shard0 = out_dir / "engram-00000-of-00002.amf"
            shard1 = out_dir / "engram-00001-of-00002.amf"
            self.assertTrue(shard0.is_file() and shard1.is_file())
            for sp in (shard0, shard1):
                with AmfReader(sp) as er:
                    eng_entries = list(er.entries)
                self.assertGreater(len(eng_entries), 0)
                for em in eng_entries:
                    self.assertEqual(em.kind, TensorKind.SPARSE_TABLE)
                    self.assertEqual(em.dtype, DType.FP8_E4M3_ROW)
                _verify_amf_with_zig(self, sp, len(eng_entries))

            with AmfReader(out_dir / "model.amf") as mr:
                bb_entries = list(mr.entries)
            mnames = {m.name for m in bb_entries}
            self.assertTrue(any("routed_expert_up_proj" in n for n in mnames))
            self.assertTrue(any("combine_output_norm" in n for n in mnames))
            self.assertTrue(any("e_score_correction_bias" in n for n in mnames))
            self.assertFalse(any("multi_head_embedding" in n for n in mnames))
            self.assertFalse(any(is_vision_tensor(n) for n in mnames))
            _verify_amf_with_zig(self, out_dir / "model.amf", len(bb_entries))

            with AmfReader(out_dir / "vision.amf") as vr:
                vis_entries = list(vr.entries)
            vnames = {m.name for m in vis_entries}
            self.assertTrue(any(n.startswith("vision.blocks.26.") for n in vnames))
            _verify_amf_with_zig(self, out_dir / "vision.amf", len(vis_entries))

    def test_synth_hy4_vl_40l_vision_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "hy4_vl"
            b = generate_synth_hy4_bundle(out_dir, preset="tiny-hy4-vl", seed=13)
            self.assertIsNotNone(b.vision_amf)
            assert b.vision_amf is not None
            with AmfReader(b.vision_amf) as vr:
                vnames = {m.name for m in vr.entries}
                self.assertTrue(any(n.startswith("vision.blocks.39.") for n in vnames))
            _verify_amf_with_zig(self, b.vision_amf, len(b.vision_metas))


class TestInkling2FormatsAndConverters(unittest.TestCase):
    def test_nvfp4_mxfp8_controllers_cold_experts_vision_sidecar_and_companion_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            src_dir = root / "hf_inkling2_mini"
            src_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "architectures": ["Inkling2LoopForConditionalGeneration"],
                "model_type": "inkling2_loop",
                "text_config": {
                    "model_type": "inkling2_loop_text",
                    "hidden_size": 128,
                    "num_hidden_layers": 2,
                    "vocab_size": 256,
                    "unpadded_vocab_size": 250,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 64,
                    "intermediate_size": 128,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "sliding_window_size": 1024,
                    "local_layer_ids": [0],
                    "route_scale": 8.0,
                    "loop_config": {"core_start": 0, "core_end": 1, "num_passes": 3, "gate_rank": 64},
                },
                "mtp_config": {
                    "num_hidden_layers": 3,
                    "hidden_size": 128,
                    "block_size": 8,
                    "markov_rank": 256,
                    "markov_head_type": "rnn_hidden",
                    "target_layer_ids": [0, 1],
                },
                "vision_config": {"model_type": "inkling2_vision", "hidden_size": 64, "num_hidden_layers": 23},
            }
            (src_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            # NVFP4 packed weight (`[64, 128]` -> packed `[64, 64]` U8, scale `[64, 8]` U8, global scale F32).
            nvfp4_packed = bytes((i * 17 + 3) & 0xFF for i in range(64 * 64))
            nvfp4_scale = bytes([0x38] * (64 * 8))
            nvfp4_gs = struct.pack("<f", 0.5)

            # MXFP8 `[128, 64]` (`F8_E4M3` + `U8` UE8M0 `[128, 2]`).
            mxfp8_w = bytes(_f32_to_e4m3_byte(0.25) for _ in range(128 * 64))
            mxfp8_sc = bytes([127] * (128 * 2))

            bb_tensors: dict[str, tuple[str, list[int], bytes]] = {
                "model.language_model.embed_tokens.weight": ("BF16", [256, 128], _f32_to_bf16_bytes(_make_f32_vals(1, 256 * 128))),
                "model.language_model.norm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(2, 128, centered_one=True))),
                "lm_head.weight": ("BF16", [256, 128], _f32_to_bf16_bytes(_make_f32_vals(3, 256 * 128))),
                "model.language_model.layers.0.input_layernorm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(4, 128, centered_one=True))),
                "model.language_model.layers.0.self_attn.q_proj.weight_packed": ("U8", [64, 64], nvfp4_packed),
                "model.language_model.layers.0.self_attn.q_proj.weight_scale": ("U8", [64, 8], nvfp4_scale),
                "model.language_model.layers.0.self_attn.q_proj.weight_global_scale": ("F32", [1], nvfp4_gs),
                "model.language_model.layers.0.self_attn.o_proj.weight": ("F8_E4M3", [128, 64], mxfp8_w),
                "model.language_model.layers.0.self_attn.o_proj.weight_scale": ("U8", [128, 2], mxfp8_sc),
                "model.language_model.layers.0.self_attn.sconv.convolution.weight": ("BF16", [128, 4], _f32_to_bf16_bytes(_make_f32_vals(5, 128 * 4))),
                "model.language_model.layers.0.controllers.attn.down_weight": ("BF16", [128, 64], _f32_to_bf16_bytes(_make_f32_vals(6, 128 * 64))),
                "model.language_model.layers.0.controllers.attn.forget_weight": ("BF16", [64, 128], _f32_to_bf16_bytes(_make_f32_vals(7, 64 * 128))),
                "model.language_model.layers.0.controllers.attn.gain_weight": ("BF16", [64, 128], _f32_to_bf16_bytes(_make_f32_vals(8, 64 * 128))),
                "model.language_model.layers.0.mlp.gate.weight": ("F32", [4, 128], struct.pack("<" + "f" * (4 * 128), *_make_f32_vals(9, 4 * 128))),
                "model.language_model.layers.0.mlp.gate.bias": ("F32", [4], struct.pack("<4f", 0.1, -0.1, 0.2, 0.0)),
                "model.language_model.layers.0.mlp.experts.down_proj": ("BF16", [4, 128, 128], _f32_to_bf16_bytes(_make_f32_vals(10, 4 * 128 * 128))),
                "model.language_model.layers.1.input_layernorm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(11, 128, centered_one=True))),
                "model.language_model.layers.1.mlp.gate.weight": ("F32", [4, 128], struct.pack("<" + "f" * (4 * 128), *_make_f32_vals(12, 4 * 128))),
                "model.language_model.layers.1.mlp.experts.down_proj": ("BF16", [4, 128, 128], _f32_to_bf16_bytes(_make_f32_vals(13, 4 * 128 * 128))),
                "model.visual.backbone.layers.0.attn.qkv.weight": ("BF16", [192, 64], _f32_to_bf16_bytes(_make_f32_vals(14, 192 * 64))),
                "model.visual.backbone.layers.22.mlp.fc1.weight": ("BF16", [256, 64], _f32_to_bf16_bytes(_make_f32_vals(15, 256 * 64))),
            }
            _write_safetensors(src_dir / "model.safetensors", bb_tensors)

            mtp_tensors: dict[str, tuple[str, list[int], bytes]] = {
                "mtp.fuse_weight": ("F32", [2], struct.pack("<2f", 0.5, 0.5)),
                "mtp.mask_embed": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(16, 128))),
                "mtp.markov_head.markov_w1.weight": ("BF16", [256, 128], _f32_to_bf16_bytes(_make_f32_vals(17, 256 * 128))),
                "mtp.layers.0.input_layernorm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(18, 128, centered_one=True))),
                "mtp.layers.0.self_attn.q_proj.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(19, 128 * 128))),
            }
            _write_safetensors(src_dir / "mtp.safetensors", mtp_tensors)

            out_bb = root / "out" / "model.amf"
            out_vis = root / "out" / "vision.amf"
            out_df = root / "out" / "dflare.amf"
            rep = convert_inkling2_checkpoint(
                src_dir,
                out_bb,
                vision_sidecar=out_vis,
                dflare_output=out_df,
                cold_experts_per_layer=2,
            )
            self.assertGreater(rep.sme2_controller_count, 0)
            self.assertEqual(rep.cold_expert_count, 2 * 2)  # 2 layers * 2 cold experts/layer (suffix-less 3D splits per-expert)
            self.assertTrue(out_vis.is_file() and out_df.is_file())
            self.assertIsNotNone(rep.dflare_config_path)

            with AmfReader(out_bb) as r:
                by_name = {m.name: m for m in r.entries}
                self.assertEqual(by_name["layers.0.self_attn.q_proj"].dtype, DType.NVFP4_G16)
                self.assertEqual(by_name["layers.0.self_attn.o_proj"].dtype, DType.MXFP8_G32)
                self.assertEqual(by_name["layers.0.controllers.attn.down_weight"].layout, TensorLayout.SME2_PANEL64)
                self.assertEqual(by_name["layers.0.controllers.attn.forget_weight"].layout, TensorLayout.SME2_PANEL64)
                self.assertEqual(by_name["layers.0.controllers.attn.gain_weight"].layout, TensorLayout.SME2_PANEL64)
                self.assertEqual(by_name["layers.0.mlp.experts.2.down_proj.weight"].kind, TensorKind.COLD_EXPERT)
                self.assertEqual(by_name["layers.0.mlp.experts.3.down_proj.weight"].kind, TensorKind.COLD_EXPERT)
                self.assertFalse(any(is_vision_tensor(n) for n in by_name))
                self.assertFalse(any("markov_w1" in n for n in by_name))

            with AmfReader(out_vis) as vr:
                vnames = {m.name for m in vr.entries}
                self.assertIn("visual.blocks.0.attn.qkv.weight", vnames)
                self.assertIn("visual.blocks.22.mlp.fc1.weight", vnames)

            with AmfReader(out_df) as dr:
                dnames = {m.name for m in dr.entries}
                self.assertIn("markov_head.markov_w1.weight", dnames)
                self.assertIn("fuse_weight", dnames)

            # Explicit `--sme2-pack none` keeps controllers ROW_MAJOR.
            out_none = root / "out_none" / "model.amf"
            convert_inkling2_checkpoint(src_dir, out_none, sme2_pack="none", strip_vision=True)
            with AmfReader(out_none) as rn:
                ctrl_n = [m for m in rn.entries if is_inkling2_controller_tensor(m.name)]
                self.assertTrue(all(m.layout == TensorLayout.ROW_MAJOR for m in ctrl_n))

            _verify_amf_with_zig(self, out_bb, len(rep.backbone_metas))
            _verify_amf_with_zig(self, out_vis, len(rep.vision_metas))
            _verify_amf_with_zig(self, out_df, len(rep.dflare_metas))

    def test_nvfp4_stacked_experts_scalar_denom_broadcast_and_oob_reject(self) -> None:
        from aminfer_tools.amf import _ByteRange, _ShardReaders, _decode_scalar_f32_at

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            src_dir = root / "hf_nvfp4_scalar_denom"
            src_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "architectures": ["Inkling2LoopForConditionalGeneration"],
                "model_type": "inkling2_loop",
                "text_config": {
                    "model_type": "inkling2_loop_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 1,
                    "vocab_size": 64,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "intermediate_size": 32,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "local_layer_ids": [0],
                },
            }
            (src_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            # 4 stacked NVFP4 experts (`[4, 64, 32]` U8 -> K=64) with per-expert `[4]` amax
            # and a scalar `[1]` outer_scale_denominator (`2688.0`) placed immediately before
            # an adjacent sentinel tensor (`-999.0` bytes) that would corrupt OOB reads.
            num_exp = 4
            amax_vals = [0.125, 0.25, 0.375, 0.5]
            denom_val = 2688.0
            w13_packed = bytes((i * 13 + 7) & 0xFF for i in range(num_exp * 64 * 32))
            w13_scale = bytes([0x38] * (num_exp * 64 * 4))
            bb_tensors: dict[str, tuple[str, list[int], bytes]] = {
                "model.llm.embed_tokens.weight": ("BF16", [64, 64], _f32_to_bf16_bytes(_make_f32_vals(1, 64 * 64))),
                "model.llm.norm.weight": ("BF16", [64], _f32_to_bf16_bytes(_make_f32_vals(2, 64, centered_one=True))),
                "lm_head.weight": ("BF16", [64, 64], _f32_to_bf16_bytes(_make_f32_vals(3, 64 * 64))),
                "model.llm.layers.0.mlp.experts.w13_weight": ("U8", [num_exp, 64, 32], w13_packed),
                "model.llm.layers.0.mlp.experts.w13_weight_scale": ("F8_E4M3", [num_exp, 64, 4], w13_scale),
                "model.llm.layers.0.mlp.experts.w13_weight_amax": ("F32", [num_exp], struct.pack("<4f", *amax_vals)),
                "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator": (
                    "F32",
                    [1],
                    struct.pack("<f", denom_val),
                ),
                # Adjacent negative/NaN sentinel right after scalar denom in safetensors stream.
                "model.llm.layers.0.mlp.gate.bias": (
                    "F32",
                    [num_exp],
                    struct.pack("<4f", -999.0, -777.0, float("nan"), -555.0),
                ),
            }
            sf_path = src_dir / "model.safetensors"
            _write_safetensors(sf_path, bb_tensors)

            out_amf = root / "out" / "model.amf"
            convert_inkling2_checkpoint(src_dir, out_amf, strip_vision=True)

            with AmfReader(out_amf) as r:
                by_name = {m.name: m for m in r.entries}
                with out_amf.open("rb") as f:
                    for e in range(num_exp):
                        meta = by_name[f"layers.0.mlp.experts.{e}.w13"]
                        self.assertEqual(meta.dtype, DType.NVFP4_G16)
                        f.seek(meta.blob_offset + meta.params_offset + meta.params_bytes - 4)
                        got_outer = struct.unpack("<f", f.read(4))[0]
                        self.assertAlmostEqual(got_outer, amax_vals[e] / denom_val, places=7)

            # Direct unit check on `_decode_scalar_f32_at`: scalar `[1]` broadcasts, vector `[2]` rejects OOB.
            vec_path = root / "vec.bin"
            vec_path.write_bytes(struct.pack("<3f", 10.0, 20.0, 30.0))
            readers = _ShardReaders()
            try:
                br_scalar = _ByteRange(path=vec_path, offset=0, length=4, dtype="F32")
                self.assertEqual(_decode_scalar_f32_at(readers, br_scalar, 0), 10.0)
                self.assertEqual(_decode_scalar_f32_at(readers, br_scalar, 127), 10.0)
                with self.assertRaises(ValueError):
                    _decode_scalar_f32_at(readers, br_scalar, -1)

                br_vec2 = _ByteRange(path=vec_path, offset=0, length=8, dtype="F32")
                self.assertEqual(_decode_scalar_f32_at(readers, br_vec2, 0), 10.0)
                self.assertEqual(_decode_scalar_f32_at(readers, br_vec2, 1), 20.0)
                with self.assertRaises(ValueError):
                    _decode_scalar_f32_at(readers, br_vec2, 2)
            finally:
                readers.close()

    def test_mlx_inkling2_long_names_and_gguf_loop_dedup_plus_dflash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            # 1) MLX Inkling 2 4-bit real HF names compact cleanly to <= 47B & preserve controller/LM_HEAD roles.
            mlx_dir = root / "mlx_inkling2"
            mlx_tensors: dict[str, tuple[str, list[int], bytes]] = {
                "model.language_model.embed.weight": ("BF16", [128, 64], _f32_to_bf16_bytes(_make_f32_vals(1, 128 * 64))),
                "model.language_model.unembed.weight": ("BF16", [128, 64], _f32_to_bf16_bytes(_make_f32_vals(2, 128 * 64))),
                "model.language_model.final_input.query": ("F32", [64], struct.pack("<64f", *_make_f32_vals(3, 64))),
                "model.language_model.layers.6.q_message_gate.down_proj.weight": (
                    "BF16",
                    [64, 64],
                    _f32_to_bf16_bytes(_make_f32_vals(4, 64 * 64)),
                ),
                "model.language_model.layers.0.post_attention_convolution.conv1d.weight": (
                    "BF16",
                    [64, 4],
                    _f32_to_bf16_bytes(_make_f32_vals(5, 64 * 4)),
                ),
                "model.language_model.layers.6.attention_input.query": (
                    "F32",
                    [64],
                    struct.pack("<64f", *_make_f32_vals(6, 64)),
                ),
                "model.language_model.layers.0.self_attn.q_proj.weight": (
                    "U32",
                    [64, 8],
                    b"\x11\x22\x33\x44" * (64 * 8),
                ),
                "model.language_model.layers.0.self_attn.q_proj.scales": ("F16", [64, 1], _f32_to_f16_bytes([0.1] * 64)),
                "model.language_model.layers.0.self_attn.q_proj.biases": ("F16", [64, 1], _f32_to_f16_bytes([0.0] * 64)),
            }
            _write_safetensors(mlx_dir / "model.safetensors", mlx_tensors)
            mlx_amf = root / "mlx.amf"
            metas_mlx = pack_amf_from_safetensors(mlx_dir, mlx_amf, model_id="mlx-inkling2", tie_word_embeddings=False)
            self.assertTrue(all(len(m.name.encode("utf-8")) <= 47 for m in metas_mlx))
            by_mlx = {m.name: m for m in metas_mlx}
            self.assertEqual(by_mlx["lm_head.weight"].kind, TensorKind.LM_HEAD)
            self.assertEqual(by_mlx["layers.6.controllers.attn.down_weight"].layout, TensorLayout.SME2_PANEL64)
            self.assertIn("layers.0.attn_sconv.weight", by_mlx)
            self.assertIn("attn_res.attn_queries.6", by_mlx)
            self.assertIn("attn_res.output_query", by_mlx)
            _verify_amf_with_zig(self, mlx_amf, len(metas_mlx))

            # 2) GGUF `arch=inkling2` unrolled `block_count=48` -> deduplicated cleanly to `24` physical layers (0..23).
            gguf_path = root / "inkling2_mini_48.gguf"
            f16_eye_64 = _f32_to_f16_bytes(_make_f32_vals(20, 64 * 64))
            gguf_tensors: list[tuple[str, tuple[int, ...], int, bytes]] = [
                ("token_embd.weight", (64, 128), 1, _f32_to_f16_bytes(_make_f32_vals(21, 128 * 64))),
                ("output_norm.weight", (64,), 0, struct.pack("<64f", *_make_f32_vals(22, 64, centered_one=True))),
                ("output.weight", (64, 128), 1, _f32_to_f16_bytes(_make_f32_vals(23, 128 * 64))),
            ]

            def _phys_index(step_l: int) -> int:
                if step_l < 6:
                    return step_l
                if step_l < 42:
                    return 6 + ((step_l - 6) % 12)
                return 18 + (step_l - 42)

            for step_l in range(48):
                phys = _phys_index(step_l)
                attn_payload = _f32_to_f16_bytes(_make_f32_vals(100 + phys, 64 * 64))
                norm_payload = struct.pack("<64f", *_make_f32_vals(200 + phys, 64, centered_one=True))
                ctrl_payload = _f32_to_f16_bytes(_make_f32_vals(300 + phys, 64 * 64))
                gguf_tensors.append((f"blk.{step_l}.attn_norm.weight", (64,), 0, norm_payload))
                gguf_tensors.append((f"blk.{step_l}.attn_q.weight", (64, 64), 1, attn_payload))
                if step_l == 6:
                    gguf_tensors.append((f"blk.{step_l}.controllers.attn.down_weight", (64, 64), 1, ctrl_payload))
                elif 18 <= step_l < 42 and (step_l - 6) % 12 == 0:
                    gguf_tensors.append((f"blk.{step_l}.controllers.attn.down_weight", (64, 64), 1, ctrl_payload))
                if step_l == 0:
                    gguf_tensors.append(("blk.0.attn_sconv.weight", (64, 4), 1, f16_eye_64[: 64 * 4 * 2]))

            _write_gguf_file(
                gguf_path,
                version=3,
                kv=[
                    ("general.architecture", 8, "inkling2"),
                    ("general.name", 8, "inkling2-mini-test"),
                    ("inkling2.block_count", 4, 48),
                    ("inkling2.context_length", 4, 4096),
                    ("inkling2.embedding_length", 4, 64),
                    ("inkling2.attention.head_count", 4, 1),
                    ("inkling2.attention.head_count_kv", 4, 1),
                    ("inkling2.loop.core_start", 4, 6),
                    ("inkling2.loop.core_end", 4, 18),
                    ("inkling2.loop.num_passes", 4, 3),
                    ("inkling2.loop.gate_rank", 4, 64),
                ],
                tensors=gguf_tensors,
            )
            gguf_amf = root / "inkling2_from_gguf.amf"
            gguf_stats: dict[str, int] = {}
            metas_gguf = pack_amf_from_gguf(
                gguf_path,
                gguf_amf,
                embed_metadata=True,
                stats_out=gguf_stats,
            )
            self.assertEqual(gguf_stats["deduplicated_layers"], 24)
            self.assertEqual(gguf_stats["physical_layers"], 24)
            gnames = {m.name for m in metas_gguf}
            # Coda layers 42..47 re-indexed cleanly to contiguous 18..23!
            self.assertIn("layers.23.self_attn.q_proj.weight", gnames)
            self.assertFalse(any(n.startswith("layers.24.") or n.startswith("layers.42.") for n in gnames))
            by_g = {m.name: m for m in metas_gguf}
            self.assertEqual(by_g["layers.6.controllers.attn.down_weight"].layout, TensorLayout.SME2_PANEL64)

            with AmfReader(gguf_amf) as gr:
                gcfg = json.loads((gr.read_metadata_bytes("__meta.config_json") or b"{}").decode("utf-8"))
                self.assertEqual(gcfg["model_type"], "inkling2_loop")
                self.assertEqual(gcfg["num_hidden_layers"], 24)
                self.assertEqual(gcfg["effective_num_steps"], 48)
                self.assertEqual(gcfg["loop_config"]["core_start"], 6)
            _verify_amf_with_zig(self, gguf_amf, len(metas_gguf))

            # 3) Standalone `arch=dflash` GGUF -> `D14` `dflare.amf` + `dflare_config.json`.
            dflash_gguf = root / "dflash_draft.gguf"
            _write_gguf_file(
                dflash_gguf,
                version=3,
                kv=[
                    ("general.architecture", 8, "dflash"),
                    ("dflash.block_count", 4, 3),
                    ("dflash.embedding_length", 4, 64),
                    ("dflash.attention.head_count", 4, 1),
                    ("dflash.attention.head_count_kv", 4, 1),
                    ("dflash.block_size", 4, 8),
                    ("dflash.markov_rank", 4, 256),
                    ("dflash.target_layer_ids", 9, (4, [3, 12, 21])),
                ],
                tensors=[
                    ("mask_embed", (64,), 0, struct.pack("<64f", *_make_f32_vals(41, 64))),
                    ("markov_head.markov_w1.weight", (64, 256), 1, _f32_to_f16_bytes(_make_f32_vals(42, 64 * 256))),
                    ("blk.0.attn_norm.weight", (64,), 0, struct.pack("<64f", *_make_f32_vals(43, 64, centered_one=True))),
                    ("blk.2.attn_q.weight", (64, 64), 1, _f32_to_f16_bytes(_make_f32_vals(44, 64 * 64))),
                ],
            )
            df_out_dir = root / "dflare_from_gguf"
            df_amf, df_cfg_path, df_metas = export_dflare_bundle_from_gguf(dflash_gguf, df_out_dir)
            self.assertTrue(df_amf.is_file() and df_cfg_path.is_file())
            df_cfg = json.loads(df_cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(df_cfg["draft_kind"], "dflash")
            self.assertEqual(df_cfg["dflare_format"], "dflare")
            self.assertEqual(df_cfg["num_hidden_layers"], 3)
            self.assertEqual(df_cfg["block_size"], 8)
            self.assertEqual(df_cfg["markov_rank"], 256)
            self.assertEqual(df_cfg["target_layer_ids"], [3, 12, 21])
            _verify_amf_with_zig(self, df_amf, len(df_metas))


class TestHy4FormatsAndConverters(unittest.TestCase):
    def test_hy4_scale_inv_fp8_engram_e4m3_row_shards_latent_moe_and_dfly2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            src_dir = root / "hf_hy4_flash"
            src_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "architectures": ["HyV4FlashForCausalLM"],
                "model_type": "hy_v4_flash",
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "vocab_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "q_lora_rank": 64,
                "kv_lora_rank": 64,
                "qk_nope_head_dim": 32,
                "qk_rope_head_dim": 32,
                "v_head_dim": 64,
                "hc_mult": 4,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "moe_intermediate_size": 64,
                "routed_scaling_factor": 3.998,
                "swiglu_limit": 10.0,
                "index_pool_compress_ratio": 4,
                "index_pool_group_topk": 512,
                "index_pool_candidate_topk": 2048,
                "dfly_num_hidden_layers": 5,
                "dfly_block_size": 7,
                "dfly_target_kv_layers": [0, 1],
                "vision_config": {"hidden_size": 64, "num_hidden_layers": 27},
            }
            (src_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            # `[128, 128]` FP8 weight + `[1, 1]` BF16 `weight_scale_inv` block scale (`FP8_BLOCK128`).
            fp8_w = bytes(_f32_to_e4m3_byte(0.125) for _ in range(128 * 128))
            scale_inv = _f32_to_bf16_bytes([0.25])

            # BF16 Engram table `[16, 128]` -> quantized to `fp8_e4m3_row` (`DType.FP8_E4M3_ROW`) across 2 shards.
            eng_vals_bf16 = _make_f32_vals(50, 16 * 128, scale=0.8)
            eng_raw_bf16 = _f32_to_bf16_bytes(eng_vals_bf16)
            expected_eng_vals = _bf16_bytes_to_f32(eng_raw_bf16)

            # Native F8_E4M3 Engram table `[8, 128]` + BF16 `[8]` row scale -> preserved losslessly as `FP8_E4M3_ROW`.
            eng_f8_bytes = bytes(_f32_to_e4m3_byte((i % 7 - 3) * 0.5) for i in range(8 * 128))
            eng_f8_scales = [0.02 + 0.01 * r for r in range(8)]

            tensors: dict[str, tuple[str, list[int], bytes]] = {
                "model.embed_tokens.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(51, 128 * 128))),
                "model.norm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(52, 128, centered_one=True))),
                "lm_head.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(53, 128 * 128))),
                "model.layers.0.input_layernorm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(54, 128, centered_one=True))),
                "model.layers.0.self_attn.o_proj.weight": ("F8_E4M3", [128, 128], fp8_w),
                "model.layers.0.self_attn.o_proj.weight_scale_inv": ("BF16", [1, 1], scale_inv),
                "model.layers.0.self_attn.learnable_sink_param": ("F32", [2], struct.pack("<2f", 0.1, 0.2)),
                "model.layers.0.self_attn.linear_gate.weight": ("BF16", [64, 128], _f32_to_bf16_bytes(_make_f32_vals(55, 64 * 128))),
                "model.layers.1.input_layernorm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(56, 128, centered_one=True))),
                "model.layers.1.mlp.experts.down_proj": ("BF16", [4, 128, 64], _f32_to_bf16_bytes(_make_f32_vals(57, 4 * 128 * 64))),
                "model.layers.1.mlp.routed_expert_up_proj.weight": ("BF16", [64, 128], _f32_to_bf16_bytes(_make_f32_vals(58, 64 * 128))),
                "model.layers.1.mlp.combine_output_norm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(59, 128, centered_one=True))),
                "model.layers.1.mlp.gate.e_score_correction_bias": ("F32", [4], struct.pack("<4f", 0.1, 0.0, -0.1, 0.2)),
                "model.layers.1.engram_layer.multi_head_embedding.embedding.weight": ("BF16", [16, 128], eng_raw_bf16),
                "model.layers.1.engram_layer.multi_head_embedding.weight": ("F8_E4M3", [8, 128], eng_f8_bytes),
                "model.layers.1.engram_layer.multi_head_embedding.weight_scale": ("BF16", [8], _f32_to_bf16_bytes(eng_f8_scales)),
                "model.layers.1.engram_layer.value_proj.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(60, 128 * 128))),
                "model.dfly_layers.0.input_layernorm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(61, 128, centered_one=True))),
                "model.dfly_layers.0.confidence_head.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(62, 128 * 128))),
                "model.dfly_layers.4.hidden_correction.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(63, 128 * 128))),
                "model.vit.layers.0.attn.qkv.weight": ("BF16", [192, 64], _f32_to_bf16_bytes(_make_f32_vals(64, 192 * 64))),
                "model.vit.layers.26.mlp.fc2.weight": ("BF16", [64, 256], _f32_to_bf16_bytes(_make_f32_vals(65, 64 * 256))),
            }
            _write_safetensors(src_dir / "model.safetensors", tensors)

            out_bb = root / "out" / "model.amf"
            out_vis = root / "out" / "vision.amf"
            out_eng = root / "out" / "engram.amf"
            out_df = root / "out" / "dfly2.amf"
            rep = convert_hy4_checkpoint(
                src_dir,
                out_bb,
                vision_sidecar=out_vis,
                engram_output=out_eng,
                engram_shards=2,
                engram_dtype="fp8_e4m3_row",
                dfly2_output=out_df,
                cold_experts_per_layer=2,
            )
            self.assertEqual(len(rep.engram_paths), 2)
            self.assertTrue(out_vis.is_file() and out_df.is_file())

            with AmfReader(out_bb) as mr:
                by_m = {m.name: m for m in mr.entries}
                self.assertEqual(by_m["layers.0.self_attn.o_proj"].dtype, DType.FP8_E4M3_BLOCK)
                self.assertIn("layers.1.engram_layer.value_proj.weight", by_m)
                self.assertEqual(by_m["layers.1.mlp.experts.2.down_proj.weight"].kind, TensorKind.COLD_EXPERT)
                self.assertFalse(any("multi_head_embedding" in n for n in by_m))
                self.assertFalse(any(is_vision_tensor(n) for n in by_m))

            # Verify FP8_E4M3_ROW reconstruction accuracy & native F8_E4M3 scale preservation across both shards!
            recon_rows: list[list[float]] = []
            native_rows_seen = 0
            for sp in rep.engram_paths:
                with AmfReader(sp) as er:
                    eng_entries = list(er.entries)
                    for em in eng_entries:
                        self.assertEqual(em.kind, TensorKind.SPARSE_TABLE)
                        self.assertEqual(em.dtype, DType.FP8_E4M3_ROW)
                        rows, cols = em.shape
                        blob = er.read_blob(em)
                        data_part = blob[: em.data_bytes]
                        scale_part = blob[em.params_offset : em.params_offset + em.params_bytes]
                        self.assertEqual(len(data_part), rows * cols)
                        self.assertEqual(len(scale_part), rows * 4)
                        if ".t1." in em.name:
                            # Shard slice of native 8-row F8_E4M3 table keeps exact byte match + BF16->F32 row scale.
                            native_rows_seen += rows
                        else:
                            for r in range(rows):
                                (row_scale,) = struct.unpack("<f", scale_part[r * 4 : (r + 1) * 4])
                                qrow = data_part[r * cols : (r + 1) * cols]
                                recon_rows.append([e4m3_byte_to_f32(b) * row_scale for b in qrow])
                _verify_amf_with_zig(self, sp, len(eng_entries))

            self.assertEqual(native_rows_seen, 8)
            self.assertEqual(len(recon_rows), 16)
            for r_idx, row in enumerate(recon_rows):
                orig = expected_eng_vals[r_idx * 128 : (r_idx + 1) * 128]
                max_err = max(abs(a - b) for a, b in zip(row, orig))
                self.assertLess(max_err, 0.08)

            with AmfReader(out_vis) as vr:
                vnames = {m.name for m in vr.entries}
                self.assertIn("vision.blocks.26.mlp.fc2.weight", vnames)

            with AmfReader(out_df) as dr:
                dnames = {m.name for m in dr.entries}
                self.assertIn("layers.0.confidence_head.weight", dnames)
                self.assertIn("layers.4.hidden_correction.weight", dnames)

            _verify_amf_with_zig(self, out_bb, len(rep.backbone_metas))
            _verify_amf_with_zig(self, out_vis, len(rep.vision_metas))
            _verify_amf_with_zig(self, out_df, len(rep.dfly2_metas))

    def test_hy4_gguf_engram_sharding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            gguf_path = root / "hy4_flash.gguf"
            _write_gguf_file(
                gguf_path,
                version=3,
                kv=[
                    ("general.architecture", 8, "hy4"),
                    ("hy4.block_count", 4, 2),
                    ("hy4.embedding_length", 4, 64),
                    ("hy4.attention.head_count", 4, 1),
                    ("hy4.attention.head_count_kv", 4, 1),
                ],
                tensors=[
                    ("token_embd.weight", (64, 64), 1, _f32_to_f16_bytes(_make_f32_vals(70, 64 * 64))),
                    ("output_norm.weight", (64,), 0, struct.pack("<64f", *_make_f32_vals(71, 64, centered_one=True))),
                    ("output.weight", (64, 64), 1, _f32_to_f16_bytes(_make_f32_vals(72, 64 * 64))),
                    ("blk.0.attn_norm.weight", (64,), 0, struct.pack("<64f", *_make_f32_vals(73, 64, centered_one=True))),
                    ("blk.0.attn_q.weight", (64, 64), 1, _f32_to_f16_bytes(_make_f32_vals(74, 64 * 64))),
                    ("per_layer_token_embd.weight", (64, 16), 0, struct.pack("<" + "f" * (16 * 64), *_make_f32_vals(75, 16 * 64))),
                ],
            )
            bb_amf = root / "hy4_bb.amf"
            eng_base = root / "hy4_engram.amf"
            stats: dict[str, int] = {}
            metas = pack_amf_from_gguf(
                gguf_path,
                bb_amf,
                engram_output=eng_base,
                engram_shards=2,
                stats_out=stats,
            )
            self.assertFalse(any("per_layer_token_embd" in m.name for m in metas))
            s0 = root / "hy4_engram-00000-of-00002.amf"
            s1 = root / "hy4_engram-00001-of-00002.amf"
            self.assertTrue(s0.is_file() and s1.is_file())
            total_eng_rows = 0
            for sp in (s0, s1):
                with AmfReader(sp) as er:
                    eng_entries = list(er.entries)
                for em in eng_entries:
                    self.assertEqual(em.kind, TensorKind.SPARSE_TABLE)
                    self.assertEqual(em.dtype, DType.FP8_E4M3_ROW)
                    total_eng_rows += em.shape[0]
                _verify_amf_with_zig(self, sp, len(eng_entries))
            self.assertEqual(total_eng_rows, 16)
            _verify_amf_with_zig(self, bb_amf, len(metas))


class TestGsqRcoV2AndRealMetadata(unittest.TestCase):
    def test_gsq_rco_v2_protects_sconv_controllers_dflare_dfly2_latent_moe_and_shared_norms(self) -> None:
        protected_names = [
            "layers.0.self_attn.sconv.convolution.weight",
            "layers.0.self_attn.sconv.weight_proj.weight",
            "layers.6.controllers.attn.down_weight",
            "layers.6.controllers.mlp.forget_weight",
            "layers.0.attn_in.attn_mgate.gain_weight",
            "middle_cycle_cross_pass_queries",
            "middle_cycle_cross_pass_kv_queries",
            "markov_head.markov_w1.weight",
            "markov_head.rnn_proj.weight",
            "layers.0.confidence_head.weight",
            "layers.0.hidden_correction.weight",
            "fuse_weight",
            "mask_embed",
            "layers.1.mlp.routed_expert_up_proj.weight",
            "layers.1.mlp.routed_expert_down_proj.weight",
            "layers.1.engram_layer.value_proj.weight",
            "layers.1.mlp.combine_output_norm.weight",
            "layers.1.self_attn.post_proj_norm.weight",
            "layers.1.mlp.post_fc2_norm.weight",
            "markov_head.dflare_norm.weight",
            "layers.1.engram_layer.unified_pre_gamma",
        ]
        for name in protected_names:
            role = _role_of_tensor(name, _classify_tensor(name, set(), False)[0])
            self.assertIn(role, _PROTECTED_HIGH_PRECISION_ROLES, msg=f"{name!r} classified as {role!r}, expected protected")

        self.assertEqual(_role_of_tensor("layers.0.mlp.experts.5.down_proj.weight", TensorKind.HOT_EXPERT), "moe_expert")

        # Aggressive Q2/Q3 budget (`2.55 bpw`) still preserves protected Inkling2/Hy4 roles at Q8_AFFINE / BF16.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sf_dir = root / "gsq_src"
            tensors: dict[str, tuple[str, list[int], bytes]] = {
                "model.embed_tokens.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(80, 128 * 128))),
                "model.norm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(81, 128, centered_one=True))),
                "lm_head.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(82, 128 * 128))),
                "layers.0.self_attn.sconv.weight_proj.weight": ("BF16", [128, 128], _f32_to_bf16_bytes(_make_f32_vals(83, 128 * 128))),
                "layers.0.controllers.attn.down_weight": ("BF16", [128, 64], _f32_to_bf16_bytes(_make_f32_vals(84, 128 * 64))),
                "layers.0.mlp.routed_expert_up_proj.weight": ("BF16", [64, 128], _f32_to_bf16_bytes(_make_f32_vals(85, 64 * 128))),
                "layers.0.mlp.combine_output_norm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(86, 128, centered_one=True))),
                "layers.0.mlp.post_fc2_norm.weight": ("BF16", [128], _f32_to_bf16_bytes(_make_f32_vals(87, 128, centered_one=True))),
            }
            for e in range(16):
                tensors[f"layers.0.mlp.experts.{e}.down_proj.weight"] = (
                    "BF16",
                    [128, 128],
                    _f32_to_bf16_bytes(_make_f32_vals(100 + e, 128 * 128, scale=0.08)),
                )
            _write_safetensors(sf_dir / "model.safetensors", tensors)
            out_amf = root / "gsq_out.amf"
            with self.assertRaises(ValueError):
                gsq_convert(
                    sf_dir,
                    root / "gsq_too_low.amf",
                    target_bits=2.55,
                    group_size=64,
                    enforce_backbone_floor=True,
                    fine_rungs=True,
                    max_q2_share=0.80,
                    sme2_pack="controllers",
                )
            rep = gsq_convert(
                sf_dir,
                out_amf,
                target_bits=2.85,
                group_size=64,
                enforce_backbone_floor=True,
                fine_rungs=True,
                max_q2_share=0.80,
                sme2_pack="controllers",
            )
            by_name = {m.name: m for m in rep.metas}
            sconv_meta = by_name.get("layers.0.self_attn.sconv.weight_proj") or by_name["layers.0.self_attn.sconv.weight_proj.weight"]
            latent_up_meta = by_name.get("layers.0.mlp.routed_expert_up_proj") or by_name["layers.0.mlp.routed_expert_up_proj.weight"]
            self.assertNotIn(sconv_meta.dtype, (DType.Q2_AFFINE, DType.Q3_AFFINE))
            self.assertNotIn(by_name["layers.0.controllers.attn.down_weight"].dtype, (DType.Q2_AFFINE, DType.Q3_AFFINE))
            self.assertEqual(by_name["layers.0.controllers.attn.down_weight"].layout, TensorLayout.SME2_PANEL64)
            self.assertNotIn(latent_up_meta.dtype, (DType.Q2_AFFINE, DType.Q3_AFFINE))
            self.assertEqual(by_name["layers.0.mlp.combine_output_norm.weight"].dtype, DType.BF16)
            self.assertEqual(by_name["layers.0.mlp.post_fc2_norm.weight"].dtype, DType.BF16)
            self.assertTrue(any(m.dtype in (DType.Q2_AFFINE, DType.Q3_AFFINE) for m in rep.metas if ".experts." in m.name))
            _verify_amf_with_zig(self, out_amf, len(rep.metas))

        # Verify GSQ-RCO v2 tier profiles for Inkling-2-Small and Inkling-2-Mini.
        p_small_24 = resolve_inkling2_tier_profile("24gb-m4", num_hidden_layers=48, n_routed_experts=384)
        p_small_128 = resolve_inkling2_tier_profile("128gb-max", num_hidden_layers=48, n_routed_experts=384)
        p_small_256 = resolve_inkling2_tier_profile("256gb-studio", num_hidden_layers=48, n_routed_experts=384)
        self.assertLess(p_small_24.target_bits, p_small_128.target_bits)
        self.assertLess(p_small_128.target_bits, p_small_256.target_bits)
        self.assertGreater(p_small_24.cold_tail_fraction, 0.5)
        self.assertEqual(p_small_256.cold_tail_fraction, 0.0)
        p_mini_24 = resolve_inkling2_tier_profile("24gb-m4", num_hidden_layers=24, n_routed_experts=128)
        self.assertAlmostEqual(p_mini_24.cold_tail_fraction, 0.125, places=3)

    @unittest.skipUnless(
        REAL_METADATA_ROOT is not None and REAL_METADATA_ROOT.is_dir(),
        "real Hugging Face metadata cache not present (set AMINFER_METADATA_DIR or AMINFER_MODELS_DIR)",
    )
    def test_validate_all_real_metadata_dirs(self) -> None:
        assert REAL_METADATA_ROOT is not None
        mini = validate_inkling2_metadata_dir(REAL_METADATA_ROOT / "thinkingmachines__Inkling-2-Mini")
        self.assertEqual(mini.model_type, "inkling2_loop")
        self.assertEqual(mini.num_hidden_layers, 24)
        self.assertEqual(mini.effective_num_steps, 48)
        self.assertEqual(mini.n_routed_experts, 128)
        self.assertEqual(mini.num_experts_per_tok, 8)
        self.assertEqual(mini.intermediate_size, 1152)
        self.assertEqual(mini.local_layer_count, 20)
        self.assertEqual(mini.sliding_window_size, 1024)
        self.assertEqual(mini.dflare_layers, 3)
        self.assertEqual(mini.vision_layers, 23)
        self.assertEqual(mini.long_backbone_names, ())

        small = validate_inkling2_metadata_dir(REAL_METADATA_ROOT / "thinkingmachines__Inkling-2-Small")
        self.assertEqual(small.model_type, "inkling2")
        self.assertEqual(small.num_hidden_layers, 48)
        self.assertEqual(small.n_routed_experts, 384)
        self.assertEqual(small.local_layer_count, 40)
        self.assertEqual(small.dflare_layers, 5)
        self.assertEqual(small.long_backbone_names, ())

        mini_mlx = validate_inkling2_metadata_dir(REAL_METADATA_ROOT / "mlx-community__Inkling-2-Mini-4bit")
        self.assertEqual(mini_mlx.model_type, "inkling2_loop")
        self.assertEqual(mini_mlx.num_hidden_layers, 24)
        self.assertEqual(mini_mlx.long_backbone_names, ())

        for dirname in (
            "tencent__Hy4",
            "tencent__Hy4-preview",
            "tencent__Hy4-flash",
            "mlx-community__Hy4-preview-4bit",
            "yujiepan__hy4-preview-tiny-random",
        ):
            rep = validate_hy4_metadata_dir(REAL_METADATA_ROOT / dirname)
            self.assertEqual(rep.long_backbone_names, (), msg=f"{dirname} had >47B names: {rep.long_backbone_names}")
            if dirname in ("tencent__Hy4", "tencent__Hy4-preview"):
                self.assertEqual(rep.n_routed_experts, 256)
            elif dirname == "tencent__Hy4-flash":
                self.assertEqual(rep.n_routed_experts, 512)
                self.assertTrue(rep.uses_index_pool)
                self.assertGreater(rep.engram_table_tensor_count, 0)

        # CLI `inspect-metadata --arch auto --json` smoke check on real metadata dirs.
        for dirname in (
            "thinkingmachines__Inkling-2-Mini",
            "thinkingmachines__Inkling-2-Small",
            "tencent__Hy4-flash",
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main([
                    "inspect-metadata",
                    "--input",
                    str(REAL_METADATA_ROOT / dirname),
                    "--arch",
                    "auto",
                    "--json",
                ])
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["index"]["long_backbone_names"], [])

    @unittest.skipUnless(
        REAL_HY4_TINY_RANDOM is not None and (REAL_HY4_TINY_RANDOM / "model.safetensors").is_file(),
        "staged hy4-preview-tiny-random absent (set AMINFER_MODELS_DIR)",
    )
    def test_real_hy4_tiny_random_smoke(self) -> None:
        assert REAL_HY4_TINY_RANDOM is not None
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "hy4_tiny"
            rep = convert_hy4_checkpoint(
                REAL_HY4_TINY_RANDOM,
                out_dir / "model.amf",
                vision_sidecar=out_dir / "vision.amf",
                engram_output=out_dir / "engram.amf",
                dfly2_output=out_dir / "dfly2.amf",
            )
            self.assertGreater(len(rep.backbone_metas), 0)
            with AmfReader(rep.backbone_amf) as r:
                entries = list(r.entries)
            self.assertTrue(all(len(m.name.encode("utf-8")) <= 47 for m in entries))
            self.assertTrue(any("experts." in m.name and ".down_proj." in m.name for m in entries))
            self.assertTrue(any("q_a_proj" in m.name or "kv_a_proj_with_mqa" in m.name or "kv_a_mqa" in m.name for m in entries))
            _verify_amf_with_zig(self, rep.backbone_amf, len(rep.backbone_metas))


if __name__ == "__main__":
    unittest.main()
