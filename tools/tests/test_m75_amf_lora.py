"""Unit & CLI integration tests for M7.5 W18 embedded metadata and W19 `.amlora` packing."""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import tempfile
import unittest

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    FLAG_EMBEDDED_METADATA,
    META_ADAPTER_CONFIG_JSON,
    META_CONFIG_JSON,
    META_TOKENIZER_JSON,
    DType,
    TensorKind,
    TensorLayout,
    embed_metadata_into_amf,
    load_amf_metadata,
    pack_amf_from_safetensors,
    read_amf_metadata,
    unpack_sme2_panel64_f16_panel,
)
from aminfer_tools.cli import main as cli_main
from aminfer_tools.lora import (
    MAX_LORA_L2_POOL_BYTES,
    generate_synthetic_lora,
    pack_lora_from_peft,
)
from aminfer_tools.synth import PRESETS, generate_qwen3_bundle


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    cursor = 0
    chunks: list[bytes] = []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(raw)],
        }
        chunks.append(raw)
        cursor += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for ch in chunks:
            fh.write(ch)


class TestW18EmbeddedMetadata(unittest.TestCase):
    def test_pack_amf_embed_metadata_and_backward_compat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ckpt = root / "hf_ckpt"
            ckpt.mkdir()
            norm_bytes = struct.pack("<64e", *[1.0] * 64)
            q_bytes = struct.pack("<4096e", *[(i % 13) * 0.01 for i in range(4096)])
            _write_safetensors(
                ckpt / "model.safetensors",
                {
                    "model.norm.weight": ("F16", [64], norm_bytes),
                    "model.layers.0.self_attn.q_proj.weight": ("F16", [64, 64], q_bytes),
                },
            )
            cfg_obj = {
                "model_type": "qwen3",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "vocab_size": 64,
                "tie_word_embeddings": True,
            }
            tok_obj = {"version": "1.0", "model": {"type": "BPE", "vocab": {"hello": 0}, "merges": []}}
            cfg_bytes = (json.dumps(cfg_obj) + "\n").encode("utf-8")
            tok_bytes = (json.dumps(tok_obj) + "\n").encode("utf-8")
            (ckpt / "config.json").write_bytes(cfg_bytes)
            (ckpt / "tokenizer.json").write_bytes(tok_bytes)

            # 1. Default pack (embed_metadata=False): legacy v0 compatibility (`flags == 0`, no `__meta.*`).
            legacy_out = root / "legacy.amf"
            legacy_metas = pack_amf_from_safetensors(ckpt, legacy_out, model_id="legacy-qwen3")
            hdr_leg, metas_leg = load_amf_metadata(legacy_out)
            self.assertEqual(hdr_leg.flags, 0)
            self.assertEqual(len(legacy_metas), 2)
            self.assertEqual(len(metas_leg), 2)
            self.assertIsNone(read_amf_metadata(legacy_out, META_CONFIG_JSON))

            # 2. Single-file pack (`--embed-metadata`): embeds both `__meta.config_json` & `__meta.tokenizer_json`.
            sf_out = root / "single.amf"
            rc = cli_main(
                [
                    "pack-amf",
                    "--input",
                    str(ckpt),
                    "--output",
                    str(sf_out),
                    "--model-id",
                    "single-qwen3",
                    "--embed-metadata",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(sf_out.stat().st_size % APPLE_PAGE_SIZE, 0)

            hdr_sf, metas_sf = load_amf_metadata(sf_out)
            self.assertEqual(hdr_sf.flags & FLAG_EMBEDDED_METADATA, FLAG_EMBEDDED_METADATA)
            meta_entries = [m for m in metas_sf if m.kind == TensorKind.METADATA]
            self.assertEqual(len(meta_entries), 2)
            for me in meta_entries:
                self.assertEqual(me.blob_offset % APPLE_PAGE_SIZE, 0)
                self.assertEqual(me.blob_bytes % APPLE_PAGE_SIZE, 0)
                self.assertEqual(me.dtype, DType.U32)
                self.assertEqual(me.shape, (me.data_bytes,))

            self.assertEqual(read_amf_metadata(sf_out, META_CONFIG_JSON), cfg_bytes)
            self.assertEqual(read_amf_metadata(sf_out, META_TOKENIZER_JSON), tok_bytes)

            # 3. Replacing metadata via `embed_metadata_into_amf` updates payload and keeps weight blobs bit-exact.
            override_cfg = b'{"model_type":"qwen3","hidden_size":64,"override":true}'
            embed_metadata_into_amf(sf_out, config_bytes=override_cfg)
            self.assertEqual(read_amf_metadata(sf_out, META_CONFIG_JSON), override_cfg)
            self.assertEqual(read_amf_metadata(sf_out, META_TOKENIZER_JSON), tok_bytes)

            # Cross-verify single-file `.amf` directly with native Zig `aminfer verify-amf` if built.
            repo_root = pathlib.Path(__file__).resolve().parents[2]
            aminfer_bin = repo_root / "zig-out" / "bin" / "aminfer"
            if aminfer_bin.is_file():
                proc = subprocess.run(
                    [str(aminfer_bin), "verify-amf", str(sf_out)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"aminfer verify-amf failed:\n{proc.stdout}\n{proc.stderr}")
                self.assertIn("amf.status       ok", proc.stdout)

            # 4. Corrupting metadata payload byte triggers CRC-32 failure.
            bad_out = root / "corrupt.amf"
            raw = bytearray(sf_out.read_bytes())
            cfg_meta = next(m for m in load_amf_metadata(sf_out)[1] if m.name == META_CONFIG_JSON)
            raw[cfg_meta.blob_offset] ^= 0x55
            bad_out.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "CRC mismatch"):
                read_amf_metadata(bad_out, META_CONFIG_JSON)

    def test_synth_qwen3_embed_metadata_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = pathlib.Path(td) / "synth_sf"
            rc = cli_main(
                [
                    "synth-qwen3",
                    "--preset",
                    "tiny-dense",
                    "--output-dir",
                    str(out_dir),
                    "--embed-metadata",
                ]
            )
            self.assertEqual(rc, 0)
            amf_path = out_dir / "model.amf"
            hdr, metas = load_amf_metadata(amf_path)
            self.assertEqual(hdr.flags & FLAG_EMBEDDED_METADATA, FLAG_EMBEDDED_METADATA)
            self.assertEqual(read_amf_metadata(amf_path, META_CONFIG_JSON), (out_dir / "config.json").read_bytes())
            self.assertEqual(
                read_amf_metadata(amf_path, META_TOKENIZER_JSON), (out_dir / "tokenizer.json").read_bytes()
            )
            self.assertTrue(any(m.kind == TensorKind.METADATA for m in metas))

            # Verify default `generate_qwen3_bundle(..., embed_metadata=False)` leaves flags == 0.
            out_leg = pathlib.Path(td) / "synth_leg"
            generate_qwen3_bundle(out_leg, PRESETS["tiny-dense"], embed_metadata=False)
            hdr_leg, _ = load_amf_metadata(out_leg / "model.amf")
            self.assertEqual(hdr_leg.flags, 0)


class TestW19PackLora(unittest.TestCase):
    def test_pack_lora_from_peft_row_major_and_sme2_panel64(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            peft_dir = root / "peft_sql"
            peft_dir.mkdir()

            rank = 4
            k_dim = 64
            n_dim = 64
            a_vals = [((i % 9) - 4) * 0.125 for i in range(rank * k_dim)]
            b_vals = [((i % 7) - 3) * 0.25 for i in range(n_dim * rank)]
            a_f32 = struct.pack(f"<{len(a_vals)}f", *a_vals)
            b_f32 = struct.pack(f"<{len(b_vals)}f", *b_vals)

            (peft_dir / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "r": rank,
                        "lora_alpha": 16,
                        "target_modules": ["q_proj", "v_proj"],
                        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
                    }
                ),
                encoding="utf-8",
            )
            _write_safetensors(
                peft_dir / "adapter_model.safetensors",
                {
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": ("F32", [rank, k_dim], a_f32),
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": ("F32", [n_dim, rank], b_f32),
                    "base_model.model.model.layers.1.self_attn.v_proj.lora_A.default.weight": (
                        "F32",
                        [rank, k_dim],
                        a_f32,
                    ),
                    "base_model.model.model.layers.1.self_attn.v_proj.lora_B.default.weight": (
                        "F32",
                        [n_dim, rank],
                        b_f32,
                    ),
                },
            )

            # 1. Pack as row_major f16 `.amlora`
            rm_out = root / "sql_rm.amlora"
            rep_rm = pack_lora_from_peft(peft_dir, rm_out, adapter_id="sql-rm", layout="row_major", dtype="f16")
            self.assertEqual(rep_rm.rank, 4)
            self.assertEqual(rep_rm.lora_alpha, 16.0)
            self.assertEqual(rep_rm.scale, 4.0)
            self.assertEqual(rep_rm.num_pairs, 2)
            self.assertEqual(rep_rm.target_modules, ("q_proj", "v_proj"))
            self.assertTrue(rep_rm.fits_l2_4mib)
            self.assertLessEqual(rep_rm.total_wired_bytes, MAX_LORA_L2_POOL_BYTES)
            self.assertEqual(rm_out.stat().st_size % APPLE_PAGE_SIZE, 0)

            hdr, metas = load_amf_metadata(rm_out)
            self.assertEqual(hdr.flags & FLAG_EMBEDDED_METADATA, FLAG_EMBEDDED_METADATA)
            cfg_raw = read_amf_metadata(rm_out, META_ADAPTER_CONFIG_JSON)
            self.assertIsNotNone(cfg_raw)
            cfg_parsed = json.loads(cfg_raw.decode("utf-8"))
            self.assertEqual(cfg_parsed["r"], 4)
            self.assertEqual(cfg_parsed["lora_alpha"], 16.0)
            self.assertEqual(cfg_parsed["base_model_name_or_path"], "Qwen/Qwen3-0.6B")

            by_name = {m.name: m for m in metas}
            self.assertEqual(by_name["layers.0.self_attn.q_proj.lora_a"].shape, (rank, k_dim))
            self.assertEqual(by_name["layers.0.self_attn.q_proj.lora_b"].shape, (n_dim, rank))
            self.assertEqual(by_name["layers.0.self_attn.q_proj.lora_a"].layout, TensorLayout.ROW_MAJOR)

            # 2. Pack as sme2_panel64 f16 `.amlora` via CLI and verify bit-exact unpacked panel values.
            sme_out = root / "sql_sme2.amlora"
            rc = cli_main(
                [
                    "pack-lora",
                    "--input",
                    str(peft_dir),
                    "--output",
                    str(sme_out),
                    "--adapter-id",
                    "sql-sme2",
                    "--layout",
                    "sme2_panel64",
                    "--dtype",
                    "f16",
                ]
            )
            self.assertEqual(rc, 0)
            _, sme_metas = load_amf_metadata(sme_out)
            sme_by_name = {m.name: m for m in sme_metas}
            b_meta = sme_by_name["layers.0.self_attn.q_proj.lora_b"]
            self.assertEqual(b_meta.layout, TensorLayout.SME2_PANEL64)
            self.assertEqual(b_meta.shape, (n_dim, 64))

            # Unpack panel 0 (rows 0..15) of lora_b and check cols 0..3 match b_vals & cols 4..63 are zero!
            raw_sme = sme_out.read_bytes()
            b_blob = raw_sme[b_meta.blob_offset : b_meta.blob_offset + b_meta.data_bytes]
            panel0_rows = unpack_sme2_panel64_f16_panel(b_blob[:2048], 64, num_rows=16)
            for r_idx in range(16):
                unpacked = struct.unpack("<64e", panel0_rows[r_idx])
                for c in range(rank):
                    self.assertAlmostEqual(unpacked[c], b_vals[r_idx * rank + c], places=3)
                for c in range(rank, 64):
                    self.assertEqual(unpacked[c], 0.0)

    def test_pack_lora_synthetic_cli_and_zig_verify_amf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            out_path = root / "synth.amlora"
            rc = cli_main(
                [
                    "pack-lora",
                    "--synthetic",
                    "--output",
                    str(out_path),
                    "--adapter-id",
                    "code-lora",
                    "--num-layers",
                    "2",
                    "--hidden-size",
                    "64",
                    "--q-dim",
                    "128",
                    "--kv-dim",
                    "64",
                    "--rank",
                    "4",
                    "--lora-alpha",
                    "8",
                    "--target-modules",
                    "q_proj,o_proj",
                    "--layout",
                    "sme2_panel64",
                    "--dtype",
                    "bf16",
                ]
            )
            self.assertEqual(rc, 0)
            hdr, metas = load_amf_metadata(out_path)
            self.assertEqual(hdr.model_id, "code-lora")
            self.assertEqual(len(metas), 2 * 2 * 2 + 1)  # 2 layers * 2 targets * (A+B) + 1 metadata
            self.assertIsNotNone(read_amf_metadata(out_path, META_ADAPTER_CONFIG_JSON))

            # Also test programmatic `generate_synthetic_lora` with zero_init_b.
            zero_path = root / "zero.amlora"
            rep_z = generate_synthetic_lora(
                zero_path,
                adapter_id="zero-lora",
                num_layers=1,
                hidden_size=64,
                rank=4,
                lora_alpha=4.0,
                target_modules=("q_proj",),
                layout="row_major",
                dtype="f16",
                zero_init_b=True,
            )
            self.assertEqual(rep_z.num_pairs, 1)
            b_meta = next(m for m in rep_z.metas if m.name.endswith(".lora_b"))
            raw_z = zero_path.read_bytes()
            self.assertTrue(all(b == 0 for b in raw_z[b_meta.blob_offset : b_meta.blob_offset + b_meta.data_bytes]))

            # If `zig-out/bin/aminfer` is built in the repo, cross-validate the Python `.amlora` directly in Zig!
            repo_root = pathlib.Path(__file__).resolve().parents[2]
            aminfer_bin = repo_root / "zig-out" / "bin" / "aminfer"
            if aminfer_bin.is_file():
                proc = subprocess.run(
                    [str(aminfer_bin), "verify-amf", str(out_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"aminfer verify-amf failed:\n{proc.stdout}\n{proc.stderr}")
                self.assertIn("amf.status       ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
