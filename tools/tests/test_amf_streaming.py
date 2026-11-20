"""Tests for the M2.5 streaming multi-shard safetensors importer (ADR-0016, W12)."""

from __future__ import annotations

import io
import json
import math
import pathlib
import random
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from aminfer_tools import APPLE_PAGE_SIZE, amf
from aminfer_tools.amf import (
    ENTRY_STRUCT,
    HEADER_STRUCT,
    DType,
    TensorKind,
    bf16_bytes_to_f16_bytes,
    f16_bytes_to_bf16_bytes,
    pack_amf_from_safetensors,
    resolve_checkpoint,
)
from aminfer_tools.cli import main

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "mlx"}}
    chunks: list[bytes] = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in chunks:
            f.write(chunk)


def bf16(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", v)[2:4] for v in values)


def f16(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}e", *values)


def quant_triple(
    prefix: str, rng: random.Random, n: int, k: int, bits: int, experts: int | None = None
) -> dict[str, tuple[str, list[int], bytes]]:
    """MLX affine triple: U32 packed weight plus BF16 scales/biases (optionally 3D-stacked)."""
    packed = k * bits // 32
    groups = k // 64
    lead = [experts] if experts is not None else []
    count = experts or 1
    w = rng.randbytes(count * n * packed * 4)
    scales = bf16([0.01 * (1 + rng.randrange(64)) for _ in range(count * n * groups)])
    biases = bf16([-0.05 * (1 + rng.randrange(64)) for _ in range(count * n * groups)])
    return {
        f"{prefix}.weight": ("U32", lead + [n, packed], w),
        f"{prefix}.scales": ("BF16", lead + [n, groups], scales),
        f"{prefix}.biases": ("BF16", lead + [n, groups], biases),
    }


def read_amf(path: pathlib.Path) -> tuple[str, dict[str, dict[str, object]], bytes]:
    data = path.read_bytes()
    hdr = HEADER_STRUCT.unpack_from(data, 0)
    count, dir_off = hdr[5], hdr[7]
    model_id = hdr[11].rstrip(b"\x00").decode()
    entries: dict[str, dict[str, object]] = {}
    for i in range(count):
        f = ENTRY_STRUCT.unpack_from(data, dir_off + i * ENTRY_STRUCT.size)
        name = f[0].rstrip(b"\x00").decode()
        rank = f[4]
        entries[name] = {
            "kind": TensorKind(f[1]),
            "dtype": DType(f[2]),
            "quant_bits": f[3],
            "group_size": f[5],
            "layer": f[6],
            "expert": f[7],
            "shape": tuple(f[10 : 10 + rank]),
            "blob_offset": f[14],
            "blob_bytes": f[15],
            "data_bytes": f[16],
            "params_offset": f[17],
            "params_bytes": f[18],
        }
    return model_id, entries, data


def blob_data(data: bytes, e: dict[str, object]) -> bytes:
    off = int(e["blob_offset"])  # type: ignore[arg-type]
    return data[off : off + int(e["data_bytes"])]  # type: ignore[arg-type]


def blob_params(data: bytes, e: dict[str, object]) -> bytes:
    off = int(e["blob_offset"]) + int(e["params_offset"])  # type: ignore[arg-type]
    return data[off : off + int(e["params_bytes"])]  # type: ignore[arg-type]


def half2(scales_bf16: bytes, biases_bf16: bytes) -> bytes:
    s = bf16_bytes_to_f16_bytes(scales_bf16)
    b = bf16_bytes_to_f16_bytes(biases_bf16)
    return b"".join(s[2 * i : 2 * i + 2] + b[2 * i : 2 * i + 2] for i in range(len(s) // 2))


class ConversionTests(unittest.TestCase):
    def test_f16_to_bf16_round_to_nearest_even(self) -> None:
        cases = [
            (1.0, 0x3F80),
            (-2.5, 0xC020),
            (1.0 + 2.0**-8, 0x3F80),  # exact tie, even LSB -> round down
            (1.0 + 2.0**-7 + 2.0**-8, 0x3F82),  # exact tie, odd LSB -> round up
            (1.0 + 2.0**-8 + 2.0**-10, 0x3F81),  # above tie -> round up
            (65504.0, 0x4780),  # f16 max rounds to 65536 in bf16
            (2.0**-24, 0x3380),  # smallest f16 subnormal is exact in bf16
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (math.inf, 0x7F80),
            (-math.inf, 0xFF80),
        ]
        got = struct.unpack(f"<{len(cases)}H", f16_bytes_to_bf16_bytes(f16([v for v, _ in cases])))
        self.assertEqual(list(got), [bits for _, bits in cases])
        nan_bits = struct.unpack("<H", f16_bytes_to_bf16_bytes(f16([math.nan])))[0]
        self.assertEqual(nan_bits & 0x7F80, 0x7F80)
        self.assertNotEqual(nan_bits & 0x007F, 0)
        with self.assertRaises(ValueError):
            f16_bytes_to_bf16_bytes(b"\x00")

    def test_bf16_to_f16_rne_and_overflow(self) -> None:
        vals = [0.0, 0.25, -1.5, 3.5, 1.0 + 2.0**-7, 65280.0]
        out = struct.unpack(f"<{len(vals)}e", bf16_bytes_to_f16_bytes(bf16(vals)))
        self.assertEqual(list(out), vals)
        with self.assertRaises(ValueError):
            bf16_bytes_to_f16_bytes(bf16([1.0e5]))
        # bf16 -> f16 -> bf16 is lossless for values representable in both.
        self.assertEqual(f16_bytes_to_bf16_bytes(bf16_bytes_to_f16_bytes(bf16(vals))), bf16(vals))


class StreamingPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def verify_with_zig(self, path: pathlib.Path, tensors: int) -> str:
        self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
        proc = subprocess.run([str(AMINFER_BIN), "verify-amf", str(path)], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("amf.status       ok", proc.stdout)
        self.assertIn(f"amf.tensors      {tensors}\n", proc.stdout)
        return proc.stdout

    def build_moe_checkpoint(self, tie: bool) -> tuple[pathlib.Path, dict[str, tuple[str, list[int], bytes]]]:
        """Two-shard MLX-style MoE checkpoint with index.json and config.json."""
        rng = random.Random(1234)
        ckpt = self.tmp / "ckpt"
        ckpt.mkdir()
        shard1: dict[str, tuple[str, list[int], bytes]] = {}
        shard1.update(quant_triple("model.embed_tokens", rng, 8, 64, 4))
        shard1.update(quant_triple("model.layers.0.mlp.gate", rng, 4, 64, 3))
        shard1["model.layers.0.input_layernorm.weight"] = ("BF16", [64], bf16([1.0 + i / 128 for i in range(64)]))
        shard1["model.norm.weight"] = ("F16", [64], f16([0.5] * 64))
        shard1.update(quant_triple("lm_head", rng, 8, 64, 4))
        shard2: dict[str, tuple[str, list[int], bytes]] = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shard2.update(quant_triple(f"model.layers.0.mlp.switch_mlp.{proj}", rng, 64, 64, 3, experts=4))
        shard2["model.layers.1.mlp.switch_mlp.up_proj.weight"] = (
            "BF16",
            [2, 16, 64],
            bf16([rng.uniform(-1, 1) for _ in range(2 * 16 * 64)]),
        )
        write_safetensors(ckpt / "model-00001-of-00002.safetensors", shard1)
        write_safetensors(ckpt / "model-00002-of-00002.safetensors", shard2)
        weight_map = {n: "model-00001-of-00002.safetensors" for n in shard1}
        weight_map.update({n: "model-00002-of-00002.safetensors" for n in shard2})
        (ckpt / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )
        (ckpt / "config.json").write_text(json.dumps({"tie_word_embeddings": tie, "model_type": "qwen3_moe"}))
        return ckpt, {**shard1, **shard2}

    def test_multishard_switch_mlp_router_embed_and_tie(self) -> None:
        ckpt, src = self.build_moe_checkpoint(tie=True)
        out = self.tmp / "moe.amf"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = main(
                ["pack-amf", "--input", str(ckpt), "--output", str(out), "--model-id", "moe-smoke", "--cold-experts", "0:3"]
            )
        self.assertEqual(rc, 0)
        self.assertIn("packed 18 tensors", stdout.getvalue())
        model_id, entries, data = read_amf(out)
        self.assertEqual(model_id, "moe-smoke")
        self.assertFalse((self.tmp / "moe.amf.partial").exists())

        # tie_word_embeddings=true in config.json: lm_head is deduplicated away.
        self.assertNotIn("lm_head", entries)
        # 1 embed + 1 router + 1 norm + 1 final norm + 3 projs x 4 experts + 2 unquantized experts.
        self.assertEqual(len(entries), 4 + 12 + 2)

        embed = entries["embed_tokens"]
        self.assertEqual((embed["kind"], embed["dtype"], embed["quant_bits"]), (TensorKind.EMBEDDING, DType.Q4_AFFINE, 4))
        self.assertEqual(embed["shape"], (8, 64))
        self.assertEqual(blob_data(data, embed), src["model.embed_tokens.weight"][2])
        self.assertEqual(
            blob_params(data, embed),
            half2(src["model.embed_tokens.scales"][2], src["model.embed_tokens.biases"][2]),
        )

        router = entries["layers.0.mlp.gate"]
        self.assertEqual((router["kind"], router["dtype"], router["layer"]), (TensorKind.ROUTER, DType.Q3_AFFINE, 0))
        self.assertEqual(router["shape"], (4, 64))

        norm = entries["layers.0.input_layernorm.weight"]
        self.assertEqual((norm["kind"], norm["dtype"], norm["quant_bits"]), (TensorKind.NORM, DType.BF16, 0))
        self.assertEqual(blob_data(data, norm), src["model.layers.0.input_layernorm.weight"][2])
        self.assertEqual(entries["norm.weight"]["dtype"], DType.F16)

        for proj in ("gate_proj", "up_proj", "down_proj"):
            w_all = src[f"model.layers.0.mlp.switch_mlp.{proj}.weight"][2]
            s_all = src[f"model.layers.0.mlp.switch_mlp.{proj}.scales"][2]
            b_all = src[f"model.layers.0.mlp.switch_mlp.{proj}.biases"][2]
            w_step, sb_step = len(w_all) // 4, len(s_all) // 4
            for e in range(4):
                ent = entries[f"layers.0.mlp.experts.{e}.{proj}"]
                self.assertEqual(ent["kind"], TensorKind.COLD_EXPERT if e == 3 else TensorKind.HOT_EXPERT)
                self.assertEqual((ent["layer"], ent["expert"], ent["shape"]), (0, e, (64, 64)))
                self.assertEqual((ent["dtype"], ent["quant_bits"], ent["group_size"]), (DType.Q3_AFFINE, 3, 64))
                self.assertEqual(ent["params_offset"] % 64, 0)
                self.assertEqual(blob_data(data, ent), w_all[e * w_step : (e + 1) * w_step])
                self.assertEqual(
                    blob_params(data, ent),
                    half2(s_all[e * sb_step : (e + 1) * sb_step], b_all[e * sb_step : (e + 1) * sb_step]),
                )

        up_all = src["model.layers.1.mlp.switch_mlp.up_proj.weight"][2]
        for e in range(2):
            ent = entries[f"layers.1.mlp.experts.{e}.up_proj.weight"]
            self.assertEqual((ent["dtype"], ent["shape"], ent["expert"]), (DType.BF16, (16, 64), e))
            self.assertEqual(blob_data(data, ent), up_all[e * 2048 : (e + 1) * 2048])

        # Blobs are 16 KiB aligned and laid out layer-contiguously in natural order.
        names = list(entries)
        self.assertEqual(names[0], "embed_tokens")
        expert_pos = [names.index(f"layers.0.mlp.experts.{e}.gate_proj") for e in range(4)]
        self.assertEqual(expert_pos, sorted(expert_pos))
        self.assertLess(max(i for i, n in enumerate(names) if n.startswith("layers.0.")),
                        min(i for i, n in enumerate(names) if n.startswith("layers.1.")))
        for ent in entries.values():
            self.assertEqual(int(ent["blob_offset"]) % APPLE_PAGE_SIZE, 0)  # type: ignore[arg-type]

        stdout = self.verify_with_zig(out, 18)
        self.assertIn("amf.model_id     moe-smoke", stdout)
        self.assertIn(f"amf.cold_bytes   {3 * APPLE_PAGE_SIZE}", stdout)

        # The index.json path and the directory path produce byte-identical files.
        out_idx = self.tmp / "moe_idx.amf"
        pack_amf_from_safetensors(
            ckpt / "model.safetensors.index.json", out_idx, model_id="moe-smoke", cold_experts={(0, 3)}
        )
        self.assertEqual(out_idx.read_bytes(), data)

        # --keep-tied-lm-head overrides config.json and emits lm_head.
        out_keep = self.tmp / "moe_keep.amf"
        with redirect_stdout(io.StringIO()):
            rc = main(["pack-amf", "--input", str(ckpt), "--output", str(out_keep), "--keep-tied-lm-head"])
        self.assertEqual(rc, 0)
        _, keep_entries, _ = read_amf(out_keep)
        self.assertEqual(keep_entries["lm_head"]["kind"], TensorKind.LM_HEAD)
        self.assertEqual(keep_entries["lm_head"]["dtype"], DType.Q4_AFFINE)
        self.verify_with_zig(out_keep, 19)

    def test_untied_config_keeps_lm_head_and_flag_forces_tie(self) -> None:
        ckpt, _ = self.build_moe_checkpoint(tie=False)
        metas = pack_amf_from_safetensors(ckpt, self.tmp / "a.amf", model_id="untied")
        self.assertIn("lm_head", {m.name for m in metas})
        with redirect_stdout(io.StringIO()):
            rc = main(["pack-amf", "--input", str(ckpt), "--output", str(self.tmp / "b.amf"), "--tie-word-embeddings"])
        self.assertEqual(rc, 0)
        _, entries, _ = read_amf(self.tmp / "b.amf")
        self.assertNotIn("lm_head", entries)

    def build_dense_bf16(self) -> tuple[pathlib.Path, dict[str, tuple[str, list[int], bytes]]]:
        """Single-file unquantized Hugging Face style checkpoint (tied embeddings)."""
        rng = random.Random(7)
        ckpt = self.tmp / "dense"
        ckpt.mkdir()
        tensors = {
            "model.embed_tokens.weight": ("BF16", [32, 64], bf16([rng.uniform(-0.5, 0.5) for _ in range(32 * 64)])),
            "lm_head.weight": ("BF16", [32, 64], bf16([rng.uniform(-0.5, 0.5) for _ in range(32 * 64)])),
            "model.layers.0.self_attn.q_proj.weight": (
                "BF16",
                [128, 64],
                bf16([rng.uniform(-1, 1) for _ in range(128 * 64)]),
            ),
            "model.layers.0.self_attn.q_norm.weight": ("F16", [16], f16([1.0 + i / 64 for i in range(16)])),
            "model.layers.0.mlp.gate.weight": ("F32", [4, 64], struct.pack("<256f", *[i / 256 for i in range(256)])),
        }
        write_safetensors(ckpt / "model.safetensors", tensors)
        (ckpt / "config.json").write_text(json.dumps({"tie_word_embeddings": True, "torch_dtype": "bfloat16"}))
        return ckpt, tensors

    def test_dtype_modes_auto_f16_bf16(self) -> None:
        ckpt, src = self.build_dense_bf16()
        q_src = src["model.layers.0.self_attn.q_proj.weight"][2]
        n_src = src["model.layers.0.self_attn.q_norm.weight"][2]
        r_src = src["model.layers.0.mlp.gate.weight"][2]

        auto = self.tmp / "auto.amf"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["pack-amf", "--input", str(ckpt), "--output", str(auto), "--model-id", "dense-bf16"]), 0)
        _, entries, data = read_amf(auto)
        self.assertNotIn("lm_head.weight", entries)  # tied in config.json
        self.assertEqual(entries["embed_tokens.weight"]["dtype"], DType.BF16)
        self.assertEqual(entries["embed_tokens.weight"]["kind"], TensorKind.EMBEDDING)
        self.assertEqual(entries["layers.0.self_attn.q_proj.weight"]["dtype"], DType.BF16)
        self.assertEqual(blob_data(data, entries["layers.0.self_attn.q_proj.weight"]), q_src)
        self.assertEqual(entries["layers.0.self_attn.q_norm.weight"]["dtype"], DType.F16)
        self.assertEqual(blob_data(data, entries["layers.0.self_attn.q_norm.weight"]), n_src)
        self.assertEqual(entries["layers.0.mlp.gate.weight"]["dtype"], DType.F32)
        self.assertEqual(entries["layers.0.mlp.gate.weight"]["kind"], TensorKind.ROUTER)
        self.assertEqual(blob_data(data, entries["layers.0.mlp.gate.weight"]), r_src)
        for e in entries.values():
            self.assertEqual((e["quant_bits"], e["group_size"], e["params_bytes"]), (0, 0, 0))
        self.verify_with_zig(auto, 4)

        to_f16 = self.tmp / "f16.amf"
        pack_amf_from_safetensors(ckpt, to_f16, model_id="dense-f16", dtype_mode="f16")
        _, entries, data = read_amf(to_f16)
        self.assertEqual(entries["layers.0.self_attn.q_proj.weight"]["dtype"], DType.F16)
        self.assertEqual(blob_data(data, entries["layers.0.self_attn.q_proj.weight"]), bf16_bytes_to_f16_bytes(q_src))
        self.assertEqual(entries["layers.0.mlp.gate.weight"]["dtype"], DType.F32)
        self.verify_with_zig(to_f16, 4)

        to_bf16 = self.tmp / "bf16.amf"
        pack_amf_from_safetensors(ckpt, to_bf16, model_id="dense-bf16", dtype_mode="bf16")
        _, entries, data = read_amf(to_bf16)
        self.assertEqual(entries["layers.0.self_attn.q_norm.weight"]["dtype"], DType.BF16)
        self.assertEqual(blob_data(data, entries["layers.0.self_attn.q_norm.weight"]), f16_bytes_to_bf16_bytes(n_src))
        self.assertEqual(blob_data(data, entries["layers.0.self_attn.q_proj.weight"]), q_src)
        self.verify_with_zig(to_bf16, 4)

        # Single .safetensors file input finds config.json next to it.
        single = self.tmp / "single.amf"
        pack_amf_from_safetensors(ckpt / "model.safetensors", single, model_id="dense-bf16")
        self.assertEqual(single.read_bytes(), auto.read_bytes())

    def test_tiny_chunks_produce_identical_output(self) -> None:
        """Multi-chunk streaming (copy, conversion and scale/bias paths) is byte-identical."""
        ckpt, _ = self.build_moe_checkpoint(tie=False)
        ref = self.tmp / "ref.amf"
        pack_amf_from_safetensors(ckpt, ref, model_id="chunks", dtype_mode="f16")
        small = self.tmp / "small.amf"
        with mock.patch.object(amf, "_COPY_CHUNK_BYTES", 96), mock.patch.object(amf, "_CONVERT_CHUNK_ELEMS", 7):
            pack_amf_from_safetensors(ckpt, small, model_id="chunks", dtype_mode="f16")
        self.assertEqual(small.read_bytes(), ref.read_bytes())
        _, entries, _ = read_amf(small)
        self.assertEqual(entries["layers.1.mlp.experts.0.up_proj.weight"]["dtype"], DType.F16)
        self.verify_with_zig(small, 19)

    def test_pass1_reads_only_headers(self) -> None:
        """Planning touches only shard JSON headers: tensor bytes are never read."""
        ckpt, _ = self.build_moe_checkpoint(tie=True)
        resolved = resolve_checkpoint(ckpt)
        self.assertEqual(len(resolved.shard_paths), 2)
        self.assertEqual(resolved.config_path, ckpt / "config.json")
        with mock.patch.object(amf, "_read_exact", side_effect=AssertionError("tensor data read in pass 1")):
            plan = amf.plan_amf_entries(amf._collect_tensors(resolved.shard_paths), tie_word_embeddings=True)
        self.assertEqual(len(plan), 18)
        self.assertTrue(all(p.blob_bytes % APPLE_PAGE_SIZE == 0 for p in plan))

    def test_invalid_inputs_are_rejected(self) -> None:
        rng = random.Random(3)
        bad = self.tmp / "bad"
        bad.mkdir()
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(bad)

        # 3D quantized tensor that is not a switch_mlp expert stack.
        write_safetensors(bad / "model.safetensors", quant_triple("model.layers.0.self_attn.q_proj", rng, 8, 64, 4, experts=2))
        with self.assertRaisesRegex(ValueError, "switch_mlp"):
            pack_amf_from_safetensors(bad, self.tmp / "x.amf", model_id="bad")
        self.assertFalse((self.tmp / "x.amf").exists())
        self.assertFalse((self.tmp / "x.amf.partial").exists())

        # Duplicate tensor across shards.
        dup = self.tmp / "dup"
        dup.mkdir()
        t = {"model.norm.weight": ("F16", [64], f16([1.0] * 64))}
        write_safetensors(dup / "a.safetensors", t)
        write_safetensors(dup / "b.safetensors", t)
        (dup / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.norm.weight": "a.safetensors", "x": "b.safetensors"}})
        )
        with self.assertRaisesRegex(ValueError, "appears in both"):
            pack_amf_from_safetensors(dup, self.tmp / "y.amf", model_id="dup")

        # Missing shard listed in the index.
        (dup / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
        with self.assertRaises(FileNotFoundError):
            pack_amf_from_safetensors(dup, self.tmp / "z.amf", model_id="dup")

        # Header data_offsets inconsistent with dtype/shape.
        corrupt = self.tmp / "corrupt.safetensors"
        write_safetensors(corrupt, {"w": ("BF16", [64], bf16([0.0] * 32))})
        with self.assertRaisesRegex(ValueError, "data_offsets"):
            pack_amf_from_safetensors(corrupt, self.tmp / "c.amf", model_id="c")

        with self.assertRaises(ValueError):
            pack_amf_from_safetensors(dup / "a.safetensors", self.tmp / "d.amf", model_id="d", dtype_mode="fp8")


if __name__ == "__main__":
    unittest.main()
