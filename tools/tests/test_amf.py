"""Tests for .amf v0 packing, MLX safetensors import, and Zig round-trip validation."""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import tempfile
import unittest

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    DType,
    TensorKind,
    bf16_bytes_to_f16_bytes,
    pack_amf_from_safetensors,
    pack_mlx_affine_3bit,
    pack_mlx_affine_4bit,
)
from aminfer_tools.cli import main

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """Write a minimal valid safetensors file for testing."""
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        end = offset + len(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        chunks.append(raw)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in chunks:
            f.write(chunk)


def f32_to_bf16_bytes(values: list[float]) -> bytes:
    out = bytearray()
    for v in values:
        raw = struct.pack("<f", v)
        out.extend(raw[2:4])  # top 16 bits of little-endian IEEE 754 float32
    return bytes(out)


class AmfTests(unittest.TestCase):
    def test_bf16_to_f16_conversion(self) -> None:
        vals = [0.0, 0.25, -1.5, 3.5]
        bf16 = f32_to_bf16_bytes(vals)
        f16 = bf16_bytes_to_f16_bytes(bf16)
        unpacked = list(struct.unpack(f"<{len(vals)}e", f16))
        for expected, got in zip(vals, unpacked):
            self.assertAlmostEqual(expected, got, places=3)

    def test_pack_mlx_4bit_and_3bit_bit_layouts(self) -> None:
        codes4 = [i % 16 for i in range(64)]
        p4 = pack_mlx_affine_4bit(codes4)
        self.assertEqual(len(p4), 32)
        self.assertEqual(p4[0], 0x10)
        self.assertEqual(p4[1], 0x32)

        codes3 = [(i * 5 + 1) % 8 for i in range(64)]
        p3 = pack_mlx_affine_3bit(codes3)
        self.assertEqual(len(p3), 24)
        word24 = sum((codes3[j] << (3 * j)) for j in range(8))
        self.assertEqual(p3[0], word24 & 0xFF)
        self.assertEqual(p3[1], (word24 >> 8) & 0xFF)
        self.assertEqual(p3[2], (word24 >> 16) & 0xFF)

    def test_mlx_safetensors_to_amf_roundtrip_verified_by_zig(self) -> None:
        n4, k4 = 8, 64
        n3, k3 = 6, 128

        # 4-bit dense projection (F16 scales/biases)
        q4_bytes = b"".join(pack_mlx_affine_4bit([(r + c) % 16 for c in range(k4)]) for r in range(n4))
        s4_bytes = struct.pack(f"<{n4 * (k4 // 64)}e", *([0.25] * (n4 * (k4 // 64))))
        b4_bytes = struct.pack(f"<{n4 * (k4 // 64)}e", *([-1.0] * (n4 * (k4 // 64))))

        # 3-bit cold expert projection (BF16 scales/biases, as in Qwen3 MLX checkpoints)
        q3_bytes = b"".join(
            pack_mlx_affine_3bit([(r * 3 + c) % 8 for c in range(64)])
            for r in range(n3)
            for _ in range(k3 // 64)
        )
        s3_bytes = f32_to_bf16_bytes([0.125] * (n3 * (k3 // 64)))
        b3_bytes = f32_to_bf16_bytes([-0.5] * (n3 * (k3 // 64)))

        # Unquantized F16 norm weight
        norm_bytes = struct.pack("<64e", *([1.0] * 64))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            st_path = tmp / "model.safetensors"
            amf_path = tmp / "model.amf"

            write_safetensors(
                st_path,
                {
                    "model.layers.0.self_attn.q_proj.weight": ("U32", [n4, k4 * 4 // 32], q4_bytes),
                    "model.layers.0.self_attn.q_proj.scales": ("F16", [n4, k4 // 64], s4_bytes),
                    "model.layers.0.self_attn.q_proj.biases": ("F16", [n4, k4 // 64], b4_bytes),
                    "model.layers.0.mlp.experts.2.gate_proj.weight": ("U32", [n3, k3 * 3 // 32], q3_bytes),
                    "model.layers.0.mlp.experts.2.gate_proj.scales": ("BF16", [n3, k3 // 64], s3_bytes),
                    "model.layers.0.mlp.experts.2.gate_proj.biases": ("BF16", [n3, k3 // 64], b3_bytes),
                    "model.layers.0.input_layernorm.weight": ("F16", [64], norm_bytes),
                },
            )

            rc = main(
                [
                    "pack-amf",
                    "--input",
                    str(st_path),
                    "--output",
                    str(amf_path),
                    "--model-id",
                    "qwen3-smoke",
                    "--cold-experts",
                    "0:2",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(amf_path.exists())
            self.assertEqual(amf_path.stat().st_size % APPLE_PAGE_SIZE, 0)

            # Verify the packed metadata directly.
            meta = pack_amf_from_safetensors(
                st_path, tmp / "model2.amf", model_id="qwen3-smoke", cold_experts={(0, 2)}
            )
            by_name = {m.name: m for m in meta}
            self.assertEqual(by_name["layers.0.self_attn.q_proj"].dtype, DType.Q4_AFFINE)
            self.assertEqual(by_name["layers.0.mlp.experts.2.gate_proj"].dtype, DType.Q3_AFFINE)
            self.assertEqual(by_name["layers.0.mlp.experts.2.gate_proj"].kind, TensorKind.COLD_EXPERT)
            self.assertEqual(by_name["layers.0.input_layernorm.weight"].kind, TensorKind.NORM)

            # Cross-language round-trip: Zig validates header, directory, 16 KiB alignment, CRCs and wired budget.
            self.assertTrue(AMINFER_BIN.exists(), f"missing {AMINFER_BIN}; run `zig build` first")
            proc = subprocess.run(
                [str(AMINFER_BIN), "verify-amf", str(amf_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            self.assertIn("amf.status       ok", proc.stdout)
            self.assertIn("amf.model_id     qwen3-smoke", proc.stdout)
            self.assertIn("amf.tensors      3", proc.stdout)
            self.assertIn(f"amf.cold_bytes   {APPLE_PAGE_SIZE}", proc.stdout)
            self.assertIn(f"amf.wired_bytes  {2 * APPLE_PAGE_SIZE}", proc.stdout)


if __name__ == "__main__":
    unittest.main()
