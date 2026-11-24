"""Comprehensive M8.1 tests for W21 formats/streaming/GSQ, W22 GGUF repacker, and W24 chat templates."""

from __future__ import annotations

import functools
import http.server
import io
import json
import math
import pathlib
import random
import struct
import subprocess
import tempfile
import threading
import unittest
from contextlib import redirect_stdout

from aminfer_tools import APPLE_PAGE_SIZE, gguf as gguf_mod
from aminfer_tools.amf import (
    LEGACY_CHAT_TEMPLATE_METADATA_NAME,
    META_CHAT_TEMPLATE_JSON,
    AmfReader,
    DType,
    HttpRangeStreamReader,
    MemoryRangeReader,
    ShardStreamSource,
    TensorKind,
    dequantize_nvfp4_row_f32,
    e2m1_nibble_to_f32,
    e4m3_byte_to_f32,
    is_vision_tensor,
    pack_amf_from_safetensors,
    pack_amf_from_stream_sources,
    pack_mlx_affine_5bit,
    pack_mlx_affine_6bit,
    pack_mlx_affine_8bit,
    ue8m0_byte_to_f32,
)
from aminfer_tools.chat_template import (
    compile_chat_template_bytes,
    compile_chat_template_from_dir,
    compile_chat_template_spec,
)
from aminfer_tools.cli import main as cli_main
from aminfer_tools.gguf import (
    expected_ggml_tensor_bytes,
    normalize_gguf_tensor_name,
    pack_amf_from_gguf,
    read_gguf_header,
)
from aminfer_tools.gsq_rco import gsq_convert, parse_gguf_imatrix

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"
REAL_METADATA_ROOT = pathlib.Path("/Volumes/Akari/models/metadata")


def _f32_to_bf16_bytes(vals: list[float]) -> bytes:
    out = bytearray()
    for v in vals:
        (u,) = struct.unpack("<I", struct.pack("<f", float(v)))
        rounding_bias = 0x7FFF + ((u >> 16) & 1)
        out.extend(struct.pack("<H", ((u + rounding_bias) >> 16) & 0xFFFF))
    return bytes(out)


def _f32_to_f16_bytes(vals: list[float]) -> bytes:
    return b"".join(struct.pack("<e", float(v)) for v in vals)


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
    path.write_bytes(blob)
    return blob


def _safetensors_bytes(
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
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)


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
    override_trailing_padding: int | None = None,
) -> None:
    """Write a compliant GGUF v2 or v3 binary file with aligned tensor payloads."""
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
    if override_trailing_padding is not None:
        buf.extend(b"\x00" * override_trailing_padding)
    path.write_bytes(bytes(buf))


def _verify_amf_with_zig(test: unittest.TestCase, path: pathlib.Path, expected_tensors: int) -> None:
    if not AMINFER_BIN.is_file():
        return
    proc = subprocess.run([str(AMINFER_BIN), "verify-amf", str(path)], capture_output=True, text=True, check=False)
    test.assertEqual(proc.returncode, 0, msg=f"verify-amf failed for {path}:\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    test.assertIn("amf.status       ok", proc.stdout)
    test.assertIn(f"amf.tensors      {expected_tensors}\n", proc.stdout)


class _RangeHttpHandler(http.server.BaseHTTPRequestHandler):
    blobs: dict[str, bytes] = {}
    range_requests: list[tuple[str, int, int]] = []

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def _resolve_blob(self) -> bytes | None:
        return self.blobs.get(self.path.lstrip("/"))

    def do_HEAD(self) -> None:  # noqa: N802
        blob = self._resolve_blob()
        if blob is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        blob = self._resolve_blob()
        if blob is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes=") :]
            s_str, e_str = spec.split("-", 1)
            start = int(s_str)
            end = int(e_str) if e_str else len(blob) - 1
            end = min(end, len(blob) - 1)
            chunk = blob[start : end + 1]
            self.range_requests.append((self.path.lstrip("/"), start, end))
            self.send_response(206)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(blob)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


class M81FormatsPackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_mlx_affine_q5_q6_q8_all_group_sizes(self) -> None:
        rng = random.Random(2026)
        for gs in (32, 64, 128):
            ckpt = self.tmp / f"mlx_g{gs}"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "hidden_size": gs * 2,
                        "quantization": {"group_size": gs, "bits": 6},
                    }
                ),
                encoding="utf-8",
            )
            n_rows = 4
            k = gs * 2
            groups = k // gs
            tensors: dict[str, tuple[str, list[int], bytes]] = {}
            expected: dict[str, list[list[float]]] = {}

            specs = [
                ("model.layers.0.self_attn.q_proj", 5, pack_mlx_affine_5bit, DType.Q5_AFFINE),
                ("model.layers.0.self_attn.k_proj", 6, pack_mlx_affine_6bit, DType.Q6_AFFINE),
                ("model.layers.0.mlp.down_proj", 8, pack_mlx_affine_8bit, DType.Q8_AFFINE),
            ]
            for prefix, bits, packer, exp_dtype in specs:
                mod = 1 << bits
                w_buf = bytearray()
                sc_vals: list[float] = []
                bi_vals: list[float] = []
                exp_rows: list[list[float]] = []
                for _r in range(n_rows):
                    codes = [(rng.randrange(mod)) for _ in range(k)]
                    w_buf.extend(packer(codes))
                    row_sc = [round(0.02 * (g + 1) + 0.005 * rng.randint(1, 5), 4) for g in range(groups)]
                    row_bi = [round(-0.25 + 0.05 * rng.randint(0, 8), 4) for _ in range(groups)]
                    # Quantize scales & biases through f16 exactly as stored
                    f16_sc = [struct.unpack("<e", struct.pack("<e", v))[0] for v in row_sc]
                    f16_bi = [struct.unpack("<e", struct.pack("<e", v))[0] for v in row_bi]
                    sc_vals.extend(f16_sc)
                    bi_vals.extend(f16_bi)
                    exp_rows.append([codes[i] * f16_sc[i // gs] + f16_bi[i // gs] for i in range(k)])
                packed_cols = (k * bits) // 32
                tensors[f"{prefix}.weight"] = ("U32", [n_rows, packed_cols], bytes(w_buf))
                tensors[f"{prefix}.scales"] = ("F16", [n_rows, groups], _f32_to_f16_bytes(sc_vals))
                tensors[f"{prefix}.biases"] = ("F16", [n_rows, groups], _f32_to_f16_bytes(bi_vals))
                expected[f"{prefix}:{exp_dtype.name}"] = exp_rows

            _write_safetensors(ckpt / "model.safetensors", tensors, metadata={"format": "mlx"})
            out_amf = ckpt / "out.amf"
            metas = pack_amf_from_safetensors(ckpt, out_amf, model_id=f"test-g{gs}", group_size=None)
            self.assertEqual(len(metas), 3)
            _verify_amf_with_zig(self, out_amf, 3)

            with AmfReader(out_amf) as reader:
                by_name = {m.name: m for m in reader.entries}
                for key, exp_rows in expected.items():
                    prefix, dname = key.split(":")
                    norm = prefix.replace("model.", "")
                    m = by_name[norm]
                    self.assertEqual(m.dtype.name, dname)
                    self.assertEqual(m.group_size, gs)
                    got = reader.read_matrix_f32(m)
                    for r in range(n_rows):
                        for c in range(k):
                            self.assertAlmostEqual(got[r][c], exp_rows[r][c], places=4)

    def test_fp8_block_bf16_scale_inv_promotion_and_row_and_tensor(self) -> None:
        ckpt = self.tmp / "fp8_variants"
        ckpt.mkdir()
        # 0x38 -> 1.0, 0x40 -> 2.0, 0xB8 -> -1.0, 0x3C -> 1.5
        n_blk, k_blk = 130, 140
        blk_codes = bytes([0x38 if (i % 2 == 0) else 0x40 for i in range(n_blk * k_blk)])
        # Block scales shape: [ceil(130/128)=2, ceil(140/128)=2], stored in BF16 (`*_weight_scale_inv`)
        bf16_block_scales = [0.5, 1.5, 2.0, 4.0]
        # Engram row FP8 + F32[N]
        n_engram, k_engram = 6, 32
        row_codes = bytes([0xB8] * (n_engram * k_engram))
        row_scales = [float(i + 1) * 0.25 for i in range(n_engram)]
        # Tensor FP8 + scalar F32[1]
        n_t, k_t = 4, 16
        ten_codes = bytes([0x3C] * (n_t * k_t))
        ten_scale = [3.0]

        tensors = {
            "model.layers.0.mlp.gate_proj.weight": ("F8_E4M3", [n_blk, k_blk], blk_codes),
            "model.layers.0.mlp.gate_proj.weight_scale_inv": ("BF16", [2, 2], _f32_to_bf16_bytes(bf16_block_scales)),
            "model.layers.0.engram.key_proj.weight": ("F8_E4M3", [n_engram, k_engram], row_codes),
            "model.layers.0.engram.key_proj.weight_scale": ("F32", [n_engram], struct.pack(f"<{n_engram}f", *row_scales)),
            "model.layers.0.self_attn.v_proj.weight": ("F8_E4M3", [n_t, k_t], ten_codes),
            "model.layers.0.self_attn.v_proj.weight_scale": ("F32", [1], struct.pack("<f", ten_scale[0])),
        }
        _write_safetensors(ckpt / "model.safetensors", tensors)
        out_amf = ckpt / "fp8.amf"
        metas = pack_amf_from_safetensors(ckpt, out_amf, model_id="fp8-test")
        _verify_amf_with_zig(self, out_amf, len(metas))

        with AmfReader(out_amf) as reader:
            by_name = {m.name: m for m in reader.entries}
            m_blk = by_name["layers.0.mlp.gate_proj"]
            self.assertEqual(m_blk.dtype, DType.FP8_E4M3_BLOCK)
            self.assertEqual(m_blk.params_bytes, 16)
            got_blk = reader.read_rows_f32(m_blk, [0, 129])
            # Row 0 col 0: block (0,0) scale=0.5, code=0x38(1.0) => 0.5; col 129: block (0,1) scale=1.5, code=0x40(2.0) => 3.0
            self.assertAlmostEqual(got_blk[0][0], 0.5, places=5)
            self.assertAlmostEqual(got_blk[0][129], 3.0, places=5)
            # Row 129 col 129: block (1,1) scale=4.0, code=0x40(2.0) => 8.0
            self.assertAlmostEqual(got_blk[1][129], 8.0, places=5)

            m_row = by_name["layers.0.engram.key_proj"]
            self.assertEqual(m_row.dtype, DType.FP8_E4M3_ROW)
            got_row = reader.read_matrix_f32(m_row)
            self.assertAlmostEqual(got_row[3][0], -1.0 * 1.0, places=5)

            m_ten = by_name["layers.0.self_attn.v_proj"]
            self.assertEqual(m_ten.dtype, DType.FP8_E4M3_TENSOR)
            got_ten = reader.read_matrix_f32(m_ten)
            self.assertAlmostEqual(got_ten[2][5], 1.5 * 3.0, places=5)

        self.assertTrue(math.isnan(e4m3_byte_to_f32(0x7F)))
        self.assertTrue(math.isnan(e4m3_byte_to_f32(0xFF)))
        self.assertEqual(e4m3_byte_to_f32(0x7E), 448.0)

    def test_mxfp8_g32_1x32_and_32x32_tile_scales(self) -> None:
        ckpt = self.tmp / "mxfp8"
        ckpt.mkdir()
        n1, k1 = 4, 64
        n2, k2 = 34, 64
        # 0x38 = 1.0; UE8M0: 127->1.0, 129->4.0
        w1 = bytes([0x38] * (n1 * k1))
        s1 = bytes([127 if (g % 2 == 0) else 129 for _ in range(n1) for g in range(k1 // 32)])
        w2 = bytes([0x40] * (n2 * k2))  # 2.0
        # 2D [ceil(34/32)=2, 2] UE8M0 tile scales: tile (0,0)=127(1.0), (0,1)=128(2.0), (1,0)=126(0.5), (1,1)=130(8.0)
        s2 = bytes([127, 128, 126, 130])
        tensors = {
            "model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [n1, k1], w1),
            "model.layers.0.mlp.up_proj.weight_scale": ("U8", [n1, k1 // 32], s1),
            "model.layers.0.mlp.down_proj.weight": ("F8_E4M3", [n2, k2], w2),
            "model.layers.0.mlp.down_proj.weight_scale": ("U8", [2, 2], s2),
        }
        _write_safetensors(ckpt / "model.safetensors", tensors)
        out_amf = ckpt / "mxfp8.amf"
        metas = pack_amf_from_safetensors(ckpt, out_amf, model_id="mxfp8-test")
        _verify_amf_with_zig(self, out_amf, len(metas))

        with AmfReader(out_amf) as reader:
            by_name = {m.name: m for m in reader.entries}
            m1 = by_name["layers.0.mlp.up_proj"]
            self.assertEqual(m1.dtype, DType.MXFP8_G32)
            self.assertEqual(m1.group_size, 32)
            got1 = reader.read_matrix_f32(m1)
            self.assertAlmostEqual(got1[0][0], 1.0, places=5)
            self.assertAlmostEqual(got1[0][32], 4.0, places=5)

            m2 = by_name["layers.0.mlp.down_proj"]
            self.assertEqual(m2.dtype, DType.MXFP8_G32)
            got2 = reader.read_rows_f32(m2, [0, 33])
            self.assertAlmostEqual(got2[0][0], 2.0 * 1.0, places=5)
            self.assertAlmostEqual(got2[0][32], 2.0 * 2.0, places=5)
            self.assertAlmostEqual(got2[1][0], 2.0 * 0.5, places=5)
            self.assertAlmostEqual(got2[1][32], 2.0 * 8.0, places=5)

        self.assertTrue(math.isnan(ue8m0_byte_to_f32(0xFF)))
        self.assertEqual(ue8m0_byte_to_f32(127), 1.0)

    def test_nvfp4_g16_variants_and_3d_stacked_experts(self) -> None:
        ckpt = self.tmp / "nvfp4"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(
            json.dumps({"model_type": "qwen3_moe", "quantization_config": {"quant_method": "compressed-tensors"}}),
            encoding="utf-8",
        )
        n_rows, k = 4, 32
        # Nibbles: low=0x2 (1.0), high=0x3 (1.5) -> byte 0x32
        u8_payload = bytes([0x32] * (n_rows * (k // 2)))
        e4m3_micro = bytes([0x38, 0x40] * n_rows)  # group0 scale 1.0, group1 scale 2.0

        # MLX mode: "nvfp4" stores U32 [N, K/8]: pack byte 0x32 across 4 bytes per u32 word
        mlx_u32 = bytes([0x32] * (n_rows * (k // 2)))

        # 3D stacked experts: E=2, N=4, K=32 via `w13_weight` + `w13_weight_scale` + `w13_weight_scale_2`
        e3_bytes = bytes([0x32] * (2 * n_rows * (k // 2)))
        e3_scales = bytes([0x38, 0x40] * (2 * n_rows))
        e3_s2 = struct.pack("<2f", 2.0, 0.5)

        tensors = {
            # Variant 1: compressed-tensors divisor `weight_global_scale` (outer = 1.0 / 4.0 = 0.25)
            "model.layers.0.self_attn.q_proj.weight": ("U8", [n_rows, k // 2], u8_payload),
            "model.layers.0.self_attn.q_proj.weight_scale": ("F8_E4M3", [n_rows, k // 16], e4m3_micro),
            "model.layers.0.self_attn.q_proj.weight_global_scale": ("F32", [1], struct.pack("<f", 4.0)),
            # Variant 2: Inkling-2 `weight_amax` / `weight_outer_scale_denominator` (outer = 12.0 / 6.0 = 2.0)
            "model.layers.0.self_attn.k_proj.weight": ("U8", [n_rows, k // 2], u8_payload),
            "model.layers.0.self_attn.k_proj.weight_scale": ("F8_E4M3", [n_rows, k // 16], e4m3_micro),
            "model.layers.0.self_attn.k_proj.weight_amax": ("F32", [1], struct.pack("<f", 12.0)),
            "model.layers.0.self_attn.k_proj.weight_outer_scale_denominator": ("F32", [1], struct.pack("<f", 6.0)),
            # Variant 3: MLX `mode: "nvfp4"` (`global_scale = 2688.0 * 0.5 = 1344.0`, outer = 0.5)
            "model.layers.0.self_attn.o_proj.weight": ("U32", [n_rows, k // 8], mlx_u32),
            "model.layers.0.self_attn.o_proj.scales": ("U8", [n_rows, k // 16], e4m3_micro),
            "model.layers.0.self_attn.o_proj.global_scale": ("F32", [1], struct.pack("<f", 1344.0)),
            # Variant 4: 3D stacked experts ModelOpt (`weight_scale_2`)
            "model.layers.0.mlp.experts.w13_weight": ("U8", [2, n_rows, k // 2], e3_bytes),
            "model.layers.0.mlp.experts.w13_weight_scale": ("F8_E4M3", [2, n_rows, k // 16], e3_scales),
            "model.layers.0.mlp.experts.w13_weight_scale_2": ("F32", [2], e3_s2),
        }
        _write_safetensors(ckpt / "model.safetensors", tensors, metadata={"format": "mlx"})
        out_amf = ckpt / "nvfp4.amf"
        metas = pack_amf_from_safetensors(ckpt, out_amf, model_id="nvfp4-test")
        _verify_amf_with_zig(self, out_amf, len(metas))

        with AmfReader(out_amf) as reader:
            by_name = {m.name: m for m in reader.entries}
            q = reader.read_matrix_f32(by_name["layers.0.self_attn.q_proj"])
            # group 0: 1.0 * 0.25 -> even=1.0*0.25=0.25, odd=1.5*0.25=0.375; group 1: 2.0 * 0.25 -> 0.5, 0.75
            self.assertAlmostEqual(q[0][0], 0.25, places=5)
            self.assertAlmostEqual(q[0][1], 0.375, places=5)
            self.assertAlmostEqual(q[0][16], 0.5, places=5)
            self.assertAlmostEqual(q[0][17], 0.75, places=5)

            k_mat = reader.read_matrix_f32(by_name["layers.0.self_attn.k_proj"])
            self.assertAlmostEqual(k_mat[0][0], 1.0 * 1.0 * 2.0, places=5)

            o_mat = reader.read_matrix_f32(by_name["layers.0.self_attn.o_proj"])
            self.assertAlmostEqual(o_mat[0][0], 1.0 * 1.0 * 0.5, places=5)

            e0 = reader.read_matrix_f32(by_name["layers.0.mlp.experts.0.w13"])
            e1 = reader.read_matrix_f32(by_name["layers.0.mlp.experts.1.w13"])
            self.assertAlmostEqual(e0[0][0], 2.0, places=5)
            self.assertAlmostEqual(e1[0][0], 0.5, places=5)

        self.assertEqual(e2m1_nibble_to_f32(0x2), 1.0)
        self.assertEqual(e2m1_nibble_to_f32(0xF), -6.0)

    def test_bounded_disk_stream_packing_memory_and_http_range(self) -> None:
        shard1_bytes = _safetensors_bytes(
            {
                "model.embed_tokens.weight": ("BF16", [8, 16], _f32_to_bf16_bytes([0.25] * 128)),
            }
        )
        shard2_bytes = _safetensors_bytes(
            {
                "model.layers.0.mlp.gate_proj.weight": ("F8_E4M3", [8, 32], bytes([0x38] * 256)),
                "model.layers.0.mlp.gate_proj.weight_scale": ("U8", [8, 1], bytes([128] * 8)),  # 2.0
            }
        )
        out_mem = self.tmp / "mem_stream.amf"
        sources_mem = [
            ShardStreamSource("s1.safetensors", lambda: MemoryRangeReader(shard1_bytes, "s1")),
            ShardStreamSource("s2.safetensors", lambda: MemoryRangeReader(shard2_bytes, "s2")),
        ]
        metas_mem = pack_amf_from_stream_sources(sources_mem, out_mem, model_id="mem-stream")
        self.assertEqual(len(metas_mem), 2)
        _verify_amf_with_zig(self, out_mem, 2)

        _RangeHttpHandler.blobs = {"s1.safetensors": shard1_bytes, "s2.safetensors": shard2_bytes}
        _RangeHttpHandler.range_requests = []
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHttpHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            out_http = self.tmp / "http_stream.amf"
            sources_http = [
                ShardStreamSource(
                    "http://s1",
                    lambda p=port: HttpRangeStreamReader(f"http://127.0.0.1:{p}/s1.safetensors", max_chunk_bytes=64),
                ),
                ShardStreamSource(
                    "http://s2",
                    lambda p=port: HttpRangeStreamReader(f"http://127.0.0.1:{p}/s2.safetensors", max_chunk_bytes=64),
                ),
            ]
            metas_http = pack_amf_from_stream_sources(sources_http, out_http, model_id="mem-stream")
            self.assertEqual(len(metas_http), 2)
            self.assertGreater(len(_RangeHttpHandler.range_requests), 2)
            self.assertEqual(out_mem.read_bytes(), out_http.read_bytes())
        finally:
            srv.shutdown()
            srv.server_close()


class M81GsqRcoAndImatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_parse_gguf_imatrix_formats_dat_gguf_and_json(self) -> None:
        # 1) Legacy `.dat`
        dat_path = self.tmp / "imatrix.dat"
        name_b = b"blk.0.attn_q.weight"
        dat_bytes = bytearray(struct.pack("<i", 1))
        dat_bytes.extend(struct.pack("<i", len(name_b)) + name_b)
        dat_bytes.extend(struct.pack("<2i", 4, 3))
        dat_bytes.extend(struct.pack("<3f", 4.0, 8.0, 12.0))
        dat_path.write_bytes(bytes(dat_bytes))
        parsed_dat = parse_gguf_imatrix(dat_path)
        self.assertEqual(parsed_dat["layers.0.self_attn.q_proj.weight"], [1.0, 2.0, 3.0])

        # 2) GGUF imatrix
        gguf_im = self.tmp / "imatrix.gguf"
        _write_gguf_file(
            gguf_im,
            version=3,
            kv=[("general.type", 8, "imatrix")],
            tensors=[
                ("blk.0.ffn_down.weight.in_sum2", (4,), 0, struct.pack("<4f", 2.0, 4.0, 6.0, 8.0)),
                ("blk.0.ffn_down.weight.counts", (1,), 0, struct.pack("<1f", 2.0)),
            ],
        )
        parsed_gguf = parse_gguf_imatrix(gguf_im)
        self.assertEqual(parsed_gguf["layers.0.mlp.down_proj.weight"], [1.0, 2.0, 3.0, 4.0])

        # 3) JSON imatrix
        js_path = self.tmp / "imatrix.json"
        js_path.write_text(json.dumps({"layers.0.mlp.gate": [0.5, 1.5]}), encoding="utf-8")
        parsed_js = parse_gguf_imatrix(js_path)
        self.assertEqual(parsed_js["layers.0.mlp.gate"], [0.5, 1.5])

    def test_gsq_rco_protects_special_roles_and_preserves_fp8_nvfp4(self) -> None:
        rng = random.Random(99)
        ckpt = self.tmp / "gsq_ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(json.dumps({"model_type": "hy4"}), encoding="utf-8")

        def bf16_matrix(rows: int, cols: int) -> tuple[str, list[int], bytes]:
            vals = [rng.uniform(-0.8, 0.8) for _ in range(rows * cols)]
            return ("BF16", [rows, cols], _f32_to_bf16_bytes(vals))

        tensors: dict[str, tuple[str, list[int], bytes]] = {
            "model.embed_tokens.weight": bf16_matrix(16, 64),
            "model.layers.0.input_layernorm.weight": ("BF16", [64], _f32_to_bf16_bytes([1.0] * 64)),
            "model.layers.0.self_attn.q_proj.weight": bf16_matrix(16, 64),
            "model.layers.0.mlp.gate.weight": bf16_matrix(8, 64),
            "model.layers.0.mlp.shared_expert.gate_proj.weight": bf16_matrix(16, 64),
            "model.layers.0.controller_a.weight": bf16_matrix(8, 64),
            "model.layers.0.sconv.weight": bf16_matrix(8, 64),
            "model.layers.0.ihc_gate.weight": bf16_matrix(8, 64),
            "model.layers.0.mtp.dflare_proj.weight": bf16_matrix(8, 64),
            "model.layers.0.dfly_proj.weight": bf16_matrix(8, 64),
            "visual.blocks.0.attn.qkv.weight": bf16_matrix(16, 64),
            # Native NVFP4 input on an already-quantized matrix (varied pattern so NVFP4 stays preserved)
            "model.layers.0.mlp.down_proj.weight": (
                "U8",
                [16, 32],
                bytes([((i * 17 + 3) & 0x77) | 0x11 for i in range(16 * 32)]),
            ),
            "model.layers.0.mlp.down_proj.weight_scale": ("F8_E4M3", [16, 4], bytes([0x38, 0x3C, 0x40, 0x34] * 16)),
            "model.layers.0.mlp.down_proj.weight_scale_2": ("F32", [1], struct.pack("<f", 0.5)),
        }
        # Populate enough MoE experts (e=0..5, 64x64) so average bits ~3.8 can protect control/norm/attn roles.
        for e_idx in range(6):
            tensors[f"model.layers.0.mlp.experts.{e_idx}.gate_proj.weight"] = bf16_matrix(64, 64)
            tensors[f"model.layers.0.mlp.experts.{e_idx}.up_proj.weight"] = bf16_matrix(64, 64)

        _write_safetensors(ckpt / "model.safetensors", tensors)
        imatrix_p = self.tmp / "imp.json"
        imatrix_p.write_text(json.dumps({"layers.0.self_attn.q_proj": [10.0], "layers.0.mlp.experts.0.gate_proj": [0.1]}), encoding="utf-8")

        out_amf = self.tmp / "gsq.amf"
        rep = gsq_convert(
            ckpt,
            out_amf,
            target_bits=3.8,
            group_size=64,
            imatrix_path=imatrix_p,
            exclude_tensors=(),
        )
        _verify_amf_with_zig(self, out_amf, len(rep.metas))
        by_name = {m.name: m for m in rep.metas}

        protected_stems = (
            "layers.0.controller_a",
            "layers.0.sconv",
            "layers.0.ihc_gate",
            "layers.0.mtp.dflare_proj",
            "layers.0.dfly_proj",
            "layers.0.mlp.gate",
            "layers.0.mlp.shared_expert.gate_proj",
            "embed_tokens",
        )
        for stem in protected_stems:
            entry = by_name.get(stem) or by_name[f"{stem}.weight"]
            self.assertIn(
                entry.dtype,
                (DType.Q4_AFFINE, DType.Q5_AFFINE, DType.Q6_AFFINE, DType.Q8_AFFINE, DType.F16, DType.BF16),
                msg=stem,
            )
        self.assertEqual(by_name["layers.0.mlp.down_proj"].dtype, DType.NVFP4_G16)
        exp0 = by_name.get("layers.0.mlp.experts.0.gate_proj") or by_name["layers.0.mlp.experts.0.gate_proj.weight"]
        self.assertEqual(exp0.group_size, 64)


class M81GgufRepackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_gguf_v2_and_v3_all_raw_dtypes_and_exact_math(self) -> None:
        k_wide = 256
        n_rows = 2

        def f16_blk_payload(ggml_type: int, n_elem_per_block: int, blk_bytes: int) -> bytes:
            blocks = k_wide // n_elem_per_block
            one_blk = bytearray(blk_bytes)
            # First two bytes are valid f16 scalar `d=0.5` where present (Q/K/IQ/TQ blocks start with d or end/middle)
            if ggml_type == 14:  # Q6_K has d in last 2 bytes of 210B block
                one_blk[208:210] = struct.pack("<e", 0.5)
            elif ggml_type in (34, 35):  # TQ1_0 / TQ2_0 have d in last 2 bytes!
                one_blk[blk_bytes - 2 : blk_bytes] = struct.pack("<e", 0.5)
            elif ggml_type == 39:  # MXFP4: 1B UE8M0 d (=127 -> 1.0) + 16B nibbles
                one_blk[0] = 127
                for j in range(1, 17):
                    one_blk[j] = 0x10  # low index=0 (0.0), high index=1 (0.5)
            elif ggml_type == 40:  # NVFP4: 4B E4M3 scales (0x38=1.0) + 32B split-half nibbles (0x32: low=2->1.0, high=3->1.5)
                one_blk[0:4] = b"\x38\x38\x38\x38"
                for j in range(4, 36):
                    one_blk[j] = 0x32
            elif ggml_type == 42:  # Q2_0: 2B f16 d + 16B nibbles + 16B wait (actually Q2_0 block is 18B: 2B f16 d + 16B 2-bit quants)
                one_blk[0:2] = struct.pack("<e", 0.5)
                for j in range(2, 18):
                    one_blk[j] = 0x55
            elif ggml_type == 29:  # IQ1_M (56B): scales[8] at offset 48..56 (exponent bits 12..15 per u16 -> ~7 for 0.5)
                one_blk[48:56] = struct.pack("<4H", 0x7000, 0x7000, 0x7000, 0x7000)
            else:
                one_blk[0:2] = struct.pack("<e", 0.5)
                if blk_bytes >= 4 and ggml_type in (3, 7, 10, 11, 12, 13):
                    one_blk[2:4] = struct.pack("<e", 0.25)
            return bytes(one_blk) * (n_rows * blocks)

        raw_specs: list[tuple[str, int, DType, int, int]] = [
            ("blk.0.q4_0.weight", 2, DType.GGUF_Q4_0, 32, 18),
            ("blk.0.q4_1.weight", 3, DType.GGUF_Q4_1, 32, 20),
            ("blk.0.q5_0.weight", 6, DType.GGUF_Q5_0, 32, 22),
            ("blk.0.q5_1.weight", 7, DType.GGUF_Q5_1, 32, 24),
            ("blk.0.q8_0.weight", 8, DType.GGUF_Q8_0, 32, 34),
            ("blk.0.q2_k.weight", 10, DType.GGUF_Q2_K, 256, 84),
            ("blk.0.q3_k.weight", 11, DType.GGUF_Q3_K, 256, 110),
            ("blk.0.q4_k.weight", 12, DType.GGUF_Q4_K, 256, 144),
            ("blk.0.q5_k.weight", 13, DType.GGUF_Q5_K, 256, 176),
            ("blk.0.q6_k.weight", 14, DType.GGUF_Q6_K, 256, 210),
            ("blk.0.iq2_xxs.weight", 16, DType.GGUF_IQ2_XXS, 256, 66),
            ("blk.0.iq2_xs.weight", 17, DType.GGUF_IQ2_XS, 256, 74),
            ("blk.0.iq3_xxs.weight", 18, DType.GGUF_IQ3_XXS, 256, 98),
            ("blk.0.iq1_s.weight", 19, DType.GGUF_IQ1_S, 256, 50),
            ("blk.0.iq4_nl.weight", 20, DType.GGUF_IQ4_NL, 32, 18),
            ("blk.0.iq3_s.weight", 21, DType.GGUF_IQ3_S, 256, 110),
            ("blk.0.iq2_s.weight", 22, DType.GGUF_IQ2_S, 256, 82),
            ("blk.0.iq4_xs.weight", 23, DType.GGUF_IQ4_XS, 256, 136),
            ("blk.0.iq1_m.weight", 29, DType.GGUF_IQ1_M, 256, 56),
            ("blk.0.tq1_0.weight", 34, DType.GGUF_TQ1_0, 256, 54),
            ("blk.0.tq2_0.weight", 35, DType.GGUF_TQ2_0, 256, 66),
            ("blk.0.mxfp4.weight", 39, DType.GGUF_MXFP4, 32, 17),
            ("blk.0.nvfp4.weight", 40, DType.NVFP4_G16, 64, 36),
            ("blk.0.q2_0.weight", 42, DType.GGUF_Q2_0, 64, 18),
        ]
        tensors_in: list[tuple[str, tuple[int, ...], int, bytes]] = []
        for tname, gtype, _amf_dt, blk_elems, blk_b in raw_specs:
            tensors_in.append((tname, (k_wide, n_rows), gtype, f16_blk_payload(gtype, blk_elems, blk_b)))
        # Add float & F8_E4M3 tensors (standalone tensor + paired row scale)
        tensors_in.extend(
            [
                ("token_embd.weight", (k_wide, n_rows), 30, _f32_to_bf16_bytes([0.25] * (n_rows * k_wide))),
                ("output_norm.weight", (k_wide,), 0, struct.pack(f"<{k_wide}f", *([1.0] * k_wide))),
                ("blk.0.fp8_t.weight", (k_wide, n_rows), 43, bytes([0x38] * (n_rows * k_wide))),
                ("blk.0.fp8_r.weight", (k_wide, n_rows), 43, bytes([0x40] * (n_rows * k_wide))),
                ("blk.0.fp8_r.weight.scale", (n_rows,), 0, struct.pack("<2f", 1.5, 3.0)),
            ]
        )

        # Test BOTH GGUF v2 and GGUF v3!
        for version in (2, 3):
            gpath = self.tmp / f"all_v{version}.gguf"
            _write_gguf_file(
                gpath,
                version=version,
                kv=[
                    ("general.architecture", 8, "llama"),
                    ("general.name", 8, f"all-v{version}"),
                    ("llama.block_count", 4, 1),
                    ("llama.embedding_length", 4, k_wide),
                    ("llama.attention.head_count", 4, 4),
                    ("tokenizer.ggml.tokens", 9, (8, ["<pad>", "a", "b", "c"])),
                ],
                tensors=tensors_in,
            )
            out_amf = self.tmp / f"all_v{version}.amf"
            metas = pack_amf_from_gguf(gpath, out_amf, embed_metadata=True)
            _verify_amf_with_zig(self, out_amf, len(metas))

            with AmfReader(out_amf) as reader:
                by_name = {m.name: m for m in reader.entries}
                for tname, _gt, exp_dt, _be, _bb in raw_specs:
                    norm = normalize_gguf_tensor_name(tname)
                    m = by_name[norm]
                    self.assertEqual(m.dtype, exp_dt, msg=tname)
                    mat = reader.read_matrix_f32(m)
                    self.assertEqual(len(mat), n_rows)
                    self.assertEqual(len(mat[0]), k_wide)
                    for v in mat[0]:
                        self.assertTrue(math.isfinite(v), msg=f"non-finite in {tname}: {v}")
                # Exact analytical checks for MXFP4, NVFP4(40), F8_E4M3(43) row scale
                mxfp4_mat = reader.read_matrix_f32(by_name[normalize_gguf_tensor_name("blk.0.mxfp4.weight")])
                self.assertEqual(mxfp4_mat[0][0], 0.0)
                self.assertEqual(mxfp4_mat[0][16], 0.5)
                nvfp4_mat = reader.read_matrix_f32(by_name[normalize_gguf_tensor_name("blk.0.nvfp4.weight")])
                self.assertAlmostEqual(nvfp4_mat[0][0], 1.0, places=5)
                self.assertAlmostEqual(nvfp4_mat[0][8], 1.5, places=5)
                fp8_r_mat = reader.read_matrix_f32(by_name[normalize_gguf_tensor_name("blk.0.fp8_r.weight")])
                self.assertAlmostEqual(fp8_r_mat[0][0], 2.0 * 1.5, places=5)
                self.assertAlmostEqual(fp8_r_mat[1][0], 2.0 * 3.0, places=5)

    def test_gguf_split_shards_and_unrolled_recurrent_deduplication(self) -> None:
        k, n = 64, 4
        row_q8_bytes = n * (k // 32) * 34
        # Give each physical layer 0..23 unique weights, and mirror blk.(L+24) identically.
        t_s1: list[tuple[str, tuple[int, ...], int, bytes]] = []
        t_s2: list[tuple[str, tuple[int, ...], int, bytes]] = []
        for layer in range(24):
            payload = bytes([((i + layer * 13 + 1) & 0x7F) for i in range(row_q8_bytes)])
            t_s1.append((f"blk.{layer}.attn_q.weight", (k, n), 8, payload))
            t_s2.append((f"blk.{layer + 24}.attn_q.weight", (k, n), 8, payload))

        s1 = self.tmp / "hy4-00001-of-00002.gguf"
        s2 = self.tmp / "hy4-00002-of-00002.gguf"
        base_kv = [
            ("general.architecture", 8, "hy4"),
            ("hy4.block_count", 4, 48),
            ("hy4.embedding_length", 4, k),
            ("hy4.attention.head_count", 4, 2),
            ("split.no", 4, 0),
            ("split.count", 4, 2),
        ]
        _write_gguf_file(s1, version=3, kv=base_kv, tensors=t_s1)
        _write_gguf_file(s2, version=3, kv=[("split.no", 4, 1), ("split.count", 4, 2)], tensors=t_s2)

        stats: dict[str, int] = {}
        out_amf = self.tmp / "hy4_dedup.amf"
        metas = pack_amf_from_gguf(s1, out_amf, model_id="hy4-dedup", stats_out=stats)
        self.assertEqual(stats.get("deduplicated_layers"), 24)
        self.assertEqual(stats.get("physical_layers"), 24)
        self.assertEqual(len(metas), 24)
        self.assertEqual(metas[0].name, "layers.0.self_attn.q_proj.weight")
        _verify_amf_with_zig(self, out_amf, 24)

    def test_gguf_byte_size_check_rejects_stq1_0_and_unknown_forks(self) -> None:
        # 1) Colliding fork `STQ1_0` overloading ggml_type 43 at ~1.312 bpw instead of F8_E4M3 8.0 bpw
        bad_stq = self.tmp / "bad_stq1_0.gguf"
        k, n = 256, 4
        small_payload = b"\x00" * 168  # ~1.3125 bpw (far below expected 1024 B for F8_E4M3)
        _write_gguf_file(
            bad_stq,
            version=3,
            kv=[("general.architecture", 8, "llama")],
            tensors=[("blk.0.attn_q.weight", (k, n), 43, small_payload)],
        )
        with self.assertRaises(ValueError) as ctx:
            read_gguf_header(bad_stq)
        self.assertIn("STQ1_0", str(ctx.exception))

        # 2) Unknown / colliding fork metadata `vq-gguf` or unknown ggml_type 99
        bad_fork = self.tmp / "bad_fork.gguf"
        _write_gguf_file(
            bad_fork,
            version=3,
            kv=[("general.architecture", 8, "vq-gguf")],
            tensors=[("output_norm.weight", (32,), 0, b"\x00" * 128)],
        )
        with self.assertRaises(ValueError):
            read_gguf_header(bad_fork)

        bad_type = self.tmp / "bad_type.gguf"
        _write_gguf_file(
            bad_type,
            version=3,
            kv=[("general.architecture", 8, "llama")],
            tensors=[("blk.0.x.weight", (64, 2), 99, b"\x00" * 64)],
        )
        with self.assertRaises(ValueError):
            read_gguf_header(bad_type)

    def test_gguf_clip_mmproj_dflash_draft_and_hf_tokenizer_crosscheck_and_cli(self) -> None:
        # 1) clip mmproj routes via W20 policy + VISION tensor kinds
        clip_gguf = self.tmp / "mmproj.gguf"
        _write_gguf_file(
            clip_gguf,
            version=3,
            kv=[("general.architecture", 8, "clip"), ("clip.vision.embedding_length", 4, 32)],
            tensors=[
                ("v.blk.0.attn_q.weight", (32, 8), 1, _f32_to_f16_bytes([0.5] * 256)),
                ("mm.0.weight", (32, 8), 1, _f32_to_f16_bytes([0.25] * 256)),
            ],
        )
        clip_amf = self.tmp / "mmproj.amf"
        clip_metas = pack_amf_from_gguf(clip_gguf, clip_amf)
        _verify_amf_with_zig(self, clip_amf, len(clip_metas))
        weight_metas = [m for m in clip_metas if m.kind != TensorKind.METADATA]
        self.assertGreater(len(weight_metas), 0)
        self.assertTrue(all(is_vision_tensor(m.name) for m in weight_metas))

        # 2) dflash draft GGUF emits draft `.amf` bundle
        dflash_gguf = self.tmp / "dflash_draft.gguf"
        _write_gguf_file(
            dflash_gguf,
            version=3,
            kv=[("general.architecture", 8, "dflash"), ("dflash.block_count", 4, 2), ("dflash.embedding_length", 4, 32)],
            tensors=[("blk.0.attn_q.weight", (32, 4), 1, _f32_to_f16_bytes([0.5] * 128))],
        )
        dflash_amf = self.tmp / "dflash_draft.amf"
        dflash_metas = pack_amf_from_gguf(dflash_gguf, dflash_amf, embed_metadata=True)
        _verify_amf_with_zig(self, dflash_amf, len(dflash_metas))
        with AmfReader(dflash_amf) as reader:
            cfg = json.loads(reader.read_metadata_bytes("__meta.config_json").decode("utf-8"))
            self.assertEqual(cfg.get("draft_kind"), "dflash")
            self.assertEqual(cfg.get("model_type"), "dflash")

        # 3) External HF tokenizer.json precedence + strict vocab mismatch + pack-gguf / pack-amf CLI auto-detect
        llama_gguf = self.tmp / "small.gguf"
        _write_gguf_file(
            llama_gguf,
            version=3,
            kv=[
                ("general.architecture", 8, "llama"),
                ("tokenizer.ggml.tokens", 9, (8, ["<bos>", "a", "b", "c"])),
                ("tokenizer.ggml.bos_token_id", 4, 0),
            ],
            tensors=[("blk.0.attn_q.weight", (32, 4), 1, _f32_to_f16_bytes([0.25] * 128))],
        )
        bad_tok = self.tmp / "bad_tokenizer.json"
        bad_tok.write_text(json.dumps({"model": {"vocab": {"<bos>": 0, "a": 1}}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            pack_amf_from_gguf(llama_gguf, self.tmp / "fail.amf", tokenizer_path=bad_tok, embed_metadata=True)

        good_tok = self.tmp / "good_tokenizer.json"
        good_tok.write_text(
            json.dumps({"model": {"vocab": {"<bos>": 0, "a": 1, "b": 2, "c": 3}}, "added_tokens": []}),
            encoding="utf-8",
        )
        out_cli = self.tmp / "cli_gguf.amf"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["pack-amf", "--input", str(llama_gguf), "--output", str(out_cli), "--tokenizer", str(good_tok), "--embed-metadata"])
        self.assertEqual(rc, 0)
        self.assertIn("packed 3 tensors into", buf.getvalue())
        with AmfReader(out_cli) as r_cli:
            _verify_amf_with_zig(self, out_cli, len(r_cli.entries))

        out_pack_gguf = self.tmp / "pack_gguf_cmd.amf"
        buf_gguf = io.StringIO()
        with redirect_stdout(buf_gguf):
            rc2 = cli_main(["pack-gguf", "--input", str(llama_gguf), "--output", str(out_pack_gguf), "--tokenizer", str(bad_tok), "--no-strict-vocab", "--embed-metadata"])
        self.assertEqual(rc2, 0)
        self.assertIn("from GGUF", buf_gguf.getvalue())
        with AmfReader(out_pack_gguf) as r_gg:
            _verify_amf_with_zig(self, out_pack_gguf, len(r_gg.entries))


class M81ChatTemplateSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_real_metadata_checkpoints_and_deterministic_embedding(self) -> None:
        expected_dialects = {
            "thinkingmachines__Inkling-2-Mini": "inkling2",
            "tencent__Hy4-flash": "hy4",
            "tencent__Hy4-preview": "hy4",
            "Qwen__Qwen3.8-Flash-Next": "chatml",
            "deepseek-ai__DeepSeek-V4.1-Flash-1017": "deepseek",
        }
        for model_dir_name, exp_dialect in expected_dialects.items():
            d = REAL_METADATA_ROOT / model_dir_name
            if not d.is_dir():
                continue
            spec1 = compile_chat_template_from_dir(d)
            spec2 = compile_chat_template_from_dir(d)
            self.assertIsNotNone(spec1, msg=model_dir_name)
            assert spec1 is not None
            self.assertEqual(spec1, spec2)
            self.assertEqual(spec1["schema_version"], 1)
            self.assertEqual(spec1["dialect"], exp_dialect, msg=model_dir_name)
            self.assertTrue(spec1["eos_token"])
            self.assertIn("thinking", spec1)
            self.assertIn("tool_calling", spec1)

    def test_synthetic_dialects_cli_and_amf_metadata_embedding(self) -> None:
        ckpt = self.tmp / "inkling_ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text(json.dumps({"model_type": "inkling2", "eos_token_id": 2}), encoding="utf-8")
        (ckpt / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}), encoding="utf-8")
        (ckpt / "chat_template.jinja").write_text(
            "{% if reasoning_effort == 'high' %}[THINK]{% else %}[SKIP]{% endif %}{{ message['content'] }}<|return|>",
            encoding="utf-8",
        )
        _write_safetensors(ckpt / "model.safetensors", {"model.embed_tokens.weight": ("BF16", [2, 16], _f32_to_bf16_bytes([1.0] * 32))})

        out_json = self.tmp / "ct_spec.json"
        with redirect_stdout(io.StringIO()):
            rc = cli_main(["compile-chat-template", "--input", str(ckpt), "--output", str(out_json)])
        self.assertEqual(rc, 0)
        spec_json = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(spec_json["dialect"], "inkling2")

        out_amf = self.tmp / "with_ct.amf"
        with redirect_stdout(io.StringIO()):
            rc2 = cli_main(["pack-amf", "--input", str(ckpt), "--output", str(out_amf), "--embed-metadata"])
        self.assertEqual(rc2, 0)
        _verify_amf_with_zig(self, out_amf, 4)
        with AmfReader(out_amf) as reader:
            raw_ct = reader.read_metadata_bytes(META_CHAT_TEMPLATE_JSON)
            parsed = json.loads(raw_ct.decode("utf-8"))
            self.assertEqual(parsed["dialect"], "inkling2")
            # Alias lookup also works cleanly
            self.assertEqual(reader.read_metadata_bytes(LEGACY_CHAT_TEMPLATE_METADATA_NAME), raw_ct)

        # Ensure absent template keeps W18 2-blob invariant intact
        plain_ckpt = self.tmp / "plain_ckpt"
        plain_ckpt.mkdir()
        (plain_ckpt / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        (plain_ckpt / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0}}}), encoding="utf-8")
        _write_safetensors(plain_ckpt / "model.safetensors", {"model.embed_tokens.weight": ("BF16", [2, 16], _f32_to_bf16_bytes([1.0] * 32))})
        plain_amf = self.tmp / "plain.amf"
        pack_amf_from_safetensors(plain_ckpt, plain_amf, model_id="plain", embed_metadata=True)
        with AmfReader(plain_amf) as reader:
            self.assertEqual(sum(1 for m in reader.entries if m.kind == TensorKind.METADATA), 2)

        # Custom template fallback
        custom = compile_chat_template_spec(jinja_text="<custom_user>{{ message['content'] }}</custom_user>", tokenizer_config={"eos_token": "<custom_eos>"})
        self.assertEqual(custom["dialect"], "custom")
        self.assertEqual(custom["eos_token"], "<custom_eos>")


if __name__ == "__main__":
    unittest.main()
