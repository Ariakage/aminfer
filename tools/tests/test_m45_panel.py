"""Tests for W15 `sme2_panel64` packing, `repack-sme2`, and Zig round-trip validation."""

from __future__ import annotations

import io
import pathlib
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import (
    ENTRY_STRUCT,
    HEADER_STRUCT,
    NO_EXPERT,
    NO_LAYER,
    SME2_F16_GROUP_BYTES,
    SME2_Q2_GROUP_BYTES,
    SME2_Q3_GROUP_BYTES,
    SME2_Q4_GROUP_BYTES,
    AmfReader,
    DType,
    QuantParamKind,
    TensorKind,
    TensorLayout,
    _crc32,
    decode_reserved0,
    dequantize_q3_row_f32,
    dequantize_q4_row_f32,
    dequantize_sme2_panel64_q2_panel_f32,
    dequantize_sme2_panel64_q3_panel_f32,
    dequantize_sme2_panel64_q4_panel_f32,
    encode_reserved0,
    interleave_scale_bias_f16,
    load_amf_metadata,
    pack_mlx_affine_3bit,
    pack_mlx_affine_4bit,
    pack_sme2_lut16_header,
    pack_sme2_panel64_f16_panel,
    pack_sme2_panel64_q2_panel,
    pack_sme2_panel64_q3_panel,
    pack_sme2_panel64_q4_panel,
    repack_amf_sme2,
    retag_amf_cold_experts,
    sme2_panel64_data_bytes,
    unpack_sme2_panel64_f16_panel,
    write_amf,
)
from aminfer_tools.cli import main

from tests.test_amf import f32_to_bf16_bytes, write_safetensors

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AMINFER_BIN = REPO_ROOT / "zig-out" / "bin" / "aminfer"


def _run_verify_amf(amf_path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    if not AMINFER_BIN.exists():
        raise AssertionError(f"missing {AMINFER_BIN}; run `zig build` first")
    return subprocess.run(
        [str(AMINFER_BIN), "verify-amf", str(amf_path)],
        check=False,
        capture_output=True,
        text=True,
    )


class Sme2Panel64Tests(unittest.TestCase):
    def test_reserved0_encoding_and_struct_sizes(self) -> None:
        self.assertEqual(HEADER_STRUCT.size, 128)
        self.assertEqual(ENTRY_STRUCT.size, 128)
        self.assertEqual(encode_reserved0(TensorLayout.ROW_MAJOR, QuantParamKind.AFFINE_SCALE_BIAS), 0x0000)
        self.assertEqual(encode_reserved0(TensorLayout.SME2_PANEL64, QuantParamKind.AFFINE_SCALE_BIAS), 0x0001)
        self.assertEqual(encode_reserved0(TensorLayout.SME2_PANEL64, QuantParamKind.LUT16_CODEBOOK), 0x0101)
        self.assertEqual(decode_reserved0(0x0000), (TensorLayout.ROW_MAJOR, QuantParamKind.AFFINE_SCALE_BIAS))
        self.assertEqual(decode_reserved0(0x0001), (TensorLayout.SME2_PANEL64, QuantParamKind.AFFINE_SCALE_BIAS))
        self.assertEqual(decode_reserved0(0x0101), (TensorLayout.SME2_PANEL64, QuantParamKind.LUT16_CODEBOOK))
        with self.assertRaises(ValueError):
            decode_reserved0(0x0002)
        with self.assertRaises(ValueError):
            decode_reserved0(0x0201)

    def test_sme2_panel64_data_bytes_and_n_padding(self) -> None:
        self.assertEqual(sme2_panel64_data_bytes(DType.BF16, 16, 64, 0), SME2_F16_GROUP_BYTES)
        self.assertEqual(sme2_panel64_data_bytes(DType.F16, 17, 128, 0), 2 * 2 * SME2_F16_GROUP_BYTES)
        self.assertEqual(sme2_panel64_data_bytes(DType.Q4_AFFINE, 17, 128, 64), 2 * 2 * SME2_Q4_GROUP_BYTES)
        self.assertEqual(sme2_panel64_data_bytes(DType.Q3_AFFINE, 33, 128, 64), 3 * 2 * SME2_Q3_GROUP_BYTES)
        with self.assertRaises(ValueError):
            sme2_panel64_data_bytes(DType.F32, 16, 64, 0)
        with self.assertRaises(ValueError):
            sme2_panel64_data_bytes(DType.BF16, 16, 96, 0)
        with self.assertRaises(ValueError):
            sme2_panel64_data_bytes(DType.Q4_AFFINE, 16, 64, 0)

    def test_f16_bf16_panel_exact_lane_order_and_zero_padding(self) -> None:
        n_active = 5
        k = 128
        rows = [
            struct.pack(f"<{k}H", *[(r * 256 + col + 1) & 0xFFFF for col in range(k)])
            for r in range(n_active)
        ]
        packed = pack_sme2_panel64_f16_panel(rows, k)
        self.assertEqual(len(packed), (k // 64) * SME2_F16_GROUP_BYTES)

        # Verify exact `[W[col_0,k0], W[col_0,k1], ..., W[col_15,k0], W[col_15,k1]]` layout.
        for g in range(k // 64):
            g_off = g * SME2_F16_GROUP_BYTES
            for kp in range(32):
                k0 = g * 64 + 2 * kp
                k1 = k0 + 1
                lanes = struct.unpack_from("<32H", packed, g_off + kp * 64)
                for c in range(16):
                    exp0 = ((c * 256 + k0 + 1) & 0xFFFF) if c < n_active else 0
                    exp1 = ((c * 256 + k1 + 1) & 0xFFFF) if c < n_active else 0
                    self.assertEqual(lanes[2 * c], exp0)
                    self.assertEqual(lanes[2 * c + 1], exp1)

        # Round-trip unpack must match active rows bit-exact and zero rows 5..15.
        unpacked_all = unpack_sme2_panel64_f16_panel(packed, k, num_rows=16)
        for r in range(n_active):
            self.assertEqual(unpacked_all[r], rows[r])
        for r in range(n_active, 16):
            self.assertEqual(unpacked_all[r], bytes(k * 2))

    def test_q4_panel_exact_header_codes_and_lut16(self) -> None:
        n_active = 7
        k = 128
        groups = k // 64
        codes_mat = [[(r * 5 + col * 3 + 1) % 16 for col in range(k)] for r in range(n_active)]
        scales_mat = [[0.125 * (r + 1 + g) for g in range(groups)] for r in range(n_active)]
        biases_mat = [[-0.25 * (r + 1) + 0.5 * g for g in range(groups)] for r in range(n_active)]

        data_rows = [pack_mlx_affine_4bit(codes_mat[r]) for r in range(n_active)]
        param_rows = [
            interleave_scale_bias_f16(
                struct.pack(f"<{groups}e", *scales_mat[r]),
                struct.pack(f"<{groups}e", *biases_mat[r]),
            )
            for r in range(n_active)
        ]

        packed = pack_sme2_panel64_q4_panel(data_rows, param_rows, k)
        self.assertEqual(len(packed), groups * SME2_Q4_GROUP_BYTES)

        for g in range(groups):
            g_off = g * SME2_Q4_GROUP_BYTES
            scale_dup = struct.unpack_from("<32e", packed, g_off)
            bias_dup = struct.unpack_from("<32e", packed, g_off + 64)
            for c in range(16):
                exp_s = struct.unpack("<e", struct.pack("<e", scales_mat[c][g]))[0] if c < n_active else 0.0
                exp_b = struct.unpack("<e", struct.pack("<e", biases_mat[c][g]))[0] if c < n_active else 0.0
                self.assertEqual(scale_dup[2 * c], exp_s)
                self.assertEqual(scale_dup[2 * c + 1], exp_s)
                self.assertEqual(bias_dup[2 * c], exp_b)
                self.assertEqual(bias_dup[2 * c + 1], exp_b)
            code_off = g_off + 128
            for kp in range(32):
                k0 = g * 64 + 2 * kp
                k1 = k0 + 1
                for c in range(16):
                    exp_byte = (
                        (codes_mat[c][k0] & 0x0F) | ((codes_mat[c][k1] & 0x0F) << 4)
                        if c < n_active
                        else 0
                    )
                    self.assertEqual(packed[code_off + kp * 16 + c], exp_byte)

        deq_panel = dequantize_sme2_panel64_q4_panel_f32(packed, k, num_rows=16)
        for r in range(n_active):
            ref = dequantize_q4_row_f32(data_rows[r], param_rows[r], k, group_size=64)
            self.assertEqual(deq_panel[r], ref)
        for r in range(n_active, 16):
            self.assertEqual(deq_panel[r], [0.0] * k)

        # Verify LUT16_CODEBOOK header duplication (`[v0,v0,...,v15,v15]` at 0..64) and dequant.
        cb_vals = [float(i) * 0.25 - 2.0 for i in range(16)]
        cb_bytes = struct.pack("<16e", *cb_vals)
        hdr = pack_sme2_lut16_header(cb_bytes)
        self.assertEqual(len(hdr), 128)
        lut_dup = struct.unpack_from("<32e", hdr, 0)
        for i in range(16):
            self.assertEqual(lut_dup[2 * i], cb_vals[i])
            self.assertEqual(lut_dup[2 * i + 1], cb_vals[i])
        self.assertEqual(hdr[64:128], bytes(64))

        packed_lut = pack_sme2_panel64_q4_panel(
            data_rows,
            None,
            k,
            param_kind=QuantParamKind.LUT16_CODEBOOK,
            group_headers=[hdr] * groups,
        )
        deq_lut = dequantize_sme2_panel64_q4_panel_f32(
            packed_lut, k, num_rows=n_active, param_kind=QuantParamKind.LUT16_CODEBOOK
        )
        for r in range(n_active):
            self.assertEqual(deq_lut[r], [cb_vals[codes_mat[r][col]] for col in range(k)])

    def test_q3_panel_exact_bitplane_contract_and_roundtrip(self) -> None:
        n_active = 9
        k = 128
        groups = k // 64
        codes_mat = [[(r * 7 + col * 5 + 2) % 8 for col in range(k)] for r in range(n_active)]
        scales_mat = [[0.25 * (r + 1) for g in range(groups)] for r in range(n_active)]
        biases_mat = [[-1.0 + 0.125 * (r + g) for g in range(groups)] for r in range(n_active)]

        data_rows = [pack_mlx_affine_3bit(codes_mat[r]) for r in range(n_active)]
        param_rows = [
            interleave_scale_bias_f16(
                struct.pack(f"<{groups}e", *scales_mat[r]),
                struct.pack(f"<{groups}e", *biases_mat[r]),
            )
            for r in range(n_active)
        ]

        packed = pack_sme2_panel64_q3_panel(data_rows, param_rows, k)
        self.assertEqual(len(packed), groups * SME2_Q3_GROUP_BYTES)

        # Verify against explicit reference bit-packing of #84 contract.
        for g in range(groups):
            g_off = g * SME2_Q3_GROUP_BYTES
            code_off = g_off + 128
            for h in range(2):
                kp_base = h * 16
                quads = [[0] * 64 for _ in range(4)]
                for q in range(4):
                    for i in range(64):
                        kp = kp_base + q * 4 + (i // 16)
                        c = i % 16
                        if c < n_active:
                            k0 = g * 64 + 2 * kp
                            k1 = k0 + 1
                            quads[q][i] = (codes_mat[c][k0] & 7) | ((codes_mat[c][k1] & 7) << 4)
                half_off = code_off + h * 192
                for i in range(64):
                    exp_b0 = quads[0][i] | ((quads[3][i] & 0x01) << 3) | ((quads[3][i] & 0x10) << 3)
                    exp_b1 = quads[1][i] | ((quads[3][i] & 0x02) << 2) | ((quads[3][i] & 0x20) << 2)
                    exp_b2 = quads[2][i] | ((quads[3][i] & 0x04) << 1) | ((quads[3][i] & 0x40) << 1)
                    self.assertEqual(packed[half_off + i], exp_b0)
                    self.assertEqual(packed[half_off + 64 + i], exp_b1)
                    self.assertEqual(packed[half_off + 128 + i], exp_b2)

        deq_panel = dequantize_sme2_panel64_q3_panel_f32(packed, k, num_rows=16)
        for r in range(n_active):
            ref = dequantize_q3_row_f32(data_rows[r], param_rows[r], k, group_size=64)
            self.assertEqual(deq_panel[r], ref)
        for r in range(n_active, 16):
            self.assertEqual(deq_panel[r], [0.0] * k)

    def test_q2_panel_exact_luti2_crumb_order(self) -> None:
        n_active = 11
        k = 128
        groups = k // 64
        codes_mat = [[(r * 3 + col * 7 + 1) & 3 for col in range(k)] for r in range(n_active)]
        scales_mat = [[0.5 * (r + 1) for g in range(groups)] for r in range(n_active)]
        biases_mat = [[-0.5 * (r + g) for g in range(groups)] for r in range(n_active)]
        param_rows = [
            interleave_scale_bias_f16(
                struct.pack(f"<{groups}e", *scales_mat[r]),
                struct.pack(f"<{groups}e", *biases_mat[r]),
            )
            for r in range(n_active)
        ]

        packed = pack_sme2_panel64_q2_panel(codes_mat, param_rows, k)
        self.assertEqual(len(packed), groups * SME2_Q2_GROUP_BYTES)

        for g in range(groups):
            code_off = g * SME2_Q2_GROUP_BYTES + 128
            for kp in range(32):
                k0 = g * 64 + 2 * kp
                k1 = k0 + 1
                for b in range(8):
                    c0 = 2 * b
                    c1 = c0 + 1
                    v00 = codes_mat[c0][k0] if c0 < n_active else 0
                    v01 = codes_mat[c0][k1] if c0 < n_active else 0
                    v10 = codes_mat[c1][k0] if c1 < n_active else 0
                    v11 = codes_mat[c1][k1] if c1 < n_active else 0
                    exp = v00 | (v01 << 2) | (v10 << 4) | (v11 << 6)
                    self.assertEqual(packed[code_off + kp * 8 + b], exp)

        deq = dequantize_sme2_panel64_q2_panel_f32(packed, k, num_rows=16)
        for r in range(n_active):
            expected = [
                biases_mat[r][col // 64] + codes_mat[r][col] * scales_mat[r][col // 64]
                for col in range(k)
            ]
            self.assertEqual(deq[r], expected)
        for r in range(n_active, 16):
            self.assertEqual(deq[r], [0.0] * k)

    def _build_mixed_safetensors(self, st_path: pathlib.Path) -> None:
        k = 128
        groups = k // 64

        embed_bf16 = f32_to_bf16_bytes([((r * 13 + c) % 19 - 9) * 0.125 for r in range(32) for c in range(k)])
        norm_bf16 = f32_to_bf16_bytes([1.0 + (c % 5) * 0.0625 for c in range(k)])
        router0_bf16 = f32_to_bf16_bytes([((r * 7 + c * 3) % 17 - 8) * 0.125 for r in range(17) for c in range(k)])

        # Quantized router (Q4, N=19 -> N%16!=0)
        r1_n = 19
        r1_q4 = b"".join(pack_mlx_affine_4bit([(r + c * 3) % 16 for c in range(k)]) for r in range(r1_n))
        r1_s = struct.pack(f"<{r1_n * groups}e", *[0.125 * (1 + (i % 4)) for i in range(r1_n * groups)])
        r1_b = struct.pack(f"<{r1_n * groups}e", *[-0.5 + 0.125 * (i % 3) for i in range(r1_n * groups)])

        # Dense Q4 projection (N=33 -> N%16!=0)
        q_n = 33
        q_q4 = b"".join(pack_mlx_affine_4bit([(r * 5 + c) % 16 for c in range(k)]) for r in range(q_n))
        q_s = f32_to_bf16_bytes([0.25 for _ in range(q_n * groups)])
        q_b = f32_to_bf16_bytes([-1.0 for _ in range(q_n * groups)])

        # Dense F16 projection (N=17 -> N%16!=0)
        o_n = 17
        o_f16 = struct.pack(f"<{o_n * k}e", *[((r * 3 + c) % 11 - 5) * 0.25 for r in range(o_n) for c in range(k)])

        # Hot & cold Q3 routed experts (N=21 -> N%16!=0)
        e_n = 21
        e0_q3 = b"".join(pack_mlx_affine_3bit([(r * 3 + c * 5) % 8 for c in range(k)]) for r in range(e_n))
        e0_s = f32_to_bf16_bytes([0.125 * (1 + (i % 3)) for i in range(e_n * groups)])
        e0_b = f32_to_bf16_bytes([-0.25 for _ in range(e_n * groups)])

        e1_q3 = b"".join(pack_mlx_affine_3bit([(r * 7 + c + 3) % 8 for c in range(k)]) for r in range(e_n))
        e1_s = struct.pack(f"<{e_n * groups}e", *[0.25 for _ in range(e_n * groups)])
        e1_b = struct.pack(f"<{e_n * groups}e", *[-0.75 for _ in range(e_n * groups)])

        lm_head_bf16 = f32_to_bf16_bytes([((r + c * 2) % 15 - 7) * 0.125 for r in range(32) for c in range(k)])
        d2t_i64 = struct.pack("<16q", *[i * 2 for i in range(16)])

        write_safetensors(
            st_path,
            {
                "model.embed_tokens.weight": ("BF16", [32, k], embed_bf16),
                "model.layers.0.input_layernorm.weight": ("BF16", [k], norm_bf16),
                "model.layers.0.mlp.gate.weight": ("BF16", [17, k], router0_bf16),
                "model.layers.1.mlp.gate.weight": ("U32", [r1_n, k * 4 // 32], r1_q4),
                "model.layers.1.mlp.gate.scales": ("F16", [r1_n, groups], r1_s),
                "model.layers.1.mlp.gate.biases": ("F16", [r1_n, groups], r1_b),
                "model.layers.0.self_attn.q_proj.weight": ("U32", [q_n, k * 4 // 32], q_q4),
                "model.layers.0.self_attn.q_proj.scales": ("BF16", [q_n, groups], q_s),
                "model.layers.0.self_attn.q_proj.biases": ("BF16", [q_n, groups], q_b),
                "model.layers.0.self_attn.o_proj.weight": ("F16", [o_n, k], o_f16),
                "model.layers.0.mlp.experts.0.gate_proj.weight": ("U32", [e_n, k * 3 // 32], e0_q3),
                "model.layers.0.mlp.experts.0.gate_proj.scales": ("BF16", [e_n, groups], e0_s),
                "model.layers.0.mlp.experts.0.gate_proj.biases": ("BF16", [e_n, groups], e0_b),
                "model.layers.0.mlp.experts.1.gate_proj.weight": ("U32", [e_n, k * 3 // 32], e1_q3),
                "model.layers.0.mlp.experts.1.gate_proj.scales": ("F16", [e_n, groups], e1_s),
                "model.layers.0.mlp.experts.1.gate_proj.biases": ("F16", [e_n, groups], e1_b),
                "lm_head.weight": ("BF16", [32, k], lm_head_bf16),
                "d2t": ("I64", [16], d2t_i64),
            },
        )

    def test_pack_amf_repack_sme2_and_zig_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            st_path = tmp / "model.safetensors"
            self._build_mixed_safetensors(st_path)

            amf_none = tmp / "none.amf"
            amf_routers = tmp / "routers.amf"
            amf_all = tmp / "all.amf"
            repack_routers = tmp / "repack_routers.amf"
            repack_all = tmp / "repack_all.amf"
            repack_all_twice = tmp / "repack_all_twice.amf"

            for out_path, mode in (
                (amf_none, "none"),
                (amf_routers, "routers"),
                (amf_all, "all-linear"),
            ):
                rc = main(
                    [
                        "pack-amf",
                        "--input",
                        str(st_path),
                        "--output",
                        str(out_path),
                        "--model-id",
                        f"m45-{mode}",
                        "--cold-experts",
                        "0:1",
                        "--sme2-pack",
                        mode,
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertEqual(out_path.stat().st_size % APPLE_PAGE_SIZE, 0)

            # Verify default (`none`) keeps 100% reserved0 == 0 backward compatibility.
            _, none_entries = load_amf_metadata(amf_none)
            self.assertEqual(len(none_entries), 10)
            for m in none_entries:
                self.assertEqual(m.layout, TensorLayout.ROW_MAJOR)
                self.assertEqual(m.quant_param_kind, QuantParamKind.AFFINE_SCALE_BIAS)

            # Verify `routers` packs ONLY router tensors (single-owner: same tensor count & names).
            _, router_entries = load_amf_metadata(amf_routers)
            self.assertEqual([m.name for m in router_entries], [m.name for m in none_entries])
            for m in router_entries:
                if m.kind == TensorKind.ROUTER:
                    self.assertEqual(m.layout, TensorLayout.SME2_PANEL64)
                    self.assertEqual(m.params_offset, 0)
                    self.assertEqual(m.params_bytes, 0)
                    self.assertEqual(
                        m.data_bytes,
                        sme2_panel64_data_bytes(m.dtype, m.shape[0], m.shape[1], m.group_size),
                    )
                else:
                    self.assertEqual(m.layout, TensorLayout.ROW_MAJOR)

            # Verify `all-linear` packs ROUTER/DENSE-2D/HOT_EXPERT/COLD_EXPERT/LM_HEAD,
            # leaving EMBEDDING, NORM, and 1D U32 `d2t` as ROW_MAJOR.
            _, all_entries = load_amf_metadata(amf_all)
            self.assertEqual([m.name for m in all_entries], [m.name for m in none_entries])
            expected_row_major = {
                "embed_tokens.weight",
                "layers.0.input_layernorm.weight",
                "d2t",
            }
            for m in all_entries:
                if m.name in expected_row_major:
                    self.assertEqual(m.layout, TensorLayout.ROW_MAJOR, msg=m.name)
                else:
                    self.assertEqual(m.layout, TensorLayout.SME2_PANEL64, msg=m.name)
                    self.assertEqual(m.params_offset, 0)
                    self.assertEqual(m.params_bytes, 0)
                    self.assertEqual(
                        m.data_bytes,
                        sme2_panel64_data_bytes(m.dtype, m.shape[0], m.shape[1], m.group_size),
                    )

            # Verify `repack-sme2` CLI produces exact parity with `pack-amf --sme2-pack`.
            rc_rr = main(
                [
                    "repack-sme2",
                    "--input",
                    str(amf_none),
                    "--output",
                    str(repack_routers),
                    "--targets",
                    "routers",
                ]
            )
            self.assertEqual(rc_rr, 0)

            rc_ra = main(
                [
                    "repack-sme2",
                    "--input",
                    str(amf_none),
                    "--output",
                    str(repack_all),
                    "--targets",
                    "all-linear",
                ]
            )
            self.assertEqual(rc_ra, 0)

            # Idempotency check: repacking an already-repacked file must be byte-for-byte identical.
            rc_ra2 = main(
                [
                    "repack-sme2",
                    "--input",
                    str(repack_all),
                    "--output",
                    str(repack_all_twice),
                    "--targets",
                    "all-linear",
                ]
            )
            self.assertEqual(rc_ra2, 0)
            self.assertEqual(repack_all.read_bytes(), repack_all_twice.read_bytes())

            # Compare all tensor blobs byte-for-byte between direct pack-amf and repack-sme2
            # (headers differ only in model_id string).
            for direct_p, repacked_p in ((amf_routers, repack_routers), (amf_all, repack_all)):
                with AmfReader(direct_p) as r_dir, AmfReader(repacked_p) as r_rep:
                    self.assertEqual(len(r_dir.entries), len(r_rep.entries))
                    for md, mr in zip(r_dir.entries, r_rep.entries):
                        self.assertEqual(md.name, mr.name)
                        self.assertEqual(md.layout, mr.layout)
                        self.assertEqual(md.quant_param_kind, mr.quant_param_kind)
                        self.assertEqual(md.data_bytes, mr.data_bytes)
                        self.assertEqual(md.params_offset, mr.params_offset)
                        self.assertEqual(md.params_bytes, mr.params_bytes)
                        self.assertEqual(md.blob_crc32, mr.blob_crc32)
                        self.assertEqual(r_dir.read_blob(md, verify_crc=True), r_rep.read_blob(mr, verify_crc=True))

            # Verify AmfReader decodes ROW_MAJOR and SME2_PANEL64 to identical f32 & f16 values.
            with AmfReader(amf_none) as r_row, AmfReader(repack_all) as r_sme:
                for m_row in r_row.entries:
                    m_sme = r_sme.by_name[m_row.name]
                    if len(m_row.shape) == 2 and m_row.dtype != DType.U32:
                        self.assertEqual(r_row.read_matrix_f32(m_row), r_sme.read_matrix_f32(m_sme), msg=m_row.name)
                        sample_rows = [0, m_row.shape[0] // 2, m_row.shape[0] - 1]
                        self.assertEqual(
                            r_row.read_rows_f32(m_row, sample_rows),
                            r_sme.read_rows_f32(m_sme, sample_rows),
                            msg=m_row.name,
                        )
                        self.assertEqual(
                            r_row.read_tensor_f16_bytes(m_row),
                            r_sme.read_tensor_f16_bytes(m_sme),
                            msg=m_row.name,
                        )

            # Verify `retag-cold` preserves `SME2_PANEL64` reserved0 tags on experts.
            retagged = tmp / "retagged.amf"
            hot_cnt, cold_cnt = retag_amf_cold_experts(repack_all, {(0, 0)}, output_path=retagged)
            self.assertEqual((hot_cnt, cold_cnt), (1, 1))
            _, retagged_entries = load_amf_metadata(retagged)
            by_name = {m.name: m for m in retagged_entries}
            self.assertEqual(by_name["layers.0.mlp.experts.0.gate_proj"].kind, TensorKind.COLD_EXPERT)
            self.assertEqual(by_name["layers.0.mlp.experts.0.gate_proj"].layout, TensorLayout.SME2_PANEL64)
            self.assertEqual(by_name["layers.0.mlp.experts.1.gate_proj"].kind, TensorKind.HOT_EXPERT)
            self.assertEqual(by_name["layers.0.mlp.experts.1.gate_proj"].layout, TensorLayout.SME2_PANEL64)

            # Cross-language Zig `verify-amf` must accept EVERY single generated .amf file!
            for amf_file in (amf_none, amf_routers, amf_all, repack_routers, repack_all, retagged):
                proc = _run_verify_amf(amf_file)
                self.assertEqual(
                    proc.returncode,
                    0,
                    msg=f"verify-amf failed for {amf_file.name}:\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}",
                )
                self.assertIn("amf.status       ok", proc.stdout)
                self.assertIn("amf.tensors      10", proc.stdout)

    def test_zig_accepts_lut16_codebook_and_rejects_corrupted_reserved0(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            k = 64
            n = 17
            cb = struct.pack("<16e", *[float(i) * 0.25 for i in range(16)])
            hdr = pack_sme2_lut16_header(cb)
            rows_p0 = [pack_mlx_affine_4bit([(r + c) % 16 for c in range(k)]) for r in range(16)]
            rows_p1 = [pack_mlx_affine_4bit([(16 + c) % 16 for c in range(k)])]
            blob_data = (
                pack_sme2_panel64_q4_panel(rows_p0, None, k, param_kind=QuantParamKind.LUT16_CODEBOOK, group_headers=[hdr])
                + pack_sme2_panel64_q4_panel(rows_p1, None, k, param_kind=QuantParamKind.LUT16_CODEBOOK, group_headers=[hdr])
            )

            lut_amf = tmp / "lut16.amf"
            write_amf(
                lut_amf,
                "lut16-smoke",
                [
                    (
                        "layers.0.mlp.gate",
                        TensorKind.ROUTER,
                        DType.Q4_AFFINE,
                        4,
                        64,
                        0,
                        NO_EXPERT,
                        (n, k),
                        blob_data,
                        b"",
                        TensorLayout.SME2_PANEL64,
                        QuantParamKind.LUT16_CODEBOOK,
                    )
                ],
            )
            proc = _run_verify_amf(lut_amf)
            self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            self.assertIn("amf.status       ok", proc.stdout)

            # Mutate reserved0 at byte offset 58 of directory entry 0, recompute CRCs, and assert Zig rejects.
            raw_good = bytearray(lut_amf.read_bytes())
            for bad_reserved0, label in ((0x0002, "bad_layout"), (0x0201, "bad_param_kind")):
                mut = bytearray(raw_good)
                struct.pack_into("<H", mut, APPLE_PAGE_SIZE + 58, bad_reserved0)
                dir_crc = _crc32(mut[APPLE_PAGE_SIZE : APPLE_PAGE_SIZE + ENTRY_STRUCT.size])
                struct.pack_into("<I", mut, 48, dir_crc)
                struct.pack_into("<I", mut, 52, 0)
                hdr_crc = _crc32(mut[0 : HEADER_STRUCT.size])
                struct.pack_into("<I", mut, 52, hdr_crc)
                bad_path = tmp / f"{label}.amf"
                bad_path.write_bytes(mut)
                bad_proc = _run_verify_amf(bad_path)
                self.assertNotEqual(bad_proc.returncode, 0, msg=f"expected Zig to reject {label}")

    def test_repack_sme2_rejects_corrupted_source_blob_and_cli_help(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            st_path = tmp / "model.safetensors"
            self._build_mixed_safetensors(st_path)
            src_amf = tmp / "src.amf"
            self.assertEqual(
                main(["pack-amf", "--input", str(st_path), "--output", str(src_amf), "--model-id", "crc-check"]),
                0,
            )
            # Corrupt first byte of first blob; repack-sme2 must detect source CRC mismatch.
            raw = bytearray(src_amf.read_bytes())
            raw[2 * APPLE_PAGE_SIZE] ^= 0xFF
            src_amf.write_bytes(raw)
            with self.assertRaises(ValueError):
                repack_amf_sme2(src_amf, tmp / "out.amf", targets="all-linear")

        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            main(["pack-amf", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        for token in ("--sme2-pack", "none", "routers", "all-linear"):
            self.assertIn(token, out.getvalue())

        out2 = io.StringIO()
        with redirect_stdout(out2), self.assertRaises(SystemExit) as ctx2:
            main(["repack-sme2", "--help"])
        self.assertEqual(ctx2.exception.code, 0)
        for token in ("--input", "--output", "--targets", "routers", "all-linear"):
            self.assertIn(token, out2.getvalue())

    def test_multipage_sme2_panel_dtype_conversion_and_inplace_repack(self) -> None:
        """Exercise >16 KiB multi-page SME2 blobs, bf16->f16 conversion, and atomic in-place repack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            st_path = tmp / "big.safetensors"
            n, k = 65, 256  # N_pad = 80 (5 panels), 4 groups -> 5 * 4 * 2048 = 40960 B (3 pages = 49152 B)
            vals = [((r * 11 + c * 3) % 31 - 15) * 0.125 for r in range(n) for c in range(k)]
            write_safetensors(
                st_path,
                {
                    "model.layers.0.mlp.gate.weight": ("BF16", [n, k], f32_to_bf16_bytes(vals)),
                },
            )

            amf_f16_row = tmp / "f16_row.amf"
            amf_f16_sme = tmp / "f16_sme.amf"
            self.assertEqual(
                main(
                    [
                        "pack-amf",
                        "--input",
                        str(st_path),
                        "--output",
                        str(amf_f16_row),
                        "--model-id",
                        "big-f16",
                        "--dtype",
                        "f16",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "pack-amf",
                        "--input",
                        str(st_path),
                        "--output",
                        str(amf_f16_sme),
                        "--model-id",
                        "big-f16",
                        "--dtype",
                        "f16",
                        "--sme2-pack",
                        "routers",
                    ]
                ),
                0,
            )

            # In-place repack on amf_f16_row must produce a byte-for-byte identical file to direct pack.
            self.assertEqual(
                main(
                    [
                        "repack-sme2",
                        "--input",
                        str(amf_f16_row),
                        "--output",
                        str(amf_f16_row),
                        "--targets",
                        "routers",
                    ]
                ),
                0,
            )
            self.assertEqual(amf_f16_row.read_bytes(), amf_f16_sme.read_bytes())

            _, metas = load_amf_metadata(amf_f16_sme)
            self.assertEqual(len(metas), 1)
            m = metas[0]
            self.assertEqual(m.dtype, DType.F16)
            self.assertEqual(m.layout, TensorLayout.SME2_PANEL64)
            self.assertEqual(m.data_bytes, 5 * 4 * SME2_F16_GROUP_BYTES)
            self.assertEqual(m.blob_bytes, 3 * APPLE_PAGE_SIZE)

            proc = _run_verify_amf(amf_f16_sme)
            self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            self.assertIn("amf.status       ok", proc.stdout)

            with AmfReader(amf_f16_sme) as reader:
                got = reader.read_matrix_f32(m)
                for r in range(n):
                    self.assertEqual(got[r], vals[r * k : (r + 1) * k])

    def test_python_repack_sme2_zig_numerical_decode_spec_and_trng_parity(self) -> None:
        """End-to-end cross-language Python repack-sme2 -> Zig SME2 decode/spec-decode numerical parity & D11 TRNG."""
        if not AMINFER_BIN.exists():
            raise AssertionError(f"missing {AMINFER_BIN}; run `zig build` first")

        def parse_field(stdout: str, key: str) -> str:
            for line in stdout.splitlines():
                if line.startswith(key):
                    return line[len(key) :].strip()
            raise AssertionError(f"missing {key!r} in output:\n{stdout}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            for preset, target_modes in (("tiny-moe", ("routers", "all-linear")), ("tiny-dense", ("all-linear",))):
                out_dir = tmp / preset
                rc = main(["synth-qwen3", "--preset", preset, "--output-dir", str(out_dir)])
                self.assertEqual(rc, 0)
                amf_row = out_dir / "model.amf"
                cfg_path = out_dir / "config.json"
                tok_path = out_dir / "tokenizer.json"

                base_dec = subprocess.run(
                    [
                        str(AMINFER_BIN),
                        "decode",
                        "--amf",
                        str(amf_row),
                        "--config",
                        str(cfg_path),
                        "--tokenizer",
                        str(tok_path),
                        "--prompt",
                        "hello world",
                        "--max-tokens",
                        "6",
                        "--ignore-eos",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self.assertEqual(base_dec.returncode, 0, msg=f"{base_dec.stderr}\n{base_dec.stdout}")
                base_toks = parse_field(base_dec.stdout, "decode.token_ids")
                self.assertEqual(parse_field(base_dec.stdout, "decode.swapouts_delta"), "0")

                for tmode in target_modes:
                    amf_sme = out_dir / f"model_{tmode.replace('-', '_')}.amf"
                    rc_rp = main(
                        [
                            "repack-sme2",
                            "--input",
                            str(amf_row),
                            "--output",
                            str(amf_sme),
                            "--targets",
                            tmode,
                        ]
                    )
                    self.assertEqual(rc_rp, 0)

                    sme_dec = subprocess.run(
                        [
                            str(AMINFER_BIN),
                            "decode",
                            "--amf",
                            str(amf_sme),
                            "--config",
                            str(cfg_path),
                            "--tokenizer",
                            str(tok_path),
                            "--prompt",
                            "hello world",
                            "--max-tokens",
                            "6",
                            "--ignore-eos",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    self.assertEqual(sme_dec.returncode, 0, msg=f"{sme_dec.stderr}\n{sme_dec.stdout}")
                    self.assertEqual(parse_field(sme_dec.stdout, "decode.token_ids"), base_toks)
                    self.assertEqual(parse_field(sme_dec.stdout, "decode.sme2_weights"), "yes")
                    self.assertEqual(parse_field(sme_dec.stdout, "decode.swapouts_delta"), "0")

                    sme_spec = subprocess.run(
                        [
                            str(AMINFER_BIN),
                            "spec-decode",
                            "--amf",
                            str(amf_sme),
                            "--config",
                            str(cfg_path),
                            "--tokenizer",
                            str(tok_path),
                            "--prompt",
                            "hello world",
                            "--max-tokens",
                            "6",
                            "--ignore-eos",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    self.assertEqual(sme_spec.returncode, 0, msg=f"{sme_spec.stderr}\n{sme_spec.stdout}")
                    self.assertEqual(parse_field(sme_spec.stdout, "spec.token_ids"), base_toks)
                    self.assertEqual(parse_field(sme_spec.stdout, "spec.swapouts_delta"), "0")
                    if preset == "tiny-moe":
                        self.assertIn("sme2_single_owner", parse_field(sme_spec.stdout, "spec.router_pred_backend"))

            # Verify D11 Secure Enclave TRNG seeding & bit-exact explicit replay via CLI `decode`.
            dense_dir = tmp / "tiny-dense"
            trng_runs = []
            for _ in range(2):
                p = subprocess.run(
                    [
                        str(AMINFER_BIN),
                        "decode",
                        "--amf",
                        str(dense_dir / "model.amf"),
                        "--config",
                        str(dense_dir / "config.json"),
                        "--tokenizer",
                        str(dense_dir / "tokenizer.json"),
                        "--prompt",
                        "hello world",
                        "--max-tokens",
                        "6",
                        "--temperature",
                        "0.8",
                        "--ignore-eos",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self.assertEqual(p.returncode, 0, msg=f"{p.stderr}\n{p.stdout}")
                self.assertEqual(parse_field(p.stdout, "decode.seed_source"), "hardware_trng")
                trng_runs.append((parse_field(p.stdout, "decode.seed_used"), parse_field(p.stdout, "decode.token_ids")))
            self.assertNotEqual(trng_runs[0][0], trng_runs[1][0])

            replay = subprocess.run(
                [
                    str(AMINFER_BIN),
                    "decode",
                    "--amf",
                    str(dense_dir / "model.amf"),
                    "--config",
                    str(dense_dir / "config.json"),
                    "--tokenizer",
                    str(dense_dir / "tokenizer.json"),
                    "--prompt",
                    "hello world",
                    "--max-tokens",
                    "6",
                    "--temperature",
                    "0.8",
                    "--seed",
                    trng_runs[0][0],
                    "--ignore-eos",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self.assertEqual(replay.returncode, 0, msg=f"{replay.stderr}\n{replay.stdout}")
            self.assertEqual(parse_field(replay.stdout, "decode.seed_source"), "explicit")
            self.assertEqual(parse_field(replay.stdout, "decode.seed_used"), trng_runs[0][0])
            self.assertEqual(parse_field(replay.stdout, "decode.token_ids"), trng_runs[0][1])


if __name__ == "__main__":
    unittest.main()
