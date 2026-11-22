"""Tests for W11 DeepSeek-V4.1-Flash-1017 (`deepseek41`) synth packer, shard partitioner, and converter."""

from __future__ import annotations

import json
import pathlib
import struct
import tempfile
import unittest

from aminfer_tools import APPLE_PAGE_SIZE
from aminfer_tools.amf import AmfReader, DType, NO_EXPERT, TensorKind
from aminfer_tools.cli import main as cli_main
from aminfer_tools.deepseek41 import (
    DEEPSEEK41_PRESETS,
    DeepSeek41Config,
    HardwareTierError,
    convert_deepseek41_checkpoint,
    evaluate_hardware_tier,
    generate_synth_deepseek41_bundle,
    partition_deepseek41_amf_shards,
    validate_config_hardware_tier,
)


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header: dict[str, object] = {}
    cursor = 0
    chunks: list[bytes] = []
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(payload)],
        }
        chunks.append(payload)
        cursor += len(payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for c in chunks:
            fh.write(c)


def _make_bf16_payload(count: int, seed_offset: int = 1) -> bytes:
    vals = []
    for i in range(count):
        f = ((i + seed_offset) % 17 - 8) * 0.0625
        u32 = struct.unpack("<I", struct.pack("<f", f))[0]
        vals.append(u32 >> 16)
    return struct.pack(f"<{count}H", *vals)


def _create_multi_shard_deepseek41_checkpoint(root: pathlib.Path, *, full_552b_config_only: bool = False) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    h = 64
    vocab = 128
    kv_lora = 32
    q_lora = 64
    moe_inter = 64
    num_experts = 4

    shard1_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
        "model.embed_tokens.weight": ("BF16", (vocab, h), _make_bf16_payload(vocab * h, 1)),
        "model.layers.0.input_layernorm.weight": ("BF16", (h,), _make_bf16_payload(h, 2)),
        "model.layers.0.mhc.pre_mix.weight": ("BF16", (4, h), _make_bf16_payload(4 * h, 3)),
        "model.layers.0.self_attn.q_a_proj.weight": ("BF16", (q_lora, h), _make_bf16_payload(q_lora * h, 4)),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": ("BF16", (kv_lora + 16, h), _make_bf16_payload((kv_lora + 16) * h, 5)),
        "model.layers.0.mlp.gate.weight": ("BF16", (num_experts, h), _make_bf16_payload(num_experts * h, 6)),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": ("BF16", (moe_inter, h), _make_bf16_payload(moe_inter * h, 7)),
        "model.layers.0.mlp.shared_expert.up_proj.weight": ("BF16", (moe_inter, h), _make_bf16_payload(moe_inter * h, 8)),
        "model.layers.0.mlp.shared_expert.down_proj.weight": ("BF16", (h, moe_inter), _make_bf16_payload(h * moe_inter, 9)),
        "model.enc_dec.global_kv_a_proj.weight": ("BF16", (kv_lora + 16, h), _make_bf16_payload((kv_lora + 16) * h, 10)),
        "visual.patch_embed.weight": ("BF16", (16, 16), _make_bf16_payload(256, 11)),
    }
    shard2_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
        "model.norm.weight": ("BF16", (h,), _make_bf16_payload(h, 12)),
        "lm_head.weight": ("BF16", (vocab, h), _make_bf16_payload(vocab * h, 13)),
        # Packed Engram conditional memory table: [32 rows, 2 * h] ([k_h || v_h])
        "model.engram.layers.0.weight": ("BF16", (32, h * 2), _make_bf16_payload(32 * h * 2, 14)),
        # Optional DSpark head weights
        "dspark.enc_proj.weight": ("BF16", (h, h), _make_bf16_payload(h * h, 15)),
        "dspark.slot.0.proj.weight": ("BF16", (h, h), _make_bf16_payload(h * h, 16)),
        "dspark.slot.0.embed.weight": ("BF16", (h,), _make_bf16_payload(h, 17)),
    }
    for e in range(num_experts):
        shard2_tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = (
            "BF16",
            (moe_inter, h),
            _make_bf16_payload(moe_inter * h, 20 + e),
        )
        shard2_tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = (
            "BF16",
            (moe_inter, h),
            _make_bf16_payload(moe_inter * h, 30 + e),
        )
        shard2_tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = (
            "BF16",
            (h, moe_inter),
            _make_bf16_payload(h * moe_inter, 40 + e),
        )

    s1_name = "model-00001-of-00002.safetensors"
    s2_name = "model-00002-of-00002.safetensors"
    _write_safetensors(root / s1_name, shard1_tensors)
    _write_safetensors(root / s2_name, shard2_tensors)

    weight_map = {k: s1_name for k in shard1_tensors}
    weight_map.update({k: s2_name for k in shard2_tensors})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "pt"}, "weight_map": weight_map}, indent=2) + "\n",
        encoding="utf-8",
    )

    if full_552b_config_only:
        cfg_dict = DEEPSEEK41_PRESETS["full-deepseek41-552b"].to_config_dict()
    else:
        cfg_dict = {
            "model_type": "deepseek41",
            "backbone_kind": "causal_enc_dec",
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "hidden_size": h,
            "vocab_size": vocab,
            "num_experts": num_experts,
            "encoder_moe_top_k": 2,
            "decoder_moe_top_k": 2,
            "moe_intermediate_size": moe_inter,
            "kv_lora_rank": kv_lora,
            "q_lora_rank": q_lora,
            "residual_kind": "mhc_single_pass",
            "mhc_num_streams": 4,
            "memory_lookup_kind": "engram",
            "engram_num_heads": 3,
            "engram_orders": [2, 3, 4],
            "tie_word_embeddings": False,
        }
    (root / "config.json").write_text(json.dumps(cfg_dict, indent=2) + "\n", encoding="utf-8")
    return root


class TestDeepSeek41ConfigAndTier(unittest.TestCase):
    def test_full_552b_preset_geometry_and_890_byte_fp4_kv(self) -> None:
        cfg = DEEPSEEK41_PRESETS["full-deepseek41-552b"]
        cfg.validate()
        self.assertEqual(cfg.model_type, "deepseek41")
        self.assertEqual(cfg.backbone_kind, "causal_enc_dec")
        self.assertEqual(cfg.num_encoder_layers, 20)
        self.assertEqual(cfg.num_decoder_layers, 20)
        self.assertEqual(cfg.total_layers, 40)
        self.assertEqual(cfg.num_experts, 512)
        self.assertEqual(cfg.encoder_moe_top_k, 8)
        self.assertEqual(cfg.decoder_moe_top_k, 16)
        self.assertEqual(cfg.residual_kind, "mhc_single_pass")
        self.assertEqual(cfg.mhc_num_streams, 4)
        self.assertEqual(cfg.memory_lookup_kind, "engram")
        self.assertTrue(cfg.fp4_kv)
        # ADR-0015 / G11 exact compressed FP4 MLA KV footprint: 890 B/token.
        self.assertEqual(cfg.fp4_kv_bytes_per_token(), 890)

        total_b, non_exp_b, exp_b = cfg.estimated_backbone_bytes()
        total_gib = total_b / (1024**3)
        self.assertGreater(total_gib, 200.0)
        self.assertLess(total_gib, 300.0)
        self.assertGreater(exp_b, non_exp_b)

    def test_config_validation_rejects_invalid_engram_and_csa2(self) -> None:
        base = DEEPSEEK41_PRESETS["tiny-deepseek41"]
        with self.assertRaises(ValueError):
            base.with_overrides(engram_num_heads=1, engram_orders=(2,)).validate()
        with self.assertRaises(ValueError):
            base.with_overrides(engram_num_heads=3, engram_orders=(2, 3, 6)).validate()
        with self.assertRaises(ValueError):
            base.with_overrides(csa2_modes=("full", "reuse", "reuse", "reindex")).validate()
        with self.assertRaises(ValueError):
            base.with_overrides(mhc_num_streams=1).validate()

    def test_hardware_tier_rejects_24gb_m4_full_552b_and_allows_512gb_or_cluster(self) -> None:
        full_cfg = DEEPSEEK41_PRESETS["full-deepseek41-552b"]
        with self.assertRaises(HardwareTierError) as ctx:
            validate_config_hardware_tier(full_cfg, target_tier="24gb-m4", world_size=1)
        msg = str(ctx.exception)
        self.assertIn("512gb-ultra", msg)
        self.assertIn("world-size", msg)
        self.assertIn("24gb-m4", msg)

        # Single-node 512 GB Ultra passes cleanly.
        ultra_ok = validate_config_hardware_tier(full_cfg, target_tier="512gb-ultra", world_size=1)
        self.assertTrue(ultra_ok.fits)
        self.assertEqual(ultra_ok.target_tier, "512gb-ultra")

        # 4-node 128 GB Max Thunderbolt 5 RDMA cluster passes cleanly (~65 GiB/rank <= 96 GiB).
        cluster_ok = validate_config_hardware_tier(full_cfg, target_tier="cluster", host_ram_gib=128.0, world_size=4)
        self.assertTrue(cluster_ok.fits)
        self.assertEqual(cluster_ok.world_size, 4)

        # Direct byte check also catches 24 GB M4 auto-detection.
        with self.assertRaises(HardwareTierError):
            evaluate_hardware_tier(242 * (1024**3), target_tier="auto", host_ram_gib=24.0, world_size=1)


class TestDeepSeek41SynthAndPartition(unittest.TestCase):
    def test_generate_synth_tiny_determinism_alignment_engram_and_dspark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d1 = pathlib.Path(tmp) / "run1"
            d2 = pathlib.Path(tmp) / "run2"

            b1 = generate_synth_deepseek41_bundle(d1, preset="tiny-deepseek41", seed=123, cold_experts_per_layer=2)
            b2 = generate_synth_deepseek41_bundle(d2, preset="tiny-deepseek41", seed=123, cold_experts_per_layer=2)

            # Bit-exact determinism across runs.
            self.assertEqual(b1.amf_path.read_bytes(), b2.amf_path.read_bytes())
            self.assertEqual(b1.engram_paths[0].read_bytes(), b2.engram_paths[0].read_bytes())
            self.assertEqual(b1.dspark_amf_path.read_bytes(), b2.dspark_amf_path.read_bytes())

            # Validate backbone .amf alignment, CRC32, CED/mHC/CSA2/MoE tensors, and name length <= 47 B.
            self.assertEqual(b1.amf_path.stat().st_size % APPLE_PAGE_SIZE, 0)
            with AmfReader(b1.amf_path, verify_header_crc=True, verify_dir_crc=True) as r:
                names = set(r.by_name.keys())
                self.assertIn("embed_tokens.weight", names)
                self.assertIn("enc_dec.global_kv_a_proj", names)
                self.assertIn("enc_dec.global_kv_b_proj", names)
                self.assertIn("layers.0.mhc.pre_mix.weight", names)
                self.assertIn("layers.0.self_attn.csa2_idx_proj", names)  # layer 0 is "full"
                self.assertNotIn("layers.1.self_attn.csa2_idx_proj", names)  # layer 1 is "reuse"
                self.assertIn("layers.2.self_attn.csa2_idx_proj", names)  # decoder layer 0 is "full"
                self.assertIn("layers.3.self_attn.csa2_idx_proj", names)  # decoder layer 1 is "reindex"
                self.assertIn("layers.0.mlp.shared_expert.gate_proj", names)
                self.assertIn("layers.3.mlp.experts.7.down_proj", names)

                cold_count = 0
                hot_count = 0
                for e in r.entries:
                    self.assertLessEqual(len(e.name.encode("utf-8")), 47)
                    self.assertEqual(e.blob_offset % APPLE_PAGE_SIZE, 0)
                    self.assertEqual(e.blob_bytes % APPLE_PAGE_SIZE, 0)
                    r.read_blob(e, verify_crc=True)
                    if e.kind == TensorKind.COLD_EXPERT:
                        cold_count += 1
                    elif e.kind == TensorKind.HOT_EXPERT:
                        hot_count += 1
                self.assertGreater(cold_count, 0)
                self.assertGreater(hot_count, 0)

            # Validate sharded Engram .amf (`SPARSE_TABLE`, packed [rows, 2 * hidden_size]).
            self.assertEqual(len(b1.engram_paths), 2)
            total_l0_rows = 0
            for ep in b1.engram_paths:
                self.assertEqual(ep.stat().st_size % APPLE_PAGE_SIZE, 0)
                with AmfReader(ep, verify_header_crc=True, verify_dir_crc=True) as er:
                    self.assertEqual(len(er.entries), b1.config.total_layers)
                    for e in er.entries:
                        self.assertEqual(e.kind, TensorKind.SPARSE_TABLE)
                        self.assertEqual(e.shape[1], b1.config.hidden_size * 2)
                        self.assertTrue(e.name.startswith("engram.layers."))
                        er.read_blob(e, verify_crc=True)
                        if e.layer == 0:
                            total_l0_rows += e.shape[0]
            self.assertEqual(total_l0_rows, b1.config.engram_num_rows)

            # Validate DSpark head .amf and config.json / tokenizer.json.
            self.assertIsNotNone(b1.dspark_amf_path)
            with AmfReader(b1.dspark_amf_path, verify_header_crc=True, verify_dir_crc=True) as dr:
                self.assertIn("dspark.enc_proj.weight", dr.by_name)
                self.assertIn("dspark.slot.3.proj.weight", dr.by_name)
                for e in dr.entries:
                    dr.read_blob(e, verify_crc=True)

            cfg_json = json.loads(b1.config_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg_json["model_type"], "deepseek41")
            self.assertEqual(cfg_json["backbone_kind"], "causal_enc_dec")
            self.assertEqual(cfg_json["residual_kind"], "mhc_single_pass")
            self.assertEqual(cfg_json["memory_lookup_kind"], "engram")
            self.assertEqual(cfg_json["engram_orders"], [2, 3, 4])
            self.assertEqual(len(cfg_json["engram_files"]), 2)
            tok_json = json.loads(b1.tokenizer_path.read_text(encoding="utf-8"))
            self.assertEqual(tok_json["model"]["type"], "BPE")

    def test_512_expert_synth_and_huge_synth_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Guard rejects full-552b disk materialization by default.
            with self.assertRaises(HardwareTierError):
                generate_synth_deepseek41_bundle(pathlib.Path(tmp) / "huge", preset="full-deepseek41-552b")

            # Verify 512 routed experts end-to-end with 1 enc + 1 dec layer.
            cfg_512 = DEEPSEEK41_PRESETS["scale-deepseek41-512e"].with_overrides(
                num_encoder_layers=1,
                num_decoder_layers=1,
                csa2_modes=("full", "full"),
                include_dspark=False,
                engram_shards=1,
                engram_num_rows=32,
            )
            bundle = generate_synth_deepseek41_bundle(pathlib.Path(tmp) / "e512", config=cfg_512, seed=7)
            with AmfReader(bundle.amf_path, verify_header_crc=True, verify_dir_crc=True) as r:
                expert_ids = {e.expert for e in r.entries if e.expert != NO_EXPERT}
                self.assertEqual(len(expert_ids), 512)
                self.assertEqual(min(expert_ids), 0)
                self.assertEqual(max(expert_ids), 511)

    def test_partition_deepseek41_amf_shards_contiguous_and_round_robin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = pathlib.Path(tmp) / "base"
            bundle = generate_synth_deepseek41_bundle(base_dir, preset="tiny-deepseek41", seed=99)

            for world_size, strategy in ((2, "contiguous_by_rank"), (4, "round_robin")):
                shard_dir = pathlib.Path(tmp) / f"shards_{world_size}_{strategy}"
                rep = partition_deepseek41_amf_shards(
                    bundle.amf_path,
                    world_size,
                    shard_dir,
                    strategy=strategy,
                )
                self.assertEqual(rep.world_size, world_size)
                self.assertEqual(rep.num_experts, 8)
                self.assertEqual(len(rep.rank_amf_paths), world_size)

                manifest = json.loads(rep.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["world_size"], world_size)
                self.assertEqual(manifest["strategy"], strategy)
                self.assertEqual(len(manifest["ranks"]), world_size)
                self.assertEqual(len(manifest["expert_map"]), 8)

                seen_experts: set[int] = set()
                for r_idx, r_path in enumerate(rep.rank_amf_paths):
                    self.assertEqual(r_path.stat().st_size % APPLE_PAGE_SIZE, 0)
                    with AmfReader(r_path, verify_header_crc=True, verify_dir_crc=True) as rr:
                        has_embed = "embed_tokens.weight" in rr.by_name
                        self.assertEqual(has_embed, r_idx == 0)
                        rank_experts = {e.expert for e in rr.entries if e.expert != NO_EXPERT}
                        self.assertEqual(sorted(rank_experts), rep.owned_experts_by_rank[r_idx])
                        self.assertTrue(seen_experts.isdisjoint(rank_experts))
                        seen_experts.update(rank_experts)
                        for e in rr.entries:
                            rr.read_blob(e, verify_crc=True)

                    r_meta = manifest["ranks"][r_idx]
                    expected_src = "local_wired_ram" if r_idx == 0 else "thunderbolt_peer"
                    self.assertEqual(r_meta["expert_source_on_rank0"], expected_src)

                self.assertEqual(seen_experts, set(range(8)))


class TestDeepSeek41ConverterAndCli(unittest.TestCase):
    def test_convert_multi_shard_checkpoint_q3_q4_bf16_and_gsq_rco(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ckpt = _create_multi_shard_deepseek41_checkpoint(root / "ckpt")

            # 1. q3_affine + 2 Engram shards + 2-rank expert partition
            out_q3 = root / "out_q3" / "model.amf"
            rep_q3 = convert_deepseek41_checkpoint(
                ckpt,
                out_q3,
                model_id="ds41-test-q3",
                target_dtype="q3_affine",
                engram_shards=2,
                world_size=2,
            )
            self.assertEqual(rep_q3.stripped_vision_tensors, 1)
            self.assertEqual(len(rep_q3.engram_paths), 2)
            self.assertIsNotNone(rep_q3.dspark_amf)
            self.assertIsNotNone(rep_q3.partition_report)
            with AmfReader(out_q3, verify_header_crc=True, verify_dir_crc=True) as r:
                self.assertEqual(r.by_name["layers.0.mlp.experts.0.gate_proj"].dtype, DType.Q3_AFFINE)
                self.assertEqual(r.by_name["layers.0.mlp.shared_expert.gate_proj"].kind, TensorKind.SHARED_EXPERT)
                self.assertNotIn("visual.patch_embed.weight", r.by_name)
                for e in r.entries:
                    r.read_blob(e, verify_crc=True)
            for ep in rep_q3.engram_paths:
                with AmfReader(ep, verify_header_crc=True, verify_dir_crc=True) as er:
                    self.assertEqual(len(er.entries), 1)
                    self.assertEqual(er.entries[0].kind, TensorKind.SPARSE_TABLE)
                    self.assertEqual(er.entries[0].shape, (16, 128))
                    er.read_blob(er.entries[0], verify_crc=True)

            # 2. q4_affine
            out_q4 = root / "out_q4" / "model.amf"
            convert_deepseek41_checkpoint(ckpt, out_q4, target_dtype="q4_affine")
            with AmfReader(out_q4, verify_header_crc=True, verify_dir_crc=True) as r:
                self.assertEqual(r.by_name["layers.0.mlp.experts.1.up_proj"].dtype, DType.Q4_AFFINE)

            # 3. bf16
            out_bf16 = root / "out_bf16" / "model.amf"
            convert_deepseek41_checkpoint(ckpt, out_bf16, target_dtype="bf16")
            with AmfReader(out_bf16, verify_header_crc=True, verify_dir_crc=True) as r:
                self.assertEqual(r.by_name["layers.0.mlp.experts.1.up_proj.weight"].dtype, DType.BF16)

            # 4. --quant-profile gsq-rco
            out_gsq = root / "out_gsq" / "model.amf"
            manifest_gsq = root / "out_gsq" / "gsq_manifest.json"
            convert_deepseek41_checkpoint(
                ckpt,
                out_gsq,
                quant_profile="gsq-rco",
                target_bits=3.5,
                manifest_path=manifest_gsq,
            )
            self.assertTrue(manifest_gsq.is_file())
            with AmfReader(out_gsq, verify_header_crc=True, verify_dir_crc=True) as r:
                quant_dtypes = {
                    e.dtype for e in r.entries if e.kind in (TensorKind.HOT_EXPERT, TensorKind.DENSE)
                }
                self.assertTrue(
                    bool(quant_dtypes & {DType.Q2_AFFINE, DType.Q3_AFFINE, DType.Q4_AFFINE}),
                    f"expected GSQ quantized dtypes, got {quant_dtypes}",
                )

    def test_convert_rejects_full_552b_on_24gb_m4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ckpt_552b = _create_multi_shard_deepseek41_checkpoint(root / "ckpt_552b", full_552b_config_only=True)
            with self.assertRaises(HardwareTierError):
                convert_deepseek41_checkpoint(
                    ckpt_552b,
                    root / "out" / "model.amf",
                    target_dtype="q3_affine",
                    target_tier="24gb-m4",
                    world_size=1,
                )

            # Same checkpoint config succeeds when targeting 512 GB Ultra!
            rep = convert_deepseek41_checkpoint(
                ckpt_552b,
                root / "out_ultra" / "model.amf",
                target_dtype="q3_affine",
                target_tier="512gb-ultra",
                world_size=1,
            )
            self.assertTrue(rep.tier_check.fits)
            self.assertEqual(rep.tier_check.target_tier, "512gb-ultra")

    def test_cli_subcommands_synth_partition_and_pack_deepseek41(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)

            # synth-deepseek41 with --world-size 2
            synth_dir = root / "cli_synth"
            rc = cli_main(
                [
                    "synth-deepseek41",
                    "--preset",
                    "tiny-deepseek41",
                    "--output-dir",
                    str(synth_dir),
                    "--world-size",
                    "2",
                    "--seed",
                    "77",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((synth_dir / "model.amf").is_file())
            self.assertTrue((synth_dir / "shards" / "cluster_manifest.json").is_file())

            # partition-experts
            part_dir = root / "cli_part"
            rc = cli_main(
                [
                    "partition-experts",
                    "--input",
                    str(synth_dir / "model.amf"),
                    "--world-size",
                    "4",
                    "--strategy",
                    "round_robin",
                    "--output-dir",
                    str(part_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((part_dir / "cluster_manifest.json").is_file())
            self.assertTrue((part_dir / "rank-03-of-04.amf").is_file())

            # pack-deepseek41 (valid local checkpoint)
            ckpt = _create_multi_shard_deepseek41_checkpoint(root / "cli_ckpt")
            packed = root / "cli_packed" / "model.amf"
            rc = cli_main(
                [
                    "pack-deepseek41",
                    "--input",
                    str(ckpt),
                    "--output",
                    str(packed),
                    "--dtype",
                    "q4_affine",
                    "--engram-shards",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(packed.is_file())

            # pack-deepseek41 rejects full 552B on 24gb-m4 with exit code 2
            ckpt_552b = _create_multi_shard_deepseek41_checkpoint(root / "cli_552b", full_552b_config_only=True)
            rc = cli_main(
                [
                    "pack-deepseek41",
                    "--input",
                    str(ckpt_552b),
                    "--output",
                    str(root / "fail" / "model.amf"),
                    "--target-tier",
                    "24gb-m4",
                ]
            )
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
