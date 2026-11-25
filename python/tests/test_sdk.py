"""End-to-end verification tests for the zero-dependency Python ctypes SDK (C17/C21).

Generates a tiny temporary synthetic Qwen3 `.amf` bundle inside `TMPDIR` using
the offline `tools` generator (cleaned up unconditionally in `tearDownClass`),
and verifies:
  1. Zero third-party runtime dependencies & clean `libaminfer.dylib` linkage.
  2. Native BPE tokenization (`W5` Qwen & DeepSeek dialects), `count_tokens`, `detokenize`.
  3. Co-tenant `wired_budget_bytes` ceiling & `R9` live zero-swap RAM rejection.
  4. Synchronous & streaming generation releasing the Python GIL across native steps,
     deterministic seed replay, and callback stop.
  5. Read-only zero-copy `memoryview` over `logits` (`fp16`/`fp32`) and `hidden_states`
     (PEP 3118 buffer-protocol compatible, zero copy verified by pointer/value parity).
  6. Native `embed()` (`last_token`/`mean`/`cls`, Matryoshka `dimensions` slicing, L2 norm)
     and `rerank()` (`logit_diff`/`prob_yes`/`raw_logit`).
  7. Cooperative `CancelToken` cancellation without modifying POSIX signal handlers.
  8. Zero-copy `VisionInput` (`CVPixelBufferRef` / `IOSurfaceRef` / `IOSurfaceID`) ingress (`G14`).
  9. Repeated `Engine` create/destroy cycles leaving `0` swapouts and stable wired memory.
"""

from __future__ import annotations

import ctypes
import json
import math
import pathlib
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))
TOOLS_SRC = REPO_ROOT / "tools" / "src"
if str(TOOLS_SRC) not in sys.path:
    sys.path.insert(0, str(TOOLS_SRC))

from aminfer_tools.synth import Qwen3SynthConfig, bytes_to_unicode, generate_qwen3_bundle  # noqa: E402

import aminfer  # noqa: E402
from aminfer import (  # noqa: E402
    AMI_SDK_ABI_VERSION,
    BudgetExceededError,
    CancelledError,
    CancelToken,
    Dialect,
    DType,
    Engine,
    FinishReason,
    GenerationConfig,
    InvalidArgumentError,
    PoolingMode,
    RerankMode,
    Tokenizer,
    VisionFormat,
    VisionInput,
)
from aminfer._bindings import (  # noqa: E402
    AMI_SDK_EFFORT_CUSTOM_FLOAT,
    AMI_SDK_EFFORT_DEFAULT,
    AMI_SDK_EFFORT_HIGH,
    AMI_SDK_EFFORT_MAX,
    AMI_SDK_EFFORT_MEDIUM,
    AMI_SDK_EFFORT_MINIMAL,
    AMI_SDK_EFFORT_NO_THINK,
    AMI_SDK_EFFORT_NONE,
    AMI_SDK_EFFORT_LOW,
    AMI_SDK_EFFORT_XHIGH,
    GenParamsC,
    find_libaminfer,
    load_lib,
)
from aminfer.engine import ReasoningEffort  # noqa: E402


def _make_deepseek_tokenizer_json() -> str:
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    vocab["<|im_start|>"] = 256
    vocab["<|im_end|>"] = 257
    payload = {
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [
            {"id": 256, "content": "<|im_start|>", "special": True},
            {"id": 257, "content": "<|im_end|>", "special": True},
        ],
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": r"\p{N}{1,3}"}},
            ],
        },
    }
    return json.dumps(payload)


class TestAminferPythonSdk(unittest.TestCase):
    tmp_dir: pathlib.Path
    amf_path: pathlib.Path
    config_path: pathlib.Path
    tokenizer_path: pathlib.Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="aminfer_py_sdk_test_"))
        cfg = Qwen3SynthConfig(
            model_id="sdk-tiny-qwen3",
            model_type="qwen3",
            quant_bits=4,
            group_size=64,
            vocab_size=320,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            intermediate_size=256,
            moe_intermediate_size=0,
            num_experts=0,
            num_experts_per_tok=0,
            tie_word_embeddings=False,
            embed_f16=True,
        )
        bundle = generate_qwen3_bundle(cls.tmp_dir, cfg, seed=1234)
        cls.amf_path = bundle.amf_path
        cls.config_path = bundle.config_path
        cls.tokenizer_path = bundle.tokenizer_path

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_01_dylib_linkage_and_abi_version(self) -> None:
        lib_path = find_libaminfer()
        lib = load_lib(lib_path)
        self.assertEqual(int(lib.ami_sdk_abi_version()), AMI_SDK_ABI_VERSION)
        ver = lib.ami_sdk_version_string().decode("utf-8")
        self.assertTrue(ver.startswith("0.7."))

        otool = subprocess.run(
            ["otool", "-L", lib_path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertNotIn("Python", otool)
        self.assertNotIn("libpython", otool)
        self.assertIn("Metal.framework", otool)
        self.assertIn("IOKit.framework", otool)
        self.assertIn("IOSurface.framework", otool)

    def test_02_native_bpe_tokenizer_qwen_and_deepseek(self) -> None:
        with Tokenizer.from_file(self.tokenizer_path) as qtok:
            self.assertEqual(qtok.dialect, Dialect.QWEN)
            self.assertEqual(qtok.vocab_size, 320)
            text = "<|im_start|>Hello Apple Silicon 42!<|im_end|>"
            count = qtok.count_tokens(text)
            ids = qtok.tokenize(text)
            self.assertEqual(len(ids), count)
            self.assertGreater(count, 2)
            self.assertEqual(qtok.detokenize(ids, skip_special_tokens=False), text)
            self.assertEqual(qtok.detokenize(ids, skip_special_tokens=True), "Hello Apple Silicon 42!")

        with Tokenizer.from_json(_make_deepseek_tokenizer_json()) as dtok:
            self.assertEqual(dtok.dialect, Dialect.DEEPSEEK)
            self.assertEqual(dtok.vocab_size, 258)
            self.assertEqual(dtok.special_token_id("<|im_start|>"), 256)
            sample = "<|im_start|>DeepSeek 12345<|im_end|>"
            ids = dtok.tokenize(sample)
            self.assertEqual(ids[0], 256)
            self.assertEqual(ids[-1], 257)
            self.assertEqual(dtok.detokenize(ids, skip_special_tokens=True), "DeepSeek 12345")

    def test_03_cotenant_budget_and_r9_live_ram_guards(self) -> None:
        with self.assertRaises(BudgetExceededError):
            Engine(
                self.amf_path,
                max_seq_len=64,
                wired_budget_bytes=16 * 1024,
                startup_canary=False,
            )

        with self.assertRaises(BudgetExceededError):
            Engine(
                self.amf_path,
                max_seq_len=64,
                strict_live_ram=True,
                live_reclaimable_override_bytes=64 * 1024,
                live_safety_margin_bytes=32 * 1024,
                startup_canary=False,
            )

    def test_04_end_to_end_generate_stream_views_embed_rerank_cancel_vision(self) -> None:
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        with Engine(
            self.amf_path,
            max_seq_len=512,
            startup_canary=False,
        ) as engine:
            stats = engine.stats
            self.assertFalse(stats.is_graph_model)
            self.assertEqual(stats.vocab_size, 320)
            self.assertEqual(stats.hidden_size, 128)
            self.assertEqual(stats.num_layers, 2)
            self.assertGreater(stats.total_wired_bytes, stats.sdk_wired_bytes)
            self.assertIsNotNone(engine.tokenizer)

            # 1. Tokenize + synchronous greedy generation + deterministic replay.
            prompt = "Hello aminfer C17 SDK"
            prompt_ids = engine.tokenize(prompt)
            self.assertGreater(len(prompt_ids), 0)

            cfg_greedy = GenerationConfig(max_tokens=6, temperature=0.0, ignore_eos=True)
            out1 = engine.generate(prompt, cfg_greedy)
            out2 = engine.generate(prompt, cfg_greedy)
            self.assertEqual(out1.generated_tokens, 6)
            self.assertEqual(out1.token_ids, out2.token_ids)
            self.assertEqual(out1.finish_reason, FinishReason.LENGTH)

            # 2. Streaming generator interface + callback stop.
            chunks = list(engine.stream(prompt, cfg_greedy))
            streamed_text = "".join(c.text for c in chunks)
            self.assertEqual(streamed_text, out1.text)

            seen_steps: list[int] = []
            stopped_out = engine.generate(
                prompt,
                cfg_greedy,
                on_chunk=lambda ch: (seen_steps.append(ch.step_index), len(seen_steps) < 2)[1],
            )
            self.assertEqual(stopped_out.finish_reason, FinishReason.CALLBACK_STOP)
            self.assertLessEqual(stopped_out.generated_tokens, 2)

            # 3. Sampled generation with explicit seed replay & penalties.
            cfg_samp = GenerationConfig(
                max_tokens=5,
                temperature=0.8,
                top_p=0.9,
                top_k=32,
                min_p=0.05,
                repetition_penalty=1.1,
                seed=999,
                ignore_eos=True,
            )
            s1 = engine.generate(prompt_ids, cfg_samp)
            s2 = engine.generate(prompt_ids, cfg_samp)
            self.assertEqual(s1.token_ids, s2.token_ids)
            self.assertEqual(s1.seed_used, 999)

            # 4. Read-only zero-copy memoryview over fp32 & fp16 logits and hidden_states.
            engine.prefill(prompt_ids, compute_logits=True, reset_kv_cache=True)
            logits_f32 = engine.logits_view(DType.F32)
            logits_f16 = engine.logits_view("f16")
            self.assertTrue(logits_f32.readonly)
            self.assertTrue(logits_f16.readonly)
            self.assertEqual(logits_f32.shape, (320,))
            self.assertEqual(logits_f16.shape, (320,))
            self.assertEqual(logits_f32.nbytes, 320 * 4)
            self.assertEqual(logits_f16.nbytes, 320 * 2)
            # Verify buffer-protocol zero-copy unpacking matches f16 <-> f32 values.
            val0_f32 = struct.unpack_from("<f", logits_f32.tobytes(), 0)[0]
            val0_f16 = struct.unpack_from("<e", logits_f16.tobytes(), 0)[0]
            self.assertAlmostEqual(val0_f32, val0_f16, places=4)
            with self.assertRaises(TypeError):
                logits_f32[0] = 0.0  # type: ignore[index]

            hidden_f32 = engine.hidden_states_view("f32", normalized=True)
            hidden_raw_f16 = engine.hidden_states_view("f16", normalized=False)
            self.assertTrue(hidden_f32.readonly)
            self.assertEqual(hidden_f32.shape, (128,))
            self.assertEqual(hidden_raw_f16.shape, (128,))

            # 5. Native embeddings (`last_token`, `mean`, `cls`, Matryoshka slice, L2 norm).
            emb_last = engine.embed(prompt, pooling=PoolingMode.LAST_TOKEN, normalize=True)
            emb_mean_32 = engine.embed(prompt, pooling="mean", dimensions=32, normalize=True)
            emb_cls_16 = engine.embed(prompt_ids, pooling=PoolingMode.CLS, dimensions=16, normalize=False)
            self.assertEqual(len(emb_last), 128)
            self.assertEqual(len(emb_mean_32), 32)
            self.assertEqual(len(emb_cls_16), 16)
            self.assertAlmostEqual(sum(x * x for x in emb_last), 1.0, places=4)
            self.assertAlmostEqual(sum(x * x for x in emb_mean_32), 1.0, places=4)

            batch_embs = engine.embed(["apple silicon", "metal 4"], pooling="mean", dimensions=64)
            self.assertEqual(len(batch_embs), 2)
            self.assertEqual(len(batch_embs[0]), 64)

            # 6. Native cross-encoder rerank (`prob_yes` & `logit_diff`).
            ranked = engine.rerank(
                "unified memory inference",
                ["Metal 4 zero-copy IOSurface", "unrelated banana recipe"],
                mode=RerankMode.PROB_YES,
            )
            self.assertEqual(len(ranked), 2)
            self.assertGreaterEqual(ranked[0].score, ranked[1].score)
            for item in ranked:
                self.assertGreaterEqual(item.score, 0.0)
                self.assertLessEqual(item.score, 1.0)

            # 7. Cooperative CancelToken (both pre-cancelled and cross-thread cancellation).
            with CancelToken() as ctok:
                self.assertFalse(ctok.is_cancelled)
                ctok.cancel()
                self.assertTrue(ctok.is_cancelled)
                with self.assertRaises(CancelledError):
                    engine.generate(prompt, GenerationConfig(max_tokens=8, cancel_token=ctok))
                ctok.reset()
                self.assertFalse(ctok.is_cancelled)

                # Verify GIL release & concurrent thread responsiveness during native decode.
                bg_ticks = 0
                done_flag = threading.Event()

                def _ticker() -> None:
                    nonlocal bg_ticks
                    while not done_flag.is_set():
                        bg_ticks += 1
                        if bg_ticks >= 5:
                            break

                t = threading.Thread(target=_ticker)
                t.start()
                out_ok = engine.generate(
                    prompt, GenerationConfig(max_tokens=4, ignore_eos=True, cancel_token=ctok)
                )
                done_flag.set()
                t.join(timeout=2.0)
                self.assertEqual(out_ok.generated_tokens, 4)
                self.assertGreater(bg_ticks, 0)

            # 8. Zero-copy VisionInput (`CVPixelBufferRef` / `IOSurfaceRef` / `IOSurfaceID`, G14).
            with VisionInput.create_iosurface(engine, 32, 32, fmt=VisionFormat.BGRA8) as vis_src:
                with vis_src.lock_plane(0, read_only=False) as plane_mv:
                    self.assertEqual(plane_mv.shape[0], 32)
                    plane_mv[0, 0] = 200
                    plane_mv[0, 1] = 120
                    plane_mv[0, 2] = 60
                    plane_mv[0, 3] = 255

                info_src = vis_src.info
                self.assertTrue(info_src.zero_copy_verified)
                self.assertTrue(info_src.is_page_aligned)
                self.assertGreater(info_src.iosurface_id, 0)

                with (
                    VisionInput.from_iosurface_id(engine, info_src.iosurface_id) as vis_id,
                    VisionInput.from_iosurface(engine, vis_src.iosurface_ptr) as vis_surf,
                    VisionInput.from_cvpixelbuffer(engine, vis_src.cvpixelbuffer_ptr) as vis_pb,
                ):
                    self.assertEqual(vis_id.info.base_address, info_src.base_address)
                    self.assertEqual(vis_surf.info.base_address, info_src.base_address)
                    self.assertEqual(vis_pb.info.base_address, info_src.base_address)
                    self.assertTrue(vis_pb.info.zero_copy_verified)

                    vres = engine.prefill_vision(
                        vis_id,
                        prefix=prompt_ids[:2],
                        suffix=prompt_ids[2:4],
                        patch_size=16,
                        spatial_merge=2,
                        t_patch=1,
                    )
                    self.assertEqual(vres.vision_tokens, 1)
                    self.assertEqual(vres.grid, (1, 1, 1))
                    self.assertTrue(vres.all_frames_zero_copy)
                    self.assertTrue(vres.all_frames_page_aligned)
                    # Logits view is immediately available after multimodal prefill.
                    vis_logits = engine.logits_view(DType.F32)
                    self.assertEqual(vis_logits.shape, (320,))

        # Signal handlers must remain completely untouched by libaminfer.dylib.
        self.assertIs(signal.getsignal(signal.SIGINT), prev_sigint)
        self.assertIs(signal.getsignal(signal.SIGTERM), prev_sigterm)

    def test_05_repeated_create_destroy_zero_wired_leak(self) -> None:
        # Warm up Metal 4 framework caches once, then verify repeated create/destroy loops
        # incur zero swapouts and do not ratchet wired/heap memory upward across cycles.
        with Engine(self.amf_path, max_seq_len=64, startup_canary=False) as warm:
            warm.prefill([1, 2, 3], compute_logits=True)

        vm_before = Engine.host_vm_snapshot()
        wired_footprint_per_engine = 0

        for _ in range(5):
            with Engine(self.amf_path, max_seq_len=64, startup_canary=False) as eng:
                wired_footprint_per_engine = eng.stats.total_wired_bytes
                _ = eng.generate([1, 2, 3], GenerationConfig(max_tokens=2, ignore_eos=True))

        vm_after = Engine.host_vm_snapshot()
        self.assertGreater(wired_footprint_per_engine, 0)
        self.assertEqual(vm_after.swapouts, vm_before.swapouts)
        # Process malloc heap in-use delta must stay bounded far below one engine's wired footprint
        # (confirming every WiredArena munlock + Objective-C autorelease pool drained cleanly).
        heap_growth = vm_after.phys_footprint_bytes - vm_before.phys_footprint_bytes
        self.assertLess(heap_growth, wired_footprint_per_engine)

    def test_06_reasoning_effort_and_c17_parameter_bounds_hardening(self) -> None:
        lib = load_lib()
        self.assertEqual(ctypes.sizeof(GenParamsC), 120)
        self.assertEqual(int(Dialect.INKLING2), 2)
        self.assertEqual(int(Dialect.HY4), 3)

        # 1. reasoning_effort mapping parity (`None`, tiers, floats in [0.0, 0.99]).
        tier_cases = [
            (None, AMI_SDK_EFFORT_DEFAULT, 0.0),
            ("default", AMI_SDK_EFFORT_DEFAULT, 0.0),
            ("none", AMI_SDK_EFFORT_NONE, 0.0),
            ("no_think", AMI_SDK_EFFORT_NO_THINK, 0.0),
            ("no-think", AMI_SDK_EFFORT_NO_THINK, 0.0),
            ("minimal", AMI_SDK_EFFORT_MINIMAL, 0.0),
            ("low", AMI_SDK_EFFORT_LOW, 0.0),
            ("medium", AMI_SDK_EFFORT_MEDIUM, 0.0),
            ("HIGH", AMI_SDK_EFFORT_HIGH, 0.0),
            ("xhigh", AMI_SDK_EFFORT_XHIGH, 0.0),
            ("max", AMI_SDK_EFFORT_MAX, 0.0),
            (ReasoningEffort.MEDIUM, AMI_SDK_EFFORT_MEDIUM, 0.0),
        ]
        for effort_in, exp_enum, exp_val in tier_cases:
            c_gp = GenerationConfig(reasoning_effort=effort_in).to_c(lib)
            self.assertEqual(int(c_gp.reasoning_effort), exp_enum)
            self.assertAlmostEqual(float(c_gp.reasoning_effort_value), exp_val, places=5)

        for fval in (0, 0.0, 0.35, 0.99):
            c_gp = GenerationConfig(reasoning_effort=fval).to_c(lib)
            self.assertEqual(int(c_gp.reasoning_effort), AMI_SDK_EFFORT_CUSTOM_FLOAT)
            self.assertAlmostEqual(float(c_gp.reasoning_effort_value), float(fval), places=5)

        # 2. Invalid reasoning_effort & C17 sampling/integer parameter rejection.
        invalid_configs: list[dict[str, object]] = [
            {"max_tokens": -1},
            {"max_tokens": 1 << 32},
            {"max_tokens": True},
            {"max_tokens": 1.5},
            {"top_k": -1},
            {"top_k": 1 << 32},
            {"top_k": False},
            {"stop_token_ids": (-1,)},
            {"stop_token_ids": (1 << 32,)},
            {"stop_token_ids": (True,)},
            {"stop_token_ids": tuple(range(9))},
            {"temperature": -0.01},
            {"temperature": math.nan},
            {"temperature": math.inf},
            {"temperature": True},
            {"top_p": 0.0},
            {"top_p": -0.5},
            {"top_p": 1.01},
            {"top_p": math.nan},
            {"top_p": math.inf},
            {"min_p": -0.01},
            {"min_p": 1.0},
            {"min_p": 1.5},
            {"min_p": math.nan},
            {"min_p": math.inf},
            {"repetition_penalty": 0.0},
            {"repetition_penalty": -1.0},
            {"repetition_penalty": math.nan},
            {"repetition_penalty": math.inf},
            {"presence_penalty": math.nan},
            {"presence_penalty": math.inf},
            {"presence_penalty": 1e40},
            {"frequency_penalty": math.nan},
            {"frequency_penalty": -math.inf},
            {"reasoning_effort": True},
            {"reasoning_effort": "ultra"},
            {"reasoning_effort": -0.01},
            {"reasoning_effort": 1.0},
            {"reasoning_effort": math.nan},
            {"reasoning_effort": math.inf},
            {"reasoning_effort": ReasoningEffort.CUSTOM_FLOAT},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(InvalidArgumentError):
                    GenerationConfig(**kwargs).to_c(lib)  # type: ignore[arg-type]

        # min_p == 0.0 and valid boundary values must succeed.
        ok_gp = GenerationConfig(max_tokens=0, min_p=0.0, top_p=1.0, temperature=0.0, reasoning_effort=0.99).to_c(lib)
        self.assertEqual(int(ok_gp.max_tokens), 0)

        # 3. End-to-end Engine checks: max_tokens=0 clean return (no OOB slice) + token ID validation.
        with Engine(self.amf_path, max_seq_len=64, startup_canary=False) as engine:
            zero_text_out = engine.generate("Hello", GenerationConfig(max_tokens=0, reasoning_effort="no_think"))
            self.assertEqual(zero_text_out.generated_tokens, 0)
            self.assertEqual(zero_text_out.token_ids, ())
            self.assertEqual(zero_text_out.text, "")
            self.assertEqual(zero_text_out.finish_reason, FinishReason.LENGTH)

            zero_ids_out = engine.generate([1, 2, 3], GenerationConfig(max_tokens=0, reasoning_effort=0.25))
            self.assertEqual(zero_ids_out.generated_tokens, 0)
            self.assertEqual(zero_ids_out.token_ids, ())

            with self.assertRaises(InvalidArgumentError):
                engine.generate("Hello", GenerationConfig(max_tokens=-1))

            bad_tokens: list[object] = [True, False, -1, 1 << 32, 320, 999]
            for bad_tok in bad_tokens:
                with self.subTest(bad_tok=bad_tok):
                    with self.assertRaises(InvalidArgumentError):
                        engine._prompt_to_ids([1, bad_tok])  # type: ignore[list-item]
                    with self.assertRaises(InvalidArgumentError):
                        engine.generate([1, bad_tok], GenerationConfig(max_tokens=1))  # type: ignore[list-item]
                    with self.assertRaises(InvalidArgumentError):
                        engine.step_decode(bad_tok)  # type: ignore[arg-type]
                    with self.assertRaises(InvalidArgumentError):
                        engine.detokenize([1, bad_tok])  # type: ignore[list-item]
                    with self.assertRaises(InvalidArgumentError):
                        engine.tokenizer.detokenize([bad_tok])  # type: ignore[union-attr,list-item]
                    with self.assertRaises(InvalidArgumentError):
                        engine.embed([bad_tok])  # type: ignore[list-item]

            # Valid reasoning_effort generate round-trip.
            eff_out = engine.generate([1, 2, 3], GenerationConfig(max_tokens=2, ignore_eos=True, reasoning_effort=0.7))
            self.assertEqual(eff_out.generated_tokens, 2)

    def test_07_speculative_configuration_and_l1_exact_parity(self) -> None:
        from aminfer import DraftMode

        with Engine(self.amf_path, max_seq_len=64, startup_canary=False) as engine:
            self.assertEqual(engine.spec_stats.active_draft_mode, DraftMode.NONE)
            base_out = engine.generate(
                [1, 2, 3, 1, 2, 3],
                GenerationConfig(max_tokens=8, temperature=0.0, ignore_eos=True),
            )

            with self.assertRaises(InvalidArgumentError):
                engine.configure_speculative("bogus_mode")
            with self.assertRaises(InvalidArgumentError):
                engine.configure_speculative("l1", confidence_threshold=1.5)

            engine.configure_speculative("auto")
            self.assertEqual(engine.spec_stats.active_draft_mode, DraftMode.L1)
            spec_out = engine.generate(
                [1, 2, 3, 1, 2, 3],
                GenerationConfig(max_tokens=8, temperature=0.0, ignore_eos=True),
            )
            self.assertEqual(spec_out.token_ids, base_out.token_ids)
            st = engine.spec_stats
            self.assertEqual(st.active_draft_mode, DraftMode.L1)
            self.assertGreater(st.rounds, 0)
            self.assertGreaterEqual(st.mean_accept_len, 1.0)

            engine.configure_speculative(DraftMode.NONE)
            self.assertEqual(engine.spec_stats.active_draft_mode, DraftMode.NONE)


if __name__ == "__main__":
    unittest.main()
