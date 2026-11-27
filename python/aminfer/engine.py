"""Pythonic zero-dependency ctypes SDK for aminfer (ADR-0022 C17/C21).

Provides Engine, Tokenizer, CancelToken, GenerationConfig, streaming iterators
and callbacks, read-only zero-copy memoryview access to logits and hidden_states
(PEP 3118 buffer-protocol compatible with NumPy np.frombuffer / np.asarray),
native embeddings (lm_head elided) & cross-encoder reranking, and zero-copy
CVPixelBufferRef / IOSurfaceRef / IOSurfaceID vision ingress (G14).
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import enum
import math
import os
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any, overload

from aminfer._bindings import (
    AMI_SDK_ABI_VERSION,
    AMI_SDK_DIALECT_DEEPSEEK,
    AMI_SDK_DIALECT_HY4,
    AMI_SDK_DIALECT_INKLING2,
    AMI_SDK_DIALECT_QWEN,
    AMI_SDK_DRAFT_AUTO,
    AMI_SDK_DRAFT_DFLARE,
    AMI_SDK_DRAFT_DFLY2,
    AMI_SDK_DRAFT_L1,
    AMI_SDK_DRAFT_NONE,
    AMI_SDK_EFFORT_CUSTOM_FLOAT,
    AMI_SDK_EFFORT_DEFAULT,
    AMI_SDK_EFFORT_HIGH,
    AMI_SDK_EFFORT_LOW,
    AMI_SDK_EFFORT_MAX,
    AMI_SDK_EFFORT_MEDIUM,
    AMI_SDK_EFFORT_MINIMAL,
    AMI_SDK_EFFORT_NONE,
    AMI_SDK_EFFORT_NO_THINK,
    AMI_SDK_EFFORT_XHIGH,
    AMI_SDK_ERR_BUFFER_TOO_SMALL,
    AMI_SDK_MAX_STOP_TOKENS,
    AMI_SDK_OK,
    AMI_SDK_TOKEN_NONE,
    STREAM_CB_TYPE,
    EmbedParamsC,
    EngineOptionsC,
    EngineStatsC,
    GenParamsC,
    GenResultC,
    HostVmInfoC,
    InvalidArgumentError,
    LoraInfoC,
    RerankParamsC,
    SpecOptionsC,
    SpecStatsC,
    StreamEventC,
    TensorViewC,
    VisionInfoC,
    VisionParamsC,
    VisionResultC,
    check_status,
    load_lib,
)

# CPython read-only memoryview helper (falls back to ctypes buffer view on non-CPython).
_PYBUF_READ = 0x100
_PYBUF_WRITE = 0x200
_HAS_PY_MEMVIEW = hasattr(ctypes, "pythonapi") and hasattr(
    ctypes.pythonapi, "PyMemoryView_FromMemory"
)
if _HAS_PY_MEMVIEW:
    _PyMemoryView_FromMemory = ctypes.pythonapi.PyMemoryView_FromMemory
    _PyMemoryView_FromMemory.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]
    _PyMemoryView_FromMemory.restype = ctypes.py_object


def _make_memoryview(addr: int, byte_len: int, *, read_only: bool, fmt: str, shape: tuple[int, ...]) -> memoryview:
    if _HAS_PY_MEMVIEW:
        flags = _PYBUF_READ if read_only else _PYBUF_WRITE
        raw_mv: memoryview = _PyMemoryView_FromMemory(
            ctypes.c_void_p(addr), ctypes.c_ssize_t(byte_len), ctypes.c_int(flags)
        )
        return raw_mv.cast(fmt, shape=shape)
    raw_arr = (ctypes.c_uint8 * byte_len).from_address(addr)
    return memoryview(raw_arr).cast(fmt, shape=shape)


class DType(enum.IntEnum):
    F16 = 0
    F32 = 1


class Dialect(enum.IntEnum):
    QWEN = AMI_SDK_DIALECT_QWEN
    DEEPSEEK = AMI_SDK_DIALECT_DEEPSEEK
    INKLING2 = AMI_SDK_DIALECT_INKLING2
    HY4 = AMI_SDK_DIALECT_HY4


class ReasoningEffort(enum.IntEnum):
    DEFAULT = AMI_SDK_EFFORT_DEFAULT
    NONE = AMI_SDK_EFFORT_NONE
    NO_THINK = AMI_SDK_EFFORT_NO_THINK
    MINIMAL = AMI_SDK_EFFORT_MINIMAL
    LOW = AMI_SDK_EFFORT_LOW
    MEDIUM = AMI_SDK_EFFORT_MEDIUM
    HIGH = AMI_SDK_EFFORT_HIGH
    XHIGH = AMI_SDK_EFFORT_XHIGH
    MAX = AMI_SDK_EFFORT_MAX
    CUSTOM_FLOAT = AMI_SDK_EFFORT_CUSTOM_FLOAT


class DeltaKind(enum.IntEnum):
    TEXT = 0
    REASONING = 1
    TOOL_CALL = 2
    TOOL_CALL_END = 3


class FinishReason(enum.IntEnum):
    LENGTH = 0
    STOP = 1
    TOOL_CALLS = 2
    CANCELLED = 3
    CALLBACK_STOP = 4


class PoolingMode(enum.IntEnum):
    LAST_TOKEN = 0
    MEAN = 1
    CLS = 2


class RerankMode(enum.IntEnum):
    LOGIT_DIFF = 0
    PROB_YES = 1
    RAW_LOGIT = 2


class VisionFormat(enum.IntEnum):
    BGRA8 = 1
    NV12 = 2
    F16 = 3


class DraftMode(enum.IntEnum):
    NONE = AMI_SDK_DRAFT_NONE
    AUTO = AMI_SDK_DRAFT_AUTO
    L1 = AMI_SDK_DRAFT_L1
    DFLARE = AMI_SDK_DRAFT_DFLARE
    DFLY2 = AMI_SDK_DRAFT_DFLY2


def _coerce_draft_mode(mode: DraftMode | str) -> DraftMode:
    if isinstance(mode, DraftMode):
        return mode
    if not isinstance(mode, str):
        raise _invalid_arg(f"unsupported draft_mode: {mode!r}")
    norm = mode.strip().lower()
    if norm in ("none", "off", "disabled"):
        return DraftMode.NONE
    if norm == "auto":
        return DraftMode.AUTO
    if norm in ("l1", "l1_only", "sam"):
        return DraftMode.L1
    if norm in ("dflare", "dflash"):
        return DraftMode.DFLARE
    if norm in ("dfly2", "dfly"):
        return DraftMode.DFLY2
    raise _invalid_arg(f"unsupported draft_mode: {mode!r}")


def _coerce_dtype(dtype: DType | str) -> DType:
    if isinstance(dtype, DType):
        return dtype
    norm = dtype.strip().lower()
    if norm in ("f16", "fp16", "float16", "half"):
        return DType.F16
    if norm in ("f32", "fp32", "float32", "float"):
        return DType.F32
    raise InvalidArgumentError(-1, "AMI_SDK_ERR_INVALID_ARGUMENT", f"unsupported dtype: {dtype!r}")


def _coerce_pooling(pooling: PoolingMode | str) -> PoolingMode:
    if isinstance(pooling, PoolingMode):
        return pooling
    norm = pooling.strip().lower()
    if norm in ("last_token", "last"):
        return PoolingMode.LAST_TOKEN
    if norm == "mean":
        return PoolingMode.MEAN
    if norm in ("cls", "first_token", "first"):
        return PoolingMode.CLS
    raise InvalidArgumentError(
        -1, "AMI_SDK_ERR_INVALID_ARGUMENT", f"unsupported pooling mode: {pooling!r}"
    )


def _coerce_rerank_mode(mode: RerankMode | str) -> RerankMode:
    if isinstance(mode, RerankMode):
        return mode
    norm = mode.strip().lower()
    if norm in ("logit_diff", "diff"):
        return RerankMode.LOGIT_DIFF
    if norm in ("prob_yes", "prob", "probability"):
        return RerankMode.PROB_YES
    if norm in ("raw_logit", "raw"):
        return RerankMode.RAW_LOGIT
    raise InvalidArgumentError(
        -1, "AMI_SDK_ERR_INVALID_ARGUMENT", f"unsupported rerank mode: {mode!r}"
    )


def _as_utf8(text: str | bytes) -> bytes:
    if isinstance(text, bytes):
        return text
    if isinstance(text, str):
        return text.encode("utf-8")
    raise TypeError(f"expected str or bytes, got {type(text).__name__}")


@dataclasses.dataclass(frozen=True, slots=True)
class HostVmSnapshot:
    wired_bytes: int
    free_bytes: int
    live_reclaimable_bytes: int
    swapouts: int
    pageouts: int
    phys_footprint_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class EngineStats:
    is_graph_model: bool
    vocab_size: int
    hidden_size: int
    num_layers: int
    max_seq_len: int
    seq_len: int
    total_wired_bytes: int
    sdk_wired_bytes: int
    wired_budget_bytes: int
    live_reclaimable_bytes: int
    live_budget_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class SpecStats:
    active_draft_mode: DraftMode
    rounds: int
    emitted_tokens: int
    drafted_nodes: int
    dflare_rounds: int
    dflare_core3_batches: int
    dfly2_rounds: int
    dfly2_early_stops: int
    mean_accept_len: float


@dataclasses.dataclass(frozen=True, slots=True)
class LoraInfo:
    active_adapters: int
    active_slot: int | None
    max_adapters: int
    pool_capacity_bytes: int
    pool_used_bytes: int


_LORA_UNSET: object = object()


_EFFORT_NAME_MAP: dict[str, int] = {
    "default": AMI_SDK_EFFORT_DEFAULT,
    "none": AMI_SDK_EFFORT_NONE,
    "no_think": AMI_SDK_EFFORT_NO_THINK,
    "nothink": AMI_SDK_EFFORT_NO_THINK,
    "minimal": AMI_SDK_EFFORT_MINIMAL,
    "low": AMI_SDK_EFFORT_LOW,
    "medium": AMI_SDK_EFFORT_MEDIUM,
    "high": AMI_SDK_EFFORT_HIGH,
    "xhigh": AMI_SDK_EFFORT_XHIGH,
    "max": AMI_SDK_EFFORT_MAX,
}


def _invalid_arg(msg: str) -> InvalidArgumentError:
    return InvalidArgumentError(-1, "AMI_SDK_ERR_INVALID_ARGUMENT", msg)


def _validate_u32(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise _invalid_arg(f"{name} must be an integer in [0, 0xFFFFFFFF], got {val!r}")
    iv = int(val)
    if iv < 0 or iv > 0xFFFFFFFF:
        raise _invalid_arg(f"{name} out of u32 range [0, 0xFFFFFFFF]: {iv}")
    return iv


def _validate_u64(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise _invalid_arg(f"{name} must be an integer in [0, 0xFFFFFFFFFFFFFFFF], got {val!r}")
    iv = int(val)
    if iv < 0 or iv > 0xFFFFFFFFFFFFFFFF:
        raise _invalid_arg(f"{name} out of u64 range: {iv}")
    return iv


def _validate_finite_float(val: Any, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise _invalid_arg(f"{name} must be a finite real number, got {val!r}")
    fv = float(val)
    if not math.isfinite(fv) or not math.isfinite(float(ctypes.c_float(fv).value)):
        raise _invalid_arg(f"{name} must be finite, got {val!r}")
    return fv


def _coerce_reasoning_effort(val: ReasoningEffort | str | float | None) -> tuple[int, float]:
    if val is None:
        return (AMI_SDK_EFFORT_DEFAULT, 0.0)
    if isinstance(val, bool):
        raise _invalid_arg(f"reasoning_effort cannot be bool: {val!r}")
    if isinstance(val, ReasoningEffort):
        if val == ReasoningEffort.CUSTOM_FLOAT:
            raise _invalid_arg("ReasoningEffort.CUSTOM_FLOAT requires a float in [0.0, 0.99]")
        return (int(val), 0.0)
    if isinstance(val, str):
        norm = val.strip().lower().replace("-", "_")
        mapped = _EFFORT_NAME_MAP.get(norm)
        if mapped is None:
            raise _invalid_arg(f"unsupported reasoning_effort string: {val!r}")
        return (mapped, 0.0)
    if isinstance(val, (int, float)):
        fv = float(val)
        if not math.isfinite(fv) or fv < 0.0 or fv > 0.99:
            raise _invalid_arg(f"numeric reasoning_effort must be finite and in [0.0, 0.99], got {val!r}")
        return (AMI_SDK_EFFORT_CUSTOM_FLOAT, fv)
    raise _invalid_arg(f"unsupported reasoning_effort type: {type(val).__name__}")


def _validate_token_id(tid: Any, *, vocab_size: int | None = None, name: str = "token_id") -> int:
    iv = _validate_u32(tid, name)
    if vocab_size is not None and vocab_size > 0 and iv >= vocab_size:
        raise _invalid_arg(f"{name} ({iv}) >= vocab_size ({vocab_size})")
    return iv


def _validate_token_ids(
    token_ids: Sequence[int],
    *,
    vocab_size: int | None = None,
    name: str = "token_ids",
) -> list[int]:
    if isinstance(token_ids, (str, bytes, bytearray)):
        raise _invalid_arg(f"{name} must be a sequence of integer token IDs, got {type(token_ids).__name__}")
    return [_validate_token_id(x, vocab_size=vocab_size, name=name) for x in token_ids]


@dataclasses.dataclass(slots=True)
class GenerationConfig:
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] | Sequence[int] = ()
    ignore_eos: bool = False
    start_in_think: bool = False
    reset_kv_cache: bool = True
    skip_special_tokens: bool = True
    reasoning_effort: ReasoningEffort | str | float | None = None
    cancel_token: CancelToken | None = None

    def to_c(self, lib: ctypes.CDLL) -> GenParamsC:
        max_tokens_u32 = _validate_u32(self.max_tokens, "max_tokens")
        top_k_u32 = _validate_u32(self.top_k, "top_k")

        temperature = _validate_finite_float(self.temperature, "temperature")
        if temperature < 0.0:
            raise _invalid_arg(f"temperature must be >= 0.0, got {temperature}")

        top_p = _validate_finite_float(self.top_p, "top_p")
        if top_p <= 0.0 or top_p > 1.0:
            raise _invalid_arg(f"top_p must be in (0.0, 1.0], got {top_p}")

        min_p = _validate_finite_float(self.min_p, "min_p")
        if min_p < 0.0 or min_p >= 1.0:
            raise _invalid_arg(f"min_p must be in [0.0, 1.0), got {min_p}")

        repetition_penalty = _validate_finite_float(self.repetition_penalty, "repetition_penalty")
        if repetition_penalty <= 0.0:
            raise _invalid_arg(f"repetition_penalty must be > 0.0, got {repetition_penalty}")

        presence_penalty = _validate_finite_float(self.presence_penalty, "presence_penalty")
        frequency_penalty = _validate_finite_float(self.frequency_penalty, "frequency_penalty")

        effort_enum, effort_val = _coerce_reasoning_effort(self.reasoning_effort)

        if isinstance(self.stop_token_ids, (str, bytes, bytearray)):
            raise _invalid_arg("stop_token_ids must be a sequence of token IDs")
        stops = list(self.stop_token_ids)
        if len(stops) > AMI_SDK_MAX_STOP_TOKENS:
            raise _invalid_arg(f"at most {AMI_SDK_MAX_STOP_TOKENS} stop_token_ids are supported")
        validated_stops = [
            _validate_u32(sid, f"stop_token_ids[{i}]") for i, sid in enumerate(stops)
        ]

        params = GenParamsC()
        lib.ami_sdk_gen_params_init(ctypes.byref(params))
        params.max_tokens = max_tokens_u32
        params.temperature = temperature
        params.top_p = top_p
        params.top_k = top_k_u32
        params.min_p = min_p
        params.repetition_penalty = repetition_penalty
        params.presence_penalty = presence_penalty
        params.frequency_penalty = frequency_penalty
        if self.seed is not None:
            params.has_seed = 1
            params.seed = _validate_u64(self.seed, "seed")
        else:
            params.has_seed = 0
            params.seed = 0
        params.stop_token_count = len(validated_stops)
        for i, sid in enumerate(validated_stops):
            params.stop_token_ids[i] = sid
        params.ignore_eos = 1 if self.ignore_eos else 0
        params.start_in_think = 1 if self.start_in_think else 0
        params.reset_kv_cache = 1 if self.reset_kv_cache else 0
        params.skip_special_tokens = 1 if self.skip_special_tokens else 0
        params.reasoning_effort = effort_enum
        params.reasoning_effort_value = effort_val
        params.cancel_token = self.cancel_token._require_open() if self.cancel_token is not None else None
        return params


@dataclasses.dataclass(frozen=True, slots=True)
class StreamChunk:
    token_id: int | None
    step_index: int
    kind: DeltaKind
    text: str
    is_final_flush: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationOutput:
    text: str
    reasoning: str
    tool_calls: tuple[str, ...]
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: FinishReason
    seed_used: int
    prefill_ns: float
    decode_ns: float

    @property
    def tokens_per_second(self) -> float:
        if self.decode_ns <= 0.0 or self.generated_tokens == 0:
            return 0.0
        return float(self.generated_tokens) / (self.decode_ns * 1e-9)


@dataclasses.dataclass(frozen=True, slots=True)
class RerankItem:
    index: int
    score: float


@dataclasses.dataclass(frozen=True, slots=True)
class VisionFrameInfo:
    format: VisionFormat
    cv_pixel_format: int
    width: int
    height: int
    plane_count: int
    iosurface_id: int
    zero_copy_verified: bool
    is_page_aligned: bool
    base_address: int
    surface_alloc_size: int


@dataclasses.dataclass(frozen=True, slots=True)
class VisionPrefillResult:
    first_token_id: int
    prefix_tokens: int
    vision_tokens: int
    suffix_tokens: int
    total_prompt_tokens: int
    grid: tuple[int, int, int]
    all_frames_zero_copy: bool
    all_frames_page_aligned: bool
    vision_encode_us: float


class CancelToken:
    """Thread-safe atomic cooperative cancellation token (0 POSIX signal changes)."""

    def __init__(self, *, lib_path: str | os.PathLike[str] | None = None) -> None:
        self._lib = load_lib(lib_path)
        handle = ctypes.c_void_p()
        check_status(self._lib, self._lib.ami_sdk_cancel_token_create(ctypes.byref(handle)))
        self._handle: ctypes.c_void_p | None = handle

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._lib.ami_sdk_cancel_token_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> CancelToken:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("CancelToken is closed")
        return self._handle

    def cancel(self) -> None:
        self._lib.ami_sdk_cancel_token_cancel(self._require_open())

    def reset(self) -> None:
        self._lib.ami_sdk_cancel_token_reset(self._require_open())

    @property
    def is_cancelled(self) -> bool:
        return bool(self._lib.ami_sdk_cancel_token_is_cancelled(self._require_open()))


class Tokenizer:
    """Native Zig Byte-Level BPE tokenizer supporting Qwen and DeepSeek dialects (W5)."""

    def __init__(
        self,
        _lib: ctypes.CDLL,
        _handle: ctypes.c_void_p,
        *,
        _owned: bool,
        _owner: Any = None,
    ) -> None:
        self._lib = _lib
        self._handle: ctypes.c_void_p | None = _handle
        self._owned = _owned
        self._owner = _owner

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        lib_path: str | os.PathLike[str] | None = None,
    ) -> Tokenizer:
        lib = load_lib(lib_path)
        handle = ctypes.c_void_p()
        raw_path = os.fsencode(os.fspath(path))
        check_status(lib, lib.ami_sdk_tokenizer_from_file(raw_path, ctypes.byref(handle)))
        return cls(lib, handle, _owned=True)

    @classmethod
    def from_json(
        cls,
        json_data: str | bytes,
        *,
        lib_path: str | os.PathLike[str] | None = None,
    ) -> Tokenizer:
        lib = load_lib(lib_path)
        raw = _as_utf8(json_data)
        handle = ctypes.c_void_p()
        check_status(
            lib,
            lib.ami_sdk_tokenizer_from_json(raw, len(raw), ctypes.byref(handle)),
        )
        return cls(lib, handle, _owned=True)

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            if self._owned:
                self._lib.ami_sdk_tokenizer_destroy(self._handle)
            self._handle = None
            self._owner = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Tokenizer:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("Tokenizer is closed")
        return self._handle

    @property
    def vocab_size(self) -> int:
        return int(self._lib.ami_sdk_tokenizer_vocab_size(self._require_open()))

    @property
    def dialect(self) -> Dialect:
        return Dialect(int(self._lib.ami_sdk_tokenizer_dialect(self._require_open())))

    def special_token_id(self, token: str | bytes) -> int | None:
        raw = _as_utf8(token)
        out_id = ctypes.c_uint32()
        rc = self._lib.ami_sdk_tokenizer_special_id(
            self._require_open(), raw, len(raw), ctypes.byref(out_id)
        )
        if rc != AMI_SDK_OK:
            return None
        return int(out_id.value)

    def count_tokens(self, text: str | bytes, *, allow_special_tokens: bool = True) -> int:
        raw = _as_utf8(text)
        if not raw:
            return 0
        out_count = ctypes.c_size_t()
        check_status(
            self._lib,
            self._lib.ami_sdk_count_tokens(
                self._require_open(),
                raw,
                len(raw),
                1 if allow_special_tokens else 0,
                ctypes.byref(out_count),
            ),
        )
        return int(out_count.value)

    def tokenize(self, text: str | bytes, *, allow_special_tokens: bool = True) -> list[int]:
        raw = _as_utf8(text)
        if not raw:
            return []
        handle = self._require_open()
        allow = 1 if allow_special_tokens else 0
        cap = max(16, len(raw))
        buf = (ctypes.c_uint32 * cap)()
        out_len = ctypes.c_size_t()
        rc = self._lib.ami_sdk_tokenize(
            handle, raw, len(raw), allow, buf, cap, ctypes.byref(out_len)
        )
        if rc == AMI_SDK_ERR_BUFFER_TOO_SMALL:
            needed = int(out_len.value)
            buf = (ctypes.c_uint32 * needed)()
            rc = self._lib.ami_sdk_tokenize(
                handle, raw, len(raw), allow, buf, needed, ctypes.byref(out_len)
            )
        check_status(self._lib, rc)
        return list(buf[: int(out_len.value)])

    def detokenize_bytes(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = True
    ) -> bytes:
        handle = self._require_open()
        vocab = self.vocab_size
        validated = _validate_token_ids(
            token_ids, vocab_size=vocab if vocab > 0 else None, name="token_ids"
        )
        n = len(validated)
        if n == 0:
            return b""
        ids_arr = (ctypes.c_uint32 * n)(*validated)
        skip = 1 if skip_special_tokens else 0
        needed = ctypes.c_size_t()
        check_status(
            self._lib,
            self._lib.ami_sdk_detokenize(
                handle, ids_arr, n, skip, None, 0, ctypes.byref(needed)
            ),
        )
        byte_len = int(needed.value)
        if byte_len == 0:
            return b""
        out_buf = (ctypes.c_uint8 * byte_len)()
        written = ctypes.c_size_t()
        check_status(
            self._lib,
            self._lib.ami_sdk_detokenize(
                handle,
                ids_arr,
                n,
                skip,
                ctypes.cast(out_buf, ctypes.c_void_p),
                byte_len,
                ctypes.byref(written),
            ),
        )
        return bytes(out_buf[: int(written.value)])

    def detokenize(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self.detokenize_bytes(token_ids, skip_special_tokens=skip_special_tokens).decode(
            "utf-8", errors="replace"
        )


class VisionInput:
    """Zero-copy IOSurface / CVPixelBuffer vision frame handle for G14 multimodal ingress."""

    def __init__(self, _lib: ctypes.CDLL, _handle: ctypes.c_void_p, _engine: Engine) -> None:
        self._lib = _lib
        self._handle: ctypes.c_void_p | None = _handle
        self._engine = _engine

    @staticmethod
    def _coerce_ptr(ptr: int | ctypes.c_void_p) -> ctypes.c_void_p:
        if isinstance(ptr, ctypes.c_void_p):
            return ptr
        if isinstance(ptr, int):
            return ctypes.c_void_p(ptr)
        raise TypeError(f"expected int pointer or ctypes.c_void_p, got {type(ptr).__name__}")

    @classmethod
    def from_cvpixelbuffer(cls, engine: Engine, cv_pixel_buffer_ptr: int | ctypes.c_void_p) -> VisionInput:
        eng_h = engine._require_open()
        handle = ctypes.c_void_p()
        check_status(
            engine._lib,
            engine._lib.ami_sdk_vision_from_cvpixelbuffer(
                eng_h, cls._coerce_ptr(cv_pixel_buffer_ptr), ctypes.byref(handle)
            ),
        )
        return cls(engine._lib, handle, engine)

    @classmethod
    def from_iosurface(cls, engine: Engine, iosurface_ptr: int | ctypes.c_void_p) -> VisionInput:
        eng_h = engine._require_open()
        handle = ctypes.c_void_p()
        check_status(
            engine._lib,
            engine._lib.ami_sdk_vision_from_iosurface(
                eng_h, cls._coerce_ptr(iosurface_ptr), ctypes.byref(handle)
            ),
        )
        return cls(engine._lib, handle, engine)

    @classmethod
    def from_iosurface_id(cls, engine: Engine, iosurface_id: int) -> VisionInput:
        eng_h = engine._require_open()
        handle = ctypes.c_void_p()
        check_status(
            engine._lib,
            engine._lib.ami_sdk_vision_from_iosurface_id(
                eng_h, ctypes.c_uint32(int(iosurface_id)), ctypes.byref(handle)
            ),
        )
        return cls(engine._lib, handle, engine)

    @classmethod
    def create_iosurface(
        cls,
        engine: Engine,
        width: int,
        height: int,
        *,
        fmt: VisionFormat = VisionFormat.BGRA8,
    ) -> VisionInput:
        eng_h = engine._require_open()
        handle = ctypes.c_void_p()
        check_status(
            engine._lib,
            engine._lib.ami_sdk_vision_create_iosurface(
                eng_h, int(fmt), int(width), int(height), ctypes.byref(handle)
            ),
        )
        return cls(engine._lib, handle, engine)

    @classmethod
    def from_image_file(cls, engine: Engine, path: str | os.PathLike[str]) -> VisionInput:
        eng_h = engine._require_open()
        handle = ctypes.c_void_p()
        raw_path = os.fsencode(os.fspath(path))
        check_status(
            engine._lib,
            engine._lib.ami_sdk_vision_load_image(eng_h, raw_path, ctypes.byref(handle)),
        )
        return cls(engine._lib, handle, engine)

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._lib.ami_sdk_vision_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> VisionInput:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("VisionInput is closed")
        return self._handle

    @property
    def info(self) -> VisionFrameInfo:
        raw = VisionInfoC(struct_size=ctypes.sizeof(VisionInfoC))
        check_status(
            self._lib, self._lib.ami_sdk_vision_query(self._require_open(), ctypes.byref(raw))
        )
        return VisionFrameInfo(
            format=VisionFormat(int(raw.format)),
            cv_pixel_format=int(raw.cv_pixel_format),
            width=int(raw.width),
            height=int(raw.height),
            plane_count=int(raw.plane_count),
            iosurface_id=int(raw.iosurface_id),
            zero_copy_verified=bool(raw.zero_copy_verified),
            is_page_aligned=bool(raw.is_page_aligned),
            base_address=int(raw.base_address),
            surface_alloc_size=int(raw.surface_alloc_size),
        )

    @property
    def iosurface_id(self) -> int:
        return self.info.iosurface_id

    @property
    def cvpixelbuffer_ptr(self) -> int:
        ptr = self._lib.ami_sdk_vision_cvpixelbuffer(self._require_open())
        return int(ptr or 0)

    @property
    def iosurface_ptr(self) -> int:
        ptr = self._lib.ami_sdk_vision_iosurface(self._require_open())
        return int(ptr or 0)

    @contextlib.contextmanager
    def lock_plane(self, plane: int = 0, *, read_only: bool = False) -> Iterator[memoryview]:
        handle = self._require_open()
        ro_flag = 1 if read_only else 0
        base_ptr = ctypes.c_void_p()
        bpr = ctypes.c_size_t()
        height = ctypes.c_uint32()
        check_status(
            self._lib,
            self._lib.ami_sdk_vision_lock_plane(
                handle,
                int(plane),
                ro_flag,
                ctypes.byref(base_ptr),
                ctypes.byref(bpr),
                ctypes.byref(height),
            ),
        )
        try:
            byte_len = int(bpr.value) * int(height.value)
            mv = _make_memoryview(
                int(base_ptr.value or 0),
                byte_len,
                read_only=read_only,
                fmt="B",
                shape=(int(height.value), int(bpr.value)),
            )
            yield mv
        finally:
            check_status(self._lib, self._lib.ami_sdk_vision_unlock(handle, ro_flag))


class Engine:
    """In-process aminfer engine bound via ctypes.CDLL (zero runtime dependencies)."""

    def __init__(
        self,
        amf_path: str | os.PathLike[str],
        *,
        config_json_path: str | os.PathLike[str] | None = None,
        tokenizer_json_path: str | os.PathLike[str] | None = None,
        wired_budget_bytes: int = 0,
        headroom_bytes: int = 0,
        live_safety_margin_bytes: int = 0,
        live_reclaimable_override_bytes: int | None = None,
        strict_live_ram: bool = True,
        max_seq_len: int = 2048,
        verify_crcs: bool = True,
        startup_canary: bool = True,
        draft_mode: DraftMode | str | None = None,
        draft_amf_path: str | os.PathLike[str] | None = None,
        draft_config_path: str | os.PathLike[str] | None = None,
        spec_max_tree_nodes: int = 16,
        spec_enable_pruning: bool = False,
        spec_confidence_threshold: float = 0.0,
        cold_expert_min_weight: float = 0.0,
        lib_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._lib = load_lib(lib_path)
        self._handle: ctypes.c_void_p | None = None
        self._tokenizer_view: Tokenizer | None = None

        opts = EngineOptionsC()
        self._lib.ami_sdk_engine_options_init(ctypes.byref(opts))
        opts.struct_size = ctypes.sizeof(EngineOptionsC)
        opts.abi_version = AMI_SDK_ABI_VERSION

        amf_bytes = os.fsencode(os.fspath(amf_path))
        cfg_bytes = os.fsencode(os.fspath(config_json_path)) if config_json_path is not None else None
        tok_bytes = (
            os.fsencode(os.fspath(tokenizer_json_path)) if tokenizer_json_path is not None else None
        )

        opts.amf_path = amf_bytes
        opts.config_json_path = cfg_bytes
        opts.tokenizer_json_path = tok_bytes
        opts.wired_budget_bytes = int(wired_budget_bytes)
        opts.headroom_bytes = int(headroom_bytes)
        opts.live_safety_margin_bytes = int(live_safety_margin_bytes)
        if live_reclaimable_override_bytes is not None:
            opts.has_live_reclaimable_override = 1
            opts.live_reclaimable_override_bytes = int(live_reclaimable_override_bytes)
        opts.strict_live_ram = 1 if strict_live_ram else 0
        opts.max_seq_len = int(max_seq_len)
        opts.verify_crcs = 1 if verify_crcs else 0
        opts.startup_canary = 1 if startup_canary else 0

        handle = ctypes.c_void_p()
        check_status(self._lib, self._lib.ami_sdk_engine_create(ctypes.byref(opts), ctypes.byref(handle)))
        self._handle = handle

        tok_ptr = self._lib.ami_sdk_engine_tokenizer(self._handle)
        if tok_ptr:
            self._tokenizer_view = Tokenizer(
                self._lib, ctypes.c_void_p(tok_ptr), _owned=False, _owner=self
            )
        self._vocab_size: int = self.stats.vocab_size
        self._active_lora: str | None = None
        try:
            self.set_cold_expert_min_weight(cold_expert_min_weight)
        except Exception:
            self.close()
            raise
        if draft_mode is not None or draft_amf_path is not None or draft_config_path is not None:
            try:
                self.configure_speculative(
                    draft_mode=draft_mode if draft_mode is not None else DraftMode.AUTO,
                    draft_amf_path=draft_amf_path,
                    draft_config_path=draft_config_path,
                    max_tree_nodes=spec_max_tree_nodes,
                    enable_pruning=spec_enable_pruning,
                    confidence_threshold=spec_confidence_threshold,
                )
            except Exception:
                self.close()
                raise

    @staticmethod
    def host_vm_snapshot(*, lib_path: str | os.PathLike[str] | None = None) -> HostVmSnapshot:
        lib = load_lib(lib_path)
        raw = HostVmInfoC(struct_size=ctypes.sizeof(HostVmInfoC))
        check_status(lib, lib.ami_sdk_host_vm_snapshot(ctypes.byref(raw)))
        return HostVmSnapshot(
            wired_bytes=int(raw.wired_bytes),
            free_bytes=int(raw.free_bytes),
            live_reclaimable_bytes=int(raw.live_reclaimable_bytes),
            swapouts=int(raw.swapouts),
            pageouts=int(raw.pageouts),
            phys_footprint_bytes=int(raw.phys_footprint_bytes),
        )

    def close(self) -> None:
        if getattr(self, "_tokenizer_view", None) is not None:
            self._tokenizer_view.close()
            self._tokenizer_view = None
        if getattr(self, "_handle", None) is not None:
            self._lib.ami_sdk_engine_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("Engine is closed")
        return self._handle

    @property
    def tokenizer(self) -> Tokenizer | None:
        self._require_open()
        return self._tokenizer_view

    def _require_tokenizer(self) -> Tokenizer:
        tok = self.tokenizer
        if tok is None:
            raise RuntimeError("Engine was initialized without a tokenizer.json")
        return tok

    @property
    def stats(self) -> EngineStats:
        raw = EngineStatsC(struct_size=ctypes.sizeof(EngineStatsC))
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_query_stats(self._require_open(), ctypes.byref(raw)),
        )
        return EngineStats(
            is_graph_model=bool(raw.is_graph_model),
            vocab_size=int(raw.vocab_size),
            hidden_size=int(raw.hidden_size),
            num_layers=int(raw.num_layers),
            max_seq_len=int(raw.max_seq_len),
            seq_len=int(raw.seq_len),
            total_wired_bytes=int(raw.total_wired_bytes),
            sdk_wired_bytes=int(raw.sdk_wired_bytes),
            wired_budget_bytes=int(raw.wired_budget_bytes),
            live_reclaimable_bytes=int(raw.live_reclaimable_bytes),
            live_budget_bytes=int(raw.live_budget_bytes),
        )

    def configure_speculative(
        self,
        draft_mode: DraftMode | str = DraftMode.AUTO,
        *,
        draft_amf_path: str | os.PathLike[str] | None = None,
        draft_config_path: str | os.PathLike[str] | None = None,
        max_tree_nodes: int = 16,
        enable_pruning: bool = False,
        confidence_threshold: float = 0.0,
    ) -> None:
        mode = _coerce_draft_mode(draft_mode)
        max_nodes_u32 = _validate_u32(max_tree_nodes, "max_tree_nodes")
        conf_f = _validate_finite_float(confidence_threshold, "confidence_threshold")
        if conf_f < 0.0 or conf_f > 1.0:
            raise _invalid_arg(f"confidence_threshold must be in [0.0, 1.0], got {conf_f}")

        opts = SpecOptionsC()
        self._lib.ami_sdk_spec_options_init(ctypes.byref(opts))
        opts.struct_size = ctypes.sizeof(SpecOptionsC)
        opts.draft_mode = int(mode)
        amf_b = os.fsencode(os.fspath(draft_amf_path)) if draft_amf_path is not None else None
        cfg_b = os.fsencode(os.fspath(draft_config_path)) if draft_config_path is not None else None
        opts.draft_amf_path = amf_b
        opts.draft_config_path = cfg_b
        opts.max_tree_nodes = max_nodes_u32
        opts.enable_pruning = 1 if enable_pruning else 0
        opts.confidence_threshold = float(conf_f)
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_configure_speculative(self._require_open(), ctypes.byref(opts)),
        )

    @property
    def spec_stats(self) -> SpecStats:
        raw = SpecStatsC(struct_size=ctypes.sizeof(SpecStatsC))
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_query_spec_stats(self._require_open(), ctypes.byref(raw)),
        )
        return SpecStats(
            active_draft_mode=DraftMode(int(raw.active_draft_mode)),
            rounds=int(raw.rounds),
            emitted_tokens=int(raw.emitted_tokens),
            drafted_nodes=int(raw.drafted_nodes),
            dflare_rounds=int(raw.dflare_rounds),
            dflare_core3_batches=int(raw.dflare_core3_batches),
            dfly2_rounds=int(raw.dfly2_rounds),
            dfly2_early_stops=int(raw.dfly2_early_stops),
            mean_accept_len=float(raw.mean_accept_len),
        )

    def init_lora_pool(self, pool_bytes: int = 0) -> None:
        """Preallocates the 16 KiB-aligned wired LoRA pool on this engine."""
        if isinstance(pool_bytes, bool) or not isinstance(pool_bytes, int) or pool_bytes < 0:
            raise _invalid_arg("pool_bytes must be a non-negative int")
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_init_lora_pool(self._require_open(), int(pool_bytes)),
        )

    def load_lora(
        self,
        name: str,
        path: str | os.PathLike[str],
        *,
        pool_bytes: int | None = None,
    ) -> int:
        """Hot-loads an `.amlora` (LoRA or DoRA) bundle under `name` and returns its slot ID."""
        handle = self._require_open()
        if not isinstance(name, str) or not name:
            raise _invalid_arg("LoRA adapter name must be a non-empty string")
        if pool_bytes is not None:
            self.init_lora_pool(pool_bytes)
        name_b = name.encode("utf-8")
        path_b = os.fsencode(os.fspath(path))
        out_slot = ctypes.c_uint32(0)
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_load_lora(handle, name_b, path_b, ctypes.byref(out_slot)),
        )
        return int(out_slot.value)

    def unload_lora(self, name: str) -> None:
        """Hot-unloads adapter `name`, returning its wired 16 KiB pages to the pool."""
        handle = self._require_open()
        if not isinstance(name, str) or not name:
            raise _invalid_arg("LoRA adapter name must be a non-empty string")
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_unload_lora(handle, name.encode("utf-8")),
        )
        if self._active_lora == name:
            self._active_lora = None

    def set_lora(self, name: str | None) -> None:
        """Activates loaded adapter `name`, or restores bit-exact base model when `name is None`."""
        handle = self._require_open()
        if name is None:
            check_status(self._lib, self._lib.ami_sdk_engine_set_active_lora(handle, None))
            self._active_lora = None
            return
        if not isinstance(name, str):
            raise _invalid_arg("LoRA adapter name must be a string or None")
        name_b = name.encode("utf-8") if name else None
        check_status(self._lib, self._lib.ami_sdk_engine_set_active_lora(handle, name_b))
        self._active_lora = name if name else None

    @property
    def lora_info(self) -> LoraInfo:
        """Queries current wired LoRA pool usage and active adapter state."""
        raw = LoraInfoC(struct_size=ctypes.sizeof(LoraInfoC))
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_query_lora_info(self._require_open(), ctypes.byref(raw)),
        )
        slot_val = int(raw.active_slot)
        return LoraInfo(
            active_adapters=int(raw.active_adapters),
            active_slot=None if slot_val == AMI_SDK_TOKEN_NONE else slot_val,
            max_adapters=int(raw.max_adapters),
            pool_capacity_bytes=int(raw.pool_capacity_bytes),
            pool_used_bytes=int(raw.pool_used_bytes),
        )

    @property
    def vocab_size(self) -> int:
        return self.stats.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.stats.hidden_size

    @property
    def num_layers(self) -> int:
        return self.stats.num_layers

    @property
    def max_seq_len(self) -> int:
        return self.stats.max_seq_len

    @property
    def seq_len(self) -> int:
        return self.stats.seq_len

    def tokenize(self, text: str | bytes, *, allow_special_tokens: bool = True) -> list[int]:
        return self._require_tokenizer().tokenize(text, allow_special_tokens=allow_special_tokens)

    def count_tokens(self, text: str | bytes, *, allow_special_tokens: bool = True) -> int:
        return self._require_tokenizer().count_tokens(text, allow_special_tokens=allow_special_tokens)

    def detokenize(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self._require_tokenizer().detokenize(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def reset_kv_cache(self) -> None:
        check_status(self._lib, self._lib.ami_sdk_engine_reset_kv(self._require_open()))

    def set_cold_expert_min_weight(self, min_weight: float) -> None:
        """Sets the C27 opt-in cold-tail skipping weight threshold in [0.0, 1.0)."""
        handle = self._require_open()
        if isinstance(min_weight, bool) or not isinstance(min_weight, (int, float)):
            raise _invalid_arg("cold_expert_min_weight must be a finite float in [0.0, 1.0)")
        val = float(min_weight)
        if not math.isfinite(val) or val < 0.0 or val >= 1.0:
            raise _invalid_arg("cold_expert_min_weight must be finite in [0.0, 1.0)")
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_set_cold_expert_min_weight(handle, ctypes.c_float(val)),
        )

    @property
    def cold_expert_min_weight(self) -> float:
        """Queries the active C27 cold-tail skipping weight threshold (0.0 = lossless)."""
        out = ctypes.c_float(0.0)
        check_status(
            self._lib,
            self._lib.ami_sdk_engine_get_cold_expert_min_weight(
                self._require_open(), ctypes.byref(out)
            ),
        )
        return float(out.value)

    def _prompt_to_ids(self, prompt: str | bytes | Sequence[int]) -> list[int]:
        if isinstance(prompt, (str, bytes)):
            return self.tokenize(prompt, allow_special_tokens=True)
        v = self._vocab_size if self._vocab_size > 0 else None
        return _validate_token_ids(prompt, vocab_size=v, name="prompt")

    def prefill(
        self,
        prompt: str | bytes | Sequence[int],
        *,
        compute_logits: bool = True,
        reset_kv_cache: bool = True,
        cancel_token: CancelToken | None = None,
    ) -> None:
        ids = self._prompt_to_ids(prompt)
        if not ids:
            raise InvalidArgumentError(-1, "AMI_SDK_ERR_INVALID_ARGUMENT", "prompt is empty")
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        ctok = cancel_token._require_open() if cancel_token is not None else None
        check_status(
            self._lib,
            self._lib.ami_sdk_prefill_tokens(
                self._require_open(),
                arr,
                len(ids),
                1 if compute_logits else 0,
                1 if reset_kv_cache else 0,
                ctok,
            ),
        )

    def step_decode(self, token_id: int) -> int:
        handle = self._require_open()
        v = self._vocab_size if self._vocab_size > 0 else None
        tid = _validate_token_id(token_id, vocab_size=v, name="token_id")
        out_next = ctypes.c_uint32()
        check_status(
            self._lib,
            self._lib.ami_sdk_step_decode(
                handle, tid, ctypes.byref(out_next)
            ),
        )
        return int(out_next.value)

    def generate(
        self,
        prompt: str | bytes | Sequence[int],
        config: GenerationConfig | None = None,
        *,
        on_chunk: Callable[[StreamChunk], bool | None] | None = None,
        lora: str | None | object = _LORA_UNSET,
    ) -> GenerationOutput:
        handle = self._require_open()
        lora_switched = lora is not _LORA_UNSET
        prev_lora = self._active_lora
        if lora_switched:
            if lora is not None and not isinstance(lora, str):
                raise _invalid_arg("lora must be a string or None")
            self.set_lora(lora)
        try:
            return self._generate_inner(handle, prompt, config, on_chunk=on_chunk)
        finally:
            if lora_switched:
                self.set_lora(prev_lora)

    def _generate_inner(
        self,
        handle: ctypes.c_void_p,
        prompt: str | bytes | Sequence[int],
        config: GenerationConfig | None = None,
        *,
        on_chunk: Callable[[StreamChunk], bool | None] | None = None,
    ) -> GenerationOutput:
        cfg = config if config is not None else GenerationConfig()
        c_params = cfg.to_c(self._lib)

        max_toks = int(c_params.max_tokens)
        out_ids = (ctypes.c_uint32 * max_toks)() if max_toks > 0 else None
        c_res = GenResultC(struct_size=ctypes.sizeof(GenResultC))

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[str] = []
        current_tool_parts: list[str] = []
        cb_exc: BaseException | None = None

        @STREAM_CB_TYPE
        def _c_stream_cb(ev_ptr: Any, _ud: int | None) -> int:
            nonlocal cb_exc
            try:
                ev = ev_ptr.contents
                kind = DeltaKind(int(ev.delta_kind))
                raw_len = int(ev.utf8_len)
                piece = (
                    ctypes.string_at(ev.utf8_ptr, raw_len).decode("utf-8", errors="replace")
                    if raw_len > 0
                    else ""
                )
                if kind == DeltaKind.TEXT:
                    if piece:
                        text_parts.append(piece)
                elif kind == DeltaKind.REASONING:
                    if piece:
                        reasoning_parts.append(piece)
                elif kind == DeltaKind.TOOL_CALL:
                    if piece:
                        current_tool_parts.append(piece)
                elif kind == DeltaKind.TOOL_CALL_END:
                    tool_calls.append("".join(current_tool_parts))
                    current_tool_parts.clear()

                if on_chunk is not None:
                    tok_val = int(ev.token_id)
                    chunk = StreamChunk(
                        token_id=None if tok_val == AMI_SDK_TOKEN_NONE else tok_val,
                        step_index=int(ev.step_index),
                        kind=kind,
                        text=piece,
                        is_final_flush=bool(ev.is_final_flush),
                    )
                    keep_going = on_chunk(chunk)
                    if keep_going is False:
                        return 1
                return 0
            except BaseException as exc:  # noqa: BLE001
                cb_exc = exc
                return 1

        cb_ptr = ctypes.cast(_c_stream_cb, ctypes.c_void_p)
        if isinstance(prompt, (str, bytes)):
            raw_prompt = _as_utf8(prompt)
            if not raw_prompt:
                raise _invalid_arg("prompt is empty")
            rc = self._lib.ami_sdk_generate_text(
                handle,
                raw_prompt,
                len(raw_prompt),
                ctypes.byref(c_params),
                cb_ptr,
                None,
                out_ids,
                max_toks,
                ctypes.byref(c_res),
            )
        else:
            v = self._vocab_size if self._vocab_size > 0 else None
            ids = _validate_token_ids(prompt, vocab_size=v, name="prompt")
            if not ids:
                raise _invalid_arg("prompt is empty")
            ids_arr = (ctypes.c_uint32 * len(ids))(*ids)
            rc = self._lib.ami_sdk_generate_tokens(
                handle,
                ids_arr,
                len(ids),
                ctypes.byref(c_params),
                cb_ptr,
                None,
                out_ids,
                max_toks,
                ctypes.byref(c_res),
            )

        if cb_exc is not None:
            raise cb_exc
        check_status(self._lib, rc)

        if current_tool_parts:
            tool_calls.append("".join(current_tool_parts))

        gen_count = int(c_res.generated_tokens)
        if max_toks == 0 or gen_count == 0 or out_ids is None:
            gen_tuple: tuple[int, ...] = ()
        else:
            gen_tuple = tuple(int(x) for x in out_ids[: min(gen_count, max_toks)])
        text_str = "".join(text_parts)
        if not text_str and not reasoning_parts and not tool_calls and self._tokenizer_view is None:
            text_str = ""

        return GenerationOutput(
            text=text_str,
            reasoning="".join(reasoning_parts),
            tool_calls=tuple(tool_calls),
            token_ids=gen_tuple,
            prompt_tokens=int(c_res.prompt_tokens),
            generated_tokens=gen_count,
            finish_reason=FinishReason(int(c_res.finish_reason)),
            seed_used=int(c_res.seed_used),
            prefill_ns=float(c_res.prefill_ns),
            decode_ns=float(c_res.decode_ns),
        )

    def stream(
        self,
        prompt: str | bytes | Sequence[int],
        config: GenerationConfig | None = None,
        *,
        lora: str | None | object = _LORA_UNSET,
    ) -> Iterator[StreamChunk]:
        """Streams generation chunks in real time while releasing the GIL on native steps."""
        self._require_open()
        q: queue.SimpleQueue[StreamChunk | BaseException | None] = queue.SimpleQueue()
        stop_event = threading.Event()

        def _on_chunk(chunk: StreamChunk) -> bool:
            if stop_event.is_set():
                return False
            q.put(chunk)
            return not stop_event.is_set()

        def _worker() -> None:
            try:
                self.generate(prompt, config, on_chunk=_on_chunk, lora=lora)
            except BaseException as exc:  # noqa: BLE001
                q.put(exc)
            finally:
                q.put(None)

        worker = threading.Thread(target=_worker, name="aminfer-stream", daemon=True)
        worker.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop_event.set()
            worker.join()

    def logits_view(self, dtype: DType | str = DType.F32) -> memoryview:
        """Returns a read-only zero-copy PEP 3118 memoryview over wired logits."""
        dt = _coerce_dtype(dtype)
        view = TensorViewC(struct_size=ctypes.sizeof(TensorViewC))
        check_status(
            self._lib,
            self._lib.ami_sdk_get_logits_view(self._require_open(), int(dt), ctypes.byref(view)),
        )
        fmt = "e" if dt == DType.F16 else "f"
        cols = int(view.shape[1])
        return _make_memoryview(
            int(view.data or 0),
            int(view.byte_len),
            read_only=True,
            fmt=fmt,
            shape=(cols,),
        )

    def hidden_states_view(
        self, dtype: DType | str = DType.F32, *, normalized: bool = True
    ) -> memoryview:
        """Returns a read-only zero-copy PEP 3118 memoryview over the latest hidden state."""
        dt = _coerce_dtype(dtype)
        view = TensorViewC(struct_size=ctypes.sizeof(TensorViewC))
        check_status(
            self._lib,
            self._lib.ami_sdk_get_hidden_state_view(
                self._require_open(),
                int(dt),
                1 if normalized else 0,
                ctypes.byref(view),
            ),
        )
        fmt = "e" if dt == DType.F16 else "f"
        cols = int(view.shape[1])
        return _make_memoryview(
            int(view.data or 0),
            int(view.byte_len),
            read_only=True,
            fmt=fmt,
            shape=(cols,),
        )

    @overload
    def embed(
        self,
        inputs: str | bytes | Sequence[int],
        *,
        pooling: PoolingMode | str = ...,
        dimensions: int | None = ...,
        normalize: bool = ...,
        cancel_token: CancelToken | None = ...,
    ) -> list[float]: ...

    @overload
    def embed(
        self,
        inputs: Sequence[str],
        *,
        pooling: PoolingMode | str = ...,
        dimensions: int | None = ...,
        normalize: bool = ...,
        cancel_token: CancelToken | None = ...,
    ) -> list[list[float]]: ...

    def embed(
        self,
        inputs: str | bytes | Sequence[int] | Sequence[str],
        *,
        pooling: PoolingMode | str = PoolingMode.LAST_TOKEN,
        dimensions: int | None = None,
        normalize: bool = True,
        cancel_token: CancelToken | None = None,
    ) -> list[float] | list[list[float]]:
        """Extracts pooled embeddings while eliding lm_head (C21)."""
        handle = self._require_open()
        pool_mode = _coerce_pooling(pooling)
        c_params = EmbedParamsC()
        self._lib.ami_sdk_embed_params_init(ctypes.byref(c_params))
        c_params.pooling = int(pool_mode)
        c_params.dimensions = 0 if dimensions is None else int(dimensions)
        c_params.normalize = 1 if normalize else 0
        c_params.cancel_token = (
            cancel_token._require_open() if cancel_token is not None else None
        )

        target_cap = self.hidden_size if not dimensions else int(dimensions)
        if target_cap <= 0:
            raise InvalidArgumentError(
                -1, "AMI_SDK_ERR_INVALID_ARGUMENT", "dimensions must be positive"
            )

        def _embed_one(item: str | bytes | Sequence[int]) -> list[float]:
            out_buf = (ctypes.c_float * target_cap)()
            out_dims = ctypes.c_size_t()
            if isinstance(item, (str, bytes)):
                raw = _as_utf8(item)
                check_status(
                    self._lib,
                    self._lib.ami_sdk_embed_text(
                        handle,
                        raw,
                        len(raw),
                        ctypes.byref(c_params),
                        out_buf,
                        target_cap,
                        ctypes.byref(out_dims),
                    ),
                )
            else:
                v = self._vocab_size if self._vocab_size > 0 else None
                ids = _validate_token_ids(item, vocab_size=v, name="token_ids")
                if not ids:
                    raise _invalid_arg("token_ids sequence is empty")
                ids_arr = (ctypes.c_uint32 * len(ids))(*ids)
                check_status(
                    self._lib,
                    self._lib.ami_sdk_embed_tokens(
                        handle,
                        ids_arr,
                        len(ids),
                        ctypes.byref(c_params),
                        out_buf,
                        target_cap,
                        ctypes.byref(out_dims),
                    ),
                )
            return [float(x) for x in out_buf[: int(out_dims.value)]]

        if isinstance(inputs, (str, bytes)):
            return _embed_one(inputs)
        if len(inputs) > 0 and isinstance(inputs[0], str):
            return [_embed_one(s) for s in inputs]  # type: ignore[arg-type]
        return _embed_one(inputs)  # type: ignore[arg-type]

    def rerank(
        self,
        query: str | bytes | Sequence[int],
        documents: Sequence[str | bytes] | Sequence[Sequence[int]],
        *,
        mode: RerankMode | str = RerankMode.LOGIT_DIFF,
        positive_token_id: int | None = None,
        negative_token_id: int | None = None,
        top_n: int | None = None,
        cancel_token: CancelToken | None = None,
    ) -> list[RerankItem]:
        """Scores (query, document) pairs via C21 cross-encoder logit scoring and sorts descending."""
        handle = self._require_open()
        num_docs = len(documents)
        if num_docs == 0:
            return []

        rmode = _coerce_rerank_mode(mode)
        c_params = RerankParamsC()
        self._lib.ami_sdk_rerank_params_init(ctypes.byref(c_params))
        c_params.score_mode = int(rmode)
        c_params.positive_token_id = (
            AMI_SDK_TOKEN_NONE
            if positive_token_id is None
            else _validate_u32(positive_token_id, "positive_token_id")
        )
        c_params.negative_token_id = (
            AMI_SDK_TOKEN_NONE
            if negative_token_id is None
            else _validate_u32(negative_token_id, "negative_token_id")
        )
        c_params.cancel_token = (
            cancel_token._require_open() if cancel_token is not None else None
        )

        scores_arr = (ctypes.c_float * num_docs)()

        if isinstance(query, (str, bytes)):
            q_bytes = _as_utf8(query)
            encoded_docs = [_as_utf8(d) for d in documents]  # type: ignore[arg-type]
            doc_ptrs = (ctypes.c_char_p * num_docs)(*encoded_docs)
            doc_lens = (ctypes.c_size_t * num_docs)(*(len(b) for b in encoded_docs))
            check_status(
                self._lib,
                self._lib.ami_sdk_rerank_texts(
                    handle,
                    q_bytes,
                    len(q_bytes),
                    doc_ptrs,
                    doc_lens,
                    num_docs,
                    ctypes.byref(c_params),
                    scores_arr,
                ),
            )
        else:
            v = self._vocab_size if self._vocab_size > 0 else None
            q_ids = _validate_token_ids(query, vocab_size=v, name="query")
            q_arr = (ctypes.c_uint32 * len(q_ids))(*q_ids) if q_ids else None
            c_u32_p = ctypes.POINTER(ctypes.c_uint32)
            doc_arrays = []
            doc_ptrs_tok = (c_u32_p * num_docs)()
            doc_lens_tok = (ctypes.c_size_t * num_docs)()
            for i, d_seq in enumerate(documents):
                d_ids = _validate_token_ids(d_seq, vocab_size=v, name=f"documents[{i}]")  # type: ignore[arg-type]
                arr = (ctypes.c_uint32 * len(d_ids))(*d_ids) if d_ids else None
                doc_arrays.append(arr)
                doc_ptrs_tok[i] = ctypes.cast(arr, c_u32_p) if arr is not None else c_u32_p()
                doc_lens_tok[i] = len(d_ids)
            check_status(
                self._lib,
                self._lib.ami_sdk_rerank_tokens(
                    handle,
                    q_arr,
                    len(q_ids),
                    doc_ptrs_tok,
                    doc_lens_tok,
                    num_docs,
                    ctypes.byref(c_params),
                    scores_arr,
                ),
            )

        items = [RerankItem(index=i, score=float(scores_arr[i])) for i in range(num_docs)]
        items.sort(key=lambda it: it.score, reverse=True)
        if top_n is not None:
            return items[: max(0, int(top_n))]
        return items

    def prefill_vision(
        self,
        frames: VisionInput | Sequence[VisionInput],
        *,
        prefix: str | bytes | Sequence[int] = (),
        suffix: str | bytes | Sequence[int] = (),
        patch_size: int = 16,
        t_patch: int = 1,
        spatial_merge: int = 2,
        compute_logits: bool = True,
        cancel_token: CancelToken | None = None,
    ) -> VisionPrefillResult:
        """Runs G14 zero-copy IOSurface / CVPixelBuffer vision ingress + multimodal prefill."""
        handle = self._require_open()
        frame_list = [frames] if isinstance(frames, VisionInput) else list(frames)
        if not frame_list:
            raise InvalidArgumentError(-1, "AMI_SDK_ERR_INVALID_ARGUMENT", "frames is empty")

        prefix_ids = self._prompt_to_ids(prefix) if prefix else []
        suffix_ids = self._prompt_to_ids(suffix) if suffix else []

        p_arr = (ctypes.c_uint32 * len(prefix_ids))(*prefix_ids) if prefix_ids else None
        s_arr = (ctypes.c_uint32 * len(suffix_ids))(*suffix_ids) if suffix_ids else None
        f_ptrs = (ctypes.c_void_p * len(frame_list))(*(f._require_open() for f in frame_list))

        c_params = VisionParamsC()
        self._lib.ami_sdk_vision_params_init(ctypes.byref(c_params))
        c_params.patch_size = int(patch_size)
        c_params.t_patch = int(t_patch)
        c_params.spatial_merge = int(spatial_merge)
        c_params.compute_logits = 1 if compute_logits else 0
        c_params.cancel_token = (
            cancel_token._require_open() if cancel_token is not None else None
        )

        c_res = VisionResultC(struct_size=ctypes.sizeof(VisionResultC))
        check_status(
            self._lib,
            self._lib.ami_sdk_prefill_vision(
                handle,
                p_arr,
                len(prefix_ids),
                f_ptrs,
                len(frame_list),
                s_arr,
                len(suffix_ids),
                ctypes.byref(c_params),
                ctypes.byref(c_res),
            ),
        )
        return VisionPrefillResult(
            first_token_id=int(c_res.first_token_id),
            prefix_tokens=int(c_res.prefix_tokens),
            vision_tokens=int(c_res.vision_tokens),
            suffix_tokens=int(c_res.suffix_tokens),
            total_prompt_tokens=int(c_res.total_prompt_tokens),
            grid=(int(c_res.grid_t), int(c_res.grid_h), int(c_res.grid_w)),
            all_frames_zero_copy=bool(c_res.all_frames_zero_copy),
            all_frames_page_aligned=bool(c_res.all_frames_page_aligned),
            vision_encode_us=float(c_res.vision_encode_us),
        )
