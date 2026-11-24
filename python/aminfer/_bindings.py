"""Low-level pure-stdlib ctypes bindings for libaminfer.dylib (ADR-0022 C17/C21).

Uses ctypes.CDLL so Python's Global Interpreter Lock (GIL) is released across
every native prefill, decode, embedding, reranking, and vision call.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import pathlib
from typing import Final

AMI_SDK_ABI_VERSION: Final[int] = 1
AMI_SDK_MAX_STOP_TOKENS: Final[int] = 8
AMI_SDK_TOKEN_NONE: Final[int] = 0xFFFFFFFF

# Tokenizer dialects (must match include/aminfer.h).
AMI_SDK_DIALECT_QWEN: Final[int] = 0
AMI_SDK_DIALECT_DEEPSEEK: Final[int] = 1
AMI_SDK_DIALECT_INKLING2: Final[int] = 2
AMI_SDK_DIALECT_HY4: Final[int] = 3

# Reasoning effort tiers (must match include/aminfer.h).
AMI_SDK_EFFORT_DEFAULT: Final[int] = 0
AMI_SDK_EFFORT_NONE: Final[int] = 1
AMI_SDK_EFFORT_NO_THINK: Final[int] = 2
AMI_SDK_EFFORT_MINIMAL: Final[int] = 3
AMI_SDK_EFFORT_LOW: Final[int] = 4
AMI_SDK_EFFORT_MEDIUM: Final[int] = 5
AMI_SDK_EFFORT_HIGH: Final[int] = 6
AMI_SDK_EFFORT_XHIGH: Final[int] = 7
AMI_SDK_EFFORT_MAX: Final[int] = 8
AMI_SDK_EFFORT_CUSTOM_FLOAT: Final[int] = 9

# Status codes (must match include/aminfer.h).
AMI_SDK_OK: Final[int] = 0
AMI_SDK_ERR_INVALID_ARGUMENT: Final[int] = -1
AMI_SDK_ERR_ABI_MISMATCH: Final[int] = -2
AMI_SDK_ERR_OUT_OF_MEMORY: Final[int] = -3
AMI_SDK_ERR_IO: Final[int] = -4
AMI_SDK_ERR_FORMAT: Final[int] = -5
AMI_SDK_ERR_BUDGET_EXCEEDED: Final[int] = -6
AMI_SDK_ERR_LIVE_RAM_EXCEEDED: Final[int] = -7
AMI_SDK_ERR_METAL: Final[int] = -8
AMI_SDK_ERR_MODEL: Final[int] = -9
AMI_SDK_ERR_TOKENIZER: Final[int] = -10
AMI_SDK_ERR_BUFFER_TOO_SMALL: Final[int] = -11
AMI_SDK_ERR_SEQUENCE_FULL: Final[int] = -12
AMI_SDK_ERR_VISION: Final[int] = -13
AMI_SDK_ERR_CANCELLED: Final[int] = -14
AMI_SDK_ERR_UNSUPPORTED: Final[int] = -15


class AminferError(RuntimeError):
    """Base exception raised when a libaminfer C-ABI call returns non-zero status."""

    def __init__(self, status: int, status_name: str, detail: str) -> None:
        self.status = status
        self.status_name = status_name
        self.detail = detail
        msg = f"{status_name} ({status}): {detail}" if detail else f"{status_name} ({status})"
        super().__init__(msg)


class BudgetExceededError(AminferError):
    """Raised when co-tenant wired_budget_bytes or R9 live zero-swap RAM is exceeded."""


class CancelledError(AminferError):
    """Raised when an active request is cooperatively cancelled via CancelToken."""


class InvalidArgumentError(AminferError):
    """Raised on invalid arguments or out-of-range token IDs."""


class SequenceFullError(AminferError):
    """Raised when prompt or decode tokens exceed max_seq_len."""


class VisionError(AminferError):
    """Raised when zero-copy IOSurface / CVPixelBuffer vision ingress fails."""


class HostVmInfoC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("wired_bytes", ctypes.c_uint64),
        ("free_bytes", ctypes.c_uint64),
        ("live_reclaimable_bytes", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("phys_footprint_bytes", ctypes.c_uint64),
    ]


class EngineOptionsC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("amf_path", ctypes.c_char_p),
        ("config_json_path", ctypes.c_char_p),
        ("tokenizer_json_path", ctypes.c_char_p),
        ("wired_budget_bytes", ctypes.c_uint64),
        ("headroom_bytes", ctypes.c_uint64),
        ("live_safety_margin_bytes", ctypes.c_uint64),
        ("live_reclaimable_override_bytes", ctypes.c_uint64),
        ("has_live_reclaimable_override", ctypes.c_uint32),
        ("strict_live_ram", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("verify_crcs", ctypes.c_uint32),
        ("startup_canary", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class EngineStatsC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("is_graph_model", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("hidden_size", ctypes.c_uint32),
        ("num_layers", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("seq_len", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("total_wired_bytes", ctypes.c_uint64),
        ("sdk_wired_bytes", ctypes.c_uint64),
        ("wired_budget_bytes", ctypes.c_uint64),
        ("live_reclaimable_bytes", ctypes.c_uint64),
        ("live_budget_bytes", ctypes.c_uint64),
    ]


class GenParamsC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("max_tokens", ctypes.c_uint32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("min_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("has_seed", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("stop_token_ids", ctypes.c_uint32 * AMI_SDK_MAX_STOP_TOKENS),
        ("stop_token_count", ctypes.c_uint32),
        ("ignore_eos", ctypes.c_uint32),
        ("start_in_think", ctypes.c_uint32),
        ("reset_kv_cache", ctypes.c_uint32),
        ("skip_special_tokens", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("reasoning_effort", ctypes.c_uint32),
        ("reasoning_effort_value", ctypes.c_float),
        ("cancel_token", ctypes.c_void_p),
    ]


class StreamEventC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("token_id", ctypes.c_uint32),
        ("step_index", ctypes.c_uint32),
        ("delta_kind", ctypes.c_uint32),
        ("utf8_ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("utf8_len", ctypes.c_size_t),
        ("is_final_flush", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


STREAM_CB_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.POINTER(StreamEventC),
    ctypes.c_void_p,
)


class GenResultC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("prompt_tokens", ctypes.c_uint32),
        ("generated_tokens", ctypes.c_uint32),
        ("finish_reason", ctypes.c_uint32),
        ("seed_used", ctypes.c_uint64),
        ("prefill_ns", ctypes.c_double),
        ("decode_ns", ctypes.c_double),
    ]


class TensorViewC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("byte_len", ctypes.c_size_t),
        ("ndim", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("shape", ctypes.c_size_t * 2),
    ]


class EmbedParamsC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("pooling", ctypes.c_uint32),
        ("dimensions", ctypes.c_uint32),
        ("normalize", ctypes.c_uint32),
        ("cancel_token", ctypes.c_void_p),
    ]


class RerankParamsC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("score_mode", ctypes.c_uint32),
        ("positive_token_id", ctypes.c_uint32),
        ("negative_token_id", ctypes.c_uint32),
        ("cancel_token", ctypes.c_void_p),
    ]


class VisionInfoC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
        ("cv_pixel_format", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("plane_count", ctypes.c_uint32),
        ("iosurface_id", ctypes.c_uint32),
        ("zero_copy_verified", ctypes.c_uint32),
        ("is_page_aligned", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("base_address", ctypes.c_uint64),
        ("surface_alloc_size", ctypes.c_uint64),
    ]


class VisionParamsC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("patch_size", ctypes.c_uint32),
        ("t_patch", ctypes.c_uint32),
        ("spatial_merge", ctypes.c_uint32),
        ("mean_rgb", ctypes.c_float * 3),
        ("inv_std_rgb", ctypes.c_float * 3),
        ("compute_logits", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("cancel_token", ctypes.c_void_p),
    ]


class VisionResultC(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("first_token_id", ctypes.c_uint32),
        ("prefix_tokens", ctypes.c_uint32),
        ("vision_tokens", ctypes.c_uint32),
        ("suffix_tokens", ctypes.c_uint32),
        ("total_prompt_tokens", ctypes.c_uint32),
        ("grid_t", ctypes.c_uint32),
        ("grid_h", ctypes.c_uint32),
        ("grid_w", ctypes.c_uint32),
        ("all_frames_zero_copy", ctypes.c_uint32),
        ("all_frames_page_aligned", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("vision_encode_us", ctypes.c_double),
    ]


_LIB_CACHE: dict[str, ctypes.CDLL] = {}


def find_libaminfer(lib_path: str | os.PathLike[str] | None = None) -> str:
    """Resolves the path to libaminfer.dylib."""
    if lib_path is not None:
        resolved = pathlib.Path(lib_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"libaminfer.dylib not found at {resolved}")
        return str(resolved)

    env_override = os.environ.get("AMINFER_LIB_PATH")
    if env_override:
        resolved = pathlib.Path(env_override).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"AMINFER_LIB_PATH does not exist: {resolved}")
        return str(resolved)

    pkg_dir = pathlib.Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parents[1] / "zig-out" / "lib" / "libaminfer.dylib",
        pkg_dir / "libaminfer.dylib",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)

    found = ctypes.util.find_library("aminfer")
    if found:
        return found

    raise FileNotFoundError(
        "Unable to locate libaminfer.dylib. Run `zig build` or set AMINFER_LIB_PATH."
    )


def _configure_cdll(lib: ctypes.CDLL) -> None:
    c_u32_p = ctypes.POINTER(ctypes.c_uint32)
    c_f32_p = ctypes.POINTER(ctypes.c_float)
    c_size_p = ctypes.POINTER(ctypes.c_size_t)
    c_void_pp = ctypes.POINTER(ctypes.c_void_p)

    lib.ami_sdk_abi_version.argtypes = []
    lib.ami_sdk_abi_version.restype = ctypes.c_uint32

    lib.ami_sdk_version_string.argtypes = []
    lib.ami_sdk_version_string.restype = ctypes.c_char_p

    lib.ami_sdk_status_string.argtypes = [ctypes.c_int32]
    lib.ami_sdk_status_string.restype = ctypes.c_char_p

    lib.ami_sdk_last_error.argtypes = []
    lib.ami_sdk_last_error.restype = ctypes.c_char_p

    lib.ami_sdk_clear_last_error.argtypes = []
    lib.ami_sdk_clear_last_error.restype = None

    lib.ami_sdk_host_vm_snapshot.argtypes = [ctypes.POINTER(HostVmInfoC)]
    lib.ami_sdk_host_vm_snapshot.restype = ctypes.c_int32

    lib.ami_sdk_engine_options_init.argtypes = [ctypes.POINTER(EngineOptionsC)]
    lib.ami_sdk_engine_options_init.restype = None

    lib.ami_sdk_gen_params_init.argtypes = [ctypes.POINTER(GenParamsC)]
    lib.ami_sdk_gen_params_init.restype = None

    lib.ami_sdk_embed_params_init.argtypes = [ctypes.POINTER(EmbedParamsC)]
    lib.ami_sdk_embed_params_init.restype = None

    lib.ami_sdk_rerank_params_init.argtypes = [ctypes.POINTER(RerankParamsC)]
    lib.ami_sdk_rerank_params_init.restype = None

    lib.ami_sdk_vision_params_init.argtypes = [ctypes.POINTER(VisionParamsC)]
    lib.ami_sdk_vision_params_init.restype = None

    # CancelToken
    lib.ami_sdk_cancel_token_create.argtypes = [c_void_pp]
    lib.ami_sdk_cancel_token_create.restype = ctypes.c_int32
    lib.ami_sdk_cancel_token_destroy.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_cancel_token_destroy.restype = None
    lib.ami_sdk_cancel_token_cancel.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_cancel_token_cancel.restype = None
    lib.ami_sdk_cancel_token_reset.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_cancel_token_reset.restype = None
    lib.ami_sdk_cancel_token_is_cancelled.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_cancel_token_is_cancelled.restype = ctypes.c_uint32

    # Tokenizer
    lib.ami_sdk_tokenizer_from_file.argtypes = [ctypes.c_char_p, c_void_pp]
    lib.ami_sdk_tokenizer_from_file.restype = ctypes.c_int32
    lib.ami_sdk_tokenizer_from_json.argtypes = [ctypes.c_char_p, ctypes.c_size_t, c_void_pp]
    lib.ami_sdk_tokenizer_from_json.restype = ctypes.c_int32
    lib.ami_sdk_tokenizer_destroy.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_tokenizer_destroy.restype = None
    lib.ami_sdk_tokenizer_vocab_size.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_tokenizer_vocab_size.restype = ctypes.c_uint32
    lib.ami_sdk_tokenizer_dialect.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_tokenizer_dialect.restype = ctypes.c_uint32
    lib.ami_sdk_tokenizer_special_id.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        c_u32_p,
    ]
    lib.ami_sdk_tokenizer_special_id.restype = ctypes.c_int32
    lib.ami_sdk_tokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        c_u32_p,
        ctypes.c_size_t,
        c_size_p,
    ]
    lib.ami_sdk_tokenize.restype = ctypes.c_int32
    lib.ami_sdk_count_tokens.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        c_size_p,
    ]
    lib.ami_sdk_count_tokens.restype = ctypes.c_int32
    lib.ami_sdk_detokenize.argtypes = [
        ctypes.c_void_p,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        c_size_p,
    ]
    lib.ami_sdk_detokenize.restype = ctypes.c_int32

    # Engine
    lib.ami_sdk_engine_create.argtypes = [ctypes.POINTER(EngineOptionsC), c_void_pp]
    lib.ami_sdk_engine_create.restype = ctypes.c_int32
    lib.ami_sdk_engine_destroy.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_engine_destroy.restype = None
    lib.ami_sdk_engine_reset_kv.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_engine_reset_kv.restype = ctypes.c_int32
    lib.ami_sdk_engine_query_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(EngineStatsC)]
    lib.ami_sdk_engine_query_stats.restype = ctypes.c_int32
    lib.ami_sdk_engine_tokenizer.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_engine_tokenizer.restype = ctypes.c_void_p

    lib.ami_sdk_prefill_tokens.argtypes = [
        ctypes.c_void_p,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.ami_sdk_prefill_tokens.restype = ctypes.c_int32

    lib.ami_sdk_step_decode.argtypes = [ctypes.c_void_p, ctypes.c_uint32, c_u32_p]
    lib.ami_sdk_step_decode.restype = ctypes.c_int32

    lib.ami_sdk_generate_tokens.argtypes = [
        ctypes.c_void_p,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.POINTER(GenParamsC),
        ctypes.c_void_p,
        ctypes.c_void_p,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.POINTER(GenResultC),
    ]
    lib.ami_sdk_generate_tokens.restype = ctypes.c_int32

    lib.ami_sdk_generate_text.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(GenParamsC),
        ctypes.c_void_p,
        ctypes.c_void_p,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.POINTER(GenResultC),
    ]
    lib.ami_sdk_generate_text.restype = ctypes.c_int32

    # Views
    lib.ami_sdk_get_logits_view.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(TensorViewC),
    ]
    lib.ami_sdk_get_logits_view.restype = ctypes.c_int32

    lib.ami_sdk_get_hidden_state_view.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(TensorViewC),
    ]
    lib.ami_sdk_get_hidden_state_view.restype = ctypes.c_int32

    # Embeddings & Reranking
    lib.ami_sdk_embed_tokens.argtypes = [
        ctypes.c_void_p,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.POINTER(EmbedParamsC),
        c_f32_p,
        ctypes.c_size_t,
        c_size_p,
    ]
    lib.ami_sdk_embed_tokens.restype = ctypes.c_int32

    lib.ami_sdk_embed_text.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(EmbedParamsC),
        c_f32_p,
        ctypes.c_size_t,
        c_size_p,
    ]
    lib.ami_sdk_embed_text.restype = ctypes.c_int32

    lib.ami_sdk_rerank_tokens.argtypes = [
        ctypes.c_void_p,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.POINTER(c_u32_p),
        c_size_p,
        ctypes.c_size_t,
        ctypes.POINTER(RerankParamsC),
        c_f32_p,
    ]
    lib.ami_sdk_rerank_tokens.restype = ctypes.c_int32

    lib.ami_sdk_rerank_texts.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
        c_size_p,
        ctypes.c_size_t,
        ctypes.POINTER(RerankParamsC),
        c_f32_p,
    ]
    lib.ami_sdk_rerank_texts.restype = ctypes.c_int32

    # Vision Ingress (G14)
    lib.ami_sdk_vision_from_cvpixelbuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, c_void_pp]
    lib.ami_sdk_vision_from_cvpixelbuffer.restype = ctypes.c_int32
    lib.ami_sdk_vision_from_iosurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p, c_void_pp]
    lib.ami_sdk_vision_from_iosurface.restype = ctypes.c_int32
    lib.ami_sdk_vision_from_iosurface_id.argtypes = [ctypes.c_void_p, ctypes.c_uint32, c_void_pp]
    lib.ami_sdk_vision_from_iosurface_id.restype = ctypes.c_int32
    lib.ami_sdk_vision_create_iosurface.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        c_void_pp,
    ]
    lib.ami_sdk_vision_create_iosurface.restype = ctypes.c_int32
    lib.ami_sdk_vision_load_image.argtypes = [ctypes.c_void_p, ctypes.c_char_p, c_void_pp]
    lib.ami_sdk_vision_load_image.restype = ctypes.c_int32
    lib.ami_sdk_vision_destroy.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_vision_destroy.restype = None
    lib.ami_sdk_vision_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(VisionInfoC)]
    lib.ami_sdk_vision_query.restype = ctypes.c_int32
    lib.ami_sdk_vision_cvpixelbuffer.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_vision_cvpixelbuffer.restype = ctypes.c_void_p
    lib.ami_sdk_vision_iosurface.argtypes = [ctypes.c_void_p]
    lib.ami_sdk_vision_iosurface.restype = ctypes.c_void_p
    lib.ami_sdk_vision_lock_plane.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        c_void_pp,
        c_size_p,
        c_u32_p,
    ]
    lib.ami_sdk_vision_lock_plane.restype = ctypes.c_int32
    lib.ami_sdk_vision_unlock.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.ami_sdk_vision_unlock.restype = ctypes.c_int32
    lib.ami_sdk_prefill_vision.argtypes = [
        ctypes.c_void_p,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        c_u32_p,
        ctypes.c_size_t,
        ctypes.POINTER(VisionParamsC),
        ctypes.POINTER(VisionResultC),
    ]
    lib.ami_sdk_prefill_vision.restype = ctypes.c_int32


def load_lib(lib_path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    """Loads and caches libaminfer.dylib via ctypes.CDLL (releasing the Python GIL)."""
    resolved = find_libaminfer(lib_path)
    cached = _LIB_CACHE.get(resolved)
    if cached is not None:
        return cached

    lib = ctypes.CDLL(resolved)
    _configure_cdll(lib)
    got_abi = int(lib.ami_sdk_abi_version())
    if got_abi != AMI_SDK_ABI_VERSION:
        raise AminferError(
            AMI_SDK_ERR_ABI_MISMATCH,
            "AMI_SDK_ERR_ABI_MISMATCH",
            f"expected ABI version {AMI_SDK_ABI_VERSION}, got {got_abi}",
        )
    _LIB_CACHE[resolved] = lib
    return lib


def check_status(lib: ctypes.CDLL, status: int) -> None:
    """Raises a typed AminferError subclass if status != AMI_SDK_OK."""
    if status == AMI_SDK_OK:
        return
    raw_name = lib.ami_sdk_status_string(status)
    status_name = raw_name.decode("utf-8", errors="replace") if raw_name else "AMI_SDK_ERR_UNKNOWN"
    raw_err = lib.ami_sdk_last_error()
    detail = raw_err.decode("utf-8", errors="replace") if raw_err else ""

    if status in (AMI_SDK_ERR_BUDGET_EXCEEDED, AMI_SDK_ERR_LIVE_RAM_EXCEEDED):
        raise BudgetExceededError(status, status_name, detail)
    if status == AMI_SDK_ERR_CANCELLED:
        raise CancelledError(status, status_name, detail)
    if status == AMI_SDK_ERR_INVALID_ARGUMENT:
        raise InvalidArgumentError(status, status_name, detail)
    if status == AMI_SDK_ERR_SEQUENCE_FULL:
        raise SequenceFullError(status, status_name, detail)
    if status == AMI_SDK_ERR_VISION:
        raise VisionError(status, status_name, detail)
    raise AminferError(status, status_name, detail)
