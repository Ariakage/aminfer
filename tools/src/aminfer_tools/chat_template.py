"""Offline chat-template compiler (W24) from Hugging Face templates into declarative JSON.

Compiles Hugging Face `chat_template.jinja` / `tokenizer_config.json` (+ optional
`config.json` hints) into a compact, deterministic `chat_template_spec` v1 JSON
object suitable for embedding into `.amf` metadata (`__meta.chat_template_json`,
legacy alias `__aminfer_chat_template__`) and zero-allocation parsing in Zig.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

CHAT_TEMPLATE_SCHEMA_VERSION = 1

SUPPORTED_DIALECTS = ("chatml", "deepseek", "inkling2", "hy4", "custom")


def _extract_token_str(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw if raw else None
    if isinstance(raw, dict):
        content = raw.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def _resolve_hy4_suffix(template_text: str) -> str:
    m = re.search(r"set\s+HYTK\s*=\s*['\"]([^'\"]*)['\"]", template_text)
    if m:
        return m.group(1)
    return ":opensource"


_JINJA_ESCAPE_MAP: dict[str, str] = {
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
}

_JINJA_ESCAPE_RE = re.compile(
    r"\\(U[0-9a-fA-F]{8}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|[0-7]{1,3}|.)"
)


def _unescape_jinja_literal(raw: str) -> str:
    """Decode backslash escapes in a Jinja string literal while preserving UTF-8 characters."""
    if "\\" not in raw:
        return raw

    def _replace(match: re.Match[str]) -> str:
        seq = match.group(1)
        if seq in _JINJA_ESCAPE_MAP:
            return _JINJA_ESCAPE_MAP[seq]
        if seq[0] in ("x", "u", "U"):
            return chr(int(seq[1:], 16))
        if "0" <= seq[0] <= "7":
            return chr(int(seq, 8))
        return "\\" + seq

    return _JINJA_ESCAPE_RE.sub(_replace, raw)


def _resolve_jinja_string_vars(template_text: str) -> dict[str, str]:
    """Extract simple `{-% set name = '...' -%}` string/format variables."""
    hytk = _resolve_hy4_suffix(template_text)
    vars_map: dict[str, str] = {"HYTK": hytk}
    for m in re.finditer(
        r"\{%-?\s*set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])((?:\\.|(?!\2).)*)\2(?:\.format\(([A-Za-z_][A-Za-z0-9_]*)\))?\s*-?%\}",
        template_text,
    ):
        var_name = m.group(1)
        literal = _unescape_jinja_literal(m.group(3))
        fmt_arg = m.group(4)
        if fmt_arg is not None:
            arg_val = vars_map.get(fmt_arg, hytk if fmt_arg == "HYTK" else "")
            literal = literal.replace("{}", arg_val)
        vars_map[var_name] = literal
    return vars_map


def _detect_dialect(
    template_text: str,
    tokenizer_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> str:
    """Identify the canonical chat template dialect deterministically."""
    if "<|message_model|>" in template_text or "<|content_invoke_tool_json|>" in template_text:
        return "inkling2"
    if "<｜hy_start" in template_text or "<｜reasoning_mode" in template_text or "HYTK" in template_text:
        return "hy4"

    eos = _extract_token_str(tokenizer_cfg.get("eos_token")) or ""
    bos = _extract_token_str(tokenizer_cfg.get("bos_token")) or ""
    add_special = tokenizer_cfg.get("additional_special_tokens") or []
    special_blob = " ".join(str(x) for x in add_special if isinstance(x, str))

    model_type = str(model_cfg.get("model_type") or "").lower()
    if "inkling" in model_type or eos == "<|content_model_end_sampling|>":
        return "inkling2"
    if "hunyuan" in model_type or "hy4" in model_type or "<｜hy_start" in bos or "<｜hy_end" in eos:
        return "hy4"

    if "<|im_start|>" in template_text or "<|im_end|>" in template_text:
        return "chatml"
    if "<|im_end|>" in (eos, special_blob) or "<|im_start|>" in special_blob or "qwen" in model_type:
        return "chatml"

    if (
        "<｜begin▁of▁sentence｜>" in template_text
        or "<｜User｜>" in template_text
        or "<｜Assistant｜>" in template_text
        or "<｜begin▁of▁sentence｜>" in (bos, eos)
        or "deepseek" in model_type
    ):
        return "deepseek"

    return "custom"


def _compile_inkling2(
    template_text: str,
    tokenizer_cfg: dict[str, Any],
) -> dict[str, Any]:
    effort_map: dict[str, float] = {
        "none": 0.0,
        "minimal": 0.1,
        "low": 0.2,
        "medium": 0.7,
        "high": 0.9,
        "xhigh": 0.99,
        "max": 0.99,
    }
    m_effort = re.search(r"effort_map\s*=\s*(\{[^}]+\})", template_text)
    if m_effort:
        try:
            parsed = json.loads(m_effort.group(1))
            if isinstance(parsed, dict) and parsed:
                effort_map = {str(k): float(v) for k, v in parsed.items()}
        except (ValueError, TypeError):
            pass

    default_effort = "high"
    m_def = re.search(r"else\s+([0-9]+(?:\.[0-9]+)?)\b", template_text)
    if m_def:
        def_num = float(m_def.group(1))
        for k, v in effort_map.items():
            if abs(v - def_num) < 1e-6:
                default_effort = k
                break

    bos = _extract_token_str(tokenizer_cfg.get("bos_token"))
    eos = _extract_token_str(tokenizer_cfg.get("eos_token")) or "<|content_model_end_sampling|>"
    img_tok = "<|content_image|><|unused_200054|><|end_message|>"
    m_img = re.search(r"<\|content_image\|><\|unused_[0-9]+\|><\|end_message\|>", template_text)
    if m_img:
        img_tok = m_img.group(0)

    return {
        "schema_version": CHAT_TEMPLATE_SCHEMA_VERSION,
        "dialect": "inkling2",
        "bos_token": bos,
        "eos_token": eos,
        "add_generation_prompt": True,
        "role_tokens": {
            "system_prefix": "<|message_system|><|content_text|>",
            "system_suffix": "<|end_message|>",
            "user_prefix": "<|message_user|><|content_text|>",
            "user_suffix": "<|end_message|>",
            "assistant_prefix": "<|message_model|><|content_text|>",
            "assistant_suffix": f"<|end_message|>{eos}",
            "tool_prefix": "<|message_tool|><|content_text|>",
            "tool_suffix": "<|end_message|>",
            "content_text_tag": "<|content_text|>",
            "end_message_tag": "<|end_message|>",
            "header_role_tokens": {
                "system": "<|message_system|>",
                "user": "<|message_user|>",
                "assistant": "<|message_model|>",
                "tool": "<|message_tool|>",
            },
        },
        "thinking": {
            "supported": True,
            "open_tag": "<|content_thinking|>",
            "close_tag": "<|end_message|>",
            "prefill_think": "",
            "prefill_no_think": "",
            "effort_mode": "inkling2_effort_directive",
            "default_effort": default_effort,
            "effort_map": effort_map,
            "effort_directive_prefix": "<|message_system|><|content_text|>Thinking effort level: ",
            "effort_directive_suffix": "<|end_message|>",
            "preserved_thinking": False,
            "reasoning_mode_token": None,
        },
        "tool_calling": {
            "format": "inkling2_xml_json",
            "declare_prefix": "<|message_system|>tool_declare<|content_xml|>",
            "declare_suffix": "<|end_message|>",
            "call_open": "<|content_invoke_tool_json|>",
            "call_close": "<|end_message|>",
            "response_open": "<|message_tool|>",
            "response_close": "<|end_message|>",
        },
        "multimodal": {
            "image_tag": img_tok,
            "video_tag": None,
        },
    }


def _compile_hy4(
    template_text: str,
    tokenizer_cfg: dict[str, Any],
) -> dict[str, Any]:
    vars_map = _resolve_jinja_string_vars(template_text)
    hytk = vars_map.get("HYTK", ":opensource")
    hy_start = vars_map.get("hy_start_token", f"<｜hy_start{hytk}｜>")
    hy_middle = vars_map.get("hy_middle_token", f"<｜hy_middle{hytk}｜>")
    hy_end = vars_map.get("hy_end_token", f"<｜hy_end{hytk}｜>")
    think_begin = vars_map.get("think_begin_token", f"<think{hytk}>")
    think_end = vars_map.get("think_end_token", f"</think{hytk}>")
    toolcalls_begin = vars_map.get("toolcalls_begin_token", f"<tool_calls{hytk}>")
    toolcalls_end = vars_map.get("toolcalls_end_token", f"</tool_calls{hytk}>")
    toolcall_begin = vars_map.get("toolcall_begin_token", f"<tool_call{hytk}>")
    toolcall_end = vars_map.get("toolcall_end_token", f"</tool_call{hytk}>")
    argkey_begin = vars_map.get("argkey_begin_token", f"<arg_key{hytk}>")
    argkey_end = vars_map.get("argkey_end_token", f"</arg_key{hytk}>")
    argvalue_begin = vars_map.get("argvalue_begin_token", f"<arg_value{hytk}>")
    argvalue_end = vars_map.get("argvalue_end_token", f"</arg_value{hytk}>")
    toolresp_begin = vars_map.get("toolresponse_begin_token", f"<tool_response{hytk}>")
    toolresp_end = vars_map.get("toolresponse_end_token", f"</tool_response{hytk}>")
    reasoning_mode_token = vars_map.get("reasoning_mode_token", f"<｜reasoning_mode{hytk}｜>")

    has_image = "image_start_token" in vars_map or "hy_image" in template_text
    has_video = "video_start_token" in vars_map or "hy_video" in template_text
    image_tag = (
        (
            vars_map.get("image_start_token", f"<｜hy_image▁start{hytk}｜>")
            + vars_map.get("image_pad_token", f"<｜hy_image▁pad{hytk}｜>")
            + vars_map.get("image_end_token", f"<｜hy_image▁end{hytk}｜>")
        )
        if has_image
        else None
    )
    video_tag = (
        (
            vars_map.get("video_start_token", f"<｜hy_video▁start{hytk}｜>")
            + vars_map.get("video_pad_token", f"<｜hy_video▁pad{hytk}｜>")
            + vars_map.get("video_end_token", f"<｜hy_video▁end{hytk}｜>")
        )
        if has_video
        else None
    )

    bos = _extract_token_str(tokenizer_cfg.get("bos_token")) or hy_start
    eos = _extract_token_str(tokenizer_cfg.get("eos_token")) or hy_end

    declare_prefix = (
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
    )
    declare_suffix = (
        f"\n</tools>\n\nFor function call returns, you should first print {toolcalls_begin}"
        f"\nFor each function call, you should return object like:\n"
        f"{toolcall_begin}{{function-name}}"
        f"{argkey_begin}{{arg-key-1}}{argkey_end}"
        f"{argvalue_begin}{{arg-value-1}}{argvalue_end}..."
        f"{toolcall_end}"
        f"\nAt the end of function call returns, you should print {toolcalls_end}"
    )

    return {
        "schema_version": CHAT_TEMPLATE_SCHEMA_VERSION,
        "dialect": "hy4",
        "bos_token": bos,
        "eos_token": eos,
        "add_generation_prompt": True,
        "role_tokens": {
            "system_prefix": f"{hy_start}system{hy_middle}",
            "system_suffix": hy_end,
            "user_prefix": f"{hy_start}user{hy_middle}",
            "user_suffix": hy_end,
            "assistant_prefix": f"{hy_start}assistant{hy_middle}",
            "assistant_suffix": hy_end,
            "tool_prefix": f"{hy_start}tool{hy_middle}",
            "tool_suffix": hy_end,
            "hy4_start": hy_start,
            "hy4_middle": hy_middle,
            "hy4_end": hy_end,
            "header_role_tokens": {
                "system": "system",
                "user": "user",
                "assistant": "assistant",
                "tool": "tool",
            },
        },
        "thinking": {
            "supported": True,
            "open_tag": think_begin,
            "close_tag": think_end,
            "prefill_think": think_begin,
            "prefill_no_think": f"{think_begin}{think_end}",
            "effort_mode": "hy4_reasoning_mode",
            "default_effort": "high",
            "effort_map": {"none": 0.0, "high": 1.0},
            "preserved_thinking": "preserved_thinking = true" in template_text,
            "reasoning_mode_token": reasoning_mode_token,
        },
        "tool_calling": {
            "format": "hy4_xml_arg",
            "declare_prefix": declare_prefix,
            "declare_suffix": declare_suffix,
            "call_open": toolcalls_begin,
            "call_close": toolcalls_end,
            "response_open": toolresp_begin,
            "response_close": toolresp_end,
        },
        "multimodal": {
            "image_tag": image_tag,
            "video_tag": video_tag,
        },
    }


def _compile_chatml(
    template_text: str,
    tokenizer_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    bos = _extract_token_str(tokenizer_cfg.get("bos_token"))
    eos = _extract_token_str(tokenizer_cfg.get("eos_token")) or "<|im_end|>"
    resp_tpl = tokenizer_cfg.get("response_template")
    fields = resp_tpl.get("fields") if isinstance(resp_tpl, dict) else {}
    if not isinstance(fields, dict):
        fields = {}

    reason_cfg = fields.get("reasoning_content")
    think_open = "<think>"
    think_close = "</think>"
    has_think = "<think>" in template_text or isinstance(reason_cfg, dict) or "qwen" in str(model_cfg.get("model_type") or "").lower()
    if isinstance(reason_cfg, dict) and isinstance(reason_cfg.get("open"), str):
        think_open = reason_cfg["open"]
        cp = str(reason_cfg.get("close_pattern") or "</think>")
        think_close = cp.split("|", 1)[0] or "</think>"

    enable_thinking_mode = "enable_thinking" in template_text or has_think
    prefill_no_think = f"{think_open}\n\n{think_close}\n\n" if "enable_thinking" in template_text else f"{think_open}{think_close}"
    prefill_think = f"{think_open}\n" if "enable_thinking" in template_text else think_open

    extra_special = tokenizer_cfg.get("extra_special_tokens") or {}
    add_special = set(tokenizer_cfg.get("additional_special_tokens") or [])
    vis_bos = extra_special.get("vision_bos_token") or ("<|vision_start|>" if "<|vision_start|>" in add_special else "")
    vis_eos = extra_special.get("vision_eos_token") or ("<|vision_end|>" if "<|vision_end|>" in add_special else "")
    img_tok_raw = extra_special.get("image_token") or ("<|image_pad|>" if "<|image_pad|>" in add_special else None)
    vid_tok_raw = extra_special.get("video_token") or ("<|video_pad|>" if "<|video_pad|>" in add_special else None)
    image_tag = f"{vis_bos}{img_tok_raw}{vis_eos}" if img_tok_raw else None
    video_tag = f"{vis_bos}{vid_tok_raw}{vis_eos}" if vid_tok_raw else None

    return {
        "schema_version": CHAT_TEMPLATE_SCHEMA_VERSION,
        "dialect": "chatml",
        "bos_token": bos,
        "eos_token": eos,
        "add_generation_prompt": True,
        "role_tokens": {
            "system_prefix": "<|im_start|>system\n",
            "system_suffix": "<|im_end|>\n",
            "user_prefix": "<|im_start|>user\n",
            "user_suffix": "<|im_end|>\n",
            "assistant_prefix": "<|im_start|>assistant\n",
            "assistant_suffix": "<|im_end|>\n",
            "tool_prefix": "<|im_start|>user\n<tool_response>\n",
            "tool_suffix": "\n</tool_response><|im_end|>\n",
        },
        "thinking": {
            "supported": has_think,
            "open_tag": think_open,
            "close_tag": think_close,
            "prefill_think": prefill_think if has_think else "",
            "prefill_no_think": prefill_no_think if has_think else "",
            "effort_mode": "boolean_think_prefill" if enable_thinking_mode else "none",
            "default_effort": "high" if has_think else "none",
            "effort_map": {"none": 0.0, "high": 1.0} if has_think else {},
            "preserved_thinking": False,
            "reasoning_mode_token": None,
        },
        "tool_calling": {
            "format": "qwen_xml_json",
            "declare_prefix": "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
            "declare_suffix": "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
            "call_open": "<tool_call>",
            "call_close": "</tool_call>",
            "response_open": "<tool_response>",
            "response_close": "</tool_response>",
        },
        "multimodal": {
            "image_tag": image_tag,
            "video_tag": video_tag,
        },
    }


def _compile_deepseek(
    template_text: str,
    tokenizer_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    bos = _extract_token_str(tokenizer_cfg.get("bos_token")) or "<｜begin▁of▁sentence｜>"
    eos = _extract_token_str(tokenizer_cfg.get("eos_token")) or "<｜end▁of▁sentence｜>"
    think_open = "<think>"
    think_close = "</think>"
    if "<｜thought｜>" in template_text:
        think_open = "<｜thought｜>"
        think_close = "</｜thought｜>"

    extra_special = tokenizer_cfg.get("extra_special_tokens") or {}
    img_tag = extra_special.get("image_token")
    if img_tag is None and "image_token_id" in model_cfg:
        img_tag = "<image>"

    return {
        "schema_version": CHAT_TEMPLATE_SCHEMA_VERSION,
        "dialect": "deepseek",
        "bos_token": bos,
        "eos_token": eos,
        "add_generation_prompt": True,
        "role_tokens": {
            "system_prefix": "",
            "system_suffix": "\n\n",
            "user_prefix": "<｜User｜>",
            "user_suffix": "",
            "assistant_prefix": "<｜Assistant｜>",
            "assistant_suffix": eos,
            "tool_prefix": "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>",
            "tool_suffix": "<｜tool▁output▁end｜><｜tool▁outputs▁end｜>",
        },
        "thinking": {
            "supported": True,
            "open_tag": think_open,
            "close_tag": think_close,
            "prefill_think": f"{think_open}\n",
            "prefill_no_think": f"{think_open}\n{think_close}\n",
            "effort_mode": "boolean_think_prefill",
            "default_effort": "high",
            "effort_map": {"none": 0.0, "high": 1.0},
            "preserved_thinking": False,
            "reasoning_mode_token": None,
        },
        "tool_calling": {
            "format": "deepseek",
            "declare_prefix": "",
            "declare_suffix": "",
            "call_open": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
            "call_close": "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            "response_open": "<｜tool▁output▁begin｜>",
            "response_close": "<｜tool▁output▁end｜>",
        },
        "multimodal": {
            "image_tag": img_tag,
            "video_tag": extra_special.get("video_token"),
        },
    }


def _compile_custom(
    template_text: str,
    tokenizer_cfg: dict[str, Any],
) -> dict[str, Any]:
    bos = _extract_token_str(tokenizer_cfg.get("bos_token"))
    eos = _extract_token_str(tokenizer_cfg.get("eos_token")) or "</s>"
    think_open = "<think>"
    think_close = "</think>"
    has_think = think_open in template_text and think_close in template_text
    return {
        "schema_version": CHAT_TEMPLATE_SCHEMA_VERSION,
        "dialect": "custom",
        "bos_token": bos,
        "eos_token": eos,
        "add_generation_prompt": True,
        "role_tokens": {
            "system_prefix": "<|system|>\n",
            "system_suffix": f"{eos}\n",
            "user_prefix": "<|user|>\n",
            "user_suffix": f"{eos}\n",
            "assistant_prefix": "<|assistant|>\n",
            "assistant_suffix": f"{eos}\n",
            "tool_prefix": "<|tool|>\n",
            "tool_suffix": f"{eos}\n",
        },
        "thinking": {
            "supported": has_think,
            "open_tag": think_open,
            "close_tag": think_close,
            "prefill_think": think_open if has_think else "",
            "prefill_no_think": f"{think_open}{think_close}" if has_think else "",
            "effort_mode": "boolean_think_prefill" if has_think else "none",
            "default_effort": "high" if has_think else "none",
            "effort_map": {"none": 0.0, "high": 1.0} if has_think else {},
            "preserved_thinking": False,
            "reasoning_mode_token": None,
        },
        "tool_calling": {
            "format": "custom",
            "declare_prefix": "",
            "declare_suffix": "",
            "call_open": "<tool_call>",
            "call_close": "</tool_call>",
            "response_open": "<tool_response>",
            "response_close": "</tool_response>",
        },
        "multimodal": {
            "image_tag": None,
            "video_tag": None,
        },
    }


def compile_chat_template_spec(
    *,
    jinja_text: str | None = None,
    tokenizer_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    dialect_override: str | None = None,
) -> dict[str, Any]:
    """Compile HF `chat_template.jinja` / `tokenizer_config.json` into declarative `chat_template_spec`."""
    tok_cfg = tokenizer_config if isinstance(tokenizer_config, dict) else {}
    mod_cfg = model_config if isinstance(model_config, dict) else {}
    raw_tpl = jinja_text
    if not raw_tpl:
        ct = tok_cfg.get("chat_template")
        if isinstance(ct, str):
            raw_tpl = ct
        elif isinstance(ct, list) and ct:
            # HF list-of-{name, template} form: prefer "default" or first entry
            chosen = next((item.get("template") for item in ct if isinstance(item, dict) and item.get("name") == "default"), None)
            if not isinstance(chosen, str) and isinstance(ct[0], dict):
                chosen = ct[0].get("template")
            raw_tpl = chosen if isinstance(chosen, str) else ""
        else:
            raw_tpl = ""

    dialect = dialect_override or _detect_dialect(raw_tpl, tok_cfg, mod_cfg)
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(f"unsupported chat template dialect {dialect!r}; expected one of {SUPPORTED_DIALECTS}")

    if dialect == "inkling2":
        return _compile_inkling2(raw_tpl, tok_cfg)
    if dialect == "hy4":
        return _compile_hy4(raw_tpl, tok_cfg)
    if dialect == "chatml":
        return _compile_chatml(raw_tpl, tok_cfg, mod_cfg)
    if dialect == "deepseek":
        return _compile_deepseek(raw_tpl, tok_cfg, mod_cfg)
    return _compile_custom(raw_tpl, tok_cfg)


def compile_chat_template_from_dir(
    model_dir_or_file: pathlib.Path,
    *,
    tokenizer_config_path: pathlib.Path | None = None,
    config_path: pathlib.Path | None = None,
    chat_template_path: pathlib.Path | None = None,
) -> dict[str, Any] | None:
    """Compile `chat_template_spec` from a Hugging Face checkpoint directory or explicit template file.

    Returns `None` only when none of `chat_template.jinja`, `tokenizer_config.json`, or `config.json`
    exist on disk.
    """
    target = pathlib.Path(model_dir_or_file)
    base_dir = target if target.is_dir() else target.parent

    jinja_p = pathlib.Path(chat_template_path) if chat_template_path is not None else base_dir / "chat_template.jinja"
    tok_cfg_p = pathlib.Path(tokenizer_config_path) if tokenizer_config_path is not None else base_dir / "tokenizer_config.json"
    cfg_p = pathlib.Path(config_path) if config_path is not None else base_dir / "config.json"

    jinja_text: str | None = None
    if jinja_p.is_file():
        jinja_text = jinja_p.read_text(encoding="utf-8")
    elif chat_template_path is not None:
        raise FileNotFoundError(f"chat_template.jinja not found: {jinja_p}")

    tok_cfg: dict[str, Any] = {}
    if tok_cfg_p.is_file():
        loaded = json.loads(tok_cfg_p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            tok_cfg = loaded
    elif tokenizer_config_path is not None:
        raise FileNotFoundError(f"tokenizer_config.json not found: {tok_cfg_p}")

    mod_cfg: dict[str, Any] = {}
    if cfg_p.is_file():
        loaded = json.loads(cfg_p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            mod_cfg = loaded

    if jinja_text is None and not tok_cfg and not mod_cfg:
        return None

    return compile_chat_template_spec(
        jinja_text=jinja_text,
        tokenizer_config=tok_cfg,
        model_config=mod_cfg,
    )


def compile_chat_template_bytes(
    model_dir_or_file: pathlib.Path,
    **kwargs: Any,
) -> bytes | None:
    """Serialize compiled `chat_template_spec` deterministically to compact UTF-8 JSON bytes."""
    spec = compile_chat_template_from_dir(model_dir_or_file, **kwargs)
    if spec is None:
        return None
    return json.dumps(spec, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")
