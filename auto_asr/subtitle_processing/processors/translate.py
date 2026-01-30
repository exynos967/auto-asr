from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import json_repair

from auto_asr.subtitle_processing.base import (
    ProcessorContext,
    SubtitleProcessor,
    register_processor,
)
from auto_asr.subtitle_processing.prompts import get_prompt
from auto_asr.subtitles import SubtitleLine

logger = logging.getLogger(__name__)

MAX_STEPS = 3


def _language_label(code: str) -> str:
    key = (code or "").strip().lower()
    return {
        "zh": "Simplified Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }.get(key, key or "Simplified Chinese")


def _validate_llm_response(
    *, response_obj: object, subtitle_dict: dict[str, str], is_reflect: bool
) -> tuple[bool, str]:
    if not isinstance(response_obj, dict):
        return (
            False,
            f"Output must be a dict, got {type(response_obj).__name__}. Use format: {{'0': 'text', '1': 'text'}}",
        )

    expected_keys = set(subtitle_dict.keys())
    actual_keys = set(str(k) for k in response_obj.keys())

    def _sort_keys(keys: set[str]) -> list[str]:
        return sorted(keys, key=lambda x: int(x) if x.isdigit() else x)

    if expected_keys != actual_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        error_parts: list[str] = []
        if missing:
            error_parts.append(
                f"Missing keys {_sort_keys(missing)} - you must translate these items"
            )
        if extra:
            error_parts.append(
                f"Extra keys {_sort_keys(extra)} - these keys are not in input, remove them"
            )
        return False, "; ".join(error_parts)

    if is_reflect:
        for key in expected_keys:
            value = response_obj.get(key)
            if not isinstance(value, dict):
                return (
                    False,
                    f"Key '{key}': value must be a dict with 'native_translation' field. Got {type(value).__name__}.",
                )
            if "native_translation" not in value:
                available_keys = list(value.keys())
                return (
                    False,
                    f"Key '{key}': missing 'native_translation' field. Found keys: {available_keys}. Must include 'native_translation'.",
                )

    return True, ""


def _agent_loop(
    *,
    ctx: ProcessorContext,
    system_prompt: str,
    subtitle_dict: dict[str, str],
    is_reflect: bool,
) -> dict[str, object]:
    if ctx.chat_text is None:
        raise RuntimeError("ctx.chat_text is required for TranslateProcessor")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(subtitle_dict, ensure_ascii=False)},
    ]

    last_response: dict[str, object] | None = None

    for _ in range(MAX_STEPS):
        content = str(ctx.chat_text(messages=messages) or "").strip()
        response_obj = json_repair.loads(content)
        if isinstance(response_obj, dict):
            last_response = {str(k): v for k, v in response_obj.items()}
        ok, error_message = _validate_llm_response(
            response_obj=response_obj,
            subtitle_dict=subtitle_dict,
            is_reflect=is_reflect,
        )
        if ok and isinstance(response_obj, dict):
            return {str(k): v for k, v in response_obj.items()}

        assistant_content = (
            json.dumps(response_obj, ensure_ascii=False)
            if isinstance(response_obj, (dict, list))
            else str(response_obj)
        )
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Error: {error_message}\n\n"
                    f"Fix the errors above and output ONLY a valid JSON dictionary with ALL {len(subtitle_dict)} keys"
                ),
            }
        )

    if last_response is not None:
        return last_response
    raise RuntimeError("LLM translation failed: empty response")


def _translate_single(
    *, ctx: ProcessorContext, system_prompt: str, text: str
) -> str:
    if ctx.chat_text is None:
        raise RuntimeError("ctx.chat_text is required for TranslateProcessor")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    return str(ctx.chat_text(messages=messages, temperature=0.7) or "").strip()


def _iter_batches(items: list[tuple[str, str]], batch_size: int) -> list[dict[str, str]]:
    if batch_size <= 0:
        batch_size = 20
    out: list[dict[str, str]] = []
    for i in range(0, len(items), batch_size):
        out.append(dict(items[i : i + batch_size]))
    return out


@register_processor
class TranslateProcessor(SubtitleProcessor):
    name = "translate"

    def process(
        self, lines: list[SubtitleLine], *, ctx: ProcessorContext, options: dict
    ) -> list[SubtitleLine]:
        target_language = str(options.get("target_language") or "").strip() or "zh"
        custom_prompt = str(options.get("custom_prompt") or "").strip()
        is_reflect = bool(options.get("reflect"))
        batch_size = int(options.get("batch_size") or 20)
        concurrency = int(options.get("concurrency") or 4)
        concurrency = max(1, min(32, concurrency))

        prompt_path = "translate/reflect" if is_reflect else "translate/standard"
        system_prompt = get_prompt(
            prompt_path,
            target_language=_language_label(target_language),
            custom_prompt=custom_prompt,
        )
        single_prompt = get_prompt(
            "translate/single", target_language=_language_label(target_language)
        )

        indexed = [(str(i), line.text) for i, line in enumerate(lines, 1)]
        batches = _iter_batches(indexed, batch_size)

        results: dict[str, str] = {}

        def run_batch(payload: dict[str, str]) -> dict[str, str]:
            raw = _agent_loop(
                ctx=ctx,
                system_prompt=system_prompt,
                subtitle_dict=payload,
                is_reflect=is_reflect,
            )
            out: dict[str, str] = {}
            for k, v in raw.items():
                if is_reflect and isinstance(v, dict):
                    out[str(k)] = str(v.get("native_translation", v) or "")
                else:
                    out[str(k)] = str(v or "")
            return out

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(run_batch, b) for b in batches]
            for idx, fut in enumerate(futs):
                batch = batches[idx]
                try:
                    results.update({str(k): str(v) for k, v in fut.result().items()})
                except Exception as e:
                    if getattr(e, "status_code", None) in {401, 403}:
                        raise
                    logger.exception(
                        "translate batch failed, fallback to single translator: %s", e
                    )
                    for k, v in batch.items():
                        try:
                            results[str(k)] = _translate_single(
                                ctx=ctx, system_prompt=single_prompt, text=str(v)
                            )
                        except Exception:
                            results[str(k)] = str(v)

        out: list[SubtitleLine] = []
        for i, line in enumerate(lines, 1):
            new_text = results.get(str(i), line.text)
            out.append(SubtitleLine(start_s=line.start_s, end_s=line.end_s, text=new_text))
        return out


__all__ = ["TranslateProcessor"]
