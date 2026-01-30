from __future__ import annotations

import difflib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import json_repair

from auto_asr.subtitle_processing.alignment import SubtitleAligner
from auto_asr.subtitle_processing.base import (
    ProcessorContext,
    SubtitleProcessor,
    register_processor,
)
from auto_asr.subtitle_processing.prompts import get_prompt
from auto_asr.subtitles import SubtitleLine

logger = logging.getLogger(__name__)

MAX_STEPS = 3


def _iter_batches(items: list[tuple[str, str]], batch_size: int) -> list[dict[str, str]]:
    if batch_size <= 0:
        batch_size = 20
    out: list[dict[str, str]] = []
    for i in range(0, len(items), batch_size):
        out.append(dict(items[i : i + batch_size]))
    return out


def _count_words(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    if re.search(r"\s", t):
        return len([p for p in re.split(r"\s+", t) if p])
    return len(t)


def _is_change_too_large(original: str, optimized: str) -> bool:
    orig = re.sub(r"\s+", " ", (original or "")).strip()
    opt = re.sub(r"\s+", " ", (optimized or "")).strip()

    if not orig and not opt:
        return False
    if not orig or not opt:
        return True

    ratio = difflib.SequenceMatcher(None, orig, opt).ratio()
    threshold = 0.3 if _count_words(orig) <= 10 else 0.7
    return ratio < threshold


def _validate_optimization_result(
    *, original_chunk: dict[str, str], optimized_chunk: dict[str, str]
) -> tuple[bool, str]:
    expected_keys = set(original_chunk.keys())
    actual_keys = set(optimized_chunk.keys())

    if expected_keys != actual_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys

        error_parts: list[str] = []
        if missing:
            error_parts.append(f"Missing keys: {sorted(missing)}")
        if extra:
            error_parts.append(f"Extra keys: {sorted(extra)}")
        error_msg = (
            "\n".join(error_parts)
            + f"\nRequired keys: {sorted(expected_keys)}\n"
            f"Please return the COMPLETE optimized dictionary with ALL {len(expected_keys)} keys."
        )
        return False, error_msg

    excessive_changes: list[str] = []
    for key in expected_keys:
        original_text = original_chunk[key]
        optimized_text = optimized_chunk[key]

        original_cleaned = re.sub(r"\s+", " ", original_text).strip()
        optimized_cleaned = re.sub(r"\s+", " ", optimized_text).strip()

        matcher = difflib.SequenceMatcher(None, original_cleaned, optimized_cleaned)
        similarity = matcher.ratio()
        similarity_threshold = 0.3 if _count_words(original_text) <= 10 else 0.7

        if similarity < similarity_threshold:
            excessive_changes.append(
                f"Key '{key}': similarity {similarity:.1%} < {similarity_threshold:.0%}. "
                f"Original: '{original_text}' → Optimized: '{optimized_text}' "
            )

    if excessive_changes:
        error_msg = ";\n".join(excessive_changes)
        error_msg += (
            "\n\nYour optimizations changed the text too much. "
            "Keep high similarity (≥70% for normal text) by making MINIMAL changes: "
            "only fix recognition errors and improve clarity, "
            "but preserve the original wording, length and structure as much as possible."
        )
        return False, error_msg

    return True, ""


def _repair_subtitle(
    *, original: dict[str, str], optimized: dict[str, str]
) -> dict[str, str]:
    try:
        aligner = SubtitleAligner()
        original_list = list(original.values())
        optimized_list = list(optimized.values())
        aligned_source, aligned_target = aligner.align_texts(original_list, optimized_list)
        if len(aligned_source) != len(aligned_target):
            return optimized

        # Keys inside one chunk are consecutive; preserve start index.
        start_id = next(iter(original.keys()))
        return {str(int(start_id) + i): text for i, text in enumerate(aligned_target)}
    except Exception:
        return optimized


def _agent_loop_optimize(
    *,
    ctx: ProcessorContext,
    subtitle_chunk: dict[str, str],
    custom_prompt: str,
) -> dict[str, str]:
    if ctx.chat_text is None:
        raise RuntimeError("ctx.chat_text is required for OptimizeProcessor")

    user_prompt = (
        "Correct the following subtitles. Keep the original language, do not translate:\n"
        f"<input_subtitle>{json.dumps(subtitle_chunk, ensure_ascii=False)}</input_subtitle>"
    )
    if custom_prompt:
        user_prompt += f"\nReference content:\n<reference>{custom_prompt}</reference>"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": get_prompt("optimize/subtitle")},
        {"role": "user", "content": user_prompt},
    ]

    last_result: dict[str, str] | None = None

    for _ in range(MAX_STEPS):
        result_text = str(ctx.chat_text(messages=messages, temperature=0.2) or "")
        if not result_text:
            raise ValueError("LLM returned empty output")

        parsed_result = json_repair.loads(result_text)
        if not isinstance(parsed_result, dict):
            raise ValueError(
                f"LLM returned invalid type: expected dict, got {type(parsed_result).__name__}"
            )

        result_dict = {str(k): str(v) for k, v in parsed_result.items()}
        last_result = result_dict

        ok, error_message = _validate_optimization_result(
            original_chunk=subtitle_chunk, optimized_chunk=result_dict
        )
        if ok:
            return _repair_subtitle(original=subtitle_chunk, optimized=result_dict)

        messages.append({"role": "assistant", "content": result_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Validation failed: {error_message}\n"
                    "Please fix the errors and output ONLY a valid JSON dictionary."
                ),
            }
        )

    return _repair_subtitle(original=subtitle_chunk, optimized=last_result or subtitle_chunk)


@register_processor
class OptimizeProcessor(SubtitleProcessor):
    name = "optimize"

    def process(
        self, lines: list[SubtitleLine], *, ctx: ProcessorContext, options: dict
    ) -> list[SubtitleLine]:
        custom_prompt = str(options.get("custom_prompt") or "").strip()
        batch_size = int(options.get("batch_size") or 20)
        concurrency = int(options.get("concurrency") or 4)
        concurrency = max(1, min(32, concurrency))

        indexed = [(str(i), line.text) for i, line in enumerate(lines, 1)]
        batches = _iter_batches(indexed, batch_size)

        results: dict[str, str] = {}

        def run_batch(payload: dict[str, str]) -> dict[str, str]:
            return _agent_loop_optimize(ctx=ctx, subtitle_chunk=payload, custom_prompt=custom_prompt)

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(run_batch, b) for b in batches]
            for idx, fut in enumerate(futs):
                batch = batches[idx]
                try:
                    results.update({str(k): str(v) for k, v in fut.result().items()})
                except Exception as e:
                    if getattr(e, "status_code", None) in {401, 403}:
                        raise
                    logger.exception("optimize batch failed, fallback to original: %s", e)
                    results.update(batch)

        out: list[SubtitleLine] = []
        for i, line in enumerate(lines, 1):
            candidate = results.get(str(i), line.text)
            if _is_change_too_large(line.text, candidate):
                candidate = line.text
            out.append(SubtitleLine(start_s=line.start_s, end_s=line.end_s, text=candidate))
        return out


__all__ = ["OptimizeProcessor"]
