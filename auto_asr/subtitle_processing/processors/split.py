from __future__ import annotations

import difflib
import math
import re
from concurrent.futures import ThreadPoolExecutor

from auto_asr.subtitle_processing.prompts import get_prompt
from auto_asr.subtitle_processing.base import (
    ProcessorContext,
    SubtitleProcessor,
    register_processor,
)
from auto_asr.subtitles import SubtitleLine

# ==================== VideoCaptioner-compatible validation ====================

_NO_SPACE_LANGUAGES = r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u0900-\u0dff]"


def _is_mainly_cjk(text: str, threshold: float = 0.5) -> bool:
    if not text:
        return False

    no_space_count = len(re.findall(_NO_SPACE_LANGUAGES, text))
    total_chars = len("".join(text.split()))
    return no_space_count / total_chars > threshold if total_chars > 0 else False


def _count_words(text: str) -> int:
    if not text:
        return 0

    char_count = len(re.findall(_NO_SPACE_LANGUAGES, text))
    word_text = re.sub(_NO_SPACE_LANGUAGES, " ", text)
    word_count = len(word_text.strip().split())
    return char_count + word_count


MAX_STEPS = 2


def _validate_split_result(
    *,
    original_text: str,
    split_result: list[str],
    max_word_count_cjk: int,
    max_word_count_english: int,
) -> tuple[bool, str]:
    if not split_result:
        return False, "No segments found. Split the text with <br> tags."

    original_cleaned = re.sub(r"\s+", " ", original_text)
    text_is_cjk = _is_mainly_cjk(original_cleaned)

    merged_char = "" if text_is_cjk else " "
    merged = merged_char.join(split_result)
    merged_cleaned = re.sub(r"\s+", " ", merged)

    matcher = difflib.SequenceMatcher(None, original_cleaned, merged_cleaned)
    similarity_ratio = matcher.ratio()
    if similarity_ratio < 0.96:
        return (
            False,
            f"Content modified (similarity: {similarity_ratio:.1%}). "
            "Keep original text unchanged, only insert <br> between words.",
        )

    violations: list[str] = []
    for i, segment in enumerate(split_result, 1):
        word_count = _count_words(segment)
        max_allowed = max_word_count_cjk if text_is_cjk else max_word_count_english
        if word_count > max_allowed:
            segment_preview = segment[:40] + "..." if len(segment) > 40 else segment
            violations.append(
                f"Segment {i} '{segment_preview}': {word_count} "
                f"{'chars' if text_is_cjk else 'words'} > {max_allowed} limit"
            )

    if violations:
        error_msg = "Length violations:\n" + "\n".join(f"- {v}" for v in violations)
        error_msg += (
            "\n\nSplit these long segments further with <br>, then output the COMPLETE text with ALL segments (not just the fixed ones)."
        )
        return False, error_msg

    return True, ""


def _split_with_agent_loop(
    *,
    ctx: ProcessorContext,
    text: str,
    prompt_path: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    delimiter: str = "<br>",
) -> list[str]:
    if ctx.chat_text is None:
        raise RuntimeError("ctx.chat_text is required for SplitProcessor")

    system_prompt = get_prompt(
        prompt_path,
        max_word_count_cjk=max_word_count_cjk,
        max_word_count_english=max_word_count_english,
    )
    user_prompt = (
        f"Please use multiple {delimiter} tags to separate the following sentence:\n{text}"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_result: list[str] | None = None

    for _step in range(MAX_STEPS):
        result_text = str(ctx.chat_text(messages=messages, temperature=0.1) or "")
        result_text_cleaned = re.sub(r"\n+", "", result_text)
        split_result = [
            segment.strip()
            for segment in result_text_cleaned.split(delimiter)
            if segment.strip()
        ]

        last_result = split_result if split_result else [text]

        ok, error_message = _validate_split_result(
            original_text=text,
            split_result=split_result,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
        )
        if ok:
            return split_result

        messages.append({"role": "assistant", "content": result_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Error: {error_message}\n"
                    "Fix the errors above and output the COMPLETE corrected text with <br> tags "
                    "(include ALL segments, not just the fixed ones), no explanation."
                ),
            }
        )

    return [text]


def split_text_by_delimiter(text: str, delimiter: str = "<br>") -> list[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in raw.split(delimiter)]
    return [p for p in parts if p]


def _segment_weight(text: str) -> int:
    # Use a simple character-based weight to allocate time.
    cleaned = "".join((text or "").split())
    return max(1, len(cleaned))


def split_line_to_cues(line: SubtitleLine, parts: list[str]) -> list[SubtitleLine]:
    """Split one cue into multiple cues and allocate timestamps proportionally."""
    if not parts:
        return [line]

    start_s = float(line.start_s)
    end_s = float(line.end_s)
    if end_s <= start_s:
        end_s = start_s + 0.001

    total_ms = max(1, round((end_s - start_s) * 1000.0))
    weights = [_segment_weight(p) for p in parts]
    total_w = sum(weights)

    out: list[SubtitleLine] = []
    cur_ms = 0
    for i, txt in enumerate(parts):
        w = weights[i]
        next_ms = total_ms if i == len(parts) - 1 else cur_ms + math.floor(total_ms * (w / total_w))

        seg_start_s = start_s + (cur_ms / 1000.0)
        seg_end_s = start_s + (next_ms / 1000.0)
        if seg_end_s <= seg_start_s:
            seg_end_s = seg_start_s + 0.001

        out.append(SubtitleLine(start_s=seg_start_s, end_s=seg_end_s, text=txt))
        cur_ms = next_ms

    # Ensure last end matches original end (avoid rounding drift).
    last = out[-1]
    out[-1] = SubtitleLine(start_s=last.start_s, end_s=end_s, text=last.text)
    return out


def _merge_parts_like_original(original: str, parts: list[str]) -> str:
    if re.search(r"\s", original or ""):
        return " ".join(parts)
    return "".join(parts)


@register_processor
class SplitProcessor(SubtitleProcessor):
    name = "split"

    def process(
        self, lines: list[SubtitleLine], *, ctx: ProcessorContext, options: dict
    ) -> list[SubtitleLine]:
        strategy = str(options.get("strategy") or "").strip() or "semantic"
        if strategy not in {"semantic", "sentence"}:
            strategy = "semantic"

        mode = str(options.get("mode") or "").strip() or "inplace_newlines"
        if mode not in {"inplace_newlines", "split_to_cues"}:
            mode = "inplace_newlines"

        delimiter = str(options.get("delimiter") or "").strip() or "<br>"
        concurrency = int(options.get("concurrency") or 4)
        concurrency = max(1, min(32, concurrency))

        max_word_count_cjk = int(options.get("max_word_count_cjk") or 18)
        max_word_count_english = int(options.get("max_word_count_english") or 12)
        max_word_count_cjk = max(1, min(200, max_word_count_cjk))
        max_word_count_english = max(1, min(200, max_word_count_english))

        prompt_path = "split/sentence" if strategy == "sentence" else "split/semantic"

        indexed = [(str(i), line) for i, line in enumerate(lines, 1)]

        def split_one(key: str, line: SubtitleLine) -> tuple[str, list[str]]:
            try:
                parts = _split_with_agent_loop(
                    ctx=ctx,
                    text=line.text,
                    prompt_path=prompt_path,
                    max_word_count_cjk=max_word_count_cjk,
                    max_word_count_english=max_word_count_english,
                    delimiter=delimiter,
                )
                return key, parts
            except Exception as e:
                if getattr(e, "status_code", None) in {401, 403}:
                    raise
                return key, [line.text]

        parts_map: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(split_one, k, line) for k, line in indexed]
            for idx, fut in enumerate(futs):
                k, line = indexed[idx]
                try:
                    key, parts = fut.result()
                except Exception as e:
                    if getattr(e, "status_code", None) in {401, 403}:
                        raise
                    key, parts = k, [line.text]
                parts_map[key] = parts

        out_lines: list[SubtitleLine] = []
        for i, line in enumerate(lines, 1):
            parts = parts_map.get(str(i), [line.text])
            if mode == "split_to_cues" and len(parts) > 1:
                out_lines.extend(split_line_to_cues(line, parts))
                continue

            merged = "\n".join(p.strip() for p in parts if p.strip())
            out_lines.append(SubtitleLine(start_s=line.start_s, end_s=line.end_s, text=merged))

        return out_lines


__all__ = ["SplitProcessor", "split_line_to_cues", "split_text_by_delimiter"]
