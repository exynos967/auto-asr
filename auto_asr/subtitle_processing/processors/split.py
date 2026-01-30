from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor

from auto_asr.subtitle_processing.base import (
    ProcessorContext,
    SubtitleProcessor,
    register_processor,
)
from auto_asr.subtitles import SubtitleLine


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

        if strategy == "sentence":
            strategy_instruction = "Split at natural sentence boundaries (punctuation/pauses)."
            strategy_hint = "sentence boundaries"
        else:
            strategy_instruction = (
                "Split at semantic boundaries for readability (you may split within a sentence)."
            )
            strategy_hint = "semantic boundaries"

        system_prompt = (
            "You are a professional subtitle splitter.\n"
            f"{strategy_instruction}\n"
            f"Strategy: {strategy_hint}\n"
            f"Insert `{delimiter}` into the text to split it.\n"
            "Keep the original text unchanged except inserting the delimiter.\n"
            "Return ONLY a valid JSON dictionary mapping the SAME keys to the processed text.\n"
            "Do not add or remove keys.\n"
        )

        indexed = [(str(i), line) for i, line in enumerate(lines, 1)]

        def split_one(key: str, line: SubtitleLine) -> tuple[str, list[str]]:
            payload = {key: line.text}
            out = ctx.chat_json(system_prompt=system_prompt, payload=payload)
            candidate = str(out.get(key, line.text) or line.text)
            parts = split_text_by_delimiter(candidate, delimiter=delimiter)
            if not parts:
                return key, [line.text]

            merged = _merge_parts_like_original(line.text, parts)
            orig_norm = re.sub(r"\s+", " ", line.text).strip()
            merged_norm = re.sub(r"\s+", " ", merged).strip()
            if orig_norm and merged_norm and orig_norm != merged_norm:
                # Content changed; fallback to original.
                return key, [line.text]

            return key, parts

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
