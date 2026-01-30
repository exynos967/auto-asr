from __future__ import annotations

import difflib
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from auto_asr.subtitle_processing.base import (
    ProcessorContext,
    SubtitleProcessor,
    register_processor,
)
from auto_asr.subtitles import SubtitleLine

logger = logging.getLogger(__name__)


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

        system_prompt = (
            "You are a subtitle proofreader.\n"
            "Correct recognition errors and improve readability with MINIMAL edits.\n"
            "Keep the original language. Do NOT translate.\n"
            "Return ONLY a valid JSON dictionary mapping the SAME keys to corrected text.\n"
            "Do not add or remove keys.\n"
        )
        if custom_prompt:
            system_prompt += f"\nReference:\n{custom_prompt}\n"

        indexed = [(str(i), line.text) for i, line in enumerate(lines, 1)]
        batches = _iter_batches(indexed, batch_size)

        results: dict[str, str] = {}

        def run_batch(payload: dict[str, str]) -> dict[str, str]:
            return ctx.chat_json(system_prompt=system_prompt, payload=payload)

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
