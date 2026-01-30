from __future__ import annotations

import logging
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


@register_processor
class TranslateProcessor(SubtitleProcessor):
    name = "translate"

    def process(
        self, lines: list[SubtitleLine], *, ctx: ProcessorContext, options: dict
    ) -> list[SubtitleLine]:
        target_language = str(options.get("target_language") or "").strip() or "zh"
        custom_prompt = str(options.get("custom_prompt") or "").strip()
        batch_size = int(options.get("batch_size") or 20)
        concurrency = int(options.get("concurrency") or 4)
        concurrency = max(1, min(32, concurrency))

        system_prompt = (
            "You are a professional subtitle translator.\n"
            f"Translate each value to {target_language}.\n"
            "Return ONLY a valid JSON dictionary mapping the SAME keys to translated text.\n"
            "Do not add or remove keys.\n"
            "Keep formatting and line breaks when appropriate.\n"
        )
        if custom_prompt:
            system_prompt += f"\nAdditional requirements:\n{custom_prompt}\n"

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
                    logger.exception("translate batch failed, fallback to original: %s", e)
                    results.update(batch)

        out: list[SubtitleLine] = []
        for i, line in enumerate(lines, 1):
            new_text = results.get(str(i), line.text)
            out.append(SubtitleLine(start_s=line.start_s, end_s=line.end_s, text=new_text))
        return out


__all__ = ["TranslateProcessor"]
