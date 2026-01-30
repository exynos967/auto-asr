from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from openai import OpenAI

from auto_asr.llm.client import call_chat_json_agent_loop, normalize_base_url
from auto_asr.subtitle_io import load_subtitle_file
from auto_asr.subtitle_processing.base import ProcessorContext, get_processor
from auto_asr.subtitles import SubtitleLine, compose_srt, compose_vtt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubtitleProcessingResult:
    out_path: str
    preview_text: str
    debug: str


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_openai_chat_json(
    *,
    api_key: str,
    base_url: str | None,
    llm_model: str,
    llm_temperature: float = 0.2,
) -> Callable[..., dict[str, str]]:
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("请在 Web UI 中填写 OpenAI API Key。")

    base_url_norm = None
    if base_url and base_url.strip():
        base_url_norm = normalize_base_url(base_url)

    client = (
        OpenAI(api_key=api_key, base_url=base_url_norm)
        if base_url_norm
        else OpenAI(api_key=api_key)
    )

    # Basic progressive backoff for rate limits:
    # - On 429: sleep 2s -> 4s -> 8s -> ... up to 10s
    # - On any successful request: reset backoff to 2s
    # - On 401/403: fail fast
    backoff_lock = Lock()
    backoff_s = 2.0

    def _status_code(err: BaseException) -> int | None:
        code = getattr(err, "status_code", None)
        if isinstance(code, int):
            return code
        resp = getattr(err, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            if isinstance(code, int):
                return code
        return None

    def chat_fn(messages, *, model: str, temperature: float):
        nonlocal backoff_s

        while True:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,  # pyright: ignore[reportArgumentType]
                    temperature=temperature,
                )

                with backoff_lock:
                    backoff_s = 2.0

                return resp.choices[0].message.content or ""
            except Exception as e:
                code = _status_code(e)
                if code in {401, 403}:
                    err = RuntimeError(f"LLM 提供商鉴权失败（HTTP {code}）。")
                    try:
                        setattr(err, "status_code", int(code))
                    except Exception:
                        pass
                    raise err from e
                if code == 429:
                    with backoff_lock:
                        delay_s = max(0.0, min(10.0, float(backoff_s)))
                        backoff_s = min(backoff_s * 2.0, 10.0)
                    logger.warning(
                        "LLM 请求触发限流(HTTP 429)，将暂停 %.1fs 后重试（最大 10s）。", delay_s
                    )
                    time.sleep(delay_s)
                    continue
                raise

    def chat_json(*, system_prompt: str, payload: dict[str, str], **kwargs) -> dict[str, str]:
        temperature = float(kwargs.get("temperature", llm_temperature))
        max_steps = int(kwargs.get("max_steps", 3))
        return call_chat_json_agent_loop(
            chat_fn=chat_fn,
            system_prompt=system_prompt,
            payload=payload,
            model=llm_model,
            temperature=temperature,
            max_steps=max_steps,
        )

    return chat_json


def process_subtitle_file(
    in_path: str,
    *,
    processor: str,
    out_dir: str,
    options: dict,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    openai_api_key: str = "",
    openai_base_url: str | None = None,
    chat_json: Callable[..., dict[str, str]] | None = None,
) -> SubtitleProcessingResult:
    """Process a subtitle file and write processed output to disk.

    This pipeline is designed for WebUI usage; it keeps LLM wiring outside processors.
    """
    processor_cls = get_processor(processor)
    proc = processor_cls()

    in_path_p = Path(in_path)
    lines: list[SubtitleLine] = load_subtitle_file(str(in_path_p))

    if chat_json is None:
        chat_json = _make_openai_chat_json(
            api_key=openai_api_key,
            base_url=openai_base_url,
            llm_model=(llm_model or "").strip() or "gpt-4o-mini",
            llm_temperature=float(llm_temperature),
        )

    ctx = ProcessorContext(chat_json=chat_json)

    started = time.time()
    out_lines = proc.process(lines, ctx=ctx, options=options or {})
    elapsed_ms = round((time.time() - started) * 1000.0)

    ext = in_path_p.suffix.lower().lstrip(".") or "srt"
    if ext not in {"srt", "vtt"}:
        ext = "srt"

    out_text = compose_vtt(out_lines) if ext == "vtt" else compose_srt(out_lines)

    out_name = f"{in_path_p.stem}--{processor}--{time.strftime('%Y%m%d-%H%M%S')}.{ext}"
    out_path = Path(out_dir) / out_name
    _write_text(out_path, out_text)

    preview_lines = [line.text for line in out_lines[:20] if (line.text or "").strip()]
    preview = "\n".join(preview_lines).strip()
    debug = (
        f"processor={processor}, cues_in={len(lines)}, cues_out={len(out_lines)}, "
        f"elapsed={elapsed_ms}ms"
    )
    logger.info("subtitle processing done: %s", debug)

    return SubtitleProcessingResult(out_path=str(out_path), preview_text=preview, debug=debug)


def process_subtitle_file_multi(
    in_path: str,
    *,
    processors: list[str],
    out_dir: str,
    options_by_processor: dict[str, dict] | None,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    openai_api_key: str = "",
    openai_base_url: str | None = None,
    chat_json: Callable[..., dict[str, str]] | None = None,
) -> SubtitleProcessingResult:
    """Process a subtitle file with multiple processors in order and write output to disk."""
    if not processors:
        raise ValueError("processors must not be empty")

    in_path_p = Path(in_path)
    lines: list[SubtitleLine] = load_subtitle_file(str(in_path_p))

    if chat_json is None:
        chat_json = _make_openai_chat_json(
            api_key=openai_api_key,
            base_url=openai_base_url,
            llm_model=(llm_model or "").strip() or "gpt-4o-mini",
            llm_temperature=float(llm_temperature),
        )

    ctx = ProcessorContext(chat_json=chat_json)

    started = time.time()
    step_debug: list[str] = []
    for name in processors:
        processor_cls = get_processor(name)
        proc = processor_cls()
        opts = (options_by_processor or {}).get(name, {})
        before = len(lines)
        lines = proc.process(lines, ctx=ctx, options=opts)
        after = len(lines)
        step_debug.append(f"{name}:{before}->{after}")

    elapsed_ms = round((time.time() - started) * 1000.0)

    ext = in_path_p.suffix.lower().lstrip(".") or "srt"
    if ext not in {"srt", "vtt"}:
        ext = "srt"

    out_text = compose_vtt(lines) if ext == "vtt" else compose_srt(lines)

    chain = "+".join(processors)
    out_name = f"{in_path_p.stem}--{chain}--{time.strftime('%Y%m%d-%H%M%S')}.{ext}"
    out_path = Path(out_dir) / out_name
    _write_text(out_path, out_text)

    preview_lines = [line.text for line in lines[:20] if (line.text or "").strip()]
    preview = "\n".join(preview_lines).strip()
    debug = (
        f"processors={processors}, steps={','.join(step_debug)}, "
        f"cues_out={len(lines)}, elapsed={elapsed_ms}ms"
    )
    logger.info("subtitle processing done: %s", debug)

    return SubtitleProcessingResult(out_path=str(out_path), preview_text=preview, debug=debug)


__all__ = ["SubtitleProcessingResult", "process_subtitle_file", "process_subtitle_file_multi"]
