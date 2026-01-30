from __future__ import annotations

import json

from auto_asr.subtitle_processing.base import ProcessorContext
from auto_asr.subtitle_processing.processors.optimize import OptimizeProcessor
from auto_asr.subtitle_processing.processors.translate import TranslateProcessor
from auto_asr.subtitles import SubtitleLine


def test_translate_processor_batches_and_preserves_timestamps():
    calls: list[dict[str, str]] = []

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        payload = json.loads(messages[-1]["content"])
        calls.append(payload)
        return json.dumps({k: f"{v}-T" for k, v in payload.items()})

    proc = TranslateProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [
        SubtitleLine(start_s=0.0, end_s=1.0, text="a"),
        SubtitleLine(start_s=1.0, end_s=2.0, text="b"),
        SubtitleLine(start_s=2.0, end_s=3.0, text="c"),
    ]
    out = proc.process(
        lines,
        ctx=ctx,
        options={"target_language": "en", "batch_size": 2, "concurrency": 1},
    )
    assert [x.text for x in out] == ["a-T", "b-T", "c-T"]
    assert out[0].start_s == 0.0 and out[0].end_s == 1.0
    assert len(calls) == 2
    assert set(calls[0].keys()) == {"1", "2"}
    assert set(calls[1].keys()) == {"3"}


def test_translate_processor_reflect_mode_uses_native_translation():
    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        payload = json.loads(messages[-1]["content"])
        return json.dumps(
            {
                k: {
                    "initial_translation": f"{v}-I",
                    "reflection": "x",
                    "native_translation": f"{v}-N",
                }
                for k, v in payload.items()
            }
        )

    proc = TranslateProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [SubtitleLine(start_s=0.0, end_s=1.0, text="a")]
    out = proc.process(
        lines,
        ctx=ctx,
        options={
            "target_language": "en",
            "batch_size": 10,
            "concurrency": 1,
            "reflect": True,
        },
    )
    assert out[0].text == "a-N"


def test_translate_processor_falls_back_to_single_on_error():
    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        system_prompt = messages[0]["content"]
        if "Return the translation result directly" in system_prompt:
            return "SINGLE"
        raise RuntimeError("boom")

    proc = TranslateProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [
        SubtitleLine(start_s=0.0, end_s=1.0, text="a"),
        SubtitleLine(start_s=1.0, end_s=2.0, text="b"),
    ]
    out = proc.process(
        lines,
        ctx=ctx,
        options={
            "target_language": "en",
            "batch_size": 10,
            "concurrency": 1,
        },
    )
    assert [x.text for x in out] == ["SINGLE", "SINGLE"]


def test_optimize_processor_keeps_original_when_change_too_large():
    calls: list[int] = []

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        calls.append(len(messages))
        if len(calls) == 1:
            return json.dumps(
                {
                    "1": "COMPLETELY DIFFERENT",
                    "2": "hello wor1d",
                }
            )
        return json.dumps(
            {
                "1": "hello world",
                "2": "hello wor1d",
            }
        )

    proc = OptimizeProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [
        SubtitleLine(start_s=0.0, end_s=1.0, text="hello world"),
        SubtitleLine(start_s=1.0, end_s=2.0, text="hello world"),
    ]
    out = proc.process(lines, ctx=ctx, options={"batch_size": 10, "concurrency": 1})
    assert [x.text for x in out] == ["hello world", "hello wor1d"]
    assert calls == [2, 4]
