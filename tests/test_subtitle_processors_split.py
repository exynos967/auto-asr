from __future__ import annotations

from auto_asr.subtitle_processing.base import ProcessorContext
from auto_asr.subtitle_processing.processors.split import (
    SplitProcessor,
    split_line_to_cues,
    split_text_by_delimiter,
)
from auto_asr.subtitles import SubtitleLine


def test_split_text_by_delimiter_basic():
    assert split_text_by_delimiter("a<br>b") == ["a", "b"]


def test_split_text_by_delimiter_normalizes_newlines_and_drops_empty():
    assert split_text_by_delimiter("a<br><br> b\r\n<br>") == ["a", "b"]


def test_split_line_to_cues_allocates_time_proportionally():
    line = SubtitleLine(start_s=0.0, end_s=10.0, text="a<br>bb")
    cues = split_line_to_cues(line, ["a", "bb"])
    assert len(cues) == 2
    assert cues[0].start_s == 0.0
    assert cues[-1].end_s == 10.0
    # a : bb => 1/3 : 2/3
    assert abs(cues[0].end_s - 3.333) < 0.01
    assert abs(cues[1].start_s - 3.333) < 0.01


def test_split_processor_inplace_newlines():
    calls: list[list[dict[str, str]]] = []

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        calls.append(messages)
        return "a<br>b"

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="a b")]
    out = proc.process(lines, ctx=ctx, options={"mode": "inplace_newlines", "concurrency": 1})
    assert len(out) == 1
    assert out[0].text == "a\nb"
    assert out[0].start_s == 0.0 and out[0].end_s == 2.0
    assert len(calls) == 1
    assert calls[0][0]["role"] == "system"


def test_split_processor_strategy_semantic_uses_semantic_prompt():
    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        system_prompt = messages[0]["content"]
        assert "语义自然断点" in system_prompt
        return "a<br>b"

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="a b")]
    out = proc.process(
        lines,
        ctx=ctx,
        options={"mode": "inplace_newlines", "strategy": "semantic", "concurrency": 1},
    )
    assert out[0].text == "a\nb"


def test_split_processor_strategy_sentence_uses_sentence_prompt():
    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        system_prompt = messages[0]["content"]
        assert "句子边界" in system_prompt
        return "a<br>b"

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="a b")]
    out = proc.process(
        lines,
        ctx=ctx,
        options={"mode": "inplace_newlines", "strategy": "sentence", "concurrency": 1},
    )
    assert out[0].text == "a\nb"


def test_split_processor_falls_back_when_content_changes():
    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        return "a<br>DIFFERENT"

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="a b")]
    out = proc.process(lines, ctx=ctx, options={"mode": "inplace_newlines", "concurrency": 1})
    assert len(out) == 1
    assert out[0].text == "a b"


def test_split_processor_agent_loop_retries_when_invalid_then_succeeds():
    calls: list[int] = []

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        calls.append(len(messages))
        # First attempt returns content-changed result -> should trigger retry.
        if len(calls) == 1:
            return "a<br>DIFFERENT"
        return "a<br>b"

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="a b")]
    out = proc.process(lines, ctx=ctx, options={"mode": "inplace_newlines", "concurrency": 1})
    assert out[0].text == "a\nb"
    assert calls == [2, 4]  # history grows after first invalid attempt


def test_split_processor_retries_when_length_violation_then_succeeds():
    calls: list[int] = []

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        calls.append(len(messages))
        if len(calls) == 1:
            return "hello world"  # no split -> 2 words
        return "hello<br>world"

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="hello world")]
    out = proc.process(
        lines,
        ctx=ctx,
        options={
            "mode": "inplace_newlines",
            "strategy": "sentence",
            "concurrency": 1,
            "max_word_count_english": 1,
        },
    )
    assert out[0].text == "hello\nworld"
    assert calls == [2, 4]


def test_split_processor_split_to_cues_allocates_timestamps():
    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        return "a<br>b"

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=lambda **_kw: {}, chat_text=chat_text)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="a b")]
    out = proc.process(lines, ctx=ctx, options={"mode": "split_to_cues", "concurrency": 1})
    assert len(out) == 2
    assert out[0].start_s == 0.0
    assert abs(out[0].end_s - 1.0) < 0.01
    assert abs(out[1].start_s - 1.0) < 0.01
    assert out[1].end_s == 2.0
