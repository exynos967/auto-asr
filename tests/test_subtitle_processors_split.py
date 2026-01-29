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
    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        # Echo one line with <br> split
        key = next(iter(payload.keys()))
        return {key: "a<br>b"}

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=chat_json)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="ab")]
    out = proc.process(lines, ctx=ctx, options={"mode": "inplace_newlines", "concurrency": 1})
    assert len(out) == 1
    assert out[0].text == "a\nb"
    assert out[0].start_s == 0.0 and out[0].end_s == 2.0


def test_split_processor_falls_back_when_content_changes():
    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        key = next(iter(payload.keys()))
        return {key: "a<br>DIFFERENT"}

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=chat_json)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="ab")]
    out = proc.process(lines, ctx=ctx, options={"mode": "inplace_newlines", "concurrency": 1})
    assert len(out) == 1
    assert out[0].text == "ab"


def test_split_processor_split_to_cues_allocates_timestamps():
    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        key = next(iter(payload.keys()))
        return {key: "a<br>b"}

    proc = SplitProcessor()
    ctx = ProcessorContext(chat_json=chat_json)
    lines = [SubtitleLine(start_s=0.0, end_s=2.0, text="ab")]
    out = proc.process(lines, ctx=ctx, options={"mode": "split_to_cues", "concurrency": 1})
    assert len(out) == 2
    assert out[0].start_s == 0.0
    assert abs(out[0].end_s - 1.0) < 0.01
    assert abs(out[1].start_s - 1.0) < 0.01
    assert out[1].end_s == 2.0
