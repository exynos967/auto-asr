from __future__ import annotations

from auto_asr.subtitle_io import load_subtitle_file


def test_load_srt_basic(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    lines = load_subtitle_file(str(p))
    assert len(lines) == 1
    assert lines[0].start_s == 0.0
    assert lines[0].end_s == 1.0
    assert lines[0].text == "hello"


def test_load_vtt_basic(tmp_path):
    p = tmp_path / "a.vtt"
    p.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n\n", encoding="utf-8")
    lines = load_subtitle_file(str(p))
    assert len(lines) == 1
    assert lines[0].start_s == 0.0
    assert lines[0].end_s == 1.0
    assert lines[0].text == "hi"

