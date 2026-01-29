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


def test_load_srt_without_index_line(tmp_path):
    p = tmp_path / "no_index.srt"
    p.write_text(
        "00:00:00,000 --> 00:00:01,000\nhello\n\n",
        encoding="utf-8",
    )
    lines = load_subtitle_file(str(p))
    assert len(lines) == 1
    assert lines[0].text == "hello"


def test_load_vtt_basic(tmp_path):
    p = tmp_path / "a.vtt"
    p.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n\n", encoding="utf-8")
    lines = load_subtitle_file(str(p))
    assert len(lines) == 1
    assert lines[0].start_s == 0.0
    assert lines[0].end_s == 1.0
    assert lines[0].text == "hi"


def test_load_vtt_skips_note_and_cue_identifier(tmp_path):
    p = tmp_path / "note.vtt"
    p.write_text(
        "\n".join(
            [
                "WEBVTT",
                "",
                "NOTE this is a note",
                "more note lines",
                "",
                "cue-1",
                "00:00:00.000 --> 00:00:01.000",
                "hello",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = load_subtitle_file(str(p))
    assert len(lines) == 1
    assert lines[0].text == "hello"


def test_load_unknown_extension_detects_vtt_by_header(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n\n", encoding="utf-8")
    lines = load_subtitle_file(str(p))
    assert len(lines) == 1
    assert lines[0].text == "hi"


def test_load_unknown_extension_defaults_to_srt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    lines = load_subtitle_file(str(p))
    assert len(lines) == 1
    assert lines[0].text == "hello"
