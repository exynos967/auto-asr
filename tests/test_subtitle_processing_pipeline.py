from __future__ import annotations

import pytest

from auto_asr.subtitle_processing.pipeline import process_subtitle_file


def test_pipeline_requires_processor(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", encoding="utf-8")
    with pytest.raises(KeyError):
        process_subtitle_file(
            str(p),
            processor="nope",
            out_dir=str(tmp_path),
            options={},
            llm_model="gpt-test",
            openai_api_key="sk-test",
            openai_base_url=None,
            chat_json=None,
        )


def test_pipeline_runs_split_and_writes_file(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nab\n\n", encoding="utf-8")

    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        key = next(iter(payload.keys()))
        return {key: "a<br>b"}

    res = process_subtitle_file(
        str(p),
        processor="split",
        out_dir=str(tmp_path),
        options={"mode": "inplace_newlines", "concurrency": 1},
        llm_model="gpt-test",
        openai_api_key="sk-test",
        openai_base_url=None,
        chat_json=chat_json,
    )
    assert res.out_path.endswith(".srt")
    out_text = (tmp_path / res.out_path.split("/")[-1]).read_text(encoding="utf-8")
    assert "a\nb" in out_text


def test_pipeline_allows_omitting_api_key_when_custom_chat_json(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nab\n\n", encoding="utf-8")

    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        key = next(iter(payload.keys()))
        return {key: "a<br>b"}

    res = process_subtitle_file(
        str(p),
        processor="split",
        out_dir=str(tmp_path),
        options={"mode": "inplace_newlines", "concurrency": 1},
        llm_model="gpt-test",
        chat_json=chat_json,
    )
    assert res.out_path.endswith(".srt")


def test_pipeline_default_chat_json_requires_api_key(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nab\n\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        process_subtitle_file(
            str(p),
            processor="split",
            out_dir=str(tmp_path),
            options={"mode": "inplace_newlines", "concurrency": 1},
            llm_model="gpt-test",
            chat_json=None,
        )


def test_pipeline_non_subtitle_input_falls_back_to_srt_suffix(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nab\n\n", encoding="utf-8")

    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        key = next(iter(payload.keys()))
        return {key: "a<br>b"}

    res = process_subtitle_file(
        str(p),
        processor="split",
        out_dir=str(tmp_path),
        options={"mode": "inplace_newlines", "concurrency": 1},
        llm_model="gpt-test",
        chat_json=chat_json,
    )
    assert res.out_path.endswith(".srt")
