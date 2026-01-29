from __future__ import annotations

from pathlib import Path

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


def test_pipeline_multi_processors_runs_in_order(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nab\n\n", encoding="utf-8")

    from auto_asr.subtitle_processing.pipeline import process_subtitle_file_multi

    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        key = next(iter(payload.keys()))
        txt = payload[key]
        if "proofreader" in system_prompt:
            return {key: txt}
        if "subtitle splitter" in system_prompt:
            return {key: "a<br>b"}
        if "translator" in system_prompt:
            return {key: txt.upper()}
        return {key: txt}

    res = process_subtitle_file_multi(
        str(p),
        processors=["optimize", "split", "translate"],
        out_dir=str(tmp_path),
        options_by_processor={
            "split": {"mode": "inplace_newlines", "concurrency": 1},
            "translate": {"target_language": "en", "batch_size": 10, "concurrency": 1},
            "optimize": {"batch_size": 10, "concurrency": 1},
        },
        llm_model="gpt-test",
        chat_json=chat_json,
    )
    assert res.out_path.endswith(".srt")
    out_text = (tmp_path / Path(res.out_path).name).read_text(encoding="utf-8")
    assert "A\nB" in out_text


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


def test_pipeline_passes_llm_temperature_to_agent_loop(tmp_path, monkeypatch):
    import auto_asr.subtitle_processing.pipeline as pipeline

    captured: dict[str, float] = {}

    def fake_agent_loop(
        *,
        chat_fn,
        system_prompt: str,
        payload: dict[str, str],
        model: str,
        temperature: float,
        max_steps: int,
    ):
        captured["temperature"] = float(temperature)
        return payload

    monkeypatch.setattr(pipeline, "call_chat_json_agent_loop", fake_agent_loop)

    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nab\n\n", encoding="utf-8")

    process_subtitle_file(
        str(p),
        processor="optimize",
        out_dir=str(tmp_path),
        options={"batch_size": 10, "concurrency": 1},
        llm_model="gpt-test",
        llm_temperature=0.25,
        openai_api_key="sk-test",
        openai_base_url=None,
        chat_json=None,
    )

    assert captured["temperature"] == 0.25


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
