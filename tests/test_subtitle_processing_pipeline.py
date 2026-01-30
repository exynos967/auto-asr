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


def test_subtitle_openai_chat_json_retries_on_429_with_progressive_sleep_and_reset(monkeypatch):
    import auto_asr.subtitle_processing.pipeline as pipeline

    sleeps: list[float] = []
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: sleeps.append(float(s)))

    class DummyHTTPError(RuntimeError):
        def __init__(self, status_code: int):
            super().__init__(f"http {status_code}")
            self.status_code = status_code

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            self._actions = [
                429,
                429,
                '{"0":"ok"}',
                429,
                '{"0":"ok"}',
            ]

            class _Chat:
                def __init__(self, parent):
                    self.completions = _Completions(parent)

            class _Completions:
                def __init__(self, parent):
                    self._parent = parent

                def create(self, **_kwargs):
                    action = self._parent._actions.pop(0)
                    if isinstance(action, int):
                        raise DummyHTTPError(action)

                    class _Msg:
                        def __init__(self, content: str):
                            self.content = content

                    class _Choice:
                        def __init__(self, content: str):
                            self.message = _Msg(content)

                    class _Resp:
                        def __init__(self, content: str):
                            self.choices = [_Choice(content)]

                    return _Resp(action)

            self.chat = _Chat(self)

    monkeypatch.setattr(pipeline, "OpenAI", DummyOpenAI)

    chat_json = pipeline._make_openai_chat_json(
        api_key="sk-test",
        base_url=None,
        llm_model="gpt-test",
        llm_temperature=0.2,
    )

    out1 = chat_json(system_prompt="x", payload={"0": "ok"}, max_steps=1)
    assert out1 == {"0": "ok"}

    out2 = chat_json(system_prompt="x", payload={"0": "ok"}, max_steps=1)
    assert out2 == {"0": "ok"}

    # 429 backoff: 2s -> 4s, then reset to 2s after a successful request
    assert sleeps == [2.0, 4.0, 2.0]


@pytest.mark.parametrize("status_code", [401, 403])
def test_subtitle_openai_chat_json_401_403_error_no_retry(monkeypatch, status_code):
    import auto_asr.subtitle_processing.pipeline as pipeline

    sleeps: list[float] = []
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: sleeps.append(float(s)))

    class DummyHTTPError(RuntimeError):
        def __init__(self, status_code: int):
            super().__init__(f"http {status_code}")
            self.status_code = status_code

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            class _Chat:
                def __init__(self, parent):
                    self.completions = _Completions(parent)

            class _Completions:
                def __init__(self, parent):
                    self._parent = parent

                def create(self, **_kwargs):
                    raise DummyHTTPError(status_code)

            self.chat = _Chat(self)

    monkeypatch.setattr(pipeline, "OpenAI", DummyOpenAI)

    chat_json = pipeline._make_openai_chat_json(
        api_key="sk-test",
        base_url=None,
        llm_model="gpt-test",
        llm_temperature=0.2,
    )

    with pytest.raises(RuntimeError):
        chat_json(system_prompt="x", payload={"0": "ok"}, max_steps=1)

    assert sleeps == []
