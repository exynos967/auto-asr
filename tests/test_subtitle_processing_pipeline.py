from __future__ import annotations

import json
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
        return payload

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        return "a<br>b"

    res = process_subtitle_file(
        str(p),
        processor="split",
        out_dir=str(tmp_path),
        options={"mode": "inplace_newlines", "concurrency": 1},
        llm_model="gpt-test",
        openai_api_key="sk-test",
        openai_base_url=None,
        chat_json=chat_json,
        chat_text=chat_text,
    )
    assert res.out_path.endswith(".srt")


def test_pipeline_multi_processors_runs_in_order(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\na b\n\n", encoding="utf-8")

    from auto_asr.subtitle_processing.pipeline import process_subtitle_file_multi

    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        key = next(iter(payload.keys()))
        txt = payload[key]
        if "proofreader" in system_prompt:
            return {key: txt}
        return {key: txt}

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        system_prompt = messages[0]["content"]
        if "字幕分句专家" in system_prompt or "字幕分段专家" in system_prompt:
            return "a<br>b"
        if "professional subtitle translator" in system_prompt:
            payload = json.loads(messages[-1]["content"])
            return json.dumps({k: str(v).upper() for k, v in payload.items()})
        return ""

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
        chat_text=chat_text,
    )
    assert res.out_path.endswith(".srt")
    out_text = (tmp_path / Path(res.out_path).name).read_text(encoding="utf-8")
    assert "A\nB" in out_text


def test_pipeline_allows_omitting_api_key_when_custom_chat_json(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nab\n\n", encoding="utf-8")

    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        return payload

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        return "a<br>b"

    res = process_subtitle_file(
        str(p),
        processor="split",
        out_dir=str(tmp_path),
        options={"mode": "inplace_newlines", "concurrency": 1},
        llm_model="gpt-test",
        chat_json=chat_json,
        chat_text=chat_text,
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


def test_pipeline_passes_llm_temperature_to_chat_text_default(tmp_path, monkeypatch):
    import auto_asr.subtitle_processing.pipeline as pipeline

    captured: dict[str, float] = {}

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            class _Chat:
                def __init__(self, parent):
                    self.completions = _Completions(parent)

            class _Completions:
                def __init__(self, parent):
                    self._parent = parent

                def create(self, *, temperature: float, **_kwargs):
                    captured["temperature"] = float(temperature)

                    class _Msg:
                        def __init__(self, content: str):
                            self.content = content

                    class _Choice:
                        def __init__(self, content: str):
                            self.message = _Msg(content)

                    class _Resp:
                        def __init__(self, content: str):
                            self.choices = [_Choice(content)]

                    return _Resp('{"1":"ok"}')

            self.chat = _Chat(self)

    monkeypatch.setattr(pipeline, "OpenAI", DummyOpenAI)

    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nab\n\n", encoding="utf-8")

    process_subtitle_file(
        str(p),
        processor="translate",
        out_dir=str(tmp_path),
        options={"target_language": "en", "batch_size": 10, "concurrency": 1},
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
        return payload

    def chat_text(*, messages: list[dict[str, str]], **_kwargs):
        return "a<br>b"

    res = process_subtitle_file(
        str(p),
        processor="split",
        out_dir=str(tmp_path),
        options={"mode": "inplace_newlines", "concurrency": 1},
        llm_model="gpt-test",
        chat_json=chat_json,
        chat_text=chat_text,
    )
    assert res.out_path.endswith(".srt")


def test_subtitle_openai_chat_text_retries_on_429_with_progressive_sleep_and_reset(monkeypatch):
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

    chat_text = pipeline._make_openai_chat_text(
        api_key="sk-test",
        base_url=None,
        llm_model="gpt-test",
        llm_temperature=0.2,
    )

    out1 = chat_text(system_prompt="x", user_prompt="hello")
    assert out1 == '{"0":"ok"}'

    out2 = chat_text(system_prompt="x", user_prompt="hello")
    assert out2 == '{"0":"ok"}'

    # 429 backoff: 2s -> 4s, then reset to 2s after a successful request
    assert sleeps == [2.0, 4.0, 2.0]


def test_subtitle_openai_chat_text_accepts_messages_kwarg(monkeypatch):
    import auto_asr.subtitle_processing.pipeline as pipeline

    captured: dict[str, object] = {}

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            class _Chat:
                def __init__(self, parent):
                    self.completions = _Completions(parent)

            class _Completions:
                def __init__(self, parent):
                    self._parent = parent

                def create(self, *, messages, **_kwargs):
                    captured["messages"] = messages

                    class _Msg:
                        def __init__(self, content: str):
                            self.content = content

                    class _Choice:
                        def __init__(self, content: str):
                            self.message = _Msg(content)

                    class _Resp:
                        def __init__(self, content: str):
                            self.choices = [_Choice(content)]

                    return _Resp("ok")

            self.chat = _Chat(self)

    monkeypatch.setattr(pipeline, "OpenAI", DummyOpenAI)

    chat_text = pipeline._make_openai_chat_text(
        api_key="sk-test",
        base_url=None,
        llm_model="gpt-test",
        llm_temperature=0.2,
    )

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    out = chat_text(messages=msgs)
    assert out == "ok"
    assert captured["messages"] == msgs


@pytest.mark.parametrize("status_code", [401, 403])
def test_subtitle_openai_chat_text_401_403_error_no_retry(monkeypatch, status_code):
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

    chat_text = pipeline._make_openai_chat_text(
        api_key="sk-test",
        base_url=None,
        llm_model="gpt-test",
        llm_temperature=0.2,
    )

    with pytest.raises(RuntimeError):
        chat_text(system_prompt="x", user_prompt="hello")

    assert sleeps == []
