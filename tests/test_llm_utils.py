import pytest

from auto_asr.llm.client import call_chat_json_agent_loop, normalize_base_url


def test_normalize_base_url_adds_v1():
    assert normalize_base_url("https://api.openai.com") == "https://api.openai.com/v1"


def test_normalize_base_url_keeps_v1_and_strips_trailing_slash():
    assert normalize_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert normalize_base_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"


def test_normalize_base_url_appends_v1_to_custom_path():
    assert normalize_base_url("http://127.0.0.1:8000/custom") == "http://127.0.0.1:8000/custom/v1"


def test_call_chat_json_agent_loop_retries_until_keys_match():
    calls: list[int] = []

    def chat_fn(messages, *, model, temperature):
        calls.append(len(messages))
        if len(calls) == 1:
            return '{"1":"b"}'
        return '{"0":"a","1":"b"}'

    payload = {"0": "a", "1": "b"}
    out = call_chat_json_agent_loop(
        chat_fn=chat_fn,
        system_prompt="Return JSON only.",
        payload=payload,
        model="gpt-test",
    )
    assert out == payload
    assert len(calls) == 2


def test_call_chat_json_agent_loop_retries_when_non_dict_then_succeeds():
    calls: list[int] = []

    def chat_fn(messages, *, model, temperature):
        calls.append(len(messages))
        if len(calls) == 1:
            return "[]"
        return '{"0":"a"}'

    out = call_chat_json_agent_loop(
        chat_fn=chat_fn,
        system_prompt="Return JSON only.",
        payload={"0": "a"},
        model="gpt-test",
        max_steps=3,
    )
    assert out == {"0": "a"}
    assert len(calls) == 2


def test_call_chat_json_agent_loop_raises_when_never_dict():
    def chat_fn(messages, *, model, temperature):
        return "[]"

    with pytest.raises(RuntimeError):
        call_chat_json_agent_loop(
            chat_fn=chat_fn,
            system_prompt="Return JSON only.",
            payload={"0": "a"},
            model="gpt-test",
            max_steps=2,
        )
