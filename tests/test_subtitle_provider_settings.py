from __future__ import annotations

from pathlib import Path

from auto_asr import config


def test_save_subtitle_provider_settings_saves_values(tmp_path, monkeypatch):
    from auto_asr.subtitle_processing.settings import save_subtitle_provider_settings

    cfg_path = tmp_path / ".auto_asr_config.json"
    monkeypatch.setattr(config, "get_config_path", lambda: Path(cfg_path))

    save_subtitle_provider_settings(
        provider="openai",
        openai_api_key="sk-test",
        openai_base_url="https://api.openai.com/v1",
        llm_model="gpt-test",
        llm_temperature=0.2,
        split_strategy="semantic",
    )

    loaded = config.load_config()
    assert loaded["subtitle_provider"] == "openai"
    assert loaded["subtitle_openai_api_key"] == "sk-test"
    assert loaded["subtitle_openai_base_url"] == "https://api.openai.com/v1"
    assert loaded["subtitle_llm_model"] == "gpt-test"
    assert loaded["subtitle_llm_temperature"] == 0.2
    assert loaded["subtitle_split_strategy"] == "semantic"


def test_save_subtitle_provider_settings_preserves_api_key_when_blank(tmp_path, monkeypatch):
    from auto_asr.subtitle_processing.settings import save_subtitle_provider_settings

    cfg_path = tmp_path / ".auto_asr_config.json"
    monkeypatch.setattr(config, "get_config_path", lambda: Path(cfg_path))

    config.save_config({"subtitle_openai_api_key": "sk-old"})

    save_subtitle_provider_settings(
        provider="openai",
        openai_api_key="",
        openai_base_url="",
        llm_model="gpt-test",
        llm_temperature=0.2,
        split_strategy="semantic",
    )

    loaded = config.load_config()
    assert loaded["subtitle_openai_api_key"] == "sk-old"
