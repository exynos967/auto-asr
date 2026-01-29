from __future__ import annotations

from pathlib import Path

from auto_asr import config


def test_update_config_preserves_api_key_when_blank(tmp_path, monkeypatch):
    cfg_path = tmp_path / ".auto_asr_config.json"
    monkeypatch.setattr(config, "get_config_path", lambda: Path(cfg_path))

    config.save_config({"openai_api_key": "sk-old", "x": 1})
    config.update_config({"openai_api_key": "", "llm_model": "gpt-4o-mini"})

    loaded = config.load_config()
    assert loaded["openai_api_key"] == "sk-old"
    assert loaded["llm_model"] == "gpt-4o-mini"


def test_update_config_overwrites_api_key_when_non_empty(tmp_path, monkeypatch):
    cfg_path = tmp_path / ".auto_asr_config.json"
    monkeypatch.setattr(config, "get_config_path", lambda: Path(cfg_path))

    config.save_config({"openai_api_key": "sk-old"})
    config.update_config({"openai_api_key": "sk-new"})

    loaded = config.load_config()
    assert loaded["openai_api_key"] == "sk-new"
