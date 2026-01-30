from __future__ import annotations

from pathlib import Path

from auto_asr import config


def test_save_subtitle_processing_settings_saves_values(tmp_path, monkeypatch):
    from auto_asr.subtitle_processing.settings import save_subtitle_processing_settings

    cfg_path = tmp_path / ".auto_asr_config.json"
    monkeypatch.setattr(config, "get_config_path", lambda: Path(cfg_path))

    save_subtitle_processing_settings(
        processors=["optimize", "translate"],
        batch_size=42,
        concurrency=7,
        target_language="ja",
        split_mode="split_to_cues",
        split_max_word_count_cjk=18,
        split_max_word_count_english=12,
        translate_reflect=True,
        custom_prompt="keep terms",
    )

    loaded = config.load_config()
    assert loaded["subtitle_processors"] == ["optimize", "translate"]
    assert loaded["subtitle_batch_size"] == 42
    assert loaded["subtitle_concurrency"] == 7
    assert loaded["subtitle_target_language"] == "ja"
    assert loaded["subtitle_split_mode"] == "split_to_cues"
    assert loaded["subtitle_split_max_word_count_cjk"] == 18
    assert loaded["subtitle_split_max_word_count_english"] == 12
    assert loaded["subtitle_translate_reflect"] is True
    assert loaded["subtitle_custom_prompt"] == "keep terms"
