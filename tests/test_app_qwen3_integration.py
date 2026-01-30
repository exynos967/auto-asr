from __future__ import annotations

from dataclasses import dataclass


def test_app_load_qwen3_model_ui_builds_config(monkeypatch):
    import app

    seen = {}

    def fake_preload(*, cfg):
        seen["cfg"] = cfg
        return object()

    monkeypatch.setattr(app, "preload_qwen3_model", fake_preload)

    msg = app.load_qwen3_model_ui(
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-ForcedAligner-0.6B",
        "cpu",
    )
    assert "已加载 Qwen3-ASR 模型" in msg
    assert seen["cfg"].model == "Qwen/Qwen3-ASR-1.7B"
    assert seen["cfg"].forced_aligner == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert seen["cfg"].device == "cpu"


def test_app_run_asr_passes_qwen3_options(monkeypatch):
    import app

    captured = {}

    def fake_update_config(_cfg):
        captured["saved_config"] = _cfg
        return "/tmp/.auto_asr_config.json"

    @dataclass
    class DummyRes:
        preview_text: str = "p"
        full_text: str = "t"
        subtitle_file_path: str = "out.srt"
        debug: str = "d"

    def fake_transcribe_to_subtitles(**kwargs):
        captured["call_kwargs"] = kwargs
        return DummyRes()

    monkeypatch.setattr(app, "update_config", fake_update_config)
    monkeypatch.setattr(app, "transcribe_to_subtitles", fake_transcribe_to_subtitles)

    out = app.run_asr(
        "dummy.wav",
        "qwen3asr",
        "",  # openai_api_key not required for qwen3
        "",
        "whisper-1",
        "iic/SenseVoiceSmall",
        "cpu",
        "auto",
        True,
        False,
        True,
        "srt",
        "en",
        "",
        True,
        120,
        180,
        0.25,
        100,
        300,
        400,
        "vad_speech",
        8,
        100,
        "wav",
        1,
        # qwen3 settings
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-ForcedAligner-0.6B",
        "cpu",
    )
    assert out[0] == "p"
    assert captured["call_kwargs"]["asr_backend"] == "qwen3asr"
    assert captured["call_kwargs"]["qwen3_model"] == "Qwen/Qwen3-ASR-1.7B"
    assert captured["call_kwargs"]["qwen3_forced_aligner"] == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert captured["call_kwargs"]["qwen3_device"] == "cpu"

    assert captured["saved_config"]["asr_backend"] == "qwen3asr"
    assert captured["saved_config"]["qwen3_model"] == "Qwen/Qwen3-ASR-1.7B"
