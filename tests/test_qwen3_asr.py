from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def test_resolve_qwen3_language_maps_common_codes():
    from auto_asr.qwen3_asr import resolve_qwen3_language

    assert resolve_qwen3_language(None) is None
    assert resolve_qwen3_language("auto") is None
    assert resolve_qwen3_language("zh") == "Chinese"
    assert resolve_qwen3_language("en") == "English"
    assert resolve_qwen3_language("ja") == "Japanese"


def test_transcribe_chunks_qwen3_builds_segments_from_time_stamps(monkeypatch):
    from auto_asr.openai_asr import ASRSegment
    from auto_asr.qwen3_asr import Qwen3ASRConfig, transcribe_chunks_qwen3

    @dataclass
    class TS:
        text: str
        start_time: float
        end_time: float

    @dataclass
    class R:
        language: str
        text: str
        time_stamps: list[TS] | None

    class FakeModel:
        def transcribe(self, *, audio, language, return_time_stamps):
            assert return_time_stamps is True
            assert isinstance(audio, list) and len(audio) == 1
            assert language == ["English"]
            return [
                R(
                    language="en",
                    text="hello world",
                    time_stamps=[TS("hello", 0.0, 0.5), TS("world", 0.5, 1.0)],
                )
            ]

    def fake_make_model(_cfg):
        return FakeModel()

    monkeypatch.setattr("auto_asr.qwen3_asr._make_model", fake_make_model)

    wav = np.zeros(16000, dtype=np.float32)
    cfg = Qwen3ASRConfig(
        model="Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        device="cpu",
    )
    out = transcribe_chunks_qwen3(
        chunks=[wav],
        cfg=cfg,
        language="en",
        return_time_stamps=True,
        sample_rate=16000,
    )
    assert len(out) == 1
    assert out[0].text == "hello world"
    assert out[0].segments == [
        ASRSegment(start_s=0.0, end_s=0.5, text="hello"),
        ASRSegment(start_s=0.5, end_s=1.0, text="world"),
    ]

