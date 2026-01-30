from __future__ import annotations

import numpy as np


def test_transcribe_to_subtitles_qwen3_backend_uses_model_timestamps(tmp_path, monkeypatch):
    import auto_asr.pipeline as pipeline
    from auto_asr.openai_asr import ASRResult, ASRSegment
    from auto_asr.vad_split import AudioChunk

    wav = np.zeros(16000, dtype=np.float32)

    def should_not_call_load_and_split(**_kwargs):
        raise AssertionError("qwen3asr should use silence splitting, not VAD splitting")

    def fake_load_and_split_silence(**_kwargs):
        return [AudioChunk(start_sample=0, end_sample=16000, wav=wav)], False

    def fake_transcribe_chunks_qwen3(**_kwargs):
        assert _kwargs["cfg"].max_inference_batch_size == 4
        return [
            ASRResult(
                text="hello world",
                segments=[
                    ASRSegment(start_s=0.0, end_s=0.5, text="hello"),
                    ASRSegment(start_s=0.5, end_s=1.0, text="world"),
                ],
            )
        ]

    monkeypatch.setattr(pipeline, "load_and_split", should_not_call_load_and_split)
    monkeypatch.setattr(
        pipeline,
        "load_and_split_silence",
        fake_load_and_split_silence,
        raising=False,
    )
    monkeypatch.setattr(pipeline, "transcribe_chunks_qwen3", fake_transcribe_chunks_qwen3)

    res = pipeline.transcribe_to_subtitles(
        input_audio_path="dummy.wav",
        asr_backend="qwen3asr",
        output_format="srt",
        model="whisper-1",
        language="en",
        prompt=None,
        enable_vad=False,
        vad_segment_threshold_s=120,
        vad_max_segment_threshold_s=180,
        vad_threshold=0.5,
        vad_min_speech_duration_ms=200,
        vad_min_silence_duration_ms=200,
        vad_speech_pad_ms=200,
        timeline_strategy="chunk",
        vad_speech_max_utterance_s=20,
        vad_speech_merge_gap_ms=300,
        upload_audio_format="wav",
        upload_mp3_bitrate_kbps=192,
        api_concurrency=1,
        outputs_dir=str(tmp_path),
        qwen3_model="Qwen/Qwen3-ASR-1.7B",
        qwen3_forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        qwen3_device="cpu",
        qwen3_max_inference_batch_size=4,
    )

    out_text = (tmp_path / res.subtitle_file_path.split("/")[-1]).read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,500" in out_text
    assert "hello" in out_text
    assert "00:00:00,500 --> 00:00:01,000" in out_text
    assert "world" in out_text
