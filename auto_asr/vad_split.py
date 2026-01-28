from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from auto_asr.audio_tools import WAV_SAMPLE_RATE, load_audio, process_vad, save_audio_file


@dataclass(frozen=True)
class AudioChunk:
    start_sample: int
    end_sample: int
    wav: np.ndarray

    @property
    def start_s(self) -> float:
        return self.start_sample / float(WAV_SAMPLE_RATE)

    @property
    def end_s(self) -> float:
        return self.end_sample / float(WAV_SAMPLE_RATE)

    @property
    def duration_s(self) -> float:
        return (self.end_sample - self.start_sample) / float(WAV_SAMPLE_RATE)


_VAD_MODEL: object | None = None

logger = logging.getLogger(__name__)


def get_vad_model() -> object | None:
    global _VAD_MODEL
    if _VAD_MODEL is not None:
        return _VAD_MODEL

    # silero_vad pulls in heavy deps (PyTorch/ONNXRuntime). If it fails to load at runtime,
    # we fall back to fixed chunking to keep the app usable.
    try:
        from silero_vad import load_silero_vad  # type: ignore

        logger.info("加载 Silero VAD 模型中（onnx=True）...")
        _VAD_MODEL = load_silero_vad(onnx=True)
        logger.info("Silero VAD 模型加载完成。")
    except Exception as e:
        logger.info("无法加载 silero_vad，将使用固定分段切分。原因: %s", e)
        _VAD_MODEL = None
    return _VAD_MODEL


def load_and_split(
    *,
    file_path: str,
    enable_vad: bool,
    vad_segment_threshold_s: int,
    vad_max_segment_threshold_s: int,
    vad_threshold: float = 0.5,
    vad_min_speech_duration_ms: int = 200,
    vad_min_silence_duration_ms: int = 200,
    vad_speech_pad_ms: int = 200,
    vad_min_duration_s: int = 180,
) -> tuple[list[AudioChunk], bool]:
    wav = load_audio(file_path)

    duration_s = len(wav) / float(WAV_SAMPLE_RATE)
    logger.info(
        "切分参数: enable_vad=%s, duration=%.2fs, min_duration=%ds, target=%ds, max=%ds",
        enable_vad,
        duration_s,
        vad_min_duration_s,
        vad_segment_threshold_s,
        vad_max_segment_threshold_s,
    )
    if not enable_vad or duration_s < vad_min_duration_s:
        logger.info("不进行切分（或音频较短），直接整段转写。")
        return [AudioChunk(start_sample=0, end_sample=len(wav), wav=wav)], False

    vad_model = get_vad_model()
    parts, used_vad = process_vad(
        wav,
        vad_model,
        segment_threshold_s=vad_segment_threshold_s,
        max_segment_threshold_s=vad_max_segment_threshold_s,
        vad_threshold=vad_threshold,
        vad_min_speech_duration_ms=vad_min_speech_duration_ms,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
    )
    logger.info("切分完成: chunks=%d, used_vad=%s", len(parts), used_vad)
    return [AudioChunk(start_sample=s, end_sample=e, wav=w) for (s, e, w) in parts], used_vad


__all__ = [
    "WAV_SAMPLE_RATE",
    "AudioChunk",
    "get_vad_model",
    "load_and_split",
    "save_audio_file",
]
