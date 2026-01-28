"""
Audio loading and (optional) VAD-based splitting utilities.

This module includes code derived from Qwen3-ASR-Toolkit (MIT License).
See `THIRD_PARTY_NOTICES.md` for license details.
"""

from __future__ import annotations

import inspect
import io
import logging
import os
import subprocess

import numpy as np
import soundfile as sf

try:
    from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
except Exception:  # pragma: no cover
    get_ffmpeg_exe = None  # type: ignore[assignment]

try:
    from silero_vad import get_speech_timestamps  # type: ignore
except Exception:  # pragma: no cover
    get_speech_timestamps = None  # type: ignore[assignment]


WAV_SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


def _ffmpeg_bin() -> str:
    if get_ffmpeg_exe is None:
        return "ffmpeg"
    try:
        return get_ffmpeg_exe()
    except Exception:  # pragma: no cover
        return "ffmpeg"


def load_audio(file_path: str) -> np.ndarray:
    if file_path.startswith(("http://", "https://")):
        raise ValueError("暂不支持远程 URL，请先下载到本地文件。")

    logger.info("读取音频: %s", file_path)

    command = [
        _ffmpeg_bin(),
        "-i",
        file_path,
        "-ar",
        str(WAV_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，请安装 `imageio-ffmpeg` 或系统 ffmpeg。") from e

    stdout_data, stderr_data = process.communicate()
    if process.returncode != 0:
        msg = stderr_data.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg 处理音频失败：{msg}")

    with io.BytesIO(stdout_data) as data_io:
        wav_data, _sr = sf.read(data_io, dtype="float32")

    if wav_data.ndim == 2:
        wav_data = wav_data.mean(axis=1)

    logger.info(
        "音频已解码: samples=%d, sr=%d, duration=%.2fs",
        len(wav_data),
        WAV_SAMPLE_RATE,
        len(wav_data) / float(WAV_SAMPLE_RATE),
    )
    return wav_data


def process_vad(
    wav: np.ndarray,
    worker_vad_model: object | None,
    segment_threshold_s: int = 120,
    max_segment_threshold_s: int = 180,
    *,
    vad_threshold: float = 0.5,
    vad_min_speech_duration_ms: int = 200,
    vad_min_silence_duration_ms: int = 200,
    vad_speech_pad_ms: int = 200,
) -> tuple[list[tuple[int, int, np.ndarray]], bool]:
    """
    Segment long audio using Silero VAD timestamps when available, otherwise fall back
    to fixed-size chunking.
    """

    try:
        if worker_vad_model is None or get_speech_timestamps is None:
            raise RuntimeError("VAD model not available.")

        def _filtered_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
            try:
                sig = inspect.signature(get_speech_timestamps)
                return {k: v for k, v in kwargs.items() if k in sig.parameters}
            except Exception:  # pragma: no cover
                return kwargs

        vad_params = {
            "sampling_rate": WAV_SAMPLE_RATE,
            "return_seconds": False,
            "threshold": float(vad_threshold),
            "min_speech_duration_ms": int(vad_min_speech_duration_ms),
            "min_silence_duration_ms": int(vad_min_silence_duration_ms),
            "speech_pad_ms": int(vad_speech_pad_ms),
        }

        speech_timestamps = get_speech_timestamps(
            wav, worker_vad_model, **_filtered_kwargs(vad_params)
        )
        if not speech_timestamps:
            raise ValueError("No speech segments detected by VAD.")

        potential_split_points = {0, len(wav)}
        for ts in speech_timestamps:
            start_of_next = int(ts["start"])
            potential_split_points.add(start_of_next)
        sorted_potential_splits = sorted(potential_split_points)

        final_split_points = {0, len(wav)}
        segment_threshold_samples = int(segment_threshold_s) * WAV_SAMPLE_RATE
        target = segment_threshold_samples
        while target < len(wav):
            closest_point = min(sorted_potential_splits, key=lambda p: abs(p - target))
            final_split_points.add(int(closest_point))
            target += segment_threshold_samples
        final_ordered_splits = sorted(final_split_points)

        max_segment_threshold_samples = int(max_segment_threshold_s) * WAV_SAMPLE_RATE
        split_points: list[float] = [0.0]

        for i in range(1, len(final_ordered_splits)):
            start = final_ordered_splits[i - 1]
            end = final_ordered_splits[i]
            segment_length = end - start

            if segment_length <= max_segment_threshold_samples:
                split_points.append(float(end))
                continue

            num_subsegments = int(np.ceil(segment_length / max_segment_threshold_samples))
            subsegment_length = segment_length / num_subsegments
            for j in range(1, num_subsegments):
                split_points.append(start + j * subsegment_length)
            split_points.append(float(end))

        segmented_wavs: list[tuple[int, int, np.ndarray]] = []
        for i in range(len(split_points) - 1):
            start_sample = int(split_points[i])
            end_sample = int(split_points[i + 1])
            segmented_wavs.append((start_sample, end_sample, wav[start_sample:end_sample]))
        return segmented_wavs, True

    except Exception as e:
        logger.info("VAD 分段失败，降级为固定分段: %s", e)
        segmented_wavs: list[tuple[int, int, np.ndarray]] = []
        total_samples = len(wav)
        max_chunk_size_samples = int(max_segment_threshold_s) * WAV_SAMPLE_RATE

        for start_sample in range(0, total_samples, max_chunk_size_samples):
            end_sample = min(start_sample + max_chunk_size_samples, total_samples)
            segment = wav[start_sample:end_sample]
            if len(segment) > 0:
                segmented_wavs.append((start_sample, end_sample, segment))
        return segmented_wavs, False


def process_vad_speech(
    wav: np.ndarray,
    worker_vad_model: object,
    *,
    max_utterance_s: int = 20,
    merge_gap_ms: int = 300,
    vad_threshold: float = 0.5,
    vad_min_speech_duration_ms: int = 200,
    vad_min_silence_duration_ms: int = 200,
    vad_speech_pad_ms: int = 200,
) -> list[tuple[int, int, np.ndarray]]:
    """
    Split by VAD speech regions (better subtitle time alignment).

    - merge_gap_ms: merge adjacent speech regions if silence gap is short.
    - max_utterance_s: cap each region length; long regions are subdivided.
    """
    if get_speech_timestamps is None:
        raise RuntimeError("silero_vad is not available.")

    def _filtered_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
        try:
            sig = inspect.signature(get_speech_timestamps)
            return {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:  # pragma: no cover
            return kwargs

    vad_params = {
        "sampling_rate": WAV_SAMPLE_RATE,
        "return_seconds": False,
        "threshold": float(vad_threshold),
        "min_speech_duration_ms": int(vad_min_speech_duration_ms),
        "min_silence_duration_ms": int(vad_min_silence_duration_ms),
        "speech_pad_ms": int(vad_speech_pad_ms),
    }
    timestamps = get_speech_timestamps(wav, worker_vad_model, **_filtered_kwargs(vad_params))
    if not timestamps:
        return []

    # Merge close regions.
    merged: list[tuple[int, int]] = []
    gap_samples = int(merge_gap_ms * WAV_SAMPLE_RATE / 1000)
    for ts in timestamps:
        start = int(ts["start"])
        end = int(ts["end"])
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        if start - last_end <= gap_samples:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    # Subdivide long regions.
    max_samples = int(max_utterance_s) * WAV_SAMPLE_RATE
    regions: list[tuple[int, int, np.ndarray]] = []
    for start, end in merged:
        if end <= start:
            continue
        if max_samples > 0 and (end - start) > max_samples:
            cur = start
            while cur < end:
                nxt = min(cur + max_samples, end)
                regions.append((cur, nxt, wav[cur:nxt]))
                cur = nxt
        else:
            regions.append((start, end, wav[start:end]))
    return regions


def save_audio_file(wav: np.ndarray, file_path: str) -> None:
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    # Use PCM_16 to keep files small (faster disk I/O + less chance to hit upstream size limits).
    sf.write(file_path, wav, WAV_SAMPLE_RATE, subtype="PCM_16")


def transcode_wav_to_mp3(
    *,
    input_wav_path: str,
    output_mp3_path: str,
    bitrate_kbps: int = 64,
) -> str:
    """
    Convert a WAV file to MP3 with ffmpeg.

    This is mainly used to avoid upstream file size limits (PCM WAV is large).
    """
    bitrate_kbps = max(8, int(bitrate_kbps))
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        input_wav_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(WAV_SAMPLE_RATE),
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate_kbps}k",
        output_mp3_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，无法进行 MP3 转码。") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg 转码失败：{stderr}") from e
    return output_mp3_path


__all__ = [
    "WAV_SAMPLE_RATE",
    "load_audio",
    "process_vad",
    "process_vad_speech",
    "save_audio_file",
    "transcode_wav_to_mp3",
]
