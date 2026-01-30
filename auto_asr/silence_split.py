from __future__ import annotations

import logging

import numpy as np

from auto_asr.audio_tools import load_audio
from auto_asr.vad_split import WAV_SAMPLE_RATE, AudioChunk

logger = logging.getLogger(__name__)


def _get_rms(
    y: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Compute RMS with a sliding window.

    This is adapted from GPT-SoVITS' `tools/slicer2.py` (numpy-only implementation).
    """

    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + (y.strides[axis],)

    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + (frame_length,)
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)

    target_axis = axis - 1 if axis < 0 else axis + 1
    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the framing axis.
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


class SilenceSlicer:
    """Silence-based slicer (RMS threshold), aligned with GPT-SoVITS behavior."""

    def __init__(
        self,
        *,
        sr: int,
        threshold_db: float = -40.0,
        min_length_ms: int = 5000,
        min_interval_ms: int = 300,
        hop_size_ms: int = 20,
        max_sil_kept_ms: int = 5000,
    ) -> None:
        if not min_length_ms >= min_interval_ms >= hop_size_ms:
            raise ValueError("min_length_ms >= min_interval_ms >= hop_size_ms must hold")
        if not max_sil_kept_ms >= hop_size_ms:
            raise ValueError("max_sil_kept_ms >= hop_size_ms must hold")

        min_interval_samples = sr * min_interval_ms / 1000.0
        self.threshold = 10 ** (float(threshold_db) / 20.0)
        self.hop_size = round(sr * hop_size_ms / 1000.0)
        self.win_size = min(round(min_interval_samples), 4 * self.hop_size)

        self.min_length_frames = round(sr * min_length_ms / 1000.0 / self.hop_size)
        self.min_interval_frames = round(min_interval_samples / self.hop_size)
        self.max_sil_kept_frames = round(sr * max_sil_kept_ms / 1000.0 / self.hop_size)

    def slice(self, waveform: np.ndarray) -> list[tuple[int, int]]:
        if waveform.ndim > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        rms_list = _get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        total_frames = int(rms_list.shape[0])

        sil_tags: list[tuple[int, int]] = []
        silence_start: int | None = None
        clip_start = 0

        for i, rms in enumerate(rms_list):
            if float(rms) < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue

            if silence_start is None:
                continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept_frames
            need_slice_middle = (
                i - silence_start >= self.min_interval_frames and i - clip_start >= self.min_length_frames
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept_frames:
                pos = int(rms_list[silence_start : i + 1].argmin()) + silence_start
                sil_tags.append((0, pos) if silence_start == 0 else (pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept_frames * 2:
                pos = int(rms_list[i - self.max_sil_kept_frames : silence_start + self.max_sil_kept_frames + 1].argmin())
                pos += i - self.max_sil_kept_frames
                pos_l = int(rms_list[silence_start : silence_start + self.max_sil_kept_frames + 1].argmin()) + silence_start
                pos_r = int(rms_list[i - self.max_sil_kept_frames : i + 1].argmin()) + i - self.max_sil_kept_frames
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = int(rms_list[silence_start : silence_start + self.max_sil_kept_frames + 1].argmin()) + silence_start
                pos_r = int(rms_list[i - self.max_sil_kept_frames : i + 1].argmin()) + i - self.max_sil_kept_frames
                sil_tags.append((0, pos_r) if silence_start == 0 else (pos_l, pos_r))
                clip_start = pos_r

            silence_start = None

        if silence_start is not None and total_frames - silence_start >= self.min_interval_frames:
            silence_end = min(total_frames, silence_start + self.max_sil_kept_frames)
            pos = int(rms_list[silence_start : silence_end + 1].argmin()) + silence_start
            sil_tags.append((pos, total_frames + 1))

        if not sil_tags:
            return [(0, int(samples.shape[0]))]

        segments: list[tuple[int, int]] = []
        if sil_tags[0][0] > 0:
            segments.append((0, int(sil_tags[0][0] * self.hop_size)))
        for i in range(len(sil_tags) - 1):
            segments.append(
                (
                    int(sil_tags[i][1] * self.hop_size),
                    int(sil_tags[i + 1][0] * self.hop_size),
                )
            )
        if sil_tags[-1][1] < total_frames:
            segments.append((int(sil_tags[-1][1] * self.hop_size), int(total_frames * self.hop_size)))

        max_sample = int(samples.shape[0])
        out: list[tuple[int, int]] = []
        for start, end in segments:
            s = max(0, min(int(start), max_sample))
            e = max(0, min(int(end), max_sample))
            if e > s:
                out.append((s, e))
        return out


def _split_by_max_len(
    *,
    wav: np.ndarray,
    start_sample: int,
    end_sample: int,
    max_segment_samples: int,
) -> list[AudioChunk]:
    out: list[AudioChunk] = []
    for s in range(int(start_sample), int(end_sample), int(max_segment_samples)):
        e = min(s + int(max_segment_samples), int(end_sample))
        if e > s:
            out.append(AudioChunk(start_sample=s, end_sample=e, wav=wav[s:e]))
    return out


def load_and_split_silence(
    *,
    file_path: str,
    max_segment_s: int = 300,
    threshold_db: float = -40.0,
    min_length_ms: int = 5000,
    min_interval_ms: int = 300,
    hop_size_ms: int = 20,
    max_sil_kept_ms: int = 5000,
) -> tuple[list[AudioChunk], bool]:
    """Load audio and split by silence; hard-limit each chunk within `max_segment_s`.

    Returns:
      (chunks, used_split)
    """

    wav = load_audio(file_path)
    duration_s = len(wav) / float(WAV_SAMPLE_RATE)

    max_segment_s = max(1, int(max_segment_s))
    max_segment_samples = int(max_segment_s) * WAV_SAMPLE_RATE

    logger.info(
        "静音切分参数: duration=%.2fs, max_segment=%ss, threshold_db=%.1f, "
        "min_interval=%dms, max_sil_kept=%dms",
        duration_s,
        max_segment_s,
        float(threshold_db),
        int(min_interval_ms),
        int(max_sil_kept_ms),
    )

    if len(wav) <= max_segment_samples:
        return [AudioChunk(start_sample=0, end_sample=len(wav), wav=wav)], False

    slicer = SilenceSlicer(
        sr=WAV_SAMPLE_RATE,
        threshold_db=float(threshold_db),
        min_length_ms=int(min_length_ms),
        min_interval_ms=int(min_interval_ms),
        hop_size_ms=int(hop_size_ms),
        max_sil_kept_ms=int(max_sil_kept_ms),
    )
    segments = slicer.slice(wav)

    chunks: list[AudioChunk] = []
    for start, end in segments:
        if end - start <= max_segment_samples:
            chunks.append(AudioChunk(start_sample=start, end_sample=end, wav=wav[start:end]))
            continue
        chunks.extend(
            _split_by_max_len(
                wav=wav,
                start_sample=start,
                end_sample=end,
                max_segment_samples=max_segment_samples,
            )
        )

    if not chunks:
        chunks = [AudioChunk(start_sample=0, end_sample=len(wav), wav=wav)]

    used_split = len(chunks) > 1
    logger.info("静音切分完成: chunks=%d", len(chunks))
    return chunks, used_split

