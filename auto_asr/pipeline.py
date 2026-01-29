from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from threading import local as thread_local
from typing import Any

from auto_asr.audio_tools import load_audio, process_vad_speech, transcode_wav_to_mp3
from auto_asr.funasr_asr import release_funasr_resources, transcribe_file_funasr
from auto_asr.funasr_models import is_funasr_nano
from auto_asr.openai_asr import make_openai_client, transcribe_file_verbose
from auto_asr.subtitles import SubtitleLine, compose_srt, compose_txt, compose_vtt
from auto_asr.vad_split import (
    WAV_SAMPLE_RATE,
    get_vad_model,
    load_and_split,
    save_audio_file,
)

logger = logging.getLogger(__name__)


def _resolve_funasr_device(device: str) -> str:
    d = (device or "").strip().lower()
    if d in {"", "auto"}:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"
    return device


@dataclass(frozen=True)
class PipelineResult:
    preview_text: str
    full_text: str
    subtitle_file_path: str
    debug: str


def _safe_stem(path: str) -> str:
    stem = Path(path).stem
    # Avoid empty/odd filenames in outputs.
    return stem if stem else "audio"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _check_cancel(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("已停止转写。")


def transcribe_to_subtitles(
    *,
    input_audio_path: str,
    asr_backend: str = "openai",
    openai_api_key: str = "",
    openai_base_url: str | None = None,
    output_format: str,
    model: str = "whisper-1",
    language: str | None = None,
    prompt: str | None = None,
    # FunASR local inference
    funasr_model: str = "iic/SenseVoiceSmall",
    funasr_device: str = "auto",
    funasr_language: str = "auto",
    funasr_use_itn: bool = True,
    funasr_enable_vad: bool = True,
    funasr_enable_punc: bool = True,
    enable_vad: bool = True,
    vad_segment_threshold_s: int = 120,
    vad_max_segment_threshold_s: int = 180,
    vad_threshold: float = 0.5,
    vad_min_speech_duration_ms: int = 200,
    vad_min_silence_duration_ms: int = 200,
    vad_speech_pad_ms: int = 200,
    timeline_strategy: str = "vad_speech",
    vad_speech_max_utterance_s: int = 20,
    vad_speech_merge_gap_ms: int = 300,
    upload_audio_format: str = "wav",
    upload_mp3_bitrate_kbps: int = 192,
    api_concurrency: int = 4,
    outputs_dir: str = "outputs",
    cancel_event: Event | None = None,
) -> PipelineResult:
    if output_format not in {"srt", "vtt", "txt"}:
        raise ValueError("output_format must be one of: srt, vtt, txt")
    if asr_backend not in {"openai", "funasr"}:
        raise ValueError("asr_backend must be one of: openai, funasr")
    if timeline_strategy not in {"chunk", "vad_speech"}:
        raise ValueError("timeline_strategy must be one of: chunk, vad_speech")
    if upload_audio_format not in {"wav", "mp3"}:
        raise ValueError("upload_audio_format must be one of: wav, mp3")
    api_concurrency = max(1, int(api_concurrency))

    logger.info(
        "开始转写: backend=%s, file=%s, format=%s, model=%s, language=%s, vad=%s, "
        "timeline_strategy=%s, upload_audio_format=%s, vad_threshold=%.2f, "
        "vad_min_speech_duration_ms=%d, vad_min_silence_duration_ms=%d, vad_speech_pad_ms=%d, "
        "api_concurrency=%d",
        asr_backend,
        input_audio_path,
        output_format,
        model,
        language or "auto",
        enable_vad,
        timeline_strategy,
        upload_audio_format,
        float(vad_threshold),
        int(vad_min_speech_duration_ms),
        int(vad_min_silence_duration_ms),
        int(vad_speech_pad_ms),
        api_concurrency,
    )
    _check_cancel(cancel_event)

    if asr_backend == "funasr":
        try:
            _check_cancel(cancel_event)
            wav_for_duration = load_audio(input_audio_path)
            duration_s = len(wav_for_duration) / float(WAV_SAMPLE_RATE)

            resolved_device = _resolve_funasr_device(funasr_device)
            lang = (funasr_language or "").strip() or "auto"
            if language:
                # if user set language in UI, prefer it over funasr_language
                lang = language
            _check_cancel(cancel_event)

            # FunASR-Nano 长音频如果整段推理, 容易因注意力矩阵过大导致 CUDA OOM.
            # 当用户启用了 VAD 语音段时间轴策略时, 直接走 VAD 语音段逐段转写以避免 OOM.
            if (
                is_funasr_nano(funasr_model)
                and enable_vad
                and timeline_strategy == "vad_speech"
                and duration_s > float(vad_speech_max_utterance_s)
            ):
                vad_model = get_vad_model()
                if vad_model is None:
                    logger.info(
                        "FunASR-Nano 长音频检测到但 VAD 模型不可用，将尝试整段推理(可能 OOM)。"
                    )
                else:
                    regions = process_vad_speech(
                        wav_for_duration,
                        vad_model,
                        max_utterance_s=int(vad_speech_max_utterance_s),
                        merge_gap_ms=int(vad_speech_merge_gap_ms),
                        vad_threshold=float(vad_threshold),
                        vad_min_speech_duration_ms=int(vad_min_speech_duration_ms),
                        vad_min_silence_duration_ms=int(vad_min_silence_duration_ms),
                        vad_speech_pad_ms=int(vad_speech_pad_ms),
                    )
                    if regions:
                        subtitle_lines: list[SubtitleLine] = []
                        full_text_parts: list[str] = []

                        logger.info(
                            "FunASR-Nano 长音频启用 VAD 语音段模式以避免 OOM: regions=%d, "
                            "max_utterance=%ss, merge_gap=%dms",
                            len(regions),
                            int(vad_speech_max_utterance_s),
                            int(vad_speech_merge_gap_ms),
                        )

                        with TemporaryDirectory(prefix="auto-asr-funasr-") as tmp_dir:
                            for idx, (start_sample, end_sample, wav_region) in enumerate(regions):
                                _check_cancel(cancel_event)
                                region_wav_path = os.path.join(tmp_dir, f"region_{idx:06d}.wav")
                                save_audio_file(wav_region, region_wav_path)

                                region_dur_s = (end_sample - start_sample) / float(WAV_SAMPLE_RATE)
                                seg_asr = transcribe_file_funasr(
                                    file_path=region_wav_path,
                                    model=funasr_model,
                                    device=resolved_device,
                                    language=lang,
                                    use_itn=bool(funasr_use_itn),
                                    enable_vad=bool(funasr_enable_vad),
                                    enable_punc=bool(funasr_enable_punc),
                                    duration_s=region_dur_s,
                                )

                                seg_text = (seg_asr.text or "").strip()
                                if seg_text:
                                    full_text_parts.append(seg_text)

                                if output_format in {"srt", "vtt"}:
                                    offset_s = start_sample / float(WAV_SAMPLE_RATE)
                                    if seg_asr.segments:
                                        for seg in seg_asr.segments:
                                            subtitle_lines.append(
                                                SubtitleLine(
                                                    start_s=offset_s + seg.start_s,
                                                    end_s=offset_s + seg.end_s,
                                                    text=seg.text,
                                                )
                                            )
                                    else:
                                        subtitle_lines.append(
                                            SubtitleLine(
                                                start_s=offset_s,
                                                end_s=offset_s + max(region_dur_s, 0.01),
                                                text=seg_text or seg_asr.text,
                                            )
                                        )

                        subtitle_lines.sort(key=lambda x: (x.start_s, x.end_s))
                        full_text = "\n".join([t for t in full_text_parts if t]).strip()

                        if output_format == "srt":
                            subtitle_text = compose_srt(subtitle_lines)
                            ext = "srt"
                        elif output_format == "vtt":
                            subtitle_text = compose_vtt(subtitle_lines)
                            ext = "vtt"
                        else:
                            subtitle_text = compose_txt(full_text)
                            ext = "txt"

                        out_base = (
                            f"{_safe_stem(input_audio_path)}-{time.strftime('%Y%m%d-%H%M%S')}"
                        )
                        out_path = Path(outputs_dir) / f"{out_base}.{ext}"
                        _write_text(out_path, subtitle_text)

                        preview = subtitle_text[:5000]
                        seg_count = len(subtitle_lines) if output_format in {"srt", "vtt"} else 0
                        debug = (
                            f"backend=funasr, model={funasr_model}, device={resolved_device}, "
                            f"segments={seg_count}, duration_s={duration_s:.2f}, "
                            "vad_speech_fallback=on(force=nano)"
                        )
                        logger.info(
                            "转写完成(funasr/nano_vad_speech): out=%s, segments=%d, duration=%.2fs",
                            out_path,
                            seg_count,
                            duration_s,
                        )
                        return PipelineResult(
                            preview_text=preview,
                            full_text=full_text,
                            subtitle_file_path=str(out_path),
                            debug=debug,
                        )
                    logger.info("VAD 未检测到语音段，将尝试整段推理(可能 OOM)。")

            asr = transcribe_file_funasr(
                file_path=input_audio_path,
                model=funasr_model,
                device=resolved_device,
                language=lang,
                use_itn=bool(funasr_use_itn),
                enable_vad=bool(funasr_enable_vad),
                enable_punc=bool(funasr_enable_punc),
                duration_s=duration_s,
            )

            subtitle_lines: list[SubtitleLine] = []
            full_text_parts: list[str] = []
            used_vad_speech_fallback = False

            # If FunASR doesn't provide timestamps (segments=0), fall back to Silero VAD speech
            # regions to build a reliable subtitle time axis.
            if (
                output_format in {"srt", "vtt"}
                and (not asr.segments)
                and enable_vad
                and timeline_strategy == "vad_speech"
            ):
                _check_cancel(cancel_event)
                vad_model = get_vad_model()
                if vad_model is None:
                    logger.info("FunASR segments=0 且 VAD 模型不可用，降级为整段字幕。")
                else:
                    regions = process_vad_speech(
                        wav_for_duration,
                        vad_model,
                        max_utterance_s=int(vad_speech_max_utterance_s),
                        merge_gap_ms=int(vad_speech_merge_gap_ms),
                        vad_threshold=float(vad_threshold),
                        vad_min_speech_duration_ms=int(vad_min_speech_duration_ms),
                        vad_min_silence_duration_ms=int(vad_min_silence_duration_ms),
                        vad_speech_pad_ms=int(vad_speech_pad_ms),
                    )
                    if regions:
                        used_vad_speech_fallback = True
                        logger.info(
                            "FunASR segments=0，启用 VAD 时间轴回退: regions=%d, "
                            "max_utterance=%ss, merge_gap=%dms",
                            len(regions),
                            int(vad_speech_max_utterance_s),
                            int(vad_speech_merge_gap_ms),
                        )
                        with TemporaryDirectory(prefix="auto-asr-funasr-") as tmp_dir:
                            for idx, (start_sample, end_sample, wav_region) in enumerate(regions):
                                _check_cancel(cancel_event)
                                region_wav_path = os.path.join(tmp_dir, f"region_{idx:06d}.wav")
                                save_audio_file(wav_region, region_wav_path)

                                region_dur_s = (end_sample - start_sample) / float(WAV_SAMPLE_RATE)
                                seg_asr = transcribe_file_funasr(
                                    file_path=region_wav_path,
                                    model=funasr_model,
                                    device=resolved_device,
                                    language=lang,
                                    use_itn=bool(funasr_use_itn),
                                    enable_vad=bool(funasr_enable_vad),
                                    enable_punc=bool(funasr_enable_punc),
                                    duration_s=region_dur_s,
                                )

                                seg_text = (seg_asr.text or "").strip()
                                if seg_text:
                                    full_text_parts.append(seg_text)

                                offset_s = start_sample / float(WAV_SAMPLE_RATE)
                                if seg_asr.segments:
                                    for seg in seg_asr.segments:
                                        subtitle_lines.append(
                                            SubtitleLine(
                                                start_s=offset_s + seg.start_s,
                                                end_s=offset_s + seg.end_s,
                                                text=seg.text,
                                            )
                                        )
                                else:
                                    subtitle_lines.append(
                                        SubtitleLine(
                                            start_s=offset_s,
                                            end_s=offset_s + max(region_dur_s, 0.01),
                                            text=seg_text or seg_asr.text,
                                        )
                                    )
                    else:
                        logger.info("VAD 未检测到语音段，降级为整段字幕。")

            if used_vad_speech_fallback and not subtitle_lines:
                # If all VAD segments produced empty text, fall back to one-shot output.
                used_vad_speech_fallback = False

            if not used_vad_speech_fallback:
                if output_format in {"srt", "vtt"}:
                    if asr.segments:
                        for seg in asr.segments:
                            subtitle_lines.append(
                                SubtitleLine(start_s=seg.start_s, end_s=seg.end_s, text=seg.text)
                            )
                    else:
                        subtitle_lines.append(
                            SubtitleLine(start_s=0.0, end_s=max(duration_s, 0.01), text=asr.text)
                        )
                full_text = (asr.text or "").strip()
            else:
                subtitle_lines.sort(key=lambda x: (x.start_s, x.end_s))
                full_text = "\n".join([t for t in full_text_parts if t]).strip()

            subtitle_lines.sort(key=lambda x: (x.start_s, x.end_s))

            if output_format == "srt":
                subtitle_text = compose_srt(subtitle_lines)
                ext = "srt"
            elif output_format == "vtt":
                subtitle_text = compose_vtt(subtitle_lines)
                ext = "vtt"
            else:
                subtitle_text = compose_txt(full_text)
                ext = "txt"

            out_base = f"{_safe_stem(input_audio_path)}-{time.strftime('%Y%m%d-%H%M%S')}"
            out_path = Path(outputs_dir) / f"{out_base}.{ext}"
            _write_text(out_path, subtitle_text)

            preview = subtitle_text[:5000]
            seg_count = len(subtitle_lines) if output_format in {"srt", "vtt"} else 0
            debug = (
                f"backend=funasr, model={funasr_model}, device={resolved_device}, "
                f"segments={seg_count}, duration_s={duration_s:.2f}, "
                f"vad_speech_fallback={'on' if used_vad_speech_fallback else 'off'}"
            )
            logger.info(
                "转写完成(funasr): out=%s, segments=%d, duration=%.2fs, vad_speech_fallback=%s",
                out_path,
                seg_count,
                duration_s,
                used_vad_speech_fallback,
            )
            return PipelineResult(
                preview_text=preview,
                full_text=full_text,
                subtitle_file_path=str(out_path),
                debug=debug,
            )
        finally:
            try:
                release_funasr_resources()
            except Exception as e:  # pragma: no cover
                logger.info("FunASR 资源清理失败(忽略): %s", e)

    client = make_openai_client(api_key=openai_api_key, base_url=openai_base_url)

    # Speed optimization for "vad_speech" timeline strategy:
    # - do VAD once on the full waveform
    # - upload speech-region WAV directly (PCM_16) to avoid per-region MP3 transcode overhead
    #
    # This keeps subtitle axis accurate while significantly reducing local compute time.
    if output_format in {"srt", "vtt"} and timeline_strategy == "vad_speech" and enable_vad:
        _check_cancel(cancel_event)
        vad_model = get_vad_model()
        if vad_model is None:
            logger.info("VAD 模型不可用，降级为分段整段模式。")
        else:
            _check_cancel(cancel_event)
            wav = load_audio(input_audio_path)
            regions = process_vad_speech(
                wav,
                vad_model,
                max_utterance_s=int(vad_speech_max_utterance_s),
                merge_gap_ms=int(vad_speech_merge_gap_ms),
                vad_threshold=float(vad_threshold),
                vad_min_speech_duration_ms=int(vad_min_speech_duration_ms),
                vad_min_silence_duration_ms=int(vad_min_silence_duration_ms),
                vad_speech_pad_ms=int(vad_speech_pad_ms),
            )
            if regions:
                subtitle_lines: list[SubtitleLine] = []
                full_text_parts: list[str] = []
                total_segments = 0
                used_vad_speech = True
                used_vad = True

                logger.info(
                    "VAD 语音段模式(单次VAD): regions=%d, concurrency=%d, "
                    "max_utterance=%ss, merge_gap=%dms",
                    len(regions),
                    api_concurrency,
                    int(vad_speech_max_utterance_s),
                    int(vad_speech_merge_gap_ms),
                )

                _tl = thread_local()

                def _get_thread_client():
                    c = getattr(_tl, "client", None)
                    if c is None:
                        c = make_openai_client(api_key=openai_api_key, base_url=openai_base_url)
                        _tl.client = c
                    return c

                with TemporaryDirectory(prefix="auto-asr-") as tmp_dir:
                    results: dict[int, tuple[float, float, Any]] = {}

                    def _worker(
                        r_idx: int, r_start: int, r_end: int, r_wav: Any
                    ) -> tuple[int, float, float, Any]:
                        _check_cancel(cancel_event)
                        abs_start_s = r_start / float(WAV_SAMPLE_RATE)
                        abs_end_s = r_end / float(WAV_SAMPLE_RATE)

                        region_wav_path = os.path.join(tmp_dir, f"region_{r_idx:06d}.wav")
                        save_audio_file(r_wav, region_wav_path)

                        # For speed: always upload speech regions as WAV (PCM_16).
                        upload_path = region_wav_path

                        _check_cancel(cancel_event)
                        asr = transcribe_file_verbose(
                            _get_thread_client(),
                            file_path=upload_path,
                            model=model,
                            language=language,
                            prompt=prompt,
                        )
                        return r_idx, abs_start_s, abs_end_s, asr

                    tasks = [(i, s, e, w) for i, (s, e, w) in enumerate(regions)]
                    ex = ThreadPoolExecutor(max_workers=api_concurrency)
                    futures = [ex.submit(_worker, *t) for t in tasks]
                    cancelled = False
                    try:
                        for fut in as_completed(futures):
                            if cancel_event is not None and cancel_event.is_set():
                                cancelled = True
                                break
                            try:
                                r_idx, abs_start_s, abs_end_s, asr = fut.result()
                            except Exception as e:
                                raise RuntimeError(f"语音段并发转写失败：{e}") from e

                            results[int(r_idx)] = (abs_start_s, abs_end_s, asr)
                            logger.info(
                                "语音段 %d/%d 完成: text_len=%d, start=%.2fs end=%.2fs",
                                r_idx + 1,
                                len(regions),
                                len(getattr(asr, "text", "") or ""),
                                abs_start_s,
                                abs_end_s,
                            )
                    finally:
                        if cancelled:
                            for f in futures:
                                f.cancel()
                            ex.shutdown(wait=False, cancel_futures=True)
                        else:
                            ex.shutdown(wait=True, cancel_futures=True)

                    _check_cancel(cancel_event)

                    for r_idx in range(len(regions)):
                        abs_start_s, abs_end_s, asr = results[r_idx]
                        # Preserve chronological text order (matches region order).
                        full_text_parts.append(asr.text.strip())

                        if asr.segments:
                            for seg in asr.segments:
                                subtitle_lines.append(
                                    SubtitleLine(
                                        start_s=abs_start_s + seg.start_s,
                                        end_s=abs_start_s + seg.end_s,
                                        text=seg.text,
                                    )
                                )
                            total_segments += len(asr.segments)
                        else:
                            subtitle_lines.append(
                                SubtitleLine(
                                    start_s=abs_start_s,
                                    end_s=abs_end_s,
                                    text=asr.text,
                                )
                            )
                            total_segments += 1

                subtitle_lines.sort(key=lambda x: (x.start_s, x.end_s))

                full_text = "\n".join([t for t in full_text_parts if t]).strip()

                if output_format == "srt":
                    subtitle_text = compose_srt(subtitle_lines)
                    ext = "srt"
                else:
                    subtitle_text = compose_vtt(subtitle_lines)
                    ext = "vtt"

                out_base = f"{_safe_stem(input_audio_path)}-{time.strftime('%Y%m%d-%H%M%S')}"
                out_path = Path(outputs_dir) / f"{out_base}.{ext}"
                _write_text(out_path, subtitle_text)

                preview = subtitle_text[:5000]
                debug = (
                    f"regions={len(regions)}, segments={total_segments}, "
                    f"vad=on(used={used_vad}), vad_speech_used={used_vad_speech}, "
                    f"vad_threshold={float(vad_threshold):.2f}, "
                    f"vad_min_speech_duration_ms={int(vad_min_speech_duration_ms)}, "
                    f"vad_min_silence_duration_ms={int(vad_min_silence_duration_ms)}, "
                    f"vad_speech_pad_ms={int(vad_speech_pad_ms)}, "
                    f"vad_speech_max_utterance_s={int(vad_speech_max_utterance_s)}, "
                    f"vad_speech_merge_gap_ms={int(vad_speech_merge_gap_ms)}, "
                    f"timeline_strategy={timeline_strategy}, upload_audio_format=wav, "
                    f"api_concurrency={api_concurrency}"
                )
                logger.info(
                    "转写完成(vad_speech): out=%s, regions=%d, segments=%d",
                    out_path,
                    len(regions),
                    total_segments,
                )
                return PipelineResult(
                    preview_text=preview,
                    full_text=full_text,
                    subtitle_file_path=str(out_path),
                    debug=debug,
                )

            logger.info("VAD 未检测到语音段，降级为分段整段模式。")

    _check_cancel(cancel_event)
    chunks, used_vad = load_and_split(
        file_path=input_audio_path,
        enable_vad=enable_vad,
        vad_segment_threshold_s=vad_segment_threshold_s,
        vad_max_segment_threshold_s=vad_max_segment_threshold_s,
        vad_threshold=float(vad_threshold),
        vad_min_speech_duration_ms=int(vad_min_speech_duration_ms),
        vad_min_silence_duration_ms=int(vad_min_silence_duration_ms),
        vad_speech_pad_ms=int(vad_speech_pad_ms),
    )

    subtitle_lines: list[SubtitleLine] = []
    full_text_parts: list[str] = []
    total_segments = 0
    used_vad_speech = False

    logger.info("分段信息: chunks=%d, used_vad=%s", len(chunks), used_vad)

    with TemporaryDirectory(prefix="auto-asr-") as tmp_dir:
        vad_model = get_vad_model() if enable_vad and timeline_strategy == "vad_speech" else None
        for idx, chunk in enumerate(chunks):
            _check_cancel(cancel_event)
            logger.info(
                "处理分段 %d/%d: start=%.2fs end=%.2fs duration=%.2fs",
                idx + 1,
                len(chunks),
                chunk.start_s,
                chunk.end_s,
                chunk.duration_s,
            )

            # When upstream doesn't return segments, we can improve subtitle time axis by
            # transcribing VAD speech regions and using VAD timestamps as SRT/VTT axis.
            #
            # Note: this increases API calls a lot (one call per speech region).
            if (
                output_format in {"srt", "vtt"}
                and timeline_strategy == "vad_speech"
                and vad_model is not None
            ):
                _check_cancel(cancel_event)
                regions = process_vad_speech(
                    chunk.wav,
                    vad_model,
                    max_utterance_s=int(vad_speech_max_utterance_s),
                    merge_gap_ms=int(vad_speech_merge_gap_ms),
                    vad_threshold=float(vad_threshold),
                    vad_min_speech_duration_ms=int(vad_min_speech_duration_ms),
                    vad_min_silence_duration_ms=int(vad_min_silence_duration_ms),
                    vad_speech_pad_ms=int(vad_speech_pad_ms),
                )
                if regions:
                    used_vad_speech = True
                    logger.info(
                        "VAD 语音段模式: regions=%d, max_utterance=%ss, merge_gap=%dms",
                        len(regions),
                        int(vad_speech_max_utterance_s),
                        int(vad_speech_merge_gap_ms),
                    )
                    for r_idx, (r_start, r_end, r_wav) in enumerate(regions):
                        _check_cancel(cancel_event)
                        abs_start_s = (chunk.start_sample + r_start) / float(WAV_SAMPLE_RATE)
                        abs_end_s = (chunk.start_sample + r_end) / float(WAV_SAMPLE_RATE)
                        region_wav_path = os.path.join(tmp_dir, f"region_{idx:04d}_{r_idx:04d}.wav")
                        save_audio_file(r_wav, region_wav_path)
                        upload_path = region_wav_path
                        if upload_audio_format == "mp3":
                            upload_path = os.path.join(tmp_dir, f"region_{idx:04d}_{r_idx:04d}.mp3")
                            try:
                                transcode_wav_to_mp3(
                                    input_wav_path=region_wav_path,
                                    output_mp3_path=upload_path,
                                    bitrate_kbps=int(upload_mp3_bitrate_kbps),
                                )
                            except Exception as e:
                                logger.info("MP3 转码失败，改用 WAV 上传: %s", e)
                                upload_path = region_wav_path

                        try:
                            size_bytes = os.path.getsize(upload_path)
                            logger.info(
                                "语音段 %d/%d 文件大小: %.2f MiB (%d bytes)",
                                r_idx + 1,
                                len(regions),
                                size_bytes / 1024.0 / 1024.0,
                                size_bytes,
                            )
                        except Exception:
                            pass

                        _check_cancel(cancel_event)
                        asr = transcribe_file_verbose(
                            client,
                            file_path=upload_path,
                            model=model,
                            language=language,
                            prompt=prompt,
                        )
                        full_text_parts.append(asr.text.strip())

                        if asr.segments:
                            for seg in asr.segments:
                                subtitle_lines.append(
                                    SubtitleLine(
                                        start_s=abs_start_s + seg.start_s,
                                        end_s=abs_start_s + seg.end_s,
                                        text=seg.text,
                                    )
                                )
                            total_segments += len(asr.segments)
                            logger.info(
                                "语音段 %d/%d 完成: segments=%d, text_len=%d, "
                                "start=%.2fs end=%.2fs",
                                r_idx + 1,
                                len(regions),
                                len(asr.segments),
                                len(asr.text or ""),
                                abs_start_s,
                                abs_end_s,
                            )
                        else:
                            subtitle_lines.append(
                                SubtitleLine(
                                    start_s=abs_start_s,
                                    end_s=abs_end_s,
                                    text=asr.text,
                                )
                            )
                            total_segments += 1
                            logger.info(
                                "语音段 %d/%d 完成: segments=0(用VAD时间轴), text_len=%d, "
                                "start=%.2fs end=%.2fs",
                                r_idx + 1,
                                len(regions),
                                len(asr.text or ""),
                                abs_start_s,
                                abs_end_s,
                            )
                    continue
                logger.info("VAD 未检测到语音段，降级为整段转写。")

            _check_cancel(cancel_event)
            chunk_wav_path = os.path.join(tmp_dir, f"chunk_{idx:04d}.wav")
            save_audio_file(chunk.wav, chunk_wav_path)
            upload_path = chunk_wav_path
            if upload_audio_format == "mp3":
                upload_path = os.path.join(tmp_dir, f"chunk_{idx:04d}.mp3")
                try:
                    transcode_wav_to_mp3(
                        input_wav_path=chunk_wav_path,
                        output_mp3_path=upload_path,
                        bitrate_kbps=int(upload_mp3_bitrate_kbps),
                    )
                except Exception as e:
                    logger.info("MP3 转码失败，改用 WAV 上传: %s", e)
                    upload_path = chunk_wav_path

            try:
                size_bytes = os.path.getsize(upload_path)
                logger.info(
                    "分段文件大小: %.2f MiB (%d bytes)",
                    size_bytes / 1024.0 / 1024.0,
                    size_bytes,
                )
            except Exception:
                pass

            _check_cancel(cancel_event)
            asr = transcribe_file_verbose(
                client,
                file_path=upload_path,
                model=model,
                language=language,
                prompt=prompt,
            )

            full_text_parts.append(asr.text.strip())

            offset_s = chunk.start_sample / float(WAV_SAMPLE_RATE)
            if asr.segments:
                for seg in asr.segments:
                    subtitle_lines.append(
                        SubtitleLine(
                            start_s=offset_s + seg.start_s,
                            end_s=offset_s + seg.end_s,
                            text=seg.text,
                        )
                    )
                total_segments += len(asr.segments)
                logger.info(
                    "分段 %d/%d 完成: segments=%d, text_len=%d",
                    idx + 1,
                    len(chunks),
                    len(asr.segments),
                    len(asr.text or ""),
                )
            else:
                # Fallback: coarse chunk-level subtitle.
                subtitle_lines.append(
                    SubtitleLine(
                        start_s=chunk.start_s,
                        end_s=chunk.end_s,
                        text=asr.text,
                    )
                )
                total_segments += 1
                logger.info(
                    "分段 %d/%d 完成: segments=0(降级为整段), text_len=%d",
                    idx + 1,
                    len(chunks),
                    len(asr.text or ""),
                )

    _check_cancel(cancel_event)
    subtitle_lines.sort(key=lambda x: (x.start_s, x.end_s))

    full_text = "\n".join([t for t in full_text_parts if t]).strip()

    if output_format == "srt":
        subtitle_text = compose_srt(subtitle_lines)
        ext = "srt"
    elif output_format == "vtt":
        subtitle_text = compose_vtt(subtitle_lines)
        ext = "vtt"
    else:
        subtitle_text = compose_txt(full_text)
        ext = "txt"

    out_base = f"{_safe_stem(input_audio_path)}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_path = Path(outputs_dir) / f"{out_base}.{ext}"
    _write_text(out_path, subtitle_text)

    preview = subtitle_text[:5000]
    debug = (
        f"chunks={len(chunks)}, segments={total_segments}, "
        f"vad={'on' if enable_vad else 'off'}(used={used_vad}), "
        f"vad_speech_used={used_vad_speech}, "
        f"vad_segment_threshold_s={vad_segment_threshold_s}, "
        f"vad_max_segment_threshold_s={vad_max_segment_threshold_s}, "
        f"vad_threshold={float(vad_threshold):.2f}, "
        f"vad_min_speech_duration_ms={int(vad_min_speech_duration_ms)}, "
        f"vad_min_silence_duration_ms={int(vad_min_silence_duration_ms)}, "
        f"vad_speech_pad_ms={int(vad_speech_pad_ms)}, "
        f"timeline_strategy={timeline_strategy}, "
        f"upload_audio_format={upload_audio_format}"
    )
    logger.info(
        "转写完成: out=%s, chunks=%d, segments=%d, used_vad=%s, vad_speech=%s",
        out_path,
        len(chunks),
        total_segments,
        used_vad,
        used_vad_speech,
    )
    return PipelineResult(
        preview_text=preview,
        full_text=full_text,
        subtitle_file_path=str(out_path),
        debug=debug,
    )
