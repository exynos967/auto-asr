from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from auto_asr.audio_tools import process_vad_speech, transcode_wav_to_mp3
from auto_asr.openai_asr import make_openai_client, transcribe_file_verbose
from auto_asr.subtitles import SubtitleLine, compose_srt, compose_txt, compose_vtt
from auto_asr.vad_split import (
    WAV_SAMPLE_RATE,
    get_vad_model,
    load_and_split,
    save_audio_file,
)

logger = logging.getLogger(__name__)


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


def transcribe_to_subtitles(
    *,
    input_audio_path: str,
    openai_api_key: str,
    openai_base_url: str | None = None,
    output_format: str,
    model: str = "whisper-1",
    language: str | None = None,
    prompt: str | None = None,
    enable_vad: bool = True,
    vad_segment_threshold_s: int = 120,
    vad_max_segment_threshold_s: int = 180,
    timeline_strategy: str = "vad_speech",
    vad_speech_max_utterance_s: int = 20,
    vad_speech_merge_gap_ms: int = 300,
    upload_audio_format: str = "mp3",
    upload_mp3_bitrate_kbps: int = 64,
    outputs_dir: str = "outputs",
) -> PipelineResult:
    if output_format not in {"srt", "vtt", "txt"}:
        raise ValueError("output_format must be one of: srt, vtt, txt")
    if timeline_strategy not in {"chunk", "vad_speech"}:
        raise ValueError("timeline_strategy must be one of: chunk, vad_speech")
    if upload_audio_format not in {"wav", "mp3"}:
        raise ValueError("upload_audio_format must be one of: wav, mp3")

    logger.info(
        "开始转写: file=%s, format=%s, model=%s, language=%s, vad=%s, "
        "timeline_strategy=%s, upload_audio_format=%s",
        input_audio_path,
        output_format,
        model,
        language or "auto",
        enable_vad,
        timeline_strategy,
        upload_audio_format,
    )

    client = make_openai_client(api_key=openai_api_key, base_url=openai_base_url)

    chunks, used_vad = load_and_split(
        file_path=input_audio_path,
        enable_vad=enable_vad,
        vad_segment_threshold_s=vad_segment_threshold_s,
        vad_max_segment_threshold_s=vad_max_segment_threshold_s,
    )

    subtitle_lines: list[SubtitleLine] = []
    full_text_parts: list[str] = []
    total_segments = 0
    used_vad_speech = False

    logger.info("分段信息: chunks=%d, used_vad=%s", len(chunks), used_vad)

    with TemporaryDirectory(prefix="auto-asr-") as tmp_dir:
        vad_model = get_vad_model() if enable_vad and timeline_strategy == "vad_speech" else None
        for idx, chunk in enumerate(chunks):
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
                regions = process_vad_speech(
                    chunk.wav,
                    vad_model,
                    max_utterance_s=int(vad_speech_max_utterance_s),
                    merge_gap_ms=int(vad_speech_merge_gap_ms),
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
                        abs_start_s = (chunk.start_sample + r_start) / float(WAV_SAMPLE_RATE)
                        abs_end_s = (chunk.start_sample + r_end) / float(WAV_SAMPLE_RATE)
                        region_wav_path = os.path.join(
                            tmp_dir, f"region_{idx:04d}_{r_idx:04d}.wav"
                        )
                        save_audio_file(r_wav, region_wav_path)
                        upload_path = region_wav_path
                        if upload_audio_format == "mp3":
                            upload_path = os.path.join(
                                tmp_dir, f"region_{idx:04d}_{r_idx:04d}.mp3"
                            )
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
