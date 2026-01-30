from __future__ import annotations

import logging
from pathlib import Path
from threading import Event, Lock

import gradio as gr

from auto_asr.config import get_config_path, load_config, update_config
from auto_asr.funasr_asr import (
    download_funasr_model,
    preload_funasr_model,
    release_funasr_resources,
)
from auto_asr.model_hub import get_models_dir
from auto_asr.pipeline import transcribe_to_subtitles
from auto_asr.qwen3_asr import (
    Qwen3ASRConfig,
    download_qwen3_models,
    preload_qwen3_model,
    release_qwen3_resources,
)
from auto_asr.subtitle_processing.pipeline import (
    process_subtitle_file,
    process_subtitle_file_multi,
)
from auto_asr.subtitle_processing.settings import (
    save_subtitle_processing_settings,
    save_subtitle_provider_settings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_SAVED_CONFIG = load_config()
_CONFIG_PATH = get_config_path()


def _str(v: object | None) -> str:
    return "" if v is None else str(v)


def _int(v: object | None, default: int) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


DEFAULT_OPENAI_API_KEY = _str(_SAVED_CONFIG.get("openai_api_key")).strip()
DEFAULT_OPENAI_BASE_URL = _str(_SAVED_CONFIG.get("openai_base_url")).strip()
DEFAULT_MODEL = _str(_SAVED_CONFIG.get("model", "whisper-1")).strip() or "whisper-1"
DEFAULT_ASR_BACKEND = _str(_SAVED_CONFIG.get("asr_backend", "qwen3asr")).strip() or "qwen3asr"
if DEFAULT_ASR_BACKEND not in {"openai", "funasr", "qwen3asr"}:
    DEFAULT_ASR_BACKEND = "qwen3asr"

DEFAULT_SUBTITLE_PROVIDER = (
    _str(_SAVED_CONFIG.get("subtitle_provider", "openai")).strip() or "openai"
)
DEFAULT_SUBTITLE_OPENAI_API_KEY = _str(
    _SAVED_CONFIG.get("subtitle_openai_api_key", DEFAULT_OPENAI_API_KEY)
).strip()
DEFAULT_SUBTITLE_OPENAI_BASE_URL = _str(
    _SAVED_CONFIG.get("subtitle_openai_base_url", DEFAULT_OPENAI_BASE_URL)
).strip()
DEFAULT_SUBTITLE_LLM_MODEL = _str(
    _SAVED_CONFIG.get("subtitle_llm_model", _SAVED_CONFIG.get("llm_model", "gpt-4o-mini"))
).strip() or "gpt-4o-mini"
try:
    DEFAULT_SUBTITLE_LLM_TEMPERATURE = float(_SAVED_CONFIG.get("subtitle_llm_temperature", 0.2))
except Exception:
    DEFAULT_SUBTITLE_LLM_TEMPERATURE = 0.2
DEFAULT_SUBTITLE_LLM_TEMPERATURE = max(0.0, min(2.0, DEFAULT_SUBTITLE_LLM_TEMPERATURE))

DEFAULT_SUBTITLE_TARGET_LANGUAGE = _str(_SAVED_CONFIG.get("subtitle_target_language", "zh")).strip()
if DEFAULT_SUBTITLE_TARGET_LANGUAGE not in {"zh", "en", "ja", "ko", "fr", "de", "es", "ru"}:
    DEFAULT_SUBTITLE_TARGET_LANGUAGE = "zh"
DEFAULT_SUBTITLE_SPLIT_MODE = _str(
    _SAVED_CONFIG.get("subtitle_split_mode", "inplace_newlines")
).strip() or "inplace_newlines"
if DEFAULT_SUBTITLE_SPLIT_MODE not in {"inplace_newlines", "split_to_cues"}:
    DEFAULT_SUBTITLE_SPLIT_MODE = "inplace_newlines"
DEFAULT_SUBTITLE_CUSTOM_PROMPT = _str(_SAVED_CONFIG.get("subtitle_custom_prompt", ""))
DEFAULT_SUBTITLE_BATCH_SIZE = _clamp_int(_int(_SAVED_CONFIG.get("subtitle_batch_size"), 30), 1, 200)
DEFAULT_SUBTITLE_CONCURRENCY = _clamp_int(_int(_SAVED_CONFIG.get("subtitle_concurrency"), 4), 1, 16)

_saved_subtitle_processors = _SAVED_CONFIG.get("subtitle_processors", ["optimize"])
if isinstance(_saved_subtitle_processors, list):
    DEFAULT_SUBTITLE_PROCESSORS = [
        str(x)
        for x in _saved_subtitle_processors
        if str(x) in {"optimize", "translate", "split"}
    ] or ["optimize"]
else:
    DEFAULT_SUBTITLE_PROCESSORS = ["optimize"]
DEFAULT_SUBTITLE_SPLIT_STRATEGY = (
    _str(_SAVED_CONFIG.get("subtitle_split_strategy", "semantic")).strip() or "semantic"
)
if DEFAULT_SUBTITLE_SPLIT_STRATEGY not in {"semantic", "sentence"}:
    DEFAULT_SUBTITLE_SPLIT_STRATEGY = "semantic"

DEFAULT_SUBTITLE_SPLIT_MAX_WORD_COUNT_CJK = _clamp_int(
    _int(_SAVED_CONFIG.get("subtitle_split_max_word_count_cjk"), 18), 1, 200
)
DEFAULT_SUBTITLE_SPLIT_MAX_WORD_COUNT_ENGLISH = _clamp_int(
    _int(_SAVED_CONFIG.get("subtitle_split_max_word_count_english"), 12), 1, 200
)
DEFAULT_SUBTITLE_TRANSLATE_REFLECT = bool(_SAVED_CONFIG.get("subtitle_translate_reflect", False))

DEFAULT_FUNASR_MODEL = _str(_SAVED_CONFIG.get("funasr_model", "iic/SenseVoiceSmall")).strip()
DEFAULT_FUNASR_DEVICE = _str(_SAVED_CONFIG.get("funasr_device", "auto")).strip() or "auto"
if DEFAULT_FUNASR_DEVICE not in {"auto", "cpu", "cuda:0"}:
    DEFAULT_FUNASR_DEVICE = "auto"
DEFAULT_FUNASR_LANGUAGE = _str(_SAVED_CONFIG.get("funasr_language", "auto")).strip() or "auto"
DEFAULT_FUNASR_USE_ITN = bool(_SAVED_CONFIG.get("funasr_use_itn", True))
DEFAULT_FUNASR_ENABLE_VAD = bool(_SAVED_CONFIG.get("funasr_enable_vad", False))
DEFAULT_FUNASR_ENABLE_PUNC = bool(_SAVED_CONFIG.get("funasr_enable_punc", True))

DEFAULT_QWEN3_MODEL = _str(_SAVED_CONFIG.get("qwen3_model", "Qwen/Qwen3-ASR-1.7B")).strip()
DEFAULT_QWEN3_FORCED_ALIGNER = _str(
    _SAVED_CONFIG.get("qwen3_forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B")
).strip()
QWEN3_ASR_HF_URL = "https://huggingface.co/Qwen/Qwen3-ASR-1.7B"
QWEN3_FORCED_ALIGNER_HF_URL = "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_QWEN3_USE_FORCED_ALIGNER = bool(_SAVED_CONFIG.get("qwen3_use_forced_aligner", True))
DEFAULT_QWEN3_DEVICE = _str(_SAVED_CONFIG.get("qwen3_device", "auto")).strip() or "auto"
if DEFAULT_QWEN3_DEVICE not in {"auto", "cpu", "cuda:0"}:
    DEFAULT_QWEN3_DEVICE = "auto"
DEFAULT_QWEN3_MAX_INFERENCE_BATCH_SIZE = _clamp_int(
    _int(_SAVED_CONFIG.get("qwen3_max_inference_batch_size"), 8), 1, 64
)
DEFAULT_OUTPUT_FORMAT = _str(_SAVED_CONFIG.get("output_format", "srt")).strip() or "srt"
if DEFAULT_OUTPUT_FORMAT not in {"srt", "vtt", "txt"}:
    DEFAULT_OUTPUT_FORMAT = "srt"

DEFAULT_LANGUAGE = _str(_SAVED_CONFIG.get("language", "auto")).strip() or "auto"
if DEFAULT_LANGUAGE not in {"auto", "zh", "en", "ja", "ko", "fr", "de", "es", "ru"}:
    DEFAULT_LANGUAGE = "auto"
DEFAULT_ENABLE_VAD = bool(_SAVED_CONFIG.get("enable_vad", True))
DEFAULT_VAD_SEGMENT_THRESHOLD_S = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_segment_threshold_s"), 120), 30, 240
)
DEFAULT_VAD_MAX_SEGMENT_THRESHOLD_S = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_max_segment_threshold_s"), 180), 60, 360
)
try:
    DEFAULT_VAD_THRESHOLD = float(_SAVED_CONFIG.get("vad_threshold", 0.25))
except Exception:
    DEFAULT_VAD_THRESHOLD = 0.25
DEFAULT_VAD_THRESHOLD = max(0.1, min(0.9, DEFAULT_VAD_THRESHOLD))

DEFAULT_VAD_MIN_SPEECH_DURATION_MS = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_min_speech_duration_ms"), 100), 50, 2000
)
DEFAULT_VAD_MIN_SILENCE_DURATION_MS = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_min_silence_duration_ms"), 300), 50, 2000
)
DEFAULT_VAD_SPEECH_PAD_MS = _clamp_int(_int(_SAVED_CONFIG.get("vad_speech_pad_ms"), 400), 0, 2000)
DEFAULT_TIMELINE_STRATEGY = _str(_SAVED_CONFIG.get("timeline_strategy", "vad_speech")).strip()
if DEFAULT_TIMELINE_STRATEGY not in {"chunk", "vad_speech"}:
    DEFAULT_TIMELINE_STRATEGY = "vad_speech"

DEFAULT_UPLOAD_AUDIO_FORMAT = _str(_SAVED_CONFIG.get("upload_audio_format", "wav")).strip()
if DEFAULT_UPLOAD_AUDIO_FORMAT not in {"wav", "mp3"}:
    DEFAULT_UPLOAD_AUDIO_FORMAT = "wav"

UPLOAD_MP3_BITRATE_KBPS = 192

DEFAULT_VAD_SPEECH_MAX_UTTERANCE_S = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_speech_max_utterance_s"), 8), 5, 60
)
DEFAULT_VAD_SPEECH_MERGE_GAP_MS = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_speech_merge_gap_ms"), 100), 0, 2000
)
DEFAULT_API_CONCURRENCY = _clamp_int(_int(_SAVED_CONFIG.get("api_concurrency"), 4), 1, 16)
CONFIG_NOTE = f"配置文件：`{_CONFIG_PATH}`"


def _detect_cuda() -> tuple[bool, str]:
    try:
        import torch  # type: ignore
    except Exception as e:
        return False, f"torch 未安装：{e}"

    try:
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None) or "n/a"
        available = bool(torch.cuda.is_available())
        if not available:
            return False, f"torch={torch_version}, cuda={cuda_version}, available=False"

        count = int(torch.cuda.device_count())
        names: list[str] = []
        for i in range(min(count, 4)):
            try:
                names.append(str(torch.cuda.get_device_name(i)))
            except Exception:
                continue
        devices = ", ".join(names) if names else "unknown"
        return True, f"torch={torch_version}, cuda={cuda_version}, devices={count}, names={devices}"
    except Exception as e:
        return False, f"CUDA 检测异常：{e}"


CUDA_AVAILABLE, CUDA_DETAILS = _detect_cuda()
CUDA_NOTE = (
    f"CUDA 检测：{'可用' if CUDA_AVAILABLE else '不可用'}（{CUDA_DETAILS}）"
    if CUDA_DETAILS
    else "CUDA 检测：未知"
)


def _load_theme():
    """Load the bundled 'miku' Gradio theme if present; fallback to Soft()."""

    try:
        base = Path(__file__).resolve().parent
        candidates = [
            base / "theme" / "miku" / "theme_schema@1.2.2.json",
            base / "miku" / "themes" / "theme_schema@1.2.2.json",
        ]
        for theme_path in candidates:
            if theme_path.exists():
                return gr.Theme.load(str(theme_path))
    except Exception as e:
        logger.info("加载 miku 主题失败，回退默认主题: %s", e)
    return gr.themes.Soft()


THEME = _load_theme()

logger.info(
    "auto-asr 启动: config=%s, exists=%s, api_key_saved=%s",
    _CONFIG_PATH,
    _CONFIG_PATH.exists(),
    bool(DEFAULT_OPENAI_API_KEY),
)
logger.info("auto-asr CUDA 检测: available=%s, details=%s", CUDA_AVAILABLE, CUDA_DETAILS)

_CANCEL_LOCK = Lock()
_CURRENT_CANCEL_EVENT: Event | None = None


def _set_current_cancel_event(ev: Event | None) -> None:
    global _CURRENT_CANCEL_EVENT
    with _CANCEL_LOCK:
        _CURRENT_CANCEL_EVENT = ev


def stop_transcribe() -> str:
    with _CANCEL_LOCK:
        ev = _CURRENT_CANCEL_EVENT
    if ev is None:
        return "当前没有正在进行的转写。"
    ev.set()
    logger.info("收到停止转写请求。")
    return "已发送停止信号（后台会尽快停止）。"


def _resolve_funasr_device_ui(device: str) -> str:
    d = (device or "").strip()
    if d in {"", "auto"}:
        return "cuda:0" if CUDA_AVAILABLE else "cpu"
    return d


def _resolve_qwen3_device_ui(device: str) -> str:
    d = (device or "").strip()
    if d in {"", "auto"}:
        return "cuda:0" if CUDA_AVAILABLE else "cpu"
    return d


def load_funasr_model_ui(
    funasr_model: str,
    funasr_device: str,
    funasr_enable_vad: bool,
    funasr_enable_punc: bool,
) -> str:
    resolved_device = _resolve_funasr_device_ui(funasr_device)
    try:
        preload_funasr_model(
            model=(funasr_model or "").strip(),
            device=resolved_device,
            enable_vad=bool(funasr_enable_vad),
            enable_punc=bool(funasr_enable_punc),
        )
    except Exception as e:
        logger.exception("加载 FunASR 模型失败: model=%s, device=%s", funasr_model, resolved_device)
        return f"加载失败：{e}"

    return f"已加载 FunASR 模型：{(funasr_model or '').strip()}（device={resolved_device}）"


def download_funasr_model_ui(
    funasr_model: str,
    funasr_enable_vad: bool,
    funasr_enable_punc: bool,
) -> str:
    try:
        download_funasr_model(
            model=(funasr_model or "").strip(),
            enable_vad=bool(funasr_enable_vad),
            enable_punc=bool(funasr_enable_punc),
        )
    except Exception as e:
        logger.exception("下载 FunASR 模型失败: model=%s", funasr_model)
        return f"下载失败：{e}"

    return f"下载完成：模型文件已下载到项目目录 `{get_models_dir()}`。"


def load_qwen3_model_ui(
    qwen3_model: str,
    qwen3_forced_aligner: str,
    qwen3_device: str,
    qwen3_max_inference_batch_size: int,
    qwen3_use_forced_aligner: bool,
) -> str:
    resolved_device = _resolve_qwen3_device_ui(qwen3_device)
    use_forced_aligner = bool(qwen3_use_forced_aligner)
    try:
        preload_qwen3_model(
            cfg=Qwen3ASRConfig(
                model=(qwen3_model or "").strip() or "Qwen/Qwen3-ASR-1.7B",
                forced_aligner=(
                    (qwen3_forced_aligner or "").strip() or "Qwen/Qwen3-ForcedAligner-0.6B"
                )
                if use_forced_aligner
                else "",
                device=resolved_device,
                max_inference_batch_size=max(1, int(qwen3_max_inference_batch_size)),
            )
        )
    except Exception as e:
        logger.exception(
            "加载 Qwen3-ASR 模型失败: model=%s, aligner=%s, device=%s",
            qwen3_model,
            qwen3_forced_aligner,
            resolved_device,
        )
        return f"加载失败：{e}"

    model_name = (qwen3_model or "").strip()
    aligner_name = (qwen3_forced_aligner or "").strip()
    if use_forced_aligner:
        return f"已加载 Qwen3-ASR 模型：{model_name} + {aligner_name}（device={resolved_device}）"
    return f"已加载 Qwen3-ASR 模型：{model_name}（未启用 Forced Aligner；device={resolved_device}）"


def download_qwen3_models_ui(
    qwen3_model: str,
    qwen3_forced_aligner: str,
    qwen3_use_forced_aligner: bool,
) -> str:
    try:
        use_forced_aligner = bool(qwen3_use_forced_aligner)
        download_qwen3_models(
            model=(qwen3_model or "").strip() or "Qwen/Qwen3-ASR-1.7B",
            forced_aligner=(
                (qwen3_forced_aligner or "").strip() or "Qwen/Qwen3-ForcedAligner-0.6B"
            )
            if use_forced_aligner
            else "",
        )
    except Exception as e:
        logger.exception("下载 Qwen3-ASR 模型失败: model=%s", qwen3_model)
        return f"下载失败：{e}"

    return f"下载完成：模型文件已下载到项目目录 `{get_models_dir()}`。"


def release_cuda_ui() -> str:
    try:
        release_funasr_resources()
        release_qwen3_resources()
    except Exception as e:
        logger.exception("释放显存失败")
        return f"释放失败：{e}"
    return "已释放 FunASR/Qwen3 模型缓存/显存（如仍显示占用，通常是 PyTorch 缓存行为）。"


def _save_subtitle_provider_settings_ui(
    subtitle_provider: str,
    subtitle_openai_api_key: str,
    subtitle_openai_base_url: str,
    subtitle_llm_model: str,
    subtitle_llm_temperature: float,
    split_strategy: str,
):
    try:
        save_subtitle_provider_settings(
            provider=subtitle_provider,
            openai_api_key=subtitle_openai_api_key,
            openai_base_url=subtitle_openai_base_url,
            llm_model=subtitle_llm_model,
            llm_temperature=float(subtitle_llm_temperature),
            split_strategy=split_strategy,
        )
    except Exception as e:
        logger.exception("保存字幕处理 LLM 配置失败: %s", e)
        return f"保存字幕处理配置失败：{e}"
    return None


def _save_subtitle_processing_settings_ui(
    subtitle_processors: list[str] | None,
    batch_size: int,
    concurrency: int,
    target_language: str,
    translate_reflect: bool,
    split_mode: str,
    split_max_word_count_cjk: int,
    split_max_word_count_english: int,
    custom_prompt: str,
):
    try:
        save_subtitle_processing_settings(
            processors=subtitle_processors,
            batch_size=int(batch_size),
            concurrency=int(concurrency),
            target_language=target_language,
            translate_reflect=bool(translate_reflect),
            split_mode=split_mode,
            split_max_word_count_cjk=int(split_max_word_count_cjk),
            split_max_word_count_english=int(split_max_word_count_english),
            custom_prompt=custom_prompt,
        )
    except Exception as e:
        logger.exception("保存字幕处理参数失败: %s", e)
        return f"保存字幕处理参数失败：{e}"
    return None


def _auto_save_settings(
    *,
    asr_backend: str,
    openai_api_key: str,
    openai_base_url: str,
    model: str,
    funasr_model: str,
    funasr_device: str,
    funasr_language: str,
    funasr_use_itn: bool,
    funasr_enable_vad: bool,
    funasr_enable_punc: bool,
    qwen3_model: str,
    qwen3_forced_aligner: str,
    qwen3_device: str,
    qwen3_max_inference_batch_size: int,
    qwen3_use_forced_aligner: bool,
    output_format: str,
    language: str,
    enable_vad: bool,
    vad_segment_threshold_s: int,
    vad_max_segment_threshold_s: int,
    vad_threshold: float,
    vad_min_speech_duration_ms: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    timeline_strategy: str,
    vad_speech_max_utterance_s: int,
    vad_speech_merge_gap_ms: int,
    upload_audio_format: str,
    api_concurrency: int,
) -> None:
    api_key = (openai_api_key or "").strip()
    if not api_key:
        # Avoid wiping the saved key due to an accidental empty input.
        api_key = DEFAULT_OPENAI_API_KEY

    config = {
        "asr_backend": (asr_backend or "").strip() or "openai",
        "enable_vad": bool(enable_vad),
        "funasr_device": (funasr_device or "").strip() or "auto",
        "funasr_enable_punc": bool(funasr_enable_punc),
        "funasr_enable_vad": bool(funasr_enable_vad),
        "funasr_language": (funasr_language or "").strip() or "auto",
        "funasr_model": (funasr_model or "").strip() or "iic/SenseVoiceSmall",
        "funasr_use_itn": bool(funasr_use_itn),
        "language": (language or "").strip() or "auto",
        "model": (model or "").strip() or "whisper-1",
        "openai_api_key": api_key,
        "openai_base_url": (openai_base_url or "").strip(),
        "output_format": (output_format or "").strip() or "srt",
        "qwen3_device": (qwen3_device or "").strip() or "auto",
        "qwen3_forced_aligner": (qwen3_forced_aligner or "").strip()
        or "Qwen/Qwen3-ForcedAligner-0.6B",
        "qwen3_max_inference_batch_size": int(qwen3_max_inference_batch_size),
        "qwen3_model": (qwen3_model or "").strip() or "Qwen/Qwen3-ASR-1.7B",
        "qwen3_use_forced_aligner": bool(qwen3_use_forced_aligner),
        "timeline_strategy": (timeline_strategy or "").strip() or "vad_speech",
        "upload_audio_format": (upload_audio_format or "").strip() or "wav",
        "upload_mp3_bitrate_kbps": int(UPLOAD_MP3_BITRATE_KBPS),
        "vad_threshold": float(vad_threshold),
        "vad_min_speech_duration_ms": int(vad_min_speech_duration_ms),
        "vad_min_silence_duration_ms": int(vad_min_silence_duration_ms),
        "vad_speech_pad_ms": int(vad_speech_pad_ms),
        "vad_speech_max_utterance_s": int(vad_speech_max_utterance_s),
        "vad_speech_merge_gap_ms": int(vad_speech_merge_gap_ms),
        "vad_max_segment_threshold_s": int(vad_max_segment_threshold_s),
        "vad_segment_threshold_s": int(vad_segment_threshold_s),
        "api_concurrency": int(api_concurrency),
    }

    path = update_config(config)
    logger.info("配置已自动保存: path=%s", path)


def run_asr(
    audio_path: str | None,
    asr_backend: str,
    openai_api_key: str,
    openai_base_url: str,
    model: str,
    funasr_model: str,
    funasr_device: str,
    funasr_language: str,
    funasr_use_itn: bool,
    funasr_enable_vad: bool,
    funasr_enable_punc: bool,
    output_format: str,
    language: str,
    prompt: str,
    enable_vad: bool,
    vad_segment_threshold_s: int,
    vad_max_segment_threshold_s: int,
    vad_threshold: float,
    vad_min_speech_duration_ms: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    timeline_strategy: str,
    vad_speech_max_utterance_s: int,
    vad_speech_merge_gap_ms: int,
    upload_audio_format: str,
    api_concurrency: int,
    qwen3_model: str,
    qwen3_forced_aligner: str,
    qwen3_device: str,
    qwen3_max_inference_batch_size: int,
    qwen3_use_forced_aligner: bool,
):
    if not audio_path:
        raise gr.Error("请先上传或录制一段音频。")
    asr_backend = (asr_backend or "").strip() or "openai"
    if asr_backend == "openai" and not (openai_api_key or "").strip():
        raise gr.Error("请先填写 OpenAI API Key。")

    lang = None if language == "auto" else language
    prompt = (prompt or "").strip() or None
    model = (model or "").strip() or "whisper-1"
    base_url = (openai_base_url or "").strip() or None

    logger.info(
        "收到转写请求: backend=%s, file=%s, format=%s, language=%s, model=%s, base_url=%s, "
        "enable_vad=%s, target=%ss, max=%ss, timeline_strategy=%s, upload=%s, "
        "vad_threshold=%.2f, vad_min_speech_ms=%d, vad_min_silence_ms=%d, vad_pad_ms=%d, "
        "api_concurrency=%d",
        asr_backend,
        audio_path,
        output_format,
        lang or "auto",
        model,
        base_url or "(default)",
        enable_vad,
        vad_segment_threshold_s,
        vad_max_segment_threshold_s,
        timeline_strategy,
        f"{upload_audio_format}/{UPLOAD_MP3_BITRATE_KBPS}k",
        float(vad_threshold),
        int(vad_min_speech_duration_ms),
        int(vad_min_silence_duration_ms),
        int(vad_speech_pad_ms),
        int(api_concurrency),
    )

    _auto_save_settings(
        asr_backend=asr_backend,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        model=model,
        funasr_model=funasr_model,
        funasr_device=funasr_device,
        funasr_language=funasr_language,
        funasr_use_itn=funasr_use_itn,
        funasr_enable_vad=funasr_enable_vad,
        funasr_enable_punc=funasr_enable_punc,
        qwen3_model=qwen3_model,
        qwen3_forced_aligner=qwen3_forced_aligner,
        qwen3_device=qwen3_device,
        qwen3_max_inference_batch_size=qwen3_max_inference_batch_size,
        qwen3_use_forced_aligner=bool(qwen3_use_forced_aligner),
        output_format=output_format,
        language=language,
        enable_vad=enable_vad,
        vad_segment_threshold_s=vad_segment_threshold_s,
        vad_max_segment_threshold_s=vad_max_segment_threshold_s,
        vad_threshold=vad_threshold,
        vad_min_speech_duration_ms=vad_min_speech_duration_ms,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        timeline_strategy=timeline_strategy,
        vad_speech_max_utterance_s=vad_speech_max_utterance_s,
        vad_speech_merge_gap_ms=vad_speech_merge_gap_ms,
        upload_audio_format=upload_audio_format,
        api_concurrency=api_concurrency,
    )

    cancel_event = Event()
    _set_current_cancel_event(cancel_event)

    try:
        resolved_funasr_model = (funasr_model or "").strip() or DEFAULT_FUNASR_MODEL
        resolved_qwen3_model = (qwen3_model or "").strip() or DEFAULT_QWEN3_MODEL
        resolved_qwen3_forced_aligner = (
            (qwen3_forced_aligner or "").strip() or DEFAULT_QWEN3_FORCED_ALIGNER
        )
        resolved_qwen3_max_batch = _clamp_int(int(qwen3_max_inference_batch_size), 1, 64)
        resolved_qwen3_use_forced_aligner = bool(qwen3_use_forced_aligner)
        result = transcribe_to_subtitles(
            input_audio_path=audio_path,
            asr_backend=asr_backend,
            openai_api_key=openai_api_key,
            openai_base_url=base_url,
            output_format=output_format,
            model=model,
            language=lang,
            prompt=prompt,
            funasr_model=resolved_funasr_model,
            funasr_device=(funasr_device or "").strip() or DEFAULT_FUNASR_DEVICE,
            funasr_language=(funasr_language or "").strip() or DEFAULT_FUNASR_LANGUAGE,
            funasr_use_itn=bool(funasr_use_itn),
            funasr_enable_vad=bool(funasr_enable_vad),
            funasr_enable_punc=bool(funasr_enable_punc),
            qwen3_model=resolved_qwen3_model,
            qwen3_forced_aligner=resolved_qwen3_forced_aligner,
            qwen3_device=(qwen3_device or "").strip() or DEFAULT_QWEN3_DEVICE,
            qwen3_max_inference_batch_size=resolved_qwen3_max_batch,
            qwen3_use_forced_aligner=resolved_qwen3_use_forced_aligner,
            enable_vad=enable_vad,
            vad_segment_threshold_s=int(vad_segment_threshold_s),
            vad_max_segment_threshold_s=int(vad_max_segment_threshold_s),
            vad_threshold=float(vad_threshold),
            vad_min_speech_duration_ms=int(vad_min_speech_duration_ms),
            vad_min_silence_duration_ms=int(vad_min_silence_duration_ms),
            vad_speech_pad_ms=int(vad_speech_pad_ms),
            timeline_strategy=(timeline_strategy or "").strip() or "vad_speech",
            vad_speech_max_utterance_s=int(vad_speech_max_utterance_s),
            vad_speech_merge_gap_ms=int(vad_speech_merge_gap_ms),
            upload_audio_format=(upload_audio_format or "").strip() or "wav",
            upload_mp3_bitrate_kbps=int(UPLOAD_MP3_BITRATE_KBPS),
            api_concurrency=int(api_concurrency),
            cancel_event=cancel_event,
        )
    except Exception as e:
        if cancel_event.is_set() or "已停止转写" in str(e):
            logger.info("转写已停止: %s", e)
            return "", "", None, "已停止转写"
        raise gr.Error(f"转写失败：{e}") from e
    finally:
        _set_current_cancel_event(None)

    return result.preview_text, result.full_text, result.subtitle_file_path, result.debug


def run_subtitle_processing(
    subtitle_path: str | None,
    subtitle_processors: list[str] | None,
    subtitle_provider: str,
    subtitle_openai_api_key: str,
    subtitle_openai_base_url: str,
    subtitle_llm_model: str,
    subtitle_llm_temperature: float,
    target_language: str,
    translate_reflect: bool,
    split_strategy: str,
    split_mode: str,
    split_max_word_count_cjk: int,
    split_max_word_count_english: int,
    custom_prompt: str,
    batch_size: int,
    concurrency: int,
):
    if not subtitle_path:
        raise gr.Error("请先上传字幕文件（SRT/VTT）。")

    save_subtitle_processing_settings(
        processors=subtitle_processors,
        batch_size=int(batch_size),
        concurrency=int(concurrency),
        target_language=target_language,
        translate_reflect=bool(translate_reflect),
        split_mode=split_mode,
        split_max_word_count_cjk=int(split_max_word_count_cjk),
        split_max_word_count_english=int(split_max_word_count_english),
        custom_prompt=custom_prompt,
    )

    processor_order = ["optimize", "translate", "split"]
    selected = [str(x).strip() for x in (subtitle_processors or []) if str(x).strip()]
    selected = [p for p in processor_order if p in set(selected)]
    if not selected:
        raise gr.Error("请至少选择一个处理类型。")

    provider = (subtitle_provider or "").strip() or "openai"
    if provider != "openai":
        raise gr.Error(f"暂不支持该字幕处理提供商：{provider!r}")

    api_key = (subtitle_openai_api_key or "").strip()
    if not api_key:
        raise gr.Error("请先在「字幕处理」中填写 API Key。")

    base_url = (subtitle_openai_base_url or "").strip() or None
    llm_model = (subtitle_llm_model or "").strip() or DEFAULT_SUBTITLE_LLM_MODEL
    try:
        llm_temperature = float(subtitle_llm_temperature)
    except Exception:
        llm_temperature = DEFAULT_SUBTITLE_LLM_TEMPERATURE
    llm_temperature = max(0.0, min(2.0, llm_temperature))

    common = {"concurrency": int(concurrency)}
    options_by_processor: dict[str, dict] = {}
    if "translate" in selected:
        options_by_processor["translate"] = {
            **common,
            "target_language": (target_language or "").strip() or "zh",
            "reflect": bool(translate_reflect),
            "custom_prompt": (custom_prompt or "").strip(),
            "batch_size": int(batch_size),
        }
    if "optimize" in selected:
        options_by_processor["optimize"] = {
            **common,
            "custom_prompt": (custom_prompt or "").strip(),
            "batch_size": int(batch_size),
        }
    if "split" in selected:
        options_by_processor["split"] = {
            **common,
            "strategy": (split_strategy or "").strip() or "semantic",
            "mode": (split_mode or "").strip() or "inplace_newlines",
            "max_word_count_cjk": int(split_max_word_count_cjk),
            "max_word_count_english": int(split_max_word_count_english),
        }

    save_subtitle_provider_settings(
        provider=provider,
        openai_api_key=api_key,
        openai_base_url=subtitle_openai_base_url,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        split_strategy=split_strategy,
    )

    out_dir = str(Path("outputs") / "processed")
    if len(selected) == 1:
        name = selected[0]
        res = process_subtitle_file(
            subtitle_path,
            processor=name,
            out_dir=out_dir,
            options=options_by_processor.get(name, common),
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            openai_api_key=api_key,
            openai_base_url=base_url,
            chat_json=None,
        )
    else:
        res = process_subtitle_file_multi(
            subtitle_path,
            processors=selected,
            out_dir=out_dir,
            options_by_processor=options_by_processor,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            openai_api_key=api_key,
            openai_base_url=base_url,
            chat_json=None,
        )

    return res.preview_text, res.out_path, res.debug


with gr.Blocks(
    title="Auto-ASR",
) as demo:
    gr.Markdown(
        "\n".join(
            [
                "# Auto-ASR",
                "上传/录制音频 -> ASR -> 导出 SRT / VTT / TXT。",
                "",
                "- 若上游不返回时间戳，可用「VAD 语音段」模式生成更准的字幕轴。",
                "- VAD 语音段模式会增加调用次数（按语音段逐段转写）。",
                "",
                CONFIG_NOTE,
            ]
        )
    )

    with gr.Tabs():
        with gr.Tab("转写", id="tab_transcribe"):
            with gr.Row():
                audio_in = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="音频",
                )

            with gr.Row():
                asr_backend = gr.Dropdown(
                    choices=[
                        ("Qwen3-ASR（本地推理）", "qwen3asr"),
                        ("FunASR（本地推理）", "funasr"),
                        ("OpenAI API（远程）", "openai"),
                    ],
                    value=DEFAULT_ASR_BACKEND,
                    label="ASR 引擎",
                )
                output_format = gr.Dropdown(
                    choices=[
                        ("SRT 字幕", "srt"),
                        ("VTT 字幕", "vtt"),
                        ("纯文本", "txt"),
                    ],
                    value=DEFAULT_OUTPUT_FORMAT,
                    label="输出格式",
                )
                language = gr.Dropdown(
                    choices=[
                        ("自动检测", "auto"),
                        ("中文", "zh"),
                        ("英语", "en"),
                        ("日语", "ja"),
                        ("韩语", "ko"),
                        ("法语", "fr"),
                        ("德语", "de"),
                        ("西语", "es"),
                        ("俄语", "ru"),
                    ],
                    value=DEFAULT_LANGUAGE,
                    label="语言",
                )

            prompt = gr.Textbox(
                label="提示词（可选）",
                placeholder="可填写术语/人名/地名等上下文，提升识别效果。",
            )

            run_btn = gr.Button("开始转写", variant="primary")
            stop_btn = gr.Button("停止转写", variant="stop")

            with gr.Row():
                preview = gr.Textbox(label="字幕预览", lines=12)
            with gr.Row():
                full_text = gr.Textbox(label="完整文本", lines=12)
            with gr.Row():
                out_file = gr.File(label="下载")
                debug = gr.Textbox(label="调试信息", lines=2)

        with gr.Tab("字幕处理", id="tab_subtitle"):
            gr.Markdown(
                "\n".join(
                    [
                        "对已有字幕文件（SRT/VTT）进行 **校正 / 翻译 / 分割**。",
                        "可多选处理类型，按顺序依次执行：**校正 -> 翻译 -> 分割**。",
                    ]
                )
            )

            with gr.Accordion("LLM 提供商（仅字幕处理）", open=True):
                subtitle_provider = gr.Dropdown(
                    choices=[
                        ("OpenAI 兼容", "openai"),
                    ],
                    value=DEFAULT_SUBTITLE_PROVIDER,
                    label="提供商",
                )
                subtitle_openai_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="sk-...",
                    value=DEFAULT_SUBTITLE_OPENAI_API_KEY,
                )
                subtitle_openai_base_url = gr.Textbox(
                    label="Base URL",
                    placeholder="例如：https://api.openai.com/v1",
                    value=DEFAULT_SUBTITLE_OPENAI_BASE_URL,
                )
                subtitle_llm_model = gr.Textbox(
                    label="模型名",
                    value=DEFAULT_SUBTITLE_LLM_MODEL,
                )
                subtitle_llm_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_SUBTITLE_LLM_TEMPERATURE,
                    step=0.05,
                    label="温度（越低越稳定，越高越发散）",
                )
                subtitle_llm_settings_state = gr.State(None)

            subtitle_in = gr.File(
                label="字幕文件（SRT/VTT）",
                file_types=[".srt", ".vtt"],
                type="filepath",
            )
            subtitle_processors = gr.CheckboxGroup(
                choices=[
                    ("字幕校正（LLM）", "optimize"),
                    ("字幕翻译（LLM）", "translate"),
                    ("智能断句（LLM）", "split"),
                ],
                value=DEFAULT_SUBTITLE_PROCESSORS,
                label="处理类型（可多选）",
            )

            with gr.Row():
                target_language = gr.Dropdown(
                    choices=[
                        ("中文", "zh"),
                        ("英语", "en"),
                        ("日语", "ja"),
                        ("韩语", "ko"),
                        ("法语", "fr"),
                        ("德语", "de"),
                        ("西语", "es"),
                        ("俄语", "ru"),
                    ],
                    value=DEFAULT_SUBTITLE_TARGET_LANGUAGE,
                    label="目标语言（仅翻译）",
                )
                translate_reflect = gr.Checkbox(
                    value=DEFAULT_SUBTITLE_TRANSLATE_REFLECT,
                    label="反思翻译（更自然）",
                )
                split_strategy = gr.Dropdown(
                    choices=[
                        ("语义断句（更易读，适合长句/无标点）", "semantic"),
                        ("按句子断句（更保守，尽量按标点）", "sentence"),
                    ],
                    value=DEFAULT_SUBTITLE_SPLIT_STRATEGY,
                    label="断句方式（仅断句）",
                )
                split_mode = gr.Dropdown(
                    choices=[
                        ("只插入换行（不改变时间轴）", "inplace_newlines"),
                        ("拆分为多条字幕（重新分配时间轴）", "split_to_cues"),
                    ],
                    value=DEFAULT_SUBTITLE_SPLIT_MODE,
                    label="输出形式（仅断句）",
                )

            with gr.Row():
                split_max_word_count_cjk = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=DEFAULT_SUBTITLE_SPLIT_MAX_WORD_COUNT_CJK,
                    step=1,
                    label="每段最大字数（CJK，如中/日/韩）",
                )
                split_max_word_count_english = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=DEFAULT_SUBTITLE_SPLIT_MAX_WORD_COUNT_ENGLISH,
                    step=1,
                    label="每段最大单词数（英文等）",
                )

            custom_prompt = gr.Textbox(
                label="自定义提示词（可选）",
                placeholder="可补充术语/风格要求/注意事项等。",
                value=DEFAULT_SUBTITLE_CUSTOM_PROMPT,
            )

            with gr.Row():
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=DEFAULT_SUBTITLE_BATCH_SIZE,
                    step=1,
                    label="Batch Size（翻译/校正）",
                )
                subtitle_concurrency = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=DEFAULT_SUBTITLE_CONCURRENCY,
                    step=1,
                    label="并发数",
                )

            subtitle_run_btn = gr.Button("开始处理字幕", variant="primary")

            with gr.Row():
                subtitle_preview = gr.Textbox(label="预览", lines=12)
            with gr.Row():
                subtitle_out_file = gr.File(label="下载")
                subtitle_debug = gr.Textbox(label="调试信息", lines=2)

        with gr.Tab("引擎配置", id="tab_engine"):
            gr.Markdown(CUDA_NOTE)
            with gr.Row():
                release_cuda_btn = gr.Button("释放显存", variant="secondary")
            release_cuda_status = gr.Markdown()

            with gr.Accordion("OpenAI 配置", open=True):
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    value=DEFAULT_OPENAI_API_KEY,
                )
                openai_base_url = gr.Textbox(
                    label="Base URL",
                    placeholder="例如：https://api.openai.com/v1",
                    value=DEFAULT_OPENAI_BASE_URL,
                )
                model = gr.Textbox(
                    label="模型名",
                    value=DEFAULT_MODEL,
                )

            with gr.Accordion("Qwen3-ASR 本地推理", open=False):
                gr.Markdown("首次使用需安装：`uv sync --extra transformers`")
                gr.Markdown(f"模型下载目录（项目内）：`{get_models_dir()}`")
                gr.Markdown(
                    "\n".join(
                        [
                            "模型地址（HuggingFace）：",
                            f"- {QWEN3_ASR_HF_URL}",
                            f"- {QWEN3_FORCED_ALIGNER_HF_URL}",
                        ]
                    )
                )
                qwen3_model = gr.Dropdown(
                    choices=[
                        ("Qwen3-ASR-1.7B（Qwen/Qwen3-ASR-1.7B）", "Qwen/Qwen3-ASR-1.7B"),
                    ],
                    value=DEFAULT_QWEN3_MODEL,
                    label="ASR 模型（HuggingFace RepoID）",
                    allow_custom_value=True,
                )
                qwen3_use_forced_aligner = gr.Checkbox(
                    value=DEFAULT_QWEN3_USE_FORCED_ALIGNER,
                    label="输出字幕轴（使用 Forced Aligner）",
                    info="开启：输出 SRT/VTT 时用 Forced Aligner 生成更细的时间轴；关闭：不加载/不调用 Forced Aligner，仅按切分段落生成粗略时间轴。",
                )
                qwen3_forced_aligner = gr.Dropdown(
                    choices=[
                        (
                            "ForcedAligner-0.6B（Qwen/Qwen3-ForcedAligner-0.6B）",
                            "Qwen/Qwen3-ForcedAligner-0.6B",
                        ),
                    ],
                    value=DEFAULT_QWEN3_FORCED_ALIGNER,
                    label="Forced Aligner（用于字幕轴，可选）",
                    allow_custom_value=True,
                )
                with gr.Row():
                    download_qwen3_btn = gr.Button("下载模型", variant="secondary")
                    load_qwen3_btn = gr.Button("加载模型", variant="primary")
                download_qwen3_status = gr.Markdown()
                load_qwen3_status = gr.Markdown()
                qwen3_device = gr.Dropdown(
                    choices=[
                        ("自动", "auto"),
                        ("CPU", "cpu"),
                        ("CUDA:0", "cuda:0"),
                    ],
                    value=DEFAULT_QWEN3_DEVICE,
                    label="设备",
                )
                qwen3_max_inference_batch_size = gr.Slider(
                    minimum=1,
                    maximum=64,
                    value=DEFAULT_QWEN3_MAX_INFERENCE_BATCH_SIZE,
                    step=1,
                    label="推理 Batch Size（并行；越大越快但更吃显存）",
                )

            with gr.Accordion("FunASR 本地推理", open=False):
                gr.Markdown("首次使用需安装：`uv sync --extra funasr`")
                gr.Markdown(f"模型下载目录（项目内）：`{get_models_dir()}`")
                funasr_model = gr.Dropdown(
                    choices=[
                        ("SenseVoiceSmall（iic/SenseVoiceSmall）", "iic/SenseVoiceSmall"),
                        (
                            "Fun-ASR-Nano-2512（FunAudioLLM/Fun-ASR-Nano-2512）",
                            "FunAudioLLM/Fun-ASR-Nano-2512",
                        ),
                    ],
                    value=DEFAULT_FUNASR_MODEL,
                    label="模型（HuggingFace RepoID）",
                    allow_custom_value=True,
                )
                with gr.Row():
                    download_model_btn = gr.Button("下载模型", variant="secondary")
                    load_model_btn = gr.Button("加载模型", variant="primary")
                download_model_status = gr.Markdown()
                load_model_status = gr.Markdown()
                funasr_device = gr.Dropdown(
                    choices=[
                        ("自动", "auto"),
                        ("CPU", "cpu"),
                        ("CUDA:0", "cuda:0"),
                    ],
                    value=DEFAULT_FUNASR_DEVICE,
                    label="设备",
                )
                funasr_language = gr.Dropdown(
                    choices=[
                        ("自动", "auto"),
                        ("中文", "zh"),
                        ("粤语", "yue"),
                        ("英语", "en"),
                        ("日语", "ja"),
                        ("韩语", "ko"),
                        ("不说话/无语音", "nospeech"),
                    ],
                    value=DEFAULT_FUNASR_LANGUAGE,
                    label="语言（FunASR）",
                )
                funasr_use_itn = gr.Checkbox(
                    value=DEFAULT_FUNASR_USE_ITN,
                    label="启用 ITN（数字/符号归一化）",
                )
                with gr.Row():
                    funasr_enable_vad = gr.Checkbox(
                        value=DEFAULT_FUNASR_ENABLE_VAD,
                        label="启用内置 VAD",
                    )
                    funasr_enable_punc = gr.Checkbox(
                        value=DEFAULT_FUNASR_ENABLE_PUNC,
                        label="启用标点恢复（推荐）",
                    )

        with gr.Tab("切分与字幕轴", id="tab_vad"):
            with gr.Accordion("长音频切分", open=True):
                enable_vad = gr.Checkbox(
                    value=DEFAULT_ENABLE_VAD,
                    label="启用 VAD（用于长音频切分 / 语音段模式）",
                )
                vad_segment_threshold_s = gr.Slider(
                    minimum=30,
                    maximum=240,
                    value=DEFAULT_VAD_SEGMENT_THRESHOLD_S,
                    step=10,
                    label="目标分段时长（秒）",
                )
                vad_max_segment_threshold_s = gr.Slider(
                    minimum=60,
                    maximum=360,
                    value=DEFAULT_VAD_MAX_SEGMENT_THRESHOLD_S,
                    step=10,
                    label="最大分段时长（秒）",
                )

            with gr.Accordion("VAD 灵敏度（漏字可调低阈值/缩短时长）", open=False):
                vad_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=DEFAULT_VAD_THRESHOLD,
                    step=0.05,
                    label="VAD 阈值（越低越敏感，越不容易漏语气词）",
                )
                vad_min_speech_duration_ms = gr.Slider(
                    minimum=50,
                    maximum=2000,
                    value=DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
                    step=50,
                    label="最小语音时长（ms）",
                )
                vad_min_silence_duration_ms = gr.Slider(
                    minimum=50,
                    maximum=2000,
                    value=DEFAULT_VAD_MIN_SILENCE_DURATION_MS,
                    step=50,
                    label="最小静音时长（ms）",
                )
                vad_speech_pad_ms = gr.Slider(
                    minimum=0,
                    maximum=2000,
                    value=DEFAULT_VAD_SPEECH_PAD_MS,
                    step=50,
                    label="语音段边缘填充（ms，避免切掉开头/结尾字）",
                )

            with gr.Accordion("字幕轴（模型不支持字幕轴时）", open=True):
                timeline_strategy = gr.Dropdown(
                    choices=[
                        ("按 VAD 语音段（更准，调用更多）", "vad_speech"),
                        ("按分段整段（省调用，可能粗）", "chunk"),
                    ],
                    value=DEFAULT_TIMELINE_STRATEGY,
                    label="时间轴策略",
                )
                vad_speech_max_utterance_s = gr.Slider(
                    minimum=5,
                    maximum=60,
                    value=DEFAULT_VAD_SPEECH_MAX_UTTERANCE_S,
                    step=1,
                    label="语音段最大时长（秒）",
                )
                vad_speech_merge_gap_ms = gr.Slider(
                    minimum=0,
                    maximum=2000,
                    value=DEFAULT_VAD_SPEECH_MERGE_GAP_MS,
                    step=50,
                    label="合并相邻语音段的静音阈值（毫秒）",
                )

        with gr.Tab("性能", id="tab_perf"):
            with gr.Accordion("上传限制", open=True):
                upload_audio_format = gr.Dropdown(
                    choices=[
                        ("WAV 无压缩", "wav"),
                        ("MP3 压缩", "mp3"),
                    ],
                    value=DEFAULT_UPLOAD_AUDIO_FORMAT,
                    label="上传音频格式",
                )

            with gr.Accordion("性能", open=True):
                api_concurrency = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=DEFAULT_API_CONCURRENCY,
                    step=1,
                    label="并发请求数",
                )

    download_model_btn.click(
        fn=download_funasr_model_ui,
        inputs=[funasr_model, funasr_enable_vad, funasr_enable_punc],
        outputs=[download_model_status],
    )

    load_model_btn.click(
        fn=load_funasr_model_ui,
        inputs=[funasr_model, funasr_device, funasr_enable_vad, funasr_enable_punc],
        outputs=[load_model_status],
    )

    download_qwen3_btn.click(
        fn=download_qwen3_models_ui,
        inputs=[qwen3_model, qwen3_forced_aligner, qwen3_use_forced_aligner],
        outputs=[download_qwen3_status],
    )

    load_qwen3_btn.click(
        fn=load_qwen3_model_ui,
        inputs=[
            qwen3_model,
            qwen3_forced_aligner,
            qwen3_device,
            qwen3_max_inference_batch_size,
            qwen3_use_forced_aligner,
        ],
        outputs=[load_qwen3_status],
    )
    release_cuda_btn.click(
        fn=release_cuda_ui, inputs=[], outputs=[release_cuda_status], queue=False
    )

    subtitle_provider.change(
        fn=_save_subtitle_provider_settings_ui,
        inputs=[
            subtitle_provider,
            subtitle_openai_api_key,
            subtitle_openai_base_url,
            subtitle_llm_model,
            subtitle_llm_temperature,
            split_strategy,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    subtitle_openai_api_key.change(
        fn=_save_subtitle_provider_settings_ui,
        inputs=[
            subtitle_provider,
            subtitle_openai_api_key,
            subtitle_openai_base_url,
            subtitle_llm_model,
            subtitle_llm_temperature,
            split_strategy,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    subtitle_openai_base_url.change(
        fn=_save_subtitle_provider_settings_ui,
        inputs=[
            subtitle_provider,
            subtitle_openai_api_key,
            subtitle_openai_base_url,
            subtitle_llm_model,
            subtitle_llm_temperature,
            split_strategy,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    subtitle_llm_model.change(
        fn=_save_subtitle_provider_settings_ui,
        inputs=[
            subtitle_provider,
            subtitle_openai_api_key,
            subtitle_openai_base_url,
            subtitle_llm_model,
            subtitle_llm_temperature,
            split_strategy,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    subtitle_llm_temperature.change(
        fn=_save_subtitle_provider_settings_ui,
        inputs=[
            subtitle_provider,
            subtitle_openai_api_key,
            subtitle_openai_base_url,
            subtitle_llm_model,
            subtitle_llm_temperature,
            split_strategy,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    split_strategy.change(
        fn=_save_subtitle_provider_settings_ui,
        inputs=[
            subtitle_provider,
            subtitle_openai_api_key,
            subtitle_openai_base_url,
            subtitle_llm_model,
            subtitle_llm_temperature,
            split_strategy,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )

    subtitle_processors.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    batch_size.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    subtitle_concurrency.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    target_language.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    translate_reflect.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    split_mode.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    split_max_word_count_cjk.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    split_max_word_count_english.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )
    custom_prompt.change(
        fn=_save_subtitle_processing_settings_ui,
        inputs=[
            subtitle_processors,
            batch_size,
            subtitle_concurrency,
            target_language,
            translate_reflect,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
        ],
        outputs=[subtitle_llm_settings_state],
        queue=False,
    )

    subtitle_run_btn.click(
        fn=run_subtitle_processing,
        inputs=[
            subtitle_in,
            subtitle_processors,
            subtitle_provider,
            subtitle_openai_api_key,
            subtitle_openai_base_url,
            subtitle_llm_model,
            subtitle_llm_temperature,
            target_language,
            translate_reflect,
            split_strategy,
            split_mode,
            split_max_word_count_cjk,
            split_max_word_count_english,
            custom_prompt,
            batch_size,
            subtitle_concurrency,
        ],
        outputs=[subtitle_preview, subtitle_out_file, subtitle_debug],
        concurrency_limit=1,
    )

    run_event = run_btn.click(
        fn=run_asr,
        inputs=[
            audio_in,
            asr_backend,
            openai_api_key,
            openai_base_url,
            model,
            funasr_model,
            funasr_device,
            funasr_language,
            funasr_use_itn,
            funasr_enable_vad,
            funasr_enable_punc,
            output_format,
            language,
            prompt,
            enable_vad,
            vad_segment_threshold_s,
            vad_max_segment_threshold_s,
            vad_threshold,
            vad_min_speech_duration_ms,
            vad_min_silence_duration_ms,
            vad_speech_pad_ms,
            timeline_strategy,
            vad_speech_max_utterance_s,
            vad_speech_merge_gap_ms,
            upload_audio_format,
            api_concurrency,
            qwen3_model,
            qwen3_forced_aligner,
            qwen3_device,
            qwen3_max_inference_batch_size,
            qwen3_use_forced_aligner,
        ],
        outputs=[preview, full_text, out_file, debug],
        concurrency_limit=1,
    )

    stop_btn.click(
        fn=stop_transcribe,
        inputs=[],
        outputs=[debug],
        cancels=[run_event],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch(theme=THEME)
