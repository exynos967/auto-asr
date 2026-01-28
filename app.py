from __future__ import annotations

import logging
from threading import Event, Lock

import gradio as gr

from auto_asr.config import get_config_path, load_config, save_config
from auto_asr.funasr_asr import download_funasr_model, preload_funasr_model
from auto_asr.pipeline import transcribe_to_subtitles

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
DEFAULT_ASR_BACKEND = _str(_SAVED_CONFIG.get("asr_backend", "openai")).strip() or "openai"
if DEFAULT_ASR_BACKEND not in {"openai", "funasr"}:
    DEFAULT_ASR_BACKEND = "openai"

DEFAULT_FUNASR_MODEL = _str(_SAVED_CONFIG.get("funasr_model", "iic/SenseVoiceSmall")).strip()
DEFAULT_FUNASR_DEVICE = _str(_SAVED_CONFIG.get("funasr_device", "auto")).strip() or "auto"
if DEFAULT_FUNASR_DEVICE not in {"auto", "cpu", "cuda:0"}:
    DEFAULT_FUNASR_DEVICE = "auto"
DEFAULT_FUNASR_LANGUAGE = _str(_SAVED_CONFIG.get("funasr_language", "auto")).strip() or "auto"
DEFAULT_FUNASR_USE_ITN = bool(_SAVED_CONFIG.get("funasr_use_itn", True))
DEFAULT_FUNASR_ENABLE_VAD = bool(_SAVED_CONFIG.get("funasr_enable_vad", True))
DEFAULT_FUNASR_ENABLE_PUNC = bool(_SAVED_CONFIG.get("funasr_enable_punc", True))
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
    DEFAULT_VAD_THRESHOLD = float(_SAVED_CONFIG.get("vad_threshold", 0.5))
except Exception:
    DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_THRESHOLD = max(0.1, min(0.9, DEFAULT_VAD_THRESHOLD))

DEFAULT_VAD_MIN_SPEECH_DURATION_MS = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_min_speech_duration_ms"), 200), 50, 2000
)
DEFAULT_VAD_MIN_SILENCE_DURATION_MS = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_min_silence_duration_ms"), 200), 50, 2000
)
DEFAULT_VAD_SPEECH_PAD_MS = _clamp_int(_int(_SAVED_CONFIG.get("vad_speech_pad_ms"), 200), 0, 2000)
DEFAULT_TIMELINE_STRATEGY = _str(_SAVED_CONFIG.get("timeline_strategy", "vad_speech")).strip()
if DEFAULT_TIMELINE_STRATEGY not in {"chunk", "vad_speech"}:
    DEFAULT_TIMELINE_STRATEGY = "vad_speech"

DEFAULT_UPLOAD_AUDIO_FORMAT = _str(_SAVED_CONFIG.get("upload_audio_format", "wav")).strip()
if DEFAULT_UPLOAD_AUDIO_FORMAT not in {"wav", "mp3"}:
    DEFAULT_UPLOAD_AUDIO_FORMAT = "wav"

UPLOAD_MP3_BITRATE_KBPS = 192

DEFAULT_VAD_SPEECH_MAX_UTTERANCE_S = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_speech_max_utterance_s"), 20), 5, 60
)
DEFAULT_VAD_SPEECH_MERGE_GAP_MS = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_speech_merge_gap_ms"), 300), 0, 2000
)
DEFAULT_API_CONCURRENCY = _clamp_int(_int(_SAVED_CONFIG.get("api_concurrency"), 4), 1, 16)
CONFIG_NOTE = f"配置文件：`{_CONFIG_PATH}`（自动保存，明文保存 key，删除该文件即可重置）"


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
THEME = gr.themes.Soft()

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


def load_funasr_model_ui(
    funasr_model: str,
    funasr_device: str,
    funasr_enable_vad: bool,
    funasr_enable_punc: bool,
) -> str:
    resolved_device = _resolve_funasr_device_ui(funasr_device)
    preload_funasr_model(
        model=(funasr_model or "").strip(),
        device=resolved_device,
        enable_vad=bool(funasr_enable_vad),
        enable_punc=bool(funasr_enable_punc),
    )
    return f"已加载 FunASR 模型：{(funasr_model or '').strip()}（device={resolved_device}）"


def download_funasr_model_ui(
    funasr_model: str,
    funasr_enable_vad: bool,
    funasr_enable_punc: bool,
) -> str:
    download_funasr_model(
        model=(funasr_model or "").strip(),
        enable_vad=bool(funasr_enable_vad),
        enable_punc=bool(funasr_enable_punc),
    )
    return "下载/初始化完成（已写入缓存目录）。若下次仍需下载，通常是网络/缓存目录权限问题。"


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

    path = save_config(config)
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


with gr.Blocks(
    title="Auto-ASR",
) as demo:
    gr.Markdown(
        "\n".join(
            [
                "# Auto-ASR",
                "上传/录制音频 -> ASR -> 导出 SRT / VTT / TXT。",
                "",
                "- 若上游不返回 segments（无时间戳），可用「VAD 语音段」模式生成更准的字幕轴。",
                "- VAD 语音段模式会增加调用次数（按语音段逐段转写）。",
                "- 为了加速，语音段模式默认使用 WAV(PCM16) 上传；分段整段模式可选 MP3 压缩。",
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
                        ("OpenAI API（远程）", "openai"),
                        ("FunASR（本地推理）", "funasr"),
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
                preview = gr.Textbox(label="字幕预览（前约 5000 字符）", lines=12)
            with gr.Row():
                full_text = gr.Textbox(label="完整文本", lines=12)
            with gr.Row():
                out_file = gr.File(label="下载")
                debug = gr.Textbox(label="调试信息", lines=2)

        with gr.Tab("引擎配置", id="tab_engine"):
            gr.Markdown(CUDA_NOTE)
            with gr.Group(visible=DEFAULT_ASR_BACKEND == "openai") as openai_group:
                gr.Markdown("## OpenAI 配置")
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key（会明文保存在本机配置文件里）",
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
                    label="模型（默认 whisper-1）",
                    value=DEFAULT_MODEL,
                )

            with gr.Group(visible=DEFAULT_ASR_BACKEND == "funasr") as funasr_group:
                gr.Markdown("## FunASR 本地推理")
                gr.Markdown("首次使用需安装：`uv sync --extra funasr`")
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
                        label="启用内置 VAD（推荐）",
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

            with gr.Accordion("字幕轴（上游无 segments 时）", open=True):
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

        with gr.Tab("性能与上传", id="tab_perf"):
            with gr.Accordion("上传优化（避免上游文件过大）", open=True):
                upload_audio_format = gr.Dropdown(
                    choices=[
                        ("WAV 无压缩（可能触发上游大小限制）", "wav"),
                        ("MP3 压缩（固定 192kbps）", "mp3"),
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
                    label="并发请求数（仅 VAD 语音段模式生效）",
                )

    def _toggle_asr_backend(backend: str):
        backend = (backend or "").strip()
        return gr.update(visible=backend == "openai"), gr.update(visible=backend == "funasr")

    asr_backend.change(
        fn=_toggle_asr_backend,
        inputs=[asr_backend],
        outputs=[openai_group, funasr_group],
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
