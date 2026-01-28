from __future__ import annotations

import logging

import gradio as gr

from auto_asr.config import delete_config, get_config_path, load_config, save_config
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

DEFAULT_UPLOAD_AUDIO_FORMAT = _str(_SAVED_CONFIG.get("upload_audio_format", "mp3")).strip()
if DEFAULT_UPLOAD_AUDIO_FORMAT not in {"wav", "mp3"}:
    DEFAULT_UPLOAD_AUDIO_FORMAT = "mp3"

DEFAULT_UPLOAD_MP3_BITRATE_KBPS = _clamp_int(
    _int(_SAVED_CONFIG.get("upload_mp3_bitrate_kbps"), 64), 16, 192
)

DEFAULT_VAD_SPEECH_MAX_UTTERANCE_S = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_speech_max_utterance_s"), 20), 5, 60
)
DEFAULT_VAD_SPEECH_MERGE_GAP_MS = _clamp_int(
    _int(_SAVED_CONFIG.get("vad_speech_merge_gap_ms"), 300), 0, 2000
)
DEFAULT_REMEMBER_API_KEY = bool(DEFAULT_OPENAI_API_KEY)
INITIAL_CONFIG_STATUS = (
    f"配置文件：`{_CONFIG_PATH}`"
    + ("（已加载）" if _CONFIG_PATH.exists() else "（尚未保存）")
)

logger.info(
    "auto-asr 启动: config=%s, exists=%s, api_key_saved=%s",
    _CONFIG_PATH,
    _CONFIG_PATH.exists(),
    bool(DEFAULT_OPENAI_API_KEY),
)

def run_asr(
    audio_path: str | None,
    openai_api_key: str,
    openai_base_url: str,
    model: str,
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
    upload_mp3_bitrate_kbps: int,
):
    if not audio_path:
        raise gr.Error("请先上传或录制一段音频。")
    if not (openai_api_key or "").strip():
        raise gr.Error("请先填写 OpenAI API Key。")

    lang = None if language == "auto" else language
    prompt = (prompt or "").strip() or None
    model = (model or "").strip() or "whisper-1"
    base_url = (openai_base_url or "").strip() or None

    logger.info(
        "收到转写请求: file=%s, format=%s, language=%s, model=%s, base_url=%s, "
        "enable_vad=%s, target=%ss, max=%ss, timeline_strategy=%s, upload=%s, "
        "vad_threshold=%.2f, vad_min_speech_ms=%d, vad_min_silence_ms=%d, vad_pad_ms=%d",
        audio_path,
        output_format,
        lang or "auto",
        model,
        base_url or "(default)",
        enable_vad,
        vad_segment_threshold_s,
        vad_max_segment_threshold_s,
        timeline_strategy,
        f"{upload_audio_format}/{int(upload_mp3_bitrate_kbps)}k",
        float(vad_threshold),
        int(vad_min_speech_duration_ms),
        int(vad_min_silence_duration_ms),
        int(vad_speech_pad_ms),
    )

    try:
        result = transcribe_to_subtitles(
            input_audio_path=audio_path,
            openai_api_key=openai_api_key,
            openai_base_url=base_url,
            output_format=output_format,
            model=model,
            language=lang,
            prompt=prompt,
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
            upload_audio_format=(upload_audio_format or "").strip() or "mp3",
            upload_mp3_bitrate_kbps=int(upload_mp3_bitrate_kbps),
        )
    except Exception as e:
        raise gr.Error(f"转写失败：{e}") from e

    return result.preview_text, result.full_text, result.subtitle_file_path, result.debug


def save_settings(
    openai_api_key: str,
    openai_base_url: str,
    model: str,
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
    upload_mp3_bitrate_kbps: int,
    remember_api_key: bool,
) -> str:
    config = {
        "enable_vad": bool(enable_vad),
        "language": (language or "").strip() or "auto",
        "model": (model or "").strip() or "whisper-1",
        "openai_base_url": (openai_base_url or "").strip(),
        "output_format": (output_format or "").strip() or "srt",
        "timeline_strategy": (timeline_strategy or "").strip() or "vad_speech",
        "upload_audio_format": (upload_audio_format or "").strip() or "mp3",
        "upload_mp3_bitrate_kbps": int(upload_mp3_bitrate_kbps),
        "vad_threshold": float(vad_threshold),
        "vad_min_speech_duration_ms": int(vad_min_speech_duration_ms),
        "vad_min_silence_duration_ms": int(vad_min_silence_duration_ms),
        "vad_speech_pad_ms": int(vad_speech_pad_ms),
        "vad_speech_max_utterance_s": int(vad_speech_max_utterance_s),
        "vad_speech_merge_gap_ms": int(vad_speech_merge_gap_ms),
        "vad_max_segment_threshold_s": int(vad_max_segment_threshold_s),
        "vad_segment_threshold_s": int(vad_segment_threshold_s),
    }
    if remember_api_key:
        config["openai_api_key"] = (openai_api_key or "").strip()
    else:
        config["openai_api_key"] = ""

    path = save_config(config)
    logger.info("配置已保存: path=%s, remember_api_key=%s", path, remember_api_key)
    return f"已保存配置到 `{path}`。下次启动会自动加载。"


def clear_settings():
    deleted = delete_config()
    logger.info("配置已清除: deleted=%s", deleted)
    msg = "已清除已保存配置。" if deleted else "未找到已保存配置。"
    return (
        "",
        "",
        "whisper-1",
        "srt",
        "auto",
        True,
        120,
        180,
        0.5,
        200,
        200,
        200,
        "vad_speech",
        20,
        300,
        "mp3",
        64,
        False,
        msg,
    )


with gr.Blocks(
    title="Auto-ASR",
    theme=gr.themes.Base(primary_hue=gr.themes.utils.colors.blue),
) as demo:
    gr.Markdown(
        "\n".join(
            [
                "# Auto-ASR",
                "上传/录制音频 -> OpenAI ASR -> 导出 SRT / VTT / TXT。",
                "",
                "- 若上游不返回 segments（无时间戳），可用 VAD 语音段模式生成更准的字幕轴。",
                "- 语音段模式会增加调用次数（按语音段逐段转写）。",
                "- 为了加速，语音段模式默认使用 WAV(PCM16) 上传；分段整段模式可选 MP3 压缩。",
                "- 部分上游对音频文件大小有限制，建议上传格式选 MP3 压缩。",
                "- 本项目默认依赖 Silero VAD（首次安装体积较大，会拉取 PyTorch/ONNXRuntime）。",
            ]
        )
    )

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
            label="模型（默认 whisper-1）",
            value=DEFAULT_MODEL,
        )

    with gr.Accordion("配置保存", open=False):
        remember_api_key = gr.Checkbox(
            value=DEFAULT_REMEMBER_API_KEY,
            label="同时保存 API Key（明文保存在本机）",
        )
        with gr.Row():
            save_btn = gr.Button("保存配置")
            clear_btn = gr.Button("清除已保存配置")
        config_status = gr.Markdown(value=INITIAL_CONFIG_STATUS)

    with gr.Row():
        audio_in = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="音频",
        )

    with gr.Row():
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

    with gr.Accordion("长音频切分", open=True):
        enable_vad = gr.Checkbox(
            value=DEFAULT_ENABLE_VAD, label="启用 VAD（用于长音频切分 / 语音段模式）"
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

    with gr.Accordion("上传优化（避免上游文件过大）", open=False):
        upload_audio_format = gr.Dropdown(
            choices=[
                ("MP3 压缩（推荐）", "mp3"),
                ("WAV 无压缩（可能触发上游大小限制）", "wav"),
            ],
            value=DEFAULT_UPLOAD_AUDIO_FORMAT,
            label="上传音频格式",
        )
        upload_mp3_bitrate_kbps = gr.Slider(
            minimum=16,
            maximum=192,
            value=DEFAULT_UPLOAD_MP3_BITRATE_KBPS,
            step=16,
            label="MP3 码率（kbps）",
        )

    run_btn = gr.Button("开始转写")

    with gr.Row():
        preview = gr.Textbox(label="字幕预览（前约 5000 字符）", lines=12)
    with gr.Row():
        full_text = gr.Textbox(label="完整文本", lines=12)
    with gr.Row():
        out_file = gr.File(label="下载")
        debug = gr.Textbox(label="调试信息", lines=2)

    run_btn.click(
        fn=run_asr,
        inputs=[
            audio_in,
            openai_api_key,
            openai_base_url,
            model,
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
            upload_mp3_bitrate_kbps,
        ],
        outputs=[preview, full_text, out_file, debug],
    )

    save_btn.click(
        fn=save_settings,
        inputs=[
            openai_api_key,
            openai_base_url,
            model,
            output_format,
            language,
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
            upload_mp3_bitrate_kbps,
            remember_api_key,
        ],
        outputs=[config_status],
    )

    clear_btn.click(
        fn=clear_settings,
        inputs=[],
        outputs=[
            openai_api_key,
            openai_base_url,
            model,
            output_format,
            language,
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
            upload_mp3_bitrate_kbps,
            remember_api_key,
            config_status,
        ],
    )


if __name__ == "__main__":
    demo.launch()
