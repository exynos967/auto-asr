from __future__ import annotations

import gradio as gr

from auto_asr.config import delete_config, get_config_path, load_config, save_config
from auto_asr.pipeline import transcribe_to_subtitles

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
DEFAULT_REMEMBER_API_KEY = bool(DEFAULT_OPENAI_API_KEY)
INITIAL_CONFIG_STATUS = (
    f"配置文件：`{_CONFIG_PATH}`"
    + ("（已加载）" if _CONFIG_PATH.exists() else "（尚未保存）")
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
):
    if not audio_path:
        raise gr.Error("请先上传或录制一段音频。")
    if not (openai_api_key or "").strip():
        raise gr.Error("请先填写 OpenAI API Key。")

    lang = None if language == "auto" else language
    prompt = (prompt or "").strip() or None
    model = (model or "").strip() or "whisper-1"
    base_url = (openai_base_url or "").strip() or None

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
    remember_api_key: bool,
) -> str:
    config = {
        "enable_vad": bool(enable_vad),
        "language": (language or "").strip() or "auto",
        "model": (model or "").strip() or "whisper-1",
        "openai_base_url": (openai_base_url or "").strip(),
        "output_format": (output_format or "").strip() or "srt",
        "vad_max_segment_threshold_s": int(vad_max_segment_threshold_s),
        "vad_segment_threshold_s": int(vad_segment_threshold_s),
    }
    if remember_api_key:
        config["openai_api_key"] = (openai_api_key or "").strip()
    else:
        config["openai_api_key"] = ""

    path = save_config(config)
    return f"已保存配置到 `{path}`。下次启动会自动加载。"


def clear_settings():
    deleted = delete_config()
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
        False,
        msg,
    )


with gr.Blocks(title="auto-asr（OpenAI 转字幕）", theme=gr.themes.Ocean()) as demo:
    gr.Markdown(
        "\n".join(
            [
                "# auto-asr 音频转字幕",
                "上传/录制音频 -> OpenAI ASR -> 导出 SRT / VTT / TXT。",
                "",
                "- API 配置在页面中填写，不依赖环境变量。",
                "- 长音频自动切分：内置切分算法（源自 Qwen3-ASR-Toolkit，MIT）。",
                "- 如需真正的 VAD（Silero），安装：`uv sync --extra vad`（体积大，会拉 PyTorch）。",
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
            label="Base URL（可选）",
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
            value=DEFAULT_ENABLE_VAD, label="启用切分（音频 >= 180 秒时生效）"
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
            remember_api_key,
            config_status,
        ],
    )


if __name__ == "__main__":
    demo.launch()
