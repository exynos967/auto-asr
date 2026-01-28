# auto-asr (Gradio + OpenAI / FunASR)

Upload/record audio in a Gradio UI, transcribe via OpenAI API or local FunASR, and download subtitles as `srt` / `vtt` / `txt`.

Long-audio splitting logic is adapted from Qwen3-ASR-Toolkit (MIT License).
See `THIRD_PARTY_NOTICES.md`.

## Setup

```bash
cd /root/workdir/auto-asr
uv sync
```

启动后在页面的「OpenAI 配置」里填写 `OpenAI API Key`（以及可选的 `Base URL`）。
配置会在每次点击「开始转写」时自动保存到项目根目录的 `.auto_asr_config.json`（明文保存 key，已加入 `.gitignore`）。如需重置配置，删除该文件即可。

如需使用「FunASR（本地推理）」：

```bash
uv sync --extra funasr
```

Windows + Python 3.12 可能会遇到 `llvmlite/numba` 依赖不兼容导致安装失败（例如报错 *Cannot install on Python version 3.12*）。
建议改用 Python 3.11（或 3.10）创建环境后再安装：

```bash
# 安装/使用指定 Python 版本（示例：3.11）
uv python install 3.11
uv sync --extra funasr -p 3.11
```

如需使用 NVIDIA 显卡（CUDA）加速本地推理，可额外安装 CUDA 版 PyTorch（示例：CUDA 12.1；按你的驱动/CUDA 版本选择对应的 cuXXX）：

```bash
# 示例：安装 CUDA 12.1 版 torch（含 torchaudio）
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchaudio
```

## Run

```bash
uv run python app.py
# or
# uv run --script app.py
```

Then open the printed local URL.

## Notes

- `imageio-ffmpeg` is used to provide an `ffmpeg` binary (no system `ffmpeg` required).
- `silero_vad` is installed by default (it may pull in PyTorch/ONNXRuntime and is large; first install can be slow).
- Optional: install FunASR runtime deps for local inference: `uv sync --extra funasr` (also large).

- Output files are written to `./outputs/`.
