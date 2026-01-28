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

FunASR 本地模型（HuggingFace）：

- SenseVoiceSmall: `https://huggingface.co/iic/SenseVoiceSmall`
- Fun-ASR-Nano-2512: `https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512`

模型存放位置：

- 在 WebUI 的「引擎配置 -> FunASR」点击「下载模型」会将模型下载到缓存目录（由 FunASR/底层下载器决定）。
- 常见缓存位置：
  - HuggingFace: Linux/macOS `~/.cache/huggingface/hub`；Windows `%USERPROFILE%\\.cache\\huggingface\\hub`
  - ModelScope: Linux/macOS `~/.cache/modelscope/hub`；Windows `%USERPROFILE%\\.cache\\modelscope\\hub`

Windows + Python 3.12 可能会遇到 `llvmlite/numba` 依赖不兼容导致安装失败（例如报错 *Cannot install on Python version 3.12*）。
建议改用 Python 3.11（或 3.10）创建环境后再安装（本项目已将 `numpy` 约束到 `numpy<2.2`，以避免 `numba` 选择到不兼容版本）：

```bash
# 安装/使用指定 Python 版本（示例：3.11）
uv python install 3.11
uv sync --extra funasr -p 3.11
```

如需使用 NVIDIA 显卡（CUDA）加速本地推理，可额外安装 CUDA 版 PyTorch（示例：CUDA 12.1；按你的驱动/CUDA 版本选择对应的 cuXXX）：

```bash
# 推荐：用 uv 的 PyTorch 后端选择（避免手动拼 index-url）
uv pip install --upgrade --torch-backend cu121 torch torchaudio
```

如果你使用 `cu130` 这类 index-url 发现 “没有更改包”，通常是因为 PyTorch/uv 当前并没有对应的 cu130 轮子（即使你本机安装了 CUDA 13.0 Toolkit，也不代表有 cu130 的 torch 轮子），或你已安装了同版本的 torch/torchaudio。
可用下面命令检查当前 torch 是否真正在用 CUDA：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Run

```bash
uv run python app.py
# or
# uv run --script app.py
```

Then open the printed local URL.

WebUI 使用提示：

- 选择「FunASR（本地推理）」时，可在「引擎配置 -> FunASR」点击「加载模型」预加载到显存/内存，减少首次转写等待。
- 转写过程中可点击「停止转写」发送停止信号（后台会尽快停止，已发出的请求可能仍需等待返回）。

## Notes

- `imageio-ffmpeg` is used to provide an `ffmpeg` binary (no system `ffmpeg` required).
- `silero_vad` is installed by default (it may pull in PyTorch/ONNXRuntime and is large; first install can be slow).
- Optional: install FunASR runtime deps for local inference: `uv sync --extra funasr` (also large).

- Output files are written to `./outputs/`.
