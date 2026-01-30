# Third Party Notices

本项目包含来自第三方开源项目的代码/提示词（Prompts）复用或改写。请在分发与二次开发时自行确认并遵守对应许可证条款。

## VideoCaptioner

- Project: https://github.com/WEIFENG2333/VideoCaptioner
- License: GNU GPL v3.0（见 `VideoCaptioner/LICENSE`）
- Used/Adapted:
  - `auto_asr/subtitle_processing/prompts/*`（提示词文本来源：`VideoCaptioner/app/core/prompts/*`）
  - `auto_asr/subtitle_processing/alignment.py`（字幕对齐修复逻辑来源：`VideoCaptioner/app/core/split/alignment.py`）

## GPT-SoVITS

- Project: https://github.com/RVC-Boss/GPT-SoVITS
- Used/Adapted:
  - `auto_asr/silence_split.py` 中 RMS 静音切分的 numpy 实现参考了其 `tools/slicer2.py` 的思路与实现细节

