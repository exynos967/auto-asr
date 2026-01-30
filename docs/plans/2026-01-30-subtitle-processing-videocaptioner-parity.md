# Subtitle Processing (LLM) VideoCaptioner Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 Auto-ASR 的「字幕校正(LLM) / 字幕翻译(LLM) / 智能断句(LLM)」与 `VideoCaptioner/` 的行为与选项尽量一致（提示词、agent loop 校验/反馈、降级策略、时间轴重分配）。

**Architecture:** 保持 Auto-ASR 现有 `subtitle_processing` 的 pipeline/processor 架构不变；引入 VideoCaptioner 的 prompt 文件与关键算法（agent loop 校验、SubtitleAligner、split_by_llm），通过新增 `chat_text` 接口解耦 LLM 提供商与处理器逻辑。

**Tech Stack:** Python 3.10+, OpenAI compatible API (openai sdk), json-repair, pytest.

---

### Task 1: 引入 Prompt 文件与加载器

**Files:**
- Create: `auto_asr/subtitle_processing/prompts.py`
- Create: `auto_asr/subtitle_processing/prompts/translate/standard.md`
- Create: `auto_asr/subtitle_processing/prompts/translate/reflect.md`
- Create: `auto_asr/subtitle_processing/prompts/translate/single.md`
- Create: `auto_asr/subtitle_processing/prompts/optimize/subtitle.md`
- Create: `auto_asr/subtitle_processing/prompts/split/sentence.md`
- Create: `auto_asr/subtitle_processing/prompts/split/semantic.md`

**Step 1: Write failing test**
- Test: `tests/test_subtitle_prompts.py`
  - `get_prompt("split/sentence", max_word_count_cjk=18, ...)` 会替换模板变量并返回字符串
  - `get_prompt("not-exists")` 抛 `FileNotFoundError`

**Step 2: Run test (RED)**
- Run: `uv run pytest -q tests/test_subtitle_prompts.py`
- Expected: FAIL (模块不存在/函数不存在)

**Step 3: Minimal implementation**
- 实现 `get_prompt()`：读取 md，`Template.safe_substitute()`，LRU cache
- 拷贝 VideoCaptioner 对应 prompt 文本（必要时调整占位符名保持一致）

**Step 4: Run test (GREEN)**
- Run: `uv run pytest -q tests/test_subtitle_prompts.py`
- Expected: PASS

---

### Task 2: Subtitle LLM Context 改为 `chat_text`（保留现有 backoff/auth 行为）

**Files:**
- Modify: `auto_asr/subtitle_processing/base.py`
- Modify: `auto_asr/subtitle_processing/pipeline.py`
- Test: `tests/test_subtitle_processing_pipeline.py`

**Step 1: Write failing tests**
- 调整 pipeline 测试：注入 `chat_text`（而不是 `chat_json`）
- 验证：429 退避、401/403 直报错逻辑依旧

**Step 2: Run test (RED)**
- Run: `uv run pytest -q tests/test_subtitle_processing_pipeline.py`
- Expected: FAIL（签名/上下文不匹配）

**Step 3: Minimal implementation**
- `ProcessorContext` 增加 `chat_text`
- pipeline 内部提供 `_make_openai_chat_text()`，并继续复用现有 backoff/auth 行为
- `process_subtitle_file(_multi)` 构造 `ProcessorContext(chat_text=...)`

**Step 4: Run test (GREEN)**
- Run: `uv run pytest -q tests/test_subtitle_processing_pipeline.py`
- Expected: PASS

---

### Task 3: 智能断句（LLM）对齐 VideoCaptioner split_by_llm

**Files:**
- Modify: `auto_asr/subtitle_processing/processors/split.py`
- Modify: `app.py`
- Modify: `auto_asr/subtitle_processing/settings.py`
- Test: `tests/test_subtitle_processors_split.py`

**Step 1: Write failing tests**
- split 使用 prompt 文件（sentence/semantic）并通过 `chat_text` 返回 `<br>` 文本
- agent loop：内容改动过大时回退原文；超出字数限制时给出 feedback 并重试
- mode=`split_to_cues`：仍按比例重分配时间轴（保留）

**Step 2: Run test (RED)**
- Run: `uv run pytest -q tests/test_subtitle_processors_split.py`
- Expected: FAIL

**Step 3: Minimal implementation**
- 引入 `MAX_STEPS=2` + `_validate_split_result()`（参考 VideoCaptioner）
- 支持 `max_word_count_cjk/max_word_count_english` 两个选项（默认 18/12）并持久化
- UI 增加两个 slider/number 输入（文字更贴近 VideoCaptioner）

**Step 4: Run test (GREEN)**
- Run: `uv run pytest -q tests/test_subtitle_processors_split.py`
- Expected: PASS

---

### Task 4: 字幕翻译（LLM）对齐 VideoCaptioner（standard/reflect + single fallback）

**Files:**
- Modify: `auto_asr/subtitle_processing/processors/translate.py`
- Modify: `app.py`
- Modify: `auto_asr/subtitle_processing/settings.py`
- Test: `tests/test_subtitle_processors_translate_optimize.py`

**Step 1: Write failing tests**
- reflect 模式：LLM 返回嵌套 dict 时提取 `native_translation`
- 翻译 batch 失败时回退 `single`（每条直接输出文本）

**Step 2: Run test (RED)**
- Run: `uv run pytest -q tests/test_subtitle_processors_translate_optimize.py`
- Expected: FAIL

**Step 3: Minimal implementation**
- 使用 prompt 文件 `translate/standard|reflect|single`
- 复刻 VideoCaptioner 的 agent loop + key 校验逻辑（json_repair）
- 异常时执行单条翻译降级
- UI 增加 “反思翻译（更自然）” 勾选，并持久化

**Step 4: Run test (GREEN)**
- Run: `uv run pytest -q tests/test_subtitle_processors_translate_optimize.py`
- Expected: PASS

---

### Task 5: 字幕校正（LLM）对齐 VideoCaptioner（agent loop + SubtitleAligner repair）

**Files:**
- Create: `auto_asr/subtitle_processing/alignment.py`
- Modify: `auto_asr/subtitle_processing/processors/optimize.py`
- Test: `tests/test_subtitle_processors_translate_optimize.py`

**Step 1: Write failing tests**
- optimize 走 agent loop：key 不匹配时会反馈并重试
- 输出相似度过低会反馈并重试；最终仍失败则回退原文
- 对齐修复：当 LLM 少返回一条时，repair 后输出仍保持与输入同长度

**Step 2: Run test (RED)**
- Run: `uv run pytest -q tests/test_subtitle_processors_translate_optimize.py`
- Expected: FAIL

**Step 3: Minimal implementation**
- 拷贝/改写 `SubtitleAligner`（来自 VideoCaptioner）
- optimize 复刻 VideoCaptioner agent loop + 校验 + repair 逻辑

**Step 4: Run test (GREEN)**
- Run: `uv run pytest -q tests/test_subtitle_processors_translate_optimize.py`
- Expected: PASS

---

### Task 6: 合规与回归验证

**Files:**
- Modify: `THIRD_PARTY_NOTICES.md`（补充来自 VideoCaptioner 的 prompts/算法复用说明）

**Steps:**
- Run: `uv run pytest -q` 期望全绿
- （可选）Run: `uv run ruff check .`（如果项目已有 ruff）

