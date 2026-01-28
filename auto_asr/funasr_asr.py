from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any

from auto_asr.openai_asr import ASRResult, ASRSegment

logger = logging.getLogger(__name__)

_PUNCT_END = set(".!?。！？")


@dataclass(frozen=True)
class FunASRConfig:
    model: str
    device: str
    language: str
    use_itn: bool
    # Whether to enable FunASR built-in VAD/punc if supported by the model.
    enable_vad: bool = True
    enable_punc: bool = True


_MODEL_CACHE: dict[tuple[str, str, bool, bool], Any] = {}


def _needs_trust_remote_code(model: str) -> bool:
    """
    Some FunASR models require `trust_remote_code=True` to load custom code from the model repo.

    We only enable it for known models to reduce noisy warnings and avoid unexpected failures.
    """

    m = (model or "").lower()
    return ("sensevoice" in m) or ("fun-asr-nano" in m) or ("funasrnano" in m)


def _remote_code_candidates(model: str) -> list[str]:
    """
    Candidate `remote_code` values for FunASR AutoModel.

    Some repos (e.g. Fun-ASR-Nano) ship model definitions in `model.py` and are not registered in
    FunASR's built-in registry. In that case AutoModel needs `trust_remote_code=True` and a proper
    `remote_code` hint.
    """

    m = (model or "").lower()
    if ("fun-asr-nano" in m) or ("funasrnano" in m):
        return ["./model.py", "model.py"]
    return []


def _is_not_registered_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "is not registered" in msg or "not registered" in msg


def _raise_if_missing_tokenizers_deps(exc: BaseException) -> None:
    # FunASR 1.3.1 has a known failure mode when `transformers` is missing: it raises
    # UnboundLocalError about `AutoTokenizer` not being associated with a value.
    if "AutoTokenizer" in str(exc):
        raise RuntimeError(
            "FunASR 依赖缺失: 似乎未安装 transformers/sentencepiece. "
            "请执行 `uv sync --extra funasr` 重新安装依赖。"
        ) from exc


def _import_funasr() -> Any:
    try:
        from funasr import AutoModel  # type: ignore

        return AutoModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未安装 FunASR 依赖，无法使用本地推理。请先安装：`uv sync --extra funasr`"
        ) from e


def _maybe_postprocess_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # SenseVoice 系列模型可能会输出 rich transcription 标签, 例如:
    # "< | ja | > < | EMO _ UNKNOWN | > < | S pee ch | > < | withi tn | >"
    # 我们先把这些带空格的标签规整成 "<|...|>", 再尝试使用官方后处理移除它们。
    def _normalize_rich_tags(s: str) -> str:
        def _repl(m: re.Match[str]) -> str:
            inner = re.sub(r"\s+", "", m.group(1))
            return f"<|{inner}|>"

        # Some upstream outputs miss the trailing ">" (e.g. "< | ja |  < | ... |"),
        # so we make it optional here to keep normalization robust.
        return re.sub(r"<\s*\|\s*([^|<>]+?)\s*\|\s*>?", _repl, s)

    text = _normalize_rich_tags(text)
    try:
        from funasr.utils.postprocess_utils import (  # type: ignore
            rich_transcription_postprocess,
        )

        text = rich_transcription_postprocess(text).strip()
    except Exception:
        pass

    # Best-effort: strip remaining tags if upstream postprocess didn't remove them,
    # or when importing the postprocess helper failed.
    # Remove both normalized "<|...|>" and loose "< | ... |" variants.
    text = re.sub(r"<\s*\|\s*[^|<>]+?\s*\|\s*>?", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Best-effort filter kwargs against a callable signature.

    FunASR's `AutoModel` frequently accepts arbitrary `**kwargs` (its signature may only expose a
    VAR_KEYWORD parameter). In that case, filtering by explicit parameter names would incorrectly
    drop required keys like `model`, causing runtime assertions.
    """

    try:
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:  # pragma: no cover
        return kwargs


def _make_model(cfg: FunASRConfig) -> Any:
    """
    Create a FunASR AutoModel with some sensible defaults.

    We keep it best-effort and rely on runtime feature detection, since different models expose
    different knobs.
    """

    key = (cfg.model, cfg.device, bool(cfg.enable_vad), bool(cfg.enable_punc))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    AutoModel = _import_funasr()
    trust_remote_code = _needs_trust_remote_code(cfg.model)
    remote_code_candidates = _remote_code_candidates(cfg.model)

    model_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "device": cfg.device,
        # Some models (e.g. SenseVoiceSmall) require remote code to enable full features.
        "trust_remote_code": trust_remote_code,
        # FunASR 会在初始化时做版本更新检查 (可能较慢), 这里默认禁用。
        "disable_update": True,
    }
    if trust_remote_code and remote_code_candidates:
        model_kwargs["remote_code"] = remote_code_candidates[0]

    # Try to enable built-in VAD / punctuation when requested. These are common in FunASR.
    if cfg.enable_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": 60000}
    if cfg.enable_punc:
        model_kwargs["punc_model"] = "ct-punc"

    try:
        model = AutoModel(**_filter_kwargs(AutoModel, model_kwargs))
    except Exception as e:
        _raise_if_missing_tokenizers_deps(e)
        if _is_not_registered_error(e):
            if not model_kwargs.get("trust_remote_code"):
                logger.warning(
                    "FunASR 模型未注册，尝试启用 trust_remote_code 重试: model=%s", cfg.model
                )
                model_kwargs["trust_remote_code"] = True
                if remote_code_candidates and "remote_code" not in model_kwargs:
                    model_kwargs["remote_code"] = remote_code_candidates[0]
                try:
                    model = AutoModel(**_filter_kwargs(AutoModel, model_kwargs))
                except Exception as e2:
                    _raise_if_missing_tokenizers_deps(e2)
                    raise
            else:
                # trust_remote_code=True but still not registered: try different remote_code values.
                if not remote_code_candidates:
                    raise
                last_exc: Exception = e
                for cand in remote_code_candidates:
                    if model_kwargs.get("remote_code") == cand:
                        continue
                    logger.warning(
                        "FunASR 模型未注册，尝试 remote_code=%s 重试: model=%s", cand, cfg.model
                    )
                    model_kwargs["remote_code"] = cand
                    try:
                        model = AutoModel(**_filter_kwargs(AutoModel, model_kwargs))
                        break
                    except Exception as e2:
                        _raise_if_missing_tokenizers_deps(e2)
                        last_exc = e2
                else:  # no break
                    raise last_exc
        else:
            raise
    _MODEL_CACHE[key] = model
    logger.info("FunASR 模型已加载: model=%s, device=%s, vad=%s, punc=%s", *key)
    return model


def _extract_segments_from_result(res: Any, *, duration_s: float) -> tuple[str, list[ASRSegment]]:
    """
    Parse FunASR `generate()` output into (full_text, segments).

    FunASR output varies across models. We support common patterns:
    - `res` is a list, first element is a dict with `text` and `sentence_info`.
    - `res` is a list of dict segments each containing `text` + start/end fields.
    """
    if not isinstance(res, list) or not res:
        return "", []

    # Case 1: a single dict with aggregated info.
    if len(res) == 1 and isinstance(res[0], dict):
        item = res[0]
        full_text = _maybe_postprocess_text(str(item.get("text", "") or ""))

        def _scale_ts(max_end: float) -> float:
            # Many FunASR models return timestamps in milliseconds.
            return 0.001 if max_end > max(duration_s, 1.0) * 10 else 1.0

        def _merge_caption_units(
            units: list[tuple[float, float, str]],
            *,
            joiner: str = "",
            max_chars: int = 28,
            max_dur_s: float = 6.0,
        ) -> list[ASRSegment]:
            merged: list[ASRSegment] = []
            buf_text: list[str] = []
            buf_start: float | None = None
            buf_end: float | None = None

            def _flush() -> None:
                nonlocal buf_text, buf_start, buf_end
                if not buf_text or buf_start is None or buf_end is None:
                    buf_text = []
                    buf_start = None
                    buf_end = None
                    return
                text = joiner.join(buf_text).strip()
                if text:
                    merged.append(ASRSegment(start_s=buf_start, end_s=buf_end, text=text))
                buf_text = []
                buf_start = None
                buf_end = None

            for start_s, end_s, t in units:
                t = _maybe_postprocess_text(t)
                if not t:
                    continue
                if buf_start is None:
                    buf_start = start_s
                buf_end = end_s
                buf_text.append(t)

                cur_text = joiner.join(buf_text)
                cur_dur = (
                    (buf_end - buf_start)
                    if (buf_end is not None and buf_start is not None)
                    else 0.0
                )
                if (
                    (cur_text and cur_text[-1] in _PUNCT_END)
                    or len(cur_text) >= max_chars
                    or cur_dur >= max_dur_s
                ):
                    _flush()

            _flush()
            return merged

        sent_info = item.get("sentence_info") or item.get("stamp_sents")
        if isinstance(sent_info, list) and sent_info:
            raw: list[tuple[float, float, str]] = []
            for s in sent_info:
                if not isinstance(s, dict):
                    continue
                text = _maybe_postprocess_text(str(s.get("text", "") or ""))
                if not text:
                    continue
                start = s.get("start", s.get("begin_time", s.get("start_time", None)))
                end = s.get("end", s.get("end_time", s.get("finish_time", None)))
                if start is None or end is None:
                    continue
                try:
                    raw.append((float(start), float(end), text))
                except Exception:
                    continue

            if raw:
                max_end = max(e for _s, e, _t in raw)
                scale = _scale_ts(max_end)
                segments = [
                    ASRSegment(start_s=s * scale, end_s=e * scale, text=t) for (s, e, t) in raw
                ]
                return full_text, segments

        # Case 1b: `timestamp` (word/token level timestamps). Seen in some FunASR models.
        timestamp = item.get("timestamp")
        if isinstance(timestamp, list) and timestamp:
            token_entries: list[tuple[float, float, str]] = []
            for it in timestamp:
                # Common formats:
                # - [start_ms, end_ms]
                # - [token, start_s, end_s]
                # - [token, start_ms, end_ms]
                if isinstance(it, (list, tuple)):
                    if len(it) >= 3 and isinstance(it[0], str):
                        token = it[0]
                        try:
                            start = float(it[1])
                            end = float(it[2])
                        except Exception:
                            continue
                        token_entries.append((start, end, token))
                    elif len(it) >= 2:
                        try:
                            start = float(it[0])
                            end = float(it[1])
                        except Exception:
                            continue
                        # If no token is provided, we fall back to splitting by whitespace later.
                        token_entries.append((start, end, ""))
                elif isinstance(it, dict):
                    start = it.get("start", it.get("begin_time", it.get("start_time", None)))
                    end = it.get("end", it.get("end_time", it.get("finish_time", None)))
                    token = it.get("text", it.get("token", ""))
                    if start is None or end is None:
                        continue
                    try:
                        token_entries.append((float(start), float(end), str(token or "")))
                    except Exception:
                        continue

            if token_entries:
                max_end = max(e for _s, e, _t in token_entries)
                scale = _scale_ts(max_end)
                token_entries = [(s * scale, e * scale, t) for (s, e, t) in token_entries]

                # If timestamp entries include tokens, do a simple SentencePiece-style merge.
                if any(t for _s, _e, t in token_entries):
                    words: list[tuple[float, float, str]] = []
                    cur_word = ""
                    cur_start: float | None = None
                    cur_end: float | None = None

                    def _flush_word() -> None:
                        nonlocal cur_word, cur_start, cur_end
                        if cur_word and cur_start is not None and cur_end is not None:
                            words.append((cur_start, cur_end, cur_word))
                        cur_word = ""
                        cur_start = None
                        cur_end = None

                    for s, e, tok in token_entries:
                        tok = str(tok or "")
                        if tok == "▁":
                            continue
                        is_new = tok.startswith("▁")
                        piece = tok[1:] if is_new else tok
                        if is_new and cur_word:
                            _flush_word()
                        if cur_start is None:
                            cur_start = s
                        cur_end = e
                        cur_word += piece

                    _flush_word()
                    # SentencePiece-style markers (e.g. "▁") indicate word boundaries.
                    # Use a space joiner for latin words, otherwise keep it compact for CJK.
                    joiner = " " if any(re.search(r"[A-Za-z]", w) for _s, _e, w in words) else ""
                    segments = _merge_caption_units(words, joiner=joiner)
                    if segments:
                        logger.info(
                            "FunASR timestamp parsed (token->word): words=%d, segments=%d",
                            len(words),
                            len(segments),
                        )
                        merged_text = "\n".join(seg.text for seg in segments).strip() or full_text
                        return merged_text, segments

                # No token text: fall back to whitespace split.
                tokens = [t for t in re.split(r"\s+", full_text) if t]
                if len(tokens) == len(token_entries):
                    units: list[tuple[float, float, str]] = []
                    for tok, (s, e, _t) in zip(tokens, token_entries, strict=False):
                        units.append((s, e, tok))
                    segments = _merge_caption_units(units, joiner=" ")
                    if segments:
                        logger.info(
                            "FunASR timestamp parsed (whitespace tokens): tokens=%d, segments=%d",
                            len(tokens),
                            len(segments),
                        )
                        merged_text = "\n".join(seg.text for seg in segments).strip() or full_text
                        return merged_text, segments

        return full_text, []

    # Case 2: list of dict segments.
    if all(isinstance(x, dict) for x in res):
        raw: list[tuple[float, float, str]] = []
        texts: list[str] = []
        for item in res:
            text = _maybe_postprocess_text(str(item.get("text", "") or ""))
            if text:
                texts.append(text)

            start = item.get("start", item.get("begin_time", item.get("start_time", None)))
            end = item.get("end", item.get("end_time", item.get("finish_time", None)))
            if start is None or end is None or not text:
                continue
            try:
                raw.append((float(start), float(end), text))
            except Exception:
                continue

        full_text = "\n".join([t for t in texts if t]).strip()
        if raw:
            max_end = max(e for _s, e, _t in raw)
            scale = 0.001 if max_end > max(duration_s, 1.0) * 10 else 1.0
            segments = [ASRSegment(start_s=s * scale, end_s=e * scale, text=t) for (s, e, t) in raw]
            return full_text, segments

        return full_text, []

    return "", []


def transcribe_file_funasr(
    *,
    file_path: str,
    model: str,
    device: str,
    language: str,
    use_itn: bool,
    enable_vad: bool,
    enable_punc: bool,
    duration_s: float,
) -> ASRResult:
    cfg = FunASRConfig(
        model=(model or "").strip(),
        device=(device or "").strip(),
        language=(language or "").strip() or "auto",
        use_itn=bool(use_itn),
        enable_vad=bool(enable_vad),
        enable_punc=bool(enable_punc),
    )
    if not cfg.model:
        raise RuntimeError("请先选择 FunASR 本地模型。")

    model_obj = _make_model(cfg)

    gen_kwargs: dict[str, Any] = {
        "input": file_path,
        "cache": {},
        "language": cfg.language,
        "use_itn": bool(cfg.use_itn),
        # Sensible defaults for long audio; ignored if the model doesn't accept them.
        "batch_size_s": 60,
        "merge_vad": True,
        "merge_length_s": 15,
    }
    try:
        res = model_obj.generate(**_filter_kwargs(model_obj.generate, gen_kwargs))
    except Exception as e:
        raise RuntimeError(f"FunASR 推理失败: {e}") from e

    text, segments = _extract_segments_from_result(res, duration_s=duration_s)
    return ASRResult(text=text, segments=segments)


def preload_funasr_model(
    *,
    model: str,
    device: str,
    enable_vad: bool = True,
    enable_punc: bool = True,
) -> None:
    """
    Preload a FunASR model into process memory (and cache it) to reduce first-run latency.
    """
    cfg = FunASRConfig(
        model=(model or "").strip(),
        device=(device or "").strip(),
        language="auto",
        use_itn=True,
        enable_vad=bool(enable_vad),
        enable_punc=bool(enable_punc),
    )
    if not cfg.model:
        raise RuntimeError("请先选择 FunASR 本地模型。")
    _make_model(cfg)


def download_funasr_model(
    *,
    model: str,
    enable_vad: bool = True,
    enable_punc: bool = True,
) -> None:
    """
    Best-effort: trigger FunASR model download into cache without keeping it in `_MODEL_CACHE`.

    FunASR/AutoModel internally decides where to cache model files (e.g. HuggingFace / ModelScope
    cache).
    """

    model = (model or "").strip()
    if not model:
        raise RuntimeError("请先选择 FunASR 本地模型。")

    AutoModel = _import_funasr()
    trust_remote_code = _needs_trust_remote_code(model)
    remote_code_candidates = _remote_code_candidates(model)
    model_kwargs: dict[str, Any] = {
        "model": model,
        "device": "cpu",
        "trust_remote_code": trust_remote_code,
        # FunASR 会在初始化时做版本更新检查 (可能较慢), 这里默认禁用。
        "disable_update": True,
    }
    if trust_remote_code and remote_code_candidates:
        model_kwargs["remote_code"] = remote_code_candidates[0]
    if enable_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": 60000}
    if enable_punc:
        model_kwargs["punc_model"] = "ct-punc"

    try:
        _ = AutoModel(**_filter_kwargs(AutoModel, model_kwargs))
    except Exception as e:
        _raise_if_missing_tokenizers_deps(e)
        if _is_not_registered_error(e):
            if not model_kwargs.get("trust_remote_code"):
                logger.warning(
                    "FunASR 模型未注册，尝试启用 trust_remote_code 重试: model=%s", model
                )
                model_kwargs["trust_remote_code"] = True
                if remote_code_candidates and "remote_code" not in model_kwargs:
                    model_kwargs["remote_code"] = remote_code_candidates[0]
                try:
                    _ = AutoModel(**_filter_kwargs(AutoModel, model_kwargs))
                except Exception as e2:
                    _raise_if_missing_tokenizers_deps(e2)
                    raise
            else:
                if not remote_code_candidates:
                    raise
                last_exc: Exception = e
                for cand in remote_code_candidates:
                    if model_kwargs.get("remote_code") == cand:
                        continue
                    logger.warning(
                        "FunASR 模型未注册，尝试 remote_code=%s 重试: model=%s", cand, model
                    )
                    model_kwargs["remote_code"] = cand
                    try:
                        _ = AutoModel(**_filter_kwargs(AutoModel, model_kwargs))
                        break
                    except Exception as e2:
                        _raise_if_missing_tokenizers_deps(e2)
                        last_exc = e2
                else:
                    raise last_exc
        else:
            raise
    logger.info(
        "FunASR 模型已下载/初始化(未缓存到进程内): model=%s, vad=%s, punc=%s",
        model,
        bool(enable_vad),
        bool(enable_punc),
    )


__all__ = ["download_funasr_model", "preload_funasr_model", "transcribe_file_funasr"]
