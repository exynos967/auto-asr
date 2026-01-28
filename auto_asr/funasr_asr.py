from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from auto_asr.openai_asr import ASRResult, ASRSegment

logger = logging.getLogger(__name__)


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
    try:
        from funasr.utils.postprocess_utils import (  # type: ignore
            rich_transcription_postprocess,
        )

        return rich_transcription_postprocess(text).strip()
    except Exception:
        return text


def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(func)
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

    model_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "device": cfg.device,
        # Some models (e.g. SenseVoiceSmall) require remote code to enable full features.
        "trust_remote_code": True,
    }

    # Try to enable built-in VAD / punctuation when requested. These are common in FunASR.
    if cfg.enable_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": 60000}
    if cfg.enable_punc:
        model_kwargs["punc_model"] = "ct-punc"

    model = AutoModel(**_filter_kwargs(AutoModel, model_kwargs))
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
                scale = 0.001 if max_end > max(duration_s, 1.0) * 10 else 1.0
                segments = [
                    ASRSegment(start_s=s * scale, end_s=e * scale, text=t) for (s, e, t) in raw
                ]
                return full_text, segments

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
    model_kwargs: dict[str, Any] = {
        "model": model,
        "device": "cpu",
        "trust_remote_code": True,
    }
    if enable_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": 60000}
    if enable_punc:
        model_kwargs["punc_model"] = "ct-punc"

    _ = AutoModel(**_filter_kwargs(AutoModel, model_kwargs))
    logger.info(
        "FunASR 模型已下载/初始化(未缓存到进程内): model=%s, vad=%s, punc=%s",
        model,
        bool(enable_vad),
        bool(enable_punc),
    )


__all__ = ["download_funasr_model", "preload_funasr_model", "transcribe_file_funasr"]
