from __future__ import annotations

import logging
from dataclasses import dataclass
from gc import collect as gc_collect
from typing import Any

import numpy as np

from auto_asr.model_hub import configure_model_cache_env, snapshot_download
from auto_asr.openai_asr import ASRResult, ASRSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qwen3ASRConfig:
    model: str
    forced_aligner: str
    device: str = "auto"
    max_inference_batch_size: int = 8
    max_new_tokens: int = 1024


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def resolve_qwen3_language(language: str | None) -> str | None:
    """Map Auto-ASR language codes to Qwen3-ASR language names.

    Qwen3-ASR accepts names like "Chinese"/"English", or None for auto-detect.
    """
    code = (language or "").strip()
    if not code:
        return None
    code_l = code.lower()
    if code_l == "auto":
        return None

    mapping = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }
    return mapping.get(code_l, code)


def _import_qwen_asr() -> Any:
    try:
        from qwen_asr import Qwen3ASRModel  # type: ignore

        return Qwen3ASRModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未安装 Qwen3-ASR 依赖，无法使用 Qwen3 本地推理。"
            "请先安装：`uv sync --extra qwen3asr` 或 `uv pip install -U qwen-asr`"
        ) from e


def _resolve_device(device: str) -> str:
    d = (device or "").strip().lower()
    if d in {"", "auto"}:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"
    return device


def _resolve_dtype(device: str):
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        return None

    if str(device).startswith("cuda"):
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def download_qwen3_models(*, model: str, forced_aligner: str) -> tuple[str, str]:
    """Download Qwen3-ASR + forced aligner models to the project ./models cache."""
    configure_model_cache_env()
    asr_dir = snapshot_download(model)
    fa_dir = snapshot_download(forced_aligner)
    return str(asr_dir), str(fa_dir)


def _make_model(cfg: Qwen3ASRConfig) -> Any:
    key = (cfg.model, cfg.forced_aligner, _resolve_device(cfg.device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    configure_model_cache_env()

    device_map = key[2]
    dtype = _resolve_dtype(device_map)

    Qwen3ASRModel = _import_qwen_asr()

    logger.info(
        "加载 Qwen3-ASR 模型: model=%s, forced_aligner=%s, device=%s, dtype=%s",
        cfg.model,
        cfg.forced_aligner,
        device_map,
        getattr(dtype, "__name__", str(dtype)),
    )

    model = Qwen3ASRModel.from_pretrained(
        cfg.model,
        dtype=dtype,
        device_map=device_map,
        forced_aligner=cfg.forced_aligner,
        forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
        max_inference_batch_size=int(cfg.max_inference_batch_size),
        max_new_tokens=int(cfg.max_new_tokens),
    )

    _MODEL_CACHE[key] = model
    return model


def preload_qwen3_model(cfg: Qwen3ASRConfig) -> Any:
    """Preload model into cache (used by WebUI 'load model' button)."""
    return _make_model(cfg)


def release_qwen3_resources() -> int:
    """Release cached Qwen3-ASR models and try to free GPU memory."""
    cleared = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    gc_collect()

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info("Qwen3-ASR 资源清理完成: cleared_models=%d", cleared)
    return cleared


def transcribe_chunks_qwen3(
    *,
    chunks: list[np.ndarray],
    cfg: Qwen3ASRConfig,
    language: str | None,
    return_time_stamps: bool,
    sample_rate: int,
) -> list[ASRResult]:
    """Transcribe audio chunks with Qwen3-ASR (transformers backend)."""
    model = _make_model(cfg)

    lang_name = resolve_qwen3_language(language)
    lang_list = [lang_name for _ in chunks]

    audio_inputs = [(w, int(sample_rate)) for w in chunks]
    results = model.transcribe(
        audio=audio_inputs,
        language=lang_list,
        return_time_stamps=bool(return_time_stamps),
    )

    out: list[ASRResult] = []
    for r in results:
        text = str(getattr(r, "text", "") or "").strip()
        segs: list[ASRSegment] = []
        if return_time_stamps:
            ts_list = getattr(r, "time_stamps", None)
            if ts_list:
                for ts in ts_list:
                    seg_text = str(getattr(ts, "text", "") or "").strip()
                    if not seg_text:
                        continue
                    segs.append(
                        ASRSegment(
                            start_s=float(getattr(ts, "start_time", 0.0) or 0.0),
                            end_s=float(getattr(ts, "end_time", 0.0) or 0.0),
                            text=seg_text,
                        )
                    )
        out.append(ASRResult(text=text, segments=segs))
    return out


__all__ = [
    "Qwen3ASRConfig",
    "download_qwen3_models",
    "preload_qwen3_model",
    "release_qwen3_resources",
    "resolve_qwen3_language",
    "transcribe_chunks_qwen3",
]
