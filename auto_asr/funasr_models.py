from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from auto_asr.model_hub import configure_model_cache_env, link_or_copy_dir, snapshot_download

logger = logging.getLogger(__name__)

_MODEL_DIR_CACHE: dict[str, str] = {}

PostDownloadHook = Callable[[str, Path], None]
RemoteCodeHook = Callable[[str, str], list[str]]

_POST_DOWNLOAD_HOOKS: list[PostDownloadHook] = []
_REMOTE_CODE_HOOKS: list[RemoteCodeHook] = []


def register_post_download_hook(hook: PostDownloadHook) -> None:
    _POST_DOWNLOAD_HOOKS.append(hook)


def register_remote_code_hook(hook: RemoteCodeHook) -> None:
    _REMOTE_CODE_HOOKS.append(hook)


def is_funasr_nano(model: str) -> bool:
    m = (model or "").lower()
    return ("fun-asr-nano" in m) or ("funasrnano" in m)


def _ensure_funasr_nano_deps(model_dir: Path) -> None:
    """
    Fun-ASR-Nano-2512 depends on Qwen3-0.6B weights. Some deployments require it to be placed
    under the ASR model directory: `<Fun-ASR-Nano-2512>/Qwen3-0.6B`.
    """

    qwen_dst = model_dir / "Qwen3-0.6B"
    if qwen_dst.exists():
        return

    qwen_src = snapshot_download("Qwen/Qwen3-0.6B")
    logger.info("FunASR-Nano 依赖已下载: qwen=%s", qwen_src)
    link_or_copy_dir(src=qwen_src, dst=qwen_dst)
    logger.info("FunASR-Nano 依赖已放置: %s -> %s", qwen_src, qwen_dst)


def resolve_model_dir(model: str) -> str:
    """Resolve FunASR model input to a local directory (download if a RepoID is provided)."""

    model = (model or "").strip()
    if not model:
        return model

    # Ensure caches are project-local before any downstream libs initialize.
    configure_model_cache_env()

    p = Path(model)
    if p.exists():
        return str(p)

    cached = _MODEL_DIR_CACHE.get(model)
    if cached:
        return cached

    local_dir = snapshot_download(model)
    for hook in _POST_DOWNLOAD_HOOKS:
        hook(model, local_dir)

    _MODEL_DIR_CACHE[model] = str(local_dir)
    logger.info("FunASR 模型下载目录(项目内): model=%s, dir=%s", model, local_dir)
    return str(local_dir)


def get_remote_code_candidates(*, model: str, model_dir_or_id: str) -> list[str]:
    """Get remote_code candidates for FunASR AutoModel (best-effort, model-specific)."""

    model = (model or "").strip()
    candidates: list[str] = []

    for hook in _REMOTE_CODE_HOOKS:
        candidates.extend(hook(model, model_dir_or_id))

    try:
        md = Path(model_dir_or_id)
        if md.exists():
            cand = md / "model.py"
            if cand.exists():
                candidates.append(str(cand))
    except Exception:
        pass

    # Dedupe while keeping order.
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _funasr_nano_post_download(model: str, model_dir: Path) -> None:
    if is_funasr_nano(model):
        _ensure_funasr_nano_deps(model_dir)


def _funasr_nano_remote_code_candidates(model: str, _model_dir_or_id: str) -> list[str]:
    if not is_funasr_nano(model):
        return []

    try:
        # Import for side-effects: register FunASRNano into FunASR registry.
        from funasr.models.fun_asr_nano import model as nano_model  # type: ignore

        nano_file = getattr(nano_model, "__file__", None)
        if nano_file:
            return [str(Path(nano_file).resolve())]
    except Exception as e:
        logger.warning("导入 FunASRNano 失败（可能导致模型未注册）: %s", e)

    return []


register_post_download_hook(_funasr_nano_post_download)
register_remote_code_hook(_funasr_nano_remote_code_candidates)


__all__ = [
    "get_remote_code_candidates",
    "is_funasr_nano",
    "register_post_download_hook",
    "register_remote_code_hook",
    "resolve_model_dir",
]
