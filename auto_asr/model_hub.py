from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_models_dir() -> Path:
    path = get_project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_model_cache_env(models_dir: Path | None = None) -> Path:
    """Configure ModelScope/HuggingFace caches to live under the project `./models` dir."""

    models_dir = models_dir or get_models_dir()

    # ModelScope uses `${MODELSCOPE_CACHE}/hub/...`.
    os.environ.setdefault("MODELSCOPE_CACHE", str(models_dir))

    # HuggingFace/transformers caches. Keep them in a subdir to avoid mixing with ModelScope.
    hf_home = models_dir / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    # TRANSFORMERS_CACHE is deprecated but still respected by some stacks.
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    return models_dir


def link_or_copy_dir(*, src: Path, dst: Path) -> None:
    """Best-effort create a directory link; fallback to copy.

    Note: Windows often needs admin permissions for directory symlinks.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    try:
        dst.symlink_to(src, target_is_directory=True)
        return
    except Exception:
        pass

    shutil.copytree(src, dst, dirs_exist_ok=True)


def snapshot_download(model_id: str, *, models_dir: Path | None = None) -> Path:
    """Download a model to the project `./models` dir and return its local directory."""

    model_id = (model_id or "").strip()
    if not model_id:
        raise RuntimeError("model_id 为空，无法下载。")

    models_dir = configure_model_cache_env(models_dir)

    # Prefer ModelScope (often faster/more accessible in CN). Fall back to HuggingFace.
    ms_exc: Exception | None = None
    try:
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore

        local_dir = snapshot_download(model_id, cache_dir=str(models_dir))
        return Path(local_dir)
    except Exception as e:  # pragma: no cover
        ms_exc = e

    try:
        from huggingface_hub import snapshot_download  # type: ignore

        local_dir = snapshot_download(model_id, cache_dir=str(models_dir / "huggingface"))
        return Path(local_dir)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"模型下载失败: {model_id} (ModelScope/HF). modelscope_err={ms_exc}"
        ) from e


__all__ = [
    "configure_model_cache_env",
    "get_models_dir",
    "get_project_root",
    "link_or_copy_dir",
    "snapshot_download",
]
