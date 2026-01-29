from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CONFIG_FILE_NAME = ".auto_asr_config.json"


def get_config_path() -> Path:
    # Project-local config (ignored by git). Makes it easy to keep settings per project.
    root = Path(__file__).resolve().parents[1]
    return root / _CONFIG_FILE_NAME


def load_config() -> dict[str, Any]:
    path = get_config_path()
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def save_config(config: dict[str, Any]) -> Path:
    path = get_config_path()
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Best-effort: restrict permissions on POSIX.
    try:
        if os.name == "posix":
            os.chmod(path, 0o600)
    except Exception:
        pass

    return path


def update_config(patch: dict[str, Any]) -> Path:
    """Load existing config, merge a patch, and save it back.

    Special-case: if patch contains an empty API key, keep the existing one.
    """
    cfg = load_config()

    for key, value in (patch or {}).items():
        if key in {"openai_api_key", "subtitle_openai_api_key"} and not str(value or "").strip():
            continue
        cfg[key] = value

    return save_config(cfg)


def delete_config() -> bool:
    path = get_config_path()
    if not path.exists():
        return False
    path.unlink()
    return True


__all__ = ["delete_config", "get_config_path", "load_config", "save_config", "update_config"]
