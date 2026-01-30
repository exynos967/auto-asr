from __future__ import annotations

import functools
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).parent / "prompts"


@functools.lru_cache(maxsize=32)
def _load_prompt_file(prompt_path: str) -> str:
    file_path = PROMPTS_DIR / f"{prompt_path}.md"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}.md\nExpected location: {file_path}"
        )
    return file_path.read_text(encoding="utf-8")


def get_prompt(prompt_path: str, **kwargs) -> str:
    raw = _load_prompt_file(prompt_path)
    if not kwargs:
        return raw
    return Template(raw).safe_substitute(**kwargs)


def reload_cache() -> None:
    _load_prompt_file.cache_clear()


__all__ = ["get_prompt", "reload_cache"]

