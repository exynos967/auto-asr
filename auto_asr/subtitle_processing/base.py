from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from auto_asr.subtitles import SubtitleLine


@dataclass(frozen=True)
class ProcessorContext:
    """Inputs shared by all processors."""

    llm_chat_json: Callable[..., dict[str, str]]
    llm_model: str


class SubtitleProcessor:
    """Base class for subtitle processors.

    Subclasses should set `name` and implement `process`.
    """

    name: str = ""

    def process(self, lines: list[SubtitleLine], *, ctx: ProcessorContext, options: dict) -> list[SubtitleLine]:
        raise NotImplementedError


_PROCESSORS: dict[str, type[SubtitleProcessor]] = {}


def register_processor(cls: type[SubtitleProcessor]) -> type[SubtitleProcessor]:
    name = (getattr(cls, "name", "") or "").strip()
    if not name:
        raise ValueError("processor must define non-empty `name`")
    _PROCESSORS[name] = cls
    return cls


def list_processors() -> list[str]:
    return sorted(_PROCESSORS.keys())


def get_processor(name: str) -> type[SubtitleProcessor]:
    key = (name or "").strip()
    if key not in _PROCESSORS:
        raise KeyError(key)
    return _PROCESSORS[key]


__all__ = ["ProcessorContext", "SubtitleProcessor", "get_processor", "list_processors", "register_processor"]

