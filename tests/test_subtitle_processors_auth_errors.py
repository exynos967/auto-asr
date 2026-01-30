from __future__ import annotations

import pytest

from auto_asr.subtitle_processing.base import ProcessorContext
from auto_asr.subtitle_processing.processors.optimize import OptimizeProcessor
from auto_asr.subtitle_processing.processors.split import SplitProcessor
from auto_asr.subtitle_processing.processors.translate import TranslateProcessor
from auto_asr.subtitles import SubtitleLine


class DummyAuthError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("proc", "options"),
    [
        (TranslateProcessor(), {"target_language": "en", "batch_size": 10, "concurrency": 1}),
        (OptimizeProcessor(), {"batch_size": 10, "concurrency": 1}),
        (SplitProcessor(), {"mode": "inplace_newlines", "concurrency": 1}),
    ],
)
def test_subtitle_processors_raise_on_auth_error(proc, options):
    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        raise DummyAuthError(401)

    ctx = ProcessorContext(chat_json=chat_json)
    lines = [SubtitleLine(start_s=0.0, end_s=1.0, text="hello")]

    with pytest.raises(DummyAuthError):
        proc.process(lines, ctx=ctx, options=options)

