from __future__ import annotations

import pytest


def test_get_prompt_substitutes_template_vars():
    from auto_asr.subtitle_processing.prompts import get_prompt

    text = get_prompt("split/sentence", max_word_count_cjk=18, max_word_count_english=12)
    assert "18" in text
    assert "12" in text


def test_get_prompt_raises_when_missing():
    from auto_asr.subtitle_processing.prompts import get_prompt

    with pytest.raises(FileNotFoundError):
        get_prompt("nope/not-exists")

