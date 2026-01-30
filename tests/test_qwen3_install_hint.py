from __future__ import annotations

import importlib.util

import pytest


def test_qwen3_import_error_suggests_transformers_extra():
    if importlib.util.find_spec("qwen_asr") is not None:
        pytest.skip("qwen_asr is installed; missing-deps hint not applicable")

    from auto_asr import qwen3_asr

    with pytest.raises(RuntimeError) as excinfo:
        qwen3_asr._import_qwen_asr()

    msg = str(excinfo.value)
    assert "--extra transformers" in msg
    assert "--extra qwen3asr" not in msg
