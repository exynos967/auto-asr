from __future__ import annotations

from auto_asr.funasr_asr import _extract_segments_from_result, _maybe_postprocess_text


def test_extract_segments_sentence_info_ms_scaled_to_seconds():
    res = [
        {
            "text": "hello",
            "sentence_info": [
                {"start": 0, "end": 1000, "text": "hi"},
                {"start": 1000, "end": 2500, "text": "there"},
            ],
        }
    ]
    full_text, segments = _extract_segments_from_result(res, duration_s=10.0)
    assert full_text == "hello"
    assert len(segments) == 2
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 1.0
    assert segments[0].text == "hi"
    assert segments[1].start_s == 1.0
    assert segments[1].end_s == 2.5
    assert segments[1].text == "there"


def test_extract_segments_list_of_dict_seconds_kept():
    res = [
        {"start": 0.0, "end": 1.2, "text": "a"},
        {"start": 1.2, "end": 2.0, "text": "b"},
    ]
    full_text, segments = _extract_segments_from_result(res, duration_s=30.0)
    assert full_text == "a\nb"
    assert [(s.start_s, s.end_s, s.text) for s in segments] == [
        (0.0, 1.2, "a"),
        (1.2, 2.0, "b"),
    ]


def test_maybe_postprocess_text_strips_sensevoice_rich_tags_without_gt():
    raw = "< | ja |  < | EMO _ UNKNOWN |  < | S pe ech |  < | withi tn | hello"
    assert _maybe_postprocess_text(raw) == "hello"


def test_extract_segments_timestamp_tokens_sentencepiece_like():
    res = [
        {
            "text": "< | ja |  < | EMO _ UNKNOWN |  < | S pe ech |  < | withi tn | hello world",
            "timestamp": [
                ["▁hello", 0.0, 0.5],
                ["▁world", 0.6, 1.0],
            ],
        }
    ]
    full_text, segments = _extract_segments_from_result(res, duration_s=10.0)
    assert full_text == "hello world"
    assert segments
    assert segments[0].start_s == 0.0
    assert segments[-1].end_s == 1.0
