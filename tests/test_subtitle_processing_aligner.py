from __future__ import annotations


def test_subtitle_aligner_fills_missing_target_line_with_previous():
    from auto_asr.subtitle_processing.alignment import SubtitleAligner

    src = ["ab", "b", "c", "d", "e", "f", "g", "h", "i"]
    tgt = ["a", "b", "c", "d", "f", "g", "h", "i"]

    aligner = SubtitleAligner()
    aligned_source, aligned_target = aligner.align_texts(src, tgt)

    assert aligned_source == src
    assert aligned_target == ["a", "b", "c", "d", "d", "f", "g", "h", "i"]

