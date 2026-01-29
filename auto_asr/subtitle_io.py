from __future__ import annotations

from pathlib import Path

from .subtitles import SubtitleLine


def _parse_timestamp(raw: str) -> float:
    value = (raw or "").strip()
    if not value:
        raise ValueError("empty timestamp")

    sep = "," if "," in value else "."
    if sep in value:
        hms, ms_raw = value.split(sep, 1)
    else:
        hms, ms_raw = value, "0"

    parts = hms.split(":")
    if len(parts) == 2:
        h = 0
        m = int(parts[0])
        s = int(parts[1])
    elif len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
    else:
        raise ValueError(f"invalid timestamp: {raw!r}")

    ms = int((ms_raw + "000")[:3])
    return (h * 3600) + (m * 60) + s + (ms / 1000.0)


def _parse_time_range(line: str) -> tuple[float, float]:
    if "-->" not in line:
        raise ValueError(f"invalid time range: {line!r}")
    start_raw, end_raw = line.split("-->", 1)
    start_s = _parse_timestamp(start_raw.strip())
    end_part = end_raw.strip().split()[0]
    end_s = _parse_timestamp(end_part)
    return start_s, end_s


def _parse_srt(text: str) -> list[SubtitleLine]:
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[SubtitleLine] = []

    block: list[str] = []
    for line in lines + [""]:
        if line.strip():
            block.append(line)
            continue

        if not block:
            continue

        i = 0
        if block[0].strip().isdigit():
            i += 1
        if i >= len(block):
            block = []
            continue

        start_s, end_s = _parse_time_range(block[i])
        i += 1
        text_lines = block[i:]
        cue_text = "\n".join(t.rstrip() for t in text_lines).strip()
        if cue_text:
            out.append(SubtitleLine(start_s=start_s, end_s=end_s, text=cue_text))

        block = []

    return out


def _parse_vtt(text: str) -> list[SubtitleLine]:
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    idx = 0

    # Skip WEBVTT header and metadata until the first blank line.
    if idx < len(lines) and lines[idx].strip().upper().startswith("WEBVTT"):
        idx += 1
        while idx < len(lines) and lines[idx].strip():
            idx += 1
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

    out: list[SubtitleLine] = []
    block: list[str] = []

    def flush_block() -> None:
        nonlocal block
        if not block:
            return

        # NOTE blocks: skip entirely.
        if block[0].strip().upper().startswith("NOTE"):
            block = []
            return

        i = 0
        if "-->" not in block[0] and len(block) > 1 and "-->" in block[1]:
            i = 1  # cue identifier line

        if "-->" not in block[i]:
            block = []
            return

        start_s, end_s = _parse_time_range(block[i])
        cue_text = "\n".join(t.rstrip() for t in block[i + 1 :]).strip()
        if cue_text:
            out.append(SubtitleLine(start_s=start_s, end_s=end_s, text=cue_text))
        block = []

    for line in lines[idx:] + [""]:
        if line.strip():
            block.append(line)
            continue
        flush_block()

    return out


def load_subtitle_file(path: str) -> list[SubtitleLine]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8-sig")

    ext = p.suffix.lower()
    if ext == ".vtt":
        return _parse_vtt(raw)
    if ext == ".srt":
        return _parse_srt(raw)

    # Fallback: guess by header.
    if raw.lstrip().upper().startswith("WEBVTT"):
        return _parse_vtt(raw)
    return _parse_srt(raw)


__all__ = ["load_subtitle_file"]

