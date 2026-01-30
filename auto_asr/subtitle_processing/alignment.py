from __future__ import annotations

import difflib


class SubtitleAligner:
    """Align two subtitle text sequences and fill missing target items.

    This is adapted from VideoCaptioner to keep subtitle line counts stable when
    LLM outputs accidentally merge/drop items. When a target line is missing, the
    aligner fills it using the previous target line.
    """

    def __init__(self):
        self.line_numbers = [0, 0]

    def align_texts(self, source_text: list[str], target_text: list[str]):
        diff_iterator = difflib.ndiff(source_text, target_text)
        return self._pair_lines(diff_iterator)

    def _pair_lines(self, diff_iterator):
        source_lines: list[str] = []
        target_lines: list[str] = []
        flag = 0

        for source_line, target_line, _ in self._line_iterator(diff_iterator):
            if source_line is not None:
                if source_line[1] == "\n":
                    flag += 1
                    continue
                source_lines.append(source_line[1])
            if target_line is not None:
                if flag > 0:
                    flag -= 1
                    continue
                target_lines.append(target_line[1])

        for i in range(1, len(target_lines)):
            if target_lines[i] == "\n":
                target_lines[i] = target_lines[i - 1]

        return source_lines, target_lines

    def _line_iterator(self, diff_iterator):
        lines: list[str] = []
        blank_lines_pending = 0
        blank_lines_to_yield = 0

        while True:
            while len(lines) < 4:
                lines.append(next(diff_iterator, "X"))

            diff_type = "".join([line[0] for line in lines])

            if diff_type.startswith("X"):
                blank_lines_to_yield = blank_lines_pending
            elif diff_type.startswith("-?+?"):
                yield (
                    self._format_line(lines, "?", 0),
                    self._format_line(lines, "?", 1),
                    True,
                )
                continue
            elif diff_type.startswith("--++"):
                blank_lines_pending -= 1
                yield self._format_line(lines, "-", 0), None, True
                continue
            elif diff_type.startswith(("--?+", "--+", "- ")):
                source_line, target_line = self._format_line(lines, "-", 0), None
                blank_lines_to_yield, blank_lines_pending = blank_lines_pending - 1, 0
            elif diff_type.startswith("-+?"):
                yield (
                    self._format_line(lines, None, 0),
                    self._format_line(lines, "?", 1),
                    True,
                )
                continue
            elif diff_type.startswith("-?+"):
                yield (
                    self._format_line(lines, "?", 0),
                    self._format_line(lines, None, 1),
                    True,
                )
                continue
            elif diff_type.startswith("-"):
                blank_lines_pending -= 1
                yield self._format_line(lines, "-", 0), None, True
                continue
            elif diff_type.startswith("+--"):
                blank_lines_pending += 1
                yield None, self._format_line(lines, "+", 1), True
                continue
            elif diff_type.startswith(("+ ", "+-")):
                source_line, target_line = None, self._format_line(lines, "+", 1)
                blank_lines_to_yield, blank_lines_pending = blank_lines_pending + 1, 0
            elif diff_type.startswith("+"):
                blank_lines_pending += 1
                yield None, self._format_line(lines, "+", 1), True
                continue
            elif diff_type.startswith(" "):
                yield (
                    self._format_line(lines[:], None, 0),
                    self._format_line(lines, None, 1),
                    False,
                )
                continue

            while blank_lines_to_yield < 0:
                blank_lines_to_yield += 1
                yield None, ("", "\n"), True
            while blank_lines_to_yield > 0:
                blank_lines_to_yield -= 1
                yield ("", "\n"), None, True

            if diff_type.startswith("X"):
                return
            yield source_line, target_line, True

    def _format_line(self, lines, format_key, side):
        self.line_numbers[side] += 1
        if format_key is None:
            return self.line_numbers[side], lines.pop(0)[2:]
        if format_key == "?":
            text = lines.pop(0)
            lines.pop(0)  # Skip markers line
            text = text[2:]
        else:
            text = lines.pop(0)[2:]
            if not text:
                text = ""
        return self.line_numbers[side], text


__all__ = ["SubtitleAligner"]

