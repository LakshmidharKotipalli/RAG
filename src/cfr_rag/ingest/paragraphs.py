"""Infer the nesting of CFR paragraph designations: (a) > (1) > (i) > (A) > (1) > (i).

The eCFR XML stores each paragraph as a flat ``<P>`` whose text starts with its
markers; the hierarchy is implied by the marker sequence. The tricky cases are
letters that are also roman numerals (``(i)``, ``(v)``, ``(x)``): ``(i)`` after
``(h)`` is a letter, but ``(i)`` after ``(1)`` is a roman numeral. We resolve
those by preferring, in order: the next sibling at the current level, the first
child of the current level, then the next value of an ancestor level.
"""

from __future__ import annotations

import re

LOWER, DIGIT, ROMAN, UPPER, IDIGIT, IROMAN = range(6)

ITALIC_OPEN = "\x01"
ITALIC_CLOSE = "\x02"

_ROMAN_RE = re.compile(r"^(?=[ivxlc])(c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
_MARKER_RE = re.compile(
    rf"^\s*\(({ITALIC_OPEN}?)([a-z]{{1,5}}|[A-Z]{{1,3}}|\d{{1,3}})({ITALIC_CLOSE}?)\)\s*"
)

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def is_roman(s: str) -> bool:
    return bool(s) and bool(_ROMAN_RE.match(s))


def roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN_VALUES[ch]
        nxt = _ROMAN_VALUES[s[i + 1]] if i + 1 < len(s) else 0
        total += -v if v < nxt else v
    return total


def int_to_roman(n: int) -> str:
    vals = [(100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"),
            (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def _next_alpha(s: str) -> str:
    # CFR convention: z is followed by aa, bb, ...
    if len(set(s)) == 1 and s[0] in "zZ":
        return ("a" if s[0] == "z" else "A") * (len(s) + 1)
    if len(set(s)) == 1:
        return chr(ord(s[0]) + 1) * len(s)
    return s


def next_marker(level: int, marker: str) -> str:
    """Return the marker expected after ``marker`` at ``level``."""
    if level in (DIGIT, IDIGIT):
        return str(int(marker) + 1)
    if level in (ROMAN, IROMAN):
        return int_to_roman(roman_to_int(marker) + 1)
    return _next_alpha(marker)


FIRST = {LOWER: "a", DIGIT: "1", ROMAN: "i", UPPER: "A", IDIGIT: "1", IROMAN: "i"}


def candidate_levels(marker: str, italic: bool) -> list[int]:
    if marker.isdigit():
        return [IDIGIT] if italic else [DIGIT, IDIGIT]
    if marker.islower():
        if italic:
            return [IROMAN]
        return [LOWER, ROMAN, IROMAN] if is_roman(marker) else [LOWER]
    return [UPPER]


class ParagraphNester:
    """Stateful resolver turning a stream of markers into full paragraph paths."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []
        self.saw_italic = False  # once italic markers appear, plain markers never use italic levels

    def push(self, marker: str, italic: bool = False) -> int:
        """Place ``marker`` in the hierarchy and return its level."""
        self.saw_italic = self.saw_italic or italic
        levels = candidate_levels(marker, italic)
        if self.saw_italic and not italic:
            levels = [lvl for lvl in levels if lvl not in (IDIGIT, IROMAN)] or levels
        if self.stack:
            top_level, top_marker = self.stack[-1]
            # 1. next sibling at the current level
            if top_level in levels and marker == next_marker(top_level, top_marker):
                self.stack[-1] = (top_level, marker)
                return top_level
            # 2. first child of the current level
            for lvl in levels:
                if lvl > top_level and marker == FIRST[lvl] and lvl == self._child_of(top_level, levels):
                    self.stack.append((lvl, marker))
                    return lvl
            # 3. next value of an ancestor level
            for idx in range(len(self.stack) - 2, -1, -1):
                lvl, m = self.stack[idx]
                if lvl in levels and marker == next_marker(lvl, m):
                    del self.stack[idx + 1:]
                    self.stack[-1] = (lvl, marker)
                    return lvl
            # 4. fallback: same level as an existing ancestor of that kind, else deeper
            for idx in range(len(self.stack) - 1, -1, -1):
                lvl, _ = self.stack[idx]
                if lvl in levels:
                    del self.stack[idx:]
                    self.stack.append((lvl, marker))
                    return lvl
            lvl = next((x for x in levels if x > top_level), levels[0])
            if lvl <= top_level:
                self.stack.clear()
            self.stack.append((lvl, marker))
            return lvl
        lvl = levels[0]
        self.stack.append((lvl, marker))
        return lvl

    @staticmethod
    def _child_of(level: int, levels: list[int]) -> int:
        deeper = [x for x in levels if x > level]
        return min(deeper) if deeper else -1

    @property
    def path(self) -> str:
        return "".join(f"({m})" for _, m in self.stack)


def split_leading_markers(text: str) -> tuple[list[tuple[str, bool]], str]:
    """Strip chained leading markers such as ``(b)(1)`` from ``text``.

    Returns the list of ``(marker, italic)`` pairs and the remaining text.
    """
    markers: list[tuple[str, bool]] = []
    rest = text
    while True:
        m = _MARKER_RE.match(rest)
        if not m:
            break
        markers.append((m.group(2), bool(m.group(1) or m.group(3))))
        rest = rest[m.end():]
    return markers, rest
