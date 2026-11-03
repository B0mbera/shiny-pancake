"""Structure-aware chunking: paragraphs first, then sentences, then hard splits; with overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int
    end: int

    @property
    def chars(self) -> int:
        return len(self.text)


def _segments(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Split into (start, end) spans no longer than max_chars, preferring natural boundaries."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in [*_PARAGRAPH_BREAK.finditer(text), None]:
        end = match.start() if match else len(text)
        if end > pos:
            spans.extend(_split_long(text, pos, end, max_chars))
        pos = match.end() if match else len(text)
    return spans


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    if end - start <= max_chars:
        return [(start, end)]
    out: list[tuple[int, int]] = []
    cur = start
    boundaries = [m.end() + start for m in _SENTENCE_END.finditer(text[start:end])] + [end]
    last = cur
    for b in boundaries:
        if b - cur > max_chars:
            if last > cur:
                out.append((cur, last))
                cur = last
            while b - cur > max_chars:  # a single enormous "sentence": hard split on whitespace
                cut = text.rfind(" ", cur, cur + max_chars)
                cut = cut if cut > cur else cur + max_chars
                out.append((cur, cut))
                cur = cut
        last = b
    if end > cur:
        out.append((cur, end))
    return out


def chunk_text(text: str, *, max_chars: int = 6000, overlap: int = 200) -> list[Chunk]:
    """Pack natural segments into chunks of at most ``max_chars`` (+ overlap from the previous chunk)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [Chunk(0, text, 0, len(text))]
    chunks: list[Chunk] = []
    start = end = None
    for s, e in _segments(text, max_chars):
        if start is None:
            start, end = s, e
        elif e - start <= max_chars:
            end = e
        else:
            chunks.append(_make(text, len(chunks), start, end, overlap))
            start, end = s, e
    if start is not None:
        chunks.append(_make(text, len(chunks), start, end, overlap))
    return chunks


def _make(text: str, index: int, start: int, end: int, overlap: int) -> Chunk:
    real_start = start
    if index > 0 and overlap > 0:
        # carry the last complete sentence(s) of the previous chunk, never a fragment
        window = text[max(0, start - overlap) : start]
        boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("\n"))
        if boundary != -1:
            real_start = start - len(window) + boundary + 1
    return Chunk(index, text[real_start:end].strip(), real_start, end)
