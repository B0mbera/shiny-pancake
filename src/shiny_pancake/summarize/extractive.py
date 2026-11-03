"""A tiny frequency-based extractive summarizer (no model, no dependencies).

Used by the offline demo provider so the whole multi-worker pipeline produces *meaningful*
output without an API key, and handy as a cheap pre-filter for very long inputs.
"""

from __future__ import annotations

import re
from collections import Counter

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ0-9\"'(\[])")
_WORD = re.compile(r"[\wáéíóöőúüű']+", re.IGNORECASE)

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have he her his how i if in
    into is it its just may me more most my no not of on or our out over so some such than that the their
    them then there these they this those to too up us was we were what when where which while who why will
    with would you your also each other only very about after before between both during through under
    should must shall all any own same few being
    az egy és hogy nem is de meg mint vagy ha már még csak ez ezt azt itt ott van volt lesz kell lehet
    """.split()
)

_ACTION = re.compile(
    r"\b(must|should|need to|needs to|will|todo|action|decide[ds]?|owner|deadline|by (mon|tue|wed|thu|fri)"
    r"|next step|follow[- ]up|assign(ed)?|ship|kell|teendő|határidő|felelős)\b",
    re.IGNORECASE,
)
_RISK = re.compile(
    r"\b(risk|risks|danger|fail(s|ure)?|caveat|concern|unknown|open question|assum(e|ption)|cost|latency"
    r"|security|injection|leak|outage|limit(s|ation)?|kockázat|veszély)\b",
    re.IGNORECASE,
)


_LIST_ITEM = re.compile(r"^\s*(?:[-*•>]|\d+[.)]|- \[[ xX]\])\s+")


def _units(text: str) -> list[str]:
    """Paragraphs with hard-wrapped lines re-joined; list items stay separate units."""
    units: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            units.append(" ".join(buffer))
            buffer.clear()

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "```", "|")) or set(stripped) <= set("-=|:_ "):
            flush()  # blank lines, headings, code fences and tables end a paragraph
            continue
        if _LIST_ITEM.match(raw):
            flush()
            buffer.append(_LIST_ITEM.sub("", raw).strip())
            continue
        buffer.append(stripped)
    flush()
    return units


_OWNER = re.compile(
    r"^(?:[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+|The team|We)\s+(?:will|must|should|needs? to|is going to)\b"
)
_DEADLINE = re.compile(
    r"\b(?:by|before|until|határidő)\s+(?:\w+\s+)?(?:\d{1,2}|mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|"
    r"jul|aug|sep|oct|nov|dec|end of|next|the)\w*",
    re.IGNORECASE,
)


def action_signals(sentence: str) -> int:
    """How strongly a sentence looks like an action item: owner + modal verb, deadline, verbs."""
    score = len(_ACTION.findall(sentence))
    score += 3 if _OWNER.search(sentence) else 0
    score += 2 if _DEADLINE.search(sentence) else 0
    score -= 3 if re.search(r"\baction items?\b", sentence, re.IGNORECASE) else 0  # meta sentences
    return score


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for unit in _units(text):
        sentences.extend(s.strip() for s in _SENTENCE.split(unit) if len(s.strip()) > 2)
    return sentences


def _keywords(text: str) -> Counter[str]:
    return Counter(w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS and len(w) > 2)


def rank_sentences(text: str, *, focus: str | None = None) -> list[tuple[float, int, str]]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    freq = _keywords(text)
    top = max(freq.values(), default=1)
    focus_words = set(_keywords(focus or ""))
    ranked = []
    for idx, sentence in enumerate(sentences):
        words = [w.lower() for w in _WORD.findall(sentence)]
        content = [w for w in words if w in freq]
        if not content:
            continue
        score = sum(freq[w] / top for w in content) / (len(words) ** 0.5)
        score += 0.6 * len(focus_words & set(content))
        if idx < 3:
            score *= 1.15  # leads tend to carry the thesis
        if len(words) < 5:
            score *= 0.6  # headings and fragments
        ranked.append((score, idx, sentence))
    ranked.sort(key=lambda r: r[0], reverse=True)
    return ranked


def extractive_summary(
    text: str, *, max_sentences: int = 5, focus: str | None = None, mode: str = "key_points"
) -> list[str]:
    """Return the most informative sentences in document order.

    ``mode`` biases selection: ``action_items`` and ``risks`` prefer matching sentences.
    """
    ranked = rank_sentences(text, focus=focus)
    if mode in {"action_items", "risks"}:
        pattern = _ACTION if mode == "action_items" else _RISK
        signals = action_signals if mode == "action_items" else (lambda s: len(pattern.findall(s)))
        ranked = [r for r in ranked if pattern.search(r[2])]
        ranked.sort(key=lambda r: (signals(r[2]), r[0]), reverse=True)
    seen: set[str] = set()
    chosen = []
    for _score, idx, sentence in ranked:
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((idx, sentence))
        if len(chosen) >= max_sentences:
            break
    chosen.sort()
    return [s for _, s in chosen]
