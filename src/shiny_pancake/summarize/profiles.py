"""Summary profiles: the "angle" a worker summarizes from. Ensemble runs several in parallel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryProfile:
    name: str
    title: str
    instruction: str


PROFILES: dict[str, SummaryProfile] = {
    p.name: p
    for p in (
        SummaryProfile(
            "key_points",
            "Key points",
            "List the most important facts, claims and numbers as concise Markdown bullet points.",
        ),
        SummaryProfile(
            "executive",
            "Executive summary",
            "Write 3-5 sentences for a busy decision maker: what it is, why it matters, what to do.",
        ),
        SummaryProfile(
            "action_items",
            "Action items",
            "Extract concrete action items and decisions as a Markdown checklist ('- [ ] ...'), "
            "with owners and deadlines when the text states them. If there are none, say so.",
        ),
        SummaryProfile(
            "risks",
            "Risks & open questions",
            "List risks, caveats, open questions and hidden assumptions as bullet points.",
        ),
        SummaryProfile("tldr", "TL;DR", "Capture the essence in one or two sentences."),
    )
}

DEFAULT_ENSEMBLE = ("key_points", "action_items", "risks")


def get_profiles(names: list[str] | tuple[str, ...] | None) -> list[SummaryProfile]:
    names = list(names or ["key_points"])
    unknown = [n for n in names if n not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown summary profile(s) {unknown}; available: {sorted(PROFILES)}")
    return [PROFILES[n] for n in dict.fromkeys(names)]
