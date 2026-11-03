import asyncio

import pytest

from shiny_pancake.llm.base import LLMError
from shiny_pancake.llm.mock_provider import OfflineDemoProvider
from shiny_pancake.summarize import SummaryError, SummaryOrchestrator, chunk_text
from shiny_pancake.summarize.extractive import extractive_summary, split_sentences
from shiny_pancake.tracing import Tracer

from .conftest import LONG_DOC


def test_chunking_respects_budget_and_covers_text():
    chunks = chunk_text(LONG_DOC, max_chars=700, overlap=120)
    assert len(chunks) > 3
    assert all(c.chars <= 700 + 120 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for i in range(1, 13):  # every section survives chunking
        assert any(f"Section {i}." in c.text for c in chunks)
    assert chunks[1].text[0].isupper()  # overlap starts on a sentence boundary, not mid-word
    assert chunk_text("short text", max_chars=700) == chunk_text("short text", max_chars=700)
    assert len(chunk_text("short text", max_chars=700)) == 1
    assert chunk_text("   ") == []


def test_chunking_hard_splits_giant_sentences():
    blob = "word " * 2000
    chunks = chunk_text(blob, max_chars=600, overlap=0)
    assert all(c.chars <= 600 for c in chunks) and len(chunks) >= 16


def test_extractive_rejoins_wrapped_lines_and_finds_actions():
    text = "This sentence is hard\nwrapped across lines.\n\n- Anna will send the report by Friday.\n# Heading"
    assert split_sentences(text) == [
        "This sentence is hard wrapped across lines.",
        "Anna will send the report by Friday.",
    ]
    assert extractive_summary(text, mode="action_items") == ["Anna will send the report by Friday."]


class FlakyOnce(OfflineDemoProvider):
    """Fails the first call for chunk 2, then behaves."""

    def __init__(self):
        super().__init__()
        self.failed = False

    async def complete(self, **kw):
        if not self.failed and 'part="2"' in kw["messages"][-1].text:
            self.failed = True
            raise LLMError("overloaded", retryable=True, status_code=529)
        return await super().complete(**kw)


class SlowOffline(OfflineDemoProvider):
    def __init__(self):
        super().__init__()
        self.in_flight = self.peak = 0

    async def complete(self, **kw):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        try:
            return await super().complete(**kw)
        finally:
            self.in_flight -= 1


async def test_map_reduce_fans_out_and_tree_reduces():
    llm = SlowOffline()
    tracer = Tracer()
    result = await SummaryOrchestrator(llm, tracer, workers=4, chunk_chars=600).summarize(
        LONG_DOC, strategy="map_reduce"
    )
    assert result.strategy == "map_reduce" and result.chunks > 4
    assert result.levels >= 3  # map level + at least two reduce levels
    assert len(result.workers) == 4 and llm.peak == 4  # really parallel, bounded by the pool size
    phases = [p.phase for p in result.partials]
    assert phases.count("map") == result.chunks and "reduce" in phases
    assert result.summary.startswith("-")
    assert any(e.kind == "pool_batch" for e in tracer.events)


async def test_ensemble_uses_editor_and_sections():
    tracer = Tracer()
    result = await SummaryOrchestrator(OfflineDemoProvider(), tracer, workers=3, chunk_chars=900).summarize(
        LONG_DOC, strategy="ensemble", title="Release plan"
    )
    assert result.profiles == ["key_points", "action_items", "risks"]
    assert set(result.sections) == set(result.profiles)
    assert result.summary.startswith("# Release plan")
    assert "## Action items" in result.summary and "- [ ]" in result.summary
    assert result.partials[-1].phase == "edit"
    assert "editor" in {e.agent for e in tracer.events if e.kind == "llm_call"}


async def test_auto_strategy_and_refine():
    orch = SummaryOrchestrator(OfflineDemoProvider(), workers=2, chunk_chars=600)
    assert (await orch.summarize("Short note. Bence will fix it by Friday.")).strategy == "stuff"
    assert (await orch.summarize(LONG_DOC)).strategy == "map_reduce"
    assert (await orch.summarize(LONG_DOC, profiles=["tldr", "risks"])).strategy == "ensemble"
    refined = await orch.summarize(LONG_DOC, strategy="refine")
    assert refined.levels == refined.chunks
    assert [p.phase for p in refined.partials][1:] == ["refine"] * (refined.chunks - 1)


async def test_pool_retries_a_failed_job_on_another_attempt():
    llm = FlakyOnce()
    tracer = Tracer()
    result = await SummaryOrchestrator(llm, tracer, workers=2, chunk_chars=600).summarize(
        LONG_DOC, strategy="map_reduce"
    )
    assert llm.failed and result.failed_jobs == 0
    assert sum(1 for p in result.partials if p.phase == "map") == result.chunks


async def test_total_failure_raises_and_bad_input_is_rejected():
    class Dead(OfflineDemoProvider):
        async def complete(self, **kw):
            raise LLMError("auth", retryable=False, status_code=401)

    with pytest.raises(SummaryError):
        await SummaryOrchestrator(Dead(), workers=2, chunk_chars=600).summarize(
            LONG_DOC, strategy="map_reduce"
        )
    with pytest.raises(SummaryError):
        await SummaryOrchestrator(OfflineDemoProvider()).summarize("   ")
    with pytest.raises(ValueError):
        await SummaryOrchestrator(OfflineDemoProvider()).summarize("x", strategy="magic")
    with pytest.raises(ValueError):
        await SummaryOrchestrator(OfflineDemoProvider()).summarize("x", profiles=["poetry"])
