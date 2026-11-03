"""The summarization orchestrator: chunk -> fan out to workers -> reduce -> (edit).

Strategies
----------
``stuff``       one call over the whole text (short inputs).
``map_reduce``  chunks are summarized in parallel by the worker pool, then reduced
                hierarchically (tree reduction) until one summary is left.
``refine``      sequential: a running summary is updated chunk by chunk (keeps narrative flow).
``ensemble``    several profiles (key points, action items, risks, ...) run map-reduce in
                parallel on the same pool, then an *editor* model merges them into one document.
``auto``        picks ``stuff`` / ``map_reduce`` / ``ensemble`` from input size and profiles.

Model routing: map/reduce/refine calls use the (cheap, fast) worker model; the final ensemble
edit uses the main model, where quality matters most.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..agents import prompts
from ..llm.base import LLMProvider, Message, Usage
from .chunking import Chunk, chunk_text
from .profiles import DEFAULT_ENSEMBLE, PROFILES, SummaryProfile, get_profiles
from .workers import PartialSummary, SummaryError, SummaryWorker, SummaryWorkerPool

STRATEGIES = ("auto", "stuff", "map_reduce", "refine", "ensemble")


@dataclass
class SummaryResult:
    summary: str
    strategy: str
    profiles: list[str]
    chunks: int
    levels: int
    sections: dict[str, str] = field(default_factory=dict)
    workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    partials: list[PartialSummary] = field(default_factory=list)
    failed_jobs: int = 0
    usage: Usage = field(default_factory=Usage)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "strategy": self.strategy,
            "profiles": self.profiles,
            "chunks": self.chunks,
            "levels": self.levels,
            "sections": self.sections,
            "workers": self.workers,
            "partials": [p.to_dict() for p in self.partials],
            "failed_jobs": self.failed_jobs,
            "usage": self.usage.to_dict(),
            "duration_ms": round(self.duration_ms, 1),
        }


class SummaryOrchestrator:
    def __init__(
        self,
        llm: LLMProvider,
        tracer: Any = None,
        *,
        editor_llm: LLMProvider | None = None,
        workers: int = 4,
        chunk_chars: int = 6000,
        overlap: int = 200,
        max_tokens: int = 4096,
        effort: str | None = "low",
        editor_max_tokens: int = 8000,
        editor_effort: str | None = None,
    ):
        self.llm = llm
        self.editor_llm = editor_llm or llm
        self.tracer = tracer
        self.n_workers = max(1, workers)
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.max_tokens = max_tokens
        self.effort = effort
        self.editor_max_tokens = editor_max_tokens
        self.editor_effort = editor_effort

    # ------------------------------------------------------------------ public API
    async def summarize(
        self,
        text: str,
        *,
        strategy: str = "auto",
        profiles: Sequence[str] | None = None,
        focus: str | None = None,
        title: str | None = None,
    ) -> SummaryResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; choose one of {STRATEGIES}")
        started = time.perf_counter()
        chosen = get_profiles(profiles)
        chunks = chunk_text(text, max_chars=self.chunk_chars, overlap=self.overlap)
        if not chunks:
            raise SummaryError("Nothing to summarize: the input is empty.")
        if strategy == "auto":
            if len(chosen) > 1:
                strategy = "ensemble"
            else:
                strategy = "stuff" if len(chunks) == 1 else "map_reduce"
        if strategy == "ensemble" and len(chosen) == 1 and profiles is None:
            chosen = [PROFILES[n] for n in DEFAULT_ENSEMBLE]

        workers = [
            SummaryWorker(
                f"worker-{i + 1}", self.llm, self.tracer, max_tokens=self.max_tokens, effort=self.effort
            )
            for i in range(self.n_workers)
        ]
        pool = SummaryWorkerPool(workers, tracer=self.tracer)
        self._log(
            "summary_started",
            strategy=strategy,
            chunks=len(chunks),
            workers=len(workers),
            profiles=[p.name for p in chosen],
            chars=len(text),
        )

        run = _Run(self, pool, chunks, focus)
        if strategy == "stuff":
            sections = await run.stuff(chosen)
        elif strategy == "refine":
            sections = {p.name: await run.refine(p) for p in chosen}
        else:  # map_reduce / ensemble share the same fan-out
            sections = await run.map_reduce(chosen)

        if len(chosen) > 1:
            summary = await run.edit(chosen, sections, title)
        else:
            summary = sections[chosen[0].name]

        result = SummaryResult(
            summary=summary.strip(),
            strategy=strategy,
            profiles=[p.name for p in chosen],
            chunks=len(chunks),
            levels=run.levels,
            sections=sections,
            partials=run.partials,
            failed_jobs=run.failed,
            workers={w.id: w.stats() for w in workers if w.calls},
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        for p in run.partials:
            result.usage = result.usage + p.usage
        self._log(
            "summary_finished",
            strategy=strategy,
            chunks=len(chunks),
            levels=run.levels,
            failed_jobs=run.failed,
            chars=len(result.summary),
            duration_ms=round(result.duration_ms, 1),
        )
        return result

    def _log(self, kind: str, **data: Any) -> None:
        if self.tracer is not None:
            self.tracer.log("summarizer", kind, **data)


class _Run:
    """State of one summarization run (kept off the orchestrator so it stays reentrant)."""

    def __init__(
        self, owner: SummaryOrchestrator, pool: SummaryWorkerPool, chunks: list[Chunk], focus: str | None
    ):
        self.owner = owner
        self.pool = pool
        self.chunks = chunks
        self.focus = focus
        self.partials: list[PartialSummary] = []
        self.failed = 0
        self.levels = 0

    def _collect(self, outcomes: Sequence[Any]) -> list[PartialSummary]:
        ok = [o.value for o in outcomes if o.ok and o.value is not None]
        self.failed += sum(1 for o in outcomes if not o.ok)
        self.partials.extend(ok)
        return ok

    async def stuff(self, profiles: Sequence[SummaryProfile]) -> dict[str, str]:
        whole = Chunk(0, "\n\n".join(c.text for c in self.chunks), 0, 0)
        outcomes = await self.pool.run(
            [lambda w, p=p: w.map(whole, 1, p, self.focus) for p in profiles], label="stuff"
        )
        self._collect(outcomes)
        sections = {}
        for profile, outcome in zip(profiles, outcomes, strict=True):
            if not outcome.ok or outcome.value is None:
                raise SummaryError(f"stuff summary failed for '{profile.name}': {outcome.error}")
            sections[profile.name] = outcome.value.text
        self.levels = 1
        return sections

    async def refine(self, profile: SummaryProfile) -> str:
        worker = self.pool.workers[0]  # refine is inherently sequential
        current = await worker.map(self.chunks[0], len(self.chunks), profile, self.focus)
        self.partials.append(current)
        for chunk in self.chunks[1:]:
            current = await worker.refine(current.text, chunk, profile, self.focus)
            self.partials.append(current)
        self.levels = len(self.chunks)
        return current.text

    async def map_reduce(self, profiles: Sequence[SummaryProfile]) -> dict[str, str]:
        total = len(self.chunks)
        # one flat batch: every (profile, chunk) pair is an independent job on the same pool
        jobs = [(p, c) for p in profiles for c in self.chunks]
        outcomes = await self.pool.run(
            [lambda w, p=p, c=c: w.map(c, total, p, self.focus) for p, c in jobs], label="map"
        )
        by_profile: dict[str, list[tuple[int, str]]] = {p.name: [] for p in profiles}
        for (profile, chunk), outcome in zip(jobs, outcomes, strict=True):
            if outcome.ok and outcome.value is not None:
                by_profile[profile.name].append((chunk.index, outcome.value.text))
        self._collect(outcomes)
        for name, items in by_profile.items():
            if not items:
                raise SummaryError(f"every map job failed for profile '{name}'")
        # tree-reduce all profiles level by level, sharing the pool
        current = {name: [t for _, t in sorted(items)] for name, items in by_profile.items()}
        level = 1
        while any(len(texts) > 1 for texts in current.values()):
            batch: list[tuple[str, list[str]]] = []
            for name, texts in current.items():
                for group in self._groups(texts):
                    batch.append((name, group))
            reduce_jobs = [
                (lambda w, n=name, g=group, lv=level: w.reduce(g, PROFILES[n], self.focus, lv))
                if len(group) > 1
                else None
                for name, group in batch
            ]
            runnable = [j for j in reduce_jobs if j is not None]
            outcomes = iter(self._wrap(await self.pool.run(runnable, label=f"reduce-L{level}")))
            nxt: dict[str, list[str]] = {name: [] for name in current}
            for (name, group), job in zip(batch, reduce_jobs, strict=True):
                if job is None:
                    nxt[name].append(group[0])  # odd one out passes through to the next level
                    continue
                text = next(outcomes)
                nxt[name].append(text if text is not None else "\n".join(group))  # degrade, don't fail
            current = nxt
            level += 1
        self.levels = level
        return {name: texts[0] for name, texts in current.items()}

    def _wrap(self, outcomes: Sequence[Any]) -> list[str | None]:
        self._collect(outcomes)
        return [o.value.text if o.ok and o.value is not None else None for o in outcomes]

    def _groups(self, texts: list[str]) -> list[list[str]]:
        """Group consecutive summaries so each reduce input stays under the chunk budget."""
        budget = self.owner.chunk_chars
        groups: list[list[str]] = []
        current: list[str] = []
        size = 0
        for text in texts:
            if current and (size + len(text) > budget) and len(current) >= 2:
                groups.append(current)
                current, size = [], 0
            current.append(text)
            size += len(text)
        if current:
            if len(current) == 1 and groups:
                groups[-1].append(current[0])  # never leave a lone straggler when it can merge
            else:
                groups.append(current)
        return groups

    async def edit(
        self, profiles: Sequence[SummaryProfile], sections: dict[str, str], title: str | None
    ) -> str:
        owner = self.owner
        payload = {p.name: (p.title, sections[p.name]) for p in profiles}
        user = prompts.editor_prompt(payload, focus=self.focus, title=title)
        started = time.perf_counter()
        response = await owner.editor_llm.complete(
            system=prompts.SUMMARY_EDITOR,
            messages=[Message.user(user)],
            max_tokens=owner.editor_max_tokens,
            effort=owner.editor_effort,
            cache="prefix",
        )
        duration = (time.perf_counter() - started) * 1000
        if owner.tracer is not None:
            owner.tracer.log(
                "editor",
                "llm_call",
                phase="edit",
                model=response.model or owner.editor_llm.model,
                provider=owner.editor_llm.name,
                stop_reason=response.stop_reason,
                usage=response.usage.to_dict(),
                duration_ms=round(duration, 1),
                text_preview=response.text[:200],
            )
        self.partials.append(
            PartialSummary(
                "editor",
                "edit",
                "+".join(payload),
                response.text,
                None,
                0,
                response.usage,
                duration,
                response.model,
            )
        )
        if not response.text:
            # fall back to a mechanical merge rather than losing the workers' output
            return "\n\n".join(f"## {heading}\n{text}" for heading, text in payload.values())
        return response.text
