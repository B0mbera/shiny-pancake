"""Summary workers and the pool that schedules jobs onto them.

A :class:`SummaryWorker` is a named, stateless unit that turns text into a summary with one LLM
call. The :class:`SummaryWorkerPool` runs N workers concurrently over an ``asyncio.Queue``:
jobs are pulled by whichever worker is free (natural load balancing), results come back in
submission order, and failures are captured per job instead of failing the whole batch.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..agents import prompts
from ..llm.base import LLMError, LLMProvider, Message, Usage
from .chunking import Chunk
from .profiles import SummaryProfile

T = TypeVar("T")


@dataclass
class PartialSummary:
    worker: str
    phase: str  # map | reduce | refine | edit
    profile: str
    text: str
    chunk_index: int | None = None
    level: int = 0
    usage: Usage = field(default_factory=Usage)
    duration_ms: float = 0.0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker": self.worker,
            "phase": self.phase,
            "profile": self.profile,
            "chunk": self.chunk_index,
            "level": self.level,
            "chars": len(self.text),
            "tokens": self.usage.total,
            "duration_ms": round(self.duration_ms, 1),
            "model": self.model,
        }


class SummaryError(RuntimeError):
    pass


class SummaryWorker:
    def __init__(
        self,
        worker_id: str,
        llm: LLMProvider,
        tracer: Any,
        *,
        max_tokens: int = 4096,
        effort: str | None = None,
        system: str = prompts.SUMMARY_WORKER,
    ):
        self.id = worker_id
        self.llm = llm
        self.tracer = tracer
        self.max_tokens = max_tokens
        self.effort = effort
        self.system = system
        self.calls = 0
        self.usage = Usage()
        self.busy_ms = 0.0

    async def _call(
        self, user: str, *, phase: str, profile: str, chunk_index: int | None = None, level: int = 0
    ) -> PartialSummary:
        started = time.perf_counter()
        response = await self.llm.complete(
            system=self.system,
            messages=[Message.user(user)],
            max_tokens=self.max_tokens,
            effort=self.effort,
            cache="prefix",
        )
        duration = (time.perf_counter() - started) * 1000
        self.calls += 1
        self.usage = self.usage + response.usage
        self.busy_ms += duration
        model = response.model or self.llm.model
        if self.tracer is not None:
            self.tracer.log(
                self.id,
                "llm_call",
                phase=phase,
                profile=profile,
                chunk=chunk_index,
                level=level,
                model=model,
                provider=self.llm.name,
                stop_reason=response.stop_reason,
                usage=response.usage.to_dict(),
                duration_ms=round(duration, 1),
                text_preview=response.text[:200],
            )
        if response.stop_reason == "refusal" or not response.text:
            raise SummaryError(f"{self.id}: empty or refused {phase} response ({response.stop_reason})")
        return PartialSummary(
            self.id, phase, profile, response.text, chunk_index, level, response.usage, duration, model
        )

    async def map(
        self, chunk: Chunk, total: int, profile: SummaryProfile, focus: str | None
    ) -> PartialSummary:
        user = prompts.map_prompt(
            chunk.text,
            index=chunk.index,
            total=total,
            profile=profile.name,
            instruction=profile.instruction,
            focus=focus,
        )
        return await self._call(user, phase="map", profile=profile.name, chunk_index=chunk.index)

    async def reduce(
        self, texts: Sequence[str], profile: SummaryProfile, focus: str | None, level: int
    ) -> PartialSummary:
        user = prompts.reduce_prompt(
            list(texts), profile=profile.name, instruction=profile.instruction, focus=focus, level=level
        )
        return await self._call(user, phase="reduce", profile=profile.name, level=level)

    async def refine(
        self, current: str, chunk: Chunk, profile: SummaryProfile, focus: str | None
    ) -> PartialSummary:
        user = prompts.refine_prompt(
            current, chunk.text, profile=profile.name, instruction=profile.instruction, focus=focus
        )
        return await self._call(user, phase="refine", profile=profile.name, chunk_index=chunk.index)

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "tokens": self.usage.total,
            "busy_ms": round(self.busy_ms, 1),
            "model": self.llm.model,
        }


@dataclass
class JobOutcome(Generic[T]):
    index: int
    value: T | None = None
    error: BaseException | None = None
    worker: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SummaryWorkerPool:
    """N workers draining a shared queue. ``run`` preserves job order in its result list."""

    def __init__(self, workers: Sequence[SummaryWorker], *, tracer: Any = None, max_attempts: int = 2):
        if not workers:
            raise ValueError("pool needs at least one worker")
        self.workers = list(workers)
        self.tracer = tracer
        self.max_attempts = max_attempts

    async def run(
        self, jobs: Sequence[Callable[[SummaryWorker], Awaitable[T]]], *, label: str = "batch"
    ) -> list[JobOutcome[T]]:
        queue: asyncio.Queue[tuple[int, Callable[[SummaryWorker], Awaitable[T]], int]] = asyncio.Queue()
        for i, job in enumerate(jobs):
            queue.put_nowait((i, job, 1))
        outcomes: list[JobOutcome[T]] = [JobOutcome(i) for i in range(len(jobs))]
        if self.tracer is not None:
            self.tracer.log(
                "pool", "pool_batch", phase=label, jobs=len(jobs), workers=min(len(self.workers), len(jobs))
            )

        async def consume(worker: SummaryWorker) -> None:
            while True:
                await asyncio.sleep(0)  # yield so every worker gets a turn, even with instant providers
                try:
                    index, job, attempt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    outcomes[index] = JobOutcome(index, await job(worker), worker=worker.id)
                except (LLMError, SummaryError) as e:
                    retryable = isinstance(e, SummaryError) or e.retryable
                    if retryable and attempt < self.max_attempts:
                        queue.put_nowait((index, job, attempt + 1))  # another worker may pick it up
                    else:
                        outcomes[index] = JobOutcome(index, error=e, worker=worker.id)
                        if self.tracer is not None:
                            self.tracer.log(worker.id, "worker_error", phase=label, job=index, error=str(e))
                finally:
                    queue.task_done()

        await asyncio.gather(*(consume(w) for w in self.workers[: max(1, len(jobs))]))
        return outcomes
