"""Job workers: poll the hub store, claim a job, execute it through the Hub, record the result.

Run them inside the API process (``SP_INLINE_WORKERS``) or as separate processes /
containers (``shiny-pancake worker --concurrency N``) that share the SQLite volume.
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Any

from .service import Hub
from .store import Job


class JobWorker:
    def __init__(
        self,
        hub: Hub,
        *,
        worker_id: str | None = None,
        poll_interval: float = 0.5,
        kinds: tuple[str, ...] | None = None,
    ):
        self.hub = hub
        self.id = worker_id or f"{socket.gethostname()[:16]}-{os.getpid()}"
        self.poll_interval = poll_interval
        self.kinds = kinds
        self.processed = 0

    async def run_once(self) -> Job | None:
        job = await asyncio.to_thread(self.hub.store.claim_next, self.id, self.kinds)
        if job is None:
            return None
        tracer = self.hub.new_tracer(job.id)
        tracer.log("hub", "job_claimed", worker=self.id, kind=job.kind, attempt=job.attempts)
        try:
            result = await self.hub.execute(job.kind, job.payload, tracer)
            self.hub.store.complete(job.id, result.to_dict())
        except asyncio.CancelledError:
            self.hub.store.fail(job.id, "cancelled: worker shut down")
            raise
        except Exception as e:  # noqa: BLE001 - a failing job must not kill the worker
            tracer.log("hub", "job_error", worker=self.id, error=f"{type(e).__name__}: {e}")
            self.hub.store.fail(job.id, f"{type(e).__name__}: {e}")
        self.processed += 1
        return self.hub.store.get_job(job.id)

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            job = await self.run_once()
            if job is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.poll_interval)
                except TimeoutError:
                    pass


class JobWorkerPool:
    def __init__(
        self,
        hub: Hub,
        *,
        concurrency: int = 2,
        poll_interval: float = 0.5,
        name: str | None = None,
        requeue_after_s: float = 3600,
    ):
        self.hub = hub
        base = name or f"{socket.gethostname()[:16]}-{os.getpid()}"
        self.workers = [
            JobWorker(hub, worker_id=f"{base}-w{i + 1}", poll_interval=poll_interval)
            for i in range(concurrency)
        ]
        self.requeue_after_s = requeue_after_s
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []

    async def start(self) -> int:
        requeued = self.hub.store.requeue_stale(self.requeue_after_s)
        self._stop.clear()
        self._tasks = [asyncio.create_task(w.run_forever(self._stop), name=w.id) for w in self.workers]
        return requeued

    async def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if not self._tasks:
            return
        done, pending = await asyncio.wait(self._tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._tasks = []

    async def run_until(self, stop: asyncio.Event) -> None:
        await self.start()
        try:
            await stop.wait()
        finally:
            await self.stop()
