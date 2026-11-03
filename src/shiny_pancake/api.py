"""HTTP API + web UI (FastAPI). Run with ``shiny-pancake serve`` or ``uvicorn shiny_pancake.api:app``.

Synchronous endpoints (``/run``, ``/summarize``) are handy for scripts; the job endpoints
(``/jobs``) are the hub's real interface: enqueue, then follow the run live over Server-Sent
Events while job workers (in-process or separate containers) execute it.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .hub.service import Hub, validate_payload
from .hub.worker import JobWorkerPool

WEB_DIR = Path(__file__).parent / "web"


class RunRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)


class RunResponse(BaseModel):
    run_id: str
    status: str
    answer: str
    summary: dict[str, Any]


class SummarizeRequest(BaseModel):
    text: str | None = Field(default=None, max_length=400_000)
    path: str | None = Field(default=None, description="File inside the hub workspace")
    strategy: Literal["auto", "stuff", "map_reduce", "refine", "ensemble"] = "auto"
    profiles: list[str] | None = None
    focus: str | None = Field(default=None, max_length=500)
    title: str | None = Field(default=None, max_length=200)


class JobRequest(BaseModel):
    kind: Literal["ask", "summarize", "eval"]
    payload: dict[str, Any]


def create_app(
    settings: Settings | None = None, *, hub: Hub | None = None, inline_workers: int | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        the_hub = hub or Hub(settings or Settings.from_env())
        owns_hub = hub is None
        await the_hub.start()
        n_workers = the_hub.settings.inline_workers if inline_workers is None else inline_workers
        pool = JobWorkerPool(the_hub, concurrency=n_workers, name="inline") if n_workers > 0 else None
        if pool:
            await pool.start()
        app.state.hub = the_hub
        app.state.pool = pool
        try:
            yield
        finally:
            if pool:
                await pool.stop()
            if owns_hub:
                await the_hub.close()

    app = FastAPI(
        title="shiny-pancake hub",
        version=__version__,
        lifespan=lifespan,
        description="Multi-agent LLM hub: agent team, multi-worker summarization, job queue, evals.",
    )

    def get_hub(request: Request) -> Hub:
        return request.app.state.hub

    def require_key(request: Request) -> None:
        expected = request.app.state.hub.settings.api_key
        if not expected:
            return
        auth = request.headers.get("authorization", "")
        supplied = (
            request.headers.get("x-api-key")
            or (auth[7:] if auth.lower().startswith("bearer ") else None)
            or request.query_params.get("key")
        )  # EventSource cannot send headers
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    guarded = [Depends(require_key)]

    # ------------------------------------------------------------------ UI & health
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> Response:
        return Response((WEB_DIR / "pancake.svg").read_text(encoding="utf-8"), media_type="image/svg+xml")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        h = get_hub(request)
        pool = request.app.state.pool
        return {
            "status": "ok",
            "version": __version__,
            "provider": h.llm.name,
            "model": h.llm.model,
            "inline_workers": len(pool.workers) if pool else 0,
            "jobs": h.store.stats(),
        }

    @app.get("/api/info", dependencies=guarded)
    async def info(request: Request) -> dict[str, Any]:
        return get_hub(request).info()

    # ------------------------------------------------------------------ synchronous runs
    @app.post("/run", response_model=RunResponse, dependencies=guarded)
    async def run(req: RunRequest, request: Request) -> RunResponse:
        result = await get_hub(request).ask(req.goal)
        return RunResponse(
            run_id=result.run_id, status=result.status, answer=result.output, summary=result.trace
        )

    @app.post("/summarize", dependencies=guarded)
    async def summarize(req: SummarizeRequest, request: Request) -> dict[str, Any]:
        try:
            result = await get_hub(request).summarize(
                req.text,
                path=req.path,
                strategy=req.strategy,
                profiles=req.profiles,
                focus=req.focus,
                title=req.title,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return result.to_dict()

    # ------------------------------------------------------------------ job queue
    @app.post("/jobs", status_code=202, dependencies=guarded)
    async def submit(req: JobRequest, request: Request) -> dict[str, Any]:
        try:
            validate_payload(req.kind, req.payload)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        job = get_hub(request).submit(req.kind, req.payload)
        return {"job": job.to_dict(), "stream": f"/jobs/{job.id}/stream", "events": f"/jobs/{job.id}/events"}

    @app.get("/jobs", dependencies=guarded)
    async def list_jobs(
        request: Request,
        limit: int = Query(30, ge=1, le=200),
        status: str | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        h = get_hub(request)
        jobs = h.store.list_jobs(limit=limit, status=status, kind=kind)
        return {"jobs": [j.to_dict(include_result=False) for j in jobs], "stats": h.store.stats()}

    @app.get("/jobs/{job_id}", dependencies=guarded)
    async def get_job(job_id: str, request: Request) -> dict[str, Any]:
        job = get_hub(request).store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return job.to_dict()

    @app.get("/jobs/{job_id}/events", dependencies=guarded)
    async def job_events(job_id: str, request: Request, after: int = 0) -> list[dict[str, Any]]:
        h = get_hub(request)
        if h.store.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return h.store.events(job_id, after_id=after)

    @app.get("/runs/{run_id}/trace", dependencies=guarded)
    async def trace(run_id: str, request: Request) -> list[dict[str, Any]]:
        return await job_events(run_id, request)

    @app.get("/jobs/{job_id}/stream", dependencies=guarded)
    async def stream(job_id: str, request: Request, after: int = 0) -> StreamingResponse:
        h = get_hub(request)
        if h.store.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Unknown job id")

        def sse(event: dict[str, Any]) -> str:
            return f"id: {event['id']}\nevent: trace\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

        async def events() -> AsyncIterator[str]:
            last, last_beat = after, time.monotonic()
            while not await request.is_disconnected():
                batch = h.store.events(job_id, after_id=last)
                for event in batch:
                    last = event["id"]
                    yield sse(event)
                job = h.store.get_job(job_id)
                if job is not None and job.done:
                    for event in h.store.events(job_id, after_id=last):  # drain the tail
                        last = event["id"]
                        yield sse(event)
                    final = json.dumps(job.to_dict(), ensure_ascii=False, default=str)
                    yield f"event: done\ndata: {final}\n\n"
                    return
                if time.monotonic() - last_beat > 15:
                    last_beat = time.monotonic()
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
