"""Structured tracing: every LLM call, tool call and pipeline step becomes a TraceEvent.

Events fan out to pluggable sinks (JSONL file, the hub's SQLite store, callbacks for live UIs).
``Tracer.summary()`` aggregates tokens, estimated cost, latency and per-agent / per-worker stats.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .guardrails import redact_secrets
from .llm.base import Usage
from .llm.pricing import estimate_cost


@dataclass
class TraceEvent:
    run_id: str
    seq: int
    ts: float
    agent: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class JsonlSink:
    """Appends one JSON object per line to ``<trace_dir>/<run_id>.jsonl``."""

    def __init__(self, trace_dir: Path, run_id: str):
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / f"{run_id}.jsonl"
        self._lock = threading.Lock()

    def emit(self, event: TraceEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class CallbackSink:
    def __init__(self, fn: Callable[[TraceEvent], None]):
        self.fn = fn

    def emit(self, event: TraceEvent) -> None:
        self.fn(event)


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub(v) for v in value]
    return value


class Tracer:
    def __init__(
        self, run_id: str | None = None, *, trace_dir: Path | None = None, sinks: Iterable[TraceSink] = ()
    ):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.started = time.time()
        self.events: list[TraceEvent] = []
        self._sinks: list[TraceSink] = list(sinks)
        if trace_dir is not None:
            self._sinks.append(JsonlSink(Path(trace_dir), self.run_id))
        self._lock = threading.Lock()
        self._seq = 0

    def add_sink(self, sink: TraceSink) -> None:
        self._sinks.append(sink)

    def log(self, agent: str, kind: str, /, **data: Any) -> TraceEvent:
        with self._lock:
            self._seq += 1
            event = TraceEvent(self.run_id, self._seq, time.time(), agent, kind, _scrub(data))
            self.events.append(event)
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:  # noqa: BLE001 - observability must never break a run
                pass
        return event

    # ------------------------------------------------------------------ aggregation
    def summary(self) -> dict[str, Any]:
        total = Usage()
        cost = 0.0
        llm_calls = tool_calls = errors = 0
        agents: dict[str, dict[str, Any]] = {}
        models: dict[str, int] = {}
        llm_ms = 0.0

        for e in self.events:
            stats = agents.setdefault(
                e.agent, {"llm_calls": 0, "tool_calls": 0, "tokens": 0, "cost_usd": 0.0}
            )
            if e.kind == "llm_call":
                llm_calls += 1
                stats["llm_calls"] += 1
                usage = Usage(**e.data.get("usage", {}))
                total = total + usage
                model = e.data.get("model", "")
                models[model] = models.get(model, 0) + 1
                call_cost = estimate_cost(model, usage)
                cost += call_cost
                stats["tokens"] += usage.total
                stats["cost_usd"] = round(stats["cost_usd"] + call_cost, 6)
                llm_ms += float(e.data.get("duration_ms", 0.0))
            elif e.kind == "tool_call":
                tool_calls += 1
                stats["tool_calls"] += 1
                if not e.data.get("ok", True):
                    errors += 1
            elif e.kind.endswith("error"):
                errors += 1

        last_ts = self.events[-1].ts if self.events else self.started
        return {
            "run_id": self.run_id,
            "duration_s": round(max(0.0, last_ts - self.started), 3),
            "events": len(self.events),
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "errors": errors,
            "llm_time_ms": round(llm_ms, 1),
            "tokens": total.to_dict() | {"total": total.total},
            "cost_usd": round(cost, 6),
            "models": models,
            "agents": {k: v for k, v in agents.items() if v["llm_calls"] or v["tool_calls"]},
        }

    def render(self, *, width: int = 100) -> str:
        """Human-readable timeline for the CLI."""
        lines = []
        for e in self.events:
            t = e.ts - self.started
            detail = _describe(e)
            line = f"{t:7.2f}s  {e.agent:<14} {e.kind:<16} {detail}"
            lines.append(line if len(line) <= width else line[: width - 1] + "…")
        return "\n".join(lines)


def _describe(e: TraceEvent) -> str:
    d = e.data
    if e.kind == "llm_call":
        u = d.get("usage", {})
        tools = d.get("tool_calls") or []
        extra = f" -> {', '.join(tools)}" if tools else ""
        return (
            f"{d.get('model', '')} {d.get('stop_reason', '')} in={u.get('input_tokens', 0)} "
            f"out={u.get('output_tokens', 0)} {d.get('duration_ms', 0):.0f}ms{extra}"
        )
    if e.kind == "tool_call":
        status = "ok" if d.get("ok") else "ERR"
        return f"{d.get('tool')} [{status}] {d.get('duration_ms', 0):.0f}ms {d.get('preview', '')!r}"
    keys = ("goal", "task", "strategy", "phase", "chunk", "error", "status", "message")
    parts = [f"{k}={str(d[k])[:60]!r}" for k in keys if k in d]
    return " ".join(parts)
