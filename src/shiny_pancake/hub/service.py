"""The Hub: composition root and the one façade used by the CLI, the HTTP API, job workers,
the MCP server and library users (the "harness" entry point).

    from shiny_pancake import Hub, Settings

    async with Hub(Settings.from_env(provider="offline")) as hub:
        result = await hub.ask("What is 12 * (3 + 4)? Check it.")
        print(result.output, result.trace["cost_usd"])

Everything concrete (providers, stores, tools, MCP bridges) is wired here and nowhere else;
agents and pipelines only see interfaces.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..agents import SPECIALISTS, Orchestrator, SpecialistSpec
from ..agents.prompts import PROMPT_VERSION
from ..config import Settings
from ..llm import LLMProvider, create_provider
from ..memory import MemoryStore
from ..summarize import PROFILES, STRATEGIES, SummaryOrchestrator
from ..tools import Tool, ToolRegistry, allow_all, build_builtin_tools, deny_side_effects
from ..tools.registry import ApprovalPolicy
from ..tracing import Tracer
from .store import HubStore, Job, StoreSink

JOB_KINDS = ("ask", "summarize", "eval")


@dataclass
class HubResult:
    run_id: str
    kind: str
    status: str
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "output": self.output,
            "data": self.data,
            "trace": self.trace,
        }


def validate_payload(kind: str, payload: Mapping[str, Any]) -> None:
    if kind not in JOB_KINDS:
        raise ValueError(f"Unknown job kind {kind!r}; expected one of {JOB_KINDS}")
    if kind == "ask" and not str(payload.get("goal", "")).strip():
        raise ValueError("'ask' jobs need a non-empty 'goal'")
    if kind == "summarize":
        if not (payload.get("text") or payload.get("path")):
            raise ValueError("'summarize' jobs need 'text' or 'path'")
        if payload.get("strategy", "auto") not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        unknown = [p for p in payload.get("profiles") or [] if p not in PROFILES]
        if unknown:
            raise ValueError(f"unknown profiles {unknown}; available: {sorted(PROFILES)}")
    if kind == "eval" and not (payload.get("cases") or payload.get("suite")):
        raise ValueError("'eval' jobs need 'cases' or a 'suite' path")


class Hub:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMProvider | None = None,
        worker_llm: LLMProvider | None = None,
        store: HubStore | None = None,
        memory: MemoryStore | None = None,
        specialists: Mapping[str, SpecialistSpec] | None = None,
        extra_tools: Iterable[Tool] = (),
        mcp_targets: Sequence[Any] = (),
        approval: ApprovalPolicy | None = None,
    ):
        self.settings = settings or Settings.from_env()
        s = self.settings
        self.llm = llm or create_provider(s, role="main")
        self.worker_llm = worker_llm or (llm if llm is not None else create_provider(s, role="worker"))
        self._owns_store = store is None
        self.store = store or HubStore(s.db_path)
        self.memory = memory or MemoryStore(s.memory_path)
        self.specialists = dict(specialists) if specialists is not None else dict(SPECIALISTS)
        self.extra_tools = list(extra_tools)
        self.mcp_targets = list(mcp_targets)
        self.approval = approval or (deny_side_effects if s.approval == "deny" else allow_all)
        self.mcp_servers: list[dict[str, Any]] = []
        self._mcp_tools: list[Tool] = []
        self._stack: AsyncExitStack | None = None
        self._start_lock: asyncio.Lock | None = None

    @classmethod
    def from_env(cls, **overrides: Any) -> Hub:
        return cls(Settings.from_env(**overrides))

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> Hub:
        """Connect MCP servers (idempotent). Called lazily by every run."""
        if self._stack is not None:
            return self
        self._start_lock = self._start_lock or asyncio.Lock()
        async with self._start_lock:
            if self._stack is not None:
                return self
            stack = AsyncExitStack()
            if self.mcp_targets:
                from ..tools.mcp_bridge import MCPBridge

                for i, target in enumerate(self.mcp_targets):
                    bridge = await stack.enter_async_context(MCPBridge(target, prefix=f"mcp{i}_"))
                    scratch = ToolRegistry()
                    names = await bridge.register_into(scratch)
                    self._mcp_tools.extend(scratch)
                    label = target if isinstance(target, str) else type(target).__name__
                    self.mcp_servers.append({"target": label, "server": bridge.server_name, "tools": names})
            self._stack = stack
        return self

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        await self.llm.aclose()
        if self.worker_llm is not self.llm:
            await self.worker_llm.aclose()
        if self._owns_store:
            self.store.close()

    async def __aenter__(self) -> Hub:
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------ building blocks
    def new_tracer(self, run_id: str | None = None, *, persist: bool = True) -> Tracer:
        sinks = [StoreSink(self.store)] if persist else []
        trace_dir = self.settings.trace_dir if persist else None
        return Tracer(run_id, trace_dir=trace_dir, sinks=sinks)

    def summarizer(self, tracer: Tracer | None) -> SummaryOrchestrator:
        s = self.settings
        return SummaryOrchestrator(
            self.worker_llm,
            tracer,
            editor_llm=self.llm,
            workers=s.summary_workers,
            chunk_chars=s.chunk_chars,
            max_tokens=s.worker_max_tokens,
            effort=s.worker_effort,
            editor_effort=s.effort,
        )

    def tools_for(self, tracer: Tracer | None) -> ToolRegistry:
        registry = build_builtin_tools(
            self.settings.workspace,
            self.memory,
            summarizer=self.summarizer(tracer).summarize,
            approval=self.approval,
        )
        for tool in [*self._mcp_tools, *self.extra_tools]:
            registry.register(tool, replace=True)
        return registry

    def orchestrator(self, tracer: Tracer | None) -> Orchestrator:
        s = self.settings
        return Orchestrator(
            self.llm,
            self.tools_for(tracer),
            tracer,
            max_steps=s.max_steps,
            max_tokens=s.max_tokens,
            effort=s.effort,
            specialists=self.specialists,
        )

    # ------------------------------------------------------------------ high-level API
    async def ask(self, goal: str, *, run_id: str | None = None) -> HubResult:
        return await self.run("ask", {"goal": goal}, run_id=run_id)

    async def summarize(
        self,
        text: str | None = None,
        *,
        path: str | None = None,
        strategy: str = "auto",
        profiles: Sequence[str] | None = None,
        focus: str | None = None,
        title: str | None = None,
        run_id: str | None = None,
    ) -> HubResult:
        payload = {
            "text": text,
            "path": path,
            "strategy": strategy,
            "profiles": list(profiles or []) or None,
            "focus": focus,
            "title": title,
        }
        return await self.run("summarize", {k: v for k, v in payload.items() if v is not None}, run_id=run_id)

    async def run(self, kind: str, payload: dict[str, Any], *, run_id: str | None = None) -> HubResult:
        """Run now (in this process) and record it as a job so it shows up in the hub."""
        validate_payload(kind, payload)
        job = self.store.create_job(
            kind, _stored_payload(payload), job_id=run_id, status="running", worker="inline"
        )
        tracer = self.new_tracer(job.id)
        try:
            result = await self.execute(kind, payload, tracer)
        except Exception as e:
            tracer.log("hub", "run_error", error=f"{type(e).__name__}: {e}")
            self.store.fail(job.id, f"{type(e).__name__}: {e}")
            raise
        self.store.complete(job.id, result.to_dict())
        return result

    def submit(self, kind: str, payload: dict[str, Any]) -> Job:
        """Enqueue for the job workers (this process or separate `shiny-pancake worker` processes)."""
        validate_payload(kind, payload)
        return self.store.create_job(kind, payload)

    async def execute(self, kind: str, payload: Mapping[str, Any], tracer: Tracer) -> HubResult:
        """Dispatch one unit of work. Used by :meth:`run`, job workers and the eval harness."""
        await self.start()
        tracer.log(
            "hub",
            "run_started",
            kind=kind,
            provider=self.llm.name,
            model=self.llm.model,
            worker_model=self.worker_llm.model,
            prompt_version=PROMPT_VERSION,
            **({"goal": payload["goal"]} if kind == "ask" else {}),
        )
        if kind == "ask":
            agent_result = await self.orchestrator(tracer).run(str(payload["goal"]))
            result = HubResult(
                tracer.run_id, kind, agent_result.status, agent_result.output, agent_result.to_dict()
            )
        elif kind == "summarize":
            text = payload.get("text") or self._read_workspace(str(payload["path"]))
            summary = await self.summarizer(tracer).summarize(
                text,
                strategy=payload.get("strategy", "auto"),
                profiles=payload.get("profiles"),
                focus=payload.get("focus"),
                title=payload.get("title") or payload.get("path"),
            )
            result = HubResult(tracer.run_id, kind, "completed", summary.summary, summary.to_dict())
        elif kind == "eval":
            from ..evals.harness import EvalHarness, load_suite

            cases = payload.get("cases") or load_suite(Path(str(payload["suite"])))
            report = await EvalHarness(self, concurrency=int(payload.get("concurrency", 4))).run(
                cases, tracer=tracer
            )
            status = "completed" if report.passed == report.total else "failed_checks"
            result = HubResult(tracer.run_id, kind, status, report.render(), report.to_dict())
        else:
            raise ValueError(f"Unknown job kind {kind!r}")
        tracer.log("hub", "run_finished", status=result.status)
        result.trace = tracer.summary()
        return result

    def _read_workspace(self, path: str) -> str:
        root = self.settings.workspace.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"'{path}' is not a file inside the workspace")
        return target.read_text(encoding="utf-8", errors="replace")

    # ------------------------------------------------------------------ introspection
    def info(self) -> dict[str, Any]:
        tools = self.tools_for(None)
        return {
            "version": __version__,
            "prompt_version": PROMPT_VERSION,
            "provider": self.llm.describe(),
            "worker_provider": self.worker_llm.describe(),
            "agents": {
                "orchestrator": "Plans and delegates",
                **{n: s.description for n, s in self.specialists.items()},
            },
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "side_effect": t.side_effect,
                    "untrusted_output": t.untrusted_output,
                }
                for t in tools
            ],
            "summary": {
                "strategies": list(STRATEGIES),
                "profiles": {n: p.title for n, p in PROFILES.items()},
                "workers": self.settings.summary_workers,
                "chunk_chars": self.settings.chunk_chars,
            },
            "job_kinds": list(JOB_KINDS),
            "mcp_servers": self.mcp_servers,
            "settings": self.settings.public_dict(),
        }


def _stored_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Avoid storing huge documents twice (they are in the result anyway)."""
    out = dict(payload)
    if isinstance(out.get("text"), str) and len(out["text"]) > 2000:
        out["text"] = out["text"][:2000] + f"…(+{len(out['text']) - 2000} chars)"
    return out
