"""Orchestrator-workers: specialists are exposed to the orchestrator *as tools*.

Delegation therefore reuses the registry's validation, timeouts, tracing and parallel execution:
there is no special protocol, and a new specialist is one entry in ``SPECIALISTS``.
Each specialist gets a least-privilege subset of the tools.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..llm.base import LLMProvider
from ..tools.registry import Tool, ToolRegistry
from . import prompts
from .agent import Agent, AgentResult


@dataclass(frozen=True)
class SpecialistSpec:
    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    tool_prefixes: tuple[str, ...] = ()
    max_steps: int | None = None
    effort: str | None = None


SPECIALISTS: dict[str, SpecialistSpec] = {
    s.name: s
    for s in (
        SpecialistSpec(
            "researcher",
            "Finds facts in workspace files, long-term memory and external MCP tools.",
            prompts.RESEARCHER,
            tools=("list_files", "read_file", "recall", "summarize_document"),
            tool_prefixes=("mcp",),
        ),
        SpecialistSpec(
            "analyst",
            "Does exact arithmetic, estimates and date/time reasoning with a calculator.",
            prompts.ANALYST,
            tools=("calculator", "current_time"),
        ),
        SpecialistSpec(
            "critic",
            "Reviews a draft answer for errors, gaps and unsupported claims (no tools).",
            prompts.CRITIC,
        ),
    )
}

ORCHESTRATOR_TOOLS = ("remember", "recall", "summarize_document")


class Orchestrator:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        tracer: Any = None,
        *,
        max_steps: int = 8,
        max_tokens: int = 16000,
        effort: str | None = None,
        specialists: Mapping[str, SpecialistSpec] | None = None,
    ):
        self.tracer = tracer
        self.specialists = dict(specialists if specialists is not None else SPECIALISTS)
        self.team: dict[str, Agent] = {
            spec.name: Agent(
                spec.name,
                llm=llm,
                system_prompt=spec.system_prompt,
                tools=tools.subset(spec.tools, prefixes=spec.tool_prefixes),
                tracer=tracer,
                max_steps=spec.max_steps or max_steps,
                max_tokens=max_tokens,
                effort=spec.effort or effort,
                description=spec.description,
            )
            for spec in self.specialists.values()
        }
        registry = tools.subset(ORCHESTRATOR_TOOLS)
        for name, agent in self.team.items():
            registry.register(self._delegate_tool(name, agent))
        team_text = "\n".join(
            f"- {n}: {a.description} (tools: {', '.join(a.tools.names()) or 'none'})"
            for n, a in self.team.items()
        )
        self.agent = Agent(
            "orchestrator",
            llm=llm,
            system_prompt=prompts.ORCHESTRATOR.format(team=team_text),
            tools=registry,
            tracer=tracer,
            max_steps=max_steps,
            max_tokens=max_tokens,
            effort=effort,
        )

    @staticmethod
    def _delegate_tool(name: str, agent: Agent) -> Tool:
        async def delegate(task: str) -> str:
            result = await agent.run(task)
            return f"[{name} | {result.status}] {result.output}"

        return Tool(
            f"delegate_to_{name}",
            f"Delegate a self-contained sub-task to the {name}. {agent.description} "
            "The specialist has no other context, so include everything it needs.",
            {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Complete, self-contained task description"}
                },
                "required": ["task"],
            },
            delegate,
            timeout_s=900,
            tags=("delegation",),
        )

    async def run(self, goal: str) -> AgentResult:
        return await self.agent.run(goal)
