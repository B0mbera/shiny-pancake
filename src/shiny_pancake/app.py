"""Backwards-compatible composition helpers. New code should use :class:`shiny_pancake.Hub`."""

from __future__ import annotations

from collections.abc import Sequence

from .config import Settings
from .hub.service import Hub, HubResult
from .llm import LLMProvider
from .tracing import Tracer


async def run_goal(
    goal: str,
    settings: Settings,
    *,
    tracer: Tracer | None = None,
    provider: LLMProvider | None = None,
    mcp_commands: Sequence[str] = (),
) -> tuple[HubResult, Tracer]:
    """Run one goal through the agent team and return ``(result, tracer)``."""
    async with Hub(settings, llm=provider, mcp_targets=mcp_commands) as hub:
        tracer = tracer or hub.new_tracer()
        result = await hub.execute("ask", {"goal": goal}, tracer)
    return result, tracer
