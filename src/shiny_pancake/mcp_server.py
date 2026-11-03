"""Expose the hub *as* an MCP server, so other agents (Claude Desktop, Claude Code, any MCP
client) can use the agent team and the multi-worker summarizer as tools.

    shiny-pancake mcp-serve                       # stdio (for desktop/IDE clients)
    shiny-pancake mcp-serve --transport streamable-http --port 8765

Together with :mod:`shiny_pancake.tools.mcp_bridge` this makes MCP work in both directions:
the hub consumes MCP tools and provides MCP tools.
"""

from __future__ import annotations

from typing import Any

from . import __version__
from .hub.service import Hub
from .summarize import PROFILES, STRATEGIES

INSTRUCTIONS = (
    "shiny-pancake is a multi-agent LLM hub. Use ask_team for questions that need research, exact "
    "calculation or a reviewed answer; use summarize for long text (parallel summary workers, strategies: "
    f"{', '.join(STRATEGIES)}; profiles: {', '.join(PROFILES)})."
)


def _footer(result: Any) -> str:
    t = result.trace
    return (
        f"\n\n[run {result.run_id} · {t.get('llm_calls', 0)} LLM calls · "
        f"{t.get('tokens', {}).get('total', 0)} tokens · ~${t.get('cost_usd', 0):.4f}]"
    )


def build_mcp_server(hub: Hub) -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("shiny-pancake", instructions=INSTRUCTIONS, version=__version__)

    @server.tool()
    async def ask_team(goal: str) -> str:
        """Ask the agent team (orchestrator + researcher, analyst, critic). Returns a checked answer."""
        result = await hub.ask(goal)
        return result.output + _footer(result)

    @server.tool()
    async def summarize(
        text: str, strategy: str = "auto", profiles: list[str] | None = None, focus: str | None = None
    ) -> str:
        """Summarize long text with parallel summary workers (map-reduce / refine / ensemble)."""
        result = await hub.summarize(text, strategy=strategy, profiles=profiles, focus=focus)
        return result.output + _footer(result)

    @server.tool()
    async def list_recent_runs(limit: int = 10) -> str:
        """List the most recent hub runs with their status."""
        jobs = hub.store.list_jobs(limit=max(1, min(limit, 50)))
        return (
            "\n".join(f"{j.id}  {j.kind:<10} {j.status:<10} worker={j.worker or '-'}" for j in jobs)
            or "No runs yet."
        )

    return server


def serve(hub: Hub, *, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765) -> None:
    server = build_mcp_server(hub)
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=host, port=port)
