"""MCP client bridge: any MCP server's tools become ordinary registry tools at runtime.

Targets:

* a shell command, e.g. ``"npx -y @modelcontextprotocol/server-everything"`` (stdio transport)
* an ``http(s)://`` URL of a Streamable-HTTP MCP server
* an in-process ``MCPServer`` object (used by the tests, no subprocess needed)

Remote tool output is fenced as untrusted content and every call goes through the normal
registry path (validation, timeout, tracing). Requires ``pip install 'shiny-pancake[mcp]'``.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from .registry import Tool, ToolError, ToolRegistry


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name)[:64]


class MCPBridge:
    def __init__(
        self, target: Any, *, prefix: str = "mcp_", timeout_s: float = 60.0, env: dict[str, str] | None = None
    ):
        self.target = target
        self.prefix = prefix
        self.timeout_s = timeout_s
        self.env = env
        self._client: Any = None
        self.server_name: str | None = None

    def _server(self) -> Any:
        try:
            from mcp import StdioServerParameters
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "MCP support needs the optional extra: pip install 'shiny-pancake[mcp]'"
            ) from e
        if not isinstance(self.target, str):
            return self.target
        if self.target.startswith(("http://", "https://")):
            return self.target
        parts = shlex.split(self.target)
        if not parts:
            raise ValueError("Empty MCP command")
        return StdioServerParameters(command=parts[0], args=parts[1:], env=self.env)

    async def __aenter__(self) -> MCPBridge:
        from mcp import Client

        self._client = Client(self._server(), read_timeout_seconds=self.timeout_s)
        await self._client.__aenter__()
        info = getattr(self._client, "server_info", None)
        self.server_name = getattr(info, "name", None)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc)
            self._client = None

    async def list_tools(self) -> list[Any]:
        result = await self._client.list_tools()
        return list(result.tools)

    async def register_into(self, registry: ToolRegistry) -> list[str]:
        added = []
        for remote in await self.list_tools():
            local_name = _safe_name(self.prefix + remote.name)
            schema = remote.input_schema or {"type": "object", "properties": {}}

            async def handler(_remote_name: str = remote.name, **kwargs: Any) -> str:
                return await self.call(_remote_name, kwargs)

            registry.register(
                Tool(
                    local_name,
                    f"[MCP:{self.server_name or 'server'}] {remote.description or remote.name}",
                    schema,
                    handler,
                    timeout_s=self.timeout_s,
                    untrusted_output=True,
                    tags=("mcp",),
                ),
                replace=True,
            )
            added.append(local_name)
        return added

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._client.call_tool(name, arguments)
        parts = []
        for item in getattr(result, "content", None) or []:
            if getattr(item, "type", "") == "text":
                parts.append(item.text)
            else:
                parts.append(f"[{getattr(item, 'type', 'content')} content omitted]")
        structured = getattr(result, "structured_content", None)
        if structured and not parts:
            parts.append(json.dumps(structured, ensure_ascii=False))
        text = "\n".join(parts) or "(empty result)"
        if getattr(result, "is_error", False):
            raise ToolError(f"MCP tool '{name}' failed: {text}")
        return text
