"""Built-in tools: calculator, time, sandboxed workspace files, long-term memory, summarization."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..calc import CalcError, format_number, safe_eval
from ..memory import MemoryStore
from .registry import ApprovalPolicy, Tool, ToolError, ToolRegistry, allow_all

MAX_FILE_CHARS = 200_000
Summarizer = Callable[..., Awaitable[Any]]


def _sandboxed(workspace: Path, relative: str) -> Path:
    root = workspace.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ToolError(f"Access denied: '{relative}' is outside the workspace.")
    return target


def build_builtin_tools(
    workspace: Path,
    memory: MemoryStore,
    *,
    summarizer: Summarizer | None = None,
    approval: ApprovalPolicy = allow_all,
) -> ToolRegistry:
    workspace = Path(workspace)
    registry = ToolRegistry(approval=approval)

    # -------------------------------------------------------------- calculator / time
    def calculator(expression: str) -> str:
        try:
            return format_number(safe_eval(expression))
        except CalcError as e:
            raise ToolError(str(e)) from e

    registry.register(
        Tool(
            "calculator",
            "Evaluate an arithmetic expression exactly. Supports + - * / // % ** ( ), sqrt, log, round, "
            "min, max, abs, pi, e.",
            {
                "type": "object",
                "properties": {"expression": {"type": "string", "maxLength": 500}},
                "required": ["expression"],
            },
            calculator,
            timeout_s=5,
        )
    )

    def current_time(timezone: str = "UTC") -> str:
        try:
            tz = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ToolError(f"Unknown timezone '{timezone}'") from e
        return datetime.now(UTC).astimezone(tz).isoformat(timespec="seconds")

    registry.register(
        Tool(
            "current_time",
            "Current date and time in an IANA timezone (default UTC).",
            {"type": "object", "properties": {"timezone": {"type": "string"}}},
            current_time,
            timeout_s=5,
        )
    )

    # -------------------------------------------------------------- workspace (sandboxed)
    def list_files(pattern: str = "**/*") -> str:
        root = workspace.resolve()
        if not root.exists():
            return "(workspace is empty)"
        files = sorted(
            str(p.relative_to(root))
            for p in root.glob(pattern)
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
        )
        return "\n".join(files[:200]) or "(no matching files)"

    registry.register(
        Tool(
            "list_files",
            "List files in the workspace (glob pattern, default all files).",
            {"type": "object", "properties": {"pattern": {"type": "string"}}},
            list_files,
            timeout_s=10,
        )
    )

    def read_file(path: str, max_chars: int = 20000) -> str:
        target = _sandboxed(workspace, path)
        if not target.is_file():
            raise ToolError(f"No such file in workspace: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars] + (
            f"\n…[{len(text) - max_chars} more chars]" if len(text) > max_chars else ""
        )

    registry.register(
        Tool(
            "read_file",
            "Read a UTF-8 text file from the workspace.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 100, "maximum": MAX_FILE_CHARS},
                },
                "required": ["path"],
            },
            read_file,
            timeout_s=10,
            untrusted_output=True,
        )
    )

    def write_file(path: str, content: str) -> str:
        target = _sandboxed(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}."

    registry.register(
        Tool(
            "write_file",
            "Create or overwrite a text file in the workspace. Requires approval.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            write_file,
            timeout_s=10,
            side_effect=True,
        )
    )

    # -------------------------------------------------------------- memory
    def remember(text: str, tags: list[str] | None = None) -> str:
        item = memory.add(text, tags or [])
        return f"Stored memory {item.id}."

    registry.register(
        Tool(
            "remember",
            "Store a fact in long-term memory so future runs can recall it.",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 4000},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text"],
            },
            remember,
            timeout_s=5,
        )
    )

    def recall(query: str, k: int = 5) -> str:
        items = memory.search(query, k=k)
        return (
            "\n".join(f"- {i.text} (tags: {', '.join(i.tags) or '-'})" for i in items)
            or "No matching memories."
        )

    registry.register(
        Tool(
            "recall",
            "Search long-term memory for facts stored in earlier runs.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
            recall,
            timeout_s=5,
        )
    )

    # -------------------------------------------------------------- multi-worker summarization
    if summarizer is not None:

        async def summarize_document(
            path: str | None = None,
            text: str | None = None,
            strategy: str = "auto",
            profiles: list[str] | None = None,
            focus: str | None = None,
        ) -> str:
            if not path and not text:
                raise ToolError("Provide either 'path' (workspace file) or 'text'.")
            if path:
                target = _sandboxed(workspace, path)
                if not target.is_file():
                    raise ToolError(f"No such file in workspace: {path}")
                text = target.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_CHARS]
            result = await summarizer(
                text, strategy=strategy, profiles=profiles or ["key_points"], focus=focus, title=path
            )
            return (
                f"{result.summary}\n\n[strategy={result.strategy} chunks={result.chunks} "
                f"workers={len(result.workers)}]"
            )

        registry.register(
            Tool(
                "summarize_document",
                "Summarize a long document with a pool of parallel summary workers. Strategies: auto, stuff, "
                "map_reduce, refine, ensemble. Profiles: key_points, executive, action_items, risks, tldr.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Workspace file to summarize"},
                        "text": {"type": "string", "description": "Raw text to summarize (instead of path)"},
                        "strategy": {
                            "type": "string",
                            "enum": ["auto", "stuff", "map_reduce", "refine", "ensemble"],
                        },
                        "profiles": {"type": "array", "items": {"type": "string"}},
                        "focus": {"type": "string"},
                    },
                },
                summarize_document,
                timeout_s=900,
                untrusted_output=True,
            )
        )

    return registry
