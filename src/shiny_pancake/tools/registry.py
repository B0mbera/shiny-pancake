"""The tool registry: one execution path for every capability (local, delegated or remote).

``ToolRegistry.execute`` turns a model's ``tool_use`` into a ``tool_result`` and guarantees:

* unknown tools and invalid arguments are reported back to the model, never raised;
* side-effecting tools pass an approval policy first (human-in-the-loop hook);
* every call has a timeout, output is truncated, external output is fenced as untrusted;
* every call is traced with arguments, duration and outcome.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin, get_type_hints

from ..guardrails import injection_signals, wrap_untrusted
from ..llm.base import ToolResultBlock, ToolSpec, ToolUseBlock

Handler = Callable[..., Any]
ApprovalPolicy = Callable[["Tool", dict[str, Any]], bool | Awaitable[bool]]


class ToolError(Exception):
    """Raise from a handler to return a clean error message to the model."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    timeout_s: float = 30.0
    side_effect: bool = False
    untrusted_output: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


# ---------------------------------------------------------------------------- validation

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def validate_args(schema: dict[str, Any], value: Any, path: str = "") -> list[str]:
    """A deliberately small JSON-Schema subset: type, required, properties, enum, items, bounds."""
    errors: list[str] = []
    label = path or "arguments"
    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        allowed = tuple(t for name in types for t in _JSON_TYPES.get(name, (object,)))
        is_bool = isinstance(value, bool)
        if not isinstance(value, allowed) or (is_bool and "boolean" not in types):
            return [f"{label}: expected {'/'.join(types)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label}: must be one of {schema['enum']}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{label}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{label}: must be <= {schema['maximum']}")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{label}: longer than {schema['maxLength']} characters")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{label}: missing required field '{req}'")
        for key, item in value.items():
            if key in props:
                errors += validate_args(props[key], item, f"{path}.{key}" if path else key)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{label}: unexpected field '{key}'")
    if isinstance(value, list | tuple) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors += validate_args(schema["items"], item, f"{label}[{i}]")
    return errors


# ---------------------------------------------------------------------------- @function_tool

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin in (list, tuple):
        args = get_args(annotation)
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin is Union or (origin is not None and type(None) in get_args(annotation)):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        return _json_type(inner[0]) if inner else {}
    if str(origin).endswith("Literal"):
        return {"enum": list(get_args(annotation))}
    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    return {}


def function_tool(
    fn: Handler | None = None, *, name: str | None = None, description: str | None = None, **options: Any
) -> Any:
    """Build a :class:`Tool` from a typed Python function (schema from type hints, docs from docstring).

    >>> @function_tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two integers.'''
    ...     return a + b
    """

    def build(func: Handler) -> Tool:
        hints = get_type_hints(func)
        props: dict[str, Any] = {}
        required: list[str] = []
        for param in inspect.signature(func).parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            props[param.name] = _json_type(hints.get(param.name, str))
            if param.default is inspect.Parameter.empty:
                required.append(param.name)
            else:
                props[param.name]["default"] = param.default
        doc = inspect.getdoc(func) or func.__name__
        schema = {"type": "object", "properties": props, "required": required}
        return Tool(name or func.__name__, description or doc, schema, func, **options)

    return build(fn) if fn is not None else build


# ---------------------------------------------------------------------------- registry


async def allow_all(tool: Tool, args: dict[str, Any]) -> bool:
    return True


async def deny_side_effects(tool: Tool, args: dict[str, Any]) -> bool:
    return False


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        max_output_chars: int = 12000,
        approval: ApprovalPolicy = allow_all,
    ):
        self._tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars
        self.approval = approval
        for tool in tools:
            self.register(tool)

    # ------------------------------------------------------------------ management
    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def subset(self, names: Iterable[str] = (), *, prefixes: Iterable[str] = ()) -> ToolRegistry:
        """Least privilege: a registry with only the named tools (and those matching prefixes)."""
        wanted = set(names)
        prefixes = tuple(prefixes)
        picked = [
            t for t in self._tools.values() if t.name in wanted or (prefixes and t.name.startswith(prefixes))
        ]
        return ToolRegistry(picked, max_output_chars=self.max_output_chars, approval=self.approval)

    # ------------------------------------------------------------------ execution
    async def execute(
        self, call: ToolUseBlock, *, tracer: Any = None, agent: str = "agent"
    ) -> ToolResultBlock:
        started = time.perf_counter()
        tool = self._tools.get(call.name)
        ok, text = False, ""
        try:
            if tool is None:
                text = f"Unknown tool '{call.name}'. Available tools: {', '.join(self._tools) or 'none'}."
            elif errors := validate_args(tool.input_schema, call.input):
                text = "Invalid arguments: " + "; ".join(errors)
            elif tool.side_effect and not await _maybe_await(self.approval(tool, call.input)):
                text = f"The approval policy denied '{call.name}'. Do not retry; explain this to the user."
            else:
                result = await asyncio.wait_for(_invoke(tool.handler, call.input), timeout=tool.timeout_s)
                text = (
                    result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                )
                ok = True
        except TimeoutError:
            text = f"Tool '{call.name}' timed out after {tool.timeout_s if tool else '?'}s."
        except ToolError as e:
            text = str(e)
        except Exception as e:  # noqa: BLE001 - errors are data for the model
            text = f"{type(e).__name__}: {e}"

        limit = self.max_output_chars if ok else max(self.max_output_chars, 2000)  # keep errors readable
        if len(text) > limit:
            text = text[:limit] + f"\n…[truncated {len(text) - limit} chars]"
        signals = injection_signals(text) if ok and tool is not None and tool.untrusted_output else []
        if ok and tool is not None and tool.untrusted_output:
            text = wrap_untrusted(text, source=tool.name)

        if tracer is not None:
            tracer.log(
                agent,
                "tool_call",
                tool=call.name,
                args=_preview_args(call.input),
                ok=ok,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                preview=text[:240] if ok else text[:400],
            )
            if signals:
                tracer.log(
                    agent, "guardrail", tool=call.name, message="possible prompt injection", signals=signals
                )
        return ToolResultBlock(tool_use_id=call.id, content=text or "(no output)", is_error=not ok)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _invoke(handler: Handler, args: dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(getattr(handler, "func", None)):
        return await handler(**args)
    result = await asyncio.to_thread(handler, **args)
    return await result if inspect.isawaitable(result) else result


def _preview_args(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 200:
            out[key] = value[:200] + f"…(+{len(value) - 200} chars)"
        else:
            out[key] = value
    return out
