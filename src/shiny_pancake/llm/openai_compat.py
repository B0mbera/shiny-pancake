"""Any OpenAI-compatible Chat Completions endpoint: Ollama, vLLM, LM Studio, LiteLLM, OpenAI.

Implemented with plain ``httpx`` on purpose: it shows the whole translation layer (messages,
tool calls, tool results, finish reasons, usage) with no SDK magic in between.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

from .base import (
    CacheHint,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

_FINISH = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAICompatProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "base_url": self.base_url}

    # ------------------------------------------------------------------ translation
    @staticmethod
    def to_wire(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        for msg in messages:
            if msg.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
                calls = [
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {"name": b.name, "arguments": json.dumps(b.input, ensure_ascii=False)},
                    }
                    for b in msg.tool_uses
                ]
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
                continue
            # user turn: tool results become role=tool messages, text stays a user message
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    prefix = "ERROR: " if block.is_error else ""
                    out.append(
                        {"role": "tool", "tool_call_id": block.tool_use_id, "content": prefix + block.content}
                    )
            if msg.text:
                out.append({"role": "user", "content": msg.text})
        return out

    @staticmethod
    def from_wire(payload: dict[str, Any]) -> LLMResponse:
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError) as e:
            raise LLMError(f"malformed response: {str(payload)[:200]}", provider="openai") from e
        message = choice.get("message") or {}
        blocks: list[Any] = []
        if message.get("content"):
            blocks.append(TextBlock(message["content"]))
        for i, call in enumerate(message.get("tool_calls") or []):
            fn = call.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"_raw_arguments": raw}  # surfaces as a validation error the model can fix
            blocks.append(ToolUseBlock(id=call.get("id") or f"call_{i}", name=fn.get("name", ""), input=args))
        stop = _FINISH.get(choice.get("finish_reason") or "stop", "end_turn")
        if any(isinstance(b, ToolUseBlock) for b in blocks):
            stop = "tool_use"
        usage_raw = payload.get("usage") or {}
        usage = Usage(
            input_tokens=usage_raw.get("prompt_tokens", 0) or 0,
            output_tokens=usage_raw.get("completion_tokens", 0) or 0,
        )
        return LLMResponse(blocks, stop_reason=stop, usage=usage, model=payload.get("model", ""))

    # ------------------------------------------------------------------ call
    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
        effort: str | None = None,
        cache: CacheHint = "auto",
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_wire(system, messages),
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
                }
                for t in tools
            ]
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers
            )
        except httpx.TransportError as e:
            raise LLMError(f"connection error: {e}", provider=self.name, retryable=True) from e
        if resp.status_code >= 400:
            retryable = resp.status_code == 429 or resp.status_code >= 500
            raise LLMError(
                f"HTTP {resp.status_code}: {resp.text[:300]}",
                provider=self.name,
                retryable=retryable,
                status_code=resp.status_code,
            )
        result = self.from_wire(resp.json())
        result.model = result.model or self.model
        return result

    async def aclose(self) -> None:
        await self._client.aclose()
