"""Claude via the official Anthropic SDK (``anthropic`` 2.x, API version 2026-10-21).

What this adapter takes care of:

* **Canonical <-> wire translation** in one place (``_to_wire`` / ``_from_wire``).
* **Append-only history.** Blocks we do not interpret (``thinking``, ``fallback``, ...) are kept
  as :class:`OpaqueBlock` and replayed byte-for-byte, so prompt caches and thinking stay valid.
* **Prompt caching.** An explicit breakpoint on the static prefix (tools + system) plus top-level
  automatic caching for the growing conversation tail in agent loops.
* **Refusal fallbacks.** On models that support it, ``fallbacks="default"`` lets the API re-run a
  declined request on Anthropic's recommended model server-side.
* **Typed error mapping** into :class:`LLMError` with a ``retryable`` flag. Transport-level
  retries (429/5xx/connection) are delegated to the SDK's own backoff (``max_retries``).
* No ``temperature``/``top_p``/``top_k``: API version 2026-10-21 rejects them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import (
    CacheHint,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    OpaqueBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

# Models that accept server-side refusal fallbacks in the scalar "default" form.
FALLBACK_CAPABLE_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EPHEMERAL = {"type": "ephemeral"}


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: Any = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        prompt_cache: bool = True,
        refusal_fallbacks: bool = True,
    ):
        self.model = model
        self.prompt_cache = prompt_cache
        self.refusal_fallbacks = refusal_fallbacks
        if client is None:
            import anthropic

            # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / auth token / `ant auth login` profile
            client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._client = client

    # ------------------------------------------------------------------ request building
    def build_request(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
        effort: str | None = None,
        cache: CacheHint = "auto",
    ) -> dict[str, Any]:
        caching = self.prompt_cache and cache != "off"
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [m for m in (self._to_wire(msg) for msg in messages) if m["content"]],
        }
        if tools:
            # deterministic order keeps the cached prefix byte-identical between calls
            request["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
            ]
        if system:
            block: dict[str, Any] = {"type": "text", "text": system}
            if caching:
                block["cache_control"] = dict(EPHEMERAL)  # caches tools + system together
            request["system"] = [block]
        if caching and cache == "auto":
            request["cache_control"] = dict(EPHEMERAL)  # automatic breakpoint on the conversation tail
        if effort:
            request["output_config"] = {"effort": effort}
        if self.refusal_fallbacks and self.model in FALLBACK_CAPABLE_MODELS:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    @staticmethod
    def _to_wire(message: Message) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            elif isinstance(block, ToolResultBlock):
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content or "(empty)",
                        "is_error": block.is_error,
                    }
                )
            elif isinstance(block, OpaqueBlock) and block.provider == "anthropic":
                content.append(block.data)  # replay unchanged (thinking signatures, fallback markers)
        return {"role": message.role, "content": content}

    @staticmethod
    def _from_wire(response: Any) -> LLMResponse:
        blocks = []
        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                blocks.append(TextBlock(block.text))
            elif kind == "tool_use":
                blocks.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input or {})))
            else:
                data = block.to_dict() if hasattr(block, "to_dict") else dict(block)
                blocks.append(OpaqueBlock("anthropic", data))
        u = response.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return LLMResponse(
            blocks,
            stop_reason=response.stop_reason or "end_turn",
            usage=usage,
            model=getattr(response, "model", "") or "",
        )

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
        request = self.build_request(
            system=system, messages=messages, tools=tools, max_tokens=max_tokens, effort=effort, cache=cache
        )
        import anthropic

        try:
            response = await self._client.beta.messages.create(**request)
        except anthropic.RateLimitError as e:
            raise LLMError(
                f"rate limited: {e.message}", provider=self.name, retryable=True, status_code=429
            ) from e
        except anthropic.APIStatusError as e:
            retryable = e.status_code >= 500 or e.status_code in (408, 409)
            raise LLMError(
                f"API error {e.status_code}: {e.message}",
                provider=self.name,
                retryable=retryable,
                status_code=e.status_code,
            ) from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}", provider=self.name, retryable=True) from e
        return self._from_wire(response)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
