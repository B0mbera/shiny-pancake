"""Provider decorators: bounded concurrency, retries with jittered backoff, cross-vendor fallback.

They wrap any :class:`LLMProvider`, so resilience is composed at the edge instead of being
sprinkled through agent code::

    llm = FallbackProvider([ResilientProvider(primary, max_retries=2), OfflineDemoProvider()])
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from typing import Any

from .base import CacheHint, LLMError, LLMProvider, LLMResponse, Message, ToolSpec


class ResilientProvider(LLMProvider):
    """Limits in-flight calls (protects rate limits) and retries retryable failures."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int = 2,
        max_concurrency: int = 8,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sem = asyncio.Semaphore(max_concurrency)
        self._sleep = sleep
        self.retries = 0

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
        attempt = 0
        while True:
            try:
                async with self._sem:
                    return await self.inner.complete(
                        system=system,
                        messages=messages,
                        tools=tools,
                        max_tokens=max_tokens,
                        effort=effort,
                        cache=cache,
                    )
            except LLMError as e:
                if not e.retryable or attempt >= self.max_retries:
                    raise
                delay = min(self.max_delay, self.base_delay * (2**attempt)) * (0.5 + random.random() / 2)
                attempt += 1
                self.retries += 1
                await self._sleep(delay)

    def describe(self) -> dict[str, Any]:
        return self.inner.describe() | {"max_retries": self.max_retries}

    async def aclose(self) -> None:
        await self.inner.aclose()


class FallbackProvider(LLMProvider):
    """Tries providers in order; the first one that answers wins.

    Non-retryable errors such as a 400 are re-raised immediately: a malformed request would
    fail the same way everywhere, and silently switching vendors would hide the bug.
    """

    def __init__(self, providers: Sequence[LLMProvider]):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = list(providers)
        self.name = self.providers[0].name
        self.model = self.providers[0].model
        self.fallbacks_used = 0

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
        last: LLMError | None = None
        for i, provider in enumerate(self.providers):
            try:
                response = await provider.complete(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    effort=effort,
                    cache=cache,
                )
                if i:
                    self.fallbacks_used += 1
                return response
            except LLMError as e:
                last = e
                if not e.retryable and e.status_code not in (401, 403, 404, 529):
                    raise
        assert last is not None
        raise last

    def describe(self) -> dict[str, Any]:
        return self.providers[0].describe() | {"fallbacks": [p.describe() for p in self.providers[1:]]}

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()
