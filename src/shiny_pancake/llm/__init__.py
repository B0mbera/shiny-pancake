"""LLM providers behind one interface, plus the factory that wires them from :class:`Settings`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    ContentBlock,
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

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings

__all__ = [
    "ContentBlock",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "OpaqueBlock",
    "TextBlock",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "Usage",
    "create_provider",
    "build_raw_provider",
]


def build_raw_provider(settings: Settings, provider: str, model: str) -> LLMProvider:
    """One concrete provider, no wrappers."""
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model,
            api_key=settings.anthropic_api_key,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            prompt_cache=settings.prompt_cache,
            refusal_fallbacks=settings.refusal_fallbacks,
        )
    if provider == "openai":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
        )
    if provider == "offline":
        from .mock_provider import OfflineDemoProvider

        return OfflineDemoProvider(latency_s=settings.offline_latency)
    raise ValueError(f"Unknown provider: {provider}")


def create_provider(settings: Settings, *, role: str = "main") -> LLMProvider:
    """Build the provider stack for a role (``main`` or ``worker``).

    stack = [Fallback]( Resilient(primary) [, Resilient(fallback)] )

    The Anthropic SDK already retries 429/5xx with backoff, so our wrapper only adds bounded
    concurrency for it; other providers also get retries here.
    """
    from .resilience import FallbackProvider, ResilientProvider

    model = settings.worker_model if role == "worker" else settings.model

    def wrap(provider: LLMProvider) -> LLMProvider:
        retries = 0 if provider.name in ("anthropic", "offline") else settings.max_retries
        return ResilientProvider(provider, max_retries=retries, max_concurrency=settings.max_concurrency)

    primary = wrap(build_raw_provider(settings, settings.provider, model))
    fallback_name = settings.fallback_provider
    if not fallback_name or fallback_name == settings.provider:
        return primary
    from ..config import DEFAULT_MODELS

    main_default, worker_default = DEFAULT_MODELS[fallback_name]
    fallback_model = (worker_default if role == "worker" else None) or main_default
    return FallbackProvider([primary, wrap(build_raw_provider(settings, fallback_name, fallback_model))])
