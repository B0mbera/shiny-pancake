"""Vendor-neutral LLM interface and the canonical message format.

Internally every conversation is a list of :class:`Message` objects whose content is a list of
blocks (text / tool_use / tool_result / opaque). Providers translate to and from their wire
format in exactly one place, so adding a vendor never touches the agents.

``OpaqueBlock`` carries provider-specific blocks (for Claude: ``thinking`` and ``fallback``
blocks) that must be sent back *unchanged* on the next request to the same provider. Keeping
them makes every agent loop append-only, which keeps prompt caches and thinking valid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CacheHint = Literal["auto", "prefix", "off"]


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass
class OpaqueBlock:
    provider: str
    data: dict[str, Any]
    type: Literal["opaque"] = "opaque"


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | OpaqueBlock


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls("user", [TextBlock(text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls("assistant", [TextBlock(text)])

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class LLMResponse:
    content: list[ContentBlock]
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock)).strip()

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class LLMError(Exception):
    """Provider failure. ``retryable`` tells resilience wrappers whether another attempt makes sense."""

    def __init__(
        self, message: str, *, provider: str = "", retryable: bool = False, status_code: int | None = None
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class LLMProvider(ABC):
    """One method to implement: :meth:`complete`."""

    name: str = "base"
    model: str = ""

    @abstractmethod
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
        """Run one model turn.

        ``cache`` is a hint: ``auto`` caches the stable prefix *and* the growing conversation
        (agent loops), ``prefix`` caches only tools + system (one-shot calls with a unique tail),
        ``off`` disables caching. Providers without caching ignore it.
        """

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model}
