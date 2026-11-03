"""The core agent loop: LLM -> tool calls (in parallel) -> results -> repeat, under a step budget."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMError, LLMProvider, Message, ToolResultBlock, Usage
from ..tools.registry import ToolRegistry
from .prompts import PROMPT_VERSION


@dataclass
class AgentResult:
    agent: str
    status: str  # completed | max_steps | refused | error
    output: str
    steps: int
    usage: Usage = field(default_factory=Usage)
    messages: list[Message] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "status": self.status,
            "output": self.output,
            "steps": self.steps,
            "usage": self.usage.to_dict(),
        }


class Agent:
    def __init__(
        self,
        name: str,
        *,
        llm: LLMProvider,
        system_prompt: str,
        tools: ToolRegistry | None = None,
        tracer: Any = None,
        max_steps: int = 8,
        max_tokens: int = 16000,
        effort: str | None = None,
        description: str = "",
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or ToolRegistry()
        self.tracer = tracer
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.effort = effort
        self.description = description

    def _log(self, kind: str, **data: Any) -> None:
        if self.tracer is not None:
            self.tracer.log(self.name, kind, **data)

    async def run(self, task: str) -> AgentResult:
        self._log("agent_started", task=task[:500], tools=self.tools.names(), prompt_version=PROMPT_VERSION)
        messages: list[Message] = [Message.user(task)]
        usage = Usage()
        last_text = ""
        specs = self.tools.specs()

        for step in range(1, self.max_steps + 1):
            started = time.perf_counter()
            try:
                response = await self.llm.complete(
                    system=self.system_prompt,
                    messages=messages,
                    tools=specs,
                    max_tokens=self.max_tokens,
                    effort=self.effort,
                )
            except LLMError as e:
                self._log(
                    "llm_error", step=step, error=str(e), retryable=e.retryable, status_code=e.status_code
                )
                return self._finish("error", f"LLM error: {e}", step, usage, messages)
            usage = usage + response.usage
            self._log(
                "llm_call",
                step=step,
                model=response.model or self.llm.model,
                provider=self.llm.name,
                stop_reason=response.stop_reason,
                usage=response.usage.to_dict(),
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                tool_calls=[t.name for t in response.tool_uses],
                text_preview=response.text[:300],
            )
            messages.append(Message("assistant", response.content))
            if response.text:
                last_text = response.text

            if response.stop_reason == "refusal":
                return self._finish(
                    "refused", response.text or "The model declined this request.", step, usage, messages
                )
            if response.stop_reason == "pause_turn":
                continue  # server-side tool loop paused; re-send to let it continue

            tool_uses = response.tool_uses
            if not tool_uses:
                if response.stop_reason == "max_tokens":
                    self._log("warning", message="answer truncated by max_tokens", step=step)
                return self._finish("completed", response.text, step, usage, messages)

            if response.stop_reason == "max_tokens":
                # tool input may be cut off mid-JSON: never execute a truncated call
                results = [
                    ToolResultBlock(
                        t.id,
                        "Tool call was truncated by max_tokens and not executed. Retry with a shorter input.",
                        is_error=True,
                    )
                    for t in tool_uses
                ]
            else:
                results = list(
                    await asyncio.gather(
                        *(self.tools.execute(t, tracer=self.tracer, agent=self.name) for t in tool_uses)
                    )
                )
            # all results go back in ONE user message (keeps parallel tool use working well)
            messages.append(Message("user", list(results)))

        self._log("budget_exhausted", max_steps=self.max_steps)
        return self._finish(
            "max_steps",
            last_text or "Step budget exhausted before a final answer.",
            self.max_steps,
            usage,
            messages,
        )

    def _finish(
        self, status: str, output: str, steps: int, usage: Usage, messages: list[Message]
    ) -> AgentResult:
        self._log("agent_finished", status=status, steps=steps, tokens=usage.total)
        return AgentResult(self.name, status, output, steps, usage, messages)
