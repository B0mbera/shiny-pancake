"""Using shiny-pancake as a harness: bring your own model, tools, specialists and evals.

python examples/custom_harness.py            # offline
ANTHROPIC_API_KEY=... python examples/custom_harness.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from shiny_pancake import Hub, Settings
from shiny_pancake.agents import SPECIALISTS, SpecialistSpec
from shiny_pancake.evals import EvalHarness
from shiny_pancake.llm import LLMProvider, LLMResponse, Message, ToolSpec
from shiny_pancake.llm.base import CacheHint
from shiny_pancake.tools import function_tool


# 1) A custom tool: schema comes from the type hints, description from the docstring.
@function_tool
def pancake_price(stacks: int, with_syrup: bool = True) -> str:
    """Price in euros for a number of pancake stacks."""
    return f"{stacks * (3.5 if with_syrup else 3.0):.2f} EUR"


# 2) A custom specialist that only sees that tool (least privilege).
chef = SpecialistSpec(
    "chef",
    "Knows the menu and pancake prices.",
    "You are the chef. Use pancake_price for any price question and answer in one sentence.",
    tools=("pancake_price",),
)


# 3) Wrap any provider to add behaviour (here: a simple audit log) without touching agent code.
class AuditedProvider(LLMProvider):
    def __init__(self, inner: LLMProvider):
        self.inner, self.name, self.model = inner, inner.name, inner.model
        self.calls = 0

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
        self.calls += 1
        return await self.inner.complete(
            system=system, messages=messages, tools=tools, max_tokens=max_tokens, effort=effort, cache=cache
        )


async def main() -> None:
    settings = Settings.from_env()
    probe = Hub(settings)  # builds the configured provider stack (Claude, OpenAI-compatible or offline)
    llm = AuditedProvider(probe.llm)

    async with Hub(
        settings,
        llm=llm,
        worker_llm=probe.worker_llm,
        extra_tools=[pancake_price],
        specialists={**SPECIALISTS, "chef": chef},
    ) as hub:
        answer = await hub.ask("What is 12 * (3 + 4)? Check it.")
        print(answer.output, "\n", answer.trace["agents"], "\n")

        summary = await hub.summarize(path="notes/kickoff-meeting.md", strategy="ensemble")
        print(summary.output, "\n")

        report = await EvalHarness(hub).run(
            [
                {
                    "id": "math",
                    "kind": "ask",
                    "input": "What is 6 * 7?",
                    "checks": [{"type": "contains", "value": "42"}],
                },
                {
                    "id": "actions",
                    "kind": "summarize",
                    "input": "Anna will send the deck by Monday.",
                    "options": {"profiles": ["action_items"]},
                    "checks": [{"type": "contains", "value": "Anna"}],
                },
            ]
        )
        print(report.render())
        print(f"\naudited main-model calls: {llm.calls}")


if __name__ == "__main__":
    asyncio.run(main())
