from shiny_pancake.agents import Agent, Orchestrator
from shiny_pancake.llm.base import LLMError, LLMResponse, TextBlock, ToolResultBlock, ToolUseBlock
from shiny_pancake.llm.mock_provider import (
    OfflineDemoProvider,
    ScriptedProvider,
    text_response,
    tool_response,
)
from shiny_pancake.memory import MemoryStore
from shiny_pancake.tools import Tool, ToolRegistry, build_builtin_tools
from shiny_pancake.tracing import Tracer


def registry():
    return ToolRegistry(
        [
            Tool(
                "add",
                "add",
                {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
                lambda a, b: str(a + b),
            ),
        ]
    )


async def test_parallel_tool_calls_return_in_one_message():
    llm = ScriptedProvider(
        [tool_response(("add", {"a": 1, "b": 2}), ("add", {"a": 3, "b": 4})), text_response("3 and 7")]
    )
    result = await Agent("a", llm=llm, system_prompt="s", tools=registry(), tracer=Tracer()).run("go")
    assert result.status == "completed" and result.output == "3 and 7" and result.steps == 2
    second_request = llm.calls[1].messages
    tool_results = second_request[-1].tool_results
    assert [r.content for r in tool_results] == ["3", "7"]  # both results, one user message
    assert llm.calls[0].tools == ["add"]


async def test_step_budget():
    llm = ScriptedProvider([tool_response(("add", {"a": 1, "b": 1}), text="thinking") for _ in range(3)])
    result = await Agent("a", llm=llm, system_prompt="s", tools=registry(), max_steps=3).run("loop forever")
    assert result.status == "max_steps" and result.steps == 3 and result.output == "thinking"


async def test_refusal_error_and_truncated_tool_call():
    refused = ScriptedProvider([LLMResponse([TextBlock("I can't help with that.")], stop_reason="refusal")])
    assert (await Agent("a", llm=refused, system_prompt="s").run("x")).status == "refused"

    failing = ScriptedProvider([LLMError("overloaded", retryable=True, status_code=529)])
    result = await Agent("a", llm=failing, system_prompt="s").run("x")
    assert result.status == "error" and "overloaded" in result.output

    truncated = LLMResponse([ToolUseBlock("t1", "add", {"a": 1})], stop_reason="max_tokens")
    llm = ScriptedProvider([truncated, text_response("ok")])
    result = await Agent("a", llm=llm, system_prompt="s", tools=registry()).run("x")
    sent_back: ToolResultBlock = llm.calls[1].messages[-1].tool_results[0]
    assert sent_back.is_error and "truncated" in sent_back.content
    assert result.status == "completed"


async def test_pause_turn_continues():
    llm = ScriptedProvider(
        [LLMResponse([TextBlock("partial")], stop_reason="pause_turn"), text_response("done")]
    )
    result = await Agent("a", llm=llm, system_prompt="s").run("x")
    assert result.output == "done" and result.steps == 2


async def test_orchestrator_least_privilege_and_delegation(workspace, tmp_path):
    tools = build_builtin_tools(workspace, MemoryStore(None))
    tools.register(Tool("mcp0_weather", "remote", {"type": "object"}, lambda: "sunny"))
    tracer = Tracer()
    orch = Orchestrator(OfflineDemoProvider(), tools, tracer)
    assert sorted(orch.team["analyst"].tools.names()) == ["calculator", "current_time"]
    assert orch.team["critic"].tools.names() == []
    assert "mcp0_weather" in orch.team["researcher"].tools.names()
    assert "mcp0_weather" not in orch.agent.tools.names()
    assert {"delegate_to_researcher", "delegate_to_analyst", "delegate_to_critic"} <= set(
        orch.agent.tools.names()
    )

    result = await orch.run("What is 12 * (3 + 4)? Check it.")
    assert result.status == "completed" and "84" in result.output and "APPROVED" in result.output
    agents = {e.agent for e in tracer.events if e.kind == "llm_call"}
    assert agents == {"orchestrator", "analyst", "critic"}


async def test_offline_researcher_reads_workspace(workspace):
    tracer = Tracer()
    orch = Orchestrator(OfflineDemoProvider(), build_builtin_tools(workspace, MemoryStore(None)), tracer)
    result = await orch.run("What do the kickoff notes in the workspace say?")
    assert "Bence will ship the MCP bridge" in result.output
    assert [e.data["tool"] for e in tracer.events if e.kind == "tool_call" and e.agent == "researcher"] == [
        "list_files",
        "read_file",
    ]
