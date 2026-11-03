import json

import anthropic
import httpx
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from shiny_pancake.config import Settings
from shiny_pancake.llm import create_provider
from shiny_pancake.llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from shiny_pancake.llm.base import LLMError, Message, OpaqueBlock, ToolResultBlock, ToolSpec, ToolUseBlock
from shiny_pancake.llm.mock_provider import OfflineDemoProvider, ScriptedProvider, text_response
from shiny_pancake.llm.openai_compat import OpenAICompatProvider
from shiny_pancake.llm.resilience import FallbackProvider, ResilientProvider

TOOLS = [ToolSpec("add", "Add numbers", {"type": "object", "properties": {"a": {"type": "integer"}}})]


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, **kw):
        self.beta = type("Beta", (), {})()
        self.beta.messages = FakeMessages(**kw)


def sdk_message(content, stop_reason="end_turn", model="claude-sonnet-5-5"):
    return BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 10,
            },
        }
    )


def test_anthropic_request_shape_caching_fallbacks_and_effort():
    p = AnthropicProvider("claude-sonnet-5-5", client=FakeClient())
    req = p.build_request(
        system="sys", messages=[Message.user("hi")], tools=TOOLS, max_tokens=1000, effort="low"
    )
    assert req["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert req["cache_control"] == {"type": "ephemeral"}  # automatic tail caching for agent loops
    assert req["tools"][0]["input_schema"] == TOOLS[0].input_schema
    assert req["output_config"] == {"effort": "low"}
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in req  # API 2026-10-21 rejects sampling params; thinking defaults to adaptive

    prefix_only = p.build_request(system="sys", messages=[Message.user("hi")], cache="prefix")
    assert "cache_control" not in prefix_only and "cache_control" in prefix_only["system"][0]
    off = p.build_request(system="sys", messages=[Message.user("hi")], cache="off")
    assert "cache_control" not in off and "cache_control" not in off["system"][0]

    haiku = AnthropicProvider("claude-haiku-5-5", client=FakeClient())
    assert "fallbacks" not in haiku.build_request(system="", messages=[Message.user("x")])


def test_anthropic_history_replays_opaque_blocks_unchanged():
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    history = [
        Message.user("q"),
        Message(
            "assistant",
            [
                OpaqueBlock("anthropic", thinking),
                ToolUseBlock("toolu_1", "add", {"a": 1}),
                OpaqueBlock("openai", {"type": "foreign"}),
            ],
        ),
        Message("user", [ToolResultBlock("toolu_1", "", is_error=True)]),
    ]
    req = AnthropicProvider("claude-sonnet-5-5", client=FakeClient()).build_request(
        system="", messages=history
    )
    assistant = req["messages"][1]["content"]
    assert assistant[0] == thinking  # byte-identical replay keeps thinking + cache valid
    assert assistant[1] == {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1}}
    assert len(assistant) == 2  # other providers' opaque blocks are dropped
    assert req["messages"][2]["content"][0]["content"] == "(empty)"
    assert req["messages"][2]["content"][0]["is_error"] is True


async def test_anthropic_parses_sdk_response():
    msg = sdk_message(
        [
            {"type": "thinking", "thinking": "", "signature": "s1"},
            {"type": "text", "text": "Let me add."},
            {"type": "tool_use", "id": "toolu_9", "name": "add", "input": {"a": 2}},
        ],
        stop_reason="tool_use",
    )
    client = FakeClient(response=msg)
    p = AnthropicProvider("claude-sonnet-5-5", client=client)
    r = await p.complete(system="s", messages=[Message.user("x")], tools=TOOLS)
    assert r.stop_reason == "tool_use" and r.text == "Let me add." and r.model == "claude-sonnet-5-5"
    assert r.tool_uses[0].input == {"a": 2}
    assert isinstance(r.content[0], OpaqueBlock) and r.content[0].data["signature"] == "s1"
    assert (r.usage.input_tokens, r.usage.cache_read_tokens, r.usage.cache_write_tokens) == (100, 80, 10)
    assert client.beta.messages.requests[0]["model"] == "claude-sonnet-5-5"


async def test_anthropic_error_mapping():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    cases = [
        (
            anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=request), body=None),
            True,
            429,
        ),
        (
            anthropic.BadRequestError("bad", response=httpx2.Response(400, request=request), body=None),
            False,
            400,
        ),
        (
            anthropic.InternalServerError("oops", response=httpx2.Response(500, request=request), body=None),
            True,
            500,
        ),
        (anthropic.APIConnectionError(request=request), True, None),
    ]
    for error, retryable, status in cases:
        p = AnthropicProvider("claude-sonnet-5-5", client=FakeClient(error=error))
        with pytest.raises(LLMError) as info:
            await p.complete(system="", messages=[Message.user("x")])
        assert (info.value.retryable, info.value.status_code) == (retryable, status)


async def test_openai_compat_translation_roundtrip():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "model": "llama3.2",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "add", "arguments": '{"a": 5}'},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    p = OpenAICompatProvider("llama3.2", base_url="http://ollama:11434/v1/", api_key="k", client=client)
    history = [
        Message.user("add 5"),
        Message("assistant", [ToolUseBlock("c0", "add", {"a": 1})]),
        Message("user", [ToolResultBlock("c0", "1")]),
    ]
    r = await p.complete(system="be nice", messages=history, tools=TOOLS)
    body = seen["body"]
    assert seen["auth"] == "Bearer k"
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "tool"]
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c0", "content": "1"}
    assert body["tools"][0]["function"]["parameters"] == TOOLS[0].input_schema
    assert r.stop_reason == "tool_use" and r.tool_uses[0].input == {"a": 5} and r.usage.input_tokens == 12


async def test_openai_compat_errors():
    def handler(request):
        return httpx.Response(503, text="busy")

    p = OpenAICompatProvider(
        "m", base_url="http://x", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(LLMError) as info:
        await p.complete(system="", messages=[Message.user("x")])
    assert info.value.retryable and info.value.status_code == 503
    bad = OpenAICompatProvider.from_wire(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"id": "1", "function": {"name": "add", "arguments": "{not json"}}]
                    }
                }
            ]
        }
    )
    assert bad.tool_uses[0].input == {"_raw_arguments": "{not json"}


async def test_resilient_retries_only_retryable():
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    flaky = ScriptedProvider(
        [LLMError("429", retryable=True), LLMError("503", retryable=True), text_response("ok")]
    )
    p = ResilientProvider(flaky, max_retries=2, sleep=fake_sleep)
    assert (await p.complete(system="", messages=[Message.user("x")])).text == "ok"
    assert len(sleeps) == 2 and sleeps[1] >= sleeps[0] * 0.5

    broken = ScriptedProvider([LLMError("400", retryable=False)])
    with pytest.raises(LLMError):
        await ResilientProvider(broken, max_retries=5, sleep=fake_sleep).complete(system="", messages=[])
    assert len(broken.calls) == 1


async def test_fallback_provider():
    primary = ScriptedProvider([LLMError("down", retryable=True)])
    chain = FallbackProvider([primary, OfflineDemoProvider()])
    r = await chain.complete(system="", messages=[Message.user("hello there")])
    assert r.model == "offline-demo" and chain.fallbacks_used == 1

    with pytest.raises(LLMError):  # a malformed request must not silently switch vendors
        await FallbackProvider(
            [ScriptedProvider([LLMError("bad request", status_code=400)]), OfflineDemoProvider()]
        ).complete(system="", messages=[Message.user("x")])


def test_create_provider_stack(tmp_path):
    s = Settings(provider="offline", data_dir=tmp_path)
    assert isinstance(create_provider(s), ResilientProvider)
    s2 = Settings(
        provider="openai",
        model="llama3.2",
        worker_model="llama3.2",
        fallback_provider="offline",
        data_dir=tmp_path,
    )
    stack = create_provider(s2)
    assert isinstance(stack, FallbackProvider)
    assert [p.name for p in stack.providers] == ["openai", "offline"]
