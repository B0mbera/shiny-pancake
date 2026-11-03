# Using shiny-pancake as a harness

The hub is built so its pieces can be reused: bring your own model, tools, agents, documents and
evaluation cases, and get the agent loop, worker pool, job queue, tracing and UI for free.

## 1. The `Hub` façade

```python
from shiny_pancake import Hub, Settings

settings = Settings.from_env(provider="offline")        # or rely on ANTHROPIC_API_KEY / SP_* variables
async with Hub(settings) as hub:
    result = await hub.ask("What is 12 * (3 + 4)? Check it.")
    result.status          # completed | max_steps | refused | error
    result.output          # final answer
    result.trace           # tokens, estimated cost, llm/tool calls, per-agent stats
    result.run_id          # look it up in the UI, /jobs/{id}/events or data/traces/<id>.jsonl

    summary = await hub.summarize(text, strategy="ensemble", profiles=["key_points", "risks"], focus="cost")
    summary.data["workers"]  # per-worker calls, tokens, busy time

    job = hub.submit("summarize", {"path": "ai-integration-playbook.md"})   # queued for job workers
```

`Hub(...)` keyword arguments:

| Argument | Purpose |
|---|---|
| `llm`, `worker_llm` | any `LLMProvider` for the main and worker roles |
| `extra_tools` | additional `Tool` objects (registered for every run) |
| `specialists` | the agent team (`dict[str, SpecialistSpec]`), defaults to `SPECIALISTS` |
| `mcp_targets` | MCP servers to connect: shell commands, `http(s)://` URLs or in-process server objects |
| `approval` | async policy `(tool, args) -> bool` for side-effecting tools |
| `store`, `memory` | replace the SQLite store or the memory backend |

Lower-level building blocks are public too: `hub.orchestrator(tracer)`, `hub.summarizer(tracer)`,
`hub.tools_for(tracer)`, `hub.execute(kind, payload, tracer)`.

## 2. Your own model

Implement one method. Everything else (agents, workers, evals, UI) works unchanged.

```python
from shiny_pancake.llm import LLMProvider, LLMResponse, TextBlock, ToolUseBlock, Usage

class MyProvider(LLMProvider):
    name, model = "mine", "my-model-v1"

    async def complete(self, *, system, messages, tools=(), max_tokens=4096, effort=None, cache="auto"):
        ...  # call your model; translate messages/tools from the canonical format
        return LLMResponse([TextBlock("hello")], stop_reason="end_turn", usage=Usage(10, 2), model=self.model)
```

Wrap it for production behaviour:

```python
from shiny_pancake.llm.resilience import FallbackProvider, ResilientProvider
from shiny_pancake.llm.mock_provider import OfflineDemoProvider

llm = FallbackProvider([ResilientProvider(MyProvider(), max_retries=3, max_concurrency=4), OfflineDemoProvider()])
```

Raise `LLMError(..., retryable=True)` for transient failures so the wrappers can act on them.

## 3. Your own tools

```python
from shiny_pancake.tools import Tool, ToolError, function_tool

@function_tool(timeout_s=10)
def lookup_order(order_id: str, include_items: bool = False) -> dict:
    """Fetch an order from the shop database."""
    if not order_id.startswith("ord_"):
        raise ToolError("order_id must start with 'ord_'")   # returned to the model, not raised
    return {"id": order_id, "status": "shipped"}

refund = Tool("refund_order", "Refund an order.", {"type": "object", "properties": {"order_id": {"type": "string"}},
              "required": ["order_id"]}, handler=lambda order_id: "refunded", side_effect=True)
```

* Arguments are validated against the schema before the handler runs.
* `side_effect=True` routes the call through the approval policy (`SP_APPROVAL=deny` blocks it).
* `untrusted_output=True` fences the result as `<untrusted_content>` and scans it for injection phrases.
* Sync handlers run in a thread; every call has a timeout and is traced.

## 4. Your own agents

```python
from shiny_pancake.agents import SPECIALISTS, SpecialistSpec

support = SpecialistSpec(
    "support", "Looks up orders and explains their status.",
    "You are the support specialist. Use lookup_order; never guess order data.",
    tools=("lookup_order",), max_steps=4, effort="low",
)
hub = Hub(settings, extra_tools=[lookup_order], specialists={**SPECIALISTS, "support": support})
```

The orchestrator now has a `delegate_to_support` tool, and its system prompt lists the new member.

## 5. Evaluation suites

A suite is JSONL, one case per line (`#` comments allowed). `input_file` is relative to the suite.

```json
{"id": "refund-policy", "kind": "ask", "input": "Can I refund ord_123?", "tags": ["support"],
 "checks": [{"type": "used_agent", "value": "support"}, {"type": "not_contains", "value": "refunded"},
            {"type": "max_cost_usd", "value": 0.02},
            {"type": "llm_judge", "rubric": "Explains the refund policy and does not promise a refund."}]}
{"id": "digest", "kind": "summarize", "input_file": "../workspace/notes/kickoff-meeting.md",
 "options": {"strategy": "ensemble"}, "checks": [{"type": "contains_all", "values": ["Action items", "Risks"]}]}
```

| Check | Passes when |
|---|---|
| `contains`, `not_contains` | the output (case-insensitive unless `case_sensitive`) contains / lacks `value` |
| `contains_any`, `contains_all` | any / all of `values` appear |
| `regex` | `value` matches |
| `min_chars`, `max_chars` | output length is within bounds |
| `status` | run status equals `value` |
| `used_tool`, `used_agent` | the trace shows a successful call of that tool / an LLM call by that agent |
| `max_llm_calls`, `max_cost_usd` | the run stayed within the budget |
| `min_workers` | at least `value` summary workers took part |
| `llm_judge` | the main model answers PASS for the `rubric` (skipped offline) |

```bash
shiny-pancake eval evals/smoke.jsonl my-suite.jsonl --min-pass 0.95 --json report.json
```

Run it offline in CI to catch pipeline and prompt-plumbing regressions, and against the real
model before a release to catch quality and cost regressions. From Python: `EvalHarness(hub).run(cases)`.
Through the hub: `POST /jobs {"kind": "eval", "payload": {"suite": "evals/smoke.jsonl"}}`.

## 6. MCP in both directions

Consume external MCP servers (their tools go to the researcher):

```bash
shiny-pancake run "What tools do you have?" --mcp "npx -y @modelcontextprotocol/server-everything"
```

```python
Hub(settings, mcp_targets=["uvx some-mcp-server", "https://mcp.example.com/mcp"])
```

Expose the hub to other agents:

```bash
shiny-pancake mcp-serve                                         # stdio, for desktop / IDE clients
```

```bash
shiny-pancake mcp-serve --transport streamable-http --port 8765  # network clients
```

Tools offered: `ask_team(goal)`, `summarize(text, strategy, profiles, focus)`,
`list_recent_runs(limit)`. A Claude Desktop config is in
[examples/claude_desktop_config.json](../examples/claude_desktop_config.json).

## 7. Scaling out

```bash
SP_INLINE_WORKERS=0 shiny-pancake serve --host 0.0.0.0     # API + UI only
```

```bash
shiny-pancake worker --concurrency 4                        # run as many of these as you like
```

All processes share `SP_DATA_DIR/hub.db` (SQLite WAL). In Docker this is the `pancake-data`
volume; `docker compose up --scale worker=N` adds workers.
