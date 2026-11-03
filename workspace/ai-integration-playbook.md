# AI Integration Playbook

This playbook describes how our team takes a large language model from a prototype notebook to a
production service. It is written for engineers who integrate models into existing systems, and
for the product owners who have to decide what to build first.

## 1. Start from the workflow, not the model

Every successful integration we shipped started from a concrete workflow with a measurable
outcome: support tickets triaged per hour, contract clauses reviewed per day, minutes saved per
incident. The model is a component inside that workflow. Teams that start by asking "what can the
model do?" usually end up with a demo that nobody uses.

Before writing any code, write down the input, the expected output, who consumes the output and
what happens when the output is wrong. If a wrong answer is cheap to detect and cheap to fix, an
autonomous agent may be acceptable. If a wrong answer is expensive, keep a human in the loop and
design the interface around review instead of automation.

## 2. Choose the simplest architecture that works

Most tasks do not need an agent. A single, well-prompted model call handles classification,
extraction, summarization and question answering. A fixed workflow of a few calls, where our code
decides the order, handles most multi-step processes. Reach for an agent, where the model decides
which tools to call and when to stop, only when the path through the task genuinely cannot be
specified in advance.

When an agent is justified, we use the orchestrator-workers pattern. An orchestrator plans and
delegates self-contained sub-tasks to specialists. Each specialist gets only the tools it needs.
Delegation is itself a tool call, so it inherits validation, timeouts, tracing and parallelism from
the tool layer. This keeps the number of moving parts small.

## 3. Treat tools as an API you design for the model

Tool names and descriptions are prompts. A tool should do one thing, have a clear name and accept
the smallest possible set of arguments. Every argument must be validated before execution, and
every failure must be returned to the model as data instead of raising an exception. The model can
usually recover from "file not found" or "invalid date" if it is told what went wrong.

Tools with side effects, such as writing files, sending email or creating tickets, must pass an
approval policy. In development the policy can approve everything automatically. In production it
should ask a human, or deny by default, for anything irreversible.

External systems should be integrated through the Model Context Protocol (MCP) wherever possible.
An MCP bridge turns any MCP server into regular tools at runtime, so the agent code never changes
when a new system is connected. The same hub can also expose its own capabilities as an MCP
server, which lets other agents use our summarizer and agent team.

## 4. Summarization at scale

Long documents do not fit comfortably into a single prompt, and even when they fit, quality drops
for content in the middle. Our default is map-reduce: the document is split into chunks along
paragraph and sentence boundaries, a pool of summary workers summarizes the chunks in parallel, and
the partial summaries are reduced hierarchically until one summary is left. The refine strategy,
which updates a running summary chunk by chunk, keeps narrative flow but cannot be parallelized.

For decision documents we run an ensemble: several worker teams summarize the same document from
different angles, for example key points, action items and risks, and an editor model merges the
sections into one report. Workers use a small, fast model with low effort. The editor uses the
main model, because the final synthesis is where quality matters most. This routing cut the cost of
our weekly report digest by roughly 80 percent without a measurable drop in reviewer ratings.

## 5. Cost and latency budgets

Every agent run has a step budget, every tool call has a timeout and every response has a token
limit. Budgets turn runaway loops into predictable failures. We track tokens and estimated cost per
run, per agent and per worker, and we alert when a workflow exceeds its budget for three days in a
row.

Prompt caching is the cheapest optimization available. Keep the system prompt and the tool list
byte-identical between calls, put volatile content such as timestamps at the end, and cache the
growing conversation in agent loops. A cache hit is billed at a fraction of the normal input price
and also reduces latency.

Provider calls must be resilient. Retry rate limits and server errors with exponential backoff and
jitter, limit the number of concurrent requests per provider, and decide explicitly whether a
failure should fall back to another model or surface to the user. Silent fallbacks can hide real
bugs, so a malformed request must never be retried on a different vendor.

## 6. Security

Content from files, web pages and external tools is untrusted. It can contain instructions written
to manipulate the model, which is called prompt injection. We fence untrusted content in clearly
marked tags, tell the model to treat it as data, and log a guardrail event when known injection
phrases appear. The real defence is least privilege: an agent that cannot send email cannot be
tricked into sending email.

File access is sandboxed to a workspace directory. Paths are resolved before use, and anything that
escapes the sandbox is rejected. Secrets never appear in prompts or traces; the tracer redacts
anything that looks like an API key.

## 7. Observability and evaluation

Every model call and tool call is a trace event with timing, token usage and outcome. Traces are
stored with the run, streamed live to the hub dashboard and exported as JSONL. When a user reports a
bad answer, the trace shows exactly which agent called which tool with which arguments.

Prompts are code. They are versioned, reviewed and covered by an evaluation suite. Each case in the
suite has an input and a set of checks: expected content, forbidden content, tools that must be
used, a maximum number of model calls and a cost ceiling. The suite runs offline in continuous
integration against a deterministic provider, which catches pipeline regressions, and against the
real model before every release, which catches quality regressions. An LLM judge with a written
rubric is used only for criteria that cannot be checked deterministically.

## 8. Rollout checklist

The team must complete these steps before a workflow goes to production:

- The owner defines the success metric and the acceptable error rate.
- Engineering adds the workflow to the evaluation suite with at least twenty cases.
- Security reviews every tool with side effects and sets its approval policy.
- Operations sets cost and latency budgets and configures alerts.
- The product owner signs off on the human review step for high-risk outputs.

Open questions remain. We do not yet have a good way to evaluate long multi-turn conversations,
vector memory is still an experiment, and the approval interface for side-effecting tools needs
user research before we build it. The decision for this quarter is to ship the summarization hub
first, because it has the clearest metric and the lowest risk.
