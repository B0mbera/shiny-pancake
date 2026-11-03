"""Deterministic providers for tests and the no-API-key demo.

* :class:`ScriptedProvider` replays a fixed list of responses (or callables) and records every
  request, so tests can assert on exactly what the agent sent.
* :class:`OfflineDemoProvider` is a small rule-based "model" that understands the hub's roles
  (orchestrator, analyst, researcher, critic, summary worker, editor). It exercises the real
  agent loop, tool registry, worker pool and tracing end-to-end with zero network access.
"""

from __future__ import annotations

import asyncio
import itertools
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..calc import CalcError, find_expression, format_number, safe_eval
from ..summarize.extractive import STOPWORDS, extractive_summary, split_sentences
from .base import (
    CacheHint,
    LLMProvider,
    LLMResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

_ids = itertools.count(1)


def text_response(text: str, **kw: Any) -> LLMResponse:
    return LLMResponse([TextBlock(text)], stop_reason="end_turn", **kw)


def tool_response(*calls: tuple[str, dict[str, Any]], text: str = "") -> LLMResponse:
    blocks: list[Any] = [TextBlock(text)] if text else []
    blocks += [ToolUseBlock(id=f"toolu_{next(_ids):04d}", name=n, input=a) for n, a in calls]
    return LLMResponse(blocks, stop_reason="tool_use")


@dataclass
class RecordedCall:
    system: str
    messages: list[Message]
    tools: list[str]
    max_tokens: int
    effort: str | None
    cache: str


Script = LLMResponse | Callable[[RecordedCall], LLMResponse] | Exception


class ScriptedProvider(LLMProvider):
    name = "scripted"

    def __init__(self, script: Sequence[Script], *, model: str = "scripted-model"):
        self.script = list(script)
        self.model = model
        self.calls: list[RecordedCall] = []

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
        call = RecordedCall(system, list(messages), [t.name for t in tools], max_tokens, effort, cache)
        self.calls.append(call)
        if not self.script:
            raise AssertionError("ScriptedProvider ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        response = item(call) if callable(item) else item
        response.model = response.model or self.model
        return response


# ============================================================================ offline demo

_TAG = r"<{tag}[^>]*>\n?(.*?)\n?</{tag}>"
_PATH = re.compile(r"[\w./\-]+\.(?:md|txt|json|csv|py|yaml|yml)\b")
_SUMMARY_WORDS = ("summar", "összefoglal", "tl;dr", "tldr", "digest")
_RESEARCH_WORDS = (
    "file",
    "workspace",
    "playbook",
    "document",
    "notes",
    "read",
    "fájl",
    "dokumentum",
    "jegyzet",
    "find",
    "look up",
    "keress",
    "mcp",
)
_CHECK_WORDS = ("check", "verify", "review", "double-check", "ellenőr", "biztos")
_TIME_WORDS = ("time", "date", "today", "idő", "dátum", "ma ")


def _between(text: str, tag: str) -> list[str]:
    return re.findall(_TAG.format(tag=tag), text, flags=re.DOTALL)


def _attr(text: str, tag: str, attr: str) -> str | None:
    match = re.search(rf'<{tag}[^>]*\b{attr}="([^"]*)"', text)
    return match.group(1) if match else None


def _strip_untrusted(text: str) -> str:
    return re.sub(r"</?untrusted_content[^>]*>", "", text).strip()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _bullets(lines: Sequence[str], marker: str = "-") -> str:
    return "\n".join(f"{marker} {line.strip().rstrip('.')}." for line in lines if line.strip())


def _summary_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:- \[[ xX]\]|[-*•]|\d+\.)\s*", "", raw).strip()
        if line and not line.startswith("#"):
            lines.extend(split_sentences(line) or [line])
    return lines


class OfflineDemoProvider(LLMProvider):
    """Rule-based stand-in for an LLM. Deterministic, instant and free."""

    name = "offline"

    def __init__(self, model: str = "offline-demo", *, latency_s: float = 0.0):
        self.model = model
        self.latency_s = latency_s  # simulated thinking time, makes live demos and parallelism visible

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
        tool_names = [t.name for t in tools]
        if any(n.startswith("delegate_to_") for n in tool_names):
            response = self._orchestrator(messages, tool_names)
        elif "calculator" in tool_names:
            response = self._analyst(messages)
        elif "read_file" in tool_names or "list_files" in tool_names:
            response = self._researcher(messages, tool_names)
        else:
            response = self._no_tools(system, messages)
        prompt_chars = len(system) + sum(
            len(m.text) + sum(len(r.content) for r in m.tool_results) for m in messages
        )
        response.usage = Usage(
            input_tokens=_estimate_tokens("x" * prompt_chars),
            output_tokens=_estimate_tokens(response.text or "tool call"),
        )
        response.model = self.model
        if self.latency_s:
            await asyncio.sleep(self.latency_s * (1 + min(len(response.text), 2000) / 1000))
        return response

    # ------------------------------------------------------------------ history helpers
    @staticmethod
    def _history(
        messages: Sequence[Message],
    ) -> tuple[str, list[tuple[ToolUseBlock, ToolResultBlock | None]]]:
        task = messages[0].text if messages else ""
        results = {r.tool_use_id: r for m in messages for r in m.tool_results}
        calls = [(tu, results.get(tu.id)) for m in messages if m.role == "assistant" for tu in m.tool_uses]
        return task, calls

    # ------------------------------------------------------------------ orchestrator
    def _orchestrator(self, messages: Sequence[Message], tool_names: list[str]) -> LLMResponse:
        task, calls = self._history(messages)
        lower = task.lower()
        called = {tu.name for tu, _ in calls}
        expression = find_expression(task)
        path_match = _PATH.search(task)
        path = path_match.group(0) if path_match else None

        if not calls:
            plan: list[tuple[str, dict[str, Any]]] = []
            if any(w in lower for w in _SUMMARY_WORDS) and path and "summarize_document" in tool_names:
                plan.append(("summarize_document", {"path": path, "strategy": "auto"}))
            elif (
                any(w in lower for w in _RESEARCH_WORDS + _SUMMARY_WORDS)
                and "delegate_to_researcher" in tool_names
            ):
                plan.append(("delegate_to_researcher", {"task": task}))
            if (expression or any(w in lower for w in _TIME_WORDS)) and "delegate_to_analyst" in tool_names:
                detail = (
                    f"Compute exactly: {expression}." if expression else "Report the current date and time."
                )
                plan.append(("delegate_to_analyst", {"task": f"{detail} Original question: {task}"}))
            if ("remember" in lower or "jegyezd meg" in lower) and "remember" in tool_names:
                plan.append(("remember", {"text": task, "tags": ["user-note"]}))
            if not plan:
                if "delegate_to_researcher" in tool_names:
                    plan.append(("delegate_to_researcher", {"task": task}))
                else:
                    return text_response(f"I can answer directly: {task}")
            names = ", ".join(n for n, _ in plan)
            return tool_response(*plan, text=f"Plan: {names}.")

        findings = [
            self._clean_result(tu.name, r.content)
            for tu, r in calls
            if r is not None and tu.name != "delegate_to_critic"
        ]
        draft = "\n\n".join(f for f in findings if f) or "No findings."
        wants_check = any(w in lower for w in _CHECK_WORDS) or expression is not None
        if wants_check and "delegate_to_critic" in tool_names and "delegate_to_critic" not in called:
            return tool_response(
                ("delegate_to_critic", {"task": f"Check this draft answer to '{task}':\n{draft}"}),
                text="Asking the critic to double-check the draft.",
            )

        contributors = sorted({tu.name.removeprefix("delegate_to_") for tu, _ in calls})
        verdict = next(
            (
                self._clean_result(tu.name, r.content)
                for tu, r in calls
                if tu.name == "delegate_to_critic" and r is not None
            ),
            None,
        )
        answer = draft
        if verdict:
            answer += "\n\nCritic: " + " ".join(verdict.splitlines()[:2])
        answer += f"\n\n(contributors: {', '.join(contributors)})"
        return text_response(answer)

    @staticmethod
    def _clean_result(tool: str, content: str) -> str:
        content = _strip_untrusted(content)
        return re.sub(r"^\[[a-z_]+ \| [a-z_]+\]\s*", "", content).strip()

    # ------------------------------------------------------------------ analyst
    def _analyst(self, messages: Sequence[Message]) -> LLMResponse:
        task, calls = self._history(messages)
        if not calls:
            expression = find_expression(task.split("Original question:")[0]) or find_expression(task)
            if expression:
                return tool_response(
                    ("calculator", {"expression": expression}),
                    text=f"Evaluating {expression} with the calculator.",
                )
            if any(w in task.lower() for w in _TIME_WORDS):
                return tool_response(("current_time", {}))
            return text_response("I could not find an arithmetic expression to evaluate in the task.")
        tool, result = calls[-1]
        if result is None or result.is_error:
            return text_response(f"The {tool.name} tool failed: {result.content if result else 'no result'}")
        if tool.name == "calculator":
            return text_response(
                f"{tool.input['expression']} = {result.content} (checked with the calculator)."
            )
        return text_response(f"Current time: {result.content}")

    # ------------------------------------------------------------------ researcher
    def _researcher(self, messages: Sequence[Message], tool_names: list[str]) -> LLMResponse:
        task, calls = self._history(messages)
        if not calls:
            match = _PATH.search(task)
            if match and "read_file" in tool_names:
                return tool_response(("read_file", {"path": match.group(0)}))
            return tool_response(("list_files", {}), text="Looking at the workspace first.")
        tool, result = calls[-1]
        if result is None or result.is_error:
            return text_response(f"I could not access the workspace: {result.content if result else ''}")
        if tool.name == "list_files":
            files = [f for f in result.content.splitlines() if f.strip() and not f.startswith("(")]
            if not files:
                return text_response("The workspace is empty, so I found no sources.")
            words = {w for w in re.findall(r"\w+", task.lower()) if w not in STOPWORDS and len(w) > 2}
            best = max(
                files, key=lambda f: (len(words & set(re.findall(r"\w+", f.lower()))), f.endswith(".md"))
            )
            return tool_response(("read_file", {"path": best.strip()}))
        if tool.name == "read_file":
            content = _strip_untrusted(result.content)
            points = extractive_summary(content, max_sentences=4, focus=task)
            source = tool.input.get("path", "the file")
            return text_response(f"From {source}:\n{_bullets(points)}" if points else f"{source} is empty.")
        return text_response(_strip_untrusted(result.content)[:800])

    # ------------------------------------------------------------------ tool-less roles
    def _no_tools(self, system: str, messages: Sequence[Message]) -> LLMResponse:
        prompt = messages[-1].text if messages else ""
        profile = _attr(prompt, "instructions", "profile") or "key_points"
        focus_match = re.search(r"Focus especially on: (.*)", prompt)
        focus = focus_match.group(1) if focus_match else None
        if "<sections>" in prompt:
            return text_response(self._edit(prompt))
        if "<partial_summaries>" in prompt:
            texts = _between(prompt, "summary")
            return text_response(self._combine(texts, profile, focus))
        if "<current_summary>" in prompt:
            current = (_between(prompt, "current_summary") or [""])[0]
            new = (_between(prompt, "new_section") or [""])[0]
            return text_response(
                self._combine([current, self._summarize(new, profile, focus)], profile, focus)
            )
        if "<document" in prompt:
            document = (_between(prompt, "document") or [prompt])[0]
            return text_response(self._summarize(_strip_untrusted(document), profile, focus))
        if system.startswith("You are the critic"):
            return text_response(self._critic(messages[0].text))
        return text_response(f"(offline demo) {extractive_summary(prompt, max_sentences=2) or prompt[:200]}")

    @staticmethod
    def _critic(draft: str) -> str:
        findings = []
        for lhs, rhs in re.findall(r"([\d\s.+\-*/%^()]+?)\s*=\s*(-?[\d.]+)", draft):
            if not re.search(r"[+\-*/%^]", lhs):
                continue
            try:
                expected, claimed = safe_eval(lhs), float(rhs.rstrip("."))
            except (CalcError, ValueError):
                continue
            if abs(float(expected) - claimed) > 1e-9:
                findings.append(f"- {lhs.strip()} is {format_number(expected)}, not {rhs}.")
            else:
                findings.append(f"- Re-computed {lhs.strip()} = {rhs}: correct.")
        if any("not" in f for f in findings):
            return "NEEDS CHANGES\n" + "\n".join(findings)
        return "APPROVED\n" + ("\n".join(findings) or "- No arithmetic or consistency problems found.")

    @staticmethod
    def _summarize(text: str, profile: str, focus: str | None) -> str:
        if profile == "action_items":
            items = extractive_summary(text, max_sentences=5, focus=focus, mode="action_items")
            return _bullets(items, "- [ ]") if items else "- No explicit action items in this part."
        if profile == "risks":
            items = extractive_summary(text, max_sentences=4, focus=focus, mode="risks")
            return _bullets(items) if items else "- No risks or open questions stated in this part."
        if profile == "tldr":
            return " ".join(extractive_summary(text, max_sentences=1, focus=focus))
        if profile == "executive":
            return " ".join(extractive_summary(text, max_sentences=3, focus=focus))
        return _bullets(extractive_summary(text, max_sentences=4, focus=focus))

    @staticmethod
    def _combine(texts: Sequence[str], profile: str, focus: str | None) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for line in _summary_lines(text):
                key = re.sub(r"\W+", " ", line.lower()).strip()
                if not key or key.startswith(("no explicit", "no risks", "nothing to report")):
                    continue
                if any(key in other or other in key for other in seen):
                    continue  # overlap between chunks produces near-duplicates
                seen.add(key)
                lines.append(line)
        if not lines:
            return "- Nothing to report for this profile."
        freq = Counter(w for line in lines for w in re.findall(r"\w+", line.lower()) if w not in STOPWORDS)
        focus_words = set(re.findall(r"\w+", (focus or "").lower()))

        def score(line: str) -> float:
            words = re.findall(r"\w+", line.lower())
            return sum(freq[w] for w in words) / max(1, len(words)) ** 0.5 + 2 * len(focus_words & set(words))

        limits = {"tldr": 1, "executive": 3, "action_items": 8, "risks": 6}
        keep = sorted(
            sorted(range(len(lines)), key=lambda i: score(lines[i]), reverse=True)[: limits.get(profile, 6)]
        )
        chosen = [lines[i] for i in keep]
        if profile in {"tldr", "executive"}:
            return " ".join(c.rstrip(".") + "." for c in chosen)
        return _bullets(chosen, "- [ ]" if profile == "action_items" else "-")

    @staticmethod
    def _edit(prompt: str) -> str:
        title = re.search(r"Document title: (.*)", prompt)
        sections = re.findall(
            r'<section profile="([^"]*)" title="([^"]*)">\n?(.*?)\n?</section>', prompt, re.DOTALL
        )
        out = [f"# {title.group(1) if title else 'Summary'}", ""]
        for _, heading, body in sections:
            out += [f"## {heading}", body.strip(), ""]
        out.append(f"_Merged from {len(sections)} worker perspectives by the editor._")
        return "\n".join(out)
