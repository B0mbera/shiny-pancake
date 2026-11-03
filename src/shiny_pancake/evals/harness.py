"""Evaluation harness: run a suite of cases through the hub and score them with checks.

A suite is a JSONL (or JSON array) file of cases::

    {"id": "math", "kind": "ask", "input": "What is 12 * (3 + 4)?",
     "checks": [{"type": "contains", "value": "84"}, {"type": "used_tool", "value": "calculator"}]}

Checks are deterministic (string, regex, length, trace-based: tools/agents used, LLM calls,
cost) plus an optional ``llm_judge`` with a rubric. The same suite runs offline in CI (prompt and
pipeline regressions) and against a real model before a release (quality and cost budgets).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..llm.base import Message
from ..tracing import Tracer

if TYPE_CHECKING:  # pragma: no cover
    from ..hub.service import Hub, HubResult


@dataclass
class EvalCase:
    id: str
    kind: str = "ask"
    input: str | None = None
    input_file: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> EvalCase:
        case = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if case.input_file and base_dir is not None and not Path(case.input_file).is_absolute():
            candidate = base_dir / case.input_file
            if candidate.exists():
                case.input_file = str(candidate)
        if case.input is None and case.input_file is None:
            raise ValueError(f"eval case {case.id!r} needs 'input' or 'input_file'")
        return case

    def payload(self) -> dict[str, Any]:
        text = (
            self.input if self.input is not None else Path(self.input_file or "").read_text(encoding="utf-8")
        )
        if self.kind == "ask":
            return {"goal": text}
        if self.kind == "summarize":
            return {"text": text, **self.options}
        raise ValueError(f"eval kind must be 'ask' or 'summarize', got {self.kind!r}")


def load_suite(path: Path) -> list[EvalCase]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        items = json.loads(raw)
    else:
        items = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return [EvalCase.from_dict(item, path.parent) for item in items]


@dataclass
class CheckResult:
    type: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class CaseResult:
    id: str
    kind: str
    passed: bool
    status: str
    checks: list[CheckResult]
    output: str = ""
    duration_ms: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    cases: list[CaseResult]
    duration_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "duration_ms": round(self.duration_ms, 1),
            "tokens": sum(c.tokens for c in self.cases),
            "cost_usd": round(sum(c.cost_usd for c in self.cases), 6),
            "cases": [asdict(c) for c in self.cases],
        }

    def render(self) -> str:
        rows = [f"{'case':<28} {'kind':<10} {'result':<7} {'checks':<8} {'llm':>4} {'tokens':>7} {'ms':>7}"]
        for c in self.cases:
            ok = sum(1 for ch in c.checks if ch.passed and not ch.skipped)
            counted = sum(1 for ch in c.checks if not ch.skipped)
            rows.append(
                f"{c.id[:28]:<28} {c.kind:<10} {'PASS' if c.passed else 'FAIL':<7} {f'{ok}/{counted}':<8} "
                f"{c.llm_calls:>4} {c.tokens:>7} {c.duration_ms:>7.0f}"
            )
            for ch in c.checks:
                if not ch.passed and not ch.skipped:
                    rows.append(f"    ✗ {ch.type}: {ch.detail}")
            if c.error:
                rows.append(f"    ✗ error: {c.error}")
        rows.append(
            f"\n{self.passed}/{self.total} passed ({self.pass_rate:.0%}) in {self.duration_ms / 1000:.2f}s, "
            f"{sum(c.tokens for c in self.cases)} tokens, ~${sum(c.cost_usd for c in self.cases):.4f}"
        )
        return "\n".join(rows)


# ---------------------------------------------------------------------------- checks

CheckFn = Callable[[dict[str, Any], "HubResult", Tracer], CheckResult]


def _norm(spec: dict[str, Any], value: str) -> str:
    return value if spec.get("case_sensitive") else value.lower()


def _check_contains(spec, result, tracer):
    found = _norm(spec, str(spec["value"])) in _norm(spec, result.output)
    return CheckResult("contains", found, f"expected to find {spec['value']!r}")


def _check_not_contains(spec, result, tracer):
    found = _norm(spec, str(spec["value"])) in _norm(spec, result.output)
    return CheckResult("not_contains", not found, f"did not expect {spec['value']!r}")


def _check_contains_any(spec, result, tracer):
    values = [str(v) for v in spec["values"]]
    hit = [v for v in values if _norm(spec, v) in _norm(spec, result.output)]
    return CheckResult("contains_any", bool(hit), f"none of {values} found")


def _check_contains_all(spec, result, tracer):
    missing = [str(v) for v in spec["values"] if _norm(spec, str(v)) not in _norm(spec, result.output)]
    return CheckResult("contains_all", not missing, f"missing {missing}")


def _check_regex(spec, result, tracer):
    flags = 0 if spec.get("case_sensitive") else re.IGNORECASE
    return CheckResult(
        "regex", re.search(spec["value"], result.output, flags) is not None, f"no match for /{spec['value']}/"
    )


def _check_max_chars(spec, result, tracer):
    return CheckResult(
        "max_chars", len(result.output) <= int(spec["value"]), f"{len(result.output)} chars > {spec['value']}"
    )


def _check_min_chars(spec, result, tracer):
    return CheckResult(
        "min_chars", len(result.output) >= int(spec["value"]), f"{len(result.output)} chars < {spec['value']}"
    )


def _check_status(spec, result, tracer):
    return CheckResult("status", result.status == spec["value"], f"status was {result.status!r}")


def _check_used_tool(spec, result, tracer):
    used = {e.data.get("tool") for e in tracer.events if e.kind == "tool_call" and e.data.get("ok")}
    return CheckResult("used_tool", spec["value"] in used, f"tools used: {sorted(t for t in used if t)}")


def _check_used_agent(spec, result, tracer):
    agents = {e.agent for e in tracer.events if e.kind == "llm_call"}
    return CheckResult("used_agent", spec["value"] in agents, f"agents: {sorted(agents)}")


def _check_max_llm_calls(spec, result, tracer):
    n = sum(1 for e in tracer.events if e.kind == "llm_call")
    return CheckResult("max_llm_calls", n <= int(spec["value"]), f"{n} calls > {spec['value']}")


def _check_max_cost(spec, result, tracer):
    cost = tracer.summary()["cost_usd"]
    return CheckResult("max_cost_usd", cost <= float(spec["value"]), f"${cost:.4f} > ${spec['value']}")


def _check_min_workers(spec, result, tracer):
    workers = {e.agent for e in tracer.events if e.kind == "llm_call" and e.agent.startswith("worker-")}
    return CheckResult("min_workers", len(workers) >= int(spec["value"]), f"only {len(workers)} workers ran")


CHECKS: dict[str, CheckFn] = {
    "contains": _check_contains,
    "not_contains": _check_not_contains,
    "contains_any": _check_contains_any,
    "contains_all": _check_contains_all,
    "regex": _check_regex,
    "max_chars": _check_max_chars,
    "min_chars": _check_min_chars,
    "status": _check_status,
    "used_tool": _check_used_tool,
    "used_agent": _check_used_agent,
    "max_llm_calls": _check_max_llm_calls,
    "max_cost_usd": _check_max_cost,
    "min_workers": _check_min_workers,
}

JUDGE_SYSTEM = """You are a strict evaluation judge. Decide whether the answer satisfies the rubric.
Reply with PASS or FAIL on the first line, then one short sentence explaining why."""


class EvalHarness:
    def __init__(self, hub: Hub, *, concurrency: int = 4):
        self.hub = hub
        self.concurrency = max(1, concurrency)

    async def run(
        self, cases: Sequence[EvalCase | dict[str, Any]], *, tracer: Tracer | None = None
    ) -> EvalReport:
        parsed = [c if isinstance(c, EvalCase) else EvalCase.from_dict(c) for c in cases]
        sem = asyncio.Semaphore(self.concurrency)
        started = time.perf_counter()

        async def one(case: EvalCase) -> CaseResult:
            async with sem:
                result = await self.run_case(case)
            if tracer is not None:
                tracer.log(
                    "eval",
                    "eval_case",
                    case=case.id,
                    passed=result.passed,
                    status=result.status,
                    tokens=result.tokens,
                    duration_ms=round(result.duration_ms, 1),
                )
            return result

        results = await asyncio.gather(*(one(c) for c in parsed))
        return EvalReport(list(results), (time.perf_counter() - started) * 1000)

    async def run_case(self, case: EvalCase) -> CaseResult:
        tracer = self.hub.new_tracer(persist=False)
        started = time.perf_counter()
        try:
            result = await self.hub.execute(case.kind, case.payload(), tracer)
        except Exception as e:  # noqa: BLE001 - a crashing case is a failing case
            return CaseResult(
                case.id,
                case.kind,
                False,
                "error",
                [],
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.perf_counter() - started) * 1000,
                tags=case.tags,
            )
        checks = [await self._check(spec, result, tracer) for spec in case.checks]
        summary = tracer.summary()
        return CaseResult(
            case.id,
            case.kind,
            all(c.passed or c.skipped for c in checks),
            result.status,
            checks,
            output=result.output[:2000],
            duration_ms=(time.perf_counter() - started) * 1000,
            llm_calls=summary["llm_calls"],
            tokens=summary["tokens"]["total"],
            cost_usd=summary["cost_usd"],
            tags=case.tags,
        )

    async def _check(self, spec: dict[str, Any], result: HubResult, tracer: Tracer) -> CheckResult:
        kind = spec.get("type", "")
        if kind == "llm_judge":
            return await self._judge(spec, result)
        fn = CHECKS.get(kind)
        if fn is None:
            return CheckResult(
                kind, False, f"unknown check type; available: {sorted(CHECKS) + ['llm_judge']}"
            )
        try:
            return fn(spec, result, tracer)
        except (KeyError, ValueError, TypeError) as e:
            return CheckResult(kind, False, f"bad check spec: {e}")

    async def _judge(self, spec: dict[str, Any], result: HubResult) -> CheckResult:
        if self.hub.llm.name == "offline":
            return CheckResult("llm_judge", True, "skipped: offline provider cannot judge", skipped=True)
        user = f"<rubric>\n{spec['rubric']}\n</rubric>\n<answer>\n{result.output}\n</answer>"
        response = await self.hub.llm.complete(
            system=JUDGE_SYSTEM, messages=[Message.user(user)], max_tokens=2000, effort="low", cache="prefix"
        )
        verdict = response.text.strip()
        return CheckResult("llm_judge", verdict.upper().startswith("PASS"), verdict[:300])
