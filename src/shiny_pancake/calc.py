"""Safe arithmetic: an AST whitelist evaluator (never ``eval``) and an expression finder."""

from __future__ import annotations

import ast
import math
import operator
import re

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 10**100


class CalcError(ValueError):
    pass


def _normalise(expression: str) -> str:
    """Accept the symbols people actually type: ×, ÷, ^ for power, the Unicode minus."""
    return expression.replace("×", "*").replace("÷", "/").replace("^", "**").replace("−", "-")


def safe_eval(expression: str) -> float | int:
    if len(expression) > 500:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(_normalise(expression).strip(), mode="eval")
    except SyntaxError as e:
        raise CalcError(f"invalid expression: {expression!r}") from e
    return _eval(tree.body)


def _eval(node: ast.AST) -> float | int:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > MAX_EXPONENT or abs(left) > MAX_MAGNITUDE):
            raise CalcError("exponent too large")
        try:
            result = _BINOPS[type(node.op)](left, right)
        except ZeroDivisionError as e:
            raise CalcError("division by zero") from e
        if isinstance(result, int | float) and abs(result) > MAX_MAGNITUDE:
            raise CalcError("result too large")
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
    raise CalcError(f"unsupported syntax: {ast.dump(node)[:60]}")


def format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_CANDIDATE = re.compile(r"[\d(][\d\s.+\-*/%^()×÷]*[\d)]")


def find_expression(text: str) -> str | None:
    """Return the longest arithmetic expression in free text, e.g. ``12 * (3 + 4)``."""
    text = _DATE.sub(" ", text)
    best: str | None = None
    for match in _CANDIDATE.finditer(text):
        candidate = match.group(0).strip()
        if not re.search(r"\d\s*[+\-*/%^×÷]\s*[\d(]", candidate):
            continue
        # balance parentheses greedily trimmed by the regex
        while candidate.count("(") > candidate.count(")") and candidate.startswith("("):
            candidate = candidate[1:].strip()
        try:
            safe_eval(candidate)
        except (CalcError, TypeError, ValueError):
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    return best
