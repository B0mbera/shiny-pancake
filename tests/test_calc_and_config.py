import pytest

from shiny_pancake.calc import CalcError, find_expression, format_number, safe_eval
from shiny_pancake.config import Settings


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("12 * (3 + 4)", 84),
        ("2 ** 10", 1024),
        ("7 // 2", 3),
        ("-3 + 5", 2),
        ("sqrt(16)", 4.0),
        ("max(1, 9, 3)", 9),
        ("2^3", 8),
        ("6 × 7", 42),
        ("10 ÷ 4", 2.5),
    ],
)
def test_safe_eval(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "a + 1",
        "2 ** 100000",
        "1 / 0",
        "[1, 2]",
        "(lambda: 1)()",
        "True + 1",
    ],
)
def test_safe_eval_rejects_unsafe_or_invalid(expr):
    with pytest.raises(CalcError):
        safe_eval(expr)


def test_find_expression_in_free_text():
    assert find_expression("Mennyi 12 * (3 + 4)? Ellenőrizd is.") == "12 * (3 + 4)"
    assert find_expression("Compute 1250 * 0.27 + 99 please") == "1250 * 0.27 + 99"
    assert find_expression("Meeting on 2026-11-03 about pancakes") is None
    assert find_expression("No math here") is None


def test_format_number():
    assert format_number(84.0) == "84"
    assert format_number(2.5) == "2.5"


def test_settings_provider_detection(monkeypatch):
    monkeypatch.setenv("SP_MODEL", "claude-opus-5-5")
    assert Settings.from_env().provider == "offline"
    assert Settings.from_env().model == "offline-demo"  # SP_MODEL is ignored offline
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123456")
    s = Settings.from_env()
    assert (s.provider, s.model, s.worker_model) == ("anthropic", "claude-opus-5-5", "claude-haiku-5-5")
    assert s.public_dict()["anthropic_api_key"] == "set"
    assert "sk-ant" not in str(s.public_dict())
    assert Settings.from_env(provider="offline").provider == "offline"


def test_settings_validation(monkeypatch):
    monkeypatch.setenv("SP_EFFORT", "turbo")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.delenv("SP_EFFORT")
    with pytest.raises(TypeError):
        Settings.from_env(nonsense=1)
    with pytest.raises(ValueError):
        Settings.from_env(provider="gpt-banana")
