from __future__ import annotations

import json

import pytest

from agent_harness.adapters import cli_confirmation
from agent_harness.adapters.cli_confirmation import MAX_ARGS_CHARS, CliConfirmation
from agent_harness.domain.tools import PermissionLevel, ToolCall, ToolSpec

SPEC = ToolSpec("write_note", "Create a note.", {"type": "object"}, PermissionLevel.WRITE)


def _ask(answer: str | BaseException, args: dict[str, object] | None = None) -> tuple[bool, str]:
    shown: list[str] = []

    def read(prompt: str) -> str:
        if isinstance(answer, BaseException):
            raise answer
        return answer

    ok = CliConfirmation(read, shown.append).confirm(
        ToolCall("c1", "write_note", args or {"title": "plan"}), SPEC
    )
    return ok, "\n".join(shown)


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES \n"])
def test_explicit_yes_approves(answer: str) -> None:
    assert _ask(answer)[0] is True


@pytest.mark.parametrize("answer", ["", "n", "no", "yep", "ok", "sure", "y es"])
def test_anything_else_refuses(answer: str) -> None:
    assert _ask(answer)[0] is False


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_no_terminal_or_ctrl_c_refuses(error: BaseException) -> None:
    ok, shown = _ask(error)
    assert ok is False and "refused" in shown


def test_operator_sees_level_tool_and_arguments() -> None:
    _, shown = _ask("n", {"title": "plan", "body": "x"})
    assert "[write] write_note" in shown and '"title": "plan"' in shown


def test_truncated_arguments_are_refused_even_if_operator_would_say_yes() -> None:
    args: dict[str, object] = {"body": "a" * (MAX_ARGS_CHARS * 2) + "HIDDEN-SUFFIX"}
    ok, shown = _ask("yes", args)
    assert ok is False
    assert "HIDDEN-SUFFIX" not in shown and "refused" in shown


def test_truncated_arguments_never_prompt() -> None:
    asked: list[str] = []

    def read(prompt: str) -> str:
        asked.append(prompt)
        return "yes"

    call = ToolCall("c1", "write_note", {"body": "z" * 5000})
    assert CliConfirmation(read, lambda _: None).confirm(call, SPEC) is False
    assert asked == []


def test_arguments_exactly_at_the_limit_can_be_approved() -> None:
    overhead = len('{"b": ""}')
    args: dict[str, object] = {"b": "x" * (MAX_ARGS_CHARS - overhead)}
    ok, shown = _ask("y", args)
    assert ok is True and "…" not in shown
    ok, _ = _ask("y", {"b": "x" * (MAX_ARGS_CHARS - overhead + 1)})
    assert ok is False


def _confirm_with_name(name: str, args: dict[str, object]) -> tuple[bool, list[str]]:
    shown: list[str] = []
    call = ToolCall("c1", name, args)
    ok = CliConfirmation(lambda _: "y", shown.append).confirm(call, SPEC)
    return ok, shown


def _is_inert(text: str) -> bool:
    return text.isprintable()


@pytest.mark.parametrize(
    "value",
    [
        "\x1b[2J\x1b]0;owned\x07",  # CSI clear screen, OSC title
        "line1\nAllow? approved\r",  # fake extra prompt lines
        "\x9b2J\x9d0;x\x9c",  # C1 controls
        "\x7f\x00",  # DEL, NUL
        "a\u202eb\u2028c\u2066",  # bidi override, line separator, isolate
    ],
)
def test_control_sequences_in_values_are_rendered_inert(value: str) -> None:
    ok, shown = _confirm_with_name("write_note", {"body": value, value: 1})
    assert ok is True
    assert len(shown) == 1 and _is_inert(shown[0])


@pytest.mark.parametrize("name", ["write\x1b[2Jnote", "write\nnote", "write\x9bnote\u202e"])
def test_control_sequences_in_tool_name_are_rendered_inert(name: str) -> None:
    ok, shown = _confirm_with_name(name, {"title": "plan"})
    assert ok is True
    assert len(shown) == 1 and _is_inert(shown[0])


def test_refusal_message_for_hostile_tool_name_is_inert() -> None:
    shown: list[str] = []
    call = ToolCall("c1", "x\x1b[2J\ny", {"body": "z" * 5000})
    assert CliConfirmation(lambda _: "y", shown.append).confirm(call, SPEC) is False
    assert shown and all(_is_inert(line) for line in shown)


def test_overlong_tool_name_is_refused() -> None:
    ok, shown = _confirm_with_name("n" * 500, {"title": "plan"})
    assert ok is False and "n" * 500 not in "".join(shown)


def test_non_ascii_printable_text_is_preserved() -> None:
    ok, shown = _confirm_with_name("write_note", {"title": "plan ñ 日本語 ✓"})
    assert ok is True and "plan ñ 日本語 ✓" in shown[0]


def test_escaping_counts_toward_the_review_limit() -> None:
    # 100 C1 control characters render as 400 escape characters: too long to review.
    ok, _ = _confirm_with_name("write_note", {"body": "\x9b" * 100})
    assert ok is False


@pytest.mark.parametrize("value", ["a" * 20_000_000, "\x9b" * 20_000_000])
def test_oversized_strings_never_reach_json_or_inert(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized arguments were serialized")

    monkeypatch.setattr(json, "dumps", forbidden)

    def bounded_inert(text: str) -> str:
        assert len(text) <= MAX_ARGS_CHARS
        return text

    monkeypatch.setattr(cli_confirmation, "_inert", bounded_inert)
    ok, shown = _ask("yes", {"body": value})
    assert not ok and "refused" in shown


def test_many_keys_and_nested_data_reject_before_sort_or_full_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("oversized collection reached JSON sorting")

    monkeypatch.setattr(json, "dumps", forbidden)
    huge: dict[str, object] = {str(i): i for i in range(100_000)}
    assert not _ask("yes", huge)[0]
    assert not _ask("yes", {"nested": [huge]})[0]


def test_cyclic_and_non_json_objects_refuse_without_running_custom_str() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    class Hostile:
        def __str__(self) -> str:
            raise AssertionError("custom str must not run")

    assert not _ask("yes", cyclic)[0]
    assert not _ask("yes", {"body": Hostile()})[0]


def test_small_json_is_rendered_with_the_existing_sorted_format() -> None:
    args: dict[str, object] = {"z": [None, True, 1.5], "a": {"b": "ñ"}}
    ok, shown = _ask("y", args)
    assert ok and shown == '[write] write_note {"a": {"b": "ñ"}, "z": [null, true, 1.5]}'
