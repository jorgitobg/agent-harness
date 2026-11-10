from __future__ import annotations

import pytest

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


def test_long_arguments_are_truncated() -> None:
    _, shown = _ask("n", {"body": "z" * 5000})
    assert len(shown.splitlines()[0]) < MAX_ARGS_CHARS + 40 and "…" in shown
