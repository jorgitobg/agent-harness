from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agent_harness.application.guardrails import GuardedRegistry
from agent_harness.application.registry import ToolRegistry
from agent_harness.domain.policy import Decision, PermissionPolicy
from agent_harness.domain.tools import PermissionLevel, ToolCall, ToolErrorKind, ToolSpec

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


class ScriptedConfirmation:
    """Answers confirmation requests from a fixed script and records each request."""

    def __init__(self, answers: list[bool] | None = None, error: Exception | None = None) -> None:
        self._answers = list(answers or [])
        self._error = error
        self.asked: list[str] = []

    def confirm(self, call: ToolCall, spec: ToolSpec) -> bool:
        self.asked.append(call.name)
        if self._error is not None:
            raise self._error
        return self._answers.pop(0) if self._answers else False


class Recorder:
    """Handler that records whether it ran, so tests can prove a gate held."""

    def __init__(self) -> None:
        self.runs: list[Mapping[str, Any]] = []

    def __call__(self, args: Mapping[str, Any]) -> str:
        self.runs.append(args)
        return f"ran {args['path']}"


def build(
    confirmation: ScriptedConfirmation,
    policy: PermissionPolicy | None = None,
) -> tuple[GuardedRegistry, dict[str, Recorder]]:
    registry = ToolRegistry()
    recorders: dict[str, Recorder] = {}
    for name, level in [
        ("read_file", PermissionLevel.READ),
        ("write_file", PermissionLevel.WRITE),
        ("delete_file", PermissionLevel.DESTRUCTIVE),
    ]:
        rec = Recorder()
        recorders[name] = rec
        registry.register(ToolSpec(name, f"{name} tool.", SCHEMA, level), rec)
    guarded = GuardedRegistry(registry, policy or PermissionPolicy(), confirmation)
    return guarded, recorders


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(call_id="c1", name=name, arguments=arguments)


def test_read_is_allowed_without_asking() -> None:
    confirmation = ScriptedConfirmation()
    guarded, recs = build(confirmation)
    result = guarded.dispatch(call("read_file", path="a.txt"))
    assert result.ok and result.output == "ran a.txt"
    assert confirmation.asked == []
    assert len(recs["read_file"].runs) == 1


def test_write_asks_and_runs_when_approved() -> None:
    confirmation = ScriptedConfirmation(answers=[True])
    guarded, recs = build(confirmation)
    result = guarded.dispatch(call("write_file", path="a.txt"))
    assert result.ok
    assert confirmation.asked == ["write_file"]
    assert len(recs["write_file"].runs) == 1


def test_refusal_returns_structured_denied_and_handler_does_not_run() -> None:
    confirmation = ScriptedConfirmation(answers=[False])
    guarded, recs = build(confirmation)
    result = guarded.dispatch(call("delete_file", path="a.txt"))
    assert not result.ok
    assert result.error_kind is ToolErrorKind.DENIED
    assert result.for_model()["error"] == "denied"
    assert recs["delete_file"].runs == []


def test_broken_confirmation_fails_closed() -> None:
    confirmation = ScriptedConfirmation(error=RuntimeError("no tty"))
    guarded, recs = build(confirmation)
    result = guarded.dispatch(call("write_file", path="a.txt"))
    assert result.error_kind is ToolErrorKind.DENIED
    assert recs["write_file"].runs == []


def test_deny_never_asks_and_never_runs() -> None:
    confirmation = ScriptedConfirmation(answers=[True])
    guarded, recs = build(confirmation, PermissionPolicy(deny=frozenset({"write_file"})))
    result = guarded.dispatch(call("write_file", path="a.txt"))
    assert result.error_kind is ToolErrorKind.DENIED
    assert confirmation.asked == []
    assert recs["write_file"].runs == []


def test_allow_list_skips_confirmation_for_write() -> None:
    confirmation = ScriptedConfirmation()
    guarded, recs = build(confirmation, PermissionPolicy(allow=frozenset({"write_file"})))
    assert guarded.dispatch(call("write_file", path="a.txt")).ok
    assert confirmation.asked == []
    assert len(recs["write_file"].runs) == 1


def test_invalid_arguments_are_reported_without_asking_a_human() -> None:
    confirmation = ScriptedConfirmation(answers=[True])
    guarded, recs = build(confirmation)
    result = guarded.dispatch(call("delete_file", path=42))
    assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
    assert confirmation.asked == []
    assert recs["delete_file"].runs == []


def test_unknown_tool_is_reported_without_asking() -> None:
    confirmation = ScriptedConfirmation(answers=[True])
    guarded, _ = build(confirmation)
    result = guarded.dispatch(call("format_disk"))
    assert result.error_kind is ToolErrorKind.UNKNOWN_TOOL
    assert confirmation.asked == []


@pytest.mark.parametrize("decision", list(Decision))
def test_every_decision_is_reachable(decision: Decision) -> None:
    policy = PermissionPolicy(
        level_decisions={
            PermissionLevel.READ: decision,
            PermissionLevel.WRITE: decision,
            PermissionLevel.DESTRUCTIVE: decision,
        }
    )
    assert policy.decide(ToolSpec("x", "x.", SCHEMA, PermissionLevel.READ)) is decision
