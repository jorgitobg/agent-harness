from __future__ import annotations

import pytest

from agent_harness import demo
from agent_harness.adapters.confirmation import ConfirmationScriptExhausted, ScriptedConfirmation
from agent_harness.application.guardrails import GuardedRegistry
from agent_harness.domain.policy import PermissionPolicy
from agent_harness.domain.tools import ToolCall, ToolSpec


@pytest.fixture(autouse=True)
def fresh_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    # write_note mutates the module-level store; give every test its own copy.
    monkeypatch.setattr(demo, "NOTES", dict(demo.NOTES))


def _write_spec() -> ToolSpec:
    tool = demo.build_registry().get("write_note")
    assert tool is not None
    return tool.spec


def test_approved_write_runs_and_is_recorded(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main([]) == 0
    out = capsys.readouterr().out

    assert "tool write_note: ok -> saved note 'plan'" in out
    assert "confirmations asked: [('write_note', 'write')]" in out
    assert "stop_reason=final_answer steps=4" in out
    assert "plan" in demo.NOTES


def test_refused_write_is_not_executed(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main(["--deny"]) == 0
    out = capsys.readouterr().out

    assert "tool write_note: refused/failed (denied)" in out
    assert "plan" not in demo.NOTES


def test_read_tool_needs_no_confirmation() -> None:
    confirmation = ScriptedConfirmation(answers=[])
    guarded = GuardedRegistry(demo.build_registry(), PermissionPolicy(), confirmation)

    result = guarded.dispatch(ToolCall("r1", "search_notes", {"query": "todo"}))

    assert result.ok
    assert result.output == ["todo"]
    assert confirmation.asked == []


def test_scripted_confirmation_fails_loudly_when_out_of_answers() -> None:
    confirmation = ScriptedConfirmation(answers=[])
    with pytest.raises(ConfirmationScriptExhausted):
        confirmation.confirm(ToolCall("w1", "write_note", {}), _write_spec())


def test_secret_is_redacted_before_the_model_sees_it(capsys: pytest.CaptureFixture[str]) -> None:
    demo.main([])
    out = capsys.readouterr().out
    assert "[REDACTED]" in out
    assert "sk-DEMO" not in out


def test_router_flag_falls_back_to_strong(capsys: pytest.CaptureFixture[str]) -> None:
    demo.main(["--router"])
    out = capsys.readouterr().out
    assert "stop_reason=final_answer steps=4" in out
    assert out.strip().splitlines()[-1].startswith("model routes: ['cheap->strong'")


def test_subagent_demo_keeps_child_work_out_of_parent(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main(["--subagent"]) == 0
    out = capsys.readouterr().out

    assert "parent transcript: 4 messages, tool calls: ['delegate_reading']" in out
    assert "child transcript: 4 messages, steps=2, tokens=205" in out
    assert "read_note" not in out.split("child transcript")[0]
    assert "tokens=330 (own 125 + child 205)" in out
