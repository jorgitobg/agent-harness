from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from agent_harness import evals
from agent_harness.domain.loop import AgentRun, Message, ModelTurn, Role, StopReason, Usage
from agent_harness.domain.tools import ToolCall, ToolErrorKind, ToolResult

ROOT = Path(__file__).resolve().parent.parent


def _by_key(outcomes: list[evals.Outcome]) -> dict[tuple[str, str], evals.Outcome]:
    return {(o.candidate, o.task): o for o in outcomes}


def test_careful_passes_everything_and_sloppy_fails_where_it_should() -> None:
    got = _by_key(evals.run_all())
    assert all(got[("careful", t.id)].passed for t in evals.TASKS)
    assert not got[("sloppy", "lookup")].passed
    assert not got[("sloppy", "save")].passed
    assert got[("sloppy", "denied-write")].passed  # the guard protects even a sloppy model
    assert got[("sloppy", "runaway")].passed


def test_runaway_is_stopped_by_the_step_budget() -> None:
    o = _by_key(evals.run_all())[("careful", "runaway")]
    assert o.stop_reason == "max_steps" and o.steps == 3


def test_scorer_rejects_a_write_that_ignored_the_denial() -> None:
    task = next(t for t in evals.TASKS if t.id == "denied-write")
    approving = evals.Task(task.id, task.prompt, True, task.max_steps, task.check)
    turns: list[ModelTurn] = evals.CANDIDATES["careful"]["denied-write"]()
    o = evals.run_task(approving, "careful", turns)
    assert not o.passed and "operator said no" in o.reason


DENIED_TASK = next(t for t in evals.TASKS if t.id == "denied-write")
PLAN: dict[str, str] = {"title": "plan", "body": "Ship the offline demo."}
USAGE = Usage(input_tokens=1, output_tokens=1)


def _call(cid: str, name: str, args: Mapping[str, object]) -> ModelTurn:
    return ModelTurn(tool_calls=(ToolCall(cid, name, args),), usage=USAGE)


def _say(text: str) -> ModelTurn:
    return ModelTurn(text=text, usage=USAGE)


def _denied_run(turns: list[ModelTurn], max_steps: int | None = None) -> evals.Outcome:
    task = DENIED_TASK
    if max_steps is not None:
        task = evals.Task(task.id, task.prompt, task.approve_writes, max_steps, task.check)
    return evals.run_task(task, "probe", turns)


def test_denied_write_passes_for_a_real_refused_attempt_then_final_answer() -> None:
    turns = [_call("a", "write_note", PLAN), _say("Refused; nothing saved.")]
    assert _denied_run(turns).passed


def test_denied_write_tolerates_a_read_before_the_refused_attempt() -> None:
    turns = [
        _call("r", "read_note", {"title": "todo"}),
        _call("a", "write_note", PLAN),
        _say("Refused; nothing saved."),
    ]
    assert _denied_run(turns).passed


def test_denied_write_fails_when_the_model_never_attempts_the_write() -> None:
    o = _denied_run([_say("I decided not to write anything.")])
    assert not o.passed and "no write of the intended note" in o.reason


def test_denied_write_fails_on_model_error() -> None:
    o = _denied_run([])  # script exhausted -> MODEL_ERROR before any attempt
    assert not o.passed and o.stop_reason == "model_error"


def test_denied_write_fails_on_model_error_after_the_refused_attempt() -> None:
    o = _denied_run([_call("a", "write_note", PLAN)])  # no turn left for a final answer
    assert not o.passed and o.stop_reason == "model_error"


def test_denied_write_fails_when_the_step_budget_stops_the_run() -> None:
    o = _denied_run([_call("a", "write_note", PLAN), _say("late")], max_steps=1)
    assert not o.passed and o.stop_reason == "max_steps"


def test_denied_write_fails_when_only_reads_happen_until_the_budget_stops() -> None:
    turns = [_call(str(i), "read_note", {"title": "todo"}) for i in range(5)]
    assert not _denied_run(turns).passed


@pytest.mark.parametrize(
    "turn",
    [
        _call("a", "write_notes", PLAN),  # unknown tool
        _call("a", "write_note", {"title": "plan"}),  # missing body: invalid arguments
        _call("a", "write_note", {**PLAN, "extra": 1}),  # extra property: invalid
    ],
    ids=["unknown-tool", "missing-body", "extra-property"],
)
def test_validation_and_unknown_tool_errors_are_not_a_policy_denial(turn: ModelTurn) -> None:
    o = _denied_run([turn, _say("done")])
    assert not o.passed and "no write of the intended note" in o.reason


@pytest.mark.parametrize(
    "args",
    [
        {"title": "other", "body": "Ship the offline demo."},
        {"title": "plan", "body": "Something else."},
    ],
    ids=["other-title", "other-body"],
)
def test_denied_write_of_a_different_note_does_not_count(args: dict[str, str]) -> None:
    o = _denied_run([_call("a", "write_note", args), _say("done")])
    assert not o.passed and "no write of the intended note" in o.reason


def test_denied_write_needs_a_final_answer_with_text() -> None:
    o = _denied_run([_call("a", "write_note", PLAN), _say("")])
    assert not o.passed


def _forged(
    *,
    result: ToolResult | None = None,
    stop: StopReason = StopReason.FINAL_ANSWER,
    final: str | None = "done",
) -> AgentRun:
    call = ToolCall("a", "write_note", PLAN)
    refused = ToolResult.failure(call, ToolErrorKind.DENIED, "operator refused")
    transcript = (
        Message(Role.USER, content="p"),
        Message(Role.ASSISTANT, tool_calls=(call,)),
        Message(Role.TOOL, tool_result=result or refused),
        Message(Role.ASSISTANT, content="done"),
    )
    return AgentRun(stop, 2, Usage(), transcript, final)


def test_forged_run_with_a_matching_denial_passes_the_scorer() -> None:
    assert DENIED_TASK.check(_forged(), evals.SEED_NOTES)[0]


@pytest.mark.parametrize("stop", [s for s in StopReason if s is not StopReason.FINAL_ANSWER])
def test_any_non_final_stop_fails(stop: StopReason) -> None:
    assert not DENIED_TASK.check(_forged(stop=stop, final=None), evals.SEED_NOTES)[0]


def test_result_for_another_call_id_does_not_count() -> None:
    other = ToolResult.failure(
        ToolCall("zzz", "write_note", PLAN), ToolErrorKind.DENIED, "operator refused"
    )
    assert not DENIED_TASK.check(_forged(result=other), evals.SEED_NOTES)[0]


def test_result_for_another_tool_name_does_not_count() -> None:
    other = ToolResult.failure(
        ToolCall("a", "read_note", PLAN), ToolErrorKind.DENIED, "operator refused"
    )
    assert not DENIED_TASK.check(_forged(result=other), evals.SEED_NOTES)[0]


@pytest.mark.parametrize(
    "kind", [k for k in ToolErrorKind if k is not ToolErrorKind.DENIED], ids=lambda k: k.value
)
def test_other_error_kinds_do_not_count_as_denial(kind: ToolErrorKind) -> None:
    other = ToolResult.failure(ToolCall("a", "write_note", PLAN), kind, "boom")
    assert not DENIED_TASK.check(_forged(result=other), evals.SEED_NOTES)[0]


def test_successful_result_does_not_count_as_denial() -> None:
    ok = ToolResult.success(ToolCall("a", "write_note", PLAN), "saved")
    assert not DENIED_TASK.check(_forged(result=ok), evals.SEED_NOTES)[0]


@pytest.mark.parametrize(
    "notes",
    [
        {**evals.SEED_NOTES, "plan": "Ship the offline demo."},
        {**evals.SEED_NOTES, "other": "x"},
        {**evals.SEED_NOTES, "welcome": "changed"},
        {"welcome": evals.SEED_NOTES["welcome"]},
    ],
    ids=["plan-stored", "extra-note", "seed-changed", "seed-deleted"],
)
def test_any_change_to_the_note_state_fails(notes: dict[str, str]) -> None:
    assert not DENIED_TASK.check(_forged(), notes)[0]


def test_run_is_deterministic() -> None:
    assert evals.run_all() == evals.run_all()


def test_write_flag_creates_both_result_files(tmp_path: Path) -> None:
    assert evals.main(["--write"], out_dir=tmp_path) == 0
    assert (tmp_path / "results.md").read_text(encoding="utf-8").startswith("# Eval results")
    assert len(json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))) == 8


def test_committed_results_match_a_fresh_run() -> None:
    outcomes = evals.run_all()
    assert (ROOT / "evals" / "results.md").read_text(encoding="utf-8") == evals.to_markdown(
        outcomes
    )
    assert (ROOT / "evals" / "results.json").read_text(encoding="utf-8") == evals.to_json(outcomes)
