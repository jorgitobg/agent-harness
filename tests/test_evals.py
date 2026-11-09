from __future__ import annotations

import json
from pathlib import Path

from agent_harness import evals
from agent_harness.domain.loop import ModelTurn

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
