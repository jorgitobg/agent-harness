from __future__ import annotations

import pytest

from agent_harness.adapters.cost_estimate import CostEstimator
from agent_harness.adapters.scripted import ScriptedModel
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.registry import ToolRegistry
from agent_harness.domain.loop import Budget, ModelTurn, StopReason, Usage


def _est(turn: ModelTurn, inp: float = 2.0, out: float = 10.0) -> tuple[ModelTurn, CostEstimator]:
    est = CostEstimator(ScriptedModel([turn]), input_per_mtok=inp, output_per_mtok=out)
    return est.complete([], []), est


def test_fills_cost_when_the_model_reports_none() -> None:
    turn, est = _est(ModelTurn(text="x", usage=Usage(1_000_000, 500_000)))
    assert turn.usage.cost_usd == pytest.approx(2.0 + 5.0)
    assert est.estimated_turns == 1


def test_reported_cost_is_never_overwritten() -> None:
    turn, est = _est(ModelTurn(usage=Usage(1000, 1000, cost_usd=0.5)))
    assert turn.usage.cost_usd == 0.5 and est.estimated_turns == 0


def test_zero_tokens_stay_free() -> None:
    turn, _ = _est(ModelTurn(usage=Usage(cost_usd=0.0)))
    assert turn.usage.cost_usd == 0.0


def test_missing_cost_is_estimated_but_reported_zero_is_preserved() -> None:
    unknown, estimator = _est(ModelTurn(usage=Usage(10, 0, cost_usd=None)))
    assert unknown.usage.cost_usd == pytest.approx(0.00002)
    assert estimator.estimated_turns == 1

    free, estimator = _est(ModelTurn(usage=Usage(10, 0, cost_usd=0.0)))
    assert free.usage.cost_usd == 0.0
    assert estimator.estimated_turns == 0


def test_negative_prices_rejected() -> None:
    with pytest.raises(ValueError):
        CostEstimator(ScriptedModel([]), input_per_mtok=-1, output_per_mtok=0)


def test_estimate_lets_the_cost_budget_stop_a_run() -> None:
    model = CostEstimator(
        ScriptedModel([ModelTurn(text="done", usage=Usage(50_000, 0))]),
        input_per_mtok=40.0,
        output_per_mtok=0.0,
    )
    run = AgentLoop(model, ToolRegistry(), Budget(max_cost_usd=1.0)).run("hi")
    assert run.stop_reason is StopReason.MAX_COST and run.usage.cost_usd == pytest.approx(2.0)
