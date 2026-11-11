from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agent_harness.adapters.scripted import ScriptedModel, ScriptExhausted
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.registry import ToolRegistry
from agent_harness.domain.loop import Budget, Message, ModelTurn, Role, StopReason, Usage
from agent_harness.domain.tools import PermissionLevel, ToolCall, ToolErrorKind, ToolSpec

ADD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


class FakeClock:
    """Deterministic clock: each read advances time by ``step`` seconds."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.now
        self.now += self.step
        return current


class RaisingModel:
    def complete(self, messages: Any, tools: Any) -> ModelTurn:
        raise ConnectionError("upstream unavailable")


def add_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def add(args: Mapping[str, Any]) -> int:
        return int(args["a"]) + int(args["b"])

    registry.register(ToolSpec("add", "Add two integers.", ADD_SCHEMA, PermissionLevel.READ), add)
    return registry


def answer(text: str, usage: Usage | None = None) -> ModelTurn:
    return ModelTurn(text=text, usage=usage or Usage())


def call_add(call_id: str, a: Any, b: Any, usage: Usage | None = None) -> ModelTurn:
    return ModelTurn(
        text="",
        tool_calls=(ToolCall(call_id, "add", {"a": a, "b": b}),),
        usage=usage or Usage(),
    )


def test_final_answer_without_tools_stops_after_one_step() -> None:
    model = ScriptedModel([answer("hola")])
    run = AgentLoop(model, add_registry()).run("saluda")

    assert run.stop_reason is StopReason.FINAL_ANSWER
    assert run.final_answer == "hola"
    assert run.steps == 1
    assert [m.role for m in run.transcript] == [Role.USER, Role.ASSISTANT]


def test_tool_call_result_is_fed_back_before_final_answer() -> None:
    model = ScriptedModel([call_add("c1", 2, 3), answer("la suma es 5")])
    run = AgentLoop(model, add_registry()).run("suma 2 y 3")

    assert run.stop_reason is StopReason.FINAL_ANSWER
    assert run.final_answer == "la suma es 5"
    assert run.steps == 2
    tool_messages = [m for m in run.transcript if m.role is Role.TOOL]
    assert len(tool_messages) == 1
    result = tool_messages[0].tool_result
    assert result is not None and result.ok and result.output == 5
    # The second model call saw the tool result in its transcript.
    second_call = model.calls[1]
    assert any(m.role is Role.TOOL for m in second_call)


def test_invalid_arguments_go_back_to_the_model_and_the_loop_continues() -> None:
    model = ScriptedModel([call_add("c1", "dos", 3), answer("corrijo")])
    run = AgentLoop(model, add_registry()).run("suma")

    assert run.stop_reason is StopReason.FINAL_ANSWER
    result = next(m.tool_result for m in run.transcript if m.role is Role.TOOL)
    assert result is not None
    assert not result.ok
    assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS


def test_unknown_tool_is_reported_not_raised() -> None:
    model = ScriptedModel(
        [
            ModelTurn(tool_calls=(ToolCall("c1", "rm_rf", {}),)),
            answer("ok"),
        ]
    )
    run = AgentLoop(model, add_registry()).run("x")
    result = next(m.tool_result for m in run.transcript if m.role is Role.TOOL)
    assert result is not None and result.error_kind is ToolErrorKind.UNKNOWN_TOOL
    assert run.stop_reason is StopReason.FINAL_ANSWER


def test_max_steps_stops_a_model_that_never_finishes() -> None:
    turns = [call_add(f"c{i}", 1, 1) for i in range(5)]
    run = AgentLoop(ScriptedModel(turns), add_registry(), Budget(max_steps=2)).run("loop")

    assert run.stop_reason is StopReason.MAX_STEPS
    assert run.steps == 2
    assert run.final_answer is None


def test_max_tokens_stops_after_the_call_that_exceeds_it() -> None:
    model = ScriptedModel(
        [
            call_add("c1", 1, 1, Usage(input_tokens=60, output_tokens=50)),
            answer("nunca llega"),
        ]
    )
    run = AgentLoop(model, add_registry(), Budget(max_tokens=100)).run("x")

    assert run.stop_reason is StopReason.MAX_TOKENS
    assert run.steps == 1
    assert run.usage.total_tokens == 110


def test_max_tokens_is_inclusive_of_the_limit() -> None:
    model = ScriptedModel([answer("justo", Usage(input_tokens=60, output_tokens=40))])
    run = AgentLoop(model, add_registry(), Budget(max_tokens=100)).run("x")
    assert run.stop_reason is StopReason.FINAL_ANSWER


def test_max_cost_stops_when_spend_exceeds_budget() -> None:
    model = ScriptedModel(
        [
            call_add("c1", 1, 1, Usage(cost_usd=0.4)),
            call_add("c2", 1, 1, Usage(cost_usd=0.4)),
            call_add("c3", 1, 1, Usage(cost_usd=0.4)),
        ]
    )
    run = AgentLoop(model, add_registry(), Budget(max_cost_usd=1.0, max_steps=10)).run("x")

    assert run.stop_reason is StopReason.MAX_COST
    assert run.steps == 3
    assert run.usage.cost_usd == pytest.approx(1.2)


def test_timeout_stops_the_loop_using_the_injected_clock() -> None:
    # Each clock read advances 1 s. Timeout is 3 s, so the run must stop before max_steps.
    clock = FakeClock(step=1.0)
    turns = [call_add(f"c{i}", 1, 1) for i in range(10)]
    run = AgentLoop(
        ScriptedModel(turns),
        add_registry(),
        Budget(max_steps=10, timeout_s=3.0),
        clock=clock,
    ).run("slow")

    assert run.stop_reason is StopReason.TIMEOUT
    assert run.steps < 10


def test_timeout_before_first_call_never_calls_the_model() -> None:
    # Clock reads: run start = 0; first loop-top check = 5 (elapsed 5 >= timeout 1) -> stop.
    readings = iter([0.0, 5.0, 5.0, 5.0])

    def clock() -> float:
        return next(readings)

    model = ScriptedModel([answer("nunca")])
    run = AgentLoop(model, add_registry(), Budget(timeout_s=1.0), clock=clock).run("x")

    assert run.stop_reason is StopReason.TIMEOUT
    assert run.steps == 0
    assert model.calls == []


def test_model_exception_becomes_model_error_not_a_crash() -> None:
    run = AgentLoop(RaisingModel(), add_registry()).run("x")
    assert run.stop_reason is StopReason.MODEL_ERROR
    assert run.steps == 0
    assert run.final_answer is None


def test_exhausted_script_is_a_model_error() -> None:
    run = AgentLoop(ScriptedModel([]), add_registry()).run("x")
    assert run.stop_reason is StopReason.MODEL_ERROR


def test_budget_breach_wins_over_a_final_answer_in_the_same_turn() -> None:
    model = ScriptedModel([answer("respuesta cara", Usage(input_tokens=500))])
    run = AgentLoop(model, add_registry(), Budget(max_tokens=100)).run("x")

    assert run.stop_reason is StopReason.MAX_TOKENS
    assert run.final_answer is None


def test_custom_dispatcher_is_used_instead_of_the_registry() -> None:
    class Recording:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, call: ToolCall) -> Any:
            from agent_harness.domain.tools import ToolResult

            self.calls.append(call.name)
            return ToolResult.success(call, "stubbed")

    dispatcher = Recording()
    model = ScriptedModel([call_add("c1", 1, 1), answer("done")])
    run = AgentLoop(model, add_registry(), dispatcher=dispatcher).run("x")

    assert dispatcher.calls == ["add"]
    assert run.stop_reason is StopReason.FINAL_ANSWER


def test_transcript_records_every_message_in_order() -> None:
    model = ScriptedModel([call_add("c1", 1, 2), answer("3")])
    run = AgentLoop(model, add_registry()).run("suma")
    assert [m.role for m in run.transcript] == [
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    assert run.transcript[0] == Message(Role.USER, content="suma")


def test_scripted_model_records_calls_and_raises_when_empty() -> None:
    model = ScriptedModel([answer("uno")])
    model.complete((), ())
    assert len(model.calls) == 1
    with pytest.raises(ScriptExhausted):
        model.complete((), ())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_tokens": 0},
        {"max_cost_usd": -0.1},
        {"timeout_s": 0.0},
    ],
)
def test_budget_rejects_nonsense_limits(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Budget(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("tokens", [True, 1.5])
def test_usage_rejects_non_integer_token_counts(tokens: int | float | bool) -> None:
    with pytest.raises(ValueError):
        Usage(input_tokens=tokens)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [float("nan"), float("inf")])
def test_budget_rejects_non_finite_cost_limit(limit: float) -> None:
    with pytest.raises(ValueError):
        Budget(max_cost_usd=limit)


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf")])
def test_usage_rejects_non_finite_cost(cost: float) -> None:
    with pytest.raises(ValueError):
        Usage(cost_usd=cost)


def test_usage_addition_preserves_unknown_cost() -> None:
    total = Usage(input_tokens=1, cost_usd=None) + Usage(input_tokens=2, cost_usd=0.0)
    assert total.input_tokens == 3
    assert total.cost_usd is None


def test_usage_addition_rejects_overflowed_cost() -> None:
    with pytest.raises(ValueError):
        _ = Usage(cost_usd=1.0e308) + Usage(cost_usd=1.0e308)


def test_usage_rejects_negative_values_and_adds() -> None:
    with pytest.raises(ValueError):
        Usage(input_tokens=-1)
    total = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5) + Usage(
        input_tokens=3, output_tokens=4, cost_usd=0.25
    )
    assert total.total_tokens == 10
    assert total.cost_usd == pytest.approx(0.75)
