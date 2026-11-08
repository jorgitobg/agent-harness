from __future__ import annotations

import pytest

from agent_harness import demo
from agent_harness.adapters.scripted import ScriptedModel
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.registry import ToolRegistry
from agent_harness.application.tracing import (
    ReplayDispatcher,
    ReplayModel,
    TracedDispatcher,
    TracedModel,
    TraceExhausted,
    Tracer,
)
from agent_harness.domain.loop import Budget, ModelTurn, StopReason, Usage
from agent_harness.domain.tools import PermissionLevel, ToolCall, ToolSpec
from agent_harness.domain.trace import SpanKind

SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
    "additionalProperties": False,
}


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec("double", "Double x.", SCHEMA, PermissionLevel.READ), lambda a: a["x"] * 2
    )
    return reg


def _turns() -> list[ModelTurn]:
    return [
        ModelTurn("t", (ToolCall("a", "double", {"x": 4}),), Usage(10, 5, 0.01)),
        ModelTurn("bad", (ToolCall("b", "double", {"x": "no"}),), Usage(20, 5, 0.02)),
        ModelTurn("done", usage=Usage(30, 5, 0.03)),
    ]


def _tracer() -> Tracer:
    ticks = iter(float(i) for i in range(100))
    return Tracer(clock=lambda: next(ticks))


def _traced_run() -> tuple[Tracer, object]:
    tracer = _tracer()
    reg = _registry()
    run = AgentLoop(
        TracedModel(ScriptedModel(_turns()), tracer), reg, dispatcher=TracedDispatcher(reg, tracer)
    ).run("go")
    return tracer, run


def test_one_span_per_model_and_tool_call_with_usage_and_latency() -> None:
    tracer, _ = _traced_run()
    assert [(s.kind, s.name) for s in tracer.spans] == [
        (SpanKind.MODEL, "model"),
        (SpanKind.TOOL, "double"),
        (SpanKind.MODEL, "model"),
        (SpanKind.TOOL, "double"),
        (SpanKind.MODEL, "model"),
    ]
    assert all(s.latency_s == 1.0 for s in tracer.spans)  # fake clock: 1 tick per span
    total = tracer.total_usage()
    assert (total.input_tokens, total.output_tokens) == (60, 15)
    assert total.cost_usd == pytest.approx(0.06)


def test_failed_tool_span_records_the_error_kind() -> None:
    tracer, _ = _traced_run()
    failed = [s for s in tracer.spans if s.kind is SpanKind.TOOL and s.error]
    assert [s.error for s in failed] == ["invalid_arguments"]
    assert failed[0].result is not None and not failed[0].result.ok


def test_model_error_is_traced_and_still_raised() -> None:
    tracer = _tracer()
    run = AgentLoop(TracedModel(ScriptedModel([]), tracer), _registry()).run("go")
    assert run.stop_reason is StopReason.MODEL_ERROR
    assert tracer.spans[0].error == "ScriptExhausted" and tracer.spans[0].turn is None


def test_replay_through_json_reproduces_the_run_without_model_or_tools() -> None:
    tracer, original = _traced_run()
    spans = Tracer.spans_from_jsonl(tracer.to_jsonl())
    assert spans == tracer.spans  # lossless round trip

    empty = ToolRegistry()  # no tools registered: replay must not need them
    replayed = AgentLoop(ReplayModel(spans), empty, dispatcher=ReplayDispatcher(spans)).run("go")
    assert replayed == original


def test_replay_beyond_the_trace_fails_loudly() -> None:
    tracer, _ = _traced_run()
    with pytest.raises(TraceExhausted):
        ReplayDispatcher(tracer.spans).dispatch(ToolCall("zzz", "double", {"x": 1}))
    short = [s for s in tracer.spans if s.kind is SpanKind.MODEL][:1]
    run = AgentLoop(ReplayModel(short), _registry(), dispatcher=ReplayDispatcher(tracer.spans)).run(
        "go"
    )
    assert run.stop_reason is StopReason.MODEL_ERROR


def test_budget_still_applies_during_replay() -> None:
    tracer, _ = _traced_run()
    run = AgentLoop(
        ReplayModel(tracer.spans),
        _registry(),
        Budget(max_tokens=40),
        dispatcher=ReplayDispatcher(tracer.spans),
    ).run("go")
    assert run.stop_reason is StopReason.MAX_TOKENS


def test_trace_demo_prints_spans_and_identical_replay(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main(["--trace"]) == 0
    out = capsys.readouterr().out
    assert "trace: 7 spans, tokens=600" in out
    assert "replay identical to original (transcript and usage): True" in out
