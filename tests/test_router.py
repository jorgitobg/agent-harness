from __future__ import annotations

from collections.abc import Sequence

import pytest

from agent_harness.adapters.scripted import ScriptedModel
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.registry import ToolRegistry
from agent_harness.application.router import ModelRouter, Route, Tier, default_is_hard
from agent_harness.domain.loop import Message, ModelTurn, Role, StopReason
from agent_harness.domain.tools import ToolCall, ToolErrorKind, ToolResult


def _turn(text: str) -> ModelTurn:
    return ModelTurn(text=text)


class Broken:
    def complete(self, messages: Sequence[Message], tools: Sequence[object]) -> ModelTurn:
        raise RuntimeError("provider down")


def _failed_tool_message() -> Message:
    call = ToolCall("c", "t", {})
    return Message(Role.TOOL, tool_result=ToolResult.failure(call, ToolErrorKind.DENIED, "no"))


def test_routine_step_uses_cheap_model() -> None:
    router = ModelRouter(ScriptedModel([_turn("cheap")]), ScriptedModel([_turn("strong")]))
    assert router.complete([Message(Role.USER, "hi")], ()).text == "cheap"
    assert router.routes == [Route(Tier.CHEAP, Tier.CHEAP, fell_back=False)]


def test_step_after_failed_tool_uses_strong_model() -> None:
    router = ModelRouter(ScriptedModel([_turn("cheap")]), ScriptedModel([_turn("strong")]))
    msgs = [Message(Role.USER, "hi"), _failed_tool_message()]
    assert router.complete(msgs, ()).text == "strong"


def test_long_transcript_is_hard() -> None:
    msgs = [Message(Role.USER, "x")] * 9
    assert default_is_hard(msgs)
    assert not default_is_hard(msgs[:8])
    assert not default_is_hard([])


def test_falls_back_to_other_tier_on_error() -> None:
    router = ModelRouter(Broken(), ScriptedModel([_turn("strong")]))
    assert router.complete([Message(Role.USER, "hi")], ()).text == "strong"
    assert router.routes == [Route(Tier.CHEAP, Tier.STRONG, fell_back=True)]


def test_both_failing_propagates_and_loop_reports_model_error() -> None:
    router = ModelRouter(Broken(), Broken())
    with pytest.raises(RuntimeError):
        router.complete([Message(Role.USER, "hi")], ())
    run = AgentLoop(router, ToolRegistry()).run("hello")
    assert run.stop_reason is StopReason.MODEL_ERROR


def test_custom_classifier_and_loop_integration() -> None:
    router = ModelRouter(ScriptedModel([]), ScriptedModel([_turn("done")]), is_hard=lambda _m: True)
    run = AgentLoop(router, ToolRegistry()).run("hello")
    assert run.final_answer == "done"
    assert router.routes[0].served_by is Tier.STRONG
