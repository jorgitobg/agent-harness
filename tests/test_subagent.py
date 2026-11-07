from __future__ import annotations

from typing import Any

from agent_harness.adapters.scripted import ScriptedModel
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.registry import ToolRegistry
from agent_harness.application.subagent import SubAgent
from agent_harness.domain.loop import Budget, ModelTurn, Role, StopReason, Usage
from agent_harness.domain.tools import PermissionLevel, ToolCall, ToolErrorKind, ToolSpec

EMPTY: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


def _child_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec("lookup", "Look a fact up.", EMPTY), lambda _a: "fact-42")
    return reg


def _delegating_parent(sub: SubAgent) -> tuple[AgentLoop, ToolRegistry]:
    reg = ToolRegistry()
    reg.register(sub.spec, sub.handle)
    model = ScriptedModel(
        [
            ModelTurn(tool_calls=(ToolCall("p1", "research", {"task": "find the fact"}),)),
            ModelTurn(text="parent done"),
        ]
    )
    return AgentLoop(model, reg, Budget(max_steps=4)), reg


def test_only_final_answer_returns_to_parent() -> None:
    child_model = ScriptedModel(
        [
            ModelTurn(text="looking", tool_calls=(ToolCall("c1", "lookup", {}),)),
            ModelTurn(text="the fact is 42", usage=Usage(10, 5, 0.0)),
        ]
    )
    sub = SubAgent(
        "research",
        "Delegate a research task.",
        lambda: AgentLoop(child_model, _child_registry(), Budget(max_steps=3)),
    )
    parent, _ = _delegating_parent(sub)

    result = parent.run("go")

    assert result.stop_reason is StopReason.FINAL_ANSWER
    tool_msgs = [m.tool_result for m in result.transcript if m.role is Role.TOOL]
    assert len(tool_msgs) == 1 and tool_msgs[0] is not None
    assert tool_msgs[0].output == "the fact is 42"
    # the child's own tool call never entered the parent transcript
    assert all(m.tool_result is None or m.tool_result.name == "research" for m in result.transcript)
    assert len(sub.runs) == 1 and sub.runs[0].steps == 2
    assert sub.runs[0].usage.total_tokens == 15


def test_child_sees_only_its_task() -> None:
    child_model = ScriptedModel([ModelTurn(text="ok")])
    sub = SubAgent(
        "research", "d", lambda: AgentLoop(child_model, _child_registry(), Budget(max_steps=2))
    )
    parent, _ = _delegating_parent(sub)
    parent.run("secret parent context")

    first_call = child_model.calls[0]
    assert [m.content for m in first_call] == ["find the fact"]


def test_child_budget_stop_is_a_failed_tool_result_not_a_crash() -> None:
    looping = ScriptedModel(
        [ModelTurn(tool_calls=(ToolCall(f"c{i}", "lookup", {}),)) for i in range(5)]
    )
    sub = SubAgent(
        "research", "d", lambda: AgentLoop(looping, _child_registry(), Budget(max_steps=2))
    )
    parent, _ = _delegating_parent(sub)

    result = parent.run("go")

    failed = next(m.tool_result for m in result.transcript if m.tool_result is not None)
    assert not failed.ok
    assert failed.error_kind is ToolErrorKind.EXECUTION_FAILED
    assert failed.error_message is not None and "max_steps" in failed.error_message
    assert result.stop_reason is StopReason.FINAL_ANSWER  # parent carries on


def test_each_call_gets_a_fresh_child() -> None:
    built: list[int] = []

    def factory() -> AgentLoop:
        built.append(1)
        return AgentLoop(ScriptedModel([ModelTurn(text="a")]), _child_registry())

    sub = SubAgent("research", "d", factory, level=PermissionLevel.READ)
    assert sub.handle({"task": "x"}) == "a"
    assert sub.handle({"task": "y"}) == "a"
    assert len(built) == 2 and len(sub.runs) == 2
