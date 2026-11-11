from __future__ import annotations

from collections.abc import Sequence

import pytest

from agent_harness.adapters.scripted import ScriptedModel
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.context import (
    SUMMARY_PREFIX,
    ContextCompactor,
    ModelSummarizer,
    estimate_tokens,
)
from agent_harness.application.registry import ToolRegistry
from agent_harness.domain.loop import Budget, Message, ModelTurn, Role, StopReason, Usage
from agent_harness.domain.tools import PermissionLevel, ToolCall, ToolResult, ToolSpec


class FakeSummarizer:
    def __init__(self, text: str = "short", usage: Usage | None = None) -> None:
        self.text = text
        self.usage = usage or Usage()
        self.seen: list[tuple[Message, ...]] = []

    def summarize(self, messages: Sequence[Message]) -> ModelTurn:
        self.seen.append(tuple(messages))
        return ModelTurn(text=self.text, usage=self.usage)


def words(n: int) -> str:
    return "palabra " * n


def convo(n: int) -> list[Message]:
    msgs = [Message(Role.USER, content="TASK")]
    for i in range(n):
        role = Role.ASSISTANT if i % 2 == 0 else Role.USER
        msgs.append(Message(role, content=words(40)))
    return msgs


def test_estimator_grows_with_content() -> None:
    assert estimate_tokens([]) == 0
    assert estimate_tokens(convo(4)) > estimate_tokens(convo(2))


def test_under_threshold_is_untouched() -> None:
    summ = FakeSummarizer()
    c = ContextCompactor(summ, threshold_tokens=10_000)
    out = c.compact(convo(6))
    assert not out.compacted and out.messages == tuple(convo(6)) and not summ.seen
    assert out.usage == Usage(cost_usd=0.0)


def test_over_threshold_keeps_task_summary_and_last_n() -> None:
    summ = FakeSummarizer("resumen", Usage(10, 5, 0.01))
    c = ContextCompactor(summ, threshold_tokens=100, keep_last=3)
    msgs = convo(10)
    out = c.compact(msgs)
    assert out.compacted and out.usage == Usage(10, 5, 0.01)
    assert out.messages[0] == msgs[0]
    assert out.messages[1] == Message(Role.USER, content=f"{SUMMARY_PREFIX} resumen")
    assert out.messages[2:] == tuple(msgs[-3:])
    assert summ.seen[0] == tuple(msgs[1:-3])


def test_cut_never_orphans_a_tool_result() -> None:
    call = ToolCall("c1", "add", {"a": 1, "b": 2})
    msgs = [
        *convo(6),
        Message(Role.ASSISTANT, tool_calls=(call,)),
        Message(Role.TOOL, tool_result=ToolResult("c1", "add", True, "3")),
        Message(Role.ASSISTANT, content=words(40)),
    ]
    summ = FakeSummarizer()
    # keep_last=2 would start at the TOOL message; the cut must move back to the call
    out = ContextCompactor(summ, threshold_tokens=50, keep_last=2).compact(msgs)
    assert out.compacted
    assert out.messages[2].tool_calls == (call,)
    assert out.messages[3].role is Role.TOOL


def test_too_short_to_cut_returns_unchanged() -> None:
    out = ContextCompactor(FakeSummarizer(), threshold_tokens=1, keep_last=5).compact(convo(3))
    assert not out.compacted
    assert out.usage == Usage(cost_usd=0.0)


def test_non_compacting_context_does_not_hide_known_model_cost() -> None:
    model = ScriptedModel([ModelTurn(text="done", usage=Usage(1, 1, 2.0))])
    compactor = ContextCompactor(FakeSummarizer(), threshold_tokens=10_000)

    run = AgentLoop(model, ToolRegistry(), Budget(max_cost_usd=1.0), compactor=compactor).run("x")

    assert run.stop_reason is StopReason.MAX_COST
    assert run.final_answer is None


def test_useless_summary_is_discarded_but_usage_counted() -> None:
    summ = FakeSummarizer(words(5_000), Usage(1, 1, 0.0))
    out = ContextCompactor(summ, threshold_tokens=100, keep_last=2).compact(convo(8))
    assert not out.compacted and out.usage == Usage(1, 1, 0.0)
    assert out.messages == tuple(convo(8))


def test_invalid_config() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ContextCompactor(FakeSummarizer(), threshold_tokens=0)
    with pytest.raises(ValueError, match="keep_last"):
        ContextCompactor(FakeSummarizer(), threshold_tokens=5, keep_last=0)


def test_model_summarizer_sends_one_prompt_without_tools() -> None:
    model = ScriptedModel([ModelTurn(text="resumen")])
    call = ToolCall("c1", "add", {"a": 1})
    msgs = [
        Message(Role.ASSISTANT, tool_calls=(call,)),
        Message(Role.TOOL, tool_result=ToolResult("c1", "add", True, "3")),
        Message(Role.USER, content="hola"),
    ]
    turn = ModelSummarizer(model).summarize(msgs)
    assert turn.text == "resumen"
    sent = model.calls[0]
    assert len(sent) == 1 and "call add" in sent[0].content and "result" in sent[0].content
    assert "user: hola" in sent[0].content


def test_loop_compacts_before_model_call_and_counts_summary_usage() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec("echo", "Echo.", {"type": "object", "properties": {}}, PermissionLevel.READ),
        lambda _args: words(200),
    )
    call = ModelTurn(tool_calls=(ToolCall("c", "echo", {}),), usage=Usage(1, 1, 0.0))
    main = ScriptedModel([call, call, call, ModelTurn(text="done", usage=Usage(1, 1, 0.0))])
    summ = FakeSummarizer("resumen", Usage(7, 3, 0.0))
    compactor = ContextCompactor(summ, threshold_tokens=300, keep_last=2)
    run = AgentLoop(main, registry, Budget(), compactor=compactor).run("go")
    assert run.stop_reason is StopReason.FINAL_ANSWER and run.final_answer == "done"
    assert summ.seen, "compaction should have triggered"
    last_prompt = main.calls[-1]
    assert any(m.content.startswith(SUMMARY_PREFIX) for m in last_prompt)
    assert len(last_prompt) < 8
    assert run.usage.input_tokens == 4 + 7 * len(summ.seen)
