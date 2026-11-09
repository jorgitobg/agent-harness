"""Agent loop: model call -> tool calls -> model call, until a final answer or a budget stop.

Sub-agent spend (optional ``ledger``) is added after each tool round and checked against
the same token and cost limits.

Context compaction (optional) runs before each model call; its summariser usage counts
toward the budget.

Budget checks, in order, at the top of each iteration: steps, then timeout.
After each model call: timeout, total tokens, cost. A breached budget wins over a
final answer produced by the same turn, so a run never exceeds its limits silently.

A model that raises is a typed ``MODEL_ERROR`` stop, not a crash.
The timeout is cooperative: it cannot interrupt a model call that is already blocking.
Adapters should set their own request timeouts as well.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

from agent_harness.application.registry import ToolRegistry
from agent_harness.domain.loop import AgentRun, Budget, Message, ModelTurn, Role, StopReason, Usage
from agent_harness.domain.tools import ToolCall, ToolResult, ToolSpec

if TYPE_CHECKING:
    from agent_harness.application.context import ContextCompactor
    from agent_harness.application.subagent import UsageLedger


class ModelClient(Protocol):
    """Port for any model: receives the transcript and the available tools, returns one turn."""

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn: ...


class ToolDispatcher(Protocol):
    """Anything that runs one tool call and returns a result (never raises for model mistakes)."""

    def dispatch(self, call: ToolCall) -> ToolResult: ...


Clock = Callable[[], float]


class AgentLoop:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        budget: Budget | None = None,
        *,
        dispatcher: ToolDispatcher | None = None,
        clock: Clock = time.monotonic,
        compactor: ContextCompactor | None = None,
        ledger: UsageLedger | None = None,
    ) -> None:
        self._ledger = ledger
        self._compactor = compactor
        self._model = model
        self._tools = tools
        self._dispatcher: ToolDispatcher = dispatcher if dispatcher is not None else tools
        self._budget = budget or Budget()
        self._clock = clock

    def run(self, prompt: str) -> AgentRun:
        started = self._clock()
        transcript: list[Message] = [Message(Role.USER, content=prompt)]
        usage = Usage()
        steps = 0
        specs = tuple(self._tools.specs())

        def elapsed() -> float:
            return self._clock() - started

        def stop(reason: StopReason, final: str | None = None) -> AgentRun:
            return AgentRun(
                stop_reason=reason,
                steps=steps,
                usage=usage,
                transcript=tuple(transcript),
                final_answer=final,
            )

        while True:
            if steps >= self._budget.max_steps:
                return stop(StopReason.MAX_STEPS)
            if elapsed() >= self._budget.timeout_s:
                return stop(StopReason.TIMEOUT)

            if self._compactor is not None:
                compaction = self._compactor.compact(transcript)
                usage = usage + compaction.usage
                if compaction.compacted:
                    transcript[:] = compaction.messages

            try:
                turn = self._model.complete(tuple(transcript), specs)
            except Exception:  # noqa: BLE001 - a model failure is a typed stop, not a crash
                return stop(StopReason.MODEL_ERROR)

            steps += 1
            usage = usage + turn.usage
            transcript.append(
                Message(Role.ASSISTANT, content=turn.text, tool_calls=turn.tool_calls)
            )

            breach = self._breach(usage, elapsed())
            if breach is not None:
                return stop(breach)

            if not turn.tool_calls:
                return stop(StopReason.FINAL_ANSWER, final=turn.text)

            for call in turn.tool_calls:
                transcript.append(Message(Role.TOOL, tool_result=self._dispatcher.dispatch(call)))

            if self._ledger is not None:
                usage = usage + self._ledger.drain()
                breach = self._breach(usage, elapsed())
                if breach is not None:
                    return stop(breach)

    def _breach(self, usage: Usage, elapsed: float) -> StopReason | None:
        if elapsed >= self._budget.timeout_s:
            return StopReason.TIMEOUT
        if usage.total_tokens > self._budget.max_tokens:
            return StopReason.MAX_TOKENS
        if usage.cost_usd > self._budget.max_cost_usd:
            return StopReason.MAX_COST
        return None
