"""Tracing and replay, as wrappers: the loop itself is unchanged.

``TracedModel`` and ``TracedDispatcher`` record one span per call into a shared ``Tracer``.
Latency comes from an injectable clock (tests pass a fake). Cost is whatever the model
reported in its ``Usage``; nothing is estimated here.

Replay: ``ReplayModel`` hands back the recorded model turns in order and
``ReplayDispatcher`` hands back the recorded tool results by ``call_id``. Neither calls a
model or runs a tool, so a recorded run can be reproduced and inspected offline.
Replay is only as faithful as the trace: a different prompt or budget just replays the
same recorded turns, and running out of them raises (the loop reports MODEL_ERROR).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence

from agent_harness.application.agent_loop import ModelClient, ToolDispatcher
from agent_harness.domain.loop import Message, ModelTurn, Usage
from agent_harness.domain.tools import ToolCall, ToolResult, ToolSpec
from agent_harness.domain.trace import Span, SpanKind, span_from_dict, span_to_dict

Clock = Callable[[], float]


class Tracer:
    def __init__(self, clock: Clock = time.monotonic) -> None:
        self.clock = clock
        self.spans: list[Span] = []

    def total_usage(self) -> Usage:
        total = Usage(cost_usd=0.0)
        for span in self.spans:
            total = total + span.usage
        return total

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(span_to_dict(s), sort_keys=True) for s in self.spans)

    @staticmethod
    def spans_from_jsonl(text: str) -> list[Span]:
        return [span_from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]


class TracedModel:
    def __init__(self, inner: ModelClient, tracer: Tracer, name: str = "model") -> None:
        self._inner = inner
        self._tracer = tracer
        self._name = name

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        start = self._tracer.clock()
        try:
            turn = self._inner.complete(messages, tools)
        except Exception as exc:
            self._tracer.spans.append(
                Span(
                    SpanKind.MODEL,
                    self._name,
                    self._tracer.clock() - start,
                    error=type(exc).__name__,
                )
            )
            raise
        self._tracer.spans.append(
            Span(SpanKind.MODEL, self._name, self._tracer.clock() - start, turn=turn)
        )
        return turn


class TracedDispatcher:
    def __init__(self, inner: ToolDispatcher, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def dispatch(self, call: ToolCall) -> ToolResult:
        start = self._tracer.clock()
        result = self._inner.dispatch(call)
        self._tracer.spans.append(
            Span(
                SpanKind.TOOL,
                call.name,
                self._tracer.clock() - start,
                call=call,
                result=result,
                error=None if result.ok else (result.error_kind or "error").__str__(),
            )
        )
        return result


class TraceExhausted(RuntimeError):
    """Replay asked for a model turn or tool result the trace does not contain."""


class ReplayModel:
    def __init__(self, spans: Sequence[Span]) -> None:
        self._turns = [s.turn for s in spans if s.kind is SpanKind.MODEL and s.turn is not None]

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        if not self._turns:
            raise TraceExhausted("no recorded model turn left")
        turn = self._turns.pop(0)
        return turn


class ReplayDispatcher:
    def __init__(self, spans: Sequence[Span]) -> None:
        self._results = {
            s.result.call_id: s.result for s in spans if s.kind is SpanKind.TOOL and s.result
        }

    def dispatch(self, call: ToolCall) -> ToolResult:
        try:
            return self._results[call.call_id]
        except KeyError:
            raise TraceExhausted(f"no recorded result for call {call.call_id!r}") from None
