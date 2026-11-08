"""Pure trace types: one span per model call or tool call, serialisable to plain JSON.

A trace holds everything needed to replay a run without a model: the model turns
(text, tool calls, usage) and the tool results, in order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agent_harness.domain.loop import ModelTurn, Usage
from agent_harness.domain.tools import ArgumentIssue, ToolCall, ToolErrorKind, ToolResult


class SpanKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Span:
    """One timed call. Exactly one of ``turn`` (model) or ``result`` (tool) is set."""

    kind: SpanKind
    name: str
    latency_s: float
    turn: ModelTurn | None = None
    call: ToolCall | None = None
    result: ToolResult | None = None
    error: str | None = None

    @property
    def usage(self) -> Usage:
        return self.turn.usage if self.turn is not None else Usage()


def _call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"call_id": call.call_id, "name": call.name, "arguments": dict(call.arguments)}


def _call_from_dict(d: Mapping[str, Any]) -> ToolCall:
    return ToolCall(str(d["call_id"]), str(d["name"]), dict(d["arguments"]))


def span_to_dict(span: Span) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": span.kind.value,
        "name": span.name,
        "latency_s": span.latency_s,
    }
    if span.error is not None:
        out["error"] = span.error
    if span.turn is not None:
        t = span.turn
        out["turn"] = {
            "text": t.text,
            "tool_calls": [_call_to_dict(c) for c in t.tool_calls],
            "usage": {
                "input_tokens": t.usage.input_tokens,
                "output_tokens": t.usage.output_tokens,
                "cost_usd": t.usage.cost_usd,
            },
        }
    if span.call is not None:
        out["call"] = _call_to_dict(span.call)
    if span.result is not None:
        r = span.result
        out["result"] = {
            "call_id": r.call_id,
            "name": r.name,
            "ok": r.ok,
            "output": r.output,
            "error_kind": r.error_kind.value if r.error_kind else None,
            "error_message": r.error_message,
            "issues": [{"path": i.path, "message": i.message} for i in r.issues],
        }
    return out


def span_from_dict(d: Mapping[str, Any]) -> Span:
    turn = None
    if "turn" in d:
        t = d["turn"]
        turn = ModelTurn(
            text=t["text"],
            tool_calls=tuple(_call_from_dict(c) for c in t["tool_calls"]),
            usage=Usage(**t["usage"]),
        )
    result = None
    if "result" in d:
        r = d["result"]
        result = ToolResult(
            call_id=r["call_id"],
            name=r["name"],
            ok=r["ok"],
            output=r["output"],
            error_kind=ToolErrorKind(r["error_kind"]) if r["error_kind"] else None,
            error_message=r["error_message"],
            issues=tuple(ArgumentIssue(i["path"], i["message"]) for i in r["issues"]),
        )
    return Span(
        kind=SpanKind(d["kind"]),
        name=d["name"],
        latency_s=float(d["latency_s"]),
        turn=turn,
        call=_call_from_dict(d["call"]) if "call" in d else None,
        result=result,
        error=d.get("error"),
    )
