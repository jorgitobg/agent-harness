"""Context compaction (US4): keep the task and the latest turns, summarise the middle.

Policy, in order:
1. If the estimated transcript size is within ``threshold_tokens``, do nothing.
2. Keep the first message verbatim (the user's task) and the last ``keep_last`` messages.
3. Never cut between an assistant message and its tool results: if the first kept message
   is a tool result, the cut moves back so the originating assistant message is kept too.
4. Replace everything in between with one summary message produced through a
   ``Summarizer`` port. If the summary would not make the transcript smaller, the
   transcript is returned unchanged.

The summariser's own usage is returned so the loop can count it against the budget.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from agent_harness.application.agent_loop import ModelClient
from agent_harness.domain.loop import Message, ModelTurn, Role, Usage

TokenEstimator = Callable[[Sequence[Message]], int]

SUMMARY_PREFIX = "[summary of earlier turns]"


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Rough, dependency-free estimate: about 4 characters per token."""
    chars = 0
    for m in messages:
        chars += len(m.content)
        for c in m.tool_calls:
            chars += len(c.name) + len(str(c.arguments))
        if m.tool_result is not None:
            chars += len(str(m.tool_result))
    return chars // 4 + len(messages)


class Summarizer(Protocol):
    """Port: condense older messages into one text; returns a turn so usage is visible."""

    def summarize(self, messages: Sequence[Message]) -> ModelTurn: ...


class ModelSummarizer:
    """Summarizer backed by any ``ModelClient`` (no tools offered)."""

    def __init__(self, model: ModelClient) -> None:
        self._model = model

    def summarize(self, messages: Sequence[Message]) -> ModelTurn:
        lines = [f"{m.role.value}: {m.content or _describe(m)}" for m in messages]
        ask = Message(
            Role.USER,
            content="Summarise these earlier turns in a few sentences. Keep facts, "
            "decisions and tool outputs that later steps need.\n" + "\n".join(lines),
        )
        return self._model.complete((ask,), ())


def _describe(m: Message) -> str:
    if m.tool_calls:
        return "; ".join(f"call {c.name}({c.arguments})" for c in m.tool_calls)
    if m.tool_result is not None:
        return f"result {m.tool_result}"
    return ""


@dataclass(frozen=True, slots=True)
class Compaction:
    messages: tuple[Message, ...]
    usage: Usage
    compacted: bool


class ContextCompactor:
    def __init__(
        self,
        summarizer: Summarizer,
        *,
        threshold_tokens: int,
        keep_last: int = 6,
        estimator: TokenEstimator = estimate_tokens,
    ) -> None:
        if threshold_tokens < 1:
            raise ValueError("threshold_tokens must be >= 1")
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self._summarizer = summarizer
        self._threshold = threshold_tokens
        self._keep_last = keep_last
        self._estimate = estimator

    def compact(self, messages: Sequence[Message]) -> Compaction:
        original = tuple(messages)
        if self._estimate(original) <= self._threshold:
            return Compaction(original, Usage(), False)
        cut = len(original) - self._keep_last
        while cut > 1 and original[cut].role is Role.TOOL:
            cut -= 1
        if cut <= 1:
            return Compaction(original, Usage(), False)
        turn = self._summarizer.summarize(original[1:cut])
        summary = Message(Role.USER, content=f"{SUMMARY_PREFIX} {turn.text}")
        result = (original[0], summary, *original[cut:])
        if self._estimate(result) >= self._estimate(original):
            return Compaction(original, turn.usage, False)
        return Compaction(result, turn.usage, True)
