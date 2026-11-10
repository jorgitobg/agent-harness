"""Cost estimation for models that report tokens but no price.

``CostEstimator`` wraps any ``ModelClient``. When a turn reports tokens and ``cost_usd == 0``
it fills in ``tokens * price``, so ``Budget.max_cost_usd`` can still stop a run. Prices are
supplied by the caller (USD per million tokens); none are built in, because rates change and
a stale table would be a quiet lie. A turn that already reports a cost is left untouched.
The result is an estimate, not an invoice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from agent_harness.application.agent_loop import ModelClient
from agent_harness.domain.loop import Message, ModelTurn
from agent_harness.domain.tools import ToolSpec


class CostEstimator:
    def __init__(
        self, model: ModelClient, *, input_per_mtok: float, output_per_mtok: float
    ) -> None:
        if input_per_mtok < 0 or output_per_mtok < 0:
            raise ValueError("prices must be >= 0")
        self._model = model
        self._in = input_per_mtok
        self._out = output_per_mtok
        self.estimated_turns = 0

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        turn = self._model.complete(messages, tools)
        usage = turn.usage
        if usage.cost_usd > 0 or usage.total_tokens == 0:
            return turn
        cost = (usage.input_tokens * self._in + usage.output_tokens * self._out) / 1_000_000
        self.estimated_turns += 1
        return replace(turn, usage=replace(usage, cost_usd=cost))
