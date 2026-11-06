"""US7 model router: routine steps go to a cheap model, hard steps to a strong one.

A ``ModelRouter`` is itself a ``ModelClient``, so the agent loop does not change. For each
call a classifier decides whether the step is *hard*; the matching model is tried first and,
if it raises, the other one is tried (automatic fallback). If both raise, the last error
propagates and the loop reports ``MODEL_ERROR``. Every decision is recorded in ``routes`` so
runs can be inspected and asserted on.

The default classifier is deliberately simple and rule-based (no model call): a step is hard
when the previous tool call failed, or when the transcript has grown past ``long_after``
messages. It is a heuristic, not a measurement of difficulty.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from agent_harness.application.agent_loop import ModelClient
from agent_harness.domain.loop import Message, ModelTurn, Role
from agent_harness.domain.tools import ToolSpec

Classifier = Callable[[Sequence[Message]], bool]


class Tier(StrEnum):
    CHEAP = "cheap"
    STRONG = "strong"


@dataclass(frozen=True, slots=True)
class Route:
    """One routing decision: the tier tried first, the tier that answered, and if it fell back."""

    wanted: Tier
    served_by: Tier
    fell_back: bool


def default_is_hard(messages: Sequence[Message], long_after: int = 8) -> bool:
    last = messages[-1] if messages else None
    if (
        last is not None
        and last.role is Role.TOOL
        and last.tool_result is not None
        and not last.tool_result.ok
    ):
        return True
    return len(messages) > long_after


class ModelRouter:
    def __init__(
        self,
        cheap: ModelClient,
        strong: ModelClient,
        is_hard: Classifier = default_is_hard,
    ) -> None:
        self._models = {Tier.CHEAP: cheap, Tier.STRONG: strong}
        self._is_hard = is_hard
        self.routes: list[Route] = []

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelTurn:
        wanted = Tier.STRONG if self._is_hard(messages) else Tier.CHEAP
        other = Tier.CHEAP if wanted is Tier.STRONG else Tier.STRONG
        try:
            turn = self._models[wanted].complete(messages, tools)
        except Exception:  # noqa: BLE001 - fall back to the other tier
            turn = self._models[other].complete(messages, tools)
            self.routes.append(Route(wanted, other, fell_back=True))
            return turn
        self.routes.append(Route(wanted, wanted, fell_back=False))
        return turn
