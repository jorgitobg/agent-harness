"""Pure types for the agent loop: budgets, usage, transcript messages and stop reasons.

No I/O here. The application layer (``agent_harness.application.agent_loop``) drives
the loop; adapters provide the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from agent_harness.domain.tools import ToolCall, ToolResult


class StopReason(StrEnum):
    """Why a run ended. Every run ends with exactly one of these."""

    FINAL_ANSWER = "final_answer"
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Budget:
    """Hard limits for one run. Hitting any of them stops the loop with a typed reason."""

    max_steps: int = 10
    max_tokens: int = 100_000
    max_cost_usd: float = 1.0
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not math.isfinite(self.max_cost_usd) or self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be finite and >= 0")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and > 0")


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts and cost; ``None`` means the cost is unknown, not free."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.input_tokens) is not int
            or self.input_tokens < 0
            or type(self.output_tokens) is not int
            or self.output_tokens < 0
        ):
            raise ValueError("token counts must be non-negative integers")
        if self.cost_usd is not None and (
            type(self.cost_usd) not in (int, float)
            or not math.isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be finite and >= 0, or None")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        cost = (
            self.cost_usd + other.cost_usd
            if self.cost_usd is not None and other.cost_usd is not None
            else None
        )
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """What a model returns for one call: text, zero or more tool calls, and its usage."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True)
class Message:
    """One transcript entry. Assistant messages carry tool calls; tool messages carry a result."""

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Outcome of one run. ``final_answer`` is set only when the stop reason is FINAL_ANSWER."""

    stop_reason: StopReason
    steps: int
    usage: Usage
    transcript: tuple[Message, ...]
    final_answer: str | None = None
