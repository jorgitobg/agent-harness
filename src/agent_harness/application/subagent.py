"""Sub-agents: a tool that runs a child agent loop and returns only its final answer.

The child gets its own model, tool registry, transcript and budget. The parent's
transcript receives one tool result: the child's final answer (or a typed failure
the parent model can read). Nothing else crosses the boundary, so the parent's
context does not grow with the child's work.

Spend is not hidden: every child run is kept in ``SubAgent.runs``, and when a shared
``UsageLedger`` is given, the child's usage (also of failed runs) is charged to it. A parent
``AgentLoop`` built with the same ledger adds it to its own usage after each tool round,
so child spend counts toward the parent's token and cost budget.
A child cannot spawn children unless its own registry is given a sub-agent tool.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_harness.application.agent_loop import AgentLoop
from agent_harness.domain.loop import AgentRun, StopReason, Usage
from agent_harness.domain.tools import PermissionLevel, ToolSpec

TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"task": {"type": "string", "minLength": 1}},
    "required": ["task"],
    "additionalProperties": False,
}


class UsageLedger:
    """Collects usage that happened outside the parent's own model calls."""

    def __init__(self) -> None:
        self._pending = Usage(cost_usd=0.0)
        self.total = Usage(cost_usd=0.0)

    def charge(self, usage: Usage) -> None:
        self._pending = self._pending + usage
        self.total = self.total + usage

    def drain(self) -> Usage:
        """Return usage charged since the last drain and reset it (``total`` keeps counting)."""
        pending, self._pending = self._pending, Usage(cost_usd=0.0)
        return pending


class SubAgentFailed(RuntimeError):
    """The child stopped without a final answer. Surfaces to the parent as a failed tool call."""


class SubAgent:
    def __init__(
        self,
        name: str,
        description: str,
        child_factory: Callable[[], AgentLoop],
        *,
        level: PermissionLevel = PermissionLevel.READ,
        ledger: UsageLedger | None = None,
    ) -> None:
        """``child_factory`` builds a fresh loop per call, so no state leaks between runs."""
        self.spec = ToolSpec(name, description, TASK_SCHEMA, level)
        self._factory = child_factory
        self._ledger = ledger
        self.runs: list[AgentRun] = []

    def handle(self, arguments: Mapping[str, Any]) -> str:
        run = self._factory().run(str(arguments["task"]))
        self.runs.append(run)
        if self._ledger is not None:
            self._ledger.charge(run.usage)
        if run.stop_reason is not StopReason.FINAL_ANSWER or run.final_answer is None:
            raise SubAgentFailed(f"sub-agent stopped: {run.stop_reason.value}")
        return run.final_answer
