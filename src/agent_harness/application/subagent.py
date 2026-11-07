"""Sub-agents: a tool that runs a child agent loop and returns only its final answer.

The child gets its own model, tool registry, transcript and budget. The parent's
transcript receives one tool result: the child's final answer (or a typed failure
the parent model can read). Nothing else crosses the boundary, so the parent's
context does not grow with the child's work.

Spend is not hidden: every child run is kept in ``SubAgent.runs`` so the caller can
account for its usage. Charging it against the parent budget is not done yet.
A child cannot spawn children unless its own registry is given a sub-agent tool.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_harness.application.agent_loop import AgentLoop
from agent_harness.domain.loop import AgentRun, StopReason
from agent_harness.domain.tools import PermissionLevel, ToolSpec

TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"task": {"type": "string", "minLength": 1}},
    "required": ["task"],
    "additionalProperties": False,
}


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
    ) -> None:
        """``child_factory`` builds a fresh loop per call, so no state leaks between runs."""
        self.spec = ToolSpec(name, description, TASK_SCHEMA, level)
        self._factory = child_factory
        self.runs: list[AgentRun] = []

    def handle(self, arguments: Mapping[str, Any]) -> str:
        run = self._factory().run(str(arguments["task"]))
        self.runs.append(run)
        if run.stop_reason is not StopReason.FINAL_ANSWER or run.final_answer is None:
            raise SubAgentFailed(f"sub-agent stopped: {run.stop_reason.value}")
        return run.final_answer
