"""Guarded registry: applies a permission policy and a confirmation port around dispatch.

Order of checks for one call:
1. unknown tool        -> structured error (no policy lookup)
2. invalid arguments   -> structured error (never ask a human about a malformed call)
3. policy DENY         -> structured "denied" result, handler not run
4. policy ASK          -> confirmation port; refusal is a "denied" result
5. policy ALLOW / approved -> registry dispatch

Confirmation failures fail closed: an exception from the port counts as a refusal.
"""

from __future__ import annotations

from typing import Protocol

from agent_harness.application.registry import ToolRegistry
from agent_harness.domain.policy import Decision, PermissionPolicy
from agent_harness.domain.tools import ToolCall, ToolErrorKind, ToolResult, ToolSpec


class ConfirmationPort(Protocol):
    """Adapter that asks a human (CLI prompt, chat message, test double) to approve a call."""

    def confirm(self, call: ToolCall, spec: ToolSpec) -> bool: ...


class GuardedRegistry:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        confirmation: ConfirmationPort,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._confirmation = confirmation

    def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._registry.get(call.name)
        if tool is None:
            return self._registry.dispatch(call)

        issues = self._registry.validate(call.name, call.arguments)
        if issues:
            return ToolResult.failure(
                call,
                ToolErrorKind.INVALID_ARGUMENTS,
                f"arguments do not match the schema of {call.name!r}",
                issues,
            )

        decision = self._policy.decide(tool.spec)
        if decision is Decision.DENY:
            return ToolResult.failure(
                call, ToolErrorKind.DENIED, f"policy denies tool {call.name!r}"
            )
        if decision is Decision.ASK and not self._approved(call, tool.spec):
            return ToolResult.failure(
                call, ToolErrorKind.DENIED, f"operator refused tool {call.name!r}"
            )
        return self._registry.dispatch(call)

    def _approved(self, call: ToolCall, spec: ToolSpec) -> bool:
        try:
            return self._confirmation.confirm(call, spec)
        except Exception:  # noqa: BLE001 - fail closed: a broken prompt is a refusal
            return False
