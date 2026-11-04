"""Pure domain types for tools: specs, permission levels and call results.

No I/O lives here. The application layer validates and dispatches; adapters
translate these types to provider-specific formats.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

JSONObject = dict[str, Any]
ToolHandler = Callable[[Mapping[str, Any]], Any]


class PermissionLevel(StrEnum):
    """How much damage a tool can do. Guardrail policies are keyed on this."""

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What the model sees: name, description and JSON Schema for arguments."""

    name: str
    description: str
    input_schema: JSONObject
    level: PermissionLevel = PermissionLevel.READ

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid tool name: {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"tool {self.name!r} needs a description")


@dataclass(frozen=True, slots=True)
class Tool:
    """A spec bound to the function that executes it."""

    spec: ToolSpec
    handler: ToolHandler


class ToolErrorKind(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    DENIED = "denied"
    BLOCKED_BY_HOOK = "blocked_by_hook"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True, slots=True)
class ArgumentIssue:
    """One validation problem, phrased so a model can fix its call."""

    path: str
    message: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to run a tool."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Outcome of a tool call, always returned to the model (never raised)."""

    call_id: str
    name: str
    ok: bool
    output: Any = None
    error_kind: ToolErrorKind | None = None
    error_message: str | None = None
    issues: tuple[ArgumentIssue, ...] = field(default_factory=tuple)

    @classmethod
    def success(cls, call: ToolCall, output: Any) -> ToolResult:
        return cls(call_id=call.call_id, name=call.name, ok=True, output=output)

    @classmethod
    def failure(
        cls,
        call: ToolCall,
        kind: ToolErrorKind,
        message: str,
        issues: tuple[ArgumentIssue, ...] = (),
    ) -> ToolResult:
        return cls(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            error_kind=kind,
            error_message=message,
            issues=issues,
        )

    def for_model(self) -> JSONObject:
        """Compact, model-readable representation."""
        if self.ok:
            return {"ok": True, "output": self.output}
        payload: JSONObject = {
            "ok": False,
            "error": self.error_kind.value if self.error_kind else "error",
            "message": self.error_message,
        }
        if self.issues:
            payload["issues"] = [{"path": i.path, "message": i.message} for i in self.issues]
        return payload
