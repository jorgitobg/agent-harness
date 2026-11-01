"""Tool registry: registers tools, validates arguments, dispatches calls.

Guardrails (permission policy, confirmation) wrap the registry in the
application layer; the registry itself only guarantees that a handler never
runs with arguments that violate its JSON Schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from agent_harness.domain.tools import (
    ArgumentIssue,
    Tool,
    ToolCall,
    ToolErrorKind,
    ToolHandler,
    ToolResult,
    ToolSpec,
)


class ToolRegistrationError(ValueError):
    """Raised at startup when a tool cannot be registered."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Draft202012Validator] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._tools:
            raise ToolRegistrationError(f"tool already registered: {spec.name}")
        try:
            Draft202012Validator.check_schema(spec.input_schema)
        except SchemaError as exc:
            raise ToolRegistrationError(
                f"invalid input schema for {spec.name}: {exc.message}"
            ) from exc
        if spec.input_schema.get("type") != "object":
            raise ToolRegistrationError(f"input schema for {spec.name} must describe an object")
        self._tools[spec.name] = Tool(spec=spec, handler=handler)
        self._validators[spec.name] = Draft202012Validator(spec.input_schema)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        """Specs in registration order, ready to be sent to a model."""
        return [tool.spec for tool in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def validate(self, name: str, arguments: Mapping[str, Any]) -> tuple[ArgumentIssue, ...]:
        validator = self._validators[name]
        found: list[ValidationError] = list(validator.iter_errors(dict(arguments)))  # pyright: ignore[reportUnknownMemberType]
        errors = sorted(found, key=lambda e: [str(p) for p in e.absolute_path])
        return tuple(
            ArgumentIssue(path=_json_path(e.absolute_path), message=e.message) for e in errors
        )

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Validate and run one call. Never raises for model mistakes."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.failure(
                call,
                ToolErrorKind.UNKNOWN_TOOL,
                f"unknown tool {call.name!r}; available: {', '.join(self._tools) or 'none'}",
            )
        issues = self.validate(call.name, call.arguments)
        if issues:
            return ToolResult.failure(
                call,
                ToolErrorKind.INVALID_ARGUMENTS,
                f"arguments do not match the schema of {call.name!r}",
                issues,
            )
        try:
            output = tool.handler(call.arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as data
            return ToolResult.failure(
                call, ToolErrorKind.EXECUTION_FAILED, f"{type(exc).__name__}: {exc}"
            )
        return ToolResult.success(call, output)


def _json_path(parts: Iterable[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path
