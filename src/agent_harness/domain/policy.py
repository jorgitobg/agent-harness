"""Permission policy: maps tool levels and tool names to allow / ask / deny.

Pure rules, no I/O. The application layer asks the policy for a decision and
decides how to ask a human when the answer is ``ASK``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from agent_harness.domain.tools import PermissionLevel, ToolSpec


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


DEFAULT_LEVEL_DECISIONS: Mapping[PermissionLevel, Decision] = {
    PermissionLevel.READ: Decision.ALLOW,
    PermissionLevel.WRITE: Decision.ASK,
    PermissionLevel.DESTRUCTIVE: Decision.ASK,
}


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    """Resolution order: deny list, allow list, then the decision for the tool's level.

    Name lists override levels. A name listed in both lists is a configuration
    error, because silently picking one would hide a mistake in the operator's intent.
    """

    level_decisions: Mapping[PermissionLevel, Decision] = field(
        default_factory=lambda: dict(DEFAULT_LEVEL_DECISIONS)
    )
    allow: frozenset[str] = frozenset()
    deny: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        overlap = self.allow & self.deny
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"tools listed in both allow and deny: {names}")
        missing = [level for level in PermissionLevel if level not in self.level_decisions]
        if missing:
            raise ValueError(f"policy has no decision for levels: {', '.join(missing)}")

    def decide(self, spec: ToolSpec) -> Decision:
        if spec.name in self.deny:
            return Decision.DENY
        if spec.name in self.allow:
            return Decision.ALLOW
        return self.level_decisions[spec.level]
