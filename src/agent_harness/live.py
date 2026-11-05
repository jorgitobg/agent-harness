"""Run the demo agent against a real OpenAI-compatible endpoint.

    AGENT_HARNESS_API_KEY=... python -m agent_harness.live --base-url URL --model NAME "prompt"

The key is read from the environment only, never from an argument. Uses the same loop,
registry and guardrails as the offline demo; writes still ask for confirmation.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from agent_harness.adapters.confirmation import ScriptedConfirmation
from agent_harness.adapters.openai_compat import OpenAICompatConfig, OpenAICompatModel
from agent_harness.application.agent_loop import AgentLoop
from agent_harness.application.guardrails import GuardedRegistry
from agent_harness.demo import build_registry
from agent_harness.domain.loop import Budget
from agent_harness.domain.policy import PermissionPolicy

SYSTEM = "You can search and write notes with the tools provided. Be brief."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-harness-live", description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args(argv)
    config = OpenAICompatConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get("AGENT_HARNESS_API_KEY") or None,
        max_tokens=args.max_tokens,
        system_prompt=SYSTEM,
    )
    registry = build_registry()
    # No operator is attached, so any write is refused: this example only reads.
    guarded = GuardedRegistry(registry, PermissionPolicy(), ScriptedConfirmation([]))
    loop = AgentLoop(
        OpenAICompatModel(config),
        registry,
        Budget(max_steps=args.max_steps, timeout_s=120.0),
        dispatcher=guarded,
    )
    run = loop.run(args.prompt)
    print(f"stop_reason={run.stop_reason.value} steps={run.steps} tokens={run.usage.total_tokens}")
    if run.final_answer:
        print(run.final_answer)
    return 0 if run.final_answer is not None else 1


if __name__ == "__main__":
    sys.exit(main())
