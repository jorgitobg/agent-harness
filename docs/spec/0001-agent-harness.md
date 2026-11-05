# Spec 0001 — agent-harness

Status: accepted · Date: 2026-11-01

## Problem

Coding and ops agents (Claude Code, Codex, Hermes, OpenClaw...) share the same machinery:
a tool registry, a model loop with budgets, context management, memory, sub-agents,
hooks and guardrails. Most write-ups describe it from the outside. This repository
builds that machinery from scratch, small enough to read in one sitting, so every
design decision is visible and testable.

## Goals

- A readable reference harness in plain Python (no agent framework).
- Provider-agnostic: any model behind a small `ModelClient` port, including a local
  OpenAI-compatible server (llama.cpp, Ollama).
- Safe by default: typed tools, permission levels, human confirmation for destructive actions.
- Observable: every model call and tool call is traceable and replayable.
- Measured: an eval suite compares models on task success, steps and cost.

## User stories and acceptance criteria

### US1 — Typed tool registry
As an agent author I register tools with a JSON Schema so the model can only call them with valid arguments.
- Given a tool registered with a schema, when it is called with valid arguments, then the handler runs and its output is returned.
- Given invalid arguments, then the handler is not run and a structured validation error is returned to the model.
- Given an unknown tool name, then a structured "unknown tool" error is returned.
- Given two tools with the same name, then registration fails.

### US2 — Permission levels and guardrails
As an operator I decide what an agent may do without asking.
- Every tool declares a level: `read`, `write` or `destructive`.
- A policy maps each level to `allow`, `ask` or `deny`; the default policy is read=allow, write=ask, destructive=ask.
- `ask` delegates to a confirmation port (CLI prompt, test double); a refusal returns a structured "denied" result to the model.
- Allow/deny lists by tool name override levels.

### US3 — Agent loop with budgets
- The loop alternates model call → tool calls → model call until the model returns a final answer.
- Hard budgets: max steps, max total tokens, max cost, wall-clock timeout. Hitting one stops the loop with a typed reason.

### US4 — Context management
- When the transcript exceeds a token threshold, older turns are compacted into a summary while the system prompt and the latest turns are kept verbatim.

### US5 — Hooks
- Pre-tool hooks can block or rewrite a call; post-tool hooks can observe or redact results.

### US6 — Sub-agents
- A tool can spawn a sub-agent with its own isolated transcript and budget; only its final answer returns to the parent.

### US7a — OpenAI-compatible model adapter (done)
- Given a chat-completions endpoint with tool calling, when the loop calls the model, then the transcript and tool specs are sent in the OpenAI format and tool calls come back as `ToolCall`s with parsed arguments.
- Given malformed JSON, a tool call without a name, non-object arguments, an empty reply or an exhausted token budget, then the call fails with a typed `model_error` stop and a message that never contains bodies or credentials.
- Given a non-https endpoint that is not localhost, then it is refused. Responses are size-capped.
- Given a recorded real conversation, then replaying it reproduces the recorded requests exactly.

### US7 — Model router
- Routine steps go to a cheap/local model, hard steps to a stronger one, with automatic fallback on error.

### US8 — Tracing, replay and evals
- Each model and tool call emits a span (model, tokens, latency, cost).
- A run can be replayed from its trace without calling any model.
- `evals/` runs a fixed task set on two models and writes results by script.

## Non-functional requirements

- Python 3.12+, strict typing (pyright strict), ruff, pytest. Coverage ≥ 85 % on core packages.
- Tests never call real models: a scripted fake model client drives the loop.
- No network access in the core; adapters live at the edge.

## Out of scope (for now)

- A UI beyond a terminal replay viewer.
- Multi-user serving, auth, persistence beyond local files.
- Fine-tuning or training.

## Delivery order

US1 → US2 → US3 → US4 → US5 → US6 → US7 → US8. Each story ships as its own PR with tests.
