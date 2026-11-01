# ADR 0001 — Hexagonal core, no agent framework

Status: accepted · Date: 2026-11-01

## Context

The purpose of the repository is to show how an agent harness works inside. Agent
frameworks (LangChain, LlamaIndex, CrewAI...) hide exactly the parts we want to show:
the loop, budgets, permission checks and context handling.

## Decision

- Plain Python with a small set of dependencies: `pydantic` for boundary models and
  `jsonschema` for tool argument validation.
- Hexagonal layout:
  - `domain/` — pure types and rules (tool specs, permission levels, results). No I/O.
  - `application/` — use cases (registry, guardrails, loop) that depend only on ports.
  - `adapters/` — model clients, confirmation prompts, tracing exporters, MCP.
- Dependencies point inward only.

## Consequences

- More code than gluing a framework together, but each concept fits in one file and
  can be tested in isolation with fakes.
- Swapping a model provider or a confirmation UI means writing one adapter.
- We give up framework integrations; MCP support (US7/US8 timeframe) covers the
  interoperability that matters.
