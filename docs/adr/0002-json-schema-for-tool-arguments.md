# ADR 0002 — JSON Schema for tool arguments

Status: accepted · Date: 2026-11-01

## Context

Models emit tool arguments as JSON. Every major provider (Anthropic, OpenAI, Gemini,
OpenAI-compatible local servers) and MCP describe tool inputs with JSON Schema.
We need validation before a handler runs, and errors the model can act on.

## Options

1. Pydantic models per tool, schema generated from the model.
2. Raw JSON Schema per tool, validated with `jsonschema`.
3. No validation; handlers check their own inputs.

## Decision

Option 2 as the registry contract, Draft 2020-12. A Pydantic helper can generate the
schema for authors who prefer typed models, but the registry only depends on the schema.

## Consequences

- The same schema is sent to the model and used for validation: one source of truth.
- MCP tools plug in without conversion, since MCP already ships JSON Schema.
- Validation errors are returned as structured, model-readable messages
  (path + reason), so the model can retry with corrected arguments.
- Schemas are checked when a tool is registered, so a broken schema fails at startup,
  not on the first call.
