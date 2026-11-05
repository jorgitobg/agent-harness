# ADR 0003: OpenAI-compatible adapter on the standard library, HTTP behind a port

Status: accepted (2026-11-05)

The adapter targets the chat-completions format because hosted services, OpenRouter, llama.cpp and Ollama all speak it, so one adapter covers cloud and local models. It uses `urllib` rather than an HTTP or vendor SDK, which keeps the dependency list at `jsonschema` and keeps the core install small. The HTTP call sits behind a one-method `HttpPost` port, so tests replay recorded responses without patching the network.

Contract tests replay a conversation recorded from a real service and assert the adapter's requests equal the recorded ones. That checks the wire format against reality, which a hand-written mock cannot.

Failures never include bodies or credentials: a status code or a short reason only. Any failure becomes a typed `MODEL_ERROR` stop in the loop. Reasoning models can spend the whole token budget on hidden thinking and return no text; this is reported explicitly rather than as an empty answer, and the default budget is generous.

Trade-offs: synchronous only, no streaming, no retry or fallback inside the adapter (that belongs to the US7 router). Tool-call support is assumed; a model without it will simply answer in text.
