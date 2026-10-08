# Roadmap

Backlog for this library. One item per pull request. Items are ordered; take the first unchecked one unless it says `blocked`.

- [x] Prompt caching in `AnthropicProvider`: `cache_control` on the system prompt and the tool list; surface cache read/write tokens on the model span and in `Run.totals`.
- [x] `readAgentSSE(response, handlers)` — a small client-side parser for the events `agentSSE` emits, so every UI does not reimplement frame splitting. Export from `agent-runtime/client`.
- [x] Opt-in server-side refusal fallbacks in `AnthropicProvider` (`serverFallbacks: true` → beta `server-side-fallback-2026-07-01`, `fallbacks: "default"`), with a test on the request shape.
- [x] `strict: true` passthrough on tool specs (schema must carry `additionalProperties: false` + `required`); `defineTool({ strict: true })`.
- [x] OpenAI-compatible provider (`providers/openai-compatible.ts`, `fetch`-based, covers Ollama and OpenAI endpoints) implementing `ModelProvider`, with mapping tests like the Anthropic ones.
- [x] Per-tool concurrency limit and a `beforeToolCall` hook for human approval gates.
- [x] Prompt regression suite: scripted `FakeProvider` scenarios that assert the loop's transcript shape for a given system prompt, runnable in CI.
- [x] Server-side tools (`AnthropicProvider({ serverTools })`) and the `pause_turn` stop condition: a paused turn resumes with its provider blocks carried back verbatim, and no client tool runs inside one.
- [ ] `blocked` Publish as `@tientran1234/agent-runtime` on npm (needs the owner's npm login).

## Batch 2 — set by the owner, 30 Sep 2026

Same rule: one item per change, in order.

- [x] Structured final output: `runAgent({ output: zodSchema })` uses structured outputs (`output_config.format`) on the Anthropic adapter, validates the final message, retries once with the validation error fed back; `result.output` is typed.
- [x] Budget guard: `maxCostUsd` / `maxInputTokens` per run — stop with status `budget_exceeded` before the call that would exceed, using the tracer's running totals; tested against the fake provider's scripted usage.
- [x] Resumable runs: serialize loop state (transcript, iteration, pending approval) so a run paused by `beforeToolCall → "ask"` resumes in another process; an example hosting a run inside a durable-workflow step.
- [x] Sub-agent handoff: `handoffTool({ name, description, ...AgentOptions })` — a tool that runs a nested agent with its own provider and tools and returns its final text; nested spans under the parent trace with cost rolled up.
- [x] Context editing: opt-in `contextEditing: { clearToolUsesAfter: N }` — on the Anthropic adapter via `context_management`, and an equivalent in-memory strategy for other providers; test that old tool results disappear from the request while the transcript stays intact.
- [x] OpenTelemetry exporter: `OtelExporter({ endpoint })` mapping runs and spans to OTLP/HTTP JSON so traces land in Tempo or Jaeger; span attributes for model, tokens, cost, tool name.
- [x] A store for a suspended run: `RunStore` (`put`/`get`/`delete`/`pending`) with `MemoryStore` and a crash-safe `FileStore` behind it, so resuming in another process does not start with every caller writing the same adapter.
- [x] A SQL-backed `RunStore`: `SqlStore({ query, table, dialect })` over a query function the caller supplies, so a suspended run can wait in a row without this library picking a driver; held to the same contract tests as the other two.
