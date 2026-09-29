# Changelog

## 2026-09-29

- Server-side tools (`AnthropicProvider({ serverTools })`) and the `pause_turn` stop condition: a paused turn resumes with its provider blocks carried back verbatim and no client tool run inside it, so a turn of web search or code execution finishes instead of being read as a finished answer.
- Prompt regression suite: scripted `FakeProvider` scenarios that assert the loop's transcript shape for a given system prompt, runnable in CI, with the prompt and the tool list checked on every model call so a shape that matches for a prompt the model never saw cannot pass.

## 2026-09-28

- Per-tool concurrency limit and a `beforeToolCall` hook for human approval gates, so a call is judged against the input `execute` would really get and a capped tool queues its own calls without holding the turn or its own timeout.

## 2026-09-27

- OpenAI-compatible provider (`providers/openai-compatible.ts`, `fetch`-based, covers Ollama and OpenAI endpoints) implementing `ModelProvider`, exported as `agent-runtime/openai-compatible`, so the loop runs against a local model without a second SDK.

## 2026-09-26

- `strict: true` passthrough on tool specs (schema must carry `additionalProperties: false` + `required`); `defineTool({ strict: true })`, with an unenforceable schema rejected where the tool is defined rather than by the provider.

## 2026-09-25

- Opt-in server-side refusal fallbacks in `AnthropicProvider` (`serverFallbacks: true` → beta `server-side-fallback-2026-07-01`, `fallbacks: "default"`), so a declined request is retried on Anthropic's substitute for that refusal category instead of ending the run.

## 2026-09-24

- `readAgentSSE(response, handlers)` — a small client-side parser for the events `agentSSE` emits, so every UI does not reimplement frame splitting. Exported from `agent-runtime/client`.
