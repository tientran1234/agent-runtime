# Roadmap

Backlog for this library. One item per pull request. Items are ordered; take the first unchecked one unless it says `blocked`.

- [ ] Prompt caching in `AnthropicProvider`: `cache_control` on the system prompt and the tool list; surface cache read/write tokens on the model span and in `Run.totals`.
- [ ] `readAgentSSE(response, handlers)` — a small client-side parser for the events `agentSSE` emits, so every UI does not reimplement frame splitting. Export from `agent-runtime/client`.
- [ ] Opt-in server-side refusal fallbacks in `AnthropicProvider` (`serverFallbacks: true` → beta `server-side-fallback-2026-07-01`, `fallbacks: "default"`), with a test on the request shape.
- [ ] `strict: true` passthrough on tool specs (schema must carry `additionalProperties: false` + `required`); `defineTool({ strict: true })`.
- [ ] OpenAI-compatible provider (`providers/openai-compatible.ts`, `fetch`-based, covers Ollama and OpenAI endpoints) implementing `ModelProvider`, with mapping tests like the Anthropic ones.
- [ ] Per-tool concurrency limit and a `beforeToolCall` hook for human approval gates.
- [ ] Prompt regression suite: scripted `FakeProvider` scenarios that assert the loop's transcript shape for a given system prompt, runnable in CI.
- [ ] `blocked` Publish as `@tientran1234/agent-runtime` on npm (needs the owner's npm login).
