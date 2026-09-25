# Changelog

## 2026-09-25

- Opt-in server-side refusal fallbacks in `AnthropicProvider` (`serverFallbacks: true` → beta `server-side-fallback-2026-07-01`, `fallbacks: "default"`), so a declined request is retried on Anthropic's substitute for that refusal category instead of ending the run.

## 2026-09-24

- `readAgentSSE(response, handlers)` — a small client-side parser for the events `agentSSE` emits, so every UI does not reimplement frame splitting. Exported from `agent-runtime/client`.
