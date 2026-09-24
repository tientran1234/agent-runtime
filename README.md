# agent-runtime

The part of an LLM agent that is not the model: a bounded tool loop, memory
that fits a token budget, a trace of what happened and what it cost, and a
fallback chain for when a provider is down. Provider-agnostic by construction —
the Anthropic adapter is one file on the official SDK, and a scripted fake
runs the whole thing offline in tests.

```ts
import { runAgent, defineTool, Tracer, ConsoleExporter } from "agent-runtime";
import { AnthropicProvider } from "agent-runtime/anthropic";
import { z } from "zod";

const getOrder = defineTool({
  name: "get_order",
  description: "Look up an order by id",
  input: z.object({ orderId: z.string() }),
  execute: async ({ orderId }) => db.orders.find(orderId),
  timeoutMs: 5_000,
});

const result = await runAgent({
  provider: new AnthropicProvider({ model: "claude-opus-5", effort: "high" }),
  system: "You are a support agent. Use tools; do not guess order details.",
  tools: [getOrder],
  input: "Where is order ord_42?",
  maxIterations: 8,
  tracer: new Tracer({ exporters: [new ConsoleExporter()] }),
});

result.status;   // "completed" | "max_iterations" | "refused" | "truncated" | "aborted"
result.text;     // the final answer
result.usage;    // tokens across every model call
```

## What the loop guarantees

- **It ends.** `maxIterations` caps model calls. A model that never stops
  calling tools gets `status: "max_iterations"`, not an infinite bill.
- **No tool runs on bad input.** Every tool input is validated against its Zod
  schema first. Invalid input becomes an *error result* the model can read and
  correct — not an exception, and not a side effect on garbage.
- **No tool runs after a refusal or a truncated turn.** A `refusal` can cut a
  `tool_use` off mid-input; a `max_tokens` stop can leave input that parses but
  is incomplete. Both end the run with a named status.
- **No tool hangs the agent, and no tool floods the context.** Per-tool
  timeout (default 30 s) and result cap (default 16 000 chars, with a marker
  saying how much was cut).
- **Parallel tool calls stay parallel.** All `tool_use` blocks in one turn run
  concurrently, and all their results go back in one user message. Splitting
  them across messages quietly trains the model to stop parallelising.
- **Every stop is a status, never an exception.** Provider errors still throw
  — the caller has to know the difference between "the agent decided to stop"
  and "the network died".

## Design decisions

**The model provider is a port, not the architecture.** `ModelProvider` has
one method, `complete()`, over neutral message and tool types. The loop,
memory, tracing and SSE never see a vendor type. `providers/anthropic.ts` is
the only file that imports `@anthropic-ai/sdk`; it streams by default, maps
`Message` to the neutral shape, and turns the SDK's typed errors into one
`ProviderError` with a `retryable` verdict. Adding another provider is one
adapter and nothing else.

**Memory trims in turns, never in messages.** A turn starts when a human
speaks and includes every tool call and result until the next human message.
`ConversationMemory` drops whole turns from the oldest end, so a `tool_use`
and its `tool_result` are never separated — which every provider rejects. An
optional `summarize` hook folds dropped turns into a leading summary message.

**Cost is computed, and "unknown" stays unknown.** Each model span records
usage and looks up the price for the model that actually answered (which
matters once fallback is involved). A model missing from the price table makes
the run's total `null`, not a quietly smaller number.

**Fallback keys on the provider's verdict, not on status codes.** Rate limits,
overload and network errors are retryable and move to the next provider; a
400 is thrown as-is, because it will be a 400 everywhere. `response.model`
says who answered.

**Adaptive thinking is on by default** for the Anthropic adapter, with
`effort` exposed for tuning. Pass `thinking: false` for models that do not
accept it.

**Prompt caching is opt-in, and breaks the prefix in two places.**
`new AnthropicProvider({ cache: true })` puts a `cache_control` breakpoint on
the last tool and a second one on the system prompt — tools render first, so an
edited system prompt still hits the cached tool list. It stays off by default
because a cache write costs 1.25× input: it pays for itself across a tool loop,
not on a single call. Cached tokens are priced apart from plain input and are
reported on each model span and in `Run.totals.usage`.

## Layout

```
src/
  types.ts           neutral messages, tool specs, ModelProvider port, ProviderError
  tools.ts           defineTool (Zod) · executeTool: validate → timeout → truncate → result
  loop.ts            runAgent — the bounded loop, events, statuses
  memory.ts          ConversationMemory: token budget, turn-wise trimming, summariser hook
  trace.ts           Tracer / runs / spans, usage + cost, Memory and Console exporters
  pricing.ts         per-model prices, costUsd()
  fallback.ts        FallbackProvider
  sse.ts             agentSSE(): the run as a text/event-stream Response
  client.ts          readAgentSSE(): the same stream back into typed events
  providers/
    anthropic.ts     the only file importing @anthropic-ai/sdk
    fake.ts          scripted provider + builders for tests
tests/               44 tests, no network, no API key
```

## SSE

```ts
// Next.js route handler, Hono, Bun, Deno — anything that returns a Response
export const POST = async (req: Request) =>
  agentSSE({ provider, tools, input: (await req.json()).prompt });
```

Events: `text_delta`, `model_call`, `tool_call`, `tool_result`, `done`.

On the other end, `agent-runtime/client` reads that stream back:

```ts
import { readAgentSSE } from "agent-runtime/client";

const result = await readAgentSSE(await fetch("/api/agent", { method: "POST", body }), {
  text_delta: (e) => setAnswer((a) => a + e.text),
  tool_call: (e) => setStatus(`running ${e.name}…`),
  error: (e) => setError(e.message),
});
```

It buffers until a frame is whole — a chunk can split anywhere, including
mid-JSON — resolves with the `done` result, and drops event types it does not
know so a newer server stays readable by an older UI. Without an `error`
handler a failed run rejects, because a stream that broke must not look like
one that finished. The module is type-only against the loop, so importing it
pulls no server code into a bundle.

## Run

```bash
pnpm install
pnpm test        # everything runs against FakeProvider
ANTHROPIC_API_KEY=… node -e '…'   # or `ant auth login`; the SDK picks either up
```

## What is deliberately not here

- **Server-side tools** (web search, code execution). They change the loop's
  stop conditions (`pause_turn`); wire them at the adapter level when needed.
- **Persistence of runs.** Traces go to an exporter; where they land is your
  call. Pair with a workflow engine for durable multi-step agents.
- **A planner or multi-agent orchestration.** This is the runtime one agent
  runs on. Orchestration is a layer above it.
