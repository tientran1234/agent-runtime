# agent-runtime

The part of an LLM agent that is not the model: a bounded tool loop, memory
that fits a token budget, a trace of what happened and what it cost, and a
fallback chain for when a provider is down. Provider-agnostic by construction —
the Anthropic adapter is one file on the official SDK, the OpenAI-compatible one
is one file on `fetch`, and a scripted fake runs the whole thing offline in
tests.

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
- **Parallel tool calls stay parallel, up to a cap you choose.** All `tool_use`
  blocks in one turn run concurrently, and all their results go back in one user
  message. Splitting them across messages quietly trains the model to stop
  parallelising. A tool given `maxConcurrency` queues its own calls without
  holding up any other tool's.
- **A paused turn is resumed, not answered.** A provider that interrupts its
  own server-side work stops with `pause_turn`. That is an unfinished turn, so
  the loop hands it back to be finished, runs no tool inside it, and spends an
  iteration on it like any other call.
- **Nothing runs unapproved where a gate is set.** `beforeToolCall` is asked
  about every call and has to answer. A denial — or a gate that throws — becomes
  an error result the model can work around, and the tool never runs.
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

**The second adapter is where the port earns itself.**
`providers/openai-compatible.ts` speaks `POST /chat/completions` over `fetch`,
which is the shape OpenAI, Ollama, vLLM, llama.cpp and the gateways in front of
them all serve, so `new OpenAICompatibleProvider({ model, baseURL })` reaches a
local model and a hosted one with the same loop above it. No second SDK: the
wire types in that file are the whole dependency. What the neutral types hide
is real — every tool result becomes its own `tool` message and may not be
interrupted by user text; `finish_reason` arrives as `"stop"` on turns that did
call tools, so the content decides, except under `length`, where the arguments
are cut off and the turn is `truncated` no matter what it called;
`prompt_tokens` counts the cached prefix that `Usage.inputTokens` must not, or
the cheap half of a prefix gets billed at the full rate. It streams by default
and reassembles the stream into the shape the plain endpoint returns, so there
is one mapping rather than two, and a stream that ends before a `finish_reason`
is a retryable error rather than half an answer that looks whole. `maxTokens`
has no default here: the ceiling belongs to the endpoint, and a value above a
small local model's window is a 400 there.

**A gate is asked at the one moment worth asking.** `beforeToolCall` is
consulted for every call after its input has validated and before the tool runs.
After, so whoever approves sees exactly the input `execute` will get and is
never asked to judge a payload the schema would have rejected anyway; before,
because afterwards there is nothing left to approve. The hook answers
`{ allow: true }` or `{ allow: false, reason }` — there is no implicit allow,
and a hook that throws denies, because a gate whose forgotten branch means yes
is worse than no gate. A denial is an error result the model reads and works
around, so one refused call does not end the run; abort `signal` as well to stop
there.

```ts
await runAgent({
  provider,
  tools: [refundOrder],
  input: "refund ord_42",
  // `tool.description` and the validated `input` are what the human is shown.
  beforeToolCall: ({ tool, input }) => askOnCall(tool.description, input),
});
```

**A concurrency cap is per tool, and neither the queue nor the human is on the
tool's clock.** `defineTool({ maxConcurrency: 2 })` bounds how many calls of
that one tool are in flight, for a downstream that cannot take a whole turn's
worth at once; every other tool runs unaffected, and unlimited stays the default
so parallel calls stay parallel. A slot is taken when the tool runs, not while
its gate waits — one pending approval must not starve the rest of the turn.
`timeoutMs` measures the tool itself, so a call cannot expire for queueing,
while `durationMs` still spans the wait and the trace reports the latency that
was real. Limiters live for one run, which is the scope the parallelism has:
a turn's calls are the only ones the loop ever has in flight.

**A prompt is regression-testable because the model is scripted.** A
`Scenario` is a system prompt, a tool set, an input, and the answers the model
gives, run through `FakeProvider`; `assertScenario` compares what the loop did
against a transcript shape — one line per message, tool calls and results by
tool name. Payloads stay out of that shape: the script already fixes the
wording, so asserting it would only restate the script, while what a prompt edit
really moves is which tools get called, in what order, and how many turns it
takes. What the shape *cannot* see is checked separately, because the prompt and
the tool list are inputs to each call rather than messages: a run that sent the
prompt only on its first call, or offered a tool the scenario never listed,
produces the expected lines and would otherwise pass — so every request is
checked for both, and a scripted turn the run never reached is a failure too. A
mismatch prints an index-aligned diff, since a turn that moved is the
regression, plus the transcript to paste once the change is the intended one.

```ts
import { assertScenario, callTools, reply, type Scenario } from "agent-runtime";

const scenarios: Scenario[] = [
  {
    name: "looks an order up before refunding it",
    system: SUPPORT_PROMPT,
    tools: [getOrder, refundOrder],
    input: "Refund ord_42.",
    script: [
      callTools([{ name: "get_order", input: { orderId: "ord_42" } }]),
      callTools([{ name: "refund_order", input: { orderId: "ord_42" } }]),
      reply("Refunded."),
    ],
    expect: {
      status: "completed",
      transcript: [
        "user: text",
        "assistant: tool_use(get_order)",
        "user: tool_result(get_order ok)",
        "assistant: tool_use(refund_order)",
        "user: tool_result(refund_order ok)",
        "assistant: text",
      ],
    },
  },
];

for (const scenario of scenarios) it(scenario.name, () => assertScenario(scenario));
```

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

**Server-side refusal fallbacks are opt-in too.**
`new AnthropicProvider({ serverFallbacks: true })` sends `fallbacks: "default"`
under the beta that gates that form, so a request the safety classifiers decline
is re-run on Anthropic's substitute for that refusal category inside the same
call — the loop gets an answer instead of `status: "refused"`. `"default"`
routes by category rather than naming a model, so there is no pinned substitute
to migrate when one is retired. Off by default: the answer then comes from a
model the caller did not ask for, on that model's bill. `response.model` names
whoever answered, and the span's cost follows it.

**Strict tool inputs are opt-in, and checked before they are sent.**
`defineTool({ strict: true })` asks the provider to guarantee the model's input
matches the schema, instead of only describing it. A provider can only promise
that for a schema it can enforce — every object closed
(`additionalProperties: false`), every property in `required` — and rejects the
whole request over one it cannot, naming neither the tool nor the property. So
`toToolSpec` walks the generated schema first, including array items, union
branches and `$defs`, and throws with the exact path. It throws rather than
sending the tool unstrict, because a silent downgrade breaks the promise
`execute` was given while the run carries on. Express an absent value as
`.nullable()`, not `.optional()`; Zod still validates every input, strict or
not.

**Server-side tools belong to the adapter; the pause they cause belongs to the
loop.** `serverTools` hands the API the tools it runs itself — web search, code
execution, a hosted MCP server — written the way its own docs write them.

```ts
const provider = new AnthropicProvider({
  serverTools: [{ type: "web_search_20250305", name: "web_search", max_uses: 5 }],
});
```

None of that reaches `defineTool`: the model calls these inside one turn, this
runtime never executes one, and they are sent ahead of the run's own tools so
the cache breakpoint stays at the end of the list, where it closes a prefix.
What does reach the loop is the stop reason. A long stretch of server-side work
comes back as `pause_turn`, which is an unfinished turn rather than an answer,
so the loop calls again with the turn as the last message and nothing appended —
a user message between the halves would split a turn the API is still inside.
Nothing runs inside a paused turn: a tool call in one belongs to a turn still in
progress and arrives for real when it ends. The turn's own blocks travel as
opaque `server_tool` parts that the loop never reads and the adapter hands back
byte for byte, because the encrypted search result in there is what lets the
resumed turn keep the search it already paid for. A block this adapter does not
recognise is carried rather than dropped, since a server tool released after it
would otherwise vanish from a resumed turn without a word. What a pause does not
get is a way out of the bound — it spends an iteration, so `maxIterations` still
ends a model that only ever pauses. And a server tool's own charge is per
request, not per token: it is outside `Usage` and outside the run's cost, which
stay honest about tokens rather than quietly mixing in a number they do not have.

## Layout

```
src/
  types.ts           neutral messages, tool specs, ModelProvider port, ProviderError
  tools.ts           defineTool (Zod) · executeTool: validate → approve → limit → timeout → truncate
  loop.ts            runAgent — the bounded loop, events, statuses
  memory.ts          ConversationMemory: token budget, turn-wise trimming, summariser hook
  trace.ts           Tracer / runs / spans, usage + cost, Memory and Console exporters
  pricing.ts         per-model prices, costUsd()
  fallback.ts        FallbackProvider
  sse.ts             agentSSE(): the run as a text/event-stream Response
  regression.ts      scenarios: a scripted run, its transcript shape, and the diff
  client.ts          readAgentSSE(): the same stream back into typed events
  providers/
    anthropic.ts     the only file importing @anthropic-ai/sdk
    openai-compatible.ts  the same port over POST /chat/completions, on fetch
    fake.ts          scripted provider + builders for tests
tests/               119 tests, no network, no API key
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

- **Persistence of runs.** Traces go to an exporter; where they land is your
  call. Pair with a workflow engine for durable multi-step agents.
- **A planner or multi-agent orchestration.** This is the runtime one agent
  runs on. Orchestration is a layer above it.
