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
                 // | "invalid_output" | "budget_exceeded" | "suspended"
result.text;     // the final answer
result.usage;    // tokens across every model call
```

## What the loop guarantees

- **It ends.** `maxIterations` caps model calls. A model that never stops
  calling tools gets `status: "max_iterations"`, not an infinite bill.
- **It ends inside a budget, if you set one.** `maxCostUsd` and
  `maxInputTokens` are totals for the run, checked before each call rather than
  reported after it: the loop stops with `status: "budget_exceeded"` on the
  first call it can show would go over.
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
- **A long run narrows the request, never the transcript.** With
  `contextEditing: { clearToolUsesAfter: N }` only the N most recent tool
  results are sent in full and the older ones go over as a placeholder.
  `result.messages` still holds every one of them, so what a tool returned is
  not lost to having stayed inside a window.
- **A paused turn is resumed, not answered.** A provider that interrupts its
  own server-side work stops with `pause_turn`. That is an unfinished turn, so
  the loop hands it back to be finished, runs no tool inside it, and spends an
  iteration on it like any other call.
- **A structured answer is validated, never assumed.** Given
  `output: zodSchema`, `result.output` is a value that schema accepted — the
  final message is parsed here rather than taken on the provider's word. One
  miss buys a repair turn with the validation error fed back; a second is
  `status: "invalid_output"`.
- **Nothing runs unapproved where a gate is set.** `beforeToolCall` is asked
  about every call and has to answer. A denial — or a gate that throws — becomes
  an error result the model can work around, and the tool never runs.
- **A decision nobody here can make stops the run rather than being guessed.**
  A gate that answers `{ ask: true }` ends the run with `status: "suspended"`
  and a JSON snapshot of the loop. `resumeAgent` finishes it in another process,
  running only the calls that were waiting — the siblings that already ran are
  carried, not repeated.
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
`{ allow: true }`, `{ allow: false, reason }` or `{ ask: true }` — there is no
implicit allow, and a hook that throws denies, because a gate whose forgotten
branch means yes is worse than no gate. A denial is an error result the model
reads and works around, so one refused call does not end the run; abort `signal`
as well to stop there. `{ ask: true }` is the answer that is not a verdict, and
it suspends the run instead.

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

**A run that waits on a human waits as data, not as a process.** An approval
that has to come from a person is a wait of minutes or days, and nothing in a
request's lifetime holds for that: the caller times out, the container is
recycled, the deploy goes out. So the gate has a third answer. `{ ask: true }`
says the decision belongs to someone who is not in this process, and the run
ends with `status: "suspended"` and a `SuspendedRun` that is plain JSON.

```ts
const first = await runAgent({
  provider,
  tools: [getOrder, refundOrder],
  input: "Refund ord_42.",
  beforeToolCall: ({ tool }) => (tool.name === "refund_order" ? { ask: true } : { allow: true }),
});

first.status;                     // "suspended" — refund_order has not run
awaitingCalls(first.suspended!);  // [{ toolUseId: "toolu_1", name: "refund_order", input: { orderId: "ord_42" } }]

const store = new FileStore("/var/lib/approvals");
await store.put(runId, first.suspended!);
```

```ts
const result = await resumeAgent({
  provider,                                 // supplied again: none of this was serializable
  tools: [getOrder, refundOrder],
  state: await store.get(runId),
  decisions: { toolu_1: { allow: true } },  // or { allow: false, reason }, or { ask: true } to keep waiting
});
```

The snapshot is the loop's state and nothing else: the transcript through the
turn whose tools were asked about, the iterations and tokens already spent, the
budget's totals, whether the one repair round has been used, and that turn's
calls split into the ones that settled and the ones still waiting. What it
deliberately leaves out is the provider, the tools, the tracer and the memory —
those are code, and a snapshot carrying them would only be readable by the
process that wrote it, which is the opposite of the point. Resuming supplies
them again, so a stored snapshot never pins a tool to the implementation that
happened to suspend, and a week-old one still resumes against today's deploy.

**Where it waits is a port, so neither half has to own the other.** `store`
above is a `RunStore`: `put`, `get`, `delete` and `pending` over snapshots, with
`MemoryStore` for a process whose approvals do not outlive it, `FileStore`
for a host with somewhere to write, and `SqlStore` for a row. All three go
through JSON rather than holding the object, which is what makes the in-memory
one worth testing against — a snapshot a row could not have stored fails where
the fake is used, and a caller that edits what it read has not edited what is
stored. `FileStore` renames a temporary file over the target rather than writing
in place, because a half-written snapshot is not untidy but fatal: it is the
only copy of a run whose tools have not run, and there is nothing left to
rebuild it from. A run id is a name and not a path in all of them, since the id
comes from whoever owns the approval and a fake that accepts `..` where the disk
would escape its directory is how a test stops predicting the deployment.

```ts
// CREATE TABLE agent_runs (run_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
const store = new SqlStore({
  query: (sql, params) => pool.query(sql, params).then((r) => r.rows),
  dialect: "postgres",          // or "sqlite" / "mysql": the placeholder and the upsert
});
```

`SqlStore` is the upsert, the two reads and the delete, and it stops there.
What it does not take is a driver: `query` is a function you pass, so no pool is
opened inside this library, nothing is depended on to open one, and the table
arrives by whatever migration you already run. The statements are built once,
when the store is, because a table name cannot be a parameter and so is the one
part of a statement that is text — a name that is really a fragment of SQL
fails where the store was built rather than on the first run that suspends.
`put` is a single upsert rather than a read and a write, since a run that
suspends twice is one run and two processes storing it must not interleave into
a lost snapshot; `pending` hands back only ids this store could have written,
because the table is yours and may hold rows a migration or an operator left.
Which database that table lives in, and whether a snapshot should really land in
a bucket or a queue instead, is still your call — the port is four methods over
a value.

A suspension takes the whole turn, not the one call. Its siblings have already
run — all of a turn's calls go out together — so their results travel in the
snapshot as `settled` and the resumed run does not execute them again: a tool
that has had its side effect must not have a second one, and a provider will not
take a turn's results in two messages either. The transcript therefore stops at
the assistant turn, because a half-filled `tool_result` message is one every
provider rejects, and the resumed run is what completes it. An awaiting call is
not a tool error and gets no `tool_result` event: nothing failed, nobody has
answered yet.

`resumeAgent` demands a decision for every awaiting id and throws on a missing
one rather than defaulting either way. The run suspended precisely because
nobody in this process was entitled to decide, so filling the gap in here would
undo the whole exercise. `{ ask: true }` is a legal decision and suspends again,
which makes polling for an answer that has not arrived cost a read and nothing
else. The run continues rather than restarts: spent iterations still count
against `maxIterations`, `result.usage` covers both legs, and a budget resumes
against what it had already spent — a `maxCostUsd` that reset at the suspension
would bound two short runs instead of the one long run it was set on. The trace
is the exception, and is a new run with `resumedAfter` in its attributes: a span
tree cannot honestly be stitched across a gap that may have been a week.

**Hosting a suspended run is what a workflow engine is for.** The snapshot is
JSON and the decision arrives as a signal, which is exactly the shape of a step
and a wait:

```ts
import { defineSignal, defineWorkflow } from "durable-workflow";
import { awaitingCalls, resumeAgent, runAgent } from "agent-runtime";
import { z } from "zod";

const decided = defineSignal("tool-approval", z.object({ allow: z.boolean(), reason: z.string().default("") }));

export const supportAgent = defineWorkflow<{ prompt: string }, string>("support-agent", async (ctx, { prompt }) => {
  // A step's result is persisted, so the snapshot has to be JSON — which it is.
  let state = (await ctx.step("start", () => runAgent({ provider, tools, input: prompt, beforeToolCall: gate }))).suspended;

  while (state) {
    const [call] = awaitingCalls(state);
    // Costs nothing while it waits: the run is history plus a pending signal.
    const answer = await ctx.waitFor(decided, { timeoutMs: 24 * 3600_000 });
    const decisions = { [call!.toolUseId]: answer.allow ? { allow: true as const } : { allow: false as const, reason: answer.reason } };
    const next = await ctx.step(`resume-${call!.toolUseId}`, () => resumeAgent({ provider, tools, state: state!, decisions, beforeToolCall: gate }));
    if (next.status !== "suspended") return next.text;
    state = next.suspended;
  }
  return "";
});
```

The two libraries agree on one thing and need nothing else of each other: the
state in between is a value. `ctx.step` memoises the resume, so a worker that
dies after the refund went through replays that step from history rather than
refunding twice; the agent's own guarantee that a settled call is never re-run
covers the rest of the turn.

**A sub-agent is one tool call, not a second runtime.**
`handoffTool({ name, description, ...AgentOptions })` is a `ToolDefinition`
whose `execute` is a whole nested `runAgent` — its own provider, system prompt,
tools, `maxIterations` and budget. The parent sees one tool that takes one
string.

```ts
import { handoffTool, runAgent } from "agent-runtime";

const researcher = handoffTool({
  name: "research",
  description: "Hand a research task to a researcher who can read the document store",
  provider: new AnthropicProvider({ model: "claude-sonnet-5-5" }),
  system: "You are a researcher. Answer only from documents you have read.",
  tools: [searchDocs, readDoc, listSources],
  maxIterations: 12,
  maxCostUsd: 0.25,
});

const result = await runAgent({
  provider: new AnthropicProvider({ model: "claude-opus-5" }),
  system: "You are a manager. Delegate research; do not read documents yourself.",
  tools: [researcher, replyToUser],
  input: "What changed in our refund policy this year?",
  tracer,
});
```

The reason to split is what the parent stops carrying. Tool choice gets worse
the more tools there are, and the three document tools plus the instructions for
using them plus every intermediate search result are context the manager never
needs: it needs the answer. A handoff keeps all of it inside the nested run, so
the parent pays for the brief and the final text and nothing in between — and
the sub-agent can run on a cheaper model, a tighter budget and a prompt written
for one job.

The brief is therefore the whole interface. The sub-agent starts from an empty
transcript: it cannot see the parent's conversation, the files it mentioned or
any earlier tool result, so the input schema says exactly that in the place the
model will read it. A parent that points instead of explaining gets a sub-agent
working on nothing, and that is a prompt problem, not something the runtime can
paper over.

Only `status: "completed"` is an answer. Every other stop comes back as an
*error result* naming the status — `max_iterations`, `refused`,
`budget_exceeded`, a sub-agent that finished with nothing to say — because a
parent model handed half a sub-agent's work with no word of how it ended would
read it as the finished thing. As an error result rather than an exception, so
the parent can try a smaller brief or answer without it; the parent run is not
over because a handoff failed. The one stop with its own message is a nested
gate answering `{ ask: true }`: a `SuspendedRun` is one loop's state and the
parent is mid-turn in a loop of its own, so a nested suspension cannot be passed
up. Decide inside the sub-agent's gate, or gate the handoff itself in the parent
and suspend there.

Two things are taken per call rather than from the options the tool was defined
with, because the options are read once and every call runs from them: the live
abort `signal`, so an aborted parent takes its sub-agents down with it, and
`memory`, which is a factory for that reason — one `ConversationMemory` is one
conversation, and sharing it would feed each sub-agent the last one's
transcript.

A handoff is also one call in the trace. The nested run does not start a run of
its own: its spans hang under the tool call that caused them, in the parent's
run, so the parent's totals count the nested tokens and cost as what that call
cost. `ConsoleExporter` indents them.

```
run agent [ok] 3159ms
  model.call claude-opus-5               620ms in=1840 out=96 cost=$0.01392
  tool.call  research                   2131ms
    model.call claude-sonnet-5-5           910ms in=820 out=140 cost=$0.00456
    tool.call  search_docs                  41ms
    model.call claude-sonnet-5-5          1180ms in=1620 out=210 cost=$0.00801
  model.call claude-opus-5               408ms in=2010 out=64 cost=$0.01398
  totals: 4 model calls, 2 tool calls (0 failed), 6290+510 tokens, cost $0.04047
```

That is why `parentSpan` exists on `AgentOptions` at all, and why a nested run
reports no `result.trace`: the spans belong to a run somebody else opened and
will end. Where the parent is not traced, a handoff given its own `tracer`
falls back to starting a run with it.

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

**Context editing edits the request, and memory edits the window.** The two
are different cuts and they compose: `memory` decides which turns are in the
window at all, `contextEditing` decides how much of the tool traffic inside
that window goes over in full.

```ts
await runAgent({ provider, tools, input, contextEditing: { clearToolUsesAfter: 3 } });
```

A long tool loop runs out of window on its tool results rather than on the
conversation — the page a search returned four turns ago is most of what each
call carries and none of what the next answer needs. So the older results are
replaced by a placeholder on the way out, and nowhere else: `result.messages`,
the trace and a suspended run's snapshot all keep what the tool actually
returned, because the transcript is what the caller is handed back and clearing
a request is not allowed to cost that. What never goes is the `tool_result`
block itself — a `tool_use` missing its result is a 400 from every provider, so
only the content is cleared and the pairing survives. On the Anthropic adapter
this is the API's own `clear_tool_uses_20250919` edit and the clearing happens
server-side, which saves sending the results at all; `ModelProvider.editsContext`
is how an adapter says so, and the loop then leaves the request whole rather
than clearing it twice. Every other provider — the OpenAI-compatible one, a
local model, a fake — gets the same option honoured locally, which is what makes
it worth setting before you know where the run will end up.

**Cost is computed, and "unknown" stays unknown.** Each model span records
usage and looks up the price for the model that actually answered (which
matters once fallback is involved). A model missing from the price table makes
the run's total `null`, not a quietly smaller number.

**Traces ship as OTLP, without an OpenTelemetry SDK.** `OtelExporter` posts
each finished run to OTLP/HTTP, the one ingest path Tempo, Jaeger and every
collector take out of the box, with the run as the root span and its spans
beneath it — so a handoff's nesting arrives as nesting.

```ts
const tracer = new Tracer({
  exporters: [new OtelExporter({ endpoint: "http://localhost:4318", serviceName: "support-bot" })],
});
```

Model and tool spans go over under the GenAI semantic conventions a backend
already groups by: `gen_ai.request.model` and `gen_ai.response.model`, which
differ once a fallback answered, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`, `gen_ai.tool.name`. Cache tokens and cost have no
convention yet and go under `agent_runtime.*`, as do the loop's own attributes,
prefixed so `stopReason` cannot land on a name a convention means something
else by. An unpriced model leaves the cost attribute off, for the same reason
the total goes `null`.

There is no `@opentelemetry` dependency. A tracer that already models runs and
spans would otherwise carry a second one that models them differently, and the
mapping is one exported function — `toOtlpTraces(run)` — for a deployment that
ships through a queue or a sidecar instead.

A failed export is reported to `onError`, never thrown. `end()` awaits its
exporters, so an exception there would make every run that finished while the
collector was down read as a run that failed. The same reason it reads the
`partialSuccess` body of a 200: a collector that took the request and kept half
the spans would otherwise look like one that took all of them.

**A budget is checked before the call, which means forecasting it.**
`maxCostUsd` and `maxInputTokens` cap the whole run, and the cap is asked about
at the one moment a call can still be not made.

```ts
const result = await runAgent({ provider, input, tools, maxCostUsd: 0.5 });
result.status;            // "budget_exceeded" once the next call would pass $0.50
result.trace?.attributes; // { budget: "maxCostUsd $0.50000: $0.49210 spent, …" }
```

Judging a call that has not happened takes a forecast, and the forecast is the
last call's own usage: the loop appends to the transcript, so the next call
sends at least what the last one did and costs at least as much. That keeps the
guard sound rather than merely cautious — it fires only where the overrun is
already provable, never on a call that would have fit. The first call of a run
is therefore never the one stopped: nothing has been measured, and a budget that
could refuse to start would be one nobody could set. What no forecast can do is
make the cap a hard ceiling, because how long an answer runs is not knowable
before asking for it — a run can overshoot by up to one call's worth, which is
the honest bound and still the difference between a capped tool loop and an open
tab. The totals are the tracer's, priced off the tracer's own table so a
deployment's rates bind the budget too, and kept in the ledger so that setting a
cap does not oblige you to attach a tracer. The same rule that makes an unknown
price `null` there decides what happens here: a model missing from the price
table makes the spend unmeasurable, and an unmeasurable `maxCostUsd` ends the
run with the same status rather than being quietly spent through, because a cap
nobody can check is not a cap. `maxInputTokens` counts cache reads and writes
along with plain input — they are tokens the model was given, whatever rate they
were billed at. Which limit fired and what had been spent against it go on the
run, not into the result: the status is what you branch on, the numbers are what
you read when deciding to raise the cap.

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

**A structured answer is asked for and then checked anyway.**
`runAgent({ output: zodSchema })` sends the schema on as JSON Schema —
`output_config.format` on the Anthropic adapter, where decoding is constrained
to it — and `result.output` comes back parsed and typed off that schema.

```ts
const result = await runAgent({
  provider,
  input: "How hot is it in Hanoi?",
  output: z.object({ city: z.string(), celsius: z.number() }),
});

result.output;   // { city: string; celsius: number } | undefined
```

The check is not redundant with the ask. `outputSchema` is a request a provider
may have nothing to map, and a constrained decode is still not a validated
value, so the loop parses the final message itself and sets `output` only once
the schema has accepted it — which is also what makes the option work unchanged
on an adapter that cannot ask for it. A miss gets the model one more turn with
the validation error fed back, naming the field that was wrong, because a bare
"try again" is how a model repeats itself. One turn, not a loop: a model that
missed twice will not be talked into it, and each further try bills for an
answer the caller cannot use — so a second miss is `status: "invalid_output"`,
with `result.text` still carrying what came back to look at. The repair spends
an iteration like any other turn, so `maxIterations` still ends the run. Text
that is not JSON at all takes the same path as JSON the schema rejects: both are
something for the model to fix, not an exception for the caller. Without an
`output` schema `result.output` is `never`, so reading it on a run that asked
for nothing is a type error rather than a surprise.

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
  output.ts          the output schema: JSON Schema out, the final answer parsed back
  resume.ts          SuspendedRun: the loop's state as JSON, and the calls it is waiting on
  store.ts           RunStore: where a snapshot waits — a map, a directory, a row, or yours
  handoff.ts         handoffTool: a nested agent behind one tool, traced under its call
  budget.ts          BudgetLedger: what a run has spent, and whether the next call fits
  memory.ts          ConversationMemory: token budget, turn-wise trimming, summariser hook
  context.ts         clearToolUses: old tool results out of the request, transcript intact
  trace.ts           Tracer / runs / spans, usage + cost, Memory and Console exporters
  otel.ts            OtelExporter: the same runs as OTLP/HTTP JSON, for a collector
  pricing.ts         per-model prices, costUsd()
  fallback.ts        FallbackProvider
  sse.ts             agentSSE(): the run as a text/event-stream Response
  regression.ts      scenarios: a scripted run, its transcript shape, and the diff
  client.ts          readAgentSSE(): the same stream back into typed events
  providers/
    anthropic.ts     the only file importing @anthropic-ai/sdk
    openai-compatible.ts  the same port over POST /chat/completions, on fetch
    fake.ts          scripted provider + builders for tests
tests/               257 tests, no network, no API key
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

- **A database.** `SqlStore` writes four statements and `query` is a function
  you hand it, so nothing here opens a pool, depends on a driver or runs a
  migration — and traces still go to an exporter. Which database, bucket or
  queue either one really lands in is your call.
- **A planner or a router.** `handoffTool` nests one agent inside another's
  tool call, and that is the whole of it: there is no graph, no supervisor and
  nothing that decides which agent should answer. Which agent to hand to is a
  tool the model picks, and anything more is a layer above this one.
