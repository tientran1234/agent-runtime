import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  ConversationMemory,
  FakeProvider,
  MemoryExporter,
  Tracer,
  awaitingCalls,
  callTools,
  defineTool,
  reply,
  resumeAgent,
  runAgent,
  type SuspendedRun,
  type ToolDecision,
} from "../src/index.js";

/** A tool that records every time it really ran, which is what a resume must not duplicate. */
const refund = (log: string[]) =>
  defineTool({
    name: "refund_order",
    description: "Refund an order",
    input: z.object({ orderId: z.string() }),
    execute: ({ orderId }) => (log.push(orderId), `refunded ${orderId}`),
  });

const notify = (log: string[]) =>
  defineTool({
    name: "notify",
    description: "Email the customer",
    input: z.object({ to: z.string() }),
    execute: ({ to }) => (log.push(to), `mailed ${to}`),
  });

/** A snapshot as it would come back from storage, rather than from memory. */
const overTheWire = (state: SuspendedRun): SuspendedRun => JSON.parse(JSON.stringify(state)) as SuspendedRun;

describe("suspending a run on an undecided gate", () => {
  it("stops with a snapshot, and the tool does not run", async () => {
    const ran: string[] = [];
    const provider = new FakeProvider([callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }]), reply("done")]);
    const result = await runAgent({
      provider,
      input: "refund ord_42",
      tools: [refund(ran)],
      beforeToolCall: () => ({ ask: true }),
    });

    expect(ran).toEqual([]);
    expect(result.status).toBe("suspended");
    expect(provider.calls).toHaveLength(1);
    expect(result.suspended).toMatchObject({ version: 1, iterations: 1, awaiting: ["r1"], settled: [], repaired: false });
    // The transcript stops at the assistant turn: that turn's results are not
    // all in, and half a tool_result message is one no provider accepts.
    expect(result.suspended?.messages).toHaveLength(2);
    expect(result.suspended?.messages[1]?.role).toBe("assistant");
  });

  it("names what it is waiting on, reading it back out of the transcript", async () => {
    const provider = new FakeProvider([callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }])]);
    const result = await runAgent({ provider, input: "go", tools: [refund([])], beforeToolCall: () => ({ ask: true }) });
    expect(awaitingCalls(result.suspended!)).toEqual([{ toolUseId: "r1", name: "refund_order", input: { orderId: "ord_42" } }]);
  });

  it("asks the gate only after the input validated, so a bad call never suspends", async () => {
    const provider = new FakeProvider([callTools([{ name: "refund_order", input: { orderId: 7 }, id: "r1" }]), reply("ok")]);
    const result = await runAgent({ provider, input: "go", tools: [refund([])], beforeToolCall: () => ({ ask: true }) });
    expect(result.status).toBe("completed");
    const results = provider.calls[1]!.messages[2]!;
    expect(results.content[0]).toMatchObject({ isError: true });
  });

  it("carries the siblings that already settled, so resuming re-runs none of them", async () => {
    const ran: string[] = [];
    const provider = new FakeProvider([
      callTools([
        { name: "notify", input: { to: "a@b.c" }, id: "n1" },
        { name: "refund_order", input: { orderId: "ord_42" }, id: "r1" },
      ]),
    ]);
    const result = await runAgent({
      provider,
      input: "both",
      tools: [notify(ran), refund(ran)],
      beforeToolCall: ({ tool }) => (tool.name === "refund_order" ? { ask: true } : { allow: true }),
    });

    expect(ran).toEqual(["a@b.c"]);
    expect(result.suspended?.awaiting).toEqual(["r1"]);
    expect(result.suspended?.settled).toEqual([{ type: "tool_result", toolUseId: "n1", content: "mailed a@b.c", isError: false }]);
  });

  it("carries a refused sibling as the refusal it already is", async () => {
    const provider = new FakeProvider([
      callTools([
        { name: "notify", input: { to: "a@b.c" }, id: "n1" },
        { name: "refund_order", input: { orderId: "ord_42" }, id: "r1" },
      ]),
    ]);
    const result = await runAgent({
      provider,
      input: "both",
      tools: [notify([]), refund([])],
      beforeToolCall: ({ tool }) => (tool.name === "refund_order" ? { ask: true } : { allow: false, reason: "no mail today" }),
    });
    expect(result.suspended?.settled[0]).toMatchObject({ toolUseId: "n1", isError: true, content: "tool notify was not approved: no mail today" });
  });

  it("does not count an awaiting call as a tool error, because it never ran", async () => {
    const exporter = new MemoryExporter();
    const provider = new FakeProvider([callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }])]);
    await runAgent({
      provider,
      input: "go",
      tools: [refund([])],
      tracer: new Tracer({ exporters: [exporter] }),
      beforeToolCall: () => ({ ask: true }),
    });
    const run = exporter.runs[0]!;
    expect(run.totals).toMatchObject({ toolCalls: 1, toolErrors: 0 });
    expect(run.spans.find((s) => s.kind === "tool.call")?.attributes).toMatchObject({ awaiting: true });
    expect(run.attributes.suspended).toBe("awaiting a decision on refund_order (r1)");
  });

  it("emits no tool_result for a call nobody has answered", async () => {
    const types: string[] = [];
    const provider = new FakeProvider([callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }])]);
    await runAgent({
      provider,
      input: "go",
      tools: [refund([])],
      onEvent: (e) => types.push(e.type),
      beforeToolCall: () => ({ ask: true }),
    });
    expect(types).toEqual(["model_call", "tool_call", "done"]);
  });
});

describe("resuming a suspended run", () => {
  const suspend = async (tools: ReturnType<typeof refund>[], script = [callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }])]) => {
    const result = await runAgent({ provider: new FakeProvider(script), input: "refund ord_42", tools, beforeToolCall: () => ({ ask: true }) });
    return overTheWire(result.suspended!);
  };

  it("runs the approved call, feeds its result back, and finishes the run", async () => {
    const ran: string[] = [];
    const state = await suspend([refund(ran)]);
    const provider = new FakeProvider([reply("Refunded ord_42.")]);
    const result = await resumeAgent({
      provider,
      tools: [refund(ran)],
      state,
      decisions: { r1: { allow: true } },
    });

    expect(ran).toEqual(["ord_42"]);
    expect(result).toMatchObject({ status: "completed", text: "Refunded ord_42." });
    // The resumed call's result is the first thing the new process sends, in the
    // turn the model was always waiting on.
    expect(provider.calls[0]!.messages[2]).toEqual({
      role: "user",
      content: [{ type: "tool_result", toolUseId: "r1", content: "refunded ord_42", isError: false }],
    });
  });

  it("does not run a call the decision refused, and lets the model work around it", async () => {
    const ran: string[] = [];
    const state = await suspend([refund(ran)]);
    const provider = new FakeProvider([reply("A human declined the refund.")]);
    const result = await resumeAgent({
      provider,
      tools: [refund(ran)],
      state,
      decisions: { r1: { allow: false, reason: "the owner said no" } },
    });

    expect(ran).toEqual([]);
    expect(result.status).toBe("completed");
    expect(provider.calls[0]!.messages[2]).toMatchObject({
      content: [{ toolUseId: "r1", isError: true, content: "tool refund_order was not approved: the owner said no" }],
    });
  });

  it("suspends again on a decision that has still not arrived, with the same call waiting", async () => {
    const ran: string[] = [];
    const state = await suspend([refund(ran)]);
    const result = await resumeAgent({
      provider: new FakeProvider([]),
      tools: [refund(ran)],
      state,
      decisions: { r1: { ask: true } },
    });

    expect(ran).toEqual([]);
    expect(result.status).toBe("suspended");
    expect(result.suspended).toMatchObject({ iterations: 1, awaiting: ["r1"] });
  });

  it("never re-runs a sibling that settled before the suspension", async () => {
    const ran: string[] = [];
    const tools = [notify(ran), refund(ran)];
    const first = await runAgent({
      provider: new FakeProvider([
        callTools([
          { name: "notify", input: { to: "a@b.c" }, id: "n1" },
          { name: "refund_order", input: { orderId: "ord_42" }, id: "r1" },
        ]),
      ]),
      input: "both",
      tools,
      beforeToolCall: ({ tool }) => (tool.name === "refund_order" ? { ask: true } : { allow: true }),
    });

    const provider = new FakeProvider([reply("all done")]);
    await resumeAgent({ provider, tools, state: overTheWire(first.suspended!), decisions: { r1: { allow: true } } });

    expect(ran).toEqual(["a@b.c", "ord_42"]);
    // Both results in one message, in the order the model asked for the calls —
    // the carried one first even though it was run in another process.
    expect(provider.calls[0]!.messages[2]).toMatchObject({
      role: "user",
      content: [{ toolUseId: "n1" }, { toolUseId: "r1" }],
    });
  });

  it("keeps the iterations already spent, so maxIterations bounds the whole run", async () => {
    const state = await suspend([refund([])]);
    expect(state.iterations).toBe(1);
    const provider = new FakeProvider([callTools([{ name: "refund_order", input: { orderId: "ord_9" }, id: "r2" }])]);
    const result = await resumeAgent({
      provider,
      tools: [refund([])],
      state,
      maxIterations: 2,
      decisions: { r1: { allow: true } },
      beforeToolCall: () => ({ allow: true }),
    });

    expect(result.status).toBe("max_iterations");
    expect(result.iterations).toBe(2);
    expect(provider.calls).toHaveLength(1);
  });

  it("keeps the tokens already spent, so the finished run reports the whole run", async () => {
    const state = await suspend([refund([])]);
    const result = await resumeAgent({
      provider: new FakeProvider([reply("ok")]),
      tools: [refund([])],
      state,
      decisions: { r1: { allow: true } },
    });
    // 20 in / 8 out on the suspended turn, 10 in / 5 out on the one that finished it.
    expect(result.usage).toMatchObject({ inputTokens: 30, outputTokens: 13 });
  });

  it("keeps a budget binding across the suspension instead of letting it restart", async () => {
    // Two tool turns at claude-opus-5's rates cost $0.0006; a $0.0005 cap is
    // passed by the second, and only a ledger that remembers the first can see it.
    const provider = new FakeProvider([callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }])], "claude-opus-5");
    const first = await runAgent({
      provider,
      input: "refund ord_42",
      tools: [refund([])],
      maxCostUsd: 0.0005,
      beforeToolCall: () => ({ ask: true }),
    });
    const state = overTheWire(first.suspended!);
    expect(state.budget?.spentUsd).toBeCloseTo(20 * 5e-6 + 8 * 25e-6, 10);

    const result = await resumeAgent({
      provider: new FakeProvider([reply("ok")], "claude-opus-5"),
      tools: [refund([])],
      state,
      maxCostUsd: 0.0005,
      decisions: { r1: { allow: true } },
    });
    expect(result.status).toBe("budget_exceeded");
    // And the same resume without the carried budget would have run: a ledger
    // starting from zero has nothing to forecast the second call from.
    const { budget: _dropped, ...unbudgeted } = state;
    const restarted = await resumeAgent({
      provider: new FakeProvider([reply("ok")], "claude-opus-5"),
      tools: [refund([])],
      state: unbudgeted,
      maxCostUsd: 0.0005,
      decisions: { r1: { allow: true } },
    });
    expect(restarted.status).toBe("completed");
  });

  it("reports the repair round as already spent, so a resumed run cannot buy a second", async () => {
    const schema = z.object({ city: z.string() });
    const provider = new FakeProvider([
      reply("not json"),
      callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }]),
    ]);
    const first = await runAgent({
      provider,
      input: "go",
      tools: [refund([])],
      output: schema,
      beforeToolCall: () => ({ ask: true }),
    });
    expect(first.suspended?.repaired).toBe(true);

    const result = await resumeAgent({
      provider: new FakeProvider([reply("still not json")]),
      tools: [refund([])],
      state: overTheWire(first.suspended!),
      output: schema,
      decisions: { r1: { allow: true } },
    });
    expect(result.status).toBe("invalid_output");
  });

  it("refills a fresh memory from the snapshot, so the window covers the whole run", async () => {
    const state = await suspend([refund([])]);
    const memory = new ConversationMemory({ maxTokens: 10_000 });
    await resumeAgent({
      provider: new FakeProvider([reply("ok")]),
      tools: [refund([])],
      state,
      memory,
      decisions: { r1: { allow: true } },
    });
    expect(memory.all.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
  });

  it("traces the resumed leg as its own run, saying what it is resuming after", async () => {
    const exporter = new MemoryExporter();
    const state = await suspend([refund([])]);
    await resumeAgent({
      provider: new FakeProvider([reply("ok")]),
      tools: [refund([])],
      state,
      tracer: new Tracer({ exporters: [exporter] }),
      decisions: { r1: { allow: true } },
    });
    expect(exporter.runs[0]?.attributes).toMatchObject({ resumedAfter: 1 });
    expect(exporter.runs[0]?.totals.modelCalls).toBe(1);
  });

  it("refuses to resume without an answer for every awaiting call", async () => {
    const ran: string[] = [];
    const state = await suspend([refund(ran)]);
    await expect(
      resumeAgent({ provider: new FakeProvider([]), tools: [refund(ran)], state, decisions: {} }),
    ).rejects.toThrow(/no decision for refund_order \(r1\)/);
    expect(ran).toEqual([]);
  });

  it("refuses a snapshot that is waiting on nothing, and one it cannot read", async () => {
    const state = await suspend([refund([])]);
    await expect(
      resumeAgent({ provider: new FakeProvider([]), tools: [], state: { ...state, awaiting: [] }, decisions: {} }),
    ).rejects.toThrow(/not waiting on anything/);
    await expect(
      resumeAgent({ provider: new FakeProvider([]), tools: [], state: { ...state, version: 2 as 1 }, decisions: {} }),
    ).rejects.toThrow(/version 2/);
    await expect(
      resumeAgent({ provider: new FakeProvider([]), tools: [], state: { ...state, awaiting: ["ghost"] }, decisions: { ghost: { allow: true } } }),
    ).rejects.toThrow(/awaiting ghost/);
  });

  it("does not fire the approved call when the signal is already aborted", async () => {
    const ran: string[] = [];
    const state = await suspend([refund(ran)]);
    const result = await resumeAgent({
      provider: new FakeProvider([]),
      tools: [refund(ran)],
      state,
      signal: AbortSignal.abort(),
      decisions: { r1: { allow: true } },
    });
    expect(ran).toEqual([]);
    expect(result.status).toBe("aborted");
  });

  it("refuses a snapshot that lost one of the turn's calls, rather than resuming a half turn", async () => {
    const first = await runAgent({
      provider: new FakeProvider([
        callTools([
          { name: "notify", input: { to: "a@b.c" }, id: "n1" },
          { name: "refund_order", input: { orderId: "ord_42" }, id: "r1" },
        ]),
      ]),
      input: "both",
      tools: [notify([]), refund([])],
      beforeToolCall: ({ tool }) => (tool.name === "refund_order" ? { ask: true } : { allow: true }),
    });
    const state = { ...overTheWire(first.suspended!), settled: [] };
    await expect(
      resumeAgent({ provider: new FakeProvider([]), tools: [], state, decisions: { r1: { allow: true } } }),
    ).rejects.toThrow(/neither awaits nor has a result for notify \(n1\)/);
  });

  it("gives the model an error result for a tool the new process no longer has", async () => {
    const state = await suspend([refund([])]);
    const provider = new FakeProvider([reply("ok")]);
    await resumeAgent({ provider, tools: [], state, decisions: { r1: { allow: true } } });
    expect(provider.calls[0]!.messages[2]).toMatchObject({
      content: [{ toolUseId: "r1", isError: true, content: "unknown tool: refund_order" }],
    });
  });

  it("survives the trip through JSON, which is the whole point of the snapshot", async () => {
    const state = await suspend([refund([])]);
    const stored = JSON.stringify(state);
    const decisions: Record<string, ToolDecision> = { r1: { allow: true } };
    const result = await resumeAgent({
      provider: new FakeProvider([reply("ok")]),
      tools: [refund([])],
      state: JSON.parse(stored) as SuspendedRun,
      decisions,
    });
    expect(result.status).toBe("completed");
  });
});
