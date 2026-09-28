import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import {
  ConsoleExporter,
  FakeProvider,
  MemoryExporter,
  Tracer,
  callTools,
  defineTool,
  reply,
  runAgent,
  stoppedWith,
  type AgentEvent,
  type PendingToolCall,
} from "../src/index.js";

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

const weather = defineTool({
  name: "get_weather",
  description: "Weather for a city",
  input: z.object({ city: z.string() }),
  execute: ({ city }) => `${city}: 22°C`,
});

describe("runAgent", () => {
  it("returns the text when the model does not call tools", async () => {
    const provider = new FakeProvider([reply("Hello!")]);
    const result = await runAgent({ provider, input: "hi" });
    expect(result).toMatchObject({ status: "completed", text: "Hello!", iterations: 1 });
    expect(provider.calls[0]?.messages).toEqual([{ role: "user", content: [{ type: "text", text: "hi" }] }]);
  });

  it("executes a tool call and feeds the result back before the next model call", async () => {
    const provider = new FakeProvider([
      callTools([{ name: "get_weather", input: { city: "Hanoi" }, id: "t1" }]),
      reply("It is 22°C in Hanoi."),
    ]);
    const result = await runAgent({ provider, input: "weather?", tools: [weather] });

    expect(result.status).toBe("completed");
    expect(result.iterations).toBe(2);
    const second = provider.calls[1]!.messages;
    expect(second[1]).toEqual({ role: "assistant", content: [{ type: "tool_use", id: "t1", name: "get_weather", input: { city: "Hanoi" } }] });
    expect(second[2]).toEqual({ role: "user", content: [{ type: "tool_result", toolUseId: "t1", content: "Hanoi: 22°C", isError: false }] });
  });

  it("runs parallel tool calls and returns ALL results in ONE user message, in order", async () => {
    const provider = new FakeProvider([
      callTools([
        { name: "get_weather", input: { city: "A" }, id: "a" },
        { name: "get_weather", input: { city: "B" }, id: "b" },
      ]),
      reply("done"),
    ]);
    await runAgent({ provider, input: "both", tools: [weather] });
    const results = provider.calls[1]!.messages[2]!;
    expect(results.role).toBe("user");
    expect(results.content.map((p) => (p.type === "tool_result" ? p.toolUseId : p.type))).toEqual(["a", "b"]);
  });

  it("gives the model an error result for an unknown tool rather than crashing", async () => {
    const provider = new FakeProvider([callTools([{ name: "nope", input: {}, id: "x" }]), reply("ok")]);
    await runAgent({ provider, input: "go", tools: [weather] });
    const results = provider.calls[1]!.messages[2]!;
    expect(results.content[0]).toMatchObject({ type: "tool_result", isError: true, content: "unknown tool: nope" });
  });

  it("stops at maxIterations even if the model never stops calling tools", async () => {
    const forever = Array.from({ length: 50 }, () => callTools([{ name: "get_weather", input: { city: "X" } }]));
    const provider = new FakeProvider(forever);
    const result = await runAgent({ provider, input: "loop", tools: [weather], maxIterations: 3 });
    expect(result.status).toBe("max_iterations");
    expect(result.iterations).toBe(3);
    expect(provider.calls).toHaveLength(3);
  });

  it("does not run tools after a refusal or a max_tokens stop", async () => {
    let ran = 0;
    const spy = defineTool({ name: "spy", description: "", input: z.object({}), execute: () => void ran++ });
    const refused = { ...callTools([{ name: "spy", input: {} }]), stopReason: "refusal" as const };
    expect((await runAgent({ provider: new FakeProvider([refused]), input: "x", tools: [spy] })).status).toBe("refused");
    const cut = { ...callTools([{ name: "spy", input: {} }]), stopReason: "max_tokens" as const };
    expect((await runAgent({ provider: new FakeProvider([cut]), input: "x", tools: [spy] })).status).toBe("truncated");
    expect(ran).toBe(0);
    void stoppedWith;
  });

  it("honours an abort signal between iterations", async () => {
    const controller = new AbortController();
    const provider = new FakeProvider([
      () => {
        controller.abort();
        return callTools([{ name: "get_weather", input: { city: "X" } }]);
      },
      reply("never"),
    ]);
    const result = await runAgent({ provider, input: "x", tools: [weather], signal: controller.signal });
    expect(result.status).toBe("aborted");
    expect(provider.calls).toHaveLength(1);
  });

  it("emits events in order and streams text deltas", async () => {
    const provider = new FakeProvider([callTools([{ name: "get_weather", input: { city: "Hue" }, id: "t" }], "Checking…"), reply("Hue: 22°C")]);
    const events: AgentEvent[] = [];
    await runAgent({ provider, input: "x", tools: [weather], onEvent: (e) => events.push(e) });
    expect(events.map((e) => e.type)).toEqual([
      "text_delta", "model_call", "tool_call", "tool_result", "text_delta", "model_call", "done",
    ]);
  });

  it("sums usage across iterations and records a trace with cost", async () => {
    const exporter = new MemoryExporter();
    const tracer = new Tracer({ exporters: [exporter], now: () => 1000 });
    const provider = new FakeProvider(
      [callTools([{ name: "get_weather", input: { city: "X" } }]), reply("ok", { inputTokens: 100, outputTokens: 50 })],
      "claude-opus-5",
    );
    const result = await runAgent({ provider, input: "x", tools: [weather], tracer });

    expect(result.usage).toMatchObject({ inputTokens: 120, outputTokens: 58 });
    const run = exporter.runs[0]!;
    expect(run.status).toBe("ok");
    expect(run.spans.map((s) => s.kind)).toEqual(["model.call", "tool.call", "model.call"]);
    expect(run.totals).toMatchObject({ modelCalls: 2, toolCalls: 1, toolErrors: 0 });
    // 120 in × $5/M + 58 out × $25/M
    expect(run.totals.costUsd).toBeCloseTo(120 * 5e-6 + 58 * 25e-6, 9);
  });

  it("surfaces cache reads and writes on the model spans, in the totals and in the cost", async () => {
    const exporter = new MemoryExporter();
    const lines: string[] = [];
    const tracer = new Tracer({ exporters: [exporter, new ConsoleExporter((l) => lines.push(l))], now: () => 1000 });
    const provider = new FakeProvider(
      [
        // First call writes the prefix to the cache, the second one reads it back.
        { ...callTools([{ name: "get_weather", input: { city: "X" } }]), usage: { inputTokens: 10, outputTokens: 8, cacheReadTokens: 0, cacheWriteTokens: 2000 } },
        reply("ok", { inputTokens: 12, outputTokens: 5, cacheReadTokens: 2000 }),
      ],
      "claude-opus-5",
    );
    const result = await runAgent({ provider, input: "x", tools: [weather], tracer });

    const run = exporter.runs[0]!;
    const modelSpans = run.spans.filter((s) => s.kind === "model.call");
    expect(modelSpans[0]!.usage).toMatchObject({ cacheWriteTokens: 2000, cacheReadTokens: 0 });
    expect(modelSpans[1]!.usage).toMatchObject({ cacheWriteTokens: 0, cacheReadTokens: 2000 });
    expect(run.totals.usage).toEqual({ inputTokens: 22, outputTokens: 13, cacheReadTokens: 2000, cacheWriteTokens: 2000 });
    expect(result.usage).toMatchObject({ cacheReadTokens: 2000, cacheWriteTokens: 2000 });
    // cache reads are 10% of input, writes 125% — folding either into `in` would misprice the run
    expect(run.totals.costUsd).toBeCloseTo(22 * 5e-6 + 13 * 25e-6 + 2000 * 0.5e-6 + 2000 * 6.25e-6, 9);
    expect(lines.some((l) => l.includes("cache_r=2000 cache_w=2000"))).toBe(true);
  });

  it("marks the trace as failed and rethrows when the provider throws", async () => {
    const exporter = new MemoryExporter();
    const provider = new FakeProvider([new Error("api down")]);
    await expect(runAgent({ provider, input: "x", tracer: new Tracer({ exporters: [exporter] }) })).rejects.toThrow("api down");
    expect(exporter.runs[0]).toMatchObject({ status: "error", error: "api down" });
  });
});

describe("per-tool concurrency in the loop", () => {
  it("keeps a tool inside its maxConcurrency across a turn's parallel calls", async () => {
    let inFlight = 0;
    let peak = 0;
    const scrape = defineTool({
      name: "scrape",
      description: "Fetch one page",
      input: z.object({ url: z.string() }),
      maxConcurrency: 2,
      execute: async ({ url }) => {
        peak = Math.max(peak, ++inFlight);
        await sleep(5);
        inFlight--;
        return url;
      },
    });
    const urls = ["a", "b", "c", "d", "e"];
    const provider = new FakeProvider([callTools(urls.map((url) => ({ name: "scrape", input: { url }, id: url }))), reply("done")]);
    await runAgent({ provider, input: "scrape all", tools: [scrape] });

    expect(peak).toBe(2);
    // The cap queues the calls; it must not drop or reorder any of their results.
    const results = provider.calls[1]!.messages[2]!;
    expect(results.content.map((p) => (p.type === "tool_result" ? p.content : p.type))).toEqual(urls);
  });

  it("limits each tool on its own, so one tool's queue does not stall another", async () => {
    const active = new Set<string>();
    let peak = 0;
    const serial = (name: string) =>
      defineTool({
        name,
        description: "",
        input: z.object({}),
        maxConcurrency: 1,
        execute: async () => {
          active.add(name);
          await sleep(5);
          peak = Math.max(peak, active.size);
          active.delete(name);
          return name;
        },
      });
    const provider = new FakeProvider([
      callTools([
        { name: "left", input: {}, id: "l1" },
        { name: "right", input: {}, id: "r1" },
        { name: "left", input: {}, id: "l2" },
        { name: "right", input: {}, id: "r2" },
      ]),
      reply("done"),
    ]);
    await runAgent({ provider, input: "x", tools: [serial("left"), serial("right")] });
    expect(peak).toBe(2);
  });
});

describe("approval gates in the loop", () => {
  const deleteAccount = (onRun: () => void) =>
    defineTool({ name: "delete_account", description: "Delete an account", input: z.object({ id: z.string() }), execute: onRun });

  it("turns a denied call into an error result and lets the model carry on", async () => {
    let ran = 0;
    const provider = new FakeProvider([
      callTools([{ name: "delete_account", input: { id: "u1" }, id: "d1" }]),
      reply("I need a human to approve that."),
    ]);
    const seen: PendingToolCall[] = [];
    const result = await runAgent({
      provider,
      input: "delete u1",
      tools: [deleteAccount(() => void ran++)],
      beforeToolCall: (call) => (seen.push(call), { allow: false, reason: "needs a human" }),
    });

    expect(ran).toBe(0);
    expect(result.status).toBe("completed");
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ id: "d1", input: { id: "u1" } });
    expect(seen[0]!.tool.name).toBe("delete_account");
    const results = provider.calls[1]!.messages[2]!;
    expect(results.content[0]).toMatchObject({
      type: "tool_result",
      isError: true,
      content: "tool delete_account was not approved: needs a human",
    });
  });

  it("never asks the gate about a tool that does not exist", async () => {
    const asked: string[] = [];
    const provider = new FakeProvider([callTools([{ name: "nope", input: {}, id: "x" }]), reply("ok")]);
    await runAgent({
      provider,
      input: "go",
      tools: [weather],
      beforeToolCall: (call) => (asked.push(call.tool.name), { allow: true }),
    });
    expect(asked).toEqual([]);
  });

  it("does not hold a tool's concurrency slot while the gate waits", async () => {
    let release = () => {};
    const approved = new Promise<void>((resolve) => {
      release = resolve;
    });
    const asked: string[] = [];
    const oneAtATime = defineTool({
      name: "get_weather",
      description: "Weather for a city",
      input: z.object({ city: z.string() }),
      maxConcurrency: 1,
      execute: ({ city }) => `${city}: 22°C`,
    });
    const provider = new FakeProvider([
      callTools([
        { name: "get_weather", input: { city: "A" }, id: "a" },
        { name: "get_weather", input: { city: "B" }, id: "b" },
      ]),
      reply("done"),
    ]);
    const run = runAgent({
      provider,
      input: "x",
      tools: [oneAtATime],
      beforeToolCall: async ({ id }) => {
        asked.push(id);
        await approved;
        return { allow: true };
      },
    });

    // Both gates are open at once: the slot is taken when the tool runs, not
    // while a human is still deciding.
    await vi.waitFor(() => expect(asked).toEqual(["a", "b"]));
    release();
    expect((await run).status).toBe("completed");
  });
});
