import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  FakeProvider,
  MemoryExporter,
  Tracer,
  callTools,
  defineTool,
  reply,
  runAgent,
  stoppedWith,
  type AgentEvent,
} from "../src/index.js";

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

  it("marks the trace as failed and rethrows when the provider throws", async () => {
    const exporter = new MemoryExporter();
    const provider = new FakeProvider([new Error("api down")]);
    await expect(runAgent({ provider, input: "x", tracer: new Tracer({ exporters: [exporter] }) })).rejects.toThrow("api down");
    expect(exporter.runs[0]).toMatchObject({ status: "error", error: "api down" });
  });
});
