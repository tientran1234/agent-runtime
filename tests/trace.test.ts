import { describe, expect, it } from "vitest";
import { ConsoleExporter, MemoryExporter, Tracer, type Span } from "../src/index.js";

const tracer = (exporters: MemoryExporter[], now = 1000) => new Tracer({ exporters, prices: {}, now: () => now });

describe("nested spans", () => {
  it("names its parent and lands in the same run, so one trace covers both", async () => {
    const exporter = new MemoryExporter();
    const run = tracer([exporter]).startRun("outer");
    const tool = run.startSpan("tool.call", "handoff_research");
    const inner = tool.startSpan("model.call", "fake-1");
    inner.end();
    tool.end();
    const finished = await run.end();

    expect(finished.spans.map((s) => s.name)).toEqual(["handoff_research", "fake-1"]);
    expect(finished.spans[0]?.parentId).toBeUndefined();
    expect(finished.spans[1]?.parentId).toBe(finished.spans[0]?.id);
    expect(finished.spans[1]?.runId).toBe(finished.id);
  });

  it("rolls a nested call's tokens and cost into the run's totals", async () => {
    const exporter = new MemoryExporter();
    const t = new Tracer({
      exporters: [exporter],
      prices: { "sub-1": { input: 1_000_000, output: 2_000_000, cacheRead: 0, cacheWrite: 0 } },
      now: () => 0,
    });
    const run = t.startRun("outer");
    const tool = run.startSpan("tool.call", "handoff_research");
    tool
      .startSpan("model.call", "sub-1")
      .recordUsage("sub-1", { inputTokens: 2, outputTokens: 3, cacheReadTokens: 0, cacheWriteTokens: 0 })
      .end();
    tool.end();
    const finished = await run.end();

    // The nested model call is the only one there is, so the totals are its own:
    // a handoff's cost belongs to the run that paid for it.
    expect(finished.totals.modelCalls).toBe(1);
    expect(finished.totals.toolCalls).toBe(1);
    expect(finished.totals.usage.inputTokens).toBe(2);
    expect(finished.totals.costUsd).toBeCloseTo(2 + 6, 10);
  });

  it("indents by depth when printed, so the nesting survives the flat span list", async () => {
    const lines: string[] = [];
    const t = new Tracer({ exporters: [new ConsoleExporter((l) => lines.push(l))], now: () => 0 });
    const run = t.startRun("outer");
    const tool = run.startSpan("tool.call", "handoff_research");
    const nestedTool = tool.startSpan("tool.call", "search");
    nestedTool.startSpan("model.call", "deep-1").end();
    nestedTool.end();
    tool.end();
    await run.end();

    const indents = lines.slice(1, -1).map((l) => l.length - l.trimStart().length);
    expect(indents).toEqual([2, 4, 6]);
  });

  it("treats a parent it cannot see as no parent, rather than dropping the line", () => {
    const lines: string[] = [];
    const orphan: Span = {
      id: "s2",
      runId: "r1",
      parentId: "gone",
      kind: "model.call",
      name: "fake-1",
      startedAt: 0,
      endedAt: 1,
      durationMs: 1,
      attributes: {},
    };
    new ConsoleExporter((l) => lines.push(l)).export({
      id: "r1",
      name: "outer",
      attributes: {},
      startedAt: 0,
      status: "ok",
      spans: [orphan],
      totals: { modelCalls: 0, toolCalls: 0, toolErrors: 0, usage: { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 }, costUsd: 0 },
    });
    expect(lines[1]).toMatch(/^ {2}model\.call/);
  });
});
