import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  ConversationMemory,
  FakeProvider,
  MemoryExporter,
  Tracer,
  callTools,
  defineTool,
  handoffTool,
  reply,
  runAgent,
  toToolSpec,
} from "../src/index.js";

const search = (log: string[] = []) =>
  defineTool({
    name: "search",
    description: "Search the notes",
    input: z.object({ q: z.string() }),
    execute: ({ q }) => (log.push(q), `nothing about ${q}`),
  });

/** A parent that delegates once and then answers from what came back. */
const delegating = (taskText = "summarise the Q3 notes") =>
  new FakeProvider([callTools([{ name: "research", input: { task: taskText }, id: "h1" }]), reply("parent answer")]);

describe("handoffTool", () => {
  it("runs a nested agent and returns its final text to the parent", async () => {
    const sub = new FakeProvider([callTools([{ name: "search", input: { q: "q3" } }]), reply("Q3 was flat")]);
    const result = await runAgent({
      provider: delegating(),
      input: "what happened in Q3?",
      tools: [handoffTool({ name: "research", description: "Hand a research task to a researcher", provider: sub })],
    });

    expect(result.status).toBe("completed");
    expect(result.text).toBe("parent answer");
    // The parent's transcript carries the sub-agent's answer as the tool result,
    // and nothing else of the nested run.
    const toolResult = result.messages[2]!;
    expect(toolResult.content[0]).toMatchObject({ toolUseId: "h1", content: "Q3 was flat", isError: false });
  });

  it("starts the sub-agent from the brief alone, with its own prompt and tools", async () => {
    const sub = new FakeProvider([reply("done")]);
    const seen: string[] = [];
    await runAgent({
      provider: delegating("read notes/q3.md and list the three biggest drops"),
      system: "You are a manager.",
      input: "what happened in Q3?",
      tools: [
        handoffTool({
          name: "research",
          description: "Hand a research task to a researcher",
          provider: sub,
          system: "You are a researcher.",
          tools: [search(seen)],
        }),
      ],
    });

    const request = sub.calls[0]!;
    // Not one message of the parent's conversation: the brief is the whole
    // interface, which is what keeps the two context windows separate.
    expect(request.messages).toEqual([
      { role: "user", content: [{ type: "text", text: "read notes/q3.md and list the three biggest drops" }] },
    ]);
    expect(request.system).toBe("You are a researcher.");
    expect(request.tools?.map((t) => t.name)).toEqual(["search"]);
  });

  it("tells the model the sub-agent cannot see this conversation", () => {
    const spec = toToolSpec(handoffTool({ name: "research", description: "Research something", provider: new FakeProvider([]) }));
    const task = (spec.inputSchema.properties as Record<string, { description?: string }>).task;
    expect(task?.description).toMatch(/does not see this conversation/);
  });

  it("keeps a parent's tools out of the sub-agent and the sub-agent's out of the parent", async () => {
    const sub = new FakeProvider([reply("done")]);
    const parent = delegating();
    await runAgent({
      provider: parent,
      input: "go",
      tools: [
        handoffTool({ name: "research", description: "Research", provider: sub, tools: [search()] }),
        defineTool({ name: "reply_to_user", description: "", input: z.object({}), execute: () => "ok" }),
      ],
    });
    expect(parent.calls[0]!.tools?.map((t) => t.name)).toEqual(["research", "reply_to_user"]);
    expect(sub.calls[0]!.tools?.map((t) => t.name)).toEqual(["search"]);
  });
});

describe("a handoff in the parent's trace", () => {
  // The parent's model is priced too, at zero: one unknown price makes the
  // run's total null, which would hide whether the nested cost rolled up.
  const prices = {
    "fake-1": { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    "sub-1": { input: 1_000_000, output: 0, cacheRead: 0, cacheWrite: 0 },
  };

  it("hangs the nested spans under the tool call and rolls their cost into the run", async () => {
    const exporter = new MemoryExporter();
    const sub = new FakeProvider([callTools([{ name: "search", input: { q: "q3" } }]), reply("Q3 was flat")], "sub-1");
    await runAgent({
      provider: delegating(),
      input: "go",
      tools: [handoffTool({ name: "research", description: "Research", provider: sub, tools: [search()] })],
      tracer: new Tracer({ exporters: [exporter], prices, now: () => 0 }),
    });

    const run = exporter.runs[0]!;
    const handoffSpan = run.spans.find((s) => s.name === "research")!;
    const nested = run.spans.filter((s) => s.parentId === handoffSpan.id);
    expect(nested.map((s) => `${s.kind} ${s.name}`)).toEqual(["model.call sub-1", "tool.call search", "model.call sub-1"]);
    // Which provider served the handoff is on the span that caused it, so the
    // nested calls do not have to be read to see what ran there.
    expect(handoffSpan.attributes).toMatchObject({ provider: "fake", model: "sub-1", handoffStatus: "completed", handoffIterations: 2 });

    // Every call of both agents, counted once, in the run that paid for them.
    expect(run.totals.modelCalls).toBe(4);
    expect(run.totals.toolCalls).toBe(2);
    // Only the sub-agent's model is priced above zero, so the run's total is
    // exactly what the nested calls cost: 20 input tokens then 10, at $1 each.
    expect(run.totals.costUsd).toBeCloseTo(30, 10);
  });

  it("gives a nested run no trace of its own, because the spans are the parent's", async () => {
    const exporter = new MemoryExporter();
    const sub = new FakeProvider([reply("sub answer")], "sub-1");
    await runAgent({
      provider: delegating(),
      input: "go",
      tools: [handoffTool({ name: "research", description: "Research", provider: sub })],
      tracer: new Tracer({ exporters: [exporter], prices, now: () => 0 }),
    });
    expect(exporter.runs).toHaveLength(1);
  });

  it("falls back to its own tracer when the parent is not traced", async () => {
    const exporter = new MemoryExporter();
    const sub = new FakeProvider([reply("sub answer")], "sub-1");
    await runAgent({
      provider: delegating(),
      input: "go",
      tools: [
        handoffTool({
          name: "research",
          description: "Research",
          provider: sub,
          tracer: new Tracer({ exporters: [exporter], prices, now: () => 0 }),
          runName: "researcher",
        }),
      ],
    });
    expect(exporter.runs.map((r) => r.name)).toEqual(["researcher"]);
    expect(exporter.runs[0]!.spans.every((s) => s.parentId === undefined)).toBe(true);
  });
});

describe("a sub-agent that does not finish", () => {
  it("comes back as an error result naming the status, not as an answer", async () => {
    const sub = new FakeProvider([
      callTools([{ name: "search", input: { q: "a" } }]),
      callTools([{ name: "search", input: { q: "b" } }]),
    ]);
    const result = await runAgent({
      provider: delegating(),
      input: "go",
      tools: [handoffTool({ name: "research", description: "Research", provider: sub, tools: [search()], maxIterations: 2 })],
    });

    // The parent run is not over: a failed handoff is one tool result the model
    // reads and works around, like any other.
    expect(result.status).toBe("completed");
    expect(result.messages[2]!.content[0]).toMatchObject({
      isError: true,
      content: expect.stringContaining("sub-agent research stopped with status max_iterations"),
    });
  });

  it("says a nested gate cannot suspend the parent, rather than losing the decision", async () => {
    const ran: string[] = [];
    const sub = new FakeProvider([callTools([{ name: "search", input: { q: "a" } }]), reply("unreachable")]);
    const result = await runAgent({
      provider: delegating(),
      input: "go",
      tools: [
        handoffTool({
          name: "research",
          description: "Research",
          provider: sub,
          tools: [search(ran)],
          beforeToolCall: () => ({ ask: true }),
        }),
      ],
    });

    expect(ran).toEqual([]);
    expect(result.status).toBe("completed");
    expect(result.suspended).toBeUndefined();
    expect(result.messages[2]!.content[0]).toMatchObject({
      isError: true,
      content: expect.stringMatching(/suspended on an undecided approval gate, which a handoff cannot carry/),
    });
  });

  it("reports a sub-agent that completed with nothing to say", async () => {
    const sub = new FakeProvider([reply("")]);
    const result = await runAgent({
      provider: delegating(),
      input: "go",
      tools: [handoffTool({ name: "research", description: "Research", provider: sub })],
    });
    expect(result.messages[2]!.content[0]).toMatchObject({
      isError: true,
      content: expect.stringContaining("completed without returning any text"),
    });
  });
});

describe("what a handoff passes down per call", () => {
  it("takes the parent's abort signal down with it", async () => {
    const controller = new AbortController();
    const abort = defineTool({
      name: "abort_everything",
      description: "",
      input: z.object({}),
      execute: () => (controller.abort(), "aborted"),
    });
    const sub = new FakeProvider([callTools([{ name: "abort_everything", input: {} }]), reply("too late")]);
    const result = await runAgent({
      provider: delegating(),
      input: "go",
      tools: [handoffTool({ name: "research", description: "Research", provider: sub, tools: [abort] })],
      signal: controller.signal,
    });

    // The sub-agent stopped at the abort rather than making its second call,
    // which is the only way to tell the signal reached it at all.
    expect(sub.calls).toHaveLength(1);
    expect(result.status).toBe("aborted");
  });

  it("builds fresh memory for each handoff, so one sub-agent never reads another's transcript", async () => {
    const built: ConversationMemory[] = [];
    const sub = new FakeProvider([reply("first"), reply("second")]);
    const tool = handoffTool({
      name: "research",
      description: "Research",
      provider: sub,
      memory: () => {
        const memory = new ConversationMemory({ maxTokens: 1_000 });
        built.push(memory);
        return memory;
      },
    });

    expect(await tool.execute({ task: "one" }, {})).toBe("first");
    expect(await tool.execute({ task: "two" }, {})).toBe("second");
    expect(built).toHaveLength(2);
    expect(built[1]!.all.map((m) => m.content)).toEqual([
      [{ type: "text", text: "two" }],
      [{ type: "text", text: "second" }],
    ]);
  });
});
