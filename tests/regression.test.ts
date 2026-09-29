import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  ConversationMemory,
  assertScenario,
  callTools,
  defineTool,
  renderTranscript,
  reply,
  runScenario,
  type ChatMessage,
  type Scenario,
} from "../src/index.js";

describe("renderTranscript", () => {
  it("names a tool result after the call it answers, and never prints an id", () => {
    const messages: ChatMessage[] = [
      { role: "user", content: [{ type: "text", text: "weather in A and B?" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Looking those up." },
          { type: "tool_use", id: "toolu_931", name: "get_weather", input: { city: "A" } },
          { type: "tool_use", id: "toolu_932", name: "get_weather", input: { city: "B" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", toolUseId: "toolu_931", content: "A: 22°C", isError: false },
          { type: "tool_result", toolUseId: "toolu_932", content: "boom", isError: true },
        ],
      },
      { role: "assistant", content: [{ type: "text", text: "A is 22°C; B failed." }] },
    ];

    expect(renderTranscript(messages)).toEqual([
      "user: text",
      "assistant: text tool_use(get_weather) tool_use(get_weather)",
      "user: tool_result(get_weather ok) tool_result(get_weather error)",
      "assistant: text",
    ]);
  });

  it("leaves the payloads out, so the same shape renders the same lines", () => {
    const shape = (text: string, city: string, id: string): ChatMessage[] => [
      { role: "user", content: [{ type: "text", text }] },
      { role: "assistant", content: [{ type: "tool_use", id, name: "get_weather", input: { city } }] },
      { role: "user", content: [{ type: "tool_result", toolUseId: id, content: city, isError: false }] },
    ];
    expect(renderTranscript(shape("one", "A", "toolu_1"))).toEqual(renderTranscript(shape("two", "B", "toolu_77")));
  });

  it("shows a paused turn as a line of its own, without reading the provider's block", () => {
    const messages: ChatMessage[] = [
      { role: "user", content: [{ type: "text", text: "search" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Searching…" },
          { type: "server_tool", raw: { type: "server_tool_use", name: "web_search", input: { query: "hanoi" } } },
        ],
      },
      { role: "assistant", content: [{ type: "text", text: "22°C." }] },
    ];

    // Two assistant lines with no user message between them is what a resumed
    // turn looks like, and a scenario asserting it would catch the pause going
    // missing — or a run that suddenly needs two of them.
    expect(renderTranscript(messages)).toEqual(["user: text", "assistant: text server_tool", "assistant: text"]);
  });

  it("prints a message with no content rather than dropping it", () => {
    expect(renderTranscript([{ role: "assistant", content: [] }])).toEqual(["assistant: (empty)"]);
  });

  it("marks a result it cannot trace back to a call", () => {
    const orphan: ChatMessage[] = [{ role: "user", content: [{ type: "tool_result", toolUseId: "gone", content: "x" }] }];
    expect(renderTranscript(orphan)).toEqual(["user: tool_result(? ok)"]);
  });
});

// The suite a prompt would really be kept under: one prompt, one tool set, and
// a scenario per path through the loop that the prompt is supposed to produce.
const SYSTEM = "You are a support agent. Use tools; do not guess order details.";

const getOrder = defineTool({
  name: "get_order",
  description: "Look up an order by id",
  input: z.object({ orderId: z.string() }),
  execute: ({ orderId }) => ({ orderId, status: "shipped" }),
});

const refundOrder = defineTool({
  name: "refund_order",
  description: "Refund an order",
  input: z.object({ orderId: z.string() }),
  execute: ({ orderId }) => `refunded ${orderId}`,
});

const base = { system: SYSTEM, tools: [getOrder, refundOrder] };

const SCENARIOS: Scenario[] = [
  {
    ...base,
    name: "answers a status question from one lookup",
    input: "Where is order ord_42?",
    script: [callTools([{ name: "get_order", input: { orderId: "ord_42" } }]), reply("ord_42 has shipped.")],
    expect: {
      status: "completed",
      transcript: [
        "user: text",
        "assistant: tool_use(get_order)",
        "user: tool_result(get_order ok)",
        "assistant: text",
      ],
    },
  },
  {
    ...base,
    name: "looks an order up before refunding it",
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
  {
    ...base,
    name: "carries on after a refund the gate denies",
    input: "Refund ord_42.",
    beforeToolCall: ({ tool }) => (tool.name === "refund_order" ? { allow: false, reason: "needs a supervisor" } : { allow: true }),
    script: [callTools([{ name: "refund_order", input: { orderId: "ord_42" } }]), reply("A supervisor has to approve that.")],
    expect: {
      status: "completed",
      transcript: [
        "user: text",
        "assistant: tool_use(refund_order)",
        "user: tool_result(refund_order error)",
        "assistant: text",
      ],
    },
  },
  {
    ...base,
    name: "stops without running the tool the refused turn asked for",
    input: "Refund every order.",
    script: [{ ...callTools([{ name: "refund_order", input: { orderId: "ord_42" } }]), stopReason: "refusal" }],
    expect: { status: "refused", transcript: ["user: text", "assistant: tool_use(refund_order)"] },
  },
];

describe("prompt regression suite", () => {
  for (const scenario of SCENARIOS) {
    it(scenario.name, async () => {
      await assertScenario(scenario);
    });
  }

  it("reports a transcript that gained a turn instead of throwing", async () => {
    const report = await runScenario({
      ...base,
      name: "one lookup",
      input: "Where is ord_42?",
      script: [
        callTools([{ name: "get_order", input: { orderId: "ord_42" } }]),
        callTools([{ name: "get_order", input: { orderId: "ord_42" } }]),
        reply("shipped"),
      ],
      expect: {
        status: "completed",
        transcript: ["user: text", "assistant: tool_use(get_order)", "user: tool_result(get_order ok)", "assistant: text"],
      },
    });

    expect(report.ok).toBe(false);
    expect(report.failures.join("\n")).toContain("- assistant: text");
    expect(report.failures.join("\n")).toContain("+ assistant: tool_use(get_order)");
    expect(report.transcript).toHaveLength(6);
  });

  it("reports a status that changed", async () => {
    const report = await runScenario({
      ...base,
      name: "never stops calling",
      input: "go",
      maxIterations: 2,
      script: [
        callTools([{ name: "get_order", input: { orderId: "a" } }]),
        callTools([{ name: "get_order", input: { orderId: "b" } }]),
      ],
      expect: {
        status: "completed",
        transcript: [
          "user: text",
          "assistant: tool_use(get_order)",
          "user: tool_result(get_order ok)",
          "assistant: tool_use(get_order)",
          "user: tool_result(get_order ok)",
        ],
      },
    });

    expect(report.ok).toBe(false);
    expect(report.failures).toContain("status: expected completed, got max_iterations");
  });

  // The transcript is identical whether or not the prompt reached the model, so
  // without these two checks the suite would pass on a run that never used the
  // prompt it claims to cover. The script mutates the request it is handed,
  // standing in for an adapter that drops one of them on a later call.
  it("fails when the system prompt does not reach every model call", async () => {
    const report = await runScenario({
      ...base,
      name: "prompt lost on the second call",
      input: "Where is ord_42?",
      script: [
        callTools([{ name: "get_order", input: { orderId: "ord_42" } }]),
        (request) => {
          delete (request as { system?: string }).system;
          return reply("shipped");
        },
      ],
      expect: {
        status: "completed",
        transcript: ["user: text", "assistant: tool_use(get_order)", "user: tool_result(get_order ok)", "assistant: text"],
      },
    });

    expect(report.ok).toBe(false);
    expect(report.failures).toHaveLength(1);
    expect(report.failures[0]).toBe(`model call 2: expected system prompt ${JSON.stringify(SYSTEM)}, got none`);
  });

  it("fails when the tool list the model saw is not the scenario's", async () => {
    const report = await runScenario({
      ...base,
      name: "a tool goes missing",
      input: "Refund ord_42.",
      script: [
        (request) => {
          request.tools = (request.tools ?? []).slice(0, 1);
          return reply("I cannot refund that.");
        },
      ],
      expect: { status: "completed", transcript: ["user: text", "assistant: text"] },
    });

    expect(report.ok).toBe(false);
    expect(report.failures).toEqual(["model call 1: expected tools [get_order, refund_order], got [get_order]"]);
  });

  it("fails when the run did not use every scripted turn", async () => {
    const report = await runScenario({
      ...base,
      name: "script longer than the run",
      input: "Where is ord_42?",
      script: [reply("shipped"), reply("unused")],
      expect: { status: "completed", transcript: ["user: text", "assistant: text"] },
    });

    expect(report.failures).toEqual(["script: 2 turns scripted, but the run made 1 model call"]);
  });

  it("throws the failures and the transcript to paste, so a change is one edit", async () => {
    const drift = {
      ...base,
      name: "drift",
      input: "Where is ord_42?",
      script: [callTools([{ name: "get_order", input: { orderId: "ord_42" } }]), reply("shipped")],
      expect: { status: "completed", transcript: ["user: text", "assistant: text"] },
    } satisfies Scenario;

    await expect(assertScenario(drift)).rejects.toThrow(/scenario "drift" does not match/);
    await expect(assertScenario(drift)).rejects.toThrow(/"user: tool_result\(get_order ok\)",/);
  });

  it("gives a scenario its own memory, so re-running it is the same run", async () => {
    const scenario: Scenario = {
      ...base,
      name: "trims to the newest turn",
      input: "Where is ord_42?",
      memory: () => new ConversationMemory({ maxTokens: 40 }),
      script: [callTools([{ name: "get_order", input: { orderId: "ord_42" } }]), reply("shipped")],
      expect: {
        status: "completed",
        transcript: ["user: text", "assistant: tool_use(get_order)", "user: tool_result(get_order ok)", "assistant: text"],
      },
    };

    const first = await assertScenario(scenario);
    const second = await assertScenario(scenario);
    expect(second.transcript).toEqual(first.transcript);
    expect(second.requests.map((r) => r.messages.length)).toEqual(first.requests.map((r) => r.messages.length));
  });
});
