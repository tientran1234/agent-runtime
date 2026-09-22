import { describe, expect, it } from "vitest";
import { ConversationMemory, splitTurns, type ChatMessage } from "../src/index.js";

const user = (text: string): ChatMessage => ({ role: "user", content: [{ type: "text", text }] });
const assistant = (text: string): ChatMessage => ({ role: "assistant", content: [{ type: "text", text }] });
const toolTurn = (id: string): ChatMessage[] => [
  { role: "assistant", content: [{ type: "tool_use", id, name: "t", input: {} }] },
  { role: "user", content: [{ type: "tool_result", toolUseId: id, content: "r" }] },
];

// one token per message, so budgets are easy to reason about
const perMessage = () => 1;

describe("splitTurns", () => {
  it("starts a turn at each human message and keeps tool traffic inside it", () => {
    const turns = splitTurns([user("q1"), ...toolTurn("a"), assistant("a1"), user("q2"), assistant("a2")]);
    expect(turns.map((t) => t.length)).toEqual([4, 2]);
  });
});

describe("ConversationMemory", () => {
  it("returns everything while under budget", async () => {
    const m = new ConversationMemory({ maxTokens: 10, estimateTokens: perMessage });
    for (const msg of [user("q1"), assistant("a1"), user("q2")]) m.append(msg);
    expect(await m.window()).toHaveLength(3);
  });

  it("drops the oldest whole turns first and never splits a tool_use from its result", async () => {
    const m = new ConversationMemory({ maxTokens: 3, estimateTokens: perMessage });
    for (const msg of [user("q1"), ...toolTurn("a"), assistant("a1"), user("q2"), assistant("a2")]) m.append(msg);
    const w = await m.window();
    expect(w.map((x) => (x.content[0] as { text?: string }).text)).toEqual(["q2", "a2"]);
    expect(w.some((x) => x.content.some((p) => p.type === "tool_result"))).toBe(false);
  });

  it("always keeps the latest turn even if it alone exceeds the budget", async () => {
    const m = new ConversationMemory({ maxTokens: 1, estimateTokens: perMessage });
    for (const msg of [user("q1"), assistant("a1"), user("q2"), ...toolTurn("z"), assistant("a2")]) m.append(msg);
    const w = await m.window();
    expect(w).toHaveLength(4);
    expect(w[0]).toEqual(user("q2"));
  });

  it("summarises dropped turns into a leading user message", async () => {
    const seen: string[][] = [];
    const m = new ConversationMemory({
      maxTokens: 2,
      estimateTokens: perMessage,
      summarize: (dropped) => {
        seen.push(dropped.map((d) => (d.content[0] as { text: string }).text));
        return `dropped ${dropped.length} messages`;
      },
    });
    for (const msg of [user("q1"), assistant("a1"), user("q2"), assistant("a2")]) m.append(msg);
    const w = await m.window();
    expect(seen).toEqual([["q1", "a1"]]);
    expect(w[0]).toEqual({ role: "user", content: [{ type: "text", text: "[Summary of earlier conversation]\ndropped 2 messages" }] });
    expect(w).toHaveLength(3);
  });

  it("keeps the full transcript available untrimmed", async () => {
    const m = new ConversationMemory({ maxTokens: 1, estimateTokens: perMessage });
    for (const msg of [user("q1"), assistant("a1"), user("q2")]) m.append(msg);
    await m.window();
    expect(m.all).toHaveLength(3);
  });
});
