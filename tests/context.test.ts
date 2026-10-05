import { describe, expect, it } from "vitest";
import { z } from "zod";
import { CLEARED_TOOL_RESULT, FakeProvider, callTools, clearToolUses, defineTool, reply, runAgent } from "../src/index.js";
import type { ChatMessage, ModelRequest } from "../src/index.js";

const echo = defineTool({
  name: "echo",
  description: "Echo a word back",
  input: z.object({ word: z.string() }),
  execute: ({ word }) => word,
});

/** Every tool result in a window, in the order it is sent. */
function resultsIn(messages: readonly ChatMessage[]): string[] {
  const out: string[] = [];
  for (const m of messages) {
    if (m.role !== "user") continue;
    for (const p of m.content) if (p.type === "tool_result") out.push(p.content);
  }
  return out;
}

/** Three tool-calling turns and an answer, which is three tool uses to clear from. */
function threeCalls(): FakeProvider {
  return new FakeProvider([
    callTools([{ name: "echo", input: { word: "first" }, id: "toolu_a" }]),
    callTools([{ name: "echo", input: { word: "second" }, id: "toolu_b" }]),
    callTools([{ name: "echo", input: { word: "third" }, id: "toolu_c" }]),
    reply("done"),
  ]);
}

describe("clearToolUses", () => {
  const transcript: ChatMessage[] = [
    { role: "user", content: [{ type: "text", text: "go" }] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "echo", input: {} }] },
    { role: "user", content: [{ type: "tool_result", toolUseId: "t1", content: "one" }] },
    { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "echo", input: {} }] },
    { role: "user", content: [{ type: "tool_result", toolUseId: "t2", content: "two" }] },
  ];

  it("keeps the newest results in full and clears the rest", () => {
    expect(resultsIn(clearToolUses(transcript, { clearToolUsesAfter: 1 }))).toEqual([CLEARED_TOOL_RESULT, "two"]);
    expect(resultsIn(clearToolUses(transcript, { clearToolUsesAfter: 0 }))).toEqual([CLEARED_TOOL_RESULT, CLEARED_TOOL_RESULT]);
    expect(resultsIn(clearToolUses(transcript, { clearToolUsesAfter: 2 }))).toEqual(["one", "two"]);
  });

  it("leaves what it was given alone", () => {
    clearToolUses(transcript, { clearToolUsesAfter: 0 });
    expect(resultsIn(transcript)).toEqual(["one", "two"]);
  });

  it("clears the content and never the pairing", () => {
    const edited = clearToolUses(transcript, { clearToolUsesAfter: 0 });
    const ids = edited.flatMap((m) => (m.role === "user" ? m.content : [])).map((p) => (p.type === "tool_result" ? p.toolUseId : null));
    // A tool_use whose tool_result went missing is a 400, so clearing a result
    // may not drop the block it lives in.
    expect(ids).toEqual(["t1", "t2"]);
  });

  it("counts a turn's parallel calls one by one, not as one turn", () => {
    const parallel: ChatMessage[] = [
      { role: "user", content: [{ type: "text", text: "go" }] },
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "p1", name: "echo", input: {} },
          { type: "tool_use", id: "p2", name: "echo", input: {} },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", toolUseId: "p1", content: "one" },
          { type: "tool_result", toolUseId: "p2", content: "two" },
        ],
      },
    ];
    expect(resultsIn(clearToolUses(parallel, { clearToolUsesAfter: 1 }))).toEqual([CLEARED_TOOL_RESULT, "two"]);
  });

  it("leaves a result whose call is no longer in the window", () => {
    // Memory has already dropped the turn that made the call, so this text is
    // the whole of what is left of that tool use — there is nothing to clear
    // it down to.
    const orphaned: ChatMessage[] = [
      { role: "user", content: [{ type: "tool_result", toolUseId: "gone", content: "older" }] },
      ...transcript.slice(1),
    ];
    expect(resultsIn(clearToolUses(orphaned, { clearToolUsesAfter: 1 }))).toEqual(["older", CLEARED_TOOL_RESULT, "two"]);
  });

  it("rejects a window that is not a count", () => {
    expect(() => clearToolUses(transcript, { clearToolUsesAfter: -1 })).toThrow(/non-negative integer/);
    expect(() => clearToolUses(transcript, { clearToolUsesAfter: 1.5 })).toThrow(/non-negative integer/);
  });
});

describe("context editing in the loop", () => {
  it("clears old tool results out of the request while the transcript keeps them", async () => {
    const provider = threeCalls();
    const result = await runAgent({
      provider,
      tools: [echo],
      input: "go",
      contextEditing: { clearToolUsesAfter: 1 },
    });

    expect(result.status).toBe("completed");
    // The final call carries the newest result in full and the two before it
    // as placeholders.
    expect(resultsIn(provider.calls[3]!.messages)).toEqual([CLEARED_TOOL_RESULT, CLEARED_TOOL_RESULT, "third"]);
    // The guarantee the option is worth having: the run still hands back what
    // the tools actually returned.
    expect(resultsIn(result.messages)).toEqual(["first", "second", "third"]);
  });

  it("edits every call, not just the last one", async () => {
    const provider = threeCalls();
    await runAgent({ provider, tools: [echo], input: "go", contextEditing: { clearToolUsesAfter: 1 } });
    expect(resultsIn(provider.calls[1]!.messages)).toEqual(["first"]);
    expect(resultsIn(provider.calls[2]!.messages)).toEqual([CLEARED_TOOL_RESULT, "second"]);
  });

  it("sends everything when nobody asked for editing", async () => {
    const provider = threeCalls();
    await runAgent({ provider, tools: [echo], input: "go" });
    expect(resultsIn(provider.calls[3]!.messages)).toEqual(["first", "second", "third"]);
    expect(provider.calls[3]!.contextEditing).toBeUndefined();
  });

  it("hands the ask to a provider that edits context itself, and edits nothing locally", async () => {
    /** An adapter with `context_management` of its own, like the Anthropic one. */
    class ServerSideEditing extends FakeProvider {
      readonly editsContext = true;
    }
    const provider = new ServerSideEditing([
      callTools([{ name: "echo", input: { word: "first" }, id: "toolu_a" }]),
      reply("done"),
    ]);
    await runAgent({ provider, tools: [echo], input: "go", contextEditing: { clearToolUsesAfter: 1 } });

    const sent: ModelRequest = provider.calls[1]!;
    expect(sent.contextEditing).toEqual({ clearToolUsesAfter: 1 });
    // Clearing on both sides would count what is left against two windows.
    expect(resultsIn(sent.messages)).toEqual(["first"]);
  });

  it("keeps the request out of a provider that would have to ignore it", async () => {
    const provider = threeCalls();
    await runAgent({ provider, tools: [echo], input: "go", contextEditing: { clearToolUsesAfter: 1 } });
    expect(provider.calls.every((c) => c.contextEditing === undefined)).toBe(true);
  });
});
