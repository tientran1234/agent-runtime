import { describe, expect, it } from "vitest";
import { renderTranscript, type ChatMessage } from "../src/index.js";

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

  it("prints a message with no content rather than dropping it", () => {
    expect(renderTranscript([{ role: "assistant", content: [] }])).toEqual(["assistant: (empty)"]);
  });

  it("marks a result it cannot trace back to a call", () => {
    const orphan: ChatMessage[] = [{ role: "user", content: [{ type: "tool_result", toolUseId: "gone", content: "x" }] }];
    expect(renderTranscript(orphan)).toEqual(["user: tool_result(? ok)"]);
  });
});
