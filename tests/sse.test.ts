import { describe, expect, it } from "vitest";
import { FakeProvider, agentSSE, reply } from "../src/index.js";

describe("agentSSE", () => {
  it("streams events as text/event-stream and ends with done", async () => {
    const res = agentSSE({ provider: new FakeProvider([reply("hey")]), input: "hi" });
    expect(res.headers.get("content-type")).toBe("text/event-stream");
    const text = await res.text();
    const events = text.trim().split("\n\n").map((chunk) => chunk.split("\n")[0]!.replace("event: ", ""));
    expect(events).toEqual(["text_delta", "model_call", "done"]);
    expect(text).toContain('"status":"completed"');
  });
});
