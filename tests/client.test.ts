import { describe, expect, it } from "vitest";
import { z } from "zod";
import { FakeProvider, agentSSE, callTools, defineTool, reply } from "../src/index.js";
import { AgentSSEError, readAgentSSE, type AgentSSEEvent } from "../src/client.js";

const DONE_FRAME =
  'event: done\ndata: {"type":"done","result":{"status":"completed","text":"hello","messages":[],"iterations":1,' +
  '"usage":{"inputTokens":1,"outputTokens":2,"cacheReadTokens":0,"cacheWriteTokens":0}}}\n\n';

/** A stream whose chunk boundaries fall exactly where the test puts them. */
function stream(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
  );
}

describe("readAgentSSE", () => {
  it("reassembles a frame split across chunks, mid-JSON included", async () => {
    const deltas: string[] = [];
    const result = await readAgentSSE(
      stream(['event: text_delta\ndata: {"type":"text_d', 'elta","text":"hel', 'lo"}\n\n' + DONE_FRAME]),
      { text_delta: (event) => deltas.push(event.text) },
    );

    expect(deltas).toEqual(["hello"]);
    expect(result?.status).toBe("completed");
  });

  it("holds a frame back until the blank line that ends it arrives", async () => {
    const deltas: string[] = [];
    await readAgentSSE(
      stream([
        'event: text_delta\ndata: {"type":"text_delta","text":"a"}\n',
        '\nevent: text_delta\ndata: {"type":"text_delta","text":"b"}\n\n',
      ]),
      { text_delta: (event) => deltas.push(event.text) },
    );

    expect(deltas).toEqual(["a", "b"]);
  });

  it("reads a real agentSSE response event by event and returns its result", async () => {
    const weather = defineTool({
      name: "get_weather",
      description: "Weather for a city",
      input: z.object({ city: z.string() }),
      execute: ({ city }) => `${city}: 22°C`,
    });
    const response = agentSSE({
      provider: new FakeProvider([
        callTools([{ name: "get_weather", input: { city: "Hanoi" }, id: "t1" }]),
        reply("It is 22°C in Hanoi."),
      ]),
      input: "weather?",
      tools: [weather],
    });

    const seen: string[] = [];
    const calls: unknown[] = [];
    const results: string[] = [];
    const result = await readAgentSSE(response, {
      onEvent: (event) => seen.push(event.type),
      tool_call: (event) => calls.push({ name: event.name, input: event.input }),
      tool_result: (event) => results.push(event.content),
    });

    expect(seen).toEqual(["model_call", "tool_call", "tool_result", "text_delta", "model_call", "done"]);
    expect(calls).toEqual([{ name: "get_weather", input: { city: "Hanoi" } }]);
    expect(results).toEqual(["Hanoi: 22°C"]);
    expect(result).toMatchObject({ status: "completed", text: "It is 22°C in Hanoi.", iterations: 2 });
  });

  it("skips keep-alive comments and event types it does not know", async () => {
    const seen: AgentSSEEvent[] = [];
    const result = await readAgentSSE(
      stream([
        ":ping\n\n",
        'event: cache_hit\ndata: {"type":"cache_hit","tokens":12}\n\n',
        'event: text_delta\ndata: {"type":"text_delta","text":"hi"}\n\n',
      ]),
      { onEvent: (event) => seen.push(event) },
    );

    expect(seen).toEqual([{ type: "text_delta", text: "hi" }]);
    // No `done` frame ever arrived, so there is no result to report.
    expect(result).toBeUndefined();
  });

  it("joins multi-line data and still reads a frame that never got its blank line", async () => {
    const deltas: string[] = [];
    await readAgentSSE(stream(['event: text_delta\ndata: {"type":"text_delta",\ndata: "text":"split"}']), {
      text_delta: (event) => deltas.push(event.text),
    });

    expect(deltas).toEqual(["split"]);
  });

  it("hands a failed run to the error handler, with the aborted result beside it", async () => {
    const response = agentSSE({ provider: new FakeProvider([new Error("upstream exploded")]), input: "hi" });
    const errors: string[] = [];

    const result = await readAgentSSE(response, { error: (event) => errors.push(event.message) });

    expect(errors).toEqual(["upstream exploded"]);
    expect(result?.status).toBe("aborted");
  });

  it("rejects when nothing handles the error frame, so a failed run is never silent", async () => {
    const response = agentSSE({ provider: new FakeProvider([new Error("upstream exploded")]), input: "hi" });

    await expect(readAgentSSE(response)).rejects.toThrow(/upstream exploded/);
  });

  it("rejects a response that never was a stream", async () => {
    await expect(readAgentSSE(new Response("nope", { status: 500 }))).rejects.toThrow(AgentSSEError);
  });
});
