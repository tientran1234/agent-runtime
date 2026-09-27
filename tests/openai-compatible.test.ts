import { describe, expect, it } from "vitest";
import {
  OpenAICompatibleProvider,
  fromCompletion,
  readCompletionStream,
  toChatMessages,
  toFunctionTools,
  toHttpError,
  toProviderError,
  type ChatCompletion,
  type OpenAICompatibleProviderOptions,
} from "../src/providers/openai-compatible.js";
import { ProviderError } from "../src/index.js";

/** A JSON completion, as a non-streaming endpoint answers. */
function jsonResponse(completion: ChatCompletion, status = 200): Response {
  return new Response(JSON.stringify(completion), { status, headers: { "content-type": "application/json" } });
}

/** An SSE body from exactly these chunks, so a test can split one frame across two. */
function sseResponse(chunks: readonly string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );
}

const DONE: ChatCompletion = {
  model: "gpt-4o-mini",
  choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 1, completion_tokens: 1 },
};

/** One request through a stub `fetch`: what it went to, and what it carried. */
async function send(
  options: Partial<OpenAICompatibleProviderOptions> = {},
  request: Partial<ModelRequestLike> = {},
  reply: Response = jsonResponse(DONE),
) {
  let url: string | undefined;
  let init: RequestInit | undefined;
  const provider = new OpenAICompatibleProvider({
    model: "gpt-4o-mini",
    apiKey: "sk-test",
    stream: false,
    ...options,
    fetch: async (input, sent) => {
      url = String(input);
      init = sent;
      return reply;
    },
  });
  const response = await provider.complete({
    messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
    ...request,
  });
  return { url: url!, init: init!, body: JSON.parse(String(init!.body)) as Record<string, unknown>, response };
}

type ModelRequestLike = Parameters<OpenAICompatibleProvider["complete"]>[0];

describe("openai-compatible request mapping", () => {
  it("puts the system prompt first and keeps the turn order", () => {
    const messages = toChatMessages("rules", [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "assistant", content: [{ type: "text", text: "hello" }] },
    ]);
    expect(messages).toEqual([
      { role: "system", content: "rules" },
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });

  it("sends an assistant tool call as tool_calls, with the input serialised", () => {
    const [message] = toChatMessages(undefined, [
      { role: "assistant", content: [{ type: "text", text: "checking" }, { type: "tool_use", id: "t1", name: "f", input: { a: 1 } }] },
    ]);
    expect(message).toEqual({
      role: "assistant",
      content: "checking",
      tool_calls: [{ id: "t1", type: "function", function: { name: "f", arguments: '{"a":1}' } }],
    });
  });

  it("sends null content on a tool-only turn, because an empty string is rejected there", () => {
    const [message] = toChatMessages(undefined, [
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "f", input: {} }] },
    ]);
    expect(message).toMatchObject({ role: "assistant", content: null });
  });

  it("fans one user message of tool results out into a `tool` message each, ahead of any text", () => {
    // The loop returns every result from a turn in ONE user message; here each
    // result is its own message, and they may not be interrupted by user text.
    const messages = toChatMessages(undefined, [
      {
        role: "user",
        content: [
          { type: "tool_result", toolUseId: "t1", content: "42" },
          { type: "tool_result", toolUseId: "t2", content: "boom", isError: true },
          { type: "text", text: "and now?" },
        ],
      },
    ]);
    expect(messages).toEqual([
      { role: "tool", tool_call_id: "t1", content: "42" },
      // No wire form for is_error: the content already says what failed.
      { role: "tool", tool_call_id: "t2", content: "boom" },
      { role: "user", content: "and now?" },
    ]);
  });

  it("translates tool specs into functions, asking for enforcement only where it was asked for", () => {
    const [plain, strict] = toFunctionTools([
      { name: "a", description: "d", inputSchema: { type: "object", properties: {} } },
      { name: "b", description: "d", inputSchema: {}, strict: true },
    ]);
    expect(plain).toEqual({ type: "function", function: { name: "a", description: "d", parameters: { type: "object", properties: {} } } });
    expect(plain?.function).not.toHaveProperty("strict");
    expect(strict?.function).toMatchObject({ strict: true });
  });
});

describe("openai-compatible response mapping", () => {
  const completion = (
    message: NonNullable<NonNullable<ChatCompletion["choices"]>[number]["message"]>,
    finish: string | null = "stop",
    usage?: ChatCompletion["usage"],
  ): ChatCompletion => ({ model: "gpt-4o-mini", choices: [{ message, finish_reason: finish }], ...(usage ? { usage } : {}) });

  it("maps text and tool calls back, with the arguments parsed", () => {
    const out = fromCompletion(
      completion(
        {
          content: "Let me check.",
          tool_calls: [{ id: "call_1", type: "function", function: { name: "f", arguments: '{"q":1}' } }],
        },
        "tool_calls",
      ),
      "asked-for",
    );
    expect(out.model).toBe("gpt-4o-mini");
    expect(out.stopReason).toBe("tool_use");
    expect(out.content).toEqual([
      { type: "text", text: "Let me check." },
      { type: "tool_use", id: "call_1", name: "f", input: { q: 1 } },
    ]);
  });

  it("names the requested model when the server omits one", () => {
    const out = fromCompletion({ choices: [{ message: { content: "ok" }, finish_reason: "stop" }] }, "llama3.1");
    // Cost is looked up per span on this field, so it can never be empty.
    expect(out.model).toBe("llama3.1");
  });

  it("keeps arguments that are not JSON as the raw string, and treats empty ones as no arguments", () => {
    const parsed = (args: string) => {
      const out = fromCompletion(
        completion({ tool_calls: [{ id: "c", type: "function", function: { name: "f", arguments: args } }] }, "tool_calls"),
        "m",
      );
      return out.content[0]?.type === "tool_use" ? out.content[0].input : undefined;
    };
    // The schema is the next thing that sees this, and turns it into an error
    // result the model can correct. An empty object would instead look like a
    // valid call that took no arguments.
    expect(parsed('{"a":')).toBe('{"a":');
    expect(parsed("")).toEqual({});
  });

  it("invents an id for a tool call that arrives without one", () => {
    const out = fromCompletion(
      completion({ tool_calls: [{ id: "", type: "function", function: { name: "f", arguments: "{}" } }] }, "tool_calls"),
      "m",
    );
    // The loop pairs each result back to its call by id, so two empty ids in
    // one turn would collide and the model would get the wrong answer.
    expect(out.content[0]).toMatchObject({ type: "tool_use", id: "call_0" });
  });

  it("stops for tool_use even when the server called the turn finished", () => {
    // Several compatible servers answer "stop" on a turn that called tools.
    // Believing that would end the run with the calls never run.
    const out = fromCompletion(
      completion({ content: null, tool_calls: [{ id: "c", type: "function", function: { name: "f", arguments: "{}" } }] }, "stop"),
      "m",
    );
    expect(out.stopReason).toBe("tool_use");
  });

  it("keeps a truncated turn truncated, tool calls or not", () => {
    // `length` outranks the calls: their arguments are cut off, and the loop
    // must not run a tool on an input that only happens to parse.
    const out = fromCompletion(
      completion({ tool_calls: [{ id: "c", type: "function", function: { name: "f", arguments: '{"a":1}' } }] }, "length"),
      "m",
    );
    expect(out.stopReason).toBe("max_tokens");
  });

  it("maps the remaining finish reasons, unknown ones to other", () => {
    const reason = (finish: string | null) => fromCompletion(completion({ content: "x" }, finish), "m").stopReason;
    expect(reason("stop")).toBe("end_turn");
    expect(reason("content_filter")).toBe("refusal");
    expect(reason("length")).toBe("max_tokens");
    expect(reason("insufficient_system_resource")).toBe("other");
    expect(reason(null)).toBe("other");
  });

  it("splits the cached prefix out of prompt_tokens, and reports no write", () => {
    // prompt_tokens counts the cached tokens; Usage.inputTokens must not, or
    // the cheap half of the prefix gets billed at the full input rate.
    const out = fromCompletion(
      completion({ content: "ok" }, "stop", { prompt_tokens: 100, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 80 } }),
      "m",
    );
    expect(out.usage).toEqual({ inputTokens: 20, outputTokens: 20, cacheReadTokens: 80, cacheWriteTokens: 0 });
  });

  it("reports zero usage when the server sends none", () => {
    expect(fromCompletion(completion({ content: "ok" }), "m").usage).toEqual({
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
    });
  });
});

describe("openai-compatible streaming", () => {
  const frames = (...payloads: string[]) => payloads.map((p) => `data: ${p}\n\n`);

  it("reassembles text and tool arguments across chunk boundaries, forwarding text deltas", async () => {
    const body = sseResponse([
      // A chunk is a transport detail and can split anywhere, including mid-JSON.
      'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"Let me "}}]}\n\ndata: {"choi',
      'ces":[{"delta":{"content":"check."}}]}\n\n',
      ...frames(
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":"{\\"q\\":"}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}',
        '{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}',
      ),
      "data: [DONE]\n\n",
    ]).body!;

    const deltas: string[] = [];
    const completion = await readCompletionStream(body, (text) => deltas.push(text));
    expect(deltas).toEqual(["Let me ", "check."]);

    const out = fromCompletion(completion, "m");
    expect(out.model).toBe("gpt-4o-mini");
    expect(out.stopReason).toBe("tool_use");
    expect(out.content).toEqual([
      { type: "text", text: "Let me check." },
      { type: "tool_use", id: "call_1", name: "f", input: { q: 1 } },
    ]);
    expect(out.usage.inputTokens).toBe(9);
  });

  it("continues the open call when a server sends no index", async () => {
    const body = sseResponse(
      frames(
        '{"choices":[{"delta":{"tool_calls":[{"id":"c","function":{"name":"f","arguments":"{\\"a\\""}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":":1}"}}]}}]}',
        '{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
      ),
    ).body!;
    const out = fromCompletion(await readCompletionStream(body), "m");
    expect(out.content).toEqual([{ type: "tool_use", id: "c", name: "f", input: { a: 1 } }]);
  });

  it("rejects a stream that ended before the turn did", async () => {
    // A stream that broke must not look like one that finished: without a
    // finish_reason the answer is a fragment, and the fallback chain should
    // get the chance to ask someone else.
    const body = sseResponse(frames('{"choices":[{"delta":{"content":"half an ans"}}]}')).body!;
    await expect(readCompletionStream(body)).rejects.toMatchObject({ name: "ProviderError", retryable: true });
  });
});

describe("openai-compatible on the wire", () => {
  it("posts the model and the messages to {baseURL}/chat/completions, with a bearer token", async () => {
    const { url, init, body } = await send({ baseURL: "https://api.openai.com/v1/" });
    expect(url).toBe("https://api.openai.com/v1/chat/completions");
    expect(init.method).toBe("POST");
    expect(new Headers(init.headers).get("authorization")).toBe("Bearer sk-test");
    expect(body).toMatchObject({ model: "gpt-4o-mini", messages: [{ role: "user", content: "hi" }] });
  });

  it("sends no Authorization at all without a key, which is what a local server wants", async () => {
    const { init } = await send({ apiKey: "", baseURL: "http://localhost:11434/v1" });
    expect(new Headers(init.headers).has("authorization")).toBe(false);
  });

  it("omits max_tokens unless one was asked for, and lets extraBody have the last word", async () => {
    // The ceiling differs per endpoint; a value above a small local model's
    // window is a 400 there, and every server already caps output itself.
    expect((await send()).body).not.toHaveProperty("max_tokens");
    expect((await send({ maxTokens: 512 })).body).toMatchObject({ max_tokens: 512 });
    const overridden = await send({ maxTokens: 512, extraBody: { max_tokens: undefined, max_completion_tokens: 512, temperature: 0 } });
    expect(overridden.body).not.toHaveProperty("max_tokens");
    expect(overridden.body).toMatchObject({ max_completion_tokens: 512, temperature: 0 });
  });

  it("sends tools only when there are some", async () => {
    expect((await send()).body).not.toHaveProperty("tools");
    const withTools = await send({}, { tools: [{ name: "f", description: "d", inputSchema: {} }] });
    expect(withTools.body.tools).toEqual([{ type: "function", function: { name: "f", description: "d", parameters: {} } }]);
  });

  it("streams by default, and asks for the usage the final chunk would otherwise omit", async () => {
    let sent: RequestInit | undefined;
    // Built here rather than through `send`, which turns streaming off so the
    // rest of these tests can answer with one JSON body.
    const provider = new OpenAICompatibleProvider({
      model: "m",
      fetch: async (_input, init) => {
        sent = init;
        return sseResponse(['data: {"model":"m","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n', "data: [DONE]\n\n"]);
      },
    });
    const response = await provider.complete({ messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }] });
    expect(JSON.parse(String(sent!.body))).toMatchObject({ stream: true, stream_options: { include_usage: true } });
    expect(response.content).toEqual([{ type: "text", text: "ok" }]);

    expect((await send({ stream: false })).body).not.toHaveProperty("stream");
  });

  it("names itself on the trace, the endpoint's own name if it was given one", async () => {
    expect(new OpenAICompatibleProvider({ model: "m" }).name).toBe("openai-compatible");
    expect(new OpenAICompatibleProvider({ model: "m", name: "ollama" }).name).toBe("ollama");
  });
});

describe("openai-compatible errors", () => {
  it("classifies HTTP status: 429 and 5xx retryable, 400 not, with the server's message kept", () => {
    const rate = toHttpError(429, JSON.stringify({ error: { message: "slow down" } }));
    expect(rate).toMatchObject({ retryable: true, status: 429 });
    expect(rate.message).toContain("slow down");
    expect(toHttpError(500, "upstream exploded")).toMatchObject({ retryable: true, status: 500 });
    expect(toHttpError(400, JSON.stringify({ error: { message: "bad tool schema" } }))).toMatchObject({ retryable: false, status: 400 });
    // A body that is not the documented envelope still has to reach the caller.
    expect(toHttpError(503, "<html>gateway</html>").message).toContain("gateway");
  });

  it("throws the HTTP error out of complete(), not a parse failure on the body", async () => {
    await expect(send({}, {}, jsonResponse({}, 429))).rejects.toMatchObject({ name: "ProviderError", retryable: true, status: 429 });
  });

  it("calls a dead connection retryable, and the caller's own abort not", () => {
    expect(toProviderError(new TypeError("fetch failed"))).toMatchObject({ retryable: true });
    expect(toProviderError(new DOMException("aborted", "AbortError"))).toMatchObject({ retryable: false });
    expect(toProviderError(new ProviderError("already neutral", true))).toMatchObject({ retryable: true });
    expect(toProviderError("a string")).toBeInstanceOf(ProviderError);
  });
});
