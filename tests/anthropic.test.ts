import Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";
import { AnthropicProvider, SERVER_FALLBACK_BETA, fromMessage, toMessageParams, toProviderError, toSystem, toTools } from "../src/providers/anthropic.js";
import type { AnthropicProviderOptions } from "../src/providers/anthropic.js";
import { ProviderError } from "../src/index.js";

/** The beta params are a superset, so one type covers whichever endpoint was called. */
type SentParams = Anthropic.Beta.Messages.MessageCreateParams;
type Endpoint = "messages" | "beta.messages";

/** Enough of the SDK surface for `complete()` to run: capture the params, answer from a canned message. */
function stubClient(capture: (params: SentParams, endpoint: Endpoint) => void): Anthropic {
  const message = {
    model: "claude-opus-5",
    stop_reason: "end_turn",
    content: [{ type: "text", text: "ok" }],
    usage: { input_tokens: 1, output_tokens: 1 },
  } as unknown as Anthropic.Message;
  const streamOn = (endpoint: Endpoint) => (params: SentParams) => {
    capture(params, endpoint);
    return { on: () => {}, finalMessage: async () => message };
  };
  return {
    messages: { stream: streamOn("messages") },
    beta: { messages: { stream: streamOn("beta.messages") } },
  } as unknown as Anthropic;
}

describe("anthropic mapping", () => {
  it("translates neutral messages into SDK params, tool results included", () => {
    const params = toMessageParams([
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "f", input: { a: 1 } }] },
      { role: "user", content: [{ type: "tool_result", toolUseId: "t1", content: "boom", isError: true }] },
    ]);
    expect(params[2]).toEqual({ role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "boom", is_error: true }] });
    expect(params[1]).toEqual({ role: "assistant", content: [{ type: "tool_use", id: "t1", name: "f", input: { a: 1 } }] });
  });

  it("translates tool specs", () => {
    const [tool] = toTools([{ name: "f", description: "d", inputSchema: { type: "object", properties: {} } }]);
    expect(tool).toMatchObject({ name: "f", description: "d", input_schema: { type: "object" } });
  });

  it("asks for enforcement only on the tools that opted into it", () => {
    const [plain, strict] = toTools([
      { name: "a", description: "d", inputSchema: {} },
      { name: "b", description: "d", inputSchema: {}, strict: true },
    ]);
    expect(plain).not.toHaveProperty("strict");
    expect(strict).toMatchObject({ strict: true });
  });

  it("leaves the system prompt and the tools uncached by default", () => {
    expect(toSystem("rules")).toBe("rules");
    expect(toTools([{ name: "f", description: "d", inputSchema: {} }])[0]).not.toHaveProperty("cache_control");
  });

  it("breaks the cache on the system prompt and on the LAST tool only", () => {
    expect(toSystem("rules", true)).toEqual([{ type: "text", text: "rules", cache_control: { type: "ephemeral" } }]);
    const tools = toTools(
      [
        { name: "a", description: "d", inputSchema: {} },
        { name: "b", description: "d", inputSchema: {} },
      ],
      true,
    );
    expect(tools.map((t) => t.cache_control)).toEqual([undefined, { type: "ephemeral" }]);
  });

  it("sends both breakpoints in one request when cache is on, and none when it is off", async () => {
    const send = async (cache: boolean) => {
      let sent: SentParams | undefined;
      const provider = new AnthropicProvider({
        cache,
        client: stubClient((params) => (sent = params)),
        model: "claude-opus-5",
      });
      await provider.complete({
        system: "rules",
        messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
        tools: [{ name: "f", description: "d", inputSchema: {} }],
      });
      return sent!;
    };

    const cached = await send(true);
    expect(cached.system).toEqual([{ type: "text", text: "rules", cache_control: { type: "ephemeral" } }]);
    expect(cached.tools?.[0]).toMatchObject({ cache_control: { type: "ephemeral" } });

    const plain = await send(false);
    expect(plain.system).toBe("rules");
    expect(plain.tools?.[0]).not.toHaveProperty("cache_control");
  });

  it("normalises an SDK message: text + tool_use kept, thinking dropped, usage mapped", () => {
    const message = {
      id: "msg_1",
      type: "message",
      role: "assistant",
      model: "claude-opus-5",
      stop_reason: "tool_use",
      stop_sequence: null,
      content: [
        { type: "thinking", thinking: "…", signature: "s" },
        { type: "text", text: "Let me check." },
        { type: "tool_use", id: "t1", name: "f", input: { q: 1 } },
      ],
      usage: { input_tokens: 40, output_tokens: 12, cache_read_input_tokens: 30, cache_creation_input_tokens: null },
    } as unknown as Anthropic.Message;

    const out = fromMessage(message);
    expect(out.model).toBe("claude-opus-5");
    expect(out.stopReason).toBe("tool_use");
    expect(out.content).toEqual([
      { type: "text", text: "Let me check." },
      { type: "tool_use", id: "t1", name: "f", input: { q: 1 } },
    ]);
    expect(out.usage).toEqual({ inputTokens: 40, outputTokens: 12, cacheReadTokens: 30, cacheWriteTokens: 0 });
  });

  it("maps unknown stop reasons to other, known ones through", () => {
    const base = { content: [], usage: { input_tokens: 0, output_tokens: 0 }, model: "m" };
    expect(fromMessage({ ...base, stop_reason: "refusal" } as unknown as Anthropic.Message).stopReason).toBe("refusal");
    expect(fromMessage({ ...base, stop_reason: "pause_turn" } as unknown as Anthropic.Message).stopReason).toBe("other");
  });

  it("classifies SDK errors: 429/5xx/network retryable, 400 not", () => {
    const headers = new Headers();
    const rate = new Anthropic.RateLimitError(429, { type: "rate_limit_error" }, "slow down", headers);
    const server = new Anthropic.InternalServerError(500, { type: "api_error" }, "oops", headers);
    const bad = new Anthropic.BadRequestError(400, { type: "invalid_request_error" }, "nope", headers);
    const net = new Anthropic.APIConnectionError({ message: "ECONNRESET" });

    expect(toProviderError(rate)).toMatchObject({ retryable: true, status: 429 });
    expect(toProviderError(server)).toMatchObject({ retryable: true, status: 500 });
    expect(toProviderError(net)).toMatchObject({ retryable: true });
    expect(toProviderError(bad)).toMatchObject({ retryable: false, status: 400 });
    expect(toProviderError(new Error("x"))).toBeInstanceOf(ProviderError);
  });
});

describe("server-side refusal fallbacks", () => {
  /** One request through the stub: which endpoint it went to, and what it carried. */
  async function send(options: AnthropicProviderOptions = {}) {
    let params: SentParams | undefined;
    let endpoint: Endpoint | undefined;
    const provider = new AnthropicProvider({
      ...options,
      client: stubClient((sent, via) => {
        params = sent;
        endpoint = via;
      }),
    });
    await provider.complete({
      system: "rules",
      messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
      tools: [{ name: "f", description: "d", inputSchema: {} }],
    });
    return { endpoint, params: params! };
  }

  it("stays on the plain endpoint, asking for no fallback, by default", async () => {
    const { endpoint, params } = await send();
    expect(endpoint).toBe("messages");
    expect(params).not.toHaveProperty("fallbacks");
    expect(params).not.toHaveProperty("betas");
  });

  it("sends `fallbacks: \"default\"` under the one beta that gates that form", async () => {
    const { endpoint, params } = await send({ serverFallbacks: true });
    expect(endpoint).toBe("beta.messages");
    expect(params.fallbacks).toBe("default");
    // The array form has its own, earlier header; either header paired with the
    // other form is rejected, so the two travel together or not at all.
    expect(params.betas).toEqual(["server-side-fallback-2026-07-01"]);
    expect(SERVER_FALLBACK_BETA).toBe("server-side-fallback-2026-07-01");
  });

  it("leaves the rest of the request byte-identical, so the prefix still caches", async () => {
    const plain = await send({ cache: true, effort: "high" });
    const withFallback = await send({ cache: true, effort: "high", serverFallbacks: true });
    const { betas: _betas, fallbacks: _fallbacks, ...rest } = withFallback.params;
    expect(rest).toEqual(plain.params);
  });

  it("reports the substitute as the model that answered, and drops the fallback block", () => {
    const message = {
      model: "claude-opus-4-8",
      stop_reason: "end_turn",
      content: [
        { type: "fallback", from: { model: "claude-opus-5" }, to: { model: "claude-opus-4-8" } },
        { type: "text", text: "ok" },
      ],
      usage: { input_tokens: 5, output_tokens: 2 },
    } as unknown as Anthropic.Beta.BetaMessage;

    const out = fromMessage(message);
    // Cost is looked up per span on this field, so it has to be the substitute.
    expect(out.model).toBe("claude-opus-4-8");
    expect(out.content).toEqual([{ type: "text", text: "ok" }]);
  });
});
