import Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";
import { fromMessage, toMessageParams, toProviderError, toTools } from "../src/providers/anthropic.js";
import { ProviderError } from "../src/index.js";

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
