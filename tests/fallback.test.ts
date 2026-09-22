import { describe, expect, it } from "vitest";
import { FakeProvider, FallbackProvider, ProviderError, reply } from "../src/index.js";

describe("FallbackProvider", () => {
  it("uses the first provider when it works", async () => {
    const a = new FakeProvider([reply("from a")], "model-a");
    const b = new FakeProvider([reply("from b")], "model-b");
    const res = await new FallbackProvider([a, b]).complete({ messages: [] });
    expect(res.model).toBe("model-a");
    expect(b.calls).toHaveLength(0);
  });

  it("moves to the next provider on a retryable error and reports who answered", async () => {
    const a = new FakeProvider([new ProviderError("overloaded", true, 529)], "model-a");
    const b = new FakeProvider([reply("from b")], "model-b");
    const hops: string[] = [];
    const chain = new FallbackProvider([a, b], { onFallback: (from, to) => hops.push(`${from.model}->${to.model}`) });
    const res = await chain.complete({ messages: [] });
    expect(res.model).toBe("model-b");
    expect(hops).toEqual(["model-a->model-b"]);
  });

  it("does not hide a non-retryable error behind another provider", async () => {
    const a = new FakeProvider([new ProviderError("bad request", false, 400)], "model-a");
    const b = new FakeProvider([reply("from b")], "model-b");
    await expect(new FallbackProvider([a, b]).complete({ messages: [] })).rejects.toMatchObject({ status: 400 });
    expect(b.calls).toHaveLength(0);
  });

  it("throws the last error when every provider fails", async () => {
    const a = new FakeProvider([new ProviderError("a down", true)], "a");
    const b = new FakeProvider([new ProviderError("b down", true)], "b");
    await expect(new FallbackProvider([a, b]).complete({ messages: [] })).rejects.toThrow("b down");
  });
});
