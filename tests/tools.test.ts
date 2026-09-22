import { describe, expect, it } from "vitest";
import { z } from "zod";
import { defineTool, executeTool, toToolSpec } from "../src/index.js";

const add = defineTool({
  name: "add",
  description: "Add two numbers",
  input: z.object({ a: z.number(), b: z.number() }),
  execute: ({ a, b }) => a + b,
});

describe("tool spec", () => {
  it("emits a JSON Schema the model can read, without the $schema noise", () => {
    const spec = toToolSpec(add);
    expect(spec.name).toBe("add");
    expect(spec.inputSchema).toMatchObject({ type: "object", required: ["a", "b"] });
    expect(spec.inputSchema).not.toHaveProperty("$schema");
  });
});

describe("executeTool", () => {
  it("validates input and returns an error result instead of running on garbage", async () => {
    const out = await executeTool(add, { a: 1, b: "two" });
    expect(out.isError).toBe(true);
    expect(out.content).toMatch(/invalid input for add: b:/);
  });

  it("stringifies non-string results as JSON", async () => {
    const obj = defineTool({ name: "obj", description: "", input: z.object({}), execute: () => ({ ok: true, n: 3 }) });
    expect((await executeTool(obj, {})).content).toBe('{"ok":true,"n":3}');
    expect((await executeTool(add, { a: 2, b: 3 })).content).toBe("5");
  });

  it("turns a thrown error into an error result the model can read", async () => {
    const boom = defineTool({ name: "boom", description: "", input: z.object({}), execute: () => { throw new Error("db down"); } });
    expect(await executeTool(boom, {})).toMatchObject({ isError: true, content: "db down" });
  });

  it("times out a hung tool", async () => {
    const hang = defineTool({ name: "hang", description: "", input: z.object({}), timeoutMs: 20, execute: () => new Promise(() => {}) });
    const out = await executeTool(hang, {});
    expect(out.isError).toBe(true);
    expect(out.content).toMatch(/timed out after 20ms/);
  });

  it("truncates oversized output and says how much was cut", async () => {
    const big = defineTool({ name: "big", description: "", input: z.object({}), maxResultChars: 100, execute: () => "x".repeat(1_000) });
    const out = await executeTool(big, {});
    expect(out.content.length).toBeLessThan(160);
    expect(out.content).toMatch(/\[truncated 900 characters\]/);
    expect(out.isError).toBe(false);
  });
});
