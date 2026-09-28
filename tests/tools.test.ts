import { describe, expect, it } from "vitest";
import { z } from "zod";
import { defineTool, executeTool, toToolSpec } from "../src/index.js";

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

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

describe("strict tool specs", () => {
  it("asks for enforcement and keeps the schema closed", () => {
    const spec = toToolSpec(
      defineTool({
        name: "grade",
        description: "Grade an answer",
        input: z.object({ score: z.number(), notes: z.string().nullable() }),
        strict: true,
        execute: () => "ok",
      }),
    );
    expect(spec.strict).toBe(true);
    expect(spec.inputSchema).toMatchObject({ additionalProperties: false, required: ["score", "notes"] });
  });

  it("says nothing about strictness for a plain tool, so the request shape is unchanged", () => {
    expect(toToolSpec(add)).not.toHaveProperty("strict");
  });

  it("refuses an optional property instead of letting the provider reject the tool", () => {
    const optional = defineTool({
      name: "search",
      description: "",
      input: z.object({ q: z.string(), limit: z.number().optional() }),
      strict: true,
      execute: () => "ok",
    });
    expect(() => toToolSpec(optional)).toThrow(/tool search .*\(root\): .*required.*limit/s);
  });

  it("refuses an open object nested inside the schema, naming where it is", () => {
    const nested = defineTool({
      name: "annotate",
      description: "",
      input: z.object({ tags: z.record(z.string(), z.string()) }),
      strict: true,
      execute: () => "ok",
    });
    expect(() => toToolSpec(nested)).toThrow(/tags: additionalProperties/);
  });

  it("accepts a schema with no properties at all — there is nothing to leave out", () => {
    const now = defineTool({ name: "now", description: "", input: z.object({}), strict: true, execute: () => "ok" });
    expect(toToolSpec(now).strict).toBe(true);
  });
});

describe("approval gates", () => {
  const deploy = (onRun: () => void) =>
    defineTool({ name: "deploy", description: "Ship the current build", input: z.object({}), execute: onRun });

  it("asks the gate only once the input validates, so nobody approves a payload the tool would reject", async () => {
    const asked: unknown[] = [];
    const out = await executeTool(add, { a: 1, b: "two" }, {}, { approve: (input) => (asked.push(input), { allow: true }) });
    expect(asked).toEqual([]);
    expect(out.content).toMatch(/invalid input for add/);
  });

  it("shows the gate the validated input, which is what execute will get", async () => {
    const coerce = defineTool({ name: "coerce", description: "", input: z.object({ a: z.coerce.number() }), execute: ({ a }) => a * 2 });
    const asked: unknown[] = [];
    const out = await executeTool(coerce, { a: "21" }, {}, { approve: (input) => (asked.push(input), { allow: true }) });
    expect(asked).toEqual([{ a: 21 }]);
    expect(out.content).toBe("42");
  });

  it("does not run a denied tool and tells the model why", async () => {
    let ran = 0;
    const out = await executeTool(deploy(() => void ran++), {}, {}, {
      approve: () => ({ allow: false, reason: "the on-call engineer said no" }),
    });
    expect(ran).toBe(0);
    expect(out).toMatchObject({ isError: true, content: "tool deploy was not approved: the on-call engineer said no" });
  });

  it("denies when the gate itself throws, rather than falling open", async () => {
    let ran = 0;
    const out = await executeTool(deploy(() => void ran++), {}, {}, {
      approve: () => {
        throw new Error("approval service down");
      },
    });
    expect(ran).toBe(0);
    expect(out.isError).toBe(true);
    expect(out.content).toMatch(/tool deploy was not approved: .*approval service down/);
  });

  it("does not spend the tool's timeout waiting for the gate", async () => {
    const quick = defineTool({ name: "quick", description: "", input: z.object({}), timeoutMs: 30, execute: () => "ok" });
    const out = await executeTool(quick, {}, {}, { approve: async () => (await sleep(80), { allow: true }) });
    expect(out).toMatchObject({ isError: false, content: "ok" });
  });
});

describe("per-tool concurrency", () => {
  it("refuses a limit below one, which would be a tool that can never run", () => {
    expect(() => defineTool({ name: "stuck", description: "", input: z.object({}), maxConcurrency: 0, execute: () => "" })).toThrow(
      /tool stuck: maxConcurrency must be at least 1/,
    );
  });

  it("leaves a tool unlimited by default", () => {
    expect(add.maxConcurrency).toBe(Infinity);
  });
});
