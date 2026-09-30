import { describe, expect, it } from "vitest";
import { z } from "zod";
import { parseOutput, repairRequest, toOutputSchema } from "../src/index.js";

const answer = z.object({ answer: z.number(), unit: z.string() });

describe("output schemas", () => {
  it("renders a Zod schema as JSON Schema without the dialect marker", () => {
    const schema = toOutputSchema(answer);
    expect(schema).not.toHaveProperty("$schema");
    expect(schema).toMatchObject({ type: "object", required: ["answer", "unit"] });
  });

  it("parses a final message the schema accepts", () => {
    expect(parseOutput(answer, '{"answer":22,"unit":"C"}')).toEqual({ ok: true, value: { answer: 22, unit: "C" } });
  });

  it("reports text that is not JSON at all as a problem, not an exception", () => {
    const parsed = parseOutput(answer, "It is 22 degrees.");
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain("not JSON");
  });

  it("names every field the schema rejected, by path", () => {
    const parsed = parseOutput(answer, '{"answer":"22"}');
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) {
      expect(parsed.problem).toContain("answer");
      expect(parsed.problem).toContain("unit");
    }
  });

  it("keeps a schema's own coercions and defaults, so the parsed value is the schema's", () => {
    const parsed = parseOutput(z.object({ n: z.coerce.number(), ok: z.boolean().default(true) }), '{"n":"7"}');
    expect(parsed).toEqual({ ok: true, value: { n: 7, ok: true } });
  });

  it("tells the model what was wrong and to send the value alone", () => {
    const asked = repairRequest("answer: expected number");
    expect(asked).toContain("answer: expected number");
    expect(asked).toMatch(/only the JSON value/);
  });
});
