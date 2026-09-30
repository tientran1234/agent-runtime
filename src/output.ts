import { z } from "zod";

/**
 * A final answer the caller asked to be structured: the schema on its way to
 * the provider, the text on its way back, and the one repair round in between.
 *
 * The provider is asked to constrain its decoding, but the promise to the
 * caller is made here — a provider that ignores the schema, or one that honours
 * it and still drifts, is caught by the same parse.
 */
export function toOutputSchema(schema: z.ZodType): Record<string, unknown> {
  // `$schema` describes the dialect rather than the value, and no provider's
  // format field takes it.
  const { $schema: _dropped, ...json } = z.toJSONSchema(schema) as Record<string, unknown>;
  return json;
}

export type OutputParse<Output> = { ok: true; value: Output } | { ok: false; problem: string };

/**
 * The final message as the schema's value. Both failures — text that is not
 * JSON at all and JSON the schema rejects — come back as a `problem` rather
 * than an exception, because the model is the one that has to fix it.
 */
export function parseOutput<S extends z.ZodType>(schema: S, text: string): OutputParse<z.output<S>> {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (err) {
    return { ok: false, problem: `it is not JSON: ${err instanceof Error ? err.message : String(err)}` };
  }
  const parsed = schema.safeParse(json);
  if (parsed.success) return { ok: true, value: parsed.data };
  return {
    ok: false,
    problem: parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; "),
  };
}

/**
 * What the model is told after an answer the schema rejected. It names the
 * problem, because a bare "try again" is how a model repeats itself, and it
 * says to send the value alone — prose around it, or a fenced block, is the
 * most common way a final message stops being JSON.
 */
export function repairRequest(problem: string): string {
  return (
    `Your last message does not match the required output format: ${problem}. ` +
    `Reply with only the JSON value the schema describes — no prose, no code fence.`
  );
}
