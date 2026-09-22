import type { Usage } from "./types.js";

/** USD per million tokens. */
export interface Price {
  input: number;
  output: number;
  /** Defaults to 10% of input. */
  cacheRead?: number;
  /** Defaults to 125% of input. */
  cacheWrite?: number;
}

/** Anthropic first-party list prices at the time of writing. Override per deployment. */
export const PRICES: Record<string, Price> = {
  "claude-fable-5-1": { input: 10, output: 50 },
  "claude-fable-5": { input: 10, output: 50 },
  "claude-opus-5": { input: 5, output: 25 },
  "claude-opus-4-8": { input: 5, output: 25 },
  "claude-opus-4-7": { input: 5, output: 25 },
  "claude-opus-4-6": { input: 5, output: 25 },
  "claude-sonnet-5": { input: 2, output: 10 },
  "claude-sonnet-4-6": { input: 3, output: 15 },
  "claude-haiku-4-5": { input: 1, output: 5 },
};

/** Cost of one call, or null when the model is not in the table — never a silent 0. */
export function costUsd(model: string, usage: Usage, table: Record<string, Price> = PRICES): number | null {
  const price = table[model];
  if (!price) return null;
  const perToken = (perMillion: number) => perMillion / 1_000_000;
  const cacheRead = price.cacheRead ?? price.input * 0.1;
  const cacheWrite = price.cacheWrite ?? price.input * 1.25;
  return (
    usage.inputTokens * perToken(price.input) +
    usage.outputTokens * perToken(price.output) +
    usage.cacheReadTokens * perToken(cacheRead) +
    usage.cacheWriteTokens * perToken(cacheWrite)
  );
}
