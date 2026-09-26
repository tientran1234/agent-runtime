import { z } from "zod";
import type { ToolSpec } from "./types.js";

export interface ToolContext {
  signal?: AbortSignal;
  /** Free-form; the loop passes through whatever the caller supplied. */
  meta?: Record<string, unknown>;
}

export interface ToolDefinition<Input = unknown> {
  name: string;
  description: string;
  /** Validates the model's input before `execute` ever sees it. */
  schema: z.ZodType;
  // Method syntax on purpose: it makes ToolDefinition<{a: number}> assignable
  // to ToolDefinition<unknown>, so a typed tool fits an untyped tools list.
  execute(input: Input, ctx: ToolContext): Promise<unknown> | unknown;
  /** Default 30s. A hung tool must not hang the agent. */
  timeoutMs: number;
  /** Default 16 000 characters. A tool that returns a 2 MB file must not eat the context window. */
  maxResultChars: number;
  /** Default false. Have the provider enforce the schema, not just describe it. */
  strict: boolean;
}

export function defineTool<S extends z.ZodType>(definition: {
  name: string;
  description: string;
  input: S;
  execute: (input: z.output<S>, ctx: ToolContext) => Promise<unknown> | unknown;
  timeoutMs?: number;
  maxResultChars?: number;
  /**
   * Ask the provider to guarantee that `execute` only ever sees input this
   * schema accepts. The schema has to be enforceable to qualify — see
   * `toToolSpec`, which is where an unenforceable one is rejected.
   */
  strict?: boolean;
}): ToolDefinition<z.output<S>> {
  return {
    name: definition.name,
    description: definition.description,
    schema: definition.input,
    execute: definition.execute,
    timeoutMs: definition.timeoutMs ?? 30_000,
    maxResultChars: definition.maxResultChars ?? 16_000,
    strict: definition.strict ?? false,
  };
}

/** What gets sent to the model. */
export function toToolSpec(tool: ToolDefinition): ToolSpec {
  const { $schema: _dropped, ...schema } = z.toJSONSchema(tool.schema) as Record<string, unknown>;
  return { name: tool.name, description: tool.description, inputSchema: schema };
}

export interface ToolOutcome {
  content: string;
  isError: boolean;
  durationMs: number;
}

/**
 * Run one tool call the way the model needs it run: validated input, bounded
 * time, bounded output, and every failure turned into an error *result* rather
 * than an exception — the model is the one that has to decide what to do next,
 * so it must see what went wrong.
 */
export async function executeTool(
  tool: ToolDefinition,
  rawInput: unknown,
  ctx: ToolContext = {},
  now: () => number = Date.now,
): Promise<ToolOutcome> {
  const started = now();
  const done = (content: string, isError: boolean): ToolOutcome => ({
    content: truncate(content, tool.maxResultChars),
    isError,
    durationMs: now() - started,
  });

  // The SDK's tolerant JSON parser can hand back a silently truncated object;
  // the schema is the last line of defence before real side effects.
  const parsed = tool.schema.safeParse(rawInput);
  if (!parsed.success) {
    return done(`invalid input for ${tool.name}: ${parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; ")}`, true);
  }

  try {
    const result = await withTimeout(Promise.resolve(tool.execute(parsed.data, ctx)), tool.timeoutMs, tool.name);
    return done(typeof result === "string" ? result : JSON.stringify(result ?? null), false);
  } catch (err) {
    return done(err instanceof Error ? err.message : String(err), true);
  }
}

function withTimeout<T>(promise: Promise<T>, ms: number, name: string): Promise<T> {
  let timer: NodeJS.Timeout;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`tool ${name} timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  const dropped = text.length - max;
  return `${text.slice(0, max)}\n…[truncated ${dropped} characters]`;
}
