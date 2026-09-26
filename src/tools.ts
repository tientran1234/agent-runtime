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
  if (!tool.strict) return { name: tool.name, description: tool.description, inputSchema: schema };

  const problems = unenforceable(schema, "(root)");
  if (problems.length > 0) {
    throw new Error(
      `tool ${tool.name} asked for strict but its schema cannot be enforced — ${problems.join("; ")}. ` +
        `Close every object and require every property: model an absent value as .nullable() rather than .optional(), ` +
        `and an open map as a fixed set of keys.`,
    );
  }
  return { name: tool.name, description: tool.description, inputSchema: schema, strict: true };
}

/**
 * Why this is a hard failure and not a silent downgrade: strict is a promise
 * made to `execute` that it will never see input the schema rejects. Sending
 * the tool without it would keep the run going while quietly breaking that
 * promise, so the definition is wrong and has to be fixed by the author.
 */
function unenforceable(node: unknown, path: string): string[] {
  if (node === null || typeof node !== "object" || Array.isArray(node)) return [];
  const schema = node as Record<string, unknown>;
  const problems: string[] = [];
  const properties = (schema.properties ?? {}) as Record<string, unknown>;
  const names = Object.keys(properties);

  if (schema.type === "object" || names.length > 0) {
    if (schema.additionalProperties !== false) problems.push(`${path}: additionalProperties must be false`);
    const required = Array.isArray(schema.required) ? (schema.required as unknown[]) : [];
    const missing = names.filter((name) => !required.includes(name));
    if (missing.length > 0) problems.push(`${path}: every property must be required, so ${missing.join(", ")} cannot be optional`);
  }

  for (const [name, child] of Object.entries(properties)) {
    problems.push(...unenforceable(child, path === "(root)" ? name : `${path}.${name}`));
  }
  // Objects also hide under array items, union branches and $defs references.
  for (const key of ["items", "additionalItems", "not"]) {
    if (key in schema) problems.push(...unenforceable(schema[key], `${path}.${key}`));
  }
  for (const key of ["anyOf", "oneOf", "allOf", "prefixItems"]) {
    const branches = schema[key];
    if (Array.isArray(branches)) branches.forEach((branch, i) => problems.push(...unenforceable(branch, `${path}.${key}[${i}]`)));
  }
  for (const key of ["$defs", "definitions"]) {
    const defs = schema[key];
    if (defs !== null && typeof defs === "object") {
      for (const [name, def] of Object.entries(defs as Record<string, unknown>)) {
        problems.push(...unenforceable(def, `${key}.${name}`));
      }
    }
  }
  return problems;
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
