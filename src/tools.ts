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
  /** Default unlimited. Caps how many calls of this tool are in flight at once. */
  maxConcurrency: number;
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
  /**
   * Cap on how many calls of this tool run at once, for a downstream that
   * cannot take a whole turn's worth of parallel calls. Default unlimited.
   */
  maxConcurrency?: number;
}): ToolDefinition<z.output<S>> {
  const maxConcurrency = definition.maxConcurrency ?? Infinity;
  // Caught here rather than mid-run, because a limit of zero is not a slow
  // tool, it is a tool that can never run — and that is an authoring mistake.
  if (!(maxConcurrency >= 1)) {
    throw new Error(`tool ${definition.name}: maxConcurrency must be at least 1, got ${String(definition.maxConcurrency)}`);
  }
  return {
    name: definition.name,
    description: definition.description,
    schema: definition.input,
    execute: definition.execute,
    timeoutMs: definition.timeoutMs ?? 30_000,
    maxResultChars: definition.maxResultChars ?? 16_000,
    strict: definition.strict ?? false,
    maxConcurrency,
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

/** A tool call that has validated but not run yet — what an approval gate judges. */
export interface PendingToolCall {
  /** The tool the call names. Its description is what an approval UI shows a human. */
  tool: ToolDefinition;
  /** The model's id for the call, so a decision can be traced back to the transcript. */
  id: string;
  /** Exactly the input `execute` receives if this call is allowed. */
  input: unknown;
}

export type ToolDecision = { allow: true } | { allow: false; reason?: string };

/**
 * Asked about every tool call before it runs. There is no implicit allow: the
 * hook has to answer, so a branch that forgets to cannot let a side effect
 * through, and a hook that throws denies rather than falling open.
 */
export type BeforeToolCall = (call: PendingToolCall) => Promise<ToolDecision> | ToolDecision;

/** Runs `fn` once whatever it guards has room for it. */
export type Limit = <T>(fn: () => Promise<T>) => Promise<T>;

export interface ExecuteOptions {
  /**
   * Asked to approve the validated input. `runAgent` binds the rest of the
   * call's identity here from its `beforeToolCall`; a direct caller, having no
   * tool_use id to bind, supplies whatever it knows.
   */
  approve?: (input: unknown) => Promise<ToolDecision> | ToolDecision;
  /** Wraps the execution, so neither a queue nor a waiting human sits inside `timeoutMs`. */
  limit?: Limit;
  /** Injectable clock, for tests. */
  now?: () => number;
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
  options: ExecuteOptions = {},
): Promise<ToolOutcome> {
  const now = options.now ?? Date.now;
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

  // Approval sits between validation and the side effect: after it, because
  // whoever approves must see the input `execute` will really get; before it,
  // because afterwards there is nothing left to approve.
  if (options.approve) {
    let decision: ToolDecision;
    try {
      decision = await options.approve(parsed.data);
    } catch (err) {
      // Fail closed. A gate that breaks must not become an open door.
      return done(`tool ${tool.name} was not approved: the approval check failed: ${messageOf(err)}`, true);
    }
    if (!decision.allow) {
      return done(`tool ${tool.name} was not approved${decision.reason ? `: ${decision.reason}` : ""}`, true);
    }
  }

  const run = () => withTimeout(Promise.resolve(tool.execute(parsed.data, ctx)), tool.timeoutMs, tool.name);
  try {
    // `timeoutMs` is the tool's own clock, so neither the wait behind a
    // concurrency cap nor the wait on a human is inside it — a call must not
    // expire for queueing. `durationMs` still spans the whole thing, so a trace
    // reports the latency the caller actually saw.
    const result = await (options.limit ? options.limit(run) : run());
    return done(typeof result === "string" ? result : JSON.stringify(result ?? null), false);
  } catch (err) {
    return done(messageOf(err), true);
  }
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * At most `max` calls in flight; the rest wait in the order they arrived, since
 * that is the order the model asked for them in. A finishing call hands its slot
 * straight to the next waiter instead of releasing it and letting the waiter
 * re-take it — the gap between those two is where a caller arriving in between
 * would slip past the cap.
 */
export function semaphore(max: number): Limit {
  if (!(max >= 1)) throw new Error(`semaphore needs a max of at least 1, got ${String(max)}`);
  let active = 0;
  const waiting: Array<() => void> = [];
  return async <T>(fn: () => Promise<T>): Promise<T> => {
    if (active >= max) await new Promise<void>((resolve) => waiting.push(resolve));
    else active++;
    try {
      return await fn();
    } finally {
      const next = waiting.shift();
      if (next) next();
      else active--;
    }
  };
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
