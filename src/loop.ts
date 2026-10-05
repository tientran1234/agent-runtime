import type { z } from "zod";
import {
  executeTool,
  semaphore,
  toToolSpec,
  type BeforeToolCall,
  type Limit,
  type ToolContext,
  type ToolDecision,
  type ToolDefinition,
  type ToolOutcome,
} from "./tools.js";
import { BudgetLedger, type BudgetOptions } from "./budget.js";
import { clearToolUses } from "./context.js";
import type { ConversationMemory } from "./memory.js";
import { parseOutput, repairRequest, toOutputSchema } from "./output.js";
import { awaitingCalls, turnCalls, type SuspendedRun } from "./resume.js";
import type { Run, SpanHandle, SpanParent, Tracer } from "./trace.js";
import {
  EMPTY_USAGE,
  addUsage,
  textOf,
  type ChatMessage,
  type ContextEditing,
  type ModelProvider,
  type ModelResponse,
  type ToolResultPart,
  type ToolUsePart,
  type Usage,
} from "./types.js";

export type AgentEvent =
  | { type: "text_delta"; text: string }
  | { type: "model_call"; iteration: number; response: ModelResponse }
  | { type: "tool_call"; id: string; name: string; input: unknown }
  | { type: "tool_result"; id: string; name: string; content: string; isError: boolean; durationMs: number }
  // Widest result, so one event type covers a run of any output schema.
  | { type: "done"; result: AgentResult<unknown> };

export type AgentStatus =
  | "completed"
  | "max_iterations"
  | "refused"
  | "truncated"
  | "aborted"
  | "invalid_output"
  | "budget_exceeded"
  | "suspended";

export interface AgentResult<Output = never> {
  status: AgentStatus;
  /** Text of the final assistant message. */
  text: string;
  /** Full transcript for this run, including tool calls and results. */
  messages: ChatMessage[];
  iterations: number;
  usage: Usage;
  trace?: Run;
  /**
   * The final message parsed by the `output` schema. Set only on
   * `status: "completed"` of a run that asked for one — `never` otherwise, so
   * reading it without having asked is a type error rather than a surprise.
   */
  output?: Output;
  /**
   * Everything needed to finish this run elsewhere. Set only on
   * `status: "suspended"`, which is the one stop that is not an ending: the run
   * is waiting on a decision, not done, and `resumeAgent` takes it from here.
   */
  suspended?: SuspendedRun;
}

export interface AgentOptions<S extends z.ZodType = z.ZodNever> extends BudgetOptions {
  provider: ModelProvider;
  /** A string becomes the first user message. */
  input: string | ChatMessage[];
  system?: string;
  tools?: ToolDefinition[];
  /** Hard cap on model calls. Default 10. The loop has to end even if the model never says so. */
  maxIterations?: number;
  memory?: ConversationMemory;
  /**
   * Opt in to clearing old tool results out of what each call carries:
   * `{ clearToolUsesAfter: N }` sends the N most recent tool results in full
   * and the rest as a placeholder. `result.messages` still holds every one of
   * them, because what is edited is the copy of the conversation the call takes
   * — a trace, a resumed run and the caller all still see what the tools
   * returned.
   *
   * It is not `memory`, and the two compose: memory decides which turns are in
   * the window at all, this decides how much of the tool traffic inside that
   * window goes over in full. A provider that edits context on its own side is
   * asked to do it there; every other one has the loop do it.
   */
  contextEditing?: ContextEditing;
  tracer?: Tracer;
  runName?: string;
  /**
   * Hang this run's spans under a span that already exists, instead of starting
   * a run of its own. That is what a nested agent wants: one trace covers the
   * whole handoff, and the outer run's totals include what the nested calls
   * cost. A nested run reports no `result.trace` — the spans are the parent's
   * run, and the parent is the one that ends it — and `runName` has no run to
   * name, so the span's own name stands for it.
   */
  parentSpan?: SpanHandle;
  onEvent?: (event: AgentEvent) => void;
  signal?: AbortSignal;
  toolContext?: Omit<ToolContext, "signal">;
  /**
   * Approval gate, asked about every tool call once its input has validated and
   * before the tool runs. A denial becomes an error result the model can read
   * and work around, so one refused call does not end the run — abort `signal`
   * as well to stop there. `{ ask: true }` instead suspends the run.
   */
  beforeToolCall?: BeforeToolCall;
  /**
   * Schema the final answer has to match. The loop sends it to the provider to
   * constrain, validates what comes back, and gives the model one more turn
   * with the validation error if it does not fit. A second miss is
   * `status: "invalid_output"` — `result.output` is a parsed value or nothing,
   * never an unchecked one.
   */
  output?: S;
}

export interface ResumeOptions<S extends z.ZodType = z.ZodNever> extends Omit<AgentOptions<S>, "input"> {
  /** The snapshot a suspended run handed back. Its transcript stands in for `input`. */
  state: SuspendedRun;
  /**
   * The answer for every id in `state.awaiting`. A missing one throws rather
   * than defaulting either way: the run suspended precisely because nobody here
   * was entitled to decide, so inventing the answer now would undo the whole
   * exercise. `{ ask: true }` is a legal answer and suspends again, which makes
   * polling for a decision that has not arrived yet cost nothing but a read.
   */
  decisions: Record<string, ToolDecision>;
}

/**
 * The bounded tool loop: model → tools → model, until the model stops calling
 * tools or something says stop. Every stop condition is a named status, not
 * an exception, so the caller can tell "done" from "gave up" from "refused".
 */
export async function runAgent<S extends z.ZodType = z.ZodNever>(
  options: AgentOptions<S>,
): Promise<AgentResult<z.output<S>>> {
  const transcript: ChatMessage[] =
    typeof options.input === "string"
      ? [{ role: "user", content: [{ type: "text", text: options.input }] }]
      : options.input.slice();
  return loop(options, transcript, undefined);
}

/**
 * Pick a suspended run up where it stopped. The snapshot replaces `input`;
 * everything else — provider, tools, tracer, memory — is supplied fresh, because
 * none of it was serializable and all of it may have been redeployed since.
 *
 * The run continues rather than restarts: the awaited calls are the only ones
 * run from that turn, the iterations already spent still count against
 * `maxIterations`, and a budget resumes against what it had spent. The trace,
 * though, is a new run: a span tree cannot be stitched across a gap that may
 * have been a week, so the resumed run carries `resumedAfter` in its attributes
 * and `result.usage` is what spans the whole thing.
 */
export async function resumeAgent<S extends z.ZodType = z.ZodNever>(
  options: ResumeOptions<S>,
): Promise<AgentResult<z.output<S>>> {
  const { state, decisions } = options;
  if (state.version !== 1) throw new Error(`resumeAgent: cannot read a suspended run of version ${String(state.version)}`);
  // Reads the calls back out of the transcript, which also rejects a snapshot
  // that names one the transcript does not have.
  const pending = awaitingCalls(state);
  if (pending.length === 0) throw new Error("resumeAgent: this run is not waiting on anything, so there is nothing to resume");
  const missing = pending.filter((c) => decisions[c.toolUseId] === undefined);
  if (missing.length > 0) {
    throw new Error(
      `resumeAgent: no decision for ${missing.map((c) => `${c.name} (${c.toolUseId})`).join(", ")} — ` +
        `every awaiting call needs one, and { ask: true } is how you say the answer has not arrived yet.`,
    );
  }
  // A turn's results go back whole or not at all, so a snapshot that lost one of
  // its calls is caught here rather than as the provider's 400 three steps later.
  const accounted = new Set([...state.awaiting, ...state.settled.map((r) => r.toolUseId)]);
  const unaccounted = turnCalls(state).filter((c) => !accounted.has(c.id));
  if (unaccounted.length > 0) {
    throw new Error(
      `resumeAgent: this run neither awaits nor has a result for ${unaccounted.map((c) => `${c.name} (${c.id})`).join(", ")}, ` +
        `so the turn could only be resumed with a result missing.`,
    );
  }
  return loop({ ...options, input: state.messages }, state.messages.slice(), options);
}

async function loop<S extends z.ZodType>(
  options: AgentOptions<S>,
  transcript: ChatMessage[],
  resume: ResumeOptions<S> | undefined,
): Promise<AgentResult<z.output<S>>> {
  const tools = options.tools ?? [];
  const byName = new Map(tools.map((t) => [t.name, t]));
  const specs = tools.length > 0 ? tools.map(toToolSpec) : undefined;
  const maxIterations = options.maxIterations ?? 10;
  const emit = options.onEvent ?? (() => {});
  const gate = options.beforeToolCall;
  // One limiter per capped tool, scoped to this run — which is the scope the
  // parallelism has, since a turn's calls are the only ones ever in flight.
  const limits = new Map<string, Limit>();
  for (const tool of tools) {
    if (Number.isFinite(tool.maxConcurrency)) limits.set(tool.name, semaphore(tool.maxConcurrency));
  }
  const outputSchema = options.output ? toOutputSchema(options.output) : undefined;
  // One side clears or the other does, never both: a second pass would count
  // what the first left against a window that has already been applied.
  const serverEdit = options.provider.editsContext === true ? options.contextEditing : undefined;
  const localEdit = serverEdit ? undefined : options.contextEditing;
  // Only when a cap was asked for: an unbudgeted run should not pay to be
  // measured. The tracer's price table is what the cap is priced against, so a
  // deployment's own rates bind the budget as well as the trace.
  const budget =
    options.maxCostUsd !== undefined || options.maxInputTokens !== undefined
      ? new BudgetLedger(options, options.tracer?.prices, resume?.state.budget)
      : undefined;

  for (const m of transcript) options.memory?.append(m);

  const attributes = {
    provider: options.provider.name,
    model: options.provider.model,
    ...(resume ? { resumedAfter: resume.state.iterations } : {}),
  };
  // `owned` is the run this loop has to close; `run` is only where spans go.
  // They differ for a nested agent, which writes into a span somebody else
  // opened and must not end the run that span belongs to.
  const owned = options.parentSpan ? undefined : options.tracer?.startRun(options.runName ?? "agent", attributes);
  const run: SpanParent | undefined = options.parentSpan?.setAttributes(attributes) ?? owned;
  let usage = resume?.state.usage ?? EMPTY_USAGE;
  let iterations = resume?.state.iterations ?? 0;
  let output: { value: z.output<S> } | undefined;
  let repaired = resume?.state.repaired ?? false;

  const finish = async (status: AgentStatus, suspended?: SuspendedRun): Promise<AgentResult<z.output<S>>> => {
    const last = transcript[transcript.length - 1];
    const text = last?.role === "assistant" ? textOf(last.content) : "";
    const trace = owned ? await owned.end() : undefined;
    const result: AgentResult<z.output<S>> = {
      status,
      text,
      messages: transcript,
      iterations,
      usage,
      ...(trace ? { trace } : {}),
      ...(output ? { output: output.value } : {}),
      ...(suspended ? { suspended } : {}),
    };
    emit({ type: "done", result });
    return result;
  };

  /** Tool results in the order the model asked for the calls, which is the order they have to go back in. */
  const inCallOrder = (turn: readonly ToolUsePart[], parts: readonly ToolResultPart[]): ToolResultPart[] => {
    const byId = new Map(parts.map((p) => [p.toolUseId, p]));
    return turn.map((c) => byId.get(c.id)).filter((p): p is ToolResultPart => p !== undefined);
  };

  const snapshot = (turn: readonly ToolUsePart[], settled: readonly ToolResultPart[], awaiting: string[]): SuspendedRun => ({
    version: 1,
    messages: transcript,
    iterations,
    usage,
    repaired,
    ...(budget ? { budget: budget.state } : {}),
    awaiting,
    settled: inCallOrder(turn, settled),
  });

  /**
   * Run some of a turn's calls. `approverFor` is how the caller says who decides:
   * the run's own gate on a fresh turn, the stored answers on a resumed one.
   */
  const execute = async (
    calls: readonly ToolUsePart[],
    approverFor: (call: ToolUsePart, tool: ToolDefinition) => ((input: unknown) => Promise<ToolDecision> | ToolDecision) | undefined,
  ): Promise<Array<{ call: ToolUsePart; outcome: ToolOutcome }>> =>
    // All tool calls from one turn run concurrently, and ALL their results go
    // back in ONE user message. Splitting them across messages teaches the
    // model to stop making parallel calls.
    Promise.all(
      calls.map(async (call) => {
        emit({ type: "tool_call", id: call.id, name: call.name, input: call.input });
        const toolSpan = run?.startSpan("tool.call", call.name, { toolUseId: call.id });
        const tool = byName.get(call.name);
        const limit = limits.get(call.name);
        // A name with no tool behind it never reaches the gate: there is
        // nothing to approve, and the model needs the mistake back either way.
        const approve = tool ? approverFor(call, tool) : undefined;
        const outcome: ToolOutcome = tool
          ? await executeTool(
              tool,
              call.input,
              {
                ...options.toolContext,
                ...(options.signal ? { signal: options.signal } : {}),
                ...(toolSpan ? { span: toolSpan } : {}),
              },
              {
                ...(approve ? { approve } : {}),
                ...(limit ? { limit } : {}),
              },
            )
          : { content: `unknown tool: ${call.name}`, isError: true, durationMs: 0 };
        // An awaiting call did not run, so it is not a tool error and has no
        // result to report: a `tool_result` event for it would have a UI render
        // a failure where the truth is that nobody has answered yet.
        if (outcome.awaiting) toolSpan?.setAttributes({ awaiting: true, durationMs: outcome.durationMs }).end();
        else {
          toolSpan?.setAttributes({ isError: outcome.isError, durationMs: outcome.durationMs }).end(outcome.isError ? outcome.content : undefined);
          emit({ type: "tool_result", id: call.id, name: call.name, content: outcome.content, isError: outcome.isError, durationMs: outcome.durationMs });
        }
        return { call, outcome };
      }),
    );

  const resultsOf = (outcomes: Array<{ call: ToolUsePart; outcome: ToolOutcome }>): ToolResultPart[] =>
    outcomes
      .filter(({ outcome }) => !outcome.awaiting)
      .map(({ call, outcome }) => ({ type: "tool_result", toolUseId: call.id, content: outcome.content, isError: outcome.isError }));

  try {
    // The turn a suspension interrupted, finished first: only the calls that
    // were waiting are run, the rest are carried, and the run then rejoins the
    // loop at the model call that turn was always going to lead to.
    if (resume) {
      // Before the side effect, not after it: a run resumed under a signal that
      // is already aborted must not be the thing that fires the approved call.
      if (options.signal?.aborted) return finish("aborted");
      const turn = turnCalls(resume.state);
      const waiting = turn.filter((c) => resume.state.awaiting.includes(c.id));
      const outcomes = await execute(waiting, (call) => () => resume.decisions[call.id]!);
      const settled = [...resume.state.settled, ...resultsOf(outcomes)];
      const stillWaiting = outcomes.filter(({ outcome }) => outcome.awaiting).map(({ call }) => call.id);
      if (stillWaiting.length > 0) {
        run?.setAttributes({ suspended: describe(turn, stillWaiting) });
        return finish("suspended", snapshot(turn, settled, stillWaiting));
      }
      const user: ChatMessage = { role: "user", content: inCallOrder(turn, settled) };
      transcript.push(user);
      options.memory?.append(user);
    }

    while (iterations < maxIterations) {
      if (options.signal?.aborted) return finish("aborted");

      // Asked here, where the call can still be not made — a cap enforced after
      // the fact is a report, not a budget. The reason goes on the run because
      // the status says only that a limit was hit, and which one, with what
      // spent against it, is what a caller raising the cap needs.
      const overBudget = budget?.wouldExceed();
      if (overBudget !== undefined) {
        run?.setAttributes({ budget: overBudget });
        return finish("budget_exceeded");
      }
      iterations++;

      // A snapshot, so a provider that keeps the request (a fake, a logger)
      // does not see later turns appear inside it.
      const history = options.memory ? await options.memory.window() : transcript.slice();
      // Cleared on the way out and nowhere else: `transcript` is what the
      // caller is handed back, so the full results stay there.
      const messages = localEdit ? clearToolUses(history, localEdit) : history;
      const span = run?.startSpan("model.call", options.provider.model, { iteration: iterations });
      let response: ModelResponse;
      try {
        response = await options.provider.complete({
          ...(options.system !== undefined ? { system: options.system } : {}),
          messages,
          ...(specs ? { tools: specs } : {}),
          ...(outputSchema ? { outputSchema } : {}),
          ...(serverEdit ? { contextEditing: serverEdit } : {}),
          ...(options.signal ? { signal: options.signal } : {}),
          onTextDelta: (text) => emit({ type: "text_delta", text }),
        });
      } catch (err) {
        span?.end(err);
        throw err;
      }
      span?.recordUsage(response.model, response.usage).setAttributes({ stopReason: response.stopReason }).end();
      usage = addUsage(usage, response.usage);
      budget?.record(response);
      emit({ type: "model_call", iteration: iterations, response });

      const assistant: ChatMessage = { role: "assistant", content: response.content };
      transcript.push(assistant);
      options.memory?.append(assistant);

      // A refusal can cut a tool_use off mid-input; a max_tokens stop can leave
      // a tool input that parses but is incomplete. Never run those tools.
      if (response.stopReason === "refusal") return finish("refused");
      if (response.stopReason === "max_tokens") return finish("truncated");

      // A pause is not a stop: the provider interrupted a turn it is still
      // running server-side, and the turn continues when it is handed back — so
      // the loop appends nothing and calls again. Any tool_use in a paused turn
      // belongs to that unfinished turn and is not run here; it arrives for real
      // when the turn ends. Pauses cost an iteration like any other call, which
      // is what still bounds a model that only ever pauses.
      if (response.stopReason === "pause_turn") continue;

      const calls = response.content.filter((p): p is ToolUsePart => p.type === "tool_use");
      if (calls.length === 0) {
        if (!options.output) return finish("completed");
        const parsed = parseOutput(options.output, textOf(assistant.content));
        if (parsed.ok) {
          output = { value: parsed.value };
          return finish("completed");
        }
        // One repair round, not a loop: a model that missed the schema twice is
        // not going to be talked into it, and every further try is another bill
        // against an answer the caller cannot use. The miss is fed back as a
        // user message so the model sees what was wrong with what it sent, and
        // it costs an iteration like any other turn.
        if (repaired) return finish("invalid_output");
        repaired = true;
        const repair: ChatMessage = { role: "user", content: [{ type: "text", text: repairRequest(parsed.problem) }] };
        transcript.push(repair);
        options.memory?.append(repair);
        continue;
      }

      const outcomes = await execute(calls, (call, tool) => (gate ? (input: unknown) => gate({ tool, id: call.id, input }) : undefined));
      const awaiting = outcomes.filter(({ outcome }) => outcome.awaiting).map(({ call }) => call.id);
      // One undecided call suspends the whole turn, not just itself. Its
      // siblings have already run and their results are carried, because the
      // model may not have a turn's results arrive in two messages — and
      // because re-running them later is a second set of side effects.
      if (awaiting.length > 0) {
        run?.setAttributes({ suspended: describe(calls, awaiting) });
        return finish("suspended", snapshot(calls, resultsOf(outcomes), awaiting));
      }

      const user: ChatMessage = { role: "user", content: inCallOrder(calls, resultsOf(outcomes)) };
      transcript.push(user);
      options.memory?.append(user);
    }
    return finish("max_iterations");
  } catch (err) {
    await owned?.end(err);
    throw err;
  }
}

/** What the run records about a suspension: which calls are waiting, by name. */
function describe(turn: readonly ToolUsePart[], awaiting: readonly string[]): string {
  const names = new Map(turn.map((c) => [c.id, c.name]));
  return `awaiting a decision on ${awaiting.map((id) => `${names.get(id) ?? "?"} (${id})`).join(", ")}`;
}
