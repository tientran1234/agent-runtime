import type { BudgetState } from "./budget.js";
import type { ChatMessage, ToolResultPart, ToolUsePart, Usage } from "./types.js";

/**
 * A run stopped mid-turn because an approval gate answered `{ ask: true }`, in
 * plain JSON: write it to a row or a workflow step's output, and `resumeAgent`
 * finishes the run in another process, on another machine, next week.
 *
 * What it deliberately does not carry is the provider, the tools, the tracer and
 * the memory. Those are code, and a snapshot holding them would only be readable
 * by the process that wrote it — which is the opposite of the point. Resuming
 * supplies them again, so a stored snapshot also never pins a tool to the
 * implementation that happened to suspend.
 */
export interface SuspendedRun {
  /** The shape of this record, for a reader that may be older than the writer. */
  version: 1;
  /**
   * The transcript through the assistant turn whose tools were asked about. It
   * stops there on purpose: that turn's results are not all in yet, and a
   * half-filled `tool_result` message is one every provider rejects.
   */
  messages: ChatMessage[];
  /** Model calls already spent, so `maxIterations` bounds the whole run and not each half of it. */
  iterations: number;
  /** Tokens already spent, so the finished run reports what it cost rather than what the last leg cost. */
  usage: Usage;
  /** Whether the one repair round has been used, so a resumed run cannot buy a second. */
  repaired: boolean;
  /** What a budget has spent and what it forecasts from, so a cap keeps binding across the suspension. */
  budget?: BudgetState;
  /**
   * The `tool_use` ids of that turn still waiting on a decision. Their names and
   * inputs are not repeated here — the model already put them in `messages`, and
   * `awaitingCalls` reads them back out.
   */
  awaiting: string[];
  /**
   * Results from the same turn that are already final: the calls that ran before
   * the gate asked, and the ones it refused. Carried rather than recomputed,
   * because running a tool that has already had its side effect is the one thing
   * a resume must never do.
   */
  settled: ToolResultPart[];
}

/** A call a snapshot is waiting on, as the model asked for it. */
export interface AwaitingCall {
  toolUseId: string;
  name: string;
  /**
   * The input the model sent, untouched — not the validated value the gate was
   * shown. A resumed run validates again, so the raw input is the only version
   * that is certainly still true of a schema that may have moved on since.
   */
  input: unknown;
}

/**
 * What a snapshot is waiting on, for whoever has to collect the answers — an
 * approval queue, an inbox, a workflow signal. Reading it from the transcript
 * rather than from a second copy in the snapshot is what keeps the two from
 * disagreeing about the input a human was shown.
 */
export function awaitingCalls(state: SuspendedRun): AwaitingCall[] {
  const byId = new Map(turnCalls(state).map((c) => [c.id, c]));
  return state.awaiting.map((id) => {
    const call = byId.get(id);
    // A snapshot naming a call its own transcript does not have is corrupt, and
    // the damage of guessing is a tool call nobody can show was approved.
    if (!call) throw new Error(`suspended run: awaiting ${id}, which the last assistant message never called`);
    return { toolUseId: id, name: call.name, input: call.input };
  });
}

/**
 * The tool calls of the turn the snapshot stopped inside, in the order the model
 * asked for them — which is the order their results have to go back in.
 */
export function turnCalls(state: SuspendedRun): ToolUsePart[] {
  const last = state.messages[state.messages.length - 1];
  return last?.role === "assistant" ? last.content.filter((p): p is ToolUsePart => p.type === "tool_use") : [];
}
