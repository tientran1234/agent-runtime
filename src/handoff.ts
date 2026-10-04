import { z } from "zod";
import { runAgent, type AgentOptions } from "./loop.js";
import type { ConversationMemory } from "./memory.js";
import { defineTool, type ToolDefinition } from "./tools.js";

/**
 * A sub-agent as the parent sees it: a tool with a name, a description and one
 * string of a brief. Everything else is the run it starts — its own provider,
 * its own system prompt, its own tools, its own caps.
 *
 * These options are read once, when the tool is defined, and every call of it
 * runs from them. That is why `memory` is a factory and not an instance: a
 * `ConversationMemory` holds one conversation, and sharing it would feed each
 * sub-agent the last one's transcript.
 */
export interface HandoffOptions<S extends z.ZodType = z.ZodNever>
  extends Omit<AgentOptions<S>, "input" | "memory" | "parentSpan"> {
  name: string;
  description: string;
  /** Fresh memory for each handoff, if the sub-agent needs a trimming window. */
  memory?: () => ConversationMemory;
  /**
   * Default 120 s — a whole nested run, not one call, so the tool default of
   * 30 s would cut off sub-agents that were working fine. It still bounds the
   * handoff: `maxIterations` bounds the calls, this bounds the wall clock.
   */
  timeoutMs?: number;
  maxResultChars?: number;
  strict?: boolean;
  maxConcurrency?: number;
}

const brief = z.object({
  task: z
    .string()
    .describe(
      "Everything the sub-agent needs, written out in full. It does not see this conversation, " +
        "the files it mentions, or any earlier tool result.",
    ),
});

/**
 * A tool that hands a task to a nested agent and returns its final text.
 *
 * The brief is the whole interface: the sub-agent starts from an empty
 * transcript, so the parent model has to say what it wants rather than point at
 * context the sub-agent cannot see. In exchange the sub-agent's own tools,
 * prompt and model stay out of the parent's context window entirely — which is
 * the reason to hand off rather than add twenty more tools to one loop.
 *
 * Traced as one call: the nested run's model and tool spans hang under this
 * tool's span, in the parent's run, so its tokens and cost are part of what the
 * parent spent. Where the parent is not traced the sub-agent falls back to its
 * own `tracer`, if it was given one.
 *
 * Only `status: "completed"` is an answer. Every other stop comes back as an
 * error result, because a parent model handed a sub-agent's half-finished text
 * with no word of the status would read it as the answer.
 */
export function handoffTool<S extends z.ZodType = z.ZodNever>(
  options: HandoffOptions<S>,
): ToolDefinition<z.output<typeof brief>> {
  const { name, description, memory, timeoutMs, maxResultChars, strict, maxConcurrency, ...agent } = options;
  return defineTool({
    name,
    description,
    input: brief,
    timeoutMs: timeoutMs ?? 120_000,
    ...(maxResultChars !== undefined ? { maxResultChars } : {}),
    ...(strict !== undefined ? { strict } : {}),
    ...(maxConcurrency !== undefined ? { maxConcurrency } : {}),
    execute: async ({ task }, ctx) => {
      const result = await runAgent({
        ...agent,
        input: task,
        ...(memory ? { memory: memory() } : {}),
        // The live run's signal, not the one the options were defined with: an
        // aborted parent must take its sub-agents down with it.
        ...(ctx.signal ? { signal: ctx.signal } : {}),
        ...(ctx.span ? { parentSpan: ctx.span } : {}),
      });
      // On the tool's own span, so a trace says what the handoff came back with
      // without having to read the nested spans to work it out.
      ctx.span?.setAttributes({ handoffStatus: result.status, handoffIterations: result.iterations });

      // A nested gate that answers `{ ask: true }` is the one stop that cannot
      // be passed up: `SuspendedRun` holds one loop's state, and the parent is
      // mid-turn in a loop of its own. Said plainly here, because the fix is a
      // design change and not something the model can work around.
      if (result.status === "suspended") {
        throw new Error(
          `sub-agent ${name} suspended on an undecided approval gate, which a handoff cannot carry: ` +
            `a parent's snapshot has no room for a nested run. Decide inside the sub-agent's gate, ` +
            `or gate this handoff in the parent and suspend there.`,
        );
      }
      if (result.status !== "completed") {
        throw new Error(
          `sub-agent ${name} stopped with status ${result.status}` +
            (result.text ? `, having got as far as: ${result.text}` : " and said nothing"),
        );
      }
      // An empty tool_result is rejected outright by some providers, so a
      // sub-agent that finished without an answer is reported as one that did.
      if (result.text === "") throw new Error(`sub-agent ${name} completed without returning any text`);
      return result.text;
    },
  });
}
