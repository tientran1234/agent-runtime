import type { AgentResult, AgentStatus } from "./loop.js";
import type { ConversationMemory } from "./memory.js";
import type { BeforeToolCall, ToolDefinition } from "./tools.js";
import type { ChatMessage, ModelRequest, ModelResponse } from "./types.js";

/**
 * What the model answers on one call. A function sees the request, so a turn
 * can answer according to what the loop actually sent.
 *
 * No `Error` branch, unlike `FakeProvider`'s own script: a provider error
 * throws out of the loop and leaves no transcript to compare, so it is not a
 * shape a scenario can describe.
 */
export type ScriptedTurn = ModelResponse | ((request: ModelRequest) => ModelResponse);

/**
 * One scripted run of the loop, plus the shape it has to produce. A suite of
 * these is what makes a system prompt safe to edit: the script pins what the
 * model does, so the only thing left that can move the transcript is the
 * prompt, the tool list, or the loop itself.
 */
export interface Scenario {
  /** Names the scenario in a failure. */
  name: string;
  /**
   * The prompt under test. Every model call is checked to have received it
   * unchanged — a transcript that looks right for a prompt the model never saw
   * is exactly the false pass a prompt suite must not give.
   */
  system?: string;
  input: string | ChatMessage[];
  tools?: ToolDefinition[];
  /** The model's answers, in order. All of them have to be used. */
  script: ScriptedTurn[];
  maxIterations?: number;
  beforeToolCall?: BeforeToolCall;
  /** A factory, not an instance: memory accumulates, and a scenario is re-run. */
  memory?: () => ConversationMemory;
  expect: {
    status: AgentStatus;
    /** The transcript as `renderTranscript` prints it, one line per message. */
    transcript: string[];
  };
}

export interface ScenarioReport {
  name: string;
  ok: boolean;
  /** One entry per mismatch, each already formatted for printing. Empty when `ok`. */
  failures: string[];
  /** What this run produced — what `expect.transcript` should become once a change is intended. */
  transcript: string[];
  result: AgentResult;
  /** Every request the loop sent, for the checks a shape does not cover. */
  requests: readonly ModelRequest[];
}

/**
 * The transcript reduced to the shape a prompt change can move: one line per
 * message, roles in order, tool calls and results by tool name.
 *
 * Payloads are left out on purpose. The script decides what the model says, so
 * asserting its wording would only restate the script; what a prompt edit can
 * really change is which tools get called, in what order, and how many turns it
 * takes. Tool-use ids are dropped for the same reason from the other side —
 * they are generated per run, so a line carrying one would differ every time.
 * A result is named after the call it answers, which the transcript itself is
 * the only place to look up.
 */
export function renderTranscript(messages: readonly ChatMessage[]): string[] {
  const calledTool = new Map<string, string>();
  return messages.map((message) => {
    const parts = message.content.map((part) => {
      switch (part.type) {
        case "text":
          return "text";
        case "tool_use":
          calledTool.set(part.id, part.name);
          return `tool_use(${part.name})`;
        case "tool_result":
          return `tool_result(${calledTool.get(part.toolUseId) ?? "?"} ${part.isError ? "error" : "ok"})`;
      }
    });
    // A refusal or a max_tokens stop can leave a message with no content at
    // all, and an absent line would read as an absent message.
    return `${message.role}: ${parts.length > 0 ? parts.join(" ") : "(empty)"}`;
  });
}
