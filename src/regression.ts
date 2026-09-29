import { runAgent, type AgentResult, type AgentStatus } from "./loop.js";
import type { ConversationMemory } from "./memory.js";
import { FakeProvider } from "./providers/fake.js";
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

/**
 * Run one scenario and report every way it differs from its expectation. It
 * does not throw on a mismatch — a suite wants all of them at once — but a
 * provider error from the script still leaves the loop, because that is the
 * loop's contract and not a shape.
 */
export async function runScenario(scenario: Scenario): Promise<ScenarioReport> {
  // A copy: FakeProvider consumes the script it is given, and a scenario is a
  // declaration that stays runnable.
  const provider = new FakeProvider(scenario.script.slice());
  const memory = scenario.memory?.();
  const result = await runAgent({
    provider,
    input: scenario.input,
    ...(scenario.system !== undefined ? { system: scenario.system } : {}),
    ...(scenario.tools ? { tools: scenario.tools } : {}),
    ...(scenario.maxIterations !== undefined ? { maxIterations: scenario.maxIterations } : {}),
    ...(scenario.beforeToolCall ? { beforeToolCall: scenario.beforeToolCall } : {}),
    ...(memory ? { memory } : {}),
  });

  const transcript = renderTranscript(result.messages);
  const failures: string[] = [];
  if (result.status !== scenario.expect.status) {
    failures.push(`status: expected ${scenario.expect.status}, got ${result.status}`);
  }
  if (!same(scenario.expect.transcript, transcript)) {
    failures.push(`transcript:\n${diff(scenario.expect.transcript, transcript).join("\n")}`);
  }
  failures.push(...requestFailures(scenario, provider.calls));

  return { name: scenario.name, ok: failures.length === 0, failures, transcript, result, requests: provider.calls };
}

/**
 * `runScenario`, as an assertion. Throws with every mismatch and with the
 * transcript the run produced, so an intended change is a paste rather than a
 * second run to find out what the new shape was.
 */
export async function assertScenario(scenario: Scenario): Promise<ScenarioReport> {
  const report = await runScenario(scenario);
  if (!report.ok) {
    throw new Error(
      `scenario ${JSON.stringify(scenario.name)} does not match:\n\n${report.failures.join("\n\n")}\n\n` +
        `If the change is intended, this is the transcript to expect:\n${report.transcript.map((line) => `  ${JSON.stringify(line)},`).join("\n")}`,
    );
  }
  return report;
}

/**
 * What the transcript cannot show. The prompt and the tool list are inputs to
 * every call rather than messages, so a run that dropped either one still
 * produces the transcript the scenario expects — and the scenario would be
 * asserting the loop's shape for a prompt that was never in play.
 */
function requestFailures(scenario: Scenario, requests: readonly ModelRequest[]): string[] {
  const failures: string[] = [];
  const expected = scenario.tools?.map((tool) => tool.name) ?? [];
  requests.forEach((request, i) => {
    if (request.system !== scenario.system) {
      failures.push(`model call ${i + 1}: expected system prompt ${show(scenario.system)}, got ${show(request.system)}`);
    }
    const offered = request.tools?.map((spec) => spec.name) ?? [];
    if (!same(expected, offered)) {
      failures.push(`model call ${i + 1}: expected tools [${expected.join(", ")}], got [${offered.join(", ")}]`);
    }
  });
  // An unused turn means the run stopped earlier than whoever wrote the script
  // thought it would, which the transcript alone will not say if `expect` was
  // filled in from a previous run's output.
  if (requests.length < scenario.script.length) {
    failures.push(`script: ${scenario.script.length} turns scripted, but the run made ${requests.length} model call${requests.length === 1 ? "" : "s"}`);
  }
  return failures;
}

function show(prompt: string | undefined): string {
  return prompt === undefined ? "none" : JSON.stringify(prompt);
}

function same(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((line, i) => line === b[i]);
}

/**
 * Aligned by position rather than matched up: in a transcript a turn that moved
 * *is* the regression, so the first line that differs is the one to read, and a
 * matcher that quietly re-paired the rest around it would hide the move.
 */
function diff(expected: readonly string[], actual: readonly string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < Math.max(expected.length, actual.length); i++) {
    const want = expected[i];
    const got = actual[i];
    if (want === got) out.push(`  ${want}`);
    else {
      if (want !== undefined) out.push(`- ${want}`);
      if (got !== undefined) out.push(`+ ${got}`);
    }
  }
  return out;
}
