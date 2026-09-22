import { executeTool, toToolSpec, type ToolContext, type ToolDefinition } from "./tools.js";
import type { ConversationMemory } from "./memory.js";
import type { Run, Tracer } from "./trace.js";
import {
  EMPTY_USAGE,
  addUsage,
  textOf,
  type ChatMessage,
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
  | { type: "done"; result: AgentResult };

export type AgentStatus = "completed" | "max_iterations" | "refused" | "truncated" | "aborted";

export interface AgentResult {
  status: AgentStatus;
  /** Text of the final assistant message. */
  text: string;
  /** Full transcript for this run, including tool calls and results. */
  messages: ChatMessage[];
  iterations: number;
  usage: Usage;
  trace?: Run;
}

export interface AgentOptions {
  provider: ModelProvider;
  /** A string becomes the first user message. */
  input: string | ChatMessage[];
  system?: string;
  tools?: ToolDefinition[];
  /** Hard cap on model calls. Default 10. The loop has to end even if the model never says so. */
  maxIterations?: number;
  memory?: ConversationMemory;
  tracer?: Tracer;
  runName?: string;
  onEvent?: (event: AgentEvent) => void;
  signal?: AbortSignal;
  toolContext?: Omit<ToolContext, "signal">;
}

/**
 * The bounded tool loop: model → tools → model, until the model stops calling
 * tools or something says stop. Every stop condition is a named status, not
 * an exception, so the caller can tell "done" from "gave up" from "refused".
 */
export async function runAgent(options: AgentOptions): Promise<AgentResult> {
  const tools = options.tools ?? [];
  const byName = new Map(tools.map((t) => [t.name, t]));
  const specs = tools.length > 0 ? tools.map(toToolSpec) : undefined;
  const maxIterations = options.maxIterations ?? 10;
  const emit = options.onEvent ?? (() => {});

  const transcript: ChatMessage[] =
    typeof options.input === "string"
      ? [{ role: "user", content: [{ type: "text", text: options.input }] }]
      : options.input.slice();
  for (const m of transcript) options.memory?.append(m);

  const run = options.tracer?.startRun(options.runName ?? "agent", { provider: options.provider.name, model: options.provider.model });
  let usage = EMPTY_USAGE;
  let iterations = 0;

  const finish = async (status: AgentStatus, error?: unknown): Promise<AgentResult> => {
    const last = transcript[transcript.length - 1];
    const text = last?.role === "assistant" ? textOf(last.content) : "";
    const trace = run ? await run.end(error) : undefined;
    const result: AgentResult = { status, text, messages: transcript, iterations, usage, ...(trace ? { trace } : {}) };
    emit({ type: "done", result });
    return result;
  };

  try {
    while (iterations < maxIterations) {
      if (options.signal?.aborted) return finish("aborted");
      iterations++;

      // A snapshot, so a provider that keeps the request (a fake, a logger)
      // does not see later turns appear inside it.
      const messages = options.memory ? await options.memory.window() : transcript.slice();
      const span = run?.startSpan("model.call", options.provider.model, { iteration: iterations });
      let response: ModelResponse;
      try {
        response = await options.provider.complete({
          ...(options.system !== undefined ? { system: options.system } : {}),
          messages,
          ...(specs ? { tools: specs } : {}),
          ...(options.signal ? { signal: options.signal } : {}),
          onTextDelta: (text) => emit({ type: "text_delta", text }),
        });
      } catch (err) {
        span?.end(err);
        throw err;
      }
      span?.recordUsage(response.model, response.usage).setAttributes({ stopReason: response.stopReason }).end();
      usage = addUsage(usage, response.usage);
      emit({ type: "model_call", iteration: iterations, response });

      const assistant: ChatMessage = { role: "assistant", content: response.content };
      transcript.push(assistant);
      options.memory?.append(assistant);

      // A refusal can cut a tool_use off mid-input; a max_tokens stop can leave
      // a tool input that parses but is incomplete. Never run those tools.
      if (response.stopReason === "refusal") return finish("refused");
      if (response.stopReason === "max_tokens") return finish("truncated");

      const calls = response.content.filter((p): p is ToolUsePart => p.type === "tool_use");
      if (calls.length === 0) return finish("completed");

      // All tool calls from one turn run concurrently, and ALL their results go
      // back in ONE user message. Splitting them across messages teaches the
      // model to stop making parallel calls.
      const results = await Promise.all(
        calls.map(async (call): Promise<ToolResultPart> => {
          emit({ type: "tool_call", id: call.id, name: call.name, input: call.input });
          const toolSpan = run?.startSpan("tool.call", call.name, { toolUseId: call.id });
          const tool = byName.get(call.name);
          const outcome = tool
            ? await executeTool(tool, call.input, { ...options.toolContext, ...(options.signal ? { signal: options.signal } : {}) })
            : { content: `unknown tool: ${call.name}`, isError: true, durationMs: 0 };
          toolSpan?.setAttributes({ isError: outcome.isError, durationMs: outcome.durationMs }).end(outcome.isError ? outcome.content : undefined);
          emit({ type: "tool_result", id: call.id, name: call.name, ...outcome });
          return { type: "tool_result", toolUseId: call.id, content: outcome.content, isError: outcome.isError };
        }),
      );

      const user: ChatMessage = { role: "user", content: results };
      transcript.push(user);
      options.memory?.append(user);
    }
    return finish("max_iterations");
  } catch (err) {
    await run?.end(err);
    throw err;
  }
}
