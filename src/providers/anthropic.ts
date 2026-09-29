/**
 * Anthropic adapter — the only file that imports `@anthropic-ai/sdk`.
 *
 * It translates neutral requests into SDK calls and SDK messages back into
 * neutral responses. It never decides what the agent does next.
 */
import Anthropic from "@anthropic-ai/sdk";
import {
  ProviderError,
  type ChatMessage,
  type ModelProvider,
  type ModelRequest,
  type ModelResponse,
  type StopReason,
  type ToolSpec,
} from "../types.js";

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

/**
 * The beta that gates the `fallbacks: "default"` form. The header and the form
 * are one unit — this header with a `fallbacks` array, or the array's own
 * earlier header with `"default"`, is a 400 either way.
 */
export const SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01";

export interface AnthropicProviderOptions {
  client?: Anthropic;
  /** Default: claude-opus-5. */
  model?: string;
  /** Default 16 000. Raise for long outputs — the adapter streams, so timeouts are not the concern. */
  maxTokens?: number;
  effort?: Effort;
  /**
   * Adaptive thinking is on by default. Pass `false` for models that do not
   * accept `{ type: "adaptive" }` (e.g. Haiku 4.5) or to opt out explicitly.
   */
  thinking?: boolean;
  /**
   * Cache the system prompt and the tool list. Off by default: a cache write
   * costs 1.25× input and only pays for itself when the same prefix is sent
   * again — true for a tool loop, not for a single call.
   */
  cache?: boolean;
  /**
   * Re-run a request the safety classifiers decline on Anthropic's substitute
   * for that refusal category, server-side, inside the same call — so the loop
   * gets an answer instead of `status: "refused"`. Off by default: the answer
   * then comes from a model the caller did not ask for, at that model's prices.
   */
  serverFallbacks?: boolean;
  /**
   * Tools the API runs on its own side — web search, code execution, a hosted
   * MCP server — written exactly as its docs write them. The model calls them
   * inside its own turn: this runtime never executes one, and their blocks ride
   * along in the transcript so a paused turn can resume. A tool this process
   * runs is a `defineTool` handed to `runAgent`, not one of these.
   */
  serverTools?: Anthropic.ToolUnion[];
}

export class AnthropicProvider implements ModelProvider {
  readonly name = "anthropic";
  readonly model: string;
  private readonly client: Anthropic;
  private readonly maxTokens: number;
  private readonly effort: Effort | undefined;
  private readonly thinking: boolean;
  private readonly cache: boolean;
  private readonly serverFallbacks: boolean;
  private readonly serverTools: readonly Anthropic.ToolUnion[];

  constructor(options: AnthropicProviderOptions = {}) {
    this.client = options.client ?? new Anthropic();
    this.model = options.model ?? "claude-opus-5";
    this.maxTokens = options.maxTokens ?? 16_000;
    this.effort = options.effort;
    this.thinking = options.thinking ?? true;
    this.cache = options.cache ?? false;
    this.serverFallbacks = options.serverFallbacks ?? false;
    this.serverTools = options.serverTools ?? [];
  }

  async complete(request: ModelRequest): Promise<ModelResponse> {
    try {
      // Server tools go first so the run's own tools stay at the end of the
      // list, where the cache breakpoint is: a breakpoint closes a prefix, so
      // anything after it would be re-sent uncached on every call.
      const tools = [...this.serverTools, ...(request.tools ? toTools(request.tools, this.cache) : [])];
      const params = {
        model: this.model,
        max_tokens: this.maxTokens,
        ...(request.system !== undefined ? { system: toSystem(request.system, this.cache) } : {}),
        messages: toMessageParams(request.messages),
        ...(tools.length > 0 ? { tools } : {}),
        ...(this.thinking ? { thinking: { type: "adaptive" as const } } : {}),
        ...(this.effort ? { output_config: { effort: this.effort } } : {}),
      };
      const options = request.signal ? { signal: request.signal } : undefined;
      // Always stream: a long answer then cannot hit the HTTP timeout, and
      // finalMessage() gives back the complete Message either way. Each branch
      // drains its own stream because the two endpoints' helpers are unrelated
      // types, and a union of them has no callable `on`.
      if (this.serverFallbacks) {
        const stream = this.client.beta.messages.stream(
          { ...params, betas: [SERVER_FALLBACK_BETA], fallbacks: "default" },
          options,
        );
        if (request.onTextDelta) stream.on("text", request.onTextDelta);
        return fromMessage(await stream.finalMessage());
      }
      const stream = this.client.messages.stream(params, options);
      if (request.onTextDelta) stream.on("text", request.onTextDelta);
      return fromMessage(await stream.finalMessage());
    } catch (err) {
      throw toProviderError(err);
    }
  }
}

// ---- mapping, exported so the translation itself can be tested ----------

export function toMessageParams(messages: readonly ChatMessage[]): Anthropic.MessageParam[] {
  return messages.map((m): Anthropic.MessageParam => {
    if (m.role === "assistant") {
      return {
        role: "assistant",
        content: m.content.map((p): Anthropic.ContentBlockParam => {
          if (p.type === "text") return { type: "text", text: p.text };
          if (p.type === "tool_use") return { type: "tool_use", id: p.id, name: p.name, input: p.input };
          // A block this adapter produced, going back the way it came. The cast
          // is the whole point of an opaque part: nothing between here and the
          // API had any business reading it, and a turn the API paused only
          // resumes if it gets its own bytes back.
          return p.raw as Anthropic.ContentBlockParam;
        }),
      };
    }
    return {
      role: "user",
      content: m.content.map((p): Anthropic.ContentBlockParam =>
        p.type === "text"
          ? { type: "text", text: p.text }
          : { type: "tool_result", tool_use_id: p.toolUseId, content: p.content, ...(p.isError ? { is_error: true } : {}) },
      ),
    };
  });
}

const EPHEMERAL = { type: "ephemeral" as const };

/**
 * Tools render before the system prompt, so the breakpoint goes on the last
 * tool and closes a prefix of its own: editing the system prompt then still
 * leaves the tool half of the prefix cached.
 */
export function toTools(tools: readonly ToolSpec[], cache = false): Anthropic.Tool[] {
  return tools.map((t, i) => ({
    name: t.name,
    description: t.description,
    input_schema: t.inputSchema as Anthropic.Tool.InputSchema,
    ...(t.strict ? { strict: true } : {}),
    ...(cache && i === tools.length - 1 ? { cache_control: EPHEMERAL } : {}),
  }));
}

/** A plain string unless it is being cached — `cache_control` lives on blocks. */
export function toSystem(system: string, cache = false): string | Anthropic.TextBlockParam[] {
  return cache ? [{ type: "text", text: system, cache_control: EPHEMERAL }] : system;
}

/**
 * The blocks that are deliberately not carried. Thinking is the model's own and
 * the loop has never acted on it; `fallback` and `compaction` describe the call
 * rather than the turn, and who answered is already on `message.model`.
 */
const DROPPED_BLOCKS = new Set(["thinking", "redacted_thinking", "fallback", "compaction"]);

export function fromMessage(message: Anthropic.Message | Anthropic.Beta.BetaMessage): ModelResponse {
  const content: ModelResponse["content"] = [];
  for (const block of message.content) {
    if (block.type === "text") content.push({ type: "text", text: block.text });
    else if (block.type === "tool_use") content.push({ type: "tool_use", id: block.id, name: block.name, input: block.input });
    // Everything else is the API's own: a server tool's call, its result, a file
    // it produced. The loop cannot read them and does not need to, but a paused
    // turn only resumes if they go back, so they are carried rather than
    // dropped — and carried by default, since a server tool added after this
    // release would otherwise go missing quietly.
    else if (!DROPPED_BLOCKS.has(block.type)) content.push({ type: "server_tool", raw: block });
  }
  return {
    model: message.model,
    content,
    stopReason: toStopReason(message.stop_reason),
    usage: {
      inputTokens: message.usage.input_tokens,
      outputTokens: message.usage.output_tokens,
      cacheReadTokens: message.usage.cache_read_input_tokens ?? 0,
      cacheWriteTokens: message.usage.cache_creation_input_tokens ?? 0,
    },
  };
}

function toStopReason(reason: Anthropic.Message["stop_reason"] | Anthropic.Beta.BetaMessage["stop_reason"]): StopReason {
  switch (reason) {
    case "end_turn":
    case "tool_use":
    case "max_tokens":
    case "refusal":
    case "pause_turn":
      return reason;
    default:
      return "other";
  }
}

/**
 * Typed SDK errors → one neutral error with a retryable verdict. Order matters:
 * APIConnectionError is a subclass of APIError in this SDK, so it is checked first.
 */
export function toProviderError(err: unknown): ProviderError {
  if (err instanceof ProviderError) return err;
  if (err instanceof Anthropic.APIConnectionError) {
    return new ProviderError(`anthropic: connection error: ${err.message}`, true, undefined, { cause: err });
  }
  if (err instanceof Anthropic.RateLimitError || err instanceof Anthropic.InternalServerError) {
    return new ProviderError(`anthropic: ${err.message}`, true, err.status, { cause: err });
  }
  if (err instanceof Anthropic.APIError) {
    return new ProviderError(`anthropic: ${err.message}`, false, err.status, { cause: err });
  }
  return new ProviderError(err instanceof Error ? err.message : String(err), false, undefined, { cause: err });
}
