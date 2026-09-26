/**
 * The provider-neutral contract. The agent loop, memory, tracing and fallback
 * all speak these types and nothing else. A provider adapter translates them
 * to and from one vendor SDK — and is the only file that imports that SDK.
 */

export interface TextPart {
  type: "text";
  text: string;
}
export interface ToolUsePart {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
}
export interface ToolResultPart {
  type: "tool_result";
  toolUseId: string;
  content: string;
  isError?: boolean;
}

export type AssistantPart = TextPart | ToolUsePart;
export type UserPart = TextPart | ToolResultPart;

export type ChatMessage =
  | { role: "user"; content: UserPart[] }
  | { role: "assistant"; content: AssistantPart[] };

/** A tool as the model sees it: name, description, JSON Schema for the input. */
export interface ToolSpec {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  /**
   * Ask the provider to guarantee the input matches `inputSchema`. Only set
   * once the schema has been checked against the strict contract, because a
   * provider rejects the whole request over a schema it cannot enforce.
   */
  strict?: boolean;
}

export type StopReason = "end_turn" | "tool_use" | "max_tokens" | "refusal" | "other";

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
}

export interface ModelResponse {
  /** The model that actually answered — matters once a fallback chain is in play. */
  model: string;
  content: AssistantPart[];
  stopReason: StopReason;
  usage: Usage;
}

export interface ModelRequest {
  system?: string;
  messages: ChatMessage[];
  tools?: ToolSpec[];
  signal?: AbortSignal;
  /** Called with each text chunk as it streams. Optional: providers may stream regardless. */
  onTextDelta?: (text: string) => void;
}

export interface ModelProvider {
  readonly name: string;
  readonly model: string;
  complete(request: ModelRequest): Promise<ModelResponse>;
}

/**
 * The one error type that crosses the provider boundary. `retryable` is the
 * provider's verdict — rate limit, overload, network — and is what the
 * fallback chain keys on. Vendor error classes never leave their adapter.
 */
export class ProviderError extends Error {
  override readonly name = "ProviderError";
  constructor(
    message: string,
    readonly retryable: boolean,
    readonly status?: number,
    options?: { cause?: unknown },
  ) {
    super(message, options);
  }
}

export const EMPTY_USAGE: Usage = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 };

export function addUsage(a: Usage, b: Usage): Usage {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
    cacheWriteTokens: a.cacheWriteTokens + b.cacheWriteTokens,
  };
}

/** Text of an assistant message, tool calls ignored. */
export function textOf(parts: readonly AssistantPart[]): string {
  return parts
    .filter((p): p is TextPart => p.type === "text")
    .map((p) => p.text)
    .join("");
}
