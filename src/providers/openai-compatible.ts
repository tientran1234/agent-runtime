/**
 * OpenAI-compatible adapter — `fetch` and nothing else.
 *
 * `POST /chat/completions` is the one shape the rest of the ecosystem agreed
 * on: OpenAI, Ollama, vLLM, llama.cpp, Together, OpenRouter, Groq. The wire
 * types below are the whole dependency, which is the point — a second SDK for
 * a second vendor would make the port an abstraction over two SDKs instead of
 * over one contract.
 *
 * Like the Anthropic adapter, it translates and nothing more: it never decides
 * what the agent does next.
 */
import {
  ProviderError,
  type ChatMessage,
  type ModelProvider,
  type ModelRequest,
  type ModelResponse,
  type ToolSpec,
} from "../types.js";

// ---- the wire, as much of it as this adapter speaks ------------------------

export interface ChatCompletionToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export type ChatCompletionMessageParam =
  | { role: "system" | "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: ChatCompletionToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string };

export interface ChatCompletionTool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    /** OpenAI's name for the same guarantee `ToolSpec.strict` asks for. */
    strict?: boolean;
  };
}

export interface ChatCompletionUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  /** Where a cache hit is reported — and it is counted inside `prompt_tokens`. */
  prompt_tokens_details?: { cached_tokens?: number };
}

export interface ChatCompletion {
  /** Absent on some servers, which is why mapping takes the requested model too. */
  model?: string;
  choices?: Array<{
    message?: { content?: string | null; tool_calls?: ChatCompletionToolCall[] };
    finish_reason?: string | null;
  }>;
  usage?: ChatCompletionUsage | null;
}

/** One `data:` frame of a streamed completion: the same shape in fragments. */
export interface ChatCompletionChunk {
  model?: string;
  choices?: Array<{
    delta?: {
      content?: string | null;
      tool_calls?: Array<{ index?: number; id?: string; function?: { name?: string; arguments?: string } }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: ChatCompletionUsage | null;
}

export interface OpenAICompatibleProviderOptions {
  /** The model id as this endpoint names it. No default: the endpoint owns the catalogue. */
  model: string;
  /** Default: `https://api.openai.com/v1`. Ollama: `http://localhost:11434/v1`. */
  baseURL?: string;
  /** Default: `OPENAI_API_KEY`. Empty means send no `Authorization` — local servers want none. */
  apiKey?: string;
  /** Shows up on traces as the provider that answered. Default `openai-compatible`. */
  name?: string;
  /** No default: the ceiling differs per endpoint, and one above a small model's window is a 400. */
  maxTokens?: number;
  headers?: Record<string, string>;
  /** Default true. */
  stream?: boolean;
  /** Merged into the request body last, so it can also unset a field above. */
  extraBody?: Record<string, unknown>;
  /** For tests, and for a caller that wants its own retry or proxy around the call. */
  fetch?: typeof globalThis.fetch;
}

export class OpenAICompatibleProvider implements ModelProvider {
  readonly name: string;
  readonly model: string;

  constructor(_options: OpenAICompatibleProviderOptions) {
    throw new Error("not implemented");
  }

  async complete(_request: ModelRequest): Promise<ModelResponse> {
    throw new Error("not implemented");
  }
}

// ---- mapping, exported so the translation itself can be tested -------------

export function toChatMessages(
  _system: string | undefined,
  _messages: readonly ChatMessage[],
): ChatCompletionMessageParam[] {
  throw new Error("not implemented");
}

export function toFunctionTools(_tools: readonly ToolSpec[]): ChatCompletionTool[] {
  throw new Error("not implemented");
}

export function fromCompletion(_completion: ChatCompletion, _requestedModel: string): ModelResponse {
  throw new Error("not implemented");
}

export async function readCompletionStream(
  _body: ReadableStream<Uint8Array>,
  _onTextDelta?: (text: string) => void,
): Promise<ChatCompletion> {
  throw new Error("not implemented");
}

export function toHttpError(_status: number, _body: string): ProviderError {
  throw new Error("not implemented");
}

export function toProviderError(_err: unknown): ProviderError {
  throw new Error("not implemented");
}
