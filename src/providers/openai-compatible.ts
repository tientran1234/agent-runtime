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
  textOf,
  type ChatMessage,
  type ModelProvider,
  type ModelRequest,
  type ModelResponse,
  type StopReason,
  type TextPart,
  type ToolSpec,
  type ToolUsePart,
  type Usage,
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
  private readonly baseURL: string;
  private readonly apiKey: string;
  private readonly maxTokens: number | undefined;
  private readonly extraHeaders: Record<string, string>;
  private readonly stream: boolean;
  private readonly extraBody: Record<string, unknown>;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(options: OpenAICompatibleProviderOptions) {
    this.name = options.name ?? "openai-compatible";
    this.model = options.model;
    this.baseURL = (options.baseURL ?? "https://api.openai.com/v1").replace(/\/+$/, "");
    this.apiKey = options.apiKey ?? process.env["OPENAI_API_KEY"] ?? "";
    this.maxTokens = options.maxTokens;
    this.extraHeaders = options.headers ?? {};
    this.stream = options.stream ?? true;
    this.extraBody = options.extraBody ?? {};
    this.fetchImpl = options.fetch ?? globalThis.fetch;
  }

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const body = {
      model: this.model,
      messages: toChatMessages(request.system, request.messages),
      ...(request.tools && request.tools.length > 0 ? { tools: toFunctionTools(request.tools) } : {}),
      ...(this.maxTokens !== undefined ? { max_tokens: this.maxTokens } : {}),
      // A streamed answer cannot hit the HTTP timeout, and it is the only way
      // `onTextDelta` — and so `agentSSE` — sees anything from this endpoint.
      // Usage comes in a final chunk that is omitted unless asked for; a server
      // that does not know the option ignores it and reports no usage.
      ...(this.stream ? { stream: true, stream_options: { include_usage: true } } : {}),
      // Last, so it can also unset a field above: JSON.stringify drops undefined.
      ...this.extraBody,
    };

    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseURL}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          // Omitted rather than sent empty: a local server has no key to check,
          // and some reject the header outright when they cannot verify it.
          ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}),
          ...this.extraHeaders,
        },
        body: JSON.stringify(body),
        ...(request.signal ? { signal: request.signal } : {}),
      });
    } catch (err) {
      throw toProviderError(err);
    }

    if (!response.ok) {
      throw toHttpError(response.status, await response.text().catch(() => ""));
    }
    try {
      const completion =
        this.stream && response.body
          ? await readCompletionStream(response.body, request.onTextDelta)
          : ((await response.json()) as ChatCompletion);
      return fromCompletion(completion, this.model);
    } catch (err) {
      throw toProviderError(err);
    }
  }
}

// ---- mapping, exported so the translation itself can be tested -------------

/**
 * The system prompt is a message here rather than a field of its own, and a
 * turn does not map one-to-one: every tool result becomes its own `tool`
 * message, and they have to follow the assistant's `tool_calls` with nothing
 * in between — so they are emitted ahead of any text the same user message
 * carried. `is_error` has no wire form; the content already says what failed.
 */
export function toChatMessages(
  system: string | undefined,
  messages: readonly ChatMessage[],
): ChatCompletionMessageParam[] {
  const out: ChatCompletionMessageParam[] = [];
  if (system !== undefined) out.push({ role: "system", content: system });

  for (const message of messages) {
    if (message.role === "assistant") {
      const toolCalls = message.content
        .filter((p): p is ToolUsePart => p.type === "tool_use")
        .map((p): ChatCompletionToolCall => ({
          id: p.id,
          type: "function",
          function: { name: p.name, arguments: JSON.stringify(p.input ?? {}) },
        }));
      const text = textOf(message.content);
      out.push({
        role: "assistant",
        // null, not "": a few servers reject an empty string next to tool_calls.
        content: text === "" ? null : text,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }

    for (const part of message.content) {
      if (part.type === "tool_result") out.push({ role: "tool", tool_call_id: part.toolUseId, content: part.content });
    }
    const text = message.content
      .filter((p): p is TextPart => p.type === "text")
      .map((p) => p.text)
      .join("");
    if (text !== "") out.push({ role: "user", content: text });
  }
  return out;
}

/**
 * `strict` asks for the same guarantee as on the Anthropic side, and wants the
 * same schema for it — closed objects, every property required — which
 * `toToolSpec` has already checked by the time a spec gets here.
 */
export function toFunctionTools(tools: readonly ToolSpec[]): ChatCompletionTool[] {
  return tools.map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description,
      parameters: t.inputSchema,
      ...(t.strict ? { strict: true } : {}),
    },
  }));
}

/**
 * `requestedModel` stands in when the response omits `model`, which some
 * compatible servers do: the span's cost is looked up on that field, and an
 * empty name would price the call at nothing instead of at unknown.
 */
export function fromCompletion(completion: ChatCompletion, requestedModel: string): ModelResponse {
  const choice = completion.choices?.[0];
  const message = choice?.message;
  const toolCalls = message?.tool_calls ?? [];

  const content: ModelResponse["content"] = [];
  if (message?.content) content.push({ type: "text", text: message.content });
  toolCalls.forEach((call, i) => {
    content.push({
      // The loop pairs a result back to its call by id, so a server that sends
      // none gets one per position — two blank ids in a turn would collide.
      id: call.id || `call_${i}`,
      type: "tool_use",
      name: call.function.name,
      input: parseArguments(call.function.arguments),
    });
  });

  return {
    model: completion.model || requestedModel,
    content,
    stopReason: toStopReason(choice?.finish_reason, toolCalls.length > 0),
    usage: toUsage(completion.usage),
  };
}

/**
 * Read a streamed completion back into the shape the non-streaming endpoint
 * would have returned, so there is one mapping and not two. Frames are
 * buffered because a chunk can split anywhere, including mid-JSON — the same
 * reason `readAgentSSE` exists; ten lines of splitting here is cheaper than
 * coupling a browser-only module to a server adapter.
 */
export async function readCompletionStream(
  body: ReadableStream<Uint8Array>,
  onTextDelta?: (text: string) => void,
): Promise<ChatCompletion> {
  let model: string | undefined;
  let finishReason: string | null | undefined;
  let usage: ChatCompletionUsage | undefined;
  let text = "";
  const calls: ChatCompletionToolCall[] = [];
  let done = false;

  const take = (frame: string): void => {
    const payload = dataOf(frame);
    if (payload === undefined) return;
    if (payload === "[DONE]") {
      done = true;
      return;
    }
    let chunk: ChatCompletionChunk;
    try {
      chunk = JSON.parse(payload) as ChatCompletionChunk;
    } catch (err) {
      throw new ProviderError(`openai: stream sent a frame that is not JSON: ${payload.slice(0, 120)}`, false, undefined, { cause: err });
    }
    if (chunk.model) model = chunk.model;
    if (chunk.usage) usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (choice?.finish_reason) finishReason = choice.finish_reason;
    const delta = choice?.delta;
    if (delta?.content) {
      text += delta.content;
      onTextDelta?.(delta.content);
    }
    for (const fragment of delta?.tool_calls ?? []) {
      // Without an index the fragment continues the call already open: a server
      // that omits it does not interleave two calls in one turn either.
      const at = fragment.index ?? Math.max(0, calls.length - 1);
      let call = calls[at];
      if (!call) {
        call = { id: "", type: "function", function: { name: "", arguments: "" } };
        calls[at] = call;
      }
      if (fragment.id) call.id = fragment.id;
      // The name arrives in fragments too, on servers that chunk aggressively.
      if (fragment.function?.name) call.function.name += fragment.function.name;
      if (fragment.function?.arguments) call.function.arguments += fragment.function.arguments;
    }
  };

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      for (let end = FRAME_END.exec(buffer); end; end = FRAME_END.exec(buffer)) {
        const frame = buffer.slice(0, end.index);
        buffer = buffer.slice(end.index + end[0].length);
        take(frame);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== "") take(buffer);
  } finally {
    reader.releaseLock();
  }

  // A stream that broke must not look like one that finished: the turn is only
  // over once a finish_reason says so, and a fragment of an answer is worth
  // retrying elsewhere rather than returning as the whole of it.
  if (finishReason === undefined && !done) {
    throw new ProviderError("openai: stream ended before the turn did", true);
  }

  return {
    ...(model !== undefined ? { model } : {}),
    choices: [
      {
        message: {
          content: text === "" ? null : text,
          // Holes, if a server skipped an index, are not calls anyone can answer.
          ...(calls.length > 0 ? { tool_calls: calls.filter((c) => c !== undefined) } : {}),
        },
        finish_reason: finishReason ?? null,
      },
    ],
    ...(usage ? { usage } : {}),
  };
}

/** A frame ends at a blank line, whichever line ending the hop in between used. */
const FRAME_END = /\r\n\r\n|\n\n|\r\r/;

function dataOf(frame: string): string | undefined {
  const data: string[] = [];
  for (const line of frame.split(/\r\n|\r|\n/)) {
    if (line === "" || line.startsWith(":")) continue; // keep-alive comments
    const colon = line.indexOf(":");
    if (colon === -1 || line.slice(0, colon) !== "data") continue;
    data.push(line.slice(colon + 1).replace(/^ /, ""));
  }
  return data.length > 0 ? data.join("\n") : undefined;
}

/**
 * `length` outranks the tool calls in the same turn: their arguments are cut
 * off, and the loop's promise is that no tool runs on a truncated input, even
 * one that happens to parse. The other way round matters too — several servers
 * answer "stop" on a turn that did call tools, and believing that would end the
 * run with the calls never made.
 */
function toStopReason(reason: string | null | undefined, hasToolCalls: boolean): StopReason {
  switch (reason) {
    case "length":
      return "max_tokens";
    case "content_filter":
      return "refusal";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "stop":
      return hasToolCalls ? "tool_use" : "end_turn";
    default:
      return hasToolCalls ? "tool_use" : "other";
  }
}

/**
 * An empty argument string is a call that took no arguments. One that is not
 * JSON is kept as the raw string, because the schema is the next thing to see
 * it and turns it into an error result the model can correct — where `{}` would
 * instead look like a valid call and reach `execute`.
 */
function parseArguments(raw: string | undefined): unknown {
  if (raw === undefined || raw.trim() === "") return {};
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return raw;
  }
}

/**
 * `prompt_tokens` includes the cached prefix, where Anthropic's `input_tokens`
 * excludes it — and `Usage` is the Anthropic split, because the two halves are
 * priced differently. Cache writes have no counterpart: caching here is
 * automatic and never billed as one.
 */
function toUsage(usage: ChatCompletionUsage | null | undefined): Usage {
  const cacheRead = usage?.prompt_tokens_details?.cached_tokens ?? 0;
  return {
    inputTokens: Math.max(0, (usage?.prompt_tokens ?? 0) - cacheRead),
    outputTokens: usage?.completion_tokens ?? 0,
    cacheReadTokens: cacheRead,
    cacheWriteTokens: 0,
  };
}

/**
 * The error envelope is `{ error: { message } }` where a server bothers with
 * it; a gateway in front of one answers HTML. Either way the text reaches the
 * caller, because the useful half of a 400 is always in the body.
 */
export function toHttpError(status: number, body: string): ProviderError {
  let message = body.trim();
  try {
    const parsed = JSON.parse(body) as { error?: { message?: unknown } };
    if (typeof parsed.error?.message === "string") message = parsed.error.message;
  } catch {
    // not the envelope; the raw body says more than nothing
  }
  // 408 and 429 are the server asking to be asked again; 5xx is it failing to
  // answer at all. Everything else will fail the same way on the next provider.
  const retryable = status === 408 || status === 429 || status >= 500;
  return new ProviderError(`openai: HTTP ${status}${message ? `: ${message.slice(0, 500)}` : ""}`, retryable, status);
}

/**
 * `fetch` throws a bare TypeError for everything at the transport layer, so a
 * dead connection is not distinguishable from DNS or TLS — all retryable, and
 * all worth asking the next provider. An abort is the caller's own doing and
 * must not travel down the fallback chain.
 */
export function toProviderError(err: unknown): ProviderError {
  if (err instanceof ProviderError) return err;
  if (err instanceof Error && err.name === "AbortError") {
    return new ProviderError(`openai: request aborted`, false, undefined, { cause: err });
  }
  if (err instanceof TypeError) {
    return new ProviderError(`openai: connection error: ${err.message}`, true, undefined, { cause: err });
  }
  return new ProviderError(err instanceof Error ? err.message : String(err), false, undefined, { cause: err });
}
