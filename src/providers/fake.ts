import {
  EMPTY_USAGE,
  type AssistantPart,
  type ModelProvider,
  type ModelRequest,
  type ModelResponse,
} from "../types.js";

type Scripted = ModelResponse | Error | ((request: ModelRequest) => ModelResponse | Error);

/**
 * A provider that answers from a script. The whole loop — tools, memory,
 * tracing, fallback, SSE — runs against it with no network and no key, which
 * is what makes the behaviour of the loop testable at all.
 */
export class FakeProvider implements ModelProvider {
  readonly name = "fake";
  readonly calls: ModelRequest[] = [];

  constructor(
    private readonly script: Scripted[],
    readonly model = "fake-1",
  ) {}

  async complete(request: ModelRequest): Promise<ModelResponse> {
    this.calls.push(request);
    const next = this.script.shift();
    if (next === undefined) throw new Error("FakeProvider: script exhausted");
    const out = typeof next === "function" ? next(request) : next;
    if (out instanceof Error) throw out;
    if (request.onTextDelta) {
      for (const part of out.content) if (part.type === "text") request.onTextDelta(part.text);
    }
    return { ...out, model: out.model || this.model };
  }
}

// ---- builders for scripts -------------------------------------------------

let counter = 0;

export function reply(text: string, usage = { inputTokens: 10, outputTokens: 5 }): ModelResponse {
  return {
    model: "",
    content: [{ type: "text", text }],
    stopReason: "end_turn",
    usage: { ...EMPTY_USAGE, ...usage },
  };
}

export function callTools(calls: Array<{ name: string; input: unknown; id?: string }>, text?: string): ModelResponse {
  const content: AssistantPart[] = [];
  if (text) content.push({ type: "text", text });
  for (const c of calls) content.push({ type: "tool_use", id: c.id ?? `toolu_${++counter}`, name: c.name, input: c.input });
  return { model: "", content, stopReason: "tool_use", usage: { ...EMPTY_USAGE, inputTokens: 20, outputTokens: 8 } };
}

export function stoppedWith(stopReason: ModelResponse["stopReason"], text = ""): ModelResponse {
  return { model: "", content: text ? [{ type: "text", text }] : [], stopReason, usage: EMPTY_USAGE };
}
