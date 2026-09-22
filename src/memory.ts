import type { ChatMessage } from "./types.js";

export interface MemoryOptions {
  /** Budget for the messages sent per call. The system prompt is not counted here. */
  maxTokens: number;
  /** Default: ~4 characters per token over the JSON of the content. Swap in a real tokenizer if you have one. */
  estimateTokens?: (message: ChatMessage) => number;
  /**
   * When turns are dropped, produce a short summary of them. It is inserted as
   * the first user message so the model still knows what happened.
   */
  summarize?: (dropped: ChatMessage[]) => Promise<string> | string;
}

const defaultEstimate = (m: ChatMessage) => Math.ceil(JSON.stringify(m.content).length / 4);

/**
 * Conversation history with a token budget.
 *
 * Trimming works in *turns*, not messages. A turn starts at a user message
 * that contains a text part (a human speaking) and runs until the next one —
 * which means it includes every assistant tool call and every tool result in
 * between. Dropping whole turns is what keeps a tool_use and its tool_result
 * together; splitting them is a 400 from every provider.
 */
export class ConversationMemory {
  private readonly messages: ChatMessage[] = [];
  private summary: string | null = null;
  private readonly estimate: (m: ChatMessage) => number;

  constructor(private readonly options: MemoryOptions) {
    this.estimate = options.estimateTokens ?? defaultEstimate;
  }

  append(message: ChatMessage): void {
    this.messages.push(message);
  }

  /** Everything ever appended, untrimmed. */
  get all(): readonly ChatMessage[] {
    return this.messages;
  }

  /** The messages to send now: within budget, whole turns, newest kept. */
  async window(): Promise<ChatMessage[]> {
    const turns = splitTurns(this.messages);
    const budget = this.options.maxTokens - (this.summary ? this.estimateText(this.summary) : 0);

    let kept = turns.slice();
    let dropped: ChatMessage[] = [];
    while (kept.length > 1 && this.tokensOf(kept.flat()) > budget) {
      dropped = dropped.concat(kept.shift() ?? []);
    }

    if (dropped.length > 0 && this.options.summarize) {
      const fresh = await this.options.summarize(dropped);
      this.summary = this.summary ? `${this.summary}\n${fresh}` : fresh;
    }

    const out = kept.flat();
    if (this.summary) {
      out.unshift({ role: "user", content: [{ type: "text", text: `[Summary of earlier conversation]\n${this.summary}` }] });
    }
    return out;
  }

  private tokensOf(messages: ChatMessage[]): number {
    return messages.reduce((n, m) => n + this.estimate(m), 0);
  }
  private estimateText(text: string): number {
    return this.estimate({ role: "user", content: [{ type: "text", text }] });
  }
}

/** Group messages into turns. Exported for tests. */
export function splitTurns(messages: readonly ChatMessage[]): ChatMessage[][] {
  const turns: ChatMessage[][] = [];
  for (const m of messages) {
    const startsTurn = m.role === "user" && m.content.some((p) => p.type === "text");
    if (startsTurn || turns.length === 0) turns.push([m]);
    else turns[turns.length - 1]!.push(m);
  }
  return turns;
}
