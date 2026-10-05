import type { ChatMessage, ContextEditing } from "./types.js";

/**
 * What a cleared tool result says in place of what the tool returned. The
 * result goes; the block stays — a `tool_use` whose `tool_result` has gone
 * missing is a 400 from every provider, so what clearing removes is the
 * content and never the pairing.
 */
export const CLEARED_TOOL_RESULT = "[tool result cleared to stay inside the context window]";

/**
 * `ContextEditing` applied here rather than by the provider: the window that
 * goes out keeps the newest tool results in full and carries the older ones as
 * a placeholder. This is the equivalent of the Anthropic adapter's
 * `context_management` for the providers that have no such thing, and it is
 * what the loop uses for them — so `contextEditing` reads the same on a local
 * model as on the API.
 *
 * The input is never touched. The transcript is what the caller is handed back,
 * what a trace shows and what a suspended run resumes from; editing it would
 * lose what the tools actually returned, which is the one thing clearing a
 * request is not allowed to cost.
 */
export function clearToolUses(messages: readonly ChatMessage[], editing: ContextEditing): ChatMessage[] {
  const keep = editing.clearToolUsesAfter;
  // Caught here rather than being read as "clear everything": a window is a
  // count, and a fractional or negative one is an authoring mistake.
  if (!Number.isInteger(keep) || keep < 0) {
    throw new Error(`clearToolUses: clearToolUsesAfter must be a non-negative integer, got ${String(keep)}`);
  }

  // Counted over the whole window in call order, not per message: a turn's
  // parallel calls are several tool uses inside one message, and the budget is
  // on uses rather than on turns.
  const called: string[] = [];
  for (const m of messages) {
    if (m.role !== "assistant") continue;
    for (const p of m.content) if (p.type === "tool_use") called.push(p.id);
  }
  if (called.length <= keep) return messages.slice();

  const ids = new Set(called);
  const spared = new Set(called.slice(called.length - keep));
  return messages.map((m) => {
    if (m.role !== "user") return m;
    const content = m.content.map((p) =>
      // A result whose call is no longer in the window was never one of the
      // uses counted above, so it is not one of the uses cleared either: its
      // call has already gone and this text is all that is left of it.
      p.type === "tool_result" && ids.has(p.toolUseId) && !spared.has(p.toolUseId)
        ? { ...p, content: CLEARED_TOOL_RESULT }
        : p,
    );
    return content.some((p, i) => p !== m.content[i]) ? { role: "user", content } : m;
  });
}
