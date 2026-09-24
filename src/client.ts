import type { AgentEvent, AgentResult } from "./loop.js";

/**
 * The client half of `agentSSE`. Type-only imports from the loop, so this
 * module pulls no runtime code into a browser bundle: it needs `Response`
 * and nothing else, and runs the same in a worker or in Node.
 */

/**
 * The `error` frame `agentSSE` emits when the run throws. It is not an
 * `AgentEvent` — the loop never produced it — but a reader has to see it.
 */
export interface AgentErrorEvent {
  type: "error";
  message: string;
}

export type AgentSSEEvent = AgentEvent | AgentErrorEvent;

/** One optional callback per event type, plus `onEvent` for all of them. */
export type AgentSSEHandlers = {
  [E in AgentSSEEvent as E["type"]]?: (event: E) => void;
} & {
  onEvent?: (event: AgentSSEEvent) => void;
};

/** A stream that could not be read, or that reported a failed run. */
export class AgentSSEError extends Error {
  override readonly name = "AgentSSEError";
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
  }
}

const AGENT_EVENTS: ReadonlySet<string> = new Set([
  "text_delta",
  "model_call",
  "tool_call",
  "tool_result",
  "done",
]);

/** A frame ends at a blank line, whichever line ending the hop in between used. */
const FRAME_END = /\r\n\r\n|\n\n|\r\r/;

/**
 * Read an `agentSSE` response, calling a handler per event, and resolve with
 * the result the `done` frame carried. Reassembling frames across chunk
 * boundaries is the part every UI otherwise rewrites — a chunk is a transport
 * detail and can split anywhere, including mid-JSON.
 */
export async function readAgentSSE(
  response: Response,
  handlers: AgentSSEHandlers = {},
): Promise<AgentResult | undefined> {
  if (!response.ok) throw new AgentSSEError(`agent stream failed: HTTP ${response.status}`);
  if (!response.body) throw new AgentSSEError("agent stream has no body");

  let result: AgentResult | undefined;
  let failure: AgentErrorEvent | undefined;

  const dispatch = (frame: string): void => {
    const event = parseFrame(frame);
    if (!event) return;
    handlers.onEvent?.(event);
    // The per-type handlers are a union once indexed by a union key; the cast
    // is what the mapped type already guarantees.
    const handler = handlers[event.type] as ((event: AgentSSEEvent) => void) | undefined;
    handler?.(event);
    if (event.type === "done") result = event.result;
    else if (event.type === "error") failure = event;
  };

  const reader = response.body.getReader();
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
        dispatch(frame);
      }
    }
    buffer += decoder.decode();
    // A stream cut off after the last data line still had something to say.
    if (buffer.trim() !== "") dispatch(buffer);
  } finally {
    reader.releaseLock();
  }

  // A stream that failed must not look like a stream that finished: without an
  // `error` handler to take it, the failure comes back out as a rejection.
  if (failure && !handlers.error) throw new AgentSSEError(failure.message);
  return result;
}

function parseFrame(frame: string): AgentSSEEvent | undefined {
  let name: string | undefined;
  const data: string[] = [];
  for (const line of frame.split(/\r\n|\r|\n/)) {
    if (line === "" || line.startsWith(":")) continue; // keep-alive comments
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") name = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return undefined;

  const payload = data.join("\n");
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch (err) {
    throw new AgentSSEError(`agent stream sent a frame that is not JSON: ${payload.slice(0, 120)}`, { cause: err });
  }
  if (typeof parsed !== "object" || parsed === null) return undefined;

  const record = parsed as Record<string, unknown>;
  // The frame names the event; the payload repeats it for every event except
  // `error`, which carries only a message.
  const type = typeof record["type"] === "string" ? record["type"] : name;
  if (type === "error") {
    return { type: "error", message: typeof record["message"] === "string" ? record["message"] : "agent stream failed" };
  }
  // An event type this client does not know is dropped, so a newer server can
  // add one without breaking an older UI.
  if (type === undefined || !AGENT_EVENTS.has(type)) return undefined;
  return record as unknown as AgentEvent;
}
