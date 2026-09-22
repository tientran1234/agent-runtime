import { runAgent, type AgentEvent, type AgentOptions } from "./loop.js";

/**
 * Run an agent and stream its events as Server-Sent Events. Each event is
 * `event: <type>` + `data: <json>`; the final `done` event carries the result.
 * Works anywhere a web `Response` does: Next.js route handlers, Hono, Bun, Deno.
 */
export function agentSSE(options: Omit<AgentOptions, "onEvent">): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (event: AgentEvent) => {
        controller.enqueue(encoder.encode(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`));
      };
      runAgent({ ...options, onEvent: send })
        .catch((err: unknown) => {
          send({ type: "done", result: { status: "aborted", text: "", messages: [], iterations: 0, usage: { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 } } });
          controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify({ message: err instanceof Error ? err.message : String(err) })}\n\n`));
        })
        .finally(() => controller.close());
    },
  });
  return new Response(stream, {
    headers: { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" },
  });
}
