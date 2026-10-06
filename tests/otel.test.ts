import { describe, expect, it } from "vitest";
import {
  OtelExporter,
  Tracer,
  toOtlpTraces,
  type OtelExporterOptions,
  type OtlpAttribute,
  type OtlpSpan,
  type OtlpTraces,
  type Run,
} from "../src/index.js";

const PRICES = { "fake-1": { input: 1_000_000, output: 2_000_000, cacheRead: 0, cacheWrite: 0 } };

/** A run with one model call, one tool call, and a nested model call under the tool. */
async function sampleRun(now = 1_760_000_000_000): Promise<Run> {
  const tracer = new Tracer({ prices: PRICES, now: () => now });
  const run = tracer.startRun("agent", { requestId: "req_7" });
  run
    .startSpan("model.call", "fake-1", { iteration: 1 })
    .recordUsage("fake-1", { inputTokens: 2, outputTokens: 3, cacheReadTokens: 4, cacheWriteTokens: 5 })
    .setAttributes({ stopReason: "tool_use" })
    .end();
  const tool = run.startSpan("tool.call", "get_order", { toolUseId: "tu_1" });
  tool.startSpan("model.call", "fake-1").end();
  tool.end();
  return run.end();
}

const spansOf = (payload: OtlpTraces): OtlpSpan[] => payload.resourceSpans[0]?.scopeSpans[0]?.spans ?? [];
const attrs = (list: OtlpAttribute[]) => Object.fromEntries(list.map((a) => [a.key, Object.values(a.value)[0]]));
const byName = (payload: OtlpTraces, name: string) => spansOf(payload).find((s) => s.name === name);

describe("toOtlpTraces", () => {
  it("makes the run the root span and hangs every span under its real parent", async () => {
    const payload = toOtlpTraces(await sampleRun());
    const spans = spansOf(payload);

    expect(spans.map((s) => s.name)).toEqual(["agent", "fake-1", "get_order", "fake-1"]);
    const [root, first, tool, nested] = spans as [OtlpSpan, OtlpSpan, OtlpSpan, OtlpSpan];
    expect(root.parentSpanId).toBeUndefined();
    expect(first.parentSpanId).toBe(root.spanId);
    expect(tool.parentSpanId).toBe(root.spanId);
    expect(nested.parentSpanId).toBe(tool.spanId);
  });

  it("gives every span the run's trace id and an id of its own, both valid widths", async () => {
    const spans = spansOf(toOtlpTraces(await sampleRun()));

    for (const span of spans) {
      expect(span.traceId).toMatch(/^[0-9a-f]{32}$/);
      expect(span.spanId).toMatch(/^[0-9a-f]{16}$/);
    }
    expect(new Set(spans.map((s) => s.traceId)).size).toBe(1);
    expect(new Set(spans.map((s) => s.spanId)).size).toBe(spans.length);
  });

  it("writes nanosecond timestamps without losing the millisecond's low digits", async () => {
    const spans = spansOf(toOtlpTraces(await sampleRun(1_760_000_000_123)));

    // 1_760_000_000_123 * 1e6 is past Number.MAX_SAFE_INTEGER: computed as a
    // number the last digits round away and every span lands at the same
    // bogus instant.
    expect(spans[0]?.startTimeUnixNano).toBe("1760000000123000000");
    expect(spans[0]?.endTimeUnixNano).toBe("1760000000123000000");
  });

  it("carries model, tokens and cost on a model span", async () => {
    const span = byName(toOtlpTraces(await sampleRun()), "fake-1");

    expect(attrs(span?.attributes ?? [])).toMatchObject({
      "gen_ai.operation.name": "chat",
      "gen_ai.request.model": "fake-1",
      "gen_ai.response.model": "fake-1",
      "gen_ai.usage.input_tokens": "2",
      "gen_ai.usage.output_tokens": "3",
      "agent_runtime.usage.cache_read_tokens": "4",
      "agent_runtime.usage.cache_write_tokens": "5",
      "agent_runtime.cost_usd": 8,
      "agent_runtime.stop_reason": "tool_use",
    });
    expect(span?.kind).toBe(3);
  });

  it("names the tool on a tool span", async () => {
    const span = byName(toOtlpTraces(await sampleRun()), "get_order");

    expect(attrs(span?.attributes ?? [])).toMatchObject({
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "get_order",
      "agent_runtime.tool_use_id": "tu_1",
    });
    expect(span?.kind).toBe(1);
  });

  it("reports cost as a double even when it lands on a whole number", async () => {
    const span = byName(toOtlpTraces(await sampleRun()), "fake-1");
    const cost = span?.attributes.find((a) => a.key === "agent_runtime.cost_usd");

    // Cost is money: a field that is intValue on one span and doubleValue on
    // the next cannot be summed by the backend.
    expect(cost?.value).toEqual({ doubleValue: 8 });
  });

  it("puts the run's totals on the root span", async () => {
    const payload = toOtlpTraces(await sampleRun());

    expect(attrs(spansOf(payload)[0]?.attributes ?? [])).toMatchObject({
      "agent_runtime.model_calls": "2",
      "agent_runtime.tool_calls": "1",
      "agent_runtime.tool_errors": "0",
      "agent_runtime.cost_usd": 8,
      "agent_runtime.request_id": "req_7",
    });
  });

  it("leaves cost off rather than reporting zero when a model has no price", async () => {
    const tracer = new Tracer({ prices: {}, now: () => 0 });
    const run = tracer.startRun("agent");
    run.startSpan("model.call", "unpriced").recordUsage("unpriced", { inputTokens: 1, outputTokens: 1, cacheReadTokens: 0, cacheWriteTokens: 0 }).end();
    const payload = toOtlpTraces(await run.end());

    for (const span of spansOf(payload)) {
      expect(span.attributes.map((a) => a.key)).not.toContain("agent_runtime.cost_usd");
    }
  });

  it("marks a failed span and a failed run ERROR, and a healthy one UNSET", async () => {
    const tracer = new Tracer({ now: () => 0 });
    const run = tracer.startRun("agent");
    run.startSpan("tool.call", "get_order").end(new Error("timed out"));
    const payload = toOtlpTraces(await run.end(new Error("gave up")));
    const [root, tool] = spansOf(payload) as [OtlpSpan, OtlpSpan];

    expect(root.status).toEqual({ code: 2, message: "gave up" });
    expect(tool.status).toEqual({ code: 2, message: "timed out" });
    expect(spansOf(toOtlpTraces(await sampleRun()))[0]?.status).toEqual({ code: 0 });
  });

  it("re-parents a span whose parent is not in the run, so the collector keeps it", async () => {
    const run = await sampleRun();
    const orphan = run.spans[3];
    if (orphan) orphan.parentId = "gone";
    const spans = spansOf(toOtlpTraces(run));

    expect(spans[3]?.parentSpanId).toBe(spans[0]?.spanId);
  });

  it("closes a span still open when the run ended, since a collector drops one with no end", async () => {
    const tracer = new Tracer({ now: () => 1_000 });
    const run = tracer.startRun("agent");
    run.startSpan("tool.call", "never_returned");
    const spans = spansOf(toOtlpTraces(await run.end()));

    expect(spans[1]?.endTimeUnixNano).toBe(spans[0]?.endTimeUnixNano);
  });

  it("hashes an id that is not wide enough to be hex, instead of emitting an invalid one", async () => {
    const run = await sampleRun();
    run.id = "run-1";
    const first = run.spans[0];
    const second = run.spans[1];
    if (first) first.id = "span-1";
    if (second) second.id = "span-2";
    const spans = spansOf(toOtlpTraces(run));

    expect(spans[0]?.traceId).toMatch(/^[0-9a-f]{32}$/);
    expect(spans[1]?.spanId).toMatch(/^[0-9a-f]{16}$/);
    expect(spans[1]?.spanId).not.toBe(spans[2]?.spanId);
  });

  it("names the service on the resource", async () => {
    const payload = toOtlpTraces(await sampleRun(), { serviceName: "support-bot", resourceAttributes: { "deployment.environment": "prod" } });

    expect(attrs(payload.resourceSpans[0]?.resource.attributes ?? [])).toEqual({
      "service.name": "support-bot",
      "deployment.environment": "prod",
    });
  });
});

describe("OtelExporter", () => {
  const accepted = () => new Response(JSON.stringify({}), { status: 200, headers: { "content-type": "application/json" } });

  /** One export through a stub `fetch`: what it went to, what it carried, what was reported. */
  async function send(options: Partial<OtelExporterOptions> = {}, reply: Response | (() => Response) = accepted()) {
    let url: string | undefined;
    let init: RequestInit | undefined;
    const errors: Error[] = [];
    const exporter = new OtelExporter({
      endpoint: "http://localhost:4318",
      onError: (error) => errors.push(error),
      ...options,
      fetch: async (input, sent) => {
        url = String(input);
        init = sent;
        return typeof reply === "function" ? reply() : reply;
      },
    });
    await exporter.export(await sampleRun());
    return { url, init, errors, payload: init ? (JSON.parse(String(init.body)) as OtlpTraces) : undefined };
  }

  it("posts OTLP JSON to the collector's /v1/traces", async () => {
    const { url, init, payload } = await send();

    expect(url).toBe("http://localhost:4318/v1/traces");
    expect(init?.method).toBe("POST");
    expect((init?.headers as Record<string, string>)["content-type"]).toBe("application/json");
    expect(spansOf(payload ?? { resourceSpans: [] })[0]?.name).toBe("agent");
  });

  it("does not double the path when the endpoint already names it", async () => {
    const { url } = await send({ endpoint: "https://collector.example.com/v1/traces/" });

    expect(url).toBe("https://collector.example.com/v1/traces");
  });

  it("sends the configured headers", async () => {
    const { init } = await send({ headers: { "x-scope-orgid": "acme" } });

    expect((init?.headers as Record<string, string>)["x-scope-orgid"]).toBe("acme");
  });

  it("reports a collector that is down instead of failing the run", async () => {
    const errors: Error[] = [];
    const tracer = new Tracer({
      exporters: [
        new OtelExporter({
          endpoint: "http://localhost:4318",
          fetch: async () => {
            throw new Error("ECONNREFUSED");
          },
          onError: (error) => errors.push(error),
        }),
      ],
      now: () => 0,
    });

    // The run finished; whether its trace was shipped is not the run's status.
    const finished = await tracer.startRun("agent").end();
    expect(finished.status).toBe("ok");
    expect(errors.map((e) => e.message)).toEqual([expect.stringContaining("ECONNREFUSED")]);
  });

  it("reports a rejecting collector with its status and body", async () => {
    const { errors } = await send({}, new Response("bad tenant", { status: 403 }));

    expect(errors[0]?.message).toContain("403");
    expect(errors[0]?.message).toContain("bad tenant");
  });

  it("reports spans the collector took the request for but dropped", async () => {
    // A 200 that dropped half the trace looks like success without this.
    const partial = new Response(JSON.stringify({ partialSuccess: { rejectedSpans: "2", errorMessage: "trace id invalid" } }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
    const { errors } = await send({}, partial);

    expect(errors[0]?.message).toContain("2");
    expect(errors[0]?.message).toContain("trace id invalid");
  });
});
