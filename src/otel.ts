/**
 * OTLP/HTTP exporter — a run as a trace in Tempo, Jaeger, or any collector.
 *
 * OTLP over HTTP with a JSON body is the one ingest path every backend takes
 * without an SDK, so the wire types below are the whole dependency: pulling in
 * `@opentelemetry/*` would make a tracer that already models runs and spans
 * carry a second one that models them differently.
 *
 * The mapping is the interesting part and is exported on its own, because a
 * deployment that ships traces through a queue or a sidecar needs the payload
 * without the POST.
 */
import { createHash } from "node:crypto";
import type { Run, Span, TraceExporter } from "./trace.js";
import type { Usage } from "./types.js";

// ---- the wire, as much of OTLP/HTTP JSON as one trace needs ---------------

/**
 * An OTLP `AnyValue`. `intValue` is a string because the field is an int64 and
 * proto3's JSON mapping writes those as decimal strings.
 */
export type OtlpValue =
  | { stringValue: string }
  | { boolValue: boolean }
  | { intValue: string }
  | { doubleValue: number }
  | { arrayValue: { values: OtlpValue[] } };

export interface OtlpAttribute {
  key: string;
  value: OtlpValue;
}

/** UNSET, OK, ERROR — the only three OTLP has. */
export const OTLP_STATUS = { unset: 0, ok: 1, error: 2 } as const;

/** INTERNAL for work inside the process, CLIENT for a call that left it. */
export const OTLP_SPAN_KIND = { internal: 1, server: 2, client: 3 } as const;

export interface OtlpSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: number;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes: OtlpAttribute[];
  status: { code: number; message?: string };
}

export interface OtlpTraces {
  resourceSpans: Array<{
    resource: { attributes: OtlpAttribute[] };
    scopeSpans: Array<{ scope: { name: string; version?: string }; spans: OtlpSpan[] }>;
  }>;
}

/** What the collector is told about the process the spans came from. */
export interface OtlpResourceOptions {
  /** `service.name`, which is how a backend lists the trace. Default `agent-runtime`. */
  serviceName?: string;
  /** Anything else about the deployment: `service.version`, `deployment.environment`. */
  resourceAttributes?: Record<string, unknown>;
}

export interface OtelExporterOptions extends OtlpResourceOptions {
  /** Collector base URL. `/v1/traces` is appended unless it is already there. */
  endpoint: string;
  /** Sent with every export — an ingest token, a tenant header. */
  headers?: Record<string, string>;
  /** Default 10s. The run is already over; a hung collector must not hold it. */
  timeoutMs?: number;
  fetch?: typeof globalThis.fetch;
  /** Where a failed export goes. Default: one line on `console.error`. */
  onError?: (error: Error) => void;
}

/** Identifies the instrumentation to a backend that groups spans by it. */
const SCOPE_NAME = "agent-runtime";

/** A finished run as one OTLP payload: the run is the root span, its spans the children. */
export function toOtlpTraces(run: Run, options: OtlpResourceOptions = {}): OtlpTraces {
  const traceId = hexId(run.id, 32);
  const rootSpanId = hexId(run.id, 16);
  const runEndedAt = run.endedAt ?? run.startedAt;
  // A `parentId` naming a span the run does not carry — one read back from a
  // truncated export — is re-parented onto the root: a collector drops a span
  // whose parent it never sees, which would take the subtree with it.
  const known = new Set(run.spans.map((span) => span.id));

  const root: OtlpSpan = {
    traceId,
    spanId: rootSpanId,
    name: run.name,
    kind: OTLP_SPAN_KIND.internal,
    startTimeUnixNano: nanos(run.startedAt),
    endTimeUnixNano: nanos(runEndedAt),
    attributes: runAttributes(run),
    status: status(run.status === "error", run.error),
  };

  const spans = run.spans.map<OtlpSpan>((span) => ({
    traceId,
    spanId: hexId(span.id, 16),
    parentSpanId: span.parentId && known.has(span.parentId) ? hexId(span.parentId, 16) : rootSpanId,
    name: span.name,
    // A model call is the one thing here that leaves the process.
    kind: span.kind === "model.call" ? OTLP_SPAN_KIND.client : OTLP_SPAN_KIND.internal,
    startTimeUnixNano: nanos(span.startedAt),
    // A span still open when the run ended is closed at the run's end rather
    // than exported without one, which is not a span a collector will take.
    endTimeUnixNano: nanos(span.endedAt ?? runEndedAt),
    attributes: spanAttributes(span),
    status: status(span.error !== undefined || span.attributes.isError === true, span.error),
  }));

  return {
    resourceSpans: [
      {
        resource: {
          attributes: attributes({
            "service.name": options.serviceName ?? "agent-runtime",
            ...options.resourceAttributes,
          }),
        },
        scopeSpans: [{ scope: { name: SCOPE_NAME }, spans: [root, ...spans] }],
      },
    ],
  };
}

/**
 * Posts each finished run to an OTLP/HTTP collector.
 *
 * A failed export is reported, never thrown: `RunHandle.end()` awaits its
 * exporters, so a collector that is down would otherwise turn every finished
 * run into a failed one.
 */
export class OtelExporter implements TraceExporter {
  private readonly url: string;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof globalThis.fetch;
  private readonly onError: (error: Error) => void;
  private readonly resource: OtlpResourceOptions;

  constructor(options: OtelExporterOptions) {
    // `/v1/traces` is the path OTLP fixes; an endpoint that already names it is
    // left alone, because both forms are what collectors print in their docs.
    const base = options.endpoint.replace(/\/+$/, "");
    this.url = base.endsWith("/v1/traces") ? base : `${base}/v1/traces`;
    this.headers = options.headers ?? {};
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    this.onError = options.onError ?? ((error) => console.error(error.message));
    this.resource = {
      ...(options.serviceName !== undefined ? { serviceName: options.serviceName } : {}),
      ...(options.resourceAttributes !== undefined ? { resourceAttributes: options.resourceAttributes } : {}),
    };
  }

  async export(run: Run): Promise<void> {
    try {
      const response = await this.fetchImpl(this.url, {
        method: "POST",
        headers: { "content-type": "application/json", ...this.headers },
        body: JSON.stringify(toOtlpTraces(run, this.resource)),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      const body = await response.text().catch(() => "");
      if (!response.ok) {
        throw new Error(`collector answered ${response.status}${body ? `: ${body.slice(0, 500)}` : ""}`);
      }
      const rejected = rejectedSpans(body);
      if (rejected) throw new Error(rejected);
    } catch (err) {
      // Named with the endpoint: the one thing the caller cannot tell from the
      // message is which collector of theirs is not taking traces.
      const message = err instanceof Error ? err.message : String(err);
      this.onError(new Error(`otel: export to ${this.url} failed: ${message}`, { cause: err }));
    }
  }
}

/**
 * OTLP answers 200 with a `partialSuccess` body when the collector took the
 * request but kept only some of the spans. Unread, an export that lost half a
 * trace is indistinguishable from one that worked.
 */
function rejectedSpans(body: string): string | undefined {
  if (!body.includes("partialSuccess")) return undefined;
  try {
    const parsed = JSON.parse(body) as { partialSuccess?: { rejectedSpans?: string | number; errorMessage?: string } };
    const count = parsed.partialSuccess?.rejectedSpans;
    if (count === undefined || Number(count) === 0) return undefined;
    return `collector rejected ${count} span(s)${parsed.partialSuccess?.errorMessage ? `: ${parsed.partialSuccess.errorMessage}` : ""}`;
  } catch {
    return undefined;
  }
}

/**
 * Nanoseconds since the epoch, as the decimal string proto3's JSON mapping
 * writes an int64 as. `ms * 1e6` is past `Number.MAX_SAFE_INTEGER` for any
 * real clock, so the arithmetic is BigInt over whole milliseconds: exact by
 * construction, rather than by what shortest-round-trip printing happens to
 * make of a double that large.
 */
function nanos(ms: number): string {
  return (BigInt(Math.round(Number.isFinite(ms) ? ms : 0)) * 1_000_000n).toString();
}

/**
 * OTLP ids are fixed-width hex and a collector drops a span whose ids are not:
 * `randomUUID` is already 32 hex digits, so a run id is a trace id as it
 * stands and a span id is its first 16. Anything narrower — an id from a
 * custom tracer, or one a store rewrote — is hashed to the same width rather
 * than padded, which would have every short id collide on a run of zeros.
 */
function hexId(id: string, width: 16 | 32): string {
  const hex = id.replace(/[^0-9a-f]/gi, "").toLowerCase().slice(0, width);
  // All-zero is OTLP's way of saying "no id", so it cannot stand for one.
  if (hex.length === width && /[^0]/.test(hex)) return hex;
  return createHash("sha256").update(id).digest("hex").slice(0, width);
}

/** UNSET rather than OK on success: OTel reserves OK for a status set deliberately. */
function status(failed: boolean, message?: string): OtlpSpan["status"] {
  if (!failed) return { code: OTLP_STATUS.unset };
  return { code: OTLP_STATUS.error, ...(message ? { message } : {}) };
}

/** Token counts, under the GenAI semantic conventions where they have a name. */
function usageAttributes(usage: Usage): Record<string, unknown> {
  return {
    "gen_ai.usage.input_tokens": usage.inputTokens,
    "gen_ai.usage.output_tokens": usage.outputTokens,
    // No convention covers a prompt cache yet, and folding these into
    // input_tokens would misreport the call: they bill at their own rates.
    "agent_runtime.usage.cache_read_tokens": usage.cacheReadTokens,
    "agent_runtime.usage.cache_write_tokens": usage.cacheWriteTokens,
  };
}

/** The attributes a backend reads off one span: model, tokens, cost, tool name. */
function spanAttributes(span: Span): OtlpAttribute[] {
  const semantic: Record<string, unknown> =
    span.kind === "model.call"
      ? {
          "gen_ai.operation.name": "chat",
          // The span's name is the model the loop asked for; `span.model` is
          // the one that answered, and a fallback chain makes them differ.
          "gen_ai.request.model": span.name,
          ...(span.model ? { "gen_ai.response.model": span.model } : {}),
        }
      : span.kind === "tool.call"
        ? { "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": span.name }
        : {};

  return [
    ...attributes({
      ...semantic,
      ...(span.usage ? usageAttributes(span.usage) : {}),
    }),
    // `null` is "this model is not in the price table", which is not zero: the
    // attribute is left off so a dashboard sums only what it can account for.
    ...(typeof span.costUsd === "number" ? [cost(span.costUsd)] : []),
    ...attributes(prefixed(span.attributes)),
  ];
}

/** The run's own attributes plus its totals, on the root span. */
function runAttributes(run: Run): OtlpAttribute[] {
  const totals = run.totals;
  return [
    ...attributes({
      "agent_runtime.model_calls": totals.modelCalls,
      "agent_runtime.tool_calls": totals.toolCalls,
      "agent_runtime.tool_errors": totals.toolErrors,
      ...usageAttributes(totals.usage),
    }),
    ...(totals.costUsd !== null ? [cost(totals.costUsd)] : []),
    ...attributes(prefixed(run.attributes)),
  ];
}

/**
 * Cost is money: always a double, even on a whole number, so the field does
 * not change type between spans and become unsummable.
 */
function cost(costUsd: number): OtlpAttribute {
  return { key: "agent_runtime.cost_usd", value: { doubleValue: costUsd } };
}

/**
 * `stopReason` → `agent_runtime.stop_reason`. Attribute names are snake_case
 * by convention, and the prefix is what keeps a caller's own attribute from
 * landing on a name a semantic convention already means something by.
 */
function prefixed(source: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(source)) {
    out[`agent_runtime.${key.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase()}`] = value;
  }
  return out;
}

function attributes(source: Record<string, unknown>): OtlpAttribute[] {
  const out: OtlpAttribute[] = [];
  for (const [key, raw] of Object.entries(source)) {
    const value = toOtlpValue(raw);
    if (value) out.push({ key, value });
  }
  return out;
}

/**
 * One attribute value on the wire. `undefined` means "leave the attribute
 * out": OTLP has no null, and an attribute carrying nothing is noise a
 * backend still indexes. Objects go as JSON — the loop's `suspended`
 * descriptor is one, and a backend that cannot filter on it can still show it.
 */
export function toOtlpValue(value: unknown): OtlpValue | undefined {
  if (typeof value === "string") return { stringValue: value };
  if (typeof value === "boolean") return { boolValue: value };
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return undefined;
    return Number.isInteger(value) ? { intValue: String(value) } : { doubleValue: value };
  }
  if (value === null || value === undefined) return undefined;
  if (Array.isArray(value)) {
    return { arrayValue: { values: value.map(toOtlpValue).filter((v): v is OtlpValue => v !== undefined) } };
  }
  return { stringValue: JSON.stringify(value) };
}
