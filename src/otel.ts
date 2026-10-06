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
import type { Run, TraceExporter } from "./trace.js";

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

/** A finished run as one OTLP payload: the run is the root span, its spans the children. */
export function toOtlpTraces(_run: Run, _options: OtlpResourceOptions = {}): OtlpTraces {
  throw new Error("not implemented");
}

/**
 * Posts each finished run to an OTLP/HTTP collector.
 *
 * A failed export is reported, never thrown: `RunHandle.end()` awaits its
 * exporters, so a collector that is down would otherwise turn every finished
 * run into a failed one.
 */
export class OtelExporter implements TraceExporter {
  constructor(_options: OtelExporterOptions) {}

  async export(_run: Run): Promise<void> {
    throw new Error("not implemented");
  }
}
