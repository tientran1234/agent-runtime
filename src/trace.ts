import { randomUUID } from "node:crypto";
import { costUsd, type Price } from "./pricing.js";
import { EMPTY_USAGE, addUsage, type Usage } from "./types.js";

export type SpanKind = "model.call" | "tool.call" | "custom";

export interface Span {
  id: string;
  runId: string;
  kind: SpanKind;
  name: string;
  startedAt: number;
  endedAt?: number;
  durationMs?: number;
  attributes: Record<string, unknown>;
  usage?: Usage;
  model?: string;
  costUsd?: number | null;
  error?: string;
}

export interface RunTotals {
  modelCalls: number;
  toolCalls: number;
  toolErrors: number;
  usage: Usage;
  /** Null if any model call had an unknown price — a partial total would be a lie. */
  costUsd: number | null;
}

export interface Run {
  id: string;
  name: string;
  attributes: Record<string, unknown>;
  startedAt: number;
  endedAt?: number;
  durationMs?: number;
  status: "running" | "ok" | "error";
  error?: string;
  spans: Span[];
  totals: RunTotals;
}

export interface TraceExporter {
  export(run: Run): void | Promise<void>;
}

/** Keeps every finished run in memory. For tests, dashboards, and small deployments. */
export class MemoryExporter implements TraceExporter {
  readonly runs: Run[] = [];
  export(run: Run) {
    this.runs.push(run);
  }
}

/** One line per span, then totals — enough to see what an agent did and what it cost. */
export class ConsoleExporter implements TraceExporter {
  constructor(private readonly log: (line: string) => void = console.log) {}
  export(run: Run) {
    this.log(`run ${run.name} [${run.status}] ${run.durationMs}ms`);
    for (const s of run.spans) {
      const cost = s.costUsd === undefined ? "" : s.costUsd === null ? " cost=?" : ` cost=$${s.costUsd.toFixed(5)}`;
      const tokens = s.usage ? ` in=${s.usage.inputTokens} out=${s.usage.outputTokens}` : "";
      this.log(`  ${s.kind.padEnd(10)} ${s.name.padEnd(24)} ${String(s.durationMs).padStart(6)}ms${tokens}${cost}${s.error ? ` ERROR ${s.error}` : ""}`);
    }
    const t = run.totals;
    this.log(`  totals: ${t.modelCalls} model calls, ${t.toolCalls} tool calls (${t.toolErrors} failed), ${t.usage.inputTokens}+${t.usage.outputTokens} tokens, cost ${t.costUsd === null ? "unknown" : `$${t.costUsd.toFixed(5)}`}`);
  }
}

export interface TracerOptions {
  exporters?: TraceExporter[];
  prices?: Record<string, Price>;
  now?: () => number;
}

export class SpanHandle {
  constructor(
    readonly span: Span,
    private readonly tracer: Tracer,
    private readonly run: Run,
  ) {}

  setAttributes(attributes: Record<string, unknown>): this {
    Object.assign(this.span.attributes, attributes);
    return this;
  }

  /** Attach token usage; cost is looked up from the price table for `model`. */
  recordUsage(model: string, usage: Usage): this {
    this.span.model = model;
    this.span.usage = usage;
    this.span.costUsd = costUsd(model, usage, this.tracer.prices);
    return this;
  }

  end(error?: unknown): Span {
    const now = this.tracer.now();
    this.span.endedAt = now;
    this.span.durationMs = now - this.span.startedAt;
    if (error !== undefined) this.span.error = error instanceof Error ? error.message : String(error);
    this.tracer.accumulate(this.run, this.span);
    return this.span;
  }
}

export class RunHandle {
  constructor(
    readonly run: Run,
    private readonly tracer: Tracer,
  ) {}

  get id() {
    return this.run.id;
  }

  startSpan(kind: SpanKind, name: string, attributes: Record<string, unknown> = {}): SpanHandle {
    const span: Span = { id: randomUUID(), runId: this.run.id, kind, name, startedAt: this.tracer.now(), attributes };
    this.run.spans.push(span);
    return new SpanHandle(span, this.tracer, this.run);
  }

  async end(error?: unknown): Promise<Run> {
    const now = this.tracer.now();
    this.run.endedAt = now;
    this.run.durationMs = now - this.run.startedAt;
    this.run.status = error === undefined ? "ok" : "error";
    if (error !== undefined) this.run.error = error instanceof Error ? error.message : String(error);
    for (const exporter of this.tracer.exporters) await exporter.export(this.run);
    return this.run;
  }
}

export class Tracer {
  readonly exporters: TraceExporter[];
  readonly prices: Record<string, Price> | undefined;
  readonly now: () => number;

  constructor(options: TracerOptions = {}) {
    this.exporters = options.exporters ?? [];
    this.prices = options.prices;
    this.now = options.now ?? Date.now;
  }

  startRun(name: string, attributes: Record<string, unknown> = {}): RunHandle {
    const run: Run = {
      id: randomUUID(),
      name,
      attributes,
      startedAt: this.now(),
      status: "running",
      spans: [],
      totals: { modelCalls: 0, toolCalls: 0, toolErrors: 0, usage: EMPTY_USAGE, costUsd: 0 },
    };
    return new RunHandle(run, this);
  }

  /** @internal */
  accumulate(run: Run, span: Span) {
    const t = run.totals;
    if (span.kind === "model.call") {
      t.modelCalls++;
      if (span.usage) t.usage = addUsage(t.usage, span.usage);
      if (span.costUsd === null || span.costUsd === undefined) t.costUsd = null;
      else if (t.costUsd !== null) t.costUsd += span.costUsd;
    } else if (span.kind === "tool.call") {
      t.toolCalls++;
      if (span.error || span.attributes.isError === true) t.toolErrors++;
    }
  }
}
