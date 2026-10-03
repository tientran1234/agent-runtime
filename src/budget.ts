import { costUsd, type Price } from "./pricing.js";
import { EMPTY_USAGE, addUsage, type ModelResponse, type Usage } from "./types.js";

/**
 * Ceilings for one run. Both are totals across every model call rather than
 * per-call limits, and both are optional — a run that sets neither has no
 * budget and is bounded by `maxIterations` alone.
 */
export interface BudgetOptions {
  /**
   * Dollars the run may spend. A model the price table does not know makes the
   * spend unmeasurable, which ends the run too: a cap nobody can check is not a
   * cap, and carrying on would break the promise without saying so.
   */
  maxCostUsd?: number;
  /**
   * Tokens the run may send. Cache reads and writes count — they are tokens the
   * model was given, whatever rate they were billed at.
   */
  maxInputTokens?: number;
}

/**
 * A ledger's totals as plain JSON. A run that can be suspended has to carry
 * them, or its cap would restart from zero in the process that finishes it and
 * bound two short runs instead of one long one.
 */
export interface BudgetState {
  usage: Usage;
  /** Null once a call's price was unknown. Restored as null, not as zero. */
  spentUsd: number | null;
  /** The last call, which is what the next one is forecast from. Absent before the first call. */
  last?: { model: string; usage: Usage; costUsd: number | null };
}

/** Every token that went in, by whichever route it was billed. */
function inputTokensOf(usage: Usage): number {
  return usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens;
}

const dollars = (amount: number) => `$${amount.toFixed(5)}`;

/**
 * What a run has spent, and whether the next call still fits inside its budget.
 *
 * The totals are the tracer's: the same usage sum, the same `costUsd` against
 * the same price table, and the same rule that one unknown price makes the
 * total `null` instead of a quietly smaller number. The ledger keeps them
 * itself so that setting a budget does not oblige the caller to attach a
 * tracer.
 *
 * Judging a call before it is made means forecasting one that has not happened,
 * and the forecast is the last call's own usage: the loop only appends to the
 * transcript, so the next call sends at least what the last one did and costs
 * at least as much. That keeps the guard sound — it stops only where it can
 * already show the cap would be passed, never on a call that would have fit. A
 * memory window that trims breaks the "at least" and can make it stop one call
 * early, which is the side to be wrong on for a budget.
 *
 * What no forecast can do is turn the cap into a hard ceiling: how long an
 * answer runs is not knowable before asking for it, so a run may overshoot by
 * up to one call's worth. The cap bounds a run to within one call, which is
 * what stops a tool loop from spending all afternoon.
 */
export class BudgetLedger {
  private usage: Usage;
  /** Null once a call's price is unknown, exactly as the tracer's total goes null. */
  private spentUsd: number | null;
  private last: { model: string; usage: Usage; costUsd: number | null } | undefined;

  constructor(
    private readonly limits: BudgetOptions,
    private readonly prices: Record<string, Price> | undefined = undefined,
    state?: BudgetState,
  ) {
    this.usage = state?.usage ?? EMPTY_USAGE;
    // Not `?? 0`: a restored `null` means a price was already unknown, and
    // coalescing it would turn an unmeasurable spend back into a measured zero.
    this.spentUsd = state === undefined ? 0 : state.spentUsd;
    this.last = state?.last;
  }

  /** The totals as plain JSON, for a snapshot that has to carry the budget with it. */
  get state(): BudgetState {
    return { usage: this.usage, spentUsd: this.spentUsd, ...(this.last ? { last: this.last } : {}) };
  }

  /** Add a finished call to the totals, and make it the forecast for the next one. */
  record(response: ModelResponse): void {
    const cost = costUsd(response.model, response.usage, this.prices);
    this.usage = addUsage(this.usage, response.usage);
    if (cost === null) this.spentUsd = null;
    else if (this.spentUsd !== null) this.spentUsd += cost;
    this.last = { model: response.model, usage: response.usage, costUsd: cost };
  }

  /**
   * Why the next call must not be made, or `undefined` while it fits. The
   * sentence is what the run records, so it names the limit and the numbers
   * behind the verdict. Nothing recorded yet means no forecast, so the first
   * call of a run is never the one stopped — a budget that could refuse to
   * start would be one nobody could set.
   */
  wouldExceed(): string | undefined {
    const last = this.last;
    if (!last) return undefined;
    const { maxInputTokens, maxCostUsd } = this.limits;

    if (maxInputTokens !== undefined) {
      const spent = inputTokensOf(this.usage);
      const forecast = inputTokensOf(last.usage);
      if (spent + forecast > maxInputTokens) {
        return `maxInputTokens ${maxInputTokens}: ${spent} sent, the next call adds at least ${forecast}`;
      }
    }

    if (maxCostUsd !== undefined) {
      if (this.spentUsd === null || last.costUsd === null) {
        return `maxCostUsd ${dollars(maxCostUsd)}: no price for ${last.model}, so the spend cannot be checked`;
      }
      if (this.spentUsd + last.costUsd > maxCostUsd) {
        return `maxCostUsd ${dollars(maxCostUsd)}: ${dollars(this.spentUsd)} spent, the next call adds at least ${dollars(last.costUsd)}`;
      }
    }

    return undefined;
  }
}
