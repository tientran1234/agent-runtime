import type { RunStore } from "./store.js";

/**
 * How the contract gets a store to run a case against: one that holds nothing
 * yet. Every case below puts what it needs and reads back what it put, so a
 * store arriving with another case's rows in it fails on a `pending` that
 * names them — which is the contract catching the harness rather than the
 * store. A file-backed store opens on a fresh directory here, a row-backed one
 * truncates its table or uses one of its own.
 */
export type OpenRunStore = () => RunStore | Promise<RunStore>;

/**
 * One promise a `RunStore` makes, as something runnable. `run` throws if the
 * store does not keep it, with a message that says what was expected and what
 * came back, so a failure reads the same whichever runner printed it.
 */
export interface RunStoreCase {
  /** What this case holds a store to — the line a runner names the test with. */
  name: string;
  run(store: RunStore): Promise<void>;
}

/**
 * Every way a store that is not one of these three can be wrong, in no order
 * that matters.
 *
 * Handed over as cases rather than as a test file, for the same reason
 * `SqlStore.schema()` is a string rather than a migration: which runner these
 * go through is the caller's, and a suite that imported `vitest` would decide
 * that for them. A caller whose suspended runs wait in DynamoDB, in a bucket
 * or on a queue writes four methods and then has the question every port
 * leaves behind — whether what they wrote keeps the promises the loop is
 * written against — answered here instead of in the deployment.
 */
export function runStoreContract(): readonly RunStoreCase[] {
  return [];
}

/**
 * The whole contract, as one assertion: every case on a store of its own,
 * every failure at once.
 *
 * Collected rather than thrown on the first, like `assertScenario`. A store
 * that fails one case usually fails three, and the three together say which
 * promise was misread — where the first alone only says where to put a
 * breakpoint.
 */
export async function assertRunStore(open: OpenRunStore): Promise<void> {
  const failures: string[] = [];
  for (const theCase of runStoreContract()) {
    try {
      await theCase.run(await open());
    } catch (err) {
      failures.push(`${theCase.name}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  if (failures.length > 0) {
    throw new Error(
      `this is not a RunStore the agent loop can suspend into — ${failures.length} of ` +
        `${runStoreContract().length} case${runStoreContract().length === 1 ? "" : "s"} failed:\n\n` +
        failures.map((failure) => `  ${failure}`).join("\n\n"),
    );
  }
}
