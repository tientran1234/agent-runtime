import { z } from "zod";
import { resumeAgent, runAgent } from "./loop.js";
import { FakeProvider, callTools, reply } from "./providers/fake.js";
import type { SuspendedRun } from "./resume.js";
import { MAX_RUN_ID, type RunStore } from "./store.js";
import { defineTool } from "./tools.js";

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

/** The tool a suspended run is waiting on, recording every time it really ran. */
const refundOrder = (ran: string[]) =>
  defineTool({
    name: "refund_order",
    description: "Refund an order",
    input: z.object({ orderId: z.string() }),
    execute: ({ orderId }) => (ran.push(orderId), `refunded ${orderId}`),
  });

/**
 * A snapshot from a run that really suspended, rather than a literal written
 * to look like one. What a store has to survive is the shape the loop writes —
 * the transcript, the settled calls, the totals — and a fixture is a chance to
 * simplify away the part that would not have fitted.
 */
async function suspend(ran: string[]): Promise<SuspendedRun> {
  const result = await runAgent({
    provider: new FakeProvider([callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }])]),
    input: "refund ord_42",
    tools: [refundOrder(ran)],
    beforeToolCall: () => ({ ask: true }),
  });
  // Not an assertion about the store: if this fails, the loop stopped
  // suspending and every case below is testing nothing.
  if (result.suspended === undefined) throw new Error(`the contract's own run did not suspend — it ended ${result.status}`);
  return result.suspended;
}

/** Ids the port refuses, because a run id is a name and not a path. */
const NOT_A_RUN_ID = ["../escape", "runs/42", ".", "..", "", ".hidden", "a".repeat(MAX_RUN_ID + 1)];

/** A snapshot JSON cannot carry, which a store has to refuse before it replaces the one it holds. */
function unserializable(state: SuspendedRun): SuspendedRun {
  const circular = { ...state } as SuspendedRun & { self?: unknown };
  circular.self = circular;
  return circular;
}

function equal(what: string, actual: unknown, expected: unknown): void {
  const [got, wanted] = [JSON.stringify(actual), JSON.stringify(expected)];
  if (got !== wanted) throw new Error(`${what}: expected ${wanted}, got ${got}`);
}

/**
 * A call that has to be refused, and refused in a way that says why. The
 * message is matched because a store that rejects everything with one blank
 * error passes a check for "it threw" while telling an operator nothing.
 */
async function refuses(what: string, says: RegExp, call: () => Promise<unknown>): Promise<void> {
  let value: unknown;
  try {
    value = await call();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (!says.test(message)) throw new Error(`${what}: refused with ${JSON.stringify(message)}, which does not say ${String(says)}`);
    return;
  }
  throw new Error(`${what}: was not refused — it answered ${JSON.stringify(value) ?? "undefined"}`);
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
  return [
    {
      name: "hands back a snapshot a resumed run finishes from",
      async run(store) {
        // One log across both runs: the refund has to happen on the resume and
        // only there.
        const ran: string[] = [];
        await store.put("run-42", await suspend(ran));

        const state = await store.get("run-42");
        if (state === undefined) throw new Error("get answered undefined for a run that had just been put");
        const result = await resumeAgent({
          provider: new FakeProvider([reply("Refunded ord_42.")]),
          tools: [refundOrder(ran)],
          state,
          decisions: { r1: { allow: true } },
        });

        equal("the resumed run's status", result.status, "completed");
        equal("what the resumed run answered", result.text, "Refunded ord_42.");
        equal("the orders refunded across the suspended run and the resumed one", ran, ["ord_42"]);
      },
    },
    {
      name: "reports an id it does not hold as undefined rather than throwing",
      async run(store) {
        // Absent is an answer: the normal way to read this store is to poll for
        // a decision that may already have been taken and the run cleaned up.
        equal("get on an id never stored", await store.get("never-stored"), undefined);
      },
    },
    {
      name: "replaces a snapshot, because a run that suspends twice is one run",
      async run(store) {
        const state = await suspend([]);
        await store.put("run-42", state);
        await store.put("run-42", { ...state, iterations: 4 });

        equal("the iterations on the snapshot read back", (await store.get("run-42"))?.iterations, 4);
        equal("what is waiting after one run suspended twice", await store.pending(), ["run-42"]);
      },
    },
    {
      name: "forgets a run, and forgetting one twice is not an error",
      async run(store) {
        await store.put("run-42", await suspend([]));
        await store.delete("run-42");

        equal("get on a deleted run", await store.get("run-42"), undefined);
        // A resumed run is cleaned up by whoever gets there first, which may
        // well be twice; the second attempt wants the first's outcome.
        await store.delete("run-42");
        equal("what is waiting after the only run was deleted", await store.pending(), []);
      },
    },
    {
      name: "names the runs still waiting",
      async run(store) {
        const state = await suspend([]);
        await store.put("run-1", state);
        await store.put("run-2", state);

        // Sorted here, not there: the order is nobody's promise.
        equal("the ids waiting", (await store.pending()).sort(), ["run-1", "run-2"]);
      },
    },
    {
      name: "refuses a run id that is a path rather than a name",
      async run(store) {
        const state = await suspend([]);
        for (const bad of NOT_A_RUN_ID) {
          const shown = JSON.stringify(bad);
          await refuses(`put with the id ${shown}`, /not a usable run id/, () => store.put(bad, state));
          await refuses(`get with the id ${shown}`, /not a usable run id/, () => store.get(bad));
          await refuses(`delete with the id ${shown}`, /not a usable run id/, () => store.delete(bad));
        }
        // Checked from the other side as well: a store that threw but stored
        // the row anyway has escaped wherever it keeps its rows.
        equal("what is waiting after every id was refused", await store.pending(), []);
      },
    },
    {
      name: "refuses at put a snapshot no real backend could have stored",
      async run(store) {
        // At `put`, not at `get`: a snapshot that cannot be serialized must not
        // have replaced the stored one that could, and an in-memory store that
        // held the object would only find that out once a row was involved.
        const state = unserializable(await suspend([]));
        await refuses("put with a snapshot that is not JSON", /is not JSON/, () => store.put("run-42", state));
      },
    },
    {
      name: "refuses a snapshot from a writer newer than this reader",
      async run(store) {
        const state = await suspend([]);
        await store.put("run-42", { ...state, version: 2 } as unknown as SuspendedRun);

        // A snapshot from a later release is not a run this code can finish.
        // Passed through as an object, it reaches the loop half-understood and
        // fails somewhere that says nothing about why.
        await refuses("get on a snapshot written by a newer version", /run-42.*version 2/, () => store.get("run-42"));
      },
    },
  ];
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
  const cases = runStoreContract();
  const failures: string[] = [];
  for (const storeCase of cases) {
    try {
      await storeCase.run(await open());
    } catch (err) {
      failures.push(`${storeCase.name}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  if (failures.length > 0) {
    throw new Error(
      `this is not a RunStore the agent loop can suspend into — ${failures.length} of ` +
        `${cases.length} case${cases.length === 1 ? "" : "s"} failed:\n\n` +
        failures.map((failure) => `  ${failure}`).join("\n\n"),
    );
  }
}
