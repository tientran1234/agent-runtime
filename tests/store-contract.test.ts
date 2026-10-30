import { describe, expect, it } from "vitest";
import {
  MemoryStore,
  assertRunStore,
  runStoreContract,
  type RunStore,
  type SuspendedRun,
} from "../src/index.js";

/**
 * A store that is a `Map` and nothing else: it validates no id, serializes
 * nothing and checks no version. Everything it gets wrong, it gets wrong by
 * leaving a check out — which is how a store written against the port's four
 * signatures and nothing else is wrong.
 */
class LaxStore implements RunStore {
  private readonly rows = new Map<string, SuspendedRun>();

  async put(runId: string, state: SuspendedRun): Promise<void> {
    this.rows.set(runId, state);
  }

  async get(runId: string): Promise<SuspendedRun | undefined> {
    return this.rows.get(runId);
  }

  async delete(runId: string): Promise<void> {
    this.rows.delete(runId);
  }

  async pending(): Promise<string[]> {
    return [...this.rows.keys()];
  }
}

/** Stores one check away from `MemoryStore`, each dropping a different promise. */
class LosesTheSnapshot extends MemoryStore {
  override async put(): Promise<void> {}
}

class ThrowsOnAMissingRun extends MemoryStore {
  override async get(runId: string): Promise<SuspendedRun> {
    const state = await super.get(runId);
    if (state === undefined) throw new Error(`no run ${runId}`);
    return state;
  }
}

class InsertsOnlyOnce extends MemoryStore {
  override async put(runId: string, state: SuspendedRun): Promise<void> {
    if ((await super.get(runId)) !== undefined) return;
    await super.put(runId, state);
  }
}

class RefusesASecondDelete extends MemoryStore {
  override async delete(runId: string): Promise<void> {
    if ((await super.get(runId)) === undefined) throw new Error(`no run ${runId} to delete`);
    await super.delete(runId);
  }
}

class NamesNothingWaiting extends MemoryStore {
  override async pending(): Promise<string[]> {
    return [];
  }
}

/**
 * Each of these is a store someone could plausibly write, paired with the
 * cases that have to catch it. A contract nothing fails is a contract that
 * holds nobody to anything, so the cases are pinned from this side too: by a
 * store that breaks exactly one promise and the case that must say so.
 */
const BROKEN: Array<{ what: string; open: () => RunStore; caught: string[] }> = [
  {
    what: "stores nothing at all",
    open: () => new LosesTheSnapshot(),
    caught: ["hands back a snapshot a resumed run finishes from"],
  },
  {
    what: "treats an id it does not hold as an error",
    open: () => new ThrowsOnAMissingRun(),
    caught: ["reports an id it does not hold as undefined rather than throwing"],
  },
  {
    what: "inserts but never replaces",
    open: () => new InsertsOnlyOnce(),
    caught: ["replaces a snapshot, because a run that suspends twice is one run"],
  },
  {
    what: "refuses to forget a run twice",
    open: () => new RefusesASecondDelete(),
    caught: ["forgets a run, and forgetting one twice is not an error"],
  },
  {
    what: "never says what is waiting",
    open: () => new NamesNothingWaiting(),
    caught: ["names the runs still waiting"],
  },
  {
    what: "leaves out the id, serialization and version checks",
    open: () => new LaxStore(),
    caught: [
      "refuses a run id that is a path rather than a name",
      "refuses at put a snapshot no real backend could have stored",
      "refuses a snapshot from a writer newer than this reader",
    ],
  },
];

describe("the RunStore contract", () => {
  it.each(BROKEN)("catches a store that $what", async ({ open, caught }) => {
    const error = await assertRunStore(open).then(
      () => undefined,
      (err: unknown) => err as Error,
    );

    expect(error, "the contract accepted a store that is wrong").toBeDefined();
    for (const name of caught) expect(error!.message).toContain(name);
  });

  it("accepts a store that keeps every promise", async () => {
    await expect(assertRunStore(() => new MemoryStore())).resolves.toBeUndefined();
  });

  /**
   * A case nothing above fails is a case that could be asserting anything, so
   * adding one to the contract means adding the store it catches.
   */
  it("has a broken store for every case it runs", () => {
    const caught = new Set(BROKEN.flatMap((broken) => broken.caught));

    expect(runStoreContract().map((runStoreCase) => runStoreCase.name).filter((name) => !caught.has(name))).toEqual([]);
    expect(runStoreContract().length).toBeGreaterThan(0);
  });
});
