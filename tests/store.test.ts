import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { z } from "zod";
import {
  FakeProvider,
  FileStore,
  MemoryStore,
  SqlStore,
  callTools,
  defineTool,
  runAgent,
  runStoreContract,
  type RunStore,
  type SqlQuery,
  type SuspendedRun,
} from "../src/index.js";

/** A tool that records every time it really ran, which a stored-and-resumed run must not duplicate. */
const refund = (log: string[]) =>
  defineTool({
    name: "refund_order",
    description: "Refund an order",
    input: z.object({ orderId: z.string() }),
    execute: ({ orderId }) => (log.push(orderId), `refunded ${orderId}`),
  });

/**
 * A snapshot from a real run rather than a hand-written one: what a store has
 * to survive is the shape the loop actually writes, including the parts a
 * hand-rolled fixture would simplify away.
 */
async function suspend(log: string[] = []): Promise<SuspendedRun> {
  const provider = new FakeProvider([callTools([{ name: "refund_order", input: { orderId: "ord_42" }, id: "r1" }])]);
  const result = await runAgent({
    provider,
    input: "refund ord_42",
    tools: [refund(log)],
    beforeToolCall: () => ({ ask: true }),
  });
  return result.suspended!;
}

/** A snapshot JSON cannot carry, which both stores have to reject at `put`. */
function unserializable(state: SuspendedRun): SuspendedRun {
  const circular = { ...state } as SuspendedRun & { self?: unknown };
  circular.self = circular;
  return circular;
}

/**
 * A database that answers the four statements `SqlStore` writes, and nothing
 * else. What the store has to be held to is which statements it sends and how
 * it maps a snapshot either way; executing SQL is the driver's job, so the fake
 * recognises the store's own statements rather than parsing them — and one it
 * does not recognise fails loudly, because a store whose `put` quietly did
 * nothing would otherwise pass every test below.
 */
function fakeDb(): { query: SqlQuery; sent: Array<{ sql: string; params: readonly string[] }> } {
  const rows = new Map<string, string>();
  const sent: Array<{ sql: string; params: readonly string[] }> = [];
  const query: SqlQuery = async (sql, params) => {
    sent.push({ sql, params });
    if (sql.startsWith("INSERT")) {
      rows.set(params[0]!, params[1]!);
      return [];
    }
    if (sql.startsWith("SELECT snapshot")) {
      const snapshot = rows.get(params[0]!);
      return snapshot === undefined ? [] : [{ snapshot }];
    }
    if (sql.startsWith("DELETE")) {
      rows.delete(params[0]!);
      return [];
    }
    if (sql.startsWith("SELECT run_id")) return [...rows.keys()].map((run_id) => ({ run_id }));
    throw new Error(`fake database: no answer for ${sql}`);
  };
  return { query, sent };
}

describe.each([
  { name: "MemoryStore", open: (_dir: string): RunStore => new MemoryStore() },
  { name: "FileStore", open: (dir: string): RunStore => new FileStore(join(dir, "runs")) },
  { name: "SqlStore", open: (_dir: string): RunStore => new SqlStore({ query: fakeDb().query }) },
])("$name as a RunStore", ({ open }) => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "agent-runtime-store-"));
  });
  afterEach(() => rm(dir, { recursive: true, force: true }));

  // The contract this library publishes, run against the three stores it
  // ships. Written once and read from `runStoreContract` rather than kept here
  // as well, because a caller holding their own store to a copy of these cases
  // would be holding it to whichever copy stopped being edited first.
  it.each(runStoreContract())("$name", async (storeCase) => {
    await storeCase.run(open(dir));
  });
});

describe("MemoryStore", () => {
  it("hands back a copy, so a caller holding one cannot edit what is stored", async () => {
    const store = new MemoryStore();
    await store.put("run-42", await suspend());

    const first = (await store.get("run-42"))!;
    first.messages.length = 0;
    first.awaiting.length = 0;

    expect((await store.get("run-42"))?.messages).toHaveLength(2);
    expect((await store.get("run-42"))?.awaiting).toEqual(["r1"]);
  });
});

describe("FileStore", () => {
  let dir: string;
  let runs: string;
  let store: FileStore;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "agent-runtime-store-"));
    runs = join(dir, "runs");
    store = new FileStore(runs);
  });
  afterEach(() => rm(dir, { recursive: true, force: true }));

  it("leaves one file per run and no temporary residue", async () => {
    const state = await suspend();
    await store.put("run-42", state);
    await store.put("run-42", state);

    expect(await readdir(runs)).toEqual(["run-42.json"]);
  });

  it("keeps the stored snapshot when a later put cannot be written", async () => {
    await store.put("run-42", await suspend());
    await expect(store.put("run-42", unserializable(await suspend()))).rejects.toThrow(/is not JSON/);

    // The snapshot is the only copy of a run whose tools have not run, so a put
    // that fails has to leave the readable one behind rather than half of a new
    // one in its place.
    expect((await store.get("run-42"))?.awaiting).toEqual(["r1"]);
    expect(await readdir(runs)).toEqual(["run-42.json"]);
  });

  it("reads a directory nothing has suspended into as empty, not as broken", async () => {
    expect(await store.pending()).toEqual([]);
    expect(await store.get("run-42")).toBeUndefined();
  });

  it("ignores files it did not write", async () => {
    await store.put("run-42", await suspend());
    await writeFile(join(runs, "notes.txt"), "mine, not the store's", "utf8");
    await writeFile(join(runs, ".editor-swap.json"), "{}", "utf8");

    expect(await store.pending()).toEqual(["run-42"]);
  });

  it("refuses a truncated file rather than half-reading it", async () => {
    await mkdir(runs, { recursive: true });
    await writeFile(join(runs, "run-42.json"), '{"version":1,"messages":[{"role":"use', "utf8");

    await expect(store.get("run-42")).rejects.toThrow(/run-42/);
  });
});

describe("SqlStore", () => {
  it("stores a snapshot as one upsert, because a run that suspends twice is one row", async () => {
    const db = fakeDb();
    const store = new SqlStore({ query: db.query });
    const state = await suspend();
    await store.put("run-42", state);

    // One statement rather than a read and then a write: two processes putting
    // the same run must not be able to interleave into a lost snapshot.
    expect(db.sent).toHaveLength(1);
    expect(db.sent[0]?.sql).toBe(
      "INSERT INTO agent_runs (run_id, snapshot) VALUES ($1, $2) " +
        "ON CONFLICT (run_id) DO UPDATE SET snapshot = excluded.snapshot",
    );
    // The id and the snapshot travel as parameters, never as statement text.
    expect(db.sent[0]?.params).toEqual(["run-42", JSON.stringify(state)]);
  });

  it("reads, forgets and lists by id alone", async () => {
    const db = fakeDb();
    const store = new SqlStore({ query: db.query, table: "approvals.agent_runs" });
    await store.get("run-42");
    await store.delete("run-42");
    await store.pending();

    expect(db.sent.map((s) => s.sql)).toEqual([
      "SELECT snapshot FROM approvals.agent_runs WHERE run_id = $1",
      "DELETE FROM approvals.agent_runs WHERE run_id = $1",
      "SELECT run_id FROM approvals.agent_runs",
    ]);
  });

  it("spells the placeholder and the upsert the way the dialect does", async () => {
    for (const { dialect, sql } of [
      {
        dialect: "sqlite" as const,
        sql:
          "INSERT INTO agent_runs (run_id, snapshot) VALUES (?, ?) " +
          "ON CONFLICT (run_id) DO UPDATE SET snapshot = excluded.snapshot",
      },
      {
        dialect: "mysql" as const,
        sql:
          "INSERT INTO agent_runs (run_id, snapshot) VALUES (?, ?) " +
          "ON DUPLICATE KEY UPDATE snapshot = VALUES(snapshot)",
      },
    ]) {
      const db = fakeDb();
      const store = new SqlStore({ query: db.query, dialect });
      await store.put("run-42", await suspend());
      await store.get("run-42");

      expect(db.sent[0]?.sql).toBe(sql);
      expect(db.sent[1]?.sql).toBe("SELECT snapshot FROM agent_runs WHERE run_id = ?");
    }
  });

  it("refuses a table name that is really a fragment of SQL, before any statement runs", async () => {
    const db = fakeDb();
    // A table name cannot be a parameter, so it is interpolated — and a store
    // that accepted this one would send whatever followed it to the database.
    for (const bad of ["agent_runs; DROP TABLE users", "agent runs", '"agent_runs"', "", "a.b.c", "1_runs"]) {
      expect(() => new SqlStore({ query: db.query, table: bad })).toThrow(/not a usable table name/);
    }
    expect(db.sent).toEqual([]);
  });

  it("refuses a snapshot column that is not the JSON text it wrote", async () => {
    // A `json` or `jsonb` column hands back a parsed value, which has never been
    // through the version check a stored snapshot is read with. Rejecting it
    // says so once, where a cast would hide a column type nobody meant.
    for (const snapshot of [{ version: 1 }, null, 42]) {
      const store = new SqlStore({ query: async () => [{ snapshot }] });
      await expect(store.get("run-42")).rejects.toThrow(/snapshot column for run-42/);
    }
  });

  it("hands back the create table its own statements read and write, without running it", async () => {
    const db = fakeDb();
    const store = new SqlStore({ query: db.query, table: "approvals.agent_runs" });

    expect(store.schema()).toBe(
      "CREATE TABLE IF NOT EXISTS approvals.agent_runs (run_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)",
    );
    // Handed over, not executed: a migration is the caller's to apply and roll
    // back, and a store that ran its own would be opening the pool `query`
    // exists to keep out of this library.
    expect(db.sent).toEqual([]);
  });

  it("declares a key MySQL will index and a column a whole snapshot fits in", () => {
    // MySQL indexes no `TEXT` key without a prefix length, and its `TEXT` stops
    // at 64 KiB — which a transcript carrying its tool results passes long
    // before it is a remarkable run. Either one wrong is not a statement that
    // fails but a snapshot stored short, and a snapshot is the only copy of a
    // run whose tools have not run.
    expect(new SqlStore({ query: fakeDb().query, dialect: "mysql" }).schema()).toBe(
      "CREATE TABLE IF NOT EXISTS agent_runs (run_id VARCHAR(200) PRIMARY KEY, snapshot LONGTEXT NOT NULL)",
    );
    expect(new SqlStore({ query: fakeDb().query, dialect: "sqlite" }).schema()).toBe(
      "CREATE TABLE IF NOT EXISTS agent_runs (run_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)",
    );
  });

  it("declares the id column as wide as the longest id it will accept", async () => {
    const store = new SqlStore({ query: fakeDb().query, dialect: "mysql" });
    const width = Number(/VARCHAR\((\d+)\)/.exec(store.schema())?.[1]);

    // The declared width and the check in front of every statement have to
    // agree: a column narrower than the ids this store accepts is a key stored
    // truncated, and a wider one is a column holding ids `get` will not ask for.
    await store.put("r".repeat(width), await suspend());
    await expect(store.put("r".repeat(width + 1), await suspend())).rejects.toThrow(/not a usable run id/);
  });

  it("ignores rows in the table it did not write", async () => {
    const store = new SqlStore({
      query: async () => [{ run_id: "run-1" }, { run_id: "../escape" }, { run_id: 42 }, { run_id: null }],
    });

    // The table is the caller's, and may hold rows from a migration, another
    // tenant or an operator's hand. None of those is an id `get` should be
    // asked for, so `pending` names only what this store could have stored.
    expect(await store.pending()).toEqual(["run-1"]);
  });
});
