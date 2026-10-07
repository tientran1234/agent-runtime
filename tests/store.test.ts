import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { z } from "zod";
import {
  FakeProvider,
  FileStore,
  MemoryStore,
  callTools,
  defineTool,
  reply,
  resumeAgent,
  runAgent,
  type RunStore,
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

describe.each([
  { name: "MemoryStore", open: (_dir: string): RunStore => new MemoryStore() },
  { name: "FileStore", open: (dir: string): RunStore => new FileStore(join(dir, "runs")) },
])("$name as a RunStore", ({ open }) => {
  let dir: string;
  let store: RunStore;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "agent-runtime-store-"));
    store = open(dir);
  });
  afterEach(() => rm(dir, { recursive: true, force: true }));

  it("hands back a snapshot a resumed run finishes from", async () => {
    const ran: string[] = [];
    await store.put("run-42", await suspend(ran));

    const result = await resumeAgent({
      provider: new FakeProvider([reply("Refunded ord_42.")]),
      tools: [refund(ran)],
      state: (await store.get("run-42"))!,
      decisions: { r1: { allow: true } },
    });

    expect(result).toMatchObject({ status: "completed", text: "Refunded ord_42." });
    expect(ran).toEqual(["ord_42"]);
  });

  it("reports an id it does not hold as undefined rather than throwing", async () => {
    expect(await store.get("never-stored")).toBeUndefined();
  });

  it("replaces a snapshot, because a run that suspends twice is one run", async () => {
    const state = await suspend();
    await store.put("run-42", state);
    await store.put("run-42", { ...state, iterations: 4 });

    expect((await store.get("run-42"))?.iterations).toBe(4);
    expect(await store.pending()).toEqual(["run-42"]);
  });

  it("forgets a run, and forgetting one twice is not an error", async () => {
    await store.put("run-42", await suspend());
    await store.delete("run-42");

    expect(await store.get("run-42")).toBeUndefined();
    await expect(store.delete("run-42")).resolves.toBeUndefined();
    expect(await store.pending()).toEqual([]);
  });

  it("names the runs still waiting", async () => {
    const state = await suspend();
    await store.put("run-1", state);
    await store.put("run-2", state);

    expect((await store.pending()).sort()).toEqual(["run-1", "run-2"]);
  });

  it("refuses a run id that is a path rather than a name", async () => {
    const state = await suspend();
    for (const bad of ["../escape", "runs/42", ".", "..", "", ".hidden", "a".repeat(201)]) {
      await expect(store.put(bad, state)).rejects.toThrow(/not a usable run id/);
      await expect(store.get(bad)).rejects.toThrow(/not a usable run id/);
      await expect(store.delete(bad)).rejects.toThrow(/not a usable run id/);
    }
    expect(await store.pending()).toEqual([]);
  });

  it("refuses at put a snapshot no real backend could have stored", async () => {
    await expect(store.put("run-42", unserializable(await suspend()))).rejects.toThrow(/is not JSON/);
  });

  it("refuses a snapshot from a writer newer than this reader", async () => {
    const state = await suspend();
    await store.put("run-42", { ...state, version: 2 } as unknown as SuspendedRun);

    await expect(store.get("run-42")).rejects.toThrow(/run-42.*version 2/);
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
