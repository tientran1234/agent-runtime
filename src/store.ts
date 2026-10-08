import { randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { SuspendedRun } from "./resume.js";

/**
 * Where a suspended run waits. `runAgent` hands the snapshot back and
 * `resumeAgent` takes it again, and in between it has to live somewhere that
 * outlives the process that made it — a row, an object, a workflow step's
 * output. This is that one shape, so the half of an approval flow that is not
 * the agent can be written against a port rather than against whichever backend
 * happened to be to hand.
 *
 * Every method is async because none of the backends worth having are not.
 */
export interface RunStore {
  /**
   * Store the snapshot under `runId`, replacing one already there: a run that
   * suspends, is answered `{ ask: true }` and suspends again is the same run
   * waiting on a later decision, not a second run.
   */
  put(runId: string, state: SuspendedRun): Promise<void>;
  /**
   * The snapshot, or `undefined` for an id the store does not hold. Absent is
   * an answer rather than an error because polling for a decision that may
   * already have been taken is the normal way to read this.
   */
  get(runId: string): Promise<SuspendedRun | undefined>;
  /**
   * Forget a run. Forgetting one that has already gone is not an error: a
   * resumed run may well be cleaned up twice, and the second attempt wants the
   * same outcome as the first.
   */
  delete(runId: string): Promise<void>;
  /**
   * The ids still waiting, in no meaningful order — for the queue, inbox or
   * dashboard that has to show someone what is waiting on them.
   */
  pending(): Promise<string[]>;
}

/** What `FileStore` names its files, and therefore what `pending` reads ids back out of. */
const SUFFIX = ".json";

/**
 * A run id is a name, not a path. The ids come from whatever system owns the
 * approval — a ticket number, a workflow id, a customer reference — and
 * `FileStore` turns them into filenames, where a `..` or a slash would read and
 * write outside the directory it was given. Leading dots go too, so that no id
 * can collide with a dotfile or with the temporary names `put` writes under.
 *
 * Checked by every store rather than only the one that builds paths: an id that
 * works against `MemoryStore` in a test has to work against the disk in a
 * deployment, and a fake that is laxer than the real thing is how that stops
 * being true.
 */
const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/;

function assertRunId(runId: string): void {
  if (RUN_ID.test(runId)) return;
  throw new Error(
    `run store: ${JSON.stringify(runId)} is not a usable run id — letters, digits, dot, dash and underscore, ` +
      `starting with a letter or digit, up to 200 characters`,
  );
}

/**
 * Serialize before anything is stored. Both implementations here go through
 * JSON rather than holding the object, which is what makes the in-memory one a
 * faithful stand-in: a snapshot that would not have survived a row fails in the
 * tests that use the fake instead of in the deployment that uses the row.
 */
function serialize(runId: string, state: SuspendedRun): string {
  let json: string | undefined;
  try {
    json = JSON.stringify(state);
  } catch (err) {
    throw new Error(`run store: the snapshot for ${runId} is not JSON — ${err instanceof Error ? err.message : String(err)}`);
  }
  // `JSON.stringify` answers `undefined` rather than throwing for a value it
  // cannot represent at all, which would otherwise be stored as the string
  // "undefined" and read back as a corrupt run.
  if (json === undefined) throw new Error(`run store: the snapshot for ${runId} is not JSON — ${String(state)} has no representation`);
  return json;
}

/**
 * Read a snapshot back, checking the one field that says whether this reader
 * understands it. `version` is in the record for a reader that may be older
 * than the writer, and here is where it has to be acted on: passed through
 * unchecked, a shape from a later release reaches the loop as a
 * half-understood object and fails somewhere that says nothing about why.
 */
function parse(runId: string, json: string): SuspendedRun {
  let value: unknown;
  try {
    value = JSON.parse(json);
  } catch (err) {
    throw new Error(`run store: the snapshot for ${runId} did not parse — ${err instanceof Error ? err.message : String(err)}`);
  }
  if (typeof value !== "object" || value === null) {
    throw new Error(`run store: the snapshot for ${runId} is ${value === null ? "null" : typeof value}, not a suspended run`);
  }
  const version = (value as { version?: unknown }).version;
  if (version !== 1) {
    throw new Error(`run store: the snapshot for ${runId} is version ${String(version)}, and this reader understands 1`);
  }
  return value as SuspendedRun;
}

/**
 * Holds every waiting run in a map. For tests, and for a single process whose
 * approvals do not outlive it.
 */
export class MemoryStore implements RunStore {
  /** Kept as JSON, not as the object: see `serialize`, and the copy `get` owes its caller. */
  private readonly rows = new Map<string, string>();

  async put(runId: string, state: SuspendedRun): Promise<void> {
    assertRunId(runId);
    this.rows.set(runId, serialize(runId, state));
  }

  async get(runId: string): Promise<SuspendedRun | undefined> {
    assertRunId(runId);
    const json = this.rows.get(runId);
    // Parsed per read, so every caller gets its own value. Handing back one
    // shared object would let a caller that edits what it read edit what is
    // stored — a bug that cannot happen against a backend that returns bytes,
    // which is exactly the class of bug a fake must not hide.
    return json === undefined ? undefined : parse(runId, json);
  }

  async delete(runId: string): Promise<void> {
    assertRunId(runId);
    this.rows.delete(runId);
  }

  async pending(): Promise<string[]> {
    return [...this.rows.keys()];
  }
}

/**
 * One JSON file per waiting run, under a directory you choose. Enough for a
 * single host with somewhere to write, and a worked example of what the port
 * asks of a real backend.
 */
export class FileStore implements RunStore {
  constructor(private readonly dir: string) {}

  /**
   * Written to a temporary name in the same directory and renamed over the
   * target, because a rename within a directory is atomic and a write is not. A
   * process killed halfway through a write leaves a truncated file, and for a
   * suspended run that is fatal rather than inconvenient: the snapshot is the
   * only copy of a run whose tools have not run, so there is nothing to rebuild
   * it from. Serializing first is the same promise one step earlier — a
   * snapshot that is not JSON must not already have replaced the one that was.
   */
  async put(runId: string, state: SuspendedRun): Promise<void> {
    assertRunId(runId);
    const json = serialize(runId, state);
    await mkdir(this.dir, { recursive: true });
    // Same directory, so the rename cannot cross a filesystem and fall back to
    // a copy; a random suffix, so two writers for one run do not share a
    // half-written file.
    const temp = join(this.dir, `.${runId}.${randomUUID()}.tmp`);
    try {
      await writeFile(temp, json, "utf8");
      await rename(temp, this.path(runId));
    } catch (err) {
      // Swept up rather than left to accumulate: a store that collects a file
      // per failed write eventually fills the disk the next snapshot needs.
      await rm(temp, { force: true });
      throw err;
    }
  }

  async get(runId: string): Promise<SuspendedRun | undefined> {
    assertRunId(runId);
    let json: string;
    try {
      json = await readFile(this.path(runId), "utf8");
    } catch (err) {
      // No file is no run. Anything else — a permission, a dead mount — is a
      // store that is not working, and must not read as a run that has gone.
      if (isNotFound(err)) return undefined;
      throw err;
    }
    return parse(runId, json);
  }

  async delete(runId: string): Promise<void> {
    assertRunId(runId);
    await rm(this.path(runId), { force: true });
  }

  async pending(): Promise<string[]> {
    let names: string[];
    try {
      names = await readdir(this.dir);
    } catch (err) {
      // A directory that nothing has suspended into yet is an empty store, not
      // a broken one: `put` is what creates it.
      if (isNotFound(err)) return [];
      throw err;
    }
    // Only names this store would have written itself. A directory is shared
    // with whatever else lands in it — an operator's notes, an editor's swap
    // file, a temporary name from a write that died — and none of those is a
    // run id that `get` should then be asked for.
    return names
      .filter((name) => name.endsWith(SUFFIX))
      .map((name) => name.slice(0, -SUFFIX.length))
      .filter((id) => RUN_ID.test(id));
  }

  private path(runId: string): string {
    return join(this.dir, `${runId}${SUFFIX}`);
  }
}

function isNotFound(err: unknown): boolean {
  return (err as { code?: unknown } | null)?.code === "ENOENT";
}

/**
 * The one thing a store needs from a database: run this statement with these
 * parameters, hand back the rows. A driver is not asked for and no pool is
 * opened here — the caller writes the line that adapts theirs
 * (`(sql, params) => pool.query(sql, params).then((r) => r.rows)` for
 * node-postgres), which is what keeps a library that holds an agent loop from
 * also holding an opinion about connections, retries and migrations.
 */
export type SqlQuery = (sql: string, params: readonly string[]) => Promise<readonly SqlRow[]>;

/** A row as a driver hands it over: the column this store asked for, and whatever else came with it. */
export type SqlRow = Record<string, unknown>;

/**
 * Which placeholder and which upsert the statements are written with. The four
 * statements are otherwise identical across these three, and those two
 * differences are the whole reason this is an option rather than one fixed
 * string of SQL.
 */
export type SqlDialect = "postgres" | "sqlite" | "mysql";

export interface SqlStoreOptions {
  /** How a statement reaches the database. */
  query: SqlQuery;
  /**
   * The table, `agent_runs` by default, with two columns: `run_id` as the
   * primary key and `snapshot` as text. Which database, schema and migration
   * it arrives by stays the caller's — this store only reads and writes it.
   */
  table?: string;
  /** Default `postgres`. */
  dialect?: SqlDialect;
}

/** The columns a store's table has. Named once because the statements and the errors both have to say them. */
const ID_COLUMN = "run_id";
const SNAPSHOT_COLUMN = "snapshot";

/**
 * A table name goes into the statement text, because no driver takes an
 * identifier as a parameter — so it is checked against what an unquoted
 * identifier may be, optionally behind one schema qualifier. The check is in
 * the constructor rather than in `put`: a store built with a name that is
 * really a fragment of SQL must fail where it was built, not on the first run
 * that suspends into it.
 */
const TABLE = /^[A-Za-z_][A-Za-z0-9_]{0,62}(\.[A-Za-z_][A-Za-z0-9_]{0,62})?$/;

/**
 * A row per waiting run, in a table you already have. The snapshot is JSON
 * text and the id is the primary key, which is the whole schema: a suspended
 * run is read back by id and listed, and nothing here wants to query inside it.
 */
export class SqlStore implements RunStore {
  private readonly query: SqlQuery;
  /**
   * Built once, in the constructor: the table and the dialect cannot change
   * after that, and a statement assembled per call would be a place for a name
   * to reach the database unchecked.
   */
  private readonly statements: { put: string; get: string; delete: string; pending: string };

  constructor(options: SqlStoreOptions) {
    const table = options.table ?? "agent_runs";
    if (!TABLE.test(table)) {
      throw new Error(
        `run store: ${JSON.stringify(table)} is not a usable table name — letters, digits and underscore, ` +
          `starting with a letter or underscore, optionally behind one schema qualifier`,
      );
    }
    const dialect = options.dialect ?? "postgres";
    this.query = options.query;
    // Postgres numbers its placeholders; the other two do not. Nothing else
    // about these statements is dialect-specific except the upsert's tail.
    const slot = (n: number) => (dialect === "postgres" ? `$${n}` : "?");
    const onConflict =
      dialect === "mysql"
        ? `ON DUPLICATE KEY UPDATE ${SNAPSHOT_COLUMN} = VALUES(${SNAPSHOT_COLUMN})`
        : `ON CONFLICT (${ID_COLUMN}) DO UPDATE SET ${SNAPSHOT_COLUMN} = excluded.${SNAPSHOT_COLUMN}`;
    this.statements = {
      // An upsert rather than a read and then a write: a run that suspends, is
      // answered `{ ask: true }` and suspends again is the same run waiting on a
      // later decision, and two processes putting it must not be able to
      // interleave into a lost snapshot.
      put: `INSERT INTO ${table} (${ID_COLUMN}, ${SNAPSHOT_COLUMN}) VALUES (${slot(1)}, ${slot(2)}) ${onConflict}`,
      get: `SELECT ${SNAPSHOT_COLUMN} FROM ${table} WHERE ${ID_COLUMN} = ${slot(1)}`,
      delete: `DELETE FROM ${table} WHERE ${ID_COLUMN} = ${slot(1)}`,
      pending: `SELECT ${ID_COLUMN} FROM ${table}`,
    };
  }

  async put(runId: string, state: SuspendedRun): Promise<void> {
    assertRunId(runId);
    // Serialized before the statement goes out, so a snapshot JSON cannot carry
    // has not already replaced the stored one that it could.
    const json = serialize(runId, state);
    await this.query(this.statements.put, [runId, json]);
  }

  async get(runId: string): Promise<SuspendedRun | undefined> {
    assertRunId(runId);
    const rows = await this.query(this.statements.get, [runId]);
    const row = rows[0];
    // No row is no run, the same answer the other two give: polling for a
    // decision that may already have been taken is how this is normally read.
    if (row === undefined) return undefined;
    const json = row[SNAPSHOT_COLUMN];
    if (typeof json !== "string") {
      // A `json` or `jsonb` column hands back a value already parsed, which has
      // never been through the version check every read owes a stored snapshot.
      // Said once here, rather than let a cast carry a column type nobody meant
      // into the loop.
      throw new Error(
        `run store: the ${SNAPSHOT_COLUMN} column for ${runId} came back as ` +
          `${json === null ? "null" : typeof json}, and this store stores JSON text — declare the column as text`,
      );
    }
    return parse(runId, json);
  }

  async delete(runId: string): Promise<void> {
    assertRunId(runId);
    // A statement that matches nothing is not an error: a resumed run may well
    // be cleaned up twice, and the second attempt wants the first's outcome.
    await this.query(this.statements.delete, [runId]);
  }

  async pending(): Promise<string[]> {
    const rows = await this.query(this.statements.pending, []);
    // Only ids this store could have written, as `FileStore` filters the
    // directory it was given. The table is the caller's and may hold rows from
    // a migration, another writer or an operator's hand, and none of those is
    // an id to hand back for `get` to be asked for.
    return rows
      .map((row) => row[ID_COLUMN])
      .filter((id): id is string => typeof id === "string" && RUN_ID.test(id));
  }
}
