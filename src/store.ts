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

/**
 * Holds every waiting run in a map. For tests, and for a single process whose
 * approvals do not outlive it.
 */
export class MemoryStore implements RunStore {
  async put(_runId: string, _state: SuspendedRun): Promise<void> {
    throw new Error("MemoryStore: not implemented");
  }
  async get(_runId: string): Promise<SuspendedRun | undefined> {
    throw new Error("MemoryStore: not implemented");
  }
  async delete(_runId: string): Promise<void> {
    throw new Error("MemoryStore: not implemented");
  }
  async pending(): Promise<string[]> {
    throw new Error("MemoryStore: not implemented");
  }
}

/**
 * One JSON file per waiting run, under a directory you choose. Enough for a
 * single host with somewhere to write, and a worked example of what the port
 * asks of a real backend.
 */
export class FileStore implements RunStore {
  constructor(private readonly dir: string) {}
  async put(_runId: string, _state: SuspendedRun): Promise<void> {
    throw new Error("FileStore: not implemented");
  }
  async get(_runId: string): Promise<SuspendedRun | undefined> {
    throw new Error("FileStore: not implemented");
  }
  async delete(_runId: string): Promise<void> {
    throw new Error("FileStore: not implemented");
  }
  async pending(): Promise<string[]> {
    throw new Error("FileStore: not implemented");
  }
}
