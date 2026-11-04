// The whole integration with gpu4agents, for any Node/TypeScript app.
// Copy this file into your project: it needs nothing but Node 18+ and the
// `gpu` command on PATH (pipx install gpu4agents / uv tool install gpu4agents).
//
// `gpu --json setup` is the one call that matters. It rents a GPU when there
// is none, restarts it when the watchdog stopped it for being idle, waits until
// the model answers, and prints how to reach it. It is idempotent and safe to
// call from several processes at once (they wait on a file lock).

import { execFile } from "node:child_process";

export interface GpuConnection {
  instanceId: string;
  state: string;
  /** OpenAI-compatible base URL, e.g. http://1.2.3.4:40123/v1 */
  baseURL: string;
  apiKey: string;
  /** The value to pass as `model` in every request. */
  model: string;
  hourlyCostUsd: number | null;
}

export interface GpuOptions {
  /** Directory holding gpu.yaml (and .env with VAST_API_KEY). */
  cwd?: string;
  /** Explicit path to gpu.yaml. */
  config?: string;
  /** The gpu executable. Default: "gpu". */
  bin?: string;
  /**
   * Kill gpu after this long. Default: no limit, because gpu bounds itself:
   * a fresh machine usually serves in 5-15 min, but each bad host it replaces
   * can cost up to health.boot_timeout (30 min), compute.host_attempts times.
   */
  timeoutMs?: number;
}

/** Exit codes of the gpu CLI (docs/INTEGRATION.md#exit-codes). */
export const EXIT = {
  ERROR: 1,
  CONFIG: 2,
  NO_INSTANCE: 3,
  PROVIDER: 4,
  AUTH: 5,
  NO_CAPACITY: 6,
  PROVISIONING: 7,
  PROVIDER_TIMEOUT: 8,
  UNSUPPORTED: 9,
  BUDGET: 10,
  STARTUP: 11,
} as const;

export class GpuError extends Error {
  readonly exitCode: number | null;
  readonly stderr: string;

  constructor(message: string, exitCode: number | null, stderr: string) {
    super(message);
    this.name = "GpuError";
    this.exitCode = exitCode;
    this.stderr = stderr;
  }
}

function gpu(args: string[], opts: GpuOptions): Promise<string> {
  const argv = ["--json", ...(opts.config ? ["--config", opts.config] : []), ...args];
  return new Promise((resolve, reject) => {
    execFile(
      opts.bin ?? "gpu",
      argv,
      { cwd: opts.cwd, timeout: opts.timeoutMs ?? 0, maxBuffer: 16 * 1024 * 1024 },
      (err, stdout, stderr) => {
        if (err) {
          const code = typeof err.code === "number" ? err.code : null;
          const detail = stderr.trim().split("\n").pop() || err.message;
          reject(new GpuError(`gpu ${args.join(" ")} failed: ${detail}`, code, stderr));
          return;
        }
        resolve(stdout);
      },
    );
  });
}

function toConnection(raw: Record<string, any>): GpuConnection {
  if (!raw.base_url) {
    throw new GpuError("gpu returned no endpoint", EXIT.NO_INSTANCE, "");
  }
  return {
    instanceId: raw.instance_id,
    state: raw.state,
    baseURL: raw.base_url,
    apiKey: raw.api_key,
    model: raw.model,
    hourlyCostUsd: raw.hourly_cost_usd ?? null,
  };
}

/** Make sure the GPU is up and serving; returns how to reach it. Blocks until ready. */
export async function ensureGpu(opts: GpuOptions = {}): Promise<GpuConnection> {
  return toConnection(JSON.parse(await gpu(["setup"], opts)));
}

/** How to reach the GPU right now, without starting anything. Null when there is none. */
export async function currentGpu(opts: GpuOptions = {}): Promise<GpuConnection | null> {
  try {
    return toConnection(JSON.parse(await gpu(["env"], opts)));
  } catch (err) {
    if (err instanceof GpuError && err.exitCode === EXIT.NO_INSTANCE) return null;
    throw err;
  }
}

/** Stop it now instead of waiting for the idle timeout. Its disk (and model cache) is kept. */
export async function stopGpu(opts: GpuOptions = {}): Promise<void> {
  await gpu(["stop"], opts);
}

/** Delete it: nothing is billed afterwards, the next ensureGpu() rents a new machine. */
export async function destroyGpu(opts: GpuOptions = {}): Promise<void> {
  await gpu(["destroy", "--yes"], opts);
}

/** The full status as reported by `gpu --json status`. */
export async function gpuStatus(opts: GpuOptions = {}): Promise<Record<string, any>> {
  return JSON.parse(await gpu(["status"], opts));
}

/** True for errors that mean "the machine is not there": stopped while idle, or replaced. */
export function isUnreachable(err: unknown): boolean {
  const e = err as { code?: string; cause?: { code?: string } } | null;
  const code = e?.code ?? e?.cause?.code;
  if (code && ["ECONNREFUSED", "ECONNRESET", "ETIMEDOUT", "EHOSTUNREACH", "UND_ERR_CONNECT_TIMEOUT"].includes(code)) {
    return true;
  }
  // The OpenAI SDK wraps every network failure in APIConnectionError (and
  // APIConnectionTimeoutError), without setting `name`.
  const kind = (err as object | null)?.constructor?.name ?? "";
  return kind === "APIConnectionError" || kind === "APIConnectionTimeoutError";
}

/**
 * A connection that heals itself: when a call fails because the watchdog
 * stopped the idle GPU, run `gpu setup` again (restarting it) and retry once.
 *
 *   const gpu = new Gpu({ cwd: "/srv/my-app" });
 *   const answer = await gpu.call((c) => openaiFor(c).chat.completions.create(...));
 */
export class Gpu {
  private conn: Promise<GpuConnection> | null = null;
  private readonly opts: GpuOptions;

  constructor(opts: GpuOptions = {}) {
    this.opts = opts;
  }

  /** The current connection, starting the GPU when needed. Concurrent callers share one setup. */
  connection(): Promise<GpuConnection> {
    if (!this.conn) {
      this.conn = ensureGpu(this.opts).catch((err) => {
        this.conn = null;
        throw err;
      });
    }
    return this.conn;
  }

  async call<T>(fn: (conn: GpuConnection) => Promise<T>): Promise<T> {
    const conn = await this.connection();
    try {
      return await fn(conn);
    } catch (err) {
      if (!isUnreachable(err)) throw err;
      this.conn = null;
      return fn(await this.connection());
    }
  }
}
