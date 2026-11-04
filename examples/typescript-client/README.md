# TypeScript client

Any Node/TypeScript application using a GPU that `gpu` rents and guards, through the official [OpenAI SDK](https://www.npmjs.com/package/openai).

[`src/gpu.ts`](src/gpu.ts) is the whole integration: one file, no dependencies beyond Node 18+, that runs `gpu --json setup` and heals the connection when the watchdog stopped the idle GPU. Copy it into your app. The contract it relies on is [docs/INTEGRATION.md](../../docs/INTEGRATION.md).

## Run it

From the directory that holds `gpu.yaml` and `VAST_API_KEY` (in `.env` or the environment), with `gpu4agents` (or `jambu-gpu` 0.3.0 or later) on `PATH`:

```bash
npm --prefix examples/typescript-client install
node --experimental-strip-types examples/typescript-client/src/index.ts "Say hi"
```

The first call rents a machine and waits for the model, which takes a few minutes. Later calls reuse it. There is nothing to clean up: the machine stops itself after `lifecycle.idle_timeout` without requests.

## Use it in your app

```ts
import OpenAI from "openai";
import { Gpu } from "./gpu.ts";

const gpu = new Gpu({ cwd: "/srv/my-app" });

export async function complete(prompt: string): Promise<string> {
  const reply = await gpu.call((c) =>
    new OpenAI({ baseURL: c.baseURL, apiKey: c.apiKey, maxRetries: 0, timeout: 120_000 })
      .chat.completions.create({ model: c.model, messages: [{ role: "user", content: prompt }] }),
  );
  return reply.choices[0]?.message?.content ?? "";
}
```

| Export | What it does |
| --- | --- |
| `new Gpu(opts).call(fn)` | Runs `fn` with a live connection; starts the GPU on first use and restarts it and retries once when it is gone |
| `ensureGpu(opts)` | `gpu --json setup`: up and serving, then `{ baseURL, apiKey, model, instanceId, ... }` |
| `currentGpu(opts)` | The connection without starting anything, or `null` |
| `stopGpu(opts)` / `destroyGpu(opts)` | Stop now (disk kept) / delete |
| `gpuStatus(opts)` | `gpu --json status` |
| `GpuError`, `EXIT` | Failures carry the CLI's `exitCode` ([codes](../../docs/INTEGRATION.md#exit-codes)) |

`opts` is `{ cwd?, config?, bin?, timeoutMs? }`: where `gpu.yaml` is, or its path, the `gpu` executable, and a limit on how long `setup` may take (none by default: `gpu` bounds itself, see [the lifecycle table](../../docs/INTEGRATION.md#lifecycle-the-app-has-to-expect)).
