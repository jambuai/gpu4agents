# Integrating gpu into an application

How any application, in any language, uses a GPU that `gpu` rents, serves and shuts down. No SDK is needed: the app talks to two things it already knows how to use.

| Plane | What | How the app uses it |
| --- | --- | --- |
| Control | the `gpu` command | runs it as a subprocess with `--json` and reads stdout and the exit code |
| Data | vLLM's OpenAI-compatible HTTP API | any OpenAI SDK, or plain HTTP, with the base URL, key and model `gpu` printed |

The app never has to stop the GPU. The watchdog on the machine shuts it down once nothing has used it for `lifecycle.idle_timeout`, and every inference request counts as use. The app's only duty is to call `gpu setup` again when the machine it was using is gone.

## 1. Prepare the host the app runs on

```bash
pipx install gpu4agents            # or: uv tool install gpu4agents / pip install gpu4agents
cd /srv/my-app                    # the directory the app runs gpu from
gpu init --model Qwen/Qwen2.5-7B-Instruct
echo "VAST_API_KEY=..." >> .env   # or export it, or ~/.gpu/credentials
gpu validate                      # fails here, not in production, on a bad config
gpu sweep --install-timer         # optional safety net, see USAGE.md
```

Requirements:

- `gpu4agents` (or `jambu-gpu` 0.3.0 or later), on Python 3.10+. The app can be in any language.
- The directory holding `gpu.yaml` must be persistent. `gpu` keeps what it rented in `.gpu/state.json` next to it. In a container, mount it as a volume: a container that loses `.gpu/` forgets its instance, rents a second one, and leaves the first to the watchdog and the sweep.
- Before 0.3.0 these directories were `.jambu/` and `~/.jambu`. The first command after an upgrade renames them, so an instance already rented is kept and is not rented again.
- `.gpu/state.json` holds the endpoint key and the watchdog token. It is written `0600`. Add `.gpu/` and `.env` to `.gitignore` and keep them out of images.

Tune the lifecycle in `gpu.yaml` for the app. For a service with bursty traffic, a longer idle timeout avoids paying a restart on every burst:

```yaml
gpu_runtime:
  lifecycle:
    idle_timeout: 20m     # stop after 20 minutes without a request
    max_lifetime: 12h     # hard cap per session, even under traffic
    stopped_ttl: 24h      # destroy a machine that stayed stopped this long
  budget:
    max_hourly_cost_usd: 0.60
```

All fields are in [CONFIG.md](CONFIG.md).

## 2. The contract

### Commands an app calls

Always pass `--json` (before the command name). Run them from the project directory, or pass `--config /path/to/gpu.yaml`.

| Command | Use it to | Blocks | Starts billing |
| --- | --- | --- | --- |
| `gpu --json setup` | get a working endpoint: rents a machine if there is none, restarts it if it was stopped, waits until the model answers | until ready | yes |
| `gpu --json env` | read the endpoint without starting anything | no | no |
| `gpu --json status` | full state, cost so far, and what the guard would do now | no | no |
| `gpu --json stop` | stop it now instead of waiting for the idle timeout; disk and model cache are kept | until stopped | stops it |
| `gpu --json destroy --yes` | delete it; nothing is billed afterwards | until gone | stops it |
| `gpu --json lock --for 2h` | keep it up through a quiet period (a long job, a demo) | no | |
| `gpu --json unlock` | return to the normal rules | no | |

`setup` is idempotent and serialized by a file lock: ten processes calling it at once rent one machine, and the other nine wait and get the same answer.

### What `setup` and `env` print

```json
{
  "instance_id": "27581234",
  "state": "ready",
  "endpoint": "http://203.0.113.7:41022",
  "base_url": "http://203.0.113.7:41022/v1",
  "api_key": "k6V0...43 characters",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "hourly_cost_usd": 0.196
}
```

| Field | Meaning |
| --- | --- |
| `base_url` | Pass as the OpenAI SDK's `baseURL` / `base_url` |
| `api_key` | Pass as the SDK's `apiKey`. Requests without it get `401` |
| `model` | Pass as `model` in every request (`model.served_name` when set, otherwise `model.id`) |
| `state` | `ready` after `setup`. `env` reports whatever the last command saw, without asking the provider |
| `instance_id` | Changes when a new machine is rented |

`setup` also prints a `watchdog` object (URL, port, last contact; never its token). `env` exits `3` with `"state": "none"` when there is no endpoint.

`gpu env` without `--json` prints shell exports instead, for scripts:

```bash
eval "$(gpu env)"     # OPENAI_BASE_URL, OPENAI_API_KEY, JAMBU_GPU_ENDPOINT, JAMBU_MODEL_ID
gpu env --format dotenv > .env.gpu
```

### Output rules

- With `--json`, stdout carries exactly one JSON document and nothing else. Progress lines are suppressed.
- Errors go to stderr as one line, `error: <message>`, with a non-zero exit code.

### Exit codes

| Code | Meaning | What the app should do |
| --- | --- | --- |
| 0 | success | |
| 1 | unexpected error | log stderr, retry later |
| 2 | config error (`gpu.yaml` invalid, or changed since setup) | fix the config; `setup --recreate` replaces a machine built from the old one |
| 3 | no instance / no endpoint (from `env`, `endpoint`) | call `setup` |
| 4 | provider API error | retry with backoff |
| 5 | provider authentication failed | fix `VAST_API_KEY` |
| 6 | no capacity matches the config | retry later, or relax `compute.gpu` / the budget |
| 7 | provisioning failed | retry; the failed machine was already destroyed |
| 8 | provider timed out | retry with backoff |
| 9 | the provider cannot do this | change the config |
| 10 | over `budget.max_hourly_cost_usd` | raise the budget or wait for cheaper offers |
| 11 | the model server did not start (setup failed, or every host tried failed to boot) | read stderr: it carries the server's log tail; the machine was destroyed |
| 12 | another `gpu` command holds this project's lock (usually a `setup` still booting) | wait and retry; `status` and `logs` do not fail on it, they report the last saved state with `"busy": true` |

### Lifecycle the app has to expect

| Event | What the app sees | What it does |
| --- | --- | --- |
| first `setup` | blocks while a machine is rented, the image pulled and the model downloaded: usually 5 to 15 minutes. A bad host (stalled image pull, broken network) is destroyed and replaced automatically, each costing up to `health.pull_stall_timeout` (10 min) when its pull stalls, or `health.boot_timeout` (30 min) when it crawls, at most `compute.host_attempts` (3) times | wait; do not put a short timeout on `setup`, it bounds itself |
| idle for `idle_timeout` | the watchdog stops the machine, or destroys it where a stopped machine still bills storage (Vast.ai, unless `lifecycle.keep_disk: true`); requests fail with connection refused or a timeout | call `setup` again and retry the request |
| `setup` on a stopped machine | it is restarted: no new rent, no model download, under a minute for a small model (41 s measured) | wait, then use the new `base_url`: the provider may assign a new IP and port on restart |
| stopped for `stopped_ttl` | the machine is destroyed to stop the storage bill | the next `setup` rents a new one (and a new key) |
| `max_lifetime` reached | stopped even under traffic | same as idle |

The key is per machine: it survives stops and restarts, and changes only when a new machine is rented. Always take `base_url`, `api_key` and `model` from the latest `setup` output instead of caching them across restarts.

## 3. Patterns

### A short job: wrap it

The job gets the connection as environment variables and its heartbeats keep the machine up while it runs:

```bash
gpu run node dist/batch.js
gpu run python score.py --input data.jsonl
```

`gpu run` sets `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `JAMBU_GPU_ENDPOINT` and `JAMBU_MODEL_ID` ([USAGE.md](USAGE.md#running-a-workload)). Any OpenAI SDK reads the first two by itself.

### A long-running service: ensure, then heal

1. On the first request (or at boot), call `gpu --json setup` and keep the result.
2. Send requests with any OpenAI client, with a client timeout that is not too long (1 to 2 minutes), so a stopped machine fails fast.
3. When a request fails with a connection error (refused, reset, timeout), call `gpu --json setup` again and retry once with the new result. Do not do this for HTTP errors like `400` or `401`: those come from a live server.
4. Share one in-flight `setup` between concurrent requests.

The TypeScript helper below does this.

### A batch that must not be interrupted

`gpu lock --for 3h` before the batch and `gpu unlock` after it. Traffic already counts as activity, so a lock only matters if the batch has quiet stretches longer than `idle_timeout`.

## 4. Recipes

### TypeScript / Node

[`examples/typescript-client/src/gpu.ts`](../examples/typescript-client/src/gpu.ts) is a single file with no dependencies beyond Node 18+. Copy it into the app.

```ts
import OpenAI from "openai";
import { Gpu } from "./gpu.ts";

const gpu = new Gpu({ cwd: "/srv/my-app" });   // where gpu.yaml lives

const reply = await gpu.call((c) =>
  new OpenAI({ baseURL: c.baseURL, apiKey: c.apiKey, maxRetries: 0, timeout: 120_000 })
    .chat.completions.create({ model: c.model, messages: [{ role: "user", content: "hi" }] }),
);
```

`gpu.call` starts the GPU on first use, shares one `setup` between concurrent calls, and restarts the GPU and retries once when the call fails because the machine is gone. The file also exports `ensureGpu`, `currentGpu`, `stopGpu`, `destroyGpu`, `gpuStatus`, `GpuError` (with `exitCode`) and the `EXIT` codes. [`examples/typescript-client/`](../examples/typescript-client/) runs it end to end:

```bash
npm --prefix examples/typescript-client install
node --experimental-strip-types examples/typescript-client/src/index.ts "Say hi"
```

### Python

```python
import json, subprocess
from openai import OpenAI, APIConnectionError

def ensure_gpu(cwd="/srv/my-app"):
    out = subprocess.run(["gpu", "--json", "setup"], cwd=cwd, check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out)

conn = ensure_gpu()
def ask(prompt):
    global conn
    for attempt in (1, 2):
        client = OpenAI(base_url=conn["base_url"], api_key=conn["api_key"], max_retries=0, timeout=120)
        try:
            return client.chat.completions.create(
                model=conn["model"], messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
        except APIConnectionError:
            if attempt == 2:
                raise
            conn = ensure_gpu()   # stopped while idle: restart it, then retry
```

### Python: queue mode

For jobs that arrive over the day, not one machine per conversation, `gpu4agents.queue.api` submits to the queue and follows a job without shelling out. It needs `queue:` in `gpu.yaml`; a coordinator (`gpu queue run`, a service you run) does the renting. See [QUEUE.md](../QUEUE.md).

```python
from gpu4agents.queue import api

job_id = api.submit("train/", key="order-1234")      # idempotent on the key
info = api.wait(job_id, timeout_s=3600)               # a dict; "timed_out": true if still running
if info["state"] == "succeeded":
    path = api.outputs(job_id)                        # verified outputs, gpu-outputs/<job>/
```

`api.status`, `listing`, `cancel`, `retry`, `pause`, `resume`, `drain`, `audit_queue`, `stats` and `summary` match the `gpu queue` commands. Credentials (`GPU_QUEUE_URL`, `GPU_OUTPUTS_URL`, `AWS_*`) come from the environment or `~/.gpu/credentials`. An unreachable queue or store raises; handle it like exit code 4.

### Shell / curl

```bash
eval "$(gpu --json setup >/dev/null && gpu env)"
curl -s "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d "{\"model\": \"$JAMBU_MODEL_ID\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}"
```

### Any other language

Run `gpu --json setup` as a subprocess, parse the JSON, and point the language's OpenAI client (or HTTP client) at `base_url` with `Authorization: Bearer <api_key>`. On a connection error, run it again and retry once.

## 5. Security

- The endpoint requires a key. Unless `runtime.api_key` is set, `setup` generates one per machine (43 random URL-safe characters) and starts vLLM with it. Requests without it get `401`. `/health` and `/metrics` stay open: they reveal no data and the watchdog reads `/metrics`. Instances provisioned before this existed run without a key until they are recreated.
- The endpoint is plain HTTP on a public IP and port. The key and the prompts travel unencrypted. For sensitive data, put the app and the GPU on a private network or a tunnel, or front the endpoint with a TLS proxy.
- The watchdog holds the provider key on the instance so it can stop its own machine. A restricted key is the safer choice; see [LIFECYCLE.md](LIFECYCLE.md).
- `gpu.yaml` never holds credentials. `.env` and `.gpu/` do not belong in version control.

## 6. Troubleshooting

| Symptom | Look at |
| --- | --- |
| `setup` fails with exit 11 | stderr has the model server's last log lines; `gpu logs --source runtime` |
| `setup` slow on the first call | `gpu status` (works during setup) and `gpu logs --source onstart`. On Vast the default image is `vastai/vllm`, which most hosts already cache; a project that pins `compute.image: vllm/vllm-openai:...` pulls 12 GB from Docker Hub |
| `401` from the endpoint | the key changed (a new machine was rented): take it from a fresh `setup` |
| the machine stopped while the app was busy | `gpu events`; check `idle_timeout` against the longest silence between requests, or `gpu lock` |
| unsure what is still billing | `gpu ls` lists every jambu instance in the account |
