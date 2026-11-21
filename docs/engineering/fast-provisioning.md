# Fast provisioning: where `gpu setup` spends its time, and how to cut it

Our last 14 successful setups on Vast.ai took a median of 9.3 minutes to serve a 27B model under vLLM, and 17 minutes to serve a 26.6 GB GGUF under llama.cpp. Nearly all of that time is the model server fetching weights from Hugging Face. The image pull, the boot and the watchdog take about 30 seconds on a host that already has the image. We built gpu4agents, and this note is our plan to make setup fast. We will write our own parallel weight fetch and let an idle machine keep the last model's disk for a few hours. Cog, SkyPilot and a warm pool each solve a different problem, and two of them would break the one-command, self-shutdown design. The plan needs no new dependency and one paid benchmark run. It does not make a cold 27B model instant: the floor is the time to move 55.6 GB onto the machine.

Status: written as a design on 2026-11-21, before the benchmark. The section below records what the benchmark changed; the rest of the note is kept as written. Measured against `main` at d685034. Nothing was rented to write it. The offer prices were read from Vast's search API on 2026-11-21.

## Update after the benchmark (2026-11-21)

We rented two Vast machines for the benchmark and paid about $1.20 in all. On each we downloaded the 27.5 GB GGUF (with its `mmproj`) and the 55.6 GB of safetensors in three ways: with our fetcher, over one connection (what llama.cpp's `-hf` does), and with `huggingface_hub` 2.2.0 plus `hf_xet` 1.7.0, the download vLLM runs. The script is in [benchmarks/fetch-rate/](../../benchmarks/fetch-rate/) and the raw results are in [results-2026-11-21.json](../../benchmarks/fetch-rate/results-2026-11-21.json).

| Download | A100 host, 864 Mbps advertised | Quadro RTX 8000 host, 7988 Mbps advertised |
| --- | --- | --- |
| GGUF, one connection (llama.cpp today) | 67 MB/s | 119 MB/s |
| GGUF, fetcher, 32 connections | 117 MB/s | **446 MB/s** (27.5 GB in 62 s) |
| GGUF, fetcher, 64 connections | – | 366 MB/s |
| Safetensors, fetcher, 32 connections | 100 MB/s | 439 MB/s |
| Safetensors, fetcher, 64 connections | – | 373 MB/s |
| Safetensors, `huggingface_hub` + `hf_xet` (vLLM today) | 112 MB/s | **1065 MB/s** (55.6 GB in 52 s) |

Three things changed in the plan:

1. **We were wrong about vLLM's download.** The design assumed vLLM's own download was the slow part. It is not. `hf_xet` filled the 864 Mbps link and reached 1065 MB/s on the fast host, more than twice our fetcher. The 494 s it took on the slow host matches the 496 s median of the lab's real setups. So for vLLM the time was the host's link, and the fetcher would only make it slower. `runtime.fetch: auto` therefore leaves vLLM to `huggingface_hub`, and only `fetch: watchdog` forces the fetcher.
2. **llama.cpp is where the fetcher pays.** llama.cpp downloads over a single connection, at 67 and 119 MB/s, and the fetcher was 1.75 and 3.7 times faster. `auto` uses it for llama. We also found that the lab's own GGUF repo ships a vision projector (`mmproj`). The first version of the fetcher refused such repos, so it would have skipped the very model it was written for. It now fetches the single `mmproj` and passes `--mmproj`.
3. **The host's link is the biggest lever, so the bandwidth floor goes up.** The same 55.6 GB took 494 s on one host and 52 s on the other. The floor now aims at 90 s of download instead of 180 s, capped at 5000 Mbps. That cap is 4942 Mbps for the 55.6 GB model and 2444 Mbps for the 27.5 GB GGUF with its projector. On 2026-11-21, raising the floor from 500 to 5000 Mbps left the cheapest 80 GB offer at the same price.

The ~300 MB/s assumption held on the fast host: 439–446 MB/s for the fetcher and 1065 MB/s for `hf_xet`. On a host with a 1 Gbps link no downloader passes about 110 MB/s. Rows (b) in section 2 now depend on renting a fast host, and the floor exists to do that.

The benchmark also found a pricing bug. Vast priced every offer with its default few GB of disk, because our search never sent `allocated_storage`. As a result `gpu offers`, the budget check and the pinned price cap all left out the disk. An A100 quoted at $1.078/h billed $1.275/h. The fix is a separate PR, and the second rental, made with it, billed exactly what it was quoted.

32 connections beat 16 and 64 on every run, so the fetcher uses 32.

## 1. Where the time goes today

### The data

The data is the event logs of the two lab projects that rented GPUs this month, `cybersecurity/qwensploit-27b` and `cybersecurity/cyber-tiel-35b` (`.gpu/logs/events.jsonl`). They cover 33 setup requests between 2026-11-13 and 2026-11-18, all on Vast:

- 24 requests rented a machine. The other 9 stopped before renting, most of them because the pinned offer was gone.
- 14 of the 24 machines reached `runtime_ready`.

The stages come from events the code already writes:

| Stage | From → to | Emitted at |
| --- | --- | --- |
| Rent | `provision_requested` → `instance_created` | `core/engine.py` (CLI clock) |
| Image pull + boot + watchdog start | `instance_created` → `watchdog_started` | watchdog on the machine (machine clock) |
| Watchdog contact | `watchdog_started` → `runtime_starting` | CLI, `Engine._wait_runtime` |
| Weights + server start | `runtime_starting` → `runtime_ready` | CLI; also logged as `startup_seconds` (`core/engine.py` L1256-L1260) |

The models were:

- **Qwensploit-V2-27B** on vLLM. Its safetensors are 55.6 GB, by the Hugging Face tree API.
- **Cyber-Tiel-Coder-35B-A3B** on llama.cpp. Its `UD-Q5_K_XL` GGUF is 26.59 GB.

### Measured stages (successful setups only)

| Stage | vLLM 27B, 9 setups: median (range) | llama 26.6 GB, 5 setups: median (range) |
| --- | --- | --- |
| Rent (API call) | 1 s (0–2) | 1 s (1–2) |
| Image pull + boot + watchdog | 29 s (19–364) | 39 s (22–100) |
| Watchdog contact | 3 s (1–6) | 4 s (3–6) |
| Weights + server start | **496 s (316–740)** | **976 s (397–1341)** |
| **Total** | **557 s (433–1107)** | **1021 s (441–1369)** |

**Image pull.** The image is not the problem. Since 0.10 the Vast default for vLLM is `vastai/vllm:v0.32.0-cuda-13.0`, which Vast's hosts cache (`providers/vast/provider.py` L145-L148). Seven of the nine vLLM boots took 19–31 s. The two slow ones, 237 s and 364 s, were hosts that pulled the image from scratch. The llama image (`ghcr.io/ggml-org/llama.cpp:server-cuda`, pinned in `runtime/llama.py` L41-L44) is not cached by Vast. Its first boot on a host took 100–113 s. Later boots on host 15167, where the image was already cached, took 22–40 s.

**Watchdog install.** This is not a time cost either. The boot script carries the whole watchdog, compressed. It is bound by Vast's 16 KB `onstart` limit (`providers/vast/config.py` L28), and a test holds it 400 bytes under that limit (`tests/test_vast_provider.py` L315). The limit costs space, not time: from boot to a live watchdog takes a few seconds once the container runs. It does matter for this design, because there is no room to add a downloader to the boot script (section 4).

**Weights and server start.** This stage is 89% of the vLLM median and 96% of the llama median. The logs cannot split it into download and load, because the CLI only sees the health check turn green. An upper bound on the download rate is the model size over the whole stage:

- **vLLM:** 55.6 GB in 316–740 s is at least 75–176 MB/s, median 112 MB/s. Load and CUDA-graph capture are inside that time, so the real download rate is higher.
- **llama.cpp:** 26.6 GB in 397–1341 s is at least 20–67 MB/s, median 27 MB/s. GGUF loads in seconds, so this is close to the download rate itself.

The Vast default floor is 500 Mbps, which is 62 MB/s (`providers/vast/config.py` L34). The search we ran on 2026-11-21 shows a median advertised downlink of 2.9–4.9 Gbps on the offers that pass it. llama.cpp's `-hf` fetch (`runtime/llama.py` L88-L95) ran below even the floor on the same host that served vLLM faster. So the limit is the downloader, not the host's link.

**Readiness detection.** This adds latency we built ourselves. `health.interval` defaults to 60 s (`core/config.py` L277) and `wait_for_healthy` clamps it to 30 s (`core/health.py` L43). On average the CLI learns the server is up about 15 s after it is, and up to 30 s after. That lag is inside the numbers above.

**Failures.** The table above counts only successes, and failures cost more than any stage:

- Of the 24 machines rented, 10 never served.
- 4 hit a timeout: three model servers never answered within the 600 s or 1200 s budget, and one image pull stalled.
- 3 had the model server exit.
- 1 was stopped mid-setup.
- 2 have no outcome in the log.

A health timeout spends 10–20 minutes before the next host starts. We cannot tell from these logs how many of those timeouts were a slow download rather than a broken server. A fetch that reports its own progress would tell us (section 4).

**Warm restart.** This is the one fast path we have measured. A stopped machine that kept its disk was restarted by `gpu setup` and answered again 41 s later (`docs/ACCEPTANCE.md` L69). That was Qwen2.5-0.5B on an RTX 3090, so the load was trivial. A 27B model adds its load from local disk.

## 2. Target

The floor for a model nobody has fetched yet is boot plus bytes over bandwidth plus load. The targets below assume a fetch that sustains 300 MB/s. That is 2.4 Gbps, under the median advertised downlink of the offers we already rent, but it is an assumption until the benchmark run in section 5 confirms it.

| Case | Today, median | Target | How |
| --- | --- | --- | --- |
| (a) Known model, its machine parked (stopped with the disk) | not used on Vast by default (idle destroys) | **≤ 90 s** for a model up to ~30 GB, about 2 min for the 27B | restart (41 s measured) + load from local disk + 2 s readiness |
| (a') Known model, a machine already serving it | n/a | **< 5 s** | only with a warm pool, which we do not recommend now |
| (b) Cold 27B vLLM, 55.6 GB | 557 s | **≈ 300 s** | 30 s boot + 185 s fetch + ~75 s load and graph capture + 2 s |
| (b) Cold 26.6 GB GGUF | 1021 s | **≈ 150 s** | 40 s boot + 89 s fetch + ~15 s load + 2 s |
| (b) Cold 8B model, ~16 GB | not measured | **≈ 2 min** | same formula |

We give the vLLM load time as an estimate, not a measurement. vLLM's own log separates load, torch.compile and graph capture. The benchmark run records them.

## 3. Options

### Cog / Replicate-style prebuilt images

Cog packages a model and its server into a container image. The usual pattern bakes the weights into the image or fetches them in `setup()`.

- **What it buys.** A model the lab serves often would boot without a separate fetch.
- **What it costs.**
  - Our 12 GB Docker Hub vLLM image already stalled for 30 minutes on three Vast hosts in a row (`docs/ACCEPTANCE.md` L89). A 27B image with its weights inside is about 70 GB pulled through the same registry path, which is slower and less reliable than Hugging Face's CDN.
  - We would build, store and refresh an image for every model.
  - An arbitrary HF model, the case `gpu` exists for, gains nothing.
  - Replicate's hosted service is its own platform with its own billing. That is a provider we would route to, not a way to speed up the three we support.
- **Fit.** Poor. It ties each model to an image build and does not speed up the cold arbitrary case. Rejected.

### SkyPilot (or similar) for provisioning and placement

SkyPilot chooses a cloud and a region, retries on capacity errors and manages spot recovery. It runs its own setup on the node over SSH, plus a controller for managed jobs.

- **What it buys.**
  - Placement across clouds. `gpu` already does placement within one provider: ranked offers, `host_attempts`, bad-host memory and pinned quotes.
  - SkyPilot does nothing about the stage that costs the time, the weight fetch.
- **What it costs.**
  - A large Python dependency and a second lifecycle owner on the same machine. Our watchdog's point is that one rule, `core/policy.py`, decides shutdown on both sides. A second orchestrator with its own autostop splits that.
  - We have not measured SkyPilot's setup time on these providers, and we would not adopt it to save the 1 s our API call takes.
- **Fit.** Poor for speed. Worth revisiting only if cross-provider failover becomes a goal of its own.

### Provider snapshots, templates and network volumes

| Mechanism | What it buys | What it costs | Fit |
| --- | --- | --- | --- |
| Vast cached images (already in use for vLLM) | 19–31 s boots | nothing extra | in use |
| Vast template or cached image for llama.cpp | 60–75 s saved on a host's first llama boot | Vast caches the images its own templates use, so we would need a llama.cpp image they cache, or accept the first pull | small win, cheap |
| Stopped machine with its disk ("parking") | ~41 s restart, weights already on disk | disk billing while parked: median $0.22–0.33 per GB-month on today's offers, about $0.27 for 6 h for a 120 GB disk | **good**: it is `lifecycle.keep_disk` plus a shorter `stopped_ttl`, already in the code |
| RunPod network volume / Lambda persistent filesystem with weights | cross-machine weight reuse within one region | ties the rental to that region's capacity; a monthly storage bill; per-provider code | later, per provider, if one provider dominates our use |

A restart can fail on Vast. A stopped machine's GPU can be rented by someone else, and the restart then waits or fails. Parking therefore needs a fallback to a cold setup after a short wait, not the 30-minute boot budget.

### A warm pool of idle machines

- **What it buys.** Attach in seconds to a server that is already loaded.
- **What it costs.**
  - The cheapest 80 GB card that passes our filters is $0.869/h today, about $630 a month per slot.
  - It inverts the product's promise that you pay only while it works.
  - `gpu` already has the pool machinery for queue mode (`queue/pool.py`), so building one is not the hard part. Paying for it is.
- **Fit.** Only as an explicit, owner-funded, lab-level choice for a model in daily use. Not a default.

### Caching weights near the GPU

| Mechanism | What it buys | What it costs | Fit |
| --- | --- | --- | --- |
| **A parallel, progress-reporting fetch from Hugging Face** (HF's Xet/CDN with many connections, stdlib code we own) | the biggest single win: the stage that is 89–96% of setup | a few hundred lines plus tests; no new dependency | **good** |
| A bandwidth floor sized to the model | removes slow hosts before renting | little: raising the floor from 500 to 2000 Mbps leaves the cheapest ≥80 GB offer at $0.869/h and moves the cheapest ≥40 GB offer from $0.244 to $0.268/h | **good** |
| A lab mirror (R2/S3/registry) | protection against HF rate limits or outages | storage, a sync job, and Vast hosts are worldwide, so one bucket is not "near" most of them | not now: no evidence HF is the bottleneck |

## 4. Recommendation

**Build a lab-authored fast path: a fast weight fetch first, then parking the last model's disk. Do not adopt Cog, SkyPilot or a warm pool now.**

### Phase 1: the fetch (cold path)

1. **The watchdog fetches the weights before the server starts.**
   - The fetch is a watchdog extension loaded through `POST /extend` (`watchdog/agent_body.py` L26-L31, L674-L677). It cannot go in the boot script, which has 400 bytes left.
   - The extension uses Python's standard library only, like the rest of the watchdog. It downloads each file with parallel HTTP range requests and writes it into the same cache path the engine reads: `HF_HOME` for vLLM, `LLAMA_CACHE` for llama.cpp.
   - The launcher waits for a "weights complete" marker before starting the server. vLLM then finds the files in its cache, and llama.cpp starts with `-m <path>` instead of `-hf`.
   - If the extension never arrives, the launcher falls back after a short wait to today's path, where the server downloads its own weights.
2. **Progress is a signal.** The extension reports bytes and rate. The CLI shows them, and a fetch below a floor for 2 minutes counts as a host failure, the same way `health.pull_stall_timeout` treats a stalled image pull today. A slow host costs 2 minutes instead of a 10–20 minute health timeout, and the failure log finally tells a slow download from a broken server.
3. **A bandwidth floor sized to the model.** Vast's `min_inet_down_mbps` rises with model size so the fetch targets about 3 minutes, never below today's 500.
4. **Readiness in 2 s.** The watchdog already shares the container with the server. It probes `/health` every 2 s and calls the CLI back, instead of the CLI polling every 30 s.

Expected result: the cold 27B setup drops from 557 s to about 300 s, the 26.6 GB GGUF from 1021 s to about 150 s, and failed hosts cost minutes instead of a timeout. Phase 1 is done when the benchmark confirms the fetch sustains 300 MB/s.

### Phase 2: parking (known model)

1. **`lifecycle.park`**, a named preset of what exists: `keep_disk: true` with `stopped_ttl: 6h` (default). An idle machine stops instead of being destroyed, and `gpu setup` with the same fingerprint restarts it.
2. **A restart that does not hang.** If the parked machine is not running within 2 minutes, which happens when its GPU was rented by someone else, setup destroys it and does a cold setup.
3. The sweep timer stays the only thing that enforces `stopped_ttl`, so `gpu agent install` makes the timer part of turning parking on.

Parking costs disk while parked: about $0.27 for 6 h on a 120 GB disk at today's median storage price. It is opt-in for the owner to make default. Expected result: about 41 s to a few minutes for the last model you used, depending on load time.

### Migration

1. Phase 1 lands behind `runtime.fetch: auto | server`, with `auto` as the default and `server` keeping today's behaviour.
2. The tests run against a local HTTP server that serves ranges.
3. One paid benchmark on Vast measures today's path against the new one, on the 27B model and the GGUF.
4. Phase 2 follows, opt-in.
5. The site and the docs change only after the benchmark gives real numbers.

### Risks

- **Hugging Face throughput.** It varies by region and time of day. 300 MB/s is an assumption until we measure it. If HF caps a single client lower, the targets move and the mirror option comes back.
- **Gated models.** The extension needs `HF_TOKEN`, which the machine already receives for vLLM and llama.cpp. The token stays in the machine's environment and never goes into the extension's source.
- **The extension path.** It adds one more thing that must work before the server starts. The fallback to the server's own download bounds that risk.
- **Parking.** It leaves a stopped machine billing disk on Vast. That is the G1 incident's shape (`docs/handoff/2026-11-11-video-validation-fixes.md` L30-L39). The fix for G1 already reports `still_billing` and the sweep timer destroys after `stopped_ttl`. Without the timer, a parked machine bills until someone destroys it.

### Cost

- **Build.**
  - Phase 1 is 2–3 days of engineering, most of it in the extension and its tests.
  - Phase 2 is about 1 day.
  - The benchmark is about one hour of an 80 GB card plus one hour of a 48 GB card, under $5 at today's prices.
- **Run.**
  - Phase 1 adds up to $0.024/h through the higher bandwidth floor, at the cheapest end of today's offers.
  - Phase 2 adds disk while parked, about $1 a day per parked 120 GB machine.
  - A warm pool, if ever chosen, is about $630 a month per 80 GB slot.

## 5. What needs the owner

- **Spend:** approve one benchmark session on Vast, under $5, to measure the fetch before and after.
- **Provider decision:** whether parking (stop and keep the disk for 6 h, about $1 a day per parked 120 GB machine) becomes the default on Vast instead of destroying on idle.
- **Spend (not recommended now):** a warm pool, about $630 a month per 80 GB slot, only if one model is in daily use.
- **New dependency:** none. The fetch is stdlib code we own, and Cog and SkyPilot are not adopted.
