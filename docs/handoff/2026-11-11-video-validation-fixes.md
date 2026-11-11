# Fixes found while filming the gpu4agents home page video (2026-11-11)

You are fixing problems that showed up when a real Claude Code session used GPU for Agents 0.12.0 end to end.
The session was filmed for the home page video, and the same run validated the product-video plugin.
Everything below was observed, not guessed; each item says where the evidence is. Work in two repositories:

- **gpu4agents**: `/home/administrator/workspaces/Jambu/Internal/jambu-gpu` (CLI, watchdog, `SKILL.md` installed by
  `gpu agent install`, docs). Items G1–G8.
- **product-video plugin**: `/home/administrator/workspaces/Jambu/Internal/product-video`, branch
  `feat/video-capabilities`, engine in `plugins/product-video/skills/make-video/engine` (tests: `npm test` there).
  Items P1–P6. 1.2.2 already fixed a first batch (listed at the end); don't redo those.

Rules for this work: tests first for every behaviour change; keep the docs (`docs/LIFECYCLE.md`, `docs/CONFIG.md`,
`docs/AGENTS.md`, `SKILL.md`, `CHANGELOG.md`) in step with the code; never read or print provider keys (run anything
that needs Vast with `INFISICAL_PATH=/shared devrun ...`); ask the user before renting anything.

## Evidence you can read

- The two recorded agent projects (gpu4agents state, `gpu.yaml`, `.gpu/state.json`, logs):
  `/tmp/claude-1000/-home-administrator-demo-studio/9debbcee-26f6-48e3-9f2d-e99fa41c3840/scratchpad/hero-gpu/` (take 1)
  and `/tmp/claude-1000/-home-administrator-demo-studio/378f23c4-a50a-41ac-bbaf-2d154953755b/scratchpad/heroimg/` (take 2).
  Their registry is `/home/administrator/demo/home/.gpu/projects.json` (the filmed agent ran with HOME=/home/administrator/demo/home).
  Read events with: `cd <project> && INFISICAL_PATH=/shared devrun bash -c 'env -i HOME=/home/administrator/demo/home PATH=/home/administrator/demo/bin-gpu:/usr/bin:/bin VAST_API_KEY="$VAST_API_KEY" gpu events'`
- What the agent said and did, as text: `jambu-gpu/video/public/footage/with.screens.txt`, `with-bench.screens.txt`;
  full transcripts: `/home/administrator/demo/claude-clean/projects/-home-administrator-demo-studio/*.jsonl`.
- The skill the agent followed: `/home/administrator/demo/studio/.claude/skills/gpu4agents/SKILL.md` (same as the repo's).

## gpu4agents

### G1. The agent's cleanup leaves a stopped machine billing, forever (highest priority)

Observed: take 2's machine (Vast 60143985, RTX 2060) shows `instance_stopped reason=manual` at 01:10:43. The agent
itself ran `gpu stop` when done. It stayed STOPPED, billing disk, with its watchdog off. `stopped_ttl` (24h) is only
enforced from this computer (`gpu guard` / `gpu sweep` / the sweep timer), and no timer is installed here
(`systemctl --user list-timers` shows none), so nothing would ever destroy it.

Cause: `SKILL.md` rule 4 says *"offer `gpu_stop` (cheap to resume) or `gpu_destroy`… Destroy only with the user's
explicit yes."* That contradicts `docs/LIFECYCLE.md` §"A stopped Vast.ai instance still bills…": the automatic
policy destroys instead of stopping wherever a stop leaves a bill, unless `lifecycle.keep_disk: true`.

Fix:
1. `SKILL.md` rule 4: when the work is done and the outputs are pulled, the default is `gpu_destroy`. This is the
   same choice the watchdog makes, and nothing keeps billing. Offer `gpu_stop` only when the user says they'll be
   back soon, or when `keep_disk: true`, and say that it bills disk until destroyed. Teaching the user to destroy
   isn't a destructive surprise: the results are already home. Keep "ask before renting" as is.
2. In the CLI, on a provider where a stopped machine bills and `keep_disk` is false, make `gpu stop` (and the MCP
   `gpu_stop`) print and return a warning naming the bill: "stopped: still billing ~$X/day for disk; destroyed
   after stopped_ttl only if `gpu sweep --install-timer` is set up". Also consider `gpu stop --destroy-after <dur>`.
3. Make sure something enforces `stopped_ttl` on a machine with no timer. Either `gpu agent install` installs the
   sweep timer (ask once, in the install flow), or every `gpu` command runs a cheap account-wide stopped-orphan
   check and warns. Write an acceptance test for "manual stop, then nothing else runs" so it can't regress.

### G2. The agent kept its GPU project in an ephemeral session folder; later sessions see an "orphan"

Observed: both takes created `gpu.yaml` under the Claude session scratchpad (`/tmp/claude-1000/.../scratchpad/heroimg`).
A new session in the same repo reported the machine as *"an RTX 3060 running… that no project tracks… I can't use it
because I don't have its key"* (`with.screens.txt`), and `gpu ls` from anywhere else shows `PROJECT orphan`.

Fix:
1. `SKILL.md`: the project lives in the user's repository: `gpu.yaml` plus `.gpu/` at the repo root, with `.gpu/`
   in `.gitignore`. Never in a temp or scratchpad folder. Generated scripts may live anywhere; the project may not.
2. CLI: `gpu init` refuses (or warns loudly) when the directory is under the system temp dir.
3. Add `gpu adopt <instance-id>`, or `gpu setup --adopt`. It takes over an orphan this account owns (re-identifies
   its watchdog, writes local state) so a new session can reuse or destroy it. Mention it in `SKILL.md` where it
   tells the agent to report orphans.

### G3. GPU sizing: the agent picked a card too small for the work

Observed: the agent set `compute.gpu.min_vram_gb: 9` for SDXL (`heroimg/gpu.yaml`). It got a 12 GB card, and then:
- SDXL VAE decode at 1344×768 ran out of memory, so it fell back to 1024×576;
- the CUDA benchmark lost its 2 largest shapes because the idle SDXL server still held 11 GB of the same card.

Image generation took about 2.5 min an image.

Fix:
1. Add `gpu.models.yaml` profiles for the common image models (SDXL base 1.0: `min_vram_gb: 16`, and 24 to be safe
   at 1344×768 or above; FLUX: 24+) so the agent doesn't guess.
2. `SKILL.md` sizing rule:
   - size for the peak with about 1.5–2× headroom, not for the weights;
   - when a server and a job share one card, stop the server before the job, or size for both;
   - compare offers by the cost of the whole job (price/h × expected minutes on that card), not price/h alone: a
     faster card at 3–4× the hourly price often costs about the same per job and finishes far sooner.
3. Consider a `gpu offers --for <minutes-on-a-4090>` view that ranks by estimated job cost.

### G4. The card the user approved is not the card that was rented

Observed: the approval question quoted an **RTX 3060 12 GB at ~$0.08/h**. The machine was an **RTX 2060**, and the
agent only noticed in its final report. Take 2's machine was rented after a host retry.

Fix: when the agent asks for approval, pin what it quoted: the offer id, or `gpu.name_filter` for that card. If
`host_attempts` falls back to a different card or a higher price, `gpu setup` must say so in its result
(`rented_gpu`, `rented_price` beside `quoted_*`), and `SKILL.md` must tell the agent to report the change before
starting work. A higher price than the approved one needs a new yes.

### G5. The machine that vanished left no reason behind

Observed: take 1's machine (60143519) went away while nobody watched (it was running at ~00:38, gone by ~01:08).
The local log says only `state_reconciled … state=destroyed, reason=instance missing at provider`, and `gpu status`
says `watchdog  not installed`. That project's events also have no `watchdog_identified`. We can't tell whether the
watchdog destroyed it (idle, as designed) or something else did.

Fix:
1. Find out whether take 1's watchdog was installed. Take 1 went through `job_submitted` with `engine: hf`; check
   that path installs and identifies the watchdog like `gpu setup` does. If it doesn't, that's a bug.
2. Before the watchdog stops or destroys its machine, it records the reason somewhere the CLI can read after the
   machine is gone. Options:
   - the Vast instance label (set it to `jambu-…:destroyed:idle` just before the call);
   - a small final POST to a URL from `gpu.yaml`;
   - for Vast, the instance's last log line, if the API exposes it.

   On reconcile, `gpu events` then records `destroyed reason=idle_timeout (watchdog)`, not "missing at provider".

### G6. The agent enumerated the secret store

Observed: in the first (unsealed) take, the agent ran `INFISICAL_PATH=/shared devrun env | cut -d= -f1` to list
secret names. It then used an OpenAI key from there. That part was the operator's mistake: the agent had the
operator's environment.

Fix in `SKILL.md`: the only check for credentials is `gpu login --check`. Never list environment variables or a
secret manager to find keys, and never use another provider's key for work gpu4agents can do.

### G7. Idle timeout is configurable: the docs and the agent's messages should say so

The agent told the user "it stops by itself after 15 min idle" as if fixed. In `SKILL.md` and the approval message,
say "after `idle_timeout` (15 min by default; yours to change in gpu.yaml)". The video now says "you set … how long
it may sit idle".

### G8. Things that worked and must keep working (add them to `docs/ACCEPTANCE.md`)

- The agent asked before renting, with the model, card, $/h, the $1/h cap and the $5 session cap.
- It reused the running machine for a second task instead of renting another.
- It reported cost (~$0.06) and what was still billing.
- It recovered by itself from the out-of-memory error and from NVRTC missing headers.
- After the session was killed, take 1's machine was gone within about 30 min (confirm with G5).

## product-video plugin

### P1. Hyperframes layers can't be translated

`make/hyperframes.mjs` renders one file with literal `variables`, so a layer with words shows English in every
language. For this video the safety layer was made wordless and the words moved to a clip label. Fix:
- `variables` values may be story keys (`"$safety.line1"`);
- when any is, render one file per language (`layers/safety.<lang>.mp4`) and let `src/footage.tsx` pick the
  language's file;
- hash per language;
- test it.

### P2. Clip labels show `*emphasis*` literally

`Clips` in `src/footage.tsx` prints the label text raw. Render `*…*` as the accent colour, like the other scenes
(`Accented`), or strip the asterisks.

### P3. Terminal capture can't answer an agent's choice menu on purpose

Claude Code's question dialogs (`AskUserQuestion`) take arrows and Enter, not typed text. Today the script types
"Yes, go ahead." and Enter happens to pick the recommended option. Add a `{ "choose": 1 }` step: N−1 Down presses,
then Enter, when a menu (`Enter to select`) is on screen. Document it in `references/footage.md`.

### P4. Each take needs a manual reset

Between takes I had to delete the demo repo's outputs and the clean config's project memory
(`claude-clean/projects/<repo>`) by hand. Add `"before": ["git clean -fd site kernels", …]`: shell commands run in
`cwd` before recording, refusing paths outside it. Add `"fresh": true` too, to move the agent config's
`projects/<cwd-slug>` aside so the agent starts with no memory of earlier takes.

### P5. A filmed session's own GPU spend is not in the video's ledger

The filmed agent rents on the same Vast account, outside `budget.json`. After a capture, read the cost the agent's
gpu4agents project recorded (`gpu --json status` in its project), or ask the operator. Then add it to the ledger
as `note: "filmed session"` so the budget report is the real total.

### P6. Claude Code chrome in the footage

The recordings show the model banner, a "Share Claude Code and earn $10" line and auto-mode hints. Strip crops
avoid them now. A launcher option to hide tips would be cleaner (check `claude --help` and the settings for tips
and notices) and should be documented for filmed sessions.

### Already fixed in 1.2.2 (commit 2f4f154): don't redo

- Blender on a rented GPU:
  - each offer priced by the minutes that card needs, fastest within 1.5× the cheapest total;
  - the card pinned with `name_filter`;
  - 24 GB and compute 8.6 or newer;
  - `idle_timeout` 5m;
  - OptiX denoising on the card (40 s → 3 s a frame);
  - job timeout at 3× the estimate;
  - failed jobs charged to the ledger.
- `npm run check` no longer warns about a missing Blender when the clip renders on a rented GPU.
- Terminal capture:
  - `"clean"`: an empty environment with its own HOME and PATH, and only the keys named in `pass`;
  - `until` (a shell check) and `while` (a busy pattern) waits;
  - `zoom` with a start scale, and short windows centred on the stage.
