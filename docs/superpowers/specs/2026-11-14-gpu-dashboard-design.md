# `gpu dashboard` — a read-only control-plane web UI

Date: 2026-11-14
Status: proposed (awaiting approval)

## Intent and why

Operating gpu4agents today means driving the CLI (or an agent) for every
look: "what is running, what does it cost, is it idle, is the proxy
healthy, why did that request 500?" A real incident made the gap concrete
— a client saw intermittent `API Error`s that turned out to be transient
TCP connect timeouts (`Errno 110`) to the rented host; confirming it was
not our side took an agent, a shell, and a handful of probes.

The user wants to stop needing an agent for that routine watching. A
single web page, reachable from their phone, that shows every GPU, proxy
and service at a glance — cost, idle, health, and live reachability — and
keeps watching so a blip like the `Errno 110` run is visible the moment it
happens, not reconstructed after the fact.

This is **v1: read-only**. It shows; it never spends money or destroys
anything. Controls (provision / stop / destroy / restart proxy) are a
deliberate later version, gated behind their own security review.

## Goals

- One page that answers, without an agent: what is running, where, at what
  $/h, how idle, how healthy, and reachable or not — across **all** jambu
  projects on this host, not one project.
- A server-side **watch** that probes each endpoint's reachability on an
  interval and keeps a short rolling history, so transient faults
  (`Errno 110`, 5xx bursts, slow connects) are caught and shown.
- Surface the **proxy / gateway** layer too: is the loopback endpoint up,
  is its systemd service active, what is the recent error/success ratio.
- **Reachable from the user's phone**, safely: read-only, and never served
  open to the internet without a token.
- House style: no new runtime stack. A stdlib loopback HTTP server plus a
  per-user systemd service, exactly like `gpu gateway`.
- Lives in the open-source tool; no private data or secrets enter the repo.

## Non-goals (v1)

- **No control actions.** No rent, stop, destroy, lock, restart, or config
  edit from the UI. Read-only end to end.
- **No multi-host fleet.** It reports what this host can see (its registry
  + the projects on it). Aggregating several machines is later.
- **No historical datastore / metrics TSDB.** A short in-memory rolling
  window (minutes, not days) is enough to see a blip. Long-term history is
  later and probably belongs in a real metrics tool, not here.
- **No user accounts / RBAC.** A single shared bearer token gates access.
- **No write path to provider APIs.** The dashboard never holds or uses a
  provider key; it reads local state and probes endpoints only.

## Anti-goals (these would make it a failure even if it "works")

- A page that can be reached from the open internet and exposes endpoints,
  instance IDs, or costs without a token.
- Anything that can spend money or destroy a GPU slipping into v1 "just
  because it was easy."
- A heavyweight SPA / build toolchain (node, bundlers) landing in a
  stdlib-only, `uv`-installed tool.
- Secrets, tokens, endpoints or costs committed to the repo.

## Constraints

- **Stdlib on the serving side.** Like `gateway` and the on-GPU servers,
  the dashboard server uses only the Python stdlib + `httpx` (already a
  dependency) for its outbound probes. No Flask/FastAPI/uvicorn.
- **Read-only data sources already exist:**
  - the sweep **registry** + provider `list_instances()` already enumerate
    every jambu instance across projects;
  - each project's `.gpu/state.json` gives endpoint, model, $/h, session
    cost, idle, lock, watchdog;
  - the **gateway** already runs a loopback server with a `HEAD` liveness
    probe and a systemd unit whose `is_installed` / active state is
    queryable;
  - reachability is a plain connect-timed `GET /health` (or `HEAD /`), the
    same probe used to diagnose the `Errno 110` incident.
  The dashboard **reads** these; it does not reach into provider APIs.
- **No secret ever leaves local state.** The access token is generated and
  stored under the user's state dir (e.g. `~/.config/gpu4agents/` or the
  project `.gpu/`), `chmod 600`, printed once on install, never committed,
  never logged.
- Must degrade gracefully: a corrupt state file, an unreachable endpoint,
  a missing registry entry never crash the page — they render as a
  "unknown / unreachable" cell.

## Approaches considered

There is effectively one sensible build given the existing `gateway`
pattern; the alternatives below are recorded to show why.

1. **`gpu dashboard`: stdlib loopback server + systemd service, sibling to
   `gateway` (chosen).** A `ThreadingHTTPServer` on 127.0.0.1 serves (a) a
   single self-contained HTML/JS page and (b) a small JSON API
   (`/api/state`). A background thread runs the reachability watch on an
   interval and caches a rolling window. A per-user systemd unit
   (`Restart=always`, linger) keeps it alive across logout/reboot, reusing
   the `gateway/service.py` rendering almost verbatim. Phone access is via
   the user's existing private network (Tailscale) or an SSH tunnel to the
   loopback port; binding to a non-loopback host is allowed **only** with a
   token set. *Why:* zero new stack, reuses registry+state+service code,
   matches the mental model the user already has from the gateway.

2. **A static page the CLI regenerates on a timer (no server).** `gpu`
   writes an HTML snapshot to a file every N seconds; the user opens the
   file. *Rejected:* no live probing, no reachability watch (the whole
   point), and no clean way to reach it from a phone.

3. **A full web app (FastAPI + a JS SPA).** *Rejected:* drags a web
   framework and a node/bundler toolchain into a stdlib-only `uv` tool for
   a single read-only page; disproportionate, and a maintenance burden on
   the open-source side.

**Decision:** approach 1.

## The how (decided)

- **Command surface:** `gpu dashboard start | install | uninstall | status
  | logs`, mirroring `gpu gateway`. `start` runs the server in the
  foreground; `install` writes the systemd user unit; `status`/`logs` wrap
  the unit. (My call, following the gateway precedent.)
- **Config:** a `dashboard:` section in `gpu.yaml`, off by default:
  `enabled`, `listen` (default `127.0.0.1:8137`), `watch_interval`
  (default `15s`), `history` (default `20m`), `token` handling. Since the
  dashboard is host-wide (all projects), it also reads a host-level config
  if present; per-project `gpu.yaml` can still launch it. (My call; exact
  precedence noted for the builder.)
- **Data model / API:** `GET /api/state` returns JSON:
  `{ instances: [...], projects: [...], proxies: [...], probes: {...},
  generated_at }`. `instances` from the registry/provider list; `projects`
  from each discoverable `.gpu/state.json`; `proxies` from gateway liveness
  + systemd active state; `probes` the rolling reachability window per
  endpoint (last status, connect time, recent failures, e.g. the
  `Errno 110` count in the window).
- **The page:** one self-contained HTML file (inline CSS/JS, no external
  CDN), polling `/api/state` every few seconds, rendering cards/rows:
  per-GPU (model, host, $/h, session cost, idle, state, reachable?),
  per-proxy (up?, service active?, recent error ratio), and a compact
  reachability timeline that would have made the `Errno 110` run obvious.
  Theme follows the project's existing site styling where cheap.
- **Security (my calls, within the chosen scope):**
  - Bind **127.0.0.1 by default.** Reaching it from the phone is expected
    to go over the user's private network (Tailscale) or an SSH tunnel —
    documented, not a new exposure.
  - Binding to any non-loopback host **requires** a token; the server
    refuses to start open without one. Token is a bearer in `Authorization`
    (and accepted as a cookie after a one-time `?token=` hand-off so a
    phone browser stays logged in). Generated on `install`, stored
    `chmod 600` in the state dir, printed once.
  - Read-only: the server exposes no route that mutates anything; it never
    loads a provider credential.
- **Reachability watch:** background thread, `httpx` `GET /health` (fallback
  `HEAD /`) per known endpoint every `watch_interval`, short connect
  timeout; records `(ts, ok, connect_ms, err)` into a per-endpoint deque
  trimmed to `history`. This is the generalization of the probe loop used
  to diagnose this session's incident.
- **Packaging:** new `gpu4agents/dashboard/` package (server, service,
  the HTML asset, data collectors) + `gpu4agents/cli/dashboard.py`, wired
  into `cli/main.py` like `gateway`. `CHANGELOG.md` `[Unreleased]` entry.
  Tests mirror `test_gateway.py`: unit tests for the collectors and the
  probe window, an end-to-end test that boots the server against fake
  state + a fake upstream and asserts the JSON and the token gate.

## Left to the builder

- Exact host-level vs per-project config precedence for discovering
  "all projects" (a registry of project dirs vs scanning); pick the
  simplest that reuses what the registry already tracks.
- The page's visual layout and the precise fields per card, within the
  data model above.
- Port number default (8137 is a placeholder; pick an unused one and keep
  it out of the gateway/watchdog range).
- Whether `status` does a self `HEAD` probe like the gateway's.

## Out of scope / future versions

- v2: control actions (stop/destroy/restart-proxy/lock) behind the token
  **and** a confirmation step, each a mutating route added deliberately.
- Multi-host fleet aggregation.
- Long-term metrics / alerting (likely an export to a real metrics tool
  rather than growing a datastore here).
