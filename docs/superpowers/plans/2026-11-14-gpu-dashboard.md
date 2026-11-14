# GPU Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only `gpu dashboard` web UI (growing to full control) that shows every GPU, proxy and service on this host — cost, idle, health, live reachability — reachable from a phone behind a token.

**Architecture:** A stdlib `ThreadingHTTPServer` on loopback, sibling to `gpu gateway`. It serves a self-contained HTML page plus a JSON API (`/api/state`), reads host-wide data from the existing sweep registry + each project's `.gpu/state.json` + gateway liveness, and runs a background reachability watch. A per-user systemd service keeps it alive. Three phases: (1) read-only core, (2) serve & reach from phone, (3) full control.

**Tech Stack:** Python stdlib + `httpx` (already a dep) + Typer CLI + Pydantic v2 config. No web framework, no JS build.

**Spec:** `docs/superpowers/specs/2026-11-14-gpu-dashboard-design.md`

## Global Constraints

- Serving side is **stdlib + httpx only** — no Flask/FastAPI/uvicorn/node.
- **Read-only through Phase 2**: no route mutates anything; the server never loads a provider credential. Mutation arrives only in Phase 3.
- **No secret in the repo or logs.** The access token is generated on install, stored `chmod 600` under the state dir, printed once, never committed, never logged.
- Bind **127.0.0.1 by default**; binding a non-loopback host **requires** a token or the server refuses to start.
- Degrade gracefully: a corrupt state file, unreachable endpoint, or missing registry entry renders as "unknown/unreachable", never a crash.
- Follow the `gateway` package's shape (`service.py`, CLI command set `start|install|uninstall|status|logs`, config section off by default).

## Review Focus

- **A corrupt / partial `.gpu/state.json` in one project** must not blank the whole page — that project renders "unknown", others unaffected. (Task 2)
- **An endpoint that is up but slow, or refuses the connection (`Errno 110`)** must show as a failing probe in the window, not hang the watch thread or the request. (Task 3)
- **Non-loopback bind with no token** must refuse to start, not serve the fleet open. (Task 6)
- **A valid token sent as `Authorization: Bearer`, as `?token=` once, and as the resulting cookie** must all authenticate; a wrong/absent token gets 401. (Task 6)
- **A Phase-3 mutating route reached without a token, or without the confirmation field** must 401/400 and do nothing. (Task 8)

---

## Phase 1 — Read-only core

### Task 1: `DashboardConfig` in config

**Files:**
- Modify: `gpu4agents/core/config.py`
- Test: `tests/test_dashboard.py` (create)

**Interfaces:**
- Produces: `DashboardConfig(enabled: bool=False, listen: str="127.0.0.1:8137", watch_interval: str="15s", history: str="20m", token_required_off_loopback: bool=True, options: dict)`, with `listen_host_port -> tuple[str,int]` (reuse the gateway validator logic). `RuntimeConfig.dashboard: DashboardConfig`.

- [ ] **Step 1: Write the failing test** — `test_config_defaults_dashboard_off`:
```python
def test_config_defaults_dashboard_off():
    rc = RuntimeConfig.model_validate(BASE_CONFIG)
    assert rc.dashboard.enabled is False
    assert rc.dashboard.listen_host_port == ("127.0.0.1", 8137)
```
and `test_dashboard_bad_listen_rejected` asserting `DashboardConfig(listen="nope").listen_host_port` raises `ValueError`.
- [ ] **Step 2: Run to verify fail** — `pytest tests/test_dashboard.py -k config -v` → FAIL (no attribute `dashboard`).
- [ ] **Step 3: Implement** `DashboardConfig(_Base)` mirroring `GatewayConfig` (durations validated with the module's `_duration` helper; `listen_host_port` property copied from `GatewayConfig`); add `dashboard: DashboardConfig = Field(default_factory=DashboardConfig)` to `RuntimeConfig`.
- [ ] **Step 4: Run to verify pass** — same command → PASS.
- [ ] **Step 5: Commit** — `feat(dashboard): config section`.

### Task 2: Host-wide snapshot collectors

**Files:**
- Create: `gpu4agents/dashboard/__init__.py`, `gpu4agents/dashboard/collect.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `gpu4agents.core.sweep.ProjectRegistry().paths() -> list[Path]`; `RuntimeState` loaded from a project's `.gpu/state.json` (fields `instance_id, state, model_id, endpoint, hourly_cost_usd, session_cost_usd, last_activity, api_key`).
- Produces: `project_snapshots(paths: Iterable[Path], now: float) -> list[dict]` where each dict is `{project, dir, instance_id, model, endpoint, state, hourly_cost_usd, session_cost_usd, idle_seconds, reachable: None}` (reachable filled by the watch later); `api_key` is **never** included. A path whose state is missing/corrupt yields `{project, dir, error: "<reason>"}` and never raises.

- [ ] **Step 1: Write the failing test** — `test_project_snapshot_reads_state` (writes a tmp `.gpu/state.json`, asserts fields incl. `idle_seconds` computed from `last_activity`, and `"api_key" not in snap`) and `test_corrupt_state_becomes_error_row` (writes garbage, asserts the row has `error` and the call returns, no raise).
- [ ] **Step 2: Run to verify fail** → FAIL (module missing).
- [ ] **Step 3: Implement** `project_snapshots` in `collect.py`: for each path, load state defensively (try/except → error row), project name = `dir.name`, compute `idle_seconds = now - last_activity` when present. Explicitly drop `api_key`.
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat(dashboard): project snapshot collector`.

### Task 3: Reachability watch

**Files:**
- Create: `gpu4agents/dashboard/probe.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Produces: `ReachabilityWatch(interval_s: float, history_s: float, client=httpx)` with `probe_once(endpoint: str) -> tuple[bool, float|None, str|None]` (ok, connect_ms, err), `record(endpoint, result, ts)`, `snapshot(endpoint) -> dict` = `{last_ok, last_connect_ms, window: [...], failures_in_window, errno110_in_window}`, and `start()/stop()` for the background thread. A deque per endpoint trimmed to `history_s`.

- [ ] **Step 1: Write the failing test** — `test_probe_records_failure_window`: feed `record()` three results including one with err `"[Errno 110] ..."`, assert `snapshot()["failures_in_window"] == 1` and `errno110_in_window == 1`; `test_window_trims_old` asserts entries older than `history_s` drop out. Use injected timestamps — no real sleeping.
- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement** the deque bookkeeping and `snapshot()`. `probe_once` does `httpx.get(endpoint + "/health")` with a short connect timeout, falls back to `HEAD /`, returns connect ms and any exception string. The thread loop is thin; tests drive `record()` directly.
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat(dashboard): reachability watch`.

### Task 4: `/api/state` server + minimal page, token gate

**Files:**
- Create: `gpu4agents/dashboard/auth.py`, `gpu4agents/dashboard/server.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: Task 2 collectors, Task 3 watch, `DashboardConfig`.
- Produces: `TokenGate(token: str|None)` with `check(headers: dict, path_query: str) -> bool` and `cookie_header(token) -> str`; `DashboardServer(config, collectors, watch, token_gate, log)` with `serve_forever()/stop()`, serving `GET /api/state` (JSON: `{instances, projects, proxies, probes, generated_at}`), `GET /` (minimal HTML that fetches `/api/state`), `HEAD /` (200 liveness). Unknown path → 404.

- [ ] **Step 1: Write the failing test** — `test_api_state_shape` boots the server against fake collectors on a free port, GETs `/api/state`, asserts the five top-level keys and that a project row carries no `api_key`. `test_token_gate_accepts_bearer_and_cookie_rejects_absent`: with a token set, no header → 401, `Authorization: Bearer <t>` → 200, `?token=<t>` → 200 + `Set-Cookie`, the cookie alone → 200, wrong token → 401. `test_no_token_configured_is_open_on_loopback` → 200 without any token.
- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement** `auth.py` (constant-time compare, cookie name `gpu_dash`) and `server.py` (stdlib handler; JSON assembled from collectors + `watch.snapshot` per endpoint; minimal inline HTML placeholder page). Reachable fields merged: each project/instance row gets its `reachable` from the watch snapshot.
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat(dashboard): api/state server + token gate`.

### Task 5: `gpu dashboard start` wired, Phase-1 validation

**Files:**
- Create: `gpu4agents/cli/dashboard.py`
- Modify: `gpu4agents/cli/main.py` (import, `QUIET_COMMANDS`, `add_typer`)
- Modify: `CHANGELOG.md` (`[Unreleased]` entry)

**Interfaces:**
- Consumes: all Phase-1 modules; `Account`/provider from `cli/account.py` for the instance list (optional in P1 — projects alone are enough to validate).

- [ ] **Step 1: Write the failing test** — `test_cli_dashboard_start_help` asserts `gpu dashboard --help` lists `start`. (CLI smoke; the server behavior is covered by Task 4.)
- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement** `dashboard_app` Typer with `start` (build collectors from `get_context`, instantiate watch from config durations, bind, `serve_forever`); wire into `main.py` exactly as `gateway` is; add `"dashboard"` to `QUIET_COMMANDS`; CHANGELOG entry.
- [ ] **Step 4: Verify** — `pytest tests/test_dashboard.py -v` all PASS; then manual validation: run `gpu dashboard start` against this host, `curl 127.0.0.1:8137/api/state`, confirm the live 27B + proxy show up with reachability. Record the result.
- [ ] **Step 5: Commit** — `feat(dashboard): gpu dashboard start (phase 1 read-only core)`.

---

## Phase 2 — Serve & reach from phone

### Task 6: Full HTML page + non-loopback bind gate

**Files:**
- Create: `gpu4agents/dashboard/page.html` (served by `server.py`)
- Modify: `gpu4agents/dashboard/server.py` (serve the asset), `gpu4agents/cli/dashboard.py` (refuse non-loopback bind without token)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Produces: a self-contained page (inline CSS/JS, no CDN) polling `/api/state` every few seconds, rendering per-GPU cards (model, host, $/h, session cost, idle, state, reachable), per-proxy rows (up?, service active?, recent error ratio), and a compact reachability timeline.

- [ ] **Step 1: Write the failing test** — `test_non_loopback_without_token_refuses`: constructing/starting the server with `listen="0.0.0.0:8137"` and no token raises/exits with a clear error; with a token it starts. `test_page_served_and_selfcontained`: `GET /` returns HTML containing no `http://` external script/style src.
- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement** the bind guard in the start path and the `page.html` asset; serve it at `/`.
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `feat(dashboard): full page + non-loopback bind guard`.

### Task 7: systemd service + `install/uninstall/status/logs`

**Files:**
- Create: `gpu4agents/dashboard/service.py`
- Modify: `gpu4agents/cli/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Produces: `render_unit(*, project_dir, python, gpu_home=None) -> str`, `install_service/uninstall_service/is_installed/unit_name` mirroring `gateway/service.py`, but `ExecStart=... dashboard start` and unit prefix `jambu-gpu-dashboard`. `install` also generates+stores the token (`chmod 600`, printed once) when the configured bind is non-loopback.

- [ ] **Step 1: Write the failing test** — `test_service_unit_is_restarting_and_boot_safe` (asserts `Type=simple`, `Restart=always`, `WantedBy=default.target`, `dashboard` in ExecStart) and `test_service_slug_per_project` (stable, distinct per dir) — same shape as the gateway service tests.
- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement** `service.py` (copy gateway's, swap the exec args + prefix) and the four CLI commands; token generation on install.
- [ ] **Step 4: Verify** — tests PASS; manual: `gpu dashboard install`, check `systemctl --user status`, reach `/api/state` from the phone over Tailscale/tunnel with the token. Record result.
- [ ] **Step 5: Commit** — `feat(dashboard): systemd service + lifecycle commands (phase 2)`.

---

## Phase 3 — Full control

### Task 8: Mutating routes behind token + confirmation

**Files:**
- Modify: `gpu4agents/dashboard/server.py` (POST routes), `gpu4agents/dashboard/page.html` (action buttons + confirm), `gpu4agents/core/config.py` (`DashboardConfig.control_enabled: bool=False`)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: the project engines' existing operations (`stop`, `destroy`, lock) and the gateway service controls — invoked **per project** via the same engine seam the CLI uses; never a raw provider call added here.
- Produces: `POST /api/control/{action}` with JSON body `{project, confirm: true}`; actions `stop|destroy|restart-proxy|lock`. Requires the token **and** `confirm`. Mutating routes are served **only** when bound to loopback unless `control_enabled` plus token; default `control_enabled=False`.

- [ ] **Step 1: Write the failing test** — `test_control_requires_token_and_confirm`: POST without token → 401; with token but `confirm` absent → 400; with both → the injected fake action is called once. `test_control_disabled_returns_404`: with `control_enabled=False`, the route is absent (404). Use an injected action registry so no real GPU is touched.
- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement** the POST handler, the config flag, the confirm gate, and wire actions to the engine seam (injected in tests).
- [ ] **Step 4: Run to verify pass** → PASS; manual validation on the live host deferred to the user (it can destroy a GPU).
- [ ] **Step 5: Commit** — `feat(dashboard): control actions behind token + confirm (phase 3)`.

### Task 9: Whole-branch review + CHANGELOG finalize

- [ ] Run the full `tests/test_dashboard.py` suite + the repo's existing suite (minus the known pre-existing `respx` failures) and confirm green.
- [ ] Finalize the `CHANGELOG.md` `[Unreleased]` entry to cover all three phases.
- [ ] Fresh reviewer pass on the whole branch (security focus on the token gate, the bind guard, and the Phase-3 mutation routes), then open the PR.
