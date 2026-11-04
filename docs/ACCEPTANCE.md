# Acceptance criteria: where each one is implemented and tested

Eighteen guarantees, mapped to the code and the test that holds each one.

| # | Guarantee | Implementation | Test |
| --- | --- | --- | --- |
| 1 | Repeated `setup` does not duplicate infrastructure | `core/engine.py::Engine.setup` reconciles first, compares `config.fingerprint()`, and holds an exclusive `.gpu/runtime.lock` | `test_engine.py::test_setup_twice_does_not_create_two_instances`, `::test_setup_refuses_to_reuse_an_incompatible_instance` |
| 2 | Provider APIs do not leak into core orchestration | `providers/base.py` contract; all Vast wire format confined to `providers/vast/{client,mapper}.py` | `test_engine.py::test_the_core_never_branches_on_a_provider_name` (greps `core/`) |
| 3 | Missing credentials fail before provisioning | `VastProvider.validate` returns before any API call; `CredentialResolver` never reads gpu.yaml | `test_vast_provider.py::test_validate_fails_before_provisioning_without_a_key` (asserts zero HTTP calls) |
| 4 | Failed provisioning cleans up | `Engine.setup` persists state *before* the long wait, then destroys on failure when `cleanup_on_setup_failure` | `test_engine.py::test_failed_setup_cleans_up_partially_created_resources` |
| 5 | Active workloads are never stopped by the idle guard | `core/policy.py::evaluate` returns `workload_active` before any idle check | `test_policy.py::test_active_workload_is_never_stopped_by_the_idle_guard`, `test_watchdog.py::test_a_heartbeating_workload_is_never_stopped` |
| 6 | An idle unlocked instance stops within `idle_timeout + watchdog_interval` | remote watchdog loop, `watchdog/agent_body.py::guard_loop` | `test_watchdog.py::test_idle_instance_is_stopped_with_a_recorded_reason` (real subprocess) |
| 7 | `max_lifetime` bounds forgotten instances | `policy.evaluate` checks age before activity | `test_policy.py::test_max_lifetime_stops_even_when_busy`, `test_watchdog.py::test_max_lifetime_stops_a_busy_instance` |
| 8 | Locks prevent shutdown until expiry | `policy.Snapshot.lock_active` / `lock_expired`; TTL required unless `allow_indefinite_lock` | `test_policy.py::test_expired_lock_stops_protecting`, `test_watchdog.py::test_a_lock_prevents_shutdown_until_it_expires` |
| 9 | A crashed CLI does not disable protection | the guard runs *on the instance*; heartbeats go stale after `heartbeat_grace` | `test_watchdog.py::test_a_vanished_heartbeat_falls_back_to_idle`, `::test_startup_grace_expires_when_the_cli_never_confirms` |
| 10 | Remote state is reconciled | `Engine._reconcile` merges provider truth + watchdog truth and repairs local state | `test_engine.py::test_status_repairs_state_when_the_instance_vanished` |
| 11 | A new provider needs only the contract | `providers/registry.py`; the engine is driven end to end by `tests/fakes.py::FakeProvider` | `test_engine.py::test_a_new_provider_needs_only_the_contract` |
| 12 | Every automatic shutdown records its reason | `lifecycle_guard_triggered` + `instance_stopped` events carry `reason`, locally and on the instance | `test_engine.py::test_guard_stops_an_idle_unlocked_instance_and_records_the_reason`, `test_watchdog.py::test_idle_instance_is_stopped_with_a_recorded_reason` |
| 13 | A restarted instance stays guarded | `Engine._reconcile` takes running/stopped from the provider; the watchdog's `terminated` flag only counts when the provider cannot tell. `RuntimeState.session_started_at` restarts on every start, and `max_lifetime` and the session budget are measured from it | `test_engine_sessions.py::test_reconcile_trusts_the_provider_over_a_stale_terminated_flag`, `::test_a_stale_terminated_flag_does_not_disable_the_guard`, `::test_restarting_a_stopped_instance_starts_a_new_session`, `::test_guard_budget_resets_with_a_new_session`; watchdog-side session tests in `test_policy.py` / `test_watchdog.py` |
| 14 | Traffic on the model endpoint is activity | `VllmRuntime.activity_probe` hands the watchdog vLLM's `/metrics` gauges and counters; `lifecycle.traffic_counts_as_activity` turns it off | `test_engine_sessions.py::test_the_watchdog_is_told_how_to_read_vllm_traffic`, `::test_traffic_can_be_excluded_from_activity`, `test_runtime_vllm.py::test_activity_probe_reads_vllm_metrics_on_loopback`; the watchdog's metrics-reading tests in `test_watchdog.py` |
| 15 | A stopped instance does not bill storage forever | `lifecycle.stopped_ttl` -> `PolicyConfig.stopped_ttl_s`; `RuntimeState.stopped_since`; `Engine.guard_tick` destroys with reason `stopped_ttl`; `validate` warns when nothing would remove it | `test_engine_sessions.py::test_guard_destroys_an_instance_stopped_longer_than_stopped_ttl`, `::test_validate_warns_when_nothing_removes_a_billing_stopped_instance`; the stopped-TTL policy tests in `test_policy.py` |
| 16 | Orphans are found account-wide, each project keeps its own rules, and nothing else is touched | `GPUProvider.list_instances` (capability `list_instances`) returns only `jambu-` labels; `gpu setup` registers the project in `~/.gpu/projects.json`; `core/sweep.py::guard_projects` runs each registered project's own `Engine.guard_tick`; only untracked instances are orphans: running past `--orphan-max-age` is stopped (never destroyed), stopped past `--stopped-ttl` (from the sweep's own first sighting in `~/.gpu/sweep.json`) is destroyed; `gpu sweep --install-timer` schedules one working-directory-independent unit and refuses a key that is only in the shell | `test_sweep.py::test_vast_lists_only_jambu_labelled_instances`, `::test_non_jambu_instances_are_never_touched`, `::test_another_projects_restarted_and_locked_instance_is_not_destroyed`, `::test_each_project_is_judged_by_its_own_max_lifetime`, `::test_a_running_orphan_past_orphan_max_age_is_stopped_never_destroyed`, `::test_a_stopped_orphan_is_destroyed_once_seen_stopped_past_stopped_ttl`, `::test_install_timer_writes_one_unit_independent_of_the_project`, `::test_install_timer_refuses_a_key_only_in_the_shell_environment` |
| 17 | The CLI guard never stops a serving endpoint it cannot see | `Engine._idle_blind`: with the watchdog enabled and installed but unreachable, `guard_tick` evaluates without `idle_timeout` and records `idle_skipped`; `max_lifetime`, budget and `stopped_ttl` still apply | `test_engine_sessions.py::test_guard_leaves_idle_to_an_unreachable_watchdog`, `::test_max_lifetime_still_applies_when_the_watchdog_is_unreachable` |
| 18 | A fresh stop or a short API outage never costs the disk | `RuntimeState.mark_stopped` restarts `stopped_since` on evidence the instance ran since; the watchdog escalates a failing stop to destroy only after `escalate_after_s` (30m) of continuous failure and resets on renewed use | `test_engine_sessions.py::test_a_stale_stopped_since_does_not_destroy_a_fresh_stop`, `test_watchdog.py::test_an_api_blip_does_not_destroy_the_disk`, `::test_failures_reset_when_the_instance_is_back_in_use` |

## Validated on real hardware

On 2026-10-24 the lifecycle ran end to end on Vast.ai (RTX 3090, Qwen2.5-0.5B, `idle_timeout: 4m`, `stopped_ttl: 5m`), for $0.35 of Vast credit in total across every attempt:

| What was checked | What happened |
| --- | --- |
| Traffic is activity | 10 requests sent straight to the endpoint over 7.5 minutes, none through `gpu run`. The machine stayed up past the 4-minute idle timeout. |
| Idle stops | 269 s after the last request the watchdog stopped its own machine (`idle_timeout`). Vast showed it `exited`. |
| A restart is guarded | `gpu run` restarted it. The watchdog logged `session_started` with `terminated: false` and measured `max_lifetime` from the restart. After the workload it stopped the machine again on its own. |
| Stopped storage is reclaimed | 334 s after the stop, `gpu guard` destroyed it (`stopped_ttl`). |
| Failed setups leave nothing | An image built for a newer CUDA, a model server that died at import, a host whose network dropped the weights download, a 30-minute image pull, and a Ctrl-C each ended with the instance destroyed. |
| Bad hosts are skipped | Hosts that could not pull the image were destroyed, excluded and replaced, with never two instances alive at once. |

The run also found problems the unit tests had missed. Each is fixed and covered by a test:
- the boot script was over Vast's 16 KB onstart limit;
- `latest` had moved to CUDA 13;
- the `-cu129` image variant could not load torchcodec;
- a 15-minute boot timeout was too short for a 12 GB pull;
- Ctrl-C did not clean up;
- a fresh contract was reported `READY`.

### Application integration (0.2.0)

Driven end to end by [`examples/typescript-client`](../examples/typescript-client/) through the OpenAI SDK, on an RTX 3090 with Qwen2.5-0.5B:

| Check | Result |
| --- | --- |
| `ensureGpu()` on an empty account | Two bad hosts (weights download cut by the host's network; image pull stalled 30 minutes) were destroyed and replaced; the third served. 37 minutes in all, one instance alive at a time. |
| The endpoint requires its key | No key: `401`. Wrong key: `401`. The key from `gpu --json setup`: `200`. `/health` without a key: `200`. |
| Chat through the SDK | Answered. |
| Healing after a stop | `stopGpu()`, then a call on the stale connection: it failed to connect, `gpu setup` restarted the same instance, and the retry answered 41 s later with the same key. |

It found two more bugs, both fixed and tested: Vast reports a stopping instance as `running` for a few seconds, which made a setup right after a stop skip the restart; and a Vast rate limit (`429`) failed setup instead of being waited out.

## A coding agent driving it (0.4.0)

On 2026-10-25 a headless Claude Code with no prior context (`claude -p`) was given a fresh directory holding only `gpu agent install`'s skill and `.mcp.json`, plus `VAST_API_KEY` in `.env`. It was asked in one sentence to test Qwen2.5-0.5B on a rented GPU, ask it the capital of Brazil, report time and cost, and destroy the machine. It loaded the `gpu` skill by itself and called `gpu_init` (with the cap and VRAM from the request), `gpu_validate`, `gpu_offers`, then `gpu_setup` until ready, `curl` for the question, `gpu_status`, `gpu_destroy` and `gpu_ls`. The model answered "Brasília", the account was empty at the end, and the agent reported setup time and cost correctly, noting that `session_cost_usd` leaves out the replaced host.

On 2026-11-02 the self-install was checked the same way. In an empty home directory with no `uv`, `gpu` or Python tool on `PATH`, a headless Claude Code got one sentence: "Set up GPU for Agents for me. I'm using Claude Code. Fetch …/install.md with curl and follow its steps." It ran `install.sh --agent claude`, which put `gpu` 0.6.0 in `~/.local/bin`, the skill in `~/.claude/skills/gpu4agents/` and the `gpu` server in the user MCP config. `gpu login --check` exited 5, and the agent told the user to create a key and run `gpu login` in their own terminal, without asking for the key. It read the skill with `gpu agent skill` and rented nothing. Separately, `install.sh` run with `GPU4AGENTS_PYTHON=3.13` on a machine without Python 3.13 had uv download CPython 3.13 and install the tool on it.

| What was checked | What happened |
| --- | --- |
| Setup longer than any tool timeout | `gpu_setup` returned `running` every 300 s for 42 minutes and re-attached to the same background job each time: one machine at a time, never a second setup. |
| Bad hosts | The first host could not pull the image for 30 minutes; setup destroyed it and rented another, which served. |
| Cancel | Over MCP (official Python SDK client), `gpu_destroy` during a setup interrupted it like Ctrl-C: the booting machine was destroyed and `gpu_ls` came back empty, 2.3 s after the call. |
| Protocol | The official MCP Python SDK (2.3.0) client initializes, lists the 13 tools and calls them. |

The first attempt found problems the unit tests had missed. Each is fixed and tested:
- during a setup, `gpu status`, `gpu logs`, `gpu stop` and `gpu destroy` failed on the project lock (reported as "no instance"). Now `status` and `logs` report the last saved state with `busy: true`, the MCP `gpu_stop` / `gpu_destroy` cancel the setup first, and a held lock is its own exit code, 12;
- the host that failed last when setup gave up was not remembered as bad, so the next setup rented it again;
- on Vast, three hosts in a row stalled 30 minutes each pulling the 12 GB Docker Hub image, and setup gave up after 90 minutes with nothing left billing. Vast now defaults to `vastai/vllm:v0.32.0-cuda-13.0`, the same vLLM that Vast's hosts cache: in the cancel test the container was running within a minute.

## The MVP command surface

`gpu.yaml` schema, CLI, provider contract, provider registry, Vast.ai adapter,
vLLM runtime, `setup`, `run`, `status`, `stop`, `destroy`, `lock`/`unlock`, idle
timeout, max lifetime, remote watchdog and structured logs are all present. On top
of that, `ls` and `sweep` handle orphan control across the account.

Deliberately not implemented: multi-instance scheduling, multi-GPU
orchestration, Kubernetes, distributed inference, provider optimization, automatic
provider selection, job queues, web dashboard.
