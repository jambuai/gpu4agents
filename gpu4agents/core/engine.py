"""Orchestration engine.

Owns the flows in spec sections 15-18 and 23: provisioning, run, status,
reconciliation, idempotency and cleanup. It knows about the provider
*contract* and about capabilities - never about a specific provider.
"""

from __future__ import annotations

import re
import secrets
import time
from typing import Any, Callable, Optional, Sequence

from .. import watchdog as watchdog_pkg
from ..providers.base import GPUProvider
from ..providers.registry import load_builtin_providers
from ..runtime import runtime_registry
from ..runtime.base import ENDPOINT_NAME, ModelRuntime, image_tag, max_cuda
from ..watchdog.builder import build_agent_config, build_agent_source, build_onstart_script
from . import events as ev
from . import policy
from .beacon import WatchdogClient, WatchdogUnavailable
from .config import RuntimeConfig
from .credentials import CredentialResolver
from .durations import format_duration
from .errors import (
    BudgetExceededError,
    ConfigError,
    HostFailure,
    JambuError,
    LockBusy,
    ProviderError,
    RuntimeStartupError,
    StateError,
    UnsupportedCapabilityError,
    WorkloadFailed,
)
from .events import EventLog
from .execution import ExecutionManager, Heartbeat
from .health import wait_for_healthy
from .models import (
    ComputeSpec,
    GpuSpec,
    Instance,
    InstanceState,
    JAMBU_LABEL_PREFIX,
    ValidationResult,
    Workload,
)
from .hosts import BadHosts
from .state import RuntimeState, StateStore, WatchdogInfo, utc_now

Echo = Callable[[str], None]

# Signatures, in a dead model server's log, of a network that failed it - not
# of a model or config that would fail on any machine.
_HOST_NETWORK_FAILURE = re.compile(
    r"RemoteProtocolError|ConnectError|ConnectTimeout|ReadTimeout|"
    r"Connection (?:reset|refused|aborted)|TLS handshake timeout|"
    r"Temporary failure in name resolution|Network is unreachable"
)

# A pull stuck retrying a layer: its countdown changes, but it is no progress.
_PULL_RETRY = re.compile(r"\bretrying\b", re.IGNORECASE)

# Where `gpu logs --source` reads from: the watchdog serves the instance files.
LOG_SOURCES = ("runtime", "onstart", "watchdog", "provider")
# Runtime log lines quoted in the error when the model server dies at boot.
STARTUP_FAILURE_LOG_LINES = 30
# How long read-only commands wait for a busy lock before reporting the last
# saved state instead: a setup holds it for the whole boot, often 10+ minutes.
READ_LOCK_WAIT_S = 3.0


def _noop(_message: str) -> None:
    return None


def _pull_progress(message: Optional[str]) -> Optional[str]:
    """The provider's boot message when it shows the pull moving, else None.

    Vast reports the image pull's latest line (``<layer>: Download complete``,
    ``<layer>: Pull complete``). A new line means a layer moved; the same line,
    no line, or a retry countdown means it did not.
    """
    text = (message or "").strip()
    if not text or _PULL_RETRY.search(text):
        return None
    return text


class Engine:
    def __init__(
        self,
        config: RuntimeConfig,
        provider: Optional[GPUProvider] = None,
        echo: Optional[Echo] = None,
        credentials: Optional[CredentialResolver] = None,
    ) -> None:
        self.config = config
        self.echo = echo or _noop
        self.credentials = credentials or CredentialResolver(
            config.project_dir, profile=config.provider.profile
        )
        registry = load_builtin_providers()
        self.provider = provider or registry.get(config, self.credentials)
        self.runtime: ModelRuntime = runtime_registry.get(config)
        self.runtime.provider_image = self.provider.preferred_images.get(self.runtime.engine)
        self.store = StateStore(config.state_dir)
        self.events = EventLog(self.store.logs_dir).bind(provider=self.provider.name)
        # Whether the last reconcile read the watchdog's state. Traffic activity
        # lives only on the instance, so without it the local idle clock lies.
        self.watchdog_read = False

    # ------------------------------------------------------------------
    # policy plumbing
    # ------------------------------------------------------------------

    def policy_config(self, hourly_cost_usd: Optional[float] = None) -> policy.PolicyConfig:
        caps = self.provider.capabilities
        cfg = self.config
        return policy.PolicyConfig(
            idle_timeout_s=cfg.idle_timeout_s,
            max_lifetime_s=cfg.max_lifetime_s,
            heartbeat_grace_s=cfg.heartbeat_grace_s,
            auto_stop=cfg.lifecycle.auto_stop,
            auto_destroy=cfg.lifecycle.auto_destroy,
            allow_indefinite_lock=cfg.lifecycle.allow_indefinite_lock,
            stop_supported=caps.stop,
            destroy_supported=caps.destroy,
            stop_eliminates_billing=caps.stop_eliminates_billing,
            hourly_cost_usd=hourly_cost_usd,
            max_session_cost_usd=cfg.budget.max_session_cost_usd,
            stopped_ttl_s=cfg.stopped_ttl_s,
        )

    def snapshot(self, state: RuntimeState) -> policy.Snapshot:
        now = utc_now()
        return policy.Snapshot(
            now=now,
            created_at=state.created_at or now,
            last_activity=state.last_activity or state.created_at or now,
            workloads=[w.to_policy() for w in state.workloads.values()],
            starting=state.starting,
            locked=state.locked,
            locked_until=state.locked_until,
            provider_state=state.state,
            session_started_at=state.session_started_at,
            stopped_since=state.stopped_since,
        )

    def _idle_blind(self, state: RuntimeState) -> bool:
        """True when only the unreachable watchdog knows about recent activity.

        Inference traffic is counted on the instance; the local last_activity
        never sees it. Judging idleness from it would stop a serving endpoint,
        so the CLI leaves the idle rule to the watchdog. max_lifetime, the
        budget and stopped_ttl still apply.
        """
        return (
            self.config.lifecycle.watchdog.enabled
            and state.watchdog.installed
            and not self.watchdog_read
            # A stopped machine has no watchdog to reach and no idle to judge.
            and InstanceState(state.state).is_billable_compute
        )

    def beacon(self, state: RuntimeState) -> Optional[WatchdogClient]:
        info = state.watchdog
        if not info.installed or not info.url:
            return None
        return WatchdogClient(info.url, info.token)

    # ------------------------------------------------------------------
    # validate (spec section 15, steps 1-5)
    # ------------------------------------------------------------------

    def validate(self, check_provider: bool = True) -> ValidationResult:
        result = ValidationResult()
        cfg = self.config

        if cfg.lifecycle.idle_timeout is None and cfg.lifecycle.max_lifetime is None:
            result.error(
                "lifecycle needs at least one of idle_timeout or max_lifetime; "
                "without a bound an abandoned GPU bills forever",
                "lifecycle",
            )
        if not cfg.lifecycle.auto_stop and not cfg.lifecycle.auto_destroy:
            result.warn(
                "auto_stop and auto_destroy are both false: the guard will report "
                "but never shut anything down",
                "lifecycle",
            )
        if not cfg.lifecycle.watchdog.enabled:
            result.warn(
                "lifecycle.watchdog.enabled is false: shutdown depends on this CLI "
                "staying alive (spec section 14)",
                "lifecycle.watchdog",
            )
        if cfg.lifecycle.allow_indefinite_lock:
            result.warn(
                "allow_indefinite_lock is true: a forgotten lock defeats cost protection",
                "lifecycle.allow_indefinite_lock",
            )
        if cfg.budget.max_hourly_cost_usd is None:
            result.warn("no budget.max_hourly_cost_usd set", "budget")
        if (
            not self.provider.capabilities.stop_eliminates_billing
            and cfg.stopped_ttl_s is None
            and not cfg.lifecycle.auto_destroy
        ):
            result.warn(
                f"a stopped {self.provider.display_name} instance still bills storage, and "
                "with lifecycle.stopped_ttl disabled and auto_destroy false nothing ever "
                "removes it; set lifecycle.stopped_ttl (e.g. 24h)",
                "lifecycle.stopped_ttl",
            )

        result.merge(self.runtime.validate())
        if cfg.runtime.engine != "none" and image_tag(self.runtime.image()) == "latest":
            result.warn(
                f"unpinned image '{self.runtime.image()}': the CUDA it needs can change "
                "under you (pin a tag such as "
                f"'{type(self.runtime).default_image}')",
                "compute.image",
            )

        if cfg.model.hf_token_env and not self.credentials.get(cfg.model.hf_token_env):
            result.warn(
                f"{cfg.model.hf_token_env} is not set; gated models will fail to download",
                "model.hf_token_env",
            )

        if check_provider:
            try:
                # The provider previews offers from config alone; hand it the
                # CUDA floor the runtime image adds, so validate sees what setup will.
                checked = cfg
                if self.cuda_min() != cfg.compute.gpu.cuda_min:
                    checked = cfg.model_copy(deep=True)
                    checked.compute.gpu.cuda_min = self.cuda_min()
                result.merge(self.provider.validate(checked))
            except JambuError as exc:
                result.error(str(exc), "provider")
        return result

    # ------------------------------------------------------------------
    # reconciliation (spec section 18)
    # ------------------------------------------------------------------

    def reconcile(self, state: Optional[RuntimeState] = None) -> RuntimeState:
        """Merge local state with provider truth. Repairs local state in place."""
        owns_lock = state is None
        if owns_lock:
            with self.store.process_lock():
                state = self.store.load()
                reconciled = self._reconcile(state)
                self.store.save(reconciled)
                return reconciled
        return self._reconcile(state)

    def _reconcile(self, state: RuntimeState) -> RuntimeState:
        self.watchdog_read = False
        if not state.instance_id:
            state.state = InstanceState.DESTROYED.value
            return state

        instance = state.instance or Instance(id=state.instance_id, provider=self.provider.name)
        try:
            status = self.provider.status(instance)
        except ProviderError as exc:
            self.events.emit(ev.PROVIDER_ERROR, operation="status", error=str(exc))
            return state  # keep local state; the API blip is not authoritative

        previous = state.state
        if not status.exists:
            state.clear_instance(InstanceState.DESTROYED.value)
            self.events.emit(
                ev.STATE_RECONCILED,
                previous_state=previous,
                state=InstanceState.DESTROYED.value,
                reason="instance missing at provider",
            )
            return state

        current = status.instance or instance
        state.instance = current
        state.instance_id = current.id
        state.hourly_cost_usd = status.hourly_cost_usd or state.hourly_cost_usd
        if current.created_at:
            state.created_at = current.created_at
        endpoint = current.endpoint_url(ENDPOINT_NAME)
        if endpoint:
            state.endpoint = endpoint

        watchdog_url = current.endpoint_url("watchdog")
        if watchdog_url and state.watchdog.token:
            state.watchdog.url = watchdog_url
            state.watchdog.installed = True

        # The watchdog, when reachable, is the authority on activity.
        remote = None
        remote_terminated = False
        client = self.beacon(state)
        if client is not None:
            remote = client.try_state()
        if remote:
            self.watchdog_read = True
            state.watchdog.last_contact = utc_now()
            state.watchdog.last_error = None
            state.locked = bool(remote.get("locked"))
            state.locked_until = remote.get("locked_until")
            state.starting = bool(remote.get("starting"))
            if remote.get("last_activity"):
                state.last_activity = max(
                    state.last_activity or 0.0, float(remote["last_activity"])
                )
            # `terminated` is the watchdog's memory of having shut the machine
            # down. It survives a restart, so it is only a hint for when the
            # provider itself cannot say whether the instance runs.
            remote_terminated = bool(remote.get("terminated"))
            pulled = client.events(limit=200)
            if pulled:
                self.events.extend(pulled)
        elif state.watchdog.installed:
            state.watchdog.last_error = "unreachable"

        if status.state in (InstanceState.READY, InstanceState.BUSY, InstanceState.IDLE):
            # Provider truth first: a previous local STOPPED must not leak
            # into the derivation after the instance was restarted.
            state.state = status.state.value
            derived = policy.derive_state(self.snapshot(state), self.policy_config())
            state.state = derived if derived != InstanceState.UNKNOWN.value else status.state.value
        else:
            if status.state is InstanceState.UNKNOWN and remote_terminated:
                state.state = InstanceState.STOPPED.value
            else:
                state.state = status.state.value
            if status.state in (InstanceState.STOPPED, InstanceState.FAILED):
                for record in state.workloads.values():
                    if record.state in ("running", "queued"):
                        record.state = "failed"
                        record.finished_at = utc_now()

        # Track actual billable time, not wall-clock time since creation - a
        # `status` checked long after auto-stop must not keep inflating cost.
        billable = InstanceState(state.state).is_billable_compute
        state.mark_billable(billable)
        if state.state == InstanceState.STOPPED.value:
            state.mark_stopped(ran_at=[current.created_at or None])
        elif billable and state.stopped_since is not None:
            # Restarted outside this CLI (web console, another machine): a new
            # session begins now.
            state.mark_session_start()
        elif billable and state.session_started_at is None:
            # A state file from before sessions existed: never grant it a
            # fresh max_lifetime, count from creation.
            state.session_started_at = state.created_at
        if state.hourly_cost_usd:
            state.session_cost_usd = round(
                state.hourly_cost_usd * state.accrued_billable_seconds() / 3600.0, 4
            )

        if previous != state.state:
            self.events.emit(
                ev.STATE_RECONCILED, previous_state=previous, state=state.state
            )
        return state

    # ------------------------------------------------------------------
    # setup (spec section 15)
    # ------------------------------------------------------------------

    def build_spec(self, onstart: str = "") -> ComputeSpec:
        cfg = self.config
        env = dict(self.runtime.env())
        hf_token = (
            self.credentials.get(cfg.model.hf_token_env) if cfg.model.hf_token_env else None
        )
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        return ComputeSpec(
            gpu=GpuSpec(
                min_vram_gb=cfg.compute.gpu.min_vram_gb,
                count=cfg.compute.gpu.count,
                name_filter=cfg.compute.gpu.name_filter,
                cuda_min=self.cuda_min(),
                compute_cap_min=self.compute_cap_min(),
            ),
            disk_gb=cfg.compute.disk_gb,
            image=self.runtime.image(),
            ports=self._ports(),
            env=env,
            onstart=onstart,
            label=f"{JAMBU_LABEL_PREFIX}{cfg.model.id.split('/')[-1][:24]}-{cfg.fingerprint()[:6]}",
            spot=cfg.compute.spot,
            region=cfg.compute.region,
            max_hourly_cost_usd=cfg.budget.max_hourly_cost_usd,
            provider_options=dict(cfg.provider.options),
        )

    def cuda_min(self) -> Optional[str]:
        """The stricter of compute.gpu.cuda_min and what the runtime image needs."""
        configured = self.config.compute.gpu.cuda_min
        needed = max_cuda(configured, self.runtime.required_cuda())
        if needed is None:
            return configured
        if configured is not None and max_cuda(configured) == needed:
            return configured
        return f"{needed:.1f}"

    def compute_cap_min(self) -> Optional[float]:
        """The stricter of compute.gpu.compute_cap_min and the runtime's floor."""
        candidates = [
            c
            for c in (
                self.config.compute.gpu.compute_cap_min,
                self.runtime.required_compute_cap(),
            )
            if c is not None
        ]
        return max(candidates) if candidates else None

    def _ports(self) -> list[int]:
        ports = list(self.runtime.ports())
        if self.config.lifecycle.watchdog.enabled:
            ports.append(self.config.lifecycle.watchdog.port)
        return ports

    def _api_key(self) -> Optional[str]:
        """The key the model server will demand: the configured one, or a fresh one.

        The endpoint is public, and traffic counts as activity: an open one
        would let anyone use the GPU and keep it billing.
        """
        if self.config.runtime.api_key:
            return self.config.runtime.api_key
        if self.runtime.api_key_env is None:
            return None
        return secrets.token_urlsafe(32)

    def _build_onstart(self, token: str, api_key: Optional[str] = None) -> str:
        cfg = self.config
        enabled = cfg.lifecycle.watchdog.enabled
        spec = self.provider.watchdog_spec(None) if enabled else None
        if enabled and spec is None:
            raise UnsupportedCapabilityError(
                f"provider '{self.provider.name}' supplies no watchdog actions; "
                "set lifecycle.watchdog.enabled: false to proceed without remote "
                "lifecycle enforcement"
            )
        agent_source = build_agent_source(spec.actions_source if spec else None)
        probe = (
            self.runtime.activity_probe() if cfg.lifecycle.traffic_counts_as_activity else None
        )
        agent_config = build_agent_config(
            provider=self.provider.name,
            instance_id="",  # resolved by /identify or the provider's own env
            token=token,
            port=cfg.lifecycle.watchdog.port,
            interval_s=cfg.watchdog_interval_s,
            # The watchdog starts with the container, but the CLI's waits span
            # the whole boot: never let the remote guard cut a legitimate one.
            startup_grace_s=cfg.boot_timeout_s + cfg.startup_timeout_s + 300.0,
            policy=self.policy_config(),
            created_at=utc_now(),
            watchdog_spec=spec,
            activity_probe=probe,
            runtime_pid_path=self.runtime.runtime_pid_path(),
            runtime_log_path=self.runtime.runtime_log_path(),
        )
        runtime_env = dict(self.runtime.env())
        # A configured key is already on the command line; only a generated
        # one travels through the environment.
        if api_key and api_key != cfg.runtime.api_key and self.runtime.api_key_env:
            runtime_env[self.runtime.api_key_env] = api_key
        return build_onstart_script(
            agent_source=agent_source,
            agent_config=agent_config,
            runtime_start_command=self.runtime.start_command(),
            runtime_env=runtime_env,
            watchdog_enabled=enabled,
        )

    def _check_budget(self, spec: ComputeSpec, allow_override: bool) -> Optional[float]:
        ceiling = self.config.budget.max_hourly_cost_usd
        if not self.provider.capabilities.pricing_query:
            if ceiling and not allow_override:
                self.echo(
                    "! provider cannot query pricing; budget.max_hourly_cost_usd "
                    "cannot be enforced before provisioning"
                )
            return None
        offers = self.provider.find_offers(spec, limit=5)
        if not offers:
            return None
        cheapest = offers[0]
        if ceiling and cheapest.hourly_cost_usd > float(ceiling):
            if not allow_override:
                self.events.emit(
                    ev.BUDGET_REJECTED,
                    cheapest_hourly_cost_usd=cheapest.hourly_cost_usd,
                    max_hourly_cost_usd=float(ceiling),
                )
                raise BudgetExceededError(
                    f"cheapest matching offer is ${cheapest.hourly_cost_usd:.3f}/h, above "
                    f"budget.max_hourly_cost_usd (${float(ceiling):.3f}/h). "
                    "Re-run with --allow-cost-override to proceed anyway."
                )
            self.echo(
                f"! cost override: ${cheapest.hourly_cost_usd:.3f}/h exceeds the "
                f"${float(ceiling):.3f}/h budget"
            )
        return cheapest.hourly_cost_usd

    def setup(
        self,
        *,
        recreate: bool = False,
        allow_cost_override: bool = False,
        wait: bool = True,
    ) -> RuntimeState:
        cfg = self.config
        fingerprint = cfg.fingerprint()

        with self.store.process_lock():
            state = self._reconcile(self.store.load())

            # -- idempotency (spec section 23) -----------------------------
            if state.has_instance():
                if state.fingerprint != fingerprint and not recreate:
                    raise ConfigError(
                        f"instance {state.instance_id} was provisioned for a different "
                        "runtime (jambu.yaml changed since setup). Re-run with "
                        "`--recreate` to replace it, or `gpu destroy` first."
                    )
                if not recreate:
                    self.echo(f"= reusing instance {state.instance_id} ({state.state})")
                    self.store.save(state)
                    return self.ensure_ready(state=state) if wait else state
                self.echo(f"~ replacing instance {state.instance_id}")
                self._terminate(state, action=policy.ACTION_DESTROY, reason="recreate")

            # -- provision ------------------------------------------------
            token = watchdog_pkg.generate_token()
            api_key = self._api_key()
            onstart = self._build_onstart(token, api_key)

            self.events.emit(
                ev.PROVISION_REQUESTED,
                model=cfg.model.id,
                engine=cfg.runtime.engine,
                gpu_count=cfg.compute.gpu.count,
                min_vram_gb=cfg.compute.gpu.min_vram_gb,
                fingerprint=fingerprint,
            )
            self.echo(
                f"> searching {self.provider.display_name} for {cfg.compute.gpu.count}x "
                f">={cfg.compute.gpu.min_vram_gb}GB GPU, {cfg.compute.disk_gb}GB disk"
            )

            # On a GPU marketplace a bad host is normal: one that cannot pull
            # the image, or reach the registry at all, costs one attempt and
            # its machine is not rented again by this setup. Retrying means
            # destroying the bad one first, so it needs cleanup_on_setup_failure.
            attempts = cfg.compute.host_attempts if cfg.lifecycle.cleanup_on_setup_failure else 1
            bad_hosts = BadHosts()
            excluded: list[str] = bad_hosts.recent(self.provider.name)
            for attempt in range(1, attempts + 1):
                spec = self.build_spec(onstart)
                spec.exclude_hosts = list(excluded)
                # Every new offer is held to the budget, not just the first.
                estimated = self._check_budget(spec, allow_cost_override)
                if estimated:
                    self.echo(f"  cheapest match: ${estimated:.3f}/h")
                try:
                    return self._provision(state, spec, token, fingerprint, wait, api_key)
                except HostFailure as exc:
                    retry = attempt < attempts
                    self._host_failed(state, exc, attempt, attempts, retry)
                    if retry and self._discard_host(state, exc, excluded, bad_hosts):
                        continue
                    if exc.instance is not None and exc.instance.host_id:
                        # The last host failed as surely as the others: the
                        # next setup must not walk straight back onto it.
                        bad_hosts.mark(self.provider.name, exc.instance.host_id, exc.reason)
                    if retry or attempts == 1:
                        self._fail_setup(state, exc)
                        raise
                    final = HostFailure(
                        f"{attempts} machines in a row failed to boot, giving up "
                        f"(compute.host_attempts). Last: {exc}",
                        reason=exc.reason,
                        instance=exc.instance,
                    )
                    self._fail_setup(state, final)
                    raise final from exc
                except (JambuError, KeyboardInterrupt) as exc:
                    # Ctrl-C included: nobody is left to use the machine this
                    # setup was renting, so it must not stay booting on the bill.
                    self._fail_setup(state, exc)
                    raise
            raise AssertionError("unreachable: the last attempt returns or raises")

    def _provision(
        self,
        state: RuntimeState,
        spec: ComputeSpec,
        token: str,
        fingerprint: str,
        wait: bool,
        api_key: Optional[str] = None,
    ) -> RuntimeState:
        cfg = self.config
        instance = self.provider.setup(spec)
        self.echo(
            f"+ instance {instance.id} created "
            f"({instance.gpu_count or cfg.compute.gpu.count}x "
            f"{instance.gpu_name or 'GPU'}"
            + (
                f", ${instance.hourly_cost_usd:.3f}/h)"
                if instance.hourly_cost_usd
                else ")"
            )
        )

        state.provider = self.provider.name
        state.instance_id = instance.id
        state.instance = instance
        state.fingerprint = fingerprint
        state.model_id = cfg.model.id
        state.api_key = api_key
        state.created_at = instance.created_at or utc_now()
        state.last_activity = utc_now()
        state.starting = True
        state.state = InstanceState.PROVISIONING.value
        state.hourly_cost_usd = instance.hourly_cost_usd
        state.locked = False
        state.locked_until = None
        # Billing starts the moment the provider creates the instance,
        # not at the next reconcile - start the clock now so a long
        # provisioning/startup wait is never undercounted.
        state.billed_seconds = 0.0
        state.running_since = state.created_at
        state.mark_session_start(state.created_at)
        state.watchdog = WatchdogInfo(
            installed=cfg.lifecycle.watchdog.enabled,
            token=token,
            port=cfg.lifecycle.watchdog.port,
            url=instance.endpoint_url("watchdog"),
        )
        # Persist before the long wait: a crash here must still leave a
        # destroyable record (spec criterion 4).
        self.store.save(state)
        self.events.emit(
            ev.INSTANCE_CREATED,
            instance_id=instance.id,
            host_id=instance.host_id,
            gpu_name=instance.gpu_name,
            gpu_count=instance.gpu_count,
            hourly_cost_usd=instance.hourly_cost_usd,
        )

        if not wait:
            return state
        return self.ensure_ready(state=state)

    def _host_failed(
        self,
        state: RuntimeState,
        exc: HostFailure,
        attempt: int,
        attempts: int,
        retry: bool,
    ) -> None:
        instance = exc.instance
        host_id = instance.host_id if instance else None
        instance_id = instance.id if instance else state.instance_id
        self.events.emit(
            ev.HOST_FAILED,
            instance_id=instance_id,
            host_id=host_id,
            reason=exc.reason,
            error=str(exc),
            attempt=attempt,
            host_attempts=attempts,
        )
        line = f"! host {host_id or instance_id} failed ({exc.reason})"
        if retry:
            line += f"; trying another machine ({attempt + 1}/{attempts})"
        self.echo(line)

    def _discard_host(
        self,
        state: RuntimeState,
        exc: HostFailure,
        excluded: list[str],
        bad_hosts: Optional["BadHosts"] = None,
    ) -> bool:
        """Destroy a machine that failed to boot so another can be rented.

        False when it could not be destroyed: renting another would leave two
        instances billing, so setup stops there.
        """
        instance = exc.instance
        try:
            if state.instance_id:
                self._terminate(state, action=policy.ACTION_DESTROY, reason="host_failure")
            elif instance is not None:
                # Vanished: already out of the state. Destroying is idempotent
                # and makes sure it is really gone before renting another.
                self.provider.destroy(instance)
        except JambuError as cleanup_exc:
            self.echo(f"! could not destroy the failed instance: {cleanup_exc}")
            return False
        self.store.save(state)
        host_id = instance.host_id if instance else None
        if host_id:
            excluded.append(host_id)
            if bad_hosts is not None:
                bad_hosts.mark(self.provider.name, host_id, exc.reason)
        return True

    def _fail_setup(self, state: RuntimeState, exc: BaseException) -> None:
        cfg = self.config
        self.events.emit(ev.SETUP_FAILED, error=str(exc), error_type=type(exc).__name__)
        if cfg.lifecycle.cleanup_on_setup_failure and state.instance_id:
            self.echo(f"! setup failed, cleaning up instance {state.instance_id}")
            try:
                self._terminate(state, action=policy.ACTION_DESTROY, reason="setup_failure")
                self.events.emit(ev.CLEANUP_PERFORMED, instance_id=state.instance_id)
            except JambuError as cleanup_exc:
                self.events.emit(
                    ev.PROVIDER_ERROR, operation="cleanup", error=str(cleanup_exc)
                )
                self.echo(f"! cleanup failed: {cleanup_exc}")
        self.store.save(state)

    # ------------------------------------------------------------------
    # readiness
    # ------------------------------------------------------------------

    def ensure_ready(self, state: Optional[RuntimeState] = None) -> RuntimeState:
        """Guarantee the configured runtime is available (spec section 16)."""
        with self.store.process_lock():
            state = state if state is not None else self._reconcile(self.store.load())

            if not state.has_instance():
                self.store.save(state)
                return self.setup()

            instance = state.instance
            if instance is None:
                raise StateError("state has an instance id but no instance record")

            if state.state == InstanceState.STOPPED.value:
                if not self.provider.capabilities.stop:
                    raise UnsupportedCapabilityError(
                        f"provider '{self.provider.name}' cannot restart instances"
                    )
                self.echo(f"> starting instance {instance.id}")
                instance = self.provider.start(instance)
                state.instance = instance
                state.starting = True
                state.mark_billable(True)  # billing resumes now, not at the next reconcile
                state.mark_session_start()  # max_lifetime and budget restart with it
                self.events.emit(ev.INSTANCE_STARTED, instance_id=instance.id)

            instance = self._wait_running(state)
            state.instance = instance
            state.endpoint = instance.endpoint_url(ENDPOINT_NAME)
            watchdog_url = instance.endpoint_url("watchdog")
            if watchdog_url:
                state.watchdog.url = watchdog_url
            self.store.save(state)

            self._connect_watchdog(state)

            if self.config.runtime.engine != "none":
                self._wait_runtime(state)

            state.starting = False
            state.state = InstanceState.READY.value
            state.touch()
            self.store.save(state)
            return state

    def _wait_running(self, state: RuntimeState) -> Instance:
        """Wait for the machine itself (mostly the image pull), up to
        health.boot_timeout. The container has not run yet, so every failure
        here is the host's.
        """
        instance = state.instance
        assert instance is not None
        boot_timeout = self.config.boot_timeout_s
        stall_timeout = self.config.pull_stall_timeout_s
        deadline = time.time() + boot_timeout
        announced = False
        # Live, healthy pulls printed a new line every 1-8 minutes; stalled ones
        # froze (or retried one layer) until boot_timeout, 30 minutes per host.
        progress: Optional[str] = None
        progress_at = time.time()
        while time.time() < deadline:
            status = self.provider.status(instance)
            if not status.exists:
                state.clear_instance()
                self.store.save(state)
                raise HostFailure(
                    f"instance {instance.id} vanished at the provider while starting",
                    reason="vanished while booting",
                    instance=instance,
                )
            current = status.instance or instance
            if status.state is InstanceState.FAILED:
                detail = (status.message or "").strip() or "no detail"
                raise HostFailure(
                    f"instance {instance.id} failed: {detail}",
                    reason=detail.splitlines()[0][:160],
                    instance=current,
                )
            ready_ports = self._ports()
            have_ports = all(
                current.endpoints.get(name) is not None
                for name in self._endpoint_names()
            )
            if status.state in (
                InstanceState.READY,
                InstanceState.BUSY,
                InstanceState.IDLE,
            ) and (have_ports or not ready_ports):
                return current
            moved = _pull_progress(status.message)
            if moved is not None and moved != progress:
                progress, progress_at = moved, time.time()
            elif stall_timeout and time.time() - progress_at >= stall_timeout:
                last = (status.message or "").strip().splitlines()
                raise HostFailure(
                    f"instance {instance.id}'s image pull made no progress for "
                    f"{format_duration(stall_timeout)} (health.pull_stall_timeout)"
                    + (f"; last: {last[-1][:160]}" if last else ""),
                    reason=f"image pull stalled for {format_duration(stall_timeout)}",
                    instance=current,
                )
            if not announced:
                announced = True
                self.echo("  waiting for the machine to come up ...")
            time.sleep(6)
        raise HostFailure(
            f"instance {instance.id} was not running after "
            f"{format_duration(boot_timeout)} (health.boot_timeout)",
            reason=f"image pull timed out after {format_duration(boot_timeout)}",
            instance=instance,
        )

    def _endpoint_names(self) -> list[str]:
        names = []
        if self.config.runtime.engine != "none":
            names.append(ENDPOINT_NAME)
        if self.config.lifecycle.watchdog.enabled:
            names.append("watchdog")
        return names

    def _connect_watchdog(self, state: RuntimeState) -> None:
        if not self.config.lifecycle.watchdog.enabled:
            return
        client = self.beacon(state)
        if client is None:
            state.watchdog.last_error = "no watchdog endpoint published"
            self.echo("! watchdog port was not published; lifecycle guard is local-only")
            return

        deadline = time.time() + 90
        while time.time() < deadline:
            if client.ping():
                client.identify(state.instance_id or "", state.hourly_cost_usd)
                client.mark_starting()
                state.watchdog.installed = True
                state.watchdog.last_contact = utc_now()
                state.watchdog.last_error = None
                self.echo(f"+ lifecycle watchdog live at {state.watchdog.url}")
                self.store.save(state)
                return
            time.sleep(5)

        state.watchdog.last_error = "did not answer within 90s"
        self.echo(
            "! the remote watchdog never answered. The instance is NOT protected "
            "by an independent guard - check `gpu logs`, and run "
            "`gpu destroy` if this is unexpected."
        )

    def _wait_runtime(self, state: RuntimeState) -> None:
        endpoint = state.endpoint
        if not endpoint:
            raise RuntimeStartupError(
                "the model port was never published by the provider; cannot reach the runtime"
            )
        url = self.runtime.health_url(endpoint)
        self.events.emit(
            ev.RUNTIME_STARTING, endpoint=endpoint, engine=self.config.runtime.engine
        )
        self.echo(f"> waiting for {self.runtime.describe()} at {endpoint}")
        started = time.time()

        def _progress(attempt: int, remaining: float, detail: str) -> None:
            if attempt % 4 == 0:
                self.echo(
                    f"  still starting ({int(time.time() - started)}s elapsed, "
                    f"{int(remaining)}s left, last probe: {detail})"
                )

        def _abort() -> Optional[str]:
            instance = state.instance
            if instance is None:
                return None
            try:
                status = self.provider.status(instance)
            except ProviderError:
                return None
            if not status.exists:
                return "the instance disappeared at the provider"
            if status.state is InstanceState.FAILED:
                return status.message or "the instance entered a failed state"
            if status.state is InstanceState.STOPPED:
                return "the instance was stopped"
            reason = self._runtime_exited(state)
            if reason and _HOST_NETWORK_FAILURE.search(reason):
                # The model server died fetching weights over a broken network:
                # the machine's fault, so another host gets a chance.
                raise HostFailure(
                    f"model server did not start: {reason}",
                    reason="model download failed on this host's network",
                    instance=state.instance,
                )
            return reason

        wait_for_healthy(
            url,
            timeout=self.config.startup_timeout_s,
            interval=self.config.health_interval_s,
            on_attempt=_progress,
            should_abort=_abort,
        )
        client = self.beacon(state)
        if client is not None:
            client.mark_ready()
        self.events.emit(
            ev.RUNTIME_READY,
            endpoint=endpoint,
            startup_seconds=round(time.time() - started, 1),
        )
        self.echo(f"+ runtime ready: {endpoint}")

    def _runtime_exited(self, state: RuntimeState) -> Optional[str]:
        """The watchdog saw the model server process die: say why, from its log."""
        client = self.beacon(state)
        if client is None:
            return None
        snapshot = client.try_state()
        if not snapshot or snapshot.get("runtime_alive") is not False:
            return None
        reason = "the model server process exited on the instance"
        try:
            tail = client.logs("runtime", STARTUP_FAILURE_LOG_LINES)
        except WatchdogUnavailable:
            tail = []
        if not tail:
            return reason + " (its log is empty or unreadable)"
        return reason + ". Last lines of its log:\n" + "\n".join(
            "  " + line for line in tail
        )

    # ------------------------------------------------------------------
    # run (spec section 16)
    # ------------------------------------------------------------------

    def api_key_for(self, state: RuntimeState) -> str:
        # An instance from before generated keys runs without one; vLLM then
        # accepts any value, and OpenAI clients refuse an empty one.
        return state.api_key or self.config.runtime.api_key or "jambu-local"

    def served_model(self) -> str:
        """The name clients pass as `model` in OpenAI requests."""
        return self.config.model.served_name or self.config.model.id

    def connection(self, state: RuntimeState) -> dict[str, Any]:
        """How an application reaches the model: the stable integration contract."""
        endpoint = state.endpoint
        return {
            "instance_id": state.instance_id,
            "state": state.state if state.instance_id else "none",
            "endpoint": endpoint,
            "base_url": f"{endpoint.rstrip('/')}/v1" if endpoint else None,
            "api_key": self.api_key_for(state) if endpoint else None,
            "model": self.served_model(),
            "hourly_cost_usd": state.hourly_cost_usd,
        }

    def workload_env(self, state: RuntimeState, workload_id: str) -> dict[str, str]:
        endpoint = state.endpoint or ""
        env: dict[str, str] = {
            "JAMBU_PROVIDER": self.provider.name,
            "JAMBU_INSTANCE_ID": state.instance_id or "",
            "JAMBU_WORKLOAD_ID": workload_id,
            "JAMBU_MODEL_ID": self.served_model(),
        }
        if endpoint:
            env["JAMBU_GPU_ENDPOINT"] = endpoint
            env["OPENAI_BASE_URL"] = f"{endpoint.rstrip('/')}/v1"
            env["OPENAI_API_KEY"] = self.api_key_for(state)
        if state.instance and state.instance.ssh_host:
            env["JAMBU_SSH_HOST"] = state.instance.ssh_host
            env["JAMBU_SSH_PORT"] = str(state.instance.ssh_port or 22)
        env.update({k: str(v) for k, v in self.config.workload.env.items()})
        return env

    def run(self, command: Sequence[str], label: str = "") -> int:
        if not command:
            raise ConfigError("nothing to run: pass a command, e.g. `gpu run python x.py`")

        state = self.ensure_ready()
        manager = ExecutionManager(
            self.store,
            self.events,
            beacon=self.beacon(state),
            heartbeat_interval=self.config.heartbeat_interval_s,
        )
        record = manager.register(command, label=label)
        workload = Workload(
            id=record.id,
            command=list(command),
            label=record.label,
            cwd=self.config.workload.workdir or str(self.config.project_dir),
            env=self.workload_env(state, record.id),
            mode=self.config.workload.mode,
        )

        returncode: Optional[int] = None
        try:
            with Heartbeat(
                record.id,
                self.config.heartbeat_interval_s,
                self.store,
                self.beacon(state),
            ):
                if workload.mode == "remote":
                    self.provider.require_capability("remote_exec")
                    execution = self.provider.run(state.instance, workload)
                else:
                    execution = manager.run_local(workload, record)
                returncode = execution.returncode
        finally:
            manager.finish(record, returncode)

        if returncode:
            raise WorkloadFailed(
                f"workload exited with code {returncode}", returncode=returncode
            )
        return 0

    # ------------------------------------------------------------------
    # status (spec section 18)
    # ------------------------------------------------------------------

    def _observe(self) -> tuple[RuntimeState, bool]:
        """Reconciled state, or the last saved one when another command holds the lock."""
        try:
            with self.store.process_lock(timeout=READ_LOCK_WAIT_S):
                state = self._reconcile(self.store.load())
                self.store.save(state)
            return state, False
        except LockBusy:
            return self.store.load(), True

    def status(self) -> dict[str, Any]:
        state, busy = self._observe()

        cfg = self.policy_config(state.hourly_cost_usd)
        idle_skipped = self._idle_blind(state)
        if idle_skipped:
            cfg.idle_timeout_s = None
        snap = self.snapshot(state)
        decision = policy.evaluate(snap, cfg)
        if idle_skipped:
            decision.detail["idle_skipped"] = "watchdog_unreachable"
        remote = None
        client = self.beacon(state)
        if client is not None:
            remote = client.try_state()

        return {
            "provider": self.provider.name,
            "instance_id": state.instance_id,
            "state": state.state if state.instance_id else "none",
            "model": state.model_id or self.config.model.id,
            "endpoint": state.endpoint,
            "base_url": f"{state.endpoint.rstrip('/')}/v1" if state.endpoint else None,
            "created_at": state.created_at,
            "age_seconds": snap.age_seconds() if state.created_at else None,
            "idle_seconds": snap.idle_seconds(cfg),
            "locked": snap.lock_active(cfg),
            "locked_until": state.locked_until,
            "hourly_cost_usd": state.hourly_cost_usd,
            "session_cost_usd": state.session_cost_usd,
            "session_started_at": state.session_started_at,
            "stopped_since": state.stopped_since,
            "workloads": [
                {
                    "id": w.id,
                    "label": w.label,
                    "state": w.state,
                    "started_at": w.started_at,
                    "last_heartbeat": w.last_heartbeat,
                    "returncode": w.returncode,
                }
                for w in state.workloads.values()
            ],
            "decision": decision.to_dict(),
            "policy": cfg.to_dict(),
            "watchdog": {
                "enabled": self.config.lifecycle.watchdog.enabled,
                "installed": state.watchdog.installed,
                "url": state.watchdog.url,
                "reachable": remote is not None,
                "last_error": state.watchdog.last_error,
                "remote_state": remote,
            },
            "capabilities": self.provider.capabilities.__dict__,
            # True while another command (a setup) holds the lock: the state is
            # the one it last saved, not freshly asked of the provider.
            "busy": busy,
        }

    # ------------------------------------------------------------------
    # lock / unlock (spec section 11)
    # ------------------------------------------------------------------

    def lock(self, ttl_seconds: Optional[float], reason: str = "") -> dict[str, Any]:
        if ttl_seconds is None and not self.config.lifecycle.allow_indefinite_lock:
            raise ConfigError(
                "indefinite locks are disabled. Pass `--for 2h`, or set "
                "lifecycle.allow_indefinite_lock: true in jambu.yaml."
            )
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            if not state.has_instance():
                raise StateError("no instance to lock")
            until = utc_now() + ttl_seconds if ttl_seconds else None
            client = self.beacon(state)
            if client is not None and client.try_request(
                "POST", "/lock", {"ttl_seconds": ttl_seconds, "reason": reason}
            ) is None:
                state.watchdog.last_error = "unreachable"
                self.echo(
                    "! the remote watchdog did not accept the lock; it may still stop "
                    "the instance. Check `gpu status`."
                )
            state.locked = True
            state.locked_until = until
            state.lock_reason = reason
            state.touch()
            self.store.save(state)
            self.events.emit(
                ev.INSTANCE_LOCKED,
                instance_id=state.instance_id,
                locked_until=until,
                reason=reason,
            )
            return {"locked": True, "locked_until": until}

    def unlock(self, reason: str = "manual") -> dict[str, Any]:
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            client = self.beacon(state)
            if client is not None:
                client.try_request("POST", "/unlock", {"reason": reason})
            state.locked = False
            state.locked_until = None
            state.lock_reason = ""
            state.touch()
            self.store.save(state)
            self.events.emit(ev.INSTANCE_UNLOCKED, instance_id=state.instance_id, reason=reason)
            return {"locked": False}

    # ------------------------------------------------------------------
    # stop / destroy
    # ------------------------------------------------------------------

    def stop(self, reason: str = "manual") -> dict[str, Any]:
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            if not state.has_instance():
                self.echo("= nothing to stop")
                return {"changed": False, "state": state.state}
            if state.state == InstanceState.STOPPED.value:
                self.echo(f"= instance {state.instance_id} is already stopped")
                return {"changed": False, "state": state.state}
            action = (
                policy.ACTION_STOP if self.provider.capabilities.stop else policy.ACTION_DESTROY
            )
            self._terminate(state, action=action, reason=reason)
            self.store.save(state)
            return {"changed": True, "state": state.state, "action": action}

    def destroy(self, reason: str = "manual") -> dict[str, Any]:
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            if not state.instance_id:
                self.echo("= nothing to destroy")
                return {"changed": False, "state": state.state}
            self._terminate(state, action=policy.ACTION_DESTROY, reason=reason)
            self.store.save(state)
            return {"changed": True, "state": state.state}

    def _terminate(self, state: RuntimeState, action: str, reason: str) -> None:
        instance = state.instance or (
            Instance(id=state.instance_id, provider=self.provider.name)
            if state.instance_id
            else None
        )
        if instance is None:
            return
        instance_id = instance.id
        try:
            if action == policy.ACTION_DESTROY:
                self.provider.destroy(instance)
            else:
                self.provider.stop(instance)
        except ProviderError as exc:
            self.events.emit(
                ev.PROVIDER_ERROR, operation=action, instance_id=instance_id, error=str(exc)
            )
            raise

        if action == policy.ACTION_DESTROY:
            state.clear_instance(InstanceState.DESTROYED.value)
            self.events.emit(ev.INSTANCE_DESTROYED, instance_id=instance_id, reason=reason)
            self.echo(f"- instance {instance_id} destroyed ({reason})")
        else:
            state.state = InstanceState.STOPPED.value
            state.starting = False
            state.mark_billable(False)
            state.mark_stopped()
            if state.hourly_cost_usd:
                state.session_cost_usd = round(
                    state.hourly_cost_usd * state.accrued_billable_seconds() / 3600.0, 4
                )
            for record in state.workloads.values():
                if record.state in ("running", "queued"):
                    record.state = "failed"
                    record.finished_at = utc_now()
            self.events.emit(ev.INSTANCE_STOPPED, instance_id=instance_id, reason=reason)
            self.echo(f"- instance {instance_id} stopped ({reason})")

    # ------------------------------------------------------------------
    # guard (external controller path, spec section 14 alternative)
    # ------------------------------------------------------------------

    def guard_tick(self, apply: bool = True) -> dict[str, Any]:
        """Evaluate the lifecycle rule locally and optionally enforce it.

        The remote watchdog does this on the instance. This entry point exists
        for a scheduled external controller (CI job, cron) and for `status`.
        """
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            if not state.has_instance():
                self.store.save(state)
                return {"action": policy.ACTION_NONE, "reason": "no_instance"}

            cfg = self.policy_config(state.hourly_cost_usd)
            idle_skipped = self._idle_blind(state)
            if idle_skipped:
                cfg.idle_timeout_s = None
            snap = self.snapshot(state)

            if state.locked and snap.lock_expired(cfg):
                state.locked = False
                state.locked_until = None
                self.events.emit(ev.INSTANCE_UNLOCKED, reason="lock_expired")
                snap = self.snapshot(state)

            decision = policy.evaluate(snap, cfg)
            if idle_skipped:
                decision.detail["idle_skipped"] = "watchdog_unreachable"
            if decision.terminates and apply:
                self.events.emit(
                    ev.LIFECYCLE_GUARD_TRIGGERED,
                    instance_id=state.instance_id,
                    action=decision.action,
                    reason=decision.reason,
                    **decision.detail,
                )
                self._terminate(state, action=decision.action, reason=decision.reason or "guard")
            self.store.save(state)
            return decision.to_dict()

    # ------------------------------------------------------------------
    # logs
    # ------------------------------------------------------------------

    def logs(self, limit: int = 200, source: Optional[str] = None) -> list[str]:
        return self.read_logs(limit, source)[1]

    def read_logs(
        self, limit: int = 200, source: Optional[str] = None
    ) -> tuple[str, list[str]]:
        """(source used, lines). ``source`` None means the runtime log when the
        watchdog can serve it, else the provider's container log."""
        if source is not None and source not in LOG_SOURCES:
            raise ConfigError(
                f"unknown log source '{source}' (choose from {', '.join(LOG_SOURCES)})"
            )
        state, _ = self._observe()
        if not state.instance:
            raise StateError("no instance to read logs from")
        if source != "provider":
            name = source or "runtime"
            client = self.beacon(state)
            try:
                if client is None:
                    raise WatchdogUnavailable("no watchdog installed on this instance")
                return name, client.logs(name, limit)
            except WatchdogUnavailable as exc:
                if source is not None:
                    raise StateError(
                        f"cannot read the {source} log: {exc}. "
                        "`gpu logs --source provider` shows the container log instead."
                    ) from exc
        self.provider.require_capability("logs")
        stream = self.provider.logs(state.instance)
        lines = list(stream)
        return "provider", lines[-limit:]
