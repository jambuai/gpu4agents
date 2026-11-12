"""Orchestration engine.

Owns the flows in spec sections 15-18 and 23: provisioning, run, status,
reconciliation, idempotency and cleanup. It knows about the provider
*contract* and about capabilities - never about a specific provider.
"""

from __future__ import annotations

import json
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .. import watchdog as watchdog_pkg
from ..providers.base import GPUProvider
from ..providers.registry import load_builtin_providers
from ..runtime import runtime_registry
from ..runtime.base import ENDPOINT_NAME, HOST_NETWORK_FAILURE_PATTERN, ModelRuntime, image_tag, max_cuda
from ..watchdog.builder import build_agent_config, build_agent_source, build_onstart_script
from . import events as ev
from . import policy
from .beacon import WatchdogClient, WatchdogUnavailable
from .config import RuntimeConfig
from .credentials import CredentialResolver
from .durations import format_duration
from .errors import (
    BudgetExceededError,
    CapacityUnavailableError,
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
from ..runtime.service import SERVICE_LABEL
from .job import (
    JobClient,
    JobRecords,
    LOG_POLL_S,
    OUTPUTS_DIRNAME,
    build_bundle,
    load_manifest,
    new_job_id,
)
from .durations import parse_duration
from .models import (
    ComputeSpec,
    GpuSpec,
    Instance,
    InstanceState,
    JAMBU_LABEL_PREFIX,
    ValidationResult,
    Workload,
    is_jambu_label,
)
from .hosts import BadHosts
from .state import RuntimeState, StateStore, WatchdogInfo, utc_now

Echo = Callable[[str], None]

_HOST_NETWORK_FAILURE = re.compile(HOST_NETWORK_FAILURE_PATTERN)

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
        *,
        state_dir: Optional[Path] = None,
        slot: Optional[str] = None,
    ) -> None:
        self.config = config
        self.slot = slot
        self.echo = echo or _noop
        self.credentials = credentials or CredentialResolver(
            config.project_dir, profile=config.provider.profile
        )
        registry = load_builtin_providers()
        self.provider = provider or registry.get(config, self.credentials)
        self.runtime: ModelRuntime = runtime_registry.get(config)
        self.runtime.provider_image = self.provider.preferred_images.get(self.runtime.engine)
        self.store = StateStore(state_dir if state_dir is not None else config.state_dir)
        self.events = EventLog(self.store.logs_dir).bind(provider=self.provider.name)
        # Whether the last reconcile read the watchdog's state. Traffic activity
        # lives only on the instance, so without it the local idle clock lies.
        self.watchdog_read = False
        # The offer quoted for this setup (pinned with --offer, else the
        # cheapest match when setup searched) and the cheapest last seen.
        self.quote: Optional[dict[str, Any]] = None
        self._cheapest = None

    # ------------------------------------------------------------------
    # policy plumbing
    # ------------------------------------------------------------------

    def policy_config(self, hourly_cost_usd: Optional[float] = None) -> policy.PolicyConfig:
        caps = self.provider.capabilities
        cfg = self.config
        pooled = bool(self.slot and cfg.queue)
        return policy.PolicyConfig(
            idle_timeout_s=cfg.idle_timeout_s,
            max_lifetime_s=cfg.queue.max_machine_lifetime_s if pooled else cfg.max_lifetime_s,
            heartbeat_grace_s=cfg.heartbeat_grace_s,
            auto_stop=cfg.lifecycle.auto_stop,
            auto_destroy=cfg.lifecycle.auto_destroy,
            allow_indefinite_lock=cfg.lifecycle.allow_indefinite_lock,
            stop_supported=caps.stop,
            destroy_supported=caps.destroy,
            stop_eliminates_billing=caps.stop_eliminates_billing,
            hourly_cost_usd=hourly_cost_usd,
            max_session_cost_usd=None if pooled else cfg.budget.max_session_cost_usd,
            stopped_ttl_s=cfg.stopped_ttl_s,
            keep_disk=cfg.lifecycle.keep_disk,
            deadline_enforced=pooled,
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
            and cfg.lifecycle.keep_disk
            and cfg.stopped_ttl_s is None
            and not cfg.lifecycle.auto_destroy
        ):
            result.warn(
                f"a stopped {self.provider.display_name} instance still bills storage, and "
                "with lifecycle.stopped_ttl disabled and auto_destroy false nothing ever "
                "removes it; set lifecycle.stopped_ttl (e.g. 24h)",
                "lifecycle.stopped_ttl",
            )

        from .model_inspect import known_vram_floor

        floor = known_vram_floor(cfg.model.id or "") if cfg.runtime.engine != "none" else None
        if floor and cfg.compute.gpu.min_vram_gb < floor[0]:
            result.warn(
                f"compute.gpu.min_vram_gb is {cfg.compute.gpu.min_vram_gb}, but {cfg.model.id} needs "
                f"about {floor[0]} GB at its peak ({floor[1]} GB to be safe at high resolution, "
                "e.g. 1344x768 and up). Size for the peak, not the weights",
                "compute.gpu.min_vram_gb",
            )

        result.merge(self.runtime.validate())
        if cfg.runtime.engine != "none" and image_tag(self.runtime.image()) == "latest":
            result.warn(
                f"unpinned image '{self.runtime.image()}': the CUDA it needs can change "
                "under you (pin a tag such as "
                f"'{type(self.runtime).default_image}')",
                "compute.image",
            )

        if cfg.model.id and cfg.model.hf_token_env and not self.credentials.get(cfg.model.hf_token_env):
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

    def _why_missing(self, state: RuntimeState) -> Optional[str]:
        """The rule that would have had the watchdog destroy this machine by now, if any.

        A destroyed machine leaves nothing to ask: the provider forgets it and
        the watchdog went with it. What can be said is whether this project's
        own rules (idle_timeout, max_lifetime, the budget) called for a destroy
        given what was last known. None: they did not, so something else did.
        """
        if not self.config.lifecycle.watchdog.enabled or not (
            state.watchdog.installed or state.watchdog.token
        ):
            return None
        if state.state in (InstanceState.STOPPED.value, InstanceState.DESTROYED.value):
            return None
        snapshot = self.snapshot(state)
        snapshot.provider_state = policy.READY
        snapshot.starting = False
        decision = policy.evaluate(snapshot, self.policy_config(state.hourly_cost_usd))
        if decision.action != policy.ACTION_DESTROY:
            return None
        return decision.reason

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
            likely = self._why_missing(state)
            instance_id = state.instance_id
            state.clear_instance(InstanceState.DESTROYED.value)
            self.events.emit(
                ev.STATE_RECONCILED,
                previous_state=previous,
                state=InstanceState.DESTROYED.value,
                reason="instance missing at provider",
                likely_reason=f"{likely} (watchdog)" if likely else (
                    "unknown: not this project's lifecycle rules (the provider console, "
                    "another tool, the host, or the account out of credit)"
                ),
            )
            if likely:
                self.events.emit(
                    ev.INSTANCE_DESTROYED,
                    instance_id=instance_id,
                    reason=likely,
                    by="watchdog",
                    inferred=True,
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
            label=self._label(cfg),
            spot=cfg.compute.spot,
            region=cfg.compute.region,
            max_hourly_cost_usd=cfg.budget.max_hourly_cost_usd,
            provider_options=dict(cfg.provider.options),
        )

    def _label(self, cfg: RuntimeConfig) -> str:
        fp = cfg.fingerprint()[:6]
        if self.slot:
            return f"{JAMBU_LABEL_PREFIX}{self._label_name()[:20]}-{self.slot}-{fp}"
        return f"{JAMBU_LABEL_PREFIX}{self._label_name()[:24]}-{fp}"

    def _label_name(self) -> str:
        cfg = self.config
        if cfg.serves_as_service:
            # One machine may serve several models over its life (redeploys).
            return f"serve-{cfg.runtime.engine}"
        return cfg.model.id.split("/")[-1] or "jobs"

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
            stop_keeps_disk=self.provider.capabilities.stop
            and self.provider.capabilities.persistent_disk,
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
        self._cheapest = offers[0] if offers else None
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

    # -- quotes: the offer the user said yes to is the one rented -------------

    def _offers_path(self) -> Path:
        return self.store.dir / "offers.json"

    def offers(self, limit: int = 10, for_minutes_on_4090: Optional[float] = None) -> list[dict[str, Any]]:
        """Matching offers, remembered so `setup --offer` can pin one.

        Cheapest per hour first; with ``for_minutes_on_4090`` (how long the job
        takes on an RTX 4090), cheapest per job first, since a faster card at a
        higher hourly price often costs the same and finishes far sooner.
        """
        from ..providers.gpus import relative_speed

        self.provider.require_capability("pricing_query")
        spec = self.build_spec()
        searched = limit * 5 if for_minutes_on_4090 else limit
        rows = [
            {
                "id": o.id,
                "gpu": o.gpu_name,
                "count": o.gpu_count,
                "vram_gb": o.gpu_vram_gb,
                "disk_gb": o.disk_gb,
                "hourly_cost_usd": o.hourly_cost_usd,
                "region": o.region,
                "cuda_max": o.cuda_max,
            }
            for o in self.provider.find_offers(spec, limit=searched)
        ]
        if for_minutes_on_4090:
            for row in rows:
                speed = relative_speed(row["gpu"])
                minutes = for_minutes_on_4090 / speed if speed else None
                row["est_minutes"] = round(minutes, 1) if minutes else None
                row["est_job_cost_usd"] = (
                    round(row["hourly_cost_usd"] * minutes / 60.0, 4) if minutes else None
                )
            rows.sort(
                key=lambda r: (
                    r["est_job_cost_usd"] is None,
                    r["est_job_cost_usd"] or 0.0,
                    r["est_minutes"] or 0.0,
                )
            )
            rows = rows[:limit]
        try:
            path = self._offers_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"quoted_at": utc_now(), "offers": rows}, indent=2) + "\n")
        except OSError:
            pass  # a read-only project still gets its prices
        return rows

    def _quote_for(self, offer_id: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        try:
            rows = json.loads(self._offers_path().read_text()).get("offers") or []
        except (OSError, ValueError):
            rows = []
        for row in rows:
            if str(row.get("id")) == str(offer_id):
                return {"offer_id": str(offer_id), "gpu": row["gpu"], "hourly_cost_usd": row["hourly_cost_usd"]}
        if self.provider.capabilities.pricing_query:
            for offer in self.provider.find_offers(self.build_spec(), limit=50):
                if str(offer.id) == str(offer_id):
                    return {"offer_id": offer.id, "gpu": offer.gpu_name, "hourly_cost_usd": offer.hourly_cost_usd}
        raise ConfigError(
            f"offer {offer_id} is not one `gpu offers` showed for this project; run `gpu offers` "
            "again and ask the user about the card and price it shows"
        )

    @staticmethod
    def _price_tolerance(price: float) -> float:
        # Quotes are compared after rounding; anything beyond that is a new price.
        return max(price * 0.02, 0.001)

    def _pin(self, spec: ComputeSpec, quote: dict[str, Any]) -> None:
        spec.gpu.name_filter = quote["gpu"]
        cap = float(quote["hourly_cost_usd"]) + self._price_tolerance(float(quote["hourly_cost_usd"]))
        spec.max_hourly_cost_usd = min(cap, spec.max_hourly_cost_usd or cap)
        spec.preferred_offer_id = quote["offer_id"]

    def rented_vs_quoted(self, state: RuntimeState) -> dict[str, Any]:
        """What was rented next to what the user was quoted, and whether they agree."""
        instance = state.instance
        rented = {
            "instance_id": state.instance_id,
            "gpu": instance.gpu_name if instance else None,
            "hourly_cost_usd": instance.hourly_cost_usd if instance else state.hourly_cost_usd,
        }
        quote = self.quote
        out: dict[str, Any] = {"rented": rented, "quoted": quote, "matches_quote": True}
        if not quote or not rented["gpu"]:
            return out
        price = rented["hourly_cost_usd"] or 0.0
        same_card = quote["gpu"].lower() in rented["gpu"].lower()
        dearer = price > float(quote["hourly_cost_usd"]) + self._price_tolerance(float(quote["hourly_cost_usd"]))
        if same_card and not dearer:
            return out
        out["matches_quote"] = False
        out["warning"] = (
            f"rented {rented['gpu']} at ${price:.3f}/h, not the {quote['gpu']} at "
            f"${float(quote['hourly_cost_usd']):.3f}/h that was quoted: tell the user before using it"
            + ("; a higher price needs their yes" if dearer else "")
        )
        return out

    def setup(
        self,
        *,
        recreate: bool = False,
        allow_cost_override: bool = False,
        wait: bool = True,
        offer_id: Optional[str] = None,
    ) -> RuntimeState:
        cfg = self.config
        fingerprint = cfg.fingerprint()
        pinned = self._quote_for(offer_id) if offer_id else None
        self.quote = pinned

        with self.store.process_lock():
            state = self._reconcile(self.store.load())

            # -- idempotency (spec section 23) -----------------------------
            if state.has_instance():
                if state.fingerprint != fingerprint and not recreate:
                    raise ConfigError(
                        f"instance {state.instance_id} was provisioned for a different "
                        "runtime (gpu.yaml changed since setup). Re-run with "
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
                if pinned:
                    self._pin(spec, pinned)
                    if not self.provider.find_offers(spec, limit=1):
                        exc = CapacityUnavailableError(
                            f"the {pinned['gpu']} at ${float(pinned['hourly_cost_usd']):.3f}/h the user approved "
                            "is no longer available at that price. Nothing was rented: run `gpu offers` "
                            "and ask the user again"
                        )
                        self._fail_setup(state, exc)
                        raise exc
                # Every new offer is held to the budget, not just the first.
                estimated = self._check_budget(spec, allow_cost_override)
                if estimated:
                    self.echo(f"  cheapest match: ${estimated:.3f}/h")
                if self.quote is None and self._cheapest is not None:
                    cheapest = self._cheapest
                    self.quote = {
                        "offer_id": cheapest.id,
                        "gpu": cheapest.gpu_name,
                        "hourly_cost_usd": cheapest.hourly_cost_usd,
                    }
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

        compared = self.rented_vs_quoted(state)
        if compared.get("warning"):
            self.echo(f"! {compared['warning']}")
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

            if self.runtime.service:
                self._deploy_service(state)
            if self.config.runtime.engine != "none":
                self._wait_runtime(state)
            else:
                # Nothing to wait for: end the watchdog's startup grace now, or
                # an unused bare machine would bill until the grace expires.
                client = self.beacon(state)
                if client is not None:
                    client.mark_ready()

            state.starting = False
            state.state = InstanceState.READY.value
            state.touch()
            self.store.save(state)
            self._pull_watchdog_events(state)
            return state

    def _pull_watchdog_events(self, state: RuntimeState) -> None:
        """Bring the watchdog's events home now: if the machine goes while nobody
        runs gpu, they are the only record that it was guarded."""
        client = self.beacon(state)
        if client is None:
            return
        try:
            pulled = client.events(limit=200)
        except Exception:  # noqa: BLE001 - a record, never a failure
            return
        if pulled:
            self.events.extend(pulled)

    def _wait_running(self, state: RuntimeState) -> Instance:
        """Wait for the machine itself (mostly the image pull), up to
        health.boot_timeout. The container has not run yet, so every failure
        here is the host's.
        """
        instance = state.instance
        assert instance is not None
        boot_timeout = self.config.boot_timeout_s
        stall_timeout = (
            self.config.pull_stall_timeout_s if self.provider.capabilities.boot_progress else None
        )
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
            if self.runtime.service:
                # A redeploy late in a session has no startup grace left: the
                # wait itself is the activity that keeps the machine up.
                client = self.beacon(state)
                if client is not None:
                    client.try_request("POST", "/activity")
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
            "engine": self.config.runtime.engine,
            "task": self._task(),
            "routes": self.runtime.routes(),
            "hourly_cost_usd": state.hourly_cost_usd,
        }

    def _task(self) -> Optional[str]:
        engine = self.config.runtime.engine
        if engine == "vllm":
            return "chat"
        if engine == "hf":
            return self.config.model.task
        return None

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
    # jobs: any program, run on the machine itself (core/job.py)
    # ------------------------------------------------------------------

    def _job_client(self, start: bool) -> tuple[RuntimeState, JobClient]:
        """A ready machine with the current job runner loaded on its watchdog."""
        if not self.config.lifecycle.watchdog.enabled:
            raise ConfigError(
                "jobs run through the machine's watchdog; set lifecycle.watchdog.enabled: true"
            )
        if start:
            state = self.ensure_ready()
        else:
            state, _ = self._observe()
            if not state.has_instance():
                raise StateError("no machine for this project; `gpu setup` starts one")
            if state.state == InstanceState.STOPPED.value:
                raise StateError(
                    f"instance {state.instance_id} is stopped; `gpu setup` restarts it "
                    "(job files are on its disk)"
                )
        client = self.beacon(state)
        if client is None:
            raise StateError("the machine has no reachable watchdog; jobs cannot run on it")
        jobs = JobClient(client)
        jobs.ensure_runner()
        return state, jobs

    def job_env(self, state: RuntimeState) -> dict[str, str]:
        """What a job gets besides its own env: the model, over loopback."""
        env = {"JAMBU_PROVIDER": self.provider.name, "JAMBU_INSTANCE_ID": state.instance_id or ""}
        if self.config.runtime.engine != "none" and self.runtime.ports():
            port = self.config.runtime.port
            env.update(
                {
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                    "OPENAI_API_KEY": self.api_key_for(state),
                    "JAMBU_MODEL_ID": self.served_model(),
                }
            )
        return env

    def job_submit(
        self,
        directory: Path,
        command: Optional[str] = None,
        *,
        setup: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[str] = None,
        label: Optional[str] = None,
        exclude: Sequence[str] = (),
        start: bool = True,
    ) -> dict[str, Any]:
        """Pack ``directory``, upload it and start the job. Returns its record."""
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ConfigError(f"job directory not found: {directory}")
        manifest = load_manifest(directory)
        command = command or manifest.run
        if not command:
            raise ConfigError(
                "nothing to run: pass a command (`gpu job run -- python train.py`) "
                "or set run: in gpujob.yaml"
            )
        timeout = timeout or manifest.timeout
        timeout_s = parse_duration(timeout) if timeout else None
        lifetime = self.config.max_lifetime_s
        if lifetime and timeout_s and timeout_s > lifetime:
            raise ConfigError(
                f"the job may run {format_duration(timeout_s)} but lifecycle.max_lifetime "
                f"shuts the machine down after {format_duration(lifetime)}; raise "
                "max_lifetime in gpu.yaml (and the session budget, if set) to at least the timeout"
            )
        merged_env = dict(manifest.env)
        merged_env.update(env or {})

        state, jobs = self._job_client(start)
        record = self._submit(
            state,
            jobs,
            directory,
            command,
            setup=setup or manifest.setup,
            env={**self.job_env(state), **merged_env},
            exclude=list(manifest.exclude) + list(exclude),
            label=label or manifest.label or "",
            timeout_s=timeout_s,
        )
        self.echo(f"+ job {record['id']} running on instance {state.instance_id}")
        if lifetime and state.session_started_at:
            left = lifetime - (utc_now() - state.session_started_at)
            if timeout_s is None or timeout_s > left:
                self.echo(
                    f"! lifecycle.max_lifetime shuts this machine down in {format_duration(max(0.0, left))}"
                    ", even mid-job; what the job wrote by then is kept and downloaded"
                )
        return record

    def _submit(
        self,
        state: RuntimeState,
        jobs: JobClient,
        directory: Path,
        command: str,
        *,
        setup: Optional[str],
        env: dict[str, str],
        exclude: Sequence[str],
        label: str,
        timeout_s: Optional[float] = None,
        service: bool = False,
        digest: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upload ``directory`` and start ``command`` on the job runner."""
        job_id = new_job_id()
        with tempfile.TemporaryDirectory(prefix="gpu-job-") as tmp:
            bundle = Path(tmp) / "bundle.tar.gz"
            packed = build_bundle(directory, list(exclude), bundle)
            self.echo(
                f"> uploading {packed['files']} files ({packed['bytes'] / 1024**2:.1f} MB) as "
                f"{'model server' if service else 'job'} {job_id}"
            )
            if packed["files"]:
                jobs.upload(job_id, bundle)
        payload = {
            "command": command,
            "setup": setup,
            "env": env,
            "timeout_s": timeout_s,
            "label": label,
            "service": service,
            "digest": digest,
        }
        started = jobs.start(job_id, payload)["job"]
        record = JobRecords(self.store.dir).update(
            job_id,
            label=label,
            command=command,
            setup=setup,
            directory=str(directory),
            instance_id=state.instance_id,
            hourly_cost_usd=state.hourly_cost_usd,
            submitted_at=utc_now(),
            state=started.get("state"),
            files=packed["files"],
            bundle_bytes=packed["bytes"],
            service=service,
        )
        self.events.emit(
            ev.JOB_SUBMITTED,
            job_id=job_id,
            instance_id=state.instance_id,
            label=label,
            files=packed["files"],
            bundle_bytes=packed["bytes"],
            service=service,
        )
        return record

    # ------------------------------------------------------------------
    # model servers deployed as service jobs (runtime/service.py)
    # ------------------------------------------------------------------

    def _deploy_service(self, state: RuntimeState, force: bool = False) -> str:
        """Start the configured model server unless that exact one already runs."""
        client = self.beacon(state)
        if client is None or not self.config.lifecycle.watchdog.enabled:
            raise ConfigError(
                f"runtime.engine: {self.config.runtime.engine} runs its model server through "
                "the machine's watchdog; set lifecycle.watchdog.enabled: true"
            )
        jobs = JobClient(client)
        jobs.ensure_runner()
        spec = self.runtime.service_spec()  # type: ignore[attr-defined]
        spec.env["JAMBU_API_KEY"] = self.api_key_for(state)
        digest = spec.digest()
        running = [j for j in jobs.list() if j.get("service") and j.get("state") == "running"]
        if not force:
            for job in running:
                if job.get("digest") == digest:
                    self.echo(f"= model server {job['id']} already running")
                    return str(job["id"])
        for job in running:
            self.echo(f"~ stopping model server {job['id']} (the model or its code changed)")
            jobs.cancel(job["id"])
            self._wait_job_end(jobs, str(job["id"]))
        client.mark_starting()
        record = self._submit(
            state,
            jobs,
            spec.directory,
            spec.command,
            setup=spec.setup,
            env=spec.env,
            exclude=spec.exclude,
            label=SERVICE_LABEL,
            service=True,
            digest=digest,
        )
        state.model_id = self.config.model.id
        self.store.save(state)
        self.echo(f"+ model server {record['id']} starting ({self.runtime.describe()})")
        return str(record["id"])

    def _wait_job_end(self, jobs: JobClient, job_id: str, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if jobs.info(job_id).get("state") != "running":
                return
            time.sleep(1.0)
        raise RuntimeStartupError(f"the old model server {job_id} did not stop within {int(timeout)}s")

    def redeploy(self) -> RuntimeState:
        """Restart the model server on the running machine, same settings or not."""
        if not self.runtime.service:
            raise ConfigError(
                "only runtime.engine: hf or custom redeploys in place; vLLM's command is "
                "fixed at rent time (`gpu setup --recreate`)"
            )
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            if not state.has_instance() or state.state == InstanceState.STOPPED.value:
                raise StateError("no running machine; `gpu setup` starts one and deploys the server")
            self._deploy_service(state, force=True)
            self._wait_runtime(state)
            state.starting = False
            state.state = InstanceState.READY.value
            state.touch()
            self.store.save(state)
            return state

    def _resolve_job(self, job_id: Optional[str]) -> str:
        if job_id:
            return job_id
        latest = JobRecords(self.store.dir).latest()
        if latest is None:
            raise StateError("no jobs in this project yet; `gpu job run` starts one")
        return str(latest["id"])

    def _job_summary(self, info: dict[str, Any], state: RuntimeState) -> dict[str, Any]:
        rate = state.hourly_cost_usd
        if info.get("elapsed_s") is None and info.get("started_at"):
            end = info.get("finished_at") or utc_now()
            info["elapsed_s"] = round(float(end) - float(info["started_at"]), 1)
        elapsed = info.get("elapsed_s")
        info["cost_usd"] = round(rate * elapsed / 3600.0, 4) if rate and elapsed else None
        return info

    def job_info(self, job_id: Optional[str] = None) -> dict[str, Any]:
        job_id = self._resolve_job(job_id)
        state, jobs = self._job_client(start=False)
        info = self._job_summary(jobs.info(job_id), state)
        JobRecords(self.store.dir).update(
            job_id, state=info.get("state"), returncode=info.get("returncode")
        )
        return info

    def job_log(self, job_id: Optional[str] = None, offset: int = 0, limit: Optional[int] = None) -> dict[str, Any]:
        job_id = self._resolve_job(job_id)
        _, jobs = self._job_client(start=False)
        data = jobs.log(job_id, offset, limit)
        data["job_id"] = job_id
        return data

    def job_follow(
        self,
        job_id: str,
        on_output: Callable[[str], None],
        offset: int = 0,
        poll_s: float = LOG_POLL_S,
    ) -> dict[str, Any]:
        """Stream the job's log until it ends; returns its final record."""
        _, jobs = self._job_client(start=False)
        while True:
            chunk = jobs.log(job_id, offset)
            if chunk.get("data"):
                on_output(chunk["data"])
            moved = chunk["offset"] != offset
            offset = chunk["offset"]
            if chunk.get("done") and not moved:
                break
            if not moved:
                time.sleep(poll_s)
        return self.job_finished(job_id)

    def job_finished(self, job_id: str) -> dict[str, Any]:
        info = self.job_info(job_id)
        if info.get("state") != "running":
            self.events.emit(
                ev.JOB_FINISHED,
                job_id=job_id,
                state=info.get("state"),
                returncode=info.get("returncode"),
                elapsed_s=info.get("elapsed_s"),
            )
        return info

    def job_pull(self, job_id: Optional[str] = None, dest: Optional[Path] = None) -> dict[str, Any]:
        """Download a job's outputs; a stopped machine is started for it and put back."""
        job_id = self._resolve_job(job_id)
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            woken = False
            if state.has_instance() and state.state == InstanceState.STOPPED.value:
                state = self._wake_for_outputs(state)
                woken = True
            try:
                _, jobs = self._job_client(start=False)
                result = self._pull_outputs(jobs, job_id, dest)
            finally:
                if woken:
                    self._settle_after_wake(state)
            return result

    def _pull_outputs(self, jobs: JobClient, job_id: str, dest: Optional[Path] = None) -> dict[str, Any]:
        target = Path(dest) if dest else self.config.project_dir / OUTPUTS_DIRNAME / job_id
        files = jobs.download_outputs(job_id, target)
        try:
            unsaved = bool(jobs.info(job_id).get("outputs_unsaved", False))
        except (StateError, WatchdogUnavailable):
            unsaved = True
        JobRecords(self.store.dir).update(
            job_id, outputs_dir=str(target), outputs=len(files), outputs_unsaved=unsaved
        )
        self.events.emit(ev.JOB_OUTPUTS_PULLED, job_id=job_id, files=len(files), dest=str(target))
        self.echo(f"+ job {job_id}: {len(files)} output files in {target}")
        return {"job_id": job_id, "outputs_dir": str(target), "files": files}

    # ------------------------------------------------------------------
    # job results are never deleted unseen
    # ------------------------------------------------------------------

    def unsaved_job_outputs(self, state: RuntimeState, jobs: Optional[JobClient] = None) -> list[str]:
        """Jobs on this instance whose results may not exist anywhere else.

        With ``jobs`` (the machine's runner) the machine's own answer is
        recorded locally, so it is still known once the machine is stopped.
        Without it, a job never confirmed saved counts as unsaved.
        """
        if not state.instance_id:
            return []
        records = JobRecords(self.store.dir)
        mine = {
            r["id"]: r
            for r in records.all()
            if r.get("instance_id") == state.instance_id and not r.get("service")
        }
        if jobs is not None and mine:
            remote = {r.get("id"): r for r in jobs.list()}
            for job_id, record in list(mine.items()):
                found = remote.get(job_id)
                # Gone from the machine: there is nothing left there to lose.
                unsaved = bool(found.get("outputs_unsaved", True)) if found else False
                if record.get("outputs_unsaved") != unsaved:
                    mine[job_id] = records.update(
                        job_id,
                        outputs_unsaved=unsaved,
                        **({"state": found.get("state")} if found else {}),
                    )
        return sorted(j for j, r in mine.items() if r.get("outputs_unsaved", True))

    def _reachable_jobs(self, state: RuntimeState) -> Optional[JobClient]:
        if not InstanceState(state.state).is_billable_compute:
            return None
        client = self.beacon(state)
        if client is None or not client.ping():
            return None
        jobs = JobClient(client)
        try:
            jobs.ensure_runner()
        except (StateError, WatchdogUnavailable):
            return None
        return jobs

    WAKE_RETRY_S = 3600.0

    def _wake_allowed(self) -> bool:
        """The guard restarts a machine for its results at most once an hour."""
        path = self.store.dir / "outputs-wake.json"
        try:
            last = float(json.loads(path.read_text()).get("at") or 0)
        except (OSError, ValueError, AttributeError):
            last = 0.0
        if utc_now() - last < self.WAKE_RETRY_S:
            return False
        self.store.ensure_dirs()
        path.write_text(json.dumps({"at": utc_now()}))
        return True

    def _save_job_outputs(
        self, state: RuntimeState, throttle: bool = False, finished_only: bool = False
    ) -> tuple[RuntimeState, list[str]]:
        """Download every unsaved job result. Returns the state and what is still unsaved.

        A stopped machine is started for it (and left running: the caller
        decides what happens to it next).
        """
        if not state.instance_id or not JobRecords(self.store.dir).all():
            return state, []
        jobs = self._reachable_jobs(state)
        unsaved = self.unsaved_job_outputs(state, jobs)
        if not unsaved:
            return state, []
        if jobs is None and state.state == InstanceState.STOPPED.value:
            if throttle and not self._wake_allowed():
                return state, unsaved
            try:
                state = self._wake_for_outputs(state)
            except JambuError as exc:
                self.events.emit(ev.JOB_OUTPUTS_KEPT, job_ids=unsaved, error=str(exc))
                self.echo(f"! could not start instance {state.instance_id} to fetch job results: {exc}")
                return state, unsaved
            jobs = self._reachable_jobs(state)
            if jobs is not None:
                unsaved = self.unsaved_job_outputs(state, jobs)
        if jobs is None:
            self.events.emit(ev.JOB_OUTPUTS_KEPT, job_ids=unsaved, error="watchdog unreachable")
            return state, unsaved
        if finished_only:
            done = {r.get("id") for r in jobs.list() if r.get("state") != "running"}
            unsaved = [j for j in unsaved if j in done]
        left = []
        for job_id in unsaved:
            try:
                self._pull_outputs(jobs, job_id)
            except (JambuError, WatchdogUnavailable, OSError) as exc:
                self.events.emit(ev.JOB_OUTPUTS_KEPT, job_ids=[job_id], error=str(exc))
                left.append(job_id)
        # A job still running was just pulled as far as it got; what it writes
        # after this is lost with the machine, but nothing it already wrote is.
        return state, left

    def _wake_for_outputs(self, state: RuntimeState) -> RuntimeState:
        """Start a stopped machine just far enough to reach its watchdog."""
        instance = state.instance
        if instance is None or not self.provider.capabilities.stop:
            raise StateError(f"instance {state.instance_id} cannot be started again")
        self.echo(f"> starting instance {instance.id} to download job results")
        instance = self.provider.start(instance)
        state.instance = instance
        state.mark_billable(True)
        state.mark_session_start()
        self.events.emit(ev.INSTANCE_STARTED, instance_id=instance.id, reason="job_outputs")
        state.state = InstanceState.STARTING.value
        self.store.save(state)
        instance = self._wait_running(state)
        state.instance = instance
        state.endpoint = instance.endpoint_url(ENDPOINT_NAME)
        watchdog_url = instance.endpoint_url("watchdog")
        if watchdog_url:
            state.watchdog.url = watchdog_url
        self.store.save(state)
        client = self.beacon(state)
        deadline = time.time() + 120
        while client is not None and time.time() < deadline:
            if client.ping():
                state.watchdog.last_contact = utc_now()
                state.watchdog.last_error = None
                return state
            time.sleep(5)
        raise StateError(f"the watchdog on instance {instance.id} did not answer after the restart")

    def _settle_after_wake(self, state: RuntimeState) -> None:
        """Put a machine started only for its results back the way policy wants it."""
        caps = self.provider.capabilities
        billing_stop = not caps.stop_eliminates_billing and not self.config.lifecycle.keep_disk
        if billing_stop and caps.destroy and not self.unsaved_job_outputs(state):
            action = policy.ACTION_DESTROY
        else:
            action = policy.ACTION_STOP
        try:
            self._terminate(state, action=action, reason="job_outputs_saved")
        except ProviderError:
            pass  # the watchdog still shuts an idle machine down
        self.store.save(state)

    def _protect_job_outputs(
        self, state: RuntimeState, decision: policy.Decision, locked: bool
    ) -> tuple[RuntimeState, policy.Decision]:
        """No automatic shutdown deletes job results nobody downloaded.

        Before one would, the results are downloaded into gpu-outputs/. A
        stopped machine kept only for them (a stop that still bills, without
        keep_disk) is started, emptied and destroyed. What cannot be fetched
        keeps the machine: stopped where the disk survives that, else as is.
        """
        caps = self.provider.capabilities
        stopped = state.state == InstanceState.STOPPED.value
        if (
            not decision.terminates
            and InstanceState(state.state).is_billable_compute
            and self.unsaved_job_outputs(state)
        ):
            # Fetch finished jobs' results while the machine is still up: then
            # the watchdog can destroy it when idle instead of keeping a disk.
            state, _ = self._save_job_outputs(state, finished_only=True)
            return state, decision
        if not decision.terminates:
            billing_stop = not caps.stop_eliminates_billing and not self.config.lifecycle.keep_disk
            if not (stopped and billing_stop and caps.destroy and not locked):
                return state, decision
            if not self.unsaved_job_outputs(state):
                return state, decision
            state, left = self._save_job_outputs(state, throttle=True)
            detail = dict(decision.detail)
            if not left:
                return state, policy.Decision(policy.ACTION_DESTROY, "job_outputs_saved", detail)
            detail["job_outputs_unsaved"] = left
            if InstanceState(state.state).is_billable_compute:
                return state, policy.Decision(policy.ACTION_STOP, "job_outputs_unsaved", detail)
            return state, policy.Decision(policy.ACTION_NONE, "job_outputs_unsaved", detail)

        if decision.action != policy.ACTION_DESTROY and caps.persistent_disk:
            return state, decision  # a stop keeps the disk, results included
        state, left = self._save_job_outputs(state, throttle=True)
        if not left:
            return state, decision
        detail = dict(decision.detail)
        detail["job_outputs_unsaved"] = left
        running = InstanceState(state.state).is_billable_compute
        if running and caps.stop and caps.persistent_disk:
            return state, policy.Decision(policy.ACTION_STOP, decision.reason, detail)
        if not running or decision.reason == policy.REASON_IDLE:
            return state, policy.Decision(policy.ACTION_NONE, "job_outputs_unsaved", detail)
        return state, decision

    def job_cancel(self, job_id: Optional[str] = None) -> dict[str, Any]:
        job_id = self._resolve_job(job_id)
        _, jobs = self._job_client(start=False)
        result = jobs.cancel(job_id)
        result["job_id"] = job_id
        return result

    def job_list(self) -> list[dict[str, Any]]:
        """This project's jobs: the machine's records, else the local ones."""
        records = {r["id"]: r for r in JobRecords(self.store.dir).all()}
        current: Optional[str] = None
        try:
            state, jobs = self._job_client(start=False)
            current = state.instance_id
            for remote in jobs.list():
                merged = records.get(remote["id"], {})
                merged.update(self._job_summary(dict(remote), state))
                records[remote["id"]] = merged
        except (StateError, WatchdogUnavailable):
            current = self.store.load().instance_id
        for record in records.values():
            # Its machine is gone: whatever it was doing ended with it.
            if record.get("state") == "running" and record.get("instance_id") != current:
                record["state"] = "lost"
        return sorted(records.values(), key=lambda r: r.get("submitted_at") or r.get("started_at") or 0)

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
            # Jobs whose results are only on the machine (as last known here).
            "job_outputs_unsaved": self.unsaved_job_outputs(state) if state.instance_id else [],
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

    def set_deadline(self, until: float) -> float:
        """Tell the machine's watchdog to end it at `until` (it never moves back)."""
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            if not state.has_instance():
                raise StateError("no instance to set a deadline on")
            client = self.beacon(state)
            if client is None:
                raise StateError("no watchdog to carry the deadline")
            return float(client.set_deadline(until)["deadline_at"])

    def lock(self, ttl_seconds: Optional[float], reason: str = "") -> dict[str, Any]:
        if ttl_seconds is None and not self.config.lifecycle.allow_indefinite_lock:
            raise ConfigError(
                "indefinite locks are disabled. Pass `--for 2h`, or set "
                "lifecycle.allow_indefinite_lock: true in gpu.yaml."
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
            instance = state.instance
            self._terminate(state, action=action, reason=reason)
            self.store.save(state)
            result: dict[str, Any] = {"changed": True, "state": state.state, "action": action}
            if action == policy.ACTION_STOP:
                result.update(self._stopped_bill(instance))
            return result

    def adopt(self, instance_id: str) -> dict[str, Any]:
        """Take over a machine gpu rented whose project was lost (an orphan).

        Its project folder went away (a temp or session scratch folder), so no
        one held its watchdog token or model key. The provider still knows the
        machine, and some (Vast.ai) keep the boot script that carries both.
        With them the machine is reusable; without them it can still be
        watched, stopped and destroyed from here.
        """
        from . import sweep as sweep_mod
        from ..watchdog.builder import recover_boot_secrets

        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            if state.has_instance():
                raise StateError(
                    f"this project already has machine {state.instance_id}; adopt from "
                    "another directory, or `gpu destroy` this one first"
                )
            owners = sweep_mod.tracked_by_projects(sweep_mod.ProjectRegistry().paths())
            owner = owners.get((self.provider.name, str(instance_id)))
            if owner is not None and owner.is_file() and owner.resolve() != self.config.source_path.resolve():
                raise ConfigError(
                    f"project {owner.parent} already tracks machine {instance_id}; run gpu there"
                )
            status = self.provider.status(Instance(id=str(instance_id), provider=self.provider.name))
            current = status.instance if status.exists else None
            if current is None:
                raise ConfigError(f"there is no machine {instance_id} in this {self.provider.display_name} account")
            if not is_jambu_label(current.label):
                raise ConfigError(
                    f"machine {instance_id} was not created by gpu (label {current.label!r}); not adopting it"
                )

            secrets_found: dict[str, Any] = {}
            try:
                script = self.provider.boot_script(current)
            except ProviderError:
                script = None
            if script:
                secrets_found = recover_boot_secrets(script, api_key_env=self.runtime.api_key_env)

            fingerprint = self.config.fingerprint()
            same_config = current.label.endswith(f"-{fingerprint[:6]}")
            reusable = same_config and bool(secrets_found.get("token"))
            state = RuntimeState()
            state.provider = self.provider.name
            state.instance_id = current.id
            state.instance = current
            state.fingerprint = fingerprint if same_config else f"adopted-{current.label}"
            state.model_id = self.config.model.id if same_config else None
            state.api_key = secrets_found.get("api_key") or self.config.runtime.api_key
            state.created_at = current.created_at or utc_now()
            state.session_started_at = state.created_at
            state.last_activity = utc_now()
            state.hourly_cost_usd = current.hourly_cost_usd
            state.endpoint = current.endpoint_url(ENDPOINT_NAME)
            state.state = status.state.value
            state.watchdog = WatchdogInfo(
                installed=bool(secrets_found.get("token")),
                token=secrets_found.get("token"),
                port=secrets_found.get("port") or self.config.lifecycle.watchdog.port,
                url=current.endpoint_url("watchdog"),
            )
            self.store.save(state)
            sweep_mod.ProjectRegistry().register(self.config.source_path)
            self.events.emit(
                ev.INSTANCE_ADOPTED,
                instance_id=current.id,
                label=current.label,
                watchdog=bool(secrets_found.get("token")),
                same_config=same_config,
            )
            state = self._reconcile(state)
            self.store.save(state)

        if reusable:
            nxt = "gpu setup reuses it (it restarts it if stopped); gpu destroy deletes it"
        elif not same_config:
            nxt = (
                "it was set up from a different gpu.yaml, so gpu setup will not reuse it: "
                "gpu destroy deletes it (or gpu setup --recreate replaces it)"
            )
        else:
            nxt = (
                "its watchdog key could not be recovered from the provider, so it cannot be "
                "reused safely: gpu destroy deletes it"
            )
        return {
            "instance_id": current.id,
            "state": state.state,
            "gpu": f"{current.gpu_count}x {current.gpu_name}".strip(),
            "hourly_cost_usd": current.hourly_cost_usd,
            "watchdog": bool(secrets_found.get("token")),
            "reusable": reusable,
            "next": nxt,
        }

    def _stopped_bill(self, instance: Optional[Instance]) -> dict[str, Any]:
        """What a stop leaves billing, said out loud (a stop that bills is easy to forget)."""
        if self.provider.capabilities.stop_eliminates_billing:
            return {"still_billing": False}
        from ..cli import timer as timer_mod
        from .durations import format_duration

        hourly = instance.stopped_hourly_cost_usd if instance else None
        per_day = round(hourly * 24, 2) if hourly else None
        timer_on = timer_mod.is_installed()
        ttl = self.config.stopped_ttl_s
        bill = f"~${per_day:.2f}/day" if per_day else "money"
        warning = f"stopped, but still billing {bill} for its disk"
        if ttl is None:
            warning += "; lifecycle.stopped_ttl is off, so it bills until `gpu destroy`"
        elif timer_on:
            warning += f"; the sweep timer has it destroyed after stopped_ttl ({format_duration(ttl)})"
        else:
            warning += (
                f"; it is destroyed after stopped_ttl ({format_duration(ttl)}) only if "
                "`gpu sweep --install-timer` is set up, and it is not. `gpu destroy` ends the bill now"
            )
        return {
            "still_billing": True,
            "stopped_cost_usd_per_day": per_day,
            "stopped_ttl_s": ttl,
            "sweep_timer_installed": timer_on,
            "warning": warning,
        }

    def destroy(self, reason: str = "manual", discard_outputs: bool = False) -> dict[str, Any]:
        with self.store.process_lock():
            state = self._reconcile(self.store.load())
            if not state.instance_id:
                self.echo("= nothing to destroy")
                return {"changed": False, "state": state.state}
            if not discard_outputs:
                # Results a job wrote are downloaded first, waking a stopped machine.
                state, left = self._save_job_outputs(state)
                if left:
                    self.store.save(state)
                    raise StateError(
                        f"job {', '.join(left)} wrote results that are not downloaded yet and "
                        "could not be fetched now; destroying the machine would delete them. "
                        "Retry, or run `gpu job pull`; `gpu destroy --discard-outputs` deletes them."
                    )
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
            if apply:
                state, decision = self._protect_job_outputs(state, decision, snap.lock_active(cfg))
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
