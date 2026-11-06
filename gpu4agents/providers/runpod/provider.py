"""RunPod provider adapter (REST API v2).

A pod is a container on a GPU machine. Its HTTP ports are reached through
RunPod's proxy (https://POD-PORT.proxy.runpod.net), which closes a request
after 100 seconds; streaming responses are not affected. A stopped pod
releases its GPU and, with no volume, costs nothing.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from ...core.config import RuntimeConfig
from ...core.credentials import CredentialResolver
from ...core.errors import (
    AuthenticationError,
    CapacityUnavailableError,
    ConfigError,
    ProviderError,
    ProviderTimeoutError,
    ProvisioningError,
    UnsupportedCapabilityError,
)
from ...core.models import (
    ComputeSpec,
    Execution,
    Instance,
    InstanceState,
    InstanceStatus,
    LogStream,
    Offer,
    ProviderCapabilities,
    ValidationResult,
    Workload,
    is_jambu_label,
)
from ..base import GPUProvider, WatchdogSpec
from . import mapper
from .client import RunpodClient, RunpodConflict, RunpodNotFound
from .config import (
    API_KEY_ENV,
    BILLING_URL,
    KEYS_URL,
    MAX_BODY_BYTES,
    MAX_CREATE_ATTEMPTS,
    REFERRAL_LINK,
    SIGNUP_URL,
    RunpodOptions,
)

# Executed inside the remote watchdog; stdlib only. RunPod puts the pod's id
# in RUNPOD_POD_ID.
WATCHDOG_ACTIONS = '''
import json as _json
import os as _os
import time as _time
import urllib.error as _urlerror
import urllib.request as _urlrequest


def _runpod_call(method, path, body=None):
    base = _os.environ.get("JAMBU_RUNPOD_API_BASE", "https://api.runpod.io")
    key = _os.environ.get("JAMBU_RUNPOD_KEY") or _os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise RuntimeError("the RunPod API key is not available on the pod")
    payload = _json.dumps(body).encode() if body is not None else None
    request = _urlrequest.Request(base.rstrip("/") + path, data=payload, method=method)
    request.add_header("Authorization", "Bearer " + key)
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    last = None
    for attempt in range(6):
        try:
            with _urlrequest.urlopen(request, timeout=30) as response:
                raw = response.read().decode() or "{}"
            try:
                return _json.loads(raw)
            except ValueError:
                return {"raw": raw}
        except _urlerror.HTTPError as exc:
            if exc.code == 404:
                return {"already": True}
            if exc.code == 409:
                # The action does not fit the pod's status: done only if it is
                # already stopped; otherwise let the watchdog escalate.
                pod = _runpod_call("GET", path.split("/action")[0])
                if pod.get("status") in ("EXITED", "TERMINATED") or pod.get("already"):
                    return {"already": True}
                raise RuntimeError("RunPod refused %s (pod status %s)" % (path, pod.get("status")))
            last = exc
        except Exception as exc:
            last = exc
        _time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("RunPod call failed after retries: %s" % last)


def _runpod_pod_id(ctx):
    value = ctx.get("instance_id") or _os.environ.get("RUNPOD_POD_ID")
    if not value:
        raise RuntimeError("cannot determine the RunPod pod id from inside the container")
    return str(value)


def provider_stop(ctx):
    _runpod_call("POST", "/v2/pods/%s/action" % _runpod_pod_id(ctx), {"action": "stop"})


def provider_destroy(ctx):
    _runpod_call("DELETE", "/v2/pods/%s" % _runpod_pod_id(ctx))
'''


class RunpodProvider(GPUProvider):
    name = "runpod"
    display_name = "RunPod"
    required_credentials = [API_KEY_ENV]
    credentials_url = KEYS_URL
    tagline = "many GPU models, billed by the second, prepaid credit"
    signup_url = SIGNUP_URL
    setup_steps = [
        "Create a RunPod account: {signup_url}"
        + ("\n     Signing up through this link supports GPU for Agents, at no cost to you." if REFERRAL_LINK else ""),
        "Add some credit ($10 is plenty to try it): " + BILLING_URL
        + "\n     You pay only while a GPU runs, and idle ones are shut down for you.",
        "Create an API key with All permissions, and copy it: {credentials_url}",
    ]
    capabilities = ProviderCapabilities(
        stop=True,
        destroy=True,
        persistent_disk=False,
        spot_instances=False,
        ssh=False,
        public_ports=True,
        gpu_selection=True,
        pricing_query=True,
        remote_exec=False,
        logs=True,
        list_instances=True,
        stop_eliminates_billing=True,
    )

    def __init__(self, config: RuntimeConfig, credentials: CredentialResolver) -> None:
        super().__init__(config, credentials)
        try:
            self.options = RunpodOptions.from_mapping(config.provider.options)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if self.options.volume_gb:
            # A volume keeps billing while the pod is stopped.
            self.capabilities = ProviderCapabilities(
                **{**type(self).capabilities.__dict__, "persistent_disk": True, "stop_eliminates_billing": False}
            )
        self._client: Optional[RunpodClient] = None

    @classmethod
    def verify_credentials(cls, values: dict[str, str]) -> dict[str, Any]:
        client = RunpodClient(values[API_KEY_ENV])
        try:
            # A REST read proves the key can see pods; the balance is a bonus.
            client.gpu_types(1, "SECURE")
            try:
                balance = client.balance()
            except ProviderError:
                balance = None
        finally:
            client.close()
        return {"balance_usd": balance}

    # -- plumbing -----------------------------------------------------------

    @property
    def client(self) -> RunpodClient:
        if self._client is None:
            key = self.credentials.get(API_KEY_ENV)
            if not key:
                raise AuthenticationError(
                    f"{API_KEY_ENV} is not set. Run `gpu login --provider runpod` "
                    f"(the key comes from {KEYS_URL})."
                )
            self._client = RunpodClient(key, api_base=self.options.api_base, graphql_url=self.options.graphql_url)
        return self._client

    def _wanted_ports(self) -> dict[int, str]:
        ports: dict[int, str] = {}
        if self.config.runtime.engine != "none":
            ports[self.config.runtime.port] = "model"
        if self.config.lifecycle.watchdog.enabled:
            ports[self.config.lifecycle.watchdog.port] = "watchdog"
        return ports

    # -- contract -----------------------------------------------------------

    def validate(self, config: RuntimeConfig) -> ValidationResult:
        result = ValidationResult()
        missing = self.credentials.missing(self.required_credentials)
        if missing:
            result.error(
                f"missing credential(s): {', '.join(missing)}. "
                f"Searched: {', '.join(self.credentials.describe_sources())}",
                "credentials",
            )
            return result
        if config.compute.spot:
            result.error("RunPod's API no longer rents spot pods; set compute.spot: false", "compute.spot")
        if config.workload.mode == "remote":
            result.error("RunPod pods take no SSH commands from gpu; use `gpu job` instead", "workload.mode")
        try:
            balance = self.client.balance()
        except AuthenticationError as exc:
            result.error(str(exc), "credentials")
            return result
        except ProviderError:
            balance = None
        if balance is not None:
            if balance <= 0:
                result.error(f"the RunPod account has no credit (balance {balance})", "account")
            elif balance < 5:
                result.warn(f"the RunPod balance is low ({balance})", "account")
        spec = self._preview(config)
        try:
            offers = self.find_offers(spec, limit=5)
        except AuthenticationError as exc:
            result.error(str(exc), "credentials")
            return result
        except ProviderError as exc:
            result.warn(f"could not query offers: {exc}", "compute")
            return result
        if not offers:
            result.error(
                "no RunPod GPU with free capacity matches compute.gpu "
                f"({config.compute.gpu.count}x >= {config.compute.gpu.min_vram_gb} GB VRAM"
                + (f", <= ${spec.max_hourly_cost_usd}/h" if spec.max_hourly_cost_usd else "")
                + ")",
                "compute",
            )
        return result

    def _preview(self, config: RuntimeConfig) -> ComputeSpec:
        from ...core.models import GpuSpec

        return ComputeSpec(
            gpu=GpuSpec(
                min_vram_gb=config.compute.gpu.min_vram_gb,
                count=config.compute.gpu.count,
                name_filter=config.compute.gpu.name_filter,
                cuda_min=config.compute.gpu.cuda_min,
                compute_cap_min=config.compute.gpu.compute_cap_min,
            ),
            disk_gb=config.compute.disk_gb,
            region=config.compute.region,
            max_hourly_cost_usd=config.budget.max_hourly_cost_usd,
        )

    def find_offers(self, spec: ComputeSpec, limit: int = 10) -> list[Offer]:
        offers: list[Offer] = []
        for cloud in self.options.clouds():
            for gpu in self.client.gpu_types(spec.gpu.count, cloud, min_cuda=mapper.cuda_floor(spec)):
                if not mapper.available(gpu, self.options):
                    continue
                offer = mapper.map_offer(gpu, cloud, spec.gpu.count)
                if offer is not None and mapper.offer_matches(offer, spec):
                    offers.append(offer)
        # Cheapest first; at the same price, the better-stocked one.
        offers.sort(key=lambda o: (o.hourly_cost_usd, -(o.score or 0.0), o.id))
        return offers[:limit]

    def setup(self, spec: ComputeSpec) -> Instance:
        if spec.spot:
            raise UnsupportedCapabilityError("RunPod's API no longer rents spot pods")
        offers = self.find_offers(spec, limit=MAX_CREATE_ATTEMPTS)
        if not offers:
            raise CapacityUnavailableError(
                "no RunPod GPU with free capacity matches the requested compute spec "
                f"({spec.gpu.count}x >= {spec.gpu.min_vram_gb} GB VRAM)"
            )
        ports = sorted(self._wanted_ports())
        extra_env = {
            "JAMBU_RUNPOD_KEY": self.credentials.get(API_KEY_ENV) or "",
            "JAMBU_RUNPOD_API_BASE": self.options.api_base,
        }
        last_error: Optional[Exception] = None
        for offer in offers:
            payload = mapper.build_create_payload(spec, self.options, offer, ports, extra_env)
            size = len(json.dumps(payload).encode())
            if size > MAX_BODY_BYTES:
                raise ProvisioningError(
                    f"the boot request is {size} bytes, over RunPod's {MAX_BODY_BYTES}-byte limit"
                )
            asked_at = time.time()
            try:
                created = self.client.create_pod(payload)
            except CapacityUnavailableError as exc:
                last_error = exc
                continue
            except (AuthenticationError, ProvisioningError):
                raise
            except ProviderError:
                # A create whose answer was lost may still have made a pod:
                # adopt it, or it would bill with nobody tracking it.
                created = self._find_created(spec.label, asked_at)
                if created is None:
                    raise
            instance = mapper.map_instance(created, self._wanted_ports())
            instance.gpu_name = offer.gpu_name
            instance.gpu_count = instance.gpu_count or offer.gpu_count
            instance.hourly_cost_usd = instance.hourly_cost_usd or offer.hourly_cost_usd
            instance.label = instance.label or spec.label
            return instance
        raise CapacityUnavailableError(f"every matching RunPod GPU was taken before the pod started: {last_error}")

    def start(self, instance: Instance) -> Instance:
        try:
            self.client.action(instance.id, "start")
        except RunpodConflict:
            pass  # already starting or running
        except ProviderError as exc:
            if "gpu" in str(exc).lower():
                raise CapacityUnavailableError(
                    f"pod {instance.id} cannot start: its machine has no free GPU now ({exc}). "
                    "Destroy it and set up again to rent another one."
                ) from exc
            raise
        deadline = time.time() + 300
        while time.time() < deadline:
            current = self._read(instance.id)
            if current is None:
                raise ProvisioningError(f"pod {instance.id} disappeared while starting")
            if current.state is InstanceState.READY:
                return current
            if current.state is InstanceState.FAILED:
                raise ProvisioningError(f"pod {instance.id} failed to start")
            time.sleep(5)
        raise ProviderTimeoutError(f"pod {instance.id} did not start within 5 minutes")

    def run(self, instance: Instance, workload: Workload) -> Execution:
        raise UnsupportedCapabilityError("RunPod pods take no SSH commands from gpu; use `gpu job`")

    def status(self, instance: Instance) -> InstanceStatus:
        raw = self.client.get_pod(instance.id)
        if raw is None or mapper.map_state(raw) is InstanceState.DESTROYED:
            return InstanceStatus(
                state=InstanceState.DESTROYED, instance=None, exists=False,
                message="pod no longer exists on RunPod",
            )
        current = mapper.map_instance(raw, self._wanted_ports())
        return InstanceStatus(
            state=current.state,
            instance=current,
            exists=True,
            hourly_cost_usd=current.hourly_cost_usd,
            message=str(raw.get("status") or ""),
            raw=raw,
        )

    def stop(self, instance: Instance) -> None:
        try:
            self.client.action(instance.id, "stop")
        except RunpodNotFound:
            return  # gone
        except RunpodConflict:
            # 409: stop does not fit the pod's status. Fine if it is already
            # stopped; otherwise (locked, say) it is still billing: say so.
            raw = self.client.get_pod(instance.id)
            if raw is None or mapper.map_state(raw) in (InstanceState.STOPPED, InstanceState.DESTROYED):
                return
            raise ProviderError(
                f"RunPod refused to stop pod {instance.id} (status {raw.get('status')}); destroy it instead"
            )

    def destroy(self, instance: Instance) -> None:
        self.client.delete_pod(instance.id)

    def logs(self, instance: Instance) -> LogStream:
        return LogStream(self.client.logs(instance.id), source=f"runpod:{instance.id}")

    def list_instances(self) -> list[Instance]:
        wanted = self._wanted_ports()
        return [
            mapper.map_instance(raw, wanted)
            for raw in self.client.list_pods()
            if is_jambu_label(raw.get("name")) and mapper.map_state(raw) is not InstanceState.DESTROYED
        ]

    def refresh(self, instance: Instance) -> Instance:
        return self._read(instance.id) or instance

    # -- watchdog -----------------------------------------------------------

    def watchdog_spec(self, spec_or_instance: Any = None) -> Optional[WatchdogSpec]:
        # The key travels in the pod's environment (setup adds it), not
        # through the boot script.
        return WatchdogSpec(actions_source=WATCHDOG_ACTIONS, env={}, context={"provider": self.name})

    # -- internals ----------------------------------------------------------

    def _find_created(self, label: str, since: float) -> Optional[dict]:
        for attempt in range(3):
            try:
                for raw in self.client.list_pods():
                    created_at = mapper._timestamp(raw.get("createdAt")) or 0.0
                    if raw.get("name") == label and created_at >= since - 120:
                        return raw
            except ProviderError:
                pass
            time.sleep(5 * (attempt + 1))
        return None

    def _read(self, pod_id: str) -> Optional[Instance]:
        raw = self.client.get_pod(pod_id)
        return mapper.map_instance(raw, self._wanted_ports()) if raw else None
