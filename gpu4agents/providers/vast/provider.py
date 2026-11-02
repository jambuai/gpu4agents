"""Vast.ai provider adapter.

Owns: authentication, credential declaration, API communication, GPU and
pricing discovery, instance create/start/stop/destroy, network + SSH config,
state mapping and error translation.

Owns nothing about idle policy, budgets, workload semantics, model config,
runtime selection or project configuration - those live in the core.
"""

from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from typing import Any, Iterator, Optional

from ...core.config import RuntimeConfig
from ...core.credentials import CredentialResolver
from ...core.errors import (
    CapacityUnavailableError,
    ProviderError,
    ProviderTimeoutError,
    ProvisioningError,
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
from .client import VastClient, VastNotFound
from .config import API_KEY_ENV, ONSTART_MAX_BYTES, VastOptions

# Executed inside the remote watchdog; stdlib only, no package imports.
WATCHDOG_ACTIONS = '''
import json as _json
import os as _os
import time as _time
import urllib.error as _urlerror
import urllib.request as _urlrequest


def _vast_call(method, path, body=None):
    base = _os.environ.get("VAST_API_BASE", "https://console.vast.ai/api/v0")
    key = _os.environ.get("VAST_API_KEY", "")
    if not key:
        raise RuntimeError("VAST_API_KEY is not available on the instance")
    payload = _json.dumps(body).encode() if body is not None else None
    request = _urlrequest.Request(base.rstrip("/") + path, data=payload, method=method)
    request.add_header("Authorization", "Bearer " + key)
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    last = None
    for attempt in range(5):
        try:
            with _urlrequest.urlopen(request, timeout=30) as response:
                raw = response.read().decode() or "{}"
            try:
                return _json.loads(raw)
            except ValueError:
                return {"raw": raw}
        except _urlerror.HTTPError as exc:
            if exc.code == 404:
                return {"success": True, "already_gone": True}
            last = exc
        except Exception as exc:
            last = exc
        _time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("vast.ai call failed after retries: %s" % last)


def _vast_instance_id(ctx):
    value = ctx.get("instance_id")
    if value:
        return str(value)
    for name in ("VAST_CONTAINERLABEL", "CONTAINER_LABEL", "CONTAINER_ID"):
        label = str(_os.environ.get(name) or "")
        if label.startswith("C."):
            label = label[2:]
        if label.isdigit():
            return label
    raise RuntimeError("cannot determine the vast.ai instance id from inside the container")


def provider_stop(ctx):
    _vast_call("PUT", "/instances/%s/" % _vast_instance_id(ctx), {"state": "stopped"})


def provider_destroy(ctx):
    _vast_call("DELETE", "/instances/%s/" % _vast_instance_id(ctx))
'''


class VastProvider(GPUProvider):
    name = "vast"
    display_name = "Vast.ai"
    required_credentials = [API_KEY_ENV]
    capabilities = ProviderCapabilities(
        stop=True,
        destroy=True,
        persistent_disk=True,
        spot_instances=True,
        ssh=True,
        public_ports=True,
        gpu_selection=True,
        pricing_query=True,
        remote_exec=True,
        logs=True,
        list_instances=True,
        # A stopped Vast instance no longer bills GPU time but still bills
        # storage, so the core may choose to escalate to destroy.
        stop_eliminates_billing=False,
    )
    # The same vLLM as vllm/vllm-openai:v0.32.0, packaged by Vast and cached
    # on many hosts. The Docker Hub image (~12 GB) stalled past the 30-minute
    # boot timeout on several hosts in live runs; this one is usually already on the host.
    preferred_images = {"vllm": "vastai/vllm:v0.32.0-cuda-13.0"}

    def __init__(self, config: RuntimeConfig, credentials: CredentialResolver) -> None:
        super().__init__(config, credentials)
        self.options = VastOptions.from_mapping(config.provider.options)
        self._client: Optional[VastClient] = None

    # -- plumbing -----------------------------------------------------------

    @property
    def client(self) -> VastClient:
        if self._client is None:
            key = self.credentials.get(API_KEY_ENV)
            if not key:
                from ...core.errors import AuthenticationError

                raise AuthenticationError(
                    f"{API_KEY_ENV} is not set. Export it or add it to .env "
                    "(get one at https://cloud.vast.ai/account/)."
                )
            self._client = VastClient(key, api_base=self.options.api_base)
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
            return result  # never reach out to the API without a key

        if config.compute.spot and not self.capabilities.spot_instances:
            result.error("provider does not support spot instances", "compute.spot")
        if config.lifecycle.auto_stop and not self.capabilities.stop:
            result.error("provider cannot stop instances", "lifecycle.auto_stop")
        if config.workload.mode == "remote" and not self.capabilities.remote_exec:
            result.error("provider cannot execute remote workloads", "workload.mode")
        if config.compute.gpu.count > 8:
            result.warn("more than 8 GPUs on a single Vast machine is rare", "compute.gpu.count")

        try:
            user = self.client.whoami()
        except ProviderError as exc:
            result.error(str(exc), "credentials")
            return result

        balance = user.get("credit") or user.get("balance")
        if balance is not None:
            try:
                if float(balance) <= 0:
                    result.error(f"vast.ai account has no credit (balance {balance})", "account")
                elif float(balance) < 5:
                    result.warn(f"vast.ai balance is low ({balance})", "account")
            except (TypeError, ValueError):
                pass

        spec = self.build_spec_preview(config)
        try:
            offers = self.find_offers(spec, limit=5)
        except ProviderError as exc:
            result.warn(f"could not query offers: {exc}", "compute")
            return result
        if not offers:
            result.error(
                "no vast.ai offer matches compute.gpu "
                f"(>= {config.compute.gpu.min_vram_gb} GB VRAM x {config.compute.gpu.count}, "
                f"{config.compute.disk_gb} GB disk"
                + (
                    f", <= ${spec.max_hourly_cost_usd}/h"
                    if spec.max_hourly_cost_usd
                    else ""
                )
                + ")",
                "compute",
            )
        return result

    def build_spec_preview(self, config: RuntimeConfig) -> ComputeSpec:
        """A spec good enough for offer discovery during validation."""
        from ...core.models import GpuSpec

        return ComputeSpec(
            gpu=GpuSpec(
                min_vram_gb=config.compute.gpu.min_vram_gb,
                count=config.compute.gpu.count,
                name_filter=config.compute.gpu.name_filter,
                cuda_min=config.compute.gpu.cuda_min,
            ),
            disk_gb=config.compute.disk_gb,
            spot=config.compute.spot,
            region=config.compute.region,
            max_hourly_cost_usd=config.budget.max_hourly_cost_usd,
        )

    def find_offers(self, spec: ComputeSpec, limit: int = 10) -> list[Offer]:
        query = mapper.build_search_query(spec, self.options)
        raw_offers = self.client.search_offers(query)
        if not raw_offers and spec.gpu.min_vram_gb:
            # Some Vast deployments report gpu_ram in GB; retry unfiltered.
            query = mapper.build_search_query(spec, self.options, include_vram=False)
            raw_offers = self.client.search_offers(query)

        offers = [mapper.map_offer(raw) for raw in raw_offers]
        min_cuda = mapper.effective_min_cuda(spec, self.options)
        offers = [o for o in offers if mapper.offer_matches(o, spec, min_cuda)]
        offers.sort(key=lambda o: o.hourly_cost_usd)
        return offers[:limit]

    def setup(self, spec: ComputeSpec) -> Instance:
        # Fail before renting anything: Vast would refuse every offer anyway.
        if len(spec.onstart.encode()) > ONSTART_MAX_BYTES:
            raise ProvisioningError(
                f"the boot script is {len(spec.onstart.encode())} bytes, over Vast's "
                f"{ONSTART_MAX_BYTES}-byte onstart limit"
            )
        offers = self.find_offers(spec, limit=self.options.search_limit)
        if not offers:
            raise CapacityUnavailableError(
                "no vast.ai capacity matches the requested compute spec "
                f"({spec.gpu.count}x >= {spec.gpu.min_vram_gb} GB VRAM, "
                f"{spec.disk_gb} GB disk)"
            )

        ports = sorted(self._wanted_ports())
        last_error: Optional[Exception] = None

        # Offers go stale between search and create; walk the list.
        for offer in offers[:5]:
            payload = mapper.build_create_payload(
                spec, self.options, ports, spec.onstart, price=offer.hourly_cost_usd
            )
            try:
                created = self.client.create_instance(offer.id, payload)
            except (CapacityUnavailableError, VastNotFound) as exc:
                last_error = exc
                continue

            instance_id = str(created.get("new_contract"))
            instance = self._read_instance(instance_id)
            if instance is None:
                instance = Instance(
                    id=instance_id,
                    provider=self.name,
                    state=InstanceState.PROVISIONING,
                    gpu_name=offer.gpu_name,
                    gpu_count=offer.gpu_count,
                    hourly_cost_usd=offer.hourly_cost_usd,
                    label=spec.label,
                )
            instance.host_id = instance.host_id or offer.host_id
            instance.hourly_cost_usd = instance.hourly_cost_usd or offer.hourly_cost_usd
            return instance

        raise CapacityUnavailableError(
            f"every matching vast.ai offer was taken before provisioning: {last_error}"
        )

    def start(self, instance: Instance) -> Instance:
        self.client.set_instance_state(instance.id, "running")
        deadline = time.time() + 300
        while time.time() < deadline:
            current = self._read_instance(instance.id)
            if current is None:
                raise ProvisioningError(f"instance {instance.id} disappeared while starting")
            if current.state in (InstanceState.READY, InstanceState.BUSY, InstanceState.IDLE):
                return current
            if current.state is InstanceState.FAILED:
                raise ProvisioningError(
                    f"instance {instance.id} failed to start: "
                    f"{current.raw.get('status_msg', 'no detail')}"
                )
            time.sleep(5)
        raise ProviderTimeoutError(f"instance {instance.id} did not start within 5 minutes")

    def run(self, instance: Instance, workload: Workload) -> Execution:
        """Remote execution over SSH (capability: remote_exec)."""
        self.require_capability("remote_exec")
        if not instance.ssh_host or not instance.ssh_port:
            raise ProviderError(
                f"instance {instance.id} has no SSH endpoint yet; retry once it is running"
            )

        inner = " ".join(shlex.quote(part) for part in workload.command)
        if workload.cwd:
            inner = f"cd {shlex.quote(workload.cwd)} && {inner}"
        exports = " ".join(
            f"export {key}={shlex.quote(str(value))};" for key, value in workload.env.items()
        )
        remote_command = f"{exports} {inner}".strip()

        ssh_command = [
            "ssh",
            "-p",
            str(instance.ssh_port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            f"{instance.ssh_user}@{instance.ssh_host}",
            remote_command,
        ]

        execution = Execution(
            id=uuid.uuid4().hex[:12], workload_id=workload.id, started_at=time.time()
        )
        process = subprocess.run(ssh_command, check=False)
        execution.finished_at = time.time()
        execution.returncode = process.returncode
        execution.state = "finished" if process.returncode == 0 else "failed"
        return execution

    def status(self, instance: Instance) -> InstanceStatus:
        raw = self.client.get_instance(instance.id)
        if raw is None:
            return InstanceStatus(
                state=InstanceState.DESTROYED,
                instance=None,
                exists=False,
                message="instance no longer exists on vast.ai",
            )
        current = mapper.map_instance(raw, self._wanted_ports())
        return InstanceStatus(
            state=current.state,
            instance=current,
            exists=True,
            hourly_cost_usd=current.hourly_cost_usd,
            message=str(raw.get("status_msg") or "").strip(),
            raw=raw,
        )

    def stop(self, instance: Instance) -> None:
        try:
            self.client.set_instance_state(instance.id, "stopped")
        except VastNotFound:
            return  # already gone: stopping is idempotent
        except ProviderError as exc:
            message = str(exc).lower()
            if "not running" in message or "already" in message:
                return
            raise

    def destroy(self, instance: Instance) -> None:
        try:
            self.client.destroy_instance(instance.id)
        except VastNotFound:
            return

    def list_instances(self) -> list[Instance]:
        wanted = self._wanted_ports()
        return [
            mapper.map_instance(raw, wanted)
            for raw in self.client.list_instances()
            if is_jambu_label(raw.get("label"))
        ]

    def logs(self, instance: Instance) -> LogStream:
        url = self.client.request_logs(instance.id, tail=800)
        if not url:
            return LogStream(iter(()), source="vast")

        def _lines() -> Iterator[str]:
            # Vast uploads the log asynchronously; give it a few seconds.
            for attempt in range(8):
                try:
                    text = self.client.fetch_url(url)
                except ProviderError:
                    text = ""
                if text:
                    yield from text.splitlines()
                    return
                time.sleep(1.5 * (attempt + 1) / 2)

        return LogStream(_lines(), source=f"vast:{instance.id}")

    def refresh(self, instance: Instance) -> Instance:
        current = self._read_instance(instance.id)
        return current or instance

    # -- watchdog -----------------------------------------------------------

    def watchdog_spec(self, spec_or_instance: Any = None) -> Optional[WatchdogSpec]:
        key = self.credentials.get(API_KEY_ENV) or ""
        return WatchdogSpec(
            actions_source=WATCHDOG_ACTIONS,
            env={"VAST_API_KEY": key, "VAST_API_BASE": self.options.api_base},
            context={"provider": self.name},
        )

    # -- internals ----------------------------------------------------------

    def _read_instance(self, instance_id: str) -> Optional[Instance]:
        raw = self.client.get_instance(instance_id)
        if raw is None:
            return None
        return mapper.map_instance(raw, self._wanted_ports())
