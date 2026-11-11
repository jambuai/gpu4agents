"""An in-memory provider used to exercise the core without touching a network."""

from __future__ import annotations

import time
from typing import Callable, Optional

from gpu4agents.core.errors import CapacityUnavailableError, ProvisioningError
from gpu4agents.core.models import (
    ComputeSpec,
    Endpoint,
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
from gpu4agents.providers.base import GPUProvider, WatchdogSpec

FAKE_ACTIONS = '''
def provider_stop(ctx):
    return None


def provider_destroy(ctx):
    return None
'''


class FakeProvider(GPUProvider):
    name = "fake"
    display_name = "Fake"
    required_credentials: list[str] = []
    capabilities = ProviderCapabilities(
        stop=True,
        destroy=True,
        ssh=True,
        public_ports=True,
        gpu_selection=True,
        pricing_query=True,
        remote_exec=True,
        logs=True,
        spot_instances=True,
        stop_eliminates_billing=True,
        list_instances=True,
        boot_progress=True,
    )

    def __init__(self, config, credentials) -> None:
        super().__init__(config, credentials)
        self.instances: dict[str, Instance] = {}
        self.counter = 0
        self.calls: list[tuple[str, str]] = []
        self.offer_price = 0.42
        self.fail_setup: Optional[Exception] = None
        self.no_offers = False
        # The marketplace: one offer per machine, cheapest first.
        self.hosts = ["m1", "m2", "m3", "m4", "m5"]
        self.host_prices: dict[str, float] = {}
        self.host_gpus: dict[str, str] = {}
        # Machines that fail to boot, and how: "failed" (the provider reports
        # FAILED), "vanish" (the instance disappears) or "loading" (it never
        # reaches running).
        self.failing_hosts: set[str] = set()
        self.host_failure = "failed"
        # What a "loading" machine reports on its Nth status poll, as Vast
        # reports the image pull's latest line.
        self.boot_message: Optional[Callable[[int], str]] = None
        self.boot_polls = 0
        self.rented_hosts: list[str] = []
        self.searches: list[list[str]] = []
        # instance id -> the boot script, for providers that keep it (Vast.ai).
        self.boot_scripts: dict[str, str] = {}

    def fail_first_hosts(self, count: int, how: str = "failed") -> None:
        """Script the first ``count`` machines rented to fail to boot."""
        self.failing_hosts = set(self.hosts[:count])
        self.host_failure = how

    # -- contract -------------------------------------------------------

    def validate(self, config) -> ValidationResult:
        return ValidationResult()

    def find_offers(self, spec: ComputeSpec, limit: int = 10) -> list[Offer]:
        self.searches.append(list(spec.exclude_hosts))
        if self.no_offers:
            return []
        offers = [
            Offer(
                id=f"offer-{host}",
                gpu_name=self.host_gpus.get(host, "RTX 4090"),
                gpu_count=spec.gpu.count,
                gpu_vram_gb=float(spec.gpu.min_vram_gb),
                disk_gb=float(spec.disk_gb),
                hourly_cost_usd=self.host_prices.get(host, self.offer_price),
                host_id=host,
            )
            for host in self.hosts
            if host not in spec.exclude_hosts
        ]
        if spec.gpu.name_filter:
            offers = [o for o in offers if spec.gpu.name_filter.lower() in o.gpu_name.lower()]
        # Like a real search, a pinned (approved) quote is held to its price.
        if spec.max_hourly_cost_usd is not None and spec.preferred_offer_id:
            offers = [o for o in offers if o.hourly_cost_usd <= spec.max_hourly_cost_usd]
        offers.sort(key=lambda o: (o.id != spec.preferred_offer_id, o.hourly_cost_usd))
        return offers[:limit]

    def setup(self, spec: ComputeSpec) -> Instance:
        self.last_spec = spec
        if self.fail_setup is not None:
            self.counter += 1
            instance_id = f"i-{self.counter}"
            self.instances[instance_id] = Instance(
                id=instance_id, provider=self.name, state=InstanceState.PROVISIONING
            )
            raise self.fail_setup
        offers = self.find_offers(spec)
        if not offers:
            raise CapacityUnavailableError("no capacity")
        offer = offers[0]
        self.counter += 1
        instance_id = f"i-{self.counter}"
        failing = offer.host_id in self.failing_hosts
        state = InstanceState.READY
        if failing:
            state = InstanceState.STARTING if self.host_failure == "loading" else InstanceState.FAILED
        instance = Instance(
            id=instance_id,
            provider=self.name,
            state=state,
            created_at=time.time(),
            gpu_name=offer.gpu_name,
            gpu_count=spec.gpu.count,
            hourly_cost_usd=offer.hourly_cost_usd,
            host_id=offer.host_id,
            host="127.0.0.1",
            ssh_host="ssh.fake",
            ssh_port=2222,
            endpoints={
                "model": Endpoint("model", "http://127.0.0.1:9", 8000, 9),
                "watchdog": Endpoint("watchdog", "http://127.0.0.1:9", 8777, 9),
            },
            label=spec.label,
        )
        self.instances[instance_id] = instance
        self.rented_hosts.append(offer.host_id)
        self.calls.append(("setup", instance_id))
        return instance

    def start(self, instance: Instance) -> Instance:
        stored = self.instances.get(instance.id)
        if stored is None:
            raise ProvisioningError("gone")
        stored.state = InstanceState.READY
        self.calls.append(("start", instance.id))
        return stored

    def run(self, instance: Instance, workload: Workload) -> Execution:
        self.calls.append(("run", instance.id))
        return Execution(
            id="e1",
            workload_id=workload.id,
            started_at=time.time(),
            finished_at=time.time(),
            returncode=0,
            state="finished",
        )

    def status(self, instance: Instance) -> InstanceStatus:
        stored = self.instances.get(instance.id)
        if stored is not None and stored.host_id in self.failing_hosts:
            if self.host_failure == "vanish":
                self.instances.pop(instance.id)
                stored = None
        if stored is None:
            return InstanceStatus(state=InstanceState.DESTROYED, exists=False)
        message = ""
        if stored.state is InstanceState.STARTING and self.boot_message is not None:
            message = self.boot_message(self.boot_polls)
            self.boot_polls += 1
        if stored.state is InstanceState.FAILED:
            message = (
                'Error response from daemon: Get "https://registry-1.docker.io/v2/": '
                "net/http: TLS handshake timeout"
            )
        return InstanceStatus(
            state=stored.state,
            instance=stored,
            exists=True,
            hourly_cost_usd=stored.hourly_cost_usd,
            message=message,
        )

    def stop(self, instance: Instance) -> None:
        self.calls.append(("stop", instance.id))
        stored = self.instances.get(instance.id)
        if stored is not None:
            stored.state = InstanceState.STOPPED

    def destroy(self, instance: Instance) -> None:
        self.calls.append(("destroy", instance.id))
        self.instances.pop(instance.id, None)

    def list_instances(self) -> list[Instance]:
        self.calls.append(("list", ""))
        return [i for i in self.instances.values() if is_jambu_label(i.label)]

    def logs(self, instance: Instance) -> LogStream:
        return LogStream(iter(["line one", "line two"]), source="fake")

    def boot_script(self, instance: Instance):
        return self.boot_scripts.get(instance.id)

    def watchdog_spec(self, spec_or_instance=None) -> WatchdogSpec:
        return WatchdogSpec(actions_source=FAKE_ACTIONS, env={}, context={})
