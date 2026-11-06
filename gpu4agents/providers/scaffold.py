"""`gpu providers new NAME`: the starting file for an adapter written outside this package.

The template compiles and registers as is. Every method that needs the
provider's API calls ``todo(...)``, which fails with a normal gpu error (exit 9)
naming the file and what is left to write.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.errors import ConfigError

PLUGIN_DIR = "gpu_providers"

TEMPLATE = '''\
"""{display} adapter for GPU for Agents (gpu4agents).

Written against the provider contract: gpu4agents/providers/base.py.
Guide: https://github.com/jambuai/gpu4agents/blob/main/docs/ADDING_A_PROVIDER.md

Selected in gpu.yaml with:

    gpu_runtime:
      provider:
        name: {name}
        plugin: ./{plugin_dir}/{name}.py:{cls}

Replace every todo(...) call with real code, reading the provider's API docs. Keep everything provider-specific in this file: the core only sees
the normalized models (Instance, Offer, InstanceState, ...) and errors.
"""

from __future__ import annotations

from typing import Any

import httpx

from gpu4agents.core.errors import (
    AuthenticationError,
    CapacityUnavailableError,
    ProviderError,
    ProviderTimeoutError,
    ProvisioningError,
    UnsupportedCapabilityError,
)
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

API_BASE = "https://api.example.com/v1"  # TODO: the provider's API root
API_KEY_ENV = "{env}"


def todo(what: str):
    """A part of this adapter that is not written yet: a clean error, not a traceback."""
    raise UnsupportedCapabilityError(f"the {display} adapter does not do this yet: {{what}} (write it in {{__file__}})")

# Runs inside the watchdog on the rented machine, so the machine can stop or
# delete itself when idle even if nobody's laptop is on. Python stdlib only.
# The key arrives as an environment variable (see watchdog_spec below), and
# ctx["instance_id"] holds this machine's id once the CLI has seen it.
WATCHDOG_ACTIONS = \'\'\'
import json as _json
import os as _os
import urllib.request as _urlrequest


def _call(method, path, body=None):
    request = _urlrequest.Request(
        _os.environ.get("{env_base}", "{api_base}").rstrip("/") + path,
        data=_json.dumps(body).encode() if body is not None else None,
        method=method,
    )
    request.add_header("Authorization", "Bearer " + _os.environ["{env}"])
    request.add_header("Content-Type", "application/json")
    with _urlrequest.urlopen(request, timeout=30) as response:
        return response.read()


def provider_stop(ctx):
    # TODO: the provider's "stop instance" call
    _call("POST", "/instances/%s/stop" % ctx["instance_id"])


def provider_destroy(ctx):
    # TODO: the provider's "delete instance" call
    _call("DELETE", "/instances/%s" % ctx["instance_id"])
\'\'\'


class {cls}(GPUProvider):
    name = "{name}"
    display_name = "{display}"
    required_credentials = [API_KEY_ENV]
    credentials_url = ""  # TODO: the page where users create an API key
    # Declare only what the API really supports; the core reads these flags.
    capabilities = ProviderCapabilities(
        stop=True,
        destroy=True,
        ssh=True,
        public_ports=True,
        gpu_selection=True,
        pricing_query=True,
        remote_exec=False,
        logs=False,
        list_instances=True,
        # False if a stopped machine still bills (disk); the core then may destroy it.
        stop_eliminates_billing=False,
    )

    # -- HTTP ---------------------------------------------------------------

    @classmethod
    def _request(cls, key: str, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = httpx.request(
                method,
                API_BASE + path,
                headers={{"Authorization": f"Bearer {{key}}"}},
                timeout=30,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"{display} timed out: {{exc}}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{display} request failed: {{exc}}") from exc
        if response.status_code in (401, 403):
            raise AuthenticationError(f"{display} rejected the API key")
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise ProviderError(f"{display} {{method}} {{path}}: {{response.status_code}} {{response.text[:300]}}")
        return response.json() if response.content else {{}}

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        key = self.credentials.get(API_KEY_ENV)
        if not key:
            raise AuthenticationError(f"{{API_KEY_ENV}} is not set; run `gpu login --provider {name}`")
        return self._request(key, method, path, **kwargs)

    @classmethod
    def verify_credentials(cls, values: dict[str, str]) -> dict[str, Any]:
        """Called by `gpu login` before saving. Return only safe fields (no email, no name)."""
        todo("verify_credentials: call an account endpoint, e.g. GET /account, and return {{'balance_usd': ...}}")

    # -- the contract ---------------------------------------------------------

    def validate(self, config) -> ValidationResult:
        result = ValidationResult()
        missing = self.credentials.missing(self.required_credentials)
        if missing:
            result.error(f"missing credential(s): {{', '.join(missing)}}; run `gpu login --provider {name}`", "credentials")
        return result

    def find_offers(self, spec: ComputeSpec, limit: int = 10) -> list[Offer]:
        """GPUs for rent that fit spec.gpu (min_vram_gb, count), spec.disk_gb and
        spec.max_hourly_cost_usd, cheapest first. Skip offers whose host_id is in
        spec.exclude_hosts."""
        todo("find_offers: list GPU offers and map each to Offer(...)")

    def setup(self, spec: ComputeSpec) -> Instance:
        """Rent one machine: image spec.image, env spec.env, disk spec.disk_gb,
        label spec.label, publish every port in spec.ports, and run spec.onstart
        (a bash script that starts the model server and the watchdog) at boot.
        Raise CapacityUnavailableError when nothing fits."""
        todo("setup: create the instance and return self._to_instance(payload)")

    def start(self, instance: Instance) -> Instance:
        todo("start: resume a stopped instance, wait until it runs, return it")

    def run(self, instance: Instance, workload: Workload) -> Execution:
        self.require_capability("remote_exec")
        todo("run: execute workload.command on the machine")

    def status(self, instance: Instance) -> InstanceStatus:
        payload = self._call("GET", f"/instances/{{instance.id}}")
        if payload is None:
            return InstanceStatus(state=InstanceState.DESTROYED, instance=None, exists=False)
        current = self._to_instance(payload)
        return InstanceStatus(state=current.state, instance=current, hourly_cost_usd=current.hourly_cost_usd)

    def stop(self, instance: Instance) -> None:
        """Must succeed when the machine is already stopped or gone."""
        todo("stop: the provider's stop call")

    def destroy(self, instance: Instance) -> None:
        """Must succeed when the machine is already gone."""
        todo("destroy: the provider's delete call")

    def logs(self, instance: Instance) -> LogStream:
        return LogStream(iter(()), source=self.name)

    def list_instances(self) -> list[Instance]:
        """Only machines whose label starts with "jambu-": never touch anything else."""
        todo("list_instances: list instances and keep those where is_jambu_label(label)")

    def watchdog_spec(self, spec_or_instance=None) -> WatchdogSpec:
        return WatchdogSpec(
            actions_source=WATCHDOG_ACTIONS,
            env={{API_KEY_ENV: self.credentials.get(API_KEY_ENV) or ""}},
            context={{"provider": self.name}},
        )

    # -- mapping ----------------------------------------------------------------

    def _to_instance(self, payload: dict) -> Instance:
        """Provider payload -> Instance. Map the provider's status onto InstanceState
        (PROVISIONING, STARTING, READY, STOPPED, FAILED, DESTROYED), and each
        published port onto endpoints: "model" for the runtime port, "watchdog"
        for the watchdog port, as Endpoint(name, url, internal_port, external_port)."""
        todo("_to_instance: map the provider's instance payload")
'''


# Adapters that ship with gpu4agents (providers/registry.py).
BUILT_IN = ("vast", "runpod", "lambda")


def plugin_spec(name: str) -> str:
    return f"./{PLUGIN_DIR}/{name}.py:{class_name(name)}"


def class_name(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^a-zA-Z0-9]+", name) if part) + "Provider"


def write_template(name: str, directory: Path, display_name: str = "", force: bool = False) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ConfigError("provider name: lowercase letters, digits, - or _, starting with a letter")
    if name in BUILT_IN:
        raise ConfigError(f"{name} is built in; pick another name")
    path = directory / PLUGIN_DIR / f"{name}.py"
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists (use --force to overwrite)")
    display = display_name or name.replace("-", " ").replace("_", " ").title()
    env = re.sub(r"[^A-Z0-9]", "_", name.upper())
    text = TEMPLATE.format(
        name=name,
        cls=class_name(name),
        display=display,
        env=f"{env}_API_KEY",
        env_base=f"{env}_API_BASE",
        api_base="https://api.example.com/v1",
        plugin_dir=PLUGIN_DIR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
