"""Lambda Cloud provider adapter.

A Lambda machine is a whole VM, billed by the minute from launch to
terminate. It cannot be stopped, only terminated, so the core destroys it
where it would stop another provider's machine. The model image runs under
Docker on the VM (see ``mapper.build_user_data``).
"""

from __future__ import annotations

import secrets
import shlex
import socket
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ...core.config import RuntimeConfig
from ...core.credentials import CredentialResolver
from ...core.errors import (
    AuthenticationError,
    CapacityUnavailableError,
    ConfigError,
    ProviderError,
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
from ...core.paths import gpu_home
from ..base import GPUProvider, WatchdogSpec
from . import mapper
from .client import LambdaClient
from .config import (
    API_KEY_ENV,
    BILLING_URL,
    KEYS_URL,
    MAX_LAUNCH_ATTEMPTS,
    NONCE_TAG,
    RULESET_NAME_PREFIX,
    SIGNUP_URL,
    SSH_KEY_FILE,
    SSH_KEY_NAME_PREFIX,
    LambdaOptions,
)

# Executed inside the remote watchdog; stdlib only. A Lambda machine cannot
# be stopped, and powering it off from inside keeps it billing, so both
# actions terminate it through the API.
WATCHDOG_ACTIONS = '''
import json as _json
import os as _os
import time as _time
import urllib.error as _urlerror
import urllib.request as _urlrequest


def _lambda_call(method, path, body=None):
    base = _os.environ.get("JAMBU_LAMBDA_API_BASE", "https://cloud.lambda.com/api/v1")
    key = _os.environ.get("JAMBU_LAMBDA_KEY", "")
    if not key:
        raise RuntimeError("the Lambda API key is not available on the instance")
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
                return {"already_gone": True}
            last = exc
        except Exception as exc:
            last = exc
        _time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("Lambda call failed after retries: %s" % last)


def _lambda_instance_id(ctx):
    value = ctx.get("instance_id")
    if value:
        return str(value)
    nonce = _os.environ.get("JAMBU_LAMBDA_NONCE", "")
    if nonce:
        listing = _lambda_call("GET", "/instances").get("data") or []
        for item in listing:
            for tag in item.get("tags") or []:
                if tag.get("key") == "__NONCE_TAG__" and tag.get("value") == nonce:
                    return str(item.get("id"))
    raise RuntimeError("cannot determine this Lambda instance's id")


def provider_destroy(ctx):
    _lambda_call("POST", "/instance-operations/terminate", {"instance_ids": [_lambda_instance_id(ctx)]})


def provider_stop(ctx):
    provider_destroy(ctx)
'''.replace("__NONCE_TAG__", NONCE_TAG)


class LambdaProvider(GPUProvider):
    name = "lambda"
    display_name = "Lambda"
    required_credentials = [API_KEY_ENV]
    credentials_url = KEYS_URL
    tagline = "data-center GPUs (A100, H100, B200), billed by the minute to a card"
    signup_url = SIGNUP_URL
    setup_steps = [
        "Create a Lambda account: {signup_url}",
        "Add a payment card: " + BILLING_URL
        + "\n     Lambda bills each week for the minutes a GPU ran, and idle ones are shut down for you.",
        "Create an API key and copy it: {credentials_url}",
    ]
    capabilities = ProviderCapabilities(
        stop=False,
        destroy=True,
        persistent_disk=False,
        spot_instances=False,
        ssh=True,
        public_ports=True,
        gpu_selection=True,
        pricing_query=True,
        remote_exec=True,
        logs=False,
        list_instances=True,
        stop_eliminates_billing=True,
    )

    def __init__(self, config: RuntimeConfig, credentials: CredentialResolver) -> None:
        super().__init__(config, credentials)
        try:
            self.options = LambdaOptions.from_mapping(config.provider.options)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        self._client: Optional[LambdaClient] = None

    @classmethod
    def verify_credentials(cls, values: dict[str, str]) -> dict[str, Any]:
        client = LambdaClient(values[API_KEY_ENV])
        try:
            client.instance_types()
        finally:
            client.close()
        # Lambda bills afterwards, by card: there is no balance to show.
        return {}

    # -- plumbing -----------------------------------------------------------

    @property
    def client(self) -> LambdaClient:
        if self._client is None:
            key = self.credentials.get(API_KEY_ENV)
            if not key:
                raise AuthenticationError(
                    f"{API_KEY_ENV} is not set. Run `gpu login --provider lambda` "
                    f"(the key comes from {KEYS_URL})."
                )
            self._client = LambdaClient(key, api_base=self.options.api_base)
        return self._client

    def _wanted_ports(self) -> dict[int, str]:
        ports: dict[int, str] = {}
        if self.config.runtime.engine != "none":
            ports[self.config.runtime.port] = "model"
        if self.config.lifecycle.watchdog.enabled:
            ports[self.config.lifecycle.watchdog.port] = "watchdog"
        return ports

    @staticmethod
    def key_path() -> Path:
        return gpu_home() / SSH_KEY_FILE

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
            result.error("Lambda has no spot machines; set compute.spot: false", "compute.spot")
        if config.workload.mode == "remote" and not shutil.which("ssh"):
            result.error("workload.mode remote needs the ssh command on this computer", "workload.mode")
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
                "no Lambda machine with free capacity matches compute.gpu "
                f"({config.compute.gpu.count}x >= {config.compute.gpu.min_vram_gb} GB VRAM, "
                f"{config.compute.disk_gb} GB disk). Lambda types have a fixed GPU count "
                "(1, 2, 4 or 8) and run out often; try another count or later.",
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
        offers = [o for o in mapper.map_offers(self.client.instance_types(), self.options) if mapper.offer_matches(o, spec)]
        offers.sort(key=lambda o: (o.hourly_cost_usd, o.id))
        return offers[:limit]

    def setup(self, spec: ComputeSpec) -> Instance:
        if spec.spot:
            raise UnsupportedCapabilityError("Lambda has no spot machines")
        offers = self.find_offers(spec, limit=MAX_LAUNCH_ATTEMPTS * 3)
        if not offers:
            raise CapacityUnavailableError(
                "no Lambda machine with free capacity matches the requested compute spec "
                f"({spec.gpu.count}x >= {spec.gpu.min_vram_gb} GB VRAM, {spec.disk_gb} GB disk)"
            )
        key_name = self._ensure_ssh_key()
        ports = sorted(self._wanted_ports())
        nonce = secrets.token_hex(8)
        env = {str(k): str(v) for k, v in spec.env.items()}
        env.update(
            {
                "JAMBU_LAMBDA_NONCE": nonce,
                "JAMBU_LAMBDA_KEY": self.credentials.get(API_KEY_ENV) or "",
                "JAMBU_LAMBDA_API_BASE": self.options.api_base,
            }
        )
        user_data = mapper.build_user_data(spec, env, ports)

        last_error: Optional[Exception] = None
        for offer in offers[:MAX_LAUNCH_ATTEMPTS]:
            type_name, region = mapper.split_offer_id(offer.id)
            payload: dict[str, Any] = {
                "region_name": region,
                "instance_type_name": type_name,
                "ssh_key_names": [key_name],
                "name": spec.label[:64],
                "user_data": user_data,
                "tags": [{"key": NONCE_TAG, "value": nonce}],
            }
            if self.options.image_family:
                payload["image"] = {"family": self.options.image_family}
            try:
                if ports:
                    payload["firewall_rulesets"] = [{"id": self._ensure_ruleset(region, ports)}]
            except (AuthenticationError, ProvisioningError):
                raise
            except ProviderError as exc:
                last_error = exc  # this region refused the ruleset: try the next one
                continue
            try:
                instance_id = self.client.launch(payload)
            except CapacityUnavailableError as exc:
                last_error = exc
                continue
            except ProviderError as exc:
                # A launch whose answer was lost may still have made a machine:
                # find it by its nonce, or it would bill with nobody tracking it.
                instance_id = self._find_by_nonce(nonce)
                if instance_id is None:
                    raise
                del exc
            current = self._read_instance(instance_id)
            instance = current or Instance(
                id=instance_id,
                provider=self.name,
                state=InstanceState.PROVISIONING,
                label=spec.label,
            )
            instance.gpu_name = instance.gpu_name or offer.gpu_name
            instance.gpu_count = instance.gpu_count or offer.gpu_count
            instance.hourly_cost_usd = instance.hourly_cost_usd or offer.hourly_cost_usd
            return instance
        raise CapacityUnavailableError(f"every matching Lambda machine was taken before launch: {last_error}")

    def start(self, instance: Instance) -> Instance:
        raise UnsupportedCapabilityError("a Lambda machine cannot be stopped or started, only terminated")

    def run(self, instance: Instance, workload: Workload) -> Execution:
        self.require_capability("remote_exec")
        if not instance.ssh_host:
            raise ProviderError(f"instance {instance.id} has no address yet; retry once it is running")
        inner = " ".join(shlex.quote(part) for part in workload.command)
        if workload.cwd:
            inner = f"cd {shlex.quote(workload.cwd)} && {inner}"
        exports = " ".join(f"export {k}={shlex.quote(str(v))};" for k, v in workload.env.items())
        command = [
            "ssh",
            "-i",
            str(self.key_path()),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            f"{instance.ssh_user}@{instance.ssh_host}",
            f"{exports} {inner}".strip(),
        ]
        execution = Execution(id=uuid.uuid4().hex[:12], workload_id=workload.id, started_at=time.time())
        process = subprocess.run(command, check=False)
        execution.finished_at = time.time()
        execution.returncode = process.returncode
        execution.state = "finished" if process.returncode == 0 else "failed"
        return execution

    def status(self, instance: Instance) -> InstanceStatus:
        raw = self.client.get_instance(instance.id)
        if raw is None:
            return InstanceStatus(
                state=InstanceState.DESTROYED, instance=None, exists=False,
                message="instance no longer exists on Lambda",
            )
        current = self._mapped(raw)
        if current.state is InstanceState.DESTROYED:
            return InstanceStatus(state=current.state, instance=current, exists=False, message=str(raw.get("status")))
        message = str(raw.get("status") or "")
        if message == "active" and current.state is InstanceState.STARTING:
            message = "machine up; pulling the model image and starting the container"
        return InstanceStatus(
            state=current.state,
            instance=current,
            exists=True,
            hourly_cost_usd=current.hourly_cost_usd,
            message=message,
            raw=raw,
        )

    def stop(self, instance: Instance) -> None:
        raise UnsupportedCapabilityError("a Lambda machine cannot be stopped, only terminated")

    def destroy(self, instance: Instance) -> None:
        self.client.terminate(instance.id)

    def logs(self, instance: Instance) -> LogStream:
        raise UnsupportedCapabilityError("Lambda has no log API; `gpu logs` reads the machine's own logs")

    def list_instances(self) -> list[Instance]:
        wanted = self._wanted_ports()
        return [
            mapper.map_instance(raw, wanted)
            for raw in self.client.list_instances()
            if is_jambu_label(raw.get("name")) and mapper.map_state(raw) is not InstanceState.DESTROYED
        ]

    def refresh(self, instance: Instance) -> Instance:
        return self._read_instance(instance.id) or instance

    # -- watchdog -----------------------------------------------------------

    def watchdog_spec(self, spec_or_instance: Any = None) -> Optional[WatchdogSpec]:
        # The key and nonce travel in the container's environment (setup
        # writes them into user_data), not through the boot script.
        return WatchdogSpec(actions_source=WATCHDOG_ACTIONS, env={}, context={"provider": self.name})

    # -- internals ----------------------------------------------------------

    def _read_instance(self, instance_id: str) -> Optional[Instance]:
        raw = self.client.get_instance(instance_id)
        return self._mapped(raw) if raw else None

    def _mapped(self, raw: dict) -> Instance:
        """The instance, STARTING until its container is up.

        Lambda says "active" once the VM boots, but the model image is pulled
        and started after that, by the boot script. The watchdog starts first
        inside the container, so its port answering means the container runs.
        """
        current = mapper.map_instance(raw, self._wanted_ports())
        watchdog = current.endpoints.get("watchdog")
        if current.state is InstanceState.READY and watchdog is not None:
            if not self._port_open(current.host or "", watchdog.external_port):
                current.state = InstanceState.STARTING
        return current

    @staticmethod
    def _port_open(host: str, port: int, timeout: float = 3.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _find_by_nonce(self, nonce: str) -> Optional[str]:
        for attempt in range(3):
            try:
                for raw in self.client.list_instances():
                    if mapper.instance_nonce(raw) == nonce:
                        return str(raw.get("id"))
            except ProviderError:
                pass
            time.sleep(5 * (attempt + 1))
        return None

    def _ensure_ssh_key(self) -> str:
        """The name of gpu's SSH key on Lambda, creating the key pair the first time."""
        path = self.key_path()
        public = path.with_name(path.name + ".pub")
        if not public.is_file():
            if not shutil.which("ssh-keygen"):
                raise ConfigError("Lambda needs an SSH key and ssh-keygen is not installed")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.chmod(0o700)
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "gpu4agents", "-f", str(path)],
                check=True,
            )
        public_key = public.read_text().strip()
        for key in self.client.ssh_keys():
            if str(key.get("public_key") or "").split()[:2] == public_key.split()[:2]:
                return str(key.get("name"))
        name = SSH_KEY_NAME_PREFIX + secrets.token_hex(4)
        self.client.add_ssh_key(name, public_key)
        return name

    def _ensure_ruleset(self, region: str, ports: list[int]) -> str:
        """A firewall ruleset in ``region`` opening SSH and ``ports``; reused across launches."""
        name = RULESET_NAME_PREFIX + "-".join(str(p) for p in ports)
        for ruleset in self.client.firewall_rulesets():
            where = ruleset.get("region")
            where = where.get("name") if isinstance(where, dict) else where
            if ruleset.get("name") == name and where == region:
                return str(ruleset.get("id"))
        created = self.client.create_firewall_ruleset(
            name, region, mapper.firewall_rules(ports, self.options.source_network)
        )
        return str(created.get("id"))
