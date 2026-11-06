"""Translation between Lambda Cloud payloads and the normalized domain model."""

from __future__ import annotations

import base64
import gzip
import re
import shlex
import time
from datetime import datetime
from typing import Any, Optional

from ...core.models import ComputeSpec, Endpoint, Instance, InstanceState, Offer
from ..gpus import compute_cap
from .config import NONCE_TAG, LambdaOptions

PROVIDER_NAME = "lambda"

_STATE_MAP = {
    "booting": InstanceState.PROVISIONING,
    "active": InstanceState.READY,
    # A health check failed; it may pass again, so this is not FAILED.
    "unhealthy": InstanceState.UNKNOWN,
    "terminating": InstanceState.DESTROYED,
    "terminated": InstanceState.DESTROYED,
    "preempted": InstanceState.DESTROYED,
}

# VRAM per GPU in GB, from Lambda's own table, for a description without "(NN GB".
_VRAM_GB = {"b200": 180, "gh200": 96, "h100": 80, "a100": 40, "a6000": 48, "a10": 24, "rtx6000": 24, "v100": 16}


def map_state(raw: dict[str, Any]) -> InstanceState:
    return _STATE_MAP.get(str(raw.get("status") or "").lower(), InstanceState.UNKNOWN)


def vram_gb(instance_type: dict[str, Any]) -> float:
    text = f"{instance_type.get('gpu_description') or ''} {instance_type.get('description') or ''}"
    match = re.search(r"\((\d+(?:\.\d+)?)\s*GB", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    key = str(instance_type.get("name") or "").lower()
    for needle in sorted(_VRAM_GB, key=len, reverse=True):
        if needle in key:
            return float(_VRAM_GB[needle])
    return 0.0


def gpu_name(instance_type: dict[str, Any]) -> str:
    """ "8x H100 (80 GB SXM5)" -> "H100 (80 GB SXM5)"."""
    text = str(instance_type.get("gpu_description") or instance_type.get("description") or "")
    return re.sub(r"^\s*\d+\s*x\s*", "", text).strip() or str(instance_type.get("name") or "")


def offer_id(type_name: str, region: str) -> str:
    return f"{type_name}@{region}"


def split_offer_id(value: str) -> tuple[str, str]:
    type_name, _, region = value.partition("@")
    return type_name, region


def map_offers(catalog: dict[str, Any], options: LambdaOptions) -> list[Offer]:
    offers: list[Offer] = []
    for entry in catalog.values():
        kind = entry.get("instance_type") or {}
        if not options.allow_arm64 and str(kind.get("architecture") or "x86_64") != "x86_64":
            continue
        specs = kind.get("specs") or {}
        name = str(kind.get("name") or "")
        for region in entry.get("regions_with_capacity_available") or []:
            region_name = str(region.get("name") or "")
            offers.append(
                Offer(
                    id=offer_id(name, region_name),
                    gpu_name=gpu_name(kind),
                    gpu_count=int(specs.get("gpus") or 0),
                    gpu_vram_gb=vram_gb(kind),
                    disk_gb=float(specs.get("storage_gib") or 0),
                    hourly_cost_usd=round(float(kind.get("price_cents_per_hour") or 0) / 100.0, 4),
                    region=region_name,
                    compute_cap=compute_cap(f"{name} {gpu_name(kind)}"),
                    raw={"instance_type": kind, "region": region},
                )
            )
    return offers


def offer_matches(offer: Offer, spec: ComputeSpec) -> bool:
    # A Lambda type has a fixed GPU count; ask for exactly what was configured.
    if offer.gpu_count != spec.gpu.count:
        return False
    if spec.gpu.min_vram_gb and offer.gpu_vram_gb + 0.5 < float(spec.gpu.min_vram_gb):
        return False
    if offer.disk_gb and offer.disk_gb + 0.5 < float(spec.disk_gb):
        return False
    min_cc = spec.gpu.compute_cap_min
    if min_cc and offer.compute_cap is not None and offer.compute_cap + 1e-6 < float(min_cc):
        return False
    if spec.max_hourly_cost_usd and offer.hourly_cost_usd > float(spec.max_hourly_cost_usd):
        return False
    if spec.gpu.name_filter and spec.gpu.name_filter.lower() not in (
        offer.gpu_name + " " + offer.id
    ).lower():
        return False
    if spec.region:
        wanted = [r.strip().lower() for r in str(spec.region).split(",") if r.strip()]
        if wanted and not any((offer.region or "").lower().startswith(r) for r in wanted):
            return False
    return True


def _created_at(raw: dict[str, Any]) -> float:
    for key in ("first_healthy", "created"):
        value = raw.get(key)
        if value:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
    return time.time()


def map_instance(raw: dict[str, Any], wanted_ports: dict[int, str]) -> Instance:
    kind = raw.get("instance_type") or {}
    ip = raw.get("ip") or None
    endpoints = {
        name: Endpoint(name=name, url=f"http://{ip}:{port}", internal_port=port, external_port=port)
        for port, name in wanted_ports.items()
        if ip
    }
    price = kind.get("price_cents_per_hour")
    return Instance(
        id=str(raw.get("id")),
        provider=PROVIDER_NAME,
        state=map_state(raw),
        created_at=_created_at(raw),
        gpu_name=gpu_name(kind) if kind else "",
        gpu_count=int((kind.get("specs") or {}).get("gpus") or 0),
        hourly_cost_usd=round(float(price) / 100.0, 4) if price else None,
        host=ip,
        ssh_host=ip,
        ssh_port=22 if ip else None,
        ssh_user="ubuntu",
        endpoints=endpoints,
        label=str(raw.get("name") or ""),
        raw=raw,
    )


def instance_nonce(raw: dict[str, Any]) -> Optional[str]:
    for tag in raw.get("tags") or []:
        if isinstance(tag, dict) and tag.get("key") == NONCE_TAG:
            return str(tag.get("value"))
    return None


def build_user_data(spec: ComputeSpec, container_env: dict[str, str], ports: list[int]) -> str:
    """Cloud-init script: run the boot script inside the model image, on the host network.

    A Lambda machine is a VM with Docker and the NVIDIA toolkit, not a
    container: the image runs under Docker, sharing the VM's network so the
    model and watchdog ports are the VM's own.
    """
    boot = base64.b64encode(gzip.compress(spec.onstart.encode())).decode()
    env_lines = "\n".join(f"{key}={value}" for key, value in container_env.items() if "\n" not in str(value))
    image = shlex.quote(spec.image)
    return "\n".join(
        [
            "#!/bin/bash",
            "# Generated by gpu4agents: runs once, as root, on first boot.",
            "set -u",
            "mkdir -p /opt/jambu /var/log/jambu",
            "exec >> /var/log/jambu/cloud-init.log 2>&1",
            "umask 077",
            "cat > /opt/jambu/container.env <<'JAMBU_ENV_EOF'",
            env_lines,
            "JAMBU_ENV_EOF",
            f"echo '{boot}' | base64 -d | gunzip > /opt/jambu/onstart.sh",
            "chmod 700 /opt/jambu/onstart.sh",
            # The docker daemon can come up after cloud-init starts this script.
            "for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done",
            # The watchdog lives in the container: if it never starts, nothing
            # would ever end the machine, so the boot ends it itself.
            "give_up() {",
            "  echo \"jambu: $1; terminating this machine\"",
            f"  python3 - <<'JAMBU_PY_EOF'\n{_SELF_TERMINATE}JAMBU_PY_EOF",
            "  exit 1",
            "}",
            "pulled=",
            f"for i in 1 2 3; do docker pull {image} && pulled=1 && break; sleep 20; done",
            '[ -n "$pulled" ] || give_up "the model image could not be pulled"',
            "docker run -d --name jambu --restart unless-stopped --gpus all --ipc host --network host "
            "--env-file /opt/jambu/container.env -v /opt/jambu:/opt/jambu -v /var/log/jambu:/var/log/jambu "
            f"--entrypoint /bin/bash {image} -c 'bash /opt/jambu/onstart.sh; exec sleep infinity' "
            '|| give_up "the model container did not start"',
            "",
        ]
    )


# Runs on the VM (stdlib only): find this machine by its nonce tag and terminate it.
_SELF_TERMINATE = '''import json, urllib.request
env = dict(l.rstrip("\\n").split("=", 1) for l in open("/opt/jambu/container.env") if "=" in l)
base, key, nonce = env["JAMBU_LAMBDA_API_BASE"].rstrip("/"), env["JAMBU_LAMBDA_KEY"], env["JAMBU_LAMBDA_NONCE"]
def call(method, path, body=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))
for item in call("GET", "/instances").get("data") or []:
    if any(t.get("key") == "gpu4agents-nonce" and t.get("value") == nonce for t in item.get("tags") or []):
        call("POST", "/instance-operations/terminate", {"instance_ids": [item["id"]]})
'''


def firewall_rules(ports: list[int], source_network: str) -> list[dict[str, Any]]:
    rules = [
        {"protocol": "tcp", "port_range": [22, 22], "source_network": "0.0.0.0/0", "description": "SSH"},
    ]
    for port in sorted(set(ports)):
        rules.append(
            {
                "protocol": "tcp",
                "port_range": [port, port],
                "source_network": source_network,
                "description": f"gpu4agents port {port}",
            }
        )
    return rules
