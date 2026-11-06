"""Translation between RunPod v2 payloads and the normalized domain model."""

from __future__ import annotations

import base64
import gzip
import time
from datetime import datetime
from typing import Any, Optional

from ...core.models import ComputeSpec, Endpoint, Instance, InstanceState, Offer
from ..gpus import compute_cap
from .config import RunpodOptions

PROVIDER_NAME = "runpod"

_STATE_MAP = {
    "PROVISIONING": InstanceState.PROVISIONING,
    "STARTING": InstanceState.STARTING,
    "RUNNING": InstanceState.READY,
    "EXITED": InstanceState.STOPPED,
    "ERROR": InstanceState.FAILED,
    "TERMINATED": InstanceState.DESTROYED,
}

# The boot script travels gzip+base64 in this env var; the container's
# command unpacks and runs it, then keeps the container alive.
BOOT_ENV = "JAMBU_BOOT"
BOOT_COMMAND = (
    "python3 -c \"import base64,gzip,os;open('/tmp/jambu-onstart.sh','wb')"
    ".write(gzip.decompress(base64.b64decode(os.environ['JAMBU_BOOT'])))\" "
    "|| (echo \"$JAMBU_BOOT\" | base64 -d | gunzip > /tmp/jambu-onstart.sh); "
    "bash /tmp/jambu-onstart.sh; exec sleep infinity"
)


def map_state(raw: dict[str, Any]) -> InstanceState:
    return _STATE_MAP.get(str(raw.get("status") or "").upper(), InstanceState.UNKNOWN)


def _float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def max_cuda(gpu: dict[str, Any]) -> Optional[float]:
    versions = []
    for item in gpu.get("cudaVersions") or []:
        if isinstance(item, dict) and item.get("available", True) is not False:
            value = _float(item.get("version"))
            if value:
                versions.append(value)
    return max(versions) if versions else None


def offer_id(cloud: str, gpu_type_id: str) -> str:
    return f"{cloud}:{gpu_type_id}"


def split_offer_id(value: str) -> tuple[str, str]:
    cloud, _, gpu_type_id = value.partition(":")
    return cloud, gpu_type_id


def map_offer(gpu: dict[str, Any], cloud: str, count: int) -> Optional[Offer]:
    if str(gpu.get("manufacturer") or "NVIDIA").upper() not in ("NVIDIA", ""):
        return None  # AMD cards cannot run the CUDA model images
    if not gpu.get(cloud.lower(), True):
        return None
    price = _float((gpu.get("price") or {}).get(cloud.lower()))
    if not price:
        return None
    centers = [str(dc.get("id")) for dc in gpu.get("dataCenters") or [] if dc.get("availability") != "NONE"]
    name = str(gpu.get("name") or gpu.get("id") or "")
    return Offer(
        id=offer_id(cloud, str(gpu.get("id"))),
        gpu_name=name,
        gpu_count=int(count),
        gpu_vram_gb=float(gpu.get("memory") or 0),
        disk_gb=0.0,  # RunPod sizes the disk to the request
        hourly_cost_usd=round(price * int(count), 4),
        region=",".join(centers[:6]) or None,
        score={"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}.get(str(gpu.get("availability") or ""), 0.0),
        cuda_max=max_cuda(gpu),
        compute_cap=compute_cap(f"{gpu.get('id') or ''} {name}"),
        raw={"gpu": gpu, "cloud": cloud},
    )


def available(gpu: dict[str, Any], options: RunpodOptions) -> bool:
    level = str(gpu.get("availability") or "NONE").upper()
    if level == "NONE":
        return False
    return not (options.skip_low_availability and level == "LOW")


def offer_matches(offer: Offer, spec: ComputeSpec) -> bool:
    if spec.gpu.min_vram_gb and offer.gpu_vram_gb + 0.5 < float(spec.gpu.min_vram_gb):
        return False
    min_cuda = _float(spec.gpu.cuda_min)
    if min_cuda and offer.cuda_max is not None and offer.cuda_max + 1e-6 < min_cuda:
        return False
    min_cc = spec.gpu.compute_cap_min
    if min_cc and offer.compute_cap is not None and offer.compute_cap + 1e-6 < float(min_cc):
        return False
    if spec.max_hourly_cost_usd and offer.hourly_cost_usd > float(spec.max_hourly_cost_usd):
        return False
    if spec.gpu.name_filter and spec.gpu.name_filter.lower() not in (offer.gpu_name + " " + offer.id).lower():
        return False
    if spec.region and not matching_centers(offer, spec.region):
        return False
    return True


def cuda_floor(spec: ComputeSpec) -> Optional[str]:
    """compute.gpu.cuda_min as RunPod's "major.minor"."""
    if not spec.gpu.cuda_min:
        return None
    major, _, minor = str(spec.gpu.cuda_min).partition(".")
    return f"{int(major)}.{int(minor or 0)}"


def proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def _timestamp(value: Any) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def map_instance(raw: dict[str, Any], wanted_ports: dict[int, str]) -> Instance:
    pod_id = str(raw.get("id"))
    state = map_state(raw)
    endpoints: dict[str, Endpoint] = {}
    if state is InstanceState.READY:
        # RunPod's HTTP proxy answers for any published port once the pod runs.
        endpoints = {
            name: Endpoint(name=name, url=proxy_url(pod_id, port), internal_port=port, external_port=443)
            for port, name in wanted_ports.items()
        }
    gpu = raw.get("gpu") or {}
    return Instance(
        id=pod_id,
        provider=PROVIDER_NAME,
        state=state,
        created_at=_timestamp(raw.get("createdAt")) or time.time(),
        gpu_name=str(gpu.get("id") or ""),
        gpu_count=int(gpu.get("count") or 0),
        hourly_cost_usd=_float(raw.get("cost")),
        host=None,
        endpoints=endpoints,
        label=str(raw.get("name") or ""),
        raw=raw,
    )


def build_create_payload(
    spec: ComputeSpec, options: RunpodOptions, offer: Offer, ports: list[int], extra_env: dict[str, str]
) -> dict[str, Any]:
    cloud, gpu_type_id = split_offer_id(offer.id)
    env = {str(k): str(v) for k, v in spec.env.items()}
    env.update(extra_env)
    env[BOOT_ENV] = base64.b64encode(gzip.compress(spec.onstart.encode())).decode()
    gpu: dict[str, Any] = {"id": gpu_type_id, "count": int(spec.gpu.count)}
    floor = cuda_floor(spec)
    if floor:
        gpu["minCudaVersion"] = floor
    payload: dict[str, Any] = {
        "name": spec.label,
        "image": spec.image,
        "gpu": gpu,
        "cloud": cloud,
        "disk": int(spec.disk_gb),
        "ports": [f"{port}/http" for port in ports],
        "env": env,
        "entrypoint": ["/bin/bash", "-c"],
        "cmd": [BOOT_COMMAND],
    }
    if options.volume_gb:
        payload["mounts"] = {"persistent": {"size": max(10, int(options.volume_gb)), "path": "/workspace"}}
    centers = matching_centers(offer, spec.region)
    if centers:
        payload["dataCenterIds"] = centers
    return payload


def matching_centers(offer: Offer, region: Optional[str]) -> list[str]:
    """The offer's data centers that ``region`` names, by prefix ("US" -> US-TX-3, ...)."""
    if not region:
        return []
    wanted = [r.strip().upper() for r in str(region).split(",") if r.strip()]
    centers = [str(dc.get("id")) for dc in (offer.raw.get("gpu") or {}).get("dataCenters") or []]
    return [c for c in centers if any(c.upper().startswith(w) for w in wanted)]
