"""Translation between Vast.ai payloads and the normalized domain model.

Every Vast-specific field name, unit and status string is confined to this
module (spec section 21: "provider state mapping" belongs to the adapter).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ...core.models import ComputeSpec, Endpoint, Instance, InstanceState, Offer
from .config import VastOptions

PROVIDER_NAME = "vast"

# Vast `actual_status` / `cur_state` -> normalized state (spec section 8).
_STATE_MAP = {
    "created": InstanceState.PROVISIONING,
    "scheduling": InstanceState.PROVISIONING,
    "loading": InstanceState.STARTING,
    "starting": InstanceState.STARTING,
    "running": InstanceState.READY,
    "stopping": InstanceState.STOPPED,
    "stopped": InstanceState.STOPPED,
    "exited": InstanceState.STOPPED,
    "inactive": InstanceState.STOPPED,
    "offline": InstanceState.UNKNOWN,
    "error": InstanceState.FAILED,
    "failed": InstanceState.FAILED,
}


def map_state(raw: dict[str, Any]) -> InstanceState:
    """Normalize Vast's several overlapping status fields into one state."""
    actual = str(raw.get("actual_status") or "").lower()
    cur = str(raw.get("cur_state") or "").lower()
    intended = str(raw.get("intended_status") or "").lower()
    status_msg = str(raw.get("status_msg") or "").lower()

    if "error" in status_msg or "failed" in status_msg:
        return InstanceState.FAILED

    # A stop was requested: the container keeps reporting `running` for a
    # while as it goes down. Calling that READY made a setup right after a
    # stop skip the restart and then watch the machine die under it.
    if intended == "stopped" or cur == "stopped":
        return InstanceState.STOPPED

    # cur_state is the contract's intent; until Vast reports the container's
    # own status, a "running" contract is still being scheduled or pulled.
    if not actual and cur == "running":
        return InstanceState.PROVISIONING

    for value in (actual, cur):
        if value in _STATE_MAP:
            state = _STATE_MAP[value]
            # Vast reports `running` while the container is still being pulled.
            if state is InstanceState.READY and intended == "running":
                if raw.get("start_date") and not raw.get("public_ipaddr"):
                    return InstanceState.STARTING
            return state

    if intended == "stopped":
        return InstanceState.STOPPED
    return InstanceState.UNKNOWN


def vram_gb(value: Any) -> float:
    """Vast reports GPU RAM in MB on offers and instances; be tolerant anyway."""
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return round(number / 1024.0, 1) if number > 512 else round(number, 1)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _compute_cap(value: Any) -> Optional[float]:
    """Vast's ``compute_cap`` is capability ×100 (860 → 8.6); normalize it."""
    raw = _float_or_none(value)
    return raw / 100.0 if raw else None


def effective_min_cuda(spec: ComputeSpec, options: VastOptions) -> Optional[float]:
    """The stricter of the adapter's floor and what the spec (image) needs."""
    known = [
        v
        for v in (_float_or_none(options.min_cuda), _float_or_none(spec.gpu.cuda_min))
        if v
    ]
    return max(known) if known else None


def host_id(raw: dict[str, Any]) -> Optional[str]:
    """Vast's ``machine_id``: the physical host behind an offer or instance."""
    value = raw.get("machine_id")
    return str(value) if value not in (None, "") else None


def map_offer(raw: dict[str, Any]) -> Offer:
    return Offer(
        id=str(raw.get("id") or raw.get("ask_contract_id") or ""),
        gpu_name=str(raw.get("gpu_name") or "unknown"),
        gpu_count=int(raw.get("num_gpus") or 1),
        gpu_vram_gb=vram_gb(raw.get("gpu_ram")),
        disk_gb=float(raw.get("disk_space") or 0),
        hourly_cost_usd=float(raw.get("dph_total") or 0.0),
        region=raw.get("geolocation"),
        score=float(raw.get("score") or 0.0) if raw.get("score") is not None else None,
        # Highest CUDA the host driver runs; an image needing more dies at boot.
        cuda_max=_float_or_none(raw.get("cuda_max_good")),
        # Vast reports compute capability ×100 (860 = 8.6); normalize to 8.6.
        compute_cap=_compute_cap(raw.get("compute_cap")),
        host_id=host_id(raw),
        raw=raw,
    )


def extract_endpoints(raw: dict[str, Any], wanted: dict[int, str]) -> dict[str, Endpoint]:
    """Map internal container ports to their public address.

    ``wanted`` maps internal port -> endpoint name.
    """
    host = raw.get("public_ipaddr") or raw.get("ssh_host") or ""
    host = str(host).strip().rstrip("/")
    ports = raw.get("ports") or {}
    endpoints: dict[str, Endpoint] = {}
    for internal, name in wanted.items():
        binding = ports.get(f"{internal}/tcp") or ports.get(str(internal))
        if not binding or not host:
            continue
        entry = binding[0] if isinstance(binding, list) else binding
        external = entry.get("HostPort") if isinstance(entry, dict) else None
        if not external:
            continue
        endpoints[name] = Endpoint(
            name=name,
            url=f"http://{host}:{external}",
            internal_port=int(internal),
            external_port=int(external),
        )
    return endpoints


def map_instance(raw: dict[str, Any], wanted_ports: dict[int, str]) -> Instance:
    created = raw.get("start_date") or raw.get("created_at")
    try:
        created_at = float(created) if created else time.time()
    except (TypeError, ValueError):
        created_at = time.time()

    return Instance(
        id=str(raw.get("id")),
        provider=PROVIDER_NAME,
        state=map_state(raw),
        created_at=created_at,
        gpu_name=str(raw.get("gpu_name") or ""),
        gpu_count=int(raw.get("num_gpus") or 0),
        hourly_cost_usd=float(raw.get("dph_total") or 0.0) or None,
        host=raw.get("public_ipaddr"),
        ssh_host=raw.get("ssh_host"),
        ssh_port=int(raw["ssh_port"]) if raw.get("ssh_port") else None,
        ssh_user="root",
        endpoints=extract_endpoints(raw, wanted_ports),
        label=str(raw.get("label") or ""),
        host_id=host_id(raw),
        raw=raw,
    )


def build_search_query(
    spec: ComputeSpec, options: VastOptions, include_vram: bool = True
) -> dict[str, Any]:
    """Translate a normalized ComputeSpec into a Vast offer query."""
    query: dict[str, Any] = {
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "external": {"eq": False},
        "num_gpus": {"eq": int(spec.gpu.count)},
        "disk_space": {"gte": float(spec.disk_gb)},
        "type": "bid" if spec.spot else "on-demand",
        "order": [[options.order_by, options.order_dir]],
        "limit": int(options.search_limit),
    }
    if options.verified_only:
        query["verified"] = {"eq": True}
    if include_vram and spec.gpu.min_vram_gb:
        # Vast stores GPU RAM in MB.
        query["gpu_ram"] = {"gte": float(spec.gpu.min_vram_gb) * 1024.0}
    if options.min_reliability:
        query["reliability2"] = {"gte": float(options.min_reliability)}
    if options.min_inet_down_mbps:
        query["inet_down"] = {"gte": float(options.min_inet_down_mbps)}
    min_cuda = effective_min_cuda(spec, options)
    if min_cuda:
        query["cuda_max_good"] = {"gte": min_cuda}
    # Floor the GPU architecture, not just the driver's CUDA: Vast reports
    # compute capability ×100, so 8.0 -> 800. Without this the cheapest offers
    # (old Volta/Pascal cards behind modern drivers) pass cuda_max_good but
    # crash a newer runtime image at engine startup.
    if spec.gpu.compute_cap_min:
        query["compute_cap"] = {"gte": int(round(float(spec.gpu.compute_cap_min) * 100))}
    # gpu_name is deliberately NOT sent server-side: Vast's field only
    # accepts an exact match ({"eq": ...}), but name_filter is documented and
    # used everywhere else (offer_matches below) as a case-insensitive
    # substring - "A6000" must match Vast's "RTX A6000". An exact-match
    # server-side filter would silently return zero offers for any filter
    # that isn't the full canonical name. offer_matches() below does the
    # real (substring) filtering once results are back.
    if spec.max_hourly_cost_usd:
        query["dph_total"] = {"lte": float(spec.max_hourly_cost_usd)}
    excluded = [int(h) if str(h).isdigit() else h for h in spec.exclude_hosts]
    if excluded:
        # Machines that already failed to boot this setup.
        query["machine_id"] = {"notin": excluded}
    if spec.region:
        query["geolocation"] = {"in": [r.strip() for r in str(spec.region).split(",")]}
    if options.search_filters:
        query.update(options.search_filters)
    return query


def build_env(spec: ComputeSpec, ports: list[int]) -> dict[str, str]:
    """Vast expects port publications as pseudo env keys (``-p 8000:8000``)."""
    env: dict[str, str] = {str(k): str(v) for k, v in spec.env.items()}
    for port in ports:
        env[f"-p {port}:{port}"] = "1"
    if ports:
        env.setdefault("OPEN_BUTTON_PORT", str(ports[0]))
    return env


def build_create_payload(
    spec: ComputeSpec,
    options: VastOptions,
    ports: list[int],
    onstart: str,
    price: Optional[float] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_id": "me",
        "image": spec.image,
        "disk": float(spec.disk_gb),
        "label": spec.label,
        "onstart": onstart,
        "env": build_env(spec, ports),
        "runtype": options.runtype,
        "target_state": "running",
        "cancel_unavail": True,
    }
    if options.image_login:
        payload["image_login"] = options.image_login
    if spec.spot and price:
        payload["price"] = float(price)
    return payload


def offer_matches(offer: Offer, spec: ComputeSpec, min_cuda: Optional[float] = None) -> bool:
    """Client-side guard: Vast's filters are advisory, verify what we got.

    ``min_cuda`` defaults to the spec's own ``gpu.cuda_min``.
    """
    if min_cuda is None:
        min_cuda = _float_or_none(spec.gpu.cuda_min)
    # 1e-6: 12.9 must match a host reporting exactly 12.9.
    if min_cuda and offer.cuda_max is not None and offer.cuda_max + 1e-6 < min_cuda:
        return False
    # Vast's compute_cap filter is advisory; verify the architecture floor too.
    min_cc = spec.gpu.compute_cap_min
    if min_cc and offer.compute_cap is not None and offer.compute_cap + 1e-6 < float(min_cc):
        return False
    if offer.gpu_count < spec.gpu.count:
        return False
    if spec.gpu.min_vram_gb and offer.gpu_vram_gb + 0.5 < float(spec.gpu.min_vram_gb):
        return False
    if offer.disk_gb and offer.disk_gb + 0.5 < float(spec.disk_gb):
        return False
    if spec.max_hourly_cost_usd and offer.hourly_cost_usd > float(spec.max_hourly_cost_usd):
        return False
    if offer.host_id is not None and offer.host_id in spec.exclude_hosts:
        return False
    if spec.gpu.name_filter:
        if spec.gpu.name_filter.lower() not in offer.gpu_name.lower():
            return False
    return True
