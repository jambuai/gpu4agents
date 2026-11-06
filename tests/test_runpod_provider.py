"""The RunPod adapter against a mocked RunPod API (v2)."""

from __future__ import annotations

import base64
import gzip
import json

import httpx
import pytest
import respx

from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.errors import AuthenticationError, CapacityUnavailableError, ProvisioningError
from gpu4agents.core.models import ComputeSpec, GpuSpec, Instance, InstanceState
from gpu4agents.providers.runpod.config import API_BASE, GRAPHQL_URL
from gpu4agents.providers.runpod.provider import WATCHDOG_ACTIONS, RunpodProvider

GPUS = "%s/v2/catalog/gpus" % API_BASE
PODS = "%s/v2/pods" % API_BASE


def gpu(gpu_id, name, memory, secure, community, availability="HIGH", cuda=("12.8", "13.0"), centers=("US-TX-3",)):
    return {
        "id": gpu_id,
        "name": name,
        "pool": None,
        "manufacturer": "NVIDIA",
        "memory": memory,
        "secure": True,
        "community": True,
        "price": {"secure": secure, "community": community},
        "maxCount": {"secure": 8, "community": 8},
        "availability": availability,
        "dataCenters": [{"id": c, "name": c, "availability": availability} for c in centers],
        "cudaVersions": [{"version": v, "available": True} for v in cuda],
    }


CATALOG = [
    gpu("NVIDIA GeForce RTX 4090", "RTX 4090", 24, 0.69, 0.34),
    gpu("NVIDIA RTX A5000", "RTX A5000", 24, 0.27, 0.16, availability="NONE"),
    gpu("NVIDIA A100 80GB PCIe", "A100 PCIe", 80, 1.64, 1.19, centers=("EU-RO-1",)),
    gpu("Tesla V100-PCIE-16GB", "V100", 16, 0.30, 0.20),
]


def pod(status="RUNNING", pod_id="pod_abc123", name="jambu-test-abcdef"):
    return {
        "id": pod_id,
        "name": name,
        "status": status,
        "actions": ["stop", "restart", "terminate"],
        "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1},
        "cloud": "COMMUNITY",
        "dataCenterId": "US-TX-3",
        "cost": 0.34,
        "createdAt": "2026-11-06T12:00:00Z",
        "image": "vllm/vllm-openai:v0.32.0",
    }


@pytest.fixture
def provider(config, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test-key")
    return RunpodProvider(config, CredentialResolver(config.project_dir))


def spec(**kw):
    base = dict(gpu=GpuSpec(min_vram_gb=24, count=1, cuda_min="13.0"), disk_gb=80, image="vllm/vllm-openai:v0.32.0",
                onstart="#!/bin/bash\necho hi\n", label="jambu-test-abcdef", env={"HF_TOKEN": "hf_x"})
    base.update(kw)
    return ComputeSpec(**base)


def mock_catalog():
    return respx.get(GPUS).mock(return_value=httpx.Response(200, json={"gpus": CATALOG}))


@respx.mock
def test_offers_come_from_both_clouds_cheapest_first(provider):
    route = mock_catalog()
    offers = provider.find_offers(spec())
    clouds = {dict(c.request.url.params)["cloud"] for c in route.calls}
    assert clouds == {"SECURE", "COMMUNITY"}
    assert all(dict(c.request.url.params)["include"] == "AVAILABILITY" for c in route.calls)
    # A5000 has no stock, V100 is too small; the 4090 on community is cheapest.
    assert offers[0].id == "COMMUNITY:NVIDIA GeForce RTX 4090"
    assert offers[0].hourly_cost_usd == 0.34 and offers[0].gpu_vram_gb == 24
    assert offers[0].compute_cap == 8.9 and offers[0].cuda_max == 13.0
    assert not any("A5000" in o.id or "V100" in o.id for o in offers)


@respx.mock
def test_the_price_is_per_pod_not_per_gpu(provider):
    mock_catalog()
    offers = provider.find_offers(spec(gpu=GpuSpec(min_vram_gb=24, count=2)))
    assert offers[0].gpu_count == 2 and offers[0].hourly_cost_usd == 0.68


@respx.mock
def test_old_architectures_and_old_drivers_are_left_out(provider):
    respx.get(GPUS).mock(return_value=httpx.Response(200, json={"gpus": [
        gpu("Tesla V100-SXM2-32GB", "V100 SXM2", 32, 0.2, 0.1),
        gpu("NVIDIA L40S", "L40S", 48, 0.9, 0.8, cuda=("12.4",)),
    ]}))
    assert provider.find_offers(spec(gpu=GpuSpec(min_vram_gb=24, count=1, cuda_min="13.0", compute_cap_min=8.0))) == []


@respx.mock
def test_a_region_picks_matching_data_centers(provider):
    mock_catalog()
    create = respx.post(PODS).mock(return_value=httpx.Response(201, json=pod("PROVISIONING")))
    provider.setup(spec(region="EU"))
    body = json.loads(create.calls[0].request.content)
    assert body["gpu"]["id"] == "NVIDIA A100 80GB PCIe" and body["dataCenterIds"] == ["EU-RO-1"]


@respx.mock
def test_setup_creates_a_pod_that_runs_the_boot_script(provider):
    mock_catalog()
    create = respx.post(PODS).mock(return_value=httpx.Response(201, json=pod("PROVISIONING")))
    instance = provider.setup(spec())
    body = json.loads(create.calls[0].request.content)
    assert body["cloud"] == "COMMUNITY" and body["gpu"] == {"id": "NVIDIA GeForce RTX 4090", "count": 1, "minCudaVersion": "13.0"}
    assert body["ports"] == ["8000/http", "8777/http"]
    assert body["entrypoint"] == ["/bin/bash", "-c"] and "sleep infinity" in body["cmd"][0]
    assert body["disk"] == 80 and "mounts" not in body
    boot = gzip.decompress(base64.b64decode(body["env"]["JAMBU_BOOT"])).decode()
    assert boot == "#!/bin/bash\necho hi\n"
    assert body["env"]["JAMBU_RUNPOD_KEY"] == "rp-test-key" and body["env"]["HF_TOKEN"] == "hf_x"
    assert instance.id == "pod_abc123" and instance.state is InstanceState.PROVISIONING
    assert instance.hourly_cost_usd == 0.34


def test_the_boot_command_unpacks_and_runs_the_script(tmp_path):
    """The container command, run for real: it must unpack JAMBU_BOOT and run it."""
    import shutil
    import subprocess

    from gpu4agents.providers.runpod.mapper import BOOT_COMMAND

    if not shutil.which("bash"):
        pytest.skip("no bash")
    marker = tmp_path / "ran"
    script = f"#!/bin/bash\necho booted > {marker}\n"
    env = {"PATH": "/usr/bin:/bin", "JAMBU_BOOT": base64.b64encode(gzip.compress(script.encode())).decode()}
    command = BOOT_COMMAND.replace("exec sleep infinity", "true")
    subprocess.run(["/bin/bash", "-c", command], env=env, check=True, timeout=30)
    assert marker.read_text().strip() == "booted"


@respx.mock
def test_setup_walks_past_gpus_that_were_taken(provider):
    mock_catalog()
    create = respx.post(PODS).mock(side_effect=[
        httpx.Response(400, json={"title": "Bad Request", "status": 400, "detail": "There are no longer any instances available"}),
        httpx.Response(201, json=pod("PROVISIONING", pod_id="pod_second")),
    ])
    assert provider.setup(spec()).id == "pod_second"
    assert create.call_count == 2


@respx.mock
def test_no_credit_stops_the_setup(provider):
    mock_catalog()
    respx.post(PODS).mock(return_value=httpx.Response(402, json={"title": "Payment Required", "status": 402, "detail": "insufficient balance"}))
    with pytest.raises(ProvisioningError, match="credit"):
        provider.setup(spec())


@respx.mock
def test_no_capacity_is_a_normalized_error(provider):
    respx.get(GPUS).mock(return_value=httpx.Response(200, json={"gpus": []}))
    with pytest.raises(CapacityUnavailableError):
        provider.setup(spec())


@respx.mock
def test_a_bad_key_is_a_normalized_error(provider):
    respx.get(GPUS).mock(return_value=httpx.Response(401, json={"title": "Unauthorized", "status": 401, "detail": "invalid token"}))
    with pytest.raises(AuthenticationError):
        provider.find_offers(spec())


@respx.mock
def test_states_and_proxy_endpoints(provider):
    respx.get(f"{PODS}/pod_abc123").mock(side_effect=[
        httpx.Response(200, json=pod("STARTING")),
        httpx.Response(200, json=pod("RUNNING")),
        httpx.Response(200, json=pod("EXITED")),
        httpx.Response(404, json={"title": "Not Found", "status": 404, "detail": "The requested Pod does not exist."}),
    ])
    handle = Instance(id="pod_abc123", provider="runpod")
    starting = provider.status(handle)
    assert starting.state is InstanceState.STARTING and not starting.instance.endpoints
    running = provider.status(handle)
    assert running.state is InstanceState.READY
    assert running.instance.endpoint_url("model") == "https://pod_abc123-8000.proxy.runpod.net"
    assert running.instance.endpoint_url("watchdog") == "https://pod_abc123-8777.proxy.runpod.net"
    assert provider.status(handle).state is InstanceState.STOPPED
    gone = provider.status(handle)
    assert gone.state is InstanceState.DESTROYED and gone.exists is False


@respx.mock
def test_stop_and_destroy_are_idempotent(provider):
    respx.post(f"{PODS}/pod_abc123/action").mock(side_effect=[
        httpx.Response(200, json=pod("EXITED")),
        httpx.Response(409, json={"title": "Conflict", "status": 409, "detail": "pod is not running"}),
        httpx.Response(404, json={"title": "Not Found", "status": 404}),
    ])
    respx.delete(f"{PODS}/pod_abc123").mock(side_effect=[httpx.Response(204), httpx.Response(404)])
    respx.get(f"{PODS}/pod_abc123").mock(return_value=httpx.Response(200, json=pod("EXITED")))
    handle = Instance(id="pod_abc123", provider="runpod")
    for _ in range(3):
        provider.stop(handle)
    provider.destroy(handle)
    provider.destroy(handle)


@respx.mock
def test_list_instances_reports_only_ours(provider):
    respx.get(PODS).mock(return_value=httpx.Response(200, json={
        "pods": [pod(), pod(pod_id="pod_mine2", name="someone-elses"), pod("TERMINATED", pod_id="pod_old")],
        "pagination": {"nextCursor": None, "hasNextPage": False},
    }))
    assert [i.id for i in provider.list_instances()] == ["pod_abc123"]


@respx.mock
def test_login_checks_the_key_and_reads_only_the_balance():
    respx.get(GPUS).mock(return_value=httpx.Response(200, json={"gpus": CATALOG}))
    graphql = respx.post(GRAPHQL_URL).mock(return_value=httpx.Response(200, json={"data": {"myself": {"clientBalance": 12.345}}}))
    assert RunpodProvider.verify_credentials({"RUNPOD_API_KEY": "k"}) == {"balance_usd": 12.35}
    query = json.loads(graphql.calls[0].request.content)["query"]
    assert "email" not in query


@respx.mock
def test_login_rejects_a_bad_key():
    respx.get(GPUS).mock(return_value=httpx.Response(401, json={"title": "Unauthorized", "status": 401}))
    with pytest.raises(AuthenticationError):
        RunpodProvider.verify_credentials({"RUNPOD_API_KEY": "bad"})


@respx.mock
def test_logs_read_the_container_stream(provider):
    events = "".join(
        f"id: {i}\nevent: log\ndata: {json.dumps({'source': 'container', 'line': f'line {i}', 'ts': 'x'})}\n\n"
        for i in range(3)
    )
    respx.get(f"{PODS}/pod_abc123/logs").mock(return_value=httpx.Response(200, text=events, headers={"Content-Type": "text/event-stream"}))
    assert list(provider.logs(Instance(id="pod_abc123", provider="runpod"))) == ["line 0", "line 1", "line 2"]


def test_watchdog_actions_are_self_contained(provider):
    spec_ = provider.watchdog_spec()
    namespace: dict = {}
    exec(compile(spec_.actions_source, "<watchdog>", "exec"), namespace)
    assert callable(namespace["provider_stop"]) and callable(namespace["provider_destroy"])
    assert "RUNPOD_POD_ID" in spec_.actions_source and "/v2/pods/" in spec_.actions_source


def test_a_volume_makes_a_stopped_pod_bill(write_config, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    config = write_config(provider={"name": "runpod", "options": {"volume_gb": 50}})
    provider = RunpodProvider(config, CredentialResolver(config.project_dir))
    assert provider.capabilities.stop_eliminates_billing is False
    assert RunpodProvider.capabilities.stop_eliminates_billing is True


def test_the_real_boot_request_fits_runpods_limit(write_config, monkeypatch):
    from gpu4agents.core.engine import Engine
    from gpu4agents.providers.runpod import mapper
    from gpu4agents.providers.runpod.config import MAX_BODY_BYTES

    monkeypatch.setenv("RUNPOD_API_KEY", "k" * 64)
    monkeypatch.setenv("HF_TOKEN", "hf_" + "t" * 37)
    config = write_config(provider={"name": "runpod"}, model={"id": "empero-ai/Qwythos-9B-v2", "hf_token_env": "HF_TOKEN"})
    engine = Engine(config)
    built = engine.build_spec(engine._build_onstart("t" * 43))
    offer = mapper.map_offer(CATALOG[0], "SECURE", 1)
    payload = mapper.build_create_payload(built, engine.provider.options, offer, [8000, 8777], {"JAMBU_RUNPOD_KEY": "k" * 64})
    assert len(json.dumps(payload).encode()) < MAX_BODY_BYTES // 2


@respx.mock
def test_a_refused_stop_on_a_running_pod_is_an_error(provider):
    from gpu4agents.core.errors import ProviderError

    respx.post(f"{PODS}/pod_abc123/action").mock(return_value=httpx.Response(409, json={"title": "Conflict", "status": 409, "detail": "pod is locked"}))
    respx.get(f"{PODS}/pod_abc123").mock(return_value=httpx.Response(200, json=pod("RUNNING")))
    with pytest.raises(ProviderError, match="refused to stop"):
        provider.stop(Instance(id="pod_abc123", provider="runpod"))


@respx.mock
def test_a_create_whose_answer_was_lost_adopts_the_pod(provider, monkeypatch):
    monkeypatch.setattr("gpu4agents.providers.runpod.provider.time.sleep", lambda s: None)
    mock_catalog()
    respx.post(PODS).mock(return_value=httpx.Response(500, json={"title": "Internal", "status": 500}))
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    respx.get(PODS).mock(return_value=httpx.Response(200, json={
        "pods": [dict(pod("PROVISIONING", pod_id="pod_lost"), createdAt=now)],
        "pagination": {"hasNextPage": False},
    }))
    assert provider.setup(spec()).id == "pod_lost"


@respx.mock
def test_the_cuda_floor_scopes_the_catalog(provider):
    route = mock_catalog()
    provider.find_offers(spec())
    assert all(dict(c.request.url.params)["minCudaVersion"] == "13.0" for c in route.calls)


@respx.mock
def test_a_key_without_write_scope_is_not_called_capacity(provider):
    mock_catalog()
    respx.post(PODS).mock(return_value=httpx.Response(403, json={"title": "Forbidden", "status": 403, "detail": "missing scope pods:write"}))
    with pytest.raises(AuthenticationError):
        provider.setup(spec())
