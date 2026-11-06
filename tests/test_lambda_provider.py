"""The Lambda Cloud adapter against a mocked Lambda API."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from gpu4agents.core.credentials import CredentialResolver
from gpu4agents.core.errors import AuthenticationError, CapacityUnavailableError, UnsupportedCapabilityError
from gpu4agents.core.models import ComputeSpec, GpuSpec, Instance, InstanceState
from gpu4agents.providers.lambdacloud.config import API_BASE
from gpu4agents.providers.lambdacloud.provider import WATCHDOG_ACTIONS, LambdaProvider


def kind(name, desc, gpus, cents, storage=512, arch="x86_64"):
    return {
        "name": name,
        "description": desc,
        "gpu_description": desc.split(" ", 1)[1] if desc[0].isdigit() else desc,
        "price_cents_per_hour": cents,
        "specs": {"vcpus": 30, "memory_gib": 200, "storage_gib": storage, "gpus": gpus},
        "architecture": arch,
    }


CATALOG = {
    "gpu_1x_a10": {
        "instance_type": kind("gpu_1x_a10", "1x A10 (24 GB PCIe)", 1, 75, storage=1400),
        "regions_with_capacity_available": [{"name": "us-east-1", "description": "Virginia, USA"}],
    },
    "gpu_1x_h100_pcie": {
        "instance_type": kind("gpu_1x_h100_pcie", "1x H100 (80 GB PCIe)", 1, 249),
        "regions_with_capacity_available": [
            {"name": "us-west-3", "description": "Utah, USA"},
            {"name": "europe-central-1", "description": "Germany"},
        ],
    },
    "gpu_8x_h100_sxm5": {
        "instance_type": kind("gpu_8x_h100_sxm5", "8x H100 (80 GB SXM5)", 8, 2392),
        "regions_with_capacity_available": [{"name": "us-west-3", "description": "Utah, USA"}],
    },
    "gpu_1x_gh200": {
        "instance_type": kind("gpu_1x_gh200", "1x GH200 (96 GB)", 1, 149, arch="arm64"),
        "regions_with_capacity_available": [{"name": "us-east-3", "description": "DC"}],
    },
    "gpu_1x_a100": {
        "instance_type": kind("gpu_1x_a100", "1x A100 (40 GB PCIe)", 1, 129),
        "regions_with_capacity_available": [],
    },
}


def instance(status="active", ip="203.0.113.7", iid="0920582c7ff041399e34823a0be62549"):
    return {
        "id": iid,
        "name": "jambu-test-abcdef",
        "ip": ip,
        "status": status,
        "ssh_key_names": ["gpu4agents-1234"],
        "file_system_names": [],
        "region": {"name": "us-east-1", "description": "Virginia, USA"},
        "instance_type": CATALOG["gpu_1x_a10"]["instance_type"],
        "tags": [{"key": "gpu4agents-nonce", "value": "n"}],
    }


@pytest.fixture
def provider(config, monkeypatch, tmp_path):
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_test")
    # The container is up unless a test says otherwise.
    monkeypatch.setattr(LambdaProvider, "_port_open", staticmethod(lambda host, port, timeout=3.0: True))
    # A key pair already on disk: tests never run ssh-keygen.
    key = LambdaProvider.key_path()
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text("PRIVATE")
    key.with_name(key.name + ".pub").write_text("ssh-ed25519 AAAAC3NzaTEST gpu4agents\n")
    return LambdaProvider(config, CredentialResolver(config.project_dir))


def spec(**kw):
    base = dict(gpu=GpuSpec(min_vram_gb=24, count=1, compute_cap_min=8.0), disk_gb=80, image="vllm/vllm-openai:v0.32.0",
                onstart="#!/bin/bash\necho hi\n", label="jambu-test-abcdef", env={"HF_TOKEN": "hf_x"})
    base.update(kw)
    return ComputeSpec(**base)


def mock_catalog():
    return respx.get(f"{API_BASE}/instance-types").mock(return_value=httpx.Response(200, json={"data": CATALOG}))


@respx.mock
def test_offers_are_one_per_type_and_region_with_capacity(provider):
    mock_catalog()
    offers = provider.find_offers(spec())
    assert [o.id for o in offers] == [
        "gpu_1x_a10@us-east-1",
        "gpu_1x_h100_pcie@europe-central-1",
        "gpu_1x_h100_pcie@us-west-3",
    ]
    assert offers[0].hourly_cost_usd == 0.75 and offers[0].gpu_vram_gb == 24 and offers[0].compute_cap == 8.6
    # The 8-GPU type, the arm64 GH200 and the A100 with no capacity are out.


@respx.mock
def test_a_region_and_a_name_filter_narrow_the_offers(provider):
    mock_catalog()
    offers = provider.find_offers(spec(region="europe", gpu=GpuSpec(min_vram_gb=24, count=1, name_filter="H100")))
    assert [o.id for o in offers] == ["gpu_1x_h100_pcie@europe-central-1"]


@respx.mock
def test_setup_launches_with_a_key_a_firewall_and_the_boot_script(provider):
    mock_catalog()
    respx.get(f"{API_BASE}/ssh-keys").mock(return_value=httpx.Response(200, json={"data": []}))
    add_key = respx.post(f"{API_BASE}/ssh-keys").mock(return_value=httpx.Response(200, json={"data": {"id": "k1", "name": "x"}}))
    respx.get(f"{API_BASE}/firewall-rulesets").mock(return_value=httpx.Response(200, json={"data": []}))
    ruleset = respx.post(f"{API_BASE}/firewall-rulesets").mock(return_value=httpx.Response(200, json={"data": {"id": "fw1"}}))
    launch = respx.post(f"{API_BASE}/instance-operations/launch").mock(
        return_value=httpx.Response(200, json={"data": {"instance_ids": ["0920582c7ff041399e34823a0be62549"]}})
    )
    respx.get(f"{API_BASE}/instances/0920582c7ff041399e34823a0be62549").mock(
        return_value=httpx.Response(200, json={"data": instance("booting", ip=None)})
    )

    created = provider.setup(spec())

    key_body = json.loads(add_key.calls[0].request.content)
    assert key_body["public_key"].startswith("ssh-ed25519 AAAAC3NzaTEST") and key_body["name"].startswith("gpu4agents-")
    rules = json.loads(ruleset.calls[0].request.content)
    assert rules["region"] == "us-east-1"
    assert sorted(r["port_range"][0] for r in rules["rules"]) == [22, 8000, 8777]
    body = json.loads(launch.calls[0].request.content)
    assert body["region_name"] == "us-east-1" and body["instance_type_name"] == "gpu_1x_a10"
    assert body["ssh_key_names"] == [key_body["name"]] and body["firewall_rulesets"] == [{"id": "fw1"}]
    assert body["tags"][0]["key"] == "gpu4agents-nonce"
    user_data = body["user_data"]
    assert user_data.startswith("#!/bin/bash") and "docker run" in user_data and "--gpus all" in user_data
    assert "JAMBU_LAMBDA_KEY=secret_test" in user_data and f"JAMBU_LAMBDA_NONCE={body['tags'][0]['value']}" in user_data
    assert created.state is InstanceState.PROVISIONING and created.hourly_cost_usd == 0.75


@respx.mock
def test_setup_reuses_the_key_and_ruleset_and_walks_past_full_regions(provider):
    mock_catalog()
    respx.get(f"{API_BASE}/ssh-keys").mock(return_value=httpx.Response(200, json={"data": [
        {"id": "k1", "name": "gpu4agents-old", "public_key": "ssh-ed25519 AAAAC3NzaTEST other-comment"},
    ]}))
    respx.get(f"{API_BASE}/firewall-rulesets").mock(return_value=httpx.Response(200, json={"data": [
        {"id": "fw-a10", "name": "gpu4agents-8000-8777", "region": {"name": "us-east-1", "description": ""}, "rules": []},
        {"id": "fw-eu", "name": "gpu4agents-8000-8777", "region": {"name": "europe-central-1", "description": ""}, "rules": []},
    ]}))
    full = {"error": {"code": "instance-operations/launch/insufficient-capacity", "message": "Not enough capacity"}}
    launch = respx.post(f"{API_BASE}/instance-operations/launch").mock(side_effect=[
        httpx.Response(400, json=full),
        httpx.Response(200, json={"data": {"instance_ids": ["second"]}}),
    ])
    respx.get(f"{API_BASE}/instances/second").mock(return_value=httpx.Response(200, json={"data": instance("booting", ip=None, iid="second")}))

    assert provider.setup(spec()).id == "second"
    bodies = [json.loads(c.request.content) for c in launch.calls]
    assert [b["ssh_key_names"] for b in bodies] == [["gpu4agents-old"]] * 2
    assert bodies[1]["firewall_rulesets"] == [{"id": "fw-eu"}]


@respx.mock
def test_no_capacity_is_a_normalized_error(provider):
    respx.get(f"{API_BASE}/instance-types").mock(return_value=httpx.Response(200, json={"data": {}}))
    with pytest.raises(CapacityUnavailableError):
        provider.setup(spec())


@respx.mock
def test_a_bad_key_is_a_normalized_error(provider):
    respx.get(f"{API_BASE}/instance-types").mock(return_value=httpx.Response(
        401, json={"error": {"code": "global/invalid-api-key", "message": "API key was invalid"}}))
    with pytest.raises(AuthenticationError):
        LambdaProvider.verify_credentials({"LAMBDA_API_KEY": "bad"})


@respx.mock
def test_states_and_endpoints(provider):
    respx.get(f"{API_BASE}/instances/i1").mock(side_effect=[
        httpx.Response(200, json={"data": instance("booting", ip=None, iid="i1")}),
        httpx.Response(200, json={"data": instance("active", iid="i1")}),
        httpx.Response(200, json={"data": instance("terminated", iid="i1")}),
        httpx.Response(404, json={"error": {"code": "global/object-does-not-exist", "message": "nope"}}),
    ])
    handle = Instance(id="i1", provider="lambda")
    assert provider.status(handle).state is InstanceState.PROVISIONING
    ready = provider.status(handle)
    assert ready.state is InstanceState.READY
    assert ready.instance.endpoint_url("model") == "http://203.0.113.7:8000"
    assert ready.instance.ssh_user == "ubuntu" and ready.instance.ssh_host == "203.0.113.7"
    assert provider.status(handle).exists is False
    assert provider.status(handle).exists is False


@respx.mock
def test_destroy_is_idempotent_and_stop_is_unsupported(provider):
    route = respx.post(f"{API_BASE}/instance-operations/terminate").mock(side_effect=[
        httpx.Response(200, json={"data": {"terminated_instances": []}}),
        httpx.Response(404, json={"error": {"code": "global/object-does-not-exist", "message": "gone"}}),
    ])
    handle = Instance(id="i1", provider="lambda")
    provider.destroy(handle)
    provider.destroy(handle)
    assert json.loads(route.calls[0].request.content) == {"instance_ids": ["i1"]}
    assert provider.capabilities.stop is False
    with pytest.raises(UnsupportedCapabilityError):
        provider.stop(handle)


@respx.mock
def test_list_instances_reports_only_ours(provider):
    other = dict(instance(iid="x"), name="training-box")
    respx.get(f"{API_BASE}/instances").mock(return_value=httpx.Response(200, json={"data": [instance(), other, instance("terminated", iid="old")]}))
    assert [i.id for i in provider.list_instances()] == ["0920582c7ff041399e34823a0be62549"]


def test_watchdog_finds_its_own_machine_by_the_nonce(monkeypatch):
    namespace: dict = {}
    exec(compile(WATCHDOG_ACTIONS, "<watchdog>", "exec"), namespace)
    calls = []

    def fake_call(method, path, body=None):
        calls.append((method, path, body))
        if path == "/instances":
            return {"data": [
                {"id": "not-me", "tags": [{"key": "gpu4agents-nonce", "value": "other"}]},
                {"id": "me", "tags": [{"key": "gpu4agents-nonce", "value": "n1"}]},
            ]}
        return {}

    namespace["_lambda_call"] = fake_call
    namespace["provider_destroy"].__globals__["_lambda_call"] = fake_call
    monkeypatch.setenv("JAMBU_LAMBDA_NONCE", "n1")
    namespace["provider_stop"]({})
    assert calls[-1] == ("POST", "/instance-operations/terminate", {"instance_ids": ["me"]})
    namespace["provider_destroy"]({"instance_id": "known"})
    assert calls[-1][2] == {"instance_ids": ["known"]}


def test_user_data_survives_a_real_boot_script(write_config, monkeypatch):
    from gpu4agents.core.engine import Engine
    from gpu4agents.providers.lambdacloud import mapper

    monkeypatch.setenv("LAMBDA_API_KEY", "k" * 40)
    engine = Engine(write_config(provider={"name": "lambda"}))
    built = engine.build_spec(engine._build_onstart("t" * 43))
    user_data = mapper.build_user_data(built, {"A": "1"}, [8000, 8777])
    assert len(user_data.encode()) < 1_000_000
    assert "vllm/vllm-openai" in user_data and engine.runtime.image() == built.image


@respx.mock
def test_an_active_machine_is_starting_until_its_container_answers(provider, monkeypatch):
    respx.get(f"{API_BASE}/instances/i1").mock(return_value=httpx.Response(200, json={"data": instance("active", iid="i1")}))
    up = {"value": False}
    monkeypatch.setattr(LambdaProvider, "_port_open", staticmethod(lambda host, port, timeout=3.0: up["value"]))
    handle = Instance(id="i1", provider="lambda")
    pulling = provider.status(handle)
    assert pulling.state is InstanceState.STARTING and "pulling the model image" in pulling.message
    up["value"] = True
    assert provider.status(handle).state is InstanceState.READY


@respx.mock
def test_a_launch_whose_answer_was_lost_is_found_by_its_nonce(provider, monkeypatch):
    monkeypatch.setattr("gpu4agents.providers.lambdacloud.provider.time.sleep", lambda s: None)
    mock_catalog()
    respx.get(f"{API_BASE}/ssh-keys").mock(return_value=httpx.Response(200, json={"data": [
        {"id": "k1", "name": "gpu4agents-old", "public_key": "ssh-ed25519 AAAAC3NzaTEST x"}]}))
    respx.get(f"{API_BASE}/firewall-rulesets").mock(return_value=httpx.Response(200, json={"data": [
        {"id": "fw", "name": "gpu4agents-8000-8777", "region": {"name": "us-east-1"}}]}))
    launch = respx.post(f"{API_BASE}/instance-operations/launch").mock(return_value=httpx.Response(502))

    def listing(request):
        nonce = json.loads(launch.calls[0].request.content)["tags"][0]["value"]
        return httpx.Response(200, json={"data": [dict(instance("booting", ip=None, iid="lost"), tags=[{"key": "gpu4agents-nonce", "value": nonce}])]})

    respx.get(f"{API_BASE}/instances").mock(side_effect=listing)
    respx.get(f"{API_BASE}/instances/lost").mock(return_value=httpx.Response(200, json={"data": instance("booting", ip=None, iid="lost")}))
    assert provider.setup(spec()).id == "lost"


def test_a_boot_that_cannot_start_the_container_terminates_the_machine():
    from gpu4agents.providers.lambdacloud import mapper

    user_data = mapper.build_user_data(spec(), {"A": "1"}, [8000])
    assert 'give_up "the model image could not be pulled"' in user_data
    assert '|| give_up "the model container did not start"' in user_data
    assert "/instance-operations/terminate" in user_data
