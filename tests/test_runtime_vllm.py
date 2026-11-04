"""vLLM already speaks both OpenAI agent-facing APIs; this locks down that
gpu turns on the right flags for it and builds nothing of its own.
"""

from __future__ import annotations

import pytest

from gpu4agents.runtime.vllm import VllmRuntime, guess_tool_call_parser


@pytest.mark.parametrize(
    "model_id,parser",
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "llama3_json"),
        ("meta-llama/Llama-4-Scout-17B", "llama4_pythonic"),
        ("NousResearch/Hermes-3-Llama-3.1-8B", "llama3_json"),  # llama3 wins first
        ("Qwen/Qwen2.5-7B-Instruct", "hermes"),
        ("Qwen/Qwen3-Coder-30B", "qwen3_xml"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        ("ibm-granite/granite-4.0-tiny", "granite4"),
        ("ibm-granite/granite-3.1-8b-instruct", "granite"),
        ("deepseek-ai/DeepSeek-V3.1", "deepseek_v31"),
        ("deepseek-ai/DeepSeek-V3", "deepseek_v3"),
        ("Qwen/Qwen3-8B", "hermes"),  # original dash-versioned Qwen3: hermes
        # Qwen3.5+ point releases: XML function-call format (qwen3_xml), not
        # hermes - confirmed empirically against empero-ai/Qwythos-9B-v2.
        ("Qwen/Qwen3.8-27B", "qwen3_xml"),
        ("JonathanColetti/Qwen3.8-27B-Uncensored", "qwen3_xml"),
        ("huihui-ai/Huihui-Qwen3.8-27B-abliterated", "qwen3_xml"),
        ("empero-ai/Qwythos-9B-v2", None),  # unknown fine-tune (no "qwen3" in the name): no guess
    ],
)
def test_guess_tool_call_parser(model_id, parser):
    assert guess_tool_call_parser(model_id) == parser


def test_auto_enables_tool_calling_for_a_recognized_model(write_config):
    config = write_config(model={"id": "Qwen/Qwen2.5-7B-Instruct"})
    runtime = VllmRuntime(config)
    args = runtime.server_args()
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "hermes"


def test_auto_leaves_tool_calling_off_for_an_unknown_model(config):
    runtime = VllmRuntime(config)  # config fixture model.id is a fine-tune
    args = runtime.server_args()
    assert "--enable-auto-tool-choice" not in args
    assert "--tool-call-parser" not in args


def test_explicit_parser_overrides_the_guess(write_config):
    config = write_config(
        model={"id": "empero-ai/Qwythos-9B-v2"},
        runtime={"tool_call_parser": "hermes"},
    )
    runtime = VllmRuntime(config)
    args = runtime.server_args()
    assert args[args.index("--tool-call-parser") + 1] == "hermes"


def test_off_never_passes_tool_flags_even_for_a_recognized_model(write_config):
    config = write_config(
        model={"id": "Qwen/Qwen2.5-7B-Instruct"}, runtime={"tool_calling": "off"}
    )
    runtime = VllmRuntime(config)
    args = runtime.server_args()
    assert "--enable-auto-tool-choice" not in args
    assert "--tool-call-parser" not in args


def test_tool_server_is_passed_through_for_mcp_routing(write_config):
    config = write_config(
        model={"id": "Qwen/Qwen2.5-7B-Instruct"},
        runtime={"tool_server": "https://mcp.example.com"},
    )
    runtime = VllmRuntime(config)
    args = runtime.server_args()
    assert args[args.index("--tool-server") + 1] == "https://mcp.example.com"


def test_on_without_a_resolvable_parser_is_a_validation_error(config):
    config.runtime.tool_calling = "on"
    result = VllmRuntime(config).validate()
    assert not result.ok
    assert any("tool_call_parser" in (i.field or "") for i in result.errors)


def test_auto_without_a_resolvable_parser_is_only_a_warning(config):
    result = VllmRuntime(config).validate()  # default model.id is unrecognized
    assert result.ok
    assert any("tool_call_parser" in (i.field or "") for i in result.warnings)


def test_recognized_model_validates_clean(write_config):
    config = write_config(model={"id": "Qwen/Qwen2.5-7B-Instruct"})
    result = VllmRuntime(config).validate()
    assert not any("tool_call_parser" in (i.field or "") for i in result.issues)


def test_describe_surfaces_the_active_parser(write_config):
    config = write_config(model={"id": "Qwen/Qwen2.5-7B-Instruct"})
    assert "tools=hermes" in VllmRuntime(config).describe()


def test_qwythos_needs_an_explicit_override_since_its_name_hides_the_family():
    """Qwythos is Qwen3.5-based but the fine-tune name doesn't say so - auto
    can't guess it, confirmed live that qwen3_xml is the right parser for it.
    """
    assert guess_tool_call_parser("empero-ai/Qwythos-9B-v2") is None


# -- runtime.vllm_args: structured passthrough for un-modeled vLLM flags ----

from gpu4agents.runtime.vllm import flatten_vllm_args


def test_flatten_vllm_args_covers_every_value_shape():
    args = flatten_vllm_args(
        {
            "max_num_seqs": 64,
            "enable-prefix-caching": True,
            "enforce_eager": False,
            "limit-mm-per-prompt": {"image": 0, "video": 0},
            "compilation-config": [1, 2, 4],
            "skip-this": None,
            "--already-dashed": "kept",
        }
    )
    assert args == [
        "--max-num-seqs", "64",
        "--enable-prefix-caching",
        "--no-enforce-eager",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--compilation-config", "[1, 2, 4]",
        "--already-dashed", "kept",
    ]


def test_vllm_args_are_appended_to_server_args(write_config):
    config = write_config(runtime={"vllm_args": {"max_num_seqs": 32}})
    args = VllmRuntime(config).server_args()
    assert args[-2:] == ["--max-num-seqs", "32"]


def test_extra_args_win_over_vllm_args_on_conflict(write_config):
    """vllm_args, then extra_args - argparse takes the last occurrence."""
    config = write_config(
        runtime={
            "vllm_args": {"max_num_seqs": 32},
            "extra_args": ["--max-num-seqs", "64"],
        }
    )
    args = VllmRuntime(config).server_args()
    assert args[-2:] == ["--max-num-seqs", "64"]


def test_validate_warns_when_vllm_args_shadows_a_managed_flag(write_config):
    config = write_config(runtime={"vllm_args": {"port": 9000}})
    result = VllmRuntime(config).validate()
    assert any("vllm_args" in (i.field or "") for i in result.warnings)


def test_validate_is_quiet_for_unmanaged_vllm_args(write_config):
    config = write_config(runtime={"vllm_args": {"max_num_seqs": 32}})
    result = VllmRuntime(config).validate()
    assert not any("vllm_args" in (i.field or "") for i in result.issues)


def test_activity_probe_reads_vllm_metrics_on_loopback(write_config):
    runtime = VllmRuntime(write_config(runtime={"port": 8123}))
    assert runtime.activity_probe() == {
        "kind": "prometheus",
        "url": "http://127.0.0.1:8123/metrics",
        "gauges": ["vllm:num_requests_running", "vllm:num_requests_waiting"],
        "counters": ["vllm:request_success_total", "vllm:prompt_tokens_total"],
    }


def test_runtimes_without_a_traffic_signal_have_no_probe(write_config):
    from gpu4agents.runtime.base import NoRuntime

    assert NoRuntime(write_config(runtime={"engine": "none"})).activity_probe() is None


# -- the image decides the CUDA floor ----------------------------------------


@pytest.mark.parametrize(
    "image,cuda",
    [
        ("vllm/vllm-openai:v0.32.0-cu129", 12.9),
        ("vllm/vllm-openai:v0.32.0-cu130", 13.0),
        ("vllm/vllm-openai:nightly-cu134", 13.4),
        ("my.registry:5000/team/vllm:v1-cu128", 12.8),
        # bare tags carry no suffix: the known table answers
        ("vllm/vllm-openai:v0.32.0", 13.0),
        ("vllm/vllm-openai:v0.33.1", 13.0),
        ("docker.io/vllm/vllm-openai:v0.32.0", 13.0),
        ("vllm/vllm-openai:latest", 13.0),
        ("vllm/vllm-openai", 13.0),  # untagged means latest
        # unknown stays unknown, never guessed
        ("vllm/vllm-openai:v0.31.2", None),
        ("vllm/vllm-openai:nightly", None),
        ("someone/else:v0.40.0", None),
        ("someone/else:latest", None),
    ],
)
def test_the_cuda_a_vllm_image_needs_comes_from_its_tag(write_config, image, cuda):
    runtime = VllmRuntime(write_config(compute={"image": image}))
    assert runtime.required_cuda() == cuda


def test_the_default_image_is_pinned_and_carries_its_cuda_floor(config):
    runtime = VllmRuntime(config)
    assert runtime.image() == "vllm/vllm-openai:v0.32.0"
    assert runtime.required_cuda() == 13.0


def test_vllm_declares_where_its_pid_and_log_live(config):
    runtime = VllmRuntime(config)
    assert runtime.runtime_pid_path() == "/var/run/jambu/vllm.pid"
    assert runtime.runtime_log_path() == "/var/log/jambu/vllm.log"
    command = runtime.start_command()
    assert ">> /var/log/jambu/vllm.log 2>&1" in command
    assert command.endswith("echo $! > /var/run/jambu/vllm.pid")


def test_a_runtime_without_liveness_declares_nothing(write_config):
    from gpu4agents.runtime.base import NoRuntime

    runtime = NoRuntime(write_config(runtime={"engine": "none"}))
    assert runtime.runtime_pid_path() is None
    assert runtime.runtime_log_path() is None
    assert runtime.required_cuda() == 12.8  # pytorch/...-cuda12.8-...


def test_vast_published_vllm_images_carry_their_cuda(write_config):
    for image, cuda in (("vastai/vllm:v0.32.0-cuda-13.0", 13.0), ("vastai/vllm:v0.29.0-cuda-12.9", 12.9)):
        assert VllmRuntime(write_config(compute={"image": image})).required_cuda() == cuda


def test_a_vllm_image_floors_the_gpu_architecture(write_config):
    """CUDA-13 vLLM wheels drop Volta and need Ampere (8.0); 12.x still runs Volta (7.0).

    Caught on a live run: the CUDA-13 image kept landing on the cheapest offers
    (V100, GTX 1080) whose driver cleared cuda_max_good but whose architecture
    crashed vLLM's engine-core init."""
    assert VllmRuntime(write_config(compute={"image": "vastai/vllm:v0.32.0-cuda-13.0"})).required_compute_cap() == 8.0
    assert VllmRuntime(write_config(compute={"image": "vllm/vllm-openai:v0.32.0"})).required_compute_cap() == 8.0
    assert VllmRuntime(write_config(compute={"image": "vastai/vllm:v0.29.0-cuda-12.9"})).required_compute_cap() == 7.0
    # Unknown image CUDA -> no architecture claim.
    assert VllmRuntime(write_config(compute={"image": "vllm/vllm-openai:nightly"})).required_compute_cap() is None


def test_the_start_command_finds_the_python_that_has_vllm(config, tmp_path):
    """vastai/vllm puts a shim first on PATH; vllm lives under /usr/bin/python3."""
    import subprocess

    command = VllmRuntime(config).start_command()
    assert "import vllm" in command and "nohup $PY -m vllm.entrypoints" in command
    subprocess.run(["bash", "-n", "-c", command], check=True)
