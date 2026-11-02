from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu4agents.core.config import RuntimeConfig, load_config  # noqa: E402

BASE_CONFIG = {
    "version": 1,
    "provider": {"name": "vast"},
    "compute": {"gpu": {"min_vram_gb": 24, "count": 1}, "disk_gb": 80},
    "model": {"id": "empero-ai/Qwythos-9B-v2"},
    "runtime": {"engine": "vllm", "port": 8000, "context_length": 16384},
    "lifecycle": {
        "max_lifetime": "6h",
        "idle_timeout": "15m",
        "auto_stop": True,
        "auto_destroy": False,
        "watchdog": {"enabled": True, "port": 8777, "interval": "60s"},
    },
    "health": {"interval": "60s", "startup_timeout": "10m"},
}


@pytest.fixture(autouse=True)
def jambu_home(tmp_path, monkeypatch):
    """Keep the project registry and sweep state out of the real ~/.gpu."""
    home = tmp_path / "gpu-home"
    monkeypatch.setenv("GPU_HOME", str(home))
    monkeypatch.delenv("JAMBU_HOME", raising=False)
    return home


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@pytest.fixture
def write_config(tmp_path: Path):
    """Writes the primary, namespaced jambu.yaml shape by default.

    Use write_flat_config for the legacy config.yml (no gpu_runtime: wrapper)
    shape specifically.
    """

    def _write(**overrides) -> RuntimeConfig:
        data = _deep_merge(BASE_CONFIG, overrides)
        path = tmp_path / "jambu.yaml"
        path.write_text(yaml.safe_dump({"gpu_runtime": data}, sort_keys=False))
        return load_config(path)

    return _write


@pytest.fixture
def write_flat_config(tmp_path: Path):
    """Writes the legacy, unnamespaced config.yml shape."""

    def _write(**overrides) -> RuntimeConfig:
        data = _deep_merge(BASE_CONFIG, overrides)
        path = tmp_path / "config.yml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return load_config(path)

    return _write


@pytest.fixture
def config(write_config) -> RuntimeConfig:
    return write_config()
