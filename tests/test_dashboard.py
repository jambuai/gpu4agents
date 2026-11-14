from __future__ import annotations

import json

import pytest

from gpu4agents.core.config import DashboardConfig, RuntimeConfig

from conftest import BASE_CONFIG


# --- Task 1: config --------------------------------------------------------

def test_config_defaults_dashboard_off():
    rc = RuntimeConfig.model_validate(BASE_CONFIG)
    assert rc.dashboard.enabled is False
    assert rc.dashboard.listen_host_port == ("127.0.0.1", 8137)


def test_dashboard_bad_listen_rejected():
    with pytest.raises(ValueError):
        DashboardConfig(listen="nope").listen_host_port
