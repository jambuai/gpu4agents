import pytest

from gpu4agents.core.config import QueueDefaults
from gpu4agents.core.errors import ConfigError
from gpu4agents.queue.models import QueueJob
from gpu4agents.queue.preflight import (
    Limits,
    PreflightContext,
    PreflightResult,
    default_preflight,
    load_preflight,
)


def _job(**manifest):
    return QueueJob(id="j1", key="k", bundle_ref="b", manifest=manifest, submitted_at=0.0, submitted_by="t")


def _ctx(**kw):
    base = dict(hourly_usd=0.5, card_vram_gb=24.0, defaults=QueueDefaults(), attempt=1)
    base.update(kw)
    return PreflightContext(**base)


def test_job_values_win_over_defaults():
    r = default_preflight(_job(timeout="30m", max_cost_usd=2.0), _ctx(defaults=QueueDefaults(timeout="2h", max_cost_usd=9.0)))
    assert r.ok
    assert r.limits == Limits(timeout_s=1800.0, max_cost_usd=2.0, deadline_margin_s=600.0)


def test_defaults_fill_missing_values():
    r = default_preflight(_job(), _ctx(defaults=QueueDefaults(timeout="2h", max_cost_usd=5.0, deadline_margin="5m")))
    assert r.ok
    assert r.limits == Limits(timeout_s=7200.0, max_cost_usd=5.0, deadline_margin_s=300.0)


def test_rejects_a_job_bigger_than_the_card():
    r = default_preflight(_job(min_vram_gb=48), _ctx(card_vram_gb=24))
    assert not r.ok and r.limits is None
    assert r.reason == "needs 48 GB, the pool's card has 24 GB"


def test_rejects_a_cost_limit_its_timeout_cannot_meet():
    r = default_preflight(_job(timeout="2h", max_cost_usd=0.5), _ctx(hourly_usd=0.5))
    assert not r.ok
    assert "0.5" in r.reason and "below 1 " in r.reason


def test_a_custom_preflight_is_loaded_by_name():
    assert load_preflight(None) is default_preflight
    fn = load_preflight("tests.test_queue_preflight:_custom")
    assert fn(_job(), _ctx()).reason == "custom"
    with pytest.raises(ConfigError):
        load_preflight("nope_module:fn")
    with pytest.raises(ConfigError):
        load_preflight("no-colon")


def _custom(job, ctx):
    return PreflightResult(ok=False, limits=None, reason="custom")


def _boom(job, ctx):
    raise RuntimeError("bad rule")


def _wrong(job, ctx):
    return True


def test_a_crashing_custom_preflight_rejects_the_job():
    r = load_preflight("tests.test_queue_preflight:_boom")(_job(), _ctx())
    assert not r.ok and "bad rule" in r.reason
    r = load_preflight("tests.test_queue_preflight:_wrong")(_job(), _ctx())
    assert not r.ok and r.reason


@pytest.mark.parametrize(
    "manifest",
    [
        {"max_cost_usd": "lots"}, {"max_cost_usd": True}, {"min_vram_gb": "big"}, {"min_vram_gb": [1]},
        {"timeout": "soon"}, {"timeout": True}, {"timeout": [1]}, {"timeout": -5}, {"max_cost_usd": float("nan")},
    ],
)
def test_malformed_manifest_values_are_rejected_not_raised(manifest):
    r = default_preflight(_job(**manifest), _ctx())
    assert not r.ok and r.limits is None and r.reason


def _ok_without_limits(job, ctx):
    return PreflightResult(ok=True, limits=None)


def _no_reason(job, ctx):
    return PreflightResult(ok=False, limits=None, reason="")


def test_bad_custom_results_are_rejected():
    for name in ("_ok_without_limits", "_no_reason"):
        r = load_preflight(f"tests.test_queue_preflight:{name}")(_job(), _ctx())
        assert not r.ok and r.reason


def test_unloadable_custom_preflights_are_config_errors():
    with pytest.raises(ConfigError):
        load_preflight("tests.test_queue_preflight:__doc__")
    with pytest.raises(ConfigError):
        load_preflight("tests.no_such_module_xyz:fn")
