from gpu4agents.queue.scaler import MachineView, ScaleInput, decide


def _inp(**kw):
    base = dict(
        now=1000.0, queued=0, oldest_queued_age_s=None, machines=[], pool_usd_per_hour=0.0,
        next_offer_usd=0.5, min_machines=0, max_machines=4, max_usd_per_hour=None,
        scale_up_after_s=60.0, jobs_per_machine=1, last_open_at=None, open_backoff_s=30.0,
    )
    base.update(kw)
    return ScaleInput(**base)


def _m(slot, state="busy", running=1, usd=0.5):
    return MachineView(slot=slot, state=state, hourly_usd=usd, running_jobs=running)


def test_opens_when_jobs_wait_past_the_threshold():
    d = decide(_inp(queued=3, oldest_queued_age_s=120, machines=[_m("a")]))
    assert d.open == 3 and d.blocked_by is None


def test_waits_before_the_threshold():
    d = decide(_inp(queued=3, oldest_queued_age_s=10, machines=[_m("a")]))
    assert d.open == 0 and d.blocked_by is None


def test_counts_machines_still_booting_as_capacity():
    d = decide(_inp(queued=2, oldest_queued_age_s=120, machines=[_m("a", "provisioning", 0), _m("b", "checking", 0)]))
    assert d.open == 0
    d = decide(_inp(queued=2, oldest_queued_age_s=120, jobs_per_machine=2, machines=[_m("a", "provisioning", 0)]))
    assert d.open == 0


def test_stops_at_max_machines():
    d = decide(_inp(queued=5, oldest_queued_age_s=120, max_machines=2, machines=[_m("a")]))
    assert d.open == 1 and d.blocked_by == "max_machines"
    d = decide(_inp(queued=5, oldest_queued_age_s=120, max_machines=1, machines=[_m("a")]))
    assert d.open == 0 and d.blocked_by == "max_machines"


def test_stops_at_the_spend_cap():
    d = decide(_inp(queued=5, oldest_queued_age_s=120, max_usd_per_hour=1.2, pool_usd_per_hour=0.5, next_offer_usd=0.5))
    assert d.open == 1 and d.blocked_by == "max_usd_per_hour"
    d = decide(_inp(queued=5, oldest_queued_age_s=120, max_usd_per_hour=0.5, pool_usd_per_hour=0.5))
    assert d.open == 0 and d.blocked_by == "max_usd_per_hour"


def test_keeps_min_machines_open_with_an_empty_queue():
    d = decide(_inp(min_machines=2, machines=[_m("a", "ready", 0)]))
    assert d.open == 1
    d = decide(_inp(min_machines=1, machines=[_m("a", "ready", 0), _m("b", "ready", 0), _m("c", "busy", 1)]))
    assert d.open == 0 and d.keepalive == ["a"]


def test_respects_open_backoff():
    d = decide(_inp(queued=2, oldest_queued_age_s=120, last_open_at=990.0))
    assert d.open == 0 and d.blocked_by == "backoff"
    d = decide(_inp(queued=2, oldest_queued_age_s=120, last_open_at=900.0))
    assert d.open == 2


def test_no_offer_blocks_without_failing():
    d = decide(_inp(queued=2, oldest_queued_age_s=120, next_offer_usd=None))
    assert d.open == 0 and d.blocked_by == "no_offer"
    assert decide(_inp()).blocked_by is None


def test_max_machines_is_reported_before_the_spend_cap():
    d = decide(_inp(queued=5, oldest_queued_age_s=120, max_machines=1, machines=[_m("a")],
                    max_usd_per_hour=0.5, pool_usd_per_hour=0.5))
    assert d.open == 0 and d.blocked_by == "max_machines"


def test_min_machines_refill_still_waits_out_the_backoff():
    d = decide(_inp(min_machines=2, last_open_at=995.0))
    assert d.open == 0 and d.blocked_by == "backoff"


def test_jobs_per_machine_partial_capacity():
    # one machine with 3 slots, 1 busy: 2 free; 5 queued needs 3 more slots -> 2 machines of 2
    ms = [_m("a", "busy", 1)]
    d = decide(_inp(queued=5, oldest_queued_age_s=120, jobs_per_machine=3, machines=ms))
    assert d.open == 1  # ceil((5-2)/3)
    d = decide(_inp(queued=2, oldest_queued_age_s=120, jobs_per_machine=3, machines=ms))
    assert d.open == 0
