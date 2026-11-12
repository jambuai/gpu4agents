from gpu4agents.core.errors import JambuError
from gpu4agents.queue.models import (
    FINAL_STATES, Attempt, JobState, LeaseLost, Outcome, QueueJob,
)


def test_queue_job_round_trips_through_dict():
    import json
    job = QueueJob(
        id="j1", key="k", bundle_ref="b", manifest={"run": "x"}, submitted_at=1.5,
        submitted_by="me", state=JobState.RUNNING, limits={"max_cost_usd": 2},
        attempts=[Attempt(1, "m", "r", 1.0, 2.0, Outcome.TIMEOUT, 124, "slow")],
        cancel_requested=True, machine="m",
    )
    data = json.loads(json.dumps(job.to_dict()))
    assert QueueJob.from_dict(data) == job


def test_final_states_are_exactly_three():
    assert FINAL_STATES == {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


def test_lease_lost_is_a_jambu_error():
    assert issubclass(LeaseLost, JambuError)


def test_contract_suite_imports_and_lists_its_tests():
    from gpu4agents.queue.testing import QueueBackendContract
    for name in ("test_submit_is_idempotent_on_key", "test_claim_returns_oldest_first",
                 "test_a_claimed_job_is_not_claimed_again", "test_claim_a_specific_job",
                 "test_lease_expiry_requeues", "test_a_stale_token_cannot_write",
                 "test_settle_is_final", "test_requeue_records_the_attempt",
                 "test_cancel_of_a_queued_job_is_immediate",
                 "test_cancel_of_a_claimed_job_sets_the_flag", "test_paused_queue_claims_nothing",
                 "test_one_coordinator_lease_at_a_time", "test_coordinator_lease_expires",
                 "test_stats_counts_and_oldest_age", "test_concurrent_claims_never_share_a_job"):
        assert callable(getattr(QueueBackendContract, name))
