from gpu4agents.core.events import EventLog
from gpu4agents.queue.observe import Observer


def _obs(tmp_path):
    return Observer(EventLog(tmp_path / "logs"), tmp_path / "q" / "metrics.prom")


def test_observer_writes_prometheus_text_atomically(tmp_path):
    o = _obs(tmp_path)
    o.inc("gpu_queue_retries_total", cause="machine_lost")
    o.gauge("gpu_queue_depth", 3)
    o.timing("gpu_queue_job_wait_seconds", 2.0)
    o.timing("gpu_queue_job_wait_seconds", 4.0)
    o.write()
    text = (tmp_path / "q" / "metrics.prom").read_text()
    assert 'gpu_queue_retries_total{cause="machine_lost"} 1' in text
    assert "gpu_queue_depth 3" in text
    assert "gpu_queue_job_wait_seconds_count 2" in text
    assert "gpu_queue_job_wait_seconds_sum 6" in text
    assert "gpu_queue_job_wait_seconds_max 4" in text
    assert [p.name for p in (tmp_path / "q").iterdir()] == ["metrics.prom"]


def test_observer_redacts_secrets(tmp_path):
    o = _obs(tmp_path)
    rec = o.emit("x", url="redis://u:pw@h", outputs_url="s3://a:b@c", token="abc", api_key="k", n=1)
    assert rec["url"] == "redis://u:***@h"
    assert rec["outputs_url"] == "s3://a:***@c"
    assert rec["token"] == "***" and rec["api_key"] == "***" and rec["n"] == 1
    assert "pw" not in (tmp_path / "logs" / "events.jsonl").read_text()
