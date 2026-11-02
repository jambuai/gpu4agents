"""Files live in .gpu/ and ~/.gpu; a legacy .jambu is adopted, never forgotten."""

from __future__ import annotations

from gpu4agents.core.paths import gpu_home, project_state_dir
from gpu4agents.core.state import RuntimeState, StateStore


def test_a_new_project_keeps_its_state_in_dot_gpu(tmp_path):
    assert project_state_dir(tmp_path) == tmp_path / ".gpu"


def test_a_project_from_before_0_3_keeps_its_instance(tmp_path):
    StateStore(tmp_path / ".jambu").save(RuntimeState(provider="vast", instance_id="42", state="ready"))

    state_dir = project_state_dir(tmp_path)

    assert state_dir == tmp_path / ".gpu"
    assert not (tmp_path / ".jambu").exists()
    assert StateStore(state_dir).load().instance_id == "42"


def test_both_present_means_dot_gpu_wins_and_the_legacy_is_left_alone(tmp_path):
    (tmp_path / ".gpu").mkdir()
    (tmp_path / ".jambu").mkdir()
    assert project_state_dir(tmp_path) == tmp_path / ".gpu"
    assert (tmp_path / ".jambu").is_dir()


def test_the_user_home_moves_from_dot_jambu(tmp_path, monkeypatch):
    monkeypatch.delenv("GPU_HOME", raising=False)
    monkeypatch.delenv("JAMBU_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".jambu").mkdir()
    (tmp_path / ".jambu" / "projects.json").write_text("[]")

    assert gpu_home() == tmp_path / ".gpu"
    assert (tmp_path / ".gpu" / "projects.json").read_text() == "[]"


def test_gpu_home_and_the_legacy_jambu_home_override_it(tmp_path, monkeypatch):
    monkeypatch.delenv("GPU_HOME", raising=False)
    monkeypatch.setenv("JAMBU_HOME", str(tmp_path / "old"))
    assert gpu_home() == tmp_path / "old"
    monkeypatch.setenv("GPU_HOME", str(tmp_path / "new"))
    assert gpu_home() == tmp_path / "new"


def test_a_config_resolves_its_state_dir_to_dot_gpu(config):
    assert config.state_dir == config.project_dir / ".gpu"
