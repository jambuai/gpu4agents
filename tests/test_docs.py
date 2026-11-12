"""The docs say what the code does: every event, every `gpu queue` subcommand, every
guarantee's test, the published copies and the version."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gpu4agents import __version__
from gpu4agents.core import events
from gpu4agents.cli.queue import queue_app

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.14.0"


def _read(*parts: str) -> str:
    return ROOT.joinpath(*parts).read_text()


def test_every_event_constant_is_on_the_docs_page():
    page = _read("site", "web", "docs", "index.html")
    names = [v for k, v in vars(events).items() if k.isupper() and isinstance(v, str)]
    assert names
    missing = [n for n in names if f"<code>{n}</code>" not in page]
    assert not missing, f"events missing from the docs page: {missing}"


def test_every_queue_subcommand_is_in_usage():
    usage = _read("docs", "USAGE.md")
    commands = [c.name or c.callback.__name__.removesuffix("_command") for c in queue_app.registered_commands]
    assert len(commands) >= 12
    missing = [c for c in commands if f"gpu queue {c}" not in usage]
    assert not missing, f"missing from USAGE.md: {missing}"


def test_every_queue_mcp_tool_is_on_the_docs_page():
    from gpu4agents.agent.mcp import Server

    page = _read("site", "web", "docs", "index.html")
    tools = [n for n in Server(project=ROOT).tools if n.startswith("gpu_queue_")]
    assert len(tools) == 5
    assert [t for t in tools if t not in page] == []


def test_the_version_is_0_14_0_everywhere():
    assert __version__ == VERSION
    assert f'version = "{VERSION}"' in _read("pyproject.toml")
    for manifest in (".claude-plugin/plugin.json", ".cursor-plugin/plugin.json"):
        assert json.loads(_read(*manifest.split("/")))["version"] == VERSION
    assert f"## [{VERSION}] - 2026-11-12" in _read("CHANGELOG.md")
    assert f"v{VERSION}" in _read("site", "web", "docs", "index.html")
    skill = _read("gpu4agents", "agent", "skill", "SKILL.md")
    assert f"prints {VERSION} or later" in skill
    assert _read("skills", "gpu4agents", "SKILL.md") == skill


def test_the_skill_says_when_to_queue_and_never_to_start_the_coordinator_unasked():
    skill = _read("gpu4agents", "agent", "skill", "SKILL.md")
    assert "gpu queue submit" in skill
    assert re.search(r"never start `gpu queue run`", skill, re.I)


def test_queue_md_is_published_and_linked():
    assert _read("site", "web", "queue.md") == _read("QUEUE.md")
    assert "https://gpu4agents.jambu.ai/queue.md" in _read("site", "web", "llms.txt")
    for threshold in (
        "heartbeat age", "invariant violation", "10 minutes", "20%", "90%", "2 × `scale_up_after`",
    ):
        assert threshold in _read("QUEUE.md"), threshold


def test_queue_backends_guide_exists():
    guide = _read("docs", "QUEUE_BACKENDS.md")
    assert "QueueBackendContract" in guide and "OutputStore" in guide


def test_acceptance_guarantees_24_to_31_each_name_a_test_that_exists():
    text = _read("docs", "ACCEPTANCE.md")
    tests_text = "\n".join(p.read_text() for p in (ROOT / "tests").glob("test_*.py"))
    for n in range(24, 32):
        rows = [line for line in text.splitlines() if line.startswith(f"| {n} |")]
        assert len(rows) == 1, n
        names = re.findall(r"::(test_\w+)", rows[0])
        assert names, f"guarantee {n} names no test"
        for name in names:
            assert re.search(rf"def {name}\b", tests_text), f"guarantee {n}: no test {name}"
