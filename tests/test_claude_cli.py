"""`gpu claude`: wire Claude Code to the gateway from gpu.yaml (turnkey)."""

from __future__ import annotations

from gpu4agents.cli.claude import claude_env
from gpu4agents.core.config import RuntimeConfig

from conftest import BASE_CONFIG


def _cfg(**over):
    body = dict(BASE_CONFIG)
    body.update(over)
    return RuntimeConfig.model_validate(body)


def test_claude_env_from_gpu_yaml():
    cfg = _cfg(
        gateway={"enabled": True, "adapter": "anthropic-openai", "listen": "127.0.0.1:18090"},
        runtime={"engine": "llama", "quant": "Q5_K_XL", "context_length": 262144},
        model={"id": "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF"},
    )
    env = claude_env(cfg)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18090"
    assert env["ANTHROPIC_MODEL"] == "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF"
    # Claude Code needs every model alias set to the same served model
    for k in ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
              "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
        assert env[k] == "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF"
    # context window from gpu.yaml, so Claude Code doesn't assume 1M for an unknown id
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
    assert int(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"]) <= 32000


def test_claude_env_uses_served_name_when_set():
    cfg = _cfg(
        gateway={"enabled": True, "adapter": "anthropic-openai", "listen": "127.0.0.1:18080"},
        model={"id": "org/Big-Model", "served_name": "cyber"},
    )
    env = claude_env(cfg)
    assert env["ANTHROPIC_MODEL"] == "cyber"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18080"
