# Security policy

GPU for Agents handles provider API keys and starts machines that bill by the hour, so we take reports seriously.

## Reporting a vulnerability

Please do not open a public issue. Report privately through [GitHub's private vulnerability reporting](https://github.com/jambuai/gpu4agents/security/advisories/new), or email joao.gabriel@jambu.ai.

Include what you found, how to reproduce it, and the version (`gpu --version`). You will get an answer within 5 business days. Once a fix is released, we credit you in the release notes unless you prefer otherwise.

## Supported versions

Fixes go into the latest release. Upgrade with `pipx upgrade gpu4agents`.

## In scope

- Leaking `VAST_API_KEY` or the per-instance endpoint key (logs, `--json` output, MCP responses, files written to the instance).
- Anything that lets a machine keep billing past its `idle_timeout`, `max_lifetime` or budget.
- The watchdog and the generated onstart script on the instance.
