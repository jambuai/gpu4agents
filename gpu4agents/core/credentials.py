"""Credential resolution (spec section 5).

Credentials never live in gpu.yaml. They are resolved from, in order:
environment variables, a project-local ``.env``, then ``~/.gpu/credentials``.
The resolution strategy can grow (keychain, secret managers, CI secrets)
without touching gpu.yaml or any adapter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .paths import gpu_home

_DOTENV_CACHE: dict[Path, dict[str, str]] = {}


def _parse_dotenv(path: Path) -> dict[str, str]:
    if path in _DOTENV_CACHE:
        return _DOTENV_CACHE[path]
    values: dict[str, str] = {}
    if path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            values[key.strip()] = value
    _DOTENV_CACHE[path] = values
    return values


class CredentialResolver:
    """Looks up named credentials without ever writing them to disk."""

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        profile: str = "default",
        extra_sources: Optional[list[tuple[str, Path]]] = None,
    ) -> None:
        self.project_dir = Path(project_dir or Path.cwd())
        self.profile = profile
        self.sources: list[tuple[str, Path]] = [
            ("project .env", self.project_dir / ".env"),
            ("user credentials", gpu_home() / "credentials"),
            *(extra_sources or []),
        ]

    def get(self, name: str) -> Optional[str]:
        value = os.environ.get(name)
        if value:
            return value.strip()
        # Profile-scoped override: VAST_API_KEY__staging
        for _, path in self.sources:
            values = _parse_dotenv(path)
            scoped = values.get(f"{name}__{self.profile}")
            if scoped:
                return scoped
            if values.get(name):
                return values[name]
        return None

    def file_source(self, name: str) -> Optional[str]:
        """Which file (not the environment) provides ``name``, if any.

        A systemd user unit does not inherit the shell's environment, so
        anything scheduled must find its credentials in one of these files.
        """
        for label, path in self.sources:
            values = _parse_dotenv(path)
            if values.get(f"{name}__{self.profile}") or values.get(name):
                return f"{label} ({path})"
        return None

    def require(self, name: str) -> str:
        value = self.get(name)
        if not value:
            raise KeyError(name)
        return value

    def missing(self, names: list[str]) -> list[str]:
        return [name for name in names if not self.get(name)]

    def describe_sources(self) -> list[str]:
        out = ["environment variables"]
        for label, path in self.sources:
            if path.is_file():
                out.append(f"{label} ({path})")
        return out


def save_user_credential(name: str, value: str) -> Path:
    """Write ``name=value`` into ``~/.gpu/credentials`` (mode 600), replacing an earlier value.

    Other lines are kept as they are. This is the only place a credential is
    written to disk, and only when the user runs ``gpu login``.
    """
    path = gpu_home() / "credentials"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    lines = path.read_text().splitlines() if path.is_file() else []
    kept = [
        line for line in lines
        if line.strip().removeprefix("export ").partition("=")[0].strip() != name
    ]
    kept.append(f"{name}={value}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("\n".join(kept) + "\n")
    path.chmod(0o600)
    _DOTENV_CACHE.pop(path, None)
    return path
