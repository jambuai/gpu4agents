"""Where gpu keeps its files: ``.gpu/`` in a project, ``~/.gpu`` per user.

Both were named ``.jambu`` before 0.3.0. A legacy directory is renamed in
place the first time it is needed, so an instance a project already tracks
is never forgotten (and never rented twice) after an upgrade.
"""

from __future__ import annotations

import os
from pathlib import Path

STATE_DIRNAME = ".gpu"
LEGACY_STATE_DIRNAME = ".jambu"


def adopt_legacy(path: Path, legacy: Path) -> Path:
    """``path``, after moving ``legacy`` there when only the legacy one exists."""
    if not path.exists() and legacy.is_dir():
        try:
            legacy.rename(path)
        except OSError:
            # Read-only or racing another gpu process: keep using the old one.
            return path if path.exists() else legacy
    return path


def project_state_dir(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    return adopt_legacy(project_dir / STATE_DIRNAME, project_dir / LEGACY_STATE_DIRNAME)


def gpu_home() -> Path:
    """Per-user files: the project registry, sweep sightings, bad hosts, credentials.

    ``$GPU_HOME`` overrides it (``$JAMBU_HOME`` is still honored).
    """
    override = os.environ.get("GPU_HOME") or os.environ.get("JAMBU_HOME")
    if override:
        return Path(override).expanduser()
    home = Path.home()
    return adopt_legacy(home / STATE_DIRNAME, home / LEGACY_STATE_DIRNAME)


EPHEMERAL_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/private/tmp", "/private/var/tmp")


def ephemeral_reason(path: Path) -> "str | None":
    """Why ``path`` is a bad home for a gpu project, or None.

    A project kept in a temp or agent-session scratch folder is lost with it,
    and the machine it rented becomes an orphan no later session can use.
    ``GPU_ALLOW_TEMP_PROJECT=1`` (or `gpu init --allow-temp`) allows it.
    """
    import tempfile

    if os.environ.get("GPU_ALLOW_TEMP_PROJECT"):
        return None
    resolved = Path(path).expanduser().resolve()
    if "scratchpad" in resolved.parts:
        return "it is an agent session's scratchpad"
    roots = {Path(r) for r in EPHEMERAL_ROOTS} | {Path(tempfile.gettempdir())}
    for root in roots:
        try:
            resolved.relative_to(root.resolve() if root.exists() else root)
        except ValueError:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
        return f"it is under the temporary directory {root}"
    return None
