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
