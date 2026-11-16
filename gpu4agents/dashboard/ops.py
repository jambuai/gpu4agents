"""Talk to the `gpu` CLI on this host: live truth, logs, offers, and the
control actions (rent / stop / destroy / lock / restart-proxy).

Shelling out to `gpu` keeps the dashboard multi-provider for free — every
provider the CLI supports works here — and means a control action runs the
exact same code path as the terminal. The runner is injectable so tests never
spawn a process. This module is used only when the dashboard runs with
credentials (control mode); read-only mode never calls it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

# run(args, cwd, timeout) -> (returncode, stdout, stderr)
Runner = Callable[..., tuple[int, str, str]]


def _gpu_bin() -> str:
    return os.environ.get("GPU_BIN") or shutil.which("gpu") or "gpu"


def _default_run(args: list[str], cwd: Optional[str] = None, timeout: float = 60.0) -> tuple[int, str, str]:
    proc = subprocess.run([_gpu_bin(), *args], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def live_ids(run: Runner = _default_run) -> Optional[set]:
    """The set of instance ids the provider reports alive, or None if it could
    not be read (no credentials, error) — the caller then shows 'unverified'."""
    try:
        code, out, _ = run(["--json", "ls"], None, 40.0)
        if code != 0:
            return None
        data = json.loads(out or "{}")
        # `gpu --json ls` keys the id as "instance_id"; tolerate "id" too.
        ids = set()
        for inst in data.get("instances", []):
            iid = inst.get("instance_id", inst.get("id"))
            if iid is not None:
                ids.add(str(iid))
        return ids
    except Exception:  # noqa: BLE001 - unverifiable is None, never a crash
        return None


def offers(cwd: Optional[str] = None, run: Runner = _default_run) -> list[dict[str, Any]]:
    code, out, err = run(["--json", "offers"], cwd, 60.0)
    if code != 0:
        raise RuntimeError(err or out or "offers failed")
    data = json.loads(out or "{}")
    return data.get("offers", data if isinstance(data, list) else [])


def logs(project_dir: str, run: Runner = _default_run, lines: int = 200) -> str:
    code, out, err = run(["logs", "--lines", str(lines)], str(project_dir), 40.0)
    return out or err or "(no logs)"


# --- control actions (each targets one project by its own directory) --------

def rent(project_dir: str, offer_id: str, run: Runner = _default_run) -> dict:
    code, out, err = run(["--json", "setup", "--offer", str(offer_id)], str(project_dir), 1200.0)
    if code != 0:
        raise RuntimeError(err or out or "setup failed")
    return {"rented": offer_id, "output": (out or err)[-2000:]}


def stop(project_dir: str, run: Runner = _default_run) -> dict:
    code, out, err = run(["stop"], str(project_dir), 180.0)
    if code != 0:
        raise RuntimeError(err or out or "stop failed")
    return {"stopped": True}


def destroy(project_dir: str, run: Runner = _default_run) -> dict:
    code, out, err = run(["destroy", "--yes"], str(project_dir), 300.0)
    if code != 0:
        raise RuntimeError(err or out or "destroy failed")
    return {"destroyed": True}


def lock(project_dir: str, run: Runner = _default_run) -> dict:
    code, out, err = run(["lock"], str(project_dir), 60.0)
    if code != 0:
        raise RuntimeError(err or out or "lock failed")
    return {"locked": True}
