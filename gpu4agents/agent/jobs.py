"""`gpu setup` in the background, for callers that cannot block for 15 minutes.

An MCP tool call times out long before a first setup finishes, so the MCP
server starts setup as a detached job and reports on it when asked again.
The job outlives the server: a client that disconnects mid-setup leaves a
machine that either finishes booting (and is then stopped by its watchdog
when nobody uses it) or is destroyed by setup's own failure cleanup.

Files, under the project's ``.gpu/jobs/``:

- ``setup.json``  pid, argv and start time
- ``setup.out``   the command's stdout (one JSON document with --json)
- ``setup.err``   its stderr (progress is suppressed; errors land here)
- ``setup.rc``    the exit code, written last, when the command ends
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from ..core.paths import project_state_dir

JOB_DIRNAME = "jobs"
ERROR_TAIL_LINES = 40


def _alive(pid: int) -> bool:
    if os.name == "nt":  # signal 0 is CTRL_C_EVENT on Windows; trust the rc file
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tail(path: Path, lines: int) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.strip()][-lines:]


class Job:
    """One named background command per project (only ``setup`` today)."""

    def __init__(self, project_dir: Path, name: str = "setup") -> None:
        self.dir = project_state_dir(Path(project_dir)) / JOB_DIRNAME
        self.name = name
        self.meta_path = self.dir / f"{name}.json"
        self.out_path = self.dir / f"{name}.out"
        self.err_path = self.dir / f"{name}.err"
        self.rc_path = self.dir / f"{name}.rc"
        self._proc: Optional[subprocess.Popen] = None

    def meta(self) -> Optional[dict[str, Any]]:
        try:
            return json.loads(self.meta_path.read_text())
        except (OSError, ValueError):
            return None

    def exit_code(self) -> Optional[int]:
        try:
            return int(self.rc_path.read_text().strip())
        except (OSError, ValueError):
            return None

    def running(self) -> bool:
        meta = self.meta()
        if meta is None or self.exit_code() is not None:
            return False
        if self._proc is not None and self._proc.poll() is not None:
            return False  # reaped here: our own child never lingers as a zombie
        return _alive(int(meta["pid"]))

    def start(self, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
        """Start ``argv`` detached. The caller checks ``running()`` first."""
        self.dir.mkdir(parents=True, exist_ok=True)
        for path in (self.out_path, self.err_path, self.rc_path):
            path.unlink(missing_ok=True)
        runner = [sys.executable, "-m", "gpu4agents.agent.jobs", str(self.dir), self.name, "--", *argv]
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            kwargs["start_new_session"] = True
        self._proc = subprocess.Popen(
            runner,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        meta = {"pid": self._proc.pid, "argv": list(argv), "started_at": time.time()}
        self.meta_path.write_text(json.dumps(meta))
        return meta

    def wait(self, seconds: float, poll: float = 2.0) -> bool:
        """Wait up to ``seconds`` for the job to end. True when it has."""
        deadline = time.monotonic() + max(0.0, seconds)
        while self.running():
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(poll, max(0.05, deadline - time.monotonic())))
        return True

    def cancel(self, timeout: float = 120.0) -> bool:
        """Interrupt a running job like Ctrl-C and wait for it to clean up.

        ``gpu setup`` destroys the machine it was booting when interrupted, so
        a cancelled setup leaves nothing billing. True when the job has ended.
        """
        meta = self.meta()
        if meta is None or not self.running():
            return True
        pid = int(meta["pid"])
        try:
            if os.name == "nt":
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(pid, signal.SIGINT)  # the runner leads the group (start_new_session)
        except (ProcessLookupError, PermissionError, OSError):
            return not self.running()
        return self.wait(timeout, poll=1.0)

    def report(self) -> dict[str, Any]:
        """``status`` is none, running, succeeded, failed or lost (died without a code)."""
        meta = self.meta()
        if meta is None:
            return {"status": "none"}
        started = float(meta.get("started_at") or 0)
        report: dict[str, Any] = {
            "pid": meta.get("pid"),
            "started_at": started,
            "elapsed_s": round(time.time() - started, 1) if started else None,
        }
        if self.running():
            report["status"] = "running"
            return report
        code = self.exit_code()
        if code is None:
            report["status"] = "lost"
        else:
            report["status"] = "succeeded" if code == 0 else "failed"
            report["exit_code"] = code
            report["finished_at"] = self.rc_path.stat().st_mtime
            report["elapsed_s"] = round(report["finished_at"] - started, 1) if started else None
        try:
            report["result"] = json.loads(self.out_path.read_text())
        except (OSError, ValueError):
            pass
        if report["status"] != "succeeded":
            report["stderr_tail"] = _tail(self.err_path, ERROR_TAIL_LINES)
        return report


def _main(argv: list[str]) -> int:
    """``python -m gpu4agents.agent.jobs DIR NAME -- CMD...``: run CMD, record its exit code."""
    directory, name = Path(argv[0]), argv[1]
    command = argv[3:] if len(argv) > 2 and argv[2] == "--" else argv[2:]
    # A cancel sends SIGINT to the whole group. The runner must outlive it to
    # record the exit code, and must not kill the command while it cleans up.
    # A Python handler (not SIG_IGN) resets to the default in the child on exec.
    signal.signal(signal.SIGINT, lambda signum, frame: None)
    with open(directory / f"{name}.out", "wb") as out, open(directory / f"{name}.err", "wb") as err:
        try:
            proc = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        except OSError as exc:
            err.write(f"error: could not start {command[0]}: {exc}\n".encode())
            code = 127
        else:
            code = proc.wait()
    tmp = directory / f"{name}.rc.tmp"
    tmp.write_text(str(code))
    tmp.replace(directory / f"{name}.rc")
    return code


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
