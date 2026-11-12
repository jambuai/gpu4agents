"""Jobs: any program, run on the rented GPU, with its outputs brought back.

`gpu run` runs a command on this computer against the model endpoint. A job
runs on the GPU machine itself: the CLI packs a directory of code, uploads it
to the watchdog (which runs it, see watchdog/jobs_ext.py), follows its log and
downloads whatever the job wrote to ``$JAMBU_OUTPUT_DIR``.

The contract a job follows is in JOBS.md. The parts the CLI enforces:

- the code is a directory (default: the current one), packed as a tar.gz
  without ``.git``, virtualenvs, caches, ``.env`` files and earlier outputs;
- ``gpujob.yaml`` in that directory may declare ``run``, ``setup``, ``env``,
  ``timeout``, ``exclude`` and ``label``; command-line flags win;
- the job's results are the files under ``$JAMBU_OUTPUT_DIR``, and they land
  in ``gpu-outputs/<job id>/`` here.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import secrets
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .beacon import WatchdogClient, WatchdogUnavailable
from .durations import parse_duration
from .errors import ConfigError, StateError

MANIFEST_NAME = "gpujob.yaml"
OUTPUTS_DIRNAME = "gpu-outputs"
EXTENSION_NAME = "jobs"
EXTENSION_PATH = Path(__file__).resolve().parent.parent / "watchdog" / "jobs_ext.py"

# Never shipped to the GPU: version control, environments, caches, secrets,
# and what earlier jobs brought back.
DEFAULT_EXCLUDES = (
    ".git",
    ".hg",
    ".svn",
    ".gpu",
    ".jambu",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".ipynb_checkpoints",
    ".DS_Store",
    ".env",
    ".env.*",
    OUTPUTS_DIRNAME,
)
BUNDLE_WARN_BYTES = 200 * 1024**2
BUNDLE_MAX_BYTES = 2 * 1024**3
UPLOAD_TIMEOUT_S = 600.0
LOG_POLL_S = 2.0


class JobManifest(BaseModel):
    """``gpujob.yaml``: what to run, declared next to the code."""

    model_config = ConfigDict(extra="forbid")

    run: Optional[str] = None
    setup: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout: Optional[str] = None
    exclude: list[str] = Field(default_factory=list)
    label: Optional[str] = None
    max_cost_usd: Optional[float] = None
    min_vram_gb: Optional[float] = None

    @field_validator("run", "setup", mode="before")
    @classmethod
    def _join_lists(cls, value: Any) -> Any:
        # `run: [python, train.py]` reads as argv; `run: |` as a script.
        if isinstance(value, list):
            import shlex

            return " ".join(shlex.quote(str(part)) for part in value)
        return value

    @field_validator("env", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        return value

    @field_validator("timeout")
    @classmethod
    def _duration(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            parse_duration(value)
        return value


def load_manifest(directory: Path) -> JobManifest:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return JobManifest()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping (run, setup, env, timeout, exclude, label)")
    try:
        return JobManifest.model_validate(raw)
    except (ValidationError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def new_job_id() -> str:
    return time.strftime("j%Y%m%d-%H%M%S-", time.gmtime()) + secrets.token_hex(2)


# ---------------------------------------------------------------------------
# the code bundle
# ---------------------------------------------------------------------------


def _excluded(relative: str, name: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        pattern = pattern.rstrip("/")
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(relative, pattern):
            return True
    return False


def bundle_files(directory: Path, exclude: list[str]) -> Iterator[tuple[Path, str]]:
    """(path, name in the archive) for every file the job ships, sorted."""
    patterns = list(DEFAULT_EXCLUDES) + list(exclude)
    for root, dirs, files in os.walk(directory):
        base = Path(root)
        rel_root = base.relative_to(directory).as_posix()
        dirs[:] = sorted(
            d for d in dirs if not _excluded(d if rel_root == "." else f"{rel_root}/{d}", d, patterns)
        )
        for name in sorted(files):
            relative = name if rel_root == "." else f"{rel_root}/{name}"
            if _excluded(relative, name, patterns):
                continue
            path = base / name
            if path.is_symlink() or not path.is_file():
                continue
            yield path, relative


def build_bundle(directory: Path, exclude: list[str], target: Path) -> dict[str, Any]:
    """Pack ``directory`` into ``target`` (tar.gz). Returns its file count and size."""
    count = 0
    with tarfile.open(target, "w:gz") as tar:
        for path, relative in bundle_files(directory, exclude):
            tar.add(path, arcname=relative, recursive=False)
            count += 1
    size = target.stat().st_size
    if size > BUNDLE_MAX_BYTES:
        raise ConfigError(
            f"the job directory packs to {size // 1024**2} MB, over the "
            f"{BUNDLE_MAX_BYTES // 1024**3} GB limit. Exclude data in {MANIFEST_NAME} "
            "(exclude: [...]) and have the job download it instead."
        )
    return {"files": count, "bytes": size}


def safe_extract(archive: Path, dest: Path) -> list[str]:
    """Unpack a tar.gz the machine sent back, refusing paths outside ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    names: list[str] = []
    with tarfile.open(archive, "r:*") as tar:
        members = []
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if target != base and base not in target.parents:
                raise StateError(f"refusing an output outside {dest}: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                continue
            members.append(member)
            if member.isfile():
                name = member.name
                names.append(name[2:] if name.startswith("./") else name)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, members=members, filter="data")
        else:  # pragma: no cover - Python before 3.10.12
            tar.extractall(dest, members=members)
    return sorted(names)


# ---------------------------------------------------------------------------
# talking to the job runner on the machine
# ---------------------------------------------------------------------------


def extension_source() -> str:
    return EXTENSION_PATH.read_text()


def extension_digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


class JobConflict(WatchdogUnavailable):
    """The job runner already has a job with this id (HTTP 409): it was started before."""


class JobClient:
    """The watchdog's /jobs routes. Unlike the beacon, failures raise."""

    def __init__(self, watchdog: WatchdogClient) -> None:
        self.watchdog = watchdog

    def _url(self, path: str) -> str:
        return f"{self.watchdog.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = httpx.request(
                method,
                self._url(path),
                headers=self.watchdog.headers,
                timeout=kwargs.pop("timeout", 30.0),
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise WatchdogUnavailable(f"the machine's job runner is unreachable: {exc}") from exc
        return self._json(response)

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.status_code == 401:
            raise WatchdogUnavailable("the watchdog rejected the control token")
        if response.status_code == 404 and not data.get("error"):
            raise StateError(
                "this machine's watchdog has no job runner; it predates gpu jobs. "
                "`gpu setup --recreate` replaces it"
            )
        if response.status_code >= 400:
            message = data.get("error") or response.text[:200]
            if response.status_code == 404:
                raise StateError(message)
            if response.status_code == 409:
                raise JobConflict(f"job runner returned 409: {message}")
            raise WatchdogUnavailable(f"job runner returned {response.status_code}: {message}")
        return data

    def ensure_runner(self) -> None:
        """Upload the job runner unless the machine already runs this version."""
        source = extension_source()
        state = self.watchdog.state()
        if (state.get("extensions") or {}).get(EXTENSION_NAME) == extension_digest(source):
            return
        if "extensions" not in state:
            raise StateError(
                "this machine's watchdog cannot load the job runner (it was installed by "
                "gpu before 0.8.0). `gpu setup --recreate` replaces the machine"
            )
        import base64
        import zlib

        payload = base64.b64encode(zlib.compress(source.encode(), 9)).decode()
        self._request("POST", "/extend", json={"name": EXTENSION_NAME, "source": payload})

    def upload(self, job_id: str, bundle: Path) -> dict[str, Any]:
        size = bundle.stat().st_size

        def chunks() -> Iterator[bytes]:
            with open(bundle, "rb") as handle:
                while True:
                    chunk = handle.read(1 << 20)
                    if not chunk:
                        return
                    yield chunk

        headers = dict(self.watchdog.headers)
        headers["Content-Length"] = str(size)
        headers["Content-Type"] = "application/gzip"
        try:
            response = httpx.put(
                self._url(f"/jobs/{job_id}/bundle"),
                content=chunks(),
                headers=headers,
                timeout=UPLOAD_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            raise WatchdogUnavailable(f"uploading the job's code failed: {exc}") from exc
        return self._json(response)

    def start(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/jobs/{job_id}/start", json=payload)

    def info(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")

    def log(self, job_id: str, offset: int = 0, limit: Optional[int] = None) -> dict[str, Any]:
        params: dict[str, Any] = {"offset": offset}
        if limit:
            params["limit"] = limit
        return self._request("GET", f"/jobs/{job_id}/log", params=params)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/jobs/{job_id}/cancel")

    def list(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/jobs").get("jobs") or [])

    def download_outputs(self, job_id: str, dest: Path) -> list[str]:
        with tempfile.TemporaryDirectory(prefix="gpu-job-") as tmp:
            archive = Path(tmp) / "outputs.tar.gz"
            try:
                with httpx.stream(
                    "GET",
                    self._url(f"/jobs/{job_id}/outputs"),
                    headers=self.watchdog.headers,
                    timeout=UPLOAD_TIMEOUT_S,
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        self._json(response)
                    with open(archive, "wb") as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
            except httpx.HTTPError as exc:
                raise WatchdogUnavailable(f"downloading the job's outputs failed: {exc}") from exc
            return safe_extract(archive, dest)


# ---------------------------------------------------------------------------
# local records: what this project submitted, so ids outlive the machine
# ---------------------------------------------------------------------------


class JobRecords:
    def __init__(self, state_dir: Path) -> None:
        self.dir = Path(state_dir) / "job-runs"

    def save(self, record: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, indent=2, default=str))

    def load(self, job_id: str) -> Optional[dict[str, Any]]:
        try:
            return json.loads((self.dir / f"{job_id}.json").read_text())
        except (OSError, ValueError):
            return None

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        record = self.load(job_id) or {"id": job_id}
        record.update(fields)
        self.save(record)
        return record

    def all(self) -> list[dict[str, Any]]:
        if not self.dir.is_dir():
            return []
        records = []
        for path in self.dir.glob("*.json"):
            try:
                records.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda r: r.get("submitted_at") or 0)

    def latest(self) -> Optional[dict[str, Any]]:
        records = self.all()
        return records[-1] if records else None
