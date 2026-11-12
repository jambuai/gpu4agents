"""Output stores for queue jobs: bundles in, results out, every delivery checked against a sha256 manifest."""

from __future__ import annotations

import hashlib
import importlib
import contextlib
import json
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ConfigError

try:
    import fcntl
except ImportError:  # no flock on this platform (Windows): locking is a no-op there
    fcntl = None  # type: ignore[assignment]

MANIFEST = "MANIFEST.json"
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class OutputRef:
    location: str
    files: list[dict] = field(default_factory=list)  # each {"path", "bytes", "sha256"}


def _check_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not _JOB_ID.match(job_id) or ".." in job_id:
        raise ValueError(f"invalid job id {job_id!r}")
    return job_id


def _check_rel(rel: str) -> str:
    parts = rel.split("/")
    if not rel or rel.startswith("/") or "\\" in rel or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"path {rel!r} escapes the store")
    return rel


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan(src_dir: Path) -> list[dict]:
    files = []
    for dirpath, dirnames, filenames in os.walk(src_dir, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not (Path(dirpath) / d).is_symlink())
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file() or (name == MANIFEST and p.parent == Path(src_dir)):
                continue
            files.append({"path": p.relative_to(src_dir).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256_file(p)})
    return sorted(files, key=lambda f: f["path"])


def _norm(files: list[dict]) -> list[tuple]:
    return sorted((f["path"], f["bytes"], f["sha256"]) for f in files)


def _manifest_bytes(job_id: str, attempt: int, files: list[dict]) -> bytes:
    return json.dumps({"job_id": job_id, "attempt": attempt, "files": files}, indent=2, sort_keys=True).encode()


class OutputStore(ABC):
    @abstractmethod
    def put_bundle(self, job_id: str, path: Path) -> str: ...

    @abstractmethod
    def get_bundle(self, ref: str, dest: Path) -> Path: ...

    @abstractmethod
    def delete_bundle(self, ref: str) -> None:
        """Remove a bundle ``put_bundle`` stored. Missing is fine."""

    @abstractmethod
    def put_outputs(self, job_id: str, attempt: int, src_dir: Path, success: bool = True) -> OutputRef:
        """Store ``src_dir``. A successful attempt becomes the job's outputs; a failed one is kept under ``.attempt-<n>``."""

    @abstractmethod
    def verify(self, ref: OutputRef) -> list[str]:
        """Problems found re-checking every file against the manifest; [] means intact."""

    @abstractmethod
    def fetch_previous(self, ref: OutputRef, dest: Path) -> None:
        """Copy a verified earlier attempt's files into ``dest`` (for $JAMBU_PREVIOUS_OUTPUT_DIR)."""


class LocalOutputStore(OutputStore):
    def __init__(self, root: Path, bundles_dir: Path | None = None) -> None:
        self.root = Path(root)
        self.bundles_dir = Path(bundles_dir) if bundles_dir else self.root / ".bundles"

    def _inside(self, base: Path, *parts: str) -> Path:
        target = base.joinpath(*parts).resolve()
        if not target.is_relative_to(base.resolve()):
            raise ValueError(f"path {'/'.join(parts)!r} escapes the store")
        return target

    def put_bundle(self, job_id: str, path: Path) -> str:
        _check_job_id(job_id)
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        dest = self._inside(self.bundles_dir, f"{job_id}.tar.gz")
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(path, tmp)
        tmp.replace(dest)
        return str(dest)

    def get_bundle(self, ref: str, dest: Path) -> Path:
        src = Path(ref)
        if not src.is_file() or not src.resolve().is_relative_to(self.bundles_dir.resolve()):
            raise FileNotFoundError(f"bundle {ref!r} not found in this store")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    def delete_bundle(self, ref: str) -> None:
        src = Path(ref)
        if not src.resolve().is_relative_to(self.bundles_dir.resolve()):
            raise ValueError(f"bundle {ref!r} is not in this store")
        src.unlink(missing_ok=True)

    @contextlib.contextmanager
    def _lock(self, job_id: str):
        """Exclusive per-job lock, so recovery in one process never lands in the middle of another's swap."""
        if fcntl is None:
            yield
            return
        lock_dir = self.root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_dir / f"{job_id}.lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def recover(self, job_id: str) -> None:
        """Finish or undo a swap that crashed: the previous result comes back, earlier attempts are kept."""
        _check_job_id(job_id)
        if not self.root.is_dir():
            return
        with self._lock(job_id):
            self._recover_locked(job_id)

    def _recover_locked(self, job_id: str) -> None:
        if not self.root.is_dir():
            return
        pat = re.compile(rf"\.{re.escape(job_id)}\.old-\d+")
        target = self._inside(self.root, job_id)
        for aside in sorted(p for p in self.root.iterdir() if pat.fullmatch(p.name)):
            if not target.exists():
                os.replace(aside, target)
                continue
            for child in aside.iterdir():
                if child.name.startswith(".attempt-") and not (target / child.name).exists():
                    os.replace(child, target / child.name)
            shutil.rmtree(aside)
        if target.is_dir():  # a failed attempt's swap: job/..attempt-N.old-M
            for aside in sorted(target.iterdir()):
                m = re.fullmatch(r"\.(\.attempt-\d+)\.old-\d+", aside.name)
                if not m:
                    continue
                orig = target / m.group(1)
                if orig.exists():
                    shutil.rmtree(aside)
                else:
                    os.replace(aside, orig)

    def _recover_ref(self, ref: OutputRef) -> list[str]:
        """Recover under the lock; a store that cannot be written yields a problem entry, not an exception."""
        loc = Path(ref.location)
        try:
            for cand in (loc, loc.parent):
                if cand.parent.resolve() == self.root.resolve() and _JOB_ID.match(cand.name):
                    self.recover(cand.name)
                    break
        except OSError as exc:
            return [f"recovery of an interrupted swap failed: {type(exc).__name__} {exc.strerror or ''}".strip()]
        return []

    def put_outputs(self, job_id: str, attempt: int, src_dir: Path, success: bool = True) -> OutputRef:
        _check_job_id(job_id)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock(job_id):
            self._recover_locked(job_id)
            return self._put_locked(job_id, attempt, src_dir, success)

    def _put_locked(self, job_id: str, attempt: int, src_dir: Path, success: bool) -> OutputRef:
        job_dir = self._inside(self.root, job_id)
        target = job_dir if success else self._inside(self.root, job_id, f".attempt-{int(attempt)}")
        files = _scan(Path(src_dir))
        staging = job_dir.parent / f".{job_id}.staging-{attempt}"
        if staging.exists():
            shutil.rmtree(staging)
        for f in files:
            dst = staging / f["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(src_dir) / f["path"], dst)
        (staging / MANIFEST).write_bytes(_manifest_bytes(job_id, attempt, files))
        if target.exists():
            old = target.parent / f".{target.name}.old-{attempt}"
            if old.exists():
                shutil.rmtree(old)
            os.replace(target, old)  # the old result leaves the path before the new one arrives
            os.replace(staging, target)
            if success:
                for child in old.iterdir():  # failed-attempt dirs belong to the job, keep them
                    if child.name.startswith(".attempt-"):
                        os.replace(child, target / child.name)
            shutil.rmtree(old)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, target)
        return OutputRef(location=str(target), files=files)

    def verify(self, ref: OutputRef) -> list[str]:
        problems = self._recover_ref(ref)
        base = Path(ref.location)
        try:
            stored = json.loads((base / MANIFEST).read_text())["files"]
            if _norm(stored) != _norm(ref.files):
                problems.append(f"{MANIFEST} does not match the recorded files")
        except (OSError, ValueError, KeyError, TypeError):
            problems.append(f"{MANIFEST} is missing or unreadable")
        listed = set()
        for f in ref.files:
            try:
                p = self._inside(base, _check_rel(f["path"]))
            except ValueError as exc:
                problems.append(str(exc))
                continue
            listed.add(f["path"])
            if not p.is_file():
                problems.append(f"{f['path']}: missing")
            elif p.stat().st_size != f["bytes"]:
                problems.append(f"{f['path']}: size {p.stat().st_size} != {f['bytes']}")
            elif _sha256_file(p) != f["sha256"]:
                problems.append(f"{f['path']}: sha256 mismatch")
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                rel = p.relative_to(base).as_posix()
                if p.is_file() and rel != MANIFEST and not rel.startswith(".attempt-") and rel not in listed:
                    problems.append(f"{rel}: unexpected file")
        return problems

    def fetch_previous(self, ref: OutputRef, dest: Path) -> None:
        problems = self.verify(ref)
        if problems:
            raise ValueError("previous outputs failed verification: " + "; ".join(problems))
        base, dest = Path(ref.location), Path(dest)
        for f in ref.files:
            src = self._inside(base, _check_rel(f["path"]))
            out = self._inside(dest, f["path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)


class S3OutputStore(OutputStore):
    """``GPU_OUTPUTS_URL = s3://bucket/prefix``; endpoint via AWS_ENDPOINT_URL; keys via boto3's own chain."""

    def __init__(self, url: str) -> None:
        m = re.match(r"^s3://([^/]+)/?(.*)$", url or "")
        if not m:
            raise ConfigError("GPU_OUTPUTS_URL must look like s3://bucket/prefix")
        self.bucket, self.prefix = m.group(1), m.group(2).strip("/")
        try:
            import boto3
        except ImportError as exc:
            raise ConfigError(f"the s3 output store needs boto3 (pip install 'gpu4agents[queue-s3]'): {exc}") from exc
        self._s3 = boto3.client("s3")

    def __repr__(self) -> str:
        return f"S3OutputStore(s3://{self.bucket}/{self.prefix})"

    def _key(self, *parts: str) -> str:
        return "/".join(p for p in (self.prefix, *parts) if p)

    def _split(self, location: str) -> str:
        head = f"s3://{self.bucket}/"
        if not location.startswith(head):
            raise ValueError(f"{location!r} is not in this store")
        key = location[len(head):]
        if ".." in key.split("/"):
            raise ValueError(f"{location!r} escapes the store")
        return key

    def _put_file(self, key: str, path: Path) -> None:
        self._s3.upload_file(str(path), self.bucket, key)

    def put_bundle(self, job_id: str, path: Path) -> str:
        _check_job_id(job_id)
        key = self._key("bundles", f"{job_id}.tar.gz")
        self._put_file(key, Path(path))
        return f"s3://{self.bucket}/{key}"

    def get_bundle(self, ref: str, dest: Path) -> Path:
        key = self._split(ref)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket, key, str(dest))
        return dest

    def delete_bundle(self, ref: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=self._split(ref))

    def put_outputs(self, job_id: str, attempt: int, src_dir: Path, success: bool = True) -> OutputRef:
        _check_job_id(job_id)
        base = self._key(job_id) if success else self._key(job_id, f".attempt-{int(attempt)}")
        files = _scan(Path(src_dir))
        self._s3.delete_object(Bucket=self.bucket, Key=f"{base}/{MANIFEST}")  # no manifest while the set is incomplete
        for f in files:
            self._put_file(f"{base}/{f['path']}", Path(src_dir) / f["path"])
        # manifest last: its presence means every file above was written
        self._s3.put_object(Bucket=self.bucket, Key=f"{base}/{MANIFEST}", Body=_manifest_bytes(job_id, attempt, files))
        keep = {f"{base}/{f['path']}" for f in files} | {f"{base}/{MANIFEST}"}
        for key in self._list(base):
            if key not in keep:
                self._s3.delete_object(Bucket=self.bucket, Key=key)
        return OutputRef(location=f"s3://{self.bucket}/{base}", files=files)

    def _list(self, base: str) -> list[str]:
        """Keys under ``base/`` except failed-attempt dirs."""
        keys = []
        for page in self._s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=base + "/"):
            keys += [o["Key"] for o in page.get("Contents", []) if not o["Key"][len(base) + 1 :].startswith(".attempt-")]
        return keys

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        return code in ("404", "NoSuchKey", "NotFound")

    def _sha_and_size(self, key: str) -> tuple[str, int]:
        body = self._s3.get_object(Bucket=self.bucket, Key=key)["Body"]
        h, n = hashlib.sha256(), 0
        for chunk in iter(lambda: body.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
        return h.hexdigest(), n

    def verify(self, ref: OutputRef) -> list[str]:
        problems = []
        try:
            base = self._split(ref.location)
        except ValueError as exc:
            return [str(exc)]
        try:
            body = self._s3.get_object(Bucket=self.bucket, Key=f"{base}/{MANIFEST}")["Body"].read()
            if _norm(json.loads(body)["files"]) != _norm(ref.files):
                problems.append(f"{MANIFEST} does not match the recorded files")
        except Exception as exc:
            if not self._is_missing(exc) and not isinstance(exc, (ValueError, KeyError, TypeError)):
                raise
            problems.append(f"{MANIFEST} is missing or unreadable")
        listed = set()
        for f in ref.files:
            try:
                rel = _check_rel(f["path"])
            except ValueError as exc:
                problems.append(str(exc))
                continue
            listed.add(f"{base}/{rel}")
            try:
                sha, size = self._sha_and_size(f"{base}/{rel}")
            except Exception as exc:
                if not self._is_missing(exc):
                    raise
                problems.append(f"{f['path']}: missing")
                continue
            if size != f["bytes"]:
                problems.append(f"{f['path']}: size {size} != {f['bytes']}")
            elif sha != f["sha256"]:
                problems.append(f"{f['path']}: sha256 mismatch")
        for key in self._list(base):
            if key != f"{base}/{MANIFEST}" and key not in listed:
                problems.append(f"{key[len(base) + 1 :]}: unexpected file")
        return problems

    def fetch_previous(self, ref: OutputRef, dest: Path) -> None:
        problems = self.verify(ref)
        if problems:
            raise ValueError("previous outputs failed verification: " + "; ".join(problems))
        base, dest = self._split(ref.location), Path(dest)
        for f in ref.files:
            rel = _check_rel(f["path"])
            out = (dest / rel).resolve()
            if not out.is_relative_to(dest.resolve()):
                raise ValueError(f"path {rel!r} escapes the store")
            out.parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(self.bucket, f"{base}/{rel}", str(out))


def load_output_store(spec: str, config: Any) -> OutputStore:
    """Resolve "local", "s3" or "module:factory" (the factory receives the config)."""
    if spec == "local":
        return LocalOutputStore(
            config.project_dir / "gpu-outputs", bundles_dir=config.state_dir / "queue" / "bundles"
        )
    if spec == "s3":
        import os

        from ..core.credentials import CredentialResolver

        profile = getattr(getattr(config, "provider", None), "profile", "default")
        url = CredentialResolver(config.project_dir, profile=profile).get("GPU_OUTPUTS_URL") or os.environ.get("GPU_OUTPUTS_URL")
        if not url:
            raise ConfigError("GPU_OUTPUTS_URL is not set (s3://bucket/prefix); put it in the environment or ~/.gpu/credentials")
        return S3OutputStore(url)
    if ":" not in spec:
        raise ConfigError(f"unknown output store {spec!r}; use local, s3 or module:factory")
    module_name, _, attr = spec.partition(":")
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load output store {spec!r}: {exc}") from exc
    return factory(config)
