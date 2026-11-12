"""Output stores for queue jobs: bundles in, results out, every delivery checked against a sha256 manifest."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ConfigError

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
    for p in sorted(src_dir.rglob("*")):
        if p.is_file() and not p.is_symlink() and p.name != MANIFEST:
            files.append({"path": p.relative_to(src_dir).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256_file(p)})
    return files


def _manifest_bytes(job_id: str, attempt: int, files: list[dict]) -> bytes:
    return json.dumps({"job_id": job_id, "attempt": attempt, "files": files}, indent=2, sort_keys=True).encode()


class OutputStore(ABC):
    @abstractmethod
    def put_bundle(self, job_id: str, path: Path) -> str: ...

    @abstractmethod
    def get_bundle(self, ref: str, dest: Path) -> Path: ...

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

    def put_outputs(self, job_id: str, attempt: int, src_dir: Path, success: bool = True) -> OutputRef:
        _check_job_id(job_id)
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
        if success and job_dir.exists():
            for child in job_dir.iterdir():  # keep failed-attempt dirs, replace the previous result
                if not child.name.startswith(".attempt-"):
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
            for child in staging.iterdir():
                shutil.move(str(child), job_dir / child.name)
            staging.rmdir()
        else:
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(target)
        return OutputRef(location=str(target), files=files)

    def verify(self, ref: OutputRef) -> list[str]:
        base = Path(ref.location)
        problems = []
        if not (base / MANIFEST).is_file():
            problems.append(f"{MANIFEST} is missing")
        for f in ref.files:
            try:
                p = self._inside(base, _check_rel(f["path"]))
            except ValueError as exc:
                problems.append(str(exc))
                continue
            if not p.is_file():
                problems.append(f"{f['path']}: missing")
            elif p.stat().st_size != f["bytes"]:
                problems.append(f"{f['path']}: size {p.stat().st_size} != {f['bytes']}")
            elif _sha256_file(p) != f["sha256"]:
                problems.append(f"{f['path']}: sha256 mismatch")
        return problems

    def fetch_previous(self, ref: OutputRef, dest: Path) -> None:
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

    def put_outputs(self, job_id: str, attempt: int, src_dir: Path, success: bool = True) -> OutputRef:
        _check_job_id(job_id)
        base = self._key(job_id) if success else self._key(job_id, f".attempt-{int(attempt)}")
        files = _scan(Path(src_dir))
        for f in files:
            self._put_file(f"{base}/{f['path']}", Path(src_dir) / f["path"])
        # manifest last: its presence means every file above was written
        self._s3.put_object(Bucket=self.bucket, Key=f"{base}/{MANIFEST}", Body=_manifest_bytes(job_id, attempt, files))
        return OutputRef(location=f"s3://{self.bucket}/{base}", files=files)

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
            self._s3.head_object(Bucket=self.bucket, Key=f"{base}/{MANIFEST}")
        except Exception:
            problems.append(f"{MANIFEST} is missing")
        for f in ref.files:
            try:
                rel = _check_rel(f["path"])
                sha, size = self._sha_and_size(f"{base}/{rel}")
            except ValueError as exc:
                problems.append(str(exc))
                continue
            except Exception:
                problems.append(f"{f['path']}: missing")
                continue
            if size != f["bytes"]:
                problems.append(f"{f['path']}: size {size} != {f['bytes']}")
            elif sha != f["sha256"]:
                problems.append(f"{f['path']}: sha256 mismatch")
        return problems

    def fetch_previous(self, ref: OutputRef, dest: Path) -> None:
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
