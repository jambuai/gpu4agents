"""One Observer for queue events and metrics (events.jsonl + Prometheus text)."""

from __future__ import annotations

import os
import math
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from ..core.events import EventLog

_SECRET_KEYS = {"token", "password", "secret", "api_key"}


def _redact_url(value: str) -> str:
    try:
        # The userinfo ends at the LAST "@" of the authority; the password may hold "/" or "@".
        scheme, sep, rest = value.partition("://")
        if not sep:
            return value
        cut = len(rest)
        for ch in "?#":
            if ch in rest:
                cut = min(cut, rest.index(ch))
        authority_and_path, tail = rest[:cut], rest[cut:]
        at = authority_and_path.rfind("@")
        if at < 0:
            return value
        userinfo, hostpath = authority_and_path[:at], authority_and_path[at + 1:]
        if ":" not in userinfo:
            return value
        user = userinfo.split(":", 1)[0]
        return f"{scheme}://{user}:***@{hostpath}{tail}"
    except Exception:
        return "***"


def _redact_value(key: str, value: Any) -> Any:
    k = key.lower()
    if k in _SECRET_KEYS or k.endswith("_token") or k.endswith("_secret"):
        return "***"
    if (k == "url" or k.endswith("_url")) and isinstance(value, str):
        return _redact_url(value)
    return value


def _series(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    def esc(v: Any) -> str:
        return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    inner = ",".join(f'{k}="{esc(v)}"' for k, v in sorted(labels.items()))
    return f"{name}{{{inner}}}"


class Observer:
    def __init__(self, events: Optional[EventLog], metrics_path: Path) -> None:
        self.events = events
        self.metrics_path = Path(metrics_path)
        self._lock = threading.Lock()
        self._values: dict[str, float] = {}

    def emit(self, event: str, **fields: Any) -> dict:
        clean = {k: _redact_value(k, v) for k, v in fields.items()}
        if self.events is None:
            return {"event": event, **clean}
        return self.events.emit(event, **clean)

    def inc(self, name: str, by: float = 1, **labels: Any) -> None:
        key = _series(name, labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + by

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._values[_series(name, labels)] = value

    def timing(self, name: str, seconds: float, **labels: Any) -> None:
        with self._lock:
            self._values[_series(name + "_count", labels)] = self._values.get(_series(name + "_count", labels), 0) + 1
            self._values[_series(name + "_sum", labels)] = self._values.get(_series(name + "_sum", labels), 0) + seconds
            mk = _series(name + "_max", labels)
            self._values[mk] = max(self._values.get(mk, 0), seconds)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._values)

    @staticmethod
    def _fmt(v: float) -> str:
        v = float(v)
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "+Inf" if v > 0 else "-Inf"
        return repr(v)

    def render(self) -> str:
        with self._lock:
            return "".join(f"{k} {self._fmt(v)}\n" for k, v in sorted(self._values.items()))

    def write(self) -> None:
        with self._lock:
            body = "".join(f"{k} {self._fmt(v)}\n" for k, v in sorted(self._values.items()))
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.metrics_path.parent, prefix=".metrics-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as handle:
                    handle.write(body)
                os.replace(tmp, self.metrics_path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise


class NullObserver(Observer):
    """Records nothing on disk; for tests."""

    def __init__(self) -> None:
        super().__init__(None, Path(os.devnull))

    def write(self) -> None:
        return None
