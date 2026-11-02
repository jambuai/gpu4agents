"""Model server readiness checks.

A passing health check means the runtime is *available*; per spec section 12
it is explicitly NOT workload activity and never refreshes the idle timer.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import httpx

from .errors import RuntimeStartupError


def probe(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        return False, type(exc).__name__
    if response.status_code < 400:
        return True, f"{response.status_code}"
    return False, f"HTTP {response.status_code}"


def wait_for_healthy(
    url: str,
    *,
    timeout: float,
    interval: float = 10.0,
    on_attempt: Optional[Callable[[int, float, str], None]] = None,
    should_abort: Optional[Callable[[], Optional[str]]] = None,
) -> None:
    """Poll ``url`` until it answers or ``timeout`` elapses.

    ``should_abort`` lets the caller fail fast when the instance itself died
    instead of waiting out the full startup timeout.
    """
    deadline = time.time() + timeout
    attempt = 0
    last_detail = "not started"
    interval = max(2.0, min(interval, 30.0))

    while time.time() < deadline:
        attempt += 1
        if should_abort is not None:
            reason = should_abort()
            if reason:
                raise RuntimeStartupError(f"model server did not start: {reason}")
        ok, last_detail = probe(url)
        if ok:
            return
        if on_attempt is not None:
            on_attempt(attempt, max(0.0, deadline - time.time()), last_detail)
        time.sleep(interval)

    raise RuntimeStartupError(
        f"model server did not become healthy within {int(timeout)}s "
        f"({url}, last probe: {last_detail}). "
        "Check `gpu logs` - model downloads can be slow on first boot."
    )
