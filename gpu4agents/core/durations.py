"""Duration parsing/formatting shared by config and CLI flags."""

from __future__ import annotations

import re

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*([smhdw])", re.IGNORECASE)


def parse_duration(value: "str | int | float | None") -> "float | None":
    """Parse ``6h``, ``15m``, ``1h30m``, ``90`` (seconds) into seconds.

    ``None`` and the literal strings ``none``/``never``/``off`` mean "no limit".
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    if not text or text in {"none", "never", "off", "unlimited", "0s"}:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)

    matches = _TOKEN.findall(text)
    if not matches or "".join(f"{n}{u}" for n, u in matches).replace(" ", "") != text.replace(" ", ""):
        raise ValueError(f"invalid duration: {value!r} (expected e.g. 30s, 15m, 6h, 1h30m)")
    return float(sum(float(n) * _UNITS[u.lower()] for n, u in matches))


def format_duration(seconds: "float | None") -> str:
    """Render seconds as a compact human string (``1h30m``)."""
    if seconds is None:
        return "-"
    seconds = int(max(0, round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    parts: list[str] = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            amount, seconds = divmod(seconds, size)
            parts.append(f"{amount}{unit}")
        if len(parts) == 2:
            break
    return "".join(parts)
