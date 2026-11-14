"""The dashboard's single shared-token gate.

A token of ``None`` means open (the default loopback case). When a token is
set, a request authenticates by any of: ``Authorization: Bearer <token>``, a
one-time ``?token=<token>`` query (the server then sets a cookie so a phone
browser stays logged in), or the resulting ``gpu_dash`` cookie. Comparison is
constant-time.
"""

from __future__ import annotations

import hmac
from http.cookies import SimpleCookie
from typing import Optional
from urllib.parse import parse_qs

COOKIE_NAME = "gpu_dash"


class TokenGate:
    def __init__(self, token: Optional[str]) -> None:
        self.token = token

    def check(self, headers: dict, query: str) -> bool:
        if self.token is None:
            return True
        supplied = self._from_headers(headers) or self._from_query(query)
        return supplied is not None and hmac.compare_digest(supplied, self.token)

    def cookie_header(self, token: str) -> str:
        return f"{COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Strict"

    @staticmethod
    def _header(headers: dict, name: str) -> Optional[str]:
        for key, value in headers.items():
            if key.lower() == name.lower():
                return value
        return None

    def _from_headers(self, headers: dict) -> Optional[str]:
        auth = self._header(headers, "Authorization")
        if auth and auth.lower().startswith("bearer "):
            return auth[7:].strip()
        raw_cookie = self._header(headers, "Cookie")
        if raw_cookie:
            jar = SimpleCookie()
            try:
                jar.load(raw_cookie)
            except Exception:  # noqa: BLE001 - a malformed cookie is just "no token"
                return None
            if COOKIE_NAME in jar:
                return jar[COOKIE_NAME].value
        return None

    @staticmethod
    def _from_query(query: str) -> Optional[str]:
        values = parse_qs(query).get("token")
        return values[0] if values else None
