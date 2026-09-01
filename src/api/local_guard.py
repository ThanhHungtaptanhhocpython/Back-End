"""Loopback-only protection for the local management API.

Every member runs the app on their own PC; there is no login. The Settings /
Cloud Assets / provider management endpoints are instead restricted to
loopback clients and localhost origins. A request from any other address --
or a browser request whose ``Origin`` / ``Referer`` is not localhost -- is
rejected with ``403``.

``HCMAI_SETTINGS_EXTRA_HOSTS`` (comma-separated) can whitelist additional
client hosts for unusual setups (e.g. a container bridge address); it is empty
by default.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

from fastapi import HTTPException, Request, status

_LOCAL_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]", ""}


def _extra_hosts() -> set[str]:
    raw = os.environ.get("HCMAI_SETTINGS_EXTRA_HOSTS", "")
    return {h.strip() for h in raw.split(",") if h.strip()}


def _is_local_host(host: str | None) -> bool:
    if host is None:
        return False
    host = host.strip().strip("[]")
    if host.lower() in _LOCAL_NAMES:
        return True
    if host in _extra_hosts():
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_local_client(request: Request) -> None:
    client_host = request.client.host if request.client else None
    if not _is_local_host(client_host):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The management API is restricted to local (loopback) clients.",
        )
    for header in ("origin", "referer"):
        value = request.headers.get(header)
        if not value:
            continue
        host = urlparse(value).hostname
        if not _is_local_host(host):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"The management API rejects cross-origin requests ({header}: {host}).",
            )
