"""Health probing for the managed backend."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def connect_host(bind_host: str) -> str:
    """A bind address like ``0.0.0.0`` / ``::`` is not connectable -- map it to
    loopback for health checks and the frontend API base URL."""
    host = (bind_host or "").strip()
    if host in ("", "0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    return host


def health_url(bind_host: str, port: int) -> str:
    return f"http://{connect_host(bind_host)}:{int(port)}/health"


def check_once(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode("utf-8"))
        return bool(body.get("success", True))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def wait_healthy(
    url: str,
    *,
    timeout: float,
    interval: float,
    is_alive=None,
    sleep=time.sleep,
    now=time.monotonic,
) -> bool:
    """Poll ``url`` until it is healthy or ``timeout`` elapses.

    Returns ``False`` immediately if ``is_alive()`` reports the process died.
    """
    deadline = now() + max(0.1, timeout)
    while now() < deadline:
        if is_alive is not None and not is_alive():
            return False
        if check_once(url, timeout=min(3.0, interval + 1.0)):
            return True
        sleep(max(0.05, interval))
    return check_once(url)
