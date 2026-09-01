"""Restart signalling shared by the management API and the local launcher.

The API cannot restart the FastAPI process itself. Instead it drops a
``restart.request`` file in the control directory; the launcher (see
``launcher/``) polls for it, saves the pending revision, restarts the managed
processes, polls ``/health`` and -- if the new configuration does not come up
healthy within the timeout -- restores the previous revision and restarts
again. The launcher reports progress back through ``restart.status``.

When no launcher is running, ``request_restart`` still records the intent and
``read_status`` reports ``launcher_running=False`` so the UI can tell the user
to restart manually.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from src.config.app_paths import get_control_dir

_REQUEST_NAME = "restart.request"
_STATUS_NAME = "restart.status"
_HEARTBEAT_NAME = "launcher.heartbeat"
# The launcher is considered alive if its heartbeat is newer than this.
_HEARTBEAT_STALE_SECONDS = 15.0


def _path(name: str):
    return get_control_dir() / name


def request_restart(reason: str = "config-change", *, target_revision_id: int | None = None) -> dict[str, Any]:
    payload = {
        "requested_at": time.time(),
        "reason": reason,
        "target_revision_id": target_revision_id,
        "pid": os.getpid(),
    }
    tmp = _path(_REQUEST_NAME + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, _path(_REQUEST_NAME))
    return payload


def read_request() -> dict[str, Any] | None:
    try:
        return json.loads(_path(_REQUEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_request() -> None:
    try:
        _path(_REQUEST_NAME).unlink()
    except OSError:
        pass


def write_status(state: str, **fields: Any) -> dict[str, Any]:
    """Called by the launcher. ``state`` is one of:
    idle | restarting | polling-health | healthy | rolling-back | rollback-complete | failed.
    """
    payload = {"state": state, "updated_at": time.time(), **fields}
    tmp = _path(_STATUS_NAME + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, _path(_STATUS_NAME))
    return payload


def touch_heartbeat() -> None:
    _path(_HEARTBEAT_NAME).write_text(str(time.time()), encoding="utf-8")


def launcher_running() -> bool:
    try:
        beat = float(_path(_HEARTBEAT_NAME).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return (time.time() - beat) <= _HEARTBEAT_STALE_SECONDS


def read_status() -> dict[str, Any]:
    status: dict[str, Any] = {"state": "idle", "updated_at": None}
    try:
        status = json.loads(_path(_STATUS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    status["launcher_running"] = launcher_running()
    status["pending_request"] = read_request() is not None
    return status
