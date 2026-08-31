"""Per-user application-data directory resolution.

Every member runs the app locally through the launcher; the runtime
configuration store, the secret key file, and the cloud-asset cache all live
under a single per-user directory that is **outside the repository** so a
``git clean`` or a fresh checkout never wipes a machine's configuration.

Resolution order:

1. ``HCMAI_APP_DATA_DIR`` environment variable (tests and power users).
2. Platform convention:
   * Windows  -> ``%LOCALAPPDATA%\\HCMAI2026``
   * macOS    -> ``~/Library/Application Support/HCMAI2026``
   * Linux    -> ``$XDG_DATA_HOME/hcmai2026`` or ``~/.local/share/hcmai2026``
3. Last-resort fallback -> ``~/.hcmai2026``.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

APP_DIR_NAME = "HCMAI2026"
APP_DIR_NAME_POSIX = "hcmai2026"

ENV_OVERRIDE = "HCMAI_APP_DATA_DIR"


def _platform_default() -> Path:
    home = Path.home()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_DIR_NAME
        return home / "AppData" / "Local" / APP_DIR_NAME
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_DIR_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_DIR_NAME_POSIX
    return home / ".local" / "share" / APP_DIR_NAME_POSIX


@lru_cache(maxsize=1)
def get_app_data_dir() -> Path:
    """Return the per-user app-data directory, creating it if missing."""
    override = os.environ.get(ENV_OVERRIDE)
    target = Path(override).expanduser() if override else _platform_default()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        target = Path.home() / ".hcmai2026"
        target.mkdir(parents=True, exist_ok=True)
    return target


def get_config_db_path() -> Path:
    """Path to the SQLite runtime-configuration database."""
    return get_app_data_dir() / "config.db"


def get_secret_key_path() -> Path:
    """Path to the local secret-encryption key file (OS keyring fallback)."""
    return get_app_data_dir() / "secret.key"


def get_assets_cache_dir() -> Path:
    """Root directory for the local cloud-asset cache."""
    path = get_app_data_dir() / "assets-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_control_dir() -> Path:
    """Directory the launcher and the API use to exchange restart signals."""
    path = get_app_data_dir() / "control"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_cache() -> None:
    """Clear the memoised directory (tests that swap ``HCMAI_APP_DATA_DIR``)."""
    get_app_data_dir.cache_clear()
