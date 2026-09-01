"""Shared pytest fixtures / test isolation for the backend suite.

The runtime configuration store (``src/config/runtime_store.py``) resolves a
per-user app-data directory and, when enabled, layers an active SQLite revision
on top of ``.env`` and the code defaults. Neither of those belongs in a unit
test run: it would read/write a real database under the developer's profile and
make ``get_settings()`` depend on machine state.

This conftest points the app-data directory at a throwaway location and disables
the store layering by default, so every legacy test keeps seeing plain
``.env`` / environment-variable behaviour. Tests that specifically exercise the
store opt back in explicitly (see ``tests/test_runtime_config_store.py``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_APP_DATA = Path(tempfile.gettempdir()) / "hcmai2026-test-appdata"


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    _TMP_APP_DATA.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HCMAI_APP_DATA_DIR", str(_TMP_APP_DATA))
    # Legacy tests must not pick up an on-disk revision.
    os.environ.setdefault("HCMAI_DISABLE_CONFIG_STORE", "1")
