"""Phase 5 -- local runtime launcher: restart, port change, health-timeout rollback."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from launcher.core import Launcher, resolve_spec  # noqa: E402
from src.config import app_paths  # noqa: E402
from src.config import runtime_store as rs  # noqa: E402
from src.config.settings import get_settings  # noqa: E402
from src.services import launcher_control  # noqa: E402


class FakeProc:
    def __init__(self, name: str):
        self.name = name
        self.started = 0
        self.stopped = 0
        self._alive = False

    def start(self):
        self.started += 1
        self._alive = True

    def stop(self, *, timeout: float = 10.0):
        self.stopped += 1
        self._alive = False

    def is_alive(self):
        return self._alive

    @property
    def pid(self):
        return 4242


class Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(float(seconds), 0.001)


@pytest.fixture()
def env(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.delenv("HCMAI_DISABLE_CONFIG_STORE", raising=False)
    app_paths.get_app_data_dir.cache_clear()
    rs.reset_store()
    get_settings.cache_clear()
    store = rs.get_store()
    assert store is not None
    store.bootstrap_from_env(
        {
            "PORT": "3000",
            "HOST": "0.0.0.0",
            "LAUNCHER_HEALTH_TIMEOUT_SECONDS": "2",
            "LAUNCHER_HEALTH_POLL_INTERVAL_SECONDS": "0.5",
        }
    )
    get_settings.cache_clear()
    yield store
    rs.reset_store()
    app_paths.get_app_data_dir.cache_clear()
    get_settings.cache_clear()


def _make_launcher(broken: set[int], *, frontend=False):
    clock = Clock()
    fe_procs: list[FakeProc] = []
    be_procs: list[FakeProc] = []

    def backend_factory(spec):
        p = FakeProc("backend")
        be_procs.append(p)
        return p

    def frontend_factory(spec):
        p = FakeProc("frontend")
        fe_procs.append(p)
        return p

    def probe(spec):
        return spec.active_revision_id not in broken

    launcher = Launcher(
        backend_factory=backend_factory,
        frontend_factory=frontend_factory,
        health_probe=probe,
        spec_resolver=resolve_spec,
        enable_frontend=frontend,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        poll_interval=0.5,
    )
    launcher._be_procs = be_procs  # type: ignore[attr-defined]
    launcher._fe_procs = fe_procs  # type: ignore[attr-defined]
    return launcher


class TestLauncher:
    def test_boot_healthy(self, env) -> None:
        launcher = _make_launcher(broken=set())
        assert launcher.boot() is True
        assert launcher.current_revision_id == env.active_revision_id()
        assert launcher.backend.is_alive()
        assert launcher_control.read_status()["state"] == "healthy"

    def test_restart_applies_new_revision(self, env) -> None:
        launcher = _make_launcher(broken=set())
        launcher.boot()
        rev2 = env.create_revision({"PORT": "4000", "HOST": "0.0.0.0"}, source="ui")
        launcher_control.request_restart("config-change", target_revision_id=rev2)

        launcher.tick()

        assert launcher.current_revision_id == rev2
        assert launcher.spec.port == 4000
        assert launcher_control.read_request() is None
        status = launcher_control.read_status()
        assert status["state"] == "healthy" and status["active_revision_id"] == rev2

    def test_health_timeout_triggers_revision_rollback(self, env) -> None:
        launcher = _make_launcher(broken=set())
        launcher.boot()
        good = launcher.current_revision_id
        rev2 = env.create_revision({"PORT": "4000"}, source="ui")
        # mark the new revision unhealthy
        launcher._health_probe = lambda spec: spec.active_revision_id not in {rev2}
        launcher_control.request_restart("config-change", target_revision_id=rev2)

        launcher.tick()

        # a fresh revision copying the last-good one is now active, and healthy
        assert launcher.current_revision_id > rev2
        assert env.effective_values()["PORT"] == "3000"
        status = launcher_control.read_status()
        assert status["state"] == "rollback-complete"
        assert status["failed_revision_id"] == rev2

    def test_port_change_restarts_frontend(self, env) -> None:
        env.create_revision(
            {"PORT": "3000", "HOST": "0.0.0.0", "LAUNCHER_FRONTEND_ENABLED": "true",
             "LAUNCHER_HEALTH_TIMEOUT_SECONDS": "2", "LAUNCHER_HEALTH_POLL_INTERVAL_SECONDS": "0.5"},
            source="ui",
        )
        get_settings.cache_clear()
        launcher = _make_launcher(broken=set(), frontend=True)
        launcher.boot()
        assert len(launcher._fe_procs) == 1

        rev = env.create_revision(
            {"PORT": "4100", "HOST": "0.0.0.0", "LAUNCHER_FRONTEND_ENABLED": "true",
             "LAUNCHER_HEALTH_TIMEOUT_SECONDS": "2", "LAUNCHER_HEALTH_POLL_INTERVAL_SECONDS": "0.5"},
            source="ui",
        )
        launcher_control.request_restart("config-change", target_revision_id=rev)
        launcher.tick()

        assert launcher.spec.port == 4100
        assert len(launcher._fe_procs) == 2  # frontend was restarted for the new API base URL
        assert launcher._fe_procs[0].stopped == 1

    def test_supervise_restarts_crashed_backend(self, env) -> None:
        launcher = _make_launcher(broken=set())
        launcher.boot()
        launcher.backend._alive = False  # simulate crash
        launcher.tick()
        assert launcher.backend.is_alive()
        assert len(launcher._be_procs) == 2

    def test_shutdown_stops_children(self, env) -> None:
        launcher = _make_launcher(broken=set(), frontend=False)
        launcher.boot()
        backend = launcher.backend
        launcher.shutdown()
        assert backend.stopped == 1 and launcher.backend is None
        assert launcher_control.read_status()["state"] == "idle"
