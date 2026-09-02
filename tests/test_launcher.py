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
        frontend_builder=lambda spec, built_for: built_for,  # never shell out to npm in tests
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

    def test_frontend_start_failure_is_non_fatal(self, env) -> None:
        env.create_revision(
            {"PORT": "3000", "HOST": "0.0.0.0", "LAUNCHER_FRONTEND_ENABLED": "true",
             "LAUNCHER_HEALTH_TIMEOUT_SECONDS": "2", "LAUNCHER_HEALTH_POLL_INTERVAL_SECONDS": "0.5"},
            source="ui",
        )
        get_settings.cache_clear()
        launcher = _make_launcher(broken=set(), frontend=True)

        def boom(_spec):
            raise FileNotFoundError("npm not found")

        launcher._frontend_factory = boom

        assert launcher.boot() is True  # backend still healthy despite the frontend blowing up
        assert launcher.backend.is_alive()
        assert launcher.frontend is None
        assert launcher._frontend_failed is True
        # supervise must not keep retrying a permanently-failed frontend
        launcher.tick()
        assert launcher._frontend_failed is True

    def test_preview_builder_runs_once_then_skips(self, env, tmp_path) -> None:
        from launcher.core import _default_frontend_builder, RuntimeSpec

        fe_dir = tmp_path / "frontend"
        (fe_dir / "src").mkdir(parents=True)
        (fe_dir / "src" / "app.jsx").write_text("x", encoding="utf-8")

        calls = []

        def fake_run_build(cmd, *, cwd, env=None, timeout=600.0):
            import time as _t

            calls.append(cmd)
            dist = Path(cwd) / "dist"
            dist.mkdir(parents=True, exist_ok=True)
            (dist / "index.html").write_text("<html></html>", encoding="utf-8")
            future = _t.time() + 3600  # ensure dist is newer than src
            os.utime(dist / "index.html", (future, future))

        import launcher.core as lc

        orig = lc.run_build
        lc.run_build = fake_run_build
        try:
            spec = RuntimeSpec(
                host="0.0.0.0", port=3000, api_base_url="http://127.0.0.1:3000",
                frontend_enabled=True, frontend_mode="preview", frontend_dir=fe_dir,
                frontend_port=5173, health_timeout=2.0, health_poll_interval=0.5, active_revision_id=1,
            )
            built_for = _default_frontend_builder(spec, None)
            assert built_for == "http://127.0.0.1:3000" and len(calls) == 1
            # unchanged src + same url -> no rebuild
            built_for = _default_frontend_builder(spec, built_for)
            assert len(calls) == 1
            # different api url -> rebuild
            spec2 = RuntimeSpec(**{**spec.__dict__, "api_base_url": "http://127.0.0.1:4000"})
            _default_frontend_builder(spec2, built_for)
            assert len(calls) == 2
        finally:
            lc.run_build = orig
