"""The launcher state machine.

``Launcher`` owns the backend (and optional frontend) processes and reacts to
restart requests with health-checked rollout + automatic rollback. Process
creation and health probing are injected so the whole flow is unit-testable
without spawning uvicorn or npm.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from launcher.health import check_once, connect_host, health_url
from launcher.process import ManagedProcess, default_backend_cmd, default_frontend_cmd
from src.services import launcher_control

logger = logging.getLogger("launcher")

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RuntimeSpec:
    host: str
    port: int
    api_base_url: str
    frontend_enabled: bool
    frontend_dir: Path
    frontend_port: int
    health_timeout: float
    health_poll_interval: float
    active_revision_id: int | None

    def endpoint_changed(self, other: "RuntimeSpec | None") -> bool:
        return other is None or (self.host, self.port) != (other.host, other.port)


def resolve_spec() -> RuntimeSpec:
    from src.config.settings import get_settings

    get_settings.cache_clear()
    s = get_settings()
    host = s.host or "0.0.0.0"
    port = int(s.port or 3000)
    active_revision_id = None
    try:
        from src.config.runtime_store import get_store

        store = get_store()
        active_revision_id = store.active_revision_id() if store else None
    except Exception:  # noqa: BLE001
        pass
    return RuntimeSpec(
        host=host,
        port=port,
        api_base_url=f"http://{connect_host(host)}:{port}",
        frontend_enabled=bool(s.launcher_frontend_enabled),
        frontend_dir=s.get_launcher_frontend_dir(),
        frontend_port=int(s.launcher_frontend_port or 5173),
        health_timeout=float(s.launcher_health_timeout_seconds or 60.0),
        health_poll_interval=float(s.launcher_health_poll_interval_seconds or 1.0),
        active_revision_id=active_revision_id,
    )


ProcFactory = Callable[[RuntimeSpec], ManagedProcess]
HealthProbe = Callable[[RuntimeSpec], bool]


def _default_backend_factory(spec: RuntimeSpec) -> ManagedProcess:
    return ManagedProcess("backend", default_backend_cmd(spec.host, spec.port), cwd=REPO_ROOT)


def _default_frontend_factory(spec: RuntimeSpec) -> ManagedProcess:
    return ManagedProcess(
        "frontend",
        default_frontend_cmd(spec.frontend_port),
        cwd=spec.frontend_dir,
        env={"VITE_SEARCH_API_BASE_URL": spec.api_base_url, "VITE_SEARCH_MODE": "live"},
    )


def _default_health_probe(spec: RuntimeSpec) -> bool:
    return check_once(health_url(spec.host, spec.port))


class Launcher:
    def __init__(
        self,
        *,
        backend_factory: ProcFactory = _default_backend_factory,
        frontend_factory: ProcFactory = _default_frontend_factory,
        health_probe: HealthProbe = _default_health_probe,
        spec_resolver: Callable[[], RuntimeSpec] = resolve_spec,
        poll_interval: float = 1.0,
        enable_frontend: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend_factory = backend_factory
        self._frontend_factory = frontend_factory
        self._health_probe = health_probe
        self._resolve = spec_resolver
        self.poll_interval = poll_interval
        self.enable_frontend = enable_frontend
        self._sleep = sleep
        self._monotonic = monotonic

        self.spec: RuntimeSpec | None = None
        self.backend: ManagedProcess | None = None
        self.frontend: ManagedProcess | None = None
        # The revision we last confirmed healthy -- the rollback target.
        self.current_revision_id: int | None = None
        self.stop = False
        self._restart_backoff = 0

    # -- process helpers ------------------------------------------------
    def _start_backend(self, spec: RuntimeSpec) -> None:
        self.backend = self._backend_factory(spec)
        self.backend.start()
        logger.info("backend started (pid=%s) on %s:%s", self.backend.pid, spec.host, spec.port)

    def _stop_backend(self) -> None:
        if self.backend:
            self.backend.stop()
            self.backend = None

    def _start_frontend(self, spec: RuntimeSpec) -> None:
        if not (self.enable_frontend and spec.frontend_enabled):
            return
        self.frontend = self._frontend_factory(spec)
        self.frontend.start()
        logger.info("frontend started (pid=%s) api=%s", self.frontend.pid, spec.api_base_url)

    def _stop_frontend(self) -> None:
        if self.frontend:
            self.frontend.stop()
            self.frontend = None

    def _wait_healthy(self, spec: RuntimeSpec) -> bool:
        deadline = self._monotonic() + max(0.5, spec.health_timeout)
        while self._monotonic() < deadline:
            if self.backend is not None and not self.backend.is_alive():
                logger.warning("backend process exited before becoming healthy")
                return False
            if self._health_probe(spec):
                return True
            self._sleep(max(0.05, spec.health_poll_interval))
        return self._health_probe(spec)

    # -- lifecycle ----------------------------------------------------------
    def boot(self) -> bool:
        self.spec = self._resolve()
        launcher_control.write_status("restarting", phase="initial-boot",
                                     active_revision_id=self.spec.active_revision_id)
        self._start_backend(self.spec)
        launcher_control.write_status("polling-health", phase="initial-boot")
        if self._wait_healthy(self.spec):
            self.current_revision_id = self.spec.active_revision_id
            self._start_frontend(self.spec)
            launcher_control.write_status(
                "healthy", active_revision_id=self.current_revision_id,
                host=self.spec.host, port=self.spec.port, api_base_url=self.spec.api_base_url,
            )
            return True
        logger.error("initial boot did not become healthy; attempting rollback")
        return self._rollback(failed_revision_id=self.spec.active_revision_id, initial=True)

    def tick(self) -> None:
        launcher_control.touch_heartbeat()
        req = launcher_control.read_request()
        if req is not None:
            launcher_control.clear_request()
            self._handle_restart(req)
            return
        self._supervise()

    def _supervise(self) -> None:
        if self.backend is not None and not self.backend.is_alive():
            self._restart_backoff = min(self._restart_backoff + 1, 5)
            logger.warning("backend died unexpectedly; restarting (attempt %s)", self._restart_backoff)
            spec = self.spec or self._resolve()
            self._start_backend(spec)
            if self._wait_healthy(spec):
                self._restart_backoff = 0
                launcher_control.write_status("healthy", active_revision_id=self.current_revision_id,
                                             host=spec.host, port=spec.port, detail="recovered after crash")
        else:
            self._restart_backoff = 0
        if (
            self.enable_frontend
            and self.spec is not None
            and self.spec.frontend_enabled
            and self.frontend is not None
            and not self.frontend.is_alive()
        ):
            logger.warning("frontend died unexpectedly; restarting")
            self._start_frontend(self.spec)

    def _handle_restart(self, req: dict) -> None:
        old_spec = self.spec
        new_spec = self._resolve()
        logger.info("restart requested (reason=%s target_revision=%s)",
                    req.get("reason"), new_spec.active_revision_id)
        launcher_control.write_status("restarting", reason=req.get("reason"),
                                     target_revision_id=new_spec.active_revision_id,
                                     from_revision_id=self.current_revision_id)
        endpoint_changed = new_spec.endpoint_changed(old_spec)
        self._stop_backend()
        if endpoint_changed:
            self._stop_frontend()
        self._start_backend(new_spec)
        launcher_control.write_status("polling-health", target_revision_id=new_spec.active_revision_id)

        if self._wait_healthy(new_spec):
            self.spec = new_spec
            self.current_revision_id = new_spec.active_revision_id
            if endpoint_changed:
                self._start_frontend(new_spec)
            launcher_control.write_status(
                "healthy", active_revision_id=self.current_revision_id,
                host=new_spec.host, port=new_spec.port, api_base_url=new_spec.api_base_url,
                endpoint_changed=endpoint_changed,
            )
            logger.info("restart healthy on revision %s", self.current_revision_id)
            return

        logger.error("new revision %s not healthy; rolling back to %s",
                     new_spec.active_revision_id, self.current_revision_id)
        self._rollback(failed_revision_id=new_spec.active_revision_id, initial=False,
                       endpoint_changed=endpoint_changed)

    def _rollback(self, *, failed_revision_id, initial: bool, endpoint_changed: bool = False) -> bool:
        target = self.current_revision_id
        launcher_control.write_status("rolling-back", failed_revision_id=failed_revision_id,
                                     restore_target=target)
        self._stop_backend()

        restored_rev = None
        if target is not None:
            try:
                from src.config.runtime_store import get_store

                store = get_store()
                if store is not None:
                    restored_rev = store.restore_revision(target, note="launcher auto-rollback")
            except Exception as exc:  # noqa: BLE001
                logger.error("could not restore revision %s: %s", target, exc)
        elif initial:
            logger.error("no known-good revision to roll back to")

        restored_spec = self._resolve()
        self._start_backend(restored_spec)
        if self._wait_healthy(restored_spec):
            self.spec = restored_spec
            self.current_revision_id = restored_spec.active_revision_id
            if endpoint_changed or initial:
                self._stop_frontend()
                self._start_frontend(restored_spec)
            launcher_control.write_status(
                "rollback-complete", active_revision_id=self.current_revision_id,
                failed_revision_id=failed_revision_id, restored_revision_id=restored_rev,
                host=restored_spec.host, port=restored_spec.port,
            )
            logger.info("rollback healthy on revision %s", self.current_revision_id)
            return True

        launcher_control.write_status("failed", failed_revision_id=failed_revision_id,
                                      detail="rollback did not become healthy")
        logger.critical("rollback did not become healthy; manual intervention required")
        return False

    def shutdown(self) -> None:
        self.stop = True
        self._stop_frontend()
        self._stop_backend()
        launcher_control.write_status("idle", detail="launcher stopped")

    def run(self) -> int:
        healthy = self.boot()
        if not healthy:
            logger.error("app is not healthy after boot/rollback; launcher will keep supervising")
        try:
            while not self.stop:
                self.tick()
                slept = 0.0
                while slept < self.poll_interval and not self.stop:
                    self._sleep(min(0.25, self.poll_interval - slept))
                    slept += 0.25
        finally:
            self.shutdown()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m launcher", description=__doc__)
    parser.add_argument("--no-frontend", action="store_true", help="do not start the local frontend")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s [launcher] %(message)s")

    launcher = Launcher(poll_interval=args.poll_interval, enable_frontend=not args.no_frontend)

    def _handle_signal(signum, _frame):
        logger.info("received signal %s; shutting down", signum)
        launcher.stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass

    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
