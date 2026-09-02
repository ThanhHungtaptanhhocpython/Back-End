"""Thin ``subprocess.Popen`` wrapper used for the backend and frontend."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class ManagedProcess:
    def __init__(self, name: str, cmd: list[str], *, cwd: Path | None = None, env: dict | None = None):
        self.name = name
        self.cmd = cmd
        self.cwd = str(cwd) if cwd else None
        self.env = env
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        full_env = os.environ.copy()
        if self.env:
            full_env.update({k: str(v) for k, v in self.env.items()})
        creationflags = 0
        start_new_session = False
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            start_new_session = True
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=full_env,
            stdout=None,
            stderr=None,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def stop(self, *, timeout: float = 10.0) -> None:
        if not self.proc:
            return
        if self.proc.poll() is not None:
            self.proc = None
            return
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError, ValueError):
            self.proc.terminate()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.proc = None
                return
            time.sleep(0.1)
        try:
            self.proc.kill()
        except OSError:
            pass
        self.proc = None


def default_backend_cmd(host: str, port: int) -> list[str]:
    return [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", str(host), "--port", str(int(port)),
    ]


def _npm() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def default_frontend_cmd(port: int) -> list[str]:
    """Hot-reload dev server (LAUNCHER_FRONTEND_MODE=dev)."""
    return [_npm(), "run", "dev", "--", "--port", str(int(port)), "--strictPort"]


def frontend_build_cmd() -> list[str]:
    """One-shot production build -> frontend/dist."""
    return [_npm(), "run", "build"]


def frontend_preview_cmd(port: int) -> list[str]:
    """Serve the built frontend/dist (LAUNCHER_FRONTEND_MODE=preview)."""
    return [_npm(), "run", "preview", "--", "--port", str(int(port)), "--strictPort"]


def run_build(cmd: list[str], *, cwd: Path, env: dict | None = None, timeout: float = 600.0) -> None:
    """Run a blocking build step. Raises CalledProcessError / FileNotFoundError
    / TimeoutExpired on failure -- the caller decides whether that is fatal."""
    full_env = os.environ.copy()
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    subprocess.run(cmd, cwd=str(cwd), env=full_env, check=True, timeout=timeout)
