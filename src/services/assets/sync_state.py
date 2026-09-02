"""Process-wide progress state for the currently-running artifact sync.

One :class:`SyncProgress` singleton is updated from the byte-level ``progress``
callback that :func:`sync_artifacts` already emits, so two things can watch a
sync without being the thing that started it:

* the ``GET /settings/cloud/sync/status`` endpoint the Settings UI polls, and
* the startup warmer (see ``src/services/startup_warm.py``), which kicks a
  background sync the moment the app comes up in cloud mode.

Only one tracked sync runs at a time; a second :func:`run_tracked_sync` while
one is in flight raises rather than interleaving.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.services.assets.base import AssetStore, Manifest
from src.services.assets.local_cache import ArtifactCache
from src.services.assets.sync import SyncReport, sync_artifacts


@dataclass
class _ArtifactProgress:
    name: str
    total: int = 0
    done: int = 0
    status: str = "pending"  # pending|downloading|verifying|synced|cached|error
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        pct = (self.done / self.total * 100.0) if self.total else 0.0
        return {
            "name": self.name,
            "total": self.total,
            "done": min(self.done, self.total) if self.total else self.done,
            "pct": round(pct, 1),
            "status": self.status,
            "detail": self.detail,
        }


class SyncProgress:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.state = "idle"  # idle|running|done|error
        self.version: str | None = None
        self.trigger: str | None = None  # "manual" | "startup"
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str = ""
        self.report: dict | None = None
        self._artifacts: dict[str, _ArtifactProgress] = {}

    # -- lifecycle ------------------------------------------------------------
    def begin(self, manifest: Manifest, names: list[str], trigger: str) -> None:
        with self._lock:
            if self.state == "running":
                raise RuntimeError("a sync is already running")
            wanted = set(names)
            self._artifacts = {
                a.name: _ArtifactProgress(name=a.name, total=int(a.size or 0))
                for a in manifest.artifacts
                if a.name in wanted
            }
            self.state = "running"
            self.version = manifest.version
            self.trigger = trigger
            self.started_at = time.time()
            self.finished_at = None
            self.error = ""
            self.report = None

    def on_progress(self, name: str, phase: str, done: int, total: int) -> None:
        with self._lock:
            art = self._artifacts.get(name)
            if art is None:
                art = _ArtifactProgress(name=name)
                self._artifacts[name] = art
            if total:
                art.total = int(total)
            if phase == "download":
                art.status = "downloading"
                art.done = int(done)
            elif phase == "verify":
                art.status = "verifying"
                art.done = art.total or int(done)
            elif phase in ("synced", "cached"):
                art.status = phase
                art.done = art.total or int(done)
            elif phase == "error":
                art.status = "error"
                art.done = int(done)

    def finish(self, report: SyncReport | None, error: str = "") -> None:
        with self._lock:
            self.finished_at = time.time()
            if error:
                self.state = "error"
                self.error = error
                return
            self.state = "done"
            if report is not None:
                self.report = report.to_dict()
                # A validation error (unknown / empty names) never touched a
                # byte -- surface it so the run does not read as green.
                if report.errors and not self.error:
                    self.error = "; ".join(report.errors)
                for r in report.results:
                    art = self._artifacts.get(r.name)
                    if art is not None and art.status not in ("synced", "cached", "error"):
                        art.status = r.status if r.status in ("synced", "cached") else "error"
                        art.detail = r.detail

    # -- read --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            arts = [a.to_dict() for a in self._artifacts.values()]
            total = sum(a["total"] for a in arts)
            done = sum(a["done"] for a in arts)
            had_errors = bool(self.error) or any(a["status"] == "error" for a in arts) or (
                self.report is not None and not self.report.get("ok", True)
            )
            promoted = bool(self.report.get("promoted")) if self.report else False
            # A finished run with checksum/download failures is "completed with
            # errors" -- never a green success. `ok` is True only for a clean,
            # promoted (or fully-cached) run.
            ok = self.state == "done" and not had_errors
            return {
                "state": self.state,
                "version": self.version,
                "trigger": self.trigger,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "had_errors": had_errors,
                "ok": ok,
                "promoted": promoted,
                "bytes_total": total,
                "bytes_done": min(done, total) if total else done,
                "pct": round(done / total * 100.0, 1) if total else 0.0,
                "artifacts": arts,
                "report": self.report,
            }


_PROGRESS = SyncProgress()
_RUN_LOCK = threading.Lock()


def get_sync_progress() -> SyncProgress:
    return _PROGRESS


def run_tracked_sync(
    store: AssetStore,
    cache: ArtifactCache,
    names: list[str],
    manifest: Manifest,
    *,
    trigger: str,
    required: list[str] | None = None,
    promote: bool = True,
) -> SyncReport:
    """Run :func:`sync_artifacts`, mirroring byte-level progress into the
    shared :class:`SyncProgress`. Serialized: only one at a time.

    ``required`` is the profile that gates promotion (see
    :func:`sync_artifacts`); ``None`` means "whatever ``names`` downloaded".
    ``promote=False`` stages the download without ever moving ``current``
    (used for a manual, explicitly-selected subset).
    """
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("a sync is already running")
    try:
        _PROGRESS.begin(manifest, names, trigger)
        try:
            report = sync_artifacts(
                store, cache, names=names, required=required, manifest=manifest,
                progress=_PROGRESS.on_progress, promote=promote,
            )
        except Exception as exc:  # noqa: BLE001
            _PROGRESS.finish(None, error=f"{type(exc).__name__}: {exc}")
            raise
        _PROGRESS.finish(report)
        return report
    finally:
        _RUN_LOCK.release()
