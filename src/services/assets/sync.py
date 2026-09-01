"""Download + checksum-verify manifest artifacts into the local cache.

Each artifact is streamed to ``tmp/``, hashed on the way, and only
``os.replace``-d into ``artifacts/<version>/`` when both size and SHA-256
match. The version is marked *current* only once every declared artifact for
that version is present -- so a half-finished sync never shadows a good one.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from src.services.assets.base import AssetStore, AssetStoreError, Manifest
from src.services.assets.local_cache import ArtifactCache

logger = logging.getLogger(__name__)

ProgressCB = Callable[[str, str, int, int], None]  # name, phase, done_bytes, total_bytes

# status values
S_CACHED = "cached"
S_SYNCED = "synced"
S_SIZE_MISMATCH = "size_mismatch"
S_CHECKSUM_MISMATCH = "checksum_mismatch"
S_DOWNLOAD_ERROR = "download_error"


@dataclass
class ArtifactResult:
    name: str
    status: str
    bytes: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "bytes": self.bytes, "detail": self.detail}


@dataclass
class SyncReport:
    version: str
    results: list[ArtifactResult] = field(default_factory=list)
    promoted: bool = False
    current_version: str | None = None
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return all(r.status in (S_CACHED, S_SYNCED) for r in self.results)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "ok": self.ok,
            "promoted": self.promoted,
            "current_version": self.current_version,
            "duration_ms": self.duration_ms,
            "artifacts": [r.to_dict() for r in self.results],
        }


def _download_and_promote(
    store: AssetStore, cache: ArtifactCache, manifest: Manifest, art, progress: ProgressCB | None
) -> ArtifactResult:
    existing = cache.slot(manifest.version, art.name, expected_sha=art.sha256, expected_size=art.size)
    if existing.present and existing.verified:
        return ArtifactResult(art.name, S_CACHED, bytes=existing.size)

    staged = cache.stage_path(art.name.replace("/", "_"))
    h = hashlib.sha256()
    written = 0
    try:
        staged.parent.mkdir(parents=True, exist_ok=True)
        with staged.open("wb") as fh:
            for chunk in store.open_object(art.container, art.key):
                fh.write(chunk)
                h.update(chunk)
                written += len(chunk)
                if progress:
                    progress(art.name, "download", written, art.size)
    except AssetStoreError as exc:
        staged.unlink(missing_ok=True)
        return ArtifactResult(art.name, S_DOWNLOAD_ERROR, detail=str(exc))
    except OSError as exc:
        staged.unlink(missing_ok=True)
        return ArtifactResult(art.name, S_DOWNLOAD_ERROR, detail=f"write error: {type(exc).__name__}")

    if art.size and written != art.size:
        staged.unlink(missing_ok=True)
        return ArtifactResult(
            art.name, S_SIZE_MISMATCH, bytes=written,
            detail=f"expected {art.size} bytes, got {written}",
        )
    if h.hexdigest() != art.sha256:
        staged.unlink(missing_ok=True)
        return ArtifactResult(
            art.name, S_CHECKSUM_MISMATCH, bytes=written,
            detail=f"expected {art.sha256}, got {h.hexdigest()}",
        )

    try:
        cache.promote(staged, manifest.version, art.name, art.sha256)
    except ValueError as exc:
        return ArtifactResult(art.name, S_CHECKSUM_MISMATCH, bytes=written, detail=str(exc))
    return ArtifactResult(art.name, S_SYNCED, bytes=written)


def sync_artifacts(
    store: AssetStore,
    cache: ArtifactCache,
    *,
    names: list[str] | None = None,
    manifest: Manifest | None = None,
    progress: ProgressCB | None = None,
) -> SyncReport:
    started = time.perf_counter()
    manifest = manifest or store.fetch_manifest()
    wanted = manifest.artifacts if not names else [a for a in manifest.artifacts if a.name in set(names)]

    report = SyncReport(version=manifest.version)
    for art in wanted:
        result = _download_and_promote(store, cache, manifest, art, progress)
        report.results.append(result)
        logger.info("sync %s/%s -> %s", manifest.version, art.name, result.status)

    # Promotion is scoped to what was actually requested (`wanted`), not every
    # artifact the manifest happens to declare -- a member syncing only the
    # active backend's artifacts (see BACKEND_ARTIFACT_NAMES) must still reach
    # "current" without also downloading the other backend's files. Each slot
    # is re-verified by size + SHA-256 here (not just "the file exists"), so a
    # stale or half-written leftover from an earlier run can never count.
    if wanted and cache.is_version_verified(manifest.version, wanted):
        cache.set_current(manifest.version)
        report.promoted = True
    report.current_version = cache.get_current()
    report.duration_ms = int((time.perf_counter() - started) * 1000)
    return report
