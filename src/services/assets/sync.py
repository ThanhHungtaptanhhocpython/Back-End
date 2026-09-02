"""Download + checksum-verify manifest artifacts into the local cache.

Each artifact is streamed to ``tmp/``, hashed on the way, and only
``os.replace``-d into ``artifacts/<version>/`` when both size and SHA-256
match.

Promotion is **profile-atomic**. When the caller passes a ``required`` profile
(e.g. the active backend's full artifact set, ``BACKEND_ARTIFACT_NAMES``):

* **every** name in ``required`` must be declared by the manifest. If even one
  is missing, this is a broken manifest for that backend -- nothing is
  downloaded and nothing is promoted (``report.errors``). A fresh index must
  never be paired with a stale/absent parquet or meta.
* the version becomes *current* only once every ``required`` artifact is
  present with a verified size + SHA-256 at that version.
* downloading a strict subset (``names`` ⊂ ``required``) still stages those
  files, but never advances ``current`` until the whole profile verifies.

An unknown / empty ``names`` request is a hard validation error
(``report.errors``), never a silently-successful empty sync. ``promote=False``
stages downloads without ever touching the ``current`` pointer.
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
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # A validation error, or an empty result set, is never "ok" -- an
        # unknown / empty artifact request must not read as a successful sync.
        return (
            not self.errors
            and bool(self.results)
            and all(r.status in (S_CACHED, S_SYNCED) for r in self.results)
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "ok": self.ok,
            "promoted": self.promoted,
            "current_version": self.current_version,
            "duration_ms": self.duration_ms,
            "errors": list(self.errors),
            "artifacts": [r.to_dict() for r in self.results],
        }


def _download_and_promote(
    store: AssetStore, cache: ArtifactCache, manifest: Manifest, art, progress: ProgressCB | None
) -> ArtifactResult:
    def _emit(phase: str, done: int) -> None:
        if progress:
            progress(art.name, phase, done, art.size)

    existing = cache.slot(manifest.version, art.name, expected_sha=art.sha256, expected_size=art.size)
    if existing.present and existing.verified:
        _emit("cached", existing.size)
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
                _emit("download", written)
    except AssetStoreError as exc:
        staged.unlink(missing_ok=True)
        _emit("error", written)
        return ArtifactResult(art.name, S_DOWNLOAD_ERROR, detail=str(exc))
    except OSError as exc:
        staged.unlink(missing_ok=True)
        _emit("error", written)
        return ArtifactResult(art.name, S_DOWNLOAD_ERROR, detail=f"write error: {type(exc).__name__}")

    if art.size and written != art.size:
        staged.unlink(missing_ok=True)
        _emit("error", written)
        return ArtifactResult(
            art.name, S_SIZE_MISMATCH, bytes=written,
            detail=f"expected {art.size} bytes, got {written}",
        )
    _emit("verify", written)
    if h.hexdigest() != art.sha256:
        staged.unlink(missing_ok=True)
        _emit("error", written)
        return ArtifactResult(
            art.name, S_CHECKSUM_MISMATCH, bytes=written,
            detail=f"expected {art.sha256}, got {h.hexdigest()}",
        )

    try:
        cache.promote(staged, manifest.version, art.name, art.sha256)
    except ValueError as exc:
        _emit("error", written)
        return ArtifactResult(art.name, S_CHECKSUM_MISMATCH, bytes=written, detail=str(exc))
    _emit("synced", written)
    return ArtifactResult(art.name, S_SYNCED, bytes=written)


def sync_artifacts(
    store: AssetStore,
    cache: ArtifactCache,
    *,
    names: list[str] | None = None,
    required: list[str] | None = None,
    manifest: Manifest | None = None,
    progress: ProgressCB | None = None,
    promote: bool = True,
) -> SyncReport:
    """Download ``names`` (default: every manifest artifact) and, when
    ``promote`` is set, promote ``manifest.version`` to *current* iff the
    whole ``required`` profile (default: whatever was downloaded) verifies.

    ``names`` that are empty, or name an artifact the manifest does not
    declare, produce a ``report.errors`` entry and download nothing.
    """
    started = time.perf_counter()
    manifest = manifest or store.fetch_manifest()
    manifest_names = {a.name for a in manifest.artifacts}
    report = SyncReport(version=manifest.version)

    if names is not None:
        if not names:
            report.errors.append("no artifact names were requested")
            report.duration_ms = int((time.perf_counter() - started) * 1000)
            return report
        unknown = sorted(n for n in names if n not in manifest_names)
        if unknown:
            report.errors.append(
                f"requested artifact name(s) not in manifest {manifest.version}: {unknown}"
            )
            report.duration_ms = int((time.perf_counter() - started) * 1000)
            return report

    # The set that gates promotion. A `required` profile must be declared by
    # the manifest *in full* -- a manifest missing any profile artifact is a
    # broken publish for that backend, so we neither download nor promote a
    # partial profile (that is exactly what would pair a fresh index with a
    # stale/absent parquet). Downloading a strict subset of a complete profile
    # still only stages it: `current` moves once the whole profile verifies.
    if required is not None:
        missing_from_manifest = sorted(set(required) - manifest_names)
        if missing_from_manifest:
            report.errors.append(
                f"manifest {manifest.version} is missing required profile artifact(s) "
                f"{missing_from_manifest}; refusing to sync or promote a partial "
                f"backend profile"
            )
            report.duration_ms = int((time.perf_counter() - started) * 1000)
            return report
        promotion_arts: list = [manifest.artifact(n) for n in required]
    else:
        promotion_arts = None  # set below to `wanted`

    requested = set(names) if names is not None else set(manifest_names)
    wanted = [a for a in manifest.artifacts if a.name in requested]
    if promotion_arts is None:
        promotion_arts = list(wanted)

    for art in wanted:
        result = _download_and_promote(store, cache, manifest, art, progress)
        report.results.append(result)
        logger.info("sync %s/%s -> %s", manifest.version, art.name, result.status)

    # Each slot is re-verified by size + SHA-256 here (not "the file exists"),
    # so a stale or half-written leftover from an earlier run never counts.
    if promote and promotion_arts and cache.is_version_verified(manifest.version, promotion_arts):
        cache.set_current(manifest.version)
        report.promoted = True
    report.current_version = cache.get_current()
    report.duration_ms = int((time.perf_counter() - started) * 1000)
    return report
