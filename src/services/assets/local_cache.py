"""Local cache for synced artifacts and on-demand keyframes.

Layout under ``<cache-root>`` (``CLOUD_ASSETS_CACHE_PATH`` or
``<app-data>/assets-cache``)::

    artifacts/<version>/<name>          promoted artifact file
    artifacts/<version>/<name>.sha256   sidecar with the verified digest
    artifacts/current.txt              the version considered "live"
    keyframes/<frame_path>             on-demand keyframe, LRU-evicted
    keyframes/.index.json              {rel_path: {size, atime}}
    tmp/                               download staging (atomic promote source)

An artifact version is only marked *current* once every artifact the manifest
declares is present and checksum-verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class ArtifactSlot:
    name: str
    path: Path
    present: bool
    size: int
    sha256: str | None
    verified: bool


class ArtifactCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.tmp_dir = self.root / "tmp"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # -- current pointer ------------------------------------------------------
    @property
    def _current_file(self) -> Path:
        return self.artifacts_dir / "current.txt"

    def get_current(self) -> str | None:
        try:
            value = self._current_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def set_current(self, version: str) -> None:
        tmp = self._current_file.with_suffix(".txt.tmp")
        tmp.write_text(version, encoding="utf-8")
        os.replace(tmp, self._current_file)

    # -- per-artifact -------------------------------------------------------
    def version_dir(self, version: str) -> Path:
        return self.artifacts_dir / version

    def artifact_path(self, version: str, name: str) -> Path:
        return self.version_dir(version) / name

    def _sidecar(self, version: str, name: str) -> Path:
        return self.version_dir(version) / f"{name}.sha256"

    def slot(self, version: str, name: str, *, expected_sha: str | None = None,
             expected_size: int | None = None) -> ArtifactSlot:
        path = self.artifact_path(version, name)
        if not path.is_file():
            return ArtifactSlot(name, path, present=False, size=0, sha256=None, verified=False)
        size = path.stat().st_size
        sidecar = None
        try:
            sidecar = self._sidecar(version, name).read_text(encoding="utf-8").strip().lower() or None
        except OSError:
            pass
        verified = bool(
            sidecar
            and (expected_sha is None or sidecar == expected_sha.lower())
            and (expected_size is None or size == expected_size)
        )
        return ArtifactSlot(name, path, present=True, size=size, sha256=sidecar, verified=verified)

    def stage_path(self, token: str) -> Path:
        return self.tmp_dir / f"stage-{token}-{int(time.time()*1000)}"

    def promote(self, staged: Path, version: str, name: str, sha256: str) -> None:
        """Verify ``staged`` against ``sha256`` then atomically move it into place."""
        digest = sha256_file(staged)
        if digest != sha256.lower():
            staged.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {name}: expected {sha256}, got {digest}")
        target = self.artifact_path(version, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, target)
        tmp_side = self._sidecar(version, name).with_suffix(".sha256.tmp")
        tmp_side.write_text(digest, encoding="utf-8")
        os.replace(tmp_side, self._sidecar(version, name))

    def is_version_complete(self, version: str, names: list[str]) -> bool:
        """Presence-only check: every named artifact file exists on disk.

        This does NOT verify size/checksum -- use :meth:`is_version_verified`
        before promoting a version to *current*.
        """
        with self._lock:
            return all(self.slot(version, name).present for name in names)

    def is_version_verified(self, version: str, artifacts) -> bool:
        """Full check: every artifact is present with a size+SHA-256 that
        matches the manifest before a version is allowed to become *current*.

        ``artifacts`` is an iterable of objects/records with ``.name``,
        ``.sha256`` and ``.size`` (e.g. :class:`ManifestArtifact`). Unlike
        :meth:`is_version_complete`, a stale or half-written file that merely
        *exists* at the right path never counts as complete here.
        """
        with self._lock:
            return all(
                self.slot(version, art.name, expected_sha=art.sha256, expected_size=art.size).verified
                for art in artifacts
            )

    # -- maintenance ------------------------------------------------------
    def clear(self) -> int:
        import shutil

        freed = self.usage_bytes()
        with self._lock:
            for child in self.artifacts_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            for child in self.tmp_dir.iterdir():
                child.unlink(missing_ok=True)
        return freed

    def usage_bytes(self) -> int:
        total = 0
        for base in (self.artifacts_dir, self.tmp_dir):
            for path in base.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        return total

    def stats(self) -> dict:
        current = self.get_current()
        versions = sorted(
            p.name for p in self.artifacts_dir.iterdir() if p.is_dir()
        )
        return {
            "root": str(self.root),
            "current_version": current,
            "versions": versions,
            "usage_bytes": self.usage_bytes(),
        }


class KeyframeCache:
    """LRU cache of on-demand keyframe downloads, keyed by ``frame_path``."""

    # get() only touches an in-memory atime; the index file is flushed at most
    # once per this many seconds (and once on shutdown via flush()).
    _FLUSH_DEBOUNCE_SECONDS = 5.0

    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root = Path(root) / "keyframes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes) if max_bytes else 0
        self._index_path = self.root / ".index.json"
        self._lock = threading.RLock()
        self._index: dict[str, dict] = self._load_index()
        # Running total maintained by get()/put()/_evict_if_needed()/clear() so
        # a put() never re-sums the whole (potentially 6-figure) index.
        self._total_bytes: int = sum(int(e.get("size", 0)) for e in self._index.values())
        self._dirty: bool = False
        self._last_flush: float = time.time()

    def _load_index(self) -> dict[str, dict]:
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._index, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self._index_path)
        self._dirty = False
        self._last_flush = time.time()

    def _flush_debounced(self) -> None:
        """Persist the index only if it's been dirty for longer than the
        debounce window. Called from the cache-hit path, which must stay cheap."""
        if self._dirty and (time.time() - self._last_flush) >= self._FLUSH_DEBOUNCE_SECONDS:
            self._save_index()

    def flush(self) -> None:
        """Force-persist a pending in-memory index (app shutdown)."""
        with self._lock:
            if self._dirty:
                self._save_index()

    def _abs(self, rel_path: str) -> Path:
        rel = rel_path.replace("\\", "/").lstrip("/")
        root = self.root.resolve()
        path = (root / rel).resolve()
        # keep the cache inside its root -- blocks "../../x" and absolute-path
        # escapes from either writing or reading outside the cache directory.
        if root != path and root not in path.parents:
            raise ValueError(f"unsafe keyframe path: {rel_path}")
        return path

    def get(self, rel_path: str) -> Path | None:
        with self._lock:
            try:
                path = self._abs(rel_path)
            except ValueError:
                return None
            if not path.is_file():
                stale = self._index.pop(rel_path, None)
                if stale is not None:
                    self._total_bytes -= int(stale.get("size", 0))
                    self._dirty = True
                return None
            entry = self._index.get(rel_path)
            if entry is None:
                entry = {"size": path.stat().st_size}
                self._index[rel_path] = entry
                self._total_bytes += int(entry["size"])
            entry["atime"] = time.time()
            self._dirty = True
            self._flush_debounced()
            return path

    def put(self, rel_path: str, data: bytes) -> Path:
        with self._lock:
            path = self._abs(rel_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            previous = self._index.get(rel_path)
            if previous is not None:
                self._total_bytes -= int(previous.get("size", 0))
            self._index[rel_path] = {"size": len(data), "atime": time.time()}
            self._total_bytes += len(data)
            self._evict_if_needed()
            self._save_index()
            return path

    def _evict_if_needed(self) -> None:
        if self.max_bytes <= 0:
            return
        if self._total_bytes <= self.max_bytes:
            return
        by_atime = sorted(self._index.items(), key=lambda kv: kv[1].get("atime", 0.0))
        for rel_path, entry in by_atime:
            if self._total_bytes <= self.max_bytes:
                break
            try:
                self._abs(rel_path).unlink(missing_ok=True)
            except OSError:
                pass
            self._total_bytes -= int(entry.get("size", 0))
            self._index.pop(rel_path, None)

    def clear(self) -> int:
        import shutil

        with self._lock:
            freed = sum(int(e.get("size", 0)) for e in self._index.values())
            for child in self.root.iterdir():
                if child.name == ".index.json":
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            self._index = {}
            self._total_bytes = 0
            self._save_index()
            return freed

    def stats(self) -> dict:
        with self._lock:
            total = sum(int(e.get("size", 0)) for e in self._index.values())
            return {
                "root": str(self.root),
                "entries": len(self._index),
                "usage_bytes": total,
                "max_bytes": self.max_bytes,
            }
