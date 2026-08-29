"""Lightweight video playback / frame-capture metadata service.

This module is intentionally independent of the BEiT-3 retrieval stack. It
reads two on-disk assets directly:

* ``media-info`` — per-video JSON carrying the YouTube ``watch_url`` and the
  video ``length`` (seconds).
* ``map-keyframes`` — per-video CSV carrying the authoritative per-video FPS
  and the keyframe-ordinal -> frame-index mapping produced during dataset
  extraction.

Both assets may be supplied either as a directory of per-video files or as a
ZIP archive, and the archive may or may not have a wrapping top-level folder
(``media-info/L21_V001.json`` vs ``L21_V001.json``). Nothing here loads a
search model or touches FAISS.

Time <-> frame conversion
-------------------------
``source_time  = playback_time - playback_offset``
``frame_idx    = floor(source_time * fps + 1e-6)``  (0-based)

The playback offset defaults to ``0`` for every video. An override map may be
supplied for videos whose YouTube timeline has been verified to differ from
the dataset timeline (see ``Settings.playback_offsets_json``).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Small positive nudge so exact products such as ``11.7333 * 30 == 351.999``
# floor to the dataset's own frame index instead of one below it.
_FLOOR_EPSILON = 1e-6


class VideoPlaybackError(Exception):
    """Base class for playback-metadata failures."""


class VideoNotFoundError(VideoPlaybackError):
    """The requested video or one of its metadata assets does not exist (404)."""


class VideoMetadataError(VideoPlaybackError):
    """A metadata asset exists but is missing/!invalid required fields (400)."""


class VideoRequestError(VideoPlaybackError):
    """The caller supplied an out-of-range timestamp or frame index (400)."""


@dataclass(frozen=True)
class VideoPlaybackMetadata:
    """Resolved playback metadata for a single video."""

    video_id: str
    watch_url: str
    fps: float
    duration_seconds: float
    playback_offset_seconds: float


@dataclass(frozen=True)
class CaptureResult:
    """Outcome of converting a playback timestamp into a dataset frame index."""

    video_id: str
    playback_time_seconds: float
    source_time_seconds: float
    fps: float
    frame_idx: int


class _AssetSource:
    """Reads per-video files from either a directory or a ZIP archive.

    Entry lookup is tolerant of a wrapping top-level folder inside the archive:
    a request for ``L21_V001.json`` also matches ``media-info/L21_V001.json``.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._is_zip = self._path.is_file() and zipfile.is_zipfile(self._path)
        self._zip_index: dict[str, str] | None = None
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def _build_zip_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        with zipfile.ZipFile(self._path) as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                base = name.rsplit("/", 1)[-1]
                # First writer wins; per-video basenames are unique in practice.
                index.setdefault(base, name)
        return index

    def read_bytes(self, filename: str) -> bytes | None:
        """Return the bytes of ``filename`` (matched by basename) or ``None``."""
        if not self.exists():
            return None

        if self._is_zip:
            with self._lock:
                if self._zip_index is None:
                    self._zip_index = self._build_zip_index()
            entry = self._zip_index.get(filename)
            if entry is None:
                return None
            with zipfile.ZipFile(self._path) as archive:
                return archive.read(entry)

        if self._path.is_dir():
            direct = self._path / filename
            if direct.is_file():
                return direct.read_bytes()
            for match in self._path.rglob(filename):
                if match.is_file():
                    return match.read_bytes()
        return None


def _coerce_positive_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


class VideoPlaybackService:
    """Resolve playback metadata and convert timestamps to dataset frame indices."""

    def __init__(
        self,
        media_info_path: Path,
        map_keyframes_path: Path,
        offsets: dict[str, float] | None = None,
    ):
        self._media_info = _AssetSource(Path(media_info_path))
        self._map_keyframes = _AssetSource(Path(map_keyframes_path))
        self._offsets = {str(k): float(v) for k, v in (offsets or {}).items()}
        self._cache: dict[str, VideoPlaybackMetadata] = {}
        self._lock = threading.Lock()

    # -- asset parsing -----------------------------------------------------

    def _load_media_info(self, video_id: str) -> dict:
        raw = self._media_info.read_bytes(f"{video_id}.json")
        if raw is None:
            raise VideoNotFoundError(
                f"No media-info entry for '{video_id}' in {self._media_info.path}"
            )
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise VideoMetadataError(
                f"media-info for '{video_id}' is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise VideoMetadataError(f"media-info for '{video_id}' is not an object")
        return data

    def _load_fps(self, video_id: str) -> float:
        raw = self._map_keyframes.read_bytes(f"{video_id}.csv")
        if raw is None:
            raise VideoNotFoundError(
                f"No map-keyframes entry for '{video_id}' in {self._map_keyframes.path}"
            )
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise VideoMetadataError(
                f"map-keyframes for '{video_id}' is not UTF-8 decodable: {exc}"
            ) from exc

        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None or "fps" not in reader.fieldnames:
            raise VideoMetadataError(
                f"map-keyframes for '{video_id}' has no 'fps' column"
            )
        for row in reader:
            fps = _coerce_positive_float(row.get("fps"))
            if fps is not None:
                # Keep the real floating-point value: the dataset mixes 25,
                # 30, 29.97 and 26.44 FPS and rounding corrupts the mapping.
                return fps
        raise VideoMetadataError(
            f"map-keyframes for '{video_id}' has no usable 'fps' value"
        )

    # -- public API ------------------------------------------------------

    def get_metadata(self, video_id: str) -> VideoPlaybackMetadata:
        """Return resolved playback metadata for ``video_id`` (cached).

        Raises:
            VideoNotFoundError: the video or one of its assets is absent.
            VideoMetadataError: an asset exists but lacks the URL / FPS /
                length needed to drive the player.
        """
        key = str(video_id or "").strip()
        if not key:
            raise VideoNotFoundError("A non-empty video_id is required")

        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        media_info = self._load_media_info(key)
        watch_url = str(media_info.get("watch_url") or "").strip()
        if not watch_url:
            raise VideoMetadataError(f"media-info for '{key}' has no watch_url")

        duration = _coerce_positive_float(media_info.get("length"))
        if duration is None:
            raise VideoMetadataError(f"media-info for '{key}' has no valid length")

        fps = self._load_fps(key)
        offset = float(self._offsets.get(key, 0.0))

        metadata = VideoPlaybackMetadata(
            video_id=key,
            watch_url=watch_url,
            fps=fps,
            duration_seconds=duration,
            playback_offset_seconds=offset,
        )
        with self._lock:
            self._cache[key] = metadata
        return metadata

    def playback_start_seconds(self, video_id: str, frame_idx: int) -> float:
        """Return the player start time (seconds) that shows ``frame_idx``.

        This is the inverse of :meth:`capture`:
        ``playback_time = frame_idx / fps + playback_offset`` clamped at 0.
        """
        meta = self.get_metadata(video_id)
        try:
            frame = int(frame_idx)
        except (TypeError, ValueError) as exc:
            raise VideoRequestError(f"frame_idx must be an integer, got {frame_idx!r}") from exc
        if frame < 0:
            raise VideoRequestError("frame_idx must not be negative")

        source_time = frame / meta.fps
        playback_time = source_time + meta.playback_offset_seconds
        if playback_time > meta.duration_seconds + 1.0:
            raise VideoRequestError(
                f"frame_idx {frame} is beyond the video duration ({meta.duration_seconds:g}s)"
            )
        return max(0.0, playback_time)

    def capture(self, video_id: str, playback_time_seconds: float) -> CaptureResult:
        """Convert a player timestamp into a 0-based dataset frame index.

        Raises:
            VideoRequestError: the timestamp is negative or past the video end.
        """
        meta = self.get_metadata(video_id)
        try:
            playback_time = float(playback_time_seconds)
        except (TypeError, ValueError) as exc:
            raise VideoRequestError(
                f"playback_time_seconds must be a number, got {playback_time_seconds!r}"
            ) from exc
        if not math.isfinite(playback_time):
            raise VideoRequestError("playback_time_seconds must be finite")
        if playback_time < 0:
            raise VideoRequestError("playback_time_seconds must not be negative")
        if playback_time > meta.duration_seconds + 1.0:
            raise VideoRequestError(
                f"playback_time_seconds {playback_time:g} is beyond the video "
                f"duration ({meta.duration_seconds:g}s)"
            )

        source_time = playback_time - meta.playback_offset_seconds
        if source_time < 0:
            raise VideoRequestError(
                "playback_time_seconds is before the start of the dataset timeline"
            )
        frame_idx = math.floor(source_time * meta.fps + _FLOOR_EPSILON)
        return CaptureResult(
            video_id=meta.video_id,
            playback_time_seconds=playback_time,
            source_time_seconds=source_time,
            fps=meta.fps,
            frame_idx=frame_idx,
        )


_service_lock = threading.Lock()
_service: VideoPlaybackService | None = None


def get_video_playback_service() -> VideoPlaybackService:
    """Return the process-wide service built from application settings."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                from src.config.settings import get_settings

                settings = get_settings()
                _service = VideoPlaybackService(
                    media_info_path=settings.get_media_info_path(),
                    map_keyframes_path=settings.get_map_keyframes_path(),
                    offsets=settings.get_playback_offsets(),
                )
                logger.info(
                    "VideoPlaybackService ready (media-info=%s, map-keyframes=%s, offsets=%d)",
                    settings.get_media_info_path(),
                    settings.get_map_keyframes_path(),
                    len(settings.get_playback_offsets()),
                )
    return _service


def reset_video_playback_service() -> None:
    """Drop the cached singleton (used by tests after changing settings)."""
    global _service
    with _service_lock:
        _service = None


def iter_known_video_ids(source_paths: Iterable[Path]) -> set[str]:  # pragma: no cover - helper
    """Best-effort enumeration of video ids present across the given assets."""
    ids: set[str] = set()
    for path in source_paths:
        src = _AssetSource(Path(path))
        if not src.exists():
            continue
        if src._is_zip:  # noqa: SLF001 - internal helper, intentional
            with zipfile.ZipFile(src.path) as archive:
                for name in archive.namelist():
                    base = name.rsplit("/", 1)[-1]
                    if base.endswith((".json", ".csv")):
                        ids.add(base.rsplit(".", 1)[0])
        elif src.path.is_dir():
            for match in list(src.path.rglob("*.json")) + list(src.path.rglob("*.csv")):
                ids.add(match.stem)
    return ids
