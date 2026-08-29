"""Exact captured-frame preview extraction.

Given a trusted YouTube ``watch_url`` (taken from media-info metadata, never
from the client) and a target playback time, this module extracts a single
still with the FFmpeg binary and caches it as a WebP at
``<cache>/<video_id>/<frame_idx>.webp``.

Design constraints:

* Only the resulting WebP is persisted. The source stream is read directly
  from the URL yt-dlp resolves; any temporary media lands in a scratch
  directory that is deleted before this returns.
* Repeated captures of the same ``(video_id, frame_idx)`` reuse the cached
  still and perform no external fetch.
* Least-recently-used stills are evicted until the cache is at or below
  ``cache_max_bytes`` (500 MB by default).
* Every failure mode -- missing yt-dlp / FFmpeg, no network, a dead source
  video, a decode error, a timeout -- raises :class:`FramePreviewError` with a
  short human-readable reason. The caller turns that into ``preview_error`` and
  still returns the valid frame index; app startup never depends on the tools.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class FramePreviewError(Exception):
    """Preview extraction could not be completed (tools / network / source)."""


def _safe_video_id(video_id: str) -> str:
    key = str(video_id or "").strip()
    if not _VIDEO_ID_RE.match(key):
        raise FramePreviewError(f"Unsupported video id for preview: {video_id!r}")
    return key


def _touch(path: Path) -> None:
    """Mark ``path`` as most-recently-used for the LRU sweep."""
    try:
        os.utime(path, None)
    except OSError:  # pragma: no cover - best effort
        pass


@dataclass(frozen=True)
class _CacheEntry:
    path: Path
    size: int
    mtime: float


class VideoFramePreviewService:
    """Resolve and cache exact captured-frame stills from stored YouTube URLs."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        ffmpeg_bin: str = "ffmpeg",
        ytdlp_bin: str = "yt-dlp",
        timeout_seconds: float = 90.0,
        cache_max_bytes: int = 500 * 1024 * 1024,
    ):
        self._cache_dir = Path(cache_dir)
        self._ffmpeg_bin = str(ffmpeg_bin or "ffmpeg")
        self._ytdlp_bin = str(ytdlp_bin or "yt-dlp")
        self._timeout = max(1.0, float(timeout_seconds))
        self._max_bytes = max(0, int(cache_max_bytes))
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------------

    def cache_key(self, video_id: str, frame_idx: int) -> str:
        """Return the ``<video_id>/<frame_idx>.webp`` key for a still."""
        return f"{_safe_video_id(video_id)}/{int(frame_idx)}.webp"

    def cached_path(self, video_id: str, frame_idx: int) -> Path:
        return self._cache_dir / _safe_video_id(video_id) / f"{int(frame_idx)}.webp"

    def get_existing(self, video_id: str, frame_idx: int) -> Path | None:
        """Return the cached still path (bumping its LRU timestamp) or ``None``."""
        try:
            path = self.cached_path(video_id, frame_idx)
        except FramePreviewError:
            return None
        if path.is_file() and path.stat().st_size > 0:
            _touch(path)
            return path
        return None

    # -- public API -------------------------------------------------------

    def get_or_create(
        self,
        *,
        video_id: str,
        frame_idx: int,
        watch_url: str,
        target_seconds: float,
    ) -> str:
        """Return the cache key for the still at ``target_seconds``.

        Reuses a cached image when present; otherwise resolves the source
        stream with yt-dlp and decodes one frame with FFmpeg.

        Raises:
            FramePreviewError: extraction tools, network access, or the source
                video are unavailable, or the frame could not be decoded.
        """
        vid = _safe_video_id(video_id)
        frame = int(frame_idx)

        with self._lock:
            existing = self.get_existing(vid, frame)
            if existing is not None:
                return self.cache_key(vid, frame)

            url = str(watch_url or "").strip()
            if not url:
                raise FramePreviewError("No source video URL is available for this video.")

            dest = self.cached_path(vid, frame)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._extract(url, max(0.0, float(target_seconds)), dest)
            self._enforce_limit()
            return self.cache_key(vid, frame)

    # -- extraction -----------------------------------------------------

    def _resolve_ffmpeg(self) -> str:
        found = shutil.which(self._ffmpeg_bin)
        if found:
            return found
        if os.path.isfile(self._ffmpeg_bin):
            return self._ffmpeg_bin
        raise FramePreviewError("FFmpeg binary is not available on this server.")

    def _resolve_ytdlp(self) -> list[str]:
        found = shutil.which(self._ytdlp_bin)
        if found:
            return [found]
        if os.path.isfile(self._ytdlp_bin):
            return [self._ytdlp_bin]
        try:  # fall back to the pip-installed module
            import yt_dlp  # noqa: F401
        except Exception:  # noqa: BLE001
            raise FramePreviewError("yt-dlp is not installed on this server.") from None
        return [sys.executable, "-m", "yt_dlp"]

    def _run(self, cmd: list[str], *, what: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise FramePreviewError(f"{what} timed out.") from exc
        except OSError as exc:
            raise FramePreviewError(f"Could not run {what}: {exc}") from exc

    def _resolve_media_url(self, ytdlp_cmd: list[str], watch_url: str) -> str:
        cmd = [
            *ytdlp_cmd,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "-f",
            "bv*[height<=720]/best[height<=720]/best",
            "-g",
            watch_url,
        ]
        proc = self._run(cmd, what="yt-dlp")
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip().splitlines()
            reason = detail[-1] if detail else "unknown error"
            raise FramePreviewError(f"yt-dlp could not resolve the video: {reason}"[:200])
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if not lines or not lines[0].lower().startswith("http"):
            raise FramePreviewError("yt-dlp returned no playable stream URL.")
        return lines[0]

    def _extract(self, watch_url: str, target_seconds: float, dest: Path) -> None:
        ffmpeg = self._resolve_ffmpeg()
        ytdlp_cmd = self._resolve_ytdlp()

        with tempfile.TemporaryDirectory(prefix="video-capture-") as scratch:
            tmp_out = Path(scratch) / "frame.webp"
            media_url = self._resolve_media_url(ytdlp_cmd, watch_url)
            cmd = [
                ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{target_seconds:.3f}",
                "-i",
                media_url,
                "-frames:v",
                "1",
                "-update",
                "1",
                "-f",
                "webp",
                "-q:v",
                "80",
                str(tmp_out),
            ]
            proc = self._run(cmd, what="FFmpeg")
            if proc.returncode != 0 or not tmp_out.is_file() or tmp_out.stat().st_size == 0:
                detail = (proc.stderr or "").strip().splitlines()
                reason = detail[-1] if detail else "no output produced"
                raise FramePreviewError(f"FFmpeg could not decode the frame: {reason}"[:200])
            # Atomic-ish publish into the cache; scratch media is dropped on exit.
            shutil.move(str(tmp_out), str(dest))

    # -- cache maintenance --------------------------------------------

    def _enforce_limit(self) -> None:
        if self._max_bytes <= 0 or not self._cache_dir.is_dir():
            return
        entries: list[_CacheEntry] = []
        for path in self._cache_dir.rglob("*.webp"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append(_CacheEntry(path, stat.st_size, stat.st_mtime))

        total = sum(entry.size for entry in entries)
        if total <= self._max_bytes:
            return
        for entry in sorted(entries, key=lambda item: item.mtime):
            if total <= self._max_bytes:
                break
            try:
                entry.path.unlink()
            except OSError:  # pragma: no cover - concurrent sweep
                continue
            total -= entry.size


_service_lock = threading.Lock()
_service: VideoFramePreviewService | None = None


def get_video_frame_preview_service() -> VideoFramePreviewService:
    """Return the process-wide preview service built from application settings."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                from src.config.settings import get_settings

                settings = get_settings()
                _service = VideoFramePreviewService(
                    cache_dir=settings.get_video_capture_cache_path(),
                    ffmpeg_bin=settings.video_capture_ffmpeg_bin,
                    timeout_seconds=settings.video_capture_extract_timeout_seconds,
                    cache_max_bytes=settings.video_capture_cache_max_bytes,
                )
                logger.info(
                    "VideoFramePreviewService ready (cache=%s, ffmpeg=%s, max_bytes=%d)",
                    settings.get_video_capture_cache_path(),
                    settings.video_capture_ffmpeg_bin,
                    settings.video_capture_cache_max_bytes,
                )
    return _service


def reset_video_frame_preview_service() -> None:
    """Drop the cached singleton (used by tests after changing settings)."""
    global _service
    with _service_lock:
        _service = None
