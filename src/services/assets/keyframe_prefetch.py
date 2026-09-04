"""Bounded background prefetch of cloud keyframes.

Only meaningful when cloud assets are enabled. The search endpoints hand a
freshly produced result set to :func:`prefetch` (fire-and-forget); by the time
the browser asks for each thumbnail, :func:`fetch_blocking` -- which
``/keyframes/`` calls -- usually just returns the already-downloaded LRU path.
When a download for that exact key is still in flight, ``fetch_blocking`` waits
on *that* future instead of starting a second identical fetch.

Everything here is a no-op unless ``cloud_enabled(settings)`` is true; the
module-level pool is created lazily and torn down by :func:`shutdown` from the
app lifespan handler.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from src.config.settings import get_settings
from src.services import assets as _assets

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None
_INFLIGHT: dict[str, Future] = {}
_FETCH_TIMEOUT_SECONDS = 60.0


def _rel_key(frame_path: str) -> str:
    return str(frame_path or "").replace("\\", "/").strip("/")


def _get_executor(settings) -> ThreadPoolExecutor | None:
    global _EXECUTOR
    if not _assets.cloud_enabled(settings):
        return None
    with _LOCK:
        if _EXECUTOR is None:
            workers = max(
                1, int(getattr(settings, "cloud_assets_keyframe_prefetch_workers", 16) or 16)
            )
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="kf-prefetch"
            )
        return _EXECUTOR


def _already_cached(key: str, settings) -> bool:
    try:
        return _assets.get_keyframe_cache(settings).get(key) is not None
    except Exception:  # noqa: BLE001 - a cache probe must never raise here
        return False


def _download(frame_path: str, key: str, settings) -> Path | None:
    try:
        return _assets.resolve_keyframe_file(frame_path, settings=settings)
    except Exception as exc:  # noqa: BLE001
        logger.debug("keyframe prefetch failed for %s: %s", frame_path, exc)
        return None
    finally:
        with _LOCK:
            _INFLIGHT.pop(key, None)


def prefetch(frame_paths: Iterable[str], *, settings=None) -> None:
    """Fire-and-forget: warm the LRU cache for these ``frame_path`` values.

    Deduplicates against downloads already in flight and skips anything the
    cache already holds. No-op when cloud assets are off.
    """
    settings = settings or get_settings()
    executor = _get_executor(settings)
    if executor is None:
        return
    for frame_path in frame_paths:
        key = _rel_key(frame_path)
        if not key:
            continue
        with _LOCK:
            if key in _INFLIGHT:
                continue
        if _already_cached(key, settings):
            continue
        with _LOCK:
            if key in _INFLIGHT:
                continue
            _INFLIGHT[key] = executor.submit(_download, frame_path, key, settings)


def fetch_blocking(frame_path: str, *, settings=None) -> Path | None:
    """Return a local keyframe path, reusing an in-flight prefetch if there is
    one for this exact key rather than downloading a second copy.

    ``None`` when cloud assets are off or the fetch fails (the caller then
    serves its visible "missing" placeholder).
    """
    settings = settings or get_settings()
    if not _assets.cloud_enabled(settings):
        return None
    key = _rel_key(frame_path)
    with _LOCK:
        future = _INFLIGHT.get(key)
    if future is not None:
        try:
            return future.result(timeout=_FETCH_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001
            return None
    return _assets.resolve_keyframe_file(frame_path, settings=settings)


def shutdown() -> None:
    """Tear down the pool (app lifespan). Safe to call when never started."""
    global _EXECUTOR
    with _LOCK:
        executor, _EXECUTOR = _EXECUTOR, None
        _INFLIGHT.clear()
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


# Test seam: drop the pool + in-flight map so each test starts clean.
_reset_for_tests = shutdown
