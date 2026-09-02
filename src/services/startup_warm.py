"""Background warm-up of the active retrieval backend at app startup.

When cloud assets are on and ``CLOUD_ASSETS_AUTOSYNC`` is set (default), the
moment the API comes up this:

1. syncs the active backend's artifacts (see ``BACKEND_ARTIFACT_NAMES``) if the
   local cache isn't already the current manifest version -- progress is
   mirrored into the shared :class:`SyncProgress` so the Settings UI shows a
   bar, and
2. constructs the active retriever (loads its FAISS index + model),

so the first real search request isn't the one that pays a multi-GB download
or a cold model load. Everything here is best-effort: any failure is logged
and swallowed, and the app keeps serving (a later request just loads lazily,
as before).
"""

from __future__ import annotations

import logging
import os
import threading

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_STARTED = threading.Event()


def warm_active_backend_in_background(settings: Settings | None = None) -> None:
    """Spawn the warm-up daemon thread. No-op if disabled, already started, or
    running under pytest."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    settings = settings or get_settings()
    if not settings.cloud_assets_autosync:
        return
    if _STARTED.is_set():
        return
    _STARTED.set()
    threading.Thread(target=_warm, args=(settings,), name="backend-warm", daemon=True).start()


def _warm(settings: Settings) -> None:
    try:
        _sync_active_backend_artifacts(settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("startup artifact sync skipped: %s", exc)

    from src.services.assets.factory import cloud_enabled

    if not (cloud_enabled(settings) or settings.retrieval_backend == "jina_clip_v2"):
        return  # local BEiT3: keep the historical lazy-load-on-first-request

    try:
        from src.services.retrieval_backend import get_active_retriever

        get_active_retriever()
        logger.info("startup warm: %s retriever ready", settings.retrieval_backend)
    except Exception as exc:  # noqa: BLE001
        logger.warning("startup retriever warm skipped: %s", exc)


def _sync_active_backend_artifacts(settings: Settings) -> None:
    from src.services.assets.base import BACKEND_ARTIFACT_NAMES
    from src.services.assets.factory import (
        build_asset_store,
        cloud_enabled,
        get_artifact_cache,
        get_manifest,
    )
    from src.services.assets.sync_state import run_tracked_sync

    if not cloud_enabled(settings):
        return
    store = build_asset_store(settings)
    if store is None:
        return
    manifest = get_manifest(store)
    if manifest is None:
        logger.info("startup warm: no manifest yet, nothing to sync")
        return

    names = list(
        BACKEND_ARTIFACT_NAMES.get(settings.retrieval_backend, BACKEND_ARTIFACT_NAMES["beit3"])
    )
    wanted = [a for a in manifest.artifacts if a.name in set(names)]
    cache = get_artifact_cache(settings)
    if cache.get_current() == manifest.version and cache.is_version_verified(manifest.version, wanted):
        logger.info("startup warm: %s artifacts already current (%s)", settings.retrieval_backend, manifest.version)
        return

    logger.info(
        "startup warm: syncing %d %s artifact(s) for manifest %s",
        len(wanted),
        settings.retrieval_backend,
        manifest.version,
    )
    try:
        run_tracked_sync(store, cache, names, manifest, trigger="startup")
    except RuntimeError as exc:
        logger.info("startup warm: %s", exc)  # a manual sync is already running
