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
    from src.services.retrieval_backend import active_backend

    backend = active_backend(settings)
    if not (cloud_enabled(settings) or backend == "jina_clip_v2"):
        return  # local BEiT3: keep the historical lazy-load-on-first-request

    try:
        from src.services.retrieval_backend import get_active_retriever

        retriever = get_active_retriever(settings)
        # Construction only loads the FAISS index + parquet. Explicitly warm
        # the query encoder too, so "retriever ready" is truthful and the
        # first real request never pays the (pinned) model download / load.
        warm = getattr(retriever, "warm_model", None)
        if callable(warm):
            warm()
            logger.info("startup warm: %s retriever + model ready", backend)
        else:
            logger.info("startup warm: %s retriever ready", backend)
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
    from src.services.retrieval_backend import active_backend

    if not cloud_enabled(settings):
        return
    store = build_asset_store(settings)
    if store is None:
        return
    manifest = get_manifest(store)
    if manifest is None:
        logger.info("startup warm: no manifest yet, nothing to sync")
        return

    backend = active_backend(settings)
    profile = list(
        BACKEND_ARTIFACT_NAMES.get(backend, BACKEND_ARTIFACT_NAMES["jina_clip_v2"])
    )
    manifest_names = {a.name for a in manifest.artifacts}
    missing = [n for n in profile if n not in manifest_names]
    if missing:
        # A manifest that does not declare the *whole* active-backend profile
        # is a broken publish. Do NOT sync a partial profile -- that is exactly
        # what would leave a fresh index paired with a stale/absent parquet.
        logger.warning(
            "startup warm: manifest %s is missing %s profile artifact(s) %s; "
            "skipping sync (republish the manifest with the full profile)",
            manifest.version,
            backend,
            missing,
        )
        return
    wanted = [a for a in manifest.artifacts if a.name in set(profile)]
    cache = get_artifact_cache(settings)
    if cache.get_current() == manifest.version and cache.is_version_verified(manifest.version, wanted):
        logger.info("startup warm: %s artifacts already current (%s)", backend, manifest.version)
        return

    logger.info(
        "startup warm: syncing the full %d-artifact %s profile for manifest %s",
        len(profile),
        backend,
        manifest.version,
    )
    try:
        # required == the COMPLETE profile -> promotion is all-or-nothing.
        run_tracked_sync(store, cache, list(profile), manifest, trigger="startup", required=list(profile))
    except RuntimeError as exc:
        logger.info("startup warm: %s", exc)  # a manual sync is already running
