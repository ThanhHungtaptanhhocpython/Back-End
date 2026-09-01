"""Build the configured :class:`AssetStore` and the shared local caches."""

from __future__ import annotations

import threading
import time

from src.config.settings import Settings, get_settings
from src.services.assets.azure_blob import AzureBlobAssetStore
from src.services.assets.base import AssetStore, Manifest
from src.services.assets.local_cache import ArtifactCache, KeyframeCache
from src.services.assets.s3_compatible import S3AssetStore

_LOCK = threading.Lock()
_ARTIFACT_CACHE: ArtifactCache | None = None
_KEYFRAME_CACHE: KeyframeCache | None = None
_KEYFRAME_CACHE_MAX = -1
_MANIFEST_CACHE: dict = {"at": 0.0, "manifest": None}
_MANIFEST_TTL = 300.0


def cloud_enabled(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return bool(settings.cloud_assets_enabled) and settings.cloud_assets_provider in (
        "azure_blob",
        "s3_compatible",
    )


def build_asset_store(settings: Settings | None = None) -> AssetStore | None:
    settings = settings or get_settings()
    provider = settings.cloud_assets_provider
    if provider == "azure_blob":
        return AzureBlobAssetStore(settings)
    if provider == "s3_compatible":
        return S3AssetStore(settings)
    return None


def get_artifact_cache(settings: Settings | None = None) -> ArtifactCache:
    global _ARTIFACT_CACHE
    settings = settings or get_settings()
    root = settings.get_cloud_assets_cache_path()
    with _LOCK:
        if _ARTIFACT_CACHE is None or _ARTIFACT_CACHE.root != root:
            _ARTIFACT_CACHE = ArtifactCache(root)
        return _ARTIFACT_CACHE


def get_keyframe_cache(settings: Settings | None = None) -> KeyframeCache:
    global _KEYFRAME_CACHE, _KEYFRAME_CACHE_MAX
    settings = settings or get_settings()
    root = settings.get_cloud_assets_cache_path()
    max_bytes = int(settings.cloud_assets_keyframe_cache_max_bytes or 0)
    with _LOCK:
        if (
            _KEYFRAME_CACHE is None
            or _KEYFRAME_CACHE.root.parent != root
            or _KEYFRAME_CACHE_MAX != max_bytes
        ):
            _KEYFRAME_CACHE = KeyframeCache(root, max_bytes)
            _KEYFRAME_CACHE_MAX = max_bytes
        return _KEYFRAME_CACHE


def reset_caches() -> None:
    global _ARTIFACT_CACHE, _KEYFRAME_CACHE, _KEYFRAME_CACHE_MAX, _MANIFEST_CACHE
    with _LOCK:
        _ARTIFACT_CACHE = None
        _KEYFRAME_CACHE = None
        _KEYFRAME_CACHE_MAX = -1
        _MANIFEST_CACHE = {"at": 0.0, "manifest": None}


def get_manifest(store: AssetStore | None = None, *, force: bool = False) -> Manifest | None:
    global _MANIFEST_CACHE
    store = store or build_asset_store()
    if store is None:
        return None
    now = time.time()
    if not force and _MANIFEST_CACHE["manifest"] is not None and now - _MANIFEST_CACHE["at"] < _MANIFEST_TTL:
        return _MANIFEST_CACHE["manifest"]
    try:
        manifest = store.fetch_manifest()
    except Exception:  # noqa: BLE001
        return _MANIFEST_CACHE["manifest"]
    _MANIFEST_CACHE = {"at": now, "manifest": manifest}
    return manifest
