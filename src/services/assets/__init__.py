"""Cloud asset storage: read the dataset from Azure Blob or S3-compatible
storage described by a versioned ``hcmai-assets.json`` manifest.

Public helpers:

* :func:`build_asset_store` / :func:`cloud_enabled`
* :func:`sync_artifacts` -- download + checksum-verify + atomic promote
* :func:`resolve_keyframe_file` -- on-demand, LRU-cached keyframe by ``frame_path``
* :func:`resolve_artifact_path` -- local path of a synced artifact, if current
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.services.assets.base import (  # noqa: F401
    AssetStore,
    AssetStoreError,
    Manifest,
    ManifestArtifact,
    ManifestError,
    ProbeResult,
)
from src.services.assets.factory import (  # noqa: F401
    build_asset_store,
    cloud_enabled,
    get_artifact_cache,
    get_keyframe_cache,
    get_manifest,
    reset_caches,
)
from src.services.assets.manifest import parse_manifest  # noqa: F401
from src.services.assets.sync import SyncReport, sync_artifacts  # noqa: F401

logger = logging.getLogger(__name__)

_IMG_RE = re.compile(r"\.(?:webp|jpe?g|png)$", re.IGNORECASE)


def _keyframe_rel_path(item: Any) -> str:
    """Best-effort ``<namespace>/<video_id>/<frame_id>.webp`` from a result item
    or a raw string."""
    if isinstance(item, (str, Path)):
        return str(item).replace("\\", "/").lstrip("/")
    if not isinstance(item, dict):
        return ""

    def first(*keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value not in (None, "", [], {}):
                return str(value)
        return ""

    raw = first("frame_path", "image_path", "keyframe_path").replace("\\", "/").strip("/")
    if raw:
        return raw
    video_id = first("video_id", "video_key", "videoKey")
    split = first("split", "namespace", "folder_key", "folderKey") or (
        video_id.split("_")[0] if video_id else ""
    )
    frame_name = first("frame_name", "frameName")
    frame_id = first("frame_id", "frame_key", "n")
    name = frame_name or frame_id
    if name and video_id and frame_name.startswith(f"{video_id}_"):
        name = frame_name[len(video_id) + 1 :]
    if name and not _IMG_RE.search(name):
        name = f"{name}.webp"
    if split and video_id and name:
        return f"{split}/{video_id}/{name}"
    return name


def resolve_keyframe_file(item: Any, *, settings=None) -> Path | None:
    """Return a local keyframe path, downloading + LRU-caching it on a miss.

    ``None`` when cloud assets are disabled, the path can't be derived, or the
    download fails (callers then fall back to their local lookup / placeholder).
    """
    settings = settings or get_settings()
    if not cloud_enabled(settings):
        return None
    rel = _keyframe_rel_path(item)
    if not rel:
        return None

    cache = get_keyframe_cache(settings)
    hit = cache.get(rel)
    if hit is not None:
        return hit

    store = build_asset_store(settings)
    if store is None:
        return None
    manifest = get_manifest(store)
    prefix = ""
    container = "keyframes"
    if manifest is not None and isinstance(manifest.keyframes, dict):
        prefix = str(manifest.keyframes.get("prefix") or "").strip("/")
        container = str(manifest.keyframes.get("container") or "keyframes")
    key = f"{prefix}/{rel}" if prefix else rel
    try:
        data = store.read_object(container, key)
    except AssetStoreError as exc:
        logger.info("keyframe fetch failed for %s: %s", rel, exc)
        return None
    if not data:
        return None
    try:
        return cache.put(rel, data)
    except (OSError, ValueError) as exc:
        logger.info("keyframe cache write failed for %s: %s", rel, exc)
        return None


def resolve_artifact_path(name: str, *, settings=None) -> Path | None:
    """Local path of a synced artifact from the *current* manifest version."""
    settings = settings or get_settings()
    if not cloud_enabled(settings):
        return None
    cache = get_artifact_cache(settings)
    current = cache.get_current()
    if not current:
        return None
    slot = cache.slot(current, name)
    return slot.path if slot.present else None
