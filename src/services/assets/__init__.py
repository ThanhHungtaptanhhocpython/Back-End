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
    ARTIFACT_NAMES,
    BACKEND_ARTIFACT_NAMES,
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
_LAYOUT_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_]+)\}")
_LAYOUT_ALLOWED_FIELDS = {"namespace", "split", "video_id", "frame_id", "frame_name"}


def _item_field(item: dict, *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _safe_rel(raw: str) -> str:
    """Normalize a candidate relative path and reject any traversal attempt."""
    rel = raw.replace("\\", "/").strip("/")
    if not rel or ".." in rel.split("/"):
        return ""
    return rel


def _layout_rel_path(item: dict, layout: str | None) -> str:
    """Best-effort path from a manifest ``keyframes.layout`` template.

    This is a validated fallback only: it never overrides an item's own
    ``asset_key``/``frame_path``, only a whitelisted placeholder set is
    honoured, and a malformed layout or an item missing a value it needs
    yields no path rather than guessing one.
    """
    if not isinstance(layout, str) or not layout.strip():
        return ""
    layout = layout.strip()
    placeholders = set(_LAYOUT_PLACEHOLDER_RE.findall(layout))
    if not placeholders or not placeholders.issubset(_LAYOUT_ALLOWED_FIELDS):
        return ""

    video_id = _item_field(item, "video_id", "video_key", "videoKey")
    split = _item_field(item, "split", "namespace", "folder_key", "folderKey") or (
        video_id.split("_")[0] if video_id else ""
    )
    frame_id = _item_field(item, "frame_id", "frame_key", "n")
    frame_name = _item_field(item, "frame_name", "frameName") or frame_id
    values = {"namespace": split, "split": split, "video_id": video_id, "frame_id": frame_id, "frame_name": frame_name}
    if not all(values.get(name) for name in placeholders):
        return ""
    try:
        rel = layout.format(**values)
    except (KeyError, IndexError, ValueError):
        return ""
    return _safe_rel(rel)


def _keyframe_rel_path(item: Any, *, layout: str | None = None) -> str:
    """Best-effort ``<namespace>/<video_id>/<frame_id>.<ext>`` for a result
    item or a raw string.

    Priority: ``asset_key`` (the authoritative cloud key -- see
    scripts/cloud/build_jina_index.py) > ``frame_path`` / ``image_path`` /
    ``keyframe_path`` > a manifest ``keyframes.layout`` template (validated,
    see ``_layout_rel_path``) > the legacy video_id/frame_id heuristic used by
    older BEiT3 result shapes. A numeric frame id is never reformatted into a
    guessed filename like ``keyframe_0000.jpg`` -- only real mapping data
    (``asset_key``/``frame_path``, or a validated layout) produces one of
    those; the legacy heuristic below only ever emits the ``.webp`` names
    BEiT3 result rows already use.
    """
    if isinstance(item, (str, Path)):
        return _safe_rel(str(item))
    if not isinstance(item, dict):
        return ""

    asset_key = _safe_rel(_item_field(item, "asset_key"))
    if asset_key:
        return asset_key

    raw = _safe_rel(_item_field(item, "frame_path", "image_path", "keyframe_path"))
    if raw:
        return raw

    via_layout = _layout_rel_path(item, layout)
    if via_layout:
        return via_layout

    video_id = _item_field(item, "video_id", "video_key", "videoKey")
    split = _item_field(item, "split", "namespace", "folder_key", "folderKey") or (
        video_id.split("_")[0] if video_id else ""
    )
    frame_name = _item_field(item, "frame_name", "frameName")
    frame_id = _item_field(item, "frame_id", "frame_key", "n")
    name = frame_name or frame_id
    if name and video_id and frame_name.startswith(f"{video_id}_"):
        name = frame_name[len(video_id) + 1 :]
    if name and not _IMG_RE.search(name):
        name = f"{name}.webp"
    if split and video_id and name:
        return _safe_rel(f"{split}/{video_id}/{name}")
    return _safe_rel(name)


def resolve_keyframe_file(item: Any, *, settings=None) -> Path | None:
    """Return a local keyframe path, downloading + LRU-caching it on a miss.

    ``None`` when cloud assets are disabled, the path can't be derived, or the
    download fails (callers then fall back to their local lookup / placeholder).
    """
    settings = settings or get_settings()
    if not cloud_enabled(settings):
        return None

    rel = _keyframe_rel_path(item)
    store: AssetStore | None = None
    manifest: Manifest | None = None
    if not rel:
        # Only reach for the manifest's keyframes.layout fallback when the
        # item itself carries neither asset_key nor frame_path -- the common
        # case (both backends populate one of those) never pays this cost.
        store = build_asset_store(settings)
        if store is not None:
            manifest = get_manifest(store)
            layout = (
                manifest.keyframes.get("layout")
                if manifest is not None and isinstance(manifest.keyframes, dict)
                else None
            )
            rel = _keyframe_rel_path(item, layout=layout)
    if not rel:
        return None

    cache = get_keyframe_cache(settings)
    hit = cache.get(rel)
    if hit is not None:
        return hit

    store = store or build_asset_store(settings)
    if store is None:
        return None
    manifest = manifest if manifest is not None else get_manifest(store)
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
