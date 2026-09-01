"""Parse + validate the versioned ``hcmai-assets.json`` manifest."""

from __future__ import annotations

import json
import re
from typing import Any

from src.services.assets.base import (
    CONTAINER_EMBEDDINGS,
    CONTAINER_KEYFRAMES,
    CONTAINER_METADATA,
    Manifest,
    ManifestArtifact,
    ManifestError,
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Default logical container for each recognised artifact name.
_DEFAULT_CONTAINER = {
    "faiss_index": CONTAINER_EMBEDDINGS,
    "global_ids": CONTAINER_EMBEDDINGS,
    "video_metadata": CONTAINER_EMBEDDINGS,
    "index_meta": CONTAINER_EMBEDDINGS,
    "checkpoint": CONTAINER_METADATA,
    "tokenizer": CONTAINER_METADATA,
    "media_info": CONTAINER_METADATA,
    "map_keyframes": CONTAINER_METADATA,
    "jina_faiss_index": CONTAINER_EMBEDDINGS,
    "jina_global_ids": CONTAINER_EMBEDDINGS,
    "jina_index_meta": CONTAINER_EMBEDDINGS,
}

_DEFAULT_KEYFRAMES = {
    "container": CONTAINER_KEYFRAMES,
    "prefix": "",
    "layout": "{namespace}/{video_id}/{frame_id}.webp",
}


def parse_manifest(data: Any) -> Manifest:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = json.loads(data.decode("utf-8"))
        except ValueError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    elif isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")

    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ManifestError("manifest is missing a non-empty string 'version'")
    version = version.strip()

    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ManifestError("manifest 'artifacts' must be a non-empty array")

    artifacts: list[ManifestArtifact] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_artifacts):
        if not isinstance(entry, dict):
            raise ManifestError(f"artifacts[{index}] must be an object")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ManifestError(f"artifacts[{index}] is missing 'name'")
        if name in seen:
            raise ManifestError(f"duplicate artifact name: {name}")
        seen.add(name)

        key = str(entry.get("key") or "").strip()
        if not key:
            raise ManifestError(f"artifact '{name}' is missing 'key'")

        sha256 = str(entry.get("sha256") or "").strip().lower()
        if not _SHA256_RE.match(sha256):
            raise ManifestError(f"artifact '{name}' has an invalid sha256")

        size_raw = entry.get("size")
        try:
            size = int(size_raw)
        except (TypeError, ValueError):
            raise ManifestError(f"artifact '{name}' has a non-integer 'size'")
        if size < 0:
            raise ManifestError(f"artifact '{name}' has a negative 'size'")

        container = str(entry.get("container") or "").strip() or _DEFAULT_CONTAINER.get(
            name, CONTAINER_METADATA
        )
        kind = str(entry.get("kind") or "artifact").strip().lower()
        if kind not in ("artifact", "archive"):
            kind = "artifact"

        artifacts.append(
            ManifestArtifact(name=name, key=key, container=container, size=size, sha256=sha256, kind=kind)
        )

    keyframes = data.get("keyframes")
    if not isinstance(keyframes, dict):
        keyframes = dict(_DEFAULT_KEYFRAMES)
    else:
        merged = dict(_DEFAULT_KEYFRAMES)
        merged.update({k: v for k, v in keyframes.items() if v is not None})
        keyframes = merged

    generated_at = data.get("generated_at")
    if generated_at is not None and not isinstance(generated_at, str):
        generated_at = str(generated_at)

    return Manifest(
        version=version,
        generated_at=generated_at,
        artifacts=artifacts,
        keyframes=keyframes,
        raw=data,
    )
