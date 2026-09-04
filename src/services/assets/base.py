"""``AssetStore`` interface + manifest model shared by the Azure Blob and
S3-compatible adapters.

The dataset is uploaded ahead of time by an operator; this code only ever
*reads* it. A versioned ``hcmai-assets.json`` manifest in the metadata
container/bucket declares every runtime artifact (model checkpoint, FAISS
index, parquet files, tokenizer, media-info, map-keyframes) with its object
key, byte size and SHA-256.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# Logical containers referenced by the manifest; adapters map these to the
# real Azure container names / S3 key prefixes from Settings.
CONTAINER_METADATA = "metadata"
CONTAINER_EMBEDDINGS = "embeddings"
CONTAINER_KEYFRAMES = "keyframes"

# Recognised artifact names (free-form names are still allowed).
ARTIFACT_NAMES = (
    "faiss_index",
    "global_ids",
    "video_metadata",
    "index_meta",
    "checkpoint",
    "tokenizer",
    "media_info",
    "map_keyframes",
    "jina_faiss_index",
    "jina_global_ids",
    "jina_video_metadata",
    "jina_index_meta",
)


class AssetStoreError(RuntimeError):
    pass


class ManifestError(AssetStoreError):
    pass


@dataclass
class ManifestArtifact:
    name: str
    key: str
    container: str
    size: int
    sha256: str
    kind: str = "artifact"  # "artifact" | "archive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "container": self.container,
            "size": self.size,
            "sha256": self.sha256,
            "kind": self.kind,
        }


@dataclass
class Manifest:
    version: str
    generated_at: str | None
    artifacts: list[ManifestArtifact]
    keyframes: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    def artifact(self, name: str) -> ManifestArtifact | None:
        for art in self.artifacts:
            if art.name == name:
                return art
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "keyframes": self.keyframes,
        }


def err_detail(exc: BaseException, limit: int = 320) -> str:
    """A short, human string for an SDK exception -- the type plus its message
    (which for network errors usually names the host / underlying errno)."""
    name = type(exc).__name__
    msg = " ".join(str(exc).split())
    if len(msg) > limit:
        msg = msg[:limit] + "…"
    return f"{name}: {msg}" if msg and msg != name else name


@dataclass
class ProbeResult:
    ok: bool
    provider: str
    detail: str = ""
    sdk_available: bool = True
    containers: list[str] = field(default_factory=list)
    manifest_present: bool = False
    manifest_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "detail": self.detail,
            "sdk_available": self.sdk_available,
            "containers": self.containers,
            "manifest_present": self.manifest_present,
            "manifest_version": self.manifest_version,
        }


class AssetStore(ABC):
    """Read-only access to a remote dataset (Azure Blob or S3-compatible)."""

    provider_id: str = "base"

    @abstractmethod
    def probe(self) -> ProbeResult:
        """Check credentials / reachability without downloading anything."""

    @abstractmethod
    def fetch_manifest(self) -> Manifest:
        """Download + parse ``hcmai-assets.json`` from the metadata container."""

    @abstractmethod
    def open_object(self, container: str, key: str, *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        """Yield the object's bytes in chunks. Raise :class:`AssetStoreError` on failure."""

    @abstractmethod
    def object_size(self, container: str, key: str) -> int | None:
        """Return the object's size in bytes, or ``None`` if unknown/missing."""

    # -- shared helper ----------------------------------------------------
    def read_object(self, container: str, key: str) -> bytes:
        buf = bytearray()
        for chunk in self.open_object(container, key):
            buf.extend(chunk)
        return bytes(buf)
