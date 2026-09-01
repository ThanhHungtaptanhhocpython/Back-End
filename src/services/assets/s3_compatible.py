"""S3-compatible :class:`AssetStore` adapter (AWS S3, Cloudflare R2, MinIO, ...).

One bucket holds everything; the manifest's logical ``container`` becomes a key
prefix (``metadata/`` is configurable via ``S3_METADATA_PREFIX``). ``boto3`` is
imported lazily.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from src.config.settings import Settings, get_settings
from src.services.assets.base import (
    CONTAINER_EMBEDDINGS,
    CONTAINER_KEYFRAMES,
    CONTAINER_METADATA,
    AssetStore,
    AssetStoreError,
    Manifest,
    ProbeResult,
    err_detail,
)
from src.services.assets.manifest import parse_manifest


class S3AssetStore(AssetStore):
    provider_id = "s3_compatible"

    def __init__(self, settings: Settings | None = None, *, client: Any = None) -> None:
        self._settings = settings or get_settings()
        self._client = client  # injectable boto3 s3 client
        self._bucket = (self._settings.s3_bucket or "").strip()
        meta_prefix = (self._settings.s3_metadata_prefix or "").strip().strip("/")
        self._prefix_map = {
            CONTAINER_METADATA: f"{meta_prefix}/" if meta_prefix else "",
            CONTAINER_EMBEDDINGS: "embeddings/",
            CONTAINER_KEYFRAMES: "keyframes/",
        }

    def _s3(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise AssetStoreError("boto3 is not installed") from exc
        endpoint = (self._settings.s3_endpoint_url or "").strip() or None
        access = (self._settings.s3_access_key_id or "").strip()
        secret = (self._settings.s3_secret_access_key or "").strip()
        region = (self._settings.s3_region or "").strip() or None
        if not self._bucket or not access or not secret:
            raise AssetStoreError("S3 needs S3_BUCKET, S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY.")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name=region,
        )
        return self._client

    def _full_key(self, container: str, key: str) -> str:
        prefix = self._prefix_map.get(container, f"{container}/" if container else "")
        return f"{prefix}{key.lstrip('/')}"

    # -- AssetStore --------------------------------------------------------
    def probe(self) -> ProbeResult:
        try:
            s3 = self._s3()
        except AssetStoreError as exc:
            sdk = "not installed" not in str(exc)
            return ProbeResult(False, self.provider_id, detail=str(exc), sdk_available=sdk)
        try:
            s3.head_bucket(Bucket=self._bucket)
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(False, self.provider_id, detail=f"head_bucket failed: {err_detail(exc)}")
        manifest_present = False
        manifest_version = None
        try:
            manifest = self.fetch_manifest()
            manifest_present = True
            manifest_version = manifest.version
        except Exception:  # noqa: BLE001
            pass
        return ProbeResult(
            True, self.provider_id, containers=[self._bucket],
            manifest_present=manifest_present, manifest_version=manifest_version,
        )

    def fetch_manifest(self) -> Manifest:
        key = (self._settings.cloud_assets_manifest_key or "hcmai-assets.json").strip()
        return parse_manifest(self.read_object(CONTAINER_METADATA, key))

    def open_object(self, container: str, key: str, *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        s3 = self._s3()
        full = self._full_key(container, key)
        try:
            obj = s3.get_object(Bucket=self._bucket, Key=full)
            body = obj["Body"]
        except Exception as exc:  # noqa: BLE001
            raise AssetStoreError(f"s3 get_object failed for {full}: {type(exc).__name__}") from exc
        iterator = getattr(body, "iter_chunks", None)
        if callable(iterator):
            yield from iterator(chunk_size)
        else:  # fake / file-like
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def object_size(self, container: str, key: str) -> int | None:
        s3 = self._s3()
        try:
            head = s3.head_object(Bucket=self._bucket, Key=self._full_key(container, key))
            return int(head["ContentLength"])
        except Exception:  # noqa: BLE001
            return None
