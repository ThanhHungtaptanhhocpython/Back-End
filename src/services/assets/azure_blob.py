"""Azure Blob Storage :class:`AssetStore` adapter.

``azure-storage-blob`` is imported lazily; when it is missing the store still
constructs and :meth:`probe` reports ``sdk_available=False`` so the UI can tell
the member to install it.
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
)
from src.services.assets.manifest import parse_manifest


class AzureBlobAssetStore(AssetStore):
    provider_id = "azure_blob"

    def __init__(self, settings: Settings | None = None, *, client: Any = None) -> None:
        self._settings = settings or get_settings()
        self._client = client  # injectable BlobServiceClient (tests / reuse)
        self._container_map = {
            CONTAINER_METADATA: self._settings.azure_blob_container_metadata or "metadata",
            CONTAINER_EMBEDDINGS: self._settings.azure_blob_container_embeddings or "embeddings",
            CONTAINER_KEYFRAMES: self._settings.azure_blob_container_keyframes or "keyframes",
        }

    # -- client -------------------------------------------------------------
    def _service(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise AssetStoreError("azure-storage-blob is not installed") from exc

        conn = (self._settings.azure_storage_connection_string or "").strip()
        if conn:
            self._client = BlobServiceClient.from_connection_string(conn)
            return self._client
        account = (self._settings.azure_storage_account_name or "").strip()
        key = (self._settings.azure_storage_primary_key or "").strip()
        if account and key:
            self._client = BlobServiceClient(
                account_url=f"https://{account}.blob.core.windows.net", credential=key
            )
            return self._client
        raise AssetStoreError(
            "Azure Blob needs AZURE_STORAGE_CONNECTION_STRING, or "
            "AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_PRIMARY_KEY."
        )

    def _real_container(self, logical: str) -> str:
        return self._container_map.get(logical, logical)

    # -- AssetStore --------------------------------------------------------
    def probe(self) -> ProbeResult:
        try:
            service = self._service()
        except AssetStoreError as exc:
            sdk = "not installed" not in str(exc)
            return ProbeResult(False, self.provider_id, detail=str(exc), sdk_available=sdk)
        try:
            names = [c.name for c in service.list_containers()]
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(False, self.provider_id, detail=f"list_containers failed: {type(exc).__name__}")
        manifest_present = False
        manifest_version = None
        try:
            manifest = self.fetch_manifest()
            manifest_present = True
            manifest_version = manifest.version
        except Exception:  # noqa: BLE001
            pass
        return ProbeResult(
            True, self.provider_id, containers=names,
            manifest_present=manifest_present, manifest_version=manifest_version,
        )

    def fetch_manifest(self) -> Manifest:
        key = (self._settings.cloud_assets_manifest_key or "hcmai-assets.json").strip()
        data = self.read_object(CONTAINER_METADATA, key)
        return parse_manifest(data)

    def open_object(self, container: str, key: str, *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        service = self._service()
        try:
            blob = service.get_blob_client(container=self._real_container(container), blob=key)
            downloader = blob.download_blob()
        except Exception as exc:  # noqa: BLE001
            raise AssetStoreError(f"azure download failed for {container}/{key}: {type(exc).__name__}") from exc
        for chunk in downloader.chunks():
            yield chunk

    def object_size(self, container: str, key: str) -> int | None:
        service = self._service()
        try:
            blob = service.get_blob_client(container=self._real_container(container), blob=key)
            return int(blob.get_blob_properties().size)
        except Exception:  # noqa: BLE001
            return None
