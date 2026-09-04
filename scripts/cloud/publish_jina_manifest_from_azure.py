#!/usr/bin/env python
"""Publish a cloud-assets manifest for the final Jina runtime artifacts.

The merge notebook uploaded every artifact with SHA-256 blob metadata. This
script reads blob size/hash properties and can publish a validated manifest
without downloading the multi-GB FAISS index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(backend_root()))

from scripts.cloud.build_asset_manifest import build_manifest_doc  # noqa: E402
from src.services.assets.manifest import parse_manifest  # noqa: E402


ARTIFACT_FILES = {
    "jina_faiss_index": "jina_faiss.index",
    "jina_global_ids": "global_ids.parquet",
    "jina_video_metadata": "video_metadata.parquet",
    "jina_index_meta": "index_meta.json",
}


def _connection_string(cli_value: str | None) -> str | None:
    if cli_value:
        return cli_value.strip()
    value = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if value:
        return value.strip()
    try:
        from dotenv import dotenv_values

        value = dotenv_values(backend_root() / ".env").get("AZURE_STORAGE_CONNECTION_STRING")
        if value:
            return str(value).strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.config.settings import get_settings

        value = get_settings().azure_storage_connection_string
        if value:
            return str(value).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.config.runtime_store import get_store

        store = get_store()
        if store is not None:
            value = store.effective_values().get("AZURE_STORAGE_CONNECTION_STRING")
            return str(value).strip() if value else None
    except Exception:  # noqa: BLE001
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection-string")
    parser.add_argument("--embeddings-container", default="embeddings")
    parser.add_argument("--metadata-container", default="metadata")
    parser.add_argument(
        "--index-prefix",
        default="indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina",
    )
    parser.add_argument("--manifest-key", default="hcmai-assets-jina.json")
    parser.add_argument("--version", default="fine-keyframes-jina-v2-2026-09-02")
    parser.add_argument("--out", type=Path, default=Path("hcmai-assets-jina.json"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)

    connection_string = _connection_string(args.connection_string)
    if not connection_string:
        raise SystemExit(
            "Azure credentials missing. Set AZURE_STORAGE_CONNECTION_STRING in the "
            "backend environment/runtime settings or pass --connection-string."
        )

    try:
        from azure.storage.blob import BlobServiceClient, ContentSettings
    except ImportError as exc:
        raise SystemExit("azure-storage-blob is required") from exc

    service = BlobServiceClient.from_connection_string(connection_string)
    artifacts: list[dict] = []
    prefix = args.index_prefix.strip("/")
    for name, filename in ARTIFACT_FILES.items():
        key = f"{prefix}/{filename}"
        client = service.get_blob_client(container=args.embeddings_container, blob=key)
        properties = client.get_blob_properties()
        sha256 = str((properties.metadata or {}).get("sha256") or "").lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise SystemExit(
                f"{args.embeddings_container}/{key} has no valid sha256 blob metadata. "
                "Rerun the merge upload cell before publishing the manifest."
            )
        artifacts.append(
            {
                "name": name,
                "container": "embeddings",
                "key": key,
                "size": int(properties.size),
                "sha256": sha256,
            }
        )
        print(f"verified {name}: {properties.size:,} bytes sha256={sha256[:12]}...")

    doc = build_manifest_doc(
        args.version,
        artifacts,
        {
            "container": "keyframes",
            "prefix": "",
            "layout": "{namespace}/{video_id}/keyframe_{ordinal:04d}.jpg",
        },
    )
    parsed = parse_manifest(doc)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(parsed.artifacts)} artifacts)")

    if args.upload:
        payload = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
        service.get_blob_client(
            container=args.metadata_container,
            blob=args.manifest_key,
        ).upload_blob(
            payload,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        print(f"uploaded {len(payload)} bytes -> {args.metadata_container}/{args.manifest_key}")
    else:
        print("manifest was not uploaded; rerun with --upload after reviewing it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
