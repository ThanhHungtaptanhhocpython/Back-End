#!/usr/bin/env python
"""Build (and optionally upload) ``hcmai-assets.json`` for the cloud asset store.

The runtime app only *reads* the dataset; this standalone ops script is how a
team publishes the manifest that describes it. Point it at the built runtime
artifacts (FAISS index, parquet files, BEiT3 checkpoint, tokenizer,
media-info, map-keyframes), it hashes each one (SHA-256 + size) and writes a
manifest the backend can verify against.

Usage
-----
Spec file (recommended)::

    python scripts/cloud/build_asset_manifest.py --spec scripts/cloud/manifest_spec.example.json \
        --out hcmai-assets.json

Ad-hoc::

    python scripts/cloud/build_asset_manifest.py --version 2026-09-01 \
        --artifact faiss_index=/data/beit3_faiss.index@embeddings/beit3/beit3_faiss.index \
        --artifact global_ids=/data/global_ids.parquet@embeddings/beit3/global_ids.parquet \
        --artifact video_metadata=/data/video_metadata.parquet \
        --artifact index_meta=/data/index_meta.json \
        --artifact checkpoint=/data/beit3_large.pth@metadata/beit3/beit3_large.pth \
        --artifact tokenizer=/data/beit3.spm@metadata/beit3/beit3.spm \
        --keyframes-container keyframes --keyframes-layout "{namespace}/{video_id}/{frame_id}.webp" \
        --out hcmai-assets.json

Add ``--upload`` to push it to ``<metadata-container>/hcmai-assets.json`` using
the Azure connection string (``--connection-string``, then
``AZURE_STORAGE_CONNECTION_STRING`` env, then the runtime config store).
Otherwise the script prints the ``az`` / ``aws`` command to run yourself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(backend_root()))

from src.services.assets.base import ManifestError  # noqa: E402
from src.services.assets.manifest import _DEFAULT_CONTAINER, parse_manifest  # noqa: E402

_CHUNK = 1 << 20
_KEYFRAME_DEFAULTS = {
    "container": "keyframes",
    "prefix": "",
    "layout": "{namespace}/{video_id}/{frame_id}.webp",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
            total += len(block)
    return h.hexdigest(), total


def build_manifest_doc(version: str, artifacts: list[dict], keyframes: dict | None) -> dict:
    """Pure builder. Each ``artifacts`` entry needs ``name`` + ``key`` and either
    a ``local`` path (hashed here) or explicit ``sha256`` + ``size``."""
    out: list[dict] = []
    for spec in artifacts:
        name = str(spec["name"]).strip()
        key = str(spec["key"]).strip()
        if not name or not key:
            raise SystemExit(f"artifact needs both 'name' and 'key': {spec!r}")
        container = str(spec.get("container") or _DEFAULT_CONTAINER.get(name, "metadata"))
        if spec.get("sha256") and spec.get("size") is not None:
            sha256, size = str(spec["sha256"]).lower(), int(spec["size"])
        else:
            local = Path(str(spec["local"])).expanduser()
            if not local.is_file():
                raise SystemExit(f"artifact '{name}': local file not found: {local}")
            print(f"  hashing {name}: {local} …", flush=True)
            sha256, size = sha256_file(local)
        out.append(
            {
                "name": name,
                "container": container,
                "key": key,
                "size": size,
                "sha256": sha256,
                "kind": str(spec.get("kind") or "artifact"),
            }
        )

    doc: dict = {"version": str(version).strip(), "generated_at": _now(), "artifacts": out}
    kf = dict(_KEYFRAME_DEFAULTS)
    kf.update({k: v for k, v in (keyframes or {}).items() if v is not None})
    doc["keyframes"] = kf
    return doc


# ---------------------------------------------------------------------------
def _parse_artifact_flag(raw: str) -> dict:
    """``name=LOCALPATH[@CONTAINER/KEY]`` -> spec dict."""
    if "=" not in raw:
        raise SystemExit(f"--artifact must be name=path[@container/key], got: {raw!r}")
    name, _, rest = raw.partition("=")
    local, sep, target = rest.partition("@")
    spec = {"name": name.strip(), "local": local.strip()}
    if sep and target.strip():
        container, _, key = target.strip().partition("/")
        if not key:
            raise SystemExit(f"--artifact target must be container/key, got: {target!r}")
        spec["container"] = container
        spec["key"] = key
    else:
        spec["key"] = Path(local.strip()).name
    return spec


def _load_spec(path: Path) -> tuple[str, list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version", "")
    raw = data.get("artifacts", {})
    artifacts: list[dict] = []
    if isinstance(raw, dict):
        for name, body in raw.items():
            entry = {"name": name, **(body or {})}
            entry.setdefault("key", Path(str(entry.get("local", name))).name)
            artifacts.append(entry)
    elif isinstance(raw, list):
        artifacts = list(raw)
    else:
        raise SystemExit("spec 'artifacts' must be an object or an array")
    return version, artifacts, data.get("keyframes") or {}


def _resolve_connection_string(cli_value: str | None) -> str | None:
    if cli_value:
        return cli_value
    import os

    env = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if env:
        return env
    try:
        from src.config.runtime_store import get_store

        store = get_store()
        if store is not None:
            return store.effective_values().get("AZURE_STORAGE_CONNECTION_STRING") or None
    except Exception:  # noqa: BLE001
        pass
    return None


def _upload_azure(doc: dict, container: str, key: str, connection_string: str) -> None:
    from azure.storage.blob import BlobServiceClient

    payload = json.dumps(doc, indent=2).encode("utf-8")
    svc = BlobServiceClient.from_connection_string(connection_string)
    svc.get_blob_client(container=container, blob=key).upload_blob(payload, overwrite=True)
    print(f"uploaded {len(payload)} bytes -> {container}/{key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, help="JSON spec file")
    parser.add_argument("--version", help="manifest version (overrides the spec)")
    parser.add_argument("--artifact", action="append", default=[],
                        metavar="name=path[@container/key]", help="repeatable")
    parser.add_argument("--keyframes-container")
    parser.add_argument("--keyframes-prefix")
    parser.add_argument("--keyframes-layout")
    parser.add_argument("--out", type=Path, default=Path("hcmai-assets.json"))
    parser.add_argument("--manifest-key", default="hcmai-assets.json")
    parser.add_argument("--metadata-container", default="metadata")
    parser.add_argument("--upload", action="store_true", help="push to Azure <metadata>/<manifest-key>")
    parser.add_argument("--connection-string", help="Azure connection string for --upload")
    args = parser.parse_args(argv)

    version, artifacts, keyframes = ("", [], {})
    if args.spec:
        version, artifacts, keyframes = _load_spec(args.spec)
    if args.version:
        version = args.version
    artifacts += [_parse_artifact_flag(a) for a in args.artifact]
    for cli_key, kf_key in (
        ("keyframes_container", "container"),
        ("keyframes_prefix", "prefix"),
        ("keyframes_layout", "layout"),
    ):
        val = getattr(args, cli_key)
        if val is not None:
            keyframes[kf_key] = val

    if not version:
        raise SystemExit("a --version (or spec 'version') is required")
    if not artifacts:
        raise SystemExit("no artifacts given (use --spec or --artifact)")

    doc = build_manifest_doc(version, artifacts, keyframes)

    try:
        parsed = parse_manifest(json.dumps(doc))
    except ManifestError as exc:
        raise SystemExit(f"built manifest is invalid: {exc}")

    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  (version {parsed.version}, {len(parsed.artifacts)} artifacts)")
    for art in parsed.artifacts:
        print(f"  {art.name:16} {art.container}/{art.key}  {art.size:,} B")

    if args.upload:
        conn = _resolve_connection_string(args.connection_string)
        if not conn:
            raise SystemExit(
                "--upload needs an Azure connection string (--connection-string, "
                "AZURE_STORAGE_CONNECTION_STRING env, or the runtime config store)."
            )
        _upload_azure(doc, args.metadata_container, args.manifest_key, conn)
    else:
        print("\nTo publish it, run one of:")
        print(f"  az storage blob upload -c {args.metadata_container} -n {args.manifest_key} "
              f"-f {args.out} --overwrite --connection-string \"$AZURE_STORAGE_CONNECTION_STRING\"")
        print(f"  aws s3 cp {args.out} s3://<bucket>/<metadata-prefix>{args.manifest_key} "
              f"--endpoint-url <S3_ENDPOINT_URL>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
