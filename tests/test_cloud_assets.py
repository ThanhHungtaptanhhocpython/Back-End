"""Phase 4 -- cloud asset storage (manifest, caches, sync, adapters)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.config.settings import Settings  # noqa: E402
from src.services import assets  # noqa: E402
from src.services.assets.azure_blob import AzureBlobAssetStore  # noqa: E402
from src.services.assets.base import AssetStore, AssetStoreError, ManifestError  # noqa: E402
from src.services.assets.local_cache import ArtifactCache, KeyframeCache  # noqa: E402
from src.services.assets.manifest import parse_manifest  # noqa: E402
from src.services.assets.s3_compatible import S3AssetStore  # noqa: E402
from src.services.assets.sync import sync_artifacts  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _installed(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


_AZURE_INSTALLED = _installed("azure.storage.blob")
_BOTO3_INSTALLED = _installed("boto3")


# ---------------------------------------------------------------------------
# in-memory store + manifest builder
# ---------------------------------------------------------------------------
class InMemoryStore(AssetStore):
    provider_id = "memory"

    def __init__(self, objects: dict[tuple[str, str], bytes], manifest_key="hcmai-assets.json"):
        self.objects = objects
        self.manifest_key = manifest_key
        self.reads: list[tuple[str, str]] = []

    def probe(self):
        from src.services.assets.base import ProbeResult

        return ProbeResult(True, self.provider_id, containers=sorted({c for c, _ in self.objects}))

    def fetch_manifest(self):
        return parse_manifest(self.objects[("metadata", self.manifest_key)])

    def open_object(self, container: str, key: str, *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        self.reads.append((container, key))
        try:
            blob = self.objects[(container, key)]
        except KeyError as exc:
            raise AssetStoreError(f"missing object {container}/{key}") from exc
        for i in range(0, len(blob), 8):
            yield blob[i : i + 8]

    def object_size(self, container: str, key: str) -> int | None:
        blob = self.objects.get((container, key))
        return len(blob) if blob is not None else None


def _manifest(version: str, arts: list[dict], keyframes: dict | None = None) -> bytes:
    doc = {"version": version, "artifacts": arts}
    if keyframes is not None:
        doc["keyframes"] = keyframes
    return json.dumps(doc).encode("utf-8")


# ---------------------------------------------------------------------------
class TestManifestParsing:
    def _art(self, data=b"x"):
        return {"name": "faiss_index", "container": "embeddings", "key": "a.index",
                "size": len(data), "sha256": _sha(data)}

    def test_valid_manifest(self):
        m = parse_manifest(_manifest("v1", [self._art()]))
        assert m.version == "v1" and m.artifacts[0].name == "faiss_index"
        assert m.keyframes["layout"].endswith(".webp")

    def test_missing_version(self):
        with pytest.raises(ManifestError):
            parse_manifest(json.dumps({"artifacts": [self._art()]}))

    def test_bad_sha(self):
        art = self._art()
        art["sha256"] = "nothex"
        with pytest.raises(ManifestError):
            parse_manifest(_manifest("v1", [art]))

    def test_negative_size(self):
        art = self._art()
        art["size"] = -1
        with pytest.raises(ManifestError):
            parse_manifest(_manifest("v1", [art]))

    def test_duplicate_name(self):
        with pytest.raises(ManifestError):
            parse_manifest(_manifest("v1", [self._art(), self._art()]))

    def test_artifacts_not_array(self):
        with pytest.raises(ManifestError):
            parse_manifest(json.dumps({"version": "v1", "artifacts": {}}))

    def test_not_json(self):
        with pytest.raises(ManifestError):
            parse_manifest(b"\x00 not json")


# ---------------------------------------------------------------------------
class TestArtifactCache:
    def test_promote_verifies_checksum(self, tmp_path: Path):
        cache = ArtifactCache(tmp_path)
        data = b"payload-1234"
        staged = cache.stage_path("faiss_index")
        staged.write_bytes(data)
        cache.promote(staged, "v1", "faiss_index", _sha(data))
        slot = cache.slot("v1", "faiss_index", expected_sha=_sha(data), expected_size=len(data))
        assert slot.present and slot.verified and slot.path.read_bytes() == data
        assert not staged.exists()

    def test_promote_rejects_wrong_checksum(self, tmp_path: Path):
        cache = ArtifactCache(tmp_path)
        staged = cache.stage_path("faiss_index")
        staged.write_bytes(b"actual")
        with pytest.raises(ValueError):
            cache.promote(staged, "v1", "faiss_index", _sha(b"expected-something-else"))
        assert not cache.artifact_path("v1", "faiss_index").exists()

    def test_current_pointer_and_completeness(self, tmp_path: Path):
        cache = ArtifactCache(tmp_path)
        assert cache.get_current() is None
        for name in ("a", "b"):
            staged = cache.stage_path(name)
            staged.write_bytes(name.encode())
            cache.promote(staged, "v1", name, _sha(name.encode()))
        assert cache.is_version_complete("v1", ["a", "b"]) is True
        assert cache.is_version_complete("v1", ["a", "b", "c"]) is False
        cache.set_current("v1")
        assert cache.get_current() == "v1"

    def test_clear(self, tmp_path: Path):
        cache = ArtifactCache(tmp_path)
        staged = cache.stage_path("a")
        staged.write_bytes(b"12345")
        cache.promote(staged, "v1", "a", _sha(b"12345"))
        freed = cache.clear()
        assert freed >= 5 and cache.usage_bytes() == 0


class TestKeyframeCache:
    def test_put_get_roundtrip(self, tmp_path: Path):
        kf = KeyframeCache(tmp_path, max_bytes=0)
        path = kf.put("L21/L21_V001/001.webp", b"imgdata")
        assert path.read_bytes() == b"imgdata"
        assert kf.get("L21/L21_V001/001.webp") == path
        assert kf.get("missing/x.webp") is None

    def test_lru_eviction(self, tmp_path: Path):
        kf = KeyframeCache(tmp_path, max_bytes=20)
        kf.put("a.webp", b"0123456789")  # 10
        kf.get("a.webp")
        kf.put("b.webp", b"0123456789")  # 20 total
        kf.put("c.webp", b"0123456789")  # 30 -> evict LRU (a)
        assert kf.get("a.webp") is None
        assert kf.get("c.webp") is not None
        assert kf.stats()["usage_bytes"] <= 20

    def test_rejects_escape_path(self, tmp_path: Path):
        kf = KeyframeCache(tmp_path, max_bytes=0)
        with pytest.raises(ValueError):
            kf.put("../../evil.webp", b"x")


# ---------------------------------------------------------------------------
class TestSync:
    def _store(self, version="2026-01") -> tuple[InMemoryStore, dict]:
        faiss = b"FAISS-INDEX-BYTES"
        parquet = b"GLOBAL-IDS-PARQUET"
        objs = {
            ("embeddings", "beit3/faiss.index"): faiss,
            ("embeddings", "beit3/global_ids.parquet"): parquet,
        }
        arts = [
            {"name": "faiss_index", "container": "embeddings", "key": "beit3/faiss.index",
             "size": len(faiss), "sha256": _sha(faiss)},
            {"name": "global_ids", "container": "embeddings", "key": "beit3/global_ids.parquet",
             "size": len(parquet), "sha256": _sha(parquet)},
        ]
        objs[("metadata", "hcmai-assets.json")] = _manifest(version, arts)
        return InMemoryStore(objs), {"faiss": faiss, "parquet": parquet, "arts": arts}

    def test_happy_path_promotes(self, tmp_path: Path):
        store, _ = self._store()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache)
        assert report.ok and report.promoted
        assert report.current_version == "2026-01"
        assert {r.status for r in report.results} == {"synced"}
        assert cache.slot("2026-01", "faiss_index").verified

    def test_resync_reports_cached(self, tmp_path: Path):
        store, _ = self._store()
        cache = ArtifactCache(tmp_path)
        sync_artifacts(store, cache)
        report = sync_artifacts(store, cache)
        assert {r.status for r in report.results} == {"cached"}

    def test_checksum_mismatch_blocks_promotion(self, tmp_path: Path):
        store, meta = self._store()
        bad = json.loads(store.objects[("metadata", "hcmai-assets.json")])
        bad["artifacts"][0]["sha256"] = _sha(b"WRONG")
        store.objects[("metadata", "hcmai-assets.json")] = json.dumps(bad).encode()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache)
        assert report.promoted is False
        statuses = {r.name: r.status for r in report.results}
        assert statuses["faiss_index"] == "checksum_mismatch"
        assert statuses["global_ids"] == "synced"
        assert cache.get_current() is None

    def test_size_mismatch_detected(self, tmp_path: Path):
        store, _ = self._store()
        doc = json.loads(store.objects[("metadata", "hcmai-assets.json")])
        doc["artifacts"][0]["size"] = 999999
        store.objects[("metadata", "hcmai-assets.json")] = json.dumps(doc).encode()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache)
        assert {r.name: r.status for r in report.results}["faiss_index"] == "size_mismatch"

    def test_subset_sync_not_promoted(self, tmp_path: Path):
        store, _ = self._store()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=["faiss_index"])
        assert [r.name for r in report.results] == ["faiss_index"]
        assert report.promoted is False and cache.get_current() is None

    def test_download_error_surfaced(self, tmp_path: Path):
        store, _ = self._store()
        del store.objects[("embeddings", "beit3/faiss.index")]
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache)
        assert {r.name: r.status for r in report.results}["faiss_index"] == "download_error"


# ---------------------------------------------------------------------------
class TestAdaptersDegradeGracefully:
    def _settings(self, **over):
        base = dict(
            _env_file=None,
            cloud_assets_enabled=True,
            azure_storage_connection_string="cs",
            s3_bucket="b", s3_access_key_id="ak", s3_secret_access_key="sk",
        )
        base.update(over)
        return Settings(**base)

    @pytest.mark.skipif(_AZURE_INSTALLED, reason="azure-storage-blob is installed")
    def test_azure_without_sdk(self):
        probe = AzureBlobAssetStore(self._settings()).probe()
        assert probe.ok is False and probe.sdk_available is False

    @pytest.mark.skipif(_BOTO3_INSTALLED, reason="boto3 is installed")
    def test_s3_without_sdk(self):
        probe = S3AssetStore(self._settings()).probe()
        assert probe.ok is False and probe.sdk_available is False


class TestS3WithFakeClient:
    class _FakeBody(io.BytesIO):
        def iter_chunks(self, size):
            while True:
                chunk = self.read(size)
                if not chunk:
                    break
                yield chunk

    class _FakeS3:
        def __init__(self, objs):
            self.objs = objs

        def head_bucket(self, Bucket):  # noqa: N803
            return {}

        def get_object(self, Bucket, Key):  # noqa: N803
            return {"Body": TestS3WithFakeClient._FakeBody(self.objs[Key])}

        def head_object(self, Bucket, Key):  # noqa: N803
            return {"ContentLength": len(self.objs[Key])}

    def test_probe_and_manifest_with_fake_client(self):
        payload = b"FAISS"
        arts = [{"name": "faiss_index", "container": "embeddings", "key": "beit3/f.index",
                 "size": len(payload), "sha256": _sha(payload)}]
        objs = {
            "metadata/hcmai-assets.json": _manifest("v9", arts),
            "embeddings/beit3/f.index": payload,
        }
        s = Settings(_env_file=None, cloud_assets_enabled=True, cloud_assets_provider="s3_compatible",
                     s3_bucket="bkt", s3_access_key_id="ak", s3_secret_access_key="sk",
                     s3_metadata_prefix="metadata/")
        store = S3AssetStore(s, client=self._FakeS3(objs))
        assert store.probe().ok is True
        manifest = store.fetch_manifest()
        assert manifest.version == "v9"
        assert store.read_object("embeddings", "beit3/f.index") == payload


# ---------------------------------------------------------------------------
class TestResolveKeyframeFile:
    def test_disabled_returns_none(self):
        s = Settings(_env_file=None, cloud_assets_enabled=False)
        assert assets.resolve_keyframe_file({"frame_path": "a/b/c.webp"}, settings=s) is None

    def test_fetches_then_serves_from_cache(self, tmp_path: Path, monkeypatch):
        img = b"WEBPDATA"
        objs = {("keyframes", "L21/L21_V001/001.webp"): img}
        store = InMemoryStore(objs)
        monkeypatch.setattr(assets, "build_asset_store", lambda settings=None: store)
        monkeypatch.setattr(assets, "get_manifest", lambda *a, **k: None)
        assets.reset_caches()

        s = Settings(
            _env_file=None, cloud_assets_enabled=True, cloud_assets_provider="s3_compatible",
            cloud_assets_cache_path=str(tmp_path), cloud_assets_keyframe_cache_max_bytes=10_000_000,
        )
        item = {"frame_path": "L21/L21_V001/001.webp"}
        first = assets.resolve_keyframe_file(item, settings=s)
        assert first is not None and first.read_bytes() == img
        assert store.reads == [("keyframes", "L21/L21_V001/001.webp")]

        second = assets.resolve_keyframe_file(item, settings=s)
        assert second == first
        assert len(store.reads) == 1  # served from LRU cache, no second download
        assets.reset_caches()
