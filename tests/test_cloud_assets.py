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
from src.services.assets.sync_state import get_sync_progress, run_tracked_sync  # noqa: E402


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

    def test_get_rejects_escape_path_without_raising(self, tmp_path: Path):
        """`get` is called on every resolve; it must degrade to a cache miss
        for a path-traversal attempt, not raise or read outside its root."""
        kf = KeyframeCache(tmp_path, max_bytes=0)
        outside = tmp_path.parent / "escaped-secret.txt"
        outside.write_bytes(b"do-not-read-me")
        try:
            assert kf.get("../escaped-secret.txt") is None
        finally:
            outside.unlink(missing_ok=True)


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

    def test_subset_sync_promotes_when_requested_subset_is_complete(self, tmp_path: Path):
        """A scoped sync (e.g. only the active backend's artifacts, see
        BACKEND_ARTIFACT_NAMES) must be able to reach 'current' on its own --
        a member syncing only Jina artifacts must never be blocked from
        promoting just because the manifest also lists BEiT3 artifacts they
        never asked to download."""
        store, _ = self._store()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=["faiss_index"])
        assert [r.name for r in report.results] == ["faiss_index"]
        assert report.promoted is True and cache.get_current() == report.version
        # The artifact that was never requested is simply absent -- not an error.
        assert cache.slot(report.version, "global_ids").present is False

    def test_subset_sync_with_bad_artifact_not_promoted(self, tmp_path: Path):
        """Even a scoped sync only promotes once every artifact it actually
        requested is present with a verified size + SHA-256 -- a stale/corrupt
        file for one of the requested names blocks promotion of the whole
        requested set."""
        store, _ = self._store()
        bad = json.loads(store.objects[("metadata", "hcmai-assets.json")])
        bad["artifacts"][0]["sha256"] = _sha(b"WRONG")
        store.objects[("metadata", "hcmai-assets.json")] = json.dumps(bad).encode()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=["faiss_index", "global_ids"])
        assert {r.name: r.status for r in report.results}["faiss_index"] == "checksum_mismatch"
        assert report.promoted is False and cache.get_current() is None

    def test_presence_only_check_does_not_gate_promotion(self, tmp_path: Path):
        """is_version_complete (presence-only) still exists for callers that
        only need a quick existence check, but sync_artifacts itself must gate
        promotion on is_version_verified (size + SHA-256), not just presence."""
        store, _ = self._store()
        cache = ArtifactCache(tmp_path)
        # A stale file sitting at the right path, with no valid sidecar, would
        # satisfy is_version_complete but must never satisfy is_version_verified.
        stale_dir = cache.version_dir("stale-v")
        stale_dir.mkdir(parents=True, exist_ok=True)
        (stale_dir / "faiss_index").write_bytes(b"not the real bytes")
        assert cache.is_version_complete("stale-v", ["faiss_index"]) is True
        arts = [a for a in [type("A", (), {"name": "faiss_index", "sha256": _sha(b"real"), "size": 4})()]]
        assert cache.is_version_verified("stale-v", arts) is False

    def test_download_error_surfaced(self, tmp_path: Path):
        store, _ = self._store()
        del store.objects[("embeddings", "beit3/faiss.index")]
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache)
        assert {r.name: r.status for r in report.results}["faiss_index"] == "download_error"


# ---------------------------------------------------------------------------
_JINA_PROFILE = ["jina_faiss_index", "jina_global_ids", "jina_video_metadata", "jina_index_meta"]
_BEIT3_PROFILE = ["faiss_index", "global_ids", "video_metadata", "index_meta", "checkpoint", "tokenizer"]


def _profile_store(profile, version="prof-v1", *, drop=(), corrupt=()):
    """A manifest declaring `profile` (minus `drop`), with `corrupt` artifacts
    whose published sha256 will not match the bytes."""
    objs, arts = {}, []
    for name in profile:
        if name in drop:
            continue
        blob = f"{name}-bytes-{version}".encode()
        key = f"p/{name}"
        objs[("embeddings", key)] = blob
        sha = _sha(b"WRONG") if name in corrupt else _sha(blob)
        arts.append({"name": name, "container": "embeddings", "key": key,
                     "size": len(blob), "sha256": sha})
    objs[("metadata", "hcmai-assets.json")] = _manifest(version, arts)
    return InMemoryStore(objs)


def _jina_profile_store(version="jina-v9", **kw):
    return _profile_store(_JINA_PROFILE, version, **kw)


class TestProfileAtomicPromotion:
    """Promotion is all-or-nothing across the *complete* active-backend
    profile. A manifest missing any profile artifact is a broken publish:
    nothing is downloaded and `current` never moves."""

    def test_complete_jina_profile_promotes_after_all_verify(self, tmp_path: Path):
        store = _jina_profile_store()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=list(_JINA_PROFILE), required=list(_JINA_PROFILE))
        assert report.ok and report.promoted
        assert cache.get_current() == report.version == "jina-v9"
        assert not report.errors

    def test_complete_beit3_profile_still_promotes(self, tmp_path: Path):
        store = _profile_store(_BEIT3_PROFILE, "beit3-v2")
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=list(_BEIT3_PROFILE), required=list(_BEIT3_PROFILE))
        assert report.ok and report.promoted and cache.get_current() == "beit3-v2"

    def test_one_bad_checksum_blocks_profile_promotion(self, tmp_path: Path):
        store = _jina_profile_store(corrupt=("jina_global_ids",))
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=list(_JINA_PROFILE), required=list(_JINA_PROFILE))
        statuses = {r.name: r.status for r in report.results}
        assert statuses["jina_global_ids"] == "checksum_mismatch"
        assert report.promoted is False and cache.get_current() is None

    def test_incomplete_download_does_not_promote(self, tmp_path: Path):
        store = _jina_profile_store()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=["jina_faiss_index"], required=list(_JINA_PROFILE))
        assert [r.name for r in report.results] == ["jina_faiss_index"]
        assert report.results[0].status == "synced"
        assert report.promoted is False and cache.get_current() is None

    @pytest.mark.parametrize("missing", _JINA_PROFILE)
    def test_manifest_missing_any_required_artifact_is_rejected(self, tmp_path: Path, missing):
        store = _jina_profile_store(drop=(missing,))
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=list(_JINA_PROFILE), required=list(_JINA_PROFILE))
        assert report.errors and missing in report.errors[0]
        assert report.results == []          # nothing downloaded
        assert store.reads == []             # not even a byte fetched
        assert report.promoted is False and cache.get_current() is None

    def test_jina_index_plus_meta_but_no_global_ids_is_rejected(self, tmp_path: Path):
        store = _jina_profile_store(drop=("jina_global_ids", "jina_video_metadata"))
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=list(_JINA_PROFILE), required=list(_JINA_PROFILE))
        assert report.errors and "jina_global_ids" in report.errors[0]
        assert report.promoted is False and cache.get_current() is None
        assert store.reads == []

    def test_unknown_artifact_name_is_a_validation_error(self, tmp_path: Path):
        store = _jina_profile_store()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=["not_a_real_artifact"])
        assert report.errors and report.ok is False
        assert report.promoted is False and report.results == [] and cache.get_current() is None

    def test_empty_names_request_is_a_validation_error(self, tmp_path: Path):
        store = _jina_profile_store()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=[])
        assert report.errors and report.ok is False and report.promoted is False

    def test_manual_subset_stages_without_promoting_when_promote_false(self, tmp_path: Path):
        store = _jina_profile_store()
        cache = ArtifactCache(tmp_path)
        report = sync_artifacts(store, cache, names=["jina_faiss_index"], promote=False)
        assert report.results[0].status == "synced"
        assert report.promoted is False and cache.get_current() is None

    def test_partial_sync_stays_staged_then_completion_promotes(self, tmp_path: Path):
        store = _jina_profile_store()
        cache = ArtifactCache(tmp_path)

        first = sync_artifacts(store, cache, names=["jina_faiss_index"], required=list(_JINA_PROFILE))
        assert first.promoted is False and cache.get_current() is None
        s = Settings(_env_file=None, cloud_assets_enabled=True,
                     cloud_assets_provider="s3_compatible", cloud_assets_cache_path=str(tmp_path))
        assets.reset_caches()
        assert assets.resolve_artifact_path("jina_faiss_index", settings=s) is None
        assets.reset_caches()

        second = sync_artifacts(store, cache, names=list(_JINA_PROFILE), required=list(_JINA_PROFILE))
        assert second.promoted is True and cache.get_current() == "jina-v9"


# ---------------------------------------------------------------------------
class TestSyncProgress:
    def _store(self):
        a, b = b"A" * 40, b"B" * 12  # 40 + 12 = 52 bytes total
        objs = {
            ("embeddings", "k/a"): a,
            ("embeddings", "k/b"): b,
        }
        arts = [
            {"name": "jina_faiss_index", "container": "embeddings", "key": "k/a", "size": len(a), "sha256": _sha(a)},
            {"name": "jina_global_ids", "container": "embeddings", "key": "k/b", "size": len(b), "sha256": _sha(b)},
        ]
        objs[("metadata", "hcmai-assets.json")] = _manifest("2026-jina", arts)
        return InMemoryStore(objs)

    def test_tracked_sync_reports_progress_and_final_state(self, tmp_path: Path):
        store = self._store()
        cache = ArtifactCache(tmp_path)
        report = run_tracked_sync(
            store, cache, ["jina_faiss_index", "jina_global_ids"], store.fetch_manifest(), trigger="manual"
        )
        assert report.promoted is True

        snap = get_sync_progress().to_dict()
        assert snap["state"] == "done"
        assert snap["trigger"] == "manual"
        assert snap["version"] == "2026-jina"
        assert snap["bytes_total"] == 52 and snap["bytes_done"] == 52
        assert snap["pct"] == 100.0
        by = {a["name"]: a for a in snap["artifacts"]}
        assert by["jina_faiss_index"]["status"] == "synced"
        assert by["jina_global_ids"]["status"] == "synced"

    def test_tracked_sync_surfaces_error_state(self, tmp_path: Path):
        store = self._store()
        doc = json.loads(store.objects[("metadata", "hcmai-assets.json")])
        doc["artifacts"][0]["sha256"] = _sha(b"WRONG")
        store.objects[("metadata", "hcmai-assets.json")] = json.dumps(doc).encode()
        cache = ArtifactCache(tmp_path)
        report = run_tracked_sync(
            store, cache, ["jina_faiss_index", "jina_global_ids"], store.fetch_manifest(), trigger="manual"
        )
        assert report.promoted is False
        snap = get_sync_progress().to_dict()
        assert snap["state"] == "done"  # the run itself completed; one artifact failed
        by = {a["name"]: a for a in snap["artifacts"]}
        assert by["jina_faiss_index"]["status"] == "error"
        assert by["jina_global_ids"]["status"] == "synced"

    def test_second_concurrent_tracked_sync_is_refused(self, tmp_path: Path, monkeypatch):
        store = self._store()
        cache = ArtifactCache(tmp_path)
        manifest = store.fetch_manifest()

        import src.services.assets.sync_state as ss

        released = {"v": False}
        real_begin = ss.SyncProgress.begin

        def slow_begin(self, *a, **k):
            real_begin(self, *a, **k)
            # while "inside" a run, a second call must be rejected
            try:
                run_tracked_sync(store, cache, ["jina_global_ids"], manifest, trigger="manual")
                released["v"] = "NOT-REFUSED"
            except RuntimeError:
                released["v"] = "refused"

        monkeypatch.setattr(ss.SyncProgress, "begin", slow_begin)
        run_tracked_sync(store, cache, ["jina_faiss_index"], manifest, trigger="manual")
        assert released["v"] == "refused"


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
class TestKeyframeRelPathPriority:
    """The resolver must prefer asset_key over frame_path, fetch the real JPG
    key for a Jina result, and never guess a BEiT3-style .webp filename for
    an item that actually carries a Jina asset_key."""

    def test_asset_key_wins_over_frame_path(self):
        item = {"asset_key": "L21/L21_V001/keyframe_0000.jpg", "frame_path": "L21/L21_V001/000000.webp"}
        assert assets._keyframe_rel_path(item) == "L21/L21_V001/keyframe_0000.jpg"

    def test_jina_result_resolves_to_its_own_jpg_key_not_a_beit3_guess(self):
        # Shaped like a Jina search result: has legacy-looking frame_id/video_id
        # fields that WOULD trigger the old .webp-guessing heuristic, but also
        # carries the real asset_key -- the guess must never win.
        item = {
            "asset_key": "L21/L21_V001/keyframe_0000.jpg",
            "video_id": "L21_V001",
            "frame_id": "0",
            "split": "L21",
            "retrieval_backend": "jina_clip_v2",
        }
        rel = assets._keyframe_rel_path(item)
        assert rel == "L21/L21_V001/keyframe_0000.jpg"
        assert not rel.endswith(".webp")

    def test_supports_jpg_jpeg_webp_png(self):
        for ext in ("jpg", "jpeg", "webp", "png"):
            item = {"asset_key": f"L21/L21_V001/keyframe_0000.{ext}"}
            assert assets._keyframe_rel_path(item).endswith(f".{ext}")

    def test_path_traversal_in_asset_key_is_rejected(self):
        item = {"asset_key": "../../etc/passwd"}
        assert assets._keyframe_rel_path(item) == ""

    def test_legacy_beit3_item_without_asset_key_keeps_old_heuristic(self):
        item = {"video_id": "L21_V001", "frame_id": "000010", "split": "L21"}
        assert assets._keyframe_rel_path(item) == "L21/L21_V001/000010.webp"

    def test_layout_fallback_only_used_when_no_asset_key_or_frame_path(self):
        item = {"video_id": "L21_V001", "split": "L21", "frame_id": "42"}
        layout = "{namespace}/{video_id}/{frame_id}.png"
        assert assets._keyframe_rel_path(item, layout=layout) == "L21/L21_V001/42.png"
        # asset_key still wins even with a layout supplied.
        item2 = dict(item, asset_key="L21/L21_V001/keyframe_0042.jpg")
        assert assets._keyframe_rel_path(item2, layout=layout) == "L21/L21_V001/keyframe_0042.jpg"

    def test_layout_rejects_unknown_placeholders(self):
        # An invalid layout (unknown placeholder) is ignored, not used -- the
        # resolver falls through to the next fallback instead of ever
        # formatting the untrusted template.
        item = {"video_id": "L21_V001", "split": "L21", "frame_id": "42"}
        malicious_layout = "{namespace}/../{video_id}/{unknown_field}"
        rel = assets._keyframe_rel_path(item, layout=malicious_layout)
        assert ".." not in rel.split("/")
        assert rel == "L21/L21_V001/42.webp"  # fell through to the legacy heuristic

    def test_layout_traversal_via_values_is_rejected(self):
        # Even with only whitelisted placeholders, a value that injects ".."
        # must not be allowed to escape the cache root.
        item = {"video_id": "../../escape", "split": "L21", "frame_id": "42"}
        layout = "{namespace}/{video_id}/{frame_id}.jpg"
        assert assets._layout_rel_path(item, layout) == ""

    def test_fetches_real_jpg_key_from_cloud_store(self, tmp_path: Path, monkeypatch):
        img = b"JPEGDATA"
        objs = {("keyframes", "L21/L21_V001/keyframe_0000.jpg"): img}
        store = InMemoryStore(objs)
        monkeypatch.setattr(assets, "build_asset_store", lambda settings=None: store)
        monkeypatch.setattr(assets, "get_manifest", lambda *a, **k: None)
        assets.reset_caches()

        s = Settings(
            _env_file=None, cloud_assets_enabled=True, cloud_assets_provider="s3_compatible",
            cloud_assets_cache_path=str(tmp_path), cloud_assets_keyframe_cache_max_bytes=10_000_000,
        )
        item = {
            "asset_key": "L21/L21_V001/keyframe_0000.jpg",
            "frame_path": "L21/L21_V001/keyframe_0000.jpg",
            "video_id": "L21_V001",
            "retrieval_backend": "jina_clip_v2",
        }
        resolved = assets.resolve_keyframe_file(item, settings=s)
        assert resolved is not None and resolved.read_bytes() == img
        assert store.reads == [("keyframes", "L21/L21_V001/keyframe_0000.jpg")]
        assets.reset_caches()


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


class TestResolveArtifactPath:
    def test_none_when_disabled(self):
        s = Settings(_env_file=None, cloud_assets_enabled=False)
        assert assets.resolve_artifact_path("faiss_index", settings=s) is None

    def test_returns_synced_path_for_current_version(self, tmp_path: Path):
        s = Settings(_env_file=None, cloud_assets_enabled=True,
                     cloud_assets_provider="s3_compatible", cloud_assets_cache_path=str(tmp_path))
        assets.reset_caches()
        cache = assets.get_artifact_cache(s)
        staged = cache.stage_path("faiss_index")
        staged.write_bytes(b"IDX")
        cache.promote(staged, "vA", "faiss_index", _sha(b"IDX"))
        # not current yet -> None
        assert assets.resolve_artifact_path("faiss_index", settings=s) is None
        cache.set_current("vA")
        resolved = assets.resolve_artifact_path("faiss_index", settings=s)
        assert resolved is not None and resolved.read_bytes() == b"IDX"
        assert assets.resolve_artifact_path("missing", settings=s) is None
        assets.reset_caches()
