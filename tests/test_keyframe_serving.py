"""Regression tests for GET /keyframes/{path} and the keyframe prefetch pool.

The bug this locks down: after the Jina + Azure switch, ``/keyframes/`` matched
a Jina "fine keyframe" path (``L30_a/L30_V093/keyframe_0016.jpg``) against the
*different* local extraction under ``KEYFRAMES_ROOT`` and, on the inevitable
miss, served ``000000.webp`` -- frame 0 of that video -- with HTTP 200. Every
result card showed the same intro frame; many were blank splashes.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from src.config.settings import Settings
from src.services.assets import keyframe_prefetch
from src.services.assets.local_cache import KeyframeCache

client = TestClient(main.app, raise_server_exceptions=False)


def _local_dataset(tmp_path: Path) -> Path:
    """A local extraction that DOES contain the video folder Jina asks about,
    with a frame-0 file -- exactly the shape that used to be mis-served."""
    vdir = tmp_path / "L30_a" / "L30_V093"
    vdir.mkdir(parents=True)
    (vdir / "000000.webp").write_bytes(b"LOCAL-FRAME-ZERO")
    (vdir / "000123.webp").write_bytes(b"LOCAL-FRAME-123")
    return tmp_path


# ---------------------------------------------------------------------------
# Cloud mode
# ---------------------------------------------------------------------------
class TestCloudMode:
    def test_serves_from_cloud_store_and_ignores_keyframes_root(self, tmp_path, monkeypatch):
        root = _local_dataset(tmp_path / "local")
        cloud_file = tmp_path / "cache" / "L30_a" / "L30_V093" / "keyframe_0016.jpg"
        cloud_file.parent.mkdir(parents=True)
        cloud_file.write_bytes(b"REAL-CLOUD-JPEG")

        monkeypatch.setattr(
            main, "settings",
            Settings(_env_file=None, cloud_assets_enabled=True,
                     cloud_assets_provider="azure_blob", keyframes_root=str(root)),
        )
        seen = {}

        def fake_fetch_blocking(path, *, settings=None):
            seen["path"] = path
            return cloud_file

        monkeypatch.setattr(keyframe_prefetch, "fetch_blocking", fake_fetch_blocking)

        resp = client.get("/keyframes/L30_a/L30_V093/keyframe_0016.jpg")

        assert resp.status_code == 200
        assert resp.content == b"REAL-CLOUD-JPEG"  # NOT the local 000000.webp
        assert seen["path"] == "L30_a/L30_V093/keyframe_0016.jpg"
        assert "immutable" in resp.headers.get("cache-control", "")
        assert resp.headers.get("etag")

    def test_cloud_miss_is_visible_not_a_permanent_200(self, tmp_path, monkeypatch):
        root = _local_dataset(tmp_path / "local")
        monkeypatch.setattr(
            main, "settings",
            Settings(_env_file=None, cloud_assets_enabled=True,
                     cloud_assets_provider="azure_blob", keyframes_root=str(root)),
        )
        monkeypatch.setattr(keyframe_prefetch, "fetch_blocking", lambda *a, **k: None)

        resp = client.get("/keyframes/L30_a/L30_V093/keyframe_0016.jpg")

        assert resp.status_code == 404
        assert resp.headers.get("cache-control") == "no-store"
        assert resp.headers.get("x-keyframe-status") == "missing"
        # never falls back to a local frame
        assert resp.content != b"LOCAL-FRAME-ZERO"


# ---------------------------------------------------------------------------
# Local-legacy mode
# ---------------------------------------------------------------------------
class TestLocalLegacyMode:
    def test_exact_hit_served_with_cache_headers(self, tmp_path, monkeypatch):
        root = _local_dataset(tmp_path)
        monkeypatch.setattr(
            main, "settings",
            Settings(_env_file=None, cloud_assets_enabled=False, keyframes_root=str(root)),
        )
        resp = client.get("/keyframes/L30_a/L30_V093/000123.webp")
        assert resp.status_code == 200
        assert resp.content == b"LOCAL-FRAME-123"
        assert "immutable" in resp.headers.get("cache-control", "")

    def test_miss_never_substitutes_another_frame(self, tmp_path, monkeypatch):
        root = _local_dataset(tmp_path)
        monkeypatch.setattr(
            main, "settings",
            Settings(_env_file=None, cloud_assets_enabled=False, keyframes_root=str(root)),
        )
        # video folder exists, requested frame does not -> must 404, not serve
        # 000000.webp / nearest-numeric / "first file in folder".
        resp = client.get("/keyframes/L30_a/L30_V093/keyframe_0016.jpg")
        assert resp.status_code == 404
        assert resp.headers.get("x-keyframe-status") == "missing"
        assert resp.content not in (b"LOCAL-FRAME-ZERO", b"LOCAL-FRAME-123")


# ---------------------------------------------------------------------------
# Prefetch pool
# ---------------------------------------------------------------------------
class TestKeyframePrefetch:
    @pytest.fixture(autouse=True)
    def _clean_pool(self):
        keyframe_prefetch.shutdown()
        yield
        keyframe_prefetch.shutdown()

    def _settings(self):
        return Settings(_env_file=None, cloud_assets_enabled=True,
                        cloud_assets_provider="azure_blob",
                        cloud_assets_keyframe_prefetch_workers=2)

    def test_prefetch_dedupes_inflight_and_fetch_blocking_reuses_future(self, monkeypatch):
        calls: list[str] = []
        release = {"go": False}

        def slow_resolve(path, *, settings=None):
            calls.append(path)
            while not release["go"]:
                time.sleep(0.01)
            return Path(f"/cache/{path}")

        monkeypatch.setattr(keyframe_prefetch._assets, "resolve_keyframe_file", slow_resolve)
        monkeypatch.setattr(keyframe_prefetch._assets, "get_keyframe_cache",
                            lambda settings=None: type("C", (), {"get": lambda self, k: None})())

        s = self._settings()
        # queue the same path three times across two prefetch calls
        keyframe_prefetch.prefetch(["a/b/c.jpg", "a/b/c.jpg"], settings=s)
        keyframe_prefetch.prefetch(["a/b/c.jpg"], settings=s)
        time.sleep(0.05)

        assert list(keyframe_prefetch._INFLIGHT) == ["a/b/c.jpg"]
        release["go"] = True
        got = keyframe_prefetch.fetch_blocking("a/b/c.jpg", settings=s)

        assert got == Path("/cache/a/b/c.jpg")
        assert calls == ["a/b/c.jpg"]  # exactly one real download, not one per request

    def test_noop_when_cloud_disabled(self, monkeypatch):
        s = Settings(_env_file=None, cloud_assets_enabled=False)
        monkeypatch.setattr(keyframe_prefetch._assets, "resolve_keyframe_file",
                            lambda *a, **k: pytest.fail("should not download"))
        keyframe_prefetch.prefetch(["x/y/z.jpg"], settings=s)
        assert keyframe_prefetch.fetch_blocking("x/y/z.jpg", settings=s) is None
        assert not keyframe_prefetch._INFLIGHT


# ---------------------------------------------------------------------------
# KeyframeCache: cheap cache-hit path + running-total eviction
# ---------------------------------------------------------------------------
class TestKeyframeCacheHotPath:
    def test_get_does_not_write_index_on_every_hit(self, tmp_path):
        kf = KeyframeCache(tmp_path, max_bytes=0)
        kf.put("a.webp", b"0123456789")
        index_path = kf.root / ".index.json"
        mtime_after_put = index_path.stat().st_mtime_ns

        for _ in range(5):
            assert kf.get("a.webp") is not None
        # debounce window not elapsed -> no rewrite on the hit path
        assert index_path.stat().st_mtime_ns == mtime_after_put

        kf.flush()  # shutdown persists the in-memory atime
        assert index_path.stat().st_mtime_ns >= mtime_after_put

    def test_eviction_tracks_running_total(self, tmp_path):
        kf = KeyframeCache(tmp_path, max_bytes=25)
        kf.put("a.webp", b"0123456789")   # 10
        kf.get("a.webp")
        kf.put("b.webp", b"0123456789")   # 20
        kf.put("c.webp", b"0123456789")   # 30 -> over cap, evict LRU (a)
        assert kf.get("a.webp") is None
        assert kf.get("b.webp") is not None
        assert kf.get("c.webp") is not None
        assert kf._total_bytes == 20
        assert kf.stats()["usage_bytes"] == 20
