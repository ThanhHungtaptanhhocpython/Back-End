"""Startup warm-up: background artifact sync + retriever warm (feature A)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.config.settings import Settings  # noqa: E402
from src.services import startup_warm  # noqa: E402
from src.services.assets.local_cache import ArtifactCache  # noqa: E402
from tests.test_cloud_assets import InMemoryStore, _manifest, _sha  # noqa: E402


def _jina_store():
    idx, gid, meta = b"I" * 20, b"G" * 10, b"{}"
    objs = {
        ("embeddings", "k/idx"): idx,
        ("embeddings", "k/gid"): gid,
        ("embeddings", "k/meta"): meta,
        ("embeddings", "k/vmeta"): b"V" * 4,
    }
    arts = [
        {"name": "jina_faiss_index", "container": "embeddings", "key": "k/idx", "size": len(idx), "sha256": _sha(idx)},
        {"name": "jina_global_ids", "container": "embeddings", "key": "k/gid", "size": len(gid), "sha256": _sha(gid)},
        {"name": "jina_video_metadata", "container": "embeddings", "key": "k/vmeta", "size": 4, "sha256": _sha(b"VVVV")},
        {"name": "jina_index_meta", "container": "embeddings", "key": "k/meta", "size": len(meta), "sha256": _sha(meta)},
    ]
    objs[("metadata", "hcmai-assets.json")] = _manifest("jina-v1", arts)
    return InMemoryStore(objs)


class TestAutosyncGate:
    def test_pytest_guard_makes_it_a_noop(self, monkeypatch):
        # PYTEST_CURRENT_TEST is set during any test -> never spawns a thread.
        spawned = {"n": 0}
        monkeypatch.setattr(
            startup_warm.threading, "Thread",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spawn under pytest")),
        )
        startup_warm._STARTED.clear()
        startup_warm.warm_active_backend_in_background(Settings(_env_file=None, cloud_assets_autosync=True))
        assert spawned["n"] == 0

    def test_disabled_when_autosync_off(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)  # auto-restored by monkeypatch
        made = {}

        class _FakeThread:
            def __init__(self, *a, **k):
                made["target"] = k.get("target")

            def start(self):
                made["started"] = True

        monkeypatch.setattr(startup_warm.threading, "Thread", _FakeThread)
        startup_warm._STARTED.clear()
        startup_warm.warm_active_backend_in_background(Settings(_env_file=None, cloud_assets_autosync=False))
        assert "started" not in made

    def test_spawns_once_when_enabled(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        starts = {"n": 0}

        class _FakeThread:
            def __init__(self, *a, **k):
                pass

            def start(self):
                starts["n"] += 1

        monkeypatch.setattr(startup_warm.threading, "Thread", _FakeThread)
        startup_warm._STARTED.clear()
        s = Settings(_env_file=None, cloud_assets_autosync=True)
        startup_warm.warm_active_backend_in_background(s)
        startup_warm.warm_active_backend_in_background(s)  # idempotent
        assert starts["n"] == 1


class TestSyncActiveBackendArtifacts:
    def _settings(self, tmp_path, **over):
        base = dict(
            _env_file=None,
            cloud_assets_enabled=True,
            cloud_assets_provider="azure_blob",
            azure_storage_connection_string="cs",
            cloud_assets_cache_path=str(tmp_path),
            retrieval_backend="jina_clip_v2",
        )
        base.update(over)
        return Settings(**base)

    def test_syncs_only_jina_artifacts_and_promotes(self, tmp_path: Path, monkeypatch):
        from src.services import assets as assets_mod

        store = _jina_store()
        monkeypatch.setattr(assets_mod.factory, "build_asset_store", lambda settings=None: store)
        assets_mod.reset_caches()

        startup_warm._sync_active_backend_artifacts(self._settings(tmp_path))

        assert store.reads and all(c == "embeddings" for c, _k in store.reads)
        cache = ArtifactCache(tmp_path)
        assert cache.get_current() == "jina-v1"
        assets_mod.reset_caches()

    def test_noop_when_already_current(self, tmp_path: Path, monkeypatch):
        from src.services import assets as assets_mod

        store = _jina_store()
        monkeypatch.setattr(assets_mod.factory, "build_asset_store", lambda settings=None: store)
        assets_mod.reset_caches()
        s = self._settings(tmp_path)
        startup_warm._sync_active_backend_artifacts(s)
        first = len(store.reads)
        startup_warm._sync_active_backend_artifacts(s)  # second call: nothing new to fetch
        assert len(store.reads) == first
        assets_mod.reset_caches()

    def test_noop_when_cloud_disabled(self, tmp_path: Path):
        startup_warm._sync_active_backend_artifacts(
            Settings(_env_file=None, cloud_assets_enabled=False, retrieval_backend="jina_clip_v2")
        )  # must simply return, no error

    def test_noop_when_no_manifest(self, tmp_path: Path, monkeypatch):
        from src.services import assets as assets_mod

        store = _jina_store()
        monkeypatch.setattr(assets_mod.factory, "build_asset_store", lambda settings=None: store)
        monkeypatch.setattr(assets_mod.factory, "get_manifest", lambda *a, **k: None)
        assets_mod.reset_caches()
        startup_warm._sync_active_backend_artifacts(self._settings(tmp_path))  # returns cleanly
        assets_mod.reset_caches()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
