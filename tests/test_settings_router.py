"""Phase 2 -- local management API (loopback guard, config CRUD, revisions)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import main  # noqa: E402
from src.config import app_paths  # noqa: E402
from src.config import runtime_store as rs  # noqa: E402
from src.config.settings import get_settings  # noqa: E402

LOCAL = ("127.0.0.1", 51888)


def _client(client_addr=LOCAL) -> TestClient:
    return TestClient(main.app, client=client_addr)


# ---------------------------------------------------------------------------
# loopback guard
# ---------------------------------------------------------------------------
class TestLocalGuard:
    def test_remote_client_forbidden(self) -> None:
        r = _client(("203.0.113.9", 4444)).get("/settings/schema")
        assert r.status_code == 403

    def test_cross_origin_forbidden(self) -> None:
        r = _client().get("/settings/schema", headers={"origin": "http://evil.example"})
        assert r.status_code == 403

    def test_localhost_origin_allowed(self) -> None:
        r = _client().get("/settings/schema", headers={"origin": "http://localhost:5173"})
        assert r.status_code == 200

    def test_users_prefix_also_mounted(self) -> None:
        assert _client().get("/users/settings/schema").status_code == 200


# ---------------------------------------------------------------------------
# schema + read-only config (store disabled by conftest)
# ---------------------------------------------------------------------------
class TestSchemaAndConfig:
    def test_schema_has_ordered_groups(self) -> None:
        body = _client().get("/settings/schema").json()
        groups = [g["group"] for g in body["groups"]]
        assert groups[0] == "Server"
        assert "AI" in groups and "Cloud Assets" in groups
        assert body["store"]["enabled"] is False

    def test_config_masks_secrets(self) -> None:
        body = _client().get("/settings/config").json()
        assert set(body["secrets"]) >= {"OPENROUTER_API_KEY", "NIM_API_KEY"}
        assert all(isinstance(v, bool) for v in body["secrets"].values())
        for key, value in body["values"].items():
            assert value != "None"
        # a secret value is never returned in clear
        assert body["values"]["OPENROUTER_API_KEY"] in ("", "********")

    def test_config_reports_resolved_default_paths(self) -> None:
        body = _client().get("/settings/config").json()
        resolved = body["resolved"]
        assert "MEDIA_INFO_PATH" in resolved and "MAP_KEYFRAMES_PATH" in resolved
        entry = resolved["KEYFRAMES_ROOT"]
        assert set(entry) == {"path", "exists", "is_default"}
        assert isinstance(entry["exists"], bool)

    def test_validate_flags_bad_values(self) -> None:
        r = _client().post(
            "/settings/validate",
            json={"values": {"PORT": "99999", "HOST": "127.0.0.1", "BEIT3_DEVICE": "tpu"}},
        )
        body = r.json()
        assert body["ok"] is False
        assert "PORT" in body["errors"] and "BEIT3_DEVICE" in body["errors"]
        assert body["normalized"]["HOST"] == "127.0.0.1"

    def test_validate_rejects_secret_sent_as_value(self) -> None:
        body = _client().post(
            "/settings/validate", json={"values": {"OPENROUTER_API_KEY": "sk-x"}}
        ).json()
        assert body["ok"] is False
        assert "OPENROUTER_API_KEY" in body["errors"]

    def test_update_conflicts_when_store_disabled(self) -> None:
        r = _client().post("/settings/config", json={"values": {"PORT": "3001"}})
        assert r.status_code == 409

    def test_fs_browser_roots_and_listing(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
        (tmp_path / "b.zip").write_bytes(b"PK\x03\x04")
        client = _client()

        roots = client.get("/settings/fs").json()
        assert roots["roots"] and roots["path"] == ""
        assert roots["home"] and "sep" in roots

        listing = client.get("/settings/fs", params={"path": str(tmp_path)}).json()
        names = [e["name"] for e in listing["entries"]]
        assert names[0] == "sub"  # dirs sorted first
        assert {"a.txt", "b.zip"} <= set(names)
        assert listing["parent"] == str(tmp_path.parent)

    def test_fs_browser_dirs_only_and_file_to_parent(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        f = tmp_path / "a.txt"
        f.write_text("x", encoding="utf-8")
        client = _client()

        only = client.get("/settings/fs", params={"path": str(tmp_path), "dirs_only": "1"}).json()
        assert [e["name"] for e in only["entries"]] == ["sub"]

        # a file path resolves to its parent directory listing
        viafile = client.get("/settings/fs", params={"path": str(f)}).json()
        assert viafile["path"] == str(tmp_path)

    def test_fs_browser_missing_path_404(self) -> None:
        assert _client().get("/settings/fs", params={"path": "/no/such/dir/xyz123"}).status_code == 404

    def test_restart_status_without_launcher(self) -> None:
        body = _client().get("/settings/restart/status").json()
        assert body["launcher_running"] is False
        post = _client().post("/settings/restart", json={"reason": "manual"}).json()
        assert post["ok"] is False and post["detail"] == "launcher_not_running"


# ---------------------------------------------------------------------------
# writes -- store enabled and isolated
# ---------------------------------------------------------------------------
class TestConfigWrites:
    @pytest.fixture()
    def enabled_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(tmp_path / "appdata"))
        monkeypatch.delenv("HCMAI_DISABLE_CONFIG_STORE", raising=False)
        app_paths.get_app_data_dir.cache_clear()
        rs.reset_store()
        get_settings.cache_clear()
        store = rs.get_store()
        assert store is not None
        store.bootstrap_from_env({"PORT": "3000", "HOST": "0.0.0.0", "LOG_LEVEL": "INFO"})
        get_settings.cache_clear()
        yield store
        rs.reset_store()
        app_paths.get_app_data_dir.cache_clear()
        get_settings.cache_clear()

    def test_update_persists_and_applies(self, enabled_store) -> None:
        client = _client()
        r = client.post(
            "/settings/config",
            json={
                "values": {"PORT": "4321", "AI_GATEWAY_ENABLED": "true"},
                "secret_set": {"OPENROUTER_API_KEY": "sk-or-topsecret"},
                "note": "test change",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True and body["restart_required"] is True
        rev_id = body["revision_id"]

        cfg = client.get("/settings/config").json()
        assert cfg["values"]["PORT"] == "4321"
        assert cfg["values"]["AI_GATEWAY_ENABLED"] == "true"
        assert cfg["values"]["OPENROUTER_API_KEY"] == "********"
        assert cfg["secrets"]["OPENROUTER_API_KEY"] is True

        # secret plaintext must not leak through the revision detail endpoint
        detail = client.get(f"/settings/revisions/{rev_id}")
        assert "sk-or-topsecret" not in detail.text
        assert detail.json()["secrets"]["OPENROUTER_API_KEY"] is True

    def test_secret_pasted_with_env_quotes_is_stored_unquoted(self, enabled_store) -> None:
        client = _client()
        r = client.post(
            "/settings/config",
            json={"values": {}, "secret_set": {"AZURE_STORAGE_CONNECTION_STRING": '"Endp=1;X=2"'}},
        )
        assert r.status_code == 200, r.text
        rev = r.json()["revision_id"]
        stored = enabled_store.effective_values()["AZURE_STORAGE_CONNECTION_STRING"]
        assert stored == "Endp=1;X=2"  # leading/trailing quotes stripped
        assert "Endp=1;X=2" not in client.get(f"/settings/revisions/{rev}").text  # not leaked
        assert client.get("/settings/config").json()["secrets"]["AZURE_STORAGE_CONNECTION_STRING"] is True

    def test_invalid_update_is_rejected_with_400(self, enabled_store) -> None:
        r = _client().post("/settings/config", json={"values": {"PORT": "-5"}})
        assert r.status_code == 400
        assert r.json()["ok"] is False and "PORT" in r.json()["errors"]
        # nothing saved
        assert _client().get("/settings/config").json()["values"]["PORT"] == "3000"

    def test_revision_history_and_restore(self, enabled_store) -> None:
        client = _client()
        client.post("/settings/config", json={"values": {"PORT": "5000"}})
        client.post("/settings/config", json={"values": {"PORT": "6000"}})
        revs = client.get("/settings/revisions").json()["revisions"]
        assert len(revs) >= 3 and revs[0]["active"] is True

        # restore the oldest listed revision (the env-import, PORT 3000)
        oldest = revs[-1]["id"]
        restored = client.post(f"/settings/revisions/{oldest}/restore").json()
        assert restored["ok"] is True
        assert client.get("/settings/config").json()["values"]["PORT"] == "3000"

    def test_restore_missing_revision_404(self, enabled_store) -> None:
        assert _client().post("/settings/revisions/99999/restore").status_code == 404


# ---------------------------------------------------------------------------
# AI provider endpoints
# ---------------------------------------------------------------------------
class TestProviderEndpoints:
    @pytest.fixture()
    def gateway_settings(self, monkeypatch: pytest.MonkeyPatch):
        from src.api.routers import settings_router
        from src.config.settings import Settings
        from src.services.ai import openai_compatible

        s = Settings(
            _env_file=None,
            ai_gateway_enabled=True,
            groq_enabled=True, groq_api_key="k-groq", groq_text_model="groq-t", groq_vision_model="groq-v",
            nim_enabled=True, nim_api_key="k-nim", nim_text_model="nim-t",
            cloudflare_enabled=True, cloudflare_api_key="k-cf", cloudflare_text_model="cf-t",
            ai_text_priority="groq,nim", ai_vision_priority="groq",
        )
        monkeypatch.setattr(settings_router, "get_settings", lambda: s)

        class _T:
            def __init__(self):
                self.calls = []

            def request(self, method, url, headers, body, timeout):
                self.calls.append((method, url))
                import json as _j
                if url.endswith("/models"):
                    return 200, _j.dumps({"data": [{"id": "z"}, {"id": "a"}]})
                return 200, _j.dumps({"choices": [{"message": {"content": "pong"}}]})

        prev = openai_compatible._TRANSPORT
        openai_compatible.set_transport(_T())
        yield s
        openai_compatible.set_transport(prev)

    def test_list_providers_reports_chains(self, gateway_settings) -> None:
        body = _client().get("/settings/providers").json()
        assert body["gateway_enabled"] is True
        assert body["text_chain"] == ["groq", "nim"]
        assert body["vision_chain"] == ["groq"]
        cf = next(p for p in body["providers"] if p["id"] == "cloudflare")
        assert cf["configured"] is False and cf["missing_requirements"] == ["CLOUDFLARE_ACCOUNT_ID"]

    def test_provider_test_text_ok(self, gateway_settings) -> None:
        body = _client().post("/settings/providers/groq/test", json={"mode": "text"}).json()
        assert body["ok"] is True and body["model"] == "groq-t" and body["sample"] == "pong"

    def test_provider_test_vision_without_model(self, gateway_settings) -> None:
        body = _client().post("/settings/providers/nim/test", json={"mode": "vision"}).json()
        assert body["ok"] is False and body["category"] == "model_unavailable"

    def test_provider_test_unconfigured(self, gateway_settings) -> None:
        body = _client().post("/settings/providers/cloudflare/test", json={"mode": "text"}).json()
        assert body["ok"] is False and body["category"] == "not_configured"

    def test_models_discovery(self, gateway_settings) -> None:
        body = _client().get("/settings/providers/groq/models").json()
        assert body["ok"] is True and body["models"] == ["a", "z"]

    def test_unknown_provider_404(self, gateway_settings) -> None:
        assert _client().post("/settings/providers/nope/test").status_code == 404


# ---------------------------------------------------------------------------
# Cloud asset endpoints
# ---------------------------------------------------------------------------
class TestCloudEndpoints:
    def test_status_reports_disabled_by_default(self) -> None:
        body = _client().get("/settings/cloud/status").json()
        assert body["enabled"] is False and body["active"] is False
        assert "azure_blob" in body["sdk"] and "s3_compatible" in body["sdk"]

    def test_test_and_manifest_conflict_when_local(self) -> None:
        assert _client().post("/settings/cloud/test").json()["ok"] is False
        assert _client().get("/settings/cloud/manifest").status_code == 409

    def test_sync_status_endpoint_shape(self) -> None:
        body = _client().get("/settings/cloud/sync/status").json()
        # always answerable, even before any sync has run
        assert body["state"] in ("idle", "running", "done", "error")
        assert "artifacts" in body and "pct" in body and "bytes_total" in body
        assert "had_errors" in body and "ok" in body

    def test_retrieval_status_reports_the_default_jina_backend(self) -> None:
        # Default backend is now jina_clip_v2; with no Jina index configured it
        # reports not-ready (preparing/error) rather than a bare 200.
        body = _client().get("/settings/retrieval/status").json()
        assert body["backend"] == "jina_clip_v2"
        assert body["ready"] is not True

    def test_retrieval_status_ready_for_an_explicit_local_beit3_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.api.routers import settings_router
        from src.config.settings import Settings

        s = Settings(_env_file=None, retrieval_backend="beit3", cloud_assets_enabled=False)
        monkeypatch.setattr(settings_router, "get_settings", lambda: s)
        body = _client().get("/settings/retrieval/status").json()
        assert body["backend"] == "beit3"
        assert body["ready"] is True

    def test_jina_readiness_reports_the_three_checks(self) -> None:
        body = _client().get("/settings/jina/readiness").json()
        assert set(body) >= {"ok", "active_backend", "checks"}
        ids = [c["id"] for c in body["checks"]]
        assert ids == ["gpu", "model", "index"]
        for c in body["checks"]:
            assert c["status"] in {"ok", "warn", "miss"}
            assert c["summary"]
        # ``ok`` is true only when no check is a hard MISS.
        assert body["ok"] == all(c["status"] != "miss" for c in body["checks"])

    def test_jina_readiness_gpu_check_reflects_torch_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        gpu = next(
            c for c in _client().get("/settings/jina/readiness").json()["checks"] if c["id"] == "gpu"
        )
        assert gpu["resolved_device"] == "cpu"
        assert gpu["status"] in {"warn", "miss"}
        if gpu["status"] == "warn":
            assert "download.pytorch.org/whl" in gpu["fix"]

    @staticmethod
    def _full_beit3_store(version: str = "vX"):
        import hashlib
        import json as _json

        from src.services.assets.base import BACKEND_ARTIFACT_NAMES
        from tests.test_cloud_assets import InMemoryStore

        arts, objs = [], {}
        for name in BACKEND_ARTIFACT_NAMES["beit3"]:
            blob = f"BEIT3-{name}-bytes".encode()
            key = f"beit3/{name}"
            objs[("embeddings", key)] = blob
            arts.append({"name": name, "container": "embeddings", "key": key,
                         "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
        objs[("metadata", "hcmai-assets.json")] = _json.dumps({"version": version, "artifacts": arts}).encode()
        return InMemoryStore(objs)

    def test_sync_and_manifest_with_fake_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.api.routers import settings_router
        from src.config.settings import Settings
        from src.services import assets as assets_mod

        store = self._full_beit3_store("vX")
        s = Settings(_env_file=None, cloud_assets_enabled=True, cloud_assets_provider="s3_compatible",
                     cloud_assets_cache_path=str(tmp_path))
        monkeypatch.setattr(settings_router, "get_settings", lambda: s)
        # Pin the active backend so this test exercises the BEiT3 sync-profile
        # machinery (all-or-nothing promotion) rather than the cloud->jina policy.
        monkeypatch.setattr("src.services.retrieval_backend.active_backend", lambda settings=None: "beit3")
        monkeypatch.setattr(assets_mod, "build_asset_store", lambda settings=None: store)
        assets_mod.reset_caches()

        client = _client()
        manifest = client.get("/settings/cloud/manifest").json()
        assert manifest["version"] == "vX"
        assert manifest["artifacts"][0]["cached"] is False

        report = client.post("/settings/cloud/sync", json={}).json()
        assert report["ok"] is True and report["promoted"] is True
        assert report["current_version"] == "vX"

        after = client.get("/settings/cloud/manifest").json()
        assert all(a["cached"] and a["verified"] for a in after["artifacts"])

        cleared = client.post("/settings/cloud/cache/clear", json={"scope": "all"}).json()
        assert cleared["ok"] is True
        assets_mod.reset_caches()

    def test_active_profile_sync_rejects_a_manifest_missing_a_profile_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest that declares only part of the active backend's profile is
        a broken publish: POST /settings/cloud/sync must 422, not promote."""
        import json as _json

        from src.api.routers import settings_router
        from src.config.settings import Settings
        from src.services import assets as assets_mod
        from tests.test_cloud_assets import InMemoryStore

        store = self._full_beit3_store("vY")
        # drop `tokenizer` from the published manifest (blob still there)
        doc = _json.loads(store.objects[("metadata", "hcmai-assets.json")])
        doc["artifacts"] = [a for a in doc["artifacts"] if a["name"] != "tokenizer"]
        store.objects[("metadata", "hcmai-assets.json")] = _json.dumps(doc).encode()

        s = Settings(_env_file=None, cloud_assets_enabled=True, cloud_assets_provider="s3_compatible",
                     cloud_assets_cache_path=str(tmp_path))
        monkeypatch.setattr(settings_router, "get_settings", lambda: s)
        monkeypatch.setattr("src.services.retrieval_backend.active_backend", lambda settings=None: "beit3")
        monkeypatch.setattr(assets_mod, "build_asset_store", lambda settings=None: store)
        assets_mod.reset_caches()

        resp = _client().post("/settings/cloud/sync", json={})
        assert resp.status_code == 422
        assert "tokenizer" in resp.json()["detail"]
        assert assets_mod.get_artifact_cache(s).get_current() is None
        assert store.reads == []  # nothing was downloaded
        assets_mod.reset_caches()

    def test_manual_subset_sync_stages_without_promoting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.api.routers import settings_router
        from src.config.settings import Settings
        from src.services import assets as assets_mod

        store = self._full_beit3_store("vZ")
        s = Settings(_env_file=None, cloud_assets_enabled=True, cloud_assets_provider="s3_compatible",
                     cloud_assets_cache_path=str(tmp_path))
        monkeypatch.setattr(settings_router, "get_settings", lambda: s)
        monkeypatch.setattr(assets_mod, "build_asset_store", lambda settings=None: store)
        assets_mod.reset_caches()

        report = _client().post("/settings/cloud/sync", json={"names": ["faiss_index"]}).json()
        assert [a["name"] for a in report["artifacts"]] == ["faiss_index"]
        assert report["promoted"] is False
        assert report["current_version"] is None
        assets_mod.reset_caches()
