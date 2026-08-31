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
