"""Phase 1 -- SQLite runtime configuration store.

Covers: .env -> SQLite migration, active-revision precedence, secret
encryption / masking, schema+range validation, revision retention, and
restore.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.config import app_paths, field_spec  # noqa: E402
from src.config.runtime_store import MAX_REVISIONS, RuntimeConfigStore  # noqa: E402
from src.config.secret_box import SecretBox, SecretBoxError  # noqa: E402


# ---------------------------------------------------------------------------
# SecretBox
# ---------------------------------------------------------------------------
class TestSecretBox:
    def test_round_trip(self) -> None:
        box = SecretBox.from_key(b"0" * 32)
        token = box.encrypt("sk-super-secret-value")
        assert token != "sk-super-secret-value"
        assert box.decrypt(token) == "sk-super-secret-value"

    def test_unicode_round_trip(self) -> None:
        box = SecretBox.from_key(b"k" * 32)
        assert box.decrypt(box.encrypt("khóa bí mật 🔐")) == "khóa bí mật 🔐"

    def test_distinct_ciphertexts_per_call(self) -> None:
        box = SecretBox.from_key(b"n" * 32)
        assert box.encrypt("same") != box.encrypt("same")

    def test_tamper_is_detected(self) -> None:
        box = SecretBox.from_key(b"z" * 32)
        token = bytearray(__import__("base64").b64decode(box.encrypt("payload")))
        token[-1] ^= 0x01
        tampered = __import__("base64").b64encode(bytes(token)).decode()
        with pytest.raises(SecretBoxError):
            box.decrypt(tampered)

    def test_wrong_key_fails(self) -> None:
        good = SecretBox.from_key(b"a" * 32)
        bad = SecretBox.from_key(b"b" * 32)
        with pytest.raises(SecretBoxError):
            bad.decrypt(good.encrypt("hello"))


# ---------------------------------------------------------------------------
# field_spec validation
# ---------------------------------------------------------------------------
class TestFieldSpecValidation:
    def test_bool_normalisation(self) -> None:
        spec = field_spec.by_key("DEBUG")
        assert field_spec.validate_value(spec, "yes") == "true"
        assert field_spec.validate_value(spec, "0") == "false"

    def test_int_range_enforced(self) -> None:
        spec = field_spec.by_key("PORT")
        assert field_spec.validate_value(spec, "8080") == "8080"
        with pytest.raises(field_spec.ValidationError):
            field_spec.validate_value(spec, "70000")
        with pytest.raises(field_spec.ValidationError):
            field_spec.validate_value(spec, "not-a-number")

    def test_choice_enforced(self) -> None:
        spec = field_spec.by_key("BEIT3_DEVICE")
        assert field_spec.validate_value(spec, "cpu") == "cpu"
        with pytest.raises(field_spec.ValidationError):
            field_spec.validate_value(spec, "tpu")

    def test_url_kind(self) -> None:
        spec = field_spec.by_key("ELASTICSEARCH_URL")
        assert field_spec.validate_value(spec, "http://localhost:9200")
        with pytest.raises(field_spec.ValidationError):
            field_spec.validate_value(spec, "localhost:9200")

    def test_json_object_kind(self) -> None:
        spec = field_spec.by_key("PLAYBACK_OFFSETS_JSON")
        assert field_spec.validate_value(spec, '{"L21_V029": -172}') == '{"L21_V029":-172}'
        with pytest.raises(field_spec.ValidationError):
            field_spec.validate_value(spec, "[1, 2, 3]")

    def test_blank_always_allowed(self) -> None:
        spec = field_spec.by_key("PORT")
        assert field_spec.validate_value(spec, "") == ""

    def test_src_dir_is_locked(self) -> None:
        assert "SRC_DIR" in field_spec.locked_keys()

    def test_registry_field_names_match_settings(self) -> None:
        from src.config.settings import Settings

        model_fields = set(Settings.model_fields)
        # SRC_DIR maps to src_dir; every other spec must resolve to a real field.
        missing = [s.key for s in field_spec.all_specs() if s.field not in model_fields]
        assert missing == [], f"specs without a Settings field: {missing}"


# ---------------------------------------------------------------------------
# RuntimeConfigStore
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path: Path) -> RuntimeConfigStore:
    box = SecretBox.from_key(b"t" * 32)
    return RuntimeConfigStore(tmp_path / "config.db", box)


ENV_SAMPLE = {
    "HOST": "127.0.0.1",
    "PORT": "8123",
    "LOG_LEVEL": "DEBUG",
    "OPENROUTER_API_KEY": "sk-or-secret-abc",
    "NIM_API_KEY": "nvapi-secret-xyz",
    "SRC_DIR": "/should/be/ignored",
}


class TestBootstrap:
    def test_creates_revision_one_from_env(self, store: RuntimeConfigStore) -> None:
        rev_id = store.bootstrap_from_env(ENV_SAMPLE)
        assert rev_id == store.active_revision_id()
        values = store.effective_values()
        assert values["HOST"] == "127.0.0.1"
        assert values["PORT"] == "8123"
        # secret decrypts back to the original
        assert values["OPENROUTER_API_KEY"] == "sk-or-secret-abc"
        # locked key never enters the store
        assert "SRC_DIR" not in values

    def test_secret_is_encrypted_at_rest(self, store: RuntimeConfigStore) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)
        rev = store.active_revision_id()
        rows = {r["key"]: r for r in store._rows_for(rev)}  # noqa: SLF001
        stored = rows["OPENROUTER_API_KEY"]
        assert stored["is_secret"] == 1
        assert "sk-or-secret-abc" not in stored["value"]

    def test_is_idempotent(self, store: RuntimeConfigStore) -> None:
        first = store.bootstrap_from_env(ENV_SAMPLE)
        second = store.bootstrap_from_env({"HOST": "0.0.0.0"})
        assert first == second
        assert store.effective_values()["HOST"] == "127.0.0.1"

    def test_secret_status_is_boolean_only(self, store: RuntimeConfigStore) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)
        status = store.secret_status()
        assert status["OPENROUTER_API_KEY"] is True
        assert status["ANTHROPIC_API_KEY"] is False
        assert all(isinstance(v, bool) for v in status.values())


class TestRevisions:
    def test_create_revision_becomes_active_and_carries_secrets(
        self, store: RuntimeConfigStore
    ) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)
        new_id = store.create_revision(
            {"HOST": "0.0.0.0", "PORT": "3000"},
            source="ui",
            note="change host",
        )
        assert store.active_revision_id() == new_id
        values = store.effective_values()
        assert values["HOST"] == "0.0.0.0"
        # secret not mentioned -> carried forward
        assert values["OPENROUTER_API_KEY"] == "sk-or-secret-abc"

    def test_secret_set_and_clear(self, store: RuntimeConfigStore) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)
        store.create_revision(
            {"HOST": "0.0.0.0"},
            source="ui",
            secret_set={"ANTHROPIC_API_KEY": "sk-ant-new"},
            secret_clear=["OPENROUTER_API_KEY"],
        )
        values = store.effective_values()
        assert values["ANTHROPIC_API_KEY"] == "sk-ant-new"
        assert "OPENROUTER_API_KEY" not in values
        assert store.secret_status()["OPENROUTER_API_KEY"] is False

    def test_blank_secret_keeps_existing(self, store: RuntimeConfigStore) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)
        store.create_revision(
            {"HOST": "0.0.0.0"}, source="ui", secret_set={"OPENROUTER_API_KEY": ""}
        )
        assert store.effective_values()["OPENROUTER_API_KEY"] == "sk-or-secret-abc"

    def test_masked_view_never_exposes_secret(self, store: RuntimeConfigStore) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)
        masked = store.revision_values_masked(store.active_revision_id())
        assert "OPENROUTER_API_KEY" not in masked["values"]
        assert masked["secrets"]["OPENROUTER_API_KEY"] is True
        assert "sk-or-secret-abc" not in repr(masked)

    def test_retention_keeps_latest_ten(self, store: RuntimeConfigStore) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)
        for i in range(15):
            store.create_revision({"PORT": str(4000 + i)}, source="ui", note=f"r{i}")
        revs = store.list_revisions(limit=100)
        assert len(revs) == MAX_REVISIONS
        # newest is active and has the last port we wrote
        assert revs[0]["active"] is True
        assert store.effective_values()["PORT"] == "4014"

    def test_restore_creates_new_active_revision(self, store: RuntimeConfigStore) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)  # rev 1: PORT 8123
        r2 = store.create_revision({"PORT": "9999"}, source="ui")
        restored = store.restore_revision(1, note="rollback")
        assert restored != 1 and restored != r2
        assert store.active_revision_id() == restored
        values = store.effective_values()
        assert values["PORT"] == "8123"
        # secret survived the restore round-trip
        assert values["OPENROUTER_API_KEY"] == "sk-or-secret-abc"

    def test_restore_unknown_revision_raises(self, store: RuntimeConfigStore) -> None:
        store.bootstrap_from_env(ENV_SAMPLE)
        with pytest.raises(KeyError):
            store.restore_revision(999)


# ---------------------------------------------------------------------------
# get_settings() precedence
# ---------------------------------------------------------------------------
class TestSettingsPrecedence:
    @pytest.fixture()
    def isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import src.config.settings as settings_mod
        from src.config import runtime_store

        monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(tmp_path / "appdata"))
        monkeypatch.delenv("HCMAI_DISABLE_CONFIG_STORE", raising=False)
        app_paths.get_app_data_dir.cache_clear()
        runtime_store.reset_store()
        settings_mod.get_settings.cache_clear()
        yield settings_mod
        runtime_store.reset_store()
        app_paths.get_app_data_dir.cache_clear()
        settings_mod.get_settings.cache_clear()

    def test_active_revision_overrides_defaults(self, isolated, monkeypatch) -> None:
        from src.config import runtime_store

        store = runtime_store.get_store()
        assert store is not None
        store.bootstrap_from_env({"PORT": "3000", "HOST": "0.0.0.0"})
        store.create_revision(
            {"PORT": "4567", "HOST": "127.0.0.1", "AI_GATEWAY_ENABLED": "true"},
            source="ui",
        )
        isolated.get_settings.cache_clear()
        settings = isolated.get_settings()
        assert settings.port == 4567
        assert settings.host == "127.0.0.1"
        assert settings.ai_gateway_enabled is True

    def test_disabled_store_is_pure_env(self, isolated, monkeypatch) -> None:
        monkeypatch.setenv("HCMAI_DISABLE_CONFIG_STORE", "1")
        from src.config import runtime_store

        runtime_store.reset_store()
        isolated.get_settings.cache_clear()
        assert runtime_store.get_store() is None
        # no crash, plain Settings
        assert isinstance(isolated.get_settings().port, int)

    def test_empty_numeric_override_falls_back_to_default(self, isolated) -> None:
        from src.config import runtime_store

        store = runtime_store.get_store()
        store.bootstrap_from_env({"PORT": "3000"})
        store.create_revision({"PORT": ""}, source="ui")  # user cleared it
        isolated.get_settings.cache_clear()
        # blank int override is ignored -> code default, not a crash
        assert isolated.get_settings().port == 3000
