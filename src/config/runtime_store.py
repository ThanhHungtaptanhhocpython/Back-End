"""SQLite-backed runtime configuration store.

On first run the store imports the repository ``.env`` into revision 1. From
then on the **active revision** is the authoritative runtime configuration and
takes precedence over ``.env`` and the code defaults (see
:func:`src.config.settings.get_settings`).

* Up to :data:`MAX_REVISIONS` revisions are retained (oldest pruned first, the
  active one is never pruned).
* Secret values are encrypted at rest with :class:`~src.config.secret_box.SecretBox`;
  the management API only ever exposes *configured / not configured* status.
* ``SRC_DIR`` and any other locked field is never read from or written to the
  store.

Disable the whole layer with ``HCMAI_DISABLE_CONFIG_STORE=1`` (used by the test
suite so legacy tests keep seeing plain ``.env`` behaviour).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import field_spec
from src.config.app_paths import get_config_db_path
from src.config.secret_box import SecretBox, SecretBoxError

MAX_REVISIONS = 10
SCHEMA_VERSION = 1
DISABLE_ENV = "HCMAI_DISABLE_CONFIG_STORE"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source     TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    active     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS revision_values (
    revision_id INTEGER NOT NULL REFERENCES revisions(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL DEFAULT '',
    is_secret   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (revision_id, key)
);
CREATE INDEX IF NOT EXISTS idx_revision_values_rev ON revision_values(revision_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RuntimeConfigStore:
    """Thread-safe wrapper around the per-user ``config.db``."""

    def __init__(self, db_path: str | Path, secret_box: SecretBox) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._box = secret_box
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            pass
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- construction -----------------------------------------------------
    @classmethod
    def open_default(cls) -> "RuntimeConfigStore":
        return cls(get_config_db_path(), SecretBox.load())

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- meta -----------------------------------------------------------------
    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # -- queries -----------------------------------------------------------
    def has_revisions(self) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM revisions LIMIT 1").fetchone()
            return row is not None

    def active_revision_id(self) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM revisions WHERE active = 1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return int(row["id"]) if row else None

    def _rows_for(self, revision_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT key, value, is_secret FROM revision_values WHERE revision_id = ?",
            (revision_id,),
        ).fetchall()

    def effective_values(self) -> dict[str, str]:
        """Decrypted key -> value map for the active revision (locked keys excluded)."""
        with self._lock:
            rev_id = self.active_revision_id()
            if rev_id is None:
                return {}
            locked = field_spec.locked_keys()
            out: dict[str, str] = {}
            for row in self._rows_for(rev_id):
                key = row["key"]
                if key in locked:
                    continue
                if row["is_secret"]:
                    raw = row["value"]
                    if not raw:
                        continue
                    try:
                        out[key] = self._box.decrypt(raw)
                    except SecretBoxError:
                        # A secret we can no longer read must not crash startup;
                        # skip it so the field falls back to .env / default.
                        continue
                else:
                    out[key] = row["value"]
            return out

    def secret_status(self) -> dict[str, bool]:
        """For every known secret key: is a non-empty value stored in the active revision?"""
        with self._lock:
            rev_id = self.active_revision_id()
            stored: dict[str, str] = {}
            if rev_id is not None:
                stored = {
                    r["key"]: r["value"]
                    for r in self._rows_for(rev_id)
                    if r["is_secret"]
                }
            return {key: bool(stored.get(key)) for key in sorted(field_spec.secret_keys())}

    def list_revisions(self, limit: int = MAX_REVISIONS) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created_at, source, note, active FROM revisions "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                counts = self._conn.execute(
                    "SELECT "
                    "  SUM(CASE WHEN is_secret = 0 THEN 1 ELSE 0 END) AS plain, "
                    "  SUM(CASE WHEN is_secret = 1 AND value <> '' THEN 1 ELSE 0 END) AS secrets "
                    "FROM revision_values WHERE revision_id = ?",
                    (row["id"],),
                ).fetchone()
                out.append(
                    {
                        "id": int(row["id"]),
                        "created_at": row["created_at"],
                        "source": row["source"],
                        "note": row["note"],
                        "active": bool(row["active"]),
                        "field_count": int(counts["plain"] or 0),
                        "secret_count": int(counts["secrets"] or 0),
                    }
                )
            return out

    def revision_values_masked(self, revision_id: int) -> dict[str, Any]:
        """Non-secret values verbatim; secret values reduced to a boolean 'set'."""
        with self._lock:
            rows = self._rows_for(revision_id)
            if not rows and not self._revision_exists(revision_id):
                raise KeyError(f"revision {revision_id} not found")
            plain: dict[str, str] = {}
            secrets: dict[str, bool] = {}
            for row in rows:
                if row["is_secret"]:
                    secrets[row["key"]] = bool(row["value"])
                else:
                    plain[row["key"]] = row["value"]
            return {"values": plain, "secrets": secrets}

    def _revision_exists(self, revision_id: int) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM revisions WHERE id = ?", (revision_id,)
            ).fetchone()
            is not None
        )

    # -- mutations --------------------------------------------------------
    def bootstrap_from_env(self, env_values: Mapping[str, str]) -> int | None:
        """Create revision 1 from ``.env`` if the store is empty. Returns its id."""
        with self._lock, self._conn:
            if self.has_revisions():
                return self.active_revision_id()
            known = field_spec.known_keys()
            locked = field_spec.locked_keys()
            secret = field_spec.secret_keys()
            plain: dict[str, str] = {}
            secret_set: dict[str, str] = {}
            for raw_key, raw_val in env_values.items():
                key = str(raw_key).upper()
                if key in locked or raw_val is None:
                    continue
                # Import unknown-but-present keys too, as plain strings, so the
                # migration is loss-free; the UI just won't render them.
                if key in secret:
                    if str(raw_val).strip():
                        secret_set[key] = str(raw_val)
                else:
                    plain[key] = str(raw_val)
                _ = known  # (kept for readability; all keys are imported)
            rev_id = self._insert_revision(
                source="env-import",
                note="Imported from .env on first run",
                plain=plain,
                secret_ciphertext={k: self._box.encrypt(v) for k, v in secret_set.items()},
            )
            self._activate(rev_id)
            self._set_meta("env_imported", "1")
            self._set_meta("env_imported_at", _utcnow())
            return rev_id

    def create_revision(
        self,
        values: Mapping[str, str],
        *,
        source: str,
        note: str = "",
        secret_set: Mapping[str, str] | None = None,
        secret_clear: Iterable[str] | None = None,
    ) -> int:
        """Persist a new active revision.

        ``values``       -- complete desired non-secret key -> normalised string.
        ``secret_set``   -- secret key -> new plaintext (encrypted here).
        ``secret_clear`` -- secret keys to drop.
        Secrets not mentioned are carried forward from the current active revision.
        """
        secret_set = dict(secret_set or {})
        secret_clear = set(secret_clear or ())
        locked = field_spec.locked_keys()
        with self._lock, self._conn:
            carried: dict[str, str] = {}
            current = self.active_revision_id()
            if current is not None:
                for row in self._rows_for(current):
                    if row["is_secret"] and row["value"]:
                        carried[row["key"]] = row["value"]
            for key in secret_clear:
                carried.pop(key.upper(), None)
            for key, plaintext in secret_set.items():
                key = key.upper()
                if str(plaintext).strip() == "":
                    continue
                carried[key] = self._box.encrypt(str(plaintext))

            plain = {
                str(k).upper(): "" if v is None else str(v)
                for k, v in values.items()
                if str(k).upper() not in locked
            }
            rev_id = self._insert_revision(
                source=source, note=note, plain=plain, secret_ciphertext=carried
            )
            self._activate(rev_id)
            self._prune()
            return rev_id

    def restore_revision(self, revision_id: int, *, note: str = "") -> int:
        """Copy an earlier revision verbatim into a new active revision."""
        with self._lock, self._conn:
            if not self._revision_exists(revision_id):
                raise KeyError(f"revision {revision_id} not found")
            rows = self._rows_for(revision_id)
            plain = {r["key"]: r["value"] for r in rows if not r["is_secret"]}
            secret_ct = {r["key"]: r["value"] for r in rows if r["is_secret"] and r["value"]}
            new_id = self._insert_revision(
                source="restore",
                note=note or f"Restored from revision {revision_id}",
                plain=plain,
                secret_ciphertext=secret_ct,
            )
            self._activate(new_id)
            self._prune()
            return new_id

    # -- internals ------------------------------------------------------------
    def _insert_revision(
        self,
        *,
        source: str,
        note: str,
        plain: Mapping[str, str],
        secret_ciphertext: Mapping[str, str],
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO revisions(created_at, source, note, active) VALUES (?, ?, ?, 0)",
            (_utcnow(), source, note),
        )
        rev_id = int(cur.lastrowid)
        payload = [(rev_id, k, v, 0) for k, v in plain.items()]
        payload += [(rev_id, k, v, 1) for k, v in secret_ciphertext.items()]
        if payload:
            self._conn.executemany(
                "INSERT INTO revision_values(revision_id, key, value, is_secret) "
                "VALUES (?, ?, ?, ?)",
                payload,
            )
        return rev_id

    def _activate(self, revision_id: int) -> None:
        self._conn.execute("UPDATE revisions SET active = 0 WHERE active = 1")
        self._conn.execute("UPDATE revisions SET active = 1 WHERE id = ?", (revision_id,))

    def _prune(self, keep: int = MAX_REVISIONS) -> None:
        keep_ids = {
            int(r["id"])
            for r in self._conn.execute(
                "SELECT id FROM revisions ORDER BY id DESC LIMIT ?", (keep,)
            ).fetchall()
        }
        active = self.active_revision_id()
        if active is not None:
            keep_ids.add(active)
        stale = [
            int(r["id"])
            for r in self._conn.execute("SELECT id FROM revisions").fetchall()
            if int(r["id"]) not in keep_ids
        ]
        if stale:
            marks = ",".join("?" for _ in stale)
            self._conn.execute(f"DELETE FROM revisions WHERE id IN ({marks})", stale)
            # ON DELETE CASCADE clears revision_values.

    # -- diagnostics -----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "db_path": str(self.db_path),
                "secret_backend": self._box.backend,
                "schema_version": int(self._get_meta("schema_version") or SCHEMA_VERSION),
                "env_imported": self._get_meta("env_imported") == "1",
                "env_imported_at": self._get_meta("env_imported_at"),
                "active_revision_id": self.active_revision_id(),
                "revision_count": int(
                    self._conn.execute("SELECT COUNT(*) AS c FROM revisions").fetchone()["c"]
                ),
                "max_revisions": MAX_REVISIONS,
            }


# ---------------------------------------------------------------------------
# module-level glue used by src.config.settings
# ---------------------------------------------------------------------------
_STORE: RuntimeConfigStore | None = None
_STORE_LOCK = threading.Lock()


def store_enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}


def get_store() -> RuntimeConfigStore | None:
    """Return a process-wide store singleton, or ``None`` when disabled."""
    global _STORE
    if not store_enabled():
        return None
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = RuntimeConfigStore.open_default()
    return _STORE


def reset_store() -> None:
    """Drop the singleton (tests that repoint ``HCMAI_APP_DATA_DIR``)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            try:
                _STORE.close()
            except Exception:
                pass
        _STORE = None


def _read_env_file(path: Path) -> dict[str, str]:
    try:
        from dotenv import dotenv_values  # python-dotenv, already a dependency
    except Exception:
        return _read_env_file_fallback(path)
    try:
        return {k: v for k, v in dotenv_values(str(path)).items() if v is not None}
    except Exception:
        return _read_env_file_fallback(path)


def _read_env_file_fallback(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("'").strip('"')
    return out


def load_effective_overrides(repo_root: Path | None = None) -> dict[str, str]:
    """Return the runtime overrides that :func:`get_settings` layers on defaults.

    * store disabled            -> ``{}`` (pure ``.env`` / env-var behaviour)
    * store empty + ``.env``    -> bootstrap revision 1, then return its values
    * store has active revision -> that revision's decrypted values
    """
    store = get_store()
    if store is None:
        return {}
    if not store.has_revisions():
        root = repo_root or Path(__file__).resolve().parents[2]
        env_path = root / ".env"
        if env_path.is_file():
            store.bootstrap_from_env(_read_env_file(env_path))
    return store.effective_values()
