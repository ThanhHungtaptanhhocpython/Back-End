"""Symmetric encryption for secret configuration values.

Secrets (API keys, connection strings, access keys) are never stored in plain
text in the runtime-config database and are never returned by the management
API -- callers only ever learn whether a secret *is configured*.

Key material:
    * Preferred: the OS credential store via the optional ``keyring`` package
      (Windows Credential Manager / macOS Keychain / Secret Service).
    * Fallback: a 32-byte random key in ``<app-data>/secret.key`` with
      ``0600`` permissions on POSIX.

Cipher (stdlib only, no third-party crypto dependency required):
    Encrypt-then-MAC over a BLAKE2b keystream.
        enc_key = BLAKE2b("hcmai-enc", key=master)         (32 bytes)
        mac_key = BLAKE2b("hcmai-mac", key=master)         (32 bytes)
        keystream_block(i) = BLAKE2b(nonce || u64(i), key=enc_key, 64B)
        ciphertext = plaintext XOR keystream
        tag = HMAC-BLAKE2b(mac_key, nonce || ciphertext)
    Wire format: base64( 0x01 || nonce[16] || tag[32] || ciphertext ).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from pathlib import Path

from src.config.app_paths import get_secret_key_path

_KEYRING_SERVICE = "HCMAI2026"
_KEYRING_USERNAME = "runtime-config-secret-key"
_VERSION = 1
_NONCE_LEN = 16
_TAG_LEN = 32
_KEY_LEN = 32


class SecretBoxError(RuntimeError):
    """Raised when a stored secret cannot be decrypted or verified."""


def _load_from_keyring() -> bytes | None:
    try:
        import keyring  # type: ignore
    except Exception:
        return None
    try:
        stored = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    except Exception:
        return None
    if not stored:
        return None
    try:
        raw = base64.urlsafe_b64decode(stored.encode("ascii"))
    except Exception:
        return None
    return raw if len(raw) == _KEY_LEN else None


def _store_in_keyring(key: bytes) -> bool:
    try:
        import keyring  # type: ignore
    except Exception:
        return False
    try:
        keyring.set_password(
            _KEYRING_SERVICE,
            _KEYRING_USERNAME,
            base64.urlsafe_b64encode(key).decode("ascii"),
        )
        return True
    except Exception:
        return False


def _load_from_file(path: Path) -> bytes | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return raw if len(raw) == _KEY_LEN else None


def _store_in_file(path: Path, key: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write private, then move into place.
    tmp = path.with_suffix(".key.tmp")
    tmp.write_bytes(key)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class SecretBox:
    """Encrypts / decrypts secret strings with a machine-local master key."""

    def __init__(self, key: bytes, *, backend: str = "unknown") -> None:
        if len(key) != _KEY_LEN:
            raise ValueError("SecretBox master key must be 32 bytes")
        self._enc_key = hashlib.blake2b(b"hcmai-enc", key=key, digest_size=32).digest()
        self._mac_key = hashlib.blake2b(b"hcmai-mac", key=key, digest_size=32).digest()
        self.backend = backend

    # -- construction ----------------------------------------------------
    @classmethod
    def load(cls) -> "SecretBox":
        """Return a box backed by the OS keyring or the local key file."""
        key = _load_from_keyring()
        if key is not None:
            return cls(key, backend="keyring")

        key_path = get_secret_key_path()
        key = _load_from_file(key_path)
        if key is not None:
            return cls(key, backend="file")

        key = secrets.token_bytes(_KEY_LEN)
        if _store_in_keyring(key):
            return cls(key, backend="keyring")
        _store_in_file(key_path, key)
        return cls(key, backend="file")

    @classmethod
    def from_key(cls, key: bytes) -> "SecretBox":
        return cls(key, backend="explicit")

    # -- keystream -----------------------------------------------------------
    def _keystream(self, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hashlib.blake2b(
                nonce + counter.to_bytes(8, "big"),
                key=self._enc_key,
                digest_size=64,
            ).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def _tag(self, nonce: bytes, ciphertext: bytes) -> bytes:
        return hmac.new(self._mac_key, nonce + ciphertext, hashlib.blake2b).digest()[:_TAG_LEN]

    # -- API ------------------------------------------------------------------
    def encrypt(self, plaintext: str) -> str:
        data = plaintext.encode("utf-8")
        nonce = secrets.token_bytes(_NONCE_LEN)
        ciphertext = bytes(b ^ k for b, k in zip(data, self._keystream(nonce, len(data))))
        tag = self._tag(nonce, ciphertext)
        blob = bytes([_VERSION]) + nonce + tag + ciphertext
        return base64.b64encode(blob).decode("ascii")

    def decrypt(self, token: str) -> str:
        try:
            blob = base64.b64decode(token.encode("ascii"), validate=True)
        except Exception as exc:  # noqa: BLE001
            raise SecretBoxError("secret is not valid base64") from exc
        if len(blob) < 1 + _NONCE_LEN + _TAG_LEN:
            raise SecretBoxError("secret ciphertext is truncated")
        if blob[0] != _VERSION:
            raise SecretBoxError(f"unsupported secret version {blob[0]}")
        nonce = blob[1 : 1 + _NONCE_LEN]
        tag = blob[1 + _NONCE_LEN : 1 + _NONCE_LEN + _TAG_LEN]
        ciphertext = blob[1 + _NONCE_LEN + _TAG_LEN :]
        if not hmac.compare_digest(tag, self._tag(nonce, ciphertext)):
            raise SecretBoxError("secret failed integrity check (wrong key or tampered)")
        plaintext = bytes(b ^ k for b, k in zip(ciphertext, self._keystream(nonce, len(ciphertext))))
        return plaintext.decode("utf-8")
