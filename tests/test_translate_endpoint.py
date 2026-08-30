"""Tests for POST /users/translate (and the /translate root alias).

The translation provider is always mocked here: unit tests must never depend on
a live Google / OpenRouter call. ``FakeGoogleTranslator`` stands in for
``deep_translator.GoogleTranslator`` and the OpenRouter fallback is stubbed per
test.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from main import app  # noqa: E402
from src.utils import nlp_processing  # noqa: E402
from src.utils.nlp_processing import Translation  # noqa: E402

client = TestClient(app)


class FakeGoogleTranslator:
    """Scriptable stand-in for ``deep_translator.GoogleTranslator``."""

    result = ""          # str, or callable(text) -> str
    exc: Exception | None = None
    calls: list = []

    def __init__(self, source="auto", target="en"):
        self.source = source
        self.target = target

    def translate(self, text):
        FakeGoogleTranslator.calls.append((self.source, self.target, text))
        if FakeGoogleTranslator.exc is not None:
            raise FakeGoogleTranslator.exc
        res = FakeGoogleTranslator.result
        return res(text) if callable(res) else res


@pytest.fixture(autouse=True)
def _isolate_translation(monkeypatch):
    """Give every test a clean cache, a fake Google provider and no fallback."""
    Translation._cache.clear()
    FakeGoogleTranslator.result = ""
    FakeGoogleTranslator.exc = None
    FakeGoogleTranslator.calls = []
    monkeypatch.setattr(nlp_processing, "GoogleTranslator", FakeGoogleTranslator)
    # Opt-in per test; by default there is no working fallback.
    monkeypatch.setattr(Translation, "_translate_with_openrouter", lambda self, text: "")
    yield
    Translation._cache.clear()


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #
def test_google_success_is_returned_and_cached():
    FakeGoogleTranslator.result = "the woman wearing a pink shirt"

    r1 = client.post(
        "/users/translate",
        json={"text": "người phụ nữ mặc áo hồng", "from_lang": "vi", "to_lang": "en"},
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["success"] is True
    assert b1["status"] == "ok"
    assert b1["translated"] is True
    assert b1["provider"] == "google"
    assert b1["translated_text"] == "the woman wearing a pink shirt"
    assert b1["error_code"] is None
    assert len(FakeGoogleTranslator.calls) == 1

    # Second identical call is served from cache; the provider is not hit again.
    r2 = client.post(
        "/users/translate",
        json={"text": "người phụ nữ mặc áo hồng", "from_lang": "vi", "to_lang": "en"},
    )
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["provider"] == "cache"
    assert b2["translated"] is True
    assert b2["translated_text"] == "the woman wearing a pink shirt"
    assert len(FakeGoogleTranslator.calls) == 1


def test_unaccented_vietnamese_still_calls_the_provider():
    """`from_lang="vi"` is the source of truth: Latin-only text is still sent."""
    FakeGoogleTranslator.result = "the woman wearing a pink shirt"

    r = client.post(
        "/users/translate",
        json={"text": "nguoi phu nu mac ao hong", "from_lang": "vi", "to_lang": "en"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["translated"] is True
    assert body["provider"] == "google"
    assert FakeGoogleTranslator.calls[-1][2] == "nguoi phu nu mac ao hong"


def test_en_to_vi_translation_still_works():
    FakeGoogleTranslator.result = "người phụ nữ mặc áo hồng"

    r = client.post(
        "/users/translate",
        json={"text": "the woman wearing a pink shirt", "from_lang": "en", "to_lang": "vi"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["from_lang"] == "en"
    assert body["to_lang"] == "vi"
    assert body["translated"] is True
    assert body["translated_text"] == "người phụ nữ mặc áo hồng"


def test_same_language_is_the_only_identity_case():
    r = client.post(
        "/users/translate",
        json={"text": "hello world", "from_lang": "en", "to_lang": "en"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["status"] == "ok"
    assert body["translated"] is False
    assert body["provider"] == "identity"
    assert body["translated_text"] == "hello world"
    assert FakeGoogleTranslator.calls == []


# --------------------------------------------------------------------------- #
# Failure paths -- must never be cached and never disguised as success
# --------------------------------------------------------------------------- #
def test_google_echoing_the_input_is_a_failure_and_is_not_cached():
    src = "người phụ nữ mặc áo hồng"
    FakeGoogleTranslator.result = src  # provider handed the input straight back

    r1 = client.post(
        "/users/translate",
        json={"text": src, "from_lang": "vi", "to_lang": "en"},
    )
    assert r1.status_code == 503
    b1 = r1.json()
    assert b1["success"] is False
    assert b1["translated"] is False
    assert b1["provider"] == "none"
    assert b1["status"] == "provider_unavailable"
    assert b1["error_code"] == "provider_unavailable"
    assert b1["translated_text"] == src  # original kept for the UI

    # Nothing was frozen into the cache: a later working call translates fine.
    FakeGoogleTranslator.result = "the woman wearing a pink shirt"
    r2 = client.post(
        "/users/translate",
        json={"text": src, "from_lang": "vi", "to_lang": "en"},
    )
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["provider"] == "google"
    assert b2["translated"] is True
    assert b2["translated_text"] == "the woman wearing a pink shirt"


def test_google_exception_is_not_cached_and_retries_next_call():
    src = "một người đàn ông đang chạy"
    FakeGoogleTranslator.exc = RuntimeError("upstream 429")

    r1 = client.post(
        "/users/translate",
        json={"text": src, "from_lang": "vi", "to_lang": "en"},
    )
    assert r1.status_code == 503
    assert r1.json()["status"] == "provider_unavailable"
    assert Translation._cache == {}

    FakeGoogleTranslator.exc = None
    FakeGoogleTranslator.result = "a man running"
    r2 = client.post(
        "/users/translate",
        json={"text": src, "from_lang": "vi", "to_lang": "en"},
    )
    assert r2.status_code == 200
    assert r2.json()["translated_text"] == "a man running"


def test_openrouter_fallback_used_when_google_fails(monkeypatch):
    FakeGoogleTranslator.exc = RuntimeError("google down")
    monkeypatch.setattr(
        Translation,
        "_translate_with_openrouter",
        lambda self, text: "a man riding a bicycle",
    )

    r = client.post(
        "/users/translate",
        json={"text": "người đàn ông đi xe đạp", "from_lang": "vi", "to_lang": "en"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["status"] == "ok"
    assert body["provider"] == "openrouter"
    assert body["translated"] is True
    assert body["translated_text"] == "a man riding a bicycle"

    # A successful fallback is cached like any other successful translation.
    r2 = client.post(
        "/users/translate",
        json={"text": "người đàn ông đi xe đạp", "from_lang": "vi", "to_lang": "en"},
    )
    assert r2.json()["provider"] == "cache"


def test_both_providers_unavailable_reports_structured_failure(monkeypatch):
    FakeGoogleTranslator.result = ""  # google returned nothing
    monkeypatch.setattr(Translation, "_translate_with_openrouter", lambda self, text: "")
    src = "người phụ nữ mặc áo hồng"

    r = client.post(
        "/users/translate",
        json={"text": src, "from_lang": "vi", "to_lang": "en"},
    )
    assert r.status_code == 503
    body = r.json()
    assert body["success"] is False
    assert body["translated"] is False
    assert body["status"] == "provider_unavailable"
    assert body["error_code"] == "provider_unavailable"
    assert body["translated_text"] == src
    assert Translation._cache == {}


def test_blank_text_is_reported_as_invalid_input():
    r = client.post(
        "/users/translate",
        json={"text": "   ", "from_lang": "vi", "to_lang": "en"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["success"] is False
    assert body["status"] == "invalid_input"
    assert body["error_code"] == "invalid_input"


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_root_alias_is_still_mounted():
    FakeGoogleTranslator.result = "harvest pineapple"
    r = client.post(
        "/translate",
        json={"text": "thu hoạch dứa", "from_lang": "vi", "to_lang": "en"},
    )
    assert r.status_code == 200
    assert r.json()["translated_text"] == "harvest pineapple"
