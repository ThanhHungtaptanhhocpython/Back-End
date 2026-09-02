"""Fix 2 -- /users/imagesearch must reject a pivot-by-id whose provenance
(`retrieval_backend`) disagrees with the currently active backend, before any
vector is reconstructed. BEiT3 and Jina CLIP v2 have independent vector-id
spaces, so a stale card from before a backend switch must not be resolved in
the wrong index.
"""

import sys
import types
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _fake_beit3_module(retriever):
    mod = types.ModuleType("src.services.beit3_retriever")
    mod.get_beit3_retriever = MagicMock(return_value=retriever)
    return mod


def _client():
    from main import app

    return TestClient(app)


def test_stale_beit3_id_rejected_when_jina_is_active(monkeypatch):
    monkeypatch.setattr(
        "src.services.retrieval_backend.active_backend", lambda settings=None: "jina_clip_v2"
    )
    resp = _client().post(
        "/users/imagesearch",
        data={"faiss_index": "512345", "topk": "5", "retrieval_backend": "beit3"},
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["success"] is False
    assert "beit3" in body["message"] and "jina_clip_v2" in body["message"]


def test_stale_jina_id_rejected_when_beit3_is_active(monkeypatch):
    monkeypatch.setattr(
        "src.services.retrieval_backend.active_backend", lambda settings=None: "beit3"
    )
    # Would blow up if the guard let it through to reconstruction.
    boom = MagicMock()
    boom.search_by_vector_id.side_effect = AssertionError("must not reconstruct on mismatch")
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", _fake_beit3_module(boom))

    resp = _client().post(
        "/users/imagesearch",
        data={"faiss_index": "77", "topk": "5", "retrieval_backend": "jina_clip_v2"},
    )
    assert resp.status_code == 409
    boom.search_by_vector_id.assert_not_called()


def test_matching_provenance_passes_through(monkeypatch):
    monkeypatch.setattr(
        "src.services.retrieval_backend.active_backend", lambda settings=None: "beit3"
    )
    retr = MagicMock()
    retr.search_by_vector_id.return_value = [
        {"vector_id": 100, "faiss_id": 100, "video_id": "V01", "retrieval_backend": "beit3"}
    ]
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", _fake_beit3_module(retr))

    resp = _client().post(
        "/users/imagesearch",
        data={"faiss_index": "100", "topk": "5", "retrieval_backend": "beit3"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["items"][0]["vector_id"] == 100
    retr.search_by_vector_id.assert_called_once_with(100, top_k=5)


def test_missing_provenance_rejected_422_active_beit3(monkeypatch):
    monkeypatch.setattr(
        "src.services.retrieval_backend.active_backend", lambda settings=None: "beit3"
    )
    boom = MagicMock()
    boom.search_by_vector_id.side_effect = AssertionError("must not reconstruct without provenance")
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", _fake_beit3_module(boom))

    resp = _client().post("/users/imagesearch", data={"faiss_index": "5", "topk": "3"})
    assert resp.status_code == 422
    assert resp.json()["success"] is False
    boom.search_by_vector_id.assert_not_called()


def test_missing_provenance_rejected_422_active_jina(monkeypatch):
    monkeypatch.setattr(
        "src.services.retrieval_backend.active_backend", lambda settings=None: "jina_clip_v2"
    )
    fake_jina = MagicMock()
    fake_jina.get_jina_retriever.side_effect = AssertionError("must not construct Jina without provenance")
    monkeypatch.setitem(sys.modules, "src.services.jina_retriever", fake_jina)

    resp = _client().post("/users/imagesearch", data={"faiss_index": "5", "topk": "3"})
    assert resp.status_code == 422
    fake_jina.get_jina_retriever.assert_not_called()


def test_unrecognised_provenance_value_rejected_422(monkeypatch):
    monkeypatch.setattr(
        "src.services.retrieval_backend.active_backend", lambda settings=None: "beit3"
    )
    boom = MagicMock()
    boom.search_by_vector_id.side_effect = AssertionError("must not reconstruct on garbage provenance")
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", _fake_beit3_module(boom))

    resp = _client().post(
        "/users/imagesearch",
        data={"faiss_index": "5", "topk": "3", "retrieval_backend": "not-a-backend"},
    )
    assert resp.status_code == 422
    boom.search_by_vector_id.assert_not_called()


def test_uploaded_image_needs_no_provenance(monkeypatch):
    monkeypatch.setattr(
        "src.services.retrieval_backend.active_backend", lambda settings=None: "jina_clip_v2"
    )
    captured = {}

    def _fake_by_file(file_obj, k):
        captured["k"] = k
        return []

    monkeypatch.setattr("src.services.user_service.getImageSearchByFile", _fake_by_file)
    resp = _client().post(
        "/users/imagesearch",
        data={"topk": "4"},
        files={"image": ("ref.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert resp.status_code == 200
    assert captured["k"] == 4
