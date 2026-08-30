"""Endpoint tests for POST /users/videos/captures/{video_id}/{frame_idx}/similar.

The captured-frame "Similar" search re-encodes the cached WebP preview with
BEiT3's vision tower. The real retriever needs multi-GB deployment artifacts, so
it is stubbed in ``sys.modules`` (same convention as
``tests/test_imagesearch_beit3.py``). The preview service seam is monkeypatched
so nothing here touches the cache directory or the network.
"""

import sys
import types
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from main import app
from src.services import video_frame_preview_service as vfps

client = TestClient(app)


def _fake_beit3_module(retriever):
    mod = types.ModuleType("src.services.beit3_retriever")
    mod.get_beit3_retriever = MagicMock(return_value=retriever)
    return mod


class _StubPreview:
    """Stands in for VideoFramePreviewService.get_existing (cache read only)."""

    def __init__(self, path):
        self._path = path
        self.calls = []

    def get_existing(self, video_id, frame_idx):
        self.calls.append((video_id, frame_idx))
        return self._path


def test_capture_similar_uses_cached_preview_and_beit3_image_query(tmp_path, monkeypatch):
    still = tmp_path / "L21_V001" / "351.webp"
    still.parent.mkdir(parents=True)
    still.write_bytes(b"RIFF\x00\x00\x00\x00WEBPfake")

    stub_preview = _StubPreview(still)
    monkeypatch.setattr(vfps, "get_video_frame_preview_service", lambda: stub_preview)

    retriever = MagicMock()
    retriever.search_by_image.return_value = [
        {"vector_id": 900 + i, "faiss_id": 900 + i, "video_id": "L30_V017", "frame_id": f"{i:06d}"}
        for i in range(4)
    ]
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", _fake_beit3_module(retriever))

    resp = client.post("/users/videos/captures/L21_V001/351/similar", json={"topk": 4})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total_items"] == 4
    assert body["data"]["items"][0]["faiss_id"] == 900

    # The exact cached still is the query; frame_idx is never a global vector id.
    retriever.search_by_image.assert_called_once_with(str(still), top_k=4)
    retriever.search_by_vector_id.assert_not_called()
    assert stub_preview.calls == [("L21_V001", 351)]


def test_capture_similar_root_prefix_alias_works(tmp_path, monkeypatch):
    still = tmp_path / "still.webp"
    still.write_bytes(b"webp-bytes")
    monkeypatch.setattr(vfps, "get_video_frame_preview_service", lambda: _StubPreview(still))

    retriever = MagicMock()
    retriever.search_by_image.return_value = []
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", _fake_beit3_module(retriever))

    resp = client.post("/videos/captures/L21_V001/351/similar", json={"topk": 10})
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_capture_similar_missing_cache_returns_error_and_skips_retriever(monkeypatch):
    monkeypatch.setattr(vfps, "get_video_frame_preview_service", lambda: _StubPreview(None))

    retriever = MagicMock()
    fake_module = _fake_beit3_module(retriever)
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", fake_module)

    resp = client.post("/users/videos/captures/L21_V001/351/similar", json={"topk": 5})

    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert "re-capture" in body["message"].lower()
    assert body["data"]["items"] == []

    # No fallback: the retriever is not even constructed, let alone a vector-id search.
    fake_module.get_beit3_retriever.assert_not_called()
    retriever.search_by_image.assert_not_called()
    retriever.search_by_vector_id.assert_not_called()


def test_capture_similar_preview_error_is_treated_as_cache_miss(monkeypatch):
    class _RaisingPreview:
        def get_existing(self, *_args, **_kwargs):
            raise vfps.FramePreviewError("Unsupported video id for preview")

    monkeypatch.setattr(vfps, "get_video_frame_preview_service", lambda: _RaisingPreview())

    retriever = MagicMock()
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", _fake_beit3_module(retriever))

    resp = client.post("/users/videos/captures/L21_V001/351/similar", json={"topk": 5})

    assert resp.status_code == 404
    assert resp.json()["success"] is False
    retriever.search_by_image.assert_not_called()


def test_capture_similar_rejects_non_positive_topk(tmp_path, monkeypatch):
    still = tmp_path / "s.webp"
    still.write_bytes(b"w")
    monkeypatch.setattr(vfps, "get_video_frame_preview_service", lambda: _StubPreview(still))

    resp = client.post("/users/videos/captures/L21_V001/351/similar", json={"topk": 0})
    assert resp.status_code in (400, 422)
