"""Unit tests for /users/imagesearch wiring via the BEiT3 vector-id path.

The real BEiT3Retriever requires multi-GB deployment artifacts (checkpoint,
FAISS index, global_ids.parquet) that are not present in the dev environment,
so the retriever module is stubbed in sys.modules — matching the convention
used in tests/test_task3.py and tests/test_task4.py.
"""

import sys
import types
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _fake_beit3_module(retriever):
    mod = types.ModuleType("src.services.beit3_retriever")
    mod.get_beit3_retriever = MagicMock(return_value=retriever)
    return mod


def test_image_search_by_faiss_index(monkeypatch):
    mock_retriever = MagicMock()
    mock_retriever.search_by_vector_id.return_value = [
        {"vector_id": 100 + i, "faiss_id": 100 + i, "video_id": "V01", "frame_key": f"L21_V001_{i:04d}"}
        for i in range(5)
    ]

    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", _fake_beit3_module(mock_retriever))

    from main import app
    client = TestClient(app)

    response = client.post(
        "/users/imagesearch",
        data={"faiss_index": "100", "topk": "5"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]["items"]) == 5
    assert data["data"]["items"][0]["vector_id"] == 100

    # The endpoint must forward the parsed form values to the service layer.
    mock_retriever.search_by_vector_id.assert_called_once_with(100, top_k=5)
