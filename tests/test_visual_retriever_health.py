from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from main import app
from src.config.settings import Settings


def test_retrieval_health_is_shallow_by_default(monkeypatch):
    from src.api.routers import health_router

    monkeypatch.setattr(
        health_router,
        "get_settings",
        lambda: Settings(_env_file=None, visual_retriever="jina"),
    )
    response = TestClient(app).get("/health/retrieval")
    assert response.status_code == 200
    # The deprecated VISUAL_RETRIEVER=jina alias normalises to the canonical
    # RETRIEVAL_BACKEND value, and /health reports the active backend.
    assert response.json()["selected"] == "jina_clip_v2"
    assert response.json()["loaded"] is False


def test_deep_retrieval_health_reports_index(monkeypatch):
    from src.api.routers import health_router
    from src.services import visual_retriever

    retriever = MagicMock()
    retriever._index.ntotal = 693124
    retriever._index.d = 1024
    retriever.loaded_model_revision = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
    monkeypatch.setattr(
        health_router,
        "get_settings",
        lambda: Settings(_env_file=None, visual_retriever="jina"),
    )
    monkeypatch.setattr(visual_retriever, "get_visual_retriever", lambda _settings: retriever)

    response = TestClient(app).get("/health/retrieval?deep=true")
    assert response.status_code == 200
    assert response.json()["vector_count"] == 693124
    assert response.json()["embedding_dim"] == 1024
    assert response.json()["model_revision"] == "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
