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
    assert response.json()["selected"] == "jina"
    assert response.json()["loaded"] is False


def test_deep_retrieval_health_reports_index(monkeypatch):
    from src.api.routers import health_router
    from src.services import visual_retriever

    retriever = MagicMock()
    retriever._index.ntotal = 693124
    retriever._index.d = 1024
    retriever.loaded_model_revision = "abc123"
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
    assert response.json()["model_revision"] == "abc123"
