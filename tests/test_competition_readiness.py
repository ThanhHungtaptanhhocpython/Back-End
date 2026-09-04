from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from main import app
from src.config.settings import Settings
from src.services.competition_readiness import run_readiness_audit


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


ROOT = Path(__file__).resolve().parents[1]


def _case_dir() -> Path:
    root = ROOT / "scratch" / "readiness-fixtures" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo(root: Path) -> Path:
    root = root / "repo"
    _touch(root / "Dockerfile", b"FROM python:3.11-slim\n")
    _touch(root / "docker-compose.yml", b"services:\n  api:\n    build: .\n")
    _touch(root / "frontend" / ".env.example", b"# live: require FastAPI\nVITE_SEARCH_MODE=auto\n")
    _touch(root / "src" / "dict" / "ocr_results_jina.json", b"[]")
    _touch(root / "src" / "dict" / "asr_results_jina.json", b"[]")
    return root


def _settings(repo: Path, **overrides) -> Settings:
    meta = repo / "index_meta.json"
    meta.write_text(
        '{"model":"jina","vector_count":693124,"embedding_dim":1024,'
        '"metric":"inner_product_on_l2_normalized_vectors"}',
        encoding="utf-8",
    )
    media = _touch(repo / "media-info.zip", b"PK")
    maps = _touch(repo / "map-keyframes.zip", b"PK")
    values = dict(
        _env_file=None,
        src_dir=repo / "src",
        # Exercise the deprecated aliases: settings normalises VISUAL_RETRIEVER
        # -> RETRIEVAL_BACKEND and JINA_MODEL_NAME_OR_PATH -> JINA_MODEL_PATH.
        visual_retriever="jina",
        jina_model_name_or_path="jinaai/jina-clip-v2",
        jina_model_revision="e10d47f5691d0454a0fb5d13f46f2199b74cb436",
        jina_local_files_only=True,
        jina_faiss_index_path=_touch(repo / "jina_faiss.index"),
        jina_global_ids_path=_touch(repo / "global_ids.parquet"),
        jina_video_metadata_path=_touch(repo / "video_metadata.parquet"),
        jina_index_meta_path=meta,
        media_info_path=media,
        map_keyframes_path=maps,
        cloud_assets_enabled=True,
        cloud_assets_provider="azure_blob",
        cloud_assets_manifest_key="hcmai-assets.json",
        azure_storage_connection_string="UseDevelopmentStorage=true",
    )
    values.update(overrides)
    return Settings(**values)


def test_readiness_passes_for_jina_competition_config(monkeypatch):
    case = _case_dir()
    monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(case / "appdata"))
    repo = _repo(case)
    report = run_readiness_audit(settings=_settings(repo), repo_root=repo)
    assert report["ready"] is True
    assert report["summary"]["fail"] == 0


def test_readiness_passes_without_the_optional_jina_video_metadata_path(monkeypatch):
    case = _case_dir()
    monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(case / "appdata"))
    repo = _repo(case)
    report = run_readiness_audit(
        settings=_settings(repo, jina_video_metadata_path=None),
        repo_root=repo,
    )
    meta = next(item for item in report["checks"] if item["name"] == "jina_video_metadata")
    assert meta["status"] == "pass"
    assert report["ready"] is True


def test_readiness_fails_when_jina_revision_is_unpinned(monkeypatch):
    case = _case_dir()
    monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(case / "appdata"))
    repo = _repo(case)
    report = run_readiness_audit(
        settings=_settings(repo, jina_model_revision=None, jina_local_files_only=False),
        repo_root=repo,
    )
    failures = {item["name"] for item in report["checks"] if item["status"] == "fail"}
    assert report["ready"] is False
    assert "jina_model_reproducibility" in failures


def test_readiness_fails_when_cloud_manifest_key_is_blank(monkeypatch):
    case = _case_dir()
    monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(case / "appdata"))
    repo = _repo(case)
    # The canonical default 'hcmai-assets.json' is fine; a deliberate override
    # is fine; only a blank manifest key is a readiness failure.
    report = run_readiness_audit(
        settings=_settings(repo, cloud_assets_manifest_key=""),
        repo_root=repo,
    )
    failures = {item["name"] for item in report["checks"] if item["status"] == "fail"}
    assert report["ready"] is False
    assert "cloud_manifest_key" in failures


def test_readiness_accepts_the_default_cloud_manifest_key(monkeypatch):
    case = _case_dir()
    monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(case / "appdata"))
    repo = _repo(case)
    report = run_readiness_audit(
        settings=_settings(repo, cloud_assets_manifest_key="hcmai-assets.json"),
        repo_root=repo,
    )
    manifest = next(item for item in report["checks"] if item["name"] == "cloud_manifest_key")
    assert manifest["status"] == "pass"


def test_readiness_endpoint_returns_503_on_fail(monkeypatch):
    from src.api.routers import health_router

    case = _case_dir()
    repo = _repo(case)
    monkeypatch.setenv("HCMAI_APP_DATA_DIR", str(case / "appdata"))
    monkeypatch.setattr(
        health_router,
        "run_readiness_audit",
        lambda **_: {"ready": False, "summary": {"pass": 0, "warn": 0, "fail": 1}, "checks": []},
    )
    monkeypatch.setattr(health_router, "get_settings", lambda: _settings(repo))
    response = TestClient(app).get("/health/competition-readiness")
    assert response.status_code == 503
    assert response.json()["ready"] is False
