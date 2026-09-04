"""Competition readiness checks for AIC HCM 2026 runtime rehearsals."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.app_paths import get_config_db_path
from src.config.settings import Settings, get_settings


EXPECTED_JINA_VECTOR_COUNT = 693_124
EXPECTED_JINA_DIM = 1024

S_PASS = "pass"
S_WARN = "warn"
S_FAIL = "fail"


@dataclass
class ReadinessCheck:
    name: str
    status: str
    detail: str
    category: str = "runtime"

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


def _check(name: str, ok: bool, detail: str, *, category: str = "runtime") -> ReadinessCheck:
    return ReadinessCheck(name=name, status=S_PASS if ok else S_FAIL, detail=detail, category=category)


def _warn(name: str, detail: str, *, category: str = "runtime") -> ReadinessCheck:
    return ReadinessCheck(name=name, status=S_WARN, detail=detail, category=category)


def _file_check(path: Path | None, label: str, *, category: str = "assets") -> ReadinessCheck:
    if path is None:
        return ReadinessCheck(label, S_FAIL, f"{label} is not configured.", category)
    resolved = Path(path)
    if not resolved.is_file():
        return ReadinessCheck(label, S_FAIL, f"missing file: {resolved}", category)
    return ReadinessCheck(label, S_PASS, f"{resolved} ({resolved.stat().st_size} bytes)", category)


def _dir_or_file_exists(path: Path | None, label: str, *, category: str = "assets") -> ReadinessCheck:
    if path is None:
        return ReadinessCheck(label, S_FAIL, f"{label} is not configured.", category)
    resolved = Path(path)
    if not resolved.exists():
        return ReadinessCheck(label, S_FAIL, f"missing path: {resolved}", category)
    return ReadinessCheck(label, S_PASS, str(resolved), category)


def _module_check(module_name: str) -> ReadinessCheck:
    try:
        found = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        found = False
    return _check(module_name, found, "installed" if found else "not importable", category="dependencies")


def _read_json(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if path is None:
        return None, "path is not configured"
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), ""
    except Exception as exc:  # noqa: BLE001 - diagnostics should keep going
        return None, f"{type(exc).__name__}: {exc}"


def _jina_model_is_local(settings: Settings) -> bool:
    return Path(str(settings.jina_model_name_or_path)).expanduser().exists()


def _jina_checks(settings: Settings) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    selected = (settings.visual_retriever or "").strip().lower()
    checks.append(
        _check(
            "visual_retriever",
            selected == "jina",
            f"VISUAL_RETRIEVER={settings.visual_retriever!r}; competition target is 'jina'.",
            category="retrieval",
        )
    )

    for path, label in (
        (settings.jina_faiss_index_path, "jina_faiss_index"),
        (settings.jina_global_ids_path, "jina_global_ids"),
        (settings.jina_video_metadata_path, "jina_video_metadata"),
        (settings.jina_index_meta_path, "jina_index_meta"),
    ):
        checks.append(_file_check(path, label))

    meta, err = _read_json(settings.jina_index_meta_path)
    if meta is None:
        checks.append(ReadinessCheck("jina_index_meta_contract", S_FAIL, err, "assets"))
    else:
        vector_count = int(meta.get("vector_count", -1))
        dim = int(meta.get("embedding_dim", -1))
        metric = str(meta.get("metric") or "")
        model = str(meta.get("model") or "").lower()
        ok = (
            vector_count == EXPECTED_JINA_VECTOR_COUNT
            and dim == EXPECTED_JINA_DIM
            and metric == "inner_product_on_l2_normalized_vectors"
            and model == "jina"
        )
        checks.append(
            _check(
                "jina_index_meta_contract",
                ok,
                (
                    f"model={model!r}, vectors={vector_count}, dim={dim}, metric={metric!r}; "
                    f"expected jina/{EXPECTED_JINA_VECTOR_COUNT}/{EXPECTED_JINA_DIM}."
                ),
                category="assets",
            )
        )

    model_local = _jina_model_is_local(settings)
    revision_pinned = bool(settings.jina_model_revision)
    if model_local:
        checks.append(
            ReadinessCheck(
                "jina_model_reproducibility",
                S_PASS,
                f"using local model path: {settings.jina_model_name_or_path}",
                "retrieval",
            )
        )
    else:
        checks.append(
            _check(
                "jina_model_reproducibility",
                revision_pinned,
                (
                    "JINA_MODEL_REVISION is pinned."
                    if revision_pinned
                    else "JINA_MODEL_REVISION is blank; startup may use a moving HF revision."
                ),
                category="retrieval",
            )
        )
        if not settings.jina_local_files_only:
            checks.append(
                _warn(
                    "jina_offline_startup",
                    "JINA_LOCAL_FILES_ONLY=false; first startup may require Hugging Face network access.",
                    category="retrieval",
                )
            )

    return checks


def _cloud_checks(settings: Settings) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    if not settings.cloud_assets_enabled:
        return [
            _warn(
                "cloud_assets",
                "CLOUD_ASSETS_ENABLED=false; deployment must provide local artifacts explicitly.",
                category="cloud",
            )
        ]
    provider_ok = settings.cloud_assets_provider in {"azure_blob", "s3_compatible"}
    checks.append(
        _check(
            "cloud_provider",
            provider_ok,
            f"CLOUD_ASSETS_PROVIDER={settings.cloud_assets_provider!r}",
            category="cloud",
        )
    )
    manifest_key = (settings.cloud_assets_manifest_key or "").strip()
    jina_manifest = manifest_key == "hcmai-assets-jina.json"
    checks.append(
        _check(
            "cloud_jina_manifest",
            jina_manifest,
            f"CLOUD_ASSETS_MANIFEST_KEY={manifest_key!r}; expected 'hcmai-assets-jina.json' for Jina.",
            category="cloud",
        )
    )
    if settings.cloud_assets_provider == "azure_blob":
        configured = bool(
            settings.azure_storage_connection_string
            or (settings.azure_storage_account_name and settings.azure_storage_primary_key)
        )
        checks.append(
            _check(
                "azure_credentials",
                configured,
                "Azure credentials configured." if configured else "missing Azure connection string or account+key.",
                category="cloud",
            )
        )
    return checks


def _runtime_store_check() -> ReadinessCheck:
    if os.environ.get("HCMAI_DISABLE_CONFIG_STORE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return _warn(
            "runtime_config_store",
            "HCMAI_DISABLE_CONFIG_STORE is set; production must manage env/config outside the Settings UI.",
            category="ops",
        )
    db_path = get_config_db_path()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        probe = db_path.parent / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        return ReadinessCheck(
            "runtime_config_store",
            S_FAIL,
            f"config directory is not writable ({db_path.parent}): {type(exc).__name__}: {exc}",
            "ops",
        )
    return ReadinessCheck("runtime_config_store", S_PASS, f"writable: {db_path}", "ops")


def _deployment_checks(repo_root: Path) -> list[ReadinessCheck]:
    dockerfile = repo_root / "Dockerfile"
    compose = repo_root / "docker-compose.yml"
    checks = [
        _check(
            "dockerfile",
            dockerfile.is_file() and dockerfile.stat().st_size > 0,
            f"{dockerfile} ({dockerfile.stat().st_size if dockerfile.exists() else 0} bytes)",
            category="deploy",
        ),
        _file_check(compose, "docker_compose", category="deploy"),
    ]
    return checks


def _media_checks(settings: Settings) -> list[ReadinessCheck]:
    return [
        _dir_or_file_exists(settings.get_media_info_path(), "media_info"),
        _dir_or_file_exists(settings.get_map_keyframes_path(), "map_keyframes"),
    ]


def _frontend_checks(repo_root: Path) -> list[ReadinessCheck]:
    env_example = repo_root / "frontend" / ".env.example"
    if not env_example.is_file():
        return [ReadinessCheck("frontend_env_example", S_FAIL, "missing frontend/.env.example", "frontend")]
    text = env_example.read_text(encoding="utf-8", errors="ignore")
    live_hint = "VITE_SEARCH_MODE=live" in text or "live:" in text
    return [
        _check(
            "frontend_live_mode_documented",
            live_hint,
            "frontend .env example documents live mode.",
            category="frontend",
        )
    ]


def _text_evidence_checks(repo_root: Path) -> list[ReadinessCheck]:
    return [
        _file_check(repo_root / "src" / "dict" / "ocr_results_jina.json", "ocr_results_jina"),
        _file_check(repo_root / "src" / "dict" / "asr_results_jina.json", "asr_results_jina"),
    ]


def _probe_elasticsearch(settings: Settings) -> ReadinessCheck:
    url = (settings.elasticsearch_url or "").rstrip("/") + "/_cluster/health"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, socket.timeout) as exc:
        return ReadinessCheck(
            "elasticsearch_health",
            S_FAIL,
            f"{url} unavailable: {type(exc).__name__}: {exc}",
            "elasticsearch",
        )
    status = str(body.get("status") or "").lower()
    return _check(
        "elasticsearch_health",
        status in {"green", "yellow"},
        f"cluster status={status!r}",
        category="elasticsearch",
    )


def _deep_retriever_check(query: str | None) -> ReadinessCheck:
    try:
        from src.services.visual_retriever import get_visual_retriever

        retriever = get_visual_retriever()
        index = getattr(retriever, "_index", None)
        if query:
            results = retriever.search_visual(query, top_k=1)
            ok = bool(results)
            detail = f"loaded ntotal={getattr(index, 'ntotal', None)} dim={getattr(index, 'd', None)} query_hits={len(results)}"
        else:
            ok = index is not None
            detail = f"loaded ntotal={getattr(index, 'ntotal', None)} dim={getattr(index, 'd', None)}"
        return _check("deep_jina_retriever", ok, detail, category="retrieval")
    except Exception as exc:  # noqa: BLE001
        return ReadinessCheck("deep_jina_retriever", S_FAIL, f"{type(exc).__name__}: {exc}", "retrieval")


def run_readiness_audit(
    *,
    settings: Settings | None = None,
    repo_root: Path | None = None,
    deep: bool = False,
    query: str | None = None,
) -> dict[str, Any]:
    """Return a structured pass/fail report for competition readiness."""
    settings = settings or get_settings()
    repo_root = repo_root or Path(settings.src_dir).resolve().parent

    checks: list[ReadinessCheck] = []
    checks.extend(_jina_checks(settings))
    checks.extend(_cloud_checks(settings))
    checks.extend(_media_checks(settings))
    checks.extend(_text_evidence_checks(repo_root))
    checks.extend(_deployment_checks(repo_root))
    checks.extend(_frontend_checks(repo_root))
    checks.append(_runtime_store_check())
    checks.extend(
        _module_check(name)
        for name in ("faiss", "pandas", "pyarrow", "torch", "transformers", "elasticsearch")
    )

    if deep:
        checks.append(_probe_elasticsearch(settings))
        checks.append(_deep_retriever_check(query))

    fail_count = sum(1 for item in checks if item.status == S_FAIL)
    warn_count = sum(1 for item in checks if item.status == S_WARN)
    pass_count = sum(1 for item in checks if item.status == S_PASS)
    return {
        "ready": fail_count == 0,
        "deep": deep,
        "summary": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
        "checks": [item.to_dict() for item in checks],
    }
