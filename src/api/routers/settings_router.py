"""Local management API: runtime configuration, revision history, restart.

All routes are loopback-only (see :func:`require_local_client`) and never log
or return secret values -- the client only ever learns whether a secret is
configured.

Cloud-asset and AI-provider management endpoints are mounted on this same
router by their own modules (later phases).
"""

from __future__ import annotations

import logging
import os
import string
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, status

from src.api.local_guard import require_local_client
from src.config import field_spec
from src.config.runtime_store import get_store, store_enabled
from src.config.settings import get_settings
from src.schemas.settings import (
    CloudCacheClearRequest,
    CloudSyncRequest,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    GenericResponse,
    ProviderTestRequest,
    RestartRequest,
    ValidateRequest,
    ValidateResponse,
)
from src.services import launcher_control

# 1x1 red pixel, used only to exercise a provider's vision path during Test.
_TEST_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/settings",
    tags=["Local management"],
    dependencies=[Depends(require_local_client)],
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _require_store():
    store = get_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The runtime config store is disabled (HCMAI_DISABLE_CONFIG_STORE). "
                "Edit .env directly instead."
            ),
        )
    return store


def _store_info() -> dict:
    if not store_enabled():
        return {"enabled": False}
    try:
        info = _require_store().describe()
        info["enabled"] = True
        return info
    except HTTPException:
        return {"enabled": False}


def _classify(values: dict[str, str], secrets: dict[str, str]):
    """Validate a proposed change. Returns (normalized, secret_values, errors, unknown)."""
    normalized: dict[str, str] = {}
    errors: dict[str, str] = {}
    unknown: list[str] = []

    for raw_key, raw_val in (values or {}).items():
        key = str(raw_key).upper()
        spec = field_spec.by_key(key)
        if spec is None:
            unknown.append(key)
            normalized[key] = "" if raw_val is None else str(raw_val)
            continue
        if spec.locked:
            errors[key] = f"{key} is locked and cannot be changed."
            continue
        if spec.secret:
            errors[key] = f"{key} is a secret; send it via secret_set."
            continue
        try:
            normalized[key] = field_spec.validate_value(spec, raw_val)
        except field_spec.ValidationError as exc:
            errors[key] = str(exc)

    secret_values: dict[str, str] = {}
    for raw_key, raw_val in (secrets or {}).items():
        key = str(raw_key).upper()
        spec = field_spec.by_key(key)
        if spec is None or not spec.secret:
            errors[key] = f"{key} is not a known secret field."
            continue
        secret_values[key] = "" if raw_val is None else str(raw_val)

    return normalized, secret_values, errors, unknown


# ---------------------------------------------------------------------------
# schema + current config
# ---------------------------------------------------------------------------
@router.get("/schema")
def get_schema() -> dict:
    return {
        "group_order": list(field_spec.GROUP_ORDER),
        "groups": field_spec.grouped(),
        "store": _store_info(),
    }


# Path fields whose blank value is auto-resolved by a Settings.get_* method.
_RESOLVED_PATHS = {
    "KEYFRAMES_ROOT": "get_keyframes_root",
    "MEDIA_INFO_PATH": "get_media_info_path",
    "MAP_KEYFRAMES_PATH": "get_map_keyframes_path",
    "VIDEO_CAPTURE_CACHE_PATH": "get_video_capture_cache_path",
    "AGENT_VLM_CACHE_PATH": "get_agent_vlm_cache_path",
    "CLOUD_ASSETS_CACHE_PATH": "get_cloud_assets_cache_path",
    "LAUNCHER_FRONTEND_DIR": "get_launcher_frontend_dir",
}


def _resolved_paths(settings) -> dict:
    from pathlib import Path

    out: dict[str, dict] = {}
    for key, getter in _RESOLVED_PATHS.items():
        spec = field_spec.by_key(key)
        raw = getattr(settings, spec.field, None) if spec else None
        try:
            resolved = getattr(settings, getter)()
        except Exception:  # noqa: BLE001
            continue
        try:
            exists = Path(resolved).exists()
        except OSError:
            exists = False
        out[key] = {
            "path": str(resolved),
            "exists": exists,
            "is_default": raw in (None, ""),
        }
    return out


@router.get("/config")
def get_config() -> dict:
    settings = get_settings()
    store = get_store()
    if store is not None:
        secrets = store.secret_status()
    else:
        secrets = {
            s.key: bool(getattr(settings, s.field, None))
            for s in field_spec.all_specs()
            if s.secret
        }
    return {
        "values": settings.redacted_runtime_values(),
        "secrets": secrets,
        "resolved": _resolved_paths(settings),
        "store": _store_info(),
        "restart": launcher_control.read_status(),
    }


@router.get("/fs")
def browse_fs(path: str = "", dirs_only: bool = False, show_hidden: bool = False) -> dict:
    """Directory browser for the location fields.

    Loopback-only (router dependency). With no ``path`` it returns the roots
    (drive letters on Windows, ``/`` on POSIX) plus quick jumps. With a ``path``
    it lists that directory; a file path resolves to its parent listing.
    """
    settings = get_settings()
    home = str(Path.home())
    repo_root = str(Path(settings.src_dir).parent)
    sep = os.sep

    if "\x00" in path:
        raise HTTPException(status_code=400, detail="invalid path")

    if not path.strip():
        return {
            "sep": sep, "home": home, "repo_root": repo_root,
            "roots": _fs_roots(), "path": "", "parent": None, "entries": [],
        }

    target = Path(path).expanduser()
    try:
        target = target.resolve()
    except (OSError, RuntimeError):
        pass
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {target}")
    if target.is_file():
        target = target.parent
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {target}")

    entries: list[dict] = []
    error = ""
    try:
        with os.scandir(target) as scan:
            for entry in scan:
                name = entry.name
                if not show_hidden and name.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    is_dir = False
                if dirs_only and not is_dir:
                    continue
                size = None
                if not is_dir:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = None
                entries.append(
                    {"name": name, "path": str(target / name), "is_dir": is_dir, "size": size}
                )
                if len(entries) >= 5000:
                    error = "listing truncated at 5000 entries"
                    break
    except PermissionError:
        error = "permission denied"
    except OSError as exc:
        error = f"cannot read directory: {type(exc).__name__}"

    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    parent = str(target.parent) if str(target.parent) != str(target) else None
    return {
        "sep": sep, "home": home, "repo_root": repo_root,
        "path": str(target), "parent": parent, "entries": entries, "error": error,
    }


def _fs_roots() -> list[dict]:
    roots: list[dict] = []
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                roots.append({"name": drive, "path": drive})
    else:
        roots.append({"name": "/", "path": "/"})
    return roots


@router.post("/validate", response_model=ValidateResponse)
def validate_config(payload: ValidateRequest) -> ValidateResponse:
    normalized, _secret_values, errors, unknown = _classify(payload.values, payload.secrets)
    return ValidateResponse(
        ok=not errors,
        errors=errors,
        normalized=normalized,
        unknown_keys=unknown,
    )


@router.post("/config", response_model=ConfigUpdateResponse)
def update_config(payload: ConfigUpdateRequest, response: Response) -> ConfigUpdateResponse:
    store = _require_store()
    normalized, secret_values, errors, _unknown = _classify(payload.values, payload.secret_set)

    # validate secret_clear keys too
    for raw_key in payload.secret_clear:
        key = str(raw_key).upper()
        spec = field_spec.by_key(key)
        if spec is None or not spec.secret:
            errors[key] = f"{key} is not a known secret field."

    if errors:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ConfigUpdateResponse(ok=False, restart_required=False, errors=errors,
                                    detail="Validation failed; nothing was saved.")

    # Merge onto the current stored non-secret set so partial updates are safe.
    active_id = store.active_revision_id()
    base: dict[str, str] = {}
    if active_id is not None:
        base = dict(store.revision_values_masked(active_id).get("values", {}))
    base.update(normalized)

    secret_set = {k: v for k, v in secret_values.items() if str(v).strip() != ""}
    secret_clear = [str(k).upper() for k in payload.secret_clear]

    revision_id = store.create_revision(
        base,
        source="ui",
        note=payload.note or "Updated via Settings UI",
        secret_set=secret_set,
        secret_clear=secret_clear,
    )
    get_settings.cache_clear()

    restart_requested = False
    if payload.restart:
        launcher_control.request_restart("config-change", target_revision_id=revision_id)
        restart_requested = launcher_control.launcher_running()

    logger.info("Runtime config updated -> revision %s (secrets set: %d, cleared: %d)",
                revision_id, len(secret_set), len(secret_clear))
    return ConfigUpdateResponse(
        ok=True,
        revision_id=revision_id,
        restart_required=True,
        restart_requested=restart_requested,
        detail="Saved. Restart the app to apply.",
    )


# ---------------------------------------------------------------------------
# revision history
# ---------------------------------------------------------------------------
@router.get("/revisions")
def list_revisions() -> dict:
    store = _require_store()
    return {"revisions": store.list_revisions(), "active_revision_id": store.active_revision_id()}


@router.get("/revisions/{revision_id}")
def get_revision(revision_id: int) -> dict:
    store = _require_store()
    try:
        detail = store.revision_values_masked(revision_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Revision {revision_id} not found.")
    return {"revision_id": revision_id, **detail}


@router.post("/revisions/{revision_id}/restore", response_model=ConfigUpdateResponse)
def restore_revision(revision_id: int, payload: RestartRequest | None = None) -> ConfigUpdateResponse:
    store = _require_store()
    try:
        new_id = store.restore_revision(revision_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Revision {revision_id} not found.")
    get_settings.cache_clear()
    launcher_control.request_restart("revision-restore", target_revision_id=new_id)
    return ConfigUpdateResponse(
        ok=True,
        revision_id=new_id,
        restart_required=True,
        restart_requested=launcher_control.launcher_running(),
        detail=f"Restored revision {revision_id} as revision {new_id}. Restart to apply.",
    )


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------
@router.get("/restart/status")
def restart_status() -> dict:
    return launcher_control.read_status()


@router.post("/restart", response_model=GenericResponse)
def trigger_restart(payload: RestartRequest | None = None) -> GenericResponse:
    reason = (payload.reason if payload else None) or "manual"
    launcher_control.request_restart(reason)
    running = launcher_control.launcher_running()
    return GenericResponse(
        ok=running,
        detail="Restart requested." if running else "launcher_not_running",
        data={"launcher_running": running},
    )


# ---------------------------------------------------------------------------
# AI provider gateway
# ---------------------------------------------------------------------------
@router.get("/providers")
def list_providers() -> dict:
    from src.services.ai import registry

    settings = get_settings()
    return {
        "gateway_enabled": bool(settings.ai_gateway_enabled),
        "local_fallback_enabled": bool(settings.ai_local_fallback_enabled),
        "text_priority": settings.get_ai_text_priority(),
        "vision_priority": settings.get_ai_vision_priority(),
        "text_chain": [p.id for p in registry.text_chain(settings)],
        "vision_chain": [p.id for p in registry.vision_chain(settings)],
        "providers": registry.provider_status(settings),
    }


@router.get("/providers/{provider_id}/models")
def discover_models(provider_id: str) -> dict:
    from src.services.ai import registry
    from src.services.ai.base import ProviderError

    provider = registry.build_provider(provider_id, get_settings())
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider_id}'.")
    try:
        models = provider.list_models()
        return {"ok": True, "provider": provider_id, "models": models}
    except ProviderError as exc:
        return {"ok": False, "provider": provider_id, "models": [],
                "category": exc.category, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "provider": provider_id, "models": [],
                "category": "network", "detail": type(exc).__name__}


@router.post("/providers/{provider_id}/test")
def test_provider(provider_id: str, payload: ProviderTestRequest | None = None) -> dict:
    from src.services.ai import registry
    from src.services.ai.base import ProviderError

    payload = payload or ProviderTestRequest()
    vision = (payload.mode or "text").lower() == "vision"
    provider = registry.build_provider(provider_id, get_settings())
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider_id}'.")
    if not provider.is_configured():
        return {"ok": False, "provider": provider_id, "category": "not_configured",
                "detail": "Missing API key" + (
                    " / " + ", ".join(provider.missing_requirements)
                    if provider.missing_requirements else ""),
                "missing_requirements": list(provider.missing_requirements)}
    model = provider.model_for(vision)
    if not model:
        return {"ok": False, "provider": provider_id, "category": "model_unavailable",
                "detail": f"No {'vision' if vision else 'text'} model configured."}

    prompt = (payload.prompt or "").strip() or "Reply with the single word: pong"
    if vision:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _TEST_IMAGE_DATA_URL}},
        ]
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        result = provider.chat_text(messages, vision=vision, max_tokens=24, temperature=0.0)
    except ProviderError as exc:
        return {"ok": False, "provider": provider_id, "model": model,
                "category": exc.category, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "provider": provider_id, "model": model,
                "category": "network", "detail": type(exc).__name__}
    return {
        "ok": True,
        "provider": provider_id,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "sample": result.text[:200],
    }


# ---------------------------------------------------------------------------
# Cloud asset storage
# ---------------------------------------------------------------------------
def _importable(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@router.get("/cloud/status")
def cloud_status() -> dict:
    from src.services import assets

    settings = get_settings()
    art = assets.get_artifact_cache(settings)
    kf = assets.get_keyframe_cache(settings)
    return {
        "enabled": bool(settings.cloud_assets_enabled),
        "provider": settings.cloud_assets_provider,
        "active": assets.cloud_enabled(settings),
        "manifest_key": settings.cloud_assets_manifest_key,
        "sdk": {
            "azure_blob": _importable("azure.storage.blob"),
            "s3_compatible": _importable("boto3"),
        },
        "artifact_cache": art.stats(),
        "keyframe_cache": kf.stats(),
    }


@router.post("/cloud/test")
def cloud_test() -> dict:
    from src.services import assets

    settings = get_settings()
    store = assets.build_asset_store(settings)
    if store is None:
        return {"ok": False, "detail": "Cloud assets are set to 'local' / disabled.",
                "provider": settings.cloud_assets_provider}
    try:
        return store.probe().to_dict()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "provider": store.provider_id, "detail": type(exc).__name__}


@router.get("/cloud/manifest")
def cloud_manifest(refresh: bool = False) -> dict:
    from src.services import assets

    settings = get_settings()
    store = assets.build_asset_store(settings)
    if store is None:
        raise HTTPException(status_code=409, detail="Cloud assets are set to 'local' / disabled.")
    try:
        manifest = store.fetch_manifest() if refresh else (assets.get_manifest(store, force=refresh) or store.fetch_manifest())
    except assets.ManifestError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid manifest: {exc}")
    except assets.AssetStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    cache = assets.get_artifact_cache(settings)
    artifacts = []
    for art in manifest.artifacts:
        slot = cache.slot(manifest.version, art.name, expected_sha=art.sha256, expected_size=art.size)
        artifacts.append(
            {
                **art.to_dict(),
                "cached": slot.present,
                "verified": slot.verified,
                "local_path": str(slot.path) if slot.present else None,
            }
        )
    return {
        "version": manifest.version,
        "generated_at": manifest.generated_at,
        "keyframes": manifest.keyframes,
        "current_version": cache.get_current(),
        "artifacts": artifacts,
    }


@router.post("/cloud/sync")
def cloud_sync(payload: CloudSyncRequest | None = None) -> dict:
    from src.services import assets

    payload = payload or CloudSyncRequest()
    settings = get_settings()
    store = assets.build_asset_store(settings)
    if store is None:
        raise HTTPException(status_code=409, detail="Cloud assets are set to 'local' / disabled.")
    cache = assets.get_artifact_cache(settings)
    try:
        manifest = store.fetch_manifest()
    except assets.ManifestError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid manifest: {exc}")
    except assets.AssetStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    report = assets.sync_artifacts(
        store, cache, names=payload.names or None, manifest=manifest
    )
    return report.to_dict()


@router.get("/cloud/cache")
def cloud_cache() -> dict:
    from src.services import assets

    settings = get_settings()
    return {
        "artifact_cache": assets.get_artifact_cache(settings).stats(),
        "keyframe_cache": assets.get_keyframe_cache(settings).stats(),
    }


@router.post("/cloud/cache/clear", response_model=GenericResponse)
def cloud_cache_clear(payload: CloudCacheClearRequest | None = None) -> GenericResponse:
    from src.services import assets

    payload = payload or CloudCacheClearRequest()
    scope = (payload.scope or "all").lower()
    settings = get_settings()
    freed = 0
    if scope in ("artifacts", "all"):
        freed += assets.get_artifact_cache(settings).clear()
    if scope in ("keyframes", "all"):
        freed += assets.get_keyframe_cache(settings).clear()
    return GenericResponse(ok=True, detail=f"Cleared {scope} cache.", data={"freed_bytes": freed})
