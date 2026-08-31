"""Local management API: runtime configuration, revision history, restart.

All routes are loopback-only (see :func:`require_local_client`) and never log
or return secret values -- the client only ever learns whether a secret is
configured.

Cloud-asset and AI-provider management endpoints are mounted on this same
router by their own modules (later phases).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status

from src.api.local_guard import require_local_client
from src.config import field_spec
from src.config.runtime_store import get_store, store_enabled
from src.config.settings import get_settings
from src.schemas.settings import (
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
        "store": _store_info(),
        "restart": launcher_control.read_status(),
    }


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
