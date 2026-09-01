"""Provider registry: build :class:`OpenAICompatibleProvider` instances from
:class:`~src.config.settings.Settings` and expose the Text / Vision chains.

Model ids are **never** hard-coded in the app -- every provider def leaves the
text/vision model blank by default and the member fills them in (and confirms
with the per-provider Test endpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.config.settings import Settings, get_settings
from src.services.ai.openai_compatible import OpenAICompatibleProvider


@dataclass(frozen=True)
class ProviderDef:
    id: str
    label: str
    default_base_url: str
    key_field: str
    base_url_field: str | None
    text_model_field: str
    vision_model_field: str
    timeout_field: str
    enabled_field: str
    # returns (base_url, missing_requirements, extra_headers, api_key_override)
    resolver: Callable[[Settings, str], tuple[str, tuple[str, ...], dict[str, str], str | None]] | None = None


def _default_resolver(settings: Settings, base_url: str):
    return base_url, (), {}, None


def _openrouter_resolver(settings: Settings, base_url: str):
    headers = {
        "HTTP-Referer": settings.openrouter_site_url,
        "X-Title": settings.openrouter_app_name,
    }
    return base_url, (), headers, None


def _kilo_resolver(settings: Settings, base_url: str):
    # Kilo's gateway is OpenRouter-derived and accepts the same id headers.
    headers = {
        "HTTP-Referer": settings.openrouter_site_url,
        "X-Title": settings.openrouter_app_name,
    }
    return base_url, (), headers, None


def _gemini_resolver(settings: Settings, base_url: str):
    # AI Studio key, else fall back to the legacy GOOGLE_API_KEY.
    key = settings.gemini_api_key or settings.google_api_key
    return base_url, (), {}, key


def _cloudflare_resolver(settings: Settings, base_url: str):
    account_id = (settings.cloudflare_account_id or "").strip()
    if not account_id:
        return "", ("CLOUDFLARE_ACCOUNT_ID",), {}, None
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
    return url, (), {}, None


PROVIDER_DEFS: dict[str, ProviderDef] = {
    "nim": ProviderDef(
        id="nim", label="NVIDIA NIM",
        default_base_url="https://integrate.api.nvidia.com/v1",
        key_field="nim_api_key", base_url_field="nim_base_url",
        text_model_field="nim_text_model", vision_model_field="nim_vision_model",
        timeout_field="nim_timeout_seconds", enabled_field="nim_enabled",
        resolver=_default_resolver,
    ),
    "cerebras": ProviderDef(
        id="cerebras", label="Cerebras",
        default_base_url="https://api.cerebras.ai/v1",
        key_field="cerebras_api_key", base_url_field="cerebras_base_url",
        text_model_field="cerebras_text_model", vision_model_field="cerebras_vision_model",
        timeout_field="cerebras_timeout_seconds", enabled_field="cerebras_enabled",
        resolver=_default_resolver,
    ),
    "groq": ProviderDef(
        id="groq", label="Groq",
        default_base_url="https://api.groq.com/openai/v1",
        key_field="groq_api_key", base_url_field="groq_base_url",
        text_model_field="groq_text_model", vision_model_field="groq_vision_model",
        timeout_field="groq_timeout_seconds", enabled_field="groq_enabled",
        resolver=_default_resolver,
    ),
    "openrouter": ProviderDef(
        id="openrouter", label="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        key_field="openrouter_api_key", base_url_field="openrouter_base_url",
        text_model_field="openrouter_model", vision_model_field="openrouter_vision_model",
        timeout_field="openrouter_timeout_seconds", enabled_field="openrouter_enabled",
        resolver=_openrouter_resolver,
    ),
    "gemini": ProviderDef(
        id="gemini", label="Gemini AI Studio",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_field="gemini_api_key", base_url_field="gemini_base_url",
        text_model_field="gemini_text_model", vision_model_field="gemini_vision_model",
        timeout_field="gemini_timeout_seconds", enabled_field="gemini_enabled",
        resolver=_gemini_resolver,
    ),
    "cloudflare": ProviderDef(
        id="cloudflare", label="Cloudflare Workers AI",
        default_base_url="",
        key_field="cloudflare_api_key", base_url_field=None,
        text_model_field="cloudflare_text_model", vision_model_field="cloudflare_vision_model",
        timeout_field="cloudflare_timeout_seconds", enabled_field="cloudflare_enabled",
        resolver=_cloudflare_resolver,
    ),
    "kilo": ProviderDef(
        id="kilo", label="Kilo AI Gateway",
        default_base_url="https://kilocode.ai/api/openrouter",
        key_field="kilo_api_key", base_url_field="kilo_base_url",
        text_model_field="kilo_text_model", vision_model_field="kilo_vision_model",
        timeout_field="kilo_timeout_seconds", enabled_field="kilo_enabled",
        resolver=_kilo_resolver,
    ),
}


def all_provider_ids() -> list[str]:
    return list(PROVIDER_DEFS)


def build_provider(provider_id: str, settings: Settings | None = None) -> OpenAICompatibleProvider | None:
    pdef = PROVIDER_DEFS.get(provider_id)
    if pdef is None:
        return None
    settings = settings or get_settings()

    base_url = pdef.default_base_url
    if pdef.base_url_field:
        base_url = getattr(settings, pdef.base_url_field, "") or pdef.default_base_url

    resolver = pdef.resolver or _default_resolver
    base_url, missing, extra_headers, key_override = resolver(settings, base_url)

    api_key = key_override if key_override is not None else getattr(settings, pdef.key_field, None)
    return OpenAICompatibleProvider(
        provider_id=pdef.id,
        label=pdef.label,
        base_url=base_url,
        api_key=api_key,
        text_model=getattr(settings, pdef.text_model_field, "") or "",
        vision_model=getattr(settings, pdef.vision_model_field, "") or "",
        timeout=getattr(settings, pdef.timeout_field, 45.0) or 45.0,
        enabled=bool(getattr(settings, pdef.enabled_field, False)),
        extra_headers=extra_headers,
        missing_requirements=missing,
    )


def _chain(priority: list[str], settings: Settings, *, vision: bool) -> list[OpenAICompatibleProvider]:
    seen: set[str] = set()
    chain: list[OpenAICompatibleProvider] = []
    for pid in priority:
        if pid in seen or pid not in PROVIDER_DEFS:
            continue
        seen.add(pid)
        provider = build_provider(pid, settings)
        if provider is not None and provider.available_for(vision):
            chain.append(provider)
    return chain


def text_chain(settings: Settings | None = None) -> list[OpenAICompatibleProvider]:
    settings = settings or get_settings()
    return _chain(settings.get_ai_text_priority(), settings, vision=False)


def vision_chain(settings: Settings | None = None) -> list[OpenAICompatibleProvider]:
    settings = settings or get_settings()
    return _chain(settings.get_ai_vision_priority(), settings, vision=True)


def provider_status(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    text_priority = settings.get_ai_text_priority()
    vision_priority = settings.get_ai_vision_priority()
    out: list[dict[str, Any]] = []
    for pid, pdef in PROVIDER_DEFS.items():
        provider = build_provider(pid, settings)
        assert provider is not None
        out.append(
            {
                "id": pid,
                "label": pdef.label,
                "enabled": provider.enabled,
                "configured": provider.is_configured(),
                "missing_requirements": list(provider.missing_requirements),
                "base_url": provider.base_url,
                "text_model": provider.text_model,
                "vision_model": provider.vision_model,
                "timeout_seconds": provider.timeout,
                "in_text_chain": provider.available_for(False) and pid in text_priority,
                "in_vision_chain": provider.available_for(True) and pid in vision_priority,
                "text_priority_index": text_priority.index(pid) if pid in text_priority else None,
                "vision_priority_index": vision_priority.index(pid) if pid in vision_priority else None,
            }
        )
    return out
