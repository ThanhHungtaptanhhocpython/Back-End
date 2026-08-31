"""Text / Vision fallback orchestration.

``text_completion`` and ``vision_completion`` walk the configured chain, trying
each provider in order. On a :class:`ProviderError` the attempt is recorded and
the next provider is tried; when the chain is exhausted (or empty) an
:class:`AllProvidersFailed` is raised so the caller can fall back to local
behaviour.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config.settings import Settings, get_settings
from src.services.ai.base import (
    AllProvidersFailed,
    ChatResult,
    ProviderAttempt,
    ProviderError,
)
from src.services.ai.registry import text_chain, vision_chain

logger = logging.getLogger(__name__)


def text_available(settings: Settings | None = None) -> bool:
    return bool(text_chain(settings or get_settings()))


def vision_available(settings: Settings | None = None) -> bool:
    return bool(vision_chain(settings or get_settings()))


def _run_chain(
    chain: list[Any],
    *,
    label: str,
    vision: bool,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None,
    downgrade_response_format: bool,
) -> tuple[ChatResult, list[ProviderAttempt]]:
    attempts: list[ProviderAttempt] = []
    for provider in chain:
        model = provider.model_for(vision)
        try:
            result = provider.chat_text(
                messages,
                vision=vision,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        except ProviderError as exc:
            # Some routes reject response_format even though the model can still
            # follow a strict-JSON prompt -- retry once without it.
            if (
                downgrade_response_format
                and response_format is not None
                and exc.category == "bad_request"
            ):
                try:
                    result = provider.chat_text(
                        messages,
                        vision=vision,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format=None,
                    )
                    attempts.append(
                        ProviderAttempt(provider.id, ok=True, model=model, detail="no-json-schema")
                    )
                    return result, attempts
                except ProviderError as exc2:
                    exc = exc2
            attempts.append(
                ProviderAttempt(
                    provider.id, ok=False, category=exc.category, detail=str(exc), model=model
                )
            )
            logger.info("AI %s provider %s failed: %s", label, provider.id, exc)
            continue
        attempts.append(ProviderAttempt(provider.id, ok=True, model=model, latency_ms=result.latency_ms))
        return result, attempts

    raise AllProvidersFailed(attempts, chain=label)


def text_completion(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
) -> tuple[ChatResult, list[ProviderAttempt]]:
    settings = settings or get_settings()
    chain = text_chain(settings)
    return _run_chain(
        chain,
        label="text",
        vision=False,
        messages=messages,
        max_tokens=int(max_tokens or settings.ai_gateway_max_tokens or 1024),
        temperature=temperature,
        response_format=response_format,
        downgrade_response_format=True,
    )


def vision_completion(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[ProviderAttempt], str]:
    """Return (raw ``/chat/completions`` payload, attempts, provider_id)."""
    settings = settings or get_settings()
    chain = vision_chain(settings)
    result, attempts = _run_chain(
        chain,
        label="vision",
        vision=True,
        messages=messages,
        max_tokens=int(max_tokens or settings.ai_gateway_max_tokens or 1024),
        temperature=temperature,
        response_format=response_format,
        downgrade_response_format=True,
    )
    return result.raw or {}, attempts, result.provider
