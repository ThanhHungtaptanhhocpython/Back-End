"""Shared types and error classification for the AI provider gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- normalized error categories --------------------------------------------
CATEGORY_TIMEOUT = "timeout"
CATEGORY_RATE_LIMIT = "rate_limit"
CATEGORY_MODEL_UNAVAILABLE = "model_unavailable"
CATEGORY_AUTH = "auth"
CATEGORY_BAD_REQUEST = "bad_request"
CATEGORY_UPSTREAM = "upstream"
CATEGORY_NETWORK = "network"
CATEGORY_NOT_CONFIGURED = "not_configured"
CATEGORY_EMPTY_RESPONSE = "empty_response"

# Categories that clearly warrant trying the next provider in the chain.
RETRYABLE_CATEGORIES = frozenset(
    {
        CATEGORY_TIMEOUT,
        CATEGORY_RATE_LIMIT,
        CATEGORY_MODEL_UNAVAILABLE,
        CATEGORY_UPSTREAM,
        CATEGORY_NETWORK,
        CATEGORY_EMPTY_RESPONSE,
    }
)


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    latency_ms: int = 0
    raw: dict[str, Any] | None = None
    finish_reason: str = ""


@dataclass
class ProviderAttempt:
    provider: str
    ok: bool
    category: str = ""
    detail: str = ""
    latency_ms: int = 0
    model: str = ""


class ProviderError(Exception):
    """A single provider failed. ``category`` is one of the ``CATEGORY_*`` values."""

    def __init__(
        self,
        category: str,
        message: str = "",
        *,
        provider: str = "",
        status: int | None = None,
    ) -> None:
        super().__init__(message or category)
        self.category = category
        self.provider = provider
        self.status = status
        self.retryable = category in RETRYABLE_CATEGORIES

    def __str__(self) -> str:  # keep messages short; never echo request bodies
        base = super().__str__()
        return f"[{self.category}] {base}" if base != self.category else self.category


class AllProvidersFailed(Exception):
    """Every provider in the chain failed (or the chain was empty)."""

    def __init__(self, attempts: list[ProviderAttempt], *, chain: str) -> None:
        self.attempts = attempts
        self.chain = chain
        tried = ", ".join(f"{a.provider}:{a.category or 'ok'}" for a in attempts) or "<empty chain>"
        super().__init__(f"{chain} chain exhausted ({tried})")


def classify_http_status(status: int, body: str = "") -> str:
    """Map an HTTP status (+ optional body text) to a normalized category."""
    low = (body or "").lower()
    if status in (401, 403):
        return CATEGORY_AUTH
    if status == 404:
        return CATEGORY_MODEL_UNAVAILABLE
    if status == 408:
        return CATEGORY_TIMEOUT
    if status == 429:
        return CATEGORY_RATE_LIMIT
    if status in (400, 409, 422):
        if "model" in low and ("not" in low or "unknown" in low or "unsupported" in low or "does not exist" in low):
            return CATEGORY_MODEL_UNAVAILABLE
        if "context" in low and "length" in low:
            return CATEGORY_BAD_REQUEST
        return CATEGORY_BAD_REQUEST
    if status in (500, 502, 503, 504) or status >= 500:
        return CATEGORY_UPSTREAM
    return CATEGORY_UPSTREAM


@dataclass
class GatewayOutcome:
    ok: bool
    text: str = ""
    provider: str = ""
    model: str = ""
    payload: dict[str, Any] | None = None
    used_fallback: bool = False
    attempts: list[ProviderAttempt] = field(default_factory=list)
    error: str = ""
