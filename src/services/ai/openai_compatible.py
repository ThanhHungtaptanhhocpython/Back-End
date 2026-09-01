"""One provider implementation for every OpenAI-compatible chat endpoint.

NVIDIA NIM, Cerebras, Groq, OpenRouter, Gemini AI Studio and Cloudflare Workers
AI all expose ``POST {base_url}/chat/completions`` and ``GET {base_url}/models``
with ``Authorization: Bearer <key>``. They differ only in base URL, default
model ids and a couple of extra headers -- captured by the registry.

Network I/O goes through the module-level :data:`_TRANSPORT` so tests can swap
in a fake OpenAI-compatible server without monkeypatching ``urllib``.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from src.services.ai.base import (
    CATEGORY_EMPTY_RESPONSE,
    CATEGORY_MODEL_UNAVAILABLE,
    CATEGORY_NETWORK,
    CATEGORY_NOT_CONFIGURED,
    CATEGORY_TIMEOUT,
    ChatResult,
    ProviderError,
    classify_http_status,
)


class _UrllibTransport:
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, str]:
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return exc.code, detail
        except (socket.timeout, TimeoutError) as exc:
            raise ProviderError(CATEGORY_TIMEOUT, "request timed out") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise ProviderError(CATEGORY_TIMEOUT, "request timed out") from exc
            raise ProviderError(CATEGORY_NETWORK, f"network error: {type(reason).__name__}") from exc
        except OSError as exc:
            raise ProviderError(CATEGORY_NETWORK, f"network error: {type(exc).__name__}") from exc


# Swappable in tests.
_TRANSPORT: Any = _UrllibTransport()


def set_transport(transport: Any) -> Any:
    """Install a transport (``.request(method, url, headers, body, timeout)``).

    Returns the previous transport so tests can restore it.
    """
    global _TRANSPORT
    previous, _TRANSPORT = _TRANSPORT, transport
    return previous


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        provider_id: str,
        label: str,
        base_url: str,
        api_key: str | None,
        text_model: str = "",
        vision_model: str = "",
        timeout: float = 45.0,
        enabled: bool = False,
        extra_headers: dict[str, str] | None = None,
        requires: tuple[str, ...] = (),
        missing_requirements: tuple[str, ...] = (),
    ) -> None:
        self.id = provider_id
        self.label = label
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.text_model = (text_model or "").strip()
        self.vision_model = (vision_model or "").strip()
        self.timeout = float(timeout or 45.0)
        self.enabled = bool(enabled)
        self.extra_headers = dict(extra_headers or {})
        self.requires = requires
        self.missing_requirements = tuple(missing_requirements)

    # -- capability introspection ------------------------------------------
    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.base_url) and not self.missing_requirements

    def model_for(self, vision: bool) -> str:
        return self.vision_model if vision else self.text_model

    def available_for(self, vision: bool) -> bool:
        return self.enabled and self.is_configured() and bool(self.model_for(vision))

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    # -- calls -----------------------------------------------------------
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the parsed ``/chat/completions`` JSON payload or raise ProviderError."""
        if not self.is_configured():
            raise ProviderError(CATEGORY_NOT_CONFIGURED, "provider is not configured", provider=self.id)
        model = (model or "").strip()
        if not model:
            raise ProviderError(CATEGORY_MODEL_UNAVAILABLE, "no model id configured", provider=self.id)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if extra_body:
            body.update(extra_body)

        raw = json.dumps(body).encode("utf-8")
        status, text = _TRANSPORT.request(
            "POST", f"{self.base_url}/chat/completions", self._headers(), raw, self.timeout
        )
        if status >= 400:
            raise ProviderError(
                classify_http_status(status, text),
                f"HTTP {status}",
                provider=self.id,
                status=status,
            )
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ProviderError(CATEGORY_EMPTY_RESPONSE, "non-JSON response", provider=self.id) from exc
        return payload

    def chat_text(
        self,
        messages: list[dict[str, Any]],
        *,
        vision: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        model = self.model_for(vision)
        started = time.perf_counter()
        payload = self.chat_completion(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = (payload.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content")) or ""
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        content = str(content).strip()
        if not content:
            raise ProviderError(CATEGORY_EMPTY_RESPONSE, "empty completion", provider=self.id)
        return ChatResult(
            text=content,
            provider=self.id,
            model=model,
            latency_ms=latency_ms,
            raw=payload,
            finish_reason=str(choice.get("finish_reason") or ""),
        )

    def list_models(self) -> list[str]:
        if not self.is_configured():
            raise ProviderError(CATEGORY_NOT_CONFIGURED, "provider is not configured", provider=self.id)
        status, text = _TRANSPORT.request(
            "GET", f"{self.base_url}/models", self._headers(), None, self.timeout
        )
        if status >= 400:
            raise ProviderError(
                classify_http_status(status, text), f"HTTP {status}", provider=self.id, status=status
            )
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ProviderError(CATEGORY_EMPTY_RESPONSE, "non-JSON response", provider=self.id) from exc
        data = payload.get("data") if isinstance(payload, dict) else payload
        models: list[str] = []
        for entry in data or []:
            if isinstance(entry, dict) and entry.get("id"):
                models.append(str(entry["id"]))
            elif isinstance(entry, str):
                models.append(entry)
        return sorted(set(models))
