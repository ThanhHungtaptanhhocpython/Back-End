"""Explicit selector for the active visual embedding space."""

from __future__ import annotations

from typing import Any

from src.config.settings import Settings, get_settings


class VisualRetrieverConfigError(RuntimeError):
    """Raised when the configured visual retriever name is unsupported."""


def get_visual_retriever(settings: Settings | None = None) -> Any:
    """Return the configured singleton retriever.

    Selection never silently falls across embedding spaces. Set
    ``VISUAL_RETRIEVER=beit3`` explicitly to roll back from Jina.
    """
    settings = settings or get_settings()
    name = (settings.visual_retriever or "beit3").strip().lower()
    if name == "beit3":
        from src.services.beit3_retriever import get_beit3_retriever

        return get_beit3_retriever()
    if name == "jina":
        from src.services.jina_retriever import get_jina_retriever

        return get_jina_retriever()
    raise VisualRetrieverConfigError(
        f"Unsupported VISUAL_RETRIEVER={settings.visual_retriever!r}; expected 'beit3' or 'jina'."
    )
