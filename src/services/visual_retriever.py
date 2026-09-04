"""Backward-compatibility shim for the retrieval-backend selector.

Historically this module was a *second* selector, keyed off the
``VISUAL_RETRIEVER`` environment variable. That name is now a deprecated
alias for ``RETRIEVAL_BACKEND`` (see ``src/config/settings.py``), and the
single source of truth for which embedding space serves a request is
:func:`src.services.retrieval_backend.active_backend` -- ``RETRIEVAL_BACKEND``
normalised, with ``CLOUD_ASSETS_ENABLED`` forcing ``jina_clip_v2``.

To avoid ever creating a second internal selector, this module now simply
forwards to :func:`src.services.retrieval_backend.get_active_retriever`. It is
kept only so older call sites and tests that import ``get_visual_retriever``
keep working; new code should import ``get_active_retriever`` directly.
"""

from __future__ import annotations

from typing import Any

from src.config.settings import Settings, get_settings


class VisualRetrieverConfigError(RuntimeError):
    """Retained for callers that still catch it. The real
    misconfiguration error is now
    :class:`src.services.retrieval_backend.RetrievalBackendError`."""


def get_visual_retriever(settings: Settings | None = None) -> Any:
    """Deprecated alias for
    :func:`src.services.retrieval_backend.get_active_retriever`.

    Routes through the canonical ``active_backend`` policy so this shim can
    never disagree with the rest of the app about which backend is live.
    """
    from src.services.retrieval_backend import get_active_retriever

    return get_active_retriever(settings or get_settings())
