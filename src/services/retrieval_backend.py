"""Selects the active visual-retrieval backend (BEiT3 or Jina CLIP v2).

`RETRIEVAL_BACKEND` picks which encoder + FAISS index serves textual KIS,
grounded Q&A candidate retrieval, and TRAKE per-event retrieval. Both
retrievers are lazy singletons (see `beit3_retriever.get_beit3_retriever` /
`jina_retriever.get_jina_retriever`); importing this module, or calling
`get_active_retriever` for one backend, never imports or loads the other
backend's model.

Image-similarity endpoints (search-by-uploaded-image, "Similar" on a captured
frame) always use BEiT3 directly and do not go through this module -- Jina
image encoding is out of scope for this migration.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.config.settings import Settings, get_settings

BEIT3 = "beit3"
JINA_CLIP_V2 = "jina_clip_v2"
VALID_BACKENDS = (BEIT3, JINA_CLIP_V2)


class Retriever(Protocol):
    """The subset of retriever methods shared by BEiT3Retriever and
    JinaRetriever, used by the backend-agnostic call sites (textual KIS,
    grounded Q&A, TRAKE, video timeline)."""

    backend_id: str

    def search_visual(self, query: str, top_k: int = 20) -> list[dict]: ...

    def get_frame_by_vector_id(self, vector_id: int) -> dict | None: ...

    def get_nearest_frame(self, video_id: str, timestamp: float) -> dict | None: ...

    def get_video_timeline(
        self, video_id: str, around_frame_id: str | None = None, limit: int = 60
    ) -> list[dict]: ...


class RetrievalBackendError(RuntimeError):
    """Raised for an unknown/misconfigured RETRIEVAL_BACKEND value."""


def get_active_retriever(settings: Settings | None = None) -> Any:
    """Return the lazily-loaded retriever singleton for the configured backend."""
    settings = settings or get_settings()
    backend = (settings.retrieval_backend or BEIT3).strip().lower()
    if backend == JINA_CLIP_V2:
        from src.services.jina_retriever import get_jina_retriever

        return get_jina_retriever()
    if backend == BEIT3:
        from src.services.beit3_retriever import get_beit3_retriever

        return get_beit3_retriever()
    raise RetrievalBackendError(
        f"Unknown RETRIEVAL_BACKEND={backend!r}; expected one of {VALID_BACKENDS}."
    )
