"""Selects the active visual-retrieval backend (BEiT3 or Jina CLIP v2).

`RETRIEVAL_BACKEND` picks which encoder + FAISS index serves textual KIS,
grounded Q&A candidate retrieval, TRAKE per-event retrieval, the video
timeline, and the image-similarity paths (search-by-uploaded-image, "Similar"
on a captured frame, similar-by-vector-id). Jina CLIP v2 is the default;
turning `CLOUD_ASSETS_ENABLED` on forces it regardless of `RETRIEVAL_BACKEND`
(see `active_backend`). Both retrievers are lazy
singletons (see `beit3_retriever.get_beit3_retriever` /
`jina_retriever.get_jina_retriever`); importing this module, or calling
`get_active_retriever` for one backend, never imports or loads the other
backend's model. Routing every path through here is what keeps the two
vector-id spaces from ever crossing -- a Jina result's vector id is always
reconstructed in the Jina index, a BEiT3 one in the BEiT3 index.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.config.settings import Settings, get_settings

BEIT3 = "beit3"
JINA_CLIP_V2 = "jina_clip_v2"
VALID_BACKENDS = (BEIT3, JINA_CLIP_V2)


class Retriever(Protocol):
    """The retriever methods shared by BEiT3Retriever and JinaRetriever, used
    by the backend-agnostic call sites (textual KIS, grounded Q&A, TRAKE,
    video timeline, image similarity)."""

    backend_id: str

    def search_visual(self, query: str, top_k: int = 20) -> list[dict]: ...

    def search_by_image(self, image: object, top_k: int = 20) -> list[dict]: ...

    def search_by_vector_id(self, vector_id: int, top_k: int = 20) -> list[dict]: ...

    def get_frame_by_vector_id(self, vector_id: int) -> dict | None: ...

    def get_nearest_frame(self, video_id: str, timestamp: float) -> dict | None: ...

    def get_video_timeline(
        self, video_id: str, around_frame_id: str | None = None, limit: int = 60
    ) -> list[dict]: ...


class RetrievalBackendError(RuntimeError):
    """Raised for an unknown/misconfigured RETRIEVAL_BACKEND value."""


class BackendProvenanceRequired(RuntimeError):
    """A pivot-by-vector-id request arrived without a usable ``retrieval_backend``
    provenance tag (missing, blank, or an unrecognised value). BEiT3 and Jina
    ids share no space, so the server cannot guess which index to reconstruct
    in -- the request is rejected (HTTP 422) rather than run against a possibly
    wrong index."""


class BackendMismatchError(RuntimeError):
    """A caller supplied a vector id qualified with a backend that is not the
    one currently active. Reconstructing it in the active index would return
    an unrelated frame, so the request is rejected (HTTP 409)."""

    def __init__(self, claimed: str, active: str):
        self.claimed = claimed
        self.active = active
        super().__init__(
            f"This result was produced by the {claimed!r} retrieval backend but "
            f"{active!r} is active now. Re-run the search before pivoting on it "
            f"(the two backends have independent vector-id spaces)."
        )


class BackendPreparingError(RuntimeError):
    """The active backend is not ready yet -- a startup cloud sync or model
    warm is still in flight. Retryable (HTTP 503 + Retry-After)."""


def active_backend(settings: Settings | None = None) -> str:
    """The retrieval backend that is actually serving requests right now.

    Policy (single source of truth for every call site):

    * ``CLOUD_ASSETS_ENABLED`` true  -> always ``jina_clip_v2``. The cloud
      asset store exists to serve the Azure-hosted Jina index; with it on,
      BEiT3 is not used even if ``RETRIEVAL_BACKEND`` still says ``beit3``.
    * otherwise -> ``RETRIEVAL_BACKEND`` (normalised), defaulting to
      ``jina_clip_v2`` when unset. ``beit3`` is the explicit local fallback.
    """
    settings = settings or get_settings()
    if getattr(settings, "cloud_assets_enabled", False):
        return JINA_CLIP_V2
    return (settings.retrieval_backend or JINA_CLIP_V2).strip().lower() or JINA_CLIP_V2


_BACKEND_ALIASES = {
    "beit3": BEIT3, "beit-3": BEIT3, "beit_3": BEIT3,
    "jina": JINA_CLIP_V2, "jina_clip_v2": JINA_CLIP_V2, "jina-clip-v2": JINA_CLIP_V2,
}


def normalize_backend_name(value: str | None) -> str | None:
    """Canonicalise a backend name/alias, or ``None`` if unrecognised/blank."""
    key = (value or "").strip().lower()
    if not key:
        return None
    return _BACKEND_ALIASES.get(key, key if key in VALID_BACKENDS else None)


def assert_active_backend(claimed: str | None, settings: Settings | None = None) -> None:
    """Guard a pivot-by-vector-id request.

    Provenance is **mandatory** for a raw-id pivot: BEiT3 and Jina ids occupy
    independent spaces, so an id with no ``retrieval_backend`` cannot be
    reconstructed safely.

    * missing / blank / unrecognised ``claimed`` -> :class:`BackendProvenanceRequired` (HTTP 422)
    * a recognised backend that is not the active one -> :class:`BackendMismatchError` (HTTP 409)

    Callers that carry no id (uploaded-image / captured-frame search) must not
    call this -- they legitimately have nothing to qualify.
    """
    normalized = normalize_backend_name(claimed)
    if normalized is None:
        raise BackendProvenanceRequired(
            "retrieval_backend is required for an image-pivot-by-id request "
            "(BEiT3 and Jina CLIP v2 have independent vector-id spaces). "
            f"Got {claimed!r}; re-run the search to obtain a result card that carries it."
        )
    current = active_backend(settings)
    if normalized != current:
        raise BackendMismatchError(normalized, current)


def _preparing_if_syncing(exc: Exception) -> None:
    """Re-raise as :class:`BackendPreparingError` iff a tracked artifact sync
    is currently running -- otherwise the caller sees the real error."""
    try:
        from src.services.assets.sync_state import get_sync_progress

        if get_sync_progress().to_dict().get("state") == "running":
            raise BackendPreparingError(
                "The retrieval backend is still preparing (cloud asset sync in "
                "progress). Retry shortly."
            ) from exc
    except BackendPreparingError:
        raise
    except Exception:  # noqa: BLE001 - never mask the original error with a probe failure
        return


def get_active_retriever(settings: Settings | None = None) -> Any:
    """Return the lazily-loaded retriever singleton for the active backend
    (see :func:`active_backend` -- cloud assets on forces Jina CLIP v2)."""
    settings = settings or get_settings()
    backend = active_backend(settings)
    if backend == JINA_CLIP_V2:
        from src.services.jina_retriever import get_jina_retriever

        try:
            return get_jina_retriever()
        except Exception as exc:  # noqa: BLE001
            _preparing_if_syncing(exc)
            raise
    if backend == BEIT3:
        from src.services.beit3_retriever import get_beit3_retriever

        return get_beit3_retriever()
    raise RetrievalBackendError(
        f"Unknown RETRIEVAL_BACKEND={backend!r}; expected one of {VALID_BACKENDS}."
    )
