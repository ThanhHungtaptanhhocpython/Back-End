from fastapi import APIRouter, Response, status

from src.config.settings import get_settings
from src.services.competition_readiness import run_readiness_audit

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "success": True,
        "message": "ok"
    }


@router.get("/health/retrieval")
def retrieval_health(response: Response, deep: bool = False):
    """Report the selected visual corpus; optionally load and validate it."""
    settings = get_settings()
    selected = (settings.visual_retriever or "beit3").strip().lower()
    result = {
        "success": True,
        "selected": selected,
        "loaded": False,
        "message": "configuration only; use ?deep=true to load model/index",
    }
    if not deep:
        return result

    try:
        from src.services.visual_retriever import get_visual_retriever

        retriever = get_visual_retriever(settings)
        index = getattr(retriever, "_index", None)
        result.update(
            loaded=True,
            message="retriever ready",
            vector_count=int(index.ntotal) if index is not None else None,
            embedding_dim=int(index.d) if index is not None else None,
            model_revision=getattr(retriever, "loaded_model_revision", None),
        )
        return result
    except Exception as exc:  # noqa: BLE001 - readiness must report initialization failures
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        result.update(success=False, message=str(exc))
        return result


@router.get("/health/competition-readiness")
def competition_readiness(response: Response, deep: bool = False, query: str | None = None):
    """Structured readiness gate for AIC HCM competition rehearsal."""
    result = run_readiness_audit(deep=deep, query=query)
    if not result["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
