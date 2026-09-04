import logging
from typing import Optional
from fastapi import APIRouter, Form, UploadFile, File, Response, status
from src.schemas.search import (
    CaptureSimilarRequest,
    TextSearchRequest,
    TranslateRequest,
    TranslateResponse,
)
from src.schemas.temporal import TemporalSearchRequest
from src.schemas.results import AgentSearchResponse, BaseResponse, DataResponse

router = APIRouter()

def _empty_query_response() -> BaseResponse:
    return BaseResponse(success=True, message="Empty query ignored.", data=DataResponse(items=[], total_items=0))


def _prefetch_result_keyframes(items) -> None:
    """Warm the cloud keyframe LRU cache for a just-produced result set so the
    browser's thumbnail requests mostly hit locally. No-op when cloud assets
    are off; never allowed to break a search response."""
    try:
        from src.services.assets.keyframe_prefetch import prefetch

        def _rows(seq):
            for it in seq or []:
                if not isinstance(it, dict):
                    continue
                yield it
                # TRAKE / Q&A rows nest the actual frames one level down.
                for nested_key in ("frames", "items"):
                    nested = it.get(nested_key)
                    if isinstance(nested, list):
                        yield from (n for n in nested if isinstance(n, dict))

        paths = [
            path
            for row in _rows(items)
            for path in (row.get("frame_path") or row.get("image_path") or row.get("asset_key"),)
            if path
        ]
        if paths:
            prefetch(paths)
    except Exception:  # noqa: BLE001
        logging.debug("keyframe prefetch skipped", exc_info=True)

@router.post("/translate", response_model=TranslateResponse)
def handle_translate(request: TranslateRequest, response: Response):
    """Translate ``text`` from ``from_lang`` to ``to_lang``.

    The HTTP status and body together make the outcome unambiguous:
    - 200 + ``status="ok"``: a real translation (or a same-language identity).
    - 400 + ``status="invalid_input"``: the text was blank.
    - 503 + ``status="provider_unavailable"``: no provider produced a
      translation. The original text is echoed back in ``translated_text`` so
      the UI can keep the query, but ``success`` is ``False`` -- it is never
      disguised as a successful translation.
    """
    from src.utils.nlp_processing import Translation

    translator = Translation(from_lang=request.from_lang, to_lang=request.to_lang)
    translated_text = translator(request.text)
    result = translator.last_result

    status_by_outcome = {
        "invalid_input": status.HTTP_400_BAD_REQUEST,
        "provider_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    if result.status in status_by_outcome:
        response.status_code = status_by_outcome[result.status]

    return TranslateResponse(
        success=result.status == "ok",
        translated_text=translated_text,
        from_lang=request.from_lang,
        to_lang=request.to_lang,
        translated=result.translated,
        provider=result.provider,
        status=result.status,
        error_code=result.error_code,
        detail=result.detail,
    )


@router.post("/singletextsearch", response_model=BaseResponse)
def handle_single_text_search(request: TextSearchRequest):
    if not request.query.strip():
        return _empty_query_response()
    # Lazy import to avoid loading heavy models on boot
    from src.services.user_service import getImageDataSingleTextSearch
    
    res = getImageDataSingleTextSearch(request.query, request.topk)
    _prefetch_result_keyframes(res)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/qnasearch", response_model=BaseResponse)
def handle_qna_search(request: TextSearchRequest):
    if not request.query.strip():
        return _empty_query_response()
    from src.services.user_service import getGroundedQASearch
    
    res, summary = getGroundedQASearch(request.query, request.topk)
    _prefetch_result_keyframes(res)
    return BaseResponse(
        success=True,
        message=summary.get("answer"),
        data=DataResponse(items=res, total_items=len(res), meta=summary)
    )

@router.post("/imagesearch", response_model=BaseResponse)
def handle_image_search(
    response: Response,
    topk: Optional[str] = Form("100"),
    faiss_index: Optional[str] = Form("default"),
    retrieval_backend: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Image pivot search on the ACTIVE retrieval backend's 1024-d index.

    An uploaded ``image`` is encoded with the active backend's vision tower
    (no existing vector id, so no provenance needed). A ``faiss_index`` is an
    existing vector id and MUST be reconstructed in the same backend that
    produced it, so ``retrieval_backend`` (the ``retrieval_backend`` field of
    the result card; the frontend fills in ``beit3`` when a card carries none)
    is **required** on that path:

    * missing / unrecognised ``retrieval_backend`` -> HTTP 422, nothing reconstructed
    * ``retrieval_backend`` != the active backend (stale card after a switch)
      -> HTTP 409, nothing reconstructed
    """
    try:
        topk_int = int(topk) if topk is not None else 100
        if topk_int <= 0:
            raise ValueError()
    except (TypeError, ValueError):
        response.status_code = status.HTTP_400_BAD_REQUEST
        return BaseResponse(success=False, message="topk must be a positive integer.", data=DataResponse(items=[], total_items=0))

    res = []
    # FastAPI UploadFile object has filename and file (SpooledTemporaryFile)
    if image is not None and image.filename != '' and image.filename is not None:
        from src.services.user_service import getImageSearchByFile
        res = getImageSearchByFile(image.file, topk_int)
    elif faiss_index and faiss_index not in ("default", "null"):
        try:
            faiss_index_int = int(faiss_index)
        except (TypeError, ValueError):
            response.status_code = status.HTTP_400_BAD_REQUEST
            return BaseResponse(success=False, message="faiss_index must be an integer.", data=DataResponse(items=[], total_items=0))
        from src.services.retrieval_backend import (
            BackendMismatchError,
            BackendProvenanceRequired,
            assert_active_backend,
        )
        try:
            assert_active_backend(retrieval_backend)
        except BackendProvenanceRequired as exc:
            response.status_code = 422
            return BaseResponse(success=False, message=str(exc), data=DataResponse(items=[], total_items=0))
        except BackendMismatchError as exc:
            response.status_code = status.HTTP_409_CONFLICT
            return BaseResponse(success=False, message=str(exc), data=DataResponse(items=[], total_items=0))
        from src.services.user_service import getImageSearchById
        res = getImageSearchById(faiss_index_int, topk_int)
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return BaseResponse(success=False, message="Either an uploaded image file or a valid faiss_index must be provided.", data=DataResponse(items=[], total_items=0))

    return BaseResponse(success=True, data=DataResponse(items=res, total_items=len(res)))


@router.post("/videos/captures/{video_id}/{frame_idx}/similar", response_model=BaseResponse)
def handle_capture_similar_search(
    video_id: str,
    frame_idx: int,
    request: CaptureSimilarRequest,
    response: Response,
):
    """Find keyframes similar to a captured frame's exact extracted still.

    A captured frame has no global FAISS vector id -- its ``frame_idx`` is a
    per-video index -- so this re-encodes the cached WebP preview with BEiT3's
    vision tower and searches the same 1024-d index used by visual text search.

    The cached preview is the only image source. If it is missing (never
    extracted, or LRU-evicted) this returns a clear "re-capture" error and does
    NOT fall back to another keyframe or to ``search_by_vector_id``.
    """
    from src.services.video_frame_preview_service import (
        FramePreviewError,
        get_video_frame_preview_service,
    )

    try:
        still_path = get_video_frame_preview_service().get_existing(video_id, frame_idx)
    except FramePreviewError:
        still_path = None

    if still_path is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return BaseResponse(
            success=False,
            message=(
                f"No captured preview image is cached for {video_id} frame {frame_idx}. "
                "Re-capture the frame, then run Similar again."
            ),
            data=DataResponse(items=[], total_items=0),
        )

    from src.services.retrieval_backend import BackendPreparingError
    from src.services.user_service import getCaptureSimilarSearch

    try:
        res = getCaptureSimilarSearch(str(still_path), request.topk)
    except BackendPreparingError:
        raise  # -> 503 + Retry-After (main._maybe_backend_preparing)
    except Exception as exc:  # noqa: BLE001 - surface a clear message, never a random result set
        logging.error("Captured-frame Similar search failed for %s#%s: %s", video_id, frame_idx, exc)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BaseResponse(
            success=False,
            message=f"Similar search failed: {exc}",
            data=DataResponse(items=[], total_items=0),
        )

    return BaseResponse(success=True, data=DataResponse(items=res, total_items=len(res)))


@router.post("/trakesearch", response_model=BaseResponse)
@router.post("/temporalsearch", response_model=BaseResponse)
def handle_trake_search(request: TemporalSearchRequest):
    from src.services.user_service import GetImageDataTrakeSearch

    context = (request.context or "").strip()

    # Retrieve one atomic visual event at a time. The shared narrative remains
    # available to the final sequence verifier, but must not dilute every
    # individual event embedding.
    query_dicts = [
        {"query": (ev.query or "").strip(), "context": context}
        for ev in request.query
        if (ev.query or "").strip()
    ]

    res = GetImageDataTrakeSearch(query_dicts, top_results=request.topk)
    _prefetch_result_keyframes(res)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/ocrsearch", response_model=BaseResponse)
def handle_ocr_search(request: TextSearchRequest):
    if not request.query.strip():
        return _empty_query_response()
    from src.services.user_service import getTextSearchOCR
    
    res = getTextSearchOCR(request.query, request.topk)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/asrsearch", response_model=BaseResponse)
def handle_asr_search(request: TextSearchRequest):
    if not request.query.strip():
        return _empty_query_response()
    from src.services.user_service import getTextSearchASR
    
    res = getTextSearchASR(request.query, request.topk)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/ocrandodsearch", response_model=BaseResponse)
def handle_ocr_and_od_search(request: TextSearchRequest):
    if not request.query.strip():
        return _empty_query_response()
    # This acts as a fallback for the legacy endpoint name
    from src.services.user_service import getTextSearchOCR
    
    res = getTextSearchOCR(request.query, request.topk)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/agentsearch", response_model=AgentSearchResponse)
def handle_agent_search(request: TextSearchRequest):
    if not request.query.strip():
        return AgentSearchResponse(
            success=True,
            message="Empty query ignored.",
            response="Prompt rong, khong the chay Agent Search.",
            data=DataResponse(items=[], total_items=0),
            plan={},
        )
    from src.services.agent_query_coordinator import run_agent_query_search

    result = run_agent_query_search(request.query, request.topk)
    frames = result.get("frames", [])
    items = result.get("sequences") if result.get("sequences") is not None else frames
    _prefetch_result_keyframes(items)
    return AgentSearchResponse(
        success=True,
        response=result.get("answer", "Agent Search completed."),
        data=DataResponse(items=items, total_items=len(items)),
        plan=result.get("plan", {}),
    )

@router.post("/multimodalsearch", response_model=BaseResponse)
def handle_multimodal_search(request: TextSearchRequest):
    if not request.query.strip():
        return _empty_query_response()
    from src.utils.nlp_processing import QueryPlanner
    from src.services.fusion_service import multimodal_search
    
    # 1. Parse the query to get visual/ocr/asr sub-queries and weights
    plan = QueryPlanner.parse_query(request.query)
    
    # 2. Execute the multimodal search
    res = multimodal_search(
        visual_query=plan["visual_query"],
        ocr_query=plan["ocr_query"],
        asr_query=plan["asr_query"],
        weights=plan["weights"],
        topk=request.topk,
        original_query=request.query
    )
    _prefetch_result_keyframes(res)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.get("/video_keyframes/{video_id}", response_model=BaseResponse)
@router.get("/videos/{video_id}/keyframes", response_model=BaseResponse)
def handle_video_keyframes(
    video_id: str,
    around: Optional[str] = None,
    limit: Optional[int] = 60,
    scope: str = "around",
):
    from src.services.retrieval_backend import BackendPreparingError, get_active_retriever
    try:
        retriever = get_active_retriever()
        full_video = scope.strip().lower() == "full"
        items = retriever.get_video_timeline(
            video_id=video_id,
            around_frame_id=around,
            limit=limit or 60,
            full_video=full_video,
        )
        _prefetch_result_keyframes(items)
        return BaseResponse(
            success=True,
            data=DataResponse(items=items, total_items=len(items))
        )
    except BackendPreparingError:
        raise  # -> 503 + Retry-After (main._maybe_backend_preparing)
    except Exception as exc:
        logging.error(f"Error fetching timeline for video {video_id}: {exc}")
        return BaseResponse(
            success=False,
            message=str(exc),
            data=DataResponse(items=[], total_items=0)
        )
