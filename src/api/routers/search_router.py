import logging
from typing import Optional
from fastapi import APIRouter, Form, UploadFile, File, Response, status
from src.schemas.search import TextSearchRequest, TranslateRequest, TranslateResponse
from src.schemas.temporal import TemporalSearchRequest
from src.schemas.results import AgentSearchResponse, BaseResponse, DataResponse

router = APIRouter()

def _empty_query_response() -> BaseResponse:
    return BaseResponse(success=True, message="Empty query ignored.", data=DataResponse(items=[], total_items=0))

@router.post("/translate", response_model=TranslateResponse)
def handle_translate(request: TranslateRequest):
    from src.utils.nlp_processing import Translation
    translator = Translation(from_lang=request.from_lang, to_lang=request.to_lang)
    translated_text = translator(request.text)
    return TranslateResponse(
        success=True,
        translated_text=translated_text,
        from_lang=request.from_lang,
        to_lang=request.to_lang,
        translated=translator.last_translated,
        provider=translator.last_provider
    )


@router.post("/singletextsearch", response_model=BaseResponse)
def handle_single_text_search(request: TextSearchRequest):
    if not request.query.strip():
        return _empty_query_response()
    # Lazy import to avoid loading heavy models on boot
    from src.services.user_service import getImageDataSingleTextSearch
    
    res = getImageDataSingleTextSearch(request.query, request.topk)
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
    return BaseResponse(
        success=True,
        message=summary.get("answer"),
        data=DataResponse(items=res, total_items=len(res), meta=summary)
    )

@router.post("/imagesearch", response_model=BaseResponse)
def handle_image_search(
    response: Response,
    topk: Optional[str] = Form("100"),
    clip: Optional[str] = Form(None),
    clipv2: Optional[str] = Form(None),
    faiss_index: Optional[str] = Form("default"),
    image: Optional[UploadFile] = File(None)
):
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
        from src.services.user_service import getImageSearchById
        res = getImageSearchById(faiss_index_int, topk_int)
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return BaseResponse(success=False, message="Either an uploaded image file or a valid faiss_index must be provided.", data=DataResponse(items=[], total_items=0))

    return BaseResponse(success=True, data=DataResponse(items=res, total_items=len(res)))

@router.post("/trakesearch", response_model=BaseResponse)
@router.post("/temporalsearch", response_model=BaseResponse)
def handle_trake_search(request: TemporalSearchRequest):
    from src.services.user_service import GetImageDataTrakeSearch
    
    query_dicts = [{"query": ev.query} for ev in request.query]
    
    res = GetImageDataTrakeSearch(query_dicts, top_results=request.topk)
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
    return AgentSearchResponse(
        success=True,
        response=result.get("answer", "Agent Search completed."),
        data=DataResponse(items=frames, total_items=len(frames)),
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
    
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.get("/video_keyframes/{video_id}", response_model=BaseResponse)
@router.get("/videos/{video_id}/keyframes", response_model=BaseResponse)
def handle_video_keyframes(
    video_id: str,
    around: Optional[str] = None,
    limit: Optional[int] = 60
):
    from src.services.beit3_retriever import get_beit3_retriever
    try:
        retriever = get_beit3_retriever()
        items = retriever.get_video_timeline(video_id=video_id, around_frame_id=around, limit=limit or 60)
        return BaseResponse(
            success=True,
            data=DataResponse(items=items, total_items=len(items))
        )
    except Exception as exc:
        logging.error(f"Error fetching timeline for video {video_id}: {exc}")
        return BaseResponse(
            success=False,
            message=str(exc),
            data=DataResponse(items=[], total_items=0)
        )
