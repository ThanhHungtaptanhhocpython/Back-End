from typing import Optional
from fastapi import APIRouter, Form, UploadFile, File, Response, status
from src.schemas.search import TextSearchRequest
from src.schemas.temporal import TemporalSearchRequest
from src.schemas.results import BaseResponse, DataResponse

router = APIRouter()

@router.post("/singletextsearch", response_model=BaseResponse)
def handle_single_text_search(request: TextSearchRequest):
    # Lazy import to avoid loading heavy models on boot
    from src.services.user_service import getImageDataSingleTextSearch
    
    res = getImageDataSingleTextSearch(request.query, request.topk)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/qnasearch", response_model=BaseResponse)
def handle_qna_search(request: TextSearchRequest):
    from src.services.user_service import getImageDataQAndASearch
    
    res = getImageDataQAndASearch(request.query, request.topk)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
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
    from src.services.user_service import getTextSearchOCR
    
    res = getTextSearchOCR(request.query, request.topk)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/asrsearch", response_model=BaseResponse)
def handle_asr_search(request: TextSearchRequest):
    from src.services.user_service import getTextSearchASR
    
    res = getTextSearchASR(request.query, request.topk)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/ocrandodsearch", response_model=BaseResponse)
def handle_ocr_and_od_search(request: TextSearchRequest):
    # This acts as a fallback for the legacy endpoint name
    from src.services.user_service import getTextSearchOCR
    
    res = getTextSearchOCR(request.query, request.topk)
    return BaseResponse(
        success=True,
        data=DataResponse(items=res, total_items=len(res))
    )

@router.post("/multimodalsearch", response_model=BaseResponse)
def handle_multimodal_search(request: TextSearchRequest):
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
