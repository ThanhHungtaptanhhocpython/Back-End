import logging
import re
import unicodedata

from fastapi import APIRouter, HTTPException

from src.agent.memory_manager import memory_manager
from src.schemas.chat import ChatRequest, ChatResponse, DeepKeyframeSearchRequest, FeedbackRequest

router = APIRouter(
    prefix="/chat",
    tags=["Chat & Agent"]
)

logger = logging.getLogger(__name__)


def _normalise_intent_text(value: str) -> str:
    text = unicodedata.normalize("NFD", str(value or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def _looks_like_deep_search_request(message: str) -> bool:
    text = _normalise_intent_text(message)
    deep_markers = (
        "tim sau",
        "tim ky",
        "dao sau",
        "deep search",
        "khong tim duoc",
        "chua tim duoc",
        "chua tim thay",
        "khong tim thay",
        "tu thu nhieu huong",
        "retrieval agent",
    )
    retrieval_terms = ("frame", "khung hinh", "canh", "clip", "video", "query")
    return any(marker in text for marker in deep_markers) and any(term in text for term in retrieval_terms)


def _extract_user_question(message: str) -> str:
    text = str(message or "").strip()
    match = re.search(r"(?im)^user_question:\s*(.+)$", text)
    if match:
        return match.group(1).strip()
    return text


def _deep_search_response(session_id: str, result: dict) -> ChatResponse:
    return ChatResponse(
        success=True,
        session_id=session_id,
        response=result.get("answer", "Deep keyframe search completed."),
        data={
            "mode": "deep_keyframe_search",
            "queries_used": result.get("queries_used", []),
            "frames": result.get("frames", []),
            "video_results": result.get("video_results", []),
            "total_candidates": result.get("total_candidates", 0),
        },
    )


@router.post("/conversational_kis", response_model=ChatResponse)
async def conversational_kis(request: ChatRequest):
    """
    Conversational KIS endpoint.
    Maintains multi-turn context using memory_manager and LLM Planner.
    """
    try:
        import asyncio

        session_id = request.session_id
        user_message = request.message

        if _looks_like_deep_search_request(user_message):
            from src.services.deep_keyframe_search import deep_keyframe_search

            search_message = _extract_user_question(user_message)
            result = await asyncio.to_thread(deep_keyframe_search, search_message, 24, 36)
            return _deep_search_response(session_id, result)

        from src.agent.llm_planner import execute_chat_turn

        agent_result = await asyncio.to_thread(execute_chat_turn, session_id, user_message)
        history = memory_manager.get_messages(session_id)

        return ChatResponse(
            success=agent_result.get("success", True),
            session_id=session_id,
            response=agent_result.get("response", ""),
            data={
                "history_length": len(history),
                **(agent_result.get("data") or {}),
            }
        )
    except Exception as e:
        logger.error(f"Error in conversational KIS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deep_keyframe_search", response_model=ChatResponse)
async def deep_keyframe_search_endpoint(request: DeepKeyframeSearchRequest):
    """Run a deterministic multi-query keyframe retrieval pipeline for chat."""
    try:
        import asyncio
        from src.services.deep_keyframe_search import deep_keyframe_search

        session_id = request.session_id or "deep-search"
        topk = max(1, min(int(request.topk or 20), 100))
        per_query = max(5, min(int(request.per_query or 30), 100))
        result = await asyncio.to_thread(deep_keyframe_search, request.message, topk, per_query)

        return _deep_search_response(session_id, result)
    except Exception as e:
        logger.error(f"Error in deep keyframe search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Nhan phan hoi tu nguoi dung cho mot truy van.
    Su dung LLM Reflection de Agent tu rut kinh nghiem cho luot chat sau.
    """
    try:
        from src.services.feedback_service import feedback_service
        import asyncio
        result = await asyncio.to_thread(feedback_service.process_feedback, request)
        return result
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))