from fastapi import APIRouter, HTTPException, Depends
from src.schemas.chat import ChatRequest, ChatResponse, FeedbackRequest
from src.agent.memory_manager import memory_manager
import logging

router = APIRouter(
    prefix="/chat",
    tags=["Chat & Agent"]
)

logger = logging.getLogger(__name__)

@router.post("/conversational_kis", response_model=ChatResponse)
async def conversational_kis(request: ChatRequest):
    """
    Conversational KIS endpoint.
    Maintains multi-turn context using memory_manager and LLM Planner.
    """
    try:
        from src.agent.llm_planner import execute_chat_turn
        
        session_id = request.session_id
        user_message = request.message
        
        # 1. Gọi LLM Planner (Tự động cập nhật history)
        agent_result = execute_chat_turn(session_id, user_message)
        
        history = memory_manager.get_messages(session_id)
        
        return ChatResponse(
            success=agent_result.get("success", True),
            session_id=session_id,
            response=agent_result.get("response", ""),
            data={"history_length": len(history)}
        )
    except Exception as e:
        logger.error(f"Error in conversational KIS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Nhận phản hồi từ người dùng cho một truy vấn.
    Sử dụng LLM Reflection để Agent tự rút kinh nghiệm cho lượt chat sau.
    """
    try:
        from src.services.feedback_service import feedback_service
        result = feedback_service.process_feedback(request)
        return result
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

