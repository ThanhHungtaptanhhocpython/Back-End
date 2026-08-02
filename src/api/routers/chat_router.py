from fastapi import APIRouter, HTTPException, Depends
from src.schemas.chat import ChatRequest, ChatResponse
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
    Maintains multi-turn context using memory_manager.
    """
    try:
        session_id = request.session_id
        user_message = request.message
        
        # 1. Add user message to memory
        memory_manager.add_user_message(session_id, user_message)
        
        # 2. Get history context
        history = memory_manager.get_messages(session_id)
        
        # 3. Call LLM / Agent Planner (Mock for Phase 1)
        # TODO (Phase 2): Pass history to LLM Planner along with Tools (Faiss, Elasticsearch)
        ai_response_text = f"Tác nhân AI đã nhận được yêu cầu: '{user_message}'. (Lịch sử: {len(history)} tin nhắn). Đang chờ tích hợp Tool ở Phase 2."
        
        # 4. Add AI response to memory
        memory_manager.add_ai_message(session_id, ai_response_text)
        
        return ChatResponse(
            success=True,
            session_id=session_id,
            response=ai_response_text,
            data={"history_length": len(history)}
        )
    except Exception as e:
        logger.error(f"Error in conversational KIS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
