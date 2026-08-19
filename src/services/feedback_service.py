import logging
from src.schemas.chat import FeedbackRequest
from src.agent.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class FeedbackService:
    """
    Handles user feedback for Conversational KIS.
    Logs feedback and injects reflection context into the LLM Memory so the 
    Agent can adapt in subsequent turns.
    """
    def __init__(self):
        # In a real scenario, we might store this in a DB or use it to update RRF parameters
        self.feedback_logs = []

    def process_feedback(self, request: FeedbackRequest) -> dict:
        logger.info(f"Received feedback: {request.dict()}")
        self.feedback_logs.append(request.dict())
        
        # Determine sentiment
        sentiment = "tích cực (👍)" if request.feedback_score > 0 else "tiêu cực (👎)"
        
        # Construct reflection message for the LLM
        reflection_msg = f"[HỆ THỐNG GHI NHẬN PHẢN HỒI]: Người dùng vừa đánh giá {sentiment} cho kết quả tìm kiếm gần nhất."
        if request.video_key:
            reflection_msg += f" (Video được đánh giá: {request.video_key})."
        if request.feedback_text:
            reflection_msg += f" Ghi chú của người dùng: '{request.feedback_text}'."
        
        reflection_msg += " Hãy rút kinh nghiệm và điều chỉnh cách tìm kiếm/trả lời ở lượt tiếp theo."
        
        # Inject into memory as a system/user prompt so the Agent sees it next time
        try:
            memory_manager.add_user_message(request.session_id, reflection_msg)
        except Exception as e:
            logger.error(f"Failed to inject feedback into memory: {e}")
            return {"success": False, "message": str(e)}

        return {"success": True, "message": "Feedback processed and injected into LLM memory for reflection."}

feedback_service = FeedbackService()
