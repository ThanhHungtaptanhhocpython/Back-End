from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    session_id: str
    message: str
    topk: Optional[int] = 100

class ChatResponse(BaseModel):
    success: bool
    session_id: str
    response: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    feedback_score: int  # e.g., 1 for thumbs up, -1 for thumbs down
    feedback_text: Optional[str] = None
    video_key: Optional[str] = None
