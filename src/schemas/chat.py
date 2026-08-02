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
