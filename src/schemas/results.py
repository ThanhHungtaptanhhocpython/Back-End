from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field

class DataResponse(BaseModel):
    items: List[Any]
    total_items: int

class BaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: DataResponse

class AgentSearchResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    response: Optional[str] = None
    data: DataResponse
    plan: Dict[str, Any] = Field(default_factory=dict)

