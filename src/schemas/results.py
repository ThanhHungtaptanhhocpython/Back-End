from typing import List, Any, Optional
from pydantic import BaseModel

class DataResponse(BaseModel):
    items: List[Any]
    total_items: int

class BaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: DataResponse
