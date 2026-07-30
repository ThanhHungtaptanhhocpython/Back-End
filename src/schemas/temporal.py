from typing import List, Optional
from pydantic import BaseModel, Field

class TemporalEvent(BaseModel):
    query: str = Field(..., min_length=1)

class TemporalSearchRequest(BaseModel):
    query: List[TemporalEvent] = Field(..., min_items=1)
    topk: int = Field(default=100, gt=0)
    cascaded: Optional[bool] = None
