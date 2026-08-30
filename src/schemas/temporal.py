from typing import List, Optional
from pydantic import BaseModel, Field

class TemporalEvent(BaseModel):
    query: str = Field(..., min_length=1)

class TemporalSearchRequest(BaseModel):
    query: List[TemporalEvent] = Field(..., min_length=1)
    topk: int = Field(default=100, gt=0)
    cascaded: Optional[bool] = None
    # Whole-scene framing shared by every event. It is folded into each event
    # query for retrieval; it is never treated as an event of its own. Optional
    # so the historical ``{query: [...], topk}`` body stays valid.
    context: Optional[str] = None
