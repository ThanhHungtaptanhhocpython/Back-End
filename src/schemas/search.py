from typing import Optional
from pydantic import BaseModel, Field

class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    topk: int = Field(..., gt=0)
    clip: Optional[bool] = None
    clipv2: Optional[bool] = None

# For /imagesearch, we receive fields via Form data, so we don't necessarily 
# validate JSON body for it, but if needed we can define it here.
# Since we will use Form(), we might not need an explicit BaseModel 
# for the image search request itself in the router.
