from typing import Optional
from pydantic import BaseModel, Field

class TextSearchRequest(BaseModel):
    query: str = Field(default="")
    topk: int = Field(..., gt=0)

class CaptureSimilarRequest(BaseModel):
    """Body for the captured-frame "Similar" search.

    Carries no image bytes and no ``faiss_index``: the server re-encodes the
    exact cached preview still for ``(video_id, frame_idx)`` with BEiT3.
    """
    topk: int = Field(default=100, gt=0)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    from_lang: str = Field(default="vi")
    to_lang: str = Field(default="en")

class TranslateResponse(BaseModel):
    success: bool = True
    translated_text: str
    from_lang: str
    to_lang: str
    translated: Optional[bool] = None
    provider: Optional[str] = None
