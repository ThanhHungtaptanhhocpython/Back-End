from typing import Literal, Optional
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
    from_lang: Literal["vi", "en"] = Field(default="vi")
    to_lang: Literal["vi", "en"] = Field(default="en")

class TranslateResponse(BaseModel):
    success: bool = True
    translated_text: str
    from_lang: str
    to_lang: str
    translated: Optional[bool] = None
    provider: Optional[str] = None
    # Structured outcome so clients can tell a real translation from a kept
    # original: "ok" | "invalid_input" | "provider_unavailable".
    status: Optional[str] = None
    # ``None`` on success; a stable reason code on failure (mirrors ``status``).
    error_code: Optional[str] = None
    # Short, non-sensitive explanation (never carries user text or secrets).
    detail: Optional[str] = None
