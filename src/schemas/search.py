from typing import Literal, Optional
from pydantic import BaseModel, Field

class TextSearchRequest(BaseModel):
    query: str = Field(default="")
    topk: int = Field(..., gt=0)
    clip: Optional[bool] = None
    clipv2: Optional[bool] = None

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
