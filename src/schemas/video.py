"""Request/response models for the video playback + frame-capture endpoints."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PlaybackItem(BaseModel):
    """Everything the frontend player needs to open a video at a frame."""

    video_id: str
    watch_url: str
    fps: float
    duration_seconds: float
    playback_offset_seconds: float
    # Populated when the request carried a ``frame_idx``; the player start time
    # (seconds) that displays that dataset frame.
    frame_idx: Optional[int] = None
    start_seconds: Optional[float] = None


class PlaybackData(BaseModel):
    items: List[PlaybackItem]
    total_items: int


class PlaybackResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
    data: PlaybackData


class CaptureRequest(BaseModel):
    playback_time_seconds: float = Field(
        ...,
        description="Current player time in seconds when the user pressed Capture.",
    )


class CaptureResultModel(BaseModel):
    video_id: str
    playback_time_seconds: float
    source_time_seconds: float
    fps: float
    frame_idx: int
    # Populated only when an exact still was extracted for this frame: a
    # relative path served by the captured-frame route. ``None`` means the
    # frame is still valid/exportable but has no substitute preview image.
    preview_url: Optional[str] = None
    # Short human-readable reason preview extraction was skipped or failed.
    preview_error: Optional[str] = None


class CaptureResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
    data: CaptureResultModel
