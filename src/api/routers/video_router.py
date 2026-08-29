"""Video playback + frame-capture endpoints.

These routes are deliberately stateless and never initialise a search model.
They convert between YouTube playback time and dataset frame indices using the
committed ``map-keyframes`` FPS data and the ``media-info`` watch URLs.

The router is registered under both ``/users`` and the root prefix (see
``main.py``) so it works regardless of whether the frontend's API base URL
carries a trailing ``/users`` segment.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Response, status

from src.schemas.video import (
    CaptureRequest,
    CaptureResponse,
    CaptureResultModel,
    PlaybackData,
    PlaybackItem,
    PlaybackResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _status_for(exc: Exception) -> int:
    from src.services.video_playback_service import (
        VideoMetadataError,
        VideoNotFoundError,
        VideoRequestError,
    )

    if isinstance(exc, VideoNotFoundError):
        return status.HTTP_404_NOT_FOUND
    if isinstance(exc, (VideoMetadataError, VideoRequestError)):
        return status.HTTP_400_BAD_REQUEST
    return status.HTTP_400_BAD_REQUEST


@router.get("/videos/{video_id}/playback", response_model=PlaybackResponse)
def get_video_playback(
    video_id: str,
    response: Response,
    frame_idx: Optional[int] = None,
) -> PlaybackResponse:
    """Return playback metadata for ``video_id``.

    When ``frame_idx`` is supplied, ``start_seconds`` is the player time that
    shows that dataset frame.
    """
    from src.services.video_playback_service import (
        VideoPlaybackError,
        get_video_playback_service,
    )

    service = get_video_playback_service()
    try:
        meta = service.get_metadata(video_id)
        start_seconds: Optional[float] = None
        if frame_idx is not None:
            start_seconds = service.playback_start_seconds(video_id, frame_idx)
    except VideoPlaybackError as exc:
        response.status_code = _status_for(exc)
        logger.info("playback lookup failed for %s: %s", video_id, exc)
        return PlaybackResponse(
            success=False,
            message=str(exc),
            data=PlaybackData(items=[], total_items=0),
        )

    item = PlaybackItem(
        video_id=meta.video_id,
        watch_url=meta.watch_url,
        fps=meta.fps,
        duration_seconds=meta.duration_seconds,
        playback_offset_seconds=meta.playback_offset_seconds,
        frame_idx=frame_idx,
        start_seconds=start_seconds,
    )
    return PlaybackResponse(success=True, data=PlaybackData(items=[item], total_items=1))


@router.post("/videos/{video_id}/capture", response_model=CaptureResponse)
def capture_video_frame(
    video_id: str,
    request: CaptureRequest,
    response: Response,
) -> CaptureResponse:
    """Convert the current player time into a 0-based dataset frame index."""
    from src.services.video_playback_service import (
        VideoPlaybackError,
        get_video_playback_service,
    )

    service = get_video_playback_service()
    try:
        result = service.capture(video_id, request.playback_time_seconds)
    except VideoPlaybackError as exc:
        response.status_code = _status_for(exc)
        logger.info("capture failed for %s @ %ss: %s", video_id, request.playback_time_seconds, exc)
        return CaptureResponse(
            success=False,
            message=str(exc),
            data=CaptureResultModel(
                video_id=video_id,
                playback_time_seconds=request.playback_time_seconds,
                source_time_seconds=request.playback_time_seconds,
                fps=0.0,
                frame_idx=-1,
            ),
        )

    return CaptureResponse(
        success=True,
        data=CaptureResultModel(
            video_id=result.video_id,
            playback_time_seconds=result.playback_time_seconds,
            source_time_seconds=result.source_time_seconds,
            fps=result.fps,
            frame_idx=result.frame_idx,
        ),
    )
