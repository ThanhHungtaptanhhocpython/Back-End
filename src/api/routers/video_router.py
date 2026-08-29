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
from fastapi.responses import FileResponse

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


@router.get("/videos/captures/{video_id}/{frame_idx}.webp")
def serve_captured_frame(video_id: str, frame_idx: int) -> Response:
    """Serve a previously extracted captured-frame still (WebP)."""
    from src.services.video_frame_preview_service import (
        FramePreviewError,
        get_video_frame_preview_service,
    )

    try:
        path = get_video_frame_preview_service().get_existing(video_id, frame_idx)
    except FramePreviewError:
        path = None
    if path is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    return FileResponse(
        str(path),
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _build_frame_preview(video_id: str, frame_idx: int) -> tuple[Optional[str], Optional[str]]:
    """Best-effort exact still for ``frame_idx``; ``(preview_url, preview_error)``.

    Never raises: a preview must not break the capture response.
    """
    from src.services.video_frame_preview_service import (
        FramePreviewError,
        get_video_frame_preview_service,
    )
    from src.services.video_playback_service import get_video_playback_service

    try:
        playback = get_video_playback_service()
        meta = playback.get_metadata(video_id)
        # Extract at the canonical playback timestamp for this exact frame,
        # not wherever the user happened to pause between frames.
        target_seconds = playback.playback_start_seconds(video_id, frame_idx)
        key = get_video_frame_preview_service().get_or_create(
            video_id=meta.video_id,
            frame_idx=frame_idx,
            watch_url=meta.watch_url,
            target_seconds=target_seconds,
        )
        return f"videos/captures/{key}", None
    except FramePreviewError as exc:
        logger.info("frame preview unavailable for %s#%s: %s", video_id, frame_idx, exc)
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - a preview must never break capture
        logger.warning("frame preview crashed for %s#%s: %s", video_id, frame_idx, exc, exc_info=True)
        return None, "Preview extraction failed unexpectedly."


@router.post("/videos/{video_id}/capture", response_model=CaptureResponse)
def capture_video_frame(
    video_id: str,
    request: CaptureRequest,
    response: Response,
) -> CaptureResponse:
    """Convert the current player time into a 0-based dataset frame index.

    On success this also tries to attach an exact server-extracted still
    (``preview_url``); if extraction is unavailable the frame index is still
    returned with ``preview_url: null`` and a ``preview_error`` reason.
    """
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

    preview_url, preview_error = _build_frame_preview(result.video_id, result.frame_idx)

    return CaptureResponse(
        success=True,
        data=CaptureResultModel(
            video_id=result.video_id,
            playback_time_seconds=result.playback_time_seconds,
            source_time_seconds=result.source_time_seconds,
            fps=result.fps,
            frame_idx=result.frame_idx,
            preview_url=preview_url,
            preview_error=preview_error,
        ),
    )
