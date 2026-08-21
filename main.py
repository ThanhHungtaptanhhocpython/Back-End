"""FastAPI entry point for the AIC Search API.

Start with:
    uvicorn main:app --reload          (development)
    python main.py                     (uses settings from .env)
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api.routers.health_router import router as health_router
from src.api.routers.search_router import router as search_router
from src.api.routers.chat_router import router as chat_router
from src.api.middleware import RequestLoggingMiddleware, global_exception_handler
from src.config.settings import get_settings

settings = get_settings()

# Configure root logger
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(levelname)s: %(message)s",
)

app = FastAPI(
    title="AIC Search API",
    description="Multimodal video keyframe retrieval backend powered by OpenCLIP and Faiss.",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom logging/timing middleware
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a standard error response for Pydantic validation failures."""
    print(f"Validation Error: {exc.errors()}")
    print(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": f"Validation Error: {exc.errors()}",
            "data": {
                "items": [],
                "total_items": 0,
            },
        },
    )

# Register global catch-all exception handler
app.add_exception_handler(Exception, global_exception_handler)


app.include_router(health_router, prefix="")
app.include_router(search_router, prefix="/users")
app.include_router(search_router, prefix="")
app.include_router(chat_router)


keyframes_root = settings.get_keyframes_root()


def _make_svg_placeholder(video_id: str, frame_info: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#0f172a"/>
          <stop offset="100%" stop-color="#1e293b"/>
        </linearGradient>
      </defs>
      <rect width="320" height="180" fill="url(#bg)"/>
      <rect x="12" y="12" width="296" height="156" rx="8" fill="none" stroke="#334155" stroke-width="1.5" stroke-dasharray="4 4"/>
      <circle cx="160" cy="65" r="22" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
      <polygon points="154,55 172,65 154,75" fill="#38bdf8"/>
      <text x="160" y="112" fill="#f8fafc" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="13" font-weight="600" text-anchor="middle">{video_id}</text>
      <text x="160" y="132" fill="#94a3b8" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="11" text-anchor="middle">{frame_info}</text>
      <text x="160" y="152" fill="#64748b" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="9" text-anchor="middle">Vector Search Match</text>
    </svg>"""


from typing import Any

_map_keyframes_cache: dict[str, Any] = {}


def _lookup_keyframe_number(video_id: str, frame_idx: int) -> int | None:
    """Look up keyframe index 'n' (e.g. 185 -> 185.jpg) from map-keyframes CSV."""
    import pandas as pd
    from pathlib import Path

    if video_id not in _map_keyframes_cache:
        map_file = Path(__file__).resolve().parent / "src" / "dict" / "map-keyframes" / f"{video_id}.csv"
        if map_file.is_file():
            try:
                _map_keyframes_cache[video_id] = pd.read_csv(map_file)
            except Exception:
                _map_keyframes_cache[video_id] = None
        else:
            _map_keyframes_cache[video_id] = None

    df = _map_keyframes_cache.get(video_id)
    if df is not None and not df.empty and "frame_idx" in df.columns and "n" in df.columns:
        diff = (df["frame_idx"] - frame_idx).abs()
        closest_row = df.loc[diff.idxmin()]
        return int(closest_row["n"])
    return None


@app.get("/keyframes/{image_path:path}")
async def serve_keyframe(image_path: str):
    """Serve keyframe image file with multi-directory fallback or dynamic SVG badge."""
    from pathlib import Path
    from fastapi.responses import FileResponse, Response

    parts = Path(image_path).parts
    filename = parts[-1] if parts else image_path
    stem = Path(filename).stem

    video_id = "UNKNOWN_VIDEO"
    for part in parts:
        if "_V" in part.upper() or part.upper().startswith("V0"):
            video_id = part
            break
    if video_id == "UNKNOWN_VIDEO":
        for part in parts:
            if "_" in part and (part.startswith("L") or part.startswith("V")):
                video_id = part
                break

    raw_root = str(settings.get_keyframes_root() or "").strip('"\'')
    if raw_root:
        root_path = Path(raw_root)
        if root_path.exists():
            # 1. Exact relative match
            target = root_path / image_path
            if target.is_file():
                return FileResponse(str(target))

            # 2. Check in 'keyframes' subfolder
            target_nested = root_path / "keyframes" / image_path
            if target_nested.is_file():
                return FileResponse(str(target_nested))

            # 3. Match video directory
                candidate_dirs = [
                    root_path / "keyframes" / parts[0] / video_id if len(parts) > 1 else None,
                    root_path / parts[0] / video_id if len(parts) > 1 else None,
                    root_path / "keyframes" / video_id,
                    root_path / video_id,
                ]
                for vdir in candidate_dirs:
                    if vdir and vdir.is_dir():
                        # Try map-keyframes CSV lookup
                        try:
                            num = int(stem)
                            n_keyframe = _lookup_keyframe_number(video_id, num)
                            if n_keyframe is not None:
                                for candidate_name in [f"{n_keyframe:03d}.jpg", f"{n_keyframe}.jpg", f"{n_keyframe:04d}.jpg"]:
                                    test_file = vdir / candidate_name
                                    if test_file.is_file():
                                        return FileResponse(str(test_file))
                        except ValueError:
                            pass

                        for ext in [".jpg", ".webp", ".png", ".jpeg"]:
                            test_file = vdir / f"{stem}{ext}"
                            if test_file.is_file():
                                return FileResponse(str(test_file))

                        try:
                            num = int(stem)
                            for candidate_name in [f"{num:03d}.jpg", f"{num+1:03d}.jpg", f"{num:04d}.jpg", f"{num:06d}.jpg"]:
                                test_file = vdir / candidate_name
                                if test_file.is_file():
                                    return FileResponse(str(test_file))
                        except ValueError:
                            pass

                        all_frames = sorted([f for f in vdir.iterdir() if f.is_file()])
                        if all_frames:
                            return FileResponse(str(all_frames[0]))

    # Fallback to sleek SVG badge when keyframe image file is not on local disk
    svg_content = _make_svg_placeholder(video_id, f"Frame {stem}")
    return Response(content=svg_content, media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
