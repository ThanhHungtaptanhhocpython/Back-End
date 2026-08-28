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
_cors_origins = settings.get_cors_origins()
_allow_all = "*" in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _cors_origins,
    # Wildcard origins cannot be combined with credentials per the CORS spec.
    allow_credentials=not _allow_all,
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
# search_router is intentionally registered under both prefixes: the frontend
# calls search endpoints via an axios baseURL ending in "/users" (services/axios.js,
# backendSearch.js) while copilotService.js strips that suffix and hits unprefixed
# paths. Removing either registration will break one of those clients.
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


def _load_keyframe_map(video_id: str):
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
    return _map_keyframes_cache.get(video_id)


def _lookup_frame_idx_from_keyframe_number(video_id: str, keyframe_number: int) -> int | None:
    df = _load_keyframe_map(video_id)
    if df is None or df.empty or "n" not in df.columns or "frame_idx" not in df.columns:
        return None
    exact = df[df["n"].astype(int) == int(keyframe_number)]
    if not exact.empty:
        return int(exact.iloc[0]["frame_idx"])
    diff = (df["n"].astype(int) - int(keyframe_number)).abs()
    return int(df.loc[diff.idxmin()]["frame_idx"])


def _candidate_video_dirs(root_path, parts, video_id: str):
    split = parts[0] if len(parts) > 1 else ""
    candidates = []
    if split:
        candidates.extend([
            root_path / split / video_id,
            root_path / "keyframes" / split / video_id,
        ])
        if "_" not in split and split.startswith("L"):
            candidates.extend(sorted(root_path.glob(f"{split}_*/{video_id}")))
            candidates.extend(sorted((root_path / "keyframes").glob(f"{split}_*/{video_id}")) if (root_path / "keyframes").is_dir() else [])
    candidates.extend([
        root_path / video_id,
        root_path / "keyframes" / video_id,
    ])
    candidates.extend(sorted(root_path.glob(f"*/{video_id}")))
    return [path for path in candidates if path and path.is_dir()]


def _closest_numeric_frame_file(vdir, target_number: int):
    best = None
    best_diff = None
    for file_path in vdir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() not in {".webp", ".jpg", ".jpeg", ".png"}:
            continue
        try:
            number = int(file_path.stem)
        except ValueError:
            continue
        diff = abs(number - target_number)
        if best is None or diff < best_diff:
            best = file_path
            best_diff = diff
    return best


@app.get("/keyframes/{image_path:path}")
async def serve_keyframe(image_path: str):
    """Serve keyframe image file with tolerant video-folder and frame-id fallback."""
    from pathlib import Path
    from fastapi.responses import FileResponse, Response

    parts = Path(image_path).parts
    filename = parts[-1] if parts else image_path
    stem = Path(filename).stem

    video_id = "UNKNOWN_VIDEO"
    for part in parts:
        upper = part.upper()
        if "_V" in upper or upper.startswith("V0"):
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
            exact_candidates = [
                root_path / image_path,
                root_path / "keyframes" / image_path,
            ]
            for target in exact_candidates:
                if target.is_file():
                    return FileResponse(str(target))

            try:
                requested_number = int(stem)
            except ValueError:
                requested_number = None

            for vdir in _candidate_video_dirs(root_path, parts, video_id):
                if requested_number is not None:
                    mapped_frame_idx = _lookup_frame_idx_from_keyframe_number(video_id, requested_number)
                    numeric_candidates = []
                    if mapped_frame_idx is not None:
                        numeric_candidates.extend([mapped_frame_idx, requested_number])
                    else:
                        numeric_candidates.append(requested_number)

                    for number in numeric_candidates:
                        for candidate_name in [
                            f"{number:06d}.webp", f"{number:06d}.jpg",
                            f"{number:05d}.webp", f"{number:05d}.jpg",
                            f"{number:04d}.webp", f"{number:04d}.jpg",
                            f"{number:03d}.webp", f"{number:03d}.jpg",
                            f"{number}.webp", f"{number}.jpg",
                        ]:
                            test_file = vdir / candidate_name
                            if test_file.is_file():
                                return FileResponse(str(test_file))

                    closest_target = mapped_frame_idx if mapped_frame_idx is not None else requested_number
                    closest = _closest_numeric_frame_file(vdir, closest_target)
                    if closest is not None:
                        return FileResponse(str(closest))

                for ext in [".webp", ".jpg", ".png", ".jpeg"]:
                    test_file = vdir / f"{stem}{ext}"
                    if test_file.is_file():
                        return FileResponse(str(test_file))

                all_frames = sorted([f for f in vdir.iterdir() if f.is_file()])
                if all_frames:
                    return FileResponse(str(all_frames[0]))

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
