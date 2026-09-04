"""FastAPI entry point for the AIC Search API.

Start with:
    uvicorn main:app --reload          (development)
    python main.py                     (uses settings from .env)
"""

import logging
import os
import sys

# Vietnamese query text is logged/printed all over the search path. On Windows a
# cp1252 stdout/stderr raises UnicodeEncodeError on those characters, which would
# turn a normal log line into an unhandled 500. Force UTF-8 with a safe fallback.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api.routers.health_router import router as health_router
from src.api.routers.search_router import router as search_router
from src.api.routers.chat_router import router as chat_router
from src.api.routers.video_router import router as video_router
from src.api.routers.settings_router import router as settings_router
from src.api.middleware import RequestLoggingMiddleware, global_exception_handler
from src.config.settings import get_settings

settings = get_settings()

# Configure root logger
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(levelname)s: %(message)s",
)

@asynccontextmanager
async def _lifespan(_app: "FastAPI"):
    """Best-effort: sync the active backend's cloud artifacts + warm its
    retriever in the background so the first search isn't blocked. Gated by
    CLOUD_ASSETS_AUTOSYNC; a no-op unless cloud assets are enabled."""
    try:
        from src.services.startup_warm import warm_active_backend_in_background

        warm_active_backend_in_background(settings)
    except Exception as exc:  # noqa: BLE001 - never let warm-up break startup
        logging.getLogger(__name__).debug("startup warm not started: %s", exc)
    yield
    # -- shutdown -----------------------------------------------------------
    try:
        from src.services.assets.keyframe_prefetch import shutdown as _kf_prefetch_shutdown

        _kf_prefetch_shutdown()
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.services.assets import cloud_enabled, get_keyframe_cache

        if cloud_enabled(settings):
            get_keyframe_cache().flush()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(
    title="AIC Search API",
    description="Multimodal video keyframe retrieval backend powered by BEiT-3 and Faiss.",
    version="1.0.0",
    lifespan=_lifespan,
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
    # Never echo the body of a management-API request: it may carry secrets.
    if not request.url.path.startswith(("/settings", "/users/settings")):
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

async def _maybe_backend_preparing(request: Request, exc: Exception) -> JSONResponse:
    """A search that lands mid startup-warm (cloud sync / model load still in
    flight) gets a clear, retryable 503 -- never a generic 500."""
    from src.services.retrieval_backend import BackendPreparingError

    if isinstance(exc, BackendPreparingError):
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "5"},
            content={
                "success": False,
                "message": str(exc),
                "detail": "backend_preparing",
                "data": {"items": [], "total_items": 0},
            },
        )
    return await global_exception_handler(request, exc)


# Register global catch-all exception handler (backend-preparing -> 503, else 500)
app.add_exception_handler(Exception, _maybe_backend_preparing)


app.include_router(health_router, prefix="")
# search_router is intentionally registered under both prefixes: the frontend
# calls search endpoints via an axios baseURL ending in "/users" (services/axios.js,
# backendSearch.js) while copilotService.js strips that suffix and hits unprefixed
# paths. Removing either registration will break one of those clients.
app.include_router(search_router, prefix="/users")
app.include_router(search_router, prefix="")
# Video playback / frame capture: same dual-prefix treatment so the frontend
# works whether or not its API base URL ends in "/users".
app.include_router(video_router, prefix="/users")
app.include_router(video_router, prefix="")
app.include_router(chat_router)
# Local management API (loopback-only). Dual-prefixed like the search router so
# it works whether or not the frontend API base URL ends in "/users".
app.include_router(settings_router, prefix="/users")
app.include_router(settings_router, prefix="")


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

# ---------------------------------------------------------------------------
# Local-legacy keyframe lookup helpers.
#
# These serve the OLD on-disk dataset layout (``<split>/<video_id>/000000.webp``,
# named by source-frame index) and are used ONLY when cloud assets are OFF.
# In cloud mode ``serve_keyframe`` goes straight to the LRU cache + Azure and
# never touches ``KEYFRAMES_ROOT`` -- the cloud "fine keyframes" set has a
# different naming/extension (``keyframe_0016.jpg``) and letting the local
# lookup answer for it served frame 0 of every video for every result.
# ---------------------------------------------------------------------------

_map_keyframes_cache: dict[str, Any] = {}


def _load_keyframe_map(video_id: str):
    """[local-legacy] map-keyframes CSV for ``video_id`` (n -> frame_idx)."""
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


_KEYFRAME_CACHE_CONTROL = "public, max-age=604800, immutable"


def _keyframe_video_id(parts: tuple[str, ...]) -> str:
    for part in parts:
        upper = part.upper()
        if "_V" in upper or upper.startswith("V0"):
            return part
    for part in parts:
        if "_" in part and (part.startswith("L") or part.startswith("V")):
            return part
    return "UNKNOWN_VIDEO"


def _keyframe_missing_response(video_id: str, stem: str):
    """A miss is *visible*: 404 + ``no-store`` so the browser never caches it,
    the client's ``onError`` fires, and the cell self-heals on a later reload.
    The SVG body is only a friendly fallback for a direct hit in a browser tab."""
    from fastapi.responses import Response

    return Response(
        content=_make_svg_placeholder(video_id, f"Frame {stem}"),
        media_type="image/svg+xml",
        status_code=404,
        headers={"Cache-Control": "no-store", "X-Keyframe-Status": "missing"},
    )


@app.get("/keyframes/{image_path:path}")
def serve_keyframe(image_path: str):
    """Serve a keyframe image.

    Cloud mode (``CLOUD_ASSETS_ENABLED`` + a cloud provider): the image comes
    *only* from the LRU cache + cloud store (via the keyframe-prefetch helper).
    ``KEYFRAMES_ROOT`` is never consulted.

    Local-legacy mode (cloud assets off, on-disk BEiT3 dataset): the tolerant
    ``KEYFRAMES_ROOT`` lookup -- minus the two fallbacks that returned a
    *wrong* image on a miss (nearest-numeric-frame, and "first file in the
    folder"). A wrong image is worse than a visible miss.

    Sync ``def`` on purpose: the I/O here is blocking, so Starlette must run it
    in a worker thread instead of on the event loop that also serves search.
    """
    from pathlib import Path
    from fastapi.responses import FileResponse

    parts = Path(image_path).parts
    filename = parts[-1] if parts else image_path
    stem = Path(filename).stem
    video_id = _keyframe_video_id(parts)

    from src.services.assets import cloud_enabled

    if cloud_enabled(settings):
        cloud_file = None
        try:
            from src.services.assets.keyframe_prefetch import fetch_blocking

            cloud_file = fetch_blocking(image_path)
        except Exception:  # noqa: BLE001 - never let this break image serving
            cloud_file = None
        if cloud_file is not None and cloud_file.is_file():
            return FileResponse(
                str(cloud_file), headers={"Cache-Control": _KEYFRAME_CACHE_CONTROL}
            )
        return _keyframe_missing_response(video_id, stem)

    # --- Local-legacy dataset path (cloud assets off) -----------------------
    raw_root = str(settings.get_keyframes_root() or "").strip('"\'')
    if raw_root:
        root_path = Path(raw_root)
        if root_path.exists():
            for target in (root_path / image_path, root_path / "keyframes" / image_path):
                if target.is_file():
                    return FileResponse(
                        str(target), headers={"Cache-Control": _KEYFRAME_CACHE_CONTROL}
                    )

            try:
                requested_number = int(stem)
            except ValueError:
                requested_number = None

            for vdir in _candidate_video_dirs(root_path, parts, video_id):
                if requested_number is not None:
                    mapped_frame_idx = _lookup_frame_idx_from_keyframe_number(video_id, requested_number)
                    numeric_candidates = (
                        [mapped_frame_idx, requested_number]
                        if mapped_frame_idx is not None
                        else [requested_number]
                    )
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
                                return FileResponse(
                                    str(test_file),
                                    headers={"Cache-Control": _KEYFRAME_CACHE_CONTROL},
                                )

                for ext in [".webp", ".jpg", ".png", ".jpeg"]:
                    test_file = vdir / f"{stem}{ext}"
                    if test_file.is_file():
                        return FileResponse(
                            str(test_file),
                            headers={"Cache-Control": _KEYFRAME_CACHE_CONTROL},
                        )

    return _keyframe_missing_response(video_id, stem)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
