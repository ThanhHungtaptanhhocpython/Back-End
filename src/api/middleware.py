"""FastAPI middleware for logging, request timing, and error handling."""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs request timing and injects a Request ID."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate a unique Request ID
        request_id = str(uuid.uuid4())
        
        # Attach it to the request state so routes can access it if needed
        request.state.request_id = request_id
        
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
        except Exception:
            # If an exception escapes up to here, we still want to record time.
            # The exception handler will catch it after this middleware in the ASGI chain,
            # or it will bubble up. We just let it bubble up to the exception_handler.
            raise
        finally:
            process_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Fast-path for getting status code if response was created
            status_code = getattr(response, "status_code", 500) if 'response' in locals() else 500
            
            logger.info(
                f"{request.method} {request.url.path} "
                f"status={status_code} "
                f"duration={process_time_ms:.2f}ms "
                f"request_id={request_id}"
            )

        # Inject Request ID into response headers
        response.headers["X-Request-ID"] = request_id
        return response


def _cors_headers_for(request: Request) -> dict:
    """Echo CORS headers on error responses.

    ``CORSMiddleware`` only decorates responses that pass *through* it; a
    response produced by this catch-all handler is created outside that
    middleware, so without this a 500 reaches the browser with no
    ``Access-Control-Allow-Origin`` and ``fetch`` rejects with an opaque
    "Failed to fetch" instead of surfacing the real error.
    """
    origin = request.headers.get("origin")
    if not origin:
        return {}
    try:
        from src.config.settings import get_settings

        allowed = get_settings().get_cors_origins()
    except Exception:
        allowed = []
    if "*" in allowed:
        return {"Access-Control-Allow-Origin": "*"}
    if origin in allowed:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
        }
    return {}


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler to prevent leaking raw tracebacks to clients.

    Returns the standard BaseResponse schema with status 500.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the full traceback for the developer
    logger.exception(f"Unhandled Exception on {request.url.path} (request_id={request_id}): {exc}")

    # Return a sanitized error response to the client
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error. Please try again later.",
            "data": {
                "items": [],
                "total_items": 0,
            },
        },
        headers={"X-Request-ID": request_id, **_cors_headers_for(request)},
    )
