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
app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
