import os
os.environ["FASTAPI_MODE"] = "1"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api.routers.health_router import router as health_router
from src.api.routers.search_router import router as search_router

app = FastAPI(title="AIC Search API (FastAPI)", description="FastAPI migration alongside Flask.")

# Configure CORS similar to Flask
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handler to mimic existing Flask 400 response format
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": f"Validation Error: {exc.errors()}",
            "data": {
                "items": [],
                "total_items": 0
            }
        }
    )

app.include_router(health_router, prefix="")
app.include_router(search_router, prefix="/users")
