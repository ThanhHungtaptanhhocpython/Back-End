# Phase 1 - Task 3: Add FastAPI Middleware (Logging, Error Handling, Request Timing)

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules.
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Background
The FastAPI app (`main.py`) currently has minimal middleware — only CORS and a `RequestValidationError` handler. For production readiness, we need structured logging, request timing, and a global exception handler to prevent raw 500 errors from leaking stack traces to clients.

## 3. Objective
Add production-grade middleware to the FastAPI application.

## 4. Requirements

### 4.1 Structured Logging Middleware
- Create a middleware that logs every incoming request with: HTTP method, path, status code, and response time in milliseconds.
- Use the Python `logging` module (not `print`).
- Log format example: `INFO: POST /users/singletextsearch 200 142ms`

### 4.2 Global Exception Handler
- Add a catch-all `Exception` handler in `main.py` that returns the standard error response schema instead of a raw 500 error.
- Response format:
  ```json
  {
    "success": false,
    "message": "Internal server error",
    "data": { "items": [], "total_items": 0 }
  }
  ```
- Log the full exception traceback using `logging.exception(...)` so developers can debug, but do NOT expose the traceback in the HTTP response.

### 4.3 Request ID (Optional Enhancement)
- Generate a unique `X-Request-ID` header for each request using `uuid4`.
- Include it in log messages so individual requests can be traced.
- Return it in the response headers.

### Important Constraints
- Place middleware logic in a new file `src/api/middleware.py` to keep `main.py` clean.
- Do NOT modify any existing router logic or service functions.
- Do NOT modify the Flask app (`app.py`) — this is FastAPI only.

## 5. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** Files created/modified.
- **Why:** The rationale behind each middleware (e.g., why structured logging over print, why catch-all exception handler).
- **Testing Plan:** Provide `curl` commands to verify: (1) response headers contain `X-Request-ID`, (2) server logs show timing info, (3) a deliberately broken endpoint returns 500 with the standard schema instead of raw traceback.
- **Automated Testing:** You must write a Python test script using `unittest` and FastAPI `TestClient`. The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_phase1_task3.py`). Test that: responses have `X-Request-ID` header, the `/health` endpoint returns 200 with timing logged, and a simulated unhandled exception returns the standard error schema. Run it to prove your code works.
