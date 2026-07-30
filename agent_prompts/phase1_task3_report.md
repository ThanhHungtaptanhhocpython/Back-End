# Phase 1 Task 3 Report: FastAPI Middleware (Logging, Request ID, Error Handling)

## Status: ✅ Completed and Tested

---

## What Changed

1. **Created `src/api/middleware.py`**:
   - Implemented `RequestLoggingMiddleware` (inheriting from `BaseHTTPMiddleware`) which:
     - Generates a unique UUID (`X-Request-ID`) for every incoming request.
     - Measures the execution time of the request (`duration=...ms`).
     - Uses Python's `logging` module to output a structured log message.
     - Injects the `X-Request-ID` into both `request.state` and the response headers.
   - Implemented `global_exception_handler` to catch all unhandled `Exception`s that bubble up from routes. Instead of returning a raw 500 HTML page or leaking a stack trace, it logs the stack trace internally and returns our unified schema:
     ```json
     {
       "success": false,
       "message": "Internal server error. Please try again later.",
       "data": {"items": [], "total_items": 0}
     }
     ```

2. **Updated `main.py`**:
   - Imported and attached `RequestLoggingMiddleware` using `app.add_middleware()`.
   - Registered `global_exception_handler` using `app.add_exception_handler(Exception, ...)`.

3. **Added Unit Tests (`tests/test_phase1_task3.py`)**:
   - Tested that standard endpoints return `x-request-id` in the response headers.
   - Verified that `logger.info` is called with the expected formatting (`method`, `path`, `status`, `duration`, `request_id`).
   - Created a deliberate crashing endpoint (`/test-error`) to verify that the `global_exception_handler` kicks in, returns status `500`, formats the JSON correctly, and still includes the `X-Request-ID` in the response header.

---

## Why These Decisions

- **Structured Logging over Print**: Using `logger.info` integrates seamlessly with modern log aggregators. Capturing `duration` allows us to monitor which search queries are taking too long.
- **Request ID Tracking**: Having an `X-Request-ID` is essential for debugging. If a frontend user reports an error, they can provide the `X-Request-ID` from the response headers, and we can immediately search our backend logs for that exact transaction.
- **Header Injection in Exception Handler**: Starlette's `BaseHTTPMiddleware` executes "outside" the exception handling stack. To guarantee that a crashed request still returns an `X-Request-ID` to the client, the `global_exception_handler` explicitly attaches the header to its `JSONResponse`.
- **Catch-all 500 JSON schema**: Standardizing the 500 error response prevents the frontend from crashing when it tries to parse a raw HTML traceback page.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase1_task3.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 3 items

tests/test_phase1_task3.py::TestMiddleware::test_request_id_in_headers PASSED
tests/test_phase1_task3.py::TestMiddleware::test_request_logging PASSED
tests/test_phase1_task3.py::TestGlobalExceptionHandler::test_global_exception_handler_returns_standard_schema PASSED

============================= 3 passed in 0.59s ============================
```

The middleware stack is now fully functional and provides a robust foundation for production debugging.
