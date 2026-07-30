# Phase 1 - Task 2: FastAPI Endpoint Parity Testing vs Flask

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules.
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Background
The backend currently runs **two frameworks in parallel**: Flask (`app.py`) and FastAPI (`main.py`). The FastAPI routers in `src/api/routers/search_router.py` were ported from the Flask controller (`src/controllers/user_controller.py`). Before we can retire Flask, we must verify that **every FastAPI endpoint behaves identically** to its Flask counterpart — same request format, same response schema, same HTTP status codes.

## 3. Objective
Write a comprehensive parity test suite that sends the same requests to both Flask and FastAPI test clients and asserts identical behavior.

## 4. Requirements
- Inspect `src/controllers/user_controller.py` (Flask routes) and `src/api/routers/search_router.py` (FastAPI routes).
- For each endpoint, verify:
  - **`POST /users/singletextsearch`**: JSON body `{"query": "...", "topk": N}` → 200 OK with standard response.
  - **`POST /users/qnasearch`**: JSON body `{"query": "...", "topk": N}` → 200 OK with standard response.
  - **`POST /users/imagesearch`**: Form data with `image` file or `faiss_index` → 200 OK. Missing both → 400. Invalid `topk` → 400.
  - **`POST /users/temporalsearch`**: JSON body `{"query": [{"query": "event1"}, {"query": "event2"}], "topk": N}` → 200 OK. Empty/invalid query → 400.
  - **`POST /users/ocrandodsearch`**: → 501 Not Implemented.
  - **`GET /health`**: → 200 OK (FastAPI only, but should be tested).
- Check that error responses from FastAPI match the unified schema: `{"success": false, "message": "...", "data": {"items": [], "total_items": 0}}`.
- If you find any **discrepancies** between Flask and FastAPI behavior, document them clearly and fix the FastAPI route to match Flask.

## 5. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** Any fixes applied to `search_router.py` or schemas to achieve parity.
- **Why:** The rationale behind the fixes.
- **Parity Matrix:** A table showing each endpoint, Flask behavior, FastAPI behavior, and whether they match.
- **Automated Testing:** You must write a Python test script using `unittest` and FastAPI `TestClient` (from `fastapi.testclient`). The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_phase1_task2.py`). Mock heavy dependencies (`faiss`, `open_clip`, `torch`) at `sys.modules` level before importing the app (same pattern as `tests/test_task1_imagesearch.py`). Run it to prove parity.
