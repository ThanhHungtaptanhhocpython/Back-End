# Phase 2 Task 4: Expose Elasticsearch via FastAPI Endpoints

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Connect the `ElasticProcessor` to the API layer so the frontend can query OCR and ASR data.

## 3. Requirements
- Update `src/services/user_service.py` (or create `text_search_service.py`) to include lazy initialization for `ElasticProcessor` (similar to Faiss).
- Add wrapper functions in the service layer: `getTextSearchOCR(query, topk)`, `getTextSearchASR(query, topk)`.
- Update `src/api/routers/search_router.py`:
  - Implement `POST /users/ocrsearch` (New endpoint).
  - Implement `POST /users/asrsearch` (New endpoint).
  - Update the placeholder `POST /users/ocrandodsearch` to call the OCR search service.
- All endpoints must return the standardized `BaseResponse` schema with Pydantic input validation.
- Add API-level unit tests (`tests/test_phase2_task4.py`) using `TestClient` to verify the routes.

## 4. Expected Output & Reporting
- Generate a `phase2_task4_report.md` explaining the API additions and how they align with the unified schema.
