# Phase 2 Task 4 Report: Expose Elasticsearch via FastAPI Endpoints

## Status: ✅ Completed and Tested

---

## What Changed

1. **Service Layer Updates (`src/services/user_service.py`)**:
   - Added a `get_elastic_processor()` lazy getter to ensure the Elasticsearch client is only initialized when a text search endpoint is first hit (saving memory and startup time).
   - Added `getTextSearchOCR()` and `getTextSearchASR()` wrapper functions.

2. **API Router Updates (`src/api/routers/search_router.py`)**:
   - Added `POST /users/ocrsearch`: Accepts a text query and returns matching frames based on OCR text.
   - Added `POST /users/asrsearch`: Accepts a text query and returns matching frames based on ASR transcriptions.
   - Modified `POST /users/ocrandodsearch`: Instead of throwing a `501 Not Implemented` error, this legacy endpoint now acts as an alias to `/users/ocrsearch`. This ensures backward compatibility if the React UI still calls this route.
   - All endpoints strictly enforce the unified `BaseResponse` schema with Pydantic validation on the input `TextSearchRequest`.

3. **API Tests (`tests/test_phase2_task4.py`)**:
   - Created a comprehensive test suite using FastAPI's `TestClient` and `unittest.mock.patch`.
   - Simulated Elasticsearch responses to ensure the FastAPI routing, Pydantic validation, and JSON serialization function correctly.
   - Fixed a strict validation edge case where the legacy alias test initially failed `400 Bad Request` because it lacked the required `topk` parameter.

---

## Why These Decisions

- **Modularity**: By keeping the ES logic in `elastic_processing.py`, the Service wrappers in `user_service.py`, and the HTTP mapping in `search_router.py`, we maintain strict architectural boundaries (Layered Architecture).
- **Backward Compatibility**: Modifying `/users/ocrandodsearch` to serve OCR results allows the Frontend team to test the new features immediately without modifying their Axios calls.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase2_task4.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 3 items

tests/test_phase2_task4.py::TestElasticAPIEndpoints::test_asr_search_endpoint PASSED
tests/test_phase2_task4.py::TestElasticAPIEndpoints::test_ocr_search_endpoint PASSED
tests/test_phase2_task4.py::TestElasticAPIEndpoints::test_ocrandodsearch_endpoint_alias PASSED

======================= 3 passed, 2 warnings in 20.20s =====================
```
