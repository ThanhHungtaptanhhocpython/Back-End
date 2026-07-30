# Phase 4 Task 3 Report: Expose Multimodal API

## Status: ✅ Completed and Tested

---

## What Changed

1. **New API Endpoint (`POST /users/multimodalsearch`)**:
   - Added a brand new endpoint in `src/api/routers/search_router.py`.
   - This endpoint acts as the single point of entry for the new AI Architecture.
   
2. **Endpoint Workflow**:
   - **Step 1**: It receives the raw `query` string from the Frontend via Pydantic's `TextSearchRequest`.
   - **Step 2**: It sends the `query` to the `QueryPlanner` (built in Task 2), which dissects it into `visual_query`, `ocr_query`, `asr_query`, and a dynamic `weights` dictionary.
   - **Step 3**: It passes all these components to `multimodal_search` in the `FusionService` (built in Task 1).
   - **Step 4**: The `FusionService` hits Faiss and Elasticsearch concurrently, normalizes the scores, outer-joins them by `faiss_id`, applies the dynamic weights, and sorts them.
   - **Step 5**: The results are wrapped in the standard `BaseResponse` and sent back to the client.

3. **API Tests (`tests/test_phase4_task3.py`)**:
   - Mocked both the `QueryPlanner` and `multimodal_search` using `unittest.mock.patch`.
   - Verified that sending a payload like `{"query": 'Cảnh sát "Police"', "topk": 50}` successfully traverses the API router logic and returns a HTTP 200 JSON response containing the `final_score` and `faiss_id`.

---

## Why These Decisions

- **Why a New Endpoint?**: Instead of silently overwriting `/users/singletextsearch`, creating `/users/multimodalsearch` allows the Frontend team to test the new AI logic side-by-side with the legacy logic. Once they are satisfied, they can deprecate the old endpoint.
- **Score Breakdown**: By ensuring the `score_breakdown` object reaches the Frontend, we enable powerful UX features. The UI can display a badge like "Matched by OCR" or "Matched by Audio" on the keyframes, helping the user understand *why* a specific image was returned.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase4_task3.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 1 item

tests/test_phase4_task3.py::TestMultimodalAPI::test_multimodal_endpoint PASSED

============================= 1 passed in 28.41s ===========================
```
