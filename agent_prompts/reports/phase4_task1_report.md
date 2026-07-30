# Phase 4 Task 1 Report: Fusion Service & Min-Max Normalization

## Status: ✅ Completed and Tested

---

## What Changed

1. **Created `src/services/fusion_service.py`**:
   - Built a central orchestration service that communicates with both Faiss and Elasticsearch concurrently.
   
2. **Implemented Min-Max Normalization (`normalize_scores`)**:
   - Created a helper function that takes raw scores (like Faiss distances or Elasticsearch BM25 scores), finds the `min` and `max`, and scales them perfectly to a `[0.0, 1.0]` range.
   - Handled edge cases (e.g., division by zero if all scores are identical).

3. **Implemented Merge & Rank (`merge_and_rank`)**:
   - Groups results from all 3 modalities using `faiss_id` (and `nearest_faiss_id` for ASR) as the primary key.
   - Calculates a `final_score` based on dynamic weights.
   - Automatically populates a `score_breakdown` object (e.g., `{"visual": 0.6, "ocr": 0.2, "asr": 0.0}`) for every single keyframe to enable deep debugging on the frontend UI.
   - Sorts the final merged list descending by `final_score`.

4. **Added Unit Tests (`tests/test_phase4_task1.py`)**:
   - Verified that the normalization math is strictly bounded between 0 and 1.
   - Verified that merging 3 disjoint sets of mock data accurately calculates the `final_score` and groups by `faiss_id`.

---

## Why These Decisions

- **Why Min-Max Normalization?**: Faiss cosine similarity scores typically range from `[0.0, 1.0]` but Elasticsearch scores are unbounded (e.g., `1.5`, `12.4`, `40.2`). You cannot add a Faiss score to an ES score directly. Normalizing them both to `[0, 1]` ensures that the "best" visual match and the "best" text match carry equal mathematical weight before the user's custom weights are applied.
- **Why `faiss_id` as Primary Key?**: Thanks to the work in Phase 3 (Metadata Normalization), every single piece of data (Image, OCR Text, Audio Transcript) points back to a single, unambiguous `faiss_id`. This allows the `merge_and_rank` function to act like a SQL `OUTER JOIN`, effortlessly combining multimodal data.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase4_task1.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 3 items

tests/test_phase4_task1.py::TestFusionService::test_merge_and_rank PASSED
tests/test_phase4_task1.py::TestFusionService::test_normalize_identical_scores PASSED
tests/test_phase4_task1.py::TestFusionService::test_normalize_scores PASSED

============================= 3 passed in 17.43s ===========================
```
