# Phase 4 Task 2 Report: Dynamic Weights & Query Planner

## Status: ✅ Completed and Tested

---

## What Changed

1. **Created `QueryPlanner` in `src/utils/nlp_processing.py`**:
   - Developed a rule-based query parser that acts as the front door for Multimodal Search.
   - **Rule 1 (OCR)**: Any text enclosed in double quotes (e.g., `"SALE"`) is extracted from the main query. The extracted text is routed to the Elasticsearch OCR engine, and the OCR fusion weight is dynamically bumped up to `0.5` (50%).
   - **Rule 2 (ASR)**: If the query contains auditory intent keywords like `"nghe tiếng"`, `"nói rằng"`, the system routes the query to the Elasticsearch ASR engine and bumps the ASR fusion weight to `0.3` (30%).
   - **Rule 3 (Visual)**: The text that remains (excluding the quotes) is routed to the Faiss CLIP engine.

2. **Automated Edge-case Handling**:
   - If a user searches *only* with quotes (e.g., `"Cảnh sát"`), the `QueryPlanner` recognizes there is no visual context and automatically sets OCR weight to `1.0` (100%), completely disabling Faiss for that specific query.

3. **Added Unit Tests (`tests/test_phase4_task2.py`)**:
   - Tested pure visual queries, pure OCR queries, and complex multimodal queries combining visual, text, and audio intents.
   - All 5 test cases successfully passed the weighting logic.

---

## Why These Decisions

- **Why Rule-Based instead of LLM?**: Using rules (regex for quotes, keyword matching for audio) introduces **zero latency**. An LLM planner would add 1-2 seconds of delay to every search. Rule-based parsing provides an immediate, highly predictable UX. The user learns quickly: *"If I want to find text on a billboard, I just put it in quotes."*
- **Why Dynamic Weights?**: Hardcoding weights (e.g., Visual always 60%) hurts performance. If a user is purely looking for a spoken phrase, Visual shouldn't drag down the score. The `QueryPlanner` ensures the most relevant modalities get the lion's share of the score based on user intent.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase4_task2.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 5 items

tests/test_phase4_task2.py::TestQueryPlanner::test_asr_query PASSED
tests/test_phase4_task2.py::TestQueryPlanner::test_multimodal_query PASSED
tests/test_phase4_task2.py::TestQueryPlanner::test_ocr_query_with_quotes PASSED
tests/test_phase4_task2.py::TestQueryPlanner::test_only_ocr_query PASSED
tests/test_phase4_task2.py::TestQueryPlanner::test_pure_visual_query PASSED

============================= 5 passed in 22.02s ===========================
```
