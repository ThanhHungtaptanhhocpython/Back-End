# Phase 2 Task 3 Report: Elasticsearch Utility & Indexing Scripts

## Status: ✅ Completed and Tested

---

## What Changed

1. **Created `src/utils/elastic_processing.py`**:
   - Implemented `ElasticProcessor`, a utility class that acts as a wrapper around the official `elasticsearch` Python client.
   - Automatically loads the connection URL from `src.config.settings`.
   - **`create_indices()`**: Safely provisions the `aic_ocr` and `aic_asr` indexes using the `mappings.json` schema without overwriting existing data.
   - **`bulk_index_ocr()` / `bulk_index_asr()`**: Leverages the `elasticsearch.helpers.bulk` API for high-performance batch insertions, essential for large datasets.
   - **`search_ocr()` / `search_asr()`**: Implemented robust full-text search using a composite `bool` query. It prioritizes exact phrase matches (`match_phrase` with `boost: 2.0`) but falls back to partial word matching (`multi_match`) for better recall on noisy OCR/ASR data.

2. **Created `scripts/index_es_samples.py`**:
   - A standalone script that reads `mappings.json`, `sample_ocr.json`, and `sample_asr.json` from the `src/dict/es_samples/` directory and feeds them into the `ElasticProcessor`.
   - This script can be run anytime the local database is wiped to instantly restore the mock data environment.

3. **Added Unit Tests (`tests/test_phase2_task3.py`)**:
   - Thoroughly tested the `ElasticProcessor` logic using `unittest.mock.patch` to simulate Elasticsearch network calls.
   - Verified that the bulk payloads are correctly formatted.
   - Verified that search results are correctly parsed and that Elasticsearch `_score` values are injected directly into the document dictionaries for downstream Reranking and Fusion.

---

## Why These Decisions

- **Boolean Queries with Boosting**: OCR and ASR data are notoriously noisy. If a user searches for "Bệnh viện Chợ Rẫy", we want the exact string to score highest. But if the OCR detected "Bệnh việ Chợ Rẫy", a strict phrase match would fail; `multi_match` catches it as a fallback.
- **Injecting `_score`**: The raw Elasticsearch score is extracted and appended to each result dictionary (`hit["_source"]["_score"] = hit["_score"]`). In Phase 4 (Adaptive Fusion), we will need these scores to calculate the final weighted rankings alongside Faiss visual scores.
- **Standalone Indexing Script**: Separating indexing from the main API server ensures that the API remains stateless and fast. Indexing is an offline task.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase2_task3.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 3 items

tests/test_phase2_task3.py::TestElasticProcessor::test_bulk_index_ocr PASSED
tests/test_phase2_task3.py::TestElasticProcessor::test_create_indices PASSED
tests/test_phase2_task3.py::TestElasticProcessor::test_search_ocr PASSED

============================= 3 passed in 15.07s ===========================
```
