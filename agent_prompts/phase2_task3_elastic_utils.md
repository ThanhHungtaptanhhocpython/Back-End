# Phase 2 Task 3: Implement Elasticsearch Utility & Indexing Scripts

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Write the core Python utility class to interact with Elasticsearch, and create scripts to load the sample data into the local ES instance.

## 3. Requirements
- Create `src/utils/elastic_processing.py`.
  - Define an `ElasticProcessor` class.
  - Implement methods:
    - `create_indices()`: Reads `mappings.json` and creates indices if they don't exist.
    - `bulk_index_ocr(documents: list[dict])`
    - `bulk_index_asr(documents: list[dict])`
    - `search_ocr(query: str, topk: int = 100) -> list[dict]`
    - `search_asr(query: str, topk: int = 100) -> list[dict]`
  - Use `multi_match` or `match_phrase` queries for robust text search.
- Create a one-off script `scripts/index_es_samples.py` that reads the sample JSON files and uses `ElasticProcessor` to index them.
- Add unit tests for the utility class (mocking `elasticsearch.Elasticsearch`).

## 4. Expected Output & Reporting
- Generate a `phase2_task3_report.md` detailing the search query structure chosen for OCR/ASR and how the indexing script functions.
