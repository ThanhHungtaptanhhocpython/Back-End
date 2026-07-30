# Phase 2 Task 2: Define ES Index Mappings & Sample Data

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Design the schema mappings for the `aic_ocr` and `aic_asr` indices in Elasticsearch, and generate sample JSON files to populate them for testing.

## 3. Requirements
- Create a new directory `src/dict/es_samples/`.
- Create `src/dict/es_samples/mappings.json`. It should define the strict mapping schemas for:
  - `aic_ocr`: `faiss_id`, `video_id`, `frame_name`, `timestamp`, `ocr_text` (text with Vietnamese analyzer if possible, or standard).
  - `aic_asr`: `video_id`, `start_time`, `end_time`, `text`, `nearest_faiss_id`.
- Create `src/dict/es_samples/sample_ocr.json` containing at least 5 sample OCR documents.
- Create `src/dict/es_samples/sample_asr.json` containing at least 5 sample ASR documents.
- Ensure the sample data matches the fields expected by the architecture plan.

## 4. Expected Output & Reporting
- Generate a `phase2_task2_report.md` detailing the mappings and the thought process behind the field types (e.g., `keyword` vs `text`).
