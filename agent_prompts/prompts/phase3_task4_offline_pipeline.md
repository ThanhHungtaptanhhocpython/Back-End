# Phase 3 Task 4: Unified Master Indexing Script

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Combine the extraction outputs (OCR and ASR) and bulk-insert them into the running Elasticsearch instance.

## 3. Requirements
- Create a script `scripts/master_index_pipeline.py`.
- The script should act as the final stage of the offline pipeline:
  1. Load `src/dict/ocr_results.json`.
  2. Load `src/dict/asr_results.json`.
  3. Load `src/dict/metadata_clip.json`.
  4. Merge the OCR text and ASR text with their corresponding base metadata (e.g., getting `video_id`, `frame_name`, `timestamp` via the `faiss_id`).
  5. Use `ElasticProcessor` from `src.utils.elastic_processing` to push the fully formed documents into the `aic_ocr` and `aic_asr` indices on Elasticsearch.
- Ensure the script handles missing data gracefully (e.g., a frame has metadata but no OCR text).

## 4. Expected Output & Reporting
- Generate a `phase3_task4_report.md` explaining the data merge logic and proving that the data was successfully indexed into Elasticsearch.
