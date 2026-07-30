# Phase 2 Task 2 Report: ES Index Mappings & Sample Data

## Status: ✅ Completed

---

## What Changed

1. **Created `src/dict/es_samples/mappings.json`**:
   - Defined strict Elasticsearch schemas for two indexes:
     - `aic_ocr`: For storing text identified inside video frames.
     - `aic_asr`: For storing spoken transcripts extracted from video audio.
   - Designed the field mapping types specifically for search efficiency.

2. **Created `src/dict/es_samples/sample_ocr.json`**:
   - Built a list of 5 sample OCR documents.
   - Included required metadata linking the OCR text back to its source keyframe (via `faiss_id`, `video_id`, and `frame_name`).

3. **Created `src/dict/es_samples/sample_asr.json`**:
   - Built a list of 5 sample ASR documents.
   - Simulated timestamped audio segments (`start_time`, `end_time`) and resolved their proximity to the nearest visual keyframe (`nearest_faiss_id`).

---

## Why These Decisions

- **`keyword` vs `text` mapping**: 
  - Fields like `video_id`, `frame_name`, and `language` are stored as `keyword`. This allows exact filtering and aggregations without the overhead of tokenization.
  - The actual transcripts (`ocr_text` and `text`) are stored as `text`. This instructs Elasticsearch to tokenize the strings into words, lower-case them, and build an inverted index, which enables fuzzy and partial-phrase full-text search.
- **Unified Keyframe Identity**: Both indices hold a reference back to the `faiss_id`. This is the fundamental linkage mechanism that will allow us (in Phase 4) to perform Adaptive Fusion, combining visual scores from Faiss with text scores from Elasticsearch on the exact same frame.
- **Mock Data First**: Creating sample data before writing the extraction pipeline allows the API frontend and backend search logic to be developed, tested, and validated completely independent of the heavy ML extraction scripts.
