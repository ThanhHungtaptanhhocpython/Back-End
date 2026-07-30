# Phase 4 Task 1: Fusion Service & Min-Max Normalization

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Also, refer to Phase 4 in `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Create a unified service that can take search queries, hit all three retrieval backends (Faiss Visual, ES OCR, ES ASR), and merge the results into a single ranked list using score normalization and weighted fusion.

## 3. Requirements
- Create `src/services/fusion_service.py`.
- Implement a `FusionService` class or standalone functions.
- **Min-Max Normalization**: Write a helper function that takes a list of dictionary results (with raw `_score` or cosine similarity) and normalizes them to a `[0, 1]` scale.
  - Formula: `(score - min_score) / (max_score - min_score + epsilon)`
- **Merging**: Merge results from `get_cosine_faiss().text_search()`, `get_elastic_processor().search_ocr()`, and `get_elastic_processor().search_asr()`.
  - Use `faiss_id` as the primary key to merge scores for the same keyframe.
  - Handle cases where a keyframe appears in one modality but not the others (missing score = 0).
- **Weighted Scoring**: For now, use fixed weights (e.g., `visual_weight = 0.6`, `ocr_weight = 0.2`, `asr_weight = 0.2`).
  - Calculate `final_score` = `visual_weight * norm_visual + ocr_weight * norm_ocr + asr_weight * norm_asr`.
- **Score Breakdown**: The returned items must include a `score_breakdown` dictionary detailing the individual scores and the final score for debugging UI.

## 4. Expected Output & Reporting
- Generate a `phase4_task1_report.md` explaining the normalization math and how the dictionary merging logic handles edge cases (like a frame having no text).
