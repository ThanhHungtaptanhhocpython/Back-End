# Phase 5 Task 3: API Pipeline Integration (Reranking)

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Inject the VQA Reranker into the Multimodal Search API pipeline to filter out false positives before returning results to the user.

## 3. Requirements
- Update `handle_multimodal_search` in `src/api/routers/search_router.py` (or update `multimodal_search` in `fusion_service.py`).
- **Reranking Logic**:
  1. After obtaining the initial fused list (e.g., Top 100 results), slice the Top N (e.g., Top 30) for reranking to save time (running BLIP on 100 images takes too long).
  2. Generate the VQA question using `QueryPlanner.generate_vqa_question`.
  3. For each of the Top 30 items, locate its physical image on disk (`src/data/Keyframes/{split}/{frame_name}`).
  4. Call `RerankerService.score_image()` to get the VQA "yes" probability (0.0 to 1.0).
  5. Combine the VQA score with the existing `final_score` (e.g., `new_score = final_score * 0.7 + vqa_score * 0.3`).
  6. Re-sort the Top 30 items based on `new_score`.
  7. Re-attach the remaining 70 non-reranked items to the bottom of the list.
- **Important**: Add the `vqa_score` into the `score_breakdown` dictionary so the Frontend can see the effect of reranking.

## 4. Expected Output & Reporting
- Generate a `phase5_task3_report.md` detailing the reranking performance (speed vs accuracy tradeoff) and how the new scores are calculated.
