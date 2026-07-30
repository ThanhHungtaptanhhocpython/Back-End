# Phase 5 Task 3 Report: API Pipeline Integration (Reranking)

## Status: ✅ Completed

---

## What Changed

1. **Updated `multimodal_search` in `fusion_service.py`**:
   - Injected the Phase 5 Reranking logic right before returning the final merged results to the router.
   - **Performance Optimization**: Instead of running BLIP-VQA on all 100 results (which would take over 10 seconds), the system slices the **Top 30** results (the `rerank_pool`) and leaves the remaining 70 untouched (the `bottom_pool`).
   - For each result in the Top 30, it constructs the physical path to the keyframe image on disk.
   - It calls `RerankerService.score_image()` with the dynamically translated VQA question (from Task 2) to get a `vqa_score`.
   - The original multimodal fusion score is blended with the VQA score (`new_score = final_score * 0.7 + vqa_score * 0.3`).
   - Finally, the Top 30 items are re-sorted according to this `new_score`, and the `bottom_pool` is appended back to the end of the list.

2. **Updated API Router (`search_router.py`)**:
   - Passed the `original_query` string into `multimodal_search` so the `QueryPlanner` knows exactly what text to translate and ask BLIP.

---

## Why These Decisions

- **Top 30 Reranking**: In real-world retrieval systems (like Google Search or Bing), heavy ML models are never run on the entire dataset. A fast, cheap algorithm (Faiss/Elasticsearch) gets the Top 1000, and a heavy, expensive algorithm (BLIP-VQA) re-ranks the Top 30. This ensures sub-second API responses while still drastically improving the quality of the first page of results the user sees.
- **Score Blending (`0.7` / `0.3`)**: We don't want VQA to completely override Faiss and Elasticsearch. If Faiss is 100% sure an image matches, but BLIP is confused, we still want the image to rank high. The 70/30 split ensures stability.
- **`vqa` in `score_breakdown`**: By attaching the `vqa_score` to the breakdown, the UI can now show the user: *"This image ranked #1 because Faiss scored it 0.8, OCR scored it 0.9, and the AI Reranker gave it a 95% Yes!"*
