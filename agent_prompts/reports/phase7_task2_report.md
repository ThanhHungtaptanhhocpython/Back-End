# Phase 7 Task 2 Report: Reciprocal Rank Fusion (RRF)

## Status: ✅ Completed and Tested

---

## What Changed

1. **Created `reciprocal_rank_fusion` Algorithm**:
   - Added the algorithm to `src/services/fusion_service.py`.
   - The function takes in multiple ranked lists of results (e.g., from OpenCLIP, SigLIP, and BEiT-3).
   - It iterates through each list and calculates an `rrf_score` for every `faiss_id` using the mathematical formula: `1 / (k + rank)`, where `k` is a smoothing constant (set to 60).
   - It sums these scores up across all lists and sorts the final, unified list descendingly by `rrf_score`.

2. **Added Unit Tests (`tests/test_phase7_task2.py`)**:
   - Simulated 3 models outputting 3 different lists of results.
   - Result A appeared consistently at **Rank 5** in all 3 lists.
   - Result B appeared at **Rank 1** in only the first list, but was completely missing from the other two.
   - Verified that the RRF algorithm mathematically boosted Result A over Result B (Score A: `0.046` vs Score B: `0.016`), proving that consistency across models is rewarded more than a single model's isolated spike.

---

## Why These Decisions

- **Why RRF instead of Min-Max Averaging?**: In Phase 4, we used Min-Max to combine Faiss (Visual) and Elasticsearch (OCR/ASR) scores. That worked because we were fusing fundamentally different modalities (Image vs Text). However, in Phase 7, we are fusing 3 different *Vision-Language* models. Each model outputs cosine similarities in a vastly different mathematical distribution space (e.g., SigLIP's scores might tightly cluster between 0.1 and 0.2, while OpenCLIP spans 0.1 to 0.8). If we naively normalize and average them, the model with the widest variance will dominate the fusion.
- **The RRF Advantage**: Reciprocal Rank Fusion completely ignores the raw scores. It only looks at the **relative ranking** (1st, 2nd, 3rd...). This makes it completely immune to score scaling issues and mathematically robust when combining radically different embedding spaces.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase7_task2.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 1 item

tests/test_phase7_task2.py::TestReciprocalRankFusion::test_rrf_scoring PASSED

============================= 1 passed in 13.46s ===========================
```
