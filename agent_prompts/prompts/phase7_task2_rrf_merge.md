# Phase 7 Task 2: Reciprocal Rank Fusion (RRF)

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Combine the search results from OpenCLIP, SigLIP, and BEiT-3 using Reciprocal Rank Fusion (RRF) rather than naive score averaging, as different models output scores on completely different scales.

## 3. Requirements
- Update `src/services/fusion_service.py`.
- Add a new function `reciprocal_rank_fusion(lists_of_results: List[List[Dict]], k: int = 60) -> List[Dict]`.
- **RRF Formula**: `RRF_Score = sum(1 / (k + rank))`.
  - For a given `faiss_id`, find its rank (1st, 2nd, 3rd...) in the OpenCLIP list, the SigLIP list, and the BEiT-3 list.
  - Apply the formula for each list it appears in and sum the results.
- Create a unit test `tests/test_phase7_task2.py` with 3 mock result lists to verify that a result appearing consistently at rank 5 across all 3 lists beats a result that appears at rank 1 in only one list.

## 4. Expected Output & Reporting
- Generate a `phase7_task2_report.md` explaining why RRF is mathematically superior to Min-Max normalization when fusing completely different embedding spaces.
