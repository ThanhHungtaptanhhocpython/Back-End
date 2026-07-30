# Phase 6 Task 3: BLIP-VQA Sequence Validation

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Validate the final temporal sequences using the `RerankerService` (BLIP-VQA) to ensure high semantic accuracy.

## 3. Requirements
- Hook the `RerankerService` (created in Phase 5) into the end of `process_temporal_search` in `src/utils/trake_processing.py`.
- **Logic**:
  - After Beam Search and Time Decay produce the Top N (e.g., 20) valid temporal sequences.
  - For each sequence, you have a list of frames and a list of queries (e.g., Query 1 -> Frame 1, Query 2 -> Frame 2).
  - Dynamically generate VQA questions for each query using `QueryPlanner.generate_vqa_question()`.
  - Score each physical frame against its respective question using `RerankerService.score_image()`.
  - Multiply or average the VQA scores and combine them with the sequence's `FinalScore`.
  - Re-rank the Top 20 sequences based on this ultimate score.
- **Optimization**: This should only happen for the absolute final Top N sequences to avoid running the heavy VQA model hundreds of times.

## 4. Expected Output & Reporting
- Generate a `phase6_task3_report.md` summarizing how VQA acts as the final gatekeeper for temporal search.
