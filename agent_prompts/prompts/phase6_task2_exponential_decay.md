# Phase 6 Task 2: Exponential Time Decay Scoring

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Improve the temporal scoring by penalizing events that are too far apart in time.

## 3. Requirements
- Modify the scoring logic inside the newly created `beam_search_sequences` (from Task 1) in `src/utils/trake_processing.py`.
- **Decay Formula**:
  - `BaseScore = Sum of individual event scores in the sequence`.
  - `TimeGap = Time difference between the last event and the first event (in seconds)`.
  - `Penalty = exp(-alpha * TimeGap)` where `alpha` is a tunable parameter (e.g., `0.01`).
  - `FinalScore = BaseScore * Penalty`.
- **Implementation**:
  - If a sequence takes 2 seconds to unfold, it should retain ~98% of its score.
  - If a sequence takes 5 minutes (300 seconds), it should retain much less of its score.
- Update `tests/test_phase6_task1.py` (or create `test_phase6_task2.py`) to verify that given two sequences with identical base scores, the one with the tighter time gap is ranked higher.

## 4. Expected Output & Reporting
- Generate a `phase6_task2_report.md` explaining the decay formula, the chosen `alpha` value, and how it impacts the final ranking.
