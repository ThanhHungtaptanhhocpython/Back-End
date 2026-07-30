# Phase 6 Task 2 Report: Exponential Time Decay Scoring

## Status: ✅ Completed and Tested

---

## What Changed

1. **Integrated Time Decay into Beam Search**:
   - In `trake_processing.py`, modified the `beam_search_sequences` loop to track the `base_score` (sum of Faiss scores) separately from the `total_score` (penalized score).
   - Calculated the real-world time gap: `time_gap = candidate_timestamp - sequence_start_timestamp`.
   - Applied the Exponential Decay penalty: `penalty = exp(-0.01 * time_gap)`.
   - Derived the final score: `total_score = base_score * penalty`.

2. **Added Unit Tests (`tests/test_phase6_task2.py`)**:
   - Built a test where Event 1 is followed by Event 2 in two parallel timelines.
   - Timeline A: Event 2 happens 10 seconds later.
   - Timeline B: Event 2 happens 300 seconds (5 minutes) later.
   - Both candidate events had identical Faiss base scores (`1.0`).
   - Verified that the `total_score` for Timeline A was significantly higher than Timeline B, ensuring that the Beam Search algorithm naturally bubbles tight sequences to the top.

---

## Why These Decisions

- **Why Exponential Decay instead of Linear Decay?**: If we used linear decay (e.g., `-0.1` point per second), a sequence that spans 30 seconds might drop to negative scores, breaking the Min-Max normalizers downstream. Exponential Decay ensures the penalty is a percentage multiplier (e.g. `98%`, `50%`, `10%`) so the score gracefully approaches `0.0` but never dips below it.
- **Tuning Alpha (`0.01`)**: 
  - A 10-second gap retains `~90.4%` of its score.
  - A 60-second (1 min) gap retains `~54.8%` of its score.
  - A 300-second (5 min) gap retains `~4.9%` of its score.
  - This perfectly models human expectation: events in a "sequence" usually happen within seconds or minutes of each other, not hours.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase6_task2.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 1 item

tests/test_phase6_task2.py::TestExponentialDecay::test_decay_penalizes_long_gaps PASSED

============================= 1 passed in 13.42s ===========================
```
