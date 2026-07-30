# Phase 6 Task 1: Beam Search & Timestamp Utilization

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Also, refer to Phase 6 in `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Replace the slow, exhaustive recursive algorithm in `src/utils/trake_processing.py` with a highly optimized Beam Search algorithm. Utilize physical timestamps instead of relying solely on `global_frame_id`.

## 3. Requirements
- Modify `trake_processing.py`.
- **Update Metadata Extraction**: In `retrieve_top_k`, ensure you extract the `timestamp` field if available.
- **Remove Recursion**: Delete `_find_sequences_recursive`.
- **Implement Beam Search**: 
  - Write a new method `beam_search_sequences(video_id, event_candidates, beam_width=50)`.
  - At Event 1, take the Top B candidates (based on individual score).
  - For Event 2, expand each of the B sequences with the candidates of Event 2.
  - Filter out invalid branches where Event 2 happens *before* Event 1 (`candidate_time <= sequence_time`).
  - Score the new partial sequences and keep only the Top B highest-scoring branches.
  - Repeat until all events in the query are processed.
- Add a test `tests/test_phase6_task1.py` with mock event lists to ensure Beam Search returns the correct sequences without exploring all permutations.

## 4. Expected Output & Reporting
- Generate a `phase6_task1_report.md` detailing the time complexity improvement (O(N^K) vs O(N*B)) and how timestamps are now utilized.
