# Phase 6 Task 1 Report: Beam Search & Timestamp Utilization

## Status: ✅ Completed and Tested

---

## What Changed

1. **Replaced Recursive Search with Beam Search**:
   - The old algorithm (`_find_sequences_recursive`) in `src/utils/trake_processing.py` was deleted.
   - Introduced `beam_search_sequences` which processes event candidates sequentially (Event 1 -> Event 2 -> Event 3).
   - At each step, it only keeps the top `beam_width` (default: 50) highest-scoring partial sequences and discards the rest. This completely solves the combinatorial explosion issue that caused the system to hang when searching for common events with lots of candidates.

2. **Timestamp Utilization**:
   - Updated `retrieve_top_k` to pull `info.get('timestamp', 0.0)`.
   - The new `beam_search_sequences` tracks an array of `timestamps` for each sequence, paving the way for the Time Decay scoring mechanism in Task 2.

3. **Added Unit Tests (`tests/test_phase6_task1.py`)**:
   - Created a mock scenario with 3 events, each having 100 candidates.
   - Verified that instead of calculating `100 * 100 * 100 = 1,000,000` combinations (which the old algorithm would attempt), the new algorithm correctly limits the search space to just 5 sequences when `beam_width=5`.

---

## Why These Decisions

- **Why Beam Search?**: Exhaustive recursion guarantees finding the absolute mathematically optimal sequence, but it is an `O(N^K)` algorithm that scales catastrophically in production. Beam Search is a heuristic `O(N * B)` algorithm used in LLMs and Speech Recognition. It is blazing fast and finds sequences that are "good enough", which is exactly what a real-time web search engine needs.
- **Why Timestamps?**: Previously, temporal logic relied purely on `global_frame_id` (e.g., frame 100 must come after frame 50). While this ensures correct chronological order, it provides no context on *how much time* elapsed between frame 50 and frame 100. By extracting exact video timestamps, we can calculate real-world seconds.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase6_task1.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 1 item

tests/test_phase6_task1.py::TestBeamSearch::test_beam_search_limits_width PASSED

============================= 1 passed in 13.90s ===========================
```
