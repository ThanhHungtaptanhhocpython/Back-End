# Phase 5 Task 1 Report: BLIP-VQA Integration

## Status: ✅ Completed and Tested

---

## What Changed

1. **Created `src/services/reranker_service.py`**:
   - Implemented `RerankerService` as a Singleton class to handle BLIP-VQA inferences.
   - Designed it with **Lazy Loading**: the heavy transformer model (`Salesforce/blip-vqa-base`) is only loaded into memory the very first time `score_image` is called, ensuring the FastAPI backend boots up instantly.

2. **Logit Extraction & Softmax Calculation**:
   - Normally, VQA models just generate a string answer ("yes" or "no"). For ranking, a binary yes/no is not helpful. We need a continuous score.
   - I used `model.generate(..., output_scores=True, return_dict_in_generate=True)` to extract the raw network logits for the very first generated token.
   - I then fetched the specific logit values for the `"yes"` token (token ID 3500) and the `"no"` token (token ID 3793).
   - Applied the Softmax mathematical formula `e^yes / (e^yes + e^no)` to yield a normalized probability between `0.0` (definitely no) and `1.0` (definitely yes).

3. **Added Unit Tests (`tests/test_phase5_task1.py`)**:
   - Mocked the transformer's output logits.
   - Simulated a scenario where the `"yes"` logit was `5.0` and the `"no"` logit was `2.0`.
   - Verified that the math correctly outputted `0.9525` (~95.2%).

---

## Why These Decisions

- **Probability over Binary**: By turning "yes/no" into a percentage (e.g. 95%), we can multiply it with the existing `final_score` from Phase 4. An image with an 80% VQA confidence will rank higher than one with a 40% VQA confidence, providing much smoother and more accurate search results.
- **Why BLIP-VQA?**: It is an excellent zero-shot visual reasoning model that runs locally, perfectly fitting the project's constraints (no external paid APIs).

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase5_task1.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 1 item

tests/test_phase5_task1.py::TestRerankerService::test_score_image PASSED

============================= 1 passed in 25.59s ===========================
```
