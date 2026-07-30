# Phase 5 Task 2 Report: Dynamic VQA Question Formulation

## Status: ✅ Completed and Tested

---

## What Changed

1. **Updated `QueryPlanner` in `src/utils/nlp_processing.py`**:
   - Added a new `@classmethod` called `generate_vqa_question(query: str) -> str`.
   - **Step 1**: It cleans the input string by stripping out double quotes (`"`). This ensures the translation engine doesn't get confused by punctuation meant for the OCR engine.
   - **Step 2**: It utilizes the existing `Translation` utility to translate the Vietnamese search query into English. (e.g., "người đàn ông đi xe đạp" -> "man riding a bicycle").
   - **Step 3**: It wraps the translated phrase into a strict binary question format: `Is there a {translated_text}?`.

2. **Added Unit Tests (`tests/test_phase5_task2.py`)**:
   - Mocked the `Translation` class to avoid making real network calls to Google Translate during unit testing.
   - Verified that a standard Vietnamese query correctly formats into an English "Is there a..." question.
   - Verified that a query containing quotes correctly strips the quotes before translation.

---

## Why These Decisions

- **Why Translate to English?**: The BLIP-VQA model we are using (`Salesforce/blip-vqa-base`) was trained predominantly on English datasets (like VQA v2). It performs exceptionally poorly if asked questions in Vietnamese.
- **Why "Is there a..."?**: BLIP is highly sensitive to the framing of the question. Open-ended questions ("What is the man doing?") do not return a binary "yes/no" probability. By forcing the question into an `Is there a...` format, we force the model's output logits to heavily polarize on the "yes" and "no" tokens, making our Phase 1 probability math highly accurate.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase5_task2.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 2 items

tests/test_phase5_task2.py::TestVQAQuestionFormulation::test_generate_vqa_question PASSED
tests/test_phase5_task2.py::TestVQAQuestionFormulation::test_generate_vqa_question_with_quotes PASSED

============================= 2 passed in 13.73s ===========================
```
