# Phase 5 Task 2: Dynamic VQA Question Formulation

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Automatically convert a user's natural language search query into a strict Yes/No question in English so the BLIP model can evaluate it.

## 3. Requirements
- Update `src/utils/nlp_processing.py`.
- Add a new method to `QueryPlanner`: `generate_vqa_question(query: str) -> str`.
- **Logic**:
  1. The input `query` is likely in Vietnamese (e.g., "người đàn ông đi xe đạp").
  2. Use the existing `Translation` class in `nlp_processing.py` to translate the query to English ("a man riding a bicycle").
  3. Prepend "Is there a " or "Is this a picture of " to the translated query, followed by a question mark.
     - Example: "Is there a man riding a bicycle?"
- Write a unit test (`tests/test_phase5_task2.py`) to verify the translation and formatting logic.

## 4. Expected Output & Reporting
- Generate a `phase5_task2_report.md` explaining how the translation and string formatting rules ensure BLIP receives optimal questions.
