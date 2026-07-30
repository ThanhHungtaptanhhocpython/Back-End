# Phase 4 Task 2: Dynamic Weights & Query Planner

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Make the fusion weights adapt to the user's query instead of being hardcoded.

## 3. Requirements
- Create `src/utils/nlp_processing.py` (if it doesn't exist) or update `src/services/fusion_service.py`.
- Implement a `QueryPlanner` function: `parse_query(query: str) -> dict`.
- **Rule-Based Dynamic Weights**:
  - If the query contains text in quotes (e.g., `xe đạp "Bệnh viện"`), extract the quoted text as the `ocr_query` and boost `ocr_weight`.
  - If the query contains auditory keywords (e.g., "nghe tiếng", "nói rằng", "âm thanh"), boost `asr_weight`.
  - Otherwise, default to visual-heavy weights.
- The planner should return a structure like:
  ```json
  {
    "visual_query": "xe đạp",
    "ocr_query": "Bệnh viện",
    "asr_query": "",
    "weights": {"visual": 0.5, "ocr": 0.5, "asr": 0.0}
  }
  ```
- Integrate this `QueryPlanner` into the `FusionService` created in Task 1.

## 4. Expected Output & Reporting
- Generate a `phase4_task2_report.md` detailing the rules used for dynamic weighting and how it improves search precision.
