# Phase 5 Task 1: BLIP-VQA Integration

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Also, refer to Phase 5 in `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Create a service that wraps the BLIP-VQA model (Visual Question Answering) to score the relevance of an image given a specific question.

## 3. Requirements
- The model `Salesforce/blip-vqa-base` is already being used in `src/utils/vlm_processing.py` (via `VLMProcessor`). We can either refactor it or create a new dedicated `RerankerService` at `src/services/reranker_service.py`.
- Implement a method `score_image(image_path: str, question: str) -> float`.
  - The method should load the image from disk using `PIL`.
  - It should pass the image and the question to the BLIP model.
  - Instead of just getting the text answer ("yes" or "no"), it should extract the **logits/probabilities** for the "yes" token vs the "no" token.
  - The method should return the normalized probability of "yes" (a float between 0.0 and 1.0).
- **Optimization**: The model should be lazy-loaded only when needed to save memory. 
- Write a simple unit test (`tests/test_phase5_task1.py`) using a mock image and mock model output to verify the probability calculation.

## 4. Expected Output & Reporting
- Generate a `phase5_task1_report.md` detailing the implementation of the VQA scoring mechanism and any challenges encountered with extracting logits from the transformers pipeline.
