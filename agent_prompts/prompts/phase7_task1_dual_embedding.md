# Phase 7 Task 1: Dual Embedding (SigLIP & BEiT-3)

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Also, refer to Phase 7 in `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Introduce two state-of-the-art vision-language models (SigLIP and BEiT-3) to run alongside the existing OpenCLIP model. These models extract better features for complex reasoning and multilingual text.

## 3. Requirements
- Create a new utility `src/utils/siglip_processing.py` to wrap the `google/siglip-base-patch16-224` model.
- Create a new utility `src/utils/beit3_processing.py` to wrap the BEiT-3 model (or a compatible HuggingFace equivalent like `microsoft/beit-base-patch16-224-pt22k` if BEiT-3 is unavailable).
- Both utilities must have a method `get_text_features(text: str) -> np.ndarray`.
- **Note**: Do NOT run the indexing over the entire dataset yet. Just write the wrapper classes and create mock unit tests in `tests/test_phase7_task1.py` to ensure the models load and output the correct feature dimensions (e.g., 768 for SigLIP).

## 4. Expected Output & Reporting
- Generate a `phase7_task1_report.md` detailing the integration of the new models, their vector dimensions, and the expected RAM usage.
