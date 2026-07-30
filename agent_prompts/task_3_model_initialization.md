# Task 3: Optimize AI Model Initialization (Singleton Pattern)

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Objective
Prevent heavy AI models (CLIP, BLIP-VQA, Faiss indices) from being re-instantiated on every request. Model instantiation should happen exactly once when the application starts.

## 3. Requirements
- Inspect `src/services/user_service.py` and `src/utils/` (`faiss_processing.py`, `vlm_processing.py`, `trake_processing.py`).
- Identify where `MyFaiss`, `VLMProcessor`, and `TRAKE` are currently being instantiated. If they are created inside route handlers or service functions (e.g., `def getImageDataSingleTextSearch(...)`), move the instantiation out to the global module scope or an App Factory initialization block.
- Ensure that the loaded models are shared globally and thread-safe for reading.
- Do not hardcode paths; ensure they still rely on environment or config variables if applicable.

## 4. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** A clear list of files modified and the architectural changes made for initialization.
- **Why:** The rationale behind your technical decisions (e.g., why you used module-level singletons vs app context).
- **Testing Plan:** A concrete plan to test performance and correctness. Describe how to verify that the startup takes longer but subsequent API requests to `/users/singletextsearch` respond much faster without re-loading the CLIP model.
- **Automated Testing:** You must write a Python test script using `unittest` and Flask `test_client`. The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_task3.py`). Remember to configure `sys.path` at the top of your test file to correctly import `src` modules, and then run it to prove your code works.
