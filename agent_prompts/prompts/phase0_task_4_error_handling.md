# Task 4: Robust File/Data Error Handling

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Objective
Prevent the backend from crashing (returning HTTP 500) when required data files (Faiss index `.bin`, Keyframe images `.webp`, or Feature arrays `.npy`) are missing. The system should gracefully handle missing files by returning an empty result set or a standardized error response.

## 3. Requirements
- Inspect file reading operations in `src/services/user_service.py` (e.g., loading base64 images) and `src/utils/faiss_processing.py`.
- Wrap file open operations (`open(...)`, `Image.open(...)`, `np.load(...)`, `faiss.read_index(...)`) in `try...except` blocks.
- Specifically catch `FileNotFoundError` or general `Exception`.
- When an image file is missing during result generation, skip it or return a placeholder, logging the error properly using the `logging` module.
- If the Faiss index is missing at startup, catch the error and log a critical warning (or graceful fallback).

## 4. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** A clear list of files modified and the try/except strategies used.
- **Why:** The rationale behind your technical decisions (e.g., choosing to skip a missing image vs returning an error).
- **Testing Plan:** A concrete plan to verify this behavior. Provide instructions on how to temporarily rename/hide a keyframe image and run a query to ensure the API returns a `200 OK` (with the missing item omitted) instead of a `500 Internal Server Error`.
- **Automated Testing:** You must write a Python test script using `unittest` and Flask `test_client`. The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_task4.py`). Remember to configure `sys.path` at the top of your test file to correctly import `src` modules, and then run it to prove your code works.
