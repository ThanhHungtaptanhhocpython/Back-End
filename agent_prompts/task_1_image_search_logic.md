# Task 1: Fix Logic in API Image Search (`POST /users/imagesearch`)

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Objective
Fix the current logic of the `POST /users/imagesearch` endpoint. Currently, this endpoint allows an image file upload but completely ignores the uploaded file, falling back to an ID-based search (`getImageSearchById`).

## 3. Requirements
- Inspect `src/controllers/user_controller.py` and `src/services/user_service.py` related to `imagesearch`.
- Modify the endpoint to actually read the bytes of the uploaded image file.
- Pass the image file through the OpenCLIP model to extract embeddings, then query the Faiss index. 
- *Note:* If true image-to-image search via embedding extraction is not yet supported by our `faiss_processing.py`, you must rename the endpoint to something accurate (e.g., `/users/imagesearchbyid`), remove the misleading `file` upload parameter, and document the behavior clearly.

## 4. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** A clear list of files modified and the exact structural changes.
- **Why:** The rationale behind your technical decisions (e.g., why you chose to extract embeddings vs. renaming the endpoint).
- **Testing Plan:** A concrete plan to test this endpoint. Provide `curl` commands to verify that uploading an image (or passing an ID) works and returns the standard JSON response format.
- **Automated Testing:** You must write a Python test script using `unittest` and Flask `test_client`. The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_task1.py`). Remember to configure `sys.path` at the top of your test file to correctly import `src` modules, and then run it to prove your code works.
