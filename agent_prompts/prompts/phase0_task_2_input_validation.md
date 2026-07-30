# Task 2: Implement Input Validation for API Endpoints

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Objective
Add robust input validation to all HTTP endpoints in the Flask controller (`src/controllers/user_controller.py`). Currently, the endpoints assume the JSON payload is always perfectly formatted, which leads to 500 Internal Server Errors when fields are missing.

## 3. Requirements
- Inspect all `POST` routes in `src/controllers/user_controller.py` (e.g., `/singletextsearch`, `/qnasearch`, `/trakesearch`).
- Safely extract JSON data using `request.get_json(silent=True)`.
- Verify the presence of required fields (like `query`). Set sensible defaults for optional fields (like `topk = 100`).
- If a required field is missing, immediately return a `400 Bad Request` with our standard JSON error schema:
  `{ "success": false, "data": null, "error": "Missing required field: query" }`

## 4. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** A clear list of files modified and how the validation logic was integrated.
- **Why:** The rationale behind your technical decisions (e.g., why you chose a specific way to check dictionary keys).
- **Testing Plan:** A concrete plan to test the validation. Provide `curl` commands that intentionally send bad/empty payloads to verify the `400 Bad Request` behavior, followed by valid payloads to verify a `200 OK` response.
- **Automated Testing:** You must write a Python test script using `unittest` and Flask `test_client`. The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_task2.py`). Remember to configure `sys.path` at the top of your test file to correctly import `src` modules, and then run it to prove your code works.
