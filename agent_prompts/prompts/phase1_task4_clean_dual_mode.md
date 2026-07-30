# Phase 1 - Task 4: Clean Up Dual-Mode Flask/FastAPI Architecture

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules.
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Background
The backend currently uses an awkward dual-mode switch:
- `src/__init__.py` checks `os.environ.get("FASTAPI_MODE")` to decide whether to register Flask blueprints.
- `main.py` sets `os.environ["FASTAPI_MODE"] = "1"` before importing anything.
- `user_service.py` initializes heavy models (`MyFaiss`, `TRAKE`) at module-level (line 20-21), which means they load regardless of which framework is active.

This creates fragile import chains and makes testing difficult. The module-level initialization also means heavy AI models are loaded as a side-effect of importing the service module.

## 3. Objective
Refactor the initialization chain so that:
- Heavy model instances are created via a **lazy initialization pattern** (initialized on first use, not on import).
- The FastAPI app can import services cleanly without triggering Flask blueprint registration.
- The Flask app still works for backward compatibility during the transition.

## 4. Requirements

### 4.1 Lazy Service Initialization
- In `src/services/user_service.py`, replace the module-level globals:
  ```python
  CosineFaiss = MyFaiss(bin_clip_file, meta_data)  # line 20
  TrakeSearch = TRAKE(bin_clip_file, meta_data)     # line 21
  ```
  with a lazy singleton pattern. For example:
  ```python
  _cosine_faiss: MyFaiss | None = None
  _trake_search: TRAKE | None = None

  def get_cosine_faiss() -> MyFaiss:
      global _cosine_faiss
      if _cosine_faiss is None:
          _cosine_faiss = MyFaiss(bin_clip_file, meta_data)
      return _cosine_faiss
  ```
- Update all functions in `user_service.py` that reference `CosineFaiss` and `TrakeSearch` to call the getter functions instead.

### 4.2 Clean Import Chain
- Remove the `FASTAPI_MODE` environment variable hack from `src/__init__.py`.
- Instead, have `src/__init__.py` ALWAYS create the Flask app but only register blueprints if `app.py` is the entry point (or use a simple flag).
- Ensure `main.py` can import service functions without triggering Flask app creation or blueprint registration.

### 4.3 VLMProcessor Lazy Init
- In `src/services/user_service.py`, the `getImageDataQAndASearch` function creates a new `VLMProcessor()` on every call (line 145). This is extremely expensive. Move it to the same lazy singleton pattern.

### Important Constraints
- Do NOT change any API behavior or response format.
- Do NOT rename any public service functions (they are imported by both Flask controller and FastAPI router).
- Keep Flask `app.py` working during the transition.

## 5. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** Files modified with the initialization refactoring.
- **Why:** The rationale behind lazy init vs module-level (import speed, testability, memory).
- **Testing Plan:** Describe how to verify that: (1) `python app.py` still boots Flask correctly, (2) `uvicorn main:app` boots FastAPI correctly, (3) models are only loaded when an endpoint is actually called, not on import.
- **Automated Testing:** You must write a Python test script using `unittest`. The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_phase1_task4.py`). Test that importing `user_service` does NOT trigger model loading, and that calling a service function DOES trigger it. Mock the heavy dependencies at `sys.modules` level. Run it to prove your code works.
