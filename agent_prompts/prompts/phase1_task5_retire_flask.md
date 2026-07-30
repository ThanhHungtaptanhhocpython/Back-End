# Phase 1 - Task 5: Retire Flask and Finalize FastAPI as Sole Entry Point

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules.
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Background
After completing Phase 1 Tasks 1–4, the backend should have:
- A Pydantic Settings configuration (`src/config/settings.py`).
- Verified endpoint parity between Flask and FastAPI.
- Production middleware (logging, error handling, request timing).
- Clean lazy initialization (no module-level model loading).

At this point, Flask is no longer needed. This task finalizes the migration.

## 3. Objective
Remove Flask as the primary entry point and make FastAPI the sole production server.

## 4. Requirements

### 4.1 Deprecate Flask Entry Point
- Rename `app.py` to `app_flask_legacy.py` (do NOT delete it — keep as reference).
- Add a comment at the top of `app_flask_legacy.py`:
  ```python
  # DEPRECATED: This Flask entry point is kept for reference only.
  # Use `uvicorn main:app` to start the FastAPI server.
  ```

### 4.2 Update `main.py` as Primary Entry Point
- Ensure `main.py` imports the Pydantic Settings from `src/config/settings.py`.
- Add a `if __name__ == "__main__":` block at the bottom that calls `uvicorn.run(...)` using settings values:
  ```python
  if __name__ == "__main__":
      import uvicorn
      from src.config.settings import get_settings
      settings = get_settings()
      uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)
  ```

### 4.3 Clean Up `src/__init__.py`
- Remove the Flask app creation logic from `src/__init__.py` if it is no longer needed by any active code path.
- If other modules still import from `src.__init__`, keep backward compatibility but mark Flask-specific code as deprecated with comments.

### 4.4 Update `Dockerfile`
- Inspect the existing `Dockerfile` in the project root.
- Change the `CMD` or `ENTRYPOINT` from `python app.py` to `uvicorn main:app --host 0.0.0.0 --port 5000`.
- Ensure the Dockerfile still builds and installs dependencies correctly.

### 4.5 Update Documentation
- Update `README.md` to reflect FastAPI as the primary server:
  - Change "Running the Server" section from `python app.py` to `uvicorn main:app --reload` (development) or `python main.py` (uses built-in uvicorn).
  - Add a note about the auto-generated API docs at `http://localhost:5000/docs`.
- Update `.agents/AGENTS.md` Section 0 (Project Context) to reflect that FastAPI is now the primary framework (not "in progress").

### Important Constraints
- Do NOT delete `app_flask_legacy.py` — someone may need to reference it.
- Do NOT change any API endpoint paths or response schemas.
- Ensure the frontend can still connect to the same URL and port.

## 5. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** Files renamed, modified, and documentation updated.
- **Why:** The rationale behind keeping the legacy file vs deleting it.
- **Testing Plan:** 
  - Provide `curl` commands to test `GET /health` and `POST /users/singletextsearch` via the FastAPI server.
  - Verify that `http://localhost:5000/docs` shows the Swagger UI.
  - Verify the Dockerfile builds successfully.
- **Automated Testing:** You must write a Python test script using `unittest` and FastAPI `TestClient`. The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_phase1_task5.py`). Test that: (1) `/health` returns 200, (2) `/docs` returns 200, (3) all search endpoints are registered and accessible. Mock heavy dependencies at `sys.modules` level. Run it to prove your code works.
