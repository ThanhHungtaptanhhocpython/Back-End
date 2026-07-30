# Phase 1 - Task 1: Create Pydantic Settings Configuration

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules.
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Ensure your code strictly adheres to the Python Style (PEP 8, Type Hinting), Architecture Boundaries, and Web API Standards defined there.

## 2. Background
The backend is migrating from Flask to FastAPI. The FastAPI entry point (`main.py`) and routers (`src/api/routers/`) already exist. However, configuration is still handled through the legacy Flask config classes (`src/config/config.py`, `src/config/dev_config.py`, `src/config/production.py`), which use plain class attributes. FastAPI best practice requires a **Pydantic Settings** class that reads from `.env` files.

## 3. Objective
Create a proper Pydantic Settings configuration module for FastAPI at `src/config/settings.py`.

## 4. Requirements
- Inspect the existing Flask config files (`src/config/config.py`, `src/config/dev_config.py`, `src/config/production.py`) to understand what environment variables are currently used (HOST, PORT, DEBUG, database credentials, etc.).
- Create `src/config/settings.py` using `pydantic-settings` (`from pydantic_settings import BaseSettings`).
- Define all configuration fields with appropriate types and defaults (e.g., `host: str = "0.0.0.0"`, `port: int = 5000`, `debug: bool = True`).
- Add path configurations that the services need: `FAISS_INDEX_PATH`, `METADATA_PATH`, `KEYFRAMES_ROOT`, `FEATURES_ROOT`.
- Add a `model_config` with `env_file = ".env"` so variables are loaded automatically.
- Create or update `.env.example` file listing all the environment variables with placeholder values.
- Update `main.py` to import and use the new settings instead of any Flask config references.
- Do NOT delete the Flask config files (they are still used by `app.py` during the transition period).

## 5. Reporting & Testing Plan Required
Upon completing this task, you must output a report that includes:
- **What changed:** A clear list of files created/modified.
- **Why:** The rationale behind your design decisions (field names, defaults, validation rules).
- **Testing Plan:** Provide `curl` commands to hit `GET /health` on the FastAPI server (`uvicorn main:app --reload`) to confirm it boots with the new settings.
- **Automated Testing:** You must write a Python test script using `unittest`. The test file MUST be placed inside the `tests/` directory (e.g., `tests/test_phase1_task1.py`). Test that the Settings class loads defaults correctly and that overriding via environment variables works. Remember to configure `sys.path` at the top of your test file, and then run it to prove your code works.
