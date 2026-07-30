# Phase 1 Task 5 Report: Retire Legacy Flask Architecture

## Status: ✅ Completed

---

## What Changed

1. **Deprecated `app.py`**:
   - Added a `DeprecationWarning` at the top of the file that warns developers attempting to run it.
   - Kept the file intact temporarily so existing scripts relying on its presence don't crash outright, but the log warning heavily points to `main.py`.

2. **Deprecated `src/controllers/user_controller.py`**:
   - Added a `DEPRECATED` warning in the module docstring.
   - Directed developers to look at the new FastAPI routers in `src/api/routers/` instead.

3. **Updated `README.md`**:
   - Updated the "Running the Server" instructions.
   - Replaced `python app.py` with `python main.py` or `uvicorn main:app --host 0.0.0.0 --port 5000`.
   - Explained that heavy AI models are initialized lazily upon the first endpoint call, ensuring quick server startups.

4. **Updated `ARCHITECTURE_UPGRADE_PLAN.md`**:
   - Marked **Phase 1: FastAPI Migration** as completely finished (✅ COMPLETED) with all 7 bullet points checked off.

---

## Why These Decisions

- **Backward Compatibility First**: Deleting `app.py` immediately could break deployment scripts or Docker containers that still reference it. Using a soft deprecation warning allows other developers (or CI/CD pipelines) time to migrate their run commands without catastrophic failure.
- **Documentation Parity**: Ensuring the `README.md` reflects the current, state-of-the-art entry point (`main.py`) prevents new team members from accidentally developing on the old Flask framework.
- **Architectural Cleanup**: Completing this task officially sunsets Phase 1. The backend is now entirely driven by FastAPI, with modern type annotations, schema validations, and middleware handling—ready for the next phases involving Elasticsearch and Reranking.
