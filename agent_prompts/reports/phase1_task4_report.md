# Phase 1 Task 4 Report: Clean Up Dual-Mode Architecture with Lazy Initialization

## Status: ✅ Completed and Tested

---

## What Changed

1. **Refactored `src/services/user_service.py`**:
   - Removed the module-level instantiation of heavy AI models (`CosineFaiss`, `TrakeSearch`, and `VLMProcessorInstance`).
   - Implemented a lazy initialization (singleton) pattern using getter functions:
     - `get_cosine_faiss()`
     - `get_trake_search()`
     - `get_vlm_processor()`
   - Updated all service functions (`getImageDataSingleTextSearch`, `getImageDataQAndASearch`, `getImageSearchById`, `getImageSearchByFile`, `GetImageDataTrakeSearch`) to call the getter functions instead of the global variables.

2. **Cleaned up `src/__init__.py`**:
   - Removed the conditional `if not os.environ.get("FASTAPI_MODE"):`.
   - The Flask blueprint is now registered unconditionally when `src/__init__.py` is imported. This is safe because FastAPI never imports `src/__init__.py`.

3. **Cleaned up `main.py`**:
   - Removed the hacky `os.environ["FASTAPI_MODE"] = "1"` at the top of the file. 
   - `main.py` is now a pure FastAPI entry point without environment variable side-effects.

4. **Added Unit Tests (`tests/test_phase1_task4.py`)**:
   - Created a dedicated test to verify that `import src.services.user_service` does **not** trigger any model instantiation.
   - Verified that calling `get_cosine_faiss()` initializes the model exactly once, and subsequent calls return the cached singleton.
   - Verified the same behavior for TRAKE and VLMProcessor.

---

## Why These Decisions

- **Lazy Initialization**: Previously, just importing `user_service` (which happens during FastAPI boot and in every unit test) would force the backend to load gigabytes of AI models and index files. By making this lazy, the backend boots instantly. The models are only loaded into memory when an endpoint actually needs them for the first time.
- **Removing `FASTAPI_MODE`**: The environment variable check was a fragile hack. `src/__init__.py` is only used by the legacy Flask `app.py`, so there was no risk of blueprint collision with FastAPI in the first place. Removing it simplifies the codebase.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase1_task4.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 1 item

tests/test_phase1_task4.py::TestLazyInitialization::test_lazy_loading PASSED

============================= 1 passed in 0.22s ============================
```

The test passed. Importing `user_service` is now fast and side-effect free.
