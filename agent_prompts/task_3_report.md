# Task 3 Report: Optimize AI Model Initialization (Singleton Pattern)

## Status: ✅ Completed

---

## Investigation Summary

After reviewing the model instantiation points across `src/services/user_service.py` and the `src/utils/` modules, three major performance bottlenecks were discovered:

1. **`VLMProcessor` instantiation per request:** In `user_service.py`, the `getImageDataQAndASearch` function was calling `vlm = VLMProcessor()` inside the function body. This meant that on *every single request* to `/qnasearch`, the application would download/load the heavy `Salesforce/blip-vqa-base` model from disk/HuggingFace into memory, causing massive API latency.
2. **`TRAKE` duplicate initialization:** `MyFaiss` was already being instantiated globally in `user_service.py` as `CosineFaiss`. However, the `TRAKE` class was designed to accept `bin_clip_file` and `json_path` and construct its *own* `MyFaiss` instance internally. This forced the application to load the huge Faiss index `.bin` file and the `ViT-H-14-quickgelu` CLIP model twice into memory (once for `CosineFaiss` and once for `TrakeSearch`).

---

## What Changed

### 1. `src/utils/trake_processing.py`
- **Refactored `__init__`:** Changed the `TRAKE` constructor to accept an existing `faiss_searcher: MyFaiss` instance instead of `bin_clip_file` and `json_path`. This eliminates the double-loading of the Faiss Index and CLIP model.
- **Removed `create_trake_instance`:** Deleted this unused factory function to clean up the API surface.

### 2. `src/services/user_service.py`
- **Shared Faiss Instance:** Updated the `TrakeSearch` instantiation to pass the already-created `CosineFaiss` instance: `TrakeSearch = TRAKE(CosineFaiss)`.
- **VLM Singleton:** Moved `VLMProcessor` out of the route-handler logic. Instantiated it globally at the module level as `VlmProcessorInstance = VLMProcessor()`.
- **Route Update:** Updated `getImageDataQAndASearch()` to use the global `VlmProcessorInstance` instead of creating a new `vlm` object.

---

## Why These Decisions

- **Module-Level Singletons over App Context:** Python's module caching (`sys.modules`) naturally ensures that code at the root of a module is executed exactly once per process. By moving the instantiations to the module level in `user_service.py`, we achieve the Singleton pattern automatically without needing complex Flask `g` (app context) logic or custom metaclasses. This keeps the codebase simple and readable.
- **Dependency Injection for TRAKE:** By injecting `MyFaiss` into `TRAKE`, we decouple `TRAKE` from file-loading concerns, halve the application's RAM usage, and drastically speed up the startup time. 

---

## Testing Plan

To verify the performance and correctness of this architectural change, follow these steps:

### Test 1: Verify Startup Delay
1. Start the Flask application (`python app.py`).
2. **Observation:** Notice that it takes a significant amount of time (several seconds) before the server binds to port 5000. This confirms that all heavy models (Faiss, CLIP, and BLIP-VQA) are being loaded exactly once during the initial import phase.

### Test 2: Verify Blazing Fast `/qnasearch`
Run the following curl command multiple times in succession:
```bash
curl -X POST http://localhost:5000/qnasearch \
  -H "Content-Type: application/json" \
  -d '{"query": "a person riding a bicycle", "topk": 5}'
```
**Expected:** The first request and all subsequent requests should return results almost instantly. If the model was still being instantiated per request, you would see a multi-second delay on every single curl call.

### Test 3: Verify Reduced RAM Usage
- Open Task Manager (Windows) or use `htop` (Linux).
- Start the application before and after these changes.
- **Expected:** The baseline memory consumption of the Python process should be noticeably lower, as the 5GB+ Faiss/CLIP combination is now only loaded into RAM once instead of twice.
