# Task 1 Report: Fix Logic in API Image Search

## Status: ✅ Already Fixed (with minor cleanup applied)

---

## Investigation Summary

After inspecting the three key files involved in the image search pipeline, the core bug described in the task prompt — *"the endpoint ignores the uploaded file and falls back to ID-based search"* — has **already been resolved** in a prior iteration. The current codebase properly supports both image-file-based search and ID-based search.

### Evidence of the existing fix:

| Layer | File | Function/Logic | Status |
|-------|------|----------------|--------|
| **Utility** | `src/utils/faiss_processing.py` | `_get_image_features()` encodes a PIL Image via OpenCLIP into a normalized embedding vector | ✅ Present |
| **Utility** | `src/utils/faiss_processing.py` | `image_search_by_file()` calls `_get_image_features()` → Faiss search → returns results | ✅ Present |
| **Service** | `src/services/user_service.py` | `getImageSearchByFile()` opens the uploaded file as PIL Image, calls `CosineFaiss.image_search_by_file()` | ✅ Present |
| **Controller** | `src/controllers/user_controller.py` | Lines 65-82: if file uploaded → `getImageSearchByFile(file, topk)`, elif `faiss_index` → `getImageSearchById()`, else → 400 error | ✅ Present |

---

## What Changed (This Session)

### File: `src/utils/faiss_processing.py`

**Change:** Removed a duplicate `_prepare_results` method and added a Google-style docstring to `image_search_by_file`.

**Details:**
- The method `_prepare_results` was defined **twice** in the class (at lines 91 and 134). Python silently uses the last definition, making the first one dead code. Both copies were identical, so there was no runtime bug, but it violated the Clean Diff Rules in `.agents/AGENTS.md` and could cause confusion for future developers.
- Added a proper docstring to `image_search_by_file()` which was missing (violating the Docstrings rule in Section 1 of AGENTS.md).

```diff
  def image_search_by_file(self, image: Image.Image, k: int) -> tuple[...]:
+     """Searches for similar keyframes using a PIL Image as the query.
+
+     Args:
+         image: A PIL Image to encode and search with.
+         k: Number of top results to return.
+
+     Returns:
+         A tuple of (scores, image_ids, infos_query, image_paths).
+     """
      image_features = self._get_image_features(image)
      ...

- def _prepare_results(self, image_ids):  # FIRST COPY (removed)
-     ...
-     return valid_infos, image_paths
-
  def text_search(self, text, k, index=None):
      ...

  def _prepare_results(self, image_ids):  # SECOND COPY (kept)
      ...
```

---

## Why These Decisions

1. **No endpoint rename needed:** The original task suggested renaming the endpoint if image-by-file search wasn't supported. Since `image_search_by_file()` already exists and works correctly (encodes via OpenCLIP -> searches Faiss), the endpoint name `/users/imagesearch` is accurate and should be kept.

2. **Removed duplicate method:** Having two identical method definitions in the same class is a maintenance hazard. A future developer might edit one copy thinking it's the active one, only to have their changes silently ignored. Removing the dead copy eliminates this risk.

3. **Added docstring:** Per AGENTS.md Section 1 ("Every public module, class, and function must have a clear docstring in Google Style"), the `image_search_by_file` method was non-compliant.

---

## Testing Plan

### Test 1: Image file upload search
```bash
curl -X POST http://localhost:5000/users/imagesearch \
  -F "image=@/path/to/test_image.jpg" \
  -F "topk=10"
```

**Expected response (200 OK):**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": 0,
        "folder_key": "L26",
        "video_key": "V001",
        "frame_key": 12345,
        "image": "<base64_encoded_string>"
      }
    ],
    "total_items": 10
  }
}
```

### Test 2: ID-based fallback search
```bash
curl -X POST http://localhost:5000/users/imagesearch \
  -F "faiss_index=36244" \
  -F "topk=5"
```

**Expected:** Same JSON structure with results based on the Faiss ID similarity search.

### Test 3: Missing both image and faiss_index
```bash
curl -X POST http://localhost:5000/users/imagesearch \
  -F "topk=5"
```

**Expected response (400 Bad Request):**
```json
{
  "success": false,
  "message": "Either an uploaded image file or a valid faiss_index must be provided.",
  "data": { "items": [], "total_items": 0 }
}
```

### Test 4: Invalid topk
```bash
curl -X POST http://localhost:5000/users/imagesearch \
  -F "image=@/path/to/test.jpg" \
  -F "topk=-1"
```

**Expected response (400 Bad Request):**
```json
{
  "success": false,
  "message": "topk must be a positive integer.",
  "data": { "items": [], "total_items": 0 }
}
```

> **Note:** These tests require the backend to be running with valid Faiss index and keyframe data. If data assets are not yet built, run the data pipeline scripts first (`scripts/rebuild_keyframes.py` -> `scripts/build_clip_faiss_index.py`).
