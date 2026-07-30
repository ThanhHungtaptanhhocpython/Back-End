# Task 2 Report: Implement Input Validation for API Endpoints

## Status: ✅ Completed

---

## Investigation Summary

After inspecting the user controller (`src/controllers/user_controller.py`), it was found that the search endpoints (`/singletextsearch`, `/qnasearch`, etc.) were directly accessing dictionary keys from `request.get_json()` (e.g., `data["query"]`) without validation. If a client sent an empty body or a payload missing the required fields, Flask would raise a `KeyError` resulting in a 500 Internal Server Error. Additionally, the existing validation in other endpoints used inconsistent JSON schemas (e.g., `"message"` instead of `"error"`).

---

## What Changed

### File: `src/controllers/user_controller.py`

**Change:** Rewrote the controller to add robust input validation to all POST endpoints, enforcing the standard response schema and setting defaults.

**Details:**
- **Extracted helper functions:** Added `_error_response()` and `_success_response()` to centralize the unified JSON structure:
  ```json
  {
    "success": false,
    "data": null,
    "error": "..."
  }
  ```
- **Extracted `_parse_topk()`:** Centralized the validation of the `topk` parameter to ensure it is a positive integer, defaulting to `100` (`DEFAULT_TOPK`) if omitted.
- **`/singletextsearch` & `/qnasearch`:** Switched from `request.get_json()` to `request.get_json(silent=True)`. Added explicit checks to ensure `data` is a valid dictionary and that `query` is a non-empty string.
- **Removed dead/debug code:** Removed an arbitrary `print(data)` from `/qnasearch`.
- **`/imagesearch` & `/trakesearch` / `/temporalsearch`:** Preserved the original logic but refactored their error returns to use `_error_response()` to match the standard schema.
- **Type Hinting & Docstrings:** Added standard return type annotations (`-> Response`) and Google-Style docstrings to all route functions and helpers.

---

## Why These Decisions

1. **Use of `get_json(silent=True)`:** By silencing the parser, if the request body is not valid JSON, it returns `None` instead of raising an HTTP 400 exception directly from Flask. This allows us to catch it and return our *own* formatted `_error_response()`.
2. **Use of `.get()` instead of bracket syntax:** Changed `data["query"]` to `data.get("query")` to safely check for existence.
3. **Centralized Response Builders:** Hardcoding the dictionary structure in every endpoint is error-prone. Extracting it to a helper guarantees the schema is 100% consistent across all endpoints.
4. **Returning `None` from `_parse_topk`:** By making `_parse_topk` return `None` on failure, the endpoints remain in control of the HTTP response flow and the precise error message.

---

## Testing Plan

A full test suite (`test_task2.py`) was written using Flask's `app.test_client()` to simulate HTTP requests locally without needing a live network server.

### Test Coverage:
1. **`/users/singletextsearch`**
   - ❌ Missing body -> 400 Bad Request
   - ❌ Empty JSON object -> 400 Bad Request ("Missing required field: query")
   - ❌ Invalid `topk` (e.g., "abc") -> 400 Bad Request
   - ✅ Valid request -> 200 OK
2. **`/users/qnasearch`**
   - Similar coverage to singletextsearch.
3. **`/users/trakesearch`**
   - ❌ Missing body -> 400 Bad Request
   - ❌ Empty query list (`{"query": []}`) -> 400 Bad Request
   - ❌ Invalid event format in list -> 400 Bad Request
   - ✅ Valid temporal list -> 200 OK
4. **`/users/imagesearch`**
   - ❌ Missing both image and faiss_index -> 400 Bad Request
   - ✅ Valid faiss ID -> 200 OK

All endpoints now reliably return 400 with a standard JSON message for invalid inputs rather than crashing with a 500 error.
