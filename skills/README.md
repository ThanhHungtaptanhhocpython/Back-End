# Back-End Skills Overview

This repository contains a Python Flask backend for an image search and retrieval service. It uses CLIP/Faiss for semantic text-image search, a visual question-answering pipeline, and temporal event search logic.

## AI Agent quick guide

- This is a backend-only repo. There is no frontend code.
- Main app entry point: `app.py`.
- API routes live in `src/controllers/user_controller.py`.
- Search and model logic live in `src/services/user_service.py` and `src/utils/`.
- Required data dependencies are in `src/dict/` and `src/data/`.
- Use `requirements.txt` to verify runtime packages.

## Recommended use for AI agents

Use this document to understand repository structure, dependencies, API endpoints, and where the main processing logic is implemented.

### Repository structure

- `app.py`
  - Application entry point.
  - Imports the Flask app from `src/config/config.py`.
  - Runs the app with `config.HOST`, `config.PORT`, and `config.DEBUG`.
- `requirements.txt`
  - Python packages needed to run the service.
- `src/`
  - Main application source code.
- `src/config/`
  - `config.py`: creates and configures the Flask app, loads environment variables, registers the `users` blueprint.
  - `dev_config.py`: development configuration values.
  - `production.py`: production configuration values.
- `src/controllers/`
  - `user_controller.py`: defines API routes and request/response handling.
- `src/services/`
  - `user_service.py`: search logic and response generation.
- `src/utils/`
  - `faiss_processing.py`: Faiss index loading, CLIP model initialization, text and image search.
  - `vlm_processing.py`: visual question answering using Hugging Face transformers.
  - `trake_processing.py`: temporal event retrieval and ranking logic.
  - `nlp_processing.py`: translation utility used before text queries.
- `src/dict/`
  - Contains metadata and Faiss index files required for search.

## Key dependencies

- `flask`, `flask_cors` for HTTP API.
- `python-dotenv` for environment variable loading.
- `faiss-cpu` for nearest neighbor search.
- `open_clip_torch` and `transformers` for CLIP and VQA models.
- `torch`, `Pillow`, `googletrans`.

## Main runtime flow

1. `python app.py` starts the Flask app.
2. `src/config/config.py` builds the app and registers `users` blueprint.
3. Requests to `/users/*` are handled in `src/controllers/user_controller.py`.
4. `user_controller.py` calls functions in `src/services/user_service.py`.
5. Search and answer generation use:
   - `MyFaiss` in `src/utils/faiss_processing.py` for text/image search.
   - `VLMProcessor` in `src/utils/vlm_processing.py` for VQA answers.
   - `TRAKE` in `src/utils/trake_processing.py` for temporal search.

## API endpoints

### `POST /users/singletextsearch`
- Input: JSON body with `query` and `topk`.
- Behavior: text-based semantic search via Faiss.
- Response: JSON with `items` and `total_items`.

### `POST /users/qnasearch`
- Input: JSON body with `query` and `topk`.
- Behavior: text-based search plus VQA answers on returned keyframes.
- Response: JSON with `items` and `total_items`.

### `POST /users/imagesearch`
- Input: form data with `topk`, `clip`, `clipv2`, `faiss_index`, and file `image`.
- Behavior: currently ignores uploaded image and runs image search by ID using `getImageSearchById`.
- Response: JSON with `items` and `total_items`.

### `POST /users/trakesearch`
- Input: JSON body with `query` and optional `topk`.
- Behavior: temporal event retrieval using `TRAKE`.
- Response: JSON with `items` and `total_items`.

## Important implementation notes

- `src/services/user_service.py` reads the Faiss index and metadata via `MyFaiss`.
- Image files are expected under `src/data/Keyframes`.
- `getImageDataSingleTextSearch`, `getImageDataQAndASearch`, and `getImageSearchById` load keyframe images, encode them as base64, and return them in responses.
- `VLMProcessor.batch_answer` uses file paths to load images and answer questions.
- `TRAKE` uses Faiss-based retrieval and sequence ranking, but its `GetImageDataTrakeSearch` wrapper only returns processed temporal search results.
- The code currently has an empty `src/utils.py` file; this is not used by the core application.

## Guidance for agents

- Prefer editing `src/services/user_service.py` or `src/controllers/user_controller.py` for API behavior.
- Prefer editing `src/utils/faiss_processing.py`, `vlm_processing.py`, and `trake_processing.py` for search and model logic.
- Use `requirements.txt` to verify required runtime packages.
- Do not assume frontend behavior; this repo only exposes backend API endpoints.
- Treat the existing `src/dict/` metadata and Faiss files as required data dependencies.

## How to run locally

- Install dependencies:
  - `pip install -r requirements.txt`
- Start the backend:
  - `python app.py`
- Confirm the server is running and endpoint prefix is `/users`.

## Notes for maintenance

- The repo is suitable for backend enhancements such as improved search ranking, error handling, and request validation.
- The VQA and CLIP models can be expensive to load; the code currently initializes CLIP in `MyFaiss` and VQA in `VLMProcessor`.
- There is no dedicated test suite visible in the repository root.

## Why these choices?

- Use `app.py` + `src/config/config.py` because the Flask app is centralized and environment configuration is separated from route logic.
- Use `src/controllers/user_controller.py` for request handling to keep endpoint definitions clean and delegate business work to services.
- Use `src/services/user_service.py` for search logic because it isolates Faiss/VLM/TRAKE processing from HTTP concerns.
- Use `src/utils/` for model and search helper classes so the core routes stay focused on API behavior.
- Keep `src/dict/` as data dependencies because the Faiss index and metadata are required inputs for all search operations.

## Summary

This repo is a Flask backend offering semantic image search, visual-question answering, and temporal search. It has clear separation between:

- application startup/configuration,
- HTTP endpoints,
- service logic,
- model/search utilities.

## Decision rationale

This section explains why the repository is organized the way it is and why the current implementation was chosen.

- Separation of concerns: controllers handle routing and request/response formatting, while `user_service.py` handles search logic. This makes the code easier to maintain and test.
- Centralized configuration: `src/config/config.py` initializes the Flask app and loads environment variables. This is better than placing app creation logic inside individual routes.
- Reusable search utilities: `src/utils/faiss_processing.py`, `vlm_processing.py`, and `trake_processing.py` provide reusable model and search abstractions. They are preferred over putting model loading and search code directly in controllers.
- Data dependency isolation: keeping Faiss index and metadata in `src/dict/` makes it explicit that these files are required inputs, instead of hiding them inside code logic.
- Performance consideration: loading the CLIP model in `MyFaiss` is more efficient than reloading it on every request. It avoids repeated heavy initialization.
- User-facing API design: the controller routes expose a simple REST-like interface, while the service layer does the heavy lifting. That is usually preferable to having all logic in one place.

## How to improve

- Add request validation in `user_controller.py` to verify required fields and return clear error messages.
- Move heavy model initialization to app startup if not already cached, and ensure shared instances are reused across requests.
- Improve `POST /users/imagesearch` to actually use the uploaded image instead of relying on an ID-based search fallback.
- Add structured logging and error handling for model failures and missing data files.
- Create tests for endpoints and service functions to verify expected behavior.
- Document data paths and required files under `src/dict/` and `src/data/` so the repo is easier to install and run.

## What to change first

1. Fix input validation in `src/controllers/user_controller.py`.
2. Confirm the CLIP/VLM/TRAKE model instances are initialized once and reused.
3. Update `POST /users/imagesearch` so it processes the uploaded image correctly.
4. Add error handling around file loads and model calls in `src/services/user_service.py`.
5. Write simple endpoint tests to lock in behavior before refactoring further.

## Proposed code changes

### 1. Add validation in `src/controllers/user_controller.py`
- Validate JSON request bodies for `query` and `topk`.
- Return a `400` error when required fields are missing.
- Example change: check `data = request.get_json()` and `if not data or 'query' not in data`.

### 2. Reuse model instances at startup
- Ensure `MyFaiss`, `VLMProcessor`, and `TRAKE` are created once in `src/services/user_service.py` or a dedicated startup module.
- Avoid reloading the CLIP model for every request.
- Example change: move initialization outside route handlers and ensure shared variables are only created once.

### 3. Fix `POST /users/imagesearch`
- Use the actual uploaded image file rather than ignoring it.
- Extract the image bytes and pass them to a real image-encoding or CLIP feature extraction function.
- If image-based search is not feasible, document the current behavior clearly and remove misleading upload requirements.

### 4. Add robust error handling in `src/services/user_service.py`
- Wrap file access and model calls with `try/except`.
- Log or return descriptive error messages for missing files, invalid metadata, or model failures.
- Example change: if `FileNotFoundError` occurs when opening keyframes, return an empty result or error response instead of crashing.

### 5. Add simple endpoint tests
- Create tests under a new `tests/` folder or `src/tests/` if absent.
- Test happy-path calls to `/users/singletextsearch`, `/users/qnasearch`, `/users/trakesearch`, and `/users/imagesearch`.
- Use mocks for model calls if the Faiss/index files are not available in CI.

## Suggested agent questions

- Why did I choose `src/services/user_service.py` for core search logic instead of embedding it directly in controllers?
- Why does `POST /users/imagesearch` accept an image upload but currently use ID-based search?
- Why is the CLIP model loaded in `MyFaiss` rather than at request time?
- Why are the VQAs produced in `VLMProcessor` instead of in the controller layer?
- What should be changed if the repo needs a fully RESTful, production-ready API with better validation?
