## Why

The competition workstation currently renders all retrieval results from a local mock engine, so the polished preview cannot demonstrate the FastAPI multimodal retrieval system. Connecting the existing query workspace to the backend enables a credible live demo while retaining a reliable offline checkpoint when search assets or the API are unavailable.

## What Changes

- Add a frontend backend-search adapter that maps workspace query modes to the FastAPI search endpoints and normalizes the common response envelope into the existing result-card shape.
- Make the backend URL and live/demo mode configurable through Vite environment variables instead of a hard-coded tunnel URL.
- Preserve the local mock engine as an explicit fallback when live mode is disabled or the API request fails, and surface the active data source in the workstation status.
- Support text, image-pivot, temporal, OCR, ASR, and multimodal search requests through the adapter. The chat copilot remains local-demo-only.
- **BREAKING:** Live mode requires a reachable FastAPI instance with its FAISS/keyframe and optional Elasticsearch assets provisioned; without it the UI falls back to demo results.

## Capabilities

### New Capabilities

- `backend-search-integration`: Configurable, demo-safe retrieval requests from the React workstation to the FastAPI backend.

### Modified Capabilities

- None.

## Impact

- Affects the workspace search boundary (`src/shared/adapters.js`, `src/features/workspace/Workstation.jsx`) and adds a dedicated backend API adapter.
- Uses FastAPI endpoints under `/users`, especially `multimodalsearch`, `imagesearch`, and `temporalsearch`, with the common `{ success, message, data: { items, total_items } }` response envelope.
- Does not modify Back-End. Live search still depends on external FAISS/keyframe assets and Elasticsearch availability.
