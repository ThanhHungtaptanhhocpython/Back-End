## Context

The competition workstation's query tabs and result grid use `runSearch` through a mock-only adapter. The FastAPI backend already exposes typed search routes under `/users` and returns a shared `BaseResponse` envelope, but real backend deployment is conditional on heavyweight FAISS/keyframe and Elasticsearch assets.

## Goals / Non-Goals

**Goals:**

- Route each workstation search mode through one thin API boundary when live mode is enabled.
- Normalize heterogeneous backend result fields into the existing result-grid card model.
- Preserve a deterministic local mock fallback for an unavailable or failing backend.
- Make backend selection visible to the operator during a competition demo.

**Non-Goals:**

- Modify FastAPI routes, ranking logic, data assets, or Elasticsearch provisioning.
- Replace the mock chat copilot with a backend service.
- Rewrite the React workspace or migrate it to TypeScript in this change.

## Decisions

### Add a dedicated search gateway behind the existing adapter

The workspace will continue to call `runSearch(tab, pivot)`. `src/shared/adapters.js` will delegate that call to a new gateway that selects either FastAPI or the existing mock engine. This preserves the high-risk workspace caller contract identified by GitNexus.

Alternative: call Axios directly from `Workstation`. Rejected because it couples view state, request encoding, fallback behavior, and result conversion.

### Configure live mode through Vite variables

`VITE_SEARCH_API_BASE_URL` supplies the FastAPI base URL and `VITE_SEARCH_MODE` selects `live`, `demo`, or `auto`. The default is demo-safe `auto`: use FastAPI only when a base URL is supplied, otherwise use mocks.

Alternative: retain the previous hard-coded ngrok URL. Rejected because its lifetime is uncontrolled and it makes a local demo fragile.

### Normalize at the boundary

The gateway maps FastAPI's `{ success, message, data: { items, total_items } }` response and legacy result keys (`faiss_id_clip`, `frame_key`, `video_key`, `image`) into the card model once. Raw API detail such as score breakdown is retained as metadata for later UI use.

Alternative: alter every result component to understand both payload shapes. Rejected because it spreads backend coupling across the UI.

### Fallback only for transport/runtime failure

In `auto` mode, network failures and unavailable backend assets fall back to the mock engine and make that status explicit. A valid FastAPI validation response is surfaced as an error rather than silently replaced, so integration mistakes remain visible.

## Risks / Trade-offs

- [Backend assets are unavailable] → Auto mode retains mocks and shows the source/fallback reason.
- [Payload fields differ by search endpoint] → Central adapter handles nullable fields and assigns stable card IDs.
- [Large base64 images increase response cost] → Preserve existing card behavior; pagination/thumbnail transport remain separate backend work.
- [Live result quality differs from mock] → Preserve demo fallback and make source visible in the status bar.

## Migration Plan

1. Add the gateway and result normalization with demo mode as default.
2. Set `VITE_SEARCH_API_BASE_URL` and `VITE_SEARCH_MODE=live` in a local untracked environment file when a real FastAPI server is ready.
3. Validate each supported request mode against Swagger/backend assets.
4. Roll back instantly by removing `VITE_SEARCH_MODE=live` or setting it to `demo`; no code rollback is needed.

## Open Questions

- Which deployed URL and port will host the FastAPI service for the final demo?
- Which live result fields are guaranteed for temporal, OCR, and ASR results after the data pipeline is provisioned?
