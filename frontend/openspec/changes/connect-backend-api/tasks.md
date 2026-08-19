## 1. Backend Search Boundary

- [x] 1.1 Add Vite-configured search mode and FastAPI URL resolution with a demo-safe default.
- [x] 1.2 Implement request routing for text, image-pivot, temporal, OCR, ASR, and multimodal searches.
- [x] 1.3 Normalize FastAPI envelopes and result records into the workstation card model, retaining ranking metadata.

## 2. Workspace Integration

- [x] 2.1 Replace the mock-only `runSearch` adapter path while preserving its result contract and fallback behavior.
- [x] 2.2 Replace the simulated backend ping with a health endpoint probe and surface live/demo/fallback status.
- [x] 2.3 Keep the chat copilot explicitly demo-only and show usable errors for non-fallback API failures.

## 3. Verification

- [x] 3.1 Add focused adapter tests for endpoint routing, response normalization, and fallback policy.
- [x] 3.2 Run formatting/linting/build verification and inspect the production preview behavior.
- [x] 3.3 Run GitNexus change detection and record the expected affected flows before handoff.
