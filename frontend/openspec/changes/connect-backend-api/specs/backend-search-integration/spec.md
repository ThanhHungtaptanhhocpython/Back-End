## ADDED Requirements

### Requirement: Configurable backend search mode
The frontend SHALL select `demo`, `auto`, or `live` search mode from Vite environment configuration. In `auto` mode, the frontend MUST use the FastAPI backend only when a backend base URL is configured and MUST otherwise use the local mock engine.

#### Scenario: No backend configuration
- **WHEN** the application starts with no backend base URL configured
- **THEN** search requests SHALL use the local mock engine and identify the source as a demo result.

#### Scenario: Explicit live mode
- **WHEN** the application starts with live mode and a configured backend base URL
- **THEN** search requests SHALL be sent to the FastAPI backend.

### Requirement: Search request routing
The frontend SHALL map text, image-pivot, temporal, OCR, ASR, and multimodal query modes to their corresponding FastAPI `/users` routes and request formats.

#### Scenario: Multimodal text search
- **WHEN** a user submits a multimodal text query in live mode
- **THEN** the frontend SHALL POST the query and positive `topk` to `/users/multimodalsearch`.

#### Scenario: Image-pivot search
- **WHEN** a user pivots from an existing result in live mode
- **THEN** the frontend SHALL POST the result's FAISS identifier and `topk` as multipart form data to `/users/imagesearch`.

### Requirement: Backend response normalization
The frontend SHALL normalize successful FastAPI response envelopes and backend result records into the existing result-grid item shape while retaining backend score metadata.

#### Scenario: Successful backend response
- **WHEN** a FastAPI endpoint returns `success: true` with `data.items`
- **THEN** the workspace SHALL render normalized items and display the backend response's total item count.

#### Scenario: Empty backend response
- **WHEN** a FastAPI endpoint returns a successful empty `data.items` array
- **THEN** the workspace SHALL render an empty result state without substituting mock results.

### Requirement: Demo-safe fallback and error visibility
The frontend SHALL keep mock results available for demo mode and for transport/runtime failure in auto mode. It MUST NOT replace a valid backend validation error with mock results.

#### Scenario: Backend network failure in auto mode
- **WHEN** a live backend request fails because the server or required runtime assets are unavailable
- **THEN** the frontend SHALL return mock results and mark the search source as fallback demo data.

#### Scenario: Backend validation error
- **WHEN** the backend returns a structured non-success response or HTTP validation error
- **THEN** the frontend SHALL show the error and SHALL NOT hide it by using mock results.
