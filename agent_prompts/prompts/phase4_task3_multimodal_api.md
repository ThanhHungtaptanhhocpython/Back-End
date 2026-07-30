# Phase 4 Task 3: Expose Multimodal API

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Expose the new Adaptive Fusion capabilities through the FastAPI router so the frontend can use it.

## 3. Requirements
- Update `src/api/routers/search_router.py`.
- Create a new endpoint: `POST /users/multimodalsearch`.
- It should accept a `TextSearchRequest` (query string, topk).
- The route controller calls the `FusionService`, passing the user's query.
- Format the response using the `BaseResponse` schema.
- **Important**: Ensure the returned items include the image base64 (or URL), metadata (`faiss_id`, `video_id`, `timestamp`), and the `score_breakdown`.
- Write unit tests (`tests/test_phase4_task3.py`) to verify the endpoint routes correctly and triggers the query planner + fusion logic.

## 4. Expected Output & Reporting
- Generate a `phase4_task3_report.md` explaining the new API endpoint and demonstrating a sample response containing the `score_breakdown`.
