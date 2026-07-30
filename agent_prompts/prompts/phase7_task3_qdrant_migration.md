# Phase 7 Task 3: Qdrant Migration Evaluation (Optional)

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Evaluate if maintaining 3 separate `.bin` Faiss indices (OpenCLIP, SigLIP, BEiT-3) in memory is sustainable, and prepare the Qdrant database schema if migration is necessary.

## 3. Requirements
- Create a script `scripts/qdrant_migration_plan.py`.
- This script does NOT need to actually run the migration. It should define the Qdrant Collection schema using the `qdrant-client` python package.
- The schema should define a collection named `keyframes` with multiple named vectors:
  - `openclip` (Dim: 512, Distance: Cosine)
  - `siglip` (Dim: 768, Distance: Cosine)
  - `beit3` (Dim: 768, Distance: Cosine)
- Include metadata payload fields: `video_id`, `frame_name`, `timestamp`.
- Write a short mock function `mock_insert()` that demonstrates how to insert a single point containing all 3 vectors simultaneously.

## 4. Expected Output & Reporting
- Generate a `phase7_task3_report.md` documenting the Qdrant schema design and explaining the benefits of switching to a proper Vector DB (like payload filtering and multi-vector search) vs staying with raw Faiss.
