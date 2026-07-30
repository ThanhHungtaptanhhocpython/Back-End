# Phase 3 Task 1: Normalize Metadata Schema

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Also, refer to Phase 3 in `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Standardize the primary metadata file (`src/dict/metadata_clip.json`) so that all modalities (Visual, OCR, ASR) can point to the same absolute keyframe identity.

## 3. Requirements
- Create a script `scripts/normalize_metadata.py`.
- The script should read the existing `metadata_clip.json`.
- It must ensure every item has the following strict schema:
  - `faiss_id` (integer) - *Matches the index in the faiss .bin file*
  - `video_id` (string) - *Extracted from the filename (e.g. V001)*
  - `frame_name` (string) - *The raw image filename*
  - `global_frame_id` (integer) - *Extracted from filename or existing metadata*
  - `split` (string) - *The dataset split directory*
  - `timestamp` (float) - *Calculate based on `global_frame_id` / `fps` (default to 25.0)*
  - `fps` (float) - *Set to 25.0*
- The script should overwrite (or output a new version of) `metadata_clip.json` with these consistent fields.

## 4. Expected Output & Reporting
- Generate a `phase3_task1_report.md` explaining the changes made to the JSON schema and how the script was tested.
