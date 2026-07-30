# Phase 3 Task 2: OCR Extraction Pipeline

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Develop a standalone script to extract text from video keyframes using an OCR engine.

## 3. Requirements
- Create a script `scripts/extract_ocr.py`.
- Pick a fast local OCR engine (e.g., `EasyOCR` or `PaddleOCR`). You may need to add it to `requirements.txt`.
- The script should:
  1. Load the normalized `metadata_clip.json`.
  2. Iterate through a subset of keyframes in `src/data/Keyframes/` (or all of them if small enough).
  3. Run OCR on each image.
  4. Aggregate the detected text blocks into a single string per image.
  5. Output the results to a file `src/dict/ocr_results.json` mapping `faiss_id` to the extracted `ocr_text`.
- Keep in mind performance: Use batching if the chosen OCR framework supports it.

## 4. Expected Output & Reporting
- Generate a `phase3_task2_report.md` detailing which OCR library was chosen, its performance implications, and how to run the script.
