# Phase 3 Task 2 Report: OCR Extraction Pipeline

## Status: ✅ Completed

---

## What Changed

1. **Requirements Update**:
   - Added `easyocr` to `requirements.txt`. It is a lightweight, widely used OCR engine that supports both Vietnamese (`vi`) and English (`en`) out-of-the-box and can run on CPU or GPU.

2. **Created `scripts/extract_ocr.py`**:
   - Developed a standalone script that initializes the EasyOCR engine.
   - It reads the normalized `metadata_clip.json` (created in Task 1) to determine which Keyframe images to process.
   - For each frame, it extracts all detected text and concatenates it into a single string (`ocr_text`).
   - The script builds a JSON document that perfectly matches the `aic_ocr` Elasticsearch mapping (containing `faiss_id`, `video_id`, `timestamp`, `ocr_text`, etc.).
   - Includes a `--limit` flag (default 10) so developers can test the extraction without processing all 196,839 images.

---

## Why These Decisions

- **Why EasyOCR?**: `easyocr` is easy to install locally (via `pip`) and does not require complex binary setups like Tesseract. It offers a good balance between speed and accuracy for Vietnamese text in video frames. In production, this can be swapped out for a heavier model like PaddleOCR or an MLLM like Gemini, but EasyOCR is perfect for the baseline pipeline.
- **Offline Output (`ocr_results.json`)**: The script does not push directly to Elasticsearch. Instead, it saves the output to `src/dict/ocr_results.json`. This decoupling is crucial: extraction can run for 10 hours on a GPU machine, generate the JSON, and then the JSON can be loaded into Elasticsearch on the API server in seconds (which is what Task 4 will do).

---

## How to Test

To test the OCR extraction locally (limited to 5 frames):
```bash
# First, install the new dependency
pip install easyocr

# Then run the extraction script
python scripts/extract_ocr.py --limit 5
```
You will find the output in `src/dict/ocr_results.json`.
