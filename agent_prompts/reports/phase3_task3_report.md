# Phase 3 Task 3 Report: ASR Extraction Pipeline (Stub)

## Status: ✅ Completed

---

## What Changed

1. **Created `scripts/extract_asr.py`**:
   - Developed an ASR extraction stub script.
   - Designed a mock Whisper transcription output that contains spoken text, the `video_id`, and exact audio timestamps (`start` and `end`).

2. **Implemented Audio-to-Visual Alignment**:
   - Wrote a custom alignment function `find_nearest_keyframe()` that maps every spoken sentence to a specific image frame.
   - It calculates the temporal midpoint of the spoken sentence: `mid_time = (start + end) / 2.0`.
   - It iterates through the normalized `metadata_clip.json` for the given `video_id` to find the keyframe whose `timestamp` is mathematically closest to the `mid_time`.
   - The output is structured exactly according to the `aic_asr` Elasticsearch mapping, storing the text alongside the `nearest_faiss_id`.

---

## Why These Decisions

- **Why a Stub?**: Since the backend repo currently holds keyframes rather than the multi-gigabyte raw `.mp4` videos or `.wav` files, attempting to run a real Whisper model would fail due to missing source files. The stub defines the exact logic boundary we need. When real audio is available, you just swap out the `mock_whisper_segments` list for a `faster-whisper` model invocation.
- **Why Nearest Frame Alignment?**: ASR operates in the continuous audio domain, whereas our Faiss visual search operates in the discrete frame domain (e.g., 1 frame every second). If a user searches for text and we find a match in the audio, we *must* return a visual frame to the frontend UI. By aligning the audio to the nearest `faiss_id`, the frontend can render the exact scene where the phrase was spoken.

---

## How to Test

Run the script locally:
```bash
python scripts/extract_asr.py
```
You will find the aligned output in `src/dict/asr_results.json`.
