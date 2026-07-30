# Phase 3 Task 1 Report: Normalize Metadata Schema

## Status: ✅ Completed

---

## What Changed

1. **Created `scripts/normalize_metadata.py`**:
   - A standalone Python script designed to read the legacy `metadata_clip.json`, compute missing fields, and rewrite it with a strictly enforced standard schema.

2. **Executed Metadata Normalization**:
   - Successfully loaded and processed **196,839** keyframe entries.
   - Enforced the following schema for every entry:
     - `faiss_id` (Integer): Explicitly saved inside the object (converted from the dictionary key).
     - `video_id` (String): The source video ID.
     - `frame_name` (String): The original extracted image filename.
     - `frame_index` (Integer): The relative index of the keyframe in the Faiss list.
     - `split` (String): The dataset partition.
     - `global_frame_id` (Integer): The raw frame number in the original video.
     - **`timestamp` (Float)**: *[NEW]* Automatically calculated as `global_frame_id / fps`.
     - **`fps` (Float)**: *[NEW]* Fixed at `25.0` for this dataset.

---

## Why These Decisions

- **Absolute Keyframe Identity**: The new schema ensures that `faiss_id`, `video_id`, and `timestamp` are always present. This is the bedrock of Phase 3 and Phase 4. When the OCR script (Task 2) runs, it will know exactly what time in the video a piece of text appeared.
- **Timestamp Computation**: `global_frame_id` is useful internally, but temporal search (TRAKE) and ASR mapping rely heavily on real seconds (Timestamps). Calculating this once offline saves the FastAPI server from doing math on the fly for thousands of results during a search.

---

## Verification

The output sample verifies the calculation is correct (e.g., Frame 17 / 25 fps = 0.68s):

```json
{
  "faiss_id": 0,
  "video_id": "V001",
  "frame_name": "keyframe_L21_V001_0001.webp",
  "frame_index": 0,
  "split": "videos-l21-a",
  "global_frame_id": 17,
  "timestamp": 0.68,
  "fps": 25.0
}
```
