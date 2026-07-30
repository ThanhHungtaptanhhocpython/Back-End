# Phase 3 Task 3: ASR Extraction Pipeline (Stub)

## 1. Context and Coding Rules
Ensure compliance with `Backend/Back-End/.agents/AGENTS.md` and `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Develop a script to extract Automatic Speech Recognition (ASR) transcripts from audio files using Whisper, and align them to keyframe timestamps.

## 3. Requirements
- Create a script `scripts/extract_asr.py`.
- **Note:** Since we may not have the raw `.mp4` video files or audio available in the repository currently, implement this script as a robust *stub* or processor that expects an input directory of audio files.
- The script logic should:
  1. Accept an audio file path (`--audio-dir`).
  2. Simulate or use `faster-whisper` (add to requirements if necessary) to generate timestamped segments.
  3. Map each timestamped transcript segment (`start_time`, `end_time`) to the *nearest* visual keyframe by comparing timestamps with `metadata_clip.json`.
  4. Output the results to `src/dict/asr_results.json`.

## 4. Expected Output & Reporting
- Generate a `phase3_task3_report.md` outlining the ASR mapping logic (how you align audio timestamps to Faiss ID timestamps).
