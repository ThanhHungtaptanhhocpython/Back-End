"""ASR Extraction Pipeline.

Extracts text from audio files using Whisper Large-v3 Turbo and aligns
timestamped text segments to the nearest visual keyframes.
"""

import argparse
import bisect
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Try importing torch and transformers
try:
    import torch
    from transformers import pipeline
except ImportError:
    torch = None
    pipeline = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_to_prefix(split: str) -> str:
    clean = str(split or "").lower().replace("videos-", "").replace("videos_", "")
    if not clean:
        return ""
    return clean.split("-")[0].upper()


def metadata_video_key(info: dict[str, Any]) -> str:
    video_id = str(info.get("video_id") or "").strip()
    if not video_id:
        return ""
    if "_" in video_id:
        return video_id
    prefix = split_to_prefix(str(info.get("split") or info.get("namespace") or info.get("folder_key") or ""))
    return f"{prefix}_{video_id}" if prefix else video_id


def build_keyframe_index(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for faiss_id_key, raw_info in metadata.items():
        if not isinstance(raw_info, dict):
            continue
        info = dict(raw_info)
        try:
            faiss_id = int(info.get("faiss_id", faiss_id_key))
            timestamp = float(info.get("timestamp", 0.0))
        except (TypeError, ValueError):
            continue

        video_key = metadata_video_key(info)
        if not video_key:
            continue

        grouped[video_key].append(
            {
                "faiss_id": faiss_id,
                "frame_name": info.get("frame_name", ""),
                "frame_index": info.get("frame_index"),
                "global_frame_id": info.get("global_frame_id"),
                "timestamp": timestamp,
                "fps": info.get("fps"),
                "split": info.get("split"),
            }
        )

    index: dict[str, dict[str, Any]] = {}
    for video_key, frames in grouped.items():
        frames.sort(key=lambda item: item["timestamp"])
        index[video_key] = {
            "frames": frames,
            "timestamps": [item["timestamp"] for item in frames],
        }
    return index


def find_nearest_keyframe(target_time: float, video_id: str, keyframe_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Find the indexed keyframe closest to target_time for a full video id."""
    video_entry = keyframe_index.get(video_id)
    if not video_entry:
        return None

    frames = video_entry["frames"]
    timestamps = video_entry["timestamps"]
    pos = bisect.bisect_left(timestamps, target_time)
    candidates = []
    if pos < len(frames):
        candidates.append(frames[pos])
    if pos > 0:
        candidates.append(frames[pos - 1])
    return min(candidates, key=lambda item: abs(float(item["timestamp"]) - target_time)) if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Extract ASR from audio and align it to keyframes.")
    parser.add_argument("--audio-dir", type=str, default="", help="Path to directory containing audio files.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Path to metadata_clip.json.")
    parser.add_argument("--output-path", type=Path, default=None, help="Path to write aligned ASR JSON.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results.")
    args = parser.parse_args()

    if torch is None or pipeline is None:
        logger.error("transformers or torch is not installed. Please run `pip install transformers torch`.")
        sys.exit(1)

    if not args.audio_dir or not os.path.isdir(args.audio_dir):
        logger.error("Please provide a valid --audio-dir containing the audio files.")
        sys.exit(1)

    backend_root = Path(__file__).resolve().parents[3]
    metadata_path = args.metadata_path or backend_root / "src" / "dict" / "metadata_clip.json"
    output_path = args.output_path or backend_root / "src" / "dict" / "asr_results.json"

    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    logger.info("Loading metadata...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    keyframe_index = build_keyframe_index(metadata)
    logger.info(
        f"Indexed {sum(len(entry['frames']) for entry in keyframe_index.values())} keyframes "
        f"across {len(keyframe_index)} videos."
    )

    # Initialize Whisper Large-v3 Turbo model
    logger.info("Loading Whisper Large-v3 Turbo model... (this may take a while to download weights)")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        chunk_length_s=30,
        return_timestamps=True,
    )

    results = []
    audio_files = [f for f in os.listdir(args.audio_dir) if f.lower().endswith((".mp3", ".wav", ".m4a", ".mp4", ".flac"))]

    logger.info(f"Found {len(audio_files)} audio files in {args.audio_dir}.")

    for audio_file in audio_files:
        video_id = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(args.audio_dir, audio_file)

        logger.info(f"Processing {audio_file}...")
        try:
            out = pipe(audio_path, generate_kwargs={"language": "vietnamese"})
            chunks = out.get("chunks", [])

            for chunk in chunks:
                timestamp = chunk.get("timestamp", (0.0, 0.0))
                start_time, end_time = timestamp
                text = chunk.get("text", "").strip()

                if end_time is None:
                    end_time = start_time + 5.0

                mid_time = (start_time + end_time) / 2.0
                nearest_frame = find_nearest_keyframe(mid_time, video_id, keyframe_index)

                if nearest_frame:
                    doc = {
                        "video_id": video_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                        "nearest_faiss_id": nearest_frame["faiss_id"],
                        "nearest_frame_name": nearest_frame["frame_name"],
                        "nearest_global_frame_id": nearest_frame.get("global_frame_id"),
                        "nearest_timestamp": nearest_frame.get("timestamp"),
                        "alignment_delta_seconds": round(float(nearest_frame["timestamp"]) - float(mid_time), 6),
                        "alignment_source": "metadata_nearest_timestamp",
                        "split": nearest_frame.get("split"),
                        "fps": nearest_frame.get("fps"),
                    }
                    results.append(doc)
                else:
                    logger.warning(f"No keyframe metadata found for {video_id} at {mid_time:.3f}s.")
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")

    logger.info(f"Generated {len(results)} aligned ASR documents.")

    if results or args.force:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logger.info(f"Saved ASR records to {output_path}")


if __name__ == "__main__":
    main()
