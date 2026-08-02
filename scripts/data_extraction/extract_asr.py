"""ASR Extraction Pipeline (Stub).

This script simulates the extraction of ASR (Automatic Speech Recognition) 
from audio files using a model like Whisper. It aligns timestamped text segments
to the nearest visual keyframes.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_nearest_keyframe(target_time: float, video_id: str, metadata: dict) -> dict:
    """Find the keyframe in the metadata closest to the target_time for a given video."""
    best_diff = float('inf')
    best_info = None

    for faiss_id_str, info in metadata.items():
        if info.get("video_id") == video_id:
            diff = abs(info.get("timestamp", 0.0) - target_time)
            if diff < best_diff:
                best_diff = diff
                best_info = info

    return best_info

def main():
    parser = argparse.ArgumentParser(description="Extract ASR from audio (Stub).")
    parser.add_argument("--audio-dir", type=str, default="", help="Path to directory containing audio files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results.")
    args = parser.parse_args()

    backend_root = Path(__file__).resolve().parent.parent
    metadata_path = backend_root / "src" / "dict" / "metadata_clip.json"
    output_path = backend_root / "src" / "dict" / "asr_results.json"

    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    logger.info("Loading metadata...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # ------------------------------------------------------------------
    # STUB LOGIC: Simulate Whisper Output
    # In a real environment, you would use faster-whisper on args.audio_dir
    # ------------------------------------------------------------------
    logger.info("Simulating Whisper ASR extraction...")
    
    mock_whisper_segments = [
        {"video_id": "V001", "start": 0.0, "end": 2.5, "text": "Chào mừng các bạn đến với bản tin thời sự buổi tối ngày hôm nay."},
        {"video_id": "V001", "start": 2.5, "end": 5.0, "text": "Thưa quý vị, hiện tại tình hình giao thông đang ùn tắc nghiêm trọng."},
        {"video_id": "V002", "start": 8.0, "end": 12.0, "text": "Các bác sĩ tại bệnh viện Chợ Rẫy đang nỗ lực hết mình để cứu chữa."},
        {"video_id": "V003", "start": 48.0, "end": 52.0, "text": "Lực lượng chức năng đã có mặt tại hiện trường để phân luồng."},
        {"video_id": "V003", "start": 53.0, "end": 58.0, "text": "Khu vực nhà hàng bán đồ ăn nhanh này luôn đông đúc vào giờ trưa."}
    ]
    
    results = []

    logger.info("Aligning audio transcripts to visual keyframes...")
    for segment in mock_whisper_segments:
        # We align the audio segment to the visual frame closest to the middle of the spoken sentence
        mid_time = (segment["start"] + segment["end"]) / 2.0
        
        nearest_frame = find_nearest_keyframe(mid_time, segment["video_id"], metadata)
        
        if nearest_frame:
            doc = {
                "video_id": segment["video_id"],
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "nearest_faiss_id": nearest_frame["faiss_id"],
                "nearest_frame_name": nearest_frame["frame_name"]
            }
            results.append(doc)
        else:
            logger.warning(f"Could not find any keyframes for video {segment['video_id']}")

    logger.info(f"Generated {len(results)} aligned ASR documents.")
    
    # Save results
    if results or args.force:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved ASR records to {output_path}")

if __name__ == "__main__":
    main()
