"""Normalize metadata_clip.json to have a consistent schema."""

import json
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    backend_root = Path(__file__).resolve().parent.parent
    metadata_path = backend_root / "src" / "dict" / "metadata_clip.json"
    
    if not metadata_path.exists():
        logger.error(f"File not found: {metadata_path}")
        return

    # Load existing metadata
    logger.info(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} items. Normalizing...")

    fps = 25.0
    normalized_data = {}

    for faiss_id_str, info in data.items():
        faiss_id = int(faiss_id_str)
        global_frame_id = info.get("global_frame_id", 0)
        
        # Calculate timestamp based on global frame ID and standard 25.0 FPS
        timestamp = float(global_frame_id) / fps
        
        normalized_item = {
            "faiss_id": faiss_id,
            "video_id": info.get("video_id", ""),
            "frame_name": info.get("frame_name", ""),
            "frame_index": info.get("frame_index", faiss_id),
            "split": info.get("split", ""),
            "global_frame_id": global_frame_id,
            "timestamp": timestamp,
            "fps": fps
        }
        normalized_data[str(faiss_id)] = normalized_item

    # Save normalized metadata
    logger.info("Saving normalized metadata...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    logger.info("Metadata normalization complete!")
    
    # Print a sample to verify
    sample_key = list(normalized_data.keys())[0]
    logger.info(f"Sample item {sample_key}: {json.dumps(normalized_data[sample_key], indent=2)}")

if __name__ == "__main__":
    main()
