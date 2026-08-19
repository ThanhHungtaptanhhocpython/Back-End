"""ASR Extraction Pipeline.

Extracts text from audio files using Whisper Large-v3 Turbo and aligns
timestamped text segments to the nearest visual keyframes.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Try importing torch and transformers
try:
    import torch
    from transformers import pipeline
except ImportError:
    torch = None
    pipeline = None

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

    if torch is None or pipeline is None:
        logger.error("transformers or torch is not installed. Please run `pip install transformers torch`.")
        sys.exit(1)

    if not args.audio_dir or not os.path.isdir(args.audio_dir):
        logger.error("Please provide a valid --audio-dir containing the audio files.")
        sys.exit(1)

    backend_root = Path(__file__).resolve().parent.parent
    metadata_path = backend_root / "src" / "dict" / "metadata_clip.json"
    output_path = backend_root / "src" / "dict" / "asr_results.json"

    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    logger.info("Loading metadata...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

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
    audio_files = [f for f in os.listdir(args.audio_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4', '.flac'))]
    
    logger.info(f"Found {len(audio_files)} audio files in {args.audio_dir}.")

    for audio_file in audio_files:
        video_id = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(args.audio_dir, audio_file)
        
        logger.info(f"Processing {audio_file}...")
        try:
            # Assumed vietnamese language, adjust generate_kwargs if needed
            out = pipe(audio_path, generate_kwargs={"language": "vietnamese"})
            chunks = out.get("chunks", [])
            
            for chunk in chunks:
                timestamp = chunk.get("timestamp", (0.0, 0.0))
                start_time, end_time = timestamp
                text = chunk.get("text", "").strip()
                
                if end_time is None:
                    end_time = start_time + 5.0
                
                mid_time = (start_time + end_time) / 2.0
                nearest_frame = find_nearest_keyframe(mid_time, video_id, metadata)
                
                if nearest_frame:
                    doc = {
                        "video_id": video_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                        "nearest_faiss_id": nearest_frame["faiss_id"],
                        "nearest_frame_name": nearest_frame["frame_name"]
                    }
                    results.append(doc)
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")

    logger.info(f"Generated {len(results)} aligned ASR documents.")
    
    # Save results
    if results or args.force:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved ASR records to {output_path}")

if __name__ == "__main__":
    main()
