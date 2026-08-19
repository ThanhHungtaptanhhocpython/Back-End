#!/usr/bin/env python
"""Extract keyframes using Scene Boundary Detection (TransNetV2).

This script:
1. Automatically clones TransNetV2 if not available.
2. Scans videos in the given directory.
3. Uses TransNetV2 to segment the video into scenes.
4. Extracts the middle frame of each scene at its original resolution.
5. Saves frames as WebP to the output directory.
6. Builds a new metadata_clip_v2.json file mapping Faiss IDs to these new frames.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".wmv"}


def ensure_transnetv2(repo_dir: Path):
    """Ensure TransNetV2 is available, clone if not."""
    if not (repo_dir / "TransNetV2" / "inference" / "transnetv2.py").exists():
        logging.info("Cloning TransNetV2 repository into %s...", repo_dir)
        repo_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/soCzech/TransNetV2.git"],
                cwd=str(repo_dir),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            logging.error("Failed to clone TransNetV2. Please install git or clone manually: %s", e)
            sys.exit(1)
    
    # Add to sys.path so we can import it
    inference_path = str(repo_dir / "TransNetV2" / "inference")
    if inference_path not in sys.path:
        sys.path.append(inference_path)


def save_frame(frame_bgr, destination: Path) -> tuple[int, int]:
    """Save OpenCV BGR frame to lossless WebP."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    image.save(destination, "WEBP", lossless=True, quality=100, method=6)
    return image.size


def process_video(
    video_path: Path, 
    model, 
    output_root: Path, 
    split_name: str,
    video_id: str,
    global_faiss_id: int
) -> tuple[int, list[dict]]:
    """Process a single video: predict scenes, extract middle frames, return metadata."""
    logging.info("Processing video: %s", video_id)
    start_time = time.time()
    
    # 1. Detect scenes with TransNetV2
    try:
        _, single_frame_predictions, _ = model.predict_video(str(video_path))
        scenes = model.predictions_to_scenes(single_frame_predictions)
    except Exception as e:
        logging.error("TransNetV2 failed on %s: %s", video_id, e)
        return global_faiss_id, []

    # 2. Extract original frames using OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error("Cannot open video for reading: %s", video_path)
        return global_faiss_id, []

    video_metadata = []
    
    for i, scene in enumerate(scenes):
        start_frame, end_frame = scene
        middle_frame_idx = int((start_frame + end_frame) // 2)
        
        # Read the high-res frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            logging.warning("Could not read frame %d for scene %d in %s", middle_frame_idx, i, video_id)
            continue
            
        frame_name = f"keyframe_{i:04d}.webp"
        dest_path = output_root / split_name / video_id / frame_name
        
        width, height = save_frame(frame_bgr, dest_path)
        
        # Build metadata entry
        meta_entry = {
            "faiss_id": global_faiss_id,
            "split": split_name,
            "video_id": video_id,
            "frame_name": frame_name,
            "frame_index": i,
            "global_frame_id": middle_frame_idx,
            "resolution": f"{width}x{height}",
            "scene_start": int(start_frame),
            "scene_end": int(end_frame)
        }
        video_metadata.append(meta_entry)
        global_faiss_id += 1

    cap.release()
    logging.info("Extracted %d scenes from %s in %.2fs", len(scenes), video_id, time.time() - start_time)
    
    return global_faiss_id, video_metadata


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Extract keyframes using SBD (TransNetV2).")
    parser.add_argument("--video-root", type=Path, required=True, help="Directory containing source videos.")
    parser.add_argument("--output-root", type=Path, default=root / "src" / "data" / "Keyframes", help="Output Keyframes dir.")
    parser.add_argument("--metadata-out", type=Path, default=root / "src" / "dict" / "metadata_clip_v2.json", help="Output metadata file.")
    parser.add_argument("--split-name", type=str, default="Train", help="Split name to organize videos (e.g., Train, Test).")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s")

    # 1. Setup TransNetV2
    utils_dir = Path(__file__).resolve().parents[1] / "utils"
    ensure_transnetv2(utils_dir)
    
    try:
        from transnetv2 import TransNetV2
    except ImportError as e:
        logging.error("Failed to import TransNetV2: %s", e)
        return 1

    # Initialize model
    logging.info("Initializing TransNetV2 model (requires GPU)...")
    model = TransNetV2()
    
    # 2. Scan videos
    video_root = args.video_root.resolve()
    if not video_root.exists():
        logging.error("Video root does not exist: %s", video_root)
        return 1
        
    videos = [p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        logging.error("No videos found in %s", video_root)
        return 1
        
    logging.info("Found %d videos to process.", len(videos))
    
    # 3. Process each video
    global_faiss_id = 0
    all_metadata = {}
    
    for video_path in sorted(videos):
        # Infer video_id from filename (e.g., "L30_V001.mp4" -> "L30_V001")
        video_id = video_path.stem
        
        global_faiss_id, video_meta_list = process_video(
            video_path=video_path,
            model=model,
            output_root=args.output_root,
            split_name=args.split_name,
            video_id=video_id,
            global_faiss_id=global_faiss_id
        )
        
        for item in video_meta_list:
            fid = item.pop("faiss_id")
            all_metadata[str(fid)] = item

    # 4. Save metadata
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_out.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        
    logging.info("Successfully extracted %d total keyframes.", len(all_metadata))
    logging.info("Saved metadata to %s", args.metadata_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
