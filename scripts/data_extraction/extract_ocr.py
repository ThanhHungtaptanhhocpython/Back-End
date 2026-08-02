"""OCR Extraction Pipeline.

Extracts text from video keyframes using EasyOCR and saves the results
for Elasticsearch indexing.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Try importing easyocr, but allow the script to fail gracefully if not installed
try:
    import easyocr
except ImportError:
    easyocr = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Extract OCR from keyframes.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of frames to process for testing.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results.")
    args = parser.parse_args()

    if easyocr is None:
        logger.error("EasyOCR is not installed. Please run `pip install easyocr`.")
        sys.exit(1)

    backend_root = Path(__file__).resolve().parent.parent
    metadata_path = backend_root / "src" / "dict" / "metadata_clip.json"
    keyframes_root = backend_root / "src" / "data" / "Keyframes"
    output_path = backend_root / "src" / "dict" / "ocr_results.json"

    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    logger.info("Loading metadata...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Initialize EasyOCR reader (Vietnamese and English)
    logger.info("Initializing EasyOCR reader (vi, en)... This may download model weights.")
    # Use GPU if available, else CPU
    reader = easyocr.Reader(['vi', 'en'], gpu=True)

    results = []
    processed_count = 0

    for faiss_id_str, info in metadata.items():
        if processed_count >= args.limit:
            break

        # Construct image path
        # The split might dictate the subfolder, or they might all be in one folder. 
        # Checking both direct and split-based paths.
        frame_name = info.get("frame_name")
        split = info.get("split", "")
        
        # Try finding the image
        img_path = keyframes_root / split / frame_name
        if not img_path.exists():
            img_path = keyframes_root / frame_name
        
        if not img_path.exists():
            # Skip if image doesn't exist locally (common if only a subset is downloaded)
            continue

        logger.info(f"Processing image {processed_count + 1}/{args.limit}: {frame_name}")
        
        # Perform OCR
        try:
            # detail=0 returns only the text strings, not bounding boxes
            detected_text_list = reader.readtext(str(img_path), detail=0)
            
            if detected_text_list:
                full_text = " ".join(detected_text_list)
                
                # Construct document matching the ES aic_ocr schema requirements
                doc = {
                    "faiss_id": info["faiss_id"],
                    "video_id": info["video_id"],
                    "frame_name": info["frame_name"],
                    "split": info["split"],
                    "global_frame_id": info["global_frame_id"],
                    "timestamp": info["timestamp"],
                    "ocr_text": full_text,
                    "language": "vi" # default for now
                }
                results.append(doc)
        except Exception as e:
            logger.error(f"Failed to process {frame_name}: {e}")

        processed_count += 1

    logger.info(f"Extracted OCR text from {len(results)} frames.")
    
    # Save results
    if results or args.force:
        # If output already exists, we might want to merge, but for this pipeline
        # we will just write/overwrite for the subset we processed.
        # In a real run without --limit, it would overwrite everything.
        
        existing_results = []
        if output_path.exists() and not args.force:
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        # Append new results to existing, avoiding duplicates by faiss_id
        existing_dict = {doc["faiss_id"]: doc for doc in existing_results}
        for doc in results:
            existing_dict[doc["faiss_id"]] = doc
            
        final_list = list(existing_dict.values())
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(final_list)} total OCR records to {output_path}")

if __name__ == "__main__":
    main()
