"""Unified Master Indexing Script.

Reads the offline extraction outputs (OCR and ASR) and bulk inserts them 
into Elasticsearch.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Ensure backend root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.utils.elastic_processing import ElasticProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Master Indexing Pipeline...")
    
    dict_dir = BACKEND_ROOT / "src" / "dict"
    samples_dir = dict_dir / "es_samples"
    
    mappings_path = samples_dir / "mappings.json"
    ocr_path = dict_dir / "ocr_results.json"
    asr_path = dict_dir / "asr_results.json"

    processor = ElasticProcessor()

    # 1. Create or ensure indices exist
    if mappings_path.exists():
        logger.info("Loading ES mappings...")
        with open(mappings_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        processor.create_indices(mappings)
    else:
        logger.warning(f"Mappings file not found at {mappings_path}. Index creation skipped.")

    # 2. Index OCR Data
    if ocr_path.exists():
        logger.info(f"Loading OCR results from {ocr_path}...")
        with open(ocr_path, "r", encoding="utf-8") as f:
            try:
                ocr_docs = json.load(f)
                if ocr_docs:
                    logger.info(f"Bulk indexing {len(ocr_docs)} OCR documents...")
                    processor.bulk_index_ocr(ocr_docs)
                else:
                    logger.info("OCR results file is empty.")
            except json.JSONDecodeError:
                logger.error("Failed to parse OCR results JSON.")
    else:
        logger.warning(f"OCR results not found at {ocr_path}. Have you run extract_ocr.py?")

    # 3. Index ASR Data
    if asr_path.exists():
        logger.info(f"Loading ASR results from {asr_path}...")
        with open(asr_path, "r", encoding="utf-8") as f:
            try:
                asr_docs = json.load(f)
                if asr_docs:
                    logger.info(f"Bulk indexing {len(asr_docs)} ASR documents...")
                    processor.bulk_index_asr(asr_docs)
                else:
                    logger.info("ASR results file is empty.")
            except json.JSONDecodeError:
                logger.error("Failed to parse ASR results JSON.")
    else:
        logger.warning(f"ASR results not found at {asr_path}. Have you run extract_asr.py?")

    logger.info("Master Indexing Pipeline finished successfully.")

if __name__ == "__main__":
    main()
