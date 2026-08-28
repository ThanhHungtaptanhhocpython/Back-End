"""Script to index sample OCR and ASR data into Elasticsearch."""

import json
import logging
import os
import sys

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.utils.elastic_processing import ElasticProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    processor = ElasticProcessor()

    samples_dir = os.path.join(BACKEND_ROOT, "src", "dict", "es_samples")
    mappings_path = os.path.join(samples_dir, "mappings.json")
    ocr_data_path = os.path.join(samples_dir, "sample_ocr.json")
    asr_data_path = os.path.join(samples_dir, "sample_asr.json")

    # 1. Create indices based on mappings
    if os.path.exists(mappings_path):
        with open(mappings_path, "r", encoding="utf-8") as f:
            mappings_dict = json.load(f)
        processor.create_indices(mappings_dict)
    else:
        logger.error(f"Mappings file not found at {mappings_path}")
        return

    # 2. Index OCR data
    if os.path.exists(ocr_data_path):
        with open(ocr_data_path, "r", encoding="utf-8") as f:
            ocr_docs = json.load(f)
        processor.bulk_index_ocr(ocr_docs)
    else:
        logger.warning(f"OCR sample data not found at {ocr_data_path}")

    # 3. Index ASR data
    if os.path.exists(asr_data_path):
        with open(asr_data_path, "r", encoding="utf-8") as f:
            asr_docs = json.load(f)
        processor.bulk_index_asr(asr_docs)
    else:
        logger.warning(f"ASR sample data not found at {asr_data_path}")

    logger.info("Indexing process completed.")

if __name__ == "__main__":
    main()
