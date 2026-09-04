"""Reindex Jina-aligned OCR evidence into Elasticsearch.

Use ``--recreate`` after moving to a new visual corpus so stale legacy FAISS
IDs cannot remain in the ``aic_ocr`` index.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.utils.elastic_processing import ElasticProcessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    dict_dir = BACKEND_ROOT / "src" / "dict"
    parser = argparse.ArgumentParser(description="Reindex Jina-aligned OCR documents into Elasticsearch.")
    parser.add_argument("--ocr-path", type=Path, default=dict_dir / "ocr_results_jina.json")
    parser.add_argument("--mappings-path", type=Path, default=dict_dir / "es_samples" / "mappings.json")
    parser.add_argument("--index", default="aic_ocr")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the OCR index before indexing.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and print stats without touching Elasticsearch.")
    parser.add_argument("--chunk-size", type=int, default=250, help="Documents per bulk request.")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Bulk request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries for transient bulk failures.")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    ocr_path = args.ocr_path.resolve()
    mappings_path = args.mappings_path.resolve()
    if not ocr_path.is_file():
        print(f"OCR file not found: {ocr_path}", file=sys.stderr)
        return 1

    docs = load_json(ocr_path)
    if not isinstance(docs, list):
        print(f"OCR file must be a JSON list: {ocr_path}", file=sys.stderr)
        return 1

    valid_docs = [doc for doc in docs if isinstance(doc, dict)]
    missing_vector = [doc for doc in valid_docs if doc.get("vector_id") is None]
    missing_text = [doc for doc in valid_docs if not str(doc.get("ocr_text") or "").strip()]
    unique_vectors = len({doc.get("vector_id") for doc in valid_docs if doc.get("vector_id") is not None})
    print(f"OCR docs: {len(docs)}")
    print(f"Unique vector_id: {unique_vectors}")
    print(f"Docs missing vector_id: {len(missing_vector)}")
    print(f"Docs missing ocr_text: {len(missing_text)}")
    if missing_vector or missing_text:
        print("Refusing to index malformed OCR documents.", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    processor = ElasticProcessor(request_timeout=args.request_timeout, max_retries=args.max_retries)
    if args.recreate:
        if processor.es.indices.exists(index=args.index):
            processor.es.options(request_timeout=args.request_timeout).indices.delete(index=args.index)
            print(f"Deleted index: {args.index}")
        if mappings_path.exists():
            mappings = load_json(mappings_path)
            mapping = mappings.get(args.index, {}) if isinstance(mappings, dict) else {}
            processor.es.options(request_timeout=args.request_timeout).indices.create(index=args.index, body=mapping)
            print(f"Created index from mapping: {args.index}")
        else:
            processor.es.options(request_timeout=args.request_timeout).indices.create(index=args.index)
            print(f"Created index without explicit mapping: {args.index}")
    elif not processor.es.indices.exists(index=args.index):
        if mappings_path.exists():
            mappings = load_json(mappings_path)
            processor.create_indices({args.index: mappings.get(args.index, {})})
        else:
            processor.es.options(request_timeout=args.request_timeout).indices.create(index=args.index)

    indexed = processor.bulk_index_ocr(
        valid_docs,
        index=args.index,
        chunk_size=args.chunk_size,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    print(f"Indexed OCR docs: {indexed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())