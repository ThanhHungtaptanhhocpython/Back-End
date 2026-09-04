"""Reindex Jina-aligned ASR documents into Elasticsearch.

Use --recreate to delete and rebuild the aic_asr index before indexing. This is
needed once after repairing ASR alignment if the old index already contains bad
or duplicated documents.
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
    parser = argparse.ArgumentParser(description="Reindex Jina-aligned ASR documents into Elasticsearch.")
    parser.add_argument("--asr-path", type=Path, default=dict_dir / "asr_results_jina.json")
    parser.add_argument("--mappings-path", type=Path, default=dict_dir / "es_samples" / "mappings.json")
    parser.add_argument("--index", default="aic_asr")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the ASR index before indexing.")
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
    asr_path = args.asr_path.resolve()
    mappings_path = args.mappings_path.resolve()

    if not asr_path.exists():
        print(f"ASR file not found: {asr_path}", file=sys.stderr)
        return 1

    docs = load_json(asr_path)
    if not isinstance(docs, list):
        print(f"ASR file must be a JSON list: {asr_path}", file=sys.stderr)
        return 1

    missing_alignment = [doc for doc in docs if isinstance(doc, dict) and doc.get("nearest_faiss_id") is None]
    unique_nearest = len({doc.get("nearest_faiss_id") for doc in docs if isinstance(doc, dict)})
    print(f"ASR docs: {len(docs)}")
    print(f"Unique nearest_faiss_id: {unique_nearest}")
    print(f"Docs missing nearest_faiss_id: {len(missing_alignment)}")

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

    indexed = processor.bulk_index_asr(
        docs,
        index=args.index,
        chunk_size=args.chunk_size,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    print(f"Indexed ASR docs: {indexed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())