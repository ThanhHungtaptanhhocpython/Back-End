#!/usr/bin/env python
"""Map legacy OCR evidence to the final Jina keyframe corpus.

This script does not rerun OCR.  It preserves the recognized text from an old
OCR JSON file, then assigns every document to the nearest Jina keyframe in the
same video using the authoritative timestamps in ``global_ids.parquet``.

The output keeps ``faiss_id`` equal to the Jina ``vector_id`` only as a
backwards-compatible alias for current Elasticsearch consumers.  New code
should prefer ``vector_id`` and ``frame_path``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VIDEO_PREFIX_RE = re.compile(r"(L\d{2,3})", re.IGNORECASE)


def backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_video_key(video_id: Any, split: Any = "") -> str:
    """Return a stable full video key, for example ``L21_V001``."""
    video = str(video_id or "").strip().upper()
    if not video:
        return ""
    if re.match(r"^L\d{2,3}_", video):
        return video

    prefix_match = VIDEO_PREFIX_RE.search(str(split or "").upper())
    return f"{prefix_match.group(1).upper()}_{video}" if prefix_match else video


def _as_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"OCR JSON must be a list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def build_jina_keyframe_index(global_ids_path: Path) -> dict[str, dict[str, Any]]:
    """Load per-frame Jina metadata and group it by video timestamp."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas and a parquet engine (for example pyarrow) are required.") from exc

    table = pd.read_parquet(global_ids_path)
    required = {"vector_id", "video_id", "frame_path", "timestamp", "source_frame_idx"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Jina global_ids parquet is missing columns: {missing}")
    if table["vector_id"].isna().any() or not table["vector_id"].is_unique:
        raise ValueError("Jina global_ids parquet has missing or duplicate vector_id values.")
    if table[["video_id", "frame_path", "timestamp", "source_frame_idx"]].isna().any().any():
        raise ValueError("Jina global_ids parquet has missing video/frame/timestamp/source-frame values.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_dict("records"):
        timestamp = _as_finite_float(row.get("timestamp"))
        if timestamp is None:
            raise ValueError("Jina global_ids parquet contains a non-finite timestamp.")
        video_id = str(row["video_id"]).strip()
        namespace = str(row.get("parent_namespace") or "").strip()
        key = canonical_video_key(video_id, namespace)
        if not key:
            raise ValueError("Jina global_ids parquet contains an empty video_id.")
        frame_path = str(row["frame_path"]).replace("\\", "/").strip("/")
        if not frame_path:
            raise ValueError("Jina global_ids parquet contains an empty frame_path.")
        grouped[key].append(
            {
                "vector_id": int(row["vector_id"]),
                "video_id": video_id,
                "parent_namespace": namespace or frame_path.split("/", 1)[0],
                "frame_id": str(row.get("frame_id") or Path(frame_path).stem),
                "frame_path": frame_path,
                "frame_name": Path(frame_path).name,
                "timestamp": timestamp,
                "source_frame_idx": int(row["source_frame_idx"]),
            }
        )

    index: dict[str, dict[str, Any]] = {}
    for key, frames in grouped.items():
        frames.sort(key=lambda item: (item["timestamp"], item["vector_id"]))
        index[key] = {"frames": frames, "timestamps": [item["timestamp"] for item in frames]}
    return index


def nearest_frame(video_entry: dict[str, Any], timestamp: float) -> dict[str, Any] | None:
    frames = video_entry["frames"]
    timestamps = video_entry["timestamps"]
    if not frames:
        return None
    position = bisect.bisect_left(timestamps, timestamp)
    candidates: list[dict[str, Any]] = []
    if position < len(frames):
        candidates.append(frames[position])
    if position > 0:
        candidates.append(frames[position - 1])
    return min(candidates, key=lambda item: abs(item["timestamp"] - timestamp)) if candidates else None


def remap_ocr(
    ocr_docs: list[dict[str, Any]],
    keyframe_index: dict[str, dict[str, Any]],
    max_delta_seconds: float,
) -> tuple[list[dict[str, Any]], Counter]:
    """Return OCR docs enriched with final Jina frame identity fields."""
    stats: Counter = Counter()
    output: list[dict[str, Any]] = []
    best_by_key: dict[tuple[int, str], tuple[float, dict[str, Any]]] = {}

    for raw_doc in ocr_docs:
        text = str(raw_doc.get("ocr_text") or "").strip()
        source_timestamp = _as_finite_float(raw_doc.get("timestamp"))
        video_key = canonical_video_key(raw_doc.get("video_id"), raw_doc.get("split"))
        if not text:
            stats["skipped_empty_text"] += 1
            continue
        if not video_key or source_timestamp is None:
            stats["skipped_missing_video_or_timestamp"] += 1
            continue

        video_entry = keyframe_index.get(video_key)
        if video_entry is None:
            stats["skipped_video_not_in_jina"] += 1
            continue
        frame = nearest_frame(video_entry, source_timestamp)
        if frame is None:
            stats["skipped_video_without_frames"] += 1
            continue

        delta = frame["timestamp"] - source_timestamp
        if abs(delta) > max_delta_seconds:
            stats["skipped_timestamp_delta"] += 1
            continue

        doc = {
            "vector_id": frame["vector_id"],
            # Kept until all existing fusion consumers use vector_id directly.
            "faiss_id": frame["vector_id"],
            "video_id": frame["video_id"],
            "parent_namespace": frame["parent_namespace"],
            "split": frame["parent_namespace"],
            "frame_id": frame["frame_id"],
            "frame_name": frame["frame_name"],
            "frame_path": frame["frame_path"],
            "source_frame_idx": frame["source_frame_idx"],
            "global_frame_id": frame["source_frame_idx"],
            "timestamp": frame["timestamp"],
            "ocr_source_timestamp": source_timestamp,
            "alignment_delta_seconds": round(delta, 6),
            "alignment_source": "jina_nearest_timestamp",
            "ocr_text": text,
            "language": str(raw_doc.get("language") or "vi"),
            "legacy_faiss_id": raw_doc.get("faiss_id"),
            "legacy_frame_name": raw_doc.get("frame_name"),
            "legacy_global_frame_id": raw_doc.get("global_frame_id"),
        }

        dedupe_key = (frame["vector_id"], text.casefold())
        existing = best_by_key.get(dedupe_key)
        if existing is not None:
            stats["deduplicated"] += 1
            if abs(delta) >= existing[0]:
                continue
        best_by_key[dedupe_key] = (abs(delta), doc)
        stats["aligned"] += 1

    output = [entry[1] for entry in best_by_key.values()]
    output.sort(key=lambda item: (item["vector_id"], item["ocr_text"].casefold()))
    stats["output_docs"] = len(output)
    stats["jina_videos"] = len(keyframe_index)
    return output, stats


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Map legacy OCR JSON to nearest Jina keyframes by timestamp.")
    parser.add_argument("--ocr-in", type=Path, required=True, help="Legacy OCR JSON list.")
    parser.add_argument("--global-ids", type=Path, required=True, help="Final Jina global_ids.parquet.")
    parser.add_argument("--output", type=Path, default=root / "src" / "dict" / "ocr_results_jina.json")
    parser.add_argument("--max-delta-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit OCR docs for a quick smoke test.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print stats without writing JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_delta_seconds < 0:
        print("--max-delta-seconds must be non-negative.", file=sys.stderr)
        return 2
    ocr_path = args.ocr_in.resolve()
    global_ids_path = args.global_ids.resolve()
    output_path = args.output.resolve()
    if not ocr_path.is_file():
        print(f"OCR input not found: {ocr_path}", file=sys.stderr)
        return 1
    if not global_ids_path.is_file():
        print(f"Jina global_ids parquet not found: {global_ids_path}", file=sys.stderr)
        return 1

    ocr_docs = load_json_list(ocr_path)
    if args.limit is not None:
        ocr_docs = ocr_docs[: max(0, args.limit)]
    keyframe_index = build_jina_keyframe_index(global_ids_path)
    remapped, stats = remap_ocr(ocr_docs, keyframe_index, args.max_delta_seconds)

    print(f"OCR input docs: {len(ocr_docs)}")
    print(f"Jina videos: {len(keyframe_index)}")
    print(f"Stats: {dict(stats)}")
    print("Sample remapped documents:")
    for doc in remapped[:3]:
        # Windows terminals may use a legacy code page; keep samples printable.
        print(json.dumps(doc, ensure_ascii=True))

    if args.dry_run:
        return 0
    write_json(output_path, remapped)
    print(f"Jina-aligned OCR written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
