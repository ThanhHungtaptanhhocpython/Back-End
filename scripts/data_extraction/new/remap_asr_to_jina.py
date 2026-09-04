#!/usr/bin/env python
"""Map legacy ASR segments to the final Jina keyframe corpus.

Whisper transcripts are video-time based and can be reused after a keyframe
re-extraction. This script preserves every ASR segment, maps its midpoint to
the closest Jina keyframe in the same video, and replaces the legacy visual
identifier with the Jina vector ID.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:  # Import style when called from the repository root or as a script.
    from scripts.data_extraction.new.remap_ocr_to_jina import (
        _as_finite_float,
        build_jina_keyframe_index,
        canonical_video_key,
        nearest_frame,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from remap_ocr_to_jina import (  # type: ignore
        _as_finite_float,
        build_jina_keyframe_index,
        canonical_video_key,
        nearest_frame,
    )


def backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"ASR JSON must be a list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def segment_midpoint(doc: dict[str, Any]) -> float | None:
    start = _as_finite_float(doc.get("start_time", doc.get("start")))
    end = _as_finite_float(doc.get("end_time", doc.get("end")))
    if start is None:
        return None
    return start if end is None else (start + end) / 2.0


def remap_asr(
    asr_docs: list[dict[str, Any]],
    keyframe_index: dict[str, dict[str, Any]],
    max_delta_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], Counter]:
    """Attach Jina visual identity to every reusable ASR segment."""
    stats: Counter = Counter()
    output: list[dict[str, Any]] = []

    for raw_doc in asr_docs:
        doc = dict(raw_doc)
        source_video_id = str(doc.get("video_id") or "").strip()
        midpoint = segment_midpoint(doc)
        video_key = canonical_video_key(source_video_id, doc.get("split"))
        if not source_video_id or not video_key or midpoint is None:
            stats["skipped_missing_video_or_timestamp"] += 1
            continue

        video_entry = keyframe_index.get(video_key)
        if video_entry is None:
            stats["skipped_video_not_in_jina"] += 1
            continue
        frame = nearest_frame(video_entry, midpoint)
        if frame is None:
            stats["skipped_video_without_frames"] += 1
            continue

        delta = float(frame["timestamp"]) - midpoint
        if max_delta_seconds is not None and abs(delta) > max_delta_seconds:
            stats["skipped_timestamp_delta"] += 1
            continue

        legacy_id = doc.get("nearest_faiss_id")
        doc.update(
            {
                "video_id": frame["video_id"],
                "parent_namespace": frame["parent_namespace"],
                "split": frame["parent_namespace"],
                "nearest_vector_id": frame["vector_id"],
                # Existing Elasticsearch/TRAKE consumers currently read this field.
                "nearest_faiss_id": frame["vector_id"],
                "nearest_frame_id": frame["frame_id"],
                "nearest_frame_name": frame["frame_name"],
                "nearest_frame_path": frame["frame_path"],
                "nearest_source_frame_idx": frame["source_frame_idx"],
                "nearest_global_frame_id": frame["source_frame_idx"],
                "nearest_timestamp": frame["timestamp"],
                "asr_source_midpoint": midpoint,
                "alignment_delta_seconds": round(delta, 6),
                "alignment_source": "jina_nearest_timestamp",
                "legacy_nearest_faiss_id": legacy_id,
            }
        )
        output.append(doc)
        stats["aligned"] += 1

    output.sort(
        key=lambda item: (
            str(item.get("video_id") or ""),
            _as_finite_float(item.get("start_time", item.get("start"))) or 0.0,
        )
    )
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
    parser = argparse.ArgumentParser(description="Map legacy ASR JSON to nearest Jina keyframes by segment midpoint.")
    parser.add_argument("--asr-in", type=Path, default=root / "src" / "dict" / "asr_results.json")
    parser.add_argument("--global-ids", type=Path, required=True, help="Final Jina global_ids.parquet.")
    parser.add_argument("--output", type=Path, default=root / "src" / "dict" / "asr_results_jina.json")
    parser.add_argument("--max-delta-seconds", type=float, default=None, help="Optional guard; omit to preserve all ASR segments.")
    parser.add_argument("--limit", type=int, default=None, help="Limit ASR docs for a smoke test.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print stats without writing JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_delta_seconds is not None and (args.max_delta_seconds < 0 or not math.isfinite(args.max_delta_seconds)):
        print("--max-delta-seconds must be a finite non-negative value.", file=sys.stderr)
        return 2
    asr_path = args.asr_in.resolve()
    global_ids_path = args.global_ids.resolve()
    output_path = args.output.resolve()
    if not asr_path.is_file():
        print(f"ASR input not found: {asr_path}", file=sys.stderr)
        return 1
    if not global_ids_path.is_file():
        print(f"Jina global_ids parquet not found: {global_ids_path}", file=sys.stderr)
        return 1

    asr_docs = load_json_list(asr_path)
    if args.limit is not None:
        asr_docs = asr_docs[: max(0, args.limit)]
    keyframe_index = build_jina_keyframe_index(global_ids_path)
    remapped, stats = remap_asr(asr_docs, keyframe_index, args.max_delta_seconds)

    print(f"ASR input docs: {len(asr_docs)}")
    print(f"Jina videos: {len(keyframe_index)}")
    print(f"Stats: {dict(stats)}")
    print("Sample remapped documents:")
    for doc in remapped[:3]:
        # Windows terminals may use a legacy code page; keep samples printable.
        print(json.dumps(doc, ensure_ascii=True))

    if args.dry_run:
        return 0
    write_json(output_path, remapped)
    print(f"Jina-aligned ASR written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
