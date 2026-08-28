#!/usr/bin/env python
"""Realign ASR transcript segments to the nearest visual keyframes.

This repairs ASR JSON files where `nearest_faiss_id` was generated with an
incorrect or missing metadata mapping. It does not rerun Whisper; it only uses
ASR timestamps and keyframe metadata timestamps.
"""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def split_to_prefix(split: str) -> str:
    clean = str(split or "").lower().replace("videos-", "").replace("videos_", "")
    if not clean:
        return ""
    return clean.split("-")[0].upper()


def metadata_video_key(info: dict[str, Any]) -> str:
    video_id = str(info.get("video_id") or "").strip()
    if not video_id:
        return ""
    if "_" in video_id:
        return video_id
    prefix = split_to_prefix(str(info.get("split") or info.get("namespace") or info.get("folder_key") or ""))
    return f"{prefix}_{video_id}" if prefix else video_id


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_keyframe_index(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for faiss_id_key, raw_info in metadata.items():
        if not isinstance(raw_info, dict):
            continue
        info = dict(raw_info)
        try:
            faiss_id = int(info.get("faiss_id", faiss_id_key))
            timestamp = float(info.get("timestamp", 0.0))
        except (TypeError, ValueError):
            continue

        video_key = metadata_video_key(info)
        if not video_key:
            continue

        grouped[video_key].append(
            {
                "faiss_id": faiss_id,
                "video_id": video_key,
                "frame_name": info.get("frame_name", ""),
                "frame_index": info.get("frame_index"),
                "global_frame_id": info.get("global_frame_id"),
                "timestamp": timestamp,
                "fps": info.get("fps"),
                "split": info.get("split"),
                "raw_video_id": info.get("video_id"),
            }
        )

    index: dict[str, dict[str, Any]] = {}
    for video_key, frames in grouped.items():
        frames.sort(key=lambda item: item["timestamp"])
        index[video_key] = {
            "frames": frames,
            "timestamps": [item["timestamp"] for item in frames],
        }
    return index


def segment_mid_time(segment: dict[str, Any]) -> float | None:
    try:
        start = segment.get("start_time", segment.get("start", 0.0))
        end = segment.get("end_time", segment.get("end", None))
        start_f = float(start or 0.0)
        end_f = float(end) if end is not None else start_f
        return (start_f + end_f) / 2.0
    except (TypeError, ValueError):
        return None


def nearest_frame(video_entry: dict[str, Any], target_time: float) -> dict[str, Any] | None:
    frames = video_entry["frames"]
    timestamps = video_entry["timestamps"]
    if not frames:
        return None
    pos = bisect.bisect_left(timestamps, target_time)
    candidates = []
    if pos < len(frames):
        candidates.append(frames[pos])
    if pos > 0:
        candidates.append(frames[pos - 1])
    return min(candidates, key=lambda item: abs(float(item["timestamp"]) - target_time)) if candidates else None


def realign(asr_docs: list[dict[str, Any]], keyframe_index: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    stats = Counter()
    output = []

    for raw_doc in asr_docs:
        if not isinstance(raw_doc, dict):
            stats["skipped_non_dict"] += 1
            continue

        doc = dict(raw_doc)
        video_id = str(doc.get("video_id") or "").strip()
        mid_time = segment_mid_time(doc)
        if not video_id or mid_time is None:
            stats["missing_video_or_time"] += 1
            output.append(doc)
            continue

        video_entry = keyframe_index.get(video_id)
        if video_entry is None:
            stats["missing_video_in_metadata"] += 1
            output.append(doc)
            continue

        frame = nearest_frame(video_entry, mid_time)
        if frame is None:
            stats["no_keyframes_for_video"] += 1
            output.append(doc)
            continue

        old_faiss = doc.get("nearest_faiss_id")
        new_faiss = frame["faiss_id"]
        if old_faiss == new_faiss:
            stats["unchanged"] += 1
        else:
            stats["changed"] += 1

        doc["nearest_faiss_id"] = new_faiss
        doc["nearest_frame_name"] = frame.get("frame_name", "")
        doc["nearest_global_frame_id"] = frame.get("global_frame_id")
        doc["nearest_timestamp"] = frame.get("timestamp")
        doc["alignment_delta_seconds"] = round(float(frame["timestamp"]) - float(mid_time), 6)
        doc["alignment_source"] = "metadata_nearest_timestamp"
        if frame.get("split"):
            doc["split"] = frame.get("split")
        if frame.get("fps") is not None:
            doc["fps"] = frame.get("fps")

        output.append(doc)
        stats["aligned"] += 1

    return output, stats


def parse_args() -> argparse.Namespace:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Realign ASR segments to nearest keyframes using metadata timestamps.")
    parser.add_argument("--asr-in", type=Path, default=root / "src" / "dict" / "asr_results.json")
    parser.add_argument("--metadata", type=Path, default=root / "src" / "dict" / "metadata_clip.json")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON. Defaults to overwriting --asr-in.")
    parser.add_argument("--backup", action="store_true", help="Backup the overwritten ASR file first.")
    parser.add_argument("--dry-run", action="store_true", help="Only report alignment stats; do not write output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asr_path = args.asr_in.resolve()
    metadata_path = args.metadata.resolve()
    output_path = (args.output or args.asr_in).resolve()

    if not asr_path.exists():
        print(f"ASR file not found: {asr_path}", file=sys.stderr)
        return 1
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}", file=sys.stderr)
        return 1

    asr_docs = load_json(asr_path)
    if not isinstance(asr_docs, list):
        print(f"ASR file must be a list of documents: {asr_path}", file=sys.stderr)
        return 1

    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict):
        print(f"Metadata file must be an object keyed by faiss id: {metadata_path}", file=sys.stderr)
        return 1

    keyframe_index = build_keyframe_index(metadata)
    aligned_docs, stats = realign(asr_docs, keyframe_index)

    print("ASR docs:", len(asr_docs))
    print("Metadata keyframes:", len(metadata))
    print("Metadata videos:", len(keyframe_index))
    print("Stats:", dict(stats))
    unique_nearest = len({doc.get("nearest_faiss_id") for doc in aligned_docs if isinstance(doc, dict)})
    print("Unique nearest_faiss_id after alignment:", unique_nearest)
    print("Sample aligned docs:")
    for doc in aligned_docs[:3]:
        print(json.dumps({
            "video_id": doc.get("video_id"),
            "start_time": doc.get("start_time"),
            "end_time": doc.get("end_time"),
            "nearest_faiss_id": doc.get("nearest_faiss_id"),
            "nearest_frame_name": doc.get("nearest_frame_name"),
            "nearest_timestamp": doc.get("nearest_timestamp"),
            "alignment_delta_seconds": doc.get("alignment_delta_seconds"),
        }, ensure_ascii=False))

    if args.dry_run:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path == asr_path and args.backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = asr_path.with_suffix(asr_path.suffix + f".bak_{stamp}")
        shutil.copy2(asr_path, backup_path)
        print("Backup written:", backup_path)

    write_json(output_path, aligned_docs)
    print("Aligned ASR written:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
