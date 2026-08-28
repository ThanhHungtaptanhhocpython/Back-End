#!/usr/bin/env python
"""Build metadata_clip.json from an extracted keyframe directory.

Expected keyframe layout:
    KEYFRAMES_ROOT/{split}/{video_id}/{frame_name}

The script uses map-keyframes CSV files when available to recover source video
timestamps and source frame indexes. It is intended for the step after SBD
keyframe extraction and before embedding/index rebuild.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".webp", ".jpg", ".jpeg", ".png"}
logger = logging.getLogger(__name__)


def backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_frame_number(frame_name: str) -> int | None:
    stem = Path(frame_name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def load_map_keyframes(map_dir: Path | None) -> dict[str, dict[str, Any]]:
    if map_dir is None or not map_dir.exists():
        logger.warning("map-keyframes directory not found; timestamp fallback will use filenames: %s", map_dir)
        return {}

    maps: dict[str, dict[str, Any]] = {}
    csv_paths = sorted(map_dir.glob("*.csv"))
    logger.info("Loading %d map-keyframes CSV files from %s", len(csv_paths), map_dir)
    for index, csv_path in enumerate(csv_paths, 1):
        by_n: dict[int, dict[str, Any]] = {}
        by_frame_idx: dict[int, dict[str, Any]] = {}
        video_fps: float | None = None
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    n = int(str(row.get("n") or "").strip())
                    pts_time = float(str(row.get("pts_time") or "0").strip())
                    fps = float(str(row.get("fps") or "25").strip())
                    frame_idx = int(str(row.get("frame_idx") or "0").strip())
                except (TypeError, ValueError):
                    continue
                item = {"timestamp": pts_time, "fps": fps, "global_frame_id": frame_idx, "n": n}
                by_n[n] = item
                by_frame_idx[frame_idx] = item
                if video_fps is None and fps > 0:
                    video_fps = fps
        if by_n or by_frame_idx:
            maps[csv_path.stem] = {"by_n": by_n, "by_frame_idx": by_frame_idx, "fps": video_fps}
        if index == 1 or index % 100 == 0 or index == len(csv_paths):
            logger.info("Loaded map CSV %d/%d: %s", index, len(csv_paths), csv_path.name)
    return maps


def infer_time_info(
    video_id: str,
    frame_name: str,
    frame_index: int,
    maps: dict[str, dict[str, Any]],
    default_fps: float,
    allow_ordinal_fallback: bool = True,
) -> tuple[int, float, float, str]:
    video_map = maps.get(video_id)
    parsed_frame = parse_frame_number(frame_name)

    if video_map:
        if parsed_frame is not None and parsed_frame in video_map["by_frame_idx"]:
            info = video_map["by_frame_idx"][parsed_frame]
            return (
                int(info["global_frame_id"]),
                float(info["timestamp"]),
                float(info["fps"]),
                "map_frame_idx_exact",
            )

        if allow_ordinal_fallback:
            ordinal = frame_index + 1
            if ordinal in video_map["by_n"]:
                info = video_map["by_n"][ordinal]
                return (
                    int(info["global_frame_id"]),
                    float(info["timestamp"]),
                    float(info["fps"]),
                    "map_keyframe_number_exact",
                )

        video_fps = float(video_map.get("fps") or default_fps)
        if parsed_frame is not None and video_fps > 0:
            return (
                int(parsed_frame),
                float(parsed_frame) / video_fps,
                video_fps,
                "filename_frame_number_video_fps",
            )

    global_frame_id = parsed_frame if parsed_frame is not None else frame_index
    timestamp = float(global_frame_id) / float(default_fps) if default_fps else 0.0
    return int(global_frame_id), timestamp, float(default_fps), "filename_frame_number_fallback"


def scan_keyframes(keyframes_root: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    split_dirs = sorted(path for path in keyframes_root.iterdir() if path.is_dir())
    logger.info("Scanning keyframes root %s (%d split folders)", keyframes_root, len(split_dirs))
    for split_index, split_dir in enumerate(split_dirs, 1):
        split = split_dir.name
        video_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        logger.info("Scanning split %d/%d: %s (%d videos)", split_index, len(split_dirs), split, len(video_dirs))
        split_count_before = len(rows)
        for video_index, video_dir in enumerate(video_dirs, 1):
            video_id = video_dir.name
            video_count = 0
            for image_path in sorted(
                path for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                rows.append((split, video_id, image_path))
                video_count += 1
            if video_index == 1 or video_index % 25 == 0 or video_index == len(video_dirs):
                logger.info(
                    "Scanned %s video %d/%d: %s (%d keyframes, total=%d)",
                    split,
                    video_index,
                    len(video_dirs),
                    video_id,
                    video_count,
                    len(rows),
                )
        logger.info("Finished split %s: %d keyframes", split, len(rows) - split_count_before)
    logger.info("Finished scanning: %d keyframe images", len(rows))
    return rows


def build_metadata(
    keyframes_root: Path,
    map_dir: Path | None,
    default_fps: float,
    limit: int | None = None,
) -> tuple[dict[str, dict[str, Any]], Counter]:
    maps = load_map_keyframes(map_dir)
    rows = scan_keyframes(keyframes_root)
    if limit is not None:
        logger.info("Applying --limit=%d to %d scanned keyframes", limit, len(rows))
        rows = rows[:limit]

    per_video_index: dict[str, int] = defaultdict(int)
    metadata: dict[str, dict[str, Any]] = {}
    stats = Counter()

    total = len(rows)
    for faiss_id, (split, video_id, image_path) in enumerate(rows):
        frame_index = per_video_index[video_id]
        per_video_index[video_id] += 1

        global_frame_id, timestamp, fps, timestamp_source = infer_time_info(
            video_id=video_id,
            frame_name=image_path.name,
            frame_index=frame_index,
            maps=maps,
            default_fps=default_fps,
        )
        stats[timestamp_source] += 1

        frame_path = image_path.relative_to(keyframes_root).as_posix()
        metadata[str(faiss_id)] = {
            "faiss_id": faiss_id,
            "split": split,
            "video_id": video_id,
            "frame_name": image_path.name,
            "frame_path": frame_path,
            "frame_index": frame_index,
            "frame_id": f"{global_frame_id:06d}",
            "global_frame_id": global_frame_id,
            "timestamp": timestamp,
            "timestamp_s": timestamp,
            "timestamp_source": timestamp_source,
            "fps": fps,
        }
        if faiss_id == 0 or (faiss_id + 1) % 10000 == 0 or (faiss_id + 1) == total:
            logger.info("Built metadata %d/%d", faiss_id + 1, total)

    stats["videos"] = len(per_video_index)
    stats["keyframes"] = len(metadata)
    stats["map_files"] = len(maps)
    return metadata, stats


def build_metadata_from_global_ids(
    global_ids_path: Path,
    keyframes_root: Path,
    map_dir: Path | None,
    default_fps: float,
    limit: int | None = None,
) -> tuple[dict[str, dict[str, Any]], Counter]:
    """Build metadata using BEiT3 vector IDs as the canonical row mapping."""
    import pandas as pd

    maps = load_map_keyframes(map_dir)
    table = pd.read_parquet(global_ids_path)
    required = {"vector_id", "video_id", "frame_path"}
    missing_columns = sorted(required - set(table.columns))
    if missing_columns:
        raise ValueError(f"global_ids parquet is missing columns: {missing_columns}")

    table = table.sort_values("vector_id", kind="stable")
    if limit is not None:
        logger.info("Applying --limit=%d to %d global-id rows", limit, len(table))
        table = table.head(limit)

    vector_ids = [int(value) for value in table["vector_id"].tolist()]
    if len(vector_ids) != len(set(vector_ids)):
        raise ValueError("global_ids parquet contains duplicate vector_id values")
    if limit is None and vector_ids != list(range(len(vector_ids))):
        raise ValueError(
            "global_ids vector_id values are not contiguous from 0; refusing to build metadata that may not match FAISS"
        )

    per_video_index: dict[str, int] = defaultdict(int)
    metadata: dict[str, dict[str, Any]] = {}
    stats = Counter()
    total = len(table)

    for position, row in enumerate(table.to_dict("records"), 1):
        vector_id = int(row["vector_id"])
        video_id = str(row["video_id"])
        frame_path = str(row["frame_path"]).replace("\\", "/").strip("/")
        frame_name = Path(frame_path).name
        split = str(row.get("parent_namespace") or frame_path.split("/", 1)[0])
        frame_index = per_video_index[video_id]
        per_video_index[video_id] += 1

        global_frame_id, timestamp, fps, timestamp_source = infer_time_info(
            video_id=video_id,
            frame_name=frame_name,
            frame_index=frame_index,
            maps=maps,
            default_fps=default_fps,
            allow_ordinal_fallback=False,
        )
        stats[timestamp_source] += 1
        if not (keyframes_root / frame_path).is_file():
            stats["missing_images"] += 1

        source_frame_id = row.get("frame_id")
        if source_frame_id is None or (isinstance(source_frame_id, float) and pd.isna(source_frame_id)):
            source_frame_id = f"{global_frame_id:06d}"
        else:
            source_frame_id = str(source_frame_id)

        metadata[str(vector_id)] = {
            "faiss_id": vector_id,
            "vector_id": vector_id,
            "split": split,
            "video_id": video_id,
            "frame_name": frame_name,
            "frame_path": frame_path,
            "frame_index": frame_index,
            "frame_id": source_frame_id,
            "global_frame_id": global_frame_id,
            "timestamp": timestamp,
            "timestamp_s": timestamp,
            "timestamp_source": timestamp_source,
            "fps": fps,
        }
        if position == 1 or position % 10000 == 0 or position == total:
            logger.info("Built BEiT3-aligned metadata %d/%d", position, total)

    stats["videos"] = len(per_video_index)
    stats["keyframes"] = len(metadata)
    stats["map_files"] = len(maps)
    return metadata, stats


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Build metadata_clip.json from extracted keyframes.")
    parser.add_argument("--keyframes-root", type=Path, required=True)
    parser.add_argument("--map-keyframes-dir", type=Path, default=root / "src" / "dict" / "map-keyframes")
    parser.add_argument(
        "--global-ids",
        type=Path,
        default=None,
        help="Build in exact BEiT3 vector_id order from global_ids.parquet instead of scanning all images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Defaults to metadata_beit3.json with --global-ids, otherwise metadata_clip.json.",
    )
    parser.add_argument("--default-fps", type=float, default=25.0)
    parser.add_argument("--backup", action="store_true", help="Backup --output before overwriting it.")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing output.")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N keyframes for testing.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    keyframes_root = args.keyframes_root.resolve()
    map_dir = args.map_keyframes_dir.resolve() if args.map_keyframes_dir else None
    default_output_name = "metadata_beit3.json" if args.global_ids else "metadata_clip.json"
    output = (args.output or (backend_root() / "src" / "dict" / default_output_name)).resolve()

    if not keyframes_root.exists():
        print(f"Keyframes root not found: {keyframes_root}")
        return 1

    if args.global_ids:
        global_ids_path = args.global_ids.resolve()
        if not global_ids_path.is_file():
            print(f"global_ids parquet not found: {global_ids_path}")
            return 1
        metadata, stats = build_metadata_from_global_ids(
            global_ids_path=global_ids_path,
            keyframes_root=keyframes_root,
            map_dir=map_dir,
            default_fps=args.default_fps,
            limit=args.limit,
        )
    else:
        global_ids_path = None
        metadata, stats = build_metadata(
            keyframes_root=keyframes_root,
            map_dir=map_dir,
            default_fps=args.default_fps,
            limit=args.limit,
        )

    print("Keyframes root:", keyframes_root)
    print("Map-keyframes dir:", map_dir)
    print("BEiT3 global IDs:", global_ids_path)
    print("Output:", output)
    print("Stats:", dict(stats))
    print("Sample:")
    for key in list(metadata)[:3]:
        print(key, json.dumps(metadata[key], ensure_ascii=False))

    if args.dry_run:
        return 0

    if stats.get("missing_images", 0):
        print(f"Refusing to write metadata: {stats['missing_images']} mapped keyframe images are missing.")
        return 2

    if output.exists() and args.backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output.with_suffix(output.suffix + f".bak_{stamp}")
        shutil.copy2(output, backup_path)
        print("Backup written:", backup_path)

    write_json(output, metadata)
    print("Metadata written:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
