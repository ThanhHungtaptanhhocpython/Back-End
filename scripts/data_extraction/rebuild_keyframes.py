#!/usr/bin/env python
"""Rebuild backend keyframe images from source videos.

Output layout matches src/services/user_service.py:
  src/data/Keyframes/{split}/{video_id}/{frame_name}

Frames are saved at the source frame resolution. Any resizing is left to model
preprocessing in the embedding pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".wmv"}


@dataclass(frozen=True)
class MetadataItem:
    faiss_id: int
    split: str
    video_id: str
    frame_name: str
    frame_index: int
    global_frame_id: int


def backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_metadata(metadata_path: Path) -> list[MetadataItem]:
    with metadata_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected metadata object at {metadata_path}, got {type(raw).__name__}")

    items: list[MetadataItem] = []
    for key, value in raw.items():
        try:
            faiss_id = int(key)
            items.append(
                MetadataItem(
                    faiss_id=faiss_id,
                    split=str(value["split"]),
                    video_id=str(value["video_id"]),
                    frame_name=str(value["frame_name"]),
                    frame_index=int(value["frame_index"]),
                    global_frame_id=int(value["global_frame_id"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Metadata item {key} is missing required field {exc}") from exc
    return sorted(items, key=lambda item: item.faiss_id)


def parse_video_roots(values: list[str] | None) -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("AIC_VIDEO_ROOT")
    if env_root:
        roots.append(Path(env_root))
    for value in values or []:
        roots.append(Path(value))

    if roots:
        return [root.resolve() for root in roots]

    root = backend_root()
    return [
        root / "src" / "data" / "Videos",
        root / "src" / "data" / "videos",
        root / "data" / "Videos",
        root / "data" / "videos",
        root / "Videos",
        root / "videos",
    ]


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_video_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("--video-map must point to a JSON object")
    return {normalize_key(str(key)): Path(value).resolve() for key, value in raw.items()}


def metadata_video_keys(item: MetadataItem) -> set[str]:
    split_last = item.split.split("-")[1].upper() if "-" in item.split else item.split
    stem = Path(item.frame_name).stem
    parts = stem.replace("keyframe_", "").split("_")
    keys = {
        item.video_id,
        f"{item.split}/{item.video_id}",
        f"{item.split}\\{item.video_id}",
        f"{split_last}_{item.video_id}",
        stem,
    }
    if len(parts) >= 2:
        keys.add("_".join(parts[:2]))
    return {normalize_key(key) for key in keys}


def scan_video_files(roots: Iterable[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in VIDEO_EXTENSIONS:
            candidates = [root]
        else:
            candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
        for path in candidates:
            stem = normalize_key(path.stem)
            index.setdefault(stem, path.resolve())
            parent_stem = normalize_key(f"{path.parent.name}/{path.stem}")
            index.setdefault(parent_stem, path.resolve())
            grandparent_stem = normalize_key(f"{path.parent.parent.name}/{path.parent.name}/{path.stem}")
            index.setdefault(grandparent_stem, path.resolve())
    return index


def resolve_video(item: MetadataItem, explicit_map: dict[str, Path], scanned: dict[str, Path]) -> Path | None:
    keys = metadata_video_keys(item)
    for key in keys:
        mapped = explicit_map.get(key)
        if mapped:
            return mapped
    for key in keys:
        mapped = scanned.get(key)
        if mapped:
            return mapped
    return None


def output_path(output_root: Path, item: MetadataItem) -> Path:
    return output_root / item.split / item.video_id / item.frame_name


def save_frame(frame_bgr, destination: Path) -> tuple[int, int]:
    import cv2

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    suffix = destination.suffix.lower()

    if suffix == ".webp":
        image.save(destination, "WEBP", lossless=True, quality=100, method=6)
    elif suffix in {".jpg", ".jpeg"}:
        image.save(destination, "JPEG", quality=100, subsampling=0)
    elif suffix == ".png":
        image.save(destination, "PNG")
    else:
        image.save(destination)
    return image.size


def extract_video_keyframes(
    video_path: Path,
    items: list[MetadataItem],
    output_root: Path,
    frame_number_base: int,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[int, int, list[str], list[tuple[str, int, int]]]:
    import cv2

    created = 0
    skipped = 0
    failures: list[str] = []
    size_samples: list[tuple[str, int, int]] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, [f"Cannot open video: {video_path}"], size_samples

    try:
        for item in sorted(items, key=lambda meta: meta.global_frame_id):
            dest = output_path(output_root, item)
            if skip_existing and dest.exists():
                skipped += 1
                continue

            frame_number = item.global_frame_id - frame_number_base
            if frame_number < 0:
                failures.append(f"{item.faiss_id}: negative frame after base adjustment ({frame_number})")
                continue

            if dry_run:
                created += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok or frame is None:
                failures.append(f"{item.faiss_id}: failed to read frame {frame_number} from {video_path}")
                continue

            width, height = save_frame(frame, dest)
            if len(size_samples) < 5:
                size_samples.append((str(dest), width, height))
            created += 1
    finally:
        cap.release()

    return created, skipped, failures, size_samples


def build_parser() -> argparse.ArgumentParser:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Rebuild keyframe images for the current backend metadata.")
    parser.add_argument("--metadata", type=Path, default=root / "src" / "dict" / "metadata_clip.json")
    parser.add_argument("--output-root", type=Path, default=root / "src" / "data" / "Keyframes")
    parser.add_argument(
        "--video-root",
        action="append",
        help="Directory or video file to scan. Can be repeated. Also supports AIC_VIDEO_ROOT.",
    )
    parser.add_argument(
        "--video-map",
        type=Path,
        help='Optional JSON map. Keys can be "split/video_id", "video_id", or "L21_V001"; values are video paths.',
    )
    parser.add_argument("--frame-number-base", type=int, default=0, choices=(0, 1))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, help="Process only the first N metadata rows, for smoke tests.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    try:
        import cv2  # noqa: F401
    except ImportError:
        logging.error("opencv-python is required for video extraction. Install it with: pip install opencv-python")
        return 2

    items = load_metadata(args.metadata)
    if args.limit:
        items = items[: args.limit]
    logging.info("Loaded %d metadata items from %s", len(items), args.metadata)

    roots = parse_video_roots(args.video_root)
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        logging.error("No video roots exist. Checked: %s", ", ".join(str(root) for root in roots))
        return 2

    explicit_map = load_video_map(args.video_map)
    scanned = scan_video_files(existing_roots)
    if not explicit_map and not scanned:
        logging.error("No source video files found under: %s", ", ".join(str(root) for root in existing_roots))
        return 2

    grouped: dict[Path, list[MetadataItem]] = defaultdict(list)
    missing: list[MetadataItem] = []
    for item in items:
        video_path = resolve_video(item, explicit_map, scanned)
        if video_path is None:
            missing.append(item)
        else:
            grouped[video_path].append(item)

    if missing:
        examples = ", ".join(f"{item.split}/{item.video_id}" for item in missing[:10])
        logging.warning("Missing source video mapping for %d metadata items. Examples: %s", len(missing), examples)

    if not grouped:
        logging.error("No metadata items could be mapped to a source video.")
        return 2

    total_created = 0
    total_skipped = 0
    all_failures: list[str] = []
    size_samples: list[tuple[str, int, int]] = []
    for video_path, video_items in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        logging.info("Extracting %d frames from %s", len(video_items), video_path)
        created, skipped, failures, samples = extract_video_keyframes(
            video_path=video_path,
            items=video_items,
            output_root=args.output_root,
            frame_number_base=args.frame_number_base,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        total_created += created
        total_skipped += skipped
        all_failures.extend(failures)
        size_samples.extend(samples)

    logging.info("Keyframes created=%d skipped=%d failed=%d missing_video=%d", total_created, total_skipped, len(all_failures), len(missing))
    for path, width, height in size_samples[:5]:
        logging.info("Sample output resolution: %s -> %dx%d", path, width, height)
    for failure in all_failures[:20]:
        logging.warning("%s", failure)

    return 1 if all_failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
