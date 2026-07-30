#!/usr/bin/env python
"""Verify that backend search assets match the current metadata."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_metadata(metadata_path: Path) -> dict[int, dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected metadata object at {metadata_path}, got {type(raw).__name__}")
    return {int(key): value for key, value in raw.items()}


def keyframe_path(keyframes_root: Path, info: dict) -> Path:
    return keyframes_root / str(info["split"]) / str(info["video_id"]) / str(info["frame_name"])


def feature_path(features_root: Path, info: dict) -> Path:
    return features_root / str(info["split"]) / f"{info['video_id']}.npy"


def sample_items(metadata: dict[int, dict], sample_size: int) -> list[tuple[int, dict]]:
    items = sorted(metadata.items())
    if len(items) <= sample_size:
        return items
    rng = random.Random(0)
    return rng.sample(items, sample_size)


def verify_index(index_path: Path, expected_count: int) -> tuple[bool, int | None]:
    if not index_path.exists():
        logging.error("Missing FAISS index: %s", index_path)
        return False, None
    try:
        import faiss
    except ImportError:
        logging.error("Cannot import faiss; install faiss-cpu to inspect %s", index_path)
        return False, None

    index = faiss.read_index(str(index_path))
    ok = True
    if index.ntotal != expected_count:
        logging.warning("FAISS ntotal=%d, metadata items=%d", index.ntotal, expected_count)
        ok = False
    else:
        logging.info("FAISS ntotal matches metadata count: %d", index.ntotal)
    return ok, int(index.ntotal)


def verify_keyframes(samples: list[tuple[int, dict]], keyframes_root: Path) -> tuple[bool, list[tuple[str, int, int]]]:
    ok = True
    sizes: list[tuple[str, int, int]] = []
    for faiss_id, info in samples:
        path = keyframe_path(keyframes_root, info)
        if not path.exists():
            logging.error("Missing sample keyframe for id=%s: %s", faiss_id, path)
            ok = False
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
            sizes.append((str(path), width, height))
        except Exception as exc:
            logging.error("Cannot read sample keyframe for id=%s: %s (%s)", faiss_id, path, exc)
            ok = False
    return ok, sizes


def verify_features(metadata: dict[int, dict], samples: list[tuple[int, dict]], features_root: Path) -> tuple[bool, int]:
    ok = True
    video_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for info in metadata.values():
        video_groups[(str(info["split"]), str(info["video_id"]))].append(info)

    checked_files: set[Path] = set()
    for faiss_id, info in samples:
        path = feature_path(features_root, info)
        if path in checked_files:
            continue
        checked_files.add(path)
        if not path.exists():
            logging.error("Missing per-video feature file for sample id=%s: %s", faiss_id, path)
            ok = False
            continue
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception as exc:
            logging.error("Cannot load feature file %s: %s", path, exc)
            ok = False
            continue
        if arr.ndim != 2:
            logging.error("Feature file %s should be 2D, got shape %s", path, arr.shape)
            ok = False
            continue
        max_frame_index = max(int(item["frame_index"]) for item in video_groups[(str(info["split"]), str(info["video_id"]))])
        if arr.shape[0] <= max_frame_index:
            logging.error("Feature file %s has %d rows, needs frame_index %d", path, arr.shape[0], max_frame_index)
            ok = False
    return ok, len(checked_files)


def build_parser() -> argparse.ArgumentParser:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Verify backend FAISS, metadata, keyframes, and per-video features.")
    parser.add_argument("--metadata", type=Path, default=root / "src" / "dict" / "metadata_clip.json")
    parser.add_argument("--keyframes-root", type=Path, default=root / "src" / "data" / "Keyframes")
    parser.add_argument("--features-root", type=Path, default=root / "src" / "data" / "features")
    parser.add_argument("--index-path", type=Path, default=root / "src" / "dict" / "nw" / "faiss_index_clip.bin")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    metadata = load_metadata(args.metadata)
    logging.info("Loaded metadata items: %d", len(metadata))
    samples = sample_items(metadata, args.sample_size)

    index_ok, ntotal = verify_index(args.index_path, len(metadata))
    keyframes_ok, sizes = verify_keyframes(samples, args.keyframes_root)
    features_ok, checked_feature_files = verify_features(metadata, samples, args.features_root)

    for path, width, height in sizes[:5]:
        logging.info("Sample keyframe resolution: %s -> %dx%d", path, width, height)
    logging.info("Checked %d sample metadata rows and %d feature files", len(samples), checked_feature_files)
    if ntotal is not None:
        logging.info("FAISS index path: %s (ntotal=%d)", args.index_path, ntotal)

    return 0 if index_ok and keyframes_ok and features_ok else 1


if __name__ == "__main__":
    sys.exit(main())
