#!/usr/bin/env python
"""Build OpenCLIP per-video features and the backend FAISS index.

This matches src/utils/faiss_processing.py:
  model_name = "ViT-H-14-quickgelu"
  pretrained = "dfn5b"

FAISS ids are the numeric keys in metadata_clip.json. Per-video .npy files are
stored so row `frame_index` is the feature used by image_search(faiss_id).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_NAME = "ViT-H-14-quickgelu"
PRETRAINED = "dfn5b"


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
            items.append(
                MetadataItem(
                    faiss_id=int(key),
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


def keyframe_path(keyframes_root: Path, item: MetadataItem) -> Path:
    return keyframes_root / item.split / item.video_id / item.frame_name


def group_by_video(items: list[MetadataItem]) -> dict[tuple[str, str], list[MetadataItem]]:
    grouped: dict[tuple[str, str], list[MetadataItem]] = defaultdict(list)
    for item in items:
        grouped[(item.split, item.video_id)].append(item)
    return {
        key: sorted(value, key=lambda item: (item.frame_index, item.faiss_id))
        for key, value in grouped.items()
    }


def load_model(device: str):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        device="cpu",
        pretrained=PRETRAINED,
    )
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
    elif device == "cuda":
        logging.warning("CUDA requested but unavailable; using CPU")
        device = "cpu"
    return model, preprocess, device


def encode_batch(model, preprocess, image_paths: list[Path], device: str) -> np.ndarray:
    import torch

    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(preprocess(image.convert("RGB")).unsqueeze(0))
    batch = torch.cat(images).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def make_video_feature_matrix(video_items: list[MetadataItem], encoded: dict[int, np.ndarray], feature_dim: int) -> np.ndarray:
    max_index = max(item.frame_index for item in video_items)
    matrix = np.zeros((max_index + 1, feature_dim), dtype=np.float32)
    seen: set[int] = set()
    for item in video_items:
        if item.frame_index in seen:
            raise ValueError(f"Duplicate frame_index {item.frame_index} for {item.split}/{item.video_id}")
        seen.add(item.frame_index)
        matrix[item.frame_index] = encoded[item.faiss_id]
    return matrix


def build_parser() -> argparse.ArgumentParser:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Build OpenCLIP features and FAISS index for backend search.")
    parser.add_argument("--metadata", type=Path, default=root / "src" / "dict" / "metadata_clip.json")
    parser.add_argument("--keyframes-root", type=Path, default=root / "src" / "data" / "Keyframes")
    parser.add_argument("--features-root", type=Path, default=root / "src" / "data" / "features")
    parser.add_argument("--index-path", type=Path, default=root / "src" / "dict" / "nw" / "faiss_index_clip.bin")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, help="Encode only the first N metadata rows, for smoke tests.")
    parser.add_argument("--strict", action="store_true", help="Fail if any keyframe image is missing.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    try:
        import faiss
        import torch
    except ImportError as exc:
        logging.error("Missing dependency: %s", exc)
        logging.error("Install backend dependencies, including faiss-cpu, torch, open-clip-torch, and Pillow.")
        return 2

    items = load_metadata(args.metadata)
    if args.limit:
        items = items[: args.limit]
    logging.info("Loaded %d metadata items from %s", len(items), args.metadata)

    missing = [item for item in items if not keyframe_path(args.keyframes_root, item).exists()]
    if missing:
        examples = ", ".join(str(keyframe_path(args.keyframes_root, item)) for item in missing[:10])
        logging.warning("Missing %d keyframe images. Examples: %s", len(missing), examples)
        if args.strict:
            return 2
        items = [item for item in items if keyframe_path(args.keyframes_root, item).exists()]

    if not items:
        logging.error("No keyframe images available to encode.")
        return 2

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model, preprocess, device = load_model(device)
    feature_dim = int(model.visual.output_dim)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(feature_dim))

    encoded: dict[int, np.ndarray] = {}
    index_ids: list[int] = []
    index_vectors: list[np.ndarray] = []

    for start in range(0, len(items), args.batch_size):
        batch_items = items[start : start + args.batch_size]
        paths = [keyframe_path(args.keyframes_root, item) for item in batch_items]
        feats = encode_batch(model, preprocess, paths, device)
        for item, feat in zip(batch_items, feats):
            encoded[item.faiss_id] = feat
            index_ids.append(item.faiss_id)
            index_vectors.append(feat)
        logging.info("Encoded %d/%d keyframes", min(start + args.batch_size, len(items)), len(items))

    vectors = np.vstack(index_vectors).astype(np.float32)
    ids = np.asarray(index_ids, dtype=np.int64)
    index.add_with_ids(vectors, ids)

    grouped = group_by_video(items)
    args.features_root.mkdir(parents=True, exist_ok=True)
    written_features = 0
    for (split, video_id), video_items in grouped.items():
        video_feature_path = args.features_root / split / f"{video_id}.npy"
        video_feature_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = make_video_feature_matrix(video_items, encoded, feature_dim)
        np.save(video_feature_path, matrix)
        written_features += 1

    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_path))
    logging.info("Wrote FAISS index: %s (ntotal=%d)", args.index_path, index.ntotal)
    logging.info("Wrote %d per-video feature files under %s", written_features, args.features_root)
    if missing:
        logging.warning("Skipped %d missing keyframes; rebuild them before production indexing.", len(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
