#!/usr/bin/env python
"""Preflight check for backend search assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_env(path: Path) -> dict[str, str]:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def path_from(value: str | None, fallback: Path) -> Path:
    return Path(value) if value else fallback


def image_count(keyframes_root: Path) -> tuple[int, int, int]:
    exts = {".webp", ".jpg", ".jpeg", ".png"}
    splits = videos = images = 0
    if not keyframes_root.exists():
        return splits, videos, images
    for split in keyframes_root.iterdir():
        if not split.is_dir():
            continue
        splits += 1
        for video in split.iterdir():
            if not video.is_dir():
                continue
            videos += 1
            images += sum(1 for item in video.iterdir() if item.is_file() and item.suffix.lower() in exts)
    return splits, videos, images


def main() -> int:
    base = root()
    env = read_env(base / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=path_from(env.get("METADATA_PATH"), base / "src" / "dict" / "metadata_clip.json"))
    parser.add_argument("--keyframes-root", type=Path, default=path_from(env.get("KEYFRAMES_ROOT"), base / "src" / "data" / "Keyframes"))
    parser.add_argument("--map-dir", type=Path, default=base / "src" / "dict" / "map-keyframes")
    parser.add_argument("--beit3-index", type=Path, default=path_from(env.get("BEIT3_FAISS_INDEX_PATH"), base / "aic2026_05_beit3_faiss" / "beit3_faiss.index"))
    parser.add_argument("--beit3-global-ids", type=Path, default=path_from(env.get("BEIT3_GLOBAL_IDS_PATH"), base / "aic2026_05_beit3_faiss" / "global_ids.parquet"))
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    failures = 0

    print("KEYFRAMES_ROOT:", args.keyframes_root)
    splits, videos, images = image_count(args.keyframes_root)
    print(f"keyframes: splits={splits} videos={videos} images={images}")
    if not args.keyframes_root.exists() or images == 0:
        print("FAIL: keyframes root missing or empty")
        failures += 1

    print("metadata:", args.metadata)
    metadata = {}
    if args.metadata.exists():
        metadata = json.load(open(args.metadata, encoding="utf-8"))
        print("metadata rows:", len(metadata))
        for key in list(metadata)[: args.sample_size]:
            item = metadata[key]
            frame_path = item.get("frame_path")
            path = args.keyframes_root / frame_path if frame_path else args.keyframes_root / str(item.get("split", "")) / str(item.get("video_id", "")) / str(item.get("frame_name", ""))
            if not path.exists():
                print("FAIL: metadata sample missing image:", key, path)
                failures += 1
                break
    else:
        print("FAIL: metadata missing")
        failures += 1

    csv_count = len(list(args.map_dir.glob("*.csv"))) if args.map_dir.exists() else 0
    print("map-keyframes csv:", csv_count, args.map_dir)

    print("BEIT3 index:", args.beit3_index)
    print("BEIT3 global_ids:", args.beit3_global_ids)
    if args.beit3_global_ids.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(args.beit3_global_ids)
            print("BEIT3 global_ids rows:", len(df))
            if "frame_path" in df.columns:
                missing = 0
                for row in df.head(args.sample_size).to_dict("records"):
                    if not (args.keyframes_root / str(row["frame_path"])).exists():
                        missing += 1
                print(f"BEIT3 sample frame_path exists: {args.sample_size - missing}/{args.sample_size}")
                if missing:
                    failures += 1
        except Exception as exc:
            print("WARN: cannot inspect BEIT3 parquet:", exc)
    else:
        print("WARN: BEIT3 global_ids missing")

    if args.beit3_index.exists():
        try:
            import faiss

            index = faiss.read_index(str(args.beit3_index))
            print("BEIT3 FAISS:", "ntotal=", index.ntotal, "dim=", index.d)
        except Exception as exc:
            print("WARN: cannot inspect BEIT3 FAISS:", exc)
    else:
        print("WARN: BEIT3 index missing")

    print("SUMMARY:", "FAIL" if failures else "OK", "failures=", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
