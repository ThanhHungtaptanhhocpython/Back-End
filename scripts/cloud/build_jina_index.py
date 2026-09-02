#!/usr/bin/env python
"""Build the Jina CLIP v2 FAISS index + parquet runtime artifacts.

Reads the per-video artifacts the Azure Jina embedding pipeline produces
(scripts/notebooks/embed-jina-upload-azure-5jobs-disk-safe.ipynb):

    <embeddings_root>/<namespace>/<video_id>.npy      -- (N, 1024) float16/float32
    <records_root>/<namespace>/<video_id>.json        -- {"records": [...]} or a bare list

and an optional map-keyframes directory (``<video_id>.csv`` with columns
``n, pts_time, fps, frame_idx``, one-based ``n``) used only to fill in a
missing timestamp/source_frame_idx -- never to override a value the record
JSON already carries. ``n`` is matched to the 1-based *embedding row order*
(``.npy`` row order == metadata order), never to digits parsed out of a
``frame_id`` string.

``keyframe_ordinal`` is the 1-based position of a keyframe in its video's
extracted set: an explicit, validated ``record["keyframe_ordinal"]`` if
present, otherwise ``embedding_row + 1``. It is never derived from the
trailing digits of ``frame_id`` (which is often an original *video* frame
number, unrelated to extraction position).

The whole corpus's embeddings are never loaded into RAM at once: each video's
``.npy`` is opened with ``mmap_mode='r'`` and added to the FAISS index one
video at a time (see ``_iter_videos`` / ``build_index``). Only the small
scalar metadata rows accumulate across the run.

Validates (raises SystemExit with a precise message on failure):

* embedding shape is exactly ``(N, 1024)`` for every video;
* every embedding value is finite;
* the per-video ``.npy`` row count matches its record count (AGENTS.md rule:
  ".npy" row order must match the metadata's frame order exactly);
* every ``vector_id`` in the output is unique;
* every row has a non-empty ``video_id``/``split`` with no path-unsafe
  characters, and a valid (finite, non-negative) ``timestamp_ms``;
* the built FAISS index's ``ntotal`` matches the parquet row count.

Produces, under ``--out-dir``:

    jina_faiss.index
    jina_global_ids.parquet     (vector_id, split, video_id, embedding_row,
                                  keyframe_ordinal, timestamp_ms, asset_key,
                                  frame_path, source_frame_id)
    jina_video_metadata.parquet (video_id, parent_namespace, split,
                                  frame_count, embedding_dim, first_vector_id,
                                  last_vector_id)
    jina_index_meta.json        (model provenance, dimension, metric, counts)
    jina_build_report.json      (per-video counts, warnings, elapsed time)

Usage
-----
    python scripts/cloud/build_jina_index.py \\
        --embeddings-root /data/jina/embeddings \\
        --records-root /data/jina/records \\
        --map-keyframes-root /data/map-keyframes \\
        --model-id jinaai/jina-clip-v2 \\
        --model-revision <pinned-commit-sha> \\
        --out-dir ./jina_runtime

Then publish the manifest for it with scripts/cloud/build_asset_manifest.py
(artifact names ``jina_faiss_index`` / ``jina_global_ids`` /
``jina_video_metadata`` / ``jina_index_meta`` -- the full runtime profile
``BACKEND_ARTIFACT_NAMES["jina_clip_v2"]``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_DIM = 1024
_UNSAFE_ID_RE = re.compile(r"[\\/]|\.\.")
_NORM_ATOL = 2e-3  # matches the tolerance the Azure merge notebook validates with


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fail(message: str) -> "SystemExit":
    return SystemExit(f"build_jina_index: {message}")


@dataclass
class VideoJob:
    namespace: str
    video_id: str
    npy_path: Path
    records_path: Path


def _iter_videos(embeddings_root: Path, records_root: Path) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    if not embeddings_root.is_dir():
        raise _fail(f"--embeddings-root does not exist or is not a directory: {embeddings_root}")
    for npy_path in sorted(embeddings_root.rglob("*.npy")):
        namespace = npy_path.parent.name
        video_id = npy_path.stem
        records_path = records_root / namespace / f"{video_id}.json"
        if not records_path.is_file():
            raise _fail(
                f"no matching records JSON for {namespace}/{video_id}: expected {records_path}"
            )
        jobs.append(VideoJob(namespace=namespace, video_id=video_id, npy_path=npy_path, records_path=records_path))
    if not jobs:
        raise _fail(f"no per-video .npy files found under {embeddings_root}")
    # Deterministic global vector-id assignment across runs of the same corpus.
    jobs.sort(key=lambda j: (j.namespace, j.video_id))
    return jobs


def _load_records(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"could not read records JSON {path}: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise _fail(f"records JSON {path} must be a list or an object with a 'records' list")


class MapKeyframesIndex:
    """Lazy, per-video ``n -> {timestamp, fps, frame_idx}`` lookup.

    Used only to fill in a value a record is missing -- never to override
    what the record JSON already has.
    """

    def __init__(self, root: Path | None):
        self._root = root
        self._cache: dict[str, dict[int, dict[str, Any]]] = {}

    def for_video(self, video_id: str) -> dict[int, dict[str, Any]]:
        if self._root is None:
            return {}
        if video_id in self._cache:
            return self._cache[video_id]
        path = self._root / f"{video_id}.csv"
        mapping: dict[int, dict[str, Any]] = {}
        if path.is_file():
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        n = int(str(row.get("n") or "").strip())
                        mapping[n] = {
                            "timestamp": float(str(row.get("pts_time") or "").strip()),
                            "fps": float(str(row.get("fps") or "0").strip()) or None,
                            "frame_idx": int(str(row.get("frame_idx") or "").strip()),
                        }
                    except (TypeError, ValueError):
                        continue
        self._cache[video_id] = mapping
        return mapping


@dataclass
class BuildResult:
    df: pd.DataFrame
    index: Any
    videos: int
    warnings: list[str] = field(default_factory=list)


def _validate_video_or_split(value: str, label: str, context: str) -> None:
    if not value or not value.strip():
        raise _fail(f"{context}: {label} is empty")
    if _UNSAFE_ID_RE.search(value):
        raise _fail(f"{context}: {label}={value!r} contains path-unsafe characters")


def build_index(
    embeddings_root: Path,
    records_root: Path,
    map_keyframes_root: Path | None,
    *,
    storage_dtype_hint: str = "auto",
) -> BuildResult:
    import faiss

    jobs = _iter_videos(embeddings_root, records_root)
    mapkf = MapKeyframesIndex(map_keyframes_root)

    index: Any = None
    rows: list[dict] = []
    warnings: list[str] = []
    next_vector_id = 0
    seen_asset_keys: set[str] = set()

    for job in jobs:
        vectors = np.load(job.npy_path, mmap_mode="r")
        if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIM:
            raise _fail(
                f"{job.namespace}/{job.video_id}: embedding shape {vectors.shape} != (N, {EXPECTED_DIM})"
            )
        records = _load_records(job.records_path)
        if len(records) != len(vectors):
            raise _fail(
                f"{job.namespace}/{job.video_id}: {len(vectors)} embedding rows "
                f"!= {len(records)} metadata records"
            )

        # mmap-backed slice -> only this video's block ever materializes.
        block = np.asarray(vectors[:], dtype=np.float32)
        if not np.isfinite(block).all():
            raise _fail(f"{job.namespace}/{job.video_id}: embedding contains NaN/Inf values")
        norms = np.linalg.norm(block, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORM_ATOL):
            bad = int(np.sum(np.abs(norms - 1.0) > _NORM_ATOL))
            raise _fail(
                f"{job.namespace}/{job.video_id}: {bad}/{len(block)} embedding rows are not "
                f"L2-normalized (atol={_NORM_ATOL})"
            )

        if index is None:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(EXPECTED_DIM))
        elif index.d != block.shape[1]:
            raise _fail(f"{job.namespace}/{job.video_id}: dimension changed within the corpus")

        ids = np.arange(next_vector_id, next_vector_id + len(block), dtype=np.int64)
        index.add_with_ids(block, ids)

        namespace_map = mapkf.for_video(job.video_id)
        for embedding_row, (record, vector_id) in enumerate(zip(records, ids.tolist())):
            frame_id = str(record.get("frame_id") or "")
            local_position = record.get("local_position")
            if local_position is not None and int(local_position) != embedding_row:
                raise _fail(
                    f"{job.namespace}/{job.video_id}: record local_position={local_position} "
                    f"!= embedding row index {embedding_row} -- .npy row order must match "
                    f"metadata order exactly"
                )

            # `keyframe_ordinal` is the 1-based position of this keyframe in the
            # video's extracted keyframe set. Trailing digits of `frame_id`
            # (e.g. an original *video* frame number like "keyframe_015530")
            # are NOT that position, so never derive it from them. Use an
            # explicit, validated record value if given; otherwise the
            # embedding row (== extraction order, enforced above).
            explicit_ordinal = record.get("keyframe_ordinal")
            if explicit_ordinal is not None:
                try:
                    keyframe_ordinal = int(explicit_ordinal)
                except (TypeError, ValueError):
                    raise _fail(
                        f"{job.namespace}/{job.video_id}#{frame_id}: keyframe_ordinal="
                        f"{explicit_ordinal!r} is not an integer"
                    )
                if keyframe_ordinal < 1:
                    raise _fail(
                        f"{job.namespace}/{job.video_id}#{frame_id}: keyframe_ordinal="
                        f"{keyframe_ordinal} must be >= 1"
                    )
            else:
                keyframe_ordinal = embedding_row + 1
            # map-keyframes `n` is 1-based over the same extracted keyframe set,
            # i.e. the embedding row order -- key the lookup on that, not on a
            # frame_id-derived number.
            csv_row = namespace_map.get(embedding_row + 1)

            timestamp = record.get("timestamp")
            source_fps = record.get("source_fps")
            source_frame_idx = record.get("source_frame_idx")
            if timestamp is None and csv_row is not None:
                timestamp = csv_row.get("timestamp")
                warnings.append(
                    f"{job.namespace}/{job.video_id}#{frame_id}: timestamp filled from map-keyframes"
                )
            if source_frame_idx is None and csv_row is not None:
                source_frame_idx = csv_row.get("frame_idx")

            try:
                timestamp_val = float(timestamp)
            except (TypeError, ValueError):
                raise _fail(
                    f"{job.namespace}/{job.video_id}#{frame_id} (vector_id={vector_id}): "
                    f"no valid timestamp (record and map-keyframes both missing/invalid it)"
                )
            if not np.isfinite(timestamp_val) or timestamp_val < 0:
                raise _fail(
                    f"{job.namespace}/{job.video_id}#{frame_id} (vector_id={vector_id}): "
                    f"invalid timestamp {timestamp_val!r}"
                )

            split = str(record.get("parent_namespace") or job.namespace)
            video_id = str(record.get("video_id") or job.video_id)
            _validate_video_or_split(split, "split", f"{job.namespace}/{job.video_id}")
            _validate_video_or_split(video_id, "video_id", f"{job.namespace}/{job.video_id}")

            asset_key = str(record.get("frame_path") or "").replace("\\", "/").strip("/")
            if not asset_key:
                raise _fail(
                    f"{job.namespace}/{job.video_id}#{frame_id} (vector_id={vector_id}): "
                    f"record has no frame_path (asset_key)"
                )
            if ".." in asset_key.split("/"):
                raise _fail(f"{job.namespace}/{job.video_id}#{frame_id}: unsafe frame_path {asset_key!r}")
            if asset_key in seen_asset_keys:
                raise _fail(f"duplicate asset_key across the corpus: {asset_key!r}")
            seen_asset_keys.add(asset_key)

            rows.append(
                {
                    "vector_id": int(vector_id),
                    "split": split,
                    "video_id": video_id,
                    "embedding_row": int(embedding_row),
                    "keyframe_ordinal": int(keyframe_ordinal),
                    "timestamp_ms": int(round(timestamp_val * 1000.0)),
                    "asset_key": asset_key,
                    "frame_path": asset_key,  # compatibility alias, see keyframe resolver
                    "source_frame_id": int(source_frame_idx) if source_frame_idx is not None else None,
                    "source_fps": float(source_fps) if source_fps is not None else None,
                }
            )

        next_vector_id += len(block)

    df = pd.DataFrame.from_records(rows)
    if not df["vector_id"].is_unique:
        raise _fail("duplicate vector_id values in the built parquet")
    if index is None or index.ntotal != len(df):
        raise _fail(f"FAISS ntotal={0 if index is None else index.ntotal} != parquet rows={len(df)}")

    return BuildResult(df=df, index=index, videos=len(jobs), warnings=warnings)


def _build_video_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Per-video rollup published as ``jina_video_metadata.parquet``.

    One row per (split, video_id): frame count, contiguous vector-id range,
    and the embedding dimension. Emitted so the builder output, the manifest
    (``jina_video_metadata`` artifact) and the runtime profile agree.
    """
    grouped = (
        df.groupby(["split", "video_id"], as_index=False)
        .agg(
            frame_count=("vector_id", "size"),
            first_vector_id=("vector_id", "min"),
            last_vector_id=("vector_id", "max"),
        )
        .sort_values(["split", "video_id"])
        .reset_index(drop=True)
    )
    grouped["parent_namespace"] = grouped["split"]
    grouped["embedding_dim"] = EXPECTED_DIM
    return grouped[
        ["video_id", "parent_namespace", "split", "frame_count", "embedding_dim",
         "first_vector_id", "last_vector_id"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-root", type=Path, required=True,
                        help="Directory of <namespace>/<video_id>.npy files.")
    parser.add_argument("--records-root", type=Path, required=True,
                        help="Directory of <namespace>/<video_id>.json record files.")
    parser.add_argument("--map-keyframes-root", type=Path, default=None,
                        help="Optional directory of <video_id>.csv map-keyframes files, "
                             "used only to fill in a value a record is missing.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--model-id", default="jinaai/jina-clip-v2",
                        help="HuggingFace repo id or local snapshot dir the corpus was built with.")
    parser.add_argument("--model-revision", required=True,
                        help="Pinned commit hash the encoder ran at -- required, not optional: "
                             "the runtime retriever refuses to query without a matching pin.")
    parser.add_argument("--embedding-run", default="",
                        help="Free-form label for this corpus build (for index_meta.json only).")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    result = build_index(args.embeddings_root, args.records_root, args.map_keyframes_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "jina_faiss.index"
    parquet_path = args.out_dir / "jina_global_ids.parquet"
    video_meta_path = args.out_dir / "jina_video_metadata.parquet"
    meta_path = args.out_dir / "jina_index_meta.json"
    report_path = args.out_dir / "jina_build_report.json"

    import faiss

    faiss.write_index(result.index, str(index_path))
    result.df.to_parquet(parquet_path, index=False)
    _build_video_metadata(result.df).to_parquet(video_meta_path, index=False)

    elapsed = time.perf_counter() - started
    meta = {
        "backend": "jina_clip_v2",
        "embedding_run": args.embedding_run or None,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dimension": EXPECTED_DIM,
        "metric": "inner_product_on_l2_normalized_vectors",
        "normalization": "l2",
        "vector_count": int(result.index.ntotal),
        "video_count": result.videos,
        "metadata_schema_version": 1,
        "generated_at": _now(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    report = {
        **meta,
        "elapsed_seconds": elapsed,
        "warning_count": len(result.warnings),
        "warnings_sample": result.warnings[:50],
        "embeddings_root": str(args.embeddings_root),
        "records_root": str(args.records_root),
        "map_keyframes_root": str(args.map_keyframes_root) if args.map_keyframes_root else None,
        "files": {
            "index": str(index_path),
            "global_ids": str(parquet_path),
            "video_metadata": str(video_meta_path),
            "index_meta": str(meta_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Built Jina index: {result.index.ntotal} vectors across {result.videos} videos "
          f"in {elapsed:.1f}s ({len(result.warnings)} warnings)")
    print(f"  {index_path}")
    print(f"  {parquet_path}")
    print(f"  {video_meta_path}")
    print(f"  {meta_path}")
    print(f"  {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
