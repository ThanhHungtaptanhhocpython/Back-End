#!/usr/bin/env python
"""Run batch PaddleOCR over Jina keyframes stored in Azure Blob Storage.

The worker is intentionally separate from the Colab notebook kernel. It reads
the final Jina ``global_ids.parquet`` as the source of truth, downloads one
video's frames to local SSD, OCRs them in batches, uploads a per-video
checkpoint, then deletes the temporary frames. A restarted runtime resumes at
the next incomplete video.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm.auto import tqdm

# Azure's HTTP logger emits one INFO block for every frame download. Keep the
# concise OCR progress logs while still allowing Azure warnings/errors through.
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

try:  # Keep ``--help`` usable on backend environments that do not run OCR.
    from azure.storage.blob import BlobServiceClient, ContentSettings
except ImportError:  # pragma: no cover - depends on the runtime environment
    BlobServiceClient = None
    ContentSettings = None

try:  # Paddle is installed only in the isolated Colab worker environment.
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - depends on the runtime environment
    PaddleOCR = None


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR Azure-hosted Jina keyframes with PaddleOCR.")
    parser.add_argument("--work-dir", type=Path, default=Path("/content/ocr_azure_work"))
    parser.add_argument("--keyframes-container", default="keyframes")
    parser.add_argument("--embeddings-container", default="embeddings")
    parser.add_argument("--results-container", default="metadata")
    parser.add_argument(
        "--global-ids-blob",
        default="indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina/global_ids.parquet",
    )
    parser.add_argument("--run-name", default="fine_keyframes_jina_paddleocr_v1")
    parser.add_argument("--only-namespaces", default="", help="Comma-separated, for example L21_a,L22_a.")
    parser.add_argument("--max-videos", type=int, default=None, help="Useful for a smoke test.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--log-every-videos", type=int, default=10)
    parser.add_argument("--drive-output-dir", type=Path, default=None, help="Optional mounted Google Drive backup directory.")
    parser.add_argument("--drive-snapshot-every-videos", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.40)
    parser.add_argument("--merge-only", action="store_true", help="Merge completed per-video artifacts without OCR.")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def upload_json(container: Any, blob_name: str, payload: Any) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    container.get_blob_client(blob_name).upload_blob(
        encoded,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
    )


def download_json(container: Any, blob_name: str) -> dict[str, Any] | None:
    client = container.get_blob_client(blob_name)
    if not client.exists():
        return None
    return json.loads(client.download_blob().readall())


def load_global_ids(container: Any, blob_name: str, work_dir: Path) -> pd.DataFrame:
    local_path = work_dir / "global_ids.parquet"
    if not local_path.exists() or local_path.stat().st_size == 0:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Downloading Jina global IDs: {blob_name}")
        with local_path.open("wb") as handle:
            container.get_blob_client(blob_name).download_blob(max_concurrency=4).readinto(handle)
    table = pd.read_parquet(local_path)
    required = {"vector_id", "parent_namespace", "video_id", "frame_id", "frame_path", "timestamp", "source_frame_idx"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"global_ids.parquet missing columns: {missing}")
    if table["vector_id"].isna().any() or not table["vector_id"].is_unique:
        raise RuntimeError("global_ids.parquet has missing or duplicate vector_id values")
    if table[["frame_path", "timestamp", "source_frame_idx"]].isna().any().any():
        raise RuntimeError("global_ids.parquet has incomplete Jina frame mapping")
    return table.sort_values(["parent_namespace", "video_id", "timestamp", "vector_id"], kind="stable")


def parse_result(result: Any, min_confidence: float) -> list[str]:
    if result is None:
        return []
    payload = getattr(result, "json", {}) or {}
    payload = payload.get("res", payload)
    texts = payload.get("rec_texts", []) or []
    scores = payload.get("rec_scores", []) or []
    try:
        scores = scores.tolist()
    except AttributeError:
        pass
    if len(scores) != len(texts):
        scores = [1.0] * len(texts)
    return [
        str(text).strip()
        for text, score in zip(texts, scores)
        if str(text).strip() and float(score) >= min_confidence
    ]


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def download_video_frames(keyframes: Any, rows: list[dict[str, Any]], cache_dir: Path, workers: int) -> list[tuple[dict[str, Any], Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def download_one(row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        remote = str(row["frame_path"]).replace("\\", "/").strip("/")
        destination = cache_dir / Path(remote).name
        if not destination.exists() or destination.stat().st_size == 0:
            temporary = destination.with_suffix(destination.suffix + ".part")
            with temporary.open("wb") as handle:
                keyframes.get_blob_client(remote).download_blob(max_concurrency=1).readinto(handle)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty blob: {remote}")
            temporary.replace(destination)
        return row, destination

    output: list[tuple[dict[str, Any], Path]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(download_one, row) for row in rows]
        for future in as_completed(futures):
            output.append(future.result())
    output.sort(key=lambda item: (float(item[0]["timestamp"]), int(item[0]["vector_id"])))
    return output


def run_ocr_for_video(ocr: PaddleOCR, items: list[tuple[dict[str, Any], Path]], args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch in chunks(items, args.batch_size):
        paths = [str(path) for _, path in batch]
        try:
            predictions = list(ocr.predict(paths))
        except Exception as error:
            log(f"Batch failed ({error}); retrying images one by one")
            predictions = []
            for path in paths:
                one = list(ocr.predict(path))
                predictions.append(one[0] if one else None)
        if len(predictions) != len(batch):
            raise RuntimeError(f"Paddle returned {len(predictions)} results for {len(batch)} images")
        for (row, _), prediction in zip(batch, predictions):
            texts = parse_result(prediction, args.min_confidence)
            if not texts:
                continue
            frame_path = str(row["frame_path"]).replace("\\", "/").strip("/")
            records.append(
                {
                    "vector_id": int(row["vector_id"]),
                    "faiss_id": int(row["vector_id"]),
                    "video_id": str(row["video_id"]),
                    "parent_namespace": str(row["parent_namespace"]),
                    "split": str(row["parent_namespace"]),
                    "frame_id": str(row["frame_id"]),
                    "frame_name": Path(frame_path).name,
                    "frame_path": frame_path,
                    "source_frame_idx": int(row["source_frame_idx"]),
                    "global_frame_id": int(row["source_frame_idx"]),
                    "timestamp": float(row["timestamp"]),
                    "alignment_source": "jina_global_ids_exact",
                    "ocr_text": " ".join(texts),
                    "language": "vi",
                    "ocr_engine": "PaddleOCR_PP-OCRv6",
                }
            )
    return records


def checkpoint_blob(run_prefix: str, namespace: str, video_id: str) -> str:
    return f"{run_prefix}/videos/{namespace}/{video_id}.json"


def drive_video_checkpoint_path(drive_dir: Path, namespace: str, video_id: str) -> Path:
    return drive_dir / "videos" / namespace / f"{video_id}.json"


def save_drive_video_checkpoint(drive_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(drive_video_checkpoint_path(drive_dir, str(payload["namespace"]), str(payload["video_id"])), payload)


def save_drive_state(drive_dir: Path, run_name: str, completed: int, total: int, namespace: str, video_id: str) -> None:
    atomic_write_json(
        drive_dir / "ocr_state.json",
        {
            "run_name": run_name,
            "completed_video_count": completed,
            "total_video_count": total,
            "last_completed_video": f"{namespace}/{video_id}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def write_drive_snapshot(drive_dir: Path) -> int:
    records: list[dict[str, Any]] = []
    for path in sorted((drive_dir / "videos").rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "completed":
            records.extend(payload.get("records") or [])
    records.sort(key=lambda item: int(item["vector_id"]))
    if len({item["vector_id"] for item in records}) != len(records):
        raise RuntimeError("Duplicate vector_id found while building Google Drive OCR snapshot")
    atomic_write_json(drive_dir / "ocr_results_jina.json", records)
    return len(records)


def merge_completed_results(results: Any, run_prefix: str, total_videos: int) -> list[dict[str, Any]]:
    prefix = f"{run_prefix}/videos/"
    records: list[dict[str, Any]] = []
    completed = 0
    for blob in results.list_blobs(name_starts_with=prefix):
        payload = download_json(results, blob.name)
        if not payload or payload.get("status") != "completed":
            continue
        completed += 1
        records.extend(payload.get("records") or [])
    if completed != total_videos:
        raise RuntimeError(f"Only {completed}/{total_videos} videos are completed; refusing to publish partial final OCR JSON")
    records.sort(key=lambda item: int(item["vector_id"]))
    if len({item["vector_id"] for item in records}) != len(records):
        raise RuntimeError("Duplicate vector_id found while merging OCR records")
    final_blob = f"{run_prefix}/ocr_results_jina.json"
    upload_json(results, final_blob, records)
    upload_json(
        results,
        f"{run_prefix}/report.json",
        {"status": "completed", "video_count": completed, "ocr_record_count": len(records), "final_blob": final_blob},
    )
    log(f"Published {len(records):,} OCR records: {final_blob}")
    return records


def main() -> int:
    args = parse_args()
    if BlobServiceClient is None or ContentSettings is None:
        raise RuntimeError("azure-storage-blob is required; run this through the prepared Colab worker environment")
    if PaddleOCR is None and not args.merge_only:
        raise RuntimeError("paddleocr is required; run this through the prepared Colab worker environment")
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not connection_string:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING must be supplied through a Colab secret/environment variable")
    if args.batch_size < 1 or args.download_workers < 1 or args.log_every_videos < 1 or args.drive_snapshot_every_videos < 1:
        raise RuntimeError("batch size, worker count, and log/snapshot intervals must be positive")

    service = BlobServiceClient.from_connection_string(connection_string)
    keyframes = service.get_container_client(args.keyframes_container)
    embeddings = service.get_container_client(args.embeddings_container)
    results = service.get_container_client(args.results_container)
    run_prefix = f"ocr-runs/{args.run_name}"
    drive_dir = args.drive_output_dir.resolve() if args.drive_output_dir else None
    if drive_dir:
        drive_dir.mkdir(parents=True, exist_ok=True)
        log(f"Google Drive backup: {drive_dir}")
    table = load_global_ids(embeddings, args.global_ids_blob, args.work_dir)

    only_namespaces = {value.strip() for value in args.only_namespaces.split(",") if value.strip()}
    if only_namespaces:
        table = table[table["parent_namespace"].isin(only_namespaces)]
    groups = list(table.groupby(["parent_namespace", "video_id"], sort=True))
    if args.max_videos is not None:
        groups = groups[: max(0, args.max_videos)]
    log(f"Selected {len(groups):,} videos and {len(table):,} Jina keyframes")

    if args.merge_only:
        if args.max_videos is not None or only_namespaces:
            raise RuntimeError(
                "Refusing to merge a partial OCR corpus. Set --max-videos unset and --only-namespaces empty first."
            )
        records = merge_completed_results(results, run_prefix, len(groups))
        if drive_dir:
            atomic_write_json(drive_dir / "ocr_results_jina.json", records)
            save_drive_state(drive_dir, args.run_name, len(groups), len(groups), "final", "merge")
            log(f"Google Drive final OCR result saved: {drive_dir / 'ocr_results_jina.json'}")
        return 0

    import paddle

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("Paddle GPU is unavailable; stop rather than silently OCR on CPU")
    log(f"Paddle {paddle.__version__}; GPU count={paddle.device.cuda.device_count()}")
    ocr = PaddleOCR(
        lang="vi",
        ocr_version="PP-OCRv6",
        device="gpu:0",
        enable_hpi=False,
        text_recognition_batch_size=args.batch_size,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    started = time.time()
    completed = 0
    processed_this_session = 0

    def report_progress(current_namespace: str, current_video_id: str, detail: str) -> None:
        elapsed = max(time.time() - started, 1e-6)
        video_rate = processed_this_session / elapsed
        if video_rate > 0:
            eta_seconds = round((len(groups) - completed) / video_rate)
            eta_label = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        else:
            eta_label = "estimating after first completed video"
        log(
            f"Progress: {completed}/{len(groups)} videos | "
            f"current={current_namespace}/{current_video_id} | {detail} | "
            f"elapsed={time.strftime('%H:%M:%S', time.gmtime(round(elapsed)))} | ETA={eta_label}"
        )

    for position, ((namespace, video_id), frames) in enumerate(groups, 1):
        namespace, video_id = str(namespace), str(video_id)
        marker = checkpoint_blob(run_prefix, namespace, video_id)
        should_log = position == 1 or position == len(groups) or position % args.log_every_videos == 0
        existing = download_json(results, marker)
        if existing and existing.get("status") == "completed":
            if drive_dir:
                save_drive_video_checkpoint(drive_dir, existing)
            completed += 1
            if drive_dir:
                save_drive_state(drive_dir, args.run_name, completed, len(groups), namespace, video_id)
                if completed % args.drive_snapshot_every_videos == 0:
                    snapshot_count = write_drive_snapshot(drive_dir)
                    log(f"Google Drive OCR snapshot saved: {snapshot_count:,} records")
            if should_log:
                report_progress(namespace, video_id, "checkpoint already exists")
            continue

        rows = frames.to_dict("records")
        cache_dir = args.work_dir / "frames" / namespace / video_id
        if should_log:
            report_progress(namespace, video_id, f"downloading {len(rows):,} frames")
        try:
            items = download_video_frames(keyframes, rows, cache_dir, args.download_workers)
            video_started = time.time()
            records = run_ocr_for_video(ocr, items, args)
            checkpoint = {
                "status": "completed",
                "namespace": namespace,
                "video_id": video_id,
                "processed_frame_count": len(items),
                "ocr_record_count": len(records),
                "records": records,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            upload_json(results, marker, checkpoint)
            if drive_dir:
                save_drive_video_checkpoint(drive_dir, checkpoint)
            completed += 1
            processed_this_session += 1
            if drive_dir:
                save_drive_state(drive_dir, args.run_name, completed, len(groups), namespace, video_id)
                if completed % args.drive_snapshot_every_videos == 0:
                    snapshot_count = write_drive_snapshot(drive_dir)
                    log(f"Google Drive OCR snapshot saved: {snapshot_count:,} records")
            elapsed = max(time.time() - video_started, 1e-6)
            if should_log:
                report_progress(namespace, video_id, f"{len(records):,} text frames, {len(items) / elapsed:.2f} img/s")
        except Exception as error:
            upload_json(results, marker, {"status": "failed", "namespace": namespace, "video_id": video_id, "error": repr(error)})
            raise
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    log(f"OCR video checkpoints complete: {completed}/{len(groups)} in {(time.time() - started) / 60:.1f} min")
    if drive_dir:
        snapshot_count = write_drive_snapshot(drive_dir)
        log(f"Google Drive OCR snapshot saved: {snapshot_count:,} records")
    log("Run the separate --merge-only command only after the intended corpus is fully complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
