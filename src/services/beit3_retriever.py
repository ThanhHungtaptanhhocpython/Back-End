"""BEiT-3 text-to-image visual retrieval service.

Owns every piece of the real retrieval path:
    text query -> SentencePiece tokens -> BEiT3 language head ->
    normalized 1024-d query vector -> FAISS IndexIDMap2(IndexFlatIP) search ->
    global_ids.parquet lookup -> structured results.

The BEiT3 model, FAISS index, and parquet metadata are loaded exactly once
(lazy singleton, see `get_beit3_retriever`) and reused for every request.
This module intentionally does not import `src.utils.beit3_processing`
(that file wraps `bert-base-uncased`, not BEiT3, and its 768-d output is
incompatible with this 1024-d FAISS index).
"""

from __future__ import annotations

import bisect
import json
import csv
import logging
import re
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch

from src.config.settings import Settings, get_settings
from src.utils.beit3_backbone import BEiT3ForRetrieval, build_large_retrieval_config

logger = logging.getLogger(__name__)

# Historical fairseq/XLM-R offset mapping (see module docstring in the task
# spec): fairseq reserves ids 0-3 for <s>, <pad>, </s>, <unk>; every other
# SentencePiece piece id is shifted by +1, except spm's own id 0 (its <unk>)
# which collapses onto fairseq's <unk> (3).
BOS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3

EXPECTED_DIM = 1024

# Image-query preprocessing must match exactly how the BEiT3 keyframe index was
# built (see scripts/model_encoding/run_beit3_encoder.py): bicubic resize to
# 384x384, scale to [0, 1], then normalize with timm's IMAGENET_INCEPTION_MEAN /
# IMAGENET_INCEPTION_STD (both 0.5 per channel). Anything else lands the query
# in a slightly different point of the shared space than the indexed frames.
IMAGE_INPUT_SIZE = 384
IMAGE_NORM_MEAN = (0.5, 0.5, 0.5)
IMAGE_NORM_STD = (0.5, 0.5, 0.5)

_VECTOR_ID_CANDIDATES = ["global_id", "global_frame_id", "vector_id", "faiss_id", "id"]
_VIDEO_ID_CANDIDATES = ["video_id", "video"]
_FRAME_ID_CANDIDATES = ["frame_id", "frame_index", "keyframe_id", "frame_name"]
_FRAME_PATH_CANDIDATES = ["frame_path", "image_path", "keyframe_path", "path"]
_TIMESTAMP_CANDIDATES = ["timestamp", "timestamp_s", "pts", "pts_time", "time_sec", "time"]
_NAMESPACE_CANDIDATES = ["namespace", "source_namespace", "parent_namespace", "split", "scope"]


class BEiT3RetrieverError(RuntimeError):
    """Raised for any BEiT3 startup or search invariant violation.

    Callers must not catch this and silently fall back to another encoder;
    that would silently return results from the wrong embedding space.
    """


def _first_matching_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


class BEiT3Retriever:
    """Owns the BEiT3 model, tokenizer, FAISS index, and metadata tables."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._device = self._resolve_device(self._settings.beit3_device)
        self._max_seq_len = self._settings.beit3_max_seq_len

        self._tokenizer = self._load_tokenizer(self._settings.beit3_tokenizer_path)
        self._model = self._load_model(self._settings.beit3_checkpoint_path)
        self._index = self._load_faiss_index(self._settings.beit3_faiss_index_path)
        self._global_ids = self._load_global_ids(self._settings.beit3_global_ids_path)
        self._video_metadata = self._load_optional_parquet(
            self._settings.beit3_video_metadata_path, label="video_metadata.parquet"
        )
        self._media_info_by_id = self._load_media_info_dir(self._settings.src_dir / "dict" / "media-info")
        self._keyframe_time_by_video = self._load_keyframe_time_maps(self._settings.src_dir / "dict" / "map-keyframes")
        self._index_meta = self._load_optional_json(
            self._settings.beit3_index_meta_path, label="index_meta.json"
        )

        self._validate_consistency()
        self._columns = self._detect_columns(self._global_ids)
        self._id_to_row = self._build_id_lookup(self._global_ids, self._columns["vector_id"])
        self._video_meta_by_id: dict[Any, dict] | None = self._build_video_metadata_lookup()
        self._video_to_rows: dict[str, list[dict]] = self._build_video_to_rows()

        logger.info(
            "BEiT3Retriever ready: device=%s ntotal=%d rows=%d columns=%s videos=%d",
            self._device,
            self._index.ntotal,
            len(self._global_ids),
            self._columns,
            len(self._video_to_rows),
        )

    # ------------------------------------------------------------------
    # Startup / loading
    # ------------------------------------------------------------------

    def _resolve_device(self, requested: str | None) -> torch.device:
        requested_norm = (requested or "cpu").strip().lower()
        if requested_norm not in ("cuda", "cpu"):
            raise BEiT3RetrieverError(
                f"Invalid BEIT3_DEVICE={requested!r}; expected 'cuda' or 'cpu'."
            )
        if requested_norm == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "BEIT3_DEVICE=cuda was requested but CUDA is not available; "
                "falling back to CPU. The model architecture is unchanged."
            )
            return torch.device("cpu")
        return torch.device(requested_norm)

    def _require_path(self, path: Path | None, env_var: str) -> Path:
        if path is None:
            raise BEiT3RetrieverError(f"{env_var} is not set.")
        resolved = Path(path)
        if not resolved.exists():
            raise BEiT3RetrieverError(f"{env_var} points to a missing file: {resolved}")
        return resolved

    def _load_tokenizer(self, path: Path | None) -> spm.SentencePieceProcessor:
        resolved = self._require_path(path, "BEIT3_TOKENIZER_PATH")
        processor = spm.SentencePieceProcessor()
        ok = processor.load(str(resolved))
        if not ok:
            raise BEiT3RetrieverError(f"Failed to load SentencePiece tokenizer from {resolved}")
        return processor

    def _load_model(self, checkpoint_path: Path | None) -> BEiT3ForRetrieval:
        resolved = self._require_path(checkpoint_path, "BEIT3_CHECKPOINT_PATH")
        config = build_large_retrieval_config(img_size=384)
        model = BEiT3ForRetrieval(config)

        checkpoint = torch.load(str(resolved), map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise BEiT3RetrieverError(
                "BEiT3 checkpoint does not match the expected "
                "beit3_large_patch16_384_retrieval architecture "
                f"(missing={len(missing)}, unexpected={len(unexpected)}). "
                f"First missing keys: {missing[:5]}. First unexpected keys: {unexpected[:5]}."
            )

        model.eval()
        model.to(self._device)
        return model

    def _load_faiss_index(self, index_path: Path | None) -> faiss.Index:
        resolved = self._require_path(index_path, "BEIT3_FAISS_INDEX_PATH")
        index = faiss.read_index(str(resolved))
        if index.d != EXPECTED_DIM:
            raise BEiT3RetrieverError(
                f"FAISS index dimension {index.d} != {EXPECTED_DIM} (path={resolved})."
            )
        return index

    def _load_global_ids(self, path: Path | None) -> pd.DataFrame:
        resolved = self._require_path(path, "BEIT3_GLOBAL_IDS_PATH")
        df = pd.read_parquet(resolved)
        if df.empty:
            raise BEiT3RetrieverError(f"global_ids.parquet at {resolved} is empty.")
        return df

    def _load_optional_parquet(self, path: Path | None, label: str) -> pd.DataFrame | None:
        if path is None:
            logger.warning("%s path not configured; enrichment from it will be skipped.", label)
            return None
        resolved = self._require_path(path, f"path for {label}")
        return pd.read_parquet(resolved)

    def _load_optional_json(self, path: Path | None, label: str) -> dict | None:
        if path is None:
            return None
        resolved = self._require_path(path, f"path for {label}")
        with resolved.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as exc:
                raise BEiT3RetrieverError(f"Failed to parse {label} at {resolved}: {exc}") from exc

    def _load_media_info_dir(self, path: Path) -> dict[str, dict]:
        """Load per-video JSON metadata, including YouTube watch URLs."""
        if not path.exists():
            logger.warning("media-info directory not found; video URL enrichment skipped: %s", path)
            return {}

        lookup: dict[str, dict] = {}
        for json_path in path.glob("*.json"):
            video_id = json_path.stem
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable media-info file %s: %s", json_path, exc)
                continue
            if isinstance(data, dict):
                lookup[video_id] = data
        logger.info("Loaded media-info metadata for %d videos from %s", len(lookup), path)
        return lookup


    def _load_keyframe_time_maps(self, path: Path) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
        """Load keyframe ordinal/frame-index to timestamp mappings from CSV files."""
        if not path.exists():
            logger.warning("map-keyframes directory not found; timeline timestamps skipped: %s", path)
            return {}

        lookup: dict[str, dict[str, Any]] = {}
        for csv_path in path.glob("*.csv"):
            by_n: dict[int, dict[str, Any]] = {}
            by_frame_idx: dict[int, dict[str, Any]] = {}
            try:
                with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            n = int(str(row.get("n") or "").strip())
                            pts_time = float(str(row.get("pts_time") or "0").strip())
                            fps = float(str(row.get("fps") or "25").strip())
                            frame_idx = int(str(row.get("frame_idx") or "0").strip())
                        except (TypeError, ValueError):
                            continue
                        item = {"timestamp": pts_time, "fps": fps, "source_frame_idx": frame_idx, "keyframe_number": n}
                        by_n[n] = item
                        by_frame_idx[frame_idx] = item
            except OSError as exc:
                logger.warning("Skipping unreadable map-keyframes file %s: %s", csv_path, exc)
                continue
            lookup[csv_path.stem] = {"n": by_n, "frame_idx": by_frame_idx, "frame_idx_values": sorted(by_frame_idx)}
        logger.info("Loaded keyframe timestamp maps for %d videos from %s", len(lookup), path)
        return lookup

    @staticmethod
    def _candidate_frame_numbers(*values: Any) -> list[int]:
        candidates: list[int] = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip().replace("\\", "/")
            if not text:
                continue
            stem = Path(text).stem
            for part in (text, stem):
                groups = re.findall(r"\d+", part)
                digit_candidates = []
                if groups:
                    digit_candidates.append(groups[-1])
                    digit_candidates.append("".join(groups))
                for digits in digit_candidates:
                    if digits:
                        number = int(digits)
                        if number not in candidates:
                            candidates.append(number)
        return candidates

    def _keyframe_time_for(self, video_id: Any, *frame_values: Any) -> dict[str, Any] | None:
        mapping = self._keyframe_time_by_video.get(str(video_id))
        if not mapping:
            return None
        numbers = self._candidate_frame_numbers(*frame_values)
        for number in numbers:
            if number in mapping["frame_idx"]:
                return {**mapping["frame_idx"][number], "timestamp_source": "map_frame_idx_exact"}
        for number in numbers:
            if number in mapping["n"]:
                return {**mapping["n"][number], "timestamp_source": "map_keyframe_number_exact"}

        frame_idx_values = mapping.get("frame_idx_values") or []
        if not frame_idx_values:
            return None
        for number in numbers:
            # Treat large numeric frame ids as source frame indexes. Some corpus
            # metadata lands between sampled keyframes, so an exact CSV hit is
            # not guaranteed. A small nearest-neighbor tolerance prevents the
            # frontend from falling back to frame_id / 25 and seeking the wrong
            # point in public video streams.
            if number < 1000:
                continue
            pos = bisect.bisect_left(frame_idx_values, number)
            candidates = []
            if pos < len(frame_idx_values):
                candidates.append(frame_idx_values[pos])
            if pos > 0:
                candidates.append(frame_idx_values[pos - 1])
            if not candidates:
                continue
            nearest = min(candidates, key=lambda value: abs(value - number))
            info = mapping["frame_idx"][nearest]
            fps = float(info.get("fps") or 25.0)
            tolerance = max(12, int(round(fps * 1.25)))
            if abs(nearest - number) <= tolerance:
                return {
                    **info,
                    "timestamp_source": "map_frame_idx_nearest",
                    "timestamp_matched_frame_idx": nearest,
                    "timestamp_frame_idx_delta": nearest - number,
                }
        return None
    def _validate_consistency(self) -> None:
        ntotal = self._index.ntotal
        n_rows = len(self._global_ids)
        if ntotal != n_rows:
            raise BEiT3RetrieverError(
                f"FAISS index ntotal={ntotal} does not match global_ids.parquet "
                f"row count={n_rows}. The index and metadata are out of sync."
            )

    def _detect_columns(self, df: pd.DataFrame) -> dict[str, str | None]:
        columns = list(df.columns)
        settings = self._settings

        vector_id = settings.beit3_col_vector_id or _first_matching_column(
            columns, _VECTOR_ID_CANDIDATES
        )
        video_id = settings.beit3_col_video_id or _first_matching_column(
            columns, _VIDEO_ID_CANDIDATES
        )
        if vector_id is None:
            raise BEiT3RetrieverError(
                "Could not identify the vector-id column in global_ids.parquet "
                f"(looked for {_VECTOR_ID_CANDIDATES}, found columns={columns}). "
                "Set BEIT3_COL_VECTOR_ID explicitly."
            )
        if video_id is None:
            raise BEiT3RetrieverError(
                "Could not identify the video-id column in global_ids.parquet "
                f"(looked for {_VIDEO_ID_CANDIDATES}, found columns={columns}). "
                "Set BEIT3_COL_VIDEO_ID explicitly."
            )

        resolved = {
            "vector_id": vector_id,
            "video_id": video_id,
            "frame_id": settings.beit3_col_frame_id or _first_matching_column(columns, _FRAME_ID_CANDIDATES),
            "frame_path": settings.beit3_col_frame_path or _first_matching_column(columns, _FRAME_PATH_CANDIDATES),
            "timestamp": settings.beit3_col_timestamp or _first_matching_column(columns, _TIMESTAMP_CANDIDATES),
            "namespace": settings.beit3_col_namespace or _first_matching_column(columns, _NAMESPACE_CANDIDATES),
        }
        for logical_name in ("frame_id", "frame_path", "timestamp", "namespace"):
            if resolved[logical_name] is None:
                logger.info(
                    "global_ids.parquet has no detected '%s' column; it will be null in results.",
                    logical_name,
                )
        return resolved

    def _build_id_lookup(self, df: pd.DataFrame, vector_id_col: str) -> dict[Any, dict]:
        records = df.to_dict(orient="records")
        lookup: dict[Any, dict] = {}
        for row in records:
            lookup[row[vector_id_col]] = row
        return lookup

    def _build_video_metadata_lookup(self) -> dict[Any, dict] | None:
        if self._video_metadata is None:
            return None
        video_id_col = _first_matching_column(list(self._video_metadata.columns), _VIDEO_ID_CANDIDATES)
        if video_id_col is None:
            logger.warning(
                "video_metadata.parquet has no recognizable video-id column "
                "(columns=%s); skipping enrichment.",
                list(self._video_metadata.columns),
            )
            return None
        return {
            row[video_id_col]: row
            for row in self._video_metadata.to_dict(orient="records")
        }

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _encode_piece_ids(self, text: str) -> list[int]:
        spm_ids = self._tokenizer.encode(text, out_type=int)
        return [UNK_ID if spm_id == 0 else spm_id + 1 for spm_id in spm_ids]

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        piece_ids = self._encode_piece_ids(text)
        max_len = self._max_seq_len
        if len(piece_ids) > max_len - 2:
            piece_ids = piece_ids[: max_len - 2]

        tokens = [BOS_ID] + piece_ids + [EOS_ID]
        num_tokens = len(tokens)
        padding_mask = [False] * num_tokens + [True] * (max_len - num_tokens)
        tokens = tokens + [PAD_ID] * (max_len - num_tokens)

        token_tensor = torch.tensor([tokens], dtype=torch.long, device=self._device)
        mask_tensor = torch.tensor([padding_mask], dtype=torch.bool, device=self._device)
        return token_tensor, mask_tensor

    def _validate_query_vector(self, vec: np.ndarray) -> None:
        if vec.shape != (1, EXPECTED_DIM):
            raise BEiT3RetrieverError(
                f"Query embedding has shape {vec.shape}, expected (1, {EXPECTED_DIM})."
            )
        if not np.isfinite(vec).all():
            raise BEiT3RetrieverError("Query embedding contains NaN/Inf values.")
        norm = float(np.linalg.norm(vec))
        if not np.isclose(norm, 1.0, atol=1e-2):
            raise BEiT3RetrieverError(f"Query embedding L2 norm={norm:.4f}, expected ~1.0.")

    def encode_text(self, query: str) -> np.ndarray:
        """Encode `query` into a normalized (1, 1024) float32 vector."""
        if not query or not query.strip():
            raise BEiT3RetrieverError("Query text must be a non-empty string.")

        tokens, padding_mask = self._tokenize(query.strip())
        with torch.no_grad():
            _, language_cls = self._model(
                text_description=tokens, padding_mask=padding_mask, only_infer=True
            )
        vec = language_cls.detach().cpu().numpy().astype(np.float32)
        self._validate_query_vector(vec)
        return vec

    def _preprocess_image(self, image: Any) -> torch.Tensor:
        """Apply the exact index-time preprocessing to an image query.

        ``image`` may be a filesystem path, a file-like object, or a PIL
        ``Image``. Returns a ``(1, 3, 384, 384)`` float32 tensor on the model
        device.
        """
        from PIL import Image

        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            try:
                pil = Image.open(image).convert("RGB")
            except (OSError, ValueError) as exc:
                raise BEiT3RetrieverError(f"Could not read the query image: {exc}") from exc

        resample = getattr(Image, "Resampling", Image).BICUBIC
        pil = pil.resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), resample)

        arr = np.asarray(pil, dtype=np.float32) / 255.0
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise BEiT3RetrieverError("Query image did not decode to 3-channel RGB.")
        arr = (arr - np.asarray(IMAGE_NORM_MEAN, dtype=np.float32)) / np.asarray(
            IMAGE_NORM_STD, dtype=np.float32
        )
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0).to(self._device)

    def encode_image(self, image: Any) -> np.ndarray:
        """Encode an image query into a normalized (1, 1024) float32 vector."""
        tensor = self._preprocess_image(image)
        with torch.no_grad():
            vision_cls, _ = self._model(image=tensor, only_infer=True)
        if vision_cls is None:
            raise BEiT3RetrieverError("BEiT3 returned no vision embedding for the query image.")
        vec = vision_cls.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = (vec / np.where(norms > 0.0, norms, 1.0)).astype(np.float32)
        self._validate_query_vector(vec)
        return vec

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
        """Convert a pandas/numpy scalar to a native, JSON-safe Python value.

        pandas `.to_dict()` leaves numpy scalar dtypes (np.int64, np.float64,
        ...) in the row values, and NaN is a float, not None. Both break the
        "return null, never invent a value" contract and can fail JSON
        serialization outright, so every field that reaches the API response
        goes through this first.
        """
        if value is None:
            return None
        if isinstance(value, (np.floating, float)):
            f = float(value)
            return None if np.isnan(f) else f
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if pd.isna(value) if not isinstance(value, (list, dict)) else False:
            return None
        return value

    def _build_result(self, rank: int, score: float, vector_id: int, row: dict) -> dict:
        cols = self._columns
        video_id = self._to_json_safe(row.get(cols["video_id"]))
        frame_id = self._to_json_safe(row.get(cols["frame_id"])) if cols["frame_id"] else None
        frame_submission_id = self._normalize_submission_frame_id(frame_id)
        frame_name = f"{video_id}_{frame_id}" if (video_id and frame_id) else None

        frame_path = self._to_json_safe(row.get(cols["frame_path"])) if cols["frame_path"] else None
        timestamp = self._to_json_safe(row.get(cols["timestamp"])) if cols["timestamp"] else None
        time_info = self._keyframe_time_for(video_id, frame_id, frame_submission_id, frame_name, frame_path)
        if time_info:
            timestamp = time_info["timestamp"]
        elif timestamp is None:
            fallback_frame_numbers = self._candidate_frame_numbers(frame_id, frame_submission_id, frame_path)
            if fallback_frame_numbers:
                timestamp = fallback_frame_numbers[0] / 25.0

        result = {
            "rank": rank,
            "score": score,
            "vector_id": vector_id,
            "faiss_id": vector_id,
            "global_frame_id": frame_submission_id,
            "frame_idx": frame_submission_id,
            "video_id": video_id,
            "frame_id": frame_id,
            "frame_name": frame_name,
            "frame_path": frame_path,
            "timestamp": timestamp,
            "namespace": self._to_json_safe(row.get(cols["namespace"])) if cols["namespace"] else None,
        }
        if time_info:
            result["fps"] = time_info["fps"]
            result["source_frame_idx"] = time_info["source_frame_idx"]
            result["keyframe_number"] = time_info["keyframe_number"]
            result["timestamp_source"] = time_info.get("timestamp_source")
            if time_info.get("timestamp_matched_frame_idx") is not None:
                result["timestamp_matched_frame_idx"] = time_info.get("timestamp_matched_frame_idx")
            if time_info.get("timestamp_frame_idx_delta") is not None:
                result["timestamp_frame_idx_delta"] = time_info.get("timestamp_frame_idx_delta")
        if self._video_meta_by_id is not None and video_id in self._video_meta_by_id:
            video_row = {
                k: self._to_json_safe(v) for k, v in self._video_meta_by_id[video_id].items()
            }
            result["video_metadata"] = video_row

        media_info = self._media_info_by_id.get(str(video_id))
        if media_info:
            safe_media_info = {k: self._to_json_safe(v) for k, v in media_info.items()}
            watch_url = safe_media_info.get("watch_url")
            thumbnail_url = safe_media_info.get("thumbnail_url")
            result["media_info"] = safe_media_info
            if watch_url:
                result["watch_url"] = watch_url
                result["youtube_url"] = watch_url
                result["video_url"] = watch_url
                result["link"] = watch_url
            if thumbnail_url:
                result["video_thumbnail_url"] = thumbnail_url
        return result

    @staticmethod
    def _normalize_submission_frame_id(frame_id: Any) -> Any:
        """Return the per-video frame id expected by submission CSVs.

        FAISS vector ids are corpus-global ids. The benchmark submission format
        expects the frame id inside each video, which is stored in metadata as
        values like "003048".
        """
        if frame_id is None:
            return None
        text = str(frame_id).strip()
        if not text:
            return None
        return int(text) if text.isdigit() else text

    def search_visual(self, query: str, top_k: int = 20) -> list[dict]:
        """Encode `query` with BEiT3 and search the exact FAISS index.

        Returns a list of dicts with real FAISS inner-product scores
        (never rank-derived placeholders), ordered by descending score.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise BEiT3RetrieverError(f"top_k must be a positive integer, got {top_k!r}.")

        query_vec = self.encode_text(query)
        scores, ids = self._index.search(query_vec, top_k)
        scores = scores[0]
        ids = ids[0]

        results: list[dict] = []
        rank = 0
        for score, vector_id in zip(scores, ids):
            if vector_id == -1:
                continue
            row = self._id_to_row.get(int(vector_id))
            if row is None:
                logger.error(
                    "FAISS returned vector_id=%s with no matching row in global_ids.parquet.",
                    vector_id,
                )
                continue
            rank += 1
            results.append(self._build_result(rank, float(score), int(vector_id), row))

        return results

    def search_by_vector_id(self, vector_id: int, top_k: int = 20) -> list[dict]:
        """Search similar keyframes using an existing vector_id in FAISS."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise BEiT3RetrieverError(f"top_k must be a positive integer, got {top_k!r}.")

        try:
            query_vec = self._index.reconstruct(int(vector_id)).reshape(1, -1)
        except Exception as exc:
            raise BEiT3RetrieverError(f"Cannot reconstruct vector for vector_id={vector_id}: {exc}") from exc

        scores, ids = self._index.search(query_vec, top_k)
        scores = scores[0]
        ids = ids[0]

        results: list[dict] = []
        rank = 0
        for score, vid in zip(scores, ids):
            if vid == -1:
                continue
            row = self._id_to_row.get(int(vid))
            if row is None:
                continue
            rank += 1
            results.append(self._build_result(rank, float(score), int(vid), row))

        return results

    def search_by_image(self, image: Any, top_k: int = 20) -> list[dict]:
        """Encode an image with the BEiT3 vision tower and search the FAISS index.

        This is the "find similar" path for a captured frame: the query is the
        exact server-extracted still, and every hit carries a real FAISS vector
        id. The captured frame's own per-video ``frame_idx`` is never treated as
        a global vector id -- that is what ``search_by_vector_id`` would require,
        and doing so returns matches for an unrelated corpus frame.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise BEiT3RetrieverError(f"top_k must be a positive integer, got {top_k!r}.")

        query_vec = self.encode_image(image)
        scores, ids = self._index.search(query_vec, top_k)
        scores = scores[0]
        ids = ids[0]

        results: list[dict] = []
        rank = 0
        for score, vector_id in zip(scores, ids):
            if vector_id == -1:
                continue
            row = self._id_to_row.get(int(vector_id))
            if row is None:
                logger.error(
                    "FAISS returned vector_id=%s with no matching row in global_ids.parquet.",
                    vector_id,
                )
                continue
            rank += 1
            results.append(self._build_result(rank, float(score), int(vector_id), row))

        return results

    def get_frame_by_vector_id(self, vector_id: int) -> dict | None:
        """Return the exact metadata row for a BEiT3 vector ID."""
        try:
            vector_id = int(vector_id)
        except (TypeError, ValueError):
            return None
        row = self._id_to_row.get(vector_id)
        if row is None:
            return None
        return self._build_result(1, 1.0, vector_id, row)

    def get_nearest_frame(self, video_id: str, timestamp: float) -> dict | None:
        """Return the keyframe nearest a timestamp in one video."""
        rows = self._video_to_rows.get(str(video_id))
        if not rows:
            return None
        try:
            target = float(timestamp)
        except (TypeError, ValueError):
            return None
        best_result = None
        best_delta = float("inf")
        for row in rows:
            vector_id = int(row.get(self._columns["vector_id"]) or 0)
            result = self._build_result(1, 1.0, vector_id, row)
            result_timestamp = result.get("timestamp")
            if result_timestamp is None:
                continue
            delta = abs(float(result_timestamp) - target)
            if delta < best_delta:
                best_delta = delta
                best_result = result
        if best_result is not None:
            best_result["timestamp_delta"] = best_delta
        return best_result

    def _build_video_to_rows(self) -> dict[str, list[dict]]:
        """Index rows by video_id in chronological frame_id order."""
        video_col = self._columns.get("video_id") or "video_id"
        frame_col = self._columns.get("frame_id") or "frame_id"
        lookup: dict[str, list[dict]] = {}
        for row in self._id_to_row.values():
            v_id = str(row.get(video_col) or "").strip()
            if v_id:
                if v_id not in lookup:
                    lookup[v_id] = []
                lookup[v_id].append(row)

        for v_id in lookup:
            lookup[v_id].sort(
                key=lambda r: (
                    str(r.get(frame_col) or "").zfill(12),
                    int(r.get(self._columns["vector_id"]) or 0),
                )
            )
        return lookup

    def get_video_timeline(
        self, video_id: str, around_frame_id: str | None = None, limit: int = 60
    ) -> list[dict]:
        """Return chronological keyframes for a given video."""
        rows = self._video_to_rows.get(video_id)
        if not rows:
            v_lower = video_id.lower()
            for k, v in self._video_to_rows.items():
                if k.lower() == v_lower:
                    rows = v
                    break
        if not rows:
            return []

        frame_col = self._columns.get("frame_id") or "frame_id"
        if around_frame_id:
            target = str(around_frame_id).strip()
            target_clean = target.lstrip("0")
            target_numbers = set(self._candidate_frame_numbers(target))
            match_idx = -1
            for i, r in enumerate(rows):
                fid = str(r.get(frame_col) or "").strip()
                row_path = r.get(self._columns["frame_path"]) if self._columns.get("frame_path") else None
                row_numbers = set(self._candidate_frame_numbers(fid, row_path))
                if (
                    fid == target
                    or (target_clean and fid.lstrip("0") == target_clean)
                    or bool(target_numbers.intersection(row_numbers))
                ):
                    match_idx = i
                    break
            if match_idx != -1:
                half = limit // 2
                start = max(0, match_idx - half)
                end = min(len(rows), start + limit)
                start = max(0, end - limit)
                selected_rows = rows[start:end]
            else:
                selected_rows = rows[:limit]
        else:
            selected_rows = rows[:limit]

        results: list[dict] = []
        for rank, row in enumerate(selected_rows, start=1):
            vid = int(row.get(self._columns["vector_id"]) or 0)
            res = self._build_result(rank, 1.0, vid, row)
            results.append(res)

        return results




_retriever: BEiT3Retriever | None = None
_retriever_lock = threading.Lock()


def get_beit3_retriever() -> BEiT3Retriever:
    """Return the process-wide BEiT3Retriever singleton, loading it on first use."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = BEiT3Retriever()
    return _retriever
