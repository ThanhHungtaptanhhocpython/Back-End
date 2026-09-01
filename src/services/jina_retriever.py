"""Jina CLIP v2 text-to-image visual retrieval service.

Owns the cloud-primary retrieval path built for this migration:
    Vietnamese/English text query -> Jina CLIP v2 `encode_text(task="retrieval.query")`
    -> normalized 1024-d query vector -> FAISS IndexIDMap2(IndexFlatIP) search ->
    jina_global_ids.parquet lookup -> structured results.

Jina CLIP v2 and BEiT3 are two independent embedding spaces built from two
independent FAISS indexes (see scripts/cloud/build_jina_index.py for the Jina
one and scripts/model_encoding/run_beit3_encoder.py-derived pipelines for the
BEiT3 one). Nothing in this module ever mixes a vector, a vector id, or a
frame path from one into the other.

The model is loaded from a **local, pinned** snapshot -- `JINA_MODEL_REVISION`
must be set and `JINA_LOCAL_FILES_ONLY` (default True) forces transformers to
resolve it from the already-downloaded local HuggingFace cache instead of
reaching the network at request time. Nothing here imports `torch` or
`transformers` at module scope, and the retriever singleton is only ever
constructed from inside a request handler (see `get_jina_retriever`), so
merely importing this module -- or starting the app -- loads nothing.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

EXPECTED_DIM = 1024

# The exact column contract produced by scripts/cloud/build_jina_index.py.
# Unlike BEiT3's global_ids.parquet (which has to tolerate several legacy
# schemas), this schema is entirely ours, so it is not auto-detected.
_COL_VECTOR_ID = "vector_id"
_COL_SPLIT = "split"
_COL_VIDEO_ID = "video_id"
_COL_EMBEDDING_ROW = "embedding_row"
_COL_KEYFRAME_ORDINAL = "keyframe_ordinal"
_COL_TIMESTAMP_MS = "timestamp_ms"
_COL_ASSET_KEY = "asset_key"
_COL_FRAME_PATH = "frame_path"
_COL_SOURCE_FRAME_ID = "source_frame_id"

_REQUIRED_COLUMNS = (
    _COL_VECTOR_ID,
    _COL_SPLIT,
    _COL_VIDEO_ID,
    _COL_EMBEDDING_ROW,
    _COL_KEYFRAME_ORDINAL,
    _COL_TIMESTAMP_MS,
    _COL_ASSET_KEY,
    _COL_FRAME_PATH,
)


class JinaRetrieverError(RuntimeError):
    """Raised for any Jina CLIP v2 startup or search invariant violation.

    Callers must not catch this and silently fall back to another backend;
    that would silently return results from the wrong embedding space.
    """


class JinaRetriever:
    """Owns the Jina CLIP v2 model, FAISS index, and parquet metadata."""

    backend_id = "jina_clip_v2"

    @staticmethod
    def _artifact(name: str, configured: Path | None) -> Path | None:
        """Prefer a synced cloud artifact for ``name`` (checksum-verified,
        current manifest version); fall back to the configured local path."""
        try:
            from src.services.assets import resolve_artifact_path

            synced = resolve_artifact_path(name)
            if synced is not None and synced.is_file():
                logger.info("Jina %s: using synced cloud artifact %s", name, synced)
                return synced
        except Exception:  # noqa: BLE001 - cloud assets must never block local mode
            pass
        return configured

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()

        self._index = self._load_faiss_index(
            self._artifact("jina_faiss_index", self._settings.jina_faiss_index_path)
        )
        self._global_ids = self._load_global_ids(
            self._artifact("jina_global_ids", self._settings.jina_global_ids_path)
        )
        self._index_meta = self._load_optional_json(
            self._artifact("jina_index_meta", self._settings.jina_index_meta_path)
        )
        self._validate_consistency()

        self._id_to_row: dict[int, dict] = {
            int(row[_COL_VECTOR_ID]): row for row in self._global_ids.to_dict(orient="records")
        }
        self._video_to_rows: dict[str, list[dict]] = self._build_video_to_rows()

        self._device = self._resolve_device(self._settings.jina_device)
        self._model = None  # loaded on first encode_text() call
        self._model_lock = threading.Lock()

        logger.info(
            "JinaRetriever ready: device=%s ntotal=%d rows=%d videos=%d",
            self._device,
            self._index.ntotal,
            len(self._global_ids),
            len(self._video_to_rows),
        )

    # ------------------------------------------------------------------
    # Startup / loading
    # ------------------------------------------------------------------

    def _resolve_device(self, requested: str | None) -> str:
        requested_norm = (requested or "cpu").strip().lower()
        if requested_norm not in ("cuda", "cpu"):
            raise JinaRetrieverError(
                f"Invalid JINA_DEVICE={requested!r}; expected 'cuda' or 'cpu'."
            )
        if requested_norm == "cuda":
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "JINA_DEVICE=cuda was requested but CUDA is not available; "
                    "falling back to CPU."
                )
                return "cpu"
        return requested_norm

    def _require_path(self, path: Path | None, env_var: str) -> Path:
        if path is None:
            raise JinaRetrieverError(f"{env_var} is not set.")
        resolved = Path(path)
        if not resolved.exists():
            raise JinaRetrieverError(f"{env_var} points to a missing file: {resolved}")
        return resolved

    def _load_faiss_index(self, index_path: Path | None):
        import faiss

        resolved = self._require_path(index_path, "JINA_FAISS_INDEX_PATH")
        index = faiss.read_index(str(resolved))
        if index.d != EXPECTED_DIM:
            raise JinaRetrieverError(
                f"Jina FAISS index dimension {index.d} != {EXPECTED_DIM} (path={resolved})."
            )
        return index

    def _load_global_ids(self, path: Path | None) -> pd.DataFrame:
        resolved = self._require_path(path, "JINA_GLOBAL_IDS_PATH")
        df = pd.read_parquet(resolved)
        if df.empty:
            raise JinaRetrieverError(f"jina_global_ids.parquet at {resolved} is empty.")
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise JinaRetrieverError(
                f"jina_global_ids.parquet at {resolved} is missing required columns: {missing}."
            )
        if not df[_COL_VECTOR_ID].is_unique:
            raise JinaRetrieverError(f"jina_global_ids.parquet at {resolved} has duplicate vector_id values.")
        return df

    def _load_optional_json(self, path: Path | None) -> dict | None:
        if path is None:
            return None
        resolved = self._require_path(path, "path for jina_index_meta.json")
        with resolved.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as exc:
                raise JinaRetrieverError(f"Failed to parse jina_index_meta.json at {resolved}: {exc}") from exc

    def _validate_consistency(self) -> None:
        ntotal = self._index.ntotal
        n_rows = len(self._global_ids)
        if ntotal != n_rows:
            raise JinaRetrieverError(
                f"Jina FAISS index ntotal={ntotal} does not match jina_global_ids.parquet "
                f"row count={n_rows}. The index and metadata are out of sync."
            )

    def _build_video_to_rows(self) -> dict[str, list[dict]]:
        lookup: dict[str, list[dict]] = {}
        for row in self._id_to_row.values():
            video_id = str(row.get(_COL_VIDEO_ID) or "").strip()
            if video_id:
                lookup.setdefault(video_id, []).append(row)
        for video_id in lookup:
            lookup[video_id].sort(key=lambda r: (r.get(_COL_TIMESTAMP_MS) or 0, r.get(_COL_EMBEDDING_ROW) or 0))
        return lookup

    def _load_model(self) -> Any:
        """Load the pinned Jina CLIP v2 model. Only ever called lazily, from
        inside `encode_text`, never at import or construction time."""
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            import torch
            from transformers import AutoModel

            revision = (self._settings.jina_model_revision or "").strip()
            if not revision:
                raise JinaRetrieverError(
                    "JINA_MODEL_REVISION is not set. The model must be pinned to an "
                    "exact revision so it can never silently drift to a newer, "
                    "differently-trained checkpoint than the one the FAISS index "
                    "was built with."
                )
            source = self._settings.jina_model_path or "jinaai/jina-clip-v2"
            logger.info(
                "Loading Jina CLIP v2 from %s@%s (local_files_only=%s)",
                source,
                revision,
                self._settings.jina_local_files_only,
            )
            model = AutoModel.from_pretrained(
                source,
                revision=revision,
                trust_remote_code=bool(self._settings.jina_trust_remote_code),
                local_files_only=bool(self._settings.jina_local_files_only),
            )
            model = model.eval().to(self._device)
            self._model = model
            return self._model

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _validate_query_vector(self, vec: np.ndarray) -> None:
        if vec.shape != (1, EXPECTED_DIM):
            raise JinaRetrieverError(
                f"Query embedding has shape {vec.shape}, expected (1, {EXPECTED_DIM})."
            )
        if vec.dtype != np.float32:
            raise JinaRetrieverError(f"Query embedding dtype={vec.dtype}, expected float32.")
        if not vec.flags["C_CONTIGUOUS"]:
            raise JinaRetrieverError("Query embedding must be C-contiguous for FAISS search.")
        if not np.isfinite(vec).all():
            raise JinaRetrieverError("Query embedding contains NaN/Inf values.")
        norm = float(np.linalg.norm(vec))
        if not np.isclose(norm, 1.0, atol=1e-2):
            raise JinaRetrieverError(f"Query embedding L2 norm={norm:.4f}, expected ~1.0.")

    def encode_text(self, query: str) -> np.ndarray:
        """Encode `query` into a normalized, contiguous (1, 1024) float32 vector.

        The query is encoded directly -- Vietnamese or English -- via Jina CLIP
        v2's official `encode_text(..., task="retrieval.query")` interface; no
        translation step is required or applied here.
        """
        if not query or not query.strip():
            raise JinaRetrieverError("Query text must be a non-empty string.")

        model = self._load_model()
        import torch

        with torch.inference_mode():
            vec = model.encode_text(
                [query.strip()],
                task="retrieval.query",
                truncate_dim=int(self._settings.jina_truncate_dim),
            )
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().numpy()
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        vec = np.ascontiguousarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = np.ascontiguousarray(vec / norm, dtype=np.float32)
        self._validate_query_vector(vec)
        return vec

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
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
        video_id = self._to_json_safe(row.get(_COL_VIDEO_ID))
        split = self._to_json_safe(row.get(_COL_SPLIT))
        asset_key = self._to_json_safe(row.get(_COL_ASSET_KEY))
        frame_path = self._to_json_safe(row.get(_COL_FRAME_PATH)) or asset_key
        source_frame_id = self._to_json_safe(row.get(_COL_SOURCE_FRAME_ID))
        keyframe_ordinal = self._to_json_safe(row.get(_COL_KEYFRAME_ORDINAL))
        timestamp_ms = self._to_json_safe(row.get(_COL_TIMESTAMP_MS))
        timestamp = (timestamp_ms / 1000.0) if timestamp_ms is not None else None

        # `frame_idx` is the per-video submission id: prefer the real source
        # video frame index when the build captured it, else the keyframe
        # ordinal. Never a FAISS vector id -- that stays in vector_id/faiss_id.
        frame_idx = source_frame_id if source_frame_id is not None else keyframe_ordinal
        frame_name = Path(str(asset_key)).name if asset_key else None

        return {
            "rank": rank,
            "score": score,
            "vector_id": vector_id,
            "faiss_id": vector_id,
            "global_frame_id": frame_idx,
            "frame_idx": frame_idx,
            "frame_id": source_frame_id if source_frame_id is not None else keyframe_ordinal,
            "video_id": video_id,
            "split": split,
            "namespace": split,
            "asset_key": asset_key,
            "frame_path": frame_path,
            "frame_name": frame_name,
            "keyframe_ordinal": keyframe_ordinal,
            "timestamp": timestamp,
            "timestamp_ms": timestamp_ms,
            "retrieval_backend": self.backend_id,
        }

    def search_visual(self, query: str, top_k: int = 20) -> list[dict]:
        """Encode `query` with Jina CLIP v2 and search the exact FAISS index.

        Returns real FAISS inner-product scores (never rank-derived
        placeholders), ordered by descending score.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise JinaRetrieverError(f"top_k must be a positive integer, got {top_k!r}.")

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
                    "Jina FAISS returned vector_id=%s with no matching row in jina_global_ids.parquet.",
                    vector_id,
                )
                continue
            rank += 1
            results.append(self._build_result(rank, float(score), int(vector_id), row))
        return results

    def get_frame_by_vector_id(self, vector_id: int) -> dict | None:
        """Return the exact metadata row for a Jina vector ID.

        Callers must only pass a Jina-space vector id here -- one obtained
        from this retriever's own search_visual/get_video_timeline results.
        A BEiT3 vector id happening to also exist as an integer key in this
        lookup would silently return an unrelated frame, so callers that
        source ids from another backend must route around this method (see
        `grounded_qa_service._candidate_from_evidence`, gated on backend_id).
        """
        try:
            vector_id = int(vector_id)
        except (TypeError, ValueError):
            return None
        row = self._id_to_row.get(vector_id)
        if row is None:
            return None
        return self._build_result(1, 1.0, vector_id, row)

    def get_nearest_frame(self, video_id: str, timestamp: float) -> dict | None:
        """Return the keyframe nearest a timestamp (seconds) in one video."""
        rows = self._video_to_rows.get(str(video_id))
        if not rows:
            return None
        try:
            target_ms = float(timestamp) * 1000.0
        except (TypeError, ValueError):
            return None
        best_result = None
        best_delta = float("inf")
        for row in rows:
            row_ts_ms = row.get(_COL_TIMESTAMP_MS)
            if row_ts_ms is None:
                continue
            delta = abs(float(row_ts_ms) - target_ms)
            if delta < best_delta:
                best_delta = delta
                best_result = row
        if best_result is None:
            return None
        vector_id = int(best_result.get(_COL_VECTOR_ID))
        result = self._build_result(1, 1.0, vector_id, best_result)
        result["timestamp_delta"] = best_delta / 1000.0
        return result

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

        if around_frame_id:
            target = str(around_frame_id).strip().lstrip("0") or "0"
            match_idx = -1
            for i, row in enumerate(rows):
                candidate = str(row.get(_COL_SOURCE_FRAME_ID) or row.get(_COL_KEYFRAME_ORDINAL) or "").lstrip("0") or "0"
                if candidate == target:
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
            vector_id = int(row.get(_COL_VECTOR_ID) or 0)
            results.append(self._build_result(rank, 1.0, vector_id, row))
        return results


_retriever: JinaRetriever | None = None
_retriever_lock = threading.Lock()


def get_jina_retriever() -> JinaRetriever:
    """Return the process-wide JinaRetriever singleton, loading it on first use.

    Nothing in this module runs at import time; the FAISS index / parquet
    load, and the Jina CLIP v2 model load, only happen the first time a
    request handler actually calls this function.
    """
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = JinaRetriever()
    return _retriever


def reset_jina_retriever() -> None:
    """Test/ops hook: drop the cached singleton so the next call reloads it."""
    global _retriever
    with _retriever_lock:
        _retriever = None
