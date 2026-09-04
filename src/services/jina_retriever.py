"""Jina CLIP v2 retrieval over the fine-keyframe FAISS corpus.

The image corpus was encoded with ``jinaai/jina-clip-v2`` and truncated to
1024 dimensions. Query text and uploaded/captured images must use that exact
embedding space; crossing it with the BEiT3 index is an invariant violation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch

from src.config.settings import Settings, get_settings
from src.services.beit3_retriever import BEiT3Retriever

logger = logging.getLogger(__name__)


@contextmanager
def _hf_offline_if(enabled: bool):
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    previous_constants: list[tuple[Any, str, Any]] = []
    if enabled:
        for key in keys:
            os.environ[key] = "1"
        for module_name, attr in (
            ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
            ("transformers.utils.hub", "_is_offline_mode"),
        ):
            try:
                module = __import__(module_name, fromlist=[attr])
                previous_constants.append((module, attr, getattr(module, attr)))
                setattr(module, attr, True)
            except Exception:
                pass
    try:
        yield
    finally:
        for module, attr, value in previous_constants:
            setattr(module, attr, value)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _cached_hf_snapshot(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    required_file: str | None = None,
) -> Path | None:
    cache_roots: list[Path] = []
    if cache_dir is not None:
        cache_roots.append(Path(cache_dir))
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    repo_dir_name = "models--" + repo_id.replace("/", "--")
    for root in cache_roots:
        repo_dir = root / repo_dir_name
        snapshots = repo_dir / "snapshots"
        if revision:
            direct = snapshots / revision
            if direct.is_dir() and (not required_file or (direct / required_file).is_file()):
                return direct
        ref = repo_dir / "refs" / "main"
        try:
            target = ref.read_text(encoding="utf-8").strip()
        except OSError:
            target = ""
        if target:
            resolved = snapshots / target
            if resolved.is_dir() and (not required_file or (resolved / required_file).is_file()):
                return resolved
        if snapshots.is_dir():
            for child in snapshots.iterdir():
                if child.is_dir() and (not required_file or (child / required_file).is_file()):
                    return child
    return None


class JinaRetrieverError(RuntimeError):
    """Raised when Jina model/index/metadata invariants are not satisfied."""


class JinaRetriever(BEiT3Retriever):
    """Own the Jina model, final FAISS index, and aligned parquet mappings.

    The BEiT3 parent supplies model-independent FAISS search and timeline
    helpers. Loading, encoding, consistency checks, and result construction are
    overridden so Jina's ``source_frame_idx`` remains the submission frame ID.
    """

    REQUIRED_COLUMNS = {
        "vector_id",
        "parent_namespace",
        "video_id",
        "frame_id",
        "frame_path",
        "timestamp",
        "source_fps",
        "source_frame_idx",
    }

    ARTIFACT_NAMES = {
        "index": "jina_faiss_index",
        "global_ids": "jina_global_ids",
        "video_metadata": "jina_video_metadata",
        "index_meta": "jina_index_meta",
    }

    @staticmethod
    def _artifact(name: str, configured: Path | None) -> Path | None:
        try:
            from src.services.assets import resolve_artifact_path

            synced = resolve_artifact_path(name)
            if synced is not None and synced.is_file():
                logger.info("Jina %s: using synced cloud artifact %s", name, synced)
                return synced
        except Exception:  # noqa: BLE001 - local mode must remain usable
            pass
        return configured

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._truncate_dim = int(self._settings.jina_truncate_dim)
        self._device = self._resolve_jina_device(self._settings.jina_device)
        self._encode_lock = threading.Lock()
        self._text_embedding_cache: dict[str, np.ndarray] = {}

        self._index = self._load_jina_index(
            self._artifact(self.ARTIFACT_NAMES["index"], self._settings.jina_faiss_index_path)
        )
        self._global_ids = self._load_parquet(
            self._artifact(self.ARTIFACT_NAMES["global_ids"], self._settings.jina_global_ids_path),
            "JINA_GLOBAL_IDS_PATH",
        )
        self._video_metadata = self._load_parquet(
            self._artifact(
                self.ARTIFACT_NAMES["video_metadata"],
                self._settings.jina_video_metadata_path,
            ),
            "JINA_VIDEO_METADATA_PATH",
        )
        self._index_meta = self._load_json(
            self._artifact(self.ARTIFACT_NAMES["index_meta"], self._settings.jina_index_meta_path),
            "JINA_INDEX_META_PATH",
        )

        self._validate_jina_consistency()
        self._columns = self._detect_jina_columns()
        self._id_to_row = self._build_id_lookup(self._global_ids, "vector_id")
        self._video_meta_by_id = self._build_video_metadata_lookup()
        self._video_to_rows = self._build_video_to_rows()
        self._keyframe_time_by_video = {}
        self._media_info_by_id = self._load_media_info_dir(self._settings.get_media_info_path())

        # Validate the cheap, local artifacts before a potentially large model
        # download. This makes configuration failures fast and deterministic.
        self._model = self._load_jina_model()

        logger.info(
            "JinaRetriever ready: model=%s revision=%s device=%s ntotal=%d dim=%d videos=%d",
            self._settings.jina_model_name_or_path,
            self.loaded_model_revision or self._settings.jina_model_revision or "unpinned",
            self._device,
            self._index.ntotal,
            self._index.d,
            len(self._video_to_rows),
        )

    @staticmethod
    def _require_file(path: Path | None, env_var: str) -> Path:
        if path is None:
            raise JinaRetrieverError(f"{env_var} is not set.")
        resolved = Path(path)
        if not resolved.is_file():
            raise JinaRetrieverError(f"{env_var} points to a missing file: {resolved}")
        return resolved

    @staticmethod
    def _resolve_jina_device(requested: str | None) -> torch.device:
        value = (requested or "cpu").strip().lower()
        if value not in {"cpu", "cuda"}:
            raise JinaRetrieverError(
                f"Invalid JINA_DEVICE={requested!r}; expected 'cpu' or 'cuda'."
            )
        if value == "cuda" and not torch.cuda.is_available():
            logger.warning("JINA_DEVICE=cuda requested without CUDA; falling back to CPU.")
            value = "cpu"
        return torch.device(value)

    def _load_jina_index(self, path: Path | None) -> faiss.Index:
        resolved = self._require_file(path, "JINA_FAISS_INDEX_PATH")
        try:
            index = faiss.read_index(str(resolved))
        except Exception as exc:
            raise JinaRetrieverError(f"Failed to read Jina FAISS index {resolved}: {exc}") from exc
        if index.d != self._truncate_dim:
            raise JinaRetrieverError(
                f"Jina FAISS dimension {index.d} != JINA_TRUNCATE_DIM={self._truncate_dim}."
            )
        return index

    def _load_parquet(self, path: Path | None, env_var: str) -> pd.DataFrame:
        resolved = self._require_file(path, env_var)
        try:
            frame = pd.read_parquet(resolved)
        except Exception as exc:
            raise JinaRetrieverError(f"Failed to read {env_var} at {resolved}: {exc}") from exc
        if frame.empty:
            raise JinaRetrieverError(f"{env_var} at {resolved} is empty.")
        return frame

    def _load_json(self, path: Path | None, env_var: str) -> dict[str, Any]:
        resolved = self._require_file(path, env_var)
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise JinaRetrieverError(f"Failed to read {env_var} at {resolved}: {exc}") from exc
        if not isinstance(value, dict):
            raise JinaRetrieverError(f"{env_var} must contain a JSON object.")
        return value

    def _detect_jina_columns(self) -> dict[str, str]:
        missing = sorted(self.REQUIRED_COLUMNS - set(self._global_ids.columns))
        if missing:
            raise JinaRetrieverError(f"Jina global_ids.parquet is missing columns: {missing}")
        return {
            "vector_id": "vector_id",
            "video_id": "video_id",
            "frame_id": "frame_id",
            "frame_path": "frame_path",
            "timestamp": "timestamp",
            "namespace": "parent_namespace",
        }

    def _validate_jina_consistency(self) -> None:
        rows = len(self._global_ids)
        if self._index.ntotal != rows:
            raise JinaRetrieverError(
                f"Jina FAISS ntotal={self._index.ntotal} != global_ids rows={rows}."
            )
        meta_dim = int(self._index_meta.get("embedding_dim", -1))
        meta_count = int(self._index_meta.get("vector_count", -1))
        if meta_dim != self._index.d or meta_count != rows:
            raise JinaRetrieverError(
                "Jina index_meta.json does not match FAISS/parquet "
                f"(meta_dim={meta_dim}, index_dim={self._index.d}, "
                f"meta_count={meta_count}, rows={rows})."
            )
        if str(self._index_meta.get("model") or "").lower() != "jina":
            raise JinaRetrieverError("JINA_INDEX_META_PATH is not a Jina index metadata file.")
        if self._index_meta.get("metric") != "inner_product_on_l2_normalized_vectors":
            raise JinaRetrieverError("Jina index metric/normalization contract is unsupported.")

        missing = sorted(self.REQUIRED_COLUMNS - set(self._global_ids.columns))
        if missing:
            raise JinaRetrieverError(f"Jina global_ids.parquet is missing columns: {missing}")
        vector_ids = self._global_ids["vector_id"]
        if vector_ids.duplicated().any() or vector_ids.tolist() != list(range(rows)):
            raise JinaRetrieverError("Jina vector_id values must be unique and contiguous from zero.")
        duplicate_keys = self._global_ids[
            ["parent_namespace", "video_id", "frame_path"]
        ].duplicated().sum()
        if duplicate_keys:
            raise JinaRetrieverError(f"Jina metadata contains {duplicate_keys} duplicate frame keys.")
        if self._global_ids[["timestamp", "source_fps", "source_frame_idx"]].isna().any().any():
            raise JinaRetrieverError("Jina timestamp/source-frame metadata contains null values.")
        if int(self._video_metadata["frame_count"].sum()) != rows:
            raise JinaRetrieverError("Jina video_metadata frame counts do not sum to vector_count.")

    def _load_jina_model(self) -> Any:
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise JinaRetrieverError("transformers is required for Jina retrieval.") from exc

        load_target = self._settings.jina_model_name_or_path
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": bool(self._settings.jina_local_files_only),
            # Jina's remote config still expects the legacy string form while
            # recent Transformers releases also accept it.
            "torch_dtype": "float16" if self._device.type == "cuda" else "float32",
        }
        if self._settings.jina_model_revision:
            kwargs["revision"] = self._settings.jina_model_revision
        if self._settings.jina_cache_dir:
            kwargs["cache_dir"] = str(self._settings.jina_cache_dir)
        if self._settings.jina_local_files_only:
            cached = _cached_hf_snapshot(
                self._settings.jina_model_name_or_path,
                revision=self._settings.jina_model_revision,
                cache_dir=self._settings.jina_cache_dir,
                required_file="config.json",
            )
            if cached is not None:
                load_target = str(cached)
                kwargs.pop("revision", None)
            try:
                with _hf_offline_if(True):
                    config = AutoConfig.from_pretrained(load_target, **kwargs)
                text_config = getattr(config, "text_config", None)
                if text_config is not None and getattr(text_config, "hf_model_name_or_path", "") == "jinaai/jina-embeddings-v3":
                    text_cached = _cached_hf_snapshot(
                        "jinaai/jina-embeddings-v3",
                        cache_dir=self._settings.jina_cache_dir,
                        required_file="config.json",
                    )
                    if text_cached is not None:
                        text_config.hf_model_name_or_path = str(text_cached)
                config._name_or_path = load_target
                kwargs["config"] = config
            except Exception as exc:
                raise JinaRetrieverError(
                    "Failed to resolve local Jina model snapshot. Check "
                    "JINA_CACHE_DIR, JINA_MODEL_REVISION, and that dependent "
                    f"HF snapshots are cached: {exc}"
                ) from exc
        try:
            with _hf_offline_if(bool(self._settings.jina_local_files_only)):
                model = AutoModel.from_pretrained(
                    load_target,
                    **kwargs,
                )
            if self._settings.jina_local_files_only:
                tokenizer_cached = _cached_hf_snapshot(
                    self._settings.jina_model_name_or_path,
                    revision=self._settings.jina_model_revision,
                    cache_dir=self._settings.jina_cache_dir,
                    required_file="tokenizer_config.json",
                )
                if tokenizer_cached is not None:
                    model.config._name_or_path = str(tokenizer_cached)
            return model.eval().to(self._device)
        except Exception as exc:
            raise JinaRetrieverError(
                "Failed to load Jina model. Check JINA_MODEL_NAME_OR_PATH, "
                "JINA_MODEL_REVISION, cache/network access, and model dependencies: "
                f"{exc}"
            ) from exc

    @property
    def loaded_model_revision(self) -> str | None:
        config = getattr(getattr(self, "_model", None), "config", None)
        value = getattr(config, "_commit_hash", None)
        return str(value) if value else None

    def _normalize_embeddings(self, value: Any, expected_rows: int) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (expected_rows, self._truncate_dim):
            raise JinaRetrieverError(
                f"Jina embedding shape {vector.shape} != ({expected_rows}, {self._truncate_dim})."
            )
        if not np.isfinite(vector).all():
            raise JinaRetrieverError("Jina embedding contains NaN/Inf values.")
        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise JinaRetrieverError("Jina returned a zero-norm embedding.")
        vector = (vector / norms).astype(np.float32, copy=False)
        return np.ascontiguousarray(vector)

    def _normalize_embedding(self, value: Any) -> np.ndarray:
        return self._normalize_embeddings(value, expected_rows=1)

    def encode_text(self, query: str) -> np.ndarray:
        return self.encode_text_batch([query])

    def encode_text_batch(self, queries: list[str]) -> np.ndarray:
        cleaned = [query.strip() for query in queries if isinstance(query, str) and query.strip()]
        if len(cleaned) != len(queries) or not cleaned:
            raise JinaRetrieverError("Query texts must be a non-empty list of non-empty strings.")

        cache = getattr(self, "_text_embedding_cache", None)
        if cache is None:
            cache = self._text_embedding_cache = {}
        missing = list(dict.fromkeys(query for query in cleaned if query not in cache))
        kwargs = {"truncate_dim": self._truncate_dim}
        task = self._settings.jina_query_task.strip()
        if task:
            kwargs["task"] = task
        if missing:
            with self._encode_lock, torch.inference_mode():
                value = self._model.encode_text(missing, **kwargs)
            encoded = self._normalize_embeddings(value, expected_rows=len(missing))
            for query, vector in zip(missing, encoded):
                cache[query] = vector.copy()
            while len(cache) > 256:
                cache.pop(next(iter(cache)))
        return np.ascontiguousarray(np.stack([cache[query] for query in cleaned]).astype(np.float32, copy=False))

    def search_visual_batch(self, queries: list[str], top_k: int = 20) -> list[list[dict]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise JinaRetrieverError(f"top_k must be a positive integer, got {top_k!r}.")
        vectors = self.encode_text_batch(queries)
        scores, ids = self._index.search(vectors, top_k)
        batches: list[list[dict]] = []
        for row_scores, row_ids in zip(scores, ids):
            results: list[dict] = []
            for score, vector_id in zip(row_scores, row_ids):
                if vector_id == -1:
                    continue
                row = self._id_to_row.get(int(vector_id))
                if row is None:
                    logger.error("FAISS returned vector_id=%s with no Jina metadata row.", vector_id)
                    continue
                results.append(self._build_result(len(results) + 1, float(score), int(vector_id), row))
            batches.append(results)
        return batches

    def _timeline_rows(self, video_id: str) -> list[dict]:
        clean_video_id = str(video_id or "").strip()
        if not clean_video_id:
            return []
        rows = self._video_to_rows.get(clean_video_id)
        if rows:
            return rows
        lowered = clean_video_id.lower()
        return next(
            (value for key, value in self._video_to_rows.items() if key.lower() == lowered),
            [],
        )

    def search_video_timelines(
        self,
        queries: list[str],
        video_ids: list[str],
        top_k: int = 20,
    ) -> dict[str, list[list[dict]]]:
        """Score several event queries over several local video timelines.

        Query text is encoded once as a batch. This is essential for TRAKE:
        anchoring twelve videos for four events must not invoke the text model
        forty-eight times on CPU.
        """
        if not queries or not video_ids:
            return {}
        query_vectors = self.encode_text_batch(queries)
        output: dict[str, list[list[dict]]] = {}
        for video_id in dict.fromkeys(str(item or "").strip() for item in video_ids):
            rows = self._timeline_rows(video_id)
            if not rows:
                output[video_id] = [[] for _ in queries]
                continue

            vector_ids: list[int] = []
            metadata_rows: list[dict] = []
            embeddings: list[np.ndarray] = []
            for row in rows:
                vector_id = int(row.get(self._columns["vector_id"]) or -1)
                if vector_id < 0:
                    continue
                try:
                    embeddings.append(self._index.reconstruct(vector_id))
                except Exception as exc:  # noqa: BLE001 - FAISS backend-specific error
                    logger.debug("Could not reconstruct Jina vector %s: %s", vector_id, exc)
                    continue
                vector_ids.append(vector_id)
                metadata_rows.append(row)
            if not embeddings:
                output[video_id] = [[] for _ in queries]
                continue

            matrix = np.asarray(embeddings, dtype=np.float32)
            batches: list[list[dict]] = []
            for query_vector in query_vectors:
                scores = matrix @ query_vector
                top_indices = np.argsort(-scores)[:max(1, int(top_k))]
                batches.append([
                    self._build_result(rank, float(scores[index]), vector_ids[index], metadata_rows[index])
                    for rank, index in enumerate(top_indices, start=1)
                ])
            output[video_id] = batches
        return output

    def search_video_timeline(self, query: str, video_id: str, top_k: int = 20) -> list[dict]:
        """Score every indexed keyframe in one video against ``query``."""
        return self.search_video_timelines([query], [video_id], top_k).get(str(video_id or "").strip(), [[]])[0]

    def encode_image(self, image: Any) -> np.ndarray:
        from PIL import Image

        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            try:
                pil = Image.open(image).convert("RGB")
            except (OSError, ValueError) as exc:
                raise JinaRetrieverError(f"Could not read the query image: {exc}") from exc
        with self._encode_lock, torch.inference_mode():
            value = self._model.encode_image([pil], truncate_dim=self._truncate_dim)
        return self._normalize_embedding(value)

    @staticmethod
    def _frame_number(frame_id: Any, frame_path: Any) -> int | None:
        for value in (frame_id, frame_path):
            match = re.search(r"(\d+)(?:\.[^.]+)?$", str(value or ""))
            if match:
                return int(match.group(1))
        return None

    def _build_result(self, rank: int, score: float, vector_id: int, row: dict) -> dict:
        video_id = self._to_json_safe(row.get("video_id"))
        frame_id = self._to_json_safe(row.get("frame_id"))
        frame_path = self._to_json_safe(row.get("frame_path"))
        namespace = self._to_json_safe(row.get("parent_namespace"))
        timestamp = self._to_json_safe(row.get("timestamp"))
        source_fps = self._to_json_safe(row.get("source_fps"))
        source_frame_idx = self._to_json_safe(row.get("source_frame_idx"))
        local_position = self._to_json_safe(row.get("local_position"))
        keyframe_number = (
            int(local_position) + 1
            if local_position is not None
            else ((self._frame_number(frame_id, frame_path) or 0) + 1)
        )

        result = {
            "rank": rank,
            "score": score,
            "vector_id": vector_id,
            "faiss_id": vector_id,
            "global_frame_id": source_frame_idx,
            "submission_frame_id": source_frame_idx,
            "frame_idx": source_frame_idx,
            "video_id": video_id,
            "frame_id": frame_id,
            "frame_name": Path(str(frame_path)).name if frame_path else frame_id,
            "frame_path": frame_path,
            "timestamp": timestamp,
            "namespace": namespace,
            "fps": source_fps,
            "source_frame_idx": source_frame_idx,
            "keyframe_number": keyframe_number,
            "timestamp_source": "jina_global_ids",
            "retriever": "jina",
        }
        if self._video_meta_by_id is not None and video_id in self._video_meta_by_id:
            result["video_metadata"] = {
                key: self._to_json_safe(value)
                for key, value in self._video_meta_by_id[video_id].items()
            }
        media_info = self._media_info_by_id.get(str(video_id))
        if media_info:
            safe_media = {key: self._to_json_safe(value) for key, value in media_info.items()}
            result["media_info"] = safe_media
            watch_url = safe_media.get("watch_url")
            if watch_url:
                result.update(
                    watch_url=watch_url,
                    youtube_url=watch_url,
                    video_url=watch_url,
                    link=watch_url,
                )
            if safe_media.get("thumbnail_url"):
                result["video_thumbnail_url"] = safe_media["thumbnail_url"]
        return result


_retriever: JinaRetriever | None = None
_retriever_lock = threading.Lock()


def get_jina_retriever() -> JinaRetriever:
    """Return the process-wide Jina retriever, loading it on first use."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = JinaRetriever()
    return _retriever


def reset_jina_retriever() -> None:
    """Clear the singleton after a runtime configuration restart/test."""
    global _retriever
    with _retriever_lock:
        _retriever = None
