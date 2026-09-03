"""Jina CLIP v2 text-to-image visual retrieval service.

Owns the cloud-primary retrieval path built for this migration:
    text (Vietnamese/English) via `encode_text(task="retrieval.query")`, or an
    image via `encode_image` -> normalized 1024-d query vector ->
    FAISS IndexIDMap2(IndexFlatIP) search -> global_ids.parquet lookup ->
    structured results. Serves textual KIS, grounded Q&A candidate retrieval,
    TRAKE per-event retrieval, the video timeline, and the image-similarity
    paths when RETRIEVAL_BACKEND=jina_clip_v2.

Jina CLIP v2 and BEiT3 are two independent embedding spaces built from two
independent FAISS indexes (see scripts/cloud/build_jina_index.py for the Jina
one and scripts/model_encoding/run_beit3_encoder.py-derived pipelines for the
BEiT3 one). Nothing in this module ever mixes a vector, a vector id, or a
frame path from one into the other.

An **immutable** commit revision (a git SHA -- never a branch/tag/'main') is
**mandatory in every environment**: from `JINA_MODEL_REVISION` or the
`model_revision` in `jina_index_meta.json`. The two are cross-checked at
construction and the loaded model's commit is verified against the pin before
the first query; any disagreement, or an unverifiable commit, is a hard
error. `JINA_MODEL_PATH` may be:

* an existing local directory -- loaded directly, no network. Its revision
  must be provable (a `jina_model_revision` sidecar file, or an HF snapshot
  dir named by its commit) and must match the pin.
* a repo id -- the exact pinned commit is fetched once via
  `huggingface_hub.snapshot_download` (`JINA_MODEL_AUTO_BOOTSTRAP`, default
  on, needs `JINA_LOCAL_FILES_ONLY=false`); no manually pre-populated cache.

Nothing here imports `torch`, `transformers` or `huggingface_hub` at module
scope, and the retriever singleton is only ever constructed from inside a
request handler / the startup warmer (see `get_jina_retriever`), so merely
importing this module -- or starting the app -- loads nothing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

EXPECTED_DIM = 1024

# ENV values that additionally require a complete jina_index_meta.json.
_RELEASE_ENVS = {"production", "prod", "release", "staging"}

# An immutable model pin: a git commit SHA (7-64 hex). Branch/tag names and
# moving refs are rejected -- Jina query embeddings must use the *exact*
# commit that produced the cloud FAISS index, in every environment.
_IMMUTABLE_REV_RE = re.compile(r"\A[0-9a-fA-F]{7,64}\Z")
_MOVING_REFS = {"main", "master", "head", "latest", "dev", "develop", "stable"}
_LOCAL_REVISION_SIDECARS = ("jina_model_revision", "jina_model_revision.txt", ".jina_model_revision")


def _revisions_match(a: str, b: str) -> bool:
    """True if two commit ids refer to the same commit, tolerating a short vs
    full SHA (one is a hex prefix of the other)."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)

# Canonical internal column contract. Two on-disk parquet schemas normalize to
# it (see `_normalize_global_ids`):
#   1. the canonical one produced by scripts/cloud/build_jina_index.py, and
#   2. the schema the Azure merge pipeline already publishes
#      (parent_namespace / timestamp-in-seconds / frame_path-as-key /
#       local_position, no asset_key / timestamp_ms / keyframe_ordinal columns).
_COL_VECTOR_ID = "vector_id"
_COL_SPLIT = "split"
_COL_VIDEO_ID = "video_id"
_COL_EMBEDDING_ROW = "embedding_row"
_COL_KEYFRAME_ORDINAL = "keyframe_ordinal"
_COL_TIMESTAMP_MS = "timestamp_ms"
_COL_ASSET_KEY = "asset_key"
_COL_FRAME_PATH = "frame_path"
_COL_SOURCE_FRAME_ID = "source_frame_id"
_COL_RAW_FRAME_ID = "raw_frame_id"  # e.g. "keyframe_0000" -- display/debug only

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


def _to_float(value: Any) -> float:
    """NaN/None/pd.NA -> 0.0; used only for order-of-magnitude sort tie-breaks
    (e.g. embedding row), never for a timestamp and never for output."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


def _to_float_or_none(value: Any) -> float | None:
    """Nullable numeric parser for timestamp-sensitive code.

    Unlike :func:`_to_float`, a missing / non-finite value stays missing
    (``None``) instead of collapsing to ``0.0`` -- an untimestamped keyframe
    must never look like a zero-second keyframe.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _normalize_global_ids(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Return a DataFrame with the canonical internal columns.

    Accepts either the canonical `build_jina_index.py` schema or the schema
    the Azure merge notebook publishes today, so a member can sync the
    already-published `global_ids.parquet` with no rebuild.
    """
    cols = set(df.columns)
    if _COL_ASSET_KEY in cols and _COL_TIMESTAMP_MS in cols and _COL_SPLIT in cols:
        return df  # already canonical

    merge_schema = {"parent_namespace", "video_id", "frame_path", "timestamp", "local_position", "vector_id"}
    if merge_schema.issubset(cols):
        out = pd.DataFrame(
            {
                _COL_VECTOR_ID: df["vector_id"].astype("int64"),
                _COL_SPLIT: df["parent_namespace"].astype(str),
                _COL_VIDEO_ID: df["video_id"].astype(str),
                _COL_EMBEDDING_ROW: df["local_position"].astype("int64"),
                _COL_KEYFRAME_ORDINAL: df["local_position"].astype("int64") + 1,
                # seconds -> ms, kept as a nullable float so a missing
                # timestamp stays missing (never invented as 0).
                _COL_TIMESTAMP_MS: pd.to_numeric(df["timestamp"], errors="coerce") * 1000.0,
                _COL_ASSET_KEY: df["frame_path"].astype(str),
                _COL_FRAME_PATH: df["frame_path"].astype(str),
                _COL_SOURCE_FRAME_ID: (
                    pd.to_numeric(df["source_frame_idx"], errors="coerce")
                    if "source_frame_idx" in cols
                    else None
                ),
                _COL_RAW_FRAME_ID: df["frame_id"].astype(str) if "frame_id" in cols else None,
            }
        )
        return out

    raise JinaRetrieverError(
        f"jina global_ids parquet at {source} has an unrecognized schema "
        f"(columns={sorted(cols)}). Expected the canonical build_jina_index.py "
        f"columns or the Azure merge schema (parent_namespace, timestamp, "
        f"local_position, frame_path, vector_id)."
    )


class JinaRetrieverError(RuntimeError):
    """Raised for any Jina CLIP v2 startup or search invariant violation.

    Callers must not catch this and silently fall back to another backend;
    that would silently return results from the wrong embedding space.
    """


def validate_immutable_model_revision(rev: str | None, where: str = "model revision") -> str:
    """Return ``rev`` stripped, or raise :class:`JinaRetrieverError`.

    The single definition of an acceptable Jina CLIP v2 model pin, shared by
    the runtime retriever, the index builder, and the local smoke-test helper
    so all three agree on it: a bare git commit SHA (7-64 hex chars). An empty
    value, a placeholder like ``smoke-test-unpinned``, or a moving ref such as
    ``main`` / ``latest`` is rejected -- query embeddings must come from the
    exact commit that produced the published cloud FAISS index.
    """
    r = (rev or "").strip()
    if not r:
        raise JinaRetrieverError(
            f"{where} is missing. Set it to the exact git commit SHA the cloud "
            f"Jina FAISS index was embedded with."
        )
    if r.lower() in _MOVING_REFS or not _IMMUTABLE_REV_RE.match(r):
        raise JinaRetrieverError(
            f"{where}={rev!r} is not an immutable commit revision. Jina query "
            f"embeddings must use the exact git commit SHA that produced the "
            f"cloud FAISS index -- not a branch, tag, or moving ref like 'main'."
        )
    return r


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
        # Resolved *before* the consistency checks: a missing / non-immutable
        # pin, or a model/index revision disagreement, must fail construction
        # -- never surface at first query. Always a validated commit SHA.
        self._expected_model_revision: str = self._resolve_expected_model_revision()
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
        """Resolve ``JINA_DEVICE`` to a concrete torch device string.

        * ``auto`` (default) / unset -> ``cuda`` when a CUDA GPU is available,
          otherwise ``cpu``.
        * ``cuda`` -> ``cuda`` if available, else ``cpu`` with a warning
          (an explicit request that can't be honoured must not be a hard error).
        * ``cpu`` -> ``cpu``.
        """
        requested_norm = (requested or "auto").strip().lower()
        if requested_norm not in ("auto", "cuda", "cpu"):
            raise JinaRetrieverError(
                f"Invalid JINA_DEVICE={requested!r}; expected 'auto', 'cuda' or 'cpu'."
            )
        if requested_norm == "cpu":
            return "cpu"

        import torch

        cuda_ok = torch.cuda.is_available()
        if requested_norm == "auto":
            resolved = "cuda" if cuda_ok else "cpu"
            logger.info("JINA_DEVICE=auto -> %s (cuda_available=%s)", resolved, cuda_ok)
            return resolved
        # requested_norm == "cuda"
        if not cuda_ok:
            logger.warning(
                "JINA_DEVICE=cuda was requested but CUDA is not available; "
                "falling back to CPU."
            )
            return "cpu"
        return "cuda"

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
            raise JinaRetrieverError(f"jina global_ids parquet at {resolved} is empty.")
        df = _normalize_global_ids(df, resolved)
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise JinaRetrieverError(
                f"jina global_ids parquet at {resolved} is missing required columns after "
                f"normalization: {missing}."
            )
        if not df[_COL_VECTOR_ID].is_unique:
            raise JinaRetrieverError(f"jina global_ids parquet at {resolved} has duplicate vector_id values.")
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

    def _is_release_mode(self) -> bool:
        return str(getattr(self._settings, "env", "") or "").strip().lower() in _RELEASE_ENVS

    def _meta_get(self, *keys: str) -> Any:
        meta = self._index_meta or {}
        for key in keys:
            if key in meta and meta[key] not in (None, ""):
                return meta[key]
        return None

    @staticmethod
    def _validate_immutable_revision(rev: str, where: str) -> str:
        return validate_immutable_model_revision(rev, where)

    def _resolve_expected_model_revision(self) -> str:
        """The exact Jina CLIP v2 commit the query encoder must resolve to.

        Precedence: ``JINA_MODEL_REVISION`` (explicit operator pin) then the
        ``model_revision`` stamped into ``jina_index_meta.json`` at build time.
        Both are validated to be immutable commit SHAs; if both are set and
        disagree, that is a hard build/config mismatch. A resolvable pin is
        **mandatory in every environment** -- there is no unpinned dev path.
        """
        configured = (getattr(self._settings, "jina_model_revision", None) or "").strip() or None
        meta_rev = self._meta_get("model_revision", "model_commit", "revision")
        meta_rev = (str(meta_rev).strip() or None) if meta_rev is not None else None

        if configured is not None:
            configured = self._validate_immutable_revision(configured, "JINA_MODEL_REVISION")
        if meta_rev is not None:
            meta_rev = self._validate_immutable_revision(
                meta_rev, "jina_index_meta.json model_revision"
            )
        if configured and meta_rev and not _revisions_match(configured, meta_rev):
            raise JinaRetrieverError(
                f"Jina model revision mismatch: JINA_MODEL_REVISION={configured!r} but "
                f"jina_index_meta.json model_revision={meta_rev!r}. The FAISS index was "
                f"built with a different checkpoint than the one pinned for querying."
            )
        expected = configured or meta_rev
        if expected is None:
            raise JinaRetrieverError(
                "A pinned Jina model revision is required for "
                "RETRIEVAL_BACKEND=jina_clip_v2: set JINA_MODEL_REVISION to the exact "
                "commit SHA the cloud index was embedded with, or publish "
                "`model_revision` in jina_index_meta.json. A published "
                "jina_index_meta.json that lacks it must be rebuilt/republished "
                "before deployment."
            )
        return expected

    def _validate_consistency(self) -> None:
        self._validate_index_semantics()
        self._validate_id_map_matches_parquet()
        self._validate_index_meta()

    def _validate_index_semantics(self) -> None:
        """The index must be an ``IndexIDMap2`` -- it is searched by, and
        reconstructed from, explicit ``vector_id`` values (see
        ``search_by_vector_id`` / ``_search``). A plain ``IndexFlat*`` returns
        positional ids and cannot ``reconstruct`` by external id, which would
        silently mis-map every hit."""
        type_name = type(self._index).__name__
        if type_name != "IndexIDMap2":
            raise JinaRetrieverError(
                f"Jina FAISS index type is {type_name!r}, expected 'IndexIDMap2'. "
                f"The retriever searches and reconstructs by explicit vector_id."
            )
        try:
            first_id = int(next(iter(self._global_ids[_COL_VECTOR_ID])))
            self._index.reconstruct(first_id)
        except StopIteration:
            raise JinaRetrieverError("jina global_ids parquet has no rows.") from None
        except Exception as exc:  # noqa: BLE001
            raise JinaRetrieverError(
                f"Jina FAISS index does not support reconstruct-by-id "
                f"(IndexIDMap2 semantics): {type(exc).__name__}: {exc}"
            ) from exc

    def _validate_id_map_matches_parquet(self) -> None:
        """Compare the *set* of FAISS ids against parquet ``vector_id`` -- not
        just the counts. Equal counts with disjoint id sets means the index and
        the metadata describe different corpora."""
        import faiss

        try:
            id_array = faiss.vector_to_array(self._index.id_map)
        except Exception as exc:  # noqa: BLE001
            raise JinaRetrieverError(
                f"Could not read the Jina FAISS id map for validation: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        index_ids = {int(x) for x in id_array}
        parquet_ids = {int(x) for x in self._global_ids[_COL_VECTOR_ID]}
        if len(id_array) != len(self._global_ids):
            raise JinaRetrieverError(
                f"Jina FAISS id map has {len(id_array)} ids but "
                f"jina_global_ids.parquet has {len(self._global_ids)} rows."
            )
        if index_ids != parquet_ids:
            only_index = sorted(index_ids - parquet_ids)[:5]
            only_parquet = sorted(parquet_ids - index_ids)[:5]
            raise JinaRetrieverError(
                "Jina FAISS id map and jina_global_ids.parquet vector_id set differ "
                f"(same count). ids only in index: {only_index}; "
                f"ids only in parquet: {only_parquet}."
            )

    def _validate_index_meta(self) -> None:
        ntotal = self._index.ntotal
        if self._index_meta is None:
            if self._is_release_mode():
                raise JinaRetrieverError(
                    "jina_index_meta.json is required when ENV is a release "
                    "environment (backend / dimension / vector count / model "
                    "revision cannot be verified without it)."
                )
            logger.warning(
                "jina_index_meta.json is absent; skipping backend/dimension/"
                "metric/model-revision cross-checks (dev only)."
            )
            return

        backend = self._meta_get("backend")
        if backend is not None and str(backend) != self.backend_id:
            raise JinaRetrieverError(
                f"jina_index_meta.json backend={backend!r}, expected {self.backend_id!r}."
            )

        dim = self._meta_get("dimension", "embedding_dim", "dim")
        if dim is not None and int(dim) != self._index.d:
            raise JinaRetrieverError(
                f"jina_index_meta.json dimension={dim} != FAISS index d={self._index.d}."
            )
        if int(getattr(self._settings, "jina_truncate_dim", EXPECTED_DIM)) != self._index.d:
            raise JinaRetrieverError(
                f"JINA_TRUNCATE_DIM={self._settings.jina_truncate_dim} != FAISS index "
                f"d={self._index.d}; query vectors would not match the corpus."
            )

        count = self._meta_get("vector_count", "ntotal", "num_vectors")
        if count is not None and int(count) != ntotal:
            raise JinaRetrieverError(
                f"jina_index_meta.json vector_count={count} != FAISS index ntotal={ntotal}."
            )

        metric = str(self._meta_get("metric") or "").strip().lower()
        if metric and not any(t in metric for t in ("inner_product", "ip", "cosine", "dot")):
            raise JinaRetrieverError(
                f"jina_index_meta.json metric={metric!r} is not an inner-product / "
                f"cosine metric; the retriever searches IndexFlatIP on L2-normalized vectors."
            )
        norm = str(self._meta_get("normalization", "norm") or "").strip().lower()
        if norm and norm not in ("l2", "l2norm", "unit", "cosine"):
            raise JinaRetrieverError(
                f"jina_index_meta.json normalization={norm!r}; expected 'l2'."
            )
        # `_resolve_expected_model_revision` (run before this) has already
        # guaranteed a validated, immutable pin exists and agrees with any
        # `model_revision` in this file.

    def _build_video_to_rows(self) -> dict[str, list[dict]]:
        lookup: dict[str, list[dict]] = {}
        for row in self._id_to_row.values():
            video_id = str(row.get(_COL_VIDEO_ID) or "").strip()
            if video_id:
                lookup.setdefault(video_id, []).append(row)
        for video_id in lookup:
            # Rows with a missing timestamp sort *after* every timestamped row
            # (never before a 0.0s frame), then by embedding row for stability.
            lookup[video_id].sort(key=self._chrono_sort_key)
        return lookup

    @staticmethod
    def _chrono_sort_key(row: dict) -> tuple[int, float, float]:
        ts = _to_float_or_none(row.get(_COL_TIMESTAMP_MS))
        return (1 if ts is None else 0, ts if ts is not None else 0.0, _to_float(row.get(_COL_EMBEDDING_ROW)))

    # ------------------------------------------------------------------
    # Model provisioning (deterministic, pinned)
    # ------------------------------------------------------------------

    def _model_source(self) -> str:
        return self._settings.jina_model_path or "jinaai/jina-clip-v2"

    def _is_local_model_dir(self) -> bool:
        """``JINA_MODEL_PATH`` points at an existing directory (not a repo id)."""
        try:
            return Path(self._model_source()).is_dir()
        except OSError:
            return False

    def _auto_bootstrap_allowed(self) -> bool:
        return bool(getattr(self._settings, "jina_model_auto_bootstrap", True)) and not bool(
            self._settings.jina_local_files_only
        )

    def _derive_local_revision(self, model_dir: Path) -> str | None:
        """Durable commit id for a local model directory, or ``None``.

        Order: a ``jina_model_revision`` sidecar file we define, then the
        HuggingFace snapshot layout (``.../snapshots/<commit-sha>/``) where the
        directory *is* named by its commit. Never guessed.
        """
        for sidecar in _LOCAL_REVISION_SIDECARS:
            p = model_dir / sidecar
            try:
                if p.is_file():
                    raw = p.read_text(encoding="utf-8").strip()
                    if raw:
                        return self._validate_immutable_revision(raw, str(p))
            except OSError:
                pass
        name = model_dir.name.strip()
        if name.lower() not in _MOVING_REFS and _IMMUTABLE_REV_RE.match(name):
            return name
        return None

    def _resolve_local_model_dir(self) -> tuple[Path, str]:
        """Validate a local ``JINA_MODEL_PATH`` and return ``(dir, revision)``,
        raising if its revision cannot be proven or does not match the pin."""
        model_dir = Path(self._model_source())
        expected = getattr(self, "_expected_model_revision", None)
        local_rev = self._derive_local_revision(model_dir)
        if local_rev is None:
            raise JinaRetrieverError(
                f"JINA_MODEL_PATH={model_dir} is a local directory but its model "
                f"revision cannot be proven. Write the exact commit SHA to "
                f"'{model_dir / _LOCAL_REVISION_SIDECARS[0]}', or point JINA_MODEL_PATH "
                f"at a HuggingFace snapshot directory named by its commit."
            )
        if expected and not _revisions_match(local_rev, expected):
            raise JinaRetrieverError(
                f"Local Jina model at {model_dir} is revision {local_rev!r} but the "
                f"index/config pin is {expected!r}. Point JINA_MODEL_PATH at the "
                f"matching snapshot or fix the pin."
            )
        return model_dir, local_rev

    def _ensure_model_snapshot(self, *, provision: bool) -> Path:
        """Return the local snapshot directory for the pinned revision of a
        *remote* repo id, downloading it via ``huggingface_hub.snapshot_download``
        when ``provision`` is set and auto-bootstrap is allowed. Raises a clear
        :class:`JinaRetrieverError` otherwise -- never a bare stack trace.
        """
        expected = getattr(self, "_expected_model_revision", None)
        source = self._model_source()
        if not expected:  # defensive: construction already guarantees a pin
            raise JinaRetrieverError("Jina model revision is not pinned.")

        allow_download = provision and self._auto_bootstrap_allowed()
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise JinaRetrieverError(
                "huggingface_hub is required to provision / verify the pinned Jina "
                "CLIP v2 model but is not installed."
            ) from exc

        try:
            path = snapshot_download(source, revision=expected, local_files_only=not allow_download)
        except Exception as exc:  # noqa: BLE001 - normalized into a clear typed error
            if allow_download:
                raise JinaRetrieverError(
                    f"Could not download the pinned Jina CLIP v2 model {source}@{expected}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            raise JinaRetrieverError(
                f"Jina CLIP v2 model {source}@{expected} is not present locally and "
                f"downloading is disabled (JINA_LOCAL_FILES_ONLY=true / "
                f"JINA_MODEL_AUTO_BOOTSTRAP=false). Provision it once with "
                f"JINA_LOCAL_FILES_ONLY=false, or pre-place the snapshot."
            ) from exc
        return Path(path)

    def ensure_model_ready(self, *, provision: bool = False) -> None:
        """Raise :class:`JinaRetrieverError` with a clear message if the query
        encoder cannot be made available. Never loads weights. Safe to call
        from a readiness probe (``provision=False``) or the startup warmer
        (``provision=True``)."""
        if self._model is not None:
            return
        if not getattr(self, "_expected_model_revision", None):
            raise JinaRetrieverError("Jina model revision is not pinned.")
        if self._is_local_model_dir():
            self._resolve_local_model_dir()  # raises on unprovable / mismatched rev
            return
        self._ensure_model_snapshot(provision=provision)

    def readiness(self) -> dict[str, Any]:
        """Structured, non-throwing readiness snapshot for the settings API."""
        state = {
            "backend": self.backend_id,
            "index_loaded": self._index is not None,
            "rows": len(self._global_ids),
            "model_loaded": self._model is not None,
            "expected_model_revision": getattr(self, "_expected_model_revision", None),
            "model_source": self._model_source(),
            "model_source_is_local_dir": self._is_local_model_dir(),
        }
        if self._model is not None:
            state["ready"] = True
            return state
        try:
            self.ensure_model_ready(provision=False)
        except JinaRetrieverError as exc:
            # A remote pinned model that just is not on disk yet is "preparing"
            # (auto-bootstrap will fetch it); everything else -- a local dir
            # with no/wrong revision, bootstrap disabled -- is a real error.
            can_bootstrap = not self._is_local_model_dir() and self._auto_bootstrap_allowed()
            state.update(
                ready=False,
                state="preparing" if can_bootstrap else "error",
                detail=str(exc),
            )
            return state
        state["ready"] = True
        return state

    def warm_model(self) -> None:
        """Explicit model warm for the startup path -- actually loads weights
        (bootstrapping the pinned snapshot first if allowed)."""
        self._load_model()

    def _verify_resolved_commit(self, resolved: str | None) -> None:
        """Fail closed unless the loaded model's commit is proven to match the
        pin. A missing / unreadable commit is a failure, never a warning."""
        expected = getattr(self, "_expected_model_revision", None)
        if not expected:
            raise JinaRetrieverError("Jina model revision is not pinned.")
        if not resolved:
            raise JinaRetrieverError(
                f"Could not determine the loaded Jina CLIP v2 model's commit to verify "
                f"it against the pinned revision {expected!r}. Refusing to serve queries "
                f"with an unverifiable model."
            )
        if not _revisions_match(str(resolved), str(expected)):
            raise JinaRetrieverError(
                f"Loaded Jina CLIP v2 commit {resolved!r} != pinned revision "
                f"{expected!r} (JINA_MODEL_REVISION / jina_index_meta.json). Query "
                f"vectors would not match the FAISS index."
            )
        logger.info("Jina CLIP v2 model commit verified against the pin: %s", resolved)

    def _load_model(self) -> Any:
        """Load the Jina CLIP v2 model. Only ever called lazily, from inside
        `encode_text` / `encode_image` / `warm_model`, never at import or
        construction time.

        * ``JINA_MODEL_PATH`` = an existing directory -> load from it directly
          (no snapshot download); its durable revision is checked against the
          pin first.
        * ``JINA_MODEL_PATH`` = a repo id -> provision the exact pinned commit
          via ``huggingface_hub.snapshot_download`` and load that snapshot.

        Either way the loaded commit is verified against the pin before the
        model is usable (:meth:`_verify_resolved_commit`).
        """
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            from transformers import AutoModel

            expected = getattr(self, "_expected_model_revision", None)
            if not expected:
                raise JinaRetrieverError(
                    "Refusing to load the Jina CLIP v2 model without a pinned commit "
                    "revision (JINA_MODEL_REVISION / jina_index_meta.json)."
                )

            if self._is_local_model_dir():
                model_dir, local_rev = self._resolve_local_model_dir()
                logger.info("Loading Jina CLIP v2 from local dir %s (revision %s)", model_dir, local_rev)
                model = AutoModel.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                    trust_remote_code=bool(self._settings.jina_trust_remote_code),
                )
                resolved = getattr(getattr(model, "config", None), "_commit_hash", None) or local_rev
            else:
                snapshot = self._ensure_model_snapshot(provision=True)
                logger.info(
                    "Loading Jina CLIP v2 from snapshot %s (%s@%s)",
                    snapshot, self._model_source(), expected,
                )
                model = AutoModel.from_pretrained(
                    str(snapshot),
                    local_files_only=True,
                    trust_remote_code=bool(self._settings.jina_trust_remote_code),
                )
                resolved = (
                    getattr(getattr(model, "config", None), "_commit_hash", None)
                    or (snapshot.name if _IMMUTABLE_REV_RE.match(snapshot.name) else None)
                )

            self._verify_resolved_commit(resolved)
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
        return self._finalize_query_vector(vec)

    def _finalize_query_vector(self, vec: Any) -> np.ndarray:
        """Common tail for encode_text / encode_image: -> contiguous,
        L2-normalized (1, 1024) float32, validated."""
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().numpy()
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        vec = np.ascontiguousarray(vec, dtype=np.float32)
        self._validate_query_vector(vec)
        return vec

    def encode_image(self, image: Any) -> np.ndarray:
        """Encode an image query into a normalized, contiguous (1, 1024)
        float32 vector using Jina CLIP v2's own image preprocessing.

        ``image`` may be a filesystem path, a file-like object, or a PIL
        ``Image``. The keyframe corpus was embedded with
        ``model.encode_image(..., truncate_dim=1024)`` (see
        scripts/notebooks/embed-jina-upload-azure-5jobs-disk-safe.ipynb), so
        the query goes through the exact same call -- no manual resize /
        normalize, which Jina's remote code applies internally.
        """
        from PIL import Image

        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            try:
                pil = Image.open(image).convert("RGB")
            except (OSError, ValueError) as exc:
                raise JinaRetrieverError(f"Could not read the query image: {exc}") from exc

        model = self._load_model()
        import torch

        with torch.inference_mode():
            vec = model.encode_image([pil], truncate_dim=int(self._settings.jina_truncate_dim))
        return self._finalize_query_vector(vec)

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
        if isinstance(source_frame_id, float):
            source_frame_id = int(source_frame_id)
        keyframe_ordinal = self._to_json_safe(row.get(_COL_KEYFRAME_ORDINAL))
        raw_frame_id = self._to_json_safe(row.get(_COL_RAW_FRAME_ID))
        timestamp_ms = self._to_json_safe(row.get(_COL_TIMESTAMP_MS))
        if isinstance(timestamp_ms, float):
            timestamp_ms = int(round(timestamp_ms))
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
            "frame_id": raw_frame_id if raw_frame_id is not None else frame_idx,
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

        return self._search(self.encode_text(query), top_k)

    def _search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
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
                    "Jina FAISS returned vector_id=%s with no matching row in jina global_ids.parquet.",
                    vector_id,
                )
                continue
            rank += 1
            results.append(self._build_result(rank, float(score), int(vector_id), row))
        return results

    def search_by_image(self, image: Any, top_k: int = 20) -> list[dict]:
        """Encode an image with Jina CLIP v2's vision tower and search the same
        FAISS index. Every hit carries a real Jina vector id; the query
        image's own per-video frame index is never treated as a vector id.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise JinaRetrieverError(f"top_k must be a positive integer, got {top_k!r}.")
        return self._search(self.encode_image(image), top_k)

    def search_by_vector_id(self, vector_id: int, top_k: int = 20) -> list[dict]:
        """Find keyframes similar to an existing Jina vector id by
        reconstructing its vector from this index and searching.

        The id must be a Jina-space vector id (from this retriever's own
        results) -- a vector id from another backend would reconstruct an
        unrelated vector here.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise JinaRetrieverError(f"top_k must be a positive integer, got {top_k!r}.")
        try:
            query_vec = self._index.reconstruct(int(vector_id)).reshape(1, -1)
        except Exception as exc:  # noqa: BLE001
            raise JinaRetrieverError(
                f"Cannot reconstruct vector for vector_id={vector_id}: {exc}"
            ) from exc
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        return self._search(query_vec, top_k)

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
            row_ts_ms = _to_float_or_none(row.get(_COL_TIMESTAMP_MS))
            if row_ts_ms is None:
                continue  # untimestamped frame -- never a 0-second evidence match
            delta = abs(row_ts_ms - target_ms)
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
            target = str(around_frame_id).strip()
            target_num = target.lstrip("0") or "0"
            match_idx = -1
            for i, row in enumerate(rows):
                candidates = set()
                for col in (_COL_RAW_FRAME_ID, _COL_SOURCE_FRAME_ID, _COL_KEYFRAME_ORDINAL, _COL_EMBEDDING_ROW):
                    value = row.get(col)
                    if value is None or (isinstance(value, float) and not np.isfinite(value)):
                        continue
                    text = str(int(value)) if isinstance(value, float) else str(value)
                    candidates.add(text)
                    candidates.add(text.lstrip("0") or "0")
                if target in candidates or target_num in candidates:
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
