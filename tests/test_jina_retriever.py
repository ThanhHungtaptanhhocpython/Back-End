"""Unit tests for the Jina CLIP v2 retrieval service (src/services/jina_retriever.py).

These tests exercise FAISS search/lookup wiring and the lazy-loading contract
WITHOUT the real Jina CLIP v2 model or a production FAISS index. Each search
test builds a bare `JinaRetriever` via `__new__` and hand-wires only the
attributes it exercises, bypassing `__init__` (which requires the real
artifacts), matching the pattern used by tests/test_beit3_retriever.py.

Run with:
    python -m pytest tests/test_jina_retriever.py -v
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import faiss

from src.config.settings import Settings
from src.services.jina_retriever import (
    EXPECTED_DIM,
    JinaRetriever,
    JinaRetrieverError,
    _normalize_global_ids,
)


def _bare_retriever(settings: Settings | None = None) -> JinaRetriever:
    """Build a JinaRetriever instance without running __init__."""
    obj = JinaRetriever.__new__(JinaRetriever)
    obj._settings = settings or Settings(debug=False)
    obj._device = "cpu"
    obj._model = None
    return obj


def _synthetic_index_and_rows(seed: int, ids: list[int], video_id: str = "L21_V001"):
    dim = EXPECTED_DIM
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(len(ids), dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    index.add_with_ids(vectors, np.array(ids, dtype=np.int64))

    df = pd.DataFrame(
        {
            "vector_id": ids,
            "split": ["L21"] * len(ids),
            "video_id": [video_id] * len(ids),
            "embedding_row": list(range(len(ids))),
            "keyframe_ordinal": [i + 1 for i in range(len(ids))],
            "timestamp_ms": [float(i) * 1000.0 for i in range(len(ids))],
            "asset_key": [f"L21/{video_id}/keyframe_{i:04d}.jpg" for i in range(len(ids))],
            "frame_path": [f"L21/{video_id}/keyframe_{i:04d}.jpg" for i in range(len(ids))],
            "source_frame_id": [i * 8 for i in range(len(ids))],
        }
    )
    return index, df, vectors


class SearchVisualIntegrationTests(unittest.TestCase):
    """Exercises FAISS search + metadata lookup with a synthetic index.

    The Jina CLIP v2 forward pass itself is stubbed out (via `encode_text`
    monkeypatch) since it requires the real model; everything downstream of
    the query vector is real.
    """

    def setUp(self):
        ids = [100, 200, 300, 400, 500]
        self.index, self.df, self.vectors = _synthetic_index_and_rows(0, ids)
        self.query_vec = self.vectors[2:3].copy()  # exact match for id=300

        r = _bare_retriever()
        r._index = self.index
        r._global_ids = self.df
        r._id_to_row = {int(row["vector_id"]): row for row in self.df.to_dict(orient="records")}
        r._video_to_rows = r._build_video_to_rows()
        r.encode_text = lambda query: self.query_vec  # bypass the real Jina forward pass
        self.retriever = r

    def test_search_visual_returns_real_scores_and_correct_vector_id_mapping(self):
        results = self.retriever.search_visual("một người đang đi xe máy", top_k=3)
        self.assertEqual(len(results), 3)

        top = results[0]
        self.assertEqual(top["rank"], 1)
        self.assertEqual(top["vector_id"], 300)
        self.assertEqual(top["faiss_id"], 300)
        self.assertAlmostEqual(top["score"], 1.0, places=4)  # exact match -> IP == 1.0
        self.assertEqual(top["video_id"], "L21_V001")
        self.assertEqual(top["asset_key"], "L21/L21_V001/keyframe_0002.jpg")
        self.assertEqual(top["frame_path"], top["asset_key"])
        self.assertEqual(top["timestamp"], 2.0)  # 2000ms -> 2.0s
        self.assertEqual(top["retrieval_backend"], "jina_clip_v2")

        # Scores must be non-increasing (real FAISS ranking, not rank-derived).
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_visual_never_returns_a_beit3_style_frame_path(self):
        results = self.retriever.search_visual("query", top_k=5)
        for row in results:
            self.assertTrue(row["asset_key"].endswith(".jpg"))
            self.assertFalse(row["asset_key"].endswith(".webp"))

    def test_search_visual_rejects_invalid_top_k(self):
        with self.assertRaises(JinaRetrieverError):
            self.retriever.search_visual("x", top_k=0)
        with self.assertRaises(JinaRetrieverError):
            self.retriever.search_visual("x", top_k=-5)

    def test_get_frame_by_vector_id(self):
        frame = self.retriever.get_frame_by_vector_id(400)
        self.assertIsNotNone(frame)
        self.assertEqual(frame["vector_id"], 400)
        self.assertIsNone(self.retriever.get_frame_by_vector_id(999999))

    def test_get_nearest_frame(self):
        frame = self.retriever.get_nearest_frame("L21_V001", 1.9)
        self.assertIsNotNone(frame)
        self.assertEqual(frame["timestamp"], 2.0)
        self.assertIsNone(self.retriever.get_nearest_frame("no_such_video", 1.0))

    def test_get_video_timeline_orders_chronologically(self):
        timeline = self.retriever.get_video_timeline("L21_V001", limit=10)
        self.assertEqual(len(timeline), 5)
        timestamps = [row["timestamp"] for row in timeline]
        self.assertEqual(timestamps, sorted(timestamps))


class MergeSchemaNormalizationTests(unittest.TestCase):
    """The retriever must consume the parquet schema the Azure merge pipeline
    already publishes (parent_namespace / timestamp-in-seconds / frame_path /
    local_position), so a member can sync it with no rebuild."""

    def _published_df(self):
        # Column layout + a real row, taken verbatim from the live Azure
        # `indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina/global_ids.parquet`.
        return pd.DataFrame(
            {
                "parent_namespace": ["L21_a", "L21_a", "L21_a"],
                "video_id": ["L21_V001", "L21_V001", "L21_V001"],
                "frame_id": ["keyframe_0000", "keyframe_0001", "keyframe_0002"],
                "frame_path": [
                    "L21_a/L21_V001/keyframe_0000.jpg",
                    "L21_a/L21_V001/keyframe_0001.jpg",
                    "L21_a/L21_V001/keyframe_0002.jpg",
                ],
                "timestamp": [0.133333, 0.3, 1.033333],
                "source_fps": [30.0, 30.0, 30.0],
                "source_frame_idx": [4, 9, 31],
                "local_position": [0, 1, 2],
                "vector_id": [0, 1, 2],
            }
        )

    def test_normalizes_published_schema_to_canonical(self):
        out = _normalize_global_ids(self._published_df(), Path("global_ids.parquet"))
        assert out["split"].tolist() == ["L21_a", "L21_a", "L21_a"]
        assert out["asset_key"].tolist()[0] == "L21_a/L21_V001/keyframe_0000.jpg"
        assert out["frame_path"].tolist()[0] == out["asset_key"].tolist()[0]
        assert out["embedding_row"].tolist() == [0, 1, 2]
        assert out["keyframe_ordinal"].tolist() == [1, 2, 3]
        assert round(out["timestamp_ms"].tolist()[0]) == 133  # 0.133333 s -> ms
        assert out["source_frame_id"].tolist() == [4, 9, 31]
        assert out["raw_frame_id"].tolist()[0] == "keyframe_0000"

    def test_canonical_schema_passes_through_unchanged(self):
        _idx, df, _v = _synthetic_index_and_rows(0, [10, 20])
        out = _normalize_global_ids(df, Path("x"))
        assert out is df

    def test_unknown_schema_raises(self):
        bad = pd.DataFrame({"vector_id": [0], "something_else": ["?"]})
        with self.assertRaises(JinaRetrieverError):
            _normalize_global_ids(bad, Path("x"))

    def test_search_over_published_schema_maps_ids_and_paths(self):
        published = self._published_df()
        norm = _normalize_global_ids(published, Path("x"))
        dim = EXPECTED_DIM
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(3, dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        index.add_with_ids(vectors, np.array([0, 1, 2], dtype=np.int64))

        r = _bare_retriever()
        r._index = index
        r._global_ids = norm
        r._id_to_row = {int(row["vector_id"]): row for row in norm.to_dict(orient="records")}
        r._video_to_rows = r._build_video_to_rows()
        r.encode_text = lambda q: vectors[1:2].copy()

        top = r.search_visual("người đàn ông", top_k=2)[0]
        assert top["vector_id"] == 1
        assert top["asset_key"] == "L21_a/L21_V001/keyframe_0001.jpg"
        assert top["frame_path"] == top["asset_key"]
        assert top["frame_id"] == "keyframe_0001"
        assert top["frame_idx"] == 9  # source_frame_idx, the submission id
        assert abs(top["timestamp"] - 0.3) < 1e-6
        assert top["split"] == "L21_a"


class ImageSearchTests(unittest.TestCase):
    """encode_image / search_by_image / search_by_vector_id: the Jina-backend
    image-similarity path used when RETRIEVAL_BACKEND=jina_clip_v2."""

    def setUp(self):
        ids = [100, 200, 300, 400]
        self.index, self.df, self.vectors = _synthetic_index_and_rows(7, ids)
        r = _bare_retriever()
        r._index = self.index
        r._global_ids = self.df
        r._id_to_row = {int(row["vector_id"]): row for row in self.df.to_dict(orient="records")}
        r._video_to_rows = r._build_video_to_rows()
        self.retriever = r
        match_vec = self.vectors[2:3].copy()  # exact match for id=300
        captured = {}

        class FakeModel:
            def encode_image(self, images, truncate_dim=None):
                captured["n_images"] = len(images)
                captured["truncate_dim"] = truncate_dim
                captured["mode"] = getattr(images[0], "mode", None)
                return match_vec

        r._model = FakeModel()
        r._model_lock = MagicMock()
        self.captured = captured

    @staticmethod
    def _img():
        from PIL import Image

        return Image.new("RGB", (9, 5), (10, 20, 30))

    def test_encode_image_uses_jina_interface_and_normalizes(self):
        vec = self.retriever.encode_image(self._img())
        self.assertEqual(self.captured["truncate_dim"], EXPECTED_DIM)
        self.assertEqual(self.captured["n_images"], 1)
        self.assertEqual(self.captured["mode"], "RGB")
        self.assertEqual(vec.shape, (1, EXPECTED_DIM))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(vec.flags["C_CONTIGUOUS"])
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_search_by_image_hits_the_faiss_index(self):
        results = self.retriever.search_by_image(self._img(), top_k=3)
        self.assertEqual(results[0]["vector_id"], 300)
        self.assertAlmostEqual(results[0]["score"], 1.0, places=4)
        self.assertEqual(results[0]["retrieval_backend"], "jina_clip_v2")

    def test_search_by_image_rejects_bad_top_k(self):
        with self.assertRaises(JinaRetrieverError):
            self.retriever.search_by_image(self._img(), top_k=0)

    def test_search_by_vector_id_reconstructs_from_this_index(self):
        results = self.retriever.search_by_vector_id(300, top_k=2)
        self.assertEqual(results[0]["vector_id"], 300)
        self.assertAlmostEqual(results[0]["score"], 1.0, places=4)

    def test_search_by_vector_id_unknown_id_raises(self):
        with self.assertRaises(JinaRetrieverError):
            self.retriever.search_by_vector_id(999999, top_k=2)


# Realistic immutable commit ids (hex, >= 7 chars).
_SHA_A = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
_SHA_B = "b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4"


def _good_meta(**over):
    meta = {
        "backend": "jina_clip_v2",
        "dimension": EXPECTED_DIM,
        "metric": "inner_product_on_l2_normalized_vectors",
        "normalization": "l2",
        "model_revision": _SHA_A,
    }
    meta.update(over)
    return meta


def _wire_for_validation(index, df, index_meta=None, settings=None):
    """Hand-wire only what `_validate_consistency` / model-revision resolution
    touch, then resolve the expected revision exactly as `__init__` does."""
    r = _bare_retriever(settings or Settings(debug=False))
    r._index = index
    r._global_ids = df
    r._index_meta = index_meta
    r._expected_model_revision = r._resolve_expected_model_revision()
    return r


class IndexMetadataModelConsistencyTests(unittest.TestCase):
    """Fix 5 -- construction-time validation must go beyond count-only equality:
    IndexIDMap2 semantics, FAISS id-set == parquet vector_id set, and the
    index_meta backend/dimension/metric/model-revision cross-checks."""

    def test_accepts_a_fully_consistent_index(self):
        index, df, _ = _synthetic_index_and_rows(0, [10, 20, 30])
        r = _wire_for_validation(
            index, df, _good_meta(vector_count=3),
            Settings(debug=False, jina_model_revision=_SHA_A),
        )
        r._validate_consistency()  # must not raise

    def test_wrong_index_type_is_rejected(self):
        _idx, df, vectors = _synthetic_index_and_rows(1, [10, 20, 30])
        plain = faiss.IndexFlatIP(EXPECTED_DIM)  # no IndexIDMap2 wrapper
        plain.add(vectors)
        r = _wire_for_validation(plain, df, _good_meta())
        with self.assertRaises(JinaRetrieverError) as ctx:
            r._validate_consistency()
        self.assertIn("IndexIDMap2", str(ctx.exception))

    def test_equal_count_but_different_id_sets_is_rejected(self):
        index, _df, _ = _synthetic_index_and_rows(2, [10, 20, 30])
        # Same row count (3), but vector_id 99 is not in the FAISS id map.
        df = pd.DataFrame(
            {
                "vector_id": [10, 20, 99],
                "split": ["L21"] * 3,
                "video_id": ["L21_V001"] * 3,
                "embedding_row": [0, 1, 2],
                "keyframe_ordinal": [1, 2, 3],
                "timestamp_ms": [0.0, 1000.0, 2000.0],
                "asset_key": [f"L21/L21_V001/keyframe_{i:04d}.jpg" for i in range(3)],
                "frame_path": [f"L21/L21_V001/keyframe_{i:04d}.jpg" for i in range(3)],
                "source_frame_id": [0, 8, 16],
            }
        )
        r = _wire_for_validation(index, df, _good_meta())
        with self.assertRaises(JinaRetrieverError) as ctx:
            r._validate_consistency()
        self.assertIn("vector_id set differ", str(ctx.exception))

    def test_wrong_model_revision_is_rejected_before_any_search(self):
        index, df, _ = _synthetic_index_and_rows(3, [1, 2, 3])
        # index_meta was built with one commit; the operator pinned another.
        meta = _good_meta(model_revision=_SHA_A)
        with self.assertRaises(JinaRetrieverError) as ctx:
            _wire_for_validation(
                index, df, meta,
                Settings(debug=False, jina_model_revision=_SHA_B),
            )
        self.assertIn("revision mismatch", str(ctx.exception))

    def test_index_meta_backend_mismatch_is_rejected(self):
        index, df, _ = _synthetic_index_and_rows(4, [1, 2, 3])
        r = _wire_for_validation(index, df, _good_meta(backend="beit3"))
        with self.assertRaises(JinaRetrieverError):
            r._validate_consistency()

    def test_index_meta_dimension_mismatch_is_rejected(self):
        index, df, _ = _synthetic_index_and_rows(5, [1, 2, 3])
        r = _wire_for_validation(index, df, _good_meta(dimension=768))
        with self.assertRaises(JinaRetrieverError):
            r._validate_consistency()

    def test_resolved_commit_mismatch_raises_on_model_load(self):
        r = _bare_retriever(Settings(debug=False, jina_model_revision=_SHA_A))
        r._expected_model_revision = _SHA_A
        import threading as _t

        r._model_lock = _t.Lock()
        fake_model = MagicMock()
        fake_model.config._commit_hash = _SHA_B  # the loaded weights are a different commit
        fake_tf = MagicMock()
        fake_tf.AutoModel.from_pretrained.return_value = fake_model
        fake_hub = MagicMock()
        fake_hub.snapshot_download.return_value = "/hf/snapshots/does-not-matter"
        old_tf = sys.modules.get("transformers")
        old_hub = sys.modules.get("huggingface_hub")
        sys.modules["transformers"] = fake_tf
        sys.modules["huggingface_hub"] = fake_hub
        try:
            with self.assertRaises(JinaRetrieverError) as ctx:
                r._load_model()
            self.assertIn("!= pinned revision", str(ctx.exception))
        finally:
            for name, old in (("transformers", old_tf), ("huggingface_hub", old_hub)):
                if old is not None:
                    sys.modules[name] = old
                else:
                    sys.modules.pop(name, None)

    def test_missing_resolved_commit_is_a_failure_not_a_warning(self):
        r = _bare_retriever(Settings(debug=False, jina_model_revision=_SHA_A))
        r._expected_model_revision = _SHA_A
        with self.assertRaises(JinaRetrieverError) as ctx:
            r._verify_resolved_commit(None)
        self.assertIn("Could not determine", str(ctx.exception))


class _HubStub:
    """Context-manager that installs a fake ``huggingface_hub`` module."""

    def __init__(self, snapshot=None, side_effect=None):
        self.calls = []
        self.mod = MagicMock()

        def _snap(source, revision=None, local_files_only=False):
            self.calls.append({"source": source, "revision": revision,
                               "local_files_only": local_files_only})
            if side_effect is not None:
                raise side_effect
            return snapshot or f"/hf/cache/snapshots/{revision}"

        self.mod.snapshot_download.side_effect = _snap

    def __enter__(self):
        self._old = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = self.mod
        return self

    def __exit__(self, *a):
        if self._old is not None:
            sys.modules["huggingface_hub"] = self._old
        else:
            sys.modules.pop("huggingface_hub", None)


class ModelProvisioningTests(unittest.TestCase):
    """A clean machine must have a deterministic, pinned way to obtain the
    model -- in every environment. No unpinned / moving-ref path."""

    def test_missing_revision_rejected_in_every_env(self):
        index, df, _ = _synthetic_index_and_rows(0, [1, 2])
        meta = _good_meta()
        meta.pop("model_revision")
        for env in ("production", "development"):
            with self.subTest(env=env):
                with self.assertRaises(JinaRetrieverError) as ctx:
                    _wire_for_validation(index, df, meta, Settings(debug=False, env=env))
                self.assertIn("pinned Jina model revision is required", str(ctx.exception))

    def test_moving_ref_revision_is_rejected(self):
        index, df, _ = _synthetic_index_and_rows(0, [1, 2])
        meta = _good_meta()
        meta.pop("model_revision")
        for bad in ("main", "master", "v2.0", "HEAD", "latest"):
            with self.subTest(rev=bad):
                with self.assertRaises(JinaRetrieverError) as ctx:
                    _wire_for_validation(
                        index, df, meta, Settings(debug=False, jina_model_revision=bad)
                    )
                self.assertIn("immutable commit revision", str(ctx.exception))

    def test_meta_revision_used_when_env_var_absent(self):
        index, df, _ = _synthetic_index_and_rows(0, [1, 2])
        r = _wire_for_validation(index, df, _good_meta(), Settings(debug=False))
        self.assertEqual(r._expected_model_revision, _SHA_A)

    def test_remote_pinned_source_downloads_the_exact_revision(self):
        r = _bare_retriever(
            Settings(debug=False, jina_model_revision=_SHA_A,
                     jina_local_files_only=False, jina_model_auto_bootstrap=True)
        )
        r._expected_model_revision = _SHA_A
        with _HubStub() as hub:
            r.ensure_model_ready(provision=True)
        self.assertEqual(hub.calls[0]["revision"], _SHA_A)
        self.assertEqual(hub.calls[0]["source"], "jinaai/jina-clip-v2")
        self.assertFalse(hub.calls[0]["local_files_only"])  # download allowed

    def test_remote_source_absent_and_locked_errors_clearly(self):
        r = _bare_retriever(
            Settings(debug=False, jina_model_revision=_SHA_A,
                     jina_local_files_only=True, jina_model_auto_bootstrap=False)
        )
        r._expected_model_revision = _SHA_A
        with _HubStub(side_effect=FileNotFoundError("not in cache")):
            with self.assertRaises(JinaRetrieverError) as ctx:
                r.ensure_model_ready(provision=False)
        self.assertIn("not present locally", str(ctx.exception))

    def test_readiness_reports_preparing_for_a_not_yet_downloaded_remote_model(self):
        r = _bare_retriever(
            Settings(debug=False, jina_model_revision=_SHA_A,
                     jina_local_files_only=False, jina_model_auto_bootstrap=True)
        )
        r._expected_model_revision = _SHA_A
        r._index = object()
        r._global_ids = [0, 1, 2]
        with _HubStub(side_effect=FileNotFoundError("not cached yet")):
            state = r.readiness()
        self.assertFalse(state["ready"])
        self.assertEqual(state["state"], "preparing")

    # -- local snapshot directory policy ---------------------------------

    def _local_snapshot(self, tmp: Path, revision: str, *, sidecar: bool) -> Path:
        d = tmp / (revision if not sidecar else "my-jina-model")
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        if sidecar:
            (d / "jina_model_revision").write_text(revision + "\n", encoding="utf-8")
        return d

    def test_local_dir_loads_directly_without_snapshot_download(self):
        import tempfile
        import threading as _t

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._local_snapshot(Path(tmp), _SHA_A, sidecar=True)
            r = _bare_retriever(
                Settings(debug=False, jina_model_revision=_SHA_A, jina_model_path=str(model_dir))
            )
            r._expected_model_revision = _SHA_A
            r._model_lock = _t.Lock()

            fake_model = MagicMock()
            fake_model.config._commit_hash = None  # local dir: transformers won't set it
            fake_tf = MagicMock()
            fake_tf.AutoModel.from_pretrained.return_value = fake_model
            old_tf = sys.modules.get("transformers")
            sys.modules["transformers"] = fake_tf
            try:
                with _HubStub() as hub:
                    r._load_model()
                self.assertEqual(hub.calls, [])  # snapshot_download NEVER called
                _args, kwargs = fake_tf.AutoModel.from_pretrained.call_args
                self.assertEqual(_args[0], str(model_dir))
                self.assertTrue(kwargs["local_files_only"])
            finally:
                if old_tf is not None:
                    sys.modules["transformers"] = old_tf
                else:
                    sys.modules.pop("transformers", None)

    def test_local_dir_revision_from_snapshot_folder_name(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._local_snapshot(Path(tmp), _SHA_A, sidecar=False)
            r = _bare_retriever(
                Settings(debug=False, jina_model_revision=_SHA_A, jina_model_path=str(model_dir))
            )
            r._expected_model_revision = _SHA_A
            got_dir, got_rev = r._resolve_local_model_dir()
            self.assertEqual(got_dir, model_dir)
            self.assertEqual(got_rev, _SHA_A)

    def test_local_dir_with_no_provable_revision_fails(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "some-model"
            model_dir.mkdir()
            r = _bare_retriever(
                Settings(debug=False, jina_model_revision=_SHA_A, jina_model_path=str(model_dir))
            )
            r._expected_model_revision = _SHA_A
            with self.assertRaises(JinaRetrieverError) as ctx:
                r.ensure_model_ready()
            self.assertIn("cannot be proven", str(ctx.exception))

    def test_local_dir_revision_mismatch_fails(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._local_snapshot(Path(tmp), _SHA_B, sidecar=True)
            r = _bare_retriever(
                Settings(debug=False, jina_model_revision=_SHA_A, jina_model_path=str(model_dir))
            )
            r._expected_model_revision = _SHA_A
            with self.assertRaises(JinaRetrieverError) as ctx:
                r.ensure_model_ready()
            self.assertIn("index/config pin", str(ctx.exception))


class MissingTimestampTests(unittest.TestCase):
    """Fix 6 -- a NaN / missing timestamp must never be treated as a
    zero-second frame, and must sort last."""

    def _retriever_with_nan_row(self):
        ids = [100, 200, 300]
        index, df, _ = _synthetic_index_and_rows(0, ids)
        # Row for id=100 has NO timestamp; the other two are at 5s and 6s.
        df.loc[df["vector_id"] == 100, "timestamp_ms"] = np.nan
        df.loc[df["vector_id"] == 200, "timestamp_ms"] = 5000.0
        df.loc[df["vector_id"] == 300, "timestamp_ms"] = 6000.0
        r = _bare_retriever()
        r._index = index
        r._global_ids = df
        r._id_to_row = {int(row["vector_id"]): row for row in df.to_dict(orient="records")}
        r._video_to_rows = r._build_video_to_rows()
        return r

    def test_nan_timestamp_is_not_selected_as_a_zero_second_frame(self):
        r = self._retriever_with_nan_row()
        near_zero = r.get_nearest_frame("L21_V001", 0.0)
        self.assertIsNotNone(near_zero)
        # The NaN-timestamp frame (id 100) must NOT win the 0-second query.
        self.assertNotEqual(near_zero["vector_id"], 100)
        self.assertEqual(near_zero["vector_id"], 200)  # nearest real timestamp (5s)

    def test_missing_timestamp_rows_sort_last(self):
        r = self._retriever_with_nan_row()
        order = [row["vector_id"] for row in r._video_to_rows["L21_V001"]]
        self.assertEqual(order[-1], 100)  # untimestamped row is last
        self.assertEqual(order[:2], [200, 300])

    def test_json_output_keeps_missing_timestamp_null(self):
        r = self._retriever_with_nan_row()
        frame = r.get_frame_by_vector_id(100)
        self.assertIsNone(frame["timestamp"])
        self.assertIsNone(frame["timestamp_ms"])


class QueryVectorValidationTests(unittest.TestCase):
    def test_rejects_wrong_shape(self):
        r = _bare_retriever()
        with self.assertRaises(JinaRetrieverError):
            r._validate_query_vector(np.zeros((1, 768), dtype=np.float32))

    def test_rejects_non_finite(self):
        r = _bare_retriever()
        vec = np.ones((1, EXPECTED_DIM), dtype=np.float32)
        vec[0, 0] = np.nan
        with self.assertRaises(JinaRetrieverError):
            r._validate_query_vector(vec)

    def test_rejects_unnormalized_vector(self):
        r = _bare_retriever()
        vec = np.ones((1, EXPECTED_DIM), dtype=np.float32)  # norm >> 1
        with self.assertRaises(JinaRetrieverError):
            r._validate_query_vector(vec)

    def test_rejects_non_float32_dtype(self):
        r = _bare_retriever()
        vec = np.zeros((1, EXPECTED_DIM), dtype=np.float64)
        vec[0, 0] = 1.0
        with self.assertRaises(JinaRetrieverError):
            r._validate_query_vector(vec)

    def test_rejects_non_contiguous_vector(self):
        r = _bare_retriever()
        base = np.zeros((1, EXPECTED_DIM * 2), dtype=np.float32)
        base[0, 0] = 1.0
        strided = base[:, ::2]  # non-contiguous view, still (1, EXPECTED_DIM)
        self.assertFalse(strided.flags["C_CONTIGUOUS"])
        with self.assertRaises(JinaRetrieverError):
            r._validate_query_vector(strided)

    def test_accepts_normalized_contiguous_float32_vector(self):
        r = _bare_retriever()
        vec = np.zeros((1, EXPECTED_DIM), dtype=np.float32)
        vec[0, 0] = 1.0
        r._validate_query_vector(vec)  # should not raise


class EncodeTextContractTests(unittest.TestCase):
    """`encode_text` must hand FAISS an L2-normalized, contiguous float32 row,
    and must use the official `encode_text(..., task="retrieval.query")`
    interface -- not some other call shape."""

    def test_encode_text_normalizes_contiguous_float32(self):
        r = _bare_retriever()
        raw = np.arange(EXPECTED_DIM, dtype=np.float64).reshape(1, -1) + 1.0  # unnormalized, wrong dtype

        captured = {}

        class FakeModel:
            def encode_text(self, texts, task=None, truncate_dim=None):
                captured["texts"] = texts
                captured["task"] = task
                captured["truncate_dim"] = truncate_dim
                return raw

        r._model = FakeModel()
        r._model_lock = MagicMock()
        r._model_lock.__enter__ = lambda *a: None
        r._model_lock.__exit__ = lambda *a: None

        vec = r.encode_text("một người đàn ông")
        self.assertEqual(captured["texts"], ["một người đàn ông"])
        self.assertEqual(captured["task"], "retrieval.query")
        self.assertEqual(vec.shape, (1, EXPECTED_DIM))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(vec.flags["C_CONTIGUOUS"])
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_encode_text_rejects_empty_query(self):
        r = _bare_retriever()
        with self.assertRaises(JinaRetrieverError):
            r.encode_text("")
        with self.assertRaises(JinaRetrieverError):
            r.encode_text("   ")


class LazyLoadingTests(unittest.TestCase):
    """Importing the module, or holding an un-constructed retriever, must
    never import torch/transformers or hit the network. The model is loaded
    only on the first real `encode_text` call, never at app startup."""

    def test_module_import_has_no_singleton_yet(self):
        import src.services.jina_retriever as jr

        self.assertIsNone(jr._retriever)

    def test_construction_does_not_load_the_model(self):
        r = _bare_retriever()
        self.assertIsNone(r._model)

    def test_load_model_refuses_without_a_pinned_revision(self):
        import threading

        # A retriever whose revision was never resolved (never reached through
        # __init__, which would have raised) must still refuse to load.
        r = _bare_retriever(Settings(debug=False, jina_model_revision=None, jina_local_files_only=False))
        r._model_lock = threading.Lock()

        fake_tf = MagicMock()
        fake_tf.AutoModel.from_pretrained.side_effect = AssertionError(
            "must not call from_pretrained without a pinned revision"
        )
        old = sys.modules.get("transformers")
        sys.modules["transformers"] = fake_tf
        try:
            with self.assertRaises(JinaRetrieverError) as ctx:
                r._load_model()
            self.assertIn("pinned commit revision", str(ctx.exception))
        finally:
            if old is not None:
                sys.modules["transformers"] = old
            else:
                sys.modules.pop("transformers", None)

    def test_reimporting_module_triggers_no_model_download(self):
        # Stub torch/transformers so that IF the module tried to eagerly load
        # anything at import time, this would blow up loudly.
        sentinel = MagicMock()
        sentinel.AutoModel.from_pretrained.side_effect = AssertionError(
            "module import must never call AutoModel.from_pretrained"
        )
        old_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = sentinel
        try:
            import src.services.jina_retriever as jr

            importlib.reload(jr)  # re-executes module body; must not raise
            self.assertIsNone(jr._retriever)
        finally:
            if old_transformers is not None:
                sys.modules["transformers"] = old_transformers
            else:
                sys.modules.pop("transformers", None)
            importlib.reload(jr)


class ValidateImmutableModelRevisionTests(unittest.TestCase):
    """The shared model-pin contract reused by the index builder and the
    scratch/jina_smoke_test.py helper -- it must match runtime behaviour.

    Symbols are pulled from the live module in each test: an earlier test in
    this file reloads ``src.services.jina_retriever``, which rebinds the
    module-level ``JinaRetrieverError`` class the file imported at load time.
    """

    def _mod(self):
        from src.services import jina_retriever as jr
        return jr

    def test_accepts_short_and_full_hex_shas(self):
        validate = self._mod().validate_immutable_model_revision
        for good in ("deadbee", _SHA_A, _SHA_B, "E10D47F5691D0454A0FB5D13F46F2199B74CB436"):
            self.assertEqual(validate(good, "x"), good.strip())

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(self._mod().validate_immutable_model_revision(f"  {_SHA_A}\n", "x"), _SHA_A)

    def test_rejects_empty_or_none_as_missing(self):
        jr = self._mod()
        for bad in (None, "", "   "):
            with self.assertRaises(jr.JinaRetrieverError) as ctx:
                jr.validate_immutable_model_revision(bad, "JINA_MODEL_REVISION")
            self.assertIn("missing", str(ctx.exception))

    def test_rejects_moving_refs(self):
        jr = self._mod()
        for bad in ("main", "master", "HEAD", "latest", "dev", "develop", "stable"):
            with self.assertRaises(jr.JinaRetrieverError) as ctx:
                jr.validate_immutable_model_revision(bad, "x")
            self.assertIn("immutable commit revision", str(ctx.exception))

    def test_rejects_placeholders_and_non_hex(self):
        jr = self._mod()
        for bad in ("smoke-test-unpinned", "v2.0", "jinaai/jina-clip-v2", "abcdefg", "z" * 12):
            with self.assertRaises(jr.JinaRetrieverError):
                jr.validate_immutable_model_revision(bad, "x")

    def test_matches_the_retriever_staticmethod(self):
        jr = self._mod()
        self.assertEqual(
            jr.JinaRetriever._validate_immutable_revision(_SHA_A, "x"),
            jr.validate_immutable_model_revision(_SHA_A, "x"),
        )


if __name__ == "__main__":
    unittest.main()
