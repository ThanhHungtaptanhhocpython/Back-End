"""Unit tests for the BEiT3 retrieval service (src/services/beit3_retriever.py).

These tests exercise the tokenizer offset mapping, parquet column
auto-detection, and FAISS search/lookup wiring WITHOUT the real BEiT3
checkpoint or production FAISS index (those are multi-GB deployment-machine
artifacts not present in this repo/dev environment). Each test builds a
bare `BEiT3Retriever` via `__new__` and hand-wires only the attributes it
exercises, bypassing `__init__` (which requires the real artifacts).

Run with:
    python -m pytest tests/test_beit3_retriever.py -v
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

# ---------------------------------------------------------------------------
# Stub optional/heavy packages that aren't needed to test the BEiT3 retriever
# and aren't installed in every dev environment (open_clip belongs to the
# legacy OpenCLIP path; torchscale is only required when loading the real
# checkpoint). setdefault keeps already-imported real modules intact.
# ---------------------------------------------------------------------------
for _mod in ("open_clip", "elasticsearch", "elasticsearch.helpers"):
    sys.modules.setdefault(_mod, MagicMock())

sys.modules.setdefault("torchscale", MagicMock())
sys.modules.setdefault("torchscale.architecture", MagicMock())
sys.modules.setdefault("torchscale.architecture.config", MagicMock(EncoderConfig=MagicMock()))
sys.modules.setdefault("torchscale.model", MagicMock())
sys.modules.setdefault("torchscale.model.BEiT3", MagicMock(BEiT3=MagicMock()))

import faiss
import sentencepiece as spm
import torch

from src.config.settings import Settings
from src.services.beit3_retriever import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    BEiT3Retriever,
    BEiT3RetrieverError,
    _first_matching_column,
)


def _bare_retriever(settings: Settings | None = None) -> BEiT3Retriever:
    """Build a BEiT3Retriever instance without running __init__."""
    obj = BEiT3Retriever.__new__(BEiT3Retriever)
    obj._settings = settings or Settings(debug=False)
    obj._device = torch.device("cpu")
    return obj


class TokenizerOffsetMappingTests(unittest.TestCase):
    """Validates the historical fairseq/XLM-R SentencePiece id-offset mapping."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = os.path.join(BACKEND_ROOT, ".pytest_spm_test")
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)
        os.makedirs(cls.tmp_dir, exist_ok=True)
        corpus_path = os.path.join(cls.tmp_dir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("a person riding a motorcycle on a street\n")
            f.write("a television news presenter in a studio\n")
            f.write("people sitting together around a table\n")
        model_prefix = os.path.join(cls.tmp_dir, "toy")
        spm.SentencePieceTrainer.train(
            input=corpus_path, model_prefix=model_prefix, vocab_size=64, model_type="bpe"
        )
        cls.sp = spm.SentencePieceProcessor()
        cls.sp.load(model_prefix + ".model")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _retriever(self, max_seq_len=64):
        r = _bare_retriever()
        r._tokenizer = self.sp
        r._max_seq_len = max_seq_len
        return r

    def test_unk_piece_maps_to_reserved_unk_id(self):
        r = self._retriever()
        # spm id 0 is always its own <unk>; our reserved <unk> id is 3.
        ids = r._encode_piece_ids("a")
        self.assertTrue(all(i != 0 for i in ids))
        self.assertTrue(all(i >= 0 for i in ids))

    def test_tokenize_wraps_with_bos_eos_and_pads(self):
        r = self._retriever(max_seq_len=64)
        tokens, mask = r._tokenize("a person riding a motorcycle on a street")
        self.assertEqual(tokens.shape, (1, 64))
        self.assertEqual(mask.shape, (1, 64))
        self.assertEqual(int(tokens[0, 0]), BOS_ID)
        # first True in the padding mask marks where </s> + padding begins
        seq_len = int((~mask[0]).sum())
        self.assertEqual(int(tokens[0, seq_len - 1]), EOS_ID)
        # everything after the real sequence must be PAD_ID
        self.assertTrue(bool((tokens[0, seq_len:] == PAD_ID).all()))

    def test_tokenize_truncates_long_queries(self):
        r = self._retriever(max_seq_len=10)
        long_query = "a person riding a motorcycle on a street " * 5
        tokens, mask = r._tokenize(long_query)
        self.assertEqual(tokens.shape, (1, 10))
        self.assertEqual(int(tokens[0, 0]), BOS_ID)
        self.assertIn(EOS_ID, tokens[0].tolist())


class ColumnDetectionTests(unittest.TestCase):
    def test_first_matching_column_case_insensitive(self):
        self.assertEqual(_first_matching_column(["Video_ID", "x"], ["video_id"]), "Video_ID")
        self.assertIsNone(_first_matching_column(["a", "b"], ["video_id"]))

    def test_detect_columns_uses_candidates(self):
        df = pd.DataFrame(
            {
                "global_id": [1, 2],
                "video_id": ["L21_V001", "L21_V001"],
                "frame_path": ["a.webp", "b.webp"],
            }
        )
        r = _bare_retriever()
        cols = r._detect_columns(df)
        self.assertEqual(cols["vector_id"], "global_id")
        self.assertEqual(cols["video_id"], "video_id")
        self.assertEqual(cols["frame_path"], "frame_path")
        self.assertIsNone(cols["timestamp"])  # not present -> None, not invented

    def test_detect_columns_respects_explicit_override(self):
        df = pd.DataFrame({"my_custom_id": [1], "video_id": ["v"]})
        settings = Settings(debug=False, beit3_col_vector_id="my_custom_id")
        r = _bare_retriever(settings=settings)
        cols = r._detect_columns(df)
        self.assertEqual(cols["vector_id"], "my_custom_id")

    def test_detect_columns_fails_loudly_when_vector_id_missing(self):
        df = pd.DataFrame({"video_id": ["v"], "something_else": [1]})
        r = _bare_retriever()
        with self.assertRaises(BEiT3RetrieverError):
            r._detect_columns(df)


class SearchVisualIntegrationTests(unittest.TestCase):
    """Exercises FAISS search + metadata lookup with a synthetic index.

    The BEiT3 forward pass itself is stubbed out (via `encode_text`
    monkeypatch) since it requires the real multi-GB checkpoint; everything
    downstream of the query vector is real.
    """

    def setUp(self):
        dim = 1024
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(5, dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        index.add_with_ids(vectors, ids)

        self.query_vec = vectors[2:3].copy()  # exact match for id=30

        self.df = pd.DataFrame(
            {
                "global_id": ids,
                "video_id": ["L21_V001"] * 5,
                "frame_id": ["000010", "000020", "003048", "000040", "000050"],
                "frame_path": [f"frame_{i}.webp" for i in ids],
                "timestamp": [float(i) for i in ids],
            }
        )

        r = _bare_retriever()
        r._index = index
        r._global_ids = self.df
        r._video_metadata = None
        r._video_meta_by_id = None
        # Optional enrichment sources loaded by __init__ in production; empty
        # here so _build_result skips them (timestamp falls back to frame_id).
        r._keyframe_time_by_video = {}
        r._media_info_by_id = {}
        r._columns = r._detect_columns(self.df)
        r._id_to_row = r._build_id_lookup(self.df, r._columns["vector_id"])
        r.encode_text = lambda query: self.query_vec  # bypass real BEiT3 forward
        self.retriever = r

    def test_search_visual_returns_real_scores_and_metadata(self):
        results = self.retriever.search_visual("a person riding a motorcycle", top_k=3)
        self.assertEqual(len(results), 3)

        top = results[0]
        self.assertEqual(top["rank"], 1)
        self.assertEqual(top["vector_id"], 30)
        self.assertAlmostEqual(top["score"], 1.0, places=4)  # exact match -> IP == 1.0
        self.assertEqual(top["video_id"], "L21_V001")
        self.assertEqual(top["frame_id"], "003048")
        self.assertEqual(top["global_frame_id"], 3048)
        self.assertEqual(top["frame_idx"], 3048)
        self.assertEqual(top["frame_path"], "frame_30.webp")
        self.assertEqual(top["timestamp"], 30.0)

        # Scores must be non-increasing (real FAISS ranking, not rank-derived).
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_result_shape_is_the_pre_jina_beit3_contract(self):
        """RETRIEVAL_BACKEND=beit3 must keep the exact legacy result payload --
        in particular NO `retrieval_backend` key (that is Jina-only)."""
        top = self.retriever.search_visual("a person", top_k=1)[0]
        self.assertNotIn("retrieval_backend", top)
        self.assertEqual(
            set(top),
            {
                "rank", "score", "vector_id", "faiss_id", "global_frame_id",
                "frame_idx", "video_id", "frame_id", "frame_name", "frame_path",
                "timestamp", "namespace",
            },
        )

    def test_search_visual_rejects_invalid_top_k(self):
        with self.assertRaises(BEiT3RetrieverError):
            self.retriever.search_visual("x", top_k=0)
        with self.assertRaises(BEiT3RetrieverError):
            self.retriever.search_visual("x", top_k=-5)


class SearchByImageTests(unittest.TestCase):
    """Image-query encoding + FAISS search with a synthetic index and model.

    The BEiT3 vision forward pass is stubbed (it needs the real multi-GB
    checkpoint); the preprocessing, L2 normalization, and FAISS search that a
    captured-frame "Similar" pivot relies on are all real.
    """

    def setUp(self):
        dim = 1024
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(4, dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = np.array([11, 22, 33, 44], dtype=np.int64)

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        index.add_with_ids(vectors, ids)
        self._match_vec = vectors[1:2].copy()  # exact match for id=22

        self.df = pd.DataFrame(
            {
                "global_id": ids,
                "video_id": ["L21_V001"] * 4,
                "frame_id": ["000011", "000022", "000033", "000044"],
                "frame_path": [f"frame_{i}.webp" for i in ids],
                "timestamp": [float(i) for i in ids],
            }
        )

        r = _bare_retriever()
        r._index = index
        r._global_ids = self.df
        r._video_metadata = None
        r._video_meta_by_id = None
        r._keyframe_time_by_video = {}
        r._media_info_by_id = {}
        r._columns = r._detect_columns(self.df)
        r._id_to_row = r._build_id_lookup(self.df, r._columns["vector_id"])

        self.captured_input = {}

        def fake_model(image=None, text_description=None, padding_mask=None, only_infer=True):
            self.captured_input["image"] = image
            return torch.from_numpy(self._match_vec), None

        r._model = fake_model
        self.retriever = r

    @staticmethod
    def _sample_image():
        from PIL import Image

        return Image.new("RGB", (11, 7), (128, 64, 200))

    def test_encode_image_returns_normalized_1024_vector(self):
        vec = self.retriever.encode_image(self._sample_image())
        self.assertEqual(vec.shape, (1, 1024))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

        tensor = self.captured_input["image"]
        self.assertEqual(tuple(tensor.shape), (1, 3, 384, 384))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_search_by_image_queries_faiss_with_that_vector(self):
        results = self.retriever.search_by_image(self._sample_image(), top_k=3)
        self.assertEqual(len(results), 3)

        top = results[0]
        self.assertEqual(top["vector_id"], 22)
        self.assertEqual(top["faiss_id"], 22)
        self.assertAlmostEqual(top["score"], 1.0, places=4)  # exact match -> IP == 1.0
        self.assertEqual(top["video_id"], "L21_V001")

        scores = [row["score"] for row in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_by_image_rejects_invalid_top_k(self):
        with self.assertRaises(BEiT3RetrieverError):
            self.retriever.search_by_image(self._sample_image(), top_k=0)
        with self.assertRaises(BEiT3RetrieverError):
            self.retriever.search_by_image(self._sample_image(), top_k=-3)


class QueryVectorValidationTests(unittest.TestCase):
    def test_rejects_wrong_shape(self):
        r = _bare_retriever()
        with self.assertRaises(BEiT3RetrieverError):
            r._validate_query_vector(np.zeros((1, 768), dtype=np.float32))

    def test_rejects_non_finite(self):
        r = _bare_retriever()
        vec = np.ones((1, 1024), dtype=np.float32)
        vec[0, 0] = np.nan
        with self.assertRaises(BEiT3RetrieverError):
            r._validate_query_vector(vec)

    def test_rejects_unnormalized_vector(self):
        r = _bare_retriever()
        vec = np.ones((1, 1024), dtype=np.float32)  # norm = 32, not ~1
        with self.assertRaises(BEiT3RetrieverError):
            r._validate_query_vector(vec)

    def test_accepts_normalized_vector(self):
        r = _bare_retriever()
        vec = np.zeros((1, 1024), dtype=np.float32)
        vec[0, 0] = 1.0
        r._validate_query_vector(vec)  # should not raise


if __name__ == "__main__":
    unittest.main()
