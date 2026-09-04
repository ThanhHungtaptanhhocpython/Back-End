"""Unit coverage for the Jina fine-keyframe retrieval contract."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import faiss
import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings
from src.services.jina_retriever import JinaRetriever, JinaRetrieverError, _hf_offline_if
from src.services.visual_retriever import get_visual_retriever


def _settings(**overrides) -> Settings:
    values = {
        "debug": False,
        "_env_file": None,
        "jina_truncate_dim": 4,
        "jina_device": "cpu",
    }
    values.update(overrides)
    return Settings(**values)


def _bare_retriever() -> JinaRetriever:
    retriever = JinaRetriever.__new__(JinaRetriever)
    retriever._settings = _settings()
    retriever._truncate_dim = 4
    retriever._media_info_by_id = {}
    retriever._keyframe_time_by_video = {}
    return retriever


def _corpus() -> tuple[faiss.Index, pd.DataFrame, np.ndarray]:
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids = np.arange(3, dtype=np.int64)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(4))
    index.add_with_ids(vectors, ids)
    rows = pd.DataFrame(
        {
            "parent_namespace": ["L22_a"] * 3,
            "video_id": ["L22_V001"] * 3,
            "frame_id": ["keyframe_0000", "keyframe_0001", "keyframe_0002"],
            "frame_path": [
                "L22_a/L22_V001/keyframe_0000.jpg",
                "L22_a/L22_V001/keyframe_0001.jpg",
                "L22_a/L22_V001/keyframe_0002.jpg",
            ],
            "timestamp": [0.4, 2.0, 4.0],
            "source_fps": [25.0] * 3,
            "source_frame_idx": [10, 50, 100],
            "local_position": [0, 1, 2],
            "vector_id": ids,
        }
    )
    return index, rows, vectors


def test_jina_result_uses_source_frame_idx_not_keyframe_ordinal():
    index, rows, vectors = _corpus()
    retriever = _bare_retriever()
    retriever._index = index
    retriever._global_ids = rows
    retriever._video_metadata = pd.DataFrame(
        {
            "video_id": ["L22_V001"],
            "parent_namespace": ["L22_a"],
            "frame_count": [3],
            "embedding_dim": [4],
            "first_vector_id": [0],
        }
    )
    retriever._columns = retriever._detect_jina_columns()
    retriever._id_to_row = retriever._build_id_lookup(rows, "vector_id")
    retriever._video_meta_by_id = retriever._build_video_metadata_lookup()
    retriever._video_to_rows = retriever._build_video_to_rows()
    retriever.encode_text = lambda _query: vectors[1:2]

    result = retriever.search_visual("test", top_k=1)[0]

    assert result["vector_id"] == 1
    assert result["frame_id"] == "keyframe_0001"
    assert result["global_frame_id"] == 50
    assert result["submission_frame_id"] == 50
    assert result["frame_idx"] == 50
    assert result["timestamp"] == 2.0
    assert result["keyframe_number"] == 2
    assert result["timestamp_source"] == "jina_global_ids"
    assert result["retriever"] == "jina"


def test_jina_normalizes_model_output():
    retriever = _bare_retriever()
    vector = retriever._normalize_embedding(np.asarray([[3.0, 4.0, 0.0, 0.0]]))
    assert vector.shape == (1, 4)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_jina_batches_text_encoding_and_reuses_query_cache():
    retriever = _bare_retriever()
    retriever._encode_lock = __import__("threading").Lock()
    retriever._text_embedding_cache = {}
    retriever._settings = _settings(jina_query_task="retrieval.query")
    retriever._model = MagicMock()
    retriever._model.encode_text.return_value = np.asarray(
        [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]],
        dtype=np.float32,
    )

    first = retriever.encode_text_batch(["event one", "event two"])
    second = retriever.encode_text_batch(["event two", "event one"])

    assert first.shape == (2, 4)
    assert np.linalg.norm(first, axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert np.allclose(second, first[::-1])
    retriever._model.encode_text.assert_called_once()


def test_jina_rejects_wrong_embedding_shape():
    retriever = _bare_retriever()
    with pytest.raises(JinaRetrieverError, match="shape"):
        retriever._normalize_embedding(np.ones((1, 8), dtype=np.float32))


def test_jina_offline_context_sets_and_restores_hf_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "old")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)

    with _hf_offline_if(True):
        import os

        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"

    import os

    assert os.environ["HF_HUB_OFFLINE"] == "old"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "HF_DATASETS_OFFLINE" not in os.environ


def test_jina_validates_index_metadata_and_alignment():
    index, rows, _vectors = _corpus()
    retriever = _bare_retriever()
    retriever._index = index
    retriever._global_ids = rows
    retriever._video_metadata = pd.DataFrame({"frame_count": [3]})
    retriever._index_meta = {
        "model": "jina",
        "embedding_dim": 4,
        "vector_count": 3,
        "metric": "inner_product_on_l2_normalized_vectors",
    }
    retriever._validate_jina_consistency()

    retriever._global_ids = rows.copy()
    retriever._global_ids.loc[0, "timestamp"] = None
    with pytest.raises(JinaRetrieverError, match="null"):
        retriever._validate_jina_consistency()


def test_selector_uses_explicit_jina_without_constructing_beit3(monkeypatch):
    jina = MagicMock(name="jina")
    jina_module = types.ModuleType("src.services.jina_retriever")
    jina_module.get_jina_retriever = MagicMock(return_value=jina)
    beit3_module = types.ModuleType("src.services.beit3_retriever")
    beit3_module.get_beit3_retriever = MagicMock()
    monkeypatch.setitem(sys.modules, "src.services.jina_retriever", jina_module)
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", beit3_module)

    selected = get_visual_retriever(_settings(visual_retriever="jina"))

    assert selected is jina
    jina_module.get_jina_retriever.assert_called_once_with()
    beit3_module.get_beit3_retriever.assert_not_called()


def test_selector_does_not_silently_fallback_when_jina_fails(monkeypatch):
    jina_module = types.ModuleType("src.services.jina_retriever")
    jina_module.get_jina_retriever = MagicMock(side_effect=JinaRetrieverError("bad index"))
    beit3_module = types.ModuleType("src.services.beit3_retriever")
    beit3_module.get_beit3_retriever = MagicMock()
    monkeypatch.setitem(sys.modules, "src.services.jina_retriever", jina_module)
    monkeypatch.setitem(sys.modules, "src.services.beit3_retriever", beit3_module)

    with pytest.raises(JinaRetrieverError, match="bad index"):
        get_visual_retriever(_settings(visual_retriever="jina"))

    beit3_module.get_beit3_retriever.assert_not_called()
