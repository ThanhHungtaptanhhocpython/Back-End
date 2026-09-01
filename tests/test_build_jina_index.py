"""Tests for scripts/cloud/build_jina_index.py -- the Jina runtime artifact
build script (per-video .npy + metadata JSON + map-keyframes -> FAISS index
+ parquet + index_meta + build report)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location(
    "build_jina_index",
    os.path.join(BACKEND_ROOT, "scripts", "cloud", "build_jina_index.py"),
)
bji = importlib.util.module_from_spec(_spec)
# Dataclasses under `from __future__ import annotations` resolve their field
# types via sys.modules[cls.__module__] at class-creation time, so the module
# must be registered there before exec_module runs its class bodies.
sys.modules["build_jina_index"] = bji
_spec.loader.exec_module(bji)


DIM = bji.EXPECTED_DIM


def _unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _write_video(
    tmp_path: Path,
    namespace: str,
    video_id: str,
    n_frames: int,
    *,
    seed: int = 0,
    dim: int = DIM,
    with_timestamp: bool = True,
    local_position_offset: int = 0,
) -> None:
    emb_dir = tmp_path / "embeddings" / namespace
    rec_dir = tmp_path / "records" / namespace
    emb_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    vectors = _unit_vectors(n_frames, dim=dim, seed=seed)
    np.save(emb_dir / f"{video_id}.npy", vectors)

    records = []
    for i in range(n_frames):
        rec = {
            "parent_namespace": namespace,
            "video_id": video_id,
            "frame_id": f"keyframe_{i:04d}",
            "frame_path": f"{namespace}/{video_id}/keyframe_{i:04d}.jpg",
            "timestamp": float(i) if with_timestamp else None,
            "source_fps": 25.0,
            "source_frame_idx": i * 8,
            "local_position": i + local_position_offset,
        }
        records.append(rec)
    (rec_dir / f"{video_id}.json").write_text(json.dumps({"records": records}), encoding="utf-8")


class TestHappyPath:
    def test_builds_valid_index_and_parquet(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 3, seed=1)
        _write_video(tmp_path, "L21_a", "L21_V002", 2, seed=2)

        result = bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)
        assert result.index.ntotal == 5
        assert len(result.df) == 5
        assert result.df["vector_id"].is_unique
        assert result.df["vector_id"].tolist() == list(range(5))
        assert set(result.df["video_id"]) == {"L21_V001", "L21_V002"}
        row = result.df.iloc[0]
        assert row["asset_key"] == "L21_a/L21_V001/keyframe_0000.jpg"
        assert row["frame_path"] == row["asset_key"]
        assert row["timestamp_ms"] == 0
        assert row["split"] == "L21_a"
        assert row["keyframe_ordinal"] == 1
        assert row["source_frame_id"] == 0

    def test_main_writes_all_four_output_files(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 3, seed=3)
        out_dir = tmp_path / "out"
        rc = bji.main([
            "--embeddings-root", str(tmp_path / "embeddings"),
            "--records-root", str(tmp_path / "records"),
            "--out-dir", str(out_dir),
            "--model-revision", "deadbeef1234",
        ])
        assert rc == 0
        assert (out_dir / "jina_faiss.index").is_file()
        assert (out_dir / "jina_global_ids.parquet").is_file()
        assert (out_dir / "jina_index_meta.json").is_file()
        assert (out_dir / "jina_build_report.json").is_file()

        meta = json.loads((out_dir / "jina_index_meta.json").read_text(encoding="utf-8"))
        assert meta["dimension"] == DIM
        assert meta["vector_count"] == 3
        assert meta["model_revision"] == "deadbeef1234"

        # dogfood: FAISS ntotal really does match the parquet on disk
        import faiss

        index = faiss.read_index(str(out_dir / "jina_faiss.index"))
        df = pd.read_parquet(out_dir / "jina_global_ids.parquet")
        assert index.ntotal == len(df)


class TestValidationCatchesRowMismatch:
    def test_npy_row_count_vs_record_count_mismatch(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 3, seed=4)
        # Truncate the records file to 2 entries -- npy still has 3 rows.
        rec_path = tmp_path / "records" / "L21_a" / "L21_V001.json"
        payload = json.loads(rec_path.read_text(encoding="utf-8"))
        payload["records"] = payload["records"][:2]
        rec_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(SystemExit, match="embedding rows"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)

    def test_local_position_out_of_order_is_rejected(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 3, seed=5, local_position_offset=1)
        with pytest.raises(SystemExit, match="local_position"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)


class TestValidationCatchesBadEmbeddings:
    def test_wrong_dimension_rejected(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 2, seed=6, dim=512)
        with pytest.raises(SystemExit, match="shape"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)

    def test_non_finite_embedding_rejected(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 2, seed=7)
        npy_path = tmp_path / "embeddings" / "L21_a" / "L21_V001.npy"
        arr = np.load(npy_path)
        arr[0, 0] = np.nan
        np.save(npy_path, arr)
        with pytest.raises(SystemExit, match="NaN/Inf"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)

    def test_non_normalized_embedding_rejected(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 2, seed=8)
        npy_path = tmp_path / "embeddings" / "L21_a" / "L21_V001.npy"
        arr = np.load(npy_path)
        arr[0] *= 5.0  # break unit-norm
        np.save(npy_path, arr)
        with pytest.raises(SystemExit, match="normalized"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)


class TestValidationCatchesBadMetadata:
    def test_missing_timestamp_without_map_keyframes_fails(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 2, seed=9, with_timestamp=False)
        with pytest.raises(SystemExit, match="timestamp"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)

    def test_missing_timestamp_is_filled_from_map_keyframes(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 2, seed=10, with_timestamp=False)
        mk_dir = tmp_path / "map-keyframes"
        mk_dir.mkdir(parents=True, exist_ok=True)
        with (mk_dir / "L21_V001.csv").open("w", encoding="utf-8", newline="") as f:
            f.write("n,pts_time,fps,frame_idx\n")
            f.write("1,0.0,25.0,0\n")
            f.write("2,1.5,25.0,8\n")

        result = bji.build_index(tmp_path / "embeddings", tmp_path / "records", mk_dir)
        assert result.df.iloc[1]["timestamp_ms"] == 1500
        assert any("filled from map-keyframes" in w for w in result.warnings)

    def test_unsafe_video_id_rejected(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 1, seed=11)
        rec_path = tmp_path / "records" / "L21_a" / "L21_V001.json"
        payload = json.loads(rec_path.read_text(encoding="utf-8"))
        payload["records"][0]["video_id"] = "../../escape"
        rec_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="video_id"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)

    def test_duplicate_asset_key_across_corpus_rejected(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 1, seed=12)
        _write_video(tmp_path, "L21_a", "L21_V002", 1, seed=13)
        # Force a collision: V002's frame_path claims to be V001's.
        rec_path = tmp_path / "records" / "L21_a" / "L21_V002.json"
        payload = json.loads(rec_path.read_text(encoding="utf-8"))
        payload["records"][0]["frame_path"] = "L21_a/L21_V001/keyframe_0000.jpg"
        rec_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="duplicate asset_key"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)

    def test_missing_records_file_rejected(self, tmp_path: Path):
        _write_video(tmp_path, "L21_a", "L21_V001", 1, seed=14)
        (tmp_path / "records" / "L21_a" / "L21_V001.json").unlink()
        with pytest.raises(SystemExit, match="no matching records"):
            bji.build_index(tmp_path / "embeddings", tmp_path / "records", None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
