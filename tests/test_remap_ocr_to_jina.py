import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data_extraction" / "new" / "remap_ocr_to_jina.py"
spec = importlib.util.spec_from_file_location("remap_ocr_to_jina", MODULE_PATH)
remap_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(remap_module)


def test_remap_ocr_uses_nearest_jina_frame_and_source_frame_index(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    global_ids_path = tmp_path / "global_ids.parquet"
    pd.DataFrame(
        [
            {
                "vector_id": 11,
                "parent_namespace": "L21_a",
                "video_id": "L21_V001",
                "frame_id": "keyframe_0009",
                "frame_path": "L21_a/L21_V001/keyframe_0009.jpg",
                "timestamp": 9.0,
                "source_frame_idx": 270,
            },
            {
                "vector_id": 12,
                "parent_namespace": "L21_a",
                "video_id": "L21_V001",
                "frame_id": "keyframe_0011",
                "frame_path": "L21_a/L21_V001/keyframe_0011.jpg",
                "timestamp": 11.0,
                "source_frame_idx": 330,
            },
        ]
    ).to_parquet(global_ids_path, index=False)

    index = remap_module.build_jina_keyframe_index(global_ids_path)
    remapped, stats = remap_module.remap_ocr(
        [
            {
                "faiss_id": 99,
                "video_id": "V001",
                "split": "L21",
                "frame_name": "004.jpg",
                "global_frame_id": 300,
                "timestamp": 10.6,
                "ocr_text": "Muc tieu",
                "language": "vi",
            }
        ],
        index,
        max_delta_seconds=2.0,
    )

    assert stats["aligned"] == 1
    assert remapped == [
        {
            "vector_id": 12,
            "faiss_id": 12,
            "video_id": "L21_V001",
            "parent_namespace": "L21_a",
            "split": "L21_a",
            "frame_id": "keyframe_0011",
            "frame_name": "keyframe_0011.jpg",
            "frame_path": "L21_a/L21_V001/keyframe_0011.jpg",
            "source_frame_idx": 330,
            "global_frame_id": 330,
            "timestamp": 11.0,
            "ocr_source_timestamp": 10.6,
            "alignment_delta_seconds": 0.4,
            "alignment_source": "jina_nearest_timestamp",
            "ocr_text": "Muc tieu",
            "language": "vi",
            "legacy_faiss_id": 99,
            "legacy_frame_name": "004.jpg",
            "legacy_global_frame_id": 300,
        }
    ]


def test_remap_ocr_skips_rows_outside_allowed_timestamp_distance():
    index = {
        "L21_V001": {
            "timestamps": [10.0],
            "frames": [
                {
                    "vector_id": 7,
                    "video_id": "L21_V001",
                    "parent_namespace": "L21_a",
                    "frame_id": "keyframe_0010",
                    "frame_path": "L21_a/L21_V001/keyframe_0010.jpg",
                    "frame_name": "keyframe_0010.jpg",
                    "timestamp": 10.0,
                    "source_frame_idx": 300,
                }
            ],
        }
    }
    remapped, stats = remap_module.remap_ocr(
        [{"video_id": "L21_V001", "timestamp": 20.0, "ocr_text": "too far"}],
        index,
        max_delta_seconds=2.0,
    )

    assert remapped == []
    assert stats["skipped_timestamp_delta"] == 1
