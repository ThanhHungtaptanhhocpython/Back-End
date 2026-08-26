import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data_extraction" / "new" / "realign_asr_keyframes.py"
spec = importlib.util.spec_from_file_location("realign_asr_keyframes", MODULE_PATH)
realign_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(realign_module)


def test_realign_matches_full_video_id_from_split_and_short_metadata_id():
    metadata = {
        "10": {
            "faiss_id": 10,
            "video_id": "V001",
            "split": "videos-l21-a",
            "frame_name": "keyframe_L21_V001_0010.webp",
            "global_frame_id": 300,
            "timestamp": 10.0,
            "fps": 25.0,
        },
        "20": {
            "faiss_id": 20,
            "video_id": "V001",
            "split": "videos-l21-a",
            "frame_name": "keyframe_L21_V001_0020.webp",
            "global_frame_id": 600,
            "timestamp": 20.0,
            "fps": 25.0,
        },
    }
    asr_docs = [
        {
            "video_id": "L21_V001",
            "start_time": 17.0,
            "end_time": 19.0,
            "text": "sample transcript",
            "nearest_faiss_id": 0,
        }
    ]

    keyframe_index = realign_module.build_keyframe_index(metadata)
    aligned_docs, stats = realign_module.realign(asr_docs, keyframe_index)

    assert stats["aligned"] == 1
    assert stats["changed"] == 1
    assert aligned_docs[0]["nearest_faiss_id"] == 20
    assert aligned_docs[0]["nearest_frame_name"] == "keyframe_L21_V001_0020.webp"
    assert aligned_docs[0]["nearest_global_frame_id"] == 600
    assert aligned_docs[0]["nearest_timestamp"] == 20.0
    assert aligned_docs[0]["alignment_delta_seconds"] == 2.0
    assert aligned_docs[0]["alignment_source"] == "metadata_nearest_timestamp"
