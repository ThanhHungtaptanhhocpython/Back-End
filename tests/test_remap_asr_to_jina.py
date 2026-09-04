from scripts.data_extraction.new.remap_asr_to_jina import remap_asr


def test_remap_asr_uses_segment_midpoint_and_preserves_text():
    keyframe_index = {
        "L21_V001": {
            "timestamps": [10.0, 20.0],
            "frames": [
                {
                    "vector_id": 100,
                    "video_id": "L21_V001",
                    "parent_namespace": "L21_a",
                    "frame_id": "keyframe_0001",
                    "frame_name": "keyframe_0001.jpg",
                    "frame_path": "L21_a/L21_V001/keyframe_0001.jpg",
                    "timestamp": 10.0,
                    "source_frame_idx": 300,
                },
                {
                    "vector_id": 101,
                    "video_id": "L21_V001",
                    "parent_namespace": "L21_a",
                    "frame_id": "keyframe_0002",
                    "frame_name": "keyframe_0002.jpg",
                    "frame_path": "L21_a/L21_V001/keyframe_0002.jpg",
                    "timestamp": 20.0,
                    "source_frame_idx": 600,
                },
            ],
        }
    }
    docs = [{"video_id": "L21_V001", "start_time": 13.0, "end_time": 17.0, "text": "hello", "nearest_faiss_id": 7}]

    remapped, stats = remap_asr(docs, keyframe_index)

    assert stats["aligned"] == 1
    assert remapped[0]["text"] == "hello"
    assert remapped[0]["nearest_vector_id"] == 101
    assert remapped[0]["nearest_faiss_id"] == 101
    assert remapped[0]["legacy_nearest_faiss_id"] == 7
    assert remapped[0]["alignment_source"] == "jina_nearest_timestamp"
