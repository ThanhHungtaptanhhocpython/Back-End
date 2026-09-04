from src.services.beit3_retriever import BEiT3Retriever


def test_full_video_timeline_uses_evenly_spaced_keyframes():
    retriever = BEiT3Retriever.__new__(BEiT3Retriever)
    retriever._video_to_rows = {
        "L24_V018": [{"frame_id": f"keyframe_{index:04d}", "vector_id": index} for index in range(100)]
    }
    retriever._columns = {"frame_id": "frame_id", "vector_id": "vector_id"}
    retriever._build_result = lambda rank, _score, vector_id, row: {
        "rank": rank,
        "vector_id": vector_id,
        "frame_id": row["frame_id"],
    }

    timeline = retriever.get_video_timeline("L24_V018", around_frame_id="keyframe_0050", limit=5, full_video=True)

    assert [item["vector_id"] for item in timeline] == [0, 24, 50, 74, 99]
