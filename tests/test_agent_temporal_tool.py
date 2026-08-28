from src.agent.tools import _compact_result, _temporal_queries


def test_temporal_queries_orders_truoc_do_event_first():
    query = "C\u00f3 4 t\u00e0i x\u1ebf trong tr\u1ea1m x\u0103ng, tr\u01b0\u1edbc \u0111\u00f3 l\u00e0 c\u1ea3nh m\u1ed9t ng\u01b0\u1eddi \u0111\u1eady n\u1eafp b\u00ecnh x\u0103ng."
    assert _temporal_queries(query) == [
        {"query": "l\u00e0 c\u1ea3nh m\u1ed9t ng\u01b0\u1eddi \u0111\u1eady n\u1eafp b\u00ecnh x\u0103ng"},
        {"query": "C\u00f3 4 t\u00e0i x\u1ebf trong tr\u1ea1m x\u0103ng"},
    ]


def test_compact_result_keeps_temporal_video_and_keyframes_without_image():
    result = {
        "id": 0,
        "video_id": "L22_V029",
        "frames": [{
            "video_key": "L22_V029",
            "frame_key": "000052",
            "timestamp": 199.933,
            "image": "large-base64-value",
        }],
    }
    assert _compact_result(result) == {
        "video_id": "L22_V029",
        "keyframes": [{
            "video_key": "L22_V029",
            "frame_key": "000052",
            "timestamp": 199.933,
        }],
    }