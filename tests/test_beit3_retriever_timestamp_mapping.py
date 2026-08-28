import os
import sys
from pathlib import Path

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.services.beit3_retriever import BEiT3Retriever


def test_keyframe_time_uses_nearest_source_frame_idx_before_fallback():
    retriever = BEiT3Retriever.__new__(BEiT3Retriever)
    retriever._keyframe_time_by_video = BEiT3Retriever._load_keyframe_time_maps(
        retriever,
        Path("src/dict/map-keyframes"),
    )

    info = retriever._keyframe_time_for("L22_V011", "15530")

    assert info is not None
    assert info["timestamp"] == 516.967
    assert info["fps"] == 30.0
    assert info["timestamp_source"] == "map_frame_idx_nearest"
    assert info["timestamp_matched_frame_idx"] == 15509
    assert info["timestamp_frame_idx_delta"] == -21