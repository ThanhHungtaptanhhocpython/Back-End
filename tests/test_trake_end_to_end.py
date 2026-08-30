import os
import sys

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.config.settings import get_settings
from src.utils.trake_processing import TRAKE


def _frame(name, timestamp, *, score=1.0, video_id="L1_V001", split="L1_a", **extra):
    frame = {
        "frame_name": name,
        "global_frame_id": extra.pop("global_frame_id", 0),
        "timestamp": timestamp,
        "score": score,
        "split": split,
        "video_id": video_id,
        "query": extra.pop("query", "event"),
        "query_en": extra.pop("query_en", "event"),
    }
    frame.update(extra)
    return frame


def test_resolve_submission_frame_id_prefers_original_index_never_vector_id():
    trake = TRAKE()
    frame = _frame("keyframe_L1_V001_0007.webp", 1.0, submission_frame_id=3048, vector_id=918273, faiss_idx=918273)
    assert trake._resolve_submission_frame_id(frame) == 3048

    frame_no_explicit = _frame("x.webp", 1.0, global_frame_id=512, vector_id=99999)
    assert trake._resolve_submission_frame_id(frame_no_explicit) == 512


def test_resolve_submission_frame_id_reads_digits_from_resolved_filename(monkeypatch):
    trake = TRAKE()
    monkeypatch.setattr(trake, "_resolve_image_path", lambda frame: "/data/L1_a/L1_V001/000660.webp")
    frame = _frame("legacy_name.webp", 1.0)
    frame.pop("global_frame_id", None)
    assert trake._resolve_submission_frame_id(frame) == 660


def test_sequence_is_valid_rejects_partial_mixed_and_out_of_order():
    trake = TRAKE()

    ok = {
        "video_id": "L1_V001",
        "frame_details": [_frame("a.webp", 10.0, global_frame_id=100), _frame("b.webp", 20.0, global_frame_id=200)],
    }
    assert trake._sequence_is_valid(ok, 2) is True

    partial = {"video_id": "L1_V001", "frame_details": [_frame("a.webp", 10.0, global_frame_id=100)]}
    assert trake._sequence_is_valid(partial, 2) is False

    mixed = {
        "video_id": "L1_V001",
        "frame_details": [
            _frame("a.webp", 10.0, global_frame_id=100),
            _frame("b.webp", 20.0, global_frame_id=200, video_id="L1_V002", split="L1_a"),
        ],
    }
    assert trake._sequence_is_valid(mixed, 2) is False

    unordered = {
        "video_id": "L1_V001",
        "frame_details": [
            _frame("a.webp", 20.0, global_frame_id=100),
            _frame("b.webp", 20.0, global_frame_id=200),
        ],
    }
    assert trake._sequence_is_valid(unordered, 2) is False


def test_format_response_exposes_per_event_contract():
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: "img"
    sequence = {
        "video_id": "L1_V001",
        "total_score": 1.8,
        "base_score": 1.6,
        "temporal_gaps": [10.0],
        "frame_details": [
            _frame("a.webp", 10.0, global_frame_id=100, query="cắt nấm", query_en="cutting mushrooms",
                   evidence_scores={"ocr": 0.4}, evidence_text={"ocr": "nấm"}),
            _frame("b.webp", 20.0, global_frame_id=200, query="bắc chảo", query_en="pan on stove"),
        ],
    }

    [formatted] = trake.format_response([sequence])

    assert formatted["video_id"] == "L1_V001"
    assert formatted["sequence_id"].startswith("L1_V001#")
    assert [f["event_index"] for f in formatted["frames"]] == [1, 2]
    assert formatted["frames"][0]["event_query"] == "cắt nấm"
    assert formatted["frames"][0]["submission_frame_id"] == 100
    assert formatted["frames"][0]["evidence"]["text"] == {"ocr": "nấm"}
    assert formatted["frames"][1]["submission_frame_id"] == 200


def test_process_temporal_search_drops_mixed_video_and_keeps_clean_sequence(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "false")
    monkeypatch.setenv("TRAKE_MAX_EVENT_GAP_SECONDS", "120")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""

    results = {
        "event one": [
            _frame("a.webp", 10.0, score=0.9, video_id="L1_V001", global_frame_id=100),
            _frame("x.webp", 10.0, score=0.8, video_id="L1_V002", global_frame_id=10),
        ],
        "event two": [
            _frame("b.webp", 25.0, score=0.9, video_id="L1_V001", global_frame_id=250),
            _frame("y.webp", 400.0, score=0.8, video_id="L1_V002", global_frame_id=40),
        ],
    }
    trake.retrieve_top_k = lambda query, k: [dict(item) for item in results[query]]

    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}], top_k=10, top_results=5
    )

    assert len(response) == 1
    assert response[0]["video_id"] == "L1_V001"
    assert [f["submission_frame_id"] for f in response[0]["frames"]] == [100, 250]
    get_settings.cache_clear()
