import os
import shutil
import sys
from pathlib import Path

from PIL import Image

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.config.settings import get_settings
from src.services import openrouter_trake_verifier
from src.services.openrouter_trake_verifier import verify_trake_sequences
from src.utils.trake_processing import TRAKE


def _candidate(name, timestamp, score=1.0, global_frame_id=0, video_id="L1_V001"):
    return {
        "frame_name": name,
        "global_frame_id": global_frame_id,
        "timestamp": timestamp,
        "score": score,
        "split": "L1_a",
        "video_id": video_id,
    }


def test_beam_search_uses_real_timestamps_instead_of_global_ids(monkeypatch):
    monkeypatch.setenv("TRAKE_MAX_EVENT_GAP_SECONDS", "60")
    get_settings.cache_clear()
    trake = TRAKE()

    sequences = trake.beam_search_sequences(
        "L1_V001",
        [
            [_candidate("first.webp", 10.0, global_frame_id=100)],
            [_candidate("second.webp", 20.0, global_frame_id=50)],
        ],
    )

    assert len(sequences) == 1
    assert sequences[0]["timestamps"] == [10.0, 20.0]
    assert sequences[0]["global_frame_ids"] == [100, 50]
    get_settings.cache_clear()


def test_beam_search_rejects_event_outside_temporal_window(monkeypatch):
    monkeypatch.setenv("TRAKE_MAX_EVENT_GAP_SECONDS", "30")
    get_settings.cache_clear()
    trake = TRAKE()

    sequences = trake.beam_search_sequences(
        "L1_V001",
        [
            [_candidate("first.webp", 10.0)],
            [_candidate("too-late.webp", 41.0)],
        ],
    )

    assert sequences == []
    get_settings.cache_clear()


def test_group_by_video_limits_each_event_candidate_pool(monkeypatch):
    monkeypatch.setenv("TRAKE_CANDIDATES_PER_EVENT_VIDEO", "2")
    get_settings.cache_clear()
    trake = TRAKE()
    candidates = [[
        _candidate("low.webp", 1.0, score=0.1),
        _candidate("high.webp", 2.0, score=0.9),
        _candidate("mid.webp", 3.0, score=0.5),
    ]]

    grouped = trake.group_by_video(candidates)

    assert [item["frame_name"] for item in grouped["L1_V001"][0]] == ["high.webp", "mid.webp"]
    get_settings.cache_clear()


def test_asr_evidence_boosts_only_nearby_candidate(monkeypatch):
    monkeypatch.setenv("TRAKE_EVIDENCE_WINDOW_SECONDS", "10")
    get_settings.cache_clear()


def test_process_temporal_search_returns_ordered_same_video_sequence(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "false")
    monkeypatch.setenv("TRAKE_MAX_EVENT_GAP_SECONDS", "60")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""
    results_by_query = {
        "event one": [
            _candidate("a.webp", 10.0, score=0.9, video_id="L1_V001"),
            _candidate("x.webp", 10.0, score=0.8, video_id="L1_V002"),
        ],
        "event two": [
            _candidate("b.webp", 20.0, score=0.9, video_id="L1_V001"),
            _candidate("y.webp", 100.0, score=0.8, video_id="L1_V002"),
        ],
    }
    trake.retrieve_top_k = lambda query, k: results_by_query[query]

    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}],
        top_k=10,
        top_results=5,
    )

    assert len(response) == 1
    assert response[0]["video_id"] == "L1_V001"
    assert response[0]["timestamps"] == [10.0, 20.0]
    assert response[0]["temporal_gaps"] == [10.0]
    assert response[0]["verification"]["method"] == "temporal_evidence"
    get_settings.cache_clear()
    trake = TRAKE()
    candidates = [
        {**_candidate("near.webp", 20.0), "visual_score": 0.5},
        {**_candidate("far.webp", 80.0), "visual_score": 0.5},
    ]
    evidence = {
        "ocr": [],
        "asr": [{"video_id": "L1_V001", "nearest_timestamp": 21.0, "_score": 4.0, "text": "target speech"}],
    }

    trake._apply_event_evidence(candidates, evidence)

    assert candidates[0]["score"] > candidates[1]["score"]
    assert candidates[0]["evidence_text"]["asr"] == "target speech"
    assert candidates[1]["evidence_scores"]["asr"] == 0.0
    get_settings.cache_clear()


def test_openrouter_sequence_verifier_reranks_and_keeps_event_contract(monkeypatch):
    scratch_root = Path("scratch") / "test_trake_vlm"
    scratch_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, color in enumerate(("red", "blue", "green", "yellow"), 1):
        path = scratch_root / f"{index}.webp"
        Image.new("RGB", (48, 48), color=color).save(path)
        paths.append(path)

    monkeypatch.setenv("AGENT_VLM_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "2")
    get_settings.cache_clear()

    def fake_request(messages, model, timeout, max_tokens):
        return {
            "items": [
                {"id": "s1", "score": 0.2, "decision": "wrong", "reason": "second event is absent", "matched_events": [1], "missing_events": [2]},
                {"id": "s2", "score": 0.95, "decision": "match", "reason": "both events match in order", "matched_events": [1, 2], "missing_events": []},
            ]
        }

    monkeypatch.setattr(openrouter_trake_verifier, "_request", fake_request)
    sequences = [
        {
            "video_id": "L1_V001",
            "timestamps": [1.0, 2.0],
            "total_score": 2.0,
            "frame_details": [{"path": str(paths[0])}, {"path": str(paths[1])}],
        },
        {
            "video_id": "L1_V002",
            "timestamps": [3.0, 4.0],
            "total_score": 1.8,
            "frame_details": [{"path": str(paths[2])}, {"path": str(paths[3])}],
        },
    ]

    verified, summary = verify_trake_sequences(sequences, ["event one", "event two"], lambda frame: frame["path"])

    assert summary["status"] == "verified"
    assert summary["evaluated"] == 2
    assert verified[0]["video_id"] == "L1_V002"
    assert verified[0]["vlm_matched_events"] == [1, 2]
    assert verified[1]["vlm_missing_events"] == [2]
    shutil.rmtree(scratch_root, ignore_errors=True)
    get_settings.cache_clear()
