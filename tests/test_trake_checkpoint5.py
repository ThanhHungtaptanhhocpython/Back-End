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
from src.services.openrouter_trake_verifier import _select_verification_sequences, verify_trake_sequences
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


def test_beam_search_keeps_fast_ground_truth_cuts(monkeypatch):
    """Rapid edits are valid when each event has its own matching frame."""
    monkeypatch.setenv("TRAKE_MIN_EVENT_GAP_SECONDS", "0")
    monkeypatch.setenv("TRAKE_MAX_EVENT_GAP_SECONDS", "60")
    get_settings.cache_clear()
    trake = TRAKE()

    sequences = trake.beam_search_sequences(
        "L30_V031",
        [
            [_candidate("keyframe_0138.jpg", 84.96, global_frame_id=2124, video_id="L30_V031")],
            [_candidate("keyframe_0140.jpg", 85.12, global_frame_id=2128, video_id="L30_V031")],
            [_candidate("keyframe_0144.jpg", 86.64, global_frame_id=2166, video_id="L30_V031")],
            [_candidate("keyframe_0149.jpg", 89.04, global_frame_id=2226, video_id="L30_V031")],
        ],
    )

    assert len(sequences) == 1
    assert sequences[0]["frames"] == [
        "keyframe_0138.jpg",
        "keyframe_0140.jpg",
        "keyframe_0144.jpg",
        "keyframe_0149.jpg",
    ]
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



def test_consecutive_context_prefers_compact_sequence_span(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "false")
    monkeypatch.setenv("TRAKE_MAX_EVENT_GAP_SECONDS", "60")
    monkeypatch.setenv("TRAKE_CONSECUTIVE_COMPACT_SPAN_SECONDS", "15")
    monkeypatch.setenv("TRAKE_CONSECUTIVE_SPAN_DECAY", "0.08")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""
    results_by_query = {
        "event one": [
            _candidate("long-a.webp", 0.0, score=1.0, video_id="L1_LONG", global_frame_id=0),
            _candidate("compact-a.webp", 10.0, score=0.7, video_id="L1_COMPACT", global_frame_id=100),
        ],
        "event two": [
            _candidate("long-b.webp", 35.0, score=1.0, video_id="L1_LONG", global_frame_id=350),
            _candidate("compact-b.webp", 18.0, score=0.7, video_id="L1_COMPACT", global_frame_id=180),
        ],
    }
    trake.retrieve_top_k = lambda query, k: [dict(item) for item in results_by_query[query]]

    response = trake.process_temporal_search(
        [
            {"query": "event one", "context": "These scenes happen consecutively."},
            {"query": "event two"},
        ],
        top_k=10,
        top_results=1,
    )

    assert len(response) == 1
    assert response[0]["video_id"] == "L1_COMPACT"
    assert response[0]["sequence_span_seconds"] == 8.0
    assert response[0]["compact_sequence_penalty"] == 1.0
    get_settings.cache_clear()

def test_temporal_search_expands_recall_when_initial_events_do_not_share_a_video(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "false")
    monkeypatch.setenv("TRAKE_RETRIEVAL_TOP_K", "10")
    monkeypatch.setenv("TRAKE_ADAPTIVE_RETRIEVAL_TOP_K", "50")
    monkeypatch.setenv("TRAKE_MAX_EVENT_GAP_SECONDS", "60")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""
    calls = []

    def retrieve(query, k):
        calls.append((query, k))
        if query == "event one":
            return [_candidate("a.webp", 10.0, video_id="L1_V001")]
        video_id = "L1_V002" if k < 50 else "L1_V001"
        return [_candidate("b.webp", 20.0, video_id=video_id)]

    trake.retrieve_top_k = retrieve
    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}],
        top_k=10,
        top_results=5,
    )

    assert [k for _query, k in calls] == [10, 10, 50, 50]
    assert len(response) == 1
    assert response[0]["video_id"] == "L1_V001"
    get_settings.cache_clear()


def test_anchor_expansion_recovers_later_event_from_an_event_anchor(monkeypatch):
    """A video sourced by one event is rescored for every other event."""
    monkeypatch.setenv("TRAKE_ANCHOR_EXPANSION_ENABLED", "true")
    monkeypatch.setenv("TRAKE_ANCHOR_VIDEO_LIMIT", "1")
    get_settings.cache_clear()
    trake = TRAKE()

    class FakeRetriever:
        def search_video_timeline(self, query, video_id, top_k):
            if query == "event two":
                return [_candidate("b.webp", 20.0, score=0.8, video_id=video_id)] if video_id == "L1_V001" else []
            return []

    import src.services.visual_retriever as visual_retriever

    monkeypatch.setattr(visual_retriever, "get_visual_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(
        trake,
        "_plan_event",
        lambda query: {"visual_query": query, "planner_source": "test"},
    )
    monkeypatch.setattr(
        trake,
        "_candidates_from_results",
        lambda _query, _plan, results, **_kwargs: [dict(item) for item in results],
    )
    monkeypatch.setattr(trake, "_rescore_event_candidates", lambda _query, candidates: candidates)

    expanded = trake._expand_anchor_videos(
        ["event one", "event two"],
        [
            [_candidate("a.webp", 10.0, score=0.9, video_id="L1_V001")],
            [_candidate("other.webp", 20.0, score=0.7, video_id="L1_V002")],
        ],
    )

    assert [item["video_id"] for item in expanded[1]] == ["L1_V002", "L1_V001"]
    assert expanded[1][1]["frame_name"] == "b.webp"
    get_settings.cache_clear()


def test_anchor_expansion_round_robins_across_events(monkeypatch):
    """A later event can contribute an anchor even when E1 points elsewhere."""
    monkeypatch.setenv("TRAKE_ANCHOR_EXPANSION_ENABLED", "true")
    monkeypatch.setenv("TRAKE_ANCHOR_VIDEO_LIMIT", "2")
    get_settings.cache_clear()
    trake = TRAKE()
    calls = []

    class FakeRetriever:
        def search_video_timelines(self, queries, video_ids, top_k):
            calls.append((queries, video_ids, top_k))
            return {video_id: [[] for _ in queries] for video_id in video_ids}

    import src.services.visual_retriever as visual_retriever

    monkeypatch.setattr(visual_retriever, "get_visual_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(
        trake,
        "_plan_event",
        lambda query: {"visual_query": query, "planner_source": "test"},
    )
    monkeypatch.setattr(trake, "_rescore_event_candidates", lambda _query, candidates: candidates)

    trake._expand_anchor_videos(
        ["event one", "event two"],
        [
            [_candidate("e1.webp", 10.0, score=0.9, video_id="L1_V001")],
            [_candidate("e2.webp", 20.0, score=0.8, video_id="L1_V002")],
        ],
    )

    assert calls == [(["event one", "event two"], ["L1_V001", "L1_V002"], 24)]
    get_settings.cache_clear()


def test_temporal_search_expands_recall_when_initial_timestamps_cannot_form_sequence(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "false")
    monkeypatch.setenv("TRAKE_RETRIEVAL_TOP_K", "10")
    monkeypatch.setenv("TRAKE_ADAPTIVE_RETRIEVAL_TOP_K", "50")
    monkeypatch.setenv("TRAKE_MAX_EVENT_GAP_SECONDS", "60")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""
    calls = []

    def retrieve(query, k):
        calls.append((query, k))
        if query == "event one":
            return [_candidate("a.webp", 20.0, video_id="L1_V001")]
        timestamp = 10.0 if k < 50 else 30.0
        return [_candidate("b.webp", timestamp, video_id="L1_V001")]

    trake.retrieve_top_k = retrieve
    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}],
        top_k=10,
        top_results=5,
    )

    assert [k for _query, k in calls] == [10, 10, 50, 50]
    assert len(response) == 1
    assert response[0]["timestamps"] == [20.0, 30.0]
    get_settings.cache_clear()


def test_sequence_verifier_samples_deeper_unique_videos():
    sequences = [
        {"video_id": f"L1_V{index:03d}", "total_score": 1.0 - (index * 0.01)}
        for index in range(12)
    ]

    selected = _select_verification_sequences(sequences, 4)

    assert [sequence["video_id"] for sequence in selected] == [
        "L1_V000",
        "L1_V001",
        "L1_V005",
        "L1_V008",
    ]



def test_sequence_verifier_prioritizes_trace_candidates():
    sequences = [
        {"video_id": "L1_TOP", "total_score": 1.0},
        {"video_id": "L1_OTHER", "total_score": 0.9},
        {"video_id": "L1_TRACE", "total_score": 0.1, "trace_verification_candidate": True},
    ]

    selected = _select_verification_sequences(sequences, 2)

    assert [sequence["video_id"] for sequence in selected] == ["L1_TRACE", "L1_TOP"]


def test_trace_videos_are_prioritized_for_vlm_even_when_ranked_lower(monkeypatch):
    get_settings.cache_clear()
    trake = TRAKE()
    ranked = [{"video_id": "L1_TOP", "total_score": 1.0}]
    target = {"video_id": "L1_TARGET", "total_score": 0.1}
    valid = ranked + [target]

    prioritized = trake._prioritise_trace_video_sequences_for_verification(
        ranked,
        valid,
        ["L1_TARGET"],
        max_promoted=1,
    )

    assert prioritized[0] is target
    assert target["trace_verification_candidate"] is True
    assert prioritized[1]["video_id"] == "L1_TOP"
    get_settings.cache_clear()
def test_trace_video_prioritization_prefers_compact_sequence_for_consecutive_query(monkeypatch):
    monkeypatch.setenv("TRAKE_CONSECUTIVE_COMPACT_SPAN_SECONDS", "15")
    get_settings.cache_clear()
    trake = TRAKE()
    high_score_long = {"video_id": "L1_TARGET", "total_score": 0.95, "timestamps": [85.0, 94.0, 98.0, 115.0]}
    low_score_compact = {"video_id": "L1_TARGET", "total_score": 0.70, "timestamps": [83.0, 85.0, 87.0, 90.0]}
    ranked = [{"video_id": "L1_TOP", "total_score": 1.0, "timestamps": [1.0, 2.0, 3.0, 4.0]}, high_score_long]

    prioritized = trake._prioritise_trace_video_sequences_for_verification(
        ranked,
        ranked + [low_score_compact],
        ["L1_TARGET"],
        max_promoted=1,
        events=["event one", "event two", "event three", "event four"],
        shared_context="These 4 scenes happen consecutively.",
    )

    assert prioritized[0] is low_score_compact
    assert prioritized[0]["trace_compact_selected"] is True
    assert trake._sequence_span_seconds(prioritized[0]) == 7.0
    get_settings.cache_clear()
def test_compact_trace_sequence_builder_uses_uncropped_event_candidates(monkeypatch):
    monkeypatch.setenv("TRAKE_CONSECUTIVE_COMPACT_SPAN_SECONDS", "15")
    get_settings.cache_clear()
    trake = TRAKE()
    candidates_list = [
        [
            _candidate("e1_far.webp", 10.0, score=0.95, video_id="L1_TARGET", global_frame_id=10),
            _candidate("e1_close.webp", 83.0, score=0.50, video_id="L1_TARGET", global_frame_id=830),
        ],
        [_candidate("e2_close.webp", 85.0, score=0.50, video_id="L1_TARGET", global_frame_id=850)],
        [_candidate("e3_close.webp", 87.0, score=0.50, video_id="L1_TARGET", global_frame_id=870)],
        [_candidate("e4_close.webp", 90.0, score=0.50, video_id="L1_TARGET", global_frame_id=900)],
    ]

    sequences = trake._build_compact_trace_sequences(
        candidates_list,
        ["L1_TARGET"],
        ["event one", "event two", "event three", "event four"],
        "These scenes happen consecutively.",
        max_sequences=1,
    )

    assert len(sequences) == 1
    assert sequences[0]["frames"] == ["e1_close.webp", "e2_close.webp", "e3_close.webp", "e4_close.webp"]
    assert sequences[0]["trace_compact_selected"] is True
    assert trake._sequence_span_seconds(sequences[0]) == 7.0
    get_settings.cache_clear()


def test_temporal_search_rejects_noncompact_vlm_match_for_consecutive_query(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "1")
    monkeypatch.setenv("TRAKE_VLM_MAX_TOTAL_SEQUENCES", "2")
    monkeypatch.setenv("AGENT_REQUIRE_VLM_MATCH", "true")
    monkeypatch.setenv("TRAKE_CONSECUTIVE_COMPACT_SPAN_SECONDS", "15")
    get_settings.cache_clear()
    trake = TRAKE()

    long_match = {
        "video_id": "L1_TARGET",
        "total_score": 1.0,
        "timestamps": [85.0, 94.0, 98.0, 115.0],
        "frame_details": [],
    }
    compact_match = {
        "video_id": "L1_TARGET",
        "total_score": 0.8,
        "timestamps": [83.0, 85.0, 87.0, 90.0],
        "frame_details": [],
    }
    ranked_sequences = [long_match, compact_match]
    calls = []

    def fake_verify(sequences, events, resolve_image_path, shared_context=""):
        calls.append(sequences[0])
        sequences[0]["vlm_score"] = 1.0
        sequences[0]["vlm_decision"] = "match"
        sequences[0]["vlm_reason"] = "all events visible"
        sequences[0]["vlm_matched_events"] = [1, 2, 3, 4]
        sequences[0]["vlm_missing_events"] = []
        sequences[0]["verification_score"] = 1.0
        return sequences, {"enabled": True, "status": "verified", "evaluated": 1, "requested": 1, "missing_images": 0}

    verified, summary = trake._verify_ranked_sequences_until_match(
        fake_verify,
        ranked_sequences,
        ["event one", "event two", "event three", "event four"],
        "These scenes happen consecutively.",
        threshold=0.45,
        require_match=True,
    )

    assert calls == [long_match, compact_match]
    assert summary["evaluated"] == 2
    assert verified[0] is compact_match
    get_settings.cache_clear()
def test_temporal_search_keeps_noncompact_vlm_match_before_unverified_fallback(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "1")
    monkeypatch.setenv("TRAKE_VLM_MAX_TOTAL_SEQUENCES", "1")
    monkeypatch.setenv("AGENT_REQUIRE_VLM_MATCH", "true")
    monkeypatch.setenv("TRAKE_CONSECUTIVE_COMPACT_SPAN_SECONDS", "15")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""

    results_by_query = {
        "event one": [_candidate("a.webp", 0.0, score=1.0, video_id="L1_LONG", global_frame_id=0)],
        "event two": [_candidate("b.webp", 10.0, score=1.0, video_id="L1_LONG", global_frame_id=100)],
        "event three": [_candidate("c.webp", 20.0, score=1.0, video_id="L1_LONG", global_frame_id=200)],
        "event four": [_candidate("d.webp", 30.0, score=1.0, video_id="L1_LONG", global_frame_id=300)],
    }
    trake.retrieve_top_k = lambda query, k: [dict(item) for item in results_by_query[query]]

    def fake_verify(sequences, events, resolve_image_path, shared_context=""):
        sequences[0]["vlm_score"] = 1.0
        sequences[0]["vlm_decision"] = "match"
        sequences[0]["vlm_reason"] = "all events visible"
        sequences[0]["vlm_matched_events"] = [1, 2, 3, 4]
        sequences[0]["vlm_missing_events"] = []
        sequences[0]["verification_score"] = 1.0
        return sequences, {"enabled": True, "status": "verified", "evaluated": 1, "requested": 1, "missing_images": 0}

    monkeypatch.setattr(openrouter_trake_verifier, "verify_trake_sequences", fake_verify)

    response = trake.process_temporal_search(
        [
            {"query": "event one", "context": "These scenes happen consecutively."},
            {"query": "event two"},
            {"query": "event three"},
            {"query": "event four"},
        ],
        top_k=10,
        top_results=1,
    )

    assert len(response) == 1
    assert response[0]["video_id"] == "L1_LONG"
    assert response[0]["verification"]["summary"]["fallback"] == "noncompact_vlm_match_after_no_compact_match"
    assert response[0]["noncompact_vlm_match"] is True
    get_settings.cache_clear()
def test_fallback_diversification_prefers_unique_videos_before_variants(monkeypatch):
    get_settings.cache_clear()
    trake = TRAKE()
    sequences = [
        {"video_id": "L1_V001", "total_score": 1.00},
        {"video_id": "L1_V001", "total_score": 0.99},
        {"video_id": "L1_V002", "total_score": 0.80},
        {"video_id": "L1_V001", "total_score": 0.70},
        {"video_id": "L1_V003", "total_score": 0.60},
    ]

    diversified = trake._diversify_fallback_sequences_by_video(sequences)

    assert [sequence["video_id"] for sequence in diversified[:3]] == ["L1_V001", "L1_V002", "L1_V003"]
    assert [sequence["video_id"] for sequence in diversified[3:]] == ["L1_V001", "L1_V001"]
    assert all(sequence.get("fallback_video_diversified") for sequence in diversified[:3])
    get_settings.cache_clear()

def test_temporal_search_keeps_wide_unverified_pool_when_vlm_rejects_checked(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "2")
    monkeypatch.setenv("AGENT_REQUIRE_VLM_MATCH", "true")
    monkeypatch.setenv("AGENT_MIN_VERIFICATION_SCORE", "0.45")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""

    results_by_query = {
        "event one": [
            _candidate("wrong-a.webp", 1.0, score=1.0, video_id="L1_WRONG", global_frame_id=10),
            _candidate("target-a.webp", 10.0, score=0.4, video_id="L1_TARGET", global_frame_id=100),
        ],
        "event two": [
            _candidate("wrong-b.webp", 2.0, score=1.0, video_id="L1_WRONG", global_frame_id=20),
            _candidate("target-b.webp", 20.0, score=0.4, video_id="L1_TARGET", global_frame_id=200),
        ],
    }
    trake.retrieve_top_k = lambda query, k: [dict(item) for item in results_by_query[query]]

    def fake_verify(sequences, events, resolve_image_path, shared_context=""):
        if sequences and sequences[0]["video_id"] == "L1_WRONG":
            sequences[0]["vlm_score"] = 0.0
            sequences[0]["vlm_decision"] = "wrong"
            sequences[0]["vlm_reason"] = "checked candidate is wrong"
            sequences[0]["vlm_matched_events"] = []
            sequences[0]["vlm_missing_events"] = [1, 2]
            sequences[0]["verification_score"] = 0.0
            return sequences, {"enabled": True, "status": "verified", "evaluated": 1, "requested": 1, "missing_images": 0}
        return sequences, {"enabled": True, "status": "fallback", "evaluated": 0, "requested": 0, "missing_images": 0}

    monkeypatch.setattr(openrouter_trake_verifier, "verify_trake_sequences", fake_verify)

    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}],
        top_k=10,
        top_results=1,
    )

    assert len(response) == 1
    assert response[0]["video_id"] == "L1_TARGET"
    assert response[0]["verification"]["summary"]["fallback"] == "unverified_pool_after_all_vlm_checked_sequences_rejected"
    get_settings.cache_clear()



def test_temporal_search_diversifies_fallback_even_when_match_not_required(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "2")
    monkeypatch.setenv("AGENT_REQUIRE_VLM_MATCH", "false")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""

    results_by_query = {
        "event one": [
            _candidate("a1.webp", 1.0, score=1.0, video_id="L1_REPEAT", global_frame_id=10),
            _candidate("b1.webp", 10.0, score=0.8, video_id="L1_OTHER", global_frame_id=100),
        ],
        "event two": [
            _candidate("a2.webp", 2.0, score=1.0, video_id="L1_REPEAT", global_frame_id=20),
            _candidate("a3.webp", 3.0, score=0.99, video_id="L1_REPEAT", global_frame_id=30),
            _candidate("b2.webp", 11.0, score=0.8, video_id="L1_OTHER", global_frame_id=110),
        ],
    }
    trake.retrieve_top_k = lambda query, k: [dict(item) for item in results_by_query[query]]

    def fake_verify(sequences, events, resolve_image_path, shared_context=""):
        return sequences, {"enabled": True, "status": "fallback", "evaluated": 0, "requested": 2, "missing_images": 0}

    monkeypatch.setattr(openrouter_trake_verifier, "verify_trake_sequences", fake_verify)

    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}],
        top_k=10,
        top_results=2,
    )

    assert [item["video_id"] for item in response] == ["L1_REPEAT", "L1_OTHER"]
    assert response[0]["verification"]["summary"]["fallback_video_diversified"] is True
    assert response[0]["verification"]["summary"]["require_match"] is False
    get_settings.cache_clear()

def test_temporal_search_verifies_next_batch_until_match(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "1")
    monkeypatch.setenv("TRAKE_VLM_MAX_TOTAL_SEQUENCES", "2")
    monkeypatch.setenv("AGENT_REQUIRE_VLM_MATCH", "true")
    monkeypatch.setenv("AGENT_MIN_VERIFICATION_SCORE", "0.45")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""

    results_by_query = {
        "event one": [
            _candidate("wrong-a.webp", 1.0, score=1.0, video_id="L1_WRONG", global_frame_id=10),
            _candidate("target-a.webp", 10.0, score=0.9, video_id="L1_TARGET", global_frame_id=100),
        ],
        "event two": [
            _candidate("wrong-b.webp", 2.0, score=1.0, video_id="L1_WRONG", global_frame_id=20),
            _candidate("target-b.webp", 12.0, score=0.9, video_id="L1_TARGET", global_frame_id=120),
        ],
    }
    trake.retrieve_top_k = lambda query, k: [dict(item) for item in results_by_query[query]]
    calls = []

    def fake_verify(sequences, events, resolve_image_path, shared_context=""):
        calls.append([sequence["video_id"] for sequence in sequences])
        if len(calls) == 1:
            sequences[0]["vlm_score"] = 0.0
            sequences[0]["vlm_decision"] = "wrong"
            sequences[0]["vlm_reason"] = "first batch is wrong"
            sequences[0]["vlm_matched_events"] = []
            sequences[0]["vlm_missing_events"] = [1, 2]
            sequences[0]["verification_score"] = 0.0
        else:
            sequences[0]["vlm_score"] = 0.95
            sequences[0]["vlm_decision"] = "match"
            sequences[0]["vlm_reason"] = "second batch matches"
            sequences[0]["vlm_matched_events"] = [1, 2]
            sequences[0]["vlm_missing_events"] = []
            sequences[0]["verification_score"] = 0.95
        return sequences, {"enabled": True, "status": "verified", "evaluated": 1, "requested": 1, "missing_images": 0}

    monkeypatch.setattr(openrouter_trake_verifier, "verify_trake_sequences", fake_verify)

    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}],
        top_k=10,
        top_results=1,
    )

    assert len(response) == 1
    assert response[0]["video_id"] == "L1_TARGET"
    assert response[0]["vlm_decision"] == "match"
    assert response[0]["verification"]["summary"]["evaluated"] == 2
    assert response[0]["verification"]["summary"]["rounds"] == 2
    assert len(calls) == 2
    get_settings.cache_clear()

def test_temporal_search_returns_temporal_pool_when_vlm_fallback_has_no_verdicts(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "2")
    monkeypatch.setenv("AGENT_REQUIRE_VLM_MATCH", "true")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""

    results_by_query = {
        "event one": [_candidate("a.webp", 1.0, score=0.8, video_id="L1_V001", global_frame_id=10)],
        "event two": [_candidate("b.webp", 2.0, score=0.8, video_id="L1_V001", global_frame_id=20)],
    }
    trake.retrieve_top_k = lambda query, k: [dict(item) for item in results_by_query[query]]

    def fake_verify(sequences, events, resolve_image_path, shared_context=""):
        return sequences, {"enabled": True, "status": "fallback", "evaluated": 0, "requested": 12}

    monkeypatch.setattr(openrouter_trake_verifier, "verify_trake_sequences", fake_verify)

    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}],
        top_k=10,
        top_results=5,
    )

    assert len(response) == 1
    assert response[0]["video_id"] == "L1_V001"
    assert response[0]["verification"]["summary"]["fallback"] == "verification_unavailable_returning_temporal_pool"
    get_settings.cache_clear()


def test_temporal_search_promotes_trace_video_after_vlm_rejects_checked_sequence(monkeypatch):
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "1")
    monkeypatch.setenv("AGENT_REQUIRE_VLM_MATCH", "true")
    get_settings.cache_clear()
    trake = TRAKE()
    trake._get_image_base64 = lambda frame: ""

    results_by_query = {
        "event one": [
            _candidate("wrong-a.webp", 1.0, score=1.0, video_id="L1_WRONG", global_frame_id=10),
            _candidate("target-a.webp", 10.0, score=0.96, video_id="L1_TARGET", global_frame_id=100),
            _candidate("distractor-a.webp", 20.0, score=0.95, video_id="L1_DISTRACTOR", global_frame_id=200),
        ],
        "event two": [
            _candidate("wrong-b.webp", 2.0, score=1.0, video_id="L1_WRONG", global_frame_id=20),
            _candidate("target-b.webp", 25.0, score=0.96, video_id="L1_TARGET", global_frame_id=250),
            _candidate("distractor-b.webp", 21.0, score=0.95, video_id="L1_DISTRACTOR", global_frame_id=210),
        ],
    }
    trake.retrieve_top_k = lambda query, k: [dict(item) for item in results_by_query[query]]

    def fake_verify(sequences, events, resolve_image_path, shared_context=""):
        if sequences and sequences[0]["video_id"] == "L1_WRONG":
            sequences[0]["vlm_score"] = 0.0
            sequences[0]["vlm_decision"] = "wrong"
            sequences[0]["vlm_reason"] = "checked candidate is wrong"
            sequences[0]["vlm_matched_events"] = []
            sequences[0]["vlm_missing_events"] = [1, 2]
            sequences[0]["verification_score"] = 0.0
            return sequences, {"enabled": True, "status": "verified", "evaluated": 1, "requested": 1, "missing_images": 0}
        return sequences, {"enabled": True, "status": "fallback", "evaluated": 0, "requested": 0, "missing_images": 0}
    monkeypatch.setattr(openrouter_trake_verifier, "verify_trake_sequences", fake_verify)

    response = trake.process_temporal_search(
        [{"query": "event one"}, {"query": "event two"}],
        top_k=10,
        top_results=1,
    )

    assert len(response) == 1
    assert response[0]["video_id"] == "L1_TARGET"
    assert response[0]["verification"]["summary"]["fallback"] == "unverified_pool_after_all_vlm_checked_sequences_rejected"
    get_settings.cache_clear()

def test_trake_vlm_extract_reports_raw_text_without_json():
    try:
        openrouter_trake_verifier._extract_trake_json("I cannot verify these images.")
    except ValueError as exc:
        assert "no JSON object in VLM response" in str(exc)
        assert "I cannot verify" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-JSON VLM response")

def test_trake_vlm_max_tokens_scales_with_batch_size():
    assert openrouter_trake_verifier._trake_max_tokens(900, 1, 2) == 900
    assert openrouter_trake_verifier._trake_max_tokens(900, 8, 4) > 900
    assert openrouter_trake_verifier._trake_max_tokens(900, 40, 4) == 3072
def test_sequence_passes_vlm_requires_complete_event_contract():
    incomplete_match = {
        "vlm_score": 0.99,
        "vlm_decision": "match",
        "vlm_matched_events": [1],
        "vlm_missing_events": [],
    }
    missing_event_match = {
        "vlm_score": 0.99,
        "vlm_decision": "match",
        "vlm_matched_events": [1, 2],
        "vlm_missing_events": [2],
    }
    complete_match = {
        "vlm_score": 0.99,
        "vlm_decision": "match",
        "vlm_matched_events": [1, 2],
        "vlm_missing_events": [],
    }

    assert not TRAKE._sequence_passes_vlm(incomplete_match, 0.45, True, event_count=2)
    assert not TRAKE._sequence_passes_vlm(missing_event_match, 0.45, True, event_count=2)
    assert TRAKE._sequence_passes_vlm(complete_match, 0.45, True, event_count=2)

def test_openrouter_sequence_verifier_reranks_and_keeps_event_contract(monkeypatch, tmp_path):
    scratch_root = tmp_path / "test_trake_vlm"
    scratch_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, color in enumerate(("red", "blue", "green", "yellow"), 1):
        path = scratch_root / f"{index}.webp"
        Image.new("RGB", (48, 48), color=color).save(path)
        paths.append(path)

    monkeypatch.setenv("AGENT_VLM_ENABLED", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "2")
    get_settings.cache_clear()

    def fake_request(messages, model, timeout, max_tokens):
        system_prompt = messages[0]["content"]
        assert "Do not simply restate the target events" in system_prompt
        assert "generic packaging" in system_prompt
        assert "under 12 words" in system_prompt
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


def test_openrouter_sequence_verifier_reports_contract_fallback_diagnostics(monkeypatch, tmp_path):
    scratch_root = tmp_path / "test_trake_vlm_bad_contract"
    scratch_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, color in enumerate(("red", "blue"), 1):
        path = scratch_root / f"{index}.webp"
        Image.new("RGB", (48, 48), color=color).save(path)
        paths.append(path)

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "1")
    get_settings.cache_clear()

    def fake_request(messages, model, timeout, max_tokens):
        return {"items": [{"id": "bad", "score": 1.0, "decision": "match", "reason": "bad id", "matched_events": [1, 2], "missing_events": []}]}

    monkeypatch.setattr(openrouter_trake_verifier, "_request", fake_request)
    sequences = [{
        "video_id": "L1_V001",
        "timestamps": [1.0, 2.0],
        "total_score": 1.0,
        "frame_details": [{"path": str(paths[0])}, {"path": str(paths[1])}],
    }]

    verified, summary = verify_trake_sequences(sequences, ["event one", "event two"], lambda frame: frame["path"])

    assert verified == sequences
    assert summary["status"] == "fallback"
    assert summary["evaluated"] == 0
    assert "unexpected or duplicate sequence id" in " ".join(summary["contract_errors"])
    assert "bad" in summary["payload_preview"]
    shutil.rmtree(scratch_root, ignore_errors=True)
    get_settings.cache_clear()
def test_openrouter_sequence_verifier_diversifies_videos_before_variants(monkeypatch, tmp_path):
    scratch_root = tmp_path / "test_trake_vlm_diverse"
    scratch_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, color in enumerate(("red", "blue", "green", "yellow", "purple", "orange"), 1):
        path = scratch_root / f"{index}.webp"
        Image.new("RGB", (48, 48), color=color).save(path)
        paths.append(path)

    monkeypatch.setenv("AGENT_VLM_ENABLED", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("TRAKE_VLM_ENABLED", "true")
    monkeypatch.setenv("TRAKE_VLM_MAX_SEQUENCES", "2")
    get_settings.cache_clear()

    def fake_request(messages, model, timeout, max_tokens):
        content = messages[1]["content"]
        text_payload = "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
        assert "video=L1_V001" in text_payload
        assert "video=L1_V002" in text_payload
        assert "timestamps=[1.5, 2.5]" not in text_payload
        return {
            "items": [
                {"id": "s1", "score": 0.1, "decision": "wrong", "reason": "wrong video", "matched_events": [], "missing_events": [1, 2]},
                {"id": "s2", "score": 0.98, "decision": "match", "reason": "both events match", "matched_events": [1, 2], "missing_events": []},
            ]
        }

    monkeypatch.setattr(openrouter_trake_verifier, "_request", fake_request)
    duplicate_sequence = {
        "video_id": "L1_V001",
        "timestamps": [1.5, 2.5],
        "total_score": 1.9,
        "frame_details": [{"path": str(paths[2])}, {"path": str(paths[3])}],
    }
    sequences = [
        {
            "video_id": "L1_V001",
            "timestamps": [1.0, 2.0],
            "total_score": 2.0,
            "frame_details": [{"path": str(paths[0])}, {"path": str(paths[1])}],
        },
        duplicate_sequence,
        {
            "video_id": "L1_V002",
            "timestamps": [3.0, 4.0],
            "total_score": 1.8,
            "frame_details": [{"path": str(paths[4])}, {"path": str(paths[5])}],
        },
    ]

    verified, summary = verify_trake_sequences(sequences, ["event one", "event two"], lambda frame: frame["path"])

    assert summary["requested"] == 2
    assert summary["evaluated"] == 2
    assert verified[0]["video_id"] == "L1_V002"
    assert duplicate_sequence.get("vlm_score") is None
    shutil.rmtree(scratch_root, ignore_errors=True)
    get_settings.cache_clear()
